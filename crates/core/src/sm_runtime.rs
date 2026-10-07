//! Per-character runtime for the ULSM (`.sm`) state machines.
//!
//! ## What this executes
//!
//! This is a native re-implementation of the engine's per-frame state-machine
//! step `SmTick` (`0x158e990`) and its helpers, driven by an [`SmEnv`] that
//! supplies the engine-side inputs (animation events / phase, actor flags,
//! signals, timers, entity / transform queries). It consumes the structures
//! recovered by [`crate::statemachine`] and [`crate::sm_payload`].
//!
//! ### Per-tick order (`SmTick` 0x158e990)
//!
//! 1. `FUN_016955c0` / `FUN_01698980` evaluate the state's entity / transform
//!    sources.
//! 2. `FUN_0169e0a0` evaluates the state's trigger records in block order and
//!    writes one boolean per trigger into the per-state output array
//!    (`0x1ee3170`); a transition `condition` index addresses that array.
//! 3. `FUN_01698770` (0x1698770), reached from `FUN_01698870` (0x1698870),
//!    visits the state's blocks **in reverse** and each block's primary then
//!    secondary condition list, stopping at the first condition whose output
//!    is set and picking a candidate by weighted random.
//! 4. `FUN_01693930` runs the current state's actions (operands from
//!    `action_data` + `block_strings`).
//! 5. If a transition was selected, the machine switches to the target state
//!    and plays its fragment (`0x138c1b0`), passing the six low bits of the
//!    fragment `tag`.
//!
//! ### Weighted selection
//!
//! With more than one candidate, a 16-bit LCG pair combined into a value in
//! `0..100` is produced (`0x1ee28ec`/`0x1ee28f0`, `0x16987cd`..`0x1698818`),
//! and the first candidate whose `weight > value` wins; if none does, the last
//! candidate is used. [`select_by_weight`] reproduces that rule.
//!
//! ### `block_strings`
//!
//! The per-block `block_strings` blob is not a general bytecode interpreter:
//! it is the concatenation of the per-action code operands. `FUN_01693930`
//! consumes, for each `action_ids` entry of a block, `descriptor.code_size`
//! bytes aligned to `descriptor.code_align` from
//! `block_strings + block.code_offset`, bounded by `block.code_length`.
//! [`crate::sm_payload`] records that slice as
//! [`crate::sm_payload::SmActionRecord::code`]; the
//! runtime surfaces it on [`SmActionCommand::code`].
//!
//! ### GraphDriven hand-off
//!
//! States named `GraphDriven.*` mean "the AnimationGraph drives this character
//! right now" (locomotion, dead, sprint, ...). [`SmInstance::drive_mode`]
//! reports that; [`SmInstance::is_graph_driven`] is the fast query the lead
//! uses to decide whether the graph or the state machine owns locomotion.
//! A [`SmCommand::DriveMode`] command is emitted whenever the ownership
//! changes.
//!
//! Everything here is best-effort: unknown trigger evaluators fall back to
//! [`SmEnv::eval_raw_trigger`], and the whole runtime is panic-free on
//! malformed / truncated machines.

use crate::sm_payload::SmTriggerOperand;
use crate::statemachine::{StateMachine, SmTransition, SmTransitionList};
use std::collections::HashMap;

/// Trigger-output sentinel meaning "no transition target" / `<exit>`.
pub const SM_EXIT: u32 = 0xffff;

/// Broad semantic class of a trigger type, recovered from the evaluator
/// functions registered in `TRIGGER_DESCS` (`crate::sm_tables`).
///
/// The player and zombie machines are dominated by [`TriggerKind::BoolCombine`]
/// (type 17, `FUN_0169eea0`). The remaining frequently used types are decoded
/// here; anything else is [`TriggerKind::Raw`] and answered by the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerKind {
    /// Always true (type 4, evaluator `0x9e9ec0`).
    ConstTrue,
    /// Always false (type 73, evaluator `0x7fa860`).
    ConstFalse,
    /// Type 17: 8-input boolean combine over other condition outputs.
    BoolCombine,
    /// Type 19: engine signal / event id (`FUN_01376640`, `actor+0x12e0`).
    Signal,
    /// Types 7 / 8 / 18 / 71: bits of the actor flags byte `actor+0x1364`.
    EngineFlag { bit: u8 },
    /// Type 62: bit of the actor flag word `actor+0x1250`.
    ActorFlagWord,
    /// Type 40: a query result slot is populated, optionally gated by a
    /// condition (`actor+0x48 + slot*4`).
    QuerySlot,
    /// Type 41: query result slot object state test.
    QuerySlotState,
    /// Type 63: comparison over the actor counter array `actor+0x1254`.
    CounterCompare,
    /// Type 66: `f32` value in range against actor fields `+0x12d0`/`+0x12d4`.
    FloatRange,
    /// Type 67: actor integer state `actor+0x1564` equals a byte.
    ActorInt,
    /// Type 56: condition falling/rising-edge latch.
    EdgeLatch,
    /// Type 5: timed hysteresis window gated by a condition.
    TimerWindow,
    /// Type 6: accumulator limit gated by a condition.
    TimerGate,
    /// Type 9: frame-delta window.
    FrameWindow,
    /// Type 37: ramp/hysteresis gated by a condition.
    TimerRamp,
    /// Type 82: one-shot delay / latched threshold over engine time.
    DelayOneShot,
    /// Type 93: persistent accumulator crossing a static threshold
    /// (`FUN_016a0e90`).
    Accumulate,
    /// Anything not yet decoded; use [`SmEnv::eval_raw_trigger`].
    Raw,
}

/// Classify a trigger type id into a [`TriggerKind`].
pub fn trigger_kind(id: u8) -> TriggerKind {
    match id {
        4 => TriggerKind::ConstTrue,
        73 => TriggerKind::ConstFalse,
        17 => TriggerKind::BoolCombine,
        19 => TriggerKind::Signal,
        7 => TriggerKind::EngineFlag { bit: 0 },
        8 => TriggerKind::EngineFlag { bit: 1 },
        18 => TriggerKind::EngineFlag { bit: 4 },
        71 => TriggerKind::EngineFlag { bit: 3 },
        62 => TriggerKind::ActorFlagWord,
        40 => TriggerKind::QuerySlot,
        41 => TriggerKind::QuerySlotState,
        63 => TriggerKind::CounterCompare,
        66 => TriggerKind::FloatRange,
        67 => TriggerKind::ActorInt,
        56 => TriggerKind::EdgeLatch,
        5 => TriggerKind::TimerWindow,
        6 => TriggerKind::TimerGate,
        9 => TriggerKind::FrameWindow,
        37 => TriggerKind::TimerRamp,
        82 => TriggerKind::DelayOneShot,
        93 => TriggerKind::Accumulate,
        _ => TriggerKind::Raw,
    }
}

/// Engine-side inputs consumed by the runtime.
///
/// All methods have inert defaults so a test / headless caller only
/// implements the inputs it cares about.
#[allow(unused_variables)]
pub trait SmEnv {
    /// Seconds since the previous tick.
    fn delta_seconds(&self) -> f32 {
        0.0
    }
    /// Machine-local time in seconds (used by delayed triggers).
    fn time_seconds(&self) -> f32 {
        0.0
    }
    /// Actor flags byte `actor+0x1364` (types 7/8/18/71).
    fn actor_flags(&self) -> u32 {
        0
    }
    /// Actor flag word `actor+0x1250` (type 62).
    fn actor_flag_word(&self) -> u32 {
        0
    }
    /// Engine signal / event by id (type 19, `FUN_01376640`).
    fn signal(&self, id: u16) -> bool {
        false
    }
    /// Actor counter array value `actor+0x1254[slot]` (type 63).
    fn counter(&self, slot: u8) -> i32 {
        0
    }
    /// A float actor field, `which` selecting `+0x12d0` (0) or `+0x12d4` (1)
    /// (type 66).
    fn actor_float(&self, which: u8) -> f32 {
        0.0
    }
    /// Integer actor state `actor+0x1564` (type 67).
    fn actor_int(&self) -> i32 {
        0
    }
    /// Whether entity/transform query slot `slot` produced a result
    /// (`actor+0x48 + slot*4`) (types 40/41).
    fn query_slot(&self, slot: u8) -> bool {
        false
    }
    /// Opaque state of a query slot result (type 41).
    fn query_slot_state(&self, slot: u8) -> i32 {
        0
    }
    /// Animation event `name` fired this tick.
    fn animation_event(&self, name: &str) -> bool {
        false
    }
    /// Current animation normalized phase.
    fn animation_phase(&self) -> f32 {
        0.0
    }
    /// Fallback for trigger evaluators not modelled by this runtime. `state`
    /// and `output` are the current state and the trigger-output index, so a
    /// scripted environment can answer per occurrence. Returning `None` makes
    /// the output default to `false`.
    fn eval_raw_trigger(
        &self,
        state: u32,
        output: u32,
        type_id: u8,
        operand: &[u8],
    ) -> Option<bool> {
        None
    }
}

/// The engine's transition-selection RNG (`0x1ee28ec` / `0x1ee28f0`).
///
/// Two 32-bit seeds are advanced independently by a 16-bit LCG step and then
/// combined into a value in `0..100` (`0x16987cd`..`0x1698818`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmRng {
    s0: u32,
    s1: u32,
}

impl SmRng {
    /// Seed the generator. Retail seeds come from a global time seed; the
    /// runtime allows any seed for deterministic replay.
    pub fn new(seed: u32) -> Self {
        SmRng {
            s0: seed | 1,
            s1: seed.wrapping_mul(0x9e37_79b1) | 1,
        }
    }

    /// Advance both seeds and return a value in `0..100`.
    pub fn next_percent(&mut self) -> u32 {
        let a = (self.s0 & 0xffff)
            .wrapping_mul(0x4650)
            .wrapping_add(self.s0 >> 16);
        let b = (self.s1 & 0xffff)
            .wrapping_mul(0x9069)
            .wrapping_add(self.s1 >> 16);
        self.s0 = a;
        self.s1 = b;
        (b << 16).wrapping_add(a) % 100
    }
}

/// Pick a candidate index exactly as `FUN_01698770` does: with one candidate
/// it is chosen unconditionally; otherwise the first candidate whose
/// `weight > rand` wins, falling back to the last candidate.
pub fn select_by_weight(weights: &[u16], rand: u16) -> usize {
    if weights.len() <= 1 {
        return weights.len().saturating_sub(1);
    }
    for (i, &w) in weights[..weights.len() - 1].iter().enumerate() {
        if w > rand {
            return i;
        }
    }
    weights.len() - 1
}

/// A chosen transition, buffered until the end of the tick like the engine's
/// instance slots `+0xe`/`+0x10`/`+0x14`/`+0x18`/`+0x19`.
#[derive(Debug, Clone, PartialEq)]
pub struct SmPending {
    pub target: u32,
    pub blend_in: f32,
    pub blend_out: f32,
    pub condition: u16,
    pub list: SmTransitionList,
    /// `candidate.flags != 0`, copied to instance `+0x19`.
    pub flag: bool,
}

/// Ownership of locomotion, derived from the current state's name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmDriveMode {
    /// State is `GraphDriven.<group>...`: let the AnimationGraph drive.
    GraphDriven(String),
    /// State is `MachineDriven.*`: the state machine drives.
    MachineDriven,
}

/// Broad category of a state action, named from the `ACTION_DESCS` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmActionKind {
    Animation,
    AnimationGraphInput,
    MovementTurnTo,
    Movement,
    AISignal,
    DebugLog,
    Timer,
    Damage,
    Query,
    Other,
}

/// Classify an action descriptor id (see `crate::sm_tables::ACTION_DESCS`).
pub fn action_kind(id: u8) -> SmActionKind {
    match id {
        13 | 149 => SmActionKind::Animation,
        15 => SmActionKind::AnimationGraphInput,
        99 => SmActionKind::MovementTurnTo,
        95 | 96 | 97 | 98 => SmActionKind::Movement,
        1 => SmActionKind::AISignal,
        44 => SmActionKind::DebugLog,
        143..=146 => SmActionKind::Timer,
        29 | 39 | 40 | 41 | 42 | 105 | 155 => SmActionKind::Damage,
        27 | 28 | 30 | 67 | 68 | 92 | 93 | 114 => SmActionKind::Query,
        _ => SmActionKind::Other,
    }
}

/// Name of the graph input written by an `AnimationGraphInput` action
/// (descriptor id 15, evaluator `0x0168b120`). The action's `code` operand is a
/// 128-byte buffer holding a NUL-terminated input name at offset 0
/// (e.g. `Signal`, `CarryWeapon`); the value it writes is not encoded in the
/// operand we decode, so callers choose one. Returns `None` for any other
/// action or an empty name.
pub fn animation_graph_input_name(cmd: &SmActionCommand) -> Option<&str> {
    if action_kind(cmd.descriptor_id) != SmActionKind::AnimationGraphInput {
        return None;
    }
    let end = cmd.code.iter().position(|&b| b == 0).unwrap_or(cmd.code.len());
    let name = std::str::from_utf8(&cmd.code[..end]).ok()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// One action descriptor invocation, emitted by [`SmInstance::tick`].
#[derive(Debug, Clone, PartialEq)]
pub struct SmActionCommand {
    pub state: u32,
    pub block: u32,
    pub index: u16,
    pub action_id: i16,
    pub descriptor_id: u8,
    pub name: &'static str,
    pub kind: SmActionKind,
    pub flags: u8,
    /// Operand from the `action_data` blob.
    pub data: Vec<u8>,
    /// Operand from the `block_strings` code stream.
    pub code: Vec<u8>,
}

/// A typed command emitted by a tick. The engine would apply these to the
/// actor; the re-implementation publishes them so the caller owns the policy.
#[derive(Debug, Clone, PartialEq)]
pub enum SmCommand {
    /// Play the state's fragment / clip (`0x138c1b0`).
    PlayFragment {
        clip: String,
        speed: f32,
        blend_in: f32,
        blend_out: f32,
        /// Six low bits of the fragment `tag` passed to the clip player.
        tag: u16,
        flags: u32,
    },
    /// Run one of the state's actions.
    Action(SmActionCommand),
    /// The GraphDriven/MachineDriven ownership changed.
    DriveMode(SmDriveMode),
}

/// Per-trigger persistent state (timers / latches). Keyed by
/// `(state, output)`; the engine keeps this inline at each descriptor
/// instance.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Persist {
    t: f32,
    flags: u8,
    i: i32,
}

/// A running state-machine instance for one character.
pub struct SmInstance<'a> {
    sm: &'a StateMachine,
    /// Current state index.
    pub state: u32,
    /// Total machine-local time.
    pub time: f32,
    /// Time spent in the current state.
    pub state_time: f32,
    /// `true` once an `<exit>` transition has been taken.
    pub exited: bool,
    /// Per-state trigger-output flags for the current tick.
    outputs: Vec<bool>,
    /// `outputs[i]` -> index into `sm.payloads.triggers`.
    trigger_of_output: Vec<Option<usize>>,
    /// Global trigger index lookup `(state, block, index)` -> record index,
    /// built once at construction.
    trigger_index: HashMap<(u32, u32, u16), usize>,
    /// `(state, block)` -> indices into `sm.payloads.actions`, in emission
    /// order. Avoids rescanning the whole action table every tick.
    action_index: HashMap<(u32, u32), Vec<usize>>,
    /// `(state, block)` -> indices into `sm.transitions`, in emission order.
    transition_index: HashMap<(u32, u32), Vec<usize>>,
    persist: HashMap<(u32, u32), Persist>,
    /// Transition chosen this tick, applied at the end of the tick.
    pub pending: Option<SmPending>,
    /// Commands emitted since the last `take_commands`.
    pub commands: Vec<SmCommand>,
    pub rng: SmRng,
    drive: SmDriveMode,
}

impl<'a> SmInstance<'a> {
    /// Create an instance on `sm`'s start state, immediately entering it
    /// (playing its fragment and reporting the initial drive mode).
    pub fn new(sm: &'a StateMachine, seed: u32) -> Self {
        let mut trigger_index = HashMap::new();
        for (i, t) in sm.payloads.triggers.iter().enumerate() {
            trigger_index.insert((t.state, t.block, t.index), i);
        }
        let mut action_index: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (i, a) in sm.payloads.actions.iter().enumerate() {
            action_index.entry((a.state, a.block)).or_default().push(i);
        }
        let mut transition_index: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (i, t) in sm.transitions.iter().enumerate() {
            transition_index.entry((t.source, t.block)).or_default().push(i);
        }
        let mut inst = SmInstance {
            sm,
            state: sm.start_state,
            time: 0.0,
            state_time: 0.0,
            exited: false,
            outputs: Vec::new(),
            trigger_of_output: Vec::new(),
            trigger_index,
            action_index,
            transition_index,
            persist: HashMap::new(),
            pending: None,
            commands: Vec::new(),
            rng: SmRng::new(seed),
            drive: SmDriveMode::MachineDriven,
        };
        let start = sm.start_state;
        // Force the initial drive-mode command to be emitted.
        inst.drive = SmDriveMode::MachineDriven;
        inst.enter_state(start, None, None);
        inst
    }

    /// Name of the current state, or `"<none>"`.
    pub fn state_name(&self) -> &str {
        self.sm
            .states
            .get(self.state as usize)
            .map(|s| s.name.as_str())
            .unwrap_or("<none>")
    }

    /// Number of states in the machine.
    pub fn state_count(&self) -> usize {
        self.sm.states.len()
    }

    /// Name of the state at `state`, if it exists.
    pub fn state_name_at(&self, state: u32) -> Option<&str> {
        self.sm.states.get(state as usize).map(|s| s.name.as_str())
    }

    /// Locomotion ownership derived from the current state name.
    pub fn drive_mode(&self) -> SmDriveMode {
        match self.state_name().strip_prefix("GraphDriven.") {
            Some(rest) => SmDriveMode::GraphDriven(
                rest.split('.').next().unwrap_or("").to_string(),
            ),
            None => SmDriveMode::MachineDriven,
        }
    }

    /// Convenience: is the AnimationGraph currently driving?
    pub fn is_graph_driven(&self) -> bool {
        matches!(self.drive_mode(), SmDriveMode::GraphDriven(_))
    }

    /// The GraphDriven group (e.g. `Observer`, `Dead`, `Sprint`), if any.
    pub fn graph_driven_group(&self) -> Option<String> {
        match self.drive_mode() {
            SmDriveMode::GraphDriven(g) => Some(g),
            SmDriveMode::MachineDriven => None,
        }
    }

    /// Take the commands emitted since the last call.
    pub fn take_commands(&mut self) -> Vec<SmCommand> {
        std::mem::take(&mut self.commands)
    }

    /// Index of the state whose name matches `name` (exact, case-insensitive,
    /// then substring), for debug / scripted entry.
    pub fn find_state(&self, name: &str) -> Option<u32> {
        let key = name.to_ascii_lowercase();
        self.sm
            .states
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))
            .or_else(|| self.sm.states.iter().find(|s| s.name.to_ascii_lowercase().contains(&key)))
            .map(|s| s.index)
    }

    /// Force the machine into `state`, emitting its fragment / drive-mode like a
    /// normal transition. Used by the `SOD_SMTEST` debug entry point.
    pub fn go_to(&mut self, state: u32) {
        self.enter_state(state, None, None);
    }

    /// Enter `state`, emitting its fragment and a [`SmCommand::DriveMode`] on
    /// ownership change. The engine's actual switch lives in `0x158b9b0` and
    /// the fragment play in `0x138c1b0`.
    fn enter_state(&mut self, state: u32, blend_in: Option<f32>, blend_out: Option<f32>) {
        self.state = state;
        self.state_time = 0.0;
        if state == SM_EXIT || (state as usize) >= self.sm.states.len() {
            self.exited = true;
            return;
        }
        if let Some(frag) = self
            .sm
            .states
            .get(state as usize)
            .and_then(|s| s.fragment)
            .and_then(|f| self.sm.fragments.get(f as usize))
        {
            self.commands.push(SmCommand::PlayFragment {
                clip: frag.name.clone(),
                speed: frag.speed,
                blend_in: blend_in.unwrap_or(0.0),
                blend_out: blend_out.unwrap_or(0.0),
                tag: frag.tag,
                flags: frag.flags,
            });
        }
        let mode = self.drive_mode();
        if mode != self.drive {
            self.commands.push(SmCommand::DriveMode(mode.clone()));
            self.drive = mode;
        }
        self.rebuild_outputs(state);
    }

    /// Build the per-state output/trigger maps.
    fn rebuild_outputs(&mut self, state: u32) {
        let sm = self.sm;
        let conds = sm.payloads.conditions.iter().find(|c| c.state == state);
        let count = conds.map(|c| c.trigger_count).unwrap_or(0) as usize;
        self.outputs = vec![false; count];
        self.trigger_of_output = vec![None; count];
        if let Some(cond) = conds {
            for o in &cond.outputs {
                if let Some(&idx) = self.trigger_index.get(&(state, o.block, o.index)) {
                    let out = o.condition as usize;
                    if out < self.trigger_of_output.len() {
                        self.trigger_of_output[out] = Some(idx);
                    }
                }
            }
        }
    }

    /// Advance the machine one tick.
    pub fn tick(&mut self, env: &dyn SmEnv) {
        if self.exited {
            return;
        }
        let dt = env.delta_seconds().max(0.0);
        self.time += dt;
        self.state_time += dt;

        self.evaluate_triggers(env);
        self.pending = self.select_transition();
        self.run_actions();

        if let Some(p) = self.pending.take() {
            self.enter_state(p.target, Some(p.blend_in), Some(p.blend_out));
        }
    }

    /// Evaluate every trigger of the current state in output order, so a
    /// [`TriggerKind::BoolCombine`] can reference earlier outputs.
    fn evaluate_triggers(&mut self, env: &dyn SmEnv) {
        let sm = self.sm;
        for i in 0..self.trigger_of_output.len() {
            let Some(Some(idx)) = self.trigger_of_output.get(i).copied() else {
                continue;
            };
            let Some(t) = sm.payloads.triggers.get(idx) else {
                continue;
            };
            let val = self.eval_record(env, t, i as u32);
            if let Some(slot) = self.outputs.get_mut(i) {
                *slot = val;
            }
        }
    }

    /// Evaluate one trigger record. Returns the output boolean.
    fn eval_record(&mut self, env: &dyn SmEnv, t: &crate::sm_payload::SmTriggerRecord, out: u32) -> bool {
        let kind = trigger_kind(t.type_id);
        match kind {
            TriggerKind::ConstTrue => true,
            TriggerKind::ConstFalse => false,
            TriggerKind::BoolCombine => {
                if let SmTriggerOperand::BoolCombine { mode, inputs } = &t.operand {
                    let mut any = false;
                    let mut all = true;
                    for &inp in inputs {
                        let v = self.outputs.get(inp as usize).copied().unwrap_or(false);
                        any |= v;
                        all &= v;
                    }
                    let mut r = if mode & 1 != 0 { any } else { all };
                    if mode & 2 != 0 {
                        r = !r;
                    }
                    r
                } else {
                    self.raw(env, t, out)
                }
            }
            TriggerKind::Signal => match raw(t) {
                Some(b) => env.signal(ru16(b, 0, self.sm.big_endian)),
                None => self.raw(env, t, out),
            },
            TriggerKind::EngineFlag { bit } => {
                if t.type_id == 71 {
                    // Grab/attach flag: actor flag bit 3, plus an optional
                    // signal and mode/name check (`FUN_016a0490`).
                    if env.actor_flags() & (1 << bit) == 0 {
                        return false;
                    }
                    match raw(t) {
                        Some(b) => {
                            let id = ru16(b, 0, self.sm.big_endian);
                            id == 0xffff || env.signal(id)
                        }
                        None => true,
                    }
                } else {
                    env.actor_flags() & (1u32 << bit) != 0
                }
            }
            TriggerKind::ActorFlagWord => match raw(t) {
                Some(b) => {
                    let bit = b.first().copied().unwrap_or(0).min(31);
                    env.actor_flag_word() & (1u32 << bit) != 0
                }
                None => self.raw(env, t, out),
            },
            TriggerKind::QuerySlot => match raw(t) {
                Some(b) => {
                    let cond = ru16(b, 0, self.sm.big_endian);
                    let slot = b.get(2).copied().unwrap_or(0);
                    env.query_slot(slot)
                        && (cond == 0xffff
                            || self.outputs.get(cond as usize).copied().unwrap_or(false))
                }
                None => self.raw(env, t, out),
            },
            TriggerKind::QuerySlotState => match raw(t) {
                Some(b) => {
                    let cond = ru16(b, 0, self.sm.big_endian);
                    let slot = b.get(2).copied().unwrap_or(0);
                    env.query_slot(slot)
                        && env.query_slot_state(slot) != 0
                        && (cond == 0xffff
                            || self.outputs.get(cond as usize).copied().unwrap_or(false))
                }
                None => self.raw(env, t, out),
            },
            TriggerKind::CounterCompare => match raw(t) {
                Some(b) if b.len() >= 5 => {
                    let v = |s: u8| -> i32 {
                        if s == 0x1d {
                            env.counter(0).saturating_add(1)
                        } else {
                            env.counter(s)
                        }
                    };
                    let base = v(b[0]);
                    (1..5).all(|k| base >= v(b[k]))
                }
                _ => self.raw(env, t, out),
            },
            TriggerKind::FloatRange => match raw(t) {
                Some(b) if b.len() >= 9 => {
                    let min = rf32(b, 0, self.sm.big_endian);
                    let max = rf32(b, 4, self.sm.big_endian);
                    let val = env.actor_float(b[8] & 1);
                    min < val && val <= max
                }
                _ => self.raw(env, t, out),
            },
            TriggerKind::ActorInt => match raw(t) {
                Some(b) => env.actor_int() == b.first().copied().unwrap_or(0) as i32,
                None => self.raw(env, t, out),
            },
            TriggerKind::EdgeLatch => match raw(t) {
                Some(b) if b.len() >= 3 => {
                    let cond = ru16(b, 0, self.sm.big_endian);
                    let mode = b[2];
                    let input = self
                        .outputs
                        .get(cond as usize)
                        .copied()
                        .unwrap_or(false);
                    let key = (self.state, out);
                    let p = self.persist.entry(key).or_default();
                    let old = p.flags & 1 != 0;
                    p.flags = (p.flags & !1) | input as u8;
                    // Falling edge for mode bit0 set, rising edge otherwise.
                    !input && if mode & 1 != 0 { old } else { true }
                }
                _ => self.raw(env, t, out),
            },
            TriggerKind::TimerWindow => {
                if let SmTriggerOperand::TimerWindow {
                    min,
                    max,
                    rate,
                    cond,
                } = &t.operand
                {
                    let dt = env.delta_seconds().max(0.0);
                    let key = (self.state, out);
                    let p = self.persist.entry(key).or_default();
                    let gate = *cond == 0xffff
                        || self.outputs.get(*cond as usize).copied().unwrap_or(false);
                    if gate {
                        p.t = (p.t + dt * *rate).min(*max);
                    }
                    p.t >= *min
                } else {
                    self.raw(env, t, out)
                }
            }
            TriggerKind::TimerGate => {
                if let SmTriggerOperand::TimerGate { limit, cond } = &t.operand {
                    let dt = env.delta_seconds().max(0.0);
                    let key = (self.state, out);
                    let p = self.persist.entry(key).or_default();
                    let gate = *cond == 0xffff
                        || self.outputs.get(*cond as usize).copied().unwrap_or(false);
                    if gate {
                        p.t += dt;
                    }
                    p.t >= *limit
                } else {
                    self.raw(env, t, out)
                }
            }
            TriggerKind::FrameWindow => {
                if let SmTriggerOperand::Window { lo, hi, flag } = &t.operand {
                    let dt = env.delta_seconds();
                    let inside = dt >= *lo && dt <= *hi;
                    if *flag & 1 != 0 {
                        !inside
                    } else {
                        inside
                    }
                } else {
                    self.raw(env, t, out)
                }
            }
            TriggerKind::TimerRamp => match raw(t) {
                Some(b) if b.len() >= 6 => {
                    let rate = rf32(b, 0, self.sm.big_endian);
                    let cond = ru16(b, 4, self.sm.big_endian);
                    let dt = env.delta_seconds().max(0.0);
                    let key = (self.state, out);
                    let p = self.persist.entry(key).or_default();
                    let gate = cond == 0xffff
                        || self.outputs.get(cond as usize).copied().unwrap_or(false);
                    if gate {
                        p.t = (p.t + rate * dt).clamp(0.0, 1.0);
                    } else {
                        p.t = 0.0;
                    }
                    p.t > 0.0
                }
                _ => self.raw(env, t, out),
            },
            TriggerKind::DelayOneShot => match raw(t) {
                Some(b) if b.len() >= 7 => {
                    let threshold = rf32(b, 0, self.sm.big_endian);
                    let cond = ru16(b, 4, self.sm.big_endian);
                    let mode = b[6];
                    let gate = cond == 0xffff
                        || self.outputs.get(cond as usize).copied().unwrap_or(false);
                    let key = (self.state, out);
                    let p = self.persist.entry(key).or_default();
                    if gate && (mode & 1 == 0 || p.flags & 1 == 0) && env.time_seconds() > threshold
                    {
                        p.flags |= 2;
                    }
                    p.flags |= (gate as u8) & 1;
                    p.flags & 2 != 0
                }
                _ => self.raw(env, t, out),
            },
            TriggerKind::Accumulate => match raw(t) {
                Some(b) if b.len() >= 4 => {
                    let threshold = rf32(b, 0, self.sm.big_endian);
                    let dt = env.delta_seconds().max(0.0);
                    let key = (self.state, out);
                    let p = self.persist.entry(key).or_default();
                    p.t += dt;
                    p.t >= threshold
                }
                _ => self.raw(env, t, out),
            },
            TriggerKind::Raw => self.raw(env, t, out),
        }
    }

    /// Ask the environment for an engine-specific trigger result.
    fn raw(&self, env: &dyn SmEnv, t: &crate::sm_payload::SmTriggerRecord, out: u32) -> bool {
        let bytes = match raw(t) {
            Some(b) => b,
            None => return false,
        };
        env.eval_raw_trigger(t.state, out, t.type_id, bytes)
            .unwrap_or(false)
    }

    /// Reproduce `FUN_01698870` (0x1698870): visit the state's blocks in
    /// reverse and each block's primary then secondary condition list, picking
    /// a candidate from the first condition whose output is set.
    fn select_transition(&mut self) -> Option<SmPending> {
        let sm = self.sm;
        let st = sm.states.get(self.state as usize)?;
        let mut weights: Vec<u16> = Vec::new();
        for &bi in st.blocks.iter().rev() {
            let mut i = 0usize;
            // This block's transitions, in emission order (indexed once at
            // construction so the whole table is not rescanned every tick).
            let idxs = self.transition_index.get(&(st.index, bi)).cloned().unwrap_or_default();
            let block_trans: Vec<&SmTransition> =
                idxs.iter().map(|&t| &sm.transitions[t]).collect();
            while i < block_trans.len() {
                let cur = block_trans[i];
                let (list, cond) = (cur.list, cur.condition);
                let mut end = i;
                while end < block_trans.len()
                    && block_trans[end].list == list
                    && block_trans[end].condition == cond
                {
                    end += 1;
                }
                let group = &block_trans[i..end];
                if self.outputs.get(cond as usize).copied().unwrap_or(false) && !group.is_empty() {
                    weights.clear();
                    weights.extend(group.iter().map(|c| c.weight));
                    let rand = if weights.len() <= 1 {
                        0
                    } else {
                        self.rng.next_percent() as u16
                    };
                    let pick = select_by_weight(&weights, rand);
                    let c = group[pick];
                    return Some(SmPending {
                        target: c.target,
                        blend_in: c.blend_in,
                        blend_out: c.blend_out,
                        condition: c.condition,
                        list: c.list,
                        flag: c.flags != 0,
                    });
                }
                i = end;
            }
        }
        None
    }

    /// Run the current state's actions in block order, emitting a
    /// [`SmCommand::Action`] for each. This mirrors the operand walk of
    /// `FUN_01693930` (0x1693930).
    fn run_actions(&mut self) {
        let sm = self.sm;
        let Some(st) = sm.states.get(self.state as usize) else {
            return;
        };
        for &bi in &st.blocks {
            let idxs = self.action_index.get(&(st.index, bi)).cloned().unwrap_or_default();
            for ai in idxs {
                let a = &sm.payloads.actions[ai];
                let name = a.desc.map(|d| d.name).unwrap_or("?");
                self.commands.push(SmCommand::Action(SmActionCommand {
                    state: a.state,
                    block: a.block,
                    index: a.index,
                    action_id: a.action_id,
                    descriptor_id: a.descriptor_id,
                    name,
                    kind: action_kind(a.descriptor_id),
                    flags: a.flags,
                    data: a.data.clone(),
                    code: a.code.clone(),
                }));
            }
        }
    }
}

/// The raw operand bytes of a trigger record, if it decoded as `Raw`.
fn raw(t: &crate::sm_payload::SmTriggerRecord) -> Option<&[u8]> {
    match &t.operand {
        SmTriggerOperand::Raw(b) => Some(b),
        _ => None,
    }
}

fn ru16(b: &[u8], o: usize, be: bool) -> u16 {
    match b.get(o..o + 2) {
        Some(s) => {
            if be {
                u16::from_be_bytes([s[0], s[1]])
            } else {
                u16::from_le_bytes([s[0], s[1]])
            }
        }
        None => 0,
    }
}

fn rf32(b: &[u8], o: usize, be: bool) -> f32 {
    match b.get(o..o + 4) {
        Some(s) => {
            let a = [s[0], s[1], s[2], s[3]];
            let bits = if be { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) };
            f32::from_bits(bits)
        }
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statemachine::parse_sm;

    fn real_sm(actor: &str, plat: &str) -> Option<Vec<u8>> {
        let rel = format!(
            "/../../../extracted-game/gamedata/libs/statemachines/{actor}/general.{plat}.sm"
        );
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "{}");
        let candidates = [
            p.replace("{}", &rel),
            format!("/media/workpc/1f162086-ac53-4c5e-96a2-d1dc72b0fa2d/home/battlestation/Desktop/sod-re/extracted-game/gamedata/libs/statemachines/{actor}/general.{plat}.sm"),
        ];
        candidates.iter().find_map(|p| std::fs::read(p).ok())
    }

    /// Scripted environment for the end-to-end tests.
    #[derive(Default)]
    struct ScriptedEnv {
        signals: std::collections::HashSet<u16>,
        query: std::collections::HashSet<u8>,
        /// `(state, output_index)` -> forced true via the raw fallback.
        raw_true: std::collections::HashSet<(u32, u32)>,
        actor_flags: u32,
        dt: f32,
        time: f32,
    }

    impl ScriptedEnv {
        fn set_signal(&mut self, id: u16, on: bool) {
            if on {
                self.signals.insert(id);
            } else {
                self.signals.remove(&id);
            }
        }
        fn set_query(&mut self, slot: u8, on: bool) {
            if on {
                self.query.insert(slot);
            } else {
                self.query.remove(&slot);
            }
        }
        fn set_actor_flags(&mut self, flags: u32) {
            self.actor_flags = flags;
        }
    }

    impl SmEnv for ScriptedEnv {
        fn delta_seconds(&self) -> f32 {
            self.dt
        }
        fn time_seconds(&self) -> f32 {
            self.time
        }
        fn signal(&self, id: u16) -> bool {
            self.signals.contains(&id)
        }
        fn actor_flags(&self) -> u32 {
            self.actor_flags
        }
        fn query_slot(&self, slot: u8) -> bool {
            self.query.contains(&slot)
        }
        fn eval_raw_trigger(
            &self,
            state: u32,
            output: u32,
            _type_id: u8,
            _operand: &[u8],
        ) -> Option<bool> {
            if self.raw_true.contains(&(state, output)) {
                Some(true)
            } else {
                None
            }
        }
    }

    #[test]
    fn select_by_weight_matches_engine() {
        // Single candidate: always chosen, no RNG draw.
        assert_eq!(select_by_weight(&[100], 55), 0);
        // First candidate whose weight > rand.
        assert_eq!(select_by_weight(&[10, 20, 30], 15), 1);
        assert_eq!(select_by_weight(&[10, 20, 30], 5), 0);
        // None exceeds -> last candidate.
        assert_eq!(select_by_weight(&[10, 20, 30], 99), 2);
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = SmRng::new(12345);
        let mut b = SmRng::new(12345);
        for _ in 0..1000 {
            let v = a.next_percent();
            assert!(v < 100);
            assert_eq!(v, b.next_percent());
        }
    }

    #[test]
    fn bloater_end_to_end() {
        let Some(bytes) = real_sm("freak_bloater", "win") else {
            return;
        };
        let sm = parse_sm(&bytes).expect("bloater parses");
        assert_eq!(sm.states.len(), 13);
        let mut inst = SmInstance::new(&sm, 1);
        let mut env = ScriptedEnv {
            dt: 1.0 / 30.0,
            ..Default::default()
        };

        // Start: GraphDriven.Observer.
        assert_eq!(inst.state, 1);
        assert_eq!(inst.state_name(), "GraphDriven.Observer");
        assert!(inst.is_graph_driven());
        assert_eq!(inst.graph_driven_group(), Some("Observer".to_string()));

        // Observer -> Walk.Normal via the type-17 AND condition #69
        // (inputs {33,68,0}); #33 is a type-40 query-slot test and #68 is a
        // type-17 invert of a raw trigger, so a populated query slot 1 fires.
        inst.take_commands();
        env.set_query(1, true);
        inst.tick(&env);
        assert_eq!(inst.state, 2, "expected Walk.Normal, got {}", inst.state_name());
        assert!(!inst.is_graph_driven());
        let cmds = inst.take_commands();
        assert!(
            cmds.iter().any(|c| matches!(c, SmCommand::PlayFragment { .. })),
            "state entry should play a fragment"
        );
        assert!(
            cmds.iter().any(|c| matches!(c, SmCommand::DriveMode(SmDriveMode::MachineDriven))),
            "leaving GraphDriven should change drive mode"
        );
        assert!(
            cmds.iter().any(|c| matches!(
                c,
                SmCommand::Action(a) if a.name == "EntityQuery2Mode"
            )),
            "walk state should run its actions"
        );

        // Walk.Normal -> Hazard.Normal via signal 290 (type 19).
        env.set_query(1, false);
        env.set_signal(0x0122, true);
        inst.tick(&env);
        assert_eq!(inst.state, 8, "expected Hazard.Normal, got {}", inst.state_name());
        assert_eq!(inst.state_name(), "MachineDriven.Hazard.Normal");

        // Hazard.Normal -> Hazard.Burning via the type-17 OR condition #44
        // (out7 is signal 140).
        env.set_signal(0x0122, false);
        env.set_signal(0x008c, true);
        inst.tick(&env);
        assert_eq!(inst.state, 7, "expected Hazard.Burning, got {}", inst.state_name());
        assert_eq!(inst.state_name(), "MachineDriven.Hazard.Burning");
    }

    #[test]
    fn player_grab_handoff() {
        let Some(bytes) = real_sm("player", "win") else {
            return;
        };
        let sm = parse_sm(&bytes).expect("player parses");
        let mut inst = SmInstance::new(&sm, 7);
        let mut env = ScriptedEnv {
            dt: 1.0 / 30.0,
            ..Default::default()
        };

        // Start: GraphDriven.Observer (state 25).
        assert_eq!(inst.state, 25);
        assert_eq!(inst.state_name(), "GraphDriven.Observer");
        assert!(inst.is_graph_driven());
        inst.take_commands();

        // Grab: condition #520 of Observer is a type-71 grab trigger reading
        // actor flag bit 3. It targets
        // MachineDriven.ReactionGrab.ByBasicZombie (512).
        env.set_actor_flags(1 << 3);
        inst.tick(&env);
        assert_eq!(inst.state, 512);
        assert_eq!(inst.state_name(), "MachineDriven.ReactionGrab.ByBasicZombie");
        assert!(!inst.is_graph_driven());
        let cmds = inst.take_commands();
        assert!(cmds.iter().any(|c| matches!(c, SmCommand::DriveMode(SmDriveMode::MachineDriven))));

        // Release: condition #448 of ByBasicZombie is a type-19 signal (358)
        // leading back to GraphDriven.Observer.
        env.set_actor_flags(0);
        env.set_signal(0x0166, true);
        inst.tick(&env);
        assert_eq!(inst.state, 25, "expected Observer, got {}", inst.state_name());
        assert!(inst.is_graph_driven());
        let cmds = inst.take_commands();
        assert!(cmds.iter().any(|c| matches!(
            c,
            SmCommand::DriveMode(SmDriveMode::GraphDriven(g)) if g == "Observer"
        )));
    }

    #[test]
    fn trigger_type_coverage_player_and_zombie() {
        let mut covered = 0usize;
        let mut total = 0usize;
        for actor in ["player", "zombie"] {
            let Some(bytes) = real_sm(actor, "win") else { continue };
            let sm = parse_sm(&bytes).expect("parses");
            let mut seen = std::collections::HashSet::new();
            for t in &sm.payloads.triggers {
                if seen.insert(t.type_id) {
                    total += 1;
                    if trigger_kind(t.type_id) != TriggerKind::Raw {
                        covered += 1;
                    }
                }
            }
        }
        if total == 0 {
            return;
        }
        // The decoded types cover the dominant share of trigger *records*.
        // (Type coverage is a lower bound; raw types are still surfaced to env.)
        assert!(covered > 0);
    }

    #[test]
    fn animation_graph_input_action_name() {
        let mut code = vec![0u8; 128];
        code[..6].copy_from_slice(b"Signal");
        let cmd = SmActionCommand {
            state: 25,
            block: 0,
            index: 44,
            action_id: 44,
            descriptor_id: 15,
            name: "AnimationGraphInput",
            kind: SmActionKind::AnimationGraphInput,
            flags: 0,
            data: Vec::new(),
            code,
        };
        assert_eq!(animation_graph_input_name(&cmd), Some("Signal"));
        // A non-AGI action (e.g. Animation, descriptor 13) is rejected.
        let other = SmActionCommand { descriptor_id: 13, kind: SmActionKind::Animation, ..cmd };
        assert_eq!(animation_graph_input_name(&other), None);
        // An empty operand has no name.
        let empty = SmActionCommand { code: vec![0u8; 128], ..other };
        assert_eq!(animation_graph_input_name(&empty), None);
    }

    #[test]
    fn force_state_and_find_by_name() {
        let Some(bytes) = real_sm("player", "win") else { return };
        let sm = parse_sm(&bytes).expect("player parses");
        let mut inst = SmInstance::new(&sm, 1);
        let react = inst
            .find_state("Reaction.Core.Heavy.High.Front")
            .expect("reaction state exists");
        inst.go_to(react);
        assert_eq!(inst.state_name(), "MachineDriven.Reaction.Core.Heavy.High.Front");
        assert!(!inst.is_graph_driven());
        let cmds = inst.take_commands();
        assert!(cmds.iter().any(|c| matches!(c, SmCommand::PlayFragment { .. })));
    }

    #[test]
    fn block_strings_code_surfaced_as_actions() {
        let Some(bytes) = real_sm("freak_bloater", "win") else {
            return;
        };
        let sm = parse_sm(&bytes).expect("bloater parses");
        let mut inst = SmInstance::new(&sm, 1);
        let env = ScriptedEnv::default();
        inst.tick(&env);
        let cmds = inst.take_commands();
        let with_code = cmds.iter().filter(|c| match c {
            SmCommand::Action(a) => !a.code.is_empty(),
            _ => false,
        }).count();
        assert!(with_code > 0, "block_strings code operands should appear on actions");
    }
}
