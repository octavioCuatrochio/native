//! Runtime goal-pipe execution: `CPipeUser` / `CGoalPipe` / `CGoalOp` stepping.
//!
//! The script boot in [`super::ai`] fills [`super::ai_data::AiState`] with the
//! goal pipes the game authors, but nothing runs them. This module is the
//! executor: it attaches a pipe to an agent, steps its goals each frame and
//! reports the original return semantics.
//!
//! ## What the original does
//!
//! * `CScriptBind_AI::PushGoal` (`FUN_00a04510`, primary) looks a goal name up
//!   in the goal-op registry and constructs the matching `CGoalOp`. The registry
//!   is the 57-entry table at `0x18fb128` (`{index, name*}`), searched by
//!   `FUN_00a03ee0` (name -> index, sentinel index `0x39`) and enumerated by
//!   `FUN_00a03f30` (index -> name). The per-index constructors are the jump
//!   tables at `0xa03e08` (indices 0..=0x34) and `0xa0520c` (0..=0x38, includes
//!   the Undead Labs ops).
//! * Each `CGoalOp` has a two-word vtable; slot 2 (`+0x8`) is `Execute`. The
//!   result enum is an integer: `1` = in progress, `4` = done, `2` = failed
//!   (see `COPBranch::Execute` `0x9f6810`, `COPTimeout` `0x9fafd0`,
//!   `COPSignal` `0x9fa100`, `COPWaitSignal` `0x9fb100`).
//! * `CGoalPipe::Execute` is `FUN_009f4500`. It walks the goal array
//!   (`[pipe+0x30]..[pipe+0x34]`, stride `0x50`), skipping goals already marked
//!   done (`[flags+i*2]`), calls `Execute`, and:
//!   * result `1`: if the goal's blocking flag (`[flags+i*2+1]`) is set the walk
//!     stops; otherwise the walk continues to the next goal;
//!   * any other result: the goal is marked done and the walk continues.
//!   It returns `4` (done) only when no goal returned `1` this frame, else `1`.
//!
//! This module reproduces that walk exactly. Movement and perception ops do not
//! invent physics: they take their inputs from a [`PipeEnv`] the lead
//! implements (agent position, target position, `dt`, path-follow status, ...).
//! Signals raised by `signal` goals go to an outbox the lead routes to the
//! script `OnSignal` handlers.

use super::ai_data::{AiState, Goal};
use super::world::EntityId;
use std::collections::HashMap;

/// `EGoalOpResult` (`GoalOp.h`). Codes recovered from the goal ops' `Execute`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalResult {
    /// `0`: no result yet.
    None,
    /// `1`: the goal is still running.
    InProgress,
    /// `2`: the goal failed.
    Failed,
    /// `3`: legacy done (not emitted by the ops this module implements).
    Done,
    /// `4`: the goal completed; the pipe treats any non-`1` as finished.
    Succeeded,
}

impl GoalResult {
    pub fn code(self) -> i32 {
        match self {
            GoalResult::None => 0,
            GoalResult::InProgress => 1,
            GoalResult::Failed => 2,
            GoalResult::Done => 3,
            GoalResult::Succeeded => 4,
        }
    }

    pub fn from_code(c: i32) -> Self {
        match c {
            1 => GoalResult::InProgress,
            2 => GoalResult::Failed,
            3 => GoalResult::Done,
            4 => GoalResult::Succeeded,
            _ => GoalResult::None,
        }
    }

    /// The original marks a goal done whenever its `Execute` does not return
    /// `eGOR_IN_PROGRESS` (`FUN_009f4500`, `cmp eax,0x1`).
    fn is_finished(self) -> bool {
        self != GoalResult::InProgress && self != GoalResult::None
    }
}

/// A signal raised by a `signal` goal, for the lead to hand to script
/// `OnSignal`-style handlers.
#[derive(Clone, Debug, PartialEq)]
pub struct RaisedSignal {
    pub sender: EntityId,
    pub name: String,
    pub filter: i64,
    /// The goal's first parameter (the `signal` op's mode flag).
    pub mode: i64,
}

/// Result the lead's path follower reports for a movement request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PathStatus {
    #[default]
    InProgress,
    Reached,
    Failed,
}

/// Result of a `locate` / `acqtarget` / `ul_locatevisible` request.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum LocateResult {
    Found([f32; 3]),
    #[default]
    NotFound,
    Failed,
}

/// Non-movement commands an op asks the lead to perform.
#[derive(Clone, Debug, PartialEq)]
pub enum AgentCommand {
    /// `run` / `speed`: set the desired locomotion speed (0..1-ish).
    SetSpeed(f32),
    /// `bodypos`: set the body position/stance (`BODYPOS_*`).
    SetBodyPos(i64),
    /// `firecmd`: set the fire mode and its flags (`FIREMODE_*`).
    SetFireMode { mode: i64, flags: i64 },
    /// `lookat`: turn toward the current target; angles are degrees.
    LookAt { yaw_min: f32, yaw_max: f32, continuous: bool },
    /// `+lookat` with no target: stop looking.
    ClearLookAt,
    /// `acqtarget`: re-acquire a target by filter string.
    AcquireTarget { filter: String },
    /// `tacticalpos`: ask the tactical-position system for a point.
    TacticalPos { query: String },
    /// `ul_face`: face the current target.
    FaceTarget { flag: i64 },
    /// `ul_runforward`: toggle the run-forward movement flag.
    RunForward(bool),
    /// `devalue`: devalue/clear tactical points (`devalue` op).
    Devalue { a: i64, b: i64 },
    /// `backoff`: move away from the current target.
    BackOff { dist: f32, flags: i64 },
}

/// The environment the lead implements. Movement/perception ops pull every
/// world input through here; the executor never invents physics or pathfinding.
pub trait PipeEnv {
    /// Frame time in seconds.
    fn dt(&self) -> f32;

    /// World position of the agent (CryEngine Z-up metres). `[0,0,0]` if unknown.
    fn agent_pos(&self, agent: EntityId) -> [f32; 3];

    /// The agent's current target position (attention/ref-point), if any.
    fn target_pos(&self, agent: EntityId) -> Option<[f32; 3]>;

    /// Move the agent toward `pos` at `speed`, honouring the `AILASTOPRES_*` /
    /// `AI_REQUEST_*` flag bitfield. The lead drives pathfinding and animation.
    fn move_to(&mut self, agent: EntityId, pos: [f32; 3], speed: f32, flags: i64) -> PathStatus;

    /// Stick to the agent's current target (`stick`). `mode` is the
    /// `STICK_*` bitfield, `dist` the desired stand-off distance.
    fn stick(&mut self, agent: EntityId, dist: f32, flags: i64, mode: i64) -> PathStatus;

    /// Resolve a `locate` / `acqtarget` / `ul_locatevisible` object kind.
    fn locate(&mut self, agent: EntityId, kind: &str, radius: f32, flags: i64) -> LocateResult;

    /// Ask the tactical-position system for a point from a registered query.
    fn tactical_pos(&mut self, agent: EntityId, query: &str) -> Option<[f32; 3]>;

    /// Drain a delivered signal for the agent; `true` if it matched `name`.
    fn take_signal(&mut self, agent: EntityId, name: &str) -> bool;

    /// Perform a non-movement command (`run`, `bodypos`, `firecmd`, ...).
    fn command(&mut self, agent: EntityId, cmd: AgentCommand);
}

/// Which goal op a name refers to, after the leading `+` (non-blocking marker)
/// is stripped. `Unknown` names are recorded so the lead can see gaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalOpKind {
    Signal,
    Timeout,
    Wait,
    WaitSignal,
    Branch,
    Run,
    Clear,
    Locate,
    AcqTarget,
    Approach,
    Stick,
    Backoff,
    LookAt,
    BodyPos,
    FireCmd,
    TacticalPos,
    Devalue,
    UlFace,
    UlRunForward,
    UlLocateVisible,
    UlLocateEntity,
    UlLocateZombie,
    Unknown,
}

impl GoalOpKind {
    pub fn parse(name: &str) -> Self {
        let n = name.strip_prefix('+').unwrap_or(name);
        match n.to_ascii_lowercase().as_str() {
            "signal" => GoalOpKind::Signal,
            "timeout" => GoalOpKind::Timeout,
            "wait" => GoalOpKind::Wait,
            "waitsignal" => GoalOpKind::WaitSignal,
            "branch" => GoalOpKind::Branch,
            "run" | "speed" => GoalOpKind::Run,
            "clear" => GoalOpKind::Clear,
            "locate" => GoalOpKind::Locate,
            "acqtarget" => GoalOpKind::AcqTarget,
            "approach" => GoalOpKind::Approach,
            "stick" => GoalOpKind::Stick,
            "backoff" => GoalOpKind::Backoff,
            "lookat" => GoalOpKind::LookAt,
            "bodypos" => GoalOpKind::BodyPos,
            "firecmd" => GoalOpKind::FireCmd,
            "tacticalpos" => GoalOpKind::TacticalPos,
            "devalue" => GoalOpKind::Devalue,
            "ul_face" => GoalOpKind::UlFace,
            "ul_runforward" => GoalOpKind::UlRunForward,
            "ul_locatevisible" => GoalOpKind::UlLocateVisible,
            "ul_locateentity" => GoalOpKind::UlLocateEntity,
            "ul_locatezombie" => GoalOpKind::UlLocateZombie,
            _ => GoalOpKind::Unknown,
        }
    }

    /// Whether this module implements the op (used by the coverage report and
    /// the "no unknown ops" test).
    pub fn is_implemented(self) -> bool {
        !matches!(self, GoalOpKind::Unknown)
    }
}

fn param_num(g: &Goal, i: usize) -> Option<f64> {
    g.params.get(i).and_then(|p| p.as_num())
}

fn param_i64(g: &Goal, i: usize) -> Option<i64> {
    param_num(g, i).map(|n| n as i64)
}

fn param_f32(g: &Goal, i: usize) -> Option<f32> {
    param_num(g, i).map(|n| n as f32)
}

fn param_str(g: &Goal, i: usize) -> Option<&str> {
    g.params.get(i).and_then(|p| p.as_str())
}

/// Branch types (`AIGOALPIPE_*` / `IF_*`), from the constants table.
const BRANCH_ALWAYS: i64 = 46;
const IF_LASTOP_FAILED: i64 = 44;
const IF_LASTOP_SUCCEED: i64 = 45;

/// Per-run state of one goal inside a [`PipeUser`]. Mirrors the two bytes the
/// original keeps per goal (`[flags+i*2]` done, `[flags+i*2+1]` blocking) plus
/// the op-specific scratch the goal needs between frames.
#[derive(Clone, Debug, Default)]
struct GoalState {
    done: bool,
    started: bool,
    /// Accumulated seconds (timeout/wait).
    elapsed: f32,
    /// A one-shot flag: signal raised, target acquired, etc.
    flag: bool,
    /// Cached random timeout duration.
    duration: f32,
    /// Cached last move/locate result.
    path: Option<PathStatus>,
}

/// A goal pipe attached to one agent (`CPipeUser`'s current pipe).
#[derive(Clone, Debug)]
pub struct PipeUser {
    pub agent: EntityId,
    /// Name of the pipe definition currently running.
    pub pipe: String,
    /// Per-goal run state, parallel to the pipe's goal list.
    states: Vec<GoalState>,
    /// The result of the most recently finished goal (`m_lastResult`).
    pub last_result: GoalResult,
    /// The position `locate` resolved, used by `approach`/`stick`/`lookat`.
    pub target: Option<[f32; 3]>,
    /// Set when the pipe has run to its end and should be popped.
    pub finished: bool,
}

impl PipeUser {
    fn new(agent: EntityId, pipe: &str, goals: usize) -> Self {
        PipeUser {
            agent,
            pipe: pipe.to_string(),
            states: vec![GoalState::default(); goals],
            last_result: GoalResult::None,
            target: None,
            finished: false,
        }
    }

    /// Reset all goals so the pipe restarts (used when re-selecting a pipe).
    pub fn reset(&mut self) {
        for s in &mut self.states {
            *s = GoalState::default();
        }
        self.last_result = GoalResult::None;
        self.target = None;
        self.finished = false;
    }
}

/// The goal-pipe executor: owns the pipe definitions copied out of
/// [`AiState`], the running pipes and the raised-signal outbox.
pub struct PipeExecutor {
    /// Pipe definitions keyed by lower-case name.
    pipes: HashMap<String, Vec<Goal>>,
    /// Label -> goal index, per pipe (built lazily).
    labels: HashMap<String, HashMap<String, usize>>,
    /// Running pipes keyed by agent.
    users: HashMap<EntityId, PipeUser>,
    /// Signals raised this frame, for the lead to route to script handlers.
    pub signals: Vec<RaisedSignal>,
    /// Goal-op names seen that this module does not implement, with counts.
    pub unknown: HashMap<String, u32>,
}

impl PipeExecutor {
    /// Build from the script registry.
    pub fn new(state: &AiState) -> Self {
        let mut pipes = HashMap::new();
        for p in &state.goal_pipes {
            pipes.insert(p.name.to_ascii_lowercase(), p.goals.clone());
        }
        PipeExecutor {
            pipes,
            labels: HashMap::new(),
            users: HashMap::new(),
            signals: Vec::new(),
            unknown: HashMap::new(),
        }
    }

    pub fn pipe_names(&self) -> impl Iterator<Item = &str> {
        self.pipes.keys().map(|s| s.as_str())
    }

    pub fn has_pipe(&self, name: &str) -> bool {
        self.pipes.contains_key(&name.to_ascii_lowercase())
    }

    fn labels_for(&mut self, pipe: &str) -> &HashMap<String, usize> {
        let key = pipe.to_ascii_lowercase();
        if !self.labels.contains_key(&key) {
            let mut map = HashMap::new();
            if let Some(goals) = self.pipes.get(&key) {
                for (i, g) in goals.iter().enumerate() {
                    if g.label {
                        if let Some(name) = g.params.first().and_then(|p| p.as_str()) {
                            map.insert(name.to_ascii_lowercase(), i);
                        }
                    }
                }
            }
            self.labels.insert(key.clone(), map);
        }
        self.labels.get(&key).unwrap()
    }

    /// Push a pipe onto an agent (`AI.SetGoalPipe` / `CGoalPipe::PushGoal`).
    pub fn set_pipe(&mut self, agent: EntityId, pipe: &str) -> bool {
        let key = pipe.to_ascii_lowercase();
        let Some(goals) = self.pipes.get(&key) else {
            return false;
        };
        let n = goals.len();
        self.users.insert(agent, PipeUser::new(agent, &key, n));
        true
    }

    pub fn user(&self, agent: EntityId) -> Option<&PipeUser> {
        self.users.get(&agent)
    }

    pub fn remove(&mut self, agent: EntityId) {
        self.users.remove(&agent);
    }

    /// Step one agent's pipe for a frame. Returns the pipe-level result:
    /// `InProgress` while goals remain, `Succeeded` when the pipe runs out.
    pub fn step(&mut self, agent: EntityId, env: &mut dyn PipeEnv) -> GoalResult {
        let Some(pipe_name) = self.users.get(&agent).map(|u| u.pipe.clone()) else {
            return GoalResult::None;
        };
        let Some(goals) = self.pipes.get(&pipe_name).cloned() else {
            return GoalResult::None;
        };
        let n = goals.len();
        let labels = self.labels_for(&pipe_name).clone();
        let user = self.users.get_mut(&agent).unwrap();

        let mut in_progress = 0usize;
        let mut i = 0usize;
        while i < n {
            let goal = &goals[i];
            let st = &user.states[i];

            // Labels are markers, never executed.
            if goal.label || st.done {
                i += 1;
                continue;
            }

            let kind = GoalOpKind::parse(&goal.name);
            if !kind.is_implemented() {
                *self.unknown.entry(goal.name.clone()).or_insert(0) += 1;
            }

            // A branch goal never reports in progress; it jumps (or not) and the
            // walk continues at the target label. `COPBranch::Execute`
            // (0x9f6810) always returns done.
            if kind == GoalOpKind::Branch {
                let ty = param_i64(goal, 1).unwrap_or(BRANCH_ALWAYS);
                let take = match ty {
                    IF_LASTOP_FAILED => user.last_result == GoalResult::Failed,
                    IF_LASTOP_SUCCEED => {
                        matches!(user.last_result, GoalResult::Succeeded | GoalResult::Done)
                    }
                    _ => true, // BRANCH_ALWAYS / IF_ACTIVE_GOALS
                };
                user.states[i].done = true;
                user.last_result = GoalResult::Succeeded;
                if take {
                    if let Some(next) = branch_target(goal, i, n, &labels) {
                        if next > i {
                            // Forward branch: skip every goal up to the target.
                            for k in i..next {
                                user.states[k].done = true;
                            }
                        } else {
                            // Backward branch (loop): reset from the label on;
                            // the frame ends and the next frame re-runs them.
                            for s in user.states.iter_mut().skip(next) {
                                *s = GoalState::default();
                            }
                        }
                        break;
                    }
                }
                i += 1;
                continue;
            }

            let result = execute_goal(
                kind,
                goal,
                &mut user.states[i],
                agent,
                &mut user.target,
                env,
                &mut self.signals,
            );
            if result != GoalResult::InProgress {
                user.last_result = result;
            }

            if result == GoalResult::InProgress {
                in_progress += 1;
                if goal.blocking {
                    break;
                }
            } else if result.is_finished() {
                user.states[i].done = true;
            }

            i += 1;
        }

        if in_progress == 0 {
            user.finished = true;
            GoalResult::Succeeded
        } else {
            GoalResult::InProgress
        }
    }
}

/// Resolve a branch goal's jump target: a named label, or a relative/absolute
/// goal index (a negative offset is relative to the branch goal; `-1` loops).
fn branch_target(
    goal: &Goal,
    i: usize,
    n: usize,
    labels: &HashMap<String, usize>,
) -> Option<usize> {
    if let Some(target) = param_str(goal, 0) {
        return labels.get(&target.to_ascii_lowercase()).copied();
    }
    let off = param_i64(goal, 0)?;
    if off < 0 {
        Some(((i as i64 + 1 + off).max(0)) as usize)
    } else {
        Some((off as usize).min(n.saturating_sub(1)))
    }
}

/// The per-goal `Execute` (vtable slot 2). Implements the ops the shipped pipes
/// use; env-dependent ops pull their inputs from [`PipeEnv`].
fn execute_goal(
    kind: GoalOpKind,
    goal: &Goal,
    st: &mut GoalState,
    agent: EntityId,
    target: &mut Option<[f32; 3]>,
    env: &mut dyn PipeEnv,
    signals: &mut Vec<RaisedSignal>,
) -> GoalResult {
    match kind {
        // -- pure flow ------------------------------------------------------
        GoalOpKind::Timeout | GoalOpKind::Wait => {
            if !st.started {
                st.started = true;
                st.elapsed = 0.0;
                let lo = param_f32(goal, 0).unwrap_or(0.0);
                let hi = param_f32(goal, 1).unwrap_or(lo);
                st.duration = if hi <= lo { lo } else { random_between(lo, hi, agent, goal) };
            }
            st.elapsed += env.dt();
            if st.elapsed >= st.duration {
                GoalResult::Succeeded
            } else {
                GoalResult::InProgress
            }
        }
        GoalOpKind::Signal => {
            // COPSignal (0x9fa100): raise on the first Execute, return done on
            // the next.
            if st.flag {
                st.flag = false;
                return GoalResult::Succeeded;
            }
            let mode = param_i64(goal, 0).unwrap_or(0);
            let name = param_str(goal, 1).unwrap_or("").to_string();
            let filter = param_i64(goal, 2).unwrap_or(0);
            signals.push(RaisedSignal { sender: agent, name, filter, mode });
            st.flag = true;
            GoalResult::InProgress
        }
        GoalOpKind::WaitSignal => {
            if st.flag {
                st.flag = false;
                return GoalResult::Succeeded;
            }
            let name = param_str(goal, 0).unwrap_or("");
            if env.take_signal(agent, name) {
                st.flag = true;
            }
            GoalResult::InProgress
        }
        GoalOpKind::Branch => {
            // Handled in `PipeExecutor::step` (it needs the pipe's label map).
            GoalResult::Succeeded
        }
        GoalOpKind::Run => {
            let speed = param_f32(goal, 0).unwrap_or(1.0);
            env.command(agent, AgentCommand::SetSpeed(speed));
            GoalResult::Succeeded
        }
        GoalOpKind::Clear => {
            *target = None;
            env.command(agent, AgentCommand::ClearLookAt);
            GoalResult::Succeeded
        }

        // -- movement / perception (via PipeEnv) ---------------------------
        GoalOpKind::Locate | GoalOpKind::AcqTarget => {
            let kind_name = param_str(goal, 0).unwrap_or("");
            let radius = param_f32(goal, 1).unwrap_or(0.0);
            let flags = param_i64(goal, 2).unwrap_or(0);
            match env.locate(agent, kind_name, radius, flags) {
                LocateResult::Found(p) => {
                    *target = Some(p);
                    GoalResult::Succeeded
                }
                LocateResult::NotFound => GoalResult::Failed,
                LocateResult::Failed => GoalResult::Failed,
            }
        }
        GoalOpKind::Approach => {
            let speed = param_f32(goal, 0).unwrap_or(1.0);
            let flags = param_i64(goal, 1).unwrap_or(0);
            let Some(pos) = target.or_else(|| env.target_pos(agent)) else {
                return GoalResult::Failed;
            };
            let r = env.move_to(agent, pos, speed, flags);
            st.path = Some(r);
            match r {
                PathStatus::InProgress => GoalResult::InProgress,
                PathStatus::Reached => GoalResult::Succeeded,
                PathStatus::Failed => GoalResult::Failed,
            }
        }
        GoalOpKind::Stick => {
            let dist = param_f32(goal, 0).unwrap_or(0.0);
            let flags = param_i64(goal, 1).unwrap_or(0);
            let mode = param_i64(goal, 2).unwrap_or(0);
            let r = env.stick(agent, dist, flags, mode);
            st.path = Some(r);
            match r {
                PathStatus::InProgress => GoalResult::InProgress,
                PathStatus::Reached => GoalResult::Succeeded,
                PathStatus::Failed => GoalResult::Failed,
            }
        }
        GoalOpKind::Backoff => {
            let dist = param_f32(goal, 0).unwrap_or(0.0);
            let flags = param_i64(goal, 1).unwrap_or(0);
            env.command(agent, AgentCommand::BackOff { dist, flags });
            GoalResult::Succeeded
        }
        GoalOpKind::LookAt => {
            if goal.params.is_empty() {
                env.command(agent, AgentCommand::ClearLookAt);
                return GoalResult::Succeeded;
            }
            let yaw_min = param_f32(goal, 0).unwrap_or(0.0);
            let yaw_max = param_f32(goal, 1).unwrap_or(yaw_min);
            let continuous = param_i64(goal, 4).unwrap_or(0) != 0
                || param_i64(goal, 3).unwrap_or(0) != 0;
            env.command(
                agent,
                AgentCommand::LookAt { yaw_min, yaw_max, continuous },
            );
            GoalResult::Succeeded
        }
        GoalOpKind::BodyPos => {
            let bp = param_i64(goal, 0).unwrap_or(0);
            env.command(agent, AgentCommand::SetBodyPos(bp));
            GoalResult::Succeeded
        }
        GoalOpKind::FireCmd => {
            let mode = param_i64(goal, 0).unwrap_or(0);
            let flags = param_i64(goal, 1).unwrap_or(0);
            env.command(agent, AgentCommand::SetFireMode { mode, flags });
            GoalResult::Succeeded
        }
        GoalOpKind::TacticalPos => {
            let query = param_str(goal, 0).unwrap_or("").to_string();
            match env.tactical_pos(agent, &query) {
                Some(p) => {
                    *target = Some(p);
                    env.command(agent, AgentCommand::TacticalPos { query });
                    GoalResult::Succeeded
                }
                None => GoalResult::Failed,
            }
        }
        GoalOpKind::Devalue => {
            let a = param_i64(goal, 0).unwrap_or(0);
            let b = param_i64(goal, 1).unwrap_or(0);
            env.command(agent, AgentCommand::Devalue { a, b });
            GoalResult::Succeeded
        }
        GoalOpKind::UlFace => {
            let flag = param_i64(goal, 0).unwrap_or(0);
            env.command(agent, AgentCommand::FaceTarget { flag });
            GoalResult::Succeeded
        }
        GoalOpKind::UlRunForward => {
            let on = param_i64(goal, 0).unwrap_or(0) != 0;
            env.command(agent, AgentCommand::RunForward(on));
            GoalResult::Succeeded
        }
        GoalOpKind::UlLocateVisible | GoalOpKind::UlLocateEntity | GoalOpKind::UlLocateZombie => {
            let kind_name = param_str(goal, 0).unwrap_or("");
            let radius = param_f32(goal, 1).unwrap_or(0.0);
            let flags = param_i64(goal, 2).unwrap_or(0);
            match env.locate(agent, kind_name, radius, flags) {
                LocateResult::Found(p) => {
                    *target = Some(p);
                    GoalResult::Succeeded
                }
                _ => GoalResult::Failed,
            }
        }

        GoalOpKind::Unknown => GoalResult::Failed,
    }
}

/// Deterministic stand-in for the engine RNG used to pick a `timeout` duration
/// in `[lo, hi]`. Tests need reproducible pipes, so this is a pure hash.
fn random_between(lo: f32, hi: f32, agent: EntityId, goal: &Goal) -> f32 {
    let mut x = agent as u64 ^ 0x9e37_79b9_7f4a_7c15;
    for b in goal.name.bytes() {
        x = (x ^ b as u64).wrapping_mul(0x1000_0000_01b3);
    }
    x ^= x >> 29;
    let frac = (x & 0xffff) as f32 / 65535.0;
    lo + (hi - lo) * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cry::ai_data::{GoalParam, GoalPipe};
    use crate::cry::{ai, ScriptHost};
    use crate::crypak::Vfs;
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;

    fn goal(name: &str, blocking: bool, params: Vec<GoalParam>) -> Goal {
        Goal { name: name.into(), blocking, params, label: false }
    }

    fn label(name: &str) -> Goal {
        Goal {
            name: String::new(),
            blocking: false,
            params: vec![GoalParam::Str(name.into())],
            label: true,
        }
    }

    fn executor(pipes: Vec<(&str, Vec<Goal>)>) -> PipeExecutor {
        let mut st = AiState::default();
        for (name, goals) in pipes {
            st.goal_pipes.push(GoalPipe { name: name.into(), goals });
        }
        st.pipe_index = st
            .goal_pipes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.to_ascii_lowercase(), i))
            .collect();
        PipeExecutor::new(&st)
    }

    /// Records every request and lets tests script the replies.
    #[derive(Default)]
    struct StubEnv {
        dt: f32,
        pos: [f32; 3],
        target: Option<[f32; 3]>,
        path: PathStatus,
        locate: LocateResult,
        tactical: Option<[f32; 3]>,
        signals: Vec<String>,
        commands: Vec<AgentCommand>,
    }

    impl PipeEnv for StubEnv {
        fn dt(&self) -> f32 {
            self.dt
        }
        fn agent_pos(&self, _a: EntityId) -> [f32; 3] {
            self.pos
        }
        fn target_pos(&self, _a: EntityId) -> Option<[f32; 3]> {
            self.target
        }
        fn move_to(&mut self, _a: EntityId, _p: [f32; 3], _s: f32, _f: i64) -> PathStatus {
            self.path
        }
        fn stick(&mut self, _a: EntityId, _d: f32, _f: i64, _m: i64) -> PathStatus {
            self.path
        }
        fn locate(&mut self, _a: EntityId, _k: &str, _r: f32, _f: i64) -> LocateResult {
            self.locate
        }
        fn tactical_pos(&mut self, _a: EntityId, _q: &str) -> Option<[f32; 3]> {
            self.tactical
        }
        fn take_signal(&mut self, _a: EntityId, name: &str) -> bool {
            if let Some(i) = self.signals.iter().position(|s| s == name) {
                self.signals.remove(i);
                true
            } else {
                false
            }
        }
        fn command(&mut self, _a: EntityId, cmd: AgentCommand) {
            self.commands.push(cmd);
        }
    }

    #[test]
    fn timeout_runs_then_completes() {
        let mut ex = executor(vec![(
            "p",
            vec![goal("timeout", true, vec![GoalParam::Num(1.0), GoalParam::Num(1.0)])],
        )]);
        ex.set_pipe(7, "p");
        let mut env = StubEnv { dt: 0.25, ..Default::default() };
        assert_eq!(ex.step(7, &mut env), GoalResult::InProgress);
        assert_eq!(ex.step(7, &mut env), GoalResult::InProgress);
        assert_eq!(ex.step(7, &mut env), GoalResult::InProgress);
        assert_eq!(ex.step(7, &mut env), GoalResult::Succeeded);
        assert!(ex.user(7).unwrap().finished);
    }

    #[test]
    fn non_blocking_run_and_bodypos_complete_in_one_frame() {
        let mut ex = executor(vec![(
            "p",
            vec![
                goal("run", false, vec![GoalParam::Num(2.0)]),
                goal("bodypos", false, vec![GoalParam::Num(0.0)]),
            ],
        )]);
        ex.set_pipe(1, "p");
        let mut env = StubEnv::default();
        assert_eq!(ex.step(1, &mut env), GoalResult::Succeeded);
        assert_eq!(
            env.commands,
            vec![
                AgentCommand::SetSpeed(2.0),
                AgentCommand::SetBodyPos(0),
            ]
        );
    }

    #[test]
    fn blocking_signal_raises_then_advances() {
        let mut ex = executor(vec![(
            "p",
            vec![
                goal(
                    "signal",
                    true,
                    vec![
                        GoalParam::Num(1.0),
                        GoalParam::Str("GO".into()),
                        GoalParam::Num(0.0),
                    ],
                ),
                goal("run", false, vec![GoalParam::Num(1.0)]),
            ],
        )]);
        ex.set_pipe(3, "p");
        let mut env = StubEnv::default();
        // Frame 1: signal raised, blocking -> in progress.
        assert_eq!(ex.step(3, &mut env), GoalResult::InProgress);
        assert_eq!(ex.signals.len(), 1);
        assert_eq!(ex.signals[0].name, "GO");
        // Frame 2: signal reports done, run executes.
        assert_eq!(ex.step(3, &mut env), GoalResult::Succeeded);
        assert_eq!(env.commands, vec![AgentCommand::SetSpeed(1.0)]);
    }

    #[test]
    fn branch_always_loops_to_label() {
        let mut ex = executor(vec![(
            "loop",
            vec![
                label("TOP"),
                goal("run", false, vec![GoalParam::Num(1.0)]),
                goal(
                    "branch",
                    false,
                    vec![
                        GoalParam::Str("TOP".into()),
                        GoalParam::Num(BRANCH_ALWAYS as f64),
                    ],
                ),
            ],
        )]);
        ex.set_pipe(9, "loop");
        let mut env = StubEnv::default();
        for _ in 0..5 {
            assert_eq!(ex.step(9, &mut env), GoalResult::Succeeded);
        }
        // run executes every frame because the branch resets the label target.
        assert_eq!(env.commands.len(), 5);
    }

    #[test]
    fn branch_on_failed_takes_only_on_failure() {
        // locate fails (NotFound) -> branch on failed jumps to the fallback.
        let mut ex = executor(vec![(
            "p",
            vec![
                goal("locate", false, vec![GoalParam::Str("enemy".into())]),
                goal(
                    "+branch",
                    false,
                    vec![
                        GoalParam::Str("FALLBACK".into()),
                        GoalParam::Num(IF_LASTOP_FAILED as f64),
                    ],
                ),
                goal("bodypos", false, vec![GoalParam::Num(1.0)]),
                label("FALLBACK"),
                goal("bodypos", false, vec![GoalParam::Num(2.0)]),
            ],
        )]);
        ex.set_pipe(4, "p");
        let mut env = StubEnv { locate: LocateResult::NotFound, ..Default::default() };
        // Frame 1: locate fails, branch jumps forward past the non-fallback goal.
        assert_eq!(ex.step(4, &mut env), GoalResult::Succeeded);
        assert!(env.commands.is_empty());
        // Frame 2: execution resumes at the FALLBACK label.
        assert_eq!(ex.step(4, &mut env), GoalResult::Succeeded);
        assert_eq!(env.commands, vec![AgentCommand::SetBodyPos(2)]);
    }

    #[test]
    fn approach_uses_path_status() {
        let mut ex = executor(vec![(
            "p",
            vec![goal("approach", true, vec![GoalParam::Num(1.0), GoalParam::Num(0.0)])],
        )]);
        ex.set_pipe(5, "p");
        let mut env = StubEnv {
            target: Some([1.0, 2.0, 3.0]),
            path: PathStatus::InProgress,
            ..Default::default()
        };
        assert_eq!(ex.step(5, &mut env), GoalResult::InProgress);
        env.path = PathStatus::Reached;
        assert_eq!(ex.step(5, &mut env), GoalResult::Succeeded);
    }

    #[test]
    fn locate_then_stick_and_lookat() {
        let mut ex = executor(vec![(
            "p",
            vec![
                goal("+locate", false, vec![GoalParam::Str("player".into())]),
                goal("+stick", false, vec![GoalParam::Num(1.5), GoalParam::Num(0.0), GoalParam::Num(0.0)]),
                goal("+lookat", false, vec![GoalParam::Num(0.0), GoalParam::Num(0.0)]),
                goal("ul_runforward", false, vec![GoalParam::Num(1.0)]),
                goal("ul_face", false, vec![GoalParam::Num(0.0)]),
                goal("firecmd", false, vec![GoalParam::Num(1.0), GoalParam::Num(0.0)]),
                goal("devalue", false, vec![GoalParam::Num(1.0), GoalParam::Num(0.0)]),
                goal("backoff", false, vec![GoalParam::Num(5.0), GoalParam::Num(2.0)]),
                goal("tacticalpos", false, vec![GoalParam::Str("q".into())]),
            ],
        )]);
        ex.set_pipe(6, "p");
        let mut env = StubEnv {
            locate: LocateResult::Found([9.0, 9.0, 9.0]),
            path: PathStatus::Reached,
            tactical: Some([1.0, 1.0, 1.0]),
            ..Default::default()
        };
        let r = ex.step(6, &mut env);
        assert_eq!(r, GoalResult::Succeeded);
        assert!(ex.unknown.is_empty(), "unexpected unknown ops: {:?}", ex.unknown);
        assert!(env.commands.contains(&AgentCommand::RunForward(true)));
        assert!(env.commands.contains(&AgentCommand::FaceTarget { flag: 0 }));
    }

    #[test]
    fn waitsignal_completes_when_delivered() {
        let mut ex = executor(vec![(
            "p",
            vec![goal("waitsignal", true, vec![GoalParam::Str("PING".into())])],
        )]);
        ex.set_pipe(8, "p");
        let mut env = StubEnv::default();
        assert_eq!(ex.step(8, &mut env), GoalResult::InProgress);
        env.signals.push("PING".into());
        assert_eq!(ex.step(8, &mut env), GoalResult::InProgress); // sets flag
        assert_eq!(ex.step(8, &mut env), GoalResult::Succeeded); // reports done
    }

    // -- real registry ------------------------------------------------------

    const DEFAULT_GAME: &str =
        "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";

    fn collect_paks(root: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(root) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    collect_paks(&p, out);
                } else if p.extension().map(|x| x == "pak").unwrap_or(false) {
                    out.push(p);
                }
            }
        }
    }

    fn game_vfs() -> Option<Rc<Vfs>> {
        let root = std::env::var("SOD_GAME_DIR").unwrap_or_else(|_| DEFAULT_GAME.to_string());
        let root = Path::new(&root);
        if !root.is_dir() {
            return None;
        }
        let mut paks = Vec::new();
        collect_paks(root, &mut paks);
        if paks.is_empty() {
            return None;
        }
        paks.sort();
        let mut vfs = Vfs::new();
        for p in &paks {
            let _ = vfs.mount(p);
        }
        Some(Rc::new(vfs))
    }

    #[test]
    fn real_zombie_pipe_steps_without_unknown_ops() {
        let Some(vfs) = game_vfs() else {
            eprintln!("skipping: game paks not present");
            return;
        };
        let host = ScriptHost::new(vfs).expect("script host");
        ai::install(&host.lua, &host.ctx).expect("ai install");
        ai::boot(&host.lua, &host.ctx).expect("ai boot");
        let state = ai::installed_state().expect("ai state");
        let state = state.borrow();

        let mut ex = PipeExecutor::new(&state);
        assert!(ex.has_pipe("UL_Zombie_HuntPlayer_Fast"), "pipe missing");

        // A representative zombie pipe: run -> locate -> stick.
        assert!(ex.set_pipe(42, "UL_Zombie_HuntPlayer_Fast"));
        let mut env = StubEnv {
            dt: 0.1,
            pos: [0.0, 0.0, 0.0],
            target: Some([10.0, 0.0, 0.0]),
            path: PathStatus::InProgress,
            locate: LocateResult::Found([10.0, 0.0, 0.0]),
            ..Default::default()
        };
        for _ in 0..30 {
            ex.step(42, &mut env);
        }
        assert!(
            ex.unknown.is_empty(),
            "real zombie pipe hit unknown ops: {:?}",
            ex.unknown
        );

        // Also walk every registered pipe for one frame to surface any other
        // op this module does not model (does not assert, just reports).
        let names: Vec<String> = ex.pipe_names().map(|s| s.to_string()).collect();
        let mut env = StubEnv::default();
        for (i, n) in names.iter().enumerate() {
            ex.set_pipe(1000 + i as EntityId, n);
            ex.step(1000 + i as EntityId, &mut env);
        }
        eprintln!("[ai-runtime] unknown goal ops across all pipes: {:?}", ex.unknown);
    }

    #[test]
    fn report_used_ops_are_implemented() {
        // Ops the shipped pipes actually push (parsed from the pipe files) must
        // all be implemented.
        for name in [
            "signal", "timeout", "locate", "run", "branch", "tacticalpos", "stick",
            "firecmd", "acqtarget", "approach", "lookat", "ul_locatevisible",
            "bodypos", "ul_runforward", "ul_face", "devalue", "backoff",
        ] {
            let k = GoalOpKind::parse(name);
            assert!(k.is_implemented(), "{name} not implemented");
        }
    }
}
