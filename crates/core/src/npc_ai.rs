//! Survivor / community NPC AI — a faithful, Wine-free port of the Class3
//! `CContentManager` mission-actor verbs and the `NSCommunity` group/stat layer.
//!
//! This is the *friendly* half of the AI: mission actors, followers/leads,
//! watchtower/barricade defenders, and the home/community stat hooks. The
//! hostile zombie ecology lives in [`crate::ai`].
//!
//! Sources / parity
//! ----------------
//! * `notes/ghidra/class3.md` §ContentManager (102 fns) and §Community (51 fns).
//! * `notes/ghidra/class3_decomp/decomp/`:
//!   * `01350290` `ActionActorAiAnimateAtPoint` → `ModeAnimateAtPoint`
//!   * `01350480` `ActionActorAiBarricade` → `ModeBarricade`
//!   * `013507c0` `ActionActorAiDefend` → `ModeDefendGroup` (+ `ModeRangedAttacker`)
//!   * `013509b0` `ActionActorAiFollowPlayer` → `ModeFollowCharacter`
//!   * `01350a90` `ActionActorAiMoveTo` → `ModeMoveTo`
//!   * `01350e10` `ActionActorAiTeleportTo` (instant placement)
//!   * `01351050` `ActionActorAiWatchtower` → `ModeWatchtower` + `ForceNewWatchPoint`
//!   * `01351650` `ActionActorSetCoreNpc` (promote member to the core follower)
//!   * `01351dc0` `MissionActorAIFollow::Apply`
//!   * `01351ec0` `MissionActorAIMoveTo::Apply`
//!   * `01352220` `MissionActorAITeleportTo::Apply`
//!   * `01352330` `MissionActorAIWatchtower::Apply`
//!   * `01352430` `MissionActorAiAnimateAtPoint::Apply`
//!   * `01352540` `MissionActorAiDefend::Apply`
//!   * `013f1b10` `NSCommunity::CNpcGroup::ModeSet`
//!   * `013f5ee0` `NSCommunity::CNpcGroup::SetCoreNpc`
//!   * `013f6e00` `NSCommunity::CNpcGroup::SetupGroup`
//!   * `013e3600` / `013e3650` `Enclave::MemberInDangerDec/Inc`
//!   * `013e0130` `NSCommunity::AddEnclaveStat`
//!   * `013faae0` `NSCommunity::Enclave::UpdateTrust`
//! * `notes/mission-flow.md` §2.3 (mission actor index) and §2.5 (`LeadNpcs`,
//!   `Tour`, `Siege` objective actors).
//! * `notes/ghidra/rts-sim.md` §5.6 / §6 (community stat hooks, morale clamp
//!   `0x0d000013` ∈ [0,100], `Family.Influence`, `Enclave.*`).
//! * `notes/inventory-community.md` §4.5 / §5.4 (survivors, community model).
//!
//! Design
//! ------
//! The native code does not run one monolithic `update`: each mission action
//! pushes a *mode string* onto the actor's AI (`CContentManager::SendAISignal`
//! @ `0135bf40` → `npcAiActor->ModeSet`). [`NpcOrder`] is that recovered verb
//! set, [`update`] is the per-frame steering/integration those modes imply, and
//! [`NpcAgent::order_mode_name`] maps back to the exact strings the decomp
//! pushes (`"ModeFollowCharacter"`, `"ModeMoveTo"`, …) for parity tooling.
//!
//! Coordinates are the engine's Z-up world space: `x`/`y` are the ground plane
//! and `z` is height, matching [`crate::nav`] and [`crate::physics`].

use std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// Movement tuning
// ---------------------------------------------------------------------------

/// Survivor run speed (m/s). The shipped `player.lua` `MovementTuning` walk/run
/// triplet is used for the player controller; NPCs share the same gait bands,
/// so we keep a fixed nominal here rather than threading the player tuning in.
pub const RUN_SPEED: f32 = 3.8;
/// Walk speed used when wandering/idle (m/s).
pub const WALK_SPEED: f32 = 1.5;
/// Flee speed (m/s) — a survivor sprinting away.
pub const FLEE_SPEED: f32 = 4.5;
/// Distance within which a steering target is considered reached (m).
pub const ARRIVE_EPS: f32 = 0.05;
/// Radius a `Defend` actor will not leave (the native `0x40a00000` = 5.0 m
/// defend-group radius; we hold a tighter 2 m so the actor visibly returns).
pub const DEFEND_RADIUS: f32 = 2.0;
/// Distance at which an actor turns to face a threat (m).
pub const FACE_RANGE: f32 = 8.0;
/// Melee/ranged engagement distance (m).
pub const ATTACK_RANGE: f32 = 1.5;
/// Seconds between attack actions.
pub const ATTACK_COOLDOWN: f32 = 1.0;
/// Idle wander leg length (m).
pub const WANDER_DIST: f32 = 10.0;
/// Seconds before an idle actor picks a new wander leg.
pub const WANDER_PERIOD: f32 = 8.0;
/// Hard clamp on a single simulation step (s), so a stalled frame cannot fling
/// an actor across the map.
pub const MAX_DT: f32 = 0.25;

/// Morale stat `0x0d000013` clamps to `[0, 100]` (`rts-sim.md` §6).
pub const MORALE_MAX: f32 = 100.0;
/// Lower morale bound.
pub const MORALE_MIN: f32 = 0.0;
/// Morale recovered per second with nobody in danger.
pub const MORALE_RECOVER_PER_SEC: f32 = 0.5;
/// Morale lost per second while members are in danger.
pub const MORALE_DANGER_PER_SEC: f32 = 2.0;
/// Morale penalty contributed per member currently in danger.
pub const MORALE_DANGER_PENALTY: f32 = 20.0;
/// `Family.Influence` accrued per member per second (a stat hook; the exact
/// attrition formula lives in the non-decompiled RTS tick — `rts-sim.md` §7).
pub const INFLUENCE_PER_MEMBER_PER_SEC: f32 = 0.02;

// ---------------------------------------------------------------------------
// Orders (the recovered `MissionActorAI*` / `Mode*` verbs)
// ---------------------------------------------------------------------------

/// The real `MissionActorAI*` verbs, one per `CContentManager::ActionActorAi*`
/// entry point in `class3.md`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NpcOrder {
    /// `TO_IDLE` — no AI mode; stand and (for non-followers) wander.
    Idle,
    /// `ModeFollowCharacter` — trail `leader`, stopping `distance` metres back.
    Follow { leader: u32, distance: f32 },
    /// `LeadNpcs` escort — this actor leads the party to `to`.
    Lead { to: [f32; 3] },
    /// `ModeMoveTo` — walk to an explicit point.
    MoveTo { to: [f32; 3] },
    /// `ModeDefendGroup` — hold position around `point`, engage threats.
    Defend { point: [f32; 3] },
    /// Break contact and run directly away from `from`.
    Flee { from: [f32; 3] },
    /// `ModeBarricade` — man a barricade (stationary, engage threats).
    Barricade,
    /// `ModeWatchtower` + `ForceNewWatchPoint` — hold a watch point.
    Watchtower,
    /// `ModeAnimateAtPoint` — play a scripted animation in place.
    AnimateAtPoint,
    /// Instant relocation (`ActionActorAiTeleportTo`).
    TeleportTo { to: [f32; 3] },
}

impl NpcOrder {
    /// The exact mode string the native `ModeSet` pushes for this order
    /// (`PTR_s_ModeFollowCharacter_0190d9b8`-relative table and the
    /// `ActionActorAi*` decomp). `None` for `Idle`/`TeleportTo`, which are not
    /// persistent AI modes.
    pub fn order_mode_name(self) -> Option<&'static str> {
        match self {
            NpcOrder::Idle => Some("TO_IDLE"),
            NpcOrder::Follow { .. } => Some("ModeFollowCharacter"),
            NpcOrder::Lead { .. } => Some("ModeFollowCharacter"),
            NpcOrder::MoveTo { .. } => Some("ModeMoveTo"),
            NpcOrder::Defend { .. } => Some("ModeDefendGroup"),
            NpcOrder::Flee { .. } => Some("ModeMoveTo"),
            NpcOrder::Barricade => Some("ModeBarricade"),
            NpcOrder::Watchtower => Some("ModeWatchtower"),
            NpcOrder::AnimateAtPoint => Some("ModeAnimateAtPoint"),
            NpcOrder::TeleportTo { .. } => None,
        }
    }

    /// Whether this order pins the actor in place.
    pub fn is_stationary(self) -> bool {
        matches!(
            self,
            NpcOrder::Idle
                | NpcOrder::Barricade
                | NpcOrder::Watchtower
                | NpcOrder::AnimateAtPoint
                | NpcOrder::TeleportTo { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

/// Who the actor is, which gates auto-engagement and speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpcRole {
    /// Ordinary home survivor.
    Survivor,
    /// Party leader (`LeadNpcs` / `Tour` `LeaderActorIndex`).
    Leader,
    /// Core NPC promoted by `ActionActorSetCoreNpc` (`01351650`).
    Core,
    /// Non-combatant; flees rather than fights.
    Civilian,
    /// Rival/enclave hostile (rogue `NPC_GROUP_FLAG_ROGUE`).
    Hostile,
}

impl NpcRole {
    /// Whether this role will engage a threat with its own weapon.
    pub fn engages(self) -> bool {
        matches!(self, NpcRole::Leader | NpcRole::Core | NpcRole::Hostile)
    }

    /// Ground-speed multiplier for the role.
    pub fn speed_scale(self) -> f32 {
        match self {
            NpcRole::Civilian => 0.85,
            _ => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// Per-actor timers/state carried between frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NpcCooldowns {
    /// Seconds until the next attack action.
    pub attack: f32,
    /// Seconds until the current order may be re-issued.
    pub order: f32,
    /// Seconds until the idle actor picks a new wander leg.
    pub wander: f32,
    /// Monotonic counter seeding deterministic wander directions.
    pub wander_seq: u32,
    /// Current idle wander destination.
    pub wander_target: [f32; 3],
}

impl Default for NpcCooldowns {
    fn default() -> Self {
        NpcCooldowns {
            attack: 0.0,
            order: 0.0,
            wander: 0.0,
            wander_seq: 0,
            wander_target: [0.0; 3],
        }
    }
}

/// A single survivor / mission actor.
#[derive(Debug, Clone, PartialEq)]
pub struct NpcAgent {
    pub id: u32,
    /// Z-up world position; `x`/`y` ground plane, `z` height.
    pub pos: [f32; 3],
    /// Facing yaw (radians, `atan2(y, x)`).
    pub yaw: f32,
    pub order: NpcOrder,
    pub alive: bool,
    pub role: NpcRole,
    pub cooldowns: NpcCooldowns,
}

impl NpcAgent {
    /// Create a living actor in `Idle`.
    pub fn new(id: u32, pos: [f32; 3], role: NpcRole) -> Self {
        NpcAgent {
            id,
            pos,
            yaw: 0.0,
            order: NpcOrder::Idle,
            alive: true,
            role,
            cooldowns: NpcCooldowns::default(),
        }
    }

    /// Issue a new order, resetting the order timer (mirrors `CNpcGroup::ModeSet`
    /// @ `013f1b10`, which re-pushes the mode string to every member).
    pub fn set_order(&mut self, order: NpcOrder) {
        self.order = order;
        self.cooldowns.order = ATTACK_COOLDOWN;
    }

    /// Promote/demote this actor to the group's core follower, mirroring
    /// `CNpcGroup::SetCoreNpc` (`013f5ee0`): the core NPC follows the player.
    pub fn set_core_npc(&mut self, leader: u32, distance: f32) {
        self.role = NpcRole::Core;
        self.set_order(NpcOrder::Follow { leader, distance });
    }

    /// Planar distance to another point.
    pub fn dist_to(&self, p: [f32; 3]) -> f32 {
        dist2(self.pos, p)
    }
}

// ---------------------------------------------------------------------------
// Input / output
// ---------------------------------------------------------------------------

/// Per-frame perception handed to [`update`]. The world provides the leader and
/// the nearest threat; the actor owns its own order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NpcInput {
    /// Position of the followed leader, if the leader is resolvable this frame.
    pub leader_pos: Option<[f32; 3]>,
    /// Nearest hostile position, if any.
    pub threat_pos: Option<[f32; 3]>,
    /// Frame delta in seconds.
    pub dt: f32,
}

impl Default for NpcInput {
    fn default() -> Self {
        NpcInput { leader_pos: None, threat_pos: None, dt: 0.0 }
    }
}

/// Steering result consumed by the animation/movement controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NpcOutput {
    /// Unit planar movement direction (`[x, y]`); `[0,0]` when stationary.
    pub move_dir: [f32; 2],
    /// Desired ground speed (m/s); `0` when stationary.
    pub speed: f32,
    /// Unit planar facing direction (`[x, y]`).
    pub face: [f32; 2],
    /// True on the frame an attack action fires.
    pub attack: bool,
}

impl Default for NpcOutput {
    fn default() -> Self {
        NpcOutput { move_dir: [0.0; 2], speed: 0.0, face: [1.0, 0.0], attack: false }
    }
}

// ---------------------------------------------------------------------------
// Math helpers (total, NaN-safe)
// ---------------------------------------------------------------------------

#[inline]
fn finite3(p: [f32; 3]) -> Option<[f32; 3]> {
    if p.iter().all(|v| v.is_finite()) {
        Some(p)
    } else {
        None
    }
}

#[inline]
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    (dx * dx + dy * dy).sqrt()
}

/// Deterministic wander angle in `[0, TAU)` from the actor id and a sequence
/// number, so idle NPCs diverge without an RNG dependency.
#[inline]
fn wander_angle(id: u32, seq: u32) -> f32 {
    let mut h = id.wrapping_mul(0x9e37_79b1) ^ seq.wrapping_mul(0x0000_c24f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0 * TAU
}

/// Advance `agent` toward `target`, moving at most `max_step` metres. Fills the
/// movement half of `out` and updates `yaw`.
fn step_toward(agent: &mut NpcAgent, target: [f32; 3], speed: f32, dt: f32, max_step: f32, out: &mut NpcOutput) {
    let dx = target[0] - agent.pos[0];
    let dy = target[1] - agent.pos[1];
    let d = (dx * dx + dy * dy).sqrt();
    if !(d > 1e-4) || !d.is_finite() || !(max_step > 0.0) || dt <= 0.0 {
        return;
    }
    let dir = [dx / d, dy / d];
    let step = (speed * dt).min(max_step);
    agent.pos[0] += dir[0] * step;
    agent.pos[1] += dir[1] * step;
    agent.yaw = dir[1].atan2(dir[0]);
    out.move_dir = dir;
    out.speed = speed;
}

/// Idle wander: pick a deterministic leg every [`WANDER_PERIOD`] and walk it.
fn wander(agent: &mut NpcAgent, dt: f32, out: &mut NpcOutput) {
    let id = agent.id;
    let pos = agent.pos;
    let cd = &mut agent.cooldowns;
    cd.wander -= dt;
    if !(cd.wander > 0.0) {
        cd.wander_seq = cd.wander_seq.wrapping_add(1);
        cd.wander = WANDER_PERIOD;
        let a = wander_angle(id, cd.wander_seq);
        cd.wander_target = [pos[0] + a.cos() * WANDER_DIST, pos[1] + a.sin() * WANDER_DIST, pos[2]];
    }
    let target = cd.wander_target;
    let d = dist2(pos, target);
    if d > ARRIVE_EPS {
        step_toward(agent, target, WALK_SPEED, dt, d, out);
    }
}

/// Whether `role` may fire an attack under `order`.
fn can_attack(role: NpcRole, order: NpcOrder) -> bool {
    if role.engages() {
        return true;
    }
    match role {
        NpcRole::Survivor => matches!(order, NpcOrder::Defend { .. } | NpcOrder::Watchtower | NpcOrder::Barricade),
        NpcRole::Civilian => false,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// The per-frame update
// ---------------------------------------------------------------------------

/// Advance one NPC by `input.dt`, mutating its position/yaw/cooldowns and
/// returning the steering result. Total: non-finite state or input degrades to
/// a finite, stationary output instead of propagating NaN.
pub fn update(agent: &mut NpcAgent, input: &NpcInput) -> NpcOutput {
    // --- sanitize -----------------------------------------------------------
    let dt = if input.dt.is_finite() { input.dt.clamp(0.0, MAX_DT) } else { 0.0 };
    for c in agent.pos.iter_mut() {
        if !c.is_finite() {
            *c = 0.0;
        }
    }
    if !agent.yaw.is_finite() {
        agent.yaw = 0.0;
    }
    agent.cooldowns.attack = (agent.cooldowns.attack - dt).max(0.0);
    agent.cooldowns.order = (agent.cooldowns.order - dt).max(0.0);

    let threat = input.threat_pos.and_then(finite3);
    let leader = input.leader_pos.and_then(finite3);

    let mut out = NpcOutput::default();
    if !agent.alive {
        out.face = [agent.yaw.cos(), agent.yaw.sin()];
        return out;
    }

    let order = agent.order;
    let mut face_target: Option<[f32; 3]> = None;

    // --- order steering -----------------------------------------------------
    match order {
        NpcOrder::Idle => wander(agent, dt, &mut out),

        NpcOrder::Follow { distance, .. } => {
            // `ModeFollowCharacter`: close on the leader, hold a standoff.
            if let Some(lp) = leader {
                let d = dist2(agent.pos, lp);
                let standoff = if distance.is_finite() { distance.max(0.0) } else { 0.0 };
                let remaining = d - standoff;
                if remaining > ARRIVE_EPS {
                    step_toward(agent, lp, RUN_SPEED * agent.role.speed_scale(), dt, remaining, &mut out);
                }
                face_target = Some(lp);
            }
            // No resolvable leader → hold idle (the required "no leader idles").
        }

        NpcOrder::Lead { to } | NpcOrder::MoveTo { to } => {
            if let Some(t) = finite3(to) {
                let d = dist2(agent.pos, t);
                if d > ARRIVE_EPS {
                    step_toward(agent, t, RUN_SPEED * agent.role.speed_scale(), dt, d, &mut out);
                }
                face_target = Some(t);
            }
        }

        NpcOrder::Defend { point } => {
            if let Some(p) = finite3(point) {
                let d = dist2(agent.pos, p);
                if d > DEFEND_RADIUS {
                    step_toward(agent, p, RUN_SPEED * agent.role.speed_scale(), dt, d - DEFEND_RADIUS, &mut out);
                }
                // Defenders watch the threat, else the defended point.
                face_target = threat.or(Some(p));
            }
        }

        NpcOrder::Flee { from } => {
            if let Some(f) = finite3(from) {
                let dx = agent.pos[0] - f[0];
                let dy = agent.pos[1] - f[1];
                let d = (dx * dx + dy * dy).sqrt();
                let dir = if d > 1e-4 && d.is_finite() { [dx / d, dy / d] } else { [1.0, 0.0] };
                let speed = FLEE_SPEED * agent.role.speed_scale();
                out.move_dir = dir;
                out.speed = speed;
                agent.pos[0] += dir[0] * speed * dt;
                agent.pos[1] += dir[1] * speed * dt;
                agent.yaw = dir[1].atan2(dir[0]);
                face_target = threat.or(Some(f));
            }
        }

        NpcOrder::Barricade | NpcOrder::Watchtower | NpcOrder::AnimateAtPoint => {
            // Stationary modes: hold the point and watch the threat.
            face_target = threat;
        }

        NpcOrder::TeleportTo { to } => {
            // `ActionActorAiTeleportTo` (`01350e10`) places the actor outright.
            if let Some(t) = finite3(to) {
                agent.pos = t;
            }
            face_target = threat;
        }
    }

    // --- threat response (overrides facing, may attack) ---------------------
    if let Some(t) = threat {
        let d = dist2(agent.pos, t);
        if d <= FACE_RANGE {
            face_target = Some(t);
        }
        if d <= ATTACK_RANGE && agent.cooldowns.attack <= 0.0 && can_attack(agent.role, order) {
            out.attack = true;
            agent.cooldowns.attack = ATTACK_COOLDOWN;
        }
    }

    // --- facing -------------------------------------------------------------
    if let Some(ft) = face_target {
        let dx = ft[0] - agent.pos[0];
        let dy = ft[1] - agent.pos[1];
        let l = (dx * dx + dy * dy).sqrt();
        if l > 1e-4 && l.is_finite() {
            out.face = [dx / l, dy / l];
            agent.yaw = out.face[1].atan2(out.face[0]);
        } else {
            out.face = [agent.yaw.cos(), agent.yaw.sin()];
        }
    } else if out.move_dir != [0.0, 0.0] {
        out.face = out.move_dir;
    } else {
        out.face = [agent.yaw.cos(), agent.yaw.sin()];
    }

    out
}

// ---------------------------------------------------------------------------
// Community / enclave stat layer
// ---------------------------------------------------------------------------

/// Pure result of one [`Community::tick`] — the per-step deltas and income the
/// caller may display or commit via [`Community::apply_tick`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommunityTick {
    /// Morale change this step (already time-scaled).
    pub morale_delta: f32,
    /// `Family.Influence` gained this step.
    pub influence_gain: f32,
}

/// The home/community stat layer from `rts-sim.md` §5.6/§6: membership plus the
/// `Confidence.Morale` (id `0x0d000013`, clamped `[0,100]`), `Family.Influence`,
/// and `Enclave.*Trust` hooks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Community {
    /// Member actor ids, in join order.
    pub members: Vec<u32>,
    /// `Confidence.Morale`, always `[0, 100]`.
    pub morale: f32,
    /// `Family.Influence` pool (`>= 0`).
    pub influence: f32,
    /// Average member trust (`Enclave::UpdateTrust`, `013faae0`), `[0, 100]`.
    pub trust: f32,
    /// Member count (`m_count`).
    pub count: u32,
    /// Members currently in danger (`m_numMembersInDanger`, `013e3600/013e3650`).
    pub members_in_danger: u32,
}

impl Community {
    /// An empty community at full morale.
    pub fn new() -> Self {
        Community { morale: MORALE_MAX, ..Default::default() }
    }

    /// Add a member. Returns `false` if the id is already present.
    pub fn add_member(&mut self, id: u32) -> bool {
        if self.members.contains(&id) {
            return false;
        }
        self.members.push(id);
        self.count = self.members.len() as u32;
        true
    }

    /// Remove a member. Returns `false` if the id was not present.
    pub fn remove_member(&mut self, id: u32) -> bool {
        if let Some(i) = self.members.iter().position(|&m| m == id) {
            self.members.swap_remove(i);
            self.count = self.members.len() as u32;
            self.members_in_danger = self.members_in_danger.min(self.count);
            true
        } else {
            false
        }
    }

    /// Whether `id` belongs to this community.
    pub fn contains(&self, id: u32) -> bool {
        self.members.contains(&id)
    }

    /// Set morale, clamped to `[0, 100]`.
    pub fn set_morale(&mut self, v: f32) {
        self.morale = if v.is_finite() { v.clamp(MORALE_MIN, MORALE_MAX) } else { self.morale };
    }

    /// Add (or subtract) morale, clamped to `[0, 100]`.
    pub fn add_morale(&mut self, d: f32) {
        self.set_morale(self.morale + if d.is_finite() { d } else { 0.0 });
    }

    /// Add influence; the pool never goes below zero.
    pub fn add_influence(&mut self, d: f32) {
        if d.is_finite() {
            self.influence = (self.influence + d).max(0.0);
        }
    }

    /// Spend influence, clamped at zero; returns the amount actually spent.
    pub fn spend_influence(&mut self, amount: f32) -> f32 {
        if !amount.is_finite() || amount <= 0.0 {
            return 0.0;
        }
        let spent = amount.min(self.influence);
        self.influence -= spent;
        spent
    }

    /// `MemberInDangerInc` (`013e3650`): assert-bounded by `m_count`.
    pub fn member_in_danger_inc(&mut self) -> bool {
        if self.members_in_danger < self.count {
            self.members_in_danger += 1;
            true
        } else {
            false
        }
    }

    /// `MemberInDangerDec` (`013e3600`): assert-bounded at zero.
    pub fn member_in_danger_dec(&mut self) -> bool {
        if self.members_in_danger > 0 {
            self.members_in_danger -= 1;
            true
        } else {
            false
        }
    }

    /// `Enclave::UpdateTrust` (`013faae0`): store the mean of per-member trust.
    pub fn update_trust(&mut self, per_member: &[f32]) -> f32 {
        let vals: Vec<f32> = per_member.iter().copied().filter(|v| v.is_finite()).collect();
        if vals.is_empty() {
            return self.trust;
        }
        let avg = vals.iter().sum::<f32>() / vals.len() as f32;
        self.trust = avg.clamp(0.0, 100.0);
        self.trust
    }

    /// Stat hook: the per-step morale delta and influence income. Non-mutating,
    /// so it can be sampled by UI without advancing the simulation.
    ///
    /// Morale drifts toward `100 - members_in_danger * MORALE_DANGER_PENALTY`
    /// (`rts-sim.md` §6 pins the `[0,100]` clamp on id `0x0d000013`); influence
    /// accrues `count * INFLUENCE_PER_MEMBER_PER_SEC` scaled by trust.
    pub fn tick(&self, dt: f32) -> CommunityTick {
        let dt = if dt.is_finite() && dt > 0.0 { dt.min(1.0) } else { 0.0 };
        let target = (MORALE_MAX - self.members_in_danger as f32 * MORALE_DANGER_PENALTY)
            .clamp(MORALE_MIN, MORALE_MAX);
        let morale_delta = if self.morale < target {
            (target - self.morale).min(MORALE_RECOVER_PER_SEC * dt)
        } else {
            -((self.morale - target).min(MORALE_DANGER_PER_SEC * dt))
        };
        let trust_scale = 0.5 + self.trust.clamp(0.0, 100.0) / 200.0;
        let influence_gain = self.count as f32 * INFLUENCE_PER_MEMBER_PER_SEC * trust_scale * dt;
        CommunityTick { morale_delta, influence_gain }
    }

    /// Commit [`Community::tick`] to the community stats.
    pub fn apply_tick(&mut self, dt: f32) {
        let t = self.tick(dt);
        self.add_morale(t.morale_delta);
        self.add_influence(t.influence_gain);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn run(agent: &mut NpcAgent, input: &NpcInput, steps: usize) -> NpcOutput {
        let mut out = NpcOutput::default();
        for _ in 0..steps {
            out = update(agent, input);
        }
        out
    }

    fn all_finite(o: &NpcOutput) -> bool {
        o.move_dir.iter().all(|v| v.is_finite())
            && o.face.iter().all(|v| v.is_finite())
            && o.speed.is_finite()
    }

    #[test]
    fn follow_keeps_distance_behind_leader() {
        let mut a = NpcAgent::new(1, [0.0, 0.0, 0.0], NpcRole::Survivor);
        a.order = NpcOrder::Follow { leader: 2, distance: 3.0 };
        let leader = [10.0, 0.0, 0.0];
        let input = NpcInput { leader_pos: Some(leader), threat_pos: None, dt: 0.1 };
        run(&mut a, &input, 300);
        let d = a.dist_to(leader);
        assert!((d - 3.0).abs() < 0.1, "expected standoff ~3.0, got {d}");
    }

    #[test]
    fn move_to_reaches_point() {
        let mut a = NpcAgent::new(1, [0.0, 0.0, 0.0], NpcRole::Survivor);
        let to = [5.0, 5.0, 0.0];
        a.order = NpcOrder::MoveTo { to };
        let input = NpcInput { leader_pos: None, threat_pos: None, dt: 0.1 };
        run(&mut a, &input, 300);
        assert!(a.dist_to(to) < 0.1, "did not reach {to:?}: {:?}", a.pos);
    }

    #[test]
    fn flee_increases_distance_from_threat() {
        let mut a = NpcAgent::new(1, [0.0, 0.0, 0.0], NpcRole::Survivor);
        let threat = [1.0, 0.0, 0.0];
        a.order = NpcOrder::Flee { from: threat };
        let input = NpcInput { leader_pos: None, threat_pos: Some(threat), dt: 0.1 };
        let d0 = a.dist_to(threat);
        run(&mut a, &input, 20);
        let d1 = a.dist_to(threat);
        assert!(d1 > d0 + 1.0, "flee did not open distance: {d0} -> {d1}");
    }

    #[test]
    fn defend_holds_near_point() {
        let mut a = NpcAgent::new(1, [0.0, 0.0, 0.0], NpcRole::Survivor);
        let point = [2.0, 2.0, 0.0];
        a.order = NpcOrder::Defend { point };
        let input = NpcInput { leader_pos: None, threat_pos: None, dt: 0.1 };
        run(&mut a, &input, 300);
        assert!(a.dist_to(point) <= DEFEND_RADIUS + 0.05, "left defend radius: {:?}", a.pos);
    }

    #[test]
    fn teleport_to_sets_position() {
        let mut a = NpcAgent::new(1, [0.0, 0.0, 0.0], NpcRole::Survivor);
        a.order = NpcOrder::TeleportTo { to: [7.0, 8.0, 9.0] };
        let input = NpcInput::default();
        update(&mut a, &input);
        assert_eq!(a.pos, [7.0, 8.0, 9.0]);
    }

    #[test]
    fn follower_without_leader_idles() {
        let mut a = NpcAgent::new(1, [0.0, 0.0, 0.0], NpcRole::Survivor);
        a.order = NpcOrder::Follow { leader: 99, distance: 3.0 };
        let input = NpcInput { leader_pos: None, threat_pos: None, dt: 0.1 };
        let out = run(&mut a, &input, 30);
        assert_eq!(out.move_dir, [0.0, 0.0]);
        assert_eq!(out.speed, 0.0);
        assert_eq!(a.pos, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn nan_total_is_finite_and_stationary() {
        let mut a = NpcAgent::new(1, [f32::NAN, f32::NAN, f32::NAN], NpcRole::Survivor);
        a.order = NpcOrder::Follow { leader: 2, distance: 3.0 };
        let input = NpcInput {
            leader_pos: Some([f32::NAN; 3]),
            threat_pos: Some([f32::INFINITY; 3]),
            dt: f32::NAN,
        };
        let out = update(&mut a, &input);
        assert!(all_finite(&out), "NaN leaked into output: {out:?}");
        assert!(a.pos.iter().all(|v| v.is_finite()));

        // A non-finite destination must also no-op rather than poison the actor.
        let mut b = NpcAgent::new(2, [1.0, 1.0, 0.0], NpcRole::Survivor);
        b.order = NpcOrder::MoveTo { to: [f32::NAN, 0.0, 0.0] };
        let out2 = update(&mut b, &NpcInput { leader_pos: None, threat_pos: None, dt: 0.1 });
        assert!(all_finite(&out2));
        assert_eq!(b.pos, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn hostile_attacks_threat_in_range() {
        let mut a = NpcAgent::new(1, [0.0, 0.0, 0.0], NpcRole::Hostile);
        let input = NpcInput { leader_pos: None, threat_pos: Some([1.0, 0.0, 0.0]), dt: 0.1 };
        let out = update(&mut a, &input);
        assert!(out.attack, "hostile in range should attack");
        // Cooldown gates the next swing.
        let out2 = update(&mut a, &input);
        assert!(!out2.attack, "attack should be on cooldown");
    }

    #[test]
    fn civilian_never_attacks() {
        let mut a = NpcAgent::new(1, [0.0, 0.0, 0.0], NpcRole::Civilian);
        let input = NpcInput { leader_pos: None, threat_pos: Some([0.5, 0.0, 0.0]), dt: 0.1 };
        let out = update(&mut a, &input);
        assert!(!out.attack);
    }

    #[test]
    fn order_mode_names_match_recovered_strings() {
        assert_eq!(NpcOrder::Follow { leader: 0, distance: 1.0 }.order_mode_name(), Some("ModeFollowCharacter"));
        assert_eq!(NpcOrder::MoveTo { to: [0.0; 3] }.order_mode_name(), Some("ModeMoveTo"));
        assert_eq!(NpcOrder::Defend { point: [0.0; 3] }.order_mode_name(), Some("ModeDefendGroup"));
        assert_eq!(NpcOrder::Barricade.order_mode_name(), Some("ModeBarricade"));
        assert_eq!(NpcOrder::Watchtower.order_mode_name(), Some("ModeWatchtower"));
        assert_eq!(NpcOrder::AnimateAtPoint.order_mode_name(), Some("ModeAnimateAtPoint"));
        assert_eq!(NpcOrder::Idle.order_mode_name(), Some("TO_IDLE"));
    }

    #[test]
    fn community_membership() {
        let mut c = Community::new();
        assert_eq!(c.morale, MORALE_MAX);
        assert!(c.add_member(7));
        assert!(!c.add_member(7));
        assert!(c.contains(7));
        assert_eq!(c.count, 1);
        assert!(c.remove_member(7));
        assert!(!c.remove_member(7));
        assert_eq!(c.count, 0);
    }

    #[test]
    fn community_morale_clamps_and_danger_bounds() {
        let mut c = Community::new();
        c.add_member(1);
        c.add_member(2);
        c.add_morale(-1000.0);
        assert_eq!(c.morale, MORALE_MIN);
        c.add_morale(1000.0);
        assert_eq!(c.morale, MORALE_MAX);

        assert!(c.member_in_danger_inc());
        assert!(c.member_in_danger_inc());
        assert!(!c.member_in_danger_inc()); // m_count bound
        assert!(c.member_in_danger_dec());
        assert!(c.member_in_danger_dec());
        assert!(!c.member_in_danger_dec()); // zero bound
    }

    #[test]
    fn community_tick_and_trust_are_finite() {
        let mut c = Community::new();
        c.add_member(1);
        c.add_member(2);
        c.add_member(3);
        assert!((c.update_trust(&[80.0, 60.0, 100.0]) - 80.0).abs() < 1e-3);
        let t = c.tick(0.5);
        assert!(t.morale_delta.is_finite() && t.influence_gain.is_finite());
        assert!(t.influence_gain > 0.0);
        c.apply_tick(0.5);
        assert!(c.influence > 0.0);
        // NaN dt is a no-op.
        let z = c.tick(f32::NAN);
        assert_eq!(z, CommunityTick { morale_delta: 0.0, influence_gain: 0.0 });
    }
}
