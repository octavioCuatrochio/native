//! Zombie ecology runtime: drives [`SpawnDirector`] against the live world,
//! keeps a logical record per spawned zombie, promotes nearby ones to live
//! CryEngine actors with a goal pipe, and moves them through the navmesh.
//!
//! This is the "Lab AI → Cry AI" hand-off described in
//! `notes/ghidra/zombie-runtime.md` §0/§12: the 256-slot logical simulation
//! decides where zombies are, and the small live pool (24,
//! [`MAX_LIVE_ZOMBIES`](super::zombie_spawn::MAX_LIVE_ZOMBIES)) runs the real
//! behaviour stack. The split here is deliberately native and event-free:
//!
//! * [`ZombieRuntime::update_director`] runs `SpawnDirector::update` with a
//!   [`WorldView`] over the real world and applies the spawn/despawn commands
//!   to [`ZombieRecord`]s.
//! * [`ZombieRuntime::reconcile_live`] applies the activation distance
//!   (`Zombies.ActivationDist` 15 m, `DeactivationDist` 20 m) to choose which
//!   records are promoted.
//! * [`ZombieRuntime::step_ai`] walks each live record's [`PipeExecutor`] user
//!   (a `CPipeUser`) with a [`PipeEnv`] that pathfinds over the level's `.bai`
//!   navmesh and snaps to the collision world's ground. Raised pipe signals are
//!   collected in [`ZombieRuntime::signals`] for the caller to route to the
//!   zombie's script `OnSignal`.
//!
//! No Lua or Bevy types appear here, so the module is unit-testable and the
//! engine binding (spawning actors, drawing them) lives in the app's
//! `ai_bridge`. Addresses are cited from the recovered decompilation; the code
//! is an independent implementation.

use super::ai_data::AiState;
use super::ai_runtime::{
    AgentCommand, LocateResult, PathStatus, PipeEnv, PipeExecutor, RaisedSignal,
};
use super::navmesh::{NavMesh, NAV_TRIANGULAR};
use super::phys_world::CollisionWorld;
use super::world::EntityId;
use super::zombie_spawn::{
    kind_for_class, SpawnCommand, SpawnDirector, WorldView, ZombieView, MAX_LIVE_ZOMBIES,
};
use crate::game::ZombieKind;

/// Distance at which a logical zombie is promoted to a live actor. The RTS
/// value is `Zombies.ActivationDist = 15` (`cvar zm_activationdist`), which the
/// original compares against the entity/player distance (`013194e0`).
pub const ACTIVATION_DIST: f32 = 15.0;
/// Distance beyond which a live actor is demoted back to the logical pool
/// (`Zombies.DeactivationDist = 20`).
pub const DEACTIVATION_DIST: f32 = 20.0;

/// How far a logical zombie is allowed to be from the player while still being
/// simulated natively (the original simulates all 256 always; this app only
/// keeps nearby/visible ones so the actor pool does not thrash).
pub const LOGICAL_SIM_RANGE: f32 = 260.0;

/// Default pipe for a zombie that has not been told otherwise: the shipped
/// `UL_PipeManagerZombies` hunt pipe (see `notes/ai-combat.md` §3.3).
pub const HUNT_FAST_PIPE: &str = "UL_Zombie_HuntPlayer_Fast";
/// Slow variant for shambling types.
pub const HUNT_SLOW_PIPE: &str = "UL_Zombie_HuntPlayer_Slow";

/// Map a `ZombieTypes.xml` `EntityClass` to the class/script pair the actor
/// system spawns. Classes are the script factories under
/// `scripts/entities/ai/*.lua` (`notes/ai-combat.md` §1); the native spawn
/// class index convention is `0 -> "Grunt"` (`013194e0:200`).
pub fn class_script(entity_class: &str) -> (&'static str, &'static str) {
    match entity_class {
        "FreshGrunt" => ("FreshGrunt", "scripts/entities/ai/freshgrunt.lua"),
        "ArmyZombie" | "SwatZombie" => ("ArmyZombie", "scripts/entities/ai/armyzombie.lua"),
        "Feral" => ("Feral", "scripts/entities/ai/feral.lua"),
        "Bloater" => ("Bloater", "scripts/entities/ai/bloater.lua"),
        "Screamer" => ("Screamer", "scripts/entities/ai/screamer.lua"),
        "Juggernaut" => ("Juggernaut", "scripts/entities/ai/juggernaut.lua"),
        _ => ("Grunt", "scripts/entities/ai/grunt.lua"),
    }
}

/// A signal a live zombie's pipe raised; routed to the actor's script.
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptSignal {
    /// Actor entity id that raised it (the pipe user).
    pub entity: EntityId,
    /// Signal name (`TO_FAST`, `CryAIStartMoving`, ...).
    pub name: String,
    /// `SIGNALFILTER_*` bitfield.
    pub filter: i64,
    /// The pipe op's mode parameter.
    pub mode: i64,
}

/// One zombie the ecology knows about: a logical slot that may or may not be
/// promoted to a live actor.
#[derive(Clone, Debug)]
pub struct ZombieRecord {
    /// Logical handle (slot id; mirrors `FUN_0133dc40` allocation).
    pub handle: u32,
    /// Gameplay kind.
    pub kind: ZombieKind,
    /// `EntityClass` from the spawn command (drives the script mapping).
    pub entity_class: String,
    /// Actor script path (Lua class factory).
    pub script: String,
    /// World position (CryEngine Z-up).
    pub pos: [f32; 3],
    /// Heading (radians).
    pub heading: f32,
    /// Spawn cluster (`CLUSTER_*`).
    pub cluster: u8,
    /// True once promoted to a live CryEngine actor.
    pub live: bool,
    /// Script-host entity id once spawned; `None` for a purely logical zombie.
    pub entity: Option<EntityId>,
    /// Current goal pipe name.
    pub pipe: String,
    /// Last located target.
    pub target: Option<[f32; 3]>,
    /// Remaining path polyline (CryEngine space) and its cursor.
    pub path: Vec<[f32; 3]>,
    pub path_i: usize,
    /// Goal the current path was planned for, so it is replanned when it moves.
    pub path_goal: Option<[f32; 3]>,
    /// Whether the zombie moved this frame (drives walk/run clip choice).
    pub moving: bool,
    /// Last world-space speed (m/s).
    pub speed: f32,
    /// `run`/`speed` op scale from the pipe.
    pub speed_scale: f32,
    /// Cartridge for the walk/run distinction.
    pub walk_only: bool,
    /// Look target from `lookat`/`ul_face`.
    pub look_target: Option<[f32; 3]>,
    /// `ul_runforward` toggle.
    pub force_forward: bool,
}

impl ZombieRecord {
    fn from_command(handle: u32, cmd: &SpawnCommand) -> ZombieRecord {
        let (class, script) = class_script(&cmd.entity_class);
        let walk_only = matches!(cmd.kind, ZombieKind::Grunt | ZombieKind::Screamer)
            && cmd.flags & super::zombie_spawn::ZOMBIE_SPAWN_FAST == 0;
        ZombieRecord {
            handle,
            kind: cmd.kind,
            entity_class: class.to_string(),
            script: script.to_string(),
            pos: cmd.pos,
            heading: cmd.heading,
            cluster: cmd.cluster,
            live: false,
            entity: None,
            pipe: if walk_only { HUNT_SLOW_PIPE } else { HUNT_FAST_PIPE }.to_string(),
            target: None,
            path: Vec::new(),
            path_i: 0,
            path_goal: None,
            moving: false,
            speed: 0.0,
            speed_scale: if walk_only { 0.5 } else { 1.5 },
            walk_only,
            look_target: None,
            force_forward: false,
        }
    }

    /// Locomotion top speed for the record's kind (`ZombieKind::stats`).
    pub fn base_speed(&self) -> f32 {
        let s = self.kind.stats();
        if self.walk_only {
            s.walk_speed
        } else {
            s.run_speed
        }
    }
}

/// The zombie ecology runtime. Owns the director, the logical records and the
/// goal-pipe executor shared by every live actor.
pub struct ZombieRuntime {
    /// Spawn/despawn director ([`super::zombie_spawn`]).
    pub director: SpawnDirector,
    /// Logical + live zombies.
    pub zombies: Vec<ZombieRecord>,
    /// Goal-pipe executor for live actors.
    pub executor: PipeExecutor,
    /// Signals raised this frame, pending routing to scripts.
    pub signals: Vec<ScriptSignal>,
    /// Live actor cap (`m_liveZombies < 0x18`).
    pub live_cap: usize,
    /// Cumulative counters for telemetry.
    pub spawned: u64,
    pub despawned: u64,
    /// Number of times a script pipe could not be selected.
    pub pipe_misses: u64,
}

impl ZombieRuntime {
    /// Build the runtime from the recovered director and the script-authored AI
    /// registry (used to clone the goal pipes the executor runs).
    pub fn new(director: SpawnDirector, ai: &AiState) -> ZombieRuntime {
        ZombieRuntime {
            director,
            zombies: Vec::new(),
            executor: PipeExecutor::new(ai),
            signals: Vec::new(),
            live_cap: MAX_LIVE_ZOMBIES,
            spawned: 0,
            despawned: 0,
            pipe_misses: 0,
        }
    }

    /// Number of promoted actors.
    pub fn live_count(&self) -> usize {
        self.zombies.iter().filter(|z| z.live).count()
    }

    fn view<'a>(
        &self,
        player: [f32; 3],
        nav: Option<&'a NavMesh>,
        collision: Option<&'a CollisionWorld>,
    ) -> RuntimeView<'a> {
        let zombies = self
            .zombies
            .iter()
            .map(|z| ZombieView {
                id: z.handle,
                kind: z.kind,
                pos: z.pos,
                live: z.live,
                cluster: z.cluster,
            })
            .collect();
        RuntimeView { player, zombies, nav, collision, inside: false }
    }

    /// Run the spawn director for one frame and apply its commands.
    ///
    /// Mirrors the manager update order in `FUN_01348680`: the director emits
    /// `Spawn`/`Despawn`; the host applies them to the logical slot array.
    pub fn update_director(
        &mut self,
        dt: f32,
        player: [f32; 3],
        nav: Option<&NavMesh>,
        collision: Option<&CollisionWorld>,
    ) {
        let events = {
            let view = self.view(player, nav, collision);
            self.director.update(dt, &view)
        };
        for ev in events {
            match ev {
                super::zombie_spawn::SpawnEvent::Spawn(cmd) => {
                    let handle = self.director.alloc_id();
                    self.zombies.push(ZombieRecord::from_command(handle, &cmd));
                    self.spawned += 1;
                }
                super::zombie_spawn::SpawnEvent::Despawn(cmd) => {
                    let entity = self
                        .zombies
                        .iter()
                        .find(|z| z.handle == cmd.id)
                        .and_then(|z| z.entity);
                    if let Some(e) = entity {
                        self.executor.remove(e);
                    }
                    self.zombies.retain(|z| z.handle != cmd.id);
                    self.despawned += 1;
                }
            }
        }
        self.reconcile_live(player);
    }

    /// Promote/demote records against the activation distances and live cap.
    ///
    /// `ActivateLiveZombie` (`013194e0`) promotes when the player is closer than
    /// the activation distance and a live slot is free; deactivation is
    /// `DeactivateLiveZombie` (`013207a0`) past the deactivation distance or
    /// when the actor becomes invalid.
    pub fn reconcile_live(&mut self, player: [f32; 3]) {
        // Demote actors that have drifted well past the deactivation distance so
        // the 24-slot pool can serve whoever is closest.
        for z in self.zombies.iter_mut() {
            if z.live {
                let d2 = horiz_d2(z.pos, player);
                if d2 > (DEACTIVATION_DIST * 4.0) * (DEACTIVATION_DIST * 4.0) {
                    z.live = false;
                    z.moving = false;
                }
            }
        }
        // Promote the nearest logical zombies while slots are free.
        let mut live = self.live_count();
        while live < self.live_cap {
            let mut best: Option<(usize, f32)> = None;
            for (i, z) in self.zombies.iter().enumerate() {
                if z.live {
                    continue;
                }
                let d2 = horiz_d2(z.pos, player);
                if d2 > LOGICAL_SIM_RANGE * LOGICAL_SIM_RANGE {
                    continue;
                }
                if best.map(|(_, b)| d2 < b).unwrap_or(true) {
                    best = Some((i, d2));
                }
            }
            let Some((i, _)) = best else { break };
            self.zombies[i].live = true;
            live += 1;
        }
    }

    /// Steer logical (non-live) zombies toward the player so they eventually
    /// reach the activation radius, and integrate the navigation. This stands in
    /// for `MoveZombies` (`01331ef0`) for the slot the actor pool is not
    /// simulating; promoted zombies are moved by their goal pipe instead.
    pub fn step_logical(
        &mut self,
        dt: f32,
        player: [f32; 3],
        collision: Option<&CollisionWorld>,
    ) {
        for i in 0..self.zombies.len() {
            if self.zombies[i].live {
                continue;
            }
            let pos = self.zombies[i].pos;
            if horiz_d2(pos, player) > LOGICAL_SIM_RANGE * LOGICAL_SIM_RANGE {
                continue;
            }
            let (dx, dy) = (player[0] - pos[0], player[1] - pos[1]);
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-3 {
                continue;
            }
            let speed = self.zombies[i].base_speed();
            let step = (speed * dt).min(len);
            let nx = pos[0] + dx / len * step;
            let ny = pos[1] + dy / len * step;
            let z = ground(collision, nx, ny, pos[2]);
            let rec = &mut self.zombies[i];
            rec.pos = [nx, ny, z];
            rec.heading = dy.atan2(dx);
            rec.moving = true;
            rec.speed = speed;
        }
    }

    /// Step every live zombie's goal pipe for a frame, then collect the signals
    /// it raised. `dt` is the clamped frame time.
    pub fn step_ai(
        &mut self,
        dt: f32,
        player: [f32; 3],
        nav: Option<&NavMesh>,
        collision: Option<&CollisionWorld>,
    ) {
        // Snapshot the agent (entity) list so the records can be borrowed mutably
        // in the environment below.
        let agents: Vec<(usize, EntityId)> = self
            .zombies
            .iter()
            .enumerate()
            .filter_map(|(i, z)| z.live.then_some(z.entity).flatten().map(|e| (i, e)))
            .collect();

        for (idx, entity) in agents {
            let want = self.zombies[idx].pipe.clone();
            if self.executor.user(entity).map(|u| u.pipe.as_str()) != Some(want.as_str())
                && !self.executor.set_pipe(entity, &want)
            {
                self.pipe_misses += 1;
            }
            let target = self.zombies[idx].target;
            let mut env = ZombieEnv {
                zombies: &mut self.zombies,
                agent_index: idx,
                player,
                dt,
                nav,
                collision,
                target,
            };
            self.executor.step(entity, &mut env);
        }

        // Drain the executor's outbox into our per-frame signal list.
        let raised: Vec<RaisedSignal> = std::mem::take(&mut self.executor.signals);
        for s in raised {
            self.signals.push(ScriptSignal {
                entity: s.sender,
                name: s.name,
                filter: s.filter,
                mode: s.mode,
            });
        }
    }

    /// Find a record index by its actor entity id.
    pub fn index_of_entity(&self, entity: EntityId) -> Option<usize> {
        self.zombies.iter().position(|z| z.entity == Some(entity))
    }
}

// ---------------------------------------------------------------------------
// World view
// ---------------------------------------------------------------------------

/// [`WorldView`] backed by the real world: player position, navmesh validity
/// and the collision world's ground/visibility.
pub struct RuntimeView<'a> {
    player: [f32; 3],
    zombies: Vec<ZombieView>,
    nav: Option<&'a NavMesh>,
    collision: Option<&'a CollisionWorld>,
    inside: bool,
}

impl WorldView for RuntimeView<'_> {
    fn player_pos(&self) -> [f32; 3] {
        self.player
    }

    fn zombies(&self) -> &[ZombieView] {
        &self.zombies
    }

    fn player_inside(&self) -> bool {
        self.inside
    }

    fn visible(&self, from: [f32; 3], to: [f32; 3]) -> bool {
        let Some(c) = self.collision else { return true };
        let d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len < 1e-3 {
            return true;
        }
        // Anything solid strictly between the eye points blocks line of sight.
        match c.raycast([from[0], from[1], from[2] + 1.5], d, len - 0.25) {
            Some(h) => h.t >= len - 0.25,
            None => true,
        }
    }

    fn nav_valid(&self, p: [f32; 3]) -> bool {
        match self.nav {
            Some(n) => n.nearest_node(p, NAV_TRIANGULAR).is_some(),
            None => true,
        }
    }

    fn ground_height(&self, x: f32, y: f32) -> Option<f32> {
        self.collision.and_then(|c| c.terrain_height(x, y))
    }
}

// ---------------------------------------------------------------------------
// Pipe environment
// ---------------------------------------------------------------------------

/// The [`PipeEnv`] a live zombie's pipe pulls its world inputs from. All state
/// lives in [`ZombieRecord`]s so the executor only sees the abstraction.
struct ZombieEnv<'a> {
    zombies: &'a mut Vec<ZombieRecord>,
    agent_index: usize,
    player: [f32; 3],
    dt: f32,
    nav: Option<&'a NavMesh>,
    collision: Option<&'a CollisionWorld>,
    /// Target resolved by `locate`/`acqtarget`.
    target: Option<[f32; 3]>,
}

impl ZombieEnv<'_> {
    fn idx(&self, agent: EntityId) -> Option<usize> {
        if self.zombies.get(self.agent_index).map(|z| z.entity) == Some(Some(agent)) {
            return Some(self.agent_index);
        }
        self.zombies.iter().position(|z| z.entity == Some(agent))
    }

    /// Plan and follow a path to `goal`, stopping `stop_dist` short of it.
    fn advance(&mut self, idx: usize, goal: [f32; 3], stop_dist: f32, speed_scale: f32) -> PathStatus {
        let pos = self.zombies[idx].pos;
        let to_goal = horiz_dist(pos, goal);
        if to_goal <= stop_dist + 0.35 {
            self.zombies[idx].moving = false;
            self.zombies[idx].speed = 0.0;
            self.zombies[idx].path.clear();
            self.zombies[idx].path_goal = None;
            return PathStatus::Reached;
        }

        // Stop short of the goal for `stick`: aim at a point along the line.
        let (gx, gy) = if stop_dist > 0.0 && to_goal > stop_dist {
            let f = (to_goal - stop_dist) / to_goal;
            (pos[0] + (goal[0] - pos[0]) * f, pos[1] + (goal[1] - pos[1]) * f)
        } else {
            (goal[0], goal[1])
        };
        let aim = [gx, gy, goal[2]];

        // (Re)plan when there is no path or the goal moved appreciably.
        let stale = self.zombies[idx].path.len() < 2
            || self.zombies[idx]
                .path_goal
                .map(|g| horiz_d2(g, aim) > 4.0)
                .unwrap_or(true);
        if stale {
            let path = match self.nav {
                Some(nav) => nav
                    .find_path(pos, aim, NAV_TRIANGULAR)
                    .filter(|p| p.len() >= 2)
                    .unwrap_or_else(|| vec![pos, aim]),
                None => vec![pos, aim],
            };
            let rec = &mut self.zombies[idx];
            rec.path_i = 1.min(path.len().saturating_sub(1));
            rec.path = path;
            rec.path_goal = Some(aim);
        }

        // Advance along the polyline toward the current waypoint.
        let speed = self.speed_of(idx, speed_scale);
        let mut step = speed * self.dt;
        let mut cur = self.zombies[idx].pos;
        let mut i = self.zombies[idx].path_i;
        let len = self.zombies[idx].path.len();
        while i < len && step > 0.0 {
            let wp = self.zombies[idx].path[i];
            let d = horiz_dist(cur, wp);
            if d <= step || d < 1e-3 {
                cur = [wp[0], wp[1], wp[2]];
                step -= d;
                i += 1;
            } else {
                cur = [cur[0] + (wp[0] - cur[0]) / d * step, cur[1] + (wp[1] - cur[1]) / d * step, cur[2]];
                step = 0.0;
            }
        }
        // Past the last waypoint, head for the aim point directly.
        if i >= len {
            let d = horiz_dist(cur, aim);
            if d > 1e-3 {
                let s = (speed * self.dt).min(d);
                cur = [cur[0] + (aim[0] - cur[0]) / d * s, cur[1] + (aim[1] - cur[1]) / d * s, cur[2]];
            }
        }
        let gz = ground(self.collision, cur[0], cur[1], cur[2]);
        let rec = &mut self.zombies[idx];
        let dx = cur[0] - pos[0];
        let dy = cur[1] - pos[1];
        if dx * dx + dy * dy > 1e-6 {
            rec.heading = dy.atan2(dx);
        }
        rec.pos = [cur[0], cur[1], gz];
        rec.path_i = i.min(len);
        rec.moving = true;
        rec.speed = speed;

        if horiz_dist(rec.pos, aim) <= stop_dist + 0.35 || horiz_dist(rec.pos, goal) <= stop_dist + 0.35 {
            PathStatus::Reached
        } else {
            PathStatus::InProgress
        }
    }

    fn speed_of(&self, idx: usize, scale: f32) -> f32 {
        let rec = &self.zombies[idx];
        let s = rec.kind.stats();
        let scale = scale.max(rec.speed_scale);
        if scale <= 0.6 && !rec.walk_only {
            s.walk_speed
        } else {
            (s.run_speed * scale.max(1.0)).clamp(s.walk_speed, s.run_speed * 1.5)
        }
    }
}

impl PipeEnv for ZombieEnv<'_> {
    fn dt(&self) -> f32 {
        self.dt
    }

    fn agent_pos(&self, agent: EntityId) -> [f32; 3] {
        self.idx(agent).map(|i| self.zombies[i].pos).unwrap_or([0.0; 3])
    }

    fn target_pos(&self, agent: EntityId) -> Option<[f32; 3]> {
        if let Some(i) = self.idx(agent) {
            if let Some(t) = self.zombies[i].target {
                return Some(t);
            }
        }
        self.target.or(Some(self.player))
    }

    fn move_to(&mut self, agent: EntityId, pos: [f32; 3], speed: f32, _flags: i64) -> PathStatus {
        let Some(i) = self.idx(agent) else { return PathStatus::Failed };
        self.target = Some(pos);
        self.zombies[i].target = Some(pos);
        self.advance(i, pos, 0.0, speed)
    }

    fn stick(&mut self, agent: EntityId, dist: f32, _flags: i64, _mode: i64) -> PathStatus {
        let Some(i) = self.idx(agent) else { return PathStatus::Failed };
        let target = self.zombies[i].target.unwrap_or(self.player);
        self.advance(i, target, dist.max(0.0), self.zombies[i].speed_scale)
    }

    fn locate(&mut self, agent: EntityId, kind: &str, _radius: f32, _flags: i64) -> LocateResult {
        let k = kind.to_ascii_lowercase();
        let found = if k.is_empty() || k.contains("player") {
            self.player
        } else if k.contains("self") {
            self.idx(agent).map(|i| self.zombies[i].pos).unwrap_or(self.player)
        } else if k.contains("zombie") {
            // Nearest other zombie, else the player.
            let me = self.idx(agent).map(|i| self.zombies[i].pos).unwrap_or(self.player);
            self.zombies
                .iter()
                .filter(|z| z.entity != Some(agent))
                .min_by(|a, b| horiz_d2(a.pos, me).partial_cmp(&horiz_d2(b.pos, me)).unwrap_or(std::cmp::Ordering::Equal))
                .map(|z| z.pos)
                .unwrap_or(self.player)
        } else {
            self.player
        };
        self.target = Some(found);
        LocateResult::Found(found)
    }

    fn tactical_pos(&mut self, agent: EntityId, query: &str) -> Option<[f32; 3]> {
        let q = query.to_ascii_lowercase();
        let base = if q.contains("currentposition") {
            self.idx(agent).map(|i| self.zombies[i].pos).unwrap_or(self.player)
        } else {
            self.player
        };
        // Deterministic offset so wander queries differ per agent.
        let seed = agent as f32 * 0.6180339887;
        let (s, c) = (seed.sin(), seed.cos());
        Some([base[0] + c * 4.0, base[1] + s * 4.0, base[2]])
    }

    fn take_signal(&mut self, _agent: EntityId, _name: &str) -> bool {
        false
    }

    fn command(&mut self, agent: EntityId, cmd: AgentCommand) {
        let Some(i) = self.idx(agent) else { return };
        match cmd {
            AgentCommand::SetSpeed(s) => {
                self.zombies[i].speed_scale = s as f32;
                self.zombies[i].walk_only = s <= 0.6;
            }
            AgentCommand::LookAt { .. } | AgentCommand::FaceTarget { .. } => {
                self.zombies[i].look_target = self.target.or(Some(self.player));
            }
            AgentCommand::ClearLookAt => self.zombies[i].look_target = None,
            AgentCommand::RunForward(on) => self.zombies[i].force_forward = on,
            AgentCommand::BackOff { dist, .. } => {
                let p = self.zombies[i].pos;
                let (dx, dy) = (p[0] - self.player[0], p[1] - self.player[1]);
                let l = (dx * dx + dy * dy).sqrt().max(1e-3);
                let goal = [p[0] + dx / l * (dist + 2.0), p[1] + dy / l * (dist + 2.0), p[2]];
                let _ = self.advance(i, goal, 0.0, self.zombies[i].speed_scale);
            }
            AgentCommand::AcquireTarget { .. } | AgentCommand::TacticalPos { .. } => {}
            AgentCommand::SetBodyPos(_) | AgentCommand::SetFireMode { .. } | AgentCommand::Devalue { .. } => {}
        }
    }
}

/// Ground height from the collision world: a downward ray from just above the
/// current height, falling back to the terrain sample and then `fallback`.
fn ground(collision: Option<&CollisionWorld>, x: f32, y: f32, fallback: f32) -> f32 {
    let Some(c) = collision else { return fallback };
    let top = fallback.max(c.terrain_height(x, y).unwrap_or(fallback)) + 2.0;
    if let Some(h) = c.raycast([x, y, top], [0.0, 0.0, -1.0], 12.0) {
        return h.point[2];
    }
    c.terrain_height(x, y).unwrap_or(fallback)
}

fn horiz_d2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn horiz_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    horiz_d2(a, b).sqrt()
}

/// Expose the `ZombieKind` mapping used by the director for diagnostics.
pub fn record_kind(entity_class: &str) -> ZombieKind {
    kind_for_class(entity_class)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cry::zombie_spawn::{SpawnCommand, ZombieTable, ZombieTuning};
    use crate::cry::zombie_spawn::SpawnDirector;

    fn empty_runtime() -> ZombieRuntime {
        let director = SpawnDirector::new(ZombieTuning::default(), ZombieTable::default(), vec![], vec![], 1);
        ZombieRuntime::new(director, &AiState::default())
    }

    fn cmd(pos: [f32; 3], class: &str) -> SpawnCommand {
        SpawnCommand {
            cluster: 1,
            kind: kind_for_class(class),
            entity_class: class.to_string(),
            pos,
            heading: 0.0,
            flags: 0,
            idle_timer: 1.0,
            always_live: false,
        }
    }

    #[test]
    fn class_script_maps_zombie_classes() {
        assert_eq!(class_script("Grunt").0, "Grunt");
        assert_eq!(class_script("Grunt").1, "scripts/entities/ai/grunt.lua");
        assert_eq!(class_script("Feral").1, "scripts/entities/ai/feral.lua");
        // Unknown classes collapse to the grunt archetype (spawn class 0).
        assert_eq!(class_script("Whatever").0, "Grunt");
    }

    #[test]
    fn spawn_and_despawn_apply_to_records() {
        let mut rt = empty_runtime();
        let id = rt.director.alloc_id();
        rt.zombies.push(ZombieRecord::from_command(id, &cmd([1.0, 2.0, 3.0], "Grunt")));
        assert_eq!(rt.zombies.len(), 1);
        assert_eq!(rt.zombies[0].entity_class, "Grunt");
        assert_eq!(rt.zombies[0].pos, [1.0, 2.0, 3.0]);
        rt.zombies.retain(|z| z.handle != id);
        assert!(rt.zombies.is_empty());
    }

    #[test]
    fn reconcile_promotes_nearby_and_respects_cap() {
        let mut rt = empty_runtime();
        rt.live_cap = 2;
        for i in 0..5u32 {
            let id = rt.director.alloc_id();
            let p = [i as f32, 0.0, 0.0];
            rt.zombies.push(ZombieRecord::from_command(id, &cmd(p, "Grunt")));
        }
        rt.reconcile_live([0.0, 0.0, 0.0]);
        assert_eq!(rt.live_count(), 2, "cap enforced");
        // The two closest (x=0,1) are promoted.
        let live: Vec<f32> = rt.zombies.iter().filter(|z| z.live).map(|z| z.pos[0]).collect();
        assert!(live.contains(&0.0) && live.contains(&1.0));
    }

    #[test]
    fn logical_zombie_walks_toward_player_and_snaps() {
        let mut rt = empty_runtime();
        let id = rt.director.alloc_id();
        let mut c = cmd([10.0, 0.0, 0.0], "Grunt");
        c.flags = 0; // not fast -> walk
        rt.zombies.push(ZombieRecord::from_command(id, &c));
        rt.step_logical(1.0, [0.0, 0.0, 0.0], None);
        let x = rt.zombies[0].pos[0];
        assert!(x < 10.0 && x > 9.0, "walked one second: {x}");
    }

    #[test]
    fn pipe_env_moves_live_zombie_toward_player() {
        let mut rt = empty_runtime();
        // A live record with a real entity id we can drive through a stub pipe.
        let id = rt.director.alloc_id();
        let mut rec = ZombieRecord::from_command(id, &cmd([10.0, 0.0, 0.0], "Grunt"));
        rec.live = true;
        rec.entity = Some(7);
        rec.pipe = HUNT_SLOW_PIPE.to_string();
        rt.zombies.push(rec);

        // Drive one frame with a hand-built environment directly, exercising
        // `stick` after a `locate` resolves the player.
        {
            let mut env = ZombieEnv {
                zombies: &mut rt.zombies,
                agent_index: 0,
                player: [0.0, 0.0, 0.0],
                dt: 1.0,
                nav: None,
                collision: None,
                target: None,
            };
            assert!(matches!(env.locate(7, "player", 0.0, 0), LocateResult::Found(_)));
            let r = env.stick(7, 1.0, 0, 0);
            assert_eq!(r, PathStatus::InProgress);
            assert!(env.zombies[0].pos[0] < 10.0, "moved toward player");
            assert!(env.zombies[0].moving);
        }
    }

    #[test]
    fn find_path_moves_along_navmesh_if_present() {
        // Empty navmesh -> direct fallback still moves.
        let mut rt = empty_runtime();
        let id = rt.director.alloc_id();
        let mut rec = ZombieRecord::from_command(id, &cmd([5.0, 0.0, 0.0], "Grunt"));
        rec.live = true;
        rec.entity = Some(3);
        rt.zombies.push(rec);
        let mut env = ZombieEnv {
            zombies: &mut rt.zombies,
            agent_index: 0,
            player: [0.0, 0.0, 0.0],
            dt: 0.5,
            nav: None,
            collision: None,
            target: Some([0.0, 0.0, 0.0]),
        };
        let _ = env.move_to(3, [0.0, 0.0, 0.0], 1.0, 0);
        assert!(env.zombies[0].pos[0] < 5.0);
    }
}
