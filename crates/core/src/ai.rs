//! Zombie AI: perception, behaviour state machine, horde direction.
//!
//! Faithful, Wine-free reimplementation of the native `NSZombieManager`
//! ecology + the CryAI half that `notes/ghidra/zombie-runtime.md` and
//! `notes/ai-combat.md` describe.
//!
//! Sources / parity:
//!  * `notes/ai-combat.md` §2 (senses, per-type AggroDist/FOV, sound radii),
//!    §3.2–§3.6 (target acquisition, chase, wander, concrete FSM), §5 (spawn
//!    caps, master tuning, freak caps).
//!  * `notes/ghidra/zombie-runtime.md` §5 (DeterminePerception ranges/flags,
//!    LOS), §6 (priority behaviour table), §3 (caps), §4 (hordes/ambush),
//!    §10 (attack initiation), §12 (consolidated state machine).
//!
//! Movement uses the anim-group locomotion speeds re-exported by
//! [`crate::game::ZombieKind::stats`] (walk / run) — the same triplets the
//! native `MoveZombies` integrator consumes.

use crate::game::ZombieKind;
use std::f32::consts::{PI, TAU};

// ---------------------------------------------------------------------------
// Perception tuning (ai-combat.md §2, zombies.xml global block)
// ---------------------------------------------------------------------------

/// CryAI close-range detection radius: inside this the FOV cone is ignored
/// (`DeterminePerception` checks range before the lateral cone).
pub const CLOSE_RANGE: f32 = 2.0;

/// SoD global `DeaggroDist` (zombies.xml): beyond this a chase is abandoned.
pub const DEAGGRO_DIST: f32 = 30.0;
/// `InvestigationTimeout` (zombies.xml): Search lasts this long.
pub const SEARCH_TIMEOUT: f32 = 8.0;
/// `InvestigationDist` (zombies.xml): "reached last known position" radius.
pub const INVESTIGATION_DIST: f32 = 6.0;
/// Lost-target post timer from `DeterminePerception` (0.5 s): how long a
/// chasing zombie tolerates losing LOS before it starts searching.
pub const LOST_GIVEUP: f32 = 0.5;

// ---------------------------------------------------------------------------
// Behaviour tuning (ai-combat.md §3.3–§3.6)
// ---------------------------------------------------------------------------

const ALERT_RATE: f32 = 4.0; // alert builds to 1.0 in ~0.25 s while sensed
const ALERT_DECAY: f32 = 1.0; // alert decays at 1/s when not sensed
const CONFIRM_ALERT: f32 = 1.0; // confidence needed to leave Alert -> Chase
const ALERT_GIVEUP: f32 = 1.5; // unconfirmed contact forgotten after this long
const ATTACK_COOLDOWN: f32 = 1.0; // Timer_CombatCooldown equivalent
const ATTACK_REENGAGE: f32 = 1.5; // drop back to Chase past range*this
const IDLE_PERIOD: f32 = 5.0; // IdlePeriod
const WANDER_DIST: f32 = 10.0; // WanderDist
const WANDER_TIMEOUT: f32 = 10.0;
const EYE_HEIGHT: f32 = 1.5;
const TURN_RATE: f32 = 4.0; // rad/s, approximates maxTurnRadius

// Screamer special (ai-combat.md §3.6 / §5.2, Zombie.CreateEcho).
const SCREAMER_FLEE_DIST: f32 = 6.0;
const SCREAMER_SAFE_DIST: f32 = 14.0;
const SCREAMER_NOISE: f32 = 8.0; // CombatSoundDist
const SCREAM_INTERVAL: f32 = 2.0;

// Horde director (zombies.xml Horde block + §3 caps).
const HORDE_MAX_SIZE: u32 = 8;
const HORDE_MAX_AMBIENT: u32 = 3;
const FREAK_CAP: u32 = 4; // Feral/Juggernaut `Max`
const AMBIENT_RESPAWN_RATE: f32 = 15.0; // AmbientRespawnRate

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The implementable zombie FSM states (ai-combat.md §3.6, §12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiState {
    Idle,
    Wander,
    Alert,
    Chase,
    Attack,
    Search,
    Flee,
    Dead,
}

/// Sense cone / hearing model. `sight_range` is inclusive, `fov_cos` is the
/// cosine of the half-angle of the primary cone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Perception {
    pub sight_range: f32,
    pub fov_cos: f32,
    pub hearing_range: f32,
}

impl Perception {
    /// Build from a sight range, a full cone angle in degrees, and hearing.
    pub fn new(sight_range: f32, fov_degrees: f32, hearing_range: f32) -> Self {
        Perception {
            sight_range,
            fov_cos: (fov_degrees.to_radians() * 0.5).cos(),
            hearing_range,
        }
    }

    /// Per-`ZombieDef` defaults from `ZombieTypes.xml` §2.2:
    /// ambient/horde/brains 60° & AggroDist 15, freaks 50° & AggroDist 6.
    /// The ZombieManager loader has no per-type FOV (ghidra §4.1/§13), so the
    /// 50/60 column is the data layer's best available value. Hearing uses the
    /// combat-sound radius (8 m) with freak adjustments.
    pub fn for_kind(kind: ZombieKind) -> Self {
        let (sight, fov, hearing) = match kind {
            ZombieKind::Grunt => (15.0, 60.0, 8.0),
            ZombieKind::ArmyZombie => (15.0, 60.0, 8.0),
            ZombieKind::Screamer => (15.0, 60.0, 10.0),
            ZombieKind::Bloater => (15.0, 60.0, 6.0),
            ZombieKind::Feral => (6.0, 50.0, 12.0),
            ZombieKind::Juggernaut => (6.0, 50.0, 8.0),
        };
        Perception::new(sight, fov, hearing)
    }

    /// Range gate + FOV cosine test against a caller-supplied eye and forward
    /// vector. Very close targets are always detected (`CLOSE_RANGE`).
    pub fn can_see(&self, eye: [f32; 3], forward: [f32; 3], target: [f32; 3]) -> bool {
        let d = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
        let dist_sq = dot3(d, d);
        if dist_sq > self.sight_range * self.sight_range {
            return false;
        }
        if dist_sq <= CLOSE_RANGE * CLOSE_RANGE {
            return true;
        }
        let fl = len3(forward);
        let dl = dist_sq.sqrt();
        if fl < 1e-6 || dl < 1e-6 {
            return true;
        }
        (dot3(d, forward) / (fl * dl)) >= self.fov_cos
    }

    /// `noise` is a loudness multiplier (`0` = silent). A source is heard when
    /// within `hearing_range * noise`.
    pub fn can_hear(&self, dist: f32, noise: f32) -> bool {
        noise > 0.0 && dist <= self.hearing_range * noise
    }
}

/// Per-agent countdown timers. All values are seconds remaining / elapsed.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AiTimers {
    pub idle: f32,
    pub search: f32,
    pub lost: f32,
    pub scream: f32,
    pub state_time: f32,
}

/// One simulated zombie brain.
#[derive(Debug, Clone)]
pub struct AiAgent {
    pub state: AiState,
    pub kind: ZombieKind,
    pub alert: f32,
    pub pos: [f32; 3],
    pub yaw: f32,
    pub last_known: Option<[f32; 3]>,
    pub attack_cd: f32,
    pub timers: AiTimers,
    /// Per-kind sense cone (defaults from [`Perception::for_kind`]).
    pub perception: Perception,
    /// Current wander waypoint while in [`AiState::Wander`].
    pub wander_target: Option<[f32; 3]>,
}

impl AiAgent {
    pub fn new(kind: ZombieKind, pos: [f32; 3]) -> Self {
        AiAgent {
            state: AiState::Idle,
            kind,
            alert: 0.0,
            pos,
            yaw: 0.0,
            last_known: None,
            attack_cd: 0.0,
            timers: AiTimers { idle: IDLE_PERIOD, ..AiTimers::default() },
            perception: Perception::for_kind(kind),
            wander_target: None,
        }
    }
}

/// External stimulus for one tick.
#[derive(Debug, Clone, Copy)]
pub struct AiInput {
    pub player_pos: [f32; 3],
    pub player_noise: f32,
    pub dt: f32,
    pub random: f32,
}

/// Steering + action result for one tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AiOutput {
    pub move_dir: [f32; 2],
    pub speed: f32,
    pub attack: bool,
    pub state: AiState,
    /// Noise this zombie emits this tick (Screamer scream feeds the director).
    pub noise: f32,
}

// ---------------------------------------------------------------------------
// Deterministic xorshift64* RNG
// ---------------------------------------------------------------------------

/// Tiny deterministic xorshift64* generator used by the horde director and
/// tests. Reproducible across platforms.
#[derive(Debug, Clone)]
pub struct SmallRng {
    state: u64,
}

impl SmallRng {
    pub fn new(seed: u64) -> Self {
        SmallRng {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

// ---------------------------------------------------------------------------
// Horde director
// ---------------------------------------------------------------------------

/// A batch of zombies the director decided to spawn in response to noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HordeSpawn {
    pub ambient: u32,
    pub freaks: u32,
}

impl HordeSpawn {
    pub fn count(&self) -> u32 {
        self.ambient + self.freaks
    }
}

/// Ambient/horde refill + noise-response director. Approximates
/// `AddHorde`/roaming-horde refill (`0131af80`/`0135c080`) and the §4 spawn
/// caps: 8 horde slots, 3 max ambient, 4 freak cap.
#[derive(Debug, Clone)]
pub struct HordeDirector {
    pub ambient_cap: u32,
    pub freak_cap: u32,
    pub max_size: u32,
    pub ambient: u32,
    pub freaks: u32,
    pub total_spawned: u32,
    pub refill_timer: f32,
}

impl HordeDirector {
    pub fn new() -> Self {
        HordeDirector {
            ambient_cap: HORDE_MAX_AMBIENT,
            freak_cap: FREAK_CAP,
            max_size: HORDE_MAX_SIZE,
            ambient: 0,
            freaks: 0,
            total_spawned: 0,
            refill_timer: 0.0,
        }
    }

    /// Respond to a noise event. Spawn count scales with `noise`, is jittered
    /// by `rng`, and can never exceed the ambient/freak caps. Returns `None`
    /// when nothing could be spawned (silence or the horde is full).
    pub fn on_noise(&mut self, noise: f32, rng: &mut SmallRng) -> Option<HordeSpawn> {
        if noise <= 0.0 {
            return None;
        }
        let base = (noise * 0.5).floor() as i64;
        let jitter = rng.next_f32() as i64; // 0..1
        let desired = (base + jitter).clamp(1, self.max_size as i64) as u32;

        let ambient_room = self.ambient_cap.saturating_sub(self.ambient);
        let freak_room = self.freak_cap.saturating_sub(self.freaks);
        let mut remaining = desired;
        let a = remaining.min(ambient_room);
        remaining -= a;
        let f = remaining.min(freak_room);

        if a == 0 && f == 0 {
            return None;
        }
        self.ambient += a;
        self.freaks += f;
        self.total_spawned += a + f;
        Some(HordeSpawn { ambient: a, freaks: f })
    }

    /// Refill the ambient population one zombie at a time every
    /// `AmbientRespawnRate` seconds, clamped to the ambient cap.
    pub fn tick(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        self.refill_timer += dt;
        while self.refill_timer >= AMBIENT_RESPAWN_RATE && self.ambient < self.ambient_cap {
            self.refill_timer -= AMBIENT_RESPAWN_RATE;
            self.ambient += 1;
        }
        if self.ambient >= self.ambient_cap {
            self.refill_timer = self.refill_timer.min(AMBIENT_RESPAWN_RATE);
        }
    }
}

impl Default for HordeDirector {
    fn default() -> Self {
        HordeDirector::new()
    }
}

// ---------------------------------------------------------------------------
// Steering helpers
// ---------------------------------------------------------------------------

fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    len3([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
}

fn yaw_forward(yaw: f32) -> [f32; 3] {
    [yaw.sin(), 0.0, yaw.cos()]
}

/// Normalised horizontal `[x, z]` direction from `from` toward `to`.
fn horiz_dir(from: [f32; 3], to: [f32; 3]) -> [f32; 2] {
    let dx = to[0] - from[0];
    let dz = to[2] - from[2];
    let l = (dx * dx + dz * dz).sqrt();
    if l < 1e-4 {
        [0.0, 0.0]
    } else {
        [dx / l, dz / l]
    }
}

fn horiz_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dz = b[2] - a[2];
    (dx * dx + dz * dz).sqrt()
}

fn set_state(agent: &mut AiAgent, s: AiState) {
    if agent.state != s {
        agent.state = s;
        agent.timers.state_time = 0.0;
    }
}

/// Rotate `yaw` toward the horizontal direction `dir` at `TURN_RATE`.
fn turn_toward(agent: &mut AiAgent, dir: [f32; 2], dt: f32) {
    if dir[0] == 0.0 && dir[1] == 0.0 {
        return;
    }
    let desired = dir[0].atan2(dir[1]);
    let mut diff = desired - agent.yaw;
    while diff > PI {
        diff -= TAU;
    }
    while diff < -PI {
        diff += TAU;
    }
    let max = TURN_RATE * dt;
    agent.yaw = (agent.yaw + diff.clamp(-max, max)).rem_euclid(TAU);
}

// ---------------------------------------------------------------------------
// Per-tick update
// ---------------------------------------------------------------------------

/// Advance one zombie brain by `input.dt`, returning its steering/action.
pub fn update(agent: &mut AiAgent, input: &AiInput) -> AiOutput {
    let dt = input.dt.max(0.0);
    let mut out = AiOutput {
        move_dir: [0.0, 0.0],
        speed: 0.0,
        attack: false,
        state: agent.state,
        noise: 0.0,
    };

    agent.timers.state_time += dt;
    agent.attack_cd = (agent.attack_cd - dt).max(0.0);

    if agent.state == AiState::Dead {
        return out;
    }

    let prev = agent.state;
    let stats = agent.kind.stats();
    let player = input.player_pos;
    let dist = dist3(agent.pos, player);

    // --- senses (ai-combat.md §2, ghidra §5) --------------------------------
    let forward = yaw_forward(agent.yaw);
    let eye = [agent.pos[0], agent.pos[1] + EYE_HEIGHT, agent.pos[2]];
    let seen = agent.perception.can_see(eye, forward, player);
    let heard = agent.perception.can_hear(dist, input.player_noise);
    let sensed = seen || heard;

    if sensed {
        agent.last_known = Some(player);
        agent.timers.lost = 0.0;
        agent.alert = (agent.alert + dt * ALERT_RATE).min(1.0);
    } else {
        agent.timers.lost += dt;
        agent.alert = (agent.alert - dt * ALERT_DECAY).max(0.0);
    }

    // --- Screamer special: flee + scream (ai-combat.md §3.6) ----------------
    if agent.kind == ZombieKind::Screamer
        && agent.state != AiState::Flee
        && agent.state != AiState::Dead
        && sensed
        && dist < SCREAMER_FLEE_DIST
    {
        set_state(agent, AiState::Flee);
        agent.timers.scream = 0.0;
    }

    // --- transitions ---------------------------------------------------------
    let attack_range = stats.attack_range;
    match agent.state {
        AiState::Idle => {
            if sensed {
                set_state(agent, AiState::Alert);
            } else {
                agent.timers.idle -= dt;
                if agent.timers.idle <= 0.0 {
                    let ang = input.random.rem_euclid(1.0) * TAU;
                    let r = WANDER_DIST * (0.5 + input.random.rem_euclid(1.0) * 0.5);
                    agent.wander_target = Some([
                        agent.pos[0] + ang.sin() * r,
                        agent.pos[1],
                        agent.pos[2] + ang.cos() * r,
                    ]);
                    set_state(agent, AiState::Wander);
                }
            }
        }
        AiState::Wander => {
            if sensed {
                set_state(agent, AiState::Alert);
            } else {
                let arrived = match agent.wander_target {
                    Some(w) => horiz_dist(agent.pos, w) < 1.0,
                    None => true,
                };
                if arrived || agent.timers.state_time > WANDER_TIMEOUT {
                    set_state(agent, AiState::Idle);
                }
            }
        }
        AiState::Alert => {
            if agent.alert >= CONFIRM_ALERT {
                set_state(agent, AiState::Chase);
            } else if !sensed && agent.timers.lost > ALERT_GIVEUP {
                set_state(agent, AiState::Idle);
            }
        }
        AiState::Chase => {
            if dist <= attack_range {
                set_state(agent, AiState::Attack);
            } else if agent.timers.lost > LOST_GIVEUP || dist > DEAGGRO_DIST {
                agent.timers.search = SEARCH_TIMEOUT;
                set_state(agent, AiState::Search);
            }
        }
        AiState::Attack => {
            if dist > attack_range * ATTACK_REENGAGE {
                set_state(agent, AiState::Chase);
            } else if agent.timers.lost > LOST_GIVEUP {
                agent.timers.search = SEARCH_TIMEOUT;
                set_state(agent, AiState::Search);
            }
        }
        AiState::Search => {
            if sensed {
                set_state(agent, AiState::Alert);
            } else {
                agent.timers.search -= dt;
                if agent.timers.search <= 0.0 {
                    set_state(agent, AiState::Idle);
                }
            }
        }
        AiState::Flee => {
            if dist >= SCREAMER_SAFE_DIST {
                if sensed {
                    set_state(agent, AiState::Alert);
                } else {
                    set_state(agent, AiState::Idle);
                }
            }
            agent.timers.scream -= dt;
            if agent.timers.scream <= 0.0 {
                out.noise = SCREAMER_NOISE;
                agent.timers.scream = SCREAM_INTERVAL;
            }
        }
        AiState::Dead => {}
    }

    // --- attack initiation (ghidra §10) -------------------------------------
    if agent.state == AiState::Attack
        && sensed
        && dist <= attack_range * ATTACK_REENGAGE
        && agent.attack_cd <= 0.0
    {
        out.attack = true;
        agent.attack_cd = ATTACK_COOLDOWN;
    }

    // --- steering / speed ----------------------------------------------------
    match agent.state {
        AiState::Idle | AiState::Dead => {}
        AiState::Wander => {
            if let Some(w) = agent.wander_target {
                out.move_dir = horiz_dir(agent.pos, w);
            }
            out.speed = stats.walk_speed;
        }
        AiState::Alert | AiState::Search => {
            if let Some(t) = agent.last_known {
                out.move_dir = horiz_dir(agent.pos, t);
            }
            out.speed = stats.walk_speed;
        }
        AiState::Chase => {
            if let Some(t) = agent.last_known {
                out.move_dir = horiz_dir(agent.pos, t);
            }
            out.speed = stats.run_speed;
        }
        AiState::Attack => {
            // Face the target but hold position for the strike.
            if let Some(t) = agent.last_known {
                turn_toward(agent, horiz_dir(agent.pos, t), dt);
            }
        }
        AiState::Flee => {
            let away = horiz_dir(agent.pos, player);
            out.move_dir = [-away[0], -away[1]];
            out.speed = stats.run_speed;
        }
    }

    // --- per-kind locomotion parity -----------------------------------------
    if out.speed > 0.0 {
        match agent.kind {
            // Ferals are permanently "Fast" (AlwaysLive / Fast=true).
            ZombieKind::Feral => out.speed = stats.run_speed,
            // Bloaters never accelerate past their slow pursue speed.
            ZombieKind::Bloater => out.speed = out.speed.min(stats.run_speed),
            _ => {}
        }
    }

    // Face the movement direction (except while attacking, handled above).
    if agent.state != AiState::Attack {
        turn_toward(agent, out.move_dir, dt);
    }

    // Re-arm the idle clock whenever we settle back into Idle.
    if agent.state == AiState::Idle && prev != AiState::Idle {
        agent.timers.idle = IDLE_PERIOD + input.random.rem_euclid(1.0) * IDLE_PERIOD;
    }

    out.state = agent.state;
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn input(player: [f32; 3], noise: f32, dt: f32) -> AiInput {
        AiInput { player_pos: player, player_noise: noise, dt, random: 0.5 }
    }

    #[test]
    fn sees_inside_cone_not_behind() {
        let p = Perception::new(20.0, 60.0, 8.0);
        let eye = [0.0, 0.0, 0.0];
        let fwd = [0.0, 0.0, 1.0];
        assert!(p.can_see(eye, fwd, [0.0, 0.0, 10.0]), "in front and in range");
        assert!(!p.can_see(eye, fwd, [0.0, 0.0, -10.0]), "directly behind");
        assert!(!p.can_see(eye, fwd, [0.0, 0.0, 25.0]), "beyond sight range");
    }

    #[test]
    fn close_range_ignores_fov() {
        let p = Perception::new(20.0, 60.0, 8.0);
        let eye = [0.0, 0.0, 0.0];
        let fwd = [0.0, 0.0, 1.0];
        // Directly behind, but inside CLOSE_RANGE -> still detected.
        assert!(p.can_see(eye, fwd, [0.0, 0.0, -1.5]));
    }

    #[test]
    fn can_hear_scales_with_noise() {
        let p = Perception::new(15.0, 60.0, 8.0);
        assert!(p.can_hear(20.0, 3.0), "20 m within 8*3 m");
        assert!(!p.can_hear(20.0, 1.0), "20 m beyond 8 m");
        assert!(!p.can_hear(1.0, 0.0), "silence is never heard");
    }

    #[test]
    fn loud_noise_alerts_outside_sight() {
        let mut a = AiAgent::new(ZombieKind::Grunt, [0.0, 0.0, 0.0]);
        let inp = input([0.0, 0.0, 30.0], 5.0, 0.1);
        assert!(!a.perception.can_see([0.0, EYE_HEIGHT, 0.0], [0.0, 0.0, 1.0], inp.player_pos));
        let out = update(&mut a, &inp);
        assert_eq!(out.state, AiState::Alert);
        assert!(a.last_known.is_some());
    }

    #[test]
    fn alert_confirms_to_chase() {
        let mut a = AiAgent::new(ZombieKind::Grunt, [0.0, 0.0, 0.0]);
        let inp = input([0.0, 0.0, 5.0], 0.0, 0.1);
        assert_eq!(update(&mut a, &inp).state, AiState::Alert);
        let mut state = AiState::Alert;
        for _ in 0..20 {
            state = update(&mut a, &inp).state;
            if state == AiState::Chase {
                break;
            }
        }
        assert_eq!(state, AiState::Chase);
    }

    #[test]
    fn chase_transitions_to_attack_in_range() {
        let mut a = AiAgent::new(ZombieKind::Grunt, [0.0, 0.0, 0.0]);
        a.state = AiState::Chase;
        let inp = input([0.0, 0.0, 1.0], 0.0, 0.05);
        let out = update(&mut a, &inp);
        assert_eq!(out.state, AiState::Attack);
        assert!(out.attack, "a strike is initiated within attack_range");
    }

    #[test]
    fn loses_sight_enters_search_then_idle() {
        let mut a = AiAgent::new(ZombieKind::Grunt, [0.0, 0.0, 0.0]);
        a.state = AiState::Chase;
        a.last_known = Some([0.0, 0.0, 0.0]);
        // Player far outside sight/hearing, so the target is lost.
        let inp = input([0.0, 0.0, 100.0], 0.0, 0.1);

        let mut entered_search = false;
        for _ in 0..20 {
            if update(&mut a, &inp).state == AiState::Search {
                entered_search = true;
                break;
            }
        }
        assert!(entered_search, "expected Search after losing the target");

        let mut returned_idle = false;
        for _ in 0..120 {
            if update(&mut a, &inp).state == AiState::Idle {
                returned_idle = true;
                break;
            }
        }
        assert!(returned_idle, "Search should time out into Idle");
    }

    #[test]
    fn feral_runs_fast_bloater_stays_slow() {
        let mut feral = AiAgent::new(ZombieKind::Feral, [0.0, 0.0, 0.0]);
        feral.state = AiState::Chase;
        let mut bloater = AiAgent::new(ZombieKind::Bloater, [0.0, 0.0, 0.0]);
        bloater.state = AiState::Chase;
        let inp = input([0.0, 0.0, 5.0], 0.0, 0.05);

        let f = update(&mut feral, &inp);
        let b = update(&mut bloater, &inp);
        assert_eq!(f.state, AiState::Chase);
        assert_eq!(b.state, AiState::Chase);
        assert_eq!(f.speed, ZombieKind::Feral.stats().run_speed);
        assert_eq!(b.speed, ZombieKind::Bloater.stats().run_speed);
        assert!(f.speed > b.speed, "feral must outpace a bloater");
    }

    #[test]
    fn screamer_flees_and_emits_noise() {
        let mut a = AiAgent::new(ZombieKind::Screamer, [0.0, 0.0, 0.0]);
        a.state = AiState::Alert;
        let inp = input([0.0, 0.0, 3.0], 0.0, 0.1);
        let out = update(&mut a, &inp);
        assert_eq!(out.state, AiState::Flee);
        assert!(out.noise > 0.0, "screamer emits alert noise");
        assert!(out.move_dir[1] < 0.0, "flees away from a +z player");
    }

    #[test]
    fn horde_spawn_scales_with_noise_and_respects_caps() {
        let mut rng = SmallRng::new(0xDEAD_BEEF);

        let mut low_dir = HordeDirector::new();
        let low = low_dir.on_noise(1.0, &mut rng).expect("low noise still spawns");
        let mut high_dir = HordeDirector::new();
        let high = high_dir.on_noise(100.0, &mut rng).expect("high noise spawns");
        assert!(high.count() > low.count(), "louder noise -> bigger spawn");

        let mut d = HordeDirector::new();
        let mut ambient = 0;
        let mut freaks = 0;
        for _ in 0..100 {
            match d.on_noise(1000.0, &mut rng) {
                Some(s) => {
                    ambient += s.ambient;
                    freaks += s.freaks;
                }
                None => break,
            }
        }
        assert_eq!(ambient, d.ambient_cap, "ambient cap respected");
        assert_eq!(freaks, d.freak_cap, "freak cap respected");
        assert!(d.on_noise(1000.0, &mut rng).is_none(), "full horde spawns nothing");
    }

    #[test]
    fn horde_tick_refills_ambient() {
        let mut d = HordeDirector::new();
        d.ambient = 0;
        d.tick(1000.0);
        assert_eq!(d.ambient, d.ambient_cap);
        // Never exceeds the cap.
        d.tick(1000.0);
        assert_eq!(d.ambient, d.ambient_cap);
    }

    #[test]
    fn small_rng_is_deterministic_and_in_unit_range() {
        let mut a = SmallRng::new(42);
        let mut b = SmallRng::new(42);
        for _ in 0..64 {
            let x = a.next_f32();
            assert!((0.0..1.0).contains(&x));
            assert_eq!(x, b.next_f32());
        }
        assert_ne!(SmallRng::new(1).next_u64(), SmallRng::new(2).next_u64());
    }
}
