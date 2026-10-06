//! 1:1 port of the Class3 zombie AI: `NSZombieManager` ("Lab AI") plus the
//! CryAI hand-off, and the horde/ambush spawn director.
//!
//! This is the *real* native design, not the earlier approximation in
//! [`crate::ai`].  It follows the recovered binary:
//!
//! * **Two-tier model** (`zombie-runtime.md` §0): 256 logical zombies always
//!   simulated; a small pool of 24 is promoted to real CryEngine entities
//!   ("Cry AI") when the player is close.  Thresholds are `ActivationDist`
//!   (15 m) and `DeactivationDist` (20 m).
//! * **Behaviour table** (§6, `NSZombieManager::BehaviorsSet` @ `0131cb00`):
//!   a fixed, priority-ordered table.  [`determine_behavior`] implements the
//!   exact selection rule
//!   `(required & 0x1fffffff) == (trigger & mask)`.
//! * **Perception** (§5, `DeterminePerception` @ `01321d20`): per-def aggro
//!   range + global aggro, FOV cone, hearing, LOS, last-known position and the
//!   `DAT_01defd58` flag word.
//! * **Movement** (§7, `MoveZombies` @ `01331ef0`): steering/integration at the
//!   anim-group locomotion speeds exposed by [`game::ZombieKind::stats`].
//! * **Attack initiation** (§10): the native side only raises the trigger; no
//!   damage is applied here.
//! * **Spawn director** (§4): ambush/ambient cadence, freak caps, horde refill
//!   (`AddHorde` @ `0131af80`, roaming refill @ `0135c080`).
//!
//! Movement speeds are the anim-group triplets (`ai-combat.md` §1.3), which is
//! what the native `MoveZombies` integrator consumes.

use crate::game::ZombieKind;
use std::f32::consts::{PI, TAU};

// ---------------------------------------------------------------------------
// Caps (`zombie-runtime.md` §3)
// ---------------------------------------------------------------------------

/// Logical zombie slots (`MAX_ZOMBIES`).
pub const MAX_ZOMBIES: usize = 256;
/// Promoted / live entities (`m_liveZombies`).
pub const MAX_LIVE_ZOMBIES: usize = 24;
/// Delayed-release entity pool (`MAX_DEAD_ZOMBIES`).
pub const MAX_DEAD_ZOMBIES: usize = 24;
/// Horde slots (`MAX_HORDES`).
pub const MAX_HORDES: usize = 8;
/// Nav-path pool (`NAV_PATH_MAX_COUNT`).
pub const NAV_PATH_MAX_COUNT: usize = 48;
/// Behaviour table capacity (`BEHAVIOR_MAX`).
pub const BEHAVIOR_MAX: usize = 64;
/// Echo target slots (`MAX_ECHOES`).
pub const MAX_ECHOES: usize = 8;
/// Draw-aggro slots (`MAX_DRAW_AGGRO`).
pub const MAX_DRAW_AGGRO: usize = 2;
/// Disturbance slots (`MAX_DISTURBANCES`).
pub const MAX_DISTURBANCES: usize = 2;
/// Ambush tags per spawn point (`MAX_AMBUSH_TAGS`).
pub const MAX_AMBUSH_TAGS: usize = 16;

// ---------------------------------------------------------------------------
// Perception / behaviour tuning (`ai-combat.md` §2, `zombies.xml` globals)
// ---------------------------------------------------------------------------

/// Global `BaseFOV` (degrees).  `DeterminePerception` uses one global cone,
/// not a per-type FOV (`zombie-runtime.md` §4.1/§13).
pub const BASE_FOV: f32 = 80.0;
/// Global `AggroDist` added to each def's base range.
pub const GLOBAL_AGGRO_DIST: f32 = 4.0;
/// Beyond `DeaggroDist` a chase is abandoned.
pub const DEAGGRO_DIST: f32 = 30.0;
/// Lab → Cry promotion distance (`ActivationDist`).
pub const ACTIVATION_DIST: f32 = 15.0;
/// Cry → Lab demotion distance (`DeactivationDist`).
pub const DEACTIVATION_DIST: f32 = 20.0;
/// Crouch/inside stealth FOV multiplier.
pub const STEALTH_FOV_MULTIPLIER: f32 = 0.5;
/// "Reached last-known position" radius (`InvestigationDist`).
pub const INVESTIGATION_DIST: f32 = 6.0;
/// Search duration (`InvestigationTimeout`).
pub const INVESTIGATION_TIMEOUT: f32 = 8.0;
/// Lost-target post timer from `DeterminePerception` (0.5 s).
pub const LOST_GIVEUP: f32 = 0.5;
/// Inside this range the FOV cone is ignored (dist² < 1 branch, §5.3).
pub const CLOSE_RANGE: f32 = 2.0;
/// Melee reach (`melee.damageRadius`).
pub const ATTACK_RANGE: f32 = 1.1;
/// `Timer_CombatCooldown` equivalent.
pub const ATTACK_COOLDOWN: f32 = 1.0;
/// Alert confidence build rate / decay (`CONFIRM_ALERT` reached in ~0.25 s).
pub const ALERT_RATE: f32 = 4.0;
pub const ALERT_DECAY: f32 = 1.0;
pub const CONFIRM_ALERT: f32 = 1.0;
/// Unconfirmed contact forgotten after this long.
pub const ALERT_GIVEUP: f32 = 1.5;
/// `IdlePeriod` / `WanderDist` (`ZombieTypes.xml`).
pub const IDLE_PERIOD: f32 = 5.0;
pub const WANDER_DIST: f32 = 10.0;
pub const WANDER_TIMEOUT: f32 = 10.0;
/// Eye offset used by the LOS/range test.
pub const EYE_HEIGHT: f32 = 1.5;
/// Heading turn rate (rad/s), approximates `maxTurnRadius`.
pub const TURN_RATE: f32 = 4.0;
/// Movement sound radii (`zombies.xml`): walk / run / sprint.
pub const SOUND_WALK: f32 = 2.0;
pub const SOUND_RUN: f32 = 4.0;
pub const SOUND_SPRINT: f32 = 6.0;
/// `CombatSoundDist` — the radius of a Screamer's scream echo.
pub const COMBAT_SOUND_DIST: f32 = 8.0;
/// Screamer special (`ai-combat.md` §3.6 / `Zombie.CreateEcho`).
pub const SCREAMER_FLEE_DIST: f32 = 6.0;
pub const SCREAMER_SAFE_DIST: f32 = 14.0;
pub const SCREAM_INTERVAL: f32 = 2.0;

// ---------------------------------------------------------------------------
// Horde tuning (`ai-combat.md` §5.1, `zombie-runtime.md` §4)
// ---------------------------------------------------------------------------

/// `Zombies.Horde.MaxSize`.
pub const HORDE_MAX_SIZE: u32 = 8;
/// `Zombies.Horde.MaxAmbient`.
pub const HORDE_MAX_AMBIENT: u32 = 3;
/// Feral / Juggernaut `Max` (freak caps).
pub const FREAK_CAP: u32 = 4;
/// `Zombies.AmbientRespawnRate`.
pub const AMBIENT_RESPAWN_RATE: f32 = 15.0;
/// Horde spawn retry delay (`0135c080:242`).
pub const HORDE_RETRY: f32 = 5.0;
/// Roaming-horde per-tick target (clamped to 6 in `0135c080`).
pub const HORDE_REFILL_TARGET: u32 = 6;

// ---------------------------------------------------------------------------
// Behaviour enum + the native priority table
// ---------------------------------------------------------------------------

/// Zombie behaviour.
///
/// The first block is the native `NSZombieManager::BehaviorsSet` table (real
/// names, registration order = priority, `zombie-runtime.md` §6.1).  The second
/// block is the higher-level FSM state (`ai-combat.md` §3.6) that the native
/// table maps onto and that [`ZombieOutput`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZombieBehavior {
    // --- native table (priority order) ---
    BarricadeWatch,
    Barricade,
    CryAi,
    Ambush,
    VehicleColl,
    VehicleTimerExp,
    Vehicle,
    /// Row 6, name pointer `01840f88` (unrecovered).
    State100,
    BlkdTargetClose,
    UnknownMarkup,
    /// Row 9, name pointer `01840fb0` (unrecovered).
    State5000000,
    Markup,
    BadPath,
    CollisionIdle,
    CollisionPath,
    FollowPath,
    TargetSeenKnown,
    Investigate,
    InvestigateMove,
    TargetKnown,
    TargetSeen,
    TargetNear,
    MissionAnchor,
    Disturb,
    EchoMoveTo,
    EchoLookAt,
    EchoNotice,
    TrackPlayer,
    DisturbWander,
    HordeTimerExp,
    WanderEnd,
    /// Row 34, name pointer `01771cf0` (the idle/default behaviour).
    Default,

    // --- high-level FSM states (`ai-combat.md` §3.6) ---
    Dead,
    Idle,
    Wander,
    Alert,
    Chase,
    Attack,
    Search,
    Flee,
    Grab,
    Feasting,
}

/// One row of the native behaviour table: `(name, behaviour, trigger, required)`.
pub type BehaviorDef = (&'static str, ZombieBehavior, u32, u32);

/// The real `BehaviorsSet` table in registration order (`0131cb00`).
/// Index 0 has the highest priority.
pub const ZOMBIE_BEHAVIOR_TABLE: [BehaviorDef; 35] = [
    ("BarricadeWatch", ZombieBehavior::BarricadeWatch, 0x0008_0200, 0x0008_0000),
    ("Barricade", ZombieBehavior::Barricade, 0x0004_0200, 0x0004_0000),
    ("CryAi", ZombieBehavior::CryAi, 0x0000_0800, 0x0000_0800),
    ("Ambush", ZombieBehavior::Ambush, 0x0010_0000, 0x0010_0000),
    ("VehicleColl", ZombieBehavior::VehicleColl, 0x0000_3001, 0x0000_3000),
    ("VehicleTimerExp", ZombieBehavior::VehicleTimerExp, 0x0000_1001, 0x0000_1001),
    // Inlined manual registration at `0131cc0a` (row 6, not 33 as an earlier
    // reading of the decomp had it).
    ("Vehicle", ZombieBehavior::Vehicle, 0x0000_3000, 0x0000_1000),
    ("?01840f88", ZombieBehavior::State100, 0x0000_0100, 0x0000_0100),
    ("BlkdTargetClose", ZombieBehavior::BlkdTargetClose, 0x0000_0048, 0x0000_0048),
    ("UnknownMarkup", ZombieBehavior::UnknownMarkup, 0x1000_0000, 0x1000_0000),
    ("?01840fb0", ZombieBehavior::State5000000, 0x0500_0000, 0x0500_0000),
    ("Markup", ZombieBehavior::Markup, 0x0100_0000, 0x0100_0000),
    ("BadPath", ZombieBehavior::BadPath, 0x0200_0002, 0x0200_0000),
    ("CollisionIdle", ZombieBehavior::CollisionIdle, 0x00a3_201c, 0x0000_2000),
    ("CollisionPath", ZombieBehavior::CollisionPath, 0x0000_2020, 0x0000_2000),
    ("FollowPath", ZombieBehavior::FollowPath, 0x0000_0020, 0x0000_0020),
    ("TargetSeenKnown", ZombieBehavior::TargetSeenKnown, 0x0800_000c, 0x0000_000c),
    ("Investigate", ZombieBehavior::Investigate, 0x0800_000a, 0x0800_000a),
    ("InvestigateMove", ZombieBehavior::InvestigateMove, 0x0800_0008, 0x0800_0008),
    ("Investigate", ZombieBehavior::Investigate, 0x0000_008a, 0x0000_008a),
    ("InvestigateMove", ZombieBehavior::InvestigateMove, 0x0000_0088, 0x0000_0088),
    ("TargetKnown", ZombieBehavior::TargetKnown, 0x0000_000c, 0x0000_0008),
    ("TargetSeen", ZombieBehavior::TargetSeen, 0x0000_000c, 0x0000_0004),
    ("TargetNear", ZombieBehavior::TargetNear, 0x0000_0040, 0x0000_0040),
    ("MissionAnchor", ZombieBehavior::MissionAnchor, 0x0080_0000, 0x0080_0000),
    ("Disturb", ZombieBehavior::Disturb, 0x0020_0000, 0x0020_0000),
    ("EchoMoveTo", ZombieBehavior::EchoMoveTo, 0x0001_0000, 0x0001_0000),
    ("EchoLookAt", ZombieBehavior::EchoLookAt, 0x0000_4000, 0x0000_4000),
    ("EchoNotice", ZombieBehavior::EchoNotice, 0x0000_8000, 0x0000_8000),
    ("TrackPlayer", ZombieBehavior::TrackPlayer, 0x0000_0010, 0x0000_0010),
    ("DisturbWander", ZombieBehavior::DisturbWander, 0x0040_0000, 0x0040_0000),
    ("HordeTimerExp", ZombieBehavior::HordeTimerExp, 0x0002_0000, 0x0002_0000),
    ("WanderEnd", ZombieBehavior::WanderEnd, 0x0000_0003, 0x0000_0003),
    ("Wander", ZombieBehavior::Wander, 0x0000_0003, 0x0000_0001),
    ("?01771cf0", ZombieBehavior::Default, 0x0000_0000, 0x0000_0000),
];

// Perception flag word `DAT_01defd58` (`zombie-runtime.md` §5.3).
/// Has an active attention target.
pub const PERCEPT_ATTENTION: u32 = 0x0001;
/// Aware / alerted.
pub const PERCEPT_AWARE: u32 = 0x0002;
/// Heard combat / echo.
pub const PERCEPT_HEARD: u32 = 0x0004;
/// Investigating / lost-target.
pub const PERCEPT_INVESTIGATE: u32 = 0x0008;
/// Target currently in LOS.
pub const PERCEPT_LOS: u32 = 0x0010;
/// Focus / attackable (in melee range).
pub const PERCEPT_FOCUS: u32 = 0x0040;
/// Reached / attack trigger.
pub const PERCEPT_REACHED: u32 = 0x0400;
/// CryAI promotion bit.
pub const PERCEPT_CRYAI: u32 = 0x0800;

/// Pick the first behaviour in [`ZOMBIE_BEHAVIOR_TABLE`] whose required bits
/// match the mask, exactly as `DetermineBehaviors` (`01321bd0`) does:
/// `(required & 0x1fffffff) == (trigger & mask)`.
pub fn determine_behavior(mask: u32) -> ZombieBehavior {
    for (_, behavior, trigger, required) in ZOMBIE_BEHAVIOR_TABLE.iter() {
        if (required & 0x1fff_ffff) == (trigger & mask) {
            return *behavior;
        }
    }
    ZombieBehavior::Default
}

// ---------------------------------------------------------------------------
// Senses
// ---------------------------------------------------------------------------

/// Sense cone / hearing model for one zombie.
///
/// `sight_range` is inclusive; `fov_cos` is the cosine of the half-angle of the
/// primary cone; `hearing_range` is multiplied by a noise loudness.  `los`
/// holds the result of the last [`ZombieSenses::determine_perception`] call
/// (whether the target was geometrically in the cone and range).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZombieSenses {
    pub sight_range: f32,
    pub fov_cos: f32,
    pub hearing_range: f32,
    pub los: bool,
}

impl ZombieSenses {
    /// Build from a sight range, a full cone angle in degrees, and hearing.
    pub fn new(sight_range: f32, fov_degrees: f32, hearing_range: f32) -> Self {
        ZombieSenses {
            sight_range,
            fov_cos: (fov_degrees.to_radians() * 0.5).cos(),
            hearing_range,
            los: false,
        }
    }

    /// Defaults for a zombie kind.
    ///
    /// Range is the per-`ZombieDef` `AggroDist` (`ai-combat.md` §2.2) plus the
    /// global `AggroDist` (4 m), matching `DeterminePerception`'s
    /// `base + globalAggro`.  The cone is the global `BaseFOV` — the native
    /// loader has no per-type FOV (`zombie-runtime.md` §4.1/§13).  Hearing uses
    /// the combat-sound radius with small freak adjustments.
    pub fn for_kind(kind: ZombieKind) -> Self {
        let (aggro, hearing) = match kind {
            ZombieKind::Grunt => (15.0, 8.0),
            ZombieKind::ArmyZombie => (15.0, 8.0),
            ZombieKind::Screamer => (15.0, 10.0),
            ZombieKind::Bloater => (15.0, 6.0),
            ZombieKind::Feral => (6.0, 12.0),
            ZombieKind::Juggernaut => (6.0, 8.0),
        };
        ZombieSenses::new(aggro + GLOBAL_AGGRO_DIST, BASE_FOV, hearing)
    }

    /// Range gate + FOV cone test.  Targets within [`CLOSE_RANGE`] are always
    /// detected regardless of the cone (`dist² < 1` branch, §5.3).
    pub fn can_see(&self, eye: [f32; 3], forward: [f32; 3], target: [f32; 3]) -> bool {
        let d = sub3(target, eye);
        let dist_sq = dot3(d, d);
        if !dist_sq.is_finite() || dist_sq > self.sight_range * self.sight_range {
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

    /// `noise` is a loudness multiplier (`0` = silent).  A source is heard when
    /// within `hearing_range * noise`.
    pub fn can_hear(&self, dist: f32, noise: f32) -> bool {
        noise > 0.0 && dist.is_finite() && dist <= self.hearing_range * noise
    }

    /// Run the full `DeterminePerception` test for one target and store the
    /// geometric LOS result.  Returns `true` if seen *or* heard.
    pub fn determine_perception(
        &mut self,
        eye: [f32; 3],
        forward: [f32; 3],
        target: [f32; 3],
        noise: f32,
    ) -> bool {
        let seen = self.can_see(eye, forward, target);
        self.los = seen;
        let dist = dist3(eye, target);
        seen || self.can_hear(dist, noise)
    }
}

// ---------------------------------------------------------------------------
// Agent state
// ---------------------------------------------------------------------------

/// Per-agent countdown timers (seconds remaining / elapsed).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ZombieTimers {
    /// Idle period before a wander leg (`IdlePeriod ± IdleVariance`).
    pub idle: f32,
    /// Remaining search time (`InvestigationTimeout`).
    pub search: f32,
    /// Time since the target was last sensed.
    pub lost: f32,
    /// Time spent in the current behaviour.
    pub state_time: f32,
    /// Scream re-arm timer (Screamer).
    pub scream: f32,
}

/// Per-agent cooldowns (seconds remaining).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ZombieCooldowns {
    /// `Timer_CombatCooldown` before another strike may start.
    pub attack: f32,
    /// Target-switch cooldown (`Timer_TargetSwitchCooldown`).
    pub target_switch: f32,
}

/// One simulated zombie brain.  The logical (Lab AI) representation.
#[derive(Debug, Clone)]
pub struct ZombieAgent {
    pub kind: ZombieKind,
    pub behavior: ZombieBehavior,
    pub pos: [f32; 3],
    pub yaw: f32,
    /// Selected target / last-known position (`DAT_01e1bea0`).
    pub target: Option<[f32; 3]>,
    /// Alert confidence `0..=1`.
    pub alert: f32,
    pub timers: ZombieTimers,
    pub cooldowns: ZombieCooldowns,
    pub senses: ZombieSenses,
    /// Two-tier promotion: `true` while handed to CryAI (`DAT_01df4b80` bit).
    pub live: bool,
    /// Desired behaviour bitmask `DAT_01df2130` from the last perception tick.
    pub behavior_mask: u32,
    /// Native behaviour selected from [`ZOMBIE_BEHAVIOR_TABLE`] this tick.
    pub native_behavior: ZombieBehavior,
    /// Current wander waypoint.
    pub wander_target: Option<[f32; 3]>,
}

impl ZombieAgent {
    /// Create a logical zombie at `pos` with idle timers.
    pub fn new(kind: ZombieKind, pos: [f32; 3]) -> Self {
        ZombieAgent {
            kind,
            behavior: ZombieBehavior::Idle,
            pos,
            yaw: 0.0,
            target: None,
            alert: 0.0,
            timers: ZombieTimers { idle: IDLE_PERIOD, ..ZombieTimers::default() },
            cooldowns: ZombieCooldowns::default(),
            senses: ZombieSenses::for_kind(kind),
            live: false,
            behavior_mask: 0,
            native_behavior: ZombieBehavior::Default,
            wander_target: None,
        }
    }
}

/// External stimulus for one tick.
#[derive(Debug, Clone, Copy)]
pub struct ZombieInput {
    pub player_pos: [f32; 3],
    pub player_noise: f32,
    pub dt: f32,
    /// Uniform random value in `[0, 1)` supplied by the caller.
    pub rng: f32,
}

/// Steering + action result for one tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZombieOutput {
    /// Normalised horizontal `[x, z]` move direction.
    pub move_dir: [f32; 2],
    pub speed: f32,
    pub attack: bool,
    pub behavior: ZombieBehavior,
    /// Noise this zombie emits this tick (movement tier, or a Screamer scream).
    pub noise: f32,
}

// ---------------------------------------------------------------------------
// Deterministic RNG
// ---------------------------------------------------------------------------

/// Tiny deterministic xorshift64* generator (reproducible across platforms).
#[derive(Debug, Clone)]
pub struct SmallRng {
    state: u64,
}

impl SmallRng {
    pub fn new(seed: u64) -> Self {
        SmallRng { state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed } }
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
// Horde director (two-tier spawn ecology, §4)
// ---------------------------------------------------------------------------

/// A batch of zombies the director decided to spawn.
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

/// Ambient/ambush + horde refill director.
///
/// Models the two-tier ecology: a large logical population (`MAX_ZOMBIES`)
/// with a small promoted live pool (`MAX_LIVE_ZOMBIES`), ambient cap
/// (`HORDE_MAX_AMBIENT`), freak cap (`FREAK_CAP`), 8 horde slots and the 5 s
/// roaming-horde refill retry from `0135c080`.
#[derive(Debug, Clone)]
pub struct HordeDirector {
    /// Total logical zombies (0..=[`MAX_ZOMBIES`]).
    pub logical: u32,
    /// Promoted live entities (0..=[`MAX_LIVE_ZOMBIES`]).
    pub live: u32,
    /// Current ambient count.
    pub ambient: u32,
    /// Current freak count.
    pub freaks: u32,
    /// Current horde count.
    pub horde: u32,
    pub ambient_cap: u32,
    pub freak_cap: u32,
    pub max_size: u32,
    pub max_logical: u32,
    pub max_live: u32,
    pub max_hordes: u32,
    /// Ambient respawn accumulator.
    pub refill_timer: f32,
    /// Roaming-horde spawn timer.
    pub horde_spawn_timer: f32,
    /// Horde respawn delay (decayed accumulator).
    pub respawn_delay: f32,
    pub total_spawned: u32,
}

impl HordeDirector {
    pub fn new() -> Self {
        HordeDirector {
            logical: 0,
            live: 0,
            ambient: 0,
            freaks: 0,
            horde: 0,
            ambient_cap: HORDE_MAX_AMBIENT,
            freak_cap: FREAK_CAP,
            max_size: HORDE_MAX_SIZE,
            max_logical: MAX_ZOMBIES as u32,
            max_live: MAX_LIVE_ZOMBIES as u32,
            max_hordes: MAX_HORDES as u32,
            refill_timer: 0.0,
            horde_spawn_timer: HORDE_RETRY,
            respawn_delay: 0.0,
            total_spawned: 0,
        }
    }

    /// Respond to a noise event (echo / disturbance / ambush trigger).
    ///
    /// Spawn count scales with `noise`, is jittered by `rng`, and is clamped to
    /// the ambient/freak/logical caps.  Returns `None` when nothing could be
    /// spawned (silence or the horde is full).
    pub fn on_noise(&mut self, noise: f32, rng: &mut SmallRng) -> Option<HordeSpawn> {
        if !(noise > 0.0) || !noise.is_finite() {
            return None;
        }
        let base = (noise * 0.5).floor();
        let jitter = rng.next_f32();
        let desired = (base + jitter).clamp(1.0, self.max_size as f32) as u32;

        let mut remaining = desired;
        let mut a = remaining.min(self.ambient_cap.saturating_sub(self.ambient));
        remaining -= a;
        let mut f = remaining.min(self.freak_cap.saturating_sub(self.freaks));

        // Logical population cap (`MAX_ZOMBIES`), reduce freaks first.
        let logical_room = self.max_logical.saturating_sub(self.logical);
        while a + f > logical_room {
            if f > 0 {
                f -= 1;
            } else if a > 0 {
                a -= 1;
            } else {
                break;
            }
        }

        if a == 0 && f == 0 {
            return None;
        }
        self.ambient += a;
        self.freaks += f;
        self.logical += a + f;
        self.total_spawned += a + f;
        Some(HordeSpawn { ambient: a, freaks: f })
    }

    /// Promote `n` logical zombies into the live CryAI pool, clamped to
    /// [`MAX_LIVE_ZOMBIES`].  Returns how many were actually promoted.
    pub fn promote(&mut self, n: u32) -> u32 {
        let room = self.max_live.saturating_sub(self.live);
        let promoted = n.min(room);
        self.live += promoted;
        promoted
    }

    /// Demote `n` live zombies back to the logical pool.
    pub fn demote(&mut self, n: u32) -> u32 {
        let demoted = n.min(self.live);
        self.live -= demoted;
        demoted
    }

    /// Per-frame refill (`0135c080` + ambient respawn).
    pub fn tick(&mut self, dt: f32) {
        if !(dt > 0.0) || !dt.is_finite() {
            return;
        }
        // Ambient respawn: one zombie every `AmbientRespawnRate` seconds.
        self.refill_timer += dt;
        while self.refill_timer >= AMBIENT_RESPAWN_RATE {
            self.refill_timer -= AMBIENT_RESPAWN_RATE;
            if self.ambient < self.ambient_cap && self.logical < self.max_logical {
                self.ambient += 1;
                self.logical += 1;
                self.total_spawned += 1;
            }
        }
        // Roaming-horde refill with a 5 s retry on expiry (`0135c080:242`).
        self.horde_spawn_timer -= dt;
        if self.horde_spawn_timer <= 0.0 {
            let target = self.max_size.min(HORDE_REFILL_TARGET);
            if self.horde < target && self.logical < self.max_logical {
                self.horde += 1;
                self.logical += 1;
                self.total_spawned += 1;
                self.horde_spawn_timer = HORDE_RETRY;
            } else {
                self.horde_spawn_timer = 0.0;
            }
        }
        // Respawn-delay decay.
        self.respawn_delay = (self.respawn_delay - dt).max(0.0);
    }
}

impl Default for HordeDirector {
    fn default() -> Self {
        HordeDirector::new()
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    len3(sub3(b, a))
}

fn yaw_forward(yaw: f32) -> [f32; 3] {
    [yaw.sin(), 0.0, yaw.cos()]
}

/// Normalised horizontal `[x, z]` direction from `from` toward `to`.
fn horiz_dir(from: [f32; 3], to: [f32; 3]) -> [f32; 2] {
    let dx = to[0] - from[0];
    let dz = to[2] - from[2];
    let l = (dx * dx + dz * dz).sqrt();
    if !l.is_finite() || l < 1e-4 {
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

/// Rotate `yaw` toward the horizontal direction `dir` at [`TURN_RATE`].
fn turn_toward(agent: &mut ZombieAgent, dir: [f32; 2], dt: f32) {
    if !dir[0].is_finite() || !dir[1].is_finite() || (dir[0] == 0.0 && dir[1] == 0.0) {
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
// Behaviour mapping + transitions
// ---------------------------------------------------------------------------

/// Map a native behaviour selection to the high-level FSM state (`§3.6`).
fn classify(native: ZombieBehavior, in_range: bool, sensed: bool) -> ZombieBehavior {
    match native {
        ZombieBehavior::TargetNear => ZombieBehavior::Attack,
        ZombieBehavior::CryAi => {
            if in_range {
                ZombieBehavior::Attack
            } else {
                ZombieBehavior::Chase
            }
        }
        ZombieBehavior::TargetSeenKnown
        | ZombieBehavior::TargetSeen
        | ZombieBehavior::TrackPlayer => ZombieBehavior::Chase,
        ZombieBehavior::TargetKnown => {
            if sensed {
                ZombieBehavior::Chase
            } else {
                ZombieBehavior::Search
            }
        }
        ZombieBehavior::Investigate | ZombieBehavior::InvestigateMove => ZombieBehavior::Search,
        ZombieBehavior::EchoNotice
        | ZombieBehavior::EchoLookAt
        | ZombieBehavior::EchoMoveTo
        | ZombieBehavior::Disturb
        | ZombieBehavior::DisturbWander
        | ZombieBehavior::Ambush => ZombieBehavior::Alert,
        ZombieBehavior::Wander | ZombieBehavior::WanderEnd => ZombieBehavior::Wander,
        _ => ZombieBehavior::Idle,
    }
}

/// Build the desired-behaviour mask `DAT_01df2130` from the current senses.
fn build_mask(agent: &ZombieAgent, seen: bool, heard: bool, in_range: bool) -> u32 {
    let mut mask = 0u32;
    if seen || heard {
        mask |= PERCEPT_ATTENTION;
    }
    if seen {
        mask |= PERCEPT_AWARE | PERCEPT_LOS;
    }
    if heard {
        mask |= PERCEPT_HEARD;
    }
    if agent.behavior == ZombieBehavior::Search {
        mask |= PERCEPT_INVESTIGATE;
    }
    if in_range {
        mask |= PERCEPT_FOCUS | PERCEPT_REACHED;
    }
    if agent.live {
        mask |= PERCEPT_CRYAI;
    }
    mask
}

fn random_waypoint(agent: &ZombieAgent, rng: f32) -> [f32; 3] {
    let a = (rng.rem_euclid(1.0)) * TAU;
    let r = WANDER_DIST * (0.5 + (rng * 1.7).rem_euclid(1.0) * 0.5);
    [agent.pos[0] + a.sin() * r, agent.pos[1], agent.pos[2] + a.cos() * r]
}

/// Enter `next`, running the native enter-hook side effects (timer re-arms,
/// wander waypoint selection, scream re-arm).
fn enter(agent: &mut ZombieAgent, next: ZombieBehavior, rng: f32) {
    if agent.behavior == next {
        return;
    }
    agent.behavior = next;
    agent.timers.state_time = 0.0;
    match next {
        ZombieBehavior::Search => {
            agent.timers.search = INVESTIGATION_TIMEOUT;
        }
        ZombieBehavior::Wander => {
            agent.wander_target = Some(random_waypoint(agent, rng));
        }
        ZombieBehavior::Idle => {
            agent.timers.idle = IDLE_PERIOD + rng.rem_euclid(1.0) * IDLE_PERIOD;
        }
        ZombieBehavior::Flee => {
            agent.timers.scream = 0.0;
        }
        _ => {}
    }
}

/// Resolve the next high-level behaviour from the native selection plus the
/// explicit `§12` transition rules.
fn resolve(
    agent: &ZombieAgent,
    desired: ZombieBehavior,
    sensed: bool,
    in_range: bool,
    dist: f32,
) -> ZombieBehavior {
    use ZombieBehavior::*;
    match agent.behavior {
        Dead => Dead,
        Flee => {
            if dist >= SCREAMER_SAFE_DIST {
                if sensed {
                    Alert
                } else {
                    Idle
                }
            } else {
                Flee
            }
        }
        Attack => {
            if !in_range {
                Chase
            } else if !sensed && agent.timers.lost > LOST_GIVEUP {
                Search
            } else {
                Attack
            }
        }
        Chase => {
            if in_range && sensed {
                Attack
            } else if !sensed && (agent.timers.lost > LOST_GIVEUP || dist > DEAGGRO_DIST) {
                Search
            } else {
                Chase
            }
        }
        Search => {
            if sensed {
                Alert
            } else if agent.timers.search <= 0.0 {
                Idle
            } else {
                Search
            }
        }
        Alert => {
            if agent.alert >= CONFIRM_ALERT {
                if desired == Attack {
                    Attack
                } else if desired == Search {
                    Search
                } else {
                    Chase
                }
            } else if !sensed && agent.timers.lost > ALERT_GIVEUP {
                Idle
            } else {
                Alert
            }
        }
        Idle | Wander | Grab | Feasting => {
            if sensed {
                if agent.alert >= CONFIRM_ALERT && desired != Idle {
                    desired
                } else {
                    Alert
                }
            } else {
                match agent.behavior {
                    Idle => {
                        if agent.timers.idle <= 0.0 {
                            Wander
                        } else {
                            Idle
                        }
                    }
                    Wander => {
                        let arrived = agent
                            .wander_target
                            .map_or(true, |w| horiz_dist(agent.pos, w) < 1.0);
                        if arrived || agent.timers.state_time > WANDER_TIMEOUT {
                            Idle
                        } else {
                            Wander
                        }
                    }
                    other => other,
                }
            }
        }
        // Native-only states never persist as high-level state.
        _ => desired,
    }
}

// ---------------------------------------------------------------------------
// Per-tick update
// ---------------------------------------------------------------------------

/// Advance one zombie brain by `input.dt` and return its steering/action.
pub fn update(agent: &mut ZombieAgent, input: &ZombieInput) -> ZombieOutput {
    let dt = if input.dt.is_finite() { input.dt.max(0.0) } else { 0.0 };
    let rng = if input.rng.is_finite() { input.rng.rem_euclid(1.0) } else { 0.0 };

    let mut out = ZombieOutput {
        move_dir: [0.0, 0.0],
        speed: 0.0,
        attack: false,
        behavior: agent.behavior,
        noise: 0.0,
    };

    if agent.behavior == ZombieBehavior::Dead {
        return out;
    }

    agent.timers.state_time += dt;
    agent.timers.lost += dt;
    agent.timers.search = (agent.timers.search - dt).max(0.0);
    agent.timers.idle = (agent.timers.idle - dt).max(0.0);
    agent.cooldowns.attack = (agent.cooldowns.attack - dt).max(0.0);
    agent.cooldowns.target_switch = (agent.cooldowns.target_switch - dt).max(0.0);

    let stats = agent.kind.stats();
    let player = input.player_pos;
    let dist = dist3(agent.pos, player);
    let attack_range = stats.attack_range.max(ATTACK_RANGE * 0.5);
    let in_range = dist.is_finite() && dist <= attack_range;

    // --- perception (DeterminePerception, §5) --------------------------------
    let forward = yaw_forward(agent.yaw);
    let eye = [agent.pos[0], agent.pos[1] + EYE_HEIGHT, agent.pos[2]];
    let seen = agent.senses.can_see(eye, forward, player);
    let heard = agent.senses.can_hear(dist, input.player_noise);
    agent.senses.los = seen;
    let sensed = seen || heard;

    if sensed {
        agent.target = Some(player);
        agent.timers.lost = 0.0;
        agent.alert = (agent.alert + dt * ALERT_RATE).min(1.0);
    } else {
        agent.alert = (agent.alert - dt * ALERT_DECAY).max(0.0);
    }

    // --- two-tier promotion (§0) ---------------------------------------------
    if dist.is_finite() {
        if dist < ACTIVATION_DIST {
            agent.live = true;
        } else if dist > DEACTIVATION_DIST {
            agent.live = false;
        }
    }

    // --- Screamer special: flee + scream (`ai-combat.md` §3.6) ---------------
    if agent.kind == ZombieKind::Screamer
        && agent.behavior != ZombieBehavior::Dead
        && agent.behavior != ZombieBehavior::Flee
        && sensed
        && dist < SCREAMER_FLEE_DIST
    {
        enter(agent, ZombieBehavior::Flee, rng);
    }

    // --- behaviour table selection (§6) --------------------------------------
    let mask = build_mask(agent, seen, heard, in_range);
    agent.behavior_mask = mask;
    let native = determine_behavior(mask);
    agent.native_behavior = native;

    let desired = classify(native, in_range, sensed);
    let next = resolve(agent, desired, sensed, in_range, dist);
    enter(agent, next, rng);

    // --- attack initiation (ghidra §10) --------------------------------------
    if agent.behavior == ZombieBehavior::Attack
        && sensed
        && in_range
        && agent.cooldowns.attack <= 0.0
    {
        out.attack = true;
        agent.cooldowns.attack = ATTACK_COOLDOWN;
    }

    // --- steering / speed (§7, anim-group locomotion) ------------------------
    match agent.behavior {
        ZombieBehavior::Idle
        | ZombieBehavior::Dead
        | ZombieBehavior::Grab
        | ZombieBehavior::Feasting => {}
        ZombieBehavior::Wander => {
            if let Some(w) = agent.wander_target {
                out.move_dir = horiz_dir(agent.pos, w);
            }
            out.speed = stats.walk_speed;
        }
        ZombieBehavior::Alert | ZombieBehavior::Search => {
            if let Some(t) = agent.target {
                out.move_dir = horiz_dir(agent.pos, t);
            }
            out.speed = stats.walk_speed;
        }
        ZombieBehavior::Chase => {
            if let Some(t) = agent.target {
                out.move_dir = horiz_dir(agent.pos, t);
            }
            out.speed = stats.run_speed;
        }
        ZombieBehavior::Attack => {
            // Face the target but hold position for the strike.
            if let Some(t) = agent.target {
                turn_toward(agent, horiz_dir(agent.pos, t), dt);
            }
        }
        ZombieBehavior::Flee => {
            let away = horiz_dir(agent.pos, player);
            out.move_dir = [-away[0], -away[1]];
            out.speed = stats.run_speed;
            // Scream echo (`Zombie.CreateEcho`), re-armed every interval.
            agent.timers.scream -= dt;
            if agent.timers.scream <= 0.0 {
                out.noise = COMBAT_SOUND_DIST;
                agent.timers.scream = SCREAM_INTERVAL;
            }
        }
        _ => {}
    }

    // --- movement noise tiers (`zombies.xml` sound radii) --------------------
    if out.noise <= 0.0 {
        if out.speed >= stats.run_speed && out.speed > 0.0 {
            out.noise = SOUND_RUN;
        } else if out.speed > 0.0 {
            out.noise = SOUND_WALK;
        }
    }

    // Face the movement direction (Attack handled above).
    if agent.behavior != ZombieBehavior::Attack {
        turn_toward(agent, out.move_dir, dt);
    }

    // --- NaN / non-finite guards ---------------------------------------------
    if !out.speed.is_finite() {
        out.speed = 0.0;
    }
    if !out.move_dir[0].is_finite() || !out.move_dir[1].is_finite() {
        out.move_dir = [0.0, 0.0];
    }
    if !out.noise.is_finite() {
        out.noise = 0.0;
    }

    out.behavior = agent.behavior;
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn input(player: [f32; 3], noise: f32, dt: f32) -> ZombieInput {
        ZombieInput { player_pos: player, player_noise: noise, dt, rng: 0.5 }
    }

    #[test]
    fn perception_sees_in_fov_not_behind() {
        let mut s = ZombieSenses::new(20.0, 60.0, 8.0);
        let eye = [0.0, 0.0, 0.0];
        let fwd = [0.0, 0.0, 1.0];
        assert!(
            s.determine_perception(eye, fwd, [0.0, 0.0, 10.0], 0.0),
            "in front and in range"
        );
        assert!(s.los, "LOS flag set for a visible target");
        assert!(
            !s.determine_perception(eye, fwd, [0.0, 0.0, -10.0], 0.0),
            "directly behind is outside the cone"
        );
        assert!(!s.los);
        assert!(
            !s.determine_perception(eye, fwd, [0.0, 0.0, 25.0], 0.0),
            "beyond sight range"
        );
        // Directly behind but inside CLOSE_RANGE -> still detected.
        assert!(s.determine_perception(eye, fwd, [0.0, 0.0, -1.5], 0.0));
    }

    #[test]
    fn hearing_scales_with_noise() {
        let s = ZombieSenses::new(15.0, 60.0, 8.0);
        assert!(s.can_hear(20.0, 3.0), "20 m within 8*3 m");
        assert!(!s.can_hear(20.0, 1.0), "20 m beyond 8 m");
        assert!(!s.can_hear(1.0, 0.0), "silence is never heard");
    }

    #[test]
    fn behavior_table_matches_native_registration_order() {
        // Names and order are the recovered `BehaviorsSet` table (`0131cb00`),
        // verified against the binary: the inlined `Vehicle` row sits at index 6
        // (between `VehicleTimerExp` and `?01840f88`).
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE.len(), 35);
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[0].0, "BarricadeWatch");
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[1].0, "Barricade");
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[2].0, "CryAi");
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[5].0, "VehicleTimerExp");
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[6].0, "Vehicle");
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[7].0, "?01840f88");
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[22].0, "TargetSeen");
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[23].0, "TargetNear");
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[33].0, "Wander");
        assert_eq!(ZOMBIE_BEHAVIOR_TABLE[34].0, "?01771cf0");
    }

    #[test]
    fn behavior_priority_prefers_earlier_table_rows() {
        // CryAi (row 2) outranks TargetNear (row 22) and TargetSeen (row 21).
        assert_eq!(
            determine_behavior(PERCEPT_CRYAI | PERCEPT_FOCUS | PERCEPT_LOS),
            ZombieBehavior::CryAi
        );
        // TargetNear wins over TrackPlayer once in range.
        assert_eq!(
            determine_behavior(PERCEPT_ATTENTION | PERCEPT_AWARE | PERCEPT_LOS | PERCEPT_FOCUS),
            ZombieBehavior::TargetNear
        );
        // Seen at medium range -> TrackPlayer.
        assert_eq!(
            determine_behavior(PERCEPT_ATTENTION | PERCEPT_AWARE | PERCEPT_LOS),
            ZombieBehavior::TrackPlayer
        );
        // Heard only -> TargetSeen.
        assert_eq!(
            determine_behavior(PERCEPT_ATTENTION | PERCEPT_HEARD),
            ZombieBehavior::TargetSeen
        );
        // Nothing sensed -> the default row.
        assert_eq!(determine_behavior(0), ZombieBehavior::Default);
    }

    #[test]
    fn chasing_zombie_in_range_attacks() {
        let mut a = ZombieAgent::new(ZombieKind::Grunt, [0.0, 0.0, 0.0]);
        a.behavior = ZombieBehavior::Chase;
        let out = update(&mut a, &input([0.0, 0.0, 1.0], 0.0, 0.05));
        assert_eq!(out.behavior, ZombieBehavior::Attack);
        assert!(out.attack, "a strike is initiated within attack range");
        // At 1 m the zombie is inside ActivationDist, so the native selection
        // is the CryAi hand-off; TargetNear is the Lab-AI in-range row.
        assert!(
            matches!(
                a.native_behavior,
                ZombieBehavior::CryAi | ZombieBehavior::TargetNear
            ),
            "in-range selection is TargetNear (Lab) or CryAi (promoted)"
        );
    }

    #[test]
    fn loses_sight_enters_search_then_idle() {
        let mut a = ZombieAgent::new(ZombieKind::Grunt, [0.0, 0.0, 0.0]);
        a.behavior = ZombieBehavior::Chase;
        a.target = Some([0.0, 0.0, 0.0]);
        let far = input([0.0, 0.0, 100.0], 0.0, 0.1);

        let mut entered_search = false;
        for _ in 0..20 {
            if update(&mut a, &far).behavior == ZombieBehavior::Search {
                entered_search = true;
                break;
            }
        }
        assert!(entered_search, "expected Search after losing the target");

        let mut returned_idle = false;
        for _ in 0..120 {
            if update(&mut a, &far).behavior == ZombieBehavior::Idle {
                returned_idle = true;
                break;
            }
        }
        assert!(returned_idle, "Search should time out into Idle");
    }

    #[test]
    fn screamer_flees_and_screams() {
        let mut a = ZombieAgent::new(ZombieKind::Screamer, [0.0, 0.0, 0.0]);
        a.behavior = ZombieBehavior::Alert;
        let out = update(&mut a, &input([0.0, 0.0, 3.0], 0.0, 0.1));
        assert_eq!(out.behavior, ZombieBehavior::Flee);
        assert!(out.noise > 0.0, "screamer emits alert noise");
        assert!(out.move_dir[1] < 0.0, "flees away from a +z player");
    }

    #[test]
    fn juggernaut_is_slower_than_grunt() {
        let mut grunt = ZombieAgent::new(ZombieKind::Grunt, [0.0, 0.0, 0.0]);
        grunt.behavior = ZombieBehavior::Chase;
        let mut jug = ZombieAgent::new(ZombieKind::Juggernaut, [0.0, 0.0, 0.0]);
        jug.behavior = ZombieBehavior::Chase;
        let inp = input([0.0, 0.0, 5.0], 0.0, 0.05);

        let g = update(&mut grunt, &inp);
        let j = update(&mut jug, &inp);
        assert_eq!(g.behavior, ZombieBehavior::Chase);
        assert_eq!(j.behavior, ZombieBehavior::Chase);
        assert_eq!(g.speed, ZombieKind::Grunt.stats().run_speed);
        assert_eq!(j.speed, ZombieKind::Juggernaut.stats().run_speed);
        assert!(g.speed > j.speed, "juggernaut must be slower than a grunt");
    }

    #[test]
    fn idle_builds_alert_then_chases() {
        let mut a = ZombieAgent::new(ZombieKind::Grunt, [0.0, 0.0, 0.0]);
        let inp = input([0.0, 0.0, 8.0], 0.0, 0.1);
        assert_eq!(update(&mut a, &inp).behavior, ZombieBehavior::Alert);
        let mut reached = ZombieBehavior::Alert;
        for _ in 0..20 {
            reached = update(&mut a, &inp).behavior;
            if reached == ZombieBehavior::Chase {
                break;
            }
        }
        assert_eq!(reached, ZombieBehavior::Chase);
    }

    #[test]
    fn horde_spawn_scales_with_noise_and_respects_caps() {
        let mut rng = SmallRng::new(0xDEAD_BEEF);

        let mut low_dir = HordeDirector::new();
        let low = low_dir.on_noise(1.0, &mut rng).expect("low noise still spawns");
        let mut high_dir = HordeDirector::new();
        let high = high_dir.on_noise(100.0, &mut rng).expect("high noise spawns");
        assert!(high.count() > low.count(), "louder noise -> bigger spawn");
        assert!(high.count() <= HORDE_MAX_SIZE, "never exceeds Horde.MaxSize");

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
    fn horde_tick_refills_ambient_and_horde() {
        let mut d = HordeDirector::new();
        d.tick(1000.0);
        assert_eq!(d.ambient, d.ambient_cap, "ambient refilled to cap");
        assert!(d.horde >= 1, "horde refilled");
        assert!(d.logical >= d.ambient + d.horde);
        d.tick(1000.0);
        assert_eq!(d.ambient, d.ambient_cap, "never exceeds the ambient cap");
    }

    #[test]
    fn director_is_deterministic_with_fixed_seed() {
        let mut a = HordeDirector::new();
        let mut b = HordeDirector::new();
        let mut ra = SmallRng::new(42);
        let mut rb = SmallRng::new(42);
        for i in 0..32 {
            let sa = a.on_noise(10.0 + i as f32, &mut ra);
            let sb = b.on_noise(10.0 + i as f32, &mut rb);
            assert_eq!(sa, sb, "same seed -> identical spawns");
        }
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

    #[test]
    fn nan_input_is_total() {
        let mut a = ZombieAgent::new(ZombieKind::Grunt, [f32::NAN; 3]);
        let inp = ZombieInput {
            player_pos: [f32::NAN; 3],
            player_noise: f32::NAN,
            dt: f32::NAN,
            rng: f32::NAN,
        };
        let out = update(&mut a, &inp);
        assert!(out.speed.is_finite(), "speed sanitised");
        assert!(out.move_dir[0].is_finite() && out.move_dir[1].is_finite());
        assert!(out.noise.is_finite());
        // Negative dt must not run time backwards.
        let neg = ZombieInput {
            player_pos: [0.0, 0.0, 0.0],
            player_noise: 0.0,
            dt: -5.0,
            rng: 0.5,
        };
        let out2 = update(&mut a, &neg);
        assert!(out2.speed.is_finite());
    }

    #[test]
    fn two_tier_promotion_tracks_player_distance() {
        let mut a = ZombieAgent::new(ZombieKind::Grunt, [0.0, 0.0, 0.0]);
        a.behavior = ZombieBehavior::Chase;
        // Near player -> promoted to CryAI.
        update(&mut a, &input([0.0, 0.0, 5.0], 0.0, 0.05));
        assert!(a.live, "promoted inside ActivationDist");
        // Far player -> demoted back to Lab AI.
        update(&mut a, &input([0.0, 0.0, 100.0], 0.0, 0.05));
        assert!(!a.live, "demoted beyond DeactivationDist");
    }
}
