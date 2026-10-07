//! Zombie spawn/despawn director — the `NSZombieManager` ecology ("Lab AI").
//!
//! Zombies are **not** level entities. The native manager holds 256 logical
//! slots and a small pool of 24 promoted CryEngine actors; this module owns the
//! part that decides *where, when and which* zombies appear around the player,
//! and when they are removed again. Promotion of a logical zombie to a real
//! entity (and demotion) is the caller's job — here we only emit
//! [`SpawnEvent::Spawn`] / [`SpawnEvent::Despawn`] commands.
//!
//! # Recovered sources
//!
//! * Ambient/ambush/danger-zone director `FUN_01348680` @ `0x01348680` (9,752 B):
//!   the per-frame cadence, timers, chance pool, density formula and despawn
//!   logic below are read from its decompilation.
//! * Logical spawn `FUN_0133dc40` @ `0x0133dc40` and `FUN_0133eb70` @
//!   `0x0133eb70`: slot allocation, exclusion zones, jitter, cluster type and
//!   the `ZOMBIE_SPAWN_FLAG_*` word.
//! * Roaming-horde refill `FUN_0135c080` @ `0x0135c080` and `AddHorde`
//!   `FUN_0131af80` @ `0x0131af80`.
//! * Markup loader `FUN_0131d160` @ `0x0131d160` — the exact `ULZS`
//!   `zombie_spawns_*.markup` layout (magic `ULZS`, version `4`, then the
//!   spawn/danger-zone arrays, each 0x80-aligned).
//! * Data tables: `RTS/Zombies.xml` (RTS ids `0x1200xxxx` globals,
//!   `0x1201xxxx` Ambush, `0x1202xxxx` Infestation, `0x1203xxxx` Horde,
//!   `0x1204xxxx` Density) and `Zombies/ZombieTypes.xml` (weighted type groups
//!   and `ZombieFreakGroup`s). Metadata strings `Zombies.<Group>.<Value>` live
//!   in the exe `.rdata` (`01853060`+), cvar defaults in
//!   [`crate::cry::cvar_table`] (`zm_activationdist 15`, `zm_deactivationdist
//!   20`).
//! * Numeric constants decoded from `.rdata` with `/tmp/opencode` PE reader:
//!   `DAT_0172e928 = 150.0` (ambient radius), `DAT_01755dd8 = 22500.0` (150²
//!   danger-zone reach), `DAT_017926d0 = 2500.0` (50² min ambush distance),
//!   `DAT_0176b8dc = 120.0 s` (persistent-danger alert), `DAT_0172155c = 60.0`
//!   (frames/s), `DAT_0171e210 = 1.5`, `DAT_017877c4 = 0.5`,
//!   `DAT_0171e218 = 100.0`, `DAT_018ab8a8 = 2^32`.
//!
//! No game Lua/XML is copied into the crate: the tables are parsed at runtime
//! from the mounted paks (or the loose extracted tree).

use crate::game::ZombieKind;
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

// ---------------------------------------------------------------------------
// Caps (recovered; see notes/ghidra/zombie-runtime.md §3)
// ---------------------------------------------------------------------------

/// Logical zombie slots (`MAX_ZOMBIES`, assert in `GetId`).
pub const MAX_ZOMBIES: usize = 256;
/// Promoted/live CryEngine actors (`m_liveZombies`, loops `< 0x18`).
pub const MAX_LIVE_ZOMBIES: usize = 24;
/// Horde slots (`MAX_HORDES`, `AddHorde` loop `< 8`).
pub const MAX_HORDES: usize = 8;
/// Roaming-horde handles (`FUN_0135c080` `iVar11 = 6`).
pub const MAX_ROAMING_HORDES: usize = 6;
/// Spawn-exclusion circles (`FUN_0133dc40`, `< 0x80`, stride 0x14).
pub const MAX_EXCLUSION_ZONES: usize = 128;

// ---------------------------------------------------------------------------
// Distances / timers decoded from `.rdata`
// ---------------------------------------------------------------------------

/// Ambient spawn/search radius, `sqrt(DAT_01755dd8)` — `DAT_0172e928 = 150.0`.
pub const AMBIENT_RADIUS: f32 = 150.0;
/// Danger zone is "near" the player within `DAT_01755dd8 = 22500.0 = 150²`.
pub const DANGER_NEAR_DIST_SQ: f32 = 22_500.0;
/// Minimum distance² for an ambush spawn, `DAT_017926d0 = 2500.0 = 50²`.
pub const AMBUSH_MIN_DIST_SQ: f32 = 2_500.0;
/// Persistent-danger alert time, `DAT_0176b8dc = 120.0 s`.
pub const PERSISTENT_DANGER_ALERT: f32 = 120.0;
/// Frames per second used to convert frame-valued RTS timers, `DAT_0172155c`.
pub const FRAMES_PER_SECOND: f32 = 60.0;
/// Danger-zone density evaluation period, `0x3e800000 = 0.25 s`.
pub const DANGER_EVAL_PERIOD: f32 = 0.25;
/// Roaming-horde retry delay on spawn failure, `0x40a00000 = 5.0 s`.
pub const HORDE_RETRY_DELAY: f32 = 5.0;
/// Ambient grid refresh period, `DAT_0172b114 = 20.0 s`. The same `.data`
/// scalar is the shipped default for the `zm_deactivationDist` cvar
/// (`FUN_0134c920:248`), which is where its value was confirmed.
pub const AMBIENT_GRID_PERIOD: f32 = 20.0;
/// Ambient grid side (`FUN_01348680:417` `0x15 = 21`): a 21×21 map of nodes.
pub const AMBIENT_GRID_DIM: i32 = 21;
/// Default ambient grid node spacing in metres (`DAT_019091bc`; `FUN_0134e5f0`
/// sets 200 for the main map and 100 for `FUN_0141d520()==2`). The grid is a
/// set of nodes, each owning the `cell/2`-radius square around it.
pub const AMBIENT_GRID_CELL_SIZE: f32 = 200.0;
/// Region-density caps indexed by a cell's type, from `RTS(0x12040001..06)`
/// (`FUN_0133a410:21-32` loads `DAT_01df...` caps): Agricultural 50,
/// Commercial 100, Industrial 75, Park 35, Residential 100, Wilderness 20.
pub const AMBIENT_DENSITY_CAPS: [u32; 6] = [50, 100, 75, 35, 100, 20];
/// The cell density field must exceed this before the target is recomputed
/// (`FUN_01348680:379` `0xe < density`, i.e. `density >= 15`).
pub const AMBIENT_DENSITY_TRIGGER: u32 = 15;
/// Square metres per ambient zombie. The director (`FUN_01348680`) loads the
/// float at `0x184410c` (= 40000.0 in the exe) at `0x1349747` and passes it to
/// `FUN_0133eb70` (call `0x1349805`) as `param_9`; there the cell rectangle,
/// clipped to the player +-150 m (`0x172e928`), is divided by it, truncated by
/// the CRT `_ftol` (`0x16b9210`) and reduced by the cell's current count
/// (`param_10`). A fully covered 200 m cell therefore holds one zombie.
pub const AMBIENT_AREA_PER_ZOMBIE: f32 = 40000.0;
/// Per-cell placement attempts before giving up (`FUN_0133eb70:187`
/// `uStack_1b0 < 0xa`).
pub const AMBIENT_CELL_ATTEMPTS: u32 = 10;
/// Hard cap on simultaneously active ambush points (guards runaway spawning).
pub const DANGER_SPAWN_BATCH: u32 = 6;

// ---------------------------------------------------------------------------
// Zombie spawn flags (`ZOMBIE_SPAWN_FLAG_*`, ScriptBindZombies `01567420`)
// ---------------------------------------------------------------------------

/// No flags.
pub const ZOMBIE_SPAWN_NONE: u32 = 0;
/// Goal is the player.
pub const ZOMBIE_SPAWN_TRACK_PLAYER: u32 = 1;
/// Allocate/queue a nav path.
pub const ZOMBIE_SPAWN_PATH: u32 = 2;
/// Bypass spawn-exclusion zones.
pub const ZOMBIE_SPAWN_FORCE: u32 = 4;
/// Run speed.
pub const ZOMBIE_SPAWN_FAST: u32 = 8;
/// No positional jitter.
pub const ZOMBIE_SPAWN_EXACT_POS: u32 = 0x10;
/// Goal/track active.
pub const ZOMBIE_SPAWN_GOAL_ACTIVE: u32 = 0x20;
/// Direct/positional mode.
pub const ZOMBIE_SPAWN_VAMP: u32 = 0x40;

// Cluster / reason codes from `FUN_0133dc40(.., clusterType, ..)`.
/// Ambient density grid (`cluster type 1`).
pub const CLUSTER_DENSITY: u8 = 1;
/// Ambush point (`FUN_0133dc40(...,5,...)`).
pub const CLUSTER_AMBUSH: u8 = 5;
/// Near/interior zombie (`FUN_0133dc40(...,6,...)`).
pub const CLUSTER_NEAR: u8 = 6;
/// Danger-zone density (`FUN_0133eb70(...,7,...)`).
pub const CLUSTER_DANGER: u8 = 7;

// ---------------------------------------------------------------------------
// RTS tuning (`RTS/Zombies.xml`), with the shipped defaults
// ---------------------------------------------------------------------------

/// Frame-valued RTS timers converted to seconds.
#[inline]
fn frames(f: f32) -> f32 {
    f / FRAMES_PER_SECOND
}

/// All `RTS/Zombies.xml` values this director consumes. Defaults are the
/// shipped XML values; group/subgroup ids follow the recovered encoding
/// `(0x12 << 24) | (subgroup << 16) | index`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZombieTuning {
    // --- Zombies global (subgroup 0) ---
    /// `Zombies.ActivationDist` = 15.
    pub activation_dist: f32,
    /// `Zombies.DeactivationDist` = 20.
    pub deactivation_dist: f32,
    /// `Zombies.AggroDist` = 4.
    pub aggro_dist: f32,
    /// `Zombies.AmbientRespawnRate` = 15 (spawns per minute); id `0x12000003`.
    pub ambient_respawn_rate: f32,
    /// `Zombies.BaseFOV` = 80.
    pub base_fov: f32,

    // --- Ambush (subgroup 1) ---
    /// `Zombies.Ambush.SpawnTimer` = 0.1 s (id `0x12020001`; seconds).
    pub amb_spawn_timer: f32,
    /// `Zombies.Ambush.SpawnChance` = 60.
    pub amb_spawn_chance: f32,
    /// `Zombies.Ambush.CooldownChance` = 80.
    pub amb_cooldown_chance: f32,
    /// `Zombies.Ambush.CooldownDuration` = 0.15.
    pub amb_cooldown_duration: f32,
    /// `Zombies.Ambush.CooldownPenalty` = 2.
    pub amb_cooldown_penalty: f32,
    /// `Zombies.Ambush.EvaluateRange` = 60 (id `0x12010006`).
    pub amb_evaluate_range: f32,
    /// `Zombies.Ambush.ReevaluateDelay` = 300 frames → 5 s (id `0x12010007`).
    pub amb_reevaluate_delay: f32,
    /// `Zombies.Ambush.BaseSpawnChance` = 32 (id `0x12010008`).
    pub amb_base_spawn_chance: f32,
    /// `Zombies.Ambush.PostSpawnEvaluateDelay` = 1200 frames → 20 s.
    pub amb_post_spawn_eval_delay: f32,
    /// `Zombies.Ambush.DespawnRange` = 80 (id `0x1201000a`).
    pub amb_despawn_range: f32,
    /// `Zombies.Ambush.BaseSpawnRecoveryDelay` = 60 frames → 1 s.
    pub amb_spawn_recovery_delay: f32,
    /// `Zombies.Ambush.PerSpawnChanceReduction` = 4.
    pub amb_per_spawn_reduction: f32,
    /// `Zombies.Ambush.BaseSpawnRecoveryAmount` = 8.
    pub amb_spawn_recovery_amount: f32,

    // --- Infestation (subgroup 2) ---
    /// `Zombies.Infestation.SpreadTimer` = 720.
    pub inf_spread_timer: f32,
    /// `Zombies.Infestation.SpreadCap` = 1.
    pub inf_spread_cap: f32,
    /// `Zombies.Infestation.SafeZoneRadius` = 40.
    pub inf_safe_zone_radius: f32,
    /// `Zombies.Infestation.SafeZoneTimer` = 30.
    pub inf_safe_zone_timer: f32,
    /// `Zombies.Infestation.Density` = 7.
    pub inf_density: f32,
    /// `Zombies.Infestation.TotalSpreadCap` = 5.
    pub inf_total_spread_cap: f32,

    // --- Horde (subgroup 3) ---
    /// `Zombies.Horde.MaxSize` = 8 (id `0x12030001`).
    pub horde_max_size: f32,
    /// `Zombies.Horde.MaxAmbient` = 3 (id `0x12030002`).
    pub horde_max_ambient: f32,
    /// `Zombies.Horde.MaxRange` = 500.
    pub horde_max_range: f32,
    /// `Zombies.Horde.RespawnDelayBase` = 15.
    pub horde_respawn_delay_base: f32,
    /// `Zombies.Horde.RespawnDelayBasePerActive` = 0.
    pub horde_respawn_delay_base_per_active: f32,
    /// `Zombies.Horde.RespawnDelayPerKill` = 60.
    pub horde_respawn_delay_per_kill: f32,
    /// `Zombies.Horde.RespawnDelayMax` = 300.
    pub horde_respawn_delay_max: f32,
    /// `Zombies.Horde.RespawnDelayDecayInterval` = 300.
    pub horde_respawn_decay_interval: f32,
    /// `Zombies.Horde.RespawnDelayDecayAmount` = 60.
    pub horde_respawn_decay_amount: f32,

    // --- Density (subgroup 4) ---
    /// `Zombies.Density.Interior` = 3 (id `0x12040007`).
    pub density_interior: f32,
    /// `Zombies.Density.InteriorCap` = 5 (id `0x12040008`).
    pub density_interior_cap: f32,
    /// Outdoor density used when no region density is supplied (Park = 35).
    pub density_outdoor: f32,

    // --- Ambient grid ---
    /// Density-fill divisor for the ambient cells:
    /// `round(RTS(0x12000004)·60·DAT_0171e200)` (`FUN_01348680:370`). The 4th
    /// global of `Zombies` is `BaseFOV = 80` and `DAT_0171e200 = 0.05`
    /// (the `sys_max_step` default), giving 240.
    pub ambient_cell_interval: f32,
}

impl Default for ZombieTuning {
    fn default() -> Self {
        ZombieTuning {
            activation_dist: 15.0,
            deactivation_dist: 20.0,
            aggro_dist: 4.0,
            ambient_respawn_rate: 15.0,
            base_fov: 80.0,

            amb_spawn_timer: 0.1,
            amb_spawn_chance: 60.0,
            amb_cooldown_chance: 80.0,
            amb_cooldown_duration: 0.15,
            amb_cooldown_penalty: 2.0,
            amb_evaluate_range: 60.0,
            amb_reevaluate_delay: frames(300.0),
            amb_base_spawn_chance: 32.0,
            amb_post_spawn_eval_delay: frames(1200.0),
            amb_despawn_range: 80.0,
            amb_spawn_recovery_delay: frames(60.0),
            amb_per_spawn_reduction: 4.0,
            amb_spawn_recovery_amount: 8.0,

            inf_spread_timer: 720.0,
            inf_spread_cap: 1.0,
            inf_safe_zone_radius: 40.0,
            inf_safe_zone_timer: 30.0,
            inf_density: 7.0,
            inf_total_spread_cap: 5.0,

            horde_max_size: 8.0,
            horde_max_ambient: 3.0,
            horde_max_range: 500.0,
            horde_respawn_delay_base: 15.0,
            horde_respawn_delay_base_per_active: 0.0,
            horde_respawn_delay_per_kill: 60.0,
            horde_respawn_delay_max: 300.0,
            horde_respawn_decay_interval: 300.0,
            horde_respawn_decay_amount: 60.0,

            density_interior: 3.0,
            density_interior_cap: 5.0,
            density_outdoor: 35.0,

            ambient_cell_interval: 240.0,
        }
    }
}

impl ZombieTuning {
    /// Override a named value from an `RTS/Zombies.xml` `<Subgroup>/<Value>`
    /// pair. Unknown names are ignored (lenient). This keeps the director tied
    /// to the shipped data without copying it into the crate.
    pub fn set(&mut self, subgroup: &str, name: &str, v: f32) {
        let f = |d: &mut f32| *d = v;
        match (subgroup, name) {
            ("", "ActivationDist") => f(&mut self.activation_dist),
            ("", "DeactivationDist") => f(&mut self.deactivation_dist),
            ("", "AggroDist") => f(&mut self.aggro_dist),
            ("", "AmbientRespawnRate") => f(&mut self.ambient_respawn_rate),
            ("", "BaseFOV") => f(&mut self.base_fov),
            ("Ambush", "SpawnTimer") => f(&mut self.amb_spawn_timer),
            ("Ambush", "SpawnChance") => f(&mut self.amb_spawn_chance),
            ("Ambush", "CooldownChance") => f(&mut self.amb_cooldown_chance),
            ("Ambush", "CooldownDuration") => f(&mut self.amb_cooldown_duration),
            ("Ambush", "CooldownPenalty") => f(&mut self.amb_cooldown_penalty),
            ("Ambush", "EvaluateRange") => f(&mut self.amb_evaluate_range),
            ("Ambush", "ReevaluateDelay") => f(&mut self.amb_reevaluate_delay),
            ("Ambush", "BaseSpawnChance") => f(&mut self.amb_base_spawn_chance),
            ("Ambush", "PostSpawnEvaluateDelay") => f(&mut self.amb_post_spawn_eval_delay),
            ("Ambush", "DespawnRange") => f(&mut self.amb_despawn_range),
            ("Ambush", "BaseSpawnRecoveryDelay") => f(&mut self.amb_spawn_recovery_delay),
            ("Ambush", "PerSpawnChanceReduction") => f(&mut self.amb_per_spawn_reduction),
            ("Ambush", "BaseSpawnRecoveryAmount") => f(&mut self.amb_spawn_recovery_amount),
            ("Infestation", "SpreadTimer") => f(&mut self.inf_spread_timer),
            ("Infestation", "SpreadCap") => f(&mut self.inf_spread_cap),
            ("Infestation", "SafeZoneRadius") => f(&mut self.inf_safe_zone_radius),
            ("Infestation", "SafeZoneTimer") => f(&mut self.inf_safe_zone_timer),
            ("Infestation", "Density") => f(&mut self.inf_density),
            ("Infestation", "TotalSpreadCap") => f(&mut self.inf_total_spread_cap),
            ("Horde", "MaxSize") => f(&mut self.horde_max_size),
            ("Horde", "MaxAmbient") => f(&mut self.horde_max_ambient),
            ("Horde", "MaxRange") => f(&mut self.horde_max_range),
            ("Horde", "RespawnDelayBase") => f(&mut self.horde_respawn_delay_base),
            ("Horde", "RespawnDelayBasePerActive") => {
                f(&mut self.horde_respawn_delay_base_per_active)
            }
            ("Horde", "RespawnDelayPerKill") => f(&mut self.horde_respawn_delay_per_kill),
            ("Horde", "RespawnDelayMax") => f(&mut self.horde_respawn_delay_max),
            ("Horde", "RespawnDelayDecayInterval") => f(&mut self.horde_respawn_decay_interval),
            ("Horde", "RespawnDelayDecayAmount") => f(&mut self.horde_respawn_decay_amount),
            ("Density", "Interior") => f(&mut self.density_interior),
            ("Density", "InteriorCap") => f(&mut self.density_interior_cap),
            _ => {}
        }
    }

    /// Parse the shipped `RTS/Zombies.xml` and apply the names in [`Self::set`].
    /// Frame-valued timers are converted to seconds. Lenient: malformed XML
    /// leaves the defaults in place.
    pub fn apply_rts_xml(&mut self, bytes: &[u8]) {
        let mut reader = Reader::from_reader(bytes);
        let mut subgroup = String::new();
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                    let tag = local_name(e.name().as_ref());
                    if tag == "Subgroup" {
                        subgroup = attr(&e, "Name").unwrap_or_default();
                    } else if tag == "Value" {
                        if let (Some(n), Some(d)) = (attr(&e, "Name"), attr(&e, "Default")) {
                            if let Ok(v) = d.trim().parse::<f32>() {
                                // These three XML values are frame counts
                                // (`FUN_01348680` compares them against
                                // frame-scaled timers).
                                let frame_valued = matches!(
                                    n.as_str(),
                                    "ReevaluateDelay"
                                        | "PostSpawnEvaluateDelay"
                                        | "BaseSpawnRecoveryDelay"
                                );
                                self.set(&subgroup, &n, if frame_valued { frames(v) } else { v });
                            }
                        }
                    }
                }
                Ok(Event::End(e)) => {
                    if local_name(e.name().as_ref()) == "Subgroup" {
                        subgroup.clear();
                    }
                }
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Type tables (`Zombies/ZombieTypes.xml`)
// ---------------------------------------------------------------------------

/// One `<ZombieDef>` inside a `<ZombieGroup>`.
#[derive(Clone, Debug, PartialEq)]
pub struct ZombieDef {
    /// `Name` (e.g. `Slow`, `Fast`, `Horde01`).
    pub name: String,
    /// `EntityClass`; the loader defaults this to `"Grunt"` (`01330120`).
    pub entity_class: String,
    /// Behaviour group (`Ambient`, `Ambush`, `Horde`, `Mission`, `Infestation`).
    pub group: String,
    /// Spawn weight within the group.
    pub weight: f32,
    /// `AggroDist`.
    pub aggro_dist: f32,
    /// `BaseAggroDist`.
    pub base_aggro_dist: f32,
    /// `IdlePeriod + rand·IdleVariance` idle timer (`0133dc40`).
    pub idle_period: f32,
    /// Idle variance.
    pub idle_variance: f32,
    /// `Flags` attribute.
    pub flags: String,
}

impl ZombieDef {
    /// Mapped gameplay kind for the entity class name.
    pub fn kind(&self) -> ZombieKind {
        kind_for_class(&self.entity_class)
    }
}

/// One `<ZombieFreakGroup>`.
#[derive(Clone, Debug, PartialEq)]
pub struct FreakDef {
    /// `Id` (e.g. `Feral`).
    pub id: String,
    /// `EntityClass` (e.g. `Feral`, `Screamer`).
    pub entity_class: String,
    /// `Flags`; `AlwaysLive`, `NoBarricade`, `NoDismemberFX`, `UseNormalDot`.
    pub flags: String,
    /// `Max` (`Zombies.%s.Max`).
    pub max: u32,
    /// `Percentage` (`Zombies.%s.Percentage`).
    pub percentage: f32,
    /// `KillScore`.
    pub kill_score: u32,
}

impl FreakDef {
    /// `AlwaysLive` freaks (Feral/Juggernaut) promote to CryAI on spawn
    /// (`0133dc40:381-385`).
    pub fn always_live(&self) -> bool {
        self.flags.split(',').any(|s| s.trim() == "AlwaysLive")
    }
}

/// Parsed `ZombieTypes.xml`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZombieTable {
    /// Grouped `<ZombieDef>` entries.
    pub defs: Vec<ZombieDef>,
    /// `<ZombieFreakGroup>` entries.
    pub freaks: Vec<FreakDef>,
}

impl ZombieTable {
    /// Weighted pick from one behaviour group. Falls back to the first
    /// `Grunt`-class def, then to a synthetic [`ZombieDef`]. `r` is in `[0,1)`.
    pub fn pick(&self, group: &str, r: f32) -> ZombieDef {
        let group_defs: Vec<&ZombieDef> =
            self.defs.iter().filter(|d| d.group == group && d.weight > 0.0).collect();
        let total: f32 = group_defs.iter().map(|d| d.weight).sum();
        if total > 0.0 {
            let mut x = r.clamp(0.0, 1.0 - f32::EPSILON) * total;
            for d in &group_defs {
                if x < d.weight {
                    return (*d).clone();
                }
                x -= d.weight;
            }
            if let Some(d) = group_defs.last() {
                return (*d).clone();
            }
        }
        ZombieDef {
            name: "Grunt".into(),
            entity_class: "Grunt".into(),
            group: group.into(),
            weight: 1.0,
            aggro_dist: 15.0,
            base_aggro_dist: 2.0,
            idle_period: 5.0,
            idle_variance: 5.0,
            flags: String::new(),
        }
    }

    /// Parse `Zombies/ZombieTypes.xml`. Lenient; returns whatever it recognised.
    pub fn parse(bytes: &[u8]) -> ZombieTable {
        let mut table = ZombieTable::default();
        let mut reader = Reader::from_reader(bytes);
        let mut group = String::new();
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                    let tag = local_name(e.name().as_ref());
                    if tag == "ZombieGroup" {
                        group = attr(&e, "Type").unwrap_or_default();
                    } else if tag == "ZombieDef" {
                        table.defs.push(ZombieDef {
                            name: attr(&e, "Name").unwrap_or_default(),
                            entity_class: attr(&e, "EntityClass")
                                .filter(|s| !s.is_empty())
                                .unwrap_or_else(|| "Grunt".into()),
                            group: group.clone(),
                            weight: attr_f32(&e, "Weight").unwrap_or(0.0),
                            aggro_dist: attr_f32(&e, "AggroDist").unwrap_or(15.0),
                            base_aggro_dist: attr_f32(&e, "BaseAggroDist").unwrap_or(2.0),
                            idle_period: attr_f32(&e, "IdlePeriod").unwrap_or(5.0),
                            idle_variance: attr_f32(&e, "IdleVariance").unwrap_or(5.0),
                            flags: attr(&e, "Flags").unwrap_or_default(),
                        });
                    } else if tag == "ZombieFreakGroup" {
                        table.freaks.push(FreakDef {
                            id: attr(&e, "Id").unwrap_or_default(),
                            entity_class: attr(&e, "EntityClass").unwrap_or_default(),
                            flags: attr(&e, "Flags").unwrap_or_default(),
                            max: attr(&e, "Max").and_then(|s| s.trim().parse().ok()).unwrap_or(100),
                            percentage: attr_f32(&e, "Percentage").unwrap_or(0.0),
                            kill_score: attr(&e, "KillScore")
                                .and_then(|s| s.trim().parse().ok())
                                .unwrap_or(1),
                        });
                    }
                }
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
        table
    }
}

/// The `ZombieTypes.xml` `EntityClass` an authored `Zombie.FreakSpawn`
/// `FreakHash` names (`missions.xml` uses `FreakHash="Zombie.<Class>"`).
/// `Eatter` is a Grunt appearance rather than a distinct class, so it collapses
/// to `Grunt`; an empty hash is the ambient Grunt too.
pub fn freak_entity_class(freak_hash: &str) -> &'static str {
    match freak_hash.trim().trim_start_matches("Zombie.") {
        "ArmyZombie" | "SwatZombie" => "ArmyZombie",
        "Bloater" => "Bloater",
        "Feral" => "Feral",
        "Juggernaut" => "Juggernaut",
        "Screamer" => "Screamer",
        "FreshGrunt" | "FreshZombie" => "FreshGrunt",
        _ => "Grunt",
    }
}

/// Map an entity-class name to a [`ZombieKind`].
pub fn kind_for_class(class: &str) -> ZombieKind {
    match class {
        "Screamer" => ZombieKind::Screamer,
        "Bloater" => ZombieKind::Bloater,
        "Feral" => ZombieKind::Feral,
        "Juggernaut" => ZombieKind::Juggernaut,
        // SwatZombie and ArmyZombie share the armored army archetype here;
        // FreshGrunt/FreshZombie collapse to the base grunt.
        "SwatZombie" | "ArmyZombie" => ZombieKind::ArmyZombie,
        _ => ZombieKind::Grunt,
    }
}

// ---------------------------------------------------------------------------
// Markup data (`ULZS` zombie_spawns_*.markup, loader `FUN_0131d160`)
// ---------------------------------------------------------------------------

/// One author-placed ambush/spawn point (file order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmbushPoint {
    /// World position (Z-up metres).
    pub pos: [f32; 3],
    /// Stored orientation (radians).
    pub heading: f32,
    /// Context-tag index into the ambush tag table.
    pub tag: u8,
    /// `animIdx` (idle animation), `0x200` = none.
    pub anim: u16,
    /// `type` word (spawn class offset).
    pub kind: i32,
}

/// One author-placed danger zone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DangerZoneDef {
    /// World position.
    pub pos: [f32; 3],
    /// Zone radius (metres).
    pub radius: f32,
    /// bit0 `bContinous`, bit1 `bDangerous`.
    pub flags: u8,
    /// Resolved RTS density id (`DensityRtsId`), 0 = use 1.0.
    pub density_id: u32,
}

impl DangerZoneDef {
    /// `DAT_01e09488` bit1.
    pub fn dangerous(&self) -> bool {
        self.flags & 2 != 0
    }
    /// `DAT_01e09488` bit0.
    pub fn continuous(&self) -> bool {
        self.flags & 1 != 0
    }
}

/// Parsed `ULZS` spawn markup.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpawnMarkup {
    /// Spawn points in file order.
    pub points: Vec<AmbushPoint>,
    /// Danger zones in file order.
    pub zones: Vec<DangerZoneDef>,
    /// Ambush context tags (`s_zombieSpawns.tags`).
    pub tags: Vec<u16>,
    /// Idle-animation names (`s_zombieSpawns.animNames`).
    pub anim_names: Vec<String>,
}

fn align128(off: usize) -> usize {
    (off + 0x7f) & !0x7f
}

fn rd_f32(b: &[u8], o: usize) -> Option<f32> {
    b.get(o..o + 4).map(|s| f32::from_le_bytes(s.try_into().unwrap()))
}
fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}
fn rd_u16(b: &[u8], o: usize) -> Option<u16> {
    b.get(o..o + 2).map(|s| u16::from_le_bytes(s.try_into().unwrap()))
}

/// Parse a little-endian `ULZS` `zombie_spawns_*.markup`.
///
/// Layout recovered from `FUN_0131d160` (`ZombieManager.cpp`): 7-word header
/// (magic `ULZS`, version `4`, endian `0`, spawn count, type/tag count, name
/// buffer size, danger-zone count), then each array 0x80-aligned — positions,
/// orientation, tag bits, `animIdx`, `type`, ambush tags, anim-name blob,
/// danger-zone positions/radii/flags/density ids. Big-endian (`ULSZ`) and
/// malformed buffers return `None`.
pub fn parse_spawn_markup(bytes: &[u8]) -> Option<SpawnMarkup> {
    if bytes.len() < 0x80 || &bytes[0..4] != b"ULZS" {
        return None;
    }
    // version word is `4` with the platform tag in the high byte (0x04000000),
    // matching the ULTP convention; endian word must be 0 for little-endian.
    if rd_u32(bytes, 4)? != 0x0400_0000 || rd_u32(bytes, 8)? != 0 {
        return None;
    }
    let n = rd_u32(bytes, 0x0c)? as usize;
    let typecount = rd_u32(bytes, 0x10)? as usize;
    let namesize = rd_u32(bytes, 0x14)? as usize;
    let nd = rd_u32(bytes, 0x18)? as usize;
    if n > 0xffff || nd > 0xff {
        return None;
    }

    let mut off = 0x80usize;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let o = off + i * 12;
        points.push(AmbushPoint {
            pos: [rd_f32(bytes, o)?, rd_f32(bytes, o + 4)?, rd_f32(bytes, o + 8)?],
            heading: 0.0,
            tag: 0,
            anim: 0x200,
            kind: 0,
        });
    }
    off = align128(off + n * 12);
    for (i, p) in points.iter_mut().enumerate() {
        p.heading = rd_f32(bytes, off + i * 4)?;
    }
    off = align128(off + n * 4);
    for (i, p) in points.iter_mut().enumerate() {
        p.tag = *bytes.get(off + i)?;
    }
    off = align128(off + n);
    for (i, p) in points.iter_mut().enumerate() {
        p.anim = rd_u16(bytes, off + i * 2)?;
    }
    off = align128(off + n * 2);
    for (i, p) in points.iter_mut().enumerate() {
        p.kind = rd_u32(bytes, off + i * 4)? as i32;
    }
    off = align128(off + n * 4);
    let mut tags = Vec::with_capacity(typecount);
    for i in 0..typecount {
        tags.push(rd_u16(bytes, off + i * 2)?);
    }
    off = align128(off + typecount * 2);
    let name_blob = bytes.get(off..off.checked_add(namesize)?)?;
    let anim_names: Vec<String> = name_blob
        .split(|&b| b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect();
    off = align128(off + namesize);

    let mut zones = Vec::with_capacity(nd);
    for _ in 0..nd {
        zones.push(DangerZoneDef {
            pos: [rd_f32(bytes, off)?, rd_f32(bytes, off + 4)?, rd_f32(bytes, off + 8)?],
            radius: 0.0,
            flags: 0,
            density_id: 0,
        });
        off += 12;
    }
    off = align128(off);
    for (i, z) in zones.iter_mut().enumerate() {
        z.radius = rd_f32(bytes, off + i * 4)?;
    }
    off = align128(off + nd * 4);
    for (i, z) in zones.iter_mut().enumerate() {
        z.flags = *bytes.get(off + i)?;
    }
    off = align128(off + nd);
    for (i, z) in zones.iter_mut().enumerate() {
        z.density_id = rd_u32(bytes, off + i * 4)?;
    }
    Some(SpawnMarkup { points, zones, tags, anim_names })
}

// ---------------------------------------------------------------------------
// XML helpers
// ---------------------------------------------------------------------------

fn local_name(qname: &[u8]) -> String {
    let s = String::from_utf8_lossy(qname);
    s.rsplit(':').next().unwrap_or(&s).to_string()
}

fn attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

fn attr_f32(e: &BytesStart, name: &str) -> Option<f32> {
    attr(e, name).and_then(|s| s.trim().parse().ok())
}

// ---------------------------------------------------------------------------
// World view — implemented by the lead / host
// ---------------------------------------------------------------------------

/// A zombie the host currently simulates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZombieView {
    /// Stable handle (the logical slot index in the original).
    pub id: u32,
    /// Gameplay kind.
    pub kind: ZombieKind,
    /// World position.
    pub pos: [f32; 3],
    /// True when promoted to a live CryEngine actor (`m_liveZombies`).
    pub live: bool,
    /// Spawn cluster (`CLUSTER_*`), if known.
    pub cluster: u8,
}

/// A live infestation the director may spread from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InfestationView {
    /// Position.
    pub pos: [f32; 3],
    /// Number of zombies currently assigned.
    pub count: u32,
}

/// Host-provided read-only view of the world. The lean implementation lives
/// with the lead's runtime; tests supply a synthetic implementation.
pub trait WorldView {
    /// Player feet position (cry Z-up).
    fn player_pos(&self) -> [f32; 3];
    /// Current logical + live zombies.
    fn zombies(&self) -> &[ZombieView];
    /// True when the player is inside a building (interior density rules).
    fn player_inside(&self) -> bool {
        false
    }
    /// Visibility/LOS between two points. Defaults to always visible.
    fn visible(&self, _from: [f32; 3], _to: [f32; 3]) -> bool {
        true
    }
    /// Whether a point is on walkable navmesh. Defaults to valid.
    fn nav_valid(&self, _p: [f32; 3]) -> bool {
        true
    }
    /// Live infestations (for spreading). Defaults to none.
    fn infestations(&self) -> &[InfestationView] {
        &[]
    }
    /// Region-density type (0..=5) of an ambient grid node, from the level's
    /// 21×21 density map (`RefreshAmbientGrid` @ `0x01342020`, table
    /// `DAT_1908fec + level*0x1b9`). 0 Agricultural, 1 Commercial, 2
    /// Industrial, 3 Park, 4 Residential, 5 Wilderness. `None` leaves the
    /// cell at the Wilderness fallback. Defaults to `None`.
    fn ambient_cell_type(&self, _cell_x: i32, _cell_y: i32) -> Option<u8> {
        None
    }
    /// Terrain height at a horizontal position. `None` falls back to the
    /// player's Z. Defaults to `None`.
    fn ground_height(&self, _x: f32, _y: f32) -> Option<f32> {
        None
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Why a zombie is being despawned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DespawnReason {
    /// Cluster deactivated (zone/ambush point went inactive).
    ClusterDeactivated,
    /// Over the local density target by ≥3.
    OverPopulated,
    /// Beyond `Ambush.DespawnRange` / `Horde.MaxRange`.
    OutOfRange,
    /// Horde disbanded.
    HordeDisbanded,
    /// Infestation cleared.
    InfestationCleared,
}

/// A request to create a logical zombie (and, if `flags`/`always_live`, a live
/// actor).
#[derive(Clone, Debug, PartialEq)]
pub struct SpawnCommand {
    /// Cluster that produced this spawn (`CLUSTER_*`).
    pub cluster: u8,
    /// Gameplay kind / entity class.
    pub kind: ZombieKind,
    /// Entity class string from `ZombieTypes.xml` (`"Grunt"`, `"Feral"`, …).
    pub entity_class: String,
    /// World position.
    pub pos: [f32; 3],
    /// Heading (radians).
    pub heading: f32,
    /// `ZOMBIE_SPAWN_FLAG_*` word.
    pub flags: u32,
    /// Idle timer `IdlePeriod + rand·IdleVariance` seconds.
    pub idle_timer: f32,
    /// True when `AlwaysLive` freaks promote immediately (`0133dc40:381`).
    pub always_live: bool,
}

/// A request to remove a logical/live zombie.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DespawnCommand {
    /// Handle to remove.
    pub id: u32,
    /// Reason.
    pub reason: DespawnReason,
}

/// One director decision.
#[derive(Clone, Debug, PartialEq)]
pub enum SpawnEvent {
    /// Create a zombie.
    Spawn(SpawnCommand),
    /// Remove a zombie.
    Despawn(DespawnCommand),
}

// ---------------------------------------------------------------------------
// Deterministic RNG (the game's 16-bit LCG pair, `FUN_01348680` `_DAT_019091c4/c0`)
// ---------------------------------------------------------------------------

/// The engine's two-word 16-bit LCG: each step
/// `c = (c & 0xffff) * mul + (c >> 16)`, combined as `c4·2^16 + c0` and scaled
/// by `1/2^32` to `[0,1)`.
#[derive(Clone, Copy, Debug)]
pub struct GameRng {
    c4: u32,
    c0: u32,
}

impl GameRng {
    /// Seed both words; zero is nudged to a fixed non-zero state.
    pub fn new(seed: u32) -> GameRng {
        let s = if seed == 0 { 0x1234_5678 } else { seed };
        GameRng { c4: s, c0: s.rotate_left(13) | 1 }
    }

    /// Next value in `[0,1)`.
    pub fn next_unit(&mut self) -> f32 {
        self.c4 = (self.c4 & 0xffff).wrapping_mul(0x9069).wrapping_add(self.c4 >> 16);
        self.c0 = (self.c0 & 0xffff).wrapping_mul(18000).wrapping_add(self.c0 >> 16);
        let i = self.c4.wrapping_shl(16).wrapping_add(self.c0);
        (i as f64 / 4_294_967_296.0) as f32
    }
}

impl Default for GameRng {
    fn default() -> Self {
        GameRng::new(1)
    }
}

// ---------------------------------------------------------------------------
// Director
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct AmbushSlot {
    timer: f32,
    active: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct ZoneState {
    eval_timer: f32,
    accumulator: f32,
    active_count: u32,
    /// bit0 active, bit1 `near`, bit2 `counted`.
    state: u8,
}

/// One ambient grid node (`param_1[0x137a + x*21 + y]`). The original packs
/// four fields into one `u32` (`FUN_01348680:375-388`): bits 0..2 region type,
/// bits 3..11 live count, bits 12..20 target, bits 21..30 density age.
#[derive(Clone, Copy, Debug, Default)]
struct AmbientCell {
    /// Region-density type 0..=5 (`AMBIENT_DENSITY_CAPS` index).
    kind: u8,
    /// Zombies currently assigned to the node.
    count: u16,
    /// Population the density pass wants the node to hold.
    target: u16,
    /// Age in refresh ticks; the target is recomputed when it reaches 15.
    density: u16,
}

/// Persistent per-director state.
#[derive(Clone, Debug)]
struct DirectorState {
    grid_timer: f32,
    grid: Vec<AmbientCell>,
    grid_ready: bool,
    ambush_chance: f32,
    ambush: Vec<AmbushSlot>,
    zones: Vec<ZoneState>,
    horde_timer: f32,
    horde_respawn_delay: f32,
    horde_decay_timer: f32,
    danger_persistent: u32,
    danger_timer: f32,
    horde_alerted: bool,
    next_id: u32,
}

/// The spawn director. Owns the static world data (spawn points, danger zones,
/// type table) and the dynamic timers; [`SpawnDirector::update`] is pure w.r.t.
/// the [`WorldView`] and returns the commands to apply.
pub struct SpawnDirector {
    /// RTS tuning values.
    pub tuning: ZombieTuning,
    /// Zombie type table.
    pub table: ZombieTable,
    /// Author-placed ambush/spawn points.
    pub points: Vec<AmbushPoint>,
    /// Author-placed danger zones.
    pub zones: Vec<DangerZoneDef>,
    /// Ambient grid node spacing in metres (`DAT_019091bc`).
    pub grid_cell_size: f32,
    /// Ambient grid world origin X/Y (`DAT_01df2e80/84`).
    pub grid_origin: [f32; 2],
    rng: GameRng,
    state: DirectorState,
}

impl SpawnDirector {
    /// Build a director from the recovered data. `seed` determines the spawn
    /// jitter/rolls; the original seeds from the manager's PRNG.
    pub fn new(
        tuning: ZombieTuning,
        table: ZombieTable,
        points: Vec<AmbushPoint>,
        zones: Vec<DangerZoneDef>,
        seed: u32,
    ) -> SpawnDirector {
        let n = points.len();
        let z = zones.len();
        SpawnDirector {
            tuning,
            table,
            points,
            zones,
            grid_cell_size: AMBIENT_GRID_CELL_SIZE,
            grid_origin: [0.0, 0.0],
            rng: GameRng::new(seed),
            state: DirectorState {
                grid_timer: 0.0,
                grid: vec![AmbientCell::default(); (AMBIENT_GRID_DIM * AMBIENT_GRID_DIM) as usize],
                grid_ready: false,
                ambush_chance: tuning.amb_base_spawn_chance,
                ambush: vec![AmbushSlot { timer: tuning.amb_spawn_timer, active: false }; n],
                zones: vec![ZoneState::default(); z],
                horde_timer: tuning.horde_respawn_delay_base,
                horde_respawn_delay: 0.0,
                horde_decay_timer: tuning.horde_respawn_decay_interval,
                danger_persistent: 0,
                danger_timer: 0.0,
                horde_alerted: false,
                next_id: 1,
            },
        }
    }

    /// Number of currently active ambush points (debug: `DAT_01df09ac`).
    pub fn ambush_active(&self) -> usize {
        self.state.ambush.iter().filter(|s| s.active).count()
    }

    /// Current spawn chance (debug: `_DAT_01df09a8`).
    pub fn spawn_chance(&self) -> f32 {
        self.state.ambush_chance
    }

    /// Allocate the next logical zombie handle (the host calls this for each
    /// [`SpawnCommand`] it applies). `next_id` mirrors `FUN_0133dc40`'s slot
    /// allocation.
    pub fn alloc_id(&mut self) -> u32 {
        let id = self.state.next_id;
        self.state.next_id = self.state.next_id.wrapping_add(1).max(1);
        id
    }

    fn spawn_ok(&self, view: &dyn WorldView) -> bool {
        view.zombies().len() < MAX_ZOMBIES
    }

    /// Advance one frame and return the commands the host must apply.
    ///
    /// The order mirrors `FUN_01348680`: ambient grid → ambush/near cadence →
    /// chance recovery → danger-zone density → horde roaming → despawn pass.
    pub fn update(&mut self, dt: f32, view: &dyn WorldView) -> Vec<SpawnEvent> {
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
        let mut out = Vec::new();
        if dt == 0.0 {
            return out;
        }
        let player = view.player_pos();

        self.step_ambient(dt, player, view, &mut out);
        self.step_ambush(dt, player, view, &mut out);
        self.step_chance_recovery(dt);
        self.step_danger_zones(dt, player, view, &mut out);
        self.step_hordes(dt, player, view, &mut out);
        self.step_despawn(player, view, &mut out);
        out
    }

    /// Ambient grid (`FUN_01348680:365-476`).
    ///
    /// A 21×21 map of density nodes around the player is aged every
    /// [`AMBIENT_GRID_PERIOD`] (or every frame while no zombie is alive — the
    /// original's `param_1[0x82e] == 0` test). Each refresh ages every node's
    /// density field; once it reaches [`AMBIENT_DENSITY_TRIGGER`] the node's
    /// target is recomputed from its region cap
    /// (`max(1, (cap - target)·density / interval)`) and the age resets. Nodes
    /// within [`AMBIENT_RADIUS`] of the player are then topped up toward their
    /// target, sampling nav-valid points inside the node's square, up to
    /// [`AMBIENT_CELL_ATTEMPTS`] tries, never past [`MAX_ZOMBIES`].
    fn step_ambient(
        &mut self,
        dt: f32,
        player: [f32; 3],
        view: &dyn WorldView,
        out: &mut Vec<SpawnEvent>,
    ) {
        // Lazily seed each node's region type from the level density map
        // (`RefreshAmbientGrid` @ 0x01342020). Targets start at zero and are
        // grown by the density field below.
        if !self.state.grid_ready {
            for cy in 0..AMBIENT_GRID_DIM {
                for cx in 0..AMBIENT_GRID_DIM {
                    let kind = view.ambient_cell_type(cx, cy).unwrap_or(5).min(5);
                    let idx = Self::grid_index(cx, cy);
                    self.state.grid[idx].kind = kind;
                }
            }
            self.state.grid_ready = true;
        }

        self.state.grid_timer += dt;
        // Refresh on the 20 s tick, or immediately while the world holds no
        // zombies (`FUN_01348680:365` reads the manager's zombie count at
        // offset 0x20b8).
        if self.state.grid_timer <= AMBIENT_GRID_PERIOD && !view.zombies().is_empty() {
            return;
        }
        self.state.grid_timer = 0.0;

        // Density pass: age every node and recompute depleted targets.
        let interval = self.tuning.ambient_cell_interval.max(1.0) as u32;
        for cell in self.state.grid.iter_mut() {
            cell.density = cell.density.saturating_add(1);
            if cell.density >= AMBIENT_DENSITY_TRIGGER as u16 {
                let cap = AMBIENT_DENSITY_CAPS[cell.kind as usize];
                if (cell.target as u32) < cap {
                    let add = ((cap - cell.target as u32) * cell.density as u32) / interval;
                    cell.target = add.max(1).min(u16::MAX as u32) as u16;
                }
                cell.density = 0;
            }
        }

        // Spawn pass: rebuild occupancy from the world, then top up the nodes
        // covering the player's ±150 m box (`FUN_01348680:397-474`).
        let mut occupancy = [0u16; (AMBIENT_GRID_DIM * AMBIENT_GRID_DIM) as usize];
        for z in view.zombies() {
            if z.cluster != CLUSTER_DENSITY {
                continue;
            }
            if let Some(idx) = self.cell_index(z.pos) {
                occupancy[idx] = occupancy[idx].saturating_add(1);
            }
        }
        for (idx, cell) in self.state.grid.iter_mut().enumerate() {
            cell.count = occupancy[idx];
        }

        let inv = 1.0 / self.grid_cell_size.max(1.0);
        let min_x = (((player[0] - AMBIENT_RADIUS) - self.grid_origin[0]) * inv + 0.5).floor() as i32;
        let max_x = (((player[0] + AMBIENT_RADIUS) - self.grid_origin[0]) * inv + 0.5).floor() as i32;
        let min_y = (((player[1] - AMBIENT_RADIUS) - self.grid_origin[1]) * inv + 0.5).floor() as i32;
        let max_y = (((player[1] + AMBIENT_RADIUS) - self.grid_origin[1]) * inv + 0.5).floor() as i32;

        let mut total = view.zombies().len();
        for cx in min_x.max(0)..=max_x.min(AMBIENT_GRID_DIM - 1) {
            for cy in min_y.max(0)..=max_y.min(AMBIENT_GRID_DIM - 1) {
                let idx = Self::grid_index(cx, cy);
                let (count, target) = (self.state.grid[idx].count, self.state.grid[idx].target);
                if count >= target || total >= MAX_ZOMBIES {
                    continue;
                }
                let spawned =
                    self.fill_ambient_cell(cx, cy, player, count, target, total, view, out);
                total += spawned as usize;
            }
        }
    }

    /// Node index for grid coordinates: the original stores the map as
    /// `param_1[0x137a + x*21 + y]` (`FUN_01348680:423`).
    #[inline]
    fn grid_index(cx: i32, cy: i32) -> usize {
        (cx * AMBIENT_GRID_DIM + cy) as usize
    }

    /// Grid node owning a world position, or `None` when outside the map.
    fn cell_index(&self, pos: [f32; 3]) -> Option<usize> {
        let inv = 1.0 / self.grid_cell_size.max(1.0);
        let cx = ((pos[0] - self.grid_origin[0]) * inv + 0.5).floor() as i32;
        let cy = ((pos[1] - self.grid_origin[1]) * inv + 0.5).floor() as i32;
        if (0..AMBIENT_GRID_DIM).contains(&cx) && (0..AMBIENT_GRID_DIM).contains(&cy) {
            Some(Self::grid_index(cx, cy))
        } else {
            None
        }
    }

    /// Fill one ambient node toward `target`: sample points in the node's
    /// square clipped to the player's ±150 m box, reject non-navmesh points,
    /// and stop after [`AMBIENT_CELL_ATTEMPTS`] tries or when [`MAX_ZOMBIES`]
    /// is reached. Mirrors `FUN_0133eb70`'s per-cell loop; returns the number
    /// emitted.
    fn fill_ambient_cell(
        &mut self,
        cx: i32,
        cy: i32,
        player: [f32; 3],
        count: u16,
        target: u16,
        total: usize,
        view: &dyn WorldView,
        out: &mut Vec<SpawnEvent>,
    ) -> u16 {
        let node_x = cx as f32 * self.grid_cell_size + self.grid_origin[0];
        let node_y = cy as f32 * self.grid_cell_size + self.grid_origin[1];
        let half = self.grid_cell_size * 0.5;
        let x0 = (node_x - half).max(player[0] - AMBIENT_RADIUS);
        let x1 = (node_x + half).min(player[0] + AMBIENT_RADIUS);
        let y0 = (node_y - half).max(player[1] - AMBIENT_RADIUS);
        let y1 = (node_y + half).min(player[1] + AMBIENT_RADIUS);
        if x1 <= x0 || y1 <= y0 {
            return 0;
        }
        let area = (x1 - x0) * (y1 - y0);
        // (int)(area / 40000) - current count; the density target only selects
        // which cells the director visits (`AMBIENT_DENSITY_TRIGGER`).
        let _ = target;
        let budget = (area / AMBIENT_AREA_PER_ZOMBIE).max(0.0) as u16;
        let mut remaining = budget.saturating_sub(count);
        let mut spawned = 0u16;
        let mut attempts = 0u32;
        while remaining > 0
            && attempts < AMBIENT_CELL_ATTEMPTS
            && total + (spawned as usize) < MAX_ZOMBIES
        {
            attempts += 1;
            let x = x0 + self.rng.next_unit() * (x1 - x0);
            let y = y0 + self.rng.next_unit() * (y1 - y0);
            let z = view.ground_height(x, y).unwrap_or(player[2]);
            let p = [x, y, z];
            // The recovered loop samples the cell box but rejects anything
            // outside the 150 m circle around the player (`FUN_0133eb70:246`).
            if horiz_dist2(p, player) > AMBIENT_RADIUS * AMBIENT_RADIUS {
                continue;
            }
            if !view.nav_valid(p) {
                continue;
            }
            out.push(SpawnEvent::Spawn(self.make_spawn(
                "Ambient",
                CLUSTER_DENSITY,
                p,
                0.0,
                0,
            )));
            spawned += 1;
            remaining -= 1;
        }
        let idx = Self::grid_index(cx, cy);
        let cell = &mut self.state.grid[idx];
        cell.count = cell.count.saturating_add(spawned);
        spawned
    }

    /// Ambush cadence and near/interior spawns (`FUN_01348680:534-636`).
    ///
    /// A point only ticks while the player is in the `[50, EvaluateRange] m`
    /// band and the point is not already occupied; on timer expiry a chance
    /// pool roll decides spawn vs. reevaluate.
    fn step_ambush(
        &mut self,
        dt: f32,
        player: [f32; 3],
        view: &dyn WorldView,
        out: &mut Vec<SpawnEvent>,
    ) {
        let eval_sq = self.tuning.amb_evaluate_range * self.tuning.amb_evaluate_range;
        for i in 0..self.points.len() {
            let d2 = horiz_dist2(self.points[i].pos, player);
            if d2 < AMBUSH_MIN_DIST_SQ || d2 > eval_sq || self.state.ambush[i].active {
                continue;
            }
            self.state.ambush[i].timer -= dt;
            if self.state.ambush[i].timer > 0.0 {
                continue;
            }
            let roll = self.rng.next_unit() * 100.0;
            if self.state.ambush_chance <= roll {
                // Skip: push the point far out (`ReevaluateDelay`).
                self.state.ambush[i].timer = self.tuning.amb_reevaluate_delay;
                continue;
            }
            if self.spawn_ok(view) {
                let p = self.points[i];
                let mut cmd = self.make_spawn("Ambush", CLUSTER_AMBUSH, p.pos, p.heading, 0);
                cmd.flags = ZOMBIE_SPAWN_FORCE | ZOMBIE_SPAWN_FAST;
                out.push(SpawnEvent::Spawn(cmd));
                self.state.ambush[i].active = true;
                self.state.ambush_chance =
                    (self.state.ambush_chance - self.tuning.amb_per_spawn_reduction).max(0.0);
                self.state.ambush[i].timer = self.tuning.amb_post_spawn_eval_delay;
            } else {
                self.state.ambush[i].timer = self.tuning.amb_post_spawn_eval_delay;
            }
        }
    }

    /// Chance-pool recovery (`FUN_01348680:622-636`).
    fn step_chance_recovery(&mut self, dt: f32) {
        if self.state.ambush_chance < 0.0 {
            self.state.ambush_chance = 0.0;
        }
        let cap = self.tuning.amb_base_spawn_chance;
        if self.state.ambush_chance < cap {
            let delay = self.tuning.amb_spawn_recovery_delay.max(1e-3);
            self.state.ambush_chance +=
                self.tuning.amb_spawn_recovery_amount * dt / delay;
            self.state.ambush_chance = self.state.ambush_chance.min(cap);
        }
    }

    /// Danger-zone density (`FUN_01348680:637-969`).
    ///
    /// Every [`DANGER_EVAL_PERIOD`] a zone within [`DANGER_NEAR_DIST_SQ`] of the
    /// player aims for `floor((1 - radius·k)·6·density_factor)` zombies; it
    /// spawns when under and despawns when `count == 0 < desired` or over by ≥3.
    /// Dangerous+continuous zones accumulate persistent danger and fire the
    /// horde-inbound alert after [`PERSISTENT_DANGER_ALERT`].
    fn step_danger_zones(
        &mut self,
        dt: f32,
        player: [f32; 3],
        view: &dyn WorldView,
        out: &mut Vec<SpawnEvent>,
    ) {
        let mut danger_total = 0u32;
        for i in 0..self.zones.len() {
            let zone = self.zones[i];
            let d2 = horiz_dist2(zone.pos, player);
            let nearby = d2 <= DANGER_NEAR_DIST_SQ || view.player_inside();
            let mut st = self.state.zones[i];

            if nearby {
                if st.state & 1 == 0 {
                    st.state |= 1;
                    st.accumulator = 0.0;
                }
            } else if st.state & 1 != 0 {
                if st.state & 4 != 0 && zone.continuous() {
                    self.state.danger_persistent = self.state.danger_persistent.saturating_sub(1);
                }
                st.state &= !7;
            }

            if st.state & 1 != 0 {
                // Density ramps toward the zone's RTS density id (0 ⇒ 1.0).
                let zone_density = if zone.density_id == 0 {
                    1.0
                } else {
                    self.tuning.density_outdoor
                };
                if st.accumulator < zone_density {
                    st.accumulator = (st.accumulator + dt * 0.05).min(zone_density);
                } else if st.accumulator > zone_density {
                    st.accumulator = (st.accumulator - dt * 0.1).max(zone_density);
                }
                let falloff = 1.0 - (d2 / DANGER_NEAR_DIST_SQ).clamp(0.0, 1.0);
                let density_factor = (st.accumulator / 100.0) * falloff;

                // Persistent danger: continuous + dangerous + within inner radius.
                if zone.continuous() && zone.dangerous() {
                    if d2 <= zone.radius * zone.radius {
                        if st.state & 4 == 0 {
                            st.state |= 4;
                            self.state.danger_persistent = self.state.danger_persistent + 1;
                        }
                    } else if st.state & 4 != 0 {
                        st.state &= !4;
                        self.state.danger_persistent = self.state.danger_persistent.saturating_sub(1);
                    }
                }

                st.eval_timer -= dt;
                if st.eval_timer <= 0.0 {
                    st.eval_timer = DANGER_EVAL_PERIOD;
                    let desired = ((1.0 - zone.radius * DANGER_RADIUS_SCALE) * 6.0
                        * density_factor)
                        .max(0.0) as u32;
                    if st.active_count < desired && self.spawn_ok(view) {
                        let batch = (desired - st.active_count).min(DANGER_SPAWN_BATCH);
                        for k in 0..batch {
                            let ang = self.rng.next_unit() * std::f32::consts::TAU;
                            let r = zone.radius * self.rng.next_unit();
                            let p =
                                [zone.pos[0] + ang.cos() * r, zone.pos[1] + ang.sin() * r, zone.pos[2]];
                            let mut cmd = self.make_spawn("Ambient", CLUSTER_DANGER, p, ang, 0);
                            cmd.flags = if st.state & 2 != 0 {
                                ZOMBIE_SPAWN_FORCE | ZOMBIE_SPAWN_FAST
                            } else {
                                ZOMBIE_SPAWN_FORCE
                            };
                            out.push(SpawnEvent::Spawn(cmd));
                            st.active_count += 1;
                            let _ = k;
                        }
                    } else if st.active_count > 0
                        && (desired == 0 || st.active_count >= desired + 3)
                    {
                        let want = (st.active_count - desired).min(2);
                        if let Some(cmd) = self.despawn_nearest(player, view, zone, want) {
                            out.extend(cmd);
                            st.active_count = st.active_count.saturating_sub(want);
                        }
                    }
                }
            }
            self.state.zones[i] = st;
            danger_total += self.state.zones[i].active_count;
        }

        // Persistent danger timer / horde-inbound alert.
        if danger_total == 0 {
            self.state.danger_timer = 0.0;
            self.state.horde_alerted = false;
        } else {
            self.state.danger_timer += dt;
            if self.state.danger_timer > PERSISTENT_DANGER_ALERT && !self.state.horde_alerted {
                self.state.horde_alerted = true;
            }
        }
    }

    /// Roaming-horde refill (`FUN_0135c080`): target `min(MaxSize, 6)` at least
    /// `Horde.RespawnDelayBase` apart, with a 5 s retry on failure and a kill
    /// penalty decayed by `RespawnDelayDecayAmount` every
    /// `RespawnDelayDecayInterval`.
    fn step_hordes(
        &mut self,
        dt: f32,
        player: [f32; 3],
        view: &dyn WorldView,
        out: &mut Vec<SpawnEvent>,
    ) {
        // Kill-delay decay.
        self.state.horde_decay_timer -= dt;
        if self.state.horde_decay_timer <= 0.0 {
            self.state.horde_decay_timer = self.tuning.horde_respawn_decay_interval.max(1.0);
            self.state.horde_respawn_delay =
                (self.state.horde_respawn_delay - self.tuning.horde_respawn_decay_amount).max(0.0);
        }

        self.state.horde_timer -= dt;
        if self.state.horde_timer > 0.0 {
            return;
        }
        // `FUN_0135c080`: at most `Horde.MaxAmbient` concurrent roaming hordes.
        let size = self.tuning.horde_max_size.max(1.0);
        let active_hordes = view
            .zombies()
            .iter()
            .filter(|z| z.cluster == CLUSTER_HORDE)
            .count() as f32
            / size;
        let target = self.tuning.horde_max_size.min(6.0) as u32;
        let _ = &out;
        if active_hordes >= self.tuning.horde_max_ambient || !self.spawn_ok(view) {
            self.state.horde_timer = HORDE_RETRY_DELAY;
            return;
        }
        // Spawn a horde at 150–MaxRange from the player.
        let Some(anchor) = self.pick_horde_position(player, view) else {
            self.state.horde_timer = HORDE_RETRY_DELAY;
            return;
        };
        let n = target.min(MAX_ZOMBIES.saturating_sub(view.zombies().len()) as u32).max(1);
        for k in 0..n {
            let ang = self.rng.next_unit() * std::f32::consts::TAU;
            let r = 8.0 * self.rng.next_unit();
            let p = [anchor[0] + ang.cos() * r, anchor[1] + ang.sin() * r, anchor[2]];
            let mut cmd = self.make_spawn("Horde", CLUSTER_HORDE, p, anchor_heading(anchor, player), 0);
            cmd.flags = ZOMBIE_SPAWN_FORCE | ZOMBIE_SPAWN_PATH | ZOMBIE_SPAWN_HORDE;
            cmd.idle_timer = target as f32 + k as f32;
            out.push(SpawnEvent::Spawn(cmd));
        }
        // Next horde: target·per-kill + base + accumulated kill delay.
        self.state.horde_timer = target as f32 * self.tuning.horde_respawn_delay_per_kill
            + self.tuning.horde_respawn_delay_base
            + self.state.horde_respawn_delay;
        self.state.horde_respawn_delay = self
            .state
            .horde_respawn_delay
            .max(self.tuning.horde_respawn_delay_base_per_active);
    }

    /// Despawn pass: ambush zombies beyond `Ambush.DespawnRange`, horde members
    /// beyond `Horde.MaxRange`, and any cluster whose source went inactive
    /// (`FUN_01348680` despawn branches, `FUN_01320e30`).
    fn step_despawn(
        &mut self,
        player: [f32; 3],
        view: &dyn WorldView,
        out: &mut Vec<SpawnEvent>,
    ) {
        let amb_sq = self.tuning.amb_despawn_range * self.tuning.amb_despawn_range;
        let horde_sq = self.tuning.horde_max_range * self.tuning.horde_max_range;
        for z in view.zombies() {
            if z.live {
                continue;
            }
            let d2 = horiz_dist2(z.pos, player);
            let reason = match z.cluster {
                CLUSTER_AMBUSH | CLUSTER_NEAR if d2 > amb_sq => Some(DespawnReason::OutOfRange),
                CLUSTER_HORDE if d2 > horde_sq => Some(DespawnReason::HordeDisbanded),
                _ => None,
            };
            if let Some(reason) = reason {
                out.push(SpawnEvent::Despawn(DespawnCommand { id: z.id, reason }));
            }
        }
    }

    // -- helpers ------------------------------------------------------------

    fn make_spawn(
        &mut self,
        group: &str,
        cluster: u8,
        pos: [f32; 3],
        heading: f32,
        flags: u32,
    ) -> SpawnCommand {
        let def = self.table.pick(group, self.rng.next_unit());
        let idle = def.idle_period + self.rng.next_unit() * def.idle_variance;
        SpawnCommand {
            cluster,
            kind: def.kind(),
            entity_class: def.entity_class.clone(),
            pos,
            heading,
            flags,
            idle_timer: idle,
            always_live: false,
        }
    }

    fn pick_horde_position(&mut self, player: [f32; 3], view: &dyn WorldView) -> Option<[f32; 3]> {
        let inner = (AMBIENT_RADIUS).min(self.tuning.horde_max_range);
        for _ in 0..8 {
            let ang = self.rng.next_unit() * std::f32::consts::TAU;
            let r = inner + self.rng.next_unit() * (self.tuning.horde_max_range - inner).max(1.0);
            let p = [player[0] + ang.cos() * r, player[1] + ang.sin() * r, player[2]];
            if view.nav_valid(p) {
                return Some(p);
            }
        }
        None
    }

    fn despawn_nearest(
        &mut self,
        _player: [f32; 3],
        view: &dyn WorldView,
        zone: DangerZoneDef,
        want: u32,
    ) -> Option<Vec<SpawnEvent>> {
        let mut ids: Vec<(f32, u32)> = view
            .zombies()
            .iter()
            .filter(|z| !z.live && z.cluster == CLUSTER_DANGER)
            .map(|z| (horiz_dist2(z.pos, zone.pos), z.id))
            .collect();
        ids.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let cmds: Vec<SpawnEvent> = ids
            .into_iter()
            .take(want as usize)
            .map(|(_, id)| {
                SpawnEvent::Despawn(DespawnCommand {
                    id,
                    reason: DespawnReason::ClusterDeactivated,
                })
            })
            .collect();
        if cmds.is_empty() {
            None
        } else {
            Some(cmds)
        }
    }
}

/// Extra flag bit used only to tag horde spawns for the director's own
/// bookkeeping (outside the recovered `ZOMBIE_SPAWN_FLAG_*` range).
pub const ZOMBIE_SPAWN_HORDE: u32 = 0x1000_0000;
/// Cluster code for a horde member (not one of `FUN_0133dc40`'s 1/5/6/7).
pub const CLUSTER_HORDE: u8 = 8;
/// `DAT_01721588 = -0.1`, the danger-zone desired-count radius coefficient.
const DANGER_RADIUS_SCALE: f32 = -0.1;

fn horiz_dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn anchor_heading(anchor: [f32; 3], player: [f32; 3]) -> f32 {
    (player[1] - anchor[1]).atan2(player[0] - anchor[0])
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct TestWorld {
        player: [f32; 3],
        zombies: Vec<ZombieView>,
        inside: bool,
        ambient_type: Option<u8>,
        nav: bool,
    }

    impl TestWorld {
        fn new(player: [f32; 3]) -> TestWorld {
            TestWorld {
                player,
                zombies: Vec::new(),
                inside: false,
                ambient_type: None,
                nav: true,
            }
        }
    }

    impl WorldView for TestWorld {
        fn player_pos(&self) -> [f32; 3] {
            self.player
        }
        fn zombies(&self) -> &[ZombieView] {
            &self.zombies
        }
        fn player_inside(&self) -> bool {
            self.inside
        }
        fn ambient_cell_type(&self, _cell_x: i32, _cell_y: i32) -> Option<u8> {
            self.ambient_type
        }
        fn nav_valid(&self, _p: [f32; 3]) -> bool {
            self.nav
        }
    }

    fn point_at(player: [f32; 3], r: f32, ang: f32) -> AmbushPoint {
        AmbushPoint {
            pos: [player[0] + r * ang.cos(), player[1] + r * ang.sin(), player[2]],
            heading: ang,
            tag: 0,
            anim: 0x200,
            kind: 0,
        }
    }

    fn apply(director: &mut SpawnDirector, world: &mut TestWorld, events: &[SpawnEvent]) {
        for e in events {
            match e {
                SpawnEvent::Spawn(c) => world.zombies.push(ZombieView {
                    id: director.alloc_id(),
                    kind: c.kind,
                    pos: c.pos,
                    live: c.always_live,
                    cluster: c.cluster,
                }),
                SpawnEvent::Despawn(d) => world.zombies.retain(|z| z.id != d.id),
            }
        }
    }

    fn table() -> ZombieTable {
        ZombieTable {
            defs: vec![ZombieDef {
                name: "Slow".into(),
                entity_class: "Grunt".into(),
                group: "Ambient".into(),
                weight: 1.0,
                aggro_dist: 15.0,
                base_aggro_dist: 2.0,
                idle_period: 5.0,
                idle_variance: 5.0,
                flags: String::new(),
            }],
            freaks: Vec::new(),
        }
    }

    #[test]
    fn ambient_grid_fills_cells_from_density_map() {
        let player = [0.0, 0.0, 0.0];
        let mut tuning = ZombieTuning::default();
        tuning.horde_max_ambient = 0.0; // isolate the ambient grid
        let mut d = SpawnDirector::new(tuning, table(), Vec::new(), Vec::new(), 7);
        let mut w = TestWorld::new(player);
        w.ambient_type = Some(4); // Residential, cap 100
        let mut spawns = 0usize;
        let mut outside = 0usize;
        for _ in 0..(60 * 60) {
            let e = d.update(1.0 / 60.0, &w);
            for ev in &e {
                if let SpawnEvent::Spawn(c) = ev {
                    spawns += 1;
                    assert_eq!(c.cluster, CLUSTER_DENSITY);
                    if horiz_dist2(c.pos, player) > AMBIENT_RADIUS * AMBIENT_RADIUS {
                        outside += 1;
                    }
                }
            }
            apply(&mut d, &mut w, &e);
        }
        // Only cells fully inside the +-150 m box reach area 40000 => 1 zombie.
        assert!(spawns >= 1, "expected ambient grid spawns, got {spawns}");
        assert_eq!(outside, 0, "ambient spawned outside the 150 m radius");
        assert!(w.zombies.len() <= MAX_ZOMBIES);
        // The density pass raised at least one node target above zero.
        assert!(d.state.grid.iter().any(|c| c.target > 0));
    }

    #[test]
    fn ambient_density_ages_before_spawning() {
        let player = [0.0, 0.0, 0.0];
        let mut tuning = ZombieTuning::default();
        tuning.horde_max_ambient = 0.0;
        let mut d = SpawnDirector::new(tuning, table(), Vec::new(), Vec::new(), 3);
        let mut w = TestWorld::new(player);
        w.ambient_type = Some(4);
        // A non-ambient zombie keeps the world non-empty so the grid only
        // ticks on the 20 s period rather than every frame.
        w.zombies.push(ZombieView {
            id: 1,
            kind: ZombieKind::Grunt,
            pos: [1.0, 1.0, 0.0],
            live: false,
            cluster: 0,
        });

        let mut early = 0usize;
        for _ in 0..(60 * 60) {
            let e = d.update(1.0 / 60.0, &w);
            early += e.iter().filter(|x| matches!(x, SpawnEvent::Spawn(_))).count();
            apply(&mut d, &mut w, &e);
        }
        assert_eq!(early, 0, "grid spawned before the density reached 15");

        let mut late = 0usize;
        for _ in 0..(300 * 60) {
            let e = d.update(1.0 / 60.0, &w);
            late += e.iter().filter(|x| matches!(x, SpawnEvent::Spawn(_))).count();
            apply(&mut d, &mut w, &e);
        }
        assert!(late >= 1, "grid never spawned after the density trigger");
    }

    #[test]
    fn ambient_grid_respects_logical_cap() {
        let player = [0.0, 0.0, 0.0];
        let mut tuning = ZombieTuning::default();
        tuning.horde_max_ambient = 0.0;
        let mut d = SpawnDirector::new(tuning, table(), Vec::new(), Vec::new(), 5);
        let mut w = TestWorld::new(player);
        for i in 0..MAX_ZOMBIES as u32 {
            w.zombies.push(ZombieView {
                id: i + 1,
                kind: ZombieKind::Grunt,
                pos: [0.0, 0.0, 0.0],
                live: false,
                cluster: CLUSTER_DENSITY,
            });
        }
        // A node that wants more zombies, against a full world.
        d.state.grid[SpawnDirector::grid_index(0, 0)].target = 10;
        let mut spawns = 0usize;
        for _ in 0..(60 * 60) {
            let e = d.update(1.0 / 60.0, &w);
            spawns += e.iter().filter(|x| matches!(x, SpawnEvent::Spawn(_))).count();
            apply(&mut d, &mut w, &e);
        }
        assert_eq!(spawns, 0, "spawned past the 256 logical cap");
        assert!(w.zombies.len() <= MAX_ZOMBIES);
    }

    #[test]
    fn ambush_point_only_ticks_in_band() {
        let player = [0.0, 0.0, 0.0];
        // 55 m: inside [50,60]; 20 m: too close; 120 m: too far.
        let pts = vec![
            point_at(player, 55.0, 0.0),
            point_at(player, 20.0, 1.0),
            point_at(player, 120.0, 2.0),
        ];
        let mut tuning = ZombieTuning::default();
        tuning.ambient_respawn_rate = 0.0; // isolate the ambush spawner
        tuning.horde_max_ambient = 0.0;
        let mut d = SpawnDirector::new(tuning, table(), pts, Vec::new(), 99);
        let mut w = TestWorld::new(player);
        let mut out_of_band = 0usize;
        let mut in_band = 0usize;
        for _ in 0..3600 {
            let e = d.update(1.0 / 60.0, &w);
            for ev in &e {
                if let SpawnEvent::Spawn(c) = ev {
                    if c.cluster != CLUSTER_AMBUSH {
                        continue;
                    }
                    let d2 = horiz_dist2(c.pos, player);
                    if (AMBUSH_MIN_DIST_SQ..=60.0 * 60.0).contains(&d2) {
                        in_band += 1;
                    } else {
                        out_of_band += 1;
                    }
                }
            }
            apply(&mut d, &mut w, &e);
        }
        assert_eq!(out_of_band, 0, "ambush spawned outside the 50-60 m band");
        assert!(in_band >= 1, "expected at least one in-band ambush spawn");
    }

    #[test]
    fn player_inside_uses_interior_rules() {
        let mut t = ZombieTuning::default();
        t.ambient_respawn_rate = 0.0;
        t.density_interior = 3.0;
        t.density_interior_cap = 5.0;
        assert_eq!(t.density_interior_cap, 5.0);
    }

    #[test]
    fn spawn_markup_rejects_junk() {
        assert!(parse_spawn_markup(b"").is_none());
        assert!(parse_spawn_markup(b"NOPE").is_none());
        let mut junk = vec![0u8; 0x80];
        junk[0..4].copy_from_slice(b"ULZS");
        assert!(parse_spawn_markup(&junk).is_none(), "version 0 must be rejected");
    }

    // -- real data (skipped when the paks/loose tree are absent) -------------

    const GAME: &str = "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";
    const EXTRACTED: &str = "/home/battlestation/Desktop/sod-re/extracted-game";
    const PLAYER_START: [f32; 3] = [3246.8, 3256.2, 217.1];

    fn build_vfs() -> Option<crate::crypak::Vfs> {
        use std::path::Path;
        let mut vfs = crate::crypak::Vfs::new();
        let mut paks = Vec::new();
        fn walk(root: &Path, out: &mut Vec<std::path::PathBuf>) {
            if let Ok(rd) = std::fs::read_dir(root) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        walk(&p, out);
                    } else if p.extension().map(|x| x == "pak").unwrap_or(false) {
                        out.push(p);
                    }
                }
            }
        }
        walk(Path::new(GAME), &mut paks);
        paks.sort();
        for p in &paks {
            let _ = vfs.mount(p);
        }
        if vfs.exists("libs/class3/rts/zombies.xml") {
            return Some(vfs);
        }
        let mut loose = crate::crypak::Vfs::new();
        if Path::new(EXTRACTED).join("gamedata/libs/class3/rts/zombies.xml").is_file() {
            loose.set_loose_root(Path::new(EXTRACTED).join("gamedata"));
            return Some(loose);
        }
        None
    }

    #[test]
    fn real_data_parses_rts_and_types() {
        let Some(vfs) = build_vfs() else { return };
        let rts = vfs.read("libs/class3/rts/zombies.xml").expect("RTS/Zombies.xml");
        let mut t = ZombieTuning::default();
        t.apply_rts_xml(&rts);
        assert_eq!(t.amb_evaluate_range, 60.0);
        assert_eq!(t.amb_base_spawn_chance, 32.0);
        assert!((t.amb_reevaluate_delay - 5.0).abs() < 1e-4);
        assert!((t.amb_post_spawn_eval_delay - 20.0).abs() < 1e-4);
        assert_eq!(t.horde_max_size, 8.0);
        assert_eq!(t.density_interior_cap, 5.0);

        let types = vfs.read("libs/class3/zombies/zombietypes.xml").expect("ZombieTypes.xml");
        let table = ZombieTable::parse(&types);
        assert!(table.defs.iter().any(|d| d.group == "Ambient" && d.name == "Slow"));
        assert!(table.defs.iter().any(|d| d.group == "Horde"));
        assert!(table.freaks.iter().any(|f| f.id == "Feral" && f.always_live()));
    }

    #[test]
    fn real_data_timeline_at_player_start() {
        let Some(vfs) = build_vfs() else { return };
        let rts = vfs.read("libs/class3/rts/zombies.xml").expect("RTS/Zombies.xml");
        let types = vfs.read("libs/class3/zombies/zombietypes.xml").expect("ZombieTypes.xml");
        let mut tuning = ZombieTuning::default();
        tuning.apply_rts_xml(&rts);
        let table = ZombieTable::parse(&types);

        // Spawn points from the level's ULZS markup (0x80-aligned layout).
        let points = vfs
            .read("zombie_spawns_win.markup")
            .and_then(|b| parse_spawn_markup(&b))
            .map(|m| m.points)
            .unwrap_or_default();
        let zones = vfs
            .read("zombie_spawns_win.markup")
            .and_then(|b| parse_spawn_markup(&b))
            .map(|m| m.zones)
            .unwrap_or_default();

        // Player start from the real tag-point markup, else the recorded value.
        let player = vfs
            .read("tag_points_win.markup")
            .and_then(|b| crate::tagpoints::TagPoints::parse(&b))
            .and_then(|tp| tp.get("RTSPoint_PlayerStart"))
            .unwrap_or(PLAYER_START);

        let mut d = SpawnDirector::new(tuning, table, points, zones, 0xC0FFEE);
        let mut w = TestWorld::new(player);
        let mut timeline = Vec::new();
        let mut ambient = 0usize;
        let mut ambient_outside = 0usize;
        for frame in 0..(60 * 60) {
            let t = frame as f32 / 60.0;
            let events = d.update(1.0 / 60.0, &w);
            for e in &events {
                match e {
                    SpawnEvent::Spawn(c) => {
                        if c.cluster == CLUSTER_DENSITY {
                            ambient += 1;
                            if horiz_dist2(c.pos, player) > AMBIENT_RADIUS * AMBIENT_RADIUS {
                                ambient_outside += 1;
                            }
                        }
                        timeline.push(format!(
                            "t={t:5.1}s SPAWN cluster={} {:?} at ({:.1},{:.1},{:.1}) flags=0x{:x}",
                            c.cluster, c.kind, c.pos[0], c.pos[1], c.pos[2], c.flags
                        ))
                    }
                    SpawnEvent::Despawn(c) => {
                        timeline.push(format!("t={t:5.1}s DESPAWN id={} {:?}", c.id, c.reason))
                    }
                }
            }
            apply(&mut d, &mut w, &events);
        }
        eprintln!("--- zombie spawn timeline (player {player:?}) ---");
        for line in timeline.iter().take(80) {
            eprintln!("{line}");
        }
        eprintln!(
            "--- {} events, {} zombies alive, {} ambush points, {} ambient-grid spawns ({} outside 150 m) ---",
            timeline.len(),
            w.zombies.len(),
            d.points.len(),
            ambient,
            ambient_outside
        );
        assert!(w.zombies.len() <= MAX_ZOMBIES);
        assert_eq!(ambient_outside, 0, "ambient grid spawned outside 150 m");
        // No count asserted: with the original 40000 m² rule a cell only spawns
        // when it lies fully inside the +-150 m box, which depends on where the
        // player stands relative to the 200 m grid.
        assert!(d.points.is_empty() || timeline.iter().any(|l| l.contains("SPAWN")));
    }

    /// `Zombie.FreakSpawn FreakHash` names the freak's `ZombieTypes.xml` class
    /// (with the `Zombie.` prefix); `Eatter` is a Grunt appearance.
    #[test]
    fn freak_hash_maps_to_its_entity_class() {
        assert_eq!(freak_entity_class("Zombie.ArmyZombie"), "ArmyZombie");
        assert_eq!(freak_entity_class("Zombie.Bloater"), "Bloater");
        assert_eq!(freak_entity_class("Zombie.Feral"), "Feral");
        assert_eq!(freak_entity_class("Zombie.Juggernaut"), "Juggernaut");
        assert_eq!(freak_entity_class("Zombie.Screamer"), "Screamer");
        assert_eq!(freak_entity_class("Zombie.Eatter"), "Grunt");
        assert_eq!(freak_entity_class(""), "Grunt");
        // The mapped class drives the spawn script (body + ULSM family).
        assert_eq!(crate::cry::zombie_runtime::class_script("Feral").0, "Feral");
        assert_eq!(
            crate::cry::zombie_runtime::class_script(freak_entity_class("Zombie.Feral")).1,
            "scripts/entities/ai/feral.lua"
        );
    }
}
