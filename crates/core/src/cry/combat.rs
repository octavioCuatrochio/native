//! The original combat / damage pipeline (`g_gameRules`), ported natively.
//!
//! The shipped scripts drive every hit through one path
//! (`scripts/gamerules/singleplayer.lua`, `scripts/entities/actor/basicactor.lua`):
//!
//! ```text
//! attacker -> CreateHit(...)                singleplayer.lua:332
//!   -> g_gameRules.game:ServerHit(...)      singleplayer.lua:354
//!   -> SinglePlayer.Server:OnHit(hit)       singleplayer.lua:997
//!   -> BasicActor.Server:OnHit(hit)         basicactor.lua:1081
//!        died = ProcessActorDamage(hit)     singleplayer.lua:797
//!   -> if died: ProcessDeath -> target:Kill  singleplayer.lua:959
//! ```
//!
//! `CreateHit`'s argument list is the transport struct recovered from
//! `FUN_013813b0` (`notes/ghidra/combat.md` §2.1): the caller supplies the raw
//! `damage`, the hit `type` and `material`, the hit `part`, the position/dir and
//! a `radius`; the engine does no arithmetic beyond the per-class rules in
//! [`process_actor_damage`]. This module implements that struct and the damage
//! rules; the actor health lives in [`super::actor`] and the app wires the two.
//!
//! Addresses are cited from the recovered decompilation / the shipped scripts;
//! the code is an independent Rust implementation.

use super::world::EntityId;
use crate::game::ZombieKind;

/// Player health pool, `g_playerHealthValue = 200.0` in every
/// `config/config/diff_*.cfg` (the entity's `ResetCommon` reads it, and the
/// player is exempt from the per-class `Damage.health`).
pub const PLAYER_MAX_HEALTH: f32 = 200.0;

/// Zombie melee hit-test radius: `melee.damageRadius = 1.1` in every
/// `scripts/entities/ai/*_x.lua` (`notes/ai-combat.md` §1.1, §4.4).
pub const MELEE_DAMAGE_RADIUS: f32 = 1.1;

/// `Zombies.Bloater.HeadshotKill` RTS toggle. The shipped default enables the
/// bloater headshot instant kill (`singleplayer.lua:868-871`); a headshot on any
/// other zombie subtracts full damage.
pub const BLOATER_HEADSHOT_KILL: bool = true;

/// A `CHitInfo` hit type. The names and order are exactly `InitHitTypes`
/// (`singleplayer.lua:101-124`); the engine id is the 1-based registration
/// index (`RegisterHitType` in [`super::gamerules`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HitType {
    Normal,
    Repair,
    Lockpick,
    Bullet,
    Fire,
    Radiation,
    Melee,
    Tac,
    Frag,
    Fall,
    Collision,
    DisableCollisions,
    Explosion,
    C4,
    Event,
    Punish,
    AvMine,
    MoacBullet,
    TrooperMelee,
    ScoutMoac,
    AaCannon,
}

impl HitType {
    /// Registration order from `InitHitTypes` (`singleplayer.lua:102-124`).
    pub const NAMES: &'static [&'static str] = &[
        "normal",
        "repair",
        "lockpick",
        "bullet",
        "fire",
        "radiation",
        "melee",
        "tac",
        "frag",
        "fall",
        "collision",
        "disableCollisions",
        "explosion",
        "c4",
        "event",
        "punish",
        "avmine",
        "moacbullet",
        "trooper_melee",
        "scout_moac",
        "aacannon",
    ];

    /// The 1-based id `GetHitTypeId` returns for this type (the registration
    /// index; `name_id` in [`super::gamerules`] is 1-based).
    pub fn id(self) -> i64 {
        match self {
            HitType::Normal => 1,
            HitType::Repair => 2,
            HitType::Lockpick => 3,
            HitType::Bullet => 4,
            HitType::Fire => 5,
            HitType::Radiation => 6,
            HitType::Melee => 7,
            HitType::Tac => 8,
            HitType::Frag => 9,
            HitType::Fall => 10,
            HitType::Collision => 11,
            HitType::DisableCollisions => 12,
            HitType::Explosion => 13,
            HitType::C4 => 14,
            HitType::Event => 15,
            HitType::Punish => 16,
            HitType::AvMine => 17,
            HitType::MoacBullet => 18,
            HitType::TrooperMelee => 19,
            HitType::ScoutMoac => 20,
            HitType::AaCannon => 21,
        }
    }

    pub fn from_id(id: i64) -> HitType {
        match id {
            2 => HitType::Repair,
            3 => HitType::Lockpick,
            4 => HitType::Bullet,
            5 => HitType::Fire,
            6 => HitType::Radiation,
            7 => HitType::Melee,
            8 => HitType::Tac,
            9 => HitType::Frag,
            10 => HitType::Fall,
            11 => HitType::Collision,
            12 => HitType::DisableCollisions,
            13 => HitType::Explosion,
            14 => HitType::C4,
            15 => HitType::Event,
            16 => HitType::Punish,
            17 => HitType::AvMine,
            18 => HitType::MoacBullet,
            19 => HitType::TrooperMelee,
            20 => HitType::ScoutMoac,
            21 => HitType::AaCannon,
            _ => HitType::Normal,
        }
    }

    /// Parse a `GetHitTypeId` name; unknown names default to `normal`, as
    /// `CreateHit` does when `type` is nil (`singleplayer.lua:350-351`).
    pub fn parse(name: &str) -> HitType {
        let low = name.to_ascii_lowercase();
        match Self::NAMES.iter().position(|n| n.eq_ignore_ascii_case(&low)) {
            Some(0) => HitType::Normal,
            Some(1) => HitType::Repair,
            Some(2) => HitType::Lockpick,
            Some(3) => HitType::Bullet,
            Some(4) => HitType::Fire,
            Some(5) => HitType::Radiation,
            Some(6) => HitType::Melee,
            Some(7) => HitType::Tac,
            Some(8) => HitType::Frag,
            Some(9) => HitType::Fall,
            Some(10) => HitType::Collision,
            Some(11) => HitType::DisableCollisions,
            Some(12) => HitType::Explosion,
            Some(13) => HitType::C4,
            Some(14) => HitType::Event,
            Some(15) => HitType::Punish,
            Some(16) => HitType::AvMine,
            Some(17) => HitType::MoacBullet,
            Some(18) => HitType::TrooperMelee,
            Some(19) => HitType::ScoutMoac,
            Some(20) => HitType::AaCannon,
            _ => HitType::Normal,
        }
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self.id() as usize - 1]
    }

    pub fn is_melee(self) -> bool {
        matches!(self, HitType::Melee | HitType::TrooperMelee)
    }

    pub fn is_bullet(self) -> bool {
        matches!(self, HitType::Bullet | HitType::MoacBullet | HitType::AaCannon | HitType::ScoutMoac)
    }
}

/// Body-part class a hit material belongs to. The material names are the
/// `InitHitMaterials` list (`singleplayer.lua:42-92`); `IsHeadShot`
/// (`singleplayer.lua:135-137`) tests the material's `head`/`helmet` type, so
/// the pipeline only needs the part class, not the exact material id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HitMaterial {
    Head,
    Torso,
    ArmLeft,
    ArmRight,
    HandLeft,
    HandRight,
    LegLeft,
    LegRight,
    FootLeft,
    FootRight,
    Unknown,
}

impl HitMaterial {
    /// Classify a registered material name. All the `mat_*`, `zed_*`,
    /// `zed_juggernaut_*`, `zed_feral_*` and `zed_bloater_*` spellings map by
    /// their trailing body-part word.
    pub fn parse(name: &str) -> HitMaterial {
        let low = name.to_ascii_lowercase();
        let low = low.trim_start_matches("mat_");
        if low.ends_with("head") || low.ends_with("helmet") {
            HitMaterial::Head
        } else if low.ends_with("torso") || low.ends_with("spine") || low.ends_with("chest") {
            HitMaterial::Torso
        } else if low.ends_with("arm_left") {
            HitMaterial::ArmLeft
        } else if low.ends_with("arm_right") {
            HitMaterial::ArmRight
        } else if low.ends_with("hand_left") {
            HitMaterial::HandLeft
        } else if low.ends_with("hand_right") {
            HitMaterial::HandRight
        } else if low.ends_with("leg_left") {
            HitMaterial::LegLeft
        } else if low.ends_with("leg_right") {
            HitMaterial::LegRight
        } else if low.ends_with("foot_left") {
            HitMaterial::FootLeft
        } else if low.ends_with("foot_right") {
            HitMaterial::FootRight
        } else {
            HitMaterial::Unknown
        }
    }

    pub fn is_head(self) -> bool {
        matches!(self, HitMaterial::Head)
    }

    pub fn is_limb(self) -> bool {
        matches!(
            self,
            HitMaterial::ArmLeft
                | HitMaterial::ArmRight
                | HitMaterial::HandLeft
                | HitMaterial::HandRight
                | HitMaterial::LegLeft
                | HitMaterial::LegRight
                | HitMaterial::FootLeft
                | HitMaterial::FootRight
        )
    }

    /// Location bit used by the dismember / reaction vocabulary
    /// (`notes/ghidra/combat.md` §2.3: arm right 1, arm left 2, head 4,
    /// leg left 8, leg right 16, torso 32, torso split 64).
    pub fn location_bit(self) -> u32 {
        match self {
            HitMaterial::ArmRight => 1,
            HitMaterial::ArmLeft => 2,
            HitMaterial::Head => 4,
            HitMaterial::LegLeft => 8,
            HitMaterial::LegRight => 16,
            HitMaterial::Torso => 32,
            _ => 0,
        }
    }
}

/// One hit record. Mirrors the `CreateHit` argument list
/// (`singleplayer.lua:332`) / `CHitInfo` (`0x013813b0`).
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub target: EntityId,
    pub shooter: Option<EntityId>,
    pub weapon: Option<EntityId>,
    pub damage: f32,
    pub radius: f32,
    pub hit_type: HitType,
    pub material: HitMaterial,
    pub pos: [f32; 3],
    pub dir: [f32; 3],
    pub part_id: i32,
    pub explosion: bool,
}

impl Hit {
    /// A melee hit at `target` (`CreateHit(..., dmg, radius, material, part,
    /// "melee", ...)`; radius = `melee.damageRadius`).
    pub fn melee(target: EntityId, shooter: EntityId, damage: f32, material: HitMaterial, pos: [f32; 3], dir: [f32; 3]) -> Hit {
        Hit {
            target,
            shooter: Some(shooter),
            weapon: None,
            damage,
            radius: MELEE_DAMAGE_RADIUS,
            hit_type: HitType::Melee,
            material,
            pos,
            dir,
            part_id: -1,
            explosion: false,
        }
    }

    /// A bullet hit at `target` (`hit.type = "bullet"`).
    pub fn bullet(target: EntityId, shooter: EntityId, weapon: EntityId, damage: f32, material: HitMaterial, pos: [f32; 3], dir: [f32; 3]) -> Hit {
        Hit {
            target,
            shooter: Some(shooter),
            weapon: Some(weapon),
            damage,
            radius: 0.0,
            hit_type: HitType::Bullet,
            material,
            pos,
            dir,
            part_id: -1,
            explosion: false,
        }
    }

    /// `IsHeadShot(hit)` (`singleplayer.lua:135`).
    pub fn is_headshot(&self) -> bool {
        self.material.is_head()
    }
}

/// Which damage-rule branch applies. Zombies use their [`ZombieKind`]; the
/// player and non-zombie actors take the default full-damage path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DamageTarget {
    Player,
    Zombie(ZombieKind),
}

impl DamageTarget {
    /// Map a spawned `EntityClass` name to its damage branch
    /// (`singleplayer.lua:866-881`; `FreshGrunt` shares `Grunt`'s rules).
    pub fn from_class(class: &str) -> DamageTarget {
        match class {
            "FreshGrunt" | "Grunt" => DamageTarget::Zombie(ZombieKind::Grunt),
            "Screamer" => DamageTarget::Zombie(ZombieKind::Screamer),
            "Feral" => DamageTarget::Zombie(ZombieKind::Feral),
            "ArmyZombie" | "SwatZombie" => DamageTarget::Zombie(ZombieKind::ArmyZombie),
            "Bloater" => DamageTarget::Zombie(ZombieKind::Bloater),
            "Juggernaut" => DamageTarget::Zombie(ZombieKind::Juggernaut),
            _ => DamageTarget::Player,
        }
    }

    fn kind(self) -> Option<ZombieKind> {
        match self {
            DamageTarget::Player => None,
            DamageTarget::Zombie(k) => Some(k),
        }
    }

    /// `KillableByCollision` (`notes/ai-combat.md` §1.1): every zombie except
    /// the Juggernaut. The player is killable by collision too.
    pub fn killable_by_collision(self) -> bool {
        match self.kind() {
            Some(ZombieKind::Juggernaut) => false,
            _ => true,
        }
    }
}

/// Result of the damage rules for one hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageOutcome {
    /// Health after the hit (never negative).
    pub health: f32,
    /// Health actually removed.
    pub dealt: f32,
    /// `ProcessActorDamage` return (`singleplayer.lua:911`).
    pub dead: bool,
    pub headshot: bool,
}

/// The `ProcessActorDamage` rules (`singleplayer.lua:797-912`), with the
/// engine's base subtraction applied first (the shipped Lua reads the
/// already-reduced `target.actor:GetHealth()`; the native actor is authoritative
/// here).
///
/// * collision + `KillableByCollision` + `damage > health` -> 0;
/// * `frag` explosion inside the 8 m kill band -> 0;
/// * `event` -> 0;
/// * headshot: Grunt/FreshGrunt/Screamer/Feral/Juggernaut full damage; Bloater
///   instant kill when `Zombies.Bloater.HeadshotKill > 0`;
/// * Juggernaut body: `damage * 0.5`, but clamped to a floor of 2;
/// * everything else: full damage.
pub fn process_actor_damage(
    health: f32,
    target: DamageTarget,
    hit: &Hit,
    bloater_headshot_kill: bool,
) -> DamageOutcome {
    let headshot = hit.is_headshot();
    let kind = target.kind();
    let health = health.max(0.0);
    let mut new_health = health - hit.damage.max(0.0);

    if hit.hit_type == HitType::Event {
        new_health = 0.0;
    } else if hit.hit_type == HitType::Collision {
        if target.killable_by_collision() && hit.damage > health {
            new_health = 0.0;
        }
    } else if hit.explosion && hit.hit_type == HitType::Frag {
        // Frag bands 0-3 m / 3-8 m force health 0; beyond 8 m only an impulse.
        // `hit.radius` carries the caller's explosion kill radius when set.
        let kill_band = if hit.radius > 0.0 { hit.radius } else { 8.0 };
        let dist = vec_dist(hit.pos, hit.dir);
        if dist <= kill_band {
            new_health = 0.0;
        }
    }

    if headshot {
        if kind == Some(ZombieKind::Bloater) {
            if bloater_headshot_kill {
                new_health = 0.0;
            }
        }
        // Grunt / FreshGrunt / Screamer / Feral / Juggernaut: full damage,
        // which is already applied above.
    } else if kind == Some(ZombieKind::Juggernaut) {
        let temp = hit.damage.max(0.0) * 0.5;
        new_health = if temp < health { health - temp } else { 2.0 };
    }

    let new_health = new_health.max(0.0);
    DamageOutcome {
        health: new_health,
        dealt: (health - new_health).max(0.0),
        dead: new_health <= 0.0,
        headshot,
    }
}

/// Melee damage a zombie deals per landed strike.
///
/// The exact scalar is C++-side and absent from both the shipped Lua and the
/// decompiled subset (`notes/ai-combat.md` §4.4, `notes/ghidra/combat.md`
/// §3.5). These values are calibrated against the 200 HP player pool so a normal
/// grunt takes many bites, freaks hit harder and the Juggernaut is worst.
pub fn zombie_melee_damage(kind: ZombieKind) -> f32 {
    match kind {
        ZombieKind::Grunt => 10.0,
        ZombieKind::Screamer => 8.0,
        ZombieKind::ArmyZombie => 15.0,
        ZombieKind::Bloater => 20.0,
        ZombieKind::Feral => 25.0,
        ZombieKind::Juggernaut => 50.0,
    }
}

/// Seconds between a zombie's melee strikes. The native manager has no attack
/// cooldown (`notes/ghidra/zombie_ai.md` §4); the CryAI combat state's
/// `Timer_CombatCooldown` owns the cadence, so this mirrors the recovered
/// strike timing (`[BEST-EFFORT]`).
pub fn zombie_melee_cooldown(kind: ZombieKind) -> f32 {
    match kind {
        ZombieKind::Grunt => 1.5,
        ZombieKind::Screamer => 2.0,
        ZombieKind::ArmyZombie => 1.4,
        ZombieKind::Bloater => 2.5,
        ZombieKind::Feral => 0.8,
        ZombieKind::Juggernaut => 2.8,
    }
}

/// English helper for the frag-band distance; `dir` is overloaded by the caller
/// as the explosion centre (the fields are positional in `CreateHit`).
fn vec_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// The engine-side melee test `CScriptBind_AI::CheckMeleeDamage`
/// (`0x0096ff70`, `notes/ghidra/combat.md` §2.2): the target must be inside a
/// forward cone of `half_angle_deg` and inside the reach band
/// `[min_reach, max_reach]`. Returns the hit angle in degrees when it lands, so
/// the caller can weight reactions, or `None`.
///
/// The forward vector is expected to be horizontal-normalised; the engine
/// normalises both the forward axis and the delta and compares
/// `acos(dot) <= deg2rad(half_angle_deg)`.
pub fn melee_arc_test(
    attacker: [f32; 3],
    forward: [f32; 3],
    target: [f32; 3],
    half_angle_deg: f32,
    min_reach: f32,
    max_reach: f32,
) -> Option<f32> {
    let d = [target[0] - attacker[0], target[1] - attacker[1], target[2] - attacker[2]];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if dist < min_reach || dist > max_reach {
        return None;
    }
    let fl = (forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]).sqrt();
    if fl < 1e-6 || dist < 1e-6 {
        return None;
    }
    let dot = (forward[0] * d[0] + forward[1] * d[1] + forward[2] * d[2]) / (fl * dist);
    let dot = dot.clamp(-1.0, 1.0);
    let angle_deg = dot.acos().to_degrees();
    if angle_deg <= half_angle_deg {
        Some(angle_deg)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn torso() -> HitMaterial {
        HitMaterial::Torso
    }

    #[test]
    fn hit_type_names_match_registration_order() {
        assert_eq!(HitType::NAMES.len(), 21);
        assert_eq!(HitType::parse("melee"), HitType::Melee);
        assert_eq!(HitType::parse("BULLET"), HitType::Bullet);
        assert_eq!(HitType::parse("collision").id(), 11);
        assert_eq!(HitType::from_id(HitType::Frag.id()), HitType::Frag);
        assert_eq!(HitType::parse("nonsense"), HitType::Normal);
        assert!(HitType::TrooperMelee.is_melee());
    }

    #[test]
    fn material_classifies_head_and_limbs() {
        assert!(HitMaterial::parse("head").is_head());
        assert!(HitMaterial::parse("zed_head").is_head());
        assert!(HitMaterial::parse("zed_feral_head").is_head());
        assert!(HitMaterial::parse("mat_head").is_head());
        assert_eq!(HitMaterial::parse("zed_juggernaut_leg_left"), HitMaterial::LegLeft);
        assert_eq!(HitMaterial::parse("mat_arm_right"), HitMaterial::ArmRight);
        assert_eq!(HitMaterial::parse("zed_bloater_torso"), HitMaterial::Torso);
        assert_eq!(HitMaterial::Head.location_bit(), 4);
        assert_eq!(HitMaterial::ArmRight.location_bit(), 1);
    }

    #[test]
    fn machete_one_hits_a_grunt() {
        // singleplayer.lua worked example: 80 melee vs a 25 HP grunt.
        let h = Hit::melee(1, 2, 80.0, torso(), [0.0; 3], [1.0, 0.0, 0.0]);
        let out = process_actor_damage(25.0, DamageTarget::Zombie(ZombieKind::Grunt), &h, true);
        assert!(out.dead);
        assert_eq!(out.health, 0.0);
        assert_eq!(out.dealt, 25.0);
    }

    #[test]
    fn juggernaut_body_is_halved_with_floor_two() {
        let h = Hit::melee(1, 2, 80.0, torso(), [0.0; 3], [1.0, 0.0, 0.0]);
        let body = process_actor_damage(2000.0, DamageTarget::Zombie(ZombieKind::Juggernaut), &h, true);
        assert_eq!(body.health, 1960.0);
        assert!(!body.dead);

        let tiny = Hit::melee(1, 2, 1.0, torso(), [0.0; 3], [1.0, 0.0, 0.0]);
        let floored = process_actor_damage(2000.0, DamageTarget::Zombie(ZombieKind::Juggernaut), &tiny, true);
        assert_eq!(floored.health, 1999.5);

        // Lethal body blow clamps to 2, never below.
        let heavy = Hit::melee(1, 2, 4000.0, torso(), [0.0; 3], [1.0, 0.0, 0.0]);
        let clamp = process_actor_damage(10.0, DamageTarget::Zombie(ZombieKind::Juggernaut), &heavy, true);
        assert_eq!(clamp.health, 2.0);
        assert!(!clamp.dead);
    }

    #[test]
    fn juggernaut_headshot_is_full_damage() {
        let head = Hit::melee(1, 2, 100.0, HitMaterial::Head, [0.0; 3], [1.0, 0.0, 0.0]);
        let out = process_actor_damage(2000.0, DamageTarget::Zombie(ZombieKind::Juggernaut), &head, true);
        assert_eq!(out.health, 1900.0);
        assert!(out.headshot);
    }

    #[test]
    fn bloater_headshot_instakill_is_rts_gated() {
        let head = Hit::melee(1, 2, 80.0, HitMaterial::Head, [0.0; 3], [1.0, 0.0, 0.0]);
        let on = process_actor_damage(1000.0, DamageTarget::Zombie(ZombieKind::Bloater), &head, true);
        assert!(on.dead && on.health == 0.0);
        let off = process_actor_damage(1000.0, DamageTarget::Zombie(ZombieKind::Bloater), &head, false);
        assert!(!off.dead);
        assert_eq!(off.health, 920.0);
    }

    #[test]
    fn collision_kills_only_killable_targets() {
        let mut h = Hit::melee(1, 2, 30.0, torso(), [0.0; 3], [1.0, 0.0, 0.0]);
        h.hit_type = HitType::Collision;
        let grunt = process_actor_damage(25.0, DamageTarget::Zombie(ZombieKind::Grunt), &h, true);
        assert!(grunt.dead);
        // Juggernaut is not KillableByCollision: normal damage path.
        let jug = process_actor_damage(2000.0, DamageTarget::Zombie(ZombieKind::Juggernaut), &h, true);
        assert_eq!(jug.health, 1985.0);
    }

    #[test]
    fn event_hit_is_fatal() {
        let mut h = Hit::melee(1, 2, 0.0, torso(), [0.0; 3], [1.0, 0.0, 0.0]);
        h.hit_type = HitType::Event;
        let out = process_actor_damage(100.0, DamageTarget::Player, &h, true);
        assert!(out.dead);
    }

    #[test]
    fn frag_in_kill_band_is_fatal() {
        let mut h = Hit::melee(1, 2, 10.0, torso(), [0.0, 0.0, 0.0], [6.0, 0.0, 0.0]);
        h.hit_type = HitType::Frag;
        h.explosion = true;
        h.radius = 8.0;
        let near = process_actor_damage(100.0, DamageTarget::Zombie(ZombieKind::Grunt), &h, true);
        assert!(near.dead);
        let mut far = h.clone();
        far.dir = [20.0, 0.0, 0.0];
        let out = process_actor_damage(100.0, DamageTarget::Zombie(ZombieKind::Grunt), &far, true);
        assert!(!out.dead);
    }

    #[test]
    fn player_takes_full_damage() {
        let h = Hit::melee(1, 2, 25.0, torso(), [0.0; 3], [1.0, 0.0, 0.0]);
        let out = process_actor_damage(200.0, DamageTarget::Player, &h, true);
        assert_eq!(out.health, 175.0);
        assert_eq!(out.dealt, 25.0);
    }

    #[test]
    fn melee_arc_uses_cone_and_reach() {
        let attacker = [0.0, 0.0, 0.0];
        let forward = [1.0, 0.0, 0.0];
        // Straight ahead, in reach.
        let ang = melee_arc_test(attacker, forward, [1.0, 0.0, 0.0], 60.0, 0.3, 2.0).unwrap();
        assert!(ang < 1.0);
        // Behind: outside the cone.
        assert!(melee_arc_test(attacker, forward, [-1.0, 0.0, 0.0], 60.0, 0.3, 2.0).is_none());
        // In the cone but too far.
        assert!(melee_arc_test(attacker, forward, [5.0, 0.0, 0.0], 60.0, 0.3, 2.0).is_none());
        // 45 deg right at 60 deg half-angle lands.
        assert!(melee_arc_test(attacker, forward, [1.0, 1.0, 0.0], 60.0, 0.3, 2.0).is_some());
    }

    #[test]
    fn damage_target_from_class_maps_freaks() {
        assert_eq!(DamageTarget::from_class("FreshGrunt"), DamageTarget::Zombie(ZombieKind::Grunt));
        assert_eq!(DamageTarget::from_class("ArmyZombie"), DamageTarget::Zombie(ZombieKind::ArmyZombie));
        assert_eq!(DamageTarget::from_class("Survivor"), DamageTarget::Player);
    }
}
