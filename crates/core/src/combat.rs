//! Combat: actor vitals, melee, firearms, zombie attacks and hit reactions.
//!
//! This is the Rust side of the recovered damage pipeline described in
//! `notes/ai-combat.md §4` and `notes/ghidra/combat.md §2/§3`:
//!
//! ```text
//! attacker (melee / bullet)
//!   -> base `damage` + hit location
//!   -> g_gameRules:ProcessActorDamage   (singleplayer.lua:797-912)
//!   -> Juggernaut body ×0.5 floor 2, headshots full damage
//!   -> SetHealth / Kill -> hit/death reaction (Flinch/Stumble/Dismember/Death)
//! ```
//!
//! Zombie HP math is delegated to [`crate::game::apply_zombie_damage`] so the
//! two modules cannot drift.  Values that the shipped data does not pin
//! (player melee reach/arc, stamina costs, zombie attack scalars) are marked
//! `[BEST-EFFORT]` and calibrated against the recovered constants.

use crate::game::{self, HitLocation, ZombieKind};
use log::debug;

/// Player health pool, `g_playerHealthValue` in `config/config/diff_*.cfg`.
pub const PLAYER_MAX_HEALTH: f32 = 200.0;

/// Stamina regained per second while alive (`[BEST-EFFORT]`; no shipped scalar).
pub const STAMINA_REGEN_PER_SEC: f32 = 15.0;

/// Rate at which a bleed wound closes, in bleed-units per second.
pub const BLEED_RECOVER_PER_SEC: f32 = 2.0;

/// Head hits are the "heavy" survivor wound (`[BEST-EFFORT]`).
pub const HEAD_DAMAGE_MULT: f32 = 2.0;

/// Arm/leg hits are the "light" survivor wound (`[BEST-EFFORT]`).
pub const LIMB_DAMAGE_MULT: f32 = 0.5;

/// Fraction of dealt damage converted into a bleed wound.
pub const BLEED_FRACTION: f32 = 0.1;

/// Fraction of `max_hp` restored by [`Vitals::revive`].
pub const REVIVE_HP_FRACTION: f32 = 0.5;

/// Base stamina channeled by a melee swing (`[BEST-EFFORT]`).
pub const MELEE_STAMINA_BASE: f32 = 5.0;

/// A swing made with insufficient stamina deals this fraction of damage.
pub const TIRED_STAMINA_MULT: f32 = 0.5;

/// A single blow at or above this base damage can sever a limb (`§2.3`).
pub const SEVER_DAMAGE_THRESHOLD: f32 = 100.0;

/// Damage fraction at/above which a surviving hit triggers a stumble.
pub const STUMBLE_DAMAGE_FRACTION: f32 = 0.35;

/// Damage fraction at/above which a lethal limb hit dismembers (`§2.3`).
pub const DISMEMBER_DAMAGE_FRACTION: f32 = 0.6;

/// `Zombies.Bloater.HeadshotKill` RTS toggle. The shipped default enables the
/// bloater headshot instant-kill (`singleplayer.lua:864-881`).
pub const BLOATER_HEADSHOT_KILL: bool = true;

// ---------------------------------------------------------------------------
// Vitals
// ---------------------------------------------------------------------------

/// Health + stamina for a survivable actor, plus persistent wound state.
///
/// `injuries` is a bitmask of [`HitLocation`] constants; `bleed` is in HP per
/// second and is drained by [`Vitals::tick`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vitals {
    pub hp: f32,
    pub max_hp: f32,
    pub stamina: f32,
    pub max_stamina: f32,
    /// Bleed severity (HP/second) applied while alive.
    pub bleed: f32,
    /// Bitmask of injured limbs (`game::HitLocation::*`).
    pub injuries: u32,
}

impl Vitals {
    /// Fresh vitals at full health and stamina, no wounds.
    pub fn new(max_hp: f32, max_stamina: f32) -> Self {
        Vitals {
            hp: max_hp,
            max_hp,
            stamina: max_stamina,
            max_stamina,
            bleed: 0.0,
            injuries: 0,
        }
    }

    /// Advance time: bleeding drains HP, stamina regenerates. A downed actor
    /// does not regenerate.
    pub fn tick(&mut self, dt: f32) {
        if self.hp <= 0.0 {
            return;
        }
        if self.bleed > 0.0 {
            self.hp = (self.hp - self.bleed * dt).max(0.0);
            self.bleed = (self.bleed - BLEED_RECOVER_PER_SEC * dt).max(0.0);
        }
        self.stamina = (self.stamina + STAMINA_REGEN_PER_SEC * dt).min(self.max_stamina);
    }

    /// Apply a wound at `location`. Arm/leg hits are lightened, head hits are
    /// heavy, torso hits apply nominally. The location is recorded in
    /// `injuries` and a proportional bleed wound is opened. Returns the HP
    /// actually dealt.
    pub fn apply_hit(&mut self, dmg: f32, location: u32) -> f32 {
        let mult = if location & HitLocation::HEAD != 0 {
            HEAD_DAMAGE_MULT
        } else if is_limb(location) {
            LIMB_DAMAGE_MULT
        } else {
            1.0
        };
        let dealt = (dmg * mult).max(0.0);
        self.hp = (self.hp - dealt).max(0.0);
        self.injuries |= location;
        let bleed_add = dealt * BLEED_FRACTION * if is_limb(location) { 0.5 } else { 1.0 };
        self.bleed = self.bleed.max(bleed_add);
        debug!(
            "Vitals hit loc {:#x}: dealt {:.2} hp={:.2} bleed={:.2}",
            location, dealt, self.hp, self.bleed
        );
        dealt
    }

    /// True when the actor has been reduced to 0 HP.
    pub fn is_downed(&self) -> bool {
        self.hp <= 0.0
    }

    /// Bring a downed actor back at [`REVIVE_HP_FRACTION`] of max HP, closing
    /// any active bleed. Existing injuries are retained.
    pub fn revive(&mut self) {
        self.hp = (self.max_hp * REVIVE_HP_FRACTION).max(1.0);
        self.bleed = 0.0;
    }

    /// Restore up to `amount` HP, returning the amount actually restored.
    pub fn heal(&mut self, amount: f32) -> f32 {
        let before = self.hp;
        self.hp = (self.hp + amount.max(0.0)).min(self.max_hp);
        self.hp - before
    }
}

/// True when `location` names a severable limb bit.
pub fn is_limb(location: u32) -> bool {
    const LIMBS: u32 = HitLocation::ARM_RIGHT
        | HitLocation::ARM_LEFT
        | HitLocation::LEG_LEFT
        | HitLocation::LEG_RIGHT;
    location & LIMBS != 0
}

/// True when `location` can be severed by a hit. Per `ghidra/combat.md §2.3`
/// arms, legs and the split torso ("guts") have sever effect tags; the plain
/// torso and head do not.
pub fn is_severable(location: u32) -> bool {
    is_limb(location) || location & HitLocation::TORSO_SPLIT != 0
}

// ---------------------------------------------------------------------------
// Melee
// ---------------------------------------------------------------------------

/// A player melee weapon. Reach/arc are `[BEST-EFFORT]`; damage is verbatim
/// from the weapon XMLs (`[FISTS|MACHETE|SLEDGEHAMMER] = 80`, `ai-combat.md
/// §4.5`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeleeWeapon {
    pub name: &'static str,
    pub damage: f32,
    pub range: f32,
    pub arc_deg: f32,
    pub durability_cost: f32,
}

impl MeleeWeapon {
    pub const fn new(
        name: &'static str,
        damage: f32,
        range: f32,
        arc_deg: f32,
        durability_cost: f32,
    ) -> Self {
        MeleeWeapon {
            name,
            damage,
            range,
            arc_deg,
            durability_cost,
        }
    }

    /// Bare hands: 80 melee (`weapons/fists.xml`), no durability wear.
    pub const FISTS: MeleeWeapon = MeleeWeapon::new("Fists", game::melee::FISTS, 1.1, 60.0, 0.0);

    /// Machete: 80 melee (`meleemachete02.xml`), moderate wear.
    pub const MACHETE: MeleeWeapon =
        MeleeWeapon::new("Machete", game::melee::MACHETE, 1.3, 90.0, 1.0);

    /// Sledgehammer: 80 melee (`meleesledgehammer.xml`), heavy wear.
    pub const SLEDGEHAMMER: MeleeWeapon =
        MeleeWeapon::new("Sledgehammer", game::melee::SLEDGEHAMMER, 1.6, 70.0, 2.5);

    pub const fn fists() -> Self {
        Self::FISTS
    }
    pub const fn machete() -> Self {
        Self::MACHETE
    }
    pub const fn sledgehammer() -> Self {
        Self::SLEDGEHAMMER
    }

    /// Stamina channeled by one swing: a base cost plus wear.
    pub fn stamina_cost(&self) -> f32 {
        MELEE_STAMINA_BASE + self.durability_cost * 2.0
    }
}

/// Result of resolving one melee swing against a zombie.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeleeHit {
    /// HP actually removed after the zombie damage model is applied.
    pub damage: f32,
    pub killed: bool,
    /// Severed limb location bit, if the blow dismembered (`ghidra/combat.md §2.3`).
    pub sever: Option<u32>,
    pub stamina_cost: f32,
}

/// Resolve `w` striking a zombie at `location`.
///
/// HP math is delegated to [`game::apply_zombie_damage`] (Juggernaut body ×0.5
/// floor 2, full-damage headshots). A swing made with less stamina than
/// [`MeleeWeapon::stamina_cost`] is "tired" and deals reduced damage. A
/// severable location dismembers when the blow kills or is heavy enough.
pub fn melee_swing(
    w: &MeleeWeapon,
    target: ZombieKind,
    target_hp: f32,
    location: u32,
    stamina: f32,
) -> MeleeHit {
    let stamina_cost = w.stamina_cost();
    let tired = stamina < stamina_cost;
    let base = if tired {
        w.damage * TIRED_STAMINA_MULT
    } else {
        w.damage
    };
    let remaining = game::apply_zombie_damage(target_hp, target, base, location, BLOATER_HEADSHOT_KILL);
    let dealt = (target_hp - remaining).max(0.0);
    let killed = remaining <= 0.0;
    let sever = if is_severable(location) && (killed || base >= SEVER_DAMAGE_THRESHOLD) {
        Some(location)
    } else {
        None
    };
    debug!(
        "melee {} -> {} hp {:.1}->{:.1} (dealt {:.1}, killed {}, sever {:?})",
        w.name,
        target.name(),
        target_hp,
        remaining,
        dealt,
        killed,
        sever
    );
    MeleeHit {
        damage: dealt,
        killed,
        sever,
        stamina_cost,
    }
}

// ---------------------------------------------------------------------------
// Firearms
// ---------------------------------------------------------------------------

/// A ranged weapon. Damage / rate / clip / reload are verbatim from the weapon
/// XML `default` firemode (`ai-combat.md §4.5`); reach is
/// `damage_drop_min_distance` and spread is the XML max radius converted from
/// radians to degrees.
#[derive(Debug, Clone, PartialEq)]
pub struct Firearm {
    pub name: String,
    pub damage: f32,
    pub mag_size: u32,
    pub rpm: f32,
    pub spread_deg: f32,
    /// Headshot multiplier. The Lua path subtracts full `hit.damage` on a
    /// headshot, so shipped weapons carry `1.0`.
    pub headshot_mult: f32,
    /// Full-damage reach (`damage_drop_min_distance`).
    pub range: f32,
    /// Reload duration in seconds (`reload_time`).
    pub reload_time: f32,
}

impl Firearm {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        damage: f32,
        mag_size: u32,
        rpm: f32,
        spread_deg: f32,
        headshot_mult: f32,
        range: f32,
        reload_time: f32,
    ) -> Self {
        Firearm {
            name: name.into(),
            damage,
            mag_size,
            rpm,
            spread_deg,
            headshot_mult,
            range,
            reload_time,
        }
    }

    /// TacMaster 1911: 200 dmg, 600 rpm, 7 rounds, 0.5 s reload
    /// (`pistol_tacmaster1911.xml`).
    pub fn pistol_tacmaster1911() -> Self {
        Firearm::new("TacMaster 1911", 200.0, 7, 600.0, 0.0, 1.0, 20.0, 0.5)
    }

    /// AK47: 300 dmg, 760 rpm, 30 rounds, 1.2 s reload; max spread 0.12 rad
    /// ≈ 6.875° (`assaultrifle_ak47.xml`).
    pub fn ak47() -> Self {
        Firearm::new("AK47", 300.0, 30, 760.0, 6.875, 1.0, 32.0, 1.2)
    }

    /// Damage for a shot at `location` (headshots apply `headshot_mult`).
    pub fn damage_for(&self, location: u32) -> f32 {
        if location & HitLocation::HEAD != 0 {
            self.damage * self.headshot_mult
        } else {
            self.damage
        }
    }

    /// True if `dist` (m) is within full-damage reach.
    pub fn in_range(&self, dist: f32) -> bool {
        dist <= self.range
    }

    /// Minimum seconds between shots for this fire rate.
    pub fn shot_interval(&self) -> f32 {
        if self.rpm > 0.0 {
            60.0 / self.rpm
        } else {
            0.0
        }
    }
}

/// Ammo/timing state for one equipped [`Firearm`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GunState {
    pub in_mag: u32,
    pub reserve: u32,
    /// Seconds until the next shot is allowed.
    pub cooldown: f32,
    /// Seconds left in the current reload.
    pub reload_left: f32,
}

impl GunState {
    pub fn new(in_mag: u32, reserve: u32) -> Self {
        GunState {
            in_mag,
            reserve,
            cooldown: 0.0,
            reload_left: 0.0,
        }
    }

    /// Attempt to fire, advancing timers by `dt` seconds first. Returns the
    /// base damage of the shot, or `None` when empty, reloading or cooling
    /// down.
    pub fn fire(&mut self, gun: &Firearm, dt: f32) -> Option<f32> {
        self.tick(dt);
        if self.reload_left > 0.0 || self.cooldown > 0.0 || self.in_mag == 0 {
            return None;
        }
        self.in_mag -= 1;
        self.cooldown = gun.shot_interval();
        Some(gun.damage)
    }

    /// Begin a reload if the magazine is not full, no reload is running and
    /// reserve ammo exists. Rounds are transferred immediately; the weapon is
    /// unusable until `reload_left` reaches 0. Returns whether one started.
    pub fn reload(&mut self, gun: &Firearm) -> bool {
        if self.reload_left > 0.0 || self.in_mag >= gun.mag_size || self.reserve == 0 {
            return false;
        }
        let need = gun.mag_size - self.in_mag;
        let take = need.min(self.reserve);
        self.in_mag += take;
        self.reserve -= take;
        self.reload_left = gun.reload_time;
        debug!(
            "reloaded {}: mag {}/{} reserve {}",
            gun.name, self.in_mag, gun.mag_size, self.reserve
        );
        true
    }

    /// Advance weapon timers without firing.
    pub fn tick(&mut self, dt: f32) {
        self.cooldown = (self.cooldown - dt).max(0.0);
        self.reload_left = (self.reload_left - dt).max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Zombie attacks
// ---------------------------------------------------------------------------

/// Melee damage a zombie deals to a survivor per landed strike.
///
/// `[BEST-EFFORT]` — `ai-combat.md §4.4` and `ghidra/combat.md §3.5` both
/// confirm the scalar was never present in the shipped data or the decompiled
/// subset. These values are calibrated against [`PLAYER_MAX_HEALTH`] so a
/// normal grunt needs many bites to down a survivor, freaks hit harder, and a
/// Juggernaut is the most dangerous.
pub fn zombie_attack_damage(kind: ZombieKind) -> f32 {
    match kind {
        ZombieKind::Grunt => 10.0,
        ZombieKind::Screamer => 8.0,
        ZombieKind::ArmyZombie => 15.0,
        ZombieKind::Bloater => 20.0,
        ZombieKind::Feral => 25.0,
        ZombieKind::Juggernaut => 50.0,
    }
}

/// Seconds between zombie melee strikes.
///
/// `[BEST-EFFORT]` — derived from the combat-state strike cadence
/// (`ai-combat.md §3.5`, `Timer_CombatCooldown`); the data does not pin it.
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

// ---------------------------------------------------------------------------
// Hit reactions
// ---------------------------------------------------------------------------

/// Discrete reaction selected for a landed hit (`ai-combat.md §4.3`,
/// `ghidra/combat.md §2.3/§3.4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reaction {
    /// Light hit: recoil only.
    Flinch,
    /// Heavy hit that did not kill: lost footing.
    Stumble,
    /// A severable location was dismembered.
    Dismember(u32),
    Death,
}

/// Choose the reaction for a hit. `damage_fraction` is the dealt damage divided
/// by the target's max HP. A lethal severable location dismembers when the
/// blow was heavy enough, otherwise death; a non-lethal hit stumbles when
/// heavy, else flinches.
pub fn reaction_for(location: u32, damage_fraction: f32, killed: bool) -> Reaction {
    if killed {
        if is_severable(location) && damage_fraction >= DISMEMBER_DAMAGE_FRACTION {
            Reaction::Dismember(location)
        } else {
            Reaction::Death
        }
    } else if damage_fraction >= STUMBLE_DAMAGE_FRACTION {
        Reaction::Stumble
    } else {
        Reaction::Flinch
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::melee as game_melee;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-3, "expected {b}, got {a}");
    }

    #[test]
    fn machete_one_hits_grunt() {
        // Worked example `ai-combat.md §4.6`: 80 melee vs 25 HP grunt.
        let hit = melee_swing(
            &MeleeWeapon::MACHETE,
            ZombieKind::Grunt,
            25.0,
            HitLocation::TORSO,
            100.0,
        );
        assert!(hit.killed);
        approx(hit.damage, 25.0); // clamped at the target's HP
        assert_eq!(hit.sever, None); // torso does not sever
        approx(hit.stamina_cost, MeleeWeapon::MACHETE.stamina_cost());
    }

    #[test]
    fn juggernaut_body_damage_halved_with_floor() {
        // 2000 HP juggernaut: 80 melee body -> 40 dealt (×0.5).
        let body = melee_swing(
            &MeleeWeapon::MACHETE,
            ZombieKind::Juggernaut,
            2000.0,
            HitLocation::TORSO,
            100.0,
        );
        approx(body.damage, 40.0);
        assert!(!body.killed);

        // Fists tiny-damage floor (2) is exercised through the same path.
        let mut w = MeleeWeapon::FISTS;
        w.damage = 1.0;
        let floored = melee_swing(&w, ZombieKind::Juggernaut, 2000.0, HitLocation::TORSO, 100.0);
        approx(floored.damage, 2.0);
    }

    #[test]
    fn head_shot_does_full_damage() {
        // Headshots bypass the Juggernaut ×0.5 body penalty (singleplayer.lua).
        let head = melee_swing(
            &MeleeWeapon::MACHETE,
            ZombieKind::Juggernaut,
            2000.0,
            HitLocation::HEAD,
            100.0,
        );
        approx(head.damage, game_melee::MACHETE);
        let body = melee_swing(
            &MeleeWeapon::MACHETE,
            ZombieKind::Juggernaut,
            2000.0,
            HitLocation::TORSO,
            100.0,
        );
        assert!(head.damage > body.damage);
    }

    #[test]
    fn tired_swing_does_less_damage() {
        let cost = MeleeWeapon::MACHETE.stamina_cost();
        let fresh = melee_swing(
            &MeleeWeapon::MACHETE,
            ZombieKind::Juggernaut,
            2000.0,
            HitLocation::TORSO,
            cost,
        );
        let tired = melee_swing(
            &MeleeWeapon::MACHETE,
            ZombieKind::Juggernaut,
            2000.0,
            HitLocation::TORSO,
            cost - 0.5,
        );
        approx(tired.damage, fresh.damage * TIRED_STAMINA_MULT);
    }

    #[test]
    fn lethal_limb_hit_severs() {
        let hit = melee_swing(
            &MeleeWeapon::MACHETE,
            ZombieKind::Grunt,
            25.0,
            HitLocation::LEG_LEFT,
            100.0,
        );
        assert!(hit.killed);
        assert_eq!(hit.sever, Some(HitLocation::LEG_LEFT));
        assert!(is_severable(HitLocation::ARM_RIGHT));
        assert!(is_severable(HitLocation::TORSO_SPLIT));
        assert!(!is_severable(HitLocation::TORSO));
        assert!(!is_severable(HitLocation::HEAD));
    }

    #[test]
    fn gun_mag_depletes_then_reloads() {
        let gun = Firearm::pistol_tacmaster1911();
        approx(gun.shot_interval(), 0.1);
        let mut gs = GunState::new(7, 21);

        for _ in 0..7 {
            assert_eq!(gs.fire(&gun, 1.0), Some(200.0));
        }
        assert_eq!(gs.in_mag, 0);
        assert_eq!(gs.fire(&gun, 1.0), None); // empty

        assert!(gs.reload(&gun));
        assert_eq!(gs.in_mag, 7);
        assert_eq!(gs.reserve, 14);
        assert_eq!(gs.fire(&gun, 0.0), None); // still reloading
        gs.tick(0.5);
        assert_eq!(gs.fire(&gun, 0.0), Some(200.0));
        assert_eq!(gs.in_mag, 6);

        // Cannot reload a full magazine.
        let mut full = GunState::new(7, 21);
        assert!(!full.reload(&gun));
    }

    #[test]
    fn gun_cooldown_gates_rate_of_fire() {
        let gun = Firearm::ak47();
        let mut gs = GunState::new(30, 0);
        assert_eq!(gs.fire(&gun, 0.0), Some(300.0));
        assert_eq!(gs.fire(&gun, 0.0), None); // cooldown not elapsed
        assert_eq!(gs.fire(&gun, gun.shot_interval()), Some(300.0));
    }

    #[test]
    fn firearm_headshot_mult_and_range() {
        let mut gun = Firearm::ak47();
        approx(gun.damage_for(HitLocation::TORSO), 300.0);
        gun.headshot_mult = 2.5;
        approx(gun.damage_for(HitLocation::HEAD), 750.0);
        assert!(gun.in_range(32.0));
        assert!(!gun.in_range(33.0));
    }

    #[test]
    fn bleed_drains_hp_over_time() {
        let mut v = Vitals::new(500.0, 100.0);
        v.apply_hit(50.0, HitLocation::TORSO);
        approx(v.hp, 450.0);
        approx(v.bleed, 5.0); // 10% of 50
        v.tick(2.0);
        approx(v.hp, 440.0); // 5 bleed * 2 s
        approx(v.bleed, 1.0); // recovered 2/s
        v.tick(1.0);
        approx(v.bleed, 0.0);
        approx(v.hp, 439.0);
    }

    #[test]
    fn limb_hit_lightens_and_head_hit_heavies() {
        let mut torso = Vitals::new(200.0, 100.0);
        approx(torso.apply_hit(40.0, HitLocation::TORSO), 40.0);
        let mut limb = Vitals::new(200.0, 100.0);
        approx(limb.apply_hit(40.0, HitLocation::ARM_LEFT), 20.0);
        let mut head = Vitals::new(200.0, 100.0);
        approx(head.apply_hit(40.0, HitLocation::HEAD), 80.0);
        assert!(limb.injuries & HitLocation::ARM_LEFT != 0);
        assert!(head.injuries & HitLocation::HEAD != 0);
    }

    #[test]
    fn stamina_regenerates_up_to_max() {
        let mut v = Vitals::new(200.0, 100.0);
        v.stamina = 40.0;
        v.tick(1.0);
        approx(v.stamina, 55.0);
        v.tick(10.0);
        approx(v.stamina, 100.0); // clamped
    }

    #[test]
    fn revive_heals_and_heal_clamps() {
        let mut v = Vitals::new(200.0, 100.0);
        v.apply_hit(500.0, HitLocation::TORSO);
        assert!(v.is_downed());
        assert!(v.bleed > 0.0);
        v.revive();
        approx(v.hp, 100.0);
        approx(v.bleed, 0.0);
        assert!(!v.is_downed());
        approx(v.heal(1000.0), 100.0);
        approx(v.hp, 200.0);
    }

    #[test]
    fn reaction_for_dismember_and_death() {
        assert_eq!(
            reaction_for(HitLocation::LEG_LEFT, 0.9, true),
            Reaction::Dismember(HitLocation::LEG_LEFT)
        );
        assert_eq!(reaction_for(HitLocation::HEAD, 1.0, true), Reaction::Death);
        assert_eq!(reaction_for(HitLocation::TORSO, 0.9, true), Reaction::Death);
        // A lethal limb graze is still a death, not a dismember.
        assert_eq!(reaction_for(HitLocation::LEG_LEFT, 0.1, true), Reaction::Death);
        assert_eq!(reaction_for(HitLocation::TORSO, 0.5, false), Reaction::Stumble);
        assert_eq!(reaction_for(HitLocation::TORSO, 0.1, false), Reaction::Flinch);
    }

    #[test]
    fn zombie_attack_scaling_is_sane() {
        let grunt = zombie_attack_damage(ZombieKind::Grunt);
        let feral = zombie_attack_damage(ZombieKind::Feral);
        let jug = zombie_attack_damage(ZombieKind::Juggernaut);
        assert!(grunt > 0.0 && feral > grunt && jug > feral);
        // A grunt needs many bites to down a survivor at full player HP.
        assert!(grunt * 10.0 < PLAYER_MAX_HEALTH);
        assert!(zombie_melee_cooldown(ZombieKind::Feral) > 0.0);
    }
}
