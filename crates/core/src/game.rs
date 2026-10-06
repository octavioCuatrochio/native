//! Game rules / simulation data recovered from the shipped Lua and the Ghidra
//! decompilation of the proprietary `Class3` module.
//!
//! Sources:
//!  * `notes/ai-combat.md` (zombie stats, melee numbers, damage application Lua).
//!  * `notes/ghidra/rts-sim.md` (resource model, caps, action opcodes).
//!  * `notes/ghidra/combat.md` (CHitInfo transport, sever/location enums).
//!  * `notes/ghidra/zombie-runtime.md` (two-tier spawn model).
//!  * `notes/inventory-community.md` (items, home tiers).
//!
//! Values are best-effort parity; where the exact formula lives in C++ that was
//! not in the decompiled subset, the source is noted.

/// The five community resources, in the game's canonical order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resource {
    Ammo = 0,
    Food = 1,
    Fuel = 2,
    Materials = 3,
    Medicine = 4,
}

pub const RESOURCE_COUNT: usize = 5;

impl Resource {
    pub fn name(self) -> &'static str {
        match self {
            Resource::Ammo => "Ammo",
            Resource::Food => "Food",
            Resource::Fuel => "Fuel",
            Resource::Materials => "Materials",
            Resource::Medicine => "Medicine",
        }
    }
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Home-site tiers (from `RTS/HomeSiteLimits.xml`, per `rts-sim.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HomeTier {
    Rv,
    Tiny,
    Small,
    Medium,
    Large,
}

/// Stockpile capacity per resource for a home tier. The recovered table gives a
/// single cap for most resources with Food doubled on the larger tiers; Storage
/// facilities add +20 to every resource. Best-effort where the exact per-resource
/// table was not fully pinned.
pub fn home_tier_caps(tier: HomeTier) -> [f32; RESOURCE_COUNT] {
    let food_mult = match tier {
        HomeTier::Rv => 1.0,
        HomeTier::Tiny => 2.0,
        HomeTier::Small => 2.0,
        HomeTier::Medium => 2.0,
        HomeTier::Large => 2.0,
    };
    let base = match tier {
        HomeTier::Rv => 25.0,
        HomeTier::Tiny => 25.0,
        HomeTier::Small => 50.0,
        HomeTier::Medium => 80.0,
        HomeTier::Large => 100.0,
    };
    // Order: Ammo, Food, Fuel, Materials, Medicine
    [base, base * food_mult, base, base, base]
}

/// A community stockpile with clamped add/remove.
#[derive(Debug, Clone)]
pub struct Stockpile {
    pub counts: [f32; RESOURCE_COUNT],
    pub caps: [f32; RESOURCE_COUNT],
}

impl Stockpile {
    pub fn new(tier: HomeTier) -> Self {
        Stockpile { counts: [0.0; RESOURCE_COUNT], caps: home_tier_caps(tier) }
    }
    /// Add and return the amount actually stored (>= 0).
    pub fn add(&mut self, r: Resource, amount: f32) -> f32 {
        let i = r.index();
        let before = self.counts[i];
        self.counts[i] = (self.counts[i] + amount).clamp(0.0, self.caps[i]);
        self.counts[i] - before
    }
    /// Remove and return the amount actually removed (>= 0).
    pub fn remove(&mut self, r: Resource, amount: f32) -> f32 {
        let i = r.index();
        let before = self.counts[i];
        self.counts[i] = (self.counts[i] - amount).clamp(0.0, self.caps[i]);
        before - self.counts[i]
    }
    pub fn apply_storage_bonus(&mut self, per_resource: f32) {
        for c in self.caps.iter_mut() {
            *c += per_resource;
        }
    }
}

/// Zombie kinds present in the game, with stats from `ai-combat.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZombieKind {
    Grunt,
    Screamer,
    Feral,
    ArmyZombie,
    Bloater,
    Juggernaut,
}

#[derive(Debug, Clone, Copy)]
pub struct ZombieStats {
    pub hp: f32,
    pub walk_speed: f32,
    pub run_speed: f32,
    /// Melee reach (m). `melee.damageRadius` = 1.1 for all zombies (Lua).
    pub attack_range: f32,
    /// Body-damage multiplier (Juggernaut 0.5) and damage floor (Juggernaut 2).
    pub body_damage_mult: f32,
    pub damage_floor: f32,
    /// Killed by vehicle collision (KillableByCollision).
    pub killable_by_collision: bool,
}

impl ZombieKind {
    pub fn stats(self) -> ZombieStats {
        match self {
            ZombieKind::Grunt => ZombieStats { hp: 25.0, walk_speed: 0.98, run_speed: 3.68, attack_range: 1.1, body_damage_mult: 1.0, damage_floor: 0.0, killable_by_collision: true },
            ZombieKind::Screamer => ZombieStats { hp: 100.0, walk_speed: 0.95, run_speed: 0.95, attack_range: 1.1, body_damage_mult: 1.0, damage_floor: 0.0, killable_by_collision: false },
            ZombieKind::Feral => ZombieStats { hp: 250.0, walk_speed: 3.68, run_speed: 7.26, attack_range: 1.1, body_damage_mult: 1.0, damage_floor: 0.0, killable_by_collision: false },
            ZombieKind::ArmyZombie => ZombieStats { hp: 250.0, walk_speed: 0.98, run_speed: 4.5, attack_range: 1.1, body_damage_mult: 1.0, damage_floor: 0.0, killable_by_collision: false },
            ZombieKind::Bloater => ZombieStats { hp: 1000.0, walk_speed: 2.0, run_speed: 2.0, attack_range: 1.1, body_damage_mult: 1.0, damage_floor: 0.0, killable_by_collision: false },
            ZombieKind::Juggernaut => ZombieStats { hp: 2000.0, walk_speed: 2.49, run_speed: 2.49, attack_range: 1.1, body_damage_mult: 0.5, damage_floor: 2.0, killable_by_collision: false },
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            ZombieKind::Grunt => "Grunt",
            ZombieKind::Screamer => "Screamer",
            ZombieKind::Feral => "Feral",
            ZombieKind::ArmyZombie => "ArmyZombie",
            ZombieKind::Bloater => "Bloater",
            ZombieKind::Juggernaut => "Juggernaut",
        }
    }
}

/// Hit location bits (from `combat.md`, `cs_location` enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HitLocation;
impl HitLocation {
    pub const ARM_RIGHT: u32 = 1;
    pub const ARM_LEFT: u32 = 2;
    pub const HEAD: u32 = 4;
    pub const LEG_LEFT: u32 = 8;
    pub const LEG_RIGHT: u32 = 16;
    pub const TORSO: u32 = 32;
    pub const TORSO_SPLIT: u32 = 64;
}

/// Melee weapon damage (from `singleplayer.lua` per `ai-combat.md`).
pub mod melee {
    pub const FISTS: f32 = 80.0;
    pub const MACHETE: f32 = 80.0;
    pub const SLEDGEHAMMER: f32 = 80.0;
}

/// Apply damage to a zombie, mirroring the Lua `ProcessActorDamage` model:
/// body damage is scaled by `body_damage_mult` with a floor; headshots do full
/// damage (except the Bloater headshot-kill RTS flag). Returns remaining HP.
pub fn apply_zombie_damage(hp: f32, kind: ZombieKind, base: f32, location: u32, bloater_headshot_kill: bool) -> f32 {
    let st = kind.stats();
    let head = location & HitLocation::HEAD != 0;
    let mut dmg = base;
    if kind == ZombieKind::Bloater && head && bloater_headshot_kill {
        return 0.0;
    }
    if !head {
        dmg = (base * st.body_damage_mult).max(st.damage_floor);
    }
    (hp - dmg).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machete_one_hits_grunt() {
        assert_eq!(apply_zombie_damage(25.0, ZombieKind::Grunt, melee::MACHETE, HitLocation::TORSO, false), 0.0);
    }

    #[test]
    fn juggernaut_body_damage_is_halved_with_floor() {
        // 2000 HP juggernaut, 80 dmg body -> 40 per hit (mult 0.5)
        let hp = apply_zombie_damage(2000.0, ZombieKind::Juggernaut, 80.0, HitLocation::TORSO, false);
        assert_eq!(hp, 1960.0);
        // tiny damage floored to 2
        let hp2 = apply_zombie_damage(2000.0, ZombieKind::Juggernaut, 1.0, HitLocation::TORSO, false);
        assert_eq!(hp2, 1998.0);
    }

    #[test]
    fn stockpile_clamps_to_caps() {
        let mut s = Stockpile::new(HomeTier::Rv);
        let stored = s.add(Resource::Ammo, 100.0);
        assert_eq!(stored, 25.0);
        assert_eq!(s.counts[Resource::Ammo.index()], 25.0);
        let removed = s.remove(Resource::Ammo, 10.0);
        assert_eq!(removed, 10.0);
        assert_eq!(s.counts[Resource::Ammo.index()], 15.0);
    }

    #[test]
    fn storage_bonus_raises_caps() {
        let mut s = Stockpile::new(HomeTier::Tiny);
        s.apply_storage_bonus(20.0);
        assert_eq!(s.caps[Resource::Food.index()], 70.0); // 25*2 + 20
    }
}
