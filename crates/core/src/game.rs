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

/// Stockpile capacity per resource for a home tier, taken verbatim from the
/// shipped `RTS/HomeSiteLimits.xml` (`libs/class3/rts/homesitelimits.xml`).
///
/// The runtime reads the same table natively: `NSHome` indexes a stride-0x34
/// record table at `DAT_0190dc10` (base `+0x04` = Ammo cap, `+0x08` = Food,
/// `+0x0C` = Fuel, `+0x10` = Materials, `+0x14` = Medicine; see
/// `notes/community.md` §3.2 / §7 and `notes/ghidra/rts-sim.md` §3.3). The
/// earlier port wrongly doubled Food on every tier above RV and left Materials
/// undoubled. Storage facilities add +20 to every resource on top of these caps.
///
/// Values:
/// ```text
///          Ammo Food Fuel Materials Medicine
/// RV         25   25   25        25       25
/// Tiny       25   25   25        50       25
/// Small      50   50   50       100       50
/// Medium     80   80   80       160       80
/// Large     100  100  100       200      100
/// ```
/// The real-data test `home_tier_caps_match_homesitelimits_xml` re-parses the
/// shipped XML through the VFS and asserts this table matches field-for-field.
pub fn home_tier_caps(tier: HomeTier) -> [f32; RESOURCE_COUNT] {
    // Order: Ammo, Food, Fuel, Materials, Medicine (canonical C++ stockpile order).
    match tier {
        HomeTier::Rv => [25.0, 25.0, 25.0, 25.0, 25.0],
        HomeTier::Tiny => [25.0, 25.0, 25.0, 50.0, 25.0],
        HomeTier::Small => [50.0, 50.0, 50.0, 100.0, 50.0],
        HomeTier::Medium => [80.0, 80.0, 80.0, 160.0, 80.0],
        HomeTier::Large => [100.0, 100.0, 100.0, 200.0, 100.0],
    }
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
        // `homesitelimits.xml`: Tiny Food cap = 25 (not doubled); Storage adds +20.
        assert_eq!(s.caps[Resource::Food.index()], 45.0);
        assert_eq!(s.caps[Resource::Materials.index()], 70.0); // 50 + 20
    }

    #[test]
    fn home_tier_caps_are_the_shipped_table() {
        // Hard-coded expectation mirroring `libs/class3/rts/homesitelimits.xml`.
        assert_eq!(home_tier_caps(HomeTier::Rv), [25.0, 25.0, 25.0, 25.0, 25.0]);
        assert_eq!(home_tier_caps(HomeTier::Tiny), [25.0, 25.0, 25.0, 50.0, 25.0]);
        assert_eq!(home_tier_caps(HomeTier::Small), [50.0, 50.0, 50.0, 100.0, 50.0]);
        assert_eq!(home_tier_caps(HomeTier::Medium), [80.0, 80.0, 80.0, 160.0, 80.0]);
        assert_eq!(home_tier_caps(HomeTier::Large), [100.0, 100.0, 100.0, 200.0, 100.0]);
    }

    /// Real-data parity: read `homesitelimits.xml` through the VFS and assert the
    /// parsed `Max{Ammo,Food,Fuel,Materials,Medicine}` match `home_tier_caps`
    /// exactly for all five tiers. Skipped when the (gitignored) extraction is
    /// not checked out next to the repo so the suite stays self-contained.
    #[test]
    fn home_tier_caps_match_homesitelimits_xml() {
        use crate::Vfs;
        use quick_xml::events::Event;
        use quick_xml::Reader;

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/gamedata");
        if !root.join("libs/class3/rts/homesitelimits.xml").is_file() {
            return;
        }
        let mut vfs = Vfs::new();
        vfs.set_loose_root(&root);
        let bytes = vfs
            .read("libs/class3/rts/homesitelimits.xml")
            .expect("homesitelimits.xml via VFS");

        let mut reader = Reader::from_reader(bytes.as_slice());
        let mut buf = Vec::new();
        let mut size: Option<String> = None;
        let mut field: Option<String> = None;
        // size-name -> [ammo, food, fuel, materials, medicine]
        let mut parsed: Vec<(String, [f32; RESOURCE_COUNT])> = Vec::new();
        let mut cur = [0.0f32; RESOURCE_COUNT];
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => match e.name().as_ref() {
                    b"HomeSiteLimit" => {
                        for a in e.attributes().flatten() {
                            if a.key.as_ref() == b"Size" {
                                size = Some(String::from_utf8_lossy(&a.value).into_owned());
                            }
                        }
                        cur = [0.0; RESOURCE_COUNT];
                    }
                    b"MaxAmmo" | b"MaxFood" | b"MaxFuel" | b"MaxMaterials" | b"MaxMedicine" => {
                        field = Some(String::from_utf8_lossy(e.name().as_ref()).into_owned());
                    }
                    _ => {}
                },
                Ok(Event::Text(t)) => {
                    if size.is_some() {
                        if let Some(field) = field.as_deref() {
                            let v: f32 = t.unescape().unwrap().trim().parse().unwrap();
                            let idx = match field {
                                "MaxAmmo" => 0,
                                "MaxFood" => 1,
                                "MaxFuel" => 2,
                                "MaxMaterials" => 3,
                                "MaxMedicine" => 4,
                                _ => unreachable!(),
                            };
                            cur[idx] = v;
                        }
                    }
                }
                Ok(Event::End(e)) => match e.name().as_ref() {
                    b"MaxAmmo" | b"MaxFood" | b"MaxFuel" | b"MaxMaterials" | b"MaxMedicine" => {
                        field = None;
                    }
                    b"HomeSiteLimit" => {
                        if let Some(s) = size.take() {
                            parsed.push((s, cur));
                        }
                    }
                    _ => {}
                },
                Ok(Event::Eof) => break,
                Err(e) => panic!("homesitelimits.xml parse error: {e}"),
                _ => {}
            }
            buf.clear();
        }

        let tier_of = |name: &str| match name {
            "RV" => HomeTier::Rv,
            "Tiny" => HomeTier::Tiny,
            "Small" => HomeTier::Small,
            "Medium" => HomeTier::Medium,
            "Large" => HomeTier::Large,
            other => panic!("unknown home size {other}"),
        };
        assert_eq!(parsed.len(), 5, "five HomeSiteLimit rows");
        for (name, caps) in &parsed {
            assert_eq!(
                home_tier_caps(tier_of(name)),
                *caps,
                "tier {name} caps must match homesitelimits.xml"
            );
        }
    }
}
