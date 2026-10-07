//! Player weapon definitions, read from the shipped game data.
//!
//! The original game keeps the per-weapon numbers in the item scripts, not in
//! native code: the fire modes (damage, rate, clip, reload, ammo type, sound
//! radius) live in `scripts/entities/items/xml/weapons/<name>.xml`
//! (`notes/ai-combat.md` §4.5), and the condition/expertise data
//! (`DurabilityMax`, `DurabilityCriticalMin/Max`, `DurabilityRating`,
//! `Expertise`) lives in `libs/class3/items/items.xml`
//! (`notes/inventory-community.md` §1.2/§1.3).
//!
//! This module parses both at boot so the native player model is driven by the
//! same numbers the original scripts use. The recovered headline weapons are
//! also available as built-in constants so the model still works when the data
//! is unavailable (tests, `SOD_CRY=0`).
//!
//! Only numeric values are recovered from the data; no XML text is copied into
//! this crate.

use std::collections::HashMap;
use std::sync::OnceLock;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// Fallback melee weapon when no data is loaded (`weapons/fists.xml`).
pub const DEFAULT_MELEE_WEAPON: &str = "Machete";
/// Fallback firearm (`pistol_tacmaster1911.xml`).
pub const DEFAULT_FIREARM: &str = "Pistol_TacMaster1911";

// ---------------------------------------------------------------------------
// Definition types
// ---------------------------------------------------------------------------

/// Melee fire-mode parameters (`<firemode type="Melee"><melee>`), verbatim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeleeParams {
    /// Base melee damage (`melee.damage`). Every shipped melee is 80.
    pub damage: f32,
    /// Attack impulse (`melee.impulse`), used by the reaction/ragdoll model.
    pub impulse: f32,
    /// Wind-up before the hit test fires (`melee.delay`), seconds.
    pub delay: f32,
    /// Swing clip duration (`melee.duration`), seconds.
    pub duration: f32,
}

impl Default for MeleeParams {
    fn default() -> Self {
        MeleeParams { damage: 0.0, impulse: 0.0, delay: 0.0, duration: 0.0 }
    }
}

/// Ranged fire-mode parameters (`<firemode type="default"><fire>`), verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct FireParams {
    /// `ammo_type` item id, e.g. `bullet_45cal`.
    pub ammo_type: String,
    /// Rounds per minute (`rate`).
    pub rate_rpm: f32,
    /// Reload duration (`reload_time`), seconds.
    pub reload_time: f32,
    /// Magazine capacity (`clip_size`).
    pub clip_size: i32,
    /// Base damage at/inside the full-damage band (`damage`).
    pub damage: f32,
    /// Distance before damage starts dropping (`damage_drop_min_distance`), m.
    pub drop_min_distance: f32,
    /// Damage lost per metre beyond `drop_min_distance` (`damage_drop_per_meter`).
    pub drop_per_meter: f32,
    /// Rounds moved into the chamber on reload (`bullet_chamber`).
    pub bullet_chamber: i32,
}

impl Default for FireParams {
    fn default() -> Self {
        FireParams {
            ammo_type: String::new(),
            rate_rpm: 0.0,
            reload_time: 0.0,
            clip_size: 0,
            damage: 0.0,
            drop_min_distance: 0.0,
            drop_per_meter: 0.0,
            bullet_chamber: 0,
        }
    }
}

/// One `<item>` weapon script: identity, the `params` block, the AI
/// descriptor and the fire modes. A weapon is a firearm when it has a
/// [`FireParams`], a melee weapon when it has a [`MeleeParams`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WeaponDef {
    /// Root `<item name>` — also the `EntityClass` in `items.xml`.
    pub name: String,
    /// Root `class` attribute (`Weapon`, `Fists`, ...).
    pub class_name: String,
    /// Root `category` attribute (`pistol`, `rifle`, `small`, ...).
    pub category: String,
    /// Localised `display_name` param.
    pub display_name: String,
    /// Reserve ammo capacity (`ammoMax`).
    pub ammo_max: i32,
    /// `disturbanceValue`: the AI disturbance this weapon causes.
    pub disturbance: f32,
    /// Unsafe sound radius (`soundRange`), metres — the zombie-hearing range.
    pub sound_range: f32,
    /// Silenced sound radius (`soundRangeSilenced`), metres.
    pub sound_range_silenced: f32,
    /// Draw/holster time (`draw_time`), seconds.
    pub draw_time: f32,
    /// `ai_descriptor` burst count bounds.
    pub burst_min: i32,
    pub burst_max: i32,
    /// `ai_descriptor` burst pause bounds, seconds.
    pub burst_pause_min: f32,
    pub burst_pause_max: f32,
    /// `ai_descriptor` minimum trigger time between single shots, seconds.
    pub single_fire_trigger: f32,
    /// Melee fire mode, if any.
    pub melee: Option<MeleeParams>,
    /// Ranged fire mode, if any (the `default` mode).
    pub fire: Option<FireParams>,
}

impl WeaponDef {
    pub fn is_firearm(&self) -> bool {
        self.fire.is_some()
    }

    pub fn is_melee(&self) -> bool {
        self.melee.is_some() && self.fire.is_none()
    }

    /// Display name, falling back to the script name.
    pub fn title(&self) -> &str {
        if self.display_name.is_empty() { &self.name } else { &self.display_name }
    }

    /// Minimum seconds between shots (`rate`, 60/rate).
    pub fn shot_interval(&self) -> f32 {
        match &self.fire {
            Some(f) if f.rate_rpm > 0.0 => 60.0 / f.rate_rpm,
            Some(_) => 0.1,
            None => 0.0,
        }
    }

    /// Magazine capacity, 0 for melee.
    pub fn clip_size(&self) -> i32 {
        self.fire.as_ref().map(|f| f.clip_size.max(0)).unwrap_or(0)
    }

    /// Base melee damage (`0` for a firearm).
    pub fn melee_damage(&self) -> f32 {
        self.melee.as_ref().map(|m| m.damage).unwrap_or(0.0)
    }

    /// Damage of a ranged shot after the linear distance falloff: full damage
    /// inside `damage_drop_min_distance`, then `damage_drop_per_meter` per
    /// metre, clamped at 0.
    pub fn damage_at(&self, distance: f32) -> f32 {
        match &self.fire {
            Some(f) => {
                let over = (distance - f.drop_min_distance).max(0.0);
                (f.damage - f.drop_per_meter * over).max(0.0)
            }
            None => self.melee_damage(),
        }
    }

    /// Zombie-hearing radius for an unsilenced shot (`soundRange`).
    pub fn noise_radius(&self) -> f32 {
        self.sound_range
    }

    /// Zombie-hearing radius for a silenced shot (`soundRangeSilenced`).
    pub fn silenced_noise_radius(&self) -> f32 {
        self.sound_range_silenced
    }
}

// ---------------------------------------------------------------------------
// Condition / expertise data (items.xml)
// ---------------------------------------------------------------------------

/// Weapon condition bounds from a `WeaponDefinition`/`GunDefinition`. Current
/// condition is the item entity's health (`notes/inventory-community.md` §1.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Durability {
    /// New-item condition (`DurabilityMax`).
    pub max: f32,
    /// Floor condition (`DurabilityMin`).
    pub min: f32,
    /// Condition band where "critical" begins (`DurabilityCriticalMin`).
    pub critical_min: f32,
    /// Condition band where "critical" ends (`DurabilityCriticalMax`).
    pub critical_max: f32,
    /// Wear multiplier (`DurabilityRating`).
    pub rating: f32,
}

impl Default for Durability {
    fn default() -> Self {
        Durability { max: 100.0, min: 0.0, critical_min: 0.0, critical_max: 0.0, rating: 1.0 }
    }
}

/// Per-entity condition/expertise record from `items.xml`.
#[derive(Debug, Clone, PartialEq)]
pub struct WeaponItemInfo {
    /// `EntityClass` (`Machete`, `Pistol_TacMaster1911`, ...).
    pub entity_class: String,
    /// `Expertise` / `WeaponType` (`EdgedLightBasic`, `Pistol Basic`, ...).
    pub expertise: String,
    /// True for a `GunDefinition`.
    pub is_gun: bool,
    /// Condition bounds.
    pub durability: Durability,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// All weapon scripts plus the per-entity condition rows, keyed by their
/// lower-cased name/entity class.
#[derive(Debug, Clone, Default)]
pub struct WeaponData {
    defs: HashMap<String, WeaponDef>,
    items: HashMap<String, WeaponItemInfo>,
}

impl WeaponData {
    /// Recovered headline weapons. These are the shipped numbers (numeric
    /// only) so the model still runs without the paks.
    pub fn builtin() -> Self {
        let mut d = WeaponData::default();
        // Fists (`weapons/fists.xml`): melee 80, impulse 100, delay 0.5 s.
        d.add_def(WeaponDef {
            name: "Fists".into(),
            class_name: "Fists".into(),
            category: "small".into(),
            display_name: "Fists".into(),
            melee: Some(MeleeParams { damage: 80.0, impulse: 100.0, delay: 0.5, duration: 0.85 }),
            ..Default::default()
        });
        // Machete (`weapons/meleemachete02.xml`, items.xml `WeaponDefinition`).
        d.add_def(WeaponDef {
            name: "Machete".into(),
            class_name: "Fists".into(),
            category: "small".into(),
            display_name: "Rusty Machete".into(),
            melee: Some(MeleeParams { damage: 80.0, impulse: 0.0, delay: 0.0, duration: 0.0 }),
            ..Default::default()
        });
        // Sledgehammer (`weapons/meleesledgehammer.xml`).
        d.add_def(WeaponDef {
            name: "Sledgehammer".into(),
            class_name: "Fists".into(),
            category: "small".into(),
            display_name: "Sledgehammer".into(),
            melee: Some(MeleeParams { damage: 80.0, impulse: 0.0, delay: 0.0, duration: 0.0 }),
            ..Default::default()
        });
        // TacMaster 1911 (`weapons/pistol_tacmaster1911.xml`).
        d.add_def(WeaponDef {
            name: "Pistol_TacMaster1911".into(),
            class_name: "Weapon".into(),
            category: "pistol".into(),
            display_name: "Tac Master 1911".into(),
            ammo_max: 120,
            disturbance: 5.0,
            sound_range: 30.0,
            sound_range_silenced: 5.0,
            draw_time: 0.35,
            burst_min: 1,
            burst_max: 2,
            burst_pause_min: 0.5,
            burst_pause_max: 1.5,
            single_fire_trigger: 0.5,
            fire: Some(FireParams {
                ammo_type: "bullet_45cal".into(),
                rate_rpm: 600.0,
                reload_time: 0.5,
                clip_size: 7,
                damage: 200.0,
                drop_min_distance: 20.0,
                drop_per_meter: 10.0,
                bullet_chamber: 1,
            }),
            ..Default::default()
        });
        // AK47 (`weapons/assaultrifle_ak47.xml`).
        d.add_def(WeaponDef {
            name: "AssaultRifle_AK47".into(),
            class_name: "Weapon".into(),
            category: "assaultrifle".into(),
            display_name: "AK47".into(),
            ammo_max: 180,
            disturbance: 10.0,
            sound_range: 50.0,
            sound_range_silenced: 5.0,
            fire: Some(FireParams {
                ammo_type: "bullet_762mm".into(),
                rate_rpm: 760.0,
                reload_time: 1.2,
                clip_size: 30,
                damage: 300.0,
                drop_min_distance: 32.0,
                drop_per_meter: 15.0,
                bullet_chamber: 1,
            }),
            ..Default::default()
        });
        // Condition rows (`items.xml`).
        d.add_item(WeaponItemInfo {
            entity_class: "Fists".into(),
            expertise: "BluntLightBasic".into(),
            is_gun: false,
            durability: Durability { max: 1000.0, min: 0.0, critical_min: 0.0, critical_max: 0.0, rating: 1.0 },
        });
        d.add_item(WeaponItemInfo {
            entity_class: "Machete".into(),
            expertise: "EdgedLightBasic".into(),
            is_gun: false,
            durability: Durability { max: 60.0, min: 20.0, critical_min: 10.0, critical_max: 40.0, rating: 0.4 },
        });
        d.add_item(WeaponItemInfo {
            entity_class: "Sledgehammer".into(),
            expertise: "BluntHeavyBasic".into(),
            is_gun: false,
            durability: Durability { max: 200.0, min: 140.0, critical_min: 20.0, critical_max: 40.0, rating: 1.0 },
        });
        d.add_item(WeaponItemInfo {
            entity_class: "Pistol_TacMaster1911".into(),
            expertise: "Pistol Basic".into(),
            is_gun: true,
            durability: Durability { max: 265.0, min: 75.0, critical_min: 15.0, critical_max: 35.0, rating: 0.4 },
        });
        d
    }

    /// Load every weapon script under `scripts/entities/items/xml/weapons/`
    /// plus the condition rows from `libs/class3/items/items.xml`, on top of
    /// the built-in headline weapons. Missing data leaves the built-ins.
    pub fn load(vfs: &crate::crypak::Vfs) -> Self {
        let mut data = WeaponData::builtin();
        for p in vfs.paths_with_ext(".xml") {
            let norm = p.replace('\\', "/");
            if !norm.to_ascii_lowercase().contains("entities/items/xml/weapons/") {
                continue;
            }
            if let Some(bytes) = vfs.read(&p) {
                if let Some(def) = parse_weapon_xml(&bytes) {
                    data.add_def(def);
                }
            }
        }
        for path in ["libs/class3/items/items.xml", "gamedata/libs/class3/items/items.xml"] {
            if let Some(bytes) = vfs.read(path) {
                parse_items_xml(&bytes, &mut data);
                break;
            }
        }
        data
    }

    pub fn add_def(&mut self, def: WeaponDef) {
        if !def.name.is_empty() {
            self.defs.insert(def.name.to_ascii_lowercase(), def);
        }
    }

    pub fn add_item(&mut self, info: WeaponItemInfo) {
        if !info.entity_class.is_empty() {
            self.items.insert(info.entity_class.to_ascii_lowercase(), info);
        }
    }

    /// Weapon script by root name, case-insensitive.
    pub fn get(&self, name: &str) -> Option<&WeaponDef> {
        self.defs.get(&name.to_ascii_lowercase())
    }

    /// Condition/expertise row by entity class, case-insensitive.
    pub fn item(&self, entity_class: &str) -> Option<&WeaponItemInfo> {
        self.items.get(&entity_class.to_ascii_lowercase())
    }

    pub fn def_count(&self) -> usize {
        self.defs.len()
    }

    pub fn item_count(&self) -> usize {
        self.items.len()
    }
}

static WEAPONS: OnceLock<WeaponData> = OnceLock::new();

/// The registry for `vfs`, loaded once per process.
pub fn data_for(vfs: &crate::crypak::Vfs) -> &'static WeaponData {
    WEAPONS.get_or_init(|| WeaponData::load(vfs))
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn attr_map(e: &BytesStart) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for a in e.attributes().flatten() {
        let k = String::from_utf8_lossy(a.key.as_ref()).to_ascii_lowercase();
        let v = a.unescape_value().map(|c| c.into_owned()).unwrap_or_default();
        out.push((k, v));
    }
    out
}

fn attr<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

fn f32_of(s: Option<&str>) -> f32 {
    s.map(|s| s.trim()).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0)
}

fn i32_of(s: Option<&str>) -> i32 {
    s.map(|s| s.trim())
        .and_then(|s| s.parse::<i32>().ok().or_else(|| s.parse::<f32>().ok().map(|f| f as i32)))
        .unwrap_or(0)
}

/// Parse one weapon script XML. Returns `None` for a file with no `<item>`.
pub fn parse_weapon_xml(xml: &[u8]) -> Option<WeaponDef> {
    let xml = xml.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let mut def = WeaponDef::default();
    let mut have_item = false;

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };
        let (e, is_empty) = match ev {
            Event::Start(e) => (e, false),
            Event::Empty(e) => (e, true),
            Event::End(_) => {
                stack.pop();
                continue;
            }
            _ => continue,
        };
        let name = String::from_utf8_lossy(e.name().as_ref()).to_ascii_lowercase();
        let attrs = attr_map(&e);
        apply_weapon_elem(&stack, &name, &attrs, &mut def, &mut have_item);
        if !is_empty {
            stack.push(name);
        }
    }

    if have_item && !def.name.is_empty() {
        Some(def)
    } else {
        None
    }
}

fn apply_weapon_elem(
    stack: &[String],
    name: &str,
    attrs: &[(String, String)],
    def: &mut WeaponDef,
    have_item: &mut bool,
) {
    match name {
        "item" => {
            *have_item = true;
            if let Some(v) = attr(attrs, "name") {
                def.name = v.to_string();
            }
            if let Some(v) = attr(attrs, "class") {
                def.class_name = v.to_string();
            }
            if let Some(v) = attr(attrs, "category") {
                def.category = v.to_string();
            }
        }
        "ai_descriptor" => {
            def.draw_time = f32_of(attr(attrs, "draw_time"));
            def.burst_min = i32_of(attr(attrs, "burstbulletcountmin"));
            def.burst_max = i32_of(attr(attrs, "burstbulletcountmax"));
            def.burst_pause_min = f32_of(attr(attrs, "burstpausetimemin"));
            def.burst_pause_max = f32_of(attr(attrs, "burstpausetimemax"));
            def.single_fire_trigger = f32_of(attr(attrs, "singlefiretriggertime"));
        }
        "param" => {
            let pname = attr(attrs, "name").unwrap_or("").to_ascii_lowercase();
            let value = attr(attrs, "value").unwrap_or("");
            match stack.last().map(|s| s.as_str()) {
                Some("fire") => apply_fire_param(def, &pname, value),
                Some("melee") => apply_melee_param(def, &pname, value),
                Some("params") if stack.len() >= 2 && stack[stack.len() - 2] == "item" => {
                    apply_item_param(def, &pname, value)
                }
                _ => {}
            }
        }
        _ => {}
    }
}

fn apply_item_param(def: &mut WeaponDef, name: &str, value: &str) {
    match name {
        "display_name" => def.display_name = value.to_string(),
        "ammomax" => def.ammo_max = i32_of(Some(value)),
        "disturbancevalue" => def.disturbance = f32_of(Some(value)),
        "soundrange" => def.sound_range = f32_of(Some(value)),
        "soundrangesilenced" => def.sound_range_silenced = f32_of(Some(value)),
        _ => {}
    }
}

fn apply_fire_param(def: &mut WeaponDef, name: &str, value: &str) {
    let f = def.fire.get_or_insert_with(FireParams::default);
    match name {
        "ammo_type" => f.ammo_type = value.to_string(),
        "rate" => f.rate_rpm = f32_of(Some(value)),
        "reload_time" => f.reload_time = f32_of(Some(value)),
        "clip_size" => f.clip_size = i32_of(Some(value)),
        "damage" => f.damage = f32_of(Some(value)),
        "damage_drop_min_distance" => f.drop_min_distance = f32_of(Some(value)),
        "damage_drop_per_meter" => f.drop_per_meter = f32_of(Some(value)),
        "bullet_chamber" => f.bullet_chamber = i32_of(Some(value)),
        _ => {}
    }
}

fn apply_melee_param(def: &mut WeaponDef, name: &str, value: &str) {
    let m = def.melee.get_or_insert_with(MeleeParams::default);
    match name {
        "damage" => m.damage = f32_of(Some(value)),
        "impulse" => m.impulse = f32_of(Some(value)),
        "delay" => m.delay = f32_of(Some(value)),
        "duration" => m.duration = f32_of(Some(value)),
        _ => {}
    }
}

/// Fill `data`'s condition rows from `items.xml` `WeaponDefinition` /
/// `GunDefinition` elements. Only the attributes are read.
pub fn parse_items_xml(xml: &[u8], data: &mut WeaponData) {
    let xml = xml.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };
        let (e, name) = match &ev {
            Event::Start(e) | Event::Empty(e) => {
                (e, String::from_utf8_lossy(e.name().as_ref()).to_ascii_lowercase())
            }
            _ => continue,
        };
        let is_gun = name == "gundefinition";
        if name != "weapondefinition" && !is_gun {
            continue;
        }
        let attrs = attr_map(e);
        let Some(class) = attr(&attrs, "entityclass").map(|s| s.to_string()) else { continue };
        let durability = Durability {
            max: f32_of(attr(&attrs, "durabilitymax")),
            min: f32_of(attr(&attrs, "durabilitymin")),
            critical_min: f32_of(attr(&attrs, "durabilitycriticalmin")),
            critical_max: f32_of(attr(&attrs, "durabilitycriticalmax")),
            rating: {
                let r = f32_of(attr(&attrs, "durabilityrating"));
                if r <= 0.0 { 1.0 } else { r }
            },
        };
        data.add_item(WeaponItemInfo {
            entity_class: class,
            expertise: attr(&attrs, "expertise").unwrap_or("").to_string(),
            is_gun,
            durability,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A hand-written fire-mode fixture (structure only; values are arbitrary).
    const FIXTURE: &str = r#"<?xml version="1.0"?>
<item name="TestGun" class="Weapon" category="rifle">
  <ai_descriptor burstBulletCountMin="2" burstBulletCountMax="3"
      burstPauseTimeMin="0.25" burstPauseTimeMax="0.75"
      singleFireTriggerTime="0.4" draw_time="0.3" />
  <params>
    <param name="display_name" value="Test Gun" />
    <param name="ammoMax" value="90" />
    <param name="disturbanceValue" value="7.5" />
    <param name="soundRange" value="42.0" />
    <param name="soundRangeSilenced" value="4.0" />
  </params>
  <firemodes>
    <firemode type="default">
      <fire>
        <param name="ammo_type" value="bullet_test" />
        <param name="rate" value="300" />
        <param name="reload_time" value="1.5" />
        <param name="clip_size" value="10" />
        <param name="damage" value="120" />
        <param name="damage_drop_min_distance" value="15" />
        <param name="damage_drop_per_meter" value="5.0" />
      </fire>
    </firemode>
  </firemodes>
</item>"#;

    const MELEE_FIXTURE: &str = r#"<item name="TestBlade" class="Fists" category="small">
  <params><param name="display_name" value="Test Blade" /></params>
  <firemodes>
    <firemode type="Melee"><melee><param name="damage" value="75" />
      <param name="impulse" value="90" /></melee></firemode>
  </firemodes>
</item>"#;

    #[test]
    fn parses_firemode_and_params() {
        let d = parse_weapon_xml(FIXTURE.as_bytes()).expect("weapon");
        assert_eq!(d.name, "TestGun");
        assert_eq!(d.category, "rifle");
        assert_eq!(d.display_name, "Test Gun");
        assert_eq!(d.ammo_max, 90);
        assert_eq!(d.noise_radius(), 42.0);
        assert_eq!(d.silenced_noise_radius(), 4.0);
        assert!(d.is_firearm() && !d.is_melee());
        let f = d.fire.as_ref().unwrap();
        assert_eq!(f.ammo_type, "bullet_test");
        assert_eq!(f.clip_size, 10);
        assert_eq!(f.damage, 120.0);
        assert_eq!(f.reload_time, 1.5);
        // 300 rpm -> 0.2 s between shots.
        assert!((d.shot_interval() - 0.2).abs() < 1e-6);
        // Damage drop starts at 15 m: 5.0 per metre beyond.
        assert_eq!(d.damage_at(15.0), 120.0);
        assert_eq!(d.damage_at(25.0), 70.0);
        assert_eq!(d.damage_at(1000.0), 0.0);
    }

    #[test]
    fn parses_melee_firemode() {
        let d = parse_weapon_xml(MELEE_FIXTURE.as_bytes()).expect("melee");
        assert!(d.is_melee() && !d.is_firearm());
        assert_eq!(d.melee_damage(), 75.0);
        let m = d.melee.as_ref().unwrap();
        assert_eq!(m.impulse, 90.0);
        assert_eq!(d.clip_size(), 0);
    }

    #[test]
    fn builtin_recovered_headline_values() {
        let d = WeaponData::builtin();
        let pistol = d.get("Pistol_TacMaster1911").unwrap();
        assert_eq!(pistol.fire.as_ref().unwrap().damage, 200.0);
        assert_eq!(pistol.clip_size(), 7);
        assert!((pistol.shot_interval() - 0.1).abs() < 1e-6);
        assert_eq!(pistol.noise_radius(), 30.0);
        let ak = d.get("assaultrifle_ak47").unwrap();
        assert_eq!(ak.fire.as_ref().unwrap().damage, 300.0);
        assert_eq!(ak.clip_size(), 30);
        assert_eq!(ak.noise_radius(), 50.0);
        assert_eq!(d.get("machete").unwrap().melee_damage(), 80.0);
        // Condition rows.
        let m = d.item("Machete").unwrap();
        assert_eq!(m.durability.max, 60.0);
        assert_eq!(m.expertise, "EdgedLightBasic");
        let tg = d.item("Pistol_TacMaster1911").unwrap();
        assert!(tg.is_gun);
        assert_eq!(tg.durability.max, 265.0);
    }

    #[test]
    fn parses_items_condition_rows() {
        let xml = r#"<ItemData><WeaponDefinitions>
            <WeaponDefinition DurabilityCriticalMax="40" DurabilityCriticalMin="20"
              DurabilityMax="120" DurabilityMin="60" DurabilityRating="0.6"
              EntityClass="Bat_Wood" Expertise="BluntLightBasic" />
            </WeaponDefinitions>
            <GunDefinitions>
              <GunDefinition DurabilityCriticalMax="30" DurabilityCriticalMin="10"
                DurabilityMax="275" DurabilityMin="85" DurabilityRating="0.4"
                EntityClass="Pistol_M9" Expertise="Pistol Basic" />
            </GunDefinitions></ItemData>"#;
        let mut data = WeaponData::default();
        parse_items_xml(xml.as_bytes(), &mut data);
        let bat = data.item("Bat_Wood").unwrap();
        assert!(!bat.is_gun);
        assert_eq!(bat.durability.max, 120.0);
        assert_eq!(bat.durability.min, 60.0);
        assert_eq!(bat.durability.rating, 0.6);
        let gun = data.item("Pistol_M9").unwrap();
        assert!(gun.is_gun);
        assert_eq!(gun.expertise, "Pistol Basic");
    }
}
