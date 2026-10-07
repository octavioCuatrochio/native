//! Player stamina and the weapon runtime bound to an actor.
//!
//! Stamina is a native actor stat (it does not appear in any shipped Lua):
//! `notes/ghidra/hud.md` §3.1 reads the actor's current/permanent/max stamina
//! from vtable slots `+0x7c/+0x80/+0x8c`, and `notes/ghidra/combat.md` §2.5
//! registers the debug cvars `pl_infinite_stamina` next to `pl_infinite_health`.
//! The magnitudes are shipped data in `libs/class3/actorconstants.xml`
//! (`DamageValues Stat="Stamina"`): sprint drain per level, regen per level,
//! the per-strike costs (`Stamina_Light_Normal` .. `Stamina_Massive_Normal` and
//! their `_Reduced` variants), dodge/massive costs and stamina DoTs.
//!
//! The weapon side joins [`crate::cry::weapons`] (what the XML says) with
//! per-actor mutable state (magazine, reserve, reload timer, condition) and the
//! stamina costs, so the app can replace the old hand-written approximation.

use std::sync::OnceLock;

use super::weapons::{self, WeaponData, WeaponDef};
use crate::crypak::Vfs;

/// Base stamina pool. The HUD's stamina bar and the `actorconstants` strike
/// costs (4..55) fit a 100-point pool; the xml does not pin it.
pub const STAMINA_MAX: f32 = 100.0;

/// Exhaustion clears once stamina recovers to this fraction of the pool. The
/// player state machine has explicit `MachineDriven.Combat.*.Exhausted.*`
/// states and the `Combat_NoStamina` / `HeroNoStamina` UI triggers, so the
/// exhausted flag is latched at 0 and released with hysteresis.
pub const EXHAUST_EXIT: f32 = 25.0;

/// A swing with insufficient stamina deals this fraction of its damage
/// (the SM's `Exhausted` branch).
pub const EXHAUSTED_DAMAGE_SCALE: f32 = 0.5;

/// Weapon condition lost per melee swing / ranged shot. The data gives the
/// condition bounds and a wear multiplier but no explicit per-use amount
/// (`WeaponUseDurability` is a data-less SM action), so this is the
/// `[BEST-EFFORT]` base.
pub const DURABILITY_WEAR_PER_USE: f32 = 1.0;

/// Melee cooldown when the script does not name a swing duration.
pub const DEFAULT_MELEE_INTERVAL: f32 = 0.6;

/// Regen pause after spending stamina (sprint/shots/swings), seconds.
pub const REGEN_DELAY: f32 = 0.35;

// ---------------------------------------------------------------------------
// Swing tiers (actorconstants stamina costs)
// ---------------------------------------------------------------------------

/// The `Stamina_*` cost tier a strike uses. Chosen from the weapon's
/// `Expertise` (`notes/inventory-community.md` §1.3) or a zombie damage profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwingTier {
    Light,
    Standard,
    Heavy,
    Special,
    Massive,
}

impl SwingTier {
    pub const COUNT: usize = 5;
    pub const ALL: [SwingTier; 5] = [
        SwingTier::Light,
        SwingTier::Standard,
        SwingTier::Heavy,
        SwingTier::Special,
        SwingTier::Massive,
    ];

    pub fn index(self) -> usize {
        match self {
            SwingTier::Light => 0,
            SwingTier::Standard => 1,
            SwingTier::Heavy => 2,
            SwingTier::Special => 3,
            SwingTier::Massive => 4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SwingTier::Light => "Light",
            SwingTier::Standard => "Standard",
            SwingTier::Heavy => "Heavy",
            SwingTier::Special => "Special",
            SwingTier::Massive => "Massive",
        }
    }

    /// Map a `WeaponDefinition.Expertise` / `GunDefinition.Expertise` string
    /// (`BluntLightBasic`, `EdgedHeavyBasic`, ...) to a cost tier. Unknown
    /// strings fall back to `Standard`.
    pub fn from_expertise(expertise: &str) -> SwingTier {
        let e = expertise.to_ascii_lowercase();
        if e.contains("heavy") {
            SwingTier::Heavy
        } else if e.contains("massive") {
            SwingTier::Massive
        } else if e.contains("special") {
            SwingTier::Special
        } else if e.contains("light") {
            SwingTier::Light
        } else {
            SwingTier::Standard
        }
    }
}

// ---------------------------------------------------------------------------
// Stamina constants (actorconstants.xml)
// ---------------------------------------------------------------------------

/// The recovered stamina magnitudes. Defaults are the level-0 shipped values;
/// [`StaminaConstants::load`] overrides them from
/// `libs/class3/actorconstants.xml`.
#[derive(Debug, Clone, PartialEq)]
pub struct StaminaConstants {
    /// `StaminaRegen_Level00` per second.
    pub regen: f32,
    /// `StaminaDegen_Sprint_Level00_Normal` per second.
    pub sprint_drain: f32,
    /// `..._Encumbered` per second.
    pub sprint_drain_encumbered: f32,
    /// `..._Marathon` per second.
    pub sprint_drain_marathon: f32,
    /// `..._MarathonEncumbered` per second.
    pub sprint_drain_marathon_encumbered: f32,
    /// `StaminaRegen_Stealth_Stopped` per second.
    pub stealth_stopped: f32,
    /// `StaminaRegen_Stealth_Moving` per second.
    pub stealth_moving: f32,
    /// Per-strike costs, indexed by [`SwingTier::index`].
    pub swing: [f32; SwingTier::COUNT],
    /// `_Reduced` per-strike costs (the exhausted/low-tier variant).
    pub swing_reduced: [f32; SwingTier::COUNT],
    /// `Stamina_Dodge_Normal`.
    pub dodge: f32,
    /// `Stamina_Dodge_Reduced`.
    pub dodge_reduced: f32,
    /// `StaminaDegen_Toxic` per second.
    pub toxic: f32,
    /// `StaminaGain_Item_Food_Normal` (instant food gain).
    pub food_gain: f32,
    /// `Game.RTS.StaminaTankHitSmall` — stamina lost to a small hit.
    pub tank_hit_small: f32,
    /// `Game.RTS.StaminaTankHitLarge` — stamina lost to a large hit.
    pub tank_hit_large: f32,
}

impl Default for StaminaConstants {
    fn default() -> Self {
        StaminaConstants {
            regen: 6.0,
            sprint_drain: 6.0,
            sprint_drain_encumbered: 15.0,
            sprint_drain_marathon: 3.0,
            sprint_drain_marathon_encumbered: 7.5,
            stealth_stopped: 16.0,
            stealth_moving: 12.0,
            swing: [4.0, 6.0, 12.0, 20.0, 55.0],
            swing_reduced: [2.85, 3.75, 7.0, 16.0, 45.0],
            dodge: 16.0,
            dodge_reduced: 10.0,
            toxic: 8.0,
            food_gain: 30.0,
            // `rtsdata.xml` `Game.RTS.StaminaTankHitSmall/Large`.
            tank_hit_small: 300.0,
            tank_hit_large: 600.0,
        }
    }
}

impl StaminaConstants {
    pub fn builtin() -> Self {
        Self::default()
    }

    /// Cost of one strike of `tier`; `reduced` picks the `_Reduced` variant.
    pub fn cost(&self, tier: SwingTier, reduced: bool) -> f32 {
        let i = tier.index();
        if reduced { self.swing_reduced[i] } else { self.swing[i] }
    }

    /// Load `libs/class3/actorconstants.xml` over the built-in defaults.
    pub fn load(vfs: &Vfs) -> Self {
        let mut c = StaminaConstants::default();
        for path in [
            "libs/class3/actorconstants.xml",
            "gamedata/libs/class3/actorconstants.xml",
        ] {
            if let Some(bytes) = vfs.read(path) {
                c.apply_xml(&bytes);
                break;
            }
        }
        c
    }

    /// Apply the `Stat="Stamina"` `DamageValues` rows.
    pub fn apply_xml(&mut self, xml: &[u8]) {
        use quick_xml::events::Event;
        use quick_xml::Reader;
        let xml = xml.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(xml);
        let mut reader = Reader::from_reader(xml);
        let mut buf = Vec::new();
        loop {
            let ev = match reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => break,
                Err(_) => break,
                Ok(ev) => ev,
            };
            let e = match &ev {
                Event::Start(e) | Event::Empty(e) => e,
                _ => continue,
            };
            if !e.name().as_ref().eq_ignore_ascii_case(b"DamageValues") {
                continue;
            }
            let mut stat = String::new();
            let mut name = String::new();
            let mut max = 0.0f32;
            let mut per_second = 0.0f32;
            for a in e.attributes().flatten() {
                let k = String::from_utf8_lossy(a.key.as_ref()).to_ascii_lowercase();
                let v = a.unescape_value().map(|c| c.into_owned()).unwrap_or_default();
                match k.as_str() {
                    "stat" => stat = v,
                    "name" => name = v,
                    "max" => max = v.trim().parse().unwrap_or(0.0),
                    "persecond" => per_second = v.trim().parse().unwrap_or(0.0),
                    _ => {}
                }
            }
            if !stat.eq_ignore_ascii_case("stamina") {
                continue;
            }
            // Instant strike/gain rows carry the amount in `Max`; drains and
            // regen carry their rate in `PerSecond`.
            let amount = if per_second.abs() > 0.0 { per_second } else { max };
            self.apply_row(&name, amount);
        }
    }

    fn apply_row(&mut self, name: &str, per_second: f32) {
        let n = name.to_ascii_lowercase();
        // Level-0 rows only; higher levels are character progression.
        if n == "staminaregen_level00" {
            self.regen = per_second;
        } else if n == "staminadegen_sprint_level00_normal" {
            self.sprint_drain = per_second;
        } else if n == "staminadegen_sprint_level00_encumbered" {
            self.sprint_drain_encumbered = per_second;
        } else if n == "staminadegen_sprint_level00_marathon" {
            self.sprint_drain_marathon = per_second;
        } else if n == "staminadegen_sprint_level00_marathonencumbered" {
            self.sprint_drain_marathon_encumbered = per_second;
        } else if n == "staminaregen_stealth_stopped" {
            self.stealth_stopped = per_second;
        } else if n == "staminaregen_stealth_moving" {
            self.stealth_moving = per_second;
        } else if n == "staminadegen_toxic" {
            self.toxic = per_second;
        } else if n == "staminagain_item_food_normal" {
            self.food_gain = per_second.max(0.0);
        } else if let Some((tier, reduced)) = parse_strike_row(&n) {
            if reduced {
                self.swing_reduced[tier.index()] = per_second;
            } else {
                self.swing[tier.index()] = per_second;
            }
        }
    }
}

/// `stamina_<tier>_normal` / `stamina_<tier>_reduced` -> (tier, reduced).
fn parse_strike_row(n: &str) -> Option<(SwingTier, bool)> {
    let rest = n.strip_prefix("stamina_")?;
    let (tier_s, reduced) = if let Some(t) = rest.strip_suffix("_normal") {
        (t, false)
    } else if let Some(t) = rest.strip_suffix("_reduced") {
        (t, true)
    } else {
        return None;
    };
    let tier = match tier_s {
        "light" => SwingTier::Light,
        "standard" => SwingTier::Standard,
        "heavy" => SwingTier::Heavy,
        "special" => SwingTier::Special,
        "massive" => SwingTier::Massive,
        _ => return None,
    };
    Some((tier, reduced))
}

static STAMINA: OnceLock<StaminaConstants> = OnceLock::new();

/// The stamina constants for `vfs`, loaded once per process.
pub fn constants_for(vfs: &Vfs) -> &'static StaminaConstants {
    STAMINA.get_or_init(|| StaminaConstants::load(vfs))
}

// ---------------------------------------------------------------------------
// Stamina state
// ---------------------------------------------------------------------------

/// Per-actor stamina. `max` is `STAMINA_MAX + bonus`; `exhausted` is latched
/// at zero and released at [`EXHAUST_EXIT`].
#[derive(Debug, Clone, PartialEq)]
pub struct StaminaState {
    pub current: f32,
    /// `Character.StaminaBonus` — permanent pool bonus.
    pub bonus: f32,
    pub exhausted: bool,
    /// Seconds before regen resumes after spending.
    pub regen_delay: f32,
    /// Set by the movement code each frame.
    pub sprinting: bool,
    pub encumbered: bool,
    pub marathon: bool,
    /// Stealth stance: uses the higher stealth regen rows.
    pub stealth: bool,
}

impl Default for StaminaState {
    fn default() -> Self {
        StaminaState {
            current: STAMINA_MAX,
            bonus: 0.0,
            exhausted: false,
            regen_delay: 0.0,
            sprinting: false,
            encumbered: false,
            marathon: false,
            stealth: false,
        }
    }
}

impl StaminaState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_total(&self) -> f32 {
        (STAMINA_MAX + self.bonus).max(1.0)
    }

    pub fn ratio(&self) -> f32 {
        (self.current / self.max_total()).clamp(0.0, 1.0)
    }

    /// Force the current value, clamped to the pool. Never latches exhaustion
    /// (scripts use this for save/restore).
    pub fn set(&mut self, v: f32) {
        self.current = v.clamp(0.0, self.max_total());
    }

    /// Set the permanent bonus; the current value moves by the same delta.
    pub fn set_bonus(&mut self, bonus: f32) {
        let delta = bonus - self.bonus;
        self.bonus = bonus.max(0.0);
        self.current = (self.current + delta).clamp(0.0, self.max_total());
    }

    pub fn add(&mut self, amount: f32) {
        self.current = (self.current + amount).clamp(0.0, self.max_total());
        if self.exhausted && self.current >= EXHAUST_EXIT {
            self.exhausted = false;
        }
    }

    /// Spend `amount`. Returns `true` when the actor could afford it. An
    /// unaffordable spend empties the pool and latches exhaustion, so the
    /// caller can apply the reduced-damage branch.
    pub fn spend(&mut self, amount: f32) -> bool {
        let affordable = self.current >= amount;
        self.current = (self.current - amount).max(0.0);
        self.regen_delay = REGEN_DELAY;
        if self.current <= 0.0 {
            self.exhausted = true;
        } else if self.exhausted && self.current >= EXHAUST_EXIT {
            self.exhausted = false;
        }
        affordable
    }

    /// ABI hits drained by `StaminaTankHitSmall`/`Large`
    /// (`Game.RTS.*`); flat amounts in the data, applied directly.
    pub fn hit(&mut self, amount: f32) {
        self.current = (self.current - amount.max(0.0)).max(0.0);
        self.regen_delay = REGEN_DELAY;
        if self.current <= 0.0 {
            self.exhausted = true;
        }
    }

    /// Sprint drain for the current state flags, per second.
    pub fn sprint_drain_per_sec(&self, c: &StaminaConstants) -> f32 {
        match (self.encumbered, self.marathon) {
            (true, true) => c.sprint_drain_marathon_encumbered,
            (true, false) => c.sprint_drain_encumbered,
            (false, true) => c.sprint_drain_marathon,
            (false, false) => c.sprint_drain,
        }
    }

    /// Advance one frame: sprint drains, otherwise regen after `regen_delay`.
    pub fn tick(&mut self, dt: f32, c: &StaminaConstants) {
        self.regen_delay = (self.regen_delay - dt).max(0.0);
        if self.sprinting {
            let drain = self.sprint_drain_per_sec(c) * dt;
            self.current = (self.current - drain).max(0.0);
            self.regen_delay = REGEN_DELAY;
        } else if self.regen_delay <= 0.0 {
            let rate = if self.stealth { c.stealth_moving } else { c.regen };
            self.current = (self.current + rate * dt).min(self.max_total());
        }
        if self.current <= 0.0 {
            self.exhausted = true;
        } else if self.exhausted && self.current >= EXHAUST_EXIT {
            self.exhausted = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Weapon runtime
// ---------------------------------------------------------------------------

/// The definition pair the player carries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Loadout {
    pub melee: Option<WeaponDef>,
    pub gun: Option<WeaponDef>,
}

/// Resolve the default loadout, honouring `SOD_MELEE` / `SOD_GUN` overrides.
pub fn default_loadout(data: &WeaponData) -> Loadout {
    let melee = std::env::var("SOD_MELEE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| weapons::DEFAULT_MELEE_WEAPON.to_string());
    let gun = std::env::var("SOD_GUN")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| weapons::DEFAULT_FIREARM.to_string());
    loadout_named(data, &melee, &gun)
}

/// Resolve a named melee/firearm loadout, falling back to Fists / no gun.
pub fn loadout_named(data: &WeaponData, melee: &str, gun: &str) -> Loadout {
    let melee = data
        .get(melee)
        .cloned()
        .or_else(|| data.get(weapons::DEFAULT_MELEE_WEAPON).cloned())
        .or_else(|| data.get("Fists").cloned());
    Loadout { melee, gun: data.get(gun).cloned() }
}

/// One melee weapon plus its condition.
#[derive(Debug, Clone, PartialEq)]
pub struct MeleeSlot {
    pub def: WeaponDef,
    pub tier: SwingTier,
    pub condition: f32,
    pub condition_max: f32,
    pub critical_min: f32,
    /// `DurabilityRating` wear multiplier.
    pub rating: f32,
}

/// One firearm plus its magazine, reserve ammo and condition.
#[derive(Debug, Clone, PartialEq)]
pub struct GunSlot {
    pub def: WeaponDef,
    pub in_mag: i32,
    pub reserve: i32,
    /// Seconds left in the current reload (the rounds transfer immediately).
    pub reload_left: f32,
    pub condition: f32,
    pub condition_max: f32,
    pub critical_min: f32,
    /// `DurabilityRating` wear multiplier.
    pub rating: f32,
}

/// What one melee swing did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeleeAttack {
    /// A swing landed; `damage` already includes the exhaustion scale.
    Swing { damage: f32, cost: f32, tier: SwingTier, exhausted: bool, broke: bool },
    /// Still in the swing cooldown.
    Cooling,
    /// The weapon is broken; use the next weapon.
    Broken,
}

impl MeleeAttack {
    pub fn damage(self) -> Option<f32> {
        match self {
            MeleeAttack::Swing { damage, .. } => Some(damage),
            _ => None,
        }
    }
}

/// What one trigger pull did.
#[derive(Debug, Clone, PartialEq)]
pub enum FireOutcome {
    /// A round left the barrel.
    Shot { damage: f32, noise: f32 },
    Empty,
    Reloading,
    Cooling,
    NoWeapon,
    Broken,
}

/// Per-actor weapon state: the equipped melee weapon and firearm, their
/// condition, and the firearm's magazine/reload/cooldown.
#[derive(Debug, Clone, PartialEq)]
pub struct WeaponRuntime {
    pub melee: Option<MeleeSlot>,
    pub gun: Option<GunSlot>,
    pub fire_cd: f32,
    pub melee_cd: f32,
    /// Noise radius of the most recent shot, metres.
    pub last_noise: f32,
}

impl Default for WeaponRuntime {
    fn default() -> Self {
        WeaponRuntime { melee: None, gun: None, fire_cd: 0.0, melee_cd: 0.0, last_noise: 0.0 }
    }
}

impl WeaponRuntime {
    /// Build the runtime from a loadout and the condition rows.
    pub fn from_loadout(loadout: Loadout, data: &WeaponData) -> Self {
        let melee = loadout.melee.map(|def| {
            let info = data.item(&def.name);
            MeleeSlot {
                tier: SwingTier::from_expertise(info.map(|i| i.expertise.as_str()).unwrap_or("")),
                condition_max: info.map(|i| i.durability.max).unwrap_or(100.0),
                condition: info.map(|i| i.durability.max).unwrap_or(100.0),
                critical_min: info.map(|i| i.durability.critical_min).unwrap_or(0.0),
                rating: info.map(|i| i.durability.rating).unwrap_or(1.0),
                def,
            }
        });
        let gun = loadout.gun.map(|def| {
            let info = data.item(&def.name);
            let clip = def.clip_size();
            let reserve = def.ammo_max.max(0);
            GunSlot {
                in_mag: clip,
                reserve,
                reload_left: 0.0,
                condition_max: info.map(|i| i.durability.max).unwrap_or(100.0),
                condition: info.map(|i| i.durability.max).unwrap_or(100.0),
                critical_min: info.map(|i| i.durability.critical_min).unwrap_or(0.0),
                rating: info.map(|i| i.durability.rating).unwrap_or(1.0),
                def,
            }
        });
        WeaponRuntime { melee, gun, fire_cd: 0.0, melee_cd: 0.0, last_noise: 0.0 }
    }

    /// True when the melee weapon is broken (condition at the floor).
    pub fn melee_broken(&self) -> bool {
        self.melee.as_ref().map(|m| m.condition <= 0.0).unwrap_or(true)
    }

    /// True when the firearm is broken (condition at the floor).
    pub fn gun_broken(&self) -> bool {
        self.gun.as_ref().map(|g| g.condition <= 0.0).unwrap_or(true)
    }

    /// Advance timers; completed reloads become usable.
    pub fn tick(&mut self, dt: f32) {
        self.fire_cd = (self.fire_cd - dt).max(0.0);
        self.melee_cd = (self.melee_cd - dt).max(0.0);
        if let Some(g) = self.gun.as_mut() {
            g.reload_left = (g.reload_left - dt).max(0.0);
        }
    }

    fn wear(condition: &mut f32, rating: f32) {
        let scale = if rating <= 0.0 { 1.0 } else { (1.0 / rating).min(4.0) };
        *condition = (*condition - DURABILITY_WEAR_PER_USE * scale).max(0.0);
    }

    /// Start a reload when the magazine is not full, reserve exists and no
    /// reload is running. Rounds transfer immediately; the busy timer follows
    /// `reload_time`. Returns whether one started.
    pub fn reload(&mut self) -> bool {
        let Some(g) = self.gun.as_mut() else { return false };
        if g.reload_left > 0.0 || g.condition <= 0.0 {
            return false;
        }
        let clip = g.def.clip_size();
        if clip <= 0 || g.in_mag >= clip || g.reserve <= 0 {
            return false;
        }
        let need = clip - g.in_mag;
        let take = need.min(g.reserve);
        g.in_mag += take;
        g.reserve -= take;
        g.reload_left = g.def.fire.as_ref().map(|f| f.reload_time).unwrap_or(0.0);
        true
    }

    /// Pull the trigger. Timers should be advanced by [`WeaponRuntime::tick`].
    pub fn try_fire(&mut self) -> FireOutcome {
        let Some(g) = self.gun.as_mut() else { return FireOutcome::NoWeapon };
        if g.condition <= 0.0 {
            return FireOutcome::Broken;
        }
        if g.reload_left > 0.0 {
            return FireOutcome::Reloading;
        }
        if self.fire_cd > 0.0 {
            return FireOutcome::Cooling;
        }
        if g.in_mag <= 0 {
            return FireOutcome::Empty;
        }
        g.in_mag -= 1;
        self.fire_cd = g.def.shot_interval().max(0.02);
        Self::wear(&mut g.condition, g.rating);
        let noise = g.def.noise_radius();
        let damage = g.def.fire.as_ref().map(|f| f.damage).unwrap_or(0.0);
        self.last_noise = noise;
        FireOutcome::Shot { damage, noise }
    }

    /// Swing the melee weapon, charging stamina. A swing made while exhausted
    /// costs the reduced tier and deals [`EXHAUSTED_DAMAGE_SCALE`] damage.
    pub fn melee_attack(
        &mut self,
        stamina: &mut StaminaState,
        c: &StaminaConstants,
    ) -> MeleeAttack {
        let Some(m) = self.melee.as_mut() else { return MeleeAttack::Broken };
        if m.condition <= 0.0 {
            return MeleeAttack::Broken;
        }
        if self.melee_cd > 0.0 {
            return MeleeAttack::Cooling;
        }
        let reduced = stamina.exhausted;
        let cost = c.cost(m.tier, reduced);
        let affordable = stamina.spend(cost);
        let exhausted = !affordable || stamina.exhausted;
        let damage = if exhausted {
            m.def.melee_damage() * EXHAUSTED_DAMAGE_SCALE
        } else {
            m.def.melee_damage()
        };
        let cooldown = m
            .def
            .melee
            .as_ref()
            .map(|p| if p.duration > 0.0 { p.duration } else { DEFAULT_MELEE_INTERVAL })
            .unwrap_or(DEFAULT_MELEE_INTERVAL);
        self.melee_cd = cooldown.max(0.05);
        Self::wear(&mut m.condition, m.rating);
        let broke = m.condition <= 0.0;
        MeleeAttack::Swing { damage, cost, tier: m.tier, exhausted, broke }
    }

    /// Refresh the magazine/reserve from the definition (debug/loadout change).
    pub fn refill(&mut self) {
        if let Some(g) = self.gun.as_mut() {
            g.in_mag = g.def.clip_size();
            g.reserve = g.def.ammo_max.max(0);
            g.reload_left = 0.0;
        }
    }

    pub fn gun_name(&self) -> &str {
        self.gun.as_ref().map(|g| g.def.title()).unwrap_or("(none)")
    }

    pub fn melee_name(&self) -> &str {
        self.melee.as_ref().map(|m| m.def.title()).unwrap_or("(none)")
    }

    pub fn mag_info(&self) -> Option<(i32, i32, i32)> {
        self.gun.as_ref().map(|g| (g.in_mag, g.def.clip_size(), g.reserve))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn consts() -> StaminaConstants {
        StaminaConstants::builtin()
    }

    #[test]
    fn builtin_costs_match_actorconstants() {
        let c = consts();
        assert_eq!(c.cost(SwingTier::Light, false), 4.0);
        assert_eq!(c.cost(SwingTier::Standard, false), 6.0);
        assert_eq!(c.cost(SwingTier::Heavy, false), 12.0);
        assert_eq!(c.cost(SwingTier::Special, false), 20.0);
        assert_eq!(c.cost(SwingTier::Massive, false), 55.0);
        assert_eq!(c.cost(SwingTier::Heavy, true), 7.0);
        assert_eq!(c.regen, 6.0);
        assert_eq!(c.sprint_drain, 6.0);
    }

    #[test]
    fn parses_stamina_rows_from_fixture() {
        let xml = r#"<ActorConstantsData><DamageConstants><DamageValues>
            <DamageValues Max="0.0" Min="0.0" Name="StaminaRegen_Level00" PerSecond="9.0" Stat="Stamina" />
            <DamageValues Max="0.0" Min="0.0" Name="StaminaDegen_Sprint_Level00_Normal" PerSecond="4.5" Stat="Stamina" />
            <DamageValues Max="7.0" Min="7.0" Name="Stamina_Heavy_Reduced" PerSecond="0.0" Stat="Stamina" />
            <DamageValues Max="100.0" Min="100.0" Name="Health_Explosion" PerSecond="0.0" Stat="Health" />
        </DamageValues></DamageConstants></ActorConstantsData>"#;
        let mut c = StaminaConstants::default();
        c.apply_xml(xml.as_bytes());
        assert_eq!(c.regen, 9.0);
        assert_eq!(c.sprint_drain, 4.5);
        assert_eq!(c.cost(SwingTier::Heavy, true), 7.0);
        // Health rows are ignored.
        assert_eq!(c.cost(SwingTier::Light, false), 4.0);
    }

    #[test]
    fn expertise_maps_to_tier() {
        assert_eq!(SwingTier::from_expertise("EdgedLightBasic"), SwingTier::Light);
        assert_eq!(SwingTier::from_expertise("BluntHeavyBasic"), SwingTier::Heavy);
        assert_eq!(SwingTier::from_expertise("EdgedHeavyBasic"), SwingTier::Heavy);
        assert_eq!(SwingTier::from_expertise("mystery"), SwingTier::Standard);
    }

    #[test]
    fn sprint_drains_then_regen_waits() {
        let c = consts();
        let mut s = StaminaState::new();
        s.sprinting = true;
        s.tick(1.0, &c);
        // 6.0/s for one second.
        approx(s.current, 94.0);
        s.sprinting = false;
        // The regen delay (0.35 s) blocks the first tick.
        s.tick(0.2, &c);
        approx(s.current, 94.0);
        s.regen_delay = 0.0;
        s.tick(0.8, &c);
        // 0.8 s of regen * 6.0 = 4.8.
        approx(s.current, 98.8);
    }

    #[test]
    fn exhaustion_latches_and_clears_with_hysteresis() {
        let c = consts();
        let mut s = StaminaState::new();
        assert!(s.spend(200.0) == false);
        assert!(s.exhausted);
        approx(s.current, 0.0);
        // Regen to just under the exit threshold: still exhausted.
        s.current = EXHAUST_EXIT - 1.0;
        s.regen_delay = 0.0;
        s.tick(0.01, &c);
        assert!(s.exhausted);
        // Cross the threshold: cleared.
        s.current = EXHAUST_EXIT;
        s.tick(0.01, &c);
        assert!(!s.exhausted);
    }

    fn test_data() -> WeaponData {
        WeaponData::builtin()
    }

    fn runtime() -> WeaponRuntime {
        let data = test_data();
        WeaponRuntime::from_loadout(
            loadout_named(&data, "Machete", "Pistol_TacMaster1911"),
            &data,
        )
    }

    #[test]
    fn loadout_uses_recovered_weapons() {
        let data = test_data();
        let l = loadout_named(&data, "Machete", "Pistol_TacMaster1911");
        assert_eq!(l.melee.as_ref().unwrap().name, "Machete");
        assert_eq!(l.gun.as_ref().unwrap().clip_size(), 7);
        let mut rt = WeaponRuntime::from_loadout(l, &data);
        // Machete is EdgedLight -> Light tier; mag starts full.
        assert_eq!(rt.melee.as_ref().unwrap().tier, SwingTier::Light);
        assert_eq!(rt.mag_info(), Some((7, 7, 120)));
        // One swing drains the Light cost.
        let c = consts();
        let mut st = StaminaState::new();
        match rt.melee_attack(&mut st, &c) {
            MeleeAttack::Swing { damage, cost, .. } => {
                assert_eq!(damage, 80.0);
                assert_eq!(cost, 4.0);
            }
            other => panic!("expected swing, got {other:?}"),
        }
        approx(st.current, 96.0);
    }

    #[test]
    fn fire_consumes_magazine_and_empty_triggers() {
        let c = consts();
        let _ = c;
        let mut rt = runtime();
        for _ in 0..7 {
            rt.fire_cd = 0.0;
            match rt.try_fire() {
                FireOutcome::Shot { damage, noise } => {
                    assert_eq!(damage, 200.0);
                    assert_eq!(noise, 30.0);
                }
                other => panic!("expected shot, got {other:?}"),
            }
        }
        rt.fire_cd = 0.0;
        assert_eq!(rt.try_fire(), FireOutcome::Empty);
        assert!(rt.reload());
        assert_eq!(rt.mag_info(), Some((7, 7, 113)));
        // The reload busy timer blocks firing until it elapses.
        assert_eq!(rt.try_fire(), FireOutcome::Reloading);
        rt.tick(0.5);
        rt.fire_cd = 0.0;
        assert!(matches!(rt.try_fire(), FireOutcome::Shot { .. }));
    }

    #[test]
    fn exhausted_swing_deals_reduced_damage() {
        let c = consts();
        let mut rt = runtime();
        let mut st = StaminaState::new();
        st.current = 1.0;
        st.exhausted = true;
        let dmg = match rt.melee_attack(&mut st, &c) {
            MeleeAttack::Swing { damage, exhausted, .. } => {
                assert!(exhausted);
                damage
            }
            other => panic!("expected swing, got {other:?}"),
        };
        approx(dmg, 40.0);
    }

    #[test]
    fn sprint_state_selects_drain_variant() {
        let c = consts();
        let mut s = StaminaState::new();
        s.sprinting = true;
        assert_eq!(s.sprint_drain_per_sec(&c), 6.0);
        s.encumbered = true;
        assert_eq!(s.sprint_drain_per_sec(&c), 15.0);
        s.marathon = true;
        assert_eq!(s.sprint_drain_per_sec(&c), 7.5);
        s.encumbered = false;
        assert_eq!(s.sprint_drain_per_sec(&c), 3.0);
    }

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-3, "expected {b}, got {a}");
    }
}
