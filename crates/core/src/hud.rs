//! HUD state model (Class3HUD: radar, reticle, notifications, prompts, banners).
//!
//! This is the native, testable model the renderer draws.  It mirrors what the
//! real `Class3HUD` front-end widgets show, anchored by the `Class3HUD*` strings
//! in `notes/ghidra/class3.md` §HUD (160 functions) and the decompiled tick
//! `FUN_0143c080` (`Class3HUDStats`, `SetHealth`/`SetStamina`):
//!
//! * **Radar** (`Class3HUDRadar` @ `0185aa40`, `FUN_013c1820` / `FUN_013c5060`):
//!   blips are stored player-relative; `FUN_01434480` subtracts the player from
//!   each blip, tests it against the aggro radius and fires `PlayAlert`.
//!   `FUN_0143b390` drives `SetAggroRadius`, `FUN_01438c80`
//!   `SetSiegeThreatLevel`, `FUN_01428d00` `ShowRadarBanner`.
//! * **CenterPrompts** (`0174eb38`): `FUN_01434730` `ShowHint`,
//!   `FUN_0142d4f0` `HideHint`, `FUN_0142d770` `HideSubtitle`.  The tick holds a
//!   prompt TTL in `DAT_01e5ac90` which is decremented by `dt` and hidden at 0.
//! * **Notifications** (`0174eb50`): `FUN_014332e0` `ShowGameNotifications`,
//!   `FUN_01436730` `ClearGameNotifications`, `FUN_01424670` `MapLabelClear`.
//!   Entries are transient, so each carries a `ttl` aged by the tick.
//! * **Banners** (`0174eb24`): `FUN_01434e20` `ShowWarningBanner`,
//!   `FUN_01439e40` `MissionBannerStart`, `FUN_01425940`
//!   `HideDangerZoneBanner`.
//! * **Reticle** (`0174eb8c`): `FUN_015e5da0` (`ResetReticle` / `OnGunFired`).
//! * **Stats** (`01865358`): `FUN_0143c080` sets the clamped health/stamina bars.
//!
//! Self-contained: `std` plus the already-present [`crate::combat`] and
//! [`crate::inventory`] modules.

use crate::combat;
use crate::inventory;

// ---------------------------------------------------------------------------
// Vitals
// ---------------------------------------------------------------------------

/// The player's vitals as the HUD draws them (`Class3HUDStats`).
///
/// Ratios are always clamped to `0.0..=1.0` and never NaN, matching the
/// `FUN_0143cb10` bar-fill helper invoked by the stats tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudVitals {
    pub hp: f32,
    pub max_hp: f32,
    pub stamina: f32,
    pub max_stamina: f32,
    pub injured: bool,
    pub bleeding: bool,
}

impl HudVitals {
    /// Full health/stamina, no wounds.
    pub fn new() -> Self {
        HudVitals {
            hp: 100.0,
            max_hp: 100.0,
            stamina: 100.0,
            max_stamina: 100.0,
            injured: false,
            bleeding: false,
        }
    }

    /// Convert the combat model's [`combat::Vitals`] into the HUD view.
    ///
    /// Non-finite inputs are sanitised so the bars can never render NaN.
    pub fn from_combat(v: &combat::Vitals) -> Self {
        let max_hp = sane(v.max_hp, 0.0).max(0.0);
        let max_stamina = sane(v.max_stamina, 0.0).max(0.0);
        HudVitals {
            hp: sane(v.hp, 0.0).clamp(0.0, max_hp),
            max_hp,
            stamina: sane(v.stamina, 0.0).clamp(0.0, max_stamina),
            max_stamina,
            injured: v.injuries != 0,
            bleeding: sane(v.bleed, 0.0) > 0.0,
        }
    }

    /// HP fraction `0.0..=1.0` (0 when max is non-positive or NaN).
    pub fn hp_ratio(&self) -> f32 {
        ratio(self.hp, self.max_hp)
    }

    /// Stamina fraction `0.0..=1.0` (0 when max is non-positive or NaN).
    pub fn stamina_ratio(&self) -> f32 {
        ratio(self.stamina, self.max_stamina)
    }

    /// True at 0 HP.
    pub fn is_downed(&self) -> bool {
        sane(self.hp, 0.0) <= 0.0
    }
}

impl Default for HudVitals {
    fn default() -> Self {
        HudVitals::new()
    }
}

// ---------------------------------------------------------------------------
// Widgets
// ---------------------------------------------------------------------------

/// The kind of centre prompt (`Class3HUDCenterPrompts`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptKind {
    Interact,
    Search,
    Climb,
    Vehicle,
    Custom(String),
}

/// A centre prompt with an optional hold progress `0.0..=1.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct CenterPrompt {
    pub kind: PromptKind,
    pub text: String,
    pub progress: f32,
}

/// A transient HUD notification (`Class3HUDNotifications`).
#[derive(Debug, Clone, PartialEq)]
pub struct Notification {
    pub text: String,
    /// Seconds remaining before the entry is dropped.
    pub ttl: f32,
    /// Caller-defined severity/category (mirrors the notification kind bits).
    pub kind: u8,
}

/// One radar blip, player-relative (`FUN_01434480`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadarBlip {
    /// `[right, forward]` in metres relative to the player, rotated by yaw.
    pub rel: [f32; 2],
    /// Blip kind (zombie kind / marker type).
    pub kind: u8,
    /// Aggro/threat confidence `0.0..=1.0`.
    pub threat: f32,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// The complete drawable HUD state.
#[derive(Debug, Clone)]
pub struct HudState {
    pub vitals: HudVitals,
    pub objective: Option<String>,
    pub zombie_count: u32,
    pub blips: Vec<RadarBlip>,
    pub prompts: Vec<CenterPrompt>,
    pub notifications: Vec<Notification>,
    pub banner: Option<String>,
    pub reticle_spread: f32,
    pub interaction: Option<String>,
    pub inventory_summary: String,
    /// Remaining seconds for [`HudState::banner`]; private because the public
    /// surface models the banner as a single optional string.
    banner_ttl: f32,
}

impl HudState {
    /// An empty HUD at full vitals.
    pub fn new() -> Self {
        HudState {
            vitals: HudVitals::new(),
            objective: None,
            zombie_count: 0,
            blips: Vec::new(),
            prompts: Vec::new(),
            notifications: Vec::new(),
            banner: None,
            reticle_spread: 0.0,
            interaction: None,
            inventory_summary: String::new(),
            banner_ttl: 0.0,
        }
    }
}

impl Default for HudState {
    fn default() -> Self {
        HudState::new()
    }
}

// ---------------------------------------------------------------------------
// Per-frame input
// ---------------------------------------------------------------------------

/// One nearby actor offered to the radar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudNpc {
    pub pos: [f32; 3],
    /// Zombie kind / blip type.
    pub kind: u8,
    /// Aggro/threat confidence `0.0..=1.0` (from the zombie AI).
    pub threat: f32,
}

/// Everything the HUD needs for one frame.
pub struct HudCtx<'a> {
    pub vitals: combat::Vitals,
    pub player_pos: [f32; 3],
    pub player_yaw: f32,
    pub npcs: &'a [HudNpc],
    pub objective: Option<&'a str>,
    pub inventory: Option<&'a inventory::Inventory>,
    pub db: Option<&'a inventory::ItemDb>,
    /// Current reticle spread in degrees/pixels (renderer-defined units).
    pub reticle_spread: f32,
    /// Interaction prompt text for the actor under the crosshair.
    pub interaction: Option<&'a str>,
    pub dt: f32,
}

// ---------------------------------------------------------------------------
// Mutators
// ---------------------------------------------------------------------------

/// Queue a transient notification (`ShowGameNotifications` @ `FUN_014332e0`).
pub fn push_notification(state: &mut HudState, text: impl Into<String>, kind: u8, ttl: f32) {
    state.notifications.push(Notification {
        text: text.into(),
        ttl: sane(ttl, 0.0).max(0.0),
        kind,
    });
}

/// Drop every notification (`ClearGameNotifications` @ `FUN_01436730`).
pub fn clear_notifications(state: &mut HudState) {
    state.notifications.clear();
}

/// Set the active objective line.
pub fn set_objective(state: &mut HudState, text: impl Into<String>) {
    state.objective = Some(text.into());
}

/// Clear the objective line.
pub fn clear_objective(state: &mut HudState) {
    state.objective = None;
}

/// Replace the centre prompt (`ShowHint` @ `FUN_01434730`).
pub fn set_prompt(state: &mut HudState, kind: PromptKind, text: impl Into<String>, progress: f32) {
    state.prompts.clear();
    state.prompts.push(CenterPrompt {
        kind,
        text: text.into(),
        progress: sane(progress, 0.0).clamp(0.0, 1.0),
    });
}

/// Clear all centre prompts (`HideHint` @ `FUN_0142d4f0`).
pub fn clear_prompts(state: &mut HudState) {
    state.prompts.clear();
}

/// Show a banner for `ttl` seconds (`ShowWarningBanner` @ `FUN_01434e20`).
pub fn set_banner(state: &mut HudState, text: impl Into<String>, ttl: f32) {
    state.banner = Some(text.into());
    state.banner_ttl = sane(ttl, 0.0).max(0.0);
}

/// Hide the banner.
pub fn clear_banner(state: &mut HudState) {
    state.banner = None;
    state.banner_ttl = 0.0;
}

// ---------------------------------------------------------------------------
// Frame update
// ---------------------------------------------------------------------------

/// Refresh the HUD for one frame: vitals, radar blips, notifications, banner,
/// prompts, interaction and inventory summary.
pub fn update(state: &mut HudState, ctx: &HudCtx) {
    let dt = sane(ctx.dt, 0.0).max(0.0);

    // --- Stats (`FUN_0143c080`: SetHealth / SetStamina) ---
    state.vitals = HudVitals::from_combat(&ctx.vitals);

    // --- Objective (authoritative only when supplied) ---
    if let Some(text) = ctx.objective {
        state.objective = Some(text.to_string());
    }

    // --- Radar (`FUN_01434480`: player-relative blips + aggro radius) ---
    let px = sane(ctx.player_pos[0], 0.0);
    let pz = sane(ctx.player_pos[2], 0.0);
    let yaw = sane(ctx.player_yaw, 0.0);
    let (sin_y, cos_y) = yaw.sin_cos();
    state.blips.clear();
    state.blips.reserve(ctx.npcs.len());
    for npc in ctx.npcs {
        let dx = sane(npc.pos[0], 0.0) - px;
        let dz = sane(npc.pos[2], 0.0) - pz;
        // Forward is +Z at yaw 0; right is +X.  Rotate world delta into the
        // player's frame so "up" on the radar is where the player faces.
        let forward = dx * sin_y + dz * cos_y;
        let right = dx * cos_y - dz * sin_y;
        state.blips.push(RadarBlip {
            rel: [right, forward],
            kind: npc.kind,
            threat: sane(npc.threat, 0.0).clamp(0.0, 1.0),
        });
    }
    state.zombie_count = ctx.npcs.len() as u32;

    // --- Notifications: age by dt, drop at 0 ---
    if dt > 0.0 {
        for n in &mut state.notifications {
            n.ttl = sane(n.ttl, 0.0) - dt;
        }
        state.notifications.retain(|n| n.ttl > 0.0);
    }

    // --- Banner TTL ---
    if state.banner.is_some() {
        state.banner_ttl -= dt;
        if state.banner_ttl <= 0.0 {
            state.banner = None;
            state.banner_ttl = 0.0;
        }
    }

    // --- Centre prompts: keep progress in range ---
    for p in &mut state.prompts {
        p.progress = sane(p.progress, 0.0).clamp(0.0, 1.0);
    }

    // --- Reticle (`FUN_015e5da0`) ---
    state.reticle_spread = sane(ctx.reticle_spread, 0.0).max(0.0);

    // --- Interaction prompt ---
    state.interaction = ctx.interaction.map(str::to_string);

    // --- Inventory summary ---
    state.inventory_summary = inventory_summary(ctx);
}

fn inventory_summary(ctx: &HudCtx) -> String {
    match (ctx.inventory, ctx.db) {
        (Some(inv), Some(db)) => {
            let items = inv.slots().iter().filter(|s| !s.is_empty()).count();
            let weight = sane(inv.total_weight(db), 0.0).max(0.0);
            format!("Backpack: {items} items, {weight:.1} lb")
        }
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Replace a non-finite value with `fallback`.
#[inline]
fn sane(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

/// Clamped `value / max`, safe against zero/negative/NaN maxima.
#[inline]
fn ratio(value: f32, max: f32) -> f32 {
    let m = sane(max, 0.0);
    if m > 0.0 {
        (sane(value, 0.0) / m).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::ZombieKind;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
    }

    fn ctx<'a>(npcs: &'a [HudNpc], yaw: f32, dt: f32) -> HudCtx<'a> {
        HudCtx {
            vitals: combat::Vitals::new(100.0, 100.0),
            player_pos: [0.0, 0.0, 0.0],
            player_yaw: yaw,
            npcs,
            objective: None,
            inventory: None,
            db: None,
            reticle_spread: 0.0,
            interaction: None,
            dt,
        }
    }

    #[test]
    fn blips_are_player_relative() {
        let npcs = [HudNpc { pos: [3.0, 0.0, 4.0], kind: 1, threat: 0.5 }];
        let mut s = HudState::new();
        update(&mut s, &ctx(&npcs, 0.0, 0.0));

        assert_eq!(s.blips.len(), 1);
        approx(s.blips[0].rel[0], 3.0); // right
        approx(s.blips[0].rel[1], 4.0); // forward
        assert_eq!(s.blips[0].kind, 1);
        approx(s.blips[0].threat, 0.5);
        assert_eq!(s.zombie_count, 1);

        // Player moving to the npc's position zeroes the blip.
        let npcs2 = [HudNpc { pos: [3.0, 0.0, 4.0], kind: 1, threat: 0.0 }];
        let mut c = ctx(&npcs2, 0.0, 0.0);
        c.player_pos = [3.0, 0.0, 4.0];
        update(&mut s, &c);
        approx(s.blips[0].rel[0], 0.0);
        approx(s.blips[0].rel[1], 0.0);
    }

    #[test]
    fn blips_rotate_by_yaw() {
        use std::f32::consts::FRAC_PI_2;
        let npcs = [HudNpc { pos: [0.0, 0.0, 4.0], kind: 0, threat: 1.0 }];
        let mut s = HudState::new();

        // Facing +Z (yaw 0): blip is straight ahead.
        update(&mut s, &ctx(&npcs, 0.0, 0.0));
        approx(s.blips[0].rel[0], 0.0);
        approx(s.blips[0].rel[1], 4.0);

        // Turn right 90°: the same world target is now to the player's left.
        update(&mut s, &ctx(&npcs, FRAC_PI_2, 0.0));
        approx(s.blips[0].rel[0], -4.0);
        approx(s.blips[0].rel[1], 0.0);

        // A target on world +X sits straight ahead when facing +X.
        let npcs2 = [HudNpc { pos: [4.0, 0.0, 0.0], kind: 0, threat: 1.0 }];
        update(&mut s, &ctx(&npcs2, FRAC_PI_2, 0.0));
        approx(s.blips[0].rel[0], 0.0);
        approx(s.blips[0].rel[1], 4.0);
    }

    #[test]
    fn notifications_expire_after_ttl() {
        let npcs: [HudNpc; 0] = [];
        let mut s = HudState::new();
        push_notification(&mut s, "Found a backpack", 1, 1.0);
        assert_eq!(s.notifications.len(), 1);

        update(&mut s, &ctx(&npcs, 0.0, 0.4));
        assert_eq!(s.notifications.len(), 1);
        approx(s.notifications[0].ttl, 0.6);

        update(&mut s, &ctx(&npcs, 0.0, 0.7));
        assert!(s.notifications.is_empty(), "ttl elapsed should drop it");

        // Clearing is immediate.
        push_notification(&mut s, "x", 0, 5.0);
        clear_notifications(&mut s);
        assert!(s.notifications.is_empty());
    }

    #[test]
    fn hp_and_stamina_ratios_clamp() {
        let mut v = HudVitals::new();
        v.hp = 150.0;
        v.max_hp = 100.0;
        approx(v.hp_ratio(), 1.0);
        v.hp = -5.0;
        approx(v.hp_ratio(), 0.0);
        v.hp = 25.0;
        approx(v.hp_ratio(), 0.25);

        v.stamina = 200.0;
        v.max_stamina = 100.0;
        approx(v.stamina_ratio(), 1.0);

        // Zero / NaN maxima never divide by zero.
        v.max_hp = 0.0;
        approx(v.hp_ratio(), 0.0);
        v.max_stamina = f32::NAN;
        approx(v.stamina_ratio(), 0.0);
        v.max_hp = f32::NAN;
        approx(v.hp_ratio(), 0.0);
    }

    #[test]
    fn from_combat_maps_wounds() {
        let mut cv = combat::Vitals::new(120.0, 80.0);
        cv.hp = 60.0;
        cv.stamina = 20.0;
        cv.injuries = crate::game::HitLocation::HEAD;
        cv.bleed = 2.0;
        let hv = HudVitals::from_combat(&cv);
        approx(hv.hp, 60.0);
        approx(hv.max_hp, 120.0);
        approx(hv.stamina_ratio(), 0.25);
        assert!(hv.injured);
        assert!(hv.bleeding);
        assert!(!hv.is_downed());

        // A downed actor reads as downed.
        cv.hp = 0.0;
        cv.injuries = 0;
        cv.bleed = 0.0;
        let hv = HudVitals::from_combat(&cv);
        assert!(hv.is_downed());
        assert!(!hv.injured);
        assert!(!hv.bleeding);
    }

    #[test]
    fn prompt_set_and_clear() {
        let mut s = HudState::new();
        set_prompt(&mut s, PromptKind::Search, "Search container", 0.5);
        assert_eq!(s.prompts.len(), 1);
        assert_eq!(s.prompts[0].kind, PromptKind::Search);
        assert_eq!(s.prompts[0].text, "Search container");
        approx(s.prompts[0].progress, 0.5);

        // A new prompt replaces the previous one.
        set_prompt(&mut s, PromptKind::Custom("Loot".into()), "Take all", 2.0);
        assert_eq!(s.prompts.len(), 1);
        approx(s.prompts[0].progress, 1.0); // clamped

        clear_prompts(&mut s);
        assert!(s.prompts.is_empty());
    }

    #[test]
    fn banner_set_and_expire() {
        let npcs: [HudNpc; 0] = [];
        let mut s = HudState::new();
        set_banner(&mut s, "Danger zone", 2.0);
        assert_eq!(s.banner.as_deref(), Some("Danger zone"));

        update(&mut s, &ctx(&npcs, 0.0, 1.0));
        assert_eq!(s.banner.as_deref(), Some("Danger zone"));

        update(&mut s, &ctx(&npcs, 0.0, 1.5));
        assert!(s.banner.is_none(), "banner should expire after its ttl");

        set_banner(&mut s, "Again", 3.0);
        clear_banner(&mut s);
        assert!(s.banner.is_none());
    }

    #[test]
    fn objective_interaction_and_inventory() {
        let npcs: [HudNpc; 0] = [];
        let mut s = HudState::new();

        let mut c = ctx(&npcs, 0.0, 0.0);
        c.objective = Some("Reach the church");
        c.interaction = Some("Open door");
        update(&mut s, &c);
        assert_eq!(s.objective.as_deref(), Some("Reach the church"));
        assert_eq!(s.interaction.as_deref(), Some("Open door"));

        // A None objective leaves a manually set one untouched.
        set_objective(&mut s, "Manual");
        update(&mut s, &ctx(&npcs, 0.0, 0.0));
        assert_eq!(s.objective.as_deref(), Some("Manual"));

        clear_objective(&mut s);
        assert!(s.objective.is_none());
        assert!(s.inventory_summary.is_empty());
    }

    #[test]
    fn nan_total_is_sanitised() {
        let nan_npc = HudNpc { pos: [f32::NAN, f32::NAN, f32::NAN], kind: 2, threat: f32::NAN };
        let npcs = [nan_npc];
        let mut s = HudState::new();
        let mut c = ctx(&npcs, f32::NAN, f32::NAN);
        c.player_pos = [f32::NAN; 3];
        c.reticle_spread = f32::NAN;
        c.vitals = combat::Vitals {
            hp: f32::NAN,
            max_hp: f32::NAN,
            stamina: f32::NAN,
            max_stamina: f32::NAN,
            bleed: f32::NAN,
            injuries: 0,
        };
        update(&mut s, &c);

        assert!(s.vitals.hp.is_finite());
        assert!(s.vitals.max_hp.is_finite());
        assert!(s.vitals.hp_ratio().is_finite());
        assert!(s.vitals.stamina_ratio().is_finite());
        assert!(s.reticle_spread.is_finite());
        assert_eq!(s.blips.len(), 1);
        assert!(s.blips[0].rel[0].is_finite());
        assert!(s.blips[0].rel[1].is_finite());
        assert!(s.blips[0].threat.is_finite());
        assert_eq!(s.blips[0].threat, 0.0);
    }

    #[test]
    fn zombie_kind_round_trips_through_blips() {
        let npcs = [HudNpc {
            pos: [1.0, 0.0, 1.0],
            kind: ZombieKind::Bloater as u8,
            threat: 0.25,
        }];
        let mut s = HudState::new();
        update(&mut s, &ctx(&npcs, 0.0, 0.0));
        assert_eq!(s.blips[0].kind, ZombieKind::Bloater as u8);
    }
}
