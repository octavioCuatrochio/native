//! Game → Flash HUD bridge.
//!
//! The engine drives the HUD through `IFlashPlayer::Invoke` on the element
//! movies (`class3hud.xml` lists the callable `<function>`s per element; the
//! `class3hud/*.as` sources are their implementations).  This module maps our
//! simulation state onto the same ActionScript functions, resolving `@label`
//! localization first.  Continuous values (health/stamina/rotation) are pushed
//! every change; one-shot messages (banners/notifications/prompts) are
//! edge-triggered so their animations do not replay.
//!
//! Which game→HUD calls exist and which are wired by this slice is enumerated
//! in `notes/ui-runtime.md`; the raw call list is parsed from the shipped
//! `class3hud.xml`, not copied into source.

use super::loc::Loc;
use super::player::{invoke, FlashHud, FlashMovie, UiValue};

/// One frame's HUD-relevant state, gathered by the app's `hud_update` system.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HudPush {
    pub health: f32,
    pub max_health: f32,
    pub health_penalty: f32,
    pub health_shadow: f32,
    pub stamina: f32,
    pub max_stamina: f32,
    pub stamina_penalty: f32,
    pub stamina_shadow: f32,
    pub character_visible: bool,

    pub has_gun: bool,
    pub ammo: i32,
    pub ammo_reserve: i32,
    pub fire_mode: i32,

    pub yaw_deg: f32,
    pub zoom: f32,
    pub zombies: i32,
    pub zombies_killed: i32,
    pub threat: i32,

    pub objective: String,
    pub location: String,

    pub prompt: String,
    pub prompt_key: String,

    pub banner_type: String,
    pub banner_name: String,
    pub banner_text: String,

    pub notification: String,
}

impl HudPush {
    fn health_i(&self) -> i32 {
        self.health.round().clamp(0.0, 200.0) as i32
    }
    fn max_health_i(&self) -> i32 {
        self.max_health.round().clamp(0.0, 200.0) as i32
    }
    fn stamina_i(&self) -> i32 {
        self.stamina.round().clamp(0.0, 200.0) as i32
    }
    fn max_stamina_i(&self) -> i32 {
        self.max_stamina.round().clamp(0.0, 200.0) as i32
    }
}

fn movie_mut<'a>(hud: &'a mut FlashHud, name: &str) -> Option<&'a mut FlashMovie> {
    hud.movies.iter_mut().find(|m| m.name == name)
}

fn resolved(loc: Option<&Loc>, s: &str) -> String {
    match loc {
        Some(l) => l.resolve(s),
        None => s.to_string(),
    }
}

/// Push `state` into the live HUD movies.
pub fn apply(hud: &mut FlashHud, state: &HudPush, loc: Option<&Loc>) {
    let last = hud.last.take().unwrap_or_default();
    let changed = &last != state;

    // --- Class3HUDStats (Class3_Stats.gfx): health/stamina/ammo ------------
    if let Some(m) = movie_mut(hud, "Class3HUDStats") {
        if state.health != last.health
            || state.max_health != last.max_health
            || state.health_penalty != last.health_penalty
            || state.health_shadow != last.health_shadow
        {
            invoke(&mut m.player, "SetHealth", &[
                UiValue::Int(state.health_i()),
                UiValue::Int(state.max_health_i()),
                UiValue::Int(state.health_penalty.round() as i32),
                UiValue::Int(state.health_shadow.round() as i32),
                UiValue::Int(0),
            ]);
        }
        if state.stamina != last.stamina
            || state.max_stamina != last.max_stamina
            || state.stamina_penalty != last.stamina_penalty
            || state.stamina_shadow != last.stamina_shadow
        {
            invoke(&mut m.player, "SetStamina", &[
                UiValue::Int(state.stamina_i()),
                UiValue::Int(state.max_stamina_i()),
                UiValue::Int(state.stamina_penalty.round() as i32),
                UiValue::Int(state.stamina_shadow.round() as i32),
                UiValue::Int(0),
            ]);
        }
        if state.character_visible && !last.character_visible {
            invoke(&mut m.player, "SetCharacterUiVisible", &[UiValue::Bool(true)]);
        }
        if state.has_gun != last.has_gun
            || state.ammo != last.ammo
            || state.ammo_reserve != last.ammo_reserve
            || state.fire_mode != last.fire_mode
        {
            if state.has_gun {
                invoke(&mut m.player, "GunStatsShow", &[
                    UiValue::Int(state.ammo),
                    UiValue::Int(state.ammo_reserve),
                    UiValue::Int(0),
                    UiValue::Int(state.fire_mode),
                ]);
            } else {
                invoke(&mut m.player, "GunStatsHide", &[]);
            }
        }
    }

    // --- Class3HUDRadar (Class3_Radar.gfx): minimap + objective ------------
    if let Some(m) = movie_mut(hud, "Class3HUDRadar") {
        if state.yaw_deg != last.yaw_deg {
            invoke(&mut m.player, "SetMapRotation", &[UiValue::Float(state.yaw_deg as f64)]);
        }
        if state.zoom != last.zoom {
            invoke(&mut m.player, "SetMapZoom", &[UiValue::Float(state.zoom as f64)]);
        }
        if state.objective != last.objective {
            let text = resolved(loc, &state.objective);
            invoke(&mut m.player, "SetMissionObjectiveText", &[UiValue::Str(text)]);
        }
        if state.zombies != last.zombies || state.zombies_killed != last.zombies_killed {
            invoke(&mut m.player, "ShowZombieCounter", &[
                UiValue::Int(state.zombies),
                UiValue::Int(state.zombies_killed),
            ]);
        }
        if state.threat != last.threat {
            invoke(&mut m.player, "SetThreatLevel", &[UiValue::Int(state.threat)]);
        }
        if !last.character_visible && state.character_visible {
            invoke(&mut m.player, "SetMinimapUiVisible", &[UiValue::Bool(true)]);
        }
    }

    // --- Class3HUDBanners: mission banners ---------------------------------
    if let Some(m) = movie_mut(hud, "Class3HUDBanners") {
        if state.banner_text != last.banner_text && !state.banner_text.is_empty() {
            let ty = resolved(loc, &state.banner_type);
            let name = resolved(loc, &state.banner_name);
            let text = resolved(loc, &state.banner_text);
            invoke(&mut m.player, "MissionBannerStart", &[
                UiValue::Str(ty),
                UiValue::Str(name),
                UiValue::Str(text),
            ]);
        }
    }

    // --- Class3HUDCenterPrompts: interaction prompt ------------------------
    if let Some(m) = movie_mut(hud, "Class3HUDCenterPrompts") {
        if state.prompt != last.prompt {
            if state.prompt.is_empty() {
                invoke(&mut m.player, "HideUseButton", &[]);
            } else {
                let text = resolved(loc, &state.prompt);
                let key = if state.prompt_key.is_empty() { "E".to_string() } else { resolved(loc, &state.prompt_key) };
                invoke(&mut m.player, "ShowUseButton", &[UiValue::Str(text), UiValue::Str(key)]);
            }
        }
    }

    // --- Class3HUDNotifications: reward/status messages --------------------
    if let Some(m) = movie_mut(hud, "Class3HUDNotifications") {
        if state.notification != last.notification && !state.notification.is_empty() {
            let text = resolved(loc, &state.notification);
            invoke(&mut m.player, "AddRewardMessage", &[UiValue::Str(text)]);
        }
    }

    let _ = changed;
    hud.last = Some(state.clone());
}
