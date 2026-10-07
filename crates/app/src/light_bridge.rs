//! Bevy bridge for the recovered CryEngine lighting.
//!
//! The core crate `sod_core::render_light` reproduces what the original
//! renderer computes for one instant: the TOD sun radiance, sky/ground
//! ambient, volumetric-fog parameters, the HDR exposure curve and the
//! post tone-map colour grade. This module takes that [`LightingFrame`] and
//! drives a stock Bevy scene from it:
//!
//! * one `DirectionalLight` for the sun, oriented from the recovered sun
//!   vector and coloured by the frame's *tone-mapped* sun (per-channel
//!   `ExpMapping` + s-curve, `HDRPostProcess.cfx`), so the warm authored
//!   `Sun color` lands near-neutral on screen the way the engine's HDR chain
//!   makes it;
//! * sky ambient (`AmbientLight`) from `Sky color * multiplier`;
//! * `DistanceFog` from the TOD fog colour;
//! * camera `Exposure` (EV100) derived from the recovered `HDRLevel /
//!   (1 + HDROffset*adapted)` exposure;
//! * `Tonemapping` (Bevy's neutral `KhronosPbrNeutral`, no LUT required) and
//!   `ColorGrading` for the parts of the grade that map onto Bevy's model.
//!
//! It also spawns the level's baked `.lights` near the player as Bevy
//! `PointLight`/`SpotLight`s. The engine's `LSUpdate` (`0x15ba8a0`) culls
//! lights by distance; `FUN_015bb8e0` (`0x15bb8e0`) uploads radius and range
//! scale, and projector lights derive their direction from the record's Euler
//! angles (`FUN_005e4a30`, `0x5e4a30`). Cubemap-only entries are skipped
//! because they need the deferred cubemap pass that is not ported.
//!
//! # Unit mapping
//!
//! The TOD radiance is relative (the engine has no physical calibration), so
//! this bridge fixes a scale and documents it:
//!
//! * `SUN_LUX_PER_UNIT = 10_000`: one unit of `LightingFrame::sun_radiance`
//!   luminance becomes 10 000 lux of directional illuminance. The shipped
//!   level's noon luminance (~8.25) therefore lands at ~82 000 lux, i.e.
//!   daylight.
//! * `SKY_LUX_PER_UNIT = 1_000`: one unit of `sky_ambient` luminance becomes
//!   1 000 cd/m² of ambient light so shaded surfaces keep some fill without
//!   flattening the scene.
//! * `LUMEN_PER_UNIT = 150`: one unit of a level light's range scale becomes
//!   150 lm; a mean level light (`range_scale*0.5` ≈ 45) is ~7 000 lm.
//! * `Exposure` EV100 is solved back from the core exposure factor `E`:
//!   `ev100 = -log2(1.2 * E / SUN_LUX_PER_UNIT) + EXPOSURE_BIAS_STOPS`, i.e.
//!   the stops needed to bring `SUN_LUX_PER_UNIT * sun_luminance` to the same
//!   display level the engine's own exposure produced for the relative frame,
//!   plus a small `EXPOSURE_BIAS_STOPS` calibration so the shipped level keeps
//!   contrast instead of sitting on the tone-map shoulder.
//!
//! Everything is emitted once, at the level's authored time; the plugin does
//! not advance the clock.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::CascadeShadowConfig;
use bevy::prelude::*;
use bevy::render::view::{ColorGrading, ColorGradingGlobal, ColorGradingSection};
use sod_core::render_light::{luminance, tonemap, LightKind, LevelLightParams, ShadowSettings};

/// Directional illuminance (lux) per unit of TOD sun radiance luminance.
const SUN_LUX_PER_UNIT: f32 = 10_000.0;
/// Ambient brightness (cd/m²) per unit of TOD sky-ambient luminance. Bevy
/// consumes `AmbientLight::brightness` as incident irradiance, so a pure
/// luminance→irradiance conversion would fold in a factor of pi; 1 000 keeps
/// shaded surfaces filled without flattening the directional contrast.
const SKY_LUX_PER_UNIT: f32 = 1_000.0;
/// Stops applied on top of the derived EV100 to make Bevy's physically-based
/// shading reproduce the engine's relative exposure.
///
/// The engine shades relative radiance with no `1/PI` factor, while Bevy's
/// Lambertian `Fd_Burley` divides the directional contribution by `PI`. The
/// derived `ev100` already folds in `SUN_LUX_PER_UNIT`, so the missing factor
/// is exactly one stop count of `-log2(PI)`: with it, `SUN_LUX_PER_UNIT *
/// sun_luminance` and the recovered `ExposureParams::exposure()` reproduce the
/// same midtone the engine's `HDRLevel/(1+HDROffset*adapted)` curve produced
/// for the relative scene. Using `+0.4` here made noon ~2 stops too dark.
const EXPOSURE_BIAS_STOPS: f32 = -1.651_496_3; // -log2(PI)
/// Luminous power (lm) per unit of a level light's uploaded intensity.
const LUMEN_PER_UNIT: f32 = 150.0;
/// Level lights are gathered within this many metres of the player start.
const LIGHT_RADIUS_M: f32 = 150.0;
/// Hard cap on spawned level lights, nearest first, to bound frame cost.
const MAX_LEVEL_LIGHTS: usize = 96;
/// Hard cap on level lights that cast shadows (each adds a shadow pass).
const MAX_SHADOW_LIGHTS: usize = 8;
/// Projector cone half-angle (radians) when a light only carries a texture.
const PROJECTOR_OUTER_ANGLE: f32 = 0.6;
/// Fraction of the outer angle used as the spot's inner (full-brightness)
/// cone.
const PROJECTOR_INNER_FRACTION: f32 = 0.7;

/// Installs the bridge. Added by `main.rs` next to the other plugins.
pub struct LightBridgePlugin;

impl Plugin for LightBridgePlugin {
    fn build(&self, app: &mut App) {
        // The shipped spec level drives the sun's cascaded shadow maps. The
        // spec is resolved once here (same source as `post.rs`) so the sun and
        // the post chain agree on the group.
        let spec = crate::post::spec_from_env();
        app.insert_resource(ShadowConfig(ShadowSettings::for_spec(spec)))
            .init_resource::<LightingApplied>()
            .init_resource::<FrameStats>()
            .add_systems(Update, (apply_lighting, log_frame_time));
    }
}

/// The sun's shadow CVarGroup for the active spec level.
#[derive(Resource)]
struct ShadowConfig(ShadowSettings);

/// Guard so the one-shot lighting setup runs a single time, after
/// `setup_world` has spawned the camera.
#[derive(Resource, Default)]
struct LightingApplied(bool);

/// Rolling frame-time accumulator, logged so the before/after numbers in the
/// report come from the running app rather than the wall clock.
#[derive(Resource, Default)]
struct FrameStats {
    frames: u32,
    accum: f32,
    last_log: u32,
}

/// The Cry-to-Bevy basis change: the engine is Z-up, Bevy Y-up. Matches the
/// conversion `main.rs` already uses for world positions.
fn cry_to_bevy(v: [f32; 3]) -> Vec3 {
    Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2) * Vec3::new(v[0], v[1], v[2])
}

/// Projector emission direction from the record's Euler angles.
///
/// `FUN_005e4a30` (`0x5e4a30`) builds the row-major `Rz*Ry*Rx` matrix; the
/// second row is the light's local Y axis, which is CryEngine's forward.
fn projector_forward(euler: [f32; 3]) -> Vec3 {
    let (rx, ry, rz) = (euler[0], euler[1], euler[2]);
    let (sx, cx) = (rx.sin(), rx.cos());
    let (sy, cy) = (ry.sin(), ry.cos());
    let (sz, cz) = (rz.sin(), rz.cos());
    let local_y = [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx];
    cry_to_bevy(local_y).normalize_or_zero()
}

fn apply_lighting(
    mut lighting_applied: ResMut<LightingApplied>,
    lighting: Option<Res<crate::LevelLighting>>,
    shadow_cfg: Res<ShadowConfig>,
    camera: Query<Entity, With<Camera3d>>,
    mut ambient: Query<&mut AmbientLight, With<Camera3d>>,
    mut commands: Commands,
) {
    if lighting_applied.0 {
        return;
    }
    let Some(lighting) = lighting else {
        // No mission TOD: leave the scene to whatever main.rs set up.
        lighting_applied.0 = true;
        return;
    };
    let Ok(cam) = camera.single() else {
        return;
    };

    let frame = lighting.frame;

    // --- sun ---------------------------------------------------------------
    // The authored `Sun color` is deliberately warm. The engine's per-channel
    // exponential + s-curve tone map saturates it to near-neutral daylight
    // (`render_light::tonemap`); use that display colour as the Bevy light
    // colour and let the illuminance carry the physical scale.
    let sun_display = tonemap(frame.sun_radiance, &frame.tonemap);
    let display_lum = luminance(sun_display);
    let sun_dir = cry_to_bevy(frame.sun_dir).normalize_or_zero();
    let sh = shadow_cfg.0;
    let shadows_on = sh.gsm_lods > 0 && std::env::var("SOD_NOSHADOWS").is_err();
    if display_lum > 1e-4 && frame.sun_radiance.iter().any(|c| *c > 0.0) {
        let color = Vec3::from_array(sun_display) / display_lum;
        let sun_lum = luminance(frame.sun_radiance).max(0.0);
        // Sun cascaded shadow maps from `sys_spec_shadows.cfg`: the cascade
        // far bounds are the accumulated `e_GsmRange * e_GsmRangeStep^n` areas
        // and `e_ShadowsMaxTexRes` sets the per-cascade map size (a global
        // resource, set by `post.rs`). Biases come from
        // `e_ShadowsConstBias` / `e_ShadowsSlopeBias`.
        //
        // The shipped High group configures 5 cascades (`e_GsmLodsNum`), but
        // Bevy's directional shadow array is capped at 4
        // (`bevy_pbr::render::light` warns and ignores the rest). Merge the two
        // farthest cascades instead of dropping the last: keep the first three
        // near bounds and stretch the fourth to the 5th's far bound, so the
        // full `r_ShadowsFadeDistMax` range is still covered.
        let mut bounds = sh.cascade_bounds();
        if bounds.len() > 4 {
            let far = *bounds.last().unwrap();
            bounds.truncate(3);
            bounds.push(far);
        }
        let cascade = CascadeShadowConfig {
            bounds,
            overlap_proportion: 0.1,
            minimum_distance: 0.1,
        };
        commands.spawn((
            DirectionalLight {
                color: Color::linear_rgb(color.x, color.y, color.z),
                illuminance: sun_lum * SUN_LUX_PER_UNIT,
                shadow_maps_enabled: shadows_on,
                shadow_depth_bias: sh.depth_bias(),
                shadow_normal_bias: sh.normal_bias(),
                ..default()
            },
            cascade,
            Transform::default().looking_to(sun_dir, Vec3::Y),
        ));
    }

    // --- sky ambient -------------------------------------------------------
    let sky_lum = luminance(frame.sky_ambient).max(0.0);
    let sky_color = if sky_lum > 1e-4 {
        Vec3::from_array(frame.sky_ambient) / sky_lum
    } else {
        Vec3::ONE
    };
    let ambient_light = AmbientLight {
        color: Color::linear_rgb(sky_color.x, sky_color.y, sky_color.z),
        brightness: sky_lum * SKY_LUX_PER_UNIT,
        affects_lightmapped_meshes: true,
    };
    if let Ok(mut a) = ambient.single_mut() {
        *a = ambient_light;
    } else {
        commands.entity(cam).insert(ambient_light);
    }

    // --- fog ---------------------------------------------------------------
    // Bevy has one fog colour; the engine's gradient has a ground and a top
    // term, so blend them for a single distance-fog tint. Exponential falloff
    // is the closest match to the engine's volumetric fog; visibility comes
    // from the static `Environment.Fog.ViewDistance` when present.
    let fog_mix = |i: usize| 0.35 * frame.fog.color[i] + 0.65 * frame.fog.color_top[i];
    let fog_color = Color::linear_rgb(fog_mix(0), fog_mix(1), fog_mix(2));
    let falloff = if frame.fog.view_distance > 0.0 {
        FogFalloff::from_visibility(frame.fog.view_distance)
    } else {
        FogFalloff::Exponential { density: 0.0006 }
    };
    commands.entity(cam).insert(DistanceFog {
        color: fog_color,
        falloff,
        ..default()
    });

    // --- exposure ----------------------------------------------------------
    // `frame.exposure.exposure()` is the relative factor the engine applied
    // before its tone map; converting it to a physical EV100 lets Bevy's
    // camera exposure reproduce the same midtone for the scaled scene.
    let e = frame.exposure.exposure().max(1e-6);
    let bias = std::env::var("SOD_EXPOSURE_BIAS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(EXPOSURE_BIAS_STOPS);
    let ev100 = -(1.2 * e / SUN_LUX_PER_UNIT).log2() + bias;
    commands.entity(cam).insert(Exposure { ev100 });
    if std::env::var("SOD_R3B_DIAG").is_ok() {
        let g = &frame.grading;
        eprintln!(
            "R3B-LIGHT: sun_rad {:?} lum {:.3} | sky {:?} ground {:?} | scene_lum {:.3} \
             adapted {:.4} exp {:.4} ev100 {:.2} | grade sat {:.3} contrast {:.3} bright {:.3} \
             gamma {:.3} min_in {:.3} max_in {:.3} min_out {:.3} max_out {:.3} photo_d {:.3}",
            frame.sun_radiance,
            luminance(frame.sun_radiance),
            frame.sky_ambient,
            frame.ambient_ground,
            frame.scene_luminance,
            frame.exposure.adapted_luminance,
            e,
            ev100,
            g.saturation,
            g.contrast,
            g.brightness,
            g.levels_gamma,
            g.levels_min_in,
            g.levels_max_in,
            g.levels_min_out,
            g.levels_max_out,
            g.photofilter_density,
        );
    }

    // --- tone map + colour grade ------------------------------------------
    if std::env::var("SOD_NOTONEMAP").is_ok() {
        commands.entity(cam).insert(Tonemapping::None);
    } else {
        commands.entity(cam).insert(Tonemapping::KhronosPbrNeutral);
    }
    if std::env::var("SOD_NOGRADE").is_err() {
        commands.entity(cam).insert(grading_from(&frame.grading));
    }

    // --- level lights ------------------------------------------------------
    let spawned = spawn_level_lights(&mut commands, lighting.as_ref(), sh);

    info!(
        "[light_bridge] hour {:.2}: sun dir {sun_dir:?} sun_col {sun_display:?} \
         illuminance {:.0} lx, ambient {:.0} cd/m2, ev100 {ev100:.2}, fog {fog_color:?}, \
         {spawned} level lights within {LIGHT_RADIUS_M:.0}m",
        lighting.hour,
        luminance(frame.sun_radiance) * SUN_LUX_PER_UNIT,
        sky_lum * SKY_LUX_PER_UNIT,
    );
    lighting_applied.0 = true;
}

/// Map the recovered grade onto Bevy's `ColorGrading` where the models agree.
///
/// Bevy has no per-hue selective colour or photo filter, so only the global
/// saturation/contrast/brightness and the levels are carried over. The
/// level's strong orange photo filter is intentionally not applied: the
/// built-in HDR chain already makes the authored warm sun read as neutral
/// daylight, and reproducing the filter would tint the whole scene orange.
fn grading_from(g: &sod_core::render_light::GradingParams) -> ColorGrading {
    let brightness_stops = if g.brightness > 1e-4 {
        g.brightness.log2()
    } else {
        0.0
    };
    ColorGrading {
        global: ColorGradingGlobal {
            exposure: brightness_stops,
            post_saturation: g.saturation.clamp(0.0, 4.0),
            ..default()
        },
        shadows: ColorGradingSection {
            contrast: g.contrast,
            // Photoshop levels gamma brightens midtones for gamma > 1; the
            // ASC-CDL `gamma` is the exponent, so invert it.
            gamma: if g.levels_gamma > 1e-4 {
                1.0 / g.levels_gamma
            } else {
                1.0
            },
            gain: g.levels_max_out,
            lift: g.levels_min_out,
            ..default()
        },
        midtones: ColorGradingSection {
            contrast: g.contrast,
            gamma: if g.levels_gamma > 1e-4 {
                1.0 / g.levels_gamma
            } else {
                1.0
            },
            gain: g.levels_max_out,
            lift: g.levels_min_out,
            ..default()
        },
        highlights: ColorGradingSection {
            contrast: g.contrast,
            gamma: if g.levels_gamma > 1e-4 {
                1.0 / g.levels_gamma
            } else {
                1.0
            },
            gain: g.levels_max_out,
            lift: g.levels_min_out,
            ..default()
        },
    }
}

/// Spawn the nearest live level lights around the player start.
///
/// `LSUpdate` (`0x15ba8a0`) culls by distance and `level_light_params`
/// already applies the bounce time window, so this only adds the render-side
/// cap. Returns how many were spawned.
fn spawn_level_lights(
    commands: &mut Commands,
    lighting: &crate::LevelLighting,
    shadows: ShadowSettings,
) -> usize {
    let Some(lv) = &lighting.lights else {
        return 0;
    };
    let hour = lighting.hour;
    let center = lighting.center;

    let mut picks: Vec<(f32, Vec3, LevelLightParams, [f32; 3])> = Vec::new();
    for light in &lv.lights {
        let Some(p) = sod_core::render_light::level_light_params(lv, light, hour) else {
            continue;
        };
        // Cubemap-only entries need the deferred cubemap pass; not ported.
        if p.kind == LightKind::Cubemap {
            continue;
        }
        let pos = cry_to_bevy(p.position);
        let dist = pos.distance(center);
        if dist > LIGHT_RADIUS_M + p.radius {
            continue;
        }
        picks.push((dist, pos, p, light.direction));
    }
    picks.sort_by(|a, b| a.0.total_cmp(&b.0));
    picks.truncate(MAX_LEVEL_LIGHTS);

    let mut spawned = 0;
    let mut shadow_casters = 0;
    for (_, pos, p, euler) in &picks {
        // The blend colour can be an HDR magnitude; split it into a unit hue
        // and fold the magnitude into the luminous power.
        let rgb = Vec3::new(p.color[0].max(0.0), p.color[1].max(0.0), p.color[2].max(0.0));
        let lum = luminance([rgb.x, rgb.y, rgb.z]);
        if lum < 1e-4 {
            continue;
        }
        let color = Color::linear_rgb(rgb.x / lum, rgb.y / lum, rgb.z / lum);
        let intensity = p.intensity.max(0.0) * LUMEN_PER_UNIT * lum;
        let range = p.radius.max(1.0);
        // The `.lights` record has no explicit shadow bit (only projector /
        // cubemap / bounce / geometry / active), so shadow casting follows the
        // kind: authored projector spot lights cast, plain omni lights do not
        // (CryEngine point-light cube shadows are the expensive path and are
        // off in the shipped config). Capped to bound the frame cost.
        let cast = shadows.gsm_lods > 0
            && std::env::var("SOD_NOSHADOWS").is_err()
            && p.kind == LightKind::Projector
            && shadow_casters < MAX_SHADOW_LIGHTS;
        if cast {
            shadow_casters += 1;
        }
        if p.kind == LightKind::Projector {
            commands.spawn((
                SpotLight {
                    color,
                    intensity,
                    range,
                    radius: 0.1,
                    shadow_maps_enabled: cast,
                    shadow_depth_bias: shadows.depth_bias(),
                    shadow_normal_bias: shadows.normal_bias(),
                    outer_angle: PROJECTOR_OUTER_ANGLE,
                    inner_angle: PROJECTOR_OUTER_ANGLE * PROJECTOR_INNER_FRACTION,
                    ..default()
                },
                Transform::from_translation(*pos)
                    .looking_to(projector_forward(*euler), Vec3::Y),
            ));
        } else {
            commands.spawn((
                PointLight {
                    color,
                    intensity,
                    range,
                    radius: 0.1,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_translation(*pos),
            ));
        }
        spawned += 1;
    }
    spawned
}

fn log_frame_time(time: Res<Time>, mut stats: ResMut<FrameStats>) {
    stats.frames += 1;
    stats.accum += time.delta_secs();
    if stats.frames >= stats.last_log + 100 {
        let avg = stats.accum / (stats.frames - stats.last_log) as f32;
        info!(
            "[light_bridge] frame {} avg {:.2} ms ({:.0} fps)",
            stats.frames,
            avg * 1000.0,
            1.0 / avg.max(1e-6)
        );
        stats.last_log = stats.frames;
        stats.accum = 0.0;
    }
}
