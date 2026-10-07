//! CryEngine 3 post-processing bridge (workerR3).
//!
//! The original renderer runs a fixed HDR post chain described by the
//! `sys_spec_postprocessing.cfg` CVarGroup and implemented in
//! `hwscripts/cryfx/HDRPostProcess.cfx` + `posteffects.cfx`. The recovered
//! numeric settings live in [`sod_core::render_light::PostSettings`]; this
//! module turns them into Bevy components:
//!
//! * `Hdr` — the scene colour must be linear HDR for bloom / eye adaptation.
//! * [`Bloom`] — the bright pass + additive glow (`r_HDRBrightLevel`,
//!   `r_HDRBrightOffset`, `r_HDRBrightThreshold`, `r_HDRBloomMul`).
//! * [`DepthOfField`] — `r_DepthOfField` (on at the shipped High group).
//! * [`ScreenSpaceAmbientOcclusion`] — `r_ssao` / `r_SSAOQuality`
//!   (`r_SSAOObscuranceRadius` / `Omega` have no Bevy equivalent).
//! * [`ShadowFilteringMethod`] — `r_ShadowsPCFiltering` (Gaussian PCF).
//! * [`CryPostMaterial`] — our own fullscreen WGSL (`post.wgsl`) for the parts
//!   Bevy does not ship: screen-space sun shafts (`r_sunshafts`) and the HDR
//!   blue shift / film grain from `HDRPostProcess.cfx`.
//!
//! The recovered exposure / eye adaptation (`r_HDRLevel`, `r_HDROffset`,
//! `r_EyeAdaptationBase/Factor`) and the tone map are already applied by
//! `light_bridge` (camera `Exposure` + `Tonemapping`); the filmic s-curve is
//! Bevy's neutral tonemap standing in for `r_HDRRendering=2`, with the sun
//! colour pre-mapped by [`sod_core::render_light::tonemap`]. Colour grading
//! (the level's colour chart) is carried by Bevy's `ColorGrading` in
//! `light_bridge`.
//!
//! Spec level: `SOD_SPEC=1..4` selects the shipped group (`sys_spec_full.cfg`
//! `[default]=3` = High); anything else falls back to High.

use std::sync::OnceLock;

use bevy::camera::Hdr;
use bevy::core_pipeline::fullscreen_material::{FullscreenMaterial, FullscreenMaterialPlugin};
use bevy::light::{DirectionalLight, DirectionalLightShadowMap, NotShadowCaster, ShadowFilteringMethod};
use bevy::pbr::ScreenSpaceAmbientOcclusion;
use bevy::pbr::ScreenSpaceAmbientOcclusionQualityLevel;
use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
use bevy::post_process::dof::DepthOfField;
use bevy::prelude::*;
use bevy::ecs::schedule::ScheduleConfigs;
use bevy::ecs::system::BoxedSystem;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::mesh::VertexAttributeValues;
use bevy::render::render_resource::ShaderType;
use bevy::render::view::Msaa;
use bevy::shader::{Shader, ShaderRef};
use sod_core::render_light::{PostSettings, ShadowSettings, SPEC_DEFAULT};

/// The custom post shader, embedded so the binary is self-contained (the same
/// pattern the terrain / character material extensions use).
pub const POST_WGSL: &str = include_str!("post.wgsl");

static POST_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

/// Register the WGSL as a shader asset before any post pipeline is built.
pub fn install_post_shader(app: &mut App) {
    let shader = Shader::from_wgsl(POST_WGSL, "sod_post.wgsl");
    let handle = app
        .world_mut()
        .resource_mut::<Assets<Shader>>()
        .add(shader);
    let _ = POST_SHADER.set(handle);
}

/// The uniform read by `post.wgsl`. Layout must match the WGSL struct.
#[derive(Component, Clone, Copy, Default, ShaderType, ExtractComponent)]
pub struct CryPostMaterial {
    /// xy sun screen uv, z visible flag, w `r_sunshafts` strength.
    pub sun: Vec4,
    /// x `r_HDRBlueShift`, y `r_HDRGrainAmount`, z frame counter, w reserved.
    pub params: Vec4,
    /// rgb sun tint for the shafts, a reserved.
    pub tint: Vec4,
    /// x sample-count normaliser, y..w reserved.
    pub misc: Vec4,
}

impl FullscreenMaterial for CryPostMaterial {
    fn fragment_shader() -> ShaderRef {
        match POST_SHADER.get() {
            Some(h) => ShaderRef::Handle(h.clone()),
            None => ShaderRef::Default,
        }
    }

    /// Run after bloom and before tonemapping.
    ///
    /// Bevy's bloom is `.before(tonemapping).in_set(PostProcess)`, the same set
    /// and the same only-ordering as the default here. Without an explicit edge
    /// the graph is free to interleave this pass with bloom's multi-pass
    /// ping-pong, so this pass would sample a bloom intermediate (bright) and
    /// white out the scene. Ordering after bloom makes the source the
    /// bloom-composited HDR scene, which is what the tone-mapped result
    /// expects.
    fn schedule_configs(
        system: ScheduleConfigs<BoxedSystem>,
    ) -> ScheduleConfigs<BoxedSystem> {
        system
            .in_set(bevy::core_pipeline::Core3dSystems::PostProcess)
            .after(bevy::post_process::bloom::bloom)
            .before(bevy::core_pipeline::tonemapping::tonemapping)
    }
}

/// Resolved spec level + settings, shared with `light_bridge` through the
/// public helper [`spec_from_env`].
#[derive(Resource)]
pub struct PostConfig {
    pub spec: u32,
    pub settings: PostSettings,
    pub shadows: ShadowSettings,
}

/// Read `SOD_SPEC` (1..4), defaulting to the shipped High group.
pub fn spec_from_env() -> u32 {
    match std::env::var("SOD_SPEC").ok().and_then(|v| v.trim().parse::<u32>().ok()) {
        Some(s) if matches!(s, 1 | 2 | 3 | 4) => s,
        _ => SPEC_DEFAULT,
    }
}

#[derive(Resource, Default)]
struct PostApplied(bool);

pub struct PostPlugin;

impl Plugin for PostPlugin {
    fn build(&self, app: &mut App) {
        let spec = spec_from_env();
        let settings = PostSettings::for_spec(spec);
        let shadows = ShadowSettings::for_spec(spec);
        info!(
            "[post] spec {spec} (High=3): hdr {} bloom {:.3} ssao {} dof {} sunshafts {} \
             blue_shift {:.2} shadow_map {}px {} cascades",
            settings.hdr_rendering,
            settings.bloom_intensity(),
            settings.ssao,
            settings.dof,
            settings.sunshafts,
            settings.hdr_blue_shift,
            shadows.max_tex_res,
            shadows.gsm_lods,
        );
        if std::env::var("SOD_R3B_DIAG").is_ok() {
            app.insert_resource(bevy::render::error_handler::RenderErrorHandler(|e, world, _| {
                eprintln!("R3B-DIAG render error: {:?} :: {} :: {:?}", e.ty, e.description, e.source);
                world.write_message(bevy::app::AppExit::error());
                bevy::render::error_handler::RenderErrorPolicy::StopRendering
            }));
        }
        app.insert_resource(PostConfig { spec, settings, shadows })
            .init_resource::<PostApplied>()
            .add_plugins(FullscreenMaterialPlugin::<CryPostMaterial>::default())
            .add_systems(Update, (apply_post_chain, update_sun_uniform).chain())
            .add_systems(Last, guard_shadow_casters);
    }
}

/// Insert the post components on the camera once it exists. The sun's
/// shadow-map resolution is a global resource, so it is set here too.
fn apply_post_chain(
    mut applied: ResMut<PostApplied>,
    cfg: Res<PostConfig>,
    camera: Query<Entity, With<Camera3d>>,
    mut commands: Commands,
) {
    if applied.0 {
        return;
    }
    let Ok(cam) = camera.single() else {
        return;
    };
    // Debug A/B switches for per-pass profiling (see REPORT.md). Each turns
    // off one pass while leaving the rest of the chain in place.
    let debug_off = |name: &str| std::env::var(name).is_ok();

    let s = &cfg.settings;
    let sh = &cfg.shadows;

    commands.insert_resource(DirectionalLightShadowMap {
        size: sh.max_tex_res.max(256) as usize,
    });

    let mut e = commands.entity(cam);
    if std::env::var("SOD_NOHDR").is_err() {
        e.insert(Hdr);
    }
    // `r_ShadowsPCFiltering = 1` -> the Gaussian PCF kernel.
    e.insert(ShadowFilteringMethod::Gaussian);

    // Bloom: the CryEngine bright pass is `max(0, x - threshold)` mapped
    // through `x/(offset+x)`; Bevy's prefilter threshold is the same kind of
    // knee, so map the authored pair onto the [0,1] fraction of the HDR range
    // it represents.
    if s.hdr_rendering >= 1 && !debug_off("SOD_NOBLOOM") {
        let knee = if s.hdr_bright_threshold + s.hdr_bright_offset > 0.0 {
            (s.hdr_bright_threshold / (s.hdr_bright_threshold + s.hdr_bright_offset))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        e.insert(Bloom {
            intensity: s.bloom_intensity(),
            prefilter: BloomPrefilter { threshold: knee, threshold_softness: 0.5 },
            composite_mode: BloomCompositeMode::Additive,
            ..default()
        });
    }

    // Depth of field: on at High (`r_DepthOfField = 3`). Kept physically
    // shallow so it reads as the engine's mild gameplay DOF rather than a
    // heavy portrait blur.
    if s.dof > 0 && !debug_off("SOD_NODOFF") {
        e.insert(DepthOfField {
            focal_distance: 12.0,
            aperture_f_stops: 16.0,
            max_circle_of_confusion_diameter: 32.0,
            ..default()
        });
    }

    // SSAO: `r_ssao` 3 (High) / 5 (Ultra). Bevy's SSAO needs the depth+normal
    // prepass and MSAA off; the authored radius/omega are not exposed by Bevy.
    if s.ssao > 0 && !debug_off("SOD_NOSSAO") {
        let quality = match s.ssao_quality {
            0 => ScreenSpaceAmbientOcclusionQualityLevel::Low,
            1 => ScreenSpaceAmbientOcclusionQualityLevel::Medium,
            2 => ScreenSpaceAmbientOcclusionQualityLevel::High,
            _ => ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
        };
        e.insert(Msaa::Off);
        e.insert(ScreenSpaceAmbientOcclusion {
            quality_level: quality,
            constant_object_thickness: s.ssao_radius,
        });
    }

    if !debug_off("SOD_NOPOST") {
        e.insert(CryPostMaterial::default());
    }
    applied.0 = true;
}

/// Make normal-less meshes safe for Bevy's shadow *and* normal prepasses.
///
/// Turning on the sun's cascaded shadow maps and SSAO makes Bevy build a
/// prepass pipeline per visible mesh. Its vertex stage reads
/// `ATTRIBUTE_NORMAL`, so a mesh with no normal attribute (the unlit particle
/// quads from `particle_bridge.rs`, and the terrain root whose setup mesh is
/// replaced with an empty placeholder by `terrain_bridge.rs`) makes the vertex
/// stage emit no `@location(0)`; wgpu then rejects `pbr_prepass_pipeline` and
/// aborts the frame, which showed as the flat grey capture.
///
/// This checks every `Mesh3d` each frame rather than relying on `Added<Mesh3d>`
/// or asset events: `terrain_bridge` replaces the root mesh *in place* (the
/// entity's `Mesh3d` component never changes) and streaming spawns meshes from
/// a background thread, so both filters raced the first render and let the
/// validation error through. A mesh with real geometry gets computed normals;
/// an empty placeholder gets a canonical empty position+normal layout.
/// Referencing entities are marked `NotShadowCaster` (the authored unlit
/// geometry never wrote the sun shadow map).
fn guard_shadow_casters(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    q: Query<(Entity, &Mesh3d)>,
) {
    for (e, m) in &q {
        let id = m.0.id();
        // A mesh is bad when it has no normal attribute at all, or when the
        // normal attribute is empty while positions exist (some decoders emit
        // an empty normal list). Both make the prepass vertex interface miss
        // the world-normal varying.
        let (has_normal, bad) = match meshes.get(id) {
            Some(mesh) => {
                let nrm = mesh
                    .try_attribute_option(Mesh::ATTRIBUTE_NORMAL)
                    .ok()
                    .flatten();
                let pos = mesh
                    .try_attribute_option(Mesh::ATTRIBUTE_POSITION)
                    .ok()
                    .flatten();
                let bad = match (nrm, pos) {
                    (Some(n), Some(p)) => n.is_empty() && !p.is_empty(),
                    (Some(n), None) => !n.is_empty(),
                    _ => false,
                };
                (nrm.is_some(), bad)
            }
            None => (true, false),
        };
        if has_normal && !bad {
            continue;
        }
        let Some(mut mesh) = meshes.get_mut(id) else { continue };
        let has_positions = mesh
            .try_attribute_option(Mesh::ATTRIBUTE_POSITION)
            .ok()
            .flatten()
            .map(|p| !p.is_empty())
            .unwrap_or(false);
        if has_positions {
            // `try_compute_normals` is flat for non-indexed geometry and
            // smooth for indexed. Unlit materials ignore the result.
            let _ = mesh.try_compute_normals();
        } else {
            // Empty placeholder: declare a canonical empty layout so the
            // prepass pipeline's vertex interface matches (nothing draws).
            if mesh.try_attribute_option(Mesh::ATTRIBUTE_POSITION).ok().flatten().is_none() {
                mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(Vec::new()));
            }
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(Vec::new()));
        }
        commands.entity(e).insert(NotShadowCaster);
    }
}

/// Project the sun to screen space each frame and push the post uniforms.
fn update_sun_uniform(
    cfg: Res<PostConfig>,
    time: Res<Time>,
    cam: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    sun: Query<&GlobalTransform, With<DirectionalLight>>,
    mut mats: Query<&mut CryPostMaterial>,
) {
    let Ok((camera, cam_tf)) = cam.single() else {
        return;
    };
    let Ok(mut mat) = mats.single_mut() else {
        return;
    };

    let mut uv = Vec2::ZERO;
    let mut visible = 0.0;
    if let Ok(sun_tf) = sun.single() {
        // A directional light's forward points along the light; the sun itself
        // sits opposite, effectively at infinity.
        let sun_world = cam_tf.translation() - sun_tf.forward() * 10_000.0;
        if let Some(ndc) = camera.world_to_ndc(cam_tf, sun_world) {
            if ndc.z > 0.0 && ndc.z < 1.0 && ndc.x.abs() < 1.5 && ndc.y.abs() < 1.5 {
                uv = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                visible = 1.0;
            }
        }
    }

    let sunshafts = cfg.settings.sunshafts && std::env::var("SOD_NOSHAFTS").is_err();
    mat.sun = Vec4::new(uv.x, uv.y, visible, if sunshafts { 1.0 } else { 0.0 });
    mat.tint = Vec4::new(1.0, 0.97, 0.9, 0.0);
    let blue = if std::env::var("SOD_NOBLUE").is_ok() {
        0.0
    } else {
        cfg.settings.hdr_blue_shift
    };
    mat.params = Vec4::new(blue, cfg.settings.hdr_grain, time.elapsed_secs() * 60.0, 0.0);
    mat.misc = if std::env::var("SOD_PASSTHRU").is_ok() {
        Vec4::ZERO
    } else {
        Vec4::ONE
    };
}
