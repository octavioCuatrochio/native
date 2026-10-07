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
use bevy::render::extract_component::ExtractComponent;
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
    let s = &cfg.settings;
    let sh = &cfg.shadows;

    commands.insert_resource(DirectionalLightShadowMap {
        size: sh.max_tex_res.max(256) as usize,
    });

    let mut e = commands.entity(cam);
    e.insert(Hdr);
    // `r_ShadowsPCFiltering = 1` -> the Gaussian PCF kernel.
    e.insert(ShadowFilteringMethod::Gaussian);

    // Bloom: the CryEngine bright pass is `max(0, x - threshold)` mapped
    // through `x/(offset+x)`; Bevy's prefilter threshold is the same kind of
    // knee, so map the authored pair onto the [0,1] fraction of the HDR range
    // it represents.
    if s.hdr_rendering >= 1 {
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
    if s.dof > 0 {
        e.insert(DepthOfField {
            focal_distance: 12.0,
            aperture_f_stops: 16.0,
            max_circle_of_confusion_diameter: 32.0,
            ..default()
        });
    }

    // SSAO: `r_ssao` 3 (High) / 5 (Ultra). Bevy's SSAO needs the depth+normal
    // prepass and MSAA off; the authored radius/omega are not exposed by Bevy.
    if s.ssao > 0 {
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

    e.insert(CryPostMaterial::default());
    applied.0 = true;
}

/// Mark meshes without vertex normals as non-shadow-casters.
///
/// Turning on the sun's cascaded shadow maps makes Bevy build a shadow
/// (prepass) pipeline per shadow-casting mesh. That pipeline's fragment stage
/// reads the world normal, so a mesh with no `ATTRIBUTE_NORMAL` (the unlit
/// particle quads built by `particle_bridge.rs`, a `Mesh` with only position /
/// uv / colour) makes the vertex stage emit no `@location(0)` and wgpu rejects
/// the pipeline with a validation error that aborts the frame. CryEngine's
/// unlit/particle geometry likewise does not write the sun shadow map, so such
/// meshes are marked `NotShadowCaster`. The `Added` filter keeps the scan to
/// newly spawned meshes.
fn guard_shadow_casters(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    q: Query<(Entity, &Mesh3d), (Added<Mesh3d>, Without<NotShadowCaster>)>,
) {
    for (e, m) in &q {
        if let Some(mesh) = meshes.get(&m.0) {
            if !mesh.attributes().any(|(a, _)| *a == Mesh::ATTRIBUTE_NORMAL) {
                commands.entity(e).insert(NotShadowCaster);
            }
        }
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

    mat.sun = Vec4::new(
        uv.x,
        uv.y,
        visible,
        if cfg.settings.sunshafts { 1.0 } else { 0.0 },
    );
    mat.tint = Vec4::new(1.0, 0.97, 0.9, 0.0);
    mat.params = Vec4::new(
        cfg.settings.hdr_blue_shift,
        cfg.settings.hdr_grain,
        time.elapsed_secs() * 60.0,
        0.0,
    );
    mat.misc = Vec4::ONE;
}
