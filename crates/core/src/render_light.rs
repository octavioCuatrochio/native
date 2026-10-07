//! CryEngine 3 lighting math for State of Decay (YOSE), reconstructed from the
//! shipped renderer and shaders.
//!
//! This module is deliberately engine-agnostic: it turns a sampled
//! [`TodSample`] plus the level [`Environment`] into the linear-light values a
//! renderer needs ([`LightingFrame`]) and reproduces the HDR post-process chain
//! (eye adaptation, exposure, tone mapping, colour grading) that the original
//! applies after the scene is lit. Nothing here knows about Bevy; the lead
//! wires the outputs into renderer resources.
//!
//! # Evidence
//!
//! * The 79 time-of-day variables and their fixed array slots are registered by
//!   `FUN_0051fef0` at `0x51fef0` (index is the 5th argument of `FUN_0051f470`;
//!   scalar slots use index `n`, colour slots `n`). The names below are the
//!   exact `Variable Name` strings.
//! * HDR eye adaptation, exposure, blue shift and the tone-map operator come
//!   from `Engine/shaders.pak` shader `hwscripts/cryfx/HDRPostProcess.cfx`,
//!   functions `EyeAdaption` (line 176), `ApplyBlueShift` (532),
//!   `TonemapOperator` (559), `ExpMapping` (567) and the final pass
//!   `HDRFinalScenePS` (850). The relevant variant is the "exponential tone map
//!   with s-curve" path (`%_RT_SAMPLE2`), selected by `r_HDRRendering=2`
//!   (default registered at `0x011a1a30`).
//! * Exposure form `HDRLevel / (1 + HDRLevelOffset * adaptedLum)` is
//!   `ExpMapping`; cvar defaults (`r_HDRLevel=6`, `r_HDROffset=10`,
//!   `r_HDRBrightLevel=1.25`, `r_HDRBrightThreshold=6`, `r_HDRBrightOffset=5`,
//!   `r_EyeAdaptationFactor=0.85`, `r_HDRSCurveMax=0.95`) are the registration
//!   table at `0x011a1a30` (descriptions carry the numeric defaults).
//! * Fog is `volumetricfog.cfi`: `ComputeVolumetricFog` (line 17) and
//!   `GetVolumetricFogColorDistanceBased` (line 61), driven by
//!   `PB_VolumetricFogParams`/`PB_VolumetricFogRampParams`/`PB_FogColGradColBase`.
//! * The final colour transform is a CPU-built 4x4 `mColorMatrix` applied in
//!   `posteffects.cfx` (line 665); the TOD "Color Filter"/"Color Levels"/
//!   "Selective Color"/"Filters" variables feed it.
//!
//! # What is approximate
//!
//! The engine computes a few intermediate scales on the CPU that were not
//! individually traced (the HDR range rescale `HDRParams2.y`, the exact
//! luminance-map encoding, and the precise CPU construction of `mColorMatrix`).
//! Where that matters we say so inline and use the documented defaults. The
//! numbers a renderer ultimately needs are all here.

use crate::level_lights::{BounceLightInfo, LevelLight, LevelLights, LightStyle};
use crate::tod::{Environment, TodSample};

/// Rec.709 luminance weights. The HDR shader computes tone-map luminance with
/// `dot(cColor, float3(0.2126, 0.7152, 0.0722))` (`ExpMapping`, line 615).
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------

#[inline]
fn saturate(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

#[inline]
fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

#[inline]
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Rec.709 luminance.
#[inline]
pub fn luminance(c: [f32; 3]) -> f32 {
    c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = saturate((x - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Frame: linear light for one instant
// ---------------------------------------------------------------------------

/// Fog parameters for one frame. Mirrors the shader's `PB_...` constants:
/// `color` is the ground-level gradient base, `color_top` the zenith end (the
/// `Fog color (top)` variable), `density`/`atmosphere_height`/`density_offset`
/// map to `vfParams` and `ramp_*` to `vfRampParams`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogParams {
    /// `Fog color` * `Fog color multiplier` (linear).
    pub color: [f32; 3],
    /// `Fog color (top)` * its multiplier (linear).
    pub color_top: [f32; 3],
    /// `Volumetric fog: Global density` (`vfParams.z`).
    pub density: f32,
    /// `Volumetric fog: Atmosphere height` (metres); the shader's
    /// `atmosphereScale` is `1/height`.
    pub atmosphere_height: f32,
    /// `Volumetric fog: Density offset (in reality 0)` (`vfParams.w`).
    pub density_offset: f32,
    /// `Volumetric fog: Ramp start` (metres).
    pub ramp_start: f32,
    /// `Volumetric fog: Ramp end` (metres).
    pub ramp_end: f32,
    /// `Volumetric fog: Ramp influence`.
    pub ramp_influence: f32,
    /// Static `Environment.Fog.ViewDistance`.
    pub view_distance: f32,
    /// `Environment.Fog.UseFogColorGradient`.
    pub use_gradient: bool,
}

impl Default for FogParams {
    fn default() -> Self {
        FogParams {
            color: [0.0, 0.0, 0.0],
            color_top: [0.0, 0.0, 0.0],
            density: 1.0,
            atmosphere_height: 4000.0,
            density_offset: 0.0,
            ramp_start: 0.0,
            ramp_end: 1.0,
            ramp_influence: 0.0,
            view_distance: 0.0,
            use_gradient: false,
        }
    }
}

/// Eye-adaptation + exposure state. `adapted_luminance` is the value the
/// renderer would store in the luminance map for this frame (already in the
/// range-reduced display domain; see [`LightingFrame::from_tod`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExposureParams {
    /// `r_EyeAdaptationFactor`, default `0.85` (`0x011a1a30`).
    pub eye_adapt_factor: f32,
    /// `r_EyeAdaptationBase`, default `0.25` (shader comment at line 181).
    pub eye_adapt_base: f32,
    /// `r_HDRLevel`, default `6.0`.
    pub level: f32,
    /// `r_HDROffset`, default `10.0`.
    pub offset: f32,
    /// Range-reduced scene luminance used for adaptation.
    pub adapted_luminance: f32,
}

impl Default for ExposureParams {
    fn default() -> Self {
        ExposureParams {
            eye_adapt_factor: 0.85,
            eye_adapt_base: 0.25,
            level: 6.0,
            offset: 10.0,
            adapted_luminance: 0.0,
        }
    }
}

impl ExposureParams {
    /// `EyeAdaption` (HDRPostProcess.cfx line 176): blend the adaptation base
    /// toward the scene luminance, then `ExpMapping`'s exposure
    /// `HDRLevel / (1 + HDROffset * adapted)`.
    pub fn adapted(&self, scene_luminance: f32) -> f32 {
        lerp(self.eye_adapt_base, scene_luminance, self.eye_adapt_factor)
    }

    /// The scalar the tone mapper multiplies linear radiance by.
    pub fn exposure(&self) -> f32 {
        self.level / (1e-6 + 1.0 + self.offset * self.adapted_luminance)
    }
}

/// Tone-map controls. The default reproduces the `%_RT_SAMPLE2` exponential +
/// s-curve path of `ExpMapping`, which is what `r_HDRRendering=2` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemapMode {
    /// `1 - exp2(-x)` per channel followed by a per-channel smoothstep
    /// (`ExpMapping` `%_RT_SAMPLE2`). Per-channel, so bright warm light
    /// saturates to neutral rather than staying orange.
    ExponentialSCurve,
    /// Luminance-only Reinhard with white point `2.4` (`TonemapOperator`,
    /// line 559). Hue preserving.
    ReinhardLuminance,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TonemapParams {
    pub mode: TonemapMode,
    pub exposure: ExposureParams,
    /// `r_HDRSCurveMin`, default `0.0`.
    pub s_curve_min: f32,
    /// `r_HDRSCurveMax`, default `0.95`.
    pub s_curve_max: f32,
    /// `r_HDRSaturation`, default `1.0`.
    pub saturation: f32,
    /// `r_HDRBrightness`, default `1.0`.
    pub brightness: f32,
    /// `HDR blueshift` TOD colour, default `[0.8,0.8,1.4]`.
    pub blueshift: [f32; 3],
    /// `HDR blueshift threshold` TOD scalar.
    pub blueshift_threshold: f32,
    /// Reinhard white point; the shader hard-codes `2.4`.
    pub white_point: f32,
}

impl Default for TonemapParams {
    fn default() -> Self {
        TonemapParams {
            mode: TonemapMode::ExponentialSCurve,
            exposure: ExposureParams::default(),
            s_curve_min: 0.0,
            s_curve_max: 0.95,
            saturation: 1.0,
            brightness: 1.0,
            blueshift: [0.8, 0.8, 1.4],
            blueshift_threshold: 0.0,
            white_point: 2.4,
        }
    }
}

/// Post tone-map colour grading. `mColorMatrix` in `posteffects.cfx` is the
/// combined saturation/contrast/brightness transform; levels, selective colour
/// and the photo filter are the surrounding Photoshop-style operators the TOD
/// block exposes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradingParams {
    /// `Color: saturation`, 1.0 = identity.
    pub saturation: f32,
    /// `Color: contrast`, 1.0 = identity.
    pub contrast: f32,
    /// `Color: brightness`, 1.0 = identity.
    pub brightness: f32,
    /// `Levels: min input` (0..1).
    pub levels_min_in: f32,
    /// `Levels: gamma`.
    pub levels_gamma: f32,
    /// `Levels: max input` (0..1).
    pub levels_max_in: f32,
    /// `Levels: minimum output`.
    pub levels_min_out: f32,
    /// `Levels: max output`.
    pub levels_max_out: f32,
    /// `Selective Color: color` reference RGB.
    pub selective_color: [f32; 3],
    /// `Selective Color` adjustments `[cyans, magentas, yellows, blacks]`
    /// (percent, as authored, e.g. 17.5 = +17.5%).
    pub selective: [f32; 4],
    /// `Filters: photofilter color` (0..1).
    pub photofilter_color: [f32; 3],
    /// `Filters: photofilter density`.
    pub photofilter_density: f32,
    /// `Filters: grain`.
    pub grain: f32,
    /// `Filters: sharpening`.
    pub sharpening: f32,
}

impl Default for GradingParams {
    fn default() -> Self {
        GradingParams {
            saturation: 1.0,
            contrast: 1.0,
            brightness: 1.0,
            levels_min_in: 0.0,
            levels_gamma: 1.0,
            levels_max_in: 1.0,
            levels_min_out: 0.0,
            levels_max_out: 1.0,
            selective_color: [0.0, 0.0, 0.0],
            selective: [0.0; 4],
            photofilter_color: [1.0, 1.0, 1.0],
            photofilter_density: 0.0,
            grain: 0.0,
            sharpening: 0.0,
        }
    }
}

/// Everything a renderer needs to light and post-process one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightingFrame {
    /// Direction the sunlight travels (unit, from sun toward the scene).
    pub sun_dir: [f32; 3],
    /// Direct sun radiance in linear RGB: `Sun color * Sun color multiplier`.
    /// This is intentionally warm at noon; the tone map below is what turns it
    /// into neutral daylight on screen.
    pub sun_radiance: [f32; 3],
    /// Sky/hemisphere ambient: `Sky color * Sky color multiplier`.
    pub sky_ambient: [f32; 3],
    /// Ground bounce ambient: `Ambient ground color * multiplier`.
    pub ambient_ground: [f32; 3],
    /// `Ambient min height` / `Ambient max height` for height-based ambient.
    pub ambient_min_height: f32,
    pub ambient_max_height: f32,
    pub fog: FogParams,
    pub exposure: ExposureParams,
    pub tonemap: TonemapParams,
    pub grading: GradingParams,
    /// Estimated scene luminance used for exposure (linear).
    pub scene_luminance: f32,
}

impl LightingFrame {
    /// Build a frame from a sampled time of day.
    ///
    /// `hour` is the wall clock in hours (0..24). `sample` must have been
    /// produced by [`crate::tod::TimeOfDay::sample`] at the same `hour` so the
    /// splines are evaluated consistently.
    ///
    /// The exposure's adapted luminance is the range-reduced scene luminance
    /// `L/(1+L)` rather than raw `L`, because the engine's luminance map is
    /// built after the HDR scene has already been range-reduced (`vSample *=
    /// HDRParams2.y` in `HDRFinalScenePS`, line 915). Without that reduction
    /// the bright noon sun would over-expose; with it the sun lands in the
    /// saturating part of the curve, which is why the original noon look is
    /// neutral and not solid orange.
    pub fn from_tod(sample: &TodSample, env: &Environment, hour: f32) -> LightingFrame {
        let sun_color = sample.color("Sun color").unwrap_or([1.0, 1.0, 1.0]);
        let sun_mult = sample.scalar("Sun color multiplier").unwrap_or(1.0);
        let sun_radiance = scale3(sun_color, sun_mult);

        let sky_color = sample.color("Sky color").unwrap_or([1.0, 1.0, 1.0]);
        let sky_mult = sample.scalar("Sky color multiplier").unwrap_or(1.0);
        let sky_ambient = scale3(sky_color, sky_mult);

        let ground_color = sample
            .color("Ambient ground color")
            .unwrap_or([0.0, 0.0, 0.0]);
        let ground_mult = sample
            .scalar("Ambient ground color multiplier")
            .unwrap_or(1.0);
        let ambient_ground = scale3(ground_color, ground_mult);

        let fog = FogParams {
            color: scale3(
                sample.color("Fog color").unwrap_or([0.0, 0.0, 0.0]),
                sample.scalar("Fog color multiplier").unwrap_or(1.0),
            ),
            color_top: scale3(
                sample.color("Fog color (top)").unwrap_or([0.0, 0.0, 0.0]),
                sample.scalar("Fog color (top) multiplier").unwrap_or(1.0),
            ),
            density: sample
                .scalar("Volumetric fog: Global density")
                .unwrap_or(1.0),
            atmosphere_height: sample
                .scalar("Volumetric fog: Atmosphere height")
                .unwrap_or(4000.0),
            density_offset: sample
                .scalar("Volumetric fog: Density offset (in reality 0)")
                .unwrap_or(0.0),
            ramp_start: sample.scalar("Volumetric fog: Ramp start").unwrap_or(0.0),
            ramp_end: sample.scalar("Volumetric fog: Ramp end").unwrap_or(1.0),
            ramp_influence: sample
                .scalar("Volumetric fog: Ramp influence")
                .unwrap_or(0.0),
            view_distance: env.fog.view_distance,
            use_gradient: env.fog.use_fog_color_gradient,
        };

        // Estimate the (linear) scene luminance from the direct and ambient
        // light the TOD authors. A hemisphere actually receives roughly half
        // the sky term; the constant is a calibration, not a recovered engine
        // value.
        let scene_luminance = luminance(add3(sun_radiance, scale3(sky_ambient, 0.5)));

        let mut exposure = ExposureParams::default();
        exposure.adapted_luminance = exposure.adapted(scene_luminance / (1.0 + scene_luminance));

        let tonemap = TonemapParams {
            exposure,
            saturation: 1.0,
            brightness: 1.0,
            blueshift: sample.color("HDR blueshift").unwrap_or([0.8, 0.8, 1.4]),
            blueshift_threshold: sample.scalar("HDR blueshift threshold").unwrap_or(0.0),
            ..TonemapParams::default()
        };

        // `Levels` are authored 0..255; the shader works in 0..1.
        let inv255 = 1.0 / 255.0;
        let grading = GradingParams {
            saturation: sample.scalar("Color: saturation").unwrap_or(1.0),
            contrast: sample.scalar("Color: contrast").unwrap_or(1.0),
            brightness: sample.scalar("Color: brightness").unwrap_or(1.0),
            levels_min_in: sample.scalar("Levels: min input").unwrap_or(0.0) * inv255,
            levels_gamma: sample.scalar("Levels: gamma").unwrap_or(1.0),
            levels_max_in: sample.scalar("Levels: max input").unwrap_or(255.0) * inv255,
            levels_min_out: sample.scalar("Levels: minimum output").unwrap_or(0.0) * inv255,
            levels_max_out: sample.scalar("Levels: max output").unwrap_or(255.0) * inv255,
            selective_color: sample
                .color("Selective Color: color")
                .unwrap_or([0.0, 0.0, 0.0]),
            selective: [
                sample.scalar("Selective Color: cyans").unwrap_or(0.0),
                sample.scalar("Selective Color: magentas").unwrap_or(0.0),
                sample.scalar("Selective Color: yellows").unwrap_or(0.0),
                sample.scalar("Selective Color: blacks").unwrap_or(0.0),
            ],
            photofilter_color: sample
                .color("Filters: photofilter color")
                .unwrap_or([1.0, 1.0, 1.0]),
            photofilter_density: sample.scalar("Filters: photofilter density").unwrap_or(0.0),
            grain: sample.scalar("Filters: grain").unwrap_or(0.0),
            sharpening: sample.scalar("Filters: sharpening").unwrap_or(0.0),
        };

        LightingFrame {
            sun_dir: env.lighting.sun_direction(hour),
            sun_radiance,
            sky_ambient,
            ambient_ground,
            ambient_min_height: sample.scalar("Ambient min height").unwrap_or(0.0),
            ambient_max_height: sample.scalar("Ambient max height").unwrap_or(1.0),
            fog,
            exposure,
            tonemap,
            grading,
            scene_luminance,
        }
    }
}

// ---------------------------------------------------------------------------
// Tone mapping
// ---------------------------------------------------------------------------

/// `ApplyBlueShift` (HDRPostProcess.cfx line 532). Blends toward
/// `target * luminance` by `saturate(threshold)`. In the shader the target is
/// `HDRParams4` whose alpha is the threshold and which is also multiplied by
/// the scene luminance.
fn apply_blue_shift(c: [f32; 3], p: &TonemapParams) -> [f32; 3] {
    let l = luminance(c);
    // `cBlueTarget *= fLum` scales the threshold (alpha) as well as the colour,
    // and the blend is `lerp(target, scene, saturate(alpha))`.
    let target = scale3(p.blueshift, l);
    lerp3(target, c, saturate(p.blueshift_threshold * l))
}

/// `TonemapOperator` (HDRPostProcess.cfx line 559): Reinhard with a white
/// point, `L * (1 + L/Lwhite^2) / (1 + L)`.
fn tonemap_operator(l: f32, lwhite: f32) -> f32 {
    let lwhite2 = lwhite * lwhite;
    (l * (1.0 + l / lwhite2)) / (1.0 + l)
}

/// Apply the HDR post-process tone map to a linear RGB colour.
///
/// `rgb_linear` is scene radiance (already multiplied by the light colours);
/// exposure and eye adaptation come from `params.exposure`. The returned value
/// is display-referred (`0..1`), before [`grade`].
pub fn tonemap(rgb_linear: [f32; 3], params: &TonemapParams) -> [f32; 3] {
    let f = params.exposure.exposure();
    let mut c = scale3(rgb_linear, f);
    c = apply_blue_shift(c, params);

    match params.mode {
        TonemapMode::ExponentialSCurve => {
            // ExpMapping `%_RT_SAMPLE2`: saturation/brightness about luminance,
            // exponential tonemap, then per-channel smoothstep.
            let l = luminance(c);
            c = scale3(lerp3([l, l, l], c, params.saturation), params.brightness);
            c = [
                1.0 - (-c[0]).exp2(),
                1.0 - (-c[1]).exp2(),
                1.0 - (-c[2]).exp2(),
            ];
            [
                smoothstep(params.s_curve_min, params.s_curve_max, c[0]),
                smoothstep(params.s_curve_min, params.s_curve_max, c[1]),
                smoothstep(params.s_curve_min, params.s_curve_max, c[2]),
            ]
        }
        TonemapMode::ReinhardLuminance => {
            // ExpMapping `#else`: tone map luminance only, rescale RGB.
            let l = luminance(c);
            if l <= 0.0 {
                return [0.0, 0.0, 0.0];
            }
            let nl = tonemap_operator(l, params.white_point);
            scale3(c, nl / l).map(saturate)
        }
    }
}

// ---------------------------------------------------------------------------
// Colour grading
// ---------------------------------------------------------------------------

/// Photoshop-style "Selective Color" adjustment for one of the primary hue
/// bands. `t` is a relative amount in percent (as authored, e.g. 17.5 for
/// +17.5%). The band weight is the normalized C/M/Y component the reference
/// colour contributes to the channel.
fn selective_adjust(c: [f32; 3], reference: [f32; 3], amount: f32) -> [f32; 3] {
    let a = amount / 100.0;
    // Cyan lives in G+B vs R, magenta in R+B vs G, yellow in R+G vs B.
    let cyan = saturate(1.0 - reference[0] - reference[2]);
    let magenta = saturate(1.0 - reference[0] - reference[1]);
    let yellow = saturate(1.0 - reference[1] - reference[2]);
    let mut out = c;
    out[0] = saturate(out[0] - a * magenta * out[0] + a * yellow * out[0]);
    out[1] = saturate(out[1] - a * cyan * out[1] + a * magenta * out[1]);
    out[2] = saturate(out[2] - a * yellow * out[2] + a * cyan * out[2]);
    out
}

/// Apply the post tone-map colour grade. Order matches the reference chain:
/// input levels + gamma + output levels, then saturation/contrast/brightness,
/// then selective colour and the photo filter. This is the operator that
/// `mColorMatrix` (posteffects.cfx line 665) implements for the colour-filter
/// part.
pub fn grade(rgb: [f32; 3], g: &GradingParams) -> [f32; 3] {
    let mut c = rgb;

    // Levels.
    let span = g.levels_max_in - g.levels_min_in;
    if span.abs() > f32::EPSILON {
        for ch in c.iter_mut() {
            *ch = saturate((*ch - g.levels_min_in) / span);
            if (g.levels_gamma - 1.0).abs() > f32::EPSILON && *ch > 0.0 {
                *ch = ch.powf(1.0 / g.levels_gamma);
            }
            *ch = lerp(g.levels_min_out, g.levels_max_out, *ch);
        }
    }

    // Saturation / contrast / brightness (the mColorMatrix transform).
    let l = luminance(c);
    c = lerp3([l, l, l], c, g.saturation);
    c = [
        (c[0] - 0.5) * g.contrast + 0.5,
        (c[1] - 0.5) * g.contrast + 0.5,
        (c[2] - 0.5) * g.contrast + 0.5,
    ];
    c = scale3(c, g.brightness);

    // Selective colour bands.
    c = selective_adjust(c, g.selective_color, g.selective[0]);
    c = selective_adjust(c, g.selective_color, g.selective[1]);
    c = selective_adjust(c, g.selective_color, g.selective[2]);

    // Photo filter: blend toward the filter colour by density.
    let d = saturate(g.photofilter_density);
    c = lerp3(c, mul3(c, g.photofilter_color), d);

    // Black band adjustment scales shadows.
    let black = g.selective[3] / 100.0;
    if black.abs() > f32::EPSILON {
        let l = luminance(c);
        let shadow = (1.0 - l).clamp(0.0, 1.0);
        c = scale3(c, 1.0 + black * shadow);
    }

    [saturate(c[0]), saturate(c[1]), saturate(c[2])]
}

// ---------------------------------------------------------------------------
// Level lights
// ---------------------------------------------------------------------------

/// The kind of a baked level light, from `flags` in `lights_*.lights`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightKind {
    /// Plain point/omni light.
    Point,
    /// `FLAG_PROJECTOR`: textured projector.
    Projector,
    /// `FLAG_CUBEMAP`: deferred cubemap.
    Cubemap,
    /// `FLAG_BOUNCE`: day/night window light.
    Bounce,
}

/// A light ready to upload. `intensity` is the style's `range_scale` (the
/// engine halves it at upload, `0x15bb8e0`); `projector_texture` is the
/// resolved path (never game XML).
#[derive(Debug, Clone, PartialEq)]
pub struct LevelLightParams {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    pub kind: LightKind,
    pub projector_texture: Option<String>,
}

/// Resolve one `.lights` entry to concrete light parameters at `hour`.
///
/// The extra `lights` argument (relative to the one in the brief) is needed to
/// reach the shared style and name tables; `LevelLight` alone carries only
/// indices.
///
/// * Non-bounce lights always resolve.
/// * Bounce lights are enabled only while
///   `time_start <= hour < time_end`. `LSUpdate` (`0x15ba8a0`) computes
///   `before = hour < time_start` and `past = hour >= time_end` and disables
///   the light when either is true, so outside that half-open window this
///   returns `None`. (The earlier reading — that the window only *blended*
///   colour while the light stayed on all day — was wrong; it is what made the
///   noon/midnight counts differ by one.)
/// * Inside the window the colour blends the base and bounce styles with
///   [`bounce_day_weight`] (`FUN_015bb580`, `0x15bb580`).
pub fn level_light_params(
    lights: &LevelLights,
    light: &LevelLight,
    hour: f32,
) -> Option<LevelLightParams> {
    let kind = if light.is_bounce() {
        LightKind::Bounce
    } else if light.has_projector() {
        LightKind::Projector
    } else if light.has_cubemap() {
        LightKind::Cubemap
    } else {
        LightKind::Point
    };

    let texture = lights.texture_path(light).map(|s| s.to_string());

    match lights.style(light)? {
        LightStyle::Info(info) => Some(LevelLightParams {
            position: light.position,
            color: info.color,
            intensity: info.range_scale * 0.5,
            radius: light.radius,
            kind,
            projector_texture: texture,
        }),
        LightStyle::Bounce(b) => {
            // `LSUpdate` disables the light outside `[time_start, time_end)`
            // (`0x15ba8a0`: `before = hour < start`, `past = hour >= end`).
            if hour < b.time_start || hour >= b.time_end {
                return None;
            }
            let (color, intensity) = bounce_style(light, b, hour);
            Some(LevelLightParams {
                position: light.position,
                color,
                intensity,
                radius: light.radius,
                kind,
                projector_texture: texture,
            })
        }
    }
}

/// Day weight for a bounce light inside its active window.
///
/// Two independent factors from `FUN_015bb580` (`0x15bb580`) multiply:
///
/// * `w` — a tent peaking at `time_mid`: `smoothstep` from `time_start` up to
///   `time_mid`, then `1 - smoothstep` down to `time_end` (the disassembly
///   builds `3t^2 - 2t^3` from `t^2`/`t^3` with `3.0` at `0x1720210` and `2.0`
///   at `0x172020c`). The earlier code had this inverted and treated the whole
///   window as a day->night ramp, which is why midnight still lit almost
///   everything.
/// * `s` — a sun-angle factor `(2 + 3*param_34)^((1 + cos(a))/2)` where
///   `a = 0.5 * (-2*pi*smoothstep(hour/24) + 0.4363 - 2*direction.z)` and the
///   constants are `-2*pi` (`0x186ece0`), `0.4363` (`0x1898098`) and `0.5`
///   (`0x17877c4`). `direction.z` is the record's third Euler angle, used here
///   as the light's sun angle.
///
/// The product is clamped to `[0,1]` so the base/night term `(1 - d)` can
/// never go negative (the engine writes the raw value into the light's HDR
/// multiplier at `+0x120`, which can overshoot).
fn bounce_day_weight(light: &LevelLight, b: &BounceLightInfo, hour: f32) -> f32 {
    let (start, mid, end) = (b.time_start, b.time_mid, b.time_end);
    if end <= start {
        return 0.0;
    }
    let w = if hour <= mid {
        smoothstep(0.0, 1.0, (hour - start) / (mid - start).max(f32::EPSILON))
    } else {
        1.0 - smoothstep(0.0, 1.0, (hour - mid) / (end - mid).max(f32::EPSILON))
    };
    let day = smoothstep(0.0, 1.0, (hour / 24.0).clamp(0.0, 1.0));
    let a = 0.5 * (-2.0 * std::f32::consts::PI * day + 0.4363 - 2.0 * light.direction[2]);
    let s = (2.0 + 3.0 * b.param_34.max(0.0)).powf((1.0 + a.cos()) * 0.5);
    (w * s).clamp(0.0, 1.0)
}

/// Blend a `BounceLightInfo` (17 floats) at `hour`.
///
/// `FUN_015bb580` writes the light colour at `+0xe0` as
/// `base_color * base_intensity * (1 - d) + bounce_color * bounce_intensity * d`
/// with `d` the day weight, and the range scale at `+0x15c` as
/// `range_scale * 0.5` (`0x15bb8e0`). The engine's separate HDR ratio written
/// to `+0x120` is not folded in here.
fn bounce_style(light: &LevelLight, b: &BounceLightInfo, hour: f32) -> ([f32; 3], f32) {
    let d = bounce_day_weight(light, b, hour);
    let n = 1.0 - d;
    let color = [
        b.bounce_color[0] * b.bounce_intensity * d + b.color[0] * b.base_intensity * n,
        b.bounce_color[1] * b.bounce_intensity * d + b.color[1] * b.base_intensity * n,
        b.bounce_color[2] * b.bounce_intensity * d + b.color[2] * b.base_intensity * n,
    ];
    (color, b.range_scale * 0.5)
}

// ---------------------------------------------------------------------------
// Shipped spec level: shadows and post-processing
// ---------------------------------------------------------------------------
//
// The game groups its render settings into `sys_spec_*` CVarGroups
// (`Engine/Config/CVarGroups/sys_spec_*.cfg`; the shipped paks place them at
// `config/config/cvargroups/`). `sys_spec_full.cfg` names group `3` as its
// `[default]`, i.e. the "High" preset, and the groups below are read from
// `sys_spec_shadows.cfg` / `sys_spec_postprocessing.cfg`. Each non-default
// group section lists only the values it overrides; anything absent inherits
// the `[default]` (= group 3) value, which is why every group here is built
// from the group-3 struct with field overrides.

/// The shipped CVarGroup the game starts at: `sys_spec_full.cfg` `[default] = 3`
/// (High). `SOD_SPEC=1..4` selects another group.
pub const SPEC_DEFAULT: u32 = 3;

/// Sun-shadow / cascaded-shadow-map settings, from
/// `sys_spec_shadows.cfg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowSettings {
    /// `e_GsmLodsNum` (0..5), the number of sun cascades.
    pub gsm_lods: u32,
    /// `e_GsmRange`: size of the LOD 0 GSM area (metres).
    pub gsm_range: f32,
    /// `e_GsmRangeStep`: each cascade's size is the previous times this.
    pub gsm_range_step: f32,
    /// `e_ShadowsMaxTexRes`: maximum shadow-map resolution per cascade
    /// (256/512/1024 in the cvar description).
    pub max_tex_res: u32,
    /// `e_ShadowsResScale`: engine-internal resolution scale. CryEngine shares
    /// one pool across cascades; a fixed per-cascade Bevy shadow map cannot
    /// express the fractional scale, so it is carried but not applied.
    pub res_scale: f32,
    /// `e_ShadowsConstBias`: constant depth bias for shadowgen.
    pub const_bias: f32,
    /// `e_ShadowsSlopeBias`: slope-scaled bias for shadowgen.
    pub slope_bias: f32,
    /// `r_ShadowBlur`: 0 none, 1..3 blur (3 = leak-free).
    pub blur: u32,
    /// `r_ShadowJittering`: shadow-map dither amount.
    pub jittering: f32,
    /// `r_ShadowsFadeDistMin` (metres).
    pub fade_min: f32,
    /// `r_ShadowsFadeDistMax` (metres).
    pub fade_max: f32,
    /// `e_ShadowsCastViewDistRatio`.
    pub cast_view_dist_ratio: f32,
    /// `e_ShadowsOnAlphaBlend`.
    pub on_alpha_blend: bool,
    /// `r_ShadowsPCFiltering`.
    pub pcf: bool,
}

impl ShadowSettings {
    /// Resolve the shadow CVarGroup for a shipped spec level.
    pub fn for_spec(spec: u32) -> Self {
        // Group 3 (empty `[3]` section) = all `[default]` values.
        let d = ShadowSettings {
            gsm_lods: 5,
            gsm_range: 2.0,
            gsm_range_step: 2.4,
            max_tex_res: 1024,
            res_scale: 3.4,
            const_bias: 0.1,
            slope_bias: 10.0,
            blur: 3,
            jittering: 1.5,
            fade_min: 50.0,
            fade_max: 150.0,
            cast_view_dist_ratio: 1.0,
            on_alpha_blend: true,
            pcf: true,
        };
        match spec {
            // [1]: 4 lods, 4 m LOD0, 2.25 step, 512 map, no blur, alpha-blend
            // shadows off, shorter fade.
            1 => ShadowSettings {
                gsm_lods: 4,
                gsm_range: 4.0,
                gsm_range_step: 2.25,
                max_tex_res: 512,
                blur: 0,
                jittering: 0.5,
                fade_min: 30.0,
                fade_max: 100.0,
                on_alpha_blend: false,
                ..d
            },
            // [2]: tighter LOD0 (1.7 m), 512 map, alpha-blend off.
            2 => ShadowSettings {
                gsm_range: 1.7,
                gsm_range_step: 2.4,
                max_tex_res: 512,
                on_alpha_blend: false,
                fade_min: 40.0,
                fade_max: 120.0,
                jittering: 1.0,
                ..d
            },
            // [4]: 6 lods, 2.3 m LOD0, 2.1 step, terrain casting, longer fade.
            4 => ShadowSettings {
                gsm_lods: 6,
                gsm_range: 2.3,
                gsm_range_step: 2.1,
                jittering: 3.0,
                fade_max: 200.0,
                ..d
            },
            _ => d,
        }
    }

    /// Far-plane distance of each cascade, in metres.
    ///
    /// `e_GsmRange` is the *size* of the LOD 0 area, so the first far bound is
    /// `gsm_range`; each following area is `gsm_range_step` times larger and
    /// the bounds accumulate. At High this gives
    /// `[2.0, 6.8, 18.32, 45.97, 112.32]` m, which lands just inside the
    /// `r_ShadowsFadeDistMax` of 150 m.
    pub fn cascade_bounds(&self) -> Vec<f32> {
        let n = self.gsm_lods.clamp(1, 8) as usize;
        let step = self.gsm_range_step.max(1.0);
        let mut size = self.gsm_range.max(0.1);
        let mut acc = 0.0;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            acc += size;
            out.push(acc);
            size *= step;
        }
        out
    }

    /// Bevy `DirectionalLight::shadow_depth_bias`.
    ///
    /// CryEngine's `e_ShadowsConstBias` is in a different unit; calibrated so
    /// the shipped High value (0.1) reproduces Bevy's tuned default (0.02).
    pub fn depth_bias(&self) -> f32 {
        self.const_bias * 0.2
    }

    /// Bevy `DirectionalLight::shadow_normal_bias`. Calibrated so the shipped
    /// High slope bias (10) maps onto Bevy's default (1.8).
    pub fn normal_bias(&self) -> f32 {
        self.slope_bias * 0.18
    }
}

/// Post-processing settings, from `sys_spec_postprocessing.cfg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostSettings {
    /// The spec group these came from.
    pub spec: u32,
    /// `r_HDRRendering`: 2 = exponential + s-curve tone map (`%_RT_SAMPLE2`).
    pub hdr_rendering: u32,
    /// `r_HDRLevel` (group value).
    pub hdr_level: f32,
    /// `r_HDROffset`.
    pub hdr_offset: f32,
    /// `r_EyeAdaptationBase` (group value).
    pub eye_adapt_base: f32,
    /// `r_EyeAdaptationFactor` (group value).
    pub eye_adapt_factor: f32,
    /// `r_HDRBrightLevel`.
    pub hdr_bright_level: f32,
    /// `r_HDRBrightOffset`.
    pub hdr_bright_offset: f32,
    /// `r_HDRBrightThreshold`.
    pub hdr_bright_threshold: f32,
    /// `r_HDRBloomMul`: bloom colour multiplier.
    pub hdr_bloom_mul: f32,
    /// `r_HDRBlueShift`.
    pub hdr_blue_shift: f32,
    /// `r_HDRGrainAmount`.
    pub hdr_grain: f32,
    /// `r_colorgrading`: 0 off, 2 on.
    pub color_grading: u32,
    /// `r_sunshafts`.
    pub sunshafts: bool,
    /// `r_ssao`: 0 off, 3/5 enabled.
    pub ssao: u32,
    /// `r_SSAOQuality`.
    pub ssao_quality: u32,
    /// `r_SSAODownscale`.
    pub ssao_downscale: u32,
    /// `r_SSAOObscuranceRadius`.
    pub ssao_radius: f32,
    /// `r_SSAOObscuranceOmega`.
    pub ssao_omega: f32,
    /// `r_DepthOfField`: 0 off, 3 on.
    pub dof: u32,
    /// `r_MotionBlur`.
    pub motion_blur: u32,
    /// `r_GlowAnamorphicFlares`.
    pub anamorphic_flares: bool,
    /// `e_SkyType`: 1 = HDR sky.
    pub sky_type: u32,
}

impl PostSettings {
    /// Resolve the post-processing CVarGroup for a shipped spec level.
    pub fn for_spec(spec: u32) -> Self {
        // Unknown specs fall back to the shipped default group (High).
        let spec = if matches!(spec, 1 | 2 | 3 | 4) {
            spec
        } else {
            SPEC_DEFAULT
        };
        // Invariants + group 3 `[default]` values.
        let d = PostSettings {
            spec,
            hdr_rendering: 2,
            hdr_level: 9.0,
            hdr_offset: 10.0,
            eye_adapt_base: 0.1,
            eye_adapt_factor: 0.9,
            hdr_bright_level: 0.85,
            hdr_bright_offset: 8.0,
            hdr_bright_threshold: 9.0,
            hdr_bloom_mul: 0.8,
            hdr_blue_shift: 0.5,
            hdr_grain: 0.0,
            color_grading: 2,
            sunshafts: true,
            ssao: 3,
            ssao_quality: 2,
            ssao_downscale: 0,
            ssao_radius: 0.33,
            ssao_omega: 0.25,
            dof: 3,
            motion_blur: 0,
            anamorphic_flares: true,
            sky_type: 1,
        };
        match spec {
            // [1]: no DOF, no flares, half-res SSAO (quality 0), slower eye
            // adaptation. `r_HDRLevel` is not listed here, so it inherits the
            // `[default]` value of 9 (the `r_HDRLevel=6` engine registration
            // default is only the fallback before CVarGroups load).
            1 => PostSettings {
                eye_adapt_base: 0.25,
                eye_adapt_factor: 0.85,
                dof: 0,
                anamorphic_flares: false,
                ssao_downscale: 1,
                ssao_quality: 0,
                ..d
            },
            // [2]: no DOF / flares, standard eye adaptation.
            2 => PostSettings {
                eye_adapt_base: 0.25,
                eye_adapt_factor: 0.85,
                dof: 0,
                anamorphic_flares: false,
                ..d
            },
            // [4]: stronger SSAO and motion blur on.
            4 => PostSettings {
                ssao: 5,
                motion_blur: 1,
                ..d
            },
            _ => d,
        }
    }

    /// Sun-shadow settings for the same spec level (convenience for the app).
    pub fn shadows(&self) -> ShadowSettings {
        ShadowSettings::for_spec(self.spec)
    }

    /// Bloom intensity for a Bevy `Bloom`, derived from the CryEngine bright
    /// pass.
    ///
    /// The original bright pass subtracts `r_HDRBrightThreshold`, maps the
    /// rest through `x / (offset + x)` and scales the additive composite by
    /// `HDRBrightLevel * HDRBloomMul` (`hdrpostprocess.cfx` `HDRBrightPassFilterPS`
    /// line 268 and the `cBloom *= HDRParams5 * 3/8` composite, line 922). The
    /// `3/8` is the engine's bloom tap weight; folding it in keeps the same
    /// order of magnitude for Bevy's additive composite.
    pub fn bloom_intensity(&self) -> f32 {
        (self.hdr_bright_level * self.hdr_bloom_mul * 0.375).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level_lights;
    use crate::tod::{parse_mission_lighting, TimeOfDay};

    fn env_default() -> Environment {
        Environment::default()
    }

    /// Build a synthetic `TodSample` by name/value.
    fn sample(pairs: &[(&str, crate::tod::Value)]) -> TodSample {
        TodSample {
            hour: 12.0,
            values: pairs.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
        }
    }

    #[test]
    fn frame_uses_sun_and_sky_multipliers() {
        use crate::tod::Value;
        let s = sample(&[
            ("Sun color", Value::Color([1.0, 0.5, 0.25])),
            ("Sun color multiplier", Value::Scalar(4.0)),
            ("Sky color", Value::Color([0.5, 0.5, 1.0])),
            ("Sky color multiplier", Value::Scalar(2.0)),
        ]);
        let f = LightingFrame::from_tod(&s, &env_default(), 12.0);
        assert_eq!(f.sun_radiance, [4.0, 2.0, 1.0]);
        assert_eq!(f.sky_ambient, [1.0, 1.0, 2.0]);
        // Sun at noon is above the horizon => light direction z negative.
        assert!(f.sun_dir[2] < 0.0);
    }

    #[test]
    fn exposure_is_positive_and_falls_with_brightness() {
        let e = ExposureParams::default();
        let dim = ExposureParams {
            adapted_luminance: e.adapted(0.2),
            ..e
        };
        let bright = ExposureParams {
            adapted_luminance: e.adapted(0.9),
            ..e
        };
        assert!(dim.exposure() > bright.exposure());
        assert!(bright.exposure() > 0.0);
    }

    #[test]
    fn s_curve_tonemap_drives_bright_warm_light_neutral() {
        // Raw sun hue at the shipped level start is strongly orange. The
        // per-channel exponential + s-curve path must lift it to near-white.
        let p = TonemapParams {
            exposure: ExposureParams {
                adapted_luminance: 0.4,
                ..ExposureParams::default()
            },
            ..TonemapParams::default()
        };
        let raw = tonemap([3.0, 1.0, 0.1], &p);
        let spread = raw[0] - raw[2];
        assert!(spread < 0.35, "spread {spread} (rgb {raw:?})");
        assert!(raw[0] > 0.5);
    }

    #[test]
    fn reinhard_preserves_hue_but_s_curve_does_not() {
        let p = TonemapParams {
            exposure: ExposureParams {
                adapted_luminance: 0.4,
                ..ExposureParams::default()
            },
            mode: TonemapMode::ReinhardLuminance,
            // Neutralise blue shift so only the operator's hue behaviour is
            // under test.
            blueshift_threshold: 1.0e9,
            ..TonemapParams::default()
        };
        // Small enough not to clamp, so a hue-preserving operator keeps the
        // R:G ratio of exactly 3.
        let c = tonemap([0.3, 0.1, 0.01], &p);
        assert!((c[0] / c[1] - 3.0).abs() < 0.05, "got {c:?}");
    }

    #[test]
    fn grade_identity_and_saturation() {
        let g = GradingParams::default();
        let c = grade([0.2, 0.4, 0.8], &g);
        assert!((c[0] - 0.2).abs() < 1e-4);
        assert!((c[2] - 0.8).abs() < 1e-4);

        let grey = GradingParams {
            saturation: 0.0,
            ..GradingParams::default()
        };
        let c = grade([0.2, 0.4, 0.8], &grey);
        assert!((c[0] - c[1]).abs() < 1e-4 && (c[1] - c[2]).abs() < 1e-4);
    }

    #[test]
    fn grade_levels_compress_range() {
        let g = GradingParams {
            levels_min_in: 0.0,
            levels_max_in: 0.5,
            levels_min_out: 0.1,
            levels_max_out: 0.9,
            ..GradingParams::default()
        };
        let c = grade([0.25, 0.5, 0.75], &g);
        assert!((c[0] - 0.5).abs() < 1e-3);
        assert!((c[1] - 0.9).abs() < 1e-3);
        assert!((c[2] - 0.9).abs() < 1e-3);
    }

    #[test]
    fn bounce_window_is_a_daytime_tent() {
        use crate::level_lights::{BounceLightInfo, LevelLight, FLAG_BOUNCE};
        let light = LevelLight {
            position: [0.0; 3],
            direction: [0.0; 3],
            radius: 1.0,
            texture_offset: 0xffff,
            geom_offset: 0xffff,
            info_index: 0,
            flags: FLAG_BOUNCE,
            clip_volume: [0, 0],
        };
        // Day colour is carried on `bounce_color` and `bounce_intensity`; the
        // base half is the night style (here blue, zero intensity).
        let b = BounceLightInfo {
            range_scale: 1.0,
            color: [0.0, 0.0, 1.0],
            base_intensity: 1.0,
            base_ratio: 1.0,
            specular: 0.0,
            bounce_color: [1.0, 1.0, 1.0],
            time_start: 6.0,
            time_mid: 12.0,
            time_end: 18.0,
            param_34: 0.0,
            bounce_intensity: 4.0,
            param_3c: 0.0,
            param_40: 0.0,
        };

        // Weight is zero at the window edges and peaks at the midpoint.
        assert_eq!(bounce_day_weight(&light, &b, 6.0), 0.0);
        assert_eq!(bounce_day_weight(&light, &b, 18.0), 0.0);
        assert!(bounce_day_weight(&light, &b, 12.0) > 0.9);
        assert!(bounce_day_weight(&light, &b, 9.0) > 0.0);
        assert!(bounce_day_weight(&light, &b, 9.0) < bounce_day_weight(&light, &b, 12.0));

        // Midday is the bright day colour; the window edges fall back to the
        // base (night) colour with no bounce contribution.
        let (c_day, i_day) = bounce_style(&light, &b, 12.0);
        assert!(c_day[0] > 1.0, "day colour {c_day:?}");
        assert!(i_day > 0.0);
        let (c_edge, _) = bounce_style(&light, &b, 6.0);
        assert_eq!(c_edge, [0.0, 0.0, 1.0]);
    }

    // -- real level ---------------------------------------------------------

    fn level_paths() -> Option<(Vec<u8>, Vec<u8>)> {
        let steam = "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";
        let xml = std::env::var("SOD_MISSION_XML").unwrap_or_else(|_| {
            "/home/battlestation/Desktop/sod-re/extracted-game/level/mission_mission0.xml"
                .to_string()
        });
        let lights = std::env::var("SOD_LIGHTS").unwrap_or_else(|_| {
            "/home/battlestation/Desktop/sod-re/extracted-game/level/lights_win.lights".to_string()
        });
        let xml = std::fs::read(&xml).ok()?;
        let lights = std::fs::read(&lights).ok()?;
        // The brief asks to skip when the game data is not installed.
        if !std::path::Path::new(steam).exists() {
            eprintln!("skipping real-level test: game install not present");
            return None;
        }
        Some((xml, lights))
    }

    #[test]
    fn real_level_noon_is_neutral_not_orange() {
        let Some((xml, _lights)) = level_paths() else {
            eprintln!("skipping integration test: level data absent");
            return;
        };
        let ml = parse_mission_lighting(&xml);
        let start = if ml.time_of_day.time > 0.0 {
            ml.time_of_day.time
        } else {
            12.0
        };
        let s = ml.time_of_day.sample(start);
        let frame = LightingFrame::from_tod(&s, &ml.environment, start);

        let mapped = tonemap(frame.sun_radiance, &frame.tonemap);
        let graded = grade(mapped, &frame.grading);
        let raw_spread = frame.sun_radiance[0] - frame.sun_radiance[2];
        let mapped_spread = mapped[0] - mapped[2];

        eprintln!(
            "hour {start:.3}: raw sun {:?} (spread {raw_spread:.3}), tone-mapped {:?} (spread {mapped_spread:.3}), graded {:?}, exposure {:.4}",
            frame.sun_radiance,
            mapped,
            graded,
            frame.exposure.exposure()
        );

        // Raw radiance is deliberately warm; after the original HDR tone map
        // the sun lands in the saturating part of the per-channel curve and
        // reads as near-neutral daylight, not saturated orange.
        assert!(raw_spread > 1.0, "raw sun should be warm");
        assert!(
            mapped_spread < 0.25,
            "tone-mapped sun should be near neutral, got {mapped:?}"
        );
        assert!(mapped.iter().all(|c| (0.0..=1.0).contains(c)));
    }

    #[test]
    fn real_level_noon_has_more_lights_than_midnight() {
        let Some((_xml, lights)) = level_paths() else {
            eprintln!("skipping integration test: level data absent");
            return;
        };
        let lv = level_lights::parse(&lights).expect("lights parse");

        let count = |hour: f32| {
            lv.lights
                .iter()
                .filter(|l| {
                    level_light_params(&lv, l, hour)
                        .map(|p| p.intensity > 0.0)
                        .unwrap_or(false)
                })
                .count()
        };
        let noon = count(12.0);
        let midnight = count(0.0);
        let bounce = lv.lights.iter().filter(|l| l.is_bounce()).count();
        eprintln!(
            "active lights: noon {noon}, midnight {midnight} ({bounce} bounce, {} always-on)",
            lv.lights.len() - bounce
        );
        // Every bounce window in the shipped level lies in daytime, so all of
        // them are off at midnight: the midnight count is exactly the
        // non-bounce lights.
        assert_eq!(midnight, lv.lights.len() - bounce);
        assert!(noon > midnight);
        assert!(noon <= lv.lights.len());

        // Also exercise the full frame build at both extremes.
        let env = env_default();
        let s = TimeOfDay::default().sample(12.0);
        let _ = LightingFrame::from_tod(&s, &env, 12.0);
    }

    #[test]
    fn default_frame_tod_has_sane_values() {
        let s = TimeOfDay::default().sample(12.0);
        let f = LightingFrame::from_tod(&s, &env_default(), 12.0);
        assert!(f.sun_radiance.iter().all(|c| c.is_finite()));
        assert!(f.exposure.exposure() > 0.0);
        assert!(f.sun_dir.iter().all(|c| c.is_finite()));
    }

    // -- shipped spec level --------------------------------------------------

    #[test]
    fn high_spec_csm_bounds_are_geometric_and_reach_the_fade_distance() {
        let s = ShadowSettings::for_spec(SPEC_DEFAULT);
        assert_eq!(s.gsm_lods, 5);
        let bounds = s.cascade_bounds();
        assert_eq!(bounds.len(), 5);
        // Strictly increasing.
        assert!(bounds.windows(2).all(|w| w[0] < w[1]));
        // First bound is the LOD 0 size (e_GsmRange).
        assert!((bounds[0] - 2.0).abs() < 1e-4);
        // Each size scales by e_GsmRangeStep: bound[1] - bound[0] = 2.0*2.4.
        assert!((bounds[1] - bounds[0] - 4.8).abs() < 1e-4);
        // Last cascade stays within r_ShadowsFadeDistMax (150 m at High).
        assert!(*bounds.last().unwrap() < s.fade_max);
    }

    #[test]
    fn shadow_group_overrides_follow_the_shipped_cfgs() {
        let low = ShadowSettings::for_spec(1);
        assert_eq!((low.gsm_lods, low.max_tex_res), (4, 512));
        assert_eq!(low.blur, 0);
        assert!(!low.on_alpha_blend);
        let ultra = ShadowSettings::for_spec(4);
        assert_eq!((ultra.gsm_lods, ultra.gsm_range), (6, 2.3));
        assert!((ultra.fade_max - 200.0).abs() < 1e-4);
        // Unknown spec falls back to the default (High) group.
        assert_eq!(ShadowSettings::for_spec(99), ShadowSettings::for_spec(3));
    }

    #[test]
    fn high_spec_bias_calibration_matches_bevy_defaults() {
        // The shipped High values (0.1 / 10) are calibrated onto Bevy's own
        // tuned defaults (0.02 / 1.8) so the mapping is neutral at High.
        let s = ShadowSettings::for_spec(3);
        assert!((s.depth_bias() - 0.02).abs() < 1e-5);
        assert!((s.normal_bias() - 1.8).abs() < 1e-4);
    }

    #[test]
    fn post_high_enables_the_shipped_chain() {
        let p = PostSettings::for_spec(SPEC_DEFAULT);
        assert_eq!(p.hdr_rendering, 2);
        assert!(p.sunshafts);
        assert!(p.ssao > 0);
        assert!(p.dof > 0);
        assert_eq!(p.color_grading, 2);
        assert!((p.eye_adapt_base - 0.1).abs() < 1e-4);
        assert!((p.eye_adapt_factor - 0.9).abs() < 1e-4);
        assert!((p.hdr_level - 9.0).abs() < 1e-4);
        assert!(p.bloom_intensity() > 0.0 && p.bloom_intensity() < 1.0);
    }

    #[test]
    fn post_low_disables_dof_and_flares() {
        let p = PostSettings::for_spec(1);
        assert_eq!(p.dof, 0);
        assert!(!p.anamorphic_flares);
        assert_eq!(p.ssao_downscale, 1);
        // Sunshafts are an invariant of the group: on at every spec.
        assert!(p.sunshafts);
        assert_eq!(PostSettings::for_spec(99), PostSettings::for_spec(3));
    }

    #[test]
    fn post_group_variants_only_override_listed_cvars() {
        // CVarGroup variants inherit every unlisted key from `[default]`
        // (= group 3). Groups 1/2 do not list `r_HDRLevel`, so they must keep
        // the shipped `[default]` value of 9 rather than the cvar-registration
        // fallback of 6.
        let high = PostSettings::for_spec(3);
        for spec in [1u32, 2] {
            let p = PostSettings::for_spec(spec);
            assert!((p.hdr_level - 9.0).abs() < 1e-4, "spec {spec}");
            assert!((p.hdr_level - high.hdr_level).abs() < 1e-4);
            assert_eq!(p.hdr_rendering, high.hdr_rendering);
            assert_eq!(p.color_grading, high.color_grading);
            assert_eq!(p.ssao, high.ssao);
        }
        // Group 4 lists `r_MotionBlur=1` and `r_ssao=5`; everything else stays.
        let ultra = PostSettings::for_spec(4);
        assert_eq!(ultra.motion_blur, 1);
        assert_eq!(ultra.ssao, 5);
        assert!((ultra.hdr_level - 9.0).abs() < 1e-4);
    }

    #[test]
    fn post_spec_selection_is_clamped_to_known_groups() {
        // The app exposes the spec level through `SOD_SPEC`; anything outside
        // the shipped 1..4 groups falls back to High.
        assert_eq!(PostSettings::for_spec(0).spec, SPEC_DEFAULT);
        assert_eq!(PostSettings::for_spec(7).spec, SPEC_DEFAULT);
        assert_eq!(ShadowSettings::for_spec(7), ShadowSettings::for_spec(SPEC_DEFAULT));
    }
}
