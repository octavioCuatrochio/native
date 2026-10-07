// CryEngine 3 post-processing bridge (workerR3).
//
// This is the custom fullscreen pass that Bevy has no stock equivalent for:
// the screen-space sun shafts (`r_sunshafts`) and the HDR blue shift / film
// grain from `hwscripts/cryfx/HDRPostProcess.cfx`. It runs on the HDR scene
// colour before Bevy's tonemap, so the values are linear and can exceed 1.
//
// The recovered CryEngine settings live in `crates/core/src/render_light.rs`
// (`PostSettings`) and are fed in through the uniform below; the mappings are
// documented there and in the module header of `post.rs`. No game source is
// copied here: this is a reimplementation from the CVar semantics.
//
// Bindings match `bevy_core_pipeline::fullscreen_material::FullscreenMaterial`
// (group 0: scene colour, filtering sampler, uniform). The vertex stage is
// Bevy's built-in `fullscreen_vertex_shader`.
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct CryPostUniform {
    // xy: sun screen uv (0..1, y down); z: 1 when the sun is in front of the
    // camera, 0 otherwise; w: sunshaft strength (`r_sunshafts`).
    sun: vec4<f32>,
    // x: `r_HDRBlueShift`; y: `r_HDRGrainAmount`; z: frame index for grain;
    // w: reserved.
    params: vec4<f32>,
    // rgb: sun colour used to tint the shafts; a: reserved.
    tint: vec4<f32>,
    // x: shafts sample count normaliser; y..w reserved.
    misc: vec4<f32>,
};

@group(0) @binding(0) var screen: texture_2d<f32>;
@group(0) @binding(1) var screen_sampler: sampler;
@group(0) @binding(2) var<uniform> post: CryPostUniform;

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Cheap value noise for the film grain (`r_HDRGrainAmount`); zero at High.
fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q = q + dot(q, q + 45.32);
    return fract(q.x * q.y);
}

// Screen-space sun shafts: march from the pixel toward the sun's projected
// position, accumulating scene luminance with exponential falloff. CryEngine
// occludes the shafts with the depth buffer (`r_sunshafts`); the scene colour
// already has the occluders darkened, so weighting each sample by its
// luminance gives the same bright-sky-only shafts without a depth read.
fn sun_shafts(uv: vec2<f32>, sun_uv: vec2<f32>, strength: f32) -> vec3<f32> {
    let delta = uv - sun_uv;
    let dist = length(delta);
    let dir = delta / max(dist, 1e-4);
    let samples = 28;
    var accum = vec3<f32>(0.0);
    var decay = 1.0;
    for (var i = 1; i <= samples; i = i + 1) {
        let t = f32(i) / f32(samples);
        let p = uv - dir * (dist * t);
        let s = textureSample(screen, screen_sampler, p).rgb;
        let l = dot(s, LUMA);
        accum = accum + s * decay * smoothstep(0.7, 2.6, l);
        decay = decay * 0.965;
    }
    // Fade the shafts out toward the screen edge and with distance from the
    // sun so only the region near the sun blooms.
    let falloff = exp(-dist * 1.8);
    return accum * (strength * falloff * (1.0 / f32(samples)) * 4.0);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    var c = textureSample(screen, screen_sampler, uv).rgb;
    // Debug: pass-through copy (SOD_PASSTHRU). Isolates the pass plumbing from
    // the effect maths.
    if (post.misc.x < 0.5) {
        return vec4<f32>(c, 1.0);
    }

    // --- r_sunshafts -------------------------------------------------------
    if (post.sun.z > 0.5 && post.sun.w > 0.0) {
        c = c + sun_shafts(uv, post.sun.xy, post.sun.w) * post.tint.rgb;
    }

    // --- HDR blue shift (`ApplyBlueShift`, HDRPostProcess.cfx) --------------
    // High luminances are lerped toward the blue corner; the amount is
    // `r_HDRBlueShift` (0.5 at High). Applied in HDR before the tonemap.
    let lum = dot(c, LUMA);
    let shift = saturate(lum - 1.0) * post.params.x;
    c = mix(c, vec3<f32>(lum * 0.55, lum * 0.78, lum * 1.25), shift * 0.5);

    // --- film grain (`r_HDRGrainAmount`) -----------------------------------
    if (post.params.y > 0.0) {
        let n = hash21(uv * 4096.0 + vec2<f32>(post.params.z, post.params.z * 0.7));
        c = c + (n - 0.5) * post.params.y;
    }

    return vec4<f32>(max(c, vec3<f32>(0.0)), 1.0);
}
