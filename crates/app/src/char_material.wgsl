// Character material extension: reproduces the colour pipeline of the
// CryEngine character shaders shipped in Engine/shaders.pak:
//   shaders/hwscripts/cryfx/ul_character.cfx   (garments / heads / hats)
//   shaders/hwscripts/cryfx/ul_test_skin.cfx   (bare skin)
//   shaders/hwscripts/cryfx/ul_partselection.cfi (part mask, applied on the CPU)
//
// The base `StandardMaterial` supplies the sampled `Diffuse` (base_color_texture)
// and the `Bumpmap` normal map; this fragment shader replaces the albedo with
// the authored region recolour / decal / blood / melanin result and feeds it
// back into the standard PBR lighting so the level lighting stays consistent.
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing, alpha_discard},
    forward_io::{VertexOutput, FragmentOutput},
}

struct CharUniform {
    mat_dif: vec4<f32>,  // MatDifColor rgb
    blood: vec4<f32>,    // x BloodBlendStart, y BloodBlendEnd, z AlphaRef, w DecalMaskId
    flags: vec4<f32>,    // x is_skin, y melanin, z has_custom, w has_decal
    c0: vec4<f32>,       // region 0: Brightness, Contrast, Saturation, Hue
    c1: vec4<f32>,       // region 1
    c2: vec4<f32>,       // region 2
    c3: vec4<f32>,       // region 3
    s0: vec4<f32>,       // region 0: FresnelBias, FresnelScale, FresnelPower, Gloss
    s1: vec4<f32>,
    s2: vec4<f32>,
    s3: vec4<f32>,
    spec0: vec4<f32>,    // region 0 SpecColor rgb, glossiness
    spec1: vec4<f32>,
    spec2: vec4<f32>,
    spec3: vec4<f32>,
    decal: vec4<f32>,    // tile u, tile v, offset u, offset v
    misc: vec4<f32>,     // x SpecularStrenght, y RimStrenght
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> ch: CharUniform;

@group(#{MATERIAL_BIND_GROUP}) @binding(110)
var custom_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(111)
var custom_smp: sampler;

@group(#{MATERIAL_BIND_GROUP}) @binding(112)
var decal_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(113)
var decal_smp: sampler;

// `UL_Character.cfx` builds the region recolour as the matrix product
// brightness * contrast * saturation * hue and applies it to the diffuse in
// row-vector order (brightness then contrast then saturation then hue), then
// saturates. Each stage is expanded to the exact per-component arithmetic.
fn comp_color(rgb_in: vec3<f32>, p: vec4<f32>) -> vec3<f32> {
    let brightness = p.x;
    let contrast = p.y;
    let sat = p.z;
    let hue = p.w;

    // scaleMat(brightness)
    var rgb = rgb_in * brightness;

    // build_contrast_mat(1 - contrast) = translate(+0.5) * scale(s) * translate(-0.5)
    // applied to a row vector: ((rgb + 0.5) * s) - 0.5
    let s = 1.0 - contrast;
    rgb = (rgb + vec3<f32>(0.5)) * s - vec3<f32>(0.5);

    // build_saturation_mat(sat): luminance weights (0.3086, 0.6094, 0.0820)
    let rw = 0.3086;
    let gw = 0.6094;
    let bw = 0.0820;
    let i = 1.0 - sat;
    let sc0 = vec3<f32>(i * rw + sat, i * gw, i * bw);
    let sc1 = vec3<f32>(i * rw, i * gw + sat, i * bw);
    let sc2 = vec3<f32>(i * rw, i * gw, i * bw + sat);
    rgb = vec3<f32>(dot(rgb, sc0), dot(rgb, sc1), dot(rgb, sc2));

    // build_hue_mat(hue): rotate rgb about (1,1,1)
    if (hue != 0.0) {
        let d = normalize(vec3<f32>(1.0, 1.0, 1.0));
        let ca = cos(hue);
        let sa = sin(hue);
        let r0 = vec3<f32>(
            d.x * d.x * (1.0 - ca) + ca,
            d.x * d.y * (1.0 - ca) - d.z * sa,
            d.x * d.z * (1.0 - ca) + d.y * sa);
        let r1 = vec3<f32>(
            d.y * d.x * (1.0 - ca) + d.z * sa,
            d.y * d.y * (1.0 - ca) + ca,
            d.y * d.z * (1.0 - ca) - d.x * sa);
        let r2 = vec3<f32>(
            d.z * d.x * (1.0 - ca) - d.y * sa,
            d.z * d.y * (1.0 - ca) + d.x * sa,
            d.z * d.z * (1.0 - ca) + ca);
        // row-vector * matrix: component j is the dot with matrix column j
        let hc0 = vec3<f32>(r0.x, r1.x, r2.x);
        let hc1 = vec3<f32>(r0.y, r1.y, r2.y);
        let hc2 = vec3<f32>(r0.z, r1.z, r2.z);
        rgb = vec3<f32>(dot(rgb, hc0), dot(rgb, hc1), dot(rgb, hc2));
    }

    return saturate(rgb);
}

// Region glossiness (0..255) from the matching Glossiness1..3 / Shininess0.
fn region_gloss(region: f32) -> f32 {
    if (region > 2.5) { return ch.s3.w; }
    if (region > 1.5) { return ch.s2.w; }
    if (region > 0.5) { return ch.s1.w; }
    return ch.s0.w;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, is_front);

#ifdef VERTEX_UVS_A
    let uv = in.uv;
#else
    let uv = vec2<f32>(0.0);
#endif
    let texel = pbr.material.base_color;
    var rgb = texel.rgb;
    var alpha = texel.a;

    let has_custom = ch.flags.z > 0.5;
    let has_decal = ch.flags.w > 0.5;
    let custom = select(
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
        textureSample(custom_tex, custom_smp, uv),
        has_custom);
    // UL_Character: channelId in {0, 1/3, 2/3, 1} -> region round(channelId*3).
    let region = round(clamp(custom.b, 0.0, 1.0) * 3.0);
    var gloss = ch.s0.w;

    if (ch.flags.x > 0.5) {
        // ---- UL_Test_Skin ----
        // ApplyDecal multiplies the diffuse; then SkinSinthesys applies melanin.
        if (has_decal) {
            let duv = uv * ch.decal.xy + ch.decal.zw;
            let d = textureSample(decal_tex, decal_smp, duv);
            rgb = rgb * d.rgb;
        }
        let m = ch.flags.y;
        let lum = dot(rgb, vec3<f32>(0.33, 0.59, 0.11));
        rgb = rgb * mix(vec3<f32>(1.0), rgb * lum, vec3<f32>(m));
    } else {
        // ---- UL_Character ----
        if (has_custom) {
            var p = ch.c0;
            if (region > 2.5) { p = ch.c3; }
            else if (region > 1.5) { p = ch.c2; }
            else if (region > 0.5) { p = ch.c1; }
            rgb = comp_color(rgb, p);

            // ApplyDecal, only on the region named by DecalMaskIdChoice.
            if (has_decal && abs(region - ch.blood.w) < 0.5) {
                let duv = uv * ch.decal.xy + ch.decal.zw;
                let d = textureSample(decal_tex, decal_smp, duv);
                rgb = mix(rgb, d.rgb, d.a);
            }

            // Blood blend from the Custom map green (load alpha / blood).
            let ba = custom.g;
            if (ba > ch.blood.x) {
                let range = max(ch.blood.y - ch.blood.x, 0.0001);
                let amt = saturate((ba - ch.blood.x) / range);
                let blood_color = vec3<f32>(0.05, 0.002, 0.002);
                rgb = mix(rgb, blood_color, amt);
                rgb = rgb * (1.0 - amt);
            }
            gloss = region_gloss(region);
            // Alpha is the load-alpha / blood channel; AlphaRef cuts it.
            alpha = custom.g;
        }
    }

    rgb = rgb * ch.mat_dif.rgb;
    pbr.material.base_color = vec4<f32>(rgb, alpha);

    // Authored glossiness (0..255) -> perceptual roughness.
    let g = clamp(gloss, 1.0, 255.0);
    pbr.material.perceptual_roughness = clamp(1.0 - g / 255.0, 0.04, 1.0);
    pbr.material.metallic = 0.0;

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr);
    out.color = main_pass_post_lighting_processing(pbr, out.color);
    out.color = alpha_discard(pbr.material, out.color);
    return out;
}
