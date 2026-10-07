//! Character / zombie material: a Bevy [`ExtendedMaterial`] over
//! [`StandardMaterial`] whose fragment shader reproduces the CryEngine
//! `UL_Character` and `UL_Test_Skin` colour pipeline.
//!
//! Both shaders (`Engine/shaders.pak`,
//! `shaders/hwscripts/cryfx/ul_character.cfx` / `ul_test_skin.cfx`, including
//! `ul_partselection.cfi`) sample a near-neutral `Diffuse`, recolour it by the
//! four regions encoded in the `Custom` mask map (`*_m.dds`, blue channel),
//! blend the `Decal` map and blood (green channel), and apply a normal map. The
//! base `StandardMaterial` is still used for lighting so characters match the
//! level; this module only feeds it the authored albedo / normal / alpha.
//!
//! The WGSL lives in `char_material.wgsl` (see its header for the exact port).

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, ImageSampler};
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
};
use bevy::shader::{Shader, ShaderRef};
use sod_core::material::MaterialSlot;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Per-material uniform mirroring the `UL_Character` / `UL_Test_Skin`
/// `PublicParams` plus the `Material` diffuse/specular attributes.
#[derive(Clone, Default, ShaderType, Reflect, Debug)]
pub struct CharUniform {
    /// `MatDifColor` rgb.
    pub mat_dif: Vec4,
    /// `x` BloodBlendStart, `y` BloodBlendEnd, `z` AlphaRef, `w` DecalMaskId.
    pub blood: Vec4,
    /// `x` is_skin, `y` Melanin, `z` has_custom, `w` has_decal.
    pub flags: Vec4,
    /// region 0 `(Brightness, Contrast, Saturation, Hue)`.
    pub c0: Vec4,
    pub c1: Vec4,
    pub c2: Vec4,
    pub c3: Vec4,
    /// region 0 `(FresnelBias, FresnelScale, FresnelPower, Glossiness)`.
    pub s0: Vec4,
    pub s1: Vec4,
    pub s2: Vec4,
    pub s3: Vec4,
    /// region `SpecColor.rgb, Glossiness`.
    pub spec0: Vec4,
    pub spec1: Vec4,
    pub spec2: Vec4,
    pub spec3: Vec4,
    /// `Decal` `(TileU, TileV, OffsetU, OffsetV)`.
    pub decal: Vec4,
    /// `x` SpecularStrenght, `y` RimStrenght, `z` debug mode, `w` unused.
    pub misc: Vec4,
}

/// Extension bindings: the `Custom` mask map and the `Decal` overlay.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct CharMaterialExt {
    #[uniform(100)]
    pub params: CharUniform,
    #[texture(110)]
    #[sampler(111)]
    pub custom: Handle<Image>,
    #[texture(112)]
    #[sampler(113)]
    pub decal: Handle<Image>,
    /// Second view of the `Custom` map bound with a POINT sampler, for the
    /// region id (see `char_material.wgsl`).
    #[texture(114)]
    #[sampler(115)]
    pub custom_point: Handle<Image>,
}

/// Standard PBR lighting + the CryEngine character albedo pipeline.
pub type CharMaterial = ExtendedMaterial<StandardMaterial, CharMaterialExt>;

static CHAR_SHADER: OnceLock<Handle<Shader>> = OnceLock::new();

/// Debug visualisation mode from `SOD_CHAR_DEBUG` (0 = off). Used to prove
/// which texture / channel the shader actually reads (see `char_material.wgsl`).
fn debug_mode() -> f32 {
    static MODE: OnceLock<f32> = OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("SOD_CHAR_DEBUG")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(0.0)
    })
}

/// Per-material dump required to diagnose the character pipeline: the shader,
/// every resolved texture path, the decoded `Custom` channel averages, and all
/// `PublicParams`. Enabled by `SOD_CHAR_DUMP=1`.
fn dump_material<'a, F>(slot: &MaterialSlot, is_skin: bool, has_custom: bool, has_decal: bool, get_tex: F)
where
    F: Fn(&str) -> Option<&'a [u8]>,
{
    if std::env::var("SOD_CHAR_DUMP").map(|v| v != "1").unwrap_or(true) {
        return;
    }
    let c = &slot.char;
    println!(
        "[char] slot={:?} shader={:?} skin={} custom={} decal={} dif={:?} spec={:?} shin={}",
        slot.name, slot.shader, is_skin, has_custom, has_decal,
        slot.diffuse_color, slot.specular_color, c.glossiness[0]
    );
    for (m, p) in &slot.textures {
        println!("[char]   tex {m} -> {p}");
    }
    if let Some(path) = slot.texture("Custom") {
        if let Some(b) = get_tex(path) {
            if let Some(d) = sod_core::asset::decode_dds_rgba(b) {
                let n = (d.width as usize * d.height as usize).max(1);
                let mut sum = [0u64; 4];
                let mut mx = [0u8; 4];
                for px in d.rgba.chunks_exact(4) {
                    for k in 0..4 {
                        sum[k] += px[k] as u64;
                        mx[k] = mx[k].max(px[k]);
                    }
                }
                println!(
                    "[char]   Custom {}x{} avg R={:.1} G={:.1} B={:.1} A={:.1} max={:?}",
                    d.width, d.height,
                    sum[0] as f64 / n as f64, sum[1] as f64 / n as f64,
                    sum[2] as f64 / n as f64, sum[3] as f64 / n as f64, mx
                );
            }
        }
    }
    println!(
        "[char]   params bright={:?} contrast={:?} sat={:?} hue={:?} blood=({},{}) alphaRef={} decalId={} melanin={}",
        c.brightness, c.contrast, c.saturation, c.hue,
        c.blood_blend_start, c.blood_blend_end, c.alpha_ref, c.decal_mask_id, c.melanin
    );
    println!(
        "[char]   genmask={:#x} stringMask(decal={} pattern={})",
        c.gen_mask, c.blend_decal_map, c.blend_pattern_map
    );
}

/// The extension WGSL (loaded as one shader asset, like the terrain material).
pub const CHAR_MATERIAL_WGSL: &str = include_str!("char_material.wgsl");

/// Register the character shader before any [`CharMaterial`] is specialized.
pub fn install_char_shader(app: &mut App) {
    let shader = Shader::from_wgsl(CHAR_MATERIAL_WGSL, "sod_char_material.wgsl");
    let handle = app.world_mut().resource_mut::<Assets<Shader>>().add(shader);
    let _ = CHAR_SHADER.set(handle);
}

impl MaterialExtension for CharMaterialExt {
    fn fragment_shader() -> ShaderRef {
        match CHAR_SHADER.get() {
            Some(h) => ShaderRef::Handle(h.clone()),
            None => ShaderRef::Default,
        }
    }
}

/// Neutral 1x1 white fallback for a missing `Custom` / `Decal` map (memoised so
/// every material that lacks the map shares one GPU texture).
fn fallback_white(images: &mut Assets<Image>, memo: &mut HashMap<String, Handle<Image>>) -> Handle<Image> {
    if let Some(h) = memo.get("#char_white") {
        return h.clone();
    }
    let img = Image::new(
        Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        TextureDimension::D2,
        vec![255u8, 255, 255, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );
    let h = images.add(img);
    memo.insert("#char_white".to_string(), h.clone());
    h
}

/// Whether the CryEngine `_ddn` normal maps need their green channel flipped.
/// The engine bakes DirectX-convention normals (`+Y` down); Bevy's glTF/MikkTSpace
/// tangent frame is OpenGL-convention (`+Y` up). Overridable for A/B tests.
fn normal_flip_y() -> bool {
    std::env::var("SOD_CHAR_NOFLIPY").is_err()
}

/// Decode one character texture into a Bevy image with an explicit GPU format
/// and sampler.
///
/// * `srgb` — `Rgba8UnormSrgb` for the authored colour maps (`Diffuse`,
///   `Decal`); plain `Rgba8Unorm` for the linear `Custom` mask
///   (`sRGBLookup = false` in `UL_Character.cfx`).
/// * `two_component` — pack only `R,G` into `Rg8Unorm`. The `_ddn` maps ship as
///   `ATI2`/BC5 (two slopes); `decode_dds_rgba` leaves `B = 128`, which Bevy's
///   three-component path turns into `Z = 0.004` — a normal lying in the
///   surface, lit as the harsh black/grey streaks. `Rg8Unorm` sets Bevy's
///   `TWO_COMPONENT_NORMAL_MAP` flag so `Z = sqrt(1 - x² - y²)` is rebuilt.
/// * `sampler` — nearest for the `LayoutSamplerPoint` region-id lookup, linear
///   otherwise.
fn char_tex(
    images: &mut Assets<Image>,
    cache: Option<&sod_core::cache::TextureCache>,
    memo: &mut HashMap<String, Handle<Image>>,
    key: &str,
    bytes: &[u8],
    srgb: bool,
    two_component: bool,
    sampler: ImageSampler,
) -> Option<Handle<Image>> {
    if let Some(h) = memo.get(key) {
        return Some(h.clone());
    }
    let d = match cache {
        Some(c) => c.get_or_decode(key.as_bytes(), bytes)?,
        None => sod_core::asset::decode_dds_rgba(bytes)?,
    };
    let (format, data) = if two_component {
        let mut rg = Vec::with_capacity(d.width as usize * d.height as usize * 2);
        for px in d.rgba.chunks_exact(4) {
            rg.push(px[0]);
            rg.push(px[1]);
        }
        (TextureFormat::Rg8Unorm, rg)
    } else {
        let f = if srgb { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm };
        (f, d.rgba)
    };
    let mut img = Image::new(
        Extent3d { width: d.width.max(1), height: d.height.max(1), depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::default(),
    );
    img.sampler = sampler;
    let h = images.add(img);
    memo.insert(key.to_string(), h.clone());
    Some(h)
}

/// Build a [`CharMaterial`] for one palette slot, feeding every `PublicParams`
/// value and the `Diffuse` / `Custom` / `Bumpmap` / `Decal` textures.
///
/// `get_tex` resolves a pak texture path to its raw DDS bytes (the caller owns
/// the byte maps; the zombie templates keep them behind `Arc`).
pub fn build_char_material<'a, F>(
    materials: &mut Assets<CharMaterial>,
    images: &mut Assets<Image>,
    cache: Option<&sod_core::cache::TextureCache>,
    memo: &mut HashMap<String, Handle<Image>>,
    slot: &MaterialSlot,
    get_tex: F,
) -> Handle<CharMaterial>
where
    F: Fn(&str) -> Option<&'a [u8]>,
{
    let is_skin = slot.shader.eq_ignore_ascii_case("ul_test_skin");

    // Alpha mode mirrors the ZPass alpha cut:
    //   UL_Character  `step(AlphaRef, Custom.g)` vs `Material AlphaTest`
    //   UL_Test_Skin  `Diffuse.a` vs `Material AlphaTest`
    let alpha_mode = if is_skin {
        match slot.alpha_test {
            Some(a) if a > 0.0 => AlphaMode::Mask(a),
            _ => AlphaMode::Opaque,
        }
    } else if slot.char.alpha_ref > 0.0 {
        AlphaMode::Mask(slot.char.alpha_ref)
    } else {
        AlphaMode::Opaque
    };

    let mut base = StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.9,
        metallic: 0.0,
        cull_mode: None,
        alpha_mode,
        ..default()
    };
    if let Some(path) = slot.texture("Diffuse") {
        if let Some(bytes) = get_tex(path) {
            if let Some(h) = char_tex(images, cache, memo, path, bytes, true, false, ImageSampler::linear()) {
                base.base_color_texture = Some(h);
            }
        }
    }
    // Normal map: `_ddn` is ATI2/BC5. Load only R,G into `Rg8Unorm` so Bevy
    // reconstructs the tangent-space Z, and flip Y for the DirectX convention.
    if let Some(path) = slot.texture("Bumpmap").or_else(|| slot.texture("Normal")) {
        if let Some(bytes) = get_tex(path) {
            let key = format!("{path}#normal");
            if let Some(h) = char_tex(images, cache, memo, &key, bytes, false, true, ImageSampler::linear()) {
                base.normal_map_texture = Some(h);
                base.flip_normal_map_y = normal_flip_y();
            }
        }
    }

    let custom = slot
        .texture("Custom")
        .and_then(|p| get_tex(p).map(|b| (p.to_string(), b)))
        .and_then(|(p, b)| char_tex(images, cache, memo, &p, b, false, false, ImageSampler::linear()));
    let has_custom = custom.is_some();
    let custom = custom.unwrap_or_else(|| fallback_white(images, memo));
    // Second view of `Custom` with a nearest sampler: the region id
    // (`LayoutSamplerPoint`) must not blend across region boundaries.
    let custom_point = slot
        .texture("Custom")
        .and_then(|p| get_tex(p).map(|b| (p.to_string(), b)))
        .and_then(|(p, b)| {
            char_tex(images, cache, memo, &format!("{p}#point"), b, false, false, ImageSampler::nearest())
        })
        .unwrap_or_else(|| custom.clone());

    // The `Decal` overlay is only compiled into the shader when the material's
    // `StringGenMask` sets `%BLEND_DECAL_MAP` (`UL_Character.cfx`). The baseball
    // cap ships a `Decal` texture but no such flag, so it must be ignored.
    let decal = if slot.char.blend_decal_map {
        slot.texture("Decal")
            .and_then(|p| get_tex(p).map(|b| (p.to_string(), b)))
            .and_then(|(p, b)| char_tex(images, cache, memo, &p, b, true, false, ImageSampler::linear()))
    } else {
        None
    };
    let has_decal = decal.is_some();
    let decal = decal.unwrap_or_else(|| fallback_white(images, memo));

    dump_material(slot, is_skin, has_custom, has_decal, &get_tex);

    let params = uniform_for(slot, is_skin, has_custom, has_decal);

    materials.add(ExtendedMaterial {
        base,
        extension: CharMaterialExt { params, custom, decal, custom_point },
    })
}

/// Pack one slot's authored parameters into the shader uniform.
fn uniform_for(slot: &MaterialSlot, is_skin: bool, has_custom: bool, has_decal: bool) -> CharUniform {
    let c = &slot.char;
    let v4 = |v: [f32; 4]| Vec4::new(v[0], v[1], v[2], v[3]);
    let sc = |i: usize| Vec4::new(c.spec_color[i][0], c.spec_color[i][1], c.spec_color[i][2], c.glossiness[i]);
    let (tu, tv) = slot.texture_tile("Decal");
    let (ou, ov) = slot.texture_offset("Decal");
    CharUniform {
        mat_dif: Vec4::new(slot.diffuse_color[0], slot.diffuse_color[1], slot.diffuse_color[2], 1.0),
        blood: Vec4::new(c.blood_blend_start, c.blood_blend_end, c.alpha_ref, c.decal_mask_id),
        flags: Vec4::new(
            if is_skin { 1.0 } else { 0.0 },
            c.melanin,
            if has_custom { 1.0 } else { 0.0 },
            if has_decal { 1.0 } else { 0.0 },
        ),
        c0: v4([c.brightness[0], c.contrast[0], c.saturation[0], c.hue[0]]),
        c1: v4([c.brightness[1], c.contrast[1], c.saturation[1], c.hue[1]]),
        c2: v4([c.brightness[2], c.contrast[2], c.saturation[2], c.hue[2]]),
        c3: v4([c.brightness[3], c.contrast[3], c.saturation[3], c.hue[3]]),
        s0: Vec4::new(c.fresnel_bias[0], c.fresnel_scale[0], c.fresnel_power[0], c.glossiness[0]),
        s1: Vec4::new(c.fresnel_bias[1], c.fresnel_scale[1], c.fresnel_power[1], c.glossiness[1]),
        s2: Vec4::new(c.fresnel_bias[2], c.fresnel_scale[2], c.fresnel_power[2], c.glossiness[2]),
        s3: Vec4::new(c.fresnel_bias[3], c.fresnel_scale[3], c.fresnel_power[3], c.glossiness[3]),
        spec0: Vec4::new(
            slot.specular_color[0],
            slot.specular_color[1],
            slot.specular_color[2],
            c.glossiness[0],
        ),
        spec1: sc(1),
        spec2: sc(2),
        spec3: sc(3),
        decal: Vec4::new(tu, tv, ou, ov),
        misc: Vec4::new(c.specular_strength, c.rim_strength, debug_mode(), 0.0),
    }
}
