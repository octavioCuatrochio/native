//! Terrain surface-type palette from the level's `leveldata.xml`.
//!
//! `<LevelData>/<SurfaceTypes>` lists one `<SurfaceType>` per terrain layer,
//! e.g.
//!
//! ```xml
//! <SurfaceType Name="Terrain/TerrainMaterials/lawn_green_01"
//!              SurfaceTypeID="4" DetailTexture=""
//!              DetailMaterial="Terrain/TerrainMaterials/lawn_green_01"
//!              ProjectAxis="2" Bumpmap="" ProjAxis="Z"/>
//! ```
//!
//! The ground mesh references these by the small integer `SurfaceTypeID`, so a
//! renderer needs the id→material table plus a colour to fall back on when the
//! real detail textures are unavailable.
//!
//! This is a **native fallback**: the shipping engine colours/textures terrain
//! from the `DetailMaterial` textures (and `DetailTexture` detail maps), not
//! from these RGBA values. [`fallback_color`] only exists so a Wine-free
//! renderer that cannot load those materials still gets a plausible ground
//! colour.
//!
//! Parsing is lenient (`quick-xml`, errors end the walk, never panics); missing
//! attributes default to empty and entries without a `SurfaceTypeID` are
//! skipped.

use crate::{asset, crypak, material, terrain};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::collections::HashMap;

/// One terrain surface layer from `<SurfaceTypes>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceType {
    /// `SurfaceTypeID` — the value the terrain data indexes by.
    pub id: u32,
    /// `Name` (e.g. `Terrain/TerrainMaterials/lawn_green_01`).
    pub name: String,
    /// `DetailMaterial` (falls back to `Name` when absent).
    pub material: String,
    /// `DetailTexture` detail map.
    pub detail: String,
}

/// Look up one attribute by exact case-sensitive name, unescaping entities.
fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

/// Build a [`SurfaceType`] from a `<SurfaceType>` element. Returns `None` when
/// the element has no usable `SurfaceTypeID`.
fn surface_from(e: &BytesStart) -> Option<SurfaceType> {
    let id = get_attr(e, "SurfaceTypeID")
        .and_then(|s| s.trim().parse::<u32>().ok())?;
    let name = get_attr(e, "Name").unwrap_or_default();
    let material = get_attr(e, "DetailMaterial")
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| name.clone());
    let detail = get_attr(e, "DetailTexture").unwrap_or_default();
    Some(SurfaceType {
        id,
        name,
        material,
        detail,
    })
}

/// Parse the `<SurfaceTypes>` palette out of raw `leveldata.xml` bytes.
///
/// Lenient: unknown elements/attributes are skipped, malformed XML ends the
/// walk with whatever was collected, and entries without a `SurfaceTypeID` are
/// dropped. Never panics.
pub fn parse_surface_types(xml: &[u8]) -> Vec<SurfaceType> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut stack: Vec<Vec<u8>> = Vec::new();
    let mut types: Vec<SurfaceType> = Vec::new();

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            // Lenient: malformed XML stops the walk, keeping what we have.
            Err(_) => break,
            Ok(ev) => ev,
        };

        let is_start = matches!(ev, Event::Start(_));

        match ev {
            Event::Start(e) | Event::Empty(e) => {
                let name = e.name().as_ref().to_vec();
                let parent_is_surface_types =
                    stack.last().map(|p| p.as_slice() == b"SurfaceTypes").unwrap_or(false);
                if name.as_slice() == b"SurfaceType" && parent_is_surface_types {
                    if let Some(st) = surface_from(&e) {
                        types.push(st);
                    }
                }
                if is_start {
                    stack.push(name);
                }
            }
            Event::End(_) => {
                stack.pop();
            }
            _ => {}
        }
        buf.clear();
    }

    types
}

/// The palette entry with `id`, if present.
pub fn lookup<'a>(types: &'a [SurfaceType], id: u8) -> Option<&'a SurfaceType> {
    types.iter().find(|t| t.id == id as u32)
}

/// Native fallback RGBA base colour for a surface-type/material `name`.
///
/// Case-insensitive substring matching against the material name, e.g.
/// `lawn_green_01` → green, `asphalt2b` → dark grey, `mud_cracks` → brown,
/// `cliff_rock` → grey-brown, `gravel_lite` → grey. Anything unrecognised gets a
/// neutral grey. This is only used when the real detail material textures are
/// unavailable (see the module docs); the engine itself textures terrain from
/// the material assets, not from these values.
pub fn fallback_color(name: &str) -> [f32; 4] {
    let n = name.to_ascii_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|k| n.contains(k));

    // Order matters: more specific terrain words are tested before broader
    // ones (e.g. "riverrock" is a rock, not open water).
    if has(&["gravel"]) {
        [0.52, 0.50, 0.47, 1.0] // grey gravel
    } else if has(&["grass", "lawn", "meadow", "foliage", "turf"]) {
        [0.30, 0.55, 0.22, 1.0] // green
    } else if has(&["asphalt", "road", "tarmac", "highway", "street"]) {
        [0.18, 0.18, 0.19, 1.0] // dark grey
    } else if has(&["mud", "dirt", "soil"]) {
        [0.45, 0.33, 0.20, 1.0] // brown
    } else if has(&["rock", "cliff", "stone", "boulder"]) {
        [0.48, 0.44, 0.38, 1.0] // grey-brown rock
    } else if has(&["sand", "beach", "desert"]) {
        [0.82, 0.74, 0.52, 1.0] // tan
    } else if has(&["concrete", "cement", "pavement", "sidewalk"]) {
        [0.60, 0.60, 0.58, 1.0] // concrete grey
    } else if has(&["water", "river", "lake", "stream", "ocean", "sea"]) {
        [0.16, 0.35, 0.60, 1.0] // blue
    } else {
        [0.50, 0.50, 0.50, 1.0] // neutral grey
    }
}

/// Fallback colour for surface `id`: the parsed [`SurfaceType`]'s name mapped
/// through [`fallback_color`], or neutral grey when the id is unknown.
pub fn color_for(types: &[SurfaceType], id: u8) -> [f32; 4] {
    match lookup(types, id) {
        Some(t) => fallback_color(&t.name),
        None => [0.50, 0.50, 0.50, 1.0],
    }
}

// ---------------------------------------------------------------------------
// Baked ground albedo
// ---------------------------------------------------------------------------
//
// The renderer used to tile one texture (`grass_dirt_mix01`) over the whole
// terrain and tint it with [`fallback_color`]. [`build_ground_albedo`] instead
// bakes a single CPU image that samples the *real* ground texture for each
// surface type, so grass/asphalt/gravel/rock/dirt read correctly. The per-type
// choice is a lenient name→file substring map ([`texture_hint`]); a missing
// texture falls back to that type's [`color_for`] colour.

/// World metres covered by one repeat of a ground texture.
const GROUND_UV_METERS: f32 = 4.0;
/// Hard cap on distinct ground textures decoded per bake (bounds memory). The
/// Class3 palette has 33 layers; this admits most of the distinct tiles while
/// keeping the decoded-texture memory bounded.
const MAX_GROUND_TEXTURES: usize = 24;
/// Hard cap on the baked albedo side (bounds the output allocation).
const MAX_ALBEDO_DIM: u32 = 4096;

const GROUND_GRASS: &str = "textures/ground/grass_dirt_mix01_diffuse.dds";
const GROUND_GRASS2: &str = "textures/ground/grass_dirt_mix02_diffuse.dds";
const GROUND_DIRT: &str = "textures/ground/dirt_dry01_diffuse.dds";
const GROUND_ASPHALT: &str = "textures/ground/asphalt01_diffuse.dds";
const GROUND_GRAVEL: &str = "textures/ground/gravel_light01_diffuse.dds";
const GROUND_ROCK: &str = "textures/ground/gravel_medium01_diffuse.dds";
const GROUND_CONCRETE: &str = "textures/ground/gravel_parking12x12_diffuse.dds";

/// Documented ground-texture repeat: one texture tile per this many world metres.
pub fn uv_scale() -> f32 {
    GROUND_UV_METERS
}

/// A surface type's real terrain-layer material, resolved from `leveldata.xml`'s
/// `DetailMaterial` -> `terrain/terrainmaterials/<name>.mtl`.
///
/// This is the shipping terrain pipeline: each `SurfaceTypeID` paints one
/// `Terrain.Layer` material whose `Diffuse`/`Bumpmap` textures tile over the
/// world at the authored `TexMod TileU/TileV` scale (repeats per metre), not a
/// single baked flat colour. [`resolve_surface_textures`] returns one row per
/// palette entry; the renderer can tile `diffuse` at `tile_u`/`tile_v` in the
/// shader, or feed the baked fallback ([`build_ground_albedo`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceTexture {
    pub id: u32,
    /// Resolved `..._diffuse.dds` pak path, if the material could be found.
    pub diffuse: Option<String>,
    pub normal: Option<String>,
    pub specular: Option<String>,
    /// `TexMod` repeats per world metre (default `1 / GROUND_UV_METERS`).
    pub tile_u: f32,
    pub tile_v: f32,
    pub alpha_test: Option<f32>,
}

/// Candidate `.mtl` paths for a `<SurfaceType>`'s detail material.
fn surface_mtl_candidates(st: &SurfaceType) -> Vec<String> {
    let mat = if st.material.is_empty() { st.name.as_str() } else { st.material.as_str() };
    let mat_base = mat.rsplit('/').next().unwrap_or(mat);
    let name_base = st.name.rsplit('/').next().unwrap_or(&st.name);
    let mut c = vec![
        format!("{mat}.mtl"),
        format!("terrain/terrainmaterials/{mat_base}.mtl"),
        format!("{}.mtl", st.name),
        format!("terrain/terrainmaterials/{name_base}.mtl"),
    ];
    c.dedup();
    c
}

/// Resolve the real terrain-layer material for one surface type, or `None`.
fn resolve_surface_material(vfs: &crypak::Vfs, st: &SurfaceType) -> Option<material::MaterialSlot> {
    for c in surface_mtl_candidates(st) {
        let slots = material::load_mtl_path(vfs, &c);
        if let Some(s) = slots.into_iter().find(|s| s.texture("Diffuse").is_some()) {
            return Some(s);
        }
    }
    None
}

/// Resolve every surface type to its authored terrain-layer textures + tiling.
///
/// Entries with no findable material fall back to [`texture_hint`] (generic
/// ground textures) and the default tiling, so the result always has one row per
/// input (same order). Never panics; a missing VFS entry simply yields `None`s.
pub fn resolve_surface_textures(
    vfs: &crypak::Vfs,
    types: &[SurfaceType],
) -> Vec<SurfaceTexture> {
    types
        .iter()
        .map(|st| match resolve_surface_material(vfs, st) {
            Some(s) => {
                let (tu, tv) = s.texture_tile("Diffuse");
                let (tu, tv) = if tu == 1.0 && tv == 1.0 { (1.0 / GROUND_UV_METERS, 1.0 / GROUND_UV_METERS) } else { (tu, tv) };
                SurfaceTexture {
                    id: st.id,
                    diffuse: s.texture("Diffuse").map(str::to_string),
                    normal: s
                        .texture("Bumpmap")
                        .or_else(|| s.texture("Normal"))
                        .map(str::to_string),
                    specular: s.texture("Specular").map(str::to_string),
                    tile_u: tu,
                    tile_v: tv,
                    alpha_test: s.alpha_test,
                }
            }
            None => {
                let name = if st.material.is_empty() { st.name.as_str() } else { st.material.as_str() };
                SurfaceTexture {
                    id: st.id,
                    diffuse: Some(texture_hint(name).to_string()),
                    normal: None,
                    specular: None,
                    tile_u: 1.0 / GROUND_UV_METERS,
                    tile_v: 1.0 / GROUND_UV_METERS,
                    alpha_test: None,
                }
            }
        })
        .collect()
}

/// Map a surface name or material path to the virtual path of its diffuse
/// ground texture.
///
/// Case-insensitive substring matching (mirrors [`fallback_color`]'s ordering,
/// so `riverrock` is a rock rather than water). Unrecognised names get the
/// default grass/dirt mix. The returned path is always a valid
/// `textures/ground/*_diffuse.dds` string; the file may or may not exist in the
/// mounted VFS.
pub fn texture_hint(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|k| n.contains(k));

    if has(&["gravel"]) {
        GROUND_GRAVEL
    } else if has(&["grass", "lawn", "meadow", "foliage", "turf"]) {
        GROUND_GRASS
    } else if has(&["asphalt", "road", "tarmac", "highway", "street"]) {
        GROUND_ASPHALT
    } else if has(&["mud", "dirt", "soil"]) {
        GROUND_DIRT
    } else if has(&["rock", "cliff", "stone", "boulder"]) {
        GROUND_ROCK
    } else if has(&["sand", "beach", "desert"]) {
        GROUND_DIRT
    } else if has(&["concrete", "cement", "pavement", "sidewalk"]) {
        GROUND_CONCRETE
    } else if has(&["water", "river", "lake", "stream", "ocean", "sea"]) {
        GROUND_GRASS2
    } else {
        GROUND_GRASS
    }
}

#[inline]
fn channel_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[inline]
fn rgba8(c: [f32; 4]) -> [u8; 4] {
    [channel_u8(c[0]), channel_u8(c[1]), channel_u8(c[2]), channel_u8(c[3])]
}

/// Tiled point sample (nearest texel, wrapping UVs) of a decoded image.
#[inline]
fn sample_rgba(img: &asset::DdsImageRgba, u: f32, v: f32) -> [u8; 4] {
    let w = img.width as usize;
    let h = img.height as usize;
    if w == 0 || h == 0 || img.rgba.len() < w.saturating_mul(h).saturating_mul(4) {
        return [128, 128, 128, 255];
    }
    let uu = u.rem_euclid(1.0);
    let vv = v.rem_euclid(1.0);
    let tx = ((uu * w as f32) as usize).min(w - 1);
    let ty = ((vv * h as f32) as usize).min(h - 1);
    let o = (ty * w + tx) * 4;
    let p = &img.rgba[o..o + 4];
    [p[0], p[1], p[2], p[3]]
}

/// Decode `path` on first use (memoised in `cache`), honouring
/// [`MAX_GROUND_TEXTURES`], then tiled-sample it at UV `(u, v)`.
fn sample_ground_texture(
    vfs: &crypak::Vfs,
    cache: &mut HashMap<String, Option<asset::DdsImageRgba>>,
    decoded: &mut usize,
    path: &str,
    u: f32,
    v: f32,
) -> Option<[u8; 4]> {
    let slot = cache.entry(path.to_string()).or_insert_with(|| {
        if *decoded >= MAX_GROUND_TEXTURES {
            return None;
        }
        *decoded += 1;
        vfs.read(path).and_then(|b| asset::decode_dds_rgba(&b))
    });
    Some(sample_rgba(slot.as_ref()?, u, v))
}

/// Bake a single RGBA8 ground-albedo texture for the whole terrain.
///
/// Each output texel maps to a world `(x, y)`, reads [`terrain::Terrain::surface_at`],
/// picks a **coloured** ground texture for that surface via [`texture_hint`],
/// decodes it (memoised, and capped at [`MAX_GROUND_TEXTURES`]) and samples it
/// at a fixed 4 m repeat. When a texture cannot be read/decoded the surface's
/// [`color_for`] colour is written instead.
///
/// This is a native fallback for the current single-texture renderer. The
/// shipping game instead blends the `Terrain.Layer` materials returned by
/// [`resolve_surface_textures`] at their authored `TexMod` tiling; a renderer
/// that tiles per-layer in world space should use those instead of this bake.
///
/// Returns `(out_size, out_size, rgba)` with `out_size²·4` bytes, or `None` for
/// an empty/degenerate terrain or an out-of-range `out_size`. Never panics.
pub fn build_ground_albedo(
    vfs: &crypak::Vfs,
    ter: &terrain::Terrain,
    types: &[SurfaceType],
    out_size: u32,
) -> Option<(u32, u32, Vec<u8>)> {
    if out_size == 0 || out_size > MAX_ALBEDO_DIM {
        return None;
    }
    if ter.samples == 0 || !(ter.spacing > 0.0) || ter.world_min[0].is_nan() || ter.world_min[1].is_nan()
    {
        return None;
    }
    let n = out_size as usize;
    let bytes = n.checked_mul(n)?.checked_mul(4)?;
    let mut out = vec![0u8; bytes];

    // Per-id lookup tables, computed once (no per-texel allocation).
    //
    // NOTE: the shipping `terrain/terrainmaterials/*_diffuse.dds` are *neutral
    // detail* maps (mean ~127 grey); the layer colour is applied by the
    // `Terrain.Layer` shader (material `Diffuse` colour + painted layer
    // coverage). Baking them directly as albedo produces a grey ground, so the
    // bake uses the coloured generic ground textures and the real per-layer
    // material metadata stays in [`resolve_surface_textures`] for a proper
    // tiled/blended terrain shader (see the report).
    let mut path_by_id: [Option<String>; 256] = std::array::from_fn(|_| None);
    let mut color_by_id: [[u8; 4]; 256] = [[128, 128, 128, 255]; 256];
    for id in 0..=255u8 {
        color_by_id[id as usize] = rgba8(color_for(types, id));
    }
    for t in types {
        if t.id <= 255 {
            let name = if t.material.is_empty() { &t.name } else { &t.material };
            path_by_id[t.id as usize] = Some(texture_hint(name).to_string());
        }
    }

    let span_x = ter.samples as f32 * ter.spacing;
    let span_y = ter.samples as f32 * ter.spacing;
    let (x0, y0) = (ter.world_min[0], ter.world_min[1]);
    let inv = 1.0 / out_size as f32;
    // The baked colormap repeats the detail texture at a fixed 4 m so adjacent
    // output texels (4 m apart at 1024) do not alias the tile away. The
    // authored per-layer `TexMod` scale is for the renderer's world-space tiling
    // (`SurfaceTexture::tile_u`), not for this full-world bake.
    let bake_tile = 1.0 / GROUND_UV_METERS;

    let mut cache: HashMap<String, Option<asset::DdsImageRgba>> = HashMap::new();
    let mut decoded = 0usize;

    for oy in 0..n {
        let wy = y0 + (oy as f32 + 0.5) * inv * span_y;
        for ox in 0..n {
            let wx = x0 + (ox as f32 + 0.5) * inv * span_x;
            let id = ter.surface_at(wx, wy) as usize;
            let px = path_by_id[id]
                .as_deref()
                .and_then(|path| {
                    sample_ground_texture(
                        vfs,
                        &mut cache,
                        &mut decoded,
                        path,
                        wx * bake_tile,
                        wy * bake_tile,
                    )
                })
                .unwrap_or(color_by_id[id]);
            let o = (oy * n + ox) * 4;
            out[o..o + 4].copy_from_slice(&px);
        }
    }
    Some((out_size, out_size, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<LevelData SandboxVersion="3.3.8.3040">
  <LevelInfo Name="Class3"/>
  <SurfaceTypes>
    <SurfaceType Name="Materials/material_terrain_default" SurfaceTypeID="0"
      DetailTexture="" DetailMaterial="Materials/material_terrain_default" ProjectAxis="2"/>
    <SurfaceType Name="Terrain/TerrainMaterials/cliff_rock" SurfaceTypeID="2"
      DetailTexture="detail_rock" DetailMaterial="Terrain/TerrainMaterials/cliff_rock"/>
    <SurfaceType Name="Terrain/TerrainMaterials/lawn_green_01" SurfaceTypeID="4"
      DetailMaterial="Terrain/TerrainMaterials/lawn_green_01"/>
    <SurfaceType Name="Terrain/TerrainMaterials/asphalt2b" SurfaceTypeID="6"/>
    <SurfaceType Name="Terrain/TerrainMaterials/mud_cracks" SurfaceTypeID="11"/>
    <SurfaceType Name="NoIdHere"/>
  </SurfaceTypes>
</LevelData>"#;

    #[test]
    fn parses_ids_names_materials_and_lookup() {
        let types = parse_surface_types(SAMPLE.as_bytes());
        // The entry without SurfaceTypeID is skipped.
        assert_eq!(types.len(), 5);
        assert_eq!(types[0].id, 0);
        assert_eq!(types[0].name, "Materials/material_terrain_default");
        assert_eq!(types[0].material, "Materials/material_terrain_default");
        assert_eq!(types[1].id, 2);
        assert_eq!(types[1].detail, "detail_rock");
        // Missing DetailMaterial falls back to the name.
        assert_eq!(types[2].material, "Terrain/TerrainMaterials/lawn_green_01");

        let st = lookup(&types, 2).expect("id 2 present");
        assert_eq!(st.name, "Terrain/TerrainMaterials/cliff_rock");
        assert!(lookup(&types, 99).is_none());
    }

    #[test]
    fn fallback_maps_representative_names() {
        let grass = fallback_color("Terrain/TerrainMaterials/lawn_green_01");
        let asphalt = fallback_color("game/terrain/terrainmaterials/asphalt01");
        let mud = fallback_color("Terrain/mud_cracks");
        let rock = fallback_color("Terrain/cliff_rock");
        let sand = fallback_color("beach_sand");
        let concrete = fallback_color("concrete_wall");
        let water = fallback_color("river");
        let gravel = fallback_color("gravel_lite");
        let unknown = fallback_color("mystery_layer");

        // Representative colours have a dominant channel in the expected hue.
        assert!(grass[1] > grass[0] && grass[1] > grass[2], "grass is green");
        assert!(asphalt[0] < 0.3 && asphalt[1] < 0.3, "asphalt is dark");
        assert!(mud[0] > mud[2], "mud is brown");
        assert!(rock[0] > rock[2], "rock is warm grey-brown");
        assert!(sand[0] > 0.7 && sand[2] < sand[0], "sand is tan");
        assert!(concrete[0] > 0.5 && concrete[1] > 0.5 && concrete[2] > 0.5);
        assert!(water[2] > water[0], "water is blue");
        assert!(gravel[0] > 0.4 && gravel[0] < 0.6);

        // All keys yield alpha 1.0 and the representative colours are distinct.
        for c in [grass, asphalt, mud, rock, sand, concrete, water, gravel, unknown] {
            assert_eq!(c[3], 1.0);
        }
        assert_ne!(grass, asphalt);
        assert_ne!(mud, rock);
        assert_ne!(sand, concrete);
        assert_ne!(water, gravel);
        assert_ne!(grass, unknown);
        assert_eq!(unknown, [0.5, 0.5, 0.5, 1.0]);
        // Case-insensitive.
        assert_eq!(fallback_color("LAWN"), fallback_color("lawn"));
    }

    #[test]
    fn lenient_and_never_panics() {
        assert!(parse_surface_types(b"").is_empty());
        assert!(parse_surface_types(b"<not xml").is_empty());
        assert!(parse_surface_types(b"<LevelData><SurfaceTypes/></LevelData>").is_empty());
        // Entries outside <SurfaceTypes> are ignored.
        let stray = parse_surface_types(br#"<X><SurfaceType Name="a" SurfaceTypeID="1"/></X>"#);
        assert!(stray.is_empty());
        // Bad id is skipped, good sibling still parsed.
        let mixed = parse_surface_types(
            br#"<LevelData><SurfaceTypes>
                 <SurfaceType Name="bad" SurfaceTypeID="nope"/>
                 <SurfaceType Name="ok" SurfaceTypeID="3"/>
               </SurfaceTypes></LevelData>"#,
        );
        assert_eq!(mixed.len(), 1);
        assert_eq!(mixed[0].id, 3);
    }

    #[test]
    fn color_for_uses_lookup_then_fallback() {
        let types = parse_surface_types(SAMPLE.as_bytes());
        assert_eq!(color_for(&types, 2), fallback_color("Terrain/TerrainMaterials/cliff_rock"));
        assert_eq!(color_for(&types, 200), [0.50, 0.50, 0.50, 1.0]);
    }

    #[test]
    fn real_leveldata_if_present() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../extracted-game/level/leveldata.xml"
        );
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return, // extraction absent: skip silently
        };
        let types = parse_surface_types(&bytes);
        assert!(types.len() > 5, "expected the real palette, got {}", types.len());
        let zero = lookup(&types, 0).expect("id 0 must resolve");
        assert!(!zero.name.is_empty());
    }

    // -----------------------------------------------------------------------
    // baked ground albedo tests
    // -----------------------------------------------------------------------

    /// A 4×4, 4 m-spacing terrain split half grass (id 4) / half asphalt (id 6),
    /// matching [`SAMPLE`]'s palette.
    fn tiny_terrain() -> crate::terrain::Terrain {
        let samples = 4;
        let cells = samples * samples;
        let mut surfaces = vec![4u8; cells];
        for (k, s) in surfaces.iter_mut().enumerate() {
            *s = if k < cells / 2 { 4 } else { 6 };
        }
        crate::terrain::Terrain {
            samples,
            spacing: 4.0,
            world_min: [0.0, 0.0, 0.0],
            heights: vec![0.0; cells],
            surfaces,
            min: 0.0,
            max: 0.0,
        }
    }

    #[test]
    fn texture_hint_maps_representatives() {
        // Exact expected virtual paths for the representative categories.
        assert_eq!(texture_hint("Terrain/TerrainMaterials/lawn_green_01"), super::GROUND_GRASS);
        assert_eq!(texture_hint("dirt_dry"), super::GROUND_DIRT);
        assert_eq!(texture_hint("asphalt2b"), super::GROUND_ASPHALT);
        assert_eq!(texture_hint("gravel_lite"), super::GROUND_GRAVEL);
        assert_eq!(texture_hint("cliff_rock"), super::GROUND_ROCK);
        assert_eq!(texture_hint("beach_sand"), super::GROUND_DIRT);
        assert_eq!(texture_hint("concrete_wall"), super::GROUND_CONCRETE);
        assert_eq!(texture_hint("river"), super::GROUND_GRASS2);
        assert_eq!(texture_hint("mystery_layer"), super::GROUND_GRASS);
        assert_eq!(texture_hint(""), super::GROUND_GRASS);

        // Every returned path is a non-empty ground diffuse path.
        for name in ["grass", "mud", "road", "gravel", "rock", "sand", "concrete", "lake", "???"] {
            let p = texture_hint(name);
            assert!(p.starts_with("textures/ground/"), "{name} -> {p}");
            assert!(p.ends_with("_diffuse.dds"), "{name} -> {p}");
        }
        // Case-insensitive.
        assert_eq!(texture_hint("LAWN_GREEN"), texture_hint("lawn_green"));
        // Distinct categories map to distinct files (except the documented
        // dirt/sand and water/grass2 aliases).
        assert_ne!(texture_hint("grass"), texture_hint("asphalt"));
        assert_ne!(texture_hint("gravel"), texture_hint("rock"));
        assert_ne!(texture_hint("rock"), texture_hint("concrete"));
        assert_eq!(uv_scale(), 4.0);
    }

    #[test]
    fn ground_albedo_bakes_varied_output() {
        let vfs = crate::crypak::Vfs::new();
        let types = parse_surface_types(SAMPLE.as_bytes());
        let ter = tiny_terrain();
        let (w, h, rgba) = build_ground_albedo(&vfs, &ter, &types, 8).expect("albedo");
        assert_eq!((w, h), (8, 8));
        assert_eq!(rgba.len(), 8 * 8 * 4);
        // No real textures in this VFS, so each surface uses its fallback colour;
        // the two surfaces must differ, giving a varied bake.
        let mut uniq = std::collections::HashSet::new();
        for p in rgba.chunks(4) {
            uniq.insert([p[0], p[1], p[2], p[3]]);
        }
        assert!(uniq.len() >= 2, "expected varied colours, got {}", uniq.len());
    }

    #[test]
    fn ground_albedo_is_total() {
        let vfs = crate::crypak::Vfs::new();
        let types = parse_surface_types(SAMPLE.as_bytes());
        let mut ter = tiny_terrain();

        assert!(build_ground_albedo(&vfs, &ter, &types, 0).is_none());
        // Refused without allocating the huge buffer.
        assert!(build_ground_albedo(&vfs, &ter, &types, 100_000).is_none());

        // Degenerate terrain.
        ter.samples = 0;
        assert!(build_ground_albedo(&vfs, &ter, &types, 8).is_none());
        ter.samples = 4;
        ter.spacing = 0.0;
        assert!(build_ground_albedo(&vfs, &ter, &types, 8).is_none());

        // Empty palette still produces a valid (default/grey) bake, no panic.
        let good = tiny_terrain();
        let (w, h, rgba) = build_ground_albedo(&vfs, &good, &[], 4).expect("empty palette ok");
        assert_eq!((w, h), (4, 4));
        assert_eq!(rgba.len(), 4 * 4 * 4);
    }

    #[test]
    fn resolves_real_terrain_layer_textures_if_present() {
        let level = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../extracted-game/level/leveldata.xml");
        let terrain_root =
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../extracted-game/terrain"));
        if !terrain_root.is_dir() {
            return; // extraction absent: skip silently
        }
        let Ok(bytes) = std::fs::read(level) else { return };
        let types = parse_surface_types(&bytes);
        assert!(!types.is_empty());

        let mut vfs = crate::crypak::Vfs::new();
        vfs.set_loose_root(terrain_root);
        let stex = resolve_surface_textures(&vfs, &types);
        assert_eq!(stex.len(), types.len());

        // Most authored layers must resolve to a real terrainmaterials diffuse.
        let with_tex = stex.iter().filter(|s| s.diffuse.is_some()).count();
        assert!(
            with_tex * 2 >= types.len(),
            "only {with_tex}/{} surfaces resolved a texture",
            types.len()
        );

        // id 4 == lawn_green_01: diffuse + authored 0.5 repeat/m.
        let id4 = stex.iter().find(|s| s.id == 4).expect("id 4");
        let d = id4.diffuse.as_deref().unwrap_or("");
        assert!(d.contains("lawn_green_01"), "id4 diffuse {d}");
        assert!((id4.tile_u - 0.5).abs() < 1e-3, "id4 tile_u {}", id4.tile_u);
        assert!(id4.normal.is_some(), "id4 should carry a Bumpmap");
    }

    #[test]
    fn resolve_surface_textures_falls_back_for_unknown() {
        let types = parse_surface_types(SAMPLE.as_bytes());
        let vfs = crate::crypak::Vfs::new();
        let stex = resolve_surface_textures(&vfs, &types);
        assert_eq!(stex.len(), types.len());
        // With no VFS every entry still gets a generic fallback diffuse.
        for s in &stex {
            assert!(s.diffuse.is_some(), "id {} should fall back", s.id);
            assert!(s.diffuse.as_deref().unwrap().starts_with("textures/ground/"));
        }
    }

    #[test]
    fn ground_albedo_decodes_real_textures_if_present() {
        // Loose files are nested one level deeper than the pak virtual path.
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../extracted-game/textures");
        let root = std::path::Path::new(root);
        if !root.is_dir() {
            return; // extraction absent: skip silently
        }
        let mut vfs = crate::crypak::Vfs::new();
        vfs.set_loose_root(root);
        let types = parse_surface_types(SAMPLE.as_bytes());
        let ter = tiny_terrain();
        let (w, h, rgba) = build_ground_albedo(&vfs, &ter, &types, 16).expect("real albedo");
        assert_eq!((w, h), (16, 16));
        assert_eq!(rgba.len(), 16 * 16 * 4);
        // At least one real texture decoded (the terrain is grass+asphalt).
        let mut uniq = std::collections::HashSet::new();
        for p in rgba.chunks(4) {
            uniq.insert([p[0], p[1], p[2], p[3]]);
        }
        assert!(uniq.len() > 1, "expected real-texture variation, got {}", uniq.len());
    }
}
