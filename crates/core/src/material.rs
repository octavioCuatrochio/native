//! CryEngine `.mtl` (XML) material parsing and texture-path resolution.
//!
//! Chain (validated in `notes/material-resolve-rust.md`):
//!   Mesh -> owning Node -> Node.materialChunkID -> root MtlName -> `<name>` ->
//!   `<dir(cgf)>/<name>.mtl` -> ordered `<SubMaterials>` palette ->
//!   MeshSubset.material indexes the palette -> slot `<Texture Map= File=>`.
//!
//! Texture `File=` paths are normalised (`\`->`/`), the source extension is
//! swapped to `.dds`, and looked up case-insensitively in the pak set.

use crate::crypak::Vfs;
use quick_xml::events::Event;
use quick_xml::Reader;

#[derive(Debug, Clone, Default)]
pub struct MaterialSlot {
    pub name: String,
    pub shader: String,
    /// (map, resolved pak path), e.g. ("Diffuse", "objects/.../camper.dds").
    pub textures: Vec<(String, String)>,
    /// `AlphaTest` threshold from the `.mtl` root/child (vegetation, decals).
    /// `None` = additive/opaque. The renderer's mask cutoff should use this
    /// instead of a fixed 0.5 so foliage cutout matches the authored value.
    pub alpha_test: Option<f32>,
    /// Per-texture `TexMod` tiling (map lowercased, tile U, tile V). Terrain
    /// layer materials author their ground tiling here.
    pub texture_tiles: Vec<(String, f32, f32)>,
    /// Character part-selection mask (`PublicParams` `VertColorMaskX/Y/Z/W`).
    ///
    /// CryEngine's character shaders (`UL_Character`, `UL_Test_Skin`) include
    /// `UL_PartSelection.cfi`: the per-vertex colour bytes are a 32-bit "which
    /// part am I" tag, and this 4-byte material mask selects the parts to drop
    /// (`clip(-Color.r)`). `[0,0,0,0]` = nothing selected, so no vertex is
    /// dropped by the mask. Non-zero for clothing/skin variants (e.g. the
    /// `med_sleeve_vNN` skins use `X=64` to hide the torso/upper-arm geometry
    /// that the outer garments already cover).
    pub vert_color_mask: [u8; 4],
}

impl MaterialSlot {
    pub fn texture(&self, map: &str) -> Option<&str> {
        self.textures
            .iter()
            .find(|(m, _)| m.eq_ignore_ascii_case(map))
            .map(|(_, p)| p.as_str())
    }

    /// `TexMod` tiling for `map`, defaulting to `(1.0, 1.0)` when unauthored.
    pub fn texture_tile(&self, map: &str) -> (f32, f32) {
        self.texture_tiles
            .iter()
            .find(|(m, _, _)| m.eq_ignore_ascii_case(map))
            .map(|(_, u, v)| (*u, *v))
            .unwrap_or((1.0, 1.0))
    }

    /// A `NoDraw` material (physics proxy / editor helper): never rendered.
    pub fn is_nodraw(&self) -> bool {
        self.shader.eq_ignore_ascii_case("nodraw")
    }

    /// Whether the part-selection mask drops the vertex carrying colour bytes
    /// `c` (BGRA/RGBA stream order, component 0 first).
    ///
    /// Mirrors `UL_PartSelection.cfi`: high bit of component 0 marks a "gore"
    /// item that is hidden unless the mask selects it; a normal item is hidden
    /// when the mask selects it. A zero mask never drops anything.
    pub fn hides_vertex(&self, c: [u8; 4]) -> bool {
        let m = self.vert_color_mask;
        if m == [0, 0, 0, 0] {
            return false;
        }
        let gore = c[0] >= 128;
        let cc = if gore {
            [c[0].wrapping_sub(128), c[1], c[2], c[3]]
        } else {
            c
        };
        let mut selected = false;
        for i in 0..4 {
            if m[i] & cc[i] != 0 {
                selected = true;
            }
        }
        if gore {
            !selected
        } else {
            selected
        }
    }
}

/// Resolve a CryEngine texture reference to an existing pak path.
fn resolve_tex(vfs: &Vfs, file: &str) -> Option<String> {
    let f = file.replace('\\', "/");
    let f = f.trim();
    if f.is_empty() {
        return None;
    }
    let lower = f.to_ascii_lowercase();
    let mut cands = Vec::new();
    if let Some(pos) = lower.rfind('.') {
        cands.push(format!("{}.dds", &lower[..pos])); // .tif/.tga/... -> .dds
        cands.push(lower.clone());
    } else {
        cands.push(format!("{lower}.dds"));
    }
    for c in cands {
        if vfs.exists(&c) {
            return Some(c);
        }
    }
    None
}

fn attr_map(e: &quick_xml::events::BytesStart) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for a in e.attributes().flatten() {
        let k = String::from_utf8_lossy(a.key.as_ref()).to_ascii_lowercase();
        let v = a
            .unescape_value()
            .map(|c| c.into_owned())
            .unwrap_or_default();
        out.push((k, v));
    }
    out
}

/// Parse `.mtl` bytes into the ordered material palette.
///
/// Palette rule (verified against `tools/mat_dump.py`, see
/// `notes/material-mapping.md` and `mesh.rs`): the root `<Material>` is a
/// container and is **not** a slot even when it carries `Name=`. The palette is
/// the root's direct `<SubMaterials>` children, in document order, where each
/// `<Material>` (named or not) **and** each `<MaterialRef/>` occupies exactly one
/// slot. When there is no `<SubMaterials>`, the palette is the single root
/// `<Material>` at slot 0. `MeshSubset.material` indexes this list directly.
pub fn parse_mtl_slots(bytes: &[u8], vfs: &Vfs) -> Vec<MaterialSlot> {
    let mut reader = Reader::from_reader(bytes);
    let mut buf = Vec::new();
    let mut st = MtlState::default();

    loop {
        match reader.read_event_into(&mut buf) {
            Err(_) | Ok(Event::Eof) => break,
            Ok(Event::Start(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_ascii_lowercase();
                let attrs = attr_map(&e);
                st.apply(&name, &attrs, true, vfs);
            }
            Ok(Event::Empty(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_ascii_lowercase();
                let attrs = attr_map(&e);
                st.apply(&name, &attrs, false, vfs);
            }
            Ok(Event::End(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_ascii_lowercase();
                st.end(&name);
            }
            _ => {}
        }
        buf.clear();
    }

    st.finish()
}

#[derive(Default)]
struct MtlState {
    depth: i32,
    in_sub: bool,
    root_slot: Option<MaterialSlot>,
    slots: Vec<MaterialSlot>,
    last_slot: Option<usize>,
    /// True after a `<Texture>` whose owning slot was the root container
    /// (no `<SubMaterials>`); lets a following `<TexMod>` update the right slot.
    last_tex_root: bool,
    /// Lowercased `Map` of the most recent `<Texture>`, if any.
    last_tex_map: Option<String>,
}

impl MtlState {
    fn apply(&mut self, name: &str, attrs: &[(String, String)], opening: bool, vfs: &Vfs) {
        let get = |k: &str| attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        let alpha_test = || get("alphatest").and_then(|v| v.trim().parse::<f32>().ok());
        match name {
            "material" => {
                // Both `<Material ...>` and self-closing `<Material .../>` sit one
                // level below the current open element, so classify them at
                // `depth + 1` regardless of `opening`.
                let d = self.depth + 1;
                if opening {
                    self.depth = d;
                }
                if d == 1 {
                    self.root_slot = Some(MaterialSlot {
                        name: get("name").unwrap_or_default(),
                        shader: get("shader").unwrap_or_default(),
                        alpha_test: alpha_test(),
                        ..Default::default()
                    });
                } else if self.in_sub && d == 2 {
                    self.slots.push(MaterialSlot {
                        name: get("name").unwrap_or_default(),
                        shader: get("shader").unwrap_or_default(),
                        alpha_test: alpha_test(),
                        ..Default::default()
                    });
                    self.last_slot = Some(self.slots.len() - 1);
                }
            }
            "submaterials" => {
                if self.depth == 1 {
                    self.in_sub = true;
                }
            }
            "materialref" => {
                if self.in_sub && self.depth == 1 {
                    self.slots.push(MaterialSlot {
                        name: get("name").unwrap_or_default(),
                        shader: String::new(),
                        ..Default::default()
                    });
                    self.last_slot = Some(self.slots.len() - 1);
                }
            }
            "texture" => {
                if let (Some(map), Some(file)) = (get("map"), get("file")) {
                    let map_l = map.to_ascii_lowercase();
                    if let Some(resolved) = resolve_tex(vfs, &file) {
                        if let Some(i) = self.last_slot {
                            self.slots[i].textures.push((map, resolved));
                            self.last_tex_root = false;
                        } else if let Some(rs) = self.root_slot.as_mut() {
                            rs.textures.push((map, resolved));
                            self.last_tex_root = true;
                        }
                    } else {
                        self.last_tex_root = self.last_slot.is_none();
                    }
                    // Tracked independently of resolution so a TexMod still
                    // applies when the source texture is missing from the VFS.
                    self.last_tex_map = Some(map_l);
                }
            }
            "publicparams" => {
                // Character part-selection mask. `PublicParams` sits inside the
                // `<Material>` it belongs to, so a child material's params target
                // `last_slot` while a root container's (no SubMaterials) target
                // the root slot.
                let set_mask = |s: &mut MaterialSlot| {
                    for (i, key) in ["vertcolormaskx", "vertcolormasky", "vertcolormaskz", "vertcolormaskw"]
                        .iter()
                        .enumerate()
                    {
                        if let Some(v) = get(key) {
                            if let Ok(b) = v.trim().parse::<u16>() {
                                s.vert_color_mask[i] = b.min(255) as u8;
                            }
                        }
                    }
                };
                if self.depth >= 2 {
                    if let Some(i) = self.last_slot {
                        if let Some(s) = self.slots.get_mut(i) {
                            set_mask(s);
                        }
                    }
                } else if let Some(rs) = self.root_slot.as_mut() {
                    set_mask(rs);
                }
            }
            "texmod" => {
                let Some(map) = self.last_tex_map.clone() else { return };
                let tu = get("tileu").and_then(|v| v.trim().parse::<f32>().ok());
                let tv = get("tilev").and_then(|v| v.trim().parse::<f32>().ok());
                let (u, v) = match (tu, tv) {
                    (Some(u), Some(v)) => (u, v),
                    (Some(u), None) => (u, u),
                    (None, Some(v)) => (v, v),
                    (None, None) => return,
                };
                let slot = if self.last_tex_root {
                    self.root_slot.as_mut()
                } else {
                    self.last_slot.and_then(|i| self.slots.get_mut(i))
                };
                if let Some(s) = slot {
                    s.texture_tiles.push((map, u, v));
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, name: &str) {
        if name == "material" {
            self.depth -= 1;
        } else if name == "submaterials" {
            self.in_sub = false;
        }
    }

    fn finish(mut self) -> Vec<MaterialSlot> {
        if self.slots.is_empty() {
            if let Some(rs) = self.root_slot.take() {
                self.slots.push(rs);
            }
        }
        self.slots
    }
}

/// Load and resolve the material palette for a model given its root material
/// name and the `.cgf`'s virtual path.
pub fn load_mtl(vfs: &Vfs, cgf_path: &str, name: &str) -> Vec<MaterialSlot> {
    if name.is_empty() {
        return Vec::new();
    }
    let dir = cgf_path.rfind('/').map(|p| &cgf_path[..p]).unwrap_or("");
    let mut candidates = Vec::new();
    if name.contains('/') {
        candidates.push(format!("{name}.mtl"));
    } else {
        candidates.push(format!("{dir}/{name}.mtl"));
        candidates.push(format!("{name}.mtl"));
    }
    for c in candidates {
        if let Some(bytes) = vfs.read(&c) {
            return parse_mtl_slots(&bytes, vfs);
        }
    }
    Vec::new()
}

/// Load a `.mtl` by its virtual path (extension optional), e.g. a terrain layer
/// material `terrain/terrainmaterials/cliff_rock.mtl`. Returns the parsed
/// palette, or empty if the file is absent.
pub fn load_mtl_path(vfs: &Vfs, mtl_path: &str) -> Vec<MaterialSlot> {
    let lower = mtl_path.replace('\\', "/").to_ascii_lowercase();
    let mut cands = Vec::new();
    if lower.ends_with(".mtl") {
        cands.push(lower.clone());
    } else {
        cands.push(format!("{lower}.mtl"));
        cands.push(lower.clone());
    }
    for c in cands {
        if let Some(bytes) = vfs.read(&c) {
            return parse_mtl_slots(&bytes, vfs);
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal stub VFS: `resolve_tex` only needs `exists`, so a fresh empty VFS
    /// still records the map names even though no texture resolves. To exercise
    /// resolution we hand-build the candidate paths that do exist.
    fn parse(bytes: &[u8]) -> Vec<MaterialSlot> {
        // Empty VFS: textures are dropped, but alpha/tile metadata is still kept.
        parse_mtl_slots(bytes, &Vfs::new())
    }

    #[test]
    fn captures_alpha_test_and_shader() {
        let slots = parse(
            br#"<Material MtlFlags="0">
                 <SubMaterials>
                   <Material Name="m_LOD" Shader="Vegetation" AlphaTest="0.81"/>
                   <Material Name="m_trunk" Shader="Illum"/>
                   <Material Name="proxy" Shader="NoDraw"/>
                 </SubMaterials>
               </Material>"#,
        );
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].alpha_test, Some(0.81));
        assert!(slots[1].alpha_test.is_none());
        assert!(slots[1].is_nodraw() == false);
        assert!(slots[2].is_nodraw());
    }

    #[test]
    fn captures_vert_color_part_mask() {
        let slots = parse(
            br#"<Material>
                 <SubMaterials>
                   <Material Name="skinShader" Shader="UL_Test_Skin">
                     <PublicParams VertColorMaskX="64" VertColorMaskY="0" VertColorMaskZ="0" VertColorMaskW="0"/>
                   </Material>
                   <Material Name="shirtShader" Shader="UL_Character"/>
                 </SubMaterials>
               </Material>"#,
        );
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].vert_color_mask, [64, 0, 0, 0]);
        assert_eq!(slots[1].vert_color_mask, [0, 0, 0, 0]);
        // Normal item carrying the selected bit is hidden; one without it kept.
        assert!(slots[0].hides_vertex([64, 0, 0, 255]));
        assert!(!slots[0].hides_vertex([0, 0, 0, 255]));
        // A zero mask never hides, even a high-bit (gore) tag.
        assert!(!slots[1].hides_vertex([64, 0, 0, 255]));
        assert!(!slots[1].hides_vertex([255, 0, 0, 255]));
    }

    #[test]
    fn gore_tag_hidden_unless_selected_by_mask() {
        let slots = parse(
            br#"<Material>
                 <SubMaterials>
                   <Material Name="s" Shader="UL_Test_Skin">
                     <PublicParams VertColorMaskX="64"/>
                   </Material>
                 </SubMaterials>
               </Material>"#,
        );
        let s = &slots[0];
        // R>=128 is a gore item: hidden unless the mask selects it.
        assert!(s.hides_vertex([128, 0, 0, 255]));
        assert!(s.hides_vertex([130, 0, 0, 255]));
        // (R-128)==64 selects it -> visible.
        assert!(!s.hides_vertex([192, 0, 0, 255]));
    }

    #[test]
    fn captures_tex_mod_tiles_for_terrain_layers() {
        // Tile attrs use `<TexMod TileU= TileV=>` under a `<Texture>`.
        let slots = parse(
            br#"<Material>
                 <SubMaterials>
                   <Material Name="x" Shader="Terrain.Layer">
                     <Textures>
                       <Texture Map="Diffuse" File="terrain/terrainmaterials/lawn_green_01_diffuse.tif">
                         <TexMod TileU="0.5" TileV="0.5"/>
                       </Texture>
                       <Texture Map="Bumpmap" File="terrain/terrainmaterials/lawn_green_01_normal.tif">
                         <TexMod TileU="0.25" TileV="0.75"/>
                       </Texture>
                     </Textures>
                   </Material>
                 </SubMaterials>
               </Material>"#,
        );
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].texture_tile("Diffuse"), (0.5, 0.5));
        assert_eq!(slots[0].texture_tile("Bumpmap"), (0.25, 0.75));
        // Default when unauthored.
        assert_eq!(slots[0].texture_tile("Specular"), (1.0, 1.0));
    }

    #[test]
    fn single_tex_mod_value_is_uniform() {
        let slots = parse(
            br#"<Material>
                 <Textures>
                   <Texture Map="Diffuse" File="a.tif"><TexMod TileU="0.2"/></Texture>
                 </Textures>
               </Material>"#,
        );
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].texture_tile("Diffuse"), (0.2, 0.2));
    }

    #[test]
    fn material_ref_still_occupies_a_slot() {
        let slots = parse(
            br#"<Material>
                 <SubMaterials>
                   <MaterialRef Name="ref"/>
                   <Material Name="b" Shader="Illum"/>
                 </SubMaterials>
               </Material>"#,
        );
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].name, "ref");
        assert_eq!(slots[1].name, "b");
    }
}
