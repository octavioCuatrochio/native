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
}

impl MaterialSlot {
    pub fn texture(&self, map: &str) -> Option<&str> {
        self.textures
            .iter()
            .find(|(m, _)| m.eq_ignore_ascii_case(map))
            .map(|(_, p)| p.as_str())
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
}

impl MtlState {
    fn apply(&mut self, name: &str, attrs: &[(String, String)], opening: bool, vfs: &Vfs) {
        let get = |k: &str| attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        match name {
            "material" => {
                if opening {
                    self.depth += 1;
                }
                if self.depth == 1 {
                    self.root_slot = Some(MaterialSlot {
                        name: get("name").unwrap_or_default(),
                        shader: get("shader").unwrap_or_default(),
                        textures: Vec::new(),
                    });
                } else if self.in_sub && self.depth == 2 {
                    self.slots.push(MaterialSlot {
                        name: get("name").unwrap_or_default(),
                        shader: get("shader").unwrap_or_default(),
                        textures: Vec::new(),
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
                        textures: Vec::new(),
                    });
                    self.last_slot = Some(self.slots.len() - 1);
                }
            }
            "texture" => {
                if let (Some(map), Some(file)) = (get("map"), get("file")) {
                    if let Some(resolved) = resolve_tex(vfs, &file) {
                        if let Some(i) = self.last_slot {
                            self.slots[i].textures.push((map, resolved));
                        } else if let Some(rs) = self.root_slot.as_mut() {
                            rs.textures.push((map, resolved));
                        }
                    }
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
