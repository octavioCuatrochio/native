//! Decode CryEngine chunk-file geometry (`.cgf`/`.cga`/`.chr`) into triangle
//! meshes suitable for a renderer.
//!
//! Layouts per `notes/cgf-format.md`, `notes/cgf-mesh-decode.md` and
//! `notes/material-resolve-rust.md` (validated across the shipped corpus):
//!   * Mesh payload stream refs at +0x1C (16 u32); slot index == DataStream type.
//!   * DataStream: flags,type,numElements,bytesPerElement,res1,res2,body.
//!   * Positions (type 0): 8 B = 4x half-float (x,y,z,1.0) or 12 B = 3x f32.
//!   * Normals (1) 12 B f32x3; Uvs (2) 8 B f32x2; Indices (5) 2 B u16.
//!   * MeshSubsets (0x17): flags,num,res,res, then N x 36 B subsets
//!     (firstIndex,numIndices,firstVertex,numVertices,material,radius,center).
//!   * Node (0x0B): objectChunkID +0x40, parentChunkID +0x44, materialChunkID +0x4C.
//!
//! # Subset material indexing (exact rule)
//!
//! `Submesh.material` is the raw `MeshSubset.material` u32 (the u32 at
//! MeshSubsets-payload `+0x10 + 36*i + 0x10`, i.e. subset field `+0x10`) kept
//! verbatim; it is a direct 0-based index into the model's `.mtl` **ordered
//! SubMaterials palette** with no remapping. The palette order is the only
//! thing that matters:
//!
//!   1. If the root `<Material>` has a `<SubMaterials>` child, the palette is
//!      the document-order list of that element's children, where **every**
//!      `<Material Name=…>` and **every** `<MaterialRef/>` occupies exactly one
//!      slot (`MaterialRef` keeps its index but has no textures). The root
//!      `<Material>` itself is *not* a slot here, even when it carries a
//!      `Name=` attribute.
//!   2. If there is no `<SubMaterials>`, the palette is the single root
//!      `<Material>` at slot 0.
//!
//! The index is never an index into the cgf's own `MtlName` children. Ground
//! truth for the corpus is `tools/mat_dump.py`; see `notes/material-mapping.md`
//! and `notes/material-resolve-rust.md`.
//!
//! Caveat (palette construction, `material::parse_mtl_slots`): as of this
//! writing that parser emits a slot for a *named* root `<Material>` (rule 1)
//! and skips `<MaterialRef/>` (rule 1's empty slot). Both shift the palette and
//! therefore make some surfaces resolve to the wrong slot: `wrench.cgf` (named
//! root `wrench`) is off by +1, and `barrel_toxic.cgf` (a `MaterialRef` at
//! index 0) is off by -1. The mesh side below is exact; fixing the index space
//! means fixing palette construction to the rules above.

use crate::asset::{parse_cry_chunk_file, CryChunkFile};
use std::collections::HashMap;

const NS_GEOM: u16 = 0xCCCC;

#[derive(Debug, Clone, Default)]
pub struct Submesh {
    pub material: u32,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct DecodedMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    /// Per-material index ranges (material = palette slot).
    pub submeshes: Vec<Submesh>,
}

#[derive(Debug, Clone, Default)]
pub struct DecodedModel {
    pub meshes: Vec<DecodedMesh>,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// Root material name (-> `<name>.mtl`), if resolved from a Node/MtlName.
    pub material_name: Option<String>,
}

impl DecodedModel {
    pub fn vertex_count(&self) -> usize {
        self.meshes.iter().map(|m| m.positions.len()).sum()
    }
    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.indices.len() / 3).sum()
    }
}

/// Every `Submesh.material` in draw order (meshes in decode order, submeshes in
/// `MeshSubsets` order). These are direct 0-based indices into the model's
/// ordered `.mtl` `<SubMaterials>` palette; see the module docs for the exact
/// palette rule. Useful for diagnostics/validation against a resolved palette.
pub fn subset_material_indices(model: &DecodedModel) -> Vec<u32> {
    model
        .meshes
        .iter()
        .flat_map(|m| m.submeshes.iter().map(|s| s.material))
        .collect()
}

fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
fn rd_f32(b: &[u8], o: usize) -> Option<f32> {
    rd_u32(b, o).map(f32::from_bits)
}

/// IEEE-754 binary16 -> f32.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    let val = match exp {
        0 => mant * 2f32.powi(-24),
        0x1f => {
            if mant == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

#[derive(Clone, Copy)]
struct Mat4([f32; 16]);

impl Mat4 {
    fn identity() -> Self {
        Mat4([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])
    }
    /// Row-vector convention (CryEngine): result = self * rhs.
    fn mul(&self, rhs: &Mat4) -> Mat4 {
        let mut r = [0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += self.0[i * 4 + k] * rhs.0[k * 4 + j];
                }
                r[i * 4 + j] = s;
            }
        }
        Mat4(r)
    }
    fn xform_point(&self, p: [f32; 3]) -> [f32; 3] {
        [
            p[0] * self.0[0] + p[1] * self.0[4] + p[2] * self.0[8] + self.0[12],
            p[0] * self.0[1] + p[1] * self.0[5] + p[2] * self.0[9] + self.0[13],
            p[0] * self.0[2] + p[1] * self.0[6] + p[2] * self.0[10] + self.0[14],
        ]
    }
}

/// Generate smooth per-vertex normals when a mesh has none/mismatched ones.
pub fn generate_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut n = vec![[0f32; 3]; positions.len()];
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let nn = cross(sub(positions[i1], positions[i0]), sub(positions[i2], positions[i0]));
        for &i in tri {
            let i = i as usize;
            n[i][0] += nn[0];
            n[i][1] += nn[1];
            n[i][2] += nn[2];
        }
    }
    for v in n.iter_mut() {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 1e-12 {
            v[0] /= len;
            v[1] /= len;
            v[2] /= len;
        } else {
            *v = [0.0, 1.0, 0.0];
        }
    }
    n
}

struct DataStream<'a> {
    ty: u32,
    num: usize,
    bpe: usize,
    body: &'a [u8],
}

fn read_stream<'a>(
    cf: &CryChunkFile,
    file: &'a [u8],
    by_id: &HashMap<u32, usize>,
    id: u32,
) -> Option<DataStream<'a>> {
    if id == 0 {
        return None;
    }
    let idx = *by_id.get(&id)?;
    let rec = &cf.chunks[idx];
    let p = rec.payload(file)?;
    Some(DataStream {
        ty: rd_u32(p, 4)?,
        num: rd_u32(p, 8)? as usize,
        bpe: rd_u32(p, 12)? as usize,
        body: p.get(24..)?,
    })
}

fn decode_positions(s: &DataStream) -> Option<Vec<[f32; 3]>> {
    if s.ty != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.num);
    match s.bpe {
        12 => {
            for i in 0..s.num {
                let o = i * 12;
                out.push([rd_f32(s.body, o)?, rd_f32(s.body, o + 4)?, rd_f32(s.body, o + 8)?]);
            }
        }
        8 => {
            for i in 0..s.num {
                let o = i * 8;
                let x = u16::from_le_bytes([s.body[o], s.body[o + 1]]);
                let y = u16::from_le_bytes([s.body[o + 2], s.body[o + 3]]);
                let z = u16::from_le_bytes([s.body[o + 4], s.body[o + 5]]);
                out.push([half_to_f32(x), half_to_f32(y), half_to_f32(z)]);
            }
        }
        _ => return None,
    }
    Some(out)
}

fn decode_normals(s: &DataStream) -> Option<Vec<[f32; 3]>> {
    if s.ty != 1 || s.bpe < 12 {
        return None;
    }
    let mut out = Vec::with_capacity(s.num);
    for i in 0..s.num {
        let o = i * s.bpe;
        out.push([rd_f32(s.body, o)?, rd_f32(s.body, o + 4)?, rd_f32(s.body, o + 8)?]);
    }
    Some(out)
}

fn decode_uvs(s: &DataStream) -> Option<Vec<[f32; 2]>> {
    if s.ty != 2 || s.bpe < 8 {
        return None;
    }
    let mut out = Vec::with_capacity(s.num);
    for i in 0..s.num {
        let o = i * s.bpe;
        out.push([rd_f32(s.body, o)?, rd_f32(s.body, o + 4)?]);
    }
    Some(out)
}

fn decode_indices(s: &DataStream) -> Option<Vec<u32>> {
    if s.ty != 5 || s.bpe != 2 {
        return None;
    }
    let mut out = Vec::with_capacity(s.num);
    for i in 0..s.num {
        let o = i * 2;
        out.push(u16::from_le_bytes([*s.body.get(o)?, *s.body.get(o + 1)?]) as u32);
    }
    Some(out)
}

struct NodeInfo {
    object_chunk: u32,
    parent: i32,
    material_chunk: u32,
    local: Mat4,
    /// Node name (payload +0x00, 64-byte NUL-padded). Used to identify physics
    /// proxies / in-file LOD helpers that must not be drawn.
    name: String,
}

/// Node-name substrings that mark a **non-renderable** node: the physics hit
/// proxy (`*_hitref`, `$physics_proxy*`) or an in-file LOD helper
/// (`*__lod1_geometry`, `*$lod1*`). The engine writes a separate low-poly mesh
/// node for collision and, in a few files, extra LOD nodes next to the LOD0
/// geometry; both carry their own `Mesh` chunk. Decoding them together with the
/// render mesh draws the collision shell over the real model (and overlaps LODs),
/// which is exactly the "low-poly" look.
pub fn node_is_render_geometry(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    if n.is_empty() {
        return true;
    }
    if n.contains("hitref") || n.contains("physics_proxy") {
        return false;
    }
    if n.starts_with("$physics") || n.starts_with("$lod") {
        return false;
    }
    if n.contains("__lod") || n.contains("_lod1") || n.contains("_lod2") || n.contains("_lod3") {
        return false;
    }
    true
}

fn parse_nodes(cf: &CryChunkFile, file: &[u8]) -> Vec<NodeInfo> {
    let mut nodes = Vec::new();
    for ch in &cf.chunks {
        if ch.namespace() != NS_GEOM || ch.kind() != 0x0B {
            continue;
        }
        let Some(p) = ch.payload(file) else { continue };
        let object_chunk = rd_u32(p, 0x40).unwrap_or(0);
        let parent = rd_u32(p, 0x44).map(|v| v as i32).unwrap_or(-1);
        let material_chunk = rd_u32(p, 0x4C).unwrap_or(0);
        let raw_name = p.get(0..64).unwrap_or(&[]);
        let end = raw_name.iter().position(|&b| b == 0).unwrap_or(raw_name.len());
        let name = String::from_utf8_lossy(&raw_name[..end]).trim().to_string();
        let mut m = [0f32; 16];
        let mut ok = true;
        for (i, slot) in m.iter_mut().enumerate() {
            match rd_f32(p, 0x54 + i * 4) {
                Some(v) => *slot = v,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        nodes.push(NodeInfo {
            object_chunk,
            parent,
            material_chunk,
            local: if ok { Mat4(m) } else { Mat4::identity() },
            name,
        });
    }
    nodes
}

/// Parse a MeshSubsets chunk into (material, first_index, num_indices) tuples.
fn parse_subsets(cf: &CryChunkFile, file: &[u8], by_id: &HashMap<u32, usize>, id: u32) -> Vec<(u32, usize, usize)> {
    if id == 0 {
        return Vec::new();
    }
    let Some(&idx) = by_id.get(&id) else { return Vec::new() };
    let Some(p) = cf.chunks[idx].payload(file) else { return Vec::new() };
    let n = rd_u32(p, 4).unwrap_or(0) as usize;
    let mut out = Vec::new();
    for i in 0..n {
        let o = 0x10 + i * 36;
        let (Some(first_index), Some(num_indices), Some(material)) =
            (rd_u32(p, o), rd_u32(p, o + 4), rd_u32(p, o + 16))
        else {
            break;
        };
        out.push((material, first_index as usize, num_indices as usize));
    }
    out
}

/// Read the fixed 128-byte NUL-terminated name from an MtlName payload.
fn mtlname_name(cf: &CryChunkFile, file: &[u8], by_id: &HashMap<u32, usize>, id: u32) -> Option<String> {
    let &idx = by_id.get(&id)?;
    let ch = &cf.chunks[idx];
    if ch.namespace() != NS_GEOM || ch.kind() != 0x14 {
        return None;
    }
    let p = ch.payload(file)?;
    let raw = p.get(0x08..0x08 + 128)?;
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    let s = String::from_utf8_lossy(&raw[..end]).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Decode all meshes of a chunk file into world-space triangles.
pub fn decode_model(file: &[u8]) -> Option<DecodedModel> {
    decode_model_full(file, true, false)
}

/// Like [`decode_model`], but optionally skips Node transform composition
/// (needed for skinned meshes, whose vertices must stay in bind model space).
pub fn decode_model_with(file: &[u8], apply_node_transforms: bool) -> Option<DecodedModel> {
    decode_model_full(file, apply_node_transforms, false)
}

/// Decode only the **renderable LOD0** geometry: like [`decode_model`], but
/// skips the physics/collision proxy mesh node (`*_hitref`, `$physics_proxy*`)
/// and any in-file LOD helper node (`*__lod1_geometry`, `$lod1*`) that a few
/// models carry next to the LOD0 geometry. See [`node_is_render_geometry`].
///
/// Use this for drawing; use [`decode_model`] when the caller wants every mesh
/// (e.g. to include collision proxies in a physics build).
pub fn decode_model_render(file: &[u8]) -> Option<DecodedModel> {
    decode_model_full(file, true, true)
}

pub fn decode_model_full(
    file: &[u8],
    apply_node_transforms: bool,
    render_only: bool,
) -> Option<DecodedModel> {
    let cf = parse_cry_chunk_file(file)?;
    let mut by_id: HashMap<u32, usize> = HashMap::new();
    for (i, c) in cf.chunks.iter().enumerate() {
        by_id.insert(c.id, i);
    }
    let nodes = parse_nodes(&cf, file);
    // Map node chunk id -> index within `nodes` (same order parse_nodes used).
    let mut node_index_by_id: HashMap<u32, usize> = HashMap::new();
    {
        let mut idx = 0usize;
        for c in &cf.chunks {
            if c.namespace() == NS_GEOM && c.kind() == 0x0B {
                node_index_by_id.insert(c.id, idx);
                idx += 1;
            }
        }
    }
    let mut node_of_object: HashMap<u32, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        node_of_object.entry(n.object_chunk).or_insert(i);
    }

    let mut model = DecodedModel::default();
    let mut bmin = [f32::INFINITY; 3];
    let mut bmax = [f32::NEG_INFINITY; 3];

    for ch in &cf.chunks {
        if ch.namespace() != NS_GEOM || ch.kind() != 0x00 {
            continue;
        }
        let Some(mp) = ch.payload(file) else { continue };

        // Render path: drop physics-proxy / in-file LOD nodes. A mesh with no
        // owning node is kept (nothing to classify it by).
        let owning_node = node_of_object.get(&ch.id).copied();
        if render_only {
            if let Some(ni) = owning_node {
                if !node_is_render_geometry(&nodes[ni].name) {
                    continue;
                }
            }
        }

        let ref_at = |slot: usize| rd_u32(mp, 0x1C + slot * 4).unwrap_or(0);

        let Some(pos_stream) = read_stream(&cf, file, &by_id, ref_at(0)) else {
            continue;
        };
        let Some(mut positions) = decode_positions(&pos_stream) else {
            continue;
        };
        let normals = read_stream(&cf, file, &by_id, ref_at(1)).and_then(|s| decode_normals(&s));
        let uvs = read_stream(&cf, file, &by_id, ref_at(2)).and_then(|s| decode_uvs(&s));
        let Some(mut indices) =
            read_stream(&cf, file, &by_id, ref_at(5)).and_then(|s| decode_indices(&s))
        else {
            continue;
        };

        // Apply the owning node's world transform (row-vector convention).
        if let Some(ni) = owning_node {
            if !apply_node_transforms {
                // skinned path: keep bind model space
            } else {
            let mut world = nodes[ni].local;
            let mut parent = nodes[ni].parent;
            let mut guard = 0;
            while parent >= 0 && guard < 256 {
                match node_index_by_id.get(&(parent as u32)) {
                    Some(&pi) => {
                        world = world.mul(&nodes[pi].local);
                        parent = nodes[pi].parent;
                    }
                    None => break,
                }
                guard += 1;
            }
            for p in positions.iter_mut() {
                *p = world.xform_point(*p);
            }
            }
        }

        indices.retain(|&i| (i as usize) < positions.len());
        for p in &positions {
            for k in 0..3 {
                bmin[k] = bmin[k].min(p[k]);
                bmax[k] = bmax[k].max(p[k]);
            }
        }

        // Subsets -> per-material index groups.
        let subsets = parse_subsets(&cf, file, &by_id, rd_u32(mp, 0x14).unwrap_or(0));
        let submeshes = if subsets.is_empty() {
            vec![Submesh { material: 0, indices: indices.clone() }]
        } else {
            subsets
                .iter()
                .map(|&(material, first, count)| {
                    let end = (first + count).min(indices.len());
                    Submesh { material, indices: indices[first.min(end)..end].to_vec() }
                })
                .collect()
        };

        // Root material name (Node.materialChunkID -> MtlName.name).
        if model.material_name.is_none() {
            if let Some(ni) = owning_node {
                model.material_name = mtlname_name(&cf, file, &by_id, nodes[ni].material_chunk);
            }
        }

        let normals = match normals {
            Some(n) if n.len() == positions.len() => n,
            _ => generate_normals(&positions, &indices),
        };

        model.meshes.push(DecodedMesh {
            positions,
            normals,
            uvs: uvs.unwrap_or_default(),
            indices,
            submeshes,
        });
    }

    if model.meshes.is_empty() {
        return None;
    }
    if bmin[0].is_finite() {
        model.bounds_min = bmin;
        model.bounds_max = bmax;
    }
    Some(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh_with(materials: &[u32]) -> DecodedMesh {
        DecodedMesh {
            submeshes: materials
                .iter()
                .map(|&material| Submesh { material, indices: vec![0, 1, 2] })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn subset_material_indices_keep_parsed_values_in_draw_order() {
        let model = DecodedModel {
            meshes: vec![mesh_with(&[3, 1, 6, 0])],
            ..Default::default()
        };
        assert_eq!(subset_material_indices(&model), vec![3, 1, 6, 0]);
    }

    #[test]
    fn subset_material_indices_span_meshes_in_decode_order() {        let model = DecodedModel {
            meshes: vec![mesh_with(&[2, 0]), mesh_with(&[1])],
            ..Default::default()
        };
        assert_eq!(subset_material_indices(&model), vec![2, 0, 1]);
    }

    /// Ground-truth mapping for real multi-material models: raw
    /// `MeshSubset.material` values, in draw order, from `tools/mat_dump.py`
    /// (`notes/material-mapping.md`). The indices address the documented
    /// `<SubMaterials>` palette, *not* the cgf's own `MtlName` list:
    ///   * wrench (named root `wrench`): 0          -> palette[0] "Pipes"
    ///   * skull_full:                  3          -> palette[3] "skull"
    ///   * camper:                     0,2,1       -> m_camper, m_glass, m_camper_b
    ///   * barrel_toxic (`MaterialRef`@0): 2,1,3   -> phong1, lambert4, m_collision_metal
    #[test]
    fn real_models_subset_material_mapping() {
        let cases: &[(&str, &[u32])] = &[
            ("objects/objects/weapons/wrench/wrench.cgf", &[0]),
            ("particles/particles/geometry/skull_full/skull_full.cgf", &[3]),
            ("objects/objects/decor/vehicles/camper/camper.cgf", &[0, 2, 1]),
            (
                "BeltwayC/Game/objects/class3_2/objects/barrel_toxic/barrel_toxic.cgf",
                &[2, 1, 3],
            ),
        ];
        // The repo's extracted-asset tree; tests skip silently when absent.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game");
        if !root.is_dir() {
            eprintln!("skip: {} not present", root.display());
            return;
        }
        for &(rel, expected) in cases {
            let path = root.join(rel);
            let Ok(bytes) = std::fs::read(&path) else {
                eprintln!("skip: {} missing", path.display());
                continue;
            };
            let model = decode_model(&bytes)
                .unwrap_or_else(|| panic!("decode failed: {}", path.display()));
            let got = subset_material_indices(&model);
            assert_eq!(got.as_slice(), expected, "{}", path.display());
        }
    }

    #[test]
    fn render_geometry_classifier() {
        // Physics hit proxies / in-file LOD helpers are not drawable.
        assert!(!node_is_render_geometry("ranger_station_01_hitref"));
        assert!(!node_is_render_geometry("$physics_proxy_hitref"));
        assert!(!node_is_render_geometry("$physics_proxy_$lod1_short03_hitref"));
        assert!(!node_is_render_geometry("Physics_Proxy"));
        assert!(!node_is_render_geometry("desert_brush__lod1_geometry"));
        assert!(!node_is_render_geometry("group__lod1_blaines_grocery"));
        // A render node whose *name* merely mentions collision is still drawn.
        assert!(node_is_render_geometry("boxwood_clump_no_collision"));
        assert!(node_is_render_geometry("geometry"));
        assert!(node_is_render_geometry("ranger_station_01"));
        assert!(node_is_render_geometry("")); // unclassified -> keep
    }

    /// A real building whose cgf carries both a `*_hitref` collision mesh and the
    /// LOD0 render mesh: the render decoder must drop the proxy, while the full
    /// decoder keeps it for physics callers.
    #[test]
    fn render_skips_hitref_proxy_on_real_building() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game");
        let path = root.join(
            "objects/objects/architecture/buildings/civic/ranger_station_01/ranger_station_01.cgf",
        );
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skip: {} missing", path.display());
            return;
        };
        let full = decode_model(&bytes).expect("full decode");
        let render = decode_model_render(&bytes).expect("render decode");

        // Dropping at least one whole mesh (the hitref proxy).
        assert!(
            render.meshes.len() < full.meshes.len(),
            "render {} meshes, full {}",
            render.meshes.len(),
            full.meshes.len()
        );
        // The biggest LOD0 mesh survives; total triangles drop but stay large.
        assert!(full.triangle_count() > render.triangle_count());
        assert!(render.triangle_count() > 5_000, "lod0 mesh must survive");
        // Every render submesh's triangle count is real geometry, not the
        // low-poly proxy (the proxy here is a few hundred triangles).
        assert!(render.triangle_count() >= full.triangle_count() * 8 / 10);
    }

    /// A proxy-only cgf decodes to `None` on the render path (nothing drawable).
    #[test]
    fn render_only_proxy_file_decodes_to_none() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game");
        let path =
            root.join("objects/objects/architecture/buildings/agricultural/windmill_01/windmill_01_proxy.cgf");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skip: {} missing", path.display());
            return;
        };
        // The proxy node's mesh is zero-vertex in this file; accept either None
        // or an empty mesh set as "nothing to draw".
        match decode_model_render(&bytes) {
            None => {}
            Some(m) => assert_eq!(m.triangle_count(), 0, "proxy file must have no render tris"),
        }
    }
}
