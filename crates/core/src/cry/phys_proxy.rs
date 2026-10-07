//! Static collision proxy geometry decoded from CryEngine `.cgf`/`.cga` chunk
//! files (the `MeshPhysicsData` chunk, type `0xCCCC0018`, version `0x800`).
//!
//! The render-side chunk reader lives in [`crate::mesh`]; this module reuses the
//! same container parser ([`crate::asset::parse_cry_chunk_file`]) but decodes the
//! *physics* payload that CryEngine's `CryPhysics` turns into the static world:
//!
//! * one `MeshPhysicsData` blob per physicalized mesh, referenced by the owning
//!   `Mesh` chunk's four `physicsDataChunkID` slots at payload `+0x5C`;
//! * a 72-byte `PhysicsData` header (inertia / quaternion / center / mass /
//!   primitive type) followed by a primitive-specific shape body;
//! * primitive types observed in the shipped corpus: `0` = oriented box,
//!   `1` = convex polyhedron (triangle mesh), `4` = sphere, `5` = cylinder,
//!   `6` = cylinder variant. No capsule body occurs in SoD's files (documented
//!   gap), but the shape model carries a capsule form for callers that have one.
//!
//! The container/header layout is validated in `notes/cgf-format.md` §5.8; the
//! primitive bodies were re-validated here against all 3,218 physics chunks in
//! the shipped tree (1,085 boxes, 1,877 polyhedra, 255 cylinders, 1 sphere) and
//! close exactly (box 132 B, sphere 84 B, cylinder 104 B after the header).
//!
//! Node property flags and per-material physics types (`-1` NONE, `0x1000`
//! DEFAULT, `0x1001` NOCOLLIDE, `0x1002` OBSTRUCT, `0x1100` DEFAULT_PROXY,
//! matching `notes/cgf-format.md` §10) decide whether a proxy blocks the player.
//!
//! Non-GoL: every entry point is total — malformed input yields `None` / an
//! empty vector and never panics.

use crate::asset::parse_cry_chunk_file;
use std::collections::HashMap;

const NS_GEOM: u16 = 0xCCCC;
const CHUNK_MESH: u16 = 0x00;
const CHUNK_NODE: u16 = 0x0B;
const CHUNK_DATASTREAM: u16 = 0x16;
const CHUNK_MESH_PHYSICS: u16 = 0x18;

/// `MtlName.physicsType` when the polygons are not physicalized.
pub const PHYS_NONE: u32 = 0xFFFF_FFFF;
/// `MtlName.physicsType` — blocks player/vehicles (solid, shader assigned).
pub const PHYS_DEFAULT: u32 = 0x1000;
/// `MtlName.physicsType` — does not block, used for bullet/ray hits only.
pub const PHYS_NOCOLLIDE: u32 = 0x1001;
/// `MtlName.physicsType` — blocks AI view only; player passes through.
pub const PHYS_OBSTRUCT: u32 = 0x1002;
/// `MtlName.physicsType` — solid but invisible (no shader).
pub const PHYS_DEFAULT_PROXY: u32 = 0x1100;

const PRIM_BOX: u32 = 0;
const PRIM_POLYHEDRON: u32 = 1;
const PRIM_SPHERE: u32 = 4;
const PRIM_CYLINDER: u32 = 5;
const PRIM_CYLINDER2: u32 = 6;

/// A primitive collision shape in CryEngine Z-up metres.
///
/// All fields are in the space the decoder handed back (mesh-local until a node
/// transform is applied, model-local until an instance transform is applied).
#[derive(Debug, Clone, PartialEq)]
pub enum PhysShape {
    /// Oriented box: `world = center + basis * local`, `local` inside
    /// `[-half_extents, +half_extents]`. `basis` is row-major and orthonormal.
    Box {
        center: [f32; 3],
        half_extents: [f32; 3],
        basis: [f32; 9],
    },
    /// Sphere.
    Sphere { center: [f32; 3], radius: f32 },
    /// Finite cylinder: axis through `center`, half-length along `axis`.
    Cylinder {
        center: [f32; 3],
        axis: [f32; 3],
        radius: f32,
        half_height: f32,
    },
    /// Capsule (not emitted by SoD's files; kept for callers/completeness).
    Capsule {
        center: [f32; 3],
        axis: [f32; 3],
        radius: f32,
        half_height: f32,
    },
    /// Triangle soup (convex polyhedron / shared render mesh). `indices` is a
    /// triangle list into `vertices`.
    TriMesh {
        vertices: Vec<[f32; 3]>,
        indices: Vec<u32>,
    },
}

impl PhysShape {
    /// Axis-aligned bounds of the shape.
    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        match self {
            PhysShape::Box { center, half_extents, basis } => {
                let mut ext = [0.0f32; 3];
                for i in 0..3 {
                    let mut e = 0.0;
                    for j in 0..3 {
                        e += basis[i * 3 + j].abs() * half_extents[j].abs();
                    }
                    ext[i] = e;
                }
                (
                    [center[0] - ext[0], center[1] - ext[1], center[2] - ext[2]],
                    [center[0] + ext[0], center[1] + ext[1], center[2] + ext[2]],
                )
            }
            PhysShape::Sphere { center, radius } => {
                let r = radius.abs();
                ([center[0] - r, center[1] - r, center[2] - r], [center[0] + r, center[1] + r, center[2] + r])
            }
            PhysShape::Cylinder { center, axis, radius, half_height }
            | PhysShape::Capsule { center, axis, radius, half_height } => {
                let r = radius.abs();
                let h = half_height.abs();
                let mut ext = [0.0f32; 3];
                for i in 0..3 {
                    ext[i] = axis[i].abs() * h + r;
                }
                ([center[0] - ext[0], center[1] - ext[1], center[2] - ext[2]], [center[0] + ext[0], center[1] + ext[1], center[2] + ext[2]])
            }
            PhysShape::TriMesh { vertices, .. } => {
                let mut mn = [f32::INFINITY; 3];
                let mut mx = [f32::NEG_INFINITY; 3];
                for v in vertices {
                    for k in 0..3 {
                        if v[k].is_finite() {
                            mn[k] = mn[k].min(v[k]);
                            mx[k] = mx[k].max(v[k]);
                        }
                    }
                }
                if !mn[0].is_finite() {
                    mn = [0.0; 3];
                    mx = [0.0; 3];
                }
                (mn, mx)
            }
        }
    }
}

/// One decoded static collision proxy.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsProxy {
    pub shape: PhysShape,
    /// True when this proxy blocks the player / solid world. Cleared by a node
    /// `no_collide` / `no_player_collide` property or a NOCOLLIDE/OBSTRUCT
    /// material.
    pub collide: bool,
    /// `PhysicsData.Unknown 10` (interpreted as mass by the reference exporter).
    pub mass: f32,
    /// `MtlName.physicsType` of the owning node's material (or [`PHYS_NONE`]).
    pub material_physics: u32,
    /// True when the node carried a `no_hit_refinement` property (bullet/ray
    /// proxies are still collidable but not refined against the render mesh).
    pub no_hit_refinement: bool,
}

// ---------------------------------------------------------------------------
// minimal row-vector 4x4 (same convention as mesh.rs)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Mat4([f32; 16]);

impl Mat4 {
    fn identity() -> Mat4 {
        Mat4([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])
    }

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
}

/// Apply a 3x3 row-major matrix (row dot) to a vector is done via `mat3_mul`
/// for composition; direct application is only needed through `Transform`.
fn mat3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut r = [0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            r[i * 3 + j] = s;
        }
    }
    r
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l.is_finite() && l > 1e-9 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 1.0]
    }
}

// ---------------------------------------------------------------------------
// bounds-checked primitive readers
// ---------------------------------------------------------------------------

fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    let s = b.get(o..o + 4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
fn rd_i32(b: &[u8], o: usize) -> Option<i32> {
    rd_u32(b, o).map(|v| v as i32)
}
fn rd_f32(b: &[u8], o: usize) -> Option<f32> {
    rd_u32(b, o).map(f32::from_bits)
}
fn rd_f32n<const N: usize>(b: &[u8], o: usize) -> Option<[f32; N]> {
    let mut out = [0f32; N];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = rd_f32(b, o + i * 4)?;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// primitive shape bodies
// ---------------------------------------------------------------------------

/// Decode one primitive shape body (the bytes after the 72-byte `PhysicsData`
/// header). `prim` is the `primitiveType` field at `PhysicsData + 0x44`.
///
/// `shared_verts`/`shared_indices` are the owning render mesh's vertex and index
/// streams, needed by polyhedra that recycle the render mesh (`UseDataStream`).
pub fn decode_physics_shape(
    prim: u32,
    body: &[u8],
    shared_verts: &[[f32; 3]],
    shared_indices: &[u32],
) -> Option<PhysShape> {
    match prim {
        PRIM_BOX => decode_box(body),
        PRIM_POLYHEDRON => decode_polyhedron(body, shared_verts, shared_indices),
        PRIM_SPHERE => decode_sphere(body),
        PRIM_CYLINDER | PRIM_CYLINDER2 => decode_cylinder(body),
        _ => None,
    }
}

/// `PhysicsCube`: two identical `PhysicsStruct1` blocks (`Matrix33` + `i32` +
/// `f32[6]`) plus a trailing `i32`; the `f32[6]` is `center[3], half[3]`.
fn decode_box(body: &[u8]) -> Option<PhysShape> {
    if body.len() < 132 {
        return None;
    }
    let basis = rd_f32n::<9>(body, 0)?;
    let f = rd_f32n::<6>(body, 40)?;
    let center = [f[0], f[1], f[2]];
    let half_extents = [f[3].abs(), f[4].abs(), f[5].abs()];
    if !center.iter().all(|v| v.is_finite()) || !half_extents.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(PhysShape::Box { center, half_extents, basis: sanitize_basis(basis) })
}

fn sanitize_basis(mut m: [f32; 9]) -> [f32; 9] {
    if !m.iter().all(|v| v.is_finite()) {
        return [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    }
    // Orthonormalise rows so degenerate exporter data cannot amplify bounds.
    let rows = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
    let r0 = normalize(rows[0]);
    let mut r1 = rows[1];
    let d = r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2];
    r1 = [r1[0] - r0[0] * d, r1[1] - r0[1] * d, r1[2] - r0[2] * d];
    let r1 = normalize(r1);
    let r2 = [
        r0[1] * r1[2] - r0[2] * r1[1],
        r0[2] * r1[0] - r0[0] * r1[2],
        r0[0] * r1[1] - r0[1] * r1[0],
    ];
    m = [r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2]];
    m
}

/// `PhysicsPolyhedron`. `nv` vertices, `nt` triangles; when
/// `UseDataStream == 0` the vertices/triangles are stored inline, otherwise the
/// triangles index the owning render mesh (`shared_indices` picked by the
/// per-triangle `Triangle Map`, vertices from `shared_verts`).
fn decode_polyhedron(
    body: &[u8],
    shared_verts: &[[f32; 3]],
    shared_indices: &[u32],
) -> Option<PhysShape> {
    let mut o = 0usize;
    let nv = rd_u32(body, o)? as usize;
    o += 4;
    let nt = rd_u32(body, o)? as usize;
    o += 4;
    let _u17 = rd_i32(body, o)?;
    o += 4;
    let _u18 = rd_i32(body, o)?;
    o += 4;
    let has_map = *body.get(o)?;
    o += 1;
    if has_map == 1 {
        // Vertex Map: nv u16 entries, ignored (only used for welding).
        o = o.checked_add(nv.checked_mul(2)?)?;
    }
    let use_stream = *body.get(o)?;
    o += 1;
    if nv > 1_000_000 || nt > 2_000_000 {
        return None;
    }

    if use_stream == 0 {
        let mut verts = Vec::with_capacity(nv);
        for i in 0..nv {
            verts.push(rd_f32n::<3>(body, o.checked_add(i * 12)?)?);
        }
        o = o.checked_add(nv.checked_mul(12)?)?;
        let mut tris = Vec::with_capacity(nt.saturating_mul(3));
        for i in 0..nt.saturating_mul(3) {
            let v = body.get(o.checked_add(i * 2)?..o.checked_add(i * 2 + 2)?)?;
            tris.push(u16::from_le_bytes([v[0], v[1]]) as u32);
        }
        if verts.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
            return None;
        }
        Some(PhysShape::TriMesh { vertices: verts, indices: tris })
    } else {
        // Triangle Map: one u16 per triangle selecting the render triangle.
        let mut map = Vec::with_capacity(nt);
        for i in 0..nt {
            let v = body.get(o.checked_add(i * 2)?..o.checked_add(i * 2 + 2)?)?;
            map.push(u16::from_le_bytes([v[0], v[1]]) as usize);
        }
        let render_tris = shared_indices.len() / 3;
        let mut indices = Vec::with_capacity(nt.saturating_mul(3));
        for &t in &map {
            if t >= render_tris {
                continue;
            }
            let base = t * 3;
            indices.extend_from_slice(&shared_indices[base..base + 3]);
        }
        if shared_verts.is_empty() || indices.is_empty() {
            return None;
        }
        Some(PhysShape::TriMesh { vertices: shared_verts.to_vec(), indices })
    }
}

/// `Primitive Type == 4`: `center` + `radius` then a `PhysicsStruct1` copy.
fn decode_sphere(body: &[u8]) -> Option<PhysShape> {
    if body.len() < 16 {
        return None;
    }
    let center = rd_f32n::<3>(body, 0)?;
    let radius = rd_f32(body, 12)?;
    if !center.iter().all(|v| v.is_finite()) || !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    Some(PhysShape::Sphere { center, radius })
}

/// `PhysicsCylinder`: `f32[8]` (`center, axis, radius, height`) + `i32`, then a
/// `PhysicsDataType2` (`Matrix33 + int + f32[6] + int`) whose `f32[6]` repeats
/// `center, radius, radius, height`.
fn decode_cylinder(body: &[u8]) -> Option<PhysShape> {
    if body.len() < 104 {
        return None;
    }
    let f = rd_f32n::<8>(body, 0)?;
    let center = [f[0], f[1], f[2]];
    let axis = normalize([f[3], f[4], f[5]]);
    let radius = f[6].abs();
    let height = f[7].abs();
    if !center.iter().all(|v| v.is_finite()) || !radius.is_finite() || radius <= 0.0 || !height.is_finite() {
        return None;
    }
    Some(PhysShape::Cylinder {
        center,
        axis,
        radius,
        half_height: height * 0.5,
    })
}

// ---------------------------------------------------------------------------
// nodes / materials / flags
// ---------------------------------------------------------------------------

struct NodeRec {
    object: i32,
    parent: i32,
    local: Mat4,
    no_collide: bool,
    no_hit_refinement: bool,
    material: u32,
}

fn rd_matrix44(b: &[u8], o: usize) -> Option<Mat4> {
    rd_f32n::<16>(b, o).map(Mat4)
}

/// Split a CryEngine node property string into lower-cased keys. Both `\n` and
/// real newlines are separators, matching how the exporter writes them.
fn property_keys(s: &str) -> Vec<String> {
    s.split(['\n', '\r', ','])
        .map(|l| {
            let l = l.trim();
            let key = l.split('=').next().unwrap_or(l).trim().to_ascii_lowercase();
            key
        })
        .filter(|k| !k.is_empty())
        .collect()
}

fn parse_nodes(cf: &crate::asset::CryChunkFile, file: &[u8]) -> (Vec<NodeRec>, HashMap<u32, usize>) {
    let mut nodes = Vec::new();
    let mut by_id = HashMap::new();
    for ch in &cf.chunks {
        if ch.namespace() != NS_GEOM || ch.kind() != CHUNK_NODE {
            continue;
        }
        let Some(p) = ch.payload(file) else { continue };
        let object = rd_i32(p, 0x40).unwrap_or(-1);
        let parent = rd_i32(p, 0x44).unwrap_or(-1);
        let material = rd_u32(p, 0x4C).unwrap_or(0);
        let local = rd_matrix44(p, 0x54).unwrap_or_else(Mat4::identity);
        let mut no_collide = false;
        let mut no_hit_refinement = false;
        if let Some(len) = rd_u32(p, 0xC8) {
            let len = len as usize;
            if let Some(text) = p.get(0xCC..0xCC + len) {
                let s = String::from_utf8_lossy(text);
                for k in property_keys(&s) {
                    match k.as_str() {
                        "no_collide" | "no_player_collide" | "no_collision" => no_collide = true,
                        "no_hit_refinement" => no_hit_refinement = true,
                        _ => {}
                    }
                }
            }
        }
        by_id.insert(ch.id, nodes.len());
        nodes.push(NodeRec { object, parent, local, no_collide, no_hit_refinement, material });
    }
    (nodes, by_id)
}

/// Accumulated local-to-model transform of a node (row-vector, `child * parent`).
fn node_world(nodes: &[NodeRec], by_id: &HashMap<u32, usize>, start: usize) -> Mat4 {
    let mut world = nodes[start].local;
    let mut parent = nodes[start].parent;
    let mut guard = 0;
    while parent >= 0 && guard < 256 {
        match by_id.get(&(parent as u32)) {
            Some(&pi) => {
                world = world.mul(&nodes[pi].local);
                parent = nodes[pi].parent;
            }
            None => break,
        }
        guard += 1;
    }
    world
}

/// `MtlName.physicsType` of a material chunk id, or [`PHYS_NONE`].
fn node_material_type(
    cf: &crate::asset::CryChunkFile,
    file: &[u8],
    by_id: &HashMap<u32, usize>,
    mat_id: u32,
) -> u32 {
    let Some(&idx) = by_id.get(&mat_id) else { return PHYS_NONE };
    let Some(p) = cf.chunks[idx].payload(file) else { return PHYS_NONE };
    rd_i32(p, 0x88).map(|v| v as u32).unwrap_or(PHYS_NONE)
}

// ---------------------------------------------------------------------------
// proxy decoding
// ---------------------------------------------------------------------------

/// Decode every static collision proxy in a `.cgf`/`.cga` file.
///
/// The proxies are returned in model space (the owning node's accumulated
/// transform applied), ready to be placed by an instance transform. Physics
/// proxies whose node property or material marks them non-solid are still
/// returned but with `collide == false`, so callers can decide.
pub fn decode_physics_proxies(file: &[u8]) -> Vec<PhysicsProxy> {
    let Some(cf) = parse_cry_chunk_file(file) else { return Vec::new() };
    let mut by_id: HashMap<u32, usize> = HashMap::new();
    for (i, c) in cf.chunks.iter().enumerate() {
        by_id.insert(c.id, i);
    }
    let (nodes, node_by_id) = parse_nodes(&cf, file);
    // node object id -> node index (first wins).
    let mut node_of_object: HashMap<i32, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        node_of_object.entry(n.object).or_insert(i);
    }

    let mut out = Vec::new();
    for ch in &cf.chunks {
        if ch.namespace() != NS_GEOM || ch.kind() != CHUNK_MESH {
            continue;
        }
        let Some(mp) = ch.payload(file) else { continue };
        let owner = node_of_object.get(&(ch.id as i32)).copied();
        let world = owner.map(|i| node_world(&nodes, &node_by_id, i)).unwrap_or_else(Mat4::identity);
        let no_collide = owner.map(|i| nodes[i].no_collide).unwrap_or(false);
        let no_hit_refinement = owner.map(|i| nodes[i].no_hit_refinement).unwrap_or(false);
        let phys_type = owner
            .map(|i| node_material_type(&cf, file, &by_id, nodes[i].material))
            .unwrap_or(PHYS_NONE);

        // Render streams, only populated lazily when a polyhedron reuses them.
        let mut shared: Option<(Vec<[f32; 3]>, Vec<u32>)> = None;

        for slot in 0..4usize {
            let Some(pid) = rd_u32(mp, 0x5C + slot * 4) else { break };
            if pid == 0 {
                continue;
            }
            let Some(&idx) = by_id.get(&pid) else { continue };
            let Some(pp) = cf.chunks[idx].payload(file) else { continue };
            if cf.chunks[idx].namespace() != NS_GEOM || cf.chunks[idx].kind() != CHUNK_MESH_PHYSICS {
                continue;
            }
            let Some((shape, mass)) = decode_mesh_physics(pp, || {
                shared.get_or_insert_with(|| read_render_streams(&cf, file, &by_id, mp)).clone()
            }) else {
                continue;
            };
            let shape = transform_shape(shape, &Transform::from_mat4(&world));
            let collide = !no_collide
                && !matches!(phys_type, PHYS_NOCOLLIDE | PHYS_OBSTRUCT);
            out.push(PhysicsProxy {
                shape,
                collide,
                mass,
                material_physics: phys_type,
                no_hit_refinement,
            });
        }
    }
    out
}

/// Read the owning Mesh chunk's vertex (slot 0) and index (slot 5) streams.
fn read_render_streams(
    cf: &crate::asset::CryChunkFile,
    file: &[u8],
    by_id: &HashMap<u32, usize>,
    mesh_payload: &[u8],
) -> (Vec<[f32; 3]>, Vec<u32>) {
    let refs = |slot: usize| rd_u32(mesh_payload, 0x1C + slot * 4).unwrap_or(0);
    let verts = read_stream(cf, file, by_id, refs(0))
        .map(|(ty, num, bpe, body)| decode_positions(ty, num, bpe, body))
        .unwrap_or_default();
    let indices = read_stream(cf, file, by_id, refs(5))
        .map(|(ty, num, bpe, body)| decode_indices(ty, num, bpe, body))
        .unwrap_or_default();
    (verts, indices)
}

fn read_stream<'a>(
    cf: &'a crate::asset::CryChunkFile,
    file: &'a [u8],
    by_id: &HashMap<u32, usize>,
    id: u32,
) -> Option<(u32, usize, usize, &'a [u8])> {
    if id == 0 {
        return None;
    }
    let &idx = by_id.get(&id)?;
    if cf.chunks[idx].namespace() != NS_GEOM || cf.chunks[idx].kind() != CHUNK_DATASTREAM {
        return None;
    }
    let p = cf.chunks[idx].payload(file)?;
    Some((
        rd_u32(p, 4)?,
        rd_u32(p, 8)? as usize,
        rd_u32(p, 12)? as usize,
        p.get(24..)?,
    ))
}

fn decode_positions(ty: u32, num: usize, bpe: usize, body: &[u8]) -> Vec<[f32; 3]> {
    if ty != 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(num);
    match bpe {
        12 => {
            for i in 0..num {
                let Some(o) = i.checked_mul(12) else { break };
                let Some(v) = rd_f32n::<3>(body, o) else { break };
                out.push(v);
            }
        }
        8 => {
            for i in 0..num {
                let Some(o) = i.checked_mul(8) else { break };
                let Some(s) = body.get(o..o + 8) else { break };
                let x = u16::from_le_bytes([s[0], s[1]]);
                let y = u16::from_le_bytes([s[2], s[3]]);
                let z = u16::from_le_bytes([s[4], s[5]]);
                out.push([
                    crate::mesh::half_to_f32(x),
                    crate::mesh::half_to_f32(y),
                    crate::mesh::half_to_f32(z),
                ]);
            }
        }
        _ => {}
    }
    out
}

fn decode_indices(ty: u32, num: usize, bpe: usize, body: &[u8]) -> Vec<u32> {
    if ty != 5 || bpe != 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(num);
    for i in 0..num {
        let Some(o) = i.checked_mul(2) else { break };
        let Some(s) = body.get(o..o + 2) else { break };
        out.push(u16::from_le_bytes([s[0], s[1]]) as u32);
    }
    out
}

/// Decode the `MeshPhysicsData` payload: 24-byte chunk header, 72-byte
/// `PhysicsData` header, then the primitive body.
fn decode_mesh_physics<F: FnOnce() -> (Vec<[f32; 3]>, Vec<u32>)>(
    pp: &[u8],
    shared: F,
) -> Option<(PhysShape, f32)> {
    if pp.len() < 24 {
        return None;
    }
    let pdsize = rd_u32(pp, 0)? as usize;
    if pdsize < 72 {
        return None;
    }
    let end = 24usize.checked_add(pdsize)?;
    if end > pp.len() {
        return None;
    }
    let mass = rd_f32(pp, 24 + 48)?;
    let prim = rd_u32(pp, 24 + 68)?;
    let body = pp.get(24 + 72..end)?;
    let (verts, indices) = if prim == PRIM_POLYHEDRON {
        shared()
    } else {
        (Vec::new(), Vec::new())
    };
    let shape = decode_physics_shape(prim, body, &verts, &indices)?;
    Some((shape, mass))
}

// ---------------------------------------------------------------------------
// shape transform
// ---------------------------------------------------------------------------

/// Apply a node/instance transform to a shape. `Transform` maps model-local
/// points as `world = rot * local + pos` (column-vector convention, matching the
/// renderer's `Transform { translation, rotation, scale }`). Primitive extents
/// are folded with the mean scale, so rigid transforms are exact.
pub fn transform_shape(shape: PhysShape, m: &Transform) -> PhysShape {
    match shape {
        PhysShape::Box { center, half_extents, basis } => {
            let s = m.mean_scale();
            PhysShape::Box {
                center: m.xform_point(center),
                half_extents: [half_extents[0] * s, half_extents[1] * s, half_extents[2] * s],
                basis: mat3_mul(&m.rot, &basis),
            }
        }
        PhysShape::Sphere { center, radius } => PhysShape::Sphere {
            center: m.xform_point(center),
            radius: radius * m.mean_scale(),
        },
        PhysShape::Cylinder { center, axis, radius, half_height } => {
            let s = m.mean_scale();
            PhysShape::Cylinder {
                center: m.xform_point(center),
                axis: normalize(m.xform_dir(axis)),
                radius: radius * s,
                half_height: half_height * s,
            }
        }
        PhysShape::Capsule { center, axis, radius, half_height } => {
            let s = m.mean_scale();
            PhysShape::Capsule {
                center: m.xform_point(center),
                axis: normalize(m.xform_dir(axis)),
                radius: radius * s,
                half_height: half_height * s,
            }
        }
        PhysShape::TriMesh { vertices, indices } => PhysShape::TriMesh {
            vertices: vertices.into_iter().map(|v| m.xform_point(v)).collect(),
            indices,
        },
    }
}

/// A placement transform: `world = rot * local + pos`, `rot` row-major.
#[derive(Clone, Copy)]
pub struct Transform {
    rot: [f32; 9],
    pos: [f32; 3],
}

impl Transform {
    pub fn identity() -> Transform {
        Transform { rot: [1., 0., 0., 0., 1., 0., 0., 0., 1.], pos: [0.0; 3] }
    }

    /// From a row-vector CryEngine `Matrix44` (as stored in `Node` chunks and
    /// applied by `mesh.rs`): `world = N^T * local + t`.
    fn from_mat4(m: &Mat4) -> Transform {
        let n = [
            m.0[0], m.0[1], m.0[2],
            m.0[4], m.0[5], m.0[6],
            m.0[8], m.0[9], m.0[10],
        ];
        Transform { rot: transpose9(&n), pos: [m.0[12], m.0[13], m.0[14]] }
    }

    /// From position, a unit quaternion `[x,y,z,w]` and per-axis scale, matching
    /// the renderer's `Transform { translation, rotation, scale }`.
    pub fn from_trs(pos: [f32; 3], q: [f32; 4], scale: [f32; 3]) -> Transform {
        let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
        let r = [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ];
        // Scale the columns: `world = R * (S * local)`.
        let mut rot = [0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                rot[i * 3 + j] = r[i * 3 + j] * scale[j];
            }
        }
        if rot.iter().chain(pos.iter()).all(|v| v.is_finite()) {
            Transform { rot, pos }
        } else {
            Transform::identity()
        }
    }

    /// `world = rot * p + pos`.
    pub fn xform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let r = &self.rot;
        [
            r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + self.pos[0],
            r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + self.pos[1],
            r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + self.pos[2],
        ]
    }

    /// Direction transform (translation omitted).
    pub fn xform_dir(&self, v: [f32; 3]) -> [f32; 3] {
        let r = &self.rot;
        [
            r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
            r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
            r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
        ]
    }

    /// Mean column norm (the per-axis scale folded into `rot`).
    pub fn mean_scale(&self) -> f32 {
        let n = |a: f32, b: f32, c: f32| (a * a + b * b + c * c).sqrt();
        let s = (n(self.rot[0], self.rot[3], self.rot[6])
            + n(self.rot[1], self.rot[4], self.rot[7])
            + n(self.rot[2], self.rot[5], self.rot[8]))
            / 3.0;
        if s.is_finite() && s > 1e-9 {
            s
        } else {
            1.0
        }
    }

    /// Compose `self` (outer) with `inner` (applied first).
    #[allow(dead_code)]
    pub fn compose(&self, inner: &Transform) -> Transform {
        Transform { rot: mat3_mul(&self.rot, &inner.rot), pos: self.xform_point(inner.pos) }
    }
}

fn transpose9(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_body_box(center: [f32; 3], half: [f32; 3]) -> Vec<u8> {
        let mut b = vec![0u8; 132];
        // identity basis
        for (i, v) in [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        b[36..40].copy_from_slice(&1i32.to_le_bytes());
        for i in 0..3 {
            b[40 + i * 4..44 + i * 4].copy_from_slice(&center[i].to_le_bytes());
            b[52 + i * 4..56 + i * 4].copy_from_slice(&half[i].to_le_bytes());
        }
        b
    }

    #[test]
    fn box_decodes_center_and_half_extents() {
        let b = flat_body_box([1.0, 2.0, 3.0], [0.5, 1.0, 2.0]);
        match decode_physics_shape(PRIM_BOX, &b, &[], &[]).unwrap() {
            PhysShape::Box { center, half_extents, .. } => {
                assert_eq!(center, [1.0, 2.0, 3.0]);
                assert_eq!(half_extents, [0.5, 1.0, 2.0]);
            }
            s => panic!("expected box, got {s:?}"),
        }
    }

    #[test]
    fn sphere_decodes_radius() {
        let mut b = vec![0u8; 84];
        for (i, v) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        b[12..16].copy_from_slice(&2.5f32.to_le_bytes());
        match decode_physics_shape(PRIM_SPHERE, &b, &[], &[]).unwrap() {
            PhysShape::Sphere { center, radius } => {
                assert_eq!(center, [1.0, 2.0, 3.0]);
                assert!((radius - 2.5).abs() < 1e-6);
            }
            s => panic!("expected sphere, got {s:?}"),
        }
    }

    #[test]
    fn cylinder_decodes_axis_radius_height() {
        let mut b = vec![0u8; 104];
        for (i, v) in [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.25, 2.0].iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        match decode_physics_shape(PRIM_CYLINDER, &b, &[], &[]).unwrap() {
            PhysShape::Cylinder { center, axis, radius, half_height } => {
                assert_eq!(center, [0.0, 0.0, 1.0]);
                assert!((axis[2] - 1.0).abs() < 1e-6);
                assert!((radius - 0.25).abs() < 1e-6);
                assert!((half_height - 1.0).abs() < 1e-6);
            }
            s => panic!("expected cylinder, got {s:?}"),
        }
    }

    #[test]
    fn polyhedron_inline_vertices_and_triangles() {
        let mut b = Vec::new();
        b.extend_from_slice(&3u32.to_le_bytes()); // nv
        b.extend_from_slice(&1u32.to_le_bytes()); // nt
        b.extend_from_slice(&0i32.to_le_bytes());
        b.extend_from_slice(&0i32.to_le_bytes());
        b.push(0); // no vertex map
        b.push(0); // use data stream
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
        for i in [0u16, 1, 2] {
            b.extend_from_slice(&i.to_le_bytes());
        }
        b.push(0);
        b.push(0);
        match decode_physics_shape(PRIM_POLYHEDRON, &b, &[], &[]).unwrap() {
            PhysShape::TriMesh { vertices, indices } => {
                assert_eq!(vertices.len(), 3);
                assert_eq!(indices, vec![0, 1, 2]);
            }
            s => panic!("expected trimesh, got {s:?}"),
        }
    }

    #[test]
    fn polyhedron_shared_stream_uses_triangle_map() {
        // Two render triangles; map selects the second using streams not needed.
        let verts = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inds = vec![0u32, 1, 2, 0, 1, 3];
        let mut b = Vec::new();
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0i32.to_le_bytes());
        b.extend_from_slice(&0i32.to_le_bytes());
        b.push(1); // vertex map present
        for _ in 0..4 {
            b.extend_from_slice(&0u16.to_le_bytes());
        }
        b.push(1); // use data stream
        b.extend_from_slice(&1u16.to_le_bytes()); // triangle map -> triangle #1
        match decode_physics_shape(PRIM_POLYHEDRON, &b, &verts, &inds).unwrap() {
            PhysShape::TriMesh { indices, .. } => assert_eq!(indices, vec![0, 1, 3]),
            s => panic!("expected trimesh, got {s:?}"),
        }
    }

    #[test]
    fn transform_box_translates_and_scales() {
        let shape = PhysShape::Box {
            center: [0.0, 0.0, 0.0],
            half_extents: [1.0, 1.0, 1.0],
            basis: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        };
        let t = Transform::from_trs([10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0]);
        match transform_shape(shape, &t) {
            PhysShape::Box { center, half_extents, .. } => {
                assert_eq!(center, [10.0, 20.0, 30.0]);
                assert_eq!(half_extents, [2.0, 2.0, 2.0]);
            }
            s => panic!("expected box, got {s:?}"),
        }
    }

    /// The placement transform must match the renderer's
    /// `Transform { translation, rotation, scale }` (column-vector), not its
    /// inverse: a +90 deg yaw about Z maps local +X to world +Y.
    #[test]
    fn instance_rotation_matches_renderer_convention() {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let q = [0.0, 0.0, s, s]; // +90 deg about Z
        let t = Transform::from_trs([5.0, 0.0, 0.0], q, [1.0, 1.0, 1.0]);
        let x = t.xform_point([1.0, 0.0, 0.0]);
        assert!((x[0] - 5.0).abs() < 1e-5 && (x[1] - 1.0).abs() < 1e-5, "x={x:?}");
        let y = t.xform_point([0.0, 1.0, 0.0]);
        assert!((y[0] - 4.0).abs() < 1e-5 && y[1].abs() < 1e-5, "y={y:?}");
    }

    #[test]
    fn malformed_is_total() {
        assert!(decode_physics_shape(PRIM_BOX, &[], &[], &[]).is_none());
        assert!(decode_physics_shape(PRIM_SPHERE, &[0u8; 4], &[], &[]).is_none());
        assert!(decode_physics_shape(PRIM_CYLINDER, &[0u8; 8], &[], &[]).is_none());
        assert!(decode_physics_shape(999, &[0u8; 200], &[], &[]).is_none());
        assert!(decode_physics_proxies(&[]).is_empty());
        assert!(decode_physics_proxies(&[0xff; 64]).is_empty());
    }

    /// Corpus census (run with `--ignored`): decode every shipped `.cgf`/`.cga`
    /// and tally the proxy shape types and collision flags.
    #[test]
    #[ignore]
    fn corpus_shape_census() {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if matches!(
                    p.extension().and_then(|x| x.to_str()).map(|x| x.to_ascii_lowercase()).as_deref(),
                    Some("cgf") | Some("cga")
                ) {
                    out.push(p);
                }
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game");
        if !root.is_dir() {
            return;
        }
        let mut files = Vec::new();
        walk(&root, &mut files);
        let mut boxes = 0usize;
        let mut polys = 0usize;
        let mut cyls = 0usize;
        let mut spheres = 0usize;
        let mut solid = 0usize;
        let mut with_phys = 0usize;
        let mut max_tris = 0usize;
        for f in &files {
            let Ok(bytes) = std::fs::read(f) else { continue };
            let proxies = decode_physics_proxies(&bytes);
            if !proxies.is_empty() {
                with_phys += 1;
            }
            for p in &proxies {
                if p.collide {
                    solid += 1;
                }
                match &p.shape {
                    PhysShape::Box { .. } => boxes += 1,
                    PhysShape::TriMesh { indices, .. } => {
                        polys += 1;
                        max_tris = max_tris.max(indices.len() / 3);
                    }
                    PhysShape::Cylinder { .. } => cyls += 1,
                    PhysShape::Sphere { .. } => spheres += 1,
                    PhysShape::Capsule { .. } => {}
                }
            }
        }
        eprintln!(
            "[phys-census] files={} with_physics={} box={} poly={} cylinder={} sphere={} solid={} max_tris={}",
            files.len(),
            with_phys,
            boxes,
            polys,
            cyls,
            spheres,
            solid,
            max_tris
        );
        assert!(boxes > 0 && polys > 0 && cyls > 0);
    }

    #[test]
    fn real_proxies_if_available() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game");
        let cases: &[(&str, u32)] = &[
            ("objects/objects/invisible_wall/you_shall_not_pass.cgf", PRIM_BOX),
            ("particles/particles/geometry/skullshards/zed_skullshards.cgf", PRIM_POLYHEDRON),
            (
                "BeltwayC/Game/objects/class3_2/objects/vehicles/xplash_truck/xplash_ad_truck_can.cgf",
                PRIM_CYLINDER,
            ),
            ("level/brush/121.cgf", PRIM_POLYHEDRON),
            (
                "BeltwayC/Game/objects/class3_2/roads/breakable_highway_marker/breakable_highway_marker.cgf",
                PRIM_BOX,
            ),
        ];
        if !root.is_dir() {
            return;
        }
        for &(rel, prim) in cases {
            let Ok(bytes) = std::fs::read(root.join(rel)) else { continue };
            let proxies = decode_physics_proxies(&bytes);
            assert!(!proxies.is_empty(), "{rel}");
            let got = match &proxies[0].shape {
                PhysShape::Box { .. } => PRIM_BOX,
                PhysShape::TriMesh { .. } => PRIM_POLYHEDRON,
                PhysShape::Cylinder { .. } => PRIM_CYLINDER,
                PhysShape::Sphere { .. } => PRIM_SPHERE,
                PhysShape::Capsule { .. } => PRIM_CYLINDER2,
            };
            assert_eq!(got, prim, "{rel}");
            if let PhysShape::TriMesh { vertices, indices } = &proxies[0].shape {
                assert!(!vertices.is_empty() && indices.len() >= 3, "{rel} empty trimesh");
                assert!(indices.iter().all(|&i| (i as usize) < vertices.len()), "{rel} bad index");
            }
        }
    }
}
