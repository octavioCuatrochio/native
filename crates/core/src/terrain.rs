//! Decode CryEngine 3 `terrain/terrain.dat` for *State of Decay: YOSE*.
//!
//! Port of the validated algorithms in `notes/terrain-decode.md` and
//! `tools/terrain_inspect.py`:
//!
//! * `STerrainChunkHeader` + `STerrainInfo` (32 bytes) and the three Undead Labs
//!   prototype tables, used to locate the terrain/object streams.
//! * The `CTerrainNode` heightmap **quadtree**: every node begins with a 44-byte
//!   `STerrainNodeChunk`; `nSize == 0` means an internal node with exactly four
//!   children, otherwise it is a leaf carrying an `nSize*nSize` little-endian
//!   `u16` field. Height is node-local: `height_m = fOffset + (raw & 0xFFF0) * fRange`
//!   (the low 4 bits are the local surface-type palette index). After the field (or
//!   none, for internal nodes) come `bit_shift` `f32` geometry errors and `nSurfaceTypesNum`
//!   palette bytes, each block 4-byte aligned with `0xDE` fill.
//! * The `COctreeNode` static-object **octree**: 32-byte `SOcTreeNodeChunk`, an
//!   object block of `nObjectsBlockSize` bytes, then the children selected by
//!   `ucChildsMask`. Object blocks are type-sorted (`int32 eType` + type chunk); the
//!   `Brush` chunk yields a `Matrix34` transform.
//!
//! The reconstructed grid is `1024 x 1024` cells (a 4 m lattice over the 4096 m
//! world), matching the note's validation: Class3 `min 21.32 / max 322.50 /
//! mean 140.89 m`, `0` missing cells.
//!
//! # Coordinate convention
//!
//! Everything is **Z-up, right-handed** (CryEngine native). [`build_mesh`] emits
//! vertex positions `[x, y, z]` with `z` = height in metres. Triangles are wound
//! **counter-clockwise when viewed from +Z** (up-facing, `+Z` normal by the
//! right-hand rule). `samples` per side are cell-centre height values on a
//! regular lattice of `spacing` metres.
//!
//! All entry points are total: malformed input yields `None` / an empty vector,
//! and no decode path can panic.

use std::collections::HashMap;

/// Header length of `STerrainChunkHeader` + `STerrainInfo`.
const HEADER_LEN: usize = 0x20;
/// Length of `STerrainNodeChunk`.
const TERRAIN_NODE_LEN: usize = 44;
/// Length of `SOcTreeNodeChunk`.
const OBJECT_NODE_LEN: usize = 32;
/// Validated reconstruction resolution (samples per side).
const RECON_GRID: usize = 1024;
/// Hard recursion-depth guard for both trees.
const MAX_DEPTH: usize = 64;
/// Hard total-node guards, so malformed data cannot run unbounded.
const MAX_TERRAIN_NODES: usize = 5_000_000;
const MAX_OBJECT_NODES: usize = 5_000_000;

/// Surface id stored for samples that are undefined (`surface_local == 14`) or a
/// hole (`surface_local == 15`), or whose local index is outside the node palette.
pub const SURFACE_UNDEFINED: u8 = 255;

/// A reconstructed terrain heightmap.
#[derive(Debug, Clone)]
pub struct Terrain {
    /// Samples per side (the grid is `samples * samples`).
    pub samples: usize,
    /// Metres between adjacent samples.
    pub spacing: f32,
    /// AABB minimum of the terrain; `x`/`y` are the world origin, `z` is the
    /// lowest reconstructed height.
    pub world_min: [f32; 3],
    /// `samples * samples` heights in metres, row-major with `y` as the row and
    /// `x` as the column: `heights[y * samples + x]`.
    pub heights: Vec<f32>,
    /// `samples * samples` global surface ids, row-major with the same indexing as
    /// [`Terrain::heights`]: `surfaces[y * samples + x]` is the `leveldata.xml`
    /// `SurfaceTypeID` of the sample, and [`SURFACE_UNDEFINED`] (`255`) marks an
    /// undefined/hole sample. Empty only if the source had no leaves.
    pub surfaces: Vec<u8>,
    /// Minimum height (m).
    pub min: f32,
    /// Maximum height (m).
    pub max: f32,
}

/// A placed static render node, currently the `Brush` type.
///
/// `prototype` indexes the object prototype table (Table B, `IStatObj`), matching
/// the `m_nObjectTypeIndex` field of the render node. Other render-node types are
/// walked (so block boundaries stay exact) but not emitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticObject {
    /// Prototype index into Table B.
    pub prototype: u16,
    /// World translation `(x, y, z)` in metres.
    pub pos: [f32; 3],
    /// Orientation quaternion in `[x, y, z, w]` (scalar-last) order.
    pub rot: [f32; 4],
    /// Best-effort uniform scale (mean of the three `Matrix34` row norms).
    pub scale: f32,
}

// ---------------------------------------------------------------------------
// primitive readers (all bounds-checked)
// ---------------------------------------------------------------------------

#[inline]
fn arr<const N: usize>(b: &[u8], o: usize) -> Option<[u8; N]> {
    let end = o.checked_add(N)?;
    let s = b.get(o..end)?;
    let mut a = [0u8; N];
    a.copy_from_slice(s);
    Some(a)
}

#[inline]
fn rd_u16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(arr::<2>(b, o)?))
}

#[inline]
fn rd_i16(b: &[u8], o: usize) -> Option<i16> {
    Some(i16::from_le_bytes(arr::<2>(b, o)?))
}

#[inline]
fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(arr::<4>(b, o)?))
}

#[inline]
fn rd_i32(b: &[u8], o: usize) -> Option<i32> {
    Some(i32::from_le_bytes(arr::<4>(b, o)?))
}

#[inline]
fn rd_f32(b: &[u8], o: usize) -> Option<f32> {
    rd_u32(b, o).map(f32::from_bits)
}

#[inline]
fn align4(x: usize) -> usize {
    x.saturating_add(3) & !3
}

// ---------------------------------------------------------------------------
// header + prototype tables
// ---------------------------------------------------------------------------

struct Header {
    world: f64,
    sector: f64,
    geom_shift: usize,
    grid: usize,
}

impl Header {
    fn parse(dat: &[u8]) -> Option<Header> {
        if dat.len() < HEADER_LEN {
            return None;
        }
        // bit0 of nFlags = SERIALIZATION_FLAG_BIG_ENDIAN; the xbox360 variant is
        // untested and would make every read below wrong, so reject it.
        let flags = *dat.get(2)?;
        if flags & 1 != 0 {
            return None;
        }
        let n_height = rd_i32(dat, 0x08)?;
        let n_unit = rd_i32(dat, 0x0C)?;
        let n_sector = rd_i32(dat, 0x10)?;
        if n_height <= 0 || n_unit <= 0 || n_sector <= 0 {
            return None;
        }
        if n_sector % n_unit != 0 {
            return None;
        }
        let units = (n_sector / n_unit) as u32;
        if !units.is_power_of_two() {
            return None;
        }
        let n_flag2 = rd_i32(dat, 4)?;
        if n_flag2 <= 0 {
            return None;
        }
        Some(Header {
            world: n_height as f64 * n_unit as f64,
            sector: n_sector as f64,
            geom_shift: units.trailing_zeros() as usize,
            grid: RECON_GRID,
        })
    }
}

/// Offset of the terrain quadtree, immediately after the three prototype tables.
fn data_offset(dat: &[u8]) -> Option<usize> {
    let count_a = rd_u32(dat, 0x20)? as usize;
    let off_b_count = 0x24usize.checked_add(count_a.checked_mul(0x150)?)?;
    let count_b = rd_u32(dat, off_b_count)? as usize;
    let off_c_count = off_b_count
        .checked_add(4)?
        .checked_add(count_b.checked_mul(0x100)?)?;
    let count_c = rd_u32(dat, off_c_count)? as usize;
    let off_data = off_c_count
        .checked_add(4)?
        .checked_add(count_c.checked_mul(0x100)?)?;
    if off_data > dat.len() {
        return None;
    }
    Some(off_data)
}

// ---------------------------------------------------------------------------
// terrain quadtree
// ---------------------------------------------------------------------------

struct Leaf {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    offset: f64,
    range: f64,
    size: usize,
    raw: Vec<u16>,
    /// `pTypes[nSurfaceTypesNum]`: node-local palette of global `SurfaceTypeID`s.
    /// The low 4 bits of a raw sample index this (`14` undefined, `15` hole).
    types: Vec<u8>,
}

/// Parse the full quadtree. Returns `(end_offset, leaves)`.
///
/// With `collect == false` the leaf sample arrays are not materialised (only the
/// byte stream is advanced), which is all [`decode_static_objects`] needs.
fn walk_terrain(dat: &[u8], hdr: &Header, collect: bool) -> Option<(usize, Vec<Leaf>)> {
    let start = data_offset(dat)?;
    let mut pos = start;
    let mut budget = MAX_TERRAIN_NODES;
    let mut leaves = Vec::new();
    walk_node(dat, &mut pos, hdr.geom_shift, 0, &mut budget, collect, &mut leaves)?;
    Some((pos, leaves))
}

fn walk_node(
    dat: &[u8],
    pos: &mut usize,
    geom: usize,
    depth: usize,
    budget: &mut usize,
    collect: bool,
    leaves: &mut Vec<Leaf>,
) -> Option<()> {
    if depth > MAX_DEPTH || *budget == 0 {
        return None;
    }
    *budget -= 1;

    let base = *pos;
    if base.checked_add(TERRAIN_NODE_LEN)? > dat.len() {
        return None;
    }
    let _version = rd_i16(dat, base)?;
    let min_x = rd_f32(dat, base + 4)? as f64;
    let min_y = rd_f32(dat, base + 8)? as f64;
    let max_x = rd_f32(dat, base + 0x10)? as f64;
    let max_y = rd_f32(dat, base + 0x14)? as f64;
    let offset = rd_f32(dat, base + 0x1C)? as f64;
    let range = rd_f32(dat, base + 0x20)? as f64;
    let n_size = rd_i32(dat, base + 0x24)?;
    let n_surf = rd_i32(dat, base + 0x28)?;
    if n_size < 0 || n_surf < 0 {
        return None;
    }
    let size = n_size as usize;
    let nsurf = n_surf as usize;
    if size > (1 << 20) || nsurf > dat.len() {
        return None;
    }

    let mut p = base + TERRAIN_NODE_LEN;

    if size > 0 {
        let count = size.checked_mul(size)?;
        let bytes = count.checked_mul(2)?;
        if p.checked_add(bytes)? > dat.len() {
            return None;
        }
        // pGeomErrors follow the (4-byte aligned) raw field, then pTypes.
        let raw_off = p;
        let geom_off = align4(raw_off.checked_add(bytes)?);
        let types_off = geom_off.checked_add(4 * geom)?;
        let end = align4(types_off.checked_add(nsurf)?);
        if end > dat.len() {
            return None;
        }
        if collect {
            let mut raw = Vec::with_capacity(count);
            for k in 0..count {
                raw.push(rd_u16(dat, raw_off + 2 * k)?);
            }
            let mut types = Vec::with_capacity(nsurf);
            for k in 0..nsurf {
                types.push(*dat.get(types_off + k)?);
            }
            leaves.push(Leaf {
                min_x,
                min_y,
                max_x,
                max_y,
                offset,
                range,
                size,
                raw,
                types,
            });
        }
        p = end;
    } else {
        // m_pGeomErrors[bit_shift], then pTypes[nSurfaceTypesNum], both 4-byte aligned.
        p = p.checked_add(4 * geom)?;
        p = align4(p.checked_add(nsurf)?);
        if p > dat.len() {
            return None;
        }
    }

    if size == 0 {
        for _ in 0..4 {
            walk_node(dat, &mut p, geom, depth + 1, budget, collect, leaves)?;
        }
    }

    *pos = p;
    Some(())
}

/// Bilinear height at world `(x, y)` inside one leaf.
fn height_at(lf: &Leaf, x: f64, y: f64) -> Option<f64> {
    let ns = lf.size;
    if ns == 0 {
        return Some(lf.offset);
    }
    let fx = if lf.max_x <= lf.min_x {
        0.0
    } else {
        (x - lf.min_x) / (lf.max_x - lf.min_x) * (ns - 1) as f64
    };
    let fy = if lf.max_y <= lf.min_y {
        0.0
    } else {
        (y - lf.min_y) / (lf.max_y - lf.min_y) * (ns - 1) as f64
    };
    let i0 = (fx.floor() as isize).clamp(0, ns as isize - 1) as usize;
    let j0 = (fy.floor() as isize).clamp(0, ns as isize - 1) as usize;
    let i1 = (i0 + 1).min(ns - 1);
    let j1 = (j0 + 1).min(ns - 1);
    let tx = fx - i0 as f64;
    let ty = fy - j0 as f64;

    let h = |i: usize, j: usize| -> Option<f64> {
        lf.raw
            .get(i * ns + j)
            .map(|&r| lf.offset + ((r & 0xFFF0) as f64) * lf.range)
    };
    let h00 = h(i0, j0)?;
    let h10 = h(i1, j0)?;
    let h01 = h(i0, j1)?;
    let h11 = h(i1, j1)?;
    let top = h00 * (1.0 - tx) + h10 * tx;
    let bot = h01 * (1.0 - tx) + h11 * tx;
    Some(top * (1.0 - ty) + bot * ty)
}

/// Map a raw sample's low-nibble local palette index to its global
/// `SurfaceTypeID`. Local `14` (undefined) and `15` (hole), and any index outside
/// the node palette, map to [`SURFACE_UNDEFINED`].
fn global_surface(lf: &Leaf, local: u8) -> u8 {
    match local {
        14 | 15 => SURFACE_UNDEFINED,
        _ => lf.types.get(local as usize).copied().unwrap_or(SURFACE_UNDEFINED),
    }
}

/// Nearest-sample global surface id at world `(x, y)` inside one leaf.
///
/// Unlike [`height_at`] the surface is categorical, so the nearest lattice point
/// (not a bilinear blend) is used; `raw` is indexed `i * size + j` with `i` from
/// `x` and `j` from `y`, exactly as in [`height_at`].
fn surface_global_at(lf: &Leaf, x: f64, y: f64) -> Option<u8> {
    let ns = lf.size;
    if ns == 0 {
        return None;
    }
    let fx = if lf.max_x <= lf.min_x {
        0.0
    } else {
        (x - lf.min_x) / (lf.max_x - lf.min_x) * (ns - 1) as f64
    };
    let fy = if lf.max_y <= lf.min_y {
        0.0
    } else {
        (y - lf.min_y) / (lf.max_y - lf.min_y) * (ns - 1) as f64
    };
    let i = (fx.round() as isize).clamp(0, ns as isize - 1) as usize;
    let j = (fy.round() as isize).clamp(0, ns as isize - 1) as usize;
    let raw = *lf.raw.get(i * ns + j)?;
    Some(global_surface(lf, (raw & 0x000F) as u8))
}

fn reconstruct(hdr: &Header, leaves: &[Leaf]) -> Option<Terrain> {
    let grid = hdr.grid;
    if grid == 0 {
        return None;
    }
    let mut index: HashMap<(i32, i32), usize> = HashMap::with_capacity(leaves.len() * 2);
    for (i, lf) in leaves.iter().enumerate() {
        if lf.min_x < 0.0 || lf.min_y < 0.0 {
            continue;
        }
        let sx = (lf.min_x / hdr.sector).round() as i32;
        let sy = (lf.min_y / hdr.sector).round() as i32;
        index.insert((sx, sy), i);
    }
    if index.is_empty() {
        return None;
    }

    let cell = hdr.world / grid as f64;
    let mut heights = vec![0f32; grid * grid];
    let mut surfaces = vec![SURFACE_UNDEFINED; grid * grid];
    let mut hmin = f64::INFINITY;
    let mut hmax = f64::NEG_INFINITY;
    let mut count = 0usize;
    let mut missing = 0usize;

    for j in 0..grid {
        let y = (j as f64 + 0.5) * cell;
        let sy = (y / hdr.sector).floor() as i32;
        for i in 0..grid {
            let x = (i as f64 + 0.5) * cell;
            let sx = (x / hdr.sector).floor() as i32;
            let leaf = index.get(&(sx, sy)).copied();
            let h = leaf.and_then(|li| height_at(&leaves[li], x, y));
            match h {
                Some(hh) if hh.is_finite() => {
                    heights[j * grid + i] = hh as f32;
                    if let Some(li) = leaf {
                        surfaces[j * grid + i] =
                            surface_global_at(&leaves[li], x, y).unwrap_or(SURFACE_UNDEFINED);
                    }
                    hmin = hmin.min(hh);
                    hmax = hmax.max(hh);
                    count += 1;
                }
                _ => {
                    missing += 1;
                    heights[j * grid + i] = 0.0;
                }
            }
        }
    }
    if count == 0 || missing > 0 {
        return None;
    }
    Some(Terrain {
        samples: grid,
        spacing: cell as f32,
        world_min: [0.0, 0.0, hmin as f32],
        heights,
        surfaces,
        min: hmin as f32,
        max: hmax as f32,
    })
}

/// Reconstruct the full-resolution heightmap from a `terrain.dat` image.
///
/// Returns `None` for malformed input (truncated header/tree, non-power-of-two
/// sector units, big-endian variant, or an incomplete grid).
pub fn decode_terrain(dat: &[u8]) -> Option<Terrain> {
    let hdr = Header::parse(dat)?;
    let (_, leaves) = walk_terrain(dat, &hdr, true)?;
    reconstruct(&hdr, &leaves)
}

impl Terrain {
    /// Global surface id nearest world `(x, y)`, or [`SURFACE_UNDEFINED`] for an
    /// undefined/hole sample (or when the point is out of range/malformed).
    ///
    /// Uses the same nearest-sample convention as the app's `height_at`:
    /// `i = round(coord / spacing)` clamped to the grid.
    pub fn surface_at(&self, x: f32, y: f32) -> u8 {
        let cells = self.samples.saturating_mul(self.samples);
        if self.samples == 0 || !(self.spacing > 0.0) || self.surfaces.len() < cells {
            return SURFACE_UNDEFINED;
        }
        let ix = ((x / self.spacing).round() as isize).clamp(0, self.samples as isize - 1) as usize;
        let iy = ((y / self.spacing).round() as isize).clamp(0, self.samples as isize - 1) as usize;
        self.surfaces[iy * self.samples + ix]
    }
}

// ---------------------------------------------------------------------------
// mesh generation
// ---------------------------------------------------------------------------

/// Build a triangle mesh from the heightmap, keeping every `stride`-th sample in
/// each direction (`stride == 0` is treated as `1`).
///
/// Coordinates are Z-up right-handed as described in the module docs: vertex
/// `[x, y, z]`, `z` = height. Winding is counter-clockwise from `+Z` (up-facing).
/// Vertices are laid out row-major with `x` fastest.
pub fn build_mesh(t: &Terrain, stride: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
    let stride = stride.max(1);
    let s = t.samples;
    if s == 0 || t.heights.len() < s.saturating_mul(s) {
        return (Vec::new(), Vec::new());
    }
    let nv = (s + stride - 1) / stride;
    if nv == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut positions = Vec::with_capacity(nv.saturating_mul(nv));
    for gj in 0..nv {
        let sj = (gj * stride).min(s - 1);
        for gi in 0..nv {
            let si = (gi * stride).min(s - 1);
            let x = t.world_min[0] + (si as f32 + 0.5) * t.spacing;
            let y = t.world_min[1] + (sj as f32 + 0.5) * t.spacing;
            let z = t.heights[sj * s + si];
            positions.push([x, y, z]);
        }
    }

    let quads = nv.saturating_sub(1);
    let mut indices = Vec::with_capacity(quads.saturating_mul(quads).saturating_mul(6));
    for gj in 0..quads {
        for gi in 0..quads {
            let a = (gj * nv + gi) as u32;
            let b = (gj * nv + gi + 1) as u32;
            let c = ((gj + 1) * nv + gi + 1) as u32;
            let d = ((gj + 1) * nv + gi) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    (positions, indices)
}

// ---------------------------------------------------------------------------
// static-object octree
// ---------------------------------------------------------------------------

/// Best-effort list of placed `Brush` render nodes with their transforms.
///
/// The entire octree is traversed, but only `eERType_Brush` (1) entries are
/// emitted; other render-node chunks are sized so the walk stays exact, and an
/// unrecognised type ends that node's block (its remaining bytes are skipped via
/// the stored block size). Returns an empty vector on malformed input.
pub fn decode_static_objects(dat: &[u8]) -> Vec<StaticObject> {
    let mut out = Vec::new();
    let Some(hdr) = Header::parse(dat) else {
        return out;
    };
    let Some((start, _)) = walk_terrain(dat, &hdr, false) else {
        return out;
    };
    let mut pos = start;
    let mut budget = MAX_OBJECT_NODES;
    let _ = walk_object_node(dat, &mut pos, 0, &mut budget, &mut out);
    out
}

fn walk_object_node(
    dat: &[u8],
    pos: &mut usize,
    depth: usize,
    budget: &mut usize,
    out: &mut Vec<StaticObject>,
) -> Option<()> {
    if depth > MAX_DEPTH || *budget == 0 {
        return None;
    }
    *budget -= 1;

    let base = *pos;
    if base.checked_add(OBJECT_NODE_LEN)? > dat.len() {
        return None;
    }
    let _version = rd_i16(dat, base)?;
    let childs = rd_i16(dat, base + 2)? as u16;
    let block_size = rd_i32(dat, base + 0x1C)?;
    if block_size < 0 {
        return None;
    }
    let block_size = block_size as usize;
    let body = base + OBJECT_NODE_LEN;
    let block_end = body.checked_add(block_size)?;
    if block_end > dat.len() {
        return None;
    }

    if block_size > 0 {
        parse_object_block(dat, body, block_end, out);
    }

    let mut p = block_end;
    for bit in 0..8 {
        if childs & (1 << bit) != 0 {
            walk_object_node(dat, &mut p, depth + 1, budget, out)?;
        }
    }
    *pos = p;
    Some(())
}

/// Walk one type-sorted object block. Mirrors `terrain_inspect.py::_parse_block`.
fn parse_object_block(dat: &[u8], start: usize, end: usize, out: &mut Vec<StaticObject>) {
    let mut p = start;
    while p.checked_add(4).is_some_and(|e| e <= end) {
        let Some(etype) = rd_i32(dat, p) else { break };
        let q = p + 4;
        match etype {
            1 => {
                // Brush: SRenderNodeChunk (36) + Matrix34 (48) + 4 + 8.
                if q.checked_add(96).is_none_or(|e| e > end) {
                    break;
                }
                let proto = rd_u16(dat, q + 32).unwrap_or(0);
                if let Some(matrix) = read_matrix34(dat, q + 36) {
                    out.push(brush_transform(proto, matrix));
                }
                p = q + 96;
            }
            2 => {
                // Vegetation: 36 + 12 + 4 + 4.
                if q.checked_add(56).is_none_or(|e| e > end) {
                    break;
                }
                p = q + 56;
            }
            7 => {
                // Decal: 112 bytes.
                if q.checked_add(112).is_none_or(|e| e > end) {
                    break;
                }
                p = q + 112;
            }
            11 => {
                // Road: fixed 64 + nVerts Vec3.
                if q.checked_add(40).is_none_or(|e| e > end) {
                    break;
                }
                let Some(nv) = rd_i32(dat, q + 36) else { break };
                if nv < 0 {
                    break;
                }
                let Some(adv) = (nv as usize).checked_mul(12).and_then(|b| 64usize.checked_add(b))
                else {
                    break;
                };
                if q.checked_add(adv).is_none_or(|e| e > end) {
                    break;
                }
                p = q + adv;
            }
            12 => {
                // DistanceCloud: 64 bytes.
                if q.checked_add(64).is_none_or(|e| e > end) {
                    break;
                }
                p = q + 64;
            }
            23 => {
                // MergedMesh: fixed 40 + nGroups * 8.
                if q.checked_add(40).is_none_or(|e| e > end) {
                    break;
                }
                let Some(ng) = rd_u32(dat, q + 36) else { break };
                let Some(adv) =
                    (ng as usize).checked_mul(8).and_then(|b| 64usize.checked_add(b))
                else {
                    break;
                };
                if q.checked_add(adv).is_none_or(|e| e > end) {
                    break;
                }
                p = q + adv;
            }
            _ => break,
        }
    }
}

fn read_matrix34(dat: &[u8], o: usize) -> Option<[f32; 12]> {
    let mut m = [0f32; 12];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = rd_f32(dat, o + i * 4)?;
    }
    Some(m)
}

/// `Matrix34` is row-major `[r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz]`.
fn brush_transform(prototype: u16, m: [f32; 12]) -> StaticObject {
    let pos = [m[3], m[7], m[11]];
    let rows = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
    let norm = |v: [f32; 3]| -> f64 {
        ((v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2)).sqrt()
    };
    let sx = norm(rows[0]);
    let sy = norm(rows[1]);
    let sz = norm(rows[2]);
    let scale = ((sx + sy + sz) / 3.0) as f32;

    let unit = |v: [f32; 3]| -> [f64; 3] {
        let l = norm(v);
        if l > 1e-12 {
            [v[0] as f64 / l, v[1] as f64 / l, v[2] as f64 / l]
        } else {
            [0.0, 0.0, 0.0]
        }
    };
    let r0 = unit(rows[0]);
    let r1 = unit(rows[1]);
    let r2 = unit(rows[2]);

    let tr = r0[0] + r1[1] + r2[2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        if s <= 1e-12 {
            [0.0, 0.0, 0.0, 1.0]
        } else {
            [
                (r1[2] - r2[1]) / s,
                (r2[0] - r0[2]) / s,
                (r0[1] - r1[0]) / s,
                0.25 * s,
            ]
        }
    } else if r0[0] > r1[1] && r0[0] > r2[2] {
        let s = (1.0 + r0[0] - r1[1] - r2[2]).sqrt() * 2.0;
        if s <= 1e-12 {
            [0.0, 0.0, 0.0, 1.0]
        } else {
            [
                0.25 * s,
                (r1[0] + r0[1]) / s,
                (r2[0] + r0[2]) / s,
                (r1[2] - r2[1]) / s,
            ]
        }
    } else if r1[1] > r2[2] {
        let s = (1.0 + r1[1] - r0[0] - r2[2]).sqrt() * 2.0;
        if s <= 1e-12 {
            [0.0, 0.0, 0.0, 1.0]
        } else {
            [
                (r1[0] + r0[1]) / s,
                0.25 * s,
                (r2[1] + r1[2]) / s,
                (r2[0] - r0[2]) / s,
            ]
        }
    } else {
        let s = (1.0 + r2[2] - r0[0] - r1[1]).sqrt() * 2.0;
        if s <= 1e-12 {
            [0.0, 0.0, 0.0, 1.0]
        } else {
            [
                (r2[0] + r0[2]) / s,
                (r2[1] + r1[2]) / s,
                0.25 * s,
                (r0[1] - r1[0]) / s,
            ]
        }
    };

    StaticObject {
        prototype,
        pos,
        rot: [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32],
        scale,
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CLASS3: &str = "/home/battlestation/Desktop/sod-re/extracted-game/level/terrain/terrain.dat";
    const BELTWAY: &str =
        "/home/battlestation/Desktop/sod-re/extracted-game/BeltwayU/Game/levels/C3_beltway/terrain/terrain.dat";

    fn load(path: &str) -> Option<Vec<u8>> {
        std::fs::read(path).ok()
    }

    fn mean(h: &[f32]) -> f64 {
        if h.is_empty() {
            return 0.0;
        }
        h.iter().map(|&x| x as f64).sum::<f64>() / h.len() as f64
    }

    #[test]
    fn malformed_is_total() {
        assert!(decode_terrain(&[]).is_none());
        assert!(decode_terrain(&[0u8; 10]).is_none());
        assert!(decode_terrain(&[0u8; 64]).is_none());
        assert!(decode_static_objects(&[]).is_empty());
        assert!(decode_static_objects(&[0xffu8; 40]).is_empty());
    }

    #[test]
    fn synthetic_mesh() {
        let heights = vec![
            0., 1., 2., 3., //
            1., 2., 3., 4., //
            2., 3., 4., 5., //
            3., 4., 5., 6.,
        ];
        let t = Terrain {
            samples: 4,
            spacing: 1.0,
            world_min: [0.0, 0.0, 0.0],
            heights,
            surfaces: vec![SURFACE_UNDEFINED; 16],
            min: 0.0,
            max: 6.0,
        };
        let (v, i) = build_mesh(&t, 1);
        assert_eq!(v.len(), 16);
        assert_eq!(i.len(), 3 * 3 * 6);
        // up-facing winding: (b-a) x (c-a) must have +Z
        let a = v[i[0] as usize];
        let b = v[i[1] as usize];
        let c = v[i[2] as usize];
        let nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        assert!(nz > 0.0, "expected up-facing winding, got {nz}");
        let (v4, _) = build_mesh(&t, 4);
        assert_eq!(v4.len(), 1);
    }

    #[test]
    fn class3() {
        let Some(dat) = load(CLASS3) else { return };
        let t = decode_terrain(&dat).expect("Class3 decode");
        assert_eq!(t.samples, 1024);
        assert_eq!(t.heights.len(), 1024 * 1024);
        assert!((t.spacing - 4.0).abs() < 1e-4, "spacing {}", t.spacing);
        assert!((t.min - 21.3242).abs() < 0.05, "min {}", t.min);
        assert!((t.max - 322.5).abs() < 0.05, "max {}", t.max);
        let m = mean(&t.heights);
        assert!((m - 140.889).abs() < 0.05, "mean {m}");
        assert_eq!(t.world_min, [0.0, 0.0, t.min]);

        let (v1, i1) = build_mesh(&t, 1);
        assert_eq!(v1.len(), 1024 * 1024);
        assert_eq!(i1.len(), 1023 * 1023 * 6);
        let (v4, i4) = build_mesh(&t, 4);
        assert_eq!(v4.len(), 256 * 256);
        assert_eq!(i4.len(), 255 * 255 * 6);

        let objs = decode_static_objects(&dat);
        assert_eq!(objs.len(), 36_662);

        println!(
            "Class3: samples={} spacing={:.3} min={:.4} max={:.4} mean={:.4} verts1={} verts4={} objects={}",
            t.samples,
            t.spacing,
            t.min,
            t.max,
            m,
            v1.len(),
            v4.len(),
            objs.len()
        );
    }

    #[test]
    fn beltway() {
        let Some(dat) = load(BELTWAY) else { return };
        let t = decode_terrain(&dat).expect("Beltway decode");
        assert_eq!(t.samples, 1024);
        assert!((t.min - 30.0).abs() < 0.05, "min {}", t.min);
        assert!((t.max - 136.03125).abs() < 0.05, "max {}", t.max);
        let m = mean(&t.heights);
        assert!((m - 56.0174).abs() < 0.05, "mean {m}");

        let (v1, i1) = build_mesh(&t, 1);
        assert_eq!(v1.len(), 1024 * 1024);
        assert_eq!(i1.len(), 1023 * 1023 * 6);
        let (v4, _) = build_mesh(&t, 4);
        assert_eq!(v4.len(), 256 * 256);

        let objs = decode_static_objects(&dat);
        assert_eq!(objs.len(), 16_444);

        println!(
            "Beltway: samples={} spacing={:.3} min={:.4} max={:.4} mean={:.4} verts1={} verts4={} objects={}",
            t.samples,
            t.spacing,
            t.min,
            t.max,
            m,
            v1.len(),
            v4.len(),
            objs.len()
        );
    }

    // -----------------------------------------------------------------------
    // surface-type tests
    // -----------------------------------------------------------------------

    /// Build a minimal well-formed `terrain.dat` holding a single leaf node of
    /// `size x size` samples over a `world_len` metre square, with `types` as its
    /// `pTypes` palette and `raw` as its raw height/surface samples.
    fn synthetic_dat(size: usize, world_len: f32, types: &[u8], raw: &[u16]) -> Vec<u8> {
        assert_eq!(raw.len(), size * size);
        let mut dat = vec![0u8; 0x2C];
        // Header: unit = 1 m, so world = sector = world_len and geom_shift = 2.
        dat[0x04..0x08].copy_from_slice(&1234i32.to_le_bytes()); // nChunkSize (> 0)
        dat[0x08..0x0C].copy_from_slice(&(world_len as i32).to_le_bytes()); // nHeightMapSize
        dat[0x0C..0x10].copy_from_slice(&1i32.to_le_bytes()); // nUnitSize
        dat[0x10..0x14].copy_from_slice(&(world_len as i32).to_le_bytes()); // nSectorSize
        dat[0x14..0x18].copy_from_slice(&1i32.to_le_bytes()); // nSectorsTableSize
        // Prototype table counts stay zero, so terrain data begins at 0x2C.
        dat.extend_from_slice(&6i16.to_le_bytes()); // nChunkVersion
        dat.extend_from_slice(&0i16.to_le_bytes()); // bHasHoles
        for v in [0.0f32, 0.0, 0.0, world_len, world_len, 0.0] {
            dat.extend_from_slice(&v.to_le_bytes()); // AABB
        }
        dat.extend_from_slice(&0.0f32.to_le_bytes()); // fOffset
        dat.extend_from_slice(&1.0f32.to_le_bytes()); // fRange
        dat.extend_from_slice(&(size as i32).to_le_bytes()); // nSize
        dat.extend_from_slice(&(types.len() as i32).to_le_bytes()); // nSurfaceTypesNum
        for &r in raw {
            dat.extend_from_slice(&r.to_le_bytes());
        }
        while dat.len() % 4 != 0 {
            dat.push(0);
        }
        for _ in 0..2 {
            dat.extend_from_slice(&0.0f32.to_le_bytes()); // pGeomErrors[2]
        }
        dat.extend_from_slice(types);
        while dat.len() % 4 != 0 {
            dat.push(0);
        }
        dat
    }

    #[test]
    fn synthetic_leaf_palette_mapping() {
        let lf = Leaf {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 4.0,
            max_y: 4.0,
            offset: 0.0,
            range: 1.0,
            size: 3,
            // raw is indexed `i * size + j`, `i` from x, `j` from y.
            raw: vec![0x0002, 0x000E, 0x000F, 0x0001, 0x0007, 0, 0, 0, 0],
            types: vec![10, 11, 12, 13],
        };
        assert_eq!(surface_global_at(&lf, 0.0, 0.0), Some(12)); // raw[0], local 2
        assert_eq!(surface_global_at(&lf, 0.0, 2.0), Some(SURFACE_UNDEFINED)); // local 14
        assert_eq!(surface_global_at(&lf, 0.0, 4.0), Some(SURFACE_UNDEFINED)); // local 15
        assert_eq!(surface_global_at(&lf, 2.0, 0.0), Some(11)); // raw[3], local 1
        assert_eq!(surface_global_at(&lf, 2.0, 2.0), Some(SURFACE_UNDEFINED)); // local 7 OOR
    }

    #[test]
    fn synthetic_decode_surfaces() {
        let types = [10u8, 11, 12, 13];
        // Every sample's low nibble is 1 => global types[1] == 11.
        let raw: Vec<u16> = (0..9).map(|k| ((k as u16) << 4) | 1).collect();
        let dat = synthetic_dat(3, 4.0, &types, &raw);
        let t = decode_terrain(&dat).expect("synthetic decode");
        assert_eq!(t.surfaces.len(), t.heights.len());
        assert_eq!(t.surfaces.len(), t.samples * t.samples);
        assert!(t.surfaces.iter().all(|&s| s == 11), "expected all 11");
        assert_eq!(t.surface_at(0.0, 0.0), 11);
        assert_eq!(t.surface_at(2.0, 2.0), 11);
        assert_eq!(t.surface_at(-100.0, -100.0), 11); // clamped, no panic
    }

    #[test]
    fn class3_surfaces() {
        let Some(dat) = load(CLASS3) else { return };
        let t = decode_terrain(&dat).expect("Class3 decode");
        assert_eq!(t.surfaces.len(), t.heights.len());
        assert_eq!(t.surfaces.len(), 1024 * 1024);

        let mut hist = [0u64; 256];
        for &s in &t.surfaces {
            hist[s as usize] += 1;
        }
        let defined: u64 = (0..256)
            .filter(|&s| s != SURFACE_UNDEFINED as usize)
            .map(|s| hist[s])
            .sum();
        let ids_ge_32: u64 = (32..SURFACE_UNDEFINED as usize).map(|s| hist[s]).sum();
        let distinct = (0..SURFACE_UNDEFINED as usize)
            .filter(|&s| hist[s] > 0)
            .count();
        assert!(defined > 0, "no defined surfaces");
        assert!(
            (ids_ge_32 as f64) < (defined as f64) * 0.1,
            "most ids should be < 32; {ids_ge_32}/{defined} are >= 32"
        );
        assert!(distinct >= 2, "expected >= 2 distinct surface ids, got {distinct}");
        println!(
            "Class3 surfaces: distinct={distinct} defined={defined} undefined={} ids>=32={ids_ge_32}",
            hist[SURFACE_UNDEFINED as usize]
        );
    }
}
