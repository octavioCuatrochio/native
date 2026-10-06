//! Static architecture: Table B prototypes + placed brushes from `terrain.dat`.
//!
//! This decodes the *placed static architecture* (buildings, walls, rocks,
//! fences, …) that CryEngine stores in the `COctreeNode` static-object octree
//! at the tail of `terrain/terrain.dat`. Each object block is a type-sorted list
//! of render nodes; the `eERType_Brush` entries carry a 16-bit
//! `m_nObjectTypeIndex` into **Table B** (the `IStatObj` prototype list, whose
//! records are `0x100`-byte NUL-padded `.cgf` paths) plus a row-major `Matrix34`
//! transform.
//!
//! Format references (see `notes/terrain-decode.md` §1, §2, §4):
//!
//! * `STerrainChunkHeader` + `STerrainInfo` — 32 bytes at offset 0; bit0 of
//!   `nFlags` marks the little/big-endian variant. The tables follow directly.
//! * Table A (StatInstGroup, stride `0x150`), Table B (IStatObj, stride `0x100`)
//!   and Table C (materials, stride `0x100`) are each `u32 count` + `count`
//!   fixed-stride records. Counts/offsets are derived from the header, not
//!   hard-coded.
//! * `CTerrainNode` quadtree — walked only to find where the object octree
//!   begins (`nSize == 0` ⇒ 4 children; `0xDE` 4-byte-aligned leaf tails).
//! * `SOcTreeNodeChunk` — 32-byte node header: `int16 nChunkVersion`,
//!   `int16 ucChildsMask`, `AABB nodeBox`, `int32 nObjectsBlockSize`; then the
//!   object block, then the children selected by `ucChildsMask`.
//!
//! The module follows the rest of `sod-core`: every entry point is total and a
//! malformed image yields `None`/an empty vector rather than a panic.
//!
//! # Coordinates
//!
//! Vectors are CryEngine-native **Z-up** right-handed: `pos = [x, y, z]` with
//! `x`/`y` in `[0, 4096]` metres and `z` the height. `rot` is the orientation
//! quaternion `[x, y, z, w]` (scalar last) obtained by orthonormalising the
//! `Matrix34` basis; `scale` is the mean of its three row norms.

/// `STerrainChunkHeader` + `STerrainInfo` length.
const HEADER_LEN: usize = 0x20;
/// `STerrainNodeChunk` length.
const TERRAIN_NODE_LEN: usize = 44;
/// `SOcTreeNodeChunk` length.
const OBJECT_NODE_LEN: usize = 32;
/// Table A record stride.
const STRIDE_A: usize = 0x150;
/// Table B record stride (`IStatObj` prototype paths).
const STRIDE_B: usize = 0x100;
/// Table C record stride.
const STRIDE_C: usize = 0x100;
/// Hard recursion-depth guard for both trees.
const MAX_DEPTH: usize = 64;
/// Hard node budgets, so malformed data cannot run unbounded.
const MAX_TERRAIN_NODES: usize = 5_000_000;
const MAX_OBJECT_NODES: usize = 5_000_000;

/// One placed static `Brush` render node resolved against Table B.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedBrush {
    /// Table B model path, forward-slash normalised (e.g.
    /// `objects/landscape/hill/.../rock_large_01_new.cgf`).
    pub model: String,
    /// World translation `(x, y, z)` in metres.
    pub pos: [f32; 3],
    /// Orientation quaternion `[x, y, z, w]` (scalar last).
    pub rot: [f32; 4],
    /// Best-effort uniform scale (mean of the three `Matrix34` row norms).
    pub scale: f32,
}

/// Parsed `STerrainChunkHeader` + `STerrainInfo`, exposed for introspection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header {
    /// `.dat` chunk version (`0x18` on SoD).
    pub version: i8,
    /// `nDummy`.
    pub dummy: i8,
    /// `SERIALIZATION_FLAG_*` bits; bit0 is `BIG_ENDIAN`.
    pub flags: i8,
    /// `TCH_FLAG2_*` bits.
    pub flags2: i8,
    /// `nChunkSize` (equals the file size).
    pub chunk_size: i32,
    /// `nHeightMapSize_InUnits`.
    pub height_map_size: i32,
    /// `nUnitSize_InMeters`.
    pub unit_size: i32,
    /// `nSectorSize_InMeters`.
    pub sector_size: i32,
    /// `nSectorsTableSize_InSectors`.
    pub sectors_table_size: i32,
    /// `fHeightmapZRatio` (legacy global Z scale).
    pub heightmap_z_ratio: f32,
    /// `fOceanWaterLevel`.
    pub ocean_water_level: f32,
    /// True when bit0 of [`flags`](Self::flags) is set (xbox360 variant).
    pub big_endian: bool,
}

impl Header {
    /// Parse the 32-byte header; `None` if the image is too short.
    ///
    /// Big-endian images parse (the flag is reported) but every multi-byte field
    /// would be wrong, so the decoders below reject them explicitly.
    pub fn parse(dat: &[u8]) -> Option<Header> {
        if dat.len() < HEADER_LEN {
            return None;
        }
        let flags = *dat.get(2)? as i8;
        Some(Header {
            version: *dat.get(0)? as i8,
            dummy: *dat.get(1)? as i8,
            flags,
            flags2: *dat.get(3)? as i8,
            chunk_size: rd_i32(dat, 4)?,
            height_map_size: rd_i32(dat, 8)?,
            unit_size: rd_i32(dat, 0x0C)?,
            sector_size: rd_i32(dat, 0x10)?,
            sectors_table_size: rd_i32(dat, 0x14)?,
            heightmap_z_ratio: rd_f32(dat, 0x18)?,
            ocean_water_level: rd_f32(dat, 0x1C)?,
            big_endian: (flags as u8) & 1 != 0,
        })
    }

    /// World extent in metres (`nHeightMapSize * nUnitSize`).
    pub fn world_size(&self) -> f64 {
        self.height_map_size as f64 * self.unit_size as f64
    }
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
// prototype tables
// ---------------------------------------------------------------------------

/// Table B location, derived from the header's Table A count.
struct TableB {
    offset: usize,
    count: usize,
}

fn table_b(dat: &[u8]) -> Option<TableB> {
    let count_a = rd_u32(dat, 0x20)? as usize;
    let off_b_count = 0x24usize.checked_add(count_a.checked_mul(STRIDE_A)?)?;
    let count = rd_u32(dat, off_b_count)? as usize;
    Some(TableB {
        offset: off_b_count.checked_add(4)?,
        count,
    })
}

/// Offset of the terrain quadtree, immediately after the three prototype tables.
fn data_offset(dat: &[u8]) -> Option<usize> {
    let count_a = rd_u32(dat, 0x20)? as usize;
    let off_b_count = 0x24usize.checked_add(count_a.checked_mul(STRIDE_A)?)?;
    let count_b = rd_u32(dat, off_b_count)? as usize;
    let off_c_count = off_b_count
        .checked_add(4)?
        .checked_add(count_b.checked_mul(STRIDE_B)?)?;
    let count_c = rd_u32(dat, off_c_count)? as usize;
    let off_data = off_c_count
        .checked_add(4)?
        .checked_add(count_c.checked_mul(STRIDE_C)?)?;
    if off_data > dat.len() {
        return None;
    }
    Some(off_data)
}

/// Decode one fixed-stride, NUL-padded path field into a forward-slash string.
///
/// Bytes are treated as Latin-1 (one `char` per byte) exactly like the reference
/// reader; on this data the paths are plain ASCII.
fn decode_path(slot: &[u8]) -> String {
    let end = slot.iter().position(|&b| b == 0).unwrap_or(slot.len());
    let mut s = String::with_capacity(end);
    for &b in &slot[..end] {
        s.push(if b == b'\\' { '/' } else { b as char });
    }
    s
}

/// Table B object-model paths, forward-slash normalised, in table order.
///
/// `Class3` yields exactly 2131 entries (e.g. `guardrail_new`,
/// `cladding_crag_07`, `rock_large_01_new`). Returns an empty vector for a
/// malformed/truncated image or the untested big-endian variant.
pub fn prototypes(dat: &[u8]) -> Vec<String> {
    let Some(hdr) = Header::parse(dat) else {
        return Vec::new();
    };
    if hdr.big_endian {
        return Vec::new();
    }
    let Some(t) = table_b(dat) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(t.count.min(1 << 20));
    for i in 0..t.count {
        let Some(o) = i
            .checked_mul(STRIDE_B)
            .and_then(|d| t.offset.checked_add(d))
        else {
            break;
        };
        let Some(slot) = o.checked_add(STRIDE_B).and_then(|e| dat.get(o..e)) else {
            break;
        };
        out.push(decode_path(slot));
    }
    out
}

// ---------------------------------------------------------------------------
// terrain quadtree (walked only to locate the object octree)
// ---------------------------------------------------------------------------

fn terrain_end(dat: &[u8], hdr: &Header) -> Option<usize> {
    if hdr.big_endian || hdr.unit_size <= 0 || hdr.sector_size <= 0 {
        return None;
    }
    if hdr.sector_size % hdr.unit_size != 0 {
        return None;
    }
    let units = (hdr.sector_size / hdr.unit_size) as u32;
    if !units.is_power_of_two() {
        return None;
    }
    let geom = units.trailing_zeros() as usize;
    let mut pos = data_offset(dat)?;
    let mut budget = MAX_TERRAIN_NODES;
    walk_terrain(dat, &mut pos, geom, 0, &mut budget)?;
    Some(pos)
}

fn walk_terrain(
    dat: &[u8],
    pos: &mut usize,
    geom: usize,
    depth: usize,
    budget: &mut usize,
) -> Option<()> {
    if depth > MAX_DEPTH || *budget == 0 {
        return None;
    }
    *budget -= 1;

    let base = *pos;
    if base.checked_add(TERRAIN_NODE_LEN)? > dat.len() {
        return None;
    }
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
        let bytes = size.checked_mul(size)?.checked_mul(2)?;
        p = align4(p.checked_add(bytes)?);
    }
    // m_pGeomErrors[bit_shift] then pTypes[nSurfaceTypesNum], both 4-byte aligned.
    p = p.checked_add(4 * geom)?;
    p = align4(p.checked_add(nsurf)?);
    if p > dat.len() {
        return None;
    }

    if size == 0 {
        for _ in 0..4 {
            walk_terrain(dat, &mut p, geom, depth + 1, budget)?;
        }
    }
    *pos = p;
    Some(())
}

// ---------------------------------------------------------------------------
// static-object octree
// ---------------------------------------------------------------------------

/// All placed `Brush` render nodes, resolved to their Table B model path.
///
/// The whole octree is traversed (so block boundaries stay exact); only
/// `eERType_Brush` (1) entries are emitted, and a brush whose
/// `m_nObjectTypeIndex` is out of range is skipped. Returns an empty vector on
/// malformed input.
pub fn decode_brushes(dat: &[u8]) -> Vec<PlacedBrush> {
    let mut out = Vec::new();
    let Some(hdr) = Header::parse(dat) else {
        return out;
    };
    if hdr.big_endian {
        return out;
    }
    let names = prototypes(dat);
    if names.is_empty() {
        return out;
    }
    let Some(start) = terrain_end(dat, &hdr) else {
        return out;
    };
    let mut pos = start;
    let mut budget = MAX_OBJECT_NODES;
    let _ = walk_objects(dat, &mut pos, 0, &mut budget, &names, &mut out);
    out
}

/// Brushes whose **horizontal** position lies within `radius` of `center`.
///
/// The query is 2-D over the ground plane. Following the documented format the
/// world is Z-up, so the horizontal axes are `X`/`Y` (the Z-up convention; the
/// task's "XZ" shorthand names the ground plane). `radius < 0` yields nothing.
pub fn brushes_in_region(dat: &[u8], center: [f32; 3], radius: f32) -> Vec<PlacedBrush> {
    let r2 = (radius as f64) * (radius as f64);
    decode_brushes(dat)
        .into_iter()
        .filter(|b| {
            let dx = b.pos[0] as f64 - center[0] as f64;
            let dy = b.pos[1] as f64 - center[1] as f64;
            dx * dx + dy * dy <= r2
        })
        .collect()
}

fn walk_objects(
    dat: &[u8],
    pos: &mut usize,
    depth: usize,
    budget: &mut usize,
    names: &[String],
    out: &mut Vec<PlacedBrush>,
) -> Option<()> {
    if depth > MAX_DEPTH || *budget == 0 {
        return None;
    }
    *budget -= 1;

    let base = *pos;
    if base.checked_add(OBJECT_NODE_LEN)? > dat.len() {
        return None;
    }
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
        parse_object_block(dat, body, block_end, names, out);
    }

    let mut p = block_end;
    for bit in 0..8 {
        if childs & (1 << bit) != 0 {
            walk_objects(dat, &mut p, depth + 1, budget, names, out)?;
        }
    }
    *pos = p;
    Some(())
}

/// Walk one type-sorted object block, emitting only `Brush` entries.
fn parse_object_block(dat: &[u8], start: usize, end: usize, names: &[String], out: &mut Vec<PlacedBrush>) {
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
                if let Some(model) = names.get(proto as usize) {
                    if let Some(m) = read_matrix34(dat, q + 36) {
                        let (pos, rot, scale) = brush_transform(m);
                        out.push(PlacedBrush {
                            model: model.clone(),
                            pos,
                            rot,
                            scale,
                        });
                    }
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
                // Road: fixed 64 + nVerts * Vec3.
                if q.checked_add(40).is_none_or(|e| e > end) {
                    break;
                }
                let Some(nv) = rd_i32(dat, q + 36) else { break };
                if nv < 0 {
                    break;
                }
                let Some(adv) = (nv as usize)
                    .checked_mul(12)
                    .and_then(|b| 64usize.checked_add(b))
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
                // MergedMesh: fixed 64 + nGroups * 8.
                if q.checked_add(40).is_none_or(|e| e > end) {
                    break;
                }
                let Some(ng) = rd_u32(dat, q + 36) else { break };
                let Some(adv) = (ng as usize)
                    .checked_mul(8)
                    .and_then(|b| 64usize.checked_add(b))
                else {
                    break;
                };
                if q.checked_add(adv).is_none_or(|e| e > end) {
                    break;
                }
                p = q + adv;
            }
            // Unrecognised type: the block boundary is stored separately, so
            // skipping the remainder keeps the octree walk exact.
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
///
/// Returns `(translation, quaternion [x,y,z,w], uniform scale)`.
fn brush_transform(m: [f32; 12]) -> ([f32; 3], [f32; 4], f32) {
    let pos = [m[3], m[7], m[11]];
    let rows = [
        [m[0], m[1], m[2]],
        [m[4], m[5], m[6]],
        [m[8], m[9], m[10]],
    ];
    let norm = |v: [f32; 3]| -> f64 {
        ((v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2)).sqrt()
    };
    let scale = ((norm(rows[0]) + norm(rows[1]) + norm(rows[2])) / 3.0) as f32;

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

    (
        pos,
        [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32],
        scale,
    )
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Path relative to this crate's `CARGO_MANIFEST_DIR`.
    const CLASS3_REL: &str = "../../../extracted-game/level/terrain/terrain.dat";

    fn load_class3() -> Option<Vec<u8>> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CLASS3_REL);
        std::fs::read(p).ok()
    }

    #[test]
    fn malformed_is_total() {
        assert!(Header::parse(&[]).is_none());
        assert!(Header::parse(&[0u8; 10]).is_none());
        assert!(prototypes(&[]).is_empty());
        assert!(prototypes(&[0xff; 64]).is_empty());
        assert!(decode_brushes(&[]).is_empty());
        assert!(brushes_in_region(&[], [0.0; 3], 100.0).is_empty());
    }

    #[test]
    fn class3_prototypes_and_brushes() {
        let Some(dat) = load_class3() else {
            eprintln!("Class3 terrain.dat absent; skipping real-data test");
            return;
        };

        let hdr = Header::parse(&dat).expect("header");
        assert!(!hdr.big_endian);
        assert!((hdr.world_size() - 4096.0).abs() < 1e-6);

        let protos = prototypes(&dat);
        assert_eq!(protos.len(), 2131, "Class3 Table B count");
        assert!(
            protos.iter().all(|p| !p.is_empty()),
            "every prototype path must be non-empty"
        );
        assert!(
            protos
                .iter()
                .any(|p| p.contains("guardrail") || p.contains("rock")),
            "expected guardrail/rock prototypes"
        );

        let brushes = decode_brushes(&dat);
        assert!(
            (30_000..40_000).contains(&brushes.len()),
            "brush count {} outside 30k..40k",
            brushes.len()
        );
        for b in &brushes {
            assert!(!b.model.is_empty(), "empty model path");
            assert!(
                b.pos[0] >= 0.0 && b.pos[0] <= 4096.0,
                "x {} out of range",
                b.pos[0]
            );
            assert!(
                b.pos[1] >= 0.0 && b.pos[1] <= 4096.0,
                "y {} out of range",
                b.pos[1]
            );
        }

        // Region query: centred on a real brush, radius 1 m must find it.
        let some = brushes.first().expect("at least one brush");
        let near = brushes_in_region(&dat, some.pos, 1.0);
        assert!(
            near.iter().any(|b| b.model == some.model && b.pos == some.pos),
            "region query missed its own centre"
        );

        let guardrail = protos.iter().filter(|p| p.contains("guardrail")).count();
        println!(
            "Class3 architecture: prototypes={} brushes={} guardrail_protos={} first={:?}",
            protos.len(),
            brushes.len(),
            guardrail,
            some.model
        );
    }
}
