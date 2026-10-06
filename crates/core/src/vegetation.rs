//! Placed vegetation (trees, bushes, grass) from `terrain.dat`.
//!
//! This is the *planted foliage* half of the static-object octree: each
//! `eERType_Vegetation` (2) render node carries its transform inline (a world
//! position, a uniform scale and three packed `uint8` angles) and a 16-bit
//! `m_nObjectTypeIndex` into **Table A**, the `StatInstGroup` prototype list
//! (`notes/terrain-decode.md` §2, §4).
//!
//! Format (see `architecture.rs`, which decodes the sibling `Brush` nodes):
//!
//! * `STerrainChunkHeader` + `STerrainInfo` — 32 bytes; bit0 of `nFlags` is the
//!   big-endian (`xbox360`) flag and is rejected here.
//! * Table A — `u32 count` at `0x20` (147 for `Class3`), then `count` records of
//!   `0x150` bytes; a NUL-padded `.cgf` path occupies the record prefix.
//! * `CTerrainNode` heightmap quadtree — walked only to find the octree start.
//! * `SOcTreeNodeChunk` — 32-byte header + object block + 8 masked children.
//!   Vegetation chunks are fixed **56 bytes** after the type word:
//!   `SRenderNodeChunk` (36) + `Vec3 m_vPos` + `f32 m_fScale` +
//!   `u8 m_ucBright, m_ucAngle, m_ucAngleX, m_ucAngleY`.
//!
//! # Rotation scaling
//!
//! `m_ucAngle` is a packed 8-bit yaw. We map the full `0..=255` range onto one
//! turn: `rot_y = m_ucAngle * (2*pi / 256)`, i.e. byte `64` is a quarter turn.
//! This preserves the byte's relative ordering and is monotonic in the raw
//! value; the absolute phase convention is arbitrary for rendering.
//!
//! Every entry point is total — malformed input yields an empty vector, never a
//! panic.

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

/// One placed vegetation instance resolved against Table A.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedVegetation {
    /// Table A model path, forward-slash normalised (e.g.
    /// `objects/landscape/vegetation/trees/maple_02/maple_02.cgf`).
    pub model: String,
    /// World position `(x, y, z)` in metres.
    pub pos: [f32; 3],
    /// Uniform scale `m_fScale`.
    pub scale: f32,
    /// Yaw in radians from `m_ucAngle` (`byte * 2*pi / 256`; see module docs).
    pub rot_y: f32,
    /// Brightness / tint byte `m_ucBright`.
    pub bright: u8,
}

impl PlacedVegetation {
    /// Broad render category of this instance's Table A model path.
    ///
    /// Convenience wrapper around [`kind_for_model`].
    pub fn kind(&self) -> VegetationKind {
        kind_for_model(&self.model)
    }
}

/// Broad render category for a placed vegetation model.
///
/// The renderer typically keeps [`Tree`](VegetationKind::Tree) and
/// [`Shrub`](VegetationKind::Shrub) instances and culls the mass of flat
/// ground-cover [`Grass`](VegetationKind::Grass) cards.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum VegetationKind {
    /// Trunked / canopy vegetation (`trees/…`).
    Tree,
    /// Low woody plants (`bushes/…`, blackberry, bramble, sage, …).
    Shrub,
    /// Flat ground cover (`groundcover/…`: grass, ferns, weeds, flowers, …).
    Grass,
    /// Anything else (rocks, decor, crops, guts, …).
    Other,
}

/// Model-path substrings that mark a [`Tree`](VegetationKind::Tree).
const TREE_WORDS: &[&str] = &[
    "tree", "pine", "oak", "maple", "cedar", "willow", "birch", "spruce",
    "fir", "elm", "aspen", "honey_mesquite", "poplar", "beech", "cypress",
    "redwood", "sequoia", "juniper", "hemlock", "larch", "sycamore",
    "cottonwood", "eucalyptus", "magnolia", "walnut", "hickory", "chestnut",
];

/// Model-path substrings that mark a [`Shrub`](VegetationKind::Shrub).
const SHRUB_WORDS: &[&str] = &[
    "bush", "shrub", "blackberry", "bramble", "sage", "azalea", "boxwood",
    "hawthorn", "myrtle", "hydrangea", "holly", "huckleberry", "blueberry",
];

/// Model-path substrings that mark [`Grass`](VegetationKind::Grass).
const GRASS_WORDS: &[&str] = &[
    "grass", "fern", "weed", "flower", "leaf", "litter", "clover",
    "dandelion", "reed", "cattail", "moss", "thistle", "bracken", "nettle",
];

/// Classify a Table A model path into a [`VegetationKind`].
///
/// Case-insensitive substring matching, tested in the priority order
/// `Tree` → `Shrub` → `Grass` → `Other`. Total: any input yields a kind, and
/// the empty string is [`Other`](VegetationKind::Other).
pub fn kind_for_model(path: &str) -> VegetationKind {
    let lower = path.to_ascii_lowercase();
    if TREE_WORDS.iter().any(|w| lower.contains(w)) {
        VegetationKind::Tree
    } else if SHRUB_WORDS.iter().any(|w| lower.contains(w)) {
        VegetationKind::Shrub
    } else if GRASS_WORDS.iter().any(|w| lower.contains(w)) {
        VegetationKind::Grass
    } else {
        VegetationKind::Other
    }
}

/// Borrow the instances whose [`kind`](PlacedVegetation::kind) is in `keep`.
///
/// An empty `keep` keeps every instance. Order is preserved.
pub fn filter_kinds<'a>(
    veg: &'a [PlacedVegetation],
    keep: &[VegetationKind],
) -> Vec<&'a PlacedVegetation> {
    if keep.is_empty() {
        return veg.iter().collect();
    }
    veg.iter()
        .filter(|v| keep.contains(&kind_for_model(&v.model)))
        .collect()
}

/// Count instances by kind, in the order `[Tree, Shrub, Grass, Other]`.
///
/// The four counts sum to `veg.len()`.
pub fn count_by_kind(veg: &[PlacedVegetation]) -> [(VegetationKind, usize); 4] {
    let mut counts = [
        (VegetationKind::Tree, 0usize),
        (VegetationKind::Shrub, 0usize),
        (VegetationKind::Grass, 0usize),
        (VegetationKind::Other, 0usize),
    ];
    for v in veg {
        let k = kind_for_model(&v.model);
        match k {
            VegetationKind::Tree => counts[0].1 += 1,
            VegetationKind::Shrub => counts[1].1 += 1,
            VegetationKind::Grass => counts[2].1 += 1,
            VegetationKind::Other => counts[3].1 += 1,
        }
    }
    counts
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

/// `None` if the image is too short to carry the header; otherwise whether the
/// image is the (unsupported) big-endian variant.
fn big_endian(dat: &[u8]) -> Option<bool> {
    if dat.len() < HEADER_LEN {
        return None;
    }
    Some(dat.get(2)? & 1 != 0)
}

/// Decode one fixed-stride, NUL-padded path field into a forward-slash string.
fn decode_path(slot: &[u8]) -> String {
    let end = slot.iter().position(|&b| b == 0).unwrap_or(slot.len());
    let mut s = String::with_capacity(end);
    for &b in &slot[..end] {
        s.push(if b == b'\\' { '/' } else { b as char });
    }
    s
}

// ---------------------------------------------------------------------------
// prototype table A
// ---------------------------------------------------------------------------

/// Table A (`StatInstGroup`) model paths, forward-slash normalised, in order.
///
/// `Class3` yields exactly 147 entries. Returns an empty vector for a malformed
/// / truncated image or the big-endian variant.
pub fn prototypes_a(dat: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    match big_endian(dat) {
        Some(false) => {}
        _ => return out,
    }
    let Some(count) = rd_u32(dat, 0x20).map(|c| c as usize) else {
        return out;
    };
    for i in 0..count {
        let Some(o) = i
            .checked_mul(STRIDE_A)
            .and_then(|d| 0x24usize.checked_add(d))
        else {
            break;
        };
        let Some(slot) = o.checked_add(STRIDE_A).and_then(|e| dat.get(o..e)) else {
            break;
        };
        out.push(decode_path(slot));
    }
    out
}

// ---------------------------------------------------------------------------
// terrain quadtree (walked only to locate the object octree)
// ---------------------------------------------------------------------------

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

/// End of the terrain quadtree == start of the object octree.
fn terrain_end(dat: &[u8]) -> Option<usize> {
    if big_endian(dat)? {
        return None;
    }
    let unit = rd_i32(dat, 0x0C)?;
    let sector = rd_i32(dat, 0x10)?;
    if unit <= 0 || sector <= 0 || sector % unit != 0 {
        return None;
    }
    let units = (sector / unit) as u32;
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
    // m_pGeomErrors[bit_shift], then pTypes[nSurfaceTypesNum], both 4-byte aligned.
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

/// All placed vegetation nodes resolved to their Table A model path.
///
/// The whole octree is traversed (so block boundaries stay exact); only
/// `eERType_Vegetation` (2) entries are emitted, and a node whose
/// `m_nObjectTypeIndex` is out of range is skipped. Returns an empty vector on
/// malformed input or the big-endian variant.
pub fn decode_vegetation(dat: &[u8]) -> Vec<PlacedVegetation> {
    let mut out = Vec::new();
    match big_endian(dat) {
        Some(false) => {}
        _ => return out,
    }
    let names = prototypes_a(dat);
    if names.is_empty() {
        return out;
    }
    let Some(start) = terrain_end(dat) else {
        return out;
    };
    let mut pos = start;
    let mut budget = MAX_OBJECT_NODES;
    let _ = walk_objects(dat, &mut pos, 0, &mut budget, &names, &mut out);
    out
}

/// Vegetation whose **horizontal** position lies within `radius` of `center`.
///
/// The query is 2-D over the ground plane (`X`/`Y`, since the world is Z-up).
/// `radius < 0` yields nothing.
pub fn vegetation_in_region(dat: &[u8], center: [f32; 3], radius: f32) -> Vec<PlacedVegetation> {
    let r2 = (radius as f64) * (radius as f64);
    decode_vegetation(dat)
        .into_iter()
        .filter(|v| {
            let dx = v.pos[0] as f64 - center[0] as f64;
            let dy = v.pos[1] as f64 - center[1] as f64;
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
    out: &mut Vec<PlacedVegetation>,
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

/// Walk one type-sorted object block, emitting only `Vegetation` entries.
fn parse_object_block(
    dat: &[u8],
    start: usize,
    end: usize,
    names: &[String],
    out: &mut Vec<PlacedVegetation>,
) {
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
                p = q + 96;
            }
            2 => {
                // Vegetation: SRenderNodeChunk (36) + Vec3 pos (12) + f32 scale
                // (4) + 4 packed angle/bright bytes = 56.
                if q.checked_add(56).is_none_or(|e| e > end) {
                    break;
                }
                let proto = rd_u16(dat, q + 32).unwrap_or(0);
                if let Some(model) = names.get(proto as usize) {
                    if let (Some(x), Some(y), Some(z), Some(scale)) = (
                        rd_f32(dat, q + 36),
                        rd_f32(dat, q + 40),
                        rd_f32(dat, q + 44),
                        rd_f32(dat, q + 48),
                    ) {
                        let bright = dat.get(q + 52).copied().unwrap_or(0);
                        let angle = dat.get(q + 53).copied().unwrap_or(0);
                        out.push(PlacedVegetation {
                            model: model.clone(),
                            pos: [x, y, z],
                            scale,
                            rot_y: angle as f32 * (std::f32::consts::TAU / 256.0),
                            bright,
                        });
                    }
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
            // Unrecognised type: the block size is stored separately, so skipping
            // the remainder keeps the octree walk exact.
            _ => break,
        }
    }
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
        assert!(prototypes_a(&[]).is_empty());
        assert!(prototypes_a(&[0xff; 64]).is_empty());
        assert!(decode_vegetation(&[]).is_empty());
        assert!(decode_vegetation(&[0xff; 64]).is_empty());
        assert!(vegetation_in_region(&[], [0.0; 3], 100.0).is_empty());
        assert!(vegetation_in_region(&[0xff; 64], [0.0; 3], 100.0).is_empty());
    }

    #[test]
    fn big_endian_rejected() {
        // A 0x20-byte header whose nFlags bit0 is set (xbox360 variant).
        let mut be = vec![0u8; HEADER_LEN];
        be[2] = 0x01;
        assert!(prototypes_a(&be).is_empty());
        assert!(decode_vegetation(&be).is_empty());
        assert!(vegetation_in_region(&be, [0.0; 3], 4096.0).is_empty());
    }

    #[test]
    fn class3_vegetation() {
        let Some(dat) = load_class3() else {
            eprintln!("Class3 terrain.dat absent; skipping real-data test");
            return;
        };

        let protos = prototypes_a(&dat);
        assert_eq!(protos.len(), 147, "Class3 Table A count");
        // Table A has one unused/empty slot (index 70) that is never referenced,
        // so only the *placed* model paths are required to be non-empty.
        assert!(
            protos.iter().filter(|p| p.is_empty()).count() <= 1,
            "expected at most one empty Table A slot"
        );
        assert!(
            protos.iter().any(|p| p.contains("maple") || p.contains("tree")),
            "expected tree prototypes in Table A"
        );

        let veg = decode_vegetation(&dat);
        assert!(
            (40_000..60_000).contains(&veg.len()),
            "vegetation count {} outside 40k..60k",
            veg.len()
        );
        for v in &veg {
            assert!(!v.model.is_empty(), "empty model path");
            assert!(
                v.pos[0] >= 0.0 && v.pos[0] <= 4096.0,
                "x {} out of range",
                v.pos[0]
            );
            assert!(
                v.pos[1] >= 0.0 && v.pos[1] <= 4096.0,
                "y {} out of range",
                v.pos[1]
            );
            assert!(v.scale.is_finite(), "non-finite scale");
            assert!(v.rot_y >= 0.0 && v.rot_y <= std::f32::consts::TAU, "yaw");
        }
        assert!(
            veg.iter().any(|v| {
                let m = v.model.to_ascii_lowercase();
                m.contains("tree")
                    || m.contains("bush")
                    || m.contains("honey_mesquite")
                    || m.contains("blackberry")
            }),
            "expected at least one tree/bush prototype among placed vegetation"
        );

        // Region query: centred on a real instance, radius 1 m must find it.
        let some = veg.first().expect("at least one vegetation node");
        let near = vegetation_in_region(&dat, some.pos, 1.0);
        assert!(
            near.iter().any(|v| v.model == some.model && v.pos == some.pos),
            "region query missed its own centre"
        );
        assert!(near.len() <= veg.len(), "region query is not a subset");

        println!(
            "Class3 vegetation: prototypes={} placed={} first={:?}",
            protos.len(),
            veg.len(),
            some.model
        );
    }

    // -----------------------------------------------------------------------
    // kind classification
    // -----------------------------------------------------------------------

    fn pv(model: &str) -> PlacedVegetation {
        PlacedVegetation {
            model: model.to_string(),
            pos: [0.0; 3],
            scale: 1.0,
            rot_y: 0.0,
            bright: 0,
        }
    }

    #[test]
    fn kind_for_model_representative() {
        assert_eq!(
            kind_for_model("objects/landscape/vegetation/trees/maple_small_01/maple_02.cgf"),
            VegetationKind::Tree
        );
        assert_eq!(
            kind_for_model("objects/landscape/vegetation/bushes/blackberry/blackberry.cgf"),
            VegetationKind::Shrub
        );
        assert_eq!(
            kind_for_model(
                "objects/landscape/vegetation/groundcover/grass/grass_clumps_01/grass_clump_01.cgf"
            ),
            VegetationKind::Grass
        );
        assert_eq!(
            kind_for_model("objects/decor/tombstones/tombstones_01.cgf"),
            VegetationKind::Other
        );
        // Case-insensitive.
        assert_eq!(
            kind_for_model("OBJECTS/LANDSCAPE/VEGETATION/TREES/PINE/TREEPINE1.CGF"),
            VegetationKind::Tree
        );
        // Empty / unknown is total.
        assert_eq!(kind_for_model(""), VegetationKind::Other);
    }

    #[test]
    fn filter_kinds_keeps_only_requested() {
        let veg = [
            pv("objects/landscape/vegetation/trees/pine/treepine1.cgf"),
            pv("objects/landscape/vegetation/bushes/sagebrush/sagebrush.cgf"),
            pv("objects/landscape/vegetation/groundcover/grass/grass_clumps_01/grass_clump_01.cgf"),
            pv("objects/decor/tombstones/tombstones_01.cgf"),
        ];
        assert_eq!(filter_kinds(&veg, &[]).len(), veg.len());

        let keep = filter_kinds(&veg, &[VegetationKind::Tree, VegetationKind::Shrub]);
        assert_eq!(keep.len(), 2);
        assert!(keep.iter().all(|v| matches!(
            v.kind(),
            VegetationKind::Tree | VegetationKind::Shrub
        )));

        assert_eq!(
            filter_kinds(&veg, &[VegetationKind::Grass]).len(),
            1
        );
        assert_eq!(
            filter_kinds(&veg, &[VegetationKind::Other]).len(),
            1
        );
    }

    #[test]
    fn count_by_kind_sums_to_len() {
        let veg = [
            pv("objects/landscape/vegetation/trees/pine/treepine1.cgf"),
            pv("objects/landscape/vegetation/bushes/sagebrush/sagebrush.cgf"),
            pv("objects/landscape/vegetation/groundcover/grass/grass_clumps_01/grass_clump_01.cgf"),
            pv("objects/decor/tombstones/tombstones_01.cgf"),
        ];
        let counts = count_by_kind(&veg);
        assert_eq!(counts[0], (VegetationKind::Tree, 1));
        assert_eq!(counts[1], (VegetationKind::Shrub, 1));
        assert_eq!(counts[2], (VegetationKind::Grass, 1));
        assert_eq!(counts[3], (VegetationKind::Other, 1));
        assert_eq!(counts.iter().map(|(_, n)| n).sum::<usize>(), veg.len());

        assert_eq!(count_by_kind(&[]).iter().map(|(_, n)| n).sum::<usize>(), 0);
    }

    #[test]
    fn class3_kinds_real_data() {
        let Some(dat) = load_class3() else {
            eprintln!("Class3 terrain.dat absent; skipping real-data kind test");
            return;
        };

        let protos = prototypes_a(&dat);
        let tree_protos = protos
            .iter()
            .filter(|m| kind_for_model(m) == VegetationKind::Tree)
            .count();
        assert!(tree_protos > 0, "expected tree prototypes in Table A");

        let veg = decode_vegetation(&dat);
        if veg.is_empty() {
            eprintln!("Class3 vegetation absent; skipping real-data kind test");
            return;
        }

        let n = veg.len().min(2000);
        let head = count_by_kind(&veg[..n]);
        let distinct = head.iter().filter(|(_, c)| *c > 0).count();
        assert!(
            distinct >= 2,
            "expected >=2 kinds in first {n} entries, got {head:?}"
        );

        let full = count_by_kind(&veg);
        assert_eq!(
            full.iter().map(|(_, c)| c).sum::<usize>(),
            veg.len(),
            "kind counts must sum to placed count"
        );
        println!("Class3 first-{n} kinds: {head:?}");
        println!("Class3 full kinds: {full:?}");
    }
}
