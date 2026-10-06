//! Skinned-mesh decode + linear-blend skinning for CryEngine `.chr`.
//!
//! Per `notes/skinning.md` (validated: identity-pose error 1.3e-7):
//!   * `CompiledBones` (`0xACDC0000`): 32 B pad + N x 584 B; fields controllerID,
//!     worldToBone @0x0D8, boneToWorld @0x108, name[256] @0x138, offsetParent
//!     @0x23C. `parent = i + offsetParent` (0 => root).
//!   * `Matrix3x4` = 12 row-major f32, interpreted as a **column-vector** affine
//!     `p' = M*[p;1]` (translation at flat 3,7,11).
//!   * BONEMAP (DataStream type 9): `u8 boneIds[4]; u8 weights[4]` (sum 255).
//!     Ids are local to the subset's bone map: `global = subsetMap[local]`.
//!
//! ## Playback binding
//!
//! An influence's `global` id indexes the *owning file's* `CompiledBones`. A
//! character `.chr` is only a subset of the master rig (`notes/skinning.md`
//! §2), and different files order that subset differently, so the positional
//! index is **not** portable between a part `.chr` and the assembled master
//! skeleton. Every bone instead carries a stable `controllerID ==
//! crc32(boneName)`; [`controller_slot`] hashes that id into a fixed table, and
//! influences store the slot rather than the positional index. Skinning a part
//! against the master skeleton then binds each vertex to the correct bone of
//! the same controller id (see [`Skeleton::skin_matrices`]).

use crate::asset::parse_cry_chunk_file;
use std::collections::HashMap;

const NS_GEOM: u16 = 0xCCCC;

/// Number of slots in the controller-id lookup table.
///
/// `controllerID == crc32(boneName)`. The game ships 116 distinct character
/// bones; their low 13 bits are collision-free (asserted by
/// `controller_slots_are_unique_on_real_rigs`), so 8192 slots key every bone
/// correctly while keeping the per-frame table small (~512 KiB).
pub const CTRL_SLOTS: usize = 1 << 13;

/// Stable table slot for a bone controller id (`crc32(boneName)`).
///
/// Used both when decoding influences ([`decode_influences`]) and when building
/// skin matrices ([`Skeleton::skin_matrices`]); because both sides hash the
/// *same* controller id, a vertex always binds to its owning bone even when the
/// part `.chr` and the posed skeleton order their bones differently.
#[inline]
pub fn controller_slot(controller_id: u32) -> u16 {
    (controller_id as usize & (CTRL_SLOTS - 1)) as u16
}

/// `controllerID` for every bone of the file's `CompiledBones`, in file order.
/// Returns an empty vector when the file has no skeleton chunk.
fn bone_controller_ids(cf: &crate::asset::CryChunkFile, file: &[u8]) -> Vec<u32> {
    let Some(cb) = cf.chunks.iter().find(|c| c.chunk_type == 0xACDC0000) else {
        return Vec::new();
    };
    let Some(p) = cb.payload(file) else { return Vec::new() };
    if p.len() < 32 {
        return Vec::new();
    }
    let n = (p.len() - 32) / 584;
    (0..n).map(|i| rd_u32(p, 32 + i * 584)).collect()
}


#[derive(Clone, Copy)]
pub struct Mat4(pub [f32; 16]);

impl Mat4 {
    pub fn identity() -> Self {
        Mat4([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])
    }
    pub fn from_3x4(m: &[f32]) -> Self {
        Mat4([m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], 0., 0., 0., 1.])
    }
    pub fn from_quat(q: [f32; 4]) -> Self {
        let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
        Mat4([
            1. - 2. * (y * y + z * z),
            2. * (x * y - z * w),
            2. * (x * z + y * w),
            0.,
            2. * (x * y + z * w),
            1. - 2. * (x * x + z * z),
            2. * (y * z - x * w),
            0.,
            2. * (x * z - y * w),
            2. * (y * z + x * w),
            1. - 2. * (x * x + y * y),
            0.,
            0.,
            0.,
            0.,
            1.,
        ])
    }
    pub fn translation(&self) -> [f32; 3] {
        [self.0[3], self.0[7], self.0[11]]
    }
    pub fn with_translation(mut self, t: [f32; 3]) -> Self {
        self.0[3] = t[0];
        self.0[7] = t[1];
        self.0[11] = t[2];
        self
    }
    pub fn mul(&self, rhs: &Mat4) -> Mat4 {
        let (a, b) = (&self.0, &rhs.0);
        let mut r = [0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += a[i * 4 + k] * b[k * 4 + j];
                }
                r[i * 4 + j] = s;
            }
        }
        Mat4(r)
    }
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
            m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
        ]
    }
    pub fn transform_dir(&self, p: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
            m[4] * p[0] + m[5] * p[1] + m[6] * p[2],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2],
        ]
    }
    /// Inverse of a rigid affine (orthonormal rotation).
    pub fn inverse_affine(&self) -> Mat4 {
        let m = &self.0;
        let r = [
            [m[0], m[1], m[2]],
            [m[4], m[5], m[6]],
            [m[8], m[9], m[10]],
        ];
        // transpose (inverse rotation)
        let mut inv = [0f32; 16];
        for i in 0..3 {
            for j in 0..3 {
                inv[i * 4 + j] = r[j][i];
            }
        }
        let t = [m[3], m[7], m[11]];
        inv[3] = -(inv[0] * t[0] + inv[1] * t[1] + inv[2] * t[2]);
        inv[7] = -(inv[4] * t[0] + inv[5] * t[1] + inv[6] * t[2]);
        inv[11] = -(inv[8] * t[0] + inv[9] * t[1] + inv[10] * t[2]);
        inv[15] = 1.0;
        Mat4(inv)
    }
}

#[derive(Clone)]
pub struct Bone {
    pub name: String,
    pub parent: i32,
    pub controller_id: u32,
    pub bind_local: Mat4,
    pub w2b: Mat4,
}

#[derive(Clone, Default)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
    pub by_controller: HashMap<u32, usize>,
}

#[derive(Clone, Copy, Default)]
pub struct BoneWeight {
    /// Controller-id slot (`controller_slot`), **not** a positional bone index.
    /// Resolves against the table built by [`Skeleton::skin_matrices`], so it is
    /// valid across `.chr` files that order their bone subsets differently.
    pub bone: u16,
    pub weight: f32,
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_bits(rd_u32(b, o))
}

/// Parse `CompiledBones` into a skeleton with bind-local transforms.
pub fn parse_skeleton(file: &[u8]) -> Option<Skeleton> {
    let cf = parse_cry_chunk_file(file)?;
    let cb = cf.chunks.iter().find(|c| c.chunk_type == 0xACDC0000)?;
    let p = cb.payload(file)?;
    if p.len() < 32 {
        return None;
    }
    let n = (p.len() - 32) / 584;
    let mut bones = Vec::with_capacity(n);
    for i in 0..n {
        let o = 32 + i * 584;
        let controller_id = rd_u32(p, o);
        let mut w2b = [0f32; 12];
        let mut b2w = [0f32; 12];
        for k in 0..12 {
            w2b[k] = rd_f32(p, o + 0x0D8 + k * 4);
            b2w[k] = rd_f32(p, o + 0x108 + k * 4);
        }
        let name_raw = &p[o + 0x138..o + 0x138 + 256];
        let end = name_raw.iter().position(|&b| b == 0).unwrap_or(256);
        let name = String::from_utf8_lossy(&name_raw[..end]).to_string();
        let offset_parent = rd_u32(p, o + 0x23C) as i32;
        let parent = if offset_parent == 0 { -1 } else { i as i32 + offset_parent };
        bones.push(Bone {
            name,
            parent,
            controller_id,
            bind_local: Mat4::identity(), // filled below
            w2b: Mat4::from_3x4(&w2b),
        });
        // stash b2w temporarily in bind_local (overwritten next)
        bones[i].bind_local = Mat4::from_3x4(&b2w);
    }
    // Convert b2w -> bind_local.
    let b2w: Vec<Mat4> = bones.iter().map(|b| b.bind_local).collect();
    for i in 0..bones.len() {
        let parent = bones[i].parent;
        bones[i].bind_local = if parent < 0 {
            b2w[i]
        } else {
            b2w[parent as usize].inverse_affine().mul(&b2w[i])
        };
    }
    let mut by_controller = HashMap::new();
    for (i, b) in bones.iter().enumerate() {
        by_controller.insert(b.controller_id, i);
    }
    Some(Skeleton { bones, by_controller })
}

fn stream_payload<'a>(
    cf: &crate::asset::CryChunkFile,
    file: &'a [u8],
    by_id: &HashMap<u32, usize>,
    id: u32,
) -> Option<(u32, usize, usize, &'a [u8])> {
    if id == 0 {
        return None;
    }
    let &idx = by_id.get(&id)?;
    let p = cf.chunks[idx].payload(file)?;
    let ty = rd_u32(p, 4);
    let num = rd_u32(p, 8) as usize;
    let bpe = rd_u32(p, 12) as usize;
    Some((ty, num, bpe, p.get(24..)?))
}

/// Decode per-vertex bone influences for each mesh (in chunk order).
///
/// Each influence's [`BoneWeight::bone`] is the **controller-id slot**
/// (`controller_slot`), resolved through the file's own `CompiledBones`, so the
/// same influence binds correctly whether it is skinned against this file's
/// skeleton or a master skeleton with a different bone order.
pub fn decode_influences(file: &[u8]) -> Option<Vec<Vec<[BoneWeight; 4]>>> {
    let cf = parse_cry_chunk_file(file)?;
    let ctrl_ids = bone_controller_ids(&cf, file);
    let mut by_id: HashMap<u32, usize> = HashMap::new();
    for (i, c) in cf.chunks.iter().enumerate() {
        by_id.insert(c.id, i);
    }
    let mut out = Vec::new();
    for ch in &cf.chunks {
        if ch.namespace() != NS_GEOM || ch.kind() != 0x00 {
            continue;
        }
        let Some(mp) = ch.payload(file) else { continue };
        let ref_at = |slot: usize| rd_u32(mp, 0x1C + slot * 4);
        let Some((_, nverts, _, body)) = stream_payload(&cf, file, &by_id, ref_at(9)) else {
            continue;
        };
        // MeshSubsets + bone maps.
        let subs_id = rd_u32(mp, 0x14);
        let mut subsets: Vec<(usize, usize, usize)> = Vec::new(); // firstVertex, numVertices, mapIndex
        let mut maps: Vec<Vec<u16>> = Vec::new();
        if subs_id != 0 {
            if let Some(&si) = by_id.get(&subs_id) {
                if let Some(sp) = cf.chunks[si].payload(file) {
                    let flags = rd_u32(sp, 0);
                    let nsub = rd_u32(sp, 4) as usize;
                    for i in 0..nsub {
                        let o = 0x10 + i * 36;
                        subsets.push((rd_u32(sp, o + 8) as usize, rd_u32(sp, o + 12) as usize, i));
                    }
                    if flags & 0x2 != 0 {
                        let base = 0x10 + nsub * 36;
                        for i in 0..nsub {
                            let o = base + i * 260;
                            let cnt = rd_u32(sp, o) as usize;
                            let mut ids = Vec::with_capacity(cnt);
                            for k in 0..cnt {
                                ids.push(rd_u16(sp, o + 4 + k * 2));
                            }
                            maps.push(ids);
                        }
                    }
                }
            }
        }
        // Per-vertex influences.
        let mut infl = vec![[BoneWeight::default(); 4]; nverts];
        for v in 0..nverts {
            let o = v * 8;
            let Some(b) = body.get(o..o + 8) else { break };
            // subset index for this vertex
            let mut map = None;
            for &(first_v, num_v, midx) in &subsets {
                if v >= first_v && v < first_v + num_v {
                    map = maps.get(midx);
                    break;
                }
            }
            for k in 0..4 {
                let w = b[4 + k];
                if w == 0 {
                    continue;
                }
                let local = b[k] as usize;
                let global = match map {
                    Some(m) if local < m.len() => m[local] as usize,
                    _ => local,
                };
                // Bind by controller id, not positional index: the owning file's
                // bone order is a subset/permutation of the posed skeleton's.
                let slot = ctrl_ids
                    .get(global)
                    .copied()
                    .map(controller_slot)
                    .unwrap_or(global as u16);
                infl[v][k] = BoneWeight { bone: slot, weight: w as f32 / 255.0 };
            }
        }
        out.push(infl);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

impl Skeleton {
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Build posed world matrices from a per-controller sample. Bones without a
    /// controller keep their bind-local transform; a controller with rotation
    /// only keeps the bind translation.
    pub fn pose(&self, sample: &HashMap<u32, (Option<[f32; 3]>, [f32; 4])>) -> Vec<Mat4> {
        let n = self.bones.len();
        let mut local = Vec::with_capacity(n);
        for b in &self.bones {
            let l = match sample.get(&b.controller_id) {
                Some((pos, q)) => {
                    let rot = Mat4::from_quat(*q);
                    let t = pos.unwrap_or_else(|| b.bind_local.translation());
                    rot.with_translation(t)
                }
                None => b.bind_local,
            };
            local.push(l);
        }
        let mut posed = vec![Mat4::identity(); n];
        for i in 0..n {
            let par = self.bones[i].parent;
            posed[i] = if par < 0 { local[i] } else { posed[par as usize].mul(&local[i]) };
        }
        posed
    }

    /// Like [`pose`](Self::pose) but **anchored**: the root/hip bone keeps the
    /// clip's rotation but its translation is pinned to the bind local, so the
    /// clip's root motion / hip drift does not slide the mesh relative to the
    /// entity transform (CryEngine `DecoupledCatchUp` player locomotion: the
    /// controller moves the entity, the clip only poses the skeleton).
    pub fn pose_anchored(
        &self,
        sample: &HashMap<u32, (Option<[f32; 3]>, [f32; 4])>,
    ) -> Vec<Mat4> {
        let n = self.bones.len();
        let root = crate::locomotion::root_bone(self);
        let mut local = Vec::with_capacity(n);
        for (i, b) in self.bones.iter().enumerate() {
            let l = match sample.get(&b.controller_id) {
                Some((pos, q)) => {
                    let rot = Mat4::from_quat(*q);
                    let t = if Some(i) == root {
                        b.bind_local.translation()
                    } else {
                        pos.unwrap_or_else(|| b.bind_local.translation())
                    };
                    rot.with_translation(t)
                }
                None => b.bind_local,
            };
            local.push(l);
        }
        let mut posed = vec![Mat4::identity(); n];
        for i in 0..n {
            let par = self.bones[i].parent;
            posed[i] = if par < 0 { local[i] } else { posed[par as usize].mul(&local[i]) };
        }
        posed
    }

    /// Per-bone skin matrices, indexed by [`controller_slot`].
    ///
    /// Returns a table of `CTRL_SLOTS` matrices (identity for unused slots) so
    /// that an influence decoded from *any* `.chr` in the same rig resolves to
    /// its bone by controller id — the positional bone order of a part file and
    /// the posed master skeleton differ. `boneToWorld * bindWorldToBone`.
    pub fn skin_matrices(&self, posed: &[Mat4]) -> Vec<Mat4> {
        let mut table = vec![Mat4::identity(); CTRL_SLOTS];
        for (i, p) in posed.iter().enumerate() {
            let Some(b) = self.bones.get(i) else { break };
            table[controller_slot(b.controller_id) as usize] = p.mul(&b.w2b);
        }
        table
    }

    pub fn bind_pose(&self) -> Vec<Mat4> {
        self.bones
            .iter()
            .map(|_| Mat4::identity())
            .collect()
    }

    /// Convenience: sample `caf` at an absolute tick and pose this skeleton.
    ///
    /// Equivalent to `self.pose(&caf.sample_tick(tick))`; the per-bone world
    /// matrices returned are in `self.bones` order.
    pub fn pose_tick(&self, caf: &crate::anim::Caf, tick: f32) -> Vec<Mat4> {
        self.pose(&caf.sample_tick(tick))
    }
}

/// Deform model-space positions with the skin matrices.
pub fn deform(
    positions: &[[f32; 3]],
    influences: &[[BoneWeight; 4]],
    skin: &[Mat4],
) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(positions.len());
    for (i, p) in positions.iter().enumerate() {
        let mut acc = [0f32; 3];
        let infl = influences.get(i);
        if let Some(infl) = infl {
            for bw in infl {
                if bw.weight <= 0.0 {
                    continue;
                }
                let m = &skin[bw.bone as usize % skin.len()];
                let q = m.transform_point(*p);
                acc[0] += q[0] * bw.weight;
                acc[1] += q[1] * bw.weight;
                acc[2] += q[2] * bw.weight;
            }
        } else {
            acc = *p;
        }
        out.push(acc);
    }
    out
}

/// Deform normals (rotation part only, renormalised).
pub fn deform_normals(
    normals: &[[f32; 3]],
    influences: &[[BoneWeight; 4]],
    skin: &[Mat4],
) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(normals.len());
    for (i, nrm) in normals.iter().enumerate() {
        let mut acc = [0f32; 3];
        if let Some(infl) = influences.get(i) {
            for bw in infl {
                if bw.weight <= 0.0 {
                    continue;
                }
                let m = &skin[bw.bone as usize % skin.len()];
                let q = m.transform_dir(*nrm);
                acc[0] += q[0] * bw.weight;
                acc[1] += q[1] * bw.weight;
                acc[2] += q[2] * bw.weight;
            }
        } else {
            acc = *nrm;
        }
        let l = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt().max(1e-9);
        out.push([acc[0] / l, acc[1] / l, acc[2] / l]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const SKIN_CHR: &str = "characters/characters/male/zombie/skin/skin.chr";
    const LOCO_CAF: &str =
        "animations/animations/characters/zombie_male/behavior/look_around01.caf";

    /// Resolve a path under `<repo>/extracted-game`, or `None` when the real
    /// game tree is not present in this checkout (test then skips silently).
    fn extracted(rel: &str) -> Option<PathBuf> {
        let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        root.pop(); // crates
        root.pop(); // native
        root.pop(); // repo root
        let p = root.join("extracted-game").join(rel);
        if p.is_file() {
            Some(p)
        } else {
            None
        }
    }

    /// The largest skinned mesh of `model` paired with the influence group of
    /// the same vertex count (the two decoders walk the mesh chunks separately,
    /// so pair by size rather than by index).
    fn largest_mesh_pair(chr: &[u8]) -> Option<(Vec<[f32; 3]>, Vec<[BoneWeight; 4]>)> {
        let model = crate::mesh::decode_model_with(chr, false)?;
        let infl = decode_influences(chr)?;
        let mut best: Option<(usize, usize)> = None;
        for (mi, m) in model.meshes.iter().enumerate() {
            if m.positions.is_empty() {
                continue;
            }
            if let Some(ii) = infl.iter().position(|g| g.len() == m.positions.len()) {
                let n = m.positions.len();
                if best.map_or(true, |(_, bn)| n > bn) {
                    best = Some((mi, ii));
                }
            }
        }
        let (mi, ii) = best?;
        Some((model.meshes[mi].positions.clone(), infl[ii].clone()))
    }

    #[test]
    fn real_caf_binds_to_skeleton_and_deforms() {
        let (Some(chr_path), Some(caf_path)) = (extracted(SKIN_CHR), extracted(LOCO_CAF)) else {
            return; // real game data absent; skip silently
        };
        let chr = std::fs::read(chr_path).expect("read .chr");
        let caf_bytes = std::fs::read(caf_path).expect("read .caf");

        let skeleton = parse_skeleton(&chr).expect("parse_skeleton");
        let caf = crate::anim::Caf::parse(&caf_bytes).expect("Caf::parse");

        // 1. The clip must bind to this rig: controllerID == crc32(boneName).
        let (matched, total) = crate::anim::controller_coverage(&caf, &skeleton);
        assert!(total > 0, "clip has no controllers");
        assert!(
            matched > 0,
            "no caf controller matched skeleton.by_controller ({matched}/{total})"
        );

        // 2. Posing + skinning at two different ticks must move the vertices.
        let (positions, influences) = largest_mesh_pair(&chr).expect("skinned mesh");
        assert!(!positions.is_empty());

        let skin_at = |tick: f32| {
            let posed = skeleton.pose_tick(&caf, tick);
            skeleton.skin_matrices(&posed)
        };
        let a = deform(&positions, &influences, &skin_at(0.0));
        let b = deform(&positions, &influences, &skin_at(72.0));

        let max_delta = a
            .iter()
            .zip(&b)
            .map(|(p, q)| {
                ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-3,
            "skinned vertices did not move between ticks (max delta {max_delta:.6}); \
             pose is identity (coverage {matched}/{total})"
        );
    }

    const MASTER_CHR: &str = "characters/characters/male/human/skeleton/skeleton.chr";
    const BODY_CHR: &str = "characters/characters/male/human/body/tshirt_jeans/tshirt_jeans.chr";

    fn max_abs_diff(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(p, q)| {
                ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
            })
            .fold(0.0f32, f32::max)
    }

    /// The assembled player poses the master `skeleton.chr` (79 bones) but its
    /// body/skin/head attachments each carry their own `CompiledBones` subset in
    /// a different order. Binding influences by controller id must make a part
    /// skinned against the master identical to the same part skinned against
    /// its own skeleton. (Before the controller-slot fix this differed by ~0.08 m
    /// and the player visibly exploded.)
    #[test]
    fn real_cdf_part_binds_to_master_skeleton_by_controller_id() {
        let (Some(master_p), Some(body_p), Some(caf_p)) =
            (extracted(MASTER_CHR), extracted(BODY_CHR), extracted(LOCO_CAF))
        else {
            return; // real game data absent; skip silently
        };
        let master_bytes = std::fs::read(master_p).unwrap();
        let body_bytes = std::fs::read(body_p).unwrap();
        let caf_bytes = std::fs::read(caf_p).unwrap();

        let master = parse_skeleton(&master_bytes).expect("master skeleton");
        let body = parse_skeleton(&body_bytes).expect("body skeleton");
        let caf = crate::anim::Caf::parse(&caf_bytes).expect("caf");
        let (positions, influences) = largest_mesh_pair(&body_bytes).expect("body mesh");

        // The part must overlap the master (it is a subset of the same rig).
        let (matched, total) = crate::anim::controller_coverage(&caf, &master);
        assert!(matched > 0 && total > 0);

        for tick in [0.0f32, 72.0] {
            let own_posed = body.pose_tick(&caf, tick);
            let own = deform(&positions, &influences, &body.skin_matrices(&own_posed));
            let master_posed = master.pose_tick(&caf, tick);
            let cross = deform(&positions, &influences, &master.skin_matrices(&master_posed));
            let d = max_abs_diff(&own, &cross);
            assert!(
                d < 1e-3,
                "part-vs-master binding mismatch at tick {tick}: {d:.6} m \
                 (influences not bound by controller id)"
            );
        }

        // And the part really animates across ticks.
        let own0 = {
            let p = body.pose_tick(&caf, 0.0);
            deform(&positions, &influences, &body.skin_matrices(&p))
        };
        let own1 = {
            let p = body.pose_tick(&caf, 72.0);
            deform(&positions, &influences, &body.skin_matrices(&p))
        };
        assert!(max_abs_diff(&own0, &own1) > 1e-3, "part did not animate");
    }

    /// Guard the `CTRL_SLOTS` invariant: the shipped rigs' controller ids must
    /// not collide in the lookup table, or two bones would share a slot.
    #[test]
    fn controller_slots_are_unique_on_real_rigs() {
        for rel in [SKIN_CHR, MASTER_CHR, BODY_CHR] {
            let Some(p) = extracted(rel) else { continue };
            let bytes = std::fs::read(p).unwrap();
            let sk = parse_skeleton(&bytes).expect("skeleton");
            let mut seen = std::collections::HashSet::new();
            for b in &sk.bones {
                assert!(
                    seen.insert(controller_slot(b.controller_id)),
                    "controller slot collision for {} in {rel}",
                    b.name
                );
            }
        }
    }
}
