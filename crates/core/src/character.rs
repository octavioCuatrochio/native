//! Character loading (model + submesh materials + skeleton + animation clips).
//!
//! Assembles everything a renderer needs to draw *and animate* a CryEngine
//! character (player survivor or zombie) in one shot:
//!   * the decoded skinned mesh (`decode_model_with(bytes, false)`, so vertices
//!     stay in bind model space for skinning),
//!   * the ordered `.mtl` material palette plus a per-`Submesh` resolution so
//!     every submesh gets its own textured slot,
//!   * the `Skeleton` and per-vertex bone influences,
//!   * a locomotion `ClipSet` scanned from the mounted paks, so the caller can
//!     `pick` an idle/walk/run/attack clip that matches the rig's controllers.

use crate::anim;
use crate::crypak::Vfs;
use crate::material::{self, MaterialSlot};
use crate::mesh;
use crate::skin;
use std::collections::{HashMap, HashSet};

/// Default cap on `.caf` files inspected by [`pick_caf`].
const DEFAULT_MAX_CAF: usize = 4096;

/// A fully loaded, animatable character.
///
/// `influences` is stored as a flat stream of bone weights (4 per vertex, in
/// vertex order). Use [`Character::influence_groups`] to get the `[BoneWeight; 4]`
/// form expected by [`skin::deform`].
#[derive(Default)]
pub struct Character {
    pub model: mesh::DecodedModel,
    pub slots: Vec<MaterialSlot>,
    pub skeleton: skin::Skeleton,
    pub influences: Vec<Vec<skin::BoneWeight>>,
    pub clips: anim::ClipSet,
}

impl Character {
    /// Re-group the flat influence stream into the `[BoneWeight; 4]` arrays used
    /// by the skinning helpers.
    pub fn influence_groups(&self) -> Vec<Vec<[skin::BoneWeight; 4]>> {
        self.influences
            .iter()
            .map(|mesh| {
                mesh.chunks(4)
                    .map(|chunk| {
                        let mut group = [skin::BoneWeight::default(); 4];
                        for (i, w) in chunk.iter().take(4).enumerate() {
                            group[i] = *w;
                        }
                        group
                    })
                    .collect()
            })
            .collect()
    }
}

/// Build a [`anim::ClipSet`] from `(path, controller_ids)` pairs, skipping
/// clips that animate no controllers. Each path is classified by keyword.
///
/// The player's real locomotion clips live under `.../hero_male/test/`, while
/// static aim/look poses (`relaxed_idleaimposes_*`, `lookposes_*`) also parse
/// into the same rigs. Those poses are dropped whenever at least one genuine
/// locomotion clip is present, so the bank cannot degrade to a single frozen
/// pose; they are only kept (and played) when nothing else exists.
pub fn collect_clips(entries: impl Iterator<Item = (String, Vec<u32>)>) -> anim::ClipSet {
    let all: Vec<(String, Vec<u32>)> = entries.filter(|(_, c)| !c.is_empty()).collect();
    let has_locomotion = all.iter().any(|(path, _)| !anim::is_static_pose(path));
    let mut set = anim::ClipSet::new();
    for (path, controllers) in all {
        if has_locomotion && anim::is_static_pose(&path) {
            continue;
        }
        let locomotion = anim::ClipSet::classify(&path);
        set.add(path, controllers, locomotion);
    }
    set
}

/// Resolve one material slot per `Submesh`, in mesh/submesh order. The renderer
/// walks `model.meshes[*].submeshes` in the same order and pairs each with the
/// returned slot. Out-of-range material indices yield an empty default slot.
pub fn submesh_materials(model: &mesh::DecodedModel, slots: &[MaterialSlot]) -> Vec<MaterialSlot> {
    let mut out = Vec::new();
    for m in &model.meshes {
        for sub in &m.submeshes {
            out.push(slots.get(sub.material as usize).cloned().unwrap_or_default());
        }
    }
    out
}

/// Merge `.$caf` and plain `.caf` path lists, de-duplicating by a normalized
/// key so a `.$caf` file shadows an identically named `.caf` regardless of
/// mount order. The `.$caf` list is treated as authoritative/primary.
fn merge_caf_paths(dollar_caf: Vec<String>, plain_caf: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(dollar_caf.len() + plain_caf.len());
    for path in dollar_caf.into_iter().chain(plain_caf) {
        let key = path.to_ascii_lowercase().replace(".$caf", ".caf");
        if seen.insert(key) {
            out.push(path);
        }
    }
    out
}

/// Every CAF candidate in the VFS, accepting both the CryEngine `.$caf`
/// extension (the hero/zombie locomotion clips ship this way) and plain `.caf`.
/// The previous scan only asked for `.caf`, so it silently skipped the entire
/// `hero_male/test/` locomotion set and matched only static aim/look poses.
///
/// The list is then stably sorted by [`anim::prefer_locomotion`] (highest
/// first) so genuine gaits are inspected *before* the [`scan_caf_ids`] cap is
/// applied. Pak order is arbitrary — `hero_run_forward.$caf` sits behind ~955
/// other `.$caf` entries — so without this the cap could stop the scan before
/// the hero set, leaving only static aim/look poses in the bank (bind pose).
fn scan_paths(vfs: &Vfs) -> Vec<String> {
    let mut paths = merge_caf_paths(vfs.paths_with_ext(".$caf"), vfs.paths_with_ext(".caf"));
    paths.sort_by(|a, b| anim::prefer_locomotion(b).cmp(&anim::prefer_locomotion(a)));
    paths
}

/// Scan the mounted paks for `.caf`/`.$caf` clips, keeping those whose
/// controller ids are non-empty. `rig`, when given, keeps only paths containing
/// it (case-insensitive). At most `max_caf` clips are collected.
fn scan_caf_ids(vfs: &Vfs, rig: Option<&str>, max_caf: usize) -> Vec<(String, Vec<u32>)> {
    let mut out = Vec::new();
    if max_caf == 0 {
        return out;
    }
    let rig_lc = rig
        .map(|r| r.to_ascii_lowercase())
        .filter(|r| !r.is_empty());
    for path in scan_paths(vfs) {
        if out.len() >= max_caf {
            break;
        }
        if let Some(rig) = &rig_lc {
            if !path.to_ascii_lowercase().contains(rig.as_str()) {
                continue;
            }
        }
        let Some(bytes) = vfs.read(&path) else { continue };
        let ids = anim::controller_ids(&bytes);
        if ids.is_empty() {
            continue;
        }
        out.push((path, ids));
    }
    out
}

/// Load a character from its `.chr`/`.cgf` and assemble its animation clips.
///
/// Returns `None` when the asset cannot be read or lacks the skinned-mesh
/// chunks (model, skeleton, influences).
pub fn load(vfs: &Vfs, chr_path: &str, rig: Option<&str>, max_caf: usize) -> Option<Character> {
    let bytes = vfs.read(chr_path)?;
    let model = mesh::decode_model_with(&bytes, false)?;
    let skeleton = skin::parse_skeleton(&bytes)?;
    let raw = skin::decode_influences(&bytes)?;
    let influences: Vec<Vec<skin::BoneWeight>> = raw
        .into_iter()
        .map(|mesh| mesh.into_iter().flatten().collect())
        .collect();
    let slots = model
        .material_name
        .as_deref()
        .map(|name| material::load_mtl(vfs, chr_path, name))
        .unwrap_or_default();
    let clips = collect_clips(scan_caf_ids(vfs, rig, max_caf).into_iter());
    Some(Character { model, slots, skeleton, influences, clips })
}

/// Convenience: pick the best default clip (preferring `Idle`, else the first)
/// for a character and return it ready to animate.
pub fn pick_caf(vfs: &Vfs, chr_path: &str, rig: Option<&str>) -> Option<(String, anim::Caf)> {
    let entries = scan_caf_ids(vfs, rig, DEFAULT_MAX_CAF);
    if entries.is_empty() {
        return None;
    }
    let clips = collect_clips(entries.iter().cloned());
    let skeleton = vfs.read(chr_path).and_then(|b| skin::parse_skeleton(&b));
    let chosen = skeleton
        .as_ref()
        .and_then(|sk| clips.pick(sk, anim::Locomotion::Idle))
        .map(|entry| entry.path.clone())
        .or_else(|| entries.first().map(|(path, _)| path.clone()))?;
    let bytes = vfs.read(&chosen)?;
    let caf = anim::Caf::parse(&bytes)?;
    Some((chosen, caf))
}

/// The locomotion classes a [`ClipBank`] eagerly preloads. `Death` is left out
/// (it is not a looping locomotion state); request it via `caf_for` when the
/// caller needs it and it happens to be present.
const BANK_CLASSES: [anim::Locomotion; 5] = [
    anim::Locomotion::Idle,
    anim::Locomotion::Walk,
    anim::Locomotion::Run,
    anim::Locomotion::Sprint,
    anim::Locomotion::Attack,
];

/// Fallback order for a wanted class: the first class in the chain that the
/// bank actually holds is used. Sprint degrades through Run and Walk to Idle;
/// Attack only degrades to Idle; Walk prefers Run if Walk is absent.
fn fallback_chain(want: anim::Locomotion) -> &'static [anim::Locomotion] {
    use anim::Locomotion::*;
    match want {
        Sprint => &[Sprint, Run, Walk, Idle],
        Run => &[Run, Walk, Idle],
        Walk => &[Walk, Run, Idle],
        Attack => &[Attack, Idle],
        Death => &[Death, Idle],
        Idle => &[Idle],
    }
}

/// A preloaded bank with (at most) one parsed [`anim::Caf`] per locomotion
/// class, already matched to a character's skeleton.
///
/// Building it once at load time makes locomotion switching trivial for the
/// app: `bank.caf_for(want)` picks the wanted clip or the nearest fallback,
/// with no further `.caf` scanning or parsing.
#[derive(Default, Clone)]
pub struct ClipBank {
    clips: HashMap<anim::Locomotion, (String, anim::Caf)>,
}

impl ClipBank {
    /// Scan the mounted paks for `.caf` clips, and for every class in
    /// [`BANK_CLASSES`] pick the clip that best overlaps the character's
    /// skeleton, parse it, and store it. Classes with no matching clip are
    /// simply absent from the bank.
    pub fn build(vfs: &Vfs, chr_path: &str, rig: Option<&str>, max_caf: usize) -> ClipBank {
        let mut bank = ClipBank::default();
        let entries = scan_caf_ids(vfs, rig, max_caf);
        if entries.is_empty() {
            return bank;
        }
        let clips = collect_clips(entries.into_iter());
        let Some(skeleton) = vfs.read(chr_path).and_then(|b| skin::parse_skeleton(&b)) else {
            return bank;
        };
        for loc in BANK_CLASSES {
            let Some(entry) = clips.pick(&skeleton, loc) else { continue };
            let path = entry.path.clone();
            let Some(caf) = vfs.read(&path).and_then(|b| anim::Caf::parse(&b)) else { continue };
            bank.clips.insert(loc, (path, caf));
        }
        bank
    }

    /// Test-only constructor from already-parsed clips.
    #[cfg(test)]
    pub(crate) fn from_parts(parts: Vec<(anim::Locomotion, String, anim::Caf)>) -> ClipBank {
        let mut bank = ClipBank::default();
        for (loc, path, caf) in parts {
            bank.clips.insert(loc, (path, caf));
        }
        bank
    }

    /// The parsed clip stored for exactly this class, if present.
    pub fn get(&self, loc: anim::Locomotion) -> Option<&anim::Caf> {
        self.clips.get(&loc).map(|(_, caf)| caf)
    }

    /// The source path of the clip stored for exactly this class, if present.
    pub fn path(&self, loc: anim::Locomotion) -> Option<&str> {
        self.clips.get(&loc).map(|(path, _)| path.as_str())
    }

    /// Resolve `want` through its fallback chain, returning the first class the
    /// bank actually holds.
    pub fn resolve(&self, want: anim::Locomotion) -> Option<anim::Locomotion> {
        fallback_chain(want)
            .iter()
            .copied()
            .find(|loc| self.clips.contains_key(loc))
    }

    /// The parsed clip to play for `want`, following the fallback chain.
    pub fn caf_for(&self, want: anim::Locomotion) -> Option<&anim::Caf> {
        self.resolve(want).and_then(|loc| self.get(loc))
    }

    /// Number of classes held (at most one clip each).
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }
}

/// Load a character *and* its preloaded [`ClipBank`]: `load` plus
/// [`ClipBank::build`]. Returns `None` under the same conditions as `load`.
pub fn load_full(
    vfs: &Vfs,
    chr_path: &str,
    rig: Option<&str>,
    max_caf: usize,
) -> Option<(Character, ClipBank)> {
    let character = load(vfs, chr_path, rig, max_caf)?;
    let bank = ClipBank::build(vfs, chr_path, rig, max_caf);
    Some((character, bank))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skel(ids: &[u32]) -> skin::Skeleton {
        let mut s = skin::Skeleton::default();
        for (i, id) in ids.iter().enumerate() {
            s.by_controller.insert(*id, i);
        }
        s
    }

    #[test]
    fn collect_clips_builds_set_and_picks_best_overlap() {
        let entries = vec![
            ("a/idle_bad.caf".to_string(), vec![1u32, 2, 3]),
            ("b/idle_good.caf".to_string(), vec![10u32, 20, 30]),
            ("c/walk_fwd.caf".to_string(), vec![10u32]),
            ("d/empty.caf".to_string(), vec![]),
        ];
        let set = collect_clips(entries.into_iter());
        // The controller-less clip is dropped.
        assert_eq!(set.len(), 3);

        let sk = skel(&[10, 20, 30, 1]);
        let got = set.pick(&sk, anim::Locomotion::Idle).expect("idle clip");
        assert_eq!(got.path, "b/idle_good.caf");
        assert_eq!(got.controllers, vec![10, 20, 30]);
        // No Run-clips in the set.
        assert!(set.pick(&sk, anim::Locomotion::Run).is_none());
    }

    #[test]
    fn missing_chr_returns_none() {
        let vfs = Vfs::new();
        assert!(load(&vfs, "characters/does_not_exist.chr", None, 16).is_none());
        assert!(pick_caf(&vfs, "characters/does_not_exist.chr", None).is_none());
    }

    #[test]
    fn submesh_materials_maps_indices() {
        let mk = |material: u32| mesh::Submesh { material, indices: vec![0, 1, 2] };
        let model = mesh::DecodedModel {
            meshes: vec![mesh::DecodedMesh {
                submeshes: vec![mk(0), mk(2), mk(9)],
                ..Default::default()
            }],
            ..Default::default()
        };
        let slots = vec![
            MaterialSlot { name: "zero".into(), ..Default::default() },
            MaterialSlot { name: "one".into(), ..Default::default() },
            MaterialSlot { name: "two".into(), ..Default::default() },
        ];
        let got = submesh_materials(&model, &slots);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].name, "zero");
        assert_eq!(got[1].name, "two");
        assert_eq!(got[2].name, "", "out-of-range material -> empty default slot");
    }

    fn test_caf() -> anim::Caf {
        anim::Caf { start: 0, end: 30, secs_per_tick: 1.0 / 30.0, controllers: Vec::new() }
    }

    fn test_bank() -> ClipBank {
        ClipBank::from_parts(vec![
            (anim::Locomotion::Idle, "idle.caf".to_string(), test_caf()),
            (anim::Locomotion::Walk, "walk.caf".to_string(), test_caf()),
        ])
    }

    #[test]
    fn clip_bank_resolve_fallback_order() {
        let bank = test_bank();
        assert_eq!(bank.len(), 2);
        assert!(!bank.is_empty());

        // Sprint -> Run -> Walk -> Idle: Run is absent, so Walk wins.
        assert_eq!(bank.resolve(anim::Locomotion::Sprint), Some(anim::Locomotion::Walk));
        // Run -> Walk (Walk present).
        assert_eq!(bank.resolve(anim::Locomotion::Run), Some(anim::Locomotion::Walk));
        // Attack -> Idle.
        assert_eq!(bank.resolve(anim::Locomotion::Attack), Some(anim::Locomotion::Idle));
        // Walk itself is present.
        assert_eq!(bank.resolve(anim::Locomotion::Walk), Some(anim::Locomotion::Walk));
    }

    #[test]
    fn clip_bank_caf_for_and_accessors() {
        let bank = test_bank();
        // caf_for follows the same chain and returns a parsed clip.
        let caf = bank.caf_for(anim::Locomotion::Sprint).expect("sprint falls back to walk");
        assert_eq!(caf.end, 30);
        assert!(bank.caf_for(anim::Locomotion::Attack).is_some());

        assert_eq!(bank.path(anim::Locomotion::Walk), Some("walk.caf"));
        assert!(bank.get(anim::Locomotion::Walk).is_some());
        // Absent exact classes return None (even though resolve finds a fallback).
        assert!(bank.get(anim::Locomotion::Run).is_none());
        assert!(bank.path(anim::Locomotion::Run).is_none());
    }

    #[test]
    fn clip_bank_missing_locomotion_resolves_through_chain() {
        // Only Idle present: every class should resolve to Idle.
        let bank = ClipBank::from_parts(vec![(
            anim::Locomotion::Idle,
            "idle.caf".to_string(),
            test_caf(),
        )]);
        for want in [
            anim::Locomotion::Idle,
            anim::Locomotion::Walk,
            anim::Locomotion::Run,
            anim::Locomotion::Sprint,
            anim::Locomotion::Attack,
            anim::Locomotion::Death,
        ] {
            assert_eq!(bank.resolve(want), Some(anim::Locomotion::Idle), "want {want:?}");
            assert!(bank.caf_for(want).is_some());
        }
    }

    #[test]
    fn clip_bank_empty_resolves_none() {
        let bank = ClipBank::from_parts(Vec::new());
        assert!(bank.is_empty());
        assert_eq!(bank.len(), 0);
        assert_eq!(bank.resolve(anim::Locomotion::Sprint), None);
        assert_eq!(bank.resolve(anim::Locomotion::Death), None);
        assert!(bank.caf_for(anim::Locomotion::Attack).is_none());
        assert!(bank.path(anim::Locomotion::Idle).is_none());
    }

    #[test]
    fn clip_bank_build_without_assets_is_empty() {
        // No `.caf` entries and no `.chr`: build must degrade to an empty bank.
        let vfs = Vfs::new();
        let bank = ClipBank::build(&vfs, "characters/does_not_exist.chr", None, 16);
        assert!(bank.is_empty());
        assert_eq!(bank.len(), 0);
    }

    #[test]
    fn load_full_missing_chr_returns_none() {
        let vfs = Vfs::new();
        assert!(load_full(&vfs, "characters/does_not_exist.chr", None, 16).is_none());
    }

    #[test]
    fn influence_groups_roundtrip() {
        let c = Character {
            influences: vec![vec![
                skin::BoneWeight { bone: 1, weight: 0.5 },
                skin::BoneWeight { bone: 2, weight: 0.5 },
                skin::BoneWeight::default(),
                skin::BoneWeight::default(),
            ]],
            ..Default::default()
        };
        let groups = c.influence_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 1);
        assert_eq!(groups[0][0][0].bone, 1);
        assert_eq!(groups[0][0][1].bone, 2);
    }

    /// Optional real-asset smoke test: set `SOD_CHR` to a loose `.chr` file to
    /// exercise the full loader; silently skipped when unset.
    #[test]
    fn real_character_optional() {
        let Some(path) = std::env::var_os("SOD_CHR") else { return };
        let path = std::path::PathBuf::from(path);
        let (Some(dir), Some(name)) =
            (path.parent().map(|p| p.to_path_buf()), path.file_name().and_then(|n| n.to_str()).map(|s| s.to_string()))
        else {
            return;
        };
        let mut vfs = Vfs::new();
        vfs.set_loose_root(dir);
        if let Some(c) = load(&vfs, &name, None, 32) {
            assert!(!c.model.meshes.is_empty());
            assert!(!c.skeleton.bones.is_empty());
        }
    }

    // -- scan fix: `.caf` + `.$caf` -----------------------------------------

    #[test]
    fn merge_caf_paths_accepts_dollar_caf_and_shadows_plain() {
        let merged = merge_caf_paths(
            vec![
                "a/hero_run_forward.$caf".to_string(),
                "a/pose_only.$caf".to_string(),
            ],
            vec![
                // Same stem as the `.$caf` above: must be shadowed.
                "a/hero_run_forward.caf".to_string(),
                "a/lookposes_basic.caf".to_string(),
            ],
        );
        assert!(merged.iter().any(|p| p.ends_with("hero_run_forward.$caf")));
        assert!(
            !merged.iter().any(|p| p.ends_with("hero_run_forward.caf")),
            "plain .caf should be shadowed by the .$caf of the same stem"
        );
        assert!(merged.iter().any(|p| p.ends_with("pose_only.$caf")));
        assert!(merged.iter().any(|p| p.ends_with("lookposes_basic.caf")));
    }

    #[test]
    fn collect_clips_drops_static_poses_when_locomotion_present() {
        let entries = vec![
            (
                "characters/hero_male/test/hero_run_forward.$caf".to_string(),
                vec![1u32, 2, 3],
            ),
            (
                "characters/hero_male/aim/relaxed_idleaimposes_pistol.caf".to_string(),
                vec![1u32, 2, 3],
            ),
            (
                "characters/hero_male/aim/relaxed_idleaimposes_rifle.caf".to_string(),
                vec![1u32, 2, 3],
            ),
        ];
        let set = collect_clips(entries.into_iter());
        assert_eq!(set.len(), 1, "static poses must be dropped when a gait exists");
        let sk = skel(&[1, 2, 3]);
        assert_eq!(
            set.pick(&sk, anim::Locomotion::Run).map(|e| e.path.as_str()),
            Some("characters/hero_male/test/hero_run_forward.$caf")
        );

        // Nothing but poses: they are preserved as a last resort.
        let only_poses = vec![(
            "characters/hero_male/aim/relaxed_idleaimposes_pistol.caf".to_string(),
            vec![1u32, 2, 3],
        )];
        assert_eq!(collect_clips(only_poses.into_iter()).len(), 1);
    }

    #[test]
    fn hero_idle_dollar_caf_classifies_as_idle() {
        assert_eq!(
            anim::ClipSet::classify("animations/characters/hero_male/test/hero_idle.$caf"),
            anim::Locomotion::Idle
        );
    }

    // -- real hero `.$caf` assets (skipped silently when absent) ------------

    const MASTER_CHR: &str = "characters/characters/male/human/skeleton/skeleton.chr";
    const BODY_CHR: &str = "characters/characters/male/human/body/tshirt_jeans/tshirt_jeans.chr";
    const HERO_RUN_CAF: &str =
        "animations/animations/characters/hero_male/test/hero_run_forward.$caf";

    fn extracted(rel: &str) -> Option<std::path::PathBuf> {
        let mut root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
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

    fn largest_mesh_pair(
        chr: &[u8],
    ) -> Option<(Vec<[f32; 3]>, Vec<[skin::BoneWeight; 4]>)> {
        let model = mesh::decode_model_with(chr, false)?;
        let infl = skin::decode_influences(chr)?;
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

    /// The player's real locomotion clips are `hero_run_forward.$caf`: parse it,
    /// bind it to the human master skeleton, and prove it deforms a body mesh to
    /// different vertex positions at two ticks.
    #[test]
    fn real_hero_dollar_caf_parses_binds_and_deforms() {
        let (Some(caf_p), Some(chr_p), Some(body_p)) = (
            extracted(HERO_RUN_CAF),
            extracted(MASTER_CHR),
            extracted(BODY_CHR),
        ) else {
            return; // real game data absent; skip silently
        };

        let caf = anim::Caf::parse(&std::fs::read(caf_p).expect("read .$caf"))
            .expect("Caf::parse on hero_run_forward.$caf");
        let master =
            skin::parse_skeleton(&std::fs::read(chr_p).expect("read .chr")).expect("master skeleton");

        let (matched, total) = anim::controller_coverage(&caf, &master);
        assert!(total > 0, "hero_run_forward.$caf has no controllers");
        assert!(
            matched > 0,
            "hero_run_forward.$caf matched no bone on the human master skeleton ({matched}/{total})"
        );

        let body = std::fs::read(body_p).expect("read body .chr");
        let (positions, influences) = largest_mesh_pair(&body).expect("body skinned mesh");
        let skin_at = |tick: f32| master.skin_matrices(&master.pose_tick(&caf, tick));
        let a = skin::deform(&positions, &influences, &skin_at(caf.start as f32));
        let b = skin::deform(&positions, &influences, &skin_at((caf.start + 15) as f32));
        let max_delta = a
            .iter()
            .zip(&b)
            .map(|(p, q)| {
                ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-3,
            "hero_run_forward.$caf produced identical deformed vertices at two ticks \
             (max delta {max_delta:.6}); clip did not animate (coverage {matched}/{total})"
        );
    }

    // -- real pak-backed `ClipBank::build` (skipped silently when absent) ----

    /// The installed game directory: `SOD_GAME`, else the default Steam path.
    const DEFAULT_GAME: &str =
        "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";

    fn game_dir() -> Option<std::path::PathBuf> {
        let p = std::env::var_os("SOD_GAME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_GAME));
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    }

    /// Mount the two paks the bank needs (locomotion clips + master skeleton).
    fn game_vfs() -> Option<Vfs> {
        let game = game_dir()?;
        let mut vfs = Vfs::new();
        let mut mounted = 0;
        for rel in ["Game/animations.pak", "Game/characters.pak"] {
            let p = game.join(rel);
            if p.is_file() && vfs.mount(&p).is_ok() {
                mounted += 1;
            }
        }
        if mounted == 2 {
            Some(vfs)
        } else {
            None
        }
    }

    /// Build a `ClipBank` from the *real* paks for the human master skeleton and
    /// assert each gait selects the player's canonical `hero_male/test` clip.
    ///
    /// The cap is deliberately small (500): in raw pak order the first
    /// `hero_male/test` `.$caf` sits behind ~955 earlier `.$caf` entries, so a
    /// naive capped scan would never see them and the bank would remain empty
    /// (or fall back to static aim/look poses). It passes only because
    /// `scan_paths` front-loads genuine locomotion before the cap.
    #[test]
    fn clip_bank_build_picks_hero_locomotion_for_human_master() {
        let Some(vfs) = game_vfs() else { return };
        let master = "characters/male/human/skeleton/skeleton.chr";
        let bank = ClipBank::build(&vfs, master, None, 500);
        assert!(!bank.is_empty(), "no clips banked for the human master skeleton");
        // The cap is 500, but the hero set must be front-loaded well inside it.
        let paths = scan_paths(&vfs);
        let first_hero = paths
            .iter()
            .position(|p| p.contains("hero_male/test/"))
            .expect("hero_male/test clips missing from scan");
        assert!(first_hero < 500, "locomotion not front-loaded (first hero at {first_hero})");
        eprintln!(
            "PROBE bank Idle={:?} Walk={:?} Run={:?} Sprint={:?}",
            bank.path(anim::Locomotion::Idle),
            bank.path(anim::Locomotion::Walk),
            bank.path(anim::Locomotion::Run),
            bank.path(anim::Locomotion::Sprint),
        );
        for (loc, suffix) in [
            (anim::Locomotion::Idle, "hero_idle.$caf"),
            (anim::Locomotion::Walk, "hero_walk_forward.$caf"),
            (anim::Locomotion::Run, "hero_run_forward.$caf"),
            (anim::Locomotion::Sprint, "hero_sprint_run.$caf"),
        ] {
            let p = bank.path(loc).unwrap_or_else(|| panic!("no {loc:?} clip banked"));
            assert!(p.ends_with(suffix), "{loc:?} banked {p} (want *{suffix})");
            assert!(bank.get(loc).is_some(), "{loc:?} has no parsed clip");
            let lp = p.to_ascii_lowercase();
            assert!(
                !lp.contains("aimpose") && !lp.contains("lookpose") && !lp.contains("pose"),
                "{loc:?} banked a static pose: {p}"
            );
        }
    }
}
