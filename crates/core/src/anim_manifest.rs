//! Deterministic gait -> animation clip manifest for one character rig.
//!
//! The renderer previously resolved hero locomotion to static aim/look poses
//! (a T-pose). This module maps the four base gaits (`Idle`, `Walk`, `Run`,
//! `Sprint`) onto the real `.$caf`/`.caf` clips shipped under the rig, using
//! the same VFS scan the clip loader uses (`character::scan_paths`). Selection
//! is pure/name-driven so it is deterministic; [`build_for`] is the only
//! VFS-reading entry point.

use crate::crypak;

/// A base locomotion gait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gait {
    Idle,
    Walk,
    Run,
    Sprint,
}

impl Gait {
    /// Canonical order used for every manifest.
    pub const ALL: [Gait; 4] = [Gait::Idle, Gait::Walk, Gait::Run, Gait::Sprint];

    /// Filename keyword that must appear for a clip to serve this gait.
    fn keyword(self) -> &'static str {
        match self {
            Gait::Idle => "idle",
            Gait::Walk => "walk",
            Gait::Run => "run",
            Gait::Sprint => "sprint",
        }
    }

    /// Highest-priority name fragment for this gait.
    fn primary(self) -> &'static str {
        match self {
            Gait::Idle => "idle",
            Gait::Walk => "walk_forward",
            Gait::Run => "run_forward",
            Gait::Sprint => "sprint_run",
        }
    }
}

/// One resolved gait -> clip path pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipRef {
    pub gait: Gait,
    pub path: String,
}

/// The pairings chosen for a rig, in [`Gait::ALL`] order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClipManifest {
    pub clips: Vec<ClipRef>,
}

impl ClipManifest {
    /// Best clip for `gait`, if one was found.
    pub fn get(&self, gait: Gait) -> Option<&str> {
        self.clips
            .iter()
            .find(|c| c.gait == gait)
            .map(|c| c.path.as_str())
    }

    /// Every resolved pairing.
    pub fn all(&self) -> &[ClipRef] {
        &self.clips
    }
}

/// Build a manifest for `rig_substr` from the VFS's `.$caf`/`.caf` entries.
pub fn build_for(vfs: &crypak::Vfs, rig_substr: &str) -> ClipManifest {
    let mut paths = vfs.paths_with_ext(".$caf");
    paths.extend(vfs.paths_with_ext(".caf"));
    from_paths(&paths, rig_substr)
}

/// Pure selection over an explicit path list (what [`build_for`] feeds).
///
/// Keeps only `.$caf`/`.caf` paths containing `rig_substr` (case-insensitive;
/// an empty rig means no filter) and picks, per gait, the best-scoring clip:
/// `*_{gait}_forward` / `_{gait}` / `_{gait}_run` rank first, while
/// `aimposes`/`lookposes`/`injured`/item/lean variants are pushed down.
pub fn from_paths(paths: &[String], rig_substr: &str) -> ClipManifest {
    let rig = rig_substr.trim().to_ascii_lowercase();
    let tag = rig.split('_').find(|s| !s.is_empty()).unwrap_or("").to_string();

    let mut clips = Vec::new();
    for gait in Gait::ALL {
        // (score, name length, index) of the current best candidate.
        let mut best: Option<(i32, usize, usize)> = None;
        for (i, path) in paths.iter().enumerate() {
            let lp = path.to_ascii_lowercase();
            if !(lp.ends_with(".$caf") || lp.ends_with(".caf")) {
                continue;
            }
            if !rig.is_empty() && !lp.contains(rig.as_str()) {
                continue;
            }
            let name = stem_of(&lp);
            let sc = score(name, gait, &tag);
            if sc == i32::MIN {
                continue;
            }
            let better = match best {
                None => true,
                Some((bs, blen, bi)) => {
                    sc > bs
                        || (sc == bs
                            && (name.len() < blen
                                || (name.len() == blen && lp < paths[bi].to_ascii_lowercase())))
                }
            };
            if better {
                best = Some((sc, name.len(), i));
            }
        }
        if let Some((_, _, i)) = best {
            clips.push(ClipRef { gait, path: paths[i].clone() });
        }
    }
    ClipManifest { clips }
}

/// Filename stem of an already-lowercased path, with `.$caf`/`.caf` stripped.
fn stem_of(lower_path: &str) -> &str {
    let base = lower_path.rsplit('/').next().unwrap_or(lower_path);
    base.strip_suffix(".$caf")
        .or_else(|| base.strip_suffix(".caf"))
        .unwrap_or(base)
}

/// Rank a clip stem for `gait`. `i32::MIN` means "not this gait".
fn score(stem: &str, gait: Gait, tag: &str) -> i32 {
    let kw = gait.keyword();
    if !stem.contains(kw) {
        return i32::MIN;
    }
    let mut s = if stem.contains(gait.primary()) {
        300
    } else if stem.contains(&format!("_{kw}")) {
        200
    } else {
        100
    };
    const PEN: [(&str, i32); 14] = [
        ("aimposes", 500),
        ("lookposes", 500),
        ("_injured", 200),
        ("injured", 150),
        ("aim", 120),
        ("rifle", 80),
        ("lean", 60),
        ("water", 70),
        ("npc", 50),
        ("basic", 40),
        ("turn", 50),
        ("crouch", 60),
        ("_stop", 40),
        ("_start", 40),
    ];
    for (needle, pen) in PEN {
        if stem.contains(needle) {
            s -= pen;
        }
    }
    if !tag.is_empty() && stem.starts_with(tag) {
        s += 30;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    const REAL_DIR: &str =
        "extracted-game/animations/animations/characters/hero_male/test";

    fn real_paths() -> Option<Vec<String>> {
        let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        root.pop();
        root.pop();
        root.pop();
        let dir = root.join(REAL_DIR);
        if !dir.is_dir() {
            return None;
        }
        let mut out = Vec::new();
        for e in fs::read_dir(&dir).ok()?.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let lp = name.to_ascii_lowercase();
            if lp.ends_with(".$caf") || lp.ends_with(".caf") {
                out.push(format!("animations/animations/characters/hero_male/test/{name}"));
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Minimal central-directory-only ZIP so a `Vfs` can list loose clip names
    /// without shipping fixtures. `build_for` only needs names, not contents.
    fn cd_zip(paths: &[String]) -> Vec<u8> {
        const CD: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
        const EOCD: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
        let mut out = Vec::new();
        for p in paths {
            let name = p.replace('\\', "/");
            let nb = name.as_bytes();
            out.extend_from_slice(&CD);
            out.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&[0, 0, 0, 0]); // crc
            out.extend_from_slice(&[0, 0, 0, 0]); // csize
            out.extend_from_slice(&[0, 0, 0, 0]); // usize
            out.extend_from_slice(&(nb.len() as u16).to_le_bytes());
            out.extend_from_slice(&[0, 0, 0, 0]); // elen, clen
            out.extend_from_slice(&[0, 0]); // disk
            out.extend_from_slice(&[0, 0]); // internal attrs
            out.extend_from_slice(&[0, 0, 0, 0]); // external attrs
            out.extend_from_slice(&[0, 0, 0, 0]); // local offset
            out.extend_from_slice(nb);
        }
        let cd_size = out.len() as u32;
        out.extend_from_slice(&EOCD);
        out.extend_from_slice(&[0, 0, 0, 0]); // disk, cd disk
        out.extend_from_slice(&(paths.len() as u16).to_le_bytes());
        out.extend_from_slice(&(paths.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&[0, 0]); // comment length
        out
    }

    fn vfs_from_paths(paths: &[String]) -> Option<crypak::Vfs> {
        let bytes = cd_zip(paths);
        let file = std::env::temp_dir()
            .join(format!("sod_anim_manifest_{}.pak", std::process::id()));
        fs::write(&file, &bytes).ok()?;
        let mut vfs = crypak::Vfs::new();
        let mounted = vfs.mount(&file).ok();
        let _ = fs::remove_file(&file);
        match mounted {
            Some(n) if n > 0 => Some(vfs),
            _ => None,
        }
    }

    #[test]
    fn synthetic_picks_real_locomotion_clips() {
        let paths: Vec<String> = [
            "animations/characters/hero_male/test/aimposes_relaxed_idle.$caf",
            "animations/characters/hero_male/test/hero_idle.$caf",
            "animations/characters/hero_male/test/hero_idle_deep_water.$caf",
            "animations/characters/hero_male/test/hero_walk_forward.$caf",
            "animations/characters/hero_male/test/hero_walk_forward_injured.$caf",
            "animations/characters/hero_male/test/hero_run_forward.$caf",
            "animations/characters/hero_male/test/hero_run_forward_aim.$caf",
            "animations/characters/hero_male/test/hero_sprint_run.$caf",
            "animations/characters/hero_male/test/hero_sprint_run_injured.$caf",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let m = from_paths(&paths, "hero_male");
        assert_eq!(m.clips.len(), 4, "clips={:?}", m.clips);
        assert_eq!(
            m.get(Gait::Idle),
            Some("animations/characters/hero_male/test/hero_idle.$caf")
        );
        assert_eq!(
            m.get(Gait::Walk),
            Some("animations/characters/hero_male/test/hero_walk_forward.$caf")
        );
        assert_eq!(
            m.get(Gait::Run),
            Some("animations/characters/hero_male/test/hero_run_forward.$caf")
        );
        assert_eq!(
            m.get(Gait::Sprint),
            Some("animations/characters/hero_male/test/hero_sprint_run.$caf")
        );

        for c in m.all() {
            let lp = c.path.to_ascii_lowercase();
            assert!(lp.ends_with(".$caf") || lp.ends_with(".caf"));
            assert!(lp.contains(c.gait.keyword()), "{:?} -> {}", c.gait, c.path);
            assert!(!lp.contains("aimposes") && !lp.contains("lookposes"));
        }
        // A rig that has no matching clips resolves to nothing.
        assert!(from_paths(&paths, "zombie_female").clips.is_empty());
    }

    #[test]
    fn real_hero_male_manifest() {
        let Some(paths) = real_paths() else {
            return; // real tree not present; skip silently
        };
        let Some(vfs) = vfs_from_paths(&paths) else {
            return;
        };
        let m = build_for(&vfs, "hero_male");
        for gait in Gait::ALL {
            let p = m
                .get(gait)
                .unwrap_or_else(|| panic!("no clip for {gait:?}; have {:?}", m.clips));
            let lp = p.to_ascii_lowercase();
            assert!(lp.ends_with(".$caf") || lp.ends_with(".caf"), "bad ext: {p}");
            assert!(lp.contains("hero_male"), "not the rig: {p}");
            assert!(lp.contains(gait.keyword()), "{gait:?} -> {p}");
        }
        assert_eq!(m.clips.len(), 4);
        eprintln!("hero_male manifest:");
        for c in m.all() {
            eprintln!("  {:?} -> {}", c.gait, c.path);
        }
    }
}
