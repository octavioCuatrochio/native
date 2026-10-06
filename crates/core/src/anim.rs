//! CryEngine 3 `.caf` animation decode + sampling.
//!
//! Per `notes/caf-format.md` (validated over 1,924 files): each `Controller`
//! (`0xCCCC000D`, version `0x829`) has a 16-byte descriptor followed by up to
//! four tracks (rotation values/times, position values/times). Rotation is
//! stored as 4xf32 or a drop-largest packed quaternion; times are f32/u16/u8
//! ticks. `SpeedInfo` (`0xAAFC0002`) carries the clip window and 1/30 s tick.

use crate::asset::parse_cry_chunk_file;
use std::collections::HashMap;

const RANGE: f32 = 0.707106781186;

#[derive(Clone)]
pub struct Controller {
    pub id: u32,
    rot_keys: Vec<[f32; 4]>,
    rot_times: Vec<f32>,
    pos_keys: Vec<[f32; 3]>,
    pos_times: Vec<f32>,
    has_pos: bool,
}

#[derive(Clone)]
pub struct Caf {
    pub start: i32,
    pub end: i32,
    pub secs_per_tick: f32,
    pub controllers: Vec<Controller>,
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

fn reverse_bits(mut v: u32) -> u32 {
    let mut r = 0;
    for _ in 0..32 {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

/// Drop-largest packed quaternion, 48-bit form (3 x u16).
fn quat48(m: &[u8]) -> [f32; 4] {
    let m1 = rd_u16(m, 0) as u32;
    let m2 = rd_u16(m, 2) as u32;
    let m3 = rd_u16(m, 4) as u32;
    let index = (m3 >> 14) & 0x3;
    let off: Vec<usize> = (0..4).filter(|&i| i as u32 != index).collect();
    let mut v = [0f32; 4];
    v[off[0]] = (m1 & 0x7FFF) as f32 / 23170.0 - RANGE;
    v[off[1]] = (((m1 >> 15) + (m2 << 1)) & 0x7FFF) as f32 / 23170.0 - RANGE;
    v[off[2]] = (((m2 >> 14) + (m3 << 2)) & 0x7FFF) as f32 / 23170.0 - RANGE;
    let sum = v[off[0]] * v[off[0]] + v[off[1]] * v[off[1]] + v[off[2]] * v[off[2]];
    v[index as usize] = (1.0 - sum).max(0.0).sqrt();
    v
}

/// Drop-largest packed quaternion, 64-bit extended form (21/21/20 bits).
fn quat64ext(m: &[u8]) -> [f32; 4] {
    let m1 = rd_u32(m, 0);
    let m2 = rd_u32(m, 4);
    let index = (m2 >> 30) & 0x3;
    let off: Vec<usize> = (0..4).filter(|&i| i as u32 != index).collect();
    let mut v = [0f32; 4];
    v[off[0]] = (m1 & 0x1FFFFF) as f32 / 1482909.0 - RANGE;
    v[off[1]] = (((m1 >> 21) + (m2 << 11)) & 0x1FFFFF) as f32 / 1482909.0 - RANGE;
    v[off[2]] = ((m2 >> 10) & 0xFFFFF) as f32 / 741454.0 - RANGE;
    let sum = v[off[0]] * v[off[0]] + v[off[1]] * v[off[1]] + v[off[2]] * v[off[2]];
    v[index as usize] = (1.0 - sum).max(0.0).sqrt();
    v
}

fn rot_value_size(fmt: u8) -> usize {
    match fmt {
        0 | 1 => 16,
        3 | 4 | 5 | 7 => 6,
        6 | 8 | 9 => 8,
        _ => 16,
    }
}
fn time_size(fmt: u8) -> usize {
    match fmt {
        0 | 3 => 4,
        1 | 4 => 2,
        2 | 5 => 1,
        _ => 4,
    }
}

fn decode_rot(m: &[u8], fmt: u8) -> [f32; 4] {
    match fmt {
        5 => quat48(m),
        6 => quat64ext(m),
        8 => quat64ext(m),
        _ => [rd_f32(m, 0), rd_f32(m, 4), rd_f32(m, 8), rd_f32(m, 12)],
    }
}

fn decode_times(body: &[u8], count: usize, fmt: u8) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    match fmt {
        0 => {
            for i in 0..count {
                out.push(rd_f32(body, i * 4));
            }
        }
        1 => {
            for i in 0..count {
                out.push(rd_u16(body, i * 2) as f32);
            }
        }
        2 => {
            for i in 0..count {
                out.push(body[i] as f32);
            }
        }
        _ => {}
    }
    out
}

impl Caf {
    pub fn parse(file: &[u8]) -> Option<Caf> {
        let cf = parse_cry_chunk_file(file)?;
        let mut start = 0i32;
        let mut end = 0i32;
        let mut secs_per_tick = 1.0 / 30.0;
        let mut controllers = Vec::new();

        for ch in &cf.chunks {
            let Some(p) = ch.payload(file) else { continue };
            // SpeedInfo 0xAAFC0002: m_nTicksPerFrame@0x08, m_fSecsPerTick@0x0C,
            // m_nStart@0x10, m_nEnd@0x14.
            if ch.chunk_type == 0xAAFC0002 && p.len() >= 0x18 {
                secs_per_tick = rd_f32(p, 0x0C);
                start = rd_u32(p, 0x10) as i32;
                end = rd_u32(p, 0x14) as i32;
                continue;
            }
            if ch.namespace() != 0xCCCC || ch.kind() != 0x0D || p.len() < 16 {
                continue;
            }
            let id = rd_u32(p, 0);
            let nrot = rd_u16(p, 4) as usize;
            let npos = rd_u16(p, 6) as usize;
            let rot_fmt = p[8];
            let rot_tfmt = p[9];
            let pos_fmt = p[10];
            let pos_info = p[11];
            let pos_tfmt = p[12];

            let mut o = 16usize;
            let mut rot_keys = Vec::with_capacity(nrot);
            for i in 0..nrot {
                let sz = rot_value_size(rot_fmt);
                rot_keys.push(decode_rot(&p[o + i * sz..], rot_fmt));
            }
            o += nrot * rot_value_size(rot_fmt);

            let rot_tsz = time_size(rot_tfmt);
            let rot_times = decode_times(&p[o..], nrot, rot_tfmt);
            o += nrot * rot_tsz;

            let psz = if pos_fmt == 2 || pos_fmt == 0 { 12 } else { 12 };
            let mut pos_keys = Vec::with_capacity(npos);
            for i in 0..npos {
                let b = &p[o + i * psz..];
                pos_keys.push([rd_f32(b, 0), rd_f32(b, 4), rd_f32(b, 8)]);
            }
            o += npos * psz;

            let pos_times = if pos_info != 0 {
                let tsz = time_size(pos_tfmt);
                let t = decode_times(&p[o..], npos, pos_tfmt);
                o += npos * tsz;
                t
            } else {
                rot_times.clone()
            };

            controllers.push(Controller {
                id,
                rot_keys,
                rot_times,
                pos_keys,
                pos_times,
                has_pos: npos > 0,
            });
        }

        Some(Caf { start, end, secs_per_tick, controllers })
    }

    pub fn duration_secs(&self) -> f32 {
        (self.end - self.start).max(0) as f32 * self.secs_per_tick
    }

    pub fn tick_from_seconds(&self, secs: f32) -> f32 {
        self.start as f32 + secs / self.secs_per_tick
    }

    /// Sample every controller at an absolute tick.
    pub fn sample_tick(&self, tick: f32) -> HashMap<u32, (Option<[f32; 3]>, [f32; 4])> {
        let mut out = HashMap::with_capacity(self.controllers.len());
        for c in &self.controllers {
            let q = sample_quat(&c.rot_times, &c.rot_keys, tick).unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let p = if c.has_pos {
                sample_vec(&c.pos_times, &c.pos_keys, tick)
            } else {
                None
            };
            out.insert(c.id, (p, q));
        }
        out
    }
}

fn find_key(times: &[f32], t: f32) -> (usize, f32) {
    if times.is_empty() {
        return (0, 0.0);
    }
    if t <= times[0] {
        return (0, 0.0);
    }
    let last = times.len() - 1;
    if t >= times[last] {
        return (last, 0.0);
    }
    for i in 1..=last {
        if t < times[i] {
            let span = times[i] - times[i - 1];
            let f = if span > 0.0 { (t - times[i - 1]) / span } else { 0.0 };
            return (i, f);
        }
    }
    (last, 0.0)
}

fn sample_vec(times: &[f32], keys: &[[f32; 3]], t: f32) -> Option<[f32; 3]> {
    if keys.is_empty() {
        return None;
    }
    let (i, f) = find_key(times, t);
    if i == 0 || f == 0.0 {
        return Some(keys[i]);
    }
    let a = keys[i - 1];
    let b = keys[i];
    Some([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f])
}

fn sample_quat(times: &[f32], keys: &[[f32; 4]], t: f32) -> Option<[f32; 4]> {
    if keys.is_empty() {
        return None;
    }
    let (i, f) = find_key(times, t);
    if i == 0 || f == 0.0 {
        return Some(normalize(keys[i]));
    }
    let mut a = keys[i - 1];
    let mut b = keys[i];
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
    }
    Some(normalize([
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
        a[3] + (b[3] - a[3]) * f,
    ]))
}

fn normalize(q: [f32; 4]) -> [f32; 4] {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l > 1e-12 {
        [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// How well a parsed `.caf`'s controllers bind to a skeleton.
///
/// Returns `(matched, total)` where `total` is the number of animation
/// controllers in the clip and `matched` is how many of their ids are present
/// in `skeleton.by_controller`. A `.caf` and a `.chr` share one id space:
/// `controllerID == crc32(boneName)` (see `notes/skinning.md` §4.1), so a
/// locomotion clip authored for a rig should match every controller.
pub fn controller_coverage(
    caf: &Caf,
    skeleton: &crate::skin::Skeleton,
) -> (usize, usize) {
    let total = caf.controllers.len();
    let matched = caf
        .controllers
        .iter()
        .filter(|c| skeleton.by_controller.contains_key(&c.id))
        .count();
    (matched, total)
}

/// Quick scan: just the controller ids in a `.caf` (for matching a skeleton).
pub fn controller_ids(file: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    let Some(cf) = parse_cry_chunk_file(file) else { return out };
    for ch in &cf.chunks {
        if ch.namespace() == 0xCCCC && ch.kind() == 0x0D {
            if let Some(p) = ch.payload(file) {
                if p.len() >= 4 {
                    out.push(rd_u32(p, 0));
                }
            }
        }
    }
    out
}

#[allow(dead_code)]
fn _use_reverse_bits() {
    let _ = reverse_bits(0);
}

// ---------------------------------------------------------------------------
// Clip database + locomotion selection
// ---------------------------------------------------------------------------
//
// A rig is identified by the gameplay-bone `controller` ids in its `Skeleton`
// (`skin::Skeleton::by_controller`). Every `.caf` clip carries the controller
// ids it animates (`controller_ids`). We keep a flat list of `ClipEntry`s
// tagged with a `Locomotion` class and, at play time, pick the clip of the
// wanted class that best overlaps the character's rig. This replaces the old
// single-keyword heuristic in the app.

/// Coarse gameplay class of a clip, derived from its path/name.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Locomotion {
    Idle,
    Walk,
    Run,
    Sprint,
    Attack,
    Death,
}

/// Classification precedence. More specific classes come first; `Idle` is
/// checked before `Death` so that "soldier_idle" is not caught by "die".
/// `classify`/`from_name` both scan in this order.
const CLASSIFY_ORDER: [Locomotion; 6] = [
    Locomotion::Sprint,
    Locomotion::Attack,
    Locomotion::Walk,
    Locomotion::Run,
    Locomotion::Idle,
    Locomotion::Death,
];

impl Locomotion {
    /// Substrings (lower-case) that identify this class in a clip name.
    pub fn keyword(self) -> &'static [&'static str] {
        match self {
            Locomotion::Idle => &["idle", "breath"],
            Locomotion::Walk => &["walk"],
            Locomotion::Run => &["run", "jog"],
            Locomotion::Sprint => &["sprint"],
            Locomotion::Attack => &["attack", "melee", "swing"],
            Locomotion::Death => &["death", "die", "dead"],
        }
    }

    /// Classify a name (usually a `.caf` file stem or path) case-insensitively.
    /// Returns `None` when no keyword matches.
    pub fn from_name(name: &str) -> Option<Locomotion> {
        let lower = name.to_ascii_lowercase();
        for loco in CLASSIFY_ORDER {
            if loco.keyword().iter().any(|k| lower.contains(k)) {
                return Some(loco);
            }
        }
        None
    }
}

/// Priority handed to a static pose clip (`*aimposes*`, `*lookposes*`, `*pose*`).
/// Such a `.caf` parses fine but holds a single bind-ish pose, so it must rank
/// below every genuine locomotion clip and is only played as a last resort.
pub const POSE_PRIORITY: i32 = -100;

/// True when `path` names a static aim/look pose rather than a looping
/// locomotion clip. Case-insensitive.
pub fn is_static_pose(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.contains("aimpose") || lower.contains("lookpose") || lower.contains("pose")
}

/// Base name of a `.caf`/`.$caf` path: no directories, no extension, lower-case.
fn clip_stem(path: &str) -> String {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path).to_ascii_lowercase();
    name.strip_suffix(".$caf")
        .or_else(|| name.strip_suffix(".caf"))
        .unwrap_or(&name)
        .to_string()
}

fn ends_with_token(stem: &str, token: &str) -> bool {
    stem == token || stem.ends_with(&format!("_{token}"))
}

/// Canonical forward/idle/sprint gait stems (`hero_idle`, `hero_walk_forward`,
/// `hero_run_forward`, `hero_sprint_run`, ...).
const GAIT_TOKENS: &[&str] = &[
    "idle",
    "walk_forward",
    "walk_fwd",
    "run_forward",
    "run_fwd",
    "sprint_run",
];

/// Canonical strafe/reverse gait stems (`hero_walk_left`, `hero_run_right`,
/// `hero_walk_backward`, ...).
const STRAFE_TOKENS: &[&str] = &[
    "walk_left",
    "walk_right",
    "run_left",
    "run_right",
    "walk_backward",
    "run_backward",
];

/// True when `path` lives in the player's canonical locomotion set
/// (`.../hero_male/test/...`), i.e. the clips authored for the human master
/// skeleton. The human rig shares controller ids with several zombie/freak
/// rigs, so this is what keeps the player's own gait clips winning on a tie.
fn is_player_locomotion(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.contains("hero_male/test/") || lower.contains("hero_male\\test\\")
}

/// Rank a clip path as a locomotion source. Higher is better:
///
///   * `70` canonical player gait in `hero_male/test/` with a `hero_` stem —
///     `hero_idle`, `hero_walk_forward`, `hero_run_forward`, `hero_sprint_run`,
///     ...,
///   * `60` canonical player strafe with a `hero_` stem, or a shorter player
///     gait stem (`idle`, `walk_forward`, ...) without the `hero_` prefix,
///   * `50` canonical player strafe without the `hero_` prefix,
///   * `30` canonical gait for any other rig (`zombie_male/wander_idle`,
///     `hero_male/npc/run_forward`, ...),
///   * `20` canonical strafe for any other rig,
///   * `15`/`10` other clips carrying a locomotion keyword (`hero_run_injured`,
///     `hero_sprint_start`, `crouch_idle`, ...),
///   * `0`  non-locomotion clips,
///   * [`POSE_PRIORITY`] static `*aimposes*`/`*lookposes*`/`*pose*` clips.
///
/// [`ClipSet::pick`] uses this as its primary key, ahead of controller overlap,
/// so a genuine (if slightly lower-overlap) gait still beats a static pose, and
/// the player's own canonical `hero_` gaits beat look-alike variants
/// (`crouch_idle`) and zombie/freak clips that share controller ids.
pub fn prefer_locomotion(path: &str) -> i32 {
    if is_static_pose(path) {
        return POSE_PRIORITY;
    }
    let stem = clip_stem(path);
    let player = is_player_locomotion(path);
    let hero_named = stem.starts_with("hero_");
    let gait = GAIT_TOKENS.iter().any(|g| ends_with_token(&stem, g));
    let strafe = STRAFE_TOKENS.iter().any(|s| ends_with_token(&stem, s));
    let lower = path.to_ascii_lowercase();
    let keyword = ["idle", "walk", "run", "sprint"].iter().any(|k| lower.contains(k));
    match (player, gait, strafe, hero_named, keyword) {
        (true, true, _, true, _) => 70,
        (true, true, _, false, _) => 60,
        (true, _, true, true, _) => 60,
        (true, _, true, false, _) => 50,
        (false, true, _, _, _) => 30,
        (false, _, true, _, _) => 20,
        (true, _, _, _, true) => 15,
        (false, _, _, _, true) => 10,
        _ => 0,
    }
}

/// Backwards-compatible alias for [`prefer_locomotion`].
pub fn clip_priority(path: &str) -> i32 {
    prefer_locomotion(path)
}

/// One clip in the database: its path, the controller ids it animates, and its
/// gameplay class.
pub struct ClipEntry {
    pub path: String,
    pub controllers: Vec<u32>,
    pub locomotion: Locomotion,
}

/// A flat collection of clips that can be filtered by locomotion and matched
/// against a rig's controller ids.
#[derive(Default)]
pub struct ClipSet {
    entries: Vec<ClipEntry>,
}

/// Overlap of a clip against a rig: `(matched_count, matched / total)`.
///
/// Factored out so it can be unit-tested without a full `.caf`/`.chr`.
fn clip_overlap(controllers: &[u32], skeleton: &crate::skin::Skeleton) -> (usize, f32) {
    if controllers.is_empty() {
        return (0, 0.0);
    }
    let matched = controllers
        .iter()
        .filter(|id| skeleton.by_controller.contains_key(id))
        .count();
    (matched, matched as f32 / controllers.len() as f32)
}

impl ClipSet {
    pub fn new() -> Self {
        ClipSet { entries: Vec::new() }
    }

    pub fn add(&mut self, path: impl Into<String>, controllers: Vec<u32>, locomotion: Locomotion) {
        self.entries.push(ClipEntry { path: path.into(), controllers, locomotion });
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Pick the entry of `want` that is the best animation source. Entries are
    /// ranked first by [`prefer_locomotion`] (genuine gait > decorated variant >
    /// static pose), then by controller overlap with the skeleton (highest match
    /// count, then highest match ratio), and finally by shortest path so the
    /// choice is deterministic regardless of scan order. Returns `None` if no
    /// entry of that class shares any controller id with the skeleton.
    pub fn pick(
        &self,
        skeleton: &crate::skin::Skeleton,
        want: Locomotion,
    ) -> Option<&ClipEntry> {
        let mut best: Option<(&ClipEntry, i32, usize, f32)> = None;
        for entry in self.entries.iter().filter(|e| e.locomotion == want) {
            let (count, ratio) = clip_overlap(&entry.controllers, skeleton);
            if count == 0 {
                continue;
            }
            let priority = prefer_locomotion(&entry.path);
            let better = match best {
                None => true,
                Some((be, bp, bc, br)) => {
                    priority > bp
                        || (priority == bp
                            && (count > bc
                                || (count == bc
                                    && (ratio > br
                                        || (ratio == br && entry.path.len() < be.path.len())))))
                }
            };
            if better {
                best = Some((entry, priority, count, ratio));
            }
        }
        best.map(|(entry, _, _, _)| entry)
    }

    /// Classify a `.caf` path by keyword (case-insensitive). Falls back to
    /// `Idle` when nothing matches, so callers always get a usable class.
    pub fn classify(path: &str) -> Locomotion {
        Locomotion::from_name(path).unwrap_or(Locomotion::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skin::Skeleton;

    fn skel(ids: &[u32]) -> Skeleton {
        let mut s = Skeleton::default();
        for (i, id) in ids.iter().enumerate() {
            s.by_controller.insert(*id, i);
        }
        s
    }

    #[test]
    fn classify_real_style_paths() {
        assert_eq!(
            ClipSet::classify("animations/characters/zombie_male/melee/attack_light_01.caf"),
            Locomotion::Attack
        );
        assert_eq!(
            ClipSet::classify("animations/characters/human/locomotion/run_forward.caf"),
            Locomotion::Run
        );
        assert_eq!(
            ClipSet::classify("animations/characters/human/locomotion/sprint_forward.caf"),
            Locomotion::Sprint
        );
        assert_eq!(
            ClipSet::classify("animations/characters/human/locomotion/idle_relaxed.caf"),
            Locomotion::Idle
        );
        assert_eq!(
            ClipSet::classify("animations/characters/human/walk/walk_fwd.caf"),
            Locomotion::Walk
        );
        assert_eq!(
            ClipSet::classify("animations/characters/human/death/death_01.caf"),
            Locomotion::Death
        );
        // Case-insensitive.
        assert_eq!(
            ClipSet::classify("Animations/Characters/Human/Locomotion/Sprint_Fwd.CAF"),
            Locomotion::Sprint
        );
        // Unknown paths fall back to Idle.
        assert_eq!(ClipSet::classify("animations/misc/whatever.caf"), Locomotion::Idle);
    }

    #[test]
    fn from_name_maps_keywords() {
        assert_eq!(Locomotion::from_name("breathing"), Some(Locomotion::Idle));
        assert_eq!(Locomotion::from_name("jog_forward"), Some(Locomotion::Run));
        assert_eq!(Locomotion::from_name("melee_swing"), Some(Locomotion::Attack));
        assert_eq!(Locomotion::from_name("turn_left"), None);
        // "soldier" contains "die" but is classified as Idle, not Death.
        assert_eq!(Locomotion::from_name("soldier_idle"), Some(Locomotion::Idle));
    }

    #[test]
    fn add_and_pick_best_overlap() {
        let mut set = ClipSet::new();
        assert!(set.is_empty());
        set.add("a/run_bad.caf", vec![1, 2, 3], Locomotion::Run);
        set.add("b/run_good.caf", vec![10, 20, 30], Locomotion::Run);
        set.add("c/run_worse.caf", vec![10, 999], Locomotion::Run);
        set.add("d/idle.caf", vec![10], Locomotion::Idle);
        assert_eq!(set.len(), 4);

        let sk = skel(&[10, 20, 30, 1]);
        let got = set.pick(&sk, Locomotion::Run).expect("run clip");
        assert_eq!(got.path, "b/run_good.caf");
        assert_eq!(got.controllers, vec![10, 20, 30]);
    }

    #[test]
    fn pick_ratio_breaks_count_tie() {
        let mut set = ClipSet::new();
        set.add("small.caf", vec![1], Locomotion::Attack); // 1/1 overlap
        set.add("big.caf", vec![1, 2, 3, 4], Locomotion::Attack); // 1/4 overlap
        let sk = skel(&[1]);
        assert_eq!(set.pick(&sk, Locomotion::Attack).unwrap().path, "small.caf");
    }

    #[test]
    fn pick_none_when_no_match() {
        let mut set = ClipSet::new();
        set.add("idle.caf", vec![7], Locomotion::Idle);
        let sk = skel(&[7]);
        // No entries of the wanted class.
        assert!(set.pick(&sk, Locomotion::Sprint).is_none());
        // Entry exists but shares no controller id with the rig.
        let stranger = skel(&[999]);
        assert!(set.pick(&stranger, Locomotion::Idle).is_none());
    }

    // -- scoring / static-pose preference -----------------------------------

    #[test]
    fn clip_priority_prefers_gaits_over_static_poses() {
        let pose = clip_priority("animations/characters/hero_male/aim/relaxed_idleaimposes_pistol.caf");
        let look = clip_priority("animations/characters/hero_male/lookposes/lookposes_sdk_head.$caf");
        let idle = clip_priority("animations/characters/hero_male/test/hero_idle.$caf");
        let walk = clip_priority("animations/characters/hero_male/test/hero_walk_forward.$caf");
        let run = clip_priority("animations/characters/hero_male/test/hero_run_forward.$caf");
        let sprint = clip_priority("animations/characters/hero_male/test/hero_sprint_run.$caf");
        let strafe = clip_priority("animations/characters/hero_male/test/hero_run_left.$caf");
        let decorated = clip_priority("animations/characters/hero_male/test/hero_run_forward_injured.$caf");

        assert!(is_static_pose(".../relaxed_idleaimposes_pistol.caf"));
        assert!(is_static_pose(".../lookposes_sdk_head.$caf"));
        assert_eq!(pose, POSE_PRIORITY);
        assert_eq!(look, POSE_PRIORITY);
        assert!(run > decorated, "canonical run must outrank decorated run");
        assert!(run > strafe, "forward gait must outrank a strafe");
        assert!(strafe > decorated, "canonical strafe must outrank decorated run");
        assert!(decorated > pose, "any gait must outrank a static pose");
        assert!(idle > pose && walk > pose && sprint > pose);
    }

    #[test]
    fn prefer_locomotion_ranks_player_gaits_first() {
        let hero_run = prefer_locomotion("animations/characters/hero_male/test/hero_run_forward.$caf");
        let hero_idle = prefer_locomotion("animations/characters/hero_male/test/hero_idle.$caf");
        let hero_strafe = prefer_locomotion("animations/characters/hero_male/test/hero_run_left.$caf");
        let hero_injured =
            prefer_locomotion("animations/characters/hero_male/test/hero_run_forward_injured.$caf");
        let npc_run = prefer_locomotion("animations/characters/hero_male/npc/run_forward.$caf");
        let zombie_idle = prefer_locomotion("animations/characters/zombie_male/wander_idle.$caf");
        let pose = prefer_locomotion("animations/characters/hero_male/aim/relaxed_idleaimposes_pistol.caf");
        let look = prefer_locomotion("animations/characters/hero_male/lookposes/lookposes_sdk_head.$caf");

        assert_eq!(pose, POSE_PRIORITY);
        assert_eq!(look, POSE_PRIORITY);
        // Player canonical gaits beat the same gait on another rig...
        assert!(hero_run > npc_run, "player run must beat npc run");
        assert!(hero_idle > zombie_idle, "player idle must beat zombie idle");
        // ...and beat decorated variants and static poses.
        assert!(hero_run > hero_injured);
        assert!(hero_strafe > hero_injured);
        assert!(hero_injured > pose);
        // The alias keeps the old public API working.
        assert_eq!(clip_priority("x/hero_idle.$caf"), prefer_locomotion("x/hero_idle.$caf"));
    }

    /// A static pose with a perfect controller overlap must still lose to a
    /// genuine gait whose overlap is lower but non-zero.
    #[test]
    fn pick_prefers_gait_over_better_overlapping_pose() {
        let mut set = ClipSet::new();
        set.add("aim/relaxed_idleaimposes_pistol.caf", vec![1, 2, 3, 4], Locomotion::Idle);
        set.add("test/hero_idle.$caf", vec![1], Locomotion::Idle);
        let sk = skel(&[1, 2, 3, 4]);
        let got = set.pick(&sk, Locomotion::Idle).expect("idle pick");
        assert!(
            got.path.ends_with("hero_idle.$caf"),
            "static pose beat genuine gait: {}",
            got.path
        );
    }

    // -- real hero ./$caf assets (skipped silently when absent) --------------

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

    const HERO_DIR: &str = "animations/animations/characters/hero_male/test";
    const HERO_RUN: &str = "animations/animations/characters/hero_male/test/hero_run_forward.$caf";
    const HERO_IDLE: &str = "animations/animations/characters/hero_male/test/hero_idle.$caf";
    const HERO_WALK: &str = "animations/animations/characters/hero_male/test/hero_walk_forward.$caf";
    const HERO_SPRINT: &str = "animations/animations/characters/hero_male/test/hero_sprint_run.$caf";
    const MASTER_CHR: &str = "characters/characters/male/human/skeleton/skeleton.chr";

    #[test]
    fn hero_dollar_caf_parses_and_binds_master_skeleton() {
        let (Some(caf_path), Some(chr_path)) = (extracted(HERO_RUN), extracted(MASTER_CHR)) else {
            return; // real game data absent; skip silently
        };
        let caf = Caf::parse(&std::fs::read(caf_path).expect("read .$caf"))
            .expect("Caf::parse on hero_run_forward.$caf");
        assert!(caf.controllers.len() > 4, "too few controllers: {}", caf.controllers.len());
        assert!(caf.end > caf.start, "clip window is empty");

        let skeleton = crate::skin::parse_skeleton(&std::fs::read(chr_path).expect("read .chr"))
            .expect("master skeleton");
        let (matched, total) = controller_coverage(&caf, &skeleton);
        eprintln!("PROBE hero_run_forward.$caf coverage {matched}/{total} window {}..{} controllers {}", caf.start, caf.end, caf.controllers.len());
        assert!(total > 0, "no controllers in clip");
        assert!(matched > 0, "hero_run_forward.$caf matched no master bone ({matched}/{total})");

        // Sampling at two ticks must produce different controller rotations.
        let a = caf.sample_tick(caf.start as f32);
        let b = caf.sample_tick((caf.start + 15) as f32);
        let moved = a.iter().any(|(id, (_, q))| {
            b.get(id).map_or(false, |(_, q2)| {
                (q[0] - q2[0]).abs()
                    + (q[1] - q2[1]).abs()
                    + (q[2] - q2[2]).abs()
                    + (q[3] - q2[3]).abs()
                    > 1e-4
            })
        });
        assert!(moved, "hero_run_forward.$caf sampled identically at two ticks");
    }

    #[test]
    fn hero_locomotion_names_classify_to_expected_gait() {
        assert_eq!(Locomotion::from_name(HERO_IDLE), Some(Locomotion::Idle));
        assert_eq!(Locomotion::from_name(HERO_WALK), Some(Locomotion::Walk));
        assert_eq!(Locomotion::from_name(HERO_RUN), Some(Locomotion::Run));
        assert_eq!(Locomotion::from_name(HERO_SPRINT), Some(Locomotion::Sprint));
        assert_eq!(
            Locomotion::from_name("animations/characters/hero_male/test/hero_walk_left.$caf"),
            Some(Locomotion::Walk)
        );
        assert_eq!(crate::locomotion::Gait::from_name(HERO_IDLE), Some(crate::locomotion::Gait::Idle));
        assert_eq!(crate::locomotion::Gait::from_name(HERO_RUN), Some(crate::locomotion::Gait::Run));
    }

    /// Build a `ClipSet` from the *real* hero locomotion `.$caf` files and check
    /// each gait resolves to its canonical clip.
    #[test]
    fn hero_clipset_picks_canonical_clip_per_gait() {
        let Some(chr_path) = extracted(MASTER_CHR) else { return };
        let Some(dir) = extracted(HERO_DIR) else { return };
        let skeleton =
            crate::skin::parse_skeleton(&std::fs::read(chr_path).expect("read")).expect("skeleton");

        // Feed every hero `.$caf` whose controllers match the rig, exactly as
        // `character::collect_clips` would after the scan fix.
        let mut set = ClipSet::new();
        let mut seen = 0usize;
        for e in std::fs::read_dir(dir).expect("read hero dir").flatten() {
            let p = e.path();
            if !p.to_string_lossy().ends_with(".$caf") {
                continue;
            }
            let Ok(bytes) = std::fs::read(&p) else { continue };
            let ids = controller_ids(&bytes);
            if ids.is_empty() {
                continue;
            }
            seen += 1;
            let path = format!("animations/animations/characters/hero_male/test/{}", p.file_name().unwrap().to_string_lossy());
            set.add(path, ids, ClipSet::classify(&format!("{}", p.file_name().unwrap().to_string_lossy())));
        }
        assert!(seen > 0, "no hero .$caf clips parsed");

        let check = |want, suffix: &str| {
            let got = set.pick(&skeleton, want).unwrap_or_else(|| panic!("no {want:?} clip"));
            assert!(
                got.path.ends_with(suffix),
                "{want:?} picked {} (want *{suffix})",
                got.path
            );
        };
        for want in [Locomotion::Idle, Locomotion::Walk, Locomotion::Run, Locomotion::Sprint] {
            if let Some(e) = set.pick(&skeleton, want) {
                eprintln!("PROBE gait {want:?} -> {} (priority {}, {} controllers)", e.path, clip_priority(&e.path), e.controllers.len());
            }
        }
        check(Locomotion::Idle, "hero_idle.$caf");
        check(Locomotion::Walk, "hero_walk_forward.$caf");
        check(Locomotion::Run, "hero_run_forward.$caf");
        check(Locomotion::Sprint, "hero_sprint_run.$caf");
    }
}
