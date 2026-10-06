//! Animation/speed matching (foot-slide-free locomotion) + root motion.
//!
//! State of Decay drives locomotion from the animation graph: a clip plays at
//! a rate proportional to ground speed so the planted foot does not slide, and
//! the root/hip bone's animated translation is extracted as *root motion* and
//! fed back into the character transform. This module is the native equivalent
//! of both halves.
//!
//! Speeds come from [`crate::tuning::MovementTuning`] (shipped `player.lua`);
//! clip classes come from [`crate::anim::Locomotion`].

use crate::anim::{Caf, Locomotion};
use crate::skin::{Mat4, Skeleton};
use crate::tuning::MovementTuning;
use std::f32::consts::PI;

/// Gameplay gait selected from a ground speed.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Gait {
    Idle,
    Creep,
    Walk,
    Run,
    Sprint,
}

fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + (b - a) * f
}

impl Gait {
    /// Classify a ground speed (m/s) into a gait.
    ///
    /// The boundaries are the shipped gait speeds, placed using `player.lua`'s
    /// controller thresholds: the creep/walk transition sits
    /// `threshold_creep_walk` of the way from `creep` to `walk`, and the
    /// walk/run transition `threshold_walk_run` of the way from `walk` to
    /// `run`. Anything at or above `sprint` sprints.
    ///
    /// * `speed <= 0` (and non-finite input) → [`Gait::Idle`]
    /// * below the creep/walk boundary → [`Gait::Creep`]
    /// * below the walk/run boundary → [`Gait::Walk`]
    /// * below `sprint` → [`Gait::Run`]
    /// * `sprint` and above → [`Gait::Sprint`]
    pub fn from_speed(speed: f32, t: &MovementTuning) -> Gait {
        if !(speed > 0.0) {
            return Gait::Idle;
        }
        let creep_walk = lerp(t.creep, t.walk, t.threshold_creep_walk);
        let walk_run = lerp(t.walk, t.run, t.threshold_walk_run);
        if speed < creep_walk {
            Gait::Creep
        } else if speed < walk_run {
            Gait::Walk
        } else if speed < t.sprint {
            Gait::Run
        } else {
            Gait::Sprint
        }
    }

    /// Nominal ground speed an animation of this gait is authored for (m/s).
    /// `Idle` is stationary (`0.0`).
    pub fn base_speed(self, t: &MovementTuning) -> f32 {
        match self {
            Gait::Idle => 0.0,
            Gait::Creep => t.creep,
            Gait::Walk => t.walk,
            Gait::Run => t.run,
            Gait::Sprint => t.sprint,
        }
    }

    /// Map an [`Locomotion`] clip class to a gait. Attack/Death clips have no
    /// locomotion gait.
    pub fn from_locomotion(loco: Locomotion) -> Option<Gait> {
        match loco {
            Locomotion::Idle => Some(Gait::Idle),
            Locomotion::Walk => Some(Gait::Walk),
            Locomotion::Run => Some(Gait::Run),
            Locomotion::Sprint => Some(Gait::Sprint),
            Locomotion::Attack | Locomotion::Death => None,
        }
    }

    /// Map a clip name/path (e.g. a `.caf` stem) to a gait, reusing
    /// [`Locomotion::from_name`]'s keyword classification.
    pub fn from_name(name: &str) -> Option<Gait> {
        Locomotion::from_name(name).and_then(Gait::from_locomotion)
    }
}

/// Gait plus the animation playback rate that keeps the feet planted.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct SpeedMatch {
    pub gait: Gait,
    pub playback_rate: f32,
}

/// Match a ground speed to a gait and a clip playback rate.
///
/// `playback_rate = clamp(speed / base_speed, 0.5..=2.0)`; `Idle` (and any
/// non-positive/non-finite speed) always plays at `1.0`. Never panics.
pub fn match_speed(speed: f32, t: &MovementTuning) -> SpeedMatch {
    let gait = Gait::from_speed(speed, t);
    let base = gait.base_speed(t);
    let playback_rate = if gait == Gait::Idle || !(base > 0.0) || !base.is_finite() {
        1.0
    } else {
        (speed / base).clamp(0.5, 2.0)
    };
    SpeedMatch { gait, playback_rate }
}

/// Root/hip bone motion between two clip ticks.
///
/// `translation` is the delta in CryEngine world axes (Z up); `yaw` is the
/// shortest signed rotation delta about Z, in radians.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct RootMotion {
    pub translation: [f32; 3],
    pub yaw: f32,
}

impl RootMotion {
    /// No motion — also the documented fallback when a root cannot be found.
    pub const ZERO: RootMotion = RootMotion { translation: [0.0, 0.0, 0.0], yaw: 0.0 };
}

/// Index of the skeleton's identifiable root/hip bone, if any.
///
/// Bones are matched case-insensitively by name, in this order: `root`,
/// `bip01`, `hip`, `pelvis`. A skeleton with none of these (including
/// [`Skeleton::default`]) has no identifiable root.
pub fn root_bone(skeleton: &Skeleton) -> Option<usize> {
    const WANT: [&str; 4] = ["root", "bip01", "hip", "pelvis"];
    for want in WANT {
        if let Some((i, _)) = skeleton
            .bones
            .iter()
            .enumerate()
            .find(|(_, b)| b.name.to_ascii_lowercase().contains(want))
        {
            return Some(i);
        }
    }
    None
}

/// Yaw of a posed bone about Z (the local +X axis maps to `(m[0], m[4], m[8])`).
fn yaw_of(m: &Mat4) -> f32 {
    m.0[4].atan2(m.0[0])
}

/// Normalise an angle to `(-PI, PI]`; non-finite input becomes `0.0`.
fn wrap_pi(mut a: f32) -> f32 {
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    if a.is_finite() {
        a
    } else {
        0.0
    }
}

/// Extract the root/hip bone's world delta between two absolute clip ticks.
///
/// Both ticks are sampled with [`Caf::sample_tick`] and posed with
/// [`Skeleton::pose`]; the translation is `posed_to - posed_from` in CryEngine
/// world axes (Z up) and `yaw` is the shortest signed Z rotation delta.
///
/// Returns [`RootMotion::ZERO`] when the skeleton has no identifiable root/hip
/// bone or the clip yields no pose. Never panics.
pub fn root_motion(caf: &Caf, skeleton: &Skeleton, tick_from: f32, tick_to: f32) -> RootMotion {
    let root = match root_bone(skeleton) {
        Some(i) => i,
        None => return RootMotion::ZERO,
    };
    let a = skeleton.pose(&caf.sample_tick(tick_from));
    let b = skeleton.pose(&caf.sample_tick(tick_to));
    let (from, to) = match (a.get(root), b.get(root)) {
        (Some(from), Some(to)) => (from, to),
        _ => return RootMotion::ZERO,
    };
    let (ta, tb) = (from.translation(), to.translation());
    let yaw = wrap_pi(yaw_of(to) - yaw_of(from));
    RootMotion {
        translation: [tb[0] - ta[0], tb[1] - ta[1], tb[2] - ta[2]],
        yaw,
    }
}

/// Apply extracted root motion to a character transform `(pos, yaw)`.
///
/// `rm.translation` is rotated by `yaw` (Z-up, standard positive direction),
/// added to `pos`, and `rm.yaw` is added to `yaw` (normalised to `(-PI, PI]`).
pub fn apply_root_motion(pos: [f32; 3], yaw: f32, rm: &RootMotion) -> ([f32; 3], f32) {
    let (s, c) = yaw.sin_cos();
    let t = rm.translation;
    (
        [
            pos[0] + t[0] * c - t[1] * s,
            pos[1] + t[0] * s + t[1] * c,
            pos[2] + t[2],
        ],
        wrap_pi(yaw + rm.yaw),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::Caf;
    use crate::skin::Bone;
    use crate::tuning::default_movement;

    fn empty_caf() -> Caf {
        Caf { start: 0, end: 30, secs_per_tick: 1.0 / 30.0, controllers: Vec::new() }
    }

    /// A skeleton with one named root bone and no animation data.
    fn skel_with_root(name: &str) -> Skeleton {
        let mut s = Skeleton::default();
        s.bones.push(Bone {
            name: name.to_string(),
            parent: -1,
            controller_id: 42,
            bind_local: Mat4::identity(),
            w2b: Mat4::identity(),
        });
        s.by_controller.insert(42, 0);
        s
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn from_speed_boundaries() {
        let t = default_movement();
        assert_eq!(Gait::from_speed(0.0, &t), Gait::Idle);
        assert_eq!(Gait::from_speed(0.8, &t), Gait::Creep);
        assert_eq!(Gait::from_speed(1.66, &t), Gait::Walk);
        assert_eq!(Gait::from_speed(4.0, &t), Gait::Run);
        assert_eq!(Gait::from_speed(6.8, &t), Gait::Sprint);
        // Just above/below the shipped gait speeds.
        assert_eq!(Gait::from_speed(0.79, &t), Gait::Creep);
        assert_eq!(Gait::from_speed(6.81, &t), Gait::Sprint);
    }

    #[test]
    fn from_speed_handles_negative_and_nan() {
        let t = default_movement();
        assert_eq!(Gait::from_speed(-3.0, &t), Gait::Idle);
        assert_eq!(Gait::from_speed(f32::NAN, &t), Gait::Idle);
        // Negative tuning values must not panic either.
        let bad = MovementTuning { creep: -1.0, ..t };
        assert_eq!(Gait::from_speed(-0.5, &bad), Gait::Idle);
    }

    #[test]
    fn match_speed_rates() {
        let t = default_movement();
        // speed == base speed → 1.0, for every moving gait.
        assert_eq!(match_speed(t.creep, &t), SpeedMatch { gait: Gait::Creep, playback_rate: 1.0 });
        assert_eq!(match_speed(t.walk, &t), SpeedMatch { gait: Gait::Walk, playback_rate: 1.0 });
        assert_eq!(match_speed(t.run, &t), SpeedMatch { gait: Gait::Run, playback_rate: 1.0 });
        assert_eq!(match_speed(t.sprint, &t), SpeedMatch { gait: Gait::Sprint, playback_rate: 1.0 });
        // Idle / zero always 1.0.
        assert_eq!(match_speed(0.0, &t), SpeedMatch { gait: Gait::Idle, playback_rate: 1.0 });
        assert_eq!(match_speed(f32::NAN, &t), SpeedMatch { gait: Gait::Idle, playback_rate: 1.0 });
        // Clamped at both extremes.
        let slow = match_speed(t.creep * 0.1, &t);
        assert_eq!(slow.gait, Gait::Creep);
        assert_eq!(slow.playback_rate, 0.5);
        let fast = match_speed(1000.0, &t);
        assert_eq!(fast.gait, Gait::Sprint);
        assert_eq!(fast.playback_rate, 2.0);
    }

    #[test]
    fn gait_from_clip_names() {
        assert_eq!(Gait::from_name("idle_relaxed"), Some(Gait::Idle));
        assert_eq!(Gait::from_name("walk_forward"), Some(Gait::Walk));
        assert_eq!(Gait::from_name("run_fwd"), Some(Gait::Run));
        assert_eq!(Gait::from_name("sprint_01"), Some(Gait::Sprint));
        assert_eq!(Gait::from_name("melee_attack"), None);
    }

    #[test]
    fn root_motion_zero_fallback() {
        let caf = empty_caf();
        // No skeleton bones at all → no identifiable root.
        let empty = Skeleton::default();
        assert!(root_bone(&empty).is_none());
        assert_eq!(root_motion(&caf, &empty, 0.0, 10.0), RootMotion::ZERO);

        // A named root exists, but the clip has no controllers: the bind pose is
        // identical at both ticks, so the delta is zero.
        let sk = skel_with_root("Bip01_Root");
        assert_eq!(root_bone(&sk), Some(0));
        assert_eq!(root_motion(&caf, &sk, 0.0, 30.0), RootMotion::ZERO);
        // Non-finite ticks do not panic.
        let _ = root_motion(&caf, &sk, f32::NAN, f32::INFINITY);
    }

    #[test]
    fn apply_root_motion_rotates_and_adds_yaw() {
        let rm = RootMotion { translation: [1.0, 0.0, 0.0], yaw: 0.0 };
        let (p, y) = apply_root_motion([10.0, 20.0, 30.0], 0.0, &rm);
        assert!(approx(p[0], 11.0) && approx(p[1], 20.0) && approx(p[2], 30.0));
        assert!(approx(y, 0.0));

        // +90° yaw rotates +X into +Y.
        let (p, y) = apply_root_motion([0.0, 0.0, 0.0], PI / 2.0, &rm);
        assert!(approx(p[0], 0.0) && approx(p[1], 1.0) && approx(p[2], 0.0));
        assert!(approx(y, PI / 2.0));

        // Yaw deltas add and wrap into (-PI, PI].
        let turn = RootMotion { translation: [0.0, 0.0, 0.0], yaw: 3.0 * PI / 2.0 };
        let (_, y) = apply_root_motion([0.0, 0.0, 0.0], 0.0, &turn);
        assert!(approx(y, -PI / 2.0));
    }
}
