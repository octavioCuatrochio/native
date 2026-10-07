//! Expensive locomotion candidate cost (`CAnimationGraph` state selection).
//!
//! When the animation graph has `expensiveLocomotionSelection="1"` (default)
//! and the character is moving, `FUN_007bd3a0` (0x7bd3a0) scores every
//! candidate state with `FUN_007cb5f0` (0x7cb5f0) and keeps the minimum in
//! `FUN_007cbaf0` (0x7cbaf0). This is what orders `RelaxedWalkNW` against
//! `RelaxedRunNW`, whose graph `SelectWhen` criteria are identical.
//!
//! The score is a weighted distance between the candidate animation's motion
//! profile (`CandidateMotion`, the engine's per-state animation segment struct,
//! +0x00..+0x60) and the character's current movement (`DesiredMotion`, built
//! by `FUN_007cb460` from `CAnimatedCharacter`). The weighted sums, range
//! clamping and per-field weights are recovered from the two functions; see
//! `notes/ag-locomotion-cost.md` for the address-by-address derivation.
//!
//! Weights (resolved from the image):
//! * speed range (candidate +0x48/+0x4c): `1000.0` (`0x172b138`)
//! * motion-profile ranges (+0x08..+0x1c): `50.0` (`0x1722354`)
//! * angular ranges (+0x20..+0x34, and the two derived angles): `0.05`
//!   (`0x171e200`)
//! * transition bonus: `100.0` (`0x171e218`) scaled by the same `0.05`
//! * tick → seconds: `1e-5` (`0x171e0e8`)
//!
//! `FUN_007cbaf0` additionally applies a small token-transition term (e.g.
//! `WALK`→`I2M_` gets `+100`) on top of this cost when the graph's current
//! state is known; that layer is documented but not part of
//! [`locomotion_cost`].

/// Weight on the candidate's authored movement-speed range (`+0x48/+0x4c`).
/// Recovered from `0x172b138`.
pub const SPEED_WEIGHT: f32 = 1000.0;
/// Weight on the 3 motion-profile ranges (`+0x08..+0x1c`). Recovered from
/// `0x1722354`.
pub const PROFILE_WEIGHT: f32 = 50.0;
/// Weight on the angular / derived-phase ranges. Recovered from `0x171e200`.
pub const ANGLE_WEIGHT: f32 = 0.05;
/// 64-bit timestamp tick → seconds. Recovered from `0x171e0e8`.
pub const TICKS_TO_SECONDS: f32 = 1e-5;
/// Transition bonus magnitude. Recovered from `0x171e218`.
pub const TRANSITION_BONUS: f32 = 100.0;
/// Degrees at which the two derived-angle comparisons wrap. `0x172022c`.
pub const ANGLE_WRAP: f32 = 180.0;
/// Full turn used by the wrap. `0x1720230`.
pub const ANGLE_FULL: f32 = 360.0;

/// Per-candidate animation-motion profile.
///
/// Layout mirrors the engine's per-state animation segment struct used by
/// `FUN_007cb5f0` (0x7cb5f0). Each `(lo, hi)` is a `(min, max)` range; the
/// score is the distance of the character's current value from that range.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CandidateMotion {
    /// Eight motion-profile ranges at `+0x00..+0x3c`. `profile[0]` carries
    /// weight 0 (it vanishes in the engine's sum); `profile[1..=3]` weight
    /// [`PROFILE_WEIGHT`]; `profile[4..=6]` weight [`ANGLE_WEIGHT`];
    /// `profile[7]` is the `(travel_yaw - move_yaw)` corridor.
    pub profile: [(f32, f32); 8],
    /// Range at `+0x40/+0x44`, compared against `travel_yaw - angle_deviation`.
    pub turn_extra: (f32, f32),
    /// Authored movement-speed range at `+0x48/+0x4c`, weighted
    /// [`SPEED_WEIGHT`].
    pub speed: (f32, f32),
    /// Scalar at `+0x58`, scales the outgoing-transition blend factor.
    pub blend_out: f32,
    /// Scalar at `+0x5c`, scales the incoming-transition blend factor.
    pub blend_in: f32,
    /// 4-character motion token at `+0x60` (`IDLE`, `WALK`, `RUN_`, `I2W_`,
    /// `I2M_`, `ISTP`, `IROT`). Only used by the `FUN_007cbaf0` token term.
    pub token: [u8; 4],
}

impl CandidateMotion {
    /// A candidate whose only populated data is its authored speed range.
    /// This is the form used when a state's movement profile is derived from
    /// its locomotion clip's root motion.
    pub fn from_speed(min: f32, max: f32) -> Self {
        CandidateMotion {
            speed: (min, max),
            token: *b"    ",
            ..Default::default()
        }
    }

    /// Convenience for a clip authored at one speed.
    pub fn from_clip_speed(speed: f32) -> Self {
        Self::from_speed(speed, speed)
    }
}

/// The character's current movement, as assembled by `FUN_007cb460`
/// (0x7cb460) from `CAnimatedCharacter`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DesiredMotion {
    /// `m[0]` (`+0x00`): movement-track phase.
    pub phase: f32,
    /// `m[1]` (`+0x04`): `CAnimatedCharacter+0x860`, the current movement
    /// magnitude (m/s).
    pub actual_speed: f32,
    /// `m[2]` (`+0x08`): magnitude of the primary movement vector.
    pub move_magnitude: f32,
    /// `m[3]` (`+0x0c`): magnitude of the secondary (strafe) vector.
    pub strafe_magnitude: f32,
    /// `m[4]` (`+0x10`): primary movement yaw (degrees).
    pub move_yaw: f32,
    /// `m[5]` (`+0x14`): travel yaw (degrees).
    pub travel_yaw: f32,
    /// `m[6]` (`+0x18`): movement angle deviation (degrees).
    pub angle_deviation: f32,
    /// `m[7]` (`+0x1c`): requested locomotion speed from the movement
    /// controller for the current state (`CAnimatedCharacter+0x2c` →
    /// vtable `+0x68`). This is the dominant term.
    pub requested_speed: f32,
    /// `param_4` of `FUN_007cb5f0`: whether the candidate is the graph's
    /// current state.
    pub is_current: bool,
    /// `CAnimatedCharacter+0x310` (64-bit current tick).
    pub now_ticks: i64,
    /// `CAnimatedCharacter+0x318`: tick the incoming transition started
    /// (0 = none).
    pub to_ticks: i64,
    /// `CAnimatedCharacter+0x320`: incoming-transition duration in ticks
    /// (0 or negative = none).
    pub to_duration_ticks: i64,
    /// `CAnimatedCharacter+0x330`: tick the outgoing transition started
    /// (0 = none).
    pub from_ticks: i64,
    /// `CAnimatedCharacter+0x338`: outgoing-transition duration in ticks
    /// (0 or negative = none).
    pub from_duration_ticks: i64,
}

/// Distance of `x` from the inclusive range `(lo, hi)`:
/// `max(x - hi, lo - x, 0)`.
///
/// The engine tests each difference with `comiss`/`ja`, so NaN compares
/// `false` and collapses to `0.0`, matching `!(d > 0.0)`.
fn range_dist(x: f32, range: (f32, f32)) -> f32 {
    let (lo, hi) = range;
    let mut a = x - hi;
    if !(a > 0.0) {
        a = 0.0;
    }
    let mut b = -(x - lo);
    if !(b > 0.0) {
        b = 0.0;
    }
    if !(a > b) {
        a = b;
    }
    a
}

/// Normalise an angle into `(-180, 180]` with the engine's single-wrap
/// `if (180 < a) a -= 360; if (a < -180) a += 360;` (`0x7cba12..0x7cba34`).
fn wrap_angle(mut a: f32) -> f32 {
    if ANGLE_WRAP < a {
        a -= ANGLE_FULL;
    }
    if a < -ANGLE_WRAP {
        a += ANGLE_FULL;
    }
    a
}

/// The engine's transition progress factor for one endpoint
/// (`FUN_007cb5f0` lines decoding `0x7cb83b..0x7cb943`): given the elapsed
/// seconds `t` and the transition duration, returns
/// `clamp((t - dur) / dur, 0, 1)` — `1.0` when the transition is absent.
fn transition_progress(t: f64, duration_ticks: i64, present: bool) -> f32 {
    if !present {
        return 1.0;
    }
    let base = duration_ticks as f64 * TICKS_TO_SECONDS as f64;
    let r = (t - base) / base;
    let r = ((r + 1.0) - (r - 1.0).abs()) * 0.5;
    ((r.abs() + r) * 0.5) as f32
}

/// `FUN_007cb5f0` (0x7cb5f0): the expensive-locomotion candidate score.
///
/// Lower is better; `FUN_007cbaf0` (0x7cbaf0) keeps the minimum.
pub fn locomotion_cost(candidate: &CandidateMotion, desired: &DesiredMotion) -> f32 {
    // Weighted profile distances. `d0` is deliberately dropped (its weight is
    // the zero in `xmm1` at 0x7cb67d).
    let d0 = range_dist(desired.phase, candidate.profile[0]);
    let d1 = range_dist(desired.actual_speed, candidate.profile[1]);
    let d2 = range_dist(desired.move_magnitude, candidate.profile[2]);
    let d3 = range_dist(desired.strafe_magnitude, candidate.profile[3]);
    let d4 = range_dist(desired.move_yaw, candidate.profile[4]);
    let d5 = range_dist(desired.travel_yaw, candidate.profile[5]);
    let d6 = range_dist(desired.angle_deviation, candidate.profile[6]);
    let d9 = range_dist(desired.requested_speed, candidate.speed);

    // Incoming-transition factor `q1` (0x7cb7e9..0x7cb8fe). `to_duration <= 0`
    // is the engine's "no transition" sentinel.
    let to_present = desired.to_duration_ticks > 0;
    let t_to = if desired.to_ticks == 0 {
        0.0
    } else {
        (desired.now_ticks - desired.to_ticks) as f64 * TICKS_TO_SECONDS as f64
    };
    let q1 = transition_progress(t_to, desired.to_duration_ticks, to_present);

    // Outgoing-transition factor `q2` (0x7cb8fe..0x7cb943).
    let from_present = desired.from_duration_ticks > 0;
    let t_from = if desired.from_ticks == 0 {
        0.0
    } else {
        (desired.now_ticks - desired.from_ticks) as f64 * TICKS_TO_SECONDS as f64
    };
    let q2 = transition_progress(t_from, desired.from_duration_ticks, from_present);

    // `tr = (1 - q2) * candidate[0x58]` (0x7cb95a, 0x7cb96c).
    let tr = (1.0 - q2) * candidate.blend_out;

    // Angular transition bonus on `profile[6]` (0x7cb97f..0x7cb9b9):
    // when the candidate is the current state (or the outgoing transition is
    // absent) and the deviation matches the transition band, add
    // `tr * 100 * 0.05`.
    let mut d6_term = d6 * ANGLE_WEIGHT;
    let from_absent = desired.from_ticks == 0;
    if (!desired.is_current || from_absent)
        && candidate.profile[6].0 <= -tr
        && -tr != candidate.profile[6].0
        && tr < candidate.profile[6].1
        && desired.angle_deviation.abs() < tr
    {
        d6_term += tr * TRANSITION_BONUS * ANGLE_WEIGHT;
    }

    // Straight-ahead transition bonus on `profile[3]` (0x7cb9bf..0x7cb9fc):
    // `x = (1 - q1) * candidate[0x5c]`, added as `x * 100 * 50`.
    let x = (1.0 - q1) * candidate.blend_in;
    let mut d3_term = d3 * PROFILE_WEIGHT;
    if x < candidate.profile[3].1 && desired.strafe_magnitude.abs() < x {
        d3_term += x * TRANSITION_BONUS * PROFILE_WEIGHT;
    }

    // Two derived-angle corridors (0x7cba02..0x7cbab6):
    // `travel_yaw - move_yaw` against `profile[7]`, and
    // `travel_yaw - angle_deviation` against `turn_extra`.
    let e1 = wrap_angle(desired.travel_yaw - desired.move_yaw);
    let d7 = range_dist(e1, candidate.profile[7]);
    let e2 = wrap_angle(desired.travel_yaw - desired.angle_deviation);
    let d8 = range_dist(e2, candidate.turn_extra);

    d8 * ANGLE_WEIGHT
        + d0 * 0.0
        + d9 * SPEED_WEIGHT
        + d1 * PROFILE_WEIGHT
        + d2 * PROFILE_WEIGHT
        + d3_term
        + d4 * ANGLE_WEIGHT
        + d5 * ANGLE_WEIGHT
        + d6_term
        + d7 * ANGLE_WEIGHT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_range_is_distance_from_authored_speed() {
        let walk = CandidateMotion::from_speed(1.66, 1.66);
        let run = CandidateMotion::from_speed(4.0, 4.0);
        let mut d = DesiredMotion { requested_speed: 1.66, ..Default::default() };
        assert_eq!(locomotion_cost(&walk, &d), 0.0);
        assert_eq!(locomotion_cost(&run, &d), (4.0 - 1.66) * 1000.0);

        d.requested_speed = 4.0;
        assert_eq!(locomotion_cost(&run, &d), 0.0);
        assert_eq!(locomotion_cost(&walk, &d), (4.0 - 1.66) * 1000.0);
    }

    #[test]
    fn range_inside_band_is_zero_but_speed_is_tight() {
        // A wide authored range scores 0 inside it; outside it is the clamp.
        let c = CandidateMotion::from_speed(1.0, 2.0);
        let mut d = DesiredMotion { requested_speed: 1.5, ..Default::default() };
        assert_eq!(locomotion_cost(&c, &d), 0.0);
        d.requested_speed = 3.0;
        assert_eq!(locomotion_cost(&c, &d), 1000.0);
        d.requested_speed = 0.5;
        assert_eq!(locomotion_cost(&c, &d), 500.0);
    }

    #[test]
    fn profile_weights_match_engine_constants() {
        // Only `actual_speed` (m[1] → profile[1]) is nonzero.
        let mut c = CandidateMotion::default();
        c.profile[1] = (0.0, 0.0);
        let d = DesiredMotion { actual_speed: 2.0, ..Default::default() };
        assert_eq!(locomotion_cost(&c, &d), 2.0 * PROFILE_WEIGHT);

        // The angular fields use the 0.05 weight. `move_yaw` (profile[4]) is
        // one such field; it also feeds the derived `travel_yaw - move_yaw`
        // corridor (profile[7]), so account for both.
        let c = CandidateMotion::default();
        let d = DesiredMotion { move_yaw: 10.0, ..Default::default() };
        assert!((locomotion_cost(&c, &d) - 2.0 * 10.0 * ANGLE_WEIGHT).abs() < 1e-6);
    }

    #[test]
    fn zero_weight_phase_is_ignored() {
        // `profile[0]` carries weight 0, so a huge mismatch costs nothing.
        let c = CandidateMotion { profile: [(100.0, 200.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], ..Default::default() };
        let d = DesiredMotion { phase: 1000.0, ..Default::default() };
        assert_eq!(locomotion_cost(&c, &d), 0.0);
    }

    #[test]
    fn derived_angle_corridors_wrap_at_180() {
        // travel_yaw - move_yaw = 350 - 10 = 340 -> wraps to -20.
        let mut c = CandidateMotion::default();
        c.profile[4] = (10.0, 10.0);
        c.profile[5] = (350.0, 350.0);
        c.profile[7] = (-20.0, -20.0);
        // `travel_yaw - angle_deviation` = 350 -> -10, so point `turn_extra` at -10.
        c.turn_extra = (-10.0, -10.0);
        let d = DesiredMotion { travel_yaw: 350.0, move_yaw: 10.0, ..Default::default() };
        assert_eq!(locomotion_cost(&c, &d), 0.0);

        // travel_yaw - angle_deviation = 350 - 10 -> -20.
        let mut c = CandidateMotion::default();
        c.profile[5] = (350.0, 350.0);
        c.profile[6] = (10.0, 10.0); // m[6] itself
        c.profile[7] = (-10.0, -10.0); // wrap(350 - 0) = -10
        c.turn_extra = (-20.0, -20.0);
        let d = DesiredMotion { travel_yaw: 350.0, angle_deviation: 10.0, ..Default::default() };
        assert_eq!(locomotion_cost(&c, &d), 0.0);
    }

    #[test]
    fn a_missing_transition_leaves_only_the_profile_distance() {
        // to/from transitions absent -> q1 == q2 == 1 -> blend terms vanish.
        let c = CandidateMotion {
            profile: [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            speed: (5.0, 5.0),
            blend_in: 10.0,
            blend_out: 10.0,
            ..Default::default()
        };
        let d = DesiredMotion { actual_speed: 2.0, requested_speed: 5.0, ..Default::default() };
        // |2-1|*50 = 50; speed matches.
        assert_eq!(locomotion_cost(&c, &d), 50.0);
    }

    fn extracted(rel: &str) -> Option<std::path::PathBuf> {
        let mut root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        root.pop(); // crates
        root.pop(); // native
        root.pop(); // repo root
        let p = root.join("extracted-game").join(rel);
        if p.is_file() { Some(p) } else { None }
    }

    /// Root-motion ground speed (m/s) of a real clip: horizontal root/hip
    /// translation over the clip window divided by its duration. Empty clips
    /// (no identifiable root, zero window) yield `0`.
    fn clip_speed(caf: &crate::anim::Caf, skeleton: &crate::skin::Skeleton) -> f32 {
        let rm = crate::locomotion::root_motion(caf, skeleton, caf.start as f32, caf.end as f32);
        let dur = (caf.end - caf.start) as f32 * caf.secs_per_tick;
        if !(dur > 0.0) {
            return 0.0;
        }
        (rm.translation[0] * rm.translation[0] + rm.translation[1] * rm.translation[1]).sqrt() / dur
    }

    /// The real animation graph's relaxed/no-weapon locomotion candidate set,
    /// plus real hero walk/run clip root-motion speeds.
    ///
    /// The shipped `RelaxedWalkNW`/`RelaxedRunNW` reference the *virtual*
    /// animations `_RELAXED_WALKSTRAFE_NW_01` / `_RELAXED_RUNSTRAFE_NW_01`,
    /// which are not present as files in the shipped/extracted corpus (only
    /// the compiled graph names them). The lookup path the engine uses is
    /// state → template `animation` → clip → root motion, so when the exact
    /// virtual clip is unavailable the canonical extracted hero walk/run
    /// forward clips stand in as the real root-motion source.
    #[test]
    fn real_relaxed_nw_walk_run_picked_by_clip_root_motion() {
        // Real graph.
        let graph_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/animations/animations/graphs/playerfullbody.xml");
        let Ok(xml) = std::fs::read(&graph_path) else { return };
        let g = crate::ag_runtime::Graph::parse(&xml).expect("parse playerfullbody");

        let walk_idx = g.state_index("RelaxedWalkNW").expect("RelaxedWalkNW state");
        let run_idx = g.state_index("RelaxedRunNW").expect("RelaxedRunNW state");

        // Real clips. Skip silently when the game data is absent.
        let (Some(chr), Some(walk_path), Some(run_path)) = (
            extracted("characters/characters/male/human/skeleton/skeleton.chr"),
            extracted("animations/animations/characters/hero_male/test/hero_walk_forward.$caf"),
            extracted("animations/animations/characters/hero_male/test/hero_run_forward.$caf"),
        ) else {
            return;
        };
        let skeleton = crate::skin::parse_skeleton(&std::fs::read(chr).unwrap()).unwrap();
        let walk_caf = crate::anim::Caf::parse(&std::fs::read(walk_path).unwrap()).unwrap();
        let run_caf = crate::anim::Caf::parse(&std::fs::read(run_path).unwrap()).unwrap();
        let walk_speed = clip_speed(&walk_caf, &skeleton);
        let run_speed = clip_speed(&run_caf, &skeleton);
        assert!(walk_speed > 0.0 && run_speed > walk_speed, "walk={walk_speed} run={run_speed}");

        // Candidate set for a relaxed, unarmed, moving player inside the shared
        // `PseudoSpeed [0,1.5]` band.
        let mut v = g.default_inputs();
        v.set_key("Action", "idle");
        v.set_key("Stance", "relaxed");
        v.set_key("Item", "nw");
        v.set_number("Health", 100.0);
        v.set_number("PseudoSpeed", 1.0);
        let cands = g.candidates(&v);
        assert!(cands.contains(&walk_idx) && cands.contains(&run_idx));

        // Per-candidate motion: the two hand-authored locomotion states get the
        // real clip root-motion speeds; every other candidate is left with a
        // zero authored speed (its transition clips are not modelled here).
        let motion = |i: usize| -> Option<CandidateMotion> {
            Some(if i == walk_idx {
                CandidateMotion::from_clip_speed(walk_speed)
            } else if i == run_idx {
                CandidateMotion::from_clip_speed(run_speed)
            } else {
                CandidateMotion::from_speed(0.0, 0.0)
            })
        };

        // `player.lua` walk/run speeds (1.66 / 4.0) drive the character; the
        // cost must resolve walk vs run from the real clip root motion.
        for (speed, want) in [(1.66f32, walk_idx), (4.0f32, run_idx)] {
            let desired = DesiredMotion { requested_speed: speed, ..Default::default() };
            let picked = g
                .pick_candidate_by_locomotion(&cands, motion, &desired)
                .unwrap();
            assert_eq!(
                g.states[picked].id, g.states[want].id,
                "desired {speed} m/s picked {} (walk {walk_speed}, run {run_speed})",
                g.states[picked].id
            );
        }
    }

    #[test]
    fn outgoing_transition_bonus_fires_only_for_non_current_candidates() {
        // q2 = 0 (`now == from`, duration 1 s) -> tr = 1.0 * blend_out.
        let c = CandidateMotion {
            profile: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-2.0, 3.0), (0.0, 0.0)],
            turn_extra: (-0.5, -0.5), // travel_yaw - angle_deviation = -0.5
            blend_out: 1.0,
            ..Default::default()
        };
        let d = DesiredMotion {
            angle_deviation: 0.5, // |0.5| < tr = 1
            now_ticks: 100_000,
            from_ticks: 100_000,
            from_duration_ticks: 100_000,
            is_current: false,
            ..Default::default()
        };
        // tr * 100 * 0.05 = 5.0; d6 itself is 0 (0 is inside [-2, 3]).
        assert_eq!(locomotion_cost(&c, &d), 5.0);

        // The same candidate as the *current* state does not take the bonus.
        let d = DesiredMotion { is_current: true, ..d };
        assert_eq!(locomotion_cost(&c, &d), 0.0);
    }

    #[test]
    fn incomplete_transition_adds_the_transition_bonus() {
        // now - to = 5e4 ticks = 0.5 s; duration 1e5 ticks = 1 s -> r = -0.5
        // -> q1 = 0. So x = 2.0; make profile[3] max 3.0 and strafe 1.0.
        let c = CandidateMotion {
            profile: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 3.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            blend_in: 2.0,
            speed: (0.0, 0.0),
            ..Default::default()
        };
        let d = DesiredMotion {
            strafe_magnitude: 1.0,
            now_ticks: 50_000,
            to_ticks: 100_000,
            to_duration_ticks: 100_000,
            ..Default::default()
        };
        // x = (1 - 0) * 2 = 2; 2 < 3 and 1 < 2 -> bonus 2 * 5000 = 10000.
        assert_eq!(locomotion_cost(&c, &d), 10_000.0);
    }
}
