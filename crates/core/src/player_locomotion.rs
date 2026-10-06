//! Player locomotion animation state machine — the Idle ↔ Walk ↔ Run ↔ Sprint
//! selection the retail game performs in the AnimationGraph.
//!
//! The real game feeds a normalised **`PseudoSpeed`** input (`+ Stance`/`Item`/
//! `Dir`) into the graph; the graph's `SelectWhen` bands choose the locomotion
//! state, and the clip plays at a speed-matched rate so the feet do not slide
//! (`notes/ghidra/animation.md` §2/§5). The shipped player graph uses
//! `MovementControlMethod = DecoupledCatchUp` (`movementdecoupled.xml`), so the
//! entity is moved by the controller and Movement↔Movement changes blend through
//! a decoupled catch-up rather than snapping.
//!
//! ## Recovered PseudoSpeed bands (source: shipped `graphs/*.xml`)
//! * **Idle** `[0, 0.2]` — `playerfullbody.xml` idle states `<PseudoSpeed min="0"
//!   max="0.2"/>`.
//! * **Walk → Run** at `0.7` — `humanmalefullbody.xml` `Walk ...` selects
//!   `PseudoSpeed 0.1..0.71`, `Run ...` selects `0.70..3`.
//! * **Run → Sprint** at `2.5` — `playerfullbody.xml` `SprintDecoupled` overrides
//!   `PseudoSpeedMin="2.5"` (run strafe `1.5..2.5`, sprint `2.5..3.5`).
//!
//! ## Recovered transition times (source: shipped `animationgraphtemplates/`)
//! * Idle→Movement `0.1` s (`idletomovement.xml`).
//! * Movement→Idle `0.2` s (`movementtoidle.xml`).
//! * Movement↔Movement decoupled catch-up `0.3` s (`playerfullbody.xml` run/walk
//!   strafe `transition_time="0.3"`; `movementdecoupled.xml` uses `0.2`).

use crate::locomotion::Gait;
use crate::tuning::MovementTuning;

/// PseudoSpeed below which the graph picks a walk clip; above it, run.
pub const WALK_RUN_PS: f32 = 0.7;
/// PseudoSpeed below which the graph picks a run clip; above it, sprint.
pub const RUN_SPRINT_PS: f32 = 2.5;
/// Hysteresis margin (PseudoSpeed units) so bands do not oscillate at a boundary.
pub const PS_HYSTERESIS: f32 = 0.05;

/// Idle→Movement transition duration in seconds (`idletomovement.xml`).
pub const IDLE_TO_MOVE_TIME: f32 = 0.1;
/// Movement→Idle transition duration in seconds (`movementtoidle.xml`).
pub const MOVE_TO_IDLE_TIME: f32 = 0.2;
/// Movement↔Movement decoupled catch-up duration in seconds.
pub const DECOUPLED_TIME: f32 = 0.3;

/// Ground speed (m/s) at or below which the character counts as stopped.
const STOP_SPEED: f32 = 0.05;

/// Locomotion animation state.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum LocoState {
    Idle,
    IdleToMovement,
    Walk,
    Run,
    Sprint,
    MovementToIdle,
    /// Movement↔Movement catch-up blend used when the graph's
    /// `MovementControlMethod` is `DecoupledCatchUp` (the player graph).
    Decoupled,
}

/// Player locomotion animation state machine.
#[derive(Clone, Debug)]
pub struct PlayerLocomotion {
    pub state: LocoState,
    /// Gait of the clip currently playing.
    pub gait: Gait,
    /// Speed-matched clip playback rate (`1.0` for idle).
    pub playback_rate: f32,
    /// Transition progress `0 → 1` for the active blend (`1.0` when settled).
    pub transition: f32,
    /// Last `PseudoSpeed` fed in (sanitised, finite, non-negative).
    pub pseudo_speed: f32,
    tuning: MovementTuning,
    decoupled: bool,
    target: LocoState,
}

impl PlayerLocomotion {
    /// New state machine for a decoupled (player) graph.
    pub fn new(tuning: MovementTuning) -> Self {
        Self::with_decoupled(tuning, true)
    }

    /// `decoupled` mirrors the graph using `DecoupledCatchUp` (the shipped player
    /// graph). When true, Movement↔Movement gait changes blend through
    /// [`LocoState::Decoupled`]; when false they snap.
    pub fn with_decoupled(tuning: MovementTuning, decoupled: bool) -> Self {
        PlayerLocomotion {
            state: LocoState::Idle,
            gait: Gait::Idle,
            playback_rate: 1.0,
            transition: 1.0,
            pseudo_speed: 0.0,
            tuning,
            decoupled,
            target: LocoState::Idle,
        }
    }

    /// Advance one frame. Non-finite inputs are treated as `0`; never panics.
    pub fn update(&mut self, speed: f32, pseudo_speed: f32, stance_aiming: bool, dt: f32) {
        let speed = finite_nonneg(speed);
        let ps = finite_nonneg(pseudo_speed);
        let dt = finite_nonneg(dt);
        self.pseudo_speed = ps;

        let creep_walk = lerp(
            self.tuning.creep,
            self.tuning.walk,
            self.tuning.threshold_creep_walk,
        );
        let creep_walk = if creep_walk.is_finite() { creep_walk } else { 0.0 };

        match self.state {
            LocoState::Idle => {
                if speed > creep_walk {
                    self.begin_idle_to_move(ps);
                } else {
                    self.gait = Gait::Idle;
                }
            }
            LocoState::IdleToMovement => {
                if speed <= STOP_SPEED {
                    self.begin_move_to_idle();
                } else {
                    self.target = band_state(ps);
                    self.gait = gait_of(self.target);
                    self.transition = advance(self.transition, dt, IDLE_TO_MOVE_TIME);
                    if self.transition >= 1.0 {
                        self.state = self.target;
                    }
                }
            }
            LocoState::Walk | LocoState::Run | LocoState::Sprint => {
                if speed <= STOP_SPEED {
                    self.begin_move_to_idle();
                } else {
                    let want = steady_target(self.state, ps);
                    if want == self.state {
                        self.gait = gait_of(self.state);
                    } else if self.decoupled {
                        self.state = LocoState::Decoupled;
                        self.target = want;
                        self.transition = 0.0;
                        self.gait = gait_of(want);
                    } else {
                        self.state = want;
                        self.transition = 1.0;
                        self.gait = gait_of(want);
                    }
                }
            }
            LocoState::Decoupled => {
                if speed <= STOP_SPEED {
                    self.begin_move_to_idle();
                } else {
                    self.gait = gait_of(self.target);
                    self.transition = advance(self.transition, dt, DECOUPLED_TIME);
                    if self.transition >= 1.0 {
                        self.state = self.target;
                    }
                }
            }
            LocoState::MovementToIdle => {
                if speed > creep_walk {
                    self.begin_idle_to_move(ps);
                } else {
                    self.gait = Gait::Idle;
                    self.transition = advance(self.transition, dt, MOVE_TO_IDLE_TIME);
                    if self.transition >= 1.0 {
                        self.state = LocoState::Idle;
                    }
                }
            }
        }

        self.playback_rate = self.rate(speed, stance_aiming);
    }

    fn begin_idle_to_move(&mut self, ps: f32) {
        self.state = LocoState::IdleToMovement;
        self.transition = 0.0;
        self.target = band_state(ps);
        self.gait = gait_of(self.target);
    }

    fn begin_move_to_idle(&mut self) {
        self.state = LocoState::MovementToIdle;
        self.transition = 0.0;
        self.gait = Gait::Idle;
    }

    /// `clamp(speed / gait_base_speed, 0.5, 2.0)`, `1.0` for idle.
    fn rate(&self, speed: f32, aiming: bool) -> f32 {
        let base = base_speed(self.gait, aiming, &self.tuning);
        if !(base > 0.0) || !base.is_finite() {
            return 1.0;
        }
        let r = speed / base;
        if r.is_finite() {
            r.clamp(0.5, 2.0)
        } else {
            1.0
        }
    }

    /// Gait of the currently playing clip.
    pub fn clip_gait(&self) -> Gait {
        self.gait
    }

    /// Stable, lowercase state name.
    pub fn state_name(&self) -> &'static str {
        match self.state {
            LocoState::Idle => "idle",
            LocoState::IdleToMovement => "idle_to_movement",
            LocoState::Walk => "walk",
            LocoState::Run => "run",
            LocoState::Sprint => "sprint",
            LocoState::MovementToIdle => "movement_to_idle",
            LocoState::Decoupled => "decoupled",
        }
    }
}

fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + (b - a) * f
}

fn finite_nonneg(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

fn advance(t: f32, dt: f32, dur: f32) -> f32 {
    if dur > 0.0 {
        (t + dt / dur).min(1.0)
    } else {
        1.0
    }
}

fn gait_of(s: LocoState) -> Gait {
    match s {
        LocoState::Walk => Gait::Walk,
        LocoState::Run => Gait::Run,
        LocoState::Sprint => Gait::Sprint,
        _ => Gait::Idle,
    }
}

/// Band selection with no memory (used when settling out of Idle/transition).
fn band_state(ps: f32) -> LocoState {
    if ps >= RUN_SPRINT_PS {
        LocoState::Sprint
    } else if ps >= WALK_RUN_PS {
        LocoState::Run
    } else {
        LocoState::Walk
    }
}

/// Steady-state band selection with hysteresis around `from`.
fn steady_target(from: LocoState, ps: f32) -> LocoState {
    match from {
        LocoState::Walk => {
            if ps > RUN_SPRINT_PS + PS_HYSTERESIS {
                LocoState::Sprint
            } else if ps > WALK_RUN_PS + PS_HYSTERESIS {
                LocoState::Run
            } else {
                LocoState::Walk
            }
        }
        LocoState::Run => {
            if ps < WALK_RUN_PS - PS_HYSTERESIS {
                LocoState::Walk
            } else if ps > RUN_SPRINT_PS + PS_HYSTERESIS {
                LocoState::Sprint
            } else {
                LocoState::Run
            }
        }
        LocoState::Sprint => {
            if ps < RUN_SPRINT_PS - PS_HYSTERESIS {
                if ps < WALK_RUN_PS - PS_HYSTERESIS {
                    LocoState::Walk
                } else {
                    LocoState::Run
                }
            } else {
                LocoState::Sprint
            }
        }
        _ => band_state(ps),
    }
}

/// Nominal speed the clip of `gait` is authored for; aiming uses the slower
/// `stance[2]` speeds from `player.lua`.
fn base_speed(gait: Gait, aiming: bool, t: &MovementTuning) -> f32 {
    match gait {
        Gait::Idle => 0.0,
        Gait::Creep => {
            if aiming {
                t.aim_creep
            } else {
                t.creep
            }
        }
        Gait::Walk => {
            if aiming {
                t.aim_walk
            } else {
                t.walk
            }
        }
        Gait::Run => {
            if aiming {
                t.aim_run
            } else {
                t.run
            }
        }
        Gait::Sprint => {
            if aiming {
                t.aim_sprint
            } else {
                t.sprint
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::default_movement;

    #[test]
    fn idle_walk_run_sprint_progression() {
        let t = default_movement();
        let mut p = PlayerLocomotion::new(t);
        assert_eq!(p.state_name(), "idle");
        assert_eq!(p.clip_gait(), Gait::Idle);

        // Speed crosses the creep/walk threshold -> IdleToMovement.
        p.update(t.walk, 0.4, false, 0.016);
        assert_eq!(p.state, LocoState::IdleToMovement);
        assert_eq!(p.clip_gait(), Gait::Walk);

        // Transition completes (0.1 s) -> Walk.
        p.update(t.walk, 0.4, false, 0.2);
        assert_eq!(p.state, LocoState::Walk);
        assert_eq!(p.clip_gait(), Gait::Walk);

        // Walk -> Run is a decoupled Movement↔Movement blend.
        p.update(t.run, 1.5, false, 0.016);
        assert_eq!(p.state, LocoState::Decoupled);
        p.update(t.run, 1.5, false, 0.5);
        assert_eq!(p.state, LocoState::Run);
        assert_eq!(p.clip_gait(), Gait::Run);

        // Run -> Sprint.
        p.update(t.sprint, 3.0, false, 0.016);
        assert_eq!(p.state, LocoState::Decoupled);
        p.update(t.sprint, 3.0, false, 0.5);
        assert_eq!(p.state, LocoState::Sprint);
        assert_eq!(p.clip_gait(), Gait::Sprint);
    }

    #[test]
    fn sprint_run_hysteresis_no_flip_flop() {
        let t = default_movement();
        let mut p = PlayerLocomotion::new(t);
        p.update(t.sprint, 3.0, false, 0.2);
        p.update(t.sprint, 3.0, false, 0.2);
        assert_eq!(p.state, LocoState::Sprint);

        // Oscillate inside the ±0.05 band around the 2.5 run/sprint boundary.
        for i in 0..100 {
            let ps = if i % 2 == 0 { 2.5 } else { 2.48 };
            p.update(t.sprint, ps, false, 0.1);
            assert_eq!(p.state, LocoState::Sprint, "flip-flopped at ps={ps}");
        }

        // Crossing the lower hysteresis bound settles into Run.
        p.update(t.sprint, 2.4, false, 0.016);
        assert_eq!(p.state, LocoState::Decoupled);
        p.update(t.sprint, 2.4, false, 0.5);
        assert_eq!(p.state, LocoState::Run);

        // And back at the boundary it stays Run (no flip-flop).
        for _ in 0..20 {
            p.update(t.sprint, 2.5, false, 0.1);
            assert_eq!(p.state, LocoState::Run);
        }
    }

    #[test]
    fn stop_goes_through_movement_to_idle() {
        let t = default_movement();
        let mut p = PlayerLocomotion::new(t);
        p.update(t.run, 1.0, false, 0.2);
        p.update(t.run, 1.0, false, 0.2);
        assert_eq!(p.state, LocoState::Run);

        p.update(0.0, 0.0, false, 0.016);
        assert_eq!(p.state, LocoState::MovementToIdle);
        assert_eq!(p.clip_gait(), Gait::Idle);

        // Still blending before MOVE_TO_IDLE_TIME (0.2 s).
        p.update(0.0, 0.0, false, 0.1);
        assert_eq!(p.state, LocoState::MovementToIdle);

        p.update(0.0, 0.0, false, 0.2);
        assert_eq!(p.state, LocoState::Idle);
    }

    #[test]
    fn playback_rate_base_and_clamped() {
        let t = default_movement();
        let mut p = PlayerLocomotion::new(t);
        p.update(t.walk, 0.4, false, 0.2);
        p.update(t.walk, 0.4, false, 0.2);
        assert_eq!(p.state, LocoState::Walk);
        assert!((p.playback_rate - 1.0).abs() < 1e-6, "rate={}", p.playback_rate);

        // Clamped high.
        p.update(t.walk * 100.0, 0.4, false, 0.016);
        assert_eq!(p.playback_rate, 2.0);

        // Clamped low, but still moving (above the stop threshold).
        p.update(0.2, 0.4, false, 0.016);
        assert_eq!(p.playback_rate, 0.5);

        // Idle always plays at 1.0.
        p.update(0.0, 0.0, false, 0.016);
        p.update(0.0, 0.0, false, 0.2);
        assert_eq!(p.state, LocoState::Idle);
        assert_eq!(p.playback_rate, 1.0);
    }

    #[test]
    fn non_decoupled_graph_snaps_gaits() {
        let t = default_movement();
        let mut p = PlayerLocomotion::with_decoupled(t, false);
        p.update(t.walk, 0.4, false, 0.2);
        p.update(t.walk, 0.4, false, 0.2);
        assert_eq!(p.state, LocoState::Walk);
        p.update(t.run, 1.5, false, 0.016);
        assert_eq!(p.state, LocoState::Run);
    }

    #[test]
    fn nan_inputs_are_total() {
        let t = default_movement();
        let mut p = PlayerLocomotion::new(t);
        for _ in 0..10 {
            p.update(f32::NAN, f32::NAN, false, f32::NAN);
            assert!(p.playback_rate.is_finite());
            assert!(p.transition.is_finite());
            assert!(p.pseudo_speed.is_finite());
        }
        p.update(f32::INFINITY, f32::NEG_INFINITY, true, f32::INFINITY);
        assert!(p.playback_rate.is_finite());
        assert!(p.transition.is_finite());
        assert!(p.pseudo_speed.is_finite());
        assert_eq!(p.state, LocoState::Idle);
    }
}
