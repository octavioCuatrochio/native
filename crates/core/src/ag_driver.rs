//! Drives a character's AnimationGraph from its movement state, the way the
//! `CAnimatedCharacter` does each frame: write the graph inputs, query the
//! candidate states, pick the target (hysteresis, previous target, expensive
//! locomotion cost, random), and walk the link path.
//!
//! Input mapping: the animated character caches 24 movement inputs
//! (`notes/ghidra/animation.md` section 2: Requested*/Actual* speeds in the
//! character's local frame, Dir4 keys, turn speeds, AnimPhase, Action,
//! PseudoSpeed, Stance). The single per-frame setter that computes them was
//! not isolated, so the vector inputs here follow their names and types
//! (INFERRED): `LX` = local right, `LY` = local forward, `LH` = horizontal
//! magnitude, `LZ` = turn rate, `Dir4LH` = quadrant of the local direction.
//! `PseudoSpeed` comes from the ported controller (`UpdateNormal`).

use crate::ag_cost::{CandidateMotion, DesiredMotion};
use crate::ag_runtime::{Graph, GraphInstance, InputValues};

/// Below this horizontal speed (m/s) the Dir4 keys read `Idle`. INFERRED.
pub const DIR4_IDLE_SPEED: f32 = 0.1;

/// One frame of a character's movement, in CryEngine space (Z-up).
#[derive(Debug, Clone, PartialEq)]
pub struct MovementSnapshot {
    /// Velocity the controller asks for (m/s, world XY).
    pub requested_vel: [f32; 2],
    /// Velocity actually achieved this frame (m/s, world XY).
    pub actual_vel: [f32; 2],
    /// Body facing: radians about +Z, with local forward = +Y rotated by it
    /// (CryEngine convention: yaw 0 faces +Y).
    pub body_yaw: f32,
    /// Requested and actual turn rates (rad/s about +Z).
    pub requested_turn: f32,
    pub actual_turn: f32,
    /// `PseudoSpeed` from the controller.
    pub pseudo_speed: f32,
    pub stance: String,
    pub item: String,
    pub action: String,
    /// 0..1.
    pub aiming: f32,
    pub health: f32,
}

impl Default for MovementSnapshot {
    fn default() -> Self {
        MovementSnapshot {
            requested_vel: [0.0; 2],
            actual_vel: [0.0; 2],
            body_yaw: 0.0,
            requested_turn: 0.0,
            actual_turn: 0.0,
            pseudo_speed: 0.0,
            stance: "relaxed".into(),
            item: "nw".into(),
            action: "idle".into(),
            aiming: 0.0,
            health: 100.0,
        }
    }
}

/// World XY velocity -> character-local `(right, forward)`.
fn to_local(v: [f32; 2], yaw: f32) -> (f32, f32) {
    let (s, c) = yaw.sin_cos();
    // forward = (-sin, cos), right = (cos, sin) for yaw about +Z, forward +Y.
    let forward = -v[0] * s + v[1] * c;
    let right = v[0] * c + v[1] * s;
    (right, forward)
}

fn dir4(right: f32, forward: f32) -> &'static str {
    if (right * right + forward * forward).sqrt() < DIR4_IDLE_SPEED {
        "Idle"
    } else if forward.abs() >= right.abs() {
        if forward >= 0.0 { "Fwd" } else { "Bwd" }
    } else if right >= 0.0 {
        "Rgt"
    } else {
        "Lft"
    }
}

/// Graph input values for this frame: the graph's declared defaults, then
/// everything the animated character writes.
pub fn graph_inputs(g: &Graph, m: &MovementSnapshot) -> InputValues {
    let mut v = g.default_inputs();
    let (rx, ry) = to_local(m.requested_vel, m.body_yaw);
    let (ax, ay) = to_local(m.actual_vel, m.body_yaw);
    let rh = (rx * rx + ry * ry).sqrt();
    let ah = (ax * ax + ay * ay).sqrt();
    v.set_number("RequestedMoveSpeedLX", rx);
    v.set_number("RequestedMoveSpeedLY", ry);
    v.set_number("RequestedMoveSpeedLH", rh);
    v.set_number("RequestedMoveSpeed", rh);
    v.set_number("RequestedTurnSpeedLZ", m.requested_turn);
    v.set_key("RequestedMoveDir4LH", dir4(rx, ry));
    v.set_number("ActualMoveSpeedLX", ax);
    v.set_number("ActualMoveSpeedLY", ay);
    v.set_number("ActualMoveSpeedLH", ah);
    v.set_number("ActualMoveSpeed", ah);
    v.set_number("ActualTurnSpeedLZ", m.actual_turn);
    v.set_key("ActualMoveDir4LH", dir4(ax, ay));
    v.set_number("PseudoSpeed", m.pseudo_speed);
    v.set_key("Stance", &m.stance);
    v.set_key("Item", &m.item);
    v.set_key("Action", &m.action);
    v.set_number("Aiming", m.aiming);
    v.set_number("Health", m.health);
    v
}

/// The cost function's "desired motion" for this frame.
pub fn desired_motion(m: &MovementSnapshot, is_current: bool) -> DesiredMotion {
    let (ax, ay) = to_local(m.actual_vel, m.body_yaw);
    let (rx, ry) = to_local(m.requested_vel, m.body_yaw);
    let deg = |a: f32| a.to_degrees();
    DesiredMotion {
        phase: 0.0,
        actual_speed: (ax * ax + ay * ay).sqrt(),
        move_magnitude: (rx * rx + ry * ry).sqrt(),
        strafe_magnitude: rx.abs(),
        move_yaw: deg(rx.atan2(ry)),
        travel_yaw: deg(ax.atan2(ay)),
        angle_deviation: 0.0,
        requested_speed: (rx * rx + ry * ry).sqrt(),
        is_current,
        now_ticks: 0,
        to_ticks: 0,
        to_duration_ticks: 0,
        from_ticks: 0,
        from_duration_ticks: 0,
    }
}

/// A character's graph instance plus the engine's target-pick order.
pub struct CharacterGraph {
    pub graph: Graph,
    pub inst: GraphInstance,
    /// `graph+0xd2`: expensive locomotion selection (on by default).
    pub expensive_locomotion: bool,
}

impl CharacterGraph {
    pub fn new(graph: Graph) -> Self {
        CharacterGraph { graph, inst: GraphInstance::default(), expensive_locomotion: true }
    }

    pub fn current_name(&self) -> Option<&str> {
        self.inst.current.map(|i| self.graph.states[i].id.as_str())
    }

    pub fn target_name(&self) -> Option<&str> {
        self.inst.target.map(|i| self.graph.states[i].id.as_str())
    }

    /// One frame. `motion_of(state)` gives a state's motion profile (from its
    /// clips' root motion), or `None` when unknown; the cost pick only runs
    /// when every candidate has one. `step` advances one link of the path.
    pub fn update<F>(&mut self, m: &MovementSnapshot, motion_of: F, step: bool)
    where
        F: Fn(usize) -> Option<CandidateMotion>,
    {
        let values = graph_inputs(&self.graph, m);
        let cands = self.graph.candidates(&values);
        let current = self.inst.current;
        // 1. hysteresis, 2. previous target.
        let keep = current.filter(|c| cands.contains(c)).or_else(|| self.inst.target.filter(|t| cands.contains(t)));
        let target = match keep {
            Some(t) => Some(t),
            None => {
                // 3. expensive locomotion selection, when profiles are known.
                let costed = if self.expensive_locomotion && cands.len() > 1 && cands.iter().all(|&i| motion_of(i).is_some()) {
                    self.graph.pick_candidate_by_locomotion(&cands, |i| motion_of(i), &desired_motion(m, false))
                } else {
                    None
                };
                // 4. random among the equals.
                costed.or_else(|| self.graph.pick_candidate(&cands, None, None, &mut self.inst.rng))
            }
        };
        if target != self.inst.target {
            self.inst.target = target;
            self.inst.queue = match (self.inst.current, target) {
                (Some(c), Some(t)) => self.graph.path(c, t).unwrap_or_default(),
                _ => Vec::new(),
            };
            if self.inst.current.is_none() {
                self.inst.current = target;
            }
        }
        if step && !self.inst.queue.is_empty() {
            let li = self.inst.queue.remove(0);
            self.inst.current = Some(self.graph.links[li].to);
            self.inst.blend = self.graph.links[li].transition_time;
        } else if self.inst.queue.is_empty() && target.is_some() && self.inst.current != target {
            self.inst.current = target;
            self.inst.blend = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_frame_and_dir4() {
        // Yaw 0 faces +Y: world +Y velocity is forward, +X is right.
        assert_eq!(dir4(to_local([0.0, 4.0], 0.0).0, to_local([0.0, 4.0], 0.0).1), "Fwd");
        let (r, f) = to_local([4.0, 0.0], 0.0);
        assert!((r - 4.0).abs() < 1e-5 && f.abs() < 1e-5);
        assert_eq!(dir4(r, f), "Rgt");
        // Facing +X (yaw -90 deg): world +X is forward.
        let (r, f) = to_local([4.0, 0.0], -std::f32::consts::FRAC_PI_2);
        assert!(f > 3.99 && r.abs() < 1e-4, "{r} {f}");
        assert_eq!(dir4(0.01, 0.02), "Idle");
    }

    /// The real graph's standing candidates include the idle state; with
    /// motion profiles supplied, the engine's pick order (cost step) picks
    /// idle / walk / run by requested speed. The profiles here are STAND-IN
    /// speeds until clip root motion is resolved (worker A1); the real
    /// end-to-end check replaces them then.
    #[test]
    fn real_player_graph_pick_order_with_motion_profiles() {
        let p = "/home/battlestation/Desktop/sod-re/extracted-game/animations/animations/graphs/playerfullbody.xml";
        let Ok(xml) = std::fs::read(p) else { return };
        let g = Graph::parse(&xml).expect("parse");
        let standing = graph_inputs(&g, &MovementSnapshot::default());
        let cands = g.candidates(&standing);
        let idle = g.state_index("RelaxedIdleNW").unwrap();
        let walk = g.state_index("RelaxedWalkNW").unwrap();
        let run = g.state_index("RelaxedRunNW").unwrap();
        assert!(cands.contains(&idle), "idle must be a standing candidate");

        // Stand-in profiles: idle 0, walk 1.66, run 4.0 m/s; other candidates
        // get a profile far from any speed so they never win.
        let motion = |i: usize| {
            Some(if i == idle {
                CandidateMotion::from_speed(0.0, 0.0)
            } else if i == walk {
                CandidateMotion::from_clip_speed(1.66)
            } else if i == run {
                CandidateMotion::from_clip_speed(4.0)
            } else {
                CandidateMotion::from_speed(50.0, 50.0)
            })
        };
        for (speed, want) in [(0.0f32, "RelaxedIdleNW"), (1.66, "RelaxedWalkNW"), (4.0, "RelaxedRunNW")] {
            let mut cg = CharacterGraph::new(Graph::parse(&xml).unwrap());
            let snap = MovementSnapshot {
                requested_vel: [0.0, speed],
                actual_vel: [0.0, speed],
                pseudo_speed: if speed > 0.0 { 1.0 } else { 0.0 },
                ..Default::default()
            };
            cg.update(&snap, motion, false);
            assert_eq!(cg.target_name(), Some(want), "requested {speed} m/s");
        }
    }
}
