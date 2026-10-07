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
use crate::anim_alias::{BlendType, Resolved, ResolvedKind};

/// Below this horizontal speed (m/s) the Dir4 keys read `Idle`: the input setter
/// compares sqrt(LX²+LY²) with the float at `0x17877c4` (= 0.5) at `0x876225`, then
/// writes `"Idle"` (`0x1771cf0`) to RequestedMoveDir4LH and zeroes LX/LY/LH.
pub const DIR4_IDLE_SPEED: f32 = 0.5;

/// Graph-template `name`s that play a one-shot transition rather than a loop:
/// the engine holds the state until the clip finishes before walking the next
/// link (`relaxed_idleToWalk_nw_01` → `IdleToMovement`, …). WorkerA2b.
const ONE_SHOT_TEMPLATES: &[&str] = &[
    "IdleToMovement",
    "MovementToIdle",
    "WalkToIdle",
    "RunToIdle",
    "IdleToRun",
    "Start",
    "Stop",
];

/// Animation-name / template metadata for one graph state.
///
/// The player graph names a clip through a `<Template … animation="…"/>`
/// (`MovementDecoupled`, `IdleToMovement`, …). Idle states carry the empty
/// `HumanAlive`/`Default` template instead, so `animation` stays `None` and the
/// caller falls back to the character clip bank.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StateTemplate {
    /// Template `name` (`MovementDecoupled`, `IdleToMovement`, …).
    pub name: Option<String>,
    /// First non-empty `animation` attribute.
    pub animation: Option<String>,
    /// Parsed `playback_speed` (`None` when absent/zero).
    pub playback_speed: Option<f32>,
    /// `structure` attribute (`StandMoving`, `StandIdle`, …).
    pub structure: Option<String>,
}

impl StateTemplate {
    /// `1.0` when the template carries no `playback_speed`.
    pub fn speed(&self) -> f32 {
        self.playback_speed.filter(|s| *s > 0.0).unwrap_or(1.0)
    }

    /// One-shot transition (must finish before the link path advances).
    pub fn one_shot(&self) -> bool {
        matches!(&self.name, Some(n) if ONE_SHOT_TEMPLATES.iter().any(|t| t.eq_ignore_ascii_case(n)))
    }
}

/// Read a state's template metadata (first template's `name`, first non-empty
/// `animation`, `playback_speed`, `structure`).
pub fn state_template(g: &Graph, state: usize) -> StateTemplate {
    let mut out = StateTemplate::default();
    let Some(st) = g.states.get(state) else { return out };
    for (name, attrs) in &st.templates {
        if out.name.is_none() && !name.is_empty() {
            out.name = Some(name.clone());
        }
        let get = |k: &str| attrs.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        if out.animation.is_none() {
            if let Some(a) = get("animation") {
                if !a.trim().is_empty() {
                    out.animation = Some(a.trim().to_string());
                }
            }
        }
        if out.playback_speed.is_none() {
            if let Some(f) = get("playback_speed").and_then(|v| v.trim().parse::<f32>().ok()) {
                if f > 0.0 {
                    out.playback_speed = Some(f);
                }
            }
        }
        if out.structure.is_none() {
            if let Some(s) = get("structure") {
                if !s.trim().is_empty() {
                    out.structure = Some(s.trim().to_string());
                }
            }
        }
    }
    out
}

/// `CandidateMotion` for a resolved state animation: a single clip is a point,
/// a blend space the `min..max` over its clips' root-motion speeds (workerA2b).
pub fn motion_from_speeds(kind: &ResolvedKind, speeds: &[f32]) -> Option<CandidateMotion> {
    let s: Vec<f32> = speeds.iter().copied().filter(|v| v.is_finite()).collect();
    if s.is_empty() {
        return None;
    }
    match kind {
        ResolvedKind::Single => Some(CandidateMotion::from_clip_speed(s[0])),
        ResolvedKind::Blend(_) => {
            let min = s.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            Some(CandidateMotion::from_speed(min, max))
        }
    }
}

/// [`CandidateMotion`] for a resolved animation, using the clip paths.
///
/// A single clip is a point. A LocomotionGroup's authored speed band is the
/// `min..max` over the clips that match its `CAPS` gait (`relaxed_walk.lmg`'s
/// `WALK` group carries both a walk row and a run row, but the walk state is
/// authored at the walk speed). When no clip matches the caps, all clips are
/// used. This is what separates `RelaxedWalkNW` (≈1.72) from `RelaxedRunNW`
/// (≈4.14) in the engine's expensive locomotion cost.
pub fn motion_from_resolved(resolved: &Resolved, speeds: &[f32]) -> Option<CandidateMotion> {
    let ResolvedKind::Blend(info) = &resolved.kind else {
        return motion_from_speeds(&resolved.kind, speeds);
    };
    let want = gait_from_caps(&info.caps);
    let mut vals: Vec<f32> = Vec::new();
    for (i, s) in speeds.iter().enumerate() {
        if !s.is_finite() {
            continue;
        }
        let path = resolved.clips.get(i).map(|(p, _)| p.as_str()).unwrap_or("");
        if let (Some(g), Some(pg)) = (want, crate::locomotion::Gait::from_name(path)) {
            if g != pg {
                continue;
            }
        }
        vals.push(*s);
    }
    if vals.is_empty() {
        vals = speeds.iter().copied().filter(|v| v.is_finite()).collect();
    }
    if vals.is_empty() {
        return None;
    }
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    Some(CandidateMotion::from_speed(min, max))
}

/// The gait a LocomotionGroup's `CAPS` code names (`WALK`, `RUN_`, `SPRINT`,
/// `IROT`), if any.
fn gait_from_caps(caps: &str) -> Option<crate::locomotion::Gait> {
    use crate::locomotion::Gait;
    let c = caps.trim().to_ascii_lowercase();
    if c.is_empty() {
        None
    } else if c.contains("sprint") {
        Some(Gait::Sprint)
    } else if c.contains("run") {
        Some(Gait::Run)
    } else if c.contains("walk") || c.contains("creep") {
        Some(Gait::Walk)
    } else if c.contains("idle") || c.contains("irot") {
        Some(Gait::Idle)
    } else {
        None
    }
}

/// Per-example blend weights for a LocomotionGroup at a local movement.
///
/// `lateral` is `-1..=1` (right positive), `gait` is `-1..=1` (walk negative,
/// run positive). `FLR1`/`STF1`/`IROT` blend along the first axis of the
/// authored `Position`s; `FLR2`/`STF2` interpolate bilinearly over
/// `(lateral, gait)` exactly as `relaxed_walk.lmg` lays its walk (`y=-1`) and
/// run (`y=+1`) rows out. A single-example group plays at full weight.
pub fn blend_weights(resolved: &Resolved, lateral: f32, gait: f32) -> Vec<f32> {
    match &resolved.kind {
        ResolvedKind::Single => vec![1.0],
        ResolvedKind::Blend(info) => {
            blend_weights_for(Some(&info.blend_type), &info.params, lateral, gait)
        }
    }
}

/// [`blend_weights`] from the raw blend pieces (so a caller can cache only the
/// blend type + positions rather than the whole [`Resolved`]).
pub fn blend_weights_for(
    blend_type: Option<&BlendType>,
    params: &[[f32; 3]],
    lateral: f32,
    gait: f32,
) -> Vec<f32> {
    let n = params.len();
    if n == 0 {
        Vec::new()
    } else if n == 1 {
        vec![1.0]
    } else if matches!(blend_type, Some(BlendType::Flr2) | Some(BlendType::Stf2)) {
        blend_2d(params, lateral, gait)
    } else {
        blend_1d(params, (0..n).collect(), lateral)
    }
}

/// One-dimensional linear blend along the `x` coordinate over the given
/// example indices (empty weight elsewhere).
fn blend_1d(params: &[[f32; 3]], mut idxs: Vec<usize>, p: f32) -> Vec<f32> {
    let mut out = vec![0.0; params.len()];
    if idxs.is_empty() {
        return out;
    }
    idxs.sort_by(|&a, &b| params[a][0].partial_cmp(&params[b][0]).unwrap_or(std::cmp::Ordering::Equal));
    if p <= params[idxs[0]][0] {
        out[idxs[0]] = 1.0;
        return out;
    }
    if p >= params[*idxs.last().unwrap()][0] {
        out[*idxs.last().unwrap()] = 1.0;
        return out;
    }
    for w in idxs.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (xa, xb) = (params[a][0], params[b][0]);
        if p >= xa && p <= xb {
            let span = if xb > xa { xb - xa } else { 1.0 };
            let t = (p - xa) / span;
            out[a] += 1.0 - t;
            out[b] += t;
            return out;
        }
    }
    out[idxs[0]] = 1.0;
    out
}

/// Bilinear blend over the `(x, y)` grid the examples define.
fn blend_2d(params: &[[f32; 3]], px: f32, py: f32) -> Vec<f32> {
    let n = params.len();
    let mut ys: Vec<f32> = params.iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
    if ys.is_empty() {
        return vec![0.0; n];
    }
    let row = |y: f32| -> Vec<usize> { (0..n).filter(|&i| (params[i][1] - y).abs() < 1e-4).collect() };
    let (ylo, yhi) = if py <= ys[0] {
        (ys[0], ys[0])
    } else if py >= *ys.last().unwrap() {
        (*ys.last().unwrap(), *ys.last().unwrap())
    } else {
        let mut hi = 1;
        while hi < ys.len() && ys[hi] < py {
            hi += 1;
        }
        (ys[hi - 1], ys[hi])
    };
    if (yhi - ylo).abs() < 1e-5 {
        return blend_1d(params, row(ylo), px);
    }
    let t = (py - ylo) / (yhi - ylo);
    let lo = blend_1d(params, row(ylo), px);
    let hi = blend_1d(params, row(yhi), px);
    let mut out = vec![0.0; n];
    for i in 0..n {
        out[i] = lo[i] * (1.0 - t) + hi[i] * t;
    }
    out
}

/// Engine expensive-locomotion pick (`FUN_007cbaf0`, `0x7cbaf0`): minimum
/// `locomotion_cost`. Ties (common when only the authored speed band differs)
/// prefer the candidate whose speed midpoint is closest to the desired
/// requested speed; exact ties keep document order. This is what separates
/// `RelaxedRunNW` (`_RUN_NW`, midpoint ≈ run) from `RelaxedWalkNW`
/// (`_WALK_NW`, midpoint ≈ walk), whose speed bands overlap.
fn pick_by_cost<F>(cands: &[usize], motion_of: &F, desired: &DesiredMotion) -> Option<usize>
where
    F: Fn(usize) -> Option<CandidateMotion>,
{
    let want = desired.requested_speed;
    let mut best: Option<(usize, f32, f32)> = None;
    for &i in cands {
        let Some(m) = motion_of(i) else { continue };
        let cost = crate::ag_cost::locomotion_cost(&m, desired);
        let mid = (m.speed.0 + m.speed.1) * 0.5;
        let take = match best {
            None => true,
            Some((_, bc, bm)) => {
                if cost < bc - 1e-4 {
                    true
                } else if cost > bc + 1e-4 {
                    false
                } else {
                    (mid - want).abs() < (bm - want).abs()
                }
            }
        };
        if take {
            best = Some((i, cost, mid));
        }
    }
    best.map(|(i, _, _)| i)
}

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
        self.update_with_inputs(m, |_| {}, motion_of, step)
    }

    /// [`update`](Self::update), letting the caller overwrite graph inputs
    /// before the candidates are evaluated. The player's state machine uses
    /// this to apply its `AnimationGraphInput` actions (workerA3b).
    pub fn update_with_inputs<F, A>(
        &mut self,
        m: &MovementSnapshot,
        apply: A,
        motion_of: F,
        step: bool,
    ) where
        F: Fn(usize) -> Option<CandidateMotion>,
        A: FnOnce(&mut InputValues),
    {
        let mut values = graph_inputs(&self.graph, m);
        apply(&mut values);
        let cands = self.graph.candidates(&values);
        let current = self.inst.current;
        // Expensive locomotion selection (`FUN_007cb5f0`/`FUN_007cbaf0`): when
        // the graph enables it (default), the engine scores *every* candidate
        // and keeps the minimum instead of holding the current state by
        // hysteresis (`FUN_007bd3a0`). That is what lets the idle state release
        // once the requested speed rises. Candidates without a resolvable
        // motion profile are skipped.
        let costed = if self.expensive_locomotion {
            pick_by_cost(&cands, &motion_of, &desired_motion(m, false))
        } else {
            None
        };
        // Fallback: 1. hysteresis, 2. previous target, 3. random among equals.
        let target = match costed {
            Some(t) => Some(t),
            None => {
                let keep = current
                    .filter(|c| cands.contains(c))
                    .or_else(|| self.inst.target.filter(|t| cands.contains(t)));
                match keep {
                    Some(t) => Some(t),
                    None => self.graph.pick_candidate(&cands, None, None, &mut self.inst.rng),
                }
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

    fn blend(blend_type: BlendType, params: &[[f32; 3]]) -> Resolved {
        Resolved {
            clips: params.iter().map(|_| (String::new(), 0.0)).collect(),
            kind: ResolvedKind::Blend(crate::anim_alias::BlendInfo {
                blend_type,
                caps: String::new(),
                params: params.to_vec(),
            }),
            alias: "test".into(),
        }
    }

    #[test]
    fn motion_from_speeds_single_and_blend() {
        let single = Resolved { clips: vec![("a".into(), 1.0)], kind: ResolvedKind::Single, alias: "a".into() };
        let m = motion_from_speeds(&single.kind, &[4.138]).unwrap();
        assert_eq!((m.speed.0, m.speed.1), (4.138, 4.138));

        let b = blend(BlendType::Flr2, &[[-1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]);
        let m = motion_from_speeds(&b.kind, &[1.723, 4.138]).unwrap();
        assert_eq!((m.speed.0, m.speed.1), (1.723, 4.138));
        assert!(motion_from_speeds(&b.kind, &[]).is_none());
    }

    #[test]
    fn blend_weights_1d_run_lean() {
        // relaxed_run.lmg: leanleft (-1), forward (0), leanright (1).
        let r = blend(BlendType::Flr1, &[[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]);
        assert_eq!(blend_weights(&r, 0.0, 1.0), vec![0.0, 1.0, 0.0]);
        let w = blend_weights(&r, 0.5, 1.0);
        assert!((w[1] - 0.5).abs() < 1e-5 && (w[2] - 0.5).abs() < 1e-5, "{w:?}");
        // Outside the range clamps to the end example.
        assert_eq!(blend_weights(&r, -2.0, 1.0), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn blend_weights_2d_walk_run_rows() {
        // relaxed_walk.lmg: run row at y=1, walk row at y=-1.
        let r = blend(
            BlendType::Flr2,
            &[
                [-1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
            ],
        );
        // Straight forward at run gait -> the run-forward example (index 1).
        assert_eq!(blend_weights(&r, 0.0, 1.0), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        // Straight forward at walk gait -> the walk-forward example (index 4).
        assert_eq!(blend_weights(&r, 0.0, -1.0), vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        // Half-way between the rows splits the weight.
        let w = blend_weights(&r, 0.0, 0.0);
        assert!((w[1] - 0.5).abs() < 1e-5 && (w[4] - 0.5).abs() < 1e-5, "{w:?}");
    }

    #[test]
    fn probe_relaxed_locomotion_clip_speeds() {
        use crate::anim_alias::{AnimInputs, AnimNameTable};
        use crate::crypak::Vfs;
        let game = std::env::var_os("SOD_GAME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(
                "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE",
            ));
        let graph_path = "/home/battlestation/Desktop/sod-re/extracted-game/animations/animations/graphs/playerfullbody.xml";
        let Ok(xml) = std::fs::read(graph_path) else { return };
        let g = Graph::parse(&xml).expect("parse");
        let mut vfs = Vfs::new();
        for rel in ["Game/animations.pak", "Game/characters.pak"] {
            let p = game.join(rel);
            if p.is_file() {
                let _ = vfs.mount(&p);
            }
        }
        let Some(table) = AnimNameTable::load_character(&vfs, "characters/male/human/ul_hero_male.cdf") else { return };
        let Some(skel) = vfs
            .read("characters/male/human/skeleton/skeleton.chr")
            .and_then(|b| crate::skin::parse_skeleton(&b))
        else {
            return;
        };
        let inputs = AnimInputs::new().with("stance", "relaxed").with("item", "nw");
        for name in ["RelaxedRunNW", "RelaxedWalkNW", "relaxed_idleToWalk_nw_01"] {
            let Some(&i) = g.by_id.get(&name.to_ascii_lowercase()) else { continue };
            let tpl = state_template(&g, i);
            let anim = tpl.animation.as_deref().unwrap_or("<none>");
            let mut speeds = Vec::new();
            if let Some(res) = tpl.animation.as_deref().and_then(|a| table.resolve(a, &inputs)) {
                for (path, _) in &res.clips {
                    if let Some(caf) = vfs.read(path).and_then(|b| crate::anim::Caf::parse(&b)) {
                        let rm = crate::locomotion::root_motion(&caf, &skel, caf.start as f32, caf.end as f32);
                        let d = caf.duration_secs().max(1e-6);
                        speeds.push(((rm.translation[0] * rm.translation[0] + rm.translation[1] * rm.translation[1]).sqrt() / d));
                    }
                }
                if let Some(m) = motion_from_resolved(&res, &speeds) {
                    eprintln!(
                        "PROBE {name}: anim={anim} caps={:?} motion.speed=({:.3},{:.3}) clips={}",
                        match &res.kind {
                            ResolvedKind::Blend(i) => i.caps.as_str(),
                            ResolvedKind::Single => "",
                        },
                        m.speed.0,
                        m.speed.1,
                        res.clips.len()
                    );
                }
            } else {
                eprintln!("PROBE {name}: anim={anim} unresolved");
            }
        }
    }

    #[test]
    fn state_template_reads_animation_and_playback_speed() {
        let path = "/home/battlestation/Desktop/sod-re/extracted-game/animations/animations/graphs/playerfullbody.xml";
        let Ok(xml) = std::fs::read(path) else { return };
        let g = Graph::parse(&xml).expect("parse");
        let run = g.state_index("RelaxedRunNW").unwrap();
        let t = state_template(&g, run);
        assert_eq!(t.animation.as_deref(), Some("_RELAXED_RUNSTRAFE_NW_01"));
        assert_eq!(t.name.as_deref(), Some("MovementDecoupled"));
        assert!(!t.one_shot());

        let itw = g.state_index("relaxed_idleToWalk_nw_01").unwrap();
        let t = state_template(&g, itw);
        assert_eq!(t.animation.as_deref(), Some("_RELAXED_IDLE2WALK_NW_01"));
        assert!(t.one_shot());
        assert!((t.speed() - 1.6).abs() < 1e-6);
    }
}
