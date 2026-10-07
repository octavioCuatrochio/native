//! Repeatable replay parity against the recorded retail-game captures in
//! `notes/telemetry` (see its `README.md` and `notes/parity.md`).
//!
//! Unlike [`crate::parity`] (which diffs two *canonical* [`crate::parity::Trace`]
//! CSVs), this module replays the raw captures produced by
//! `tools/game_probe.py` / `tools/camera_capture.py` through the native
//! controllers and scores the result:
//!
//! * **movement** — the binary/analog keyboard schedule of each `dyn.csv` /
//!   `session3.csv` / `turnpad.csv` / `turnjump.csv` phase is replayed through
//!   [`class3_move::update_normal`] with the shipped [`crate::tuning`] tuning at
//!   the recorded sample times, and the resulting speed/slide is compared with
//!   the actor positions (`px,py,pz`) in the capture.
//! * **camera stick** — each deflection level is replayed through
//!   [`camera::update`] and the camera yaw/pitch rate is compared with the
//!   derivative of the recorded `c1`/`c2` floats (`cam_stick*.csv`), or of
//!   `cam_qy`/`cam_qz` for the gamepad-turn captures (`turnpad.csv`,
//!   `turnjump.csv`).
//! * **camera mouse** — the recorded mouse counts are accumulated per game
//!   frame and replayed through [`camera::update`]; the per-frame yaw/pitch
//!   pose is compared with the recorded camera floats (`cam_mouse*.csv`,
//!   `cam_msweep_session2_*.csv`). Frame boundaries come from the recorded pose
//!   changes (or a file-level frame clock for gate-suppressed phases); see
//!   [`replay_mouse`]. The residual is frame-timing quantisation, not the
//!   rotation law.
//!
//! Everything is `std`-only and total: a missing column or empty phase is
//! skipped rather than panicking. The optional telemetry directory is located
//! through [`default_telemetry_dir`] (`CARGO_MANIFEST_DIR/../../../notes/telemetry`),
//! so unit tests silently skip when the captures are not checked out.

use crate::{camera, class3_move, tuning};
use std::path::{Path, PathBuf};

/// Default capture location relative to the crate manifest
/// (`native/crates/core/../../../notes/telemetry`).
pub fn default_telemetry_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../notes/telemetry")
}

// ---------------------------------------------------------------------------
// Pass/fail tolerances
// ---------------------------------------------------------------------------

/// Steady-state speed tolerance (m/s).
pub const STEADY_TOL: f32 = 0.5;
/// Acceleration ramp distance tolerance over the first 0.5 s (m).
pub const RAMP_DIST_TOL: f32 = 0.5;
/// Stop-slide distance tolerance (m).
pub const SLIDE_TOL: f32 = 0.5;
/// Camera stick rate tolerance (deg/s), vector error of yaw and pitch.
pub const STICK_RATE_TOL_DEG: f32 = 2.5;
/// Camera mouse per-frame RMS tolerance (deg).
///
/// The decoded per-frame law is exact, but the recorded 100/250 Hz captures do
/// not timestamp the game frame: the camera pose only changes on the frames
/// whose accumulated input clears the `|x|/|y| > 0.05` gate, so a frame's
/// duration can only be inferred from the gaps between pose changes (and game
/// hitches stretch a frame beyond one sample). The residual is therefore
/// dominated by frame-timing quantisation, not by the rotation model. The
/// measured mean per-phase RMS is ~6 deg across the four captures.
pub const MOUSE_FRAME_RMS_TOL_DEG: f32 = 35.0;
/// Longest frame time fed to the mouse rotation law (s). `Camera::Update`
/// consumes the real frame `dt`; a capture hitch (a 0.15 s gap between pose
/// changes) still produced < `dt`-proportional rotation, consistent with the
/// engine acting on a clamped frame time.
pub const MOUSE_DT_CLAMP: f32 = 0.08;
/// A phase with fewer than this many recorded pose changes is treated as
/// gate-suppressed (the camera updated but every frame's input stayed under
/// the dead zone), so its frames cannot be recovered from the pose and the
/// file-level frame clock is used instead.
pub const MOUSE_MIN_CHANGES: usize = 3;
/// Window used to turn the recorded actor positions into a speed curve (s).
pub const SPEED_WINDOW: f32 = 0.15;

// ---------------------------------------------------------------------------
// CSV plumbing
// ---------------------------------------------------------------------------

fn io_err(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

/// Split a capture CSV into a header and raw string rows (both trimmed).
fn read_table(path: &Path) -> std::io::Result<(Vec<String>, Vec<Vec<String>>)> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .ok_or_else(|| io_err("empty csv"))?
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let rows = lines
        .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
        .collect();
    Ok((header, rows))
}

fn col(cols: &[String], name: &str) -> Option<usize> {
    cols.iter().position(|c| c == name)
}

fn fv(row: &[String], i: usize) -> f32 {
    row.get(i).and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Captured sample types
// ---------------------------------------------------------------------------

/// One row of a `game_probe.py` capture (`t, phase, desired, px, py, pz,
/// cam_qy, cam_qz`).
#[derive(Clone, Debug, PartialEq)]
pub struct MovementSample {
    pub t: f32,
    pub phase: String,
    /// `controller+0x2c8`; zero exactly when the controller has no movement
    /// input, `~100.5` (the stance `maxSpeed`) when it does.
    pub desired: f32,
    /// Actor feet (`comp+0x24`), world Z-up metres.
    pub pos: [f32; 3],
    /// Camera pitch float (`cam_qy`, `c1`).
    pub cam_pitch: f32,
    /// Camera yaw float (`cam_qz`, `c2`).
    pub cam_yaw: f32,
}

/// One row of a `camera_capture.py` capture (`t, phase, stick_x, stick_y,
/// mouse_dx, mouse_dy, c1, c2`).
#[derive(Clone, Debug, PartialEq)]
pub struct CameraSample {
    pub t: f32,
    pub phase: String,
    pub stick: [f32; 2],
    pub mouse: [f32; 2],
    /// `c1` — camera pitch (rad).
    pub pitch: f32,
    /// `c2` — camera yaw (rad), opposite sign to [`camera::CameraState::yaw`].
    pub yaw: f32,
}

/// A camera rotation-rate sample (either format, see [`parse_rate_samples`]).
#[derive(Clone, Debug, PartialEq)]
pub struct RateSample {
    pub t: f32,
    pub phase: String,
    pub pitch: f32,
    pub yaw: f32,
    pub stick: [f32; 2],
}

/// Parse a `game_probe.py`-format movement/telemetry capture.
pub fn parse_movement(path: &Path) -> std::io::Result<Vec<MovementSample>> {
    let (cols, rows) = read_table(path)?;
    let ti = col(&cols, "t").ok_or_else(|| io_err("missing t"))?;
    let pi = col(&cols, "phase").ok_or_else(|| io_err("missing phase"))?;
    let di = col(&cols, "desired").ok_or_else(|| io_err("missing desired"))?;
    let xi = col(&cols, "px").ok_or_else(|| io_err("missing px"))?;
    let yi = col(&cols, "py").ok_or_else(|| io_err("missing py"))?;
    let zi = col(&cols, "pz").ok_or_else(|| io_err("missing pz"))?;
    let cpi = col(&cols, "cam_qy").ok_or_else(|| io_err("missing cam_qy"))?;
    let cyi = col(&cols, "cam_qz").ok_or_else(|| io_err("missing cam_qz"))?;
    Ok(rows
        .iter()
        .map(|r| MovementSample {
            t: fv(r, ti),
            phase: r.get(pi).cloned().unwrap_or_default(),
            desired: fv(r, di),
            pos: [fv(r, xi), fv(r, yi), fv(r, zi)],
            cam_pitch: fv(r, cpi),
            cam_yaw: fv(r, cyi),
        })
        .collect())
}

/// Parse a `camera_capture.py`-format capture.
pub fn parse_camera(path: &Path) -> std::io::Result<Vec<CameraSample>> {
    let (cols, rows) = read_table(path)?;
    let ti = col(&cols, "t").ok_or_else(|| io_err("missing t"))?;
    let pi = col(&cols, "phase").ok_or_else(|| io_err("missing phase"))?;
    let sxi = col(&cols, "stick_x").ok_or_else(|| io_err("missing stick_x"))?;
    let syi = col(&cols, "stick_y").ok_or_else(|| io_err("missing stick_y"))?;
    let mxi = col(&cols, "mouse_dx").ok_or_else(|| io_err("missing mouse_dx"))?;
    let myi = col(&cols, "mouse_dy").ok_or_else(|| io_err("missing mouse_dy"))?;
    let c1i = col(&cols, "c1").ok_or_else(|| io_err("missing c1"))?;
    let c2i = col(&cols, "c2").ok_or_else(|| io_err("missing c2"))?;
    Ok(rows
        .iter()
        .map(|r| CameraSample {
            t: fv(r, ti),
            phase: r.get(pi).cloned().unwrap_or_default(),
            stick: [fv(r, sxi), fv(r, syi)],
            mouse: [fv(r, mxi), fv(r, myi)],
            pitch: fv(r, c1i),
            yaw: fv(r, c2i),
        })
        .collect())
}

/// Parse either camera-capture or gamepad-telemetry format for rate analysis.
///
/// `cam_stick*.csv` record `stick_x/stick_y` and `c1/c2`; `turnpad.csv` /
/// `turnjump.csv` are `game_probe.py` rows where the same floats are the
/// `cam_qy`/`cam_qz` columns and the stick is not captured (it is supplied by
/// [`rate_overrides`]).
pub fn parse_rate_samples(path: &Path) -> std::io::Result<Vec<RateSample>> {
    let (cols, _) = read_table(path)?;
    if col(&cols, "stick_x").is_some() {
        Ok(parse_camera(path)?
            .into_iter()
            .map(|c| RateSample {
                t: c.t,
                phase: c.phase,
                pitch: c.pitch,
                yaw: c.yaw,
                stick: c.stick,
            })
            .collect())
    } else {
        Ok(parse_movement(path)?
            .into_iter()
            .map(|m| RateSample {
                t: m.t,
                phase: m.phase,
                pitch: m.cam_pitch,
                yaw: m.cam_yaw,
                stick: [0.0, 0.0],
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Movement replay
// ---------------------------------------------------------------------------

/// One movement phase to replay. The key schedule is binary (the retail
/// keyboard is digital); the controller's shipped inertia produces the ramp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveScenario {
    pub file: &'static str,
    pub phase: &'static str,
    pub forward: f32,
    pub strafe: f32,
    pub sprint: bool,
    pub crouch: bool,
    /// `true` for the `*_start_stop` phases that release the key mid-phase.
    pub released: bool,
}

/// The movement phases the harness scores, reconstructed from
/// `tools/game_probe.py::scenario_default` / `scenario_dyn` and
/// `cmd_turnjump`.
pub fn movement_scenarios() -> Vec<MoveScenario> {
    use MoveScenario as S;
    vec![
        S { file: "dyn.csv", phase: "run_start_stop", forward: 1.0, strafe: 0.0, sprint: false, crouch: false, released: true },
        S { file: "dyn.csv", phase: "sprint_start_stop", forward: 1.0, strafe: 0.0, sprint: true, crouch: false, released: true },
        S { file: "dyn.csv", phase: "back_start_stop", forward: -1.0, strafe: 0.0, sprint: false, crouch: false, released: true },
        S { file: "dyn.csv", phase: "strafeA_start_stop", forward: 0.0, strafe: 1.0, sprint: false, crouch: false, released: true },
        S { file: "dyn.csv", phase: "creep_probe", forward: 1.0, strafe: 0.0, sprint: false, crouch: true, released: false },
        S { file: "session3.csv", phase: "run_W", forward: 1.0, strafe: 0.0, sprint: false, crouch: false, released: false },
        S { file: "session3.csv", phase: "run_W2", forward: 1.0, strafe: 0.0, sprint: false, crouch: false, released: false },
        S { file: "session3.csv", phase: "sprint_W_shift", forward: 1.0, strafe: 0.0, sprint: true, crouch: false, released: false },
        S { file: "session3.csv", phase: "sprint2", forward: 1.0, strafe: 0.0, sprint: true, crouch: false, released: false },
        S { file: "session3.csv", phase: "back_S", forward: -1.0, strafe: 0.0, sprint: false, crouch: false, released: false },
        S { file: "session3.csv", phase: "strafe_A", forward: 0.0, strafe: 1.0, sprint: false, crouch: false, released: false },
        S { file: "session3.csv", phase: "strafe_D", forward: 0.0, strafe: -1.0, sprint: false, crouch: false, released: false },
        S { file: "session3.csv", phase: "diag_WA", forward: 1.0, strafe: 1.0, sprint: false, crouch: false, released: false },
        S { file: "session3.csv", phase: "creep_ctrl_W", forward: 1.0, strafe: 0.0, sprint: false, crouch: true, released: false },
        S { file: "turnpad.csv", phase: "run_rstick", forward: 1.0, strafe: 0.0, sprint: false, crouch: false, released: false },
        S { file: "turnjump.csv", phase: "run_W", forward: 1.0, strafe: 0.0, sprint: false, crouch: false, released: false },
        S { file: "turnjump.csv", phase: "run_turn_r", forward: 1.0, strafe: 0.0, sprint: false, crouch: false, released: false },
        S { file: "turnjump.csv", phase: "run_turn_l", forward: 1.0, strafe: 0.0, sprint: false, crouch: false, released: false },
        S { file: "turnjump.csv", phase: "run_jump", forward: 1.0, strafe: 0.0, sprint: false, crouch: false, released: false },
    ]
}

/// Per-scenario movement error summary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MovementError {
    pub steady_sim: f32,
    pub steady_rec: f32,
    pub steady_err: f32,
    /// RMS of (recorded speed - simulated speed) over the ramp window (m/s).
    pub ramp_rms: f32,
    /// Max of the same (m/s).
    pub ramp_max: f32,
    /// |distance travelled in the first 0.5 s| error (m).
    pub ramp_dist_err: f32,
    pub slide_sim: Option<f32>,
    pub slide_rec: Option<f32>,
    pub slide_err: Option<f32>,
    pub samples: usize,
}

impl MovementError {
    /// All three robust metrics within tolerance.
    pub fn pass(&self) -> bool {
        self.steady_err <= STEADY_TOL
            && self.ramp_dist_err <= RAMP_DIST_TOL
            && self.slide_err.map_or(true, |e| e <= SLIDE_TOL)
    }
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn arc_len(pos: &[[f32; 2]], a: usize, b: usize) -> f32 {
    if b <= a {
        return 0.0;
    }
    (a + 1..=b).map(|i| dist2(pos[i - 1], pos[i])).sum()
}

/// Replay one movement phase and score it against the recorded actor path.
///
/// The controller's input-active window is recovered from the recorded
/// `desired` field: it transitions from `0` to `~100.5` when the movement
/// controller receives input (this includes the capture rig's own latency), so
/// replaying that window reproduces the exact input timing the game acted on.
/// Returns `None` when the phase is absent or no input edge is found.
pub fn replay_movement(samples: &[MovementSample], sc: &MoveScenario) -> Option<MovementError> {
    let ph: Vec<&MovementSample> = samples.iter().filter(|s| s.phase == sc.phase).collect();
    if ph.len() < 12 {
        return None;
    }
    let t0 = ph[0].t;
    let ts: Vec<f32> = ph.iter().map(|s| s.t - t0).collect();
    let pos_rec: Vec<[f32; 2]> = ph.iter().map(|s| [s.pos[0], s.pos[1]]).collect();
    let desired: Vec<f32> = ph.iter().map(|s| s.desired).collect();

    let press = desired.iter().position(|&d| d > 1.0)?;
    let release = if sc.released {
        (press + 1..ph.len()).find(|&i| desired[i] <= 1.0)
    } else {
        None
    };
    let n = ph.len();

    // Recorded speed curve: arc length over a sliding window, divided by the
    // actual elapsed time (smooth enough to absorb the 50 Hz game / 100 Hz
    // sampler beat).
    let mut rec_speed = vec![0.0f32; n];
    let mut j = 0usize;
    for i in 0..n {
        while ts[i] - ts[j] > SPEED_WINDOW {
            j += 1;
        }
        let dt = ts[i] - ts[j];
        rec_speed[i] = if dt > 1e-6 {
            dist2(pos_rec[i], pos_rec[j]) / dt
        } else {
            0.0
        };
    }

    // Replay through the shipped controller.
    let tuning = tuning::default_movement();
    let z0 = ph[0].pos[2];
    let ground = move |_x: f32, _y: f32| z0;
    let mut state = class3_move::MoveState::new([pos_rec[0][0], pos_rec[0][1], z0]);
    let mut sim_speed = vec![0.0f32; n];
    let mut sim_pos = vec![[0.0f32; 2]; n];
    for i in 0..n {
        let dt = if i == 0 {
            0.0
        } else {
            (ph[i].t - ph[i - 1].t).clamp(0.0, 0.2)
        };
        let active = i >= press && release.map_or(true, |r| i < r);
        let input = class3_move::MoveInput {
            forward: if active { sc.forward } else { 0.0 },
            strafe: if active { sc.strafe } else { 0.0 },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            crouch: sc.crouch,
            sprint: sc.sprint,
            dt,
            aim_yaw: None,
        };
        class3_move::update_normal(&mut state, &input, &tuning, &ground, &[]);
        sim_speed[i] = (state.vel[0] * state.vel[0] + state.vel[1] * state.vel[1]).sqrt();
        sim_pos[i] = [state.pos[0], state.pos[1]];
    }

    // Simulated curve smoothed with the same window so the ramp comparison is
    // apples-to-apples.
    let mut sim_win = vec![0.0f32; n];
    let mut j = 0usize;
    for i in 0..n {
        while ts[i] - ts[j] > SPEED_WINDOW {
            j += 1;
        }
        let dt = ts[i] - ts[j];
        sim_win[i] = if dt > 1e-6 {
            dist2(sim_pos[i], sim_pos[j]) / dt
        } else {
            0.0
        };
    }

    // Steady window: 0.9 s after the press to just before the release (or end).
    let hold_end = release.unwrap_or(n - 1);
    let a = (press..n).find(|&i| ts[i] >= ts[press] + 0.9).unwrap_or(hold_end);
    let b = hold_end.saturating_sub(2).min(n - 1);
    let steady_rec = if b > a {
        rec_speed[a..b].iter().sum::<f32>() / (b - a) as f32
    } else {
        rec_speed.get(hold_end).copied().unwrap_or(0.0)
    };
    let steady_sim = if b > a {
        sim_speed[a..b].iter().sum::<f32>() / (b - a) as f32
    } else {
        sim_speed.get(hold_end).copied().unwrap_or(0.0)
    };

    // Ramp: distance travelled in the first 0.5 s, both sides.
    let i05 = (press..n)
        .find(|&i| ts[i] >= ts[press] + 0.5)
        .unwrap_or(n - 1);
    let ramp_dist_err =
        (arc_len(&pos_rec, press, i05) - arc_len(&sim_pos, press, i05)).abs();
    let (mut ramp_rms, mut ramp_max) = (0.0f32, 0.0f32);
    let ramp_n = i05.saturating_sub(press).max(1);
    for i in press..i05 {
        let e = (rec_speed[i] - sim_win[i]).abs();
        ramp_rms += e * e;
        ramp_max = ramp_max.max(e);
    }
    ramp_rms = (ramp_rms / ramp_n as f32).sqrt();

    // Stop slide: remaining travel after the release until the actor stops.
    let (slide_sim, slide_rec, slide_err) = if let Some(r) = release {
        let rec_end = (r..n).filter(|&i| rec_speed[i] > 0.15).last().unwrap_or(r);
        let sim_end = (r..n).filter(|&i| sim_speed[i] > 0.15).last().unwrap_or(r);
        let sr = arc_len(&pos_rec, r, rec_end);
        let ss = arc_len(&sim_pos, r, sim_end);
        (Some(ss), Some(sr), Some((ss - sr).abs()))
    } else {
        (None, None, None)
    };

    Some(MovementError {
        steady_sim,
        steady_rec,
        steady_err: (steady_sim - steady_rec).abs(),
        ramp_rms,
        ramp_max,
        ramp_dist_err,
        slide_sim,
        slide_rec,
        slide_err,
        samples: n,
    })
}

// ---------------------------------------------------------------------------
// Camera stick replay
// ---------------------------------------------------------------------------

/// A stick level supplied by the phase name when the capture did not record
/// the device value (the gamepad-turn captures).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateOverride {
    pub file: &'static str,
    pub phase: &'static str,
    pub stick: [f32; 2],
}

/// Gamepad right-stick levels for `turnpad.csv` / `turnjump.csv`
/// (`tools/game_probe.py::cmd_turnjump`, `virtual_gamepad.py`), full deflection.
pub fn rate_overrides() -> Vec<RateOverride> {
    vec![
        RateOverride { file: "turnpad.csv", phase: "rstick_right", stick: [1.0, 0.0] },
        RateOverride { file: "turnpad.csv", phase: "rstick_left", stick: [-1.0, 0.0] },
        RateOverride { file: "turnpad.csv", phase: "run_rstick", stick: [1.0, 0.0] },
        RateOverride { file: "turnjump.csv", phase: "run_turn_r", stick: [1.0, 0.0] },
        RateOverride { file: "turnjump.csv", phase: "run_turn_l", stick: [-1.0, 0.0] },
        RateOverride { file: "turnjump.csv", phase: "idle_turn", stick: [1.0, 0.0] },
    ]
}

/// Per-phase camera rotation-rate error (degrees/second).
#[derive(Clone, Debug, PartialEq)]
pub struct RateError {
    pub file: String,
    pub phase: String,
    pub pred_yaw: f32,
    pub meas_yaw: f32,
    pub pred_pitch: f32,
    pub meas_pitch: f32,
    /// Vector (yaw, pitch) error magnitude, degrees/second.
    pub err: f32,
}

impl RateError {
    pub fn pass(&self) -> bool {
        self.err <= STICK_RATE_TOL_DEG
    }
}

/// Unwrap a sequence of angles to a continuous ramp starting at the first value.
fn unwrap(v: &[f32]) -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    let pi = std::f32::consts::PI;
    let mut out = Vec::with_capacity(v.len());
    if let Some(&first) = v.first() {
        out.push(first);
    }
    for &x in v.iter().skip(1) {
        let last = *out.last().unwrap();
        let mut d = x - last;
        while d > pi {
            d -= tau;
        }
        while d < -pi {
            d += tau;
        }
        out.push(last + d);
    }
    out
}

/// Least-squares slope of `v` against `t`.
fn regress(t: &[f32], v: &[f32]) -> f32 {
    if t.len() < 2 || t.len() != v.len() {
        return 0.0;
    }
    let n = t.len() as f32;
    let mt = t.iter().sum::<f32>() / n;
    let mv = v.iter().sum::<f32>() / n;
    let num: f32 = t.iter().zip(v).map(|(a, b)| (a - mt) * (b - mv)).sum();
    let den: f32 = t.iter().map(|a| (a - mt) * (a - mt)).sum();
    if den > 1e-9 {
        num / den
    } else {
        0.0
    }
}

/// Measured yaw/pitch rate (rad/s, [`camera::CameraState`] sign convention) of a
/// phase, from its recorded camera floats.
///
/// The recorded `c2` yaw runs opposite to [`camera::CameraState::yaw`] (a
/// positive stick/`mouse_dx` decreases `c2`), so the yaw slope is negated;
/// pitch has the same sign.
pub fn measured_rate(ts: &[f32], yaw_c2: &[f32], pitch_c1: &[f32]) -> (f32, f32) {
    let y = unwrap(yaw_c2);
    let p = unwrap(pitch_c1);
    let n = ts.len();
    let (a, b) = (n * 2 / 10, n * 8 / 10);
    if b <= a {
        return (0.0, 0.0);
    }
    (-regress(&ts[a..b], &y[a..b]), regress(&ts[a..b], &p[a..b]))
}

/// Replay every phase of a rotation capture through [`camera::update`] and
/// compare the steady rate with the recorded camera float derivative.
pub fn replay_rates(samples: &[RateSample], overrides: &[RateOverride], file: &str) -> Vec<RateError> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < samples.len() {
        let phase = samples[i].phase.clone();
        let a = i;
        while i < samples.len() && samples[i].phase == phase {
            i += 1;
        }
        let b = i;
        if phase == "idle" || b - a < 10 {
            continue;
        }
        let stick = overrides
            .iter()
            .find(|o| o.file == file && o.phase == phase.as_str())
            .map(|o| o.stick)
            .unwrap_or(samples[a].stick);
        if stick == [0.0, 0.0] {
            continue;
        }

        let ts: Vec<f32> = samples[a..b].iter().map(|s| s.t).collect();
        let yaw: Vec<f32> = samples[a..b].iter().map(|s| s.yaw).collect();
        let pitch: Vec<f32> = samples[a..b].iter().map(|s| s.pitch).collect();
        let (meas_yaw, meas_pitch) = measured_rate(&ts, &yaw, &pitch);

        let mut st = camera::CameraState::new();
        let input = camera::CameraInput { stick, dt: 1.0, ..Default::default() };
        camera::update(&mut st, &input, &tuning::default_camera(), [0.0; 3]);
        let pred_yaw = st.yaw_vel;
        let pred_pitch = st.pitch_vel;

        let dy = pred_yaw - meas_yaw;
        let dp = pred_pitch - meas_pitch;
        let err = (dy * dy + dp * dp).sqrt().to_degrees();

        out.push(RateError {
            file: file.to_string(),
            phase,
            pred_yaw: pred_yaw.to_degrees(),
            meas_yaw: meas_yaw.to_degrees(),
            pred_pitch: pred_pitch.to_degrees(),
            meas_pitch: meas_pitch.to_degrees(),
            err,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Camera mouse replay
// ---------------------------------------------------------------------------

/// Per-phase mouse replay error (degrees).
#[derive(Clone, Debug, PartialEq)]
pub struct MouseError {
    pub file: String,
    pub phase: String,
    /// RMS of the per-frame camera pose error, degrees.
    pub frame_rms: f32,
    /// Max of the same, degrees.
    pub frame_max: f32,
    /// Accumulated yaw predicted by [`camera::update`], degrees.
    pub total_pred: f32,
    /// Accumulated yaw observed in `c2`, degrees.
    pub total_rec: f32,
    /// Final accumulated yaw error, degrees.
    pub total_err: f32,
    pub samples: usize,
}

impl MouseError {
    pub fn pass(&self) -> bool {
        self.frame_rms <= MOUSE_FRAME_RMS_TOL_DEG
    }
}

/// Infer a file-level game-frame clock (Hz) from the recorded pose updates.
///
/// The game frame is the frame on which the camera pose changes, so the busiest
/// mouse phase is a lower bound on the frame rate. Returns at least 1 Hz, and
/// 60 Hz when the file has no usable phases.
pub fn infer_mouse_fps(samples: &[CameraSample]) -> f32 {
    let mut best = 0.0f32;
    let mut i = 0;
    while i < samples.len() {
        let phase = samples[i].phase.clone();
        let a = i;
        while i < samples.len() && samples[i].phase == phase {
            i += 1;
        }
        let b = i;
        if phase == "idle" || b - a < 10 {
            continue;
        }
        let changes = (a + 1..b)
            .filter(|&k| samples[k].yaw != samples[k - 1].yaw || samples[k].pitch != samples[k - 1].pitch)
            .count() as f32;
        let dur = (samples[b - 1].t - samples[a].t).max(1e-3);
        best = best.max(changes / dur);
    }
    if best.is_finite() && best >= 1.0 {
        best.min(1000.0)
    } else {
        60.0
    }
}

/// Wrap an angle difference to `(-pi, pi]`.
fn wrap_pi(mut d: f32) -> f32 {
    while d > core::f32::consts::PI {
        d -= core::f32::consts::TAU;
    }
    while d <= -core::f32::consts::PI {
        d += core::f32::consts::TAU;
    }
    d
}

/// Replay the recorded mouse counts through [`camera::update`] and compare the
/// pose to the recorded `c1`/`c2`.
///
/// The engine consumes mouse counts **once per game frame**: `0x1588ee0`
/// accumulates events into `player+0xc8/+0xcc` and `0x15ea59c` applies the
/// clamped value with the frame `dt`. The capture samples at 100/250 Hz and
/// does not record the game frame, but the camera pose changes exactly on the
/// frames whose input clears the `0.05` gate, so the gaps between pose changes
/// give the frame boundaries. For gate-suppressed phases (fewer than
/// [`MOUSE_MIN_CHANGES`] pose changes but non-zero input) the frames are
/// invisible and the file-level [`infer_mouse_fps`] clock is used instead.
/// Frame times are capped at [`MOUSE_DT_CLAMP`].
///
/// The recorded pitch bounds are not static (they come from the camera-spec
/// behaviour object), so the observed `c1` range of the file seeds
/// [`camera::CameraState::pitch_min`]/`pitch_max`.
pub fn replay_mouse(samples: &[CameraSample], file: &str) -> Vec<MouseError> {
    let fps = infer_mouse_fps(samples);
    let period = if fps.is_finite() && fps > 0.0 { 1.0 / fps } else { 1.0 / 60.0 };
    let (plo, phi) = samples
        .iter()
        .map(|s| s.pitch)
        .filter(|v| v.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
    let (plo, phi) = if plo.is_finite() && phi.is_finite() && plo <= phi {
        (plo, phi)
    } else {
        (-camera::PITCH_LIMIT, camera::PITCH_LIMIT)
    };

    let mut out = Vec::new();
    let mut i = 0;
    while i < samples.len() {
        let phase = samples[i].phase.clone();
        let a = i;
        while i < samples.len() && samples[i].phase == phase {
            i += 1;
        }
        let b = i;
        if phase == "idle" || b - a < 10 {
            continue;
        }
        let any_mouse = samples[a..b]
            .iter()
            .any(|s| s.mouse[0] != 0.0 || s.mouse[1] != 0.0);
        if !any_mouse {
            continue;
        }
        let changes = (a + 1..b)
            .filter(|&k| samples[k].yaw != samples[k - 1].yaw || samples[k].pitch != samples[k - 1].pitch)
            .count();
        let use_fixed = changes < MOUSE_MIN_CHANGES;

        let yaw0 = -samples[a].yaw;
        let mut st = camera::CameraState::new();
        st.yaw = yaw0;
        st.pitch = samples[a].pitch;
        st.pitch_min = plo;
        st.pitch_max = phi;

        let mut acc = [0.0f32; 2];
        let mut last_t = samples[a].t;
        let mut ss = 0.0f32;
        let mut mx = 0.0f32;
        let mut cnt = 0usize;
        let mut total_rec = 0.0f32;
        let mut last_pred = (st.yaw, st.pitch);
        for k in a..b {
            acc[0] += samples[k].mouse[0];
            acc[1] += samples[k].mouse[1];
            if k > a {
                total_rec += wrap_pi((-samples[k].yaw) - (-samples[k - 1].yaw));
            }
            let changed = k > a
                && (samples[k].yaw != samples[k - 1].yaw || samples[k].pitch != samples[k - 1].pitch);
            let apply = if use_fixed {
                samples[k].t - last_t >= period
            } else {
                changed
            };
            if apply {
                let dt = (samples[k].t - last_t).clamp(0.0, MOUSE_DT_CLAMP);
                let input = camera::CameraInput {
                    mouse_dx: acc[0],
                    mouse_dy: acc[1],
                    stick: [0.0, 0.0],
                    aim: false,
                    dt,
                };
                camera::update(&mut st, &input, &tuning::default_camera(), [0.0; 3]);
                acc = [0.0, 0.0];
                last_t = samples[k].t;
            }
            // `CameraState::yaw` runs opposite to the recorded `c2`.
            let dy = wrap_pi(st.yaw - (-samples[k].yaw));
            let dp = st.pitch - samples[k].pitch;
            let e = (dy * dy + dp * dp).sqrt().to_degrees();
            ss += e * e;
            mx = mx.max(e);
            cnt += 1;
            last_pred = (st.yaw, st.pitch);
        }
        let total_pred = (last_pred.0 - yaw0).to_degrees();
        let total_rec = total_rec.to_degrees();
        out.push(MouseError {
            file: file.to_string(),
            phase,
            frame_rms: (ss / cnt.max(1) as f32).sqrt(),
            frame_max: mx,
            total_pred,
            total_rec,
            total_err: (total_pred - total_rec).abs(),
            samples: b - a,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_dir() -> Option<PathBuf> {
        let d = default_telemetry_dir();
        if d.join("dyn.csv").is_file() {
            Some(d)
        } else {
            None
        }
    }

    #[test]
    fn unwrap_is_continuous_across_pi() {
        let v = vec![3.0, -3.0, -3.0 + 0.1];
        let u = unwrap(&v);
        assert!((u[1] - (3.0 + (std::f32::consts::TAU - 6.0))).abs() < 1e-5, "{u:?}");
        for w in u.windows(2) {
            assert!((w[1] - w[0]).abs() <= std::f32::consts::PI + 1e-5);
        }
    }

    #[test]
    fn regress_recovers_a_slope() {
        let t = [0.0, 1.0, 2.0, 3.0];
        let v = [1.0, 3.0, 5.0, 7.0];
        assert!((regress(&t, &v) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn stick_full_deflection_is_ninety_degrees_per_second() {
        let mut st = camera::CameraState::new();
        let input = camera::CameraInput { stick: [1.0, 0.0], dt: 1.0, ..Default::default() };
        camera::update(&mut st, &input, &tuning::default_camera(), [0.0; 3]);
        assert!((st.yaw_vel.to_degrees() - 90.0).abs() < 0.01, "{}", st.yaw_vel);
    }

    /// The `cam_stick*.csv` sweep must be reproduced to within the tolerance.
    #[test]
    fn stick_captures_pass() {
        let Some(dir) = diag_dir() else { return };
        for file in ["cam_stick.csv", "cam_stick_fine.csv"] {
            let samples = parse_rate_samples(&dir.join(file)).expect("parse");
            let errs = replay_rates(&samples, &[], file);
            assert!(!errs.is_empty(), "no phases in {file}");
            for e in &errs {
                assert!(e.pass(), "{file}/{}: err={:.2} deg/s", e.phase, e.err);
            }
        }
    }

    /// The gamepad-turn captures (full-stick phases) must also pass.
    #[test]
    fn turn_captures_pass() {
        let Some(dir) = diag_dir() else { return };
        let ov = rate_overrides();
        for file in ["turnpad.csv", "turnjump.csv"] {
            let samples = parse_rate_samples(&dir.join(file)).expect("parse");
            let errs = replay_rates(&samples, &ov, file);
            for e in errs.iter().filter(|e| !e.phase.starts_with("run_") || e.phase == "run_rstick") {
                assert!(e.pass(), "{file}/{}: err={:.2} deg/s", e.phase, e.err);
            }
        }
    }

    /// Clean flat-ground run/sprint phases must reproduce the shipped speeds.
    #[test]
    fn clean_movement_phases_pass() {
        let Some(dir) = diag_dir() else { return };
        let mut cache: Vec<(String, Vec<MovementSample>)> = Vec::new();
        for sc in movement_scenarios() {
            let samples = match cache.iter().find(|(f, _)| f == sc.file) {
                Some((_, s)) => s.clone(),
                None => {
                    let s = parse_movement(&dir.join(sc.file)).expect("parse");
                    cache.push((sc.file.to_string(), s.clone()));
                    s
                }
            };
            let e = replay_movement(&samples, &sc).expect("phase present");
            let clean_run = matches!(
                (sc.file, sc.phase),
                ("dyn.csv", "run_start_stop")
                    | ("session3.csv", "run_W")
                    | ("session3.csv", "run_W2")
                    | ("session3.csv", "sprint_W_shift")
                    | ("session3.csv", "sprint2")
                    | ("turnjump.csv", "run_W")
            );
            if clean_run {
                assert!(e.pass(), "{}/{}: {e:?}", sc.file, sc.phase);
            }
        }
    }

    /// The mouse pipeline (per-frame accumulation, `±1` clamp, `dt` scaling and
    /// the `0.05` gate) must reproduce the virtual-mouse sweeps and the guided
    /// captures. The per-phase frame RMS tolerance
    /// ([`MOUSE_FRAME_RMS_TOL_DEG`]) is dominated by the capture's frame-timing
    /// quantisation, so the aggregate mean is also locked tighter.
    #[test]
    fn mouse_captures_pass() {
        let Some(dir) = diag_dir() else { return };
        let files = [
            "cam_mouse.csv",
            "cam_mouse_flat.csv",
            "cam_msweep_session2_100hz.csv",
            "cam_msweep_session2_250hz.csv",
        ];
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for file in files {
            let samples = match parse_camera(&dir.join(file)) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let errs = replay_mouse(&samples, file);
            assert!(!errs.is_empty(), "no mouse phases in {file}");
            for e in &errs {
                assert!(e.pass(), "{file}/{}: frame rms {:.2} deg", e.phase, e.frame_rms);
                sum += e.frame_rms;
                n += 1;
            }
        }
        assert!(n > 0, "no mouse captures present");
        let mean = sum / n as f32;
        assert!(mean <= 8.0, "mean per-phase mouse frame RMS {mean:.2} deg");
    }

    /// The virtual-mouse sweep is the strongest constraint on the decoded
    /// pipeline: 1/2/4 counts/event grow super-linearly, 8/16/32 saturate, and
    /// 100/400-count flicks give the same (clamped) turn.
    #[test]
    fn mouse_sweep_nonlinearity_is_reproduced() {
        let Some(dir) = diag_dir() else { return };
        let path = dir.join("cam_msweep_session2_100hz.csv");
        let samples = match parse_camera(&path) {
            Ok(s) => s,
            Err(_) => return,
        };
        // Net recorded yaw per right-turn phase must be monotone in counts and
        // saturate: |yaw(32)| > |yaw(16)| > |yaw(4)| > |yaw(1)|.
        let net = |phase: &str| -> Option<f32> {
            let p: Vec<&CameraSample> = samples.iter().filter(|s| s.phase == phase).collect();
            if p.len() < 10 {
                return None;
            }
            let mut acc = 0.0f32;
            for w in p.windows(2) {
                acc += wrap_pi((-w[1].yaw) - (-w[0].yaw));
            }
            Some(acc.to_degrees().abs())
        };
        let a = net("right_1").unwrap();
        let b = net("right_4").unwrap();
        let c = net("right_16").unwrap();
        let d = net("right_32").unwrap();
        assert!(a < b, "1 vs 4 counts: {a:.1} !< {b:.1}");
        assert!(b < c, "4 vs 16 counts: {b:.1} !< {c:.1}");
        assert!(d <= c + 20.0, "32 should saturate near 16: {c:.1} vs {d:.1}");
    }
}
