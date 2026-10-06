//! Deterministic fixed-step parity traces + tolerance comparison.
//!
//! A [`Trace`] is a spawn point plus a fixed-step sequence of [`StateSample`]s
//! produced from a matching sequence of [`InputFrame`]s. Traces serialise to a
//! stable, round-trippable CSV so a native controller and real-game telemetry
//! (see `tools/parity_real.py`) can be diffed objectively with [`compare`].

use std::fmt::Write as _;
use std::path::Path;

/// One fixed-step input command applied to a controller.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct InputFrame {
    pub forward: f32,
    pub strafe: f32,
    pub yaw_delta: f32,
    pub sprint: bool,
    pub crouch: bool,
    pub jump: bool,
    pub dt: f32,
}

/// The observable state produced by one controller step.
#[derive(Clone, PartialEq, Debug)]
pub struct StateSample {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub yaw: f32,
    pub speed: f32,
    pub gait: u8,
    pub grounded: bool,
}

/// A deterministic controller advanced by fixed-step [`InputFrame`]s.
pub trait Controller {
    fn reset(&mut self, spawn: [f32; 3]);
    fn step(&mut self, input: &InputFrame) -> StateSample;
}

/// A recorded trace: spawn, per-step inputs, and per-step samples.
#[derive(Clone, PartialEq, Debug)]
pub struct Trace {
    pub spawn: [f32; 3],
    pub samples: Vec<StateSample>,
    pub inputs: Vec<InputFrame>,
}

/// Run `inputs` through `ctrl` from `spawn`, capturing a trace.
pub fn record<C: Controller>(ctrl: &mut C, spawn: [f32; 3], inputs: &[InputFrame]) -> Trace {
    ctrl.reset(spawn);
    let mut samples = Vec::with_capacity(inputs.len());
    for input in inputs {
        samples.push(ctrl.step(input));
    }
    Trace { spawn, samples, inputs: inputs.to_vec() }
}

// --- CSV serialisation (stable column order) --------------------------------

const HEADER: &str =
    "tick,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,yaw,speed,gait,grounded,\
in_forward,in_strafe,in_yaw_delta,in_sprint,in_crouch,in_jump,in_dt";

/// Serialise a trace to the canonical CSV format.
pub fn to_csv(t: &Trace) -> String {
    let mut out = String::new();
    out.push_str("#sod-parity-trace v1\n");
    let _ = writeln!(out, "spawn,{},{},{}", t.spawn[0], t.spawn[1], t.spawn[2]);
    out.push_str(HEADER);
    out.push('\n');
    let n = t.samples.len().max(t.inputs.len());
    for i in 0..n {
        let s = &t.samples[i];
        let f = &t.inputs[i];
        let _ = writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            i,
            s.pos[0], s.pos[1], s.pos[2],
            s.vel[0], s.vel[1], s.vel[2],
            s.yaw, s.speed, s.gait, s.grounded as u8,
            f.forward, f.strafe, f.yaw_delta,
            f.sprint as u8, f.crouch as u8, f.jump as u8, f.dt
        );
    }
    out
}

fn csv_row(line: &str) -> Option<Vec<&str>> {
    let v: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    if v.len() == 18 {
        Some(v)
    } else {
        None
    }
}

/// Parse a trace from the canonical CSV format. Returns `None` on malformed input.
pub fn from_csv(s: &str) -> Option<Trace> {
    let mut lines = s
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let spawn_line = lines.next()?;
    let sp: Vec<&str> = spawn_line.split(',').map(|s| s.trim()).collect();
    if sp.len() != 4 || sp[0] != "spawn" {
        return None;
    }
    let spawn = [sp[1].parse().ok()?, sp[2].parse().ok()?, sp[3].parse().ok()?];

    let header = lines.next()?;
    if !header.starts_with("tick,") {
        return None;
    }

    let mut samples = Vec::new();
    let mut inputs = Vec::new();
    for line in lines {
        let v = csv_row(line)?;
        let g = |i: usize| -> Option<f32> { v[i].parse().ok() };
        let b = |i: usize| v[i] != "0";
        samples.push(StateSample {
            pos: [g(1)?, g(2)?, g(3)?],
            vel: [g(4)?, g(5)?, g(6)?],
            yaw: g(7)?,
            speed: g(8)?,
            gait: v[9].parse().ok()?,
            grounded: b(10),
        });
        inputs.push(InputFrame {
            forward: g(11)?,
            strafe: g(12)?,
            yaw_delta: g(13)?,
            sprint: b(14),
            crouch: b(15),
            jump: b(16),
            dt: g(17)?,
        });
    }
    Some(Trace { spawn, samples, inputs })
}

// --- Files -------------------------------------------------------------------

/// Write a trace as canonical CSV.
pub fn write_trace(t: &Trace, path: &Path) -> std::io::Result<()> {
    std::fs::write(path, to_csv(t))
}

/// Read a trace from canonical CSV.
pub fn read_trace(path: &Path) -> std::io::Result<Option<Trace>> {
    let s = std::fs::read_to_string(path)?;
    Ok(from_csv(&s))
}

// --- Comparison --------------------------------------------------------------

/// Per-tick error summary between two traces.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Diff {
    pub max_pos_err: f32,
    pub max_vel_err: f32,
    pub max_yaw_err: f32,
    pub first_divergence: Option<usize>,
}

fn dist3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn yaw_err(a: f32, b: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut d = (a - b) % tau;
    if d > std::f32::consts::PI {
        d -= tau;
    } else if d < -std::f32::consts::PI {
        d += tau;
    }
    d.abs()
}

/// Tick-by-tick tolerance comparison. `first_divergence` is the first index
/// whose position or velocity error exceeds its tolerance, or the overlap
/// length when the traces have different lengths.
pub fn compare(a: &Trace, b: &Trace, pos_tol: f32, vel_tol: f32) -> Diff {
    let overlap = a.samples.len().min(b.samples.len());
    let mut d = Diff::default();
    let mut diverged = None;
    for i in 0..overlap {
        let pe = dist3(&a.samples[i].pos, &b.samples[i].pos);
        let ve = dist3(&a.samples[i].vel, &b.samples[i].vel);
        let ye = yaw_err(a.samples[i].yaw, b.samples[i].yaw);
        d.max_pos_err = d.max_pos_err.max(pe);
        d.max_vel_err = d.max_vel_err.max(ve);
        d.max_yaw_err = d.max_yaw_err.max(ye);
        if diverged.is_none() && (pe > pos_tol || ve > vel_tol) {
            diverged = Some(i);
        }
    }
    if diverged.is_none() && a.samples.len() != b.samples.len() {
        diverged = Some(overlap);
    }
    d.first_divergence = diverged;
    d
}

// --- Reference controller ----------------------------------------------------

/// A deterministic canned-speed oscillator, used to make reproducible traces.
#[derive(Clone, Debug)]
pub struct SinController {
    pub base_speed: f32,
    pub amp: f32,
    pos: [f32; 3],
    vel: [f32; 3],
    yaw: f32,
    phase: f32,
    grounded: bool,
}

impl Default for SinController {
    fn default() -> Self {
        Self { base_speed: 3.0, amp: 0.25, pos: [0.0; 3], vel: [0.0; 3], yaw: 0.0, phase: 0.0, grounded: true }
    }
}

impl SinController {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_speed(base_speed: f32) -> Self {
        Self { base_speed, ..Self::default() }
    }
}

impl Controller for SinController {
    fn reset(&mut self, spawn: [f32; 3]) {
        self.pos = spawn;
        self.vel = [0.0; 3];
        self.yaw = 0.0;
        self.phase = 0.0;
        self.grounded = true;
    }

    fn step(&mut self, input: &InputFrame) -> StateSample {
        self.phase += input.dt;
        let osc = (self.phase * 3.0).sin();
        let drive = input.forward + self.amp * osc;
        let scale = if input.sprint { 1.6 } else { 1.0 } * if input.crouch { 0.5 } else { 1.0 };
        let spd = self.base_speed * scale;
        self.yaw += input.yaw_delta;

        let (s, c) = self.yaw.sin_cos();
        self.vel[0] = (s * drive + c * input.strafe) * spd;
        self.vel[2] = (c * drive - s * input.strafe) * spd;

        if input.jump && self.grounded {
            self.vel[1] = 5.0;
            self.grounded = false;
        } else if !self.grounded {
            self.vel[1] -= 9.81 * input.dt;
        } else {
            self.vel[1] = 0.0;
        }

        for k in 0..3 {
            self.pos[k] += self.vel[k] * input.dt;
        }
        if self.pos[1] <= 0.0 {
            self.pos[1] = 0.0;
            self.vel[1] = 0.0;
            self.grounded = true;
        }

        let speed = (self.vel[0] * self.vel[0] + self.vel[2] * self.vel[2]).sqrt();
        let gait = if speed < 0.1 {
            0
        } else if speed < 2.0 {
            1
        } else if speed < 4.0 {
            2
        } else {
            3
        };

        StateSample { pos: self.pos, vel: self.vel, yaw: self.yaw, speed, gait, grounded: self.grounded }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(n: usize) -> Vec<InputFrame> {
        (0..n)
            .map(|i| InputFrame {
                forward: 1.0,
                strafe: (i as f32 * 0.1).sin(),
                yaw_delta: 0.02,
                sprint: i % 5 == 0,
                crouch: i % 7 == 0,
                jump: i == 10,
                dt: 1.0 / 60.0,
            })
            .collect()
    }

    #[test]
    fn record_is_deterministic() {
        let ins = inputs(200);
        let a = record(&mut SinController::new(), [1.0, 0.0, 2.0], &ins);
        let b = record(&mut SinController::new(), [1.0, 0.0, 2.0], &ins);
        assert_eq!(a, b);
        assert_eq!(a.samples.len(), ins.len());
    }

    #[test]
    fn csv_round_trips_exactly() {
        let t = record(&mut SinController::new(), [3.0, 1.0, -4.0], &inputs(64));
        let back = from_csv(&to_csv(&t)).expect("parse");
        assert_eq!(back, t);
    }

    #[test]
    fn compare_identical_is_zero() {
        let t = record(&mut SinController::new(), [0.0; 3], &inputs(64));
        let d = compare(&t, &t, 0.001, 0.001);
        assert_eq!(d.max_pos_err, 0.0);
        assert_eq!(d.max_vel_err, 0.0);
        assert_eq!(d.max_yaw_err, 0.0);
        assert_eq!(d.first_divergence, None);
    }

    #[test]
    fn compare_perturbed_finds_divergence() {
        let a = record(&mut SinController::new(), [0.0; 3], &inputs(64));
        let mut b = a.clone();
        b.samples[7].pos[0] += 5.0;
        let d = compare(&a, &b, 0.01, 0.01);
        assert!(d.max_pos_err > 0.0);
        assert_eq!(d.first_divergence, Some(7));
    }

    #[test]
    fn compare_handles_length_mismatch() {
        let a = record(&mut SinController::new(), [0.0; 3], &inputs(64));
        let mut b = a.clone();
        b.samples.truncate(20);
        b.inputs.truncate(20);
        let d = compare(&a, &b, 0.001, 0.001);
        assert_eq!(d.first_divergence, Some(20));
        assert_eq!(d.max_pos_err, 0.0);
    }
}
