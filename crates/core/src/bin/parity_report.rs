//! `parity_report` — replay the recorded retail captures in `notes/telemetry`
//! through the native controllers and print per-scenario error metrics.
//!
//! Usage:
//! ```text
//! cargo run -p sod-core --bin parity_report [TELEMETRY_DIR]
//! ```
//!
//! Defaults to `notes/telemetry` next to the workspace
//! ([`sod_core::parity_replay::default_telemetry_dir`]). Exits non-zero if any
//! scored scenario fails, so it can gate CI. See `crates/core/src/parity_replay.rs`
//! for the algorithms and tolerances, and `notes/parity.md` for the write-up.

use sod_core::parity_replay as pr;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(pr::default_telemetry_dir);

    println!("== sod parity replay ==");
    println!("telemetry dir: {}", dir.display());
    println!(
        "tolerances: steady<= {:.2} m/s, ramp-dist<= {:.2} m, slide<= {:.2} m, \
         stick-rate<= {:.2} deg/s, mouse-frame-rms<= {:.2} deg",
        pr::STEADY_TOL,
        pr::RAMP_DIST_TOL,
        pr::SLIDE_TOL,
        pr::STICK_RATE_TOL_DEG,
        pr::MOUSE_FRAME_RMS_TOL_DEG,
    );

    let mut failed: Vec<String> = Vec::new();
    let mut passed = 0usize;
    let mut total = 0usize;

    // ---- movement ---------------------------------------------------------
    println!("\n-- movement (class3_move::update_normal) --");
    let mut cache: Vec<(String, Vec<pr::MovementSample>)> = Vec::new();
    for sc in pr::movement_scenarios() {
        let samples = match cache.iter().find(|(f, _)| f == sc.file) {
            Some((_, s)) => s.clone(),
            None => match pr::parse_movement(&dir.join(sc.file)) {
                Ok(s) => {
                    cache.push((sc.file.to_string(), s.clone()));
                    s
                }
                Err(e) => {
                    println!("  [SKIP] {}/{}: {e}", sc.file, sc.phase);
                    continue;
                }
            },
        };
        total += 1;
        match pr::replay_movement(&samples, &sc) {
            None => {
                println!("  [SKIP] {}/{}: phase absent or no input edge", sc.file, sc.phase);
                total -= 1;
            }
            Some(e) => {
                let ok = e.pass();
                if ok {
                    passed += 1;
                } else {
                    failed.push(format!(
                        "movement {}/{}: steady {:.2} vs {:.2} (e {:.2}), \
                         ramp-dist e {:.2}, ramp-rms {:.2}, slide {}",
                        sc.file,
                        sc.phase,
                        e.steady_sim,
                        e.steady_rec,
                        e.steady_err,
                        e.ramp_dist_err,
                        e.ramp_rms,
                        e.slide_err.map_or("-".to_string(), |s| format!("e {s:.2}")),
                    ));
                }
                println!(
                    "  [{}] {}/{}: steady sim {:.2} rec {:.2} (err {:.2}); \
                     ramp-dist err {:.2}; ramp rms {:.2} max {:.2}; slide {}",
                    if ok { "PASS" } else { "FAIL" },
                    sc.file,
                    sc.phase,
                    e.steady_sim,
                    e.steady_rec,
                    e.steady_err,
                    e.ramp_dist_err,
                    e.ramp_rms,
                    e.ramp_max,
                    e.slide_err.map_or("-".to_string(), |s| format!("err {s:.2}")),
                );
            }
        }
    }

    // ---- camera stick -----------------------------------------------------
    println!("\n-- camera stick rates (camera::update) --");
    let overrides = pr::rate_overrides();
    for file in ["cam_stick.csv", "cam_stick_fine.csv", "turnpad.csv", "turnjump.csv"] {
        let samples = match pr::parse_rate_samples(&dir.join(file)) {
            Ok(s) => s,
            Err(e) => {
                println!("  [SKIP] {file}: {e}");
                continue;
            }
        };
        let errs = pr::replay_rates(&samples, &overrides, file);
        let mut worst = 0.0f32;
        let mut worst_phase = String::new();
        for e in &errs {
            total += 1;
            if e.pass() {
                passed += 1;
            } else {
                failed.push(format!(
                    "stick {}/{}: pred yaw {:.2} pitch {:.2} vs meas {:.2}/{:.2} deg/s (err {:.2})",
                    file, e.phase, e.pred_yaw, e.pred_pitch, e.meas_yaw, e.meas_pitch, e.err
                ));
            }
            if e.err > worst {
                worst = e.err;
                worst_phase = e.phase.clone();
            }
        }
        println!("  {file}: {} phases, worst {:.2} deg/s ({})", errs.len(), worst, worst_phase);
    }

    // ---- camera mouse -----------------------------------------------------
    println!("\n-- camera mouse deltas (camera::update) --");
    for file in ["cam_mouse.csv", "cam_mouse_flat.csv"] {
        let samples = match pr::parse_camera(&dir.join(file)) {
            Ok(s) => s,
            Err(e) => {
                println!("  [SKIP] {file}: {e}");
                continue;
            }
        };
        let errs = pr::replay_mouse(&samples, file);
        let mut worst = 0.0f32;
        let mut worst_phase = String::new();
        let mut npass = 0usize;
        for e in &errs {
            total += 1;
            if e.pass() {
                passed += 1;
                npass += 1;
            } else {
                failed.push(format!(
                    "mouse {}/{}: frame rms {:.2} max {:.2} deg, total pred {:.1} vs rec {:.1} (err {:.1})",
                    file, e.phase, e.frame_rms, e.frame_max, e.total_pred, e.total_rec, e.total_err
                ));
            }
            if e.frame_max > worst {
                worst = e.frame_max;
                worst_phase = e.phase.clone();
            }
        }
        println!(
            "  {file}: {} phases, {} pass, worst frame {:.2} deg ({})",
            errs.len(),
            npass,
            worst,
            worst_phase
        );
    }

    // ---- failures ---------------------------------------------------------
    println!("\n-- failures --");
    if failed.is_empty() {
        println!("  (none)");
    } else {
        for f in &failed {
            println!("  {f}");
        }
    }

    println!("\nsummary: {passed}/{total} scenarios pass, {} fail", failed.len());
    if failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
