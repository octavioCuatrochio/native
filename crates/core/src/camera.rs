//! 1:1 player camera (third/first person, aim, nav, `cl_cam_*` acceleration).
//!
//! CryEngine coordinates are **Z-up right-handed** (matching `physics.rs` and
//! `terrain.rs`); the app converts to Bevy's Y-up with a base X-90° rotation
//! (`crates/app/src/main.rs::follow_camera`). Yaw turns about world Z, pitch
//! about the camera right axis.
//!
//! Sources for the constants:
//! * [`crate::tuning::CameraTuning`] / `default_camera()` — the runtime cvar
//!   values (`cl_sensitivity 45` from
//!   `attributes.xml OptionCfg.cl_sensitivity`).
//! * `notes/ghidra/camera.md` §2–§4 — UL `Camera::Update` (`0x01488390`),
//!   behavior type `spec+0x78` (0 = first-person, 1 = third-person,
//!   2 = cinematic, 3 = vehicle), the `cam_*` cvar block (`0x01486d60`) and the
//!   nav/aim rotation controller (`FUN_016858c0`, cvars `0x01568d20`).
//! * `notes/ghidra/movement_input.md` §5 — mouse deltas are added after the
//!   per-frame delta clamp; pitch limit `DAT_0189f050 = 1.3962634` (80°).
//! * `follow_camera` — boom back `5.0 m`, up `2.6 m` (Aim pulls it in/down).
//!
//! Recovered binary defaults used here (the *exe* registration defaults, not the
//! profile override): `cl_cam_rotation_speed 0.2`,
//! `cl_cam_rotation_acceleration_time_yaw 0.5`, `..._pitch 0.4`,
//! `cl_cam_mouseYawScale 0.5`, `cl_cam_mousePitchScale 0.75`,
//! `cam_collisionPadding 0.15`, `cam_collisionZoomOffsetL/R 0.25`,
//! `cam_collisionZoomInTime 0.2–0.6 s`, `...OutTime 1.0–2.0 s`.
//! `tuning::default_camera()` still carries the older profile values
//! (`0.04 / 0.3 / 0.6`); [`update`] substitutes the recovered defaults when it
//! sees those exact legacy values so the port matches the exe, while any other
//! explicit override (including `0.0`, i.e. instant acceleration) is honoured.

use crate::physics;
use crate::tuning;

/// 80° pitch clamp, `DAT_0189f050 = 1.3962634` (`movement.md` §3).
pub const PITCH_LIMIT: f32 = 1.3962634;

/// Shipped `cl_sensitivity` (`attributes.xml OptionCfg.cl_sensitivity`).
pub const BASE_SENSITIVITY: f32 = 45.0;

/// `cl_cam_rotation_speed` — the **live shipped value** (client obj
/// `[*(0x01e1e634)]+0x40`, read from the running game; `meleeprototype.cfg`).
/// The exe registration default is `0.2`, but the profile/config overrides it
/// to `0.04`; the port uses the effective value (`movement_telemetry.md`).
pub const CAM_ROTATION_SPEED: f32 = 0.04;
/// `cl_cam_rotation_acceleration_time_yaw` (client `+0x48`, live `0.3`).
pub const ACCEL_TIME_YAW: f32 = 0.3;
/// `cl_cam_rotation_acceleration_time_pitch` (client `+0x4c`, live `0.6`).
pub const ACCEL_TIME_PITCH: f32 = 0.6;
/// `cl_cam_mouseYawScale` (client `+0x50`, live `0.5`).
pub const MOUSE_YAW_SCALE: f32 = 0.5;
/// `cl_cam_mousePitchScale` (client `+0x54`, live `0.75`).
pub const MOUSE_PITCH_SCALE: f32 = 0.75;

/// Nav/aim rotation-speed acceleration model (`FUN_016858c0`, `01568d20`).
///
/// Live values read at runtime (`movement_telemetry.md`); the exe registration
/// defaults (`2.0`/`2.0`) are overridden by the profile. At/above the threshold
/// the rotation speed is `accel_max`, so the full-stick nav yaw rate is
/// `input · π · 0.16`; the game's own `DAT_01ee0fd8` measured `0.473 rad/s`
/// (27.1 °/s) at full stick, confirming the model.
pub const NAV_ACCEL_MAX: f32 = 0.16;
/// `cl_cam_nav_rotation_acceleration_threshold` (live `0.9`).
pub const NAV_ACCEL_THRESHOLD: f32 = 0.9;
/// `cl_cam_aim_rotation_speed_base` (live `0.02`).
pub const AIM_SPEED_BASE: f32 = 0.02;
/// `cl_cam_aim_rotation_speed_accelerationmax` (live `0.07`).
pub const AIM_ACCEL_MAX: f32 = 0.07;
/// `cl_cam_aim_rotation_acceleration_threshold` (live `0.9`).
pub const AIM_ACCEL_THRESHOLD: f32 = 0.9;
/// `cl_cam_aim_rotation_acceleration_time_yaw` (live `0.4`).
pub const AIM_ACCEL_TIME_YAW: f32 = 0.4;
/// `cl_cam_aim_rotation_acceleration_time_pitch` (live `0.6`).
pub const AIM_ACCEL_TIME_PITCH: f32 = 0.6;


/// Boom defaults mirror `follow_camera` (`-fwd * 5.0 + up * 2.6`).
const DEFAULT_BOOM_DISTANCE: f32 = 5.0;
const DEFAULT_BOOM_HEIGHT: f32 = 2.6;

/// Aim mode pulls the boom closer and lower (over-the-shoulder).
const AIM_DISTANCE_SCALE: f32 = 0.5;
const AIM_HEIGHT_SCALE: f32 = 0.5;

/// `cam_collisionPadding` (`DAT_01e645cc`) kept between the camera and a wall.
pub const BOOM_PADDING: f32 = 0.15;
/// `cam_collisionZoomOffsetL/R` (`DAT_01e645a0/a4`) side-probe offset (m).
pub const BOOM_OFFSET: f32 = 0.25;
/// `cam_collisionZoomInTimeMin/Max` (`DAT_01e6459c/98`), seconds.
pub const BOOM_ZOOM_IN_MIN: f32 = 0.2;
pub const BOOM_ZOOM_IN_MAX: f32 = 0.6;
/// `cam_collisionZoomOutTimeMin/Max` (`DAT_01e645ac/a8`), seconds.
pub const BOOM_ZOOM_OUT_MIN: f32 = 1.0;
pub const BOOM_ZOOM_OUT_MAX: f32 = 2.0;
/// `cam_collisionZoomOutDelay` (`DAT_01e64590`), seconds.
pub const BOOM_ZOOM_OUT_DELAY: f32 = 0.1;
/// `cam_collisionZoomSlowingDist` (`DAT_01e64594`), metres.
pub const BOOM_ZOOM_SLOWING_DIST: f32 = 1.0;

/// Focal height above the player origin (metres): chest for the third-person
/// focal target (`Camera::SetFocalTarget`, `0148c140`), eye for first-person.
pub const FOCAL_CHEST: f32 = 1.4;
pub const FOCAL_EYE: f32 = 1.7;

/// Camera behaviour mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CameraMode {
    /// Free third-person navigation boom (`spec+0x78 == 1`).
    Nav,
    /// Aiming: tighter, lower boom.
    Aim,
    /// No boom; the camera sits on the target (`spec+0x78 == 0`, at the eye).
    FirstPerson,
}

/// Persistent look/boom state for the player camera.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CameraState {
    /// Yaw about world Z (radians).
    pub yaw: f32,
    /// Pitch (radians, positive = up).
    pub pitch: f32,
    /// Current yaw angular velocity (rad/s).
    pub yaw_vel: f32,
    /// Current pitch angular velocity (rad/s).
    pub pitch_vel: f32,
    /// Current mode.
    pub mode: CameraMode,
    /// Third-person boom length (m).
    pub boom_distance: f32,
    /// Third-person boom height above the target (m).
    pub boom_height: f32,
    /// Look-at target (world Z-up).
    pub target: [f32; 3],
    /// Computed camera position (world Z-up).
    pub pos: [f32; 3],
    /// Collision zoom factor: `1.0` = full boom, `0.0` = fully retracted
    /// (`cam_collisionZoom*`, `FUN_015ed600`).
    pub boom_zoom: f32,
}

impl CameraState {
    /// Fresh camera at the origin, facing +X, with the shipped boom.
    pub fn new() -> CameraState {
        CameraState {
            yaw: 0.0,
            pitch: 0.0,
            yaw_vel: 0.0,
            pitch_vel: 0.0,
            mode: CameraMode::Nav,
            boom_distance: DEFAULT_BOOM_DISTANCE,
            boom_height: DEFAULT_BOOM_HEIGHT,
            target: [0.0; 3],
            pos: [0.0; 3],
            boom_zoom: 1.0,
        }
    }
}

impl Default for CameraState {
    fn default() -> Self {
        CameraState::new()
    }
}

/// One frame of mouse/stick/aim input.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct CameraInput {
    /// Mouse X delta this frame (pixels).
    pub mouse_dx: f32,
    /// Mouse Y delta this frame (pixels).
    pub mouse_dy: f32,
    /// Right-stick deflection `[x, y]`, each `-1..=1`, same sign convention as
    /// `mouse_dx`/`mouse_dy`. Mapped through [`stick_response`].
    pub stick: [f32; 2],
    /// True while the aim button is held.
    pub aim: bool,
    /// Frame time (s).
    pub dt: f32,
}

/// Force a camera mode (e.g. [`CameraMode::FirstPerson`]); [`update`] preserves
/// it until changed.
pub fn set_mode(state: &mut CameraState, mode: CameraMode) {
    state.mode = mode;
}

/// Advance the camera by one frame.
///
/// The mouse delta becomes a *desired* angular velocity (scaled by
/// `cl_cam_rotation_speed`, `cl_cam_mouseYawScale`/`mousePitchScale` and the
/// normalised `sensitivity`); the current angular velocity accelerates toward it
/// over the `cl_cam_rotation_acceleration_time_*` (live `0.3`/`0.6`; aim uses
/// `cl_cam_aim_rotation_acceleration_time_*`, live `0.4`/`0.6`). Pitch is clamped
/// to ±[`PITCH_LIMIT`]. `aim` selects [`CameraMode::Aim`] (unless the state is
/// already first-person), which pulls the boom closer and lower. Finally the boom
/// position behind `target` is written to [`CameraState::pos`].
pub fn update(state: &mut CameraState, input: &CameraInput, t: &tuning::CameraTuning, target: [f32; 3]) {
    let mode = if state.mode == CameraMode::FirstPerson {
        CameraMode::FirstPerson
    } else if input.aim {
        CameraMode::Aim
    } else {
        CameraMode::Nav
    };
    update_mode(state, input, t, target, mode);
}

/// Advance the camera with an explicit [`CameraMode`] (see [`update`]).
pub fn update_mode(
    state: &mut CameraState,
    input: &CameraInput,
    t: &tuning::CameraTuning,
    target: [f32; 3],
    mode: CameraMode,
) {
    let target = sane3(target);
    state.target = target;
    state.mode = mode;

    let dt = if input.dt.is_finite() && input.dt > 0.0 {
        input.dt
    } else {
        0.0
    };
    let _ = t;

    // Device input -> camera input value in -1..=1 (the player-input object's
    // `+0xd8/+0xdc`, read by the camera through `0x1487870`).
    let mouse = [sane(input.mouse_dx), sane(input.mouse_dy)];
    let (v, device_scale) = if mouse != [0.0, 0.0] {
        // Mouse (rotate action handler `0x1588ee0`): raw counts this frame
        // times `0.01 + 0.025*s`, clamped to +-1; device scale `1 + 6*s`
        // (`0x150ea1c`). s = 0.5 live.
        let k = mouse_count_scale(MOUSE_SENSITIVITY_NORM);
        (
            [(mouse[0] * k).clamp(-1.0, 1.0), (mouse[1] * k).clamp(-1.0, 1.0)],
            1.0 + MOUSE_DEVICE_SCALE_PER_SENS * MOUSE_SENSITIVITY_NORM,
        )
    } else {
        // Right stick: shaped deflection (fitted curve, see `STICK_RESPONSE`).
        let [sx, sy] = [sane(input.stick[0]).clamp(-1.0, 1.0), sane(input.stick[1]).clamp(-1.0, 1.0)];
        ([sx.signum() * stick_response(sx.abs()), sy.signum() * stick_response(sy.abs())], STICK_DEVICE_SCALE)
    };

    // Camera rotation (third-person update `0x15ea59c`): only when either
    // axis exceeds the dead zone, then `speed * value * dt * device_scale`,
    // applied immediately (no ramp or smoothing).
    if v[0].abs() > CAM_INPUT_DEADZONE || v[1].abs() > CAM_INPUT_DEADZONE {
        state.yaw_vel = CAM_YAW_SPEED * v[0] * device_scale;
        state.pitch_vel = CAM_PITCH_SPEED * v[1] * device_scale;
    } else {
        state.yaw_vel = 0.0;
        state.pitch_vel = 0.0;
    }

    state.yaw = sane(state.yaw) + state.yaw_vel * dt;
    state.pitch = sane(state.pitch) + state.pitch_vel * dt;
    if state.pitch > PITCH_LIMIT {
        state.pitch = PITCH_LIMIT;
        state.pitch_vel = 0.0;
    } else if state.pitch < -PITCH_LIMIT {
        state.pitch = -PITCH_LIMIT;
        state.pitch_vel = 0.0;
    }

    // Boom sits behind the yaw heading *and* is tilted by pitch, then raised by
    // the boom height. Aim shortens/lowers it and FirstPerson collapses it to
    // zero (the camera sits on the focal target / eye).
    state.pos = orbit_position(state, target);
}

/// Boom `(distance, height)` for the current mode.
///
/// Aim halves both (over-the-shoulder, `AIM_DISTANCE_SCALE`/`AIM_HEIGHT_SCALE`)
/// and FirstPerson collapses the boom to the focal target.
fn boom_params(state: &CameraState) -> (f32, f32) {
    match state.mode {
        CameraMode::Aim => (
            state.boom_distance * AIM_DISTANCE_SCALE,
            state.boom_height * AIM_HEIGHT_SCALE,
        ),
        CameraMode::FirstPerson => (0.0, 0.0),
        CameraMode::Nav => (state.boom_distance, state.boom_height),
    }
}

/// Third-person boom position in **cry Z-up** space, orbiting `target`.
///
/// `target` is the focal target — the player's chest/head in cry coordinates,
/// as produced by [`focal_target`] from the player's feet. The camera is placed
/// opposite the yaw/pitch look direction at `boom_distance`, then raised by
/// `boom_height` along world +Z:
///
/// ```text
/// look(yaw, pitch) = [ cos(pitch)cos(yaw),  cos(pitch)sin(yaw), sin(pitch) ]
/// pos = target - look(yaw, pitch) * distance + [0, 0, height]
/// ```
///
/// Because pitch tilts the *boom* (not just the aim), looking up (`pitch > 0`)
/// drops the camera below the model and looking down (`pitch < 0`) lifts it
/// above — a true orbit around the model. With `yaw = 0` the camera rests
/// directly behind the target along -X and `height` above it. Yaw = 0, pitch =
/// 0, Nav reproduces the shipped `follow_camera` boom (`-X * 5.0 + Z * 2.6`).
///
/// Total offset is `sqrt(distance^2 + height^2)` at `pitch = 0`. Non-finite
/// state/target components are coerced to 0 so nothing NaN escapes.
pub fn orbit_position(state: &CameraState, target: [f32; 3]) -> [f32; 3] {
    let target = sane3(target);
    let (dist, height) = boom_params(state);
    let dist = sane(dist).max(0.0);
    let height = sane(height);
    let yaw = sane(state.yaw);
    let pitch = sane(state.pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    let cp = pitch.cos();
    let sp = pitch.sin();
    // -look(yaw, pitch): behind the target, and below it when looking up.
    let bx = -cp * yaw.cos();
    let by = -cp * yaw.sin();
    let bz = -sp;
    [
        target[0] + bx * dist,
        target[1] + by * dist,
        target[2] + bz * dist + height,
    ]
}

/// Advance the camera and apply [`boom_collision`] to the resulting boom.
///
/// Returns the collided camera position (also written to [`CameraState::pos`])
/// and advances [`CameraState::boom_zoom`] toward retracted/extended.
pub fn update_collide(
    state: &mut CameraState,
    input: &CameraInput,
    t: &tuning::CameraTuning,
    target: [f32; 3],
    colliders: &[physics::Aabb],
) -> [f32; 3] {
    update(state, input, t, target);
    if state.mode == CameraMode::FirstPerson {
        state.boom_zoom = 1.0;
        return state.pos;
    }

    let uncollided = state.pos;
    let collided = boom_collision(uncollided, state.target, colliders);
    let pulled = dist3(collided, state.target) + 1e-4 < dist3(uncollided, state.target);
    let wanted = if pulled { 0.0 } else { 1.0 };
    let dt = if input.dt.is_finite() && input.dt > 0.0 {
        input.dt
    } else {
        0.0
    };
    state.boom_zoom = boom_zoom_step(state.boom_zoom, wanted, dt, pulled);
    state.pos = collided;
    state.pos
}

/// Pull the camera in when the boom segment `target -> pos` crosses a collider.
///
/// Uses the recovered `cam_collisionPadding` ([`BOOM_PADDING`]) and probes the
/// focal target plus left/right [`BOOM_OFFSET`] (`cam_collisionZoomOffsetL/R`),
/// taking the nearest hit. The camera is placed `padding` metres in front of it.
/// Degenerate/NaN input is returned unchanged (or zeroed).
pub fn boom_collision(pos: [f32; 3], target: [f32; 3], colliders: &[physics::Aabb]) -> [f32; 3] {
    boom_collision_with(pos, target, colliders, BOOM_PADDING, BOOM_OFFSET)
}

/// [`boom_collision`] with explicit padding / side-probe offset (m).
pub fn boom_collision_with(
    pos: [f32; 3],
    target: [f32; 3],
    colliders: &[physics::Aabb],
    padding: f32,
    offset: f32,
) -> [f32; 3] {
    let p = sane3(pos);
    let t = sane3(target);
    let d = [p[0] - t[0], p[1] - t[1], p[2] - t[2]];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if !len.is_finite() || len < 1e-6 {
        return p;
    }
    let pad = if padding.is_finite() && padding >= 0.0 {
        padding
    } else {
        BOOM_PADDING
    };
    let off = if offset.is_finite() && offset >= 0.0 {
        offset
    } else {
        0.0
    };

    // Probe the focal target plus a horizontal left/right offset; all probes
    // share the boom direction.
    let mut origins = [[t[0], t[1], t[2]]; 3];
    let mut nprobe = 1usize;
    let horiz = (d[0] * d[0] + d[1] * d[1]).sqrt();
    if off > 1e-6 && horiz.is_finite() && horiz > 1e-6 {
        let rx = -d[1] / horiz * off;
        let ry = d[0] / horiz * off;
        origins[1] = [t[0] + rx, t[1] + ry, t[2]];
        origins[2] = [t[0] - rx, t[1] - ry, t[2]];
        nprobe = 3;
    }

    let mut best = f32::INFINITY;
    for o in origins.iter().take(nprobe) {
        for b in colliders {
            if !b.min.iter().chain(b.max.iter()).all(|v| v.is_finite()) {
                continue;
            }
            if let Some(hit) = segment_hit(*o, d, b) {
                if hit > 1e-3 && hit < best {
                    best = hit;
                }
            }
        }
    }

    if best.is_finite() {
        let tnew = ((best * len - pad) / len).clamp(0.0, 1.0);
        [
            t[0] + d[0] * tnew,
            t[1] + d[1] * tnew,
            t[2] + d[2] * tnew,
        ]
    } else {
        p
    }
}

/// Third-person focal target (`Camera::SetFocalTarget`, `0148c140`).
///
/// **Input**: the player's *feet/origin* position in cry Z-up coordinates
/// (`mstate.pos`, exactly what `follow_camera` passes). **Output**: the model's
/// chest/head anchor in the same cry Z-up space, [`FOCAL_CHEST`] metres above
/// the feet. Feed the result to [`update`] as `target`; the boom then orbits
/// this point (`player.md` §7–§8: the third-person focal target is the head
/// bone, `viewOffset.z = 1.625`). Non-finite input is coerced to 0.
pub fn focal_target(player_pos: [f32; 3]) -> [f32; 3] {
    let p = sane3(player_pos);
    [p[0], p[1], p[2] + FOCAL_CHEST]
}

/// First-person focal target: eye height ([`FOCAL_EYE`]).
///
/// Same feet-in / cry-out contract as [`focal_target`].
pub fn focal_target_head(player_pos: [f32; 3]) -> [f32; 3] {
    let p = sane3(player_pos);
    [p[0], p[1], p[2] + FOCAL_EYE]
}

/// The `(min, max)` zoom time window for retracting (`obstructed`) or extending
/// the boom (`cam_collisionZoomInTimeMin/Max` vs `...OutTimeMin/Max`).
pub fn boom_zoom_time(obstructed: bool) -> (f32, f32) {
    if obstructed {
        (BOOM_ZOOM_IN_MIN, BOOM_ZOOM_IN_MAX)
    } else {
        (BOOM_ZOOM_OUT_MIN, BOOM_ZOOM_OUT_MAX)
    }
}

/// Advance the boom zoom factor `current` toward `target` (`1.0` full boom,
/// `0.0` fully retracted) over one frame.
///
/// Retraction uses the faster in-time window (`0.2..0.6 s`) and extension the
/// slower out-time window (`1.0..2.0 s`); larger corrections take the longer end
/// of the window. Degenerate input is returned unchanged (or clamped).
pub fn boom_zoom_step(current: f32, target: f32, dt: f32, obstructed: bool) -> f32 {
    let c = sane(current).clamp(0.0, 1.0);
    let tg = sane(target).clamp(0.0, 1.0);
    if !dt.is_finite() || dt <= 0.0 {
        return c;
    }
    let (tmin, tmax) = boom_zoom_time(obstructed);
    let dist = (tg - c).abs();
    let time = tmin + (tmax - tmin) * dist;
    let alpha = if !time.is_finite() || time <= 1e-6 {
        1.0
    } else {
        (dt / time).clamp(0.0, 1.0)
    };
    c + (tg - c) * alpha
}

/// Unit camera forward direction — the direction the camera looks, i.e. from
/// [`CameraState::pos`] toward [`CameraState::target`].
///
/// After [`update`]/[`update_mode`]/[`orbit_position`] this is exactly
/// `normalize(target - pos)`, so the camera always points at the model and
/// [`basis`] stays consistent with what is rendered. When the camera sits on
/// the target (first-person, or a freshly-constructed state that has not been
/// positioned yet) the offset is degenerate and the raw yaw/pitch forward
/// vector is used instead so the basis stays orthonormal.
pub fn view_dir(state: &CameraState) -> [f32; 3] {
    let d = [
        state.target[0] - state.pos[0],
        state.target[1] - state.pos[1],
        state.target[2] - state.pos[2],
    ];
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if l.is_finite() && l > 1e-6 {
        [d[0] / l, d[1] / l, d[2] / l]
    } else {
        yaw_pitch_dir(state)
    }
}

/// Raw yaw/pitch forward direction (unit), independent of position. This is the
/// direction `orbit_position` places the camera *against* (`pos` uses
/// `-yaw_pitch_dir`), and the fallback for [`view_dir`] when `pos == target`.
fn yaw_pitch_dir(state: &CameraState) -> [f32; 3] {
    let yaw = sane(state.yaw);
    let pitch = sane(state.pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    let cp = pitch.cos();
    [cp * yaw.cos(), cp * yaw.sin(), pitch.sin()]
}

/// Orthonormal `(forward, right, up)` camera basis, for camera-relative movement.
pub fn basis(state: &CameraState) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let f = view_dir(state);
    // right = worldUp × forward, flattened to the ground plane.
    let mut r = [-f[1], f[0], 0.0];
    let rl = (r[0] * r[0] + r[1] * r[1]).sqrt();
    if rl.is_finite() && rl > 1e-6 {
        r = [r[0] / rl, r[1] / rl, 0.0];
    } else {
        r = [1.0, 0.0, 0.0];
    }
    let u = [
        f[1] * r[2] - f[2] * r[1],
        f[2] * r[0] - f[0] * r[2],
        f[0] * r[1] - f[1] * r[0],
    ];
    (f, r, u)
}

// ---------------------------------------------------------------------------
// nav / aim input model (`FUN_016858c0`, cvar registration `0x01568d20`)
// ---------------------------------------------------------------------------

/// Which recovered rotation cvar set the input controller uses
/// (`FUN_016858c0`, `camera.md` §3.2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CameraSet {
    /// Free-look navigation set (`cl_cam_nav_*`).
    Nav,
    /// Aiming set (`cl_cam_aim_*`); tighter threshold and slower ramp.
    Aim,
}

/// Camera yaw speed per unit input (rad/s), camera spec `+0x20` read live
/// at the rotation site `0x15ea59c`: pi/2.
pub const CAM_YAW_SPEED: f32 = std::f32::consts::FRAC_PI_2;
/// Camera pitch speed per unit input (rad/s), camera spec `+0x1c`: 80 deg.
pub const CAM_PITCH_SPEED: f32 = 1.396_263_4;
/// Per-axis input dead zone at the rotation site (`0x171e200`).
pub const CAM_INPUT_DEADZONE: f32 = 0.05;
/// Mouse counts -> input value: `MOUSE_COUNT_BASE + MOUSE_COUNT_PER_SENS * s`
/// (`0x171e1f8`, `0x17876fc`, handler `0x1588ee0`).
pub const MOUSE_COUNT_BASE: f32 = 0.01;
pub const MOUSE_COUNT_PER_SENS: f32 = 0.025;
/// Normalised mouse sensitivity `s` (camera spec `+0x50`, 0.5 live with the
/// default options). The mapping from the options menu is not recovered.
pub const MOUSE_SENSITIVITY_NORM: f32 = 0.5;
/// Mouse device scale `1 + MOUSE_DEVICE_SCALE_PER_SENS * s` (`0x172b108`).
pub const MOUSE_DEVICE_SCALE_PER_SENS: f32 = 6.0;
/// Stick device scale (option value read live as 1.0).
pub const STICK_DEVICE_SCALE: f32 = 1.0;

/// Input value per mouse count for a normalised sensitivity.
pub fn mouse_count_scale(s: f32) -> f32 {
    MOUSE_COUNT_BASE + MOUSE_COUNT_PER_SENS * sane(s)
}

/// Measured right-stick response `(|deflection|, fraction of max rate)`.
///
/// Yaw (10 deflections, both directions) and pitch (3 deflections) share this
/// normalised shape within 1 %, so one curve scaled per axis reproduces both.
/// Everything at or below 0.65 is dead. Fitted from retail captures, not
/// decompiled: the code that applies it is not yet located.
pub const STICK_RESPONSE: [(f32, f32); 9] = [
    (0.65, 0.0),
    (0.70, 0.0758),
    (0.75, 0.1323),
    (0.80, 0.2211),
    (0.85, 0.3366),
    (0.90, 0.5065),
    (0.95, 0.7259),
    (1.00, 1.0),
    (1.00, 1.0),
];

/// Fraction of the maximum stick rotation rate for a deflection magnitude,
/// linearly interpolated through [`STICK_RESPONSE`].
pub fn stick_response(m: f32) -> f32 {
    let m = sane(m).abs().min(1.0);
    if m <= STICK_RESPONSE[0].0 {
        return 0.0;
    }
    for w in STICK_RESPONSE.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if m <= x1 {
            return if x1 > x0 { y0 + (y1 - y0) * (m - x0) / (x1 - x0) } else { y1 };
        }
    }
    1.0
}

/// Resolved rotation parameters for a [`CameraSet`] (`camera.md` §3.2).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CameraSetParams {
    /// Rotation speed used below the acceleration threshold.
    pub base: f32,
    /// Rotation speed used at or above the acceleration threshold.
    pub accel_max: f32,
    /// Input magnitude at which the controller switches from `base` to
    /// `accel_max`.
    pub threshold: f32,
    /// `cl_cam_*_rotation_acceleration_time_yaw` (s).
    pub accel_time_yaw: f32,
    /// `cl_cam_*_rotation_acceleration_time_pitch` (s).
    pub accel_time_pitch: f32,
    /// Per-axis linear camera deadzone (`..._deadzone_pc_linearLR/UD`).
    pub deadzone_linear: f32,
    /// Radial camera deadzone (`..._deadzone_pc_radius`).
    pub deadzone_radius: f32,
}

/// `cl_input_*_camera_deadzone_pc_linearLR/UD` (nav and aim both `0.1`).
pub const CAMERA_DEADZONE_LINEAR: f32 = 0.1;
/// `cl_input_*_camera_deadzone_pc_radius` (nav and aim both `0.24`).
pub const CAMERA_DEADZONE_RADIUS: f32 = 0.24;

/// Nav rotation speed below the acceleration threshold
/// (`cl_cam_rotation_speed`, recovered exe `0.2`).
pub const NAV_SPEED_BASE: f32 = CAM_ROTATION_SPEED;

/// Resolve the cvar set for a [`CameraSet`] (`camera.md` §3.2).
pub fn camera_set_params(set: CameraSet) -> CameraSetParams {
    match set {
        CameraSet::Nav => CameraSetParams {
            base: NAV_SPEED_BASE,
            accel_max: NAV_ACCEL_MAX,
            threshold: NAV_ACCEL_THRESHOLD,
            accel_time_yaw: ACCEL_TIME_YAW,
            accel_time_pitch: ACCEL_TIME_PITCH,
            deadzone_linear: CAMERA_DEADZONE_LINEAR,
            deadzone_radius: CAMERA_DEADZONE_RADIUS,
        },
        CameraSet::Aim => CameraSetParams {
            base: AIM_SPEED_BASE,
            accel_max: AIM_ACCEL_MAX,
            threshold: AIM_ACCEL_THRESHOLD,
            accel_time_yaw: AIM_ACCEL_TIME_YAW,
            accel_time_pitch: AIM_ACCEL_TIME_PITCH,
            deadzone_linear: CAMERA_DEADZONE_LINEAR,
            deadzone_radius: CAMERA_DEADZONE_RADIUS,
        },
    }
}

/// Rotation-speed coefficient for an input magnitude: `base` below the set
/// threshold, `accel_max` at or above it (`FUN_016858c0` step 3). Total on
/// non-finite input.
pub fn camera_set_speed(set: CameraSet, input: f32) -> f32 {
    let p = camera_set_params(set);
    let m = sane(input).abs();
    if m < sane(p.threshold) {
        sane(p.base)
    } else {
        sane(p.accel_max)
    }
}

/// Recovered smoothing curve `FUN_015e98b0`: `3x² - 2x³` (a clamped
/// smoothstep). Used by the acceleration integrator and the boom easing.
#[inline]
pub fn smoothstep(x: f32) -> f32 {
    let x = sane(x).clamp(0.0, 1.0);
    3.0 * x * x - 2.0 * x * x * x
}

/// Integrate a rotation speed toward `target` over `accel_time`, using the
/// recovered smoothing curve (`FUN_016853a0`). With acceleration disabled (or
/// degenerate time) the target is returned immediately.
pub fn integrate_rotation_speed(
    current: f32,
    target: f32,
    dt: f32,
    accel_time: f32,
    accel_enabled: bool,
) -> f32 {
    let c = sane(current);
    let tg = sane(target);
    if !accel_enabled {
        return tg;
    }
    if !dt.is_finite() || dt <= 0.0 {
        return c;
    }
    if !accel_time.is_finite() || accel_time <= 1e-6 {
        return tg;
    }
    let a = smoothstep(dt / accel_time);
    c + (tg - c) * a
}

/// Apply the radial camera deadzone (`cl_input_*_camera_deadzone_pc_*`).
/// Inputs inside `radius` are zeroed (falling back to the per-axis `linear`
/// deadzone); inputs outside are remapped so the edge of the deadzone becomes
/// zero and full deflection stays full. Total on non-finite input.
pub fn camera_deadzone(stick: [f32; 2], linear: f32, radius: f32) -> [f32; 2] {
    let x = sane(stick[0]);
    let y = sane(stick[1]);
    let lin = if linear.is_finite() && linear > 0.0 {
        linear
    } else {
        CAMERA_DEADZONE_LINEAR
    };
    let r = if radius.is_finite() && radius > 0.0 {
        radius
    } else {
        CAMERA_DEADZONE_RADIUS
    };
    let mag = (x * x + y * y).sqrt();
    if !mag.is_finite() || mag <= r {
        return [
            if x.abs() <= lin { 0.0 } else { x },
            if y.abs() <= lin { 0.0 } else { y },
        ];
    }
    let span = (1.0 - r).max(1e-6);
    let scale = ((mag - r) / span / mag).max(0.0);
    [x * scale, y * scale]
}

// ---------------------------------------------------------------------------
// third-person lag / smoothing (`FUN_015ea260`)
// ---------------------------------------------------------------------------

/// Default lag scale for the third-person camera (`spec+0x4b8`).
pub const DEFAULT_LAG_SCALE: f32 = 0.15;

/// Exponential-smoothing blend fraction for one frame. `lag_scale` is the time
/// constant in seconds (`spec+0x4b8`); `<= 0` snaps.
#[inline]
pub fn lag_alpha(dt: f32, lag_scale: f32) -> f32 {
    if !dt.is_finite() || dt <= 0.0 {
        return 1.0;
    }
    if !lag_scale.is_finite() || lag_scale <= 1e-6 {
        return 1.0;
    }
    (1.0 - (-dt / lag_scale).exp()).clamp(0.0, 1.0)
}

/// Smooth a scalar toward `target` (`FUN_015ea260` orientation lag).
#[inline]
pub fn lag_smooth_scalar(current: f32, target: f32, lag_scale: f32, dt: f32) -> f32 {
    let a = lag_alpha(dt, lag_scale);
    let c = sane(current);
    let t = sane(target);
    c + (t - c) * a
}

/// Smooth a 3-vector toward `target` (`FUN_015ea260` position lag).
pub fn lag_smooth3(current: [f32; 3], target: [f32; 3], lag_scale: f32, dt: f32) -> [f32; 3] {
    let a = lag_alpha(dt, lag_scale);
    let c = sane3(current);
    let t = sane3(target);
    [
        c[0] + (t[0] - c[0]) * a,
        c[1] + (t[1] - c[1]) * a,
        c[2] + (t[2] - c[2]) * a,
    ]
}

/// Persistent lagged camera pose. The first [`CameraLag::step`] seeds the state
/// so the camera never snaps from the origin.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CameraLag {
    /// Smoothed camera position (cry Z-up).
    pub pos: [f32; 3],
    /// Smoothed yaw (rad).
    pub yaw: f32,
    /// Smoothed pitch (rad).
    pub pitch: f32,
    /// Set once the state has been seeded.
    pub seeded: bool,
}

impl CameraLag {
    /// Unseeded lag state at the origin.
    pub fn new() -> CameraLag {
        CameraLag {
            pos: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            seeded: false,
        }
    }

    /// Advance toward the target pose, returning the smoothed
    /// `(pos, yaw, pitch)`. The first call snaps to the target.
    pub fn step(
        &mut self,
        target_pos: [f32; 3],
        target_yaw: f32,
        target_pitch: f32,
        lag_scale: f32,
        dt: f32,
    ) -> ([f32; 3], f32, f32) {
        let tp = sane3(target_pos);
        let ty = sane(target_yaw);
        let tpi = sane(target_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        if !self.seeded {
            self.pos = tp;
            self.yaw = ty;
            self.pitch = tpi;
            self.seeded = true;
            return (self.pos, self.yaw, self.pitch);
        }
        let a = lag_alpha(dt, lag_scale);
        self.pos = [
            self.pos[0] + (tp[0] - self.pos[0]) * a,
            self.pos[1] + (tp[1] - self.pos[1]) * a,
            self.pos[2] + (tp[2] - self.pos[2]) * a,
        ];
        self.yaw = self.yaw + (ty - self.yaw) * a;
        self.pitch = self.pitch + (tpi - self.pitch) * a;
        (self.pos, self.yaw, self.pitch)
    }
}

impl Default for CameraLag {
    fn default() -> Self {
        CameraLag::new()
    }
}

/// Derive camera pitch from a forward basis vector, the `fpatan` in
/// `FUN_015ea260`: `atan2(z, |xy|)`, clamped to [`PITCH_LIMIT`].
pub fn pitch_from_basis(forward: [f32; 3]) -> f32 {
    let f = sane3(forward);
    let h = (f[0] * f[0] + f[1] * f[1]).sqrt();
    f[2].atan2(h).clamp(-PITCH_LIMIT, PITCH_LIMIT)
}

/// Derive camera yaw from a forward basis vector (`atan2(y, x)`).
pub fn yaw_from_basis(forward: [f32; 3]) -> f32 {
    let f = sane3(forward);
    f[1].atan2(f[0])
}

// ---------------------------------------------------------------------------
// 64-sample boom offset curve (`FUN_0148bc90`)
// ---------------------------------------------------------------------------

/// The camera's 64-sample boom offset curve (`FUN_0148bc90`): direction samples
/// at `camera+0x98` and weights at `camera+0x398`, with the focal point as the
/// `angle == 0` default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoomCurve {
    /// 64 direction samples (camera-local).
    pub dirs: [[f32; 3]; 64],
    /// 64 weights, one per direction sample.
    pub weights: [f32; 64],
    /// Default offset (the focal point) returned at `angle == 0` or when the
    /// weighted sum vanishes.
    pub default: [f32; 3],
}

impl BoomCurve {
    /// A curve whose default offset is `focal` and whose samples are empty.
    pub fn focal_point(focal: [f32; 3]) -> BoomCurve {
        BoomCurve {
            dirs: [[0.0; 3]; 64],
            weights: [0.0; 64],
            default: sane3(focal),
        }
    }

    /// Evaluate the curve at `angle`, mirroring `FUN_0148bc90`:
    ///
    /// ```text
    /// w_i = weights[i] / angle        (0 when angle == 0)
    /// if 1 - w_i < 0 { w_i = 0 }
    /// f_i = smoothstep(w_i)
    /// result = Σ dirs[i]·f_i / Σ f_i  (or default when Σ f_i == 0)
    /// ```
    pub fn evaluate(&self, angle: f32) -> [f32; 3] {
        let a = sane(angle);
        if a == 0.0 {
            return sane3(self.default);
        }
        let inv = if a >= 1e-8 { 1.0 / a } else { 0.0 };
        let mut total = 0.0f32;
        let mut sum = [0.0f32; 3];
        for i in 0..64 {
            let mut w = sane(self.weights[i]) * inv;
            if 1.0 - w < 0.0 {
                w = 0.0;
            }
            let f = smoothstep(w);
            let d = sane3(self.dirs[i]);
            total += f;
            sum[0] += d[0] * f;
            sum[1] += d[1] * f;
            sum[2] += d[2] * f;
        }
        if total > 0.0 {
            [sum[0] / total, sum[1] / total, sum[2] / total]
        } else {
            sane3(self.default)
        }
    }
}

impl Default for BoomCurve {
    fn default() -> Self {
        BoomCurve::focal_point([0.0; 3])
    }
}

/// Rotate `v` by the CryEngine quaternion `q = (x, y, z, w)`. Non-finite or
/// degenerate quaternions fall back to identity.
pub fn rotate_quat(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let q = sane4(q);
    let v = sane3(v);
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (x, y, z, w) = if n.is_finite() && n > 1e-8 {
        (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
    } else {
        (0.0, 0.0, 0.0, 1.0)
    };
    let tx = 2.0 * (y * v[2] - z * v[1]);
    let ty = 2.0 * (z * v[0] - x * v[2]);
    let tz = 2.0 * (x * v[1] - y * v[0]);
    [
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    ]
}

/// Boom offset: the [`BoomCurve`] evaluated at `angle`, rotated by the camera
/// basis quaternion (`spec+0x54..0x60`).
pub fn boom_offset(curve: &BoomCurve, angle: f32, basis_quat: [f32; 4]) -> [f32; 3] {
    rotate_quat(basis_quat, curve.evaluate(angle))
}

// ---------------------------------------------------------------------------
// secondary focal position + 64-entry focal history (`0148c140`)
// ---------------------------------------------------------------------------

/// `cam_secondaryFocalPosWeight` (`DAT_01e645d0`) — blend of the secondary
/// focal position.
pub const SECONDARY_FOCAL_WEIGHT: f32 = 0.4;
/// Focal-position history depth (`DAT_01e646c8`, stride `0xc`).
pub const FOCAL_HISTORY_LEN: usize = 64;

/// Linear blend `a -> b` by `weight` (clamped to `[0, 1]`). Total.
pub fn blend_weighted(a: [f32; 3], b: [f32; 3], weight: f32) -> [f32; 3] {
    let w = sane(weight).clamp(0.0, 1.0);
    let a = sane3(a);
    let b = sane3(b);
    [
        a[0] + (b[0] - a[0]) * w,
        a[1] + (b[1] - a[1]) * w,
        a[2] + (b[2] - a[2]) * w,
    ]
}

/// Blend the secondary focal position into the primary at
/// [`SECONDARY_FOCAL_WEIGHT`] (`0.4`), gated by `cam_useSecondaryFocalPos`.
pub fn blend_secondary_focal(
    primary: [f32; 3],
    secondary: [f32; 3],
    use_secondary: bool,
) -> [f32; 3] {
    if use_secondary {
        blend_weighted(primary, secondary, SECONDARY_FOCAL_WEIGHT)
    } else {
        sane3(primary)
    }
}

/// The 64-entry focal-position ring plus the focal velocity derived from the
/// last two entries (`0148c140`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocalHistory {
    /// Ring buffer of the last [`FOCAL_HISTORY_LEN`] focal positions.
    pub ring: [[f32; 3]; 64],
    /// Next write slot (`DAT_01e64ac8`, mod 64).
    pub index: usize,
    /// Number of samples written so far (saturates at 64).
    pub count: usize,
    /// Focal velocity from the last two samples.
    pub velocity: [f32; 3],
}

impl FocalHistory {
    /// Empty history.
    pub fn new() -> FocalHistory {
        FocalHistory {
            ring: [[0.0; 3]; 64],
            index: 0,
            count: 0,
            velocity: [0.0; 3],
        }
    }

    /// Push a focal position, updating [`FocalHistory::velocity`]. Returns the
    /// velocity.
    pub fn push(&mut self, p: [f32; 3]) -> [f32; 3] {
        let p = sane3(p);
        let prev = if self.count > 0 {
            self.ring[(self.index + FOCAL_HISTORY_LEN - 1) % FOCAL_HISTORY_LEN]
        } else {
            p
        };
        self.ring[self.index] = p;
        self.index = (self.index + 1) % FOCAL_HISTORY_LEN;
        if self.count < FOCAL_HISTORY_LEN {
            self.count += 1;
        }
        self.velocity = [p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]];
        self.velocity
    }

    /// Most recently pushed position (or the origin when empty).
    pub fn last(&self) -> [f32; 3] {
        if self.count == 0 {
            [0.0; 3]
        } else {
            self.ring[(self.index + FOCAL_HISTORY_LEN - 1) % FOCAL_HISTORY_LEN]
        }
    }
}

impl Default for FocalHistory {
    fn default() -> Self {
        FocalHistory::new()
    }
}

// ---------------------------------------------------------------------------
// speed-scaled boom zoom (`FUN_015ed600`)
// ---------------------------------------------------------------------------

/// `cam_collisionZoomScalePitchSpeedMin` / `...YawSpeedMin` (`DAT_01e645b4/bc`).
pub const ZOOM_SPEED_SCALE_MIN: f32 = 5.0;
/// `cam_collisionZoomScalePitchSpeedMax` / `...YawSpeedMax` (`DAT_01e645b0/b8`).
pub const ZOOM_SPEED_SCALE_MAX: f32 = 20.0;
/// Angular speed (rad/s) mapped to [`ZOOM_SPEED_SCALE_MAX`].
pub const ZOOM_SPEED_REF: f32 = 1.0;

/// Map camera angular speed to the recovered retract/extend scale in
/// `[5, 20]` (`cam_collisionZoomScale*`): fast turns pull the camera in harder.
pub fn zoom_speed_scale(angular_speed: f32) -> f32 {
    let s = sane(angular_speed).abs();
    let t = (s / ZOOM_SPEED_REF).clamp(0.0, 1.0);
    ZOOM_SPEED_SCALE_MIN + (ZOOM_SPEED_SCALE_MAX - ZOOM_SPEED_SCALE_MIN) * t
}

/// [`boom_zoom_step`] with the recovered angular-speed scaling applied: the
/// effective `dt` is multiplied by `scale / min`, so a fast turn zooms faster.
pub fn boom_zoom_step_scaled(
    current: f32,
    target: f32,
    dt: f32,
    obstructed: bool,
    angular_speed: f32,
) -> f32 {
    let scale = zoom_speed_scale(angular_speed);
    let mult = scale / ZOOM_SPEED_SCALE_MIN;
    let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
    boom_zoom_step(current, target, dt * mult, obstructed)
}

/// Persistent boom-zoom state with the recovered `cam_collisionZoomOutDelay`
/// (`0.1 s`) hold before extending.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoomZoom {
    /// `1.0` = full boom, `0.0` = fully retracted.
    pub zoom: f32,
    /// Remaining out-delay before extension may start.
    pub out_delay: f32,
}

impl BoomZoom {
    /// Full boom, no delay pending.
    pub fn new() -> BoomZoom {
        BoomZoom {
            zoom: 1.0,
            out_delay: 0.0,
        }
    }

    /// Advance one frame. Retraction is immediate (scaled by angular speed);
    /// extension waits out [`BOOM_ZOOM_OUT_DELAY`]. Returns the new zoom.
    pub fn update(&mut self, obstructed: bool, dt: f32, angular_speed: f32) -> f32 {
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
        if obstructed {
            self.out_delay = BOOM_ZOOM_OUT_DELAY;
            self.zoom = boom_zoom_step_scaled(self.zoom, 0.0, dt, true, angular_speed);
        } else if self.out_delay > 0.0 {
            self.out_delay = (self.out_delay - dt).max(0.0);
        } else {
            self.zoom = boom_zoom_step_scaled(self.zoom, 1.0, dt, false, angular_speed);
        }
        self.zoom
    }
}

impl Default for BoomZoom {
    fn default() -> Self {
        BoomZoom::new()
    }
}

// ---------------------------------------------------------------------------
// doorway collision (`CamCollision::UpdateDoors 0x015ed3c0`, `FindDoorway 0x015f1080`)
// ---------------------------------------------------------------------------

/// `cam_collisionDoorDepth` (`DAT_01e645c8`), metres.
pub const DOOR_DEPTH: f32 = 2.0;
/// `cam_collisionDoorBlendTime` (`DAT_01e645c4`), seconds.
pub const DOOR_BLEND_TIME: f32 = 2.0;
/// Door ring capacity (`spec+0x44`, indices `spec+0xfc0/0xfc4`).
pub const MAX_DOOR_CANDIDATES: usize = 4;

/// A doorway candidate: the two jamb points and the plane normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Doorway {
    /// Left jamb (`DoorLeft`).
    pub left: [f32; 3],
    /// Right jamb (`DoorRight`).
    pub right: [f32; 3],
    /// Doorway plane normal.
    pub normal: [f32; 3],
}

/// Doorway collision: intersect the camera segment `cam_seg = (anchor, pos)`
/// with up to [`MAX_DOOR_CANDIDATES`] doorway planes and blend the camera
/// toward the nearest crossing by `blend` (scaled by [`DOOR_BLEND_TIME`]).
///
/// Returns `None` when the segment crosses no doorway opening. Total on
/// non-finite input.
pub fn doorway_collision(
    cam_seg: ([f32; 3], [f32; 3]),
    doors: &[Doorway],
    depth: f32,
    blend: f32,
) -> Option<[f32; 3]> {
    let a = sane3(cam_seg.0);
    let b = sane3(cam_seg.1);
    let depth = if depth.is_finite() && depth > 0.0 {
        depth
    } else {
        DOOR_DEPTH
    };
    let dir = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let seg_len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if !seg_len.is_finite() || seg_len < 1e-6 {
        return None;
    }

    let mut best_t = f32::INFINITY;
    for d in doors.iter().take(MAX_DOOR_CANDIDATES) {
        let left = sane3(d.left);
        let right = sane3(d.right);
        let n = sane3(d.normal);
        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if !nl.is_finite() || nl < 1e-6 {
            continue;
        }
        let n = [n[0] / nl, n[1] / nl, n[2] / nl];
        let denom = dir[0] * n[0] + dir[1] * n[1] + dir[2] * n[2];
        if denom.abs() < 1e-6 {
            continue;
        }
        let num = (left[0] - a[0]) * n[0] + (left[1] - a[1]) * n[1] + (left[2] - a[2]) * n[2];
        let t = num / denom;
        if !t.is_finite() || t <= 1e-4 || t > 1.0 {
            continue;
        }
        let p = [a[0] + dir[0] * t, a[1] + dir[1] * t, a[2] + dir[2] * t];
        let axis = [right[0] - left[0], right[1] - left[1], right[2] - left[2]];
        let axis_len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if !axis_len.is_finite() || axis_len < 1e-6 {
            continue;
        }
        let rel = [p[0] - left[0], p[1] - left[1], p[2] - left[2]];
        let along = (rel[0] * axis[0] + rel[1] * axis[1] + rel[2] * axis[2]) / axis_len;
        if along < 0.0 || along > axis_len {
            continue;
        }
        let vert_sq =
            (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2] - along * along).max(0.0);
        if vert_sq.sqrt() > depth {
            continue;
        }
        if t < best_t {
            best_t = t;
        }
    }

    if !best_t.is_finite() {
        return None;
    }
    let door_point = [
        a[0] + dir[0] * best_t,
        a[1] + dir[1] * best_t,
        a[2] + dir[2] * best_t,
    ];
    let bf = (sane(blend) / DOOR_BLEND_TIME).clamp(0.0, 1.0);
    Some(blend_weighted(b, door_point, bf))
}

// ---------------------------------------------------------------------------
// FOV + invert flags
// ---------------------------------------------------------------------------

/// Default render FOV (`0x42800000` = `64.0`, `Camera::Update` step 3).
pub const FOV_DEGREES: f32 = 64.0;
/// [`FOV_DEGREES`] in radians.
pub const FOV_RADIANS: f32 = FOV_DEGREES * core::f32::consts::PI / 180.0;

/// `cl_invertMouse` / `cl_invertMouseX` look-inversion flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InvertFlags {
    /// `cl_invertMouse` — invert pitch (Y).
    pub invert_mouse: bool,
    /// `cl_invertMouseX` — invert yaw (X).
    pub invert_mouse_x: bool,
}

impl Default for InvertFlags {
    fn default() -> Self {
        InvertFlags {
            invert_mouse: false,
            invert_mouse_x: false,
        }
    }
}

/// Apply [`InvertFlags`] to a raw look delta `(dx, dy)`. Total.
pub fn apply_invert(dx: f32, dy: f32, flags: InvertFlags) -> (f32, f32) {
    let x = sane(dx) * if flags.invert_mouse_x { -1.0 } else { 1.0 };
    let y = sane(dy) * if flags.invert_mouse { -1.0 } else { 1.0 };
    (x, y)
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

/// Slab test on the segment `o -> o + d` (parameter in `[0, 1]`).
fn segment_hit(o: [f32; 3], d: [f32; 3], b: &physics::Aabb) -> Option<f32> {
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;
    for i in 0..3 {
        if d[i].abs() < 1e-12 {
            if o[i] < b.min[i] || o[i] > b.max[i] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d[i];
        let mut ta = (b.min[i] - o[i]) * inv;
        let mut tb = (b.max[i] - o[i]) * inv;
        if ta > tb {
            core::mem::swap(&mut ta, &mut tb);
        }
        if ta > t0 {
            t0 = ta;
        }
        if tb < t1 {
            t1 = tb;
        }
        if t0 > t1 {
            return None;
        }
    }
    if (0.0..=1.0).contains(&t0) {
        Some(t0)
    } else {
        None
    }
}

/// Map a non-finite scalar to 0 so nothing NaN escapes.
#[inline]
fn sane(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[inline]
fn sane3(v: [f32; 3]) -> [f32; 3] {
    [sane(v[0]), sane(v[1]), sane(v[2])]
}

#[inline]
fn sane4(v: [f32; 4]) -> [f32; 4] {
    [sane(v[0]), sane(v[1]), sane(v[2]), sane(v[3])]
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> tuning::CameraTuning {
        tuning::default_camera()
    }

    fn len(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        len([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    }

    fn idle(aim: bool) -> CameraInput {
        CameraInput { aim, dt: 1.0 / 60.0, ..Default::default() }
    }

    #[test]
    fn pitch_limit_is_eighty_degrees() {
        assert!((PITCH_LIMIT.to_degrees() - 80.0).abs() < 1e-3);
    }

    #[test]
    fn pitch_clamps() {
        let t = t();
        let mut up = CameraState::new();
        for _ in 0..600 {
            let inp = CameraInput { mouse_dy: 100.0, ..idle(false) };
            update(&mut up, &inp, &t, [0.0; 3]);
        }
        assert!(up.pitch <= PITCH_LIMIT + 1e-4, "pitch {} above limit", up.pitch);
        assert!(
            (up.pitch - PITCH_LIMIT).abs() < 1e-3,
            "pitch should reach the clamp, got {}",
            up.pitch
        );

        let mut down = CameraState::new();
        for _ in 0..600 {
            let inp = CameraInput { mouse_dy: -100.0, ..idle(false) };
            update(&mut down, &inp, &t, [0.0; 3]);
        }
        assert!(down.pitch >= -PITCH_LIMIT - 1e-4, "pitch {} below limit", down.pitch);
        assert!(
            (down.pitch + PITCH_LIMIT).abs() < 1e-3,
            "pitch should reach the clamp, got {}",
            down.pitch
        );
        assert!(up.pitch.is_finite() && down.pitch.is_finite());
    }

    /// Mouse input is applied in full on the frame it arrives and stops the
    /// frame it ends (retail flicks land in a single frame, no settle).
    #[test]
    fn mouse_rotation_is_immediate() {
        let t = t();
        let inp = CameraInput { mouse_dx: 10.0, ..idle(false) };
        let mut s = CameraState::new();
        update(&mut s, &inp, &t, [0.0; 3]);
        let v1 = s.yaw_vel;
        assert!(v1 > 0.0, "expected positive yaw velocity, got {v1}");
        update(&mut s, &inp, &t, [0.0; 3]);
        assert!((s.yaw_vel - v1).abs() < 1e-6, "no ramp: {v1} -> {}", s.yaw_vel);
        update(&mut s, &idle(false), &t, [0.0; 3]);
        assert_eq!(s.yaw_vel, 0.0, "stops on release");
    }

    /// The right stick reproduces the retail capture: dead through 0.65,
    /// 90.1 °/s yaw at full deflection, same shape on pitch.
    #[test]
    fn stick_matches_retail_capture() {
        let deg = |r: f32| r.to_degrees();
        assert_eq!(stick_response(0.25), 0.0);
        assert_eq!(stick_response(0.65), 0.0);
        let cases = [(0.70, 6.85), (0.75, 11.9), (0.80, 19.9), (0.85, 30.3), (0.90, 45.6), (0.95, 65.4), (1.0, 90.1)];
        for (m, expect) in cases {
            let got = deg(stick_response(m) * CAM_YAW_SPEED);
            assert!((got - expect).abs() < 0.5, "yaw at {m}: {got} vs {expect}");
        }
        let pitch85 = deg(stick_response(0.85) * CAM_PITCH_SPEED);
        assert!((pitch85 - 27.1).abs() < 1.0, "pitch at 0.85: {pitch85}");

        // One second of full right stick turns the camera by the full rate,
        // from the first frame, and stops on release.
        let t = t();
        let mut s = CameraState::new();
        let full = CameraInput { stick: [1.0, 0.0], ..idle(false) };
        for _ in 0..60 {
            update(&mut s, &full, &t, [0.0; 3]);
        }
        assert!((s.yaw - CAM_YAW_SPEED).abs() < 1e-3, "yaw after 1 s {}", s.yaw);
        update(&mut s, &idle(false), &t, [0.0; 3]);
        assert_eq!(s.yaw_vel, 0.0);
        let mut l = CameraState::new();
        update(&mut l, &CameraInput { stick: [-0.9, 0.0], ..idle(false) }, &t, [0.0; 3]);
        assert!(l.yaw_vel < 0.0, "left stick turns the other way");
    }

    /// Expected mouse angular velocity for `(dx, dy)` counts in one frame.
    fn expected_mouse(dx: f32, dy: f32) -> (f32, f32) {
        let k = mouse_count_scale(MOUSE_SENSITIVITY_NORM);
        let z = 1.0 + MOUSE_DEVICE_SCALE_PER_SENS * MOUSE_SENSITIVITY_NORM;
        (
            CAM_YAW_SPEED * (dx * k).clamp(-1.0, 1.0) * z,
            CAM_PITCH_SPEED * (dy * k).clamp(-1.0, 1.0) * z,
        )
    }

    /// Values read live at the rotation site with the virtual mouse
    /// (`notes/telemetry/README.md`): 8 counts/frame -> input 0.18, device
    /// scale 4, yaw speed pi/2; 16 -> 0.36; >= 45 counts saturate at 1.
    #[test]
    fn mouse_matches_live_rotation_factors() {
        let k = mouse_count_scale(MOUSE_SENSITIVITY_NORM);
        assert!((k - 0.0225).abs() < 1e-7);
        assert!(((8.0 * k) - 0.18).abs() < 1e-6 && ((16.0 * k) - 0.36).abs() < 1e-6);
        assert_eq!(1.0 + MOUSE_DEVICE_SCALE_PER_SENS * MOUSE_SENSITIVITY_NORM, 4.0);

        let t = t();
        for (dx, dy) in [(8.0, 0.0), (16.0, -16.0), (100.0, 0.0), (400.0, 0.0)] {
            let mut s = CameraState::new();
            update(&mut s, &CameraInput { mouse_dx: dx, mouse_dy: dy, ..idle(false) }, &t, [0.0; 3]);
            let (ey, ep) = expected_mouse(dx, dy);
            assert!((s.yaw_vel - ey).abs() < 1e-5, "{dx}: {} vs {ey}", s.yaw_vel);
            assert!((s.pitch_vel - ep).abs() < 1e-5, "{dy}: {} vs {ep}", s.pitch_vel);
        }
        // 100 and 400 counts both saturate: identical rotation (the cap).
        let (a, _) = expected_mouse(100.0, 0.0);
        let (b, _) = expected_mouse(400.0, 0.0);
        assert_eq!(a, b);
        // A 50 fps frame at saturation turns 2*pi*0.02 rad (~7.2 deg).
        assert!((a * 0.02 - std::f32::consts::TAU * 0.02).abs() < 1e-5);
    }

    /// Both axes inside the 0.05 dead zone: no rotation at all (2 counts ->
    /// 0.045). One axis outside rotates both.
    #[test]
    fn camera_input_dead_zone() {
        let t = t();
        let mut s = CameraState::new();
        update(&mut s, &CameraInput { mouse_dx: 2.0, mouse_dy: 1.0, ..idle(false) }, &t, [0.0; 3]);
        assert_eq!((s.yaw_vel, s.pitch_vel), (0.0, 0.0));
        update(&mut s, &CameraInput { mouse_dx: 3.0, mouse_dy: 1.0, ..idle(false) }, &t, [0.0; 3]);
        assert!(s.yaw_vel > 0.0 && s.pitch_vel > 0.0, "{} {}", s.yaw_vel, s.pitch_vel);
    }

    #[test]
    fn aim_reduces_boom_distance() {
        let t = t();
        let mut s = CameraState::new();
        update(&mut s, &idle(false), &t, [0.0; 3]);
        assert_eq!(s.mode, CameraMode::Nav);
        let nav = dist(s.pos, s.target);
        let nav_h = s.pos[2] - s.target[2];

        update(&mut s, &idle(true), &t, [0.0; 3]);
        assert_eq!(s.mode, CameraMode::Aim);
        let aim = dist(s.pos, s.target);
        let aim_h = s.pos[2] - s.target[2];
        assert!(aim < nav, "aim boom {aim} should be shorter than nav {nav}");
        assert!(aim_h < nav_h, "aim boom should be lower: {aim_h} vs {nav_h}");
        // Recovered over-the-shoulder scale: both axes halve.
        assert!((aim - nav * AIM_DISTANCE_SCALE).abs() < 1e-3);
    }

    #[test]
    fn first_person_has_zero_boom() {
        let t = t();
        let mut s = CameraState::new();
        set_mode(&mut s, CameraMode::FirstPerson);
        update(&mut s, &idle(false), &t, [3.0, 4.0, 5.0]);
        assert_eq!(s.mode, CameraMode::FirstPerson);
        assert_eq!(s.pos, s.target);
        assert!(dist(s.pos, s.target) < 1e-6);

        // Explicit mode path too.
        let mut s2 = CameraState::new();
        update_mode(&mut s2, &idle(false), &t, [1.0, 2.0, 3.0], CameraMode::FirstPerson);
        assert_eq!(s2.mode, CameraMode::FirstPerson);
        assert_eq!(s2.pos, s2.target);
    }

    #[test]
    fn focal_target_is_chest_and_head() {
        let f = focal_target([1.0, 2.0, 3.0]);
        assert_eq!(f, [1.0, 2.0, 3.0 + FOCAL_CHEST]);
        let h = focal_target_head([1.0, 2.0, 3.0]);
        assert_eq!(h, [1.0, 2.0, 3.0 + FOCAL_EYE]);
        assert!(focal_target([f32::NAN; 3]).iter().all(|v| v.is_finite()));
        assert!(focal_target_head([f32::NAN; 3]).iter().all(|v| v.is_finite()));
    }

    // --- third-person orbit around the model -------------------------------

    /// With yaw = 0 the boom sits directly behind the target along -X and
    /// `boom_height` above it, in the same cry Z-up space as the target.
    #[test]
    fn orbit_behind_and_above_at_yaw_zero() {
        let mut s = CameraState::new();
        s.yaw = 0.0;
        s.pitch = 0.0;
        s.mode = CameraMode::Nav;
        let target = [10.0, 20.0, 1.5];
        let pos = orbit_position(&s, target);
        // Directly behind: no lateral drift, boom length back along -X.
        assert!((pos[1] - target[1]).abs() < 1e-4, "lateral drift: {pos:?}");
        assert!(pos[0] < target[0], "not behind along -X: {pos:?}");
        assert!(
            (target[0] - pos[0] - DEFAULT_BOOM_DISTANCE).abs() < 1e-3,
            "boom distance wrong: {pos:?}"
        );
        assert!(
            (pos[2] - (target[2] + DEFAULT_BOOM_HEIGHT)).abs() < 1e-3,
            "camera not boom_height above: {pos:?}"
        );
    }

    /// Yaw rotates the boom around the target in the ground plane.
    #[test]
    fn orbit_yaw_rotates_around_target() {
        let target = [0.0, 0.0, 0.0];
        let mut s = CameraState::new();
        s.pitch = 0.0;
        s.mode = CameraMode::Nav;
        s.yaw = 0.0;
        let p0 = orbit_position(&s, target);
        assert!(p0[0] < -1.0 && p0[1].abs() < 1e-3, "yaw 0 not -X: {p0:?}");
        s.yaw = std::f32::consts::FRAC_PI_2;
        let p1 = orbit_position(&s, target);
        assert!(p1[1] < -1.0 && p1[0].abs() < 1e-3, "yaw 90 not -Y: {p1:?}");
    }

    /// Pitch tilts the boom: looking up drops the camera below the model and
    /// looking down lifts it above — this is what the old flat-XZ boom missed.
    #[test]
    fn orbit_pitch_moves_camera_under_and_over() {
        let target = [0.0, 0.0, 1.5];
        let mut s = CameraState::new();
        s.mode = CameraMode::Nav;
        s.pitch = 0.0;
        let level = orbit_position(&s, target);
        s.pitch = 0.5; // looking up
        let up = orbit_position(&s, target);
        assert!(
            up[2] < level[2],
            "looking up should drop the camera below: {up:?} vs {level:?}"
        );
        s.pitch = -0.5; // looking down
        let down = orbit_position(&s, target);
        assert!(
            down[2] > level[2],
            "looking down should lift the camera above: {down:?} vs {level:?}"
        );
    }

    /// `boom_distance` is the horizontal reach at pitch 0 and `boom_height` the
    /// vertical offset above the target.
    #[test]
    fn orbit_respects_distance_and_height() {
        let target = [0.0, 0.0, 1.5];
        let mut s = CameraState::new();
        s.mode = CameraMode::Nav;
        s.yaw = 0.0;
        s.pitch = 0.0;
        s.boom_distance = 7.0;
        s.boom_height = 3.0;
        let pos = orbit_position(&s, target);
        let horiz = ((pos[0] - target[0]).powi(2) + (pos[1] - target[1]).powi(2)).sqrt();
        assert!((horiz - 7.0).abs() < 1e-3, "horizontal reach wrong: {pos:?}");
        assert!(
            (pos[2] - target[2] - 3.0).abs() < 1e-3,
            "height wrong: {pos:?}"
        );
    }

    /// Aim pulls the orbit in and down; FirstPerson collapses it to the target.
    #[test]
    fn orbit_aim_and_first_person() {
        let target = [0.0, 0.0, 0.0];
        let mut s = CameraState::new();
        s.pitch = 0.0;
        s.mode = CameraMode::Nav;
        let nav = dist(orbit_position(&s, target), target);
        s.mode = CameraMode::Aim;
        let aim = dist(orbit_position(&s, target), target);
        assert!(aim < nav, "aim {aim} should be shorter than nav {nav}");
        assert!((aim - nav * AIM_DISTANCE_SCALE).abs() < 1e-3);
        s.mode = CameraMode::FirstPerson;
        assert_eq!(orbit_position(&s, target), target);
    }

    /// `update` writes `state.pos` through `orbit_position`, so the camera is
    /// framed around the model (not a flat XZ offset).
    #[test]
    fn update_positions_via_orbit_position() {
        let t = t();
        let target = [4.0, -3.0, 1.6];
        let mut s = CameraState::new();
        s.yaw = 0.7;
        s.pitch = 0.35;
        update(&mut s, &idle(false), &t, target);
        assert_eq!(s.pos, orbit_position(&s, target));
    }

    /// `view_dir` is the unit direction from the camera to the target.
    #[test]
    fn view_dir_points_from_camera_to_target() {
        let t = t();
        let target = [3.0, -2.0, 1.5];
        let mut s = CameraState::new();
        update(&mut s, &idle(false), &t, target);
        let v = view_dir(&s);
        assert!((len(v) - 1.0).abs() < 1e-4, "not unit: {v:?}");
        let d = [
            s.target[0] - s.pos[0],
            s.target[1] - s.pos[1],
            s.target[2] - s.pos[2],
        ];
        let dl = len(d);
        let cos = dot(v, d) / dl;
        assert!(
            (cos - 1.0).abs() < 1e-4,
            "view_dir not at target: {v:?} vs {d:?} (cos {cos})"
        );
    }

    /// `orbit_position` is total on non-finite state/target.
    #[test]
    fn orbit_position_nan_is_total() {
        let mut s = CameraState::new();
        s.yaw = f32::NAN;
        s.pitch = f32::NAN;
        s.boom_distance = f32::NAN;
        s.boom_height = f32::NAN;
        let p = orbit_position(&s, [f32::NAN; 3]);
        assert!(p.iter().all(|v| v.is_finite()), "not total: {p:?}");
    }

    #[test]
    fn boom_collision_pulls_camera_in() {
        let target = [0.0, 0.0, 1.5];
        let pos = [0.0, -5.0, 1.5];
        let wall = physics::Aabb {
            min: [-2.0, -3.0, 0.0],
            max: [2.0, -2.0, 3.0],
        };
        let out = boom_collision(pos, target, &[wall]);
        assert!(
            dist(out, target) < dist(pos, target),
            "camera not pulled in: {out:?}"
        );
        assert!(out[1] > -2.3, "camera should stop in front of the wall: {out:?}");
        assert!(out.iter().all(|v| v.is_finite()));

        // No colliders: untouched.
        assert_eq!(boom_collision(pos, target, &[]), pos);
        // Collider not on the segment: untouched.
        let aside = physics::Aabb {
            min: [10.0, 10.0, 10.0],
            max: [11.0, 11.0, 11.0],
        };
        assert_eq!(boom_collision(pos, target, &[aside]), pos);
    }

    #[test]
    fn boom_collision_uses_padding_and_side_offsets() {
        let target = [0.0, 0.0, 1.5];
        let pos = [0.0, -5.0, 1.5];
        let wall = physics::Aabb {
            min: [-2.0, -3.0, 0.0],
            max: [2.0, -2.0, 3.0],
        };
        let out = boom_collision(pos, target, &[wall]);
        // Hit at 2.0 m along the boom, minus the recovered 0.15 m padding.
        assert!(
            (dist(out, target) - (2.0 - BOOM_PADDING)).abs() < 1e-3,
            "padding margin wrong: {}",
            dist(out, target)
        );

        // A thin wall only the +X side probe (offset 0.25) can see still pulls in.
        let side = physics::Aabb {
            min: [0.1, -3.0, 0.0],
            max: [0.4, -2.0, 3.0],
        };
        let out2 = boom_collision(pos, target, &[side]);
        assert!(
            dist(out2, target) < dist(pos, target) - 1e-3,
            "side-probe offset missed: {out2:?}"
        );
        assert!((dist(out2, target) - (2.0 - BOOM_PADDING)).abs() < 1e-3);

        // Disabling the offset (0.0) must miss that wall.
        let out3 = boom_collision_with(pos, target, &[side], BOOM_PADDING, 0.0);
        assert_eq!(out3, pos, "offset probe should be required: {out3:?}");
    }

    #[test]
    fn update_collide_applies_collision_and_zoom() {
        let t = t();
        let mut s = CameraState::new();
        s.yaw = 0.0; // boom behind at -X
        let wall = physics::Aabb {
            min: [-6.0, -2.0, 0.0],
            max: [-3.0, 2.0, 4.0],
        };
        let out = update_collide(&mut s, &idle(false), &t, [0.0, 0.0, 1.5], &[wall]);
        let free = dist([-5.0, 0.0, 1.5 + DEFAULT_BOOM_HEIGHT], s.target);
        assert!(dist(out, s.target) < free, "not pulled in: {out:?}");
        assert!(s.boom_zoom < 1.0, "zoom should retract: {}", s.boom_zoom);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn boom_zoom_retracts_faster_than_it_extends() {
        assert!(BOOM_ZOOM_IN_MIN < BOOM_ZOOM_OUT_MIN);
        assert!(BOOM_ZOOM_IN_MAX < BOOM_ZOOM_OUT_MAX);
        let retract = 1.0 - boom_zoom_step(1.0, 0.0, 0.4, true);
        let extend = boom_zoom_step(0.0, 1.0, 0.4, false);
        assert!(retract > extend, "retract {retract} extend {extend}");
        assert!(retract > 0.6 && extend < 0.3);
        assert_eq!(boom_zoom_time(true), (BOOM_ZOOM_IN_MIN, BOOM_ZOOM_IN_MAX));
        assert_eq!(boom_zoom_time(false), (BOOM_ZOOM_OUT_MIN, BOOM_ZOOM_OUT_MAX));
        assert!((BOOM_ZOOM_OUT_DELAY - 0.1).abs() < 1e-6);
    }

    #[test]
    fn basis_is_orthonormal() {
        for &(yaw, pitch) in &[(0.0, 0.0), (1.0, 0.3), (-2.0, -0.7), (3.0, 1.0)] {
            let mut s = CameraState::new();
            s.yaw = yaw;
            s.pitch = pitch;
            let (f, r, u) = basis(&s);
            for v in [f, r, u] {
                assert!((len(v) - 1.0).abs() < 1e-4, "not unit: {v:?}");
            }
            assert!(dot(f, r).abs() < 1e-4, "f·r = {}", dot(f, r));
            assert!(dot(f, u).abs() < 1e-4, "f·u = {}", dot(f, u));
            assert!(dot(r, u).abs() < 1e-4, "r·u = {}", dot(r, u));
        }
    }

    #[test]
    fn nan_inputs_are_total() {
        let t = t();
        let mut s = CameraState::new();
        let inp = CameraInput {
            mouse_dx: f32::NAN,
            mouse_dy: f32::NAN,
            aim: false,
            dt: f32::NAN,
            ..Default::default()
        };
        update(&mut s, &inp, &t, [f32::NAN; 3]);
        assert!(s.yaw.is_finite() && s.pitch.is_finite());
        assert!(s.yaw_vel.is_finite() && s.pitch_vel.is_finite());
        assert!(s.pos.iter().all(|v| v.is_finite()));
        assert!(s.target.iter().all(|v| v.is_finite()));

        let (f, r, u) = basis(&s);
        assert!(f.iter().chain(r.iter()).chain(u.iter()).all(|v| v.is_finite()));
        assert!(view_dir(&s).iter().all(|v| v.is_finite()));

        let out = boom_collision(
            [f32::NAN; 3],
            [f32::NAN; 3],
            &[physics::Aabb {
                min: [f32::NAN; 3],
                max: [f32::NAN; 3],
            }],
        );
        assert!(out.iter().all(|v| v.is_finite()));

        // New entry points are total as well.
        assert!(focal_target([f32::NAN; 3]).iter().all(|v| v.is_finite()));
        assert!(boom_zoom_step(f32::NAN, f32::NAN, f32::NAN, true).is_finite());
        update_mode(&mut s, &inp, &t, [f32::NAN; 3], CameraMode::FirstPerson);
        assert!(s.pos.iter().all(|v| v.is_finite()) && s.target.iter().all(|v| v.is_finite()));
        let out2 = update_collide(&mut s, &inp, &t, [f32::NAN; 3], &[]);
        assert!(out2.iter().all(|v| v.is_finite()));
    }

    // --- nav / aim cvar split ---------------------------------------------

    #[test]
    fn nav_aim_sets_match_recovered_cvars() {
        let nav = camera_set_params(CameraSet::Nav);
        assert_eq!(nav.accel_max, 0.16);
        assert_eq!(nav.threshold, 0.9);
        assert_eq!(nav.base, CAM_ROTATION_SPEED);
        assert_eq!(nav.accel_time_yaw, ACCEL_TIME_YAW);
        assert_eq!(nav.accel_time_pitch, ACCEL_TIME_PITCH);

        let aim = camera_set_params(CameraSet::Aim);
        assert_eq!(aim.base, 0.02);
        assert_eq!(aim.accel_max, 0.07);
        assert_eq!(aim.threshold, 0.9);
        assert_eq!(aim.accel_time_yaw, 0.4);
        assert_eq!(aim.accel_time_pitch, 0.6);

        for p in [nav, aim] {
            assert_eq!(p.deadzone_linear, 0.1);
            assert_eq!(p.deadzone_radius, 0.24);
        }
    }

    #[test]
    fn nav_aim_thresholds_switch_speed() {
        // Nav: threshold 0.9, base = cl_cam_rotation_speed (0.04).
        assert_eq!(camera_set_speed(CameraSet::Nav, 0.5), CAM_ROTATION_SPEED);
        assert_eq!(camera_set_speed(CameraSet::Nav, 0.9), NAV_ACCEL_MAX);
        assert_eq!(camera_set_speed(CameraSet::Nav, 1.5), NAV_ACCEL_MAX);
        // Aim: threshold 0.9, base 0.02.
        assert_eq!(camera_set_speed(CameraSet::Aim, 0.5), AIM_SPEED_BASE);
        assert_eq!(camera_set_speed(CameraSet::Aim, 0.9), AIM_ACCEL_MAX);
        assert_eq!(camera_set_speed(CameraSet::Aim, 1.2), AIM_ACCEL_MAX);
        // Total.
        assert!(camera_set_speed(CameraSet::Aim, f32::NAN).is_finite());
    }

    #[test]
    fn deadzone_zeroes_inside_and_remaps_outside() {
        let dead = camera_deadzone([0.05, 0.05], 0.1, 0.24);
        assert_eq!(dead, [0.0, 0.0]);
        let full = camera_deadzone([1.0, 0.0], 0.1, 0.24);
        assert!((full[0] - 1.0).abs() < 1e-5 && full[1] == 0.0);
        let half = camera_deadzone([0.5, 0.0], 0.1, 0.24);
        assert!(half[0] > 0.0 && half[0] < 0.5);
        let n = camera_deadzone([f32::NAN, f32::INFINITY], 0.1, 0.24);
        assert!(n.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn accel_integration_reaches_target_over_time() {
        let time = 1.0;
        assert_eq!(integrate_rotation_speed(0.0, 2.0, 0.0, time, true), 0.0);
        let mid = integrate_rotation_speed(0.0, 2.0, 0.5, time, true);
        assert!(mid > 0.0 && mid < 2.0, "mid {mid}");
        let done = integrate_rotation_speed(0.0, 2.0, 1.0, time, true);
        assert!((done - 2.0).abs() < 1e-5, "done {done}");
        // Acceleration disabled -> immediate.
        assert_eq!(integrate_rotation_speed(0.0, 2.0, 0.001, time, false), 2.0);
        assert!(integrate_rotation_speed(f32::NAN, f32::NAN, f32::NAN, time, true).is_finite());
    }

    // --- lag / smoothing ---------------------------------------------------

    #[test]
    fn lag_trails_then_converges() {
        let mut lag = CameraLag::new();
        // First step seeds exactly.
        let (p, _, _) = lag.step([10.0, 0.0, 0.0], 0.0, 0.0, 0.5, 1.0 / 60.0);
        assert_eq!(p, [10.0, 0.0, 0.0]);
        // Move the target: the camera trails behind it.
        let (p1, _, _) = lag.step([20.0, 0.0, 0.0], 0.0, 0.0, 0.5, 1.0 / 60.0);
        assert!(p1[0] > 10.0 && p1[0] < 20.0, "trailing: {p1:?}");
        // After enough frames it converges.
        let mut last = p1;
        for _ in 0..600 {
            last = lag.step([20.0, 0.0, 0.0], 0.0, 0.0, 0.5, 1.0 / 60.0).0;
        }
        assert!((last[0] - 20.0).abs() < 0.01, "converged: {last:?}");
        // Zero lag snaps.
        assert_eq!(lag_smooth_scalar(0.0, 5.0, 0.0, 1.0 / 60.0), 5.0);
        assert!(lag_smooth3([f32::NAN; 3], [1.0; 3], 0.5, 1.0 / 60.0)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn pitch_and_yaw_from_basis() {
        assert!((pitch_from_basis([1.0, 0.0, 0.0]) - 0.0).abs() < 1e-6);
        assert!((pitch_from_basis([0.0, 0.0, 1.0]) - PITCH_LIMIT).abs() < 1e-6);
        assert!((pitch_from_basis([0.0, 0.0, -1.0]) + PITCH_LIMIT).abs() < 1e-6);
        assert!((yaw_from_basis([0.0, 1.0, 0.0]) - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!(pitch_from_basis([f32::NAN; 3]).is_finite());
    }

    // --- boom offset curve -------------------------------------------------

    #[test]
    fn boom_curve_evaluates_and_defaults() {
        let mut c = BoomCurve::focal_point([9.0, 9.0, 9.0]);
        // Empty curve returns the default, including at angle 0.
        assert_eq!(c.evaluate(0.0), [9.0, 9.0, 9.0]);
        assert_eq!(c.evaluate(1.0), [9.0, 9.0, 9.0]);
        // One weighted sample at angle 1 -> w = 0.5 -> smoothstep(0.5) = 0.5.
        c.dirs[0] = [1.0, 0.0, 0.0];
        c.weights[0] = 0.5;
        let e = c.evaluate(1.0);
        assert!((e[0] - 1.0).abs() < 1e-4, "curve {e:?}");
        // At angle 0 it still returns the default.
        assert_eq!(c.evaluate(0.0), [9.0, 9.0, 9.0]);
        // Total on non-finite.
        assert!(c.evaluate(f32::NAN).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn boom_offset_rotates_by_basis() {
        let mut c = BoomCurve::focal_point([0.0; 3]);
        c.dirs[0] = [1.0, 0.0, 0.0];
        c.weights[0] = 0.5;
        let identity = [0.0, 0.0, 0.0, 1.0];
        let o = boom_offset(&c, 1.0, identity);
        assert!((o[0] - 1.0).abs() < 1e-4, "identity {o:?}");
        // +90 deg about Z maps +X to +Y.
        let h = std::f32::consts::FRAC_PI_4;
        let q = [0.0, 0.0, h.sin(), h.cos()];
        let r = boom_offset(&c, 1.0, q);
        assert!(
            r[0].abs() < 1e-3 && (r[1] - 1.0).abs() < 1e-3,
            "rotated {r:?}"
        );
        assert!(rotate_quat([f32::NAN; 4], [f32::NAN; 3])
            .iter()
            .all(|v| v.is_finite()));
    }

    // --- secondary focal ---------------------------------------------------

    #[test]
    fn secondary_focal_blends_at_forty_percent() {
        assert_eq!(SECONDARY_FOCAL_WEIGHT, 0.4);
        let b = blend_secondary_focal([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], true);
        assert!(
            (b[0] - 4.0).abs() < 1e-5 && (b[2] - 4.0).abs() < 1e-5,
            "{b:?}"
        );
        assert_eq!(
            blend_secondary_focal([1.0, 2.0, 3.0], [9.0; 3], false),
            [1.0, 2.0, 3.0]
        );
        assert!(blend_secondary_focal([f32::NAN; 3], [f32::NAN; 3], true)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn focal_history_tracks_velocity() {
        let mut h = FocalHistory::new();
        assert_eq!(h.push([0.0, 0.0, 0.0]), [0.0; 3]);
        assert_eq!(h.push([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
        assert_eq!(h.count, 2);
        assert_eq!(h.last(), [1.0, 2.0, 3.0]);
        for i in 0..200 {
            h.push([i as f32, 0.0, 0.0]);
        }
        assert_eq!(h.count, FOCAL_HISTORY_LEN);
        assert!(h.ring.iter().all(|v| v.iter().all(|x| x.is_finite())));
    }

    // --- speed-scaled zoom -------------------------------------------------

    #[test]
    fn zoom_speed_scale_clamps_to_five_twenty() {
        assert_eq!(zoom_speed_scale(0.0), ZOOM_SPEED_SCALE_MIN);
        assert_eq!(zoom_speed_scale(0.5), 12.5);
        assert_eq!(zoom_speed_scale(1.0), ZOOM_SPEED_SCALE_MAX);
        assert_eq!(zoom_speed_scale(10.0), ZOOM_SPEED_SCALE_MAX);
        assert!(zoom_speed_scale(f32::NAN).is_finite());
        assert_eq!(zoom_speed_scale(f32::NAN), ZOOM_SPEED_SCALE_MIN);
    }

    #[test]
    fn fast_turns_retract_harder_and_extend_after_delay() {
        let slow = boom_zoom_step_scaled(1.0, 0.0, 0.1, true, 0.0);
        let fast = boom_zoom_step_scaled(1.0, 0.0, 0.1, true, 1.0);
        assert!(fast < slow, "fast {fast} should retract more than slow {slow}");
        assert!(slow < 1.0 && fast >= 0.0);

        // The out-delay holds extension for 0.1 s.
        let mut z = BoomZoom::new();
        z.update(true, 0.01, 1.0); // obstructed -> arms the delay
        assert!(z.out_delay > 0.0);
        let held = z.update(false, 0.05, 1.0);
        assert_eq!(held, z.zoom, "held during out delay");
        z.update(false, 0.1, 1.0); // consume the remaining delay
        let before = z.zoom;
        let after = z.update(false, 0.1, 1.0);
        assert!(after > before, "extends after delay: {before} -> {after}");
        assert!(z.update(false, f32::NAN, f32::NAN).is_finite());
    }

    // --- doorway collision -------------------------------------------------

    #[test]
    fn doorway_collision_blends() {
        let anchor = [0.0, 0.0, 0.0];
        let cam = [0.0, -5.0, 0.0];
        let door = Doorway {
            left: [-1.0, -2.0, 0.0],
            right: [1.0, -2.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        };
        // Full blend (default 2.0 / 2.0) lands on the doorway crossing.
        let hit = doorway_collision((anchor, cam), &[door], DOOR_DEPTH, DOOR_BLEND_TIME).unwrap();
        assert!((hit[1] + 2.0).abs() < 1e-4, "full blend {hit:?}");
        // Zero blend keeps the free camera position.
        let free = doorway_collision((anchor, cam), &[door], DOOR_DEPTH, 0.0).unwrap();
        assert!((free[1] + 5.0).abs() < 1e-4, "zero blend {free:?}");
        // Half blend sits between.
        let half = doorway_collision((anchor, cam), &[door], DOOR_DEPTH, 1.0).unwrap();
        assert!(half[1] < -2.0 && half[1] > -5.0, "half blend {half:?}");
        // No crossing -> None.
        let beside = [5.0, -5.0, 0.0];
        assert!(doorway_collision((anchor, beside), &[door], DOOR_DEPTH, DOOR_BLEND_TIME).is_none());
        // Total on non-finite.
        assert!(
            doorway_collision(([f32::NAN; 3], [f32::NAN; 3]), &[door], DOOR_DEPTH, 1.0).is_none()
        );
    }

    #[test]
    fn doorway_uses_at_most_four_candidates() {
        let anchor = [0.0, 0.0, 0.0];
        let cam = [0.0, -5.0, 0.0];
        let far = Doorway {
            left: [-1.0, -4.0, 0.0],
            right: [1.0, -4.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        };
        let near = Doorway {
            left: [-1.0, -1.0, 0.0],
            right: [1.0, -1.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        };
        // The 5th candidate (nearest) is ignored: only the first four count.
        let doors = [far, far, far, far, near];
        let hit = doorway_collision((anchor, cam), &doors, DOOR_DEPTH, DOOR_BLEND_TIME).unwrap();
        assert!((hit[1] + 4.0).abs() < 1e-4, "5th ignored: {hit:?}");
        // With the near door inside the window it wins.
        let hit2 = doorway_collision((anchor, cam), &[far, near], DOOR_DEPTH, DOOR_BLEND_TIME).unwrap();
        assert!((hit2[1] + 1.0).abs() < 1e-4, "nearest wins: {hit2:?}");
    }

    // --- FOV + invert ------------------------------------------------------

    #[test]
    fn fov_is_sixty_four_degrees() {
        assert_eq!(FOV_DEGREES, 64.0);
        assert!((FOV_RADIANS.to_degrees() - 64.0).abs() < 1e-4);
    }

    #[test]
    fn invert_flags_flip_axes() {
        let none = InvertFlags::default();
        assert_eq!(apply_invert(1.0, 2.0, none), (1.0, 2.0));
        let y = InvertFlags {
            invert_mouse: true,
            ..none
        };
        assert_eq!(apply_invert(1.0, 2.0, y), (1.0, -2.0));
        let x = InvertFlags {
            invert_mouse_x: true,
            ..none
        };
        assert_eq!(apply_invert(1.0, 2.0, x), (-1.0, 2.0));
        let both = InvertFlags {
            invert_mouse: true,
            invert_mouse_x: true,
        };
        assert_eq!(apply_invert(1.0, 2.0, both), (-1.0, -2.0));
        assert_eq!(apply_invert(f32::NAN, f32::NAN, both), (-0.0, -0.0));
    }
}
