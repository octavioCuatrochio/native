//! CryEngine **ViewSystem** layer + view/aim modes.
//!
//! This is the generic engine-side view layer that sits *under* the UL camera
//! spec system in [`crate::camera`]. It owns the active view transform (the
//! position + quaternion the renderer and scripts consume), the view FOV, the
//! first-person body/weapon offsets, and the camera↔body coupling
//! (`Camera::CamGetQT`).
//!
//! Sources (spec, not Rust):
//! * `notes/ghidra/camera.md` §5 — ViewSystem script surface
//!   (`CScriptBind_System` registration `0x00705220`: `SetViewCamera`,
//!   `ResetToNormalCamera`, `GetViewCameraPos`, `SetViewCameraFov`;
//!   `GetViewCameraAngles` `0x0112ef80`; camera noise `0x00745de0`/`0x00745f70`).
//! * `notes/ghidra/camera.md` §2 — camera behaviour types (0 first-person,
//!   1 third-person, 2 cinematic, 3 vehicle), `cl_fpBody` (client `+0x38`,
//!   default `2`) and the first-person weapon fields `actor+0xe4/+0xf0`
//!   (`FPWeaponPos`/`FPWeaponAngles`) and `+0x114/+0x120`
//!   (`FPSecWeaponPos`/`FPSecWeaponAngles`).
//! * `notes/ghidra/player_orientation.md` §3.6 — `CamGetQT` (`0x01486940`)
//!   returns the camera position + quaternion; `CPlayerRotation::Process`
//!   copies it into the actor's `viewQuat` while sprinting (§1.3).
//!
//! Coordinate convention matches the rest of the port: right-handed **Z-up**,
//! quaternions are `[x, y, z, w]` with `w` last (`player_orientation.md` §1.1).
//! Yaw rotates about world Z, pitch about the camera right axis.
//!
//! Only `std` is used; no new dependencies.

/// Default render FOV in **degrees** (`0x42800000` = `64.0`,
/// `Camera::Update` step 3; `camera.md` §1.3/§6).
pub const DEFAULT_FOV: f32 = 64.0;

/// Smallest FOV accepted by [`set_view_fov`] (degrees). Values at or below the
/// degenerate 0° view collapse the projection, so the view layer clamps.
pub const MIN_FOV: f32 = 1.0;
/// Largest FOV accepted by [`set_view_fov`] (degrees). Below the 180° horizon
/// singularity.
pub const MAX_FOV: f32 = 179.0;

/// `DAT_017225c8` = `57.29578` rad→deg, used by `GetViewCameraAngles`
/// (`0x0112ef80` lines 41-43; `camera.md` §5/§6).
pub const RAD_TO_DEG: f32 = 57.295_78;

/// `cl_fpBody` shipped default (client `+0x38` = `2`, `camera.md` §2).
pub const FP_BODY_DEFAULT: u8 = 2;

/// `cl_camera_noise` is a per-axis angular amplitude; the shake is built from
/// two bounded sinusoids at these fixed frequencies (deterministic, no RNG).
pub const NOISE_FREQ_YAW: f32 = 1.7;
/// Pitch noise frequency (`cl_camera_noise_freq` shape, `0x00745de0`).
pub const NOISE_FREQ_PITCH: f32 = 2.3;
/// Fixed pitch phase so yaw/pitch do not peak together.
pub const NOISE_PITCH_PHASE: f32 = 0.9;

/// View kind / mode, mirroring the UL camera behaviour dispatch
/// (`spec+0x78`, `camera.md` §2) plus the scripted cinematic view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewKind {
    /// Third-person navigation boom (`spec+0x78 == 1`).
    Normal,
    /// Aiming / over-the-shoulder (`AimMode`, `SmActions` id `6`).
    Aim,
    /// First-person body camera (`spec+0x78 == 0`, `cl_fpBody`).
    FirstPerson,
    /// Scripted / fixed view (`SetViewCamera`, `View Camera … Cut-Scene`).
    Cinematic,
    /// Vehicle camera (`spec+0x78 == 3`, `CamBehaviorVehicle.cpp`).
    Vehicle,
}

/// Active view-camera state owned by the ViewSystem layer.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ViewState {
    /// Current view kind / mode.
    pub kind: ViewKind,
    /// Vertical field of view in **degrees**.
    pub fov: f32,
    /// Active view position (world Z-up).
    pub position: [f32; 3],
    /// Active view orientation, `[x, y, z, w]`.
    pub quat: [f32; 4],
    /// `cl_fpBody` value (client `+0x38`): which first-person body to render.
    pub fp_body: u8,
    /// First-person weapon position offset (`actor+0xe4`, `FPWeaponPos`).
    pub weapon_offset: [f32; 3],
    /// First-person weapon angles (`actor+0xf0`, `FPWeaponAngles`).
    pub weapon_angles: [f32; 3],
}

impl ViewState {
    /// Fresh view system: `Normal` mode, [`DEFAULT_FOV`], identity view at the
    /// origin, `cl_fpBody` default and zeroed FP weapon offsets.
    pub fn new() -> ViewState {
        ViewState {
            kind: ViewKind::Normal,
            fov: DEFAULT_FOV,
            position: [0.0; 3],
            quat: [0.0, 0.0, 0.0, 1.0],
            fp_body: FP_BODY_DEFAULT,
            weapon_offset: [0.0; 3],
            weapon_angles: [0.0; 3],
        }
    }
}

impl Default for ViewState {
    fn default() -> Self {
        ViewState::new()
    }
}

/// Set the active view camera to an explicit `position` + `quat`
/// (`CScriptBind_System::SetViewCamera`, `0x00705220`).
///
/// The quaternion is normalised and non-finite inputs are sanitised. The view
/// `kind` is left untouched so callers can drive a scripted [`ViewKind::Cinematic`]
/// view or override the current mode.
pub fn set_view_camera(state: &mut ViewState, position: [f32; 3], quat: [f32; 4]) {
    state.position = [sane(position[0]), sane(position[1]), sane(position[2])];
    state.quat = normalize_quat(quat);
}

/// Reset the view to the normal game camera (`ResetToNormalCamera`,
/// `0x00705220`): [`ViewKind::Normal`], [`DEFAULT_FOV`], identity view at the
/// origin, default `cl_fpBody` and zeroed FP weapon offsets.
pub fn reset_to_normal(state: &mut ViewState) {
    state.kind = ViewKind::Normal;
    state.fov = DEFAULT_FOV;
    state.position = [0.0; 3];
    state.quat = [0.0, 0.0, 0.0, 1.0];
    state.fp_body = FP_BODY_DEFAULT;
    state.weapon_offset = [0.0; 3];
    state.weapon_angles = [0.0; 3];
}

/// View camera `(yaw, pitch)` in **degrees**, extracted from the active view
/// quaternion (`CScriptBind_System::GetViewCameraAngles`, `0x0112ef80`, which
/// converts the active view rotation with `DAT_017225c8 = 57.29578`).
///
/// `yaw` is about world Z, `pitch` about the camera right axis. Non-finite
/// quaternions yield `(0, 0)`.
pub fn get_view_camera_angles(state: &ViewState) -> (f32, f32) {
    let q = normalize_quat(state.quat);
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let yaw = (2.0 * (x * y + w * z)).atan2(1.0 - 2.0 * (y * y + z * z));
    let pitch = (2.0 * (y * z + w * x)).clamp(-1.0, 1.0).asin();
    (yaw * RAD_TO_DEG, pitch * RAD_TO_DEG)
}

/// Set the view FOV in degrees (`SetViewCameraFov`, `0x00705220`/`0x01129dc0`),
/// clamped to [`MIN_FOV`]..=[`MAX_FOV`]. Non-finite input falls back to
/// [`DEFAULT_FOV`].
pub fn set_view_fov(state: &mut ViewState, fov: f32) {
    state.fov = if fov.is_finite() {
        fov.clamp(MIN_FOV, MAX_FOV)
    } else {
        DEFAULT_FOV
    };
}

/// Apply deterministic hand-held camera noise (`cl_camera_noise` /
/// `cl_camera_noise_freq`, `0x00745de0`/`0x00745f70`).
///
/// `amp` is the per-axis angular amplitude (radians) and `t` is the shake clock
/// (seconds). Two fixed-frequency sinusoids produce a bounded yaw/pitch delta
/// that is right-multiplied into the view quaternion. The result is fully
/// deterministic in `(amp, t)`; `amp <= 0` or non-finite `t` is a no-op, and the
/// combined perturbation angle is at most `2 * |amp|`.
pub fn apply_view_noise(state: &mut ViewState, amp: f32, t: f32) {
    if !amp.is_finite() || amp <= 0.0 {
        return;
    }
    let t = sane(t);
    let yaw = amp * (t * NOISE_FREQ_YAW).sin();
    let pitch = amp * (t * NOISE_FREQ_PITCH + NOISE_PITCH_PHASE).sin();
    let delta = quat_from_yaw_pitch(yaw, pitch);
    state.quat = normalize_quat(quat_mul(state.quat, delta));
}

/// `Camera::CamGetQT` (`0x01486940`): the active camera quaternion the actor
/// body copies while sprinting (`player_orientation.md` §1.3/§3.6). Returns the
/// normalised view quaternion.
pub fn cam_get_qt(state: &ViewState) -> [f32; 4] {
    normalize_quat(state.quat)
}

/// Couple the body orientation to the camera (`CPlayerRotation::Process`).
///
/// When `sprinting` is true the body quaternion is slerped toward `cam_quat`
/// by `blend` (clamped to `0..=1`); otherwise the body is returned unchanged
/// (normalised). `blend == 0` returns the body, `blend == 1` returns the camera.
/// All inputs are sanitised, so the function is total on NaN.
pub fn couple_body_to_camera(
    body_quat: [f32; 4],
    cam_quat: [f32; 4],
    sprinting: bool,
    blend: f32,
) -> [f32; 4] {
    let body = normalize_quat(body_quat);
    if !sprinting {
        return body;
    }
    let t = if blend.is_finite() { blend.clamp(0.0, 1.0) } else { 0.0 };
    let cam = normalize_quat(cam_quat);
    quat_slerp(body, cam, t)
}

/// First-person weapon offset: `(position, angles)`.
///
/// Mirrors `FPWeaponPos`/`FPWeaponAngles` (`actor+0xe4`/`+0xf0`) — and, for the
/// secondary weapon, `actor+0x114`/`+0x120` (`camera.md` §2).
pub fn fp_weapon_offset(state: &ViewState) -> ([f32; 3], [f32; 3]) {
    (state.weapon_offset, state.weapon_angles)
}

// ---------------------------------------------------------------------------
// Internal quaternion helpers (Z-up, `[x, y, z, w]`).
// ---------------------------------------------------------------------------

/// Non-finite → `0.0`, otherwise the value unchanged.
fn sane(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Normalise a quaternion; degenerate/non-finite input becomes identity.
fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let q = [sane(q[0]), sane(q[1]), sane(q[2]), sane(q[3])];
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len.is_finite() && len > 1e-8 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Hamilton product `a ⊗ b` (applying `b` in `a`'s local frame).
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Quaternion for `yaw` about Z then `pitch` about X (radians), half-angles:
/// `q_yaw ⊗ q_pitch`, giving `[cy·sp, sy·sp, sy·cp, cy·cp]`.
fn quat_from_yaw_pitch(yaw: f32, pitch: f32) -> [f32; 4] {
    let (sy, cy) = (sane(yaw) * 0.5).sin_cos();
    let (sp, cp) = (sane(pitch) * 0.5).sin_cos();
    [cy * sp, sy * sp, sy * cp, cy * cp]
}

/// Shortest-arc spherical interpolation from `a` to `b` by `t` (clamped
/// `0..=1`). Falls back to normalised lerp when the quaternions are nearly
/// parallel.
fn quat_slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let a = normalize_quat(a);
    let mut b = normalize_quat(b);
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }
    if dot > 0.999_5 {
        return normalize_quat([
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]);
    }
    let theta0 = dot.clamp(-1.0, 1.0).acos();
    let theta = theta0 * t;
    let sin0 = theta0.sin();
    let s0 = (theta0 - theta).sin() / sin0;
    let s1 = theta.sin() / sin0;
    normalize_quat([
        a[0] * s0 + b[0] * s1,
        a[1] * s0 + b[1] * s1,
        a[2] * s0 + b[2] * s1,
        a[3] * s0 + b[3] * s1,
    ])
}

/// Geodesic angle (radians) between two orientations, sign-insensitive.
#[cfg(test)]
fn quat_angle(a: [f32; 4], b: [f32; 4]) -> f32 {
    let a = normalize_quat(a);
    let b = normalize_quat(b);
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
        .abs()
        .clamp(0.0, 1.0);
    2.0 * dot.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn get_view_camera_angles_round_trips_known_quat() {
        // Identity is a known quat: zero yaw, zero pitch.
        let mut s = ViewState::new();
        assert_eq!(get_view_camera_angles(&s), (0.0, 0.0));

        // Round-trip a set of known yaw/pitch pairs (degrees).
        for (yaw, pitch) in [
            (0.0f32, 0.0f32),
            (30.0, 0.0),
            (0.0, 15.0),
            (-45.0, -20.0),
            (90.0, 30.0),
            (179.0, 10.0),
            (-179.0, -10.0),
        ] {
            s.quat = quat_from_yaw_pitch(yaw.to_radians(), pitch.to_radians());
            let (gy, gp) = get_view_camera_angles(&s);
            assert!(
                (gy - yaw).abs() < 1e-2,
                "yaw round-trip {yaw} -> {gy} for pitch {pitch}"
            );
            assert!(
                (gp - pitch).abs() < 1e-2,
                "pitch round-trip {pitch} -> {gp} for yaw {yaw}"
            );
        }
    }

    #[test]
    fn set_view_fov_clamps() {
        let mut s = ViewState::new();
        set_view_fov(&mut s, 90.0);
        assert!((s.fov - 90.0).abs() < EPS);

        set_view_fov(&mut s, -10.0);
        assert!((s.fov - MIN_FOV).abs() < EPS);

        set_view_fov(&mut s, 1_000.0);
        assert!((s.fov - MAX_FOV).abs() < EPS);

        set_view_fov(&mut s, f32::NAN);
        assert!((s.fov - DEFAULT_FOV).abs() < EPS);

        set_view_fov(&mut s, f32::INFINITY);
        assert!((s.fov - DEFAULT_FOV).abs() < EPS);
    }

    #[test]
    fn couple_body_to_camera_blends_only_when_sprinting() {
        let body = quat_from_yaw_pitch(0.0, 0.0);
        let cam = quat_from_yaw_pitch(80.0f32.to_radians(), 0.0);

        // Not sprinting: body unchanged regardless of blend.
        let r = couple_body_to_camera(body, cam, false, 0.5);
        assert!(quat_angle(r, body) < EPS, "non-sprint must keep the body");
        let r = couple_body_to_camera(body, cam, false, 1.0);
        assert!(quat_angle(r, body) < EPS, "non-sprint blend=1 keeps the body");

        // Sprinting, blend 0 -> body; blend 1 -> camera.
        let r0 = couple_body_to_camera(body, cam, true, 0.0);
        assert!(quat_angle(r0, body) < EPS, "blend 0 must keep the body");
        let r1 = couple_body_to_camera(body, cam, true, 1.0);
        assert!(quat_angle(r1, cam) < EPS, "blend 1 must reach the camera");

        // Sprinting, blend 0.5 -> half the geodesic arc.
        let full = quat_angle(body, cam);
        let half = couple_body_to_camera(body, cam, true, 0.5);
        let a_body = quat_angle(half, body);
        let a_cam = quat_angle(half, cam);
        assert!((a_body - a_cam).abs() < 1e-3, "midpoint must be equidistant");
        assert!((a_body - full * 0.5).abs() < 1e-3, "midpoint must be half-way");

        // Blend is clamped into 0..=1.
        let over = couple_body_to_camera(body, cam, true, 5.0);
        assert!(quat_angle(over, cam) < EPS, "blend >1 clamps to 1");
        let under = couple_body_to_camera(body, cam, true, -5.0);
        assert!(quat_angle(under, body) < EPS, "blend <0 clamps to 0");
    }

    #[test]
    fn noise_is_deterministic_and_bounded() {
        let base = ViewState::new();

        // Deterministic: identical (amp, t) yields an identical quat.
        let mut a = base;
        let mut b = base;
        apply_view_noise(&mut a, 0.05, 1.234);
        apply_view_noise(&mut b, 0.05, 1.234);
        assert_eq!(a.quat, b.quat, "noise must be deterministic");

        // Bounded: combined yaw+pitch perturbation <= 2*amp.
        for t in [0.0f32, 0.1, 0.5, 1.0, 2.7, 10.0, 123.4] {
            let mut s = base;
            apply_view_noise(&mut s, 0.05, t);
            let angle = quat_angle(s.quat, base.quat);
            assert!(angle <= 2.0 * 0.05 + EPS, "noise angle {angle} exceeded bound at t={t}");
            let len = (s.quat[0] * s.quat[0]
                + s.quat[1] * s.quat[1]
                + s.quat[2] * s.quat[2]
                + s.quat[3] * s.quat[3])
                .sqrt();
            assert!((len - 1.0).abs() < EPS, "noise quat must stay unit length");
            assert!(s.quat.iter().all(|c| c.is_finite()));
        }

        // amp <= 0 is a no-op.
        let mut z = base;
        apply_view_noise(&mut z, 0.0, 3.0);
        assert_eq!(z.quat, base.quat);
    }

    #[test]
    fn nan_inputs_are_total() {
        let nan = f32::NAN;

        // FOV.
        let mut s = ViewState::new();
        set_view_fov(&mut s, nan);
        assert!(s.fov.is_finite());

        // View camera setter with NaN position + quat.
        set_view_camera(&mut s, [nan, nan, nan], [nan, nan, nan, nan]);
        assert!(s.position.iter().all(|c| c.is_finite()));
        assert!(s.quat.iter().all(|c| c.is_finite()));
        assert!(get_view_camera_angles(&s).0.is_finite());
        assert!(get_view_camera_angles(&s).1.is_finite());
        assert!(cam_get_qt(&s).iter().all(|c| c.is_finite()));

        // Noise with NaN amp/t.
        apply_view_noise(&mut s, nan, nan);
        assert!(s.quat.iter().all(|c| c.is_finite()));

        // Body coupling with NaN quats/blend.
        let r = couple_body_to_camera([nan; 4], [nan; 4], true, nan);
        assert!(r.iter().all(|c| c.is_finite()));
        let r = couple_body_to_camera([nan; 4], [nan; 4], false, nan);
        assert!(r.iter().all(|c| c.is_finite()));

        // fp_weapon_offset is total and mirrors the stored offsets.
        let (pos, ang) = fp_weapon_offset(&s);
        assert_eq!(pos, s.weapon_offset);
        assert_eq!(ang, s.weapon_angles);
    }

    #[test]
    fn new_and_reset_defaults() {
        let mut s = ViewState::new();
        assert_eq!(s.kind, ViewKind::Normal);
        assert!((s.fov - DEFAULT_FOV).abs() < EPS);
        assert_eq!(s.quat, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.fp_body, FP_BODY_DEFAULT);

        s.kind = ViewKind::FirstPerson;
        s.fov = 30.0;
        s.position = [1.0, 2.0, 3.0];
        s.weapon_offset = [0.1, 0.2, 0.3];
        reset_to_normal(&mut s);
        assert_eq!(s.kind, ViewKind::Normal);
        assert!((s.fov - DEFAULT_FOV).abs() < EPS);
        assert_eq!(s.position, [0.0; 3]);
        assert_eq!(s.quat, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.weapon_offset, [0.0; 3]);
        assert_eq!(s.weapon_angles, [0.0; 3]);
    }

    #[test]
    fn cam_get_qt_returns_normalised_view() {
        let mut s = ViewState::new();
        set_view_camera(&mut s, [4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 2.0]);
        let q = cam_get_qt(&s);
        let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((len - 1.0).abs() < EPS);
        assert!((q[3] - 1.0).abs() < EPS);
        assert_eq!(s.position, [4.0, 5.0, 6.0]);
    }
}
