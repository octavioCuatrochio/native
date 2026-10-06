//! 1:1 port of Class3 `CPlayerMovementController` + `CPlayerRotation`.
//!
//! Recovered from `StateOfDecay.unpacked.exe` (see `notes/ghidra/movement.md`,
//! `notes/ghidra/movement_input.md` and the raw decompiles under
//! `notes/ghidra/movement/decomp/`):
//!
//! * `CPlayerMovementController::UpdateNormal` `0x01619ca0` — desired direction,
//!   stance speed, delta angles (spec §6).
//! * per-frame orchestrator `0x01505ae0` — order is
//!   `input -> ctor -> Process -> Commit` (spec §5).
//! * `CPlayerRotation::Process` `0x0161eae0` / `ProcessNormal` `0x01621070`
//!   — up-vector interpolation gain `min(dt*5,1)` (spec §7.3) and the
//!   `asin`/`atan2` delta-angle accumulation (spec §6.3 step 9); `Process`
//!   builds the Euler→quaternion with `*0.5` half-angles
//!   (`DAT_017877c4` = 0.5, spec §5 note).
//! * `CPlayerRotation::Commit` `0x0161de60` — normalize + write back the yaw
//!   basis / Euler (spec §7.5); commits the per-frame yaw.
//! * ground probe `0x01620b10` — terrain ray for lean/up (spec §7.2 step 2);
//!   [`crate::class3_physics`] reproduces the ground normal / jump integration.
//! * jump executor `0x0156a560` line 566 — initial vertical velocity
//!   `sqrt(|g| * 2 * jumpHeight) / g - v0 * 0.5`, folded through the `g * v`
//!   scale on lines 570–576 to `sqrt(2*g*h) - g*v0*0.5` (spec §6.2).
//!
//! The controller is pure and deterministic: it consumes a [`MoveState`] +
//! [`MoveInput`] and is total (no panics) for any finite or non-finite input.
//! [`CPlayerMovementController`] wraps the same [`step`] with the tuning, the
//! selectable delta clamp, and the [`pseudo_speed`](CPlayerMovementController::pseudo_speed)
//! / [`capsule`](CPlayerMovementController::capsule) accessors the animation and
//! collision layers need.
//!
//! # Facing
//! CryEngine characters are authored with local **forward = +Y**, while this
//! controller's yaw convention is **0 → +X** ([`basis_from_yaw`] forward is
//! `(cos yaw, sin yaw, 0)`). The app must therefore apply
//! [`render_yaw`] (`actor_yaw - π/2`, [`MODEL_FORWARD_OFFSET`]) to the **model
//! transform** so the authored `+Y` forward renders along the controller's
//! `+X`; the controller keeps `state.yaw` unchanged.
//!
//! # Zero input
//! A frame with `forward == 0 && strafe == 0` is exactly idle: [`step`] forces
//! `drive == 0`, `desired_speed == 0` and `desired_vel == [0, 0, 0]`, and
//! residual inertia is bled to zero, so the player never creeps. Mouse-look
//! deltas rotate the body but never contribute speed.
//!
//! # Positioning
//! [`MoveState::pos`] is the capsule **feet** in Z-up world space: the solid
//! spans `pos.z ..= pos.z + height`. `update_normal` snaps the feet onto the
//! terrain/collider surface and sets [`MoveState::grounded`]; the assembled CDF
//! has its feet at local `z = 0`, so `pos` is exactly the root translation that
//! lands the character on the ground ([`spawn_on_ground`] for spawn/teleport).
//!
//! # PseudoSpeed
//! [`CPlayerMovementController::pseudo_speed`] is the value written to the
//! animation graph input `PseudoSpeed` (`controller+0x453`, `0x01619ca0`
//! line ~1431; `animation.md` §1.1): the horizontal speed normalised by the
//! shipped run speed, so run is `1.0`. See [`pseudo_speed_from_speed`] for the
//! band table.
//!
//! Shipped `cl_movement_*` cvars are all unset in the image (`.data` globals
//! `0x190f4a8..0x190f4c4` initialised to `-1.0`), so the `player.lua` stance
//! values reproduced by [`crate::tuning::default_movement`] are the live ones
//! (`notes/ghidra/movement.md` §4, `tuning.rs` module docs).
//!
//! # Inertia
//! `player.lua` exposes `inertia = 10.0` and `inertiaAccel = 16.0`; the
//! gameParams loader `0x01509d20` maps them to the controller at `+0xea4`
//! / `+0xea8` (`notes/ghidra/movement_input.md` §4.1), and
//! `CPlayer::PostPhysicalize` `0x01504890` reads them (`0x15048b0`/`0x15048e0`)
//! and pushes them to the movement request (`vtable+0xa4`, flags `|0x58`). The
//! downstream consumer is not in the export (`player_orientation.md` §5.1).
//!
//! Real-game telemetry resolves the *effective* curve: the run displacement
//! ramps with a first-order time constant **`τ ≈ 0.2 s`** (accel ~0.6 s to 95%,
//! stop slide ≈0.8 m), not the `τ = 0.063 s` that the shipped `inertiaAccel`
//! would give (`notes/movement_telemetry.md`). [`integrate_inertia`] therefore
//! uses the measured `MovementTuning::velocity_tau_accel` / `velocity_tau_decel`.
//!
//! # Constrained rotation
//! `CPlayerRotation::Process` `0x0161eae0` dispatches on `actor+0x16c`: the free
//! `ProcessNormal` `0x01621070` path, or
//! [`process_constrained_to_markup`] (`0x0161f8a0`) when the actor is
//! constrained to a markup. [`step`] selects the latter from the controller's
//! `constrained` flag, which also selects the `π*dt` delta clamp
//! (`DAT_01914de8`) over the `20π*dt` default (`DAT_01914df0`).

use crate::class3_physics;
use crate::physics::{Aabb, CharacterController};
use crate::tuning::MovementTuning;
use std::f32::consts::PI;

/// Capsule radius used for grounding (matches the player stand stance
/// `size.x = 0.375`, `player.md` §3; [`class3_physics::PLAYER_CAPSULE_RADIUS`]).
pub const CAPSULE_RADIUS: f32 = class3_physics::PLAYER_CAPSULE_RADIUS;
/// Capsule feet-to-top height used for grounding (matches the player stand
/// stance `heightCollider + 2*size.z = 1.8`, `player.md` §3;
/// [`class3_physics::PLAYER_CAPSULE_HEIGHT`]).
pub const CAPSULE_HEIGHT: f32 = class3_physics::PLAYER_CAPSULE_HEIGHT;
/// `true` — the player stand stance ships `useCapsule = 1`
/// ([`class3_physics::PLAYER_USE_CAPSULE`]).
pub const USE_CAPSULE: bool = class3_physics::PLAYER_USE_CAPSULE;
/// Input magnitude below which no movement direction is produced.
const DIR_EPS: f32 = 1e-4;
/// Horizontal speed below which an idle (zero-input) frame snaps the residual
/// inertia to exactly zero, so an idle player can never creep.
const ZERO_INPUT_SPEED_EPS: f32 = 1e-3;

/// Default per-frame delta-angle rate: `20π` rad/s.
///
/// `DAT_01914df0` = `62.83185` (`movement_input.md` §5, `movement.md` §3). The
/// binary clamps `maxDelta = rate * dt` to `±maxDelta` unless the request is on
/// the movement-matching/anim path.
pub const MAX_DELTA_RATE_DEFAULT: f32 = 20.0 * PI;
/// Constrained per-frame delta-angle rate: `π` rad/s.
///
/// `DAT_01914de8` = `3.14159`; taken when `*(DAT_01eed558+0x13d)==0` **and** the
/// movement request carries flag bit `0x1` (`movement_input.md` §5). The `2π`
/// at `DAT_01914dec` is multiplied by `0.0` in `0x01619ca0` and is unused.
pub const MAX_DELTA_RATE_CONSTRAINED: f32 = PI;

/// `GetStanceSpeed` (`0x01501900`, actor vtable `+0x344`) actor-side global
/// speed multiplier `actor+0xe48`. The shipped player value is `1.0`
/// (`player.md` §3, `player_orientation.md` §2.5); the field itself is an
/// unnamed per-actor scalar. Passed as the first factor of
/// [`get_stance_speed`].
pub const STANCE_SPEED_ACTOR_SCALE: f32 = 1.0;
/// `GetStanceSpeed` (`0x01501900`) vtable factor `(*actor->vtable[0x288])()`.
/// The shipped player value is `1.0` (`player_orientation.md` §2.5); the slot
/// body is `[UNRECOVERED]`. Passed as the third factor of [`get_stance_speed`].
pub const STANCE_SPEED_VTABLE_SCALE: f32 = 1.0;

// --- Jump executor `0x0156a560` constants (movement_input.md §6.2) ----------

/// `DAT_0172b8b8` = `22.0` — clamp on the `jumpVec` magnitude.
pub const JUMP_VELOCITY_CLAMP: f32 = 22.0;
/// `_DAT_018903e8` = `484.0` = `22²` — `|jumpVec|²` threshold above which the
/// clamp at [`JUMP_VELOCITY_CLAMP`] is applied.
pub const JUMP_VELOCITY_CLAMP_SQ: f32 = 484.0;
/// `DAT_01730280` = `0.7` — slope cutoff on `cos²`; below it the jump vector is
/// scaled by `(0.7 - cos²) * 5.0`.
pub const JUMP_SLOPE_COS2_CUTOFF: f32 = 0.7;
/// `DAT_01720224` = `5.0` — slope response gain in `0x0156a560`.
pub const JUMP_SLOPE_GAIN: f32 = 5.0;
/// The shipped `backwardMul` / `strafeMul` / `sprintMult` jump multipliers
/// (`player.lua` `backwardMultiplier = strafeMultiplier = 1.0`,
/// `sprintMultiplier = 0` unused; `0x0156a560` writes `sprintMult = 1.0`).
pub const JUMP_MULTIPLIER_DEFAULT: f32 = 1.0;

/// Up-vector interpolation rate: `DAT_01720224` = `5.0` (`ProcessNormal`,
/// `0x01621070`; `movement.md` §7.3).
pub const UP_SLERP_RATE: f32 = 5.0;

/// Half-angle scale used when building the Euler→quaternion
/// (`DAT_017877c4` = `0.5`; `0x0161eae0` lines 219–245,
/// `movement_input.md` §5).
pub const HALF_ANGLE: f32 = 0.5;

/// Facing-convention fix: CryEngine characters are authored with local
/// **forward = +Y** (`X` right, `Y` forward, `Z` up), while this controller's
/// yaw convention is **0 → +X** ([`basis_from_yaw`] returns
/// `forward = (cos yaw, sin yaw, 0)`).
///
/// The rendered model therefore needs an extra rotation of `-π/2` about Z to
/// line its local `+Y` up with the controller's `+X`. The controller's own yaw
/// stays as-is; the app applies [`render_yaw`] to the **model transform** only.
///
/// Derivation (`Rz(θ)` maps `(0, 1, 0) → (-sin θ, cos θ, 0)`):
///
/// * `render_yaw(0) = -π/2`
/// * `Rz(render_yaw(0)) * (0, 1, 0) = (1, 0, 0)` — model forward → actor `+X`
/// * in general `Rz(render_yaw(y)) * (0, 1, 0) = (cos y, sin y, 0)`, exactly
///   [`basis_from_yaw`]'s forward row.
pub const MODEL_FORWARD_OFFSET: f32 = -core::f32::consts::FRAC_PI_2;

/// Model-space yaw for a controller-space actor yaw: `actor_yaw - π/2`.
///
/// This is the rotation to put on the assembled CDF's model transform so its
/// authored local `+Y` forward points along the controller's `+X` forward
/// (see [`MODEL_FORWARD_OFFSET`]). The controller keeps using `actor_yaw`
/// unchanged; only the render transform consumes this. Finite in and out for
/// the finite `state.yaw` the controller produces.
pub fn render_yaw(actor_yaw: f32) -> f32 {
    actor_yaw + MODEL_FORWARD_OFFSET
}

/// Below this normalised speed the [`CPlayerMovementController::pseudo_speed`]
/// reports `0.0` (idle), matching the animation graph's idle band
/// `<PseudoSpeed min=0 max=0.2>` (`playerfullbody.xml` line 922).
pub const PSEUDO_SPEED_IDLE_EPS: f32 = 1e-3;

/// Ground probe reach used to snap a freshly spawned / teleported player onto
/// the surface (metres). Generous so a spawn above the terrain still lands.
pub const SPAWN_SNAP_REACH: f32 = 100.0;

/// Ground probe reach used to keep the feet on the surface after a frame.
/// One [`class3_physics::STEP_HEIGHT`] plus a small tolerance.
pub const GROUND_SNAP_REACH: f32 = class3_physics::STEP_HEIGHT + 1e-3;

// Stance/gait selectors used by [`stance_speed`]. The stand stance's normal
// locomotion is the run speed, so index 0 = stand -> run; creep/walk/sprint are
// selected by the remaining indices. See `notes/ghidra/movement.md` §6.3 step 6:
// the stance speed is `stance+0x80 * |desired direction|`, i.e. the stance
// table's speed slot scaled by the desired direction magnitude.
/// Stand stance, normal locomotion — maps to the run speed.
pub const STANCE_STAND: u8 = 0;
/// Crouched/creep gait — maps to the creep speed.
pub const STANCE_CREEP: u8 = 1;
/// Walk gait — maps to the walk speed.
pub const STANCE_WALK: u8 = 2;
/// Sprint gait — maps to the sprint speed.
pub const STANCE_SPRINT: u8 = 3;

// The six per-stance speed slots recovered from `stance+0x84..0x98`
// (`movement_input.md` §4): creep, walk, run, sprintCreep, sprintWalk, sprint.
/// `stance+0x84` — creep speed slot.
pub const SPEED_SLOT_CREEP: usize = 0;
/// `stance+0x88` — walk speed slot.
pub const SPEED_SLOT_WALK: usize = 1;
/// `stance+0x8c` — run speed slot.
pub const SPEED_SLOT_RUN: usize = 2;
/// `stance+0x90` — sprint-while-creeping speed slot.
pub const SPEED_SLOT_SPRINT_CREEP: usize = 3;
/// `stance+0x94` — sprint-while-walking speed slot.
pub const SPEED_SLOT_SPRINT_WALK: usize = 4;
/// `stance+0x98` — sprint speed slot.
pub const SPEED_SLOT_SPRINT: usize = 5;

/// Per-frame player movement state.
///
/// `pos` is the capsule **feet** in Z-up world space (same convention as
/// [`crate::physics`]); `up` is the body up-vector; `delta_angles` is
/// `[pitch, yaw]` for the frame. `vel` is the *inertia-smoothed* velocity that
/// [`update_normal`] integrates (horizontal components) plus the vertical
/// component owned by the character controller / [`integrate_vertical`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MoveState {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub yaw: f32,
    pub up: [f32; 3],
    pub grounded: bool,
    pub stance_index: u8,
    pub delta_angles: [f32; 2],
}

impl MoveState {
    /// Fresh state at `pos` facing `+X` with an upright body and no motion.
    pub fn new(pos: [f32; 3]) -> MoveState {
        MoveState {
            pos: [sane(pos[0]), sane(pos[1]), sane(pos[2])],
            vel: [0.0; 3],
            yaw: 0.0,
            up: [0.0, 0.0, 1.0],
            grounded: false,
            stance_index: STANCE_STAND,
            delta_angles: [0.0, 0.0],
        }
    }
}

/// One frame of player input. `forward`/`strafe` are camera/body-relative
/// analog axes (typically `-1..=1`); `mouse_*` are look deltas already in
/// radians for the frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MoveInput {
    pub forward: f32,
    pub strafe: f32,
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub crouch: bool,
    pub sprint: bool,
    pub dt: f32,
}

/// The six shipped per-stance speeds, in table order
/// `stance+0x84..0x98` (`movement_input.md` §4):
/// `[creep, walk, run, sprintCreep, sprintWalk, sprint]`.
///
/// `player.lua` ships `sprintCreepSpeed == sprintWalkSpeed == sprintSpeed`
/// (`6.8`), so all three sprint slots collapse to `t.sprint`. Non-finite tuning
/// values collapse to `0.0` so the mapping is total.
pub fn stance_speeds(t: &MovementTuning) -> [f32; 6] {
    let creep = sane(t.creep).max(0.0);
    let walk = sane(t.walk).max(0.0);
    let run = sane(t.run).max(0.0);
    let sprint = sane(t.sprint).max(0.0);
    [creep, walk, run, sprint, sprint, sprint]
}

/// Read one of the six stance speed slots (`stance+0x84..0x98`); out-of-range
/// slots clamp to the sprint slot and non-finite tuning collapses to `0.0`.
pub fn stance_speed_at(slot: usize, t: &MovementTuning) -> f32 {
    let table = stance_speeds(t);
    table[slot.min(SPEED_SLOT_SPRINT)]
}

/// Map a stance/gait selector to the shipped player speed (m/s).
///
/// * [`STANCE_CREEP`] → `t.creep` (0.8)
/// * [`STANCE_WALK`] → `t.walk` (1.66)
/// * [`STANCE_STAND`] → `t.run` (4.0) — "stand -> run"; also the fallback for
///   any unknown index
/// * [`STANCE_SPRINT`] → `t.sprint` (6.8)
///
/// Cite: `notes/ghidra/movement.md` §4 (cvars unset, so `player.lua` values are
/// used) and §6.3 step 6 (stance table speed = `stance+0x80`); the six-slot
/// resolution is `movement_input.md` §4. Non-finite tuning values collapse to
/// `0.0` so the mapping is total.
pub fn stance_speed(stance_index: u8, t: &MovementTuning) -> f32 {
    let slot = match stance_index {
        STANCE_CREEP => SPEED_SLOT_CREEP,
        STANCE_WALK => SPEED_SLOT_WALK,
        STANCE_SPRINT => SPEED_SLOT_SPRINT,
        // STANCE_STAND and any unknown selector: stand stance defaults to run.
        _ => SPEED_SLOT_RUN,
    };
    stance_speed_at(slot, t)
}

/// `GetStanceSpeed` (`0x01501900`, actor vtable `+0x344`) — the full stance
/// speed product.
///
/// Recovered body (`player_orientation.md` §2.5, `player.md` §3):
///
/// ```c
/// float GetStanceSpeed(int stanceIndex) {
///     stance = (stanceIndex < 12) ? actor + stanceIndex*0xd4 + 0xa8 : &DAT_01e1ee38;
///     if (stance[0xc4] == 0) warn(...);
///     return actor[0xe48] * stance[0x80] * (*actor->vtable[0x288])();
/// }
/// ```
///
/// i.e. a **product** of three factors, not the bare `stance+0x80`:
///
/// * `actor_scale` = `actor+0xe48` — per-actor global speed multiplier
///   (shipped `1.0`, [`STANCE_SPEED_ACTOR_SCALE`]);
/// * `stance_slot_speed` = `stance+0x80` — the stance speed slot selected by
///   [`stance_speed`] (the stand stance's `+0x80` is `maxSpeed = 100.5`, but the
///   effective slot used by the controller is the gait speed from
///   `stance+0x84..0x98`);
/// * `vtable_scale` = `(*actor->vtable[0x288])()` — an unnamed speed factor
///   (shipped `1.0`, [`STANCE_SPEED_VTABLE_SCALE`]; slot body `[UNRECOVERED]`).
///
/// Non-finite factors collapse to `0.0`, so the product is total and a missing
/// factor cannot produce NaN.
pub fn get_stance_speed(actor_scale: f32, stance_slot_speed: f32, vtable_scale: f32) -> f32 {
    sane(actor_scale) * sane(stance_slot_speed) * sane(vtable_scale)
}

/// Resolve the stand-stance gait from the analog input magnitude using the
/// shipped `player.lua` thresholds, honouring the crouch/sprint overrides.
///
/// `GetStanceSpeed` (`actor->vtable+0x344`) is **not in the export** (see
/// `movement_input.md` §8.1), so the selection among `stance+0x84..0x98` is
/// reconstructed from the shipped `player.lua` controller thresholds:
/// `creepWalkStepThreshold = 0.29` and `walkRunStepThreshold = 0.69`
/// (`movement_input.md` §4.1). `drive` is `|dir|` clamped to one:
///
/// * `drive <= DIR_EPS` (idle)      → [`STANCE_STAND`]
/// * `drive < 0.29`                 → [`STANCE_CREEP`]
/// * `drive < 0.69`                 → [`STANCE_WALK`]
/// * otherwise                      → [`STANCE_STAND`] (run)
///
/// `crouch` forces [`STANCE_CREEP`] and `sprint` forces [`STANCE_SPRINT`],
/// matching the controller's explicit gait overrides. Total for any input.
pub fn gait_stance(drive: f32, crouch: bool, sprint: bool, t: &MovementTuning) -> u8 {
    if crouch {
        return STANCE_CREEP;
    }
    if sprint {
        return STANCE_SPRINT;
    }
    let d = sane(drive).max(0.0);
    if d <= DIR_EPS {
        return STANCE_STAND;
    }
    let creep_walk = sane(t.threshold_creep_walk);
    let walk_run = sane(t.threshold_walk_run);
    if d < creep_walk {
        STANCE_CREEP
    } else if d < walk_run {
        STANCE_WALK
    } else {
        STANCE_STAND
    }
}

/// The animation-graph `PseudoSpeed` input value for a horizontal speed.
///
/// `PseudoSpeed` is the **normalised locomotion band**: the recovered
/// `UpdateNormal` writes the desired velocity magnitude `|param_3[0..2]|` to
/// the graph input cached at `controller+0x453` (`0x01619ca0` line ~1431;
/// `animation.md` §1.1/§2), and the graph consumes it as a band where the run
/// speed is the unit. Dividing by the shipped run speed (`player.lua`
/// `runSpeed = 4.0`) reproduces the shipped `<PseudoSpeed>` bands in
/// `playerfullbody.xml`:
///
/// | gait   | speed (`player.lua`) | `PseudoSpeed` | graph band |
/// |--------|----------------------|---------------|------------|
/// | idle   | `0.0`                | `0.0`         | `[0, 0.2]` |
/// | creep  | `0.8`                | `0.20`        | walk       |
/// | walk   | `1.66`               | `0.415`       | `select_speed_slow=0.2` … `fast=0.8` |
/// | run    | `4.0`                | `1.0`         | `[0.6, 2.0]` (`min=0 max=1.5` state) |
/// | sprint | `6.8`                | `1.7`         | `[1.5, 3.5]` (`SprintDecoupled`) |
///
/// A non-positive `run` yields `0.0`; non-finite input collapses to `0.0`.
pub fn pseudo_speed_from_speed(speed: f32, run: f32) -> f32 {
    let run = sane(run);
    if !(run > 0.0) {
        return 0.0;
    }
    let ratio = sane(speed) / run;
    if ratio.abs() < PSEUDO_SPEED_IDLE_EPS {
        0.0
    } else {
        ratio
    }
}

/// Maximum per-frame delta angle: `rate * dt`.
///
/// The recovered controller computes `maxDelta = BASE * dt`, then clamps every
/// delta-angle component to `±maxDelta` (`notes/ghidra/movement.md` §6.3 step 9,
/// `0x01619ca0` lines 1114–1164). `BASE` is:
///
/// * [`MAX_DELTA_RATE_DEFAULT`] = `20π` (`DAT_01914df0`) for the default path;
/// * [`MAX_DELTA_RATE_CONSTRAINED`] = `π` (`DAT_01914de8`) when the request is
///   constrained to movement-matching/animation — pass `constrained = true`.
///
/// A non-positive/non-finite `dt` yields `0.0` so the clamp is a no-op.
pub fn max_delta(dt: f32, constrained: bool) -> f32 {
    let rate = if constrained {
        MAX_DELTA_RATE_CONSTRAINED
    } else {
        MAX_DELTA_RATE_DEFAULT
    };
    if dt.is_finite() && dt > 0.0 {
        rate * dt
    } else {
        0.0
    }
}

/// Up-vector interpolation gain: `min(dt * 5, 1)`.
///
/// `CPlayerRotation::ProcessNormal` `0x01621070` computes
/// `alpha = min(*param_1 * 5.0, 1.0)` before interpolating the actor up-vector
/// (`notes/ghidra/movement.md` §7.3). Clamped to `[0,1]`; non-finite → `0`.
pub fn up_slerp_gain(dt: f32) -> f32 {
    if dt.is_finite() {
        (dt * UP_SLERP_RATE).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Normalise an angle to `(-PI, PI]`; non-finite input becomes `0.0`.
pub fn wrap_angle(a: f32) -> f32 {
    if !a.is_finite() {
        return 0.0;
    }
    let tau = 2.0 * PI;
    let mut r = a % tau;
    if r > PI {
        r -= tau;
    } else if r < -PI {
        r += tau;
    }
    r
}

/// Per-axis player velocity-lag integrator (first-order exponential).
///
/// # Recovery
/// `CPlayer::PostPhysicalize` (`FUN_01504890`, `0x01504890`) reads the shipped
/// tuning at `0x15048b0` (`movss xmm0, [edi+0xea4]` = **`inertia`**) and
/// `0x15048e0` (`movss xmm0, [edi+0xea8]` = **`inertiaAccel`**), stores them into
/// a local movement-request struct and pushes it via `request->vtable+0xa4`
/// with request flags `| 0x58` (`player_orientation.md` §2.4). The **consumer**
/// that later decays the actor velocity from those request fields is not in any
/// export (`player_orientation.md` §5.1), so the exact curve was `[UNRECOVERED]`.
///
/// # Measurement
/// Real-game telemetry (`notes/movement_telemetry.md`) resolves the curve: the
/// run gait reaches ~95% of `runSpeed` in ~0.6 s from rest, and the actor slides
/// ≈0.8 m after releasing `W` from `runSpeed`. Both are a first-order lag with a
/// single time constant **`τ ≈ 0.2 s`** (`V·τ = 0.8 m`). The shipped
/// `inertiaAccel = 16` / `inertia = 10` would be `τ = 0.063 s` / `0.1 s`, i.e.
/// the game's *displacement* is ~2–3× slower than a naive `k·dt` step. We
/// therefore integrate with the measured `τ` (`MovementTuning::velocity_tau_*`),
/// which supersedes the shipped constants for parity.
///
/// # Model
/// A clamped, per-axis exponential approach. For each axis `i`:
///
/// ```text
/// speeding_up_i = |d_i| > |c_i| and (c_i == 0 or d_i and c_i share sign)
/// tau_i  = speeding_up_i ? tau_accel : tau_decel
/// alpha  = 1 - exp(-dt / tau_i)          // tau <= 0 -> snap
/// n_i    = c_i + (d_i - c_i) * alpha
/// n_i    = clamp(n_i, min(c_i, d_i), max(c_i, d_i))   // no overshoot
/// ```
///
/// Each axis is independent and can never overshoot its target. Non-finite
/// components collapse to `0`, a non-positive/non-finite `dt` leaves the
/// velocity unchanged, and a non-positive/non-finite `tau` snaps to target, so
/// the helper is total.
pub fn integrate_inertia(
    current: [f32; 3],
    desired: [f32; 3],
    tau_accel: f32,
    tau_decel: f32,
    dt: f32,
) -> [f32; 3] {
    let cur = [sane(current[0]), sane(current[1]), sane(current[2])];
    let des = [sane(desired[0]), sane(desired[1]), sane(desired[2])];
    if !(dt.is_finite() && dt > 0.0) {
        return cur;
    }
    let accel = sane(tau_accel);
    let decel = sane(tau_decel);
    let mut out = cur;
    for i in 0..3 {
        let c = cur[i];
        let d = des[i];
        // Rising target speed (same direction): use the accel time constant. A
        // sign flip or a shrinking target speed is a deceleration: use decel.
        let same_sign = c == 0.0 || d == 0.0 || (c > 0.0) == (d > 0.0);
        let speeding_up = d.abs() > c.abs() && same_sign;
        let tau = if speeding_up { accel } else { decel };
        let alpha = if tau > 0.0 {
            1.0 - (-dt / tau).exp()
        } else {
            1.0
        };
        let alpha = alpha.clamp(0.0, 1.0);
        let mut n = c + (d - c) * alpha;
        // Clamp so a single step can never overshoot the target on this axis.
        let (lo, hi) = if d >= c { (c, d) } else { (d, c) };
        n = n.clamp(lo, hi);
        out[i] = sane(n);
    }
    out
}

/// Accelerate the current velocity toward `desired` using the **measured**
/// first-order lag `τ ≈ 0.2 s` ([`MovementTuning::velocity_tau_accel`] /
/// [`MovementTuning::velocity_tau_decel`], fitted in
/// `notes/movement_telemetry.md`).
///
/// Thin wrapper over [`integrate_inertia`]. The shipped `inertia` (10) /
/// `inertiaAccel` (16) are retained on [`MovementTuning`] for provenance but are
/// not used here, because their naive `k·dt` step is ~2–3× faster than the
/// actuator's real displacement ramp.
pub fn accelerate_velocity(
    current: [f32; 3],
    desired: [f32; 3],
    t: &MovementTuning,
    dt: f32,
) -> [f32; 3] {
    integrate_inertia(current, desired, t.velocity_tau_accel, t.velocity_tau_decel, dt)
}

/// Orthonormal Z-up basis from a yaw: `[forward, right, up]`.
///
/// Mirrors the orthonormal basis `ProcessNormal` (`0x01621070`) builds from the
/// actor rotation rows before constructing the quaternion (`movement.md` §7.3);
/// in this planar port `up` is world up and `right` is `forward` rotated `-90°`.
pub fn basis_from_yaw(yaw: f32) -> [[f32; 3]; 3] {
    let (s, c) = sane(yaw).sin_cos();
    [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// Euler→quaternion using **half-angles**, matching `CPlayerRotation::Process`
/// (`0x0161eae0` lines 219–245) which multiplies `actor+0xa90..a98` by
/// `DAT_017877c4 = 0.5` (`movement_input.md` §5 note; `movement.md` §3).
///
/// `euler` is `[pitch (x), roll (y), yaw (z)]` in radians. Returns
/// `[x, y, z, w]`; non-finite components collapse to `0`.
pub fn euler_to_quat(euler: [f32; 3]) -> [f32; 4] {
    let (sp, cp) = (sane(euler[0]) * HALF_ANGLE).sin_cos();
    let (sr, cr) = (sane(euler[1]) * HALF_ANGLE).sin_cos();
    let (sy, cy) = (sane(euler[2]) * HALF_ANGLE).sin_cos();
    [
        sp * cr * cy - cp * sr * sy,
        cp * sr * cy + sp * cr * sy,
        cp * cr * sy - sp * sr * cy,
        cp * cr * cy + sp * sr * sy,
    ]
}

/// Normalise a quaternion; a degenerate/non-finite input becomes the identity
/// `[0, 0, 0, 1]`. Mirrors `CPlayerRotation::Commit`'s
/// `rsqrt`-based normalisation of every written quaternion (`0x0161de60`).
pub fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let q = [sane(q[0]), sane(q[1]), sane(q[2]), sane(q[3])];
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len.is_finite() && len > 1e-8 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Yaw (radians, wrapped) of a quaternion in the ZYX convention used by
/// [`euler_to_quat`]. For a pure yaw quaternion this is exactly the input yaw.
pub fn quat_yaw(q: [f32; 4]) -> f32 {
    wrap_angle(2.0 * sane(q[2]).atan2(sane(q[3])))
}

/// Launch vertical velocity for a jump reaching `height` metres under
/// `gravity`, retaining half of the current vertical velocity.
///
/// Recovered from `FUN_0156a560` line 566
/// `v_y = sqrt(|g| * 2 * jumpHeight) / g - v0 * 0.5`; the surrounding
/// composition (lines 570–576) multiplies by `g`, yielding the folded form
/// `sqrt(2*g*h) - g*v0*0.5` (`movement_input.md` §6.2). The `- v0*0.5` momentum
/// term is `[UNRECOVERED]` in detail (the `v0` source field is not identified);
/// this port keeps the projectile launch [`class3_physics::jump_velocity`] and
/// subtracts half of `vy`. The result is clamped to
/// `±`[`JUMP_VELOCITY_CLAMP`] (`DAT_0172b8b8` = `22.0`; the binary applies it
/// when `|jumpVec|² >` `_DAT_018903e8` = `484`, `movement_input.md` §6.2).
/// Degenerate `height`/`gravity` leave `vy` unchanged (still clamped).
pub fn jump(vy: f32, height: f32, gravity: f32) -> f32 {
    let launch = class3_physics::jump_velocity(gravity, height);
    let v = if launch <= 0.0 {
        sane(vy)
    } else {
        sane(launch - HALF_ANGLE * sane(vy))
    };
    v.clamp(-JUMP_VELOCITY_CLAMP, JUMP_VELOCITY_CLAMP)
}

/// Integrate one vertical jump/fall step against `ground_h`, grounding through
/// [`class3_physics::integrate_jump`] (`-GRAVITY * dt`, snap at the surface,
/// cancel downward velocity). Updates `state.pos[2]`, `state.vel[2]` and
/// `state.grounded`; total for any input.
pub fn integrate_vertical(state: &mut MoveState, ground_h: f32, dt: f32) {
    let mut vz = sane(state.vel[2]);
    let mut grounded = state.grounded;
    let z = class3_physics::integrate_jump(state.pos[2], &mut vz, &mut grounded, ground_h, dt);
    state.pos[2] = sane(z);
    state.vel[2] = sane(vz);
    state.grounded = grounded;
}

/// `CPlayerRotation::ProcessConstrainedToMarkup` (`0x0161f8a0`) — the
/// markup-constrained rotation path, taken by `CPlayerRotation::Process`
/// (`0x0161eae0`) when `actor+0x16c` (`isConstrainedToMarkup`) is set.
///
/// # Recovery
/// The function shares [`up_slerp_gain`]'s interpolation preamble
/// (`alpha = min(dt*5, 1)`, `DAT_01720224` = 5.0) and then computes a rotation
/// constrained to the markup orientation read from the actor's rotation
/// component (`actor+0xc` vtable `+0x90`, the world quaternion)
/// (`movement.md` §7.4, `player.md` §6.4, `player_orientation.md` §6.4). The
/// full 4.7 KB body is **not** line-reversed (`movement.md` §10, `player.md`
/// §11.6), so this port reproduces the recovered structure: rotate the actor
/// yaw toward the target direction at the **constrained** per-frame limit
/// `π*dt` ([`MAX_DELTA_RATE_CONSTRAINED`], `DAT_01914de8`) — the
/// `20π*dt` default (`DAT_01914df0`) is only for the free `ProcessNormal` path.
///
/// `target_dir` is the frame's desired world direction (horizontal). When it is
/// degenerate (`|xy| <=` [`DIR_EPS`]) the already-accumulated
/// [`MoveState::delta_angles`] yaw is applied instead (idle / mouse-look), so
/// mouse deltas added after the clamp still pass through. The up-vector slerp
/// is performed by the caller ([`step`]) because it needs the ground normal.
/// Total for any input: non-finite components collapse to `0` and the applied
/// delta is always clamped to `±π*dt`.
pub fn process_constrained_to_markup(state: &mut MoveState, target_dir: [f32; 3], dt: f32) {
    let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
    let max_d = max_delta(dt, true);
    let dx = sane(target_dir[0]);
    let dy = sane(target_dir[1]);
    let horiz = (dx * dx + dy * dy).sqrt();
    let delta = if horiz > DIR_EPS {
        // Markup target yaw, clamped to the constrained per-frame limit.
        wrap_angle(dy.atan2(dx) - state.yaw).clamp(-max_d, max_d)
    } else {
        // Idle / mouse-look: keep the pre-accumulated (already clamped) yaw.
        sane(state.delta_angles[1])
    };
    state.delta_angles[1] = delta;
    state.yaw = wrap_angle(state.yaw + delta);
}

/// Per-frame outputs of [`step`], the recovered `UpdateNormal` value set.
///
/// Mirrors the `controller+0x2c8` desired-speed cache and the `param_3[0..2]`
/// desired-velocity output (`movement_input.md` §3–4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOut {
    /// `controller+0x2c8` — `stance+0x80 * |dir|`, the desired speed.
    pub desired_speed: f32,
    /// `param_3[0..2]` — desired world velocity (`dir * desired_speed`).
    pub desired_vel: [f32; 3],
    /// The stance/gait selector resolved this frame.
    pub stance: u8,
    /// Analog input magnitude `|dir|`, clamped to one.
    pub drive: f32,
}

/// Core recovered `UpdateNormal` (`0x01619ca0`) step.
///
/// Shared by the free [`update_normal`] (which pins the movement-matching /
/// animation path) and [`CPlayerMovementController`] (which selects it from its
/// `constrained` flag). `constrained = true` uses the `π*dt` delta limit
/// (`DAT_01914de8`) and the [`process_constrained_to_markup`] rotation path
/// (`0x0161f8a0`, `actor+0x16c`); `false` uses the default `20π*dt` limit
/// (`DAT_01914df0`) and the free `ProcessNormal` quaternion commit.
pub fn step(
    state: &mut MoveState,
    input: &MoveInput,
    t: &MovementTuning,
    constrained: bool,
    ground: &dyn Fn(f32, f32) -> f32,
    colliders: &[Aabb],
) -> StepOut {
    let dt = if input.dt.is_finite() && input.dt > 0.0 {
        input.dt
    } else {
        0.0
    };

    // 1. desired world direction, relative to current facing.
    let fwd = sane(input.forward);
    let strafe = sane(input.strafe);
    let mut mag = (fwd * fwd + strafe * strafe).sqrt();
    if !mag.is_finite() {
        mag = 0.0;
    }
    // |dir| in the spec is the input direction magnitude, clamped to one so a
    // diagonal stick cannot exceed the stance speed. // [UNRECOVERED]
    //
    // Dead-zone: below `DIR_EPS` the frame is *exactly* idle, so `drive` is
    // forced to `0.0` (not merely the direction). Otherwise a tiny stick
    // residue would still yield `desired_speed = stance_speed * drive > 0`
    // while `dir == 0`, i.e. a nonzero desired speed at zero input.
    let drive = if mag > DIR_EPS { mag.min(1.0) } else { 0.0 };
    let basis = basis_from_yaw(state.yaw);
    let (dir_x, dir_y) = if drive > 0.0 {
        let inv = 1.0 / mag;
        let (lx, ly) = (fwd * inv, strafe * inv);
        // world = lx * forward + ly * right (Z-up basis built in ProcessNormal).
        (
            lx * basis[0][0] + ly * basis[1][0],
            lx * basis[0][1] + ly * basis[1][1],
        )
    } else {
        (0.0, 0.0)
    };

    // 2. stance speed = stance table slot * |dir|. Crouch/sprint override the
    //    gait; otherwise the shipped thresholds select creep/walk/run. The full
    //    speed is the recovered `GetStanceSpeed` product
    //    (`0x01501900`: actor_scale * stance_slot * vtable_scale).
    let stance = gait_stance(drive, input.crouch, input.sprint, t);
    state.stance_index = stance;
    let speed = get_stance_speed(
        STANCE_SPEED_ACTOR_SCALE,
        stance_speed(stance, t),
        STANCE_SPEED_VTABLE_SCALE,
    ) * drive;

    // 3. movement delta angles from desired-vs-current basis, clamped to
    //    max_delta. Mouse deltas are added *after* the clamp below
    //    (`0x01619ca0` line 1197), so they are not part of this clamp.
    let max_d = max_delta(dt, constrained);
    // Rotate the desired world direction into the actor basis. In this planar
    // Z-up port the vertical component is 0, so the recovered
    // `asin(clamp(z,-1,1))` pitch term is 0; the yaw term is the planar
    // equivalent of `atan2(-a, b)` (`0x01619ca0` lines 1079–1082):
    // `local_x = D·forward = cos(t-y)`, `local_y = D·right = sin(t-y)`, so
    // `atan2(local_y, local_x) = t - y`.
    let local_x = dir_x * basis[0][0] + dir_y * basis[0][1];
    let local_y = dir_x * basis[1][0] + dir_y * basis[1][1];
    let local_z = 0.0f32;
    let mut pitch_delta = sane(local_z.clamp(-1.0, 1.0).asin());
    let mut yaw_delta = if dir_x != 0.0 || dir_y != 0.0 {
        wrap_angle(local_y.atan2(local_x))
    } else {
        0.0
    };
    pitch_delta = pitch_delta.clamp(-max_d, max_d);
    yaw_delta = yaw_delta.clamp(-max_d, max_d);
    state.delta_angles = [pitch_delta, yaw_delta];

    // 4. rotation: slerp up toward the ground normal, then run the selected
    //    path (`CPlayerRotation::Process` `0x0161eae0`):
    //      * `actor+0x16c` set / `constrained` -> `ProcessConstrainedToMarkup`
    //        (`0x0161f8a0`), the `π*dt` markup-constrained rotation;
    //      * otherwise `ProcessNormal` (`0x01621070`), the free `20π*dt` path,
    //        committed through the half-angle Euler→quaternion.
    let gain = up_slerp_gain(dt);
    let ground_up = class3_physics::ground_normal(state.pos[0], state.pos[1], ground);
    state.up = slerp_up(state.up, ground_up, gain);
    if constrained {
        process_constrained_to_markup(state, [dir_x, dir_y, 0.0], dt);
    } else {
        let committed = normalize_quat(euler_to_quat([0.0, 0.0, state.yaw + yaw_delta]));
        state.yaw = wrap_angle(quat_yaw(committed));
    }
    // `controller+0x10 & 0x40` "mouse" path: added *after* the clamp (and after
    // the movement rotation commit), so a mouse look can exceed `max_delta`.
    state.delta_angles[0] += sane(input.mouse_dy);
    state.delta_angles[1] += sane(input.mouse_dx);
    state.yaw = wrap_angle(state.yaw + sane(input.mouse_dx));

    // 5. desired horizontal velocity, inertia-smoothed, then integrated.
    //    Zero movement input (`drive == 0`) must yield an *exactly* zero
    //    desired velocity: mouse-look deltas rotate the body but never
    //    contribute speed, and the desired direction is already zero above.
    let desired = if drive > 0.0 {
        [dir_x * speed, dir_y * speed, 0.0]
    } else {
        [0.0, 0.0, 0.0]
    };
    let current = [sane(state.vel[0]), sane(state.vel[1]), 0.0];
    let mut vel = accelerate_velocity(current, desired, t, dt);
    // Bleed the very last of the inertia: once the residual horizontal speed
    // is negligible on an idle frame, snap it to zero so the player cannot
    // creep. (While input is held the inertia ramp is untouched.)
    if drive == 0.0 && (vel[0] * vel[0] + vel[1] * vel[1]).sqrt() < ZERO_INPUT_SPEED_EPS {
        vel[0] = 0.0;
        vel[1] = 0.0;
    }
    let (vx, vy) = (vel[0], vel[1]);

    // 6. integrate horizontal motion, then gravity + ground via `physics`.
    //    Use the recovered `Living` gravity (9.81, `player.lua`), not the
    //    generic character-controller default.
    let mut cc = CharacterController::new(CAPSULE_RADIUS, CAPSULE_HEIGHT);
    cc.gravity = -class3_physics::GRAVITY;
    cc.velocity[2] = sane(state.vel[2]);
    let new_pos = cc.move_and_slide(state.pos, [vx * dt, vy * dt], dt, ground, colliders);
    state.pos = [sane(new_pos[0]), sane(new_pos[1]), sane(new_pos[2])];
    state.vel = [vx, vy, sane(cc.velocity[2])];
    state.grounded = cc.grounded;

    // 7. Positioning invariant: when grounded, the feet rest exactly on the
    //    surface. `move_and_slide` clamps the terrain; re-probe within one step
    //    so collider tops and sub-step rounding also land exactly. This is what
    //    keeps `MoveState.pos` the CDF's feet origin (feet at local z = 0).
    if state.grounded {
        if let Some(hit) = class3_physics::probe_ground(
            state.pos[0],
            state.pos[1],
            state.pos[2] + 1e-3,
            GROUND_SNAP_REACH,
            ground,
            colliders,
        ) {
            state.pos[2] = sane(hit.height);
        }
    }

    StepOut {
        desired_speed: speed,
        desired_vel: desired,
        stance,
        drive,
    }
}

/// Advance the player one frame along the recovered order.
///
/// 1. desired world direction = input rotated by `state.yaw` through the
///    orthonormal basis from [`basis_from_yaw`];
/// 2. `speed = get_stance_speed(actor_scale, stance_speed(stance, t), vtable) *
///    |dir|` (`GetStanceSpeed` `0x01501900`; `player.lua` stance table), with
///    the gait chosen by [`gait_stance`];
/// 3. movement delta angles from desired-vs-current basis via `asin`/`atan2`,
///    clamped to `±`[`max_delta`], **then** the mouse deltas are added
///    (`0x01619ca0` lines 1079–1200, `movement_input.md` §5);
/// 4. rotate: slerp `state.up` toward the ground normal with
///    [`up_slerp_gain`], then run [`process_constrained_to_markup`]
///    (`0x0161f8a0`) on the constrained path, or build the Euler→quaternion
///    with half-angles and commit the normalized yaw ([`euler_to_quat`] +
///    [`quat_yaw`]) on the free `ProcessNormal` path;
/// 5. accelerate `state.vel` toward the desired horizontal velocity with
///    [`integrate_inertia`] / [`accelerate_velocity`] (`inertia`/`inertiaAccel`),
///    integrate it, then gravity + ground through
///    [`CharacterController::move_and_slide`], setting `state.grounded` and
///    snapping the feet onto the surface.
///
/// This free function pins the **constrained** (movement-matching/animation)
/// path — the `π*dt` clamp and the markup rotation — used by the player
/// pipeline; use [`CPlayerMovementController`] to select either path, and
/// [`step`] to get the desired-speed/velocity outputs.
pub fn update_normal(
    state: &mut MoveState,
    input: &MoveInput,
    t: &MovementTuning,
    ground: &dyn Fn(f32, f32) -> f32,
    colliders: &[Aabb],
) {
    let _ = step(state, input, t, true, ground, colliders);
}

/// Snap `state.pos` (the feet) onto the nearest ground surface within `reach`.
///
/// Returns `true` and clears vertical velocity / sets `grounded` on a hit; a
/// non-finite position or no surface within reach leaves the state untouched
/// and returns `false`. See [`class3_physics::snap_to_ground`].
pub fn snap_to_ground(
    state: &mut MoveState,
    reach: f32,
    ground: &dyn Fn(f32, f32) -> f32,
    colliders: &[Aabb],
) -> bool {
    match class3_physics::snap_to_ground(&mut state.pos, reach, ground, colliders) {
        Some(_) => {
            state.vel[2] = 0.0;
            state.grounded = true;
            true
        }
        None => false,
    }
}

/// Resolve a spawn/teleport position so the feet land on the ground.
///
/// The assembled CDF has its feet at local `z = 0`, so returning the snapped
/// feet position is exactly the root translation to place the character with.
/// Uses [`SPAWN_SNAP_REACH`] and leaves the position unchanged when no surface
/// is found. Total for any input.
pub fn spawn_on_ground(
    pos: [f32; 3],
    ground: &dyn Fn(f32, f32) -> f32,
    colliders: &[Aabb],
) -> [f32; 3] {
    let mut p = [sane(pos[0]), sane(pos[1]), sane(pos[2])];
    let _ = class3_physics::snap_to_ground(&mut p, SPAWN_SNAP_REACH, ground, colliders);
    p
}

/// 1:1 player movement controller (`CPlayerMovementController`) + rotation
/// (`CPlayerRotation`) state.
///
/// Owns the [`MoveState`] and the shipped [`MovementTuning`] so the animation
/// layer can read [`pseudo_speed`](Self::pseudo_speed) and the collision layer
/// can read [`capsule`](Self::capsule) without re-deriving either.
#[derive(Clone, Debug, PartialEq)]
pub struct CPlayerMovementController {
    /// Current per-frame state (`pos` is the capsule feet, Z-up).
    pub state: MoveState,
    /// The shipped `player.lua` movement tuning.
    pub tuning: MovementTuning,
    /// `true` selects the movement-matching / animation-constrained pipeline:
    /// the rotation path is `CPlayerRotation::ProcessConstrainedToMarkup`
    /// (`0x0161f8a0`, `actor+0x16c` set) and the per-frame delta clamp is `π*dt`
    /// (`DAT_01914de8`). `false` selects the free `ProcessNormal` path
    /// (`0x01621070`) with the default `20π*dt` clamp (`DAT_01914df0`).
    /// Defaults to `true`, matching the player pipeline.
    pub constrained: bool,
    /// Last `stance+0x80 * |dir|` desired speed (`controller+0x2c8`).
    pub desired_speed: f32,
    /// Last `param_3[0..2]` desired world velocity.
    pub desired_vel: [f32; 3],
}

impl CPlayerMovementController {
    /// New controller at `pos` (feet) with the given tuning.
    pub fn new(pos: [f32; 3], tuning: MovementTuning) -> Self {
        CPlayerMovementController {
            state: MoveState::new(pos),
            tuning,
            constrained: true,
            desired_speed: 0.0,
            desired_vel: [0.0; 3],
        }
    }

    /// New controller wrapping an existing [`MoveState`].
    pub fn with_state(state: MoveState, tuning: MovementTuning) -> Self {
        CPlayerMovementController {
            state,
            tuning,
            constrained: true,
            desired_speed: 0.0,
            desired_vel: [0.0; 3],
        }
    }

    /// Run one `UpdateNormal` frame and cache its desired speed/velocity.
    pub fn update_normal(
        &mut self,
        input: &MoveInput,
        ground: &dyn Fn(f32, f32) -> f32,
        colliders: &[Aabb],
    ) -> StepOut {
        let out = step(
            &mut self.state,
            input,
            &self.tuning,
            self.constrained,
            ground,
            colliders,
        );
        self.desired_speed = out.desired_speed;
        self.desired_vel = out.desired_vel;
        out
    }

    /// The normalised locomotion band written to the animation graph input
    /// `PseudoSpeed` (`controller+0x453`, `0x01619ca0`; `animation.md` §1.1).
    ///
    /// `|horizontal velocity| / run_speed`, so run is `1.0`. The shipped bands
    /// (`playerfullbody.xml`) are idle `[0, 0.2]`, walk
    /// `select_speed_slow=0.2 … fast=0.8`, run `[0.6, 2.0]` and sprint
    /// `[1.5, 3.5]`; see [`pseudo_speed_from_speed`]. Finite for any state.
    pub fn pseudo_speed(&self) -> f32 {
        let vx = sane(self.state.vel[0]);
        let vy = sane(self.state.vel[1]);
        let speed = (vx * vx + vy * vy).sqrt();
        pseudo_speed_from_speed(speed, self.tuning.run)
    }

    /// Last frame's desired world velocity (`param_3[0..2]`,
    /// `controller+0x2c8`-driven) as cached by
    /// [`update_normal`](Self::update_normal). Zero when idle. Finite for any
    /// state (the controller is total).
    pub fn desired_velocity(&self) -> [f32; 3] {
        [
            sane(self.desired_vel[0]),
            sane(self.desired_vel[1]),
            sane(self.desired_vel[2]),
        ]
    }

    /// The per-frame delta-angle limit selected by [`constrained`](Self::constrained):
    /// `π*dt` when constrained (`DAT_01914de8`), else `20π*dt`
    /// (`DAT_01914df0`). This is the controller-level clamp choice
    /// (`0x01619ca0` lines 1114–1164). A non-positive/non-finite `dt` yields
    /// `0.0`.
    pub fn max_delta(&self, dt: f32) -> f32 {
        max_delta(dt, self.constrained)
    }

    /// The selected delta-angle base rate (`π` constrained, `20π` default).
    pub fn clamp_rate(&self) -> f32 {
        if self.constrained {
            MAX_DELTA_RATE_CONSTRAINED
        } else {
            MAX_DELTA_RATE_DEFAULT
        }
    }

    /// The current stance's full `GetStanceSpeed` product
    /// (`0x01501900`): actor scale × stance slot × vtable scale.
    pub fn stance_speed(&self) -> f32 {
        get_stance_speed(
            STANCE_SPEED_ACTOR_SCALE,
            stance_speed(self.state.stance_index, &self.tuning),
            STANCE_SPEED_VTABLE_SCALE,
        )
    }

    /// The collision capsule as `(feet, height)`.
    ///
    /// `feet` is [`MoveState::pos`] (world Z-up, snapped to the ground); the
    /// capsule solid spans `feet.z ..= feet.z + height`. The radius is
    /// [`CAPSULE_RADIUS`] and is returned by
    /// [`capsule_radius`](Self::capsule_radius). This is the transform used to
    /// place the assembled CDF, whose feet are at local `z = 0`.
    pub fn capsule(&self) -> ([f32; 3], f32) {
        (self.state.pos, CAPSULE_HEIGHT)
    }

    /// Capsule radius (m).
    pub fn capsule_radius(&self) -> f32 {
        CAPSULE_RADIUS
    }

    /// Snap the feet onto the ground within `reach` (see [`snap_to_ground`]).
    pub fn snap_to_ground(
        &mut self,
        reach: f32,
        ground: &dyn Fn(f32, f32) -> f32,
        colliders: &[Aabb],
    ) -> bool {
        snap_to_ground(&mut self.state, reach, ground, colliders)
    }

    /// Cast the ground probe straight down from the current feet.
    pub fn probe_ground(
        &self,
        reach: f32,
        ground: &dyn Fn(f32, f32) -> f32,
        colliders: &[Aabb],
    ) -> Option<class3_physics::GroundHit> {
        class3_physics::probe_ground(
            self.state.pos[0],
            self.state.pos[1],
            self.state.pos[2],
            reach,
            ground,
            colliders,
        )
    }

    /// Apply a jump launch (`0x0156a560` line 566) and leave the ground.
    ///
    /// Returns the launch velocity written to `state.vel[2]`. The vertical arc
    /// is then integrated by [`update_normal`](Self::update_normal) under the
    /// recovered gravity.
    pub fn jump(&mut self, height: f32) -> f32 {
        let vy = jump(self.state.vel[2], height, class3_physics::GRAVITY);
        self.state.vel[2] = vy;
        self.state.grounded = false;
        vy
    }

    /// Integrate one vertical jump/fall step (see [`integrate_vertical`]).
    pub fn integrate_vertical(&mut self, ground_h: f32, dt: f32) {
        integrate_vertical(&mut self.state, ground_h, dt);
    }
}


// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

/// Map a non-finite value to `0.0`.
#[inline]
fn sane(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Normalise a vector, falling back to `fallback` when degenerate/non-finite.
#[inline]
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len.is_finite() && len > 1e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        fallback
    }
}

/// Spherical interpolation between two direction vectors.
///
/// Mirrors the up-vector interpolation in `ProcessNormal` (`0x01621070`,
/// `alpha = min(dt*5,1)` then interpolate `actor+0xcc` into `this[0x1d..0x1f]`);
/// the exact helper `FUN_004a1d30` was not line-reversed, so a standard slerp
/// with an nlerp fallback for near-parallel vectors is used. // [UNRECOVERED]
fn slerp_up(from: [f32; 3], to: [f32; 3], alpha: f32) -> [f32; 3] {
    let a = normalize_or(from, [0.0, 0.0, 1.0]);
    let b = normalize_or(to, [0.0, 0.0, 1.0]);
    let alpha = if alpha.is_finite() { alpha.clamp(0.0, 1.0) } else { 0.0 };
    if alpha <= 0.0 {
        return a;
    }
    if alpha >= 1.0 {
        return b;
    }
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    // Near-parallel: straight nlerp is stable and effectively identical.
    if dot > 0.9995 {
        let l = [
            a[0] + (b[0] - a[0]) * alpha,
            a[1] + (b[1] - a[1]) * alpha,
            a[2] + (b[2] - a[2]) * alpha,
        ];
        return normalize_or(l, a);
    }
    if dot < -0.9995 {
        // Antiparallel: no unique arc; keep the starting vector.
        return a;
    }
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - alpha) * theta).sin() / sin_theta;
    let wb = (alpha * theta).sin() / sin_theta;
    normalize_or(
        [a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb, a[2] * wa + b[2] * wb],
        a,
    )
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::default_movement;

    fn flat(z: f32) -> impl Fn(f32, f32) -> f32 {
        move |_, _| z
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn input(forward: f32, strafe: f32, dt: f32) -> MoveInput {
        MoveInput {
            forward,
            strafe,
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            crouch: false,
            sprint: false,
            dt,
        }
    }

    /// Rotate a 2D vector by `a` about Z (radians), matching
    /// `Quat::from_rotation_z` / [`basis_from_yaw`].
    fn rotate_z(v: [f32; 2], a: f32) -> [f32; 2] {
        let (s, c) = a.sin_cos();
        [v[0] * c - v[1] * s, v[0] * s + v[1] * c]
    }

    #[test]
    fn stance_mapping_is_shipped() {
        let t = default_movement();
        assert!(approx(stance_speed(STANCE_CREEP, &t), 0.8));
        assert!(approx(stance_speed(STANCE_WALK, &t), 1.66));
        assert!(approx(stance_speed(STANCE_STAND, &t), 4.0));
        assert!(approx(stance_speed(STANCE_SPRINT, &t), 6.8));
        // Unknown selectors fall back to the stand/run speed and stay finite.
        assert!(approx(stance_speed(200, &t), 4.0));
        let bad = MovementTuning { run: f32::NAN, ..t };
        assert_eq!(stance_speed(STANCE_STAND, &bad), 0.0);
    }

    /// The six per-stance speed slots at `stance+0x84..0x98`
    /// (`movement_input.md` §4). `player.lua` ships the three sprint slots equal.
    #[test]
    fn six_speed_stance_table_is_shipped() {
        let t = default_movement();
        let table = stance_speeds(&t);
        assert!(approx(table[SPEED_SLOT_CREEP], 0.8));
        assert!(approx(table[SPEED_SLOT_WALK], 1.66));
        assert!(approx(table[SPEED_SLOT_RUN], 4.0));
        assert!(approx(table[SPEED_SLOT_SPRINT_CREEP], 6.8));
        assert!(approx(table[SPEED_SLOT_SPRINT_WALK], 6.8));
        assert!(approx(table[SPEED_SLOT_SPRINT], 6.8));
        assert!(approx(stance_speed_at(SPEED_SLOT_CREEP, &t), 0.8));
        assert!(approx(stance_speed_at(SPEED_SLOT_RUN, &t), 4.0));
        // Out-of-range slots clamp to the last (sprint) slot, never panic.
        assert!(approx(stance_speed_at(99, &t), 6.8));
    }

    #[test]
    fn clamp_helpers_match_recovered_formulas() {
        // Constrained path: DAT_01914de8 = π.
        assert!(approx(max_delta(0.5, true), PI * 0.5));
        // Default path: DAT_01914df0 = 20π.
        assert!(approx(max_delta(0.5, false), 20.0 * PI * 0.5));
        assert_eq!(max_delta(0.0, true), 0.0);
        assert_eq!(max_delta(f32::NAN, false), 0.0);
        assert_eq!(max_delta(-1.0, false), 0.0);

        assert_eq!(up_slerp_gain(0.0), 0.0);
        assert_eq!(up_slerp_gain(1.0), 1.0);
        assert!(approx(up_slerp_gain(0.1), 0.5));
        assert_eq!(up_slerp_gain(0.5), 1.0);
        assert_eq!(up_slerp_gain(f32::NAN), 0.0);
        assert_eq!(up_slerp_gain(-2.0), 0.0);

        assert!(approx(wrap_angle(0.0), 0.0));
        assert!(approx(wrap_angle(PI * 3.0), PI));
        assert!(wrap_angle(-PI * 3.0).abs() <= PI);
        assert_eq!(wrap_angle(f32::NAN), 0.0);
    }

    /// `Process` builds the Euler→quaternion with `*0.5` half-angles and
    /// `Commit` normalizes; a pure yaw must round-trip through both.
    #[test]
    fn half_angle_quat_commit_round_trips_yaw() {
        for &yaw in &[0.0f32, 0.5, -1.25, PI / 2.0, 3.0] {
            let q = euler_to_quat([0.0, 0.0, yaw]);
            // Pure yaw: z = sin(y/2), w = cos(y/2) (DAT_017877c4 = 0.5).
            assert!(approx(q[2], (yaw * 0.5).sin()), "z {q:?}");
            assert!(approx(q[3], (yaw * 0.5).cos()), "w {q:?}");
            let n = normalize_quat(q);
            assert!(approx(quat_yaw(n), wrap_angle(yaw)), "yaw {yaw}");
        }
        // Degenerate quaternion normalizes to identity and stays finite.
        assert_eq!(normalize_quat([0.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0]);
        assert!(normalize_quat([f32::NAN; 4]).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn straight_line_travel_is_speed_times_dt() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let speed = stance_speed(STANCE_STAND, &t); // 4.0 m/s
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        // Start already at the target velocity so this measures pure
        // integration (the inertia ramp is covered separately below).
        s.vel = [speed, 0.0, 0.0];
        let i = input(1.0, 0.0, dt);

        update_normal(&mut s, &i, &t, &g, &[]);
        assert!(approx(s.pos[0], speed * dt), "x={}", s.pos[0]);
        assert!(approx(s.pos[1], 0.0));
        assert!(approx(s.pos[2], 0.0));
        assert!(s.grounded);
        assert!(approx(s.yaw, 0.0));

        update_normal(&mut s, &i, &t, &g, &[]);
        assert!(approx(s.pos[0], 2.0 * speed * dt));
        assert!(approx(s.vel[0], speed));
    }

    #[test]
    fn ninety_degree_turn_clamped_over_expected_ticks() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        // Constrained rate PI*dt = 180 deg/s -> 3 deg/tick.
        let md = max_delta(dt, true);
        let target = PI / 2.0;
        // 90 deg at 180 deg/s = 0.5 s = 30 ticks at 60 Hz.
        let expected_ticks = 30usize;

        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        let mut ticks = 0usize;
        for _ in 0..1000 {
            let rem = wrap_angle(target - s.yaw);
            if rem.abs() < 1e-5 {
                break;
            }
            // Feed a fixed world target: local = R(-yaw) * (0,1) = (sin, cos).
            let (sn, cs) = s.yaw.sin_cos();
            let i = input(sn, cs, dt);
            let before = s.yaw;
            update_normal(&mut s, &i, &t, &g, &[]);
            let step = wrap_angle(s.yaw - before);
            assert!(step <= md + 1e-5, "step {step} exceeds clamp {md}");
            ticks += 1;
        }
        assert_eq!(ticks, expected_ticks, "yaw={}", s.yaw);
        assert!(approx(s.yaw, target), "yaw={}", s.yaw);
    }

    /// Mouse deltas are added *after* the movement-derived clamp
    /// (`0x01619ca0` line 1197): a huge mouse yaw must pass through unclamped.
    #[test]
    fn mouse_deltas_are_added_after_clamp() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        let mut i = input(0.0, 0.0, dt);
        i.mouse_dx = 0.4; // far above max_delta(dt, true) = π/60
        i.mouse_dy = -0.3;
        update_normal(&mut s, &i, &t, &g, &[]);
        assert!(approx(s.delta_angles[1], 0.4));
        assert!(approx(s.delta_angles[0], -0.3));
        assert!(approx(s.yaw, 0.4));
    }

    /// The measured first-order velocity lag (`velocity_tau_*`, τ≈0.2 s) drives
    /// both acceleration and deceleration; `integrate_inertia` is time-constant
    /// based (the shipped `inertia`/`inertiaAccel` are provenance only).
    #[test]
    fn acceleration_reaches_target_velocity() {
        let t = default_movement();
        let dt = 1.0 / 60.0;
        // One frame is not enough: the model ramps, it does not snap.
        let v1 = accelerate_velocity([0.0; 3], [4.0, 0.0, 0.0], &t, dt);
        assert!(v1[0] > 0.0 && v1[0] < 4.0, "v1={}", v1[0]);

        // Converges to the target within the model.
        let mut v = [0.0f32; 3];
        let target = [4.0f32, 0.0, 0.0];
        for _ in 0..240 {
            v = accelerate_velocity(v, target, &t, dt);
        }
        assert!((v[0] - 4.0).abs() < 1e-3, "v={:?}", v);

        // Deceleration also converges.
        let mut v = [4.0f32, 0.0, 0.0];
        for _ in 0..240 {
            v = accelerate_velocity(v, [0.0; 3], &t, dt);
        }
        assert!(v[0].abs() < 1e-3, "v={:?}", v);

        // A shorter accel tau is faster on the first frame than a longer decel
        // tau; the shipped/measured default taus (both 0.2) are symmetric.
        let fast = MovementTuning {
            velocity_tau_accel: 0.05,
            velocity_tau_decel: 0.4,
            ..t
        };
        let accel = integrate_inertia(
            [0.0; 3], [4.0, 0.0, 0.0], fast.velocity_tau_accel, fast.velocity_tau_decel, dt,
        )[0];
        let decel = 4.0
            - integrate_inertia(
                [4.0, 0.0, 0.0], [0.0; 3], fast.velocity_tau_accel, fast.velocity_tau_decel, dt,
            )[0];
        assert!(accel > decel, "accel {accel} vs decel {decel}");
        let sym_a = accelerate_velocity([0.0; 3], [4.0, 0.0, 0.0], &t, dt)[0];
        let sym_d = 4.0 - accelerate_velocity([4.0, 0.0, 0.0], [0.0; 3], &t, dt)[0];
        assert!(approx(sym_a, sym_d), "symmetric default {sym_a} vs {sym_d}");

        // Total: NaN/zero-dt leave the velocity unchanged; tau <= 0 snaps.
        assert_eq!(accelerate_velocity([1.0, 2.0, 3.0], [4.0; 3], &t, 0.0), [1.0, 2.0, 3.0]);
        let bad = accelerate_velocity([f32::NAN; 3], [f32::INFINITY; 3], &t, dt);
        assert!(bad.iter().all(|v| v.is_finite()));
        assert_eq!(integrate_inertia([0.0; 3], [4.0; 3], 0.0, 0.0, dt), [4.0; 3]);
    }

    /// The measured run ramp: from rest at the run gait, ~3.2 m in the first
    /// second and ~0.8 m of slide after release (`notes/movement_telemetry.md`).
    #[test]
    fn velocity_lag_matches_measured_run_ramp() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;

        // Accelerate from rest for 1 s: V*(t - tau*(1-exp(-t/tau))) = 3.21 m.
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        let i = input(1.0, 0.0, dt);
        for _ in 0..60 {
            update_normal(&mut s, &i, &t, &g, &[]);
        }
        assert!((s.pos[0] - 3.21).abs() < 0.15, "1s distance = {}", s.pos[0]);

        // Run to the plateau, then release; the slide is V*tau = 0.8 m.
        for _ in 0..240 {
            update_normal(&mut s, &i, &t, &g, &[]);
        }
        assert!(approx(s.vel[0], 4.0), "plateau {}", s.vel[0]);
        let stop_start = s.pos[0];
        let idle = input(0.0, 0.0, dt);
        for _ in 0..240 {
            update_normal(&mut s, &idle, &t, &g, &[]);
        }
        let slide = s.pos[0] - stop_start;
        assert!((slide - 0.8).abs() < 0.15, "stop slide = {slide}");
    }

    /// The whole controller accelerates from rest toward the stance speed.
    #[test]
    fn update_normal_accelerates_from_rest() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let target = stance_speed(STANCE_STAND, &t);
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        let i = input(1.0, 0.0, dt);
        update_normal(&mut s, &i, &t, &g, &[]);
        assert!(s.vel[0] > 0.0 && s.vel[0] < target, "vel={}", s.vel[0]);
        for _ in 0..240 {
            update_normal(&mut s, &i, &t, &g, &[]);
        }
        assert!(approx(s.vel[0], target), "vel={}", s.vel[0]);
        assert!(s.pos.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn jump_returns_to_ground() {
        let t = default_movement();
        let height = t.jump_height; // 1.0 m (basicactor.lua)
        let gravity = class3_physics::GRAVITY; // 9.81
        // Recovered launch: sqrt(2*g*h) (0156a560 line 566, folded).
        let launch = jump(0.0, height, gravity);
        let expected = (2.0 * gravity * height).sqrt();
        assert!((launch - expected).abs() < 1e-4, "launch {launch}");

        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        s.vel[2] = launch;
        s.grounded = false;
        let mut peak = 0.0f32;
        for _ in 0..240 {
            integrate_vertical(&mut s, 0.0, 1.0 / 60.0);
            peak = peak.max(s.pos[2]);
        }
        assert!(s.grounded, "should have landed");
        assert!(s.pos[2].abs() < 1e-4, "z={}", s.pos[2]);
        assert!(s.vel[2].abs() < 1e-4, "vz={}", s.vel[2]);
        assert!(peak > 0.9, "peak {peak}");

        // Degenerate launch height leaves the current velocity untouched.
        assert_eq!(jump(1.5, 0.0, gravity), 1.5);
        assert!(jump(0.0, 1.0, 0.0).is_finite());
    }

    #[test]
    fn zero_input_stays_still_and_grounded() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        let i = input(0.0, 0.0, dt);
        update_normal(&mut s, &i, &t, &g, &[]);
        assert!(approx(s.pos[0], 0.0) && approx(s.pos[1], 0.0) && approx(s.pos[2], 0.0));
        assert_eq!(s.vel, [0.0, 0.0, 0.0]);
        assert!(s.grounded);
    }

    /// The CryEngine model's local forward is `+Y`, but this controller's yaw
    /// convention is `0 -> +X`. [`render_yaw`] rotates the model transform by
    /// `-π/2` so the authored `+Y` forward renders along the actor's `+X`.
    #[test]
    fn render_yaw_maps_cry_plus_y_forward_to_actor_plus_x() {
        // `render_yaw(0) == -PI/2` and it maps model +Y onto +X.
        assert!(
            approx(render_yaw(0.0), -PI / 2.0),
            "render_yaw(0)={}",
            render_yaw(0.0)
        );
        assert!(approx(MODEL_FORWARD_OFFSET, -PI / 2.0));
        let model_forward = [0.0f32, 1.0];
        let mapped = rotate_z(model_forward, render_yaw(0.0));
        assert!(
            approx(mapped[0], 1.0) && approx(mapped[1], 0.0),
            "mapped={mapped:?}"
        );

        // For any actor yaw the mapped model forward equals basis_from_yaw's
        // forward row — the sign of the offset is verified here.
        for &yaw in &[0.0f32, 0.3, -1.2, PI / 2.0, 2.7, -2.9] {
            let basis = basis_from_yaw(yaw);
            let m = rotate_z(model_forward, render_yaw(yaw));
            assert!(
                approx(m[0], basis[0][0]) && approx(m[1], basis[0][1]),
                "yaw {yaw}: {m:?} vs forward {:?}",
                [basis[0][0], basis[0][1]]
            );
        }
    }

    /// Zero input must be *exactly* idle: `desired_speed == 0`,
    /// `desired_vel == [0,0,0]`, and no horizontal movement over a second of
    /// 60 Hz ticks. Mouse-look deltas must not leak into speed.
    #[test]
    fn zero_input_produces_no_movement_over_60_ticks() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;

        // First frame: exact zero desired speed/velocity, even with mouse look.
        let mut s = MoveState::new([1.0, -2.0, 0.0]);
        let mut i = input(0.0, 0.0, dt);
        i.mouse_dx = 0.5;
        i.mouse_dy = -0.25;
        let out = step(&mut s, &i, &t, true, &g, &[]);
        assert_eq!(out.desired_speed, 0.0);
        assert_eq!(out.desired_vel, [0.0, 0.0, 0.0]);
        assert_eq!(out.drive, 0.0);

        // 60 ticks at 60 Hz with no movement input: no horizontal drift.
        let start = [1.0f32, -2.0, 0.0];
        let mut s = MoveState::new(start);
        let i = input(0.0, 0.0, dt);
        for _ in 0..60 {
            let out = step(&mut s, &i, &t, true, &g, &[]);
            assert_eq!(out.desired_speed, 0.0);
            assert_eq!(out.desired_vel, [0.0, 0.0, 0.0]);
        }
        let dx = s.pos[0] - start[0];
        let dy = s.pos[1] - start[1];
        assert!(dx.abs() < 1e-3 && dy.abs() < 1e-3, "drift=({dx}, {dy})");
        let hs = (s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1]).sqrt();
        assert!(hs < 1e-3, "horizontal vel={hs}");
        assert!(s.pos.iter().all(|v| v.is_finite()));

        // Residual inertia is bled off: after a second of idle ticks the
        // horizontal velocity is essentially zero.
        let mut s = MoveState::new(start);
        s.vel = [t.run, 0.0, 0.0];
        for _ in 0..120 {
            step(&mut s, &i, &t, true, &g, &[]);
        }
        let hs = (s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1]).sqrt();
        assert!(hs < 1e-3, "residual horizontal vel={hs}");
    }

    /// Forward input (re-expressed in the actor frame each tick, as the app
    /// does) turns the actor to face `+X` and moves it along `+X`.
    #[test]
    fn forward_input_moves_along_plus_x_and_faces_plus_x() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;

        // Start facing +Y; target world direction is +X.
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        s.yaw = PI / 2.0;
        for _ in 0..240 {
            let (sn, cs) = s.yaw.sin_cos();
            // local of world +X = (forward·X, right·X) = (cos yaw, -sin yaw).
            update_normal(&mut s, &input(cs, -sn, dt), &t, &g, &[]);
        }
        assert!(approx(s.yaw, 0.0), "yaw={}", s.yaw);
        assert!(s.pos[0] > 1.0, "x={}", s.pos[0]);
        assert!(s.pos[1].abs() < 1e-2, "y={}", s.pos[1]);

        // The rendered model now faces +X: model +Y rotated by render_yaw(0).
        let mapped = rotate_z([0.0, 1.0], render_yaw(s.yaw));
        assert!(
            approx(mapped[0], 1.0) && approx(mapped[1], 0.0),
            "model forward={mapped:?}"
        );
    }

    #[test]
    fn crouch_and_sprint_use_their_stance_speeds() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;

        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        s.vel = [t.creep, 0.0, 0.0];
        let mut i = input(1.0, 0.0, dt);
        i.crouch = true;
        update_normal(&mut s, &i, &t, &g, &[]);
        assert!(approx(s.pos[0], t.creep * dt));
        assert_eq!(s.stance_index, STANCE_CREEP);

        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        s.vel = [t.sprint, 0.0, 0.0];
        let mut i = input(1.0, 0.0, dt);
        i.sprint = true;
        update_normal(&mut s, &i, &t, &g, &[]);
        assert!(approx(s.pos[0], t.sprint * dt));
        assert_eq!(s.stance_index, STANCE_SPRINT);
    }

    #[test]
    fn degenerate_input_is_total_and_finite() {
        let t = default_movement();
        let g = flat(0.0);
        let mut s = MoveState::new([f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        let i = MoveInput {
            forward: f32::NAN,
            strafe: f32::INFINITY,
            mouse_dx: f32::NAN,
            mouse_dy: f32::INFINITY,
            crouch: false,
            sprint: false,
            dt: f32::NAN,
        };
        update_normal(&mut s, &i, &t, &g, &[]);
        assert!(s.pos.iter().all(|v| v.is_finite()), "pos={:?}", s.pos);
        assert!(s.vel.iter().all(|v| v.is_finite()), "vel={:?}", s.vel);
        assert!(s.up.iter().all(|v| v.is_finite()), "up={:?}", s.up);
        assert!(s.yaw.is_finite());
        assert!(s.delta_angles.iter().all(|v| v.is_finite()));
        assert!(stance_speed(s.stance_index, &t).is_finite());
        assert!(max_delta(s.yaw, true).is_finite());
        assert!(max_delta(s.yaw, false).is_finite());
    }

    #[test]
    fn walls_block_without_panic() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let wall = Aabb {
            min: [2.0, -5.0, 0.0],
            max: [3.0, 5.0, 3.0],
        };
        let mut s = MoveState::new([1.0, 0.0, 0.0]);
        let i = input(1.0, 0.0, dt);
        for _ in 0..240 {
            update_normal(&mut s, &i, &t, &g, &[wall]);
        }
        assert!(s.pos[0] < 2.0, "passed through wall: {}", s.pos[0]);
        assert!(s.pos.iter().all(|v| v.is_finite()));
        assert!(s.grounded);
    }

    // -----------------------------------------------------------------------
    // controller / positioning tests pinned to the recovered addresses
    // -----------------------------------------------------------------------

    /// `GetStanceSpeed` (`actor->vtable+0x344`, not exported) selects the gait
    /// from the `player.lua` controller thresholds `0.29`/`0.69`
    /// (`movement_input.md` §4.1). The explicit crouch/sprint overrides win.
    #[test]
    fn gait_thresholds_select_creep_walk_run() {
        let t = default_movement();
        assert_eq!(gait_stance(0.0, false, false, &t), STANCE_STAND); // idle
        assert_eq!(gait_stance(0.1, false, false, &t), STANCE_CREEP);
        assert_eq!(gait_stance(0.29, false, false, &t), STANCE_WALK); // boundary
        assert_eq!(gait_stance(0.5, false, false, &t), STANCE_WALK);
        assert_eq!(gait_stance(0.69, false, false, &t), STANCE_STAND); // boundary
        assert_eq!(gait_stance(1.0, false, false, &t), STANCE_STAND);
        assert_eq!(gait_stance(0.5, true, false, &t), STANCE_CREEP);
        assert_eq!(gait_stance(0.5, false, true, &t), STANCE_SPRINT);
        // Total for non-finite input.
        assert_eq!(gait_stance(f32::NAN, false, false, &t), STANCE_STAND);

        // Through the free controller: a half-stick selects the walk gait.
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        update_normal(&mut s, &input(0.5, 0.0, dt), &t, &g, &[]);
        assert_eq!(s.stance_index, STANCE_WALK);
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        update_normal(&mut s, &input(1.0, 0.0, dt), &t, &g, &[]);
        assert_eq!(s.stance_index, STANCE_STAND);
    }

    /// The `PseudoSpeed` graph input (`controller+0x453`, `0x01619ca0`) is the
    /// normalised locomotion band: `|v| / runSpeed` with the shipped
    /// `playerfullbody.xml` bands idle `[0,0.2]`, walk `0.2…0.8`, run `[0.6,2]`,
    /// sprint `[1.5,3.5]`.
    #[test]
    fn pseudo_speed_bands_match_recovered_mapping() {
        let t = default_movement();
        let run = t.run; // 4.0
        let mut c = CPlayerMovementController::new([0.0, 0.0, 0.0], t);

        // idle
        assert_eq!(c.pseudo_speed(), 0.0);

        // creep 0.8 -> 0.20
        c.state.vel = [t.creep, 0.0, 0.0];
        let creep = c.pseudo_speed();
        assert!((creep - t.creep / run).abs() < 1e-6, "creep {creep}");

        // walk 1.66 -> 0.415 (~0.5)
        c.state.vel = [t.walk, 0.0, 0.0];
        let walk = c.pseudo_speed();
        assert!((walk - t.walk / run).abs() < 1e-6, "walk {walk}");
        assert!(walk > 0.3 && walk < 0.7, "walk band {walk}");
        assert!(walk > creep);

        // run 4.0 -> exactly 1.0
        c.state.vel = [t.run, 0.0, 0.0];
        assert!((c.pseudo_speed() - 1.0).abs() < 1e-6);

        // sprint 6.8 -> 1.7 (> 1.0)
        c.state.vel = [t.sprint, 0.0, 0.0];
        let sprint = c.pseudo_speed();
        assert!((sprint - t.sprint / run).abs() < 1e-6, "sprint {sprint}");
        assert!(sprint > 1.0, "sprint band {sprint}");

        // Free helper: degenerate run/speed is finite.
        assert_eq!(pseudo_speed_from_speed(1.0, 0.0), 0.0);
        assert_eq!(pseudo_speed_from_speed(f32::NAN, run), 0.0);
        assert!(pseudo_speed_from_speed(f32::INFINITY, run).is_finite());
        // Non-finite state stays finite.
        c.state.vel = [f32::NAN; 3];
        assert!(c.pseudo_speed().is_finite());
    }

    /// The controller caches the recovered `UpdateNormal` desired values
    /// (`controller+0x2c8` / `param_3[0..2]`).
    #[test]
    fn controller_reports_desired_speed_and_velocity() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let mut c = CPlayerMovementController::new([0.0, 0.0, 0.0], t);
        let out = c.update_normal(&input(1.0, 0.0, dt), &g, &[]);
        assert_eq!(out.stance, STANCE_STAND);
        assert!(approx(out.desired_speed, t.run));
        assert!(approx(out.desired_vel[0], t.run));
        assert!(approx(c.desired_speed, t.run));
        assert!(approx(c.desired_vel[0], t.run));
        // Idle reports zero desired speed.
        let out = c.update_normal(&input(0.0, 0.0, dt), &g, &[]);
        assert_eq!(out.desired_speed, 0.0);
        assert_eq!(out.stance, STANCE_STAND);
    }

    /// `maxDelta = BASE * dt` (`0x01619ca0` lines 1114–1164): the movement-
    /// matching path uses `DAT_01914de8 = π`, the default `DAT_01914df0 = 20π`.
    #[test]
    fn controller_constrained_flag_selects_clamp_rate() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 0.1; // 20π*dt = 6.28 > π/2, π*dt = 0.314

        // Constrained (animation/movement-matching) rate: one tick turns π*dt.
        let mut c = CPlayerMovementController::new([0.0, 0.0, 0.0], t);
        c.constrained = true;
        let (sn, cs) = c.state.yaw.sin_cos();
        c.update_normal(&input(sn, cs, dt), &g, &[]);
        assert!(
            (wrap_angle(c.state.yaw) - max_delta(dt, true)).abs() < 1e-5,
            "constrained yaw {}",
            c.state.yaw
        );

        // Default rate: the same +Y target is reached in a single tick.
        let mut c = CPlayerMovementController::new([0.0, 0.0, 0.0], t);
        c.constrained = false;
        let (sn, cs) = c.state.yaw.sin_cos();
        c.update_normal(&input(sn, cs, dt), &g, &[]);
        assert!(
            (wrap_angle(c.state.yaw) - PI / 2.0).abs() < 1e-4,
            "default yaw {}",
            c.state.yaw
        );
        assert!(wrap_angle(c.state.yaw) > max_delta(dt, true));
    }

    /// `MoveState.pos` is the capsule **feet**: snapped onto the terrain, and
    /// the capsule spans `pos.z ..= pos.z + height`. The assembled CDF has feet
    /// at local `z = 0`, so the feet position is the root translation.
    #[test]
    fn feet_are_snapped_to_ground() {
        let t = default_movement();
        let dt = 1.0 / 60.0;
        let g = flat(3.5);

        // Spawn already on the surface: the feet stay exactly there.
        let mut s = MoveState::new([0.0, 0.0, 3.5]);
        update_normal(&mut s, &input(0.0, 0.0, dt), &t, &g, &[]);
        assert!(s.grounded);
        assert!((s.pos[2] - 3.5).abs() < 1e-4, "z={}", s.pos[2]);

        // Spawned below the surface: lifted onto it.
        let mut s = MoveState::new([1.0, 2.0, -5.0]);
        update_normal(&mut s, &input(0.0, 0.0, dt), &t, &g, &[]);
        assert!(s.grounded);
        assert!((s.pos[2] - 3.5).abs() < 1e-4, "z={}", s.pos[2]);

        // Spawn snap from far above.
        let p = spawn_on_ground([0.0, 0.0, 100.0], &g, &[]);
        assert!((p[2] - 3.5).abs() < 1e-4, "spawn z={}", p[2]);

        // Free snap helper on MoveState clears vertical velocity + grounds.
        let mut s = MoveState::new([0.0, 0.0, 50.0]);
        s.vel[2] = -3.0;
        assert!(snap_to_ground(&mut s, SPAWN_SNAP_REACH, &g, &[]));
        assert!((s.pos[2] - 3.5).abs() < 1e-4);
        assert_eq!(s.vel[2], 0.0);
        assert!(s.grounded);

        // Capsule: feet + height, radius separate; solid spans [z, z+height].
        let c = CPlayerMovementController::new([0.0, 0.0, 3.5], t);
        let (feet, height) = c.capsule();
        assert!((feet[2] - 3.5).abs() < 1e-6);
        assert!((height - CAPSULE_HEIGHT).abs() < 1e-6);
        assert!((c.capsule_radius() - CAPSULE_RADIUS).abs() < 1e-6);
        assert!(c.probe_ground(GROUND_SNAP_REACH, &g, &[]).is_some());
    }

    /// While moving, the feet follow a slope (Z-up terrain heightfield) and the
    /// up-vector tilts toward the ground normal (`ProcessNormal`, `0x01621070`).
    #[test]
    fn feet_follow_slope_while_moving() {
        let t = default_movement();
        let ramp = |x: f32, _: f32| 0.5 * x;
        let dt = 1.0 / 60.0;
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        let i = input(1.0, 0.0, dt);
        for _ in 0..120 {
            update_normal(&mut s, &i, &t, &ramp, &[]);
        }
        assert!(s.grounded, "grounded on ramp");
        assert!(
            (s.pos[2] - 0.5 * s.pos[0]).abs() < 1e-2,
            "feet off ramp z={} x={}",
            s.pos[2],
            s.pos[0]
        );
        // Ground normal for `h = 0.5x` points toward -x; up slerps onto it.
        assert!(s.up[0] < 0.0, "up={:?}", s.up);
    }

    /// The recovered jump launch (`0x0156a560` line 566) arcs under the
    /// `Living` gravity and returns to the ground.
    #[test]
    fn controller_jump_arc_returns_to_ground() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let mut c = CPlayerMovementController::new([0.0, 0.0, 0.0], t);
        c.state.grounded = true;
        let launch = c.jump(t.jump_height);
        assert!(launch > 0.0);
        assert!(!c.state.grounded);

        let mut peak = 0.0f32;
        let i = input(0.0, 0.0, dt);
        for _ in 0..300 {
            c.update_normal(&i, &g, &[]);
            peak = peak.max(c.state.pos[2]);
        }
        assert!(c.state.grounded, "should have landed");
        assert!(c.state.pos[2].abs() < 1e-3, "z={}", c.state.pos[2]);
        assert!(peak > 0.5, "peak {peak}");
    }

    /// The whole controller converges to the stance speed (inertia 10 /
    /// inertiaAccel 16 at `+0xea4`/`+0xea8`).
    #[test]
    fn controller_inertia_converges_to_stance_speed() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let mut c = CPlayerMovementController::new([0.0, 0.0, 0.0], t);
        let i = input(1.0, 0.0, dt);
        for _ in 0..240 {
            c.update_normal(&i, &g, &[]);
        }
        assert!((c.state.vel[0] - t.run).abs() < 1e-3, "v={}", c.state.vel[0]);
        assert!((c.pseudo_speed() - 1.0).abs() < 1e-3);
    }

    /// The controller is total for non-finite input: no panic, no NaN escapes.
    #[test]
    fn controller_is_total_for_degenerate_input() {
        let t = default_movement();
        let g = flat(0.0);
        let mut c = CPlayerMovementController::new([f32::NAN, f32::INFINITY, f32::NEG_INFINITY], t);
        let i = MoveInput {
            forward: f32::NAN,
            strafe: f32::INFINITY,
            mouse_dx: f32::NAN,
            mouse_dy: f32::INFINITY,
            crouch: false,
            sprint: false,
            dt: f32::NAN,
        };
        c.update_normal(&i, &g, &[]);
        assert!(c.state.pos.iter().all(|v| v.is_finite()), "pos={:?}", c.state.pos);
        assert!(c.state.vel.iter().all(|v| v.is_finite()), "vel={:?}", c.state.vel);
        assert!(c.state.up.iter().all(|v| v.is_finite()));
        assert!(c.state.yaw.is_finite());
        assert!(c.pseudo_speed().is_finite());
        let (feet, h) = c.capsule();
        assert!(feet.iter().all(|v| v.is_finite()) && h.is_finite());
        // No ground within reach on a NaN terrain: the controller stays finite.
        let nan_h = |_: f32, _: f32| f32::NAN;
        c.update_normal(&i, &nan_h, &[]);
        assert!(c.state.pos.iter().all(|v| v.is_finite()));
    }

    // -----------------------------------------------------------------------
    // worker-2 additions: inertia integrator, markup rotation, stance product,
    // jump clamp, capsule/stance constants, controller accessors
    // -----------------------------------------------------------------------

    /// The integrator is per-axis, accel-vs-decel, first-order
    /// (`alpha = 1 - exp(-dt/tau)`) and clamped so it never overshoots.
    /// `CPlayer::PostPhysicalize` `0x01504890` reads `inertia`/`inertiaAccel`,
    /// but the measured effective lag is `notes/movement_telemetry.md`
    /// (τ≈0.2 s), passed here as the time constants.
    #[test]
    fn inertia_integrator_is_per_axis_accel_decel_clamped() {
        let dt = 1.0 / 60.0;
        let tau_a = 0.2f32;
        let tau_d = 0.4f32;
        // Rising target -> accel tau: alpha = 1 - exp(-dt/tau_a).
        let alpha_a = 1.0 - (-dt / tau_a).exp();
        let a = integrate_inertia([0.0; 3], [4.0, 0.0, 0.0], tau_a, tau_d, dt);
        assert!((a[0] - 4.0 * alpha_a).abs() < 1e-6, "accel {a:?}");
        assert!(a[0] > 0.0 && a[0] < 4.0);
        // Shrinking target -> decel tau.
        let alpha_d = 1.0 - (-dt / tau_d).exp();
        let d = integrate_inertia([4.0, 0.0, 0.0], [0.0; 3], tau_a, tau_d, dt);
        assert!((d[0] - 4.0 * (1.0 - alpha_d)).abs() < 1e-6, "decel {d:?}");
        assert!(4.0 - d[0] < a[0], "accel must be faster than decel");

        // Per-axis: axis 0 accelerates while axis 1 decelerates in one call.
        let mixed = integrate_inertia([0.0, 4.0, 0.0], [4.0, 0.0, 0.0], tau_a, tau_d, dt);
        assert!((mixed[0] - 4.0 * alpha_a).abs() < 1e-6, "mixed {mixed:?}");
        assert!((mixed[1] - 4.0 * (1.0 - alpha_d)).abs() < 1e-6, "mixed {mixed:?}");

        // Clamped: a zero tau snaps, and cannot overshoot the target.
        assert_eq!(
            integrate_inertia([0.0; 3], [4.0, 0.0, 0.0], 0.0, 0.0, dt)[0],
            4.0
        );
        // Reversal uses the decel tau and stays between current and target.
        let rev = integrate_inertia([4.0, 0.0, 0.0], [-4.0, 0.0, 0.0], tau_a, tau_d, dt);
        assert!(rev[0] < 4.0 && rev[0] > -4.0, "rev {rev:?}");

        // Total: zero dt is a no-op; NaN/inf collapse to finite.
        assert_eq!(
            integrate_inertia([1.0, 2.0, 3.0], [4.0; 3], tau_a, tau_d, 0.0),
            [1.0, 2.0, 3.0]
        );
        let bad = integrate_inertia([f32::NAN; 3], [f32::INFINITY; 3], tau_a, tau_d, dt);
        assert!(bad.iter().all(|v| v.is_finite()));
    }

    /// `CPlayerRotation::ProcessConstrainedToMarkup` `0x0161f8a0`: rotate toward
    /// the markup target at the constrained `π*dt` rate (`DAT_01914de8`), never
    /// faster; a degenerate target falls back to the accumulated (mouse) delta.
    #[test]
    fn constrained_markup_rotation_clamps_to_pi_rate() {
        let dt = 0.1; // π*dt = 0.314 << π/2
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        process_constrained_to_markup(&mut s, [0.0, 1.0, 0.0], dt);
        assert!(
            (wrap_angle(s.yaw) - max_delta(dt, true)).abs() < 1e-6,
            "yaw {}",
            s.yaw
        );
        assert!((s.delta_angles[1] - max_delta(dt, true)).abs() < 1e-6);

        // Repeated calls converge onto the target.
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        for _ in 0..200 {
            process_constrained_to_markup(&mut s, [0.0, 1.0, 0.0], dt);
        }
        assert!((wrap_angle(s.yaw) - PI / 2.0).abs() < 1e-4, "yaw {}", s.yaw);

        // Never exceeds the constrained per-call rate (target behind us).
        let dt60 = 1.0 / 60.0;
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        for _ in 0..10 {
            let before = s.yaw;
            process_constrained_to_markup(&mut s, [1.0, 0.0, 0.0], dt60);
            assert!(
                wrap_angle(s.yaw - before).abs() <= max_delta(dt60, true) + 1e-6,
                "step exceeded π*dt"
            );
        }

        // Degenerate target: apply the pre-accumulated mouse yaw unchanged.
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        s.delta_angles = [0.0, 0.5];
        process_constrained_to_markup(&mut s, [0.0, 0.0, 0.0], dt);
        assert!((wrap_angle(s.yaw) - 0.5).abs() < 1e-6, "yaw {}", s.yaw);

        // Total for non-finite input.
        let mut s = MoveState::new([0.0, 0.0, 0.0]);
        process_constrained_to_markup(&mut s, [f32::NAN, f32::INFINITY, 0.0], f32::NAN);
        assert!(s.yaw.is_finite() && s.delta_angles.iter().all(|v| v.is_finite()));
    }

    /// `GetStanceSpeed` `0x01501900` is the product
    /// `actor+0xe48 * stance+0x80 * vtable+0x288()`.
    #[test]
    fn get_stance_speed_is_recovered_product() {
        assert!((get_stance_speed(1.0, 4.0, 1.0) - 4.0).abs() < 1e-6);
        assert!((get_stance_speed(2.0, 4.0, 0.5) - 4.0).abs() < 1e-6);
        assert_eq!(get_stance_speed(0.0, 4.0, 1.0), 0.0);
        assert_eq!(get_stance_speed(1.0, 4.0, 0.0), 0.0);
        assert_eq!(get_stance_speed(f32::NAN, 4.0, 1.0), 0.0);
        assert!(get_stance_speed(f32::INFINITY, 4.0, 1.0).is_finite());

        let t = default_movement();
        let c = CPlayerMovementController::new([0.0, 0.0, 0.0], t);
        assert!((c.stance_speed() - t.run).abs() < 1e-6);
    }

    /// `0x0156a560`: the folded launch is clamped to `DAT_0172b8b8 = 22.0`
    /// (`_DAT_018903e8 = 484 = 22²`), and the slope/multiplier constants are the
    /// recovered shipped values.
    #[test]
    fn jump_velocity_is_clamped_to_recovered_22() {
        let g = class3_physics::GRAVITY;
        let v = jump(0.0, 1.0, g);
        assert!(v > 0.0 && v < JUMP_VELOCITY_CLAMP);
        // An absurd height is clamped to 22.
        assert_eq!(jump(0.0, 10_000.0, g), JUMP_VELOCITY_CLAMP);
        assert!((JUMP_VELOCITY_CLAMP * JUMP_VELOCITY_CLAMP - JUMP_VELOCITY_CLAMP_SQ).abs() < 1e-3);
        // A huge downward momentum is clamped symmetrically.
        assert_eq!(jump(1_000.0, 1.0, g), -JUMP_VELOCITY_CLAMP);
        assert_eq!(JUMP_MULTIPLIER_DEFAULT, 1.0);
        assert!((JUMP_SLOPE_COS2_CUTOFF - 0.7).abs() < 1e-6);
        assert!((JUMP_SLOPE_GAIN - 5.0).abs() < 1e-6);
    }

    /// The stand-stance capsule constants are derived from the recovered stance
    /// struct (`heightCollider 1.0`, `size {0.375, 0.375, 0.4}`).
    #[test]
    fn capsule_constants_are_stance_derived() {
        assert!((CAPSULE_RADIUS - 0.375).abs() < 1e-6);
        assert!((CAPSULE_HEIGHT - 1.8).abs() < 1e-6);
        assert!(USE_CAPSULE);
        let c = CPlayerMovementController::new([0.0, 0.0, 0.0], default_movement());
        assert!((c.capsule_radius() - class3_physics::PLAYER_SIZE_X).abs() < 1e-6);
        assert!((c.capsule().1 - class3_physics::PLAYER_CAPSULE_HEIGHT).abs() < 1e-6);
        // The step-over height is the stance `size.z`.
        assert!((class3_physics::STEP_HEIGHT - class3_physics::PLAYER_SIZE_Z).abs() < 1e-6);
    }

    /// The controller makes the clamp choice and desired velocity explicit
    /// (`20π` default vs `π` constrained; `param_3[0..2]` cache).
    #[test]
    fn controller_exposes_desired_velocity_and_clamp_choice() {
        let t = default_movement();
        let g = flat(0.0);
        let dt = 1.0 / 60.0;
        let mut c = CPlayerMovementController::new([0.0, 0.0, 0.0], t);
        assert_eq!(c.desired_velocity(), [0.0, 0.0, 0.0]);
        assert_eq!(c.clamp_rate(), MAX_DELTA_RATE_CONSTRAINED);
        assert!((c.max_delta(dt) - max_delta(dt, true)).abs() < 1e-6);

        let out = c.update_normal(&input(1.0, 0.0, dt), &g, &[]);
        assert!((c.desired_velocity()[0] - t.run).abs() < 1e-6);
        assert!((c.desired_velocity()[0] - out.desired_vel[0]).abs() < 1e-6);

        c.constrained = false;
        assert_eq!(c.clamp_rate(), MAX_DELTA_RATE_DEFAULT);
        assert!((c.max_delta(dt) - max_delta(dt, false)).abs() < 1e-6);
        assert!(c.desired_velocity().iter().all(|v| v.is_finite()));
    }
}
