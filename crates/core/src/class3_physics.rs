//! Actor grounding / step / slope physics used by Class3's player controller.
//!
//! Faithful-enough port of the recovered `CPlayerMovementController` /
//! `CPlayerRotation` ground logic:
//!
//! * `FUN_01620b10` — `CPlayerRotation::Process`'s ground/terrain probe: a short
//!   physics ray (`FUN_0043d0d0` -> `"RayWorldIntersection(Game)"`, masks
//!   `0x107`/`0x40f`) producing the ground contact used to correct the actor's
//!   up-vector/lean. [`probe_ground`] reproduces the downward segment against
//!   the terrain heightfield plus static [`crate::physics::Aabb`] colliders.
//!   The exact ray direction/masks are `[UNRECOVERED]` (the decompiles only
//!   structurally characterize the helper); a straight-down segment is used.
//! * The `ProcessNormal` up-vector slerp / `Commit` ground normal
//!   (`actor+0xd8`, registered by `FUN_01508260` as the net property
//!   `"groundNormal"`) fed by that probe.
//! * Step/slope acceptance and ballistic jump integration.
//!
//! # Gravity
//! The actor's `physicsParams.Living.gravity = 9.81` is defined in
//! `extracted/scripts/entities/actor/basicactor.lua:526` and overridden with the
//! same value in `player.lua:61`; the PE publishes it as the net property
//! `"gravity"` at `actor+0x34` in `FUN_01508260` (`0x01508260`). It is the
//! CryEngine `Living`-entity gravity magnitude, applied downwards (`-z`).
//!
//! Coordinates are Z-up: `x`/`y` are the ground plane and `z` is height,
//! matching [`crate::physics`] and [`crate::terrain`]. Every entry point is
//! total: non-finite / degenerate input yields finite output and never panics.

/// Downward acceleration of the actor's `Living` physics body (`m/s²`).
///
/// Recovered from `physicsParams.Living.gravity = 9.81`
/// (`extracted/scripts/entities/actor/basicactor.lua:526`, `player.lua:61`),
/// the CryEngine default. Applied as `-GRAVITY` along `z` by [`integrate_jump`].
pub const GRAVITY: f32 = 9.81;

/// Maximum walkable ground slope for the player (`max_climb_angle`, degrees).
///
/// Recovered from `player.lua:66` (`max_climb_angle = 40.0`, with
/// `min_slide_angle = 40.1`); `basicactor.lua` defaults to `50.0`. The Class3
/// player controller reads the player override.
pub const MAX_SLOPE_DEG: f32 = 40.0;

/// Character step-over height (m).
///
/// Recovered from the player's **stand stance** struct (`player.lua`
/// `gameParams.stance[1]`, loaded by `FUN_01381fa0`): `size = {x=0.375,
/// y=0.375, z=0.4}`, so the step-over height is the stance `size.z` = **0.4 m**
/// (`notes/ghidra/player.md` §3/§7). `movement.md` §3's claim that no step
/// height exists in the scripts is superseded by the stance struct recovery.
pub const STEP_HEIGHT: f32 = 0.4;

// ---------------------------------------------------------------------------
// Player stand-stance capsule (recovered struct, `FUN_01381fa0`)
// ---------------------------------------------------------------------------
//
// The stance struct is `0xd4` bytes (`player.md` §3); the stand entry ships
// (`extracted-game/gamedata/scripts/entities/actor/player.lua`):
//
//   heightCollider = 1.0
//   heightPivot    = 0.0
//   size           = { x = 0.375, y = 0.375, z = 0.4 }
//   useCapsule     = 1
//
// `size.x`/`size.y` are the capsule radius; `size.z` is the step-over height;
// the collider's total feet-to-top extent is `heightCollider + 2 * size.z`
// (`[UNRECOVERED]` exact CryEngine capsule packing, but this reproduces the
// shipped 1.8 m player height and keeps the step at `size.z`).

/// `stance+0x00` — `heightCollider` for the player's stand stance.
pub const PLAYER_HEIGHT_COLLIDER: f32 = 1.0;
/// `stance+0x04` — `heightPivot` for the player's stand stance.
pub const PLAYER_HEIGHT_PIVOT: f32 = 0.0;
/// `stance+0x18` `size.x` — capsule radius (X) for the player's stand stance.
pub const PLAYER_SIZE_X: f32 = 0.375;
/// `stance+0x1c` `size.y` — capsule radius (Y) for the player's stand stance.
pub const PLAYER_SIZE_Y: f32 = 0.375;
/// `stance+0x20` `size.z` — step-over height for the player's stand stance.
pub const PLAYER_SIZE_Z: f32 = STEP_HEIGHT;
/// `stance+0x08` — `useCapsule` for the player's stand stance.
pub const PLAYER_USE_CAPSULE: bool = true;
/// Capsule radius (m) derived from the stand stance `size.x`.
pub const PLAYER_CAPSULE_RADIUS: f32 = PLAYER_SIZE_X;
/// Capsule feet-to-top height (m): `heightCollider + 2 * size.z` = `1.8`.
pub const PLAYER_CAPSULE_HEIGHT: f32 = PLAYER_HEIGHT_COLLIDER + 2.0 * PLAYER_SIZE_Z;

/// Ground contact returned by [`probe_ground`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundHit {
    /// World-space height (`z`) of the contact surface.
    pub height: f32,
    /// Up-facing surface normal at the contact (unit length).
    pub normal: [f32; 3],
    /// Distance from the probe origin down to the surface (`>= 0`).
    pub dist: f32,
}

/// Cast a short segment straight down from `(x, y, z)` for `reach` metres,
/// returning the first ground contact (terrain heightfield or an [`Aabb`] top).
///
/// Mirrors `FUN_01620b10`'s use of a short physics ray to find the surface the
/// actor stands on. The terrain is sampled at the fixed `(x, y)` (the segment is
/// vertical), and an `Aabb` only counts when `(x, y)` lies over it. The nearest
/// (highest) surface within `reach` wins. Degenerate input returns `None`.
pub fn probe_ground(
    x: f32,
    y: f32,
    z: f32,
    reach: f32,
    height: &dyn Fn(f32, f32) -> f32,
    colliders: &[crate::physics::Aabb],
) -> Option<GroundHit> {
    if !(x.is_finite() && y.is_finite() && z.is_finite()) {
        return None;
    }
    if !(reach.is_finite() && reach > 0.0) {
        return None;
    }
    let z = sane(z);

    // Terrain: `(x, y)` is constant along the vertical segment, so the surface
    // is a single height `g`, crossed at `t = z - g`.
    let g = sane(height(x, y));
    let t_terr = z - g;
    let mut best: Option<GroundHit> = if t_terr <= reach {
        Some(GroundHit {
            height: g,
            normal: ground_normal(x, y, height),
            dist: t_terr.max(0.0),
        })
    } else {
        None
    };

    // Static architecture: the nearest downward-facing top face under `(x, y)`.
    for b in colliders {
        if !b.min.iter().chain(b.max.iter()).all(|v| v.is_finite()) {
            continue;
        }
        if x < b.min[0] || x > b.max[0] || y < b.min[1] || y > b.max[1] {
            continue;
        }
        let top = b.max[2];
        let t = z - top;
        if t < 0.0 || t > reach {
            continue;
        }
        if best.as_ref().is_none_or(|h| t < h.dist) {
            best = Some(GroundHit {
                height: top,
                normal: [0.0, 0.0, 1.0],
                dist: t,
            });
        }
    }
    best
}

/// Snap `pos[2]` to the nearest ground surface within `reach` metres,
/// returning the contact.
///
/// The player's [`crate::class3_move::MoveState::pos`] is the **feet** of the
/// capsule (`pos.z ..= pos.z + height`), and the assembled CDF has its feet at
/// local `z = 0`; snapping the feet to the contact height is therefore what
/// lands the character on the ground. This is the spawn/teleport counterpart to
/// the per-frame clamp in [`crate::physics::CharacterController::move_and_slide`]
/// and uses the same [`probe_ground`] ray.
///
/// A non-finite `pos` or `reach`, or no surface within reach, returns `None`
/// and leaves `pos` untouched. A surface at or above the probe also counts
/// (the actor is lifted out of the ground).
pub fn snap_to_ground(
    pos: &mut [f32; 3],
    reach: f32,
    height: &dyn Fn(f32, f32) -> f32,
    colliders: &[crate::physics::Aabb],
) -> Option<GroundHit> {
    if !pos.iter().all(|v| v.is_finite()) {
        return None;
    }
    let hit = probe_ground(pos[0], pos[1], pos[2], reach, height, colliders)?;
    pos[2] = sane(hit.height);
    Some(hit)
}

/// Up-facing terrain normal from central differences (unit length).
///
/// Uses a `0.5 m` sample offset, matching `crate::physics`'s internal terrain
/// normal. Non-finite samples collapse to `0` and a degenerate gradient to
/// `[0, 0, 1]`.
pub fn ground_normal(x: f32, y: f32, height: &dyn Fn(f32, f32) -> f32) -> [f32; 3] {
    const E: f32 = 0.5;
    if !(x.is_finite() && y.is_finite()) {
        return [0.0, 0.0, 1.0];
    }
    let gx = (sane(height(x + E, y)) - sane(height(x - E, y))) / (2.0 * E);
    let gy = (sane(height(x, y + E)) - sane(height(x, y - E))) / (2.0 * E);
    let len = (gx * gx + gy * gy + 1.0).sqrt();
    if !len.is_finite() || len < 1e-12 {
        return [0.0, 0.0, 1.0];
    }
    [-gx / len, -gy / len, 1.0 / len]
}

/// True when `normal` is walkable for a maximum slope of `max_slope_deg`
/// degrees from vertical.
///
/// `normal` need not be unit length. A downward-facing or degenerate normal is
/// rejected; `max_slope_deg` is clamped to `[0, 90]`.
pub fn slope_ok(normal: [f32; 3], max_slope_deg: f32) -> bool {
    let len2 =
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if !len2.is_finite() || len2 < 1e-12 {
        return false;
    }
    if !max_slope_deg.is_finite() {
        return false;
    }
    let cos_slope = normal[2] / len2.sqrt();
    let limit = max_slope_deg.clamp(0.0, 90.0).to_radians().cos();
    cos_slope >= limit
}

/// True when a move from `prev_z` to `new_z` is a climb the actor can step over.
///
/// Ascending must fit within `step_height` (with a tiny tolerance); descending
/// is never a blocked step. Non-finite input is rejected.
pub fn step_up_ok(prev_z: f32, new_z: f32, step_height: f32) -> bool {
    if !(prev_z.is_finite() && new_z.is_finite() && step_height.is_finite()) {
        return false;
    }
    const TOL: f32 = 1e-4;
    new_z - prev_z <= step_height.max(0.0) + TOL
}

/// Launch speed for a ballistic jump reaching `height` metres under gravity `g`:
/// `sqrt(2 * g * height)`. Degenerate input returns `0`.
pub fn jump_velocity(g: f32, height: f32) -> f32 {
    if !(g.is_finite() && height.is_finite()) || g <= 0.0 || height <= 0.0 {
        return 0.0;
    }
    (2.0 * g * height).sqrt()
}

/// Integrate one vertical jump/fall step and resolve it against the ground.
///
/// `vz` is updated by `-GRAVITY * dt` and `z` by `vz * dt`; when the result
/// reaches `ground_h` the actor snaps to the ground, downward velocity is
/// cancelled and `grounded` is set. A non-positive/non-finite `dt` is a no-op.
pub fn integrate_jump(
    z: f32,
    vz: &mut f32,
    grounded: &mut bool,
    ground_h: f32,
    dt: f32,
) -> f32 {
    if !(dt.is_finite() && dt > 0.0) {
        *vz = sane(*vz);
        return sane(z);
    }
    let ground = sane(ground_h);
    let mut z = sane(z);
    let mut v = sane(*vz) - GRAVITY * dt;
    z += v * dt;

    if z <= ground {
        z = ground;
        if v < 0.0 {
            v = 0.0;
        }
        *grounded = true;
    } else {
        *grounded = false;
    }
    *vz = v;
    z
}

/// Map a non-finite value to 0 so nothing NaN escapes.
#[inline]
fn sane(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::Aabb;

    fn flat(z: f32) -> impl Fn(f32, f32) -> f32 {
        move |_, _| z
    }

    #[test]
    fn probe_ground_flat() {
        let hit = probe_ground(0.0, 0.0, 15.0, 10.0, &flat(10.0), &[])
            .expect("flat terrain hit");
        assert!((hit.height - 10.0).abs() < 1e-3, "height {}", hit.height);
        assert!((hit.dist - 5.0).abs() < 1e-3, "dist {}", hit.dist);
        assert!(hit.normal[2] > 0.99, "normal {:?}", hit.normal);

        // Reach too short to touch the surface.
        assert!(probe_ground(0.0, 0.0, 15.0, 4.0, &flat(10.0), &[]).is_none());
    }

    #[test]
    fn probe_ground_hill_normal() {
        // Linear ramp `h = 0.5 * x`; gradient (0.5, 0),
        // normal = (-0.5, 0, 1) / sqrt(1.25).
        let hill = |x: f32, _: f32| 0.5 * x;
        let hit = probe_ground(1.0, 0.0, 5.0, 10.0, &hill, &[]).expect("ramp hit");
        assert!((hit.height - 0.5).abs() < 1e-4, "height {}", hit.height);
        let expected = -0.5 / (1.25f32).sqrt();
        assert!((hit.normal[0] - expected).abs() < 1e-4, "normal {:?}", hit.normal);
        assert!(hit.normal[1].abs() < 1e-6);
        assert!(hit.normal[2] > 0.8);
    }

    #[test]
    fn probe_ground_collider_wins() {
        // Terrain far below; an AABB top at z = 3 is the nearest contact.
        let box_ = Aabb {
            min: [-1.0, -1.0, 1.0],
            max: [1.0, 1.0, 3.0],
        };
        let hit = probe_ground(0.0, 0.0, 10.0, 10.0, &flat(-100.0), &[box_])
            .expect("collider hit");
        assert!((hit.height - 3.0).abs() < 1e-4, "height {}", hit.height);
        assert!((hit.dist - 7.0).abs() < 1e-4, "dist {}", hit.dist);
        assert_eq!(hit.normal, [0.0, 0.0, 1.0]);

        // Outside the box footprint the terrain is used instead.
        let outside = probe_ground(5.0, 0.0, 0.0, 10.0, &flat(0.0), &[box_])
            .expect("terrain hit outside box");
        assert!((outside.height - 0.0).abs() < 1e-6);
    }

    #[test]
    fn slope_ok_by_angle() {
        assert!(slope_ok([0.0, 0.0, 1.0], 45.0));
        // 30 degrees from vertical: walkable under a 45 degree limit.
        let n30 = [30f32.to_radians().sin(), 0.0, 30f32.to_radians().cos()];
        assert!(slope_ok(n30, 45.0));
        // 60 degrees from vertical: rejected.
        let n60 = [60f32.to_radians().sin(), 0.0, 60f32.to_radians().cos()];
        assert!(!slope_ok(n60, 45.0));
        // Just inside / outside the 45 degree boundary.
        let n44 = [44.9f32.to_radians().sin(), 0.0, 44.9f32.to_radians().cos()];
        let n45 = [45.1f32.to_radians().sin(), 0.0, 45.1f32.to_radians().cos()];
        assert!(slope_ok(n44, 45.0));
        assert!(!slope_ok(n45, 45.0));
        // Scale-invariant: a non-unit normal behaves the same.
        assert!(slope_ok([0.0, 0.0, 7.0], 45.0));
        // Downward / degenerate normals are rejected.
        assert!(!slope_ok([0.0, 0.0, -1.0], 45.0));
        assert!(!slope_ok([0.0, 0.0, 0.0], 45.0));
    }

    #[test]
    fn step_up_ok_boundary() {
        assert!(step_up_ok(1.0, 1.0, 0.4));
        assert!(step_up_ok(1.0, 1.4, 0.4));
        assert!(!step_up_ok(1.0, 1.5, 0.4));
        // Descending is never a blocked step.
        assert!(step_up_ok(1.0, 0.5, 0.4));
        // Non-finite input is rejected.
        assert!(!step_up_ok(f32::NAN, 1.0, 0.4));
        assert!(!step_up_ok(1.0, f32::INFINITY, 0.4));
    }

    /// The stand-stance capsule (`player.lua` `gameParams.stance[1]`,
    /// `FUN_01381fa0`): `heightCollider 1.0`, `size {0.375, 0.375, 0.4}`,
    /// `useCapsule 1`. `STEP_HEIGHT` is `size.z`; the capsule radius is
    /// `size.x`; the feet-to-top height is `heightCollider + 2*size.z`.
    #[test]
    fn stance_capsule_constants_match_shipped_struct() {
        assert_eq!(STEP_HEIGHT, 0.4);
        assert_eq!(PLAYER_SIZE_Z, STEP_HEIGHT);
        assert_eq!(PLAYER_SIZE_X, 0.375);
        assert_eq!(PLAYER_SIZE_Y, 0.375);
        assert_eq!(PLAYER_HEIGHT_COLLIDER, 1.0);
        assert_eq!(PLAYER_HEIGHT_PIVOT, 0.0);
        assert!(PLAYER_USE_CAPSULE);
        assert_eq!(PLAYER_CAPSULE_RADIUS, 0.375);
        // 1.0 + 2*0.4 = 1.8 (the shipped player height).
        assert!((PLAYER_CAPSULE_HEIGHT - 1.8).abs() < 1e-6);
        // The step boundary uses the stance step height exactly.
        assert!(step_up_ok(0.0, PLAYER_SIZE_Z, STEP_HEIGHT));
        assert!(!step_up_ok(0.0, PLAYER_SIZE_Z + 0.2, STEP_HEIGHT));
    }

    #[test]
    fn jump_velocity_sane() {
        let v = jump_velocity(GRAVITY, 1.0);
        let expected = (2.0 * GRAVITY).sqrt();
        assert!(v > 0.0 && (v - expected).abs() < 1e-5, "v {v}");
        assert!((v - 4.429).abs() < 0.01, "v {v}");
        assert_eq!(jump_velocity(GRAVITY, 0.0), 0.0);
        assert_eq!(jump_velocity(-1.0, 1.0), 0.0);
        assert_eq!(jump_velocity(f32::NAN, 1.0), 0.0);
    }

    #[test]
    fn integrate_jump_returns_to_ground() {
        let mut z = 0.0f32;
        let mut vz = jump_velocity(GRAVITY, 1.0);
        let mut grounded = false;
        let mut peak = 0.0f32;
        for _ in 0..240 {
            z = integrate_jump(z, &mut vz, &mut grounded, 0.0, 1.0 / 60.0);
            peak = peak.max(z);
        }
        assert!(grounded, "actor should have landed");
        assert!(z.abs() < 1e-4, "z {z}");
        assert!(vz.abs() < 1e-4, "vz {vz}");
        // The 1 m launch should actually leave the ground (minus one step).
        assert!(peak > 0.9, "peak {peak}");
    }

    #[test]
    fn snap_to_ground_places_feet_on_surface() {
        // Flat ground at 3.5 m: feet snap exactly onto it from above or below.
        let mut p = [0.0f32, 0.0, 100.0];
        let hit = snap_to_ground(&mut p, 200.0, &flat(3.5), &[]).expect("snap from above");
        assert!((p[2] - 3.5).abs() < 1e-4, "z {}", p[2]);
        assert!((hit.height - 3.5).abs() < 1e-4);

        let mut p = [0.0f32, 0.0, -5.0];
        assert!(snap_to_ground(&mut p, 200.0, &flat(3.5), &[]).is_some());
        assert!((p[2] - 3.5).abs() < 1e-4, "z {}", p[2]);

        // Collider top wins over far terrain.
        let box_ = Aabb { min: [-1.0, -1.0, 1.0], max: [1.0, 1.0, 3.0] };
        let mut p = [0.0f32, 0.0, 10.0];
        snap_to_ground(&mut p, 20.0, &flat(-100.0), &[box_]);
        assert!((p[2] - 3.0).abs() < 1e-4, "z {}", p[2]);

        // No surface within reach leaves the position untouched.
        let mut p = [0.0f32, 0.0, 100.0];
        assert!(snap_to_ground(&mut p, 1.0, &flat(0.0), &[]).is_none());
        assert!((p[2] - 100.0).abs() < 1e-6);

        // Non-finite input is rejected without panic.
        let mut p = [f32::NAN, 0.0, 0.0];
        assert!(snap_to_ground(&mut p, 10.0, &flat(0.0), &[]).is_none());
    }

    #[test]
    fn ground_normal_helper() {
        let hill = |x: f32, _: f32| 2.0 * x;
        let n = ground_normal(0.0, 0.0, &hill);
        assert!((n[0] + 2.0 / (5.0f32).sqrt()).abs() < 1e-4, "n {:?}", n);
        assert!(n[2] > 0.0);
        assert!(n.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn degenerate_input_is_total() {
        let nan_h = |_: f32, _: f32| f32::NAN;
        // Non-finite origins / reach -> None.
        assert!(probe_ground(f32::NAN, 0.0, 5.0, 10.0, &flat(0.0), &[]).is_none());
        assert!(probe_ground(0.0, 0.0, 5.0, 0.0, &flat(0.0), &[]).is_none());
        assert!(probe_ground(0.0, 0.0, 5.0, -1.0, &flat(0.0), &[]).is_none());
        // A NaN terrain collapses to 0 rather than escaping.
        let hit = probe_ground(0.0, 0.0, 5.0, 10.0, &nan_h, &[]);
        if let Some(h) = hit {
            assert!(h.height.is_finite() && h.dist.is_finite());
        }
        // ground_normal is always finite.
        assert!(ground_normal(f32::NAN, 0.0, &nan_h).iter().all(|v| v.is_finite()));

        // integrate_jump: zero dt is a no-op, NaN input stays finite.
        let mut vz = f32::NAN;
        let mut grounded = false;
        let z = integrate_jump(f32::NAN, &mut vz, &mut grounded, 0.0, 0.0);
        assert!(z.is_finite() && vz.is_finite());
    }
}
