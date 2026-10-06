//! Lightweight, deterministic collision + capsule character controller.
//!
//! The engine previously had no collision at all: the player walked through
//! walls and either floated or sank through the terrain. This module provides
//! the small pieces the app needs each frame:
//!
//! * [`Aabb`] — axis-aligned boxes for the static architecture. `architecture.rs`
//!   returns [`crate::architecture::PlacedBrush`] transforms but no bounds, so
//!   the caller supplies colliders as AABBs (in the same Z-up world space as
//!   [`crate::terrain`]).
//! * [`CharacterController::move_and_slide`] — a capsule swept against the
//!   terrain heightfield (provided as a closure, so the caller may use
//!   [`crate::terrain::Terrain`] however it likes) and against `&[Aabb]`.
//! * [`raycast`] — nearest hit against the terrain (march + bisection) and the
//!   AABBs (slab test).
//!
//! # Conventions
//!
//! Coordinates are **Z-up right-handed**: `x`/`y` are the ground plane and `z`
//! is height, matching `terrain.rs`. The controller's position `pos` is the
//! **feet** of the capsule: the solid spans `[pos.z, pos.z + height]` in `z` and
//! a circle of `radius` around `(pos.x, pos.y)` in the ground plane.
//!
//! Every entry point is total: degenerate input (zero/negative `dt`, empty
//! colliders, non-finite values) yields finite output and never panics.

/// Axis-aligned bounding box in world space (Z-up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: [f32; 3],
    /// Maximum corner.
    pub max: [f32; 3],
}

/// Nearest intersection returned by [`raycast`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Distance along the (normalised) ray direction.
    pub t: f32,
    /// World-space hit position.
    pub point: [f32; 3],
    /// Surface normal at the hit (unit length, facing the ray origin).
    pub normal: [f32; 3],
}

/// Capsule character controller.
#[derive(Debug, Clone, PartialEq)]
pub struct CharacterController {
    /// Capsule radius (m).
    pub radius: f32,
    /// Capsule height (m), measured feet-to-top.
    pub height: f32,
    /// Largest step the controller can climb without being blocked (m).
    pub step_height: f32,
    /// Largest walkable ground slope (degrees).
    pub max_slope_deg: f32,
    /// Vertical acceleration (`m/s²`, negative = down).
    pub gravity: f32,
    /// Current velocity (`m/s`). Only `velocity[2]` is integrated by the
    /// controller; horizontal velocity is the caller's business.
    pub velocity: [f32; 3],
    /// True when the last step rested on the terrain or a collider top.
    pub grounded: bool,
}

impl Aabb {
    /// Box from centre and **full** size (each component is the extent, so the
    /// half-extent is `size / 2`). Negative sizes are made positive.
    pub fn from_center_size(center: [f32; 3], size: [f32; 3]) -> Aabb {
        let h = [
            size[0].abs() * 0.5,
            size[1].abs() * 0.5,
            size[2].abs() * 0.5,
        ];
        Aabb {
            min: [center[0] - h[0], center[1] - h[1], center[2] - h[2]],
            max: [center[0] + h[0], center[1] + h[1], center[2] + h[2]],
        }
    }

    /// Inclusive point containment.
    pub fn contains(&self, p: [f32; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] <= self.max[i])
    }

    /// True when the two boxes share any volume or face (inclusive).
    pub fn overlaps(&self, other: &Aabb) -> bool {
        (0..3).all(|i| self.min[i] <= other.max[i] && self.max[i] >= other.min[i])
    }

    /// Grow every side outwards by `r` (negative grows inwards).
    pub fn expanded(&self, r: f32) -> Aabb {
        Aabb {
            min: [self.min[0] - r, self.min[1] - r, self.min[2] - r],
            max: [self.max[0] + r, self.max[1] + r, self.max[2] + r],
        }
    }

    /// Geometric centre.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Closest point inside the box to `p` (the box itself if `p` is inside).
    pub fn closest_point(&self, p: [f32; 3]) -> [f32; 3] {
        [
            clampf(p[0], self.min[0], self.max[0]),
            clampf(p[1], self.min[1], self.max[1]),
            clampf(p[2], self.min[2], self.max[2]),
        ]
    }
}

impl CharacterController {
    /// New controller with sensible defaults: `step_height = 0.4 m`,
    /// `max_slope_deg = 50°`, `gravity = -20 m/s²`.
    pub fn new(radius: f32, height: f32) -> CharacterController {
        CharacterController {
            radius: if radius.is_finite() && radius > 0.0 {
                radius
            } else {
                0.4
            },
            height: if height.is_finite() && height > 0.0 {
                height
            } else {
                1.8
            },
            step_height: 0.4,
            max_slope_deg: 50.0,
            gravity: -20.0,
            velocity: [0.0; 3],
            grounded: false,
        }
    }

    /// True when the capsule at `pos` intersects `aabb`.
    pub fn intersects(&self, pos: [f32; 3], aabb: &Aabb) -> bool {
        capsule_intersects(pos, self.radius, self.height, aabb)
    }

    /// Advance the capsule by `dxz` metres horizontally over `dt` seconds.
    ///
    /// The horizontal move is applied, the capsule is pushed out of any
    /// overlapping `colliders` along the smallest axis (with a step-up
    /// allowance), gravity is integrated into `velocity[2]`, and the result is
    /// clamped to the terrain `height(x, y)`. Steep ground is projected/slid on
    /// so the controller cannot walk up cliffs. Returns the new feet position.
    pub fn move_and_slide(
        &mut self,
        pos: [f32; 3],
        dxz: [f32; 2],
        dt: f32,
        height: &dyn Fn(f32, f32) -> f32,
        colliders: &[Aabb],
    ) -> [f32; 3] {
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
        let old = [sane(pos[0]), sane(pos[1]), sane(pos[2])];
        let sx = sane(dxz[0]);
        let sy = sane(dxz[1]);

        // Sub-divide the horizontal move so a fast step cannot tunnel through a
        // thin wall in one discrete jump.
        let len = (sx * sx + sy * sy).sqrt();
        let max_step = (sane(self.radius) * 0.5).max(0.05);
        let n = ((len / max_step).ceil() as usize).clamp(1, 256);

        let mut p = old;
        for _ in 0..n {
            let (np, _) = self.substep(
                p,
                [sx / n as f32, sy / n as f32],
                dt / n as f32,
                height,
                colliders,
            );
            p = np;
        }
        [sane(p[0]), sane(p[1]), sane(p[2])]
    }

    /// One collision-resolving integration sub-step.
    fn substep(
        &mut self,
        pos: [f32; 3],
        dxz: [f32; 2],
        dt: f32,
        height: &dyn Fn(f32, f32) -> f32,
        colliders: &[Aabb],
    ) -> ([f32; 3], bool) {
        let r = sane(self.radius).max(0.0);
        let h = sane(self.height).max(0.0);
        let step = sane(self.step_height).max(0.0);
        let cos_max = (sane(self.max_slope_deg).clamp(0.0, 90.0).to_radians()).cos();

        let old = [sane(pos[0]), sane(pos[1]), sane(pos[2])];
        let mut dx = sane(dxz[0]);
        let mut dy = sane(dxz[1]);

        // --- project horizontal motion away from a steep uphill slope ---
        let n0 = terrain_normal(height, old[0], old[1]);
        if n0[2] < cos_max {
            let ulen = (n0[0] * n0[0] + n0[1] * n0[1]).sqrt();
            if ulen > 1e-6 {
                let (ux, uy) = (-n0[0] / ulen, -n0[1] / ulen); // uphill in XY
                let dot = dx * ux + dy * uy;
                if dot > 0.0 {
                    dx -= ux * dot;
                    dy -= uy * dot;
                }
            }
        }

        // --- horizontal move, blocked when it demands an un-steppable climb ---
        let mut nx = old[0] + dx;
        let mut ny = old[1] + dy;
        if terrain(height, nx, ny) > old[2] + step {
            let n1 = terrain_normal(height, nx, ny);
            if n1[2] < cos_max {
                nx = old[0];
                ny = old[1];
            }
        }
        let mut p = [nx, ny, old[2]];

        // --- gravity ---
        self.velocity[2] = sane(self.velocity[2]) + sane(self.gravity) * dt;
        p[2] = sane(p[2] + self.velocity[2] * dt);

        // --- push out of static colliders (a few relaxation passes) ---
        let mut on_collider = false;
        for _ in 0..4 {
            let mut moved = false;
            for b in colliders {
                match capsule_box_resolve(p, r, h, b, step) {
                    Some(Resolve::StepUp(z)) => {
                        if z > p[2] {
                            p[2] = z;
                        }
                        on_collider = true;
                        moved = true;
                    }
                    Some(Resolve::Push(v)) => {
                        p[0] += v[0];
                        p[1] += v[1];
                        p[2] += v[2];
                        if v[2] > 0.0 {
                            on_collider = true;
                        }
                        moved = true;
                    }
                    None => {}
                }
            }
            if !moved {
                break;
            }
        }

        // --- terrain clamp + grounded state ---
        let ground = terrain(height, p[0], p[1]);
        if p[2] <= ground + 1e-3 {
            p[2] = ground;
            if self.velocity[2] < 0.0 {
                self.velocity[2] = 0.0;
            }
            self.grounded = true;
        } else if !on_collider && p[2] <= ground + step && self.velocity[2] <= 0.0 {
            // Walking off a small ledge / down a slope: snap to avoid floating.
            // Never snap down off a collider top we just stepped onto.
            p[2] = ground;
            if self.velocity[2] < 0.0 {
                self.velocity[2] = 0.0;
            }
            self.grounded = true;
        } else {
            self.grounded = on_collider;
        }

        ([sane(p[0]), sane(p[1]), sane(p[2])], on_collider)
    }
}

/// Result of resolving a capsule/AABB overlap.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Resolve {
    /// Climb onto the given top height without a horizontal push.
    StepUp(f32),
    /// Translate the capsule by this vector along the smallest axis.
    Push([f32; 3]),
}

/// Distance test: true when the vertical capsule at `pos` overlaps `aabb`.
pub fn capsule_intersects(pos: [f32; 3], radius: f32, height: f32, aabb: &Aabb) -> bool {
    capsule_box_resolve(pos, radius, height, aabb, 0.0).is_some()
}

/// Alias for [`capsule_intersects`].
pub fn capsule_intersects_aabb(pos: [f32; 3], radius: f32, height: f32, aabb: &Aabb) -> bool {
    capsule_intersects(pos, radius, height, aabb)
}

/// Translation needed to separate the capsule at `pos` from `aabb`, or `None`
/// when they do not overlap. A low box (top within `step_height`) yields a pure
/// upward translation so the caller can climb it.
pub fn overlap_resolve(
    pos: [f32; 3],
    radius: f32,
    height: f32,
    aabb: &Aabb,
    step_height: f32,
) -> Option<[f32; 3]> {
    match capsule_box_resolve(pos, radius, height, aabb, step_height)? {
        Resolve::StepUp(z) => Some([0.0, 0.0, z - pos[2]]),
        Resolve::Push(v) => Some(v),
    }
}

/// True when a sphere of radius `r` at `center` intersects `aabb`.
pub fn sphere_intersects_aabb(center: [f32; 3], r: f32, aabb: &Aabb) -> bool {
    if !r.is_finite() || r < 0.0 || !center.iter().all(|c| c.is_finite()) {
        return false;
    }
    let q = aabb.closest_point(center);
    let d2 =
        (center[0] - q[0]).powi(2) + (center[1] - q[1]).powi(2) + (center[2] - q[2]).powi(2);
    d2 <= r * r
}

/// Nearest of the terrain heightfield and the `colliders` along a ray.
///
/// `dir` need not be normalised. The terrain is sampled by marching in 0.5 m
/// steps and refined with a bisection when the ray dips below the surface.
/// `max <= 0`, a zero/degenerate direction, or a total miss returns `None`.
pub fn raycast(
    origin: [f32; 3],
    dir: [f32; 3],
    max: f32,
    height: &dyn Fn(f32, f32) -> f32,
    colliders: &[Aabb],
) -> Option<RayHit> {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if !len.is_finite() || len < 1e-12 || !max.is_finite() || max <= 0.0 {
        return None;
    }
    let d = [dir[0] / len, dir[1] / len, dir[2] / len];
    let o = [sane(origin[0]), sane(origin[1]), sane(origin[2])];

    let mut best: Option<RayHit> = None;
    for b in colliders {
        if let Some((t, n)) = ray_aabb(o, d, max, b) {
            if best.as_ref().is_none_or(|h| t < h.t) {
                best = Some(RayHit {
                    t,
                    point: [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t],
                    normal: n,
                });
            }
        }
    }
    if let Some(hit) = raycast_terrain(o, d, max, height) {
        if best.as_ref().is_none_or(|h| hit.t < h.t) {
            best = Some(hit);
        }
    }
    best
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

/// NaN-safe clamp (also tolerates rejected bounds).
#[inline]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if lo > hi {
        return (lo + hi) * 0.5;
    }
    let v = if v.is_finite() { v } else { lo };
    v.max(lo).min(hi)
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

/// Terrain height, mapping a non-finite sample to 0.
#[inline]
fn terrain(height: &dyn Fn(f32, f32) -> f32, x: f32, y: f32) -> f32 {
    let h = height(x, y);
    if h.is_finite() {
        h
    } else {
        0.0
    }
}

/// Up-facing terrain normal from central differences (unit length).
fn terrain_normal(height: &dyn Fn(f32, f32) -> f32, x: f32, y: f32) -> [f32; 3] {
    const E: f32 = 0.5;
    let gx = (terrain(height, x + E, y) - terrain(height, x - E, y)) / (2.0 * E);
    let gy = (terrain(height, x, y + E) - terrain(height, x, y - E)) / (2.0 * E);
    let len = (gx * gx + gy * gy + 1.0).sqrt();
    if !len.is_finite() || len < 1e-12 {
        return [0.0, 0.0, 1.0];
    }
    [-gx / len, -gy / len, 1.0 / len]
}

/// Resolve a capsule at `pos` against one `Aabb`.
fn capsule_box_resolve(
    pos: [f32; 3],
    radius: f32,
    height: f32,
    b: &Aabb,
    step_height: f32,
) -> Option<Resolve> {
    let r = sane(radius).max(0.0);
    let h = sane(height).max(0.0);
    let p = [sane(pos[0]), sane(pos[1]), sane(pos[2])];
    if r <= 0.0 || !b.min.iter().chain(b.max.iter()).all(|v| v.is_finite()) {
        return None;
    }

    let z0 = p[2];
    let z1 = p[2] + h;
    // Vertical overlap (the capsule occupies exactly [z0, z1]).
    if !(z0 < b.max[2] && z1 > b.min[2]) {
        return None;
    }

    // Closest point on the capsule's vertical segment to the box.
    let cap = r.min(h * 0.5);
    let (sa, sb) = (z0 + cap, z1 - cap);
    let cz = if sb >= sa {
        clampf(b.center()[2], sa, sb)
    } else {
        (z0 + z1) * 0.5
    };
    let dz = if cz < b.min[2] {
        b.min[2] - cz
    } else if cz > b.max[2] {
        cz - b.max[2]
    } else {
        0.0
    };
    if dz >= r {
        return None;
    }
    let reff = (r * r - dz * dz).max(0.0).sqrt();
    if reff <= 0.0 {
        return None;
    }

    let (cx, cy) = (p[0], p[1]);
    let dx = if cx < b.min[0] {
        b.min[0] - cx
    } else if cx > b.max[0] {
        cx - b.max[0]
    } else {
        0.0
    };
    let dy = if cy < b.min[1] {
        b.min[1] - cy
    } else if cy > b.max[1] {
        cy - b.max[1]
    } else {
        0.0
    };
    if dx * dx + dy * dy >= reff * reff {
        return None;
    }

    // Step-up: a box whose top is a short climb is mounted instead of blocked.
    let up = b.max[2] - z0;
    if b.max[2] > z0 && up <= sane(step_height).max(0.0) {
        return Some(Resolve::StepUp(b.max[2]));
    }

    // Smallest-axis separator over the valid directions. Candidates are
    // restricted to the side the capsule centre is actually on, so we never
    // "separate" by teleporting across the box.
    let mut best = f32::INFINITY;
    let mut vec = [0.0f32; 3];

    if cx <= b.min[0] {
        let m = cx - (b.min[0] - reff);
        offer(&mut best, &mut vec, m, [-m, 0.0, 0.0]);
    } else if cx >= b.max[0] {
        let m = (b.max[0] + reff) - cx;
        offer(&mut best, &mut vec, m, [m, 0.0, 0.0]);
    } else {
        let ml = cx - (b.min[0] - reff);
        let mr = (b.max[0] + reff) - cx;
        offer(&mut best, &mut vec, ml, [-ml, 0.0, 0.0]);
        offer(&mut best, &mut vec, mr, [mr, 0.0, 0.0]);
    }
    if cy <= b.min[1] {
        let m = cy - (b.min[1] - reff);
        offer(&mut best, &mut vec, m, [0.0, -m, 0.0]);
    } else if cy >= b.max[1] {
        let m = (b.max[1] + reff) - cy;
        offer(&mut best, &mut vec, m, [0.0, m, 0.0]);
    } else {
        let mb = cy - (b.min[1] - reff);
        let mf = (b.max[1] + reff) - cy;
        offer(&mut best, &mut vec, mb, [0.0, -mb, 0.0]);
        offer(&mut best, &mut vec, mf, [0.0, mf, 0.0]);
    }
    offer(&mut best, &mut vec, up, [0.0, 0.0, up]);
    let down = z1 - b.min[2];
    offer(&mut best, &mut vec, down, [0.0, 0.0, -down]);

    Some(Resolve::Push(vec))
}

/// Keep the smallest non-negative separating translation.
#[inline]
fn offer(best: &mut f32, vec: &mut [f32; 3], m: f32, v: [f32; 3]) {
    if m.is_finite() && m >= 0.0 && m < *best {
        *best = m;
        *vec = v;
    }
}

/// Slab test. Returns `(t, normal)` for the nearest entry in `[0, max]`.
fn ray_aabb(o: [f32; 3], d: [f32; 3], max: f32, b: &Aabb) -> Option<(f32, [f32; 3])> {
    let mut t0 = 0.0f32;
    let mut t1 = max;
    let mut normal = [0.0f32; 3];
    for i in 0..3 {
        if d[i].abs() < 1e-9 {
            if o[i] < b.min[i] || o[i] > b.max[i] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d[i];
        let mut ta = (b.min[i] - o[i]) * inv;
        let mut tb = (b.max[i] - o[i]) * inv;
        let mut sign = -1.0f32; // +d enters through the min face
        if ta > tb {
            core::mem::swap(&mut ta, &mut tb);
            sign = 1.0;
        }
        if ta > t0 {
            t0 = ta;
            normal = [0.0; 3];
            normal[i] = sign;
        }
        if tb < t1 {
            t1 = tb;
        }
        if t0 > t1 {
            return None;
        }
    }
    if t1 < 0.0 || t0 > max {
        return None;
    }
    Some((t0.max(0.0), normal))
}

/// March the terrain heightfield, then bisect the sign change.
fn raycast_terrain(
    o: [f32; 3],
    d: [f32; 3],
    max: f32,
    height: &dyn Fn(f32, f32) -> f32,
) -> Option<RayHit> {
    let f = |t: f32| -> f32 {
        let x = o[0] + d[0] * t;
        let y = o[1] + d[1] * t;
        let z = o[2] + d[2] * t;
        z - terrain(height, x, y)
    };
    if f(0.0) <= 0.0 {
        let n = terrain_normal(height, o[0], o[1]);
        return Some(RayHit {
            t: 0.0,
            point: o,
            normal: n,
        });
    }
    const STEP: f32 = 0.5;
    let iters = ((max / STEP).ceil() as usize).min(1 << 16);
    let mut prev = 0.0f32;
    for i in 1..=iters {
        let t = i as f32 * STEP;
        if t > max {
            break;
        }
        if f(t) <= 0.0 {
            let mut lo = prev;
            let mut hi = t;
            for _ in 0..28 {
                let mid = (lo + hi) * 0.5;
                if f(mid) <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let px = o[0] + d[0] * hi;
            let py = o[1] + d[1] * hi;
            let pz = o[2] + d[2] * hi;
            let n = terrain_normal(height, px, py);
            return Some(RayHit {
                t: hi,
                point: [px, py, pz],
                normal: n,
            });
        }
        prev = t;
    }
    None
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(z: f32) -> impl Fn(f32, f32) -> f32 {
        move |_, _| z
    }

    #[test]
    fn aabb_basics() {
        let b = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 4.0, 6.0],
        };
        assert_eq!(b.center(), [1.0, 2.0, 3.0]);
        assert!(b.contains([0.0, 0.0, 0.0]));
        assert!(b.contains([2.0, 4.0, 6.0]));
        assert!(!b.contains([2.1, 0.0, 0.0]));
        assert!(!b.contains([0.0, -0.1, 0.0]));

        let c = Aabb {
            min: [1.0, 1.0, 1.0],
            max: [3.0, 3.0, 3.0],
        };
        assert!(b.overlaps(&c));
        let far = Aabb {
            min: [10.0, 10.0, 10.0],
            max: [11.0, 11.0, 11.0],
        };
        assert!(!b.overlaps(&far));

        assert_eq!(b.expanded(1.0).min, [-1.0, -1.0, -1.0]);
        assert_eq!(b.expanded(1.0).max, [3.0, 5.0, 7.0]);

        assert_eq!(b.closest_point([-5.0, 2.0, 100.0]), [0.0, 2.0, 6.0]);
        assert_eq!(b.closest_point([1.0, 1.0, 1.0]), [1.0, 1.0, 1.0]);

        let d = Aabb::from_center_size([0.0, 0.0, 10.0], [2.0, 2.0, 4.0]);
        assert_eq!(d.min, [-1.0, -1.0, 8.0]);
        assert_eq!(d.max, [1.0, 1.0, 12.0]);
    }

    #[test]
    fn sphere_aabb() {
        let b = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        assert!(sphere_intersects_aabb([0.5, 0.5, 0.5], 0.1, &b));
        assert!(sphere_intersects_aabb([-0.3, 0.5, 0.5], 0.4, &b));
        assert!(!sphere_intersects_aabb([-0.5, 0.5, 0.5], 0.4, &b));
        assert!(!sphere_intersects_aabb([0.5, 0.5, 0.5], -1.0, &b));
    }

    #[test]
    fn capsule_intersect_helpers() {
        let b = Aabb {
            min: [0.0, -1.0, 0.0],
            max: [1.0, 1.0, 2.0],
        };
        // Feet at origin, body along the box.
        assert!(capsule_intersects([0.5, 0.0, 0.0], 0.4, 1.8, &b));
        // Far away.
        assert!(!capsule_intersects([5.0, 0.0, 0.0], 0.4, 1.8, &b));
        // Above the box (no vertical overlap).
        assert!(!capsule_intersects([0.5, 0.0, 5.0], 0.4, 1.8, &b));
        assert!(capsule_intersects_aabb([0.5, 0.0, 0.0], 0.4, 1.8, &b));
        assert!(overlap_resolve([0.5, 0.0, 0.0], 0.4, 1.8, &b, 0.0).is_some());
        assert!(overlap_resolve([5.0, 0.0, 0.0], 0.4, 1.8, &b, 0.0).is_none());
    }

    #[test]
    fn falls_to_flat_ground() {
        let mut c = CharacterController::new(0.4, 1.8);
        let h = flat(10.0);
        let mut p = [0.0, 0.0, 20.0];
        let mut saw_negative = false;
        for _ in 0..600 {
            p = c.move_and_slide(p, [0.0, 0.0], 1.0 / 60.0, &h, &[]);
            if c.velocity[2] < 0.0 {
                saw_negative = true;
            }
        }
        assert!(saw_negative, "gravity should have pulled the capsule down");
        assert!((p[2] - 10.0).abs() < 1e-3, "landed at {}", p[2]);
        assert!(c.grounded, "expected grounded on flat terrain");
        assert!(c.velocity[2].abs() < 1e-4, "velocity {}", c.velocity[2]);
        assert!(p.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gravity_velocity_then_zero() {
        let mut c = CharacterController::new(0.4, 1.8);
        let h = flat(0.0);
        let mut p = [0.0, 0.0, 100.0];
        p = c.move_and_slide(p, [0.0, 0.0], 0.1, &h, &[]);
        let v1 = c.velocity[2];
        assert!(v1 < 0.0, "velocity should fall, got {v1}");
        p = c.move_and_slide(p, [0.0, 0.0], 0.1, &h, &[]);
        let v2 = c.velocity[2];
        assert!(p.iter().all(|v| v.is_finite()));
        assert!(v2 < v1, "velocity should keep increasing: {v1} -> {v2}");

        // Land and confirm the downward velocity is cancelled.
        let mut p = [0.0, 0.0, 3.0];
        for _ in 0..600 {
            p = c.move_and_slide(p, [0.0, 0.0], 1.0 / 60.0, &flat(0.0), &[]);
        }
        assert!((p[2] - 0.0).abs() < 1e-3);
        assert_eq!(c.velocity[2], 0.0);
        assert!(c.grounded);
    }

    #[test]
    fn wall_slides_not_passes() {
        let mut c = CharacterController::new(0.4, 1.8);
        let h = flat(0.0);
        let wall = Aabb {
            min: [2.0, -5.0, 0.0],
            max: [3.0, 5.0, 3.0],
        };
        let mut p = [1.0, 0.0, 0.0];
        p = c.move_and_slide(p, [1.0, 0.5], 1.0 / 60.0, &h, &[wall]);
        assert!(p[0] + 1e-3 < 2.0, "capsule passed through wall x={}", p[0]);
        assert!(p[0] > 1.4, "capsule should rest near the wall surface: {}", p[0]);
        assert!(p[1] > 0.4, "capsule should slide along y: {}", p[1]);

        // Repeated pushes must never tunnel through.
        for _ in 0..120 {
            p = c.move_and_slide(p, [1.0, 0.0], 1.0 / 60.0, &h, &[wall]);
            assert!(p[0] + 1e-3 < 2.0, "tunnelled to x={}", p[0]);
        }
    }

    #[test]
    fn low_step_is_climbed() {
        let mut c = CharacterController::new(0.4, 1.8);
        let h = flat(0.0);
        let step = Aabb {
            min: [2.0, -1.0, 0.0],
            max: [3.0, 1.0, 0.3],
        };
        let mut p = [1.0, 0.0, 0.0];
        p = c.move_and_slide(p, [1.0, 0.0], 1.0 / 60.0, &h, &[step]);
        assert!((p[2] - 0.3).abs() < 1e-3, "did not step up, z={}", p[2]);
        // Continue across the top.
        p = c.move_and_slide(p, [1.0, 0.0], 1.0 / 60.0, &h, &[step]);
        assert!(p[0] > 2.5, "did not advance onto the step, x={}", p[0]);
        assert!((p[2] - 0.3).abs() < 1e-3, "lost the step height, z={}", p[2]);
    }

    #[test]
    fn high_wall_blocks() {
        let mut c = CharacterController::new(0.4, 1.8);
        let h = flat(0.0);
        let wall = Aabb {
            min: [2.0, -1.0, 0.0],
            max: [2.4, 1.0, 4.0],
        };
        let mut p = [1.0, 0.0, 0.0];
        for _ in 0..120 {
            p = c.move_and_slide(p, [2.0, 0.0], 1.0 / 60.0, &h, &[wall]);
        }
        assert!(p[0] < 2.0, "capsule got into/through the wall: {}", p[0]);
        assert!(p[0] < 1.7, "capsule should be pushed clear of the wall: {}", p[0]);
    }

    #[test]
    fn steep_slope_blocks_ascent() {
        let mut c = CharacterController::new(0.4, 1.8);
        // Height ramps up at slope atan(2) ~ 63.4deg (> max 50deg) everywhere.
        let h = |x: f32, _: f32| 2.0 * x;
        c.grounded = true;
        let p = c.move_and_slide([0.0, 0.0, 0.0], [0.5, 0.0], 1.0 / 60.0, &h, &[]);
        assert!(p[0] < 0.1, "climbed a steep slope to x={}", p[0]);
        assert!(p[2] < 0.5, "gained height on a steep slope: z={}", p[2]);

        // Motion along the contour (y) is still allowed.
        let p2 = c.move_and_slide([0.0, 0.0, 0.0], [0.0, 0.5], 1.0 / 60.0, &h, &[]);
        assert!(p2[1] > 0.4, "contour motion blocked: y={}", p2[1]);
    }

    #[test]
    fn raycast_hits_terrain() {
        let hits = raycast([0.0, 0.0, 20.0], [0.0, 0.0, -1.0], 100.0, &flat(10.0), &[])
            .expect("terrain hit");
        assert!((hits.t - 10.0).abs() < 1e-2, "t={}", hits.t);
        assert!((hits.point[2] - 10.0).abs() < 1e-2, "z={}", hits.point[2]);
        assert!(hits.normal[2] > 0.9, "normal={:?}", hits.normal);
    }

    #[test]
    fn raycast_hits_aabb() {
        // Terrain far below so the box wins.
        let h = flat(-1000.0);
        let b = Aabb {
            min: [-1.0, -1.0, 4.0],
            max: [1.0, 1.0, 6.0],
        };
        let hits =
            raycast([0.0, 0.0, 20.0], [0.0, 0.0, -1.0], 100.0, &h, &[b]).expect("aabb hit");
        assert!((hits.t - 14.0).abs() < 1e-3, "t={}", hits.t);
        assert!((hits.point[2] - 6.0).abs() < 1e-3, "z={}", hits.point[2]);
        assert_eq!(hits.normal, [0.0, 0.0, 1.0]);

        // Nearest of terrain + box wins when both are candidates.
        let near = Aabb {
            min: [-1.0, -1.0, 12.0],
            max: [1.0, 1.0, 13.0],
        };
        let hits2 = raycast(
            [0.0, 0.0, 20.0],
            [0.0, 0.0, -1.0],
            100.0,
            &flat(10.0),
            &[near],
        )
        .expect("nearest hit");
        assert!((hits2.t - 7.0).abs() < 1e-3, "nearest t={}", hits2.t);
    }

    #[test]
    fn raycast_misses() {
        assert!(
            raycast([0.0, 0.0, 20.0], [1.0, 0.0, 0.0], 5.0, &flat(10.0), &[]).is_none()
        );
        assert!(
            raycast([0.0, 0.0, 20.0], [0.0, 0.0, -1.0], 0.0, &flat(10.0), &[]).is_none()
        );
        assert!(
            raycast([0.0, 0.0, 20.0], [0.0, 0.0, 0.0], 10.0, &flat(10.0), &[]).is_none()
        );
    }

    #[test]
    fn degenerate_input_is_total() {
        let mut c = CharacterController::new(0.4, 1.8);
        let h = flat(0.0);
        // Zero dt, empty colliders.
        let a = c.move_and_slide([1.0, 2.0, 3.0], [0.0, 0.0], 0.0, &h, &[]);
        assert_eq!(a, [1.0, 2.0, 3.0]); // no time passes, no gravity
        // Negative dt behaves like zero.
        let b = c.move_and_slide([1.0, 2.0, 5.0], [0.0, 0.0], -1.0, &h, &[]);
        assert!(b.iter().all(|v| v.is_finite()));
        // NaN inputs stay finite.
        let d = c.move_and_slide([f32::NAN; 3], [f32::NAN, 0.0], f32::NAN, &h, &[]);
        assert!(d.iter().all(|v| v.is_finite()), "got {d:?}");
        let e = raycast([f32::NAN; 3], [3.0, 4.0, 0.0], f32::NAN, &h, &[]);
        assert!(e.is_none());
        assert!(!sphere_intersects_aabb(
            [f32::NAN; 3],
            1.0,
            &Aabb {
                min: [0.0; 3],
                max: [1.0; 3]
            }
        ));
    }
}
