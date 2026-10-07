//! A static collision world for the native engine: a terrain heightfield plus
//! every placed architectural brush / static prop, each carrying the physics
//! proxy shapes decoded by [`crate::cry::phys_proxy`].
//!
//! This is the native replacement for the part of CryPhysics the game code
//! observes through `RayWorldIntersection` / `IPhysicalWorld`:
//!
//! * a uniform XY grid broadphase over the instance AABBs;
//! * [`CollisionWorld::raycast`] — nearest hit with point, unit normal and a
//!   surface/material id (terrain surface type or [`SURFACE_SHAPE`]);
//! * [`CollisionWorld::overlap_capsule`] / [`CollisionWorld::sweep_capsule`] —
//!   the player capsule against the terrain and the static proxies, returning
//!   contact normal + penetration depth.
//!
//! Everything is CryEngine-native **Z-up right-handed, metres**, matching
//! [`crate::terrain`], [`crate::architecture`] and [`crate::physics`].
//!
//! All entry points are total: degenerate input returns `None`/an empty result
//! and never panics.

use crate::cry::phys_proxy::{self, PhysShape, Transform};
use crate::crypak::Vfs;
use crate::terrain::Terrain;
use std::collections::HashMap;

/// Surface id reported for non-terrain (proxy) hits.
pub const SURFACE_SHAPE: u8 = 254;
/// Instance id used for the terrain in [`RayHit`] / [`Contact`].
pub const INSTANCE_TERRAIN: u32 = u32::MAX;

/// One placement to include in the world.
#[derive(Debug, Clone)]
pub struct InstanceSpec {
    /// VFS model path (`.cgf`/`.cga`).
    pub model: String,
    /// World translation (m).
    pub pos: [f32; 3],
    /// Orientation quaternion `[x, y, z, w]`.
    pub rot: [f32; 4],
    /// Per-axis scale.
    pub scale: [f32; 3],
}

impl InstanceSpec {
    /// Instance with uniform scale and identity rotation.
    pub fn at(model: impl Into<String>, pos: [f32; 3], scale: f32) -> InstanceSpec {
        InstanceSpec { model: model.into(), pos, rot: [0.0, 0.0, 0.0, 1.0], scale: [scale; 3] }
    }
}

/// Nearest ray/world intersection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Distance along the normalised ray direction.
    pub t: f32,
    /// World hit position.
    pub point: [f32; 3],
    /// Unit surface normal (faces the ray origin).
    pub normal: [f32; 3],
    /// Terrain surface id, or [`SURFACE_SHAPE`] for a proxy.
    pub surface: u8,
    /// Instance index, or [`INSTANCE_TERRAIN`].
    pub instance: u32,
}

/// A capsule/shape overlap contact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contact {
    /// Unit normal that separates the capsule from the obstacle.
    pub normal: [f32; 3],
    /// Penetration depth (>= 0).
    pub depth: f32,
    /// Approximate world contact point.
    pub point: [f32; 3],
    /// Terrain surface id, or [`SURFACE_SHAPE`].
    pub surface: u8,
    /// Instance index, or [`INSTANCE_TERRAIN`].
    pub instance: u32,
}

/// Result of [`CollisionWorld::sweep_capsule`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sweep {
    /// Feet position after sliding.
    pub pos: [f32; 3],
    /// Deepest contact encountered, if any.
    pub contact: Option<Contact>,
}

struct Instance {
    shape: PhysShape,
    min: [f32; 3],
    max: [f32; 3],
}

/// Uniform XY grid over the instance bounds.
struct Grid {
    origin: [f32; 2],
    cell: f32,
    nx: usize,
    ny: usize,
    cells: Vec<Vec<u32>>,
}

impl Grid {
    fn build(instances: &[Instance]) -> Grid {
        let cell = 16.0f32;
        let mut mn = [f32::INFINITY; 2];
        let mut mx = [f32::NEG_INFINITY; 2];
        for ins in instances {
            for k in 0..2 {
                if ins.min[k].is_finite() && ins.max[k].is_finite() {
                    mn[k] = mn[k].min(ins.min[k]);
                    mx[k] = mx[k].max(ins.max[k]);
                }
            }
        }
        if !mn[0].is_finite() {
            return Grid { origin: [0.0, 0.0], cell, nx: 1, ny: 1, cells: vec![Vec::new()] };
        }
        let origin = [mn[0] - cell, mn[1] - cell];
        let nx = (((mx[0] - origin[0]) / cell).ceil() as isize + 1).clamp(1, 2048) as usize;
        let ny = (((mx[1] - origin[1]) / cell).ceil() as isize + 1).clamp(1, 2048) as usize;
        let mut grid = Grid { origin, cell, nx, ny, cells: vec![Vec::new(); nx * ny] };
        for (id, ins) in instances.iter().enumerate() {
            let (x0, x1, y0, y1) = grid.cell_range(ins.min, ins.max);
            for cy in y0..=y1 {
                for cx in x0..=x1 {
                    grid.cells[cy * grid.nx + cx].push(id as u32);
                }
            }
        }
        grid
    }

    fn cell_range(&self, min: [f32; 3], max: [f32; 3]) -> (usize, usize, usize, usize) {
        let c = |v: f32, o: f32, n: usize| -> usize {
            (((v - o) / self.cell).floor().max(0.0) as usize).min(n - 1)
        };
        (
            c(min[0], self.origin[0], self.nx),
            c(max[0], self.origin[0], self.nx),
            c(min[1], self.origin[1], self.ny),
            c(max[1], self.origin[1], self.ny),
        )
    }

    /// Cells whose footprint the ray's XY projection crosses, in order.
    fn ray_cells(&self, o: [f32; 3], d: [f32; 3], max: f32) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        // Clamp the XY origin into the grid; a ray from far outside in Z is the
        // common case (player ground probes), and XY outside is harmless.
        let x = ((o[0] - self.origin[0]) / self.cell)
            .floor()
            .clamp(0.0, (self.nx - 1) as f32);
        let y = ((o[1] - self.origin[1]) / self.cell)
            .floor()
            .clamp(0.0, (self.ny - 1) as f32);
        let mut cx = x as usize;
        let mut cy = y as usize;

        let step_x: isize = if d[0] > 1e-9 { 1 } else if d[0] < -1e-9 { -1 } else { 0 };
        let step_y: isize = if d[1] > 1e-9 { 1 } else if d[1] < -1e-9 { -1 } else { 0 };
        if step_x == 0 && step_y == 0 {
            out.push((cx, cy));
            return out;
        }

        let ox = (cx as f32 + 0.5) * self.cell + self.origin[0];
        let oy = (cy as f32 + 0.5) * self.cell + self.origin[1];
        let next_x = if step_x > 0 {
            self.origin[0] + (cx + 1) as f32 * self.cell
        } else {
            self.origin[0] + cx as f32 * self.cell
        };
        let next_y = if step_y > 0 {
            self.origin[1] + (cy + 1) as f32 * self.cell
        } else {
            self.origin[1] + cy as f32 * self.cell
        };
        let mut t_dx = if step_x != 0 { ((next_x - ox) / d[0]).abs() } else { f32::INFINITY };
        let mut t_dy = if step_y != 0 { ((next_y - oy) / d[1]).abs() } else { f32::INFINITY };
        let dt_x = if step_x != 0 { (self.cell / d[0]).abs() } else { f32::INFINITY };
        let dt_y = if step_y != 0 { (self.cell / d[1]).abs() } else { f32::INFINITY };

        let mut t = 0.0f32;
        out.push((cx, cy));
        let mut guard = 0;
        while t <= max && guard < 100_000 {
            guard += 1;
            if t_dx < t_dy {
                t = t_dx;
                cx = (cx as isize + step_x).clamp(0, self.nx as isize) as usize;
                if cx >= self.nx {
                    break;
                }
                t_dx += dt_x;
            } else {
                t = t_dy;
                cy = (cy as isize + step_y).clamp(0, self.ny as isize) as usize;
                if cy >= self.ny {
                    break;
                }
                t_dy += dt_y;
            }
            out.push((cx, cy));
        }
        out
    }
}

/// The static collision world.
pub struct CollisionWorld {
    terrain: Option<Terrain>,
    instances: Vec<Instance>,
    grid: Grid,
}

impl CollisionWorld {
    /// A world with no terrain and no instances.
    pub fn empty() -> CollisionWorld {
        let grid = Grid { origin: [0.0, 0.0], cell: 16.0, nx: 1, ny: 1, cells: vec![Vec::new()] };
        CollisionWorld { terrain: None, instances: Vec::new(), grid }
    }

    /// Build from explicit shapes (terrain optional). Useful for tests.
    pub fn from_shapes(terrain: Option<Terrain>, shapes: Vec<PhysShape>) -> CollisionWorld {
        let instances: Vec<Instance> = shapes
            .into_iter()
            .filter_map(|shape| {
                let (min, max) = shape.aabb();
                if min.iter().chain(max.iter()).all(|v| v.is_finite()) {
                    Some(Instance { shape, min, max })
                } else {
                    None
                }
            })
            .collect();
        let grid = Grid::build(&instances);
        CollisionWorld { terrain, instances, grid }
    }

    /// Build from a list of placed models, decoding each unique model's physics
    /// proxies once from `vfs`.
    pub fn build(terrain: Option<Terrain>, vfs: &Vfs, specs: &[InstanceSpec]) -> CollisionWorld {
        let mut cache: HashMap<String, Vec<phys_proxy::PhysicsProxy>> = HashMap::new();
        let mut shapes: Vec<PhysShape> = Vec::new();
        for spec in specs {
            let proxies = cache.entry(spec.model.clone()).or_insert_with(|| {
                vfs.read(&spec.model)
                    .map(|b| phys_proxy::decode_physics_proxies(&b))
                    .unwrap_or_default()
            });
            let t = Transform::from_trs(spec.pos, spec.rot, spec.scale);
            for p in proxies.iter() {
                if !p.collide {
                    continue;
                }
                shapes.push(phys_proxy::transform_shape(p.shape.clone(), &t));
            }
        }
        CollisionWorld::from_shapes(terrain, shapes)
    }

    /// Number of collision instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Reference to the terrain, when present.
    pub fn terrain(&self) -> Option<&Terrain> {
        self.terrain.as_ref()
    }

    // -----------------------------------------------------------------------
    // terrain sampling
    // -----------------------------------------------------------------------

    /// Bilinear terrain height at `(x, y)`, or `None` when there is no terrain.
    pub fn terrain_height(&self, x: f32, y: f32) -> Option<f32> {
        let t = self.terrain.as_ref()?;
        let s = t.samples;
        if s == 0 || t.spacing <= 0.0 || t.heights.len() < s * s {
            return None;
        }
        let fx = (x / t.spacing - 0.5).clamp(0.0, s as f32 - 1.0);
        let fy = (y / t.spacing - 0.5).clamp(0.0, s as f32 - 1.0);
        let i0 = fx.floor() as usize;
        let j0 = fy.floor() as usize;
        let i1 = (i0 + 1).min(s - 1);
        let j1 = (j0 + 1).min(s - 1);
        let tx = fx - i0 as f32;
        let ty = fy - j0 as f32;
        let h = |i: usize, j: usize| t.heights[j * s + i];
        let top = h(i0, j0) * (1.0 - tx) + h(i1, j0) * tx;
        let bot = h(i0, j1) * (1.0 - tx) + h(i1, j1) * tx;
        Some(top * (1.0 - ty) + bot * ty)
    }

    /// Up-facing unit terrain normal (or `[0,0,1]` without terrain).
    pub fn terrain_normal(&self, x: f32, y: f32) -> [f32; 3] {
        const E: f32 = 0.5;
        let (Some(hx1), Some(hx0), Some(hy1), Some(hy0)) = (
            self.terrain_height(x + E, y),
            self.terrain_height(x - E, y),
            self.terrain_height(x, y + E),
            self.terrain_height(x, y - E),
        ) else {
            return [0.0, 0.0, 1.0];
        };
        let gx = (hx1 - hx0) / (2.0 * E);
        let gy = (hy1 - hy0) / (2.0 * E);
        let len = (gx * gx + gy * gy + 1.0).sqrt();
        if len.is_finite() && len > 1e-9 {
            [-gx / len, -gy / len, 1.0 / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    /// Terrain surface id at `(x, y)`.
    pub fn terrain_surface(&self, x: f32, y: f32) -> u8 {
        self.terrain.as_ref().map(|t| t.surface_at(x, y)).unwrap_or(SURFACE_SHAPE)
    }

    // -----------------------------------------------------------------------
    // raycast
    // -----------------------------------------------------------------------

    /// Nearest hit of the ray `origin + t * dir` (`t` in `[0, max]`), where the
    /// direction need not be normalised.
    pub fn raycast(&self, origin: [f32; 3], dir: [f32; 3], max: f32) -> Option<RayHit> {
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if !len.is_finite() || len < 1e-12 || !max.is_finite() || max <= 0.0 {
            return None;
        }
        if !origin.iter().all(|v| v.is_finite()) {
            return None;
        }
        let d = [dir[0] / len, dir[1] / len, dir[2] / len];

        let mut best = self.raycast_terrain(origin, d, max);
        for (cx, cy) in self.grid.ray_cells(origin, d, max) {
            for &id in &self.grid.cells[cy * self.grid.nx + cx] {
                let ins = &self.instances[id as usize];
                if let Some((t, normal)) = ray_shape(origin, d, max, &ins.shape) {
                    if best.as_ref().is_none_or(|h: &RayHit| t < h.t) {
                        best = Some(RayHit {
                            t,
                            point: [origin[0] + d[0] * t, origin[1] + d[1] * t, origin[2] + d[2] * t],
                            normal,
                            surface: SURFACE_SHAPE,
                            instance: id,
                        });
                    }
                }
            }
        }
        best
    }

    fn raycast_terrain(&self, o: [f32; 3], d: [f32; 3], max: f32) -> Option<RayHit> {
        let t = self.terrain.as_ref()?;
        let _ = t;
        let f = |t: f32| -> f32 {
            let x = o[0] + d[0] * t;
            let y = o[1] + d[1] * t;
            let z = o[2] + d[2] * t;
            z - self.terrain_height(x, y).unwrap_or(f32::NEG_INFINITY)
        };
        if f(0.0) <= 0.0 {
            return Some(RayHit {
                t: 0.0,
                point: o,
                normal: self.terrain_normal(o[0], o[1]),
                surface: self.terrain_surface(o[0], o[1]),
                instance: INSTANCE_TERRAIN,
            });
        }
        const STEP: f32 = 0.5;
        let iters = ((max / STEP).ceil() as usize).min(1 << 16);
        let mut prev = 0.0f32;
        for i in 1..=iters {
            let ti = i as f32 * STEP;
            if ti > max {
                break;
            }
            if f(ti) <= 0.0 {
                let mut lo = prev;
                let mut hi = ti;
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
                return Some(RayHit {
                    t: hi,
                    point: [px, py, pz],
                    normal: self.terrain_normal(px, py),
                    surface: self.terrain_surface(px, py),
                    instance: INSTANCE_TERRAIN,
                });
            }
            prev = ti;
        }
        None
    }

    // -----------------------------------------------------------------------
    // capsule overlap / sweep
    // -----------------------------------------------------------------------

    /// All overlaps of the vertical capsule (feet at `pos`, given `radius` and
    /// `height`) with the terrain and static proxies.
    pub fn overlap_capsule_all(&self, pos: [f32; 3], radius: f32, height: f32) -> Vec<Contact> {
        let mut out = Vec::new();
        if !pos.iter().all(|v| v.is_finite())
            || !radius.is_finite()
            || radius <= 0.0
            || !height.is_finite()
            || height <= 0.0
        {
            return out;
        }
        let r = radius;
        let (sample_lo, sample_hi) = capsule_sphere_range(pos, r, height);
        let n = (((sample_hi - sample_lo) / (r * 0.5)).ceil() as usize).clamp(1, 64);

        // Terrain: the capsule bottom / feet vs the ground.
        if let Some(ground) = self.terrain_height(pos[0], pos[1]) {
            if pos[2] < ground {
                let mut c = Contact {
                    normal: self.terrain_normal(pos[0], pos[1]),
                    depth: ground - pos[2],
                    point: [pos[0], pos[1], ground],
                    surface: self.terrain_surface(pos[0], pos[1]),
                    instance: INSTANCE_TERRAIN,
                };
                if !c.normal.iter().all(|v| v.is_finite()) {
                    c.normal = [0.0, 0.0, 1.0];
                }
                out.push(c);
            }
        }

        // Proxies: sample spheres along the capsule axis.
        let aabb = capsule_aabb(pos, r, height);
        let (x0, x1, y0, y1) = self.grid.cell_range(aabb.0, aabb.1);
        for cy in y0..=y1 {
            for cx in x0..=x1 {
                for &id in &self.grid.cells[cy * self.grid.nx + cx] {
                    let ins = &self.instances[id as usize];
                    if !aabb_overlap(aabb.0, aabb.1, ins.min, ins.max) {
                        continue;
                    }
                    for k in 0..=n {
                        let frac = if n == 0 { 0.5 } else { k as f32 / n as f32 };
                        let cz = sample_lo + (sample_hi - sample_lo) * frac;
                        let center = [pos[0], pos[1], cz];
                        if let Some((normal, depth, point)) = sphere_shape_contact(center, r, &ins.shape) {
                            out.push(Contact {
                                normal,
                                depth,
                                point,
                                surface: SURFACE_SHAPE,
                                instance: id,
                            });
                        }
                    }
                }
            }
        }
        out
    }

    /// Deepest capsule overlap, if any.
    pub fn overlap_capsule(&self, pos: [f32; 3], radius: f32, height: f32) -> Option<Contact> {
        self.overlap_capsule_all(pos, radius, height)
            .into_iter()
            .max_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Sweep the capsule by `delta`, sliding out of anything it meets. Returns
    /// the final feet position and the deepest contact encountered.
    pub fn sweep_capsule(
        &self,
        pos: [f32; 3],
        delta: [f32; 3],
        radius: f32,
        height: f32,
    ) -> Sweep {
        let mut p = pos;
        if !pos.iter().all(|v| v.is_finite()) || !delta.iter().all(|v| v.is_finite()) {
            return Sweep { pos, contact: None };
        }
        let len = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        let max_step = (radius * 0.5).max(0.05);
        let n = ((len / max_step).ceil() as usize).clamp(1, 64);
        let mut best: Option<Contact> = None;
        for _ in 0..n {
            p[0] += delta[0] / n as f32;
            p[1] += delta[1] / n as f32;
            p[2] += delta[2] / n as f32;
            if let Some(c) = self.overlap_capsule(p, radius, height) {
                p[0] += c.normal[0] * c.depth;
                p[1] += c.normal[1] * c.depth;
                p[2] += c.normal[2] * c.depth;
                if best.as_ref().is_none_or(|b| c.depth > b.depth) {
                    best = Some(c);
                }
            }
        }
        Sweep { pos: p, contact: best }
    }
}

/// Sphere-centre range for a vertical capsule (`feet .. feet+height`).
fn capsule_sphere_range(pos: [f32; 3], r: f32, height: f32) -> (f32, f32) {
    let lo = pos[2] + r;
    let hi = pos[2] + height - r;
    if hi <= lo {
        let mid = pos[2] + height * 0.5;
        (mid, mid)
    } else {
        (lo, hi)
    }
}

fn capsule_aabb(pos: [f32; 3], r: f32, height: f32) -> ([f32; 3], [f32; 3]) {
    (
        [pos[0] - r, pos[1] - r, pos[2]],
        [pos[0] + r, pos[1] + r, pos[2] + height],
    )
}

fn aabb_overlap(a0: [f32; 3], a1: [f32; 3], b0: [f32; 3], b1: [f32; 3]) -> bool {
    (0..3).all(|i| a0[i] <= b1[i] && a1[i] >= b0[i])
}

// ---------------------------------------------------------------------------
// primitive intersection
// ---------------------------------------------------------------------------

/// Ray vs any proxy shape. Returns `(t, unit normal facing the origin)`.
fn ray_shape(o: [f32; 3], d: [f32; 3], max: f32, shape: &PhysShape) -> Option<(f32, [f32; 3])> {
    match shape {
        PhysShape::Box { center, half_extents, basis } => ray_box(o, d, max, *center, *half_extents, basis),
        PhysShape::Sphere { center, radius } => ray_sphere(o, d, max, *center, *radius),
        PhysShape::Cylinder { center, axis, radius, half_height } => {
            ray_cylinder(o, d, max, *center, *axis, *radius, *half_height)
        }
        PhysShape::Capsule { center, axis, radius, half_height } => {
            ray_capsule(o, d, max, *center, *axis, *radius, *half_height)
        }
        PhysShape::TriMesh { vertices, indices } => ray_trimesh(o, d, max, vertices, indices),
    }
}

fn transpose_apply(b: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        b[0] * v[0] + b[3] * v[1] + b[6] * v[2],
        b[1] * v[0] + b[4] * v[1] + b[7] * v[2],
        b[2] * v[0] + b[5] * v[1] + b[8] * v[2],
    ]
}

fn apply3(b: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        b[0] * v[0] + b[1] * v[1] + b[2] * v[2],
        b[3] * v[0] + b[4] * v[1] + b[5] * v[2],
        b[6] * v[0] + b[7] * v[1] + b[8] * v[2],
    ]
}

fn ray_box(
    o: [f32; 3],
    d: [f32; 3],
    max: f32,
    center: [f32; 3],
    half: [f32; 3],
    basis: &[f32; 9],
) -> Option<(f32, [f32; 3])> {
    let rel = [o[0] - center[0], o[1] - center[1], o[2] - center[2]];
    let lo = transpose_apply(basis, rel);
    let ld = transpose_apply(basis, d);
    let mut t0 = 0.0f32;
    let mut t1 = max;
    let mut n_local = [0.0f32; 3];
    for i in 0..3 {
        if ld[i].abs() < 1e-9 {
            if lo[i] < -half[i] || lo[i] > half[i] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / ld[i];
        let mut ta = (-half[i] - lo[i]) * inv;
        let mut tb = (half[i] - lo[i]) * inv;
        let mut sign = -1.0f32;
        if ta > tb {
            core::mem::swap(&mut ta, &mut tb);
            sign = 1.0;
        }
        if ta > t0 {
            t0 = ta;
            n_local = [0.0; 3];
            n_local[i] = sign;
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
    let t = t0.max(0.0);
    let mut n = apply3(basis, n_local);
    normal_face(&mut n, o, d);
    Some((t, n))
}

fn ray_sphere(o: [f32; 3], d: [f32; 3], max: f32, center: [f32; 3], radius: f32) -> Option<(f32, [f32; 3])> {
    if radius <= 0.0 {
        return None;
    }
    let m = [o[0] - center[0], o[1] - center[1], o[2] - center[2]];
    let a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let b = m[0] * d[0] + m[1] * d[1] + m[2] * d[2];
    let c = m[0] * m[0] + m[1] * m[1] + m[2] * m[2] - radius * radius;
    let disc = b * b - a * c;
    if disc < 0.0 || a <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let mut t = (-b - sq) / a;
    if t < 0.0 {
        t = (-b + sq) / a;
    }
    if t < 0.0 || t > max {
        return None;
    }
    let p = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
    let mut n = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    normal_normalize(&mut n);
    normal_face(&mut n, o, d);
    Some((t, n))
}

fn ray_cylinder(
    o: [f32; 3],
    d: [f32; 3],
    max: f32,
    center: [f32; 3],
    axis: [f32; 3],
    radius: f32,
    half_height: f32,
) -> Option<(f32, [f32; 3])> {
    if radius <= 0.0 {
        return None;
    }
    let m = [o[0] - center[0], o[1] - center[1], o[2] - center[2]];
    let da = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
    let ma = m[0] * axis[0] + m[1] * axis[1] + m[2] * axis[2];
    let dp = [d[0] - axis[0] * da, d[1] - axis[1] * da, d[2] - axis[2] * da];
    let mp = [m[0] - axis[0] * ma, m[1] - axis[1] * ma, m[2] - axis[2] * ma];
    let a = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2];
    let b = 2.0 * (mp[0] * dp[0] + mp[1] * dp[1] + mp[2] * dp[2]);
    let c = mp[0] * mp[0] + mp[1] * mp[1] + mp[2] * mp[2] - radius * radius;

    let mut best: Option<(f32, [f32; 3])> = None;
    if a > 1e-12 {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            for t in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
                if t < 0.0 || t > max {
                    continue;
                }
                let y = ma + t * da;
                if y.abs() <= half_height + 1e-6 {
                    let p = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
                    let q = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                    let qa = q[0] * axis[0] + q[1] * axis[1] + q[2] * axis[2];
                    let mut n = [q[0] - axis[0] * qa, q[1] - axis[1] * qa, q[2] - axis[2] * qa];
                    normal_normalize(&mut n);
                    if best.as_ref().is_none_or(|(bt, _)| t < *bt) {
                        best = Some((t, n));
                    }
                }
            }
        }
    }
    if da.abs() > 1e-9 {
        for (plane, sign) in [(half_height, 1.0f32), (-half_height, -1.0)] {
            let t = (plane - ma) / da;
            if t < 0.0 || t > max {
                continue;
            }
            let p = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
            let q = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
            let qa = q[0] * axis[0] + q[1] * axis[1] + q[2] * axis[2];
            let rp = [q[0] - axis[0] * qa, q[1] - axis[1] * qa, q[2] - axis[2] * qa];
            if rp[0] * rp[0] + rp[1] * rp[1] + rp[2] * rp[2] <= radius * radius + 1e-6 {
                let n = [axis[0] * sign, axis[1] * sign, axis[2] * sign];
                if best.as_ref().is_none_or(|(bt, _)| t < *bt) {
                    best = Some((t, n));
                }
            }
        }
    }
    let (t, mut n) = best?;
    normal_face(&mut n, o, d);
    Some((t, n))
}

fn ray_capsule(
    o: [f32; 3],
    d: [f32; 3],
    max: f32,
    center: [f32; 3],
    axis: [f32; 3],
    radius: f32,
    half_height: f32,
) -> Option<(f32, [f32; 3])> {
    if let Some((t, n)) = ray_cylinder(o, d, max, center, axis, radius, half_height) {
        return Some((t, n));
    }
    let a = [center[0] - axis[0] * half_height, center[1] - axis[1] * half_height, center[2] - axis[2] * half_height];
    let b = [center[0] + axis[0] * half_height, center[1] + axis[1] * half_height, center[2] + axis[2] * half_height];
    let s0 = ray_sphere(o, d, max, a, radius);
    let s1 = ray_sphere(o, d, max, b, radius);
    match (s0, s1) {
        (Some(x), Some(y)) if x.0 <= y.0 => Some(x),
        (Some(_), Some(y)) => Some(y),
        (Some(x), None) => Some(x),
        (None, y) => y,
    }
}

fn ray_trimesh(
    o: [f32; 3],
    d: [f32; 3],
    max: f32,
    vertices: &[[f32; 3]],
    indices: &[u32],
) -> Option<(f32, [f32; 3])> {
    let mut best: Option<(f32, [f32; 3])> = None;
    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (Some(&a), Some(&b), Some(&c)) = (
            vertices.get(tri[0] as usize),
            vertices.get(tri[1] as usize),
            vertices.get(tri[2] as usize),
        ) else {
            continue;
        };
        if let Some((t, n)) = ray_triangle(o, d, max, a, b, c) {
            if best.as_ref().is_none_or(|(bt, _)| t < *bt) {
                best = Some((t, n));
            }
        }
    }
    let (t, mut n) = best?;
    normal_face(&mut n, o, d);
    Some((t, n))
}

fn ray_triangle(
    o: [f32; 3],
    d: [f32; 3],
    max: f32,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
) -> Option<(f32, [f32; 3])> {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let pvec = cross(d, e2);
    let det = dot(e1, pvec);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = [o[0] - a[0], o[1] - a[1], o[2] - a[2]];
    let u = dot(tvec, pvec) * inv;
    if !(-1e-6..=1.0 + 1e-6).contains(&u) {
        return None;
    }
    let qvec = cross(tvec, e1);
    let v = dot(d, qvec) * inv;
    if v < -1e-6 || u + v > 1.0 + 1e-6 {
        return None;
    }
    let t = dot(e2, qvec) * inv;
    if t < 0.0 || t > max {
        return None;
    }
    let mut n = cross(e1, e2);
    normal_normalize(&mut n);
    Some((t, n))
}

// ---------------------------------------------------------------------------
// sphere vs shape (used for the capsule, sampled as spheres)
// ---------------------------------------------------------------------------

fn sphere_shape_contact(center: [f32; 3], r: f32, shape: &PhysShape) -> Option<([f32; 3], f32, [f32; 3])> {
    match shape {
        PhysShape::Box { center: bc, half_extents, basis } => sphere_box(center, r, *bc, *half_extents, basis),
        PhysShape::Sphere { center: sc, radius } => {
            let d = [center[0] - sc[0], center[1] - sc[1], center[2] - sc[2]];
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let sum = r + radius;
            if dist < sum && dist > 1e-9 {
                Some(([d[0] / dist, d[1] / dist, d[2] / dist], sum - dist, center))
            } else {
                None
            }
        }
        PhysShape::Cylinder { center: cc, axis, radius, half_height } => {
            sphere_cylinder(center, r, *cc, *axis, *radius, *half_height)
        }
        PhysShape::Capsule { center: cc, axis, radius, half_height } => {
            sphere_capsule(center, r, *cc, *axis, *radius, *half_height)
        }
        PhysShape::TriMesh { vertices, indices } => {
            let mut best: Option<([f32; 3], f32, [f32; 3])> = None;
            for tri in indices.chunks(3) {
                if tri.len() < 3 {
                    continue;
                }
                let (Some(&a), Some(&b), Some(&c)) = (
                    vertices.get(tri[0] as usize),
                    vertices.get(tri[1] as usize),
                    vertices.get(tri[2] as usize),
                ) else {
                    continue;
                };
                let q = closest_point_triangle(center, a, b, c);
                let d = [center[0] - q[0], center[1] - q[1], center[2] - q[2]];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if dist < r {
                    let depth = r - dist;
                    let n = if dist > 1e-9 {
                        [d[0] / dist, d[1] / dist, d[2] / dist]
                    } else {
                        let fn_ = cross([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
                        let l = (fn_[0] * fn_[0] + fn_[1] * fn_[1] + fn_[2] * fn_[2]).sqrt();
                        if l > 1e-9 { [fn_[0] / l, fn_[1] / l, fn_[2] / l] } else { [0.0, 0.0, 1.0] }
                    };
                    if best.as_ref().is_none_or(|(_, bd, _)| depth > *bd) {
                        best = Some((n, depth, q));
                    }
                }
            }
            best
        }
    }
}

fn sphere_box(
    center: [f32; 3],
    r: f32,
    bc: [f32; 3],
    half: [f32; 3],
    basis: &[f32; 9],
) -> Option<([f32; 3], f32, [f32; 3])> {
    let rel = [center[0] - bc[0], center[1] - bc[1], center[2] - bc[2]];
    let local = transpose_apply(basis, rel);
    let q = [
        local[0].clamp(-half[0], half[0]),
        local[1].clamp(-half[1], half[1]),
        local[2].clamp(-half[2], half[2]),
    ];
    let diff = [local[0] - q[0], local[1] - q[1], local[2] - q[2]];
    let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    if d2 > r * r {
        return None;
    }
    if d2 > 1e-12 {
        let dist = d2.sqrt();
        let n_local = [diff[0] / dist, diff[1] / dist, diff[2] / dist];
        let n = apply3(basis, n_local);
        let point = add3(bc, apply3(basis, q));
        Some((n, r - dist, point))
    } else {
        // Centre inside: push out along the least-penetrated face.
        let mut best_axis = 0usize;
        let mut best_pen = f32::INFINITY;
        let mut sign = 1.0f32;
        for i in 0..3 {
            let pen = half[i] - local[i].abs();
            if pen < best_pen {
                best_pen = pen;
                best_axis = i;
                sign = if local[i] >= 0.0 { 1.0 } else { -1.0 };
            }
        }
        let mut n_local = [0.0f32; 3];
        n_local[best_axis] = sign;
        let n = apply3(basis, n_local);
        Some((n, best_pen + r, center))
    }
}

fn sphere_cylinder(
    center: [f32; 3],
    r: f32,
    cc: [f32; 3],
    axis: [f32; 3],
    radius: f32,
    half_height: f32,
) -> Option<([f32; 3], f32, [f32; 3])> {
    let m = [center[0] - cc[0], center[1] - cc[1], center[2] - cc[2]];
    let ma = dot(m, axis);
    let radial = [m[0] - axis[0] * ma, m[1] - axis[1] * ma, m[2] - axis[2] * ma];
    let d = (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
    let radial_hat = if d > 1e-9 {
        [radial[0] / d, radial[1] / d, radial[2] / d]
    } else {
        // Any perpendicular axis.
        let t = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let c = cross(axis, t);
        let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        if l > 1e-9 { [c[0] / l, c[1] / l, c[2] / l] } else { [0.0, 0.0, 1.0] }
    };
    let inside = d <= radius && ma.abs() <= half_height;
    if inside {
        let pen_side = radius - d;
        let pen_top = half_height - ma;
        let pen_bot = half_height + ma;
        let (n0, depth) = if pen_side <= pen_top && pen_side <= pen_bot {
            (radial_hat, pen_side)
        } else if pen_top <= pen_bot {
            (axis, pen_top)
        } else {
            ([-axis[0], -axis[1], -axis[2]], pen_bot)
        };
        let mut n = n0;
        normal_normalize(&mut n);
        return Some((n, depth + r, center));
    }
    let yc = ma.clamp(-half_height, half_height);
    let q = [
        cc[0] + axis[0] * yc + radial_hat[0] * radius,
        cc[1] + axis[1] * yc + radial_hat[1] * radius,
        cc[2] + axis[2] * yc + radial_hat[2] * radius,
    ];
    let diff = [center[0] - q[0], center[1] - q[1], center[2] - q[2]];
    let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
    if dist < r && dist > 1e-9 {
        Some(([diff[0] / dist, diff[1] / dist, diff[2] / dist], r - dist, q))
    } else {
        None
    }
}

fn sphere_capsule(
    center: [f32; 3],
    r: f32,
    cc: [f32; 3],
    axis: [f32; 3],
    radius: f32,
    half_height: f32,
) -> Option<([f32; 3], f32, [f32; 3])> {
    let m = [center[0] - cc[0], center[1] - cc[1], center[2] - cc[2]];
    let ma = dot(m, axis);
    let yc = ma.clamp(-half_height, half_height);
    let q = [
        cc[0] + axis[0] * yc,
        cc[1] + axis[1] * yc,
        cc[2] + axis[2] * yc,
    ];
    let diff = [center[0] - q[0], center[1] - q[1], center[2] - q[2]];
    let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
    let sum = r + radius;
    if dist < sum {
        let n = if dist > 1e-9 {
            [diff[0] / dist, diff[1] / dist, diff[2] / dist]
        } else {
            [0.0, 0.0, 1.0]
        };
        Some((n, sum - dist, q))
    } else {
        None
    }
}

fn closest_point_triangle(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add(add(a, scale(ab, v)), scale(ac, w))
}

// ---------------------------------------------------------------------------
// small vector helpers
// ---------------------------------------------------------------------------

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    add(a, b)
}
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn normal_normalize(n: &mut [f32; 3]) {
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l.is_finite() && l > 1e-9 {
        n[0] /= l;
        n[1] /= l;
        n[2] /= l;
    } else {
        *n = [0.0, 0.0, 1.0];
    }
}
fn normal_face(n: &mut [f32; 3], o: [f32; 3], d: [f32; 3]) {
    if dot(*n, d) > 0.0 {
        n[0] = -n[0];
        n[1] = -n[1];
        n[2] = -n[2];
    }
    let _ = o;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cry::phys_proxy::{decode_physics_proxies, transform_shape};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    fn box_shape(center: [f32; 3], half: [f32; 3]) -> PhysShape {
        PhysShape::Box { center, half_extents: half, basis: [1., 0., 0., 0., 1., 0., 0., 0., 1.] }
    }

    fn flat(z: f32) -> Terrain {
        Terrain {
            samples: 4,
            spacing: 1.0,
            world_min: [0.0, 0.0, z],
            heights: vec![z; 16],
            surfaces: vec![7; 16],
            min: z,
            max: z,
        }
    }

    #[test]
    fn ray_hits_box_roof_not_terrain() {
        let world = CollisionWorld::from_shapes(Some(flat(0.0)), vec![box_shape([0.0, 0.0, 5.0], [2.0, 2.0, 5.0])]);
        let hit = world.raycast([0.0, 0.0, 50.0], [0.0, 0.0, -1.0], 100.0).expect("hit");
        assert!((hit.point[2] - 10.0).abs() < 1e-3, "z={}", hit.point[2]);
        assert!(hit.normal[2] > 0.99, "n={:?}", hit.normal);
        assert_eq!(hit.surface, SURFACE_SHAPE);
    }

    #[test]
    fn ray_hits_terrain_when_no_shape() {
        let world = CollisionWorld::from_shapes(Some(flat(2.0)), vec![]);
        let hit = world.raycast([0.0, 0.0, 50.0], [0.0, 0.0, -1.0], 100.0).expect("terrain");
        assert!((hit.point[2] - 2.0).abs() < 0.05, "z={}", hit.point[2]);
        assert_eq!(hit.surface, 7);
        assert_eq!(hit.instance, INSTANCE_TERRAIN);
    }

    #[test]
    fn ray_sphere_and_cylinder() {
        let s = PhysShape::Sphere { center: [0.0, 0.0, 3.0], radius: 1.0 };
        let world = CollisionWorld::from_shapes(None, vec![s]);
        let hit = world.raycast([0.0, 0.0, 10.0], [0.0, 0.0, -1.0], 20.0).unwrap();
        assert!((hit.point[2] - 4.0).abs() < 1e-3, "z={}", hit.point[2]);

        let c = PhysShape::Cylinder { center: [0.0, 0.0, 3.0], axis: [0.0, 0.0, 1.0], radius: 1.0, half_height: 2.0 };
        let world = CollisionWorld::from_shapes(None, vec![c]);
        let hit = world.raycast([0.0, 0.0, 10.0], [0.0, 0.0, -1.0], 20.0).unwrap();
        assert!((hit.point[2] - 5.0).abs() < 1e-3, "cap z={}", hit.point[2]);
        // Side hit.
        let hit = world.raycast([-5.0, 0.0, 3.0], [1.0, 0.0, 0.0], 20.0).unwrap();
        assert!((hit.point[0] + 1.0).abs() < 1e-3, "side x={}", hit.point[0]);
    }

    #[test]
    fn ray_trimesh_hit() {
        let vertices = vec![[-1.0f32, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0u32, 1, 2];
        let world = CollisionWorld::from_shapes(None, vec![PhysShape::TriMesh { vertices, indices }]);
        let hit = world.raycast([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 10.0).unwrap();
        assert!((hit.point[2]).abs() < 1e-4, "z={}", hit.point[2]);
    }

    #[test]
    fn capsule_overlaps_wall() {
        let world = CollisionWorld::from_shapes(Some(flat(0.0)), vec![box_shape([2.0, 0.0, 1.0], [0.5, 5.0, 1.0])]);
        let c = world.overlap_capsule([1.9, 0.0, 0.0], 0.4, 1.8).expect("overlap");
        assert!(c.depth > 0.0, "depth={}", c.depth);
        assert!(c.normal[0] < -0.5, "normal={:?}", c.normal);

        let free = world.overlap_capsule([-5.0, 0.0, 0.0], 0.4, 1.8);
        assert!(free.is_none());
    }

    #[test]
    fn capsule_sweep_stops_at_wall() {
        let world = CollisionWorld::from_shapes(Some(flat(0.0)), vec![box_shape([2.0, 0.0, 1.0], [0.5, 5.0, 1.0])]);
        let s = world.sweep_capsule([0.0, 0.0, 0.0], [5.0, 0.0, 0.0], 0.4, 1.8);
        assert!(s.pos[0] < 2.0, "passed through: {}", s.pos[0]);
        assert!(s.contact.is_some());
    }

    #[test]
    fn terrain_capsule_contact() {
        let world = CollisionWorld::from_shapes(Some(flat(1.0)), vec![]);
        let c = world.overlap_capsule([0.0, 0.0, 0.5], 0.4, 1.8).expect("ground");
        assert!((c.depth - 0.5).abs() < 1e-3, "depth={}", c.depth);
        assert!(c.normal[2] > 0.9);
    }

    #[test]
    fn degenerate_is_total() {
        let world = CollisionWorld::from_shapes(None, vec![]);
        assert!(world.raycast([f32::NAN; 3], [0.0, 0.0, -1.0], 10.0).is_none());
        assert!(world.raycast([0.0; 3], [0.0, 0.0, 0.0], 10.0).is_none());
        assert!(world.raycast([0.0; 3], [0.0, 0.0, -1.0], 0.0).is_none());
        assert!(world.overlap_capsule([f32::NAN; 3], 0.4, 1.8).is_none());
        assert!(world.overlap_capsule([0.0; 3], -1.0, 1.8).is_none());
        assert!(world.raycast([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 10.0).is_none());
    }

    // -----------------------------------------------------------------------
    // integration: the real Mt. Tanner area (skips when the paks are absent)
    // -----------------------------------------------------------------------

    fn game_dir() -> Option<PathBuf> {
        let p = std::env::var("SOD_GAME")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE",
                )
            });
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    }

    fn collect_paks(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_paks(&p, out);
            } else if p.extension().map(|x| x == "pak").unwrap_or(false) {
                out.push(p);
            }
        }
    }

    /// Build the Mt. Tanner (RTSPoint_PlayerStart = 3246.8, 3256.2, 217.1,
    /// radius 200 m) collision world and confirm a downward ray over a known
    /// building hits its roof, not the terrain. Also times the build and 10k
    /// raycasts. Silently returns when the shipped paks are not mounted.
    #[test]
    fn mt_tanner_area_build_and_roof_ray() {
        let Some(game) = game_dir() else { return };
        let mut paks = Vec::new();
        collect_paks(&game.join("Game"), &mut paks);
        paks.sort();
        if paks.is_empty() {
            return;
        }
        let mut vfs = Vfs::new();
        for p in &paks {
            let _ = vfs.mount(p);
        }

        let level_pak = game.join("Game/levels/Class3/level.pak");
        let Ok(level) = crate::crypak::Pak::open(&level_pak) else { return };
        let Some(terrain_bytes) =
            level.find("terrain/terrain.dat").and_then(|i| level.read_idx(i).ok())
        else {
            return;
        };
        let terrain = crate::terrain::decode_terrain(&terrain_bytes).expect("Class3 terrain");

        let start = [3246.8f32, 3256.2, 217.1];
        let brushes = crate::architecture::brushes_in_region(&terrain_bytes, start, 200.0);
        if brushes.is_empty() {
            return;
        }
        let specs: Vec<InstanceSpec> = brushes
            .iter()
            .map(|b| InstanceSpec {
                model: b.model.clone(),
                pos: b.pos,
                rot: b.rot,
                scale: [b.scale; 3],
            })
            .collect();

        let t_build = Instant::now();
        let world = CollisionWorld::build(Some(terrain), &vfs, &specs);
        let build_ms = t_build.elapsed().as_secs_f32() * 1000.0;
        assert!(world.instance_count() > 0, "no collision proxies decoded");

        // Pick a known tall building: decode each brush's proxies (cached) and
        // choose the proxy whose world-space top rises highest above its own
        // footprint, then cast down through that footprint's centre.
        let mut cache: HashMap<String, Vec<phys_proxy::PhysicsProxy>> = HashMap::new();
        let mut roof: Option<([f32; 2], f32, f32)> = None;
        for b in &brushes {
            let proxies = cache.entry(b.model.clone()).or_insert_with(|| {
                vfs.read(&b.model)
                    .map(|x| decode_physics_proxies(&x))
                    .unwrap_or_default()
            });
            let t = Transform::from_trs(b.pos, b.rot, [b.scale; 3]);
            for p in proxies.iter() {
                if !p.collide {
                    continue;
                }
                let (mn, mx) = transform_shape(p.shape.clone(), &t).aabb();
                let cx = (mn[0] + mx[0]) * 0.5;
                let cy = (mn[1] + mx[1]) * 0.5;
                let ground = world.terrain_height(cx, cy).unwrap_or(0.0);
                // A building-like proxy: clearly above ground but not one of
                // the huge skybox/terrain-cover proxies.
                if mx[2] > ground + 4.0 && mx[2] < ground + 50.0 && roof.is_none() {
                    roof = Some(([cx, cy], mx[2], ground));
                }
            }
        }
        let (xy, top, ground) = roof.expect("a building taller than 4 m in the zone");

        let origin = [xy[0], xy[1], top + 25.0];
        let max = top + 25.0 - ground + 5.0;
        let hit = world
            .raycast(origin, [0.0, 0.0, -1.0], max)
            .expect("downward roof ray");
        assert_eq!(hit.surface, SURFACE_SHAPE, "roof should be an architectural proxy");
        assert!(
            hit.point[2] > ground + 2.0,
            "roof ray hit at z={} (ground {})",
            hit.point[2],
            ground
        );
        assert!(
            (hit.point[2] - top).abs() < 3.0,
            "expected roof near z={top}, hit z={}",
            hit.point[2]
        );

        // Performance: 10k downward rays over the zone.
        let t_rays = Instant::now();
        let mut hits = 0u32;
        for i in 0..10_000u32 {
            let a = (i as f32) * 0.7;
            let x = start[0] + (a.cos() * (i as f32 % 190.0));
            let y = start[1] + (a.sin() * (i as f32 % 190.0));
            if world.raycast([x, y, 400.0], [0.0, 0.0, -1.0], 400.0).is_some() {
                hits += 1;
            }
        }
        let ray_ms = t_rays.elapsed().as_secs_f32() * 1000.0;
        eprintln!(
            "[phys] Mt. Tanner: {} brushes -> {} instances, build {:.1} ms, 10k rays {:.1} ms ({}/10000 hit)",
            brushes.len(),
            world.instance_count(),
            build_ms,
            ray_ms,
            hits
        );
        assert!(hits > 0, "no downward ray hit the world");
    }
}
