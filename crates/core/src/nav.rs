//! Walkability grid + A* pathfinding.
//!
//! Actors currently steer in a straight line and snag on buildings. This module
//! bakes the terrain heightfield plus the static `physics::Aabb` obstacles into
//! a coarse cell grid, then answers navigational queries:
//!
//! * [`build`] samples a caller `height(x, y)` closure, rejects cells on steep
//!   local slopes, and rejects cells whose footprint (grown by `agent_radius`)
//!   overlaps an obstacle. The grid is capped at [`MAX_NODES`] cells.
//! * [`NavGrid::path`] runs 8-connected A* (diagonal cost `sqrt 2`, no corner
//!   cutting) and then string-pulls the result with a line-of-sight test so a
//!   clear straight run collapses to a couple of waypoints.
//!
//! Coordinates are Z-up: `x`/`y` are the ground plane and `z` is height. All
//! entry points are total — empty or out-of-range input yields `false`/`None`,
//! and no query can panic.

use crate::physics::Aabb;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f32::consts::SQRT_2;

/// Hard cap on `width * height`, so a pathological `cell`/`size` cannot allocate
/// an unbounded grid. Cell size is grown (doubled) until the grid fits.
pub const MAX_NODES: usize = 2_000_000;
/// Smallest accepted cell size. Smaller inputs are clamped up to this.
const MIN_CELL: f32 = 0.25;
/// Chebyshev radius (in cells) searched when snapping a query point.
const SNAP_CELLS: i32 = 8;
/// Vertical extent assumed for an agent when testing obstacle overlap.
const AGENT_HEIGHT: f64 = 2.0;

/// A baked walkability grid.
#[derive(Debug, Clone)]
pub struct NavGrid {
    origin: [f32; 2],
    size: [f32; 2],
    cell: f32,
    w: usize,
    h: usize,
    walkable: Vec<bool>,
    heights: Vec<f32>,
}

/// Node in the A* open set; ordered so `BinaryHeap` pops the lowest `f` first.
#[derive(PartialEq)]
struct Open {
    f: f32,
    idx: usize,
}

impl Eq for Open {}

impl Ord for Open {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .partial_cmp(&self.f)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Grid dimensions for a world extent and cell size; `0` if degenerate.
fn dims(sx: f64, sy: f64, cell: f64) -> (usize, usize) {
    if !(sx > 0.0) || !(sy > 0.0) || !(cell > 0.0) {
        return (0, 0);
    }
    let w = (sx / cell).ceil();
    let h = (sy / cell).ceil();
    if !w.is_finite() || !h.is_finite() {
        return (0, 0);
    }
    (w as usize, h as usize)
}

/// Bake a [`NavGrid`].
///
/// `origin` is the minimum `(x, y)` corner, `size` the extent in metres. A cell
/// is unwalkable when its sampled height is unavailable (non-finite), when the
/// slope to any of its 4 neighbours exceeds `max_slope_deg`, or when its square
/// footprint grown by `agent_radius` overlaps an obstacle at the cell's ground
/// height. `cell` is clamped to at least [`MIN_CELL`] and grown until the grid
/// is at most [`MAX_NODES`] cells.
pub fn build(
    origin: [f32; 2],
    size: [f32; 2],
    cell: f32,
    height: &dyn Fn(f32, f32) -> f32,
    obstacles: &[Aabb],
    max_slope_deg: f32,
    agent_radius: f32,
) -> NavGrid {
    let ox = if origin[0].is_finite() { origin[0] } else { 0.0 };
    let oy = if origin[1].is_finite() { origin[1] } else { 0.0 };
    let sx = if size[0].is_finite() && size[0] > 0.0 { size[0] as f64 } else { 0.0 };
    let sy = if size[1].is_finite() && size[1] > 0.0 { size[1] as f64 } else { 0.0 };

    let mut cell = if cell.is_finite() && cell > 0.0 { cell.max(MIN_CELL) } else { 1.0 };
    let (mut w, mut h) = dims(sx, sy, cell as f64);
    let mut guard = 0;
    while (w as u128) * (h as u128) > MAX_NODES as u128 && guard < 128 {
        cell *= 2.0;
        let d = dims(sx, sy, cell as f64);
        w = d.0;
        h = d.1;
        guard += 1;
    }
    let empty = !cell.is_finite()
        || (w as u128) * (h as u128) > MAX_NODES as u128
        || w == 0
        || h == 0;
    if empty {
        return NavGrid {
            origin: [ox, oy],
            size: [sx as f32, sy as f32],
            cell,
            w: 0,
            h: 0,
            walkable: Vec::new(),
            heights: Vec::new(),
        };
    }

    let cell64 = cell as f64;
    let n = w * h;
    let mut heights = vec![0f32; n];
    let mut avail = vec![false; n];
    let mut walkable = vec![true; n];

    for cy in 0..h {
        let wy = oy as f64 + (cy as f64 + 0.5) * cell64;
        for cx in 0..w {
            let wx = ox as f64 + (cx as f64 + 0.5) * cell64;
            let idx = cy * w + cx;
            let hv = height(wx as f32, wy as f32);
            if hv.is_finite() {
                heights[idx] = hv;
                avail[idx] = true;
            } else {
                walkable[idx] = false;
            }
        }
    }

    // Slope to the 4 orthogonal neighbours.
    let tan_limit = if max_slope_deg.is_finite() {
        let d = max_slope_deg.clamp(0.0, 90.0) as f64;
        if d >= 90.0 {
            f64::INFINITY
        } else {
            d.to_radians().tan()
        }
    } else {
        f64::INFINITY
    };
    let max_dh = tan_limit * cell64;
    for cy in 0..h {
        for cx in 0..w {
            let idx = cy * w + cx;
            if !walkable[idx] {
                continue;
            }
            let hc = heights[idx] as f64;
            let mut steep = false;
            for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let nidx = ny as usize * w + nx as usize;
                if !avail[nidx] {
                    continue;
                }
                if (heights[nidx] as f64 - hc).abs() > max_dh {
                    steep = true;
                    break;
                }
            }
            if steep {
                walkable[idx] = false;
            }
        }
    }

    // Obstacle overlap (footprint grown by agent_radius, at cell ground height).
    if !obstacles.is_empty() {
        let radius = if agent_radius.is_finite() && agent_radius > 0.0 {
            agent_radius as f64
        } else {
            0.0
        };
        for cy in 0..h {
            let y0 = oy as f64 + cy as f64 * cell64 - radius;
            let y1 = oy as f64 + (cy as f64 + 1.0) * cell64 + radius;
            for cx in 0..w {
                let idx = cy * w + cx;
                if !walkable[idx] {
                    continue;
                }
                let x0 = ox as f64 + cx as f64 * cell64 - radius;
                let x1 = ox as f64 + (cx as f64 + 1.0) * cell64 + radius;
                let hc = heights[idx] as f64;
                for ob in obstacles {
                    let (ax0, ax1) = (ob.min[0] as f64, ob.max[0] as f64);
                    let (ay0, ay1) = (ob.min[1] as f64, ob.max[1] as f64);
                    let (az0, az1) = (ob.min[2] as f64, ob.max[2] as f64);
                    let horizontal =
                        ax0 <= x1 && ax1 >= x0 && ay0 <= y1 && ay1 >= y0;
                    let vertical = az0 <= hc + AGENT_HEIGHT && az1 >= hc;
                    if horizontal && vertical {
                        walkable[idx] = false;
                        break;
                    }
                }
            }
        }
    }

    NavGrid {
        origin: [ox, oy],
        size: [sx as f32, sy as f32],
        cell,
        w,
        h,
        walkable,
        heights,
    }
}

impl NavGrid {
    /// Cell indices for `x`/`y`, or `None` when out of range / degenerate.
    fn cell_of(&self, p: [f32; 3]) -> Option<usize> {
        if self.w == 0 || self.h == 0 || !(self.cell > 0.0) {
            return None;
        }
        let fx = ((p[0] - self.origin[0]) / self.cell).floor();
        let fy = ((p[1] - self.origin[1]) / self.cell).floor();
        if !fx.is_finite() || !fy.is_finite() || fx < 0.0 || fy < 0.0 {
            return None;
        }
        let cx = fx as usize;
        let cy = fy as usize;
        if cx >= self.w || cy >= self.h {
            return None;
        }
        Some(cy * self.w + cx)
    }

    /// Is the cell containing `p` walkable? `p`'s `z` is ignored (obstacles are
    /// already baked in) and out-of-range points are not walkable.
    pub fn walkable(&self, p: [f32; 3]) -> bool {
        match self.cell_of(p) {
            Some(i) => self.walkable.get(i).copied().unwrap_or(false),
            None => false,
        }
    }

    /// Nearest walkable cell to `p` within [`SNAP_CELLS`] cells, snapping out of
    /// obstacles. Returns `None` for an empty grid or when nothing is close.
    pub fn nearest_walkable(&self, p: [f32; 3]) -> Option<[usize; 2]> {
        if self.w == 0 || self.h == 0 || !(self.cell > 0.0) {
            return None;
        }
        let fx = ((p[0] - self.origin[0]) / self.cell).floor();
        let fy = ((p[1] - self.origin[1]) / self.cell).floor();
        let fx = if fx.is_finite() { fx } else { 0.0 };
        let fy = if fy.is_finite() { fy } else { 0.0 };
        let cx0 = (fx as isize).clamp(0, self.w as isize - 1);
        let cy0 = (fy as isize).clamp(0, self.h as isize - 1);

        let px = p[0] as f64;
        let py = p[1] as f64;
        // Snap window is centred on the clamped cell, so also cap the true
        // distance: a point far outside the grid has no walkable cell "nearby".
        let max_d2 = ((SNAP_CELLS as f64 + 0.5) * self.cell as f64).powi(2);
        let mut best: Option<(f64, usize, usize)> = None;
        for dy in -SNAP_CELLS..=SNAP_CELLS {
            let ny = cy0 + dy as isize;
            if ny < 0 || ny >= self.h as isize {
                continue;
            }
            for dx in -SNAP_CELLS..=SNAP_CELLS {
                let nx = cx0 + dx as isize;
                if nx < 0 || nx >= self.w as isize {
                    continue;
                }
                let nx = nx as usize;
                let ny = ny as usize;
                if !self.walkable[ny * self.w + nx] {
                    continue;
                }
                let c = self.cell_center(nx, ny);
                let d2 = (c[0] as f64 - px).powi(2) + (c[1] as f64 - py).powi(2);
                if best.map_or(true, |(bd, _, _)| d2 < bd) {
                    best = Some((d2, nx, ny));
                }
            }
        }
        best.filter(|(d2, _, _)| *d2 <= max_d2).map(|(_, x, y)| [x, y])
    }

    /// World-space centre of cell `(cx, cy)`, with the stored terrain height.
    /// Out-of-range coordinates clamp to the grid and yield `z = 0.0`.
    pub fn cell_center(&self, cx: usize, cy: usize) -> [f32; 3] {
        let x = self.origin[0] + (cx as f32 + 0.5) * self.cell;
        let y = self.origin[1] + (cy as f32 + 0.5) * self.cell;
        let z = if self.w > 0 && self.h > 0 && cx < self.w && cy < self.h {
            self.heights[cy * self.w + cx]
        } else {
            0.0
        };
        [x, y, z]
    }

    /// Grid dimensions `(width, height)` in cells.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// Cell size in metres.
    pub fn cell_size(&self) -> f32 {
        self.cell
    }

    /// World extent `[x, y]` in metres (kept for callers that size overlays).
    pub fn size(&self) -> [f32; 2] {
        self.size
    }

    /// Line of sight for smoothing: every sample along the horizontal segment
    /// must land in a walkable cell.
    fn los(&self, a: [f32; 3], b: [f32; 3]) -> bool {
        if self.w == 0 || self.h == 0 {
            return false;
        }
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let d = (dx * dx + dy * dy).sqrt();
        let step = (self.cell * 0.5).max(1e-3);
        let n = ((d / step).ceil() as usize).clamp(1, 100_000);
        for k in 0..=n {
            let t = k as f32 / n as f32;
            let p = [a[0] + dx * t, a[1] + dy * t, a[2] + (b[2] - a[2]) * t];
            if !self.walkable(p) {
                return false;
            }
        }
        true
    }

    /// Greedy string-pull: keep the first point, then emit a corner only when
    /// the next line to the current anchor would be blocked.
    fn smooth(&self, pts: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let n = pts.len();
        if n <= 2 {
            return pts.to_vec();
        }
        let mut out = Vec::with_capacity(n);
        out.push(pts[0]);
        let mut anchor = 0usize;
        let mut i = 2usize;
        while i < n {
            if !self.los(pts[anchor], pts[i]) {
                out.push(pts[i - 1]);
                anchor = i - 1;
            }
            i += 1;
        }
        out.push(pts[n - 1]);
        out
    }

    /// Octile distance heuristic for 8-connected movement.
    fn hcost(a: [usize; 2], b: [usize; 2]) -> f32 {
        let dx = (a[0] as isize - b[0] as isize).unsigned_abs() as f32;
        let dy = (a[1] as isize - b[1] as isize).unsigned_abs() as f32;
        let (lo, hi) = if dx < dy { (dx, dy) } else { (dy, dx) };
        hi + (SQRT_2 - 1.0) * lo
    }

    fn reconstruct(&self, came: &[usize], start: [usize; 2], goal: [usize; 2]) -> Vec<[usize; 2]> {
        let s = start[1] * self.w + start[0];
        let mut cur = goal[1] * self.w + goal[0];
        let mut out = Vec::new();
        while cur != s {
            out.push([cur % self.w, cur / self.w]);
            let p = came[cur];
            if p == usize::MAX {
                break;
            }
            cur = p;
        }
        out.push([start[0], start[1]]);
        out.reverse();
        out
    }

    /// Raw 8-connected A* (no smoothing). `None` when no route exists.
    fn astar(&self, start: [usize; 2], goal: [usize; 2]) -> Option<Vec<[usize; 2]>> {
        let (w, h) = (self.w, self.h);
        if w == 0 || h == 0 {
            return None;
        }
        let s = start[1] * w + start[0];
        let g = goal[1] * w + goal[0];
        if s >= self.walkable.len() || g >= self.walkable.len() {
            return None;
        }
        if !self.walkable[s] || !self.walkable[g] {
            return None;
        }
        if s == g {
            return Some(vec![[start[0], start[1]]]);
        }

        let n = w * h;
        let mut gs = vec![f32::INFINITY; n];
        let mut came = vec![usize::MAX; n];
        let mut closed = vec![false; n];
        gs[s] = 0.0;
        let mut heap = BinaryHeap::new();
        heap.push(Open { f: Self::hcost(start, goal), idx: s });

        const NEIGH: [(isize, isize); 8] =
            [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)];

        while let Some(Open { idx, .. }) = heap.pop() {
            if idx == g {
                return Some(self.reconstruct(&came, start, goal));
            }
            if closed[idx] {
                continue;
            }
            closed[idx] = true;
            let cx = idx % w;
            let cy = idx / w;
            for (dx, dy) in NEIGH {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let nx = nx as usize;
                let ny = ny as usize;
                let ni = ny * w + nx;
                if !self.walkable[ni] {
                    continue;
                }
                // No corner cutting: both orthogonal neighbours must be open.
                if dx != 0 && dy != 0 {
                    let ortho_x = cy * w + (cx as isize + dx) as usize;
                    let ortho_y = (cy as isize + dy) as usize * w + cx;
                    if !self.walkable[ortho_x] || !self.walkable[ortho_y] {
                        continue;
                    }
                }
                let cost = if dx != 0 && dy != 0 { SQRT_2 } else { 1.0 };
                let tentative = gs[idx] + cost;
                if tentative < gs[ni] {
                    gs[ni] = tentative;
                    came[ni] = idx;
                    let f = tentative + Self::hcost([nx, ny], goal);
                    heap.push(Open { f, idx: ni });
                }
            }
        }
        None
    }

    /// Route from `from` to `to`, returning smoothed world-space waypoints
    /// (start first, goal last). `None` when either endpoint has no walkable cell
    /// within [`SNAP_CELLS`], or when no route exists.
    pub fn path(&self, from: [f32; 3], to: [f32; 3]) -> Option<Vec<[f32; 3]>> {
        let start = self.nearest_walkable(from)?;
        let goal = self.nearest_walkable(to)?;

        let dx = to[0] - from[0];
        let dy = to[1] - from[1];
        let dz = to[2] - from[2];
        if dx * dx + dy * dy + dz * dz < 1e-6 {
            return Some(vec![from]);
        }
        if start == goal {
            return Some(vec![from, to]);
        }

        let cells = self.astar(start, goal)?;
        let pts: Vec<[f32; 3]> =
            cells.iter().map(|&[cx, cy]| self.cell_center(cx, cy)).collect();
        let mut out = self.smooth(&pts);
        if let Some(first) = out.first_mut() {
            *first = from;
        }
        if let Some(last) = out.last_mut() {
            *last = to;
        }
        Some(out)
    }

    /// Total 3D length of a polyline.
    pub fn path_len(&self, path: &[[f32; 3]]) -> f32 {
        let mut total = 0.0f32;
        for seg in path.windows(2) {
            let d = [seg[1][0] - seg[0][0], seg[1][1] - seg[0][1], seg[1][2] - seg[0][2]];
            total += (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        }
        total
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(_x: f32, _y: f32) -> f32 {
        0.0
    }

    fn box_at(min: [f32; 3], max: [f32; 3]) -> Aabb {
        Aabb { min, max }
    }

    fn flat_grid(obstacles: &[Aabb]) -> NavGrid {
        build([0.0, 0.0], [64.0, 64.0], 1.0, &flat, obstacles, 45.0, 0.5)
    }

    fn assert_path_walkable(g: &NavGrid, path: &[[f32; 3]]) {
        for w in path {
            assert!(g.walkable(*w), "waypoint in unwalkable cell: {w:?}");
        }
        for seg in path.windows(2) {
            for k in 0..=8 {
                let t = k as f32 / 8.0;
                let p = [
                    seg[0][0] + (seg[1][0] - seg[0][0]) * t,
                    seg[0][1] + (seg[1][1] - seg[0][1]) * t,
                    seg[0][2] + (seg[1][2] - seg[0][2]) * t,
                ];
                assert!(g.walkable(p), "path crosses an unwalkable cell at {p:?}");
            }
        }
    }

    #[test]
    fn flat_straight_path() {
        let g = flat_grid(&[]);
        let from = [4.0, 4.0, 0.0];
        let to = [60.0, 60.0, 0.0];
        let path = g.path(from, to).expect("flat ground is routable");
        assert_path_walkable(&g, &path);
        assert!(path.len() >= 2);
        assert!((path[0][0] - from[0]).abs() < 1e-3 && (path[0][1] - from[1]).abs() < 1e-3);
        let last = path[path.len() - 1];
        assert!((last[0] - to[0]).abs() < 1e-3 && (last[1] - to[1]).abs() < 1e-3);

        let py = to[1] - from[1];
        let px = to[0] - from[0];
        let straight = (px * px + py * py).sqrt();
        let pl = g.path_len(&path);
        assert!((pl - straight).abs() < 0.5, "len {pl} vs straight {straight}");
        // A clear run should string-pull down to a couple of waypoints.
        assert!(path.len() <= 3, "smoothing should collapse, got {}", path.len());
    }

    #[test]
    fn path_around_wall_is_longer_and_clear() {
        // Wall blocks y in [0, 50); the gap is at the top.
        let wall = box_at([28.0, 0.0, 0.0], [32.0, 50.0, 10.0]);
        let g = flat_grid(&[wall]);
        let from = [10.0, 10.0, 0.0];
        let to = [50.0, 10.0, 0.0];
        let path = g.path(from, to).expect("there is a route around the wall");
        assert_path_walkable(&g, &path);
        let straight = to[0] - from[0];
        assert!(
            g.path_len(&path) > straight,
            "detour {} should exceed straight {straight}",
            g.path_len(&path)
        );
        // It must actually leave the wall's y-range to get past the barrier.
        assert!(path.iter().any(|p| p[1] > 50.0), "path should round the wall end");
    }

    #[test]
    fn fully_blocking_wall_is_unreachable() {
        let wall = box_at([30.0, -1.0, 0.0], [32.0, 65.0, 10.0]);
        let g = flat_grid(&[wall]);
        assert!(g.path([10.0, 32.0, 0.0], [50.0, 32.0, 0.0]).is_none());
    }

    #[test]
    fn nearest_walkable_snaps_out_of_obstacle() {
        let block = box_at([10.0, 10.0, 0.0], [14.0, 14.0, 10.0]);
        let g = flat_grid(&[block]);
        let inside = [12.0, 12.0, 0.0];
        assert!(!g.walkable(inside), "obstacle interior is unwalkable");
        let cell = g.nearest_walkable(inside).expect("snaps to a nearby cell");
        let centre = g.cell_center(cell[0], cell[1]);
        assert!(g.walkable(centre), "snapped cell is walkable");
        assert!(!g.walkable(inside));
        // And a route out of the obstacle is still found.
        assert!(g.path(inside, [40.0, 40.0, 0.0]).is_some());
    }

    #[test]
    fn steep_hill_cells_are_unwalkable() {
        let cliff = |x: f32, _y: f32| if x > 16.0 { 10.0 } else { 0.0 };
        let g = build([0.0, 0.0], [32.0, 32.0], 1.0, &cliff, &[], 20.0, 0.5);
        let (w, h) = g.dimensions();
        assert!(w > 0 && h > 0);
        let mut walk = 0;
        let mut unwalk = 0;
        for cy in 0..h {
            for cx in 0..w {
                if g.walkable(g.cell_center(cx, cy)) {
                    walk += 1;
                } else {
                    unwalk += 1;
                }
            }
        }
        assert!(walk > 0, "flat regions stay walkable");
        assert!(unwalk > 0, "the cliff strip is unwalkable");
        // The cliff spans every row, so no route crosses it.
        assert!(g.path([5.0, 5.0, 0.0], [27.0, 5.0, 0.0]).is_none());
    }

    #[test]
    fn smoothing_reduces_waypoint_count() {
        let g = flat_grid(&[]);
        let from = [0.5, 0.5, 0.0];
        let to = [40.5, 40.5, 0.0];
        let smoothed = g.path(from, to).expect("route");
        let s = g.nearest_walkable(from).unwrap();
        let e = g.nearest_walkable(to).unwrap();
        let raw = g.astar(s, e).expect("raw route");
        assert!(
            smoothed.len() < raw.len(),
            "smoothed {} should be shorter than raw {}",
            smoothed.len(),
            raw.len()
        );
    }

    #[test]
    fn empty_and_out_of_range_are_total() {
        let g = build([0.0, 0.0], [0.0, 0.0], 1.0, &flat, &[], 45.0, 0.5);
        assert_eq!(g.dimensions(), (0, 0));
        assert!(!g.walkable([0.0, 0.0, 0.0]));
        assert!(g.nearest_walkable([0.0, 0.0, 0.0]).is_none());
        assert!(g.path([0.0, 0.0, 0.0], [1.0, 1.0, 0.0]).is_none());
        let _ = g.cell_center(5, 5);
        assert_eq!(g.path_len(&[]), 0.0);
        assert!(g.path_len(&[[1.0, 1.0, 0.0]]) == 0.0);

        // Huge extent: the node cap grows the cell instead of exploding.
        let g2 = build([0.0, 0.0], [1.0e9, 1.0e9], 0.01, &flat, &[], 45.0, 0.5);
        let (w, h) = g2.dimensions();
        assert!((w as u128) * (h as u128) <= MAX_NODES as u128);

        // Non-finite heights are treated as unavailable.
        let hole = |x: f32, _y: f32| if x > 30.0 { f32::NAN } else { 0.0 };
        let gh = build([0.0, 0.0], [64.0, 64.0], 1.0, &hole, &[], 45.0, 0.5);
        assert!(!gh.walkable([40.0, 10.0, 0.0]));
        assert!(gh.walkable([10.0, 10.0, 0.0]));

        // Out-of-range queries never panic.
        assert!(!g2.walkable([-100.0, -100.0, 0.0]));
        assert!(g2.nearest_walkable([f32::MAX, f32::MAX, 0.0]).is_none());
        assert!(g2.path([1.0e12, 1.0e12, 0.0], [2.0e12, 2.0e12, 0.0]).is_none());
    }
}
