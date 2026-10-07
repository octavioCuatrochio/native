//! Undead Labs `.bai` AI navigation mesh: reader + A* path query.
//!
//! The level ships a family of `<tag>mission0.bai` files built by the loader at
//! `0x0095ee50` (string table `%s/net%s.bai`, `%s/verts%s.bai`, ... at
//! `0x01788874..`). For ground agents the two that matter are:
//!
//! * `verts*.bai`  - a flat vertex array (tag `1`, `u32 count`, then `count`
//!   records of 32 bytes; the first 12 bytes are the world-space `f32` position,
//!   read by `0x0092a4c0`: exact `fread(count*0x20)` and per-record
//!   `src+0x00/0x04/0x08 -> x/y/z`).
//! * `net*.bai`    - the triangular navigation graph (`NAV_TRIANGULAR`), tag
//!   `0x36` (`0x00901820` checks `== 0x36`). The file is exactly
//!   `4 + 12 + 12 + 4 + N*60 + 4 + L*44` bytes:
//!
//! ```text
//! u32   0x36                       // tag / version (FUN_00901820)
//! f32[3] min                       // bounds
//! f32[3] max
//! u32   node_count  = 88252
//!        node_count * 60-byte nodes
//! u32   link_count  = 266105
//!        link_count * 44-byte links
//! ```
//!
//! The 60-byte node layout is fixed by `FUN_00901820`, which copies 15 dwords
//! and then switches on the `u16` at `+0x38` (2 = triangular). For a triangular
//! node the three `u32` vertex indices live at `+0x2c/+0x30/+0x34` (they index
//! `verts*.bai`, case 2 of the switch) and the triangle representative point
//! (centroid-ish) at `+0x1c .. +0x24`. Field `+0x00` is a 1-based self id.
//!
//! The 44-byte link records give the adjacency graph directly (`from`/`to`
//! 1-based node ids + the shared-edge portal position). The loader reads them as
//! `count * 0x2c`; they are essentially symmetric (266,104 of 266,105 records
//! have their reverse) with ~3 neighbours per node, as expected for a
//! triangle mesh.
//!
//! [`NavMesh::find_path`] runs A* over this graph (cost = 3D distance between
//! node representative points, heuristic = 3D straight-line distance to the
//! goal: both admissible and consistent) and then greedily string-pulls the
//! result across the shared portal edges so clear runs collapse to straight
//! segments. The returned polyline is in CryEngine space (Z-up, metres).
//!
//! Nav type constants are the engine's bit flags. `net*.bai` nodes carry
//! `2 == NAV_TRIANGULAR`; the same flags are used as the `agent_type` bitmask.

use std::collections::BinaryHeap;

/// Nav flags recovered from the loader switch (`0x00901820`) and the
/// `NAV_*` strings at `.rdata` `0x01788ae8..`.
pub const NAV_UNSET: u32 = 1;
pub const NAV_TRIANGULAR: u32 = 2;
pub const NAV_WAYPOINT_HUMAN: u32 = 4;
pub const NAV_WAYPOINT_3DSURFACE: u32 = 8;
pub const NAV_FLIGHT: u32 = 16;
pub const NAV_VOLUME: u32 = 32;
pub const NAV_ROAD: u32 = 64;
pub const NAV_SMARTOBJECT: u32 = 128;
pub const NAV_FREE_2D: u32 = 256;

const NET_TAG: u32 = 0x36;
const VERT_TAG: u32 = 1;
const NODE_SIZE: usize = 60;
const LINK_SIZE: usize = 44;

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// One navigation graph node. Triangular nodes (`nav_type & NAV_TRIANGULAR`)
/// reference three `verts*.bai` vertices and carry a representative point.
#[derive(Clone, Debug, PartialEq)]
pub struct NavNode {
    /// 1-based id as stored in the file.
    pub id: u32,
    /// Nav type bit flags (`+0x38`).
    pub nav_type: u16,
    /// Small subtype/flags byte (`+0x3a`).
    pub flags: u8,
    /// Representative point (`+0x1c`), world space.
    pub center: [f32; 3],
    /// Indices into the vertex array (valid for `NAV_TRIANGULAR`).
    pub tri: [u32; 3],
}

/// A directed adjacency record (one per direction of a shared edge, plus a few
/// off-mesh links). References 1-based node ids.
#[derive(Clone, Debug, PartialEq)]
pub struct NavLink {
    pub from: u32,
    pub to: u32,
    /// Shared-edge portal point (`+0x08`).
    pub portal: [f32; 3],
    /// Extra per-link weights stored at `+0x14`/`+0x18`.
    pub a: f32,
    pub b: f32,
}

/// Raw `net*.bai` contents.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NetData {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub nodes: Vec<NavNode>,
    pub links: Vec<NavLink>,
}

impl NetData {
    /// Number of nodes carrying `NAV_TRIANGULAR`.
    pub fn triangle_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.nav_type as u32 & NAV_TRIANGULAR != 0)
            .count()
    }
}

/// Parse `net*.bai`. Returns `None` on bad tag or truncation; never panics.
pub fn parse_net(b: &[u8]) -> Option<NetData> {
    if b.len() < 32 || rd_u32(b, 0) != NET_TAG {
        return None;
    }
    let mut min = [0.0f32; 3];
    let mut max = [0.0f32; 3];
    for i in 0..3 {
        min[i] = rd_f32(b, 4 + i * 4);
        max[i] = rd_f32(b, 16 + i * 4);
    }
    let node_count = rd_u32(b, 28) as usize;
    let node_bytes = node_count.checked_mul(NODE_SIZE)?;
    let mut o = 32usize;
    if b.len() < o.checked_add(node_bytes)? {
        return None;
    }
    let mut nodes = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let r = o + i * NODE_SIZE;
        nodes.push(NavNode {
            id: rd_u32(b, r),
            nav_type: rd_u16(b, r + 0x38),
            flags: b[r + 0x3a],
            center: [rd_f32(b, r + 0x1c), rd_f32(b, r + 0x20), rd_f32(b, r + 0x24)],
            tri: [rd_u32(b, r + 0x2c), rd_u32(b, r + 0x30), rd_u32(b, r + 0x34)],
        });
    }
    o += node_bytes;
    if b.len() < o + 4 {
        return None;
    }
    let link_count = rd_u32(b, o) as usize;
    o += 4;
    let link_bytes = link_count.checked_mul(LINK_SIZE)?;
    if b.len() < o.checked_add(link_bytes)? {
        return None;
    }
    let mut links = Vec::with_capacity(link_count);
    for i in 0..link_count {
        let r = o + i * LINK_SIZE;
        links.push(NavLink {
            from: rd_u32(b, r),
            to: rd_u32(b, r + 4),
            portal: [rd_f32(b, r + 8), rd_f32(b, r + 12), rd_f32(b, r + 16)],
            a: rd_f32(b, r + 20),
            b: rd_f32(b, r + 24),
        });
    }
    Some(NetData { min, max, nodes, links })
}

/// Parse `verts*.bai` into world-space positions. Returns `None` on bad tag or
/// truncation; never panics.
pub fn parse_verts(b: &[u8]) -> Option<Vec<[f32; 3]>> {
    if b.len() < 8 || rd_u32(b, 0) != VERT_TAG {
        return None;
    }
    let count = rd_u32(b, 4) as usize;
    if b.len() < 8 + count.checked_mul(32)? {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let o = 8 + i * 32;
        out.push([rd_f32(b, o), rd_f32(b, o + 4), rd_f32(b, o + 8)]);
    }
    Some(out)
}

/// A decoded, queryable navigation mesh.
#[derive(Clone, Debug, Default)]
pub struct NavMesh {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub vertices: Vec<[f32; 3]>,
    pub nodes: Vec<NavNode>,
    pub links: Vec<NavLink>,
    /// `adjacency[node_id]` -> link indices leaving that node (1-based ids).
    adjacency: Vec<Vec<u32>>,
}

impl NavMesh {
    /// Combine a decoded `verts*.bai` and `net*.bai`.
    pub fn from_parts(vertices: Vec<[f32; 3]>, net: NetData) -> NavMesh {
        let mut adjacency = vec![Vec::new(); net.nodes.len() + 1];
        for (i, l) in net.links.iter().enumerate() {
            let from = l.from as usize;
            if from < adjacency.len() {
                adjacency[from].push(i as u32);
            }
        }
        NavMesh {
            min: net.min,
            max: net.max,
            vertices,
            nodes: net.nodes,
            links: net.links,
            adjacency,
        }
    }

    /// Node at 1-based id.
    pub fn node(&self, id: u32) -> Option<&NavNode> {
        let i = id as usize;
        if i >= 1 && i <= self.nodes.len() {
            Some(&self.nodes[i - 1])
        } else {
            None
        }
    }

    /// Nearest eligible node to a world point. Eligibility is
    /// `nav_type & agent_type != 0`.
    pub fn nearest_node(&self, p: [f32; 3], agent_type: u32) -> Option<u32> {
        let mut best = f32::INFINITY;
        let mut best_id = 0u32;
        for n in &self.nodes {
            if n.nav_type as u32 & agent_type == 0 {
                continue;
            }
            let d = dist2(p, n.center);
            if d < best {
                best = d;
                best_id = n.id;
            }
        }
        if best_id == 0 {
            None
        } else {
            Some(best_id)
        }
    }

    /// Two endpoints of the portal shared by two triangular nodes, if any.
    fn portal_endpoints(&self, a: &NavNode, b: &NavNode) -> Option<([f32; 3], [f32; 3])> {
        let mut shared = [u32::MAX; 2];
        let mut k = 0;
        for &va in &a.tri {
            if b.tri.contains(&va) && k < 2 {
                shared[k] = va;
                k += 1;
            }
        }
        if k != 2 {
            return None;
        }
        let p0 = *self.vertices.get(shared[0] as usize)?;
        let p1 = *self.vertices.get(shared[1] as usize)?;
        Some((p0, p1))
    }

    /// A* + string-pull path between two world points for the given nav bitmask.
    ///
    /// Returns `None` if the mesh is empty, either endpoint has no eligible node,
    /// or the goal is unreachable. A zero `agent_type` defaults to
    /// [`NAV_TRIANGULAR`]. The returned polyline starts at `from` and ends at
    /// `to` (CryEngine Z-up space).
    pub fn find_path(
        &self,
        from: [f32; 3],
        to: [f32; 3],
        agent_type: u32,
    ) -> Option<Vec<[f32; 3]>> {
        if self.nodes.is_empty() {
            return None;
        }
        let agent_type = if agent_type == 0 { NAV_TRIANGULAR } else { agent_type };
        let start = self.nearest_node(from, agent_type)?;
        let goal = self.nearest_node(to, agent_type)?;
        if start == goal {
            return Some(vec![from, to]);
        }

        let n = self.nodes.len();
        let mut g = vec![f32::INFINITY; n + 1];
        let mut came = vec![0u32; n + 1];
        let mut closed = vec![false; n + 1];
        let mut open: BinaryHeap<Scored> = BinaryHeap::new();

        g[start as usize] = 0.0;
        open.push(Scored { f: self.h(start, goal), id: start });

        while let Some(Scored { id, .. }) = open.pop() {
            if id == goal {
                break;
            }
            if closed[id as usize] {
                continue;
            }
            closed[id as usize] = true;
            let curnode = &self.nodes[(id - 1) as usize];
            for &li in &self.adjacency[id as usize] {
                let link = &self.links[li as usize];
                let nb = link.to;
                if nb == 0 || nb as usize > n || closed[nb as usize] {
                    continue;
                }
                let nbnode = &self.nodes[(nb - 1) as usize];
                if nbnode.nav_type as u32 & agent_type == 0 {
                    continue;
                }
                let step = dist(curnode.center, link.portal) + dist(link.portal, nbnode.center);
                let tentative = g[id as usize] + step;
                if tentative < g[nb as usize] {
                    g[nb as usize] = tentative;
                    came[nb as usize] = id;
                    open.push(Scored { f: tentative + self.h(nb, goal), id: nb });
                }
            }
        }

        if !g[goal as usize].is_finite() {
            return None;
        }

        let mut ids = vec![goal];
        let mut cur = goal;
        while cur != start {
            cur = came[cur as usize];
            if cur == 0 {
                return None;
            }
            ids.push(cur);
        }
        ids.reverse();

        let mut centers: Vec<[f32; 3]> = ids
            .iter()
            .map(|&i| self.nodes[(i - 1) as usize].center)
            .collect();
        self.string_pull(&ids, &mut centers);
        centers.insert(0, from);
        centers.push(to);
        Some(centers)
    }

    fn h(&self, id: u32, goal: u32) -> f32 {
        dist(self.nodes[(id - 1) as usize].center, self.nodes[(goal - 1) as usize].center)
    }

    /// Greedy string-pull: keep only the waypoints needed so clear runs become
    /// straight segments. A straight segment is accepted when it crosses every
    /// shared portal edge on the sub-path (in order), which keeps it inside the
    /// navmesh corridor.
    fn string_pull(&self, ids: &[u32], pts: &mut Vec<[f32; 3]>) {
        if ids.len() <= 2 || pts.len() != ids.len() {
            return;
        }
        // Does the straight segment anchor -> k cross every shared portal on the
        // sub-path (in order)? If so it stays inside the navmesh corridor.
        let los = |anchor: usize, k: usize, pts: &[[f32; 3]]| -> bool {
            for m in anchor..k {
                let a = &self.nodes[(ids[m] - 1) as usize];
                let b = &self.nodes[(ids[m + 1] - 1) as usize];
                if let Some((p0, p1)) = self.portal_endpoints(a, b) {
                    if !segments_cross_xy(pts[anchor], pts[k], p0, p1) {
                        return false;
                    }
                }
            }
            true
        };
        let mut out: Vec<[f32; 3]> = Vec::with_capacity(pts.len());
        out.push(pts[0]);
        let mut anchor = 0usize;
        while anchor + 1 < pts.len() {
            let mut best = anchor + 1;
            let mut k = anchor + 2;
            while k < pts.len() {
                if los(anchor, k, pts) {
                    best = k;
                    k += 1;
                } else {
                    break;
                }
            }
            out.push(pts[best]);
            anchor = best;
        }
        out.dedup();
        if out.len() >= 2 {
            *pts = out;
        }
    }
}

/// Open-set entry; ordered so the smallest `f` pops first.
#[derive(PartialEq)]
struct Scored {
    f: f32,
    id: u32,
}
impl Eq for Scored {}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .f
            .partial_cmp(&self.f)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    dist2(a, b).sqrt()
}
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Proper/normalised orientation in the XY plane.
fn orient(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Do segment `p1p2` and `p3p4` (projected to XY) cross strictly?
fn segments_cross_xy(p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], p4: [f32; 3]) -> bool {
    let d1 = orient(p3, p4, p1);
    let d2 = orient(p3, p4, p2);
    let d3 = orient(p1, p2, p3);
    let d4 = orient(p1, p2, p4);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// Load `verts<suffix>.bai` then `net<suffix>.bai` from a [`Vfs`] and build a
/// mesh. `suffix` is the level's mission tag, e.g. `"mission0"`.
pub fn load_from_vfs(vfs: &crate::crypak::Vfs, suffix: &str) -> Option<NavMesh> {
    let verts = parse_verts(&vfs.read(&format!("verts{suffix}.bai"))?)?;
    let net = parse_net(&vfs.read(&format!("net{suffix}.bai"))?)?;
    Some(NavMesh::from_parts(verts, net))
}

/// Load from a loose directory holding the extracted `.bai` files.
pub fn load_from_dir(dir: &std::path::Path, suffix: &str) -> Option<NavMesh> {
    let verts = parse_verts(&std::fs::read(dir.join(format!("verts{suffix}.bai"))).ok()?)?;
    let net = parse_net(&std::fs::read(dir.join(format!("net{suffix}.bai"))).ok()?)?;
    Some(NavMesh::from_parts(verts, net))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn put_f32(v: &mut Vec<u8>, x: f32) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    /// Build a tiny square mesh: 4 verts, 2 triangles, 2 directed links.
    fn sample_net() -> Vec<u8> {
        let mut b = Vec::new();
        put_u32(&mut b, NET_TAG);
        for x in [-1.0f32, -1.0, 0.0] {
            put_f32(&mut b, x);
        }
        for x in [1.0f32, 1.0, 0.0] {
            put_f32(&mut b, x);
        }
        put_u32(&mut b, 2); // node_count
        // node 1 (triangle 0,1,2) center (0,0,0)
        let node = |id: u32, tri: [u32; 3], c: [f32; 3]| -> Vec<u8> {
            let mut r = Vec::new();
            put_u32(&mut r, id);
            for _ in 0..6 {
                put_u32(&mut r, 0);
            }
            for k in 0..3 {
                put_f32(&mut r, c[k]);
            }
            put_u32(&mut r, 0);
            for t in tri {
                put_u32(&mut r, t);
            }
            r.extend_from_slice(&NAV_TRIANGULAR.to_le_bytes()[..2]);
            r.push(0);
            r.push(0);
            assert_eq!(r.len(), NODE_SIZE);
            r
        };
        b.extend(node(1, [0, 1, 2], [0.0, 0.0, 0.0]));
        b.extend(node(2, [1, 3, 2], [1.0, 0.0, 0.0]));
        put_u32(&mut b, 2); // link_count
        let link = |from: u32, to: u32, p: [f32; 3]| -> Vec<u8> {
            let mut r = Vec::new();
            put_u32(&mut r, from);
            put_u32(&mut r, to);
            for k in 0..3 {
                put_f32(&mut r, p[k]);
            }
            put_f32(&mut r, 0.0);
            put_f32(&mut r, 0.0);
            put_f32(&mut r, 0.0);
            put_f32(&mut r, 0.0);
            put_f32(&mut r, 0.0);
            put_f32(&mut r, 0.0);
            assert_eq!(r.len(), LINK_SIZE);
            r
        };
        b.extend(link(1, 2, [0.5, 0.0, 0.0]));
        b.extend(link(2, 1, [0.5, 0.0, 0.0]));
        b
    }

    fn sample_verts() -> Vec<u8> {
        let mut b = Vec::new();
        put_u32(&mut b, VERT_TAG);
        put_u32(&mut b, 4);
        let verts = [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        for v in verts {
            for k in 0..3 {
                put_f32(&mut b, v[k]);
            }
            // remaining 20 bytes of the 32-byte record
            b.extend_from_slice(&[0u8; 20]);
        }
        b
    }

    #[test]
    fn parses_net_and_verts() {
        let net = parse_net(&sample_net()).expect("net parses");
        assert_eq!(net.nodes.len(), 2);
        assert_eq!(net.links.len(), 2);
        assert_eq!(net.nodes[0].nav_type, NAV_TRIANGULAR as u16);
        assert_eq!(net.triangle_count(), 2);
        assert_eq!(net.nodes[1].tri, [1, 3, 2]);
        let verts = parse_verts(&sample_verts()).expect("verts parse");
        assert_eq!(verts.len(), 4);
        assert_eq!(verts[3], [1.0, 1.0, 0.0]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse_net(b"").is_none());
        assert!(parse_net(&[0u8; 64]).is_none());
        assert!(parse_verts(b"").is_none());
        assert!(parse_verts(&[0u8; 64]).is_none());
    }

    #[test]
    fn a_star_finds_path_between_triangles() {
        let mesh = NavMesh::from_parts(
            parse_verts(&sample_verts()).unwrap(),
            parse_net(&sample_net()).unwrap(),
        );
        let path = mesh
            .find_path([0.1, 0.1, 0.0], [0.9, 0.9, 0.0], NAV_TRIANGULAR)
            .expect("path");
        assert!(path.len() >= 2);
        assert_eq!(path[0], [0.1, 0.1, 0.0]);
        assert_eq!(*path.last().unwrap(), [0.9, 0.9, 0.0]);
    }

    #[test]
    fn zero_agent_type_defaults_to_triangular() {
        let mesh = NavMesh::from_parts(
            parse_verts(&sample_verts()).unwrap(),
            parse_net(&sample_net()).unwrap(),
        );
        assert!(mesh.find_path([0.1, 0.1, 0.0], [0.9, 0.9, 0.0], 0).is_some());
    }

    #[test]
    fn unreachable_agent_type_returns_none() {
        let mesh = NavMesh::from_parts(
            parse_verts(&sample_verts()).unwrap(),
            parse_net(&sample_net()).unwrap(),
        );
        assert!(mesh
            .find_path([0.1, 0.1, 0.0], [0.9, 0.9, 0.0], NAV_ROAD)
            .is_none());
    }

    // -- real level data (skipped when the extracted level is absent) --------

    const EXTRACTED_LEVEL: &str = "/home/battlestation/Desktop/sod-re/extracted-game/level";

    #[test]
    fn real_level_decodes_and_paths_between_tagpoints() {
        use crate::tagpoints::TagPoints;
        let dir = std::path::Path::new(EXTRACTED_LEVEL);
        let Some(mesh) = load_from_dir(dir, "mission0") else {
            eprintln!("level .bai absent; skipping real-data nav test");
            return;
        };
        eprintln!(
            "navmesh: {} nodes ({} triangles), {} links, {} verts, bounds min={:?} max={:?}",
            mesh.nodes.len(),
            mesh.nodes
                .iter()
                .filter(|n| n.nav_type as u32 & NAV_TRIANGULAR != 0)
                .count(),
            mesh.links.len(),
            mesh.vertices.len(),
            mesh.min,
            mesh.max
        );
        assert!(mesh.nodes.len() > 50_000);
        assert!(mesh.vertices.len() > 10_000);

        let tp = std::fs::read(dir.join("tag_points_win.markup"))
            .ok()
            .and_then(|b| TagPoints::parse(&b));
        let (Some(tp), _) = (tp, ()) else {
            eprintln!("tag points absent; skipping path check");
            return;
        };
        let start = tp.get("RTSPoint_PlayerStart").expect("PlayerStart");
        let goal = tp.get("RTSPoint_RangerStation").expect("RangerStation");
        let path = mesh
            .find_path(start, goal, NAV_TRIANGULAR)
            .expect("a path should exist between PlayerStart and RangerStation");
        let len: f32 = path
            .windows(2)
            .map(|w| dist(w[0], w[1]))
            .sum();
        eprintln!(
            "path PlayerStart {:?} -> RangerStation {:?}: {} points, length {:.1} m",
            start,
            goal,
            path.len(),
            len
        );
        assert!(path.len() >= 2);
        assert!(len > 1.0 && len < 20_000.0);
    }
}
