//! Per-frame terrain LOD selection and streaming (CryEngine `CTerrainNode`).
//!
//! The original does not mesh the terrain once: every frame
//! `CTerrain::RenderTerrain` walks the `CTerrainNode` quadtree, runs
//! `CheckVis`/`SetLOD` per visible node and rebuilds the mesh of every node
//! whose chosen geometry level (`m_cNewGeomMML`) changed. This module is the
//! presentation half of that: the engine-agnostic recursion and meshing live in
//! `sod_core::terrain` ([`TerrainVis`]); here we hold the per-node mesh cache,
//! rebuild only the nodes whose stitching signature changed, and write the
//! merged attributes back into the single terrain `Mesh` asset.
//!
//! Budgeting: the amount of geometry rebuilt per frame is capped
//! ([`MAX_NODE_BUILDS_PER_FRAME`]) so a teleport can never stall a frame; the
//! remaining nodes fill in over the next frames. At walking speed only a
//! handful of sectors cross an LOD threshold per frame, so updates are both
//! rare and small.
//!
//! Holes: `sod_core::terrain` now keeps the authored hole flag (raw low nibble
//! `15`) distinct and drops the quads whose corners touch it, matching
//! `CTerrainNode::BuildIndices`; those are the real source holes the overview
//! camera shows at the map edge, not a decode artifact.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use sod_core::surfaces::SurfaceType;
use sod_core::terrain::{self, Terrain, TerrainLodParams, TerrainSector, TerrainVis};
use std::collections::HashMap;
use std::time::Instant;

/// Maximum number of sector meshes rebuilt in a single frame. A camera move
/// typically changes 0-4; the cap only matters on a teleport, where it turns a
/// single large rebuild into a few bounded ones.
const MAX_NODE_BUILDS_PER_FRAME: usize = 6;

/// One cached sector mesh, keyed by [`TerrainSector::id`].
struct Chunk {
    /// Stitching signature: own LOD plus neighbour LODs along the borders.
    /// Unchanged signature => identical geometry, so the entity is kept as is.
    sig: u64,
    /// Child entity of the terrain entity drawing this sector, and its mesh.
    entity: Entity,
    mesh: Handle<Mesh>,
    verts: usize,
    tris: usize,
}

/// Marks the per-sector child entities.
#[derive(Component)]
pub struct TerrainSectorTag;

/// Dynamic terrain LOD state, inserted by `run_world`.
#[derive(Resource)]
pub struct TerrainLodState {
    /// The decoded `CTerrainNode` quadtree.
    pub root: TerrainSector,
    /// `leveldata.xml` surface palette, for per-vertex tint.
    pub surface_types: Vec<SurfaceType>,
    /// Base LOD parameters (camera is overwritten each frame).
    pub params: TerrainLodParams,
    cache: HashMap<u32, Chunk>,
    /// Current rendered nodes, in draw order.
    order: Vec<u32>,
    /// Terrain entity (owns the shared material); sectors are its children.
    terrain: Option<(Entity, Handle<crate::TerrainMaterial>)>,
    frame: u32,
    /// `SOD_TELEPORT=x,y,z` scripted move for verification captures.
    teleport: Option<[f32; 3]>,
    teleported: bool,
    /// Diagnostics.
    last_ms: f32,
    worst_ms: f32,
    updates: u64,
}

impl TerrainLodState {
    /// Build the state from the decoded tree. `SOD_TELEPORT` moves the player
    /// to a fixed spot for scripted far-apart captures.
    pub fn new(root: TerrainSector, surface_types: Vec<SurfaceType>, params: TerrainLodParams) -> Self {
        let teleport = std::env::var("SOD_TELEPORT").ok().and_then(|s| {
            let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse::<f32>().ok()).collect();
            (v.len() >= 3).then(|| [v[0], v[1], v[2]])
        });
        Self {
            root,
            surface_types,
            params,
            cache: HashMap::new(),
            order: Vec::new(),
            terrain: None,
            frame: 0,
            teleport,
            teleported: false,
            last_ms: 0.0,
            worst_ms: 0.0,
            updates: 0,
        }
    }
}

/// Analytic ground normal from the reconstructed heightmap (central
/// differences). Unlike `mesh::generate_normals` this is independent of the
/// per-node triangulation, so LOD borders cannot show a shading seam.
fn terrain_normal(t: &Terrain, x: f32, y: f32) -> [f32; 3] {
    let d = t.spacing.max(1e-3);
    let hx = t.height_interp(x + d, y) - t.height_interp(x - d, y);
    let hy = t.height_interp(x, y + d) - t.height_interp(x, y - d);
    let n = Vec3::new(-hx, -hy, 2.0 * d);
    let n = n.normalize_or_zero();
    [n.x, n.y, n.z]
}

/// One sector's mesh asset.
fn sector_mesh(positions: Vec<[f32; 3]>, normals: Vec<[f32; 3]>, colors: Vec<[f32; 4]>, uvs: Vec<[f32; 2]>, indices: Vec<u32>) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(positions));
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(normals));
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, VertexAttributeValues::Float32x4(colors));
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs));
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Per-frame terrain LOD update. Runs after the camera systems (and after
/// `setup_world`), so it always sees this frame's camera transform.
#[allow(clippy::too_many_arguments)]
pub fn update_terrain_lod(
    mut state: ResMut<TerrainLodState>,
    world: Res<crate::World>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    camera: Query<&Transform, With<crate::MainCamera>>,
    terrain_q: Query<(Entity, &Mesh3d, &MeshMaterial3d<crate::TerrainMaterial>), Without<TerrainSectorTag>>,
    mut player: Option<ResMut<crate::Player>>,
    mut cam_state: Option<ResMut<crate::CameraStateRes>>,
) {
    let debug = std::env::var("SOD_TERRAIN_DEBUG").is_ok();

    state.frame = state.frame.wrapping_add(1);

    // Scripted teleport for the verification captures. Applied a few frames in
    // (after the menu has auto-started), once.
    if let Some(tp) = state.teleport {
        if !state.teleported && state.frame >= 8 {
            if let Some(p) = player.as_mut() {
                p.mstate.pos = tp;
                p.mstate.vel = [0.0; 3];
                p.pos = [tp[0], tp[1]];
            }
            if let Some(cs) = cam_state.as_mut() {
                cs.0.pos = tp;
            }
            state.teleported = true;
            println!("[terrain] SOD_TELEPORT -> {tp:?}");
        }
    }

    let Some(cam) = camera.iter().next() else { return };
    // Bevy is Y-up here; the terrain/engine are Z-up (see `run_world`).
    let t = cam.translation;
    let camera_cry = [t.x, -t.z, t.y];

    let ter = &world.terrain;
    let extent = (ter.samples as f32 * ter.spacing).max(1.0);
    let plane_uv = world.ground_layers.is_some() || world.ground_albedo.is_some();
    let uv_scale = if plane_uv { 1.0 / extent } else { 1.0 / 8.0 };

    let t0 = Instant::now();

    // Split the resource borrows so `vis` (borrowing `root`) coexists with the
    // mutable `cache`.
    let TerrainLodState { root, surface_types, params, cache, order, terrain, last_ms, worst_ms, updates, .. } = &mut *state;
    // Resolve the terrain entity once; its single setup-time mesh is emptied so
    // only the per-sector children draw.
    if terrain.is_none() {
        let Some((e, m, mat)) = terrain_q.iter().next() else { return };
        // Drop the setup-time mesh from the entity (an attribute-less mesh would
        // break the shadow/prepass pipeline); the entity stays as the parent.
        meshes.remove(&m.0);
        commands.entity(e).remove::<Mesh3d>();
        *terrain = Some((e, mat.0.clone()));
    }
    let (terrain_e, terrain_mat) = terrain.clone().unwrap();
    let vis = TerrainVis::new(ter, root, &TerrainLodParams { camera: camera_cry, ..*params });

    let first_build = order.is_empty();
    let budget = if first_build { usize::MAX } else { MAX_NODE_BUILDS_PER_FRAME };
    let mut built = 0usize;
    let mut new_order: Vec<u32> = Vec::with_capacity(vis.nodes.len());
    let mut changed = first_build;

    for i in 0..vis.nodes.len() {
        let s = vis.nodes[i].0;
        let sig = vis.signature_at(root, i);
        let valid = matches!(cache.get(&s.id), Some(c) if c.sig == sig);
        if !valid {
            if built >= budget {
                // Defer: keep drawing the previous geometry (if any) until a
                // later frame rebuilds it, so no hole appears in between.
                if cache.contains_key(&s.id) {
                    new_order.push(s.id);
                }
                changed = true;
                continue;
            }
            let m = vis.build_at(ter, root, i);
            if m.positions.is_empty() {
                continue;
            }
            let mut normals = Vec::with_capacity(m.positions.len());
            let mut colors = Vec::with_capacity(m.positions.len());
            let mut uvs = Vec::with_capacity(m.positions.len());
            for p in &m.positions {
                normals.push(terrain_normal(ter, p[0], p[1]));
                let c = sod_core::surfaces::color_for(surface_types, ter.surface_at(p[0], p[1]));
                colors.push([0.55 + 0.45 * c[0], 0.55 + 0.45 * c[1], 0.55 + 0.45 * c[2], 1.0]);
                uvs.push([p[0] * uv_scale, p[1] * uv_scale]);
            }
            let (verts, tris) = (m.positions.len(), m.indices.len() / 3);
            let handle = meshes.add(sector_mesh(m.positions, normals, colors, uvs, m.indices));
            if let Some(old) = cache.remove(&s.id) {
                commands.entity(old.entity).despawn();
                meshes.remove(&old.mesh);
            }
            let entity = commands
                .spawn((Mesh3d(handle.clone()), MeshMaterial3d(terrain_mat.clone()), Transform::default(), TerrainSectorTag, ChildOf(terrain_e)))
                .id();
            cache.insert(s.id, Chunk { sig, entity, mesh: handle, verts, tris });
            built += 1;
            changed = true;
        }
        new_order.push(s.id);
    }

    // Drop sectors that left the view or are no longer rendered.
    let stale: Vec<u32> = cache.keys().copied().filter(|id| !new_order.contains(id)).collect();
    if !stale.is_empty() {
        for id in stale {
            if let Some(old) = cache.remove(&id) {
                commands.entity(old.entity).despawn();
                meshes.remove(&old.mesh);
            }
        }
        changed = true;
    }

    if new_order != *order {
        *order = new_order;
        changed = true;
    }

    if !changed {
        return;
    }

    let (verts, tris) = order.iter().filter_map(|id| cache.get(id)).fold((0, 0), |a, c| (a.0 + c.verts, a.1 + c.tris));

    let ms = t0.elapsed().as_secs_f32() * 1000.0;
    *last_ms = ms;
    *worst_ms = worst_ms.max(ms);
    *updates += 1;
    if debug || ms > 5.0 {
        println!(
            "[terrain] LOD update #{}: {} sectors, {} verts, {} tris, built {} chunks in {:.2} ms (worst {:.2})",
            updates, order.len(), verts, tris, built, ms, worst_ms
        );
    }
}
