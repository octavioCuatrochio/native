//! Bevy renderer for the recovered CryEngine particle system.
//!
//! The core crate owns the engine-agnostic simulation
//! ([`sod_core::particles`]): it parses the shipped `libs/particles/*.xml`
//! libraries (whose effects are plain XML with `Params` micro-format
//! attributes) and runs one emitter simulation per placed effect. This module
//! is the presentation boundary:
//!
//! * once, it loads every `libs/particles/*.xml` from the Cry VFS into a
//!   [`ParticleLibrary`];
//! * every frame it re-reads the entity world's `SlotKind::Particle` slots
//!   (`LoadParticleEffect` in `particleeffect.lua` records each effect's name
//!   and transform) and keeps the effects within `SOD_PARTICLE_RADIUS` metres
//!   of the camera alive in a [`ParticleSystem`];
//! * it advances the simulation by `dt`;
//! * it draws the live particles as camera-facing billboards. Billboards from
//!   one `(texture, blend)` pair are packed into a single world-space mesh, so
//!   the whole level costs a handful of draw calls instead of one per particle.
//!
//! Blend modes map from the authored `BlendType`/`AlphaTest`
//! (`sod_core::particles::BlendMode`) onto Bevy's `AlphaMode`: additive fire
//! and embers use `Add`, smoke/dust/water use `Blend`, and masked cutouts use
//! `Mask`. Textures are the emitter's `Texture=` (`.tif` resolved to the
//! shipped `.dds`) or, failing that, the `Diffuse` map of its `Material=`.
//!
//! Because the level ships 818 `ParticleEffect` entities, effects are streamed
//! by distance and capped; `SOD_PARTICLE_RADIUS`, `SOD_PARTICLE_MAX` and
//! `SOD_PARTICLE_BUDGET` tune the load. `SOD_NOPARTICLES=1` disables it.

use bevy::asset::RenderAssetUsages;
use bevy::material::AlphaMode;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use sod_core::cry::render::visible_slots;
use sod_core::cry::world::SlotKind;
use sod_core::material;
use sod_core::particles::{BlendMode, ParticleLibrary, ParticleSystem};
use std::collections::HashMap;
use std::rc::Rc;

/// How often the set of live effects is re-evaluated (seconds), plus a
/// distance trigger so fast travel refreshes immediately.
const REFRESH_PERIOD: f32 = 0.5;
const REFRESH_MOVE: f32 = 4.0;
/// Default radius (metres) around the camera in which effects are simulated.
const DEFAULT_RADIUS: f32 = 200.0;
/// Default cap on live effect instances.
const DEFAULT_MAX_EFFECTS: usize = 48;
/// Default cap on displayed particles per frame across all effects.
const DEFAULT_BUDGET: usize = 12000;

/// Installs the bridge; added next to the other plugins in `main.rs`.
pub struct ParticleBridgePlugin;

impl Plugin for ParticleBridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleRender>()
            .add_systems(Update, update_particles.after(crate::cry_bridge::CrySet::Tick));
    }
}

/// The Cry-to-Bevy basis change used everywhere at the render boundary.
fn base_rot() -> Quat {
    Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)
}

/// A billboard's `(texture, material, blend)` identity. The renderer keeps one
/// mesh + material per distinct key.
#[derive(Clone, PartialEq, Eq, Hash)]
struct GroupKey {
    texture: Option<String>,
    material: Option<String>,
    blend: u32,
}

/// One live billboard, in CryEngine world space.
struct PQuad {
    pos: [f32; 3],
    size: f32,
    rot: f32,
    color: [f32; 4],
    tiling: (f32, f32),
    dist2: f32,
}

/// One rendered `(texture, blend)` group: the entity whose mesh we rewrite
/// every frame, plus the material we keep until its texture changes.
struct Group {
    entity: Entity,
    texture: Option<Handle<Image>>,
    material: Handle<StandardMaterial>,
    /// One mesh per group, rewritten in place each frame (a fresh asset per
    /// frame churned the render slab allocator: use-after-free errors).
    mesh: Handle<Mesh>,
}

/// All renderer state. The simulation is `None` until the library loads.
#[derive(Resource, Default)]
struct ParticleRender {
    sys: Option<ParticleSystem>,
    /// `(entity, slot)` -> simulation instance id.
    tracked: HashMap<(u32, usize), usize>,
    groups: HashMap<GroupKey, Group>,
    /// Resolved `.dds` path -> decoded GPU handle.
    textures: HashMap<String, Option<Handle<Image>>>,
    /// Neutral soft-dot texture used when an emitter has no usable texture.
    fallback: Option<Handle<Image>>,
    refresh: f32,
    last_center: [f32; 3],
    stats_frames: u32,
    stats_accum: f32,
    stats_logged: u32,
}

/// Read the two tuning knobs as `f32`/`usize` environment overrides.
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Blend tag for the group key (stable, quantises the alpha-test cutoff).
fn blend_tag(b: BlendMode) -> u32 {
    match b {
        BlendMode::AlphaBlend => 0,
        BlendMode::Additive => 1,
        BlendMode::Opaque => 2,
        BlendMode::Mask(a) => 3 + (a.clamp(0.0, 1.0) * 1000.0) as u32 * 4,
    }
}

fn alpha_mode(b: BlendMode) -> AlphaMode {
    match b {
        BlendMode::AlphaBlend => AlphaMode::Blend,
        BlendMode::Additive => AlphaMode::Add,
        BlendMode::Opaque => AlphaMode::Opaque,
        BlendMode::Mask(a) => AlphaMode::Mask(a.clamp(0.0, 1.0)),
    }
}

/// Resolve an authored texture reference to a shipped `.dds` path.
fn resolve_dds(vfs: &sod_core::crypak::Vfs, raw: &str) -> Option<String> {
    let lower = raw.replace('\\', "/").trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }
    if vfs.exists(&lower) {
        return Some(lower);
    }
    if let Some(pos) = lower.rfind('.') {
        let cand = format!("{}.dds", &lower[..pos]);
        if vfs.exists(&cand) {
            return Some(cand);
        }
    }
    let cand = format!("{lower}.dds");
    if vfs.exists(&cand) {
        return Some(cand);
    }
    None
}

/// The `Diffuse` map a `Material=` reference names, if it resolves.
fn material_diffuse(vfs: &sod_core::crypak::Vfs, m: &str) -> Option<String> {
    let mtl = format!("{}.mtl", m.trim_end_matches(".mtl"));
    let slots = material::load_mtl_path(vfs, &mtl);
    for s in &slots {
        if let Some(t) = s.texture("Diffuse") {
            if let Some(path) = resolve_dds(vfs, t) {
                return Some(path);
            }
        }
    }
    None
}

/// Build (or fetch) the GPU image for a resolved `.dds` path.
fn texture_handle(
    images: &mut Assets<Image>,
    cache: &mut HashMap<String, Option<Handle<Image>>>,
    vfs: &sod_core::crypak::Vfs,
    path: &str,
) -> Option<Handle<Image>> {
    if let Some(h) = cache.get(path) {
        return h.clone();
    }
    let decoded = vfs.read(path).and_then(|b| sod_core::asset::decode_dds_rgba(&b));
    let handle = decoded.and_then(|d| {
        let img = Image::new(
            Extent3d { width: d.width.max(1), height: d.height.max(1), depth_or_array_layers: 1 },
            TextureDimension::D2,
            d.rgba,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        Some(images.add(img))
    });
    cache.insert(path.to_string(), handle.clone());
    handle
}

/// A soft radial dot, used when an emitter's texture cannot be resolved.
fn fallback_texture(images: &mut Assets<Image>) -> Handle<Image> {
    let n = 64u32;
    let mut data = vec![0u8; (n * n * 4) as usize];
    let half = n as f32 / 2.0;
    for y in 0..n {
        for x in 0..n {
            let dx = (x as f32 + 0.5 - half) / half;
            let dy = (y as f32 + 0.5 - half) / half;
            let r = (dx * dx + dy * dy).sqrt();
            let a = (1.0 - r).clamp(0.0, 1.0);
            let a = (a * a * 255.0) as u8;
            let o = ((y * n + x) * 4) as usize;
            data[o] = 255;
            data[o + 1] = 255;
            data[o + 2] = 255;
            data[o + 3] = a;
        }
    }
    images.add(Image::new(
        Extent3d { width: n, height: n, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ))
}

/// Gather the nearest live particle slots around `center` (Cry space).
fn gather_slots(
    world: &sod_core::cry::world::EntityWorld,
    lib: &ParticleLibrary,
    center: [f32; 3],
    radius: f32,
    max: usize,
) -> Vec<(u32, usize, String, [f32; 3], [f32; 4], f32)> {
    let r2 = radius * radius;
    let mut picks = Vec::new();
    for s in visible_slots(world) {
        if s.kind != SlotKind::Particle || s.path.is_empty() {
            continue;
        }
        let dx = s.pos[0] - center[0];
        let dy = s.pos[1] - center[1];
        let dz = s.pos[2] - center[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 > r2 {
            continue;
        }
        // Only effects the shipped libraries define are simulated.
        if lib.get(&s.path).is_none() {
            continue;
        }
        picks.push((d2, (s.entity, s.slot, s.path, s.pos, s.rot, s.scale[0])));
    }
    picks.sort_by(|a, b| a.0.total_cmp(&b.0));
    picks.truncate(max);
    picks.into_iter().map(|(_, v)| v).collect()
}

fn update_particles(
    mut render: ResMut<ParticleRender>,
    cry: Option<NonSend<crate::cry_bridge::CryHost>>,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    mut commands: Commands,
) {
    if std::env::var("SOD_NOPARTICLES").is_ok() {
        return;
    }
    let Some(cry) = cry else { return };
    let Ok(cam) = cameras.single() else { return };
    // Borrow the resource's fields directly so the simulation and the
    // tracked/groups maps can be mutated independently.
    let render: &mut ParticleRender = &mut render;

    let vfs: Rc<sod_core::crypak::Vfs> = cry.host.ctx.vfs.clone();

    // --- load the libraries once ------------------------------------------
    if render.sys.is_none() {
        let t0 = std::time::Instant::now();
        let mut lib = ParticleLibrary::new();
        let mut files = 0usize;
        for path in vfs.paths_with_ext(".xml") {
            if !path.to_ascii_lowercase().starts_with("libs/particles/") {
                continue;
            }
            if let Some(bytes) = vfs.read(&path) {
                if lib.add_xml(&bytes).is_ok() {
                    files += 1;
                }
            }
        }
        println!(
            "[particles] loaded {} libraries, {} effects in {:.0} ms",
            files,
            lib.len(),
            t0.elapsed().as_secs_f32() * 1000.0
        );
        render.sys = Some(ParticleSystem::new(lib));
        render.fallback = Some(fallback_texture(&mut images));
    }

    let raw_dt = time.delta_secs();
    let dt = raw_dt.min(0.1);
    let radius = env_f32("SOD_PARTICLE_RADIUS", DEFAULT_RADIUS);
    let max_effects = env_usize("SOD_PARTICLE_MAX", DEFAULT_MAX_EFFECTS);
    let budget = env_usize("SOD_PARTICLE_BUDGET", DEFAULT_BUDGET);
    // Debug hook: place one named effect a few metres in front of the camera
    // instead of streaming the level's slots. Used to photograph a specific
    // effect (e.g. `Class3Effects.smoke_and_fire.burn_barrel`).
    let focus = std::env::var("SOD_PARTICLE_FOCUS").ok();
    let focus_dist = env_f32("SOD_PARTICLE_FOCUS_DIST", 12.0);

    // Camera position in CryEngine space is the streaming centre.
    let center = (base_rot().inverse() * cam.translation()).to_array();

    // --- refresh the live effect set --------------------------------------
    render.refresh += dt;
    let moved2 = {
        let d = [
            center[0] - render.last_center[0],
            center[1] - render.last_center[1],
            center[2] - render.last_center[2],
        ];
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    if render.refresh >= REFRESH_PERIOD || moved2 >= REFRESH_MOVE * REFRESH_MOVE {
        render.refresh = 0.0;
        render.last_center = center;
        let picks = if let Some(name) = focus.clone() {
            let fwd = base_rot().inverse() * (cam.rotation() * -Vec3::Z);
            let p = Vec3::from_array(center) + fwd * focus_dist;
            vec![(0u32, 0usize, name, p.to_array(), [0.0, 0.0, 0.0, 1.0], 1.0f32)]
        } else {
            let sys = render.sys.as_ref().unwrap();
            gather_slots(&cry.host.ctx.world.borrow(), &sys.lib, center, radius, max_effects)
        };
        let sys = render.sys.as_mut().unwrap();
        let mut desired: HashMap<(u32, usize), usize> = HashMap::with_capacity(picks.len());
        for (entity, slot, effect, pos, rot, scale) in picks {
            let key = (entity, slot);
            if let Some(&id) = render.tracked.get(&key) {
                sys.set_transform(id, pos, rot, scale);
                desired.insert(key, id);
            } else if let Some(id) = sys.add(&effect, pos, rot, scale) {
                render.tracked.insert(key, id);
                desired.insert(key, id);
            }
        }
        let stale: Vec<(u32, usize)> =
            render.tracked.keys().filter(|k| !desired.contains_key(*k)).copied().collect();
        for k in stale {
            if let Some(id) = render.tracked.remove(&k) {
                sys.remove(id);
            }
        }
    }

    if let Some(sys) = render.sys.as_mut() {
        sys.step(dt);
    }

    // --- pack the live particles into (texture, blend) bucket meshes -------
    let mut buckets: HashMap<GroupKey, (BlendMode, Vec<PQuad>)> = HashMap::new();
    let mut shown = 0usize;
    {
        let sys = render.sys.as_ref().unwrap();
        let cam_bevy = cam.translation();
        'outer: for (_, inst) in sys.instances() {
            let Some(def) = sys.lib.get(&inst.effect) else { continue };
            // Whole-effect distance cull for effects that author one (skipped
            // for the debug focus effect, which is meant to be seen).
            let max_view = def.max_view_distance();
            if focus.is_none() && max_view > 0.0 {
                let p = base_rot() * Vec3::from_array(inst.origin);
                if p.distance(cam_bevy) > max_view {
                    continue;
                }
            }
            for st in &inst.emitters {
                let Some(em) = def.emitters.get(st.def) else { continue };
                let p = &em.params;
                let key = GroupKey {
                    texture: p.texture.clone(),
                    material: if p.texture.is_none() { p.material.clone() } else { None },
                    blend: blend_tag(p.blend),
                };
                let entry = buckets.entry(key).or_insert_with(|| (p.blend, Vec::new()));
                for part in &st.particles {
                    if shown >= budget {
                        break 'outer;
                    }
                    if part.alpha <= 0.003 || part.size <= 0.001 {
                        continue;
                    }
                    let world = base_rot() * Vec3::from_array(part.pos);
                    let b = p.brightness;
                    let col = [
                        (part.color[0] * b).clamp(0.0, 8.0),
                        (part.color[1] * b).clamp(0.0, 8.0),
                        (part.color[2] * b).clamp(0.0, 8.0),
                        part.alpha.clamp(0.0, 1.0),
                    ];
                    entry.1.push(PQuad {
                        pos: part.pos,
                        size: part.size,
                        rot: part.rot,
                        color: col,
                        tiling: p.texture_tiling,
                        dist2: world.distance_squared(cam_bevy),
                    });
                    shown += 1;
                }
            }
        }
    }

    // --- upload one mesh per group; material only when its texture changes --
    let right = cam.rotation() * Vec3::X;
    let up = cam.rotation() * Vec3::Y;
    let fallback = render.fallback.clone();
    for (key, (blend, mut quads)) in buckets {
        // Resolve the texture for this group (cached by path).
        let tex = if let Some(path) = key.texture.as_deref().and_then(|t| resolve_dds(&vfs, t)) {
            texture_handle(&mut images, &mut render.textures, &vfs, &path)
        } else if let Some(path) = key.material.as_deref().and_then(|m| material_diffuse(&vfs, m)) {
            texture_handle(&mut images, &mut render.textures, &vfs, &path)
        } else {
            fallback.clone()
        };
        // Ensure the group entity + material exist (recreate the material only
        // when the texture handle changes, so most frames reuse it).
        let entity = match render.groups.entry(key.clone()) {
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let g = o.get_mut();
                if g.texture != tex {
                    g.texture = tex.clone();
                    g.material = materials.add(StandardMaterial {
                        base_color: Color::WHITE,
                        base_color_texture: tex,
                        alpha_mode: alpha_mode(blend),
                        unlit: true,
                        cull_mode: None,
                        ..default()
                    });
                }
                g.entity
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                let mat = materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    base_color_texture: tex.clone(),
                    alpha_mode: alpha_mode(blend),
                    unlit: true,
                    cull_mode: None,
                    ..default()
                });
                let mesh = meshes.add(build_group_mesh(&quads, right, up));
                let e = commands
                    .spawn((Transform::default(), Visibility::Inherited, Mesh3d(mesh.clone()), MeshMaterial3d(mat.clone())))
                    .id();
                v.insert(Group { entity: e, texture: tex, material: mat.clone(), mesh });
                e
            }
        };
        let g = render.groups.get(&key).unwrap();
        let (mat, mesh) = (g.material.clone(), g.mesh.clone());
        quads.sort_by(|a, b| b.dist2.total_cmp(&a.dist2));
        // Never leave a zero-vertex mesh for the renderer: hide the group instead.
        if quads.is_empty() {
            commands.entity(entity).insert(Visibility::Hidden);
            continue;
        }
        if let Some(mut m) = meshes.get_mut(&mesh) {
            *m = build_group_mesh(&quads, right, up);
        }
        commands.entity(entity).insert((MeshMaterial3d(mat), Visibility::Inherited));
    }

    // --- stats -------------------------------------------------------------
    render.stats_frames += 1;
    render.stats_accum += raw_dt.min(0.5);
    if render.stats_frames >= render.stats_logged + 120 {
        let live = render.sys.as_ref().map(|s| s.instances().count()).unwrap_or(0);
        let avg = render.stats_accum / (render.stats_frames - render.stats_logged) as f32;
        info!(
            "[particles] {} effects, {} particles shown, {:.1} ms/frame",
            live,
            shown,
            avg * 1000.0
        );
        render.stats_logged = render.stats_frames;
        render.stats_accum = 0.0;
    }
}

/// Pack camera-facing quads into one world-space mesh. `quads` are sorted
/// back-to-front first so alpha blending reads correctly.
fn build_group_mesh(quads: &[PQuad], right: Vec3, up: Vec3) -> Mesh {
    let base = base_rot();
    let mut pos: Vec<[f32; 3]> = Vec::with_capacity(quads.len() * 4);
    let mut uv: Vec<[f32; 2]> = Vec::with_capacity(quads.len() * 4);
    let mut col: Vec<[f32; 4]> = Vec::with_capacity(quads.len() * 4);
    let mut idx: Vec<u32> = Vec::with_capacity(quads.len() * 6);
    for q in quads {
        let c = base * Vec3::from_array(q.pos);
        let (s, co) = q.rot.sin_cos();
        let rx = right * co + up * s;
        let uy = -right * s + up * co;
        let h = q.size * 0.5;
        let base_i = pos.len() as u32;
        for (dx, dy, u, v) in [
            (-1.0f32, -1.0f32, 0.0f32, 0.0f32),
            (1.0, -1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
            (-1.0, 1.0, 0.0, 1.0),
        ] {
            let corner = c + rx * (dx * h) + uy * (dy * h);
            pos.push(corner.to_array());
            uv.push([u * q.tiling.0, v * q.tiling.1]);
            col.push(q.color);
        }
        idx.extend([base_i, base_i + 1, base_i + 2, base_i, base_i + 2, base_i + 3]);
    }
    let mut m = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    if !pos.is_empty() {
        m.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(pos));
        m.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uv));
        m.insert_attribute(Mesh::ATTRIBUTE_COLOR, VertexAttributeValues::Float32x4(col));
        m.insert_indices(Indices::U32(idx));
    }
    m
}
