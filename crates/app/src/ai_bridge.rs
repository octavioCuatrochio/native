//! Bevy side of the zombie ecology: runs the native [`ZombieRuntime`] against
//! the live world, spawns the original zombie actor scripts through the
//! [`crate::cry_bridge::CryHost`], resolves their living capsules against the
//! static world and each other, and draws/skins the promoted actors.
//!
//! Pipeline per frame (ordered after `move_player`):
//!
//! 1. `zombie_bridge_tick` — lazy-boot the runtime from the mounted paks
//!    (`ZombieTypes.xml`, `RTS/Zombies.xml`, `zombie_spawns_win.markup`, the
//!    AI goal-pipe registry and the `.bai` navmesh), run the spawn director
//!    over a [`WorldView`](sod_core::cry::zombie_spawn::WorldView), reconcile
//!    live actors (spawn/despawn through `ScriptHost::spawn_actor` using the
//!    recovered class → script map), step each live actor's goal pipe, then
//!    collide every zombie against the static collision world and separate the
//!    actor-actor / actor-player capsule overlaps.
//! 2. `zombie_visual_sync` — lazily load the per-class character (its `.cdf`
//!    skeleton, attachments and `.mtl` palette) and spawn a skinned Bevy entity
//!    per newly promoted worker.
//! 3. `animate_zombie_visuals` — position it and play the idle/walk clip chosen
//!    from the runtime's `moving` flag.
//! 4. `zombie_hud` — feed the live count and the status line into the HUD,
//!    which otherwise only sees the legacy `Npc` entities.
//!
//! The script host remains the authority for zombie behaviour; this file only
//! feeds it positions, runs the goal pipes it authored, and renders the result.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use sod_core::cry::ai;
use sod_core::cry::ai_data::AiState;
use sod_core::cry::navmesh::{self, NavMesh};
use sod_core::cry::phys_world::CollisionWorld;
use sod_core::cry::world::EntityId;
use sod_core::cry::zombie_actor::{self, Body};
use sod_core::cry::zombie_runtime::ZombieRuntime;
use sod_core::cry::zombie_spawn::{parse_spawn_markup, SpawnDirector, ZombieTable, ZombieTuning};
use sod_core::crypak::Vfs;
use sod_core::material::MaterialSlot;
use sod_core::mesh::DecodedModel;
use sod_core::skin::{self, BoneWeight, Skeleton};
use sod_core::anim;
use std::collections::{HashMap, HashSet};

use crate::cry_bridge::{CryHost, CryPlayerPos};
use crate::{AnimPart, Collision, Player, TextureCacheRes};

/// How many individual spawn/despawn lines to print before summarising.
const VERBOSE_SPAWN_LOG: u64 = 60;
/// Separation relaxation passes per frame (chains of overlapping actors).
const SEPARATION_ITERATIONS: usize = 3;

/// Loaded zombie character for one class, shared by every promoted actor of
/// that class.
struct ZombieTemplate {
    model: DecodedModel,
    slots: Vec<MaterialSlot>,
    tex_bytes: HashMap<String, Vec<u8>>,
    skeleton: Skeleton,
    influences: Vec<Vec<[BoneWeight; 4]>>,
    /// Neutral idle loop (never the burning/reaction clips the bank may pick).
    idle: anim::Caf,
    /// Forward shambling walk loop.
    walk: anim::Caf,
    /// Built once on the first render spawn.
    mats: Vec<Handle<StandardMaterial>>,
}

/// The whole bridge state: runtime, navmesh, per-class character templates and
/// the handle → script/Bevy entity maps.
#[derive(Resource, Default)]
pub struct ZombieBridge {
    rt: Option<ZombieRuntime>,
    nav: Option<NavMesh>,
    /// `EntityClass` → loaded character (or a cached load failure).
    templates: HashMap<String, Option<ZombieTemplate>>,
    /// Logical handle → script-host entity id.
    host_entities: HashMap<u32, EntityId>,
    /// Logical handle → Bevy render entity.
    visuals: HashMap<u32, Entity>,
    /// Script-host entity → last pipe name (transition logging).
    last_pipe: HashMap<EntityId, String>,
    /// Script-host entity → last goal result code (transition logging).
    last_result: HashMap<EntityId, i32>,
    tex_memo: HashMap<String, Handle<Image>>,
    /// Number of spawn/despawn lines still allowed through.
    verbose: u64,
    /// Telemetry timer.
    log_timer: f32,
    /// True once init has been attempted (success or not).
    tried_init: bool,
}

/// One promoted zombie's render-side skinned character.
#[derive(Component)]
struct ZombieVisual {
    handle: u32,
    parts: Vec<AnimPart>,
    skeleton: Skeleton,
    idle: anim::Caf,
    walk: anim::Caf,
    time: f32,
    moving: bool,
}

impl ZombieBridge {
    /// Number of promoted actors (what the HUD counter shows).
    pub fn live_count(&self) -> usize {
        self.rt.as_ref().map(|rt| rt.live_count()).unwrap_or(0)
    }
}

/// Register the bridge. Call inside `run_world`'s play branch once the script
/// host exists.
pub fn install(app: &mut App) {
    app.init_resource::<ZombieBridge>().add_systems(
        Update,
        (
            zombie_bridge_tick.after(crate::move_player),
            zombie_visual_sync,
            animate_zombie_visuals,
            zombie_hud.after(crate::hud_update),
        )
            .chain(),
    );
}

/// Lazy-boot the runtime from the mounted data. Returns `false` until ready.
fn ensure_init(br: &mut ZombieBridge, cry: &CryHost) -> bool {
    if br.rt.is_some() {
        return true;
    }
    if br.tried_init {
        return false;
    }
    br.tried_init = true;
    let vfs = cry.host.ctx.vfs.clone();

    br.nav = navmesh::load_from_vfs(&vfs, "mission0");
    let table = vfs
        .read("libs/class3/zombies/zombietypes.xml")
        .map(|b| ZombieTable::parse(&b))
        .unwrap_or_default();
    let mut tuning = ZombieTuning::default();
    if let Some(b) = vfs.read("libs/class3/rts/zombies.xml") {
        tuning.apply_rts_xml(&b);
    }
    let mut points = Vec::new();
    let mut zones = Vec::new();
    for path in ["zombie_spawns_win.markup", "levels/class3/zombie_spawns_win.markup"] {
        if let Some(b) = vfs.read(path) {
            if let Some(m) = parse_spawn_markup(&b) {
                points = m.points;
                zones = m.zones;
                break;
            }
        }
    }
    let director = SpawnDirector::new(tuning, table, points, zones, 0xC0FFEE);
    let ai_state = ai::installed_state();
    let rt = if let Some(st) = &ai_state {
        let guard = st.borrow();
        ZombieRuntime::new(director, &guard)
    } else {
        ZombieRuntime::new(director, &AiState::default())
    };
    println!(
        "[zombie] runtime: {} spawn points, {} danger zones, {} goal pipes, navmesh={}",
        rt.director.points.len(),
        rt.director.zones.len(),
        rt.executor.pipe_names().count(),
        br.nav.is_some()
    );
    br.rt = Some(rt);
    true
}

/// Load the character for `class` once and cache it (including failures, so a
/// missing asset is not re-read every frame).
fn ensure_template(br: &mut ZombieBridge, cry: &CryHost, class: &str) -> bool {
    if let Some(t) = br.templates.get(class) {
        return t.is_some();
    }
    let vfs = cry.host.ctx.vfs.clone();
    let def = zombie_actor::character_for_class(class);
    let loaded = load_template(&vfs, def);
    match &loaded {
        Some(t) => println!(
            "[zombie] character {class} -> {} ({} meshes, {} slots, {} textures)",
            def.path,
            t.model.meshes.len(),
            t.slots.len(),
            t.tex_bytes.len()
        ),
        None => eprintln!("[zombie] WARNING: character {class} ({}) failed to load", def.path),
    }
    let ok = loaded.is_some();
    br.templates.insert(class.to_string(), loaded);
    ok
}

/// Assemble a class's character with its full material palette. `.cdf` files go
/// through the shared CDF assembler (skeleton + all attachments + `.mtl`
/// overrides); the clip bank is then rebuilt filtered to the class rig so freaks
/// do not animate on human clips.
fn load_template(vfs: &Vfs, def: zombie_actor::CharacterDef) -> Option<ZombieTemplate> {
    if def.path.to_ascii_lowercase().ends_with(".cdf") {
        let (model, slots, tex_bytes, skeleton, influences, _caf, bank) =
            crate::load_cdf_character(vfs, def.path)?;
        let idle = pick_clip(vfs, def.rig, IDLE_CLIPS)
            .or_else(|| bank.caf_for(anim::Locomotion::Idle).cloned())?;
        let walk = pick_clip(vfs, def.rig, WALK_CLIPS)
            .or_else(|| bank.caf_for(anim::Locomotion::Walk).cloned())
            .unwrap_or_else(|| idle.clone());
        Some(ZombieTemplate { model, slots, tex_bytes, skeleton, influences, idle, walk, mats: Vec::new() })
    } else {
        crate::load_any_character(vfs, def.path, Some(def.rig)).map(
            |(model, slots, tex_bytes, skeleton, influences, _caf, bank)| {
                let idle = pick_clip(vfs, def.rig, IDLE_CLIPS)
                    .or_else(|| bank.caf_for(anim::Locomotion::Idle).cloned())
                    .unwrap_or_else(|| anim::Caf { start: 0, end: 1, secs_per_tick: 1.0, controllers: Vec::new() });
                let walk = pick_clip(vfs, def.rig, WALK_CLIPS)
                    .or_else(|| bank.caf_for(anim::Locomotion::Walk).cloned())
                    .unwrap_or_else(|| idle.clone());
                ZombieTemplate {
                    model,
                    slots,
                    tex_bytes,
                    skeleton,
                    influences,
                    idle,
                    walk,
                    mats: Vec::new(),
                }
            },
        )
    }
}

/// Idle clip preferences, best first. The zombie `.$caf` set is huge; the
/// generic `ClipSet::pick` lands on `react/burning_idle` (a writhing
/// on-fire reaction), so the neutral loops are named explicitly.
const IDLE_CLIPS: &[&str] = &["relaxed_basic_idle", "wander_idle", "dormant_idle", "idle"];
/// Walk clip preferences, best first.
const WALK_CLIPS: &[&str] =
    &["wander_walk_01", "wander_walk", "relaxed_walk_forward_slow", "walk_forward", "walk"];

/// Find and parse the first `.$caf` under `rig` whose path contains one of
/// `needles` (in preference order). Used to pin the zombie locomotion loops the
/// shipped scripts expect instead of whatever the generic bank picks.
fn pick_clip(vfs: &Vfs, rig: &str, needles: &[&str]) -> Option<anim::Caf> {
    let rig = rig.to_ascii_lowercase();
    let mut paths: Vec<String> = vfs.paths_with_ext(".$caf");
    paths.sort();
    for needle in needles {
        for p in &paths {
            let pl = p.to_ascii_lowercase();
            if pl.contains(&rig) && pl.contains(needle) {
                if let Some(b) = vfs.read(p) {
                    if let Some(c) = anim::Caf::parse(&b) {
                        return Some(c);
                    }
                }
            }
        }
    }
    None
}

/// Player feet position (CryEngine space). Falls back to `CryPlayerPos`.
fn player_pos(player: Option<&Player>, pos: Option<&CryPlayerPos>) -> Option<[f32; 3]> {
    if let Some(p) = player {
        return Some(p.mstate.pos);
    }
    pos.and_then(|p| p.0)
}

fn zombie_bridge_tick(
    mut commands: Commands,
    cry: Option<NonSendMut<CryHost>>,
    player: Option<Res<Player>>,
    cpos: Option<Res<CryPlayerPos>>,
    collision: Option<Res<Collision>>,
    time: Res<Time>,
    mut bridge: ResMut<ZombieBridge>,
) {
    let Some(mut cry) = cry else { return };
    let Some(ppos) = player_pos(player.as_deref(), cpos.as_deref()) else { return };
    if !ensure_init(&mut bridge, &cry) {
        return;
    }
    let dt = time.delta_secs().min(0.1);
    let collision = collision.as_ref().map(|c| &c.0);

    {
        let ZombieBridge { rt, nav, .. } = &mut *bridge;
        let rt = rt.as_mut().unwrap();
        rt.update_director(dt, ppos, nav.as_ref(), collision);
    }
    reconcile_host_entities(&mut bridge, &mut cry, &mut commands);
    // Snapshot feet positions before the movement steps so the world sweep can
    // cast the frame's delta.
    let prev: Vec<[f32; 3]> = bridge.rt.as_ref().unwrap().zombies.iter().map(|z| z.pos).collect();
    {
        let ZombieBridge { rt, nav, .. } = &mut *bridge;
        let rt = rt.as_mut().unwrap();
        rt.step_logical(dt, ppos, collision);
        rt.step_ai(dt, ppos, nav.as_ref(), collision);
        apply_actor_physics(rt, ppos, collision, &prev);
    }

    // Push actor transforms into the script world and route raised signals to
    // the zombie's `OnSignal` handler.
    let signals: Vec<_> = {
        let rt = bridge.rt.as_mut().unwrap();
        let signals: Vec<_> = rt.signals.drain(..).collect();
        for z in rt.zombies.iter() {
            if let Some(e) = z.entity {
                let world = cry.host.ctx.world.clone();
                let mut w = world.borrow_mut();
                w.set_pos(e, z.pos);
                let half = 0.5 * z.heading;
                w.set_rot(e, [0.0, 0.0, half.sin(), half.cos()]);
            }
        }
        signals
    };
    let mut signal_log = 0u32;
    for s in &signals {
        if signal_log < 12 {
            println!("[zombie] signal {} -> entity {} (filter {})", s.name, s.entity, s.filter);
            signal_log += 1;
        }
        let _ = cry.host.event(s.entity, "OnSignal", (s.name.clone(), s.filter, s.mode));
    }

    log_transitions(&mut bridge);
    bridge.log_timer += dt;
    if bridge.log_timer >= 2.0 {
        bridge.log_timer = 0.0;
        let rt = bridge.rt.as_ref().unwrap();
        let live = rt.live_count();
        let nearest = rt
            .zombies
            .iter()
            .map(|z| {
                let dx = z.pos[0] - ppos[0];
                let dy = z.pos[1] - ppos[1];
                (dx * dx + dy * dy).sqrt()
            })
            .fold(f32::INFINITY, f32::min);
        println!(
            "[zombie] live={} logical={} total={} spawned={} despawned={} nearest={:.1}m",
            live,
            rt.zombies.len() - live,
            rt.zombies.len(),
            rt.spawned,
            rt.despawned,
            if nearest.is_finite() { nearest } else { -1.0 },
        );
    }
}

/// Resolve the actors' living capsules for this frame:
/// 1. sweep each zombie's movement delta through the static collision world
///    (terrain + placed proxies) so it slides instead of walking through walls,
/// 2. separate overlapping actor-actor and actor-player capsules, and
/// 3. re-snap every foot to the ground after the horizontal pushes.
///
/// `prev` is the feet position before this frame's movement, parallel to
/// `rt.zombies` (the record set is stable across the movement steps).
fn apply_actor_physics(
    rt: &mut ZombieRuntime,
    player: [f32; 3],
    collision: Option<&CollisionWorld>,
    prev: &[[f32; 3]],
) {
    if let Some(c) = collision {
        for (z, p0) in rt.zombies.iter_mut().zip(prev.iter().copied()) {
            let cap = zombie_actor::capsule_for_class(&z.entity_class);
            let delta = [z.pos[0] - p0[0], z.pos[1] - p0[1], z.pos[2] - p0[2]];
            if delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] > 1e-8 {
                z.pos = zombie_actor::sweep_body(c, p0, delta, cap);
            }
            z.pos[2] = ground(c, z.pos[0], z.pos[1], z.pos[2]);
        }
    }

    // Actor-actor + actor-player separation. The player is a fixed body so the
    // player controller (main.rs) stays authoritative.
    let mut bodies: Vec<Body> = rt
        .zombies
        .iter()
        .map(|z| {
            let cap = zombie_actor::capsule_for_class(&z.entity_class);
            Body { pos: z.pos, radius: cap.radius, height: cap.height, fixed: false }
        })
        .collect();
    bodies.push(Body {
        pos: player,
        radius: sod_core::class3_physics::PLAYER_CAPSULE_RADIUS,
        height: sod_core::class3_physics::PLAYER_CAPSULE_HEIGHT,
        fixed: true,
    });
    zombie_actor::separate_bodies(&mut bodies, SEPARATION_ITERATIONS);
    for (z, b) in rt.zombies.iter_mut().zip(bodies.iter()) {
        z.pos[0] = b.pos[0];
        z.pos[1] = b.pos[1];
    }
    if let Some(c) = collision {
        for z in rt.zombies.iter_mut() {
            z.pos[2] = ground(c, z.pos[0], z.pos[1], z.pos[2]);
        }
    }
}

/// Ground height from the collision world (same probe the runtime uses):
/// downward ray from just above the current height, falling back to terrain.
fn ground(c: &CollisionWorld, x: f32, y: f32, fallback: f32) -> f32 {
    let top = fallback.max(c.terrain_height(x, y).unwrap_or(fallback)) + 2.0;
    if let Some(h) = c.raycast([x, y, top], [0.0, 0.0, -1.0], 12.0) {
        return h.point[2];
    }
    c.terrain_height(x, y).unwrap_or(fallback)
}

/// Create/destroy script-host actors so every live record owns an entity and no
/// stale entity outlives its record.
fn reconcile_host_entities(br: &mut ZombieBridge, cry: &mut CryHost, commands: &mut Commands) {
    let live: HashSet<u32> = br
        .rt
        .as_ref()
        .unwrap()
        .zombies
        .iter()
        .filter(|z| z.live)
        .map(|z| z.handle)
        .collect();

    let stale: Vec<u32> =
        br.host_entities.keys().copied().filter(|h| !live.contains(h)).collect();
    for h in stale {
        if let Some(id) = br.host_entities.remove(&h) {
            cry.host.ctx.world.borrow_mut().remove(id);
            br.last_pipe.remove(&id);
            br.last_result.remove(&id);
        }
        if let Some(e) = br.visuals.remove(&h) {
            commands.entity(e).despawn();
        }
        if br.verbose < VERBOSE_SPAWN_LOG {
            br.verbose += 1;
            println!("[zombie] despawn #{}", h);
        }
    }

    let rt = br.rt.as_mut().unwrap();
    for z in rt.zombies.iter_mut() {
        if !z.live || z.entity.is_some() {
            continue;
        }
        let name = format!("ULZombie{}", z.handle);
        match cry.host.spawn_actor(&z.entity_class, &z.script, &name, z.pos, false) {
            Ok(id) => {
                z.entity = Some(id);
                br.host_entities.insert(z.handle, id);
                if br.verbose < VERBOSE_SPAWN_LOG {
                    br.verbose += 1;
                    println!(
                        "[zombie] spawn #{} {} <{}> at [{:.1}, {:.1}, {:.1}] cluster {}",
                        z.handle, z.entity_class, z.script, z.pos[0], z.pos[1], z.pos[2], z.cluster
                    );
                }
            }
            Err(e) => {
                eprintln!("[zombie] spawn_actor #{} {}: {e}", z.handle, z.entity_class);
                z.live = false;
            }
        }
    }
}

/// Log pipe transitions for live actors (goal-result changes are tracked for
/// telemetry and printed alongside).
fn log_transitions(br: &mut ZombieBridge) {
    let rt = br.rt.as_ref().unwrap();
    for z in rt.zombies.iter() {
        let Some(entity) = z.entity else { continue };
        let Some(u) = rt.executor.user(entity) else { continue };
        if br.last_pipe.get(&entity) != Some(&u.pipe) {
            println!("[zombie] pipe #{} entity {} -> {} (last {:?})", z.handle, entity, u.pipe, u.last_result);
            br.last_pipe.insert(entity, u.pipe.clone());
        }
        let code = u.last_result.code();
        if br.last_result.get(&entity) != Some(&code) {
            br.last_result.insert(entity, code);
        }
    }
}

/// Ensure every live record has a Bevy visual of its own class; lifecycle only
/// (positions and skinning are handled in `animate_zombie_visuals`).
fn zombie_visual_sync(
    mut commands: Commands,
    cry: Option<NonSendMut<CryHost>>,
    mut bridge: ResMut<ZombieBridge>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    cache: Option<Res<TextureCacheRes>>,
) {
    if bridge.rt.is_none() {
        return;
    }
    let Some(cry) = cry else { return };

    // Which classes currently need a visual, and load their characters.
    let classes: HashSet<String> = bridge
        .rt
        .as_ref()
        .unwrap()
        .zombies
        .iter()
        .filter(|z| z.live)
        .map(|z| z.entity_class.clone())
        .collect();
    for class in &classes {
        ensure_template(&mut bridge, &cry, class);
    }

    let cache = cache.as_deref().map(|c| &c.0);
    for class in &classes {
        ensure_materials(&mut bridge, class, &mut materials, &mut images, cache);
    }

    let wanted: Vec<(u32, String, [f32; 3], f32)> = bridge
        .rt
        .as_ref()
        .unwrap()
        .zombies
        .iter()
        .filter(|z| z.live && !bridge.visuals.contains_key(&z.handle))
        .map(|z| (z.handle, z.entity_class.clone(), z.pos, z.heading))
        .collect();

    let base = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    for (handle, class, pos, heading) in wanted {
        let Some(t) = bridge.templates.get(&class).and_then(|t| t.as_ref()) else { continue };
        let (parts, skeleton, idle, walk) =
            (build_parts(t, &mut meshes), t.skeleton.clone(), t.idle.clone(), t.walk.clone());
        let tf = Transform {
            translation: base * Vec3::new(pos[0], pos[1], pos[2]),
            rotation: base * Quat::from_rotation_z(sod_core::class3_move::render_yaw(heading)),
            scale: Vec3::ONE,
        };
        let child_parts = parts.clone();
        let entity = commands
            .spawn((
                tf,
                Visibility::default(),
                ZombieVisual {
                    handle,
                    parts,
                    skeleton,
                    idle,
                    walk,
                    time: 0.0,
                    moving: false,
                },
            ))
            .with_children(|parent| {
                for p in &child_parts {
                    parent.spawn((Mesh3d(p.handle.clone()), MeshMaterial3d(p.mat.clone())));
                }
            })
            .id();
        bridge.visuals.insert(handle, entity);
    }

    let live: HashSet<u32> =
        bridge.rt.as_ref().unwrap().zombies.iter().filter(|z| z.live).map(|z| z.handle).collect();
    let stale: Vec<u32> =
        bridge.visuals.keys().copied().filter(|h| !live.contains(h)).collect();
    for h in stale {
        if let Some(e) = bridge.visuals.remove(&h) {
            commands.entity(e).despawn();
        }
    }
}

/// Build shared StandardMaterials from a class's character palette once.
fn ensure_materials(
    br: &mut ZombieBridge,
    class: &str,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    cache: Option<&sod_core::cache::TextureCache>,
) {
    let ZombieBridge { templates, tex_memo, .. } = br;
    let Some(t) = templates.get_mut(class).and_then(|t| t.as_mut()) else { return };
    if !t.mats.is_empty() {
        return;
    }
    let mut memo = std::mem::take(tex_memo);
    let mut handles = Vec::new();
    for slot in &t.slots {
        // Neutral base so the authored Diffuse texture is not tinted brown.
        let mut m = StandardMaterial {
            base_color: Color::srgb(0.85, 0.82, 0.78),
            perceptual_roughness: 0.9,
            cull_mode: None,
            ..default()
        };
        if let Some(tp) = slot.texture("Diffuse") {
            if let Some(bytes) = t.tex_bytes.get(tp) {
                if let Some(h) = crate::cached_tex(images, cache, &mut memo, tp, bytes, true) {
                    m.base_color_texture = Some(h);
                }
            }
        }
        handles.push(materials.add(m));
    }
    if handles.is_empty() {
        handles.push(materials.add(StandardMaterial {
            base_color: Color::srgb(0.72, 0.68, 0.62),
            perceptual_roughness: 0.9,
            cull_mode: None,
            ..default()
        }));
    }
    t.mats = handles;
    *tex_memo = memo;
}

/// Copy a class character's meshes into per-instance Bevy meshes so each can be
/// skinned independently. Mirrors the NPC spawn path in `main.rs`.
fn build_parts(t: &ZombieTemplate, meshes: &mut Assets<Mesh>) -> Vec<AnimPart> {
    let mut parts = Vec::new();
    for (mi, m) in t.model.meshes.iter().enumerate() {
        if m.positions.is_empty() {
            continue;
        }
        let infl = t.influences.get(mi).cloned().unwrap_or_default();
        let groups: Vec<(usize, Vec<u32>)> = if m.submeshes.is_empty() {
            vec![(0, m.indices.clone())]
        } else {
            m.submeshes.iter().map(|s| (s.material as usize, s.indices.clone())).collect()
        };
        for (mat_i, idx) in groups {
            if idx.is_empty() {
                continue;
            }
            let mut bm = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
            bm.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(m.positions.clone()),
            );
            bm.insert_attribute(
                Mesh::ATTRIBUTE_NORMAL,
                VertexAttributeValues::Float32x3(m.normals.clone()),
            );
            if m.uvs.len() == m.positions.len() {
                bm.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(m.uvs.clone()));
            }
            bm.insert_indices(Indices::U32(idx));
            let handle = meshes.add(bm);
            let mat = t.mats.get(mat_i).or_else(|| t.mats.first()).cloned().unwrap_or_default();
            parts.push(AnimPart {
                handle,
                bind_pos: m.positions.clone(),
                bind_nrm: m.normals.clone(),
                infl: infl.clone(),
                mat,
            });
        }
    }
    parts
}

/// Position and skin every zombie visual; clip is idle or walk.
fn animate_zombie_visuals(
    time: Res<Time>,
    bridge: Res<ZombieBridge>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut q: Query<(&mut Transform, &mut ZombieVisual)>,
) {
    let base = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let rt = bridge.rt.as_ref();
    for (mut tf, mut v) in &mut q {
        if let Some(rt) = rt {
            if let Some(z) = rt.zombies.iter().find(|z| z.handle == v.handle) {
                tf.translation = base * Vec3::new(z.pos[0], z.pos[1], z.pos[2]);
                tf.rotation =
                    base * Quat::from_rotation_z(sod_core::class3_move::render_yaw(z.heading));
                v.moving = z.moving;
            }
        }
        if v.parts.is_empty() {
            continue;
        }
        let moving = v.moving;
        v.time += time.delta_secs();
        let caf = if moving { &v.walk } else { &v.idle };
        let dur = caf.duration_secs().max(1.0 / 30.0);
        let tick = caf.tick_from_seconds(v.time % dur);
        let sample = caf.sample_tick(tick);
        let posed = v.skeleton.pose_anchored(&sample);
        let skin_m = v.skeleton.skin_matrices(&posed);
        for part in &v.parts {
            let npos = skin::deform(&part.bind_pos, &part.infl, &skin_m);
            let nnrm = skin::deform_normals(&part.bind_nrm, &part.infl, &skin_m);
            if let Some(mut m) = meshes.get_mut(&part.handle) {
                m.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(npos));
                m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(nnrm));
            }
        }
    }
}

/// Feed the native zombie count (and status line) into the HUD. `main.rs`'s
/// `hud_update` only counts the legacy `Npc` entities, so the counter would
/// otherwise read zero while the spawn director is running. Runs after it.
fn zombie_hud(
    bridge: Res<ZombieBridge>,
    hud: Option<ResMut<crate::HudRes>>,
    mut lines: Query<(&mut Text, &crate::HudLine)>,
) {
    let Some(mut hud) = hud else { return };
    let n = bridge.live_count();
    hud.0.zombie_count = n as u32;
    let hp = hud.0.vitals.hp;
    let maxhp = hud.0.vitals.max_hp;
    let stam = hud.0.vitals.stamina;
    for (mut t, line) in &mut lines {
        if matches!(line, crate::HudLine::Status) {
            t.0 = format!("HP {hp:.0}/{maxhp:.0}   Stamina {stam:.0}   Zombies {n}");
        }
    }
}
