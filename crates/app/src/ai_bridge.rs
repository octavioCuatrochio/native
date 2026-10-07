//! Bevy side of the zombie ecology: runs the native [`ZombieRuntime`] against
//! the live world, spawns the original zombie actor scripts through the
//! [`crate::cry_bridge::CryHost`], and draws/skins the promoted actors.
//!
//! Pipeline per frame (ordered after `move_player`):
//!
//! 1. `zombie_bridge_tick` — lazy-boot the runtime from the mounted paks
//!    (`ZombieTypes.xml`, `RTS/Zombies.xml`, `zombie_spawns_win.markup`, the
//!    AI goal-pipe registry and the `.bai` navmesh), run the spawn director
//!    over a [`WorldView`](sod_core::cry::zombie_spawn::WorldView), reconcile
//!    live actors (spawn/despawn through `ScriptHost::spawn_actor` using the
//!    recovered class → script map), step each live actor's goal pipe, and
//!    route the raised signals back to the actor's script.
//! 2. `zombie_visual_sync` — spawn a skinned Bevy entity per newly promoted
//!    worker by copying the zombie character's meshes/palette.
//! 3. `animate_zombie_visuals` — position it and play the idle/walk clip chosen
//!    from the runtime's `moving` flag.
//!
//! The script host remains the authority for zombie behaviour; this file only
//! feeds it positions and runs the goal pipes it authored.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use sod_core::cry::ai;
use sod_core::cry::ai_data::AiState;
use sod_core::cry::navmesh::{self, NavMesh};
use sod_core::cry::world::EntityId;
use sod_core::cry::zombie_runtime::ZombieRuntime;
use sod_core::cry::zombie_spawn::{parse_spawn_markup, SpawnDirector, ZombieTable, ZombieTuning};
use sod_core::material::MaterialSlot;
use sod_core::mesh::DecodedModel;
use sod_core::skin::{self, BoneWeight, Skeleton};
use sod_core::{anim, character};
use std::collections::{HashMap, HashSet};

use crate::cry_bridge::{CryHost, CryPlayerPos};
use crate::{AnimPart, Collision, Player, TextureCacheRes};

/// The zombie character the native spawn system promotes (`013194e0:263`):
/// class 0 is the Grunt whose body is `characters/male/zombie/skin/skin.chr`.
const ZOMBIE_CHARACTER: &str = "characters/male/zombie/skin/skin.chr";
/// Rig name the clip bank resolves against.
const ZOMBIE_RIG: &str = "zombie_male";
/// How many individual spawn/despawn lines to print before summarising.
const VERBOSE_SPAWN_LOG: u64 = 60;

/// Loaded zombie character, shared by every promoted actor.
struct ZombieTemplate {
    model: DecodedModel,
    slots: Vec<MaterialSlot>,
    tex_bytes: HashMap<String, Vec<u8>>,
    skeleton: Skeleton,
    influences: Vec<Vec<[BoneWeight; 4]>>,
    caf: anim::Caf,
    bank: character::ClipBank,
    /// Built once on the first render spawn.
    mats: Vec<Handle<StandardMaterial>>,
}

/// The whole bridge state: runtime, navmesh, character template and the
/// handle → script/Bevy entity maps.
#[derive(Resource, Default)]
pub struct ZombieBridge {
    rt: Option<ZombieRuntime>,
    nav: Option<NavMesh>,
    template: Option<ZombieTemplate>,
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
    caf: anim::Caf,
    bank: character::ClipBank,
    loc: anim::Locomotion,
    time: f32,
    moving: bool,
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

    br.template = crate::load_any_character(&vfs, ZOMBIE_CHARACTER, Some(ZOMBIE_RIG)).map(
        |(model, slots, tex_bytes, skeleton, influences, caf, bank)| ZombieTemplate {
            model,
            slots,
            tex_bytes,
            skeleton,
            influences,
            caf,
            bank,
            mats: Vec::new(),
        },
    );
    if br.template.is_none() {
        eprintln!("[zombie] WARNING: zombie character {ZOMBIE_CHARACTER} failed to load");
    }
    true
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
    {
        let ZombieBridge { rt, nav, .. } = &mut *bridge;
        let rt = rt.as_mut().unwrap();
        rt.step_logical(dt, ppos, collision);
        rt.step_ai(dt, ppos, nav.as_ref(), collision);
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

/// Ensure every live record has a Bevy visual; lifecycle only (positions and
/// skinning are handled in `animate_zombie_visuals`).
fn zombie_visual_sync(
    mut commands: Commands,
    mut bridge: ResMut<ZombieBridge>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    cache: Option<Res<TextureCacheRes>>,
) {
    if bridge.rt.is_none() || bridge.template.is_none() {
        return;
    }
    ensure_materials(&mut bridge, &mut materials, &mut images, cache.as_deref().map(|c| &c.0));

    let wanted: Vec<(u32, [f32; 3], f32)> = bridge
        .rt
        .as_ref()
        .unwrap()
        .zombies
        .iter()
        .filter(|z| z.live && !bridge.visuals.contains_key(&z.handle))
        .map(|z| (z.handle, z.pos, z.heading))
        .collect();

    let base = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    for (handle, pos, heading) in wanted {
        let (parts, skeleton, caf, bank) = {
            let t = bridge.template.as_ref().unwrap();
            (build_parts(t, &mut meshes), t.skeleton.clone(), t.caf.clone(), t.bank.clone())
        };
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
                    caf,
                    bank,
                    loc: anim::Locomotion::Idle,
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

/// Build shared StandardMaterials from the zombie character's palette once.
fn ensure_materials(
    br: &mut ZombieBridge,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    cache: Option<&sod_core::cache::TextureCache>,
) {
    if br.template.as_ref().map(|t| !t.mats.is_empty()).unwrap_or(true) {
        return;
    }
    let mut memo = std::mem::take(&mut br.tex_memo);
    let mut handles = Vec::new();
    {
        let t = br.template.as_ref().unwrap();
        for slot in &t.slots {
            let mut m = StandardMaterial {
                base_color: Color::srgb(0.62, 0.55, 0.5),
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
    }
    br.tex_memo = memo;
    if handles.is_empty() {
        handles.push(materials.add(StandardMaterial {
            base_color: Color::srgb(0.62, 0.55, 0.5),
            perceptual_roughness: 0.9,
            cull_mode: None,
            ..default()
        }));
    }
    br.template.as_mut().unwrap().mats = handles;
}

/// Copy the zombie character's meshes into per-instance Bevy meshes so each can
/// be skinned independently. Mirrors the NPC spawn path in `main.rs`.
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
        let want = if v.moving { anim::Locomotion::Walk } else { anim::Locomotion::Idle };
        if want != v.loc {
            if let Some(c) = v.bank.caf_for(want).cloned() {
                v.caf = c;
                v.loc = want;
            }
        }
        v.time += time.delta_secs();
        let dur = v.caf.duration_secs().max(1.0 / 30.0);
        let tick = v.caf.tick_from_seconds(v.time % dur);
        let sample = v.caf.sample_tick(tick);
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
