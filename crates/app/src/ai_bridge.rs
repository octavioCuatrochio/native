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
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use std::path::{Path, PathBuf};
use sod_core::cry::ai;
use sod_core::cry::ai_data::AiState;
use sod_core::cry::combat::{self as cry_combat, DamageTarget, HitMaterial, MeleeProfile};
use sod_core::cry::combat_perception::{self as perception, Senses, SoundStimulus};
use sod_core::game::ZombieKind;
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
use sod_core::sm_runtime::{self, SmCommand, SmEnv, SmInstance};
use sod_core::statemachine::StateMachine;
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
    /// Best available death clip (played once on the corpse).
    death: Option<anim::Caf>,
    /// Best available hit-reaction clip (played once on a hit).
    react: Option<anim::Caf>,
    /// Built once on the first render spawn.
    mats: Vec<Handle<StandardMaterial>>,
}

/// Per-live-handle combat state driven by the native perception/attack loop.
#[derive(Clone, Debug, Default)]
struct ZombieCombat {
    /// Seconds until the next melee strike is allowed (the strike gap).
    attack_cd: f32,
    /// True once the zombie's perception acquired the player.
    alerted: bool,
    /// Hit-reaction timer (seconds); drives the visible knock-back.
    reaction: f32,
}

/// One in-flight state-machine strike for a live zombie. The machine is forced
/// into an attack state (chosen by perception + range) and the hit is emitted
/// when that state's strike action (`FxMaterialEffect`, e.g. `zed_clawattack`)
/// runs; after the strike window the machine is handed back to the graph.
#[derive(Clone, Debug)]
struct SmStrike {
    /// Attack state the machine was forced into.
    state: u32,
    /// Seconds left in the forced strike window.
    left: f32,
    /// The strike event has fired this swing (the hit is applied once).
    struck: bool,
    /// Damage tier named by the attack state's own action block.
    profile: MeleeProfile,
}

/// Player attack intent published by `main.rs::player_attack` and resolved by
/// the bridge (which owns the actor health and the script host).
#[derive(Resource, Default)]
pub struct PlayerAttackRequest {
    /// `Some(base_damage)` when a melee swing landed this frame.
    pub melee: Option<f32>,
    /// `Some(base_damage)` when a firearm round fired this frame.
    pub shot: Option<f32>,
    /// CryEngine-space player position.
    pub origin: [f32; 3],
    /// Player body yaw (radians).
    pub yaw: f32,
}

/// The outcome of one frame of combat, for HUD/telemetry.
#[derive(Resource, Default, Debug, Clone)]
pub struct CombatStats {
    pub hits_dealt: u64,
    pub hits_taken: u64,
    pub kills: u64,
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
    /// Per-live-handle perception/attack state.
    combat: HashMap<u32, ZombieCombat>,
    /// Leaked parsed state machines, keyed by actor family (`zombie`,
    /// `freak_feral`, ...). Each live zombie runs the machine for its class.
    sm_machines: HashMap<&'static str, &'static StateMachine>,
    /// Logical handle -> the zombie's running ULSM instance.
    sm_instances: HashMap<u32, SmInstance<'static>>,
    /// Logical handle -> the in-flight attack (present while a strike runs).
    sm_strikes: HashMap<u32, SmStrike>,
    /// Logical handle -> strike clip to start on the visual (drained by
    /// `animate_zombie_visuals`).
    pending_strikes: HashMap<u32, anim::Caf>,
    /// Dead handles kept as corpses for this many seconds before despawn.
    dead: HashMap<u32, f32>,
    /// Handles killed this run; never re-promoted (the director does not model
    /// kills, so without this a dead slot would respawn on the next tick).
    killed: HashSet<u32>,
    /// Player actor health was pushed into the script host.
    player_ready: bool,
    /// Monotonic frame counter (deterministic hit-material selection).
    frame: u64,
    /// Cumulative combat counters.
    stats: CombatStats,
    /// One-shot combat screenshots (`SOD_HITSHOT`) already taken.
    shot_player: bool,
    shot_zombie: bool,
}

/// One promoted zombie's render-side skinned character.
#[derive(Component)]
struct ZombieVisual {
    handle: u32,
    parts: Vec<AnimPart>,
    skeleton: Skeleton,
    idle: anim::Caf,
    walk: anim::Caf,
    death: Option<anim::Caf>,
    react: Option<anim::Caf>,
    /// One-shot strike clip requested by the zombie's own state machine
    /// (a `MachineDriven.Combat.*` fragment). Played over the graph.
    strike: Option<anim::Caf>,
    time: f32,
    moving: bool,
    /// True once the actor is dead (plays the death clip, then holds).
    dying: bool,
    /// True while a hit-reaction clip is playing.
    reacting: bool,
    /// True while the state-machine strike fragment is playing.
    striking: bool,
    /// Previous frame's reaction timer (to restart the clip on a fresh hit).
    last_react: f32,
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
    app.init_resource::<ZombieBridge>()
        .init_resource::<PlayerAttackRequest>()
        .init_resource::<CombatStats>()
        .add_systems(
            Update,
            (
                zombie_bridge_tick.after(crate::move_player).after(crate::player_attack),
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

    // The shipped per-class ULSM machines. A live zombie runs its own class's
    // machine so its `MachineDriven.Combat.*` attack states drive the strike.
    for (family, rel) in [
        ("zombie", "libs/statemachines/zombie/general.win.sm"),
        ("freak_feral", "libs/statemachines/freak_feral/general.win.sm"),
        ("freak_juggernaut", "libs/statemachines/freak_juggernaut/general.win.sm"),
        ("freak_bloater", "libs/statemachines/freak_bloater/general.win.sm"),
        ("freak_screamer", "libs/statemachines/freak_screamer/general.win.sm"),
    ] {
        let bytes = vfs.read(rel).or_else(|| vfs.read(&format!("gamedata/{rel}")));
        match bytes.and_then(|b| sod_core::statemachine::parse_sm(&b)) {
            Some(sm) => {
                let leaked: &'static StateMachine = Box::leak(Box::new(sm));
                println!(
                    "[zombie] state machine {}: {} states, start [{}] {}",
                    family,
                    leaked.states.len(),
                    leaked.start_state,
                    leaked.states.get(leaked.start_state as usize).map(|s| s.name.as_str()).unwrap_or("?")
                );
                br.sm_machines.insert(family, leaked);
            }
            None => eprintln!("[zombie] WARNING: state machine {rel} unavailable"),
        }
    }
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
        let death = pick_clip(vfs, def.rig, DEATH_CLIPS);
        let react = pick_clip(vfs, def.rig, REACT_CLIPS);
        Some(ZombieTemplate { model, slots, tex_bytes, skeleton, influences, idle, walk, death, react, mats: Vec::new() })
    } else {
        crate::load_any_character(vfs, def.path, Some(def.rig)).map(
            |(model, slots, tex_bytes, skeleton, influences, _caf, bank)| {
                let idle = pick_clip(vfs, def.rig, IDLE_CLIPS)
                    .or_else(|| bank.caf_for(anim::Locomotion::Idle).cloned())
                    .unwrap_or_else(|| anim::Caf { start: 0, end: 1, secs_per_tick: 1.0, controllers: Vec::new() });
                let walk = pick_clip(vfs, def.rig, WALK_CLIPS)
                    .or_else(|| bank.caf_for(anim::Locomotion::Walk).cloned())
                    .unwrap_or_else(|| idle.clone());
                let death = pick_clip(vfs, def.rig, DEATH_CLIPS);
                let react = pick_clip(vfs, def.rig, REACT_CLIPS);
                ZombieTemplate {
                    model,
                    slots,
                    tex_bytes,
                    skeleton,
                    influences,
                    idle,
                    walk,
                    death,
                    react,
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
/// Death clip preferences, best first (react/fall first, then down/death).
const DEATH_CLIPS: &[&str] =
    &["react_death", "death_01", "death", "react/fall", "fall_forward", "death_front"];
/// Hit-reaction clip preferences, best first.
const REACT_CLIPS: &[&str] =
    &["react_mid_front", "react_mid", "react_light_front", "stagger", "flinch", "react"];

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
    mut player: Option<ResMut<Player>>,
    cpos: Option<Res<CryPlayerPos>>,
    collision: Option<Res<Collision>>,
    time: Res<Time>,
    mut req: ResMut<PlayerAttackRequest>,
    mut sm: Option<ResMut<crate::PlayerSm>>,
    mut bridge: ResMut<ZombieBridge>,
) {
    let Some(mut cry) = cry else { return };
    let Some(ppos) = player_pos(player.as_deref(), cpos.as_deref()) else { return };
    if !ensure_init(&mut bridge, &cry) {
        return;
    }
    let dt = time.delta_secs().min(0.1);
    let collision = collision.as_ref().map(|c| &c.0);
    ensure_player_health(&mut bridge, &mut cry);

    {
        let ZombieBridge { rt, nav, .. } = &mut *bridge;
        let rt = rt.as_mut().unwrap();
        rt.update_director(dt, ppos, nav.as_ref(), collision);
    }
    reconcile_host_entities(&mut bridge, &mut cry, &mut commands);
    // Snapshot feet positions before the movement steps so the world sweep can
    // cast the frame's delta.
    let prev: Vec<[f32; 3]> = bridge.rt.as_ref().unwrap().zombies.iter().map(|z| z.pos).collect();
    let killed = bridge.killed.clone();
    {
        let ZombieBridge { rt, nav, .. } = &mut *bridge;
        let rt = rt.as_mut().unwrap();
        rt.step_logical(dt, ppos, collision);
        rt.step_ai(dt, ppos, nav.as_ref(), collision);
        apply_actor_physics(rt, ppos, collision, &prev, &killed);
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

    // The original combat loop: perception -> target acquisition -> melee hits
    // through the game-rules damage pipeline.
    let request = std::mem::take(&mut *req);
    let hitshot = std::env::var("SOD_HITSHOT").ok().map(PathBuf::from);
    combat_step(
        &mut bridge,
        &mut cry,
        &mut commands,
        &hitshot,
        ppos,
        collision,
        dt,
        request,
        sm.as_deref_mut(),
        player.as_deref_mut(),
    );

    // Retire expired corpses.
    let expired: Vec<u32> =
        bridge.dead.iter().filter(|(_, t)| **t <= 0.0).map(|(h, _)| *h).collect();
    for h in expired {
        bridge.dead.remove(&h);
        if let Some(rt) = bridge.rt.as_mut() {
            if let Some(z) = rt.zombies.iter_mut().find(|z| z.handle == h) {
                z.live = false;
            }
        }
    }

    // Drop state-machine state for handles the director has removed.
    {
        let present: HashSet<u32> =
            bridge.rt.as_ref().unwrap().zombies.iter().map(|z| z.handle).collect();
        bridge.sm_instances.retain(|h, _| present.contains(h));
        bridge.sm_strikes.retain(|h, _| present.contains(h));
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
        let alerted = bridge.combat.values().filter(|c| c.alerted).count();
        println!(
            "[zombie] live={} logical={} total={} spawned={} despawned={} alerted={} hits_dealt={} hits_taken={} kills={} nearest={:.1}m",
            live,
            rt.zombies.len() - live,
            rt.zombies.len(),
            rt.spawned,
            rt.despawned,
            alerted,
            bridge.stats.hits_dealt,
            bridge.stats.hits_taken,
            bridge.stats.kills,
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
    skip: &HashSet<u32>,
) {
    if let Some(c) = collision {
        for (z, p0) in rt.zombies.iter_mut().zip(prev.iter().copied()) {
            if skip.contains(&z.handle) {
                continue;
            }
            let cap = zombie_actor::capsule_for_class(&z.entity_class);
            let delta = [z.pos[0] - p0[0], z.pos[1] - p0[1], z.pos[2] - p0[2]];
            if delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] > 1e-8 {
                z.pos = zombie_actor::sweep_body(c, p0, delta, cap);
            }
            z.pos[2] = ground(c, z.pos[0], z.pos[1], z.pos[2]);
        }
    }

    // Actor-actor + actor-player separation. The player is a fixed body so the
    // player controller (main.rs) stays authoritative. Corpses are left in
    // place and excluded so they do not wall live zombies off the player.
    let mut indices: Vec<usize> = Vec::new();
    let mut bodies: Vec<Body> = Vec::new();
    for (i, z) in rt.zombies.iter().enumerate() {
        if skip.contains(&z.handle) {
            continue;
        }
        let cap = zombie_actor::capsule_for_class(&z.entity_class);
        bodies.push(Body { pos: z.pos, radius: cap.radius, height: cap.height, fixed: false });
        indices.push(i);
    }
    bodies.push(Body {
        pos: player,
        radius: sod_core::class3_physics::PLAYER_CAPSULE_RADIUS,
        height: sod_core::class3_physics::PLAYER_CAPSULE_HEIGHT,
        fixed: true,
    });
    zombie_actor::separate_bodies(&mut bodies, SEPARATION_ITERATIONS);
    for (k, &i) in indices.iter().enumerate() {
        rt.zombies[i].pos[0] = bodies[k].pos[0];
        rt.zombies[i].pos[1] = bodies[k].pos[1];
    }
    if let Some(c) = collision {
        for &i in &indices {
            let z = &mut rt.zombies[i];
            z.pos[2] = ground(c, z.pos[0], z.pos[1], z.pos[2]);
        }
    }
}

/// How long a dead actor stays as a corpse before its entity/visual is removed.
const CORPSE_TIME: f32 = 6.0;
/// Player melee reach used by the auto/player attack resolver (m). The zombie's
/// own strike radius is [`cry_combat::MELEE_DAMAGE_RADIUS`].
const PLAYER_MELEE_REACH: f32 = 2.2;
/// Maximum firearm range the resolver will target.
const PLAYER_GUN_RANGE: f32 = 45.0;
/// Range at which the perception loop lets a zombie's state machine strike:
/// `melee.damageRadius` (1.1 m) plus the actor reach band.
const ZOMBIE_STRIKE_RANGE: f32 = 2.2;
/// How long a forced `MachineDriven.Combat.*` strike state may run before the
/// machine is handed back to the graph. Covers the longest strike fragment.
const STRIKE_WINDOW: f32 = 1.6;

/// Which leaked machine a zombie class runs.
fn machine_family(class: &str) -> &'static str {
    match class {
        "Feral" => "freak_feral",
        "Juggernaut" => "freak_juggernaut",
        "Bloater" => "freak_bloater",
        "Screamer" => "freak_screamer",
        _ => "zombie",
    }
}

/// The `MachineDriven.Combat.*` attack state this class uses, in preference
/// order. Feral swipes, the Juggernaut slams, the rest use the slow or fast
/// claw strike (`fast` = the `Fast` context tag / hunt pipe scale).
fn pick_strike_state(inst: &SmInstance, family: &str, fast: bool) -> Option<u32> {
    let candidates: &[&str] = match family {
        "freak_feral" => &[
            "Combat.Basic.DoubleSwipe",
            "Combat.Basic.JumpAttack",
            "Combat.TargetDown.Attacks",
        ],
        "freak_juggernaut" => &[
            "Combat.Slam.Normal",
            "Combat.Slam.Walking",
            "Combat.Charge.FollowupSlam",
        ],
        _ if fast => &[
            "Combat.Fast.Strike.TargetStand.Left",
            "Combat.Fast.Strike.TargetStand.Right",
        ],
        _ => &[
            "Combat.Slow.Strike.TargetStand.Left",
            "Combat.Slow.Strike.TargetStand.Right",
        ],
    };
    candidates.iter().find_map(|c| inst.find_state(c))
}

/// The damage tier named by `state`'s own action block. Only the strike tiers
/// (`Zombie, Damage*` / `BiteHit`) are accepted; the shared reaction blocks'
/// `KillCredit` / stamina profiles are ignored.
fn strike_profile(sm: &StateMachine, state: u32) -> MeleeProfile {
    for a in sm.payloads.actions.iter().filter(|a| a.state == state) {
        if let Some(t) = sm_runtime::combat_damage_profile_text(a.descriptor_id, &a.code) {
            let p = MeleeProfile::parse(t);
            if matches!(
                p,
                MeleeProfile::Light | MeleeProfile::Standard | MeleeProfile::Heavy | MeleeProfile::Bite
            ) {
                return p;
            }
        }
    }
    MeleeProfile::Unknown
}

/// Heuristic on a fragment clip name: the attack fragments are `ZombieMelee_*`
/// strike/swipe/dive clips and the freak `Combat_*` / `Grab_*` clips, as opposed
/// to locomotion loops.
fn clip_looks_like_strike(clip: &str) -> bool {
    let c = clip.to_ascii_lowercase();
    [
        "strike", "swipe", "slam", "tackle", "leap", "dive", "combo", "bite", "attack", "claw",
    ]
    .iter()
    .any(|k| c.contains(k))
}

/// Resolve a state-machine fragment clip name (e.g.
/// `ZombieMelee_strike_stand_high_left`, `Combat_combo_attacks`) to the shipped
/// `.$caf`. The fragment names carry a `ZombieMelee_` / `Combat_` / `Grab_` /
/// `Basic_` prefix the file path omits, so the trailing tokens are matched
/// against the class rig (`characters/<rig>/...`).
fn load_strike_clip(vfs: &Vfs, class: &str, clip: &str) -> Option<anim::Caf> {
    let rig = zombie_actor::character_for_class(class).rig.to_ascii_lowercase();
    let base = clip.to_ascii_lowercase();
    let mut needles: Vec<String> = vec![base.clone()];
    for p in ["zombiemelee_", "fastzombie_", "slowzombie_", "combat_", "grab_", "basic_"] {
        if let Some(rest) = base.strip_prefix(p) {
            needles.push(rest.to_string());
        }
    }
    let mut paths: Vec<String> = vfs.paths_with_ext(".$caf");
    paths.sort();
    for needle in &needles {
        for p in &paths {
            let pl = p.to_ascii_lowercase();
            if pl.contains(&rig) && pl.ends_with(&format!("/{needle}.$caf")) {
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

/// Engine inputs for a zombie's ULSM. The native perception loop drives the
/// machine (it forces the attack state with `go_to`), so the engine trigger
/// inputs are inert; only the timers / phase are fed.
struct ZedSmEnv {
    dt: f32,
    time: f32,
    phase: f32,
}

impl SmEnv for ZedSmEnv {
    fn delta_seconds(&self) -> f32 {
        self.dt
    }
    fn time_seconds(&self) -> f32 {
        self.time
    }
    fn animation_phase(&self) -> f32 {
        self.phase
    }
}

/// One-shot push of the player's `g_playerHealthValue` into the script actor.
/// The zombie scripts call `BasicActor:ResetCommon` from `OnReset`, which the
/// host does not send at spawn, so the actor otherwise keeps its 100 HP default.
fn ensure_player_health(br: &mut ZombieBridge, cry: &mut CryHost) {
    if br.player_ready {
        return;
    }
    let target = cry
        .host
        .ctx
        .cvars
        .borrow()
        .get("g_playerhealthvalue")
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(cry_combat::PLAYER_MAX_HEALTH);
    let id = cry.player;
    if let Some(a) = cry.host.ctx.actors.borrow_mut().get_mut(&id) {
        a.max_health = target;
        a.health = target;
        br.player_ready = true;
        println!("[combat] player actor #{id} health {target:.0}");
    }
}

fn actor_health(cry: &CryHost, id: EntityId) -> Option<f32> {
    cry.host.ctx.actors.borrow().get(&id).map(|a| a.health)
}

fn set_actor_health(cry: &CryHost, id: EntityId, hp: f32) {
    if let Some(a) = cry.host.ctx.actors.borrow_mut().get_mut(&id) {
        a.health = hp.max(0.0);
    }
}

/// Deterministic hit-material choice: exercise headshots as well as body hits
/// without an RNG resource. The bullet/melee `material` drives the headshot
/// branch of [`cry_combat::process_actor_damage`].
fn hit_material(frame: u64, handle: u32) -> HitMaterial {
    if (frame.wrapping_add(handle as u64)) % 4 == 0 {
        HitMaterial::Head
    } else {
        HitMaterial::Torso
    }
}

/// Dispatch the death events the zombie's own script handles: `Kill` (reached
/// from `SinglePlayer:ProcessDeath` -> `target:Kill(...)`), `OnDeadHit`
/// (`BasicActor.Server:OnDeadHit`) and a `TO_IDLE` signal so the AI drops the
/// hunt state.
fn dispatch_zombie_death(cry: &CryHost, entity: EntityId, shooter: Option<EntityId>, weapon: Option<EntityId>, headshot: bool) {
    let shooter = shooter.map(|s| s as i64).unwrap_or(0);
    let weapon = weapon.map(|w| w as i64).unwrap_or(0);
    let _ = cry.host.event(entity, "OnDeadHit", ());
    let _ = cry.host.event(entity, "Kill", (true, shooter, weapon, false, headshot));
    let _ = cry.host.event(entity, "OnSignal", ("TO_IDLE".to_string(), 0i64, 0i64));
}

/// Force the player's original `.sm` into a `MachineDriven` reaction. Reuses the
/// `SOD_SMTEST` window (`enter_frame = 90`, hold 300 frames) by rewinding the
/// debug driver's frame counter, so no animation code needs changing.
fn trigger_player_reaction(sm: Option<&mut crate::PlayerSm>, name: &str) {
    let Some(sm) = sm else { return };
    if let Some(idx) = sm.inst.find_state(name) {
        sm.test = crate::SmTest::Jump(idx);
        sm.armed = false;
        sm.released = false;
        sm.frame = 89;
        sm.next_release_frame = 389;
        println!("[combat] player reaction -> [{idx}] {}", sm.inst.state_name());
    }
}

/// Horizontal ground distance.
fn horiz(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// `SOD_HITSHOT=<base>` combat screenshot path with a suffix.
fn shot_path(base: &Path, suffix: &str) -> PathBuf {
    let stem = base.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "shot".into());
    base.with_file_name(format!("{stem}.{suffix}.png"))
}

/// Apply one already-picked player hit to zombie `idx` through
/// [`cry_combat::process_actor_damage`], then dispatch death if it was lethal.
#[allow(clippy::too_many_arguments)]
fn resolve_player_hit(
    br: &mut ZombieBridge,
    cry: &mut CryHost,
    commands: &mut Commands,
    hitshot: &Option<PathBuf>,
    idx: usize,
    player: EntityId,
    hit: &cry_combat::Hit,
    base: f32,
    label: &str,
) {
    let (handle, kind, class) = {
        let z = &br.rt.as_ref().unwrap().zombies[idx];
        (z.handle, z.kind, z.entity_class.clone())
    };
    let entity = hit.target;
    let cur = actor_health(cry, entity).unwrap_or_else(|| kind.stats().hp);
    let out = cry_combat::process_actor_damage(cur, DamageTarget::Zombie(kind), hit, cry_combat::BLOATER_HEADSHOT_KILL);
    set_actor_health(cry, entity, out.health);
    br.stats.hits_dealt += 1;
    if let Some(c) = br.combat.get_mut(&handle) {
        c.reaction = 0.35;
    }
    println!(
        "[combat] player({label}) -> zombie #{handle} {class} {base:.0} dmg ({cur:.0}->{:.0} hp){}",
        out.health,
        if out.dead { " KILL" } else if out.headshot { " headshot" } else { "" },
    );
    if out.dead {
        if let Some(z) = br.rt.as_mut().unwrap().zombies.iter_mut().find(|z| z.handle == handle) {
            z.live = false;
        }
        br.combat.remove(&handle);
        br.dead.insert(handle, CORPSE_TIME);
        br.killed.insert(handle);
        dispatch_zombie_death(cry, entity, Some(player), hit.weapon, out.headshot);
        br.stats.kills += 1;
        println!("[combat] zombie #{handle} {class} died (kills={})", br.stats.kills);
    }
    // Capture the zombie's hit/death reaction once.
    if out.dealt > 0.0 && !br.shot_zombie {
        if let Some(base) = hitshot {
            let p = shot_path(base, "zombie");
            commands.spawn(Screenshot::primary_window()).observe(save_to_disk(p.clone()));
            br.shot_zombie = true;
            println!("[combat] zombie reaction screenshot {}", p.display());
        }
    }
}

/// The original combat loop for one frame:
/// 1. zombie perception acquires the player (feeding the hunt pipe's target),
/// 2. alerted zombies in melee range strike through the damage pipeline,
/// 3. the player's queued melee/fire attack resolves against the nearest live
///    zombie, and a lethal result dispatches the victim's `Kill`/`OnDeadHit`.
#[allow(clippy::too_many_arguments)]
fn combat_step(
    br: &mut ZombieBridge,
    cry: &mut CryHost,
    commands: &mut Commands,
    hitshot: &Option<PathBuf>,
    player_pos: [f32; 3],
    collision: Option<&CollisionWorld>,
    dt: f32,
    request: PlayerAttackRequest,
    mut sm: Option<&mut crate::PlayerSm>,
    mut player_res: Option<&mut Player>,
) {
    br.frame = br.frame.wrapping_add(1);
    let player = cry.player;
    let auto = std::env::var("SOD_AUTOFIGHT").is_ok();

    // In the auto-fight harness the player does not turn (no mouse input), so
    // face the nearest live zombie before resolving the melee cone. This only
    // runs under `SOD_AUTOFIGHT`; normal play keeps the camera-driven yaw.
    let mut aim_yaw = request.yaw;
    if auto {
        if let Some(p) = player_res.as_deref_mut() {
            if let Some(rt) = br.rt.as_ref() {
                let mut best: Option<(f32, f32)> = None;
                for z in rt.zombies.iter().filter(|z| z.live && z.entity.is_some()) {
                    let d = horiz(request.origin, z.pos);
                    if best.map_or(true, |(bd, _)| d < bd) {
                        best = Some((d, (z.pos[1] - request.origin[1]).atan2(z.pos[0] - request.origin[0])));
                    }
                }
                if let Some((_, y)) = best {
                    p.mstate.yaw = y;
                    aim_yaw = y;
                }
            }
        }
    }

    for c in br.combat.values_mut() {
        c.attack_cd = (c.attack_cd - dt).max(0.0);
        c.reaction = (c.reaction - dt).max(0.0);
    }
    for t in br.dead.values_mut() {
        *t -= dt;
    }

    let forward = [aim_yaw.cos(), aim_yaw.sin()];
    let gunshot = request.shot.is_some();

    // --- 3. player attack (resolve before zombie attacks) ---
    // Melee and the firearm are resolved independently (a frame can carry
    // both) against the nearest qualifying live zombie.
    let origin = request.origin;
    if let Some(dmg) = request.melee {
        let pick = {
            let rt = br.rt.as_ref().unwrap();
            let mut best: Option<(usize, f32)> = None;
            for (i, z) in rt.zombies.iter().enumerate() {
                if !z.live || br.dead.contains_key(&z.handle) || br.killed.contains(&z.handle) {
                    continue;
                }
                if cry_combat::melee_arc_test(origin, [forward[0], forward[1], 0.0], z.pos, 70.0, 0.0, PLAYER_MELEE_REACH).is_some() {
                    let d = horiz(origin, z.pos);
                    if best.map_or(true, |(_, bd)| d < bd) {
                        best = Some((i, d));
                    }
                }
            }
            best.map(|(i, _)| i)
        };
        if let Some(i) = pick {
            let (entity, zpos, handle) = {
                let z = &br.rt.as_ref().unwrap().zombies[i];
                (z.entity.unwrap_or(0), z.pos, z.handle)
            };
            let material = hit_material(br.frame, handle);
            let hit = cry_combat::Hit::melee(entity, player, dmg, material, zpos, [forward[0], forward[1], 0.0]);
            resolve_player_hit(br, cry, commands, hitshot, i, player, &hit, dmg, "melee");
        }
    }
    if let Some(dmg) = request.shot {
        let max_range = PLAYER_GUN_RANGE;
        let pick = {
            let rt = br.rt.as_ref().unwrap();
            let mut best: Option<(usize, f32)> = None;
            for (i, z) in rt.zombies.iter().enumerate() {
                if !z.live || br.dead.contains_key(&z.handle) || br.killed.contains(&z.handle) {
                    continue;
                }
                let d = horiz(origin, z.pos);
                if d <= max_range
                    && line_of_sight(collision, origin, z.pos)
                    && best.map_or(true, |(_, bd)| d < bd)
                {
                    best = Some((i, d));
                }
            }
            best.map(|(i, _)| i)
        };
        if let Some(i) = pick {
            let (entity, zpos, handle) = {
                let z = &br.rt.as_ref().unwrap().zombies[i];
                (z.entity.unwrap_or(0), z.pos, z.handle)
            };
            let material = hit_material(br.frame, handle);
            let hit = cry_combat::Hit::bullet(entity, player, 0, dmg, material, zpos, [forward[0], forward[1], 0.0]);
            resolve_player_hit(br, cry, commands, hitshot, i, player, &hit, dmg, "gun");
        }
    }

    // --- 1 + 2. zombie perception and state-machine-driven melee ---
    //
    // The native perception loop decides *that* a zombie attacks (it feeds the
    // hunt pipe's target and the `TO_FRESHMEAT` signal); the zombie's own ULSM
    // decides *when the strike lands*. Each live actor runs its class machine
    // (`libs/statemachines/<family>/general.win.sm`). When perception has the
    // player in melee range the machine is forced into the matching
    // `MachineDriven.Combat.*` attack state, which plays its `ZombieMelee_*` /
    // `Combat_*` strike fragment and fires the strike action
    // (`FxMaterialEffect`, e.g. `zed_clawattack` / `jugg_swipe_attack`). The hit
    // is emitted on that action, using the damage tier the same state names.
    let gun_sound = gunshot.then_some(SoundStimulus { pos: player_pos, radius: perception::COMBAT_SOUND_DIST });
    let mut player_hit: Option<([f32; 3], f32, ZombieKind, HitMaterial, EntityId, MeleeProfile, String)> = None;
    {
        let vfs = cry.host.ctx.vfs.clone();
        let rt = br.rt.as_mut().unwrap();
        for i in 0..rt.zombies.len() {
            let (handle, entity, class, kind, zpos, heading, walk_only) = {
                let z = &rt.zombies[i];
                (z.handle, z.entity, z.entity_class.clone(), z.kind, z.pos, z.heading, z.walk_only)
            };
            let Some(entity) = entity else { continue };
            if br.dead.contains_key(&handle) || br.killed.contains(&handle) {
                continue;
            }
            let dist = horiz(zpos, player_pos);
            let senses = Senses::for_class(&class);
            let los = |a: [f32; 3], b: [f32; 3]| line_of_sight(collision, a, b);
            let perc = perception::perceive(zpos, heading, &senses, player_pos, false, los, gun_sound);
            {
                let entry = br.combat.entry(handle).or_default();
                if perc.has_target() {
                    if !entry.alerted {
                        entry.alerted = true;
                        println!("[combat] zombie #{} {class} acquired player at {dist:.1}m", handle);
                        let _ = cry.host.event(entity, "OnSignal", ("TO_FRESHMEAT".to_string(), 0i64, 0i64));
                    }
                    rt.zombies[i].target = Some(player_pos);
                } else {
                    entry.alerted = false;
                    rt.zombies[i].target = None;
                }
                entry.attack_cd = (entry.attack_cd - dt).max(0.0);
            }
            let alerted = br.combat.get(&handle).map(|c| c.alerted).unwrap_or(false);
            let ready = br.combat.get(&handle).map(|c| c.attack_cd <= 0.0).unwrap_or(false);

            // --- the zombie's own state machine ---
            let family = machine_family(&class);
            let Some(machine) = br.sm_machines.get(family).copied() else { continue };
            // Take the instance out of the map so the other per-handle maps are
            // free to be borrowed while it ticks.
            let mut inst = br
                .sm_instances
                .remove(&handle)
                .unwrap_or_else(|| SmInstance::new(machine, handle.wrapping_mul(0x9e37_79b1) | 1));

            // Force-enter the class's melee attack state when perception, range
            // and the strike gap agree (the engine would reach it through the
            // `TO_ATTACK` signal; the native loop exercises the same state).
            let strike_active = br.sm_strikes.contains_key(&handle);
            if alerted && !strike_active && ready && dist <= ZOMBIE_STRIKE_RANGE {
                if let Some(st) = pick_strike_state(&inst, family, !walk_only) {
                    let profile = strike_profile(machine, st);
                    if std::env::var("SOD_ZEDSMDBG").is_ok() {
                        println!(
                            "[combat] zombie #{} {class} SM -> [{}] {} (profile {:?})",
                            handle,
                            st,
                            machine.states.get(st as usize).map(|s| s.name.as_str()).unwrap_or("?"),
                            profile
                        );
                    }
                    inst.go_to(st);
                    br.sm_strikes.insert(
                        handle,
                        SmStrike { state: st, left: STRIKE_WINDOW, struck: false, profile },
                    );
                }
            }

            let env = ZedSmEnv { dt, time: inst.time, phase: 0.0 };
            inst.tick(&env);
            let mut struck = false;
            for cmd in inst.take_commands() {
                match cmd {
                    SmCommand::PlayFragment { clip, .. } => {
                        // Play the attack state's strike fragment on the zombie.
                        if br.sm_strikes.contains_key(&handle) && clip_looks_like_strike(&clip) {
                            if let Some(caf) = load_strike_clip(&vfs, &class, &clip) {
                                if std::env::var("SOD_ZEDSMDBG").is_ok() {
                                    println!(
                                        "[combat] zombie #{} {class} strike fragment {clip} ({:.2}s) -> visual",
                                        handle,
                                        caf.duration_secs()
                                    );
                                }
                                br.pending_strikes.insert(handle, caf);
                            }
                        }
                    }
                    SmCommand::Action(a) => {
                        if sm_runtime::fx_material_effect_name(&a).is_some() {
                            struck = true;
                        }
                    }
                    _ => {}
                }
            }
            if struck {
                if let Some(s) = br.sm_strikes.get_mut(&handle) {
                    s.struck = true;
                }
            }

            // End the strike: emit the hit once, then hand the machine back to
            // the graph (locomotion) and start the strike gap.
            let finish = match br.sm_strikes.get_mut(&handle) {
                Some(s) => {
                    s.left -= dt;
                    s.struck || s.left <= 0.0
                }
                None => false,
            };
            if finish {
                if let Some(s) = br.sm_strikes.remove(&handle) {
                    inst.go_to(machine.start_state);
                    if let Some(c) = br.combat.get_mut(&handle) {
                        c.attack_cd = cry_combat::melee_recovery(s.profile);
                    }
                    if s.struck {
                        let material = hit_material(br.frame, handle);
                        let state_name = machine
                            .states
                            .get(s.state as usize)
                            .map(|x| x.name.clone())
                            .unwrap_or_default();
                        player_hit = Some((
                            zpos,
                            cry_combat::melee_damage(kind, s.profile),
                            kind,
                            material,
                            entity,
                            s.profile,
                            state_name,
                        ));
                    }
                }
            }
            br.sm_instances.insert(handle, inst);
        }
    }

    if let Some((zpos, dmg, kind, material, attacker, profile, state_name)) = player_hit {
        let dir = [player_pos[0] - zpos[0], player_pos[1] - zpos[1], 0.0];
        let hit = cry_combat::Hit::melee(player, attacker, dmg, material, zpos, dir);
        let cur = actor_health(cry, player).unwrap_or(cry_combat::PLAYER_MAX_HEALTH);
        let out = cry_combat::process_actor_damage(cur, DamageTarget::Player, &hit, true);
        set_actor_health(cry, player, out.health);
        br.stats.hits_taken += 1;
        println!(
            "[combat] zombie {} {:?} strike [{}] -> player {:.0} dmg ({:.0}->{:.0} hp){}",
            kind.name(),
            profile,
            state_name,
            out.dealt,
            cur,
            out.health,
            if out.dead { " DOWNED" } else if out.headshot { " headshot" } else { "" },
        );
        trigger_player_reaction(sm.as_deref_mut(), "Reaction.Core.Heavy.High.Front");
        if !br.shot_player {
            if let Some(base) = hitshot {
                let p = shot_path(base, "player");
                commands.spawn(Screenshot::primary_window()).observe(save_to_disk(p.clone()));
                br.shot_player = true;
                println!("[combat] player reaction screenshot {}", p.display());
            }
        }
        if out.dead {
            // The player's script `Kill` path would end the run; the demo keeps
            // playing by reviving the actor to its pool after the downed event.
            let maxh = cry
                .host
                .ctx
                .actors
                .borrow()
                .get(&player)
                .map(|a| a.max_health)
                .unwrap_or(cry_combat::PLAYER_MAX_HEALTH);
            set_actor_health(cry, player, maxh);
        }
    }
}

/// Simple eye-to-eye line of sight through the static collision world (1.5 m
/// above each foot, the same eye height the runtime visibility check uses).
fn line_of_sight(collision: Option<&CollisionWorld>, from: [f32; 3], to: [f32; 3]) -> bool {
    let Some(c) = collision else { return true };
    let a = [from[0], from[1], from[2] + 1.5];
    let b = [to[0], to[1], to[2] + 1.5];
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if len < 1e-3 {
        return true;
    }
    match c.raycast(a, d, len - 0.2) {
        Some(h) => h.t >= len - 0.2,
        None => true,
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
    let mut live: HashSet<u32> = br
        .rt
        .as_ref()
        .unwrap()
        .zombies
        .iter()
        .filter(|z| z.live)
        .map(|z| z.handle)
        .collect();
    // Dead actors stay as corpses (and keep their entity/visual) for the
    // corpse timer; see `combat_step` / `zombie_bridge_tick`.
    live.extend(br.dead.keys().copied());

    let stale: Vec<u32> =
        br.host_entities.keys().copied().filter(|h| !live.contains(h)).collect();
    for h in stale {
        if let Some(id) = br.host_entities.remove(&h) {
            cry.host.ctx.world.borrow_mut().remove(id);
            br.last_pipe.remove(&id);
            br.last_result.remove(&id);
            // Clear the record's entity so a later re-promotion spawns a fresh
            // actor instead of pointing at the removed one.
            if let Some(z) = br.rt.as_mut().unwrap().zombies.iter_mut().find(|z| z.handle == h) {
                z.entity = None;
            }
        }
        if let Some(e) = br.visuals.remove(&h) {
            commands.entity(e).despawn();
        }
        if br.verbose < VERBOSE_SPAWN_LOG {
            br.verbose += 1;
            println!("[zombie] despawn #{}", h);
        }
    }

    let killed: HashSet<u32> = br.killed.clone();
    let rt = br.rt.as_mut().unwrap();
    for z in rt.zombies.iter_mut() {
        if killed.contains(&z.handle) {
            // Permanently dead: the director does not model kills, so keep the
            // slot demoted instead of letting it respawn.
            z.live = false;
            continue;
        }
        if !z.live || z.entity.is_some() {
            continue;
        }
        let name = format!("ULZombie{}", z.handle);
        match cry.host.spawn_actor(&z.entity_class, &z.script, &name, z.pos, false) {
            Ok(id) => {
                z.entity = Some(id);
                br.host_entities.insert(z.handle, id);
                // The host does not send `OnReset` at spawn, so `ResetCommon`
                // never runs and the actor keeps its 100 HP default. Seed the
                // per-class `Damage.health` from the recovered stats
                // (`ai-combat.md` §1.1) so the pipeline has the right pool.
                if let Some(a) = cry.host.ctx.actors.borrow_mut().get_mut(&id) {
                    let hp = z.kind.stats().hp;
                    a.max_health = hp;
                    a.health = hp;
                }
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
        let (parts, skeleton, idle, walk, death, react) = (
            build_parts(t, &mut meshes),
            t.skeleton.clone(),
            t.idle.clone(),
            t.walk.clone(),
            t.death.clone(),
            t.react.clone(),
        );
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
                    death,
                    react,
                    strike: None,
                    time: 0.0,
                    moving: false,
                    dying: false,
                    reacting: false,
                    striking: false,
                    last_react: 0.0,
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

    let mut live: HashSet<u32> =
        bridge.rt.as_ref().unwrap().zombies.iter().filter(|z| z.live).map(|z| z.handle).collect();
    // Keep corpses' visuals alive while their entity is held for the death clip.
    live.extend(bridge.dead.keys().copied());
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

/// Position and skin every zombie visual; clip is idle/walk, a one-shot death
/// clip on a corpse, and a brief knock-back lean on a hit reaction.
fn animate_zombie_visuals(
    time: Res<Time>,
    mut bridge: ResMut<ZombieBridge>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut q: Query<(&mut Transform, &mut ZombieVisual)>,
) {
    let base = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    // Drain this frame's state-machine strike requests before taking the
    // read-only borrows used by the loop below.
    let mut pending: HashMap<u32, anim::Caf> = std::mem::take(&mut bridge.pending_strikes);
    let rt = bridge.rt.as_ref();
    for (mut tf, mut v) in &mut q {
        let mut yaw = 0.0f32;
        if let Some(rt) = rt {
            if let Some(z) = rt.zombies.iter().find(|z| z.handle == v.handle) {
                tf.translation = base * Vec3::new(z.pos[0], z.pos[1], z.pos[2]);
                yaw = sod_core::class3_move::render_yaw(z.heading);
                v.moving = z.moving;
            }
        }
        // Corpses: switch to the one-shot death clip once.
        let dead = bridge.dead.contains_key(&v.handle);
        if dead && !v.dying {
            v.dying = true;
            v.time = 0.0;
        }
        // Victim reaction: a short back-lean and, when a reaction clip exists,
        // a one-shot flinch. A fresh hit restarts the clip.
        let reaction = bridge.combat.get(&v.handle).map(|c| c.reaction).unwrap_or(0.0);
        if reaction > 0.0 {
            if !v.reacting || reaction > v.last_react {
                v.reacting = true;
                v.time = 0.0;
            }
        } else {
            v.reacting = false;
        }
        v.last_react = reaction;
        // A strike fragment requested by the zombie's own state machine: play
        // it as a one-shot over the locomotion graph.
        if let Some(caf) = pending.remove(&v.handle) {
            v.strike = Some(caf);
            v.striking = true;
            v.time = 0.0;
        }
        let lean = (reaction / 0.35).clamp(0.0, 1.0) * 0.5;
        tf.rotation = base * Quat::from_rotation_z(yaw) * Quat::from_rotation_x(-lean);

        if v.parts.is_empty() {
            continue;
        }
        v.time += time.delta_secs();
        // A finished one-shot strike releases the machine-driven layer.
        if v.striking {
            let done = v
                .strike
                .as_ref()
                .map(|s| v.time >= s.duration_secs().max(1.0 / 30.0))
                .unwrap_or(true);
            if done {
                v.striking = false;
            }
        }
        let (caf, one_shot) = if v.dying {
            match &v.death {
                Some(d) => (d, true),
                None => (&v.idle, false),
            }
        } else if v.reacting {
            match &v.react {
                Some(r) => (r, true),
                None => (&v.idle, false),
            }
        } else if v.striking {
            match &v.strike {
                Some(s) => (s, true),
                None => (&v.idle, false),
            }
        } else if v.moving {
            (&v.walk, false)
        } else {
            (&v.idle, false)
        };
        let dur = caf.duration_secs().max(1.0 / 30.0);
        let t = if one_shot { v.time.min(dur) } else { v.time % dur };
        let tick = caf.tick_from_seconds(t);
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
    let kills = bridge.stats.kills;
    for (mut t, line) in &mut lines {
        if matches!(line, crate::HudLine::Status) {
            t.0 = format!("HP {hp:.0}/{maxhp:.0}   Stamina {stam:.0}   Zombies {n}   Kills {kills}");
        }
    }
}
