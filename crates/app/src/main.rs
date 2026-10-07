//! sod-native: native (Wine-free) Linux front-end for State of Decay: YOSE.
//!
//! Modes:
//!   `sod-native <game> world`            overview of the Class3 level
//!   `sod-native <game> play`             third-person, controllable survivor
//!   `sod-native <game> <model.cgf|.chr>` single asset

mod cry_bridge;
mod ai_bridge;
use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::image::{CompressedImageFormats, ImageSampler, ImageType};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::material::AlphaMode;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use sod_core::crypak::Vfs;
use sod_core::{ai, architecture, cache, camera, character, combat, hud, inventory, locomotion, mission, npc_ai, player_locomotion, surfaces, tagpoints, tuning, vegetation, zombie_ai};
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::ui::Display;

fn wrap_angle(a: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut r = a % tau;
    if r > std::f32::consts::PI { r -= tau; }
    if r < -std::f32::consts::PI { r += tau; }
    r
}
use sod_core::level::Placement;
use sod_core::material::{self, MaterialSlot};
use sod_core::mesh::{self, DecodedMesh, DecodedModel};
use sod_core::skin::{self, BoneWeight, Skeleton};
use sod_core::terrain::{self, Terrain};
use sod_core::anim;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const DEFAULT_GAME: &str =
    "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";
const DEFAULT_PLAYER: &str = "characters/male/human/body/tshirt_jeans/variant/v01.cdf";

#[derive(Resource)]
struct Scene {
    model: DecodedModel,
    palette: Vec<MaterialSlot>,
    tex_bytes: HashMap<String, Vec<u8>>,
}

#[derive(Clone)]
struct AnimPart {
    handle: Handle<Mesh>,
    bind_pos: Vec<[f32; 3]>,
    bind_nrm: Vec<[f32; 3]>,
    infl: Vec<[BoneWeight; 4]>,
    mat: Handle<StandardMaterial>,
}

#[derive(Resource)]
struct Anim {
    parts: Vec<AnimPart>,
    skeleton: Skeleton,
    caf: anim::Caf,
    time: f32,
}

#[derive(Resource)]
struct World {
    terrain: Terrain,
    terrain_verts: Vec<[f32; 3]>,
    terrain_colors: Vec<[f32; 4]>,
    terrain_indices: Vec<u32>,
    placements: Vec<Placement>,
    models: HashMap<String, Vec<DecodedMesh>>,
    slots: HashMap<String, Vec<MaterialSlot>>,
    tex_bytes: HashMap<String, Vec<u8>>,
    /// Static architecture (buildings, walls, rocks) around the tutorial zone.
    brushes: Vec<(String, [f32; 3], [f32; 4], f32)>,
    /// Vegetation (trees/bushes): model, cry position, scale, yaw radians.
    vegetation: Vec<(String, [f32; 3], f32, f32)>,
    /// Raw `.dds` bytes for the tiled ground diffuse texture (fallback).
    ground_tex: Option<Vec<u8>>,
    /// Baked per-surface ground albedo (width, height, RGBA8), spanning the terrain.
    ground_albedo: Option<(u32, u32, Vec<u8>)>,
}

/// Level lighting from the mission's time of day, at its start time.
/// Only the sun DIRECTION is used (verified against the level's authored
/// SunVector). TOD colours are HDR tints that the original runs through its
/// multipliers, exposure, tone mapping and colour grading; applying the raw
/// hue without that pipeline is less faithful than neutral light, so colours
/// stay neutral until the HDR pipeline is ported.
#[derive(Resource)]
struct LevelLighting {
    /// Direction the sun's light travels, in Bevy space.
    sun_dir: Vec3,
}

/// The player's AnimationGraph (playerfullbody.xml) driven from movement.
/// Clips are not taken from it yet: graph animation names resolve to clip
/// files through the template/alias chain still being recovered.
#[derive(Resource)]
struct PlayerGraph(sod_core::ag_driver::CharacterGraph);

/// Static collision world (CryPhysics replacement): terrain plus the physics
/// proxies of every placed brush and script-loaded prop.
#[derive(Resource)]
struct Collision(sod_core::cry::phys_world::CollisionWorld);

#[derive(Resource)]
struct PlayerSetup {
    model: DecodedModel,
    slots: Vec<MaterialSlot>,
    tex_bytes: HashMap<String, Vec<u8>>,
    skeleton: Skeleton,
    influences: Vec<Vec<[BoneWeight; 4]>>,
    caf: anim::Caf,
    spawn: [f32; 3],
    yaw: f32,
    bank: character::ClipBank,
}

#[derive(Resource)]
struct Player {
    yaw: f32,
    pos: [f32; 2], // cry X/Y
    parts: Vec<AnimPart>,
    skeleton: Skeleton,
    caf: anim::Caf,
    time: f32,
    moving: bool,
    rate: f32,
    bank: character::ClipBank,
    loc: anim::Locomotion,
    /// 1:1 Class3 movement controller state.
    mstate: sod_core::class3_move::MoveState,
    /// Decompiled locomotion state machine (idle/walk/run/sprint transitions).
    ploco: player_locomotion::PlayerLocomotion,
}

#[derive(Resource)]
struct Orbit {
    center: Vec3,
    radius: f32,
    height: f32,
    angle: f32,
}

#[derive(Component)]
struct Spinner;
#[derive(Component)]
struct MainCamera;
#[derive(Component)]
struct PlayerRoot;
#[derive(Resource)]
struct NpcTemplate {
    model: DecodedModel,
    skeleton: Skeleton,
    influences: Vec<Vec<[BoneWeight; 4]>>,
    caf: anim::Caf,
    slots: Vec<MaterialSlot>,
    tex_bytes: HashMap<String, Vec<u8>>,
    bank: character::ClipBank,
}
#[derive(Component)]
struct Npc {
    kind: sod_core::game::ZombieKind,
    hp: f32,
    time: f32,
    parts: Vec<AnimPart>,
    skeleton: Skeleton,
    caf: anim::Caf,
    agent: zombie_ai::ZombieAgent,
    bank: character::ClipBank,
    loc: anim::Locomotion,
}

/// Survivor follower (npc_ai).
#[derive(Component)]
struct Follower {
    agent: npc_ai::NpcAgent,
    time: f32,
    moving: bool,
    parts: Vec<AnimPart>,
    skeleton: Skeleton,
    caf: anim::Caf,
    bank: character::ClipBank,
    loc: anim::Locomotion,
}

#[derive(Resource)]
struct FollowerTemplate {
    model: DecodedModel,
    slots: Vec<MaterialSlot>,
    tex_bytes: HashMap<String, Vec<u8>>,
    skeleton: Skeleton,
    influences: Vec<Vec<[BoneWeight; 4]>>,
    caf: anim::Caf,
    bank: character::ClipBank,
}

/// Player simulation: combat vitals, weapons and the carried inventory.
#[derive(Resource)]
struct PlayerState {
    vitals: combat::Vitals,
    melee: combat::MeleeWeapon,
    gun: combat::Firearm,
    gun_state: combat::GunState,
    equipment: inventory::Equipment,
    inv: inventory::Inventory,
    db: inventory::ItemDb,
}

/// An in-progress container search.
struct SearchTask {
    state: inventory::SearchState,
}

/// Mission/tutorial + world-simulation state.
#[derive(Resource)]
struct GameState {
    tutorial: mission::Tutorial,
    rng: inventory::XorShiftRng,
    ai_rng: ai::SmallRng,
    horde: ai::HordeDirector,
    loot: inventory::LootTable,
    search: Option<SearchTask>,
    kills: u32,
    move_notified: bool,
    auto_cd: f32,
    auto_fire_cd: f32,
    auto_search_cd: f32,
    /// Tanner-arc mission labels loaded from `missions.xml` (story line for the HUD).
    tanner: Vec<String>,
}

/// 1:1 Class3 camera state (mouse-look, aim, boom).
#[derive(Resource)]
struct CameraStateRes(sod_core::camera::CameraState);

/// Camera lag/smoothing (CryEngine `FUN_015ea260`).
#[derive(Resource)]
struct CamLagRes(sod_core::camera::CameraLag);

#[derive(Component)]
struct HealthBarFill;
#[derive(Component)]
struct StaminaBarFill;
/// Marks HUD nodes that are hidden while the main menu is up.
#[derive(Component)]
struct HudHide;
#[derive(Component)]
enum HudLine {
    Status,
    Objective,
    Inventory,
    Banner,
    Prompt,
    Notifications,
}

/// Native HUD state (Class3HUD model).
#[derive(Resource)]
struct HudRes(hud::HudState);

#[derive(Component)]
struct RadarBlipNode {
    index: usize,
}
#[derive(Component)]
struct MenuUi;
#[derive(Resource)]
struct MenuRes {
    active: bool,
    index: usize,
}

#[derive(Component)]
struct MenuItem {
    index: usize,
}

/// Marks the single top-level menu root; despawning it removes the whole menu
/// (children are despawned recursively, so they must not be despawned again).
#[derive(Component)]
struct MenuRoot;

/// The game's own UI font (`gamedata/fonts/veramono.ttf`), read from the paks.
#[derive(Resource)]
struct MenuFont(pub Vec<u8>);

/// On-disk decoded-texture cache (avoids re-decoding hundreds of DDS per launch).
#[derive(Resource)]
struct TextureCacheRes(cache::TextureCache);

#[derive(Resource)]
struct ShotPlan {
    path: Option<PathBuf>,
    frame: u32,
    keys: Vec<(u32, KeyCode)>,
}

fn play_shot(path: &Path) -> PathBuf {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "shot".into());
    path.with_file_name(format!("{stem}.play.png"))
}

fn late_shot(path: &Path) -> PathBuf {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "shot".into());
    path.with_file_name(format!("{stem}.late.png"))
}

fn auto_screenshot(
    mut commands: Commands,
    mut plan: ResMut<ShotPlan>,
    menu: Option<ResMut<MenuRes>>,
    ui: Query<Entity, With<MenuRoot>>,
) {
    let Some(path) = plan.path.clone() else { return };
    plan.frame += 1;
    let play = play_shot(&path);
    let long = std::env::var("SOD_LONG").is_ok();
    match plan.frame {
        30 => {
            commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path.clone()));
        }
        100 => {
            commands.spawn(Screenshot::primary_window()).observe(save_to_disk(play.clone()));
        }
        60 => {
            // Auto-start (simulate pressing ENTER on the menu).
            if std::env::var("SOD_AUTOSTART").is_ok() {
                if let Some(mut m) = menu {
                    if m.active {
                        for e in &ui {
                            commands.entity(e).despawn();
                        }
                        m.active = false;
                        println!("[menu] auto-start");
                    }
                }
            }
        }
        200 => {
            commands.spawn(Screenshot::primary_window()).observe(save_to_disk(play.clone()));
        }
        700 => {
            if long {
                let late = late_shot(&path);
                commands.spawn(Screenshot::primary_window()).observe(save_to_disk(late));
            }
        }
        _ => {}
    }
    let done_at = if long { 730 } else { 115 };
    if plan.frame >= done_at && play.exists() {
        println!("[shot] saved {} and {}", path.display(), play.display());
        std::process::exit(0);
    }
    if plan.frame > 2000 {
        std::process::exit(1);
    }
}

fn collect_paks(root: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_paks(&p, out);
            } else if p.extension().map(|x| x == "pak").unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let game = args.next().unwrap_or_else(|| DEFAULT_GAME.to_string());
    let target = args.next().unwrap_or_else(|| "play".to_string());

    let mut paks = Vec::new();
    collect_paks(Path::new(&game), &mut paks);
    paks.sort();
    let mut vfs = Vfs::new();
    for p in &paks {
        if let Ok(n) = vfs.mount(p) {
            println!("[app] mounted {:>6} entries  {}", n, p.file_name().unwrap().to_string_lossy());
        }
    }

    match target.as_str() {
        "world" => run_world(&game, vfs, false),
        "play" => run_world(&game, vfs, true),
        other => run_model(vfs, other.to_string()),
    }
}

type LoadedCharacter = (
    DecodedModel,
    Vec<MaterialSlot>,
    HashMap<String, Vec<u8>>,
    Skeleton,
    Vec<Vec<[BoneWeight; 4]>>,
    anim::Caf,
    character::ClipBank,
);

fn load_character(vfs: &Vfs, path: &str, rig: Option<&str>) -> Option<LoadedCharacter> {
    // Full character loader (mesh + palette + skeleton + locomotion clip bank).
    let (ch, bank) = character::load_full(vfs, path, rig, 3000)?;
    let mut tex_bytes = HashMap::new();
    for slot in &ch.slots {
        for (_, p) in &slot.textures {
            if !tex_bytes.contains_key(p) {
                if let Some(b) = vfs.read(p) {
                    tex_bytes.insert(p.clone(), b);
                }
            }
        }
    }
    let caf = match std::env::var("SOD_ANIM") {
        Ok(p) => {
            let b = vfs.read(&p)?;
            println!("[anim] forced {p}");
            anim::Caf::parse(&b)?
        }
        Err(_) => bank
            .caf_for(anim::Locomotion::Idle)
            .or_else(|| bank.caf_for(anim::Locomotion::Walk))
            .cloned()?,
    };
    println!("[anim] {} clips (idle/walk/run from bank: {})", bank.len(), bank.len() > 0);
    let influences = ch.influence_groups();
    Some((ch.model, ch.slots, tex_bytes, ch.skeleton, influences, caf, bank))
}

/// Load a character from a `.cdf` (skeleton + body/skin/legs/head/hair parts),
/// merging all parts into one skeleton-driven model. Attachments are baked in
/// their supplied bind transform (cry coords).
fn load_cdf_character(vfs: &Vfs, cdf_path: &str) -> Option<LoadedCharacter> {
    let (skeleton, parts) = match sod_core::cdf::load_parts(vfs, cdf_path) {
        Some(v) => v,
        None => {
            eprintln!("[cdf] load_parts failed for {cdf_path}");
            return None;
        }
    };
    println!("[cdf] {cdf_path}: {} attachment part(s)", parts.len());
    for p in &parts {
        println!("[cdf]   part {} ({} meshes, {} slots)", p.path, p.model.meshes.len(), p.slots.len());
    }
    let mut merged = DecodedModel { meshes: Vec::new(), bounds_min: [0.0; 3], bounds_max: [0.0; 3], material_name: None };
    let mut merged_infl: Vec<Vec<[BoneWeight; 4]>> = Vec::new();
    let mut slots: Vec<MaterialSlot> = Vec::new();
    let mut tex_bytes: HashMap<String, Vec<u8>> = HashMap::new();
    for part in &parts {
        // The base `skeleton/skeleton.chr` is a rig proxy, not body geometry.
        if part.path.to_ascii_lowercase().contains("skeleton/skeleton") {
            continue;
        }
        // CDF rotations are scalar-first `(w,x,y,z)` (e.g. "1,0,0,0" = identity).
        let rot = Quat::from_xyzw(part.rotation[1], part.rotation[2], part.rotation[3], part.rotation[0]);
        let tr = Vec3::new(part.position[0], part.position[1], part.position[2]);
        let infl = vfs
            .read(&part.path)
            .and_then(|b| skin::decode_influences(&b))
            .unwrap_or_default();
        let base = slots.len() as u32;
        for (mi, m) in part.model.meshes.iter().enumerate() {
            let mut m2 = m.clone();
            for p in &mut m2.positions {
                *p = (rot * Vec3::from_array(*p) + tr).to_array();
            }
            for n in &mut m2.normals {
                *n = (rot * Vec3::from_array(*n)).normalize_or_zero().to_array();
            }
            for sm in &mut m2.submeshes {
                sm.material = sm.material.saturating_add(base);
            }
            merged.meshes.push(m2);
            merged_infl.push(infl.get(mi).cloned().unwrap_or_default());
        }
        slots.extend(part.slots.iter().cloned());
        for slot in &part.slots {
            for (_, tp) in &slot.textures {
                if !tex_bytes.contains_key(tp) {
                    if let Some(b) = vfs.read(tp) {
                        tex_bytes.insert(tp.clone(), b);
                    }
                }
            }
        }
    }
    if merged.meshes.is_empty() {
        eprintln!("[cdf] merged 0 meshes for {cdf_path}");
        return None;
    }
    {
        let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for m in &merged.meshes {
            for p in &m.positions {
                for i in 0..3 {
                    mn[i] = mn[i].min(p[i]);
                    mx[i] = mx[i].max(p[i]);
                }
            }
        }
        println!("[cdf] merged bounds {mn:?}..{mx:?}");
    }
    // Build the clip bank against the base skeleton `.chr` (the CDF path itself
    // is not a loadable character to `ClipBank::build`).
    let base_chr = parts.first().map(|p| p.path.clone()).unwrap_or_default();
    let bank = character::ClipBank::build(vfs, &base_chr, None, 3000);
    let caf = match bank
        .caf_for(anim::Locomotion::Idle)
        .or_else(|| bank.caf_for(anim::Locomotion::Walk))
        .cloned()
    {
        Some(c) => c,
        None => {
            eprintln!("[cdf] no clip bank for {base_chr}");
            return None;
        }
    };
    println!("[anim] CDF {} -> {} meshes, {} clip(s)", cdf_path, merged.meshes.len(), bank.len());
    for loc in [anim::Locomotion::Idle, anim::Locomotion::Walk, anim::Locomotion::Run, anim::Locomotion::Sprint] {
        println!("[anim]   {:?} -> {:?}", loc, bank.path(loc));
    }
    Some((merged, slots, tex_bytes, skeleton, merged_infl, caf, bank))
}

/// Load either a `.cdf` (assembled) or a single `.chr`/`.cgf`.
fn load_any_character(vfs: &Vfs, path: &str, rig: Option<&str>) -> Option<LoadedCharacter> {
    if path.to_ascii_lowercase().ends_with(".cdf") {
        load_cdf_character(vfs, path)
    } else {
        load_character(vfs, path, rig)
    }
}

fn height_at(t: &Terrain, x: f32, y: f32) -> f32 {
    t.height_interp(x, y)
}

fn run_world(game: &str, vfs: Vfs, play: bool) {
    let level_pak_path = Path::new(game).join("Game/levels/Class3/level.pak");
    let level_pak = sod_core::Pak::open(&level_pak_path).expect("open level.pak");
    println!("[world] level.pak: {} entries", level_pak.len());
    let read = |name: &str| level_pak.find(name).and_then(|i| level_pak.read_idx(i).ok());

    let terrain_bytes = read("terrain/terrain.dat").expect("terrain.dat");
    let ter = terrain::decode_terrain(&terrain_bytes).expect("terrain");
    println!("[world] terrain {}x{} @ {:.1}m  height {:.1}..{:.1} m", ter.samples, ter.samples, ter.spacing, ter.min, ter.max);
    // Terrain surface palette (leveldata.xml) -> per-vertex colour by surface type.
    let surface_types = read("leveldata.xml").map(|b| surfaces::parse_surface_types(&b)).unwrap_or_default();
    println!("[world] surface types: {}", surface_types.len());
    // The feet stand on the interpolated 4 m heightmap; the drawn ground uses
    // an 8 m grid by default (full res cost ~40 ms on p90 frames in debug).
    // Proper fix: near-player terrain LOD. SOD_TERRAIN_STRIDE overrides.
    let terrain_stride = std::env::var("SOD_TERRAIN_STRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(2usize);
    let (tverts, tindex) = terrain::build_mesh(&ter, terrain_stride);
    // Tint the (textured) ground by surface type, mixed toward white so the
    // tiled diffuse detail stays visible.
    let tcolors: Vec<[f32; 4]> = tverts.iter().map(|p| {
        let c = surfaces::color_for(&surface_types, ter.surface_at(p[0], p[1]));
        [0.55 + 0.45 * c[0], 0.55 + 0.45 * c[1], 0.55 + 0.45 * c[2], 1.0]
    }).collect();
    let ground_tex = vfs.read("textures/ground/grass_dirt_mix01_diffuse.dds");
    // Bake a per-surface ground albedo from the real ground textures.
    let ground_albedo = surfaces::build_ground_albedo(&vfs, &ter, &surface_types, 1024);
    if let Some((w, _h, _)) = &ground_albedo {
        println!("[world] ground albedo {w}px");
    }
    println!("[world] terrain mesh: {} verts", tverts.len());

    // --- locate the Mt. Tanner tutorial zone -------------------------------
    let tag_points = read("tag_points_win.markup").and_then(|b| tagpoints::TagPoints::parse(&b));
    let placements_all: Vec<Placement> = read("mission_mission0.xml")
        .map(|b| sod_core::level::parse_mission(&b))
        .unwrap_or_default();
    let total = placements_all.len();
    let start_cry = tag_points
        .as_ref()
        .and_then(|tp| tp.get("RTSPoint_PlayerStart"))
        .or_else(|| tag_points.as_ref().and_then(|tp| tp.get("RTSPoint_RangerStation")))
        .or_else(|| sod_core::level::named_centroid(&placements_all, &["Home_RangerStation", "TannerTent"]))
        .unwrap_or_else(|| {
            let (mut ax, mut ay, mut an) = (0f32, 0f32, 0f32);
            for p in &placements_all { if p.pos != [0.0; 3] { ax += p.pos[0]; ay += p.pos[1]; an += 1.0; } }
            if an > 0.0 { [ax / an, ay / an, 0.0] } else { [0.0, 0.0, 0.0] }
        });
    println!("[world] placements: {} total, {} tag points; Mt. Tanner start {:?}",
        total, tag_points.as_ref().map(|t| t.len()).unwrap_or(0), start_cry);

    // Boot the original scripts on the native entity world. What gets drawn
    // for entities is what their scripts loaded (`LoadObject`/`DrawSlot`/
    // `Hide`), as in the original renderer; `SOD_CRY=0` falls back to the
    // level XML's `object_Model` attributes.
    let cry_host = if play && cry_bridge::enabled() { cry_bridge::build_host(game, start_cry) } else { None };
    let source: Vec<Placement> = match &cry_host {
        Some(h) => {
            let mut v = sod_core::cry::render::static_placements(&h.host.ctx.world.borrow());
            // Non-entity objects (decals, etc.) still come from the level file.
            v.extend(placements_all.iter().filter(|p| !p.is_entity).cloned());
            v
        }
        None => placements_all.clone(),
    };

    // Keep only props in the tutorial zone (covers the Ranger Station at ~166 m).
    let zone_radius = 400.0f32;
    let xml_in_zone = placements_all
        .iter()
        .filter(|p| p.model.is_some())
        .filter(|p| (p.pos[0] - start_cry[0]).powi(2) + (p.pos[1] - start_cry[1]).powi(2) <= zone_radius * zone_radius)
        .count();
    let with_models: Vec<Placement> = source
        .iter()
        .filter(|p| p.model.is_some())
        .filter(|p| {
            let dx = p.pos[0] - start_cry[0];
            let dy = p.pos[1] - start_cry[1];
            dx * dx + dy * dy <= zone_radius * zone_radius
        })
        .cloned()
        .collect();
    println!("[world] {} props within {:.0}m of Mt. Tanner (script-driven: {}; level XML would give {})",
        with_models.len(), zone_radius, cry_host.is_some(), xml_in_zone);

    // Static architecture (buildings/walls/rocks) around the zone.
    let arch_radius = 360.0f32;
    let mut brushes: Vec<(String, [f32; 3], [f32; 4], f32)> = Vec::new();
    for b in architecture::brushes_in_region(&terrain_bytes, start_cry, arch_radius) {
        brushes.push((b.model, b.pos, b.rot, b.scale));
    }
    let mut unique_brush_models: Vec<String> = brushes.iter().map(|b| b.0.clone()).collect();
    unique_brush_models.sort();
    unique_brush_models.dedup();
    unique_brush_models.truncate(900);
    brushes.retain(|b| unique_brush_models.binary_search(&b.0).is_ok());
    brushes.truncate(24000);
    println!("[world] architecture: {} brush instances / {} models within {:.0}m",
        brushes.len(), unique_brush_models.len(), arch_radius);

    // Lighting from the level's own time of day (sun from the engine's TOD
    // formula, colours from the TOD splines) at its authored start time.
    let level_lighting = read("mission_mission0.xml").map(|b| {
        let ml = sod_core::tod::parse_mission_lighting(&b);
        let hour = ml.time_of_day.time;
        let tod = ml.time_of_day.sample(hour);
        let d = ml.environment.sun_direction(hour);
        let base = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        let sun_dir = (base * Vec3::new(d[0], d[1], d[2])).normalize_or_zero();
        println!("[world] time of day {hour:.2} h: sun dir (cry) {d:?}, sun {:?}, sky {:?}", tod.sun_color(), tod.sky_color());
        LevelLighting { sun_dir }
    });

    let player_graph = vfs
        .read("animations/graphs/playerfullbody.xml")
        .and_then(|b| sod_core::ag_runtime::Graph::parse(&b))
        .map(|g| PlayerGraph(sod_core::ag_driver::CharacterGraph::new(g)));
    println!("[world] player animation graph: {}", player_graph.as_ref().map(|g| g.0.graph.states.len()).unwrap_or(0));

    // Collision from the same geometry the renderer draws.
    let collision = if std::env::var("SOD_NOCOLLIDE").is_ok() {
        None
    } else {
        use sod_core::cry::phys_world::{CollisionWorld, InstanceSpec};
        let t0 = std::time::Instant::now();
        let mut specs: Vec<InstanceSpec> = brushes
            .iter()
            .map(|(m, p, r, sc)| InstanceSpec { model: m.clone(), pos: *p, rot: *r, scale: [*sc; 3] })
            .collect();
        specs.extend(with_models.iter().filter_map(|p| {
            p.model.as_ref().map(|m| InstanceSpec { model: m.clone(), pos: p.pos, rot: p.rot, scale: p.scale })
        }));
        let cw = CollisionWorld::build(Some(ter.clone()), &vfs, &specs);
        println!("[world] collision: {} instances in {:.0} ms", specs.len(), t0.elapsed().as_secs_f32() * 1000.0);
        Some(Collision(cw))
    };

    // Trees / bushes (vegetation) from the same octree.
    let veg_radius = 360.0f32;
    let mut vegetation: Vec<(String, [f32; 3], f32, f32)> = Vec::new();
    if std::env::var("SOD_NOVEG").is_err() {
        for v in vegetation::vegetation_in_region(&terrain_bytes, start_cry, veg_radius) {
            vegetation.push((v.model, v.pos, v.scale, v.rot_y));
        }
    }
    // Keep trees/shrubs; the flat grass/clutter cards are numerous and clutter
    // the view (and cost frame time) for little benefit.
    vegetation.retain(|v| matches!(
        vegetation::kind_for_model(&v.0),
        vegetation::VegetationKind::Tree | vegetation::VegetationKind::Shrub
    ));
    let mut unique_veg_models: Vec<String> = vegetation.iter().map(|v| v.0.clone()).collect();
    unique_veg_models.sort();
    unique_veg_models.dedup();
    unique_veg_models.truncate(220);
    vegetation.retain(|v| unique_veg_models.binary_search(&v.0).is_ok());
    vegetation.truncate(20000);
    println!("[world] vegetation: {} instances / {} models within {:.0}m",
        vegetation.len(), unique_veg_models.len(), veg_radius);

    // Decode every unique model (props + architecture + vegetation) once.
    let mut wanted: Vec<String> = with_models.iter().filter_map(|p| p.model.clone()).collect();
    wanted.extend(unique_brush_models.iter().cloned());
    wanted.extend(unique_veg_models.iter().cloned());
    wanted.sort();
    wanted.dedup();
    let mut models: HashMap<String, Vec<DecodedMesh>> = HashMap::new();
    let mut slots: HashMap<String, Vec<MaterialSlot>> = HashMap::new();
    let mut tex_bytes: HashMap<String, Vec<u8>> = HashMap::new();
    for path in &wanted {
        if models.contains_key(path) {
            continue;
        }
        let Some(bytes) = vfs.read(path) else { continue };
        let dm = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mesh::decode_model(&bytes))) {
            Ok(Some(dm)) => dm,
            _ => continue,
        };
        if let Some(name) = dm.material_name.as_deref() {
            let pal = material::load_mtl(&vfs, path, name);
            for slot in &pal {
                for (_, tp) in &slot.textures {
                    if !tex_bytes.contains_key(tp) {
                        if let Some(b) = vfs.read(tp) {
                            tex_bytes.insert(tp.clone(), b);
                        }
                    }
                }
            }
            slots.insert(path.clone(), pal);
        }
        models.insert(path.clone(), dm.meshes);
    }
    println!("[world] loaded {} models, {} textures", models.len(), tex_bytes.len());

    let center = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2) * Vec3::new(start_cry[0], start_cry[1], start_cry[2]);

    // Player character for play mode, spawned at Mt. Tanner.
    let player_model = std::env::var("SOD_PLAYER").unwrap_or_else(|_| DEFAULT_PLAYER.to_string());
    let mut player_setup = if play {
        let loaded = load_any_character(&vfs, &player_model, None).or_else(|| {
            eprintln!("[play] falling back to single-.chr survivor");
            load_any_character(&vfs, "characters/male/human/body/flannel_jeans/flannel_jeans.chr", None)
        });
        loaded.map(|(model, slots, tex, skeleton, influences, caf, bank)| {
            let h = height_at(&ter, start_cry[0], start_cry[1]);
            PlayerSetup { model, slots, tex_bytes: tex, skeleton, influences, caf, spawn: [start_cry[0], start_cry[1], h + 0.05], yaw: 0.0, bank }
        })
    } else {
        None
    };
    if play {
        match &player_setup {
            Some(ps) => println!("[play] player {} bones, {} meshes, anim {:.2}s, spawn {:?}",
                ps.skeleton.bone_count(), ps.model.meshes.len(), ps.caf.duration_secs(), ps.spawn),
            None => eprintln!("[play] WARNING: player character failed to load"),
        }
    }

    // NPCs (zombies + follower) are opt-in; by default only the player exists.
    let spawn_npcs = std::env::var("SOD_SPAWN_NPCS").is_ok();
    let npc_template = if play && spawn_npcs {
        load_any_character(&vfs, "characters/male/zombie/skin/skin.chr", Some("zombie_male"))
            .map(|(model, slots, tex, skeleton, influences, caf, bank)| NpcTemplate { model, skeleton, influences, caf, slots, tex_bytes: tex, bank })
    } else {
        None
    };

    // Survivor follower (Ed) — assembled from a CDF, driven by npc_ai.
    let follower_template = if play && spawn_npcs {
        load_any_character(&vfs, "characters/female/human/body/tshirt_jeans/variant/v01.cdf", None)
            .map(|(model, slots, tex, skeleton, influences, caf, bank)| FollowerTemplate { model, slots, tex_bytes: tex, skeleton, influences, caf, bank })
    } else {
        None
    };

    // --- simulation state: item DB, loot, tutorial, player vitals ----------
    let db = vfs
        .read("libs/class3/items/items.xml")
        .map(|b| inventory::ItemDb::load_from_xml(&b))
        .unwrap_or_else(inventory::ItemDb::new);
    println!("[game] item db: {} definitions", db.len());
    let mut loot = inventory::LootTable::new();
    for id in ["HealthPack", "AmmoBullet9mm", "FuelRucksack", "Machete", "Pistol_M9"] {
        if db.get(id).is_some() {
            loot.add(inventory::LootEntry::new(id, 3, 1, 2));
        }
    }
    let spawn = player_setup.as_ref().map(|p| p.spawn).unwrap_or(start_cry);
    let ranger = tag_points
        .as_ref()
        .and_then(|tp| tp.get("RTSPoint_RangerStation"))
        .or_else(|| sod_core::level::named_centroid(&placements_all, &["Home_RangerStation"]))
        .unwrap_or([spawn[0] + 42.0, spawn[1] + 26.0, spawn[2]]);
    let survivor = tag_points
        .as_ref()
        .and_then(|tp| tp.get("RTSPoint_ta10Buddies_Start"))
        .or_else(|| tag_points.as_ref().and_then(|tp| tp.get("RTSPoint_taBuddiesCarnage1")))
        .unwrap_or([spawn[0] + 12.0, spawn[1] + 8.0, spawn[2]]);
    println!("[game] Mt. Tanner: start {:?} ranger {:?} survivor {:?}", spawn, ranger, survivor);
    // Face the player toward the survivor / Ranger Station.
    if let Some(ps) = player_setup.as_mut() {
        ps.yaw = (survivor[1] - spawn[1]).atan2(survivor[0] - spawn[0]);
    }
    let opening = mission::OpeningState {
        player_start: spawn,
        survivor_pos: survivor,
        ranger_station: ranger,
        home_pos: spawn,
        kill_target: 3,
        survivor_actor: 0,
    };
    // Real Tanner-arc missions (drives the HUD story line).
    let missions = vfs
        .read("libs/class3/contentmanager/missions.xml")
        .map(|b| mission::parse_missions_xml(&b))
        .unwrap_or_default();
    let tanner: Vec<String> = mission::tanner_arc(&missions).iter().map(|m| m.label.clone()).collect();
    println!("[game] Tanner arc ({} missions): {:?}", tanner.len(), tanner);
    let mut equipment = inventory::Equipment::new();
    let _ = equipment.equip(inventory::ItemClass::Melee, inventory::ItemStack::new("Machete", 1));
    let _ = equipment.equip(inventory::ItemClass::Gun, inventory::ItemStack::new("Pistol_M9", 1));
    let player_state = PlayerState {
        vitals: combat::Vitals::new(combat::PLAYER_MAX_HEALTH, 100.0),
        melee: combat::MeleeWeapon::machete(),
        gun: combat::Firearm::pistol_tacmaster1911(),
        gun_state: combat::GunState::new(7, 21),
        equipment,
        inv: inventory::Inventory::new(8),
        db,
    };
    let game_state = GameState {
        tutorial: mission::Tutorial::new_tanner(&opening),
        rng: inventory::XorShiftRng::new(0x5eed_1234_9abc_def0),
        ai_rng: ai::SmallRng::new(0x0dd0_1234_5678_9abc),
        horde: ai::HordeDirector::new(),
        loot,
        search: None,
        kills: 0,
        move_notified: false,
        auto_cd: 0.0,
        auto_fire_cd: 1.0,
        auto_search_cd: 1.0,
        tanner,
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins
        .set(WindowPlugin {
            primary_window: Some(Window {
                title: "sod-native — State of Decay: YOSE".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        })
        // Compile the scene's render pipelines synchronously so the first
        // timed screenshots draw the world instead of an empty sky.
        .set(bevy::render::RenderPlugin { synchronous_pipeline_compilation: true, ..default() }))
    .insert_resource(ClearColor(Color::srgb(0.45, 0.62, 0.85)))
    .insert_resource(World { terrain: ter, terrain_verts: tverts, terrain_colors: tcolors, terrain_indices: tindex, placements: with_models, models, slots, tex_bytes, brushes, vegetation, ground_tex, ground_albedo })
    .insert_resource(ShotPlan { path: std::env::var("SOD_SHOT").ok().map(PathBuf::from), frame: 0, keys: Vec::new() })
    .insert_resource(MenuFont(vfs.read("fonts/veramono.ttf").unwrap_or_default()))
    .insert_resource(TextureCacheRes(cache::TextureCache::new("target/texture-cache")))
    .insert_resource(CameraStateRes(sod_core::camera::CameraState::new()))
    .insert_resource(CamLagRes(sod_core::camera::CameraLag::new()))
    .insert_resource(HudRes(hud::HudState::new()))
    .add_systems(Startup, setup_world);
    if let Some(c) = collision {
        app.insert_resource(c);
    }
    if let Some(g) = player_graph {
        app.insert_resource(g);
    }
    if let Some(l) = level_lighting {
        app.insert_resource(l);
    }
    if let Some(t) = npc_template {
        app.insert_resource(t);
    }
    if let Some(t) = follower_template {
        app.insert_resource(t);
    }
    if play {
        if let Some(h) = cry_host {
            cry_bridge::install(&mut app, h);
            ai_bridge::install(&mut app);
            app.add_systems(Update, cry_sync_player.after(move_player).in_set(cry_bridge::CrySet::Sync));
        }
        app.insert_resource(player_state);
        app.insert_resource(game_state);
        app.add_systems(Update, (
            // Fixed order: the camera must read this frame's player position.
            // Unordered, Bevy ran them either way round per frame, so the
            // camera alternately saw zero and double movement (a vertical jerk
            // on slopes).
            menu_input, move_player, animate_player.after(move_player), follow_camera.after(move_player),
            npc_ai, animate_npcs, player_attack, npc_attack,
            follower_ai, animate_followers,
            player_vitals, search_system, tutorial_system, hud_update, auto_screenshot,
        ));
    } else {
        app.insert_resource(Orbit { center, radius: 320.0, height: 210.0, angle: 0.6 });
        app.add_systems(Update, (orbit_camera, auto_screenshot));
    }
    if let Some(ps) = player_setup {
        app.insert_resource(ps);
    }
    app.run();
}

fn setup_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut fonts: ResMut<Assets<Font>>,
    menu_font: Option<Res<MenuFont>>,
    cache: Option<Res<TextureCacheRes>>,
    world: Res<World>,
    lighting: Option<Res<LevelLighting>>,
    player_setup: Option<Res<PlayerSetup>>,
    npc_template: Option<Res<NpcTemplate>>,
    follower_template: Option<Res<FollowerTemplate>>,
    cam_state: Option<ResMut<CameraStateRes>>,
    cam_lag: Option<ResMut<CamLagRes>>,
) {
    let base = Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2));
    let cache = cache.as_ref().map(|c| &c.0);
    let mut tex_memo: HashMap<String, Handle<Image>> = HashMap::new();

    // Original menu assets: the game's UI font + the STATE OF DECAY logo cropped
    // from the Scaleform frontend atlas (Class3_frontend_I1.dds).
    let font_h: Handle<Font> = menu_font
        .as_ref()
        .filter(|f| !f.0.is_empty())
        .map(|f| fonts.add(Font::from_bytes(f.0.clone())))
        .unwrap_or_default();
    let logo_h: Option<Handle<Image>> = std::fs::read("assets/menu_logo.png")
        .ok()
        .and_then(|b| {
            Image::from_buffer(&b, ImageType::Extension("png"), CompressedImageFormats::NONE, true, ImageSampler::linear(), RenderAssetUsages::default()).ok()
        })
        .map(|i| images.add(i));

    // Terrain
    let tnormals = mesh::generate_normals(&world.terrain_verts, &world.terrain_indices);
    let mut tmesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    tmesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(world.terrain_verts.clone()));
    tmesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(tnormals));
    tmesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, VertexAttributeValues::Float32x4(world.terrain_colors.clone()));
    let extent = (world.terrain.samples as f32 * world.terrain.spacing).max(1.0);
    let mut terrain_mat = StandardMaterial { base_color: Color::WHITE, perceptual_roughness: 1.0, cull_mode: None, ..default() };
    if let Some((w, h, rgba)) = &world.ground_albedo {
        // One texture spanning the whole terrain: UV is world / extent.
        let tuvs: Vec<[f32; 2]> = world.terrain_verts.iter().map(|p| [p[0] / extent, p[1] / extent]).collect();
        tmesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(tuvs));
        let img = Image::new(
            Extent3d { width: (*w).max(1), height: (*h).max(1), depth_or_array_layers: 1 },
            TextureDimension::D2,
            rgba.clone(),
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        terrain_mat.base_color_texture = Some(images.add(img));
    } else {
        // Fallback: tile a single ground texture every 8 m.
        let tuvs: Vec<[f32; 2]> = world.terrain_verts.iter().map(|p| [p[0] / 8.0, p[1] / 8.0]).collect();
        tmesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(tuvs));
        if let Some(bytes) = &world.ground_tex {
            if let Some(h) = cached_tex(&mut images, cache, &mut tex_memo, "ground_grass_dirt", bytes, true) {
                terrain_mat.base_color_texture = Some(h);
            }
        }
    }
    tmesh.insert_indices(Indices::U32(world.terrain_indices.clone()));
    let terrain_mat = materials.add(terrain_mat);

    // Prebuild model meshes + materials. One material per palette slot, and one
    // mesh per submesh, so complex multi-material models texture correctly.
    let mut built: HashMap<String, Vec<(Handle<Mesh>, Handle<StandardMaterial>)>> = HashMap::new();
    for (path, dmeshes) in &world.models {
        let mut mat_handles: Vec<Handle<StandardMaterial>> = Vec::new();
        if let Some(slots) = world.slots.get(path) {
            for slot in slots {
                let mut mat = StandardMaterial { base_color: Color::srgb(0.78, 0.76, 0.72), perceptual_roughness: 0.9, cull_mode: None, alpha_mode: AlphaMode::Mask(0.5), ..default() };
                if let Some(tp) = slot.texture("Diffuse") {
                    if let Some(bytes) = world.tex_bytes.get(tp) {
                        if let Some(h) = cached_tex(&mut images, cache, &mut tex_memo, tp, bytes, true) {
                            mat.base_color_texture = Some(h);
                        }
                    }
                }
                mat_handles.push(materials.add(mat));
            }
        }
        if mat_handles.is_empty() {
            mat_handles.push(materials.add(StandardMaterial { base_color: Color::srgb(0.78, 0.76, 0.72), perceptual_roughness: 0.9, cull_mode: None, ..default() }));
        }
        let mut list = Vec::new();
        for m in dmeshes {
            if m.positions.is_empty() { continue; }
            let build_mesh = |indices: &[u32]| {
                let mut bm = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
                bm.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(m.positions.clone()));
                bm.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(m.normals.clone()));
                if m.uvs.len() == m.positions.len() { bm.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(m.uvs.clone())); }
                bm.insert_indices(Indices::U32(indices.to_vec()));
                bm
            };
            if m.submeshes.is_empty() {
                if m.indices.is_empty() { continue; }
                list.push((meshes.add(build_mesh(&m.indices)), mat_handles[0].clone()));
            } else {
                for sm in &m.submeshes {
                    if sm.indices.is_empty() { continue; }
                    let mat = mat_handles.get(sm.material as usize).cloned().unwrap_or_else(|| mat_handles[0].clone());
                    list.push((meshes.add(build_mesh(&sm.indices)), mat));
                }
            }
        }
        built.insert(path.clone(), list);
    }

    // World root: terrain + props.
    commands.spawn((base, Visibility::default())).with_children(|root| {
        root.spawn((Mesh3d(meshes.add(tmesh)), MeshMaterial3d(terrain_mat), Transform::default()));
        let mut spawned = 0;
        for p in &world.placements {
            let Some(path) = &p.model else { continue };
            let Some(list) = built.get(path) else { continue };
            let tf = Transform {
                translation: Vec3::new(p.pos[0], p.pos[1], p.pos[2]),
                rotation: Quat::from_xyzw(p.rot[0], p.rot[1], p.rot[2], p.rot[3]),
                scale: Vec3::new(p.scale[0], p.scale[1], p.scale[2]),
            };
            for (mh, mat) in list { root.spawn((Mesh3d(mh.clone()), MeshMaterial3d(mat.clone()), tf)); }
            spawned += 1;
        }
        // Static architecture (buildings, walls, rocks) as instances.
        let mut bspawned = 0;
        for (model, pos, rot, scale) in &world.brushes {
            let Some(list) = built.get(model) else { continue };
            let tf = Transform {
                translation: Vec3::new(pos[0], pos[1], pos[2]),
                rotation: Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]),
                scale: Vec3::splat(*scale),
            };
            for (mh, mat) in list { root.spawn((Mesh3d(mh.clone()), MeshMaterial3d(mat.clone()), tf)); }
            bspawned += 1;
        }
        // Vegetation (trees/bushes).
        let mut vspawned = 0;
        for (model, pos, scale, yaw) in &world.vegetation {
            let Some(list) = built.get(model) else { continue };
            let tf = Transform {
                translation: Vec3::new(pos[0], pos[1], pos[2]),
                rotation: Quat::from_rotation_z(*yaw),
                scale: Vec3::splat(*scale),
            };
            for (mh, mat) in list { root.spawn((Mesh3d(mh.clone()), MeshMaterial3d(mat.clone()), tf)); }
            vspawned += 1;
        }
        info!("[world] spawned {} props, {} brushes, {} vegetation", spawned, bspawned, vspawned);
    });

    // Player (per-submesh palette materials).
    if let Some(ps) = player_setup {
        let mut mat_handles: Vec<Handle<StandardMaterial>> = Vec::new();
        for slot in &ps.slots {
            let mut mat = StandardMaterial { base_color: Color::srgb(0.8, 0.72, 0.62), perceptual_roughness: 0.85, cull_mode: None, ..default() };
            if let Some(tp) = slot.texture("Diffuse") {
                if let Some(bytes) = ps.tex_bytes.get(tp) {
                    if let Some(h) = cached_tex(&mut images, cache, &mut tex_memo, tp, bytes, true) { mat.base_color_texture = Some(h); }
                }
            }
            mat_handles.push(materials.add(mat));
        }
        if mat_handles.is_empty() {
            mat_handles.push(materials.add(StandardMaterial { base_color: Color::srgb(0.8, 0.72, 0.62), perceptual_roughness: 0.85, cull_mode: None, ..default() }));
        }
        let mut parts = Vec::new();
        let yaw = ps.yaw;
        // Match `move_player`: the assembled model's forward is local +Y, so the
        // render transform needs the `render_yaw` (-π/2) correction from spawn.
        let rot = base.rotation * Quat::from_rotation_z(sod_core::class3_move::render_yaw(yaw));
        let tf0 = Transform { translation: base.rotation * Vec3::new(ps.spawn[0], ps.spawn[1], ps.spawn[2]), rotation: rot, scale: Vec3::ONE };
        commands.spawn((tf0, Visibility::default(), PlayerRoot)).with_children(|parent| {
            for (mi, m) in ps.model.meshes.iter().enumerate() {
                if m.positions.is_empty() { continue; }
                let infl = ps.influences.get(mi).cloned().unwrap_or_default();
                let groups: Vec<(usize, Vec<u32>)> = if m.submeshes.is_empty() {
                    vec![(0, m.indices.clone())]
                } else {
                    m.submeshes.iter().map(|s| (s.material as usize, s.indices.clone())).collect()
                };
                for (mat_i, idx) in groups {
                    if idx.is_empty() { continue; }
                    let mut bm = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
                    bm.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(m.positions.clone()));
                    bm.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(m.normals.clone()));
                    if m.uvs.len() == m.positions.len() { bm.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(m.uvs.clone())); }
                    bm.insert_indices(Indices::U32(idx));
                    let handle = meshes.add(bm);
                    let mh = mat_handles.get(mat_i).cloned().unwrap_or_else(|| mat_handles[0].clone());
                    parent.spawn((Mesh3d(handle.clone()), MeshMaterial3d(mh.clone())));
                    parts.push(AnimPart { handle, bind_pos: m.positions.clone(), bind_nrm: m.normals.clone(), infl: infl.clone(), mat: mh });
                }
            }
        });
        let mut mstate = sod_core::class3_move::MoveState::new([ps.spawn[0], ps.spawn[1], ps.spawn[2]]);
        mstate.yaw = yaw;
        commands.insert_resource(Player { yaw, pos: [ps.spawn[0], ps.spawn[1]], parts, skeleton: ps.skeleton.clone(), caf: ps.caf.clone(), time: 0.0, moving: false, rate: 1.0, bank: ps.bank.clone(), loc: anim::Locomotion::Idle, mstate, ploco: player_locomotion::PlayerLocomotion::with_decoupled(tuning::default_movement(), true) });
        // Start the camera behind the player and re-seed the lag so it snaps.
        if let Some(mut cs) = cam_state {
            cs.0.yaw = yaw;
            cs.0.pitch = 0.0;
        }
        if let Some(mut l) = cam_lag {
            l.0.seeded = false;
        }

        // Spawn zombies around the player.
        if let Some(nt) = &npc_template {
          if std::env::var("SOD_SPAWN_NPCS").is_ok() {
            let mut nmat_handles: Vec<Handle<StandardMaterial>> = Vec::new();
            for slot in &nt.slots {
                let mut nmat = StandardMaterial { base_color: Color::srgb(0.62, 0.55, 0.5), perceptual_roughness: 0.9, cull_mode: None, ..default() };
                if let Some(tp) = slot.texture("Diffuse") {
                    if let Some(bytes) = nt.tex_bytes.get(tp) {
                        if let Some(h) = cached_tex(&mut images, cache, &mut tex_memo, tp, bytes, true) { nmat.base_color_texture = Some(h); }
                    }
                }
                nmat_handles.push(materials.add(nmat));
            }
            if nmat_handles.is_empty() {
                nmat_handles.push(materials.add(StandardMaterial { base_color: Color::srgb(0.62, 0.55, 0.5), perceptual_roughness: 0.9, cull_mode: None, ..default() }));
            }
            for i in 0..6 {
                let cx = ps.spawn[0] + 8.0 + (i % 3) as f32 * 3.5;
                let cy = ps.spawn[1] + (i as f32 - 2.5) * 2.5;
                let h = height_at(&world.terrain, cx, cy);
                let wpos = base.rotation * Vec3::new(cx, cy, h);
                let mut parts = Vec::new();
                for (mi, m) in nt.model.meshes.iter().enumerate() {
                    if m.positions.is_empty() { continue; }
                    let infl = nt.influences.get(mi).cloned().unwrap_or_default();
                    let groups: Vec<(usize, Vec<u32>)> = if m.submeshes.is_empty() {
                        vec![(0, m.indices.clone())]
                    } else {
                        m.submeshes.iter().map(|s| (s.material as usize, s.indices.clone())).collect()
                    };
                    for (mat_i, idx) in groups {
                        if idx.is_empty() { continue; }
                        let mut bm = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
                        bm.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(m.positions.clone()));
                        bm.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(m.normals.clone()));
                        if m.uvs.len() == m.positions.len() { bm.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(m.uvs.clone())); }
                        bm.insert_indices(Indices::U32(idx));
                        let handle = meshes.add(bm);
                        let mh = nmat_handles.get(mat_i).cloned().unwrap_or_else(|| nmat_handles[0].clone());
                        parts.push(AnimPart { handle, bind_pos: m.positions.clone(), bind_nrm: m.normals.clone(), infl: infl.clone(), mat: mh });
                    }
                }
                let kind = match i % 4 {
                    1 => sod_core::game::ZombieKind::Feral,
                    3 => sod_core::game::ZombieKind::Screamer,
                    _ => sod_core::game::ZombieKind::Grunt,
                };
                let agent = zombie_ai::ZombieAgent::new(kind, [cx, cy, h]);
                let npc = Npc { kind, hp: kind.stats().hp, time: i as f32 * 0.31, parts: parts.clone(), skeleton: nt.skeleton.clone(), caf: nt.caf.clone(), agent, bank: nt.bank.clone(), loc: anim::Locomotion::Idle };
                commands.spawn((Transform::from_translation(wpos).with_rotation(base.rotation), Visibility::default(), npc))
                    .with_children(|parent| {
                        for part in &parts { parent.spawn((Mesh3d(part.handle.clone()), MeshMaterial3d(part.mat.clone()))); }
                    });
            }
            println!("[game] spawned 6 zombies");
          }
        }

        // Survivor follower (Ed), driven by npc_ai.
        if let Some(ft) = &follower_template {
          if std::env::var("SOD_SPAWN_NPCS").is_ok() {
            let mut mat = StandardMaterial { base_color: Color::srgb(0.8, 0.72, 0.62), perceptual_roughness: 0.85, cull_mode: None, ..default() };
            if let Some(slot) = ft.slots.iter().find(|s| s.texture("Diffuse").is_some()) {
                if let Some(tp) = slot.texture("Diffuse") {
                    if let Some(bytes) = ft.tex_bytes.get(tp) {
                        if let Some(h) = cached_tex(&mut images, cache, &mut tex_memo, tp, bytes, true) {
                            mat.base_color_texture = Some(h);
                        }
                    }
                }
            }
            let mat_h = materials.add(mat);
            let cx = ps.spawn[0] + 2.5;
            let cy = ps.spawn[1] + 2.5;
            let h = height_at(&world.terrain, cx, cy);
            let wpos = base.rotation * Vec3::new(cx, cy, h);
            let mut parts = Vec::new();
            for (mi, m) in ft.model.meshes.iter().enumerate() {
                if m.positions.is_empty() { continue; }
                let infl = ft.influences.get(mi).cloned().unwrap_or_default();
                let groups: Vec<Vec<u32>> = if m.submeshes.is_empty() {
                    vec![m.indices.clone()]
                } else {
                    m.submeshes.iter().map(|s| s.indices.clone()).collect()
                };
                for idx in groups {
                    if idx.is_empty() { continue; }
                    let mut bm = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
                    bm.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(m.positions.clone()));
                    bm.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(m.normals.clone()));
                    if m.uvs.len() == m.positions.len() { bm.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(m.uvs.clone())); }
                    bm.insert_indices(Indices::U32(idx));
                    let handle = meshes.add(bm);
                    parts.push(AnimPart { handle, bind_pos: m.positions.clone(), bind_nrm: m.normals.clone(), infl: infl.clone(), mat: mat_h.clone() });
                }
            }
            let mut agent = npc_ai::NpcAgent::new(1, [cx, cy, h], npc_ai::NpcRole::Survivor);
            agent.set_order(npc_ai::NpcOrder::Follow { leader: 0, distance: 2.5 });
            commands.spawn((Transform::from_translation(wpos).with_rotation(base.rotation), Visibility::default(),
                Follower { agent, time: 0.0, moving: false, parts: parts.clone(), skeleton: ft.skeleton.clone(), caf: ft.caf.clone(), bank: ft.bank.clone(), loc: anim::Locomotion::Idle }))
                .with_children(|parent| {
                    for part in &parts { parent.spawn((Mesh3d(part.handle.clone()), MeshMaterial3d(part.mat.clone()))); }
                });
            println!("[game] spawned follower");
          }
        }

        // HUD: health bar + status text.
        commands.spawn((Node { position_type: PositionType::Absolute, left: Val::Px(20.0), top: Val::Px(20.0), width: Val::Px(300.0), height: Val::Px(20.0), border: UiRect::all(Val::Px(2.0)), ..default() },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)), BorderColor::all(Color::WHITE), HudHide))
            .with_children(|p| {
                p.spawn((Node { width: Val::Percent(100.0), height: Val::Percent(100.0), ..default() },
                    BackgroundColor(Color::srgb(0.75, 0.12, 0.12)), HealthBarFill));
            });
        // Stamina bar under the health bar.
        commands.spawn((Node { position_type: PositionType::Absolute, left: Val::Px(20.0), top: Val::Px(46.0), width: Val::Px(300.0), height: Val::Px(12.0), border: UiRect::all(Val::Px(2.0)), ..default() },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)), BorderColor::all(Color::WHITE), HudHide))
            .with_children(|p| {
                p.spawn((Node { width: Val::Percent(100.0), height: Val::Percent(100.0), ..default() },
                    BackgroundColor(Color::srgb(0.85, 0.75, 0.15)), StaminaBarFill));
            });
        commands.spawn((Text::new("HP 200/200   Stamina 100   Zombies 6"),
            TextFont { font_size: FontSize::Px(18.0), ..default() }, TextColor(Color::WHITE), HudLine::Status, HudHide,
            Node { position_type: PositionType::Absolute, left: Val::Px(20.0), top: Val::Px(66.0), ..default() }));
        commands.spawn((Text::new("OBJECTIVE: Use [WASD] to move."),
            TextFont { font_size: FontSize::Px(18.0), ..default() }, TextColor(Color::srgb(0.95, 0.9, 0.5)), HudLine::Objective, HudHide,
            Node { position_type: PositionType::Absolute, left: Val::Px(20.0), top: Val::Px(94.0), ..default() }));
        commands.spawn((Text::new("Backpack: empty"),
            TextFont { font_size: FontSize::Px(16.0), ..default() }, TextColor(Color::srgb(0.8, 0.85, 0.9)), HudLine::Inventory, HudHide,
            Node { position_type: PositionType::Absolute, left: Val::Px(20.0), top: Val::Px(120.0), ..default() }));
        // Class3HUD: banner, center prompt, notifications.
        commands.spawn((Text::new(""),
            TextFont { font_size: FontSize::Px(22.0), ..default() }, TextColor(Color::srgb(0.95, 0.85, 0.6)), HudLine::Banner, HudHide,
            Node { position_type: PositionType::Absolute, left: Val::Px(20.0), top: Val::Px(150.0), ..default() }));
        commands.spawn((Text::new(""),
            TextFont { font_size: FontSize::Px(20.0), ..default() }, TextColor(Color::srgb(0.9, 0.95, 1.0)), HudLine::Prompt, HudHide,
            Node { position_type: PositionType::Absolute, left: Val::Percent(38.0), top: Val::Percent(60.0), ..default() }));
        commands.spawn((Text::new(""),
            TextFont { font_size: FontSize::Px(16.0), ..default() }, TextColor(Color::srgb(0.85, 0.9, 0.7)), HudLine::Notifications, HudHide,
            Node { position_type: PositionType::Absolute, left: Val::Px(20.0), top: Val::Px(182.0), ..default() }));
        // Radar (Class3HUDRadar) with a fixed blip pool.
        commands.spawn((Node { position_type: PositionType::Absolute, left: Val::Px(20.0), bottom: Val::Px(20.0), width: Val::Px(160.0), height: Val::Px(160.0),
                border: UiRect::all(Val::Px(2.0)), ..default() },
            BackgroundColor(Color::srgba(0.05, 0.08, 0.05, 0.6)), BorderColor::all(Color::srgb(0.5, 0.7, 0.5)), HudHide))
            .with_children(|p| {
                for i in 0..32 {
                    p.spawn((Node { position_type: PositionType::Absolute, left: Val::Px(78.0), top: Val::Px(78.0), width: Val::Px(5.0), height: Val::Px(5.0), ..default() },
                        BackgroundColor(Color::srgb(0.9, 0.3, 0.2)), RadarBlipNode { index: i }));
                }
            });

        // Original main menu: real labels/order (embeddedstrings.xml) + the real
        // STATE OF DECAY logo from the Scaleform frontend atlas.
        commands.insert_resource(MenuRes { active: true, index: 0 });
        const MENU_ITEMS: [&str; 8] = [
            "Continue",
            "Start a New Game",
            "Play Breakdown",
            "Play Lifeline",
            "Achievements",
            "Help & Options",
            "Leaderboards",
            "Exit Game",
        ];
        commands
            .spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    row_gap: Val::Px(4.0),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.012, 0.016, 0.024, 0.88)),
                MenuUi,
                MenuRoot,
            ))
            .with_children(|p| {
                if let Some(logo) = &logo_h {
                    p.spawn((
                        ImageNode::new(logo.clone()),
                        Node { width: Val::Px(420.0), height: Val::Px(181.0), margin: UiRect::bottom(Val::Px(20.0)), ..default() },
                        MenuUi,
                    ));
                } else {
                    p.spawn((
                        Text::new("STATE OF DECAY"),
                        TextFont { font: bevy::text::FontSource::Handle(font_h.clone()), font_size: FontSize::Px(56.0), ..default() },
                        TextColor(Color::srgb(0.86, 0.89, 0.72)),
                        MenuUi,
                    ));
                }
                p.spawn((
                    Node { width: Val::Px(380.0), height: Val::Px(2.0), margin: UiRect::vertical(Val::Px(12.0)), ..default() },
                    BackgroundColor(Color::srgb(0.35, 0.12, 0.1)),
                    MenuUi,
                ));
                for (i, label) in MENU_ITEMS.iter().enumerate() {
                    p.spawn((
                        Text::new(*label),
                        TextFont { font: bevy::text::FontSource::Handle(font_h.clone()), font_size: FontSize::Px(25.0), ..default() },
                        TextColor(if i == 0 { Color::srgb(0.96, 0.93, 0.8) } else { Color::srgb(0.6, 0.6, 0.58) }),
                        MenuItem { index: i },
                        MenuUi,
                    ));
                }
                p.spawn((
                    Text::new("State of Decay: Year-One — native engine"),
                    TextFont { font: bevy::text::FontSource::Handle(font_h.clone()), font_size: FontSize::Px(13.0), ..default() },
                    TextColor(Color::srgb(0.38, 0.4, 0.42)),
                    Node { position_type: PositionType::Absolute, left: Val::Px(16.0), bottom: Val::Px(12.0), ..default() },
                    MenuUi,
                ));
            });
    }

    // Camera + lights. A bright key sun, a cool fill and a global ambient term
    // so the static architecture is not left in shadow.
    // CryEngine's FOV (64) is *horizontal*; Bevy's Projection is vertical. Convert
    // through the 16:9 aspect so the framing matches the game. Tunable: SOD_FOV=<horizontal deg>.
    let hfov = std::env::var("SOD_FOV").ok().and_then(|s| s.parse::<f32>().ok())
        .map(f32::to_radians)
        .unwrap_or(sod_core::camera::FOV_RADIANS);
    let vfov = 2.0 * ((hfov * 0.5).tan() * (720.0f32 / 1280.0)).atan();
    commands.spawn((Camera3d::default(),
        Projection::Perspective(PerspectiveProjection { fov: vfov, ..default() }),
        Transform::from_xyz(0.0, 100.0, 200.0).looking_at(Vec3::ZERO, Vec3::Y), Tonemapping::None,
        AmbientLight { color: Color::srgb(0.55, 0.63, 0.78), brightness: 110.0, ..default() },
        MainCamera));
    match lighting.as_ref() {
        // The level's sun direction; neutral colour, plus the previous fill
        // light until the sky-light / HDR pipeline is ported.
        Some(l) => {
            commands.spawn((
                DirectionalLight { shadow_maps_enabled: false, illuminance: 3600.0, color: Color::srgb(1.0, 0.95, 0.85), ..default() },
                Transform::default().looking_to(l.sun_dir, Vec3::Y),
            ));
            commands.spawn((DirectionalLight { shadow_maps_enabled: false, illuminance: 700.0, color: Color::srgb(0.7, 0.8, 1.0), ..default() },
                Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, -2.2, 0.0))));
        }
        None => {
            commands.spawn((DirectionalLight { shadow_maps_enabled: false, illuminance: 3600.0, color: Color::srgb(1.0, 0.95, 0.85), ..default() },
                Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -1.0, 0.5, 0.0))));
            commands.spawn((DirectionalLight { shadow_maps_enabled: false, illuminance: 700.0, color: Color::srgb(0.7, 0.8, 1.0), ..default() },
                Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, -2.2, 0.0))));
        }
    }
    info!("scene ready");
}

fn menu_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut menu: ResMut<MenuRes>,
    mut commands: Commands,
    q: Query<Entity, With<MenuRoot>>,
    mut items: Query<(&MenuItem, &mut TextColor)>,
) {
    if !menu.active {
        return;
    }
    const N: usize = 8;
    if keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::KeyS) {
        menu.index = (menu.index + 1) % N;
    }
    if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW) {
        menu.index = (menu.index + N - 1) % N;
    }
    for (item, mut color) in &mut items {
        *color = TextColor(if item.index == menu.index {
            Color::srgb(0.96, 0.93, 0.8)
        } else {
            Color::srgb(0.6, 0.6, 0.58)
        });
    }
    if keys.just_pressed(KeyCode::Enter) {
        match menu.index {
            7 => {
                println!("[menu] Exit Game");
                std::process::exit(0);
            }
            2..=6 => {
                let names = ["", "", "Play Breakdown", "Play Lifeline", "Achievements", "Help & Options", "Leaderboards"];
                println!("[menu] {} (DLC/options screen not yet implemented)", names[menu.index]);
            }
            other => {
                for e in &q {
                    commands.entity(e).despawn();
                }
                menu.active = false;
                println!("[menu] {}", if other == 0 { "Continue" } else { "Start a New Game" });
            }
        }
    }
}

fn move_player(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    world: Res<World>,
    menu: Res<MenuRes>,
    cam_state: Option<Res<CameraStateRes>>,
    mut player: ResMut<Player>,
    mut game: ResMut<GameState>,
    mut q: Query<&mut Transform, With<PlayerRoot>>,
    collision: Option<Res<Collision>>,
    graph: Option<ResMut<PlayerGraph>>,
) {
    if menu.active {
        return;
    }
    let dt = time.delta_secs();
    let t = tuning::default_movement();

    // Camera-relative input (WASD / arrows). Shift = sprint, Ctrl/C = crouch.
    let fwd: f32 = if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) { 1.0 }
        else if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) { -1.0 } else { 0.0 };
    let strafe: f32 = if keys.pressed(KeyCode::KeyD) { 1.0 } else if keys.pressed(KeyCode::KeyA) { -1.0 } else { 0.0 };
    let crouch = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) || keys.pressed(KeyCode::KeyC);
    let sprint = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    // Demo-only auto-walk for screenshots; deliberately NOT named SOD_WALK so a
    // stray env var can't make the player walk without input.
    let auto = std::env::var("SOD_AUTOWALK").is_ok();

    let (fwd, strafe) = if auto { (1.0, 0.0) } else { (fwd, strafe) };

    // Movement is camera-relative (Class3): build the desired world direction from
    // the camera yaw, then express it in the character's local frame for the
    // controller, which turns the character toward it.
    let cam_yaw = cam_state.as_ref().map(|c| c.0.yaw).unwrap_or(player.mstate.yaw);
    let (sc, cc) = cam_yaw.sin_cos();
    let wdx = cc * fwd - sc * strafe;
    let wdy = sc * fwd + cc * strafe;
    let mag = (fwd * fwd + strafe * strafe).sqrt();
    let world_yaw = if mag > 1e-4 { wdy.atan2(wdx) } else { player.mstate.yaw };
    let rel = wrap_angle(world_yaw - player.mstate.yaw);
    let (fwd_l, strafe_l) = (rel.cos() * mag, rel.sin() * mag);

    // 1:1 Class3 controller: CPlayerMovementController::UpdateNormal + CPlayerRotation.
    let input = sod_core::class3_move::MoveInput { forward: fwd_l, strafe: strafe_l, mouse_dx: 0.0, mouse_dy: 0.0, crouch, sprint, dt };
    use sod_core::class3_physics::{PLAYER_CAPSULE_HEIGHT, PLAYER_CAPSULE_RADIUS, STEP_HEIGHT};
    let feet_z = player.mstate.pos[2];
    // Ground = the first surface below the feet, reachable within one step up
    // (roofs, porches and rocks included), else the terrain.
    let ground = |x: f32, y: f32| match collision.as_ref() {
        Some(c) => c
            .0
            .raycast([x, y, feet_z + STEP_HEIGHT + 0.05], [0.0, 0.0, -1.0], 60.0)
            .map(|h| h.point[2])
            .unwrap_or_else(|| height_at(&world.terrain, x, y)),
        None => height_at(&world.terrain, x, y),
    };
    let before = player.mstate.pos;
    sod_core::class3_move::update_normal(&mut player.mstate, &input, &t, &ground, &[]);
    // Walls: slide the capsule from the previous position along the move;
    // keep the controller's grounded height and drop velocity into the wall.
    if let Some(c) = collision.as_ref() {
        let after = player.mstate.pos;
        let delta = [after[0] - before[0], after[1] - before[1], 0.0];
        let lifted = [before[0], before[1], after[2] + STEP_HEIGHT];
        let sw = c.0.sweep_capsule(lifted, delta, PLAYER_CAPSULE_RADIUS, PLAYER_CAPSULE_HEIGHT - STEP_HEIGHT);
        player.mstate.pos[0] = sw.pos[0];
        player.mstate.pos[1] = sw.pos[1];
        if let Some(ct) = sw.contact {
            let n = [ct.normal[0], ct.normal[1]];
            let nl = (n[0] * n[0] + n[1] * n[1]).sqrt();
            if nl > 1e-3 {
                let n = [n[0] / nl, n[1] / nl];
                let into = player.mstate.vel[0] * n[0] + player.mstate.vel[1] * n[1];
                if into < 0.0 {
                    player.mstate.vel[0] -= into * n[0];
                    player.mstate.vel[1] -= into * n[1];
                }
            }
        }
    }
    player.pos = [player.mstate.pos[0], player.mstate.pos[1]];
    player.yaw = player.mstate.yaw;

    // Animation graph: inputs from this frame's movement.
    if let Some(mut g) = graph {
        let req_speed = if sprint { t.sprint } else { t.run } * mag.min(1.0);
        let wl = (wdx * wdx + wdy * wdy).sqrt().max(1e-6);
        let v = player.mstate.vel;
        let snap = sod_core::ag_driver::MovementSnapshot {
            requested_vel: [wdx / wl * req_speed, wdy / wl * req_speed],
            actual_vel: [v[0], v[1]],
            body_yaw: sod_core::class3_move::render_yaw(player.mstate.yaw),
            pseudo_speed: sod_core::class3_move::pseudo_speed_from_speed((v[0] * v[0] + v[1] * v[1]).sqrt(), t.run),
            stance: if crouch { "crouch".into() } else { "relaxed".into() },
            ..Default::default()
        };
        let before = g.0.current_name().map(str::to_string);
        // INTERIM: one link per frame; the engine advances when the current
        // state's animation allows leaving it (needs the resolved clips).
        g.0.update(&snap, |_| None, true);
        if std::env::var("SOD_AGDBG").is_ok() && g.0.current_name().map(str::to_string) != before {
            eprintln!("[ag] {:?} -> {:?} (target {:?}, pseudo {:.2})", before, g.0.current_name(), g.0.target_name(), snap.pseudo_speed);
        }
    }
    let speed = (player.mstate.vel[0] * player.mstate.vel[0] + player.mstate.vel[1] * player.mstate.vel[1]).sqrt();
    player.moving = speed > 0.05;

    if player.moving && !game.move_notified {
        game.move_notified = true;
        game.tutorial.notify(&mission::TutorialTrigger::Custom("Scene.Move".to_string()));
    }

    // Decompiled locomotion state machine: real speed thresholds -> PseudoSpeed band.
    let aiming = mouse.pressed(MouseButton::Right);
    let gait = locomotion::Gait::from_speed(speed, &t);
    let ps = match gait {
        locomotion::Gait::Idle => 0.1,
        locomotion::Gait::Creep => 0.3,
        locomotion::Gait::Walk => 0.6,
        locomotion::Gait::Run => 1.8,
        locomotion::Gait::Sprint => 3.0,
    };
    player.ploco.update(speed, ps, aiming, dt);
    player.rate = player.ploco.playback_rate;
    let loc = match player.ploco.clip_gait() {
        locomotion::Gait::Idle => anim::Locomotion::Idle,
        locomotion::Gait::Creep | locomotion::Gait::Walk => anim::Locomotion::Walk,
        locomotion::Gait::Run => anim::Locomotion::Run,
        locomotion::Gait::Sprint => anim::Locomotion::Sprint,
    };
    if loc != player.loc {
        let c = player.bank.caf_for(loc).cloned();
        if let Some(c) = c {
            player.caf = c;
            player.loc = loc;
        }
    }

    let h = player.mstate.pos[2];
    let base_rot = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let world_pos = base_rot * Vec3::new(player.pos[0], player.pos[1], h);
    // CryEngine models face local +Y; the controller yaw 0 faces +X.
    let rot = base_rot * Quat::from_rotation_z(sod_core::class3_move::render_yaw(player.yaw));
    for mut t in &mut q {
        t.translation = world_pos;
        t.rotation = rot;
    }
}

fn animate_player(anim: Option<ResMut<Player>>, mut meshes: ResMut<Assets<Mesh>>, time: Res<Time>) {
    let Some(mut anim) = anim else { return };
    if anim.parts.is_empty() { return; }
    // Clip selection happens in `move_player`; here we just advance the clip at
    // the speed-matched rate.
    anim.time += time.delta_secs() * anim.rate.max(0.05);
    let dur = anim.caf.duration_secs().max(1.0 / 30.0);
    let tick = anim.caf.tick_from_seconds(anim.time % dur);
    let sample = anim.caf.sample_tick(tick);
    let posed = anim.skeleton.pose_anchored(&sample);
    let skin_m = anim.skeleton.skin_matrices(&posed);
    for part in &anim.parts {
        let npos = skin::deform(&part.bind_pos, &part.infl, &skin_m);
        let nnrm = skin::deform_normals(&part.bind_nrm, &part.infl, &skin_m);
        if let Some(mut m) = meshes.get_mut(&part.handle) {
            m.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(npos));
            m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(nnrm));
        }
    }
}

fn follow_camera(
    time: Res<Time>,
    motion: Res<AccumulatedMouseMotion>,
    mouse: Res<ButtonInput<MouseButton>>,
    menu: Res<MenuRes>,
    player: Option<Res<Player>>,
    cam_state: Option<ResMut<CameraStateRes>>,
    lag: Option<ResMut<CamLagRes>>,
    mut cam: Query<&mut Transform, (With<MainCamera>, Without<PlayerRoot>)>,
    listener: Option<Res<cry_bridge::CryListener>>,
) {
    let Some(player) = player else { return };
    let Some(mut cs) = cam_state else { return };
    // The camera takes raw mouse counts, exactly as the game's rotate action
    // does (`camera::mouse_count_scale`); SOD_SENS is an optional extra
    // multiplier (default 1). Axes are flipped to the standard (mouse right =
    // look right, mouse up = look up); set SOD_INVERT=0 for the raw signs.
    let sens = std::env::var("SOD_SENS").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
    let invert = std::env::var("SOD_INVERT").map(|v| v != "0").unwrap_or(true);
    let sgn = if invert { -1.0 } else { 1.0 };
    let (mut dx, mut dy) = (motion.delta.x * sens * sgn, motion.delta.y * sens * sgn);
    if menu.active {
        dx = 0.0;
        dy = 0.0;
    }
    let t = tuning::default_camera();
    // Focal target = player chest (cry coords); aim on right mouse.
    let target = camera::focal_target([player.mstate.pos[0], player.mstate.pos[1], player.mstate.pos[2]]);
    let aim = mouse.pressed(MouseButton::Right);
    let input = camera::CameraInput { mouse_dx: dx, mouse_dy: dy, aim, dt: time.delta_secs(), ..Default::default() };
    camera::update(&mut cs.0, &input, &t, target);
    // Recenter the camera behind the player while walking (nav camera), unless
    // the player is actively looking this frame.
    if player.moving && dx.abs() < 0.5 && dy.abs() < 0.5 {
        let d = wrap_angle(player.mstate.yaw - cs.0.yaw);
        cs.0.yaw = wrap_angle(cs.0.yaw + d * (time.delta_secs() * 2.0).min(1.0));
    }
    // CryEngine camera lag/smoothing (FUN_015ea260) so the camera trails the
    // target instead of snapping.
    let cam_pos = match lag {
        // Time constant in seconds: retail captures show the focal point
        // trailing ~0.75 m at 4 m/s run (tau ~0.19 s) and settling within
        // ~0.4 s of stopping (notes/telemetry/dyn.csv), matching the default.
        Some(mut l) => l.0.step(cs.0.pos, cs.0.yaw, cs.0.pitch, camera::DEFAULT_LAG_SCALE, time.delta_secs()).0,
        None => cs.0.pos,
    };
    if std::env::var("SOD_CAMDBG").is_ok() {
        eprintln!("[camdbg] t={:.3} x={:.3} y={:.3} feet_z={:.4} target_z={:.4} cam_z={:.4} lag_z={:.4} vz={:.3} grounded={} vx={:.3} vy={:.3}",
            time.elapsed_secs(), player.mstate.pos[0], player.mstate.pos[1], player.mstate.pos[2], target[2], cs.0.pos[2], cam_pos[2], player.mstate.vel[2], player.mstate.grounded, player.mstate.vel[0], player.mstate.vel[1]);
    }
    let base_rot = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let pos = base_rot * Vec3::new(cam_pos[0], cam_pos[1], cam_pos[2]);
    let tgt = base_rot * Vec3::new(target[0], target[1], target[2]);
    for mut tr in &mut cam {
        *tr = Transform::from_translation(pos).looking_at(tgt, Vec3::Y);
    }
    // Audio listener = the camera, in CryEngine space (Z-up).
    if let Some(l) = listener {
        let to_cry = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let fwd = to_cry * (tgt - pos).normalize_or_zero();
        let up = to_cry * Vec3::Y;
        if let Ok(mut li) = l.0.lock() {
            li.position = cam_pos;
            li.forward = fwd.to_array();
            li.up = up.to_array();
        }
    }
}

fn orbit_camera(time: Res<Time>, mut orbit: ResMut<Orbit>, mut q: Query<&mut Transform, With<MainCamera>>) {
    orbit.angle += time.delta_secs() * 0.12;
    let c = orbit.center;
    let pos = c + Vec3::new(orbit.angle.cos() * orbit.radius, orbit.height, orbit.angle.sin() * orbit.radius);
    for mut t in &mut q { *t = Transform::from_translation(pos).looking_at(c, Vec3::Y); }
}

// ---------------------------------------------------------------------------
// single-model mode
// ---------------------------------------------------------------------------

fn run_model(vfs: Vfs, model_path: String) {
    let bytes = vfs.read(&model_path).unwrap_or_else(|| panic!("{model_path} not found"));
    let skeleton = skin::parse_skeleton(&bytes);
    let influences = skin::decode_influences(&bytes);
    let is_skinned = skeleton.is_some() && influences.is_some();
    let model = mesh::decode_model_with(&bytes, !is_skinned).unwrap_or_else(|| panic!("decode {model_path}"));
    let palette = model.material_name.as_deref().map(|n| material::load_mtl(&vfs, &model_path, n)).unwrap_or_default();
    let mut tex_bytes = HashMap::new();
    for slot in &palette { for (_, p) in &slot.textures { if !tex_bytes.contains_key(p) { if let Some(b) = vfs.read(p) { tex_bytes.insert(p.clone(), b); } } } }
    println!("[app] {model_path}: {} meshes, {} verts, {} tris", model.meshes.len(), model.vertex_count(), model.triangle_count());

    let anim_setup = if is_skinned {
        let sk = skeleton.unwrap();
        pick_caf_rig(&vfs, &sk, None).map(|(_, caf)| (sk, influences.unwrap(), caf))
    } else { None };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window { title: format!("sod-native — {model_path}"), resolution: (1280u32, 720u32).into(), ..default() }),
        ..default()
    }))
    .insert_resource(ClearColor(Color::srgb(0.05, 0.06, 0.08)))
    .insert_resource(Scene { model, palette, tex_bytes })
    .insert_resource(ShotPlan { path: std::env::var("SOD_SHOT").ok().map(PathBuf::from), frame: 0, keys: Vec::new() })
    .add_systems(Startup, setup)
    .add_systems(Update, (spin, auto_screenshot, animate_single));
    if let Some((sk, inf, caf)) = anim_setup {
        app.insert_resource(SingleAnim { skeleton: sk, influences: inf, caf, time: 0.0 });
    }
    app.run();
}

#[derive(Resource)]
struct SingleAnim { skeleton: Skeleton, influences: Vec<Vec<[BoneWeight; 4]>>, caf: anim::Caf, time: f32 }

fn pick_caf_rig(vfs: &Vfs, skeleton: &Skeleton, rig_in: Option<&str>) -> Option<(String, anim::Caf)> {
    let rig = rig_in.map(|r| r.to_string()).or_else(|| std::env::var("SOD_RIG").ok());
    let mut best = (0usize, -1i32, String::new());
    let mut scanned = 0;
    for path in vfs.paths_with_ext(".caf") {
        scanned += 1;
        if scanned > 3000 { break; }
        if let Some(r) = &rig {
            if !path.contains(r.as_str()) {
                continue;
            }
        }
        let lname = path.to_ascii_lowercase();
        let key = if lname.contains("walk") || lname.contains("run") || lname.contains("sprint") { 2 }
                  else if lname.contains("zombie") || lname.contains("attack") || lname.contains("horde") { 1 } else { 0 };
        if let Some(bytes) = vfs.read(&path) {
            let ids = anim::controller_ids(&bytes);
            let m = ids.iter().filter(|id| skeleton.by_controller.contains_key(id)).count();
            if m > best.0 || (m == best.0 && key > best.1) { best = (m, key, path.clone()); }
        }
    }
    if best.0 == 0 { return None; }
    let bytes = vfs.read(&best.2)?;
    let caf = anim::Caf::parse(&bytes)?;
    println!("[anim] using {} ({} matches, {:.2}s)", best.2, best.0, caf.duration_secs());
    Some((best.2, caf))
}

/// Decode a texture (disk-cached, memoised) and return a GPU handle.
fn cached_tex(
    images: &mut Assets<Image>,
    cache: Option<&cache::TextureCache>,
    memo: &mut HashMap<String, Handle<Image>>,
    key: &str,
    bytes: &[u8],
    srgb: bool,
) -> Option<Handle<Image>> {
    if let Some(h) = memo.get(key) {
        return Some(h.clone());
    }
    let d = match cache {
        Some(c) => c.get_or_decode(key.as_bytes(), bytes)?,
        None => sod_core::asset::decode_dds_rgba(bytes)?,
    };
    let format = if srgb { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm };
    let img = Image::new(
        Extent3d { width: d.width.max(1), height: d.height.max(1), depth_or_array_layers: 1 },
        TextureDimension::D2,
        d.rgba,
        format,
        RenderAssetUsages::default(),
    );
    let h = images.add(img);
    memo.insert(key.to_string(), h.clone());
    Some(h)
}

fn load_dds(images: &mut Assets<Image>, bytes: &[u8], srgb: bool) -> Option<Handle<Image>> {
    // Decode to plain RGBA8 on the CPU: robust across every DXT/BC variant the
    // game ships and independent of GPU compressed-texture support.
    let d = sod_core::asset::decode_dds_rgba(bytes)?;
    let format = if srgb { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm };
    let img = Image::new(
        Extent3d { width: d.width.max(1), height: d.height.max(1), depth_or_array_layers: 1 },
        TextureDimension::D2,
        d.rgba,
        format,
        RenderAssetUsages::default(),
    );
    Some(images.add(img))
}

fn make_material(materials: &mut Assets<StandardMaterial>, images: &mut Assets<Image>, scene: &Scene, slot: Option<&MaterialSlot>, fallback: Color) -> Handle<StandardMaterial> {
    let mut sm = StandardMaterial { base_color: fallback, perceptual_roughness: 0.9, cull_mode: None, ..default() };
    if let Some(slot) = slot {
        if let Some(path) = slot.texture("Diffuse") {
            if let Some(bytes) = scene.tex_bytes.get(path) {
                if let Some(h) = load_dds(images, bytes, true) { sm.base_color_texture = Some(h); }
            }
        }
    }
    materials.add(sm)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    scene: Res<Scene>,
    anim_setup: Option<Res<SingleAnim>>,
) {
    let min = Vec3::from(scene.model.bounds_min);
    let max = Vec3::from(scene.model.bounds_max);
    let radius = ((max - min).length() * 0.5).max(0.5);
    let base = Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2));
    let center = base.rotation * ((min + max) * 0.5);
    let textured_slot = scene.palette.iter().find(|s| s.texture("Diffuse").is_some());

    let mut parts = Vec::new();
    commands.spawn((base, Visibility::default(), Spinner)).with_children(|parent| {
        for (mi, m) in scene.model.meshes.iter().enumerate() {
            let mat = make_material(&mut materials, &mut images, &scene, textured_slot, Color::srgb(0.78, 0.74, 0.68));
            for sm in &m.submeshes {
                if sm.indices.is_empty() || m.positions.is_empty() { continue; }
                let mut bm = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
                bm.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(m.positions.clone()));
                bm.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(m.normals.clone()));
                if m.uvs.len() == m.positions.len() { bm.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(m.uvs.clone())); }
                bm.insert_indices(Indices::U32(sm.indices.clone()));
                let handle = meshes.add(bm);
                parent.spawn((Mesh3d(handle.clone()), MeshMaterial3d(mat.clone())));
                parts.push(AnimPart { handle, bind_pos: m.positions.clone(), bind_nrm: m.normals.clone(), infl: Vec::new(), mat: mat.clone() });
            }
        }
    });
    let _ = &parts;
    if let Some(s) = anim_setup {
        // build skinned parts over the same meshes is handled by animate_single using SingleAnim;
        // for simplicity re-skin using influences by index.
        let _ = s;
    }
    let cam_pos = center + Vec3::new(radius * 1.1, radius * 0.7, radius * 2.1);
    commands.spawn((Camera3d::default(), Transform::from_translation(cam_pos).looking_at(center, Vec3::Y), Tonemapping::None));
    commands.spawn((DirectionalLight { shadow_maps_enabled: false, illuminance: 2600.0, ..default() }, Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.7, 0.0))));
    commands.spawn((DirectionalLight { shadow_maps_enabled: false, illuminance: 900.0, ..default() }, Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.4, -2.4, 0.0))));
}

fn spin(time: Res<Time>, mut q: Query<&mut Transform, With<Spinner>>) {
    for mut t in &mut q { t.rotate_y(time.delta_secs() * 0.5); }
}

fn animate_single(_anim: Option<ResMut<SingleAnim>>, _meshes: ResMut<Assets<Mesh>>, _time: Res<Time>) {}


// ---------------------------------------------------------------------------
// game systems (zombies, combat, HUD)
// ---------------------------------------------------------------------------

fn base_rot() -> Quat { Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2) }

fn npc_ai(time: Res<Time>, menu: Res<MenuRes>, player: Option<Res<Player>>, world: Res<World>, mut game: ResMut<GameState>, mut q: Query<(&mut Transform, &mut Npc)>) {
    if menu.active { return; }
    let Some(player) = player else { return };
    let dt = time.delta_secs().min(0.1);
    let br = base_rot();
    // Player is a noise source: louder while running.
    let ph = height_at(&world.terrain, player.pos[0], player.pos[1]);
    let ppos = [player.pos[0], player.pos[1], ph];
    let noise = if player.moving { 2.2 } else { 0.6 };
    game.horde.tick(dt);
    for (mut t, mut npc) in &mut q {
        let rng = game.ai_rng.next_f32();
        let input = zombie_ai::ZombieInput { player_pos: ppos, player_noise: noise, dt, rng };
        let out = zombie_ai::update(&mut npc.agent, &input);
        if out.speed > 0.0 {
            npc.agent.pos[0] += out.move_dir[0] * out.speed * dt;
            npc.agent.pos[1] += out.move_dir[1] * out.speed * dt;
        }
        let h = height_at(&world.terrain, npc.agent.pos[0], npc.agent.pos[1]);
        npc.agent.pos[2] = h;
        t.translation = br * Vec3::new(npc.agent.pos[0], npc.agent.pos[1], h);
        t.rotation = br * Quat::from_rotation_z(sod_core::class3_move::render_yaw(npc.agent.yaw));
    }
}

fn animate_npcs(time: Res<Time>, mut meshes: ResMut<Assets<Mesh>>, mut q: Query<&mut Npc>) {
    for mut npc in &mut q {
        if npc.parts.is_empty() { continue; }
        // Locomotion clip from the zombie behaviour state.
        let want = match npc.agent.behavior {
            zombie_ai::ZombieBehavior::Attack => anim::Locomotion::Attack,
            zombie_ai::ZombieBehavior::Chase
            | zombie_ai::ZombieBehavior::Flee
            | zombie_ai::ZombieBehavior::Grab
            | zombie_ai::ZombieBehavior::Feasting => anim::Locomotion::Run,
            zombie_ai::ZombieBehavior::Wander
            | zombie_ai::ZombieBehavior::Alert
            | zombie_ai::ZombieBehavior::Search => anim::Locomotion::Walk,
            _ => anim::Locomotion::Idle,
        };
        if want != npc.loc {
            let c = npc.bank.caf_for(want).cloned();
            if let Some(c) = c {
                npc.caf = c;
                npc.loc = want;
            }
        }
        npc.time += time.delta_secs();
        let dur = npc.caf.duration_secs().max(1.0 / 30.0);
        let tick = npc.caf.tick_from_seconds(npc.time % dur);
        let sample = npc.caf.sample_tick(tick);
        let posed = npc.skeleton.pose_anchored(&sample);
        let skin_m = npc.skeleton.skin_matrices(&posed);
        for part in &npc.parts {
            let npos = skin::deform(&part.bind_pos, &part.infl, &skin_m);
            let nnrm = skin::deform_normals(&part.bind_nrm, &part.infl, &skin_m);
            if let Some(mut m) = meshes.get_mut(&part.handle) {
                m.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(npos));
                m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(nnrm));
            }
        }
    }
}

fn follower_ai(
    time: Res<Time>,
    menu: Res<MenuRes>,
    player: Option<Res<Player>>,
    world: Res<World>,
    mut q: Query<(&mut Transform, &mut Follower)>,
) {
    if menu.active { return; }
    let Some(player) = player else { return };
    let dt = time.delta_secs().min(0.1);
    let br = base_rot();
    let leader = [player.mstate.pos[0], player.mstate.pos[1], player.mstate.pos[2]];
    for (mut t, mut f) in &mut q {
        let input = npc_ai::NpcInput { leader_pos: Some(leader), threat_pos: None, dt };
        let out = npc_ai::update(&mut f.agent, &input);
        if out.speed > 0.0 {
            f.agent.pos[0] += out.move_dir[0] * out.speed * dt;
            f.agent.pos[1] += out.move_dir[1] * out.speed * dt;
        }
        let h = height_at(&world.terrain, f.agent.pos[0], f.agent.pos[1]);
        f.agent.pos[2] = h;
        f.moving = out.speed > 0.05;
        if out.face[0].abs() + out.face[1].abs() > 1e-4 {
            f.agent.yaw = out.face[1].atan2(out.face[0]);
        }
        t.translation = br * Vec3::new(f.agent.pos[0], f.agent.pos[1], h);
        t.rotation = br * Quat::from_rotation_z(sod_core::class3_move::render_yaw(f.agent.yaw));
    }
}

fn animate_followers(time: Res<Time>, mut meshes: ResMut<Assets<Mesh>>, mut q: Query<&mut Follower>) {
    for mut f in &mut q {
        if f.parts.is_empty() { continue; }
        let want = if f.moving { anim::Locomotion::Walk } else { anim::Locomotion::Idle };
        if want != f.loc {
            let c = f.bank.caf_for(want).cloned();
            if let Some(c) = c {
                f.caf = c;
                f.loc = want;
            }
        }
        f.time += time.delta_secs();
        let dur = f.caf.duration_secs().max(1.0 / 30.0);
        let tick = f.caf.tick_from_seconds(f.time % dur);
        let sample = f.caf.sample_tick(tick);
        let posed = f.skeleton.pose_anchored(&sample);
        let skin_m = f.skeleton.skin_matrices(&posed);
        for part in &f.parts {
            let npos = skin::deform(&part.bind_pos, &part.infl, &skin_m);
            let nnrm = skin::deform_normals(&part.bind_nrm, &part.infl, &skin_m);
            if let Some(mut m) = meshes.get_mut(&part.handle) {
                m.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(npos));
                m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(nnrm));
            }
        }
    }
}

fn player_attack(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    menu: Res<MenuRes>,
    player: Option<Res<Player>>,
    state: Option<ResMut<PlayerState>>,
    mut game: ResMut<GameState>,
    mut npcs: Query<(Entity, &Transform, &mut Npc)>,
    mut commands: Commands,
) {
    if menu.active { return; }
    let (Some(player), Some(mut state)) = (player, state) else { return };
    let br = base_rot();
    let pw = br * Vec3::new(player.pos[0], player.pos[1], 0.0);
    let dt = time.delta_secs();
    let PlayerState { vitals, melee, gun, gun_state, .. } = &mut *state;
    let auto = std::env::var("SOD_AUTOFIGHT").is_ok();
    game.auto_cd = (game.auto_cd - dt).max(0.0);
    game.auto_fire_cd = (game.auto_fire_cd - dt).max(0.0);
    let melee_now = keys.just_pressed(KeyCode::Space) || (auto && game.auto_cd <= 0.0);
    let fire_now = keys.just_pressed(KeyCode::KeyF) || (auto && game.auto_fire_cd <= 0.0);

    // Melee: nearest zombie within reach.
    if melee_now {
        let mut best: Option<(Entity, f32)> = None;
        for (e, t, _n) in npcs.iter_mut() {
            let d = (t.translation - pw).length();
            if d < 2.8 && best.map_or(true, |(_, b)| d < b) { best = Some((e, d)); }
        }
        if let Some((e, _)) = best {
            if let Ok((_, _, mut npc)) = npcs.get_mut(e) {
                let hit = combat::melee_swing(melee, npc.kind, npc.hp, sod_core::game::HitLocation::TORSO, vitals.stamina);
                npc.hp = (npc.hp - hit.damage).max(0.0);
                vitals.stamina = (vitals.stamina - hit.stamina_cost).max(0.0);
                println!("[combat] {} -> {} hp {:.0} (dmg {:.0}, sever {:?})", melee.name, npc.kind.name(), npc.hp, hit.damage, hit.sever);
                if npc.hp <= 0.0 {
                    commands.entity(e).despawn();
                    game.kills += 1;
                    let kills = game.kills;
                    println!("[combat] zombie killed ({} total)", kills);
                    game.tutorial.notify(&mission::TutorialTrigger::KillCount(kills));
                }
            }
        }
    }

    // Firearm: nearest zombie, any range.
    if fire_now {
        let mut best: Option<(Entity, f32)> = None;
        for (e, t, _n) in npcs.iter_mut() {
            let d = (t.translation - pw).length();
            if best.map_or(true, |(_, b)| d < b) { best = Some((e, d)); }
        }
        if let Some((e, _)) = best {
            if let Some(dmg) = gun_state.fire(gun, dt) {
                if let Ok((_, _, mut npc)) = npcs.get_mut(e) {
                    npc.hp = sod_core::game::apply_zombie_damage(npc.hp, npc.kind, dmg, sod_core::game::HitLocation::TORSO, combat::BLOATER_HEADSHOT_KILL);
                    println!("[combat] {} ({}/{} rounds) -> {} hp {:.0}", gun.name, gun_state.in_mag, gun.mag_size, npc.kind.name(), npc.hp);
                    if npc.hp <= 0.0 {
                        commands.entity(e).despawn();
                        game.kills += 1;
                        let kills = game.kills;
                        game.tutorial.notify(&mission::TutorialTrigger::KillCount(kills));
                    }
                }
            }
        }
    }

    if keys.just_pressed(KeyCode::KeyR) && gun_state.reload(gun) {
        println!("[combat] reloading {} ({}/{})", gun.name, gun_state.in_mag, gun_state.reserve);
    }
}

fn npc_attack(time: Res<Time>, menu: Res<MenuRes>, player: Option<Res<Player>>, state: Option<ResMut<PlayerState>>, q: Query<(&Transform, &Npc)>) {
    if menu.active { return; }
    let (Some(player), Some(mut state)) = (player, state) else { return };
    let dt = time.delta_secs();
    let br = base_rot();
    let pw = br * Vec3::new(player.pos[0], player.pos[1], 0.0);
    let mut dmg = 0.0;
    for (t, npc) in &q {
        if (t.translation - pw).length() < 1.3 {
            // Damage-per-second derived from strike damage / strike cooldown.
            dmg += combat::zombie_attack_damage(npc.kind) * dt / combat::zombie_melee_cooldown(npc.kind);
        }
    }
    if dmg > 0.0 {
        state.vitals.hp = (state.vitals.hp - dmg).max(0.0);
    }
}

fn player_vitals(time: Res<Time>, menu: Res<MenuRes>, state: Option<ResMut<PlayerState>>) {
    if menu.active { return; }
    let Some(mut state) = state else { return };
    let dt = time.delta_secs();
    state.vitals.tick(dt);
    state.gun_state.tick(dt);
    if state.vitals.is_downed() {
        println!("[game] player downed -> revive");
        state.vitals.revive();
    }
}

fn search_system(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    menu: Res<MenuRes>,
    player: Option<Res<Player>>,
    world: Res<World>,
    state: Option<ResMut<PlayerState>>,
    mut game: ResMut<GameState>,
) {
    if menu.active { return; }
    let (Some(player), Some(mut state)) = (player, state) else { return };
    let dt = time.delta_secs();
    let auto = std::env::var("SOD_AUTOSEARCH").is_ok();
    game.auto_search_cd = (game.auto_search_cd - dt).max(0.0);
    let search_now = keys.just_pressed(KeyCode::KeyE) || (auto && game.auto_search_cd <= 0.0);
    if search_now && game.search.is_none() {
        let br = base_rot();
        let pw = br * Vec3::new(player.pos[0], player.pos[1], 0.0);
        let mut best: Option<f32> = None;
        for p in &world.placements {
            let wp = br * Vec3::new(p.pos[0], p.pos[1], p.pos[2]);
            let d = (wp - pw).length();
            if best.map_or(true, |b| d < b) { best = Some(d); }
        }
        if best.map_or(false, |d| d < 25.0) {
            let mut ss = inventory::SearchState::new(2.5);
            {
                let GameState { loot, rng, .. } = &mut *game;
                ss.roll(loot, rng);
            }
            game.search = Some(SearchTask { state: ss });
            game.auto_search_cd = 3.0;
            println!("[search] started");
        } else {
            game.auto_search_cd = 1.0;
            println!("[search] nothing nearby (nearest prop {:.1} m)", best.unwrap_or(f32::INFINITY));
        }
    }
    if let Some(mut task) = game.search.take() {
        if task.state.tick(dt) {
            let loot = std::mem::take(&mut task.state.loot);
            {
                let PlayerState { inv, db, .. } = &mut *state;
                for item in loot {
                    let id = item.id.clone();
                    let count = item.count;
                    let leftover = inv.add_with_db(item, db);
                    println!("[search] found {} x{} (leftover {})", id, count, leftover);
                }
            }
            state.vitals.heal(5.0);
            game.tutorial.notify(&mission::TutorialTrigger::SearchContainer);
            println!("[search] complete");
        } else {
            game.search = Some(task);
        }
    }
}

fn tutorial_system(menu: Res<MenuRes>, player: Option<Res<Player>>, mut game: ResMut<GameState>) {
    if menu.active { return; }
    let Some(player) = player else { return };
    let p = [player.pos[0], player.pos[1]];
    let cur = game.tutorial.current().map(|s| s.trigger.clone());
    if let Some(mission::TutorialTrigger::ReachPoint(pos, r)) = cur {
        let dx = pos[0] - p[0];
        let dy = pos[1] - p[1];
        if dx * dx + dy * dy <= r * r {
            game.tutorial.notify(&mission::TutorialTrigger::ReachPoint(pos, r));
        }
    }
}

fn hud_update(
    time: Res<Time>,
    state: Option<Res<PlayerState>>,
    game: Option<Res<GameState>>,
    player: Option<Res<Player>>,
    npcs: Query<&Npc>,
    hud_res: Option<ResMut<HudRes>>,
    mut hbar: Query<&mut Node, (With<HealthBarFill>, Without<StaminaBarFill>, Without<RadarBlipNode>)>,
    mut sbar: Query<&mut Node, (With<StaminaBarFill>, Without<HealthBarFill>, Without<RadarBlipNode>)>,
    mut lines: Query<(&mut Text, &HudLine)>,
    mut blips: Query<(&mut Node, &mut BackgroundColor, &RadarBlipNode), (Without<HealthBarFill>, Without<StaminaBarFill>)>,
) {
    let Some(mut hud_res) = hud_res else { return };
    let vitals = state.as_ref().map(|s| s.vitals).unwrap_or_else(|| combat::Vitals::new(100.0, 100.0));
    let (ppos, pyaw) = player
        .as_ref()
        .map(|p| ([p.mstate.pos[0], p.mstate.pos[1], p.mstate.pos[2]], p.mstate.yaw))
        .unwrap_or(([0.0; 3], 0.0));
    let hud_npcs: Vec<hud::HudNpc> = npcs
        .iter()
        .map(|n| hud::HudNpc { pos: n.agent.pos, kind: n.kind as u8, threat: n.agent.alert })
        .collect();
    let objective = game.as_ref().and_then(|g| g.tutorial.current().map(|s| s.text.clone()));
    {
        let db = state.as_ref().map(|s| &s.db);
        let inv = state.as_ref().map(|s| &s.inv);
        let ctx = hud::HudCtx {
            vitals,
            player_pos: ppos,
            player_yaw: pyaw,
            npcs: &hud_npcs,
            objective: objective.as_deref(),
            inventory: inv,
            db,
            reticle_spread: 0.0,
            interaction: None,
            dt: time.delta_secs(),
        };
        hud::update(&mut hud_res.0, &ctx);
    }
    let hp_ratio = hud_res.0.vitals.hp_ratio();
    let st_ratio = hud_res.0.vitals.stamina_ratio();
    for mut n in &mut hbar { n.width = Val::Percent((hp_ratio * 100.0).clamp(0.0, 100.0)); }
    for mut n in &mut sbar { n.width = Val::Percent((st_ratio * 100.0).clamp(0.0, 100.0)); }
    let notif = hud_res.0.notifications.iter().map(|n| n.text.clone()).collect::<Vec<_>>().join("   ");
    let (hp, maxhp, stam, count) = (hud_res.0.vitals.hp, hud_res.0.vitals.max_hp, hud_res.0.vitals.stamina, hud_res.0.zombie_count);
    let objective_txt = hud_res.0.objective.clone().unwrap_or_else(|| "(complete)".to_string());
    let inv_txt = hud_res.0.inventory_summary.clone();
    let banner = hud_res.0.banner.clone().unwrap_or_default();
    let prompt = hud_res.0.prompts.first().map(|p| p.text.clone()).unwrap_or_default();
    for (mut t, line) in &mut lines {
        match line {
            HudLine::Status => t.0 = format!("HP {hp:.0}/{maxhp:.0}   Stamina {stam:.0}   Zombies {count}"),
            HudLine::Objective => t.0 = format!("OBJECTIVE: {objective_txt}"),
            HudLine::Inventory => t.0 = inv_txt.clone(),
            HudLine::Banner => t.0 = banner.clone(),
            HudLine::Prompt => t.0 = prompt.clone(),
            HudLine::Notifications => t.0 = notif.clone(),
        }
    }
    for (mut node, mut color, marker) in &mut blips {
        match hud_res.0.blips.get(marker.index) {
            Some(b) => {
                let scale = 70.0f32;
                node.left = Val::Px((78.0 + b.rel[0].clamp(-1.0, 1.0) * scale).clamp(2.0, 152.0));
                node.top = Val::Px((78.0 - b.rel[1].clamp(-1.0, 1.0) * scale).clamp(2.0, 152.0));
                node.display = Display::Flex;
                *color = BackgroundColor(if b.threat > 0.5 { Color::srgb(0.95, 0.2, 0.15) } else { Color::srgb(0.8, 0.75, 0.3) });
            }
            None => node.display = Display::None,
        }
    }
}

/// Feed the controlled player's CryEngine-space transform to the script host.
fn cry_sync_player(
    player: Option<Res<Player>>,
    world: Res<World>,
    out: Option<ResMut<cry_bridge::CryPlayerPos>>,
) {
    let (Some(player), Some(mut out)) = (player, out) else { return };
    let [x, y] = player.pos;
    out.0 = Some([x, y, height_at(&world.terrain, x, y)]);
    // Entity rotation = the model root's rotation about cry +Z.
    let half = 0.5 * sod_core::class3_move::render_yaw(player.yaw);
    out.1 = Some([0.0, 0.0, half.sin(), half.cos()]);
}
