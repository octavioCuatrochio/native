//! App-side bridge for the native vehicle layer.
//!
//! The driving model lives in `sod_core::cry::vehicle_sim` (a port of Class3's
//! `ArcadeWheeled` movement, registered at `0x01552970`, factory `0x015527c0`,
//! vtable `0x018a6668`); this module is the engine side that the original game
//! scripts expect:
//!
//! 1. It reads the compiled `ULVS` spawn list (`vehicle_spawns_win.markup`,
//!    produced by `NSVehicleManager::VehicleManager` `0x0140b9a0` from the
//!    `ULVehicleSpawnEditor` level entities) and spawns every initial vehicle
//!    with its real per-class model/material, through the same
//!    `mesh::decode_model_render` + `material::load_mtl` path static props use.
//! 2. It enters/exits through the original seat helpers parsed from the class
//!    XML (`driverEnterPos`/`driverExitPos`/`driverSitPos`), the same points
//!    `Scripts/Entities/Vehicles` seat actions use.
//! 3. It drives with the existing input bindings the game uses (WASD + Space
//!    hand-brake, `E` to enter/exit) and the original per-class vehicle camera
//!    (`<View class="ActionThirdPerson">` / `<ThirdPerson>` params).
//!
//! It is always on in play mode; there is no demo flag. `SOD_AUTOSTART` (used
//! by the screenshot harness) additionally scripts an enter-and-drive so the
//! verification screenshots show a spawned and a moving car.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use std::collections::HashMap;

use sod_core::cry::vehicle::VehicleSpawnFile;
use sod_core::cry::vehicle_sim::{VehicleBody, VehicleInput};
use sod_core::cry::vehicle_xml::{parse_vehicle_xml, VehicleSetup};
use sod_core::crypak::Vfs;
use sod_core::material::{self, MaterialSlot};
use sod_core::mesh::{self, DecodedModel};

use crate::cry_bridge::CryHost;

/// Register the bridge (inside `run_world`'s play branch, after the script host
/// exists so the mounts are readable).
pub struct VehicleBridgePlugin;

impl Plugin for VehicleBridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VehicleWorld>()
            .add_systems(Startup, vehicle_spawn.after(crate::setup_world))
            .add_systems(
                Update,
                vehicle_control
                    .after(crate::move_player)
                    .before(crate::follow_camera),
            )
            .add_systems(Update, vehicle_camera.after(crate::follow_camera));
    }
}

/// A vehicle class: the parsed XML (physics, helpers, seats) plus its render
/// geometry and the recovered camera view offsets.
struct VehicleClass {
    setup: VehicleSetup,
    parts: Vec<(Handle<Mesh>, Handle<StandardMaterial>)>,
    view: VehicleView,
}

/// `<Seats><Seat><Views>` camera params, recovered from the class XML.
#[derive(Debug, Clone, Copy, Default)]
struct VehicleView {
    /// `ActionThirdPerson cameraPosOffset` (body space `[right, forward, up]`).
    pos_offset: [f32; 3],
    /// `ActionThirdPerson cameraAimOffset`.
    aim_offset: [f32; 3],
    /// `ThirdPerson distance` (how far behind the car the camera sits).
    distance: f32,
    /// `ThirdPerson heightOffset`.
    height_offset: f32,
}

/// One spawned vehicle.
struct Car {
    body: VehicleBody,
    root: Entity,
    /// Lower-cased class name, keyed into [`VehicleWorld::classes`].
    class: String,
}

/// All live vehicles plus their loaded class assets.
#[derive(Resource, Default)]
struct VehicleWorld {
    classes: HashMap<String, VehicleClass>,
    cars: Vec<Car>,
    driving: Option<usize>,
    frame: u32,
    tex_memo: HashMap<String, Handle<Image>>,
    cam_pos: Vec3,
    cam_init: bool,
}

/// Marks a spawned vehicle root (the transform the physics drives).
#[derive(Component)]
struct VehicleRoot;

/// Cap the number of initial ULVS spawns processed so a full level does not
/// stall the frame while keeping the shipped placement list authoritative.
const MAX_SPAWN: usize = 160;

// ---------------------------------------------------------------------------
// Spawn
// ---------------------------------------------------------------------------

fn vehicle_spawn(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    cache: Option<Res<crate::TextureCacheRes>>,
    cry: Option<NonSend<CryHost>>,
    mut vw: ResMut<VehicleWorld>,
) {
    let Some(cry) = cry else { return };
    let vfs = cry.host.ctx.vfs.clone();
    // Prefer the Class3 campaign placement; the DLC packs ship their own.
    let mut cands = vfs.paths_with_ext("vehicle_spawns_win.markup");
    cands.sort_by_key(|p| !p.to_ascii_lowercase().contains("class3"));
    let parsed = cands
        .iter()
        .find_map(|p| vfs.read(p))
        .and_then(|b| VehicleSpawnFile::parse(&b));
    let Some(spawns) = parsed else {
        eprintln!("[vehicle] no ULVS spawn list in the mounted paks");
        return;
    };
    println!(
        "[vehicle] ULVS parse: {} initial / {} total spawns",
        spawns.initial_spawn_count, spawns.total_spawn_count
    );

    let cache = cache.as_ref().map(|c| &c.0);
    let base = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let mut loaded: HashMap<String, VehicleClass> = HashMap::new();
    let mut spawned = 0usize;
    for sp in spawns.initial().iter().take(MAX_SPAWN) {
        let key = sp.class_name.to_ascii_lowercase();
        if !loaded.contains_key(&key) {
            match load_class(&vfs, &key, &mut meshes, &mut materials, &mut images, cache, &mut vw.tex_memo) {
                Some(c) => {
                    loaded.insert(key.clone(), c);
                }
                None => {
                    eprintln!("[vehicle] no usable class XML for {}", sp.class_name);
                    continue;
                }
            }
        }
        let class = &loaded[&key];
        // The model's local forward is +Y (body space), so the physics yaw
        // (+X at 0) is the ULVS euler yaw rotated a quarter turn.
        let yaw = sp.euler[2] + std::f32::consts::FRAC_PI_2;
        let body = VehicleBody::new(class.setup.clone(), sp.pos, yaw);
        let world_pos = base * Vec3::new(sp.pos[0], sp.pos[1], sp.pos[2]);
        let rot = base * Quat::from_rotation_z(yaw - std::f32::consts::FRAC_PI_2);
        let root = commands
            .spawn((
                Transform { translation: world_pos, rotation: rot, scale: Vec3::ONE },
                Visibility::default(),
                VehicleRoot,
            ))
            .with_children(|parent| {
                for (mh, mat) in &class.parts {
                    parent.spawn((Mesh3d(mh.clone()), MeshMaterial3d(mat.clone())));
                }
            })
            .id();
        vw.cars.push(Car { body, root, class: key.clone() });
        spawned += 1;
    }
    for (k, v) in loaded {
        vw.classes.insert(k, v);
    }
    println!("[vehicle] spawned {spawned} vehicles across {} classes", vw.classes.len());
}

/// Locate and parse the per-class XML, decode its animated model and build the
/// render parts (the same pipeline `setup_world` uses for static props).
fn load_class(
    vfs: &Vfs,
    key: &str,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    cache: Option<&sod_core::cache::TextureCache>,
    memo: &mut HashMap<String, Handle<Image>>,
) -> Option<VehicleClass> {
    let xml = vfs.read(&format!("scripts/entities/vehicles/implementations/xml/{key}.xml"))?;
    let setup = parse_vehicle_xml(&xml);
    let model = extract_animated_filename(&xml)?;
    let bytes = vfs.read(&model)?;
    let dm = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mesh::decode_model_render(&bytes)
    })) {
        Ok(Some(dm)) => dm,
        _ => return None,
    };
    let slots = dm
        .material_name
        .as_deref()
        .map(|n| material::load_mtl(vfs, &model, n))
        .unwrap_or_default();
    let mut tex_bytes: HashMap<String, Vec<u8>> = HashMap::new();
    for slot in &slots {
        for (_, tp) in &slot.textures {
            if !tex_bytes.contains_key(tp) {
                if let Some(b) = vfs.read(tp) {
                    tex_bytes.insert(tp.clone(), b);
                }
            }
        }
    }
    let parts = build_parts(&dm, &slots, &tex_bytes, meshes, materials, images, cache, memo);
    let view = parse_view(&xml);
    Some(VehicleClass { setup, parts, view })
}

fn default_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(0.78, 0.76, 0.72),
        perceptual_roughness: 0.9,
        cull_mode: None,
        ..default()
    }
}

fn build_parts(
    dm: &DecodedModel,
    slots: &[MaterialSlot],
    tex: &HashMap<String, Vec<u8>>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    cache: Option<&sod_core::cache::TextureCache>,
    memo: &mut HashMap<String, Handle<Image>>,
) -> Vec<(Handle<Mesh>, Handle<StandardMaterial>)> {
    use bevy::material::AlphaMode;
    let mut mat_handles: Vec<Handle<StandardMaterial>> = Vec::new();
    let mut nodraw: Vec<bool> = Vec::new();
    let mut has_normal = false;
    for slot in slots {
        if slot.is_nodraw() {
            mat_handles.push(materials.add(default_material()));
            nodraw.push(true);
            continue;
        }
        let alpha_mode = AlphaMode::Mask(slot.alpha_test.unwrap_or(0.5));
        let mut mat = StandardMaterial {
            base_color: Color::srgb(0.78, 0.76, 0.72),
            perceptual_roughness: 0.9,
            cull_mode: None,
            alpha_mode,
            ..default()
        };
        if let Some(tp) = slot.texture("Diffuse") {
            if let Some(b) = tex.get(tp) {
                if let Some(h) = crate::cached_tex(images, cache, memo, tp, b, true) {
                    mat.base_color_texture = Some(h);
                }
            }
        }
        if let Some(tp) = slot.texture("Bumpmap").or_else(|| slot.texture("Normal")) {
            if let Some(b) = tex.get(tp) {
                let key = format!("{tp}#normal");
                if let Some(h) = crate::cached_tex(images, cache, memo, &key, b, false) {
                    mat.normal_map_texture = Some(h);
                    has_normal = true;
                }
            }
        }
        mat_handles.push(materials.add(mat));
        nodraw.push(false);
    }
    if mat_handles.is_empty() {
        mat_handles.push(materials.add(default_material()));
        nodraw.push(false);
    }

    let mut out = Vec::new();
    for m in &dm.meshes {
        if m.positions.is_empty() {
            continue;
        }
        let build = |indices: &[u32]| {
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
            bm.insert_indices(Indices::U32(indices.to_vec()));
            if has_normal && m.uvs.len() == m.positions.len() {
                let _ = bm.generate_tangents();
            }
            bm
        };
        if m.submeshes.is_empty() {
            if m.indices.is_empty() || nodraw.first().copied().unwrap_or(false) {
                continue;
            }
            out.push((meshes.add(build(&m.indices)), mat_handles[0].clone()));
        } else {
            for sm in &m.submeshes {
                if sm.indices.is_empty() {
                    continue;
                }
                let mi = sm.material as usize;
                if nodraw.get(mi).copied().unwrap_or(false) {
                    continue;
                }
                let mat = mat_handles.get(mi).cloned().unwrap_or_else(|| mat_handles[0].clone());
                out.push((meshes.add(build(&sm.indices)), mat));
            }
        }
    }
    out
}

/// Pull the LOD0 animated model path (`<Animated filename="...">`) out of a
/// class XML.
fn extract_animated_filename(xml: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(xml);
    for (tag, attr) in [("<Animated", "filename="), ("<Animated", "filenameDestroyed=")] {
        let Some(t) = text.find(tag) else { continue };
        let rest = &text[t..];
        let Some(a) = rest.find(attr) else { continue };
        let after = &rest[a + attr.len()..];
        let after = after.trim_start_matches([' ', '=']);
        let after = after.trim_start_matches('"');
        let Some(end) = after.find('"') else { continue };
        let path = &after[..end];
        if path.to_ascii_lowercase().ends_with(".cga") || path.to_ascii_lowercase().ends_with(".cgf") {
            return Some(path.replace('\\', "/").to_ascii_lowercase());
        }
    }
    None
}

/// Recover the `ActionThirdPerson`/`ThirdPerson` view params from the class
/// XML (the `<Seats><Seat><Views>` block).
fn parse_view(xml: &[u8]) -> VehicleView {
    let text = String::from_utf8_lossy(xml);
    let mut view = VehicleView::default();
    let Some(start) = text.find("<Views>") else { return view };
    let views = &text[start..];
    if let Some(pos) = views.find("cameraPosOffset") {
        view.pos_offset = parse_vec3_attr(&views[pos..]);
    }
    if let Some(pos) = views.find("cameraAimOffset") {
        view.aim_offset = parse_vec3_attr(&views[pos..]);
    }
    if let Some(pos) = views.find("distance=") {
        view.distance = parse_scalar_attr(&views[pos..]);
    }
    if let Some(pos) = views.find("heightOffset") {
        view.height_offset = parse_scalar_attr(&views[pos..]);
    }
    view
}

fn parse_vec3_attr(rest: &str) -> [f32; 3] {
    let after = rest.split('"').nth(1).unwrap_or("");
    let p: Vec<f32> = after
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    [p.first().copied().unwrap_or(0.0), p.get(1).copied().unwrap_or(0.0), p.get(2).copied().unwrap_or(0.0)]
}

fn parse_scalar_attr(rest: &str) -> f32 {
    rest.split('"')
        .nth(1)
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Control (enter / exit / drive)
// ---------------------------------------------------------------------------

fn vehicle_control(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    collision: Option<Res<crate::Collision>>,
    world: Option<Res<crate::World>>,
    mut player: Option<ResMut<crate::Player>>,
    mut roots: Query<&mut Transform, (With<VehicleRoot>, Without<crate::PlayerRoot>)>,
    mut vis: Query<&mut Visibility, With<crate::PlayerRoot>>,
    mut vw: ResMut<VehicleWorld>,
) {
    vw.frame = vw.frame.wrapping_add(1);
    let auto = std::env::var("SOD_AUTOSTART").is_ok() || std::env::var("SOD_AUTODRIVE").is_ok();
    let dt = time.delta_secs().clamp(0.0, 1.0 / 20.0);

    let player_xy = player
        .as_ref()
        .map(|p| [p.mstate.pos[0], p.mstate.pos[1]])
        .unwrap_or([0.0, 0.0]);

    // --- enter / exit -----------------------------------------------------
    if auto {
        // Scripted harness: at frame 80 walk to the nearest car and get in.
        if vw.frame == 80 && vw.driving.is_none() {
            if let Some(i) = nearest_car(&vw, player_xy, f32::INFINITY) {
                enter_car(&mut vw, i, &mut player, &mut vis);
            }
        }
    } else if keys.just_pressed(KeyCode::KeyE) {
        if let Some(i) = vw.driving {
            exit_car(&mut vw, i, &mut player, &mut vis);
        } else if let Some(i) = nearest_car(&vw, player_xy, 5.0) {
            enter_car(&mut vw, i, &mut player, &mut vis);
        }
    }

    // --- input ------------------------------------------------------------
    let drive_input = if vw.driving.is_some() {
        if auto {
            let f = vw.frame;
            let throttle = if f > 140 { 1.0 } else { 0.0 };
            let steer = if f > 210 { (f as f32 * 0.02).sin() * 0.6 } else { 0.0 };
            VehicleInput { throttle, steer, brake: 0.0, handbrake: f > 560 }
        } else {
            let throttle = if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
                1.0
            } else if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
                -1.0
            } else {
                0.0
            };
            let steer = if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
                1.0
            } else if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
                -1.0
            } else {
                0.0
            };
            VehicleInput {
                throttle,
                steer,
                brake: 0.0,
                handbrake: keys.pressed(KeyCode::Space),
            }
        }
    } else {
        VehicleInput::default()
    };

    // --- step physics -----------------------------------------------------
    let driving = vw.driving;
    let mut stepped = 0usize;
    for (i, car) in vw.cars.iter_mut().enumerate() {
        let near = ((car.body.state.pos[0] - player_xy[0]).powi(2)
            + (car.body.state.pos[1] - player_xy[1]).powi(2))
        .sqrt()
            < 180.0;
        if Some(i) != driving && !near {
            continue;
        }
        let inp = if Some(i) == driving { drive_input } else { VehicleInput::default() };
        if let Some(c) = &collision {
            car.body.step(&inp, &c.0, dt);
        } else if let Some(w) = &world {
            car.body.step(&inp, &w.terrain, dt);
        } else {
            continue;
        }
        stepped += 1;
    }
    let _ = stepped;

    // --- write transforms -------------------------------------------------
    let base = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    for car in vw.cars.iter() {
        let Ok(mut tf) = roots.get_mut(car.root) else { continue };
        let s = &car.body.state;
        let yaw_q = base * Quat::from_rotation_z(s.yaw - std::f32::consts::FRAC_PI_2);
        let tilt = Quat::from_rotation_x(s.pitch) * Quat::from_rotation_y(s.roll);
        tf.translation = base * Vec3::new(s.pos[0], s.pos[1], s.pos[2]);
        tf.rotation = yaw_q * tilt;
    }

    // --- seat the player in the driven car --------------------------------
    if let Some(d) = vw.driving {
        let (class_key, s) = {
            let car = &vw.cars[d];
            (car.class.clone(), car.body.state)
        };
        let seat = vw
            .classes
            .get(&class_key)
            .and_then(|c| c.setup.helpers.get("driverSitPos").copied())
            .unwrap_or([0.0, 0.0, 0.5]);
        let cy = s.yaw.cos();
        let sy = s.yaw.sin();
        let fwd = [cy, sy];
        let rt = [sy, -cy];
        let wx = s.pos[0] + rt[0] * seat[0] + fwd[0] * seat[1];
        let wy = s.pos[1] + rt[1] * seat[0] + fwd[1] * seat[1];
        let wz = s.pos[2] + seat[2];
        if let Some(p) = player.as_deref_mut() {
            p.mstate.pos = [wx, wy, wz];
            p.pos = [wx, wy];
            p.mstate.vel = [0.0, 0.0, 0.0];
            p.mstate.grounded = true;
            p.mstate.yaw = s.yaw;
            p.yaw = s.yaw;
            p.moving = false;
        }
    }
}

fn nearest_car(vw: &VehicleWorld, xy: [f32; 2], max_dist: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, car) in vw.cars.iter().enumerate() {
        let d = ((car.body.state.pos[0] - xy[0]).powi(2) + (car.body.state.pos[1] - xy[1]).powi(2))
            .sqrt();
        if d <= max_dist && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

fn enter_car(
    vw: &mut VehicleWorld,
    i: usize,
    player: &mut Option<ResMut<crate::Player>>,
    vis: &mut Query<&mut Visibility, With<crate::PlayerRoot>>,
) {
    vw.driving = Some(i);
    // Teleport the walker to the driver door so the seated position is sane.
    let (class_key, s) = {
        let car = &vw.cars[i];
        (car.class.clone(), car.body.state)
    };
    let enter = vw
        .classes
        .get(&class_key)
        .and_then(|c| c.setup.helpers.get("driverEnterPos").copied())
        .unwrap_or([-1.8, 0.0, 0.0]);
    let cy = s.yaw.cos();
    let sy = s.yaw.sin();
    let fwd = [cy, sy];
    let rt = [sy, -cy];
    if let Some(p) = player.as_deref_mut() {
        p.mstate.pos = [
            s.pos[0] + rt[0] * enter[0] + fwd[0] * enter[1],
            s.pos[1] + rt[1] * enter[0] + fwd[1] * enter[1],
            s.pos[2] + enter[2],
        ];
    }
    for mut v in vis.iter_mut() {
        *v = Visibility::Hidden;
    }
}

fn exit_car(
    vw: &mut VehicleWorld,
    i: usize,
    player: &mut Option<ResMut<crate::Player>>,
    vis: &mut Query<&mut Visibility, With<crate::PlayerRoot>>,
) {
    let (class_key, s) = {
        let car = &vw.cars[i];
        (car.class.clone(), car.body.state)
    };
    let exit = vw
        .classes
        .get(&class_key)
        .and_then(|c| c.setup.helpers.get("driverExitPos").copied())
        .unwrap_or([-1.8, 0.0, 0.0]);
    let cy = s.yaw.cos();
    let sy = s.yaw.sin();
    let fwd = [cy, sy];
    let rt = [sy, -cy];
    if let Some(p) = player.as_deref_mut() {
        p.mstate.pos = [
            s.pos[0] + rt[0] * exit[0] + fwd[0] * exit[1],
            s.pos[1] + rt[1] * exit[0] + fwd[1] * exit[1],
            s.pos[2] + exit[2],
        ];
        p.pos = [p.mstate.pos[0], p.mstate.pos[1]];
        p.mstate.vel = [0.0, 0.0, 0.0];
    }
    vw.driving = None;
    for mut v in vis.iter_mut() {
        *v = Visibility::Inherited;
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

fn vehicle_camera(
    time: Res<Time>,
    vw: ResMut<VehicleWorld>,
    mut cam: Query<&mut Transform, With<crate::MainCamera>>,
) {
    let Some(d) = vw.driving else { return };
    let mut vw = vw;
    let (class_key, s) = {
        let car = &vw.cars[d];
        (car.class.clone(), car.body.state)
    };
    let view = vw.classes.get(&class_key).map(|c| c.view).unwrap_or_default();
    let base = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let yaw_q = base * Quat::from_rotation_z(s.yaw - std::f32::consts::FRAC_PI_2);
    let body = Vec3::new(s.pos[0], s.pos[1], s.pos[2]);
    // Body-space camera offset plus the third-person boom behind the car.
    let mut cam_body = Vec3::new(view.pos_offset[0], view.pos_offset[1], view.pos_offset[2]);
    cam_body.y -= if view.distance > 0.0 { view.distance } else { 6.0 };
    cam_body.z += view.height_offset.max(1.0);
    let want = base * body + yaw_q * cam_body;
    let target = base * body
        + yaw_q * Vec3::new(view.aim_offset[0], view.aim_offset[1], view.aim_offset[2] + 0.7);
    let dt = time.delta_secs().clamp(0.0, 1.0 / 20.0);
    if !vw.cam_init {
        vw.cam_pos = want;
        vw.cam_init = true;
    } else {
        let k = (dt * 6.0).min(1.0);
        let cur = vw.cam_pos;
        vw.cam_pos = cur + (want - cur) * k;
    }
    for mut t in &mut cam {
        *t = Transform::from_translation(vw.cam_pos).looking_at(target, Vec3::Y);
    }
}
