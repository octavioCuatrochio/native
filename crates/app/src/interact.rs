//! Player interaction with searchable containers.
//!
//! Replaces the retired hand-written search loop (see
//! `/tmp/opencode-e2/REPORT.md`): the player approaches/looks at a
//! `UL_Searchable` entity, gets a "Search (Hold)" prompt, holds USE, and after
//! the shipped normal search duration (`sr_searchDurationNormal 10.0`,
//! `libs/config/meleeprototype.cfg`) the container's own `ul_searchable.lua`
//! state machine moves `NotSearched -> Searched` (the script swaps the model
//! slots) and the loot rolled from `libs/class3/rts/search.xml` lands in the
//! backpack.
//!
//! The engine side (data, tables, inventory, `ItemSystem` binds) lives in
//! `sod_core::cry::items`. This module only wires it to Bevy + the `ScriptHost`.

use bevy::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use sod_core::cry::items::{self, ItemRuntime, SearchSession, BASE_BACKPACK_CAPACITY, SEARCH_REACH};
use sod_core::cry::world::EntityId;

/// One searchable container the interaction loop knows about.
struct Container {
    id: EntityId,
    site: String,
    prop: String,
    /// CryEngine-space (Z-up) position, for the proximity check.
    pos: [f32; 3],
    searched: bool,
    /// Presentation stand-in for the container model (the app does not draw
    /// `ScriptHost` slots yet); recoloured when the script state changes.
    #[allow(dead_code)]
    marker: Entity,
    marker_mat: Handle<StandardMaterial>,
}

/// Interaction state. Non-send because the shared item runtime owns a
/// `ScriptHost`-adjacent `Rc<Vfs>`.
struct InteractState {
    rt: Rc<RefCell<ItemRuntime>>,
    player: u64,
    containers: Vec<Container>,
    session: Option<SearchSession>,
    target: Option<usize>,
    prompt: String,
    notice: Option<(String, f32)>,
    spawned: bool,
    cooldown: f32,
}

/// Registers the interaction loop and installs the `ItemSystem` script binds.
///
/// Must be added after `cry_bridge::install` (the `CryHost` non-send resource
/// it reads). When no script host exists (e.g. `SOD_CRY=0`) it degrades to a
/// data-only runtime and does nothing else.
pub struct InteractPlugin;

impl Plugin for InteractPlugin {
    fn build(&self, app: &mut App) {
        let (rt, player) = match app.world().get_non_send::<crate::cry_bridge::CryHost>() {
            Some(host) => {
                let rt = Rc::new(RefCell::new(ItemRuntime::load(host.host.ctx.vfs.clone())));
                if let Err(e) = items::install(&host.host.lua, &host.host.ctx, rt.clone()) {
                    eprintln!("[interact] ItemSystem install failed: {e}");
                }
                (rt, Some(host.player as u64))
            }
            None => (
                Rc::new(RefCell::new(ItemRuntime::new(
                    Default::default(),
                    Default::default(),
                    Rc::new(sod_core::crypak::Vfs::new()),
                    0x5eed_1234_9abc_def0,
                ))),
                None,
            ),
        };

        let player = player.unwrap_or(0);
        if player != 0 {
            let mut r = rt.borrow_mut();
            r.ensure_inventory(player, BASE_BACKPACK_CAPACITY);
            // The single-player rules' `GiveItemPack(actor, "DefaultPlayer", ...)`
            // runs during `OnClientConnect`, before this plugin installs the
            // bind, so it is a no-op here. Grant the authored loadout natively
            // when asked (`SOD_ITEM_PACK=1`); the bind otherwise serves any
            // later call.
            let granted = if std::env::var("SOD_ITEM_PACK").is_ok() { r.grant_pack(player, "DefaultPlayer") } else { Vec::new() };
            println!(
                "[interact] item db: {} defs, {} item lists, {} sites; default loadout {} items",
                r.db.len(),
                r.search.list_count(),
                r.search.site_count(),
                granted.len()
            );
        }

        app.insert_non_send(InteractState {
            rt,
            player,
            containers: Vec::new(),
            session: None,
            target: None,
            prompt: String::new(),
            notice: None,
            spawned: false,
            cooldown: 0.0,
        });
        app.add_systems(
            Update,
            (
                interact_spawn.after(crate::move_player),
                interact_update.after(crate::move_player).after(interact_spawn),
                interact_hud.after(crate::hud_update),
            ),
        );
    }
}

/// Spawn the searchable containers once the player exists.
///
/// The shipped level has no authored `UL_Searchable` entities (the RTS layer
/// generates them from buildings at runtime), so this places the original
/// class near the player start, bound to real `search.xml` props.
fn interact_spawn(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    player: Option<Res<crate::Player>>,
    cry: Option<NonSendMut<crate::cry_bridge::CryHost>>,
    state: Option<NonSendMut<InteractState>>,
) {
    let (Some(player), Some(mut cry), Some(mut state)) = (player, cry, state) else { return };
    if state.spawned {
        return;
    }
    let p = player.mstate.pos;
    let (s, co) = (player.yaw.sin(), player.yaw.cos());
    let fwd = [co, s];
    let right = [-s, co];
    let specs: [(&str, &str, &str, f32, f32); 3] = [
        ("UL_Searchable_Cooler", "Any", "CoolerChest", 2.0, 0.0),
        ("UL_Searchable_Cabinet", "Any", "Cabinet", 2.6, 1.0),
        ("UL_Searchable_Crate", "Any", "Crate", 3.0, -1.0),
    ];
    let mesh = meshes.add(Cuboid::new(0.7, 0.9, 0.7));
    for (name, site, prop, dist, side) in specs {
        let pos = [p[0] + fwd[0] * dist + right[0] * side, p[1] + fwd[1] * dist + right[1] * side, p[2]];
        match items::spawn_searchable(&mut cry.host, name, pos, site, prop) {
            Ok(id) => {
                // Marker sits on the ground where the script entity is.
                let bevy_pos = crate::base_rot() * Vec3::new(pos[0], pos[1], pos[2] + 0.45);
                let mat = materials.add(StandardMaterial {
                    base_color: Color::srgb(0.55, 0.42, 0.30),
                    perceptual_roughness: 0.9,
                    ..default()
                });
                let marker = commands.spawn((Mesh3d(mesh.clone()), MeshMaterial3d(mat.clone()), Transform::from_translation(bevy_pos))).id();
                state.containers.push(Container {
                    id,
                    site: site.to_string(),
                    prop: prop.to_string(),
                    pos,
                    searched: false,
                    marker,
                    marker_mat: mat,
                });
            }
            Err(e) => eprintln!("[interact] spawn {name} failed: {e}"),
        }
    }
    state.spawned = true;
    println!("[interact] spawned {} searchable containers near the player", state.containers.len());
}

/// The per-frame interaction loop: pick the nearest container in front of the
/// player, run the hold-to-search timer and complete the search.
fn interact_update(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    menu: Option<Res<crate::MenuRes>>,
    player: Option<Res<crate::Player>>,
    cry: Option<NonSendMut<crate::cry_bridge::CryHost>>,
    state: Option<NonSendMut<InteractState>>,
) {
    let (Some(player), Some(mut cry), Some(mut state)) = (player, cry, state) else { return };
    if state.player == 0 {
        return;
    }

    // Decay the completion notice.
    if let Some((_, ttl)) = state.notice.as_mut() {
        *ttl -= time.delta_secs();
        if *ttl <= 0.0 {
            state.notice = None;
        }
    }
    state.cooldown = (state.cooldown - time.delta_secs()).max(0.0);

    if menu.map(|m| m.active).unwrap_or(false) {
        state.prompt.clear();
        state.target = None;
        return;
    }

    let dt = time.delta_secs();
    let p = player.mstate.pos;
    let fwd = [player.yaw.cos(), player.yaw.sin()];

    // Nearest unsearched container within reach and roughly in front.
    let mut best: Option<(usize, f32)> = None;
    for (i, c) in state.containers.iter().enumerate() {
        if c.searched {
            continue;
        }
        let d = [c.pos[0] - p[0], c.pos[1] - p[1]];
        let dist = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if dist > SEARCH_REACH {
            continue;
        }
        let dot = if dist > 1e-3 { (d[0] * fwd[0] + d[1] * fwd[1]) / dist } else { 1.0 };
        if dot < -0.15 {
            continue;
        }
        if best.map_or(true, |(_, bd)| dist < bd) {
            best = Some((i, dist));
        }
    }
    state.target = best.map(|(i, _)| i);

    let auto = std::env::var("SOD_AUTOSEARCH").is_ok();
    let holding = keys.pressed(KeyCode::KeyE) || auto;
    // Fast search is the sprint-while-searching mechanic; the harness forces it
    // so a screenshot lands after the containers have been searched.
    let fast = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) || auto;
    let ready = state.cooldown <= 0.0;

    let mut complete: Option<u64> = None;
    if let Some(i) = state.target {
        let (cid, site, prop) = {
            let c = &state.containers[i];
            (c.id as u64, c.site.clone(), c.prop.clone())
        };
        if state.session.as_ref().map(|s| s.target) != Some(cid) {
            state.session = None;
        }
        if state.session.is_none() && holding && ready {
            state.session = Some(SearchSession::begin(cid, site, prop));
        }
        if let Some(session) = state.session.as_mut() {
            if holding {
                if session.tick(dt, fast) {
                    complete = Some(cid);
                }
            }
        }
    } else if state.session.as_ref().map(|s| s.progress <= 0.0).unwrap_or(false) {
        state.session = None;
    }

    if let Some(cid) = complete {
        if let Some(idx) = state.containers.iter().position(|c| c.id as u64 == cid) {
            let (site, prop) = (state.containers[idx].site.clone(), state.containers[idx].prop.clone());
            let loot = state.rt.borrow_mut().complete_search(state.player, &site, &prop);
            // The flowgraph `Searched` input calls `UL_Searchable.Event_Searched`,
            // which runs `GotoState("Searched")` (model swap).
            let _ = cry.host.callback(state.containers[idx].id, "Event_Searched");
            state.containers[idx].searched = true;
            state.session = None;
            state.cooldown = items::SR_COOLDOWN_DURATION;
            // Visual stand-in for the `Searched` model swap.
            if let Some(mut m) = materials.get_mut(&state.containers[idx].marker_mat) {
                m.base_color = Color::srgb(0.28, 0.30, 0.32);
            }

            let items_txt: Vec<String> = loot.items.iter().map(|s| format!("{} x{}", s.id, s.count)).collect();
            let res_txt: Vec<String> = loot.resources.iter().map(|r| format!("{} +{}", r.resource.name(), r.amount)).collect();
            let mut msg = format!("Found: {}", items_txt.join(", "));
            if !res_txt.is_empty() {
                msg.push_str(&format!("  [{}]", res_txt.join(", ")));
            }
            println!("[interact] searched {} ({}/{}) -> {}", state.containers[idx].prop, site, prop, msg);
            state.notice = Some((msg, 5.0));
        }
    }

    // Prompt text.
    state.prompt = match (state.target, &state.session) {
        (Some(i), Some(s)) => {
            format!("Searching {}...  {:.0}%  [hold USE]", state.containers[i].prop, s.ratio() * 100.0)
        }
        (Some(i), None) => format!("Search (Hold) [E]   {}", state.containers[i].prop),
        _ => String::new(),
    };
}

/// After `hud_update`, replace the backpack line and centre prompt with the
/// live search/inventory state.
fn interact_hud(state: Option<NonSend<InteractState>>, mut lines: Query<(&mut Text, &crate::HudLine)>) {
    let Some(state) = state else { return };
    let (inv_line, db_line) = {
        let rt = state.rt.borrow();
        let inv = rt.inventory(state.player).map(|ai| ai.summary(&rt.db)).unwrap_or_default();
        let res = [
            sod_core::cry::items::ResourceType::Ammo,
            sod_core::cry::items::ResourceType::Food,
            sod_core::cry::items::ResourceType::Medicine,
            sod_core::cry::items::ResourceType::Materials,
            sod_core::cry::items::ResourceType::Fuel,
        ]
        .iter()
        .filter_map(|r| {
            let n = rt.stockpile_of(*r);
            (n > 0).then(|| format!("{} {}", n, r.name()))
        })
        .collect::<Vec<_>>()
        .join("  ");
        (if res.is_empty() { inv } else { format!("{inv}   |   {res}") }, ())
    };
    let _ = db_line;
    let prompt = state.prompt.clone();
    let notice = state.notice.as_ref().map(|(t, _)| t.clone()).unwrap_or_default();

    for (mut text, line) in &mut lines {
        match line {
            crate::HudLine::Inventory => {
                if !inv_line.is_empty() {
                    text.0 = inv_line.clone();
                }
            }
            crate::HudLine::Prompt => text.0 = prompt.clone(),
            crate::HudLine::Notifications => {
                if !notice.is_empty() {
                    text.0 = notice.clone();
                }
            }
            _ => {}
        }
    }
}
