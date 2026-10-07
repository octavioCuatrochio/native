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
//!
//! ## Models
//!
//! `ul_searchable.lua` is a two-slot prop: `LoadObject(NotSearchedSlot,
//! Properties.object_ModelNotSearched)` and `LoadObject(SearchedSlot,
//! Properties.object_ModelSearched)`, with `DrawSlot` hiding one slot and
//! showing the other per state (`NotSearched` draws slot 0, `Searched` slot 1).
//! The RTS layer fills those two properties per prop from
//! `libs/class3/rts/props.xml` (`base` before searching, `depleted` after,
//! matching the class defaults). The native RTS stand-in here resolves the
//! prop's models from that same table and renders both variants from the shared
//! [`crate::PropAssets`] handles the level pipeline already built, swapping them
//! on the script's `Searched` transition. No placeholder geometry, no tint.
//!
//! Note: the app's `EntityNative` geometry binds (`LoadObject`/`DrawSlot`) are
//! stubs in `sod_core`, so the script's slot state is not projected into the
//! entity world and drawn generically; this module mirrors the exact two-slot
//! semantics the script expresses, using the same models and the same draw
//! order the class defines.

use bevy::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use sod_core::cry::items::{self, ItemRuntime, SearchSession, BASE_BACKPACK_CAPACITY, SEARCH_REACH};
use sod_core::cry::world::EntityId;
use sod_core::crypak::Vfs;

/// Shipped RTS prop → model table (the prop distribution reads it alongside
/// `Search.xml`; see `notes/ghidra/rts-sim.md`).
const PROPS_XML: &str = "libs/class3/rts/props.xml";

/// `ul_searchable.lua`'s class defaults, used when a prop has no `props.xml`
/// entry (the script would keep these values).
const DEFAULT_NOT_SEARCHED: &str =
    "objects/decor/resources/resource_food/food_cooler_chest/food_cooler_chest.cgf";
const DEFAULT_SEARCHED: &str =
    "objects/decor/resources/resource_food/food_cooler_chest/food_cooler_chest_depleted.cgf";

/// The original RTS generates `UL_Searchable` props from buildings at runtime;
/// the shipped level has no authored instances (see `/tmp/opencode-g1/REPORT.md`).
/// This is the native stand-in, placing the real `.ent` class at the tutorial
/// start bound to real `search.xml` site/prop pairs.
const SPAWN_SPECS: [(&str, &str, &str, f32, f32); 3] = [
    ("UL_Searchable_Cooler", "Any", "CoolerChest", 2.0, 0.0),
    ("UL_Searchable_Cabinet", "Any", "Cabinet", 2.6, 1.0),
    ("UL_Searchable_Crate", "Any", "Crate", 3.0, -1.0),
];

/// One searchable container the interaction loop knows about.
struct Container {
    id: EntityId,
    site: String,
    prop: String,
    /// CryEngine-space (Z-up) position, for the proximity check.
    pos: [f32; 3],
    searched: bool,
    /// `NotSearched` slot meshes (`DrawSlot(0, 1)`).
    not_searched: Vec<Entity>,
    /// `Searched` slot meshes (`DrawSlot(1, 1)`).
    searched_slot: Vec<Entity>,
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
    /// `propId` (lowercase) -> (`object_ModelNotSearched`, `object_ModelSearched`),
    /// from `props.xml`.
    prop_models: HashMap<String, (String, String)>,
}

/// One searchable container's live state, published for the content manager.
/// `SearchBuilding` counts a building searched once **all** of its containers
/// are (`RTSWorldBuildingSetSearched` `0x013b6c20`).
#[derive(Clone, Debug)]
pub struct SearchableEntry {
    /// CryEngine-space position (Z-up).
    pub pos: [f32; 3],
    /// RTS site / building group this container belongs to.
    pub site: String,
    pub searched: bool,
}

/// Registry of live searchables, rebuilt each frame after `interact_update`.
#[derive(Resource, Default)]
pub struct SearchRegistry {
    pub entries: Vec<SearchableEntry>,
}

/// A request to place one RTS building/site's searchable container. The content
/// bridge pushes these when a `SearchBuilding` objective activates (the original
/// RTS generates the props from the building's markup volume); the interaction
/// loop owns the actual `UL_Searchable` spawns and their search state.
#[derive(Clone, Debug)]
pub struct SearchSpawnRequest {
    pub site: String,
    pub prop: String,
    pub pos: [f32; 3],
}

/// Pending searchable-spawn requests from the content manager.
#[derive(Resource, Default)]
pub struct SearchSpawnQueue {
    pub requests: Vec<SearchSpawnRequest>,
}

/// Registers the interaction loop and installs the `ItemSystem` script binds.
///
/// Must be added after `cry_bridge::install` (the `CryHost` non-send resource
/// it reads). When no script host exists (e.g. `SOD_CRY=0`) it degrades to a
/// data-only runtime and does nothing else.
pub struct InteractPlugin;

impl Plugin for InteractPlugin {
    fn build(&self, app: &mut App) {
        let (rt, player, prop_models) = match app.world().get_non_send::<crate::cry_bridge::CryHost>() {
            Some(host) => {
                let prop_models = host
                    .host
                    .ctx
                    .vfs
                    .read(PROPS_XML)
                    .map(|b| parse_props_xml(&b))
                    .unwrap_or_default();
                let rt = Rc::new(RefCell::new(ItemRuntime::load(host.host.ctx.vfs.clone())));
                if let Err(e) = items::install(&host.host.lua, &host.host.ctx, rt.clone()) {
                    eprintln!("[interact] ItemSystem install failed: {e}");
                }
                (rt, Some(host.player as u64), prop_models)
            }
            None => (
                Rc::new(RefCell::new(ItemRuntime::new(
                    Default::default(),
                    Default::default(),
                    Rc::new(sod_core::crypak::Vfs::new()),
                    0x5eed_1234_9abc_def0,
                ))),
                None,
                HashMap::new(),
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
            prop_models,
        });
        app.init_resource::<SearchRegistry>().init_resource::<SearchSpawnQueue>();
        app.add_systems(
            Update,
            (
                interact_spawn.after(crate::move_player),
                interact_mission_spawn.after(interact_spawn),
                interact_update
                    .after(crate::move_player)
                    .after(interact_mission_spawn),
                interact_hud.after(crate::hud_update),
            ),
        );
    }
}

/// `propId` -> (`object_ModelNotSearched`, `object_ModelSearched`) for every
/// self-closing `<Prop>` in `props.xml`. The first entry for a `propId` wins.
fn parse_props_xml(bytes: &[u8]) -> HashMap<String, (String, String)> {
    let text = String::from_utf8_lossy(bytes);
    let mut out = HashMap::new();
    let mut rest: &str = &text;
    while let Some(i) = rest.find("<Prop") {
        let after = &rest[i..];
        let Some(end) = after.find('>') else { break };
        let tag = &after[..=end];
        if let (Some(prop), Some(base)) = (attr(tag, "propId"), attr(tag, "base")) {
            let searched = attr(tag, "depleted").unwrap_or(base);
            out.entry(prop.to_ascii_lowercase())
                .or_insert_with(|| (base.to_string(), searched.to_string()));
        }
        rest = &after[end + 1..];
    }
    out
}

/// Value of `name="..."` inside a single XML tag.
fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let pat = format!("{name}=\"");
    let start = tag.find(&pat)? + pat.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

/// The `props.xml` model paths the native searchables need, so the level build
/// can decode their meshes/materials before the runtime spawn (see
/// `main.rs`'s `run_world`). Includes the class defaults as a fallback.
pub fn searchable_wanted_models(vfs: &Vfs) -> Vec<String> {
    let map = vfs.read(PROPS_XML).map(|b| parse_props_xml(&b)).unwrap_or_default();
    let mut out = vec![DEFAULT_NOT_SEARCHED.to_string(), DEFAULT_SEARCHED.to_string()];
    for (_, _, prop, _, _) in SPAWN_SPECS {
        if let Some((not_searched, searched)) = map.get(&prop.to_ascii_lowercase()) {
            out.push(not_searched.clone());
            out.push(searched.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Spawn the searchable containers once the player exists.
///
/// The shipped level has no authored `UL_Searchable` entities (the RTS layer
/// generates them from buildings at runtime), so this places the original
/// class near the player start, bound to real `search.xml` props and to their
/// `props.xml` models (drawn from the level's own asset handles).
fn interact_spawn(
    mut commands: Commands,
    player: Option<Res<crate::Player>>,
    assets: Option<Res<crate::PropAssets>>,
    cry: Option<NonSendMut<crate::cry_bridge::CryHost>>,
    state: Option<NonSendMut<InteractState>>,
) {
    let (Some(player), Some(assets), Some(mut cry), Some(mut state)) = (player, assets, cry, state) else { return };
    if state.spawned {
        return;
    }
    let p = player.mstate.pos;
    let (s, co) = (player.yaw.sin(), player.yaw.cos());
    let fwd = [co, s];
    let right = [-s, co];
    for (name, site, prop, dist, side) in SPAWN_SPECS {
        let pos = [p[0] + fwd[0] * dist + right[0] * side, p[1] + fwd[1] * dist + right[1] * side, p[2]];
        if let Some(c) = spawn_container(&mut commands, &assets, &mut cry, &state.prop_models, name, pos, site, prop) {
            state.containers.push(c);
        }
    }
    state.spawned = true;
    println!("[interact] spawned {} searchable containers near the player", state.containers.len());
}

/// Spawn one searchable `UL_Searchable` + its two model slots and return the
/// runtime record. Shared by the start-up placement and the content manager's
/// `SearchBuilding` building requests.
#[allow(clippy::too_many_arguments)]
fn spawn_container(
    commands: &mut Commands,
    assets: &crate::PropAssets,
    cry: &mut crate::cry_bridge::CryHost,
    prop_models: &HashMap<String, (String, String)>,
    name: &str,
    pos: [f32; 3],
    site: &str,
    prop: &str,
) -> Option<Container> {
    let (not_searched_path, searched_path) = prop_models
        .get(&prop.to_ascii_lowercase())
        .cloned()
        .unwrap_or_else(|| (DEFAULT_NOT_SEARCHED.to_string(), DEFAULT_SEARCHED.to_string()));
    let id = match items::spawn_searchable(&mut cry.host, name, pos, site, prop) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("[interact] spawn {name} failed: {e}");
            return None;
        }
    };
    // Model root at the container's world transform. CryEngine Z-up maps to
    // Bevy Y-up through `base_rot`, as the level root does.
    let root = commands
        .spawn((
            Transform {
                translation: crate::base_rot() * Vec3::new(pos[0], pos[1], pos[2]),
                rotation: crate::base_rot(),
                scale: Vec3::ONE,
            },
            Visibility::default(),
        ))
        .id();
    let mut not_searched = Vec::new();
    let mut searched_slot = Vec::new();
    commands.entity(root).with_children(|parent| {
        // NotSearched.OnBeginState: DrawSlot(0, 1), DrawSlot(1, 0).
        if let Some(list) = assets.0.get(&not_searched_path) {
            for (mesh, mat) in list {
                not_searched.push(
                    parent
                        .spawn((
                            Mesh3d(mesh.clone()),
                            MeshMaterial3d(mat.clone()),
                            Transform::default(),
                            Visibility::Visible,
                        ))
                        .id(),
                );
            }
        }
        // Searched.OnBeginState: DrawSlot(0, 0), DrawSlot(1, 1).
        if let Some(list) = assets.0.get(&searched_path) {
            for (mesh, mat) in list {
                searched_slot.push(
                    parent
                        .spawn((
                            Mesh3d(mesh.clone()),
                            MeshMaterial3d(mat.clone()),
                            Transform::default(),
                            Visibility::Hidden,
                        ))
                        .id(),
                );
            }
        }
    });
    if not_searched.is_empty() && searched_slot.is_empty() {
        eprintln!("[interact] {name}: no meshes for {not_searched_path} / {searched_path}");
    }
    Some(Container {
        id,
        site: site.to_string(),
        prop: prop.to_string(),
        pos,
        searched: false,
        not_searched,
        searched_slot,
    })
}

/// Drain the content manager's `SearchBuilding` spawn requests: place each
/// requested building's searchable container (RTS `props.xml` models) at the
/// authored position and bind it to the building's site tag.
fn interact_mission_spawn(
    mut commands: Commands,
    assets: Option<Res<crate::PropAssets>>,
    cry: Option<NonSendMut<crate::cry_bridge::CryHost>>,
    state: Option<NonSendMut<InteractState>>,
    mut queue: ResMut<SearchSpawnQueue>,
) {
    let (Some(assets), Some(mut cry), Some(mut state)) = (assets, cry, state) else { return };
    if queue.requests.is_empty() {
        return;
    }
    let requests = std::mem::take(&mut queue.requests);
    for req in requests {
        let name = format!("ULMissionSearchab{}", state.containers.len());
        if let Some(c) = spawn_container(&mut commands, &assets, &mut cry, &state.prop_models, &name, req.pos, &req.site, &req.prop) {
            println!(
                "[interact] mission searchable {name} site {} at ({:.1},{:.1},{:.1})",
                req.site, req.pos[0], req.pos[1], req.pos[2]
            );
            state.containers.push(c);
        }
    }
}

/// The per-frame interaction loop: pick the nearest container in front of the
/// player, run the hold-to-search timer and complete the search.
fn interact_update(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    menu: Option<Res<crate::MenuRes>>,
    player: Option<Res<crate::Player>>,
    cry: Option<NonSendMut<crate::cry_bridge::CryHost>>,
    state: Option<NonSendMut<InteractState>>,
    mut registry: ResMut<SearchRegistry>,
) {
    let (Some(player), Some(cry), Some(mut state)) = (player, cry, state) else { return };
    // Publish the live searchable table for the content manager's
    // `SearchBuilding` predicate before any early return.
    registry.entries = state
        .containers
        .iter()
        .map(|c| SearchableEntry { pos: c.pos, site: c.site.clone(), searched: c.searched })
        .collect();
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

    let auto = std::env::var("SOD_AUTOSEARCH").is_ok();
    // Nearest unsearched container within reach and roughly in front. The
    // harness does not steer the camera, so it ignores the facing test and
    // searches whichever container is nearest.
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
        if !auto && dot < -0.15 {
            continue;
        }
        if best.map_or(true, |(_, bd)| dist < bd) {
            best = Some((i, dist));
        }
    }
    state.target = best.map(|(i, _)| i);
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
            // which runs `GotoState("Searched")` and swaps the drawn slot.
            let _ = cry.host.callback(state.containers[idx].id, "Event_Searched");
            state.containers[idx].searched = true;
            state.session = None;
            state.cooldown = items::SR_COOLDOWN_DURATION;
            // `Searched.OnBeginState`: DrawSlot(0, 0), DrawSlot(1, 1).
            for e in &state.containers[idx].not_searched {
                commands.entity(*e).insert(Visibility::Hidden);
            }
            for e in &state.containers[idx].searched_slot {
                commands.entity(*e).insert(Visibility::Visible);
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
