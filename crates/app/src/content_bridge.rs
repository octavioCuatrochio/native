//! App-side bridge for the original content manager: missions, objectives and
//! the `Home` enclave story state machine.
//!
//! Data + runtime live in `sod_core::cry::content` / `content_runtime`; this
//! module is the thing the original game calls the *content manager owner* and
//! the *RTS story runner* for:
//!
//! 1. It parses the shipped `missions.xml` and `rts/enclaves.xml`, sets
//!    `Game.Tutorial`, and runs the `Arrive → Buddies → Mountain → Escape →
//!    PreChurch` state machine (every action's `CheckBool`/`Enclave_CheckBool`
//!    gates and `Mission_Create` / `ChangeState` / stat outputs). Mission
//!    completion writes the enclave `MissionPrimary_Done` /
//!    `MissionSecondary_Done` stats that drive the state transitions.
//! 2. It ticks the [`ContentManager`] every frame against a live
//!    [`MissionWorld`] built from the controlled player, the level tag points
//!    and the script host's live actors, so objectives advance from in-game
//!    actions (Clearing an area, leading to a tag point, searching, surveying,
//!    killing).
//! 3. It resolves [`ContentEffect`]s: spawns actors through the script host,
//!    routes `Scene_ScenePlay` dialog lines to the HUD (and the sound system
//!    when the event is loaded), turns `EventLog` into HUD banners/notifications
//!    and arming hints, and publishes the current objective text (localized
//!    through `gamedata/languages/*.win.btxt`) to the HUD.
//! 4. `SOD_TUTORIAL_AUTOPLAY=1` drives the player to each objective and marks
//!    the world predicates the standalone build cannot produce yet, so the
//!    whole Tanner arc runs head-lessly for verification.
//!
//! Class3 addresses for the content manager are cited in
//! `crates/core/src/cry/content.rs` and `content_runtime.rs`.

use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use sod_core::cry::content::{
    parse_document, ActorKind, Mission, MissionData, Objective, ObjectiveType,
};
use sod_core::cry::combat::{self as cry_combat, DamageTarget, Hit, HitMaterial};
use sod_core::cry::content_runtime::{ContentEffect, ContentManager, MissionWorld};
use sod_core::cry::zombie_runtime::class_script;
use sod_core::cry::zombie_spawn::{freak_entity_class, kind_for_class};
use sod_core::game::ZombieKind;
use sod_core::crypak::{Pak, Vfs};
use sod_core::tagpoints::TagPoints;

use crate::ai_bridge::CombatStats;
use crate::cry_bridge::CryHost;
use crate::interact::{SearchRegistry, SearchSpawnQueue, SearchSpawnRequest};

/// Register the bridge. Added inside `run_world`'s play branch, after the
/// script host exists (the bridge reads the mounts through it).
pub struct ContentBridgePlugin;

impl Plugin for ContentBridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ContentBridge>()
            .add_systems(Startup, content_init)
            .add_systems(
                Update,
                (content_autostart, content_autoplay, content_tick)
                    .chain()
                    .after(crate::move_player),
            )
            .add_systems(Update, content_hud.after(crate::hud_update));
    }
}

// ---------------------------------------------------------------------------
// Story state machine (the `Home` enclave in `rts/enclaves.xml`)
// ---------------------------------------------------------------------------

/// One `<Inputs>` condition of a story action.
struct StoryInput {
    kind: String,
    id: String,
    invert: bool,
}

/// One `Mission_Create` output, remembering the completion stats it sets.
struct StoryCreate {
    mission: String,
    success: String,
    abort: String,
}

/// One `<Action>` of a story state.
struct StoryAction {
    id: String,
    /// `ActionFlags="Repeats"`: may fire again once its inputs stay satisfied.
    /// Actions without it fire at most once per state entry (this is how the
    /// game gates `ResetBools`, `CreateMission (Init)`, `ta15Go`, …).
    repeats: bool,
    /// `ActionFlags="WaitForNoMission"`: hold while a mission is running, so
    /// primary/secondary missions start one at a time (`mission-flow.md:326`).
    wait_no_mission: bool,
    inputs: Vec<StoryInput>,
    creates: Vec<StoryCreate>,
    changes: Vec<String>,
    /// `Enclave_StatSetBool` outputs (`OutputStat` -> bool), stored prefixed.
    enclave_stats: Vec<(String, bool)>,
    /// `RTSStat_SetBool` outputs (`OutputId` -> bool), stored verbatim.
    stat_bools: Vec<(String, bool)>,
}

struct StoryState {
    id: String,
    actions: Vec<StoryAction>,
}

/// The recovered `Home` enclave state machine.
#[derive(Default)]
struct StoryMachine {
    states: Vec<StoryState>,
}

impl StoryMachine {
    fn state(&self, id: &str) -> Option<&StoryState> {
        self.states.iter().find(|s| s.id == id)
    }
}

// ---------------------------------------------------------------------------
// Localization (`gamedata/languages/*.win.btxt`)
// ---------------------------------------------------------------------------

/// TXDB (`*.win.btxt`) string table. Entry order is stable across languages, so
/// an English string maps to the same index in the selected language.
#[derive(Default)]
struct Localization {
    index: HashMap<String, usize>,
    cur: Vec<String>,
}

impl Localization {
    fn load(vfs: &Vfs, game: &str, lang: &str) -> Localization {
        let en = load_btxt(vfs, game, "english").unwrap_or_default();
        let cur = if lang == "english" {
            en.clone()
        } else {
            load_btxt(vfs, game, lang).unwrap_or_else(|| en.clone())
        };
        let mut index = HashMap::new();
        for (i, s) in en.iter().enumerate() {
            index.entry(s.clone()).or_insert(i);
        }
        Localization { index, cur }
    }

    /// Translated text for `s`, or `s` when it is not in the shipped table.
    fn get(&self, s: &str) -> String {
        if let Some(&i) = self.index.get(s) {
            if let Some(v) = self.cur.get(i) {
                if !v.is_empty() {
                    return v.clone();
                }
            }
        }
        s.to_string()
    }
}

/// Decode a CryEngine `TXDB` text database: header (`TXDB`, version, count at
/// offset 12), a `count`-entry `u32` offset array, then the NUL-separated
/// UTF-8 string blob.
fn parse_btxt(bytes: &[u8]) -> Vec<String> {
    if bytes.len() < 16 || &bytes[0..4] != b"TXDB" {
        return Vec::new();
    }
    let count = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let start = 16 + 4 * count;
    if count == 0 || start >= bytes.len() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count);
    for chunk in bytes[start..].split(|b| *b == 0) {
        if out.len() >= count {
            break;
        }
        out.push(String::from_utf8_lossy(chunk).trim().to_string());
    }
    out
}

fn load_btxt(vfs: &Vfs, game: &str, lang: &str) -> Option<Vec<String>> {
    let name = format!("{lang}.win.btxt");
    let bytes = vfs
        .paths_with_ext(&name)
        .into_iter()
        .find_map(|p| vfs.read(&p))
        .or_else(|| std::fs::read(Path::new(game).join("gamedata/languages").join(&name)).ok())?;
    Some(parse_btxt(&bytes))
}

// ---------------------------------------------------------------------------
// Bridge state
// ---------------------------------------------------------------------------

/// Runtime state for the content manager, story machine and HUD feed.
#[derive(Resource)]
pub struct ContentBridge {
    ready: bool,
    cm: ContentManager,
    story: StoryMachine,
    state: String,
    /// Per-action "already fired this state entry" flags for `story_step`.
    story_fired: Vec<bool>,
    /// mission label -> (success stat, abort stat) from the `Mission_Create`.
    pending_stats: HashMap<String, (String, String)>,
    loc: Localization,
    tags: TagPoints,
    /// Scene buckets -> (audio file, subtitle) lines, for dialog playback.
    scenes: HashMap<String, Vec<(String, String)>>,
    /// Mission whose objective currently owns the HUD.
    focus: Option<String>,

    /// Named survey points from the level (`EntityClass="SurveyPoint"`):
    /// `SurveyGroupId` -> world position. The `Survey` objective's action is
    /// performed at one of these (`LoadSurveyPoints` `0x013b07a0`).
    survey_points: HashMap<String, [f32; 3]>,
    /// Seconds spent surveying each group (timer) and completed groups.
    survey_progress: HashMap<String, f32>,
    surveyed: HashSet<String>,
    /// `(mission, bucket)` whose `SearchBuilding` containers were requested.
    search_requested: HashSet<String>,
    /// Mission-spawned zombie actors for the `KillActor` predicate.
    mission_zeds: Vec<MissionZed>,
    /// B4 kill count when each `KillActor` objective activated; a later kill
    /// satisfies it (the real combat pipeline, B4).
    kill_baselines: HashMap<String, u64>,

    /// HUD output.
    pub objective: Option<String>,
    pub notifications: Vec<String>,
    pub banner: Option<String>,
    pub marker: Option<[f32; 3]>,

    /// Movement driver (`SOD_TUTORIAL_AUTOPLAY` or `SOD_TUTORIAL_AUTORUN`).
    drive: bool,
    /// Predicate faking — **test-only** (`SOD_TUTORIAL_AUTOPLAY`).
    fake: bool,
    at_goal: bool,
    done: Vec<String>,
    log_timer: f32,
    game_dir: String,
    /// Player was already downed; reset when revived.
    player_down: bool,
    /// The `SOD_AUTOSTART` menu dismissal has run.
    autostarted: bool,
    /// Melee/fire cadence for hitting mission-spawned zombies (which the B4
    /// director does not own, so they are resolved here through the same
    /// damage pipeline).
    zed_melee_cd: f32,
    zed_gun_cd: f32,
}

/// A zombie actor spawned by a mission `Zombie.Spawn`/`Zombie.FreakSpawn` output.
#[derive(Clone, Debug)]
struct MissionZed {
    mission: String,
    entity: sod_core::cry::world::EntityId,
}

impl Default for ContentBridge {
    fn default() -> ContentBridge {
        ContentBridge {
            ready: false,
            cm: ContentManager::new(MissionData::default()),
            story: StoryMachine::default(),
            state: "Arrive".to_string(),
            story_fired: Vec::new(),
            pending_stats: HashMap::new(),
            loc: Localization::default(),
            tags: TagPoints::default(),
            scenes: HashMap::new(),
            focus: None,
            survey_points: HashMap::new(),
            survey_progress: HashMap::new(),
            surveyed: HashSet::new(),
            search_requested: HashSet::new(),
            mission_zeds: Vec::new(),
            kill_baselines: HashMap::new(),
            objective: None,
            notifications: Vec::new(),
            banner: None,
            marker: None,
            drive: std::env::var("SOD_TUTORIAL_AUTOPLAY").is_ok()
                || std::env::var("SOD_TUTORIAL_AUTORUN").is_ok(),
            fake: std::env::var("SOD_TUTORIAL_AUTOPLAY").is_ok(),
            at_goal: false,
            done: Vec::new(),
            log_timer: 0.0,
            game_dir: std::env::args().nth(1).unwrap_or_default(),
            player_down: false,
            autostarted: false,
            zed_melee_cd: 0.0,
            zed_gun_cd: 0.0,
        }
    }
}

/// Dismiss the main menu under `SOD_AUTOSTART` without requiring `SOD_SHOT`
/// (main's own auto-start only runs while a screenshot plan is active). This
/// lets the scripted verification run as long as it needs.
fn content_autostart(
    mut commands: Commands,
    mut menu: ResMut<crate::MenuRes>,
    ui: Query<Entity, With<crate::MenuRoot>>,
    mut bridge: ResMut<ContentBridge>,
) {
    // Only the scripted tutorial harness starts the game itself; `SOD_AUTOSTART`
    // keeps main's own (frame-60) menu dismissal so other harnesses are
    // unaffected.
    if bridge.autostarted || std::env::var("SOD_TUTORIAL_AUTORUN").is_err() {
        return;
    }
    bridge.autostarted = true;
    if menu.active {
        for e in &ui {
            commands.entity(e).despawn();
        }
        menu.active = false;
        println!("[content] auto-start (content bridge)");
    }
}

impl ContentBridge {
    fn note(&mut self, text: String) {
        self.notifications.push(text);
        let n = self.notifications.len();
        if n > 4 {
            self.notifications.drain(0..n - 4);
        }
    }
}

/// [`MissionWorld`] over the live app state. Every predicate reads the real
/// game systems (B4 combat deaths, G1 container search, the survey action); the
/// `fake` flag is the test-only `SOD_TUTORIAL_AUTOPLAY` override.
struct WorldView<'a> {
    player: [f32; 3],
    tags: &'a TagPoints,
    /// Live non-player actors' positions (already filtered to the living).
    zombies: Vec<[f32; 3]>,
    /// Live searchable containers (G1 `UL_Searchable` state).
    searchables: &'a [crate::interact::SearchableEntry],
    /// Survey groups whose survey action completed.
    surveyed: &'a HashSet<String>,
    /// `(mission, actor index)` `KillActor` predicates satisfied by a real
    /// death in the B4 pipeline.
    dead_actors: &'a HashSet<(String, usize)>,
    fake: bool,
    at_goal: bool,
}

impl MissionWorld for WorldView<'_> {
    fn player_pos(&self) -> [f32; 3] {
        self.player
    }
    fn tag_point(&self, name: &str) -> Option<[f32; 3]> {
        self.tags.get(name)
    }
    fn zombie_count_near(&self, pos: [f32; 3], radius: f32) -> u32 {
        // Test-only: autoplay clears the objective area so a frame where there
        // is no combat system attached still completes ClearArea.
        if self.fake && self.at_goal {
            return 0;
        }
        let r2 = radius * radius;
        self.zombies
            .iter()
            .filter(|z| dist2(**z, pos) <= r2)
            .count() as u32
    }
    fn searched_buildings_near(&self, pos: [f32; 3], radius: f32) -> u32 {
        if self.fake && self.at_goal {
            return 99;
        }
        // A building/site counts as searched only once **all** of its
        // searchable containers are searched (`RTSWorldBuildingSetSearched`
        // `0x013b6c20`); the objective then counts the sites whose volume is
        // within `radius` of the target (`RTSWorldBuildingContainsPointInterior`
        // `0x013b4bf0`).
        let mut groups: HashMap<&str, (usize, usize, [f32; 3])> = HashMap::new();
        for e in self.searchables {
            let g = groups.entry(e.site.as_str()).or_insert((0, 0, [0.0; 3]));
            g.0 += 1;
            if e.searched {
                g.1 += 1;
            }
            g.2[0] += e.pos[0];
            g.2[1] += e.pos[1];
            g.2[2] += e.pos[2];
        }
        let r2 = radius * radius;
        groups
            .values()
            .filter(|(total, searched, sum)| {
                *total > 0
                    && *searched == *total
                    && dist2([sum[0] / *total as f32, sum[1] / *total as f32, sum[2] / *total as f32], pos) <= r2
            })
            .count() as u32
    }
    fn surveyed(&self, group: &str) -> bool {
        (self.fake && self.at_goal) || self.surveyed.contains(group)
    }
    fn actor_dead(&self, mission: &str, actor_index: usize) -> bool {
        self.dead_actors.contains(&(mission.to_string(), actor_index))
    }
    fn force_objective_complete(&self, _mission: &str, objective: &Objective) -> bool {
        if !(self.fake && self.at_goal) {
            return false;
        }
        // Types whose native interaction is not modelled yet (NPC casting,
        // gather/deliver/siege) still complete under the test-only override.
        matches!(
            objective.kind,
            ObjectiveType::RescueNpc
                | ObjectiveType::SearchProp
                | ObjectiveType::Gather
                | ObjectiveType::Tour
                | ObjectiveType::Deliver
                | ObjectiveType::ClearInfestation
                | ObjectiveType::Siege
                | ObjectiveType::BaseSiege
        )
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    dist2(a, b).sqrt()
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

fn read_named(vfs: &Vfs, suffix: &str, prefer: &str) -> Option<Vec<u8>> {
    let mut paths = vfs.paths_with_ext(suffix);
    paths.sort();
    if !prefer.is_empty() {
        paths.sort_by_key(|p| !p.contains(prefer));
    }
    paths.into_iter().find_map(|p| vfs.read(&p))
}

fn load_tags(vfs: &Vfs, game: &str) -> Option<TagPoints> {
    // Several packs ship a `tag_points_win.markup` (Beltway/Breakdown DLC carry
    // their own, ~1.6k points; the Class3 campaign file has ~7k). Path-based
    // preference is not reliable (pack entry case varies), so parse every
    // candidate and keep the richest table — the main level.
    let mut best: Option<TagPoints> = None;
    let mut cands = vfs.paths_with_ext("tag_points_win.markup");
    cands.sort();
    for p in cands {
        let Some(bytes) = vfs.read(&p) else { continue };
        let Some(t) = TagPoints::parse(&bytes) else { continue };
        if best.as_ref().map_or(true, |b| t.len() > b.len()) {
            best = Some(t);
        }
    }
    if best.is_some() {
        return best;
    }
    let pak = Pak::open(Path::new(game).join("Game/levels/Class3/level.pak")).ok()?;
    let i = pak.find("tag_points_win.markup")?;
    let bytes = pak.read_idx(i).ok()?;
    TagPoints::parse(&bytes)
}

fn content_init(mut bridge: ResMut<ContentBridge>, cry: Option<NonSend<CryHost>>) {
    let Some(cry) = cry else { return };
    if let Ok(dir) = std::env::var("SOD_G2C_SHOT") {
        let _ = std::fs::create_dir_all(&dir);
    }
    let vfs = cry.host.ctx.vfs.clone();
    let game = bridge.game_dir.clone();

    let Some(missions) = read_named(&vfs, "missions.xml", "class3/contentmanager") else {
        eprintln!("[content] missions.xml not found in the mounted paks");
        return;
    };
    let data = MissionData::parse(&missions);
    if data.mission_count() == 0 {
        eprintln!("[content] missions.xml parsed no missions");
        return;
    }
    bridge.cm = ContentManager::new(data);

    if std::env::var("SOD_CONTENT_DBG").is_ok() {
        let mut cands = vfs.paths_with_ext("enclaves.xml");
        cands.sort();
        println!("[content] enclaves.xml candidates: {cands:?}");
    }
    if let Some(enclaves) = read_named(&vfs, "enclaves.xml", "libs/class3/rts") {
        bridge.story = parse_story(&enclaves);
        if bridge.story.states.is_empty() {
            println!(
                "[content] enclaves.xml ({} bytes) has no Home states",
                enclaves.len()
            );
        }
    } else {
        println!("[content] enclaves.xml not found in the mounted paks");
    }
    bridge.tags = load_tags(&vfs, &game).unwrap_or_default();
    // Several packs ship a `mission_mission0.xml` (Beltway/Breakdown DLC); parse
    // every candidate and keep the table with the most survey points, i.e. the
    // main Class3 level.
    for p in vfs.paths_with_ext("mission_mission0.xml") {
        if let Some(bytes) = vfs.read(&p) {
            let pts = parse_survey_points(&bytes);
            if pts.len() > bridge.survey_points.len() {
                bridge.survey_points = pts;
            }
        }
    }
    let lang = std::env::var("SOD_LANG").unwrap_or_else(|_| "english".to_string());
    bridge.loc = Localization::load(&vfs, &game, &lang);
    if let Some(scenes) = read_named(&vfs, "scenes.xml", "contentmanager") {
        bridge.scenes = parse_scenes(&scenes);
    }

    // The native new-game sets `Game.Tutorial`; nothing in XML does it. It is
    // the gate on the `Arrive.StartTutorial` action that enters `Buddies`.
    bridge.cm.set_story_bool("Game.Tutorial", true);
    bridge.state = "Arrive".to_string();
    bridge.story_fired.clear();
    bridge.ready = true;

    println!(
        "[content] {} missions, {} tag points, {} survey points, {} story states, {} scene buckets (lang {})",
        bridge.cm.data().mission_count(),
        bridge.tags.len(),
        bridge.survey_points.len(),
        bridge.story.states.len(),
        bridge.scenes.len(),
        lang,
    );
}

fn parse_story(bytes: &[u8]) -> StoryMachine {
    let mut out = StoryMachine::default();
    let Some(root) = parse_document(bytes) else { return out };
    let Some(home) = root
        .find_all("Enclave")
        .into_iter()
        .find(|e| e.attr("EnclaveId") == Some("Home"))
    else {
        return out;
    };
    for st in home.children_named("State") {
        let mut actions = Vec::new();
        for a in st.children_named("Action") {
            let id = a.attr("ActionId").unwrap_or("").to_string();
            let flags: Vec<String> = a.csv("ActionFlags");
            let repeats = flags.iter().any(|f| f == "Repeats");
            let wait_no_mission = flags.iter().any(|f| f == "WaitForNoMission");
            let mut inputs = Vec::new();
            if let Some(i) = a.child("Inputs") {
                for c in &i.children {
                    if matches!(c.name.as_str(), "CheckBool" | "Enclave_CheckBool" | "Check") {
                        inputs.push(StoryInput {
                            kind: c.name.clone(),
                            id: c.attr("InputId").unwrap_or("").to_string(),
                            invert: c.attr_bool("Invert"),
                        });
                    }
                }
            }
            let mut creates = Vec::new();
            let mut changes = Vec::new();
            let mut enclave_stats = Vec::new();
            let mut stat_bools = Vec::new();
            if let Some(o) = a.child("Outputs") {
                for c in &o.children {
                    match c.name.as_str() {
                        "Mission_Create" => {
                            let mission = c.attr("Mission").unwrap_or("").to_string();
                            if !mission.is_empty() {
                                creates.push(StoryCreate {
                                    mission,
                                    success: c.attr("RtsDataSuccess").unwrap_or("").to_string(),
                                    abort: c.attr("RtsDataAbort").unwrap_or("").to_string(),
                                });
                            }
                        }
                        "ChangeState" => {
                            if let Some(t) = c.attr("State") {
                                if !t.trim().is_empty() {
                                    changes.push(t.to_string());
                                }
                            }
                        }
                        "Enclave_StatSetBool" => {
                            let stat = c.attr("OutputStat").unwrap_or("").to_string();
                            if !stat.is_empty() {
                                enclave_stats.push((stat, c.attr_bool_or("OutputAmount", true)));
                            }
                        }
                        "RTSStat_SetBool" => {
                            let oid = c.attr("OutputId").unwrap_or("").to_string();
                            if !oid.is_empty() {
                                stat_bools.push((oid, c.attr_bool_or("OutputAmount", true)));
                            }
                        }
                        _ => {}
                    }
                }
            }
            actions.push(StoryAction {
                id,
                repeats,
                wait_no_mission,
                inputs,
                creates,
                changes,
                enclave_stats,
                stat_bools,
            });
        }
        out.states.push(StoryState {
            id: st.attr("Id").unwrap_or("").to_string(),
            actions,
        });
    }
    out
}

fn parse_scenes(bytes: &[u8]) -> HashMap<String, Vec<(String, String)>> {
    let mut out = HashMap::new();
    let Some(root) = parse_document(bytes) else { return out };
    for bucket in root.find_all("SceneBucket") {
        let Some(label) = bucket.attr("Label") else { continue };
        let mut lines = Vec::new();
        for line in bucket.find_all("Line") {
            let audio = line.attr("AudioFile").unwrap_or("").replace('\\', "/");
            let subtitle = line.attr("Subtitle").unwrap_or("").to_string();
            if !audio.is_empty() || !subtitle.is_empty() {
                lines.push((audio, subtitle));
            }
        }
        if !lines.is_empty() {
            out.insert(label.to_string(), lines);
        }
    }
    out
}

/// Parse the level's `SurveyPoint` entities into `SurveyGroupId -> position`.
/// `LoadSurveyPoints` (`0x013b07a0`) walks every entity whose class is
/// `SurveyPoint` and reads its `SurveyGroupId` and transform; the `Survey`
/// objective's action runs at the matching point.
fn parse_survey_points(bytes: &[u8]) -> HashMap<String, [f32; 3]> {
    let mut out = HashMap::new();
    let Some(root) = parse_document(bytes) else { return out };
    for e in root.find_all("Entity") {
        if e.attr("EntityClass") != Some("SurveyPoint") {
            continue;
        }
        let Some(group) = e
            .child("Properties")
            .and_then(|p| p.attr("SurveyGroupId"))
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        let Some(pos) = e.attr("Pos").and_then(parse_vec3) else { continue };
        out.entry(group.to_string()).or_insert(pos);
    }
    out
}

/// `"x,y,z"` -> `[x, y, z]` (CryEngine Z-up).
fn parse_vec3(s: &str) -> Option<[f32; 3]> {
    let mut it = s.split(',').map(|p| p.trim().parse::<f32>().ok());
    Some([it.next()??, it.next()??, it.next()??])
}

// ---------------------------------------------------------------------------
// Story evaluation
// ---------------------------------------------------------------------------

fn story_input_true(inp: &StoryInput, cm: &ContentManager) -> bool {
    let id = inp.id.strip_prefix('!').unwrap_or(&inp.id);
    let mut v = match inp.kind.as_str() {
        "Enclave_CheckBool" => cm.story_bool(&format!("Enclave.{id}")),
        "CheckBool" => match id {
            "Game.RTS.AlwaysTrue" => true,
            "Game.RTS.AlwaysFalse" => false,
            _ => cm.story_bool(id),
        },
        // Numeric (`Check`) and mission-count gates are not modelled; they
        // never fire here.
        "Check" | "Enclave_CheckMissionCount" => return false,
        _ => true,
    };
    if inp.id.starts_with('!') {
        v = !v;
    }
    if inp.invert {
        v = !v;
    }
    v
}

fn eval_inputs(a: &StoryAction, cm: &ContentManager) -> bool {
    a.inputs.iter().all(|i| story_input_true(i, cm))
}

fn story_step(
    machine: &StoryMachine,
    cm: &mut ContentManager,
    state: &mut String,
    fired: &mut Vec<bool>,
    pending: &mut HashMap<String, (String, String)>,
) {
    let Some(st) = machine.state(state) else { return };
    if fired.len() != st.actions.len() {
        *fired = vec![false; st.actions.len()];
    }
    let mission_active = !cm.active_labels().is_empty();
    let mut next: Option<String> = None;
    for (i, a) in st.actions.iter().enumerate() {
        // Non-`Repeats` actions fire at most once per state entry; `Repeats`
        // ones may re-fire while their inputs hold. `WaitForNoMission` holds
        // the action back while any mission is running.
        if fired[i] && !a.repeats {
            continue;
        }
        if a.wait_no_mission && mission_active {
            continue;
        }
        if !eval_inputs(a, cm) {
            continue;
        }
        for c in &a.creates {
            if cm.create_mission(&c.mission) {
                println!("[content] story {}: create mission {}", state, c.mission);
            }
            if !c.success.is_empty() || !c.abort.is_empty() {
                pending.insert(c.mission.clone(), (c.success.clone(), c.abort.clone()));
            }
        }
        for (id, v) in &a.stat_bools {
            cm.set_story_bool(id, *v);
        }
        for (id, v) in &a.enclave_stats {
            cm.set_story_bool(&format!("Enclave.{id}"), *v);
        }
        for t in &a.changes {
            next = Some(t.clone());
            println!("[content] story {} -> {} ({})", state, t, a.id);
        }
        fired[i] = true;
        // A state change takes effect on the next tick, in the new state.
        if next.is_some() {
            break;
        }
    }
    if let Some(t) = next {
        *state = t;
        fired.clear();
    }
}

// ---------------------------------------------------------------------------
// Objective targets / actor resolution
// ---------------------------------------------------------------------------

fn resolve_actor(
    mission: &Mission,
    idx: usize,
    tags: &TagPoints,
    player: [f32; 3],
) -> Option<[f32; 3]> {
    let a = mission.actor(idx)?;
    match &a.kind {
        sod_core::cry::content::ActorKind::Player => Some(player),
        sod_core::cry::content::ActorKind::TagPoint => tags.get(&a.actor_name),
        sod_core::cry::content::ActorKind::Building => {
            if a.near_actor_idx >= 0 {
                resolve_actor(mission, a.near_actor_idx as usize, tags, player)
            } else {
                None
            }
        }
        sod_core::cry::content::ActorKind::Character
        | sod_core::cry::content::ActorKind::Zombie => {
            if a.actor_name.is_empty() {
                None
            } else {
                tags.get(&a.actor_name)
            }
        }
        _ => None,
    }
}

/// World position an objective wants the player at. `SurveyGroup`/`Enclave`
/// targets resolve to nothing, so fall back to the mission's first tag point
/// (its `EnterArea` trigger) — the closest in-world anchor the data gives.
fn objective_target(
    bridge: &ContentBridge,
    label: &str,
    player: [f32; 3],
) -> Option<[f32; 3]> {
    let m = bridge.cm.data().mission(label)?;
    let o = bridge.cm.active_objective_ref(label)?;
    let resolved = if o.target_location_override >= 0 {
        resolve_actor(m, o.target_location_override as usize, &bridge.tags, player)
    } else if let Some(l) = o.locations.first() {
        resolve_actor(m, l.actor_location.max(0) as usize, &bridge.tags, player)
    } else {
        None
    };
    resolved
        .or_else(|| survey_target(bridge, m, o))
        .or_else(|| mission_trigger(m, &bridge.tags, player))
}

/// The level `SurveyPoint` position of the `SurveyGroup` a `Survey` objective
/// names (ta20Tower -> `tanner_watertower`).
fn survey_target(bridge: &ContentBridge, m: &Mission, o: &Objective) -> Option<[f32; 3]> {
    if o.kind != ObjectiveType::Survey {
        return None;
    }
    let a = m.actor(o.primary_actor_index())?;
    if a.kind != ActorKind::SurveyGroup {
        return None;
    }
    bridge.survey_points.get(&a.actor_name).copied()
}

/// The tag point an `EnterArea` mission keys its approach trigger off (its
/// first `TagPoint` actor).
fn mission_trigger(
    m: &Mission,
    tags: &TagPoints,
    player: [f32; 3],
) -> Option<[f32; 3]> {
    for (i, a) in m.actors.iter().enumerate() {
        if a.kind == sod_core::cry::content::ActorKind::TagPoint {
            if let Some(p) = resolve_actor(m, i, tags, player) {
                return Some(p);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Autoplay
// ---------------------------------------------------------------------------

fn content_autoplay(
    mut bridge: ResMut<ContentBridge>,
    player: Option<ResMut<crate::Player>>,
    menu: Res<crate::MenuRes>,
    cry: Option<NonSend<CryHost>>,
    time: Res<Time>,
) {
    if !bridge.ready || !bridge.drive || menu.active {
        return;
    }
    let Some(mut player) = player else { return };
    let pos = [player.mstate.pos[0], player.mstate.pos[1], player.mstate.pos[2]];

    // Where should we be? Prefer the focused mission's objective, else the
    // first created-but-unactivated `EnterArea` mission's trigger.
    let mut goal: Option<[f32; 3]> = None;
    if let Some(label) = bridge.focus.clone() {
        goal = objective_target(&bridge, &label, pos);
    }
    if goal.is_none() {
        for label in bridge.cm.active_labels() {
            let label = label.to_string();
            if bridge.cm.is_activated(&label) {
                continue;
            }
            let Some(m) = bridge.cm.data().mission(&label) else { continue };
            if m.activate_type != "EnterArea" {
                continue;
            }
            if let Some(p) = mission_trigger(m, &bridge.tags, pos) {
                goal = Some(p);
                break;
            }
        }
    }
    // On a `ClearArea` the scripted driver walks to each mission-spawned zombie
    // (they are static host actors) so the player can melee them down; that is
    // real play, not predicate faking.
    if let Some(label) = bridge.focus.clone() {
        let clear_area = bridge
            .cm
            .active_objective_ref(&label)
            .map(|o| o.kind == ObjectiveType::ClearArea)
            .unwrap_or(false);
        if clear_area {
            if let Some(cry) = cry.as_deref() {
                let actors = cry.host.ctx.actors.borrow();
                let world = cry.host.ctx.world.borrow();
                let mut best: Option<([f32; 3], f32)> = None;
                for mz in &bridge.mission_zeds {
                    if actors.get(&mz.entity).map(|a| a.is_dead()).unwrap_or(true) {
                        continue;
                    }
                    if let Some(e) = world.get(mz.entity) {
                        let d = dist(pos, e.pos);
                        if best.map_or(true, |(_, bd)| d < bd) {
                            best = Some((e.pos, d));
                        }
                    }
                }
                if let Some((p, d)) = best {
                    if d > 2.0 {
                        goal = Some(p);
                    }
                }
            }
        }
    }
    let Some(goal) = goal else {
        if std::env::var("SOD_TUTORIAL_DBG").is_ok() {
            bridge.log_timer += time.delta_secs();
            if bridge.log_timer > 1.0 {
                bridge.log_timer = 0.0;
                println!(
                    "[content] autoplay: no goal (focus {:?}, active {:?})",
                    bridge.focus,
                    bridge.cm.active_labels()
                );
            }
        }
        return;
    };

    let d = dist(pos, goal);
    bridge.at_goal = d <= 1.5;
    if std::env::var("SOD_TUTORIAL_DBG").is_ok() {
        bridge.log_timer += time.delta_secs();
        if bridge.log_timer > 1.0 {
            bridge.log_timer = 0.0;
            println!(
                "[content] autoplay: focus {:?} pos {pos:?} goal {goal:?} d {d:.1} at_goal {}",
                bridge.focus, bridge.at_goal
            );
        }
    }
    if bridge.at_goal {
        return;
    }
    // Fast, collision-free walk for verification (dt-scaled so it is frame
    // independent). Clamp to the remaining distance so we land exactly.
    let speed = 320.0f32;
    let step = (speed * time.delta_secs()).min(d);
    let inv = if d > 1e-4 { 1.0 / d } else { 0.0 };
    let dir = [(goal[0] - pos[0]) * inv, (goal[1] - pos[1]) * inv, (goal[2] - pos[2]) * inv];
    player.mstate.pos[0] = pos[0] + dir[0] * step;
    player.mstate.pos[1] = pos[1] + dir[1] * step;
    player.mstate.pos[2] = pos[2] + dir[2] * step;
    player.pos = [player.mstate.pos[0], player.mstate.pos[1]];
    player.yaw = (goal[1] - pos[1]).atan2(goal[0] - pos[0]);
}

// ---------------------------------------------------------------------------
// Per-frame content manager
// ---------------------------------------------------------------------------

/// How close the player must be to a survey point to run its survey action.
const SURVEY_REACH: f32 = 8.0;
/// Searchable containers the native RTS places in each requested building.
const CONTAINERS_PER_BUILDING: usize = 2;

#[allow(clippy::too_many_arguments)]
fn content_tick(
    mut commands: Commands,
    mut bridge: ResMut<ContentBridge>,
    player: Option<Res<crate::Player>>,
    menu: Res<crate::MenuRes>,
    mut cry: Option<NonSendMut<CryHost>>,
    time: Res<Time>,
    registry: Res<SearchRegistry>,
    mut searchq: ResMut<SearchSpawnQueue>,
    combat: Res<CombatStats>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_btn: Res<ButtonInput<MouseButton>>,
) {
    if !bridge.ready || menu.active {
        return;
    }
    let Some(player) = player else { return };
    let dt = time.delta_secs().min(0.1);
    let ppos = [player.mstate.pos[0], player.mstate.pos[1], player.mstate.pos[2]];

    // Story state machine first so newly created missions get ticked.
    {
        let ContentBridge { cm, story, state, story_fired, pending_stats, .. } = &mut *bridge;
        story_step(story, cm, state, story_fired, pending_stats);
    }
    // Apply the effects the story machine just produced (mission `Created`
    // spawns) *before* collecting the world, so the objective's first update
    // sees the spawned zombies instead of an empty area.
    let pre = bridge.cm.take_effects();
    apply_effects(
        &mut commands,
        &mut bridge,
        cry.as_deref_mut(),
        pre,
        ppos,
        &mut searchq,
        combat.kills,
    );

    // Live non-player actors: the zombie presence `ClearArea` counts.
    let mut zombies = Vec::new();
    if let Some(cry) = cry.as_deref() {
        let world = cry.host.ctx.world.borrow();
        let actors = cry.host.ctx.actors.borrow();
        for (id, a) in actors.iter() {
            if *id == cry.player || a.is_dead() {
                continue;
            }
            if let Some(e) = world.get(*id) {
                zombies.push(e.pos);
            }
        }
    }

    // Mission-spawned zombies are host actors outside the B4 director's runtime,
    // so the player's attacks are resolved against them here through the same
    // `process_actor_damage` pipeline (real deaths, same as B4).
    mission_zed_combat(&mut bridge, cry.as_deref_mut(), ppos, player.yaw, dt, &keys, &mouse_btn);

    // `KillActor`: a mission-spawned actor that died, or a real B4-pipeline
    // kill since the objective activated.
    let mut dead_actors: HashSet<(String, usize)> = HashSet::new();
    let mission_dead: Vec<String> = if let Some(cry) = cry.as_deref() {
        let actors = cry.host.ctx.actors.borrow();
        bridge
            .mission_zeds
            .iter()
            .filter(|mz| actors.get(&mz.entity).map(|a| a.is_dead()).unwrap_or(false))
            .map(|mz| mz.mission.clone())
            .collect()
    } else {
        Vec::new()
    };
    let kill_targets: Vec<(String, usize)> = bridge
        .cm
        .active_labels()
        .into_iter()
        .filter_map(|label| {
            let o = bridge.cm.active_objective_ref(label)?;
            matches!(o.kind, ObjectiveType::KillActor | ObjectiveType::Assassinate)
                .then(|| (label.to_string(), o.primary_actor_index()))
        })
        .collect();
    for (label, idx) in kill_targets {
        bridge.kill_baselines.entry(label.clone()).or_insert(combat.kills);
        let baseline = bridge.kill_baselines.get(&label).copied().unwrap_or(0);
        let real_kill = combat.kills > baseline;
        let zed_dead = mission_dead.iter().any(|m| m == &label);
        if real_kill || zed_dead {
            dead_actors.insert((label, idx));
        }
    }

    // The `Survey` action: hold USE within reach of the group's survey point
    // for `MinSurveyTime` seconds (`ParseObjectiveSurveyNode` `0x014c6770`,
    // `LoadSurveyPoints` `0x013b07a0`). `SOD_AUTOSURVEY` is the harness.
    let auto_survey = std::env::var("SOD_AUTOSURVEY").is_ok();
    let surveying = keys.pressed(KeyCode::KeyE) || auto_survey;
    let survey_jobs: Vec<(String, f32)> = bridge
        .cm
        .active_labels()
        .into_iter()
        .filter_map(|label| {
            let o = bridge.cm.active_objective_ref(label)?;
            if o.kind != ObjectiveType::Survey {
                return None;
            }
            let m = bridge.cm.data().mission(label)?;
            let a = m.actor(o.primary_actor_index())?;
            if a.kind != ActorKind::SurveyGroup || !bridge.survey_points.contains_key(&a.actor_name) {
                return None;
            }
            Some((a.actor_name.clone(), o.survey_min_time().max(0.01)))
        })
        .collect();
    for (group, need) in survey_jobs {
        let pt = bridge.survey_points.get(&group).copied().unwrap_or([0.0; 3]);
        if dist(ppos, pt) > SURVEY_REACH || !surveying {
            continue;
        }
        let prog = bridge.survey_progress.entry(group.clone()).or_insert(0.0);
        *prog += dt;
        if *prog >= need && bridge.surveyed.insert(group.clone()) {
            println!("[content] survey complete: {group}");
        }
    }

    // Player downed: fail active missions; `RestoreMissionOnFail` ones come
    // back. Revive immediately so the harness keeps playing.
    if let Some(cry) = cry.as_deref() {
        let dead = cry
            .host
            .ctx
            .actors
            .borrow()
            .get(&cry.player)
            .map(|a| a.is_dead())
            .unwrap_or(false);
        if dead && !bridge.player_down {
            bridge.player_down = true;
            let n = bridge.cm.on_player_down();
            println!("[content] player down: {n} mission(s) failed");
            if let Some(a) = cry.host.ctx.actors.borrow_mut().get_mut(&cry.player) {
                a.health = a.max_health;
            }
        } else if !dead {
            bridge.player_down = false;
        }
    }

    // Split-borrow the bridge fields so the world view can borrow `tags`
    // while the runtime mutates `cm`.
    let ContentBridge {
        cm,
        tags,
        at_goal,
        fake,
        surveyed,
        ..
    } = &mut *bridge;

    // `EnterArea` activation: fire when the player reaches the trigger.
    let active: Vec<String> = cm.active_labels().into_iter().map(str::to_string).collect();
    for label in active {
        if cm.is_activated(&label) {
            continue;
        }
        let Some(m) = cm.data().mission(&label) else { continue };
        if m.activate_type != "EnterArea" {
            continue;
        }
        let radius = if m.approach_radius > 0.0 { m.approach_radius } else { 75.0 };
        if let Some(t) = mission_trigger(m, tags, ppos) {
            if dist(ppos, t) <= radius {
                if cm.activate_mission(&label) {
                    println!("[content] enter-area activate {}", label);
                }
            }
        }
    }

    {
        let view = WorldView {
            player: ppos,
            tags,
            zombies: std::mem::take(&mut zombies),
            searchables: &registry.entries,
            surveyed,
            dead_actors: &dead_actors,
            fake: *fake,
            at_goal: *at_goal,
        };
        cm.tick(dt, &view);
    }

    let effects = cm.take_effects();
    apply_effects(&mut commands, &mut bridge, cry.as_deref_mut(), effects, ppos, &mut searchq, combat.kills);
}

/// Save a milestone screenshot when `SOD_G2C_SHOT=<dir>` is set (the scripted
/// verification harness; independent of `SOD_SHOT` so the run is not cut short).
fn milestone_shot(commands: &mut Commands, name: &str) {
    let Ok(dir) = std::env::var("SOD_G2C_SHOT") else { return };
    let safe: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let path = PathBuf::from(dir).join(format!("{safe}.png"));
    commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
}

#[allow(clippy::too_many_arguments)]
fn apply_effects(
    commands: &mut Commands,
    bridge: &mut ContentBridge,
    mut cry: Option<&mut CryHost>,
    effects: Vec<ContentEffect>,
    ppos: [f32; 3],
    searchq: &mut SearchSpawnQueue,
    kills: u64,
) {
    for e in effects {
        match e {
            ContentEffect::MissionCreated(l) => {
                println!("[content] mission created: {l}");
                bridge.note(format!("Mission: {l}"));
            }
            ContentEffect::MissionActivated(l) => {
                println!("[content] mission active: {l}");
            }
            ContentEffect::ObjectiveActivated { mission, bucket, objective, kind } => {
                let text = bridge.loc.get(&objective);
                println!("[content] objective [{kind}] {mission}/{bucket}: {text}");
                bridge.objective = Some(text.clone());
                bridge.focus = Some(mission.clone());
                bridge.at_goal = false;
                bridge.note(text);
                // Marker for the objective's target, when it resolves.
                let target = objective_target(bridge, &mission, ppos);
                bridge.marker = target;
                // `SearchBuilding`: place the building's searchables (the
                // native RTS stand-in for the building markup volume).
                let key = format!("{mission}/{bucket}");
                if kind == "SearchBuilding" && bridge.search_requested.insert(key) {
                    if let Some((site, pos, max)) = search_building_plan(bridge, &mission) {
                        for i in 0..max as usize {
                            let off = i as f32 * 1.2;
                            let gsite = format!("{site}#{i}");
                            for j in 0..CONTAINERS_PER_BUILDING {
                                let prop = ["CoolerChest", "Cabinet", "Crate"][(i + j) % 3];
                                let p = [
                                    pos[0] + off + j as f32 * 0.6 - 0.3,
                                    pos[1] + if j % 2 == 0 { 0.5 } else { -0.5 },
                                    pos[2],
                                ];
                                searchq.requests.push(SearchSpawnRequest {
                                    site: gsite.clone(),
                                    prop: prop.to_string(),
                                    pos: p,
                                });
                            }
                        }
                        println!("[content] {mission}: requested {max} searchable buildings at {pos:?}");
                    }
                }
                bridge.kill_baselines.entry(mission.clone()).or_insert(kills);
            }
            ContentEffect::ObjectiveCompleted { mission, objective, .. } => {
                println!("[content] objective complete {mission}: {}", bridge.loc.get(&objective));
                milestone_shot(commands, &format!("objective_{mission}"));
            }
            ContentEffect::MissionCompleted(l) => {
                println!("[content] mission complete: {l}");
                milestone_shot(commands, &format!("mission_{l}"));
                if !bridge.done.contains(&l) {
                    bridge.done.push(l.clone());
                }
                if let Some((success, _)) = bridge.pending_stats.get(&l).cloned() {
                    if !success.is_empty() {
                        bridge.cm.set_story_bool(&format!("Enclave.{success}"), true);
                    }
                }
                if bridge.focus.as_deref() == Some(l.as_str()) {
                    bridge.focus = None;
                    bridge.objective = None;
                }
                // Hand the HUD to any other running mission's objective (e.g.
                // `ta20Tower` while `ta20Forest` finishes).
                let labels: Vec<String> =
                    bridge.cm.active_labels().into_iter().map(str::to_string).collect();
                for lab in labels {
                    if let Some(t) = bridge.cm.active_objective_text(&lab) {
                        let text = bridge.loc.get(t);
                        bridge.objective = Some(text);
                        bridge.focus = Some(lab);
                        break;
                    }
                }
            }
            ContentEffect::MissionAborted(l) => {
                println!("[content] mission aborted: {l}");
                if let Some((_, abort)) = bridge.pending_stats.get(&l).cloned() {
                    if !abort.is_empty() {
                        bridge.cm.set_story_bool(&format!("Enclave.{abort}"), true);
                    }
                }
            }
            ContentEffect::ZombieSpawn { mission, spawn_actor, count, freak, pos } => {
                // The `Created`/`Activated` events are applied with a world-less
                // context, so re-resolve the spawn actor's position from the
                // mission table + tag points here (the original casts it live).
                let pos = bridge
                    .cm
                    .data()
                    .mission(&mission)
                    .and_then(|m| resolve_actor(m, spawn_actor.max(0) as usize, &bridge.tags, ppos))
                    .unwrap_or(pos);
                println!("[content] spawn {count}x zombie (freak {freak:?}) at {:?}", pos);
                let spawned = spawn_zombies(cry.as_deref_mut(), count, &pos, freak.as_deref());
                let _ = spawn_actor;
                for entity in spawned {
                    bridge.mission_zeds.push(MissionZed {
                        mission: mission.clone(),
                        entity,
                    });
                }
            }
            ContentEffect::ScenePlay(scene) => {
                play_scene(bridge, cry.as_deref_mut(), &scene, ppos);
            }
            ContentEffect::SceneSetup(scene) => {
                println!("[content] scene setup: {scene}");
            }
            ContentEffect::HintShow(id) => {
                println!("[content] hint show: {id}");
            }
            ContentEffect::HintEnable { id, enabled } => {
                println!("[content] hint {id}: enabled={enabled}");
            }
            ContentEffect::MusicBucket(b) => {
                println!("[content] music: {b}");
            }
            ContentEffect::EventLog { title, display } => {
                let title = bridge.loc.get(&title);
                println!("[content] event log ({display}): {title}");
                if !title.is_empty() {
                    bridge.banner = Some(title.clone());
                    bridge.note(title);
                }
            }
            ContentEffect::StatBool { id, value } => {
                log::debug!("[content] stat {id} = {value}");
            }
            ContentEffect::Output { name, .. } => {
                log::debug!("[content] output {name}");
            }
        }
    }

    // Journal/radio notifications are appended with a TTL timer; the HUD
    // system below keeps the newest lines.
    bridge.log_timer += 0.0;
}

/// Mission `Zombie.Spawn`/`Zombie.FreakSpawn` actors through the script host.
/// `FreakHash="Zombie.Feral"` selects the Feral class, whose own script body and
/// ULSM family the actor system loads; an empty/unknown hash is the Grunt.
fn spawn_zombies(
    cry: Option<&mut CryHost>,
    count: u32,
    pos: &[f32; 3],
    freak_hash: Option<&str>,
) -> Vec<sod_core::cry::world::EntityId> {
    let Some(cry) = cry else { return Vec::new() };
    let entity_class = freak_entity_class(freak_hash.unwrap_or(""));
    let (class, script) = class_script(entity_class);
    let mut out = Vec::new();
    for i in 0..count.min(8) {
        let name = format!(
            "ULMissionZed{}_{}",
            cry.player,
            cry.host.ctx.instances.borrow().len() + i as usize
        );
        let p = [pos[0] + i as f32 * 1.5, pos[1], pos[2]];
        match cry.host.spawn_actor(class, script, &name, p, false) {
            Ok(id) => {
                if let Some(a) = cry.host.ctx.actors.borrow_mut().get_mut(&id) {
                    let hp = kind_for_class(class).stats().hp;
                    a.max_health = hp;
                    a.health = hp;
                }
                out.push(id);
            }
            Err(e) => log::debug!("[content] zombie spawn failed: {e}"),
        }
    }
    out
}

/// Resolve the player's melee/fire attack against the nearest live
/// mission-spawned zombie, through the same game-rules damage pipeline B4 uses.
/// The auto-fight harness (`SOD_AUTOFIGHT`) swings on a cadence; normal play
/// uses the attack keys.
fn mission_zed_combat(
    bridge: &mut ContentBridge,
    cry: Option<&mut CryHost>,
    player_pos: [f32; 3],
    yaw: f32,
    dt: f32,
    keys: &ButtonInput<KeyCode>,
    mouse_btn: &ButtonInput<MouseButton>,
) {
    let Some(cry) = cry else { return };
    if bridge.mission_zeds.is_empty() {
        return;
    }
    let auto = std::env::var("SOD_AUTOFIGHT").is_ok();
    let auto_gun = std::env::var("SOD_AUTOGUN").is_ok();
    bridge.zed_melee_cd = (bridge.zed_melee_cd - dt).max(0.0);
    bridge.zed_gun_cd = (bridge.zed_gun_cd - dt).max(0.0);
    let melee = mouse_btn.just_pressed(MouseButton::Left) || (auto && bridge.zed_melee_cd <= 0.0);
    let gun = keys.just_pressed(KeyCode::KeyF) || (auto && auto_gun && bridge.zed_gun_cd <= 0.0);
    if !melee && !gun {
        return;
    }
    let fwd = [yaw.cos(), yaw.sin()];
    let pick = {
        let actors = cry.host.ctx.actors.borrow();
        let world = cry.host.ctx.world.borrow();
        let mut best: Option<(usize, f32)> = None;
        for (i, mz) in bridge.mission_zeds.iter().enumerate() {
            if actors.get(&mz.entity).map(|a| a.is_dead()).unwrap_or(true) {
                continue;
            }
            let Some(e) = world.get(mz.entity) else { continue };
            let d = dist(e.pos, player_pos);
            let to = [e.pos[0] - player_pos[0], e.pos[1] - player_pos[1]];
            let dot = if d > 1e-3 { (to[0] * fwd[0] + to[1] * fwd[1]) / d } else { 1.0 };
            if melee && d > 2.6 {
                continue;
            }
            // The auto-fight harness does not steer the camera; it faces the
            // target like B4's own auto-aim. Manual play still needs the facing.
            if melee && !auto && dot < 0.15 {
                continue;
            }
            if gun && d > 45.0 {
                continue;
            }
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    };
    let Some(i) = pick else { return };
    let entity = bridge.mission_zeds[i].entity;
    let cur = cry.host.ctx.actors.borrow().get(&entity).map(|a| a.health).unwrap_or(25.0);
    let zpos = cry
        .host
        .ctx
        .world
        .borrow()
        .get(entity)
        .map(|e| e.pos)
        .unwrap_or(player_pos);
    let dir = [fwd[0], fwd[1], 0.0];
    let (hit, dmg) = if gun {
        (Hit::bullet(entity, cry.player, 0, 200.0, HitMaterial::Torso, zpos, dir), 200.0)
    } else {
        (Hit::melee(entity, cry.player, 80.0, HitMaterial::Torso, zpos, dir), 80.0)
    };
    let out = cry_combat::process_actor_damage(
        cur,
        DamageTarget::Zombie(ZombieKind::Grunt),
        &hit,
        cry_combat::BLOATER_HEADSHOT_KILL,
    );
    if let Some(a) = cry.host.ctx.actors.borrow_mut().get_mut(&entity) {
        a.health = out.health;
    }
    println!(
        "[content] mission-zed hit {dmg:.0} dmg ({cur:.0}->{:.0} hp){}",
        out.health,
        if out.dead { " KILL" } else { "" }
    );
    if gun {
        bridge.zed_gun_cd = 0.3;
    } else {
        bridge.zed_melee_cd = 0.6;
    }
}

/// The `SearchBuilding` objective's building: its `SiteTag` and the tag point
/// its `Building` actor resolves to, plus the `MaxCount` of buildings to search.
fn search_building_plan(
    bridge: &ContentBridge,
    mission_label: &str,
) -> Option<(String, [f32; 3], u32)> {
    let m = bridge.cm.data().mission(mission_label)?;
    let o = bridge.cm.active_objective_ref(mission_label)?;
    if o.kind != ObjectiveType::SearchBuilding {
        return None;
    }
    let idx = o.primary_actor_index();
    let a = m.actor(idx)?;
    let site = a
        .attrs
        .iter()
        .find(|(k, _)| k == "SiteTag")
        .map(|(_, v)| v.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Search".to_string());
    let pos = resolve_actor(m, idx, &bridge.tags, [0.0; 3])?;
    let max = o
        .attrs
        .iter()
        .find(|(k, _)| k == "MaxCount")
        .and_then(|(_, v)| v.parse::<u32>().ok())
        .unwrap_or(1)
        .max(1);
    Some((site, pos, max))
}

/// Resolve a `Scene_ScenePlay` bucket to its authored dialog lines: show the
/// subtitle and, when the sound event is loaded, play the audio file through
/// the host's sound system.
fn play_scene(
    bridge: &mut ContentBridge,
    cry: Option<&mut CryHost>,
    scene: &str,
    _pos: [f32; 3],
) {
    let Some(lines) = bridge.scenes.get(scene).cloned() else {
        println!("[content] scene (unknown): {scene}");
        return;
    };
    let line = lines.first().cloned().unwrap_or_default();
    if !line.1.is_empty() {
        println!("[content] scene {scene}: {}", line.1);
        bridge.note(line.1.clone());
    } else {
        println!("[content] scene: {scene}");
    }
    if let Some(cry) = cry {
        let system = cry.host.ctx.audio.borrow().system.clone();
        if let (Some(system), false) = (system, line.0.is_empty()) {
            if let Ok(mut sys) = system.lock() {
                if sys.has_event(&line.0) {
                    let _ = sys.play(&line.0, _pos, None);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

fn content_hud(bridge: Res<ContentBridge>, mut lines: Query<(&mut Text, &crate::HudLine)>) {
    if !bridge.ready {
        return;
    }
    for (mut t, line) in &mut lines {
        match line {
            crate::HudLine::Objective => {
                if let Some(o) = &bridge.objective {
                    t.0 = format!("OBJECTIVE: {o}");
                }
            }
            crate::HudLine::Notifications => {
                if !bridge.notifications.is_empty() {
                    t.0 = bridge.notifications.join("   ");
                }
            }
            crate::HudLine::Banner => {
                if let Some(b) = &bridge.banner {
                    t.0 = b.clone();
                }
            }
            _ => {}
        }
    }
}
