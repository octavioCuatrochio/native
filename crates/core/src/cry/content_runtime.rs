//! Native content-manager runtime: mission lifecycle, objective evaluation and
//! the story-flag effects that chain the tutorial together.
//!
//! This is the Wine-free stand-in for `NSContentManager::CContentManager`
//! (Class3 `UL\ContentManager\ContentManager.cpp`; the relevant routines are
//! `CContentManager::MissionOpAdd` `Class3:01357420`,
//! `ChooseObjectiveFromMissionPhase` `Class3:013540d0`,
//! `CObjectiveBase::Activate` `Class3:014c94e0`,
//! `CObjectiveBase::Update` `Class3:014d93a0` and
//! `UpdateLocationEvents` `Class3:014da7d0`). The data it evaluates comes from
//! [`super::content`]; the world it queries is abstracted behind
//! [`MissionWorld`] so the tutorial can be driven head-lessly in tests.
//!
//! Flow recovered from the shipped data and decompilation:
//!
//! 1. `Mission_Create` output (from the `Home` enclave's story state machine,
//!    see `rts/enclaves.xml`) → the mission's `Created` event runs
//!    (spawns, hint setup).
//! 2. `ActivateType="OnAdd"` activates immediately; otherwise the mission is
//!    an offer that waits for its area/interact trigger.
//! 3. The next `<ObjectiveBucket>` in order supplies one objective; its
//!    `Setup`/`Activate` events fire. `ChooseObjectiveFromMissionPhase`
//!    increments `m_activeMissionObjectiveBucketIndex` (`Class3:013540d0`) and
//!    `OnObjectiveComplete` (`Class3:0135a7f0`) walks the buckets, calling
//!    `MissionComplete` (`Class3:01355d40`) once the last one is done.
//! 4. Each tick evaluates the active objective against the world:
//!    `ClearArea` clears when the zombie count at its `<Location>` reaches
//!    `CompleteThreshold`, `LeadNpcs` when the player reaches the target actor,
//!    and every objective runs its `<LocationEvent>`s when the player enters
//!    their radius (these often carry `Mission_CompleteObjective`).
//! 5. Outputs are recorded as [`ContentEffect`]s and selected ones mutate the
//!    run's story-flag table (`RTSStat_SetBool`, `Enclave_StatSetBool`).

use std::collections::HashMap;

use super::content::{
    Action, ActorKind, ActorSpec, Element, Mission, MissionData, Objective, ObjectiveType,
};

/// World queries the mission runtime needs. Implementations resolve mission
/// tag points (from the ULTP markup, see `crate::tagpoints`), the player and
/// any live actors/zombies.
pub trait MissionWorld {
    fn player_pos(&self) -> [f32; 3];
    /// Named tag point (e.g. `RTSPoint_taBuddies_Ambush1`).
    fn tag_point(&self, _name: &str) -> Option<[f32; 3]> {
        None
    }
    /// A `Character`/`Zombie` actor's live position, when the world tracks it.
    fn character_pos(&self, _actor: &ActorSpec) -> Option<[f32; 3]> {
        None
    }
    fn zombie_count_near(&self, _pos: [f32; 3], _radius: f32) -> u32 {
        0
    }
    fn searched_buildings_near(&self, _pos: [f32; 3], _radius: f32) -> u32 {
        0
    }
    fn surveyed(&self, _group: &str) -> bool {
        false
    }
    fn actor_dead(&self, _mission: &str, _actor_index: usize) -> bool {
        false
    }
    /// Completion override for objectives whose native predicate the test or
    /// debug harness cannot produce yet (e.g. `KillActor` when the actor has no
    /// live script handle). Returning `true` completes the objective on this
    /// tick, exactly as a satisfied predicate would. Production worlds leave
    /// the default.
    fn force_objective_complete(&self, _mission: &str, _objective: &Objective) -> bool {
        false
    }
}

/// Observable output of a mission action. The app can map these onto HUD,
/// audio, spawn director and RTS-stat layers.
#[derive(Clone, Debug, PartialEq)]
pub enum ContentEffect {
    MissionCreated(String),
    MissionActivated(String),
    ObjectiveActivated { mission: String, bucket: String, objective: String, kind: &'static str },
    ObjectiveCompleted { mission: String, bucket: String, objective: String },
    MissionCompleted(String),
    MissionAborted(String),
    ScenePlay(String),
    SceneSetup(String),
    HintShow(String),
    HintEnable { id: String, enabled: bool },
    MusicBucket(String),
    StatBool { id: String, value: bool },
    ZombieSpawn { spawn_actor: i32, count: u32, freak: Option<String>, pos: [f32; 3] },
    EventLog { title: String, display: String },
    /// Any output the runtime does not model yet (actor AI, rewards, …).
    Output { name: String, attrs: Vec<(String, String)> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissionRun {
    Active,
    Complete,
    Aborted,
}

struct ActiveObjective {
    spec: Objective,
    bucket_label: String,
    elapsed: f32,
    loc_entered: Vec<bool>,
    timed_fired: Vec<bool>,
}

impl ActiveObjective {
    fn new(bucket_label: String, spec: Objective) -> ActiveObjective {
        ActiveObjective {
            loc_entered: vec![false; spec.location_events.len()],
            timed_fired: vec![false; spec.timed_events.len()],
            bucket_label,
            spec,
            elapsed: 0.0,
        }
    }
}

struct ActiveMission {
    mission: Mission,
    bucket_index: usize,
    objective: Option<ActiveObjective>,
    state: MissionRun,
    activated: bool,
}

impl ActiveMission {
    fn new(mission: Mission) -> ActiveMission {
        ActiveMission {
            mission,
            bucket_index: 0,
            objective: None,
            state: MissionRun::Active,
            activated: false,
        }
    }
}

#[derive(Clone, Debug)]
enum Pending {
    Create(String),
    Abort(String),
}

/// Owner of loaded mission data and the currently running missions.
pub struct ContentManager {
    data: MissionData,
    active: Vec<ActiveMission>,
    completed: Vec<String>,
    aborted: Vec<String>,
    story: HashMap<String, bool>,
    effects: Vec<ContentEffect>,
    pending: Vec<Pending>,
}

impl ContentManager {
    pub fn new(data: MissionData) -> ContentManager {
        ContentManager {
            data,
            active: Vec::new(),
            completed: Vec::new(),
            aborted: Vec::new(),
            story: HashMap::new(),
            effects: Vec::new(),
            pending: Vec::new(),
        }
    }

    pub fn data(&self) -> &MissionData {
        &self.data
    }

    pub fn effects(&self) -> &[ContentEffect] {
        &self.effects
    }

    pub fn take_effects(&mut self) -> Vec<ContentEffect> {
        std::mem::take(&mut self.effects)
    }

    pub fn clear_effects(&mut self) {
        self.effects.clear();
    }

    pub fn story_bool(&self, id: &str) -> bool {
        self.story.get(id).copied().unwrap_or(false)
    }

    pub fn set_story_bool(&mut self, id: &str, value: bool) {
        self.story.insert(id.to_string(), value);
    }

    pub fn completed(&self) -> &[String] {
        &self.completed
    }

    pub fn aborted(&self) -> &[String] {
        &self.aborted
    }

    pub fn is_active(&self, label: &str) -> bool {
        self.active.iter().any(|m| m.mission.label == label && m.state == MissionRun::Active)
    }

    /// Whether an active mission has fired its `Activated` event. `EnterArea`
    /// missions stay `false` until the app triggers their area.
    pub fn is_activated(&self, label: &str) -> bool {
        self.active.iter().any(|m| m.mission.label == label && m.activated)
    }

    /// The active objective of a running mission (its spec), when one is up.
    pub fn active_objective_ref(&self, label: &str) -> Option<&Objective> {
        self.active
            .iter()
            .find(|m| m.mission.label == label)
            .and_then(|m| m.objective.as_ref())
            .map(|o| &o.spec)
    }

    pub fn is_complete(&self, label: &str) -> bool {
        self.completed.iter().any(|l| l == label)
    }

    /// HUD line for a running mission (its active objective's text).
    pub fn active_objective_text(&self, label: &str) -> Option<&str> {
        self.active
            .iter()
            .find(|m| m.mission.label == label)
            .and_then(|m| m.objective.as_ref())
            .map(|o| o.spec.text.as_str())
    }

    pub fn active_labels(&self) -> Vec<&str> {
        self.active.iter().map(|m| m.mission.label.as_str()).collect()
    }

    /// Start a mission: run its `Created` event, then activate it if the data
    /// says `ActivateType="OnAdd"` (or leaves the type empty). Returns `false`
    /// when the label is unknown, already running, or already resolved.
    pub fn create_mission(&mut self, label: &str) -> bool {
        if self.is_active(label)
            || self.completed.iter().any(|l| l == label)
            || self.aborted.iter().any(|l| l == label)
        {
            return false;
        }
        let Some(def) = self.data.mission(label) else { return false };
        let am = ActiveMission::new(def.clone());
        let created = def.event("Created").to_vec();
        self.effects.push(ContentEffect::MissionCreated(label.to_string()));
        {
            let mut ctx = ApplyCtx {
                mission: &am.mission,
                effects: &mut self.effects,
                pending: &mut self.pending,
                story: &mut self.story,
                world: &NONE_WORLD,
                complete: false,
            };
            ctx.run(&created);
        }
        let auto = am.mission.activate_type == "OnAdd" || am.mission.activate_type.is_empty();
        self.active.push(am);
        if auto {
            self.activate_mission(label);
        }
        true
    }

    /// Fire `Activated` and select this mission's first objective. Idempotent.
    pub fn activate_mission(&mut self, label: &str) -> bool {
        let Some(idx) = self
            .active
            .iter()
            .position(|m| m.mission.label == label && m.state == MissionRun::Active)
        else {
            return false;
        };
        if self.active[idx].activated {
            return false;
        }
        self.active[idx].activated = true;
        self.effects.push(ContentEffect::MissionActivated(label.to_string()));
        let actions = self.active[idx].mission.event("Activated").to_vec();
        {
            let am = &mut self.active[idx];
            let mut ctx = ApplyCtx {
                mission: &am.mission,
                effects: &mut self.effects,
                pending: &mut self.pending,
                story: &mut self.story,
                world: &NONE_WORLD,
                complete: false,
            };
            ctx.run(&actions);
        }
        am_choose_objective(
            &mut self.active[idx],
            &mut self.effects,
            &mut self.pending,
            &mut self.story,
        );
        true
    }

    /// Cancel a running mission, running its `Aborted` event.
    pub fn abort_mission(&mut self, label: &str) -> bool {
        let Some(idx) = self
            .active
            .iter()
            .position(|m| m.mission.label == label && m.state == MissionRun::Active)
        else {
            return false;
        };
        let mut am = self.active.remove(idx);
        let actions = am.mission.event("Aborted").to_vec();
        {
            let mut ctx = ApplyCtx {
                mission: &am.mission,
                effects: &mut self.effects,
                pending: &mut self.pending,
                story: &mut self.story,
                world: &NONE_WORLD,
                complete: false,
            };
            ctx.run(&actions);
        }
        am.state = MissionRun::Aborted;
        self.aborted.push(label.to_string());
        self.effects.push(ContentEffect::MissionAborted(label.to_string()));
        true
    }

    /// Advance every running mission one frame.
    pub fn tick(&mut self, dt: f32, world: &dyn MissionWorld) {
        self.process_pending();
        for i in 0..self.active.len() {
            if self.active[i].state != MissionRun::Active {
                continue;
            }
            // `EnterArea` missions are created but wait for the app to fire
            // `activate_mission` once the player enters the area; only then does
            // the first objective phase begin (`Class3:01355920`).
            if !self.active[i].activated {
                continue;
            }
            if self.active[i].objective.is_none() {
                am_choose_objective(
                    &mut self.active[i],
                    &mut self.effects,
                    &mut self.pending,
                    &mut self.story,
                );
            }
            if self.active[i].objective.is_some() {
                am_update_objective(
                    &mut self.active[i],
                    dt,
                    world,
                    &mut self.effects,
                    &mut self.pending,
                    &mut self.story,
                );
            }
        }
        let finished: Vec<String> = self
            .active
            .iter()
            .filter(|m| m.state == MissionRun::Complete)
            .map(|m| m.mission.label.clone())
            .collect();
        for label in finished {
            if !self.completed.contains(&label) {
                self.completed.push(label);
            }
        }
        self.active.retain(|m| m.state == MissionRun::Active);
        self.process_pending();
    }

    fn process_pending(&mut self) {
        for _ in 0..10_000 {
            let pend = std::mem::take(&mut self.pending);
            if pend.is_empty() {
                return;
            }
            for p in pend {
                match p {
                    Pending::Create(l) => {
                        self.create_mission(&l);
                    }
                    Pending::Abort(l) => {
                        self.abort_mission(&l);
                    }
                }
            }
        }
        log::warn!("content: pending output queue did not drain (possible mission cycle)");
    }
}

/// A [`MissionWorld`] that reports nothing. Used while applying lifecycle
/// outputs (position resolution is deferred to a real world when needed).
struct NoneWorld;
impl MissionWorld for NoneWorld {
    fn player_pos(&self) -> [f32; 3] {
        [0.0; 3]
    }
}
static NONE_WORLD: NoneWorld = NoneWorld;

/// Bundles the mutable run state and the immutable mission definition while
/// applying an action's outputs.
struct ApplyCtx<'a> {
    mission: &'a Mission,
    effects: &'a mut Vec<ContentEffect>,
    pending: &'a mut Vec<Pending>,
    story: &'a mut HashMap<String, bool>,
    world: &'a dyn MissionWorld,
    complete: bool,
}

impl<'a> ApplyCtx<'a> {
    fn run(&mut self, actions: &[Action]) {
        for a in actions {
            // Weighted `<Alt>` branches are siblings selected by the parent's
            // `<Alts>`; running them unconditionally would double-apply moods.
            if a.flags.iter().any(|f| f == "Alt") {
                continue;
            }
            if !self.eval_inputs(a) {
                continue;
            }
            for out in &a.outputs {
                self.apply_output(out);
            }
        }
    }

    fn eval_inputs(&self, a: &Action) -> bool {
        for inp in &a.inputs {
            let ok = match inp.name.as_str() {
                "CheckBool" | "Enclave_CheckBool" => self.check_bool(inp),
                _ => true,
            };
            if !ok {
                return false;
            }
        }
        true
    }

    fn check_bool(&self, inp: &Element) -> bool {
        let id = inp.attr("InputId").unwrap_or("");
        let mut v = match id {
            "Game.RTS.AlwaysTrue" => true,
            "Game.RTS.AlwaysFalse" => false,
            _ => self.story.get(id).copied().unwrap_or(false),
        };
        if inp.attr_bool("Invert") {
            v = !v;
        }
        v
    }

    fn apply_output(&mut self, el: &Element) {
        match el.name.as_str() {
            "Mission_CompleteObjective" => self.complete = true,
            "Mission_Create" => {
                let m = el.attr("Mission").unwrap_or("");
                if !m.is_empty() {
                    self.pending.push(Pending::Create(m.to_string()));
                }
            }
            "Mission_AbortMission" => {
                let m = el.attr("Mission").unwrap_or("");
                if !m.is_empty() {
                    self.pending.push(Pending::Abort(m.to_string()));
                }
            }
            "RTSStat_SetBool" => {
                let id = el.attr("OutputId").unwrap_or("").to_string();
                let value = el.attr_bool("OutputAmount");
                if !id.is_empty() {
                    self.story.insert(id.clone(), value);
                    self.effects.push(ContentEffect::StatBool { id, value });
                }
            }
            "Enclave_StatSetBool" => {
                let stat = el.attr("OutputStat").unwrap_or("");
                let value = el.attr_bool("OutputAmount");
                if !stat.is_empty() {
                    let id = format!("Enclave.{stat}");
                    self.story.insert(id.clone(), value);
                    self.effects.push(ContentEffect::StatBool { id, value });
                }
            }
            "Hint_Enable" => {
                let id = el.attr("HintId").unwrap_or("").to_string();
                let enabled = el.attr_bool_or("SetEnabled", true);
                self.effects.push(ContentEffect::HintEnable { id, enabled });
            }
            "Hint_Show" => {
                self.effects.push(ContentEffect::HintShow(el.attr("HintId").unwrap_or("").to_string()));
            }
            "Scene_ScenePlay" => {
                self.effects.push(ContentEffect::ScenePlay(el.attr("Scene").unwrap_or("").to_string()));
            }
            "Scene_SceneSetup" => {
                self.effects.push(ContentEffect::SceneSetup(el.attr("Scene").unwrap_or("").to_string()));
            }
            "Music_PlayBucket" => {
                self.effects
                    .push(ContentEffect::MusicBucket(el.attr("MusicBucket").unwrap_or("").to_string()));
            }
            "Zombie.Spawn" | "Zombie.FreakSpawn" => {
                let spawn_actor = el.attr_i32("SpawnPosActorIndex").unwrap_or(-1);
                let count = el.attr_u32("SpawnCount").unwrap_or(1);
                let freak = el.attr("FreakHash").filter(|s| !s.is_empty()).map(str::to_string);
                let pos = resolve_actor(self.world, self.mission, spawn_actor as usize)
                    .unwrap_or([0.0; 3]);
                self.effects.push(ContentEffect::ZombieSpawn { spawn_actor, count, freak, pos });
            }
            "EventLog" => {
                let title = el
                    .child("NameText")
                    .and_then(|n| n.attr("Text"))
                    .unwrap_or("")
                    .to_string();
                let display = el.attr("DisplayType").unwrap_or("").to_string();
                self.effects.push(ContentEffect::EventLog { title, display });
            }
            other => {
                self.effects
                    .push(ContentEffect::Output { name: other.to_string(), attrs: el.attrs.clone() });
            }
        }
    }
}

fn resolve_actor(world: &dyn MissionWorld, mission: &Mission, idx: usize) -> Option<[f32; 3]> {
    let a = mission.actor(idx)?;
    match &a.kind {
        ActorKind::Player => Some(world.player_pos()),
        ActorKind::TagPoint => world.tag_point(&a.actor_name),
        ActorKind::Building => {
            if a.near_actor_idx >= 0 {
                resolve_actor(world, mission, a.near_actor_idx as usize)
            } else {
                None
            }
        }
        ActorKind::Character | ActorKind::Zombie => world
            .character_pos(a)
            .or_else(|| if a.actor_name.is_empty() { None } else { world.tag_point(&a.actor_name) }),
        _ => None,
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

fn am_choose_objective(
    am: &mut ActiveMission,
    effects: &mut Vec<ContentEffect>,
    pending: &mut Vec<Pending>,
    story: &mut HashMap<String, bool>,
) {
    loop {
        if am.bucket_index >= am.mission.objective_buckets.len() {
            am_finish(am, effects, pending, story);
            return;
        }
        let bucket = &am.mission.objective_buckets[am.bucket_index];
        if bucket.objectives.is_empty() {
            am.bucket_index += 1;
            continue;
        }
        let bucket_label = bucket.label.clone();
        let spec = bucket.objectives[0].clone();
        effects.push(ContentEffect::ObjectiveActivated {
            mission: am.mission.label.clone(),
            bucket: bucket_label.clone(),
            objective: spec.text.clone(),
            kind: spec.kind.name(),
        });
        // Run Setup then Activate with a world-less context (positions resolve
        // later during update); output effects are already recorded.
        for ev in ["Setup", "Activate"] {
            let acts = spec.event(ev).to_vec();
            if acts.is_empty() {
                continue;
            }
            let mut ctx = ApplyCtx {
                mission: &am.mission,
                effects,
                pending,
                story,
                world: &NONE_WORLD,
                complete: false,
            };
            ctx.run(&acts);
        }
        am.objective = Some(ActiveObjective::new(bucket_label, spec));
        return;
    }
}

fn am_finish(
    am: &mut ActiveMission,
    effects: &mut Vec<ContentEffect>,
    pending: &mut Vec<Pending>,
    story: &mut HashMap<String, bool>,
) {
    effects.push(ContentEffect::MissionCompleted(am.mission.label.clone()));
    let acts = am.mission.event("Completed").to_vec();
    {
        let mut ctx = ApplyCtx {
            mission: &am.mission,
            effects,
            pending,
            story,
            world: &NONE_WORLD,
            complete: false,
        };
        ctx.run(&acts);
    }
    am.state = MissionRun::Complete;
}

fn am_update_objective(
    am: &mut ActiveMission,
    dt: f32,
    world: &dyn MissionWorld,
    effects: &mut Vec<ContentEffect>,
    pending: &mut Vec<Pending>,
    story: &mut HashMap<String, bool>,
) {
    let completed = {
        let mission = &am.mission;
        let ao = am.objective.as_mut().expect("objective present");
        let mut ctx = ApplyCtx {
            mission,
            effects,
            pending,
            story,
            world,
            complete: false,
        };
        ao.elapsed += dt;

        // Timed objective events.
        for i in 0..ao.spec.timed_events.len() {
            if ao.timed_fired[i] || ao.elapsed < ao.spec.timed_events[i].timer {
                continue;
            }
            ao.timed_fired[i] = true;
            let acts = ao.spec.timed_events[i].actions.clone();
            ctx.run(&acts);
        }

        // Proximity location events (edge-triggered on the player).
        for i in 0..ao.spec.location_events.len() {
            if ao.loc_entered[i] {
                continue;
            }
            let le = &ao.spec.location_events[i];
            let Some(target) = resolve_actor(world, mission, le.actor_index.max(0) as usize) else {
                continue;
            };
            if dist2(world.player_pos(), target) <= le.radius * le.radius {
                ao.loc_entered[i] = true;
                let acts = le.actions.clone();
                ctx.run(&acts);
            }
        }

        if !ctx.complete {
            ctx.complete = objective_condition_met(&ao.spec, world, mission, ao.elapsed);
        }
        if ctx.complete {
            let acts = ao.spec.event("Complete").to_vec();
            ctx.run(&acts);
        }
        ctx.complete
    };

    if completed {
        let (bucket, text) = am
            .objective
            .as_ref()
            .map(|o| (o.bucket_label.clone(), o.spec.text.clone()))
            .unwrap_or_default();
        effects.push(ContentEffect::ObjectiveCompleted {
            mission: am.mission.label.clone(),
            bucket,
            objective: text,
        });
        am.bucket_index += 1;
        am.objective = None;
        am_choose_objective(am, effects, pending, story);
    }
}

/// Type-specific completion test (the non-`LocationEvent` path). Mirrors the
/// per-type `Update` in `CmObjective.cpp` (`Class3:014d93a0` for the base
/// clear-area/lead path).
fn objective_condition_met(
    spec: &Objective,
    world: &dyn MissionWorld,
    mission: &Mission,
    elapsed: f32,
) -> bool {
    if world.force_objective_complete(&mission.label, spec) {
        return true;
    }
    match spec.kind {
        ObjectiveType::ClearArea => {
            if spec.locations.is_empty() {
                return false;
            }
            spec.locations.iter().all(|l| {
                let Some(p) = resolve_actor(world, mission, l.actor_location.max(0) as usize) else {
                    return false;
                };
                world.zombie_count_near(p, l.radius) <= spec.complete_threshold
            })
        }
        ObjectiveType::LeadNpcs | ObjectiveType::InteractNpc => {
            let target = lead_target(spec);
            let Some(p) = resolve_actor(world, mission, target) else { return false };
            let radius = lead_reach_radius(spec, target);
            dist2(world.player_pos(), p) <= radius * radius
        }
        ObjectiveType::Wait => {
            let wait = spec.attrs.iter().find(|(k, _)| k == "WaitTime").and_then(|(_, v)| v.parse().ok()).unwrap_or(0.0);
            elapsed >= wait
        }
        ObjectiveType::SearchBuilding => {
            let max = spec
                .attrs
                .iter()
                .find(|(k, _)| k == "MaxCount")
                .and_then(|(_, v)| v.parse::<u32>().ok())
                .unwrap_or(1);
            let pos = resolve_actor(world, mission, lead_target(spec)).unwrap_or(world.player_pos());
            let radius = spec
                .attrs
                .iter()
                .find(|(k, _)| k == "Radius")
                .and_then(|(_, v)| v.parse::<f32>().ok())
                .unwrap_or(75.0);
            world.searched_buildings_near(pos, radius) >= max
        }
        ObjectiveType::Survey => {
            let group = spec
                .attrs
                .iter()
                .find(|(k, _)| k == "SurveyGroup")
                .map(|(_, v)| v.as_str())
                .unwrap_or("");
            world.surveyed(group)
        }
        ObjectiveType::KillActor | ObjectiveType::Assassinate => {
            let target = spec
                .attrs
                .iter()
                .find(|(k, _)| k == "TargetActorIndex")
                .and_then(|(_, v)| v.parse::<usize>().ok())
                .or_else(|| {
                    spec.children
                        .iter()
                        .find(|c| c.name == "Target")
                        .and_then(|c| c.attr_i32("ActorIndex"))
                        .map(|i| i.max(0) as usize)
                })
                .unwrap_or(0);
            world.actor_dead(&mission.label, target)
        }
        _ => false,
    }
}

fn lead_target(spec: &Objective) -> usize {
    if spec.target_location_override >= 0 {
        spec.target_location_override as usize
    } else {
        spec.locations
            .first()
            .map(|l| l.actor_location.max(0) as usize)
            .unwrap_or(0)
    }
}

fn lead_reach_radius(spec: &Objective, target: usize) -> f32 {
    let r = spec
        .location_events
        .iter()
        .filter(|le| le.actor_index.max(0) as usize == target && le.radius > 0.0)
        .map(|le| le.radius)
        .fold(f32::INFINITY, f32::min);
    if r.is_finite() {
        r
    } else {
        20.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cry::content::MissionData;

    const MISSION_XML: &str = r#"<RtsMissionData>
      <MissionFrameworkBucket Label="Tanner Arc (ch00)" FragmentFile="Missions_Tanner">
        <MissionFramework Label="ta10Buddies" Type="Story" ActivateType="OnAdd" OpportunityText="Under Attack">
          <MissionActors><Actors>
            <Character ActorTraits="Trait.Story.Buddy" />
            <TagPoint ActorName="RTSPoint_RangerStation" NearActorIdx="-1" />
            <Building NearActorIdx="1" />
            <TagPoint ActorName="RTSPoint_taBuddies_Ambush1" NearActorIdx="-1" />
            <TagPoint ActorName="RTSPoint_TannerMeadow" NearActorIdx="-1" />
            <Player />
          </Actors></MissionActors>
          <MissionEvents>
            <MissionEvent Id="Created">
              <Action ActionId="Zeds" ActionType="Action.Background"><Outputs>
                <Zombie.Spawn SpawnCount="1" SpawnPosActorIndex="3" />
                <Zombie.Spawn SpawnCount="2" SpawnPosActorIndex="4" />
                <Hint_Enable HintId="Attack" SetEnabled="false" />
              </Outputs></Action>
            </MissionEvent>
            <MissionEvent Id="Activated"><Action ActionId="Banner"><Outputs>
              <EventLog DisplayType="EventBanner"><NameText Text="MISSION: Find Help" /></EventLog>
            </Outputs></Action></MissionEvent>
            <MissionEvent Id="Completed"><Action ActionId="Chain"><Outputs>
              <RTSStat_SetBool OutputId="Game.NoSave" OutputAmount="false" />
            </Outputs></Action></MissionEvent>
          </MissionEvents>
          <ObjectiveBuckets>
            <ObjectiveBucket Label="SaveEd"><Objectives>
              <Objective Type="ClearArea" ObjectiveText="Survive!" HeadlineText="UNDER ATTACK"
                  TargetLocationOverride="3" CompleteThreshold="0">
                <Location ActorLocation="3" Radius="45" ZombieCount="3" />
                <LocationEvents>
                  <LocationEvent ActorIndex="4" Radius="46"><Action ActionId="EndObjective"><Outputs>
                    <Mission_CompleteObjective />
                  </Outputs></Action></LocationEvent>
                </LocationEvents>
                <ObjectiveEvents>
                  <ObjectiveEvent Id="Activate"><Action ActionFlags="DelayOutputs" OutputDelayTime="5s"><Outputs>
                    <Hint_Enable HintId="Attack" SetEnabled="true" />
                  </Outputs></Action></ObjectiveEvent>
                  <ObjectiveEvent Id="Complete"><Action ActionId="Sprint"><Outputs>
                    <Hint_Enable HintId="Sprint" SetEnabled="true" />
                  </Outputs></Action></ObjectiveEvent>
                </ObjectiveEvents>
              </Objective>
            </Objectives></ObjectiveBucket>
          </ObjectiveBuckets>
        </MissionFramework>
      </MissionFrameworkBucket>
    </RtsMissionData>"#;

    struct TestWorld {
        player: [f32; 3],
        zombies: u32,
        points: Vec<(&'static str, [f32; 3])>,
    }

    impl TestWorld {
        fn new(player: [f32; 3], zombies: u32) -> TestWorld {
            TestWorld {
                player,
                zombies,
                points: vec![
                    ("RTSPoint_RangerStation", [100.0, 0.0, 0.0]),
                    ("RTSPoint_taBuddies_Ambush1", [0.0, 0.0, 0.0]),
                    ("RTSPoint_TannerMeadow", [0.0, 50.0, 0.0]),
                ],
            }
        }
    }

    impl MissionWorld for TestWorld {
        fn player_pos(&self) -> [f32; 3] {
            self.player
        }
        fn tag_point(&self, name: &str) -> Option<[f32; 3]> {
            self.points.iter().find(|(n, _)| *n == name).map(|(_, p)| *p)
        }
        fn zombie_count_near(&self, _pos: [f32; 3], _radius: f32) -> u32 {
            self.zombies
        }
    }

    fn manager() -> ContentManager {
        ContentManager::new(MissionData::parse(MISSION_XML.as_bytes()))
    }

    fn has_effect(cm: &ContentManager, pred: impl Fn(&ContentEffect) -> bool) -> bool {
        cm.effects().iter().any(pred)
    }

    #[test]
    fn create_activates_onadd_mission_and_runs_created() {
        let mut cm = manager();
        assert!(cm.create_mission("ta10Buddies"));
        assert!(cm.is_active("ta10Buddies"));
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::MissionCreated(l) if l == "ta10Buddies")));
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::MissionActivated(l) if l == "ta10Buddies")));
        // Created spawns.
        let spawns: u32 = cm
            .effects()
            .iter()
            .filter_map(|e| match e {
                ContentEffect::ZombieSpawn { count, .. } => Some(*count),
                _ => None,
            })
            .sum();
        assert_eq!(spawns, 3);
        // Created disabled Attack, objective Activate re-enabled it.
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::HintEnable { id, enabled } if id == "Attack" && !*enabled)));
        // The objective is active.
        assert_eq!(
            cm.active_objective_text("ta10Buddies"),
            Some("Survive!")
        );
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::ObjectiveActivated { objective, kind, .. } if objective == "Survive!" && *kind == "ClearArea")));
    }

    #[test]
    fn clear_area_completes_when_zombies_are_cleared() {
        let mut cm = manager();
        cm.create_mission("ta10Buddies");
        cm.clear_effects();
        // Zombies still around: not complete.
        cm.tick(0.1, &TestWorld::new([0.0, 0.0, 0.0], 3));
        assert!(cm.is_active("ta10Buddies"));
        assert!(!has_effect(&cm, |e| matches!(e, ContentEffect::ObjectiveCompleted { .. })));
        // Area cleared: objective completes, bucket exhausted, mission completes.
        cm.tick(0.1, &TestWorld::new([0.0, 0.0, 0.0], 0));
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::ObjectiveCompleted { objective, .. } if objective == "Survive!")));
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::MissionCompleted(l) if l == "ta10Buddies")));
        assert!(cm.is_complete("ta10Buddies"));
        assert!(!cm.is_active("ta10Buddies"));
        // Objective Complete event enabled the sprint hint.
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::HintEnable { id, enabled } if id == "Sprint" && *enabled)));
        // Mission Completed event wrote the story flag.
        assert!(!cm.story_bool("Game.NoSave"));
    }

    #[test]
    fn location_event_completes_objective_via_mission_complete_objective() {
        let mut cm = manager();
        cm.create_mission("ta10Buddies");
        cm.clear_effects();
        // Zombies never clear; player walks to the meadow event (actor 4),
        // whose LocationEvent carries Mission_CompleteObjective.
        cm.tick(0.1, &TestWorld::new([0.0, 49.0, 0.0], 5));
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::ObjectiveCompleted { objective, .. } if objective == "Survive!")));
        assert!(cm.is_complete("ta10Buddies"));
    }

    #[test]
    fn enter_area_mission_waits_for_activation() {
        let xml = r#"<RtsMissionData><MissionFrameworkBucket Label="Arc">
            <MissionFramework Label="ta20Forest" Type="Story" ActivateType="EnterArea" ApproachRadius="25">
              <MissionActors><Actors><TagPoint ActorName="RTSPoint_TannerCabins" /></Actors></MissionActors>
              <ObjectiveBuckets><ObjectiveBucket Label="Search"><Objectives>
                <Objective Type="SearchBuilding" ObjectiveText="Search the Cabins" />
              </Objectives></ObjectiveBucket></ObjectiveBuckets>
            </MissionFramework>
          </MissionFrameworkBucket></RtsMissionData>"#;
        let mut cm = ContentManager::new(MissionData::parse(xml.as_bytes()));
        assert!(cm.create_mission("ta20Forest"));
        assert!(cm.is_active("ta20Forest"));
        // Not OnAdd: created but not activated, and no objective is chosen yet.
        assert!(!cm.is_activated("ta20Forest"));
        cm.tick(0.1, &TestWorld::new([0.0; 3], 0));
        assert_eq!(cm.active_objective_text("ta20Forest"), None);
        // The app fires activation once the player reaches the area.
        assert!(cm.activate_mission("ta20Forest"));
        assert!(cm.is_activated("ta20Forest"));
        assert_eq!(cm.active_objective_text("ta20Forest"), Some("Search the Cabins"));
    }

    #[test]
    fn force_override_completes_objective() {
        struct Forced;
        impl MissionWorld for Forced {
            fn player_pos(&self) -> [f32; 3] {
                [0.0; 3]
            }
            fn force_objective_complete(&self, mission: &str, _o: &Objective) -> bool {
                mission == "ta20Tower"
            }
        }
        let xml = r#"<RtsMissionData><MissionFrameworkBucket Label="Arc">
            <MissionFramework Label="ta20Tower" Type="Story" ActivateType="OnAdd">
              <MissionActors><Actors><Player /></Actors></MissionActors>
              <ObjectiveBuckets><ObjectiveBucket Label="Survey"><Objectives>
                <Objective Type="Survey" ObjectiveText="Survey the Area" />
              </Objectives></ObjectiveBucket></ObjectiveBuckets>
            </MissionFramework>
          </MissionFrameworkBucket></RtsMissionData>"#;
        let mut cm = ContentManager::new(MissionData::parse(xml.as_bytes()));
        cm.create_mission("ta20Tower");
        cm.tick(0.1, &Forced);
        assert!(cm.is_complete("ta20Tower"));
    }

    #[test]
    fn mission_create_output_starts_chained_mission() {
        // Build a two-mission file where mission `a` creates mission `b` from
        // its Completed event, mirroring `ta20Gunshots`'s `Mission_Create
        // ta30Return` (missions.xml:2082).
        let chained = format!(
            r#"<RtsMissionData><MissionFrameworkBucket Label="Arc">
            <MissionFramework Label="a" Type="Story" ActivateType="OnAdd">
              <MissionActors><Actors><Player /></Actors></MissionActors>
              <MissionEvents><MissionEvent Id="Completed"><Action><Outputs>
                <Mission_Create Mission="b" />
              </Outputs></Action></MissionEvent></MissionEvents>
              <ObjectiveBuckets><ObjectiveBucket Label="B"><Objectives>
                <Objective Type="Wait" ObjectiveText="Wait" WaitTime="0.0" />
              </Objectives></ObjectiveBucket></ObjectiveBuckets>
            </MissionFramework>
            <MissionFramework Label="b" Type="Story" ActivateType="OnAdd">
              <MissionActors><Actors><Player /></Actors></MissionActors>
              <ObjectiveBuckets><ObjectiveBucket Label="B"><Objectives>
                <Objective Type="Wait" ObjectiveText="Second" WaitTime="0.0" />
              </Objectives></ObjectiveBucket></ObjectiveBuckets>
            </MissionFramework>
          </MissionFrameworkBucket></RtsMissionData>"#
        );
        let mut cm = ContentManager::new(MissionData::parse(chained.as_bytes()));
        cm.create_mission("a");
        let world = TestWorld::new([0.0; 3], 0);
        cm.tick(0.1, &world);
        assert!(cm.is_complete("a"));
        assert!(cm.is_active("b"), "chained mission should be created by Completed output");
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::MissionCreated(l) if l == "b")));
    }

    /// Full head-less run of the shipped `ta10Buddies` framework: the real
    /// mission XML drives `Created` (zombie spawns + hint arming), the
    /// `SaveEd` `ClearArea` objective activates with text "Survive!", and it
    /// completes when the world reports the ambush area cleared. Tag points
    /// are resolved from the shipped ULTP markup when it is present.
    #[test]
    fn runs_shipped_ta10buddies_objective() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game");
        let Ok(missions) =
            std::fs::read(root.join("gamedata/libs/class3/contentmanager/missions.xml"))
        else {
            return; // real data not present: skip silently
        };
        let tag_points = match std::fs::read(root.join("level/tag_points_win.markup"))
            .ok()
            .and_then(|b| crate::tagpoints::TagPoints::parse(&b))
        {
            Some(t) => t,
            None => return, // positions needed to resolve the ambush area: skip
        };

        struct ShippedWorld<'a> {
            player: [f32; 3],
            zombies: u32,
            tags: &'a crate::tagpoints::TagPoints,
        }
        impl MissionWorld for ShippedWorld<'_> {
            fn player_pos(&self) -> [f32; 3] {
                self.player
            }
            fn tag_point(&self, name: &str) -> Option<[f32; 3]> {
                self.tags.get(name)
            }
            fn zombie_count_near(&self, _pos: [f32; 3], _radius: f32) -> u32 {
                self.zombies
            }
        }

        let data = MissionData::parse(&missions);
        let mut cm = ContentManager::new(data);
        assert!(cm.create_mission("ta10Buddies"), "ta10Buddies is a shipped mission");
        assert_eq!(cm.active_objective_text("ta10Buddies"), Some("Survive!"));
        assert!(cm.is_active("ta10Buddies"));

        // Zombies still in the meadow: no completion.
        let mut world = ShippedWorld { player: [0.0; 3], zombies: 3, tags: &tag_points };
        cm.clear_effects();
        cm.tick(0.1, &world);
        assert!(cm.is_active("ta10Buddies"), "objective must not complete while zombies live");
        assert!(!has_effect(&cm, |e| matches!(e, ContentEffect::MissionCompleted(_))));

        // Ambush cleared: the ClearArea objective completes, its bucket is
        // exhausted, and `MissionComplete` fires.
        world.zombies = 0;
        cm.tick(0.1, &world);
        assert!(has_effect(&cm, |e| matches!(e, ContentEffect::ObjectiveCompleted { objective, .. } if objective == "Survive!")));
        assert!(cm.is_complete("ta10Buddies"));
    }
}
