//! Mission / objective flow for State of Decay: YOSE.
//!
//! SoD stores mission *logic* in `class3/contentmanager/missions.xml` (compiled
//! twin `missions.win.bmd`). The XML root is `<RtsMissionData>`; a
//! `<MissionFramework>` describes one mission (`Label`), with
//! `<MissionEvents>/<MissionEvent Id=...>` lifecycle hooks and an
//! `<ObjectiveBuckets>/<ObjectiveBucket>/<Objectives>/<Objective Type=...>`
//! objective list.
//!
//! This module is a deliberately small, Wine-free runtime:
//!
//! * [`ObjectiveKind`] mirrors the real objective-type catalogue (16 shipped
//!   types plus the procedural/neighbour objective verbs named in the notes).
//! * [`Objective`] tracks activation and either count- or time-based progress.
//! * [`Mission`] walks an ordered objective list.
//! * [`parse_mission_xml`] reads a `<MissionFramework>` out of raw XML,
//!   skipping unknown/messy nodes rather than failing.
//! * [`Tutorial`] reconstructs the opening Mt. Tanner (`ch00`) sequence.
//! * [`RadioState`] models support-call affordability/cooldowns.
//!
//! Self-contained: no other module of this crate is referenced.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

// ---------------------------------------------------------------------------
// Objective catalogue
// ---------------------------------------------------------------------------

/// Every objective verb observed in `missions.xml` (and the variants the
/// procedural generators use). `Unknown` absorbs unrecognised `Type=` values so
/// parsing stays lenient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectiveKind {
    /// Escort follower NPCs to a target (`ta15Go`).
    LeadNpcs,
    /// Kill/clear zombies around a point (`ta10Buddies` "Survive!").
    ClearArea,
    /// Fully search N buildings (`ta20Forest`).
    SearchBuilding,
    /// Climb a lookout and survey (`ta20Tower`).
    Survey,
    /// Search props for a signal (`CLEO_Mission01`).
    SearchProp,
    /// Free trapped survivors (`ar02Quarantine`).
    RescueNpc,
    /// Kill one specific actor (`ta30Return`).
    KillActor,
    /// Find and kill a special zombie (`gr03Background`).
    Assassinate,
    /// Walk out of a radius (`ar01Request`).
    LeaveArea,
    /// Wave defence (`ar04Bridges`).
    Siege,
    /// Timed scavenge under threat (`ch03Meds`).
    Gather,
    /// Follow a leader through points (`ch02Tour`).
    Tour,
    /// Create noise at a spot (`ch04Brother`).
    Distraction,
    /// Trivial timed/proximity wait (`ar04Bridges`).
    Wait,
    /// Talk to / reach an NPC (`ar06Explosives`).
    InteractNpc,
    /// Clear a site's infestation (`ar06Explosives`).
    ClearInfestation,
    /// Carry goods to a drop-off.
    Deliver,
    /// Defend a home site (`NeighborSiege`).
    BaseSiege,
    /// Escort a VIP (`encJoinEscort`).
    Escort,
    /// Intercept a roaming horde.
    InterceptHorde,
    /// Locate and rescue a stranded survivor.
    ScoutRescue,
    /// Scout a resource node.
    ScoutResource,
    /// Impromptu supply-run drop (`ImpromptuSupplyRun`).
    SupplyRun,
    /// Hold a position against an assault.
    Defend,
    /// Unrecognised `Type=`.
    Unknown,
}

impl ObjectiveKind {
    /// Parse the `Type` attribute. Case-sensitive canonical names are matched
    /// first, then a few aliases; anything else is [`ObjectiveKind::Unknown`].
    pub fn from_str(s: &str) -> ObjectiveKind {
        let t = s.trim();
        match t {
            "LeadNpcs" => ObjectiveKind::LeadNpcs,
            "ClearArea" => ObjectiveKind::ClearArea,
            "SearchBuilding" => ObjectiveKind::SearchBuilding,
            "Survey" => ObjectiveKind::Survey,
            "SearchProp" => ObjectiveKind::SearchProp,
            "RescueNpc" => ObjectiveKind::RescueNpc,
            "KillActor" => ObjectiveKind::KillActor,
            "Assassinate" => ObjectiveKind::Assassinate,
            "LeaveArea" => ObjectiveKind::LeaveArea,
            "Siege" | "encSiege" => ObjectiveKind::Siege,
            "Gather" => ObjectiveKind::Gather,
            "Tour" => ObjectiveKind::Tour,
            "Distraction" => ObjectiveKind::Distraction,
            "Wait" => ObjectiveKind::Wait,
            "InteractNpc" | "TalkTo" => ObjectiveKind::InteractNpc,
            "ClearInfestation" => ObjectiveKind::ClearInfestation,
            "Deliver" => ObjectiveKind::Deliver,
            "BaseSiege" | "NeighborSiege" => ObjectiveKind::BaseSiege,
            "Escort" | "encJoinEscort" => ObjectiveKind::Escort,
            "InterceptHorde" => ObjectiveKind::InterceptHorde,
            "ScoutRescue" | "encRescue" | "comRescue" => ObjectiveKind::ScoutRescue,
            "ScoutResource" | "encSupplies" => ObjectiveKind::ScoutResource,
            "SupplyRun" | "ImpromptuSupplyRun" | "ImpromptuSupplyRunRescue" => {
                ObjectiveKind::SupplyRun
            }
            "Defend" => ObjectiveKind::Defend,
            _ => ObjectiveKind::Unknown,
        }
    }

    /// Canonical `Type` spelling.
    pub fn name(&self) -> &'static str {
        match self {
            ObjectiveKind::LeadNpcs => "LeadNpcs",
            ObjectiveKind::ClearArea => "ClearArea",
            ObjectiveKind::SearchBuilding => "SearchBuilding",
            ObjectiveKind::Survey => "Survey",
            ObjectiveKind::SearchProp => "SearchProp",
            ObjectiveKind::RescueNpc => "RescueNpc",
            ObjectiveKind::KillActor => "KillActor",
            ObjectiveKind::Assassinate => "Assassinate",
            ObjectiveKind::LeaveArea => "LeaveArea",
            ObjectiveKind::Siege => "Siege",
            ObjectiveKind::Gather => "Gather",
            ObjectiveKind::Tour => "Tour",
            ObjectiveKind::Distraction => "Distraction",
            ObjectiveKind::Wait => "Wait",
            ObjectiveKind::InteractNpc => "InteractNpc",
            ObjectiveKind::ClearInfestation => "ClearInfestation",
            ObjectiveKind::Deliver => "Deliver",
            ObjectiveKind::BaseSiege => "BaseSiege",
            ObjectiveKind::Escort => "Escort",
            ObjectiveKind::InterceptHorde => "InterceptHorde",
            ObjectiveKind::ScoutRescue => "ScoutRescue",
            ObjectiveKind::ScoutResource => "ScoutResource",
            ObjectiveKind::SupplyRun => "SupplyRun",
            ObjectiveKind::Defend => "Defend",
            ObjectiveKind::Unknown => "Unknown",
        }
    }

    /// Objectives whose completion clock is wall time (`tick`), as opposed to a
    /// count fed through `report`.
    pub fn is_timed(&self) -> bool {
        matches!(
            self,
            ObjectiveKind::Wait | ObjectiveKind::Gather | ObjectiveKind::Survey
        )
    }
}

// ---------------------------------------------------------------------------
// Objective
// ---------------------------------------------------------------------------

/// Lifecycle state of a single objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveState {
    Inactive,
    Active,
    Complete,
    Failed,
}

/// One objective, with count- or time-based progress toward `required`.
#[derive(Debug, Clone, PartialEq)]
pub struct Objective {
    pub kind: ObjectiveKind,
    pub state: ObjectiveState,
    /// World target resolved from a tag point / actor, when known.
    pub target: Option<[f32; 3]>,
    /// Accumulated progress (items reported, or seconds elapsed).
    pub progress: f32,
    /// Amount needed to complete. `0.0` means "any report/tick completes".
    pub required: f32,
    /// Mission actor indices referenced by the objective.
    pub actors: Vec<u32>,
}

impl Objective {
    /// A fresh inactive objective of `kind` needing `required` units.
    pub fn new(kind: ObjectiveKind, required: f32) -> Objective {
        Objective {
            kind,
            state: ObjectiveState::Inactive,
            target: None,
            progress: 0.0,
            required: required.max(0.0),
            actors: Vec::new(),
        }
    }

    /// A count-based objective (e.g. search 2 buildings).
    pub fn count(kind: ObjectiveKind, required: u32) -> Objective {
        Objective::new(kind, required as f32)
    }

    /// A time-based objective (e.g. wait 5 s).
    pub fn timed(kind: ObjectiveKind, seconds: f32) -> Objective {
        Objective::new(kind, seconds)
    }

    /// Move an inactive objective to `Active`. Returns `true` on transition.
    pub fn activate(&mut self) -> bool {
        if self.state == ObjectiveState::Inactive {
            self.state = ObjectiveState::Active;
            true
        } else {
            false
        }
    }

    /// Advance a time-based objective by `dt` seconds and complete it when the
    /// requirement is met. Count-based objectives ignore ticks.
    pub fn tick(&mut self, dt: f32) {
        if self.state != ObjectiveState::Active || !self.kind.is_timed() {
            return;
        }
        self.progress += dt.max(0.0);
        if self.progress >= self.required {
            self.state = ObjectiveState::Complete;
        }
    }

    /// Add `amount` units of progress (a searched building, a kill, a surveyed
    /// fraction). Completes when the accumulated progress reaches `required`.
    pub fn report(&mut self, amount: f32) {
        if self.state != ObjectiveState::Active {
            return;
        }
        self.progress += amount.max(0.0);
        if self.progress >= self.required {
            self.state = ObjectiveState::Complete;
        }
    }

    /// Mark the objective failed (e.g. an escort target died).
    pub fn fail(&mut self) {
        if self.state != ObjectiveState::Complete {
            self.state = ObjectiveState::Failed;
        }
    }

    /// Completion ratio in `0.0..=1.0`.
    pub fn fraction(&self) -> f32 {
        match self.state {
            ObjectiveState::Complete => 1.0,
            ObjectiveState::Inactive | ObjectiveState::Failed => 0.0,
            ObjectiveState::Active => {
                if self.required <= 0.0 {
                    0.0
                } else {
                    (self.progress / self.required).clamp(0.0, 1.0)
                }
            }
        }
    }

    /// `true` once the objective has completed.
    pub fn is_complete(&self) -> bool {
        self.state == ObjectiveState::Complete
    }
}

// ---------------------------------------------------------------------------
// Mission
// ---------------------------------------------------------------------------

/// An ordered list of objectives that advances one at a time.
#[derive(Debug, Clone, PartialEq)]
pub struct Mission {
    pub name: String,
    pub objectives: Vec<Objective>,
    pub active_index: usize,
}

impl Mission {
    /// Build a mission and activate its first objective.
    pub fn new(name: impl Into<String>, objectives: Vec<Objective>) -> Mission {
        let mut m = Mission {
            name: name.into(),
            objectives,
            active_index: 0,
        };
        m.activate_current();
        m
    }

    fn activate_current(&mut self) {
        if let Some(o) = self.objectives.get_mut(self.active_index) {
            o.activate();
        }
    }

    /// The objective currently in play, or `None` once the list is exhausted.
    pub fn current(&self) -> Option<&Objective> {
        self.objectives.get(self.active_index)
    }

    /// Mutable access to the current objective.
    pub fn current_mut(&mut self) -> Option<&mut Objective> {
        self.objectives.get_mut(self.active_index)
    }

    /// Advance to the next objective, but only once the current one is finished
    /// (`Complete` or `Failed`). Activates the new current objective. Returns
    /// `true` when the active index moved.
    pub fn advance(&mut self) -> bool {
        let finished = match self.objectives.get(self.active_index) {
            Some(o) => matches!(o.state, ObjectiveState::Complete | ObjectiveState::Failed),
            None => false,
        };
        if !finished || self.active_index + 1 >= self.objectives.len() {
            return false;
        }
        self.active_index += 1;
        self.activate_current();
        true
    }

    /// `true` when there is at least one objective and every one is complete.
    pub fn is_complete(&self) -> bool {
        !self.objectives.is_empty()
            && self
                .objectives
                .iter()
                .all(|o| o.state == ObjectiveState::Complete)
    }

    /// `true` when any objective has failed.
    pub fn is_failed(&self) -> bool {
        self.objectives
            .iter()
            .any(|o| o.state == ObjectiveState::Failed)
    }

    /// Overall completion in `0.0..=1.0`, averaged over objectives.
    pub fn progress_fraction(&self) -> f32 {
        if self.objectives.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.objectives.iter().map(Objective::fraction).sum();
        (sum / self.objectives.len() as f32).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// XML parsing
// ---------------------------------------------------------------------------

/// Look up one attribute by exact case-sensitive name, unescaping entities.
fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

fn f32_attr(e: &BytesStart, name: &str) -> Option<f32> {
    get_attr(e, name).and_then(|s| s.trim().parse::<f32>().ok())
}

fn u32_attr(e: &BytesStart, name: &str) -> Option<u32> {
    get_attr(e, name).and_then(|s| s.trim().parse::<u32>().ok())
}

/// Parse `"x,y,z"` into a world point.
fn parse_vec3(s: &str, default: [f32; 3]) -> [f32; 3] {
    let v: Vec<f32> = s
        .replace(' ', "")
        .split(',')
        .filter(|p| !p.is_empty())
        .filter_map(|p| p.parse::<f32>().ok())
        .collect();
    if v.len() >= 3 {
        [v[0], v[1], v[2]]
    } else {
        default
    }
}

fn target_attr(e: &BytesStart) -> Option<[f32; 3]> {
    for name in ["TargetPos", "Target", "Pos"] {
        if let Some(s) = get_attr(e, name) {
            if s.contains(',') {
                return Some(parse_vec3(&s, [0.0, 0.0, 0.0]));
            }
        }
    }
    None
}

const ACTOR_REF_ATTRS: [&str; 9] = [
    "TargetActorIndex",
    "RescueActorIndex",
    "RescueActorIndex2",
    "RescueActorIndex3",
    "PropActorIndex",
    "LeaderActorIndex",
    "BreachActorIndex",
    "DistractionLocActorIndex",
    "ActorIndex",
];

fn collect_actors(e: &BytesStart, out: &mut Vec<u32>) {
    for name in ACTOR_REF_ATTRS {
        if let Some(v) = u32_attr(e, name) {
            if !out.contains(&v) {
                out.push(v);
            }
        }
    }
    if let Some(v) = u32_attr(e, "ActorLocation") {
        if !out.contains(&v) {
            out.push(v);
        }
    }
}

/// Derive the requirement from the type-specific attributes. Count types use
/// their count attribute; timed types use their duration attribute.
fn derive_required(kind: ObjectiveKind, e: &BytesStart) -> f32 {
    match kind {
        ObjectiveKind::Wait => f32_attr(e, "WaitTimeMax")
            .filter(|v| *v > 0.0)
            .or_else(|| f32_attr(e, "WaitTime"))
            .unwrap_or(0.0),
        ObjectiveKind::Survey => f32_attr(e, "MinSurveyTime").unwrap_or(0.0),
        ObjectiveKind::Gather => {
            let single = f32_attr(e, "GatherTime").filter(|v| *v > 0.0);
            let hi = f32_attr(e, "GatherTimeMax").filter(|v| *v > 0.0);
            let lo = f32_attr(e, "GatherTimeMin").filter(|v| *v > 0.0);
            single.or(hi).or(lo).unwrap_or(0.0)
        }
        ObjectiveKind::SearchBuilding => u32_attr(e, "MaxCount").unwrap_or(1) as f32,
        ObjectiveKind::SearchProp => u32_attr(e, "ZombieCount")
            .filter(|v| *v > 0)
            .unwrap_or(1) as f32,
        ObjectiveKind::ClearArea => f32_attr(e, "ZombieCount")
            .filter(|v| *v > 0.0)
            .unwrap_or(1.0),
        ObjectiveKind::RescueNpc => {
            let mut n = 0u32;
            for name in ["RescueActorIndex", "RescueActorIndex2", "RescueActorIndex3"] {
                if u32_attr(e, name).is_some() {
                    n += 1;
                }
            }
            n.max(1) as f32
        }
        ObjectiveKind::KillActor
        | ObjectiveKind::Assassinate
        | ObjectiveKind::InteractNpc
        | ObjectiveKind::ClearInfestation => 1.0,
        ObjectiveKind::Siege => 1.0,
        _ => 1.0,
    }
}

fn objective_from(e: &BytesStart) -> Objective {
    let kind = get_attr(e, "Type")
        .map(|s| ObjectiveKind::from_str(&s))
        .unwrap_or(ObjectiveKind::Unknown);
    let required = derive_required(kind, e);
    let mut actors = Vec::new();
    collect_actors(e, &mut actors);
    Objective {
        kind,
        state: ObjectiveState::Inactive,
        target: target_attr(e),
        progress: 0.0,
        required,
        actors,
    }
}

/// Fold a child `<Location ...>` / `<LocationEvent ...>` back into the current
/// objective (extends reach, or raises the zombie-count requirement).
fn apply_location_child(obj: &mut Objective, e: &BytesStart) {
    if let Some(z) = f32_attr(e, "ZombieCount") {
        if z > 0.0 {
            obj.required = z;
        }
    }
    collect_actors(e, &mut obj.actors);
}

/// Parse a mission from raw `missions.xml` bytes.
///
/// Finds the first `<MissionFramework Label=...>` (its `Label` is the mission
/// name, falling back to `DescriptionText`/`OpportunityText`/`FragmentFile`),
/// then collects every `<Objective Type=...>` beneath it. Unknown elements,
/// attributes and non-UTF-8 values are skipped; malformed XML or a document
/// with no `MissionFramework` yields `Err`.
pub fn parse_mission_xml(xml: &[u8]) -> Result<Mission, String> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut stack: Vec<Vec<u8>> = Vec::new();

    let mut found = false;
    let mut name = String::new();
    let mut framework_label = String::new();
    let mut objectives: Vec<Objective> = Vec::new();
    let mut cur_obj: Option<usize> = None;

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("xml parse error: {e}")),
            Ok(ev) => ev,
        };
        let is_start = matches!(ev, Event::Start(_));

        match ev {
            Event::Start(e) | Event::Empty(e) => {
                let name_bytes = e.name().as_ref().to_vec();
                let local = name_bytes.as_slice();

                if local == b"MissionFramework" && !found {
                    found = true;
                    name = get_attr(&e, "Label").unwrap_or_default();
                    if name.is_empty() {
                        name = get_attr(&e, "DescriptionText")
                            .filter(|s| !s.is_empty())
                            .or_else(|| {
                                get_attr(&e, "OpportunityText").filter(|s| !s.is_empty())
                            })
                            .unwrap_or_default();
                    }
                } else if local == b"MissionFrameworkBucket" && framework_label.is_empty() {
                    framework_label = get_attr(&e, "FragmentFile").unwrap_or_default();
                } else if local == b"Objective" && get_attr(&e, "Type").is_some() {
                    objectives.push(objective_from(&e));
                    cur_obj = Some(objectives.len() - 1);
                } else if matches!(local, b"Location" | b"LocationEvent") {
                    if let Some(i) = cur_obj {
                        if let Some(o) = objectives.get_mut(i) {
                            apply_location_child(o, &e);
                        }
                    }
                }

                if is_start {
                    stack.push(name_bytes);
                } else if local == b"Objective" {
                    // Empty element: close out the objective immediately.
                    cur_obj = None;
                }
            }
            Event::End(e) => {
                let name_bytes = e.name().as_ref().to_vec();
                stack.pop();
                if name_bytes.as_slice() == b"Objective" {
                    cur_obj = None;
                }
            }
            _ => {}
        }
        buf.clear();
    }

    if !found {
        return Err("no <MissionFramework> element found".to_string());
    }
    if name.is_empty() {
        name = framework_label;
    }
    if name.is_empty() {
        name = "mission".to_string();
    }

    Ok(Mission::new(name, objectives))
}

// ---------------------------------------------------------------------------
// Whole-file mission catalogue (`missions.xml`)
// ---------------------------------------------------------------------------

/// One entry in a mission's `MissionActors/Actors` indexed list.
///
/// `index` is the position in that list — the value every `ActorIndex=`,
/// `TargetActorIndex=`, `NearActorIdx=` … in `missions.xml` refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionActor {
    /// 0-based position within `MissionActors/Actors`.
    pub index: u32,
    /// Cast selector for a cast member (`ActorTraits`, e.g.
    /// `Trait.Story.Buddy`, falling back to `ActorName`).
    pub character: Option<String>,
    /// `ActorName` when the actor is a `TagPoint` (`RTSPoint_…`).
    pub tag_point: Option<String>,
    /// Parsed `ActorTraits` / `FunctionTags` values.
    pub traits: Vec<String>,
}

/// A single `<MissionFramework>` extracted from `missions.xml`, keyed by
/// `Label` and grouped by the enclosing `<MissionFrameworkBucket Label=...>`.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionDef {
    /// `<MissionFramework Label=...>` (e.g. `ta10Buddies`).
    pub label: String,
    /// Enclosing bucket `Label` (e.g. `Tanner Arc (ch00)`).
    pub bucket: String,
    /// Every `<Objective Type=...>` beneath the mission, in document order.
    pub objective_kinds: Vec<ObjectiveKind>,
    /// The indexed `<MissionActors>/<Actors>` list, in document order.
    pub actors: Vec<MissionActor>,
    /// Tag names emitted under `<Outputs>` of the `Created` `<MissionEvent>`
    /// (e.g. `Zombie.Spawn`, `Scene_ScenePlay`, `Hint_Enable`).
    pub created_actions: Vec<String>,
}

/// Split a comma-separated attribute value (e.g. `FunctionTags="Npc,Female"`).
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

/// Build a [`MissionActor`] from one entry of `MissionActors/Actors`.
fn mission_actor(e: &BytesStart, index: u32) -> MissionActor {
    let name = e.name().as_ref().to_vec();
    let local: &[u8] = name.as_slice();
    let actor_name = get_attr(e, "ActorName").filter(|s| !s.is_empty());
    let actor_traits = get_attr(e, "ActorTraits").unwrap_or_default();
    let fn_tags = get_attr(e, "FunctionTags").unwrap_or_default();

    let mut traits = split_csv(&actor_traits);
    for t in split_csv(&fn_tags) {
        if !traits.contains(&t) {
            traits.push(t);
        }
    }

    let is_tag_point = local == b"TagPoint";
    let character = if is_tag_point {
        None
    } else {
        let from_traits = actor_traits.trim();
        if !from_traits.is_empty() {
            Some(from_traits.to_string())
        } else {
            actor_name.clone()
        }
    };
    let tag_point = if is_tag_point { actor_name } else { None };

    MissionActor {
        index,
        character,
        tag_point,
        traits,
    }
}

/// Parse the whole `contentmanager/missions.xml` into a flat catalogue of
/// [`MissionDef`]s.
///
/// Lenient by design: unknown elements/attributes are skipped, non-UTF-8 text
/// is lossily decoded, and malformed XML simply ends the walk with whatever was
/// collected so far (this function cannot fail and never panics on the real
/// 3.67 MB file).
pub fn parse_missions_xml(xml: &[u8]) -> Vec<MissionDef> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut stack: Vec<Vec<u8>> = Vec::new();
    let mut defs: Vec<MissionDef> = Vec::new();

    let mut cur_bucket = String::new();
    let mut cur_def: Option<MissionDef> = None;
    let mut in_created = false;
    let mut outputs_level: Option<usize> = None;

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            // Lenient: a malformed document stops the walk, keeping what we have.
            Err(_) => break,
            Ok(ev) => ev,
        };
        let is_start = matches!(ev, Event::Start(_));

        match ev {
            Event::Start(e) | Event::Empty(e) => {
                let name_bytes = e.name().as_ref().to_vec();
                let local = name_bytes.as_slice();

                // Entering a bucket: remember its label (fall back to the
                // fragment file name).
                if local == b"MissionFrameworkBucket" {
                    if let Some(l) = get_attr(&e, "Label").filter(|s| !s.is_empty()).or_else(|| {
                        get_attr(&e, "FragmentFile").filter(|s| !s.is_empty())
                    }) {
                        cur_bucket = l;
                    }
                }

                // Starting a mission: flush the previous one, then begin anew.
                if local == b"MissionFramework" {
                    if let Some(d) = cur_def.take() {
                        defs.push(d);
                    }
                    let label = get_attr(&e, "Label")
                        .filter(|s| !s.is_empty())
                        .or_else(|| get_attr(&e, "DescriptionText").filter(|s| !s.is_empty()))
                        .or_else(|| get_attr(&e, "OpportunityText").filter(|s| !s.is_empty()))
                        .unwrap_or_default();
                    cur_def = Some(MissionDef {
                        label,
                        bucket: cur_bucket.clone(),
                        objective_kinds: Vec::new(),
                        actors: Vec::new(),
                        created_actions: Vec::new(),
                    });
                }

                if local == b"MissionEvent" && cur_def.is_some() {
                    if get_attr(&e, "Id").as_deref() == Some("Created") {
                        in_created = true;
                    }
                } else if local == b"Objective" && cur_def.is_some() {
                    if let Some(t) = get_attr(&e, "Type") {
                        if let Some(d) = cur_def.as_mut() {
                            d.objective_kinds.push(ObjectiveKind::from_str(&t));
                        }
                    }
                } else if local == b"Outputs" {
                    // Only Start has children; an empty `<Outputs/>` has none.
                    if in_created && is_start {
                        outputs_level = Some(stack.len() + 1);
                    }
                } else if in_created
                    && outputs_level == Some(stack.len())
                    && local != b"Outputs"
                {
                    // Direct child of a `<Outputs>` inside the Created event.
                    if let Some(d) = cur_def.as_mut() {
                        d.created_actions
                            .push(String::from_utf8_lossy(local).into_owned());
                    }
                }

                // Actor entries: direct children of `<Actors>` under
                // `<MissionActors>`.
                let parent_is_actors = stack
                    .last()
                    .map(|p| p.as_slice() == b"Actors")
                    .unwrap_or(false);
                if parent_is_actors
                    && cur_def.is_some()
                    && stack.iter().any(|n| n.as_slice() == b"MissionActors")
                {
                    let idx = cur_def.as_ref().map(|d| d.actors.len() as u32).unwrap_or(0);
                    let actor = mission_actor(&e, idx);
                    if let Some(d) = cur_def.as_mut() {
                        d.actors.push(actor);
                    }
                }

                if is_start {
                    stack.push(name_bytes);
                }
            }
            Event::End(e) => {
                let name_bytes = e.name().as_ref().to_vec();
                let local = name_bytes.as_slice();
                stack.pop();
                if local == b"Outputs" {
                    outputs_level = None;
                }
                if local == b"MissionEvent" {
                    in_created = false;
                }
                if local == b"MissionFramework" {
                    if let Some(d) = cur_def.take() {
                        defs.push(d);
                    }
                }
            }
            _ => {}
        }
        buf.clear();
    }

    if let Some(d) = cur_def.take() {
        defs.push(d);
    }
    defs
}

/// The bucket label of the opening (Mt. Tanner) tutorial arc.
const TANNER_BUCKET: &str = "Tanner Arc (ch00)";

/// Missions belonging to the Tanner-arc (`ch00`) bucket, in file order.
///
/// Matching is case-insensitive on the bucket label. If no mission carries the
/// expected label (a different extraction), this falls back to labels beginning
/// with `ta` (the tutorial arc's naming prefix).
pub fn tanner_arc(defs: &[MissionDef]) -> Vec<&MissionDef> {
    let exact: Vec<&MissionDef> = defs
        .iter()
        .filter(|d| d.bucket.eq_ignore_ascii_case(TANNER_BUCKET))
        .collect();
    if !exact.is_empty() {
        return exact;
    }
    defs.iter()
        .filter(|d| d.label.to_ascii_lowercase().starts_with("ta"))
        .collect()
}

/// The ordered objective headlines of the tutorial, following the mission
/// order of [`tanner_arc`] and each mission's objective order.
///
/// `MissionDef` retains objective *kinds*, so the canonical [`ObjectiveKind`]
/// name is used as the headline; missions with no objectives contribute their
/// label instead.
pub fn tutorial_flow(defs: &[MissionDef]) -> Vec<String> {
    let mut out = Vec::new();
    for d in tanner_arc(defs) {
        if d.objective_kinds.is_empty() {
            if !d.label.is_empty() {
                out.push(d.label.clone());
            }
        } else {
            for k in &d.objective_kinds {
                out.push(k.name().to_string());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Opening tutorial (Mt. Tanner ch00)
// ---------------------------------------------------------------------------

/// A milestone the tutorial waits for.
#[derive(Debug, Clone, PartialEq)]
pub enum TutorialTrigger {
    /// Player reached `point` within `radius` metres.
    ReachPoint([f32; 3], f32),
    /// A container/prop was searched.
    SearchContainer,
    /// A weapon was equipped.
    EquipWeapon,
    /// At least this many zombies were killed.
    KillCount(u32),
    /// Talked to actor index.
    TalkTo(u32),
    /// Native/scene call (e.g. `Scene.Move`).
    Custom(String),
}

/// One on-screen tutorial step and the event that retires it.
#[derive(Debug, Clone, PartialEq)]
pub struct TutorialStep {
    pub text: String,
    pub trigger: TutorialTrigger,
}

/// Opening-state inputs needed to build the Tanner sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct OpeningState {
    /// Where the player starts (not used for step 0, kept for callers).
    pub player_start: [f32; 3],
    /// The buddy/survivor to reach first (`Trait.Story.Buddy`).
    pub survivor_pos: [f32; 3],
    /// The Ranger Station building (`RTSPoint_RangerStation`).
    pub ranger_station: [f32; 3],
    /// The Home site to return to.
    pub home_pos: [f32; 3],
    /// Zombies that must be killed during the ambush.
    pub kill_target: u32,
    /// Actor index of the buddy to talk to.
    pub survivor_actor: u32,
}

impl Default for OpeningState {
    fn default() -> OpeningState {
        OpeningState {
            player_start: [0.0, 0.0, 0.0],
            survivor_pos: [10.0, 0.0, 0.0],
            ranger_station: [120.0, 0.0, 40.0],
            home_pos: [0.0, 0.0, 0.0],
            kill_target: 3,
            survivor_actor: 0,
        }
    }
}

/// The reconstructed opening tutorial (`ch00`).
#[derive(Debug, Clone, PartialEq)]
pub struct Tutorial {
    pub steps: Vec<TutorialStep>,
    pub index: usize,
}

impl Tutorial {
    /// Reconstruct the Mt. Tanner opening from [`OpeningState`].
    ///
    /// Mirrors §4.1: move/look, reach the buddy, survive the ambush, reach the
    /// Ranger Station, search the supply container, equip a weapon, return home.
    pub fn new_tanner(state: &OpeningState) -> Tutorial {
        let steps = vec![
            TutorialStep {
                text: "Use [WASD] to move. Use the mouse to look around.".to_string(),
                trigger: TutorialTrigger::Custom("Scene.Move".to_string()),
            },
            TutorialStep {
                text: "Reach the survivor.".to_string(),
                trigger: TutorialTrigger::ReachPoint(state.survivor_pos, 35.0),
            },
            TutorialStep {
                text: "Survive the ambush.".to_string(),
                trigger: TutorialTrigger::KillCount(state.kill_target.max(1)),
            },
            TutorialStep {
                text: "Go to the Ranger Station.".to_string(),
                trigger: TutorialTrigger::ReachPoint(state.ranger_station, 25.0),
            },
            TutorialStep {
                text: "Search the container for supplies.".to_string(),
                trigger: TutorialTrigger::SearchContainer,
            },
            TutorialStep {
                text: "Equip a weapon.".to_string(),
                trigger: TutorialTrigger::EquipWeapon,
            },
            TutorialStep {
                text: "Return home.".to_string(),
                trigger: TutorialTrigger::ReachPoint(state.home_pos, 25.0),
            },
        ];
        Tutorial { steps, index: 0 }
    }

    /// The step currently awaiting its trigger.
    pub fn current(&self) -> Option<&TutorialStep> {
        self.steps.get(self.index)
    }

    /// Move to the next step. Returns `true` when the index moved.
    pub fn advance(&mut self) -> bool {
        if self.index + 1 < self.steps.len() {
            self.index += 1;
            true
        } else if self.index < self.steps.len() {
            // Retire the final step by advancing past the end.
            self.index += 1;
            true
        } else {
            false
        }
    }

    /// `true` once every step has been retired.
    pub fn is_complete(&self) -> bool {
        !self.steps.is_empty() && self.index >= self.steps.len()
    }

    /// Feed a gameplay event. If it satisfies the current step's trigger the
    /// tutorial advances and `true` is returned.
    pub fn notify(&mut self, trigger_event: &TutorialTrigger) -> bool {
        let matched = match self.current() {
            Some(step) => triggers_match(&step.trigger, trigger_event),
            None => false,
        };
        if matched {
            self.advance()
        } else {
            false
        }
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Does `event` satisfy the pending `wanted` trigger?
fn triggers_match(wanted: &TutorialTrigger, event: &TutorialTrigger) -> bool {
    match (wanted, event) {
        (TutorialTrigger::ReachPoint(wp, r), TutorialTrigger::ReachPoint(ep, _)) => {
            dist2(*wp, *ep) <= (*r) * (*r)
        }
        (TutorialTrigger::SearchContainer, TutorialTrigger::SearchContainer) => true,
        (TutorialTrigger::EquipWeapon, TutorialTrigger::EquipWeapon) => true,
        (TutorialTrigger::KillCount(need), TutorialTrigger::KillCount(got)) => got >= need,
        (TutorialTrigger::TalkTo(a), TutorialTrigger::TalkTo(b)) => a == b,
        (TutorialTrigger::Custom(a), TutorialTrigger::Custom(b)) => a == b,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Radio support calls
// ---------------------------------------------------------------------------

/// Support strikes available from the radio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RadioOption {
    AirStrike,
    ArtilleryBarrage,
    DroneStrike,
    MortarBarrage,
}

impl RadioOption {
    /// Dense index into [`RadioState`]'s parallel arrays.
    pub fn index(&self) -> usize {
        match self {
            RadioOption::AirStrike => 0,
            RadioOption::ArtilleryBarrage => 1,
            RadioOption::DroneStrike => 2,
            RadioOption::MortarBarrage => 3,
        }
    }

    /// Cooldown applied when the option is executed, in seconds.
    pub fn cooldown(&self) -> f32 {
        RADIO_COOLDOWNS[self.index()]
    }

    /// All options in index order.
    pub fn all() -> [RadioOption; 4] {
        [
            RadioOption::AirStrike,
            RadioOption::ArtilleryBarrage,
            RadioOption::DroneStrike,
            RadioOption::MortarBarrage,
        ]
    }
}

const RADIO_COOLDOWNS: [f32; 4] = [180.0, 240.0, 300.0, 150.0];

/// Player influence pool plus per-option cost/cooldown tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct RadioState {
    /// Remaining cooldown seconds per option (0 = ready).
    pub cooldowns: [f32; 4],
    /// Influence cost per option.
    pub costs: [u32; 4],
    /// Influence currently available.
    pub influence: u32,
}

impl Default for RadioState {
    fn default() -> RadioState {
        RadioState {
            cooldowns: [0.0; 4],
            costs: [30, 50, 80, 40],
            influence: 100,
        }
    }
}

impl RadioState {
    /// Create a state with an influence pool and explicit costs.
    pub fn new(influence: u32, costs: [u32; 4]) -> RadioState {
        RadioState {
            cooldowns: [0.0; 4],
            costs,
            influence,
        }
    }

    /// `true` when the option is off cooldown *and* affordable.
    pub fn eval(&self, option: RadioOption) -> bool {
        let i = option.index();
        self.cooldowns[i] <= 0.0 && self.influence >= self.costs[i]
    }

    /// Execute the option: spend influence and start its cooldown. Returns
    /// `false` (spending nothing) when [`RadioState::eval`] fails.
    pub fn exec(&mut self, option: RadioOption) -> bool {
        if !self.eval(option) {
            return false;
        }
        let i = option.index();
        self.influence -= self.costs[i];
        self.cooldowns[i] = option.cooldown();
        true
    }

    /// Advance every cooldown by `dt` seconds, clamping at zero.
    pub fn tick(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        for c in self.cooldowns.iter_mut() {
            *c = (*c - dt).max(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Native mission runtime
// ---------------------------------------------------------------------------

/// An effect emitted by the mission runtime, meant to be applied by the app /
/// native layer (spawns, HUD hints, story flags, mission chaining, scenes).
#[derive(Debug, Clone, PartialEq)]
pub enum MissionEffect {
    SpawnZombie { kind: String, count: u32 },
    SpawnFreak { kind: String, count: u32 },
    HintShow { id: String },
    HintEnable { id: String, enabled: bool },
    SetStatBool { stat: String, value: bool },
    SetStoryFlag { flag: String },
    CompleteObjective { mission: String, objective: String },
    CreateMission { label: String },
    AbortMission { label: String },
    PlayScene { label: String },
    Log(String),
}

/// Best-effort mapping of a real `missions.xml` output verb (plus positional
/// `args`) to a [`MissionEffect`]. `args` follow the verb's natural order:
///
/// * `Zombie.Spawn` / `Zombie.FreakSpawn` – `[kind, count]`
/// * `Hint_Show` – `[id]`
/// * `Hint_Enable` – `[id, enabled]`
/// * `RTSStat_SetBool` – `[stat, value]`
/// * `SetStoryFlag` – `[flag]`
/// * `Mission_CompleteObjective` – `[mission, objective]`
/// * `Mission_Create` / `Mission_AbortMission` – `[label]`
/// * `Scene_ScenePlay` – `[label]`
/// * `EventLog` – `[text?]`
///
/// Missing args fall back to sensible defaults (count 1, enabled true); unknown
/// verbs yield `None` so parsing/runtime stays lenient.
pub fn effect_from_action(action: &str, args: &[String]) -> Option<MissionEffect> {
    let a = action.trim();
    if a.is_empty() {
        return None;
    }
    let arg = |i: usize| args.get(i).map(|s| s.as_str());
    let count = |i: usize| {
        arg(i)
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(1)
    };
    let flag = |i: usize| {
        arg(i)
            .map(|s| {
                matches!(
                    s.trim().to_ascii_lowercase().as_str(),
                    "true" | "1" | "yes" | "on"
                )
            })
            .unwrap_or(true)
    };

    match a {
        "Zombie.Spawn" => Some(MissionEffect::SpawnZombie {
            kind: arg(0).unwrap_or("Zombie.Walker").to_string(),
            count: count(1),
        }),
        "Zombie.FreakSpawn" => Some(MissionEffect::SpawnFreak {
            kind: arg(0).unwrap_or("Zombie.Freak").to_string(),
            count: count(1),
        }),
        "Hint_Show" => Some(MissionEffect::HintShow {
            id: arg(0).unwrap_or_default().to_string(),
        }),
        "Hint_Enable" => Some(MissionEffect::HintEnable {
            id: arg(0).unwrap_or_default().to_string(),
            enabled: flag(1),
        }),
        "RTSStat_SetBool" => Some(MissionEffect::SetStatBool {
            stat: arg(0).unwrap_or_default().to_string(),
            value: flag(1),
        }),
        "SetStoryFlag" => Some(MissionEffect::SetStoryFlag {
            flag: arg(0).unwrap_or_default().to_string(),
        }),
        "Mission_CompleteObjective" => Some(MissionEffect::CompleteObjective {
            mission: arg(0).unwrap_or_default().to_string(),
            objective: arg(1).unwrap_or_default().to_string(),
        }),
        "Mission_Create" => Some(MissionEffect::CreateMission {
            label: arg(0).unwrap_or_default().to_string(),
        }),
        "Mission_AbortMission" => Some(MissionEffect::AbortMission {
            label: arg(0).unwrap_or_default().to_string(),
        }),
        "Scene_ScenePlay" => Some(MissionEffect::PlayScene {
            label: arg(0).unwrap_or_default().to_string(),
        }),
        "EventLog" => Some(MissionEffect::Log(
            arg(0).unwrap_or(a).to_string(),
        )),
        _ => None,
    }
}

/// One objective currently tracked by a running [`ActiveMission`].
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveObjective {
    pub kind: ObjectiveKind,
    pub required: f32,
    pub progress: f32,
    pub state: ObjectiveState,
    /// World target, when the catalogue carried one.
    pub target: Option<[f32; 3]>,
    /// Completion radius around `target` (0 = unbounded).
    pub radius: f32,
}

/// One mission started through the [`MissionRuntime`].
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveMission {
    pub label: String,
    pub objectives: Vec<ActiveObjective>,
}

/// Default requirement for an objective whose explicit count/duration is not
/// retained by [`MissionDef`]: one unit of progress.
const DEFAULT_REQUIRED: f32 = 1.0;

/// Fixed wall-clock step applied per [`MissionRuntime::tick`] call.
pub const TICK_DT: f32 = 1.0;

/// True when `pos` is within `o.radius` of `o.target`. When the objective has
/// no resolvable target, `lenient` decides (`true` = any position matches).
fn objective_in_range(o: &ActiveObjective, pos: [f32; 3], lenient: bool) -> bool {
    match o.target {
        Some(t) if o.radius > 0.0 => dist2(t, pos) <= o.radius * o.radius,
        _ => lenient,
    }
}

/// Kinds that complete when an actor/player arrives at their target.
fn reach_completes(kind: ObjectiveKind) -> bool {
    matches!(
        kind,
        ObjectiveKind::LeadNpcs
            | ObjectiveKind::ClearArea
            | ObjectiveKind::Tour
            | ObjectiveKind::InteractNpc
            | ObjectiveKind::RescueNpc
            | ObjectiveKind::Defend
            | ObjectiveKind::Siege
            | ObjectiveKind::BaseSiege
            | ObjectiveKind::Escort
            | ObjectiveKind::Deliver
    )
}

/// A small, deterministic, total mission runtime: it starts [`MissionDef`]s on
/// demand and advances their objectives from world events, returning the
/// [`MissionEffect`]s the app should apply. It never panics on unknown labels,
/// out-of-range indices or empty objective lists.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionRuntime {
    defs: Vec<MissionDef>,
    active: Vec<ActiveMission>,
    completed: Vec<String>,
    effects: Vec<MissionEffect>,
}

impl MissionRuntime {
    /// Build a runtime over a parsed mission catalogue.
    pub fn new(defs: Vec<MissionDef>) -> Self {
        MissionRuntime {
            defs,
            active: Vec::new(),
            completed: Vec::new(),
            effects: Vec::new(),
        }
    }

    /// The catalogue this runtime was built from.
    pub fn defs(&self) -> &[MissionDef] {
        &self.defs
    }

    /// Currently running missions.
    pub fn active(&self) -> &[ActiveMission] {
        &self.active
    }

    /// Labels of missions that have completed.
    pub fn completed(&self) -> &[String] {
        &self.completed
    }

    /// Every effect ever emitted, in order.
    pub fn effects(&self) -> &[MissionEffect] {
        &self.effects
    }

    /// `true` once `label` has been started and fully completed.
    pub fn is_complete(&self, label: &str) -> bool {
        self.completed.iter().any(|l| l == label)
    }

    fn find_active(&self, label: &str) -> Option<usize> {
        self.active
            .iter()
            .position(|m| m.label == label || m.label.eq_ignore_ascii_case(label))
    }

    /// Index of a mission's current (first active) objective, if any.
    fn current_index(&self, mi: usize) -> Option<usize> {
        self.active[mi]
            .objectives
            .iter()
            .position(|o| o.state == ObjectiveState::Active)
    }

    /// Activate the first still-inactive objective of mission `mi`.
    fn activate_next(&mut self, mi: usize) {
        if let Some(pos) = self.active[mi]
            .objectives
            .iter()
            .position(|o| o.state == ObjectiveState::Inactive)
        {
            self.active[mi].objectives[pos].state = ObjectiveState::Active;
        }
    }

    /// Mark objective `oi` of mission `mi` complete, activate the next one and
    /// record mission completion. Emits `CompleteObjective` (and any derivable
    /// follow-up `CreateMission`).
    fn finish_objective(&mut self, mi: usize, oi: usize, out: &mut Vec<MissionEffect>) {
        let label = self.active[mi].label.clone();
        let objective = {
            let obj = &mut self.active[mi].objectives[oi];
            if obj.state == ObjectiveState::Complete {
                return;
            }
            obj.state = ObjectiveState::Complete;
            if obj.progress < obj.required {
                obj.progress = obj.required;
            }
            obj.kind.name().to_string()
        };
        out.push(MissionEffect::CompleteObjective {
            mission: label.clone(),
            objective,
        });

        self.activate_next(mi);
        let all_done = self.active[mi]
            .objectives
            .iter()
            .all(|o| o.state == ObjectiveState::Complete);
        if all_done && !self.completed.contains(&label) {
            self.completed.push(label.clone());
            for fu in self.follow_ups(&label) {
                out.push(MissionEffect::CreateMission { label: fu });
            }
        }
    }

    /// Follow-up missions unlocked by completing `label`.
    ///
    /// `MissionDef` retains only the `Created` event's outputs, whereas
    /// `Mission_Create` follow-ups are authored under `Completed`; without those
    /// outputs nothing is derivable here, so the runtime emits none.
    fn follow_ups(&self, label: &str) -> Vec<String> {
        let _ = label;
        Vec::new()
    }

    /// Drop finished missions from the running set.
    fn reap(&mut self) {
        if self.completed.is_empty() {
            return;
        }
        let completed = &self.completed;
        self.active.retain(|m| !completed.contains(&m.label));
    }

    fn record(&mut self, out: &mut Vec<MissionEffect>) {
        self.reap();
        self.effects.extend(out.iter().cloned());
    }

    /// Start `label`: activate its first objective and emit the mission's
    /// `Created` actions as effects. Unknown, already-running or already-finished
    /// labels are a no-op returning no effects.
    pub fn start(&mut self, label: &str) -> Vec<MissionEffect> {
        let mut out = Vec::new();
        if self.find_active(label).is_some() || self.is_complete(label) {
            return out;
        }
        let def = match self
            .defs
            .iter()
            .find(|d| d.label == label || d.label.eq_ignore_ascii_case(label))
        {
            Some(d) => d.clone(),
            None => return out,
        };

        for action in &def.created_actions {
            if let Some(e) = effect_from_action(action, &[]) {
                out.push(e);
            }
        }

        let mut objectives: Vec<ActiveObjective> = def
            .objective_kinds
            .iter()
            .map(|k| ActiveObjective {
                kind: *k,
                required: DEFAULT_REQUIRED,
                progress: 0.0,
                state: ObjectiveState::Inactive,
                target: None,
                radius: 0.0,
            })
            .collect();
        if let Some(first) = objectives.first_mut() {
            first.state = ObjectiveState::Active;
        }
        let empty = objectives.is_empty();
        self.active.push(ActiveMission {
            label: def.label.clone(),
            objectives,
        });
        if empty && !self.completed.contains(&def.label) {
            self.completed.push(def.label.clone());
            for fu in self.follow_ups(&def.label) {
                out.push(MissionEffect::CreateMission { label: fu });
            }
        }

        self.record(&mut out);
        out
    }

    /// Advance objective `objective` of `mission` by `amount`, completing it and
    /// chaining to the next objective / mission completion when the requirement
    /// is met.
    pub fn report_progress(
        &mut self,
        mission: &str,
        objective: usize,
        amount: f32,
    ) -> Vec<MissionEffect> {
        let mut out = Vec::new();
        if let Some(mi) = self.find_active(mission) {
            if objective < self.active[mi].objectives.len()
                && self.active[mi].objectives[objective].state == ObjectiveState::Active
            {
                let complete = {
                    let o = &mut self.active[mi].objectives[objective];
                    o.progress += amount.max(0.0);
                    o.progress >= o.required
                };
                if complete {
                    self.finish_objective(mi, objective, &mut out);
                }
            }
        }
        self.record(&mut out);
        out
    }

    /// Signal that the player/follower reached `actor_index` at `pos`. Completes
    /// active `LeadNpcs` / `ClearArea` (`CompleteOnPlayerReachTarget`) objectives
    /// in range.
    ///
    /// `MissionDef` does not retain per-objective actor lists, so `actor_index`
    /// is informational; completion keys off the objective's own target/radius.
    pub fn actor_entered(&mut self, actor_index: u32, pos: [f32; 3]) -> Vec<MissionEffect> {
        let _ = actor_index;
        let mut out = Vec::new();
        let n_missions = self.active.len();
        for mi in 0..n_missions {
            let oi = match self.current_index(mi) {
                Some(i) => i,
                None => continue,
            };
            let hit = {
                let o = &self.active[mi].objectives[oi];
                matches!(o.kind, ObjectiveKind::LeadNpcs | ObjectiveKind::ClearArea)
                    && objective_in_range(o, pos, true)
            };
            if hit {
                self.finish_objective(mi, oi, &mut out);
            }
        }
        self.record(&mut out);
        out
    }

    /// Report that `remaining` zombies are left at `pos`. Active `ClearArea`
    /// objectives in range complete once the count reaches zero.
    pub fn zombies_cleared(&mut self, pos: [f32; 3], remaining: u32) -> Vec<MissionEffect> {
        let mut out = Vec::new();
        let n_missions = self.active.len();
        for mi in 0..n_missions {
            let oi = match self.current_index(mi) {
                Some(i) => i,
                None => continue,
            };
            let is_clear = {
                let o = &self.active[mi].objectives[oi];
                o.kind == ObjectiveKind::ClearArea && objective_in_range(o, pos, true)
            };
            if !is_clear {
                continue;
            }
            let complete = {
                let o = &mut self.active[mi].objectives[oi];
                let cleared = (o.required - remaining as f32).max(0.0);
                if cleared > o.progress {
                    o.progress = cleared;
                }
                o.progress >= o.required
            };
            if complete {
                self.finish_objective(mi, oi, &mut out);
            }
        }
        self.record(&mut out);
        out
    }

    /// Report that `count` buildings were searched. Advances active
    /// `SearchBuilding` objectives.
    pub fn building_searched(&mut self, count: u32) -> Vec<MissionEffect> {
        let mut out = Vec::new();
        let n_missions = self.active.len();
        for mi in 0..n_missions {
            let oi = match self.current_index(mi) {
                Some(i) => i,
                None => continue,
            };
            let is_search = self.active[mi].objectives[oi].kind == ObjectiveKind::SearchBuilding;
            if !is_search {
                continue;
            }
            let complete = {
                let o = &mut self.active[mi].objectives[oi];
                o.progress += count as f32;
                o.progress >= o.required
            };
            if complete {
                self.finish_objective(mi, oi, &mut out);
            }
        }
        self.record(&mut out);
        out
    }

    /// Evaluate proximity/time objectives against the current `actors`
    /// (`[(actor_index, position)]`). Timed objectives advance by [`TICK_DT`];
    /// reach objectives with a resolvable target complete when an actor is in
    /// range.
    pub fn tick(&mut self, actors: &[(u32, [f32; 3])]) -> Vec<MissionEffect> {
        let mut out = Vec::new();
        let n_missions = self.active.len();
        for mi in 0..n_missions {
            let oi = match self.current_index(mi) {
                Some(i) => i,
                None => continue,
            };
            let timed = self.active[mi].objectives[oi].kind.is_timed();
            if timed {
                let complete = {
                    let o = &mut self.active[mi].objectives[oi];
                    o.progress += TICK_DT;
                    o.progress >= o.required
                };
                if complete {
                    self.finish_objective(mi, oi, &mut out);
                }
                continue;
            }
            let near = {
                let o = &self.active[mi].objectives[oi];
                o.target.is_some()
                    && reach_completes(o.kind)
                    && actors
                        .iter()
                        .any(|(_, p)| objective_in_range(o, *p, false))
            };
            if near {
                self.finish_objective(mi, oi, &mut out);
            }
        }
        self.record(&mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<RtsMissionData>
  <MissionFrameworkBucket FragmentFile="Missions_Tanner" Label="Tanner Arc (ch00)">
    <MissionFramework Label="ta10Buddies" Type="Story" DescriptionText="Under Attack">
      <MissionEvents>
        <MissionEvent Id="Created"><Action ActionType="Action.Background"/></MissionEvent>
        <MissionEvent Id="Activated"/>
      </MissionEvents>
      <ObjectiveBuckets>
        <ObjectiveBucket Label="SaveEd">
          <Objectives>
            <Objective Type="ClearArea" ObjectiveText="Survive!" TargetLocationOverride="4">
              <Location ActorLocation="4" Radius="45" ZombieCount="3"/>
              <LocationEvents>
                <LocationEvent ActorIndex="5" Radius="46"><Action/></LocationEvent>
              </LocationEvents>
            </Objective>
            <Objective Type="LeadNpcs" ObjectiveText="Go to the Ranger Station">
              <LocationEvents><LocationEvent ActorIndex="2" Radius="20"/></LocationEvents>
            </Objective>
            <Objective Type="SearchBuilding" MaxCount="2" Radius="75"/>
            <Objective Type="Wait" WaitTime="5" WaitTimeMax="5"/>
            <Objective Type="RescueNpc" RescueActorIndex="6" RescueActorIndex2="7"/>
            <Objective Type="KillActor" TargetActorIndex="4"/>
            <Objective Type="BogusVerb"/>
          </Objectives>
        </ObjectiveBucket>
      </ObjectiveBuckets>
    </MissionFramework>
  </MissionFrameworkBucket>
</RtsMissionData>"#;

    #[test]
    fn parses_framework_and_objectives() {
        let m = parse_mission_xml(SAMPLE.as_bytes()).expect("parse");
        assert_eq!(m.name, "ta10Buddies");
        assert_eq!(m.objectives.len(), 7);
        assert_eq!(m.objectives[0].kind, ObjectiveKind::ClearArea);
        assert_eq!(m.objectives[0].required, 3.0);
        assert_eq!(m.objectives[1].kind, ObjectiveKind::LeadNpcs);
        assert_eq!(m.objectives[2].kind, ObjectiveKind::SearchBuilding);
        assert_eq!(m.objectives[2].required, 2.0);
        assert_eq!(m.objectives[3].kind, ObjectiveKind::Wait);
        assert_eq!(m.objectives[3].required, 5.0);
        assert_eq!(m.objectives[4].kind, ObjectiveKind::RescueNpc);
        assert_eq!(m.objectives[4].required, 2.0);
        assert_eq!(m.objectives[5].kind, ObjectiveKind::KillActor);
        assert_eq!(m.objectives[5].required, 1.0);
        assert_eq!(m.objectives[6].kind, ObjectiveKind::Unknown);
        // first objective was auto-activated by Mission::new
        assert_eq!(m.active_index, 0);
        assert_eq!(m.current().unwrap().state, ObjectiveState::Active);
        // actor refs folded in from <Location>/<LocationEvent>
        assert!(m.objectives[0].actors.contains(&4));
        assert!(m.objectives[0].actors.contains(&5));
        assert!(m.objectives[1].actors.contains(&2));
    }

    #[test]
    fn parse_is_lenient_and_errors_cleanly() {
        assert!(parse_mission_xml(b"").is_err());
        assert!(parse_mission_xml(b"<not xml").is_err());
        // framework with no objectives still parses
        let m = parse_mission_xml(br#"<RtsMissionData><MissionFramework Label="x"/></RtsMissionData>"#)
            .unwrap();
        assert_eq!(m.name, "x");
        assert!(m.objectives.is_empty());
        assert!(!m.is_complete());
    }

    #[test]
    fn objective_kind_roundtrips() {
        for k in [
            ObjectiveKind::SearchBuilding,
            ObjectiveKind::SearchProp,
            ObjectiveKind::ClearInfestation,
            ObjectiveKind::RescueNpc,
            ObjectiveKind::Deliver,
            ObjectiveKind::Survey,
            ObjectiveKind::Siege,
            ObjectiveKind::BaseSiege,
            ObjectiveKind::Distraction,
            ObjectiveKind::Gather,
            ObjectiveKind::LeadNpcs,
            ObjectiveKind::Tour,
            ObjectiveKind::Escort,
            ObjectiveKind::Assassinate,
            ObjectiveKind::ClearArea,
            ObjectiveKind::InterceptHorde,
            ObjectiveKind::ScoutRescue,
            ObjectiveKind::ScoutResource,
            ObjectiveKind::SupplyRun,
            ObjectiveKind::Defend,
        ] {
            assert_eq!(ObjectiveKind::from_str(k.name()), k);
        }
        assert_eq!(ObjectiveKind::from_str("NeighborSiege"), ObjectiveKind::BaseSiege);
        assert_eq!(ObjectiveKind::from_str("nonsense"), ObjectiveKind::Unknown);
    }

    #[test]
    fn count_objective_completes_on_report() {
        let mut o = Objective::count(ObjectiveKind::SearchBuilding, 2);
        assert_eq!(o.state, ObjectiveState::Inactive);
        assert!(o.activate());
        assert!(!o.activate(), "activate is idempotent");
        o.report(1.0);
        assert_eq!(o.state, ObjectiveState::Active);
        assert!((o.fraction() - 0.5).abs() < 1e-6);
        o.report(1.5);
        assert_eq!(o.state, ObjectiveState::Complete);
        assert_eq!(o.fraction(), 1.0);
        // further reports are no-ops
        o.report(100.0);
        assert_eq!(o.progress, 2.5);
    }

    #[test]
    fn timed_objective_completes_on_tick() {
        let mut o = Objective::timed(ObjectiveKind::Wait, 5.0);
        o.activate();
        o.tick(2.0);
        assert_eq!(o.state, ObjectiveState::Active);
        assert!((o.fraction() - 0.4).abs() < 1e-6);
        o.tick(3.0);
        assert_eq!(o.state, ObjectiveState::Complete);

        // Ticks are ignored by count-based objectives.
        let mut c = Objective::count(ObjectiveKind::ClearArea, 1);
        c.activate();
        c.tick(100.0);
        assert_eq!(c.state, ObjectiveState::Active);
        // but a report finishes it.
        c.report(1.0);
        assert_eq!(c.state, ObjectiveState::Complete);
    }

    #[test]
    fn mission_advances_through_objectives() {
        let mut m = Mission::new(
            "test",
            vec![
                Objective::count(ObjectiveKind::ClearArea, 1),
                Objective::count(ObjectiveKind::LeadNpcs, 1),
                Objective::timed(ObjectiveKind::Wait, 2.0),
            ],
        );
        assert_eq!(m.current().unwrap().kind, ObjectiveKind::ClearArea);
        assert!(!m.advance(), "cannot advance before completing");

        m.current_mut().unwrap().report(1.0);
        assert!(m.advance());
        assert_eq!(m.active_index, 1);
        assert_eq!(m.current().unwrap().state, ObjectiveState::Active);
        assert!(!m.is_complete());

        m.current_mut().unwrap().report(1.0);
        assert!(m.advance());
        m.current_mut().unwrap().tick(2.0);
        assert!(m.is_complete());
        assert!(!m.advance(), "nothing left to advance to");
        assert_eq!(m.progress_fraction(), 1.0);
    }

    #[test]
    fn objective_failure_blocks_completion() {
        let mut m = Mission::new(
            "fail",
            vec![
                Objective::count(ObjectiveKind::Escort, 1),
                Objective::count(ObjectiveKind::Defend, 1),
            ],
        );
        m.current_mut().unwrap().fail();
        assert!(m.is_failed());
        assert!(m.advance());
        m.current_mut().unwrap().report(1.0);
        assert!(!m.is_complete());
        assert!((m.progress_fraction() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tutorial_new_tanner_and_trigger_matching() {
        let st = OpeningState::default();
        let mut t = Tutorial::new_tanner(&st);
        assert!(!t.steps.is_empty());
        assert_eq!(t.index, 0);
        assert_eq!(
            t.current().unwrap().trigger,
            TutorialTrigger::Custom("Scene.Move".to_string())
        );

        // wrong event does not advance
        assert!(!t.notify(&TutorialTrigger::SearchContainer));
        assert_eq!(t.index, 0);

        // matching custom event advances
        assert!(t.notify(&TutorialTrigger::Custom("Scene.Move".to_string())));
        assert_eq!(t.index, 1);

        // reach the survivor
        assert!(t.notify(&TutorialTrigger::ReachPoint(st.survivor_pos, 5.0)));
        assert_eq!(t.index, 2);
        // far away does not satisfy
        assert!(!t.notify(&TutorialTrigger::ReachPoint([999.0, 999.0, 999.0], 5.0)));

        // kill count: fewer than needed fails, enough advances
        assert!(!t.notify(&TutorialTrigger::KillCount(1)));
        assert!(t.notify(&TutorialTrigger::KillCount(st.kill_target.max(1))));
        assert_eq!(t.index, 3);

        assert!(t.notify(&TutorialTrigger::ReachPoint(st.ranger_station, 1.0)));
        assert!(t.notify(&TutorialTrigger::SearchContainer));
        assert!(t.notify(&TutorialTrigger::EquipWeapon));
        assert!(t.notify(&TutorialTrigger::ReachPoint(st.home_pos, 1.0)));
        assert!(t.is_complete());
        assert!(t.current().is_none());
    }

    #[test]
    fn talk_to_trigger_matches_actor() {
        let mut t = Tutorial {
            steps: vec![TutorialStep {
                text: "Talk".to_string(),
                trigger: TutorialTrigger::TalkTo(7),
            }],
            index: 0,
        };
        assert!(!t.notify(&TutorialTrigger::TalkTo(3)));
        assert!(t.notify(&TutorialTrigger::TalkTo(7)));
        assert!(t.is_complete());
    }

    #[test]
    fn radio_spends_and_cools_down() {
        let mut r = RadioState::new(100, [10, 20, 30, 40]);
        assert!(r.eval(RadioOption::AirStrike));
        assert!(r.exec(RadioOption::AirStrike));
        assert_eq!(r.influence, 90);
        assert_eq!(r.cooldowns[0], RadioOption::AirStrike.cooldown());
        // on cooldown -> unavailable, exec is a no-op
        assert!(!r.eval(RadioOption::AirStrike));
        assert!(!r.exec(RadioOption::AirStrike));
        assert_eq!(r.influence, 90);

        // other option still ready
        assert!(r.exec(RadioOption::MortarBarrage));
        assert_eq!(r.influence, 50);

        // cooldown expires
        r.tick(1000.0);
        assert!(r.eval(RadioOption::AirStrike));
    }

    #[test]
    fn radio_rejects_unaffordable() {
        let mut r = RadioState::new(5, [10, 20, 30, 40]);
        assert!(!r.eval(RadioOption::AirStrike));
        assert!(!r.exec(RadioOption::AirStrike));
        assert_eq!(r.influence, 5, "failed exec spends nothing");
        assert_eq!(r.cooldowns[0], 0.0, "failed exec starts no cooldown");
    }

    #[test]
    fn radio_tick_clamps_at_zero() {
        let mut r = RadioState::default();
        r.exec(RadioOption::ArtilleryBarrage);
        r.tick(10.0);
        assert!((r.cooldowns[1] - (240.0 - 10.0)).abs() < 1e-3);
        r.tick(1_000_000.0);
        assert_eq!(r.cooldowns, [0.0; 4]);
    }

    // -----------------------------------------------------------------------
    // Whole-file catalogue (`parse_missions_xml`)
    // -----------------------------------------------------------------------

    const MISSIONS_SAMPLE: &str = r#"<?xml version="1.0"?>
<RtsMissionData>
  <MissionRtsData/>
  <MissionFrameworkBucket FragmentFile="Missions_Tanner" Label="Tanner Arc (ch00)">
    <MissionFramework Label="ta10Buddies" Type="Story" OpportunityText="Under Attack">
      <MissionActors><Actors>
        <Character ActorTraits="Trait.Story.Buddy" FunctionTags="Npc" />
        <TagPoint ActorName="RTSPoint_RangerStation" NearActorIdx="-1" />
        <Player />
      </Actors></MissionActors>
      <MissionEvents>
        <MissionEvent Id="Created">
          <Action ActionType="Action.Background">
            <Actors />
            <Outputs>
              <Zombie.Spawn SpawnCount="1" />
              <Hint_Enable HintId="Sprint" SetEnabled="false" />
            </Outputs>
          </Action>
        </MissionEvent>
        <MissionEvent Id="Completed">
          <Action><Inputs /><Outputs><EventLog /></Outputs></Action>
        </MissionEvent>
      </MissionEvents>
      <ObjectiveBuckets><ObjectiveBucket Label="SaveEd"><Objectives>
        <Objective Type="ClearArea" HeadlineText="UNDER ATTACK">
          <Location ActorLocation="4" ZombieCount="3" />
        </Objective>
        <Objective Type="LeadNpcs" ObjectiveText="Go to the Ranger Station" />
      </Objectives></ObjectiveBucket></ObjectiveBuckets>
    </MissionFramework>
    <MissionFramework Label="ta15Go" Type="Story" />
  </MissionFrameworkBucket>
  <MissionFrameworkBucket FragmentFile="Missions_Army" Label="Army Arc (ar)">
    <MissionFramework Label="ar01Request" Type="Story" />
  </MissionFrameworkBucket>
</RtsMissionData>"#;

    #[test]
    fn parse_missions_xml_reads_defs_actors_and_actions() {
        let defs = parse_missions_xml(MISSIONS_SAMPLE.as_bytes());
        assert_eq!(defs.len(), 3, "three MissionFramework elements");

        let m = &defs[0];
        assert_eq!(m.label, "ta10Buddies");
        assert_eq!(m.bucket, "Tanner Arc (ch00)");
        assert_eq!(
            m.objective_kinds,
            vec![ObjectiveKind::ClearArea, ObjectiveKind::LeadNpcs]
        );

        assert_eq!(m.actors.len(), 3);
        assert_eq!(m.actors[0].index, 0);
        assert_eq!(m.actors[0].character.as_deref(), Some("Trait.Story.Buddy"));
        assert!(m.actors[0].traits.iter().any(|t| t == "Trait.Story.Buddy"));
        assert!(m.actors[0].traits.iter().any(|t| t == "Npc"));
        assert_eq!(
            m.actors[1].tag_point.as_deref(),
            Some("RTSPoint_RangerStation")
        );
        assert!(m.actors[1].character.is_none());

        // Only the Created event's outputs are collected (no EventLog leak).
        assert!(m.created_actions.iter().any(|a| a == "Zombie.Spawn"));
        assert!(m.created_actions.iter().any(|a| a == "Hint_Enable"));
        assert!(!m.created_actions.iter().any(|a| a == "EventLog"));

        assert_eq!(defs[1].label, "ta15Go");
        assert!(defs[1].objective_kinds.is_empty());
        assert_eq!(defs[2].label, "ar01Request");
        assert_eq!(defs[2].bucket, "Army Arc (ar)");
    }

    #[test]
    fn tanner_arc_and_flow_from_sample() {
        let defs = parse_missions_xml(MISSIONS_SAMPLE.as_bytes());
        let arc = tanner_arc(&defs);
        let labels: Vec<&str> = arc.iter().map(|d| d.label.as_str()).collect();
        assert_eq!(labels, vec!["ta10Buddies", "ta15Go"]);
        assert_eq!(
            tutorial_flow(&defs),
            vec![
                "ClearArea".to_string(),
                "LeadNpcs".to_string(),
                "ta15Go".to_string(),
            ]
        );
    }

    #[test]
    fn parse_missions_xml_never_panics_on_junk() {
        let _ = parse_missions_xml(b"");
        let _ = parse_missions_xml(b"<not xml");
        let _ = parse_missions_xml(b"\x00\xff\xfe<RtsMissionData>");
        assert!(parse_missions_xml(b"<RtsMissionData/>").is_empty());
    }

    fn load_real_missions() -> Option<Vec<u8>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../../extracted-game/gamedata/libs/class3/contentmanager/missions.xml",
        );
        std::fs::read(path).ok()
    }

    #[test]
    fn real_missions_parse_many_defs() {
        let bytes = match load_real_missions() {
            Some(b) => b,
            None => return, // real data not present: skip silently
        };
        let defs = parse_missions_xml(&bytes);
        assert!(
            defs.len() > 100,
            "expected >100 mission defs, got {}",
            defs.len()
        );
    }

    #[test]
    fn real_tanner_arc_has_expected_labels() {
        let bytes = match load_real_missions() {
            Some(b) => b,
            None => return,
        };
        let defs = parse_missions_xml(&bytes);
        let arc = tanner_arc(&defs);
        assert!(!arc.is_empty(), "tanner arc is empty");
        let labels: Vec<&str> = arc.iter().map(|d| d.label.as_str()).collect();
        assert!(labels.contains(&"ta10Buddies"), "labels: {labels:?}");
        assert!(labels.contains(&"ta40Bridge"), "labels: {labels:?}");
    }

    #[test]
    fn real_ta10buddies_has_clear_area_and_spawn_actions() {
        let bytes = match load_real_missions() {
            Some(b) => b,
            None => return,
        };
        let defs = parse_missions_xml(&bytes);
        let arc = tanner_arc(&defs);
        let m = arc
            .iter()
            .find(|d| d.label == "ta10Buddies")
            .expect("ta10Buddies in tanner arc");
        assert!(
            m.objective_kinds.contains(&ObjectiveKind::ClearArea),
            "kinds: {:?}",
            m.objective_kinds
        );
        assert!(
            m.created_actions
                .iter()
                .any(|a| a.contains("Zombie.Spawn") || a.contains("Hint")),
            "created actions: {:?}",
            m.created_actions
        );
    }

    // -----------------------------------------------------------------------
    // Native mission runtime
    // -----------------------------------------------------------------------

    fn mkdef(label: &str, kinds: Vec<ObjectiveKind>, actions: &[&str]) -> MissionDef {
        MissionDef {
            label: label.to_string(),
            bucket: "Test Arc".to_string(),
            objective_kinds: kinds,
            actors: Vec::new(),
            created_actions: actions.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn runtime_start_activates_first_objective_and_emits_created_actions() {
        let mut rt = MissionRuntime::new(vec![mkdef(
            "ta10Buddies",
            vec![ObjectiveKind::ClearArea, ObjectiveKind::LeadNpcs],
            &["Zombie.Spawn", "Hint_Enable"],
        )]);

        let fx = rt.start("ta10Buddies");
        assert_eq!(rt.active().len(), 1);
        assert_eq!(rt.active()[0].label, "ta10Buddies");
        assert_eq!(
            rt.active()[0].objectives[0].state,
            ObjectiveState::Active,
            "first objective auto-activated"
        );
        assert_eq!(
            rt.active()[0].objectives[1].state,
            ObjectiveState::Inactive,
            "later objectives stay inactive"
        );
        assert_eq!(
            fx,
            vec![
                MissionEffect::SpawnZombie {
                    kind: "Zombie.Walker".to_string(),
                    count: 1,
                },
                MissionEffect::HintEnable {
                    id: String::new(),
                    enabled: true,
                },
            ]
        );
        assert_eq!(rt.effects(), fx.as_slice());

        // Unknown labels and double-starts are no-ops.
        assert!(rt.start("nope").is_empty());
        assert!(rt.start("ta10Buddies").is_empty());
        assert!(!rt.is_complete("ta10Buddies"));
    }

    #[test]
    fn runtime_zombies_cleared_completes_clear_area_then_mission() {
        let mut rt = MissionRuntime::new(vec![mkdef(
            "ta10Buddies",
            vec![ObjectiveKind::ClearArea],
            &[],
        )]);
        rt.start("ta10Buddies");

        // Partial: two of three still alive, objective stays active.
        let fx = rt.zombies_cleared([0.0, 0.0, 0.0], 2);
        assert!(fx.is_empty());
        assert!(!rt.is_complete("ta10Buddies"));
        assert_eq!(rt.active()[0].objectives[0].state, ObjectiveState::Active);

        // Last zombie down: objective and the single-objective mission complete.
        let fx = rt.zombies_cleared([0.0, 0.0, 0.0], 0);
        assert!(matches!(
            fx.first(),
            Some(MissionEffect::CompleteObjective { mission, objective })
                if mission == "ta10Buddies" && objective == "ClearArea"
        ));
        assert!(rt.is_complete("ta10Buddies"));
        assert!(rt.active().is_empty(), "finished mission is reaped");
    }

    #[test]
    fn runtime_actor_entered_completes_lead_npcs() {
        let mut rt = MissionRuntime::new(vec![mkdef(
            "ta15Go",
            vec![ObjectiveKind::LeadNpcs, ObjectiveKind::ClearArea],
            &[],
        )]);
        rt.start("ta15Go");

        let fx = rt.actor_entered(2, [10.0, 0.0, 0.0]);
        assert!(matches!(
            fx.first(),
            Some(MissionEffect::CompleteObjective { mission, objective })
                if mission == "ta15Go" && objective == "LeadNpcs"
        ));
        assert_eq!(
            rt.active()[0].objectives[0].state,
            ObjectiveState::Complete
        );
        assert_eq!(
            rt.active()[0].objectives[1].state,
            ObjectiveState::Active,
            "next objective activated after the escort"
        );
        assert!(!rt.is_complete("ta15Go"), "ClearArea remains");
    }

    #[test]
    fn runtime_building_searched_completes_search_building() {
        let mut rt = MissionRuntime::new(vec![mkdef(
            "ta20Forest",
            vec![ObjectiveKind::SearchBuilding],
            &[],
        )]);
        rt.start("ta20Forest");

        let fx = rt.building_searched(1);
        assert!(fx
            .iter()
            .any(|e| matches!(e, MissionEffect::CompleteObjective { objective, .. } if objective == "SearchBuilding")));
        assert!(rt.is_complete("ta20Forest"));
    }

    #[test]
    fn runtime_report_progress_partial_then_complete() {
        let mut rt = MissionRuntime::new(vec![mkdef(
            "ch03Meds",
            vec![ObjectiveKind::Gather, ObjectiveKind::Wait],
            &[],
        )]);
        rt.start("ch03Meds");

        let fx = rt.report_progress("ch03Meds", 0, 0.4);
        assert!(fx.is_empty(), "0.4 < required 1.0");
        assert!(!rt.is_complete("ch03Meds"));
        assert!((rt.active()[0].objectives[0].progress - 0.4).abs() < 1e-6);

        // reporting a non-active objective is ignored
        assert!(rt.report_progress("ch03Meds", 1, 5.0).is_empty());
        assert_eq!(
            rt.active()[0].objectives[1].state,
            ObjectiveState::Inactive
        );

        let fx = rt.report_progress("ch03Meds", 0, 0.6);
        assert!(fx
            .iter()
            .any(|e| matches!(e, MissionEffect::CompleteObjective { objective, .. } if objective == "Gather")));
        assert_eq!(
            rt.active()[0].objectives[1].state,
            ObjectiveState::Active
        );
        assert!(!rt.is_complete("ch03Meds"));

        // Finish the final Wait objective via report.
        rt.report_progress("ch03Meds", 1, 1.0);
        assert!(rt.is_complete("ch03Meds"));
    }

    #[test]
    fn runtime_effect_from_action_maps_representative_verbs() {
        let s = |v: &str| v.to_string();
        assert_eq!(
            effect_from_action("Zombie.Spawn", &[s("Zombie.Eatter"), s("3")]),
            Some(MissionEffect::SpawnZombie {
                kind: s("Zombie.Eatter"),
                count: 3,
            })
        );
        assert_eq!(
            effect_from_action("Zombie.Spawn", &[]),
            Some(MissionEffect::SpawnZombie {
                kind: s("Zombie.Walker"),
                count: 1,
            })
        );
        assert_eq!(
            effect_from_action("Zombie.FreakSpawn", &[s("Zombie.Eatter"), s("4")]),
            Some(MissionEffect::SpawnFreak {
                kind: s("Zombie.Eatter"),
                count: 4,
            })
        );
        assert_eq!(
            effect_from_action("Hint_Show", &[s("Scene.Combat")]),
            Some(MissionEffect::HintShow {
                id: s("Scene.Combat"),
            })
        );
        assert_eq!(
            effect_from_action("Hint_Enable", &[s("Sprint"), s("false")]),
            Some(MissionEffect::HintEnable {
                id: s("Sprint"),
                enabled: false,
            })
        );
        assert_eq!(
            effect_from_action("RTSStat_SetBool", &[s("Game.StoryTA.ReadyForGunshots"), s("true")]),
            Some(MissionEffect::SetStatBool {
                stat: s("Game.StoryTA.ReadyForGunshots"),
                value: true,
            })
        );
        assert_eq!(
            effect_from_action("Mission_CompleteObjective", &[s("ta10Buddies"), s("ClearArea")]),
            Some(MissionEffect::CompleteObjective {
                mission: s("ta10Buddies"),
                objective: s("ClearArea"),
            })
        );
        assert_eq!(
            effect_from_action("Mission_Create", &[s("ta30Return")]),
            Some(MissionEffect::CreateMission {
                label: s("ta30Return"),
            })
        );
        assert_eq!(
            effect_from_action("Mission_AbortMission", &[s("ta20Forest")]),
            Some(MissionEffect::AbortMission {
                label: s("ta20Forest"),
            })
        );
        assert_eq!(
            effect_from_action("Scene_ScenePlay", &[s("ta10Buddies_Start")]),
            Some(MissionEffect::PlayScene {
                label: s("ta10Buddies_Start"),
            })
        );
        assert_eq!(
            effect_from_action("SetStoryFlag", &[s("Game.Tutorial")]),
            Some(MissionEffect::SetStoryFlag {
                flag: s("Game.Tutorial"),
            })
        );
        assert_eq!(
            effect_from_action("EventLog", &[]),
            Some(MissionEffect::Log("EventLog".to_string()))
        );
        assert_eq!(effect_from_action("Actor_ApplyBlood", &[]), None);
        assert_eq!(effect_from_action("  ", &[]), None);
    }

    #[test]
    fn runtime_tick_advances_timed_objective() {
        let mut rt = MissionRuntime::new(vec![mkdef(
            "ar04Bridges",
            vec![ObjectiveKind::Wait],
            &[],
        )]);
        rt.start("ar04Bridges");
        let fx = rt.tick(&[]);
        assert!(fx
            .iter()
            .any(|e| matches!(e, MissionEffect::CompleteObjective { objective, .. } if objective == "Wait")));
        assert!(rt.is_complete("ar04Bridges"));
    }
}
