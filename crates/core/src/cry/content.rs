//! Content-manager data model and parsers.
//!
//! State of Decay's story missions, objectives and the enclave story state
//! machine are authored as XML under `gamedata/libs/class3/`:
//!
//! * `contentmanager/missions.xml` — root `<RtsMissionData>`, 25
//!   `<MissionFrameworkBucket>`s holding 119 `<MissionFramework>` missions.
//!   Each mission owns an indexed actor table, lifecycle events
//!   (`Created`/`Activated`/`Completed`/`Aborted`/…) and an ordered list of
//!   `<ObjectiveBucket>`s whose `<Objective Type=…>` selects the runtime
//!   behaviour (`ClearArea`, `LeadNpcs`, `SearchBuilding`, `Survey`, …).
//! * `rts/enclaves.xml` — the `Home` enclave's `<State>` machine; the
//!   `Buddies → Mountain → Escape → PreChurch` chain emits `Mission_Create`
//!   outputs that start the Tanner-arc tutorial missions.
//!
//! The original code that consumes this data is the Class3
//! `NSContentManager::CContentManager` module. Object addresses recovered from
//! the unpacked image (base `0x400000`, Class3 image) are cited inline; they
//! come from `notes/ghidra/class3_decomp` (see `notes/content-manager.md`).
//! The parser here is a clean-room reading of the XML, not a copy of any game
//! file.
//!
//! Nothing in this module depends on the engine: it is pure data + parsing so
//! it can be unit-tested hermetically. The runtime lives in
//! [`super::content_runtime`].

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

fn attrs_of(e: &BytesStart) -> Vec<(String, String)> {
    e.attributes()
        .flatten()
        .map(|a| {
            let k = String::from_utf8_lossy(a.key.as_ref()).to_string();
            let v = a
                .unescape_value()
                .map(|v| v.to_string())
                .unwrap_or_else(|_| String::from_utf8_lossy(&a.value).to_string());
            (k, v)
        })
        .collect()
}

/// A generic XML element: name, attributes and child elements. Text nodes are
/// deliberately discarded — the mission schema keeps all meaning in attributes
/// (the only text is HUD copy, which the runtime does not need to reinterpret).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Element {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Element>,
}

impl Element {
    pub fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }

    pub fn attr_i32(&self, k: &str) -> Option<i32> {
        self.attr(k).and_then(|v| v.trim().parse().ok())
    }

    pub fn attr_u32(&self, k: &str) -> Option<u32> {
        self.attr(k).and_then(|v| v.trim().parse().ok())
    }

    pub fn attr_f32(&self, k: &str) -> Option<f32> {
        self.attr(k).and_then(|v| v.trim().parse().ok())
    }

    /// `True`/`true`/`1` → true. Absent attribute → `default`.
    pub fn attr_bool(&self, k: &str) -> bool {
        self.attr_bool_or(k, false)
    }

    pub fn attr_bool_or(&self, k: &str, default: bool) -> bool {
        match self.attr(k) {
            None => default,
            Some(v) => matches!(v.trim(), "True" | "true" | "1"),
        }
    }

    pub fn child(&self, name: &str) -> Option<&Element> {
        self.children.iter().find(|c| c.name == name)
    }

    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Element> {
        self.children.iter().filter(move |c| c.name == name)
    }

    /// Comma-separated attribute split into trimmed, non-empty items.
    pub fn csv(&self, k: &str) -> Vec<String> {
        split_csv(self.attr(k).unwrap_or(""))
    }

    fn descendants<'a>(&'a self, name: &str, out: &mut Vec<&'a Element>) {
        for c in &self.children {
            if c.name == name {
                out.push(c);
            }
            c.descendants(name, out);
        }
    }

    pub fn find_all<'a>(&'a self, name: &str) -> Vec<&'a Element> {
        let mut out = Vec::new();
        self.descendants(name, &mut out);
        out
    }
}

pub fn split_csv(s: &str) -> Vec<String> {
    s.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()).map(str::to_string).collect()
}

/// Parse an XML document into an [`Element`] tree. Lenient: malformed input
/// stops the scan and returns whatever tree was built so far (possibly `None`).
pub fn parse_document(xml: &[u8]) -> Option<Element> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut stack: Vec<Element> = Vec::new();
    let mut root: Option<Element> = None;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => stack.push(element_from(&e)),
            Ok(Event::Empty(e)) => {
                let el = element_from(&e);
                match stack.last_mut() {
                    Some(p) => p.children.push(el),
                    None => root = Some(el),
                }
            }
            Ok(Event::End(_)) => {
                if let Some(el) = stack.pop() {
                    match stack.last_mut() {
                        Some(p) => p.children.push(el),
                        None => root = Some(el),
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    root
}

fn element_from(e: &BytesStart) -> Element {
    Element {
        name: String::from_utf8_lossy(e.name().as_ref()).to_string(),
        attrs: attrs_of(e),
        children: Vec::new(),
    }
}

/// `"5s"` / `"0.0"` / `"1.5"` → seconds. Bare numbers are seconds.
pub fn parse_seconds(s: &str) -> f32 {
    let s = s.trim();
    if s.is_empty() {
        return 0.0;
    }
    let num = s.trim_end_matches('s').trim();
    num.parse::<f32>().unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Mission schema
// ---------------------------------------------------------------------------

/// Kind of an entry in a mission's indexed actor table. The index into the
/// table is what every `ActorIndex`/`NearActorIdx`/`OutputActorIndex` in the
/// XML refers to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActorKind {
    Player,
    Character,
    Zombie,
    TagPoint,
    Building,
    Enclave,
    HomeSite,
    SurveyGroup,
    Cage,
    Infestation,
    Unknown(String),
}

impl ActorKind {
    pub fn from_name(s: &str) -> ActorKind {
        match s {
            "Player" => ActorKind::Player,
            "Character" => ActorKind::Character,
            "Zombie" => ActorKind::Zombie,
            "TagPoint" => ActorKind::TagPoint,
            "Building" => ActorKind::Building,
            "Enclave" => ActorKind::Enclave,
            "HomeSite" => ActorKind::HomeSite,
            "SurveyGroup" => ActorKind::SurveyGroup,
            "Cage" => ActorKind::Cage,
            "Infestation" => ActorKind::Infestation,
            other => ActorKind::Unknown(other.to_string()),
        }
    }
}

/// One mission actor. Only the fields the runtime resolves positions/traits
/// from are promoted; the rest stay in [`ActorSpec::attrs`].
#[derive(Clone, Debug, Default)]
pub struct ActorSpec {
    pub kind: ActorKind,
    pub actor_name: String,
    pub actor_traits: Vec<String>,
    pub function_tags: Vec<String>,
    pub near_actor_idx: i32,
    pub actor_flags: Vec<String>,
    pub optional: bool,
    pub attrs: Vec<(String, String)>,
}

impl Default for ActorKind {
    fn default() -> Self {
        ActorKind::Unknown(String::new())
    }
}

impl ActorSpec {
    fn from_element(e: &Element) -> ActorSpec {
        ActorSpec {
            kind: ActorKind::from_name(&e.name),
            actor_name: e.attr("ActorName").unwrap_or("").to_string(),
            actor_traits: e.csv("ActorTraits"),
            function_tags: e.csv("FunctionTags"),
            near_actor_idx: e.attr_i32("NearActorIdx").unwrap_or(-1),
            actor_flags: e.csv("ActorFlags"),
            optional: e.attr_bool("Optional"),
            attrs: e.attrs.clone(),
        }
    }
}

/// One `<Action>`: a condition gate (`inputs`) and an effect list (`outputs`).
/// `alts` describes the weighted sub-actions selected by `ActionFlags="Alt"`.
#[derive(Clone, Debug, Default)]
pub struct Action {
    pub id: String,
    pub action_type: String,
    pub flags: Vec<String>,
    pub execute_time: f32,
    pub output_delay: f32,
    pub inputs: Vec<Element>,
    pub outputs: Vec<Element>,
    pub alts: Vec<(String, f32)>,
}

impl Action {
    fn from_element(e: &Element) -> Action {
        let alts = e
            .child("Alts")
            .map(|a| {
                a.children_named("Alt")
                    .map(|alt| {
                        (
                            alt.attr("AltId").unwrap_or("").to_string(),
                            alt.attr_f32("AltOdds").unwrap_or(0.0),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        Action {
            id: e.attr("ActionId").unwrap_or("").to_string(),
            action_type: e.attr("ActionType").unwrap_or("").to_string(),
            flags: e.csv("ActionFlags"),
            execute_time: e.attr("ExecuteTimeStandard").map(parse_seconds).unwrap_or(0.0),
            output_delay: e.attr("OutputDelayTime").map(parse_seconds).unwrap_or(0.0),
            inputs: e.child("Inputs").map(|i| i.children.clone()).unwrap_or_default(),
            outputs: e.child("Outputs").map(|o| o.children.clone()).unwrap_or_default(),
            alts,
        }
    }
}

/// A named lifecycle event (`Created`, `Activated`, `Completed`, `Aborted`, …).
#[derive(Clone, Debug, Default)]
pub struct MissionEvent {
    pub id: String,
    pub actions: Vec<Action>,
}

#[derive(Clone, Debug, Default)]
pub struct ObjectiveLocation {
    pub actor_location: i32,
    pub radius: f32,
    pub zombie_count: u32,
    pub in_building: bool,
    pub attrs: Vec<(String, String)>,
}

/// A proximity event attached to an objective: fires when the player reaches
/// the actor named by `actor_index` (edge-triggered). The original checks the
/// distance in `CObjectiveBase::UpdateLocationEvents` (`Class3:014da7d0`).
#[derive(Clone, Debug, Default)]
pub struct LocationEvent {
    pub actor_index: i32,
    pub radius: f32,
    pub flags: Vec<String>,
    pub actions: Vec<Action>,
    pub leave_actions: Vec<Action>,
    pub attrs: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default)]
pub struct TimedEvent {
    pub timer: f32,
    pub repeat: bool,
    pub actions: Vec<Action>,
}

/// One `<Objective>`. `kind` is the parse of `Type=`; type-specific children
/// remain in [`Objective::children`] for lookups (e.g. `Waves`, `Site`).
#[derive(Clone, Debug, Default)]
pub struct Objective {
    pub type_name: String,
    pub kind: ObjectiveType,
    pub text: String,
    pub headline: String,
    pub flags: Vec<String>,
    pub activation_scene: String,
    pub completion_scene: String,
    pub target_location_override: i32,
    pub complete_threshold: u32,
    pub locations: Vec<ObjectiveLocation>,
    pub location_events: Vec<LocationEvent>,
    pub events: Vec<(String, Vec<Action>)>,
    pub timed_events: Vec<TimedEvent>,
    /// Direct children other than the well-known sections (type-specific).
    pub children: Vec<Element>,
    pub attrs: Vec<(String, String)>,
}

impl Default for ObjectiveType {
    fn default() -> Self {
        ObjectiveType::Unknown
    }
}

impl Objective {
    pub fn event(&self, id: &str) -> &[Action] {
        self.events
            .iter()
            .find(|(k, _)| k == id)
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[])
    }

    fn from_element(e: &Element) -> Objective {
        let locations = e
            .children_named("Location")
            .map(|l| ObjectiveLocation {
                actor_location: l.attr_i32("ActorLocation").unwrap_or(-1),
                radius: l.attr_f32("Radius").unwrap_or(0.0),
                zombie_count: l.attr_u32("ZombieCount").unwrap_or(0),
                in_building: l.attr_bool("InBuilding"),
                attrs: l.attrs.clone(),
            })
            .collect();
        let location_events = e
            .child("LocationEvents")
            .map(|les| {
                les.children_named("LocationEvent")
                    .map(|le| LocationEvent {
                        actor_index: le.attr_i32("ActorIndex").unwrap_or(-1),
                        radius: le.attr_f32("Radius").unwrap_or(0.0),
                        flags: le.csv("Flags"),
                        actions: le.children_named("Action").map(Action::from_element).collect(),
                        leave_actions: le
                            .child("LeaveActions")
                            .map(|la| la.children_named("Action").map(Action::from_element).collect())
                            .unwrap_or_default(),
                        attrs: le.attrs.clone(),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let events = e
            .child("ObjectiveEvents")
            .map(|oe| {
                oe.children_named("ObjectiveEvent")
                    .map(|ev| {
                        let id = ev.attr("Id").unwrap_or("").to_string();
                        let acts =
                            ev.children_named("Action").map(Action::from_element).collect();
                        (id, acts)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let timed_events = e
            .child("TimedEvents")
            .map(|te| {
                te.children_named("TimedEvent")
                    .map(|t| TimedEvent {
                        timer: t.attr("Timer").map(parse_seconds).unwrap_or(0.0),
                        repeat: t.attr_bool("Repeat"),
                        actions: t.children_named("Action").map(Action::from_element).collect(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Keep every child that is not a parsed section, so type-specific data
        // (Siege `<Waves>`, Survey `<Site>`, LeadNpcs `<Threshold>`…) survives.
        const SECTIONS: &[&str] =
            &["Location", "LocationEvents", "ObjectiveEvents", "TimedEvents", "OverrideScenes", "ProgressScenes"];
        let children =
            e.children.iter().filter(|c| !SECTIONS.contains(&c.name.as_str())).cloned().collect();
        let type_name = e.attr("Type").unwrap_or("").to_string();
        Objective {
            kind: ObjectiveType::from_str(&type_name),
            type_name,
            text: e.attr("ObjectiveText").unwrap_or("").to_string(),
            headline: e.attr("HeadlineText").unwrap_or("").to_string(),
            flags: e.csv("Flags"),
            activation_scene: e.attr("ActivationScene").unwrap_or("").to_string(),
            completion_scene: e.attr("CompletionScene").unwrap_or("").to_string(),
            target_location_override: e.attr_i32("TargetLocationOverride").unwrap_or(-1),
            complete_threshold: e.attr_u32("CompleteThreshold").unwrap_or(0),
            locations,
            location_events,
            events,
            timed_events,
            children,
            attrs: e.attrs.clone(),
        }
    }
}

/// Objective verbs observed in `missions.xml`, plus procedural neighbours.
/// Mirrors the runtime classes registered by `NSRTSActions::RAInit`
/// (`Class3:0146daf0`) and the parse entry points in `content.rs`
/// (`NSContentManager::ParseObjective*Node`, `Class3:014c30e0`+).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjectiveType {
    ClearArea,
    LeadNpcs,
    SearchBuilding,
    Survey,
    SearchProp,
    RescueNpc,
    KillActor,
    Assassinate,
    LeaveArea,
    Siege,
    Gather,
    Tour,
    Distraction,
    Wait,
    InteractNpc,
    ClearInfestation,
    Deliver,
    BaseSiege,
    Escort,
    Unknown,
}

impl ObjectiveType {
    pub fn from_str(s: &str) -> ObjectiveType {
        match s.trim() {
            "ClearArea" => ObjectiveType::ClearArea,
            "LeadNpcs" => ObjectiveType::LeadNpcs,
            "SearchBuilding" => ObjectiveType::SearchBuilding,
            "Survey" => ObjectiveType::Survey,
            "SearchProp" => ObjectiveType::SearchProp,
            "RescueNpc" => ObjectiveType::RescueNpc,
            "KillActor" => ObjectiveType::KillActor,
            "Assassinate" => ObjectiveType::Assassinate,
            "LeaveArea" => ObjectiveType::LeaveArea,
            "Siege" => ObjectiveType::Siege,
            "Gather" => ObjectiveType::Gather,
            "Tour" => ObjectiveType::Tour,
            "Distraction" => ObjectiveType::Distraction,
            "Wait" => ObjectiveType::Wait,
            "InteractNpc" => ObjectiveType::InteractNpc,
            "ClearInfestation" => ObjectiveType::ClearInfestation,
            "Deliver" => ObjectiveType::Deliver,
            "BaseSiege" => ObjectiveType::BaseSiege,
            "Escort" => ObjectiveType::Escort,
            _ => ObjectiveType::Unknown,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ObjectiveType::ClearArea => "ClearArea",
            ObjectiveType::LeadNpcs => "LeadNpcs",
            ObjectiveType::SearchBuilding => "SearchBuilding",
            ObjectiveType::Survey => "Survey",
            ObjectiveType::SearchProp => "SearchProp",
            ObjectiveType::RescueNpc => "RescueNpc",
            ObjectiveType::KillActor => "KillActor",
            ObjectiveType::Assassinate => "Assassinate",
            ObjectiveType::LeaveArea => "LeaveArea",
            ObjectiveType::Siege => "Siege",
            ObjectiveType::Gather => "Gather",
            ObjectiveType::Tour => "Tour",
            ObjectiveType::Distraction => "Distraction",
            ObjectiveType::Wait => "Wait",
            ObjectiveType::InteractNpc => "InteractNpc",
            ObjectiveType::ClearInfestation => "ClearInfestation",
            ObjectiveType::Deliver => "Deliver",
            ObjectiveType::BaseSiege => "BaseSiege",
            ObjectiveType::Escort => "Escort",
            ObjectiveType::Unknown => "Unknown",
        }
    }
}

/// An ordered phase of a mission: `<ObjectiveBucket>` → 1..n objectives.
#[derive(Clone, Debug, Default)]
pub struct ObjectiveBucket {
    pub label: String,
    pub objectives: Vec<Objective>,
}

/// One `<MissionFramework>`.
#[derive(Clone, Debug, Default)]
pub struct Mission {
    pub label: String,
    pub type_name: String,
    pub activate_type: String,
    pub opportunity_text: String,
    pub description_text: String,
    pub opportunity_scene: String,
    pub setup_scene: String,
    pub priority: i32,
    pub min_start_dist: f32,
    pub approach_radius: f32,
    pub actors: Vec<ActorSpec>,
    pub events: Vec<MissionEvent>,
    pub objective_buckets: Vec<ObjectiveBucket>,
    pub attrs: Vec<(String, String)>,
}

impl Mission {
    pub fn event(&self, id: &str) -> &[Action] {
        self.events.iter().find(|e| e.id == id).map(|e| e.actions.as_slice()).unwrap_or(&[])
    }

    pub fn actor(&self, index: usize) -> Option<&ActorSpec> {
        self.actors.get(index)
    }

    fn from_element(e: &Element) -> Mission {
        let actors = e
            .child("MissionActors")
            .and_then(|ma| ma.child("Actors"))
            .map(|a| a.children.iter().map(ActorSpec::from_element).collect())
            .unwrap_or_default();
        let events = e
            .child("MissionEvents")
            .map(|me| {
                me.children_named("MissionEvent")
                    .map(|ev| MissionEvent {
                        id: ev.attr("Id").unwrap_or("").to_string(),
                        actions: ev.children_named("Action").map(Action::from_element).collect(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let objective_buckets = e
            .child("ObjectiveBuckets")
            .map(|ob| {
                ob.children_named("ObjectiveBucket")
                    .map(|b| ObjectiveBucket {
                        label: b.attr("Label").unwrap_or("").to_string(),
                        objectives: b
                            .child("Objectives")
                            .map(|os| os.children_named("Objective").map(Objective::from_element).collect())
                            .unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Mission {
            label: e.attr("Label").unwrap_or("").to_string(),
            type_name: e.attr("Type").unwrap_or("").to_string(),
            activate_type: e.attr("ActivateType").unwrap_or("").to_string(),
            opportunity_text: e.attr("OpportunityText").unwrap_or("").to_string(),
            description_text: e.attr("DescriptionText").unwrap_or("").to_string(),
            opportunity_scene: e.attr("OpportunityScene").unwrap_or("").to_string(),
            setup_scene: e.attr("SetupScene").unwrap_or("").to_string(),
            priority: e.attr_i32("Priority").unwrap_or(0),
            min_start_dist: e.attr_f32("MinStartDist").unwrap_or(0.0),
            approach_radius: e.attr_f32("ApproachRadius").unwrap_or(0.0),
            actors,
            events,
            objective_buckets,
            attrs: e.attrs.clone(),
        }
    }
}

/// One `<MissionFrameworkBucket>` (an "arc": Tanner, Army, Church, …).
#[derive(Clone, Debug, Default)]
pub struct MissionBucket {
    pub label: String,
    pub fragment_file: String,
    pub missions: Vec<Mission>,
}

/// Root `<RtsMissionData>`.
#[derive(Clone, Debug, Default)]
pub struct MissionData {
    pub buckets: Vec<MissionBucket>,
}

impl MissionData {
    pub fn parse(xml: &[u8]) -> MissionData {
        match parse_document(xml) {
            Some(root) => MissionData::from_element(&root),
            None => MissionData::default(),
        }
    }

    fn from_element(root: &Element) -> MissionData {
        let buckets = root
            .find_all("MissionFrameworkBucket")
            .into_iter()
            .map(|b| MissionBucket {
                label: b.attr("Label").unwrap_or("").to_string(),
                fragment_file: b.attr("FragmentFile").unwrap_or("").to_string(),
                missions: b
                    .children_named("MissionFramework")
                    .map(Mission::from_element)
                    .collect(),
            })
            .collect();
        MissionData { buckets }
    }

    pub fn missions(&self) -> impl Iterator<Item = &Mission> {
        self.buckets.iter().flat_map(|b| b.missions.iter())
    }

    pub fn mission(&self, label: &str) -> Option<&Mission> {
        self.missions().find(|m| m.label == label)
    }

    pub fn mission_count(&self) -> usize {
        self.missions().count()
    }

    /// Missions of the Tanner (`ch00`) tutorial arc, in authoring order.
    pub fn tanner_arc(&self) -> Vec<&Mission> {
        self.missions().filter(|m| m.label.starts_with("ta")).collect()
    }
}

// ---------------------------------------------------------------------------
// Enclave story progression (`rts/enclaves.xml`)
// ---------------------------------------------------------------------------

/// One condition on a story action's `<Inputs>`.
#[derive(Clone, Debug, PartialEq)]
pub struct StoryCondition {
    pub kind: String,
    pub id: String,
    pub invert: bool,
}

/// A `Mission_Create` output together with the state/action that emits it.
#[derive(Clone, Debug, PartialEq)]
pub struct MissionCreate {
    pub mission: String,
    pub state: String,
    pub action: String,
    pub conditions: Vec<StoryCondition>,
    pub success_stat: String,
    pub abort_stat: String,
}

/// A `<State>` of the Home enclave: which missions it starts and where it goes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StoryState {
    pub id: String,
    pub alertness: String,
    pub missions: Vec<String>,
    pub transitions: Vec<String>,
}

/// The recovered `Home` enclave state machine. Parsing the whole file gives
/// every enclave; the tutorial chain is the `Home` one.
#[derive(Clone, Debug, Default)]
pub struct StoryProgression {
    pub states: Vec<StoryState>,
    pub mission_creates: Vec<MissionCreate>,
    /// `EnvChange`/`ChangeState` edges in authoring order.
    pub transitions: Vec<(String, String)>,
}

impl StoryProgression {
    pub fn parse(xml: &[u8]) -> StoryProgression {
        match parse_document(xml) {
            Some(root) => StoryProgression::from_element(&root),
            None => StoryProgression::default(),
        }
    }

    fn from_element(root: &Element) -> StoryProgression {
        let home = root
            .find_all("Enclave")
            .into_iter()
            .find(|e| e.attr("EnclaveId") == Some("Home"));
        let mut out = StoryProgression::default();
        let Some(home) = home else { return out };
        for state in home.children_named("State") {
            let state_id = state.attr("Id").unwrap_or("").to_string();
            let mut st = StoryState {
                id: state_id.clone(),
                alertness: state.attr("Alertness").unwrap_or("").to_string(),
                ..Default::default()
            };
            for action in state.children_named("Action") {
                let action_id = action.attr("ActionId").unwrap_or("").to_string();
                let conditions: Vec<StoryCondition> = action
                    .child("Inputs")
                    .map(|i| {
                        i.children
                            .iter()
                            .filter(|c| {
                                matches!(c.name.as_str(), "CheckBool" | "Enclave_CheckBool" | "Check")
                            })
                            .map(|c| StoryCondition {
                                kind: c.name.clone(),
                                id: c.attr("InputId").unwrap_or("").to_string(),
                                invert: c.attr_bool("Invert"),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if let Some(outputs) = action.child("Outputs") {
                    for o in &outputs.children {
                        match o.name.as_str() {
                            "Mission_Create" => {
                                let mission = o.attr("Mission").unwrap_or("").to_string();
                                if !mission.is_empty() {
                                    st.missions.push(mission.clone());
                                    out.mission_creates.push(MissionCreate {
                                        mission,
                                        state: state_id.clone(),
                                        action: action_id.clone(),
                                        conditions: conditions.clone(),
                                        success_stat: o.attr("RtsDataSuccess").unwrap_or("").to_string(),
                                        abort_stat: o.attr("RtsDataAbort").unwrap_or("").to_string(),
                                    });
                                }
                            }
                            "ChangeState" => {
                                if let Some(to) = o.attr("State") {
                                    if !to.trim().is_empty() {
                                        st.transitions.push(to.to_string());
                                        out.transitions.push((state_id.clone(), to.to_string()));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            out.states.push(st);
        }
        out
    }

    /// The ch00 tutorial `Mission_Create` chain in state order.
    pub fn tutorial_chain(&self) -> Vec<&MissionCreate> {
        self.mission_creates.iter().filter(|m| m.mission.starts_with("ta")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MISSION_XML: &str = r#"<RtsMissionData>
      <MissionFrameworkBucket Label="Tanner Arc (ch00)" FragmentFile="Missions_Tanner">
        <MissionFramework Label="ta10Buddies" Type="Story" ActivateType="OnAdd"
            OpportunityText="Under Attack" SetupScene="ta10Buddies_Setup" Priority="0">
          <MissionActors><Actors>
            <Character ActorTraits="Trait.Story.Buddy" Optional="" />
            <TagPoint ActorName="RTSPoint_RangerStation" NearActorIdx="-1" />
            <Building NearActorIdx="1" ActorFlags="AllowMissionOp" />
            <Player />
          </Actors></MissionActors>
          <MissionEvents>
            <MissionEvent Id="Created">
              <Action ActionId="Zeds" ActionType="Action.Background">
                <Inputs />
                <Outputs><Zombie.Spawn SpawnCount="1" SpawnPosActorIndex="2" /></Outputs>
              </Action>
            </MissionEvent>
            <MissionEvent Id="Activated" />
          </MissionEvents>
          <ObjectiveBuckets>
            <ObjectiveBucket Label="SaveEd"><Objectives>
              <Objective Type="ClearArea" ObjectiveText="Survive!" HeadlineText="UNDER ATTACK"
                  Flags="ActivateBeforeScene" TargetLocationOverride="2" IgnoreNonAggroZombies="True">
                <Location ActorLocation="2" Radius="45" ZombieCount="3" />
                <LocationEvents>
                  <LocationEvent ActorIndex="1" Radius="46">
                    <Action ActionId="EndObjective"><Outputs><Mission_CompleteObjective /></Outputs></Action>
                  </LocationEvent>
                </LocationEvents>
                <ObjectiveEvents>
                  <ObjectiveEvent Id="Activate">
                    <Action ActionFlags="DelayOutputs" OutputDelayTime="5s">
                      <Outputs><Hint_Enable HintId="Attack" SetEnabled="true" /></Outputs>
                    </Action>
                  </ObjectiveEvent>
                </ObjectiveEvents>
                <TimedEvents><TimedEvent Timer="180"><Action><Outputs /></Action></TimedEvent></TimedEvents>
              </Objective>
            </Objectives></ObjectiveBucket>
          </ObjectiveBuckets>
        </MissionFramework>
      </MissionFrameworkBucket>
    </RtsMissionData>"#;

    const ENCLAVES_XML: &str = r#"<Enclaves>
      <Enclave EnclaveId="Home" Flags="Home">
        <State Alertness="Defending" Id="Arrive">
          <Action ActionId="ToBuddies"><Inputs /><Outputs><ChangeState State="Buddies" /></Outputs></Action>
        </State>
        <State Alertness="Defending" Id="Buddies">
          <Action ActionId="CreateMission (Init)"><Inputs /><Outputs>
            <Mission_Create Mission="ta10Buddies" RtsDataSuccess="MissionPrimary_Done" />
          </Outputs></Action>
          <Action ActionId="ta15Go"><Inputs>
            <Enclave_CheckBool InputId="MissionPrimary_Done" />
          </Inputs><Outputs>
            <Mission_Create Mission="ta15Go" RtsDataSuccess="MissionSecondary_Done" />
          </Outputs></Action>
          <Action ActionId="ToNextState"><Inputs /><Outputs><ChangeState State="Mountain" /></Outputs></Action>
        </State>
        <State Id="Mountain">
          <Action ActionId="CreateMissions"><Inputs /><Outputs>
            <Mission_Create Mission="ta20Forest" />
            <Mission_Create Mission="ta20Tower" />
          </Outputs></Action>
        </State>
      </Enclave>
    </Enclaves>"#;

    #[test]
    fn parses_mission_schema() {
        let data = MissionData::parse(MISSION_XML.as_bytes());
        assert_eq!(data.mission_count(), 1);
        let m = data.mission("ta10Buddies").expect("mission");
        assert_eq!(m.type_name, "Story");
        assert_eq!(m.activate_type, "OnAdd");
        assert_eq!(m.actors.len(), 4);
        assert_eq!(m.actors[0].kind, ActorKind::Character);
        assert_eq!(m.actors[0].actor_traits, vec!["Trait.Story.Buddy"]);
        assert_eq!(m.actors[1].kind, ActorKind::TagPoint);
        assert_eq!(m.actors[1].actor_name, "RTSPoint_RangerStation");
        assert_eq!(m.actors[2].near_actor_idx, 1);
        assert_eq!(m.actors[3].kind, ActorKind::Player);
        assert_eq!(m.event("Created").len(), 1);
        assert_eq!(m.event("Created")[0].outputs[0].name, "Zombie.Spawn");
        assert_eq!(m.objective_buckets.len(), 1);
        let o = &m.objective_buckets[0].objectives[0];
        assert_eq!(o.kind, ObjectiveType::ClearArea);
        assert_eq!(o.text, "Survive!");
        assert_eq!(o.locations[0].radius, 45.0);
        assert_eq!(o.locations[0].zombie_count, 3);
        assert_eq!(o.location_events[0].actor_index, 1);
        assert_eq!(o.location_events[0].actions[0].outputs[0].name, "Mission_CompleteObjective");
        assert_eq!(o.event("Activate")[0].output_delay, 5.0);
        assert_eq!(o.timed_events[0].timer, 180.0);
    }

    #[test]
    fn parses_story_progression() {
        let sp = StoryProgression::parse(ENCLAVES_XML.as_bytes());
        assert_eq!(sp.states.len(), 3);
        assert_eq!(sp.mission_creates.len(), 4);
        let chain = sp.tutorial_chain();
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[0].mission, "ta10Buddies");
        assert_eq!(chain[0].state, "Buddies");
        assert_eq!(chain[1].mission, "ta15Go");
        assert_eq!(chain[1].conditions[0].id, "MissionPrimary_Done");
        assert!(sp.transitions.contains(&("Buddies".into(), "Mountain".into())));
    }

    #[test]
    fn parse_never_panics_on_junk() {
        assert_eq!(MissionData::parse(b"").mission_count(), 0);
        assert_eq!(MissionData::parse(b"<not xml").mission_count(), 0);
        assert_eq!(MissionData::parse(b"<RtsMissionData/>").mission_count(), 0);
        assert_eq!(StoryProgression::parse(b"").states.len(), 0);
    }

    #[test]
    fn parse_seconds_handles_units() {
        assert_eq!(parse_seconds("5s"), 5.0);
        assert_eq!(parse_seconds("1.5"), 1.5);
        assert_eq!(parse_seconds(""), 0.0);
        assert_eq!(parse_seconds("0.01s"), 0.01);
    }

    fn real_bytes(rel: &str) -> Option<Vec<u8>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/gamedata/libs/class3")
            .join(rel);
        std::fs::read(path).ok()
    }

    #[test]
    fn real_missions_and_enclaves_parse_when_present() {
        let Some(bytes) = real_bytes("contentmanager/missions.xml") else { return };
        let data = MissionData::parse(&bytes);
        assert_eq!(data.buckets.len(), 25, "mission buckets in missions.xml");
        assert_eq!(data.mission_count(), 119, "mission frameworks in missions.xml");
        let m = data.mission("ta10Buddies").expect("ta10Buddies");
        assert_eq!(m.actors.len(), 12);
        assert_eq!(m.objective_buckets.len(), 1);
        let o = &m.objective_buckets[0].objectives[0];
        assert_eq!(o.kind, ObjectiveType::ClearArea);
        assert_eq!(o.text, "Survive!");
        assert_eq!(o.locations[0].radius, 45.0);
        assert_eq!(o.locations[0].zombie_count, 3);
        assert_eq!(m.event("Created").len(), 3);
        // The whole Tanner arc is present.
        assert!(data.tanner_arc().iter().any(|m| m.label == "ta40Bridge"));

        let Some(eb) = real_bytes("rts/enclaves.xml") else { return };
        let sp = StoryProgression::parse(&eb);
        let chain = sp.tutorial_chain();
        let labels: Vec<&str> = chain.iter().map(|m| m.mission.as_str()).collect();
        for want in ["ta10Buddies", "ta15Go", "ta20Forest", "ta20Tower", "ta20Gunshots", "ta40Bridge"] {
            assert!(labels.contains(&want), "story chain missing {want}: {labels:?}");
        }
    }
}
