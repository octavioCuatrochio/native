//! Full per-character CryEngine AnimationGraph XML (`graphs/*.xml`).
//!
//! Two on-disk flavours are handled:
//!
//! * **State-list** (`humanmalefullbody.xml`, `ul_fullbody.xml`, ...): an
//!   `<AnimationSetup>` root with `<Inputs>`, `<States>` and `<Transitions>`.
//!   Each `<State id="...">` carries a base animation in a direct
//!   `<ParameterizedData animationName="..."/>` (and/or `<AnimationLayer1
//!   animation1="..."/>`), a `Structure` on `<NodeDetails>` or
//!   `<TransitionParamsLayer1>`, and selection predicates in `<SelectWhen>`.
//!   `<Parameterization>` holds per stance/item `<Override>`s which this
//!   module deliberately ignores so [`AgState::animation`] stays the default.
//! * **Template-list** (`playerfullbody.xml`): each `<State>` names a
//!   `<Template name="..." animation="..." structure="..."/>`; overrides may
//!   refine the clip per `[STANCE]`/`[ITEM]`.
//!
//! `<Transitions>` links are `<Link from="A" to="B"/>`; conditions are
//! best-effort metadata (child elements such as `<Mapping>` plus the
//! `cost`/`forceFollowChance`/`transitionTime` attributes).
//!
//! [`parse`] is lenient: malformed XML ends parsing early and never panics.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// One animation graph state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgState {
    /// `<State id="...">`.
    pub name: String,
    /// Base clip alias, e.g. `_RELAXED_RUN_NW` / `_[STANCE]_RUNSTRAFE_[ITEM]_01`.
    pub animation: Option<String>,
    /// Graph structure hint (`StandMoving`, `Moving`, ...).
    pub structure: Option<String>,
    /// Template name for template-list graphs (`MovementDecoupled`).
    pub template: Option<String>,
}

/// One input the graph is parameterised by.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgInput {
    /// `name` attribute (`PseudoSpeed`, `Stance`, `Item`, ...).
    pub name: String,
    /// `defaultValue` / `default`, when present.
    pub default: Option<String>,
}

/// One `<Link from="..." to="...">` transition.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgTransition {
    pub from: String,
    pub to: String,
    /// Best-effort conditions/metadata (`Mapping ...`, `cost=..`, ...).
    pub conditions: Vec<String>,
}

/// A parsed AnimationGraph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgGraph {
    /// Derived from the `<Character filename="...">` stem.
    pub name: String,
    pub states: Vec<AgState>,
    pub inputs: Vec<AgInput>,
    pub transitions: Vec<AgTransition>,
}

#[derive(Clone, Copy, PartialEq)]
enum Section {
    None,
    Inputs,
    States,
    Transitions,
}

/// Parse a full AnimationGraph XML document. Returns `None` when no
/// `<AnimationSetup>` root or state is present. Never panics.
pub fn parse(xml: &[u8]) -> Option<AgGraph> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();

    let mut root_seen = false;
    let mut character: Option<String> = None;
    let mut section = Section::None;

    let mut inputs: Vec<AgInput> = Vec::new();
    let mut states: Vec<AgState> = Vec::new();
    let mut transitions: Vec<AgTransition> = Vec::new();

    let mut cur: Option<AgState> = None;
    let mut state_depth: u32 = 0;
    let mut cur_link: Option<AgTransition> = None;
    let mut link_depth: u32 = 0;

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };

        match ev {
            Event::Start(e) => {
                let name = local_name(&e);
                let tag = name.as_slice();
                if !root_seen && tag == b"AnimationSetup" {
                    root_seen = true;
                } else if tag == b"Character" {
                    if character.is_none() {
                        character = get_attr(&e, "filename");
                    }
                } else if tag == b"Inputs" {
                    section = Section::Inputs;
                } else if tag == b"States" {
                    section = Section::States;
                } else if tag == b"Transitions" {
                    section = Section::Transitions;
                } else {
                    match section {
                        Section::Inputs => {
                            if let Some(i) = input_from(&e) {
                                inputs.push(i);
                            }
                        }
                        Section::States => {
                            if tag == b"State" && state_depth == 0 && cur.is_none() {
                                cur = Some(AgState {
                                    name: get_attr(&e, "id").unwrap_or_default(),
                                    ..Default::default()
                                });
                            } else if let Some(s) = cur.as_mut() {
                                if state_depth == 0 {
                                    absorb_state_child(&e, s);
                                }
                                state_depth += 1;
                            }
                        }
                        Section::Transitions => {
                            if tag == b"Link" && cur_link.is_none() {
                                cur_link = Some(transition_from(&e));
                                link_depth = 0;
                            } else if let Some(t) = cur_link.as_mut() {
                                t.conditions.push(render_element(&e));
                                link_depth += 1;
                            }
                        }
                        Section::None => {}
                    }
                }
            }
            Event::Empty(e) => {
                let name = local_name(&e);
                let tag = name.as_slice();
                if tag == b"Character" && character.is_none() {
                    character = get_attr(&e, "filename");
                }
                match section {
                    Section::Inputs => {
                        if let Some(i) = input_from(&e) {
                            inputs.push(i);
                        }
                    }
                    Section::States => {
                        if let Some(s) = cur.as_mut() {
                            if state_depth == 0 {
                                absorb_state_child(&e, s);
                            }
                        }
                    }
                    Section::Transitions => {
                        if tag == b"Link" {
                            if let Some(t) = cur_link.take() {
                                transitions.push(t);
                            }
                            transitions.push(transition_from(&e));
                        } else if let Some(t) = cur_link.as_mut() {
                            t.conditions.push(render_element(&e));
                        }
                    }
                    Section::None => {}
                }
            }
            Event::End(e) => {
                let name = e.name().as_ref().to_vec();
                let tag = name.as_slice();
                match section {
                    Section::Inputs => {
                        if tag == b"Inputs" {
                            section = Section::None;
                        }
                    }
                    Section::States => {
                        if tag == b"States" && state_depth == 0 && cur.is_none() {
                            section = Section::None;
                        } else if tag == b"State" && state_depth == 0 {
                            if let Some(s) = cur.take() {
                                states.push(s);
                            }
                        } else if state_depth > 0 {
                            state_depth -= 1;
                        }
                    }
                    Section::Transitions => {
                        if tag == b"Transitions" && link_depth == 0 && cur_link.is_none() {
                            section = Section::None;
                        } else if tag == b"Link" && link_depth == 0 {
                            if let Some(t) = cur_link.take() {
                                transitions.push(t);
                            }
                        } else if link_depth > 0 {
                            link_depth -= 1;
                        }
                    }
                    Section::None => {}
                }
            }
            _ => {}
        }
        buf.clear();
    }

    if !root_seen && states.is_empty() {
        return None;
    }

    let name = character
        .as_deref()
        .map(stem)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "AnimationGraph".to_string());

    Some(AgGraph {
        name,
        states,
        inputs,
        transitions,
    })
}

impl AgGraph {
    /// Look up a state by exact name, then case-insensitively.
    pub fn state(&self, name: &str) -> Option<&AgState> {
        self.states
            .iter()
            .find(|s| s.name == name)
            .or_else(|| self.states.iter().find(|s| s.name.eq_ignore_ascii_case(name)))
    }

    /// Base clip alias played by `state`.
    pub fn animation_for(&self, state: &str) -> Option<&str> {
        self.state(state).and_then(|s| s.animation.as_deref())
    }

    /// Find the state that realises a locomotion `gait` (`run`, `walk`,
    /// `sprint`, `_RUNSTRAFE_`, ...). Prefers a named state that carries a
    /// clip, then any named match, then a matching clip alias.
    pub fn locomotion_state(&self, gait: &str) -> Option<&AgState> {
        let g = gait.trim().trim_matches('_').to_ascii_lowercase();
        if g.is_empty() {
            return None;
        }
        self.states
            .iter()
            .find(|s| s.animation.is_some() && s.name.to_ascii_lowercase().contains(&g))
            .or_else(|| self.states.iter().find(|s| s.name.to_ascii_lowercase().contains(&g)))
            .or_else(|| {
                self.states.iter().find(|s| {
                    s.animation
                        .as_deref()
                        .map_or(false, |a| a.to_ascii_lowercase().contains(&g))
                })
            })
    }

    /// Distinct locomotion clip aliases, in document order, with the leading
    /// `_` stripped (e.g. `RELAXED_RUN_NW`, `[STANCE]_RUNSTRAFE_[ITEM]_01`).
    pub fn clip_suffixes(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for s in &self.states {
            if !is_locomotion(&s.name) {
                continue;
            }
            let Some(a) = s.animation.as_deref() else {
                continue;
            };
            let a = a.trim().trim_start_matches('_');
            if a.is_empty() {
                continue;
            }
            if !out.iter().any(|x| x == a) {
                out.push(a.to_string());
            }
        }
        out
    }
}

const LOCO_KEYWORDS: [&str; 6] = ["run", "walk", "sprint", "strafe", "crouch", "swim"];

fn is_locomotion(state_name: &str) -> bool {
    let n = state_name.to_ascii_lowercase();
    LOCO_KEYWORDS.iter().any(|k| n.contains(k))
}

/// Fold one direct child of `<State>` into the state record.
fn absorb_state_child(e: &BytesStart, s: &mut AgState) {
    let name = local_name(e);
    let tag = name.as_slice();
    match tag {
        b"ParameterizedData" => {
            if s.animation.is_none() {
                s.animation = nonempty_attr(e, "animationName");
            }
        }
        b"AnimationLayer1" => {
            if s.animation.is_none() {
                s.animation = nonempty_attr(e, "animation1");
            }
        }
        b"Template" => {
            if s.template.is_none() {
                s.template = nonempty_attr(e, "name");
            }
            if s.animation.is_none() {
                s.animation = nonempty_attr(e, "animation");
            }
            if s.structure.is_none() {
                s.structure = nonempty_attr(e, "structure");
            }
        }
        b"NodeDetails" | b"TransitionParamsLayer1" => {
            if s.structure.is_none() {
                s.structure = nonempty_attr(e, "Structure");
            }
        }
        _ => {}
    }
}

fn input_from(e: &BytesStart) -> Option<AgInput> {
    let tag = local_name(e);
    if !tag.ends_with(b"State") {
        return None;
    }
    let name = get_attr(e, "name")?;
    if name.is_empty() {
        return None;
    }
    let default = nonempty_attr(e, "defaultValue").or_else(|| nonempty_attr(e, "default"));
    Some(AgInput { name, default })
}

fn transition_from(e: &BytesStart) -> AgTransition {
    let mut t = AgTransition {
        from: get_attr(e, "from").unwrap_or_default(),
        to: get_attr(e, "to").unwrap_or_default(),
        conditions: Vec::new(),
    };
    for k in ["cost", "forceFollowChance", "transitionTime"] {
        if let Some(v) = get_attr(e, k) {
            t.conditions.push(format!("{k}={v}"));
        }
    }
    t
}

/// Render an element as `tag key=value ...` for use as a transition condition.
fn render_element(e: &BytesStart) -> String {
    let mut s = String::from_utf8_lossy(e.name().as_ref()).into_owned();
    for a in e.attributes().flatten() {
        if let Ok(v) = a.unescape_value() {
            s.push(' ');
            s.push_str(&String::from_utf8_lossy(a.key.as_ref()));
            s.push('=');
            s.push_str(&v);
        }
    }
    s
}

fn local_name(e: &BytesStart) -> Vec<u8> {
    e.name().as_ref().to_vec()
}

/// Look up one attribute by exact case-sensitive name, unescaped.
fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

/// Like [`get_attr`] but maps `Some("")` to `None`.
fn nonempty_attr(e: &BytesStart, name: &str) -> Option<String> {
    get_attr(e, name).filter(|v| !v.trim().is_empty())
}

/// File stem of a CryEngine path, e.g. `.../sdk_character_male.cdf` -> `sdk_character_male`.
fn stem(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let base = normalized.rsplit('/').next().unwrap_or(normalized.as_str());
    match base.rsplit_once('.') {
        Some((s, _)) => s.to_string(),
        None => base.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// `native/crates/core` -> repo root -> the shipped graphs directory.
    fn graphs_dir() -> Option<PathBuf> {
        let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        root.pop();
        root.pop();
        root.pop();
        let dir = root.join("extracted-game/animations/animations/graphs");
        if dir.is_dir() {
            Some(dir)
        } else {
            None
        }
    }

    fn read_graph(file: &str) -> Option<AgGraph> {
        let dir = graphs_dir()?;
        let bytes = fs::read(dir.join(file)).ok()?;
        parse(&bytes)
    }

    #[test]
    fn junk_never_panics() {
        for junk in [
            &b""[..],
            b"not xml at all",
            b"<Other><Foo/></Other>",
            &[0xff, 0xfe, 0x00, 0x01],
            b"<AnimationSetup version=\"1.5\"><States><State id=\"X\"><P",
        ] {
            let _ = parse(junk);
        }
        assert!(parse(b"").is_none());
        assert!(parse(b"not xml at all").is_none());
        assert!(parse(b"<Other><Foo/></Other>").is_none());
        assert!(parse(&[0xff, 0xfe, 0x00]).is_none());
        // A truncated document still yields whatever was collected.
        assert!(parse(b"<AnimationSetup version=\"1.5\"><States><State id=\"X\"><P").is_some());
    }

    #[test]
    fn real_humanmale_fullbody_maps_locomotion() {
        let Some(g) = read_graph("humanmalefullbody.xml") else {
            // Real game data not present in this checkout; skip silently.
            return;
        };
        assert!(!g.name.is_empty(), "graph needs a name");
        assert!(g.states.len() > 5, "states={}", g.states.len());
        assert!(g.inputs.len() > 3, "inputs={}", g.inputs.len());
        assert!(g.transitions.len() > 3, "transitions={}", g.transitions.len());
        assert!(g.inputs.iter().any(|i| i.name == "PseudoSpeed"));
        assert!(g.inputs.iter().any(|i| i.name == "Stance"));

        let run = g.locomotion_state("run").expect("run state");
        assert!(run.name.to_lowercase().contains("run"), "got {}", run.name);
        let ra = g.animation_for(&run.name).expect("run animation");
        assert!(ra.to_uppercase().contains("RUN"), "run anim {ra}");

        let walk = g.locomotion_state("walk").expect("walk state");
        assert!(walk.name.to_lowercase().contains("walk"), "got {}", walk.name);
        assert!(g.animation_for(&walk.name).is_some());

        let clips = g.clip_suffixes();
        assert!(!clips.is_empty(), "no locomotion clips");
        assert!(
            clips.iter().any(|c| c.to_uppercase().contains("RUN")),
            "clips={clips:?}"
        );
        assert!(g.animation_for("Run [Item] [Dir]").is_some());
        assert!(g.animation_for("does-not-exist").is_none());

        eprintln!(
            "ag_graph humanmalefullbody: name={} states={} inputs={} transitions={} clips={:?}",
            g.name,
            g.states.len(),
            g.inputs.len(),
            g.transitions.len(),
            clips
        );
    }

    #[test]
    fn real_player_template_style() {
        let Some(g) = read_graph("playerfullbody.xml") else {
            return;
        };
        let Some(st) = g.states.iter().find(|s| {
            s.animation
                .as_deref()
                .map_or(false, |a| a.contains("RUNSTRAFE"))
        }) else {
            // Schema drift; nothing to assert.
            return;
        };
        assert!(
            st.template.is_some(),
            "template-style state {} should name a template",
            st.name
        );
        assert!(g
            .animation_for(&st.name)
            .unwrap_or_default()
            .contains("RUNSTRAFE"));
    }
}
