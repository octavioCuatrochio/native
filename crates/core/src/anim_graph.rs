//! CryEngine AnimationGraph **state templates** for State of Decay: YOSE.
//!
//! The real game ships its animation state machine as one XML file per template
//! under `gamedata/libs/animationgraphtemplates/` (`humanalive.xml`,
//! `huntermovement.xml`, `sprintdecoupled.xml`, `jumpmain.xml`,
//! `idleoneshot.xml`, `animationcontrolled.xml`, ...). Each file is a single
//! `<AGTemplate name="...">`. A template participates in state selection
//! through a `<SelectWhen>` block whose children are predicates over the
//! character's current inputs (`Health`, `Vehicle`, `Signal`, `PseudoSpeed`,
//! `Stance`, ...). The template also carries presentation hints such as
//! `<ColliderMode mode="..."/>` and
//! `<MovementControlMethod horizontal="..." vertical="..."/>`.
//!
//! This module is a deliberately small, self-contained runtime:
//!
//! * [`AgCondition`] mirrors the observed predicate vocabulary (with a
//!   `Custom` escape hatch so unknown children never fail a parse).
//! * [`AgTemplate`] is one parsed `<AGTemplate>`.
//! * [`parse`] leniently reads a single template; malformed XML ends parsing
//!   without panicking, and `None` is returned when no `AGTemplate` root exists.
//! * [`AnimGraph`] holds every template parsed by [`AnimGraph::parse_many`] and
//!   picks the first matching candidate in [`AnimGraph::select`].
//! * [`GraphParams`] is the input snapshot handed to selection.
//!
//! Templates whose `<SelectWhen>` is absent or empty are *not* selection
//! candidates: they are transitional/oneshot states invoked explicitly by the
//! state machine (often through `extend=`), so [`AnimGraph::select`] skips them.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------

/// One `<SelectWhen>` child: a predicate the character state must satisfy.
#[derive(Debug, Clone, PartialEq)]
pub enum AgCondition {
    /// `<Health min="1" max="100"/>` (or `<Health value="0"/>`).
    Health { min: f32, max: f32 },
    /// `<Vehicle value="none"/>`; `"$param"` placeholders are treated as
    /// wildcards by [`AgCondition::matches`].
    Vehicle(String),
    /// `<Signal value="none"/>`; parameterised values are wildcards.
    Signal(String),
    /// `<Speed>` / `<PseudoSpeed>` / `<ActualMoveSpeedLH>` range. Unparseable
    /// bounds (e.g. `"$speed_slow"`) widen to infinity so they never block.
    Speed { min: f32, max: f32 },
    /// `<Stance min="swim" max="swim"/>`; `"$stance"` is a wildcard.
    Stance(String),
    /// Any other child (`Action`, `DesiredSpeed`, `Firing`, ...), kept verbatim
    /// and treated as a non-blocking hint during selection.
    Custom { tag: String, value: String },
}

impl AgCondition {
    /// Evaluate this predicate against an input snapshot. Unrecognised /
    /// parameterised predicates are best-effort: they match rather than
    /// excluding an otherwise valid template.
    pub fn matches(&self, p: &GraphParams) -> bool {
        match self {
            AgCondition::Health { min, max } => p.health >= *min && p.health <= *max,
            AgCondition::Vehicle(v) => opt_match(v, p.vehicle.as_deref()),
            AgCondition::Signal(s) => opt_match(s, p.signal.as_deref()),
            AgCondition::Speed { min, max } => p.speed >= *min && p.speed <= *max,
            AgCondition::Stance(s) => {
                if is_placeholder(s) {
                    true
                } else {
                    p.stance.as_deref().map_or(false, |a| s.eq_ignore_ascii_case(a))
                }
            }
            AgCondition::Custom { .. } => true,
        }
    }
}

fn is_placeholder(s: &str) -> bool {
    s.trim_start().starts_with('$')
}

/// `"none"` matches absence; a literal matches the same value case-
/// insensitively; a `"$param"` placeholder matches anything.
fn opt_match(cond: &str, actual: Option<&str>) -> bool {
    let c = cond.trim();
    if is_placeholder(c) {
        return true;
    }
    if c.eq_ignore_ascii_case("none") {
        actual.is_none()
    } else {
        actual.map_or(false, |a| a.eq_ignore_ascii_case(c))
    }
}

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

/// One parsed `<AGTemplate>`.
#[derive(Debug, Clone, PartialEq)]
pub struct AgTemplate {
    /// The `name` attribute (`"HumanAlive"`).
    pub name: String,
    /// `<SelectWhen>` predicates, in document order.
    pub select_when: Vec<AgCondition>,
    /// `<ColliderMode mode="..."/>`, when present.
    pub collider_mode: Option<String>,
    /// `<MovementControlMethod horizontal="..." vertical="..."/>`, when present.
    pub movement_control: Option<(String, String)>,
}

/// Parse a single `<AGTemplate>` XML document.
///
/// Returns `None` if no `AGTemplate` element is seen. Malformed XML and
/// malformed attributes are tolerated: parsing stops at the error and whatever
/// was collected so far is returned.
pub fn parse(xml: &[u8]) -> Option<AgTemplate> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();

    let mut root: Option<AgTemplate> = None;
    let mut in_select = false;
    // Nesting depth under `<SelectWhen>`; only direct children are predicates.
    let mut select_depth: u32 = 0;

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            // Treat malformed XML as end-of-input rather than panicking.
            Err(_) => break,
            Ok(ev) => ev,
        };

        match ev {
            Event::Start(e) => {
                let name = e.name().as_ref().to_vec();
                if name.as_slice() == b"AGTemplate" {
                    if root.is_none() {
                        root = Some(empty_template(&e));
                    }
                } else if name.as_slice() == b"SelectWhen" {
                    in_select = true;
                    select_depth = 0;
                } else if in_select {
                    if select_depth == 0 {
                        if let Some(t) = root.as_mut() {
                            push_condition(&e, &mut t.select_when);
                        }
                    }
                    select_depth += 1;
                } else if let Some(t) = root.as_mut() {
                    handle_hint(&name, &e, t);
                }
            }

            Event::Empty(e) => {
                let name = e.name().as_ref().to_vec();
                if name.as_slice() == b"AGTemplate" {
                    if root.is_none() {
                        root = Some(empty_template(&e));
                    }
                } else if name.as_slice() == b"SelectWhen" {
                    // `<SelectWhen/>` with no predicates.
                } else if in_select {
                    if select_depth == 0 {
                        if let Some(t) = root.as_mut() {
                            push_condition(&e, &mut t.select_when);
                        }
                    }
                } else if let Some(t) = root.as_mut() {
                    handle_hint(&name, &e, t);
                }
            }

            Event::End(e) => {
                let name = e.name().as_ref().to_vec();
                if name.as_slice() == b"SelectWhen" {
                    in_select = false;
                    select_depth = 0;
                } else if in_select && select_depth > 0 {
                    select_depth -= 1;
                }
            }

            _ => {}
        }
        buf.clear();
    }

    root
}

fn empty_template(e: &BytesStart) -> AgTemplate {
    AgTemplate {
        name: get_attr(e, "name").unwrap_or_default(),
        select_when: Vec::new(),
        collider_mode: None,
        movement_control: None,
    }
}

/// Fold a non-`SelectWhen` direct child into the template: the collider mode
/// and movement-control method are the only hints we surface.
fn handle_hint(name: &[u8], e: &BytesStart, t: &mut AgTemplate) {
    if name == b"ColliderMode" {
        if let Some(m) = get_attr(e, "mode") {
            t.collider_mode = Some(m);
        }
    } else if name == b"MovementControlMethod" {
        let h = get_attr(e, "horizontal").unwrap_or_default();
        let v = get_attr(e, "vertical").unwrap_or_default();
        t.movement_control = Some((h, v));
    }
}

/// Classify one `<SelectWhen>` child into an [`AgCondition`].
fn push_condition(e: &BytesStart, out: &mut Vec<AgCondition>) {
    let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
    let value = get_attr(e, "value");
    let min = get_attr(e, "min");
    let max = get_attr(e, "max");

    let cond = match tag.as_str() {
        "Health" => {
            if min.is_none() && max.is_none() {
                match value.as_deref().and_then(parse_f32) {
                    Some(v) => AgCondition::Health { min: v, max: v },
                    None => AgCondition::Custom { tag, value: value.unwrap_or_default() },
                }
            } else {
                AgCondition::Health {
                    min: min.as_deref().and_then(parse_f32).unwrap_or(f32::NEG_INFINITY),
                    max: max.as_deref().and_then(parse_f32).unwrap_or(f32::INFINITY),
                }
            }
        }
        "Vehicle" => AgCondition::Vehicle(value.unwrap_or_else(|| "none".into())),
        "Signal" => AgCondition::Signal(value.unwrap_or_else(|| "none".into())),
        "Speed" | "PseudoSpeed" | "ActualMoveSpeedLH" | "RequestedMoveSpeedLH" => {
            AgCondition::Speed {
                min: min.as_deref().and_then(parse_f32).unwrap_or(f32::NEG_INFINITY),
                max: max.as_deref().and_then(parse_f32).unwrap_or(f32::INFINITY),
            }
        }
        "Stance" => AgCondition::Stance(
            value
                .clone()
                .or_else(|| min.clone())
                .or_else(|| max.clone())
                .unwrap_or_default(),
        ),
        _ => AgCondition::Custom {
            tag,
            value: value.or(min).or(max).unwrap_or_default(),
        },
    };
    out.push(cond);
}

fn parse_f32(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok()
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

// ---------------------------------------------------------------------------
// Graph + selection
// ---------------------------------------------------------------------------

/// The input snapshot selection runs against.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GraphParams {
    pub health: f32,
    pub vehicle: Option<String>,
    pub signal: Option<String>,
    pub speed: f32,
    pub stance: Option<String>,
}

/// Every template loaded from the animation graph template directory.
#[derive(Debug, Clone, Default)]
pub struct AnimGraph {
    templates: Vec<AgTemplate>,
}

impl AnimGraph {
    /// Parse every document in `iter`, silently dropping files with no
    /// `AGTemplate` root.
    pub fn parse_many<'a>(iter: impl Iterator<Item = &'a [u8]>) -> AnimGraph {
        let mut templates = Vec::new();
        for data in iter {
            if let Some(t) = parse(data) {
                templates.push(t);
            }
        }
        AnimGraph { templates }
    }

    /// Number of loaded templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether no templates were loaded.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Look a template up by name (exact first, then case-insensitive).
    pub fn get(&self, name: &str) -> Option<&AgTemplate> {
        self.templates
            .iter()
            .find(|t| t.name == name)
            .or_else(|| self.templates.iter().find(|t| t.name.eq_ignore_ascii_case(name)))
    }

    /// Iterate templates in load order.
    pub fn iter(&self) -> impl Iterator<Item = &AgTemplate> {
        self.templates.iter()
    }

    /// Return the first selection candidate whose predicates all match.
    ///
    /// Templates with no `<SelectWhen>` predicates are skipped: they are
    /// transitional states run explicitly, not selected by input matching.
    pub fn select(&self, p: &GraphParams) -> Option<&AgTemplate> {
        self.templates
            .iter()
            .find(|t| !t.select_when.is_empty() && t.select_when.iter().all(|c| c.matches(p)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    const HUMANALIVE: &[u8] = br#"<AGTemplate name="HumanAlive">
  <SelectWhen>
    <Health max="100" min="1" />
    <Vehicle value="none" />
    <Signal value="none"/>
  </SelectWhen>
  <ColliderMode mode="Undefined"/>
  <MovementControlMethod horizontal="Entity" vertical="Entity"/>
</AGTemplate>"#;

    #[test]
    fn parses_human_alive() {
        let t = parse(HUMANALIVE).expect("AGTemplate should parse");
        assert_eq!(t.name, "HumanAlive");
        assert_eq!(t.select_when.len(), 3);
        assert_eq!(t.collider_mode.as_deref(), Some("Undefined"));
        assert_eq!(
            t.movement_control,
            Some(("Entity".to_string(), "Entity".to_string()))
        );
        assert!(
            matches!(t.select_when[0], AgCondition::Health { min, max } if min == 1.0 && max == 100.0)
        );
        assert!(matches!(&t.select_when[1], AgCondition::Vehicle(v) if v == "none"));
        assert!(matches!(&t.select_when[2], AgCondition::Signal(s) if s == "none"));

        let p = GraphParams {
            health: 50.0,
            vehicle: None,
            signal: None,
            speed: 0.0,
            stance: None,
        };
        assert!(t.select_when.iter().all(|c| c.matches(&p)));
    }

    #[test]
    fn lenient_on_junk() {
        // Must never panic, whatever the bytes.
        for junk in [
            &b""[..],
            b"not xml at all",
            b"<Other><Foo/></Other>",
            &[0xff, 0xfe, 0x00, 0x01],
            b"<AGTemplate name=\"X\"><SelectWhen><Health min=",
        ] {
            let _ = parse(junk);
        }
        assert!(parse(b"not xml at all").is_none());
        assert!(parse(b"<Other><Foo/></Other>").is_none());
        assert!(parse(&[0xff, 0xfe, 0x00]).is_none());
        // A truncated template still yields whatever it had.
        let truncated = parse(b"<AGTemplate name=\"X\"><ColliderMode mode=\"Pushable\"/>");
        assert_eq!(truncated.map(|t| t.name), Some("X".to_string()));
    }

    /// Locate the shipped template directory from this crate's manifest dir.
    fn templates_dir() -> Option<PathBuf> {
        let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // native/crates/core -> native/crates -> native -> repo root
        root.pop();
        root.pop();
        root.pop();
        let dir = root.join("extracted-game/gamedata/libs/animationgraphtemplates");
        if dir.is_dir() {
            Some(dir)
        } else {
            None
        }
    }

    #[test]
    fn real_templates_parse_and_select() {
        let Some(dir) = templates_dir() else {
            // Real game data not present in this checkout; skip silently.
            return;
        };

        let Ok(rd) = fs::read_dir(&dir) else { return };
        let mut paths: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("xml"))
            .collect();
        paths.sort();

        let datas: Vec<Vec<u8>> = paths.iter().filter_map(|p| fs::read(p).ok()).collect();
        let graph = AnimGraph::parse_many(datas.iter().map(|d| d.as_slice()));

        assert!(graph.len() > 20, "only {} templates parsed", graph.len());
        assert!(graph.get("HumanAlive").is_some());

        let p = GraphParams {
            health: 50.0,
            vehicle: None,
            signal: None,
            speed: 0.0,
            stance: None,
        };
        let selected = graph.select(&p).expect("a template should be selected");
        assert_eq!(selected.name, "HumanAlive", "selected {}", selected.name);
    }
}
