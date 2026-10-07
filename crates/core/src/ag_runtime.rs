//! CryEngine AnimationGraph runtime (`animations/graphs/*.xml`).
//!
//! The graph XML is read at runtime from the game data. The runtime does what
//! the engine's graph does each frame:
//!
//! 1. **Select**: from the current input values, pick the target state whose
//!    effective `SelectWhen` criteria all match (only `allowSelect`,
//!    `includeInGame` states can be targets).
//! 2. **Path**: walk the `Link` graph from the current state to the target
//!    along the cheapest path; each link's `transitionTime` is the blend time.
//!
//! State inheritance: `extend="Parent"` copies the parent's flags and
//! criteria; the child's own entries override per input, and an empty
//! criterion element (`<PseudoSpeed />`) removes the inherited constraint.
//!
//! Choices not settled by the data (marked `ASSUMPTION`, to be verified
//! against the running game): float ranges are half-open `[min, max)` with the
//! input's declared maximum inclusive; integer ranges are inclusive; ties
//! between matching states go to the more specific state, weighted by input
//! priority (approximating the engine's priority-ordered decision tree).
//!
//! OPEN (blocks wiring this into the player): in `playerfullbody.xml` the
//! locomotion states `RelaxedWalkNW`, `RelaxedRunNW` and `RelaxedMoveNw` have
//! identical effective criteria, which also match at standing speed, so
//! `SelectWhen` alone cannot be how the engine picks walk vs run. The engine's
//! state query (decision tree built at load), template parameterization, or
//! the ULSM `GraphDriven.*` states requesting graph states must be resolved
//! first; until then `select` breaks such ties by document order.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone, PartialEq)]
pub enum InputKind {
    /// `KeyState`: one of a set of string keys.
    Key { keys: Vec<String> },
    /// `FloatState` with its declared range.
    Float { min: f32, max: f32 },
    /// `IntegerState` with its declared range.
    Integer { min: i32, max: i32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    pub name: String,
    pub kind: InputKind,
    pub priority: i32,
    pub default: Option<String>,
    pub signalled: bool,
}

/// One `SelectWhen` entry.
#[derive(Debug, Clone, PartialEq)]
pub enum Criterion {
    /// Empty element: explicitly unconstrained (removes an inherited one).
    Any,
    Key(String),
    Range { min: Option<f32>, max: Option<f32> },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct State {
    pub id: String,
    pub extend: Option<String>,
    /// Raw attributes as authored on this element (before inheritance).
    pub attrs: Vec<(String, String)>,
    /// Own `SelectWhen` criteria (before inheritance), by input name.
    pub criteria: Vec<(String, Criterion)>,
    /// `Template` elements: (template name, its attributes).
    pub templates: Vec<(String, Vec<(String, String)>)>,
    pub outputs: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    pub from: usize,
    pub to: usize,
    pub cost: f32,
    pub transition_time: f32,
    pub force_follow_chance: f32,
}

/// A state with inheritance applied.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Resolved {
    pub attrs: HashMap<String, String>,
    pub criteria: HashMap<String, Criterion>,
    pub templates: Vec<(String, Vec<(String, String)>)>,
}

impl Resolved {
    pub fn flag(&self, name: &str) -> bool {
        self.attrs.get(name).map(|v| v.trim() == "1").unwrap_or(false)
    }
    pub fn number(&self, name: &str) -> Option<f32> {
        self.attrs.get(name).and_then(|v| v.trim().parse().ok())
    }
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub inputs: Vec<Input>,
    pub states: Vec<State>,
    pub links: Vec<Link>,
    pub by_id: HashMap<String, usize>,
    pub resolved: Vec<Resolved>,
    /// Outgoing link indices per state.
    pub out_links: Vec<Vec<usize>>,
}

/// Current input values, by input name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InputValues {
    pub values: HashMap<String, InputValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputValue {
    Key(String),
    Number(f32),
}

impl InputValues {
    pub fn set_key(&mut self, name: &str, v: &str) {
        self.values.insert(name.to_string(), InputValue::Key(v.to_string()));
    }
    pub fn set_number(&mut self, name: &str, v: f32) {
        self.values.insert(name.to_string(), InputValue::Number(v));
    }
}

fn attrs_of(e: &BytesStart) -> Vec<(String, String)> {
    e.attributes()
        .flatten()
        .map(|a| {
            (
                String::from_utf8_lossy(a.key.as_ref()).to_string(),
                a.unescape_value().map(|v| v.to_string()).unwrap_or_default(),
            )
        })
        .collect()
}

fn get<'a>(a: &'a [(String, String)], k: &str) -> Option<&'a str> {
    a.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

fn num(a: &[(String, String)], k: &str) -> Option<f32> {
    get(a, k).and_then(|v| v.trim().parse().ok())
}

fn criterion_from(a: &[(String, String)]) -> Criterion {
    if a.is_empty() {
        return Criterion::Any;
    }
    if let Some(v) = get(a, "value") {
        return Criterion::Key(v.to_string());
    }
    Criterion::Range { min: num(a, "min"), max: num(a, "max") }
}

impl Graph {
    pub fn parse(xml: &[u8]) -> Option<Graph> {
        let mut r = Reader::from_reader(xml);
        let mut buf = Vec::new();
        let mut path: Vec<Vec<u8>> = Vec::new();
        let mut g = Graph::default();
        let mut cur_input: Option<Input> = None;
        let mut cur_state: Option<State> = None;
        let mut raw_links: Vec<(String, String, f32, f32, f32)> = Vec::new();

        loop {
            let (ev, empty) = match r.read_event_into(&mut buf) {
                Ok(Event::Eof) => break,
                Err(_) => return None,
                Ok(Event::Start(e)) => (Some(e.into_owned()), false),
                Ok(Event::Empty(e)) => (Some(e.into_owned()), true),
                Ok(Event::End(_)) => {
                    let closed = path.pop().unwrap_or_default();
                    Self::close(&closed, &path, &mut g, &mut cur_input, &mut cur_state);
                    buf.clear();
                    continue;
                }
                Ok(_) => (None, false),
            };
            let Some(e) = ev else {
                buf.clear();
                continue;
            };
            let name = e.name().as_ref().to_vec();
            let a = attrs_of(&e);
            let parent = path.last().map(|p| p.as_slice());
            let section = path.get(1).map(|p| p.as_slice());
            match (section, name.as_slice()) {
                (Some(b"Inputs"), b"KeyState" | b"FloatState" | b"IntegerState") if path.len() == 2 => {
                    let kind = match name.as_slice() {
                        b"KeyState" => InputKind::Key { keys: Vec::new() },
                        b"FloatState" => InputKind::Float {
                            min: num(&a, "min").unwrap_or(f32::MIN),
                            max: num(&a, "max").unwrap_or(f32::MAX),
                        },
                        _ => InputKind::Integer {
                            min: num(&a, "min").map(|v| v as i32).unwrap_or(i32::MIN),
                            max: num(&a, "max").map(|v| v as i32).unwrap_or(i32::MAX),
                        },
                    };
                    cur_input = Some(Input {
                        name: get(&a, "name").unwrap_or_default().to_string(),
                        kind,
                        priority: num(&a, "priority").unwrap_or(0.0) as i32,
                        default: get(&a, "defaultValue").map(str::to_string),
                        signalled: get(&a, "signalled") == Some("1"),
                    });
                }
                (Some(b"Inputs"), b"Key") => {
                    if let Some(Input { kind: InputKind::Key { keys }, .. }) = cur_input.as_mut() {
                        if let Some(v) = get(&a, "value") {
                            keys.push(v.to_string());
                        }
                    }
                }
                (Some(b"States"), b"State") if path.len() == 2 => {
                    cur_state = Some(State {
                        id: get(&a, "id").unwrap_or_default().to_string(),
                        extend: get(&a, "extend").filter(|s| !s.is_empty()).map(str::to_string),
                        attrs: a.clone(),
                        ..Default::default()
                    });
                }
                (Some(b"States"), _) if parent == Some(b"SelectWhen") => {
                    if let Some(s) = cur_state.as_mut() {
                        s.criteria.push((String::from_utf8_lossy(&name).to_string(), criterion_from(&a)));
                    }
                }
                (Some(b"States"), b"Template") if parent == Some(b"State") => {
                    if let Some(s) = cur_state.as_mut() {
                        s.templates.push((get(&a, "name").unwrap_or_default().to_string(), a.clone()));
                    }
                }
                (Some(b"States"), b"Output") if parent == Some(b"State") => {
                    if let Some(s) = cur_state.as_mut() {
                        s.outputs.push((
                            get(&a, "name").unwrap_or_default().to_string(),
                            get(&a, "value").unwrap_or_default().to_string(),
                        ));
                    }
                }
                (Some(b"Transitions"), b"Link") => {
                    raw_links.push((
                        get(&a, "from").unwrap_or_default().to_string(),
                        get(&a, "to").unwrap_or_default().to_string(),
                        num(&a, "cost").unwrap_or(100.0),
                        num(&a, "transitionTime").unwrap_or(0.0),
                        num(&a, "forceFollowChance").unwrap_or(0.0),
                    ));
                }
                _ => {}
            }
            if empty {
                Self::close(&name, &path, &mut g, &mut cur_input, &mut cur_state);
            } else {
                path.push(name);
            }
            buf.clear();
        }

        for (i, s) in g.states.iter().enumerate() {
            g.by_id.insert(s.id.to_ascii_lowercase(), i);
        }
        g.out_links = vec![Vec::new(); g.states.len()];
        for (from, to, cost, tt, ffc) in raw_links {
            if let (Some(&f), Some(&t)) = (g.by_id.get(&from.to_ascii_lowercase()), g.by_id.get(&to.to_ascii_lowercase())) {
                g.out_links[f].push(g.links.len());
                g.links.push(Link { from: f, to: t, cost, transition_time: tt, force_follow_chance: ffc });
            }
        }
        g.resolved = (0..g.states.len()).map(|i| g.resolve(i, 0)).collect();
        Some(g)
    }

    fn close(name: &[u8], path: &[Vec<u8>], g: &mut Graph, inp: &mut Option<Input>, st: &mut Option<State>) {
        let section = path.get(1).map(|p| p.as_slice());
        match (section, name) {
            (Some(b"Inputs"), b"KeyState" | b"FloatState" | b"IntegerState") if path.len() == 2 => {
                if let Some(i) = inp.take() {
                    g.inputs.push(i);
                }
            }
            (Some(b"States"), b"State") if path.len() == 2 => {
                if let Some(s) = st.take() {
                    g.states.push(s);
                }
            }
            _ => {}
        }
    }

    fn resolve(&self, i: usize, depth: usize) -> Resolved {
        let s = &self.states[i];
        let mut r = match s.extend.as_ref().and_then(|p| self.by_id.get(&p.to_ascii_lowercase())) {
            Some(&p) if p != i && depth < 64 => self.resolve(p, depth + 1),
            _ => Resolved::default(),
        };
        for (k, v) in &s.attrs {
            r.attrs.insert(k.clone(), v.clone());
        }
        for (k, c) in &s.criteria {
            r.criteria.insert(k.clone(), c.clone());
        }
        if !s.templates.is_empty() {
            r.templates = s.templates.clone();
        }
        r
    }

    pub fn state_index(&self, id: &str) -> Option<usize> {
        self.by_id.get(&id.to_ascii_lowercase()).copied()
    }

    pub fn input(&self, name: &str) -> Option<&Input> {
        self.inputs.iter().find(|i| i.name.eq_ignore_ascii_case(name))
    }

    /// Input values at their declared defaults (key inputs only declare one).
    pub fn default_inputs(&self) -> InputValues {
        let mut v = InputValues::default();
        for i in &self.inputs {
            match (&i.kind, &i.default) {
                (InputKind::Key { .. }, Some(d)) => v.set_key(&i.name, d),
                (InputKind::Key { .. }, None) => {}
                _ => v.set_number(&i.name, 0.0),
            }
        }
        v
    }

    /// Does `crit` accept the current value of `input`?
    fn matches(&self, input: &str, crit: &Criterion, values: &InputValues) -> bool {
        let decl = self.input(input);
        let value = values.values.get(input).or_else(|| {
            values.values.iter().find(|(k, _)| k.eq_ignore_ascii_case(input)).map(|(_, v)| v)
        });
        match crit {
            Criterion::Any => true,
            Criterion::Key(want) => match value {
                Some(InputValue::Key(k)) => k.eq_ignore_ascii_case(want),
                Some(InputValue::Number(n)) => want.trim().parse::<f32>().map(|w| w == *n).unwrap_or(false),
                None => decl.and_then(|d| d.default.as_ref()).map(|d| d.eq_ignore_ascii_case(want)).unwrap_or(false),
            },
            Criterion::Range { min, max } => {
                let Some(InputValue::Number(v)) = value else { return false };
                let v = *v;
                match decl.map(|d| &d.kind) {
                    // ASSUMPTION: integer ranges are inclusive.
                    Some(InputKind::Integer { .. }) => {
                        min.map(|m| v >= m).unwrap_or(true) && max.map(|m| v <= m).unwrap_or(true)
                    }
                    // ASSUMPTION: float ranges are [min, max), except the
                    // input's declared maximum, which is inclusive.
                    Some(InputKind::Float { max: decl_max, .. }) => {
                        let lo = min.map(|m| v >= m).unwrap_or(true);
                        let hi = max.map(|m| v < m || (m >= *decl_max && v <= m)).unwrap_or(true);
                        lo && hi
                    }
                    _ => min.map(|m| v >= m).unwrap_or(true) && max.map(|m| v <= m).unwrap_or(true),
                }
            }
        }
    }

    /// Specificity of a state's criteria, weighted by input priority.
    fn specificity(&self, r: &Resolved) -> i64 {
        r.criteria
            .iter()
            .filter(|(_, c)| **c != Criterion::Any)
            .map(|(k, _)| 1000 + self.input(k).map(|i| i.priority as i64).unwrap_or(0))
            .sum()
    }

    /// Every selectable state whose criteria match, best first.
    pub fn candidates(&self, values: &InputValues) -> Vec<usize> {
        let mut c: Vec<(i64, usize)> = (0..self.states.len())
            .filter(|&i| {
                let r = &self.resolved[i];
                r.flag("allowSelect")
                    && r.flag("includeInGame")
                    && r.criteria.iter().all(|(k, c)| self.matches(k, c, values))
            })
            .map(|i| (self.specificity(&self.resolved[i]), i))
            .collect();
        c.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        c.into_iter().map(|(_, i)| i).collect()
    }

    /// The target state for these inputs.
    pub fn select(&self, values: &InputValues) -> Option<usize> {
        self.candidates(values).first().copied()
    }

    /// Cheapest link path `from -> to` (link indices), or `None`.
    pub fn path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from == to {
            return Some(Vec::new());
        }
        let n = self.states.len();
        let mut dist = vec![f32::INFINITY; n];
        let mut via: Vec<Option<usize>> = vec![None; n];
        let mut heap = BinaryHeap::new();
        dist[from] = 0.0;
        heap.push(Reverse((0u64, from)));
        while let Some(Reverse((d, u))) = heap.pop() {
            let d = d as f32 / 1000.0;
            if d > dist[u] {
                continue;
            }
            if u == to {
                break;
            }
            for &li in &self.out_links[u] {
                let l = &self.links[li];
                let nd = d + l.cost.max(0.0);
                if nd < dist[l.to] {
                    dist[l.to] = nd;
                    via[l.to] = Some(li);
                    heap.push(Reverse(((nd * 1000.0) as u64, l.to)));
                }
            }
        }
        if !dist[to].is_finite() {
            return None;
        }
        let mut out = Vec::new();
        let mut cur = to;
        while let Some(li) = via[cur] {
            out.push(li);
            cur = self.links[li].from;
        }
        out.reverse();
        Some(out)
    }
}

/// Per-character graph instance: current state plus the queued path.
#[derive(Debug, Clone, Default)]
pub struct GraphInstance {
    pub current: Option<usize>,
    pub target: Option<usize>,
    /// Remaining links to traverse toward `target`.
    pub queue: Vec<usize>,
    /// Blend time of the most recent transition.
    pub blend: f32,
}

impl GraphInstance {
    /// Re-select the target from `values`; if it changed, plan a path. Then
    /// advance one link when `step` is true (the engine advances when the
    /// current state's animation allows leaving it).
    pub fn update(&mut self, g: &Graph, values: &InputValues, step: bool) {
        let target = g.select(values);
        if target != self.target {
            self.target = target;
            self.queue = match (self.current, target) {
                (Some(c), Some(t)) => g.path(c, t).unwrap_or_default(),
                _ => Vec::new(),
            };
            if self.current.is_none() {
                self.current = target;
            }
        }
        if step && !self.queue.is_empty() {
            let li = self.queue.remove(0);
            self.current = Some(g.links[li].to);
            self.blend = g.links[li].transition_time;
        } else if self.queue.is_empty() && target.is_some() && self.current != target {
            // No path in the link graph: the engine jumps directly.
            self.current = target;
            self.blend = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<AnimationGraph>
 <Character filename="test"/>
 <Inputs>
  <KeyState name="Action" priority="100" defaultValue="idle"><Key value="idle"/><Key value="move"/></KeyState>
  <FloatState name="PseudoSpeed" priority="50" min="0" max="3"/>
  <IntegerState name="Seat" priority="10" min="0" max="5"/>
 </Inputs>
 <States>
  <State id="Base" allowSelect="0" includeInGame="1" cost="100"><SelectWhen><Action value="idle"/></SelectWhen></State>
  <State id="Idle" extend="Base" allowSelect="1"><SelectWhen><PseudoSpeed min="0" max="0.2"/></SelectWhen></State>
  <State id="Walk" extend="Base" allowSelect="1"><SelectWhen><Action value="move"/><PseudoSpeed min="0.2" max="2"/></SelectWhen></State>
  <State id="Run" extend="Base" allowSelect="1"><SelectWhen><Action value="move"/><PseudoSpeed min="2" max="3"/></SelectWhen></State>
  <State id="AnyMove" extend="Walk" allowSelect="1"><SelectWhen><PseudoSpeed/></SelectWhen></State>
  <State id="Start" extend="Base" allowSelect="0"/>
  <State id="Hidden" extend="Idle" includeInGame="0"/>
 </States>
 <Transitions>
  <Link from="Idle" to="Start" cost="100" transitionTime="0.1" forceFollowChance="0"/>
  <Link from="Start" to="Walk" cost="100" transitionTime="0.2" forceFollowChance="0"/>
  <Link from="Idle" to="Walk" cost="300" transitionTime="0.3" forceFollowChance="0"/>
  <Link from="Walk" to="Run" cost="100" transitionTime="0.15" forceFollowChance="0"/>
 </Transitions>
</AnimationGraph>"#;

    fn g() -> Graph {
        Graph::parse(XML.as_bytes()).expect("parse")
    }

    #[test]
    fn parses_inputs_states_links_and_inheritance() {
        let g = g();
        assert_eq!(g.inputs.len(), 3);
        assert_eq!(g.states.len(), 7);
        assert_eq!(g.links.len(), 4);
        let idle = &g.resolved[g.state_index("Idle").unwrap()];
        assert!(idle.flag("includeInGame"), "inherited flag");
        assert_eq!(idle.criteria.get("Action"), Some(&Criterion::Key("idle".into())), "inherited criterion");
        let any = &g.resolved[g.state_index("AnyMove").unwrap()];
        assert_eq!(any.criteria.get("PseudoSpeed"), Some(&Criterion::Any), "empty element removes the range");
    }

    #[test]
    fn selection_uses_ranges_and_specificity() {
        let g = g();
        let mut v = g.default_inputs();
        assert_eq!(g.select(&v).map(|i| g.states[i].id.as_str()), Some("Idle"));
        v.set_key("Action", "move");
        v.set_number("PseudoSpeed", 1.0);
        // Walk and AnyMove both match; Walk constrains more.
        assert_eq!(g.select(&v).map(|i| g.states[i].id.as_str()), Some("Walk"));
        v.set_number("PseudoSpeed", 2.0); // half-open: 2.0 belongs to Run
        assert_eq!(g.select(&v).map(|i| g.states[i].id.as_str()), Some("Run"));
        v.set_number("PseudoSpeed", 3.0); // declared max is inclusive
        assert_eq!(g.select(&v).map(|i| g.states[i].id.as_str()), Some("Run"));
    }

    #[test]
    fn path_takes_the_cheapest_route_through_transition_states() {
        let g = g();
        let (idle, walk) = (g.state_index("Idle").unwrap(), g.state_index("Walk").unwrap());
        let p = g.path(idle, walk).unwrap();
        let ids: Vec<&str> = p.iter().map(|&l| g.states[g.links[l].to].id.as_str()).collect();
        assert_eq!(ids, ["Start", "Walk"], "200 via Start beats the direct 300 link");

        let mut inst = GraphInstance::default();
        let mut v = g.default_inputs();
        inst.update(&g, &v, false);
        assert_eq!(inst.current, Some(idle));
        v.set_key("Action", "move");
        v.set_number("PseudoSpeed", 1.0);
        inst.update(&g, &v, true);
        assert_eq!(g.states[inst.current.unwrap()].id, "Start");
        assert!((inst.blend - 0.1).abs() < 1e-6);
        inst.update(&g, &v, true);
        assert_eq!(inst.current, Some(walk));
    }
}
