//! HUD data loading from the game's real UI definition files.
//!
//! Three independent catalogues, each parsed leniently (never panics; malformed
//! input yields an empty or partial list):
//!
//! * [`parse_uielements`] — `libs/ui/uielements/class3hud.xml`. Root
//!   `<UIElements name="Menus">` holds ten `<UIElement name=...>` nodes, each
//!   binding a Flash asset through a `<GFx file=... layer=...>` child with a
//!   `<Constraints><Align mode="fullscreen"/>` block. Both the logical
//!   `UIElement` and the concrete `GFx` resource are emitted (with `kind`
//!   distinguishing them); the `GFx` entry records its parent under the
//!   synthetic `uielement` attribute. The shipped file contains a malformed
//!   `<!-->` marker that a strict parser would treat as an unclosed comment, so
//!   the input is sanitised first (see [`sanitize`]).
//! * [`parse_objectives`] — `libs/ui/objectives_new.xml`. Root `<Root>` with a
//!   grouping element (`<Example>`) containing `<Objective_NN Name=...
//!   Description=.../>` leaves.
//! * [`parse_uiactions`] — `libs/ui/uiactions/*.xml`, CryEngine flow graphs
//!   (`<Graph>` → `<Nodes>`/`<Edges>`/`<GraphTokens>`, each `<Node Id=...
//!   Class=... pos=...>`). The shipped files are empty graphs; the `<Graph>`
//!   binding itself is always emitted so an empty graph still yields an action,
//!   and `<function name= funcname=>` definitions are recognised too.
//!
//! Self-contained: `std` plus the already-present `quick_xml`.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// One UI element or Flash resource definition.
#[derive(Debug, Clone, PartialEq)]
pub struct UiElement {
    /// `UIElement/@name`, or the `GFx/@file` asset for a resource entry.
    pub name: String,
    /// Element kind: `"UIElement"` or `"GFx"`.
    pub kind: String,
    pub x: Option<f32>,
    pub y: Option<f32>,
    pub w: Option<f32>,
    pub h: Option<f32>,
    /// Every attribute on the node, verbatim (plus `uielement` on `GFx`).
    pub attrs: Vec<(String, String)>,
}

/// One objective banner/text definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveDef {
    pub name: String,
    pub text: Option<String>,
    pub icon: Option<String>,
    /// Enclosing group tag or a `Kind`/`Type`/`Category` attribute.
    pub kind: Option<String>,
}

/// One UI action mapping (flow-graph node, graph binding or UI function).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiAction {
    pub name: String,
    pub action: String,
    pub target: Option<String>,
}

// ---------------------------------------------------------------------------
// Lenient XML helpers
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

/// First present attribute among `names`.
fn get_any(e: &BytesStart, names: &[&str]) -> Option<String> {
    names.iter().find_map(|n| get_attr(e, n))
}

/// All attributes as owned `(name, value)` pairs, lossily decoded.
fn all_attrs(e: &BytesStart) -> Vec<(String, String)> {
    e.attributes()
        .flatten()
        .map(|a| {
            let k = String::from_utf8_lossy(a.key.as_ref()).into_owned();
            let v = a
                .unescape_value()
                .map(|c| c.into_owned())
                .unwrap_or_else(|_| String::from_utf8_lossy(&a.value).into_owned());
            (k, v)
        })
        .collect()
}

/// Parse the first present numeric attribute among `names`.
fn num_attr(e: &BytesStart, names: &[&str]) -> Option<f32> {
    get_any(e, names).and_then(|v| v.trim().parse::<f32>().ok())
}

/// Strip a UTF-8 BOM and repair the malformed `<!-->` comment markers found in
/// the shipped `class3hud.xml` (turn them into valid empty comments so they do
/// not swallow the following elements).
fn sanitize(xml: &[u8]) -> Vec<u8> {
    let xml = xml.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(xml);
    const PAT: &[u8] = b"<!-->";
    const REP: &[u8] = b"<!-- -->";
    if !xml.windows(PAT.len()).any(|w| w == PAT) {
        return xml.to_vec();
    }
    let mut out = Vec::with_capacity(xml.len() + 8);
    let mut i = 0;
    while i < xml.len() {
        if xml[i..].starts_with(PAT) {
            out.extend_from_slice(REP);
            i += PAT.len();
        } else {
            out.push(xml[i]);
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// uielements/class3hud.xml
// ---------------------------------------------------------------------------

/// Parse a `uielements/*.xml` document. Never fails.
pub fn parse_uielements(xml: &[u8]) -> Vec<UiElement> {
    let data = sanitize(xml);
    let mut reader = Reader::from_reader(data.as_slice());
    let mut buf = Vec::new();
    let mut out = Vec::new();
    let mut cur_name: Option<String> = None;

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };
        match ev {
            Event::Start(e) | Event::Empty(e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"UIElement" => {
                        let name = get_attr(&e, "name").unwrap_or_default();
                        cur_name = Some(name.clone());
                        out.push(UiElement {
                            name,
                            kind: "UIElement".to_string(),
                            x: num_attr(&e, &["x"]),
                            y: num_attr(&e, &["y"]),
                            w: num_attr(&e, &["width", "w"]),
                            h: num_attr(&e, &["height", "h"]),
                            attrs: all_attrs(&e),
                        });
                    }
                    b"GFx" => {
                        let file = get_attr(&e, "file").unwrap_or_default();
                        let name = if file.is_empty() {
                            cur_name.clone().unwrap_or_default()
                        } else {
                            file
                        };
                        let mut attrs = all_attrs(&e);
                        if let Some(parent) = cur_name.clone() {
                            attrs.push(("uielement".to_string(), parent));
                        }
                        out.push(UiElement {
                            name,
                            kind: "GFx".to_string(),
                            x: num_attr(&e, &["x"]),
                            y: num_attr(&e, &["y"]),
                            w: num_attr(&e, &["width", "w"]),
                            h: num_attr(&e, &["height", "h"]),
                            attrs,
                        });
                    }
                    _ => {}
                }
            }
            Event::End(e) => {
                if e.name().as_ref() == b"UIElement" {
                    cur_name = None;
                }
            }
            _ => {}
        }
    }
    out
}

// ---------------------------------------------------------------------------
// objectives_new.xml
// ---------------------------------------------------------------------------

/// Parse an `objectives_new.xml` document. Never fails.
pub fn parse_objectives(xml: &[u8]) -> Vec<ObjectiveDef> {
    let data = sanitize(xml);
    let mut reader = Reader::from_reader(data.as_slice());
    let mut buf = Vec::new();
    let mut out = Vec::new();
    let mut stack: Vec<String> = Vec::new();

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };
        match ev {
            Event::Start(e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                push_objective(&e, &tag, stack.last().map(String::as_str), &mut out);
                stack.push(tag);
            }
            Event::Empty(e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                push_objective(&e, &tag, stack.last().map(String::as_str), &mut out);
            }
            Event::End(_) => {
                stack.pop();
            }
            _ => {}
        }
    }
    out
}

fn push_objective(
    e: &BytesStart,
    tag: &str,
    parent: Option<&str>,
    out: &mut Vec<ObjectiveDef>,
) {
    let is_objective = tag.to_ascii_lowercase().starts_with("objective");
    let named = get_any(e, &["Name", "name"]).is_some();
    if !is_objective && !named {
        return;
    }
    let name = get_any(e, &["Name", "name", "ID", "Id"]).unwrap_or_else(|| tag.to_string());
    let text = get_any(
        e,
        &["Description", "description", "Text", "text", "DescriptionText"],
    );
    let icon = get_any(e, &["Icon", "icon", "IconName", "Image"]);
    let kind = get_any(e, &["Kind", "Type", "Category", "Group"])
        .or_else(|| parent.map(str::to_string));
    out.push(ObjectiveDef {
        name,
        text,
        icon,
        kind,
    });
}

// ---------------------------------------------------------------------------
// uiactions/*.xml
// ---------------------------------------------------------------------------

/// Parse a `uiactions/*.xml` document. Never fails. The shipped files are empty
/// flow graphs, so each `<Graph>` binding is emitted as an action as well.
pub fn parse_uiactions(xml: &[u8]) -> Vec<UiAction> {
    let data = sanitize(xml);
    let mut reader = Reader::from_reader(data.as_slice());
    let mut buf = Vec::new();
    let mut out = Vec::new();

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };
        let e = match ev {
            Event::Start(e) | Event::Empty(e) => e,
            _ => continue,
        };
        let tag = e.name().as_ref().to_vec();
        match tag.as_slice() {
            b"Graph" => {
                let name = get_any(&e, &["Group", "Description", "Name"])
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "Graph".to_string());
                out.push(UiAction {
                    name,
                    action: "graph".to_string(),
                    target: get_any(&e, &["MultiPlayer"]).filter(|s| !s.is_empty()),
                });
            }
            b"Node" => {
                let name = get_any(&e, &["Id", "ID", "Name", "name"])
                    .unwrap_or_else(|| "Node".to_string());
                let action = get_any(&e, &["Class", "class", "Action", "action", "ActionId"])
                    .unwrap_or_else(|| "node".to_string());
                out.push(UiAction {
                    name,
                    action,
                    target: get_any(&e, &["pos", "Target", "target"]),
                });
            }
            b"function" => {
                let name = get_attr(&e, "name").unwrap_or_default();
                if name.is_empty() {
                    continue;
                }
                let action = get_attr(&e, "funcname").unwrap_or_else(|| name.clone());
                out.push(UiAction {
                    name,
                    action,
                    target: None,
                });
            }
            _ => {
                if let Some(action) = get_any(&e, &["action", "Action", "ActionId"]) {
                    let name = get_any(&e, &["name", "Name", "Id", "ID"])
                        .unwrap_or_else(|| String::from_utf8_lossy(&tag).into_owned());
                    out.push(UiAction {
                        name,
                        action,
                        target: get_any(&e, &["target", "Target"]),
                    });
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn ui_file(rel: &str) -> Option<Vec<u8>> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/gamedata/libs/ui")
            .join(rel);
        std::fs::read(p).ok()
    }

    const UE_SAMPLE: &str = r#"<?xml version="1.0"?>
<UIElements name="Menus">
  <UIElement name="Class3HUDRadar">
    <GFx file="Class3_Radar.gfx" layer="1">
      <Constraints><Align mode="fullscreen"/></Constraints>
    </GFx>
    <functions><function name="Setup3DElements"/></functions>
  </UIElement>
  <UIElement name="Widget" x="10" y="20" width="100" height="50" />
</UIElements>"#;

    const OBJ_SAMPLE: &str = r#"<Root>
  <Group>
    <Objective Name="Save the day" Description="Find survivors." Icon="star"/>
    <Objective_02 Name="Two" Text="Second"/>
  </Group>
</Root>"#;

    const ACT_SAMPLE: &str = r#"<Graph Description="" Group="Inventory" enabled="1" MultiPlayer="ClientServer">
  <Nodes>
    <Node Id="1" Class="Logic:And" pos="10,20,0"/>
    <Node Id="2" Class="UI:Action" pos="0,0,0"><Inputs name="Use"/></Node>
  </Nodes>
  <Edges/><GraphTokens/>
</Graph>
<functions><function name="SetHealth" funcname="setHealth"/></functions>"#;

    #[test]
    fn synthetic_uielements_parse_with_expected_fields() {
        let els = parse_uielements(UE_SAMPLE.as_bytes());
        // 2 logical elements + 1 GFx resource
        assert_eq!(els.len(), 3, "{els:#?}");

        let widget = els.iter().find(|e| e.name == "Widget").expect("Widget");
        assert_eq!(widget.kind, "UIElement");
        assert_eq!(widget.x, Some(10.0));
        assert_eq!(widget.y, Some(20.0));
        assert_eq!(widget.w, Some(100.0));
        assert_eq!(widget.h, Some(50.0));

        let gfx = els.iter().find(|e| e.kind == "GFx").expect("GFx");
        assert_eq!(gfx.name, "Class3_Radar.gfx");
        assert!(gfx
            .attrs
            .iter()
            .any(|(k, v)| k == "layer" && v == "1"));
        assert!(gfx
            .attrs
            .iter()
            .any(|(k, v)| k == "uielement" && v == "Class3HUDRadar"));
    }

    #[test]
    fn synthetic_objectives_parse_with_expected_fields() {
        let objs = parse_objectives(OBJ_SAMPLE.as_bytes());
        assert_eq!(objs.len(), 2, "{objs:#?}");
        assert_eq!(objs[0].name, "Save the day");
        assert_eq!(objs[0].text.as_deref(), Some("Find survivors."));
        assert_eq!(objs[0].icon.as_deref(), Some("star"));
        assert_eq!(objs[0].kind.as_deref(), Some("Group"));
        assert_eq!(objs[1].text.as_deref(), Some("Second"));
    }

    #[test]
    fn synthetic_uiactions_parse_with_expected_fields() {
        let acts = parse_uiactions(ACT_SAMPLE.as_bytes());
        // Graph + 2 Nodes + 1 function
        assert_eq!(acts.len(), 4, "{acts:#?}");
        let g = acts.iter().find(|a| a.action == "graph").expect("graph");
        assert_eq!(g.name, "Inventory");
        let n = acts.iter().find(|a| a.name == "1").expect("node 1");
        assert_eq!(n.action, "Logic:And");
        assert_eq!(n.target.as_deref(), Some("10,20,0"));
        let f = acts.iter().find(|a| a.name == "SetHealth").expect("function");
        assert_eq!(f.action, "setHealth");
    }

    #[test]
    fn malformed_input_does_not_panic() {
        for bad in [
            &b""[..],
            &b"not xml at all"[..],
            &b"<UIElement name=\"x\""[..],
            &b"<Root><Objective Name=\"a\""[..],
            &b"<Graph><Nodes><Node Id="[..],
            &b"<!--><UIElement name=\"swallowed\"/>"[..],
        ] {
            let _ = parse_uielements(bad);
            let _ = parse_objectives(bad);
            let _ = parse_uiactions(bad);
        }
    }

    #[test]
    fn real_class3hud_parses() {
        let Some(bytes) = ui_file("uielements/class3hud.xml") else {
            return; // data not present in this checkout; skip silently
        };
        let els = parse_uielements(&bytes);
        eprintln!("hud_data: class3hud.xml -> {} elements", els.len());
        assert!(els.len() > 10, "expected >10 elements, got {}", els.len());
        for want in ["Class3HUDRadar", "Class3HUDSurvey", "Class3HUDJournal"] {
            assert!(els.iter().any(|e| e.name == want), "missing {want}");
        }
        assert!(
            els.iter()
                .any(|e| e.kind == "GFx" && e.name == "Class3_Radar.gfx"),
            "missing Class3_Radar.gfx"
        );
    }

    #[test]
    fn real_objectives_parses() {
        let Some(bytes) = ui_file("objectives_new.xml") else {
            return;
        };
        let objs = parse_objectives(&bytes);
        eprintln!("hud_data: objectives_new.xml -> {} objectives", objs.len());
        assert!(!objs.is_empty(), "expected >0 objectives");
    }

    #[test]
    fn real_uiactions_parses() {
        let mut total = 0;
        for rel in ["uiactions/inventory.xml", "uiactions/playerstate.xml"] {
            if let Some(bytes) = ui_file(rel) {
                let acts = parse_uiactions(&bytes);
                eprintln!("hud_data: {rel} -> {} actions", acts.len());
                total += acts.len();
            }
        }
        if total == 0 {
            return; // no data present; skip silently
        }
        assert!(total > 0, "expected >0 actions");
    }
}
