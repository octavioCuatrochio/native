//! CryEngine 3 `mission_mission0.xml` scene-placement parsing.
//!
//! A mission file's `<Objects>` element is a flat instance stream: each direct
//! child is either an `<Entity>` (a gameplay/archetype instance, identified by
//! `EntityClass` and/or `Pos`) carrying renderable geometry in its
//! `<Properties object_Model="...">` child, or an `<Object>` (a raw scene node
//! such as a `Decal`, `Group`, `VoxelObject`, `Occluder.*` or
//! `DistanceCloud`).
//!
//! The same file also contains `<Entity Id="..."/>` *reference stubs* nested
//! inside volume `<Entities>` lists under `<Area>`/`<EntityLinks>`; they are
//! links only (no `EntityClass`, no `Pos`, no `EntityGuid`) and must not be
//! instantiated as scene objects. Because only direct children of `<Objects>`
//! are considered, and a placement entity must carry `EntityClass` or `Pos`
//! (the reference discriminator), those stubs are excluded.
//!
//! Transforms are world-space: `Pos="x,y,z"` in metres and `Rotate="x,y,z,w"`
//! a unit quaternion with the scalar last. `Scale` is optional and defaults to
//! `(1,1,1)`. Missing transforms fall back to identity/zero rather than
//! dropping the placement, so traversal counts stay faithful to the source.
//!
//! Robustness: `quick-xml` errors are treated as end-of-input, attribute
//! values that fail to parse fall back to their defaults, and no node can
//! panic the parser.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// One placed node from `<Objects>`.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    /// `EntityClass` for an entity, `Type` for an object.
    pub class: String,
    /// The authored `Name` attribute (used to locate story landmarks such as
    /// `Home_RangerStation` / `Interior_TannerTent`); `None` when unnamed.
    pub name: Option<String>,
    /// `Properties.object_Model` for an entity (forward-slash normalised,
    /// original case preserved); `None` for objects and model-less entities.
    pub model: Option<String>,
    /// World position in metres.
    pub pos: [f32; 3],
    /// World rotation quaternion, component order `(x, y, z, w)`.
    pub rot: [f32; 4],
    /// Non-uniform scale, defaulting to `(1, 1, 1)`.
    pub scale: [f32; 3],
    /// `true` for `<Entity>`, `false` for `<Object>`.
    pub is_entity: bool,
}

const IDENTITY_ROT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const UNIT_SCALE: [f32; 3] = [1.0, 1.0, 1.0];
const ZERO_POS: [f32; 3] = [0.0, 0.0, 0.0];

/// Parse the `<Objects>` section of a mission XML into placements.
///
/// Only direct `<Entity>`/`<Object>` children of `<Objects>` are returned, in
/// document order. Nested entities (reference stubs) are ignored. Malformed
/// XML terminates parsing without panicking; malformed attributes fall back to
/// their defaults.
pub fn parse_mission(xml: &[u8]) -> Vec<Placement> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    // Names of currently open elements (bytes to avoid UTF-8 decoding work).
    let mut stack: Vec<Vec<u8>> = Vec::new();
    let mut placements: Vec<Placement> = Vec::new();
    // The `<Entity>` currently open as a direct child of `<Objects>`, whose
    // `<Properties object_Model>` we still need to fold in.
    let mut pending: Option<Placement> = None;

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
                let parent = stack.last().map(|v| v.as_slice());
                if parent == Some(&b"Objects"[..]) {
                    if name.as_slice() == b"Entity" {
                        if is_placement_entity(&e) {
                            pending = Some(entity_from(&e));
                        }
                    } else if name.as_slice() == b"Object" {
                        placements.push(object_from(&e));
                    }
                } else if parent == Some(&b"Entity"[..])
                    && name.as_slice() == b"Properties"
                {
                    if let Some(p) = pending.as_mut() {
                        if let Some(model) = get_attr(&e, "object_Model") {
                            p.model = normalize_model(&model);
                        }
                    }
                }
                stack.push(name);
            }

            Event::Empty(e) => {
                let name = e.name();
                let name = name.as_ref();
                let parent = stack.last().map(|v| v.as_slice());
                if parent == Some(&b"Objects"[..]) {
                    if name == b"Entity" {
                        if is_placement_entity(&e) {
                            placements.push(entity_from(&e));
                        }
                    } else if name == b"Object" {
                        placements.push(object_from(&e));
                    }
                } else if parent == Some(&b"Entity"[..]) && name == b"Properties" {
                    if let Some(p) = pending.as_mut() {
                        if let Some(model) = get_attr(&e, "object_Model") {
                            p.model = normalize_model(&model);
                        }
                    }
                }
            }

            Event::End(e) => {
                let name = e.name().as_ref().to_vec();
                stack.pop();
                // Close out the pending placement only when the entity we just
                // closed was itself a direct child of `<Objects>`.
                if name.as_slice() == b"Entity"
                    && stack.last().map(|v| v.as_slice()) == Some(&b"Objects"[..])
                {
                    if let Some(p) = pending.take() {
                        placements.push(p);
                    }
                }
            }

            _ => {}
        }
        buf.clear();
    }

    placements
}

/// A placement entity carries `EntityClass` and/or `Pos`; volume reference
/// stubs (`<Entity Id="11197"/>`) carry neither.
fn is_placement_entity(e: &BytesStart) -> bool {
    get_attr(e, "EntityClass").is_some() || get_attr(e, "Pos").is_some()
}

fn entity_from(e: &BytesStart) -> Placement {
    Placement {
        class: get_attr(e, "EntityClass").unwrap_or_default(),
        name: get_attr(e, "Name"),
        model: None,
        pos: parse_vec3(get_attr(e, "Pos").as_deref(), ZERO_POS),
        rot: parse_vec4(get_attr(e, "Rotate").as_deref(), IDENTITY_ROT),
        scale: parse_vec3(get_attr(e, "Scale").as_deref(), UNIT_SCALE),
        is_entity: true,
    }
}

fn object_from(e: &BytesStart) -> Placement {
    Placement {
        class: get_attr(e, "Type").unwrap_or_default(),
        name: get_attr(e, "Name"),
        model: None,
        pos: parse_vec3(get_attr(e, "Pos").as_deref(), ZERO_POS),
        rot: parse_vec4(get_attr(e, "Rotate").as_deref(), IDENTITY_ROT),
        scale: parse_vec3(get_attr(e, "Scale").as_deref(), UNIT_SCALE),
        is_entity: false,
    }
}

/// Look up one attribute by its exact case-sensitive name, unescaping XML
/// entities. Returns `None` if absent or non-UTF-8.
fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

/// Normalise a `object_Model` path to forward slashes. Case is preserved: the
/// mission mixes `objects/` and `Objects/` while the asset VFS is
/// case-insensitive, so callers fold case at lookup time.
fn normalize_model(s: &str) -> Option<String> {
    let n = s.trim().replace('\\', "/");
    if n.is_empty() {
        None
    } else {
        Some(n)
    }
}

fn parse_floats(s: &str) -> Vec<f32> {
    s.replace(' ', "")
        .split(',')
        .filter(|p| !p.is_empty())
        .filter_map(|p| p.parse::<f32>().ok())
        .collect()
}

/// Parse a comma-separated float triplet. Falls back to `default` unless all
/// three components parse.
fn parse_vec3(s: Option<&str>, default: [f32; 3]) -> [f32; 3] {
    match s {
        Some(t) => {
            let v = parse_floats(t);
            if v.len() >= 3 {
                [v[0], v[1], v[2]]
            } else {
                default
            }
        }
        None => default,
    }
}

/// Parse a comma-separated float quad (quaternion `x,y,z,w`). Falls back to
/// `default` unless all four components parse.
fn parse_vec4(s: Option<&str>, default: [f32; 4]) -> [f32; 4] {
    match s {
        Some(t) => {
            let v = parse_floats(t);
            if v.len() >= 4 {
                [v[0], v[1], v[2], v[3]]
            } else {
                default
            }
        }
        None => default,
    }
}

/// Average world position of every placement whose `Name` contains one of
/// `needles` (case-insensitive). Positions with all-zero coordinates are
/// skipped so unplaced references do not drag the centroid to the origin.
pub fn named_centroid(placements: &[Placement], needles: &[&str]) -> Option<[f32; 3]> {
    let needles: Vec<String> = needles.iter().map(|n| n.to_ascii_lowercase()).collect();
    let mut sum = [0f64; 3];
    let mut n = 0u32;
    for p in placements {
        let Some(name) = &p.name else { continue };
        let lname = name.to_ascii_lowercase();
        if !needles.iter().any(|needle| lname.contains(needle.as_str())) {
            continue;
        }
        if p.pos == ZERO_POS {
            continue;
        }
        sum[0] += p.pos[0] as f64;
        sum[1] += p.pos[1] as f64;
        sum[2] += p.pos[2] as f64;
        n += 1;
    }
    if n == 0 {
        None
    } else {
        let f = n as f64;
        Some([(sum[0] / f) as f32, (sum[1] / f) as f32, (sum[2] / f) as f32])
    }
}

/// All placements whose `Name` contains `needle` (case-insensitive).
pub fn find_named<'a>(placements: &'a [Placement], needle: &str) -> Vec<&'a Placement> {
    let needle = needle.to_ascii_lowercase();
    placements
        .iter()
        .filter(|p| p.name.as_deref().map(|n| n.to_ascii_lowercase().contains(&needle)).unwrap_or(false))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<Mission>
 <Objects>
  <Entity Name="e1" EntityClass="RoadSign" Pos="1,2,3" Rotate="0,0,0,1">
   <Properties object_Model="Objects\Decor\Sign.cgf"><Breakage/></Properties>
  </Entity>
  <Entity Name="e2" EntityClass="TagPoint" Pos="4,5,6"/>
  <Entity Id="11197"/>
  <Object Type="Decal" Pos="7,8,9" Rotate="0.1,0.2,0.3,0.9" Scale="2,2,2"/>
  <Object Type="Group" Name="g"/>
  <Area><Entities><Entity Id="22222"/></Entities></Area>
 </Objects>
</Mission>"#;

    #[test]
    fn parses_objects_and_excludes_stubs() {
        let p = parse_mission(SAMPLE.as_bytes());
        assert_eq!(p.len(), 4, "2 entities + 2 objects; stubs excluded");

        let e1 = &p[0];
        assert!(e1.is_entity);
        assert_eq!(e1.class, "RoadSign");
        assert_eq!(e1.model.as_deref(), Some("Objects/Decor/Sign.cgf"));
        assert_eq!(e1.pos, [1.0, 2.0, 3.0]);
        assert_eq!(e1.rot, IDENTITY_ROT);
        assert_eq!(e1.scale, UNIT_SCALE);

        let e2 = &p[1];
        assert!(e2.is_entity);
        assert_eq!(e2.model, None);
        assert_eq!(e2.rot, IDENTITY_ROT);

        let o = &p[2];
        assert!(!o.is_entity);
        assert_eq!(o.class, "Decal");
        assert_eq!(o.pos, [7.0, 8.0, 9.0]);
        assert_eq!(o.rot, [0.1, 0.2, 0.3, 0.9]);
        assert_eq!(o.scale, [2.0, 2.0, 2.0]);

        let g = &p[3];
        assert_eq!(g.class, "Group");
        assert_eq!(g.pos, ZERO_POS);
        assert_eq!(g.scale, UNIT_SCALE);
    }

    #[test]
    fn handles_malformed_input_without_panicking() {
        assert!(parse_mission(b"").is_empty());
        assert!(parse_mission(b"<not xml").is_empty());
        assert!(parse_mission(b"<Mission><Objects/></Mission>").is_empty());
    }

    #[test]
    fn malformed_vectors_fall_back_to_defaults() {
        let xml = br#"<Mission><Objects>
            <Entity EntityClass="X" Pos="1,bad,3" Rotate="1,2" Scale="nope"/>
        </Objects></Mission>"#;
        let p = parse_mission(xml);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].pos, ZERO_POS);
        assert_eq!(p[0].rot, IDENTITY_ROT);
        assert_eq!(p[0].scale, UNIT_SCALE);
    }
}
