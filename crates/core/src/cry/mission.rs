//! Level entity records from `mission_*.xml`, with the full `<Properties>`
//! tree so the script layer can apply it exactly like CryEngine's
//! `CScriptProxy` XML serialisation.

use mlua::{Lua, Table};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// An XML property element: attributes plus named child elements.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PropNode {
    pub attrs: Vec<(String, String)>,
    pub children: Vec<(String, PropNode)>,
}

impl PropNode {
    fn from_start(e: &BytesStart) -> PropNode {
        PropNode { attrs: attrs_of(e), children: Vec::new() }
    }

    pub fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }

    pub fn child(&self, k: &str) -> Option<&PropNode> {
        self.children.iter().find(|(a, _)| a == k).map(|(_, v)| v)
    }

    /// `{ attr = "string", ..., __children = { Name = {...} } }` for the Lua applier.
    pub fn to_lua(&self, lua: &Lua) -> mlua::Result<Table> {
        let t = lua.create_table()?;
        for (k, v) in &self.attrs {
            t.raw_set(k.as_str(), v.as_str())?;
        }
        if !self.children.is_empty() {
            let c = lua.create_table()?;
            for (k, v) in &self.children {
                c.raw_set(k.as_str(), v.to_lua(lua)?)?;
            }
            t.raw_set("__children", c)?;
        }
        Ok(t)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LevelEntity {
    pub id: u32,
    pub name: String,
    pub class: String,
    pub guid: Option<String>,
    pub layer: Option<String>,
    pub pos: [f32; 3],
    /// `(x, y, z, w)`; the XML stores `Rotate="w,x,y,z"`.
    pub rot: [f32; 4],
    pub scale: [f32; 3],
    pub attrs: Vec<(String, String)>,
    pub properties: Option<PropNode>,
    pub properties_instance: Option<PropNode>,
}

impl LevelEntity {
    pub fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
}

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

fn floats<const N: usize>(s: Option<&str>, default: [f32; N]) -> [f32; N] {
    let Some(s) = s else { return default };
    let parts: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
    if parts.len() != N {
        return default;
    }
    let mut out = default;
    out.copy_from_slice(&parts);
    out
}

fn entity_header(e: &BytesStart) -> LevelEntity {
    let attrs = attrs_of(e);
    let get = |k: &str| attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
    let wxyz = floats(get("Rotate"), [1.0, 0.0, 0.0, 0.0]);
    LevelEntity {
        id: get("EntityId").and_then(|s| s.parse().ok()).unwrap_or(0),
        name: get("Name").unwrap_or_default().to_string(),
        class: get("EntityClass").unwrap_or_default().to_string(),
        guid: get("EntityGuid").map(str::to_string),
        layer: get("Layer").map(str::to_string),
        pos: floats(get("Pos"), [0.0; 3]),
        rot: [wxyz[1], wxyz[2], wxyz[3], wxyz[0]],
        scale: floats(get("Scale"), [1.0; 3]),
        properties: None,
        properties_instance: None,
        attrs,
    }
}

/// All `<Entity>` elements directly under `<Objects>`, in document order.
pub fn parse_level_entities(xml: &[u8]) -> Vec<LevelEntity> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut stack: Vec<Vec<u8>> = Vec::new();
    let mut out = Vec::new();
    let mut cur: Option<LevelEntity> = None;
    // Open property elements beneath the current entity: (tag, node).
    let mut props: Vec<(String, PropNode)> = Vec::new();

    fn close_prop(cur: &mut Option<LevelEntity>, props: &mut Vec<(String, PropNode)>) {
        let Some((tag, node)) = props.pop() else { return };
        if let Some((_, parent)) = props.last_mut() {
            parent.children.push((tag, node));
        } else if let Some(e) = cur.as_mut() {
            match tag.as_str() {
                "Properties" => e.properties = Some(node),
                "PropertiesInstance" => e.properties_instance = Some(node),
                _ => {}
            }
        }
    }

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) | Err(_) => break,
            Ok(ev) => ev,
        };
        match ev {
            Event::Start(e) => {
                let name = e.name().as_ref().to_vec();
                let parent = stack.last().map(|v| v.as_slice());
                if parent == Some(b"Objects") && name == b"Entity" {
                    cur = Some(entity_header(&e));
                } else if cur.is_some()
                    && (!props.is_empty()
                        || (stack.len() >= 2
                            && stack[stack.len() - 1] == b"Entity"
                            && (name == b"Properties" || name == b"PropertiesInstance")))
                {
                    props.push((String::from_utf8_lossy(&name).to_string(), PropNode::from_start(&e)));
                }
                stack.push(name);
            }
            Event::Empty(e) => {
                let name = e.name().as_ref().to_vec();
                let parent = stack.last().map(|v| v.as_slice());
                if parent == Some(b"Objects") && name == b"Entity" {
                    out.push(entity_header(&e));
                } else if cur.is_some()
                    && (!props.is_empty()
                        || (parent == Some(b"Entity") && (name == b"Properties" || name == b"PropertiesInstance")))
                {
                    props.push((String::from_utf8_lossy(&name).to_string(), PropNode::from_start(&e)));
                    close_prop(&mut cur, &mut props);
                }
            }
            Event::End(_) => {
                let name = stack.pop().unwrap_or_default();
                if !props.is_empty() && props.last().map(|p| p.0.as_bytes() == name.as_slice()).unwrap_or(false) {
                    close_prop(&mut cur, &mut props);
                } else if name == b"Entity" && stack.last().map(|v| v.as_slice()) == Some(b"Objects") {
                    if let Some(e) = cur.take() {
                        out.push(e);
                    }
                }
            }
            _ => {}
        }
        buf.clear();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<Mission><Objects>
  <Entity Name="pt" Pos="1,2,3" EntityClass="TagPoint" EntityId="1092" Layer="RTS"/>
  <Entity Name="Sign" Pos="4,5,6" Rotate="0.9238795,0,0,0.38268358" EntityClass="RoadSign" EntityId="5925">
   <Properties fHealth="1" object_Model="objects/a.cgf">
    <Physics bRigidBody="1" Mass="-1"><Simulation damping="0"/></Physics>
   </Properties>
  </Entity>
</Objects></Mission>"#;

    #[test]
    fn parses_entities_with_nested_properties() {
        let v = parse_level_entities(XML.as_bytes());
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].id, 1092);
        assert_eq!(v[0].layer.as_deref(), Some("RTS"));
        let s = &v[1];
        assert_eq!(s.class, "RoadSign");
        assert!((s.rot[3] - 0.9238795).abs() < 1e-6 && (s.rot[2] - 0.38268358).abs() < 1e-6);
        let p = s.properties.as_ref().unwrap();
        assert_eq!(p.attr("object_Model"), Some("objects/a.cgf"));
        let phys = p.child("Physics").unwrap();
        assert_eq!(phys.attr("bRigidBody"), Some("1"));
        assert_eq!(phys.child("Simulation").unwrap().attr("damping"), Some("0"));
    }
}
