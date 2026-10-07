//! Console variables: the defaults the original executable registers, with the
//! shipped config files and profile options applied in the engine's load order.
//!
//! Registration order in the original engine (CryEngine `IConsole`):
//!   1. every `Register*` call in the executable's static init, setting each
//!      cvar to its default (recovered in [`crate::cry::cvar_table`]);
//!   2. the config files executed at boot (`system.cfg`, `game.cfg`, engine and
//!      game `config/` and `libs/config/` cfgs);
//!   3. the profile's `OptionCfg.*` attributes (`attributes.xml`), which map to
//!      the cvar of the same name without the prefix.
//!
//! `System.GetCVar` reads the resulting lower-cased name -> value map.

use crate::crypak::Vfs;
use std::collections::HashMap;

pub use super::cvar_table::{Def, Kind, CVARS};

/// Look up a registered cvar by case-insensitive name.
pub fn find(name: &str) -> Option<&'static Def> {
    let key = name.to_ascii_lowercase();
    CVARS.binary_search_by(|d| d.name.cmp(key.as_str())).ok().map(|i| &CVARS[i])
}

/// Registered kind of `name`, if the executable registered it.
pub fn kind(name: &str) -> Option<Kind> {
    find(name).map(|d| d.kind)
}

/// Config files executed at boot, in the engine's order. Missing files are
/// skipped; every path is resolved through the VFS (paks and loose root).
const CONFIG_FILES: &[&str] = &[
    "system.cfg",
    "game.cfg",
    "libs/config/demotuning.cfg",
    "libs/config/meleeprototype.cfg",
    "config/singleplayer.cfg",
    "config/windows.cfg",
    "config/console.cfg",
];

/// The shipped default profile; its `OptionCfg.*` entries override cvars.
const PROFILE_ATTRIBUTES: &str = "libs/config/profiles/default/attributes.xml";
/// A per-user profile, when mounted as a loose file, wins over the shipped one.
const USER_ATTRIBUTES: &str = "profiles/default/attributes.xml";

/// Strip a comment (`--`, `//`, `;`, `#`) that starts outside a quoted string.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut quote = 0u8;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if quote != 0 {
            if c == quote {
                quote = 0;
            }
        } else if c == b'"' || c == b'\'' {
            quote = c;
        } else if c == b'-' && bytes.get(i + 1) == Some(&b'-') {
            return &line[..i];
        } else if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            return &line[..i];
        } else if c == b';' || c == b'#' {
            return &line[..i];
        }
        i += 1;
    }
    line
}

/// Unquote a value and trim surrounding whitespace.
fn unquote(v: &str) -> String {
    let v = v.trim();
    if v.len() >= 2 {
        let b = v.as_bytes();
        if (b[0] == b'"' && b[v.len() - 1] == b'"') || (b[0] == b'\'' && b[v.len() - 1] == b'\'') {
            return v[1..v.len() - 1].to_string();
        }
    }
    v.to_string()
}

/// Parse a cvar config body: `name = value` or `name value`, one per line.
/// Comments (`--`, `//`, `;`, `#`) and `[section]` headers are ignored. Names
/// are lower-cased; values are unquoted.
pub fn parse_config(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = strip_comment(raw).trim();
        if line.is_empty() || line.starts_with('[') {
            continue;
        }
        let (name, value) = match line.split_once('=') {
            Some((n, v)) => (n.trim(), v.trim()),
            None => match line.split_once(char::is_whitespace) {
                Some((n, v)) => (n.trim(), v.trim()),
                None => continue,
            },
        };
        if name.is_empty() {
            continue;
        }
        out.push((name.to_ascii_lowercase(), unquote(value)));
    }
    out
}

/// Parse profile `<Attr name="OptionCfg.foo" value="bar"/>` entries into
/// `(cvar, value)` pairs. Only the `OptionCfg.` namespace maps to cvars.
pub fn parse_profile_attributes(text: &str) -> Vec<(String, String)> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut out = Vec::new();
    let mut reader = Reader::from_str(text);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                if e.name().as_ref() != b"Attr" {
                    continue;
                }
                let mut name = None;
                let mut value = None;
                for a in e.attributes().flatten() {
                    let val = a.unescape_value().map(|c| c.into_owned()).ok();
                    match a.key.as_ref() {
                        b"name" => name = val,
                        b"value" => value = val,
                        _ => {}
                    }
                }
                if let (Some(n), Some(v)) = (name, value) {
                    if let Some(cvar) = n.strip_prefix("OptionCfg.") {
                        out.push((cvar.to_ascii_lowercase(), v));
                    }
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out
}

fn apply_config(vfs: &Vfs, path: &str, map: &mut HashMap<String, String>) {
    if let Some(bytes) = vfs.read(path) {
        let text = String::from_utf8_lossy(&bytes);
        for (k, v) in parse_config(&text) {
            map.insert(k, v);
        }
    }
}

fn apply_attributes(vfs: &Vfs, path: &str, map: &mut HashMap<String, String>) {
    if let Some(bytes) = vfs.read(path) {
        let text = String::from_utf8_lossy(&bytes);
        for (k, v) in parse_profile_attributes(&text) {
            map.insert(k, v);
        }
    }
}

/// Seed every registered default, then apply the shipped configs and the
/// profile attributes. Returns lower-cased cvar name -> value string.
pub fn load(vfs: &Vfs) -> HashMap<String, String> {
    let mut map: HashMap<String, String> =
        CVARS.iter().map(|d| (d.name.to_string(), d.default.to_string())).collect();
    for path in CONFIG_FILES {
        apply_config(vfs, path, &mut map);
    }
    apply_attributes(vfs, PROFILE_ATTRIBUTES, &mut map);
    apply_attributes(vfs, USER_ATTRIBUTES, &mut map);
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn parses_equals_space_and_quotes() {
        let text = "a = 1\nb 2\nc = \"three\"\nd '4'\n";
        let got = parse_config(text);
        assert_eq!(
            got,
            vec![
                ("a".into(), "1".into()),
                ("b".into(), "2".into()),
                ("c".into(), "three".into()),
                ("d".into(), "4".into()),
            ]
        );
    }

    #[test]
    fn parses_comments_sections_and_case() {
        let text = "-- a = 1\n; b = 2\n// g_Test = 9\n# nope = 1\n[default]\n= 3\nG_PlayerHealthValue = 150\n";
        let got = parse_config(text);
        assert_eq!(got, vec![("g_playerhealthvalue".into(), "150".into())]);
    }

    #[test]
    fn parses_inline_comment_after_value() {
        let got = parse_config("r_Foo = 3 -- keep three\ns_Bar = \"x;y\"\n");
        assert_eq!(got, vec![("r_foo".into(), "3".into()), ("s_bar".into(), "x;y".into())]);
    }

    #[test]
    fn parses_optioncfg_attributes() {
        let xml = r#"<Attributes Version="7">
            <Attr name="OptionCfg.cl_sensitivity" value="45"/>
            <Attr name="Option.hud_voicemode" value="1"/>
            <Attr name="OptionCfg.g_quickGame_mode" value="InstantAction"/>
        </Attributes>"#;
        let got = parse_profile_attributes(xml);
        assert_eq!(
            got,
            vec![
                ("cl_sensitivity".into(), "45".into()),
                ("g_quickgame_mode".into(), "InstantAction".into()),
            ]
        );
    }

    #[test]
    fn known_defaults_have_kinds() {
        assert_eq!(kind("g_playerHealthValue"), Some(Kind::Float));
        assert_eq!(kind("cl_sensitivity"), Some(Kind::Float));
        assert_eq!(kind("cl_invertMouse"), Some(Kind::Int));
        assert_eq!(find("g_playerHealthValue").unwrap().default, "100");
        assert!(CVARS.len() > 1000);
        assert!(kind("this_cvar_does_not_exist").is_none());
    }

    #[test]
    fn load_applies_loose_config_overrides() {
        let dir = std::env::temp_dir().join(format!("sod_cvars_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("system.cfg"), b"g_playerHealthValue = 250\nai_debugdraw = 4\n").unwrap();
        let mut vfs = Vfs::new();
        vfs.set_loose_root(&dir);
        let map = load(&vfs);
        assert_eq!(map.get("g_playerhealthvalue").map(String::as_str), Some("250"));
        assert_eq!(map.get("ai_debugdraw").map(String::as_str), Some("4"));
        // A cvar untouched by the config keeps its registered default.
        assert_eq!(map.get("cl_sensitivity").map(String::as_str), Some("45"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    const GAME: &str = "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";

    #[test]
    fn game_paks_resolve_key_cvars_as_numbers() {
        if !Path::new(GAME).is_dir() {
            return; // game not installed; nothing to check
        }
        let mut paks = Vec::new();
        fn walk(root: &Path, out: &mut Vec<std::path::PathBuf>) {
            if let Ok(rd) = std::fs::read_dir(root) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        walk(&p, out);
                    } else if p.extension().map(|x| x == "pak").unwrap_or(false) {
                        out.push(p);
                    }
                }
            }
        }
        walk(Path::new(GAME), &mut paks);
        paks.sort();
        let mut vfs = Vfs::new();
        for p in &paks {
            let _ = vfs.mount(p);
        }
        let map = load(&vfs);
        for name in ["g_playerhealthvalue", "cl_sensitivity"] {
            let v = map.get(name).unwrap_or_else(|| panic!("{name} missing"));
            assert!(v.trim().parse::<f64>().is_ok(), "{name} = {v:?} is not numeric");
        }
        // The profile's OptionCfg.cl_sensitivity = 45 must win over the default.
        assert_eq!(map.get("cl_sensitivity").map(String::as_str), Some("45"));
    }
}
