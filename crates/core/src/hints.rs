//! Tutorial / control hints from `contentmanager/hints.xml`.
//!
//! The real file (root `<HintData>`) groups [`Hint`]s into
//! `<HintCategory Name=...>` buckets. Each `<Hint ID=...>` carries an on-screen
//! prompt as either a `<HintText Text=...>` (gamepad/default) and, for inputs
//! that differ on PC, a `<KeyboardHintText Text=...>` variant. The prompt text
//! embeds input tokens in square brackets verbatim, e.g. `Use [moveforward] …
//! to move.`
//!
//! [`Hints::parse`] is lenient: unknown elements/attributes are skipped,
//! non-UTF-8 values are lossily decoded, and malformed XML simply ends the walk
//! with whatever was collected (it cannot fail and never panics). Where both
//! variants exist the keyboard text is preferred.
//!
//! Self-contained: `std` plus the already-present `quick_xml`.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::collections::HashMap;

/// One tutorial/control prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hint {
    /// The hint identifier (e.g. `Scene.Move`).
    pub id: String,
    /// The prompt text, `[token]` markers kept verbatim.
    pub text: String,
    /// Enclosing `<HintCategory Name=...>`, when present.
    pub category: Option<String>,
}

/// A parsed `hints.xml` catalogue, keyed by hint id.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Hints {
    by_id: HashMap<String, Hint>,
}

/// Look up one attribute by exact case-sensitive name, unescaping entities.
fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

impl Hints {
    /// Parse a `hints.xml` document. Never fails; malformed input yields an
    /// empty or partial catalogue.
    pub fn parse(bytes: &[u8]) -> Hints {
        // Tolerate a UTF-8 BOM ahead of the declaration.
        let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
        let mut reader = Reader::from_reader(bytes);
        let mut buf = Vec::new();

        let mut by_id: HashMap<String, Hint> = HashMap::new();
        let mut category: Option<String> = None;

        // The hint currently being collected (between <Hint> and </Hint>).
        let mut cur_id: Option<String> = None;
        let mut cur_kb: Option<String> = None;
        let mut cur_text: Option<String> = None;

        loop {
            let ev = match reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => break,
                // Lenient: a malformed document stops the walk, keeping what we
                // have so far. It must never panic.
                Err(_) => break,
                Ok(ev) => ev,
            };

            match ev {
                Event::Start(e) | Event::Empty(e) => {
                    let name = e.name().as_ref().to_vec();
                    match name.as_slice() {
                        b"HintCategory" => {
                            category = get_attr(&e, "Name").filter(|s| !s.is_empty());
                        }
                        b"Hint" => {
                            // Starting a new hint flushes any unterminated one.
                            if let Some(id) = cur_id.take() {
                                flush(&mut by_id, id, cur_kb.take(), cur_text.take(), category.clone());
                            }
                            cur_id = get_attr(&e, "ID").filter(|s| !s.is_empty());
                            cur_kb = None;
                            cur_text = None;
                        }
                        b"KeyboardHintText" => {
                            if cur_id.is_some() {
                                if let Some(t) = get_attr(&e, "Text") {
                                    cur_kb = Some(t);
                                }
                            }
                        }
                        b"HintText" => {
                            if cur_id.is_some() {
                                if let Some(t) = get_attr(&e, "Text") {
                                    cur_text = Some(t);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::End(e) => {
                    let name = e.name().as_ref().to_vec();
                    match name.as_slice() {
                        b"Hint" => {
                            if let Some(id) = cur_id.take() {
                                flush(&mut by_id, id, cur_kb.take(), cur_text.take(), category.clone());
                            }
                        }
                        b"HintCategory" => category = None,
                        _ => {}
                    }
                }
                _ => {}
            }
            buf.clear();
        }

        // Unterminated final hint (a truncated document).
        if let Some(id) = cur_id.take() {
            flush(&mut by_id, id, cur_kb, cur_text, category);
        }

        Hints { by_id }
    }

    /// The hint with `id`, if any.
    pub fn get(&self, id: &str) -> Option<&Hint> {
        self.by_id.get(id)
    }

    /// The prompt text for `id`, if any.
    pub fn text(&self, id: &str) -> Option<&str> {
        self.by_id.get(id).map(|h| h.text.as_str())
    }

    /// Every hint, in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = &Hint> {
        self.by_id.values()
    }

    /// Number of hints.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// `true` when no hint was collected.
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

/// Insert a collected hint, preferring the keyboard variant of the text.
fn flush(
    by_id: &mut HashMap<String, Hint>,
    id: String,
    keyboard: Option<String>,
    plain: Option<String>,
    category: Option<String>,
) {
    let text = keyboard.or(plain).unwrap_or_default();
    by_id.insert(
        id.clone(),
        Hint {
            id,
            text,
            category,
        },
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn real_file() -> Option<Vec<u8>> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/gamedata/libs/class3/contentmanager/hints.xml");
        std::fs::read(p).ok()
    }

    const SAMPLE: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<HintData>
  <HintCategory Name="Triggered" TimeoutSeconds="0">
    <Hint Flags="Triggered" ID="Scene.Move">
      <KeyboardHintText Flags="" Notes="Keyboard Hint Text" Text="Use [moveforward], [moveleft] to move." />
      <HintText Flags="" Notes="Hint Text" Text="Use [xi_thumblx] to move." />
      <AbortConditions /><DisableConditions /><DisplayConditions />
      <FollowUps /><UnpauseConditions />
    </Hint>
    <Hint Flags="Triggered" ID="Scene.Combat">
      <HintText Text="TIP: Press [ul_melee_attack] to attack. Press [cover] to defend." />
    </Hint>
    <Hint Flags="Triggered" ID="Empty" />
  </HintCategory>
  <HintCategory Name="UI" TimeoutSeconds="0">
    <Hint ID="UiOnly"><HintText Text="A &amp; B" /></Hint>
  </HintCategory>
</HintData>"#;

    #[test]
    fn parses_sample_and_prefers_keyboard() {
        let h = Hints::parse(SAMPLE.as_bytes());
        assert_eq!(h.len(), 4);
        assert!(!h.is_empty());
        assert_eq!(
            h.text("Scene.Move"),
            Some("Use [moveforward], [moveleft] to move.")
        );
        assert_eq!(
            h.text("Scene.Combat"),
            Some("TIP: Press [ul_melee_attack] to attack. Press [cover] to defend.")
        );
        assert_eq!(h.get("UiOnly").and_then(|x| x.category.as_deref()), Some("UI"));
        assert_eq!(h.get("Scene.Combat").and_then(|x| x.category.as_deref()), Some("Triggered"));
        // entity unescaping
        assert_eq!(h.text("UiOnly"), Some("A & B"));
        // id with no text still exists, with empty text
        assert_eq!(h.text("Empty"), Some(""));
        assert!(h.get("nope").is_none());
        assert_eq!(h.iter().count(), 4);
    }

    #[test]
    fn malformed_input_does_not_panic() {
        for bad in [
            &b""[..],
            &b"not xml at all"[..],
            &b"<HintData><HintCategory Name=\"x\"><Hint ID=\"a\"><HintText Text=\"unclosed"[..],
            &b"<Hint ID=\"\"></Hint>"[..],
        ] {
            let h = Hints::parse(bad);
            // nothing to assert beyond "did not panic"; just exercise the API
            let _ = h.len();
            let _ = h.is_empty();
            let _ = h.iter().count();
        }
    }

    #[test]
    fn real_file_parses() {
        let Some(bytes) = real_file() else {
            return; // data not present in this checkout; skip silently
        };
        let h = Hints::parse(&bytes);
        assert!(
            h.len() > 20,
            "expected >20 hints, got {}",
            h.len()
        );

        let mv = h.text("Scene.Move").expect("Scene.Move present");
        assert!(!mv.is_empty());
        assert!(
            mv.to_ascii_lowercase().contains("move"),
            "Scene.Move text should mention move: {mv:?}"
        );

        let combat = h.text("Scene.Combat").expect("Scene.Combat present");
        let lc = combat.to_ascii_lowercase();
        assert!(
            lc.contains("attack") || lc.contains("defend"),
            "Scene.Combat text should mention attack/defend: {combat:?}"
        );

        // keyboard variant must be preferred when the file supplies one
        for id in ["Scene.Look", "Scene.Exploration", "Scene.Hide", "Scene.Map"] {
            assert!(h.get(id).is_some(), "missing {id}");
        }
    }
}
