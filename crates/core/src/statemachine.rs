//! Undead Labs `.sm` state machines (`ULSM`) and CryEngine `.chrparams`
//! animation-list includes.
//!
//! ## `.sm` / `ULSM` byte layout (recovered)
//!
//! Files live under `gamedata/libs/statemachines/<actor>/general.<plat>.sm`
//! (e.g. `player/general.win.sm`, `player/general.xboxone.sm`). The `.win` and
//! `.xboxone` builds are byte-identical little-endian; `.xbox360` is the same
//! layout with big-endian integers. All observed files start with the ASCII
//! magic `ULSM`.
//!
//! | offset | size | meaning |
//! |-------:|-----:|---------|
//! | `0x00` | 4 | magic `"ULSM"` |
//! | `0x04` | 8 | zero / reserved (version or flags) |
//! | `0x0c` | 16 | content hash / GUID, opaque |
//! | `0x1c` | 4 | unknown header word |
//! | `0x20` | 4 | **state count `N`** (number of entries in the offset array) |
//! | `0x24` | 8 | table counts (unknown) |
//! | `0x2c` | 4 | **state-name blob size** in bytes |
//! | `0x30` | 4 | file offset of the next section (secondary string table) |
//! | `0x34`… | 4 | remaining table counts/offsets (unknown) |
//! | `0x74` | 64 | **machine name**, NUL-padded fixed field (e.g. `"Player"`) |
//! | `0xb4` | `4*N` | `N` u32 **state-name start offsets**, relative to blob |
//! | `0xb4+4N` | — | packed NUL-terminated **state names** (`N` of them) |
//!
//! The offsets are monotonically increasing starts into the name blob; reading
//! `N` NUL-terminated strings beginning at the blob reproduces every state name
//! in file order. For `player/general.win.sm`: `N = 689`, blob size `33456`,
//! machine `"Player"`, first state `GraphDriven.Dead.Idle`.
//!
//! What this module recovers: the machine name and the ordered list of state
//! names (with their index). What it does **not** recover: the transitions,
//! conditions, animation bindings, per-state tables, the header hash, and the
//! sections after the name blob (offset `0x30`) — those remain opaque.
//!
//! ## `.chrparams`
//!
//! `skeleton.chrparams` and friends reference further animation lists via
//! `<AnimationList><Animation name="$Include" path="…"/></AnimationList>`.
//! Other `<Animation>` entries name real clips / wildcards
//! (`Basic_*`, `name="*"`, …) and are ignored by [`parse_chrparams`].
//!
//! Everything here is best-effort and lenient: malformed input yields `None`
//! or a partial/empty result, never a panic. Self-contained: `std` plus the
//! already-present `quick_xml`.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// One state (node) inside a [`StateMachine`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmState {
    /// Fully-qualified state name, e.g. `GraphDriven.Dead.Idle`.
    pub name: String,
    /// Position in the machine's state-name table.
    pub index: u32,
}

/// A parsed `.sm` (ULSM) state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateMachine {
    /// Machine name from the fixed header field, e.g. `Player`.
    pub name: String,
    /// States in file order.
    pub states: Vec<SmState>,
}

/// Offset of the state-count word.
const COUNT_OFF: usize = 0x20;
/// Offset of the fixed 64-byte machine-name field.
const NAME_OFF: usize = 0x74;
/// Length of the fixed machine-name field.
const NAME_BYTES: usize = 0x40;
/// Offset of the state-name start-offset array.
const OFFSET_ARR: usize = 0xb4;
/// Upper bound on a plausible state count (guards against bad endian reads).
const MAX_STATES: usize = 4_000_000;

fn read_u32(bytes: &[u8], off: usize, be: bool) -> Option<u32> {
    let s = bytes.get(off..off.checked_add(4)?)?;
    let a = [s[0], s[1], s[2], s[3]];
    Some(if be {
        u32::from_be_bytes(a)
    } else {
        u32::from_le_bytes(a)
    })
}

/// Read a NUL-terminated lossy string at `start`, capped at `max` bytes.
fn read_cstr(bytes: &[u8], start: usize, max: usize) -> String {
    let Some(rest) = bytes.get(start..) else {
        return String::new();
    };
    let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
    let end = end.min(max);
    String::from_utf8_lossy(&rest[..end]).into_owned()
}

/// Try to decode the whole file with one endianness. Returns `None` when the
/// header/table do not fit, or no state name can be recovered.
fn attempt(bytes: &[u8], be: bool) -> Option<StateMachine> {
    let count = read_u32(bytes, COUNT_OFF, be)? as usize;
    if count == 0 || count > MAX_STATES {
        return None;
    }
    let arr_end = OFFSET_ARR.checked_add(count.checked_mul(4)?)?;
    if arr_end > bytes.len() {
        return None;
    }
    let blob = arr_end;

    let name = read_cstr(bytes, NAME_OFF, NAME_BYTES);
    let mut states = Vec::with_capacity(count.min(8192));
    let mut prev = 0usize;
    for i in 0..count {
        let start = read_u32(bytes, OFFSET_ARR + i * 4, be)? as usize;
        // Starts must be non-decreasing into the blob.
        if i > 0 && start < prev {
            break;
        }
        prev = start;
        let Some(rest) = bytes.get(blob.checked_add(start)?..) else {
            break;
        };
        match rest.iter().position(|&c| c == 0) {
            Some(len) => {
                states.push(SmState {
                    name: String::from_utf8_lossy(&rest[..len]).into_owned(),
                    index: i as u32,
                });
            }
            None => {
                // Unterminated final entry: keep it, then stop.
                if !rest.is_empty() {
                    states.push(SmState {
                        name: String::from_utf8_lossy(rest).into_owned(),
                        index: i as u32,
                    });
                }
                break;
            }
        }
    }

    if states.is_empty() {
        return None;
    }
    Some(StateMachine { name, states })
}

/// Best-effort parse of a `ULSM` `.sm` file.
///
/// Returns `None` for a non-`ULSM` buffer (or one too truncated to hold the
/// header), otherwise the machine name plus as many state names as could be
/// read. Never panics. Both little-endian (PC/XB1) and big-endian (XB360)
/// integer layouts are tried; the candidate recovering the most states wins.
pub fn parse_sm(bytes: &[u8]) -> Option<StateMachine> {
    if bytes.len() < OFFSET_ARR || bytes.get(..4) != Some(&b"ULSM"[..]) {
        return None;
    }
    let le = attempt(bytes, false);
    let be = attempt(bytes, true);
    match (le, be) {
        (Some(a), Some(b)) => Some(if b.states.len() > a.states.len() {
            b
        } else {
            a
        }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn get_attr(e: &BytesStart, name: &[u8]) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

/// Collect the `path` of `<Animation name="$Include" …>` entries inside
/// `<AnimationList>` (including, leniently, unnamed `<Animation>` entries that
/// carry a `path`). Malformed XML ends the walk with whatever was collected;
/// never panics.
pub fn parse_chrparams(xml: &[u8]) -> Vec<String> {
    let xml = xml.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut out = Vec::new();
    let mut in_list = 0i32;

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };
        match ev {
            Event::Start(e) => {
                if e.name().as_ref() == b"AnimationList" {
                    in_list += 1;
                } else if in_list > 0 && e.name().as_ref() == b"Animation" {
                    let name = get_attr(&e, b"name");
                    let path = get_attr(&e, b"path");
                    let include = name
                        .as_deref()
                        .map(|n| n == "$Include" || n.is_empty())
                        .unwrap_or(true);
                    if include {
                        if let Some(p) = path.filter(|p| !p.is_empty()) {
                            out.push(p);
                        }
                    }
                }
            }
            Event::Empty(e) => {
                if in_list > 0 && e.name().as_ref() == b"Animation" {
                    let name = get_attr(&e, b"name");
                    let path = get_attr(&e, b"path");
                    let include = name
                        .as_deref()
                        .map(|n| n == "$Include" || n.is_empty())
                        .unwrap_or(true);
                    if include {
                        if let Some(p) = path.filter(|p| !p.is_empty()) {
                            out.push(p);
                        }
                    }
                }
            }
            Event::End(e) => {
                if e.name().as_ref() == b"AnimationList" {
                    in_list -= 1;
                    if in_list < 0 {
                        in_list = 0;
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

    fn real_player_sm() -> Option<Vec<u8>> {
        let candidates = [
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../extracted-game/gamedata/libs/statemachines/player/general.win.sm"
            ),
            "/media/workpc/1f162086-ac53-4c5e-96a2-d1dc72b0fa2d/home/battlestation/Desktop/sod-re/extracted-game/gamedata/libs/statemachines/player/general.win.sm",
        ];
        candidates.iter().find_map(|p| std::fs::read(p).ok())
    }

    #[test]
    fn parse_sm_rejects_junk() {
        assert!(parse_sm(b"").is_none());
        assert!(parse_sm(b"ULSM").is_none());
        assert!(parse_sm(b"this is not a state machine").is_none());
        let junk: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(37) % 256) as u8).collect();
        assert!(parse_sm(&junk).is_none());
    }

    #[test]
    fn parse_sm_never_panics() {
        let mut v = b"ULSM".to_vec();
        v.extend(std::iter::repeat(0xff).take(512));
        let _ = parse_sm(&v);
        let _ = parse_sm(&[0xff; 3]);
        let _ = parse_sm(&b"ULSM\0\0\0\0\0\0\0\0"[..]);
    }

    #[test]
    fn real_player_sm_if_present() {
        let Some(bytes) = real_player_sm() else {
            return; // data not present in this checkout: skip silently
        };
        let sm = parse_sm(&bytes).expect("real player sm should parse");
        assert_eq!(sm.name, "Player");
        assert!(!sm.states.is_empty());
        assert_eq!(sm.states.len(), 689);
        assert_eq!(sm.states[0].name, "GraphDriven.Dead.Idle");
        assert_eq!(sm.states[0].index, 0);
    }

    #[test]
    fn chrparams_inline_example() {
        let xml = br#"<?xml version="1.0" ?>
<Params>
	<AnimationList>
		<Animation name="$Include" path="characters/male/zombie/basic_zombie.chrparams"/>
	</AnimationList>
	<IK_Definition/>
</Params>"#;
        assert_eq!(
            parse_chrparams(xml),
            vec!["characters/male/zombie/basic_zombie.chrparams".to_string()]
        );
    }

    #[test]
    fn chrparams_lenient() {
        assert!(parse_chrparams(b"not xml at all <<<").is_empty());
        assert!(parse_chrparams(&[0xff, 0xfe, 0x00, 0x01]).is_empty());
        // Truncated document: keep what was already parsed.
        let m = br#"<AnimationList><Animation name="$Include" path="a/b.chrparams"><oops"#;
        assert_eq!(
            parse_chrparams(m),
            vec!["a/b.chrparams".to_string()]
        );
    }
}
