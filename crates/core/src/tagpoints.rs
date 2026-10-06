//! ULTP tag-point markup reader (story / RTS named positions).
//!
//! `level/tag_points_win.markup` is a little-endian CryEngine "markup" binary
//! that stores the named world positions used by missions and the RTS layer:
//! `RTSPoint_PlayerStart`, `RTSPoint_RangerStation`, every `RTSPoint_ta*`
//! tutorial point, plus NPC/room/scene tag points. This module recovers
//! `name -> world position`.
//!
//! # Recovered layout (`tag_points_win.markup`, 390,856 bytes)
//!
//! ```text
//! 0x00  char[4]  magic "ULTP"
//! 0x04  u32      version/flags = 0x06000000 (little-endian)
//! 0x08  u32      0
//! 0x0c  u32      n = 7088   // point count == name count
//! 0x10  u32      52         // not needed here
//! 0x14  u32      name-blob byte length = 146,251
//! 0x18  u32      763        // not needed here
//! 0x1c  u32      15         // not needed here
//! 0x20..0x80     zero padding (96 bytes)
//! 0x80  f32[3] * n          // world positions, stride 12, ends at 0x14CC0
//! ...            orientation/normal array + more tables (not needed here)
//! ...  u32[n-1]  cumulative name end offsets (relative to the name blob),
//!                strictly increasing; unused because names align by index
//! ...  char[]     name blob: n NUL-terminated ASCII identifiers
//! ```
//!
//! **The key result:** positions and names are *not* linked through the offset
//! table; they are simply index-aligned, i.e. `position[i]` belongs to
//! `name[i]` for the same `i`. This is validated by
//! `RTSPoint_RangerStation -> (3097.31, 3328.19, 230.71)`, which matches the
//! mission's own `Home_RangerStation Pos="3095.3545,3327.7961,230.70418"` to
//! within ~2 m, and by `RTSPoint_ta10Buddies_Locker -> (3099.94, 3330.05,
//! 230.74)`, next to `Facility_RangerStation_Indoor (3101.75, 3335.18, ...)`.
//!
//! # Blob location
//!
//! The header stores the blob *length* but not its offset. Rather than hard-code
//! an offset, [`TagPoints::parse`] scans for the unique byte range of the given
//! length that decodes as exactly `n` non-empty printable-ASCII, NUL-terminated
//! names (the range is preceded by a NUL and ends on a NUL). This is robust to
//! the intermediate tables changing size.
//!
//! Malformed or foreign input yields `None`; parsing never panics.

use std::collections::HashMap;

/// One `name -> world position` entry (Z-up, metres, CryEngine native).
#[derive(Debug, Clone, PartialEq)]
pub struct TagPoint {
    /// Identifier as it appears in `scenes.xml`/`missions.xml`
    /// (e.g. `RTSPoint_RangerStation`).
    pub name: String,
    /// World position in metres.
    pub pos: [f32; 3],
}

/// Parsed ULTP tag points, in file order.
#[derive(Debug, Clone, Default)]
pub struct TagPoints {
    points: Vec<TagPoint>,
    /// `name -> first index in points` (the file contains a few duplicate
    /// names; the first occurrence wins for [`TagPoints::get`]).
    index: HashMap<String, usize>,
}

/// Minimum file size: full header up to the position array.
const HEADER_LEN: usize = 0x80;
/// Expected little-endian version/flags word at offset `0x04`.
const VERSION: u32 = 0x0600_0000;

impl TagPoints {
    /// Parse a ULTP buffer. Returns `None` for a bad magic, a big-endian (or
    /// otherwise unexpected) version word, or a structurally invalid file.
    pub fn parse(bytes: &[u8]) -> Option<TagPoints> {
        if bytes.len() < HEADER_LEN || &bytes[0..4] != b"ULTP" {
            return None;
        }
        // Rejects big-endian input: a big-endian 0x06000000 and the documented
        // big-endian marker 0x01000000 both fail this equality.
        if read_u32(bytes, 4)? != VERSION {
            return None;
        }

        let n = read_u32(bytes, 0x0C)? as usize;
        let blob_len = read_u32(bytes, 0x14)? as usize;
        if n == 0 || blob_len == 0 {
            return None;
        }

        let pos_end = HEADER_LEN.checked_add(n.checked_mul(12)?)?;
        if pos_end > bytes.len() {
            return None;
        }

        // World positions: n × f32[3], stride 12, at 0x80.
        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            let o = HEADER_LEN + i * 12;
            let x = f32::from_le_bytes(bytes[o..o + 4].try_into().ok()?);
            let y = f32::from_le_bytes(bytes[o + 4..o + 8].try_into().ok()?);
            let z = f32::from_le_bytes(bytes[o + 8..o + 12].try_into().ok()?);
            positions.push([x, y, z]);
        }

        // Name blob: n NUL-terminated ASCII names, total span `blob_len`.
        let blob = locate_name_blob(bytes, HEADER_LEN, n, blob_len)?;
        let names = decode_names(bytes, blob, blob_len, n)?;

        let mut points = Vec::with_capacity(n);
        let mut index = HashMap::with_capacity(n);
        for (name, pos) in names.into_iter().zip(positions) {
            // Duplicate names: keep the first (matches `get` semantics).
            if !index.contains_key(&name) {
                index.insert(name.clone(), points.len());
            }
            points.push(TagPoint { name, pos });
        }

        Some(TagPoints { points, index })
    }

    /// World position for `name`, if present.
    pub fn get(&self, name: &str) -> Option<[f32; 3]> {
        self.index.get(name).map(|&i| self.points[i].pos)
    }

    /// All entries in file order.
    pub fn iter(&self) -> impl Iterator<Item = &TagPoint> {
        self.points.iter()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// `true` when there are no entries.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

fn read_u32(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4)
        .and_then(|s| s.try_into().ok())
        .map(u32::from_le_bytes)
}

/// Find the unique `[b, b + blob_len)` window (starting at or after `from`)
/// that holds exactly `n` non-empty printable-ASCII NUL-terminated names.
fn locate_name_blob(bytes: &[u8], from: usize, n: usize, blob_len: usize) -> Option<usize> {
    if blob_len > bytes.len() {
        return None;
    }
    let last_start = bytes.len() - blob_len;
    let mut b = from.max(1);
    while b <= last_start {
        if bytes[b] != 0
            && bytes[b - 1] == 0
            && bytes[b + blob_len - 1] == 0
            && count_names(bytes, b, blob_len) == Some(n)
        {
            return Some(b);
        }
        b += 1;
    }
    None
}

/// Number of NUL-terminated names in `[b, b+len)`, or `None` if the range holds
/// anything that is not a non-empty printable-ASCII name terminated by NUL.
fn count_names(bytes: &[u8], b: usize, len: usize) -> Option<usize> {
    let end = b.checked_add(len)?;
    if end > bytes.len() {
        return None;
    }
    let mut k = b;
    let mut count = 0usize;
    while k < end {
        let mut e = k;
        while e < end && bytes[e] != 0 {
            if !(0x20..=0x7e).contains(&bytes[e]) {
                return None;
            }
            e += 1;
        }
        if e == end || e == k {
            return None; // unterminated or empty name
        }
        count += 1;
        k = e + 1;
    }
    Some(count)
}

/// Decode exactly `n` names from a validated blob window.
fn decode_names(bytes: &[u8], b: usize, blob_len: usize, n: usize) -> Option<Vec<String>> {
    let end = b + blob_len;
    let mut names = Vec::with_capacity(n);
    let mut k = b;
    while k < end {
        let e = bytes[k..end].iter().position(|&c| c == 0)? + k;
        let s = std::str::from_utf8(&bytes[k..e]).ok()?;
        names.push(s.to_string());
        k = e + 1;
    }
    if names.len() == n {
        Some(names)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn real_file() -> Option<Vec<u8>> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/level/tag_points_win.markup");
        std::fs::read(p).ok()
    }

    #[test]
    fn rejects_non_ultp_and_big_endian() {
        assert!(TagPoints::parse(b"").is_none());
        assert!(TagPoints::parse(b"NOPE").is_none());
        // Wrong/absent magic.
        let mut junk = vec![0u8; 0x80];
        junk[0..4].copy_from_slice(b"NOPE");
        assert!(TagPoints::parse(&junk).is_none());
        // Big-endian version word: ULTP + 0x06000000 written big-endian.
        let mut be = vec![0u8; 0x80];
        be[0..4].copy_from_slice(b"ULTP");
        be[4..8].copy_from_slice(&[0x06, 0x00, 0x00, 0x00]);
        assert!(TagPoints::parse(&be).is_none());
        // Documented big-endian marker 0x01000000 (LE bytes 00 00 00 01).
        let mut marker = vec![0u8; 0x80];
        marker[0..4].copy_from_slice(b"ULTP");
        marker[4..8].copy_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        assert!(TagPoints::parse(&marker).is_none());
    }

    #[test]
    fn parses_real_file_and_links_names_to_positions() {
        let Some(bytes) = real_file() else { return }; // skip if absent
        let tp = TagPoints::parse(&bytes).expect("real ULTP should parse");

        assert_eq!(tp.len(), 7088, "file stores 7088 index-aligned points");
        assert!(!tp.is_empty());

        // Every name in the blob is retrievable. A few scene/room names repeat
        // (895 duplicated of 3149 distinct); `get` returns the first
        // occurrence, so only unique names must round-trip exactly.
        assert_eq!(tp.iter().count(), tp.len());
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for p in tp.iter() {
            assert!(
                tp.get(&p.name).is_some(),
                "name {} must be retrievable",
                p.name
            );
            assert!(p.pos.iter().all(|c| c.is_finite()));
            *counts.entry(p.name.as_str()).or_insert(0) += 1;
        }
        for p in tp.iter() {
            if counts[p.name.as_str()] == 1 {
                assert_eq!(tp.get(&p.name), Some(p.pos), "unique name round-trip");
            }
        }

        let rts = tp.iter().filter(|p| p.name.starts_with("RTSPoint_")).count();
        assert_eq!(rts, 473, "473 RTSPoint_* names");

        // Cross-file ground truth: the mission places Home_RangerStation at
        // (3095.35, 3327.80, 230.70); the tag point sits metres away.
        let rs = tp.get("RTSPoint_RangerStation").expect("RangerStation");
        assert!(rs.iter().all(|c| c.is_finite()));
        assert!(
            (3090.0..3105.0).contains(&rs[0])
                && (3320.0..3336.0).contains(&rs[1])
                && (225.0..236.0).contains(&rs[2]),
            "RangerStation {rs:?} should be at the Ranger Station home"
        );

        // Player start must be a sane ground position on the Class3 map.
        let ps = tp.get("RTSPoint_PlayerStart").expect("PlayerStart");
        assert!(ps.iter().all(|c| c.is_finite()));
        assert!(
            (0.0..=4096.0).contains(&ps[0])
                && (0.0..=4096.0).contains(&ps[1])
                && (0.0..=400.0).contains(&ps[2]),
            "PlayerStart {ps:?} outside world bounds"
        );
    }

    #[test]
    fn unknown_name_is_none() {
        let Some(bytes) = real_file() else { return };
        let tp = TagPoints::parse(&bytes).unwrap();
        assert!(tp.get("RTSPoint_DefinitelyNotAName").is_none());
    }
}
