//! CryEngine animation database (`.dba`) parser.
//!
//! A `.dba` is a `"CryTek\0"` chunk container — the same `0x18`-byte header
//! plus `record_size`-byte record table used by `.cgf`/`.caf`/`.img` (see
//! [`crate::asset::parse_cry_chunk_file`]) — but the shipped State of Decay
//! files carry exactly **one** chunk:
//!
//! ```text
//! container header
//!   0x00  char[7] "CryTek\0"
//!   0x08  u32     file_type    = 0xFFFF0000        (or 0xFFFF0001)
//!   0x0C  u32     file_version = 0x745             (0x744 also accepted)
//!   0x10  u32     record_size  = 20
//!   0x14  u32     record_count = 1
//! record[0] @ 0x18 (record_size bytes)
//!   0x18  u32 chunk_type    = 0xCCCC000D
//!   0x1C  u32 chunk_version = 0x905
//!   0x20  u32 file_offset   -> DBA blob
//!   0x24  u32 id            = 1
//!   0x28  u32 size          -> DBA blob length in bytes
//! ```
//!
//! The `0xCCCC000D` / version `0x905` chunk is the `ReadController905` DBA
//! body (CryEngine's `CAnimationDatabase`). The blob it points at starts with
//! a 16-byte chunk preamble mirroring `type|version|file_offset|id`, then its
//! own header, then the packed per-clip records:
//!
//! ```text
//! DBA blob @ file_offset
//!   +0x00  chunk preamble (type, version, file_offset, id)     [16 bytes]
//!   +0x10  u32 num_a       (e.g. 2246)   fixed-array count
//!   +0x14  u32 num_b       (e.g. 22014)  fixed-array count
//!   +0x18  u32 num_c       (e.g. 9145)   fixed-array count
//!   +0x1C  u32 num_files   packed clip count (== entry count)
//!   +0x20  ... variable-size fixed arrays (bone/controller index tables) ...
//!   then `num_files` contiguous clip records:
//!          u16  name_len
//!          u8   name[name_len]    virtual `.caf` path (length-prefixed,
//!                                 *not* NUL-terminated)
//!          u32  header[0x21]      132-byte per-clip animation header
//!          u16  reserved
//!          u16  sub_count
//!          u8   sub[sub_count * 20]   20-byte sub-entry records
//!   trailing footer (14 bytes)
//! ```
//!
//! All integers little-endian. The clip records are walked structurally using
//! `num_files`; the walk lands a few bytes short of the blob end (the footer).
//! The on-disk format stores **no** per-clip hash, so [`DbaEntry::hash`] is
//! derived: a 64-bit FNV-1a hash of the clip name (retained so callers get a
//! stable key). See `notes/dba-format.md`.

/// File signature: `"CryTek\0"`.
pub const CRY_DBA_SIG: &[u8; 7] = b"CryTek\0";
/// Offset of the record table.
pub const TABLE_OFFSET: usize = 0x18;
/// Fixed 132-byte (`0x21` x u32) per-clip animation header.
pub const ENTRY_HEADER: usize = 0x21 * 4;
/// Size of one packed sub-entry record (5 x u32).
pub const SUB_ENTRY: usize = 20;
/// Maximum accepted length-prefixed name (mirrors the loader's 0x400 buffer).
pub const MAX_NAME: usize = 0x400;
/// The single DBA chunk type.
pub const DBA_CHUNK_TYPE: u32 = 0xCCCC_000D;
/// The DBA chunk data version handled by the `ReadController905` loader.
pub const DBA_CHUNK_VERSION: u32 = 0x905;

fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// 64-bit FNV-1a hash of a clip name.
fn fnv1a(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &c in name.as_bytes() {
        h ^= c as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// True when `seg` looks like a virtual asset path (printable, has a `/` and
/// an extension dot).
fn is_path(seg: &[u8]) -> bool {
    seg.len() >= 4
        && seg.iter().all(|&c| (0x20..=0x7e).contains(&c))
        && seg.contains(&b'/')
        && seg.contains(&b'.')
}

/// One clip record of a [`Dba`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbaEntry {
    /// Virtual `.caf` path (e.g. `animations/characters/.../cover_aim_idle.caf`).
    pub name: String,
    /// Stable 64-bit FNV-1a hash of `name` (the file stores no per-clip hash).
    pub hash: u64,
    /// Absolute file offset of the clip record (the `u16` name length); the
    /// 132-byte animation header follows the name.
    pub offset: u32,
    /// 1-based position of the clip in the DBA.
    pub id: u32,
}

/// Parsed `CryTek` `.dba` animation database.
#[derive(Debug, Clone)]
pub struct Dba {
    /// Header `0x08` (observed `0xFFFF0000`/`0xFFFF0001`).
    pub file_type: u32,
    /// Header `0x0C` (observed `0x745`).
    pub version: u32,
    /// Header `0x10`, the raw record-size field (observed 20).
    pub record_size: u32,
    /// Header `0x14`, the record count (observed 1).
    pub record_count: u32,
    /// Absolute offset of the `0xCCCC000D` chunk blob.
    pub chunk_offset: u32,
    /// Byte length of the `0xCCCC000D` chunk blob.
    pub chunk_size: u32,
    /// `num_files` from the DBA header (`+0x1C` of the blob).
    pub num_files: u32,
    entries: Vec<DbaEntry>,
}

/// Walk `num` packed clip records starting at `q`, bounded by `end`.
fn walk(bytes: &[u8], mut q: usize, end: usize, num: u32) -> Option<Vec<DbaEntry>> {
    let mut out = Vec::with_capacity(num.min(0x10000) as usize);
    for i in 0..num {
        let rec = q;
        if q.checked_add(2)? > end {
            return None;
        }
        let n = rd_u16(bytes, q) as usize;
        q += 2;
        if n < 4 || n > MAX_NAME || q.checked_add(n)? > end {
            return None;
        }
        let seg = &bytes[q..q + n];
        if !is_path(seg) {
            return None;
        }
        let name = std::str::from_utf8(seg).ok()?.to_string();
        q += n;
        // 132-byte animation header, then u16 reserved + u16 sub_count.
        if q.checked_add(ENTRY_HEADER)?.checked_add(4)? > end {
            return None;
        }
        q += ENTRY_HEADER;
        let sub = rd_u16(bytes, q + 2) as usize;
        q = q.checked_add(4)?.checked_add(sub.checked_mul(SUB_ENTRY)?)?;
        if q > end {
            return None;
        }
        out.push(DbaEntry {
            hash: fnv1a(&name),
            name,
            offset: rec as u32,
            id: 1 + i,
        });
    }
    Some(out)
}

/// Locate the first clip record by finding the first length-prefixed path that
/// begins with `animations/`.
fn find_start(bytes: &[u8], start: usize, end: usize) -> Option<usize> {
    let mut q = start;
    while q.checked_add(2)? <= end {
        let n = rd_u16(bytes, q) as usize;
        if (4..=MAX_NAME).contains(&n) && q + 2 + n <= end {
            let seg = &bytes[q + 2..q + 2 + n];
            if seg.starts_with(b"animations/") && is_path(seg) {
                return Some(q);
            }
        }
        q += 1;
    }
    None
}

impl Dba {
    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the container has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries, in file order.
    pub fn entries(&self) -> &[DbaEntry] {
        &self.entries
    }

    /// First entry whose name contains `needle` (ASCII, case-insensitive).
    pub fn find(&self, needle: &str) -> Option<&DbaEntry> {
        let n = needle.to_ascii_lowercase();
        self.entries
            .iter()
            .find(|e| e.name.to_ascii_lowercase().contains(&n))
    }

    /// Iterator over all entry names.
    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.entries.iter().map(|e| e.name.as_str())
    }
}

/// Parse a `CryTek` `.dba`.
///
/// Returns `None` for a non-`CryTek` signature, a truncated/garbage table, a
/// missing `0xCCCC000D` DBA chunk, an impossible record shape, or a truncated
/// clip stream. Never panics and never reads out of bounds on arbitrary input.
pub fn parse(bytes: &[u8]) -> Option<Dba> {
    if bytes.len() < TABLE_OFFSET + 20 || &bytes[0..7] != CRY_DBA_SIG {
        return None;
    }
    let file_type = rd_u32(bytes, 0x08);
    let version = rd_u32(bytes, 0x0C);
    let record_size = rd_u32(bytes, 0x10);
    let record_count = rd_u32(bytes, 0x14);
    let stride = if (16..=64).contains(&record_size) {
        record_size as usize
    } else {
        20
    };
    let table_end = TABLE_OFFSET.checked_add((record_count as usize).checked_mul(stride)?)?;
    if record_count == 0 || table_end > bytes.len() {
        return None;
    }

    // Find the single 0xCCCC000D DBA chunk.
    let mut chunk: Option<(u32, u32)> = None;
    for i in 0..record_count as usize {
        let o = TABLE_OFFSET + i * stride;
        if o + stride > bytes.len() {
            break;
        }
        if rd_u32(bytes, o) == DBA_CHUNK_TYPE {
            let off = rd_u32(bytes, o + 0x08);
            let size = if stride >= 20 {
                rd_u32(bytes, o + 0x10)
            } else if i + 1 < record_count as usize {
                rd_u32(bytes, TABLE_OFFSET + (i + 1) * stride + 0x08).saturating_sub(off)
            } else {
                (bytes.len() as u32).saturating_sub(off)
            };
            chunk = Some((off, size));
            break;
        }
    }
    let (chunk_offset, chunk_size) = chunk?;
    let cstart = chunk_offset as usize;
    let cend = cstart.checked_add(chunk_size as usize)?;
    if cstart.checked_add(0x20)? > bytes.len() || cend > bytes.len() {
        return None;
    }

    let num_files = rd_u32(bytes, cstart + 0x1C);
    if num_files == 0 {
        return None;
    }
    let start = find_start(bytes, cstart, cend)?;
    let entries = walk(bytes, start, cend, num_files)?;

    Some(Dba {
        file_type,
        version,
        record_size,
        record_count,
        chunk_offset,
        chunk_size,
        num_files,
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn dba_file(name: &str) -> Option<Vec<u8>> {
        let p: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/animations/animations/characters")
            .join(name);
        std::fs::read(p).ok()
    }

    #[test]
    fn parses_real_hero_male() {
        let Some(bytes) = dba_file("hero_male/hero_male.dba") else {
            return; // real data absent: skip silently
        };
        let dba = parse(&bytes).expect("hero_male.dba should parse");
        assert!(dba.len() > 500, "expected >500 entries, got {}", dba.len());
        assert_eq!(dba.len(), dba.entries().len());
        assert_eq!(dba.num_files as usize, dba.len());
        assert_eq!(dba.version, 0x745);
        assert_eq!(dba.record_size, 20);
        assert_eq!(dba.record_count, 1);
        assert_eq!(dba.entries()[0].id, 1);
        assert!(dba.entries().iter().all(|e| !e.name.is_empty()));
        assert!(dba.names().all(|n| n.ends_with(".caf")));
        assert_eq!(dba.names().count(), dba.len());
        assert!(
            dba.entries()
                .iter()
                .all(|e| e.offset as usize + ENTRY_HEADER <= bytes.len()),
            "every record offset is in bounds"
        );

        let hit = dba.find("cover_aim_idle").expect("cover_aim_idle present");
        assert!(hit.name.contains("cover_aim_idle"));
        assert!(dba.find("COVER_AIM_IDLE").is_some(), "find is case-insensitive");
        assert!(dba.find("no_such_clip_xyz_123").is_none());
    }

    #[test]
    fn reports_locomotion_entries() {
        let Some(bytes) = dba_file("hero_male/hero_male.dba") else {
            return;
        };
        let dba = parse(&bytes).unwrap();
        for needle in ["hero_run_forward", "hero_walk_forward", "hero_idle"] {
            let hit = dba.find(needle);
            eprintln!(
                "dba locomotion {needle}: {}",
                hit.map(|e| format!("present id={} name={}", e.id, e.name))
                    .unwrap_or_else(|| "absent".to_string())
            );
            assert!(hit.is_some(), "{needle} should be present");
        }
        // Exact base clips recovered in hero_male.dba.
        assert!(dba
            .entries()
            .iter()
            .any(|e| e.name.ends_with("/test/hero_run_forward.caf")));
        assert!(dba
            .entries()
            .iter()
            .any(|e| e.name.ends_with("/test/hero_walk_forward.caf")));
        assert!(dba
            .entries()
            .iter()
            .any(|e| e.name.ends_with("/test/hero_idle.caf")));
    }

    #[test]
    fn rejects_non_crytek_and_truncated() {
        assert!(parse(b"").is_none());
        assert!(parse(b"not a crytek dba file").is_none());
        assert!(parse(&[0u8; 0x18]).is_none());
        assert!(parse(&[0u8; 0x2c]).is_none());

        // Correct signature but no usable record / chunk.
        let mut b = vec![0u8; 0x100];
        b[0..7].copy_from_slice(CRY_DBA_SIG);
        b[0x08..0x0C].copy_from_slice(&0xFFFF_0000u32.to_le_bytes());
        b[0x0C..0x10].copy_from_slice(&0x745u32.to_le_bytes());
        b[0x10..0x14].copy_from_slice(&20u32.to_le_bytes());
        b[0x14..0x18].copy_from_slice(&1u32.to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&DBA_CHUNK_TYPE.to_le_bytes());
        b[0x1C..0x20].copy_from_slice(&DBA_CHUNK_VERSION.to_le_bytes());
        b[0x20..0x24].copy_from_slice(&0x2Cu32.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        b[0x28..0x2C].copy_from_slice(&0xD4u32.to_le_bytes());
        // num_files == 0 -> None.
        assert!(parse(&b).is_none());
        // record_count impossible -> None.
        b[0x14..0x18].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse(&b).is_none());

        // Truncated real file must not panic and must be rejected.
        if let Some(bytes) = dba_file("hero_male/hero_male.dba") {
            assert!(parse(&bytes[..64]).is_none());
            assert!(parse(&bytes[..TABLE_OFFSET]).is_none());
            assert!(parse(&bytes[..bytes.len() / 2]).is_none());
        }
    }
}
