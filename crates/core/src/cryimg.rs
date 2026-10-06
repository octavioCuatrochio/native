//! CryEngine `CryTek` `.img` animation container (packed per-clip blobs).
//!
//! A `.img` is the **same `"CryTek\0"` chunk container** as `.cgf`/`.caf`
//! (see [`crate::asset::parse_cry_chunk_file`]): an `0x18`-byte header then a
//! `count`-entry table of 20-byte records. Unlike a loose `.caf`, the chunk
//! payloads are *not* Controller streams: every entry is a compact animation
//! header chunk (`0xAAFC0007` in `animations.img`, `0xAAFC0008` in
//! `directionalblends.img`) whose payload embeds the source `.caf` virtual
//! path at `blob_offset + 0x14`.
//!
//! Layout recovered from the shipped files (both validated end-to-end; the
//! 20-byte table + contiguous offsets/sizes + embedded paths all check out):
//!
//! ```text
//! header
//!   0x00  char[7]  "CryTek\0"            (8th byte is zero padding)
//!   0x08  u32      file_type    = 0xFFFF0000   (meaning uncertain)
//!   0x0C  u32      file_version = 0x745
//!   0x10  u32      record_size  = 20           (bytes per table record)
//!   0x14  u32      record_count = 3430         (animations.img)
//!   0x18  record[record_count], record_size bytes each:
//!           +0x00 u32 chunk_type     (0xAAFC0007 / 0xAAFC0008)
//!           +0x04 u32 chunk_version  (0x971 / 0x970)
//!           +0x08 u32 file_offset    absolute blob offset
//!           +0x0C u32 id             1-based, unique
//!           +0x10 u32 size           blob length in bytes
//! blob at file_offset (its first 16 bytes mirror type/version/offset/id):
//!           +0x10 u32 flags/param    (varies; meaning uncertain)
//!           +0x14 char[] path        NUL-terminated in a fixed 0x100-byte field
//!           ...  compact animation header (bone defaults, -10000.0 sentinels)
//! ```
//!
//! All integers little-endian. See `notes/cryimg-format.md`.

/// File signature: `"CryTek\0"`.
pub const CRY_IMG_SIG: &[u8; 7] = b"CryTek\0";
/// Offset of the record table.
pub const TABLE_OFFSET: usize = 0x18;
/// File version at/above which 20-byte records are used (older: 16).
pub const FORMAT_VERSION: u32 = 0x745;
/// Offset inside a blob where the embedded `.caf` path begins.
pub const PATH_OFFSET: usize = 0x14;
/// Size of the fixed path field inside a blob.
pub const PATH_FIELD: usize = 0x100;

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

/// One record of a [`CryImg`] table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImgEntry {
    /// 1-based id from the record (used by [`CryImg::extract`]).
    pub id: u32,
    /// Absolute byte offset of the blob in the container.
    pub offset: u32,
    /// Blob length in bytes.
    pub size: u32,
    /// The record's leading 8 bytes. In practice this is **not** a content
    /// hash: it is `chunk_type | (chunk_version << 32)` and is constant per
    /// file (`0x00000971_aafc0007` for `animations.img`).
    pub hash: u64,
    /// Source `.caf` path embedded in the blob, if a printable one was found.
    pub path: Option<String>,
}

/// Parsed `CryTek` `.img` container.
#[derive(Debug, Clone)]
pub struct CryImg {
    /// Header `0x08` (observed `0xFFFF0000`; meaning uncertain).
    pub file_type: u32,
    /// Header `0x0C` (observed `0x745`).
    pub version: u32,
    /// Header `0x10`, the raw record size field (observed 20).
    pub record_size: u32,
    /// Header `0x14`, the record count.
    pub record_count: u32,
    entries: Vec<ImgEntry>,
}

/// Read the NUL-terminated ASCII path embedded in a blob, if any.
fn read_path(bytes: &[u8], offset: u32, size: u32) -> Option<String> {
    let base = offset as usize;
    let start = base.checked_add(PATH_OFFSET)?;
    let end = base
        .checked_add(size as usize)?
        .min(bytes.len())
        .min(start.checked_add(PATH_FIELD)?);
    if start >= end || end > bytes.len() {
        return None;
    }
    let field = &bytes[start..end];
    let n = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    let name = &field[..n];
    if name.is_empty() || !name.iter().all(|&b| (0x20..=0x7e).contains(&b)) {
        return None;
    }
    std::str::from_utf8(name).ok().map(str::to_string)
}

/// Bounds-checked `bytes[offset .. offset + size]`.
pub fn extract_at<'a>(bytes: &'a [u8], offset: u32, size: u32) -> Option<&'a [u8]> {
    let start = offset as usize;
    let end = start.checked_add(size as usize)?;
    if end <= bytes.len() {
        Some(&bytes[start..end])
    } else {
        None
    }
}

impl CryImg {
    /// Number of table entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the container has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All table entries, in file order.
    pub fn entries(&self) -> &[ImgEntry] {
        &self.entries
    }

    /// Look up an entry by its 1-based `id`.
    pub fn find_id(&self, id: u32) -> Option<&ImgEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// First entry whose embedded path contains `needle` (ASCII,
    /// case-insensitive).
    pub fn find_path(&self, needle: &str) -> Option<&ImgEntry> {
        let n = needle.to_ascii_lowercase();
        self.entries.iter().find(|e| {
            e.path
                .as_deref()
                .map(|p| p.to_ascii_lowercase().contains(&n))
                .unwrap_or(false)
        })
    }

    /// Raw blob for entry `id`, bounds-checked against `bytes`.
    pub fn extract<'a>(&self, bytes: &'a [u8], id: u32) -> Option<&'a [u8]> {
        let e = self.find_id(id)?;
        extract_at(bytes, e.offset, e.size)
    }

    /// Associated-function form of [`extract_at`].
    pub fn extract_at<'a>(bytes: &'a [u8], offset: u32, size: u32) -> Option<&'a [u8]> {
        extract_at(bytes, offset, size)
    }
}

/// Parse a `CryTek` `.img`.
///
/// Returns `None` for a non-`CryTek` signature, a header/table that does not
/// fit in `bytes`, a zero/overflowing record count, or any other impossible
/// shape. Never panics and never reads out of bounds on arbitrary input.
pub fn parse(bytes: &[u8]) -> Option<CryImg> {
    if bytes.len() < TABLE_OFFSET || &bytes[0..7] != CRY_IMG_SIG {
        return None;
    }
    let file_type = rd_u32(bytes, 0x08);
    let version = rd_u32(bytes, 0x0C);
    let record_size = rd_u32(bytes, 0x10);
    let record_count = rd_u32(bytes, 0x14);

    // `0x10` is the record size in every shipped `.img`/`.caf` we checked; fall
    // back to the version-derived stride if it is implausible.
    let stride = if (16..=64).contains(&record_size) {
        record_size as usize
    } else if version >= FORMAT_VERSION {
        20
    } else {
        16
    };

    let table_end = TABLE_OFFSET.checked_add((record_count as usize).checked_mul(stride)?)?;
    if record_count == 0 || table_end > bytes.len() {
        return None;
    }

    let mut entries = Vec::with_capacity(record_count as usize);
    for i in 0..record_count as usize {
        let o = TABLE_OFFSET + i * stride;
        let hash = rd_u64(bytes, o);
        let offset = rd_u32(bytes, o + 0x08);
        let id = rd_u32(bytes, o + 0x0C);
        // Legacy 16-byte records carry no explicit size; derive it below.
        let size = if stride >= 20 {
            rd_u32(bytes, o + 0x10)
        } else {
            0
        };
        entries.push(ImgEntry {
            id,
            offset,
            size,
            hash,
            path: None,
        });
    }
    if stride < 20 {
        for i in 0..entries.len() {
            let next = if i + 1 < entries.len() {
                entries[i + 1].offset
            } else {
                bytes.len() as u32
            };
            entries[i].size = next.saturating_sub(entries[i].offset);
        }
    }

    for e in &mut entries {
        e.path = read_path(bytes, e.offset, e.size);
    }

    Some(CryImg {
        file_type,
        version,
        record_size,
        record_count,
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/animations/animations")
    }
    fn load(name: &str) -> Option<Vec<u8>> {
        std::fs::read(img_dir().join(name)).ok()
    }

    #[test]
    fn parses_real_animations_img() {
        let Some(bytes) = load("animations.img") else {
            return; // real data absent: skip silently
        };
        let img = parse(&bytes).expect("animations.img should parse");
        assert!(!img.entries().is_empty(), "entry list is non-empty");
        assert_eq!(img.len(), img.entries().len());
        assert_eq!(img.version, 0x745);
        assert_eq!(img.record_size, 20);
        assert_eq!(img.record_count as usize, img.len());

        let e0 = &img.entries()[0];
        let end = e0.offset as usize + e0.size as usize;
        assert!(end <= bytes.len(), "first entry range in bounds");
        assert!(e0.size > 0);

        let blob = img.extract(&bytes, e0.id).expect("extract by id");
        assert!(!blob.is_empty());
        assert_eq!(blob.len(), e0.size as usize);
        assert!(
            e0.path.as_deref().map(|p| p.ends_with(".caf")).unwrap_or(false),
            "first entry has a .caf path"
        );
        assert_eq!(CryImg::extract_at(&bytes, e0.offset, e0.size), Some(blob));

        // If the blob happens to be a chunk file, Caf::parse must succeed.
        if let Some(caf) = crate::anim::Caf::parse(blob) {
            assert!(caf.secs_per_tick > 0.0);
        }

        // Cross-check against the generic CryTek chunk reader.
        if let Some(cf) = crate::asset::parse_cry_chunk_file(&bytes) {
            assert_eq!(cf.num_chunks, img.record_count);
            assert_eq!(cf.chunks.len(), img.len());
        }
    }

    #[test]
    fn parses_real_directionalblends_img() {
        let Some(bytes) = load("directionalblends.img") else {
            return;
        };
        let img = parse(&bytes).expect("directionalblends.img should parse");
        assert!(img.len() >= 10);
        assert!(img
            .entries()
            .iter()
            .all(|e| e.offset as usize + e.size as usize <= bytes.len()));
        // This container has variable blob sizes (8404 / 2176 / 5116).
        let first = img.entries()[0].size;
        assert!(img.entries().iter().any(|e| e.size != first));
    }

    #[test]
    fn find_path_finds_locomotion_clips() {
        let Some(bytes) = load("animations.img") else {
            return;
        };
        let img = parse(&bytes).unwrap();
        for needle in ["walk", "idle", "run"] {
            let e = img.find_path(needle).expect("locomotion path present");
            let p = e.path.as_deref().unwrap().to_ascii_lowercase();
            assert!(p.contains(needle), "{p} should contain {needle}");
        }
        assert!(img.find_path("WALK").is_some(), "find_path is case-insensitive");
        assert!(img.find_path("no_such_clip_xyz").is_none());
        assert!(img.find_id(img.entries()[0].id).is_some());
    }

    #[test]
    fn rejects_non_crytek_and_truncated() {
        assert!(parse(b"").is_none());
        assert!(parse(b"not a crytek image").is_none());
        assert!(parse(&[0u8; 0x18]).is_none());

        // Correct signature but an impossible count -> table runs past input.
        let mut b = vec![0u8; TABLE_OFFSET];
        b[0..7].copy_from_slice(CRY_IMG_SIG);
        b[0x0C..0x10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        b[0x10..0x14].copy_from_slice(&20u32.to_le_bytes());
        b[0x14..0x18].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse(&b).is_none());

        // Truncated real file (below the table end) must not panic.
        if let Some(bytes) = load("animations.img") {
            assert!(parse(&bytes[..64]).is_none());
            assert!(parse(&bytes[..TABLE_OFFSET]).is_none());
        }
    }

    #[test]
    fn extract_bounds_and_free_fn() {
        let bytes = [1u8, 2, 3, 4, 5];
        assert_eq!(extract_at(&bytes, 1, 3), Some(&bytes[1..4]));
        assert_eq!(CryImg::extract_at(&bytes, 1, 3), Some(&bytes[1..4]));
        assert!(extract_at(&bytes, 4, 2).is_none());
        assert!(extract_at(&bytes, u32::MAX, 1).is_none());
        // Unparsed entry cannot be extracted.
        let img = CryImg {
            file_type: 0,
            version: FORMAT_VERSION,
            record_size: 20,
            record_count: 1,
            entries: vec![],
        };
        assert!(img.extract(&bytes, 1).is_none());
    }
}
