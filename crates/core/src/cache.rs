//! On-disk cache for decoded textures (avoids re-decoding ~400 DDS per launch).
//!
//! # On-disk format
//!
//! Each decoded texture is stored as a single `.rgba` file named after a
//! 64-bit FNV-1a hash of the caller's cache key (hex, 16 chars):
//!
//! ```text
//! <cache-dir>/<fnv1a64(key):016x>.rgba
//! ```
//!
//! The file is a tiny little-endian header followed by the raw payload:
//!
//! ```text
//! offset  size  field
//! 0       4     magic  = b"TXC1"
//! 4       4     u32    width
//! 8       4     u32    height
//! 12      w*h*4 RGBA8  row-major, top-left origin (asset::DdsImageRgba::rgba)
//! ```
//!
//! A cache hit must satisfy `file_len == 12 + w*h*4` and `w,h > 0`; anything
//! else (bad magic, truncation, overflow, trailing bytes) is treated as a miss
//! and the texture is decoded again. Writes go to a per-process temp file in the
//! same directory followed by a rename, so a crash cannot leave a half-written
//! entry. All IO is best-effort: if the cache directory is missing/unwritable
//! (or the disk is full) decoding still succeeds in memory.

use crate::asset::{self, DdsImageRgba};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Cache-file magic (`T`e`X`ture `C`ache v`1`).
const MAGIC: &[u8; 4] = b"TXC1";
/// Fixed on-disk header size: magic + width + height.
const HEADER: usize = 12;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv1a64(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Hash a key (its bytes, then its length) to the file stem.
fn key_hash(key: &[u8]) -> u64 {
    let h = fnv1a64(key, FNV_OFFSET);
    fnv1a64(&(key.len() as u64).to_le_bytes(), h)
}

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Decode a cache file, returning `None` for any malformed/truncated content.
fn read_cached(path: &Path) -> Option<DdsImageRgba> {
    let data = fs::read(path).ok()?;
    if data.len() < HEADER || &data[0..4] != MAGIC {
        return None;
    }
    let width = u32le(&data, 4);
    let height = u32le(&data, 8);
    if width == 0 || height == 0 {
        return None;
    }
    let expected = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)?
        .checked_add(HEADER)?;
    if data.len() != expected {
        return None;
    }
    Some(DdsImageRgba { width, height, rgba: data[HEADER..].to_vec() })
}

/// Serialise an image into the documented header+payload representation.
fn encode_cached(img: &DdsImageRgba) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + img.rgba.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&img.width.to_le_bytes());
    out.extend_from_slice(&img.height.to_le_bytes());
    out.extend_from_slice(&img.rgba);
    out
}

/// Write atomically: temp file in the same directory, then rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".{}.tmp", std::process::id()));
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, bytes)?;
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Disk cache mapping arbitrary byte keys to decoded RGBA8 textures.
pub struct TextureCache {
    dir: PathBuf,
}

impl TextureCache {
    /// Create a handle for `dir`. The directory is created lazily on first write.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        TextureCache { dir: dir.into() }
    }

    /// The cache directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_for(&self, key: &[u8]) -> PathBuf {
        self.dir.join(format!("{:016x}.rgba", key_hash(key)))
    }

    /// Return the decoded texture for `key`, loading it from disk when a valid
    /// entry exists and otherwise decoding `dds` via [`asset::decode_dds_rgba`].
    ///
    /// On a miss the result is written back to disk best-effort; cache/IO errors
    /// never prevent returning the freshly decoded image.
    pub fn get_or_decode(&self, key: &[u8], dds: &[u8]) -> Option<DdsImageRgba> {
        let path = self.path_for(key);
        if let Some(img) = read_cached(&path) {
            return Some(img);
        }
        let img = asset::decode_dds_rgba(dds)?;
        if fs::create_dir_all(&self.dir).is_ok() {
            let _ = write_atomic(&path, &encode_cached(&img));
        }
        Some(img)
    }

    /// Remove every cached entry. Missing directories are not an error.
    pub fn clear(&self) -> io::Result<()> {
        match fs::read_dir(&self.dir) {
            Ok(rd) => {
                for entry in rd {
                    let entry = entry?;
                    let p = entry.path();
                    let _ = if p.is_dir() {
                        fs::remove_dir_all(&p)
                    } else {
                        fs::remove_file(&p)
                    };
                }
                Ok(())
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Total bytes occupied by regular files in the cache directory.
    pub fn size_bytes(&self) -> u64 {
        let mut total = 0u64;
        if let Ok(rd) = fs::read_dir(&self.dir) {
            for entry in rd.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        total = total.saturating_add(meta.len());
                    }
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A fresh, unique cache directory; caller must `remove_dir_all` it.
    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "sod_cache_{}_{}_{}",
            tag,
            std::process::id(),
            nanos
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    /// 128-byte DDS header (mirrors the helper in `asset::tests`).
    fn dds_header(w: u32, h: u32, fourcc: &[u8; 4]) -> Vec<u8> {
        let mut b = vec![0u8; 128];
        b[0..4].copy_from_slice(b"DDS ");
        b[4..8].copy_from_slice(&124u32.to_le_bytes());
        b[8..12].copy_from_slice(&0x0002_1007u32.to_le_bytes());
        b[12..16].copy_from_slice(&h.to_le_bytes());
        b[16..20].copy_from_slice(&w.to_le_bytes());
        b[28..32].copy_from_slice(&1u32.to_le_bytes());
        b[76..80].copy_from_slice(&32u32.to_le_bytes());
        b[80..84].copy_from_slice(&0x0000_0004u32.to_le_bytes()); // DDPF_FOURCC
        b[84..88].copy_from_slice(fourcc);
        b
    }

    /// A valid 4x4 DXT1 texture (red/blue endpoints, four indices per row).
    fn tiny_dxt1() -> Vec<u8> {
        let mut b = dds_header(4, 4, b"DXT1");
        b.extend_from_slice(&0xF800u16.to_le_bytes()); // red
        b.extend_from_slice(&0x001Fu16.to_le_bytes()); // blue
        b.extend_from_slice(&[0xE4, 0, 0, 0]);
        b
    }

    #[test]
    fn first_call_decodes_and_writes_file() {
        let dir = temp_dir("write");
        let cache = TextureCache::new(dir.clone());
        let dds = tiny_dxt1();
        let img = cache.get_or_decode(b"tex/key/0", &dds).expect("decode");
        assert_eq!((img.width, img.height), (4, 4));
        assert_eq!(img.rgba.len(), 64);
        assert_eq!(&img.rgba[0..4], &[255, 0, 0, 255]);

        let path = cache.path_for(b"tex/key/0");
        assert!(path.is_file(), "cache file not written: {path:?}");
        assert!(cache.size_bytes() >= (HEADER + 64) as u64);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn second_call_reads_from_cache_even_with_corrupt_input() {
        let dir = temp_dir("hit");
        let cache = TextureCache::new(dir.clone());
        let good = tiny_dxt1();
        let first = cache.get_or_decode(b"key", &good).expect("first decode");

        // Destroy the source DDS; a hit must still return the cached pixels.
        let mut broken = good.clone();
        broken.truncate(0);
        let second = cache.get_or_decode(b"key", &broken).expect("cache hit");
        assert_eq!(first, second);
        assert_eq!(&second.rgba[0..4], &[255, 0, 0, 255]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unsupported_dds_returns_none() {
        let dir = temp_dir("unsupported");
        let cache = TextureCache::new(dir.clone());
        let mut bad = dds_header(4, 4, b"ZZZZ");
        bad.extend_from_slice(&[0u8; 8]);
        assert!(cache.get_or_decode(b"bad", &bad).is_none());
        // A miss for an unsupported format must not create an entry.
        assert!(!cache.path_for(b"bad").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_cache_entry_is_ignored_and_redecoded() {
        let dir = temp_dir("malformed");
        fs::create_dir_all(&dir).unwrap();
        let cache = TextureCache::new(dir.clone());
        let dds = tiny_dxt1();

        // Wrong magic.
        fs::write(cache.path_for(b"m"), b"not a cache file at all").unwrap();
        let img = cache.get_or_decode(b"m", &dds).expect("re-decoded");
        assert_eq!((img.width, img.height), (4, 4));
        // Entry should have been repaired to a valid one.
        assert!(read_cached(&cache.path_for(b"m")).is_some());

        // Valid magic but truncated payload.
        let mut short = encode_cached(&img);
        short.truncate(HEADER + 10);
        fs::write(cache.path_for(b"s"), &short).unwrap();
        let img2 = cache.get_or_decode(b"s", &dds).expect("re-decoded 2");
        assert_eq!(img, img2);

        // Valid magic but bogus dimensions / trailing bytes.
        let mut bogus = encode_cached(&img);
        bogus[4..8].copy_from_slice(&9999u32.to_le_bytes());
        fs::write(cache.path_for(b"b"), &bogus).unwrap();
        assert_eq!(cache.get_or_decode(b"b", &dds).expect("re-decoded 3"), img);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_directory_does_not_panic() {
        let file = temp_dir("bad_dir_file");
        // Make `file` itself a regular file, then use it as a directory.
        fs::write(&file, b"x").unwrap();

        let cache = TextureCache::new(file.clone());
        let dds = tiny_dxt1();
        let img = cache.get_or_decode(b"k", &dds).expect("still decodes in memory");
        assert_eq!((img.width, img.height), (4, 4));
        assert_eq!(cache.size_bytes(), 0);
        assert!(cache.clear().is_ok() || cache.clear().is_err()); // no panic either way

        let _ = fs::remove_file(&file);
    }

    #[test]
    fn clear_removes_entries() {
        let dir = temp_dir("clear");
        let cache = TextureCache::new(dir.clone());
        let dds = tiny_dxt1();
        assert!(cache.get_or_decode(b"a", &dds).is_some());
        assert!(cache.get_or_decode(b"b", &dds).is_some());
        assert!(cache.size_bytes() > 0);
        cache.clear().expect("clear");
        assert_eq!(cache.size_bytes(), 0);
        // Clearing a never-created directory is fine too.
        assert!(TextureCache::new(dir.join("missing")).clear().is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn distinct_keys_get_distinct_files() {
        let dir = temp_dir("keys");
        let cache = TextureCache::new(dir.clone());
        let dds = tiny_dxt1();
        assert!(cache.get_or_decode(b"one", &dds).is_some());
        assert!(cache.get_or_decode(b"two", &dds).is_some());
        assert_ne!(cache.path_for(b"one"), cache.path_for(b"two"));
        // Length is part of the hash.
        assert_ne!(cache.path_for(b"ab"), cache.path_for(b"a"));
        let _ = fs::remove_dir_all(&dir);
    }
}
