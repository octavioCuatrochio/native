//! Reader for CryEngine `.pak` archives.
//!
//! CryEngine paks are ZIP archives with two quirks that break stock readers:
//!   1. local-header names use backslashes and may differ from the
//!      central-directory name;
//!   2. deflate streams are zlib-wrapped (with a 2-byte zlib header).
//!
//! We parse the central directory ourselves and inflate with `flate2`.

use anyhow::{bail, Context, Result};
use flate2::read::{DeflateDecoder, ZlibDecoder};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
const CD_SIG: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
const LFH_SIG: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];

fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn find_last(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).rev().find(|&i| &hay[i..i + needle.len()] == needle)
}

#[derive(Debug, Clone)]
pub struct Entry {
    /// Path normalised to forward slashes, as stored in the archive.
    pub name: String,
    pub method: u16,
    pub crc: u32,
    pub csize: u32,
    pub usize_: u32,
    pub local_offset: u32,
}

/// A single mounted archive.
pub struct Pak {
    path: PathBuf,
    pub entries: Vec<Entry>,
    index: HashMap<String, usize>,
}

impl Pak {
    pub fn open(path: impl AsRef<Path>) -> Result<Pak> {
        let path = path.as_ref().to_path_buf();
        let mut f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let len = f.metadata()?.len();
        if len < 22 {
            bail!("{}: too small to be a zip", path.display());
        }

        let tail_len = std::cmp::min(len, 66_000) as usize;
        let mut tail = vec![0u8; tail_len];
        f.seek(SeekFrom::Start(len - tail_len as u64))?;
        f.read_exact(&mut tail)?;

        let eocd_rel = find_last(&tail, &EOCD_SIG).context("EOCD signature not found")?;
        let eocd = &tail[eocd_rel..];
        let n_total = rd_u16(eocd, 10) as usize;
        let cd_size = rd_u32(eocd, 12) as usize;
        let cd_off = rd_u32(eocd, 16) as usize;

        let mut cd = vec![0u8; cd_size];
        f.seek(SeekFrom::Start(cd_off as u64))?;
        f.read_exact(&mut cd)?;

        let mut entries = Vec::with_capacity(n_total);
        let mut index = HashMap::with_capacity(n_total);
        let mut o = 0usize;
        for _ in 0..n_total {
            if cd.get(o..o + 4) != Some(&CD_SIG) {
                bail!("bad central-directory record at offset {o}");
            }
            let method = rd_u16(&cd, o + 10);
            let crc = rd_u32(&cd, o + 16);
            let csize = rd_u32(&cd, o + 20);
            let usize_ = rd_u32(&cd, o + 24);
            let nlen = rd_u16(&cd, o + 28) as usize;
            let elen = rd_u16(&cd, o + 30) as usize;
            let clen = rd_u16(&cd, o + 32) as usize;
            let lofs = rd_u32(&cd, o + 42);
            let raw = cd.get(o + 46..o + 46 + nlen).context("truncated central dir")?;
            let name = String::from_utf8_lossy(raw).replace('\\', "/");
            index.insert(name.to_ascii_lowercase(), entries.len());
            entries.push(Entry { name, method, crc, csize, usize_, local_offset: lofs });
            o += 46 + nlen + elen + clen;
        }

        Ok(Pak { path, entries, index })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Case-insensitive lookup (pak entries are lowercase; scripts are not always).
    pub fn find(&self, name: &str) -> Option<usize> {
        self.index.get(&name.replace('\\', "/").to_ascii_lowercase()).copied()
    }

    pub fn read_idx(&self, idx: usize) -> Result<Vec<u8>> {
        let e = self.entries.get(idx).context("entry index out of range")?;
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(e.local_offset as u64))?;
        let mut lh = [0u8; 30];
        f.read_exact(&mut lh)?;
        if lh[0..4] != LFH_SIG {
            bail!("bad local header for {}", e.name);
        }
        let nlen = rd_u16(&lh, 26) as usize;
        let elen = rd_u16(&lh, 28) as usize;
        let data_off = e.local_offset as u64 + 30 + nlen as u64 + elen as u64;
        f.seek(SeekFrom::Start(data_off))?;
        let mut raw = vec![0u8; e.csize as usize];
        f.read_exact(&mut raw)?;

        match e.method {
            0 => Ok(raw),
            8 => {
                // zlib-wrapped first (CryEngine), raw deflate as fallback.
                let mut out = Vec::with_capacity(e.usize_ as usize);
                if ZlibDecoder::new(&raw[..]).read_to_end(&mut out).is_ok() {
                    return Ok(out);
                }
                let mut out = Vec::with_capacity(e.usize_ as usize);
                DeflateDecoder::new(&raw[..])
                    .read_to_end(&mut out)
                    .with_context(|| format!("deflate failed for {}", e.name))?;
                Ok(out)
            }
            m => bail!("unsupported compression method {m} for {}", e.name),
        }
    }
}

/// A virtual file system over several mounted paks. Later mounts win
/// (mirrors CryEngine's pak override precedence). A loose directory, when set,
/// takes priority over every pak.
pub struct Vfs {
    paks: Vec<Pak>,
    loose_root: Option<PathBuf>,
}

impl Default for Vfs {
    fn default() -> Self {
        Self::new()
    }
}

impl Vfs {
    pub fn new() -> Self {
        Vfs { paks: Vec::new(), loose_root: None }
    }

    /// Prefer files under `root` (mirrors disk-over-pak override).
    pub fn set_loose_root(&mut self, root: impl AsRef<Path>) {
        self.loose_root = Some(root.as_ref().to_path_buf());
    }

    pub fn mount(&mut self, pak: impl AsRef<Path>) -> Result<usize> {
        let p = Pak::open(pak)?;
        let n = p.len();
        self.paks.push(p);
        Ok(n)
    }

    pub fn pak_count(&self) -> usize {
        self.paks.len()
    }
    pub fn entry_count(&self) -> usize {
        self.paks.iter().map(|p| p.len()).sum()
    }

    pub fn read(&self, name: &str) -> Option<Vec<u8>> {
        let norm = name.replace('\\', "/");
        if let Some(root) = &self.loose_root {
            let cand = root.join(&norm);
            if cand.is_file() {
                if let Ok(d) = std::fs::read(&cand) {
                    return Some(d);
                }
            }
        }
        for pak in self.paks.iter().rev() {
            if let Some(i) = pak.find(&norm) {
                if let Ok(d) = pak.read_idx(i) {
                    return Some(d);
                }
            }
        }
        None
    }

    /// All entry paths (across paks, last-wins order not applied) ending with `ext`.
    pub fn paths_with_ext(&self, ext: &str) -> Vec<String> {
        let ext = ext.to_ascii_lowercase();
        let mut out = Vec::new();
        for pak in &self.paks {
            for e in &pak.entries {
                if e.name.to_ascii_lowercase().ends_with(&ext) {
                    out.push(e.name.clone());
                }
            }
        }
        out
    }

    pub fn exists(&self, name: &str) -> bool {
        let norm = name.replace('\\', "/");
        if let Some(root) = &self.loose_root {
            if root.join(&norm).is_file() {
                return true;
            }
        }
        self.paks.iter().rev().any(|p| p.find(&norm).is_some())
    }
}
