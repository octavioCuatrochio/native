//! FMOD Designer event banks (`.fev`) for State of Decay: YOSE.
//!
//! The game ships 18 `.fev` event projects (13 in `sounds.pak`, 5 DLC2 projects
//! in the Beltway/Breakdown paks). A `.fev` is a RIFF container whose real
//! payload is the `LGCY` chunk (FEV1 format); `STRR` holds the string table used
//! by the version's name indices. All shipped projects are RIFF FEV version
//! `0x00450000` (v69).
//!
//! This module decodes the event hierarchy (categories → event groups →
//! events), each event's sound definitions (wave bank + sub-sound index) and the
//! scalar per-event properties (volume, pitch, priority, max playbacks, mode,
//! 3D min/max distance). It then maps the path syntax the shipped Lua scripts
//! use, e.g. `Sounds/explosions:exp_specific:grenade`, onto those definitions.
//!
//! # Container layout
//!
//! ```text
//! RIFF <size> "FEV " "FMT " <4> <version u32le>   // version at 0x14
//! LIST <size> "PROJ"
//!     OBCT ...      // object table (ignored here)
//!     PROP ...      // project properties
//!     LGCY <size>   // the FEV1 database (see below)
//!     EPRP ...
//!     STRR <size>   // string table: u32 count, count*u32 offsets, buffer
//!     LANG ...
//! ```
//!
//! The `LGCY` database is a sequential stream (no offsets), read in this order:
//! project/wave-bank names, the sound-category tree, the event-group tree with
//! its events, the sound-definition pool, then reverb definitions. Names in the
//! RIFF variant are indices into `STRR` (the leaf event name and the sound-def
//! name are separate strings; the wave/stream names inside a sound definition
//! are inline length-prefixed strings).
//!
//! Every structure below is version-gated exactly as the shipped
//! `fmod_event.dll` reader gates it; because only v0x41..v0x45 occur here, the
//! branches that matter are the ones active in that window. See
//! `notes/audio-events.md` for the field-by-field layout and unknowns.
//!
//! # Resolution chain
//!
//! ```text
//! script "Sounds/<project>:<group>:<event>"
//!   -> project            (FEV file stem or project name)
//!   -> event              (group path + leaf, leaf-only fallback)
//!   -> sound defs[]       (indices stored on the event)
//!   -> (wave bank, subsound) per entry
//!   -> "<fev dir>/<wave bank>.fsb" + subsound index
//! ```

use std::collections::{HashMap, HashSet};

use crate::crypak::Vfs;

const RIFF_TAG: &[u8; 4] = b"RIFF";
const FEV_TAG: &[u8; 4] = b"FEV ";
const FMT_TAG: &[u8; 4] = b"FMT ";
const PROJ_TAG: &[u8; 4] = b"PROJ";
const LGCY_TAG: &[u8; 4] = b"LGCY";
const STRR_TAG: &[u8; 4] = b"STRR";

/// Oldest and newest RIFF FEV version this reader understands. The shipped YOSE
/// projects are all [`FEV_VERSION_MAX`] (`0x00450000`, FMOD FEV v69).
pub const FEV_VERSION_MIN: u32 = 0x0041_0000;
pub const FEV_VERSION_MAX: u32 = 0x0045_0000;

/// Upper bound on a length-prefixed FEV string; real names are far shorter.
const MAX_STRING: u32 = 1024;
/// Upper bound on any count read from the stream, to reject absurd allocations.
const MAX_ITEMS: u32 = 1_000_000;
/// Recursion guard for the category / event-group trees.
const MAX_DEPTH: u32 = 64;

#[inline]
fn u32le(b: &[u8], o: usize) -> Option<u32> {
    let e = o.checked_add(4)?;
    let s = b.get(o..e)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// A bounds-checked little-endian cursor over a FEV stream.
struct Reader<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8], o: usize) -> Self {
        Reader { b, o }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.o.checked_add(n)?;
        if end > self.b.len() {
            return None;
        }
        let s = &self.b[self.o..end];
        self.o = end;
        Some(s)
    }

    fn skip(&mut self, n: usize) -> Option<()> {
        self.take(n).map(|_| ())
    }

    fn u16(&mut self) -> Option<u16> {
        let s = self.take(2)?;
        Some(u16::from_le_bytes([s[0], s[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        let s = self.take(4)?;
        Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn f32(&mut self) -> Option<f32> {
        let s = self.take(4)?;
        Some(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// Length-prefixed string: `u32 len` (including the NUL) then the bytes.
    fn string(&mut self) -> Option<String> {
        let len = self.u32()?;
        if len == 0 {
            return Some(String::new());
        }
        if len > MAX_STRING {
            return None;
        }
        let raw = self.take(len as usize)?;
        let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
        Some(String::from_utf8_lossy(&raw[..end]).into_owned())
    }

    /// UUIDs are always present but unused; skip them.
    fn uuid(&mut self) -> Option<()> {
        self.skip(16)
    }
}

/// The RIFF `STRR` string table.
struct StringTable {
    data: Vec<u8>,
    count: usize,
}

impl StringTable {
    fn parse(bytes: &[u8], off: usize, size: usize) -> Option<StringTable> {
        let end = off.checked_add(size)?.min(bytes.len());
        let data = bytes.get(off..end)?;
        let count = u32le(data, 0)? as usize;
        if count as u64 > MAX_ITEMS as u64 {
            return None;
        }
        if 4usize.checked_add(count.checked_mul(4)?)? > data.len() {
            return None;
        }
        Some(StringTable { data: data.to_vec(), count })
    }

    fn get(&self, idx: usize) -> Option<String> {
        if idx >= self.count {
            return None;
        }
        let rel = u32le(&self.data, 4 + idx * 4)? as usize;
        let buf = 4 + self.count * 4;
        let start = buf.checked_add(rel)?;
        if start >= self.data.len() {
            return None;
        }
        let rest = &self.data[start..];
        let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
        Some(String::from_utf8_lossy(&rest[..end]).into_owned())
    }
}

/// One entry of a sound definition: a source stream and where it lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoundDefEntry {
    /// Source file name as authored (e.g. `exp_cmn/exp_cmn_base_01.wav`).
    pub stream: String,
    /// Index into [`FevProject::wave_banks`].
    pub wave_bank: u32,
    /// Sub-sound index within that wave bank's FSB.
    pub subsound: u32,
}

/// A named sound definition (a set of one or more wave/oscillator entries).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SoundDef {
    pub name: String,
    pub entries: Vec<SoundDefEntry>,
}

/// A single event and its scalar properties.
#[derive(Debug, Clone, PartialEq)]
pub struct FevEvent {
    /// Leaf event name (the last path component).
    pub name: String,
    /// Event-group path leading to the event (the `LGCY` event-category tree).
    pub group: Vec<String>,
    /// Sound categories the event belongs to (inline strings on the event).
    pub categories: Vec<String>,
    /// Indices into [`FevProject::sound_defs`].
    pub sound_defs: Vec<usize>,
    pub volume: f32,
    pub pitch: f32,
    pub priority: u32,
    pub max_playbacks: u32,
    /// Raw event mode/flags word (3D/2D/looping bits; see notes).
    pub mode: u32,
    pub min_distance: f32,
    pub max_distance: f32,
}

impl FevEvent {
    /// Canonical FMOD-style path: `/<group>/<group>/<name>`.
    pub fn path(&self) -> String {
        let mut s = String::new();
        for g in &self.group {
            s.push('/');
            s.push_str(g);
        }
        s.push('/');
        s.push_str(&self.name);
        s
    }
}

/// A decoded FEV project (one `.fev` file).
#[derive(Debug, Clone)]
pub struct FevProject {
    /// Project / wave-bank name (the FEV "bank name", e.g. `explosions`).
    pub name: String,
    pub version: u32,
    /// Directory the `.fev` was read from, used to locate sibling FSB banks.
    pub fsb_dir: String,
    /// Wave-bank (FSB) names, indexed by [`SoundDefEntry::wave_bank`].
    pub wave_banks: Vec<String>,
    pub sound_defs: Vec<SoundDef>,
    /// Flattened event list, each carrying its group path.
    pub events: Vec<FevEvent>,
}

impl FevProject {
    /// VFS path of the FSB backing `wave_bank`, or `None` if out of range.
    pub fn fsb_path(&self, wave_bank: usize) -> Option<String> {
        let wb = self.wave_banks.get(wave_bank)?;
        Some(if self.fsb_dir.is_empty() {
            format!("{wb}.fsb")
        } else {
            format!("{}/{wb}.fsb", self.fsb_dir)
        })
    }
}

// ---------------------------------------------------------------------------
// LGCY parser
// ---------------------------------------------------------------------------

/// Locate the `LGCY` and `STRR` chunks inside `RIFF > LIST > PROJ`.
/// Returns `(lgcy_data, lgcy_size, strr_data, strr_size)`.
fn find_proj_chunks(bytes: &[u8]) -> Option<(usize, usize, usize, usize)> {
    let mut lgcy = None;
    let mut strr = None;
    let mut off = 0x0Cusize;
    while off + 8 <= bytes.len() {
        let id = bytes.get(off..off + 4)?;
        let size = u32le(bytes, off + 4)? as usize;
        if id == b"LIST" {
            let typ = bytes.get(off + 8..off + 12)?;
            if typ == PROJ_TAG {
                let sub_end = (off + 8 + size).min(bytes.len());
                let mut so = off + 12;
                while so + 8 <= sub_end {
                    let sid = bytes.get(so..so + 4)?;
                    let ssz = u32le(bytes, so + 4)? as usize;
                    if sid == LGCY_TAG {
                        lgcy = Some((so + 8, ssz));
                    } else if sid == STRR_TAG {
                        strr = Some((so + 8, ssz));
                    }
                    so += 8 + ssz + (ssz & 1);
                }
            }
        }
        off += 8 + size + (size & 1);
    }
    let (lo, ls) = lgcy?;
    let (ro, rs) = strr?;
    Some((lo, ls, ro, rs))
}

fn parse_properties(r: &mut Reader) -> Option<()> {
    let n = r.u32()?;
    if n > MAX_ITEMS {
        return None;
    }
    for _ in 0..n {
        r.string()?;
        let ty = r.u32()?;
        if ty == 2 {
            r.string()?;
        } else {
            r.skip(4)?;
        }
    }
    Some(())
}

fn parse_category(r: &mut Reader, v: u32, depth: u32) -> Option<()> {
    if depth > MAX_DEPTH {
        return None;
    }
    r.string()?;
    r.skip(8)?;
    if v >= 0x0029_0000 {
        r.skip(8)?;
    }
    let n = r.u32()?;
    if n > MAX_ITEMS {
        return None;
    }
    for _ in 0..n {
        parse_category(r, v, depth + 1)?;
    }
    Some(())
}

fn parse_event_sound(r: &mut Reader, v: u32) -> Option<u16> {
    let idx = if v >= 0x0027_0000 {
        r.u16()?
    } else {
        r.string()?;
        0
    };
    r.skip(0x08)?;
    if v >= 0x001E_0000 {
        r.skip(4)?;
    }
    r.skip(4)?;
    if v >= 0x001F_0000 {
        r.skip(4)?;
    }
    r.skip(8)?;
    if v >= 0x0024_0000 {
        r.skip(4)?;
    }
    r.skip(4)?;
    if v < 0x0034_0000 {
        r.skip(8)?;
    }
    r.skip(0x0C)?;
    if v >= 0x0018_0000 {
        r.skip(8)?;
    }
    Some(idx)
}

fn parse_event_envelope(r: &mut Reader, v: u32) -> Option<()> {
    if v >= 0x0027_0000 {
        r.skip(4)?;
    } else {
        r.string()?;
        r.string()?;
    }
    r.string()?; // DSP unit name (e.g. "FMOD Lowpass Simple")
    r.skip(4)?;
    if v >= 0x0026_0000 {
        r.skip(4)?;
    }
    if v >= 0x0039_0000 {
        r.skip(4)?;
    }
    let points = r.u32()? as usize;
    let scaled = points.checked_mul(if v >= 0x0041_0000 { 4 } else { 8 })?;
    r.skip(scaled)?;
    if v < 0x0041_0000 && v >= 0x000D_0000 {
        r.skip(points.checked_mul(4)?)?;
    }
    r.skip(4)?;
    if v >= 0x001A_0000 {
        r.skip(4)?;
    }
    Some(())
}

fn parse_event_complex(r: &mut Reader, v: u32, sounds: &mut Vec<usize>) -> Option<()> {
    let layers = r.u32()?;
    if layers > MAX_ITEMS {
        return None;
    }
    for _ in 0..layers {
        if v < 0x0027_0000 {
            r.string()?;
        }
        r.skip(2)?;
        if v < 0x001D_0000 {
            r.skip(2)?;
        }
        if v >= 0x0025_0000 {
            r.skip(2)?;
        }
        if v >= 0x0027_0000 {
            r.skip(2)?;
        } else {
            r.string()?;
        }
        let (n_sounds, n_envelopes) = if v >= 0x001D_0000 {
            (r.u16()? as u32, r.u16()? as u32)
        } else {
            (r.u32()?, r.u32()?)
        };
        if n_sounds > MAX_ITEMS || n_envelopes > MAX_ITEMS {
            return None;
        }
        for _ in 0..n_sounds {
            sounds.push(parse_event_sound(r, v)? as usize);
        }
        for _ in 0..n_envelopes {
            parse_event_envelope(r, v)?;
        }
    }

    let params = r.u32()?;
    if params > MAX_ITEMS {
        return None;
    }
    for _ in 0..params {
        if v >= 0x0041_0000 {
            r.skip(4)?;
        } else {
            r.string()?;
        }
        if v < 0x0012_0000 {
            r.skip(8)?;
        }
        r.skip(0x10)?;
        if v >= 0x000B_0000 && v < 0x0010_0000 {
            r.skip(4)?;
        }
        if v >= 0x0012_0000 {
            r.skip(4)?;
        }
        r.skip(4)?;
        if v >= 0x000C_0000 {
            let n = r.u32()?;
            if n > MAX_ITEMS {
                return None;
            }
            r.skip((n as usize).checked_mul(4)?)?;
        }
    }
    parse_properties(r)
}

fn parse_event(r: &mut Reader, v: u32, strr: &StringTable) -> Option<FevEvent> {
    let event_type = if v >= 0x0034_0000 { r.u32()? } else { 0x08 };
    let name = if v >= 0x0041_0000 {
        let idx = r.u32()? as usize;
        strr.get(idx)?
    } else {
        r.string()?
    };
    if v >= 0x003A_0000 {
        r.uuid()?;
    }

    // Event property block (fields documented in notes/audio-events.md).
    let volume = r.f32()?;
    let pitch = r.f32()?;
    if v >= 0x001B_0000 {
        r.skip(4)?; // pitch randomisation
    }
    if v >= 0x0020_0000 {
        r.skip(4)?; // volume randomisation
    }
    let priority = if v >= 0x000A_0000 { r.u32()? } else { 0 };
    let max_playbacks = r.u32()?;
    if v >= 0x0038_0000 {
        r.skip(4)?; // steal priority
    }
    let mode = r.u32()?;
    let min_distance = r.f32()?;
    let max_distance = r.f32()?;
    if v >= 0x0045_0000 {
        r.skip(8)?; // distance filtering, filter centre frequency
    }
    r.skip(4)?; // music / flags
    if v >= 0x0009_0000 {
        r.skip(0x2C)?; // speaker levels[8] + cone angles/volume
    }
    if v >= 0x000B_0000 {
        r.skip(8)?;
    }
    if v >= 0x001C_0000 {
        r.skip(4)?;
    }
    if v >= 0x000B_0000 {
        r.skip(4)?;
    }
    if v >= 0x0012_0000 {
        r.skip(4)?;
    }
    if v >= 0x0013_0000 {
        r.skip(8)?;
    }
    if v >= 0x002B_0000 {
        r.skip(4)?;
    }
    if v >= 0x002D_0000 {
        r.skip(4)?;
    }
    if v >= 0x0016_0000 {
        r.skip(4)?;
    }
    if v >= 0x0044_0000 {
        r.skip(4)?;
    }
    if v >= 0x0028_0000 {
        r.skip(4)?;
    }

    let mut sound_defs = Vec::new();
    if (event_type & 0x18) == 0x08 {
        parse_event_complex(r, v, &mut sound_defs)?;
    } else if (event_type & 0x18) == 0x10 {
        r.skip(4)?;
        sound_defs.push(parse_event_sound(r, v)? as usize);
    } else {
        return None;
    }

    let n_cats = r.u32()?;
    if n_cats > MAX_ITEMS {
        return None;
    }
    let mut categories = Vec::with_capacity(n_cats as usize);
    for _ in 0..n_cats {
        categories.push(r.string()?);
    }

    Some(FevEvent {
        name,
        group: Vec::new(),
        categories,
        sound_defs,
        volume,
        pitch,
        priority,
        max_playbacks,
        mode,
        min_distance,
        max_distance,
    })
}

fn parse_event_category(
    r: &mut Reader,
    v: u32,
    strr: &StringTable,
    group: &mut Vec<String>,
    out: &mut Vec<FevEvent>,
    depth: u32,
) -> Option<()> {
    if depth > MAX_DEPTH {
        return None;
    }
    let name = if v >= 0x0041_0000 {
        strr.get(r.u32()? as usize)?
    } else {
        r.string()?
    };
    if v >= 0x0017_0000 {
        parse_properties(r)?;
    }
    let sub_groups = r.u32()?;
    let events = r.u32()?;
    if sub_groups > MAX_ITEMS || events > MAX_ITEMS {
        return None;
    }
    group.push(name);
    for _ in 0..sub_groups {
        parse_event_category(r, v, strr, group, out, depth + 1)?;
    }
    for _ in 0..events {
        let mut ev = parse_event(r, v, strr)?;
        ev.group = group.clone();
        out.push(ev);
    }
    group.pop();
    Some(())
}

fn parse_sound_def_def(r: &mut Reader, v: u32) -> Option<()> {
    r.skip(8)?;
    if v < 0x0022_0000 || v >= 0x0026_0000 {
        r.skip(4)?;
    }
    r.skip(8)?;
    if v >= 0x001B_0000 {
        r.skip(4)?;
    }
    r.skip(8)?;
    if v >= 0x001B_0000 {
        r.skip(4)?;
    }
    r.skip(4)?;
    if v >= 0x001B_0000 {
        r.skip(4)?;
    }
    r.skip(8)?;
    if v >= 0x001B_0000 {
        r.skip(4)?;
    }
    if v >= 0x003C_0000 {
        r.skip(4)?;
    }
    if v >= 0x0044_0000 {
        r.skip(4)?;
    }
    if v >= 0x002A_0000 {
        r.skip(4)?;
    }
    if v >= 0x003E_0000 {
        r.skip(4)?;
    }
    if v >= 0x003F_0000 {
        r.skip(2)?;
    }
    Some(())
}

fn parse_sound_def(r: &mut Reader, v: u32, strr: &StringTable) -> Option<SoundDef> {
    let name = if v >= 0x0041_0000 {
        strr.get(r.u32()? as usize)?
    } else {
        r.string()?
    };
    if v >= 0x002E_0000 {
        r.skip(4)?; // sound def pool index
    } else {
        parse_sound_def_def(r, v)?;
    }
    let entries = r.u32()?;
    if entries > MAX_ITEMS {
        return None;
    }
    let mut out = Vec::with_capacity(entries as usize);
    for _ in 0..entries {
        let entry_type = r.u32()?;
        if v >= 0x000E_0000 {
            r.skip(4)?; // weight
        }
        match entry_type {
            0 => {
                let stream = r.string()?;
                let wave_bank = if v >= 0x0041_0000 { r.u32()? } else { 0 };
                let subsound = r.u32()?;
                if v >= 0x0008_0000 {
                    r.skip(4)?; // length in ms
                }
                out.push(SoundDefEntry { stream, wave_bank, subsound });
            }
            1 => r.skip(8)?, // oscillator
            2 | 3 => {}      // null / programmer
            _ => return None,
        }
    }
    Some(SoundDef { name, entries: out })
}

fn parse_reverb(r: &mut Reader, v: u32) -> Option<()> {
    r.string()?;
    r.skip(0x30)?;
    if v >= 0x001C_0000 {
        r.skip(8)?;
    }
    r.skip(0x4C)?;
    Some(())
}

/// Parse a RIFF FMOD Designer `.fev` project.
///
/// Returns `None` for non-FEV data, an unsupported version, or any structural
/// inconsistency; it never panics and bounds every read and count.
pub fn parse_fev(bytes: &[u8]) -> Option<FevProject> {
    if bytes.len() < 0x18 {
        return None;
    }
    if bytes.get(0..4)? != RIFF_TAG {
        return None;
    }
    if bytes.get(8..12)? != FEV_TAG {
        return None;
    }
    if bytes.get(12..16)? != FMT_TAG {
        return None;
    }
    let version = u32le(bytes, 0x14)?;
    if version & 0xFF00_FFFF != 0 {
        return None;
    }
    if version < FEV_VERSION_MIN || version > FEV_VERSION_MAX {
        return None;
    }

    let (lgcy_off, _lgcy_size, strr_off, strr_size) = find_proj_chunks(bytes)?;
    let strr = StringTable::parse(bytes, strr_off, strr_size)?;

    let mut r = Reader::new(bytes, lgcy_off);

    // parse_fev_main
    if version >= 0x002E_0000 {
        r.skip(4)?;
    }
    if version >= 0x0032_0000 {
        r.skip(4)?;
    }
    if version == 0x0040_0000 {
        let n = r.u32()? as usize;
        r.skip(n.checked_mul(8)?)?;
    }

    let name = r.string()?;

    let mut languages = 1u32;
    let wave_bank_count = r.u32()?;
    if version >= 0x0041_0000 {
        languages = r.u32()?;
    }
    if wave_bank_count > MAX_ITEMS {
        return None;
    }
    let mut wave_banks = Vec::with_capacity(wave_bank_count as usize);
    for _ in 0..wave_bank_count {
        r.skip(4)?;
        if version >= 0x0014_0000 {
            r.skip(4)?;
        }
        if version >= 0x003D_0000 {
            r.skip((languages as usize).checked_mul(8)?)?;
        }
        if version >= 0x0041_0000 {
            r.skip((languages as usize).checked_mul(4)?)?;
        }
        wave_banks.push(r.string()?);
    }

    parse_category(&mut r, version, 0)?;

    let event_groups = r.u32()?;
    if event_groups > MAX_ITEMS {
        return None;
    }
    let mut events = Vec::new();
    let mut group = Vec::new();
    for _ in 0..event_groups {
        parse_event_category(&mut r, version, &strr, &mut group, &mut events, 0)?;
    }

    if version >= 0x002E_0000 {
        let n = r.u32()?;
        if n > MAX_ITEMS {
            return None;
        }
        for _ in 0..n {
            parse_sound_def_def(&mut r, version)?;
        }
    }

    let sound_def_count = r.u32()?;
    if sound_def_count > MAX_ITEMS {
        return None;
    }
    let mut sound_defs = Vec::with_capacity(sound_def_count as usize);
    for _ in 0..sound_def_count {
        sound_defs.push(parse_sound_def(&mut r, version, &strr)?);
    }

    if version >= 0x0015_0000 {
        let n = r.u32()?;
        if n > MAX_ITEMS {
            return None;
        }
        for _ in 0..n {
            parse_reverb(&mut r, version)?;
        }
    }

    Some(FevProject {
        name,
        version,
        fsb_dir: String::new(),
        wave_banks,
        sound_defs,
        events,
    })
}

// ---------------------------------------------------------------------------
// Database and script-path resolution
// ---------------------------------------------------------------------------

/// A sound resolved from an event down to a concrete FSB sub-sound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSound {
    /// VFS path of the FSB bank (case may differ from the on-disk name).
    pub fsb_path: String,
    pub subsound: u32,
    /// Authored source file name of the sub-sound.
    pub stream: String,
}

/// A fully resolved event: properties plus every referenced FSB sub-sound.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedEvent {
    pub project: String,
    pub path: String,
    pub name: String,
    pub group: Vec<String>,
    pub categories: Vec<String>,
    pub volume: f32,
    pub pitch: f32,
    pub priority: u32,
    pub max_playbacks: u32,
    pub mode: u32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub sounds: Vec<ResolvedSound>,
}

/// All loaded FEV projects, indexed for script-path lookup.
#[derive(Default)]
pub struct FevDb {
    projects: Vec<FevProject>,
    by_stem: HashMap<String, usize>,
    by_name: HashMap<String, usize>,
}

impl FevDb {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.projects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.projects.is_empty()
    }

    pub fn projects(&self) -> &[FevProject] {
        &self.projects
    }

    /// Total number of events across every loaded project.
    pub fn event_count(&self) -> usize {
        self.projects.iter().map(|p| p.events.len()).sum()
    }

    /// Insert a parsed project, recording the directory of `vfs_path` so sibling
    /// FSB banks can be located. Later inserts win for name/stem lookups.
    pub fn insert(&mut self, mut project: FevProject, vfs_path: &str) -> usize {
        project.fsb_dir = vfs_dir(vfs_path).to_string();
        let stem = file_stem(vfs_path).to_ascii_lowercase();
        let name = project.name.to_ascii_lowercase();
        let idx = self.projects.len();
        self.projects.push(project);
        if !stem.is_empty() {
            self.by_stem.insert(stem, idx);
        }
        if !name.is_empty() {
            self.by_name.insert(name, idx);
        }
        idx
    }

    /// Parse `bytes` and insert them; returns the project index.
    pub fn add(&mut self, vfs_path: &str, bytes: &[u8]) -> Option<usize> {
        let project = parse_fev(bytes)?;
        Some(self.insert(project, vfs_path))
    }

    /// Parse and insert every `*.fev` reachable through `vfs`; returns the count.
    pub fn load_vfs(&mut self, vfs: &Vfs) -> usize {
        let mut seen = HashSet::new();
        let mut loaded = 0;
        for path in vfs.paths_with_ext(".fev") {
            if !seen.insert(path.to_ascii_lowercase()) {
                continue;
            }
            if let Some(bytes) = vfs.read(&path) {
                if self.add(&path, &bytes).is_some() {
                    loaded += 1;
                }
            }
        }
        loaded
    }

    /// Resolve a script sound path such as
    /// `Sounds/explosions:exp_specific:grenade`.
    ///
    /// The project component matches either the FEV file stem or its project
    /// name. The event is first matched by its full group path + leaf; if that
    /// fails and the leaf is unique within the project, the leaf alone is used
    /// (DLC content is stored under renamed groups).
    pub fn resolve(&self, script_path: &str) -> Option<ResolvedEvent> {
        let low = script_path.trim().to_ascii_lowercase();
        let low = low.strip_prefix("sounds/").unwrap_or(&low);
        let mut parts = low.splitn(3, ':');
        let proj = parts.next()?;
        let group = parts.next().unwrap_or("");
        let leaf = parts.next().unwrap_or("");
        if leaf.is_empty() {
            return None;
        }

        let stem = file_stem(proj);
        let project_idx = match self
            .by_stem
            .get(stem)
            .or_else(|| self.by_name.get(stem))
            .copied()
        {
            Some(i) => i,
            None => {
                // Unknown project name: accept a unique project that owns the leaf.
                let mut found = None;
                let mut count = 0;
                for (i, p) in self.projects.iter().enumerate() {
                    if p.events.iter().any(|e| e.name.eq_ignore_ascii_case(leaf)) {
                        found = Some(i);
                        count += 1;
                    }
                }
                if count == 1 {
                    found?
                } else {
                    return None;
                }
            }
        };

        let project = &self.projects[project_idx];
        let group_parts: Vec<&str> = if group.is_empty() {
            Vec::new()
        } else {
            group.split('/').collect()
        };

        let mut event_idx = None;
        for (i, e) in project.events.iter().enumerate() {
            let group_matches = e.group.len() == group_parts.len()
                && e.group
                    .iter()
                    .zip(&group_parts)
                    .all(|(a, b)| a.eq_ignore_ascii_case(b));
            if group_matches && e.name.eq_ignore_ascii_case(leaf) {
                event_idx = Some(i);
                break;
            }
        }
        if event_idx.is_none() {
            let mut found = None;
            let mut count = 0;
            for (i, e) in project.events.iter().enumerate() {
                if e.name.eq_ignore_ascii_case(leaf) {
                    found = Some(i);
                    count += 1;
                }
            }
            if count == 1 {
                event_idx = found;
            }
        }
        let event_idx = event_idx?;
        let event = &project.events[event_idx];

        let mut sounds = Vec::new();
        for &sd_idx in &event.sound_defs {
            let Some(sd) = project.sound_defs.get(sd_idx) else {
                continue;
            };
            for entry in &sd.entries {
                if let Some(fsb_path) = project.fsb_path(entry.wave_bank as usize) {
                    sounds.push(ResolvedSound {
                        fsb_path,
                        subsound: entry.subsound,
                        stream: entry.stream.clone(),
                    });
                }
            }
        }

        Some(ResolvedEvent {
            project: project.name.clone(),
            path: event.path(),
            name: event.name.clone(),
            group: event.group.clone(),
            categories: event.categories.clone(),
            volume: event.volume,
            pitch: event.pitch,
            priority: event.priority,
            max_playbacks: event.max_playbacks,
            mode: event.mode,
            min_distance: event.min_distance,
            max_distance: event.max_distance,
            sounds,
        })
    }
}

/// Check a resolved sound against the FSB bank via [`crate::audio`]: the bank
/// must parse and the sub-sound index must be in range.
pub fn sound_exists(vfs: &Vfs, sound: &ResolvedSound) -> bool {
    let Some(bytes) = vfs.read(&sound.fsb_path) else {
        return false;
    };
    let Some(fsb) = crate::audio::parse_fsb(&bytes) else {
        return false;
    };
    (sound.subsound as usize) < fsb.num_subsounds
}

/// Directory component of a VFS path (`""` when there is none).
fn vfs_dir(p: &str) -> &str {
    match p.rfind('/') {
        Some(i) => &p[..i],
        None => "",
    }
}

/// File name without its extension.
fn file_stem(p: &str) -> &str {
    let base = match p.rfind('/') {
        Some(i) => &p[i + 1..],
        None => p,
    };
    match base.rfind('.') {
        Some(i) => &base[..i],
        None => base,
    }
}

// ---------------------------------------------------------------------------
// Lua script reference extraction
// ---------------------------------------------------------------------------

/// Sound functions whose first string argument names an event (or sample).
const SOUND_FNS: &[&str] = &[
    "PlaySoundEventEx",
    "PlaySoundEvent",
    "Load3DSound",
    "LoadSound",
    "PlayEx",
    "Precache",
    "CacheAudioFile",
    "PlaySound",
];

/// Extract the first string literal passed to each sound call in `source`,
/// skipping whole-line comments. Only names containing `:` (i.e. event paths)
/// are returned, de-duplicated in first-seen order.
pub fn script_event_refs(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for line in source.lines() {
        if line.trim_start().starts_with("--") {
            continue;
        }
        let bytes = line.as_bytes();
        for &fname in SOUND_FNS {
            let mut from = 0;
            while let Some(rel) = line[from..].find(fname) {
                let at = from + rel;
                let mut k = at + fname.len();
                while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t') {
                    k += 1;
                }
                if k < bytes.len() && bytes[k] == b'(' {
                    k += 1;
                    while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t') {
                        k += 1;
                    }
                    if k < bytes.len() && bytes[k] == b'"' {
                        let start = k + 1;
                        if let Some(end) = line[start..].find('"') {
                            let val = &line[start..start + end];
                            if val.contains(':') && seen.insert(val.to_string()) {
                                out.push(val.to_string());
                            }
                        }
                    }
                }
                from = at + fname.len();
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn empty_project(name: &str, wave_banks: &[&str], sound_defs: Vec<SoundDef>, events: Vec<FevEvent>) -> FevProject {
        FevProject {
            name: name.to_string(),
            version: FEV_VERSION_MAX,
            fsb_dir: String::new(),
            wave_banks: wave_banks.iter().map(|s| s.to_string()).collect(),
            sound_defs,
            events,
        }
    }

    fn event(name: &str, group: &[&str], sound_defs: Vec<usize>) -> FevEvent {
        FevEvent {
            name: name.to_string(),
            group: group.iter().map(|s| s.to_string()).collect(),
            categories: vec!["sfx".to_string()],
            sound_defs,
            volume: 0.5,
            pitch: 1.0,
            priority: 10,
            max_playbacks: 2,
            mode: 0x180010,
            min_distance: 1.0,
            max_distance: 100.0,
        }
    }

    #[test]
    fn rejects_non_fev_and_bad_version() {
        assert!(parse_fev(&[]).is_none());
        assert!(parse_fev(b"not a fev").is_none());
        assert!(parse_fev(b"RIFF....FEV ").is_none());
        // Right tags but an out-of-range version.
        let mut b = vec![0u8; 0x18];
        b[0..4].copy_from_slice(b"RIFF");
        b[8..12].copy_from_slice(b"FEV ");
        b[12..16].copy_from_slice(b"FMT ");
        b[0x14..0x18].copy_from_slice(&0x0099_0000u32.to_le_bytes());
        assert!(parse_fev(&b).is_none());
    }

    #[test]
    fn reader_string_is_bounds_checked() {
        let mut r = Reader::new(&[3, 0, 0, 0, b'h', b'i', 0], 0);
        assert_eq!(r.string().as_deref(), Some("hi"));
        let mut r = Reader::new(&[0xff, 0xff, 0, 0], 0);
        assert!(r.string().is_none());
        let mut r = Reader::new(&[4, 0, 0, 0, b'a'], 0);
        assert!(r.string().is_none());
    }

    #[test]
    fn path_helpers() {
        assert_eq!(vfs_dir("sounds/explosions/explosions.fev"), "sounds/explosions");
        assert_eq!(vfs_dir("explosions.fev"), "");
        assert_eq!(file_stem("sounds/explosions/explosions.fev"), "explosions");
        assert_eq!(file_stem("Game/sounds/dlc2/x/explosions_dlc2.fev"), "explosions_dlc2");
    }

    #[test]
    fn script_refs_find_calls_and_skip_comments() {
        let src = r#"
            self:PlaySoundEvent("Sounds/explosions:exp_specific:grenade", pos);
            -- self:PlaySoundEvent("sounds/interface:hud:hud_blood", pos);
            Sound.LoadSound("TurnOffSound");
            self:PlaySoundEventEx("sounds/dlc2/explosions_dlc2:exp_specific:nuke", f);
        "#;
        let refs = script_event_refs(src);
        assert_eq!(
            refs,
            vec![
                "Sounds/explosions:exp_specific:grenade",
                "sounds/dlc2/explosions_dlc2:exp_specific:nuke"
            ]
        );
    }

    #[test]
    fn resolve_prefers_exact_group_then_unique_leaf() {
        let mut db = FevDb::new();
        let defs = vec![SoundDef {
            name: "/g/grenade".to_string(),
            entries: vec![SoundDefEntry {
                stream: "g/grenade_01.wav".to_string(),
                wave_bank: 0,
                subsound: 3,
            }],
        }];
        let events = vec![
            event("grenade", &["exp_specific"], vec![0]),
            event("mortar_fire", &["exp_airstrike_dlc2"], vec![0]),
        ];
        db.insert(
            empty_project("explosions", &["exp_grenade"], defs, events),
            "sounds/explosions/explosions.fev",
        );

        let exact = db.resolve("Sounds/explosions:exp_specific:grenade").expect("exact");
        assert_eq!(exact.path, "/exp_specific/grenade");
        assert_eq!(exact.sounds.len(), 1);
        assert_eq!(exact.sounds[0].fsb_path, "sounds/explosions/exp_grenade.fsb");
        assert_eq!(exact.sounds[0].subsound, 3);
        assert_eq!(exact.priority, 10);

        // Group renamed in the script: unique leaf fallback resolves it.
        let leaf = db
            .resolve("Sounds/explosions:exp_airstrike:mortar_fire")
            .expect("leaf fallback");
        assert_eq!(leaf.path, "/exp_airstrike_dlc2/mortar_fire");

        // Unknown event.
        assert!(db.resolve("Sounds/explosions:exp_specific:nope").is_none());
        // Not an event path.
        assert!(db.resolve("TurnOffSound").is_none());
    }

    fn game_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SOD_YOSE_DIR") {
            let p = PathBuf::from(dir);
            if p.join("Game/sounds.pak").is_file() {
                return Some(p);
            }
        }
        let p = PathBuf::from(
            "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE",
        );
        if p.join("Game/sounds.pak").is_file() {
            Some(p)
        } else {
            None
        }
    }

    /// End-to-end: mount the shipped paks, decode every FEV, then resolve the
    /// event names the shipped Lua scripts actually reference and report the
    /// resolve rate. Skipped silently when the game paks are absent.
    #[test]
    fn real_fev_and_script_resolution() {
        let Some(dir) = game_dir() else {
            return; // real data absent: skip silently
        };

        let mut vfs = Vfs::new();
        for pak in [
            "Game/gamedata.pak",
            "Game/sounds.pak",
            "Game/beltway/BeltwayU.pak",
            "Game/beltway/BeltwayC.pak",
            "Game/sandbox/BreakdownC.pak",
        ] {
            let p = dir.join(pak);
            if p.is_file() {
                let _ = vfs.mount(&p);
            }
        }
        assert!(vfs.pak_count() > 0, "expected at least one mounted pak");

        let mut db = FevDb::new();
        let loaded = db.load_vfs(&vfs);
        assert!(loaded >= 13, "expected >=13 FEV projects, got {loaded}");
        assert!(db.event_count() > 3000, "expected >3000 events, got {}", db.event_count());

        // Gather event paths referenced by the shipped Lua.
        let mut refs: Vec<String> = Vec::new();
        for path in vfs.paths_with_ext(".lua") {
            if let Some(bytes) = vfs.read(&path) {
                refs.extend(script_event_refs(&String::from_utf8_lossy(&bytes)));
            }
        }
        refs.sort();
        refs.dedup();
        assert!(refs.len() >= 20, "expected many script event refs, got {}", refs.len());

        let mut resolved = 0usize;
        let mut checked = 0usize;
        let mut found = 0usize;
        for r in &refs {
            let Some(ev) = db.resolve(r) else {
                continue;
            };
            resolved += 1;
            for s in &ev.sounds {
                checked += 1;
                if sound_exists(&vfs, s) {
                    found += 1;
                }
            }
        }
        let rate = resolved as f64 / refs.len() as f64;
        let sound_rate = if checked == 0 { 1.0 } else { found as f64 / checked as f64 };
        println!(
            "FEV: {} projects, {} events; resolved {resolved}/{} script refs ({:.1}%), \
             {found}/{checked} FSB subsounds exist ({:.1}%)",
            db.len(),
            db.event_count(),
            refs.len(),
            rate * 100.0,
            sound_rate * 100.0
        );
        assert!(rate >= 0.8, "event resolve rate too low: {rate:.3} ({resolved}/{})", refs.len());
        assert!(checked > 0, "expected at least one resolved FSB reference");
        assert!(sound_rate >= 0.9, "FSB sub-sound existence too low: {sound_rate:.3}");
    }
}
