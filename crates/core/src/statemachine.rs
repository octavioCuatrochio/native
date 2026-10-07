//! Undead Labs `.sm` state machines (`ULSM`) and CryEngine `.chrparams`
//! animation-list includes.
//!
//! ## `.sm` / `ULSM` byte layout (recovered)
//!
//! Files live under `gamedata/libs/statemachines/<actor>/general.<plat>.sm`
//! (e.g. `player/general.win.sm`, `player/general.xbox360.sm`). The `.win` and
//! `.xboxone` builds are byte-identical little-endian; `.xbox360` is the same
//! layout with big-endian integers and floats. All observed files start with
//! the ASCII magic `ULSM`.
//!
//! The parser was recovered from `FUN_0158c3d0` (0x158c3d0, the actual
//! deserialiser) and its caller `FUN_0158c9b0` (0x158c9b0), reached from
//! `ULActor::CreateStateMachine` 0x1385b30 -> `FUN_0158df30` (0x158df30).
//! The runtime that consumes the result is `SmTick` (0x158e990) and the
//! per-block helpers `FUN_016955c0`/`FUN_01698870`/`FUN_01693930`/`FUN_01697a20`
//! /`FUN_01698980`/`FUN_0169e0a0`.
//!
//! ### Header (little-endian; big-endian words are byte-swapped)
//!
//! | offset | size | meaning |
//! |-------:|-----:|---------|
//! | `0x00` | 4 | magic `"ULSM"` (bytes unchanged on XB360) |
//! | `0x04` | 4 | endian marker: `0` little-endian, `1` big-endian |
//! | `0x08` | 4 | constant bytes `00 00 00 0e` |
//! | `0x0c` | 16 | fixed 16-byte signature GUID (`2ed4290c3124f610f5a8721f5ffb50bb`) |
//! | `0x1c` | 4 | constant `0xa10cebe2` (unused by the parser we traced) |
//! | `0x20` | 4 | **state count `N`** |
//! | `0x24` | 4 | **block count** (records of 0x48 bytes) |
//! | `0x28` | 4 | **start state index** |
//! | `0x2c` | 4 | state-name blob size in bytes |
//! | `0x30` | 4 | file offset of the fragment section (`0x38` records) |
//! | `0x34` | 4 | action-id count (`d_34`, 4-byte records) |
//! | `0x38` | 4 | fragment count (`e_38`, 0x4c-byte records) |
//! | `0x3c` | 4 | trigger-type bytes (`g_3c`) |
//! | `0x40` | 4 | filter count (`i_40`, 8-byte records) |
//! | `0x44` | 4 | transition-candidate count (`j_44`, 12-byte records) |
//! | `0x48` | 4 | transition-condition count (`k_48`, 4-byte records) |
//! | `0x4c` | 4 | entity-type bytes (`m_4c`) |
//! | `0x50` | 4 | transform-type bytes (`o_50`) |
//! | `0x54` | 4 | block-string/bytecode blob bytes (`c_54`) |
//! | `0x58` | 4 | trigger-data bytes (`f_58`) |
//! | `0x5c` | 4 | filter-data bytes (`h_5c`) |
//! | `0x60` | 4 | action-data bytes (`l_60`) |
//! | `0x64` | 4 | transform-data bytes (`n_64`) |
//! | `0x68` | 4 | block-index byte size (`q_68`; used even) |
//! | `0x6c` | 4 | unknown (varies per machine) |
//! | `0x70` | 4 | unknown (varies per machine) |
//! | `0x74` | 64 | **machine name**, NUL-padded fixed field |
//!
//! ### Sections
//!
//! After the fixed header the parser bump-allocates 18 sections in order from a
//! cursor starting at `0xb4` (`FUN_0131d6f0`). Each is aligned up before it is
//! placed; the cursor must land exactly on end-of-file. Only the first two
//! sections use a small alignment, everything else is 128-byte aligned.
//!
//! | # | name | count source | elem | align |
//! |--:|------|--------------|-----:|------:|
//! | 0 | `state_name_offsets` | `0x20` (`N`) | 4 | 4 |
//! | 1 | `state_names` | `0x2c` bytes | 1 | 1 |
//! | 2 | `block_strings` | `0x54` bytes | 1 | 0x80 |
//! | 3 | `action_ids` | `0x34` | 4 | 0x80 |
//! | 4 | `fragments` | `0x38` | 0x4c | 0x80 |
//! | 5 | `trigger_data` | `0x58` bytes | 1 | 0x80 |
//! | 6 | `trigger_types` | `0x3c` bytes | 1 | 0x80 |
//! | 7 | `filter_data` | `0x5c` bytes | 1 | 0x80 |
//! | 8 | `filters` | `0x40` | 8 | 0x80 |
//! | 9 | `transition_candidates` | `0x44` | 0xc | 0x80 |
//! | 10 | `transition_conditions` | `0x48` | 4 | 0x80 |
//! | 11 | `action_data` | `0x60` bytes | 1 | 0x80 |
//! | 12 | `entity_types` | `0x4c` bytes | 1 | 0x80 |
//! | 13 | `transform_data` | `0x64` bytes | 1 | 0x80 |
//! | 14 | `transform_types` | `0x50` bytes | 1 | 0x80 |
//! | 15 | `blocks` | `0x24` | 0x48 | 0x80 |
//! | 16 | `block_indices` | `0x68 & ~1` bytes | 1 | 0x80 |
//! | 17 | `state_headers` | `N` | 6 | 0x80 |
//!
//! ### Records
//!
//! * `state_headers` (6 bytes): `u16 fragmentIndex` (`0xffff` = none),
//!   `u16 blockListBase` (word index into `block_indices`),
//!   `u8 blockCount` (low 7 bits; bit 7 is a flag), `u8 reserved` (0).
//! * `block_indices` (`q_68`, `u16`): concatenated per-state lists of
//!   `blocks` indices.
//! * `fragments` (0x4c bytes): `f32 speed`, `u32 flags`, `u16 tag`, then a
//!   NUL-terminated animation/fragment name at `+0x0a`; remaining bytes zero.
//!   `SmTick` 0x158e990 plays this clip and unpacks the six low bits of `tag`.
//! * `blocks` (0x48 bytes): one "layer" of a state. Fields (all traced through
//!   the helpers above): `+0x00 u32` code offset into `block_strings`,
//!   `+0x04 u32` action-data offset, `+0x08 u32` trigger-data offset,
//!   `+0x0c u32` filter-data offset, `+0x10 u32` first transition candidate,
//!   `+0x14 u32` entity-data offset, `+0x18 u32` transform-data offset,
//!   `+0x1c u32` transform runtime offset, `+0x20 u16` action base,
//!   `+0x22 u16` trigger base, `+0x24 u16` filter base, `+0x26 u16` condition
//!   base, `+0x28 u16` entity base, `+0x2a u16` transform base,
//!   `+0x2c u16` code length, `+0x3a u16` transform count,
//!   `+0x3e u8` action count, `+0x3f u8` trigger count, `+0x40 u8` filter count,
//!   `+0x41 u8` secondary condition count, `+0x42 u8` secondary candidate
//!   delta, `+0x43 u8` primary condition count, `+0x44 u8` entity count.
//! * `transition_conditions` (`k_48`, 4 bytes): `u16 condition` (index into the
//!   per-frame trigger-output flag array `DAT_01ee3170`), `u8 candidateCount`,
//!   `u8 reserved`.
//! * `transition_candidates` (`j_44`, 12 bytes): `f32 blendIn`, `f32 blendOut`,
//!   `u16 targetState`, `u16 weight`, `u8 flags` (at `+0xb`).
//! * `action_ids` (`d_34`, 4 bytes): `i16 actionId`, `u8 descriptorId`,
//!   `u8 flags` (an interpreter opcode stream, `FUN_01693930`).
//! * `filters` (`i_40`, 8 bytes): `u32 dataOffset` into `filter_data`,
//!   `u16 dataSize`, `u8 filterType` (1..=16), `u8 reserved`.
//!
//! ### Transition evaluation (`FUN_01698870` -> `FUN_01698770`, `SmTick`)
//!
//! For the active state, every block is visited in reverse. Two condition lists
//! are tried: the primary list (`+0x26`/`+0x43`, candidates at `+0x10`) and the
//! secondary list (`condition_base + condition_count`/`+0x41`, candidates at
//! `candidate_base + +0x42`). For a condition whose trigger-output flag is set,
//! one of its `candidateCount` candidates is chosen by weighted random; the
//! chosen candidate's `targetState` is written to the instance's next-state
//! slot (offset `0xe`) and its two floats to `0x10`/`0x14`. `SmTick` then
//! switches the machine and plays the target state's fragment.
//!
//! The per-block sub-tables are now decoded too: the payload blobs
//! (trigger / entity / transform sources, filters, actions) are walked with the
//! descriptor tables in [`crate::sm_tables`] and returned in
//! [`StateMachine::payloads`] (see [`crate::sm_payload`]). What remains opaque
//! is header `0x1c`, `0x6c`, `0x70`, the script bytecode inside
//! `block_strings`, and the uninterpreted fields of most individual trigger
//! evaluators.
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
    /// Index into [`StateMachine::fragments`] of the clip this state plays
    /// (`None` for the `0xffff` sentinel).
    pub fragment: Option<u32>,
    /// Indices into [`StateMachine::blocks`] that make up this state.
    pub blocks: Vec<u32>,
}

/// A playable animation fragment / clip entry (`fragments` section).
#[derive(Debug, Clone, PartialEq)]
pub struct SmFragment {
    /// NUL-terminated clip name (e.g. `Basic_wander_walk_fast`).
    pub name: String,
    /// Playback speed multiplier (`1.0` in every observed record).
    pub speed: f32,
    /// `u32` at `+0x04`; `0`/`1` in observed records.
    pub flags: u32,
    /// `u16` at `+0x08`; `SmTick` unpacks six low bits as clip flags.
    pub tag: u16,
}

/// One block ("layer") of a state (`blocks` section, 0x48 bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmBlock {
    pub code_offset: u32,
    pub action_data_offset: u32,
    pub trigger_data_offset: u32,
    pub filter_data_offset: u32,
    pub candidate_base: u32,
    pub entity_data_offset: u32,
    pub transform_data_offset: u32,
    pub transform_runtime_offset: u32,
    pub action_base: u16,
    pub trigger_base: u16,
    pub filter_base: u16,
    pub condition_base: u16,
    pub entity_base: u16,
    pub transform_base: u16,
    pub code_length: u16,
    pub transform_count: u16,
    pub action_count: u8,
    pub trigger_count: u8,
    pub filter_count: u8,
    pub condition_count_b: u8,
    pub candidate_delta_b: u8,
    pub condition_count: u8,
    pub entity_count: u8,
}

/// Which of a block's two condition lists a transition came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmTransitionList {
    /// List at block `+0x26`/`+0x43`, candidates at `+0x10`.
    Primary,
    /// List at `+0x26 + +0x43`/`+0x41`, candidates at `+0x10 + +0x42`.
    Secondary,
}

/// A single state-to-state edge selected by a trigger output.
#[derive(Debug, Clone, PartialEq)]
pub struct SmTransition {
    /// Source state index.
    pub source: u32,
    /// Block of the source state that carries this edge.
    pub block: u32,
    /// Trigger-output id (`k_48` condition word).
    pub condition: u16,
    /// Target state index (`j_44` candidate `targetState`).
    pub target: u32,
    /// Selection weight among the condition's candidates.
    pub weight: u16,
    /// `f32` blend parameter copied to the instance.
    pub blend_in: f32,
    /// Second `f32` blend parameter.
    pub blend_out: f32,
    /// Candidate flags byte (`j_44` `+0xb`); `SmTick` copies `flags != 0` into
    /// the instance's `+0x19` latch when the candidate is selected.
    pub flags: u8,
    /// Which of the block's two condition lists produced this edge.
    pub list: SmTransitionList,
}

/// A mapped section of the `.sm` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmSection {
    pub name: &'static str,
    pub offset: u32,
    pub size: u32,
    pub count: u32,
    pub elem_size: u32,
    pub align: u32,
}

/// A parsed `.sm` (ULSM) state machine.
#[derive(Debug, Clone, PartialEq)]
pub struct StateMachine {
    /// Machine name from the fixed header field, e.g. `Player`.
    pub name: String,
    /// States in file order.
    pub states: Vec<SmState>,
    /// Index of the initial state (header `0x28`).
    pub start_state: u32,
    /// Fragment / clip table.
    pub fragments: Vec<SmFragment>,
    /// Block ("layer") table.
    pub blocks: Vec<SmBlock>,
    /// Flattened state-to-state transitions (see [`SmTransition`]).
    pub transitions: Vec<SmTransition>,
    /// All mapped sections, in file order.
    pub sections: Vec<SmSection>,
    /// Records decoded from the trigger / entity / transform / filter / action
    /// payload blobs, plus coverage accounting (see [`crate::sm_payload`]).
    pub payloads: crate::sm_payload::SmPayloads,
    /// `true` for the big-endian XB360 build.
    pub big_endian: bool,
}

/// Offset of the state-count word (fallback path).
const COUNT_OFF: usize = 0x20;
/// Offset of the fixed 64-byte machine-name field.
const NAME_OFF: usize = 0x74;
/// Length of the fixed machine-name field.
const NAME_BYTES: usize = 0x40;
/// Offset of the state-name start-offset array (fallback path).
const OFFSET_ARR: usize = 0xb4;
/// Size of the fixed header; the section arena starts here.
const HDR_SIZE: usize = 0xb4;
/// Upper bound on a plausible state count (guards against bad endian reads).
const MAX_STATES: usize = 4_000_000;
/// `fragmentIndex` sentinel meaning "no clip".
const NO_FRAGMENT: u16 = 0xffff;

/// Fixed 16-byte signature stored at header `0x0c` (from the exe at
/// `0x18919d0`, compared by `FUN_0158c3d0`).
const GUID: [u8; 16] = [
    0x2e, 0xd4, 0x29, 0x0c, 0x31, 0x24, 0xf6, 0x10, 0xf5, 0xa8, 0x72, 0x1f, 0x5f, 0xfb, 0x50, 0xbb,
];

// Section indices into `StateMachine::sections`.
const SEC_NAME_OFFSETS: usize = 0;
const SEC_NAME_BLOB: usize = 1;
const SEC_FRAGMENTS: usize = 4;
const SEC_TRANSITION_CANDIDATES: usize = 9;
const SEC_TRANSITION_CONDITIONS: usize = 10;
const SEC_BLOCKS: usize = 15;
const SEC_BLOCK_INDICES: usize = 16;
const SEC_STATE_HEADERS: usize = 17;

/// One bump-allocated section description.
struct SecSpec {
    name: &'static str,
    count: usize,
    elem: usize,
    align: usize,
}

fn read_u32(bytes: &[u8], off: usize, be: bool) -> Option<u32> {
    let s = bytes.get(off..off.checked_add(4)?)?;
    let a = [s[0], s[1], s[2], s[3]];
    Some(if be {
        u32::from_be_bytes(a)
    } else {
        u32::from_le_bytes(a)
    })
}

fn read_u16(bytes: &[u8], off: usize, be: bool) -> Option<u16> {
    let s = bytes.get(off..off.checked_add(2)?)?;
    let a = [s[0], s[1]];
    Some(if be {
        u16::from_be_bytes(a)
    } else {
        u16::from_le_bytes(a)
    })
}

fn read_f32(bytes: &[u8], off: usize, be: bool) -> Option<f32> {
    Some(f32::from_bits(read_u32(bytes, off, be)?))
}

/// Round `v` up to a power-of-two `align`.
fn align_up(v: usize, align: usize) -> Option<usize> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    v.checked_add(align - 1).map(|x| x & !(align - 1))
}

/// Read a NUL-terminated lossy string at `base + rel`, capped at `max` bytes.
fn read_cstr_rel(bytes: &[u8], base: usize, rel: usize, max: usize) -> String {
    let Some(start) = base.checked_add(rel) else {
        return String::new();
    };
    let Some(rest) = bytes.get(start..) else {
        return String::new();
    };
    let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
    String::from_utf8_lossy(&rest[..end.min(max)]).into_owned()
}

/// Read a NUL-terminated lossy string at `start`, capped at `max` bytes.
fn read_cstr(bytes: &[u8], start: usize, max: usize) -> String {
    read_cstr_rel(bytes, start, 0, max)
}

/// The 18 sections the deserialiser lays out, in file order. `hw` is the
/// header as little/big-endian words indexed by `byte_offset / 4`.
fn section_specs(n: usize, hw: &[u32]) -> Vec<SecSpec> {
    let g = |byte: usize| hw.get(byte / 4).copied().unwrap_or(0) as usize;
    vec![
        SecSpec { name: "state_name_offsets", count: n, elem: 4, align: 4 },
        SecSpec { name: "state_names", count: g(0x2c), elem: 1, align: 1 },
        SecSpec { name: "block_strings", count: g(0x54), elem: 1, align: 0x80 },
        SecSpec { name: "action_ids", count: g(0x34), elem: 4, align: 0x80 },
        SecSpec { name: "fragments", count: g(0x38), elem: 0x4c, align: 0x80 },
        SecSpec { name: "trigger_data", count: g(0x58), elem: 1, align: 0x80 },
        SecSpec { name: "trigger_types", count: g(0x3c), elem: 1, align: 0x80 },
        SecSpec { name: "filter_data", count: g(0x5c), elem: 1, align: 0x80 },
        SecSpec { name: "filters", count: g(0x40), elem: 8, align: 0x80 },
        SecSpec { name: "transition_candidates", count: g(0x44), elem: 0xc, align: 0x80 },
        SecSpec { name: "transition_conditions", count: g(0x48), elem: 4, align: 0x80 },
        SecSpec { name: "action_data", count: g(0x60), elem: 1, align: 0x80 },
        SecSpec { name: "entity_types", count: g(0x4c), elem: 1, align: 0x80 },
        SecSpec { name: "transform_data", count: g(0x64), elem: 1, align: 0x80 },
        SecSpec { name: "transform_types", count: g(0x50), elem: 1, align: 0x80 },
        SecSpec { name: "blocks", count: g(0x24), elem: 0x48, align: 0x80 },
        SecSpec { name: "block_indices", count: g(0x68) & !1, elem: 1, align: 0x80 },
        SecSpec { name: "state_headers", count: n, elem: 6, align: 0x80 },
    ]
}

/// Place every section into a `len`-byte buffer; returns `(offset, size)` per
/// section. Fails unless the final cursor lands exactly on `len` (the
/// deserialiser's "all bytes consumed" check at `0x158c84f`).
fn place_sections(specs: &[SecSpec], len: usize) -> Option<Vec<(usize, usize)>> {
    let mut cursor = HDR_SIZE;
    let mut out = Vec::with_capacity(specs.len());
    for s in specs {
        let off = align_up(cursor, s.align)?;
        let size = s.count.checked_mul(s.elem)?;
        let end = off.checked_add(size)?;
        if end > len {
            return None;
        }
        out.push((off, size));
        cursor = end;
    }
    if cursor != len {
        return None;
    }
    Some(out)
}

/// Strict full decode with one endianness.
fn parse_full(bytes: &[u8], be: bool) -> Option<StateMachine> {
    if bytes.len() < HDR_SIZE || bytes.get(..4)? != b"ULSM" {
        return None;
    }
    // 0x04: endian marker (0 little, 1 big).
    let marker = read_u32(bytes, 0x04, be)?;
    if marker != u32::from(be) {
        return None;
    }
    // 0x08: constant bytes 00 00 00 0e.
    if bytes.get(0x08..0x0c)? != &[0u8, 0, 0, 0x0e][..] {
        return None;
    }
    // 0x0c: fixed signature GUID.
    if bytes.get(0x0c..0x1c)? != &GUID[..] {
        return None;
    }

    let n = read_u32(bytes, COUNT_OFF, be)? as usize;
    if n == 0 || n > MAX_STATES {
        return None;
    }
    let mut hw = Vec::with_capacity(HDR_SIZE / 4);
    for i in 0..HDR_SIZE / 4 {
        hw.push(read_u32(bytes, i * 4, be)?);
    }
    let specs = section_specs(n, &hw);
    let placed = place_sections(&specs, bytes.len())?;

    let mut sections = Vec::with_capacity(specs.len());
    for (spec, &(off, size)) in specs.iter().zip(placed.iter()) {
        sections.push(SmSection {
            name: spec.name,
            offset: off as u32,
            size: size as u32,
            count: spec.count as u32,
            elem_size: spec.elem as u32,
            align: spec.align as u32,
        });
    }

    let name_offsets = &sections[SEC_NAME_OFFSETS];
    let name_blob = &sections[SEC_NAME_BLOB];

    // Fragments / clips.
    let fsec = &sections[SEC_FRAGMENTS];
    let mut fragments = Vec::with_capacity(fsec.count as usize);
    for i in 0..fsec.count as usize {
        let o = fsec.offset as usize + i * 0x4c;
        fragments.push(SmFragment {
            name: read_cstr_rel(bytes, o + 0x0a, 0, 0x4c - 0x0a),
            speed: read_f32(bytes, o, be)?,
            flags: read_u32(bytes, o + 4, be)?,
            tag: read_u16(bytes, o + 8, be)?,
        });
    }

    // Block index table (q_68).
    let qsec = &sections[SEC_BLOCK_INDICES];
    let mut q = Vec::with_capacity(qsec.count as usize / 2);
    for i in 0..qsec.count as usize / 2 {
        q.push(read_u16(bytes, qsec.offset as usize + i * 2, be)?);
    }

    // Blocks (p_24).
    let bsec = &sections[SEC_BLOCKS];
    let mut blocks = Vec::with_capacity(bsec.count as usize);
    for i in 0..bsec.count as usize {
        let o = bsec.offset as usize + i * 0x48;
        blocks.push(SmBlock {
            code_offset: read_u32(bytes, o, be)?,
            action_data_offset: read_u32(bytes, o + 4, be)?,
            trigger_data_offset: read_u32(bytes, o + 8, be)?,
            filter_data_offset: read_u32(bytes, o + 0x0c, be)?,
            candidate_base: read_u32(bytes, o + 0x10, be)?,
            entity_data_offset: read_u32(bytes, o + 0x14, be)?,
            transform_data_offset: read_u32(bytes, o + 0x18, be)?,
            transform_runtime_offset: read_u32(bytes, o + 0x1c, be)?,
            action_base: read_u16(bytes, o + 0x20, be)?,
            trigger_base: read_u16(bytes, o + 0x22, be)?,
            filter_base: read_u16(bytes, o + 0x24, be)?,
            condition_base: read_u16(bytes, o + 0x26, be)?,
            entity_base: read_u16(bytes, o + 0x28, be)?,
            transform_base: read_u16(bytes, o + 0x2a, be)?,
            code_length: read_u16(bytes, o + 0x2c, be)?,
            transform_count: read_u16(bytes, o + 0x3a, be)?,
            action_count: bytes.get(o + 0x3e).copied().unwrap_or(0),
            trigger_count: bytes.get(o + 0x3f).copied().unwrap_or(0),
            filter_count: bytes.get(o + 0x40).copied().unwrap_or(0),
            condition_count_b: bytes.get(o + 0x41).copied().unwrap_or(0),
            candidate_delta_b: bytes.get(o + 0x42).copied().unwrap_or(0),
            condition_count: bytes.get(o + 0x43).copied().unwrap_or(0),
            entity_count: bytes.get(o + 0x44).copied().unwrap_or(0),
        });
    }

    // Transition conditions (k_48): (condition, candidateCount).
    let ksec = &sections[SEC_TRANSITION_CONDITIONS];
    let mut k48 = Vec::with_capacity(ksec.count as usize);
    for i in 0..ksec.count as usize {
        let o = ksec.offset as usize + i * 4;
        k48.push((read_u16(bytes, o, be)?, bytes.get(o + 2).copied().unwrap_or(0)));
    }

    // Transition candidates (j_44): (blendIn, blendOut, target, weight).
    let jsec = &sections[SEC_TRANSITION_CANDIDATES];
    let mut j44 = Vec::with_capacity(jsec.count as usize);
    for i in 0..jsec.count as usize {
        let o = jsec.offset as usize + i * 12;
        j44.push((
            read_f32(bytes, o, be)?,
            read_f32(bytes, o + 4, be)?,
            read_u16(bytes, o + 8, be)?,
            read_u16(bytes, o + 0xa, be)?,
            bytes.get(o + 0xb).copied().unwrap_or(0),
        ));
    }

    // States + their block lists.
    let shsec = &sections[SEC_STATE_HEADERS];
    let mut states = Vec::with_capacity(n.min(8192));
    for i in 0..n {
        let ho = shsec.offset as usize + i * 6;
        let fragment = read_u16(bytes, ho, be)?;
        let qbase = read_u16(bytes, ho + 2, be)? as usize;
        let flags = bytes.get(ho + 4).copied().unwrap_or(0);
        let nblocks = (flags & 0x7f) as usize;
        let mut block_ids = Vec::with_capacity(nblocks.min(256));
        for b in 0..nblocks {
            if let Some(&bi) = q.get(qbase + b) {
                block_ids.push(bi as u32);
            }
        }
        let offset = read_u32(bytes, name_offsets.offset as usize + i * 4, be)? as usize;
        states.push(SmState {
            name: read_cstr_rel(bytes, name_blob.offset as usize, offset, name_blob.size as usize),
            index: i as u32,
            fragment: if fragment == NO_FRAGMENT {
                None
            } else {
                Some(fragment as u32)
            },
            blocks: block_ids,
        });
    }

    // Flatten transitions. Indices are validated against their tables so a
    // truncated/odd file can never panic.
    let mut transitions = Vec::new();
    for (si, st) in states.iter().enumerate() {
        for &bi in &st.blocks {
            let Some(blk) = blocks.get(bi as usize) else {
                continue;
            };
            let mut j = blk.candidate_base as usize;
            for k in 0..blk.condition_count as usize {
                let Some(&(condition, nc)) = k48.get(blk.condition_base as usize + k) else {
                    break;
                };
                for _ in 0..nc {
                    let Some(&(blend_in, blend_out, target, weight, flags)) = j44.get(j) else {
                        break;
                    };
                    transitions.push(SmTransition {
                        source: si as u32,
                        block: bi,
                        condition,
                        target: target as u32,
                        weight,
                        blend_in,
                        blend_out,
                        flags,
                        list: SmTransitionList::Primary,
                    });
                    j += 1;
                }
            }
            let mut j2 = blk.candidate_base as usize + blk.candidate_delta_b as usize;
            let kbase = blk.condition_base as usize + blk.condition_count as usize;
            for k in 0..blk.condition_count_b as usize {
                let Some(&(condition, nc)) = k48.get(kbase + k) else {
                    break;
                };
                for _ in 0..nc {
                    let Some(&(blend_in, blend_out, target, weight, flags)) = j44.get(j2) else {
                        break;
                    };
                    transitions.push(SmTransition {
                        source: si as u32,
                        block: bi,
                        condition,
                        target: target as u32,
                        weight,
                        blend_in,
                        blend_out,
                        flags,
                        list: SmTransitionList::Secondary,
                    });
                    j2 += 1;
                }
            }
        }
    }

    let mut sm = StateMachine {
        name: read_cstr(bytes, NAME_OFF, NAME_BYTES),
        states,
        start_state: hw.get(0x28 / 4).copied().unwrap_or(0),
        fragments,
        blocks,
        transitions,
        sections,
        payloads: crate::sm_payload::SmPayloads::empty(0),
        big_endian: be,
    };
    // Second pass: decode the opaque payload blobs now that the structural
    // tables are available.
    sm.payloads = crate::sm_payload::decode(bytes, &sm);
    Some(sm)
}

/// Lenient name-only decode (the pre-existing behaviour), used when the strict
/// full parse does not add up. Returns the machine name plus as many states as
/// could be read.
fn parse_names_only(bytes: &[u8], be: bool) -> Option<StateMachine> {
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
        if i > 0 && start < prev {
            break;
        }
        prev = start;
        let Some(rest) = bytes.get(blob.checked_add(start)?..) else {
            break;
        };
        match rest.iter().position(|&c| c == 0) {
            Some(len) => states.push(SmState {
                name: String::from_utf8_lossy(&rest[..len]).into_owned(),
                index: i as u32,
                fragment: None,
                blocks: Vec::new(),
            }),
            None => {
                if !rest.is_empty() {
                    states.push(SmState {
                        name: String::from_utf8_lossy(rest).into_owned(),
                        index: i as u32,
                        fragment: None,
                        blocks: Vec::new(),
                    });
                }
                break;
            }
        }
    }

    if states.is_empty() {
        return None;
    }
    Some(StateMachine {
        name,
        states,
        start_state: 0,
        fragments: Vec::new(),
        blocks: Vec::new(),
        transitions: Vec::new(),
        sections: Vec::new(),
        payloads: crate::sm_payload::SmPayloads::empty(bytes.len()),
        big_endian: be,
    })
}

/// Best-effort parse of a `ULSM` `.sm` file.
///
/// Returns `None` for a non-`ULSM` buffer (or one too truncated to hold the
/// header). On success the machine name, states, fragments, blocks and
/// transitions are recovered. Both little-endian (PC/XB1) and big-endian
/// (XB360) layouts are tried; the candidate with the most states wins. If the
/// strict section table does not add up, the state names are still returned
/// with the structural fields left empty. Never panics.
pub fn parse_sm(bytes: &[u8]) -> Option<StateMachine> {
    if bytes.len() < HDR_SIZE || bytes.get(..4) != Some(&b"ULSM"[..]) {
        return None;
    }
    let pick = |a: Option<StateMachine>, b: Option<StateMachine>| match (a, b) {
        (Some(a), Some(b)) => Some(if b.states.len() > a.states.len() { b } else { a }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    let full = pick(parse_full(bytes, false), parse_full(bytes, true));
    if full.is_some() {
        return full;
    }
    pick(parse_names_only(bytes, false), parse_names_only(bytes, true))
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

    fn real_sm(actor: &str, plat: &str) -> Option<Vec<u8>> {
        let rel = format!(
            "/../../../extracted-game/gamedata/libs/statemachines/{actor}/general.{plat}.sm"
        );
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "{}");
        let candidates = [
            p.replace("{}", &rel),
            format!("/media/workpc/1f162086-ac53-4c5e-96a2-d1dc72b0fa2d/home/battlestation/Desktop/sod-re/extracted-game/gamedata/libs/statemachines/{actor}/general.{plat}.sm"),
        ];
        candidates.iter().find_map(|p| std::fs::read(p).ok())
    }

    fn real_player_sm() -> Option<Vec<u8>> {
        real_sm("player", "win")
    }

    /// Build a minimal valid 1-state machine with no fragments/blocks.
    fn synth_sm() -> Vec<u8> {
        let n = 1usize;
        let mut hw = vec![0u32; HDR_SIZE / 4];
        hw[0x20 / 4] = n as u32;
        hw[0x28 / 4] = 0;
        hw[0x2c / 4] = 5; // "Idle\0"
        let specs = section_specs(n, &hw);
        let placed = place_sections(&specs, {
            // Recompute the total the same way `place_sections` will.
            let mut cursor = HDR_SIZE;
            for s in &specs {
                cursor = align_up(cursor, s.align).unwrap() + s.count * s.elem;
            }
            cursor
        })
        .unwrap();
        let mut buf = vec![0u8; placed.iter().map(|&(o, s)| o + s).max().unwrap()];
        for (i, w) in hw.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        buf[..4].copy_from_slice(b"ULSM");
        buf[0x04..0x08].copy_from_slice(&0u32.to_le_bytes());
        buf[0x08..0x0c].copy_from_slice(&[0, 0, 0, 0x0e]);
        buf[0x0c..0x1c].copy_from_slice(&GUID);
        buf[NAME_OFF..NAME_OFF + 5].copy_from_slice(b"Test\0");
        let (noff, _) = placed[SEC_NAME_OFFSETS];
        let (nb, _) = placed[SEC_NAME_BLOB];
        let (sh, _) = placed[SEC_STATE_HEADERS];
        buf[noff..noff + 4].copy_from_slice(&0u32.to_le_bytes());
        buf[nb..nb + 5].copy_from_slice(b"Idle\0");
        // fragment = 0xffff, block list base = 0, count = 0.
        buf[sh..sh + 6].copy_from_slice(&[0xff, 0xff, 0, 0, 0, 0]);
        buf
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
    fn synthetic_sm_parses() {
        let buf = synth_sm();
        let sm = parse_sm(&buf).expect("synthetic sm should parse");
        assert_eq!(sm.name, "Test");
        assert!(!sm.big_endian);
        assert_eq!(sm.start_state, 0);
        assert_eq!(sm.states.len(), 1);
        assert_eq!(sm.states[0].name, "Idle");
        assert_eq!(sm.states[0].fragment, None);
        assert!(sm.states[0].blocks.is_empty());
        assert!(sm.fragments.is_empty());
        assert!(sm.blocks.is_empty());
        assert!(sm.transitions.is_empty());
        assert_eq!(sm.sections.len(), 18);
        assert_eq!(sm.sections[SEC_NAME_OFFSETS].offset, 0xb4);
        assert_eq!(sm.sections[SEC_NAME_BLOB].offset, 0xb8);
        assert_eq!(sm.sections[SEC_STATE_HEADERS].offset, 0x100);
    }

    #[test]
    fn real_player_sm_if_present() {
        let Some(bytes) = real_player_sm() else {
            return; // data not present in this checkout: skip silently
        };
        let sm = parse_sm(&bytes).expect("real player sm should parse");
        assert_eq!(sm.name, "Player");
        assert_eq!(sm.states.len(), 689);
        assert_eq!(sm.states[0].name, "GraphDriven.Dead.Idle");
        assert_eq!(sm.states[0].index, 0);
        assert_eq!(sm.start_state, 25);
        assert_eq!(sm.states[25].name, "GraphDriven.Observer");
        assert!(!sm.fragments.is_empty());
        assert!(!sm.blocks.is_empty());
        assert!(!sm.transitions.is_empty());
        // Every index must be internally consistent.
        for st in &sm.states {
            if let Some(f) = st.fragment {
                assert!((f as usize) < sm.fragments.len());
            }
            for &b in &st.blocks {
                assert!((b as usize) < sm.blocks.len());
            }
        }
        for t in &sm.transitions {
            assert!((t.source as usize) < sm.states.len());
            assert!((t.target as usize) < sm.states.len());
            assert!((t.block as usize) < sm.blocks.len());
        }
    }

    #[test]
    fn real_bloater_matches_known_graph() {
        let Some(bytes) = real_sm("freak_bloater", "win") else {
            return;
        };
        let sm = parse_sm(&bytes).expect("bloater should parse");
        assert_eq!(sm.name, "Freak_Bloater");
        assert_eq!(sm.states.len(), 13);
        assert_eq!(sm.start_state, 1);
        assert_eq!(sm.fragments.len(), 9);
        // Observer reaches Walk.Normal (index 2) and Climb.Start (index 6).
        let observer: Vec<u32> = sm
            .transitions
            .iter()
            .filter(|t| t.source == 1)
            .map(|t| t.target)
            .collect();
        assert!(observer.contains(&2));
        assert!(observer.contains(&0));
        // Randomizer (index 9) only picks the two burning runs.
        let rand: Vec<u32> = sm
            .transitions
            .iter()
            .filter(|t| t.source == 9)
            .map(|t| t.target)
            .collect();
        assert_eq!(rand, vec![10, 11]);
    }

    #[test]
    fn real_xbox360_is_big_endian() {
        let Some(bytes) = real_sm("freak_bloater", "xbox360") else {
            return;
        };
        let sm = parse_sm(&bytes).expect("xbox360 bloater should parse");
        assert!(sm.big_endian);
        assert_eq!(sm.states.len(), 13);
        assert_eq!(sm.fragments.len(), 9);
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
