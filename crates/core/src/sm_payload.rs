//! Decode the opaque `.sm` payload blobs (trigger / entity / transform / filter
//! / action operands) into typed records.
//!
//! ## What is decoded
//!
//! `parse_sm` (in [`crate::statemachine`]) already maps every byte of a `ULSM`
//! file to one of 18 sections, but leaves eight of them as blobs:
//! `block_strings` (bytecode), `trigger_data`, `trigger_types`, `filter_data`,
//! `action_data`, `entity_types`, `transform_data`, `transform_types`.
//!
//! Three runtime walkers consume the *type arrays* and *operand blobs*, pairing
//! each type id with a descriptor from [`crate::sm_tables`] that gives the
//! record's static size and alignment. Recovered from:
//!
//! | consumer | walker | type array | operand blob | block fields |
//! |---|---|---|---|---|
//! | trigger source | `FUN_0169e0a0` 0x169e0a0 | `transform_types` | `transform_data` | base `+0x2a`, count `+0x3a`, offset `+0x18` |
//! | entity source | `FUN_016955c0` 0x16955c0 | `trigger_types` | `trigger_data` | base `+0x22`, count `+0x3f`, offset `+0x08` |
//! | transform source | `FUN_01698980` 0x1698980 | `entity_types` | `action_data` | base `+0x28`, count `+0x44`, offset `+0x14` |
//!
//! (The section names above are the historical labels from
//! [`crate::statemachine`]; the descriptor tables are named from the engine's
//! own registration strings, so the pairing is a rotation of those labels. The
//! walk was validated against every block of all seven shipped machines.)
//!
//! The walk is sequential: for a record of type `t`, the operand starts at
//! `align_up(cursor, t.static_align)` and the cursor advances by `t.static_size`.
//!
//! ## Conditions
//!
//! `FUN_0169e0a0` writes one boolean per trigger into the global flag array
//! `0x1ee3170`, numbering triggers sequentially across the *current state's*
//! blocks (running counter stored at `0x1ee3570`). A transition's `condition`
//! word is an index into that array (`FUN_0169d450` 0x169d450 reads
//! `0x1ee3170[condition]`). [`SmPayloads::conditions`] exposes the per-state
//! `condition -> (block, trigger)` mapping; every shipped transition's
//! condition lies below its state's trigger count.
//!
//! ## Typed vs. opaque
//!
//! Trigger types whose evaluator layout is clear are decoded into
//! [`SmTriggerOperand`] (e.g. type 17 is an 8-input boolean combine over other
//! conditions). Everything else is kept as [`SmTriggerOperand::Raw`] with its
//! type id. Filters, entity/transform sources and actions are decoded to their
//! operand ranges; actions also carry their registration name.

use crate::sm_tables::{
    action_desc, entity_desc, filter_desc, transform_desc, trigger_desc, SmActionDesc, SmFilterDesc,
};
use crate::statemachine::StateMachine;

/// A decoded trigger operand.
#[derive(Debug, Clone, PartialEq)]
pub enum SmTriggerOperand {
    /// Type 17: boolean combine over up to 8 other condition outputs.
    /// `mode` bit 0 selects OR (set) vs AND (clear); bit 1 inverts.
    BoolCombine { mode: u8, inputs: [u16; 8] },
    /// Type 5: hysteresis window `[min, max]` gated by another condition.
    TimerWindow { min: f32, max: f32, rate: f32, cond: u16 },
    /// Type 6: accumulator limited by `limit`, gated by `cond`.
    TimerGate { limit: f32, cond: u16 },
    /// Type 9: `[lo, hi]` window over the frame delta, carrying a flag byte.
    Window { lo: f32, hi: f32, flag: u8 },
    /// Types 7/8: reads bit `bit` of the engine actor flags at `+0x1364`.
    EngineFlag { bit: u8 },
    /// Anything not yet understood: the exact static operand bytes.
    Raw(Vec<u8>),
}

/// One evaluated trigger record (one output flag).
#[derive(Debug, Clone, PartialEq)]
pub struct SmTriggerRecord {
    pub state: u32,
    pub block: u32,
    /// Index within the block's trigger list.
    pub index: u16,
    /// Global condition-flag index within the state (the transition id space).
    pub output_id: u32,
    pub type_id: u8,
    /// Static operand size from the descriptor.
    pub size: u16,
    pub align: u8,
    pub known_type: bool,
    pub eval_fn: u32,
    pub operand: SmTriggerOperand,
}

/// Whether an [`SmSourceRecord`] came from the entity or transform walker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmSourceKind {
    Entity,
    Transform,
}

/// One entity- or transform-source record.
#[derive(Debug, Clone, PartialEq)]
pub struct SmSourceRecord {
    pub kind: SmSourceKind,
    pub state: u32,
    pub block: u32,
    pub index: u16,
    pub type_id: u8,
    pub size: u16,
    pub align: u8,
    pub known_type: bool,
    pub eval_fn: u32,
    pub data: Vec<u8>,
}

/// One filter record (`filters` section, 8 bytes).
#[derive(Debug, Clone, PartialEq)]
pub struct SmFilterRecord {
    pub block: u32,
    pub index: u16,
    pub filter_type: u8,
    pub data_offset: u32,
    pub data_size: u16,
    pub known_type: bool,
    pub desc: Option<SmFilterDesc>,
    pub data: Vec<u8>,
}

/// One action record (an `action_ids` entry plus its static operand).
#[derive(Debug, Clone, PartialEq)]
pub struct SmActionRecord {
    pub state: u32,
    pub block: u32,
    pub index: u16,
    /// `action_ids[.].actionId` (`i16`).
    pub action_id: i16,
    /// `action_ids[.].descriptorId` selects [`crate::sm_tables::ACTION_DESCS`].
    pub descriptor_id: u8,
    pub flags: u8,
    pub known_type: bool,
    pub desc: Option<SmActionDesc>,
    pub data: Vec<u8>,
}

/// One entry of a state's condition table.
#[derive(Debug, Clone, PartialEq)]
pub struct SmConditionOutput {
    /// The id written in a transition's `condition` word.
    pub condition: u32,
    pub block: u32,
    pub index: u16,
    pub type_id: u8,
}

/// The triggers of one state, in condition-id order.
#[derive(Debug, Clone, PartialEq)]
pub struct SmStateConditions {
    pub state: u32,
    pub trigger_count: u32,
    pub outputs: Vec<SmConditionOutput>,
}

/// Per-section payload coverage.
#[derive(Debug, Clone, PartialEq)]
pub struct SmSectionCoverage {
    pub name: &'static str,
    /// Total bytes in the section.
    pub total: u32,
    /// Bytes interpreted by the payload decoders.
    pub decoded: u32,
}

/// Coverage accounting for the payload blobs.
#[derive(Debug, Clone, PartialEq)]
pub struct SmCoverage {
    /// Whole-file size.
    pub file_bytes: u32,
    /// Bytes interpreted by `parse_sm` structure + payload decoders.
    pub decoded_bytes: u32,
    /// Bytes still opaque (mostly `block_strings` bytecode).
    pub opaque_bytes: u32,
    pub sections: Vec<SmSectionCoverage>,
}

/// Everything decoded from the payload blobs.
#[derive(Debug, Clone, PartialEq)]
pub struct SmPayloads {
    pub triggers: Vec<SmTriggerRecord>,
    pub entities: Vec<SmSourceRecord>,
    pub transforms: Vec<SmSourceRecord>,
    pub filters: Vec<SmFilterRecord>,
    pub actions: Vec<SmActionRecord>,
    pub conditions: Vec<SmStateConditions>,
    pub coverage: SmCoverage,
}

impl SmPayloads {
    /// An empty payload set for the non-strict parse path.
    pub fn empty(file_bytes: usize) -> Self {
        SmPayloads {
            triggers: Vec::new(),
            entities: Vec::new(),
            transforms: Vec::new(),
            filters: Vec::new(),
            actions: Vec::new(),
            conditions: Vec::new(),
            coverage: SmCoverage {
                file_bytes: file_bytes as u32,
                decoded_bytes: 0,
                opaque_bytes: file_bytes as u32,
                sections: Vec::new(),
            },
        }
    }
}

fn ru16(b: &[u8], o: usize, be: bool) -> Option<u16> {
    let s = b.get(o..o.checked_add(2)?)?;
    Some(if be { u16::from_be_bytes([s[0], s[1]]) } else { u16::from_le_bytes([s[0], s[1]]) })
}

fn ru32(b: &[u8], o: usize, be: bool) -> Option<u32> {
    let s = b.get(o..o.checked_add(4)?)?;
    let a = [s[0], s[1], s[2], s[3]];
    Some(if be { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) })
}

fn rf32(b: &[u8], o: usize, be: bool) -> Option<f32> {
    Some(f32::from_bits(ru32(b, o, be)?))
}

fn align_up(v: usize, a: u8) -> usize {
    let a = a.max(1) as usize;
    (v + a - 1) & !(a - 1)
}

fn section<'a>(sm: &'a StateMachine, name: &str) -> Option<&'a crate::statemachine::SmSection> {
    sm.sections.iter().find(|s| s.name == name)
}

fn u16_at(b: &[u8], o: usize, be: bool) -> u16 {
    ru16(b, o, be).unwrap_or(0)
}

/// A tiny bitmap used to union operand byte ranges for coverage.
struct Bits {
    bits: Vec<u8>,
}

impl Bits {
    fn new(n: usize) -> Self {
        Bits { bits: vec![0u8; (n + 7) / 8] }
    }
    fn mark(&mut self, start: usize, len: usize) {
        for i in start..start.saturating_add(len) {
            if let Some(byte) = self.bits.get_mut(i / 8) {
                *byte |= 1 << (i % 8);
            }
        }
    }
    fn count(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }
}

/// Decode the payload blobs of an already-parsed machine.
pub fn decode(bytes: &[u8], sm: &StateMachine) -> SmPayloads {
    let be = sm.big_endian;

    let blocks = &sm.blocks;
    let mut triggers = Vec::new();
    let mut entities = Vec::new();
    let mut transforms = Vec::new();
    let mut filters = Vec::new();
    let mut actions = Vec::new();

    // Section bases.
    let trig_types = section(sm, "transform_types").map(|s| (s.offset as usize, s.size as usize));
    let trig_data = section(sm, "transform_data").map(|s| (s.offset as usize, s.size as usize));
    let ent_types = section(sm, "trigger_types").map(|s| (s.offset as usize, s.size as usize));
    let ent_data = section(sm, "trigger_data").map(|s| (s.offset as usize, s.size as usize));
    let trn_types = section(sm, "entity_types").map(|s| (s.offset as usize, s.size as usize));
    let act_data = section(sm, "action_data").map(|s| (s.offset as usize, s.size as usize));
    let filters_sec = section(sm, "filters").map(|s| (s.offset as usize, s.size as usize));
    let filter_data = section(sm, "filter_data").map(|s| (s.offset as usize, s.size as usize));
    let act_ids = section(sm, "action_ids").map(|s| (s.offset as usize, s.size as usize));

    // Coverage bitmaps for the shared/opaque payload sections.
    let mut b_trig_data = Bits::new(trig_data.map(|x| x.1).unwrap_or(0));
    let mut b_ent_data = Bits::new(ent_data.map(|x| x.1).unwrap_or(0));
    let mut b_act_data = Bits::new(act_data.map(|x| x.1).unwrap_or(0));
    let mut b_filter_data = Bits::new(filter_data.map(|x| x.1).unwrap_or(0));
    let mut b_trig_types = Bits::new(trig_types.map(|x| x.1).unwrap_or(0));
    let mut b_ent_types = Bits::new(ent_types.map(|x| x.1).unwrap_or(0));
    let mut b_trn_types = Bits::new(trn_types.map(|x| x.1).unwrap_or(0));

    for (si, st) in sm.states.iter().enumerate() {
        let mut running_output: u32 = 0;
        for &bi in &st.blocks {
            let Some(block) = blocks.get(bi as usize) else { continue };
            let block_idx = bi;

            // ---- triggers (type array `transform_types`, blob `transform_data`)
            if let (Some(&(tb, tlen)), Some(&(db, dlen))) = (trig_types.as_ref(), trig_data.as_ref())
            {
                let base = block.transform_base as usize;
                let count = block.transform_count as usize;
                let mut cursor = db as usize + block.transform_data_offset as usize;
                for i in 0..count {
                    let tidx = tb + base + i;
                    if tidx >= tb + tlen {
                        break;
                    }
                    let tid = bytes[tidx];
                    b_trig_types.mark(base + i, 1);
                    let Some(desc) = trigger_desc(tid) else {
                        break;
                    };
                    cursor = align_up(cursor, desc.static_align);
                    if cursor + desc.static_size as usize > db + dlen {
                        break;
                    }
                    let operand = decode_trigger_operand(tid, bytes, cursor, desc.static_size as usize, be);
                    b_trig_data.mark(cursor - db, desc.static_size as usize);
                    triggers.push(SmTriggerRecord {
                        state: si as u32,
                        block: block_idx,
                        index: i as u16,
                        output_id: running_output + i as u32,
                        type_id: tid,
                        size: desc.static_size,
                        align: desc.static_align,
                        known_type: true,
                        eval_fn: desc.eval_fn,
                        operand,
                    });
                    cursor += desc.static_size as usize;
                }
                running_output += count as u32;
            } else {
                running_output += block.transform_count as u32;
            }

            // ---- entity sources (`trigger_types` / `trigger_data`)
            if let (Some(&(tb, tlen)), Some(&(db, dlen))) = (ent_types.as_ref(), ent_data.as_ref())
            {
                let base = block.trigger_base as usize;
                let count = block.trigger_count as usize;
                let mut cursor = db as usize + block.trigger_data_offset as usize;
                for i in 0..count {
                    let tidx = tb + base + i;
                    if tidx >= tb + tlen {
                        break;
                    }
                    let tid = bytes[tidx];
                    b_ent_types.mark(base + i, 1);
                    let Some(desc) = entity_desc(tid) else {
                        break;
                    };
                    cursor = align_up(cursor, desc.align);
                    if cursor + desc.size as usize > db + dlen {
                        break;
                    }
                    b_ent_data.mark(cursor - db, desc.size as usize);
                    entities.push(SmSourceRecord {
                        kind: SmSourceKind::Entity,
                        state: si as u32,
                        block: block_idx,
                        index: i as u16,
                        type_id: tid,
                        size: desc.size,
                        align: desc.align,
                        known_type: true,
                        eval_fn: desc.eval_fn,
                        data: bytes.get(cursor..cursor + desc.size as usize).unwrap_or(&[]).to_vec(),
                    });
                    cursor += desc.size as usize;
                }
            }

            // ---- transform sources (`entity_types` / `action_data`)
            if let (Some(&(tb, tlen)), Some(&(db, dlen))) = (trn_types.as_ref(), act_data.as_ref())
            {
                let base = block.entity_base as usize;
                let count = block.entity_count as usize;
                let mut cursor = db as usize + block.entity_data_offset as usize;
                for i in 0..count {
                    let tidx = tb + base + i;
                    if tidx >= tb + tlen {
                        break;
                    }
                    let tid = bytes[tidx];
                    b_trn_types.mark(base + i, 1);
                    let Some(desc) = transform_desc(tid) else {
                        break;
                    };
                    cursor = align_up(cursor, desc.align);
                    if cursor + desc.size as usize > db + dlen {
                        break;
                    }
                    b_act_data.mark(cursor - db, desc.size as usize);
                    transforms.push(SmSourceRecord {
                        kind: SmSourceKind::Transform,
                        state: si as u32,
                        block: block_idx,
                        index: i as u16,
                        type_id: tid,
                        size: desc.size,
                        align: desc.align,
                        known_type: true,
                        eval_fn: desc.eval_fn,
                        data: bytes.get(cursor..cursor + desc.size as usize).unwrap_or(&[]).to_vec(),
                    });
                    cursor += desc.size as usize;
                }
            }

            // ---- actions (`action_ids` + `action_data`)
            if let (Some(&(ab, alen)), Some(&(db, dlen))) = (act_ids.as_ref(), act_data.as_ref()) {
                let base = block.action_base as usize;
                let count = block.action_count as usize;
                let mut cursor = db as usize + block.action_data_offset as usize;
                for i in 0..count {
                    let rec = ab + (base + i) * 4;
                    if rec + 4 > ab + alen * 4 {
                        break;
                    }
                    let action_id = i16::from_le_bytes([bytes[rec], bytes[rec + 1]]);
                    let descriptor_id = bytes[rec + 2];
                    let flags = bytes[rec + 3];
                    let desc = action_desc(descriptor_id).copied();
                    let (size, known) = match desc {
                        Some(d) => {
                            cursor = align_up(cursor, d.data_align);
                            let ok = cursor + d.data_size as usize <= db + dlen;
                            if ok {
                                b_act_data.mark(cursor - db, d.data_size as usize);
                            }
                            (d.data_size, ok)
                        }
                        None => (0, false),
                    };
                    let data = if known {
                        bytes.get(cursor..cursor + size as usize).unwrap_or(&[]).to_vec()
                    } else {
                        Vec::new()
                    };
                    if known {
                        cursor += size as usize;
                    }
                    actions.push(SmActionRecord {
                        state: si as u32,
                        block: block_idx,
                        index: i as u16,
                        action_id,
                        descriptor_id,
                        flags,
                        known_type: known,
                        desc,
                        data,
                    });
                }
            }
        }
    }

    // ---- filters (`filters` section references `filter_data`)
    //
    // A filter record's `dataOffset` is relative to the owning block's
    // `filter_data_offset` (`+0x0c`), so one global record can describe
    // different operands depending on the block that references it. Decode per
    // (state, block) occurrence. Field order: `u32 dataOffset`, `u16 dataSize`,
    // `u8 filterType`, `u8 reserved`.
    if let (Some(&(fo, _fs)), Some(&(db, dlen))) = (filters_sec.as_ref(), filter_data.as_ref()) {
        for (si, st) in sm.states.iter().enumerate() {
            for &bi in &st.blocks {
                let Some(block) = blocks.get(bi as usize) else { continue };
                let base = block.filter_base as usize;
                let count = block.filter_count as usize;
                let blk_off = block.filter_data_offset as usize;
                for i in 0..count {
                    let fi = base + i;
                    let o = fo + fi * 8;
                    if o + 8 > bytes.len() {
                        break;
                    }
                    let rel = ru32(bytes, o, be).unwrap_or(0) as usize;
                    let data_size = ru16(bytes, o + 4, be).unwrap_or(0);
                    let filter_type = bytes.get(o + 6).copied().unwrap_or(0);
                    let desc = filter_desc(filter_type).copied();
                    let data_offset = blk_off + rel;
                    let start = db + data_offset;
                    let end = (start + data_size as usize).min(db + dlen);
                    let data = if start <= end {
                        b_filter_data.mark(data_offset, end.saturating_sub(start));
                        bytes.get(start..end).unwrap_or(&[]).to_vec()
                    } else {
                        Vec::new()
                    };
                    filters.push(SmFilterRecord {
                        block: bi,
                        index: fi as u16,
                        filter_type,
                        data_offset: data_offset as u32,
                        data_size,
                        known_type: desc.is_some(),
                        desc,
                        data,
                    });
                }
            }
        }
    }

    // ---- condition tables, grouped by state
    let mut conditions = Vec::new();
    {
        let mut by_state: std::collections::BTreeMap<u32, Vec<SmConditionOutput>> =
            std::collections::BTreeMap::new();
        for t in &triggers {
            by_state.entry(t.state).or_default().push(SmConditionOutput {
                condition: t.output_id,
                block: t.block,
                index: t.index,
                type_id: t.type_id,
            });
        }
        for st in &sm.states {
            let outputs = by_state.remove(&st.index).unwrap_or_default();
            let trigger_count = outputs.iter().map(|o| o.condition + 1).max().unwrap_or(0);
            conditions.push(SmStateConditions {
                state: st.index,
                trigger_count,
                outputs,
            });
        }
    }

    // ---- coverage
    let payload_names: [&'static str; 8] = [
        "block_strings",
        "trigger_data",
        "trigger_types",
        "filter_data",
        "action_data",
        "entity_types",
        "transform_data",
        "transform_types",
    ];
    let mut sec_cov = Vec::new();
    let mut payload_total: u64 = 0;
    let mut payload_decoded: u64 = 0;
    for name in payload_names {
        let Some(s) = section(sm, name) else { continue };
        let total = s.size as u64;
        let decoded = match name {
            "block_strings" => 0,
            "trigger_data" => b_ent_data.count() as u64,
            "trigger_types" => b_ent_types.count() as u64,
            "filter_data" => b_filter_data.count() as u64,
            "action_data" => b_act_data.count() as u64,
            "entity_types" => b_trn_types.count() as u64,
            "transform_data" => b_trig_data.count() as u64,
            "transform_types" => b_trig_types.count() as u64,
            _ => 0,
        };
        payload_total += total;
        payload_decoded += decoded;
        sec_cov.push(SmSectionCoverage {
            name,
            total: total as u32,
            decoded: decoded.min(total) as u32,
        });
    }
    let file_bytes = bytes.len() as u64;
    // Structural bytes = everything outside the eight payload blobs.
    let structural = file_bytes.saturating_sub(payload_total);
    let decoded_bytes = (structural + payload_decoded).min(file_bytes);
    let coverage = SmCoverage {
        file_bytes: file_bytes as u32,
        decoded_bytes: decoded_bytes as u32,
        opaque_bytes: (file_bytes - decoded_bytes) as u32,
        sections: sec_cov,
    };

    SmPayloads {
        triggers,
        entities,
        transforms,
        filters,
        actions,
        conditions,
        coverage,
    }
}

fn decode_trigger_operand(tid: u8, bytes: &[u8], off: usize, size: usize, be: bool) -> SmTriggerOperand {
    let raw = || SmTriggerOperand::Raw(bytes.get(off..off + size).unwrap_or(&[]).to_vec());
    match tid {
        17 if size >= 18 => {
            let mode = bytes[off];
            let mut inputs = [0u16; 8];
            for (k, slot) in inputs.iter_mut().enumerate() {
                *slot = u16_at(bytes, off + 2 + k * 2, be);
            }
            SmTriggerOperand::BoolCombine { mode, inputs }
        }
        5 if size >= 16 => SmTriggerOperand::TimerWindow {
            min: rf32(bytes, off, be).unwrap_or(0.0),
            max: rf32(bytes, off + 4, be).unwrap_or(0.0),
            rate: rf32(bytes, off + 8, be).unwrap_or(0.0),
            cond: u16_at(bytes, off + 12, be),
        },
        6 if size >= 8 => SmTriggerOperand::TimerGate {
            limit: rf32(bytes, off, be).unwrap_or(0.0),
            cond: u16_at(bytes, off + 4, be),
        },
        9 if size >= 12 => SmTriggerOperand::Window {
            lo: rf32(bytes, off, be).unwrap_or(0.0),
            hi: rf32(bytes, off + 4, be).unwrap_or(0.0),
            flag: bytes.get(off + 8).copied().unwrap_or(0),
        },
        7 => SmTriggerOperand::EngineFlag { bit: 0 },
        8 => SmTriggerOperand::EngineFlag { bit: 1 },
        _ => raw(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statemachine::parse_sm;

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

    fn all_actors() -> [&'static str; 7] {
        [
            "player",
            "civilian",
            "zombie",
            "freak_feral",
            "freak_juggernaut",
            "freak_screamer",
            "freak_bloater",
        ]
    }

    #[test]
    fn bloater_payloads_decode() {
        let Some(bytes) = real_sm("freak_bloater", "win") else {
            return;
        };
        let sm = parse_sm(&bytes).expect("bloater parses");
        let p = &sm.payloads;
        assert!(!p.triggers.is_empty(), "triggers decoded");
        assert!(!p.conditions.is_empty());
        assert_eq!(p.conditions.len(), sm.states.len());

        // Every trigger is a known type and its condition id is in range.
        for t in &p.triggers {
            assert!(t.known_type);
            assert!(trigger_desc(t.type_id).is_some());
        }
        // The per-state condition tables cover exactly the state's triggers.
        for st in &sm.states {
            let cond = p.conditions.iter().find(|c| c.state == st.index).unwrap();
            let expect: u32 = st.blocks.iter().map(|&bi| sm.blocks[bi as usize].transform_count as u32).sum();
            assert_eq!(cond.trigger_count, expect, "state {}", st.index);
        }
        // Every transition condition maps to a real trigger output of its state.
        for tr in &sm.transitions {
            let cond = p.conditions.iter().find(|c| c.state == tr.source).unwrap();
            assert!(
                (tr.condition as u32) < cond.trigger_count,
                "state {} condition {} >= {}",
                tr.source,
                tr.condition,
                cond.trigger_count
            );
        }
        assert!(p.coverage.decoded_bytes > 0);
        assert!(p.coverage.opaque_bytes > 0);
    }

    #[test]
    fn bool_combine_decoded() {
        // Type 17 is the dominant trigger type; check at least one instance
        // decodes into the typed 8-input boolean-combine variant.
        let mut seen = false;
        for actor in all_actors() {
            let Some(bytes) = real_sm(actor, "win") else { continue };
            let sm = parse_sm(&bytes).expect("parses");
            for t in &sm.payloads.triggers {
                if t.type_id == 17 {
                    let SmTriggerOperand::BoolCombine { mode, inputs } = &t.operand else {
                        panic!("type 17 should decode as BoolCombine");
                    };
                    assert_eq!(t.size, 18);
                    assert_eq!(inputs.len(), 8);
                    let _ = mode;
                    seen = true;
                }
            }
        }
        if real_sm("player", "win").is_none() {
            return;
        }
        assert!(seen, "expected at least one type-17 trigger");
    }

    #[test]
    fn all_machines_payloads_consistent() {
        let mut any = false;
        for actor in all_actors() {
            let Some(bytes) = real_sm(actor, "win") else { continue };
            any = true;
            let sm = parse_sm(&bytes).expect("parses");
            let p = &sm.payloads;
            assert_eq!(p.conditions.len(), sm.states.len(), "{actor}");

            // Filters stay inside the filter-data blob.
            for f in &p.filters {
                if f.data_size > 0 {
                    assert!(!f.data.is_empty(), "{actor} filter {}", f.index);
                }
            }
            // Action operands match their descriptor.
            for a in &p.actions {
                if let Some(d) = a.desc {
                    if a.known_type {
                        assert_eq!(a.data.len(), d.data_size as usize, "{actor} action {}", a.index);
                    }
                }
            }
            // Entity / transform operands match their descriptors.
            for r in p.entities.iter().chain(p.transforms.iter()) {
                assert_eq!(r.data.len(), r.size as usize, "{actor} source {}", r.index);
            }
            // Conditions never exceed the state trigger count.
            for tr in &sm.transitions {
                let cond = p.conditions.iter().find(|c| c.state == tr.source).unwrap();
                assert!((tr.condition as u32) < cond.trigger_count || cond.trigger_count == 0);
            }
        }
        assert!(any, "no machine files present");
    }

    #[test]
    fn action_names_recovered() {
        let a = crate::sm_tables::action_desc(13).expect("action 13");
        assert_eq!(a.name, "Animation");
        let a = crate::sm_tables::action_desc(44).expect("action 44");
        assert_eq!(a.name, "DebugLog");
        // All trigger/entity/transform/filter lookups are sorted and unique.
        for w in crate::sm_tables::TRIGGER_DESCS.windows(2) {
            assert!(w[0].id < w[1].id);
        }
        for w in crate::sm_tables::ACTION_DESCS.windows(2) {
            assert!(w[0].id < w[1].id);
        }
    }

    #[test]
    fn xbox360_payloads_decode() {
        let Some(bytes) = real_sm("freak_bloater", "xbox360") else {
            return;
        };
        let sm = parse_sm(&bytes).expect("xbox360 bloater parses");
        assert!(sm.big_endian);
        assert!(!sm.payloads.triggers.is_empty());
        assert_eq!(sm.payloads.conditions.len(), sm.states.len());
    }
}
