//! Player state-machine trigger types ported from their exe evaluators
//! (batch B). See `sm_runtime::TriggerCtx` and the `SmEnv` trait.
//!
//! The evaluator signature recovered from the registration table
//! (`FUN_0169e1e0` calls in `FUN_0169d470`, `0x169d470`) and the trigger walker
//! `FUN_0169e0a0` is `bool eval(ctx, const u8* static_operand, void* persist)`:
//! `ctx[0]` is the actor pointer, the second argument is the record's static
//! operand slice (size/align from `TRIGGER_DESCS`), and the third is the
//! per-instance persist slot (`persist_size` bytes, zeroed unless the
//! descriptor registers an init function).
//!
//! Anything the evaluator reads from live engine state is surfaced through an
//! [`SmEnv`] method (see the `batch B env` block in [`crate::sm_runtime`]);
//! when the environment has no source for a value its neutral default is used.
//! Evaluators whose whole result depends on an unavailable transform/entity
//! source return `None`, letting the caller fall back to
//! [`SmEnv::eval_raw_trigger`].

use crate::sm_runtime::{SmEnv, TriggerCtx};

fn ru16(b: &[u8], o: usize) -> u16 {
    match b.get(o..o + 2) {
        Some(s) => u16::from_le_bytes([s[0], s[1]]),
        None => 0,
    }
}

fn ri32(b: &[u8], o: usize) -> i32 {
    match b.get(o..o + 4) {
        Some(s) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        None => 0,
    }
}

fn rf32(b: &[u8], o: usize) -> f32 {
    match b.get(o..o + 4) {
        Some(s) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        None => 0.0,
    }
}

/// NUL-terminated operand string (types 45 / 47 carry an entity name).
fn cstr(b: &[u8]) -> &str {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    std::str::from_utf8(&b[..end]).unwrap_or("")
}

/// Evaluate trigger `type_id` if this batch implements it.
pub fn eval(type_id: u8, operand: &[u8], ctx: &mut TriggerCtx, env: &dyn SmEnv) -> Option<bool> {
    match type_id {
        // 0x169f560 (size 2): actor+0x48 + slot*4 handle equality.
        // `u8 slotA @0`, `u8 slotB @1`.
        43 => {
            let a = *operand.first()?;
            let b = *operand.get(1)?;
            Some(env.query_slot_handle(a) == env.query_slot_handle(b))
        }

        // 0x169f5d0 (size 32): actor+0x1490 referenced by name; empty operand
        // byte 0 selects the "handle is non-null" test.
        45 => Some(handle_name_test(env, 0, operand)),

        // 0x169f630 (size 32): as 45 but actor+0x1494.
        47 => Some(handle_name_test(env, 1, operand)),

        // 0x169ea80 (size 6): query match count in [u8@2, u8@3].
        // `u16 cond @0` (0xffff = ungated), `u8 min @2`, `u8 max @3`,
        // `u8 capacity @4`.
        14 => {
            let cond = ru16(operand, 0);
            let min = *operand.get(2)? as i32;
            let max = *operand.get(3)? as i32;
            let gate = cond == 0xffff
                || ctx.outputs.get(cond as usize).copied().unwrap_or(false);
            let count = env.query_match_count(cond);
            Some(gate && count >= min && count <= max)
        }

        // 0x169eb80 (size 44): nearby-target scan. `f32 min @0`, `f32 max @4`,
        // `u8 mode @8`, `u8 countA @9`, `u8 countB @10`, `u8 flag @11`.
        15 => Some(env.scan_nearby_targets(
            rf32(operand, 0),
            rf32(operand, 4),
            *operand.get(8).unwrap_or(&0),
            *operand.get(0xb).unwrap_or(&0),
        )),

        // 0x16a05f0 (size 2, persist 4 init -1): edge on actor+0x8 gated by a
        // condition output. `u16 cond @0`.
        74 => {
            let cond = ru16(operand, 0);
            let init = ctx.persist.flags & 0x80 == 0;
            if init {
                ctx.persist.i = -1;
                ctx.persist.flags |= 0x80;
            }
            let id = env.actor_change_id();
            if id == ctx.persist.i {
                return Some(false);
            }
            let gate = cond == 0xffff
                || ctx.outputs.get(cond as usize).copied().unwrap_or(false);
            if gate {
                ctx.persist.i = id;
                Some(true)
            } else {
                Some(false)
            }
        }

        // 0x16a0720 (size 0): actor+0x142c > 0.0.
        77 => Some(env.actor_float_142c() > 0.0),

        // 0x16a0740 (size 0): bit 0 of actor+0x1365.
        78 => Some(env.actor_flags2() & 1 != 0),

        // 0x16a0c50 (size 0): actor+0x1554 non-null.
        86 => Some(env.actor_flag_1554()),

        // 0x169f690 (size 16): integer metric of a referenced entity within
        // [i32@4, i32@8], XOR the invert flag bit 0 at @12. `i32 handle @0`
        // (the evaluator caches the resolved handle back into the operand).
        49 => {
            let handle = ri32(operand, 0);
            let min = ri32(operand, 4);
            let max = ri32(operand, 8);
            let invert = operand.get(12).copied().unwrap_or(0) & 1 != 0;
            match env.entity_ref_metric(handle) {
                Some(v) => Some((v >= min && v <= max) ^ invert),
                None => None,
            }
        }

        // 0x16a0f50 (size 4): resolve transform source `u16 @2` and test the
        // predicate `u16 @0`; falls through when no transform sources exist.
        97 => env.source_predicate(ru16(operand, 2), ru16(operand, 0)),

        // 0x16a1090 (size 12): `f32 angle @0`, `u16 srcA @4`, `u16 srcB @6`,
        // `u8 flags @8` (bit 0 = ignore height). srcB must be farther from the
        // actor than srcA and the angle between (srcA-actor) and (srcB-actor)
        // must satisfy `cos(angle) <= dot`.
        99 => {
            let angle = rf32(operand, 0);
            let a = ru16(operand, 4);
            let b = ru16(operand, 6);
            let flags = operand.get(8).copied().unwrap_or(0);
            let pa = env.transform_source_pos(a)?;
            let Some(pb) = env.transform_source_pos(b) else {
                // As in the evaluator: a missing second source reads as true.
                return Some(true);
            };
            let act = env.actor_position();
            let mut v1 = [pa[0] - act[0], pa[1] - act[1], pa[2] - act[2]];
            let mut v2 = [pb[0] - act[0], pb[1] - act[1], pb[2] - act[2]];
            if flags & 1 != 0 {
                v1[2] = 0.0;
                v2[2] = 0.0;
            }
            let l1 = dot(v1, v1);
            let l2 = dot(v2, v2);
            if l2 <= l1 {
                return Some(false);
            }
            if l1 <= 0.0 {
                return Some(true);
            }
            let n1 = [v1[0] / l1.sqrt(), v1[1] / l1.sqrt(), v1[2] / l1.sqrt()];
            let n2 = [v2[0] / l2.sqrt(), v2[1] / l2.sqrt(), v2[2] / l2.sqrt()];
            Some(dot(n1, n2) >= angle.cos())
        }

        // 0x169fee0 (size 4): environment z/up term > cos(angle).
        60 => Some(env.environment_cos() > rf32(operand, 0).cos()),

        // 0x169ff20 (size 2): entity-source slot `u16 @0` is live.
        61 => Some(env.entity_slot_present(ru16(operand, 0))),

        // 0x16a0c20 (size 0): linked-entity query at ctx+8.
        83 => Some(env.linked_entity_flag()),

        // 0x16a1ad0 (size 0): bit 6 of the actor flags byte at+0x1364.
        114 => Some((env.actor_flags() >> 6) & 1 != 0),

        _ => None,
    }
}

/// Types 45 / 47: `0x169f5d0` first tests `static[0]`: a non-zero first byte
/// means the operand is an entity name to resolve against the actor's handle
/// (`actor+0x1490` for 45, `actor+0x1494` for 47); a zero byte selects the
/// "handle is non-null" branch.
fn handle_name_test(env: &dyn SmEnv, which: u8, operand: &[u8]) -> bool {
    match operand.first() {
        Some(0) | None => env.actor_handle_nonzero(which),
        Some(_) => env.actor_handle_named(which, cstr(operand)),
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm_runtime::Persist;

    /// Environment with a writable value for every batch-B input.
    #[derive(Default)]
    struct E {
        handles: [i32; 5],
        named: Vec<(u8, String)>,
        nonzero: [bool; 2],
        change_id: i32,
        f142c: f32,
        flags2: u32,
        flag1554: bool,
        source_pred: Option<bool>,
        src_pos: std::collections::HashMap<u16, [f32; 3]>,
        actor_pos: [f32; 3],
        nearby: bool,
        match_count: i32,
        metric: Option<i32>,
        env_cos: f32,
        slot_present: std::collections::HashSet<u16>,
        linked: bool,
    }

    impl SmEnv for E {
        fn actor_flags(&self) -> u32 {
            self.flags2
        }
        fn query_slot_handle(&self, slot: u8) -> i32 {
            self.handles.get(slot as usize).copied().unwrap_or(0)
        }
        fn actor_handle_named(&self, which: u8, name: &str) -> bool {
            self.named.iter().any(|(w, n)| *w == which && n == name)
        }
        fn actor_handle_nonzero(&self, which: u8) -> bool {
            self.nonzero.get(which as usize).copied().unwrap_or(false)
        }
        fn actor_change_id(&self) -> i32 {
            self.change_id
        }
        fn actor_float_142c(&self) -> f32 {
            self.f142c
        }
        fn actor_flags2(&self) -> u32 {
            self.flags2
        }
        fn actor_flag_1554(&self) -> bool {
            self.flag1554
        }
        fn source_predicate(&self, _source: u16, _predicate: u16) -> Option<bool> {
            self.source_pred
        }
        fn transform_source_pos(&self, slot: u16) -> Option<[f32; 3]> {
            self.src_pos.get(&slot).copied()
        }
        fn actor_position(&self) -> [f32; 3] {
            self.actor_pos
        }
        fn scan_nearby_targets(&self, _min: f32, _max: f32, _mode: u8, _flag: u8) -> bool {
            self.nearby
        }
        fn query_match_count(&self, _cond: u16) -> i32 {
            self.match_count
        }
        fn entity_ref_metric(&self, _handle: i32) -> Option<i32> {
            self.metric
        }
        fn environment_cos(&self) -> f32 {
            self.env_cos
        }
        fn entity_slot_present(&self, slot: u16) -> bool {
            self.slot_present.contains(&slot)
        }
        fn linked_entity_flag(&self) -> bool {
            self.linked
        }
    }

    fn run(type_id: u8, operand: &[u8], env: &E) -> Option<bool> {
        let outputs = [true, false, false];
        let mut persist = Persist::default();
        let mut ctx = TriggerCtx {
            state_time: 0.0,
            outputs: &outputs,
            persist: &mut persist,
            big_endian: false,
        };
        eval(type_id, operand, &mut ctx, env)
    }

    #[test]
    fn type43_compares_query_slot_handles() {
        let mut env = E::default();
        env.handles[4] = 7;
        env.handles[2] = 7;
        env.handles[3] = 9;
        assert_eq!(run(43, &[4, 2], &env), Some(true));
        assert_eq!(run(43, &[4, 3], &env), Some(false));
    }

    #[test]
    fn type45_47_named_and_nonzero_handles() {
        let mut env = E::default();
        let mut name = b"Rifle_Mk15".to_vec();
        name.resize(32, 0);
        env.named.push((0, "Rifle_Mk15".to_string()));
        assert_eq!(run(45, &name, &env), Some(true));
        let mut other = b"Sledgehammer".to_vec();
        other.resize(32, 0);
        assert_eq!(run(45, &other, &env), Some(false));
        env.named.push((1, "Sledgehammer".to_string()));
        assert_eq!(run(47, &other, &env), Some(true));
        // Empty operand -> non-null branch.
        env.nonzero[0] = true;
        assert_eq!(run(45, &[0u8; 32], &env), Some(true));
    }

    #[test]
    fn type74_edges_on_change_id_gated_by_condition() {
        let outputs = [true, false, false];
        let mut persist = Persist::default();
        let mut ctx = TriggerCtx {
            state_time: 0.0,
            outputs: &outputs,
            persist: &mut persist,
            big_endian: false,
        };
        let mut env = E { change_id: 5, ..Default::default() };
        // First evaluation: persist starts uninitialised (-1), gate output 0 is
        // true -> fires and latches.
        assert_eq!(eval(74, &[0, 0], &mut ctx, &env), Some(true));
        // Same change id -> no edge.
        assert_eq!(eval(74, &[0, 0], &mut ctx, &env), Some(false));
        // New change id but a false gate (condition 1) -> does not fire.
        env.change_id = 6;
        assert_eq!(eval(74, &[1, 0], &mut ctx, &env), Some(false));
        // Ungated (0xffff) fires on a new change id.
        assert_eq!(eval(74, &[0xff, 0xff], &mut ctx, &env), Some(true));
    }

    #[test]
    fn type77_78_86_114_actor_flags() {
        let mut env = E::default();
        assert_eq!(run(77, &[], &env), Some(false));
        env.f142c = 0.5;
        assert_eq!(run(77, &[], &env), Some(true));
        env.flags2 = 0x41; // bit 0 (type 78) and bit 6 (type 114)
        assert_eq!(run(78, &[], &env), Some(true));
        assert_eq!(run(114, &[], &env), Some(true));
        assert_eq!(run(86, &[], &env), Some(false));
        env.flag1554 = true;
        assert_eq!(run(86, &[], &env), Some(true));
    }

    #[test]
    fn type97_falls_through_without_sources() {
        let env = E::default();
        assert_eq!(run(97, &[3, 0, 2, 0], &env), None);
        let env = E { source_pred: Some(true), ..Default::default() };
        assert_eq!(run(97, &[3, 0, 2, 0], &env), Some(true));
    }

    #[test]
    fn type99_distance_and_cone() {
        let mut env = E::default();
        env.actor_pos = [0.0, 0.0, 0.0];
        env.src_pos.insert(9, [1.0, 0.0, 0.0]);
        env.src_pos.insert(10, [2.0, 0.0, 0.0]);
        // Same direction, srcB farther, angle 0.0 -> cos 1.0 <= dot 1.0.
        let mut op = Vec::new();
        op.extend_from_slice(&0.0f32.to_le_bytes());
        op.extend_from_slice(&9u16.to_le_bytes());
        op.extend_from_slice(&10u16.to_le_bytes());
        op.extend_from_slice(&[0, 0, 0, 0]);
        assert_eq!(run(99, &op, &env), Some(true));
        // 90 degrees apart, angle threshold 0 -> false.
        env.src_pos.insert(10, [0.0, 2.0, 0.0]);
        assert_eq!(run(99, &op, &env), Some(false));
        // Missing first source -> fall through.
        assert_eq!(run(99, &op, &E::default()), None);
    }

    #[test]
    fn type60_environment_cos() {
        let env = E { env_cos: 0.5, ..Default::default() };
        // angle = 0 -> threshold cos 1.0; 0.5 > 1.0 false.
        assert_eq!(run(60, &0.0f32.to_le_bytes(), &env), Some(false));
        // angle = 2.0 rad -> cos ~ -0.416; 0.5 > -0.416 true.
        assert_eq!(run(60, &2.0f32.to_le_bytes(), &env), Some(true));
    }

    #[test]
    fn type14_query_count_range() {
        let mut env = E { match_count: 6, ..Default::default() };
        // cond 0 (output 0 true), min 6 max 255 (0xff).
        let op = [0u8, 0, 6, 0xff, 55, 1];
        assert_eq!(run(14, &op, &env), Some(true));
        env.match_count = 5;
        assert_eq!(run(14, &op, &env), Some(false));
        // Gated by a false condition output 1.
        let op2 = [1u8, 0, 0, 255, 0, 0];
        assert_eq!(run(14, &op2, &env), Some(false));
        // Ungated (0xffff) ignores the outputs table.
        let op3 = [0xffu8, 0xff, 0, 255, 0, 0];
        env.match_count = 100;
        assert_eq!(run(14, &op3, &env), Some(true));
    }

    #[test]
    fn type15_scan_nearby() {
        let env = E { nearby: true, ..Default::default() };
        let mut op = Vec::new();
        op.extend_from_slice(&0.0f32.to_le_bytes());
        op.extend_from_slice(&5.0f32.to_le_bytes());
        op.push(1);
        op.push(1);
        op.push(1);
        op.push(0);
        op.resize(44, 0);
        assert_eq!(run(15, &op, &env), Some(true));
        assert_eq!(run(15, &op, &E::default()), Some(false));
    }

    #[test]
    fn type49_entity_metric_range_with_invert() {
        let env = E { metric: Some(2), ..Default::default() };
        let mut op = Vec::new();
        op.extend_from_slice(&0x1234i32.to_le_bytes());
        op.extend_from_slice(&0i32.to_le_bytes());
        op.extend_from_slice(&3i32.to_le_bytes());
        op.extend_from_slice(&0i32.to_le_bytes());
        assert_eq!(run(49, &op, &env), Some(true));
        // Invert flag set -> false inside the range.
        op[12] = 1;
        assert_eq!(run(49, &op, &env), Some(false));
        assert_eq!(run(49, &op, &E::default()), None);
    }

    #[test]
    fn type61_83_entity_slots() {
        let mut env = E::default();
        assert_eq!(run(61, &[1, 0], &env), Some(false));
        env.slot_present.insert(1);
        assert_eq!(run(61, &[1, 0], &env), Some(true));
        assert_eq!(run(83, &[], &env), Some(false));
        env.linked = true;
        assert_eq!(run(83, &[], &env), Some(true));
    }
}
