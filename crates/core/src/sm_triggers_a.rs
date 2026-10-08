//! Player state-machine trigger types ported from their exe evaluators
//! (batch A). See `sm_runtime::TriggerCtx` and the `SmEnv` trait.
//!
//! Each evaluator was read from the disassembly at the address registered for
//! its type id (`FUN_0169e1e0` fills the descriptor table at `0x1ee2a40` from
//! the static table in [`crate::sm_tables::TRIGGER_DESCS`]). The evaluator
//! calling convention (from `FUN_0169e0a0`, 0x169e18e) is
//! `fn(ctx, operand, persist)` where `*ctx` is the actor, `operand` is the
//! static descriptor buffer and `persist` is the per-(state, output) state
//! block. Operands arrive here as the raw decoded bytes; `TriggerCtx` exposes
//! the condition outputs (global flag table `0x1ee3170`) and the persist block.
//!
//! The operand sizes come from `TRIGGER_DESCS` (`static_size`); the address in
//! each doc comment is the type's `eval_fn`.

use crate::sm_runtime::{SmEnv, TriggerCtx};

/// Evaluate trigger `type_id` if this batch implements it.
pub fn eval(type_id: u8, operand: &[u8], ctx: &mut TriggerCtx, env: &dyn SmEnv) -> Option<bool> {
    let be = ctx.big_endian;
    let v = match type_id {
        // 0x169ea00: `actor+0x14dc != -1`. The id is written by the animation
        // target subsystem (`0x13851d3`) and cleared to -1 on release.
        12 => env.focus_target_id() != -1,
        // 0x169efb0 -> FUN_01376a60: whether the 128-byte context tag mask at
        // `actor+0x12e0` contains every bit set in the operand (subset test).
        20 => tag_mask_subset(operand, env),
        // 0x169f0c0: `actor[0x14 + (b[8]>>1)*4]` inside [b[0], b[4]] (i32),
        // inverted by bit 0 of `b[8]`.
        24 => {
            if operand.len() < 9 {
                false
            } else {
                let slot = operand[8] >> 1;
                let val = env.actor_i32_slot(slot);
                let lo = ri32(operand, 0, be);
                let hi = ri32(operand, 4, be);
                let inside = lo <= val && val <= hi;
                inside ^ (operand[8] & 1 != 0)
            }
        }
        // 0x169f050: a nearby-entity count query at the actor's position
        // (`FUN_013298d0`'s vtable +0x64, query mask 0x1e00) whose result must
        // lie in `[b[4], b[5]]`; the f32 at `b[0]` is the query radius.
        25 => {
            if operand.len() < 6 {
                false
            } else {
                let radius = rf32(operand, 0, be);
                let count = env.nearby_count(radius, 0x1e00);
                count >= operand[4] as i32 && count <= operand[5] as i32
            }
        }
        // 0x169f2e0: `actor+0x1518 != -1 && (actor+0x151c & 1) != 0` (an
        // animation action/target id and its first flag bit).
        30 => env.action_target_id() != -1 && env.action_target_flags() & 1 != 0,
        // 0x169f380: `actor+0x1518 != -1`.
        36 => env.action_target_id() != -1,
        // 0x169f8e0 (enter 0x169e390): a decaying one-shot. The enter function
        // seeds `persist.t = FLT_MAX` so it starts false; the gating condition
        // `b[4]` resets it to 0, otherwise it accumulates the frame delta up to
        // `threshold + 1`. True while `b[0] >= persist.t`, i.e. for `b[0]`
        // seconds after the gate fires.
        52 => {
            if operand.len() < 6 {
                false
            } else {
                enter_type52(ctx);
                let threshold = rf32(operand, 0, be);
                let gate = cond(ctx, ru16(operand, 4, be));
                if gate {
                    ctx.persist.t = 0.0;
                } else if threshold > 0.0 {
                    let cap = threshold + 1.0;
                    ctx.persist.t = (ctx.persist.t + env.delta_seconds().max(0.0)).min(cap);
                }
                threshold >= ctx.persist.t
            }
        }
        // 0x16a0250: per-slot inventory test. With slot `b[0]`: false when the
        // slot's `actor+0x11a0+slot` bit 0 is clear; an empty operand name is
        // true; otherwise the slot's item name must match `b[1..]`, and a name
        // other than `DoorClimb` is true without the world query. A
        // `DoorClimb` slot runs a world query (`0x14150c0`, result `0x1414e10`).
        69 => equipment_slot_test(operand, env),
        // 0x16a0620: bit 7 of the actor flags byte `actor+0x1364`.
        75 => env.actor_flags() & (1 << 7) != 0,
        // 0x16a0760: velocity range. `b[8]` bit 1 selects the horizontal
        // magnitude from `actor+0x1420/0x1424`, otherwise the z component
        // `actor+0x1428`. `[b[0], b[4]]` is the accepted range; bit 0 of `b[8]`
        // also accepts the exact endpoints.
        79 => {
            if operand.len() < 9 {
                false
            } else {
                let vel = env.actor_velocity();
                let val = if operand[8] & 2 != 0 {
                    (vel[0] * vel[0] + vel[1] * vel[1]).sqrt()
                } else {
                    vel[2]
                };
                range_match(val, rf32(operand, 0, be), rf32(operand, 4, be), operand[8])
            }
        }
        // 0x16a0dc0: stateful gate/ramp accumulator over three conditions
        // (`b[8]`, `b[10]`, `b[12]`) with thresholds `b[0]` and `b[4]`; see
        // `eval_type92` for the exact transition sequence.
        92 => eval_type92(operand, ctx, env),
        // 0x16a0ec0 -> FUN_015910f0: input slot `b[0]` is active (flag bit 0 of
        // `actor+0x1234[slot]`) and its analog value `actor+0x11c8[slot]` is
        // positive.
        94 => {
            if operand.is_empty() {
                false
            } else {
                let slot = operand[0];
                env.input_slot_pressed(slot) && env.input_slot_value(slot) > 0.0
            }
        }
        // 0x16a0f00: resolve the packed id `b[0]` (`FUN_013958b0`) and test it
        // through the global game object's vtable +0x10 with `actor+0xc` and
        // `(b[4]>>1)&1`. The deep resolver is opaque; the environment answers
        // the whole test.
        96 => {
            if operand.len() < 5 {
                false
            } else {
                env.stat_test(ru32(operand, 0, be), (operand[4] >> 1) & 1 != 0)
            }
        }
        // 0x16a0f90: squared distance between the positions of the entity slots
        // `b[8]` and `b[10]` (resolved through `FUN_01698940`) must lie in
        // `[b[0], b[4]]`. `b[12]` bit 2 restricts to the z difference, else the
        // horizontal (x, y) difference; bit 0 also accepts the endpoints.
        98 => {
            if operand.len() < 13 {
                false
            } else {
                eval_type98(operand, env, be)
            }
        }
        // 0x16a1a30: `actor+0x14a4 != 0 && actor+0x14a8 != 0xffff` (a held /
        // equipped item is present and has a valid id).
        111 => env.equipped_item_present() && env.equipped_item_id() != 0xffff,
        // 0x16a1a80: height above ground `max(actor+0x1560 - actor+0x140c, 0)`
        // within the inclusive range `[b[0], b[4]]`.
        113 => {
            if operand.len() < 8 {
                false
            } else {
                let h = env.height_above_ground().max(0.0);
                rf32(operand, 0, be) <= h && h <= rf32(operand, 4, be)
            }
        }
        _ => return None,
    };
    Some(v)
}

/// Read the condition output `c` from the global flag table; `0xffff` is "no
/// condition" and is treated as clear.
fn cond(ctx: &TriggerCtx, c: u16) -> bool {
    c != 0xffff && ctx.outputs.get(c as usize).copied().unwrap_or(false)
}

/// Type 20 (0x169efb0 / `FUN_01376a60`): the actor's 128-byte context tag mask
/// must contain every bit set in the operand.
fn tag_mask_subset(operand: &[u8], env: &dyn SmEnv) -> bool {
    if operand.len() < 128 {
        return false;
    }
    let mask = env.actor_context_mask();
    operand[..128].iter().zip(mask.iter()).all(|(&o, &m)| (o & m) == o)
}

/// Type 69 (0x16a0250): per-slot inventory / attachment string test.
fn equipment_slot_test(operand: &[u8], env: &dyn SmEnv) -> bool {
    if operand.is_empty() {
        return false;
    }
    let slot = operand[0];
    if !env.equipment_slot_active(slot) {
        return false;
    }
    let name = match std::str::from_utf8(&operand[1..]) {
        Ok(s) => s.split('\0').next().unwrap_or(""),
        Err(_) => return false,
    };
    if name.is_empty() {
        return true;
    }
    if !env.equipment_slot_matches(slot, name) {
        return false;
    }
    if !env.equipment_slot_matches(slot, "DoorClimb") {
        return true;
    }
    let id = env.equipment_slot_query(slot);
    if id == -1 {
        return true;
    }
    env.world_query_result(id)
}

/// Type 79 / 98 range comparison. The interior `(lo, hi)` is always accepted;
/// bit 0 of `flags` additionally accepts an exact endpoint.
fn range_match(val: f32, lo: f32, hi: f32, flags: u8) -> bool {
    if lo < val && val < hi {
        return true;
    }
    flags & 1 != 0 && (val == lo || val == hi)
}

/// Type 98 (0x16a0f90): squared distance between two resolved actor entity
/// slots against `[b[0], b[4]]`.
fn eval_type98(operand: &[u8], env: &dyn SmEnv, be: bool) -> bool {
    let id_a = ru16(operand, 8, be) as u8;
    let id_b = ru16(operand, 10, be) as u8;
    let (Some(pa), Some(pb)) = (env.actor_slot_position(id_a), env.actor_slot_position(id_b))
    else {
        return false;
    };
    let (dx, dy, dz) = (pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]);
    let d2 = if operand[12] & 4 != 0 {
        dz * dz
    } else {
        dx * dx + dy * dy
    };
    range_match(d2, rf32(operand, 0, be), rf32(operand, 4, be), operand[12])
}

/// Whether the current state was (re-)entered since the last call, using the
/// same `state_time`-went-backwards rule as the type-5 timer. The engine calls
/// each trigger's `enter` function when it sets up the state's triggers
/// (`0x169d3c2`, descriptor `+0`); `persist.i` stores the previous
/// `state_time` bits so a reset to 0 is detected.
fn state_entered(ctx: &mut TriggerCtx) -> bool {
    let now = ctx.state_time.to_bits() as i32;
    let fresh = ctx.persist.i == 0 || now < ctx.persist.i;
    ctx.persist.i = now;
    fresh
}

/// Type 52's enter function (`0x169e390`): `persist.t = FLT_MAX`.
fn enter_type52(ctx: &mut TriggerCtx) {
    if state_entered(ctx) {
        ctx.persist.t = f32::from_bits(0x7f7f_ffff);
    }
}

/// Type 92's enter function (`0x169e420`): `persist.t = 0` and `persist.flags`
/// bit 1 seeded from operand byte `+0xe` bit 0.
fn enter_type92(operand: &[u8], ctx: &mut TriggerCtx) {
    if state_entered(ctx) {
        let seed = operand.get(14).copied().unwrap_or(0) & 1;
        ctx.persist.flags = (ctx.persist.flags & 0xfc) | (seed << 1);
        ctx.persist.t = 0.0;
    }
}

/// Type 92 (0x16a0dc0): stateful gate/ramp accumulator over three conditions.
///
/// Operand layout (16 bytes): `f32` threshold A at +0, `f32` threshold B at +4,
/// `u16` cond1 at +8, cond2 at +10, cond3 at +12. Persistent flags:
/// bit 0 = latched, bit 1 = ramping. The engine advances the accumulator by the
/// frame delta while ramping and clears it on the gate conditions; the trigger
/// fires while `B > t` (or `B == 0`) and both flags are set.
fn eval_type92(operand: &[u8], ctx: &mut TriggerCtx, env: &dyn SmEnv) -> bool {
    if operand.len() < 14 {
        return false;
    }
    enter_type92(operand, ctx);
    let be = ctx.big_endian;
    let a = rf32(operand, 0, be);
    let b = rf32(operand, 4, be);
    let c1 = cond(ctx, ru16(operand, 8, be));
    let c2 = cond(ctx, ru16(operand, 10, be));
    let c3 = cond(ctx, ru16(operand, 12, be));
    let dt = env.delta_seconds().max(0.0);

    let mut f0 = ctx.persist.flags & 1 != 0;
    let mut f1 = ctx.persist.flags & 2 != 0;
    let mut t = ctx.persist.t;

    if f1 {
        t += dt;
        if c2 {
            f0 = false;
            f1 = false;
            t = 0.0;
        }
    } else {
        f1 = c1;
    }

    if c3 {
        f1 = true;
        f0 = false;
        t = 0.0;
        if !(t < a) {
            f0 = true;
            t = 0.0;
        }
    } else if f1 {
        if !f0 {
            if !(t < a) {
                f0 = true;
                t = 0.0;
            }
        } else if t > b {
            t = b;
        }
    }

    ctx.persist.flags = (f1 as u8) << 1 | (ctx.persist.flags & 0xfc) | (f0 as u8);
    ctx.persist.t = t;

    let pass = b > t || b == 0.0;
    pass && f0 && f1
}

fn ru16(b: &[u8], o: usize, be: bool) -> u16 {
    match b.get(o..o + 2) {
        Some(s) => {
            if be {
                u16::from_be_bytes([s[0], s[1]])
            } else {
                u16::from_le_bytes([s[0], s[1]])
            }
        }
        None => 0,
    }
}

fn ru32(b: &[u8], o: usize, be: bool) -> u32 {
    match b.get(o..o + 4) {
        Some(s) => {
            let a = [s[0], s[1], s[2], s[3]];
            if be { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) }
        }
        None => 0,
    }
}

fn ri32(b: &[u8], o: usize, be: bool) -> i32 {
    ru32(b, o, be) as i32
}

fn rf32(b: &[u8], o: usize, be: bool) -> f32 {
    f32::from_bits(ru32(b, o, be))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm_runtime::Persist;

    struct Env {
        focus: i32,
        mask: [u8; 128],
        slot_i32: std::collections::HashMap<u8, i32>,
        nearby: i32,
        target_id: i32,
        target_flags: u32,
        actor_flags: u32,
        velocity: [f32; 3],
        positions: std::collections::HashMap<u8, [f32; 3]>,
        equip_active: std::collections::HashSet<u8>,
        equip_name: std::collections::HashMap<u8, String>,
        equip_query: i32,
        world_result: bool,
        height: f32,
        item_present: bool,
        item_id: u16,
        input_pressed: std::collections::HashSet<u8>,
        input_value: std::collections::HashMap<u8, f32>,
        stat: bool,
        dt: f32,
        st: f32,
    }

    impl Default for Env {
        fn default() -> Self {
            Env {
                focus: -1,
                mask: [0u8; 128],
                slot_i32: Default::default(),
                nearby: 0,
                target_id: -1,
                target_flags: 0,
                actor_flags: 0,
                velocity: [0.0; 3],
                positions: Default::default(),
                equip_active: Default::default(),
                equip_name: Default::default(),
                equip_query: -1,
                world_result: false,
                height: 0.0,
                item_present: false,
                item_id: 0xffff,
                input_pressed: Default::default(),
                input_value: Default::default(),
                stat: false,
                dt: 0.0,
                st: 0.0,
            }
        }
    }

    impl Env {
        fn run(&self, ty: u8, operand: &[u8], outputs: &[bool], p: &mut Persist) -> bool {
            let mut ctx = TriggerCtx {
                state_time: self.st,
                outputs,
                persist: p,
                big_endian: false,
            };
            eval(ty, operand, &mut ctx, self).unwrap_or(false)
        }
    }

    impl SmEnv for Env {
        fn delta_seconds(&self) -> f32 {
            self.dt
        }
        fn actor_flags(&self) -> u32 {
            self.actor_flags
        }
        fn focus_target_id(&self) -> i32 {
            self.focus
        }
        fn actor_context_mask(&self) -> [u8; 128] {
            self.mask
        }
        fn actor_i32_slot(&self, slot: u8) -> i32 {
            self.slot_i32.get(&slot).copied().unwrap_or(0)
        }
        fn nearby_count(&self, _radius: f32, _mask: u32) -> i32 {
            self.nearby
        }
        fn action_target_id(&self) -> i32 {
            self.target_id
        }
        fn action_target_flags(&self) -> u32 {
            self.target_flags
        }
        fn actor_velocity(&self) -> [f32; 3] {
            self.velocity
        }
        fn actor_slot_position(&self, slot: u8) -> Option<[f32; 3]> {
            self.positions.get(&slot).copied()
        }
        fn equipment_slot_active(&self, slot: u8) -> bool {
            self.equip_active.contains(&slot)
        }
        fn equipment_slot_matches(&self, slot: u8, name: &str) -> bool {
            self.equip_name.get(&slot).map(|n| n == name).unwrap_or(false)
        }
        fn equipment_slot_query(&self, _slot: u8) -> i32 {
            self.equip_query
        }
        fn world_query_result(&self, _id: i32) -> bool {
            self.world_result
        }
        fn height_above_ground(&self) -> f32 {
            self.height
        }
        fn equipped_item_present(&self) -> bool {
            self.item_present
        }
        fn equipped_item_id(&self) -> u16 {
            self.item_id
        }
        fn input_slot_pressed(&self, slot: u8) -> bool {
            self.input_pressed.contains(&slot)
        }
        fn input_slot_value(&self, slot: u8) -> f32 {
            self.input_value.get(&slot).copied().unwrap_or(0.0)
        }
        fn stat_test(&self, _packed: u32, _mode: bool) -> bool {
            self.stat
        }
    }

    fn p() -> Persist {
        Persist::default()
    }

    #[test]
    fn type12_focus_id() {
        let f = Env { focus: -1, ..Default::default() };
        assert!(!f.run(12, &[], &[], &mut p()));
        let t = Env { focus: 5, ..Default::default() };
        assert!(t.run(12, &[], &[], &mut p()));
    }

    #[test]
    fn type20_tag_mask_subset() {
        let mut e = Env::default();
        e.mask[0] = 0b1011;
        // operand requires bits 0 and 1 (subset) -> true.
        let mut op = vec![0u8; 128];
        op[0] = 0b0011;
        assert!(e.run(20, &op, &[], &mut p()));
        // operand requires bit 2 the actor lacks -> false.
        op[0] = 0b0111;
        assert!(!e.run(20, &op, &[], &mut p()));
        // all-zero operand is trivially a subset.
        assert!(e.run(20, &[0u8; 128], &[], &mut p()));
    }

    #[test]
    fn type24_i32_slot_range_with_invert() {
        let mut e = Env::default();
        e.slot_i32.insert(7, 1);
        // real operand: min=1 max=1 index byte 0x0e (idx 7, invert off).
        let op = [1, 0, 0, 0, 1, 0, 0, 0, 0x0e];
        assert!(e.run(24, &op, &[], &mut p()));
        // invert bit set -> false.
        let op_inv = [1, 0, 0, 0, 1, 0, 0, 0, 0x0f];
        assert!(!e.run(24, &op_inv, &[], &mut p()));
        // out of range.
        let op_bad = [5, 0, 0, 0, 6, 0, 0, 0, 0x0e];
        assert!(!e.run(24, &op_bad, &[], &mut p()));
    }

    #[test]
    fn type25_nearby_count_range() {
        let op = [0, 0, 0x20, 0x41, 1, 0xff, 0, 0]; // radius 10.0, range [1,255]
        let e = Env { nearby: 3, ..Default::default() };
        assert!(e.run(25, &op, &[], &mut p()));
        let e0 = Env { nearby: 0, ..Default::default() };
        assert!(!e0.run(25, &op, &[], &mut p()));
    }

    #[test]
    fn type30_36_action_target() {
        let e = Env { target_id: -1, ..Default::default() };
        assert!(!e.run(36, &[], &[], &mut p()));
        assert!(!e.run(30, &[], &[], &mut p()));
        let e1 = Env { target_id: 3, target_flags: 1, ..Default::default() };
        assert!(e1.run(36, &[], &[], &mut p()));
        assert!(e1.run(30, &[], &[], &mut p()));
        let e2 = Env { target_id: 3, target_flags: 2, ..Default::default() };
        assert!(e2.run(36, &[], &[], &mut p()));
        assert!(!e2.run(30, &[], &[], &mut p()));
    }

    #[test]
    fn type52_decaying_oneshot() {
        // Real operand: threshold 4.0, gate cond 0x52.
        let op = [0, 0, 0x80, 0x40, 0x52, 0, 0, 0];
        let mut e = Env { dt: 0.05, st: 1.0 / 30.0, ..Default::default() };
        let mut state = p();
        // Not yet gated, and the enter function seeds t = FLT_MAX -> false.
        let outs_off = vec![false; 0x53];
        assert!(!e.run(52, &op, &outs_off, &mut state));
        // Gated -> accumulator 0 -> true.
        let mut outs_on = vec![false; 0x53];
        outs_on[0x52] = true;
        e.st += e.dt;
        assert!(e.run(52, &op, &outs_on, &mut state));
        assert_eq!(state.t, 0.0);
        // Ungated: stays true for ~4s of accumulated delta.
        let mut frames = 0;
        loop {
            e.st += e.dt;
            if !e.run(52, &op, &outs_off, &mut state) {
                break;
            }
            frames += 1;
            if frames > 1000 {
                break;
            }
        }
        assert!(frames >= 79, "expected ~80 frames, got {frames}");
    }

    #[test]
    fn type69_equipment_slot() {
        // Real operand: slot 0x22, empty name -> active slot only.
        let mut op = vec![0x22u8];
        op.extend_from_slice(&[0u8; 64]);
        let mut e = Env::default();
        assert!(!e.run(69, &op, &[], &mut p()));
        e.equip_active.insert(0x22);
        assert!(e.run(69, &op, &[], &mut p()));
        // Named, matching item that is not DoorClimb -> true.
        let mut op = vec![0x22u8];
        op.extend_from_slice(b"Axe\0");
        e.equip_name.insert(0x22, "Axe".to_string());
        assert!(e.run(69, &op, &[], &mut p()));
        // Name mismatch -> false.
        let mut op_bad = vec![0x22u8];
        op_bad.extend_from_slice(b"Sword\0");
        assert!(!e.run(69, &op_bad, &[], &mut p()));
        // DoorClimb: world query result decides.
        let mut op = vec![0x22u8];
        op.extend_from_slice(b"DoorClimb\0");
        e.equip_name.insert(0x22, "DoorClimb".to_string());
        e.equip_query = -1;
        assert!(e.run(69, &op, &[], &mut p()));
        e.equip_query = 7;
        e.world_result = false;
        assert!(!e.run(69, &op, &[], &mut p()));
        e.world_result = true;
        assert!(e.run(69, &op, &[], &mut p()));
    }

    #[test]
    fn type75_flag_bit7() {
        assert!(!Env::default().run(75, &[], &[], &mut p()));
        let e = Env { actor_flags: 0x80, ..Default::default() };
        assert!(e.run(75, &[], &[], &mut p()));
    }

    #[test]
    fn type79_velocity_range() {
        // Real operand: [-FLT_MAX, -0.1], z component, exclusive.
        let op = [0xff, 0xff, 0x7f, 0xff, 0xcd, 0xcc, 0xcc, 0xbd, 0x00];
        let falling = Env { velocity: [0.0, 0.0, -2.0], ..Default::default() };
        assert!(falling.run(79, &op, &[], &mut p()));
        let rising = Env { velocity: [0.0, 0.0, 2.0], ..Default::default() };
        assert!(!rising.run(79, &op, &[], &mut p()));
        // bit 1 set -> horizontal magnitude (0.3, 0.4) = 0.5 in (0,1).
        let op_h = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x02];
        let horiz = Env { velocity: [0.3, 0.4, 0.0], ..Default::default() };
        assert!(horiz.run(79, &op_h, &[], &mut p()));
    }

    #[test]
    fn type92_gate_accumulator() {
        // A=0.2, B=0.0, c1=1, c2=0x2f, c3=0x31. c1 held, so the ramp latches
        // true after A seconds (B == 0 makes the result sticky).
        let op = [0xcd, 0xcc, 0x4c, 0x3e, 0, 0, 0, 0, 1, 0, 0x2f, 0, 0x31, 0, 0, 0];
        let mut e = Env { dt: 0.1, st: 0.1, ..Default::default() };
        let mut outs = vec![false; 0x32];
        outs[1] = true;
        let mut st = p();
        // First frame: ramping, not yet latched.
        assert!(!e.run(92, &op, &outs, &mut st));
        // After ~0.2 s the latch sets and stays set.
        let mut saw_true = false;
        for _ in 0..5 {
            e.st += e.dt;
            if e.run(92, &op, &outs, &mut st) {
                saw_true = true;
                break;
            }
        }
        assert!(saw_true, "type 92 should latch after A seconds");
        e.st += e.dt;
        assert!(e.run(92, &op, &outs, &mut st), "latch should hold");
    }

    #[test]
    fn type94_input_slot() {
        let op = [11u8];
        let mut e = Env::default();
        assert!(!e.run(94, &op, &[], &mut p()));
        e.input_pressed.insert(11);
        assert!(!e.run(94, &op, &[], &mut p())); // pressed but zero value
        e.input_value.insert(11, 0.5);
        assert!(e.run(94, &op, &[], &mut p()));
    }

    #[test]
    fn type96_stat_test() {
        let op = [0x80, 0x58, 0xaa, 0xfb, 0x00, 0, 0, 0];
        let e = Env { stat: true, ..Default::default() };
        assert!(e.run(96, &op, &[], &mut p()));
        assert!(!Env::default().run(96, &op, &[], &mut p()));
    }

    #[test]
    fn type98_entity_distance() {
        // min=0, max=1, slots 7 and 8, flags 0 -> horizontal squared distance.
        let op = [0, 0, 0, 0, 0, 0, 0x80, 0x3f, 7, 0, 8, 0, 0, 0, 0, 0];
        let mut e = Env::default();
        e.positions.insert(7, [0.0, 0.0, 0.0]);
        e.positions.insert(8, [0.5, 0.0, 0.0]);
        assert!(e.run(98, &op, &[], &mut p()));
        // 2.0 apart -> squared 4.0 > 1.0 -> false.
        e.positions.insert(8, [2.0, 0.0, 0.0]);
        assert!(!e.run(98, &op, &[], &mut p()));
        // Missing slot -> false.
        let e2 = Env::default();
        assert!(!e2.run(98, &op, &[], &mut p()));
    }

    #[test]
    fn type111_equipped_item() {
        assert!(!Env::default().run(111, &[], &[], &mut p()));
        let e = Env { item_present: true, item_id: 3, ..Default::default() };
        assert!(e.run(111, &[], &[], &mut p()));
        let bad = Env { item_present: true, item_id: 0xffff, ..Default::default() };
        assert!(!bad.run(111, &[], &[], &mut p()));
    }

    #[test]
    fn type113_height_range() {
        // Real operand: [0.3, 100.0].
        let op = [0x9a, 0x99, 0x99, 0x3e, 0x00, 0x00, 0xc8, 0x42];
        let e = Env { height: 1.5, ..Default::default() };
        assert!(e.run(113, &op, &[], &mut p()));
        let low = Env { height: 0.1, ..Default::default() };
        assert!(!low.run(113, &op, &[], &mut p()));
        // Negative height is clamped to 0 -> below min -> false.
        let neg = Env { height: -5.0, ..Default::default() };
        assert!(!neg.run(113, &op, &[], &mut p()));
    }
}
