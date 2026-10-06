//! `CScriptBind_Entity`: the native methods every entity table inherits
//! (`self:GetPos()`, `self:LoadObject(...)`, `self:SetTimer(...)`, ...).
//!
//! Methods operate on [`super::world::EntityWorld`]; anything visual is
//! recorded there and picked up by the Bevy presentation layer.

use super::host::{num, self_id, truthy, value_to_id, vec3_from, vec3_out, Ctx};
use super::world::{
    angles_from_quat, quat_from_angles, AnimState, FlowOutput, FlowValue, PhysKind, PhysParams, Slot, SlotKind,
};
use mlua::{Lua, Result, Table, Value, Variadic};

/// Resolve a slot argument (`-1`/nil = first free slot).
fn slot_index(v: &Value, free: usize) -> usize {
    match num(v) {
        Some(n) if n >= 0.0 => n as usize,
        _ => free,
    }
}

fn value_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.to_string_lossy().to_string()),
        _ => None,
    }
}

/// `ActivateOutput`/`ProcessBroadcastEvent` value -> typed flow value.
fn flow_value(v: &Value) -> FlowValue {
    match v {
        Value::Nil => FlowValue::Nil,
        Value::Boolean(b) => FlowValue::Bool(*b),
        Value::Integer(i) => FlowValue::Num(*i as f64),
        Value::Number(n) => FlowValue::Num(*n),
        Value::String(s) => FlowValue::Str(s.to_string_lossy().to_string()),
        Value::LightUserData(p) => FlowValue::Id(p.0 as usize as super::world::EntityId),
        Value::Table(_) => value_to_id(v).map(FlowValue::Id).unwrap_or(FlowValue::Nil),
        _ => FlowValue::Nil,
    }
}

/// Best-effort clip length: if `name` is a real `.caf` path (or a file stem
/// that matches exactly one `.caf` in the mounted paks) parse its duration via
/// [`crate::anim::Caf`]; otherwise 0.0.
///
/// TODO: symbolic animation names (`"Default"`, `"weaponRecoil"`) are resolved
/// through the character's animation set / CDF in the original engine; that
/// mapping is not wired up yet, so such clips report 0.0.
fn resolve_caf(vfs: &crate::crypak::Vfs, name: &str) -> Option<f32> {
    if name.is_empty() {
        return None;
    }
    let lower = name.replace('\\', "/").to_ascii_lowercase();
    if lower.ends_with(".caf") {
        if let Some(bytes) = vfs.read(&lower) {
            return crate::anim::Caf::parse(&bytes).map(|c| c.duration_secs());
        }
    }
    let stem = lower.rsplit('/').next().unwrap_or(&lower).trim_end_matches(".caf").to_string();
    let mut found = None;
    for p in vfs.paths_with_ext("caf") {
        let pl = p.replace('\\', "/").to_ascii_lowercase();
        let f = pl.rsplit('/').next().unwrap_or(&pl);
        if f.strip_suffix(".caf") == Some(stem.as_str()) {
            found = Some(p);
            break;
        }
    }
    let path = found?;
    let bytes = vfs.read(&path)?;
    crate::anim::Caf::parse(&bytes).map(|c| c.duration_secs())
}

pub fn install(lua: &Lua, ctx: &Ctx) -> Result<()> {
    let n: Table = lua.globals().get("EntityNative")?;

    // -- transform ---------------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "GetPos",
        lua.create_function(move |lua, (this, out): (Table, Option<Table>)| {
            let p = self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.pos)).unwrap_or_default();
            vec3_out(lua, out, p)
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetWorldPos",
        lua.create_function(move |lua, (this, out): (Table, Option<Table>)| {
            let p = self_id(&this).map(|id| c.world.borrow().world_pos(id)).unwrap_or_default();
            vec3_out(lua, out, p)
        })?,
    )?;
    for name in ["SetPos", "SetWorldPos"] {
        let c = ctx.clone();
        n.raw_set(
            name,
            lua.create_function(move |_, (this, p): (Table, Value)| {
                if let (Some(id), Some(p)) = (self_id(&this), vec3_from(&p)) {
                    c.world.borrow_mut().set_pos(id, p);
                }
                Ok(())
            })?,
        )?;
    }
    for name in ["GetAngles", "GetWorldAngles"] {
        let c = ctx.clone();
        n.raw_set(
            name,
            lua.create_function(move |lua, (this, out): (Table, Option<Table>)| {
                let q = self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.rot)).unwrap_or([0.0, 0.0, 0.0, 1.0]);
                vec3_out(lua, out, angles_from_quat(q))
            })?,
        )?;
    }
    for name in ["SetAngles", "SetWorldAngles"] {
        let c = ctx.clone();
        n.raw_set(
            name,
            lua.create_function(move |_, (this, a): (Table, Value)| {
                if let (Some(id), Some(a)) = (self_id(&this), vec3_from(&a)) {
                    c.world.borrow_mut().set_rot(id, quat_from_angles(a));
                }
                Ok(())
            })?,
        )?;
    }
    let c = ctx.clone();
    n.raw_set(
        "GetDirectionVector",
        lua.create_function(move |lua, (this, axis, out): (Table, Option<Value>, Option<Table>)| {
            let axis = axis.as_ref().and_then(num).unwrap_or(1.0) as usize;
            let d = self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.axis(axis))).unwrap_or([0.0, 1.0, 0.0]);
            vec3_out(lua, out, d)
        })?,
    )?;

    // -- identity ------------------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "GetName",
        lua.create_function(move |_, this: Table| {
            Ok(self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.name.clone())).unwrap_or_default())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetName",
        lua.create_function(move |_, (this, name): (Table, String)| {
            if let Some(id) = self_id(&this) {
                c.world.borrow_mut().rename(id, &name);
            }
            Ok(())
        })?,
    )?;

    // -- activation / visibility -------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "Activate",
        lua.create_function(move |_, (this, on): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.active = truthy(&on);
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "IsActive",
        lua.create_function(move |_, this: Table| {
            Ok(self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.active)).unwrap_or(false))
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "Hide",
        lua.create_function(move |_, (this, on): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                let mut w = c.world.borrow_mut();
                if let Some(e) = w.get_mut(id) {
                    e.hidden = truthy(&on);
                }
                w.events.push(super::world::WorldEvent::VisibilityChanged(id));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "IsHidden",
        lua.create_function(move |_, this: Table| {
            Ok(self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.hidden)).unwrap_or(false))
        })?,
    )?;

    // -- slots ---------------------------------------------------------------
    for (name, kind) in [
        ("LoadObject", SlotKind::Static),
        ("LoadCharacter", SlotKind::Character),
        ("LoadParticleEffect", SlotKind::Particle),
    ] {
        let c = ctx.clone();
        n.raw_set(
            name,
            lua.create_function(move |_, (this, slot, path): (Table, Value, Value)| {
                let Some(id) = self_id(&this) else { return Ok(-1i64) };
                let path = match &path {
                    Value::String(s) => s.to_string_lossy().to_string(),
                    _ => return Ok(-1),
                };
                if path.is_empty() {
                    return Ok(-1);
                }
                let mut w = c.world.borrow_mut();
                let Some(e) = w.get_mut(id) else { return Ok(-1) };
                let i = slot_index(&slot, e.free_slot_index());
                e.set_slot(i, Some(Slot::new(kind, path.replace('\\', "/"))));
                w.events.push(super::world::WorldEvent::SlotsChanged(id));
                Ok(i as i64)
            })?,
        )?;
    }
    let c = ctx.clone();
    n.raw_set(
        "FreeSlot",
        lua.create_function(move |_, (this, slot): (Table, Value)| {
            if let (Some(id), Some(i)) = (self_id(&this), num(&slot)) {
                let mut w = c.world.borrow_mut();
                if let Some(e) = w.get_mut(id) {
                    if i >= 0.0 {
                        e.set_slot(i as usize, None);
                    }
                }
                w.events.push(super::world::WorldEvent::SlotsChanged(id));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "FreeAllSlots",
        lua.create_function(move |_, this: Table| {
            if let Some(id) = self_id(&this) {
                let mut w = c.world.borrow_mut();
                if let Some(e) = w.get_mut(id) {
                    e.slots.clear();
                }
                w.events.push(super::world::WorldEvent::SlotsChanged(id));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "DrawSlot",
        lua.create_function(move |_, (this, slot, on): (Table, Value, Value)| {
            if let (Some(id), Some(i)) = (self_id(&this), num(&slot)) {
                let mut w = c.world.borrow_mut();
                if let Some(s) = w.get_mut(id).and_then(|e| e.slots.get_mut(i as usize)).and_then(|s| s.as_mut()) {
                    s.visible = truthy(&on);
                }
                w.events.push(super::world::WorldEvent::SlotsChanged(id));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "IsSlotValid",
        lua.create_function(move |_, (this, slot): (Table, Value)| {
            Ok(match (self_id(&this), num(&slot)) {
                (Some(id), Some(i)) if i >= 0.0 => {
                    c.world.borrow().get(id).and_then(|e| e.slot(i as usize)).is_some()
                }
                _ => false,
            })
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetSlotCount",
        lua.create_function(move |_, this: Table| {
            Ok(self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.slots.len())).unwrap_or(0))
        })?,
    )?;

    // -- timers / lifetime ---------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "SetTimer",
        lua.create_function(move |_, (this, timer_id, ms): (Table, Value, Value)| {
            if let Some(id) = self_id(&this) {
                let now = c.now_ms();
                c.timers.borrow_mut().set_entity(
                    now,
                    id,
                    num(&timer_id).unwrap_or(0.0) as i64,
                    num(&ms).unwrap_or(0.0) as i64,
                );
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "KillTimer",
        lua.create_function(move |_, (this, timer_id): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                c.timers.borrow_mut().kill_entity(id, num(&timer_id).unwrap_or(0.0) as i64);
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "DeleteThis",
        lua.create_function(move |_, this: Table| {
            if let Some(id) = self_id(&this) {
                c.world.borrow_mut().remove(id);
            }
            Ok(())
        })?,
    )?;

    // -- physics bookkeeping -------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "Physicalize",
        lua.create_function(move |_, (this, _slot, ptype, params): (Table, Value, Value, Value)| {
            if let Some(id) = self_id(&this) {
                let mut w = c.world.borrow_mut();
                if let Some(e) = w.get_mut(id) {
                    e.physics = PhysKind::from_pe(num(&ptype).unwrap_or(0.0) as i64);
                    if let Value::Table(p) = &params {
                        if let Some(m) = p.raw_get::<Value>("mass").ok().as_ref().and_then(num) {
                            e.mass = m as f32;
                        }
                    }
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "DestroyPhysics",
        lua.create_function(move |_, this: Table| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.physics = PhysKind::None;
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetViewDistRatio",
        lua.create_function(move |_, (this, r): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.view_dist_ratio = num(&r).unwrap_or(100.0) as i32;
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetFlags",
        lua.create_function(move |_, (this, flags, mode): (Table, Value, Option<Value>)| {
            // mode: 0 = or, 1 = and, 2 = xor (CryEngine SetFlags semantics), default 0.
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    let f = num(&flags).unwrap_or(0.0) as u32;
                    match mode.as_ref().and_then(num).unwrap_or(0.0) as i32 {
                        1 => e.flags &= f,
                        2 => e.flags ^= f,
                        _ => e.flags |= f,
                    }
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetFlags",
        lua.create_function(move |_, this: Table| {
            Ok(self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.flags)).unwrap_or(0))
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetTriggerBBox",
        lua.create_function(move |_, (this, min, max): (Table, Value, Value)| {
            if let (Some(id), Some(a), Some(b)) = (self_id(&this), vec3_from(&min), vec3_from(&max)) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.trigger_bbox = Some((a, b));
                }
            }
            Ok(())
        })?,
    )?;

    // -- streaming / network / physics bookkeeping --------------------------
    let c = ctx.clone();
    n.raw_set(
        "NetPresent",
        lua.create_function(move |_, (this, _): (Table, Variadic<Value>)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.net_present = true;
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetUpdatePolicy",
        lua.create_function(move |_, (this, policy): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.update_policy = num(&policy).unwrap_or(0.0) as i32;
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetRegisterInSectors",
        lua.create_function(move |_, (this, on): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.register_in_sectors = truthy(&on);
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "AwakePhysics",
        lua.create_function(move |_, (this, on): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.physics_awake = truthy(&on);
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "EnablePhysics",
        lua.create_function(move |_, (this, on): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.physics_enabled = truthy(&on);
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetPhysicParams",
        lua.create_function(move |_, (this, ptype, params): (Table, Value, Value)| {
            let (Some(id), Some(ptype)) = (self_id(&this), num(&ptype)) else { return Ok(()) };
            let mut pp = PhysParams::default();
            if let Value::Table(tbl) = &params {
                if let Ok(v) = tbl.raw_get::<Value>("flags") {
                    pp.flags = num(&v).unwrap_or(0.0) as u32;
                }
                if let Ok(v) = tbl.raw_get::<Value>("flags_mask") {
                    pp.flags_mask = num(&v).unwrap_or(0.0) as u32;
                }
                if let Ok(v) = tbl.raw_get::<Value>("mass") {
                    pp.mass = num(&v).unwrap_or(0.0) as f32;
                }
                if let Ok(v) = tbl.raw_get::<Value>("density") {
                    pp.density = num(&v).unwrap_or(0.0) as f32;
                }
                if let Ok(v) = tbl.raw_get::<Value>("velocity") {
                    if let Some(v) = vec3_from(&v) {
                        pp.velocity = v;
                    }
                }
                for item in tbl.pairs::<Value, Value>() {
                    let (k, v) = item?;
                    if let Value::String(s) = k {
                        if let Some(n) = num(&v) {
                            pp.values.insert(s.to_string_lossy().to_string(), n);
                        }
                    }
                }
            }
            if let Some(e) = c.world.borrow_mut().get_mut(id) {
                e.phys_params.insert(ptype as i64, pp);
            }
            Ok(())
        })?,
    )?;

    // -- sound / decals ------------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "SetSoundEffectRadius",
        lua.create_function(move |_, (this, r): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.sound_effect_radius = num(&r).unwrap_or(0.0) as f32;
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "PlaySoundEvent",
        lua.create_function(move |_, (this, name): (Table, Value)| {
            if let (Some(id), Some(name)) = (self_id(&this), value_str(&name)) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.sound_events.push(name);
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "EnableDecals",
        lua.create_function(move |_, (this, _slot, on): (Table, Value, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.decals_enabled = truthy(&on);
                }
            }
            Ok(())
        })?,
    )?;
    n.raw_set(
        "RemoveDecals",
        lua.create_function(move |_, this: Table| {
            // No decal list is kept in the engine layer yet; the call is
            // accepted so scripts do not fall through to the logging stub.
            let _ = self_id(&this);
            Ok(())
        })?,
    )?;

    // -- AI registration -----------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "SetAIName",
        lua.create_function(move |_, (this, name): (Table, Value)| {
            if let (Some(id), Some(name)) = (self_id(&this), value_str(&name)) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.ai_name = Some(name);
                }
            }
            Ok(())
        })?,
    )?;

    // -- triggers ------------------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "InvalidateTrigger",
        lua.create_function(move |_, this: Table| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.trigger_invalidated = true;
                }
            }
            Ok(())
        })?,
    )?;

    // -- flowgraph -----------------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "ActivateOutput",
        lua.create_function(move |_, (this, port, value): (Table, Value, Value)| {
            if let (Some(id), Some(port)) = (self_id(&this), value_str(&port)) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.flow_outputs.push(FlowOutput { port, value: flow_value(&value) });
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "ProcessBroadcastEvent",
        lua.create_function(move |_, (this, event): (Table, Value)| {
            if let (Some(id), Some(port)) = (self_id(&this), value_str(&event)) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.flow_outputs.push(FlowOutput { port, value: FlowValue::Nil });
                }
            }
            Ok(())
        })?,
    )?;

    // -- particle precache ---------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "PreLoadParticleEffect",
        lua.create_function(move |_, (this, name): (Table, Value)| {
            if let (Some(id), Some(name)) = (self_id(&this), value_str(&name)) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.preloaded_effects.push(name);
                }
            }
            Ok(())
        })?,
    )?;

    // -- character animation -------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "StartAnimation",
        lua.create_function(move |_, (this, args): (Table, Variadic<Value>)| {
            let Some(id) = self_id(&this) else { return Ok(()) };
            let slot = args.first().and_then(num).unwrap_or(0.0) as usize;
            let name = args.get(1).and_then(value_str).unwrap_or_default();
            let layer = args.get(2).and_then(num).unwrap_or(0.0) as i32;
            let speed = args.get(4).and_then(num).unwrap_or(1.0) as f32;
            let looping = args.get(5).map(truthy).unwrap_or(false);
            let length = resolve_caf(&c.vfs, &name).unwrap_or(0.0);
            let now = c.now_ms();
            if let Some(e) = c.world.borrow_mut().get_mut(id) {
                e.animations.insert(
                    slot,
                    AnimState { name, layer, speed, start_ms: now, looping, length, running: true },
                );
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "StopAnimation",
        lua.create_function(move |_, (this, slot, _layer): (Table, Value, Value)| {
            if let Some(id) = self_id(&this) {
                let slot = num(&slot).unwrap_or(0.0) as i64;
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    if slot < 0 {
                        for a in e.animations.values_mut() {
                            a.running = false;
                        }
                    } else if let Some(a) = e.animations.get_mut(&(slot as usize)) {
                        a.running = false;
                    }
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "ResetAnimation",
        lua.create_function(move |_, (this, slot, _layer): (Table, Value, Value)| {
            if let Some(id) = self_id(&this) {
                let slot = num(&slot).unwrap_or(0.0) as i64;
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    if slot < 0 {
                        e.animations.clear();
                    } else {
                        e.animations.remove(&(slot as usize));
                    }
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetAnimationSpeed",
        lua.create_function(move |_, (this, slot, _layer, speed): (Table, Value, Value, Value)| {
            if let Some(id) = self_id(&this) {
                let slot = num(&slot).unwrap_or(0.0) as usize;
                let speed = num(&speed).unwrap_or(1.0) as f32;
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    let a = e.animations.entry(slot).or_insert_with(AnimState::default);
                    a.speed = speed;
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetAnimationLength",
        lua.create_function(move |_, (this, _slot, name): (Table, Value, Value)| {
            let _ = self_id(&this);
            Ok(value_str(&name).and_then(|n| resolve_caf(&c.vfs, &n)).unwrap_or(0.0) as f64)
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "IsAnimationRunning",
        lua.create_function(move |_, (this, slot, _layer): (Table, Value, Value)| {
            Ok(match (self_id(&this), num(&slot)) {
                (Some(id), Some(s)) => c
                    .world
                    .borrow()
                    .get(id)
                    .and_then(|e| e.animations.get(&(s as usize)))
                    .map(|a| a.running)
                    .unwrap_or(false),
                _ => false,
            })
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetAnimationTime",
        lua.create_function(move |_, (this, slot, _layer): (Table, Value, Value)| {
            let now = c.now_ms();
            Ok(match (self_id(&this), num(&slot)) {
                (Some(id), Some(s)) => c
                    .world
                    .borrow()
                    .get(id)
                    .and_then(|e| e.animations.get(&(s as usize)))
                    .map(|a| {
                        let dt = (now - a.start_ms).max(0) as f32 / 1000.0;
                        (dt * a.speed) as f64
                    })
                    .unwrap_or(0.0),
                _ => 0.0,
            })
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetAnimationTime",
        lua.create_function(move |_, (this, slot, _layer, t): (Table, Value, Value, Value)| {
            if let Some(id) = self_id(&this) {
                let slot = num(&slot).unwrap_or(0.0) as usize;
                let t = num(&t).unwrap_or(0.0) as f32;
                let now = c.now_ms();
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    if let Some(a) = e.animations.get_mut(&slot) {
                        a.start_ms = now - (t / a.speed.max(f32::EPSILON) * 1000.0) as i64;
                    }
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "ForceCharacterUpdate",
        lua.create_function(move |_, (this, _slot, always): (Table, Value, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.character_update = truthy(&always);
                }
            }
            Ok(())
        })?,
    )?;
    // -- slot transforms -----------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "SetSlotPos",
        lua.create_function(move |_, (this, slot, p): (Table, Value, Value)| {
            if let (Some(id), Some(i), Some(p)) = (self_id(&this), num(&slot), vec3_from(&p)) {
                let mut w = c.world.borrow_mut();
                if let Some(s) = w.get_mut(id).and_then(|e| e.slots.get_mut(i as usize)).and_then(|s| s.as_mut()) {
                    s.pos = p;
                }
                w.events.push(super::world::WorldEvent::SlotsChanged(id));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetSlotAngles",
        lua.create_function(move |_, (this, slot, a): (Table, Value, Value)| {
            if let (Some(id), Some(i), Some(a)) = (self_id(&this), num(&slot), vec3_from(&a)) {
                let mut w = c.world.borrow_mut();
                if let Some(s) = w.get_mut(id).and_then(|e| e.slots.get_mut(i as usize)).and_then(|s| s.as_mut()) {
                    s.rot = quat_from_angles(a);
                }
                w.events.push(super::world::WorldEvent::SlotsChanged(id));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetSlotPos",
        lua.create_function(move |lua, (this, slot, out): (Table, Value, Option<Table>)| {
            let p = self_id(&this)
                .and_then(|id| {
                    let w = c.world.borrow();
                    let e = w.get(id)?;
                    Some(e.slot(num(&slot).unwrap_or(0.0) as usize)?.pos)
                })
                .unwrap_or_default();
            vec3_out(lua, out, p)
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetSlotAngles",
        lua.create_function(move |lua, (this, slot, out): (Table, Value, Option<Table>)| {
            let a = self_id(&this)
                .and_then(|id| {
                    let w = c.world.borrow();
                    let e = w.get(id)?;
                    Some(angles_from_quat(e.slot(num(&slot).unwrap_or(0.0) as usize)?.rot))
                })
                .unwrap_or_default();
            vec3_out(lua, out, a)
        })?,
    )?;
    for name in ["GetSlotScale", "GetWorldScale"] {
        let c = ctx.clone();
        n.raw_set(
            name,
            lua.create_function(move |lua, (this, slot, out): (Table, Option<Value>, Option<Table>)| {
                let s = self_id(&this)
                    .and_then(|id| {
                        let w = c.world.borrow();
                        let e = w.get(id)?;
                        match slot.as_ref().and_then(num) {
                            Some(i) => Some(e.slot(i as usize)?.scale),
                            None => Some(e.scale),
                        }
                    })
                    .unwrap_or([1.0; 3]);
                vec3_out(lua, out, s)
            })?,
        )?;
    }
    for name in ["SetSlotScale", "SetScale", "SetWorldScale"] {
        let c = ctx.clone();
        n.raw_set(
            name,
            lua.create_function(move |_, (this, a, b): (Table, Value, Value)| {
                let Some(id) = self_id(&this) else { return Ok(()) };
                let scale = vec3_from(&b)
                    .or_else(|| vec3_from(&a))
                    .or_else(|| num(&b).map(|n| [n as f32; 3]))
                    .or_else(|| num(&a).map(|n| [n as f32; 3]))
                    .unwrap_or([1.0; 3]);
                let slot = num(&a);
                let mut w = c.world.borrow_mut();
                if let Some(e) = w.get_mut(id) {
                    match (name, slot) {
                        ("SetSlotScale", Some(i)) => {
                            if let Some(s) = e.slots.get_mut(i as usize).and_then(|s| s.as_mut()) {
                                s.scale = scale;
                            }
                        }
                        _ => e.scale = scale,
                    }
                }
                w.events.push(super::world::WorldEvent::SlotsChanged(id));
                Ok(())
            })?,
        )?;
    }
    let c = ctx.clone();
    n.raw_set(
        "GetScale",
        lua.create_function(move |lua, (this, out): (Table, Option<Table>)| {
            let s = self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.scale)).unwrap_or([1.0; 3]);
            vec3_out(lua, out, s)
        })?,
    )?;

    // -- local transform (same as world when unparented) ---------------------
    let c = ctx.clone();
    n.raw_set(
        "GetLocalPos",
        lua.create_function(move |lua, (this, out): (Table, Option<Table>)| {
            let p = self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.pos)).unwrap_or_default();
            vec3_out(lua, out, p)
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetLocalPos",
        lua.create_function(move |_, (this, p): (Table, Value)| {
            if let (Some(id), Some(p)) = (self_id(&this), vec3_from(&p)) {
                c.world.borrow_mut().set_pos(id, p);
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetLocalAngles",
        lua.create_function(move |lua, (this, out): (Table, Option<Table>)| {
            let q = self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.rot)).unwrap_or([0.0, 0.0, 0.0, 1.0]);
            vec3_out(lua, out, angles_from_quat(q))
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetLocalAngles",
        lua.create_function(move |_, (this, a): (Table, Value)| {
            if let (Some(id), Some(a)) = (self_id(&this), vec3_from(&a)) {
                c.world.borrow_mut().set_rot(id, quat_from_angles(a));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetDirectionVector",
        lua.create_function(move |_, (this, d): (Table, Value)| {
            if let (Some(id), Some(d)) = (self_id(&this), vec3_from(&d)) {
                c.world.borrow_mut().set_rot(id, super::world::quat_from_dir(d));
            }
            Ok(())
        })?,
    )?;

    // -- physics / motion ----------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "AddImpulse",
        lua.create_function(move |_, (this, args): (Table, Variadic<Value>)| {
            let Some(id) = self_id(&this) else { return Ok(()) };
            let dir = args.get(2).and_then(vec3_from);
            let scale = args.get(3).and_then(num).unwrap_or(0.0) as f32;
            if let Some(dir) = dir {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    let mass = if e.mass > 0.0 { e.mass } else { 1.0 };
                    for i in 0..3 {
                        e.velocity[i] += dir[i] * scale / mass;
                    }
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetVelocity",
        lua.create_function(move |lua, (this, out): (Table, Option<Table>)| {
            let v = self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.velocity)).unwrap_or_default();
            vec3_out(lua, out, v)
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "RagDollize",
        lua.create_function(move |_, (this, _slot): (Table, Option<Value>)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.physics = PhysKind::Rigid;
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "IsARigidBody",
        lua.create_function(move |_, this: Table| {
            Ok(self_id(&this)
                .and_then(|id| c.world.borrow().get(id).map(|e| matches!(e.physics, PhysKind::Rigid | PhysKind::Articulated)))
                .unwrap_or(false))
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetPhysicsProperties",
        lua.create_function(move |_, (this, _slot, active): (Table, Value, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.physics_awake = truthy(&active);
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "CharacterUpdateOnRender",
        lua.create_function(move |_, (this, _slot, on): (Table, Value, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.character_update = truthy(&on);
                }
            }
            Ok(())
        })?,
    )?;

    // -- render / slots ------------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "LoadSubObject",
        lua.create_function(move |_, (this, slot, file, _geom): (Table, Value, Value, Value)| {
            let Some(id) = self_id(&this) else { return Ok(-1i64) };
            let Some(path) = value_str(&file) else { return Ok(-1) };
            if path.is_empty() {
                return Ok(-1);
            }
            let mut w = c.world.borrow_mut();
            let Some(e) = w.get_mut(id) else { return Ok(-1) };
            let i = slot_index(&slot, e.free_slot_index());
            e.set_slot(i, Some(Slot::new(SlotKind::Static, path.replace('\\', "/"))));
            w.events.push(super::world::WorldEvent::SlotsChanged(id));
            Ok(i as i64)
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "DrawObject",
        lua.create_function(move |_, (this, slot): (Table, Option<Value>)| {
            if let Some(id) = self_id(&this) {
                let mut w = c.world.borrow_mut();
                if let Some(e) = w.get_mut(id) {
                    let i = slot_index(slot.as_ref().unwrap_or(&Value::Nil), e.free_slot_index());
                    if let Some(s) = e.slots.get_mut(i).and_then(|s| s.as_mut()) {
                        s.visible = true;
                    }
                }
                w.events.push(super::world::WorldEvent::SlotsChanged(id));
            }
            Ok(())
        })?,
    )?;
    n.raw_set(
        "UpdateLightClipBounds",
        lua.create_function(move |_, (this, _slot): (Table, Option<Value>)| {
            let _ = self_id(&this);
            Ok(())
        })?,
    )?;

    // -- material parameters -------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "SetMaterialFloat",
        lua.create_function(move |_, (this, slot, sub, name, value): (Table, Value, Value, Value, Value)| {
            if let (Some(id), Some(name)) = (self_id(&this), value_str(&name)) {
                let key = format!("{}:{}:{}", num(&slot).unwrap_or(0.0) as i64, num(&sub).unwrap_or(0.0) as i64, name);
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.material_floats.insert(key, num(&value).unwrap_or(0.0) as f32);
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetMaterialFloat",
        lua.create_function(move |_, (this, slot, sub, name): (Table, Value, Value, Value)| {
            let Some(id) = self_id(&this) else { return Ok(0.0) };
            let Some(name) = value_str(&name) else { return Ok(0.0) };
            let key = format!("{}:{}:{}", num(&slot).unwrap_or(0.0) as i64, num(&sub).unwrap_or(0.0) as i64, name);
            Ok(c.world.borrow().get(id).and_then(|e| e.material_floats.get(&key).copied()).unwrap_or(0.0) as f64)
        })?,
    )?;

    // -- skeleton helpers (no skeleton in the engine layer yet) --------------
    n.raw_set(
        "GetBoneNameFromTable",
        lua.create_function(move |_, (this, _idx): (Table, Value)| {
            let _ = self_id(&this);
            Ok(Value::Nil)
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetBonePos",
        lua.create_function(move |lua, (this, _name, out): (Table, Value, Option<Table>)| {
            let p = self_id(&this).map(|id| c.world.borrow().world_pos(id)).unwrap_or_default();
            vec3_out(lua, out, p)
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetHelperPos",
        lua.create_function(move |lua, (this, _name, object_space, out): (Table, Value, Option<Value>, Option<Table>)| {
            let p = self_id(&this)
                .map(|id| {
                    let w = c.world.borrow();
                    if truthy(object_space.as_ref().unwrap_or(&Value::Nil)) {
                        w.get(id).map(|e| e.pos).unwrap_or_default()
                    } else {
                        w.world_pos(id)
                    }
                })
                .unwrap_or_default();
            vec3_out(lua, out, p)
        })?,
    )?;

    // -- links ---------------------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "CountLinks",
        lua.create_function(move |_, this: Table| {
            Ok(self_id(&this).and_then(|id| c.world.borrow().get(id).map(|e| e.links.len())).unwrap_or(0))
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetLink",
        lua.create_function(move |_, (this, idx): (Table, Value)| {
            let i = num(&idx).unwrap_or(0.0) as usize;
            Ok(self_id(&this)
                .and_then(|id| c.world.borrow().get(id).and_then(|e| e.links.get(i).map(|l| l.1)))
                .map(super::host::id_value)
                .unwrap_or(Value::Nil))
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetLinkName",
        lua.create_function(move |lua, (this, target): (Table, Value)| {
            let Some(target) = value_to_id(&target) else { return Ok(Value::Nil) };
            Ok(self_id(&this)
                .and_then(|id| {
                    c.world
                        .borrow()
                        .get(id)
                        .and_then(|e| e.links.iter().find(|l| l.1 == target).map(|l| l.0.clone()))
                })
                .map(|n| Value::String(lua.create_string(n).unwrap()))
                .unwrap_or(Value::Nil))
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetLinkTarget",
        lua.create_function(move |_, (this, name, idx): (Table, Value, Value)| {
            let Some(name) = value_str(&name) else { return Ok(Value::Nil) };
            let i = num(&idx).unwrap_or(0.0) as usize;
            Ok(self_id(&this)
                .and_then(|id| {
                    c.world
                        .borrow()
                        .get(id)
                        .and_then(|e| e.links.iter().filter(|l| l.0 == name).nth(i).map(|l| l.1))
                })
                .map(super::host::id_value)
                .unwrap_or(Value::Nil))
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "SetLinkTarget",
        lua.create_function(move |_, (this, name, target): (Table, Value, Value)| {
            let (Some(id), Some(name), Some(target)) = (self_id(&this), value_str(&name), value_to_id(&target)) else {
                return Ok(());
            };
            if let Some(e) = c.world.borrow_mut().get_mut(id) {
                if let Some(l) = e.links.iter_mut().find(|l| l.0 == name) {
                    l.1 = target;
                } else {
                    e.links.push((name, target));
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "CreateLink",
        lua.create_function(move |_, (this, a, b): (Table, Value, Value)| {
            let Some(id) = self_id(&this) else { return Ok(()) };
            let (name, target) = if value_to_id(&a).is_some() && value_str(&b).is_some() {
                (value_str(&b).unwrap(), value_to_id(&a).unwrap())
            } else if value_str(&a).is_some() && value_to_id(&b).is_some() {
                (value_str(&a).unwrap(), value_to_id(&b).unwrap())
            } else {
                return Ok(());
            };
            if let Some(e) = c.world.borrow_mut().get_mut(id) {
                e.links.push((name, target));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "RemoveLink",
        lua.create_function(move |_, (this, name): (Table, Value)| {
            if let (Some(id), Some(name)) = (self_id(&this), value_str(&name)) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.links.retain(|l| l.0 != name);
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "RemoveAllLinks",
        lua.create_function(move |_, this: Table| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.links.clear();
                }
            }
            Ok(())
        })?,
    )?;

    // -- parenting / misc ----------------------------------------------------
    let c = ctx.clone();
    n.raw_set(
        "GetParent",
        lua.create_function(move |_, this: Table| {
            Ok(self_id(&this)
                .and_then(|id| c.world.borrow().get(id).and_then(|e| e.parent))
                .map(super::host::id_value)
                .unwrap_or(Value::Nil))
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "AttachChild",
        lua.create_function(move |_, (this, child, _flags): (Table, Value, Option<Value>)| {
            if let (Some(id), Some(child)) = (self_id(&this), value_to_id(&child)) {
                if let Some(e) = c.world.borrow_mut().get_mut(child) {
                    e.parent = Some(id);
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "GetDistance",
        lua.create_function(move |_, (this, other): (Table, Value)| {
            let (Some(id), Some(other)) = (self_id(&this), value_to_id(&other)) else { return Ok(0.0) };
            let w = c.world.borrow();
            let d = super::world::dist2(w.world_pos(id), w.world_pos(other)).sqrt();
            Ok(d as f64)
        })?,
    )?;
    n.raw_set(
        "RegisterState",
        lua.create_function(move |_, (this, _name): (Table, Value)| {
            let _ = self_id(&this);
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "TriggerEvent",
        lua.create_function(move |_, (this, ev): (Table, Value)| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.flow_outputs.push(FlowOutput { port: "TriggerEvent".into(), value: flow_value(&ev) });
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "StopAllSounds",
        lua.create_function(move |_, this: Table| {
            if let Some(id) = self_id(&this) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.sound_events.clear();
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    n.raw_set(
        "PlaySoundEventEx",
        lua.create_function(move |_, (this, name): (Table, Value)| {
            if let (Some(id), Some(name)) = (self_id(&this), value_str(&name)) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.sound_events.push(name);
                }
            }
            Ok(())
        })?,
    )?;
    for name in ["AddConstraint", "BreakToPieces", "PhysicalizeSlot", "CreateBoneAttachment", "ResetAttachment"] {
        n.raw_set(name, lua.create_function(|_, _: Variadic<Value>| Ok(Value::Nil))?)?;
    }

    install_actor_compat(lua, ctx)?;
    Ok(())
}

/// The actor methods the level scripts reach during boot but that the actor
/// bind does not provide yet. They are added to a pre-created `ActorNative`
/// table; `actor::install` runs later, keeps this table (its `or {}` idiom)
/// and adds its own methods, so nothing here is clobbered.
///
/// TODO: move these into the actor bind once it owns code-event / link /
/// movement-target state.
fn install_actor_compat(lua: &Lua, ctx: &Ctx) -> Result<()> {
    let actor: Table = match lua.globals().get::<Value>("ActorNative")? {
        Value::Table(t) => t,
        _ => {
            let t = lua.create_table()?;
            lua.globals().set("ActorNative", t.clone())?;
            t
        }
    };

    let c = ctx.clone();
    actor.raw_set(
        "CreateCodeEvent",
        lua.create_function(move |_, (this, ev): (Table, Value)| {
            let Some(id) = actor_entity(&this) else { return Ok(()) };
            let port = match &ev {
                Value::Table(t) => t.raw_get::<Value>("event").ok().and_then(|v| value_str(&v)),
                v => value_str(v),
            }
            .unwrap_or_else(|| "codeevent".into());
            if let Some(e) = c.world.borrow_mut().get_mut(id) {
                e.flow_outputs.push(FlowOutput { port, value: FlowValue::Nil });
            }
            Ok(())
        })?,
    )?;

    let c = ctx.clone();
    actor.raw_set(
        "LinkToEntity",
        lua.create_function(move |_, (this, target): (Table, Value)| {
            if let (Some(id), Some(target)) = (actor_entity(&this), value_to_id(&target)) {
                if let Some(a) = c.actors.borrow_mut().get_mut(&id) {
                    a.linked_vehicle = Some(target);
                }
            }
            Ok(())
        })?,
    )?;

    let c = ctx.clone();
    actor.raw_set(
        "SetMovementTarget",
        lua.create_function(move |_, (this, pos): (Table, Value)| {
            if let (Some(id), Some(pos)) = (actor_entity(&this), vec3_from(&pos)) {
                if let Some(a) = c.actors.borrow_mut().get_mut(&id) {
                    a.stored_pos = Some(pos);
                }
            }
            Ok(())
        })?,
    )?;
    Ok(())
}

/// The entity id carried by an `ActorNative` receiver (`__id`).
fn actor_entity(this: &Table) -> Option<super::world::EntityId> {
    value_to_id(&this.raw_get::<Value>("__id").ok()?)
}

impl Default for AnimState {
    fn default() -> Self {
        AnimState {
            name: String::new(),
            layer: 0,
            speed: 1.0,
            start_ms: 0,
            looping: false,
            length: 0.0,
            running: false,
        }
    }
}
