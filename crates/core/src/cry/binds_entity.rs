//! `CScriptBind_Entity`: the native methods every entity table inherits
//! (`self:GetPos()`, `self:LoadObject(...)`, `self:SetTimer(...)`, ...).
//!
//! Methods operate on [`super::world::EntityWorld`]; anything visual is
//! recorded there and picked up by the Bevy presentation layer.

use super::host::{num, self_id, truthy, vec3_from, vec3_out, Ctx};
use super::world::{angles_from_quat, quat_from_angles, PhysKind, Slot, SlotKind};
use mlua::{Lua, Result, Table, Value};

/// Resolve a slot argument (`-1`/nil = first free slot).
fn slot_index(v: &Value, free: usize) -> usize {
    match num(v) {
        Some(n) if n >= 0.0 => n as usize,
        _ => free,
    }
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
    Ok(())
}
