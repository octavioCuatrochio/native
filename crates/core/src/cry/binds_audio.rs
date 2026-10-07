//! Script sound calls on the engine sound system (`crate::audio_play`): the
//! entity's `PlaySoundEvent`/`StopSound` and the `Sound.*` playback family.
//! Event names resolve through the FMOD event database (`crate::audio_fev`).
//!
//! The sound system is attached by the host's owner (the app opens the audio
//! device and attaches a system at the device rate). Until then every call
//! is accepted and returns no handle, so scripts behave as on a silent device.

use super::host::{num, self_id, truthy, vec3_from, Ctx};
use crate::audio_play::{SoundId, SoundSystem};
use mlua::{Lua, Result, Table, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A script-visible sound handle (`Sound.LoadSound` may create one before it
/// plays; `PlaySound` starts it).
#[derive(Debug, Clone, Default)]
pub struct Handle {
    pub event: String,
    pub playing: Option<SoundId>,
    pub position: [f32; 3],
    pub volume: Option<f32>,
    pub looping: Option<bool>,
}

#[derive(Default)]
pub struct AudioState {
    pub system: Option<Arc<Mutex<SoundSystem>>>,
    pub fev: crate::audio_fev::FevDb,
    pub handles: HashMap<i64, Handle>,
    pub next: i64,
    /// Events played so far, for diagnostics (name -> count).
    pub played: HashMap<String, u32>,
    /// Requests that did not start, by name and reason (diagnostics).
    pub failed: HashMap<String, u32>,
}

fn name_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.to_string_lossy().to_string()),
        _ => None,
    }
}

/// Resolve, load (once) and start an event. Returns the script handle.
fn start(ctx: &Ctx, event: &str, position: [f32; 3], owner: Option<u64>) -> Option<i64> {
    let mut a = ctx.audio.borrow_mut();
    let sys = a.system.clone()?;
    let Some(resolved) = a.fev.resolve(event) else {
        *a.failed.entry(format!("unresolved: {event}")).or_default() += 1;
        return None;
    };
    let path = resolved.path.clone();
    let mut s = sys.lock().ok()?;
    if !s.has_event(&path) {
        let vfs = ctx.vfs.clone();
        if !s.load_event(&resolved, |p| vfs.read(p)) {
            drop(s);
            *a.failed.entry(format!("load failed: {event}")).or_default() += 1;
            return None;
        }
    }
    let id = s.play(&path, position, owner);
    drop(s);
    *a.played.entry(path.clone()).or_default() += 1;
    a.next += 1;
    let h = a.next;
    a.handles.insert(h, Handle { event: path, playing: Some(id), position, ..Default::default() });
    Some(h)
}

/// Apply `f` to a playing handle's sound.
fn with_sound(ctx: &Ctx, h: &Value, f: impl FnOnce(&mut SoundSystem, SoundId, &mut Handle)) {
    let Some(h) = num(h).map(|n| n as i64) else { return };
    let mut a = ctx.audio.borrow_mut();
    let Some(sys) = a.system.clone() else { return };
    let Some(handle) = a.handles.get_mut(&h) else { return };
    let Some(id) = handle.playing else { return };
    let guard = sys.lock();
    if let Ok(mut s) = guard {
        f(&mut s, id, handle);
    }
}

pub fn install(lua: &Lua, ctx: &Ctx) -> Result<()> {
    {
        let mut a = ctx.audio.borrow_mut();
        a.fev.load_vfs(&ctx.vfs);
    }
    let ent: Table = lua.globals().get("EntityNative")?;
    let sound: Table = lua.globals().get("Sound")?;

    // self:PlaySoundEvent(name, offset, dir, flags, semantic) -> handle
    let c = ctx.clone();
    ent.raw_set(
        "PlaySoundEvent",
        lua.create_function(move |_, (this, name, offset): (Table, Value, Value)| {
            let (Some(id), Some(name)) = (self_id(&this), name_of(&name)) else { return Ok(Value::Nil) };
            if let Some(e) = c.world.borrow_mut().get_mut(id) {
                e.sound_events.push(name.clone());
            }
            let base = c.world.borrow().world_pos(id);
            let off = vec3_from(&offset).unwrap_or([0.0; 3]);
            let pos = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
            Ok(start(&c, &name, pos, Some(id as u64)).map(Value::Integer).unwrap_or(Value::Nil))
        })?,
    )?;
    // self:PlaySoundEventEx(name, flags, volume, offset, minRadius, maxRadius, semantic)
    // Ambient volumes start at volume 0 and raise it with Sound.SetSoundVolume
    // from the area fade, so the initial volume is applied as given.
    let c = ctx.clone();
    ent.raw_set(
        "PlaySoundEventEx",
        lua.create_function(move |_, (this, name, _flags, volume, offset): (Table, Value, Value, Value, Value)| {
            let (Some(id), Some(name)) = (self_id(&this), name_of(&name)) else { return Ok(Value::Nil) };
            if let Some(e) = c.world.borrow_mut().get_mut(id) {
                e.sound_events.push(name.clone());
            }
            let base = c.world.borrow().world_pos(id);
            let off = vec3_from(&offset).unwrap_or([0.0; 3]);
            let pos = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
            let Some(h) = start(&c, &name, pos, Some(id as u64)) else { return Ok(Value::Nil) };
            if let Some(v) = num(&volume) {
                let hv = Value::Integer(h);
                with_sound(&c, &hv, |s, sid, handle| {
                    s.set_volume(sid, v as f32);
                    handle.volume = Some(v as f32);
                });
            }
            Ok(Value::Integer(h))
        })?,
    )?;
    let c = ctx.clone();
    ent.raw_set(
        "StopSound",
        lua.create_function(move |_, (_this, h): (Table, Value)| {
            with_sound(&c, &h, |s, id, handle| {
                s.stop(id);
                handle.playing = None;
            });
            Ok(())
        })?,
    )?;

    // Sound.Play(name, pos, flags, semantic) -> handle
    let c = ctx.clone();
    sound.raw_set(
        "Play",
        lua.create_function(move |_, (name, pos): (Value, Value)| {
            let Some(name) = name_of(&name) else { return Ok(Value::Nil) };
            let pos = vec3_from(&pos).unwrap_or([0.0; 3]);
            Ok(start(&c, &name, pos, None).map(Value::Integer).unwrap_or(Value::Nil))
        })?,
    )?;
    // Sound.LoadSound(name, flags) -> handle (not yet playing)
    let c = ctx.clone();
    sound.raw_set(
        "LoadSound",
        lua.create_function(move |_, (name, _flags): (Value, Value)| {
            let Some(name) = name_of(&name) else { return Ok(Value::Nil) };
            let mut a = c.audio.borrow_mut();
            a.next += 1;
            let h = a.next;
            a.handles.insert(h, Handle { event: name, ..Default::default() });
            Ok(Value::Integer(h))
        })?,
    )?;
    // Sound.PlaySound(handle)
    let c = ctx.clone();
    sound.raw_set(
        "PlaySound",
        lua.create_function(move |_, h: Value| {
            let Some(hn) = num(&h).map(|n| n as i64) else { return Ok(()) };
            let pending = c.audio.borrow().handles.get(&hn).filter(|x| x.playing.is_none()).cloned();
            if let Some(p) = pending {
                if let Some(newh) = start(&c, &p.event, p.position, None) {
                    let mut a = c.audio.borrow_mut();
                    let started = a.handles.remove(&newh);
                    if let (Some(started), Some(slot)) = (started, a.handles.get_mut(&hn)) {
                        slot.playing = started.playing;
                        slot.event = started.event;
                    }
                    let (vol, lp, id, sys) = {
                        let slot = &a.handles[&hn];
                        (slot.volume, slot.looping, slot.playing, a.system.clone())
                    };
                    if let (Some(id), Some(sys)) = (id, sys) {
                        if let Ok(mut s) = sys.lock() {
                            if let Some(v) = vol {
                                s.set_volume(id, v);
                            }
                            if let Some(l) = lp {
                                s.set_looping(id, l);
                            }
                        }
                    }
                }
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    sound.raw_set(
        "StopSound",
        lua.create_function(move |_, h: Value| {
            with_sound(&c, &h, |s, id, handle| {
                s.stop(id);
                handle.playing = None;
            });
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    sound.raw_set(
        "SetSoundVolume",
        lua.create_function(move |_, (h, v): (Value, Value)| {
            let vol = num(&v).unwrap_or(1.0) as f32;
            if let Some(hn) = num(&h).map(|n| n as i64) {
                if let Some(x) = c.audio.borrow_mut().handles.get_mut(&hn) {
                    x.volume = Some(vol);
                }
            }
            with_sound(&c, &h, |s, id, _| s.set_volume(id, vol));
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    sound.raw_set(
        "SetSoundLoop",
        lua.create_function(move |_, (h, v): (Value, Value)| {
            let lp = truthy(&v);
            if let Some(hn) = num(&h).map(|n| n as i64) {
                if let Some(x) = c.audio.borrow_mut().handles.get_mut(&hn) {
                    x.looping = Some(lp);
                }
            }
            with_sound(&c, &h, |s, id, _| s.set_looping(id, lp));
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    sound.raw_set(
        "SetSoundPosition",
        lua.create_function(move |_, (h, p): (Value, Value)| {
            if let Some(p) = vec3_from(&p) {
                with_sound(&c, &h, |s, id, handle| {
                    s.set_position(id, p);
                    handle.position = p;
                });
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    sound.raw_set(
        "SetSoundPaused",
        lua.create_function(move |_, (h, v): (Value, Value)| {
            let paused = truthy(&v);
            with_sound(&c, &h, |s, id, _| s.pause(id, paused));
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    sound.raw_set(
        "IsPlaying",
        lua.create_function(move |_, h: Value| {
            let Some(hn) = num(&h).map(|n| n as i64) else { return Ok(false) };
            let a = c.audio.borrow();
            let Some(id) = a.handles.get(&hn).and_then(|x| x.playing) else { return Ok(false) };
            let Some(sys) = a.system.clone() else { return Ok(false) };
            let playing = sys.lock().map(|s| s.is_playing(id)).unwrap_or(false);
            Ok(playing)
        })?,
    )?;
    Ok(())
}
