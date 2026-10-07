//! `Sound.*` state setters. Playback is not implemented yet; these record the
//! per-sound state scripts set so the audio layer can apply it later.

use super::host::{num, truthy, Ctx};
use mlua::{Lua, Result, Table, Value};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SoundState {
    pub volume: f32,
    pub fade_time: f32,
    pub paused: bool,
}

fn key(v: &Value) -> Option<u64> {
    match v {
        Value::LightUserData(p) => Some(p.0 as usize as u64),
        other => num(other).map(|n| n as u64),
    }
}

pub fn install(lua: &Lua, ctx: &Ctx) -> Result<()> {
    let sound: Table = lua.globals().get("Sound")?;
    let c = ctx.clone();
    sound.raw_set(
        "SetSoundVolume",
        lua.create_function(move |_, (id, v): (Value, Value)| {
            if let Some(k) = key(&id) {
                c.sounds.borrow_mut().entry(k).or_default().volume = num(&v).unwrap_or(0.0) as f32;
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    sound.raw_set(
        "SetFadeTime",
        lua.create_function(move |_, (id, v): (Value, Value)| {
            if let Some(k) = key(&id) {
                c.sounds.borrow_mut().entry(k).or_default().fade_time = num(&v).unwrap_or(0.0) as f32;
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    sound.raw_set(
        "SetSoundPaused",
        lua.create_function(move |_, (id, v): (Value, Value)| {
            if let Some(k) = key(&id) {
                c.sounds.borrow_mut().entry(k).or_default().paused = truthy(&v);
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    sound.raw_set(
        "SetMasterVolumeScale",
        lua.create_function(move |_, v: Value| {
            *c.master_volume.borrow_mut() = num(&v).unwrap_or(1.0) as f32;
            Ok(())
        })?,
    )?;
    Ok(())
}
