//! `CScriptBind_System` and the other engine namespaces (`System.*`,
//! `CryAction.*`, `Physics.*`, `Particle.*`, `Sound.*`, `AI.*`, ...).
//!
//! Each bind mirrors the original's argument handling and return values.
//! Unimplemented members fall through to the counted stubs installed by
//! [`super::host`].

use super::host::{lua_format, num, Ctx};
use mlua::{Lua, Result, Table, Value, Variadic};

pub fn install(lua: &Lua, ctx: &Ctx) -> Result<()> {
    install_system(lua, ctx)?;
    Ok(())
}

fn install_system(lua: &Lua, ctx: &Ctx) -> Result<()> {
    let system: Table = lua.globals().get("System")?;

    for (name, level) in [("Log", "log"), ("LogAlways", "log"), ("Warning", "warn"), ("Error", "error")] {
        let c = ctx.clone();
        system.raw_set(
            name,
            lua.create_function(move |lua, args: Variadic<Value>| {
                if c.echo_log {
                    println!("[lua {level}] {}", lua_format(lua, &args));
                }
                Ok(())
            })?,
        )?;
    }

    let c = ctx.clone();
    system.raw_set(
        "GetCurrTime",
        lua.create_function(move |_, ()| Ok(c.world.borrow().clock_ms as f64 / 1000.0))?,
    )?;
    let c = ctx.clone();
    system.raw_set(
        "GetCurrAsyncTime",
        lua.create_function(move |_, ()| Ok(c.world.borrow().clock_ms as f64 / 1000.0))?,
    )?;
    let c = ctx.clone();
    system.raw_set(
        "GetFrameTime",
        lua.create_function(move |_, ()| Ok(c.world.borrow().frame_time as f64))?,
    )?;

    let c = ctx.clone();
    system.raw_set(
        "GetCVar",
        lua.create_function(move |lua, name: String| {
            let v = c.cvars.borrow().get(&name.to_ascii_lowercase()).cloned();
            Ok(match v {
                Some(s) => match s.trim().parse::<f64>() {
                    Ok(n) => Value::Number(n),
                    Err(_) => Value::String(lua.create_string(&s)?),
                },
                None => Value::Nil,
            })
        })?,
    )?;
    let c = ctx.clone();
    system.raw_set(
        "SetCVar",
        lua.create_function(move |_, (name, v): (String, Value)| {
            let s = match &v {
                Value::String(s) => s.to_string_lossy().to_string(),
                other => num(other).map(|n| n.to_string()).unwrap_or_default(),
            };
            c.cvars.borrow_mut().insert(name.to_ascii_lowercase(), s);
            Ok(())
        })?,
    )?;
    system.raw_set("IsEditor", lua.create_function(|_, ()| Ok(false))?)?;
    system.raw_set("IsEditing", lua.create_function(|_, ()| Ok(false))?)?;
    system.raw_set("IsDevModeEnable", lua.create_function(|_, ()| Ok(false))?)?;
    system.raw_set("IsMultiplayer", lua.create_function(|_, ()| Ok(false))?)?;
    Ok(())
}
