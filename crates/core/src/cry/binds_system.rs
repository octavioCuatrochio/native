//! `CScriptBind_System` and the other engine namespaces (`System.*`,
//! `CryAction.*`, `Physics.*`, `Particle.*`, `Sound.*`, `AI.*`, ...).
//!
//! Each bind mirrors the original's argument handling and return values.
//! Unimplemented members fall through to the counted stubs installed by
//! [`super::host`].

use super::host::{lua_format, num, value_to_id, Ctx};
use mlua::{Lua, Result, Table, Value, Variadic};

pub fn install(lua: &Lua, ctx: &Ctx) -> Result<()> {
    install_system(lua, ctx)?;
    install_cryaction(lua, ctx)?;
    install_ai(lua, ctx)?;
    install_net(lua, ctx)?;
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

/// `CryAction.*`: the game-framework services. A single-player host is both
/// server and client; object/flowgraph/AI bookkeeping is recorded on the
/// entity so the presentation layer can pick it up later.
fn install_cryaction(lua: &Lua, ctx: &Ctx) -> Result<()> {
    let ca: Table = lua.globals().get("CryAction")?;

    ca.raw_set("IsServer", lua.create_function(|_, ()| Ok(true))?)?;
    ca.raw_set("IsClient", lua.create_function(|_, ()| Ok(true))?)?;
    ca.raw_set("IsRMIServer", lua.create_function(|_, ()| Ok(true))?)?;
    ca.raw_set("IsGameStarted", lua.create_function(|_, ()| Ok(true))?)?;
    // The original returns a runtime flag; at the boot/level state it is on.
    ca.raw_set("IsImmersivenessEnabled", lua.create_function(|_, ()| Ok(1.0))?)?;

    let c = ctx.clone();
    ca.raw_set(
        "GetServerTime",
        lua.create_function(move |_, ()| Ok(c.world.borrow().clock_ms as f64 / 1000.0))?,
    )?;

    let c = ctx.clone();
    ca.raw_set(
        "RegisterWithAI",
        lua.create_function(move |_, (id, _rest): (Value, Variadic<Value>)| {
            if let Some(id) = value_to_id(&id) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.ai_registered = true;
                }
            }
            Ok(())
        })?,
    )?;

    // Game-object / network / flowgraph bookkeeping: accepted, no state yet.
    for name in [
        "CreateGameObjectForEntity",
        "BindGameObjectToNetwork",
        "ForceGameObjectUpdate",
        "DontSyncPhysics",
        "ActivateExtensionForGameObject",
        "ActivateEffect",
        "LoadXML",
        "SendGameplayEvent",
        "Persistant",
        "PersistantSphere",
        "PersistantArrow",
        "CacheItemGeometry",
        "CacheItemSound",
        "ClearEntityTags",
    ] {
        ca.raw_set(name, lua.create_function(|_, _: Variadic<Value>| Ok(Value::Nil))?)?;
    }
    Ok(())
}

/// `AI.*`: only the boot-time parameter reset is needed here; the goal-pipe
/// and signal API is exercised later during gameplay.
fn install_ai(lua: &Lua, ctx: &Ctx) -> Result<()> {
    let ai: Table = lua.globals().get("AI")?;
    let c = ctx.clone();
    ai.raw_set(
        "ResetParameters",
        lua.create_function(move |_, (id, _rest): (Value, Variadic<Value>)| {
            if let Some(id) = value_to_id(&id) {
                if let Some(e) = c.world.borrow_mut().get_mut(id) {
                    e.ai_registered = true;
                }
            }
            Ok(())
        })?,
    )?;
    Ok(())
}

/// `Net.Expose`: network channel setup. In a single-player host there is no
/// remote peer, so this is accepted as bookkeeping; the local `Server`/`Client`
/// replicas are already driven by the script host.
fn install_net(lua: &Lua, _ctx: &Ctx) -> Result<()> {
    let net: Table = lua.globals().get("Net")?;
    net.raw_set("Expose", lua.create_function(|_, _: Value| Ok(Value::Nil))?)?;
    Ok(())
}
