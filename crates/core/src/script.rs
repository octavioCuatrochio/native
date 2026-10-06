//! Runs the game's real Lua (CryEngine 3 used Lua 5.1) with a minimal engine
//! API shim so scripts can be loaded and exercised natively, without the game.
//!
//! Note: `mlua::Error` is not `Send`, so Lua-facing functions return
//! `mlua::Result` and are converted to `anyhow` only at the outer boundary.

use crate::crypak::Vfs;
use mlua::{Lua, Value};
use std::rc::Rc;

pub struct GameLua {
    pub lua: Lua,
    vfs: Rc<Vfs>,
}

fn fmt_value(v: &Value) -> String {
    match v {
        Value::Nil => "nil".into(),
        Value::Boolean(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.to_string_lossy().to_string(),
        other => format!("{other:?}"),
    }
}

impl GameLua {
    pub fn new(vfs: Rc<Vfs>) -> mlua::Result<Self> {
        let lua = Lua::new();

        // --- Script.* -----------------------------------------------------
        let script_t = lua.create_table()?;
        {
            let vfs2 = vfs.clone();
            script_t.set(
                "ReloadScript",
                lua.create_function(move |lua, path: String| Self::exec_vfs(lua, &vfs2, &path))?,
            )?;
            script_t.set("ReloadScripts", lua.create_function(|_, ()| Ok(()))?)?;
        }
        lua.globals().set("Script", script_t)?;

        // --- System.* -----------------------------------------------------
        let system = lua.create_table()?;
        system.set(
            "Log",
            lua.create_function(|_, v: Value| {
                println!("[Lua System.Log] {}", fmt_value(&v));
                Ok(())
            })?,
        )?;
        system.set(
            "LogAlways",
            lua.create_function(|_, v: Value| {
                println!("[Lua System.LogAlways] {}", fmt_value(&v));
                Ok(())
            })?,
        )?;
        system.set(
            "Error",
            lua.create_function(|_, v: Value| {
                eprintln!("[Lua System.Error] {}", fmt_value(&v));
                Ok(())
            })?,
        )?;
        system.set(
            "Warning",
            lua.create_function(|_, v: Value| {
                eprintln!("[Lua System.Warning] {}", fmt_value(&v));
                Ok(())
            })?,
        )?;
        system.set("GetCVar", lua.create_function(|_, _n: String| Ok(0.0f64))?)?;
        system.set("SetCVar", lua.create_function(|_, _: (String, Value)| Ok(()))?)?;
        system.set("GetCurrTime", lua.create_function(|_, ()| Ok(0.0f64))?)?;
        system.set("GetFrameTime", lua.create_function(|_, ()| Ok(0.0f64))?)?;
        system.set("GetUserName", lua.create_function(|_, ()| Ok("player".to_string()))?)?;
        system.set("IsDevModeEnable", lua.create_function(|_, ()| Ok(false))?)?;
        system.set("ShowConsole", lua.create_function(|_, _: bool| Ok(()))?)?;
        system.set("ClearConsole", lua.create_function(|_, ()| Ok(()))?)?;
        system.set("Quit", lua.create_function(|_, ()| Ok(()))?)?;
        lua.globals().set("System", system)?;

        // --- Log global (some scripts call Log(...) directly) -------------
        lua.globals().set(
            "Log",
            lua.create_function(|_, v: Value| {
                println!("[Lua Log] {}", fmt_value(&v));
                Ok(())
            })?,
        )?;

        Ok(GameLua { lua, vfs })
    }

    fn exec_vfs(lua: &Lua, vfs: &Rc<Vfs>, path: &str) -> mlua::Result<()> {
        let bytes = match vfs.read(path) {
            Some(b) => b,
            None => {
                eprintln!("[lua] (missing) {path}");
                return Ok(());
            }
        };
        let code = String::from_utf8_lossy(&bytes).to_string();
        match lua.load(&code).set_name(path).exec() {
            Ok(()) => {
                println!("[lua] ran {path} ({} bytes)", bytes.len());
                Ok(())
            }
            Err(e) => {
                eprintln!("[lua] error in {path}: {e}");
                Ok(())
            }
        }
    }

    /// Load a script from the VFS (case-insensitive).
    pub fn run(&self, path: &str) -> anyhow::Result<()> {
        Self::exec_vfs(&self.lua, &self.vfs, path).map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    /// Call a zero-argument global function if it exists; returns whether it ran.
    pub fn call_global0(&self, name: &str) -> mlua::Result<bool> {
        let f: Value = self.lua.globals().get(name)?;
        if let Value::Function(f) = f {
            match f.call::<()>(()) {
                Ok(()) => Ok(true),
                Err(e) => {
                    eprintln!("[lua] {name}() raised: {e}");
                    Ok(false)
                }
            }
        } else {
            Ok(false)
        }
    }
}
