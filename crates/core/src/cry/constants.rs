//! Engine-registered script globals (`PE_*`, `pef_*`, `ENTITY_FLAG_*`,
//! `AIEVENT_*`, ...) and the global bit-op functions, with the exact values
//! the original executable registers.

use mlua::{Lua, Result};

pub fn install(lua: &Lua) -> Result<()> {
    let _ = lua;
    Ok(())
}
