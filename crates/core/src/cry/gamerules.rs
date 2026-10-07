//! Game-rules bind (`self.game` on the rules entity): the native registries
//! the game-rules script drives — hit materials/types, teams, spawn locations
//! and objectives. Damage-pipeline methods (`ServerHit`, `ServerExplosion`,
//! ...) stay counted stubs until combat is ported.

use super::host::{id_value, value_to_id, Ctx};
use super::world::EntityId;
use mlua::{Function, Lua, Result, Table, Value, Variadic};
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct GameRulesState {
    pub hit_materials: Vec<String>,
    pub hit_types: Vec<String>,
    pub teams: HashMap<EntityId, i64>,
    pub team_names: Vec<String>,
    /// (spawn entity, group id)
    pub spawn_locations: Vec<(EntityId, Option<EntityId>)>,
    pub objectives: HashMap<String, (i64, Option<EntityId>)>,
}

fn name_id(list: &mut Vec<String>, name: &str, register: bool) -> i64 {
    if let Some(i) = list.iter().position(|n| n.eq_ignore_ascii_case(name)) {
        return i as i64 + 1;
    }
    if register {
        list.push(name.to_string());
        return list.len() as i64;
    }
    0
}

const RULES_LUA: &str = r#"
GameRulesNative = GameRulesNative or {}
local stubs = __host_stubs
local cache = {}
setmetatable(GameRulesNative, { __index = function(_, key)
    if type(key) ~= "string" or not string.find(key, "^%u") then return nil end
    local s = cache[key]
    if s == nil then
        s = setmetatable({}, { __call = function() __host_missing("GameRules:" .. key) return nil end })
        stubs[s] = true
        cache[key] = s
    end
    return s
end })
function __gamerules_new() return setmetatable({}, { __index = GameRulesNative }) end
"#;

pub fn install(lua: &Lua, ctx: &Ctx) -> Result<()> {
    lua.load(RULES_LUA).set_name("@sod_gamerules.lua").exec()?;
    let t: Table = lua.globals().get("GameRulesNative")?;

    for (reg, get, materials) in [
        ("RegisterHitMaterial", "GetHitMaterialId", true),
        ("RegisterHitType", "GetHitTypeId", false),
    ] {
        let c = ctx.clone();
        t.raw_set(
            reg,
            lua.create_function(move |_, (_this, name): (Table, String)| {
                let mut g = c.game_rules.borrow_mut();
                let list = if materials { &mut g.hit_materials } else { &mut g.hit_types };
                Ok(name_id(list, &name, true))
            })?,
        )?;
        let c = ctx.clone();
        t.raw_set(
            get,
            lua.create_function(move |_, (_this, name): (Table, String)| {
                let mut g = c.game_rules.borrow_mut();
                let list = if materials { &mut g.hit_materials } else { &mut g.hit_types };
                Ok(name_id(list, &name, false))
            })?,
        )?;
    }

    let c = ctx.clone();
    t.raw_set(
        "GetTeam",
        lua.create_function(move |_, (_this, id): (Table, Value)| {
            Ok(value_to_id(&id).and_then(|id| c.game_rules.borrow().teams.get(&id).copied()).unwrap_or(0))
        })?,
    )?;
    let c = ctx.clone();
    t.raw_set(
        "SetTeam",
        lua.create_function(move |_, (_this, team, id): (Table, i64, Value)| {
            if let Some(id) = value_to_id(&id) {
                c.game_rules.borrow_mut().teams.insert(id, team);
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    t.raw_set(
        "GetTeamId",
        lua.create_function(move |_, (_this, name): (Table, String)| {
            Ok(name_id(&mut c.game_rules.borrow_mut().team_names, &name, false))
        })?,
    )?;

    // Players: the single local actor.
    let c = ctx.clone();
    t.raw_set(
        "IsPlayer",
        lua.create_function(move |_, (_this, id): (Table, Value)| {
            Ok(value_to_id(&id).and_then(|id| c.actors.borrow().get(&id).map(|a| a.is_player)).unwrap_or(false))
        })?,
    )?;
    let c = ctx.clone();
    t.raw_set(
        "IsPlayerInGame",
        lua.create_function(move |_, (_this, id): (Table, Value)| {
            Ok(value_to_id(&id).and_then(|id| c.actors.borrow().get(&id).map(|a| a.is_player)).unwrap_or(false))
        })?,
    )?;
    let c = ctx.clone();
    t.raw_set(
        "GetPlayers",
        lua.create_function(move |lua, _this: Table| {
            let out = lua.create_table()?;
            let ids: Vec<EntityId> =
                c.actors.borrow().iter().filter(|(_, a)| a.is_player).map(|(id, _)| *id).collect();
            for (i, id) in ids.into_iter().enumerate() {
                if let Some(inst) = c.instance(id) {
                    out.raw_set(i + 1, inst)?;
                }
            }
            Ok(out)
        })?,
    )?;
    t.raw_set("IsDemoMode", lua.create_function(|_, _: Variadic<Value>| Ok(false))?)?;
    t.raw_set("IsInvulnerable", lua.create_function(|_, _: Variadic<Value>| Ok(false))?)?;

    // Spawn locations / groups.
    let c = ctx.clone();
    t.raw_set(
        "AddSpawnLocation",
        lua.create_function(move |_, (_this, id, group): (Table, Value, Value)| {
            if let Some(id) = value_to_id(&id) {
                c.game_rules.borrow_mut().spawn_locations.push((id, value_to_id(&group)));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    t.raw_set(
        "RemoveSpawnLocation",
        lua.create_function(move |_, (_this, id): (Table, Value)| {
            if let Some(id) = value_to_id(&id) {
                c.game_rules.borrow_mut().spawn_locations.retain(|(e, _)| *e != id);
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    t.raw_set(
        "RemoveSpawnGroup",
        lua.create_function(move |_, (_this, group): (Table, Value)| {
            if let Some(g) = value_to_id(&group) {
                c.game_rules.borrow_mut().spawn_locations.retain(|(_, gr)| *gr != Some(g));
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    t.raw_set(
        "GetFirstSpawnLocation",
        lua.create_function(move |_, _: Variadic<Value>| {
            Ok(c.game_rules.borrow().spawn_locations.first().map(|(e, _)| id_value(*e)).unwrap_or(Value::Nil))
        })?,
    )?;

    // Objectives.
    let c = ctx.clone();
    t.raw_set(
        "AddObjective",
        lua.create_function(move |_, (_this, _team, name, status, entity): (Table, Value, String, Option<i64>, Value)| {
            c.game_rules.borrow_mut().objectives.insert(name, (status.unwrap_or(0), value_to_id(&entity)));
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    t.raw_set(
        "SetObjectiveStatus",
        lua.create_function(move |_, (_this, _team, name, status): (Table, Value, String, i64)| {
            c.game_rules.borrow_mut().objectives.entry(name).or_insert((0, None)).0 = status;
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    t.raw_set(
        "SetObjectiveEntity",
        lua.create_function(move |_, (_this, _team, name, entity): (Table, Value, String, Value)| {
            c.game_rules.borrow_mut().objectives.entry(name).or_insert((0, None)).1 = value_to_id(&entity);
            Ok(())
        })?,
    )?;
    Ok(())
}

/// Attach `self.game` to the rules entity table.
pub fn attach(lua: &Lua, inst: &Table) -> Result<()> {
    let newf: Function = lua.globals().get("__gamerules_new")?;
    let game: Table = newf.call(())?;
    inst.raw_set("game", game)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_registry_is_one_based_and_case_insensitive() {
        let mut v = Vec::new();
        assert_eq!(name_id(&mut v, "flesh", false), 0);
        assert_eq!(name_id(&mut v, "flesh", true), 1);
        assert_eq!(name_id(&mut v, "metal", true), 2);
        assert_eq!(name_id(&mut v, "FLESH", false), 1);
    }
}
