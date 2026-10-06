//! `CScriptBind_Actor`: the `self.actor` table the actor system attaches to
//! every actor entity (player, survivors, zombies). Methods are called as
//! `self.actor:Method(...)`; the table carries the owning entity id.

use super::host::{num, value_to_id, Ctx};
use super::world::EntityId;
use mlua::{Function, Lua, Result, Table, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct ActorState {
    pub is_player: bool,
    pub health: f32,
    pub max_health: f32,
    pub stored_pos: Option<[f32; 3]>,
    pub physicalization_profile: String,
    pub spectator_mode: i32,
    pub linked_vehicle: Option<EntityId>,
}

impl ActorState {
    pub fn new(is_player: bool) -> Self {
        ActorState {
            is_player,
            health: 100.0,
            max_health: 100.0,
            stored_pos: None,
            physicalization_profile: "alive".into(),
            spectator_mode: 0,
            linked_vehicle: None,
        }
    }

    pub fn is_dead(&self) -> bool {
        self.health <= 0.0
    }
}

const ACTOR_LUA: &str = r#"
ActorNative = ActorNative or {}
local stubs = __host_stubs
local cache = {}
setmetatable(ActorNative, { __index = function(_, key)
    if type(key) ~= "string" or not string.find(key, "^%u") then return nil end
    local s = cache[key]
    if s == nil then
        s = setmetatable({}, { __call = function() __host_missing("Actor:" .. key) return nil end })
        stubs[s] = true
        cache[key] = s
    end
    return s
end })
function __actor_new(id)
    return setmetatable({ __id = id }, { __index = ActorNative })
end
"#;

fn actor_id(this: &Table) -> Option<EntityId> {
    value_to_id(&this.raw_get::<Value>("__id").ok()?)
}

/// Install the actor method table. Must run after the host prelude.
pub fn install(lua: &Lua, ctx: &Ctx) -> Result<()> {
    lua.load(ACTOR_LUA).set_name("@sod_actor.lua").exec()?;
    let t: Table = lua.globals().get("ActorNative")?;

    let with = |f: fn(&mut ActorState, &Lua, Value) -> Result<Value>| {
        let c = ctx.clone();
        lua.create_function(move |lua, (this, arg): (Table, Value)| {
            let Some(id) = actor_id(&this) else { return Ok(Value::Nil) };
            let mut actors = c.actors.borrow_mut();
            let Some(a) = actors.get_mut(&id) else { return Ok(Value::Nil) };
            f(a, lua, arg)
        })
    };

    t.raw_set("IsPlayer", with(|a, _, _| Ok(Value::Boolean(a.is_player)))?)?;
    t.raw_set("IsLocalClient", with(|a, _, _| Ok(Value::Boolean(a.is_player)))?)?;
    t.raw_set("GetHealth", with(|a, _, _| Ok(Value::Number(a.health as f64)))?)?;
    t.raw_set("GetMaxHealth", with(|a, _, _| Ok(Value::Number(a.max_health as f64)))?)?;
    t.raw_set(
        "SetHealth",
        with(|a, _, v| {
            a.health = num(&v).unwrap_or(0.0) as f32;
            Ok(Value::Nil)
        })?,
    )?;
    t.raw_set(
        "SetMaxHealth",
        with(|a, _, v| {
            a.max_health = num(&v).unwrap_or(0.0) as f32;
            a.health = a.max_health;
            Ok(Value::Nil)
        })?,
    )?;
    t.raw_set("IsDead", with(|a, _, _| Ok(Value::Boolean(a.is_dead())))?)?;
    t.raw_set(
        "Revive",
        with(|a, _, _| {
            a.health = a.max_health;
            a.physicalization_profile = "alive".into();
            Ok(Value::Nil)
        })?,
    )?;
    t.raw_set(
        "Kill",
        with(|a, _, _| {
            a.health = 0.0;
            Ok(Value::Nil)
        })?,
    )?;
    t.raw_set("GetSpectatorMode", with(|a, _, _| Ok(Value::Integer(a.spectator_mode as i64)))?)?;
    // Single-player: the local player owns channel 1, AI actors have none.
    t.raw_set("GetChannel", with(|a, _, _| Ok(Value::Integer(if a.is_player { 1 } else { 0 })))?)?;
    t.raw_set(
        "GetPhysicalizationProfile",
        with(|a, lua, _| Ok(Value::String(lua.create_string(&a.physicalization_profile)?)))?,
    )?;
    t.raw_set(
        "SetPhysicalizationProfile",
        with(|a, _, v| {
            if let Value::String(s) = v {
                a.physicalization_profile = s.to_string_lossy().to_string();
            }
            Ok(Value::Nil)
        })?,
    )?;
    t.raw_set(
        "StoredPosClear",
        with(|a, _, _| {
            a.stored_pos = None;
            Ok(Value::Nil)
        })?,
    )?;
    t.raw_set(
        "GetLinkedVehicleId",
        with(|a, _, _| Ok(a.linked_vehicle.map(super::host::id_value).unwrap_or(Value::Nil)))?,
    )?;

    // StoredPosSet() stores the entity's current world position.
    let c = ctx.clone();
    t.raw_set(
        "StoredPosSet",
        lua.create_function(move |_, this: Table| {
            if let Some(id) = actor_id(&this) {
                let p = c.world.borrow().world_pos(id);
                if let Some(a) = c.actors.borrow_mut().get_mut(&id) {
                    a.stored_pos = Some(p);
                }
            }
            Ok(())
        })?,
    )?;
    Ok(())
}

/// Create the actor record and attach `self.actor` to the entity table.
pub fn attach(lua: &Lua, ctx: &Ctx, id: EntityId, inst: &Table, is_player: bool) -> Result<()> {
    ctx.actors.borrow_mut().insert(id, ActorState::new(is_player));
    let newf: Function = lua.globals().get("__actor_new")?;
    let actor: Table = newf.call(super::host::id_value(id))?;
    inst.raw_set("actor", actor)?;
    Ok(())
}
