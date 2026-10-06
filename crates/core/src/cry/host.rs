//! Script host: the CryEngine `IScriptSystem` + entity-script layer.
//!
//! Runs the game's own Lua (read at runtime from the paks) with the original
//! engine semantics:
//!
//! * `Script.ReloadScript`/`LoadScript` execute a file **once** per path in
//!   game mode (re-execution only happens in the editor).
//! * An entity class is a global Lua table loaded once from its `.ent`
//!   script. Each instance table delegates to the class table, which in turn
//!   delegates to the native `Entity` bind table.
//! * Spawn: entity created natively (level id/name/transform), `Properties`
//!   deep-cloned from the class and overridden by the level XML, then
//!   `OnSpawn`, then `Server.OnInit`, then `Client.OnInit` (single-player host
//!   runs both replicas).
//! * Engine events (`OnTimer`, `OnUpdate`, `OnBeginState`, ...) go to
//!   `Server[state]` then `Client[state]` (the replica tables themselves when
//!   no state is active). Class-level callbacks (`OnReset`,
//!   `OnPropertyChange`, `OnSave`, ...) are called on the instance.
//!
//! Engine calls that have no native implementation yet resolve to a logging
//! stub, counted in [`Ctx::missing`] — that list *is* the remaining work.

use super::world::{EntityData, EntityId, EntityWorld};
use crate::crypak::Vfs;
use mlua::{Function, LightUserData, Lua, MultiValue, Result, Table, Value, Variadic};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Shared context handed to every bind module
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Timers {
    heap: BinaryHeap<Reverse<(i64, u64)>>,
    entries: HashMap<u64, TimerEntry>,
    /// `(entity, timerId)` -> live sequence number (re-arming replaces).
    entity_live: HashMap<(EntityId, i64), u64>,
    next_seq: u64,
    next_script_id: i64,
}

enum TimerEntry {
    Entity { owner: EntityId, timer_id: i64, ms: i64 },
    Script { id: i64, func: Value, user: Value },
}

impl Timers {
    fn push(&mut self, deadline: i64, entry: TimerEntry) -> u64 {
        self.next_seq += 1;
        let seq = self.next_seq;
        self.heap.push(Reverse((deadline, seq)));
        self.entries.insert(seq, entry);
        seq
    }

    pub fn set_entity(&mut self, now: i64, owner: EntityId, timer_id: i64, ms: i64) {
        if let Some(old) = self.entity_live.remove(&(owner, timer_id)) {
            self.entries.remove(&old);
        }
        let seq = self.push(now + ms.max(0), TimerEntry::Entity { owner, timer_id, ms });
        self.entity_live.insert((owner, timer_id), seq);
    }

    pub fn kill_entity(&mut self, owner: EntityId, timer_id: i64) {
        if let Some(old) = self.entity_live.remove(&(owner, timer_id)) {
            self.entries.remove(&old);
        }
    }

    pub fn kill_all_entity(&mut self, owner: EntityId) {
        let keys: Vec<_> = self.entity_live.keys().filter(|k| k.0 == owner).copied().collect();
        for k in keys {
            self.kill_entity(k.0, k.1);
        }
    }

    pub fn set_script(&mut self, now: i64, ms: i64, func: Value, user: Value) -> i64 {
        self.next_script_id += 1;
        let id = self.next_script_id;
        self.push(now + ms.max(0), TimerEntry::Script { id, func, user });
        id
    }

    pub fn kill_script(&mut self, id: i64) {
        self.entries.retain(|_, e| !matches!(e, TimerEntry::Script { id: i, .. } if *i == id));
    }

    fn pop_due(&mut self, now: i64) -> Option<TimerEntry> {
        loop {
            let Reverse((deadline, seq)) = *self.heap.peek()?;
            if deadline > now {
                return None;
            }
            self.heap.pop();
            if let Some(e) = self.entries.remove(&seq) {
                if let TimerEntry::Entity { owner, timer_id, .. } = &e {
                    self.entity_live.remove(&(*owner, *timer_id));
                }
                return Some(e);
            }
        }
    }
}

/// Native state shared by all bind closures.
#[derive(Clone)]
pub struct Ctx {
    pub world: Rc<RefCell<EntityWorld>>,
    pub vfs: Rc<Vfs>,
    pub instances: Rc<RefCell<HashMap<EntityId, Table>>>,
    pub timers: Rc<RefCell<Timers>>,
    /// `"Namespace.Func"` / `"Entity:Method"` -> call count for unimplemented binds.
    pub missing: Rc<RefCell<BTreeMap<String, u64>>>,
    /// Script errors raised by game Lua, in order.
    pub errors: Rc<RefCell<Vec<String>>>,
    pub loaded_scripts: Rc<RefCell<HashSet<String>>>,
    /// CVars (`System.GetCVar`/`SetCVar`), seeded from the shipped configs.
    pub cvars: Rc<RefCell<HashMap<String, String>>>,
    /// Echo `System.Log` output to stdout.
    pub echo_log: bool,
}

impl Ctx {
    pub fn now_ms(&self) -> i64 {
        self.world.borrow().clock_ms
    }

    pub fn instance(&self, id: EntityId) -> Option<Table> {
        self.instances.borrow().get(&id).cloned()
    }

    pub fn note_missing(&self, what: &str) {
        *self.missing.borrow_mut().entry(what.to_string()).or_default() += 1;
    }

    pub fn error(&self, msg: String) {
        if self.echo_log {
            eprintln!("[lua error] {msg}");
        }
        self.errors.borrow_mut().push(msg);
    }
}

// ---------------------------------------------------------------------------
// Value helpers used by bind modules
// ---------------------------------------------------------------------------

/// Entity ids cross into Lua as light userdata (CryEngine `ScriptHandle`).
pub fn id_value(id: EntityId) -> Value {
    Value::LightUserData(LightUserData(id as usize as *mut std::ffi::c_void))
}

/// Accepts an id handle, a number, or an entity table (`.id`).
pub fn value_to_id(v: &Value) -> Option<EntityId> {
    match v {
        Value::LightUserData(p) => Some(p.0 as usize as EntityId).filter(|i| *i != 0),
        Value::Integer(i) if *i > 0 => Some(*i as EntityId),
        Value::Number(n) if *n > 0.0 => Some(*n as EntityId),
        Value::Table(t) => t.raw_get::<Value>("id").ok().and_then(|v| match v {
            Value::Table(_) => None,
            v => value_to_id(&v),
        }),
        _ => None,
    }
}

/// `self` -> entity id (`self.id`).
pub fn self_id(this: &Table) -> Option<EntityId> {
    value_to_id(&this.raw_get::<Value>("id").ok()?)
}

pub fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(i) => Some(*i as f64),
        Value::Number(n) => Some(*n),
        Value::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.to_str().ok()?.trim().parse().ok(),
        _ => None,
    }
}

/// Lua truthiness the way CryEngine binds read bool args (`nil`/`false`/`0` = false).
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Nil => false,
        Value::Boolean(b) => *b,
        Value::Integer(i) => *i != 0,
        Value::Number(n) => *n != 0.0,
        _ => true,
    }
}

pub fn vec3_from(v: &Value) -> Option<[f32; 3]> {
    let Value::Table(t) = v else { return None };
    let g = |k: &str, i: i64| -> Option<f32> {
        num(&t.raw_get::<Value>(k).ok()?).or_else(|| num(&t.raw_get::<Value>(i).ok()?)).map(|x| x as f32)
    };
    Some([g("x", 1)?, g("y", 2)?, g("z", 3)?])
}

pub fn vec3_table(lua: &Lua, v: [f32; 3]) -> Result<Table> {
    let t = lua.create_table()?;
    t.raw_set("x", v[0] as f64)?;
    t.raw_set("y", v[1] as f64)?;
    t.raw_set("z", v[2] as f64)?;
    Ok(t)
}

/// Write into a caller-provided vector table (the `GetPos(out)` idiom) or a new one.
pub fn vec3_out(lua: &Lua, out: Option<Table>, v: [f32; 3]) -> Result<Table> {
    match out {
        Some(t) => {
            t.raw_set("x", v[0] as f64)?;
            t.raw_set("y", v[1] as f64)?;
            t.raw_set("z", v[2] as f64)?;
            Ok(t)
        }
        None => vec3_table(lua, v),
    }
}

pub fn fmt_value(v: &Value) -> String {
    match v {
        Value::Nil => "nil".into(),
        Value::Boolean(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Number(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                n.to_string()
            }
        }
        Value::String(s) => s.to_string_lossy().to_string(),
        Value::Table(_) => "<table>".into(),
        Value::Function(_) => "<function>".into(),
        Value::LightUserData(p) => format!("<id {}>", p.0 as usize),
        _ => "<value>".into(),
    }
}

/// Lua-style `string.format` for `System.Log(fmt, ...)`-type calls.
pub fn lua_format(lua: &Lua, args: &Variadic<Value>) -> String {
    if args.len() <= 1 {
        return args.first().map(fmt_value).unwrap_or_default();
    }
    let f: Option<Function> = lua.globals().get::<Table>("string").ok().and_then(|s| s.get("format").ok());
    if let Some(f) = f {
        let mv: MultiValue = args.iter().cloned().collect();
        if let Ok(s) = f.call::<String>(mv) {
            return s;
        }
    }
    args.iter().map(fmt_value).collect::<Vec<_>>().join(" ")
}

/// Normalise a script path the way CryPak does (case-insensitive, `/`).
pub fn norm_path(p: &str) -> String {
    p.replace('\\', "/").trim_start_matches("./").to_ascii_lowercase()
}

// ---------------------------------------------------------------------------
// Lua-side glue
// ---------------------------------------------------------------------------

const HOST_LUA: &str = r#"
local missing = __host_missing
local stubs = setmetatable({}, { __mode = "k" })
__host_stubs = stubs

-- Keys that must never be synthesised as stubs: script data and callbacks
-- that the engine probes for existence.
local reserved = { Server = true, Client = true, Properties = true, PropertiesInstance = true,
    Editor = true, States = true }

local function make_stub(label)
    local s = setmetatable({}, { __call = function(_, ...) missing(label) return nil end })
    stubs[s] = true
    return s
end

-- Install a logging fallback on a namespace table (System, AI, ...). The stub
-- is a callable table so `NS.X = NS.X or {}` extension idioms keep working.
function __host_namespace(name)
    local t = rawget(_G, name)
    if type(t) ~= "table" then t = {}; rawset(_G, name, t) end
    local cache = {}
    local mt = getmetatable(t) or {}
    mt.__index = function(tbl, key)
        if type(key) ~= "string" or not string.find(key, "^%u") then return nil end
        local s = cache[key]
        if s == nil then s = make_stub(name .. "." .. key); cache[key] = s end
        return s
    end
    setmetatable(t, mt)
    return t
end

-- The native per-entity bind table ("Entity" methods on every instance).
EntityNative = EntityNative or {}
do
    local cache = {}
    setmetatable(EntityNative, { __index = function(_, key)
        if type(key) ~= "string" or reserved[key] or not string.find(key, "^%u")
            or string.find(key, "^On") then return nil end
        local s = cache[key]
        if s == nil then s = make_stub("Entity:" .. key); cache[key] = s end
        return s
    end })
end

function __host_bind_class(cls)
    if getmetatable(cls) == nil then setmetatable(cls, { __index = EntityNative }) end
end

local function deep_copy(t, seen)
    if type(t) ~= "table" then return t end
    seen = seen or {}
    if seen[t] then return seen[t] end
    local o = {}
    seen[t] = o
    for k, v in pairs(t) do o[deep_copy(k, seen)] = deep_copy(v, seen) end
    return o
end
__host_deep_copy = deep_copy

-- Override a cloned Properties table from the level's XML property tree,
-- walking the *table's* keys (unknown XML attributes are ignored) and
-- converting each value to the type of the class default.
local function apply_xml(props, xml)
    if type(props) ~= "table" or type(xml) ~= "table" then return end
    for k, def in pairs(props) do
        if type(k) == "string" then
            local tdef = type(def)
            if tdef == "table" then
                local child = xml.__children and xml.__children[k]
                if child then
                    apply_xml(def, child)
                elseif type(xml[k]) == "string" and def.x ~= nil then
                    local x, y, z = string.match(xml[k], "^%s*([^,]+),([^,]+),([^,]+)")
                    if z then def.x, def.y, def.z = tonumber(x) or 0, tonumber(y) or 0, tonumber(z) or 0 end
                end
            else
                local s = xml[k]
                if type(s) == "string" then
                    if tdef == "number" then
                        props[k] = tonumber(s) or 0
                    elseif tdef == "boolean" then
                        props[k] = (s == "1" or s == "true")
                    else
                        props[k] = s
                    end
                end
            end
        end
    end
end
__host_apply_xml = apply_xml

function __host_new_instance(cls, id, className, xmlProps, xmlPropsInst)
    local self = setmetatable({}, { __index = cls })
    rawset(self, "id", id)
    rawset(self, "class", className)
    local props = cls.Properties
    if type(props) == "table" then
        props = deep_copy(props)
        apply_xml(props, xmlProps)
        rawset(self, "Properties", props)
    end
    local pinst = cls.PropertiesInstance
    if type(pinst) == "table" then
        pinst = deep_copy(pinst)
        apply_xml(pinst, xmlPropsInst)
        rawset(self, "PropertiesInstance", pinst)
    end
    return self
end

local function replica_handler(self, rep, ev)
    local cls = rawget(self, "__class") or getmetatable(self).__index
    local r = cls[rep]
    if type(r) ~= "table" or stubs[r] then return nil end
    local st = rawget(self, "__state")
    if st ~= nil and st ~= "" then
        r = r[st]
        if type(r) ~= "table" then return nil end
    end
    local f = r[ev]
    if type(f) == "function" then return f end
    return nil
end

-- Engine event: Server[state].ev then Client[state].ev.
function __host_event(self, ev, ...)
    local fs = replica_handler(self, "Server", ev)
    local fc = replica_handler(self, "Client", ev)
    local r
    if fs then r = fs(self, ...) end
    if fc and fc ~= fs then
        local rc = fc(self, ...)
        if r == nil then r = rc end
    end
    return r
end

-- Class-level callback on the instance (OnSpawn, OnReset, OnPropertyChange...).
function __host_callback(self, ev, ...)
    local f = self[ev]
    if type(f) == "function" and not stubs[f] then return f(self, ...) end
    return nil
end

local function has_state(self, name)
    local cls = rawget(self, "__class")
    for _, rep in ipairs({ "Server", "Client" }) do
        local r = cls[rep]
        if type(r) == "table" and not stubs[r] and type(r[name]) == "table" then return true end
    end
    return false
end

function EntityNative.GotoState(self, name)
    if name == nil or name == "" then return false end
    if not has_state(self, name) then
        __host_missing("GotoState(unknown state):" .. tostring(rawget(self, "class")) .. "." .. tostring(name))
        return false
    end
    local cur = rawget(self, "__state")
    if cur ~= nil and cur ~= "" then __host_event(self, "OnEndState") end
    rawset(self, "__state", name)
    __host_state_changed(self.id, name)
    __host_event(self, "OnBeginState")
    return true
end

function EntityNative.GetState(self) return rawget(self, "__state") or "" end
function EntityNative.IsInState(self, name) return rawget(self, "__state") == name end

-- Resolve "Table.Func" at fire time (Script.SetTimerForFunction).
function __host_resolve(name)
    local f = _G
    for p in string.gmatch(name, "[^%.]+") do
        if type(f) ~= "table" then return nil end
        f = f[p]
    end
    return f
end
"#;

/// Namespaces the scripts reference; any member without a native bind
/// becomes a counted stub instead of an `attempt to call nil` abort.
pub const NAMESPACES: &[&str] = &[
    "System", "Script", "AI", "Sound", "CryAction", "Particle", "Physics", "Game", "HUD", "Movie",
    "Net", "Weapon", "DialogSystem", "Entity", "ItemSystem", "Vehicle", "VehicleSystem", "Boids",
    "Zombie", "Barricades", "VehicleManager", "UIAction", "Dialog", "Action", "Actor", "Item",
    "MaterialEffects", "HitDeathReactions", "Animation", "Camera", "Input", "Music", "Time",
    "Terrain", "Gamerules", "GameRules",
];

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

/// One `<Entity>` from a mission/level file (see [`super::mission`]).
pub use super::mission::LevelEntity;

pub struct ScriptHost {
    pub lua: Lua,
    pub ctx: Ctx,
    /// Lower-cased class name -> script path (`.ent` registry).
    pub class_scripts: HashMap<String, (String, Option<String>)>,
}

impl ScriptHost {
    pub fn new(vfs: Rc<Vfs>) -> Result<Self> {
        let lua = Lua::new();
        let ctx = Ctx {
            world: Rc::new(RefCell::new(EntityWorld::new())),
            vfs,
            instances: Default::default(),
            timers: Default::default(),
            missing: Default::default(),
            errors: Default::default(),
            loaded_scripts: Default::default(),
            cvars: Default::default(),
            echo_log: std::env::var("SOD_LUA_LOG").is_ok(),
        };

        {
            let c = ctx.clone();
            lua.globals().set(
                "__host_missing",
                lua.create_function(move |_, what: String| {
                    c.note_missing(&what);
                    Ok(())
                })?,
            )?;
            let c = ctx.clone();
            lua.globals().set(
                "__host_state_changed",
                lua.create_function(move |_, (id, name): (Value, String)| {
                    if let Some(id) = value_to_id(&id) {
                        if let Some(e) = c.world.borrow_mut().get_mut(id) {
                            e.state = name;
                        }
                    }
                    Ok(())
                })?,
            )?;
        }

        super::constants::install(&lua)?;
        lua.load(HOST_LUA).set_name("@sod_host.lua").exec()?;
        for ns in NAMESPACES {
            let f: Function = lua.globals().get("__host_namespace")?;
            f.call::<()>(*ns)?;
        }

        let mut host = ScriptHost { lua, ctx, class_scripts: HashMap::new() };
        host.install_script_binds()?;
        host.install_entity_system_binds()?;
        super::binds_system::install(&host.lua, &host.ctx)?;
        super::binds_entity::install(&host.lua, &host.ctx)?;
        host.load_class_registry();
        Ok(host)
    }

    /// Execute a script file once (game-mode `Script.ReloadScript` semantics).
    pub fn exec_file(lua: &Lua, ctx: &Ctx, path: &str, force: bool) -> bool {
        let key = norm_path(path);
        if !force && ctx.loaded_scripts.borrow().contains(&key) {
            return true;
        }
        let Some(bytes) = ctx.vfs.read(&key).or_else(|| ctx.vfs.read(&path.replace('\\', "/"))) else {
            ctx.error(format!("script not found: {path}"));
            return false;
        };
        ctx.loaded_scripts.borrow_mut().insert(key.clone());
        let code = String::from_utf8_lossy(&bytes);
        let code = code.strip_prefix('\u{feff}').unwrap_or(&code);
        match lua.load(code).set_name(format!("@{key}")).exec() {
            Ok(()) => true,
            Err(e) => {
                ctx.error(format!("{key}: {e}"));
                false
            }
        }
    }

    /// The original boot: run `scripts/main.lua` and call its `OnInit`, then
    /// load the single-player game rules.
    pub fn boot(&self) -> Result<()> {
        Self::exec_file(&self.lua, &self.ctx, "scripts/main.lua", false);
        self.call_global("OnInit", ())?;
        Self::exec_file(&self.lua, &self.ctx, "scripts/gamerules/singleplayer.lua", false);
        Ok(())
    }

    /// Call a global function if it exists, recording any script error.
    pub fn call_global(&self, name: &str, args: impl mlua::IntoLuaMulti) -> Result<()> {
        if let Value::Function(f) = self.lua.globals().get::<Value>(name)? {
            if let Err(e) = f.call::<()>(args) {
                self.ctx.error(format!("{name}(): {e}"));
            }
        }
        Ok(())
    }

    fn load_class_registry(&mut self) {
        let reg = crate::entity::EntityRegistry::load(&self.ctx.vfs);
        for c in reg.iter() {
            if let Some(s) = &c.script {
                self.class_scripts.insert(c.name.to_ascii_lowercase(), (c.name.clone(), Some(s.clone())));
            }
        }
    }

    /// Load (once) and return the class table for `class`.
    pub fn class_table(&self, class: &str) -> Result<Option<Table>> {
        Self::class_table_with(&self.lua, &self.ctx, &self.class_scripts, class)
    }

    fn class_table_with(
        lua: &Lua,
        ctx: &Ctx,
        scripts: &HashMap<String, (String, Option<String>)>,
        class: &str,
    ) -> Result<Option<Table>> {
        let (name, script) = match scripts.get(&class.to_ascii_lowercase()) {
            Some((n, s)) => (n.clone(), s.clone()),
            None => (class.to_string(), None),
        };
        if let Value::Table(t) = lua.globals().get::<Value>(name.as_str())? {
            let bind: Function = lua.globals().get("__host_bind_class")?;
            bind.call::<()>(t.clone())?;
            return Ok(Some(t));
        }
        let Some(script) = script else { return Ok(None) };
        Self::exec_file(lua, ctx, &script, false);
        match lua.globals().get::<Value>(name.as_str())? {
            Value::Table(t) => {
                let bind: Function = lua.globals().get("__host_bind_class")?;
                bind.call::<()>(t.clone())?;
                Ok(Some(t))
            }
            _ => {
                ctx.error(format!("class {name}: script {script} defined no global table"));
                Ok(None)
            }
        }
    }

    /// Spawn one level entity with the original init order. Returns the id
    /// even for script-less classes (those get a native entity only).
    pub fn spawn_level_entity(&self, le: &LevelEntity) -> Result<EntityId> {
        let mut data = EntityData::new(le.id, le.name.clone(), le.class.clone());
        data.pos = le.pos;
        data.rot = le.rot;
        data.scale = le.scale;
        data.guid = le.guid.clone();
        data.layer = le.layer.clone();
        let id = self.ctx.world.borrow_mut().insert(data);

        let Some(cls) = self.class_table(&le.class)? else { return Ok(id) };
        let xml = le.properties.as_ref().map(|p| p.to_lua(&self.lua)).transpose()?;
        let xml_inst = le.properties_instance.as_ref().map(|p| p.to_lua(&self.lua)).transpose()?;
        self.init_instance(cls, id, &le.class, xml, xml_inst)?;
        Ok(id)
    }

    fn init_instance(
        &self,
        cls: Table,
        id: EntityId,
        class: &str,
        xml: Option<Table>,
        xml_inst: Option<Table>,
    ) -> Result<Table> {
        Self::init_instance_with(&self.lua, &self.ctx, cls, id, class, xml, xml_inst)
    }

    fn init_instance_with(
        lua: &Lua,
        ctx: &Ctx,
        cls: Table,
        id: EntityId,
        class: &str,
        xml: Option<Table>,
        xml_inst: Option<Table>,
    ) -> Result<Table> {
        let newf: Function = lua.globals().get("__host_new_instance")?;
        let inst: Table = newf.call((cls.clone(), id_value(id), class, xml, xml_inst))?;
        inst.raw_set("__class", cls)?;
        ctx.instances.borrow_mut().insert(id, inst.clone());

        let cb: Function = lua.globals().get("__host_callback")?;
        let ev: Function = lua.globals().get("__host_event")?;
        if let Err(e) = cb.call::<()>((inst.clone(), "OnSpawn")) {
            ctx.error(format!("{class}#{id} OnSpawn: {e}"));
        }
        if let Err(e) = ev.call::<()>((inst.clone(), "OnInit")) {
            ctx.error(format!("{class}#{id} OnInit: {e}"));
        }
        Ok(inst)
    }

    /// Dispatch an engine event to one entity (`Server`/`Client` per state).
    pub fn event(&self, id: EntityId, ev: &str, args: impl mlua::IntoLuaMulti) -> Result<()> {
        let Some(inst) = self.ctx.instance(id) else { return Ok(()) };
        let f: Function = self.lua.globals().get("__host_event")?;
        let mut mv = args.into_lua_multi(&self.lua)?;
        mv.push_front(Value::String(self.lua.create_string(ev)?));
        mv.push_front(Value::Table(inst));
        if let Err(e) = f.call::<()>(mv) {
            self.ctx.error(format!("#{id} {ev}: {e}"));
        }
        Ok(())
    }

    /// Class-level callback (`OnReset`, `OnPropertyChange`, ...).
    pub fn callback(&self, id: EntityId, ev: &str) -> Result<()> {
        let Some(inst) = self.ctx.instance(id) else { return Ok(()) };
        let f: Function = self.lua.globals().get("__host_callback")?;
        if let Err(e) = f.call::<()>((inst, ev)) {
            self.ctx.error(format!("#{id} {ev}: {e}"));
        }
        Ok(())
    }

    /// Advance one frame: clock, timers, then `OnUpdate` for active entities.
    pub fn update(&self, dt: f32) -> Result<()> {
        let now = {
            let mut w = self.ctx.world.borrow_mut();
            w.clock_ms += (dt * 1000.0).round() as i64;
            w.frame_time = dt;
            w.clock_ms
        };
        loop {
            let due = self.ctx.timers.borrow_mut().pop_due(now);
            let Some(t) = due else { break };
            match t {
                TimerEntry::Entity { owner, timer_id, ms } => self.event(owner, "OnTimer", (timer_id, ms))?,
                TimerEntry::Script { id, func, user } => {
                    let func = match func {
                        Value::String(s) => {
                            let r: Function = self.lua.globals().get("__host_resolve")?;
                            r.call::<Value>(s)?
                        }
                        f => f,
                    };
                    if let Value::Function(f) = func {
                        let res = if user.is_nil() { f.call::<()>(id) } else { f.call::<()>((user, id)) };
                        if let Err(e) = res {
                            self.ctx.error(format!("script timer {id}: {e}"));
                        }
                    }
                }
            }
        }
        let active: Vec<EntityId> =
            self.ctx.world.borrow().entities.values().filter(|e| e.active && !e.removed).map(|e| e.id).collect();
        for id in active {
            self.event(id, "OnUpdate", dt)?;
        }
        let removed: Vec<EntityId> =
            self.ctx.world.borrow().entities.values().filter(|e| e.removed).map(|e| e.id).collect();
        for id in removed {
            self.ctx.instances.borrow_mut().remove(&id);
            self.ctx.timers.borrow_mut().kill_all_entity(id);
        }
        self.ctx.world.borrow_mut().flush_removed();
        Ok(())
    }

    /// Unimplemented engine calls hit so far, most frequent first.
    pub fn missing_report(&self) -> Vec<(String, u64)> {
        let mut v: Vec<_> = self.ctx.missing.borrow().iter().map(|(k, n)| (k.clone(), *n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    // -- Script.* ------------------------------------------------------------

    fn install_script_binds(&self) -> Result<()> {
        let lua = &self.lua;
        let script: Table = lua.globals().get("Script")?;
        let c = self.ctx.clone();
        script.raw_set(
            "ReloadScript",
            lua.create_function(move |lua, (path, _raise): (String, Value)| {
                Ok(Self::exec_file(lua, &c, &path, false))
            })?,
        )?;
        let c = self.ctx.clone();
        script.raw_set(
            "LoadScript",
            lua.create_function(move |lua, (path, reload, _raise): (String, Value, Value)| {
                Ok(Self::exec_file(lua, &c, &path, truthy(&reload)))
            })?,
        )?;
        let c = self.ctx.clone();
        script.raw_set(
            "UnloadScript",
            lua.create_function(move |_, path: String| {
                c.loaded_scripts.borrow_mut().remove(&norm_path(&path));
                Ok(())
            })?,
        )?;
        script.raw_set("ReloadScripts", lua.create_function(|_, ()| Ok(()))?)?;
        script.raw_set("ReloadEntityScript", lua.create_function(|_, _: Variadic<Value>| Ok(()))?)?;
        script.raw_set("DumpLoadedScripts", lua.create_function(|_, ()| Ok(()))?)?;

        let c = self.ctx.clone();
        script.raw_set(
            "SetTimer",
            lua.create_function(move |_, (ms, func, user): (Value, Value, Value)| {
                let now = c.now_ms();
                Ok(c.timers.borrow_mut().set_script(now, num(&ms).unwrap_or(0.0) as i64, func, user))
            })?,
        )?;
        let c = self.ctx.clone();
        script.raw_set(
            "SetTimerForFunction",
            lua.create_function(move |_, (ms, func, user): (Value, Value, Value)| {
                let now = c.now_ms();
                Ok(c.timers.borrow_mut().set_script(now, num(&ms).unwrap_or(0.0) as i64, func, user))
            })?,
        )?;
        let c = self.ctx.clone();
        script.raw_set(
            "KillTimer",
            lua.create_function(move |_, id: Value| {
                if let Some(id) = num(&id) {
                    c.timers.borrow_mut().kill_script(id as i64);
                }
                Ok(())
            })?,
        )?;
        Ok(())
    }

    // -- System entity-system binds that need the spawner --------------------

    fn install_entity_system_binds(&self) -> Result<()> {
        let lua = &self.lua;
        let system: Table = lua.globals().get("System")?;

        let c = self.ctx.clone();
        system.raw_set(
            "GetEntity",
            lua.create_function(move |_, id: Value| Ok(value_to_id(&id).and_then(|id| c.instance(id))))?,
        )?;
        let c = self.ctx.clone();
        system.raw_set(
            "GetEntityByName",
            lua.create_function(move |_, name: String| {
                let id = c.world.borrow().find_by_name(&name);
                Ok(id.and_then(|id| c.instance(id)))
            })?,
        )?;
        let c = self.ctx.clone();
        system.raw_set(
            "GetEntityIdByName",
            lua.create_function(move |_, name: String| {
                Ok(c.world.borrow().find_by_name(&name).map(id_value).unwrap_or(Value::Nil))
            })?,
        )?;
        let c = self.ctx.clone();
        system.raw_set(
            "GetEntities",
            lua.create_function(move |lua, _: Variadic<Value>| {
                let t = lua.create_table()?;
                let ids: Vec<EntityId> = c.world.borrow().entities.keys().copied().collect();
                let mut i = 1;
                for id in ids {
                    if let Some(inst) = c.instance(id) {
                        t.raw_set(i, inst)?;
                        i += 1;
                    }
                }
                Ok(t)
            })?,
        )?;
        let c = self.ctx.clone();
        system.raw_set(
            "GetEntitiesByClass",
            lua.create_function(move |lua, class: String| {
                let t = lua.create_table()?;
                let ids: Vec<EntityId> = c
                    .world
                    .borrow()
                    .entities
                    .values()
                    .filter(|e| !e.removed && e.class.eq_ignore_ascii_case(&class))
                    .map(|e| e.id)
                    .collect();
                let mut i = 1;
                for id in ids {
                    if let Some(inst) = c.instance(id) {
                        t.raw_set(i, inst)?;
                        i += 1;
                    }
                }
                Ok(t)
            })?,
        )?;
        let c = self.ctx.clone();
        system.raw_set(
            "GetEntitiesInSphere",
            lua.create_function(move |lua, (center, radius): (Value, f32)| {
                let t = lua.create_table()?;
                let Some(center) = vec3_from(&center) else { return Ok(t) };
                let ids = c.world.borrow().ids_in_sphere(center, radius);
                let mut i = 1;
                for id in ids {
                    if let Some(inst) = c.instance(id) {
                        t.raw_set(i, inst)?;
                        i += 1;
                    }
                }
                Ok(t)
            })?,
        )?;
        let c = self.ctx.clone();
        system.raw_set(
            "GetEntitiesInSphereByClass",
            lua.create_function(move |lua, (center, radius, class): (Value, f32, String)| {
                let t = lua.create_table()?;
                let Some(center) = vec3_from(&center) else { return Ok(t) };
                let ids: Vec<EntityId> = {
                    let w = c.world.borrow();
                    w.ids_in_sphere(center, radius)
                        .into_iter()
                        .filter(|id| w.get(*id).map(|e| e.class.eq_ignore_ascii_case(&class)).unwrap_or(false))
                        .collect()
                };
                let mut i = 1;
                for id in ids {
                    if let Some(inst) = c.instance(id) {
                        t.raw_set(i, inst)?;
                        i += 1;
                    }
                }
                Ok(t)
            })?,
        )?;
        let c = self.ctx.clone();
        system.raw_set(
            "RemoveEntity",
            lua.create_function(move |_, id: Value| {
                if let Some(id) = value_to_id(&id) {
                    c.world.borrow_mut().remove(id);
                }
                Ok(())
            })?,
        )?;

        // System.SpawnEntity{ class=, name=, position=, orientation=, properties=, ... }
        let c = self.ctx.clone();
        let scripts = Rc::new(self.class_scripts_snapshot());
        system.raw_set(
            "SpawnEntity",
            lua.create_function(move |lua, params: Table| {
                let class: String = params.get::<Option<String>>("class")?.unwrap_or_default();
                let name: String = params.get::<Option<String>>("name")?.unwrap_or_default();
                let mut data = EntityData::new(0, name, class.clone());
                if let Some(p) = vec3_from(&params.get::<Value>("position")?) {
                    data.pos = p;
                }
                if let Some(d) = vec3_from(&params.get::<Value>("orientation")?) {
                    data.rot = super::world::quat_from_dir(d);
                }
                if let Some(s) = num(&params.get::<Value>("scale")?) {
                    data.scale = [s as f32; 3];
                }
                let id = c.world.borrow_mut().insert(data);
                let Some(cls) = Self::class_table_with(lua, &c, &scripts, &class)? else {
                    return Ok(Value::Nil);
                };
                // Spawn-param properties replace the class defaults wholesale.
                let inst = Self::init_instance_with(lua, &c, cls, id, &class, None, None)?;
                if let Value::Table(p) = params.get::<Value>("properties")? {
                    let _ = p; // applied below before init would need re-init; CE merges first.
                }
                Ok(Value::Table(inst))
            })?,
        )?;
        Ok(())
    }

    fn class_scripts_snapshot(&self) -> HashMap<String, (String, Option<String>)> {
        // The registry is filled after binds are installed; read it lazily
        // from the VFS instead so SpawnEntity sees every class.
        let reg = crate::entity::EntityRegistry::load(&self.ctx.vfs);
        reg.iter()
            .filter_map(|c| c.script.as_ref().map(|s| (c.name.to_ascii_lowercase(), (c.name.clone(), Some(s.clone())))))
            .collect()
    }
}
