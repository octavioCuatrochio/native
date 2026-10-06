//! Minimal native CryEngine-3 entity runtime for State of Decay: YOSE.
//!
//! Loads a real game entity Lua script, builds an instance table whose lookup
//! chain is `instance -> replica table (Client/Server) -> class table -> native
//! bindings`, drives the lifecycle (`OnInit`/`OnSpawn`) and the state machine
//! (`GotoState` -> `OnEndState`/`OnBeginState`) and pumps entity/`Script` timers.
//!
//! The design is taken from `notes/lua-entity-runtime.md` section 5.  The Lua
//! glue (`__index` resolution, state tables, timer dispatch) lives in a small
//! prelude installed into the same Lua state as the existing `script::GameLua`
//! shim, so `System.Log`, `Script.ReloadScript` and friends keep working.
//!
//! Threading: `mlua` runs Lua 5.1 without the `send` feature, so this runtime is
//! `!Send` (an `Rc`/`RefCell` world), which is all a single simulation thread
//! needs.

use crate::crypak::Vfs;
use crate::script::GameLua;
use mlua::{Function, Lua, Result, Table, Value, Variadic};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// Opaque entity handle handed back to native callers.
pub type EntityId = u64;

/// Entities live on the server replica by default (single-player is the host).
const DEFAULT_REPLICA: &str = "Server";

/// Upper bound on the number of archetypes [`EntityRegistry`] will keep, so a
/// hostile or corrupt VFS/tree cannot exhaust memory.
const MAX_ENT_CLASSES: usize = 4096;

// ---------------------------------------------------------------------------
// `.ent` archetype class registry
// ---------------------------------------------------------------------------

/// One entity archetype read from a CryEngine `.ent` file: a named class with
/// an optional Lua script, an optional model and its authored default
/// properties.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EntityClass {
    pub name: String,
    pub script: Option<String>,
    pub model: Option<String>,
    /// Authored defaults as `(key, value)` pairs, in encounter order. Includes
    /// the root element's attributes plus every `<Properties>` attribute.
    pub properties: Vec<(String, String)>,
}

impl EntityClass {
    /// Case-insensitive lookup of one authored property.
    pub fn property(&self, key: &str) -> Option<&str> {
        self.properties
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }
}

/// Leniently parse one `.ent` archetype.
///
/// Accepts both the flat `<Entity Name=".." Script=".."/>` form used by the
/// shipped `gamedata/entities/*.ent` files and the richer `<EntityClass>` form
/// with a nested `<Properties>` element. Malformed XML ends parsing and junk
/// input yields `None`; the function never panics.
pub fn parse_ent(bytes: &[u8]) -> Option<EntityClass> {
    let mut reader = Reader::from_reader(bytes);
    let mut buf = Vec::new();
    let mut root_tag: Option<String> = None;
    let mut props: Vec<(String, String)> = Vec::new();

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) | Err(_) => break,
            Ok(ev) => ev,
        };
        if let Event::Start(e) | Event::Empty(e) = &ev {
            let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
            let attrs = collect_attrs(e);
            if root_tag.is_none() {
                // The first element is the archetype root; its attributes are
                // class-level defaults.
                root_tag = Some(tag);
                for (k, v) in attrs {
                    set_prop(&mut props, &k, &v);
                }
            } else if tag.eq_ignore_ascii_case("Properties") || tag.eq_ignore_ascii_case("Property") {
                for (k, v) in attrs {
                    set_prop(&mut props, &k, &v);
                }
            }
        }
        buf.clear();
    }

    let tag = root_tag?;

    let name = prop_of(&props, "Name")
        .or_else(|| prop_of(&props, "EntityClass"))
        .or_else(|| prop_of(&props, "Class"))
        .or_else(|| {
            if tag.eq_ignore_ascii_case("Entity") || tag.eq_ignore_ascii_case("EntityClass") {
                None
            } else {
                Some(tag.as_str())
            }
        })
        .unwrap_or_default()
        .trim()
        .to_string();

    let script = prop_of(&props, "Script")
        .or_else(|| prop_of(&props, "Class"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let model = prop_of(&props, "object_Model")
        .or_else(|| prop_of(&props, "Model"))
        .map(|m| m.trim().replace('\\', "/"))
        .filter(|m| !m.is_empty());

    Some(EntityClass { name, script, model, properties: props })
}

fn collect_attrs(e: &BytesStart) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for a in e.attributes().flatten() {
        let key = String::from_utf8_lossy(a.key.as_ref()).to_string();
        let value = match a.unescape_value() {
            Ok(v) => v.into_owned(),
            Err(_) => String::from_utf8_lossy(&a.value).to_string(),
        };
        out.push((key, value));
    }
    out
}

/// Insert-or-overwrite by exact key (attributes are few, so a linear scan is
/// fine and keeps first-seen ordering).
fn set_prop(props: &mut Vec<(String, String)>, key: &str, value: &str) {
    if let Some((_, v)) = props.iter_mut().find(|(k, _)| k == key) {
        *v = value.to_string();
    } else {
        props.push((key.to_string(), value.to_string()));
    }
}

fn prop_of<'a>(props: &'a [(String, String)], key: &str) -> Option<&'a str> {
    props
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v.as_str())
}

/// A name-keyed registry of [`EntityClass`] archetypes.
#[derive(Debug, Default, Clone)]
pub struct EntityRegistry {
    classes: Vec<EntityClass>,
    index: HashMap<String, usize>,
}

impl EntityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace, by case-insensitive name) one archetype. Classes
    /// with an empty name are ignored.
    pub fn insert(&mut self, class: EntityClass) {
        if class.name.is_empty() {
            return;
        }
        let key = class.name.to_ascii_lowercase();
        match self.index.get(&key).copied() {
            Some(i) => self.classes[i] = class,
            None => {
                self.index.insert(key, self.classes.len());
                self.classes.push(class);
            }
        }
    }

    /// Scan a VFS for entity archetypes. Only `.ent` entries under `entities/`
    /// are considered; unreadable or malformed files are skipped silently.
    pub fn load(vfs: &Vfs) -> EntityRegistry {
        let paths = vfs.paths_with_ext(".ent");
        Self::load_paths(vfs, paths)
    }

    /// Load archetypes from explicit VFS paths. The loose-directory front-end
    /// of the VFS cannot enumerate its own entries, so callers that start from
    /// an extracted tree pass the discovered paths here.
    pub fn load_paths<I, S>(vfs: &Vfs, paths: I) -> EntityRegistry
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut reg = EntityRegistry::default();
        let mut paths: Vec<String> = paths
            .into_iter()
            .map(|p| p.as_ref().replace('\\', "/"))
            .filter(|p| {
                let lp = p.to_ascii_lowercase();
                lp.ends_with(".ent") && lp.contains("entities/")
            })
            .collect();
        paths.sort();
        for p in paths {
            if reg.len() >= MAX_ENT_CLASSES {
                break;
            }
            if let Some(bytes) = vfs.read(&p) {
                if let Some(class) = parse_ent(&bytes) {
                    reg.insert(class);
                }
            }
        }
        reg
    }

    /// Scan a real directory tree for `.ent` archetypes. Robust to missing
    /// directories and capped; intended for loose extracted trees.
    pub fn load_dir(root: &Path) -> EntityRegistry {
        let mut reg = EntityRegistry::default();
        let mut files: Vec<PathBuf> = Vec::new();
        walk_ent(root, 0, &mut files, MAX_ENT_CLASSES);
        files.sort();
        for f in files {
            if reg.len() >= MAX_ENT_CLASSES {
                break;
            }
            if let Ok(bytes) = std::fs::read(&f) {
                if let Some(class) = parse_ent(&bytes) {
                    reg.insert(class);
                }
            }
        }
        reg
    }

    /// Case-insensitive lookup by archetype name.
    pub fn get(&self, name: &str) -> Option<&EntityClass> {
        self.index
            .get(&name.to_ascii_lowercase())
            .and_then(|&i| self.classes.get(i))
    }

    pub fn len(&self) -> usize {
        self.classes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EntityClass> {
        self.classes.iter()
    }
}

fn walk_ent(dir: &Path, depth: usize, out: &mut Vec<PathBuf>, limit: usize) {
    if depth > 8 || out.len() >= limit {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        if out.len() >= limit {
            return;
        }
        let p = e.path();
        if p.is_dir() {
            walk_ent(&p, depth + 1, out, limit);
        } else if p.extension().map(|x| x.eq_ignore_ascii_case("ent")).unwrap_or(false) {
            out.push(p);
        }
    }
}

// ---------------------------------------------------------------------------
// Timer bookkeeping
// ---------------------------------------------------------------------------

enum TimerKind {
    /// `self:SetTimer(timerId, msec)` -> `OnTimer(timerId, msec)`.
    Entity,
    /// `Script.SetTimerForFunction(ms, "Table.Func", self)` / `Script.SetTimer(ms, cb)`.
    Script { func: Value, this: Value },
}

struct Timer {
    deadline: i64,
    seq: u64,
    owner: EntityId,
    /// Cancellation key: `(entity, timer_id)` for entity timers, `(0, id)` for script.
    cancel_key: (EntityId, i64),
    timer_id: i64,
    delay_ms: i64,
    kind: TimerKind,
}

// A min-heap on `(deadline, seq)`.  `kind` is deliberately not part of the
// ordering, so `Ord` is implemented by hand rather than derived.
impl PartialEq for Timer {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline && self.seq == other.seq
    }
}
impl Eq for Timer {}
impl PartialOrd for Timer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Timer {
    fn cmp(&self, other: &Self) -> Ordering {
        self.deadline.cmp(&other.deadline).then(self.seq.cmp(&other.seq))
    }
}

#[derive(Default)]
struct RtState {
    clock_ms: i64,
    next_id: EntityId,
    next_seq: u64,
    next_script_timer: i64,
    instances: HashMap<EntityId, Table>,
    timers: BinaryHeap<Reverse<Timer>>,
    cancelled: HashSet<(EntityId, i64)>,
    /// Every native call and namespace call, in order, for inspection/tests.
    calls: Vec<String>,
}

fn push_timer(
    st: &Rc<RefCell<RtState>>,
    owner: EntityId,
    cancel_key: (EntityId, i64),
    timer_id: i64,
    delay_ms: i64,
    kind: TimerKind,
) {
    let mut s = st.borrow_mut();
    let seq = s.next_seq;
    s.next_seq += 1;
    let deadline = s.clock_ms + delay_ms;
    s.timers.push(Reverse(Timer {
        deadline,
        seq,
        owner,
        cancel_key,
        timer_id,
        delay_ms,
        kind,
    }));
}

// ---------------------------------------------------------------------------
// Helpers shared by the native closures
// ---------------------------------------------------------------------------

fn fmt_value(v: &Value) -> String {
    match v {
        Value::Nil => "nil".into(),
        Value::Boolean(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.to_string_lossy().to_string(),
        Value::Table(_) => "<table>".into(),
        Value::Function(_) => "<function>".into(),
        _ => "<value>".into(),
    }
}

fn as_i64(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        Value::Number(n) => *n as i64,
        Value::String(s) => s.to_string_lossy().parse().unwrap_or(0),
        _ => 0,
    }
}

fn entity_id_of(this: &Table) -> Result<EntityId> {
    let v: Value = this.raw_get("id")?;
    Ok(match v {
        Value::Integer(i) => i as u64,
        Value::Number(n) => n as u64,
        _ => 0,
    })
}

fn value_entity_id(v: &Value) -> EntityId {
    match v {
        Value::Table(t) => entity_id_of(t).unwrap_or(0),
        Value::Integer(i) => *i as u64,
        Value::Number(n) => *n as u64,
        _ => 0,
    }
}

/// Methods that appear all over the script tree but need no semantics yet.
/// They are logged so a script's flow is visible and never aborts on a bind.
const STUB_METHODS: &[&str] = &[
    "Activate",
    "GetName",
    "SetFlags",
    "NetPresent",
    "PlaySoundEvent",
    "PlaySoundEventEx",
    "SetPhysicParams",
    "SetCharacterPhysicParams",
    "SetSlotPos",
    "SetSlotAngles",
    "GetSlotPos",
    "GetSlotAngles",
    "IsSlotValid",
    "RemoveDecals",
    "AddImpulse",
    "GetVelocity",
    "GetAngles",
    "GetWorldAngles",
    "GetDirectionVector",
    "SetPos",
    "SetWorldPos",
    "SetAngles",
    "SetWorldAngles",
    "LoadCharacter",
    "LoadSubObject",
    "LoadParticleEffect",
    "PreLoadParticleEffect",
    "StartAnimation",
    "StopAnimation",
    "GetAnimationLength",
    "GetAnimationTime",
    "SetAnimationSpeed",
    "TriggerEvent",
    "ProcessVehicleDeath",
    "EnablePhysics",
    "AddConstraint",
    "SetSlotWorldTM",
    "SetCharacterPhysicParams",
    "GetPhysicalStats",
    "AIDriver",
    "IsDead",
    "GetMass",
    "GetScale",
];

// ---------------------------------------------------------------------------
// Lua prelude
// ---------------------------------------------------------------------------

const RUNTIME_LUA: &str = r#"
-- sod-core entity runtime glue.  Installed after the native binding table.
EntityNative = EntityNative or {}
__entity_stubs = {}

local function deep_copy(t)
    local o = {}
    for k, v in pairs(t) do
        if type(v) == "table" then o[k] = deep_copy(v) else o[k] = v end
    end
    return o
end
__deep_copy = deep_copy

-- The instance __index chain: instance raw -> replica -> class -> native.
local function resolve(self, key)
    local class = rawget(self, "__class")
    local replica = rawget(self, "__replica")
    if replica ~= nil and class ~= nil then
        local rep = rawget(class, replica)
        if rep ~= nil then
            local v = rawget(rep, key)
            if v ~= nil then return v end
        end
    end
    if class ~= nil then
        local v = rawget(class, key)
        if v ~= nil then return v end
    end
    local nv = rawget(EntityNative, key)
    if nv ~= nil then return nv end
    -- Unknown engine call: hand back a logging stub so scripts never blow up
    -- with "attempt to call nil".  Lowercase keys stay nil so `if self.actor`
    -- style probes keep behaving as falsy.
    if type(key) == "string" and string.match(key, "^%u") ~= nil then
        local stub = rawget(__entity_stubs, key)
        if stub == nil then
            stub = function(s, ...)
                System.Log("[entity] missing native method: " .. key)
                return nil
            end
            rawset(__entity_stubs, key, stub)
        end
        return stub
    end
    return nil
end

function __entity_new(class, id, replica, instanceProps)
    local self = setmetatable({}, { __index = resolve })
    rawset(self, "id", id)
    rawset(self, "__class", class)
    rawset(self, "__replica", replica)
    local props = {}
    if class ~= nil and class.Properties ~= nil then props = deep_copy(class.Properties) end
    rawset(self, "Properties", props)
    rawset(self, "PropertiesInstance", instanceProps or {})
    if instanceProps ~= nil then
        for k, v in pairs(instanceProps) do props[k] = v end
    end
    return self
end

local function replica_table(self)
    local class = rawget(self, "__class")
    local replica = rawget(self, "__replica")
    if class ~= nil and replica ~= nil then return rawget(class, replica) end
    return nil
end

function __entity_state_table(self, name)
    local rep = replica_table(self)
    if rep ~= nil then
        local st = rawget(rep, name)
        if st ~= nil then return st end
    end
    local class = rawget(self, "__class")
    if class ~= nil then return rawget(class, name) end
    return nil
end

function __entity_goto_state(self, name)
    local prev = rawget(self, "__stateTable")
    if prev ~= nil and type(rawget(prev, "OnEndState")) == "function" then
        prev.OnEndState(self)
    end
    rawset(self, "__state", name)
    local st = __entity_state_table(self, name)
    rawset(self, "__stateTable", st)
    if st ~= nil and type(rawget(st, "OnBeginState")) == "function" then
        st.OnBeginState(self)
    end
end

function __entity_event(self, name, ...)
    -- State handlers first, then the entity/class, matching CryEngine.
    local st = rawget(self, "__stateTable")
    if st ~= nil then
        local f = rawget(st, name)
        if type(f) == "function" then return f(self, ...) end
    end
    local f = self[name]
    if type(f) == "function" then return f(self, ...) end
    return nil
end

function __entity_has(self, name)
    local rep = replica_table(self)
    if rep ~= nil and type(rawget(rep, name)) == "function" then return true end
    local class = rawget(self, "__class")
    if class ~= nil and type(rawget(class, name)) == "function" then return true end
    return type(rawget(EntityNative, name)) == "function"
end

-- Resolve a dotted name ("BasicAI.OnDeath") at fire time and call it.
function __call_named(name, this)
    local parts = {}
    for p in string.gmatch(name, "[^%.]+") do parts[#parts + 1] = p end
    local f = _G[parts[1]]
    for i = 2, #parts do
        if type(f) ~= "table" then return nil end
        f = f[parts[i]]
    end
    if type(f) == "function" then
        if this ~= nil then return f(this) end
        return f()
    end
    return nil
end

function EntityNative.GotoState(self, name) __entity_goto_state(self, name) end
function EntityNative.GetState(self) return rawget(self, "__state") end
function EntityNative.RegisterState(self, name) end
"#;

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

pub struct EntityRuntime {
    pub lua: Lua,
    state: Rc<RefCell<RtState>>,
    vfs: Rc<Vfs>,
}

impl EntityRuntime {
    pub fn new(vfs: Rc<Vfs>) -> Result<Self> {
        // Reuse the existing System/Script shim; take its Lua state.
        let shim = GameLua::new(vfs.clone())?;
        let lua = shim.lua;

        let state = Rc::new(RefCell::new(RtState::default()));

        install_script_timers(&lua, &state)?;
        install_natives(&lua, &state)?;
        install_namespaces(&lua, &state)?;
        install_constants(&lua)?;

        lua.load(RUNTIME_LUA).set_name("@entity_runtime.lua").exec()?;

        Ok(EntityRuntime { lua, state, vfs })
    }

    /// Load `script_path` and instantiate the global class `class`.
    ///
    /// Runs the replica `OnInit` (falling back to `CommonInit` when the script
    /// defines no init) and then `OnSpawn` when present.
    pub fn spawn(&self, script_path: &str, class: &str) -> Result<EntityId> {
        let bytes = self.vfs.read(script_path).ok_or_else(|| {
            mlua::Error::RuntimeError(format!("entity script not found in VFS: {script_path}"))
        })?;
        let code = String::from_utf8_lossy(&bytes).to_string();
        self.lua.load(code.as_str()).set_name(script_path).exec()?;

        let class_t = match self.lua.globals().get::<Value>(class)? {
            Value::Table(t) => t,
            _ => {
                return Err(mlua::Error::RuntimeError(format!(
                    "script {script_path} did not define global class {class}"
                )))
            }
        };

        let id = {
            let mut s = self.state.borrow_mut();
            s.next_id += 1;
            s.next_id
        };

        let props = self.lua.create_table()?;
        let new_fn: Function = self.lua.globals().get("__entity_new")?;
        let instance: Table = new_fn.call((class_t, id, DEFAULT_REPLICA, props))?;
        self.state.borrow_mut().instances.insert(id, instance.clone());

        let has: Function = self.lua.globals().get("__entity_has")?;
        let has_init: bool = has.call((instance.clone(), "OnInit"))?;
        let has_common: bool = has.call((instance.clone(), "CommonInit"))?;
        if has_init {
            self.fire(instance.clone(), "OnInit")?;
        } else if has_common {
            self.fire(instance.clone(), "CommonInit")?;
        }
        if has.call::<bool>((instance.clone(), "OnSpawn"))? {
            self.fire(instance.clone(), "OnSpawn")?;
        }

        Ok(id)
    }

    /// Instantiate an archetype from `registry` at `pos`.
    ///
    /// A class that carries a script is spawned through the normal
    /// [`Self::spawn`] path, using the archetype name as the Lua global class;
    /// its position and authored default properties are applied afterwards. A
    /// class without a script gets an inert placeholder instance, so callers
    /// still receive a valid handle.
    pub fn spawn_class(
        &self,
        registry: &EntityRegistry,
        class: &str,
        pos: [f32; 3],
    ) -> Result<EntityId> {
        let archetype = registry.get(class).ok_or_else(|| {
            mlua::Error::RuntimeError(format!("unknown entity class: {class}"))
        })?;
        let global = if archetype.name.is_empty() { class } else { archetype.name.as_str() };

        let id = match archetype.script.as_deref() {
            Some(script) if !script.trim().is_empty() => {
                let path = self.resolve_script(script);
                self.spawn(&path, global)?
            }
            _ => self.spawn_placeholder(global)?,
        };

        self.set_position(id, pos)?;
        self.apply_class_properties(id, &archetype.properties)?;
        Ok(id)
    }

    /// Store an instance's world position (native side; read back through
    /// [`Self::position`]).
    pub fn set_position(&self, id: EntityId, pos: [f32; 3]) -> Result<()> {
        let inst = self.instance(id)?;
        let t = self.lua.create_table()?;
        t.set("x", pos[0])?;
        t.set("y", pos[1])?;
        t.set("z", pos[2])?;
        inst.raw_set("__pos", t)?;
        Ok(())
    }

    /// The position last assigned with [`Self::set_position`]; zeros if unset.
    pub fn position(&self, id: EntityId) -> Result<[f32; 3]> {
        let inst = self.instance(id)?;
        let v: Value = inst.raw_get("__pos")?;
        if let Value::Table(t) = v {
            let comp = |k: &str| -> f32 {
                t.raw_get::<Value>(k)
                    .ok()
                    .and_then(|x| match x {
                        Value::Number(n) => Some(n as f32),
                        Value::Integer(i) => Some(i as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0)
            };
            Ok([comp("x"), comp("y"), comp("z")])
        } else {
            Ok([0.0, 0.0, 0.0])
        }
    }

    fn alloc_id(&self) -> EntityId {
        let mut s = self.state.borrow_mut();
        s.next_id += 1;
        s.next_id
    }

    /// Resolve a `.ent`-authored script path to something the VFS can read.
    /// Paks are case-insensitive but a loose extracted tree is not, so fall
    /// back to a lowercased path.
    fn resolve_script(&self, script: &str) -> String {
        let norm = script.replace('\\', "/");
        if self.vfs.read(&norm).is_some() {
            return norm;
        }
        let lower = norm.to_ascii_lowercase();
        if self.vfs.read(&lower).is_some() {
            return lower;
        }
        norm
    }

    /// Build an inert instance for a script-less archetype.
    fn spawn_placeholder(&self, class_name: &str) -> Result<EntityId> {
        let id = self.alloc_id();
        let class_t = self.lua.create_table()?;
        class_t.set("__name", class_name)?;
        class_t.set("Properties", self.lua.create_table()?)?;
        let instance_props = self.lua.create_table()?;
        let new_fn: Function = self.lua.globals().get("__entity_new")?;
        let instance: Table = new_fn.call((class_t, id, DEFAULT_REPLICA, instance_props))?;
        self.state.borrow_mut().instances.insert(id, instance);
        Ok(id)
    }

    /// Merge authored defaults into an instance's `Properties` table.
    fn apply_class_properties(&self, id: EntityId, props: &[(String, String)]) -> Result<()> {
        if props.is_empty() {
            return Ok(());
        }
        let inst = self.instance(id)?;
        let table = match inst.raw_get::<Value>("Properties")? {
            Value::Table(t) => t,
            _ => {
                let t = self.lua.create_table()?;
                inst.raw_set("Properties", t.clone())?;
                t
            }
        };
        for (k, v) in props {
            table.raw_set(k.as_str(), v.as_str())?;
        }
        Ok(())
    }

    /// Fire `OnEndState` on the old state and `OnBeginState` on `state`.
    pub fn goto_state(&self, id: EntityId, state: &str) -> Result<()> {
        let instance = self.instance(id)?;
        let f: Function = self.lua.globals().get("__entity_goto_state")?;
        f.call::<()>((instance, state))
    }

    pub fn get_state(&self, id: EntityId) -> Result<String> {
        let instance = self.instance(id)?;
        let v: Option<String> = instance.raw_get("__state")?;
        Ok(v.unwrap_or_default())
    }

    /// Advance the engine clock by `dt_ms` and dispatch every due timer.
    pub fn tick(&self, dt_ms: u32) -> Result<()> {
        self.state.borrow_mut().clock_ms += dt_ms as i64;

        loop {
            let due = {
                let mut s = self.state.borrow_mut();
                let now = s.clock_ms;
                let next_deadline = s.timers.peek().map(|r| r.0.deadline);
                match next_deadline {
                    Some(d) if d <= now => s.timers.pop().map(|r| r.0),
                    _ => None,
                }
            };

            let Some(timer) = due else { break };
            if self.state.borrow_mut().cancelled.remove(&timer.cancel_key) {
                continue;
            }

            match timer.kind {
                TimerKind::Entity => {
                    let instance = self.state.borrow().instances.get(&timer.owner).cloned();
                    if let Some(instance) = instance {
                        let f: Function = self.lua.globals().get("__entity_event")?;
                        f.call::<()>((instance, "OnTimer", timer.timer_id, timer.delay_ms))?;
                    }
                }
                TimerKind::Script { func, this } => match func {
                    Value::String(s) => {
                        let name = s.to_string_lossy().to_string();
                        let f: Function = self.lua.globals().get("__call_named")?;
                        f.call::<()>((name, this))?;
                    }
                    Value::Function(f) => {
                        f.call::<()>(this)?;
                    }
                    _ => {}
                },
            }
        }
        Ok(())
    }

    /// The ordered log of native/namespace calls, for tests and diagnostics.
    pub fn calls(&self) -> Vec<String> {
        self.state.borrow().calls.clone()
    }

    // -- internals ---------------------------------------------------------

    fn instance(&self, id: EntityId) -> Result<Table> {
        match self.state.borrow().instances.get(&id).cloned() {
            Some(t) => Ok(t),
            None => Err(mlua::Error::RuntimeError(format!("no entity with id {id}"))),
        }
    }

    fn fire(&self, this: Table, name: &str) -> Result<()> {
        let f: Function = self.lua.globals().get("__entity_event")?;
        f.call::<()>((this, name))
    }
}

// ---------------------------------------------------------------------------
// Native installation
// ---------------------------------------------------------------------------

fn install_natives(lua: &Lua, st: &Rc<RefCell<RtState>>) -> Result<()> {
    let natives = lua.create_table()?;

    // --- timers -----------------------------------------------------------
    {
        let st = st.clone();
        natives.set(
            "SetTimer",
            lua.create_function(move |_, (this, id, ms): (Table, Value, Value)| {
                let owner = entity_id_of(&this)?;
                let timer_id = as_i64(&id);
                let delay = as_i64(&ms);
                {
                    let mut s = st.borrow_mut();
                    s.calls.push(format!("entity {owner}: SetTimer({timer_id}, {delay})"));
                }
                push_timer(&st, owner, (owner, timer_id), timer_id, delay, TimerKind::Entity);
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "KillTimer",
            lua.create_function(move |_, (this, id): (Table, Value)| {
                let owner = entity_id_of(&this)?;
                let timer_id = as_i64(&id);
                st.borrow_mut().cancelled.insert((owner, timer_id));
                st.borrow_mut().calls.push(format!("entity {owner}: KillTimer({timer_id})"));
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "DeleteThis",
            lua.create_function(move |_, this: Table| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut().instances.remove(&owner);
                st.borrow_mut().calls.push(format!("entity {owner}: DeleteThis()"));
                Ok(())
            })?,
        )?;
    }

    // --- flowgraph output -------------------------------------------------
    {
        let st = st.clone();
        natives.set(
            "ActivateOutput",
            lua.create_function(move |_, (this, name, value): (Table, String, Value)| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut()
                    .calls
                    .push(format!("entity {owner}: ActivateOutput({name}, {})", fmt_value(&value)));
                Ok(())
            })?,
        )?;
    }

    // --- transforms (real enough to keep scripts running) -----------------
    for name in ["GetPos", "GetWorldPos"] {
        natives.set(
            name,
            lua.create_function(|lua, _this: Table| {
                let t = lua.create_table()?;
                t.set("x", 0.0)?;
                t.set("y", 0.0)?;
                t.set("z", 0.0)?;
                Ok(t)
            })?,
        )?;
    }

    // --- geometry / slots / physics / misc --------------------------------
    {
        let st = st.clone();
        natives.set(
            "LoadObject",
            lua.create_function(move |_, (this, slot, path): (Table, Value, Value)| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut().calls.push(format!(
                    "entity {owner}: LoadObject({}, {})",
                    as_i64(&slot),
                    fmt_value(&path)
                ));
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "DrawSlot",
            lua.create_function(move |_, (this, slot, on): (Table, Value, Value)| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut().calls.push(format!(
                    "entity {owner}: DrawSlot({}, {})",
                    as_i64(&slot),
                    as_i64(&on)
                ));
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "FreeSlot",
            lua.create_function(move |_, (this, slot): (Table, Value)| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut().calls.push(format!("entity {owner}: FreeSlot({})", as_i64(&slot)));
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "Hide",
            lua.create_function(move |_, (this, on): (Table, Value)| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut().calls.push(format!("entity {owner}: Hide({})", as_i64(&on)));
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "Physicalize",
            lua.create_function(move |_, (this, slot, ptype, params): (Table, Value, Value, Value)| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut().calls.push(format!(
                    "entity {owner}: Physicalize({}, {}, {})",
                    as_i64(&slot),
                    as_i64(&ptype),
                    fmt_value(&params)
                ));
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "SetTriggerBBox",
            lua.create_function(move |_, (this, min, max): (Table, Value, Value)| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut().calls.push(format!(
                    "entity {owner}: SetTriggerBBox({}, {})",
                    fmt_value(&min),
                    fmt_value(&max)
                ));
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "AwakePhysics",
            lua.create_function(move |_, (this, on): (Table, Value)| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut().calls.push(format!("entity {owner}: AwakePhysics({})", as_i64(&on)));
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "DestroyPhysics",
            lua.create_function(move |_, this: Table| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut().calls.push(format!("entity {owner}: DestroyPhysics()"));
                Ok(())
            })?,
        )?;
    }
    {
        let st = st.clone();
        natives.set(
            "SetViewDistRatio",
            lua.create_function(move |_, (this, ratio): (Table, Value)| {
                let owner = entity_id_of(&this)?;
                st.borrow_mut()
                    .calls
                    .push(format!("entity {owner}: SetViewDistRatio({})", fmt_value(&ratio)));
                Ok(())
            })?,
        )?;
    }

    // --- generic logged stubs --------------------------------------------
    for name in STUB_METHODS {
        let st = st.clone();
        let label = name.to_string();
        natives.set(
            *name,
            lua.create_function(move |_, args: Variadic<Value>| {
                let owner = args.first().map(value_entity_id).unwrap_or(0);
                let rendered: Vec<String> = args.iter().skip(1).map(fmt_value).collect();
                st.borrow_mut()
                    .calls
                    .push(format!("entity {owner}: {label}({})", rendered.join(", ")));
                Ok(())
            })?,
        )?;
    }

    lua.globals().set("EntityNative", natives)?;
    Ok(())
}

fn install_script_timers(lua: &Lua, st: &Rc<RefCell<RtState>>) -> Result<()> {
    let script: Table = lua.globals().get("Script")?;

    {
        let st = st.clone();
        script.set(
            "SetTimer",
            lua.create_function(move |_, (ms, cb): (Value, Value)| -> Result<i64> {
                let delay = as_i64(&ms);
                let id = {
                    let mut s = st.borrow_mut();
                    s.next_script_timer += 1;
                    s.next_script_timer
                };
                st.borrow_mut()
                    .calls
                    .push(format!("Script.SetTimer({delay}, {})", fmt_value(&cb)));
                push_timer(&st, 0, (0, id), id, delay, TimerKind::Script { func: cb, this: Value::Nil });
                Ok(id)
            })?,
        )?;
    }
    {
        let st = st.clone();
        script.set(
            "SetTimerForFunction",
            lua.create_function(move |_, (ms, func, this): (Value, Value, Value)| -> Result<i64> {
                let delay = as_i64(&ms);
                let id = {
                    let mut s = st.borrow_mut();
                    s.next_script_timer += 1;
                    s.next_script_timer
                };
                st.borrow_mut().calls.push(format!(
                    "Script.SetTimerForFunction({delay}, {})",
                    fmt_value(&func)
                ));
                push_timer(&st, 0, (0, id), id, delay, TimerKind::Script { func, this });
                Ok(id)
            })?,
        )?;
    }
    {
        let st = st.clone();
        script.set(
            "KillTimer",
            lua.create_function(move |_, id: Value| {
                let id = as_i64(&id);
                st.borrow_mut().cancelled.insert((0, id));
                st.borrow_mut().calls.push(format!("Script.KillTimer({id})"));
                Ok(())
            })?,
        )?;
    }

    Ok(())
}

fn install_namespaces(lua: &Lua, st: &Rc<RefCell<RtState>>) -> Result<()> {
    let sound = lua.create_table()?;
    for name in ["CacheAudioFile", "RemoveCachedAudioFile", "Precache", "LoadSound"] {
        let st = st.clone();
        let label = name.to_string();
        sound.set(
            name,
            lua.create_function(move |_, args: Variadic<Value>| {
                let rendered: Vec<String> = args.iter().map(fmt_value).collect();
                st.borrow_mut().calls.push(format!("Sound.{label}({})", rendered.join(", ")));
                Ok(())
            })?,
        )?;
    }
    lua.globals().set("Sound", sound)?;
    Ok(())
}

fn install_constants(lua: &Lua) -> Result<()> {
    let g = lua.globals();
    g.set("PE_NONE", 0i64)?;
    g.set("PE_STATIC", 1i64)?;
    g.set("PE_RIGID", 2i64)?;
    g.set("PE_LIVING", 3i64)?;
    g.set("NULL_ENTITY", 0i64)?;

    // `EmptyString` normally comes from scripts/common.lua; provide it so entity
    // scripts can be loaded standalone.
    g.set(
        "EmptyString",
        lua.create_function(|_, v: Value| {
            Ok(match v {
                Value::Nil => true,
                Value::String(s) => s.as_bytes().is_empty(),
                _ => false,
            })
        })?,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    const GAME: &str =
        "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";
    const EXTRACTED: &str = "/home/battlestation/Desktop/sod-re/extracted-game/gamedata";

    fn collect_paks(root: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(root) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    collect_paks(&p, out);
                } else if p.extension().map(|x| x == "pak").unwrap_or(false) {
                    out.push(p);
                }
            }
        }
    }

    /// Prefer the real game paks; fall back to the loose extracted-game tree so
    /// the test still runs if the paks are not mounted/accessible.
    fn build_vfs() -> (Rc<Vfs>, &'static str) {
        let mut vfs = Vfs::new();
        let mut paks = Vec::new();
        collect_paks(Path::new(GAME), &mut paks);
        paks.sort();
        for p in &paks {
            let _ = vfs.mount(p);
        }
        if vfs.read("scripts/entities/ul/ul_searchable.lua").is_some() {
            return (Rc::new(vfs), "game paks");
        }

        let mut loose = Vfs::new();
        if Path::new(EXTRACTED).join("scripts/entities/ul/ul_searchable.lua").is_file() {
            loose.set_loose_root(EXTRACTED);
            return (Rc::new(loose), "loose extracted-game tree (game paks unavailable)");
        }

        (Rc::new(loose), "EMPTY VFS (no source available!)")
    }

    #[test]
    fn ul_searchable_lifecycle_and_timer() {
        let (vfs, source) = build_vfs();
        eprintln!("[entity test] entity scripts sourced from: {source}");

        let rt = EntityRuntime::new(vfs).expect("EntityRuntime::new");
        let id = rt
            .spawn("scripts/entities/ul/ul_searchable.lua", "UL_Searchable")
            .expect("spawn UL_Searchable");

        // CommonInit -> Reload -> GotoState("NotSearched")
        assert_eq!(rt.get_state(id).unwrap(), "NotSearched");
        let calls = rt.calls();
        assert!(
            calls.iter().any(|c| c.contains("LoadObject(0,")),
            "expected a LoadObject from Reload; calls={calls:?}"
        );
        assert!(
            calls.iter().any(|c| c.contains("DrawSlot(0, 1)")),
            "NotSearched.OnBeginState should draw slot 0; calls={calls:?}"
        );

        // Real state transition, driven from the instance's own class table.
        rt.goto_state(id, "Searched").expect("goto_state");
        assert_eq!(rt.get_state(id).unwrap(), "Searched");
        let calls = rt.calls();
        assert!(
            calls.iter().any(|c| c.contains("DrawSlot(1, 1)")),
            "Searched.OnBeginState should draw slot 1; calls={calls:?}"
        );

        // Timer path: AudioCache's OnInit sets an entity timer, OnTimer deletes
        // the entity.  This exercises SetTimer -> tick -> OnTimer -> DeleteThis.
        let ac = rt
            .spawn("scripts/entities/ul/audiocache.lua", "AudioCache")
            .expect("spawn AudioCache");
        rt.tick(1000).expect("tick");
        let calls = rt.calls();
        assert!(
            calls.iter().any(|c| c.contains("SetTimer(0,")),
            "AudioCache.OnInit should SetTimer; calls={calls:?}"
        );
        assert!(
            calls.iter().any(|c| c.contains("Sound.RemoveCachedAudioFile")),
            "tick should fire OnTimer -> RemoveCachedAudioFile; calls={calls:?}"
        );
        assert!(
            !rt.state.borrow().instances.contains_key(&ac),
            "OnTimer should have DeleteThis'd the AudioCache entity"
        );

        // Unknown engine calls must log and return, not crash the script.
        let inst = rt.state.borrow().instances.get(&id).cloned().unwrap();
        rt.lua.globals().set("__probe_entity", inst).unwrap();
        rt.lua
            .load("__probe_entity:ThisMethodDoesNotExist(1, 2)")
            .exec()
            .expect("unknown native method must not raise");
    }

    // -- `.ent` class registry -------------------------------------------------

    /// Rich `<EntityClass>` form with a nested `<Properties>` element, as the
    /// CryEngine editor writes archetypes.
    const RICH_ENT: &str = r#"<?xml version="1.0"?>
<EntityClass Name="UL_Searchable" EntityClass="UL_Searchable">
  <Properties object_Model="Objects\Decor\Sign.cgf" Script="Scripts/Entities/UL/UL_Searchable.lua" CastShadow="1" bIsSaveable="1"/>
</EntityClass>"#;

    /// Flat form used by the shipped `gamedata/entities/*.ent` files.
    const FLAT_ENT: &str = r#"<Entity Name="TagPoint" Script="Scripts/Entities/AI/TagPoint.lua" Invisible="1"/>"#;

    #[test]
    fn parse_ent_rich_form_yields_name_model_and_properties() {
        let c = parse_ent(RICH_ENT.as_bytes()).expect("parse rich .ent");
        assert_eq!(c.name, "UL_Searchable");
        assert_eq!(c.model.as_deref(), Some("Objects/Decor/Sign.cgf"));
        assert_eq!(c.script.as_deref(), Some("Scripts/Entities/UL/UL_Searchable.lua"));
        // Authored properties keep their raw value; `model` is normalised.
        assert_eq!(c.property("object_Model"), Some(r"Objects\Decor\Sign.cgf"));
        // Case-insensitive property lookup.
        assert_eq!(c.property("castshadow"), Some("1"));
        assert_eq!(c.property("bisSaveable"), Some("1"));
        // Root attributes are folded in too.
        assert_eq!(c.property("EntityClass"), Some("UL_Searchable"));
        assert_eq!(c.property("does_not_exist"), None);
    }

    #[test]
    fn parse_ent_flat_form_matches_shipped_entities() {
        let c = parse_ent(FLAT_ENT.as_bytes()).expect("parse flat .ent");
        assert_eq!(c.name, "TagPoint");
        assert_eq!(c.script.as_deref(), Some("Scripts/Entities/AI/TagPoint.lua"));
        assert_eq!(c.model, None);
        assert_eq!(c.property("Invisible"), Some("1"));
    }

    #[test]
    fn parse_ent_is_lenient_on_junk() {
        assert!(parse_ent(b"").is_none());
        assert!(parse_ent(b"    \n\t").is_none());
        assert!(parse_ent(b"not xml at all").is_none());
        // Whatever these produce, they must not panic.
        for junk in [
            b"<".as_slice(),
            b"<bad attr",
            b"<Entity Name=/>",
            b"<a><b><c></a>",
            b"\xff\xfe<Entity Name=\"x\"/>",
            b"<Entity Name=\"x\" Script=\"s.lua\">",
        ] {
            let _ = parse_ent(junk);
        }
    }

    #[test]
    fn registry_load_dir_on_real_tree_if_present() {
        let dir = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../extracted-game/gamedata/entities"
        ));
        if !dir.is_dir() {
            return; // extraction absent: skip silently
        }
        let reg = EntityRegistry::load_dir(dir);
        assert!(reg.len() > 10, "expected >10 .ent classes, got {}", reg.len());
        eprintln!("[entity registry] load_dir loaded {} classes from {}", reg.len(), dir.display());
        assert!(
            reg.iter().all(|c| !c.name.is_empty()),
            "every loaded class must have a non-empty name"
        );
        // Case-insensitive lookup against a known shipped archetype.
        let searchable = reg.get("ul_searchable").expect("UL_Searchable present");
        assert_eq!(searchable.script.as_deref(), Some("Scripts/Entities/UL/UL_Searchable.lua"));
    }

    #[test]
    fn registry_load_over_vfs_and_loose_tree() {
        // 1) A VFS with the real `gamedata.pak` exercises `load`.
        let gamedata = Path::new(GAME).join("Game").join("gamedata.pak");
        if gamedata.is_file() {
            let mut vfs = Vfs::new();
            if vfs.mount(&gamedata).is_ok() {
                let reg = EntityRegistry::load(&vfs);
                assert!(reg.len() > 10, "load() found {} classes in gamedata.pak", reg.len());
                eprintln!("[entity registry] load(gamedata.pak) loaded {} classes", reg.len());
                assert!(reg.iter().all(|c| !c.name.is_empty()));
            }
        }

        // 2) The loose extracted tree: mount the gamedata dir and feed the
        //    enumerated paths through a loose-rooted VFS.
        let dir = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../extracted-game/gamedata/entities"
        ));
        if !dir.is_dir() {
            return; // extraction absent: skip silently
        }
        let mut vfs = Vfs::new();
        vfs.set_loose_root(dir.parent().expect("gamedata dir"));
        let mut paths = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                if e.path().extension().map(|x| x.eq_ignore_ascii_case("ent")).unwrap_or(false) {
                    paths.push(format!("entities/{}", e.file_name().to_string_lossy()));
                }
            }
        }
        let reg = EntityRegistry::load_paths(&vfs, paths);
        assert!(reg.len() > 10, "loose load found {} classes", reg.len());
        eprintln!("[entity registry] loose load_paths loaded {} classes", reg.len());
        assert!(reg.iter().all(|c| !c.name.is_empty()));

        // Bonus: a real script-backed class spawns through `spawn_class`.
        let rt = EntityRuntime::new(Rc::new(vfs)).expect("EntityRuntime::new");
        let id = rt
            .spawn_class(&reg, "UL_Searchable", [1.0, 2.0, 3.0])
            .expect("spawn_class UL_Searchable");
        assert_eq!(rt.get_state(id).unwrap(), "NotSearched");
        assert_eq!(rt.position(id).unwrap(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn spawn_class_placeholder_without_script() {
        let rt = EntityRuntime::new(Rc::new(Vfs::new())).expect("EntityRuntime::new");
        let mut reg = EntityRegistry::default();
        reg.insert(EntityClass {
            name: "UL_Placeholder".into(),
            script: None,
            model: Some("objects/decor/box.cgf".into()),
            properties: vec![
                ("object_Model".into(), "objects/decor/box.cgf".into()),
                ("Invisible".into(), "1".into()),
            ],
        });

        let id = rt
            .spawn_class(&reg, "UL_Placeholder", [4.0, 5.0, 6.0])
            .expect("spawn_class placeholder");
        assert_eq!(rt.position(id).unwrap(), [4.0, 5.0, 6.0]);

        let inst = rt.instance(id).expect("placeholder instance exists");
        let props: Table = inst.raw_get("Properties").expect("Properties table");
        assert_eq!(props.raw_get::<String>("Invisible").unwrap(), "1");
        assert_eq!(props.raw_get::<String>("object_Model").unwrap(), "objects/decor/box.cgf");

        // Unknown classes are reported, not silently ignored.
        assert!(rt.spawn_class(&reg, "No_Such_Class", [0.0; 3]).is_err());
    }
}
