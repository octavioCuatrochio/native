//! Native data model behind the `AI.*` script binds.
//!
//! The original `CAISystem` keeps its script-authored configuration in native
//! registries built while `scripts/ai/aiconfig.lua` and the files it loads run:
//! goal pipes (an ordered list of goals), formations (an ordered list of
//! formation points), the behaviour / character catalogues, combat classes,
//! pathfinding property sets and tactical-point queries. This module is the
//! Rust representation of those registries; the binds in [`super::ai`] fill it
//! and the runtime AI (out of scope here) would consume it.
//!
//! Nothing here is game content copied from the scripts — only the shapes the
//! scripts drive, recovered from the `AI.*` registration at `0x00968ab0` and
//! the `Load*` implementations (`0x0097e510`, `0x0097e660`, `0x0097e7a0`).

use mlua::{Table, Value};
use std::collections::HashMap;

/// One argument recorded on a goal (`AI.PushGoal` tail).
#[derive(Clone, Debug, PartialEq)]
pub enum GoalParam {
    Nil,
    Bool(bool),
    Num(f64),
    Str(String),
    /// CryEngine `ScriptHandle` (entity id) crosses as light userdata.
    Id(u64),
    Table,
}

impl GoalParam {
    pub fn from_value(v: &Value) -> Self {
        match v {
            Value::Nil => GoalParam::Nil,
            Value::Boolean(b) => GoalParam::Bool(*b),
            Value::Integer(i) => GoalParam::Num(*i as f64),
            Value::Number(n) => GoalParam::Num(*n),
            Value::String(s) => GoalParam::Str(s.to_string_lossy().to_string()),
            Value::LightUserData(p) => GoalParam::Id(p.0 as usize as u64),
            Value::Table(_) => GoalParam::Table,
            _ => GoalParam::Nil,
        }
    }

    pub fn as_num(&self) -> Option<f64> {
        match self {
            GoalParam::Num(n) => Some(*n),
            GoalParam::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GoalParam::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// One goal inside a goal pipe.
#[derive(Clone, Debug, PartialEq)]
pub struct Goal {
    pub name: String,
    /// `AI.PushGoal`'s second (blocking) flag; a label has `name == ""`.
    pub blocking: bool,
    pub params: Vec<GoalParam>,
    /// `AI.PushLabel` target (goals carry the labels they branch to instead).
    pub label: bool,
}

/// A named, ordered goal list (`CGoalPipe`).
#[derive(Clone, Debug, Default)]
pub struct GoalPipe {
    pub name: String,
    pub goals: Vec<Goal>,
}

/// One point of a formation (`AI.AddFormationPoint`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FormationPoint {
    pub sight: f64,
    pub distance: f64,
    pub offset: f64,
    pub unit_class: i64,
    pub fixed: bool,
}

/// A named formation (`AI.CreateFormation`).
#[derive(Clone, Debug, Default)]
pub struct Formation {
    pub name: String,
    pub points: Vec<FormationPoint>,
}

/// A behaviour discovered by `AI.LoadBehaviors` (name + optional base name).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BehaviorDef {
    pub name: String,
    pub parent: Option<String>,
}

/// One combat class row (`AI.AddCombatClass`).
#[derive(Clone, Debug, Default)]
pub struct CombatClass {
    pub id: i64,
    /// The 20 class-vs-class assessment multipliers.
    pub scale: Vec<f64>,
    pub seen_signal: Option<String>,
}

/// One `AI.AssignPFPropertiesToPathType` / `AI.LoadPathfindProperties` record.
#[derive(Clone, Debug, Default)]
pub struct PathfindProps {
    pub path_type: String,
    pub nav_mask: i64,
    pub params: Vec<f64>,
    pub id: i64,
}

/// A tactical-point query parameter value.
#[derive(Clone, Debug, PartialEq)]
pub enum TpsVal {
    Nil,
    Bool(bool),
    Num(f64),
    Str(String),
}

impl TpsVal {
    fn from_value(v: &Value) -> Self {
        match v {
            Value::Boolean(b) => TpsVal::Bool(*b),
            Value::Integer(i) => TpsVal::Num(*i as f64),
            Value::Number(n) => TpsVal::Num(*n),
            Value::String(s) => TpsVal::Str(s.to_string_lossy().to_string()),
            _ => TpsVal::Nil,
        }
    }
}

/// One sub-table of a tactical-point query (`Generation`/`Conditions`/`Weights`/...).
#[derive(Clone, Debug, Default)]
pub struct TpsBlock {
    pub kind: String,
    pub entries: Vec<(String, TpsVal)>,
}

/// A registered tactical-point query (`AI.RegisterTacticalPointQuery`).
#[derive(Clone, Debug, Default)]
pub struct TpsQuery {
    pub name: String,
    pub blocks: Vec<TpsBlock>,
}

/// The complete script-authored AI registry.
#[derive(Default)]
pub struct AiState {
    pub goal_pipes: Vec<GoalPipe>,
    pub pipe_index: HashMap<String, usize>,
    /// Pipe currently being appended to (`BeginGoalPipe`/`CreateGoalPipe`).
    pub current_pipe: Option<String>,

    pub formations: Vec<Formation>,
    pub formation_index: HashMap<String, usize>,

    pub behaviors: Vec<BehaviorDef>,
    pub behavior_index: HashMap<String, usize>,
    pub group_behaviors: Vec<BehaviorDef>,
    pub group_behavior_index: HashMap<String, usize>,
    pub characters: Vec<String>,
    pub character_index: HashMap<String, usize>,

    pub combat_classes: Vec<CombatClass>,
    pub assessment: HashMap<i64, f64>,
    pub pathfind_props: Vec<PathfindProps>,
    pub tps_queries: Vec<TpsQuery>,
    pub tps_index: HashMap<String, usize>,

    /// `AI.SetBehaviorVariable` storage keyed by `(entity id, name)`.
    pub behavior_vars: HashMap<(u64, String), GoalParam>,
    /// `AI.ChangeParameter` storage keyed by `(entity id, AIPARAM_*)`.
    pub ai_params: HashMap<(u64, i64), GoalParam>,
}

fn upsert<'a, T: 'a>(
    list: &'a mut Vec<T>,
    index: &mut HashMap<String, usize>,
    name: &str,
    mk: impl FnOnce() -> T,
    get: impl FnOnce(&mut T) -> &mut String,
) -> &'a mut T {
    let key = name.to_ascii_lowercase();
    let i = match index.get(&key) {
        Some(i) => *i,
        None => {
            let i = list.len();
            list.push(mk());
            index.insert(key, i);
            i
        }
    };
    let item = &mut list[i];
    *get(item) = name.to_string();
    item
}

impl AiState {
    // -- goal pipes ---------------------------------------------------------

    /// `AI.CreateGoalPipe` / `AI.BeginGoalPipe`: start (or restart) a pipe and
    /// make it current. The original clears the pipe's goal list on begin.
    pub fn begin_pipe(&mut self, name: &str) {
        let key = name.to_ascii_lowercase();
        let i = match self.pipe_index.get(&key) {
            Some(i) => *i,
            None => {
                let i = self.goal_pipes.len();
                self.goal_pipes.push(GoalPipe { name: name.to_string(), goals: Vec::new() });
                self.pipe_index.insert(key, i);
                i
            }
        };
        self.goal_pipes[i].name = name.to_string();
        self.goal_pipes[i].goals.clear();
        self.current_pipe = Some(name.to_string());
    }

    pub fn end_pipe(&mut self) {
        self.current_pipe = None;
    }

    fn pipe_mut(&mut self, name: &str) -> Option<&mut GoalPipe> {
        let i = *self.pipe_index.get(&name.to_ascii_lowercase())?;
        Some(&mut self.goal_pipes[i])
    }

    /// Append a goal to `pipe` (or the current pipe when `pipe` is `None`).
    pub fn push_goal(&mut self, pipe: Option<&str>, goal: Goal) -> bool {
        let target = pipe.map(|s| s.to_string()).or_else(|| self.current_pipe.clone());
        let Some(name) = target else { return false };
        match self.pipe_mut(&name) {
            Some(p) => {
                p.goals.push(goal);
                true
            }
            None => false,
        }
    }

    /// Append a branch label to `pipe` (or the current pipe).
    pub fn push_label(&mut self, pipe: Option<&str>, label: &str) -> bool {
        let g = Goal {
            name: String::new(),
            blocking: false,
            params: vec![GoalParam::Str(label.to_string())],
            label: true,
        };
        self.push_goal(pipe, g)
    }

    pub fn is_pipe(&self, name: &str) -> bool {
        self.pipe_index.contains_key(&name.to_ascii_lowercase())
    }

    pub fn pipe_count(&self) -> usize {
        self.goal_pipes.len()
    }

    pub fn goal_count(&self) -> usize {
        self.goal_pipes.iter().map(|p| p.goals.len()).sum()
    }

    // -- formations ---------------------------------------------------------

    pub fn create_formation(&mut self, name: &str) {
        let f = upsert(
            &mut self.formations,
            &mut self.formation_index,
            name,
            || Formation::default(),
            |f| &mut f.name,
        );
        f.points.clear();
    }

    pub fn add_formation_point(&mut self, name: &str, pt: FormationPoint) {
        let f = upsert(
            &mut self.formations,
            &mut self.formation_index,
            name,
            || Formation::default(),
            |f| &mut f.name,
        );
        f.points.push(pt);
    }

    // -- behaviours / characters -------------------------------------------

    pub fn register_behavior(&mut self, name: &str, parent: Option<String>) {
        let b = upsert(
            &mut self.behaviors,
            &mut self.behavior_index,
            name,
            || BehaviorDef::default(),
            |b| &mut b.name,
        );
        b.parent = parent;
    }

    pub fn register_group_behavior(&mut self, name: &str, parent: Option<String>) {
        let b = upsert(
            &mut self.group_behaviors,
            &mut self.group_behavior_index,
            name,
            || BehaviorDef::default(),
            |b| &mut b.name,
        );
        b.parent = parent;
    }

    pub fn register_character(&mut self, name: &str) {
        if self.character_index.contains_key(&name.to_ascii_lowercase()) {
            return;
        }
        let i = self.characters.len();
        self.characters.push(name.to_string());
        self.character_index.insert(name.to_ascii_lowercase(), i);
    }

    // -- combat classes -----------------------------------------------------

    /// `AI.AddCombatClass()` (no args) clears the table; with an id it appends.
    pub fn add_combat_class(&mut self, id: Option<i64>, scale: Vec<f64>, seen_signal: Option<String>) {
        match id {
            None => {
                self.combat_classes.clear();
            }
            Some(id) => {
                if let Some(c) = self.combat_classes.iter_mut().find(|c| c.id == id) {
                    c.scale = scale;
                    c.seen_signal = seen_signal;
                } else {
                    self.combat_classes.push(CombatClass { id, scale, seen_signal });
                }
            }
        }
    }

    // -- tactical point queries --------------------------------------------

    pub fn register_tps_query(&mut self, q: TpsQuery) {
        let key = q.name.to_ascii_lowercase();
        match self.tps_index.get(&key) {
            Some(i) => self.tps_queries[*i] = q,
            None => {
                let i = self.tps_queries.len();
                self.tps_index.insert(key, i);
                self.tps_queries.push(q);
            }
        }
    }

    pub fn destroy_tps_queries(&mut self) {
        self.tps_queries.clear();
        self.tps_index.clear();
    }
}

/// Read `Generation`/`Conditions`/`Weights`/`Parameters` sub-tables of one
/// `AI.RegisterTacticalPointQuery` entry into [`TpsBlock`]s.
pub fn read_tps_blocks(entry: &Table) -> Vec<TpsBlock> {
    const KINDS: [&str; 4] = ["Generation", "Conditions", "Weights", "Parameters"];
    let mut out = Vec::new();
    for kind in KINDS {
        if let Ok(Value::Table(sub)) = entry.raw_get::<Value>(kind) {
            let mut block = TpsBlock { kind: kind.to_string(), entries: Vec::new() };
            for pair in sub.pairs::<Value, Value>() {
                let Ok((k, v)) = pair else { continue };
                if let Value::String(k) = k {
                    block.entries.push((k.to_string_lossy().to_string(), TpsVal::from_value(&v)));
                }
            }
            out.push(block);
        }
    }
    out
}
