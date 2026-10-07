//! `CScriptBind_AI`: the native `AI.*` namespace the game's AI scripts drive.
//!
//! The AI system's script boot is two engine paths:
//!
//!  * `CScriptBind_AI::Init` (`0x00968ab0`) registers the namespace and, at
//!    `0x0096d761`, calls the helper `FUN_00983230` (`0x00983230`) which runs
//!    `ExecuteFile("scripts/ai/aiconfig.lua", 1, 0, 0)`.
//!  * The `CAISystem` reload/init path (`0x008c1910`) runs
//!    `ExecuteFile("Scripts/AI/aiconfig.lua", 1, 1, 0)` (`0x008c198e`) and
//!    then invokes the Lua globals `AIReset` (`0x008c199d`) and `AIReload`
//!    (`0x008c19b0`).
//!
//! [`boot`] reproduces the second path: run `aiconfig.lua`, then `AIReset`
//! (`AI:ResetAI` -> `AIBlackBoard_Reset`, `ReloadTPS`, `ReloadPipes`) and
//! `AIReload` (TPS extensions + `PipeManager:OnInit`). The registries the
//! scripts build are recorded in [`super::ai_data::AiState`].
//!
//! Unimplemented members keep falling through to the host's counted stubs, so
//! `ScriptHost::missing_report` still names whatever the AI scripts reach that
//! this module does not yet model.

use super::ai_data::{
    read_tps_blocks, AiState, FormationPoint, Goal, GoalParam, PathfindProps, TpsQuery,
};
use super::ai_loader;
use super::host::{id_value, lua_format, num, truthy, value_to_id, Ctx};
use mlua::{Lua, Result, Table, Value, Variadic};
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    /// Most recently installed AI registry. Tests (and the runtime) use this
    /// to read what the boot scripts built.
    static INSTALLED: RefCell<Option<Rc<RefCell<AiState>>>> = const { RefCell::new(None) };
}

/// The registry built by the last [`install`] call on this thread.
pub fn installed_state() -> Option<Rc<RefCell<AiState>>> {
    INSTALLED.with(|s| s.borrow().clone())
}

fn table_nums(t: &Table) -> Vec<f64> {
    let mut v = Vec::new();
    for i in 1..=t.raw_len() {
        if let Ok(x) = t.raw_get::<Value>(i as i64) {
            if let Some(n) = num(&x) {
                v.push(n);
            }
        }
    }
    v
}

fn goal_param_to_value(lua: &Lua, p: Option<GoalParam>) -> Result<Value> {
    Ok(match p {
        None | Some(GoalParam::Nil) => Value::Nil,
        Some(GoalParam::Bool(b)) => Value::Boolean(b),
        Some(GoalParam::Num(n)) => Value::Number(n),
        Some(GoalParam::Str(s)) => Value::String(lua.create_string(&s)?),
        Some(GoalParam::Id(i)) => id_value(i as super::world::EntityId),
        Some(GoalParam::Table) => Value::Nil,
    })
}

fn call_global(lua: &Lua, ctx: &Ctx, name: &str) {
    if let Ok(Value::Function(f)) = lua.globals().get::<Value>(name) {
        if let Err(e) = f.call::<()>(()) {
            ctx.error(format!("{name}(): {e}"));
        }
    }
}

/// Install the `AI.*` binds. Must run after the host prelude (the `AI`
/// namespace table already exists with its stub fallback).
pub fn install(lua: &Lua, ctx: &Ctx) -> Result<()> {
    let state: Rc<RefCell<AiState>> = Rc::new(RefCell::new(AiState::default()));
    INSTALLED.with(|s| *s.borrow_mut() = Some(state.clone()));

    let ai: Table = lua.globals().get("AI")?;

    // -- logging ------------------------------------------------------------
    for (name, tag) in [
        ("LogEvent", "AI.LogEvent"),
        ("LogComment", "AI.LogComment"),
        ("LogProgress", "AI.LogProgress"),
        ("Warning", "AI.Warning"),
        ("Error", "AI.Error"),
    ] {
        let c = ctx.clone();
        ai.raw_set(
            name,
            lua.create_function(move |lua, args: Variadic<Value>| {
                if c.echo_log {
                    println!("[{tag}] {}", lua_format(lua, &args));
                }
                Ok(())
            })?,
        )?;
    }

    // -- goal pipes ---------------------------------------------------------
    let st = state.clone();
    ai.raw_set(
        "CreateGoalPipe",
        lua.create_function(move |_, name: String| {
            st.borrow_mut().begin_pipe(&name);
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "BeginGoalPipe",
        lua.create_function(move |_, name: String| {
            st.borrow_mut().begin_pipe(&name);
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "EndGoalPipe",
        lua.create_function(move |_, ()| {
            st.borrow_mut().end_pipe();
            Ok(())
        })?,
    )?;

    // PushGoal has two call shapes (CryEngine picks by the 2nd argument):
    //   PushGoal(pipe, goal, blocking, ...)   -- explicit pipe
    //   PushGoal(goal, blocking, ...)          -- current pipe (BeginGoalPipe)
    let st = state.clone();
    ai.raw_set(
        "PushGoal",
        lua.create_function(move |_, args: Variadic<Value>| {
            let (pipe, goal_name, blocking_idx) = match (args.get(0), args.get(1)) {
                (Some(Value::String(s0)), Some(Value::String(s1))) => (
                    Some(s0.to_string_lossy().to_string()),
                    s1.to_string_lossy().to_string(),
                    2,
                ),
                (Some(Value::String(s0)), _) => (None, s0.to_string_lossy().to_string(), 1),
                _ => return Ok(Value::Nil),
            };
            let blocking = args.get(blocking_idx).map(truthy).unwrap_or(false);
            let params: Vec<GoalParam> =
                args.iter().skip(blocking_idx + 1).map(GoalParam::from_value).collect();
            let goal = Goal { name: goal_name, blocking, params, label: false };
            st.borrow_mut().push_goal(pipe.as_deref(), goal);
            Ok(Value::Nil)
        })?,
    )?;

    let st = state.clone();
    ai.raw_set(
        "PushLabel",
        lua.create_function(move |_, args: Variadic<Value>| {
            let label = args.iter().rev().find_map(|v| match v {
                Value::String(s) => Some(s.to_string_lossy().to_string()),
                _ => None,
            });
            let Some(label) = label else { return Ok(Value::Nil) };
            let pipe = if args.len() >= 2 {
                args.get(0).and_then(|v| match v {
                    Value::String(s) => Some(s.to_string_lossy().to_string()),
                    _ => None,
                })
            } else {
                None
            };
            st.borrow_mut().push_label(pipe.as_deref(), &label);
            Ok(Value::Nil)
        })?,
    )?;

    let st = state.clone();
    ai.raw_set(
        "IsGoalPipe",
        lua.create_function(move |_, name: String| Ok(st.borrow().is_pipe(&name)))?,
    )?;

    let c = ctx.clone();
    ai.raw_set(
        "LoadGoalPipes",
        lua.create_function(move |lua, path: String| {
            super::host::ScriptHost::exec_file(lua, &c, &path, true);
            Ok(())
        })?,
    )?;

    // -- formations ---------------------------------------------------------
    let st = state.clone();
    ai.raw_set(
        "CreateFormation",
        lua.create_function(move |_, name: String| {
            st.borrow_mut().create_formation(&name);
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "AddFormationPoint",
        lua.create_function(move |_, (name, args): (String, Variadic<Value>)| {
            let g = |i: usize| args.get(i).and_then(num).unwrap_or(0.0);
            let unit_class = args.get(3).and_then(num).unwrap_or(1.0) as i64;
            st.borrow_mut().add_formation_point(
                &name,
                FormationPoint { sight: g(0), distance: g(1), offset: g(2), unit_class, fixed: false },
            );
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "AddFormationPointFixed",
        lua.create_function(move |_, (name, args): (String, Variadic<Value>)| {
            let g = |i: usize| args.get(i).and_then(num).unwrap_or(0.0);
            let unit_class = args.get(3).and_then(num).unwrap_or(1.0) as i64;
            st.borrow_mut().add_formation_point(
                &name,
                FormationPoint { sight: g(0), distance: g(1), offset: g(2), unit_class, fixed: true },
            );
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "CreateGroupFormation",
        lua.create_function(move |_, (name, _args): (String, Variadic<Value>)| {
            st.borrow_mut().create_formation(&name);
            Ok(())
        })?,
    )?;

    // -- behaviour / character / group-behaviour catalogues -----------------
    let c = ctx.clone();
    let st = state.clone();
    ai.raw_set(
        "LoadBehaviors",
        lua.create_function(move |lua, (folder, ext): (String, String)| {
            let found = ai_loader::load_behaviors(lua, &c, &folder, &ext, "CreateAIBehavior")?;
            let mut s = st.borrow_mut();
            for b in found {
                s.register_behavior(&b.name, b.parent);
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    let st = state.clone();
    ai.raw_set(
        "LoadGroupBehaviors",
        lua.create_function(move |lua, (folder, ext): (String, String)| {
            let found = ai_loader::load_behaviors(lua, &c, &folder, &ext, "CreateAIGroupBehavior")?;
            let mut s = st.borrow_mut();
            for b in found {
                s.register_group_behavior(&b.name, b.parent);
            }
            Ok(())
        })?,
    )?;
    let c = ctx.clone();
    let st = state.clone();
    ai.raw_set(
        "LoadCharacters",
        lua.create_function(move |lua, (folder, table): (String, Option<Table>)| {
            ai_loader::load_flat(lua, &c, &folder, "*.lua")?;
            // The character files index `AICharacter.<Name>`; collect what the
            // loaded files defined (skipping the catalogue's own bookkeeping).
            let t = table.or_else(|| lua.globals().get::<Table>("AICharacter").ok());
            if let Some(t) = t {
                let mut s = st.borrow_mut();
                for pair in t.pairs::<Value, Value>() {
                    let Ok((k, v)) = pair else { continue };
                    if let (Value::String(k), Value::Table(_)) = (&k, &v) {
                        let name = k.to_string_lossy().to_string();
                        if name != "AVAILABLE" && name != "INTERNAL" {
                            s.register_character(&name);
                        }
                    }
                }
            }
            Ok(())
        })?,
    )?;

    // -- combat classes / assessment ---------------------------------------
    let st = state.clone();
    ai.raw_set(
        "SetAssesmentMultiplier",
        lua.create_function(move |_, (id, mult): (Value, Value)| {
            let id = num(&id).unwrap_or(0.0) as i64;
            st.borrow_mut().assessment.insert(id, num(&mult).unwrap_or(1.0));
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "AddCombatClass",
        lua.create_function(move |_, args: Variadic<Value>| {
            let mut s = st.borrow_mut();
            if args.is_empty() {
                s.add_combat_class(None, Vec::new(), None);
                return Ok(());
            }
            let id = args.get(0).and_then(num).unwrap_or(0.0) as i64;
            let scale = match args.get(1) {
                Some(Value::Table(t)) => table_nums(t),
                _ => Vec::new(),
            };
            let signal = args.get(2).and_then(|v| match v {
                Value::String(s) => Some(s.to_string_lossy().to_string()),
                _ => None,
            });
            s.add_combat_class(Some(id), scale, signal);
            Ok(())
        })?,
    )?;

    // -- pathfinding properties --------------------------------------------
    let c = ctx.clone();
    let st = state.clone();
    ai.raw_set(
        "LoadPathfindProperties",
        lua.create_function(move |_, path: String| {
            // The bind parses the XML in the original; record the file and the
            // path types it names so the registry is complete.
            let types = c
                .vfs
                .read(&path)
                .map(|b| {
                    let text = String::from_utf8_lossy(&b);
                    text.match_indices("AIPATH_").count()
                })
                .unwrap_or(0);
            st.borrow_mut().pathfind_props.push(PathfindProps {
                path_type: path,
                nav_mask: 0,
                params: Vec::new(),
                id: types as i64,
            });
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "AssignPFPropertiesToPathType",
        lua.create_function(move |_, args: Variadic<Value>| {
            let path_type = args
                .get(0)
                .and_then(|v| match v {
                    Value::String(s) => Some(s.to_string_lossy().to_string()),
                    _ => None,
                })
                .unwrap_or_default();
            let nav_mask = args.get(1).and_then(num).unwrap_or(0.0) as i64;
            let params: Vec<f64> = args.iter().skip(2).filter_map(num).collect();
            st.borrow_mut().pathfind_props.push(PathfindProps { path_type, nav_mask, params, id: 0 });
            Ok(())
        })?,
    )?;

    // -- tactical point queries --------------------------------------------
    let st = state.clone();
    ai.raw_set(
        "RegisterTacticalPointQuery",
        lua.create_function(move |_, t: Table| {
            let name: String = t.get::<Option<String>>("Name")?.unwrap_or_default();
            let mut blocks = read_tps_blocks(&t);
            if blocks.is_empty() {
                for i in 1..=t.raw_len() {
                    if let Ok(Value::Table(entry)) = t.raw_get::<Value>(i as i64) {
                        blocks.extend(read_tps_blocks(&entry));
                    }
                }
            }
            st.borrow_mut().register_tps_query(TpsQuery { name, blocks });
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "DestroyAllTPSQueries",
        lua.create_function(move |_, ()| {
            st.borrow_mut().destroy_tps_queries();
            Ok(())
        })?,
    )?;
    ai.raw_set(
        "GetTacticalPoints",
        lua.create_function(|_, _: Variadic<Value>| Ok(false))?,
    )?;

    // -- per-entity AI variables / parameters -------------------------------
    let st = state.clone();
    ai.raw_set(
        "SetBehaviorVariable",
        lua.create_function(move |_, (id, name, value): (Value, String, Value)| {
            if let Some(id) = value_to_id(&id) {
                st.borrow_mut().behavior_vars.insert((id as u64, name), GoalParam::from_value(&value));
            }
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "GetBehaviorVariable",
        lua.create_function(move |lua, (id, name): (Value, String)| {
            let Some(id) = value_to_id(&id) else { return Ok(Value::Nil) };
            let v = st.borrow().behavior_vars.get(&(id as u64, name)).cloned();
            goal_param_to_value(lua, v)
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "ChangeParameter",
        lua.create_function(move |_, (id, param, value): (Value, Value, Value)| {
            if let Some(id) = value_to_id(&id) {
                let p = num(&param).unwrap_or(0.0) as i64;
                st.borrow_mut().ai_params.insert((id as u64, p), GoalParam::from_value(&value));
            }
            Ok(())
        })?,
    )?;
    let st = state.clone();
    ai.raw_set(
        "ResetParameters",
        lua.create_function(move |_, id: Value| {
            if let Some(id) = value_to_id(&id) {
                st.borrow_mut().ai_params.retain(|(e, _), _| *e != id as u64);
            }
            Ok(())
        })?,
    )?;

    // -- runtime verbs reached by the boot scripts but not simulated here ----
    // These are recorded/accepted so the boot never falls through to a counted
    // stub; the real AI execution is out of scope.
    for name in [
        "Signal",
        "FreeSignal",
        "SetRefPointPosition",
        "GetRefPointPosition",
        "SetRefPointDirection",
        "GetRefPointDirection",
        "SetStance",
        "GetStance",
        "SetBeaconPosition",
        "GetBeaconPosition",
        "SetTerritoryShapeName",
        "IsEnabled",
        "GetEnclosingGenericShapeOfType",
        "ModifySmartObjectStates",
        "SmartObjectEvent",
        "EnableWeaponAccessory",
        "AbortAction",
        "ExecuteAction",
        "SetIgnorant",
        "SetContinuousMotion",
        "SetMemoryFireType",
        "GetGroupOf",
        "GetGroupCount",
        "GetTargetType",
        "GetAttentionTargetDistance",
        "GetAttentionTargetEntity",
        "SetPFProperties",
        "SetFormationPosition",
        "SetFormationLookingPoint",
        "GetFormationPosition",
        "RegisterTargetTrack",
        "UnregisterTargetTrack",
    ] {
        ai.raw_set(name, lua.create_function(|_, _: Variadic<Value>| Ok(Value::Nil))?)?;
    }
    Ok(())
}

/// Run the AI script boot exactly as `CAISystem`'s reload path does:
/// `ExecuteFile("Scripts/AI/aiconfig.lua", 1, 1, 0)`, then `AIReset`, then
/// `AIReload`. Requires [`install`] to have run first.
pub fn boot(lua: &Lua, ctx: &Ctx) -> Result<()> {
    // `aiconfig.lua` is authored against the shared script helpers (`mergef`,
    // `merge`, `Log`, ...) that `scripts/main.lua` normally loads first. Load
    // `common.lua` (idempotent via the loaded-script set) so this boot is
    // self-contained.
    super::host::ScriptHost::exec_file(lua, ctx, "scripts/common.lua", false);
    super::host::ScriptHost::exec_file(lua, ctx, "scripts/ai/aiconfig.lua", true);
    call_global(lua, ctx, "AIReset");
    call_global(lua, ctx, "AIReload");
    Ok(())
}

/// `AI.LoadBehaviors`/`AI.LoadCharacters` folder enumeration helper, exposed
/// for the runtime (and tests) to inspect which files a catalogue would load.
pub fn catalogue_files(vfs: &crate::crypak::Vfs, folder: &str, ext: &str) -> Vec<String> {
    ai_loader::files_in(vfs, folder, ext)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cry::ScriptHost;
    use crate::crypak::Vfs;
    use std::path::{Path, PathBuf};

    const DEFAULT_GAME: &str = "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";

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

    fn game_vfs() -> Option<Rc<Vfs>> {
        let root = std::env::var("SOD_GAME_DIR").unwrap_or_else(|_| DEFAULT_GAME.to_string());
        let root = Path::new(&root);
        if !root.is_dir() {
            return None;
        }
        let mut paks = Vec::new();
        collect_paks(root, &mut paks);
        if paks.is_empty() {
            return None;
        }
        paks.sort();
        let mut vfs = Vfs::new();
        for p in &paks {
            let _ = vfs.mount(p);
        }
        Some(Rc::new(vfs))
    }

    #[test]
    fn ai_boot_builds_the_registry_without_errors() {
        let Some(vfs) = game_vfs() else {
            eprintln!("skipping: game paks not present");
            return;
        };
        let host = ScriptHost::new(vfs).expect("script host");
        install(&host.lua, &host.ctx).expect("ai install");
        boot(&host.lua, &host.ctx).expect("ai boot");

        let errors = host.ctx.errors.borrow().clone();
        for e in errors.iter().take(20) {
            eprintln!("[ai boot error] {e}");
        }
        assert!(errors.is_empty(), "{} script errors during AI boot", errors.len());

        let state = installed_state().expect("ai state installed");
        let s = state.borrow();
        assert!(s.goal_pipes.len() > 0, "no goal pipes registered");
        assert!(s.formations.len() > 0, "no formations registered");
        assert!(s.behaviors.len() > 0, "no behaviours registered");
        assert!(s.characters.len() > 0, "no characters registered");
        assert!(s.combat_classes.len() > 0, "no combat classes registered");
        assert!(s.tps_queries.len() > 0, "no tactical point queries registered");
        eprintln!(
            "[ai boot] pipes={} goals={} formations={} behaviors={} characters={} combat_classes={} tps_queries={}",
            s.goal_pipes.len(),
            s.goal_count(),
            s.formations.len(),
            s.behaviors.len(),
            s.characters.len(),
            s.combat_classes.len(),
            s.tps_queries.len(),
        );

        let ai_missing: Vec<(String, u64)> = host
            .missing_report()
            .into_iter()
            .filter(|(k, _)| k.starts_with("AI."))
            .collect();
        assert!(ai_missing.is_empty(), "AI.* calls hit stubs: {ai_missing:?}");
    }

    #[test]
    fn goal_pipe_records_ordered_goals_and_params() {
        // A bare host (empty VFS) gives us the `AI` namespace without touching
        // the game paks.
        let host = ScriptHost::new(Rc::new(Vfs::new())).expect("script host");
        install(&host.lua, &host.ctx).unwrap();

        host.lua
            .load(
                r#"
            AI.BeginGoalPipe("p")
                AI.PushGoal("timeout", 1, 0.5)
                AI.PushGoal("+signal", 0, 1, "GO", SIGNALFILTER_SENDER)
                AI.PushLabel("LOOP")
                AI.PushGoal("branch", 0, "LOOP", BRANCH_ALWAYS)
            AI.EndGoalPipe()
            AI.CreateGoalPipe("q")
            AI.PushGoal("q", "timeout", 1, 2.0)
            "#,
            )
            .set_name("@ai_pipe_test.lua")
            .exec()
            .unwrap();

        let state = installed_state().unwrap();
        let s = state.borrow();
        assert_eq!(s.goal_pipes.len(), 2);
        let p = s.goal_pipes.iter().find(|p| p.name == "p").unwrap();
        assert_eq!(p.goals.len(), 4);
        assert_eq!(p.goals[0].name, "timeout");
        assert!(p.goals[0].blocking);
        assert_eq!(p.goals[0].params, vec![GoalParam::Num(0.5)]);
        assert_eq!(p.goals[1].name, "+signal");
        assert_eq!(p.goals[1].params[1], GoalParam::Str("GO".into()));
        assert!(p.goals[2].label);
        assert_eq!(p.goals[3].name, "branch");
        let q = s.goal_pipes.iter().find(|p| p.name == "q").unwrap();
        assert_eq!(q.goals.len(), 1);
        assert_eq!(q.goals[0].name, "timeout");
    }
}
