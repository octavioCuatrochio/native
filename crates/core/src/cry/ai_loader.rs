//! Script-folder loading used by the `AI.LoadBehaviors` / `AI.LoadCharacters`
//! / `AI.LoadGroupBehaviors` binds.
//!
//! The original `CScriptBind_AI` loaders (`0x0097e510` behaviors,
//! `0x0097e660` characters, `0x0097e7a0` group behaviors) recurse the mounted
//! paks for `*.lua`. Behaviours are *not* loaded in file order: the engine
//! first runs every file through a fake `CreateAIBehavior` that records
//! `name -> baseName` into `__DiscoveredBehavior` (the chunk string at
//! `0x0138e5c8`), topologically sorts by base behaviour, and only then runs
//! the real constructor. This module reproduces that two-pass scheme so
//! derived behaviours (e.g. `UL_Ally_BaseBehavior` subclasses) always find
//! their base, regardless of pak entry order.

use super::ai_data::BehaviorDef;
use super::host::{norm_path, Ctx};
use crate::crypak::Vfs;
use mlua::{Lua, Result, Table, Value};
use std::collections::{HashMap, HashSet};

/// All VFS entries under `folder` ending in `ext`, sorted and de-duplicated.
pub fn files_in(vfs: &Vfs, folder: &str, ext: &str) -> Vec<String> {
    let prefix = format!("{}/", folder.replace('\\', "/").trim_end_matches('/').to_ascii_lowercase());
    // The bind passes globs like `*.lua`; `paths_with_ext` wants the suffix.
    let ext = match ext.find('*') {
        Some(i) => &ext[i + 1..],
        None => ext,
    };
    let mut out: Vec<String> = vfs
        .paths_with_ext(ext)
        .into_iter()
        .filter(|p| p.replace('\\', "/").to_ascii_lowercase().starts_with(&prefix))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Execute a file without recording errors or touching the loaded-script set.
/// Used for the behaviour discovery pass, whose errors are expected to be
/// harmless (a file may reference a behaviour defined later in the pass).
fn run_quiet(lua: &Lua, ctx: &Ctx, path: &str) -> bool {
    let key = norm_path(path);
    let Some(bytes) = ctx.vfs.read(&key) else { return false };
    let code = String::from_utf8_lossy(&bytes);
    let code = code.strip_prefix('\u{feff}').unwrap_or(&code);
    lua.load(code).set_name(format!("@{key}")).exec().is_ok()
}

const DISCOVERY_LUA: &str = r#"
-- Reproduces the recovered probe chunk (string 0x0138e5c8): the fake
-- constructor records the behaviour name and its base name, then returns an
-- empty definition so the discovery pass has no side effects on AIBehavior.
__ai_discovery = {}
function __ai_fake_ctor(...)
    local name, baseName = select(1, ...)
    if (type(baseName) ~= "string") then baseName = true end
    __ai_discovery[name] = baseName
    return {}
end
"#;

/// Load a behaviour folder with the discovery + topological-order scheme.
/// `ctor` is the Lua global the files call (`CreateAIBehavior` /
/// `CreateAIGroupBehavior`). Returns the discovered `(name, base)` catalogue.
pub fn load_behaviors(lua: &Lua, ctx: &Ctx, folder: &str, ext: &str, ctor: &str) -> Result<Vec<BehaviorDef>> {
    let files = files_in(&ctx.vfs, folder, ext);
    if files.is_empty() {
        return Ok(Vec::new());
    }

    // Pass 1: discover every (name, base) pair and the file that defines it.
    let saved: Value = lua.globals().get(ctor)?;
    lua.load(DISCOVERY_LUA).set_name("@sod_ai_discovery.lua").exec()?;
    let fake: Value = lua.globals().get("__ai_fake_ctor")?;
    lua.globals().set(ctor, fake)?;

    let mut name_file: HashMap<String, String> = HashMap::new();
    let mut discovered: Vec<(String, Option<String>)> = Vec::new();
    for f in &files {
        let _ = run_quiet(lua, ctx, f);
        let disc: Table = lua.globals().get("__ai_discovery")?;
        for pair in disc.pairs::<Value, Value>() {
            let (k, v) = pair?;
            let Value::String(k) = k else { continue };
            let name = k.to_string_lossy().to_string();
            if name_file.insert(name.clone(), f.clone()).is_none() {
                let parent = match v {
                    Value::String(s) => Some(s.to_string_lossy().to_string()),
                    _ => None,
                };
                discovered.push((name, parent));
            }
        }
    }
    lua.globals().set(ctor, saved)?;

    // Pass 2: topological order (base before derived), stable for cycles.
    let parents: HashMap<String, Option<String>> = discovered.iter().cloned().collect();
    let known: HashSet<String> = discovered.iter().map(|(n, _)| n.to_ascii_lowercase()).collect();
    let mut placed: HashSet<String> = HashSet::new();
    let mut order: Vec<String> = Vec::new();
    let mut remaining = discovered;
    while !remaining.is_empty() {
        let mut next = Vec::new();
        let mut progressed = false;
        for (name, parent) in remaining.drain(..) {
            let ready = match &parent {
                None => true,
                Some(b) => placed.contains(&b.to_ascii_lowercase()) || !known.contains(&b.to_ascii_lowercase()),
            };
            if ready {
                placed.insert(name.to_ascii_lowercase());
                order.push(name);
                progressed = true;
            } else {
                next.push((name, parent));
            }
        }
        remaining = next;
        if !remaining.is_empty() && !progressed {
            for (n, _) in remaining.drain(..) {
                order.push(n);
            }
            break;
        }
    }

    // Pass 3: run each file once, in dependency order, recording any error.
    let mut done: HashSet<String> = HashSet::new();
    for name in &order {
        if let Some(f) = name_file.get(name) {
            if done.insert(norm_path(f)) {
                super::host::ScriptHost::exec_file(lua, ctx, f, true);
            }
        }
    }
    Ok(order
        .into_iter()
        .map(|name| {
            let parent = parents.get(&name).cloned().flatten();
            BehaviorDef { name, parent }
        })
        .collect())
}

/// Execute every `*.lua` under `folder` that is not already loaded. Skipping
/// loaded files keeps a loader script (e.g. `characters/aicharacter.lua`,
/// which itself calls `AI.LoadCharacters`) from recursing into itself.
pub fn load_flat(lua: &Lua, ctx: &Ctx, folder: &str, ext: &str) -> Result<()> {
    for f in files_in(&ctx.vfs, folder, ext) {
        if ctx.loaded_scripts.borrow().contains(&norm_path(&f)) {
            continue;
        }
        super::host::ScriptHost::exec_file(lua, ctx, &f, false);
    }
    Ok(())
}
