//! Boot the game's own scripts exactly as the engine would and spawn every
//! level entity through them, then report what the native engine layer is
//! still missing. Usage: `level_boot [game_dir] [seconds]`.

use sod_core::cry::mission::parse_level_entities;
use sod_core::cry::ScriptHost;
use sod_core::crypak::Vfs;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

const DEFAULT_GAME: &str =
    "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";

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

fn main() {
    let mut args = std::env::args().skip(1);
    let game = args.next().unwrap_or_else(|| DEFAULT_GAME.to_string());
    let seconds: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10.0);

    let mut paks = Vec::new();
    collect_paks(Path::new(&game), &mut paks);
    paks.sort();
    let mut vfs = Vfs::new();
    for p in &paks {
        let _ = vfs.mount(p);
    }
    let vfs = Rc::new(vfs);

    let mut host = ScriptHost::new(vfs.clone()).expect("script host");
    host.boot().expect("boot");
    let rules = host.spawn_game_rules().expect("game rules");
    println!("[rules] GameRules id {rules} instance={}", host.ctx.instance(rules).is_some());
    println!("[boot] scripts loaded: {}  errors: {}", host.ctx.loaded_scripts.borrow().len(), host.ctx.errors.borrow().len());

    let xml = ["mission_mission0.xml", "levels/class3/mission_mission0.xml"]
        .iter()
        .find_map(|p| vfs.read(p))
        .expect("mission_mission0.xml not found in VFS");
    let xml_bytes = xml.clone();
    let ents = parse_level_entities(&xml);
    println!("[level] {} entities", ents.len());

    let mut per_class: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for le in &ents {
        let before = host.ctx.errors.borrow().len();
        let _ = host.spawn_level_entity(le);
        let failed = host.ctx.errors.borrow().len() > before;
        let slot = per_class.entry(le.class.clone()).or_default();
        slot.0 += 1;
        if failed {
            slot.1 += 1;
        }
    }

    let errs_at_spawn = host.ctx.errors.borrow().len();

    // The player actor at the tutorial start point, walking east at run speed
    // so area volumes along the way fire enter/leave events.
    let start = [3246.8f32, 3256.2, 217.1];
    // Original flow: level loaded / start events, then the local client
    // connects and the rules script creates the player (Actor.CreateActor).
    let player = host.start_game().expect("start game").expect("rules script created no player");
    host.ctx.world.borrow_mut().set_pos(player, start);
    if let Some(e) = host.ctx.world.borrow().get(player) {
        println!("[player] created by rules script: name {:?} class {}", e.name, e.class);
    }
    println!("[player] id {player} instance={}", host.ctx.instance(player).is_some());
    if let Some(a) = host.ctx.actors.borrow().get(&player) {
        println!("[player] actor after init: {a:?}");
    }
    if let Ok(v) = host.lua.load(r#"
        local p = System.GetEntityByName("Dude")
        local d = p and p.Properties and p.Properties.Damage
        return tostring(d and d.health) .. " class.Damage=" .. tostring(Player.Properties.Damage and Player.Properties.Damage.health)
    "#).eval::<String>() {
        println!("[player] Properties.Damage.health = {v}");
    }

    let dt = 1.0 / 30.0;
    let frames = (seconds / dt) as usize;
    for f in 0..frames {
        let t = f as f32 * dt;
        host.ctx.world.borrow_mut().set_pos(player, [start[0] + 4.0 * t, start[1], start[2]]);
        host.update(dt).expect("update");
    }
    let mut areas_entered = 0;
    for a in host.areas.borrow().iter() {
        if a.inside.contains(&player) {
            areas_entered += 1;
        }
    }
    println!("[areas] {} volumes, player inside {areas_entered} at end", host.areas.borrow().len());
    println!("[events] delivered: {:?}", host.ctx.delivered.borrow());

    {
        // What scripts drew vs. what the level XML names as `object_Model`.
        use std::collections::BTreeMap;
        let w = host.ctx.world.borrow();
        let norm = |m: &str| m.replace('\\', "/").to_ascii_lowercase();
        let mut scripted: BTreeMap<String, usize> = BTreeMap::new();
        for p in sod_core::cry::render::static_placements(&w) {
            *scripted.entry(norm(p.model.as_deref().unwrap_or(""))).or_default() += 1;
        }
        let mut xml: BTreeMap<String, usize> = BTreeMap::new();
        for p in sod_core::level::parse_mission(&xml_bytes) {
            if p.is_entity {
                if let Some(m) = &p.model {
                    *xml.entry(norm(m)).or_default() += 1;
                }
            }
        }
        let s_total: usize = scripted.values().sum();
        let x_total: usize = xml.values().sum();
        println!("[render] static slots drawn by scripts: {s_total}  level-XML object_Model entities: {x_total}");
        let mut diffs = 0;
        for k in scripted.keys().chain(xml.keys()).collect::<std::collections::BTreeSet<_>>() {
            let (a, b) = (scripted.get(k).copied().unwrap_or(0), xml.get(k).copied().unwrap_or(0));
            if a != b {
                diffs += 1;
                if diffs <= 15 {
                    println!("  diff {k}: scripts {a} vs xml {b}");
                }
            }
        }
        println!("[render] models that differ: {diffs}");
    }

    let w = host.ctx.world.borrow();
    let active = w.entities.values().filter(|e| e.active).count();
    let with_slots = w.entities.values().filter(|e| e.slots.iter().any(|s| s.is_some())).count();
    let in_state = w.entities.values().filter(|e| !e.state.is_empty()).count();
    drop(w);
    println!(
        "[sim] {frames} frames: instances={} active={active} with_slots={with_slots} in_state={in_state}",
        host.ctx.instances.borrow().len()
    );

    println!("\n== per-class spawn (count / spawns that raised errors) ==");
    for (c, (n, f)) in &per_class {
        println!("{n:6} {f:6}  {c}");
    }

    let errors = host.ctx.errors.borrow();
    println!("\n== script errors: {} at spawn, {} total ==", errs_at_spawn, errors.len());
    let mut uniq: BTreeMap<String, usize> = BTreeMap::new();
    for e in errors.iter() {
        // Collapse per-entity prefixes so identical failures group.
        let key = e.split_once(": ").map(|(_, r)| r.to_string()).unwrap_or_else(|| e.clone());
        let key: String = key.lines().next().unwrap_or("").chars().take(200).collect();
        *uniq.entry(key).or_default() += 1;
    }
    let mut uniq: Vec<_> = uniq.into_iter().collect();
    uniq.sort_by(|a, b| b.1.cmp(&a.1));
    for (e, n) in uniq.iter().take(40) {
        println!("{n:6}  {e}");
    }

    let missing = host.missing_report();
    println!("\n== missing engine binds: {} distinct, {} calls ==", missing.len(), missing.iter().map(|m| m.1).sum::<u64>());
    for (m, n) in missing.iter() {
        println!("{n:8}  {m}");
    }
}
