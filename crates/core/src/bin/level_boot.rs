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

    let host = ScriptHost::new(vfs.clone()).expect("script host");
    host.boot().expect("boot");
    println!("[boot] scripts loaded: {}  errors: {}", host.ctx.loaded_scripts.borrow().len(), host.ctx.errors.borrow().len());

    let xml = ["mission_mission0.xml", "levels/class3/mission_mission0.xml"]
        .iter()
        .find_map(|p| vfs.read(p))
        .expect("mission_mission0.xml not found in VFS");
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
    let dt = 1.0 / 30.0;
    let frames = (seconds / dt) as usize;
    for _ in 0..frames {
        host.update(dt).expect("update");
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
