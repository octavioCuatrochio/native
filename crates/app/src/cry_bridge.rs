//! Bevy side of the CryEngine replacement: owns the [`ScriptHost`] (the
//! game's own Lua running on the native entity world), ticks it every frame
//! and feeds it the player's position. Positions stay in CryEngine space
//! (Z-up); conversion to Bevy happens only where things are drawn.
//!
//! Disable with `SOD_CRY=0`.

use bevy::prelude::*;
use sod_core::cry::mission::parse_level_entities;
use sod_core::cry::{EntityId, ScriptHost};
use sod_core::crypak::Vfs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// The script host (Lua is `!Send`, so this is a non-send resource).
pub struct CryHost {
    pub host: ScriptHost,
    pub player: EntityId,
}

/// The controlled player's transform in CryEngine space (position, and
/// rotation as `(x, y, z, w)`), written after movement each frame.
#[derive(Resource, Default)]
pub struct CryPlayerPos(pub Option<[f32; 3]>, pub Option<[f32; 4]>);

/// Per-frame order: game code moves the player, the result is synced into
/// the entity world, then scripts tick on up-to-date state.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum CrySet {
    Sync,
    Tick,
}

/// Live counters for HUD/diagnostics.
#[derive(Resource, Default, Debug)]
pub struct CryStats {
    pub entities: usize,
    pub areas_inside: usize,
    pub script_errors: usize,
    pub missing_binds: usize,
}

pub struct CryPlugin {
    pub game_dir: String,
    pub spawn: [f32; 3],
}

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

/// Boot the original scripts and spawn the level's entities through them.
pub fn build_host(game_dir: &str, spawn: [f32; 3]) -> Option<CryHost> {
    let mut paks = Vec::new();
    collect_paks(Path::new(game_dir), &mut paks);
    paks.sort();
    let mut vfs = Vfs::new();
    for p in &paks {
        let _ = vfs.mount(p);
    }
    let vfs = Rc::new(vfs);
    let mut host = match ScriptHost::new(vfs.clone()) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[cry] script host failed: {e}");
            return None;
        }
    };
    if let Err(e) = host.boot() {
        eprintln!("[cry] boot failed: {e}");
    }
    if let Err(e) = host.spawn_game_rules() {
        eprintln!("[cry] game rules failed: {e}");
    }
    let xml = ["mission_mission0.xml", "levels/class3/mission_mission0.xml"].iter().find_map(|p| vfs.read(p))?;
    let ents = parse_level_entities(&xml);
    for le in &ents {
        let _ = host.spawn_level_entity(le);
    }
    let player = host.spawn_actor("Player", "scripts/entities/actor/player.lua", "Dev", spawn, true).ok()?;
    host.add_area_listener(player);
    println!(
        "[cry] booted: {} entities, {} areas, {} script errors, {} missing binds",
        host.ctx.instances.borrow().len(),
        host.areas.borrow().len(),
        host.ctx.errors.borrow().len(),
        host.missing_report().len()
    );
    Some(CryHost { host, player })
}

impl Plugin for CryPlugin {
    fn build(&self, app: &mut App) {
        if std::env::var("SOD_CRY").map(|v| v == "0").unwrap_or(false) {
            return;
        }
        app.init_resource::<CryPlayerPos>().init_resource::<CryStats>();
        if let Some(h) = build_host(&self.game_dir, self.spawn) {
            app.insert_non_send_resource(h);
            app.configure_sets(Update, (CrySet::Sync, CrySet::Tick).chain());
            app.add_systems(Update, cry_tick.in_set(CrySet::Tick));
        }
    }
}

fn cry_tick(
    cry: Option<NonSendMut<CryHost>>,
    time: Res<Time>,
    pos: Res<CryPlayerPos>,
    mut stats: ResMut<CryStats>,
) {
    let Some(cry) = cry else { return };
    if let Some(p) = pos.0 {
        cry.host.ctx.world.borrow_mut().set_pos(cry.player, p);
    }
    if let Some(r) = pos.1 {
        cry.host.ctx.world.borrow_mut().set_rot(cry.player, r);
    }
    // CryEngine clamps the frame time it hands to game code.
    let dt = time.delta_secs().min(0.1);
    if let Err(e) = cry.host.update(dt) {
        eprintln!("[cry] update: {e}");
    }
    // The presentation layer consumes world events once it draws slots;
    // until then drop them so the queue stays bounded.
    cry.host.ctx.world.borrow_mut().drain_events();
    stats.entities = cry.host.ctx.instances.borrow().len();
    stats.areas_inside = cry.host.areas.borrow().iter().filter(|a| a.inside.contains(&cry.player)).count();
    stats.script_errors = cry.host.ctx.errors.borrow().len();
    stats.missing_binds = cry.host.ctx.missing.borrow().len();
}
