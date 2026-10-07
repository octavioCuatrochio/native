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

/// The listener the audio thread mixes for (CryEngine space), written by the
/// camera each frame.
#[derive(Resource, Clone)]
pub struct CryListener(pub std::sync::Arc<std::sync::Mutex<sod_core::audio_play::Listener>>);

/// Keeps the audio output stream alive (cpal streams are not `Send`).
pub struct AudioOut(#[allow(dead_code)] cpal::Stream);

/// Open the default output device and attach a sound system at its rate.
/// `SOD_MUTE=1` disables audio.
fn open_audio(host: &ScriptHost) -> Option<(AudioOut, CryListener)> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::{Arc, Mutex};
    if std::env::var("SOD_MUTE").is_ok() {
        return None;
    }
    let device = cpal::default_host().default_output_device()?;
    let config = device.default_output_config().ok()?;
    let rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let system = Arc::new(Mutex::new(sod_core::audio_play::SoundSystem::new(rate)));
    host.attach_audio(system.clone());
    let listener = Arc::new(Mutex::new(sod_core::audio_play::Listener {
        position: [0.0; 3],
        forward: [0.0, 1.0, 0.0],
        up: [0.0, 0.0, 1.0],
    }));
    let (sys, lis) = (system, listener.clone());
    // SOD_AUDIODBG=1: once per second, report how many voices mixed and the
    // loudest sample seen, so audio activity is provable without listening.
    let dbg = std::env::var("SOD_AUDIODBG").is_ok();
    let mut dbg_frames: u64 = 0;
    let mut dbg_blocks: u64 = 0;
    let mut dbg_voices: usize = 0;
    let mut dbg_peak: f32 = 0.0;
    let stream = device
        .build_output_stream(
            &config.config(),
            move |out: &mut [f32], _| {
                let frames = out.len() / channels.max(1);
                let l = lis.lock().map(|l| l.clone()).ok();
                let (mix, voices) = match (sys.lock(), l) {
                    (Ok(mut s), Some(l)) => {
                        let mix = s.render(frames, &l);
                        let n = s.voice_count();
                        (mix, n)
                    }
                    _ => (vec![0.0; frames * 2], 0),
                };
                if dbg {
                    dbg_frames += frames as u64;
                    dbg_blocks += 1;
                    dbg_voices = dbg_voices.max(voices);
                    for &s in &mix {
                        dbg_peak = dbg_peak.max(s.abs());
                    }
                    if dbg_frames >= rate as u64 {
                        eprintln!(
                            "[audiodbg] {:>5} Hz | voices={} peak={:.5} blocks={}",
                            rate, dbg_voices, dbg_peak, dbg_blocks
                        );
                        dbg_frames = 0;
                        dbg_blocks = 0;
                        dbg_voices = 0;
                        dbg_peak = 0.0;
                    }
                }
                for (f, frame) in out.chunks_mut(channels.max(1)).enumerate() {
                    let (lv, rv) = (mix.get(2 * f).copied().unwrap_or(0.0), mix.get(2 * f + 1).copied().unwrap_or(0.0));
                    for (c, sample) in frame.iter_mut().enumerate() {
                        *sample = match c {
                            0 => lv,
                            1 => rv,
                            _ => 0.5 * (lv + rv),
                        };
                    }
                }
            },
            |e| eprintln!("[audio] stream error: {e}"),
            None,
        )
        .ok()?;
    stream.play().ok()?;
    println!("[audio] output {rate} Hz, {channels} ch on '{}'", device.name().unwrap_or_default());
    Some((AudioOut(stream), CryListener(listener)))
}

/// Whether the script layer is enabled (`SOD_CRY=0` disables it).
pub fn enabled() -> bool {
    !std::env::var("SOD_CRY").map(|v| v == "0").unwrap_or(false)
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
    // Original flow: start events, then the rules script creates the local
    // player on client connect. The controller then owns its position.
    let player = match host.start_game() {
        Ok(Some(p)) => p,
        _ => {
            eprintln!("[cry] the rules script created no player");
            return None;
        }
    };
    host.ctx.world.borrow_mut().set_pos(player, spawn);
    println!(
        "[cry] booted: {} entities, {} areas, {} script errors, {} missing binds",
        host.ctx.instances.borrow().len(),
        host.areas.borrow().len(),
        host.ctx.errors.borrow().len(),
        host.missing_report().len()
    );
    Some(CryHost { host, player })
}

/// Register the booted host and its per-frame systems with the app.
pub fn install(app: &mut App, host: CryHost) {
    app.init_resource::<CryPlayerPos>().init_resource::<CryStats>();
    if let Some((out, listener)) = open_audio(&host.host) {
        app.insert_non_send_resource(out);
        app.insert_resource(listener);
    }
    app.insert_non_send_resource(host);
    app.configure_sets(Update, (CrySet::Sync, CrySet::Tick).chain());
    app.add_systems(Update, cry_tick.in_set(CrySet::Tick));
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
