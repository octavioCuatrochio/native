//! Runtime Flash (Scaleform→SWF) HUD integration.
//!
//! The original game's in-game HUD is a Scaleform `IFlashPlayer` overlay: the
//! engine loads ten `Class3_*.gfx` element movies (`libs/ui/uielements/
//! class3hud.xml`), binds each with `<Constraints><Align mode="fullscreen"/>`,
//! and drives it with `IFlashPlayer::Invoke`/`SetVariable`.  This module runs
//! the same movies through embedded Ruffle (one player per movie), renders them
//! offscreen, and composites them over the 3D view; `bridge` maps our state
//! onto the same ActionScript functions and `loc` resolves `@label` text.
//!
//! Converted movies come from a local cache (see `cache`); nothing derived from
//! the game ships in the repository.  When the cache is absent the app prints
//! how to build it and the native text HUD stays in charge.
//!
//! The Ruffle dependency is pinned to the 0.7.0 revision the `ui_convert`
//! pipeline was validated against (workerC3b/C3c).

mod cache;
pub mod bridge;
pub mod loc;
pub mod player;

use std::path::{Path, PathBuf};

use bevy::image::Image;
use bevy::prelude::*;
use bevy::ui::{Display, ZIndex};
use sod_core::crypak::Vfs;

pub use bridge::HudPush;
pub use player::FlashHud;

/// Config handed from `install` to the startup system that builds the players.
#[derive(Resource)]
struct FlashConfig {
    dir: PathBuf,
    loc: loc::Loc,
}

/// One frame's HUD state, published by `hud_update` and consumed by `flash_apply`.
#[derive(Resource, Default)]
pub struct FlashPush(pub HudPush);

/// Whether the Flash HUD exists and should currently be drawn.
#[derive(Resource, Default)]
pub struct FlashUiActive {
    pub loaded: bool,
    pub visible: bool,
}

/// Marker for the full-screen Flash overlay image nodes.
#[derive(Component)]
pub struct FlashUiNode;

/// Install the Flash HUD when `play` mode is active.
///
/// `install_dir` is the game root (used only to print the converter command);
/// `vfs` is the mounted pak set, used to build the runtime localisation table.
pub fn install(app: &mut App, install_dir: &Path, vfs: &Vfs) {
    app.init_resource::<FlashUiActive>();
    app.init_resource::<FlashPush>();

    if std::env::var_os("SOD_TEXT_HUD").is_some() {
        println!("[ui] SOD_TEXT_HUD=1: native text HUD kept, Flash HUD disabled");
        return;
    }

    let dir = cache::cache_dir();
    if !cache::is_ready(&dir, cache::HUD_MOVIES) {
        cache::print_instructions(install_dir, &dir, cache::first_missing(&dir, cache::HUD_MOVIES));
        return;
    }

    app.insert_resource(FlashConfig { dir, loc: loc::Loc::load(vfs) });
    app.add_systems(Startup, setup_flash_hud);
    app.add_systems(Update, (sync_visibility, flash_apply.after(crate::hud_update)));
}

/// Exclusive startup system: build the Ruffle players and spawn the overlays.
///
/// Exclusive because a Ruffle `Player` is `!Send` and must be inserted into the
/// world as a non-send resource (`Commands` has no non-send insert in Bevy 0.19).
fn setup_flash_hud(world: &mut World) {
    let Some(cfg) = world.remove_resource::<FlashConfig>() else { return };
    let hud = {
        let mut images = world.resource_mut::<Assets<Image>>();
        player::FlashHud::load(&cfg.dir, cache::HUD_MOVIES, &mut images, cfg.loc)
    };
    match hud {
        Ok(hud) => {
            let nodes: Vec<(Handle<Image>, i32)> = hud.movies.iter().map(|m| (m.handle.clone(), m.z)).collect();
            for (handle, z) in nodes {
                world.spawn((
                    ImageNode::new(handle),
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(0.0),
                        top: Val::Px(0.0),
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    ZIndex(z),
                    FlashUiNode,
                ));
            }
            println!("[ui] Flash HUD live: {} movies from {}", hud.movies.len(), cfg.dir.display());
            world.insert_non_send_resource(hud);
            world.insert_resource(FlashUiActive { loaded: true, visible: true });
        }
        Err(e) => eprintln!("[ui] Flash HUD failed to start: {e:#}"),
    }
}

/// Keep the overlay display state consistent with `FlashUiActive`.
pub fn sync_visibility(active: Res<FlashUiActive>, mut q: Query<&mut Node, With<FlashUiNode>>) {
    let display = if active.loaded && active.visible { Display::Flex } else { Display::None };
    for mut node in &mut q {
        if node.display != display {
            node.display = display;
        }
    }
}

/// Push the frame's state into the HUD and advance the offscreen renderer.
pub fn flash_apply(
    flash: Option<NonSendMut<FlashHud>>,
    push: Res<FlashPush>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(mut flash) = flash else { return };
    let loc = flash.loc.take();
    bridge::apply(&mut flash, &push.0, loc.as_ref());
    flash.loc = loc;
    flash.tick(&mut images, 1);
}
