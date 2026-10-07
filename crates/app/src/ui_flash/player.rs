//! One Ruffle player per HUD movie, rendered offscreen and read back.
//!
//! Ruffle 0.7.0 renders through wgpu 30 while Bevy 0.19 uses wgpu 29, so the
//! two GPU devices cannot share textures.  Instead each movie owns a Ruffle
//! `WgpuRenderBackend<TextureTarget>` on one shared device, renders to an
//! offscreen `Rgba8Unorm` target, and we read the frame back into a Bevy
//! `Image` (the same path the shipped `exporter` uses: `render()` then
//! `capture_frame()`).  Capture is budgeted to at most one movie per frame and
//! movies are refreshed on a round-robin period, keeping the Flash work off the
//! main frame budget.
//!
//! Movies are built with a forced `ShowAll` scale mode at the window
//! resolution: the shipped HUD scripts set `Stage.scaleMode = "noScale"` but
//! the engine binds every element with `<Constraints><Align mode="fullscreen"/>`
//! (`libs/ui/uielements/class3hud.xml`), i.e. the 1920×1080 authored stage is
//! fitted to the screen — exactly what `ShowAll` does.
//!
//! # Host → ActionScript
//!
//! Ruffle 0.7 exposes no public host→AVM1 invoke (the engine's
//! `IFlashPlayer::Invoke`/`SetVariable` have no equivalent API), so we use the
//! supported `ExternalInterface` path: before loading, a `DoAction` tag is
//! appended to each movie that registers every HUD function as an
//! `ExternalInterface` callback (`ExternalInterface.addCallback(name, this, fn)`).
//! The host then calls `Player::call_internal_interface`, which is exactly the
//! `GFxMovieRoot::Invoke` equivalent.

use std::any::Any;
use std::path::Path;
use std::sync::{Arc, Mutex};

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use ruffle_core::backend::log::LogBackend;
use ruffle_core::context::UpdateContext;
use ruffle_core::external::{ExternalInterfaceProvider, FsCommandProvider, Value as ExternalValue};
use ruffle_core::limits::ExecutionLimit;
use ruffle_core::tag_utils::SwfMovie;
use ruffle_core::{Color, Player, PlayerBuilder, StageScaleMode};
use ruffle_render_wgpu::backend::WgpuRenderBackend;
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::wgpu;

use super::cache::MovieSpec;
use super::loc::Loc;

/// Offscreen render/readback resolution.  Matches the app window and the
/// 16:9 authored HUD stage.
pub const TARGET_W: u32 = 1280;
pub const TARGET_H: u32 = 720;

/// ActionScript functions to expose to the host for each HUD element.  These
/// are the callable `<function>`s declared for the element in
/// `libs/ui/uielements/class3hud.xml`; registering names a movie does not
/// define is harmless (the value is `undefined` and `addCallback` returns false).
const CALLBACKS: &[&str] = &[
    // Class3HUDStats
    "Setup3DElements",
    "cry_onSetup",
    "SetHealth",
    "SetStamina",
    "SetCharacterUiVisible",
    "GunStatsShow",
    "GunStatsHide",
    "SetLocation",
    "SetBuildNumber",
    // Class3HUDRadar
    "SetMapRotation",
    "SetMapZoom",
    "SetMapLoc",
    "SetMissionObjectiveText",
    "SetThreatLevel",
    "ShowZombieCounter",
    "SetMinimapUiVisible",
    "SetMoraleBar",
    "PlayAlert",
    // Class3HUDBanners
    "MissionBannerStart",
    "MissionBannerUpdate",
    "MissionBannerHide",
    // Class3HUDCenterPrompts
    "ShowUseButton",
    "HideUseButton",
    "ShowHint",
    "HideHint",
    // Class3HUDNotifications
    "AddRewardMessage",
    // Class3HUDReticle
    "OnGunFired",
    "SetUnarmedReticleMode",
];

/// A value passed to an ActionScript function.  ActionScript numbers are
/// doubles, so integers are widened exactly.
#[derive(Clone, Debug)]
pub enum UiValue {
    Int(i32),
    Float(f64),
    Bool(bool),
    Str(String),
}

struct RenderLog;

impl LogBackend for RenderLog {
    fn avm_trace(&self, message: &str) {
        log::debug!("[ui avm] {message}");
    }
    fn avm_warning(&self, message: &str) {
        log::warn!("[ui avm] {message}");
    }
}

/// No-op `ExternalInterface` provider: its only job is to make
/// `ExternalInterface.available` true so `addCallback` accepts our stub.
struct HostBridge;

impl ExternalInterfaceProvider for HostBridge {
    fn call_method(&self, _context: &mut UpdateContext<'_>, name: &str, args: &[ExternalValue]) -> ExternalValue {
        log::debug!("[ui] ExternalInterface.call({name}, {} args)", args.len());
        ExternalValue::Null
    }

    fn on_callback_available(&self, name: &str) {
        log::debug!("[ui] callback registered: {name}");
    }

    fn get_id(&self) -> Option<String> {
        None
    }
}

/// Receives the movie→host `fscommand` calls the HUD emits (button/selection
/// events); the real game routes these to its flow graph.  Logged for now.
struct FsCommands;

impl FsCommandProvider for FsCommands {
    fn on_fs_command(&self, command: &str, args: &str) -> bool {
        log::debug!("[ui] fscommand({command}, {args})");
        true
    }
}

pub struct FlashMovie {
    pub name: &'static str,
    pub z: i32,
    pub handle: Handle<Image>,
    period: u64,
    next_due: u64,
    pub(crate) player: Arc<Mutex<Player>>,
}

/// All live HUD element players plus the shared offscreen device.
///
/// This type is `!Send` (Ruffle's `Player` uses `Rc`), so it is stored as a
/// Bevy *non-send* resource and only ever touched from the main thread.
pub struct FlashHud {
    pub movies: Vec<FlashMovie>,
    frame: u64,
    /// Runtime localisation table (resolves `@label` before every string push).
    pub(crate) loc: Option<Loc>,
    /// Previous push, for edge-triggering one-shot messages (banners etc.).
    pub(crate) last: Option<super::bridge::HudPush>,
}

impl FlashHud {
    /// Build one player per HUD movie from the converted cache in `dir`.
    ///
    /// A movie that fails to load is reported and skipped; the rest still run.
    pub fn load(
        dir: &Path,
        specs: &[MovieSpec],
        images: &mut Assets<Image>,
        loc: Loc,
    ) -> anyhow::Result<Self> {
        // One shared instance/device for every movie.
        let first_backend =
            WgpuRenderBackend::for_offscreen((TARGET_W, TARGET_H), wgpu::Backends::PRIMARY, wgpu::PowerPreference::HighPerformance)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        let descriptors: Arc<Descriptors> = first_backend.descriptors().clone();

        let mut movies = Vec::new();
        let mut first_backend = Some(first_backend);
        for spec in specs {
            let path = dir.join(spec.file);
            let data = match std::fs::read(&path) {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("[ui] failed to read {}: {e}", path.display());
                    continue;
                }
            };
            let injected = match inject_callbacks(&data, CALLBACKS) {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("[ui] failed to prepare {}: {e:#}", path.display());
                    data
                }
            };
            if let Some(dir) = std::env::var_os("SOD_UI_DUMP") {
                let _ = std::fs::write(std::path::Path::new(&dir).join(spec.file), &injected);
            }
            let url = format!("file://{}", path.display());
            let movie = match SwfMovie::from_data(&injected, url, None, None) {
                Ok(m) => m,
                Err(e) => {
                    log::warn!("[ui] failed to parse {}: {e}", path.display());
                    continue;
                }
            };

            let backend = match first_backend.take() {
                Some(b) => b,
                None => {
                    let target = TextureTarget::new(&descriptors.device, (TARGET_W, TARGET_H))
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    WgpuRenderBackend::new(descriptors.clone(), target)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                }
            };

            let player = PlayerBuilder::new()
                .with_renderer(backend)
                .with_movie(movie)
                .with_viewport_dimensions(TARGET_W, TARGET_H, 1.0)
                .with_scale_mode(StageScaleMode::ShowAll, true)
                .with_autoplay(true)
                .with_external_interface(Box::new(HostBridge))
                .with_fs_commands(Box::new(FsCommands))
                .with_log(RenderLog)
                .build();
            // The HUD composites over the 3D view.  `Player::render` clears to
            // `background_color().unwrap_or(WHITE)`, so `None` would paint an
            // opaque white blanket over the scene; clear to fully transparent
            // (alpha 0) instead.
            player.lock().unwrap().set_background_color(Some(Color::TRANSPARENT));

            let handle = images.add(Image::new(
                Extent3d { width: TARGET_W, height: TARGET_H, depth_or_array_layers: 1 },
                TextureDimension::D2,
                vec![0u8; (TARGET_W * TARGET_H * 4) as usize],
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::default(),
            ));

            movies.push(FlashMovie { name: spec.name, z: spec.z, handle, period: spec.period.max(1), next_due: 0, player });
        }

        if movies.is_empty() {
            anyhow::bail!("no HUD movies could be loaded from {}", dir.display());
        }

        let mut hud = Self { movies, frame: 0, loc: Some(loc), last: None };
        // The engine calls `cry_onSetup(false)` / `Setup3DElements(false)` once
        // after loading each element (see the `<functions>` block of
        // `class3hud.xml`).  Running setup hides the parts that are driven by
        // gameplay and positions the minimap.  A couple of frames ensure the
        // injected callback stub has run before the first host call.
        for m in &mut hud.movies {
            {
                let mut p = m.player.lock().unwrap();
                p.set_is_playing(true);
                force_root_play(&mut p);
                for _ in 0..3 {
                    p.preload(&mut ExecutionLimit::none());
                    p.run_frame();
                }
            }
            invoke(&m.player, "Setup3DElements", &[UiValue::Bool(false)]);
            invoke(&m.player, "cry_onSetup", &[UiValue::Bool(false)]);
        }
        Ok(hud)
    }

    pub fn is_empty(&self) -> bool {
        self.movies.is_empty()
    }

    /// Run every movie's timeline, then render+capture the movies that are due.
    ///
    /// `budget` caps how many GPU readbacks happen this frame (kept at 1 so the
    /// Flash work stays off the main frame budget).
    pub fn tick(&mut self, images: &mut Assets<Image>, budget: usize) {
        for m in &mut self.movies {
            let mut p = m.player.lock().unwrap();
            p.preload(&mut ExecutionLimit::none());
            p.run_frame();
        }

        let mut done = 0usize;
        while done < budget {
            let due = self
                .movies
                .iter()
                .enumerate()
                .filter(|(_, m)| m.next_due <= self.frame)
                .min_by_key(|(_, m)| m.next_due)
                .map(|(i, _)| i);
            let Some(i) = due else { break };
            self.movies[i].next_due = self.frame + self.movies[i].period;
            self.capture(i, images);
            done += 1;
        }
        self.frame = self.frame.wrapping_add(1);
    }

    fn capture(&mut self, index: usize, images: &mut Assets<Image>) {
        let m = &mut self.movies[index];
        {
            let mut p = m.player.lock().unwrap();
            p.render();
            let renderer = <dyn Any>::downcast_mut::<WgpuRenderBackend<TextureTarget>>(p.renderer_mut());
            let Some(renderer) = renderer else {
                log::warn!("[ui] {}: unexpected render backend", m.name);
                return;
            };
            let Some(frame) = renderer.capture_frame() else { return };
            let (w, h) = frame.dimensions();
            let raw = frame.into_raw();
            if let Some(dir) = std::env::var_os("SOD_UI_DUMP_IMG") {
                let dir = std::path::Path::new(&dir);
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(dir.join(format!("{}.rgba", m.name)), &raw);
                let _ = std::fs::write(dir.join(format!("{}.txt", m.name)), format!("{w}x{h}"));
            }
            if let Some(mut image) = images.get_mut(&m.handle) {
                match image.data.as_mut() {
                    Some(data) if data.len() == raw.len() => data.copy_from_slice(&raw),
                    _ => {
                        image.data = Some(raw);
                        image.texture_descriptor.size = Extent3d { width: w, height: h, depth_or_array_layers: 1 };
                    }
                }
            }
        }
    }
}

/// Invoke an ActionScript function on the movie root through the callback stub.
pub fn invoke(player: &Arc<Mutex<Player>>, func: &str, args: &[UiValue]) {
    let ext: Vec<ExternalValue> = args
        .iter()
        .map(|v| match v {
            UiValue::Int(i) => ExternalValue::from(*i),
            UiValue::Float(f) => ExternalValue::from(*f),
            UiValue::Bool(b) => ExternalValue::from(*b),
            UiValue::Str(s) => ExternalValue::from(s.clone()),
        })
        .collect();
    player.lock().unwrap().call_internal_interface(func, ext);
}

/// Append a `DoAction` tag registering each name as an `ExternalInterface`
/// callback: `ExternalInterface.addCallback(name, undefined, _root[name])`.
///
/// Uses the `swf` crate's AVM1 writer so the bytecode is well formed.  The
/// callback `this` is `undefined`; the HUD functions use `_root`/`_global`, not
/// `this`.  Injected tags are not committed; this runs on the cache at startup.
fn inject_callbacks(data: &[u8], names: &[&str]) -> anyhow::Result<Vec<u8>> {
    use swf::avm1::types::{Action, Push, Value as V};

    let buf = swf::decompress_swf(data).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let version = buf.header.swf_header().version;
    let header = buf.header.clone();
    let parsed = swf::parse_swf(&buf).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut payload = Vec::new();
    {
        let mut w = swf::avm1::write::Writer::new(&mut payload, version);
        for name in names {
            let s: &swf::SwfStr = (*name).into();
            // `ExternalInterface` is only exposed at
            // `flash.external.ExternalInterface` (avm1/globals.rs), not as a
            // top-level global.
            let ext: &swf::SwfStr = "flash.external.ExternalInterface".into();
            let add: &swf::SwfStr = "addCallback".into();
            // _root[name]  (the function value; arg 3 = method)
            w.write_action(&Action::Push(Push { values: vec![V::Str(s)] }))?;
            w.write_action(&Action::GetVariable)?;
            // this = undefined (arg 2)
            w.write_action(&Action::Push(Push { values: vec![V::Undefined] }))?;
            // display name (arg 1)
            w.write_action(&Action::Push(Push { values: vec![V::Str(s)] }))?;
            // arg count
            w.write_action(&Action::Push(Push { values: vec![V::Int(3)] }))?;
            // flash.external.ExternalInterface
            w.write_action(&Action::Push(Push { values: vec![V::Str(ext)] }))?;
            w.write_action(&Action::GetVariable)?;
            // .addCallback
            w.write_action(&Action::Push(Push { values: vec![V::Str(add)] }))?;
            w.write_action(&Action::CallMethod)?;
            w.write_action(&Action::Pop)?;
        }
        // A trace marker so `RUST_LOG=debug` shows the stub actually ran.
        let marker: &swf::SwfStr = "sod-ui-callbacks".into();
        w.write_action(&Action::Push(Push { values: vec![V::Str(marker)] }))?;
        w.write_action(&Action::Trace)?;
        w.write_action(&Action::End)?;
    }

    // Run on frame 1: insert immediately before the first `ShowFrame`, which
    // closes frame 1 (after the timeline's own function definitions).  Putting
    // it at the end would land in a later frame that a stopped HUD never plays.
    let mut tags = parsed.tags;
    let tag = swf::Tag::DoAction(&payload);
    let at = tags
        .iter()
        .position(|t| matches!(t, swf::Tag::ShowFrame))
        .or_else(|| tags.iter().position(|t| matches!(t, swf::Tag::End)));
    match at {
        Some(i) => tags.insert(i, tag),
        None => {
            tags.push(tag);
            tags.push(swf::Tag::End);
        }
    }
    let mut out = Vec::new();
    swf::write::write_swf(header.swf_header(), &tags, &mut out).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(out)
}

fn force_root_play(player: &mut Player) {
    player.mutate_with_update_context(|ctx| {
        if let Some(root) = ctx.stage.root_clip() {
            if let Some(mc) = root.as_movie_clip() {
                if !mc.playing() {
                    mc.play();
                }
            }
        }
    });
}
