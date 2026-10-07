//! Locates the one-time converted Flash UI cache.
//!
//! The `.gfx` UI shipped with State of Decay is Scaleform, not standard SWF.
//! `tools/ui_convert/convert.py` rewrites it into Ruffle-loadable `.swf` with
//! the real atlases / fonts and resolved `@strings`.  Converted game assets are
//! deliberately not committed, so the app reads them from a local cache and
//! prints how to build it when it is missing.
//!
//! The set of HUD element movies mirrors the shipped
//! `libs/ui/uielements/class3hud.xml` (one `UIElement` per in-game HUD piece,
//! each with a `Class3_*.gfx` asset and a `<Constraints><Align mode="fullscreen"/>`).

use std::path::{Path, PathBuf};

/// One HUD element movie: stable name, cache file name and draw order.
///
/// `z` follows the `layer` attribute of the matching `class3hud.xml`
/// `UIElement` (reticle/stats/survey/notifications 0, radar/banners 1,
/// frontend 2, journal 3, center prompts 4, pause 5); lower layers first.
#[derive(Clone, Copy, Debug)]
pub struct MovieSpec {
    pub name: &'static str,
    pub file: &'static str,
    pub z: i32,
    /// Roughly how many rendered frames apart this movie needs recomposing.
    /// The radar is the only continuously rotating element.
    pub period: u64,
}

/// The in-game HUD elements wired by this first runtime slice.
pub const HUD_MOVIES: &[MovieSpec] = &[
    MovieSpec { name: "Class3HUDStats", file: "class3_stats.swf", z: 10, period: 6 },
    MovieSpec { name: "Class3HUDReticle", file: "class3_reticle.swf", z: 11, period: 6 },
    MovieSpec { name: "Class3HUDRadar", file: "class3_radar.swf", z: 12, period: 1 },
    MovieSpec { name: "Class3HUDBanners", file: "class3_banners.swf", z: 13, period: 12 },
    MovieSpec { name: "Class3HUDCenterPrompts", file: "class3_centerprompts.swf", z: 14, period: 12 },
    MovieSpec { name: "Class3HUDNotifications", file: "class3_notifications.swf", z: 15, period: 12 },
];

/// `~/.cache/sod-native/ui` (honouring `XDG_CACHE_HOME`).
pub fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("sod-native").join("ui")
}

/// True when every HUD movie exists in `dir`.
pub fn is_ready(dir: &Path, specs: &[MovieSpec]) -> bool {
    specs.iter().all(|s| dir.join(s.file).is_file())
}

/// The first movie that is missing, if any (for a precise diagnostic).
pub fn first_missing(dir: &Path, specs: &[MovieSpec]) -> Option<&'static str> {
    specs.iter().find(|s| !dir.join(s.file).is_file()).map(|s| s.file)
}

/// Print the exact command that builds the cache from the user's install.
///
/// The converter lives in the research repo (`~/Desktop/sod-re/tools/ui_convert`)
/// but is invoked with a path relative to the repository root, matching its
/// README.
pub fn print_instructions(install: &Path, dir: &Path, missing: Option<&str>) {
    eprintln!("[ui] converted Flash UI cache not found at {}", dir.display());
    if let Some(m) = missing {
        eprintln!("[ui] missing movie: {m}");
    }
    eprintln!("[ui] build it once from the game install with:");
    eprintln!(
        "       python3 tools/ui_convert/convert.py \"{}\" \"{}\" --strings",
        install.display(),
        dir.display()
    );
    eprintln!("[ui] continuing without the Flash HUD (set SOD_TEXT_HUD=1 to force the text HUD)");
}
