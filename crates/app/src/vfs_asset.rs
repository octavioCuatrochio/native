//! A native Bevy 0.19.1 [`AssetReader`] over sod-core's in-memory CryEngine
//! `.pak` VFS, so `asset_server.load("pak://objects/.../camper.dds")` reads the
//! bytes straight out of the mounted paks instead of the filesystem.
//!
//! # Status / verification
//!
//! Every signature and import path below was checked against the actual
//! crates.io 0.19.1 sources resolved by `native/Cargo.lock`, located at
//! `$REG = /home/battlestation/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f`.
//! Each item carries a `// src: <crate>/<file>:<line>` tag. The code is
//! expected to compile as-is once the module is included (see step 1) — this
//! exact `impl AssetReader` shape is what Bevy itself uses in
//! `bevy_asset-0.19.1/src/io/memory.rs:374-411`.
//!
//! # Integration (exact steps)
//!
//! 1. Include this file as a module (one line in `main.rs`, which this task
//!    deliberately does not touch):
//!
//!    ```ignore
//!    mod vfs_asset;
//!    use crate::vfs_asset::vfs_asset_source;
//!    ```
//!
//! 2. Build/mount the `Vfs` exactly as the current `main.rs` does, then wrap
//!    it so it is shareable: `Arc<Mutex<Vfs>>`.
//!
//! 3. Register the named source **before** adding `DefaultPlugins`
//!    (`AssetPlugin` builds sources in its `Plugin::build`, and
//!    `register_asset_source` only seeds the `AssetSourceBuilders` resource;
//!    registering afterwards logs an error and is ignored):
//!
//!    ```ignore
//!    use bevy::prelude::*; // brings `AssetApp` (register_asset_source) into scope
//!    use sod_core::crypak::Vfs;
//!    use std::sync::{Arc, Mutex};
//!
//!    fn main() {
//!        let mut vfs = Vfs::new();
//!        vfs.mount("/path/to/game/Game.pak").unwrap();
//!        let vfs = Arc::new(Mutex::new(vfs));
//!
//!        let mut app = App::new();
//!
//!        // (a) register FIRST, before DefaultPlugins/AssetPlugin
//!        app.register_asset_source("pak", vfs_asset_source(vfs.clone()));
//!
//!        // (b) then plugins and assets
//!        app.add_plugins(DefaultPlugins);
//!        app.add_systems(Startup, |assets: Res<AssetServer>| {
//!            // scheme is `<source-name>://<path-inside-source>`
//!            let _img: Handle<Image> = assets.load("pak://objects/.../camper.dds");
//!        });
//!        app.run();
//!    }
//!    ```
//!
//! 4. Load with the `pak://` scheme. `.dds` resolves to a `Handle<Image>`
//!    because `ImageLoader` registers the `dds` extension when the `dds`
//!    feature is on (`bevy_image-0.19.1/src/image.rs:245,322-323`) and the
//!    `ImageLoader` itself is registered by `bevy_render`'s `TexturePlugin`
//!    (`bevy_render-0.19.1/src/texture/mod.rs:59`).
//!
//! ## Import-path correction vs. the task brief
//!
//! `AssetSourceBuilder` and `AssetSourceId` live in `bevy::asset::io`, **not**
//! `bevy::asset`: `bevy_asset` declares `pub mod io;` and `io/mod.rs` does
//! `pub use source::*;` (`bevy_asset-0.19.1/src/lib.rs:153`,
//! `src/io/mod.rs:26`, `src/io/source.rs:24,119`), while the crate root only
//! has a *private* `use crate::io::{...}` (`src/lib.rs:205-206`) and no
//! `pub use io::*`. `AssetApp` is defined in the crate root (`src/lib.rs:554`)
//! and re-exported from `bevy::asset::prelude` (`src/lib.rs:162-171`), so
//! `bevy::prelude::*` or `bevy::asset::AssetApp` both work.
//!
//! # Mutex / Send notes
//!
//! `AssetReader: Send + Sync + 'static` (`src/io/mod.rs:190`). `Arc<Mutex<Vfs>>`
//! satisfies this because `Vfs` is an owned `Vec<Pak>` + `Option<PathBuf>`
//! (`crates/core/src/crypak.rs:152-155`) and therefore `Send + Sync`.
//! `std::sync::MutexGuard` is `!Send`, so the guard is scoped to a block and
//! dropped before anything else; `read` has no `.await` points at all, so the
//! returned future stays `Send` under the `multi_threaded` feature.
//!
//! # Fallback plan (only if this `AssetReader` route ever fails to compile)
//!
//! If a future `AssetReader` change makes the RPITIT return types unusable,
//! bypass the source entirely with a custom [`AssetLoader`] registered for the
//! `dds` extension. It reads the `Vfs` directly; `LoadContext::path()` gives the
//! asset path. Sketch (not compiled here; needs `bevy::asset::{AssetLoader,
//! LoadContext}` and the `RenderAssetUsages`/`Image` imports already used by
//! `main.rs`):
//!
//! ```ignore
//! use bevy::asset::{AssetLoader, LoadContext};
//! use bevy::asset::io::Reader;
//! use bevy::image::{CompressedImageFormats, Image, ImageSampler, ImageType};
//! use bevy::asset::RenderAssetUsages;
//! use bevy::tasks::ConditionalSendFuture;
//!
//! #[derive(Default)]
//! struct VfsDdsLoader {
//!     vfs: std::sync::Arc<std::sync::Mutex<sod_core::crypak::Vfs>>,
//! }
//!
//! impl AssetLoader for VfsDdsLoader {
//!     type Asset = Image;
//!     type Settings = ();
//!     type Error = std::io::Error; // must impl Into<BevyError>; io::Error does
//!
//!     fn extensions(&self) -> &[&str] { &["dds"] }
//!
//!     fn load<'a>(
//!         &'a self,
//!         _reader: &'a mut dyn Reader,
//!         _settings: &'a Self::Settings,
//!         ctx: &'a mut LoadContext,
//!     ) -> impl ConditionalSendFuture<Output = Result<Image, Self::Error>> {
//!         async move {
//!             let key = ctx.path().path().to_string_lossy().replace('\\', "/");
//!             let bytes = {
//!                 let vfs = self.vfs.lock().unwrap_or_else(|e| e.into_inner());
//!                 vfs.read(&key)
//!                     .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, key))?
//!             };
//!             Image::from_buffer(
//!                 &bytes,
//!                 ImageType::Extension("dds"),
//!                 CompressedImageFormats::BC,
//!                 true,
//!                 ImageSampler::linear(),
//!                 RenderAssetUsages::default(),
//!             )
//!         }
//!     }
//! }
//! // app.add_plugins(DefaultPlugins).register_asset_loader(VfsDdsLoader { vfs });
//! // then load with a plain (non-`pak`) path, or use `load_asset`/a `LoadBuilder`.
//! ```
//!
//! [`AssetReader`]: bevy::asset::io::AssetReader
//! [`AssetLoader`]: bevy::asset::AssetLoader

use bevy::asset::io::{AssetReader, AssetReaderError, PathStream, Reader, VecReader};
// `AssetSourceBuilder` is in `io`, not the crate root. See module docs above.
// src: bevy_asset-0.19.1/src/lib.rs:153 (`pub mod io`) + src/io/mod.rs:26
//      (`pub use source::*`) + src/io/source.rs:119 (`pub struct AssetSourceBuilder`)
use bevy::asset::io::AssetSourceBuilder;
use sod_core::crypak::Vfs;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// [`AssetReader`] over a shared in-memory [`Vfs`].
///
/// `Arc<Mutex<Vfs>>` is `Send + Sync + 'static`, satisfying the
/// `AssetReader: Send + Sync + 'static` supertrait bound
/// (`bevy_asset-0.19.1/src/io/mod.rs:190`). `Clone` lets the source factory
/// hand out a fresh reader on each build.
#[derive(Clone)]
pub struct VfsAssetReader {
    vfs: Arc<Mutex<Vfs>>,
}

impl VfsAssetReader {
    /// Wrap an already-shared VFS.
    pub fn new(vfs: Arc<Mutex<Vfs>>) -> Self {
        Self { vfs }
    }

    /// Lock, read, and release. Never holds the guard across an `.await`.
    fn read_bytes(&self, key: &str) -> Option<Vec<u8>> {
        let vfs = self
            .vfs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        vfs.read(key)
    }
}

// Trait: `pub trait AssetReader: Send + Sync + 'static`
// src: bevy_asset-0.19.1/src/io/mod.rs:190
//
//   fn read<'a>(&'a self, path: &'a Path) -> impl AssetReaderFuture<Value: Reader + 'a>;
//   fn read_meta<'a>(&'a self, path: &'a Path) -> impl AssetReaderFuture<Value: Reader + 'a>;
//   fn read_directory<'a>(&'a self, path: &'a Path)
//       -> impl ConditionalSendFuture<Output = Result<Box<PathStream>, AssetReaderError>>;
//   fn is_directory<'a>(&'a self, path: &'a Path)
//       -> impl ConditionalSendFuture<Output = Result<bool, AssetReaderError>>;
// src: bevy_asset-0.19.1/src/io/mod.rs:212-224
//
// `AssetReaderFuture` is `ConditionalSendFuture<Output = Result<Value, _>>` with
// a blanket impl, which is why the idiomatic impl is a plain `async fn`
// returning `Result<...>.` src: bevy_asset-0.19.1/src/io/mod.rs:169-180
impl AssetReader for VfsAssetReader {
    async fn read<'a>(&'a self, path: &'a Path) -> Result<impl Reader + 'a, AssetReaderError> {
        // Asset paths use forward slashes; `Vfs::read` normalises backslashes
        // too (`crates/core/src/crypak.rs:187-205`), but normalise here as well
        // so the lookup key is canonical.
        let key = path.to_string_lossy().replace('\\', "/");

        // Scope the guard: `std::sync::MutexGuard` is `!Send`. There is no
        // `.await` in this future, so it is already `Send`; keeping the guard
        // inside this block keeps it that way under `multi_threaded`.
        let bytes = self.read_bytes(&key);

        match bytes {
            // `VecReader` is owned bytes and impls `AsyncRead + AsyncSeek +
            // Reader` (`ReaderNotSeekableError`-free seek support).
            // src: bevy_asset-0.19.1/src/io/mod.rs:600-651 (new at :608)
            Some(bytes) => Ok(VecReader::new(bytes)),
            // `AssetReaderError::NotFound` tells the `AssetServer` this path is
            // simply absent (no default meta / no directory).
            // src: bevy_asset-0.19.1/src/io/mod.rs:45-48
            None => Err(AssetReaderError::NotFound(path.to_path_buf())),
        }
    }

    // Concrete `VecReader` Ok-type because this body only ever returns `Err`;
    // an opaque `impl Reader` here cannot be inferred. Returning a concrete
    // type is legal for an RPITIT method and still satisfies
    // `Value: Reader + 'a` (`VecReader: 'static`).
    async fn read_meta<'a>(&'a self, path: &'a Path) -> Result<VecReader, AssetReaderError> {
        // Paks carry no `.meta` sidecars. `NotFound` (not `Io`) is what makes
        // the `AssetServer` fall back to the loader's default meta; default
        // `AssetPlugin::meta_check` is `AssetMetaCheck::Always`, so this fires
        // on every load.
        // src: bevy_asset-0.19.1/src/lib.rs:319-329;
        //      bevy_asset-0.19.1/src/server/mod.rs:1616-1629
        Err(AssetReaderError::NotFound(path.to_path_buf()))
    }

    async fn read_directory<'a>(
        &'a self,
        path: &'a Path,
    ) -> Result<Box<PathStream>, AssetReaderError> {
        // Single-file loads never hit this path (only folder loads do), and a
        // `Box<PathStream>` needs `futures_lite::Stream`, which Bevy does not
        // re-export. `NotFound` is a valid "no such directory".
        // src: bevy_asset-0.19.1/src/io/mod.rs:216-219,312
        Err(AssetReaderError::NotFound(path.to_path_buf()))
    }

    async fn is_directory<'a>(&'a self, _path: &'a Path) -> Result<bool, AssetReaderError> {
        // The VFS exposes flat leaf lookups only.
        Ok(false)
    }
}

/// Build a Bevy named asset source backed by `vfs`.
///
/// Must be handed to `app.register_asset_source("pak", ...)` **before**
/// `DefaultPlugins`/`AssetPlugin` is added.
///
/// Factory signature it satisfies:
/// `AssetSourceBuilder::new(impl FnMut() -> Box<dyn ErasedAssetReader> + Send + Sync + 'static)`.
/// src: bevy_asset-0.19.1/src/io/source.rs:151-166
///
/// `VfsAssetReader: AssetReader` gets `ErasedAssetReader` for free via the
/// blanket impl, so `Box::new(...)` unsizes into `Box<dyn ErasedAssetReader>`.
/// src: bevy_asset-0.19.1/src/io/mod.rs:271-308
///
/// Registration entry point (`AssetApp::register_asset_source`):
/// src: bevy_asset-0.19.1/src/lib.rs:563-567 (decl), 607-625 (impl)
pub fn vfs_asset_source(vfs: Arc<Mutex<Vfs>>) -> AssetSourceBuilder {
    // If type inference ever balks at the closure's `Box<...>` -> `Box<dyn
    // ErasedAssetReader>` coercion, annotate the closure return:
    //   move || -> Box<dyn bevy::asset::io::ErasedAssetReader> { ... }
    // (import `bevy::asset::io::ErasedAssetReader`). Bevy's own code relies on
    // the unannotated form, e.g. bevy_asset-0.19.1/src/lib.rs:2121.
    AssetSourceBuilder::new(move || Box::new(VfsAssetReader::new(vfs.clone())))
}
