//! Distance-based model + instance streaming.
//!
//! The whole Class3 map holds ~1.1k unique CGF models and ~90k placed
//! instances; decoding and uploading all of them at startup costs several GB
//! and seconds of blocking work (see the `world:` log lines). This module keeps
//! startup cheap by loading only what is near the player and streaming the rest
//! in, nearest-first, from a background thread.
//!
//! Split of responsibilities:
//!
//! * [`StreamInstance`] — a placed instance's transform (in CryEngine Z-up
//!   space) plus the information needed to spawn it (vegetation group index).
//! * [`ModelDecoder`] — a worker thread that reads a `.cgf` from the virtual
//!   file system, decodes its render meshes/material palette/textures and sends
//!   the plain-data payload back over a channel. The main thread never blocks
//!   on file I/O or CGF/mesh decoding.
//! * [`plan`] — the pure near-first request ordering: given each model's
//!   nearest instance distance it picks which pending models to request next.
//!
//! The Bevy-facing half (`update_streaming` in `main.rs`) turns decoded payloads
//! into `Assets`/handles on the main thread and spawns/despawns the instances,
//! bounded to a few milliseconds of work per frame.
//!
//! The numbers recovered from the engine (`GetMaxViewDist`, `GetObjectLOD`,
//! `e_ViewDistRatioVegetation`, `e_VegetationSpritesDistanceRatio`) all live in
//! `sod_core::architecture`/`sod_core::vegetation`; this module only decides
//! *when* a model and its instances enter or leave the world.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sod_core::crypak::Vfs;
use sod_core::material::{self, MaterialSlot};
use sod_core::mesh::{self, DecodedMesh};

/// One placed instance that should exist while its model is resident.
///
/// `rot` is a `[x, y, z, w]` Cry-space rotation (vegetation yaw is baked into it
/// by the loader). Positions are CryEngine-native `[x, y, z]` metres, the same
/// space as `CameraState::target`.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamInstance {
    /// Table A/B model path.
    pub model: String,
    /// Cry-space world position.
    pub pos: [f32; 3],
    /// Cry-space rotation quaternion `[x, y, z, w]`.
    pub rot: [f32; 4],
    /// Uniform / non-uniform render scale.
    pub scale: [f32; 3],
    /// `true` for vegetation (uses the group's distances); `false` for a brush
    /// or prop (uses the per-object `ViewDistRatio`/`LodRatio`).
    pub vegetation: bool,
    /// Table A group index for vegetation, otherwise ignored.
    pub group: u16,
    /// Normalised per-object `m_ucViewDistRatio/100` (brushes; 1.0 = normal).
    pub view_dist_ratio: f32,
    /// Normalised per-object `m_ucLodRatio/100` (brushes; 1.0 = normal).
    pub lod_ratio: f32,
}

/// A decoded model payload produced by [`ModelDecoder`], ready to be turned into
/// Bevy assets on the main thread.
pub struct DecodedModel {
    pub path: String,
    pub meshes: Vec<DecodedMesh>,
    pub slots: Vec<MaterialSlot>,
    /// Raw texture bytes keyed by the path in the material palette.
    pub textures: Vec<(String, Vec<u8>)>,
}

/// Background `.cgf`/`.mtl`/texture decoder.
///
/// The worker owns an `Arc<Vfs>`; the main thread owns the request `Sender` and
/// drains results from the `Mutex`-wrapped `Receiver` (kept behind a `Mutex` so
/// the whole thing is `Sync` and can live in a Bevy `Resource`).
pub struct ModelDecoder {
    tx: Sender<String>,
    rx: Mutex<Receiver<DecodedModel>>,
    _worker: JoinHandle<()>,
}

impl ModelDecoder {
    /// Spawn the decode worker.
    pub fn new(vfs: Arc<Vfs>) -> ModelDecoder {
        let (job_tx, job_rx) = channel::<String>();
        let (res_tx, res_rx) = channel::<DecodedModel>();
        let worker = std::thread::Builder::new()
            .name("sod-model-stream".into())
            .spawn(move || {
                while let Ok(path) = job_rx.recv() {
                    let Some(bytes) = vfs.read(&path) else { continue };
                    // A malformed model must not kill the worker.
                    let dm = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        mesh::decode_model_render(&bytes)
                    })) {
                        Ok(Some(dm)) => dm,
                        _ => continue,
                    };
                    let slots = dm
                        .material_name
                        .as_deref()
                        .map(|name| material::load_mtl(&vfs, &path, name))
                        .unwrap_or_default();
                    let mut textures = Vec::new();
                    for slot in &slots {
                        for (_, tp) in &slot.textures {
                            if let Some(b) = vfs.read(tp) {
                                textures.push((tp.clone(), b));
                            }
                        }
                    }
                    if res_tx
                        .send(DecodedModel { path, meshes: dm.meshes, slots, textures })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("spawn sod-model-stream worker");
        ModelDecoder { tx: job_tx, rx: Mutex::new(res_rx), _worker: worker }
    }

    /// Queue a model for decoding. Best-effort: returns `false` if the worker is gone.
    pub fn request(&self, path: &str) -> bool {
        self.tx.send(path.to_string()).is_ok()
    }

    /// Drain finished models, spending at most `budget` of wall time so a burst
    /// of completions cannot block the frame.
    pub fn drain(&self, budget: Duration) -> Vec<DecodedModel> {
        let t0 = Instant::now();
        let mut out = Vec::new();
        let rx = self.rx.lock().expect("model decoder receiver poisoned");
        loop {
            match rx.try_recv() {
                Ok(m) => out.push(m),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
            if t0.elapsed() >= budget {
                break;
            }
        }
        out
    }
}

/// Distances that drive the streamer, in metres.
#[derive(Debug, Clone, Copy)]
pub struct StreamRadii {
    /// Startup: decode models with an instance at least this close.
    pub near: f32,
    /// Request a pending model once an instance is this close.
    pub load: f32,
    /// Drop a resident model once every instance is farther than this.
    pub unload: f32,
}

impl Default for StreamRadii {
    fn default() -> Self {
        StreamRadii { near: 400.0, load: 900.0, unload: 1400.0 }
    }
}

impl StreamRadii {
    /// Read the radii from the environment (`SOD_STREAM_NEAR`/`LOAD`/`UNLOAD`).
    pub fn from_env() -> StreamRadii {
        let get = |k: &str, d: f32| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let r = StreamRadii {
            near: get("SOD_STREAM_NEAR", 400.0),
            load: get("SOD_STREAM_LOAD", 900.0),
            unload: get("SOD_STREAM_UNLOAD", 1400.0),
        };
        if r.unload < r.load {
            StreamRadii { unload: r.load, ..r }
        } else {
            r
        }
    }
}

/// Horizontal (X/Y) squared distance, since the world is Z-up and altitude is
/// irrelevant to how far the player is from a tree.
pub fn dist2_xy(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
}

/// Nearest horizontal distance from `player` to any instance of `model`.
pub fn nearest_distance(instances: &HashMap<String, Vec<StreamInstance>>, player: [f32; 3]) -> HashMap<String, f32> {
    let mut out = HashMap::new();
    for (model, list) in instances {
        let mut best = f32::INFINITY;
        for inst in list {
            let d = dist2_xy(inst.pos, player);
            if d < best {
                best = d;
            }
        }
        out.insert(model.clone(), best.sqrt());
    }
    out
}

/// Pick up to `want` not-yet-requested models within `load` of `player`,
/// nearest-first. Mutates `requested` to mark the chosen ones.
pub fn plan(
    pending: &[String],
    nearest: &HashMap<String, f32>,
    player: [f32; 3],
    instances: &HashMap<String, Vec<StreamInstance>>,
    load: f32,
    want: usize,
    requested: &mut HashSet<String>,
) -> Vec<String> {
    if want == 0 {
        return Vec::new();
    }
    let mut in_range: Vec<(f32, &String)> = pending
        .iter()
        .filter(|m| !requested.contains(*m))
        .filter_map(|m| {
            let model: &String = m;
            let d = match nearest.get(model) {
                Some(d) => *d,
                None => instances
                    .get(model)
                    .map(|l| {
                        l.iter()
                            .map(|i| dist2_xy(i.pos, player))
                            .fold(f32::INFINITY, f32::min)
                            .sqrt()
                    })
                    .unwrap_or(f32::INFINITY),
            };
            (d <= load).then_some((d, model))
        })
        .collect();
    in_range.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    in_range
        .into_iter()
        .take(want)
        .map(|(_, m)| {
            requested.insert(m.clone());
            m.clone()
        })
        .collect()
}

/// Whether every instance of a resident model is beyond `unload`, so its
/// entities/assets can be released.
pub fn should_unload(
    instances: &[StreamInstance],
    player: [f32; 3],
    unload: f32,
) -> bool {
    let u2 = unload * unload;
    !instances.iter().any(|i| dist2_xy(i.pos, player) <= u2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(model: &str, x: f32, y: f32) -> StreamInstance {
        StreamInstance {
            model: model.into(),
            pos: [x, y, 0.0],
            rot: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            vegetation: true,
            group: 0,
            view_dist_ratio: 1.0,
            lod_ratio: 1.0,
        }
    }

    #[test]
    fn nearest_uses_horizontal_distance() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), vec![inst("a", 3.0, 4.0), inst("a", 100.0, 0.0)]);
        m.insert("b".to_string(), vec![inst("b", 30.0, 40.0)]);
        let n = nearest_distance(&m, [0.0, 0.0, 999.0]);
        assert!((n["a"] - 5.0).abs() < 1e-4);
        assert!((n["b"] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn plan_is_nearest_first_and_marks_requested() {
        let mut instances = HashMap::new();
        instances.insert("far".to_string(), vec![inst("far", 500.0, 0.0)]);
        instances.insert("mid".to_string(), vec![inst("mid", 100.0, 0.0)]);
        instances.insert("near".to_string(), vec![inst("near", 10.0, 0.0)]);
        let pending: Vec<String> = vec!["far".into(), "mid".into(), "near".into()];
        let nearest = HashMap::new();
        let mut requested = HashSet::new();
        let got = plan(&pending, &nearest, [0.0, 0.0, 0.0], &instances, 1000.0, 2, &mut requested);
        assert_eq!(got, vec!["near".to_string(), "mid".to_string()]);
        assert!(requested.contains("near") && requested.contains("mid") && !requested.contains("far"));
        // Re-planning does not re-request.
        let got2 = plan(&pending, &nearest, [0.0, 0.0, 0.0], &instances, 1000.0, 5, &mut requested);
        assert_eq!(got2, vec!["far".to_string()]);
    }

    #[test]
    fn unload_only_when_all_instances_far() {
        let list = vec![inst("a", 0.0, 0.0), inst("a", 2000.0, 0.0)];
        assert!(!should_unload(&list, [0.0, 0.0, 0.0], 1000.0));
        let list = vec![inst("a", 1500.0, 0.0), inst("a", 2000.0, 0.0)];
        assert!(should_unload(&list, [0.0, 0.0, 0.0], 1000.0));
    }
}
