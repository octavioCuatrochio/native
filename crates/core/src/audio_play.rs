//! Engine-side sound system for State of Decay: FMOD event playback over FSB5 banks.
//!
//! The game authors audio as FMOD Designer *events* (`*.fev`) whose layers
//! reference wave-bank sub-sounds (`*.fsb`). [`crate::audio_fev`] resolves a
//! script path such as `Sounds/explosions:exp_specific:grenade` to a
//! [`ResolvedEvent`] (properties + `(fsb, subsound)` list); [`crate::audio`]
//! enumerates the banks and extracts the MPEG elementary stream. This module
//! closes the loop: it decodes subsounds to f32 PCM (MPEG through Symphonia,
//! PCM/IMA through [`crate::audio::decode_sample`]), caches them, and mixes
//! voices with FMOD-compatible 3D attenuation, equal-power stereo panning,
//! looping, fades and per-event playback limits.
//!
//! It is deliberately engine-agnostic: no Bevy, no device, no global state. The
//! app drives it with a fixed output rate and pulls interleaved stereo frames
//! from [`SoundSystem::render`].
//!
//! # Event `mode` is FMOD_MODE
//!
//! `audio_fev::FevEvent::mode` is the raw event mode word. Earlier notes called
//! it "not the FMOD_MODE bitfield"; measuring all 3,317 shipped 3D/2D events
//! shows the opposite: every observed word decomposes exactly into the
//! documented FMOD_MODE flags.
//!
//! | observed bits | FMOD flag | meaning |
//! |---|---|---|
//! | `0x08` / `0x10` | `FMOD_2D` / `FMOD_3D` | positioning |
//! | `0x00080000` | `FMOD_3D_WORLDRELATIVE` | world-space source (all events) |
//! | `0x00100000` | `FMOD_3D_INVERSEROLLOFF` | inverse rolloff (music, ui, environment, explosions) |
//! | `0x00200000` | `FMOD_3D_LINEARROLLOFF` | linear rolloff (weapons) |
//! | `0x00400000` | `FMOD_3D_LINEARSQUAREROLLOFF` | linear-squared (no shipped event) |
//! | `0x00800000` | `FMOD_3D_INVERSETAPEREDROLLOFF` | inverse tapered (none shipped) |
//! | `0x04000000` | `FMOD_3D_CUSTOMROLLOFF` | custom distance envelope (most 3D SFX) |
//! | `0x40000000` | `FMOD_3D_IGNOREGEOMETRY` | geometry ignored (zdx, some dialog) |
//!
//! Distribution over the shipped data: `0x00180008` (2D inverse) 426×,
//! `0x00180010` (3D inverse) 67×, `0x00280010` (3D linear) 18×,
//! `0x04080008` (2D custom) 1,010×, `0x04080010` (3D custom) 1,620×,
//! `0x44080010` (3D custom + ignore geometry) 178×. Custom-rolloff events carry
//! the actual attenuation in a 3D-distance parameter *envelope* that the FEV
//! reader does not decode, so this module falls back to a linear distance fade
//! for `Custom` (documented at [`attenuation`]).

use std::collections::HashMap;
use std::f32::consts::SQRT_2;
use std::io::Cursor;
use std::sync::Arc;

use crate::audio;
use crate::audio_fev::ResolvedEvent;

// ---------------------------------------------------------------------------
// FMOD_MODE flags (the event `mode` word)
// ---------------------------------------------------------------------------

/// `FMOD_LOOP_NORMAL` (`0x2`): looping sound.
pub const MODE_LOOP_NORMAL: u32 = 0x0000_0002;
/// `FMOD_LOOP_BIDI` (`0x4`): ping-pong looping sound.
pub const MODE_LOOP_BIDI: u32 = 0x0000_0004;
/// `FMOD_2D` (`0x8`).
pub const MODE_2D: u32 = 0x0000_0008;
/// `FMOD_3D` (`0x10`).
pub const MODE_3D: u32 = 0x0000_0010;
/// `FMOD_3D_WORLDRELATIVE` (`0x00080000`).
pub const MODE_3D_WORLDRELATIVE: u32 = 0x0008_0000;
/// `FMOD_3D_INVERSEROLLOFF` (`0x00100000`); the FMOD default model.
pub const MODE_3D_INVERSEROLLOFF: u32 = 0x0010_0000;
/// `FMOD_3D_LINEARROLLOFF` (`0x00200000`).
pub const MODE_3D_LINEARROLLOFF: u32 = 0x0020_0000;
/// `FMOD_3D_LINEARSQUAREROLLOFF` (`0x00400000`).
pub const MODE_3D_LINEARSQUAREROLLOFF: u32 = 0x0040_0000;
/// `FMOD_3D_INVERSETAPEREDROLLOFF` (`0x00800000`).
pub const MODE_3D_INVERSETAPEREDROLLOFF: u32 = 0x0080_0000;
/// `FMOD_3D_CUSTOMROLLOFF` (`0x04000000`). The custom curve itself lives in the
/// (undecoded) distance envelope, so playback falls back to a linear fade.
pub const MODE_3D_CUSTOMROLLOFF: u32 = 0x0400_0000;
/// `FMOD_3D_IGNOREGEOMETRY` (`0x40000000`).
pub const MODE_3D_IGNOREGEOMETRY: u32 = 0x4000_0000;

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Fully decoded audio, channel-interleaved, normalised to `-1.0..=1.0`.
#[derive(Debug, Clone)]
pub struct DecodedBuffer {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Source channel count (1, 2 or 4 in the shipped banks).
    pub channels: u16,
    /// Interleaved f32 samples, `frames * channels` long.
    pub samples: Vec<f32>,
}

impl DecodedBuffer {
    /// Number of PCM frames (samples per channel).
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    /// Build from interleaved signed 16-bit PCM.
    fn from_i16(sample_rate: u32, channels: u16, pcm: &[i16]) -> Self {
        let samples = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
        Self {
            sample_rate,
            channels,
            samples,
        }
    }

    /// Channel-averaged (mono) sample at fractional frame `frame`, linearly
    /// interpolated. Out-of-range positions read as silence.
    fn mono_at(&self, frame: f64) -> f32 {
        let ch = self.channels as usize;
        let total = self.frames();
        if ch == 0 || total == 0 {
            return 0.0;
        }
        let f0 = frame.floor();
        if f0 < 0.0 {
            return 0.0;
        }
        let f0 = f0 as usize;
        if f0 >= total {
            return 0.0;
        }
        let frac = (frame - f0 as f64) as f32;
        let avg = |idx: usize| -> f32 {
            let base = idx * ch;
            let mut sum = 0.0;
            for c in 0..ch {
                sum += self.samples[base + c];
            }
            sum / ch as f32
        };
        if f0 + 1 < total {
            avg(f0) * (1.0 - frac) + avg(f0 + 1) * frac
        } else {
            avg(f0)
        }
    }
}

/// Decode one FSB5 sub-sound to f32 PCM.
///
/// MPEG banks (the only codec in the shipped data, `audio::CODEC_MPEG`) are
/// demuxed through [`audio::extract_mpeg_stream`] and decoded with Symphonia's
/// MPEG audio codec. PCM8/PCM16/IMA ADPCM banks go through
/// [`audio::decode_sample`]. Other codecs return `None`. Never panics.
pub fn decode_fsb_subsound(bytes: &[u8], idx: usize) -> Option<DecodedBuffer> {
    if audio::is_mpeg(bytes) {
        let stream = audio::extract_mpeg_stream(bytes, idx)?;
        decode_mpeg_stream(&stream)
    } else {
        let decoded = audio::decode_sample(bytes, idx)?;
        Some(DecodedBuffer::from_i16(
            decoded.sample_rate,
            decoded.channels,
            &decoded.pcm,
        ))
    }
}

/// Decode a raw MPEG audio elementary stream (a sequence of MPEG frames).
fn decode_mpeg_stream(stream: &[u8]) -> Option<DecodedBuffer> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let source = MediaSourceStream::new(Box::new(Cursor::new(stream.to_vec())), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let mut format = probed.format;

    let track = format.default_track()?.clone();
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .ok()?;

    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                if sample_rate == 0 {
                    sample_rate = spec.rate;
                    channels = spec.channels.count() as u16;
                }
                let mut buf = SampleBuffer::<f32>::new(decoded.frames() as u64, spec);
                buf.copy_interleaved_ref(decoded);
                samples.extend_from_slice(buf.samples());
            }
            // A bad frame is skippable; anything else ends the stream.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }

    if sample_rate == 0 || channels == 0 || samples.is_empty() {
        return None;
    }
    Some(DecodedBuffer {
        sample_rate,
        channels,
        samples,
    })
}

// ---------------------------------------------------------------------------
// 3D attenuation and panning
// ---------------------------------------------------------------------------

/// FMOD distance-attenuation model selected by the event `mode` word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rolloff {
    /// `FMOD_3D_INVERSEROLLOFF` (and the FMOD default): `min / distance`.
    Inverse,
    /// `FMOD_3D_LINEARROLLOFF`: linear fade from `min` to `max`.
    Linear,
    /// `FMOD_3D_LINEARSQUAREROLLOFF`: squared linear fade.
    LinearSquared,
    /// `FMOD_3D_INVERSETAPEREDROLLOFF`: `min(inverse, linear-squared)`.
    InverseTapered,
    /// `FMOD_3D_CUSTOMROLLOFF`: the curve is stored in an undecoded FEV
    /// envelope, so [`attenuation`] uses the linear model as a fallback.
    Custom,
}

/// Select the rolloff model from an event `mode` word.
///
/// Rolloff bits are mutually exclusive in the shipped data; the explicit bits
/// are tested in order and everything else (including a bare `FMOD_3D`) uses
/// FMOD's inverse default.
pub fn rolloff_from_mode(mode: u32) -> Rolloff {
    if mode & MODE_3D_LINEARROLLOFF != 0 {
        Rolloff::Linear
    } else if mode & MODE_3D_LINEARSQUAREROLLOFF != 0 {
        Rolloff::LinearSquared
    } else if mode & MODE_3D_INVERSETAPEREDROLLOFF != 0 {
        Rolloff::InverseTapered
    } else if mode & MODE_3D_CUSTOMROLLOFF != 0 {
        Rolloff::Custom
    } else {
        Rolloff::Inverse
    }
}

/// FMOD distance attenuation in `0.0..=1.0` for `distance`, given the event's
/// `min`/`max` distance and rolloff `model`.
///
/// FMOD uses a global `rolloffscale` of 1.0 (the default), so:
///
/// * inverse — `min / distance` between `min` and `max`, held at its `max`
///   value beyond `max` (FMOD's inverse model keeps attenuating only up to the
///   max distance);
/// * linear — `1 - (d - min) / (max - min)`;
/// * linear-squared — the square of that;
/// * inverse-tapered — the quieter of inverse and linear-squared;
/// * custom — linear fallback (the authored distance envelope is not decoded).
///
/// `min <= 0` is clamped to a tiny positive value so the inverse model cannot
/// divide by zero; custom events (which ship `min = 0`) use the linear fallback
/// and are unaffected. A degenerate `max <= min` (the ambient volumes ship
/// `min = max = 1`) falls back to FMOD's default maximum distance so the source
/// is not faded to silence.
pub fn attenuation(model: Rolloff, distance: f32, min: f32, max: f32) -> f32 {
    let min = min.max(0.0);
    let d = distance.max(0.0);
    if d <= min {
        return 1.0;
    }
    // A degenerate authored range (`max <= min`, e.g. the ambient volumes ship
    // `min = max = 1`) would collapse a fade-to-zero model to zero width. The
    // inverse models already hold near full volume in that case, so only the
    // fade-to-zero models fall back to FMOD's default maximum distance (10000)
    // to keep the source audible.
    let span_max = if max > min { max } else { min + 1e-3 };
    let fade_max = if max > min { max } else { 10000.0_f32.max(min + 1.0) };
    let linear_nd = ((d.min(fade_max) - min) / (fade_max - min)).clamp(0.0, 1.0);
    match model {
        Rolloff::Inverse => {
            let m = if min > 1e-4 { min } else { 1e-4 };
            let dc = d.min(span_max);
            m / (m + (dc - m).max(0.0))
        }
        Rolloff::Linear | Rolloff::Custom => 1.0 - linear_nd,
        Rolloff::LinearSquared => (1.0 - linear_nd).powi(2),
        Rolloff::InverseTapered => {
            let m = if min > 1e-4 { min } else { 1e-4 };
            let inv = m / (m + (d.min(span_max) - m).max(0.0));
            inv.min((1.0 - linear_nd).powi(2))
        }
    }
}

/// Equal-power stereo gains for a pan in `-1.0` (hard left) `..= 1.0` (hard
/// right). Centre is `1/sqrt(2)` on both channels.
pub fn pan_gains(pan: f32) -> (f32, f32) {
    let p = pan.clamp(-1.0, 1.0);
    (((1.0 - p) * 0.5).sqrt(), ((1.0 + p) * 0.5).sqrt())
}

// ---------------------------------------------------------------------------
// Listener and vector helpers
// ---------------------------------------------------------------------------

/// The single listener (camera) used for 3D position, orientation and panning.
#[derive(Debug, Clone, Copy)]
pub struct Listener {
    pub position: [f32; 3],
    /// Unit forward vector (into the screen).
    pub forward: [f32; 3],
    /// Unit up vector.
    pub up: [f32; 3],
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            forward: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
        }
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len > 1e-6 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Listener right vector, `forward x up`. With the default forward `-Z` and up
/// `+Y` this is `+X`, matching CryEngine's left-handed world axes.
fn listener_right(l: &Listener) -> [f32; 3] {
    normalize(cross(l.forward, l.up))
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    dot(sub(a, b), sub(a, b)).sqrt()
}

// ---------------------------------------------------------------------------
// Voices and the sound system
// ---------------------------------------------------------------------------

/// Opaque handle to a playing voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoundId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoiceState {
    Playing,
    Paused,
    Stopping,
    Dead,
}

struct Voice {
    id: SoundId,
    event: String,
    owner: Option<u64>,
    position: [f32; 3],
    /// Target user volume (linear).
    volume: f32,
    /// Current faded user gain.
    gain: f32,
    fade_time: f32,
    pitch: f32,
    seq: u64,
    state: VoiceState,
    looping: bool,
    cursor: f64,
    buf: Arc<DecodedBuffer>,
    is_3d: bool,
    rolloff: Rolloff,
    min_distance: f32,
    max_distance: f32,
    priority: u32,
}

/// Immutable per-event data: decoded sub-sound variants and FMOD properties.
struct EventSource {
    is_3d: bool,
    rolloff: Rolloff,
    min_distance: f32,
    max_distance: f32,
    volume: f32,
    pitch: f32,
    priority: u32,
    max_playbacks: u32,
    looping: bool,
    buffers: Vec<Arc<DecodedBuffer>>,
    next_variant: usize,
}

/// The engine-side mixer.
///
/// Register resolved events with [`SoundSystem::load_event`], start them with
/// [`SoundSystem::play`], and pull interleaved stereo f32 frames from
/// [`SoundSystem::render`].
pub struct SoundSystem {
    output_rate: u32,
    max_voices: usize,
    sources: HashMap<String, EventSource>,
    cache: HashMap<(String, u32), Option<Arc<DecodedBuffer>>>,
    voices: Vec<Voice>,
    next_id: u64,
    next_seq: u64,
}

impl SoundSystem {
    /// Create a mixer that renders at `output_rate` Hz.
    pub fn new(output_rate: u32) -> Self {
        Self {
            output_rate: output_rate.max(1),
            max_voices: 64,
            sources: HashMap::new(),
            cache: HashMap::new(),
            voices: Vec::new(),
            next_id: 1,
            next_seq: 0,
        }
    }

    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }

    /// Global simultaneous-voice cap used for cross-event priority stealing.
    pub fn set_max_voices(&mut self, max_voices: usize) {
        self.max_voices = max_voices.max(1);
    }

    /// Number of event sources currently registered.
    pub fn event_count(&self) -> usize {
        self.sources.len()
    }

    /// True once an event path has been registered.
    pub fn has_event(&self, event_path: &str) -> bool {
        self.sources.contains_key(&normalise_event(event_path))
    }

    /// Decode and register every sub-sound of `ev`.
    ///
    /// `read` fetches the bytes for an FSB path (the app passes a `Vfs` read;
    /// tests pass an in-memory map). Decoded buffers are cached by
    /// `(lowercased fsb path, subsound)`. Returns `true` when at least one
    /// sub-sound decoded. The event is registered under `ev.path` (lowercased).
    pub fn load_event<F>(&mut self, ev: &ResolvedEvent, mut read: F) -> bool
    where
        F: FnMut(&str) -> Option<Vec<u8>>,
    {
        let mut buffers: Vec<Arc<DecodedBuffer>> = Vec::new();
        for sound in &ev.sounds {
            let key = (sound.fsb_path.to_ascii_lowercase(), sound.subsound);
            let decoded = match self.cache.get(&key) {
                Some(entry) => entry.clone(),
                None => {
                    let entry = read(&sound.fsb_path)
                        .and_then(|bytes| decode_fsb_subsound(&bytes, sound.subsound as usize))
                        .map(Arc::new);
                    self.cache.insert(key, entry.clone());
                    entry
                }
            };
            if let Some(buf) = decoded {
                buffers.push(buf);
            }
        }

        let has_sound = !buffers.is_empty();
        let source = EventSource {
            is_3d: ev.mode & MODE_3D != 0,
            rolloff: rolloff_from_mode(ev.mode),
            min_distance: ev.min_distance,
            max_distance: ev.max_distance,
            volume: ev.volume,
            // FEV pitch is a signed offset from unity (`0.0` = unchanged, range
            // -0.375..0.30 on the shipped data), not a bare multiplier.
            pitch: ev.pitch,
            priority: ev.priority,
            max_playbacks: ev.max_playbacks.max(1),
            looping: ev.mode & (MODE_LOOP_NORMAL | MODE_LOOP_BIDI) != 0,
            buffers,
            next_variant: 0,
        };
        self.sources.insert(normalise_event(&ev.path), source);
        has_sound
    }

    /// Start an instance of `event_path` at `position`.
    ///
    /// Returns a [`SoundId`]. If the event is unknown, has no decoded sound, or
    /// is rejected by a limit with no eligible victim, the id is returned but no
    /// voice is created (subsequent control calls on it are no-ops). When a
    /// voice is stolen the victim's [`SoundId`] is reused.
    pub fn play(&mut self, event_path: &str, position: [f32; 3], owner: Option<u64>) -> SoundId {
        let id = SoundId(self.next_id);
        self.next_id += 1;
        let key = normalise_event(event_path);

        // Snapshot the source (kept small; the Arc keeps the PCM alive).
        let (
            is_3d,
            rolloff,
            min_distance,
            max_distance,
            volume,
            pitch,
            priority,
            max_playbacks,
            looping,
            variant,
            buf,
            nbuf,
        ) = {
            let src = match self.sources.get(&key) {
                Some(s) => s,
                None => return id,
            };
            if src.buffers.is_empty() {
                return id;
            }
            let variant = src.next_variant % src.buffers.len();
            (
                src.is_3d,
                src.rolloff,
                src.min_distance,
                src.max_distance,
                src.volume,
                src.pitch,
                src.priority,
                src.max_playbacks,
                src.looping,
                variant,
                src.buffers[variant].clone(),
                src.buffers.len(),
            )
        };
        if let Some(src) = self.sources.get_mut(&key) {
            src.next_variant = (variant + 1) % nbuf;
        }

        // Per-event max playbacks: steal the least important instance (largest
        // priority number, oldest on a tie) so the event never exceeds its cap.
        let same_event = self
            .voices
            .iter()
            .filter(|v| v.event == key && v.state != VoiceState::Dead)
            .count();
        if same_event >= max_playbacks as usize {
            let victim =
                least_important(&self.voices, |v| v.event == key).expect("same_event is non-empty");
            self.restart_voice(
                victim,
                id,
                &key,
                position,
                owner,
                volume,
                pitch,
                priority,
                is_3d,
                rolloff,
                min_distance,
                max_distance,
                looping,
                buf,
            );
            return id;
        }

        // Global voice cap: steal the least important voice overall if the new
        // instance is at least as important (lower priority number wins).
        let alive = self
            .voices
            .iter()
            .filter(|v| v.state != VoiceState::Dead)
            .count();
        if alive >= self.max_voices {
            let victim =
                least_important(&self.voices, |_| true).expect("max_voices >= 1 and alive >= 1");
            if priority <= self.voices[victim].priority {
                self.restart_voice(
                    victim,
                    id,
                    &key,
                    position,
                    owner,
                    volume,
                    pitch,
                    priority,
                    is_3d,
                    rolloff,
                    min_distance,
                    max_distance,
                    looping,
                    buf,
                );
                return id;
            }
            return id; // too unimportant to steal
        }

        self.next_seq += 1;
        let seq = self.next_seq;
        self.voices.push(Voice {
            id,
            event: key,
            owner,
            position,
            volume,
            gain: volume,
            fade_time: 0.0,
            pitch,
            seq,
            state: VoiceState::Playing,
            looping,
            cursor: 0.0,
            buf,
            is_3d,
            rolloff,
            min_distance,
            max_distance,
            priority,
        });
        id
    }

    #[allow(clippy::too_many_arguments)]
    fn restart_voice(
        &mut self,
        idx: usize,
        id: SoundId,
        event: &str,
        position: [f32; 3],
        owner: Option<u64>,
        volume: f32,
        pitch: f32,
        priority: u32,
        is_3d: bool,
        rolloff: Rolloff,
        min_distance: f32,
        max_distance: f32,
        looping: bool,
        buf: Arc<DecodedBuffer>,
    ) {
        self.next_seq += 1;
        let seq = self.next_seq;
        let v = &mut self.voices[idx];
        let fade_time = v.fade_time;
        *v = Voice {
            id,
            event: event.to_string(),
            owner,
            position,
            volume,
            gain: 0.0,
            fade_time,
            pitch,
            seq,
            state: VoiceState::Playing,
            looping,
            cursor: 0.0,
            buf,
            is_3d,
            rolloff,
            min_distance,
            max_distance,
            priority,
        };
    }

    /// Stop a voice, fading out over its fade time.
    pub fn stop(&mut self, id: SoundId) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
            v.state = VoiceState::Stopping;
        }
    }

    /// Stop every live instance of an event.
    pub fn stop_event(&mut self, event_path: &str) {
        let key = normalise_event(event_path);
        for v in self.voices.iter_mut().filter(|v| v.event == key) {
            v.state = VoiceState::Stopping;
        }
    }

    /// Pause or resume a voice (position is preserved).
    pub fn pause(&mut self, id: SoundId, paused: bool) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
            if matches!(v.state, VoiceState::Playing | VoiceState::Paused) {
                v.state = if paused {
                    VoiceState::Paused
                } else {
                    VoiceState::Playing
                };
            }
        }
    }

    /// Set a voice's user volume (linear). The change fades over the voice's
    /// current fade time.
    pub fn set_volume(&mut self, id: SoundId, volume: f32) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
            v.volume = volume.max(0.0);
        }
    }

    /// Move a 3D voice.
    pub fn set_position(&mut self, id: SoundId, position: [f32; 3]) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
            v.position = position;
        }
    }

    /// Set the fade time (seconds) used for subsequent volume/stop transitions.
    pub fn set_fade_time(&mut self, id: SoundId, seconds: f32) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
            v.fade_time = seconds.max(0.0);
        }
    }

    /// Enable or disable looping for a voice.
    pub fn set_looping(&mut self, id: SoundId, looping: bool) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
            v.looping = looping;
        }
    }

    /// Live instance count for one event.
    pub fn active_count(&self, event_path: &str) -> usize {
        let key = normalise_event(event_path);
        self.voices
            .iter()
            .filter(|v| v.event == key && v.state != VoiceState::Dead)
            .count()
    }

    /// Live instance count across all events.
    pub fn voice_count(&self) -> usize {
        self.voices
            .iter()
            .filter(|v| v.state != VoiceState::Dead)
            .count()
    }

    /// The owner token a live voice was started with, if any.
    /// Whether the sound is still an active voice (not stopped or finished).
    pub fn is_playing(&self, id: SoundId) -> bool {
        self.voices.iter().any(|v| v.id == id)
    }

    pub fn voice_owner(&self, id: SoundId) -> Option<u64> {
        self.voices
            .iter()
            .find(|v| v.id == id)
            .and_then(|v| v.owner)
    }

    /// Mix `frames` stereo frames at the output rate and return them
    /// interleaved `[L, R, L, R, ...]`. The listener drives 3D attenuation and
    /// equal-power stereo panning.
    pub fn render(&mut self, frames: usize, listener: &Listener) -> Vec<f32> {
        let mut out = vec![0.0f32; frames * 2];
        if frames == 0 {
            return out;
        }
        let dt = 1.0 / self.output_rate as f32;
        let right = listener_right(listener);

        for v in self.voices.iter_mut() {
            if v.state == VoiceState::Dead {
                continue;
            }
            let total = v.buf.frames();
            if total == 0 {
                v.state = VoiceState::Dead;
                continue;
            }

            let pitch = 1.0 + v.pitch;
            let step = if pitch > 0.0 {
                (v.buf.sample_rate as f64 / self.output_rate as f64) * pitch as f64
            } else {
                1.0
            };

            let (atten, gain_l, gain_r) = if v.is_3d {
                let d = distance(v.position, listener.position);
                let atten = attenuation(v.rolloff, d, v.min_distance, v.max_distance);
                let dir = normalize(sub(v.position, listener.position));
                let (l, r) = pan_gains(dot(dir, right));
                (atten, l, r)
            } else {
                (1.0, 1.0 / SQRT_2, 1.0 / SQRT_2)
            };

            for i in 0..frames {
                if v.state == VoiceState::Paused {
                    break;
                }
                let mut cursor = v.cursor;
                if cursor >= total as f64 {
                    if v.looping {
                        cursor = cursor % total as f64;
                    } else {
                        v.state = VoiceState::Dead;
                        break;
                    }
                }
                let sample = v.buf.mono_at(cursor);
                let target = if v.state == VoiceState::Stopping {
                    0.0
                } else {
                    v.volume
                };
                let gain = advance_gain(&mut v.gain, target, dt, v.fade_time);
                if v.state == VoiceState::Stopping && gain <= 1e-4 {
                    v.state = VoiceState::Dead;
                }
                let amp = gain * atten;
                out[i * 2] += sample * amp * gain_l;
                out[i * 2 + 1] += sample * amp * gain_r;
                v.cursor = cursor + step;
            }
        }

        self.voices.retain(|v| v.state != VoiceState::Dead);
        out
    }

    /// Render `frames` stereo frames and convert to interleaved signed 16-bit,
    /// clamped to full scale. Convenience for [`crate::audio::to_wav`].
    pub fn render_i16(&mut self, frames: usize, listener: &Listener) -> Vec<i16> {
        self.render(frames, listener)
            .into_iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }
}

fn normalise_event(path: &str) -> String {
    path.trim().to_ascii_lowercase()
}

/// Index of the least important live voice matching `filter`: the largest
/// priority number wins (FMOD priority 0 is the most important), with the
/// oldest voice breaking ties.
fn least_important(voices: &[Voice], filter: impl Fn(&Voice) -> bool) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, v) in voices.iter().enumerate() {
        if v.state == VoiceState::Dead || !filter(v) {
            continue;
        }
        match best {
            None => best = Some(i),
            Some(b) => {
                let better = v.priority > voices[b].priority
                    || (v.priority == voices[b].priority && v.seq < voices[b].seq);
                if better {
                    best = Some(i);
                }
            }
        }
    }
    best
}

/// Move `gain` toward `target` at a rate that traverses the full range in
/// `fade_time` seconds. `fade_time <= 0` snaps.
fn advance_gain(gain: &mut f32, target: f32, dt: f32, fade_time: f32) -> f32 {
    if fade_time <= 1e-6 {
        *gain = target;
        return target;
    }
    let step = dt / fade_time;
    let diff = target - *gain;
    if diff.abs() <= step {
        *gain = target;
    } else {
        *gain += step * diff.signum();
    }
    *gain
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_fev::{FevDb, ResolvedSound};
    use std::path::{Path, PathBuf};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // --- synthetic FSB helpers -------------------------------------------

    /// Build a single-sub-sound FSB5 bank with constant PCM16 samples.
    fn pcm16_bank(frames: u32, channels: u16, value: i16) -> Vec<u8> {
        let ch_code: u64 = match channels {
            1 => 0,
            2 => 1,
            _ => 0,
        };
        let raw: u64 = (8u64 << 1) | (ch_code << 5) | ((frames as u64) << 34); // idx 8 = 44100
        let mut payload: Vec<u8> = Vec::new();
        for _ in 0..(frames as usize * channels as usize) {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"FSB5");
        out.extend_from_slice(&1u32.to_le_bytes()); // version
        out.extend_from_slice(&1u32.to_le_bytes()); // numSamples
        out.extend_from_slice(&8u32.to_le_bytes()); // sampleHeadersSize
        out.extend_from_slice(&0u32.to_le_bytes()); // nameTableSize
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes()); // PCM16
        out.extend_from_slice(&[0u8; 8]);
        out.extend_from_slice(&[0xAAu8; 16]);
        out.extend_from_slice(&[0u8; 8]);
        out.extend_from_slice(&raw.to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    fn resolved_event(
        name: &str,
        mode: u32,
        priority: u32,
        max_playbacks: u32,
        min_distance: f32,
        max_distance: f32,
    ) -> ResolvedEvent {
        ResolvedEvent {
            project: "test".to_string(),
            path: format!("/test/{name}"),
            name: name.to_string(),
            group: vec!["test".to_string()],
            categories: Vec::new(),
            volume: 1.0,
            pitch: 0.0,
            priority,
            max_playbacks,
            mode,
            min_distance,
            max_distance,
            sounds: vec![ResolvedSound {
                fsb_path: "test/bank.fsb".to_string(),
                subsound: 0,
                stream: "bank.wav".to_string(),
            }],
        }
    }

    // --- mode / attenuation ---------------------------------------------

    #[test]
    fn mode_bits_select_fmod_rolloff() {
        // Real shipped mode words (see module docs).
        assert_eq!(rolloff_from_mode(0x0018_0008), Rolloff::Inverse);
        assert_eq!(rolloff_from_mode(0x0018_0010), Rolloff::Inverse);
        assert_eq!(rolloff_from_mode(0x0028_0010), Rolloff::Linear);
        assert_eq!(rolloff_from_mode(0x0408_0008), Rolloff::Custom);
        assert_eq!(rolloff_from_mode(0x0408_0010), Rolloff::Custom);
        assert_eq!(rolloff_from_mode(0x4408_0010), Rolloff::Custom);
        // Explicit unshipped flags still map.
        assert_eq!(rolloff_from_mode(0x0040_0010), Rolloff::LinearSquared);
        assert_eq!(rolloff_from_mode(0x0080_0010), Rolloff::InverseTapered);
        // Bare 3D defaults to inverse.
        assert_eq!(rolloff_from_mode(MODE_3D), Rolloff::Inverse);
    }

    #[test]
    fn attenuation_curves_match_fmod() {
        // Inverse (music/ui default): unattenuated through min, min/d beyond,
        // held at the max-distance value further out.
        assert!(approx(
            attenuation(Rolloff::Inverse, 0.0, 1.0, 10000.0),
            1.0,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Inverse, 1.0, 1.0, 10000.0),
            1.0,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Inverse, 2.0, 1.0, 10000.0),
            0.5,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Inverse, 100.0, 1.0, 10000.0),
            0.01,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Inverse, 20000.0, 1.0, 10000.0),
            0.0001,
            1e-9
        ));

        // Linear (weapons): 1 at min, 0 at max.
        assert!(approx(
            attenuation(Rolloff::Linear, 3.0, 3.0, 13.0),
            1.0,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Linear, 8.0, 3.0, 13.0),
            0.5,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Linear, 13.0, 3.0, 13.0),
            0.0,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Linear, 50.0, 3.0, 13.0),
            0.0,
            1e-6
        ));

        // Linear-squared halves the linear gain at the midpoint.
        assert!(approx(
            attenuation(Rolloff::LinearSquared, 8.0, 3.0, 13.0),
            0.25,
            1e-6
        ));

        // Inverse tapered takes the quieter of inverse and linear-squared.
        let inv = attenuation(Rolloff::Inverse, 8.0, 1.0, 10.0);
        let sq = attenuation(Rolloff::LinearSquared, 8.0, 1.0, 10.0);
        assert!(approx(
            attenuation(Rolloff::InverseTapered, 8.0, 1.0, 10.0),
            inv.min(sq),
            1e-6
        ));

        // Custom falls back to a linear fade even with min == 0.
        assert!(approx(
            attenuation(Rolloff::Custom, 0.0, 0.0, 30.0),
            1.0,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Custom, 15.0, 0.0, 30.0),
            0.5,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Custom, 30.0, 0.0, 30.0),
            0.0,
            1e-6
        ));
        assert!(approx(
            attenuation(Rolloff::Custom, 60.0, 0.0, 30.0),
            0.0,
            1e-6
        ));

        // Degenerate max is guarded.
        assert!(attenuation(Rolloff::Linear, 5.0, 1.0, 1.0).is_finite());
    }

    #[test]
    fn equal_power_pan_is_constant_power() {
        for pan in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            let (l, r) = pan_gains(pan);
            assert!(approx(l * l + r * r, 1.0, 1e-5), "power at pan {pan}");
        }
        let (l, r) = pan_gains(0.0);
        assert!(approx(l, 1.0 / SQRT_2, 1e-6));
        assert!(approx(r, 1.0 / SQRT_2, 1e-6));
        let (l, r) = pan_gains(1.0);
        assert!(approx(l, 0.0, 1e-6));
        assert!(approx(r, 1.0, 1e-6));
    }

    // --- voice management ------------------------------------------------

    #[test]
    fn per_event_max_playbacks_steals_lowest_priority() {
        let bank = pcm16_bank(48000, 1, 4000);
        let ev = resolved_event("shot", 0x0408_0010, 128, 2, 0.0, 30.0);
        let mut sys = SoundSystem::new(48000);
        assert!(sys.load_event(&ev, |_| Some(bank.clone())));

        let a = sys.play(&ev.path, [0.0; 3], None);
        let b = sys.play(&ev.path, [0.0; 3], None);
        let c = sys.play(&ev.path, [0.0; 3], None);
        assert_eq!(sys.active_count(&ev.path), 2, "cap enforced");
        // The two oldest survive; the third retrigger steals one of them, so
        // exactly one of the first two ids is still live.
        let live_first = [a, b]
            .iter()
            .filter(|&&id| sys.voices.iter().any(|v| v.id == id))
            .count();
        assert_eq!(live_first, 1);
        assert!(sys.voices.iter().any(|v| v.id == c));
    }

    #[test]
    fn global_priority_stealing_prefers_less_important_voice() {
        let bank = pcm16_bank(48000, 1, 4000);
        let important = resolved_event("important", 0x0408_0010, 10, 100, 0.0, 30.0);
        let unimportant = resolved_event("unimportant", 0x0408_0010, 200, 100, 0.0, 30.0);
        let middle = resolved_event("middle", 0x0408_0010, 100, 100, 0.0, 30.0);

        let mut sys = SoundSystem::new(48000);
        sys.set_max_voices(2);
        let rd = |_p: &str| Some(bank.clone());
        sys.load_event(&important, rd);
        sys.load_event(&unimportant, rd);
        sys.load_event(&middle, rd);

        sys.play(&important.path, [0.0; 3], None); // voice 1 (prio 10)
        sys.play(&unimportant.path, [0.0; 3], None); // voice 2 (prio 200)
        assert_eq!(sys.voice_count(), 2);

        // A more important event steals the unimportant voice.
        sys.play(&middle.path, [0.0; 3], None);
        assert_eq!(sys.voice_count(), 2);
        assert_eq!(sys.active_count(&middle.path), 1);
        assert_eq!(sys.active_count(&unimportant.path), 0);
        assert_eq!(sys.active_count(&important.path), 1);
    }

    #[test]
    fn unknown_event_returns_unregistered_id() {
        let mut sys = SoundSystem::new(48000);
        let id = sys.play("/nope/nothing", [0.0; 3], None);
        assert_eq!(sys.voice_count(), 0);
        sys.stop(id); // must be a no-op, not a panic
        sys.set_volume(id, 0.5);
        assert_eq!(sys.voice_count(), 0);
    }

    #[test]
    fn stop_fades_out_and_frees_the_voice() {
        let bank = pcm16_bank(48000, 1, 4000);
        let ev = resolved_event("tone", 0x0018_0010, 128, 4, 1.0, 100.0);
        let mut sys = SoundSystem::new(48000);
        sys.load_event(&ev, |_| Some(bank.clone()));
        let id = sys.play(&ev.path, [0.0; 3], None);
        sys.set_fade_time(id, 0.05);
        sys.stop(id);
        // 0.05 s at 48 kHz = 2400 frames; render a little longer.
        sys.render(3000, &Listener::default());
        assert_eq!(sys.voice_count(), 0, "voice removed after fade");
    }

    #[test]
    fn pause_freezes_position_and_output() {
        let bank = pcm16_bank(48000, 1, 4000);
        let ev = resolved_event("tone", 0x0018_0010, 128, 4, 1.0, 100.0);
        let mut sys = SoundSystem::new(48000);
        sys.load_event(&ev, |_| Some(bank.clone()));
        let id = sys.play(&ev.path, [0.0; 3], None);
        let _ = sys.render(100, &Listener::default());
        let cursor = sys.voices.iter().find(|v| v.id == id).unwrap().cursor;
        sys.pause(id, true);
        let out = sys.render(100, &Listener::default());
        assert!(out.iter().all(|&s| s == 0.0), "paused voice is silent");
        let cursor2 = sys.voices.iter().find(|v| v.id == id).unwrap().cursor;
        assert_eq!(cursor, cursor2, "paused cursor does not advance");
    }

    // --- real-data helpers ----------------------------------------------

    /// `<repo>/extracted-game/sounds` (contains `sounds/**/*.fsb`).
    fn extracted_sounds_dir() -> Option<PathBuf> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game/sounds");
        if base.join("sounds").is_dir() {
            Some(base)
        } else {
            None
        }
    }

    fn walk_ext(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                walk_ext(&p, ext, out);
            } else if p.extension().map(|e| e == ext).unwrap_or(false) {
                out.push(p);
            }
        }
    }

    #[test]
    fn decodes_real_mpeg_subsound_with_expected_duration() {
        let Some(base) = extracted_sounds_dir() else {
            eprintln!("skip: extracted-game/sounds absent");
            return;
        };
        let mut banks = Vec::new();
        walk_ext(&base, "fsb", &mut banks);
        assert!(!banks.is_empty(), "expected shipped FSB banks");

        let mut checked = 0usize;
        for bank_path in &banks {
            let Ok(bytes) = std::fs::read(bank_path) else {
                continue;
            };
            if !audio::is_mpeg(&bytes) {
                continue;
            }
            let Some(fsb) = audio::parse_fsb(&bytes) else {
                continue;
            };
            for (idx, sample) in fsb.samples.iter().enumerate() {
                if sample.channels > 2 {
                    continue; // ambisonic beds are not MPEG stereo
                }
                let Some(decoded) = decode_fsb_subsound(&bytes, idx) else {
                    continue;
                };
                assert_eq!(
                    decoded.sample_rate,
                    sample.sample_rate,
                    "sample rate {}",
                    bank_path.display()
                );
                assert_eq!(
                    decoded.channels,
                    sample.channels,
                    "channels {}",
                    bank_path.display()
                );
                let expected = sample.num_samples as i64;
                let got = decoded.frames() as i64;
                let tol = (expected / 20).max(3000); // 5% or a few MPEG frames
                assert!(
                    (expected - got).abs() <= tol,
                    "duration mismatch in {} sub {}: header {} frames, decoded {}",
                    bank_path.display(),
                    idx,
                    expected,
                    got
                );
                checked += 1;
                if checked >= 8 {
                    break;
                }
            }
            if checked >= 8 {
                break;
            }
        }
        assert!(checked > 0, "no MPEG sub-sound decoded");
        eprintln!("decoded {checked} real MPEG sub-sounds with matching duration");
    }

    /// Resolve a real script event and render one second to a WAV, reporting RMS.
    #[test]
    fn renders_script_referenced_event_to_wav() {
        let Some(base) = extracted_sounds_dir() else {
            eprintln!("skip: extracted-game/sounds absent");
            return;
        };

        // Build a FEV database from the extracted projects.
        let mut db = FevDb::new();
        let mut fevs = Vec::new();
        walk_ext(&base, "fev", &mut fevs);
        for fev in &fevs {
            let Ok(rel) = fev.strip_prefix(&base) else {
                continue;
            };
            let Ok(bytes) = std::fs::read(fev) else {
                continue;
            };
            let vfs_path = rel.to_string_lossy().replace('\\', "/");
            db.add(&vfs_path, &bytes);
        }
        assert!(db.event_count() > 1000, "expected shipped FEV events");

        let ev = db
            .resolve("Sounds/explosions:exp_specific:grenade")
            .expect("grenade event resolves");
        assert!(!ev.sounds.is_empty());

        let mut sys = SoundSystem::new(48000);
        let loaded = sys.load_event(&ev, |p| std::fs::read(base.join(p)).ok());
        assert!(loaded, "grenade FSB sub-sounds decoded");

        // Play at the listener so attenuation is exactly 1 regardless of model.
        let listener = Listener::default();
        sys.play(&ev.path, listener.position, None);
        let mix = sys.render(48000, &listener);
        assert_eq!(mix.len(), 96000);

        let sum_sq: f64 = mix.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let rms = (sum_sq / mix.len() as f64).sqrt();
        assert!(rms > 1e-5, "rendered mix is silent (rms {rms})");

        let pcm: Vec<i16> = sys_convert(&mix);
        let da = audio::DecodedAudio {
            sample_rate: 48000,
            channels: 2,
            bits: 16,
            pcm,
        };
        let wav = audio::to_wav(&da);
        std::fs::create_dir_all("/tmp/opencode-c1").ok();
        let out = "/tmp/opencode-c1/grenade_mix.wav";
        std::fs::write(out, &wav).expect("write wav");
        eprintln!(
            "grenade 1 s mix: rms={rms:.6}, {} bytes -> {out}",
            wav.len()
        );
    }

    fn sys_convert(mix: &[f32]) -> Vec<i16> {
        mix.iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }
}
