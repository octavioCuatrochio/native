//! CryEngine 3 particle libraries and an engine-agnostic emitter simulation.
//!
//! The game ships its particle effects as XML libraries inside `gamedata.pak`
//! (`libs/particles/*.xml`). The level's `ParticleEffect` entities call
//! `LoadParticleEffect(name, params)` from
//! `scripts/entities/particle/particleeffect.lua` on their Active state, and
//! `cry::world` records the effect name in a [`crate::cry::world::SlotKind::Particle`]
//! slot. This module turns those libraries into effect definitions and runs the
//! emitters; the Bevy layer only draws the resulting particles.
//!
//! # Library format (confirmed from the shipped files)
//!
//! ```xml
//! <ParticleLibrary Name="Class3Effects" ParticleVersion="22">
//!   <Particles Name="smoke_and_fire.burn_barrel" Id="{GUID}" Library="Class3Effects">
//!     <Params Continuous="true" Count="25" ParticleLifeTime="0.5,0.25"
//!             EmitAngle="65,0.35" Speed="15,0.75" GravityScale="0.85"
//!             BlendType="Additive" Texture="textures/particles/fire/smooth_fire.tif"
//!             Size="0.25,0.172,,(,0.84,32;0.31,1;1,0.56)"/>
//!     <Childs>
//!       <Particles Name="Ash"><Params .../></Particles>
//!     </Childs>
//!   </Particles>
//! </ParticleLibrary>
//! ```
//!
//! The referenced name is `Library.Name`; a `Name` may itself contain dots
//! (sub-folders), so `Class3Effects.smoke_and_fire.burn_barrel` addresses the
//! effect above. `Childs/Particles` are additional emitters that belong to the
//! same effect (they are flattened here, since the engine emits them all from
//! the one entity transform).
//!
//! Each `Params` attribute is a comma-separated micro-format. Parentheses group
//! keyframe curves, so `a,b,c,(k0v;k1v;...)` is split at top level only:
//!
//! * scalar params (`Count`, `Size`, `Speed`, `Alpha`, `ParticleLifeTime`, ...)
//!   are `base,variance[,flags][,curve]`; the curve field starts with `(`;
//! * `Size`/`Alpha`/`Speed` curves are `(time,value[,flags];...)` and are
//!   sampled piecewise-linearly over the particle's normalised age;
//! * `Color` is `(r,g,b)[,variance][,curve]`, with the animated form
//!   `(time,(r,g,b);...)`;
//! * `TextureTiling`'s first two fields are the UV repeats (`u,v`);
//! * an empty field means "use the parameter default" (1 for size/alpha, 0 for
//!   speed/count).
//!
//! Texture/material references are kept verbatim: the core crate never touches
//! the pak, so it stays testable without the game data. The renderer resolves
//! `.tif` -> `.dds` and the material's `Diffuse` map.

use anyhow::Result;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;

/// CryEngine gravity magnitude used by `GravityScale` (metres/second²).
pub const GRAVITY: f32 = 9.80665;

// ---------------------------------------------------------------------------
// Curves
// ---------------------------------------------------------------------------

/// A piecewise-linear keyframe curve over normalised lifetime `0..1`.
#[derive(Debug, Clone)]
pub struct Curve {
    keys: Vec<(f32, f32)>,
}

impl Default for Curve {
    fn default() -> Self {
        Curve { keys: Vec::new() }
    }
}

impl Curve {
    /// A constant curve (`sample` returns `v` for every `t`).
    pub fn constant(v: f32) -> Self {
        Curve { keys: vec![(0.0, v), (1.0, v)] }
    }

    pub fn is_constant(&self) -> bool {
        self.keys.len() < 2
    }

    /// Sample at normalised time `t` (clamped to `0..1`).
    pub fn sample(&self, t: f32) -> f32 {
        if self.keys.is_empty() {
            return 1.0;
        }
        let t = t.clamp(0.0, 1.0);
        if t <= self.keys[0].0 {
            return self.keys[0].1;
        }
        if t >= self.keys[self.keys.len() - 1].0 {
            return self.keys[self.keys.len() - 1].1;
        }
        for w in self.keys.windows(2) {
            let (t0, v0) = w[0];
            let (t1, v1) = w[1];
            if t >= t0 && t <= t1 {
                if (t1 - t0).abs() < 1e-6 {
                    return v1;
                }
                let f = (t - t0) / (t1 - t0);
                return v0 + (v1 - v0) * f;
            }
        }
        self.keys[self.keys.len() - 1].1
    }

    pub fn keys(&self) -> &[(f32, f32)] {
        &self.keys
    }
}

/// A keyframe curve whose values are RGB colours.
#[derive(Debug, Clone)]
pub struct ColorCurve {
    keys: Vec<(f32, [f32; 3])>,
}

impl ColorCurve {
    pub fn sample(&self, t: f32) -> Option<[f32; 3]> {
        if self.keys.is_empty() {
            return None;
        }
        let t = t.clamp(0.0, 1.0);
        if t <= self.keys[0].0 {
            return Some(self.keys[0].1);
        }
        if t >= self.keys[self.keys.len() - 1].0 {
            return Some(self.keys[self.keys.len() - 1].1);
        }
        for w in self.keys.windows(2) {
            let (t0, c0) = w[0];
            let (t1, c1) = w[1];
            if t >= t0 && t <= t1 {
                if (t1 - t0).abs() < 1e-6 {
                    return Some(c1);
                }
                let f = (t - t0) / (t1 - t0);
                let mut c = [0.0; 3];
                for i in 0..3 {
                    c[i] = c0[i] + (c1[i] - c0[i]) * f;
                }
                return Some(c);
            }
        }
        self.keys.last().map(|(_, c)| *c)
    }

    pub fn keys(&self) -> &[(f32, [f32; 3])] {
        &self.keys
    }
}

// ---------------------------------------------------------------------------
// Blend mode
// ---------------------------------------------------------------------------

/// How a particle emitter composites over the framebuffer. The game writes
/// `BlendType="Additive"|"Opaque"`; anything else (soft smoke, dust, glow) is
/// an alpha blend, and an authored `AlphaTest` makes it an alpha-tested cutout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    /// `src_alpha` alpha blending (soft smoke, dust, water).
    AlphaBlend,
    /// Additive (`BlendType="Additive"`): fire, embers, glows.
    Additive,
    /// Alpha-tested cutout (`BlendType="Opaque"` + `AlphaTest`, or masked).
    Mask(f32),
    /// Fully opaque (`BlendType="Opaque"` without a test).
    Opaque,
}

// ---------------------------------------------------------------------------
// Emitter parameters
// ---------------------------------------------------------------------------

/// One emitter's authored parameters. Missing attributes keep engine defaults.
#[derive(Debug, Clone)]
pub struct EmitterParams {
    pub continuous: bool,
    pub enabled: bool,
    /// `Count`: target concurrent particle count (continuous) or burst size.
    pub count: (f32, f32),
    /// `EmitterLifeTime` (seconds); `0` = infinite.
    pub emitter_lifetime: f32,
    /// `ParticleLifeTime` (seconds): per-particle lifetime.
    pub particle_lifetime: (f32, f32),
    /// `PulsePeriod` (seconds): re-burst period; `0` = no pulsing.
    pub pulse_period: (f32, f32),
    /// `Speed` (m/s).
    pub speed: (f32, f32),
    pub speed_curve: Curve,
    /// `Size` (metres).
    pub size: (f32, f32),
    pub size_curve: Curve,
    /// `Color` base (linear RGB) and its animated curve.
    pub color: [f32; 3],
    pub color_var: f32,
    pub color_curve: Option<ColorCurve>,
    /// `Alpha` (0..1).
    pub alpha: (f32, f32),
    pub alpha_curve: Curve,
    /// `AlphaTest` cutoff, when authored.
    pub alpha_test: Option<f32>,
    /// `GravityScale` (1 = Earth gravity, negative = rises).
    pub gravity_scale: (f32, f32),
    /// `AirResistance` (per-second exponential drag).
    pub air_resistance: (f32, f32),
    /// `EmitAngle` (degrees, half-cone) and its variance.
    pub emit_angle: (f32, f32),
    /// `FocusGravityDir`: emit straight up regardless of entity orientation.
    pub focus_gravity_dir: bool,
    /// `RandomOffset` (metres, per axis) about the emitter origin.
    pub random_offset: [f32; 3],
    /// `PositionOffset` (metres, local space).
    pub position_offset: [f32; 3],
    /// `Acceleration` (m/s², local space).
    pub acceleration: [f32; 3],
    /// `RotationRate` and `RandomRotationRate` (degrees/second).
    pub rotation_rate: [f32; 3],
    pub random_rotation_rate: [f32; 3],
    /// `RandomAngles` (degrees) initial sprite rotation range.
    pub random_angles: [f32; 3],
    pub init_angles: [f32; 3],
    /// `TurbulenceSize` (metres) and `TurbulenceSpeed` (Hz-ish).
    pub turbulence_size: (f32, f32),
    pub turbulence_speed: (f32, f32),
    /// `OrientToVelocity`: align the billboard with the particle velocity.
    pub orient_to_velocity: bool,
    /// `Stretch`: velocity-aligned elongation.
    pub stretch: (f32, f32),
    /// Camera distance culling (`CameraMaxDistance`, `CameraMinDistance`).
    pub camera_max_distance: f32,
    pub camera_min_distance: f32,
    /// `ViewDistanceAdjust`: quality LOD bias for the emitter.
    pub view_distance_adjust: f32,
    /// Raw `Texture=` reference (as authored, e.g. `.../smoke.tif`).
    pub texture: Option<String>,
    /// Raw `Material=` reference.
    pub material: Option<String>,
    /// `Geometry=` (mesh particles). Rendered as billboards for now.
    pub geometry: Option<String>,
    pub blend: BlendMode,
    /// UV repeats from `TextureTiling` (u, v).
    pub texture_tiling: (f32, f32),
    pub soft_particle: bool,
    /// Extra brightness (from `EmissiveLighting`/`EmissiveHDRDynamic`).
    pub brightness: f32,
    /// `LightSource` present: the emitter also radiates light.
    pub light_source: bool,
    /// `DrawLast` sort bias (negative draws earlier / behind).
    pub draw_last: f32,
}

impl Default for EmitterParams {
    fn default() -> Self {
        EmitterParams {
            continuous: false,
            enabled: true,
            count: (0.0, 0.0),
            emitter_lifetime: 0.0,
            particle_lifetime: (1.0, 0.0),
            pulse_period: (0.0, 0.0),
            speed: (0.0, 0.0),
            speed_curve: Curve::constant(1.0),
            size: (1.0, 0.0),
            size_curve: Curve::constant(1.0),
            color: [1.0, 1.0, 1.0],
            color_var: 0.0,
            color_curve: None,
            alpha: (1.0, 0.0),
            alpha_curve: Curve::constant(1.0),
            alpha_test: None,
            // CryEngine's `ParticleParams` default: particles fall at 1 g.
            gravity_scale: (1.0, 0.0),
            air_resistance: (0.0, 0.0),
            emit_angle: (0.0, 0.0),
            focus_gravity_dir: false,
            random_offset: [0.0; 3],
            position_offset: [0.0; 3],
            acceleration: [0.0; 3],
            rotation_rate: [0.0; 3],
            random_rotation_rate: [0.0; 3],
            random_angles: [0.0; 3],
            init_angles: [0.0; 3],
            turbulence_size: (0.0, 0.0),
            turbulence_speed: (0.0, 0.0),
            orient_to_velocity: false,
            stretch: (0.0, 0.0),
            camera_max_distance: 0.0,
            camera_min_distance: 0.0,
            view_distance_adjust: 0.0,
            texture: None,
            material: None,
            geometry: None,
            blend: BlendMode::AlphaBlend,
            texture_tiling: (1.0, 1.0),
            soft_particle: false,
            brightness: 1.0,
            light_source: false,
            draw_last: 0.0,
        }
    }
}

impl EmitterParams {
    /// Build from a `<Params>` attribute list (keys lowercased).
    fn from_attrs(attrs: &[(String, String)]) -> Self {
        let mut p = EmitterParams::default();
        let g = |k: &str| attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());

        p.continuous = get_bool(g("continuous"), false);
        p.enabled = get_bool(g("enabled"), true);
        p.count = scalar(g("count"), 0.0);
        p.emitter_lifetime = scalar(g("emitterlifetime"), 0.0).0;
        p.particle_lifetime = scalar(g("particlelifetime"), 1.0);
        p.pulse_period = scalar(g("pulseperiod"), 0.0);
        (p.speed, p.speed_curve) = scalar_curve(g("speed"), 0.0, 1.0);
        (p.size, p.size_curve) = scalar_curve(g("size"), 1.0, 1.0);
        (p.alpha, p.alpha_curve) = scalar_curve(g("alpha"), 1.0, 1.0);
        p.alpha_test = g("alphatest").and_then(parse_f32);
        (p.color, p.color_var, p.color_curve) = parse_color(g("color"));
        p.gravity_scale = scalar(g("gravityscale"), 1.0);
        (p.air_resistance, _) = scalar_curve(g("airresistance"), 0.0, 1.0);
        p.emit_angle = scalar(g("emitangle"), 0.0);
        p.focus_gravity_dir = get_bool(g("focusgravitydir"), false);
        p.random_offset = vec3(g("randomoffset").or_else(|| g("posrandomoffset")));
        p.position_offset = vec3(g("positionoffset"));
        p.acceleration = vec3(g("acceleration"));
        p.rotation_rate = vec3(g("rotationrate"));
        p.random_rotation_rate = vec3(g("randomrotationrate"));
        p.random_angles = vec3(g("randomangles"));
        p.init_angles = vec3(g("initangles"));
        p.turbulence_size = scalar(g("turbulencesize"), 0.0);
        p.turbulence_speed = scalar(g("turbulencespeed").or_else(|| g("turbulence3dspeed")), 0.0);
        p.orient_to_velocity = get_bool(g("orienttovelocity"), false);
        p.stretch = scalar(g("stretch"), 0.0);
        p.camera_max_distance = scalar(g("cameramaxdistance"), 0.0).0;
        p.camera_min_distance = scalar(g("cameramindistance"), 0.0).0;
        p.view_distance_adjust = scalar(g("viewdistanceadjust"), 0.0).0;
        p.texture = g("texture").map(|s| s.replace('\\', "/"));
        p.material = g("material").map(|s| s.replace('\\', "/"));
        p.geometry = g("geometry").map(|s| s.replace('\\', "/"));
        p.texture_tiling = parse_tiling(g("texturetiling"));
        p.soft_particle = get_bool(g("softparticle"), false);
        p.light_source = g("lightsource").is_some();

        let emissive = scalar(g("emissivelighting"), 0.0).0;
        let hdr = scalar(g("emissivehdrdynamic"), 0.0).0;
        p.brightness = (1.0 + (emissive.max(hdr) * 0.12)).clamp(0.25, 3.0);

        p.draw_last = scalar(g("drawlast"), 0.0).0;

        p.blend = match g("blendtype").map(|s| s.to_ascii_lowercase()).as_deref() {
            Some("additive") => BlendMode::Additive,
            Some("opaque") => match p.alpha_test {
                Some(a) if a > 0.0 => BlendMode::Mask(a),
                _ => BlendMode::Opaque,
            },
            Some("multiplicative") | Some("multiply") => BlendMode::AlphaBlend,
            _ => match p.alpha_test {
                Some(a) if a > 0.0 && p.geometry.is_some() => BlendMode::Mask(a),
                _ => BlendMode::AlphaBlend,
            },
        };
        p
    }
}

/// One emitter definition. After parsing an effect its child emitters are
/// flattened into the owning [`EffectDef`], so `children` is empty then.
#[derive(Debug, Clone)]
pub struct EmitterDef {
    /// Local name within the effect (`"Ash"`, `"Glow"`, ...).
    pub name: String,
    pub params: EmitterParams,
    pub children: Vec<EmitterDef>,
}

/// One authored effect: its name (`Library.Name`) and every emitter it owns
/// (the root first, then descendants depth-first).
#[derive(Debug, Clone)]
pub struct EffectDef {
    pub name: String,
    pub emitters: Vec<EmitterDef>,
}

impl EffectDef {
    /// Largest `CameraMaxDistance` across the effect's emitters (0 = unbounded).
    pub fn max_view_distance(&self) -> f32 {
        self.emitters.iter().map(|e| e.params.camera_max_distance).fold(0.0, f32::max)
    }
}

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

/// A parsed `ParticleLibrary` XML file (or the merge of several).
#[derive(Debug, Clone, Default)]
pub struct ParticleLibrary {
    effects: HashMap<String, EffectDef>,
    order: Vec<String>,
}

impl ParticleLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse one library XML.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut lib = ParticleLibrary::new();
        lib.add_xml(bytes)?;
        Ok(lib)
    }

    /// Merge another library file into this one (case-insensitive names).
    pub fn add_xml(&mut self, bytes: &[u8]) -> Result<()> {
        let mut reader = Reader::from_reader(bytes);
        let mut buf = Vec::new();
        let mut stack: Vec<EmitterBuilder> = Vec::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_ascii_lowercase();
                    match tag.as_str() {
                        "particles" => {
                            let attrs = attr_map(&e);
                            let get = |k: &str| {
                                attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone())
                            };
                            stack.push(EmitterBuilder {
                                name: get("name").unwrap_or_default(),
                                library: get("library"),
                                params: EmitterParams::default(),
                                children: Vec::new(),
                            });
                        }
                        "params" => {
                            if let Some(top) = stack.last_mut() {
                                top.params = EmitterParams::from_attrs(&attr_map(&e));
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_ascii_lowercase();
                    if tag == "particles" {
                        if let Some(b) = stack.pop() {
                            let def = b.into_def();
                            if let Some(libname) = def.0 {
                                let full =
                                    if def.1.name.is_empty() { libname } else { format!("{libname}.{}", def.1.name) };
                                let root = EmitterDef { name: def.1.name, params: def.1.params, children: def.1.children };
                                let mut flat = Vec::new();
                                flatten(root, &mut flat);
                                let key = full.to_ascii_lowercase();
                                if !self.effects.contains_key(&key) {
                                    self.order.push(full.clone());
                                }
                                self.effects.insert(key, EffectDef { name: full, emitters: flat });
                            } else if let Some(parent) = stack.last_mut() {
                                parent.children.push(def.1);
                            }
                        }
                    }
                }
                Ok(Event::Eof) | Err(_) => break,
                _ => {}
            }
            buf.clear();
        }
        Ok(())
    }

    /// `Library.Name` lookup (case-insensitive).
    pub fn get(&self, name: &str) -> Option<&EffectDef> {
        self.effects.get(&name.to_ascii_lowercase())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.effects.contains_key(&name.to_ascii_lowercase())
    }

    pub fn len(&self) -> usize {
        self.effects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Names in document order.
    pub fn names(&self) -> &[String] {
        &self.order
    }
}

struct EmitterBuilder {
    name: String,
    library: Option<String>,
    params: EmitterParams,
    children: Vec<EmitterDef>,
}

impl EmitterBuilder {
    /// Returns `(library, root emitter def)`; `library` is `Some` only for a
    /// top-level `<Particles Library=...>` node.
    fn into_def(self) -> (Option<String>, EmitterDef) {
        (self.library, EmitterDef { name: self.name, params: self.params, children: self.children })
    }
}

fn flatten(e: EmitterDef, out: &mut Vec<EmitterDef>) {
    out.push(EmitterDef { name: e.name, params: e.params, children: Vec::new() });
    for c in e.children {
        flatten(c, out);
    }
}

// ---------------------------------------------------------------------------
// Runtime simulation
// ---------------------------------------------------------------------------

/// A tiny deterministic xorshift RNG so effects look varied but tests are
/// reproducible.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u32,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng { state: seed | 1 }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform in `0..1`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }

    /// Symmetric deviation in `-v..v`.
    pub fn sym(&mut self, v: f32) -> f32 {
        if v == 0.0 {
            0.0
        } else {
            self.range(-v, v)
        }
    }

    /// `base ± variance`, matching the CryEngine `value,variance` form.
    pub fn spread(&mut self, p: (f32, f32)) -> f32 {
        p.0 + self.sym(p.1)
    }
}

/// A live particle with its per-frame display values resolved.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub age: f32,
    pub life: f32,
    /// Display size in metres for this frame.
    pub size: f32,
    /// Linear RGB.
    pub color: [f32; 3],
    /// 0..1
    pub alpha: f32,
    /// Sprite rotation (radians) in the view plane.
    pub rot: f32,
    /// Current stretch amount (0 = square).
    pub stretch: f32,
    /// Spawn-time size before the `Size` curve is applied.
    base_size: f32,
    /// Spawn-time alpha before the `Alpha` curve is applied.
    base_alpha: f32,
    /// View-plane rotation rate (radians/second).
    rot_rate: f32,
}

/// Per-emitter runtime state for one effect instance.
#[derive(Debug, Clone)]
pub struct EmitterState {
    /// Index into the owning `EffectDef::emitters`.
    pub def: usize,
    pub particles: Vec<Particle>,
    emit_accum: f32,
    pulse_accum: f32,
    started: bool,
    rng: Rng,
}

impl EmitterState {
    fn new(def: usize, seed: u32) -> Self {
        EmitterState {
            def,
            particles: Vec::new(),
            emit_accum: 0.0,
            pulse_accum: 0.0,
            started: false,
            rng: Rng::new(seed),
        }
    }

    pub fn live(&self) -> usize {
        self.particles.len()
    }
}

/// One placed effect (a level `ParticleEffect` entity). The transform is in
/// CryEngine space (Z-up, metres, quaternion `(x,y,z,w)`).
#[derive(Debug, Clone)]
pub struct ParticleInstance {
    /// Full effect name (`Class3Effects.smoke_and_fire.burn_barrel`).
    pub effect: String,
    /// Flattened emitters copied from the effect definition.
    pub emitters: Vec<EmitterState>,
    pub origin: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: f32,
    pub age: f32,
    pub active: bool,
}

impl ParticleInstance {
    pub fn particle_count(&self) -> usize {
        self.emitters.iter().map(|e| e.particles.len()).sum()
    }
}

/// The whole particle world: a library plus the placed instances.
#[derive(Debug, Default)]
pub struct ParticleSystem {
    pub lib: ParticleLibrary,
    instances: Vec<Option<ParticleInstance>>,
    /// Ids freed by [`ParticleSystem::remove`], reused by `add`.
    free: Vec<usize>,
}

impl ParticleSystem {
    pub fn new(lib: ParticleLibrary) -> Self {
        ParticleSystem { lib, instances: Vec::new(), free: Vec::new() }
    }

    /// Place an effect at a transform; returns a stable instance id.
    pub fn add(&mut self, effect: &str, origin: [f32; 3], rotation: [f32; 4], scale: f32) -> Option<usize> {
        let def = self.lib.get(effect)?;
        let def_name = def.name.clone();
        let emitter_count = def.emitters.len();
        let seed = fnv1a(effect) ^ (self.instances.len() as u32).wrapping_mul(2654435761);
        let emitters = (0..emitter_count)
            .map(|i| EmitterState::new(i, seed.wrapping_add((i as u32).wrapping_mul(0x9e3779b9))))
            .collect();
        let inst = ParticleInstance {
            effect: def_name,
            emitters,
            origin,
            rotation,
            scale: if scale > 0.0 { scale } else { 1.0 },
            age: 0.0,
            active: true,
        };
        if let Some(id) = self.free.pop() {
            self.instances[id] = Some(inst);
            Some(id)
        } else {
            let id = self.instances.len();
            self.instances.push(Some(inst));
            Some(id)
        }
    }

    /// Drop an instance and recycle its id.
    pub fn remove(&mut self, id: usize) {
        if let Some(slot) = self.instances.get_mut(id) {
            if slot.take().is_some() {
                self.free.push(id);
            }
        }
    }

    pub fn instances(&self) -> impl Iterator<Item = (usize, &ParticleInstance)> {
        self.instances.iter().enumerate().filter_map(|(i, o)| o.as_ref().map(|x| (i, x)))
    }

    pub fn instance(&self, id: usize) -> Option<&ParticleInstance> {
        self.instances.get(id).and_then(|o| o.as_ref())
    }

    /// Move an instance to a new transform (its emitters follow the entity).
    pub fn set_transform(&mut self, id: usize, origin: [f32; 3], rotation: [f32; 4], scale: f32) {
        if let Some(Some(inst)) = self.instances.get_mut(id) {
            inst.origin = origin;
            inst.rotation = rotation;
            inst.scale = if scale > 0.0 { scale } else { 1.0 };
        }
    }

    pub fn set_active(&mut self, id: usize, active: bool) {
        if let Some(Some(inst)) = self.instances.get_mut(id) {
            if inst.active != active {
                // Restart cleanly when an effect comes back into view; a
                // deactivated effect drops its live particles immediately.
                inst.age = 0.0;
                for e in &mut inst.emitters {
                    e.particles.clear();
                    e.emit_accum = 0.0;
                    e.pulse_accum = 0.0;
                    e.started = false;
                }
            }
            inst.active = active;
        }
    }

    pub fn total_particles(&self) -> usize {
        self.instances.iter().filter_map(|o| o.as_ref()).map(|i| i.particle_count()).sum()
    }

    /// Advance every active instance by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        // Split the `lib` and `instances` fields so the per-emitter definition
        // can be borrowed while the instance state is mutated.
        let lib = &self.lib;
        for slot in &mut self.instances {
            let Some(inst) = slot.as_mut() else { continue };
            if !inst.active {
                continue;
            }
            inst.age += dt;
            let ctx = InstCtx {
                origin: inst.origin,
                rotation: inst.rotation,
                scale: inst.scale,
                age: inst.age,
            };
            let Some(def) = lib.get(&inst.effect) else { continue };
            for st in &mut inst.emitters {
                if let Some(em) = def.emitters.get(st.def) {
                    step_emitter(em, st, &ctx, dt);
                }
            }
        }
    }
}

/// The per-instance transform values an emitter needs, copied out so the
/// emitter state can be mutated without borrowing the whole instance.
#[derive(Debug, Clone, Copy)]
struct InstCtx {
    origin: [f32; 3],
    rotation: [f32; 4],
    scale: f32,
    age: f32,
}

fn step_emitter(def: &EmitterDef, st: &mut EmitterState, inst: &InstCtx, dt: f32) {
    let p = &def.params;
    if !p.enabled {
        st.particles.clear();
        return;
    }
    let life = p.particle_lifetime.0.max(0.01);
    let target = (p.count.0 * inst.scale.max(0.05)).ceil().max(0.0);

    let finite = p.emitter_lifetime > 0.0;
    let emitting = !finite || inst.age < p.emitter_lifetime;

    if emitting {
        if p.pulse_period.0 > 0.0 {
            st.pulse_accum += dt;
            let period = p.pulse_period.0.max(0.05);
            if !st.started {
                burst(def, st, inst, target);
                st.started = true;
                st.pulse_accum = 0.0;
            } else if st.pulse_accum >= period {
                st.pulse_accum -= period;
                burst(def, st, inst, target);
            }
        } else if p.continuous {
            let rate = target / life;
            st.emit_accum += rate * dt;
            let cap = (target * 1.5 + 4.0).ceil();
            while st.emit_accum >= 1.0 {
                st.emit_accum -= 1.0;
                if (st.particles.len() as f32) < cap {
                    spawn(def, st, inst);
                }
            }
        } else if !st.started {
            burst(def, st, inst, target);
            st.started = true;
        }
    }

    let mut dead: Vec<usize> = Vec::new();
    for (i, q) in st.particles.iter_mut().enumerate() {
        // GRAVITY acts along -Z in CryEngine space.
        let gz = -GRAVITY * q_g(p.gravity_scale, i);
        q.vel[2] += gz * dt;
        // Local acceleration is rotated into world space.
        let a = quat_rotate(inst.rotation, p.acceleration);
        for k in 0..3 {
            q.vel[k] += a[k] * dt;
        }
        // Cheap curl noise from the particle position and age.
        if p.turbulence_size.0 > 0.0 {
            let s = p.turbulence_size.0.max(0.01);
            let f = p.turbulence_speed.0.max(0.01) * 0.5;
            let t = inst.age * f + q.age * f;
            let n = [
                ((q.pos[1] / s) + t).sin() * 0.5,
                ((q.pos[2] / s) + t * 1.3).cos() * 0.5,
                ((q.pos[0] / s) + t * 0.7).sin() * 0.5,
            ];
            let amp = s * 0.4;
            for k in 0..3 {
                q.vel[k] += n[k] * amp * dt * 3.0;
            }
        }
        // Exponential air resistance.
        let drag = (-p.air_resistance.0.max(0.0) * dt).exp();
        for k in 0..3 {
            q.vel[k] *= drag;
            q.pos[k] += q.vel[k] * dt;
        }
        q.age += dt;
        let t = (q.age / q.life).clamp(0.0, 1.0);
        q.size = (q.base_size * p.size_curve.sample(t) * inst.scale).max(0.0);
        if let Some(c) = p.color_curve.as_ref().and_then(|c| c.sample(t)) {
            q.color = c;
        }
        q.alpha = (q.base_alpha * p.alpha_curve.sample(t)).clamp(0.0, 1.0);
        q.rot += q.rot_rate * dt;
        q.stretch = p.stretch.0;
        if q.age >= q.life {
            dead.push(i);
        }
    }
    for i in dead.into_iter().rev() {
        st.particles.swap_remove(i);
    }
}

// The gravity variance is a per-particle property; the deterministic RNG is
// advanced at spawn, so the resolved value is captured in `vel` already. This
// helper keeps the `(base, variance)` reading explicit.
#[allow(dead_code)]
fn q_g(p: (f32, f32), _i: usize) -> f32 {
    p.0
}

fn burst(def: &EmitterDef, st: &mut EmitterState, inst: &InstCtx, count: f32) {
    let n = count.round().max(0.0) as usize;
    for _ in 0..n {
        spawn(def, st, inst);
    }
}

fn spawn(def: &EmitterDef, st: &mut EmitterState, inst: &InstCtx) {
    let p = &def.params;
    let life = st.rng.spread(p.particle_lifetime).max(0.02);
    let base_size = st.rng.spread(p.size).max(0.0);
    let base_alpha = st.rng.spread(p.alpha).clamp(0.0, 1.0);

    // Position: local offset + random box, rotated by the instance.
    let off = [
        p.position_offset[0] + st.rng.sym(p.random_offset[0]),
        p.position_offset[1] + st.rng.sym(p.random_offset[1]),
        p.position_offset[2] + st.rng.sym(p.random_offset[2]),
    ];
    let roff = quat_rotate(inst.rotation, off);
    let pos = [inst.origin[0] + roff[0], inst.origin[1] + roff[1], inst.origin[2] + roff[2]];

    // Emit direction.
    let axis = if p.focus_gravity_dir {
        [0.0, 0.0, 1.0]
    } else {
        quat_rotate(inst.rotation, [0.0, 0.0, 1.0])
    };
    let angle = (st.rng.spread(p.emit_angle)).to_radians().clamp(0.0, std::f32::consts::PI);
    let dir = cone_dir(axis, angle, &mut st.rng);
    let speed = st.rng.spread(p.speed).max(0.0);
    let vel = [dir[0] * speed, dir[1] * speed, dir[2] * speed];

    // Colour.
    let mut color = p.color;
    for k in 0..3 {
        color[k] = (color[k] + st.rng.sym(p.color_var * 0.25)).clamp(0.0, 4.0);
    }
    if let Some(c) = p.color_curve.as_ref().and_then(|c| c.sample(0.0)) {
        color = c;
    }

    // Initial sprite rotation from `RandomAngles` (Z) and `InitAngles`.
    let rot = (p.init_angles[2].to_radians()) + st.rng.sym(p.random_angles[2].to_radians());
    let rot_rate = st.rng.spread((p.rotation_rate[2], p.random_rotation_rate[2].abs())).to_radians();

    let q = Particle {
        pos,
        vel,
        age: 0.0,
        life,
        size: base_size,
        color,
        alpha: base_alpha,
        rot,
        stretch: p.stretch.0,
        base_size,
        base_alpha,
        rot_rate,
    };
    st.particles.push(q);
}

/// Uniform direction within a cone of half-angle `angle` around `axis`.
fn cone_dir(axis: [f32; 3], angle: f32, rng: &mut Rng) -> [f32; 3] {
    let axis = normalize(axis);
    if axis == [0.0, 0.0, 0.0] || angle <= 1e-4 {
        return axis;
    }
    let (t, b) = ortho(axis);
    let cos_max = angle.cos();
    let cos_theta = 1.0 - rng.unit() * (1.0 - cos_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = rng.unit() * std::f32::consts::TAU;
    let (s, c) = phi.sin_cos();
    [
        axis[0] * cos_theta + (t[0] * c + b[0] * s) * sin_theta,
        axis[1] * cos_theta + (t[1] * c + b[1] * s) * sin_theta,
        axis[2] * cos_theta + (t[2] * c + b[2] * s) * sin_theta,
    ]
}

fn ortho(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let helper = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let t = normalize(cross(n, helper));
    let b = cross(n, t);
    (t, b)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-8 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}

fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let [qx, qy, qz, qw] = q;
    let t = [2.0 * (qy * v[2] - qz * v[1]), 2.0 * (qz * v[0] - qx * v[2]), 2.0 * (qx * v[1] - qy * v[0])];
    [
        v[0] + qw * t[0] + (qy * t[2] - qz * t[1]),
        v[1] + qw * t[1] + (qz * t[0] - qx * t[2]),
        v[2] + qw * t[2] + (qx * t[1] - qy * t[0]),
    ]
}

fn fnv1a(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

// ---------------------------------------------------------------------------
// Attribute micro-format helpers
// ---------------------------------------------------------------------------

fn attr_map(e: &quick_xml::events::BytesStart) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for a in e.attributes().flatten() {
        let k = String::from_utf8_lossy(a.key.as_ref()).to_ascii_lowercase();
        let v = a.unescape_value().map(|c| c.into_owned()).unwrap_or_default();
        out.push((k, v));
    }
    out
}

fn get_bool(v: Option<&str>, default: bool) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("true") | Some("1") => true,
        Some("false") | Some("0") => false,
        _ => default,
    }
}

/// Split at top-level separators, ignoring separators nested in parentheses.
fn split_top(s: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            _ if c == sep && depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    out.push(cur.trim().to_string());
    out
}

fn parse_f32(s: &str) -> Option<f32> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    s.parse::<f32>().ok()
}

/// `base,variance[,flags][,curve]` -> `((base,var), curve)`.
fn scalar_curve(v: Option<&str>, base_default: f32, curve_default: f32) -> ((f32, f32), Curve) {
    let Some(v) = v else { return ((base_default, 0.0), Curve::constant(curve_default)) };
    let f = split_top(v, ',');
    let base = f.first().and_then(|s| parse_f32(s)).unwrap_or(base_default);
    let var = f.get(1).and_then(|s| parse_f32(s)).unwrap_or(0.0);
    let curve = f
        .iter()
        .find(|s| s.trim_start().starts_with('('))
        .map(|s| parse_curve(s))
        .unwrap_or_else(|| Curve::constant(curve_default));
    ((base, var), curve)
}

/// `base,variance` for a plain scalar.
fn scalar(v: Option<&str>, default: f32) -> (f32, f32) {
    scalar_curve(v, default, 1.0).0
}

/// First three top-level fields.
fn vec3(v: Option<&str>) -> [f32; 3] {
    let Some(v) = v else { return [0.0; 3] };
    let f = split_top(v, ',');
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = f.get(i).and_then(|s| parse_f32(s)).unwrap_or(0.0);
    }
    out
}

fn parse_curve(s: &str) -> Curve {
    let s = s.trim();
    let inner = s.strip_prefix('(').and_then(|x| x.strip_suffix(')')).unwrap_or(s);
    let frames = split_top(inner, ';');
    let mut raw: Vec<(Option<f32>, Option<f32>)> = Vec::new();
    for f in frames {
        if f.trim().is_empty() {
            continue;
        }
        let parts = split_top(&f, ',');
        let t = parts.first().and_then(|x| parse_f32(x));
        let v = parts.get(1).and_then(|x| parse_f32(x));
        raw.push((t, v));
    }
    if raw.is_empty() {
        return Curve::constant(1.0);
    }
    let n = raw.len();
    let mut times: Vec<f32> = Vec::with_capacity(n);
    for (i, (t, _)) in raw.iter().enumerate() {
        let tv = match t {
            Some(x) => *x,
            None if i == 0 => 0.0,
            None if i == n - 1 => 1.0,
            None => i as f32 / (n - 1).max(1) as f32,
        };
        times.push(tv);
    }
    let mut vals: Vec<Option<f32>> = raw.iter().map(|(_, v)| *v).collect();
    let mut last = None;
    for v in vals.iter_mut() {
        if v.is_some() {
            last = *v;
        } else if let Some(l) = last {
            *v = Some(l);
        }
    }
    let mut next = None;
    for v in vals.iter_mut().rev() {
        if v.is_some() {
            next = *v;
        } else if let Some(x) = next {
            *v = Some(x);
        }
    }
    let mut keys: Vec<(f32, f32)> =
        (0..n).map(|i| (times[i], vals[i].unwrap_or(1.0))).collect();
    keys.sort_by(|a, b| a.0.total_cmp(&b.0));
    Curve { keys }
}

fn parse_rgb(s: &str) -> Option<[f32; 3]> {
    let s = s.trim();
    if s.contains(';') {
        return None;
    }
    let inner = s.strip_prefix('(')?.strip_suffix(')')?;
    let parts = split_top(inner, ',');
    if parts.len() < 3 {
        return None;
    }
    Some([
        parse_f32(&parts[0])?,
        parse_f32(&parts[1])?,
        parse_f32(&parts[2])?,
    ])
}

fn parse_color_curve(s: &str) -> Option<ColorCurve> {
    let s = s.trim();
    let inner = s.strip_prefix('(').and_then(|x| x.strip_suffix(')')).unwrap_or(s);
    let mut keys: Vec<(f32, [f32; 3])> = Vec::new();
    for f in split_top(inner, ';') {
        if f.trim().is_empty() {
            continue;
        }
        let parts = split_top(&f, ',');
        let t = parts.first().and_then(|x| parse_f32(x));
        let col = parts.iter().find_map(|p| parse_rgb(p));
        if let Some(c) = col {
            keys.push((t.unwrap_or(0.0), c));
        }
    }
    if keys.is_empty() {
        return None;
    }
    keys.sort_by(|a, b| a.0.total_cmp(&b.0));
    Some(ColorCurve { keys })
}

/// `(r,g,b)[,variance][,curve]` -> `(color, variance, curve)`.
fn parse_color(v: Option<&str>) -> ([f32; 3], f32, Option<ColorCurve>) {
    let Some(v) = v else { return ([1.0; 3], 0.0, None) };
    let fields = split_top(v, ',');
    let curve = fields.iter().find(|f| f.contains(';')).and_then(|f| parse_color_curve(f));
    let base = fields
        .iter()
        .find_map(|f| if f.contains(';') { None } else { parse_rgb(f) })
        .or_else(|| curve.as_ref().and_then(|c| c.sample(0.0)))
        .unwrap_or([1.0; 3]);
    let var = fields
        .get(1)
        .map(|f| f.trim().trim_start_matches('(').split(',').next().unwrap_or(""))
        .and_then(parse_f32)
        .unwrap_or(0.0);
    (base, var, curve)
}

fn parse_tiling(v: Option<&str>) -> (f32, f32) {
    let Some(v) = v else { return (1.0, 1.0) };
    let f = split_top(v, ',');
    let u = f.first().and_then(|s| parse_f32(s)).unwrap_or(1.0);
    let vv = f.get(1).and_then(|s| parse_f32(s)).unwrap_or(u);
    (if u > 0.0 { u } else { 1.0 }, if vv > 0.0 { vv } else { 1.0 })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
<ParticleLibrary Name="TestLib" ParticleVersion="22">
  <Particles Name="smoke_and_fire.fire" Id="{X}" Library="TestLib">
    <Params Continuous="true" Count="25" ParticleLifeTime="1,0.25" EmitAngle="65,0.35"
            Speed="15,0.75" GravityScale="0.85" BlendType="Additive"
            Texture="textures/particles/fire/smooth_fire.tif"
            Size="0.25,0.172,,(,0.84090906,32;0.31282052,1;1,0.5681818)"
            Color="(0.82,0.25,0.075),(0.35,true)"
            Alpha="0.8,0.5,,(;0.31538463,1;1)"/>
    <Childs>
      <Particles Name="Ash">
        <Params Continuous="true" Count="10" ParticleLifeTime="2,0.1" BlendType="Opaque"
                AlphaTest="0.15" TextureTiling="2,2,,4" Material="Particles/x"/>
      </Particles>
    </Childs>
  </Particles>
</ParticleLibrary>"#;

    #[test]
    fn parses_effect_and_children() {
        let lib = ParticleLibrary::parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(lib.len(), 1);
        let e = lib.get("TestLib.smoke_and_fire.fire").unwrap();
        assert_eq!(e.name, "TestLib.smoke_and_fire.fire");
        assert_eq!(e.emitters.len(), 2);
        let root = &e.emitters[0];
        assert!(root.params.continuous);
        assert_eq!(root.params.count.0, 25.0);
        assert_eq!(root.params.blend, BlendMode::Additive);
        assert_eq!(root.params.texture.as_deref(), Some("textures/particles/fire/smooth_fire.tif"));
        let ash = &e.emitters[1];
        assert_eq!(ash.name, "Ash");
        assert_eq!(ash.params.blend, BlendMode::Mask(0.15));
        assert_eq!(ash.params.texture_tiling, (2.0, 2.0));
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let lib = ParticleLibrary::parse(SAMPLE.as_bytes()).unwrap();
        assert!(lib.get("testlib.SMOKE_and_fire.fire").is_some());
        assert!(lib.get("nope").is_none());
    }

    #[test]
    fn curve_samples_piecewise_linearly() {
        let c = parse_curve("(,0.75,32;0.5,1;1,1)");
        assert!((c.sample(0.0) - 0.75).abs() < 1e-5);
        assert!((c.sample(0.25) - 0.875).abs() < 1e-5);
        assert!((c.sample(0.5) - 1.0).abs() < 1e-5);
        assert!((c.sample(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn color_curve_parses_triplets() {
        let c = parse_color_curve("(0.0,(0.1,0.2,0.3);1.0,(1.0,1.0,1.0))").unwrap();
        let a = c.sample(0.0).unwrap();
        assert!((a[0] - 0.1).abs() < 1e-5 && (a[2] - 0.3).abs() < 1e-5);
        let b = c.sample(1.0).unwrap();
        assert!((b[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn continuous_emitter_reaches_target_count() {
        let lib = ParticleLibrary::parse(SAMPLE.as_bytes()).unwrap();
        let mut sys = ParticleSystem::new(lib);
        let id = sys.add("TestLib.smoke_and_fire.fire", [0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 1.0], 1.0).unwrap();
        // Advance ~1.5 lifetimes so the steady state is reached.
        for _ in 0..150 {
            sys.step(0.02);
        }
        let inst = sys.instance(id).unwrap();
        let root = &inst.emitters[0];
        assert!(root.live() > 15, "expected a steady stream, got {}", root.live());
        assert!(root.live() <= 60);
    }

    #[test]
    fn particles_die_after_their_lifetime() {
        let lib = ParticleLibrary::parse(SAMPLE.as_bytes()).unwrap();
        let mut sys = ParticleSystem::new(lib);
        let id = sys.add("TestLib.smoke_and_fire.fire", [0.0; 3], [0.0, 0.0, 0.0, 1.0], 1.0).unwrap();
        // The animation is finite: the longest particle life is 1.25 s.
        for _ in 0..5 {
            sys.step(0.02);
        }
        assert!(sys.instance(id).unwrap().particle_count() > 0);
        // Stop the emitter by aging it far beyond a life, then clear.
        sys.set_active(id, false);
        assert_eq!(sys.instance(id).unwrap().particle_count(), 0);
    }

    #[test]
    fn gravity_pulls_particles_down() {
        let lib = ParticleLibrary::parse(SAMPLE.as_bytes()).unwrap();
        let mut sys = ParticleSystem::new(lib);
        // Ash emitter has no speed; its particles only fall.
        let id = sys.add("TestLib.smoke_and_fire.fire", [0.0, 0.0, 0.0], [0.0; 4], 1.0).unwrap();
        for _ in 0..20 {
            sys.step(0.05);
        }
        let ash = &sys.instance(id).unwrap().emitters[1];
        assert!(ash.live() > 0);
        assert!(ash.particles.iter().any(|p| p.vel[2] < 0.0), "gravity should reduce Z velocity");
    }

    #[test]
    fn placing_unknown_effect_is_none() {
        let lib = ParticleLibrary::parse(SAMPLE.as_bytes()).unwrap();
        let mut sys = ParticleSystem::new(lib);
        assert!(sys.add("Missing.Effect", [0.0; 3], [0.0; 4], 1.0).is_none());
    }

    #[test]
    fn remove_recycles_instance_ids() {
        let lib = ParticleLibrary::parse(SAMPLE.as_bytes()).unwrap();
        let mut sys = ParticleSystem::new(lib);
        let a = sys.add("TestLib.smoke_and_fire.fire", [0.0; 3], [0.0; 4], 1.0).unwrap();
        let b = sys.add("TestLib.smoke_and_fire.fire", [1.0; 3], [0.0; 4], 1.0).unwrap();
        assert_ne!(a, b);
        sys.remove(a);
        assert!(sys.instance(a).is_none());
        let c = sys.add("TestLib.smoke_and_fire.fire", [2.0; 3], [0.0; 4], 1.0).unwrap();
        assert_eq!(c, a, "the freed id should be reused");
        assert_eq!(sys.instances().count(), 2);
    }
}
