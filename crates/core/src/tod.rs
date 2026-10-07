//! CryEngine 3 time-of-day and environment model for State of Decay (YOSE).
//!
//! The level's `mission_mission0.xml` carries two global lighting blocks:
//!
//! * `<Environment>` — static settings and the run-time clock range. The engine
//!   parses this in the environment loader at `0x47add0` (attribute strings are
//!   the ones at `.rdata` RVA `0x132be54`..`0x132c20c`): `Fog`, `EnvState`,
//!   `SkyBox`, `Ocean`, `OceanAnimation`, `IceScattering`, `Moon`, `HDRSetup`
//!   and `Lighting`. `Lighting.SunRotation` lands at object `+0x114` and
//!   `Lighting.Longitude` at `+0x118` (`0x47afe6`/`0x47b028`).
//! * `<TimeOfDay>` — the animated variables. Each `<Variable>` has a base
//!   `Value`/`Color` and an optional `<Spline Keys="time:value,..."/>`. The
//!   engine reads the clock in hours (`Time`, `TimeStart`, `TimeEnd`,
//!   `TimeAnimSpeed`) and evaluates every spline at `hour / 24` (see the
//!   `* (1/24)` scale at `0x522415`, constant `0x17319c0`).
//!
//! Spline evaluation: the Scalar spline's interpolate wrapper is `0x51fc70`,
//! which calls the base blend routine `0x524bf0`; the per-type blend is
//! `0x5248d0` (scalar, stride `0x14`) / `0x524d30` (color, stride `0x2c`).
//! The blend is a cubic Hermite using the Hermite basis with tangents
//! `m0 = 3*key[i0].outSlope`, `m1 = 3*key[i1].inSlope`. When the auto slopes
//! are the one-sided differences (`outSlope = (v1-v0)/3`, `inSlope = (v1-v0)/3`
//! for the bracketing pair) the cubic terms cancel and the result is exactly
//! linear between the two keys. `0x5248d0` also short-circuits for the two
//! "step/hold" flag encodings (`flags & 0x38 == 0x18` and `flags & 7 == 3`).
//! The ToD XML never authors flags or tangents, so we implement the linear
//! reduction (values are strictly between adjacent keys; sampling on a key
//! returns that key exactly).
//!
//! Sun direction: `0x47add0` does *not* derive a direction from
//! `SunRotation`/`Longitude`; the compiled `Lighting.SunVector` attribute in
//! the XML is not even read back by the engine (there is no `SunVector`
//! string). At run time `CTimeOfDay::Update` (`0x5226e0`) rebuilds it from the
//! clock: the hour angle `(hour+12)/24 * 2pi` (`0x52298f`, constants
//! `0x17319c0`, `0x172020c`, `0x1720214`) is fed through `fsincos`
//! (`0x522a2a`) together with two orbit angles. The orbit tilt is
//! `pi/2 - Longitude*pi/180` (`0x5229de`..`0x522a21`, constants `0x1721538`,
//! `0x172c354`) and the azimuth node is `-SunRotation*pi/180`. Reconstructing
//! the resulting spherical direction gives, with `H = (hour-12)*15deg`,
//! `alt = asin(cos H * sin(90deg - Longitude))` and azimuth `SunRotation`;
//! composed as `d = (cos alt * sin az, cos alt * cos az, -sin alt)`. At the
//! level's saved `Time="11.916878"` this reproduces the authored
//! `SunVector="-0.25357094,-0.54378468,-0.79999989"` (azimuth `205deg`,
//! altitude `53.13deg`).

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

// ---------------------------------------------------------------------------
// Values and splines
// ---------------------------------------------------------------------------

/// A time-of-day variable value: either a scalar or an RGB color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Scalar(f32),
    Color([f32; 3]),
}

impl Value {
    /// Linear blend `a -> b` at fraction `t` (component-wise for colors).
    pub fn lerp(a: Value, b: Value, t: f32) -> Value {
        match (a, b) {
            (Value::Scalar(x), Value::Scalar(y)) => Value::Scalar(x + (y - x) * t),
            (Value::Color(x), Value::Color(y)) => {
                Value::Color([x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t])
            }
            // Mixed kinds cannot happen in one variable; fall back to `a`.
            _ => a,
        }
    }

    pub fn as_scalar(&self) -> f32 {
        match self {
            Value::Scalar(v) => *v,
            Value::Color(c) => c[0],
        }
    }

    pub fn as_color(&self) -> [f32; 3] {
        match self {
            Value::Color(c) => *c,
            Value::Scalar(v) => [*v, *v, *v],
        }
    }
}

/// One spline key: a normalized day time in `[0, 1]` and its value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Key {
    pub time: f32,
    pub value: Value,
}

/// A per-variable key spline. Keys are kept sorted by time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Spline {
    pub keys: Vec<Key>,
}

impl Spline {
    pub fn new(mut keys: Vec<Key>) -> Spline {
        keys.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        Spline { keys }
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Evaluate at normalized time `t`. Values before the first key and after
    /// the last key are clamped. Between keys, piecewise-linear (the linear
    /// reduction of the engine's Hermite blend).
    pub fn sample(&self, t: f32) -> Option<Value> {
        if self.keys.is_empty() {
            return None;
        }
        let first = self.keys.first().unwrap();
        let last = self.keys.last().unwrap();
        if t <= first.time {
            return Some(first.value);
        }
        if t >= last.time {
            return Some(last.value);
        }
        // Find the last key with time <= t.
        let i = self.keys.partition_point(|k| k.time <= t) - 1;
        let k0 = &self.keys[i];
        let k1 = &self.keys[i + 1];
        let dt = k1.time - k0.time;
        if dt <= 0.0 {
            return Some(k1.value);
        }
        let f = (t - k0.time) / dt;
        Some(Value::lerp(k0.value, k1.value, f))
    }
}

/// One `<Variable>` of the time-of-day graph.
#[derive(Debug, Clone, PartialEq)]
pub struct TodVariable {
    pub name: String,
    /// The base value authored on the `<Variable>` element.
    pub value: Value,
    /// The optional `<Spline>`; `None` means the base value is constant.
    pub spline: Option<Spline>,
}

impl TodVariable {
    /// Sample at normalized time `t` (`0..1`), falling back to the base value.
    pub fn sample(&self, t: f32) -> Value {
        self.spline
            .as_ref()
            .and_then(|s| s.sample(t))
            .unwrap_or(self.value)
    }
}

/// The parsed `<TimeOfDay>` block.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeOfDay {
    /// Current clock in hours, e.g. `11.916878`.
    pub time: f32,
    /// Clock range in hours (`TimeStart`, `TimeEnd`).
    pub time_start: f32,
    pub time_end: f32,
    /// Authored animation speed (`TimeAnimSpeed`).
    pub time_anim_speed: f32,
    pub variables: Vec<TodVariable>,
}

impl Default for TimeOfDay {
    fn default() -> Self {
        TimeOfDay { time: 0.0, time_start: 0.0, time_end: 24.0, time_anim_speed: 0.0, variables: Vec::new() }
    }
}

/// Result of sampling every time-of-day variable at one instant.
#[derive(Debug, Clone, PartialEq)]
pub struct TodSample {
    /// The requested hour (0..24).
    pub hour: f32,
    /// Every variable in document order.
    pub values: Vec<(String, Value)>,
}

impl TodSample {
    pub fn get(&self, name: &str) -> Option<Value> {
        self.values.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    pub fn scalar(&self, name: &str) -> Option<f32> {
        self.get(name).map(|v| v.as_scalar())
    }

    pub fn color(&self, name: &str) -> Option<[f32; 3]> {
        self.get(name).map(|v| v.as_color())
    }

    pub fn sun_color(&self) -> Option<[f32; 3]> {
        self.color("Sun color")
    }

    pub fn sky_color(&self) -> Option<[f32; 3]> {
        self.color("Sky color")
    }

    pub fn fog_color(&self) -> Option<[f32; 3]> {
        self.color("Fog color")
    }
}

impl TimeOfDay {
    /// Look up a variable by its authored name.
    pub fn variable(&self, name: &str) -> Option<&TodVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// Sample every variable at `hour` (in hours). Splines are keyed on the
    /// normalized day (`hour / 24`), matching the engine's `*(1/24)` scale.
    pub fn sample(&self, hour: f32) -> TodSample {
        let t = hour / 24.0;
        let values = self.variables.iter().map(|v| (v.name.clone(), v.sample(t))).collect();
        TodSample { hour, values }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// `<Fog>` settings.
#[derive(Debug, Clone, PartialEq)]
pub struct Fog {
    pub view_distance: f32,
    pub view_distance_low_spec: f32,
    pub ldr_global_dens_mult: f32,
    pub use_fog_color_gradient: bool,
}

impl Default for Fog {
    fn default() -> Self {
        Fog { view_distance: 0.0, view_distance_low_spec: 0.0, ldr_global_dens_mult: 1.0, use_fog_color_gradient: false }
    }
}

/// `<EnvState>` settings.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvState {
    pub wind_vector: [f32; 3],
    pub show_terrain_surface: bool,
    pub sun_shadows_min_spec: bool,
    pub cloud_shadow_texture: String,
    pub cloud_shadow_speed: f32,
    pub use_layers_activation: bool,
}

impl Default for EnvState {
    fn default() -> Self {
        EnvState {
            wind_vector: [1.0, 0.0, 0.0],
            show_terrain_surface: false,
            sun_shadows_min_spec: false,
            cloud_shadow_texture: String::new(),
            cloud_shadow_speed: 0.0,
            use_layers_activation: false,
        }
    }
}

/// `<SkyBox>` settings.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SkyBox {
    pub material: String,
    pub material_low_spec: String,
    pub angle: f32,
    pub stretching: f32,
}

/// `<Ocean>` settings.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ocean {
    pub material: String,
    pub fog_color: [f32; 3],
    pub fog_color_multiplier: f32,
    pub fog_density: f32,
    pub caustic_depth: f32,
    pub caustic_intensity: f32,
    pub caustics_tilling: f32,
}

/// `<OceanAnimation>` settings.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OceanAnimation {
    pub wind_direction: f32,
    pub wind_speed: f32,
    pub waves_speed: f32,
    pub waves_amount: f32,
    pub waves_size: f32,
}

/// `<IceScattering>` settings.
#[derive(Debug, Clone, PartialEq)]
pub struct IceScattering {
    pub scattering_direction: [f32; 3],
}

impl Default for IceScattering {
    fn default() -> Self {
        IceScattering { scattering_direction: [0.0, 0.0, 1.0] }
    }
}

/// `<Moon>` settings.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Moon {
    pub latitude: f32,
    pub longitude: f32,
    pub size: f32,
    pub texture: String,
}

/// `<HDRSetup>` settings.
#[derive(Debug, Clone, PartialEq)]
pub struct HdrSetup {
    pub bloom_color: [f32; 3],
    pub bloom_mul: f32,
    pub flares_color: [f32; 3],
    pub flares_mul: f32,
    pub streaks_color: [f32; 3],
    pub streaks_mul: f32,
    pub streak_scale: f32,
}

impl Default for HdrSetup {
    fn default() -> Self {
        HdrSetup {
            bloom_color: [255.0, 255.0, 255.0],
            bloom_mul: 1.0,
            flares_color: [255.0, 255.0, 255.0],
            flares_mul: 0.1,
            streaks_color: [255.0, 255.0, 255.0],
            streaks_mul: 0.33,
            streak_scale: 1.0,
        }
    }
}

/// `<Lighting>` settings. `sun_vector` is the authored, pre-baked direction
/// (the engine does not read it back; it is recomputed at run time, see
/// [`Lighting::sun_direction`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Lighting {
    /// Azimuth node in degrees (`SunRotation`).
    pub sun_rotation: f32,
    /// Authored `SunHeight` (degrees; the engine build does not parse it).
    pub sun_height: f32,
    pub algorithm: i32,
    pub lighting: bool,
    pub shadows: bool,
    pub shadow_intensity: f32,
    pub il_quality: i32,
    pub hemi_sample_quality: i32,
    /// Orbit tilt in degrees (`Longitude`).
    pub longitude: f32,
    pub dawn_time: f32,
    pub dawn_duration: f32,
    pub dusk_time: f32,
    pub dusk_duration: f32,
    /// Authored `SunVector` (unit direction the level was saved with).
    pub sun_vector: [f32; 3],
}

impl Default for Lighting {
    fn default() -> Self {
        Lighting {
            sun_rotation: 0.0,
            sun_height: 0.0,
            algorithm: 3,
            lighting: true,
            shadows: true,
            shadow_intensity: 100.0,
            il_quality: 1,
            hemi_sample_quality: 9,
            longitude: 0.0,
            dawn_time: 0.0,
            dawn_duration: 0.0,
            dusk_time: 0.0,
            dusk_duration: 0.0,
            sun_vector: [0.0, 0.0, -1.0],
        }
    }
}

impl Lighting {
    /// Sun direction (unit, pointing from the sun toward the scene, i.e. `z`
    /// is negative with the sun above the horizon) at `hour` (0..24).
    ///
    /// Derived from `CTimeOfDay::Update` (`0x5226e0`): the hour angle is
    /// `H = (hour-12)*15deg` (`(hour+12)/24*2pi`), the noon altitude is
    /// `90deg - Longitude` and the orbital azimuth is `SunRotation`.
    pub fn sun_direction(&self, hour: f32) -> [f32; 3] {
        let h = (hour - 12.0) * std::f32::consts::PI / 12.0;
        let max_alt = std::f32::consts::FRAC_PI_2 - self.longitude.to_radians();
        let alt = (h.cos() * max_alt.sin()).clamp(-1.0, 1.0).asin();
        let az = self.sun_rotation.to_radians();
        [alt.cos() * az.sin(), alt.cos() * az.cos(), -alt.sin()]
    }

    /// `true` when the sun's *light direction* points below the horizon (i.e.
    /// the sun itself is below the horizon — night).
    pub fn is_night(&self, hour: f32) -> bool {
        self.sun_direction(hour)[2] >= 0.0
    }
}

/// `<Terrain>` settings (level detail-layer view distance ratio).
#[derive(Debug, Clone, PartialEq)]
pub struct Terrain {
    pub detail_layers_view_dist_ratio: f32,
}

impl Default for Terrain {
    fn default() -> Self {
        Terrain { detail_layers_view_dist_ratio: 1.0 }
    }
}

/// The full `<Environment>` block.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Environment {
    pub fog: Fog,
    pub terrain: Terrain,
    pub env_state: EnvState,
    pub skybox: SkyBox,
    pub ocean: Ocean,
    pub ocean_animation: OceanAnimation,
    pub ice_scattering: IceScattering,
    pub moon: Moon,
    pub hdr: HdrSetup,
    pub lighting: Lighting,
}

impl Environment {
    pub fn sun_direction(&self, hour: f32) -> [f32; 3] {
        self.lighting.sun_direction(hour)
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// A parsed mission lighting block: `<Environment>` + `<TimeOfDay>`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MissionLighting {
    pub environment: Environment,
    pub time_of_day: TimeOfDay,
}

/// Parse the `<Environment>` and `<TimeOfDay>` sections of a mission XML.
/// Malformed XML terminates early without panicking; malformed attributes fall
/// back to their defaults.
pub fn parse_mission_lighting(xml: &[u8]) -> MissionLighting {
    let mut out = MissionLighting::default();
    out.time_of_day = TimeOfDay {
        time: 0.0,
        time_start: 0.0,
        time_end: 24.0,
        time_anim_speed: 0.0,
        variables: Vec::new(),
    };

    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut in_environment = false;
    let mut in_tod = false;
    let mut current: Option<TodVariable> = None;

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };

        match ev {
            Event::Start(e) => {
                let name = e.name();
                let name = name.as_ref();
                match name {
                    b"Environment" => in_environment = true,
                    b"TimeOfDay" => {
                        in_tod = true;
                        read_tod_attrs(&e, &mut out.time_of_day);
                    }
                    b"Variable" if in_tod => current = Some(variable_from(&e)),
                    b"Spline" => {
                        if let Some(v) = current.as_mut() {
                            if let Some(keys) = get_attr(&e, "Keys") {
                                v.spline = Some(Spline::new(parse_keys(&keys, v.value)));
                            }
                        }
                    }
                    _ if in_environment => parse_env_element(name, &e, &mut out.environment),
                    _ => {}
                }
            }
            Event::Empty(e) => {
                let name = e.name();
                let name = name.as_ref();
                match name {
                    b"TimeOfDay" => read_tod_attrs(&e, &mut out.time_of_day),
                    b"Variable" if in_tod => out.time_of_day.variables.push(variable_from(&e)),
                    b"Spline" => {
                        if let Some(v) = current.as_mut() {
                            if let Some(keys) = get_attr(&e, "Keys") {
                                v.spline = Some(Spline::new(parse_keys(&keys, v.value)));
                            }
                        }
                    }
                    _ if in_environment => parse_env_element(name, &e, &mut out.environment),
                    _ => {}
                }
            }
            Event::End(e) => {
                let name = e.name();
                let name = name.as_ref();
                match name {
                    b"Environment" => in_environment = false,
                    b"TimeOfDay" => in_tod = false,
                    b"Variable" if in_tod => {
                        if let Some(v) = current.take() {
                            out.time_of_day.variables.push(v);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        buf.clear();
    }

    out
}

fn read_tod_attrs(e: &BytesStart, tod: &mut TimeOfDay) {
    if let Some(v) = get_attr(e, "Time") {
        tod.time = v.parse().unwrap_or(tod.time);
    }
    if let Some(v) = get_attr(e, "TimeStart") {
        tod.time_start = v.parse().unwrap_or(tod.time_start);
    }
    if let Some(v) = get_attr(e, "TimeEnd") {
        tod.time_end = v.parse().unwrap_or(tod.time_end);
    }
    if let Some(v) = get_attr(e, "TimeAnimSpeed") {
        tod.time_anim_speed = v.parse().unwrap_or(tod.time_anim_speed);
    }
}

fn variable_from(e: &BytesStart) -> TodVariable {
    let name = get_attr(e, "Name").unwrap_or_default();
    // `Color` wins when present; otherwise the scalar `Value`.
    let value = if let Some(c) = get_attr(e, "Color") {
        Value::Color(parse_color(&c))
    } else {
        let s = get_attr(e, "Value").and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(0.0);
        Value::Scalar(s)
    };
    TodVariable { name, value, spline: None }
}

/// Parse a `Keys="t:v,t:v,..."` (scalar) or `t:(r:g:b),...` (color) list.
/// `base` selects the value type; malformed entries are skipped.
fn parse_keys(keys: &str, base: Value) -> Vec<Key> {
    let mut out = Vec::new();
    for entry in keys.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let Some((t, rest)) = entry.split_once(':') else { continue };
        let Ok(time) = t.trim().parse::<f32>() else { continue };
        let value = match base {
            Value::Scalar(_) => match rest.trim().parse::<f32>() {
                Ok(v) => Value::Scalar(v),
                Err(_) => continue,
            },
            Value::Color(_) => {
                let inner = rest.trim().trim_start_matches('(').trim_end_matches(')');
                let parts: Vec<f32> = inner
                    .split(':')
                    .filter_map(|p| p.trim().parse::<f32>().ok())
                    .collect();
                if parts.len() < 3 {
                    continue;
                }
                Value::Color([parts[0], parts[1], parts[2]])
            }
        };
        out.push(Key { time, value });
    }
    out
}

fn parse_env_element(name: &[u8], e: &BytesStart, env: &mut Environment) {
    match name {
        b"Fog" => {
            env.fog.view_distance = get_f32(e, "ViewDistance", env.fog.view_distance);
            env.fog.view_distance_low_spec = get_f32(e, "ViewDistanceLowSpec", env.fog.view_distance_low_spec);
            env.fog.ldr_global_dens_mult = get_f32(e, "LDRGlobalDensMult", env.fog.ldr_global_dens_mult);
            env.fog.use_fog_color_gradient = get_bool(e, "UseFogColorGradient", env.fog.use_fog_color_gradient);
        }
        b"Terrain" => {
            env.terrain.detail_layers_view_dist_ratio =
                get_f32(e, "DetailLayersViewDistRatio", env.terrain.detail_layers_view_dist_ratio);
        }
        b"EnvState" => {
            if let Some(v) = get_attr(e, "WindVector") {
                env.env_state.wind_vector = parse_vec3(&v, env.env_state.wind_vector);
            }
            env.env_state.show_terrain_surface = get_bool(e, "ShowTerrainSurface", env.env_state.show_terrain_surface);
            env.env_state.sun_shadows_min_spec = get_bool(e, "SunShadowsMinSpec", env.env_state.sun_shadows_min_spec);
            env.env_state.cloud_shadow_speed = get_f32(e, "CloudShadowSpeed", env.env_state.cloud_shadow_speed);
            env.env_state.use_layers_activation = get_bool(e, "UseLayersActivation", env.env_state.use_layers_activation);
            if let Some(v) = get_attr(e, "CloudShadowTexture") {
                env.env_state.cloud_shadow_texture = v;
            }
        }
        b"SkyBox" => {
            env.skybox.material = get_attr(e, "Material").unwrap_or_default();
            env.skybox.material_low_spec = get_attr(e, "MaterialLowSpec").unwrap_or_default();
            env.skybox.angle = get_f32(e, "Angle", env.skybox.angle);
            env.skybox.stretching = get_f32(e, "Stretching", env.skybox.stretching);
        }
        b"Ocean" => {
            env.ocean.material = get_attr(e, "Material").unwrap_or_default();
            if let Some(v) = get_attr(e, "FogColor") {
                env.ocean.fog_color = parse_color(&v);
            }
            env.ocean.fog_color_multiplier = get_f32(e, "FogColorMultiplier", env.ocean.fog_color_multiplier);
            env.ocean.fog_density = get_f32(e, "FogDensity", env.ocean.fog_density);
            env.ocean.caustic_depth = get_f32(e, "CausticDepth", env.ocean.caustic_depth);
            env.ocean.caustic_intensity = get_f32(e, "CausticIntensity", env.ocean.caustic_intensity);
            env.ocean.caustics_tilling = get_f32(e, "CausticsTilling", env.ocean.caustics_tilling);
        }
        b"OceanAnimation" => {
            env.ocean_animation.wind_direction = get_f32(e, "WindDirection", env.ocean_animation.wind_direction);
            env.ocean_animation.wind_speed = get_f32(e, "WindSpeed", env.ocean_animation.wind_speed);
            env.ocean_animation.waves_speed = get_f32(e, "WavesSpeed", env.ocean_animation.waves_speed);
            env.ocean_animation.waves_amount = get_f32(e, "WavesAmount", env.ocean_animation.waves_amount);
            env.ocean_animation.waves_size = get_f32(e, "WavesSize", env.ocean_animation.waves_size);
        }
        b"IceScattering" => {
            if let Some(v) = get_attr(e, "ScatteringDirection") {
                env.ice_scattering.scattering_direction =
                    parse_vec3(&v, env.ice_scattering.scattering_direction);
            }
        }
        b"Moon" => {
            env.moon.latitude = get_f32(e, "Latitude", env.moon.latitude);
            env.moon.longitude = get_f32(e, "Longitude", env.moon.longitude);
            env.moon.size = get_f32(e, "Size", env.moon.size);
            env.moon.texture = get_attr(e, "Texture").unwrap_or_default();
        }
        b"HDRSetup" => {
            if let Some(v) = get_attr(e, "BloomColor") {
                env.hdr.bloom_color = parse_color(&v);
            }
            env.hdr.bloom_mul = get_f32(e, "BloomMul", env.hdr.bloom_mul);
            if let Some(v) = get_attr(e, "FlaresColor") {
                env.hdr.flares_color = parse_color(&v);
            }
            env.hdr.flares_mul = get_f32(e, "FlaresMul", env.hdr.flares_mul);
            if let Some(v) = get_attr(e, "StreaksColor") {
                env.hdr.streaks_color = parse_color(&v);
            }
            env.hdr.streaks_mul = get_f32(e, "StreaksMul", env.hdr.streaks_mul);
            env.hdr.streak_scale = get_f32(e, "StreakScale", env.hdr.streak_scale);
        }
        b"Lighting" => {
            env.lighting.sun_rotation = get_f32(e, "SunRotation", env.lighting.sun_rotation);
            env.lighting.sun_height = get_f32(e, "SunHeight", env.lighting.sun_height);
            env.lighting.algorithm = get_i32(e, "Algorithm", env.lighting.algorithm);
            env.lighting.lighting = get_bool(e, "Lighting", env.lighting.lighting);
            env.lighting.shadows = get_bool(e, "Shadows", env.lighting.shadows);
            env.lighting.shadow_intensity = get_f32(e, "ShadowIntensity", env.lighting.shadow_intensity);
            env.lighting.il_quality = get_i32(e, "ILQuality", env.lighting.il_quality);
            env.lighting.hemi_sample_quality =
                get_i32(e, "HemiSamplQuality", env.lighting.hemi_sample_quality);
            env.lighting.longitude = get_f32(e, "Longitude", env.lighting.longitude);
            env.lighting.dawn_time = get_f32(e, "DawnTime", env.lighting.dawn_time);
            env.lighting.dawn_duration = get_f32(e, "DawnDuration", env.lighting.dawn_duration);
            env.lighting.dusk_time = get_f32(e, "DuskTime", env.lighting.dusk_time);
            env.lighting.dusk_duration = get_f32(e, "DuskDuration", env.lighting.dusk_duration);
            if let Some(v) = get_attr(e, "SunVector") {
                env.lighting.sun_vector = parse_vec3(&v, env.lighting.sun_vector);
            }
        }
        _ => {}
    }
}

fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

fn get_f32(e: &BytesStart, name: &str, default: f32) -> f32 {
    get_attr(e, name).and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(default)
}

fn get_i32(e: &BytesStart, name: &str, default: i32) -> i32 {
    get_attr(e, name).and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(default)
}

fn get_bool(e: &BytesStart, name: &str, default: bool) -> bool {
    match get_attr(e, name) {
        Some(s) => match s.trim() {
            "1" | "true" | "True" => true,
            "0" | "false" | "False" => false,
            _ => default,
        },
        None => default,
    }
}

fn parse_color(s: &str) -> [f32; 3] {
    parse_vec3(s, [0.0, 0.0, 0.0])
}

fn parse_vec3(s: &str, default: [f32; 3]) -> [f32; 3] {
    let v: Vec<f32> = s
        .replace(' ', "")
        .split(',')
        .filter(|p| !p.is_empty())
        .filter_map(|p| p.parse::<f32>().ok())
        .collect();
    if v.len() >= 3 {
        [v[0], v[1], v[2]]
    } else {
        default
    }
}

// ---------------------------------------------------------------------------
// lights_*.lights (binary)
// ---------------------------------------------------------------------------

/// Header of a `lights_*.lights` file. The format is `ULLI`, a flags word,
/// `count`, four extra header words and then fixed-size records (the record
/// payload is not decoded yet; see `notes/level-format.md` §5.5).
#[derive(Debug, Clone, PartialEq)]
pub struct LightsFile {
    pub flags: u32,
    pub count: u32,
    pub extra: [u32; 4],
    /// Raw record area (`count` records of `record_size` bytes).
    pub records: Vec<u8>,
    pub record_size: usize,
}

/// Parse a `lights_*.lights` file. Returns `None` when the magic is wrong or
/// the buffer is truncated.
pub fn parse_lights(data: &[u8]) -> Option<LightsFile> {
    if data.len() < 0x1c || &data[0..4] != b"ULLI" {
        return None;
    }
    let rd = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    let flags = rd(4);
    let count = rd(8);
    let extra = [rd(0xc), rd(0x10), rd(0x14), rd(0x18)];
    // Records start after the zero-padded header; the observed header is the
    // 4-word block at 0..0x1c followed by zero padding up to 0x80.
    let start = 0x80.min(data.len());
    let records = data[start.min(data.len())..].to_vec();
    let record_size = if count > 0 && records.len() >= count as usize {
        records.len() / count as usize
    } else {
        0
    };
    Some(LightsFile { flags, count, extra, records, record_size })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<Mission Time="11:55">
 <Environment>
  <Fog ViewDistance="6000" ViewDistanceLowSpec="1000" LDRGlobalDensMult="1" UseFogColorGradient="1"/>
  <Terrain DetailLayersViewDistRatio="1"/>
  <EnvState WindVector="1,0,0" ShowTerrainSurface="1" SunShadowsMinSpec="1" CloudShadowTexture="" CloudShadowSpeed="0.0" UseLayersActivation="1"/>
  <SkyBox Material="Materials/Sky/Sky" MaterialLowSpec="Materials/Sky/Sky" Angle="0" Stretching="0.5"/>
  <Ocean Material="Materials/ocean/ocean_medium_waves" FogColor="32,112,208" FogColorMultiplier="0.001" FogDensity="0.5"/>
  <Moon Latitude="240.0" Longitude="45" Size="0.5" Texture="Textures/Skys/Night/half_moon.dds"/>
  <HDRSetup BloomColor="255,255,255" BloomMul="1.0" FlaresMul="0.1" StreaksMul="0.33" StreakScale="1.0"/>
  <Lighting SunRotation="205" SunHeight="80" Algorithm="3" Lighting="1" Shadows="1" ShadowIntensity="100" ILQuality="1" HemiSamplQuality="9" Longitude="36" DawnTime="375" DawnDuration="0" DuskTime="340" DuskDuration="0" SunVector="-0.25357094,-0.54378468,-0.79999989"/>
 </Environment>
 <TimeOfDay Time="11.916878" TimeStart="0" TimeEnd="23.983334" TimeAnimSpeed="0.0034">
  <Variable Name="Sun color multiplier" Value="12.467134">
   <Spline Keys="0:3,0.5:12.5,1:3,"/>
  </Variable>
  <Variable Name="Sun color" Color="0.99139392,0.6061852,0.24370697">
   <Spline Keys="0:(0.1:0.2:0.3),0.5:(0.991393:0.605551:0.239264),1:(0.1:0.2:0.3),"/>
  </Variable>
  <Variable Name="Constant" Value="4.5"/>
 </TimeOfDay>
 <Objects/>
</Mission>"#;

    #[test]
    fn parses_environment_and_tod() {
        let m = parse_mission_lighting(SAMPLE.as_bytes());
        let e = &m.environment;
        assert_eq!(e.fog.view_distance, 6000.0);
        assert!(e.fog.use_fog_color_gradient);
        assert_eq!(e.env_state.wind_vector, [1.0, 0.0, 0.0]);
        assert_eq!(e.skybox.material, "Materials/Sky/Sky");
        assert_eq!(e.ocean.fog_color, [32.0, 112.0, 208.0]);
        assert_eq!(e.moon.latitude, 240.0);
        assert_eq!(e.hdr.bloom_color, [255.0, 255.0, 255.0]);
        assert_eq!(e.lighting.sun_rotation, 205.0);
        assert_eq!(e.lighting.longitude, 36.0);
        assert_eq!(e.lighting.sun_vector, [-0.25357094, -0.54378468, -0.79999989]);

        let t = &m.time_of_day;
        assert!((t.time - 11.916878).abs() < 1e-5);
        assert_eq!(t.time_end, 23.983334);
        assert_eq!(t.variables.len(), 3);
        assert_eq!(t.variables[0].name, "Sun color multiplier");
        assert_eq!(t.variables[0].spline.as_ref().unwrap().keys.len(), 3);
        assert_eq!(t.variables[1].value, Value::Color([0.99139392, 0.6061852, 0.24370697]));
        assert!(t.variables[2].spline.is_none());
    }

    #[test]
    fn samples_on_keys_exactly() {
        let t = parse_mission_lighting(SAMPLE.as_bytes()).time_of_day;
        let s = t.sample(12.0); // 0.5 normalized
        assert_eq!(s.scalar("Sun color multiplier"), Some(12.5));
        assert_eq!(s.color("Sun color"), Some([0.991393, 0.605551, 0.239264]));
        // Constant variable has no spline.
        assert_eq!(s.scalar("Constant"), Some(4.5));
    }

    #[test]
    fn interpolates_linearly_between_keys() {
        let t = parse_mission_lighting(SAMPLE.as_bytes()).time_of_day;
        // 6h -> 0.25, halfway between 0:3 and 0.5:12.5.
        let s = t.sample(6.0);
        let v = s.scalar("Sun color multiplier").unwrap();
        assert!((v - 7.75).abs() < 1e-4, "got {v}");
        // 18h -> 0.75, halfway between 0.5:12.5 and 1:3 (descending).
        let v = t.sample(18.0).scalar("Sun color multiplier").unwrap();
        assert!((v - 7.75).abs() < 1e-4, "got {v}");
        // Color components interpolate independently.
        let c = t.sample(6.0).color("Sun color").unwrap();
        assert!((c[0] - (0.1 + 0.891393 * 0.5)).abs() < 1e-4, "got {c:?}");
    }

    #[test]
    fn sun_below_horizon_at_night() {
        let e = parse_mission_lighting(SAMPLE.as_bytes()).environment;
        let light = &e.lighting;
        // Noon: light points down.
        let noon = light.sun_direction(12.0);
        assert!(noon[2] < 0.0, "noon z={}", noon[2]);
        // Midnight: sun is below the horizon, light direction points up.
        let night = light.sun_direction(0.0);
        assert!(night[2] > 0.0, "night z={}", night[2]);
        assert!(light.is_night(0.0));
        assert!(!light.is_night(12.0));
        // Unit length.
        let len = (noon[0] * noon[0] + noon[1] * noon[1] + noon[2] * noon[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-4, "len={len}");
    }

    #[test]
    fn sun_model_reproduces_authored_vector() {
        // The authored SunVector collapses to azimuth SunRotation and altitude
        // 53.13 deg with the level's Longitude (36 -> max altitude 54).
        let e = parse_mission_lighting(SAMPLE.as_bytes()).environment;
        let got = e.lighting.sun_direction(11.916878);
        let want = e.lighting.sun_vector;
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() < 0.02, "component {i}: got {} want {}", got[i], want[i]);
        }
    }

    #[test]
    fn malformed_input_is_safe() {
        let m = parse_mission_lighting(b"<not xml");
        assert!(m.time_of_day.variables.is_empty());
        let m = parse_mission_lighting(b"");
        assert_eq!(m.time_of_day.time_end, 24.0);
    }

    #[test]
    fn spline_clamps_and_handles_unsorted_keys() {
        let s = Spline::new(vec![
            Key { time: 0.5, value: Value::Scalar(10.0) },
            Key { time: 0.0, value: Value::Scalar(0.0) },
            Key { time: 1.0, value: Value::Scalar(20.0) },
        ]);
        assert_eq!(s.sample(-1.0), Some(Value::Scalar(0.0)));
        assert_eq!(s.sample(0.25), Some(Value::Scalar(5.0)));
        assert_eq!(s.sample(2.0), Some(Value::Scalar(20.0)));
    }

    #[test]
    fn lights_header() {
        let mut d = Vec::new();
        d.extend_from_slice(b"ULLI");
        d.extend_from_slice(&7u32.to_le_bytes());
        d.extend_from_slice(&2356u32.to_le_bytes());
        for v in [1220u32, 52, 91, 0] {
            d.extend_from_slice(&v.to_le_bytes());
        }
        d.resize(0x80 + 2356 * 46, 0);
        let l = parse_lights(&d).unwrap();
        assert_eq!(l.flags, 7);
        assert_eq!(l.count, 2356);
        assert_eq!(l.extra, [1220, 52, 91, 0]);
        assert_eq!(l.record_size, 46);
        assert!(parse_lights(b"NOPE").is_none());
    }

    #[test]
    fn integration_real_level_if_present() {
        let path = std::env::var("SOD_MISSION_XML").unwrap_or_else(|_| {
            "/home/battlestation/Desktop/sod-re/extracted-game/level/mission_mission0.xml".to_string()
        });
        let Ok(xml) = std::fs::read(&path) else {
            eprintln!("skipping integration test: {path} not found");
            return;
        };
        let m = parse_mission_lighting(&xml);
        assert!(m.environment.lighting.sun_rotation != 0.0, "SunRotation parsed");
        assert!(m.time_of_day.variables.len() > 40, "{} variables", m.time_of_day.variables.len());
        assert!(m.time_of_day.variable("Sun color").is_some());
        assert!(m.time_of_day.variable("Fog color").is_some());
        // Night is below the horizon on the real level too.
        assert!(m.environment.lighting.is_night(0.0));
        assert!(!m.environment.lighting.is_night(12.0));
    }
}
