//! Per-class vehicle XML: physics, damage and ArcadeWheeled movement params.
//!
//! Every drivable car ships a document under
//! `Scripts/Entities/Vehicles/Implementations/Xml/<class>.xml` (VFS
//! `gamedata/scripts/entities/vehicles/implementations/xml/`). The schema is
//! described by `Scripts/Entities/Vehicles/def_vehicle.xml`; the editor
//! defaults live in `veed_defaults.xml`. This module recovers the subset the
//! native simulation needs:
//!
//! * `<Physics>` / `<Simulation>` — mass, damping, sleep thresholds.
//! * `<Damages>` and `<Components>` — hull health, per-damage-type multipliers
//!   and the collision-damage threshold. The shipped values are the same
//!   numbers the engine reader uses (`CVehicle::` damage code).
//! * `<MovementParams><ArcadeWheeled>` — the whole arcade driving model.
//!   Class3 registers its own `"ArcadeWheeled"` movement class with the engine
//!   vehicle system at `0x01552970`; the constructor factory at `0x015527c0`
//!   allocates a `0x878`-byte object and chains into `0x01672070`. The
//!   parameter names parsed here are exactly the attribute names in the XML.
//! * `<Parts><SubPartWheel>` — one wheel per `<SubPartWheel>` element;
//!   `driving="1"` marks a powered wheel. The `<WheelMaster>` template is kept
//!   separately.
//! * `<Helpers>` — named attachment points; the `wheel*` helpers give the
//!   wheelbase/track used by the steering model.
//!
//! Only the human-readable XML is parsed here; no engine binary is involved.
//! Parsing is total: garbage yields `VehicleSetup::default()`.

use std::collections::BTreeMap;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// Engine RPM when the engine is idle (from `<WheeledLegacy>`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheeledLegacy {
    pub damping: f32,
    pub engine_idle_rpm: f32,
    pub engine_max_rpm: f32,
    pub engine_min_rpm: f32,
    pub stabilizer: f32,
    pub max_time_step: f32,
    pub min_energy: f32,
    pub susp_damping_min: f32,
    pub susp_damping_max: f32,
    pub susp_damping_max_speed: f32,
}

impl Default for WheeledLegacy {
    fn default() -> Self {
        WheeledLegacy {
            damping: 0.0,
            engine_idle_rpm: 0.0,
            engine_max_rpm: 0.0,
            engine_min_rpm: 0.0,
            stabilizer: 0.0,
            max_time_step: 0.0,
            min_energy: 0.0,
            susp_damping_min: 0.0,
            susp_damping_max: 0.0,
            susp_damping_max_speed: 0.0,
        }
    }
}

/// `<Handling><Power>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Power {
    /// Engine acceleration (`m/s²`).
    pub acceleration: f32,
    /// Coast deceleration (`m/s²`).
    pub decceleration: f32,
    /// Forward top speed (`m/s`).
    pub top_speed: f32,
    /// Reverse top speed (`m/s`).
    pub reverse_speed: f32,
}

/// `<Handling><WheelSpin>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WheelSpin {
    pub grip1: f32,
    pub grip2: f32,
    pub grip_recover_speed: f32,
    pub accel_multiplier1: f32,
    pub accel_multiplier2: f32,
}

/// `<Handling><HandBrake>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HandBrake {
    pub decceleration: f32,
    pub decceleration_power_lock: f32,
    pub lock_back: bool,
    pub lock_front: bool,
    pub front_friction_scale: f32,
    pub back_friction_scale: f32,
    pub ang_correction_scale: f32,
}

/// `<Handling><Brakes>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Brakes {
    pub decceleration: f32,
    pub lock_decceleration: f32,
    pub lock_min_speed: f32,
    pub lock_min_pressure: f32,
    pub front_friction_scale: f32,
    pub back_friction_scale: f32,
}

/// `<Handling><SpeedReduction>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpeedReduction {
    pub reduction_amount: f32,
    pub reduction_rate: f32,
}

/// `<Handling><Friction>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Friction {
    pub back: f32,
    pub front: f32,
    pub offset: f32,
}

/// `<Handling><Correction>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Correction {
    pub lateral_spring: f32,
    pub ang_spring: f32,
}

/// `<Handling><Compression>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Compression {
    pub friction_boost: f32,
    pub friction_boost_hand_brake: f32,
}

/// `<Handling><Inertia>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Inertia {
    pub radius: f32,
}

/// `<FakeGears>`.
#[derive(Debug, Clone, PartialEq)]
pub struct FakeGears {
    pub min_change_up_time: f32,
    pub min_change_down_time: f32,
    /// Forward gear ratios as fractions of top speed.
    pub ratios: Vec<f32>,
}

impl Default for FakeGears {
    fn default() -> Self {
        FakeGears {
            min_change_up_time: 0.0,
            min_change_down_time: 0.0,
            ratios: Vec::new(),
        }
    }
}

/// `<AirDamp>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AirDamp {
    pub damp_angle: [f32; 3],
    pub damp_ang_vel: [f32; 3],
}

/// The full `<ArcadeWheeled>` movement block.
#[derive(Debug, Clone, PartialEq)]
pub struct ArcadeWheeled {
    pub gravity: f32,
    pub gravity_in_air: f32,

    pub v_max_steer_max: f32,
    pub steer_speed_min: f32,
    pub steer_speed: f32,
    pub steer_speed_scale_min: f32,
    pub steer_speed_scale: f32,
    pub v0_steer_max: f32,
    pub kv_steer_max: f32,

    pub rv_max_steer_max: f32,
    pub reverse_steer_speed_min: f32,
    pub reverse_steer_speed: f32,
    pub reverse_steer_speed_scale_min: f32,
    pub reverse_steer_speed_scale: f32,
    pub rv0_steer_max: f32,
    pub rkv_steer_max: f32,

    pub steer_relaxation: f32,
    pub pedal_limit_max: f32,
    pub rpm_relax_speed: f32,
    pub rpm_interp_speed: f32,
    pub rpm_gear_shift_speed: f32,

    pub power: Power,
    pub wheel_spin: WheelSpin,
    pub hand_brake: HandBrake,
    pub brakes: Brakes,
    pub speed_reduction: SpeedReduction,
    pub friction: Friction,
    pub correction: Correction,
    pub compression: Compression,
    pub inertia: Inertia,
    pub fake_gears: FakeGears,
    pub wheeled_legacy: WheeledLegacy,
    pub air_damp: AirDamp,
}

impl Default for ArcadeWheeled {
    fn default() -> Self {
        ArcadeWheeled {
            gravity: -9.8,
            gravity_in_air: -9.0,
            v_max_steer_max: 0.0,
            steer_speed_min: 0.0,
            steer_speed: 0.0,
            steer_speed_scale_min: 1.0,
            steer_speed_scale: 1.0,
            v0_steer_max: 0.0,
            kv_steer_max: 0.0,
            rv_max_steer_max: 0.0,
            reverse_steer_speed_min: 0.0,
            reverse_steer_speed: 0.0,
            reverse_steer_speed_scale_min: 1.0,
            reverse_steer_speed_scale: 1.0,
            rv0_steer_max: 0.0,
            rkv_steer_max: 0.0,
            steer_relaxation: 0.0,
            pedal_limit_max: 0.0,
            rpm_relax_speed: 0.0,
            rpm_interp_speed: 0.0,
            rpm_gear_shift_speed: 0.0,
            power: Power::default(),
            wheel_spin: WheelSpin::default(),
            hand_brake: HandBrake::default(),
            brakes: Brakes::default(),
            speed_reduction: SpeedReduction::default(),
            friction: Friction::default(),
            correction: Correction::default(),
            compression: Compression::default(),
            inertia: Inertia::default(),
            fake_gears: FakeGears::default(),
            wheeled_legacy: WheeledLegacy::default(),
            air_damp: AirDamp::default(),
        }
    }
}

impl ArcadeWheeled {
    /// Maximum steering angle in **radians** at `speed` (`m/s`), following the
    /// recovered curve `v0SteerMax - min(|v|/vMaxSteerMax,1) * kvSteerMax`
    /// (degrees). Reverse uses the `rv*` pair when `speed < 0`.
    pub fn max_steer(&self, speed: f32) -> f32 {
        let fwd = speed >= 0.0;
        let (v0, k, vmax) = if fwd {
            (self.v0_steer_max, self.kv_steer_max, self.v_max_steer_max)
        } else {
            (self.rv0_steer_max, self.rkv_steer_max, self.rv_max_steer_max)
        };
        if vmax <= 0.0 || v0 == 0.0 {
            return v0.to_radians().clamp(0.0, std::f32::consts::FRAC_PI_2);
        }
        let t = (speed.abs() / vmax).min(1.0);
        (v0 - t * k).to_radians().clamp(0.0, std::f32::consts::FRAC_PI_2)
    }

    /// Engine acceleration multiplier from `<WheelSpin>`: `accelMultiplier1` at
    /// rest easing to `accelMultiplier2` at top speed.
    pub fn accel_multiplier(&self, speed: f32) -> f32 {
        let top = self.power.top_speed.max(1e-3);
        let t = (speed.abs() / top).clamp(0.0, 1.0);
        self.wheel_spin.accel_multiplier1
            + (self.wheel_spin.accel_multiplier2 - self.wheel_spin.accel_multiplier1) * t
    }
}

/// One `<SubPartWheel>` (physical wheel / suspension ray).
#[derive(Debug, Clone, PartialEq)]
pub struct Wheel {
    pub name: String,
    pub axle: u8,
    /// Powered wheel (`driving="1"`).
    pub driving: bool,
    pub can_brake: bool,
    /// Wheel body density.
    pub density: f32,
    /// Suspension damping (negative per the shipped data).
    pub damping: f32,
    /// Maximum (stretched) suspension length.
    pub len_max: f32,
    pub susp_length: f32,
    pub stiffness: f32,
    pub min_friction: f32,
    pub max_friction: f32,
    pub slip_friction_mod: f32,
    pub rim_radius: f32,
    pub torque_scale: f32,
}

impl Wheel {
    fn new(name: String) -> Wheel {
        Wheel {
            name,
            axle: 0,
            driving: false,
            can_brake: false,
            density: 0.0,
            damping: 0.0,
            len_max: 0.0,
            susp_length: 0.0,
            stiffness: 0.0,
            min_friction: 0.0,
            max_friction: 0.0,
            slip_friction_mod: 0.0,
            rim_radius: 0.0,
            torque_scale: 1.0,
        }
    }
}

/// `<Part class="MassBox"><MassBox .../></Part>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MassBox {
    pub mass: f32,
    pub size: [f32; 3],
    /// Z offset applied while driving (`drivingOffset`).
    pub driving_offset: f32,
}

/// `<Physics>` + `<Simulation>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehiclePhysics {
    pub mass: f32,
    pub pushable: bool,
    pub retain_gravity: bool,
    pub damping: f32,
    pub max_time_step: f32,
    pub min_energy: f32,
    pub water_density: f32,
    pub water_resistance: f32,
}

impl Default for VehiclePhysics {
    fn default() -> Self {
        VehiclePhysics {
            mass: 0.0,
            pushable: false,
            retain_gravity: true,
            damping: 0.0,
            max_time_step: 0.0,
            min_energy: 0.0,
            water_density: 0.0,
            water_resistance: 0.0,
        }
    }
}

/// One damageable `<Component>`.
#[derive(Debug, Clone, PartialEq)]
pub struct DamageComponent {
    pub name: String,
    /// `damageMax` — component health.
    pub damage_max: f32,
    /// Fraction of damage passed to the hull.
    pub hull_affection: f32,
    pub use_bounds_from_parts: bool,
}

/// `<Damages>` block plus the always-present `Hull` component.
#[derive(Debug, Clone, PartialEq)]
pub struct VehicleDamage {
    pub coll_damage_threshold: f32,
    pub ground_collision_min_mult: f32,
    pub ground_collision_max_mult: f32,
    pub ground_collision_min_speed: f32,
    pub ground_collision_max_speed: f32,
    pub vehicle_collision_destruction_speed: f32,
    /// `damageType -> multiplier` from the top-level `<DamageMultipliers>`.
    pub multipliers: BTreeMap<String, f32>,
    pub components: Vec<DamageComponent>,
}

impl Default for VehicleDamage {
    fn default() -> Self {
        VehicleDamage {
            coll_damage_threshold: 0.0,
            ground_collision_min_mult: 0.0,
            ground_collision_max_mult: 0.0,
            ground_collision_min_speed: 0.0,
            ground_collision_max_speed: 0.0,
            vehicle_collision_destruction_speed: 0.0,
            multipliers: BTreeMap::new(),
            components: Vec::new(),
        }
    }
}

impl VehicleDamage {
    /// Top-level multiplier for a damage type (1.0 when unspecified).
    pub fn multiplier(&self, damage_type: &str) -> f32 {
        self.multipliers
            .get(&damage_type.to_ascii_lowercase())
            .copied()
            .unwrap_or(1.0)
    }

    /// Hull component health, if a component named `Hull` exists.
    pub fn hull_damage_max(&self) -> Option<f32> {
        self.components
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case("hull"))
            .map(|c| c.damage_max)
    }
}

/// Everything recovered from one per-class vehicle XML.
#[derive(Debug, Clone, PartialEq)]
pub struct VehicleSetup {
    pub name: String,
    pub action_map: String,
    /// Root `type` attr (1 = land vehicle in the shipped data).
    pub kind: u8,
    pub physics: VehiclePhysics,
    /// Mass declared on the animated `body` part.
    pub body_mass: f32,
    pub mass_box: Option<MassBox>,
    pub damage: VehicleDamage,
    pub movement: ArcadeWheeled,
    pub wheels: Vec<Wheel>,
    /// `<WheelMaster><SubPartWheel>` template, when present.
    pub wheel_master: Option<Wheel>,
    pub helpers: BTreeMap<String, [f32; 3]>,
    pub seat_count: u8,
}

impl Default for VehicleSetup {
    fn default() -> Self {
        VehicleSetup {
            name: String::new(),
            action_map: String::new(),
            kind: 0,
            physics: VehiclePhysics::default(),
            body_mass: 0.0,
            mass_box: None,
            damage: VehicleDamage::default(),
            movement: ArcadeWheeled::default(),
            wheels: Vec::new(),
            wheel_master: None,
            helpers: BTreeMap::new(),
            seat_count: 0,
        }
    }
}

impl VehicleSetup {
    /// A neutral ArcadeWheeled car used when no per-class XML is available.
    ///
    /// The numbers are the shipped `hatchback_modern` values (mass 5000 kg,
    /// 14 m/s² / 24.59 m/s top speed, four wheels with front-wheel drive).
    pub fn fallback() -> VehicleSetup {
        let mut movement = ArcadeWheeled {
            gravity: -9.8,
            gravity_in_air: -9.0,
            v_max_steer_max: 18.0,
            steer_speed_min: 100.0,
            steer_speed: 80.0,
            v0_steer_max: 60.0,
            kv_steer_max: 50.0,
            rv_max_steer_max: 16.0,
            reverse_steer_speed_min: 100.0,
            reverse_steer_speed: 80.0,
            rv0_steer_max: 30.0,
            rkv_steer_max: -60.0,
            steer_relaxation: 200.0,
            rpm_relax_speed: 1.0,
            rpm_interp_speed: 2.5,
            rpm_gear_shift_speed: 2.0,
            power: Power {
                acceleration: 14.0,
                decceleration: 1.0,
                top_speed: 24.59,
                reverse_speed: 16.0,
            },
            wheel_spin: WheelSpin {
                grip1: 0.85,
                grip2: 1.6,
                grip_recover_speed: 2.0,
                accel_multiplier1: 1.4,
                accel_multiplier2: 0.1,
            },
            hand_brake: HandBrake {
                decceleration: 14.0,
                decceleration_power_lock: 10.0,
                lock_back: true,
                lock_front: true,
                front_friction_scale: 1.0,
                back_friction_scale: 0.001,
                ang_correction_scale: 1.0,
            },
            brakes: Brakes {
                decceleration: 16.0,
                lock_decceleration: 26.0,
                lock_min_speed: 0.0,
                lock_min_pressure: 0.9,
                front_friction_scale: 1.0,
                back_friction_scale: 0.1,
            },
            speed_reduction: SpeedReduction {
                reduction_amount: 0.1,
                reduction_rate: 0.1,
            },
            friction: Friction {
                back: 3.0,
                front: 3.0,
                offset: -1.3,
            },
            correction: Correction {
                lateral_spring: 4.0,
                ang_spring: 2.0,
            },
            compression: Compression {
                friction_boost: 1.5,
                friction_boost_hand_brake: 1.5,
            },
            inertia: Inertia { radius: 4.0 },
            fake_gears: FakeGears {
                min_change_up_time: 0.7,
                min_change_down_time: 0.5,
                ratios: Vec::new(),
            },
            wheeled_legacy: WheeledLegacy {
                damping: 0.11,
                engine_idle_rpm: 1.0,
                engine_max_rpm: 5000.0,
                engine_min_rpm: 1.0,
                stabilizer: 0.1,
                max_time_step: 0.05,
                min_energy: 0.09,
                susp_damping_min: -0.3,
                susp_damping_max: -1.0,
                susp_damping_max_speed: 10.0,
            },
            ..ArcadeWheeled::default()
        };
        movement.fake_gears.ratios = vec![0.5, 0.8, 0.9];

        let mut setup = VehicleSetup {
            name: "Hatchback_modern".into(),
            action_map: "landvehicle".into(),
            kind: 1,
            body_mass: 5000.0,
            mass_box: Some(MassBox {
                mass: 1000.0,
                size: [1.25, 2.85, 0.85],
                driving_offset: -2.5,
            }),
            movement,
            ..VehicleSetup::default()
        };
        setup.damage.coll_damage_threshold = 10.0;
        setup.damage.components.push(DamageComponent {
            name: "Hull".into(),
            damage_max: 20000.0,
            hull_affection: 1.0,
            use_bounds_from_parts: true,
        });
        let wheel = |name: &str, axle: u8, driving: bool| Wheel {
            name: name.into(),
            axle,
            driving,
            can_brake: false,
            density: 80.0,
            damping: -0.8,
            len_max: 0.15,
            susp_length: 0.025,
            stiffness: 0.0,
            min_friction: 1.0,
            max_friction: 1.0,
            slip_friction_mod: 0.3,
            rim_radius: 0.1,
            torque_scale: 1.0,
        };
        setup.wheels = vec![
            wheel("wheel1", 0, true),
            wheel("wheel2", 1, false),
            wheel("wheel3", 0, true),
            wheel("wheel4", 1, false),
        ];
        setup
    }

    /// Effective mass: the animated body mass, falling back to `<Physics mass>`,
    /// plus the mass box when one is declared. CryEngine treats the mass box as a
    /// secondary collision proxy with its own mass, so both contribute.
    pub fn mass(&self) -> f32 {
        let body = if self.body_mass > 0.0 {
            self.body_mass
        } else {
            self.physics.mass
        };
        let box_mass = self.mass_box.map(|m| m.mass).unwrap_or(0.0);
        let total = body + box_mass;
        if total > 0.0 {
            total
        } else {
            1.0
        }
    }

    /// Number of powered wheels.
    pub fn driven_wheel_count(&self) -> usize {
        self.wheels.iter().filter(|w| w.driving).count()
    }

    /// Wheelbase (m) from the `wheelFront*` / `wheelRear*` helpers when present.
    pub fn wheelbase(&self) -> Option<f32> {
        let f = self.wheel_helper_y(true)?;
        let r = self.wheel_helper_y(false)?;
        Some((f - r).abs())
    }

    /// Track width (m) from the left/right wheel helpers when present.
    pub fn track_width(&self) -> Option<f32> {
        let l = self.helper("wheelFrontLeft")?[0];
        let r = self.helper("wheelFrontRight")?[0];
        Some((l - r).abs())
    }

    fn wheel_helper_y(&self, front: bool) -> Option<f32> {
        let names: [&str; 2] = if front {
            ["wheelFrontLeft", "wheelFrontRight"]
        } else {
            ["wheelRearLeft", "wheelRearRight"]
        };
        names.iter().find_map(|n| self.helper(n).map(|p| p[1]))
    }

    fn helper(&self, name: &str) -> Option<[f32; 3]> {
        self.helpers.get(name).copied()
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a per-class vehicle XML. Returns `default()` for empty/garbage input
/// (never `None`, so callers can always fall back to [`VehicleSetup::fallback`]).
pub fn parse_vehicle_xml(xml: &[u8]) -> VehicleSetup {
    let xml = xml.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut setup = VehicleSetup::default();
    let mut stack: Vec<String> = Vec::new();
    let mut cur_component: Option<usize> = None;
    let mut cur_part_name: Option<String> = None;
    let mut cur_mass_box_mass: Option<f32> = None;
    let mut in_top_helpers = false;
    let mut in_top_seats = false;
    let mut wheel_master = false;

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };
        let is_empty = matches!(&ev, Event::Empty(_));
        match ev {
            Event::Start(e) | Event::Empty(e) => {
                let name = local_name(&e);
                let a = attr_map(&e);
                match name.as_str() {
                    "vehicle" => {
                        setup.name = attr(&a, "name").unwrap_or("").to_string();
                        setup.action_map = attr(&a, "actionmap").unwrap_or("").to_string();
                        setup.kind = u8_of(attr(&a, "type"));
                    }
                    "physics" => {
                        setup.physics.pushable = bool_of(attr(&a, "pushable"));
                        setup.physics.retain_gravity = bool_of(attr(&a, "retaingravity"));
                        setup.physics.mass = f32_of(attr(&a, "mass"));
                        setup.physics.damping = f32_of(attr(&a, "damping"));
                    }
                    "buoyancy" => {
                        setup.physics.water_density = f32_of(attr(&a, "waterdensity"));
                        setup.physics.water_resistance = f32_of(attr(&a, "waterresistance"));
                    }
                    "simulation" => {
                        setup.physics.max_time_step = f32_of(attr(&a, "maxtimestep"));
                        setup.physics.min_energy = f32_of(attr(&a, "minenergy"));
                    }
                    "damages" => {
                        setup.damage.coll_damage_threshold = f32_of(attr(&a, "colldamagethreshold"));
                        setup.damage.ground_collision_min_mult =
                            f32_of(attr(&a, "groundcollisionminmult"));
                        setup.damage.ground_collision_max_mult =
                            f32_of(attr(&a, "groundcollisionmaxmult"));
                        setup.damage.ground_collision_min_speed =
                            f32_of(attr(&a, "groundcollisionminspeed"));
                        setup.damage.ground_collision_max_speed =
                            f32_of(attr(&a, "groundcollisionmaxspeed"));
                        setup.damage.vehicle_collision_destruction_speed =
                            f32_of(attr(&a, "vehiclecollisiondestructionspeed"));
                    }
                    "damagemultiplier" => {
                        let ty = attr(&a, "damagetype").unwrap_or("").to_ascii_lowercase();
                        let mult = f32_of(attr(&a, "multiplier"));
                        if !ty.is_empty() && cur_component.is_none() {
                            setup.damage.multipliers.insert(ty, mult);
                        }
                    }
                    "component" => {
                        let c = DamageComponent {
                            name: attr(&a, "name").unwrap_or("").to_string(),
                            damage_max: f32_of(attr(&a, "damagemax")),
                            hull_affection: f32_of(attr(&a, "hullaffection")),
                            use_bounds_from_parts: bool_of(attr(&a, "useboundsfromparts")),
                        };
                        setup.damage.components.push(c);
                        if !is_empty {
                            cur_component = Some(setup.damage.components.len() - 1);
                        }
                    }
                    "part" => {
                        let class = attr(&a, "class").unwrap_or("");
                        let pname = attr(&a, "name").unwrap_or("").to_string();
                        if class.eq_ignore_ascii_case("animated")
                            && pname.eq_ignore_ascii_case("body")
                        {
                            setup.body_mass = f32_of(attr(&a, "mass"));
                        }
                        // The mass box declares its mass on the `<Part>` wrapper;
                        // the child `<MassBox>` only carries size/offset.
                        if class.eq_ignore_ascii_case("massbox") {
                            cur_mass_box_mass = Some(f32_of(attr(&a, "mass")));
                        }
                        if !pname.is_empty() {
                            cur_part_name = Some(pname);
                        }
                    }
                    "massbox" => {
                        setup.mass_box = Some(MassBox {
                            mass: cur_mass_box_mass.unwrap_or(0.0),
                            size: parse_vec3(attr(&a, "size")),
                            driving_offset: f32_of(attr(&a, "drivingoffset")),
                        });
                    }
                    "subpartwheel" => {
                        let mut w = Wheel::new(
                            cur_part_name.clone().unwrap_or_else(|| "wheel".to_string()),
                        );
                        w.axle = u8_of(attr(&a, "axle"));
                        w.driving = bool_of(attr(&a, "driving"));
                        w.can_brake = bool_of(attr(&a, "canbrake"));
                        w.density = f32_of(attr(&a, "density"));
                        w.damping = f32_of(attr(&a, "damping"));
                        w.len_max = f32_of(attr(&a, "lenmax"));
                        w.susp_length = f32_of(attr(&a, "susplength"));
                        w.stiffness = f32_of(attr(&a, "stiffness"));
                        w.min_friction = f32_of(attr(&a, "minfriction"));
                        w.max_friction = f32_of(attr(&a, "maxfriction"));
                        w.slip_friction_mod = f32_of(attr(&a, "slipfrictionmod"));
                        w.rim_radius = f32_of(attr(&a, "rimradius"));
                        w.torque_scale = f32_of(attr(&a, "torquescale"));
                        if wheel_master {
                            setup.wheel_master = Some(w);
                        } else {
                            setup.wheels.push(w);
                        }
                    }
                    "wheelmaster" => wheel_master = true,
                    "helper" => {
                        if in_top_helpers {
                            let hname = attr(&a, "name").unwrap_or("").to_string();
                            if !hname.is_empty() {
                                setup.helpers.insert(hname, parse_vec3(attr(&a, "position")));
                            }
                        }
                    }
                    "helpers" => in_top_helpers = !stack_contains(&stack, "seat"),
                    "seats" => in_top_seats = !stack_contains(&stack, "actions"),
                    "seat" => {
                        if in_top_seats {
                            setup.seat_count = setup.seat_count.saturating_add(1);
                        }
                    }
                    "arcadewheeled" => apply_arcade(&a, &mut setup.movement),
                    "power" => apply_power(&a, &mut setup.movement.power),
                    "wheelspin" => apply_wheelspin(&a, &mut setup.movement.wheel_spin),
                    "handbrake" => apply_handbrake(&a, &mut setup.movement.hand_brake),
                    "brakes" => apply_brakes(&a, &mut setup.movement.brakes),
                    "speedreduction" => {
                        setup.movement.speed_reduction = SpeedReduction {
                            reduction_amount: f32_of(attr(&a, "reductionamount")),
                            reduction_rate: f32_of(attr(&a, "reductionrate")),
                        };
                    }
                    "friction" => {
                        setup.movement.friction = Friction {
                            back: f32_of(attr(&a, "back")),
                            front: f32_of(attr(&a, "front")),
                            offset: f32_of(attr(&a, "offset")),
                        };
                    }
                    "correction" => {
                        setup.movement.correction = Correction {
                            lateral_spring: f32_of(attr(&a, "lateralspring")),
                            ang_spring: f32_of(attr(&a, "angspring")),
                        };
                    }
                    "compression" => {
                        setup.movement.compression = Compression {
                            friction_boost: f32_of(attr(&a, "frictionboost")),
                            friction_boost_hand_brake: f32_of(attr(&a, "frictionboosthandbrake")),
                        };
                    }
                    "inertia" => {
                        setup.movement.inertia = Inertia {
                            radius: f32_of(attr(&a, "radius")),
                        };
                    }
                    "fakegears" => {
                        setup.movement.fake_gears.min_change_up_time =
                            f32_of(attr(&a, "minchangeuptime"));
                        setup.movement.fake_gears.min_change_down_time =
                            f32_of(attr(&a, "minchangedowntime"));
                    }
                    "ratio" => {
                        let v = f32_of(attr(&a, "value"));
                        setup.movement.fake_gears.ratios.push(v);
                    }
                    "wheeledlegacy" => {
                        let l = &mut setup.movement.wheeled_legacy;
                        l.damping = f32_of(attr(&a, "damping"));
                        l.engine_idle_rpm = f32_of(attr(&a, "engineidlerpm"));
                        l.engine_max_rpm = f32_of(attr(&a, "enginemaxrpm"));
                        l.engine_min_rpm = f32_of(attr(&a, "engineminrpm"));
                        l.stabilizer = f32_of(attr(&a, "stabilizer"));
                        l.max_time_step = f32_of(attr(&a, "maxtimestep"));
                        l.min_energy = f32_of(attr(&a, "minenergy"));
                        l.susp_damping_min = f32_of(attr(&a, "suspdampingmin"));
                        l.susp_damping_max = f32_of(attr(&a, "suspdampingmax"));
                        l.susp_damping_max_speed = f32_of(attr(&a, "suspdampingmaxspeed"));
                    }
                    "airdamp" => {
                        setup.movement.air_damp = AirDamp {
                            damp_angle: parse_vec3(attr(&a, "dampangle")),
                            damp_ang_vel: parse_vec3(attr(&a, "dampangvel")),
                        };
                    }
                    _ => {}
                }
                if !is_empty {
                    stack.push(name);
                }
            }
            Event::End(e) => {
                let name = local_name_raw(e.name().as_ref());
                stack.pop();
                match name.as_str() {
                    "component" => cur_component = None,
                    "part" => {
                        cur_part_name = None;
                        cur_mass_box_mass = None;
                    }
                    "wheelmaster" => wheel_master = false,
                    "helpers" => in_top_helpers = false,
                    "seats" => in_top_seats = false,
                    _ => {}
                }
            }
            _ => {}
        }
    }
    setup
}

fn stack_contains(stack: &[String], needle: &str) -> bool {
    stack.iter().any(|s| s == needle)
}

fn apply_arcade(a: &[(String, String)], m: &mut ArcadeWheeled) {
    m.gravity = f32_of(attr(a, "gravity"));
    m.gravity_in_air = f32_of(attr(a, "gravityinair"));
    m.v_max_steer_max = f32_of(attr(a, "vmaxsteermax"));
    m.steer_speed_min = f32_of(attr(a, "steerspeedmin"));
    m.steer_speed = f32_of(attr(a, "steerspeed"));
    m.steer_speed_scale_min = f32_of(attr(a, "steerspeedscalemin"));
    m.steer_speed_scale = f32_of(attr(a, "steerspeedscale"));
    m.v0_steer_max = f32_of(attr(a, "v0steermax"));
    m.kv_steer_max = f32_of(attr(a, "kvsteermax"));
    m.rv_max_steer_max = f32_of(attr(a, "rvmaxsteermax"));
    m.reverse_steer_speed_min = f32_of(attr(a, "reversesteerspeedmin"));
    m.reverse_steer_speed = f32_of(attr(a, "reversesteerspeed"));
    m.reverse_steer_speed_scale_min = f32_of(attr(a, "reversesteerspeedscalemin"));
    m.reverse_steer_speed_scale = f32_of(attr(a, "reversesteerspeedscale"));
    m.rv0_steer_max = f32_of(attr(a, "rv0steermax"));
    m.rkv_steer_max = f32_of(attr(a, "rkvsteermax"));
    m.steer_relaxation = f32_of(attr(a, "steerrelaxation"));
    m.pedal_limit_max = f32_of(attr(a, "pedallimitmax"));
    m.rpm_relax_speed = f32_of(attr(a, "rpmrelaxspeed"));
    m.rpm_interp_speed = f32_of(attr(a, "rpminterpspeed"));
    m.rpm_gear_shift_speed = f32_of(attr(a, "rpmgearshiftspeed"));
}

fn apply_power(a: &[(String, String)], p: &mut Power) {
    p.acceleration = f32_of(attr(a, "acceleration"));
    p.decceleration = f32_of(attr(a, "decceleration"));
    p.top_speed = f32_of(attr(a, "topspeed"));
    p.reverse_speed = f32_of(attr(a, "reversespeed"));
}

fn apply_wheelspin(a: &[(String, String)], w: &mut WheelSpin) {
    w.grip1 = f32_of(attr(a, "grip1"));
    w.grip2 = f32_of(attr(a, "grip2"));
    w.grip_recover_speed = f32_of(attr(a, "griprecoverspeed"));
    w.accel_multiplier1 = f32_of(attr(a, "accelmultiplier1"));
    w.accel_multiplier2 = f32_of(attr(a, "accelmultiplier2"));
}

fn apply_handbrake(a: &[(String, String)], h: &mut HandBrake) {
    h.decceleration = f32_of(attr(a, "decceleration"));
    h.decceleration_power_lock = f32_of(attr(a, "deccelerationpowerlock"));
    h.lock_back = bool_of(attr(a, "lockback"));
    h.lock_front = bool_of(attr(a, "lockfront"));
    h.front_friction_scale = f32_of(attr(a, "frontfrictionscale"));
    h.back_friction_scale = f32_of(attr(a, "backfrictionscale"));
    h.ang_correction_scale = f32_of(attr(a, "angcorrectionscale"));
}

fn apply_brakes(a: &[(String, String)], b: &mut Brakes) {
    b.decceleration = f32_of(attr(a, "decceleration"));
    b.lock_decceleration = f32_of(attr(a, "lockdecceleration"));
    b.lock_min_speed = f32_of(attr(a, "lockminspeed"));
    b.lock_min_pressure = f32_of(attr(a, "lockminpressure"));
    b.front_friction_scale = f32_of(attr(a, "frontfrictionscale"));
    b.back_friction_scale = f32_of(attr(a, "backfrictionscale"));
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn attr_map(e: &BytesStart) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for a in e.attributes().flatten() {
        let k = String::from_utf8_lossy(a.key.as_ref()).to_ascii_lowercase();
        let v = a.unescape_value().map(|c| c.into_owned()).unwrap_or_default();
        out.push((k, v));
    }
    out
}

fn attr<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

fn local_name(e: &BytesStart) -> String {
    local_name_raw(e.name().as_ref())
}

fn local_name_raw(raw: &[u8]) -> String {
    let s = String::from_utf8_lossy(raw);
    let local = s.rsplit(':').next().unwrap_or(&s);
    local.to_ascii_lowercase()
}

fn f32_of(s: Option<&str>) -> f32 {
    s.and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(0.0)
}

fn u8_of(s: Option<&str>) -> u8 {
    s.and_then(|s| s.trim().parse::<u8>().ok()).unwrap_or(0)
}

fn bool_of(s: Option<&str>) -> bool {
    match s.map(|s| s.trim()) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("true"),
        None => false,
    }
}

fn parse_vec3(s: Option<&str>) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    if let Some(s) = s {
        for (i, p) in s.split(',').take(3).enumerate() {
            out[i] = p.trim().parse::<f32>().unwrap_or(0.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const HATCHBACK: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<Vehicle name="Hatchback_modern" actionMap="landvehicle" type="1">
 <Physics pushable="0" retainGravity="1">
  <Simulation maxTimeStep="0.02" minEnergy="0.0020000001"/>
 </Physics>
 <Damages collDamageThreshold="10" groundCollisionMinMult="1.3" groundCollisionMaxMult="1.4"
     groundCollisionMinSpeed="8" groundCollisionMaxSpeed="22">
  <DamageMultipliers>
   <DamageMultiplier damageType="bullet" multiplier="0.125"/>
   <DamageMultiplier damageType="collision" multiplier="1"/>
  </DamageMultipliers>
 </Damages>
 <Components>
  <Component name="Hull" damageMax="20000" useBoundsFromParts="1" useDamageLevels="0">
   <DamageMultipliers>
    <DamageMultiplier damageType="bullet" multiplier="10"/>
   </DamageMultipliers>
  </Component>
  <Component name="fuel_tank" damageMax="250" useBoundsFromParts="1" hullAffection="1"/>
 </Components>
 <Parts>
  <Part name="body" class="Animated" mass="5000">
   <Parts>
    <Part name="wheel1" class="SubPartWheel" component="wheel_1" mass="250">
     <SubPartWheel axle="0" density="80" damping="-0.8" driving="1" lenMax="0.15" maxFriction="1"
        minFriction="1" slipFrictionMod="0.3" stiffness="0" suspLength="0.025" rimRadius="0.1" torqueScale="1.0"/>
    </Part>
    <Part name="wheel2" class="SubPartWheel" component="wheel_2" mass="250">
     <SubPartWheel axle="1" density="80" damping="-0.8" driving="0" lenMax="0.15" maxFriction="1"
        minFriction="1" slipFrictionMod="0.3" stiffness="0" suspLength="0.025" rimRadius="0.1" torqueScale="1.0"/>
    </Part>
   </Parts>
  </Part>
  <Part name="massBox" class="MassBox" mass="1000" position="0,-0.15,0.65">
   <MassBox size="1.25,2.85,0.85" drivingOffset="-2.5"/>
  </Part>
 </Parts>
 <Helpers>
  <Helper name="wheelFrontLeft" position="-0.9,1.94513,0.0" direction="0,0,0" part="body"/>
  <Helper name="wheelFrontRight" position="0.9,1.94513,0.0" direction="0,0,0" part="body"/>
  <Helper name="wheelRearLeft" position="-0.9,-1.45805,0.0" direction="0,0,0" part="body"/>
  <Helper name="wheelRearRight" position="0.9,-1.45805,0.0" direction="0,0,0" part="body"/>
 </Helpers>
 <Seats>
  <Seat name="driver" isDriver="1"/>
  <Seat name="passenger"/>
 </Seats>
 <MovementParams>
  <ArcadeWheeled gravity="-9.8" gravityInAir="-9.0" vMaxSteerMax="18" steerSpeedMin="100" steerSpeed="80"
     v0SteerMax="60" kvSteerMax="50" rvMaxSteerMax="16" rv0SteerMax="30" rkvSteerMax="-60"
     steerRelaxation="200" rpmRelaxSpeed="1" rpmInterpSpeed="2.5" rpmGearShiftSpeed="2">
   <Handling>
    <Power acceleration="14" decceleration="1" topSpeed="24.59" reverseSpeed="16"/>
    <WheelSpin grip1="0.85" grip2="1.6" gripRecoverSpeed="2.0" accelMultiplier1="1.4" accelMultiplier2="0.1"/>
    <HandBrake decceleration="14" deccelerationPowerLock="10" lockBack="1" lockFront="1" frontFrictionScale="1.0" backFrictionScale="0.001" angCorrectionScale="1"/>
    <Brakes decceleration="16" lockDecceleration="26" lockMinSpeed="0.0" lockMinPressure="0.9" frontFrictionScale="1.0" backFrictionScale="0.1"/>
    <SpeedReduction reductionAmount="0.1" reductionRate="0.1"/>
    <Friction back="3.0" front="3.0" offset="-1.3"/>
    <Correction lateralSpring="4" angSpring="2"/>
    <Compression frictionBoost="1.5" frictionBoostHandBrake="1.5"/>
    <Inertia radius="4"/>
   </Handling>
   <FakeGears minChangeUpTime="0.69999999" minChangeDownTime="0.5"/>
   <WheeledLegacy damping="0.11" engineIdleRPM="1" engineMaxRPM="5000" engineMinRPM="1" stabilizer="0.1" maxTimeStep="0.05" minEnergy="0.09" suspDampingMin="-0.3" suspDampingMax="-1.0" suspDampingMaxSpeed="10"/>
  </ArcadeWheeled>
 </MovementParams>
 <WheelMaster>
  <SubPartWheel axle="1" density="100" damping="-0.7" driving="1" lenMax="0.3" maxFriction="1" minFriction="1" stiffness="0" suspLength="0.25"/>
 </WheelMaster>
</Vehicle>"#;

    #[test]
    fn parses_hatchback() {
        let s = parse_vehicle_xml(HATCHBACK.as_bytes());
        assert_eq!(s.name, "Hatchback_modern");
        assert_eq!(s.kind, 1);
        assert_eq!(s.mass(), 6000.0);
        assert_eq!(s.seat_count, 2);
        assert_eq!(s.damage.hull_damage_max(), Some(20000.0));
        assert!((s.damage.multiplier("bullet") - 0.125).abs() < 1e-6);
        assert!((s.damage.multiplier("melee") - 1.0).abs() < 1e-6);
        assert_eq!(s.damage.components.len(), 2);
        assert!((s.damage.coll_damage_threshold - 10.0).abs() < 1e-6);

        let m = &s.movement;
        assert!((m.power.acceleration - 14.0).abs() < 1e-6);
        assert!((m.power.top_speed - 24.59).abs() < 1e-6);
        assert!((m.wheel_spin.accel_multiplier1 - 1.4).abs() < 1e-6);
        assert!((m.correction.lateral_spring - 4.0).abs() < 1e-6);
        assert!((m.wheeled_legacy.engine_max_rpm - 5000.0).abs() < 1e-6);
        // At rest the full 60 deg is available; at vMaxSteerMax it drops by 50.
        let a0 = m.max_steer(0.0).to_degrees();
        let a18 = m.max_steer(18.0).to_degrees();
        assert!((a0 - 60.0).abs() < 1e-3);
        assert!((a18 - 10.0).abs() < 1e-3);

        assert_eq!(s.wheels.len(), 2);
        assert!(s.wheels[0].driving);
        assert!(!s.wheels[1].driving);
        assert_eq!(s.driven_wheel_count(), 1);
        assert!(s.wheel_master.is_some());

        let wb = s.wheelbase().expect("wheelbase");
        assert!((wb - (1.94513 + 1.45805)).abs() < 1e-3);
        assert!((s.track_width().unwrap() - 1.8).abs() < 1e-3);

        assert!((s.helpers["wheelFrontLeft"][1] - 1.94513).abs() < 1e-4);
    }

    #[test]
    fn garbage_yields_default() {
        let s = parse_vehicle_xml(b"not xml at all");
        assert!(s.name.is_empty());
        assert_eq!(s.wheels.len(), 0);
        // The fallback still drives.
        assert!(VehicleSetup::fallback().movement.power.top_speed > 0.0);
    }
}
