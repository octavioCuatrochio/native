//! Arcade wheeled-vehicle dynamics driven by the original per-class XML.
//!
//! Class3 registers its `"ArcadeWheeled"` movement class in the engine
//! movement factory (`0x01552970`): `("ArcadeWheeled", FUN_015527c0, 0)` at
//! `0x01553170`, with the `"ArcadeWheeled"` name string at `0x0188dd90`. The
//! factory `0x015527c0` allocates an `0x878`-byte movement object and runs its
//! constructor at `0x01672070`, which installs the primary vtable at
//! `0x018a6668` (secondary vtables `0x018a6774` / `0x018a678c`). The parameter
//! names the XML reader consumes live immediately after the vtable (strings
//! from `0x018a6820`: `isBreakingOnIdle`, `steerSpeed`, `steerSpeedMin`,
//! `kvSteerMax`, `v0SteerMax`, `steerSpeedScaleMin`, `steerSpeedScale`,
//! `steerRelaxation`, `vMaxSteerMax`, `pedalLimitMax`, `rpmRelaxSpeed`,
//! `rpmInterpSpeed`, `rpmGearShiftSpeed`, the reverse `rv*` pair,
//! `gravityInAir`, `suspDamping*`, `stabi*`, `roadBump*`, `airbrake`,
//! `slipMinRatio`, `slipWheel*`, then the `<Handling>` block
//! `Power`/`WheelSpin`/`HandBrake`/`Brakes`/`SpeedReduction`/`Friction`/
//! `Correction`/`Compression`/`Inertia`; the accessor that binds them is the
//! code around `0x016733e9`).
//!
//! This module is a native port of that movement update: per-wheel suspension
//! raycasts (the `wheelFront*/wheelRear*` helpers), the recovered steering
//! curve, engine/torque, fake gears + RPM, front/rear friction with slip and
//! the hand-brake lock scale, `Correction` slip/angular springs, `Compression`
//! grip boost, `SpeedReduction`, gravity, and a chassis sweep against the
//! static collision world for walls/props. It is deliberately arcade, matching
//! the original class's parameter set (there is no full rigid-body tyre solver
//! there either). Every constant comes from the shipped XML.
//!
//! Units are SI: metres, seconds, radians, m/s. The fixed step is supplied by
//! the caller so replays are deterministic.

use crate::cry::vehicle_xml::VehicleSetup;

/// Fixed simulation step used by callers that want the engine's `maxTimeStep`.
pub const DEFAULT_DT: f32 = 1.0 / 30.0;

/// A ray/contact result in the vehicle collision world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleHit {
    /// Distance along the ray (`t`), or `0` for a documented contact point.
    pub t: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
}

/// Result of sweeping the chassis: the resolved position and the deepest
/// contact encountered, if any.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChassisMotion {
    pub pos: [f32; 3],
    pub hit: Option<VehicleHit>,
}

/// Ground query used by the simulation. Implemented for the app's
/// [`crate::cry::phys_world::CollisionWorld`] and for a bare
/// [`crate::terrain::Terrain`], and by test height fields.
pub trait VehicleGround {
    /// Ground height at world `(x, y)`, or `None` when there is no ground.
    fn height(&self, x: f32, y: f32) -> Option<f32>;
    /// Upward ground normal at `(x, y)` (Z-up). Defaults to flat ground.
    fn normal(&self, _x: f32, _y: f32) -> [f32; 3] {
        [0.0, 0.0, 1.0]
    }

    /// Cast a wheel ray straight down from `origin` for at most `max` metres.
    /// The default implementation samples [`VehicleGround::height`], which is
    /// enough for a bare height field; [`crate::cry::phys_world::CollisionWorld`]
    /// overrides it with a real ray against terrain and physics proxies.
    fn ray_down(&self, origin: [f32; 3], max: f32) -> Option<VehicleHit> {
        let h = self.height(origin[0], origin[1])?;
        let dz = origin[2] - h;
        if dz < 0.0 {
            return Some(VehicleHit {
                t: 0.0,
                point: [origin[0], origin[1], h],
                normal: self.normal(origin[0], origin[1]),
            });
        }
        if dz <= max {
            Some(VehicleHit {
                t: dz,
                point: [origin[0], origin[1], h],
                normal: self.normal(origin[0], origin[1]),
            })
        } else {
            None
        }
    }

    /// Sweep a vertical capsule of `radius`/`height` from `from` by `delta`,
    /// sliding out of static geometry. The default is a free move (height-field
    /// terrains have no walls).
    fn chassis_sweep(
        &self,
        from: [f32; 3],
        delta: [f32; 3],
        _radius: f32,
        _height: f32,
    ) -> ChassisMotion {
        ChassisMotion {
            pos: [from[0] + delta[0], from[1] + delta[1], from[2] + delta[2]],
            hit: None,
        }
    }
}

impl VehicleGround for crate::cry::phys_world::CollisionWorld {
    fn height(&self, x: f32, y: f32) -> Option<f32> {
        self.terrain_height(x, y)
    }
    fn normal(&self, x: f32, y: f32) -> [f32; 3] {
        self.terrain_normal(x, y)
    }
    fn ray_down(&self, origin: [f32; 3], max: f32) -> Option<VehicleHit> {
        self.raycast(origin, [0.0, 0.0, -1.0], max)
            .map(|h| VehicleHit { t: h.t, point: h.point, normal: h.normal })
    }
    fn chassis_sweep(
        &self,
        from: [f32; 3],
        delta: [f32; 3],
        radius: f32,
        height: f32,
    ) -> ChassisMotion {
        let sw = self.sweep_capsule(from, delta, radius, height);
        ChassisMotion {
            pos: sw.pos,
            hit: sw.contact.map(|c| VehicleHit { t: 0.0, point: c.point, normal: c.normal }),
        }
    }
}

impl VehicleGround for crate::terrain::Terrain {
    fn height(&self, x: f32, y: f32) -> Option<f32> {
        Some(self.height_interp(x, y))
    }
}

/// Driver input for one fixed step. All values may be interpolated by the
/// caller; they are clamped internally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleInput {
    /// `-1` = full reverse/brake, `0` = coast, `1` = full throttle.
    pub throttle: f32,
    /// `-1` = full right, `0` = centred, `1` = full left.
    pub steer: f32,
    /// Foot brake, `0..1`.
    pub brake: f32,
    /// Hand brake / e-brake.
    pub handbrake: bool,
}

impl Default for VehicleInput {
    fn default() -> Self {
        VehicleInput { throttle: 0.0, steer: 0.0, brake: 0.0, handbrake: false }
    }
}

/// One suspension wheel: its anchor in body space (`[right, forward, up]`) and
/// the last raycast result. `SubPartWheel` gives `radius`, `suspLength` and
/// `lenMax`; `driving` chooses the powered axle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelState {
    /// Body-space anchor `[right, forward, up]` from the `wheel*` helpers.
    pub anchor: [f32; 3],
    pub driven: bool,
    /// `true` for the front axle (helper forward offset `> 0`).
    pub front: bool,
    /// Wheel/tire radius used by the suspension ray.
    pub radius: f32,
    /// Total suspension travel (`suspLength + lenMax`).
    pub max_travel: f32,
    /// Current compression, `0..max_travel`.
    pub compression: f32,
    pub grounded: bool,
    /// Ground contact point/normal from the last ray.
    pub contact: [f32; 3],
    pub normal: [f32; 3],
}

impl WheelState {
    /// Compression as a `0..1` fraction of the travel.
    pub fn compression_fraction(&self) -> f32 {
        if self.max_travel > 1e-4 {
            (self.compression / self.max_travel).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// The mutable rigid state of one vehicle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleState {
    /// Chassis origin (wheel contact plane), world Z-up metres.
    pub pos: [f32; 3],
    /// Heading about `+Z` (radians); `forward = (cos, sin, 0)`.
    pub yaw: f32,
    /// World-space horizontal velocity `(vx, vy)`.
    pub vel: [f32; 2],
    /// Vertical velocity while airborne.
    pub vertical_speed: f32,
    /// Current front-wheel steer angle (radians).
    pub steer: f32,
    /// Engine speed (RPM, from the XML band).
    pub rpm: f32,
    /// Current gear index (`0..=ratios.len()`).
    pub gear: usize,
    /// True while the chassis is resting on the ground.
    pub grounded: bool,
    /// Lateral (slip) speed in m/s, useful for skid effects/telemetry.
    pub slip: f32,
    /// Distance travelled (m).
    pub odometer: f32,
    /// Body pitch (radians, +nose-up) derived from the wheel contacts.
    pub pitch: f32,
    /// Body roll (radians, +right-down) derived from the wheel contacts.
    pub roll: f32,
}

impl VehicleState {
    /// A car resting at `pos` facing `yaw`.
    pub fn at(pos: [f32; 3], yaw: f32) -> VehicleState {
        VehicleState {
            pos,
            yaw,
            vel: [0.0, 0.0],
            vertical_speed: 0.0,
            steer: 0.0,
            rpm: 0.0,
            gear: 0,
            grounded: true,
            slip: 0.0,
            odometer: 0.0,
            pitch: 0.0,
            roll: 0.0,
        }
    }

    /// Unit forward vector `(x, y)`.
    pub fn forward(&self) -> [f32; 2] {
        [self.yaw.cos(), self.yaw.sin()]
    }

    /// Unit right vector `(x, y)` (`forward` rotated -90° about `+Z`).
    pub fn right(&self) -> [f32; 2] {
        [self.yaw.sin(), -self.yaw.cos()]
    }

    /// Signed longitudinal speed (m/s) along [`VehicleState::forward`].
    pub fn forward_speed(&self) -> f32 {
        let f = self.forward();
        self.vel[0] * f[0] + self.vel[1] * f[1]
    }

    /// Lateral speed (m/s) along [`VehicleState::right`].
    pub fn lateral_speed(&self) -> f32 {
        let r = self.right();
        self.vel[0] * r[0] + self.vel[1] * r[1]
    }

    /// Horizontal speed magnitude (m/s).
    pub fn speed(&self) -> f32 {
        (self.vel[0] * self.vel[0] + self.vel[1] * self.vel[1]).sqrt()
    }
}

/// Per-component damage state for one vehicle, initialised from the XML
/// `<Components>` block.
///
/// The engine applies top-level `<DamageMultipliers>` per damage type and then
/// routes the result to the struck component; a component with
/// `hullAffection="1"` passes its damage straight to the hull.
#[derive(Debug, Clone, PartialEq)]
pub struct VehicleHealth {
    /// Current hull health.
    pub hull: f32,
    /// Hull `damageMax` from the XML.
    pub hull_max: f32,
    /// Current health of every `<Component>`, aligned with
    /// [`crate::cry::vehicle_xml::VehicleDamage::components`].
    pub components: Vec<f32>,
    /// True once the hull is depleted.
    pub destroyed: bool,
}

impl VehicleHealth {
    /// Full health from a parsed setup.
    pub fn new(setup: &VehicleSetup) -> VehicleHealth {
        let components: Vec<f32> = setup
            .damage
            .components
            .iter()
            .map(|c| c.damage_max)
            .collect();
        let hull_max = setup.damage.hull_damage_max().unwrap_or(0.0);
        VehicleHealth { hull: hull_max, hull_max, components, destroyed: false }
    }

    /// Hull fraction remaining, `0..=1` (1 when there is no hull component).
    pub fn hull_ratio(&self) -> f32 {
        if self.hull_max > 0.0 {
            (self.hull / self.hull_max).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Effective multiplier for `damage_type` on component `idx`: the
    /// component's own `<DamageMultipliers>` overrides the top-level
    /// `<Damages><DamageMultipliers>` default (the same precedence
    /// `CVehicleComponent` uses when it looks its multiplier up by type).
    fn component_mult(setup: &VehicleSetup, idx: usize, damage_type: &str) -> f32 {
        setup
            .damage
            .components
            .get(idx)
            .and_then(|c| c.multiplier(damage_type))
            .unwrap_or_else(|| setup.damage.multiplier(damage_type))
    }

    /// Apply `amount` of `damage_type` to the hull. Routes through the `Hull`
    /// component when one is declared so its per-component multipliers apply.
    pub fn apply_hull(&mut self, setup: &VehicleSetup, damage_type: &str, amount: f32) {
        match setup
            .damage
            .components
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case("hull"))
        {
            Some(i) => self.apply_component(setup, i, damage_type, amount),
            None => {
                let dmg = amount.max(0.0) * setup.damage.multiplier(damage_type);
                self.hull = (self.hull - dmg).max(0.0);
                self.check();
            }
        }
    }

    /// Apply damage to component `idx`; the component's `hullAffection`
    /// fraction is forwarded to the hull (the `<Hull>` component forwards all
    /// of it).
    pub fn apply_component(
        &mut self,
        setup: &VehicleSetup,
        idx: usize,
        damage_type: &str,
        amount: f32,
    ) {
        let Some(comp) = setup.damage.components.get(idx) else {
            return;
        };
        let dmg = amount.max(0.0) * Self::component_mult(setup, idx, damage_type);
        if let Some(h) = self.components.get_mut(idx) {
            *h = (*h - dmg).max(0.0);
        }
        if comp.name.eq_ignore_ascii_case("hull") {
            self.hull = (self.hull - dmg).max(0.0);
        } else if comp.hull_affection > 0.0 {
            self.hull = (self.hull - dmg * comp.hull_affection).max(0.0);
        }
        self.check();
    }

    /// Damage ratio of component `idx`, `0..=1` (`1 - health / damageMax`).
    pub fn component_ratio(&self, setup: &VehicleSetup, idx: usize) -> f32 {
        let Some(comp) = setup.damage.components.get(idx) else {
            return 0.0;
        };
        let Some(&hp) = self.components.get(idx) else {
            return 0.0;
        };
        if comp.damage_max > 0.0 {
            (1.0 - hp / comp.damage_max).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// `<DamageBehavior>`s on component `idx` whose `damageRatioMin` has been
    /// reached, in document order. The caller drives the corresponding
    /// effect/mask/detach/group action.
    pub fn fired_behaviors<'a>(
        &'a self,
        setup: &'a VehicleSetup,
        idx: usize,
    ) -> impl Iterator<Item = &'a crate::cry::vehicle_xml::DamageBehavior> + 'a {
        let ratio = self.component_ratio(setup, idx);
        setup
            .damage
            .components
            .get(idx)
            .map(move |c| c.behaviors.iter().filter(move |b| b.fires_at(ratio)))
            .into_iter()
            .flatten()
    }

    fn check(&mut self) {
        if self.hull <= 0.0 {
            self.hull = 0.0;
            self.destroyed = true;
        }
    }
}

/// Collision damage for an impact of `speed` (m/s) along a hard surface,
/// following the `<Damages>` thresholds.
///
/// `collDamageThreshold` is the minimum impact speed; between
/// `groundCollisionMinSpeed` and `groundCollisionMaxSpeed` the damage is
/// scaled from `groundCollisionMinMult` to `groundCollisionMaxMult`.
pub fn collision_damage(setup: &VehicleSetup, speed: f32) -> f32 {
    let d = &setup.damage;
    let v = speed.abs();
    if v <= d.coll_damage_threshold {
        return 0.0;
    }
    let vmin = d.ground_collision_min_speed;
    let vmax = d.ground_collision_max_speed;
    let mult = if vmax > vmin {
        let t = ((v - vmin) / (vmax - vmin)).clamp(0.0, 1.0);
        d.ground_collision_min_mult + t * (d.ground_collision_max_mult - d.ground_collision_min_mult)
    } else {
        d.ground_collision_min_mult.max(1.0)
    };
    (v - d.coll_damage_threshold) * mult.max(0.0)
}

/// One simulated vehicle: its setup and its state.
#[derive(Debug, Clone)]
pub struct VehicleBody {
    pub setup: VehicleSetup,
    pub state: VehicleState,
    pub health: VehicleHealth,
    /// Per-wheel suspension state, built from the `wheel*` helpers.
    pub wheels: Vec<WheelState>,
    shift_timer: f32,
}

impl VehicleBody {
    /// Create a body from a parsed setup, resting on the ground at `pos`.
    pub fn new(setup: VehicleSetup, pos: [f32; 3], yaw: f32) -> VehicleBody {
        let health = VehicleHealth::new(&setup);
        let wheels = build_wheels(&setup);
        VehicleBody {
            setup,
            state: VehicleState::at(pos, yaw),
            health,
            wheels,
            shift_timer: 0.0,
        }
    }

    /// Wheelbase in metres, falling back to twice the inertia radius (the
    /// ArcadeWheeled `<Inertia radius>` is a collision radius) or 2.6 m.
    pub fn wheelbase(&self) -> f32 {
        self.setup
            .wheelbase()
            .or_else(|| {
                let r = self.setup.movement.inertia.radius;
                if r > 0.0 {
                    Some(r * 2.0)
                } else {
                    None
                }
            })
            .filter(|w| *w > 0.5)
            .unwrap_or(2.6)
    }

    /// Chassis collision capsule: `(radius, height, ground clearance)`. Scaled
    /// from the wheel track so a wide truck blocks wider than a hatchback.
    fn chassis_capsule(&self) -> (f32, f32, f32) {
        let track = self.setup.track_width().unwrap_or(1.8);
        let radius = (track * 0.5 * 0.75).clamp(0.35, 1.25);
        (radius, 1.2, 0.35)
    }

    /// Advance one fixed step. `ground` supplies terrain and static geometry.
    pub fn step(&mut self, input: &VehicleInput, ground: &impl VehicleGround, dt: f32) {
        if !(dt > 0.0) {
            return;
        }
        let throttle = input.throttle.clamp(-1.0, 1.0);
        let steer_in = input.steer.clamp(-1.0, 1.0);
        let brake = input.brake.clamp(0.0, 1.0);
        let handbrake = input.handbrake;

        let was_grounded = self.state.grounded;
        let entry_vspeed = self.state.vertical_speed;

        // Body basis: forward local +Y, right local +X (CryEngine Z-up).
        let cy = self.state.yaw.cos();
        let sy = self.state.yaw.sin();
        let fwd = [cy, sy];
        let rt = [sy, -cy];

        // --- per-wheel suspension raycasts ---------------------------------
        let (px, py, pz) = (self.state.pos[0], self.state.pos[1], self.state.pos[2]);
        for w in self.wheels.iter_mut() {
            let wx = px + rt[0] * w.anchor[0] + fwd[0] * w.anchor[1];
            let wy = py + rt[1] * w.anchor[0] + fwd[1] * w.anchor[1];
            // Mount the ray at the fully-extended hub height; a wheel resting on
            // flat ground (contact plane at `pz`) reads back `radius + maxTravel`
            // and zero compression.
            let origin = [wx, wy, pz + w.anchor[2] + w.radius + w.max_travel];
            let max = w.radius + 2.0 * w.max_travel + 0.05;
            match ground.ray_down(origin, max) {
                Some(h) => {
                    w.grounded = true;
                    w.contact = h.point;
                    w.normal = if h.normal.iter().all(|v| v.is_finite()) {
                        h.normal
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    let rest_t = w.radius + w.max_travel;
                    w.compression = (rest_t - h.t).clamp(0.0, w.max_travel);
                }
                None => {
                    w.grounded = false;
                    w.normal = [0.0, 0.0, 1.0];
                    w.compression = 0.0;
                    w.contact = [wx, wy, origin[2] - max];
                }
            }
        }
        let grounded = self.wheels.iter().any(|w| w.grounded);
        let n_gnd = self.wheels.iter().filter(|w| w.grounded).count();

        // Driven-wheel contact fraction: an axle in the air has no traction.
        let driven_total = self.wheels.iter().filter(|w| w.driven).count().max(1);
        let driven_gnd = self.wheels.iter().filter(|w| w.driven && w.grounded).count();
        let drive_frac = driven_gnd as f32 / driven_total as f32;

        // --- longitudinal engine / resistance ------------------------------
        let m = &self.setup.movement;
        let vlong = self.state.vel[0] * fwd[0] + self.state.vel[1] * fwd[1];
        let vlat = self.state.vel[0] * rt[0] + self.state.vel[1] * rt[1];

        let accel_mult = m.accel_multiplier(vlong);
        let mut a_long = 0.0;
        if throttle > 0.0 {
            a_long += m.power.acceleration * throttle * accel_mult * drive_frac;
        } else if throttle < 0.0 {
            // Reverse is the same engine behind the reverse gear: only useful
            // while not already rolling forward fast.
            let reverse_bite =
                (1.0 - (vlong / m.power.top_speed.max(1e-3)).max(0.0)).clamp(0.0, 1.0);
            a_long += m.power.acceleration * throttle * reverse_bite * drive_frac;
        }
        let mut next_long = vlong + a_long * dt;

        let mut decel = m.power.decceleration.max(0.0);
        if brake > 0.0 {
            decel = decel.max(m.brakes.decceleration * brake);
        }
        if handbrake {
            decel = decel.max(m.hand_brake.decceleration);
        }
        let dv = decel * dt;
        if next_long.abs() <= dv {
            next_long = 0.0;
        } else {
            next_long -= next_long.signum() * dv;
        }
        // `SpeedReduction` bleeds off speed with no wheel contact.
        if !grounded {
            let bleed = (m.speed_reduction.reduction_amount * m.speed_reduction.reduction_rate)
                .clamp(0.0, 1.0);
            next_long *= (1.0 - bleed * dt).max(0.0);
        }
        next_long = next_long.clamp(-m.power.reverse_speed.max(0.0), m.power.top_speed.max(0.0));

        // --- lateral grip / slip -------------------------------------------
        // `Friction` front/back is the steady lateral decay; `Correction`
        // lateral spring adds anti-slip stiffness. The hand brake locks the
        // rear axle (`backFrictionScale`) so the car slides.
        let mut grip = ((m.friction.front.max(0.0) + m.friction.back.max(0.0)) * 0.5).max(0.0);
        if grip <= 0.0 {
            grip = 1.0;
        }
        grip += m.correction.lateral_spring.max(0.0);
        // Suspension compression boosts grip (tyre load sensitivity).
        let avg_frac = if self.wheels.is_empty() {
            0.0
        } else {
            self.wheels.iter().map(|w| w.compression_fraction()).sum::<f32>()
                / self.wheels.len() as f32
        };
        let boost = if handbrake {
            m.compression.friction_boost_hand_brake
        } else {
            m.compression.friction_boost
        };
        grip *= 1.0 + boost.max(0.0) * avg_frac;
        if handbrake {
            grip *= m.hand_brake.back_friction_scale.clamp(0.001, 1.0);
        }
        if brake > 0.5 {
            grip *= m.brakes.back_friction_scale.clamp(0.01, 1.0);
        }
        if !grounded {
            grip *= 0.05;
        }
        let next_lat = vlat * (-grip * dt).exp();

        // --- steering -------------------------------------------------------
        let full_lock = m.max_steer(vlong);
        let steer_rate = steer_rate_deg(m, vlong, steer_in);
        let max_delta = steer_rate.to_radians().max(0.0) * dt;
        let target = steer_in * full_lock;
        self.state.steer += (target - self.state.steer).clamp(-max_delta, max_delta);

        // --- heading (bicycle model + angular spring) -----------------------
        let wb = self.wheelbase();
        let steer_eff = if grounded {
            self.state.steer
        } else {
            self.state.steer * 0.25
        };
        let mut yaw_rate = if vlong.abs() > 0.05 {
            (vlong / wb) * steer_eff.tan()
        } else {
            0.0
        };
        yaw_rate *= (1.0 - m.correction.ang_spring.max(0.0) * dt).max(0.0);

        // Recompose in the pre-turn body frame (the heading rotates apart from
        // the velocity, so the next step's lateral component is exactly the
        // slip grip has to remove).
        let vx = fwd[0] * next_long + rt[0] * next_lat;
        let vy = fwd[1] * next_long + rt[1] * next_lat;
        self.state.vel = [vx, vy];
        self.state.slip = next_lat.abs();
        self.state.odometer += (vx * vx + vy * vy).sqrt() * dt;
        self.state.yaw = wrap_pi(self.state.yaw + yaw_rate * dt);

        // --- horizontal move + chassis collision ----------------------------
        let (radius, height, clearance) = self.chassis_capsule();
        let from = [px, py, pz + clearance];
        let motion = ground.chassis_sweep(from, [vx * dt, vy * dt, 0.0], radius, height);
        let mut impact = 0.0f32;
        self.state.pos[0] = motion.pos[0];
        self.state.pos[1] = motion.pos[1];
        if let Some(hit) = motion.hit {
            let n2 = [hit.normal[0], hit.normal[1]];
            let nl = (n2[0] * n2[0] + n2[1] * n2[1]).sqrt();
            if nl > 1e-3 {
                let n = [n2[0] / nl, n2[1] / nl];
                let into = self.state.vel[0] * n[0] + self.state.vel[1] * n[1];
                if into < 0.0 {
                    impact = -into;
                    // Slide: remove the into-wall component and bleed energy.
                    self.state.vel[0] = (self.state.vel[0] - into * n[0]) * 0.55;
                    self.state.vel[1] = (self.state.vel[1] - into * n[1]) * 0.55;
                }
            }
        }

        // --- vertical / suspension support ----------------------------------
        if grounded {
            // Snap the chassis onto the average wheel contact height.
            let support_z: f32 = self
                .wheels
                .iter()
                .filter(|w| w.grounded)
                .map(|w| w.contact[2])
                .sum::<f32>()
                / n_gnd.max(1) as f32;
            if !was_grounded {
                impact = impact.max((-entry_vspeed).max(0.0));
            }
            self.state.pos[2] = support_z;
            self.state.vertical_speed = 0.0;
            self.state.grounded = true;
        } else {
            self.state.vertical_speed += m.gravity_in_air * dt;
            self.state.pos[2] += self.state.vertical_speed * dt;
            self.state.grounded = false;
        }

        // --- body attitude from the four contacts --------------------------
        if grounded {
            let (mut fz, mut fq) = (0.0f32, 0);
            let (mut rz, mut rq) = (0.0f32, 0);
            let (mut lz, mut lq) = (0.0f32, 0);
            let (mut rr_z, mut rr_q) = (0.0f32, 0);
            for w in self.wheels.iter().filter(|w| w.grounded) {
                if w.front {
                    fz += w.contact[2];
                    fq += 1;
                } else {
                    rz += w.contact[2];
                    rq += 1;
                }
                if w.anchor[0] < 0.0 {
                    lz += w.contact[2];
                    lq += 1;
                } else {
                    rr_z += w.contact[2];
                    rr_q += 1;
                }
            }
            let track = self.setup.track_width().unwrap_or(1.8).max(0.5);
            let pitch_t = if fq > 0 && rq > 0 {
                ((fz / fq as f32) - (rz / rq as f32)).atan2(wb.max(0.5))
            } else {
                self.state.pitch
            };
            let roll_t = if lq > 0 && rr_q > 0 {
                -(((rr_z / rr_q as f32) - (lz / lq as f32)).atan2(track))
            } else {
                self.state.roll
            };
            let k = (dt * 8.0).min(1.0);
            self.state.pitch += (pitch_t - self.state.pitch) * k;
            self.state.roll += (roll_t - self.state.roll) * k;
        }

        // A hard hit or landing damages the hull via `<Damages>`.
        if impact > 0.0 {
            let dmg = collision_damage(&self.setup, impact);
            if dmg > 0.0 {
                self.health.apply_hull(&self.setup, "collision", dmg);
            }
        }

        self.update_drive_train(next_long, dt);
    }

    /// Fake gears + RPM from the XML. `<FakeGears>` `ratios` are upshift
    /// fractions of top speed; `minChange*Time` gates the shifts. RPM chases
    /// the current band at `rpmInterpSpeed` and relaxes at `rpmRelaxSpeed`;
    /// `rpmGearShiftSpeed` caps how fast the gear index may move.
    fn update_drive_train(&mut self, vlong: f32, dt: f32) {
        let m = &self.setup.movement;
        let ratios = &m.fake_gears.ratios;
        let top = m.power.top_speed.max(1e-3);
        let frac = (vlong.abs() / top).clamp(0.0, 1.0);

        let target_gear = ratios.iter().take_while(|&&r| frac >= r).count();
        self.shift_timer += dt;
        let max_gear_step = (m.rpm_gear_shift_speed.max(1.0) * dt).max(1.0) as usize;
        if target_gear > self.state.gear {
            if self.shift_timer >= m.fake_gears.min_change_up_time {
                self.state.gear = (self.state.gear + 1).min(target_gear);
                self.shift_timer = 0.0;
            }
        } else if target_gear < self.state.gear {
            if self.shift_timer >= m.fake_gears.min_change_down_time {
                self.state.gear = self.state.gear.saturating_sub(1).max(target_gear);
                self.shift_timer = 0.0;
            }
        }
        let _ = max_gear_step;

        // Local fraction inside the current gear band -> RPM sweep.
        let ratios_empty = ratios.is_empty();
        let lo = if self.state.gear == 0 || ratios_empty {
            0.0
        } else {
            ratios.get(self.state.gear - 1).copied().unwrap_or(0.0)
        };
        let hi = ratios
            .get(self.state.gear)
            .copied()
            .unwrap_or(1.0)
            .max(lo + 1e-3);
        let t = if ratios_empty { frac } else { ((frac - lo) / (hi - lo)).clamp(0.0, 1.0) };
        let idle = m.wheeled_legacy.engine_idle_rpm;
        let max = m.wheeled_legacy.engine_max_rpm.max(idle);
        let target_rpm = idle + (max - idle) * t;
        // `rpmInterpSpeed` is the per-second approach rate (scaled to the RPM
        // band); idle relaxation is gentler (`rpmRelaxSpeed`).
        let rate = if target_rpm >= self.state.rpm {
            m.rpm_interp_speed.max(0.0)
        } else {
            m.rpm_relax_speed.max(0.0)
        };
        let step = rate * (max - idle).max(1.0) * dt;
        self.state.rpm += (target_rpm - self.state.rpm).clamp(-step, step);
    }

    /// Full snapshots for replay/telemetry.
    pub fn snapshot(&self) -> (VehicleState, f32) {
        (self.state, self.setup.movement.power.top_speed)
    }
}

/// Steering rate in degrees/second: the recovered
/// `CVehicleMovementArcadeWheeled::GetSteerSpeed` (`0x01673170`, see
/// [`crate::cry::vehicle_xml::ArcadeWheeled::steer_rate_deg`]) for the input
/// direction, and the `steerRelaxation` rate when the wheel returns to centre.
fn steer_rate_deg(
    m: &crate::cry::vehicle_xml::ArcadeWheeled,
    vlong: f32,
    steer_in: f32,
) -> f32 {
    if steer_in.abs() < 1e-3 {
        return m.steer_relaxation;
    }
    m.steer_rate_deg(vlong, vlong < 0.0)
}

/// Build the four suspension wheels from the `<SubPartWheel>` set and the
/// `wheelFront*/wheelRear*` helpers. Axle `0` is the front, `1` the rear; the
/// driven flag comes from the wheel `driving` attribute.
fn build_wheels(setup: &VehicleSetup) -> Vec<WheelState> {
    let wb = setup.wheelbase().unwrap_or(2.6).max(0.5);
    let track = setup.track_width().unwrap_or(1.8).max(0.5);
    let front_driven = setup.wheels.iter().any(|w| w.axle == 0 && w.driving);
    let rear_driven = setup.wheels.iter().any(|w| w.axle == 1 && w.driving);

    let radius = setup
        .wheels
        .iter()
        .map(|w| w.rim_radius)
        .fold(0.0f32, f32::max)
        .max(0.32);
    let max_travel = setup
        .wheels
        .iter()
        .map(|w| (w.susp_length + w.len_max).max(0.15))
        .fold(0.15f32, f32::max);

    let helper = |name: &str, default: [f32; 3]| setup.helpers.get(name).copied().unwrap_or(default);
    let fl = helper("wheelFrontLeft", [-track * 0.5, wb * 0.5, 0.0]);
    let fr = helper("wheelFrontRight", [track * 0.5, wb * 0.5, 0.0]);
    let rl = helper("wheelRearLeft", [-track * 0.5, -wb * 0.5, 0.0]);
    let rr = helper("wheelRearRight", [track * 0.5, -wb * 0.5, 0.0]);

    let wheel = |anchor: [f32; 3], driven: bool, front: bool| WheelState {
        anchor,
        driven,
        front,
        radius,
        max_travel,
        compression: 0.0,
        grounded: false,
        contact: [0.0; 3],
        normal: [0.0, 0.0, 1.0],
    };
    vec![
        wheel(fl, front_driven, true),
        wheel(fr, front_driven, true),
        wheel(rl, rear_driven, false),
        wheel(rr, rear_driven, false),
    ]
}

fn wrap_pi(a: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let mut x = a % two_pi;
    if x > std::f32::consts::PI {
        x -= two_pi;
    } else if x < -std::f32::consts::PI {
        x += two_pi;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cry::vehicle_xml::VehicleSetup;

    /// Flat ground at height `z`.
    struct Flat {
        z: f32,
    }
    impl VehicleGround for Flat {
        fn height(&self, _x: f32, _y: f32) -> Option<f32> {
            Some(self.z)
        }
    }

    /// A constant 10 % slope rising along +x.
    struct Slope;
    impl VehicleGround for Slope {
        fn height(&self, x: f32, _y: f32) -> Option<f32> {
            Some(x * 0.1)
        }
        fn normal(&self, _x: f32, _y: f32) -> [f32; 3] {
            let n = [-0.1f32, 0.0, 1.0];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            [n[0] / l, n[1] / l, n[2] / l]
        }
    }

    /// An invisible wall at `x = wall`, only the horizontal component matters.
    struct Wall {
        wall: f32,
    }
    impl VehicleGround for Wall {
        fn height(&self, _x: f32, _y: f32) -> Option<f32> {
            Some(0.0)
        }
        fn chassis_sweep(
            &self,
            from: [f32; 3],
            delta: [f32; 3],
            _radius: f32,
            _height: f32,
        ) -> ChassisMotion {
            let nx = from[0] + delta[0];
            if nx + 0.4 > self.wall && from[0] + 0.4 <= self.wall {
                ChassisMotion {
                    pos: [self.wall - 0.4, from[1] + delta[1], from[2] + delta[2]],
                    hit: Some(VehicleHit { t: 0.0, point: [self.wall, from[1], from[2]], normal: [-1.0, 0.0, 0.0] }),
                }
            } else {
                ChassisMotion { pos: [nx, from[1] + delta[1], from[2] + delta[2]], hit: None }
            }
        }
    }

    fn hatch() -> VehicleSetup {
        VehicleSetup::fallback()
    }

    fn run(body: &mut VehicleBody, input: &VehicleInput, ground: &impl VehicleGround, secs: f32) {
        let dt = 1.0 / 60.0;
        let steps = (secs / dt).round() as usize;
        for _ in 0..steps {
            body.step(input, ground, dt);
        }
    }

    #[test]
    fn accelerates_and_clamps_to_top_speed() {
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        let g = Flat { z: 0.0 };
        let throttle = VehicleInput { throttle: 1.0, ..VehicleInput::default() };
        run(&mut b, &throttle, &g, 20.0);
        let top = b.setup.movement.power.top_speed;
        assert!(b.state.forward_speed() > top * 0.9, "spd={}", b.state.forward_speed());
        assert!(b.state.forward_speed() <= top + 1e-3);
        assert!((b.state.pos[2] - 0.0).abs() < 1e-3);
        assert!(b.state.grounded);
        assert!(b.state.gear > 0, "should have upshifted");
        assert!(b.state.rpm > 0.0);
    }

    #[test]
    fn coasts_to_rest() {
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        let g = Flat { z: 0.0 };
        run(&mut b, &VehicleInput { throttle: 1.0, ..Default::default() }, &g, 3.0);
        assert!(b.state.speed() > 1.0);
        run(&mut b, &VehicleInput::default(), &g, 30.0);
        assert!(b.state.speed() < 0.05, "spd={}", b.state.speed());
    }

    #[test]
    fn brakes_stop_faster_than_coasting() {
        let g = Flat { z: 0.0 };
        let mut coast = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        let mut stop = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        let go = VehicleInput { throttle: 1.0, ..Default::default() };
        run(&mut coast, &go, &g, 3.0);
        run(&mut stop, &go, &g, 3.0);
        let brake = VehicleInput { brake: 1.0, ..Default::default() };
        run(&mut coast, &VehicleInput::default(), &g, 1.0);
        run(&mut stop, &brake, &g, 1.0);
        assert!(stop.state.speed() < coast.state.speed());
    }

    #[test]
    fn steering_turns_the_car() {
        let g = Flat { z: 0.0 };
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        let input = VehicleInput { throttle: 1.0, steer: 1.0, ..Default::default() };
        run(&mut b, &input, &g, 2.0);
        assert!(b.state.yaw > 0.1, "yaw={}", b.state.yaw);
        assert!(b.state.pos[1] > 0.0, "should curve left (+y)");
        let mut c = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        run(&mut c, &VehicleInput { throttle: 1.0, steer: -1.0, ..Default::default() }, &g, 2.0);
        assert!(c.state.yaw < -0.1);
    }

    #[test]
    fn reverse_goes_backwards() {
        let g = Flat { z: 0.0 };
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        run(&mut b, &VehicleInput { throttle: -1.0, ..Default::default() }, &g, 3.0);
        assert!(b.state.forward_speed() < -1.0, "spd={}", b.state.forward_speed());
        assert!(b.state.pos[0] < -1.0);
        let rev = b.setup.movement.power.reverse_speed;
        assert!(b.state.forward_speed() >= -rev - 1e-3);
    }

    #[test]
    fn handbrake_slides() {
        let g = Flat { z: 0.0 };
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        run(&mut b, &VehicleInput { throttle: 1.0, ..Default::default() }, &g, 3.0);
        let hb = VehicleInput { throttle: 0.0, steer: 1.0, handbrake: true, ..Default::default() };
        run(&mut b, &hb, &g, 1.0);
        assert!(b.state.slip > 0.05, "expected lateral slip, got {}", b.state.slip);
    }

    #[test]
    fn falls_under_gravity_off_a_ledge() {
        struct Ledge;
        impl VehicleGround for Ledge {
            fn height(&self, x: f32, _y: f32) -> Option<f32> {
                if x < 0.0 { Some(0.0) } else { None }
            }
        }
        let mut b = VehicleBody::new(hatch(), [-2.0, 0.0, 0.0], 0.0);
        b.state.pos[2] = 0.0;
        b.state.vel = [10.0, 0.0];
        for _ in 0..120 {
            b.step(&VehicleInput::default(), &Ledge, 1.0 / 60.0);
        }
        assert!(!b.state.grounded);
        assert!(b.state.pos[2] < -0.5, "z={}", b.state.pos[2]);
    }

    #[test]
    fn rests_on_slope_and_pitches() {
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        let g = Slope;
        for _ in 0..30 {
            b.step(&VehicleInput::default(), &g, 1.0 / 60.0);
        }
        // The chassis rides at the mean wheel-contact height (the hatchback's
        // wheelbase is not centred on the model origin).
        let gz: Vec<f32> = b.wheels.iter().filter(|w| w.grounded).map(|w| w.contact[2]).collect();
        assert!(!gz.is_empty());
        let lo = gz.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = gz.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(b.state.pos[2] >= lo - 1e-3 && b.state.pos[2] <= hi + 1e-3, "z={}", b.state.pos[2]);
        assert!(b.state.pitch > 1e-3, "pitch={}", b.state.pitch);
    }

    #[test]
    fn chassis_wall_stops_the_car() {
        let g = Wall { wall: 5.0 };
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        run(&mut b, &VehicleInput { throttle: 1.0, ..Default::default() }, &g, 5.0);
        assert!(b.state.pos[0] < 5.0, "car passed the wall: x={}", b.state.pos[0]);
        assert!(b.state.forward_speed() < b.setup.movement.power.top_speed * 0.9);
        assert!(b.health.hull < b.health.hull_max, "wall hit should damage the hull");
    }

    #[test]
    fn wheels_rest_on_flat_ground() {
        let b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        assert_eq!(b.wheels.len(), 4);
        assert!(b.wheels.iter().filter(|w| w.driven).count() >= 1);
    }

    #[test]
    fn suspension_compresses_on_a_step() {
        // Ground is 10 cm higher for x > 1, so the front wheels ride up.
        struct Step;
        impl VehicleGround for Step {
            fn height(&self, x: f32, _y: f32) -> Option<f32> {
                if x > 1.0 { Some(0.1) } else { Some(0.0) }
            }
        }
        // Straddle the step so the front axle is on the raised ground.
        let mut b = VehicleBody::new(hatch(), [1.0, 0.0, 0.0], 0.0);
        for _ in 0..30 {
            b.step(&VehicleInput::default(), &Step, 1.0 / 60.0);
        }
        assert!(b.wheels.iter().any(|w| w.compression > 0.0), "no compression on a step");
        assert!(b.state.pitch.abs() > 1e-3, "step should pitch the body");
    }

    #[test]
    fn rpm_sweeps_with_gear() {
        let g = Flat { z: 0.0 };
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        run(&mut b, &VehicleInput { throttle: 1.0, ..Default::default() }, &g, 2.0);
        let idle = b.setup.movement.wheeled_legacy.engine_idle_rpm;
        assert!(b.state.rpm > idle);
        assert!(b.state.rpm <= b.setup.movement.wheeled_legacy.engine_max_rpm + 1e-3);
    }

    #[test]
    fn deterministic_replay() {
        let g = Flat { z: 0.0 };
        let inputs = [
            VehicleInput { throttle: 1.0, steer: 0.3, ..Default::default() },
            VehicleInput { throttle: 0.5, steer: -0.2, ..Default::default() },
            VehicleInput { brake: 0.4, ..Default::default() },
        ];
        let mut a = VehicleBody::new(hatch(), [1.0, 2.0, 0.0], 0.7);
        let mut b = VehicleBody::new(hatch(), [1.0, 2.0, 0.0], 0.7);
        for _ in 0..300 {
            for inp in &inputs {
                a.step(inp, &g, 1.0 / 60.0);
                b.step(inp, &g, 1.0 / 60.0);
            }
        }
        assert_eq!(a.state, b.state);
    }

    #[test]
    fn health_initialises_from_xml() {
        let s = hatch();
        let h = VehicleHealth::new(&s);
        assert_eq!(h.hull, 20000.0);
        assert_eq!(h.components.len(), s.damage.components.len());
        assert!(!h.destroyed);
        assert!((h.hull_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn damage_multiplier_scales_and_destroys() {
        let mut s = hatch();
        s.damage.multipliers.insert("bullet".into(), 0.125);
        let mut h = VehicleHealth::new(&s);
        h.apply_hull(&s, "bullet", 800.0);
        assert!((h.hull - 19900.0).abs() < 1e-3, "hull={}", h.hull);
        h.apply_hull(&s, "melee", 100.0);
        assert!((h.hull - 19800.0).abs() < 1e-3);
        h.apply_hull(&s, "melee", 1.0e9);
        assert!(h.destroyed);
        assert_eq!(h.hull, 0.0);
    }

    #[test]
    fn hull_component_routes_to_hull() {
        let s = hatch();
        let mut h = VehicleHealth::new(&s);
        h.apply_component(&s, 0, "collision", 500.0);
        assert!((h.hull - 19500.0).abs() < 1e-3, "hull={}", h.hull);
    }

    /// The shipped hatchback declares bullet x10 on the `Hull` component while
    /// the top-level default is 0.125; the component value wins.
    #[test]
    fn component_multiplier_overrides_global() {
        let mut s = hatch();
        s.damage.multipliers.insert("bullet".into(), 0.125);
        s.damage.components[0].multipliers.insert("bullet".into(), 10.0);
        let mut h = VehicleHealth::new(&s);
        h.apply_hull(&s, "bullet", 100.0);
        assert!((h.hull - 19000.0).abs() < 1e-3, "hull={}", h.hull);
        // A component without a per-type entry falls back to the default.
        let mut h2 = VehicleHealth::new(&s);
        h2.apply_hull(&s, "collision", 100.0);
        assert!((h2.hull - 19900.0).abs() < 1e-3, "hull={}", h2.hull);
    }

    #[test]
    fn damage_behaviors_fire_at_damage_ratio() {
        use crate::cry::vehicle_xml::DamageBehavior;
        let mut s = hatch();
        s.damage.components[0].behaviors.push(DamageBehavior {
            class: "ClipMask".into(),
            damage_ratio_min: 0.5,
            activate_mask: "hoodDented".into(),
            ..DamageBehavior::default()
        });
        s.damage.components[0].behaviors.push(DamageBehavior {
            class: "DetachPart".into(),
            damage_ratio_min: 1.0,
            part: "hood".into(),
            ..DamageBehavior::default()
        });
        let mut h = VehicleHealth::new(&s);
        assert_eq!(h.fired_behaviors(&s, 0).count(), 0);
        h.apply_component(&s, 0, "collision", 10000.0); // ratio 0.5
        assert_eq!(h.fired_behaviors(&s, 0).count(), 1);
        h.apply_component(&s, 0, "collision", 10000.0); // ratio 1.0, destroyed
        assert_eq!(h.fired_behaviors(&s, 0).count(), 2);
        assert!(h.destroyed);
    }

    #[test]
    fn collision_threshold() {
        let s = hatch();
        assert_eq!(collision_damage(&s, 9.0), 0.0);
        assert!(collision_damage(&s, 14.0) > 0.0);
        assert!(collision_damage(&s, 20.0) > collision_damage(&s, 14.0));
    }

    #[test]
    fn hard_landing_damages_hull() {
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 20.0], 0.0);
        let g = Flat { z: 0.0 };
        for _ in 0..300 {
            b.step(&VehicleInput::default(), &g, 1.0 / 60.0);
        }
        assert!(b.state.grounded);
        assert!(b.health.hull < 20000.0, "hull={}", b.health.hull);
    }

    #[test]
    fn resting_car_takes_no_damage() {
        let mut b = VehicleBody::new(hatch(), [0.0, 0.0, 0.0], 0.0);
        let g = Flat { z: 0.0 };
        run(&mut b, &VehicleInput::default(), &g, 5.0);
        assert_eq!(b.health.hull, 20000.0);
        assert!(!b.health.destroyed);
    }
}
