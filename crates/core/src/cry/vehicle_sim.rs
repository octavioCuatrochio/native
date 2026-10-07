//! Arcade wheeled-vehicle dynamics driven by the original per-class XML.
//!
//! The engine registers Class3's `"ArcadeWheeled"` movement class at
//! `0x01552970` (factory `0x015527c0`), and the whole parameter set is the
//! `<MovementParams><ArcadeWheeled>` block recovered by
//! [`crate::cry::vehicle_xml`]. This module is a native arcade integration of
//! those parameters: a bicycle-model heading with a decaying lateral velocity
//! (the wheel grip / slip), the recovered steering curve, engine acceleration
//! with the `<WheelSpin>` multiplier, coast/brake/hand-brake resistance, fake
//! gears and gravity, evaluated against our collision world/terrain.
//!
//! It is deliberately *not* a full rigid-body tyre solver (the original arcade
//! class does not expose one either); every coefficient below comes from the
//! shipped XML:
//!
//! * `Power.acceleration` / `decceleration` / `top_speed` / `reverse_speed`
//!   — longitudinal engine and coasting terms.
//! * `WheelSpin.accel_multiplier1/2` and `grip1/2`, `grip_recover_speed` —
//!   low-speed torque boost and lateral grip.
//! * `Brakes` / `HandBrake` — additional deceleration and the locked-axle
//!   friction scale used for hand-brake slides.
//! * `FakeGears.ratios` — upshift fractions of top speed; `WheeledLegacy`
//!   supplies the RPM band.
//! * `ArcadeWheeled::max_steer` — the `v0SteerMax - t*kvSteerMax` curve,
//!   already recovered in [`crate::cry::vehicle_xml`].
//!
//! Units are SI: metres, seconds, radians, m/s. The fixed step is supplied by
//! the caller so replays are deterministic.

use crate::cry::vehicle_xml::VehicleSetup;

/// Fixed simulation step used by callers that want the engine's `maxTimeStep`.
pub const DEFAULT_DT: f32 = 1.0 / 30.0;

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
}

impl VehicleGround for crate::cry::phys_world::CollisionWorld {
    fn height(&self, x: f32, y: f32) -> Option<f32> {
        self.terrain_height(x, y)
    }
    fn normal(&self, x: f32, y: f32) -> [f32; 3] {
        self.terrain_normal(x, y)
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
        VehicleInput {
            throttle: 0.0,
            steer: 0.0,
            brake: 0.0,
            handbrake: false,
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
    /// Body pitch (radians, +nose-up) derived from the ground normal.
    pub pitch: f32,
    /// Body roll (radians, +right-down) derived from the ground normal.
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
/// `hullAffection="1"` passes its damage straight to the hull. Only the
/// top-level multipliers are used here (per-component multipliers are not
/// parsed yet).
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
        VehicleHealth {
            hull: hull_max,
            hull_max,
            components,
            destroyed: false,
        }
    }

    /// Hull fraction remaining, `0..=1` (1 when there is no hull component).
    pub fn hull_ratio(&self) -> f32 {
        if self.hull_max > 0.0 {
            (self.hull / self.hull_max).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Apply `amount` of `damage_type` to the hull (scaled by the XML
    /// top-level multiplier for that type).
    pub fn apply_hull(&mut self, setup: &VehicleSetup, damage_type: &str, amount: f32) {
        let dmg = amount.max(0.0) * setup.damage.multiplier(damage_type);
        self.hull -= dmg;
        self.check();
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
        let dmg = amount.max(0.0) * setup.damage.multiplier(damage_type);
        if let Some(h) = self.components.get_mut(idx) {
            *h -= dmg;
        }
        if comp.name.eq_ignore_ascii_case("hull") || comp.hull_affection > 0.0 {
            let affect = if comp.name.eq_ignore_ascii_case("hull") {
                1.0
            } else {
                comp.hull_affection
            };
            self.hull -= dmg * affect;
        }
        self.check();
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
    shift_timer: f32,
}

impl VehicleBody {
    /// Create a body from a parsed setup, resting on the ground at `pos`.
    pub fn new(setup: VehicleSetup, pos: [f32; 3], yaw: f32) -> VehicleBody {
        let health = VehicleHealth::new(&setup);
        VehicleBody {
            setup,
            state: VehicleState::at(pos, yaw),
            health,
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

    /// Advance one fixed step. `ground` supplies terrain height and normal.
    pub fn step(&mut self, input: &VehicleInput, ground: &impl VehicleGround, dt: f32) {
        if !(dt > 0.0) {
            return;
        }
        let m = &self.setup.movement;
        let throttle = input.throttle.clamp(-1.0, 1.0);
        let steer_in = input.steer.clamp(-1.0, 1.0);
        let brake = input.brake.clamp(0.0, 1.0);

        // --- steering -------------------------------------------------------
        // The recovered curve caps the lock by speed; reverse uses the rv* pair.
        let vlong = self.state.forward_speed();
        let full_lock = m.max_steer(vlong);
        let target = steer_in * full_lock;
        let rate_deg = if steer_in.abs() < 1e-3 {
            // Let go: spring back to centre at the relaxation rate.
            m.steer_relaxation
        } else if vlong >= 0.0 {
            m.steer_speed
        } else {
            m.reverse_steer_speed
        };
        let max_delta = rate_deg.to_radians().max(0.0) * dt;
        let delta = (target - self.state.steer).clamp(-max_delta, max_delta);
        self.state.steer += delta;

        // --- longitudinal ---------------------------------------------------
        let f = self.state.forward();
        let r = self.state.right();
        let vlat = self.state.vel[0] * r[0] + self.state.vel[1] * r[1];

        let accel_mult = m.accel_multiplier(vlong);
        let mut a_long = 0.0;
        if throttle > 0.0 {
            a_long += m.power.acceleration * throttle * accel_mult;
        } else if throttle < 0.0 {
            // Reverse is the same engine behind the reverse gear: only useful
            // while not already rolling forward fast.
            let reverse_bite = (1.0 - (vlong / m.power.top_speed.max(1e-3)).max(0.0)).clamp(0.0, 1.0);
            a_long += m.power.acceleration * throttle * reverse_bite;
        }

        let mut next_long = vlong + a_long * dt;

        // Resistance: coasting always, plus foot/hand brake.
        let mut decel = m.power.decceleration.max(0.0);
        if brake > 0.0 {
            decel = decel.max(m.brakes.decceleration * brake);
        }
        if input.handbrake {
            decel = decel.max(m.hand_brake.decceleration);
        }
        let dv = decel * dt;
        if next_long.abs() <= dv {
            next_long = 0.0;
        } else {
            next_long -= next_long.signum() * dv;
        }
        // Clamp to the forward/reverse top speeds from the XML.
        next_long = next_long.clamp(-m.power.reverse_speed, m.power.top_speed);

        // --- lateral grip / slip -------------------------------------------
        // grip1 is the steady lateral decay rate; the hand brake locks the
        // back axle (backFrictionScale) so the car slides.
        let mut grip_rate = m.wheel_spin.grip1.max(0.0);
        if grip_rate <= 0.0 {
            grip_rate = 1.0;
        }
        if input.handbrake {
            let scale = m.hand_brake.back_friction_scale.clamp(0.0, 1.0);
            grip_rate *= scale;
        }
        let next_lat = vlat * (-grip_rate * dt).exp();

        // --- heading (bicycle model) ---------------------------------------
        let wb = self.wheelbase();
        let steer_eff = if self.state.grounded { self.state.steer } else { self.state.steer * 0.25 };
        let yaw_rate = if vlong.abs() > 0.05 {
            (vlong / wb) * steer_eff.tan()
        } else {
            0.0
        };

        // Recompose world velocity in the *pre-turn* body frame: the heading
        // is rotated separately, so the lateral component next step is exactly
        // the slip that grip has to remove. (Using the post-turn frame would
        // rotate the velocity with the car and make every surface grippy.)
        let vx = f[0] * next_long + r[0] * next_lat;
        let vy = f[1] * next_long + r[1] * next_lat;
        self.state.vel = [vx, vy];
        self.state.slip = next_lat.abs();
        self.state.odometer += (vx * vx + vy * vy).sqrt() * dt;

        self.state.yaw = wrap_pi(self.state.yaw + yaw_rate * dt);

        // --- move -----------------------------------------------------------
        self.state.pos[0] += vx * dt;
        self.state.pos[1] += vy * dt;

        // --- vertical / ground ---------------------------------------------
        let gx = self.state.pos[0];
        let gy = self.state.pos[1];
        let was_airborne = !self.state.grounded;
        let mut impact = 0.0f32;
        if let Some(gh) = ground.height(gx, gy) {
            if self.state.pos[2] <= gh + 0.05 && self.state.vertical_speed <= 0.0 {
                if was_airborne {
                    impact = (-self.state.vertical_speed).max(0.0);
                }
                self.state.pos[2] = gh;
                self.state.vertical_speed = 0.0;
                self.state.grounded = true;
            } else {
                self.state.vertical_speed += m.gravity_in_air * dt;
                self.state.pos[2] += self.state.vertical_speed * dt;
                if self.state.pos[2] <= gh {
                    if was_airborne {
                        impact = (-self.state.vertical_speed).max(0.0);
                    }
                    self.state.pos[2] = gh;
                    self.state.vertical_speed = 0.0;
                    self.state.grounded = true;
                } else {
                    self.state.grounded = false;
                }
            }
            // Orient the chassis on the slope.
            let n = ground.normal(gx, gy);
            self.state.pitch = slope_pitch(self.state.forward(), n);
            self.state.roll = slope_roll(self.state.right(), n);
        } else {
            // No ground data below us: keep the current height but still fall.
            self.state.vertical_speed += m.gravity_in_air * dt;
            self.state.pos[2] += self.state.vertical_speed * dt;
            self.state.grounded = false;
        }

        // A hard landing damages the hull via the `<Damages>` thresholds.
        if impact > 0.0 {
            let dmg = collision_damage(&self.setup, impact);
            if dmg > 0.0 {
                self.health.apply_hull(&self.setup, "collision", dmg);
            }
        }

        self.update_drive_train(next_long, dt);
    }

    /// Fake gears + RPM from the XML: `ratios` are upshift fractions of top
    /// speed; the minimum change times gate the shifts.
    fn update_drive_train(&mut self, vlong: f32, dt: f32) {
        let m = &self.setup.movement;
        let ratios = &m.fake_gears.ratios;
        let top = m.power.top_speed.max(1e-3);
        let frac = (vlong / top).clamp(0.0, 1.0);

        let target_gear = ratios.iter().take_while(|&&r| frac >= r).count();
        self.shift_timer += dt;
        if target_gear > self.state.gear {
            if self.shift_timer >= m.fake_gears.min_change_up_time {
                self.state.gear = target_gear;
                self.shift_timer = 0.0;
            }
        } else if target_gear < self.state.gear {
            if self.shift_timer >= m.fake_gears.min_change_down_time {
                self.state.gear = target_gear;
                self.shift_timer = 0.0;
            }
        }

        // Local fraction inside the current gear band -> RPM sweep.
        let lo = if self.state.gear == 0 {
            0.0
        } else {
            ratios.get(self.state.gear - 1).copied().unwrap_or(0.0)
        };
        let hi = ratios
            .get(self.state.gear)
            .copied()
            .unwrap_or(1.0)
            .max(lo + 1e-3);
        let t = ((frac - lo) / (hi - lo)).clamp(0.0, 1.0);
        let idle = m.wheeled_legacy.engine_idle_rpm;
        let max = m.wheeled_legacy.engine_max_rpm.max(idle);
        self.state.rpm = idle + (max - idle) * t;
    }

    /// Full snapshots for replay/telemetry.
    pub fn snapshot(&self) -> (VehicleState, f32) {
        (self.state, self.setup.movement.power.top_speed)
    }
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

/// Nose-up pitch from the slope along `forward` (positive when climbing).
fn slope_pitch(forward: [f32; 2], n: [f32; 3]) -> f32 {
    let along = forward[0] * n[0] + forward[1] * n[1];
    along.atan2(n[2].max(1e-3))
}

/// Roll from the slope along `right` (positive when the right side is low).
fn slope_roll(right: [f32; 2], n: [f32; 3]) -> f32 {
    let along = right[0] * n[0] + right[1] * n[1];
    -along.atan2(n[2].max(1e-3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cry::vehicle_xml::VehicleSetup;

    /// Flat ground at height `z`, optionally with a constant slope.
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
        // Stays on flat ground.
        assert!((b.state.pos[2] - 0.0).abs() < 1e-4);
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
        // Right turn mirrors it.
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
        // Ground exists only for x < 0; beyond it the car free-falls.
        struct Ledge;
        impl VehicleGround for Ledge {
            fn height(&self, x: f32, _y: f32) -> Option<f32> {
                if x < 0.0 {
                    Some(0.0)
                } else {
                    None
                }
            }
        }
        let mut b = VehicleBody::new(hatch(), [-2.0, 0.0, 0.0], 0.0);
        b.state.pos[2] = 0.0;
        // Give it forward velocity so it rolls off the edge.
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
        // Let it settle.
        for _ in 0..10 {
            b.step(&VehicleInput::default(), &g, 1.0 / 60.0);
        }
        assert!((b.state.pos[2] - g.height(b.state.pos[0], b.state.pos[1]).unwrap()).abs() < 1e-3);
        assert!(b.state.pitch.abs() > 1e-3, "pitch={}", b.state.pitch);
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
        // 800 * 0.125 = 100 damage.
        assert!((h.hull - 19900.0).abs() < 1e-3, "hull={}", h.hull);
        // Melee is unspecified -> multiplier 1.
        h.apply_hull(&s, "melee", 100.0);
        assert!((h.hull - 19800.0).abs() < 1e-3);
        // Enough damage to destroy it.
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

    #[test]
    fn collision_threshold() {
        let s = hatch();
        assert_eq!(collision_damage(&s, 9.0), 0.0);
        assert!(collision_damage(&s, 14.0) > 0.0);
        // Monotonic in speed.
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
