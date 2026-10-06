//! Exact, shipped movement / camera / input tuning for State of Decay: YOSE.
//!
//! Every value below is re-derived from the shipped data, never guessed:
//!
//! * `gamedata/scripts/entities/actor/player.lua`     — `stance` table + UL tuning block
//! * `gamedata/scripts/entities/actor/basicactor.lua`  — `jumpHeight = 1.0` (metres)
//! * `gamedata/libs/config/meleeprototype.cfg`         — `cl_cam_*` tuning
//! * `gamedata/libs/config/defaultprofile.xml`         — input action map
//! * `gamedata/libs/config/profiles/default/attributes.xml` — `OptionCfg.cl_sensitivity = 45`
//!
//! The `cl_movement_*` cvars all ship **unset**. The image initialises their
//! float storage to `-1.0` (verified in `bin/StateOfDecay.unpacked.exe` at
//! `.data` globals `0x190f4a8..0x190f4c4`), and `Player::PostPhysicalize`
//! only applies an override when the value is `>= 0`. So the real controller
//! uses the `player.lua` numbers reproduced by [`default_movement`].
//! `cl_movement_jump_height` likewise defaults to `-1` (global `0x190f6d4`);
//! jump height comes from `basicactor.lua` (`1.0` m).

/// Ground and aiming movement speeds (metres per second) plus inertia.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MovementTuning {
    /// `stance[1].creepSpeed` (STANCE_STAND).
    pub creep: f32,
    /// `stance[1].walkSpeed`.
    pub walk: f32,
    /// `stance[1].runSpeed`.
    pub run: f32,
    /// `stance[1].sprintSpeed` (== sprintCreepSpeed == sprintWalkSpeed == 6.8).
    pub sprint: f32,
    /// `stance[2].creepSpeed` (STANCE_AIMING).
    pub aim_creep: f32,
    /// `stance[2].walkSpeed`.
    pub aim_walk: f32,
    /// `stance[2].runSpeed`.
    pub aim_run: f32,
    /// `stance[2].sprintSpeed` (== sprintCreepSpeed == sprintWalkSpeed == 0.5).
    pub aim_sprint: f32,
    /// `backwardMultiplier` — speed scale when moving backward.
    pub backward_mult: f32,
    /// `runAndGunMultiplier` — speed scale in run-and-gun mode.
    pub run_and_gun_mult: f32,
    /// `grabMultiplier` — speed scale while carrying the maximum load.
    pub grab_mult: f32,
    /// `inertia` — velocity-approach rate (1/s) while slowing down or turning
    /// (`v += (target - v) * min(1, inertia * dt)`), see
    /// [`crate::class3_move::integrate_inertia`].
    pub inertia: f32,
    /// `inertiaAccel` — like `inertia`, used while the target speed exceeds
    /// the current speed.
    pub inertia_accel: f32,
    /// `leanAngle` — view rotation (degrees) while leaning.
    pub lean_angle: f32,
    /// `physicsParams.Living.air_resistance`.
    pub air_resistance: f32,
    /// `creepWalkStepThreshold` — controller point below which the player creeps.
    pub threshold_creep_walk: f32,
    /// `walkRunStepThreshold` — controller point above which the player runs.
    pub threshold_walk_run: f32,
    /// `basicactor.lua` `jumpHeight` in metres.
    pub jump_height: f32,
}

/// The exact shipped player movement tuning.
pub fn default_movement() -> MovementTuning {
    MovementTuning {
        creep: 0.8,       // player.lua stance[1].creepSpeed
        walk: 1.66,       // player.lua stance[1].walkSpeed
        run: 4.0,         // player.lua stance[1].runSpeed
        sprint: 6.8,      // player.lua stance[1].sprintSpeed (also sprintCreep/SprintWalk)
        aim_creep: 0.7,   // player.lua stance[2].creepSpeed
        aim_walk: 0.5,    // player.lua stance[2].walkSpeed
        aim_run: 0.5,     // player.lua stance[2].runSpeed
        aim_sprint: 0.5,  // player.lua stance[2].sprintSpeed
        backward_mult: 1.0,     // player.lua backwardMultiplier
        run_and_gun_mult: 1.0,  // player.lua runAndGunMultiplier
        grab_mult: 0.5,         // player.lua grabMultiplier
        inertia: 10.0,          // player.lua inertia
        inertia_accel: 16.0,    // player.lua inertiaAccel (BasicActor uses 11.0)
        lean_angle: 15.0,       // player.lua leanAngle
        air_resistance: 0.0,    // player.lua physicsParams.Living.air_resistance
        threshold_creep_walk: 0.29, // player.lua creepWalkStepThreshold
        threshold_walk_run: 0.69,   // player.lua walkRunStepThreshold
        jump_height: 1.0,           // basicactor.lua jumpHeight (metres)
    }
}

// --- Lenient numeric scanning (no Lua/XML dependency) ------------------------

/// Read the first float that follows `key` on the same line, tolerating
/// `key = v`, `key v`, and `key="v"` forms plus `--`/`//` comments.
fn number_after(hay: &str, key: &str) -> Option<f32> {
    let mut from = 0;
    while let Some(rel) = hay[from..].find(key) {
        let at = from + rel;
        let after = &hay[at + key.len()..];
        // Reject matches that are only a prefix of a longer identifier
        // (e.g. `inertia` inside `inertiaAccel`).
        let prev_ok = hay[..at]
            .chars()
            .next_back()
            .map_or(true, |c| !(c.is_ascii_alphanumeric() || c == '_'));
        let next_ok = after
            .chars()
            .next()
            .map_or(true, |c| !(c.is_ascii_alphanumeric() || c == '_'));
        if prev_ok && next_ok {
            if let Some(v) = scan_number(after) {
                return Some(v);
            }
        }
        from = at + key.len();
    }
    None
}

/// Scan a single line for the first numeric literal, stopping at a comment.
fn scan_number(s: &str) -> Option<f32> {
    let end = s.find('\n').unwrap_or(s.len());
    let mut line = &s[..end];
    if let Some(i) = line.find("--") {
        line = &line[..i];
    }
    if let Some(i) = line.find("//") {
        line = &line[..i];
    }
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        let starts = c.is_ascii_digit()
            || ((c == '-' || c == '+' || c == '.')
                && i + 1 < b.len()
                && (b[i + 1] as char).is_ascii_digit());
        if starts {
            let mut j = i + 1;
            while j < b.len() {
                let d = b[j] as char;
                if d.is_ascii_digit() || matches!(d, '.' | '-' | '+' | 'e' | 'E') {
                    j += 1;
                } else {
                    break;
                }
            }
            if let Ok(v) = line[i..j].parse::<f32>() {
                return Some(v);
            }
        }
        i += 1;
    }
    None
}

/// Parse `player.lua`'s `stance` table and UL tuning block.
///
/// Purely textual: named numeric fields are looked up in document order so the
/// two `stance` entries keep their meaning. Anything missing falls back to
/// [`default_movement`]; this never panics and takes no Lua dependency.
pub fn parse_player_lua(text: &str) -> MovementTuning {
    let mut t = default_movement();
    let stand = text.find("STANCE_STAND");
    let aim = text.find("STANCE_AIMING");

    if let Some(s) = stand {
        let region = &text[s..aim.unwrap_or(text.len())];
        if let Some(v) = number_after(region, "creepSpeed") {
            t.creep = v;
        }
        if let Some(v) = number_after(region, "walkSpeed") {
            t.walk = v;
        }
        if let Some(v) = number_after(region, "runSpeed") {
            t.run = v;
        }
        if let Some(v) = number_after(region, "sprintSpeed") {
            t.sprint = v;
        }
    }
    if let Some(a) = aim {
        let region = &text[a..];
        if let Some(v) = number_after(region, "creepSpeed") {
            t.aim_creep = v;
        }
        if let Some(v) = number_after(region, "walkSpeed") {
            t.aim_walk = v;
        }
        if let Some(v) = number_after(region, "runSpeed") {
            t.aim_run = v;
        }
        if let Some(v) = number_after(region, "sprintSpeed") {
            t.aim_sprint = v;
        }
    }

    if let Some(v) = number_after(text, "runAndGunMultiplier") {
        t.run_and_gun_mult = v;
    }
    if let Some(v) = number_after(text, "backwardMultiplier") {
        t.backward_mult = v;
    }
    if let Some(v) = number_after(text, "grabMultiplier") {
        t.grab_mult = v;
    }
    if let Some(v) = number_after(text, "inertia") {
        t.inertia = v;
    }
    if let Some(v) = number_after(text, "inertiaAccel") {
        t.inertia_accel = v;
    }
    if let Some(v) = number_after(text, "leanAngle") {
        t.lean_angle = v;
    }
    if let Some(v) = number_after(text, "air_resistance") {
        t.air_resistance = v;
    }
    if let Some(v) = number_after(text, "creepWalkStepThreshold") {
        t.threshold_creep_walk = v;
    }
    if let Some(v) = number_after(text, "walkRunStepThreshold") {
        t.threshold_walk_run = v;
    }
    // `jump_height` is not in player.lua; keep basicactor's shipped default.
    t
}

/// Camera rotation speeds/acceleration and mouse sensitivity.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CameraTuning {
    /// Navigation yaw speed — `cl_cam_rotation_speed`.
    pub yaw_speed: f32,
    /// Navigation pitch speed (same shipped base as yaw).
    pub pitch_speed: f32,
    /// `cl_cam_rotation_acceleration_time_yaw`.
    pub accel_time_yaw: f32,
    /// `cl_cam_rotation_acceleration_time_pitch`.
    pub accel_time_pitch: f32,
    /// `cl_sensitivity` (option `OptionCfg.cl_sensitivity`).
    pub sensitivity: f32,
}

/// The exact shipped camera tuning (from `meleeprototype.cfg` + attributes).
pub fn default_camera() -> CameraTuning {
    CameraTuning {
        yaw_speed: 0.04,       // meleeprototype.cfg cl_cam_rotation_speed
        pitch_speed: 0.04,     // same base (no separate shipped pitch cvar)
        accel_time_yaw: 0.3,   // cl_cam_rotation_acceleration_time_yaw
        accel_time_pitch: 0.6, // cl_cam_rotation_acceleration_time_pitch
        sensitivity: 45.0,     // attributes.xml OptionCfg.cl_sensitivity (+ exe default)
    }
}

/// Parse `cl_cam_rotation_speed` and friends out of a CryEngine `.cfg`;
/// missing keys keep their [`default_camera`] value.
pub fn parse_camera_cvars(text: &str) -> CameraTuning {
    let mut c = default_camera();
    if let Some(v) = number_after(text, "cl_cam_rotation_speed") {
        c.yaw_speed = v;
        c.pitch_speed = v;
    }
    if let Some(v) = number_after(text, "cl_cam_yaw_speed") {
        c.yaw_speed = v;
    }
    if let Some(v) = number_after(text, "cl_cam_pitch_speed") {
        c.pitch_speed = v;
    }
    if let Some(v) = number_after(text, "cl_cam_rotation_acceleration_time_yaw") {
        c.accel_time_yaw = v;
    }
    if let Some(v) = number_after(text, "cl_cam_rotation_acceleration_time_pitch") {
        c.accel_time_pitch = v;
    }
    if let Some(v) = number_after(text, "cl_sensitivity") {
        c.sensitivity = v;
    }
    c
}

/// Provenance for each tuning group (file / field).
pub fn parity_notes() -> &'static [&'static str] {
    &[
        "creep/walk/run/sprint + aiming: extracted-game/gamedata/scripts/entities/actor/player.lua gameParams.stance",
        "inertia 10.0 / inertiaAccel 16.0: player.lua UL tuning block",
        "leanAngle 15 / air_resistance 0.0: player.lua UL tuning + physicsParams.Living",
        "backwardMultiplier 1.0 / runAndGunMultiplier 1.0 / grabMultiplier 0.5: player.lua",
        "creepWalkStepThreshold 0.29 / walkRunStepThreshold 0.69: player.lua",
        "jumpHeight 1.0 m: extracted-game/gamedata/scripts/entities/actor/basicactor.lua",
        "cl_movement_* cvars ship unset (-1.0) in bin/StateOfDecay.unpacked.exe .data 0x190f4a8..0x190f4c4",
        "cl_cam_rotation_speed 0.04 / accel time yaw 0.3 / pitch 0.6: gamedata/libs/config/meleeprototype.cfg",
        "cl_sensitivity 45: gamedata/libs/config/profiles/default/attributes.xml (OptionCfg.cl_sensitivity)",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_movement_is_shipped_values() {
        let m = default_movement();
        assert_eq!(m.creep, 0.8);
        assert_eq!(m.walk, 1.66);
        assert_eq!(m.run, 4.0);
        assert_eq!(m.sprint, 6.8);
        assert_eq!(m.aim_creep, 0.7);
        assert_eq!(m.aim_walk, 0.5);
        assert_eq!(m.aim_run, 0.5);
        assert_eq!(m.aim_sprint, 0.5);
        assert_eq!(m.backward_mult, 1.0);
        assert_eq!(m.run_and_gun_mult, 1.0);
        assert_eq!(m.grab_mult, 0.5);
        assert_eq!(m.inertia, 10.0);
        assert_eq!(m.inertia_accel, 16.0);
        assert_eq!(m.lean_angle, 15.0);
        assert_eq!(m.air_resistance, 0.0);
        assert_eq!(m.threshold_creep_walk, 0.29);
        assert_eq!(m.threshold_walk_run, 0.69);
        assert_eq!(m.jump_height, 1.0);
    }

    /// The key parity guarantee: parsing the real file must reproduce the
    /// shipped defaults field-for-field. Skipped silently if absent.
    #[test]
    fn player_lua_reproduces_defaults() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/gamedata/scripts/entities/actor/player.lua");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => return,
        };
        assert_eq!(parse_player_lua(&text), default_movement());
    }

    #[test]
    fn parser_is_lenient_and_falls_back() {
        assert_eq!(parse_player_lua(""), default_movement());
        assert_eq!(parse_player_lua("nonsense = = ="), default_movement());
        // tuning fields without a stance table still parse.
        let t = parse_player_lua("inertia = 7.0\ninertiaAccel = 9.0\nleanAngle = 12\n");
        assert_eq!(t.inertia, 7.0);
        assert_eq!(t.inertia_accel, 9.0);
        assert_eq!(t.lean_angle, 12.0);
    }

    #[test]
    fn camera_cvars_extract_values() {
        let cfg = "\
-- nav camera\n\
cl_cam_rotation_speed 0.05 -- base\n\
cl_cam_rotation_acceleration_time_yaw 0.25\n\
cl_cam_rotation_acceleration_time_pitch 0.55 // comment\n\
cl_sensitivity 60\n";
        let c = parse_camera_cvars(cfg);
        assert_eq!(c.yaw_speed, 0.05);
        assert_eq!(c.pitch_speed, 0.05);
        assert_eq!(c.accel_time_yaw, 0.25);
        assert_eq!(c.accel_time_pitch, 0.55);
        assert_eq!(c.sensitivity, 60.0);

        // unknown / absent keys keep defaults.
        assert_eq!(parse_camera_cvars(""), default_camera());
    }

    #[test]
    fn camera_defaults_are_shipped() {
        let c = default_camera();
        assert_eq!(c.yaw_speed, 0.04);
        assert_eq!(c.pitch_speed, 0.04);
        assert_eq!(c.accel_time_yaw, 0.3);
        assert_eq!(c.accel_time_pitch, 0.6);
        assert_eq!(c.sensitivity, 45.0);
    }
}
