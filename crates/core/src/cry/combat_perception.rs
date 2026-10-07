//! Zombie perception: the sight/sound acquisition that feeds the goal pipes'
//! target acquisition.
//!
//! Two layers drive a zombie's senses (`notes/ai-combat.md` §2,
//! `notes/ghidra/zombie_ai.md` §2):
//!
//! * the CryAI defaults on the actor (`basicaitable.lua:78-126`):
//!   `sightrange = 70`, `FOVPrimary = 80`, `FOVSecondary = 160`;
//! * the SoD `ZombieDef` gate from `libs/class3/zombies/zombietypes.xml`,
//!   evaluated per logical zombie by `DeterminePerception` (`0x01321d20`):
//!   per-type `FOV`/`AggroDist`/`BaseAggroDist`, plus the global multipliers in
//!   `libs/class3/rts/zombies.xml` (`BaseFOV = 80`, `AggroDist = 4`,
//!   `DeaggroDist = 30`, `StealthFOVMultiplier = 0.5`).
//!
//! The native runtime only has to decide *whether* a zombie notices the player
//! (or a sound) and which point to hand to the pipe; pathing then runs through
//! [`super::zombie_runtime`]. This module is pure and unit-tested.

/// Perception flag bits, matching `DAT_01defd58[z]`
/// (`notes/ghidra/zombie_ai.md` §2.3).
pub mod flags {
    /// `0x0001` active attention target.
    pub const ATTENTION: u32 = 0x0001;
    /// `0x0002` aware / alerted (confirmed target).
    pub const AWARE: u32 = 0x0002;
    /// `0x0004` heard combat/echo.
    pub const HEARD: u32 = 0x0004;
    /// `0x0008` investigating / lost target.
    pub const INVESTIGATING: u32 = 0x0008;
    /// `0x0010` target in line of sight.
    pub const LOS: u32 = 0x0010;
    /// `0x0040` focus / attackable.
    pub const FOCUS: u32 = 0x0040;
}

/// Global multipliers from `libs/class3/rts/zombies.xml` (`notes/ai-combat.md`
/// §2.2 / §5.1).
pub const BASE_FOV_DEG: f32 = 80.0;
pub const GLOBAL_AGGRO_DIST: f32 = 4.0;
pub const DEAGGRO_DIST: f32 = 30.0;
pub const STEALTH_FOV_MULT: f32 = 0.5;
pub const INVESTIGATION_DIST: f32 = 6.0;
/// Hearing radius a wandering zombie passes to its flock-mate response
/// (`ambient/zombiewander.lua:87-92`); the combat-sound radius is 8.
pub const SOUND_RANGE_WANDER: f32 = 30.0;
pub const COMBAT_SOUND_DIST: f32 = 8.0;

/// Per-type sense parameters from the `ZombieDef` table
/// (`notes/ai-combat.md` §2.2). `FOV` is the full cone in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Senses {
    pub fov_deg: f32,
    /// `AggroDist`: the gate at which the type acquires the player.
    pub sight_range: f32,
    /// `BaseAggroDist`: an inner radius that notices regardless of the cone.
    pub close_range: f32,
    /// `DeaggroDist`: lose the target beyond this.
    pub lose_range: f32,
    /// Hearing radius for sounds.
    pub hearing_range: f32,
}

impl Default for Senses {
    fn default() -> Self {
        Senses {
            fov_deg: 60.0,
            sight_range: 15.0,
            close_range: 2.0,
            lose_range: DEAGGRO_DIST,
            hearing_range: SOUND_RANGE_WANDER,
        }
    }
}

impl Senses {
    /// Per-`EntityClass` values. Ambient zombies share one row; freaks have the
    /// tighter `FreakFeral`/`FreakJuggernaut` FOV 50 / aggro 6 (`§2.2`).
    pub fn for_class(class: &str) -> Senses {
        match class {
            "Feral" | "Juggernaut" => Senses { fov_deg: 50.0, sight_range: 6.0, close_range: 1.5, ..Senses::default() },
            "Bloater" | "Screamer" => Senses { fov_deg: 60.0, sight_range: 15.0, close_range: 2.0, ..Senses::default() },
            "AmbushSitter" => Senses { fov_deg: 90.0, sight_range: 15.0, close_range: 7.5, ..Senses::default() },
            "Dormant" => Senses { fov_deg: 180.0, sight_range: 15.0, close_range: 15.0, ..Senses::default() },
            "InfestationEating" => Senses { fov_deg: 60.0, sight_range: 8.0, close_range: 2.0, ..Senses::default() },
            _ => Senses::default(),
        }
    }
}

/// A perception result: the point the goal pipe should investigate and the
/// flag word (same bit meanings as the engine's `DAT_01defd58`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Perception {
    /// Selected target point, or `None` when nothing is perceived.
    pub target: Option<[f32; 3]>,
    /// Perceived distance to the target (or `f32::INFINITY`).
    pub dist: f32,
    pub flags: u32,
}

impl Perception {
    pub fn sees_player(&self) -> bool {
        self.flags & (flags::AWARE | flags::LOS) == (flags::AWARE | flags::LOS)
    }

    pub fn has_target(&self) -> bool {
        self.target.is_some()
    }
}

/// A sound stimulus a zombie can hear (`Zombie.CreateEcho` / footsteps /
/// gunfire). `radius` is the audible range; `range` is filled by the caller
/// from `§2.2` (walk 2 / run 4 / sprint 6 / rucksack 10, combat 8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundStimulus {
    pub pos: [f32; 3],
    pub radius: f32,
}

/// `DeterminePerception` for one zombie.
///
/// * `los` decides whether the segment is unobstructed (the engine's
///   `"Zombie LOS"` raycast, `FUN_01336870`).
/// * `stealth` multiplies the sight range by
///   [`STEALTH_FOV_MULT`] while the player is crouched/hidden
///   (`notes/ghidra/zombie_ai.md` §2.2).
/// * `twitter`: whether the current player also counts as a heard stimulus for
///   the gametes.
///
/// Sight wins over hearing; inside `close_range` the zombie notices without the
/// FOV cone (the smell/touch branch).
#[allow(clippy::too_many_arguments)]
pub fn perceive(
    zombie_pos: [f32; 3],
    heading: f32,
    senses: &Senses,
    player_pos: [f32; 3],
    stealth: bool,
    los: impl Fn([f32; 3], [f32; 3]) -> bool,
    sound: Option<SoundStimulus>,
) -> Perception {
    let dx = player_pos[0] - zombie_pos[0];
    let dy = player_pos[1] - zombie_pos[1];
    let dist = (dx * dx + dy * dy).sqrt();
    let mut flags = 0u32;

    // Forward axis from the zombie's heading.
    let forward = [heading.cos(), heading.sin()];
    let in_cone = if dist < 1e-4 {
        true
    } else {
        let dot = (forward[0] * dx + forward[1] * dy) / dist;
        // Compare cos(angle) against cos(fov/2) to avoid the trig on the hot path.
        let half = (senses.fov_deg * 0.5).to_radians().cos();
        dot >= half
    };
    let mut sight = senses.sight_range;
    if stealth {
        sight *= STEALTH_FOV_MULT;
    }
    let sees = dist <= sight && (in_cone || dist <= senses.close_range) && los(zombie_pos, player_pos);
    if sees {
        flags |= flags::ATTENTION | flags::AWARE | flags::LOS | flags::FOCUS;
        return Perception { target: Some(player_pos), dist, flags };
    }

    // Hearing: the loudest reachable stimulus, investigated otherwise.
    if let Some(s) = sound {
        let sx = s.pos[0] - zombie_pos[0];
        let sy = s.pos[1] - zombie_pos[1];
        let sd = (sx * sx + sy * sy).sqrt();
        if sd <= s.radius.max(senses.hearing_range) {
            flags |= flags::HEARD | flags::INVESTIGATING;
            return Perception { target: Some(s.pos), dist: sd, flags };
        }
    }

    Perception { target: None, dist, flags }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn always(_a: [f32; 3], _b: [f32; 3]) -> bool {
        true
    }

    #[test]
    fn detects_player_straight_ahead_in_range() {
        let p = perceive([0.0, 0.0, 0.0], 0.0, &Senses::default(), [5.0, 0.0, 0.0], false, always, None);
        assert!(p.sees_player());
        assert_eq!(p.target, Some([5.0, 0.0, 0.0]));
        assert!(p.flags & flags::FOCUS != 0);
    }

    #[test]
    fn fov_cone_rejects_player_behind() {
        let p = perceive([0.0, 0.0, 0.0], 0.0, &Senses::default(), [-5.0, 0.0, 0.0], false, always, None);
        assert!(!p.has_target());
    }

    #[test]
    fn close_range_ignores_fov() {
        // 1.5 m behind is inside the 2.0 m close range.
        let p = perceive([0.0, 0.0, 0.0], 0.0, &Senses::default(), [-1.5, 0.0, 0.0], false, always, None);
        assert!(p.sees_player());
    }

    #[test]
    fn line_of_sight_blocks_sight_but_hearing_survives() {
        let blocked = |_a: [f32; 3], _b: [f32; 3]| false;
        let sound = SoundStimulus { pos: [4.0, 0.0, 0.0], radius: 10.0 };
        let p = perceive([0.0, 0.0, 0.0], 0.0, &Senses::default(), [5.0, 0.0, 0.0], false, blocked, Some(sound));
        assert!(!p.sees_player());
        assert!(p.has_target());
        assert!(p.flags & flags::HEARD != 0);
        assert_eq!(p.target, Some([4.0, 0.0, 0.0]));
    }

    #[test]
    fn stealth_halves_sight_range() {
        // 10 m is inside the 15 m range, but outside the 7.5 m stealth range.
        let vis = perceive([0.0, 0.0, 0.0], 0.0, &Senses::default(), [10.0, 0.0, 0.0], false, always, None);
        assert!(vis.sees_player());
        let hidden = perceive([0.0, 0.0, 0.0], 0.0, &Senses::default(), [10.0, 0.0, 0.0], true, always, None);
        assert!(!hidden.has_target());
    }

    #[test]
    fn freaks_have_tighter_senses() {
        let feral = Senses::for_class("Feral");
        assert_eq!(feral.fov_deg, 50.0);
        assert_eq!(feral.sight_range, 6.0);
        // A player 10 m away is invisible to a feral but seen by a grunt.
        let f = perceive([0.0, 0.0, 0.0], 0.0, &feral, [10.0, 0.0, 0.0], false, always, None);
        assert!(!f.has_target());
        let g = perceive([0.0, 0.0, 0.0], 0.0, &Senses::for_class("Grunt"), [10.0, 0.0, 0.0], false, always, None);
        assert!(g.has_target());
    }

    #[test]
    fn out_of_range_sound_is_ignored() {
        let sound = SoundStimulus { pos: [50.0, 0.0, 0.0], radius: 8.0 };
        let p = perceive([0.0, 0.0, 0.0], 0.0, &Senses::default(), [40.0, 0.0, 0.0], false, always, Some(sound));
        assert!(!p.has_target());
    }
}
