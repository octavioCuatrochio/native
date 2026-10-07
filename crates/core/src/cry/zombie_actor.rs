//! Per-class zombie actor bodies: the living-entity capsule each zombie class
//! uses in CryPhysics, the character asset that class renders, and the native
//! actor-actor separation pass.
//!
//! # Living capsules
//!
//! Every actor script under `scripts/entities/ai/` inherits `BasicActor`'s
//! `physicsParams.stance[STANCE_STAND]` unless it overrides it. The recovered
//! stand stance is `heightCollider = 1.0`, `size = {x=0.375, y=0.375, z=0.4}`,
//! `useCapsule = 1` (`BasicActor.lua`; the same struct the player uses, see
//! [`crate::class3_physics`]). The capsule radius is `size.x`; the feet-to-top
//! extent is `heightCollider + 2 * size.z` (the same packing the player's
//! capsule uses). Overrides:
//!
//! * `ArmyZombie`/`SwatZombie` (`armyzombie_x.lua`, `swatzombie_x.lua`):
//!   `size = {0.25, 0.25, 0.5}` -> radius `0.25`, height `1.0 + 1.0 = 2.0`.
//! * `Juggernaut` (`juggernaut_x.lua`): `heightCollider = 1.2`,
//!   `size = {0.65, 0.65, 0.3}` -> radius `0.65`, height `1.2 + 0.6 = 1.8`.
//!
//! `Feral`, `Bloater` and `Screamer` do not override the stance, so they use the
//! `BasicActor` default. These are the numbers the original hands to
//! `SEntityPhysicalize` as a living capsule.
//!
//! # Characters
//!
//! `Properties.fileModel` in each `*_x.lua` names the class body; classes with
//! an empty `fileModel` are assembled by the `ZombieVariant` system from a
//! gender/body/head/skin/hair table (`ZombieVariant.xml`). This module maps the
//! classes the spawn director emits to the shipped character asset (the `.cdf`
//! that carries the skeleton, every attachment and its `.mtl` palette) plus the
//! `.$caf` rig whose locomotion bank matches the skeleton:
//!
//! | class | fileModel (script) | native asset | rig |
//! |---|---|---|---|
//! | Grunt / FreshGrunt | *(variant)* | male zombie standard body CDF | `zombie_male` |
//! | ArmyZombie / SwatZombie | `characters/freaks/armored/armored.chr` | `armored.cdf` | `freaks/armored` |
//! | Feral | `characters/freaks/feral/feral.cdf` | same | `freaks/feral` |
//! | Bloater | `characters/freaks/bloater/bloater.cdf` | same | `freaks/bloater` |
//! | Screamer | `characters/freaks/screamer/screamer.cdf` | same | `freaks/screamer` |
//! | Juggernaut | `characters/freaks/juggernaut/juggernaut.cdf` | same | `freaks/juggernaut` |
//!
//! The `.cdf` sibling is preferred over a bare `fileModel` `.chr` so the
//! skeleton, all attachments and the explicit `Material=` overrides come from
//! one place, exactly as the player's character is loaded.
//!
//! # Separation
//!
//! CryPhysics resolves living-entity overlaps with the capsule contacts. The
//! native pass here separates overlapping vertical capsules horizontally (the
//! common ground case) and leaves vertically-disjoint capsules alone, matching
//! "two actors standing on different floors do not collide".

use super::phys_world::CollisionWorld;

/// `BasicActor` stand-stance capsule radius (`size.x`).
pub const DEFAULT_RADIUS: f32 = 0.375;
/// `BasicActor` stand-stance `size.z` (step-over height).
pub const DEFAULT_SIZE_Z: f32 = 0.4;
/// `BasicActor` stand-stance `heightCollider`.
pub const DEFAULT_HEIGHT_COLLIDER: f32 = 1.0;
/// Default feet-to-top extent: `heightCollider + 2 * size.z` = `1.8`.
pub const DEFAULT_HEIGHT: f32 = DEFAULT_HEIGHT_COLLIDER + 2.0 * DEFAULT_SIZE_Z;

/// A vertical living capsule (`feet .. feet + height`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    /// Cylinder radius (m).
    pub radius: f32,
    /// Feet-to-top extent (m).
    pub height: f32,
}

impl Capsule {
    /// Construct a capsule, clamping degenerate input to the default stance.
    pub fn new(radius: f32, height: f32) -> Capsule {
        if radius.is_finite() && radius > 0.0 && height.is_finite() && height > 0.0 {
            Capsule { radius, height }
        } else {
            Capsule { radius: DEFAULT_RADIUS, height: DEFAULT_HEIGHT }
        }
    }
}

impl Default for Capsule {
    fn default() -> Self {
        Capsule { radius: DEFAULT_RADIUS, height: DEFAULT_HEIGHT }
    }
}

/// The living capsule a zombie class physicalizes with.
pub fn capsule_for_class(entity_class: &str) -> Capsule {
    match entity_class {
        // `armyzombie_x.lua` / `swatzombie_x.lua`: size 0.25, size.z 0.5.
        "ArmyZombie" | "SwatZombie" => Capsule { radius: 0.25, height: 2.0 },
        // `juggernaut_x.lua`: heightCollider 1.2, size 0.65, size.z 0.3.
        "Juggernaut" => Capsule { radius: 0.65, height: 1.8 },
        // Grunt/FreshGrunt/Feral/Bloater/Screamer: BasicActor stand stance.
        _ => Capsule::default(),
    }
}

/// A zombie class's renderable character: the shipped `.cdf` and the `.caf` rig
/// its locomotion bank is filtered by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterDef {
    /// Virtual path of the `.cdf` (or `.chr`) to assemble.
    pub path: &'static str,
    /// Rig substring the `ClipBank` scan keeps (`None` would scan everything).
    pub rig: &'static str,
}

/// Map a `ZombieTypes.xml` `EntityClass` to its character asset and rig.
pub fn character_for_class(entity_class: &str) -> CharacterDef {
    match entity_class {
        "ArmyZombie" | "SwatZombie" => CharacterDef {
            // Script `fileModel = characters/freaks/armored/armored.chr`; the CDF
            // adds the freak skeleton and the armored skin layer.
            path: "characters/freaks/armored/armored.cdf",
            rig: "freaks/armored",
        },
        "Feral" => CharacterDef {
            path: "characters/freaks/feral/feral.cdf",
            rig: "freaks/feral",
        },
        "Bloater" => CharacterDef {
            path: "characters/freaks/bloater/bloater.cdf",
            rig: "freaks/bloater",
        },
        "Screamer" => CharacterDef {
            path: "characters/freaks/screamer/screamer.cdf",
            rig: "freaks/screamer",
        },
        "Juggernaut" => CharacterDef {
            path: "characters/freaks/juggernaut/juggernaut.cdf",
            rig: "freaks/juggernaut",
        },
        // Grunt / FreshGrunt: the ZombieVariant system assembles a civilian at
        // spawn. Use the shipped male standard body CDF (head + clothed body)
        // as the native stand-in; it carries the full material palette.
        _ => CharacterDef {
            path: "characters/male/zombie/body/noshirt_jeans/noshirt_jeans.cdf",
            rig: "zombie_male",
        },
    }
}

/// One actor's collision body for the separation pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    /// Feet position (CryEngine Z-up).
    pub pos: [f32; 3],
    /// Capsule radius (m).
    pub radius: f32,
    /// Feet-to-top extent (m).
    pub height: f32,
    /// `true` for bodies that must not be pushed (e.g. the player controller).
    pub fixed: bool,
}

/// Resolve overlapping actor capsules by pushing them apart horizontally.
///
/// Two capsules interact only when their vertical spans overlap; a vertical
/// pair on different floors is left alone. Coincident centres use a
/// deterministic per-index direction so the result is reproducible. Fixed
/// bodies never move; when only one body is movable it absorbs the whole
/// correction. `iterations` passes let chains of overlapping actors relax.
pub fn separate_bodies(bodies: &mut [Body], iterations: usize) {
    let n = bodies.len();
    if n < 2 {
        return;
    }
    for _ in 0..iterations.max(1) {
        for i in 0..n {
            for j in (i + 1)..n {
                let a = bodies[i];
                let b = bodies[j];
                if a.radius <= 0.0 || b.radius <= 0.0 {
                    continue;
                }
                let sum = a.radius + b.radius;
                let dx = b.pos[0] - a.pos[0];
                let dy = b.pos[1] - a.pos[1];
                let horiz = (dx * dx + dy * dy).sqrt();
                if horiz >= sum {
                    continue;
                }
                let v_overlap = (a.pos[2] + a.height)
                    .min(b.pos[2] + b.height)
                    - a.pos[2].max(b.pos[2]);
                if v_overlap <= 0.0 {
                    continue;
                }
                let pen = sum - horiz;
                let (nx, ny) = if horiz > 1e-4 {
                    (dx / horiz, dy / horiz)
                } else {
                    // Exactly coincident: fan out on a golden-angle spiral.
                    let ang = i as f32 * 2.399_963_2;
                    (ang.cos(), ang.sin())
                };
                let (wi, wj) = match (a.fixed, b.fixed) {
                    (true, true) => (0.0, 0.0),
                    (true, false) => (0.0, 1.0),
                    (false, true) => (1.0, 0.0),
                    (false, false) => (0.5, 0.5),
                };
                if wi > 0.0 {
                    bodies[i].pos[0] -= nx * pen * wi;
                    bodies[i].pos[1] -= ny * pen * wi;
                }
                if wj > 0.0 {
                    bodies[j].pos[0] += nx * pen * wj;
                    bodies[j].pos[1] += ny * pen * wj;
                }
            }
        }
    }
}

/// Sweep one actor capsule through the static collision world and return the
/// slid feet position. Thin wrapper over [`CollisionWorld::sweep_capsule`] so
/// callers use the class capsule directly.
pub fn sweep_body(
    collision: &CollisionWorld,
    from: [f32; 3],
    delta: [f32; 3],
    capsule: Capsule,
) -> [f32; 3] {
    collision.sweep_capsule(from, delta, capsule.radius, capsule.height).pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_capsules_match_scripts() {
        // Default BasicActor stand stance.
        let g = capsule_for_class("Grunt");
        assert!((g.radius - 0.375).abs() < 1e-6);
        assert!((g.height - 1.8).abs() < 1e-6);
        // Feral/Bloater/Screamer keep the default.
        assert_eq!(capsule_for_class("Feral"), g);
        assert_eq!(capsule_for_class("Bloater"), g);
        assert_eq!(capsule_for_class("Screamer"), g);
        // Armored: 0.25, heightCollider 1.0 + 2*0.5.
        let a = capsule_for_class("ArmyZombie");
        assert!((a.radius - 0.25).abs() < 1e-6);
        assert!((a.height - 2.0).abs() < 1e-6);
        assert_eq!(capsule_for_class("SwatZombie"), a);
        // Juggernaut: 0.65, 1.2 + 2*0.3.
        let j = capsule_for_class("Juggernaut");
        assert!((j.radius - 0.65).abs() < 1e-6);
        assert!((j.height - 1.8).abs() < 1e-6);
    }

    #[test]
    fn class_characters_are_distinct() {
        let grunt = character_for_class("Grunt");
        assert!(grunt.path.ends_with(".cdf"));
        assert_eq!(grunt.rig, "zombie_male");
        // Every freak maps to its own asset.
        for (class, needle, rig) in [
            ("ArmyZombie", "armored", "freaks/armored"),
            ("SwatZombie", "armored", "freaks/armored"),
            ("Feral", "feral", "freaks/feral"),
            ("Bloater", "bloater", "freaks/bloater"),
            ("Screamer", "screamer", "freaks/screamer"),
            ("Juggernaut", "juggernaut", "freaks/juggernaut"),
        ] {
            let d = character_for_class(class);
            assert!(d.path.contains(needle), "{class} -> {}", d.path);
            assert_eq!(d.rig, rig);
            assert_ne!(d.path, grunt.path, "{class} must not reuse the grunt body");
        }
        // Unknown classes collapse to the grunt stand-in.
        assert_eq!(character_for_class("Whatever").path, grunt.path);
    }

    #[test]
    fn separation_pushes_overlapping_apart() {
        let mut bodies = vec![
            Body { pos: [0.0, 0.0, 0.0], radius: 0.375, height: 1.8, fixed: false },
            Body { pos: [0.1, 0.0, 0.0], radius: 0.375, height: 1.8, fixed: false },
        ];
        separate_bodies(&mut bodies, 4);
        let d = (bodies[1].pos[0] - bodies[0].pos[0]).abs();
        assert!(d >= 0.74, "separated to {d}");
    }

    #[test]
    fn separation_leaves_disjoint_alone() {
        let before = [
            Body { pos: [0.0, 0.0, 0.0], radius: 0.375, height: 1.8, fixed: false },
            Body { pos: [5.0, 0.0, 0.0], radius: 0.375, height: 1.8, fixed: false },
        ];
        let mut bodies = before;
        separate_bodies(&mut bodies, 2);
        assert_eq!(bodies, before);
    }

    #[test]
    fn separation_ignores_different_floors() {
        let before = [
            Body { pos: [0.0, 0.0, 0.0], radius: 0.375, height: 1.8, fixed: false },
            Body { pos: [0.0, 0.0, 5.0], radius: 0.375, height: 1.8, fixed: false },
        ];
        let mut bodies = before;
        separate_bodies(&mut bodies, 2);
        assert_eq!(bodies, before, "no vertical overlap -> no collision");
    }

    #[test]
    fn fixed_body_absorbs_correction() {
        let mut bodies = vec![
            Body { pos: [0.0, 0.0, 0.0], radius: 0.375, height: 1.8, fixed: true },
            Body { pos: [0.2, 0.0, 0.0], radius: 0.375, height: 1.8, fixed: false },
        ];
        separate_bodies(&mut bodies, 4);
        assert_eq!(bodies[0].pos, [0.0, 0.0, 0.0], "fixed body must not move");
        assert!(bodies[1].pos[0] >= 0.74, "movable body took the whole push");
    }

    /// Real-asset check (skipped when the game is not installed): every class's
    /// character assembles with geometry and at least one material slot, and the
    /// freaks do not collapse to the grunt body.
    #[test]
    fn real_class_characters_load_distinct_palettes() {
        use crate::crypak::Vfs;
        use std::path::PathBuf;

        let game = std::env::var_os("SOD_GAME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(
                "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE",
            ));
        if !game.is_dir() {
            return;
        }
        let mut vfs = Vfs::new();
        let mut mounted = 0;
        for rel in ["Game/characters.pak", "Game/animations.pak"] {
            let p = game.join(rel);
            if p.is_file() && vfs.mount(&p).is_ok() {
                mounted += 1;
            }
        }
        if mounted < 2 {
            return;
        }

        let classes = ["Grunt", "ArmyZombie", "Feral", "Bloater", "Screamer", "Juggernaut"];
        let mut paths = Vec::new();
        for class in classes {
            let def = character_for_class(class);
            let Some((skel, parts)) = crate::cdf::load_parts(&vfs, def.path) else {
                eprintln!("zombie_actor: {class} ({}) did not load; skipping", def.path);
                continue;
            };
            assert!(!skel.bones.is_empty(), "{class} skeleton");
            let meshes: usize = parts.iter().map(|p| p.model.meshes.len()).sum();
            let slots: usize = parts.iter().map(|p| p.slots.len()).sum();
            assert!(meshes > 0, "{class} has no geometry");
            assert!(slots > 0, "{class} has no material slots");
            paths.push((class, def.path));
        }
        // Freaks must use their own assets, not the grunt stand-in.
        for (class, path) in &paths {
            if *class != "Grunt" {
                assert_ne!(*path, character_for_class("Grunt").path, "{class}");
            }
        }
    }

    #[test]
    fn coincident_bodies_do_not_panic_and_separate() {
        let mut bodies = vec![
            Body { pos: [1.0, 2.0, 0.0], radius: 0.375, height: 1.8, fixed: false },
            Body { pos: [1.0, 2.0, 0.0], radius: 0.375, height: 1.8, fixed: false },
            Body { pos: [1.0, 2.0, 0.0], radius: 0.375, height: 1.8, fixed: false },
        ];
        separate_bodies(&mut bodies, 8);
        assert!(bodies.iter().all(|b| b.pos.iter().all(|v| v.is_finite())));
    }
}
