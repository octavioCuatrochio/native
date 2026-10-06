//! Presentation view of the entity world: what the original renderer would
//! draw, derived only from script-driven state (slots loaded by
//! `LoadObject`/`LoadCharacter`, `DrawSlot`, `Hide`), never from the level
//! XML directly.

use super::world::{quat_rotate, EntityId, EntityWorld, SlotKind};
use crate::level::Placement;

#[derive(Debug, Clone, PartialEq)]
pub struct SlotInstance {
    pub entity: EntityId,
    pub slot: usize,
    pub kind: SlotKind,
    pub path: String,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    pub scale: [f32; 3],
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Every drawn slot of every visible entity, in world space (CryEngine axes).
pub fn visible_slots(w: &EntityWorld) -> Vec<SlotInstance> {
    let mut out = Vec::new();
    for e in w.entities.values() {
        if e.removed || e.hidden || e.invisible {
            continue;
        }
        let epos = w.world_pos(e.id);
        for (i, s) in e.slots.iter().enumerate() {
            let Some(s) = s else { continue };
            if !s.visible || s.path.is_empty() {
                continue;
            }
            let local = quat_rotate(e.rot, [s.pos[0] * e.scale[0], s.pos[1] * e.scale[1], s.pos[2] * e.scale[2]]);
            out.push(SlotInstance {
                entity: e.id,
                slot: i,
                kind: s.kind,
                path: s.path.clone(),
                pos: [epos[0] + local[0], epos[1] + local[1], epos[2] + local[2]],
                rot: quat_mul(e.rot, s.rot),
                scale: [e.scale[0] * s.scale[0], e.scale[1] * s.scale[1], e.scale[2] * s.scale[2]],
            });
        }
    }
    out
}

/// Static geometry slots as level `Placement`s, so the existing prop renderer
/// draws exactly what the scripts loaded.
pub fn static_placements(w: &EntityWorld) -> Vec<Placement> {
    visible_slots(w)
        .into_iter()
        .filter(|s| s.kind == SlotKind::Static)
        .map(|s| Placement {
            class: w.get(s.entity).map(|e| e.class.clone()).unwrap_or_default(),
            name: w.get(s.entity).map(|e| e.name.clone()),
            model: Some(s.path),
            pos: s.pos,
            rot: s.rot,
            scale: s.scale,
            is_entity: true,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cry::world::{EntityData, Slot};

    #[test]
    fn hidden_entities_and_slots_are_not_drawn() {
        let mut w = EntityWorld::new();
        let mut a = EntityData::new(1, "a", "BasicEntity");
        a.pos = [1.0, 2.0, 3.0];
        a.set_slot(0, Some(Slot::new(SlotKind::Static, "objects/a.cgf")));
        let mut hidden_slot = Slot::new(SlotKind::Static, "objects/b.cgf");
        hidden_slot.visible = false;
        a.set_slot(1, Some(hidden_slot));
        w.insert(a);
        let mut b = EntityData::new(2, "b", "BasicEntity");
        b.hidden = true;
        b.set_slot(0, Some(Slot::new(SlotKind::Static, "objects/c.cgf")));
        w.insert(b);
        let p = static_placements(&w);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].model.as_deref(), Some("objects/a.cgf"));
        assert_eq!(p[0].pos, [1.0, 2.0, 3.0]);
    }
}
