//! Native entity store: the CryEngine `IEntitySystem` replacement.
//!
//! Everything here is in **CryEngine space** (Z-up, metres, quaternions as
//! `(x, y, z, w)`). Conversion to Bevy's Y-up happens only at the render
//! boundary, so script-visible values are bit-for-bit what the original engine
//! would hand to Lua.

use std::collections::{BTreeMap, HashMap};

pub type EntityId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlotKind {
    #[default]
    Empty,
    /// `LoadObject` (`.cgf`/`.cga`).
    Static,
    /// `LoadCharacter` (`.chr`/`.cdf`).
    Character,
    /// `LoadParticleEffect`.
    Particle,
    /// `LoadLight`.
    Light,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    pub kind: SlotKind,
    pub path: String,
    /// `DrawSlot(slot, 1)`; CryEngine draws a freshly loaded slot.
    pub visible: bool,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    pub scale: [f32; 3],
}

impl Slot {
    pub fn new(kind: SlotKind, path: impl Into<String>) -> Self {
        Slot {
            kind,
            path: path.into(),
            visible: true,
            pos: [0.0; 3],
            rot: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    }
}

/// `Physicalize` type (`PE_*` in Lua).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhysKind {
    #[default]
    None,
    Static,
    Rigid,
    Living,
    WheeledVehicle,
    Particle,
    Area,
    Articulated,
    Rope,
    Soft,
}

impl PhysKind {
    pub fn from_pe(v: i64) -> Self {
        match v {
            1 => PhysKind::Static,
            2 => PhysKind::Rigid,
            3 => PhysKind::WheeledVehicle,
            4 => PhysKind::Living,
            5 => PhysKind::Particle,
            6 => PhysKind::Articulated,
            7 => PhysKind::Rope,
            8 => PhysKind::Soft,
            9 => PhysKind::Area,
            _ => PhysKind::None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EntityData {
    pub id: EntityId,
    pub name: String,
    pub class: String,
    pub guid: Option<String>,
    pub layer: Option<String>,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    pub scale: [f32; 3],
    pub parent: Option<EntityId>,
    pub hidden: bool,
    pub invisible: bool,
    pub flags: u32,
    pub slots: Vec<Option<Slot>>,
    /// `self:Activate(1)`: receives `OnUpdate` every frame.
    pub active: bool,
    pub physics: PhysKind,
    pub mass: f32,
    pub velocity: [f32; 3],
    pub view_dist_ratio: i32,
    pub lod_ratio: i32,
    /// Trigger bounds (local space) from `SetTriggerBBox`.
    pub trigger_bbox: Option<([f32; 3], [f32; 3])>,
    /// Current script state name (`""` = no state).
    pub state: String,
    pub removed: bool,
}

impl EntityData {
    pub fn new(id: EntityId, name: impl Into<String>, class: impl Into<String>) -> Self {
        EntityData {
            id,
            name: name.into(),
            class: class.into(),
            guid: None,
            layer: None,
            pos: [0.0; 3],
            rot: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            parent: None,
            hidden: false,
            invisible: false,
            flags: 0,
            slots: Vec::new(),
            active: false,
            physics: PhysKind::None,
            mass: 0.0,
            velocity: [0.0; 3],
            view_dist_ratio: 100,
            lod_ratio: 100,
            trigger_bbox: None,
            state: String::new(),
            removed: false,
        }
    }

    pub fn slot(&self, i: usize) -> Option<&Slot> {
        self.slots.get(i).and_then(|s| s.as_ref())
    }

    pub fn set_slot(&mut self, i: usize, slot: Option<Slot>) {
        if self.slots.len() <= i {
            self.slots.resize(i + 1, None);
        }
        self.slots[i] = slot;
    }

    /// Index of the first free slot (CryEngine's `-1` slot argument).
    pub fn free_slot_index(&self) -> usize {
        self.slots.iter().position(|s| s.is_none()).unwrap_or(self.slots.len())
    }

    /// World direction of a local axis (0 = X/right, 1 = Y/forward, 2 = Z/up).
    pub fn axis(&self, axis: usize) -> [f32; 3] {
        let [x, y, z, w] = self.rot;
        match axis {
            0 => [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
            2 => [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
            _ => [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
        }
    }
}

/// Change notifications for the presentation layer (Bevy).
#[derive(Debug, Clone, PartialEq)]
pub enum WorldEvent {
    Spawned(EntityId),
    Removed(EntityId),
    Moved(EntityId),
    SlotsChanged(EntityId),
    VisibilityChanged(EntityId),
}

#[derive(Default)]
pub struct EntityWorld {
    pub entities: BTreeMap<EntityId, EntityData>,
    by_name: HashMap<String, EntityId>,
    next_id: EntityId,
    pub events: Vec<WorldEvent>,
    /// Game clock in milliseconds (`System.GetCurrTime` * 1000).
    pub clock_ms: i64,
    pub frame_time: f32,
}

impl EntityWorld {
    pub fn new() -> Self {
        EntityWorld { next_id: 1, ..Default::default() }
    }

    /// Insert an entity, keeping its level-authored id when free.
    pub fn insert(&mut self, mut e: EntityData) -> EntityId {
        if e.id == 0 || self.entities.contains_key(&e.id) {
            e.id = self.alloc_id();
        }
        self.next_id = self.next_id.max(e.id + 1);
        let id = e.id;
        if !e.name.is_empty() {
            self.by_name.insert(e.name.to_ascii_lowercase(), id);
        }
        self.entities.insert(id, e);
        self.events.push(WorldEvent::Spawned(id));
        id
    }

    pub fn alloc_id(&mut self) -> EntityId {
        while self.entities.contains_key(&self.next_id) || self.next_id == 0 {
            self.next_id += 1;
        }
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn get(&self, id: EntityId) -> Option<&EntityData> {
        self.entities.get(&id).filter(|e| !e.removed)
    }

    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut EntityData> {
        self.entities.get_mut(&id).filter(|e| !e.removed)
    }

    /// `System.GetEntityByName` (case-insensitive, like CryEngine).
    pub fn find_by_name(&self, name: &str) -> Option<EntityId> {
        self.by_name
            .get(&name.to_ascii_lowercase())
            .copied()
            .filter(|id| self.get(*id).is_some())
    }

    pub fn rename(&mut self, id: EntityId, name: &str) {
        if let Some(e) = self.entities.get_mut(&id) {
            self.by_name.remove(&e.name.to_ascii_lowercase());
            e.name = name.to_string();
            self.by_name.insert(name.to_ascii_lowercase(), id);
        }
    }

    /// Mark for removal; the entity stays readable until [`Self::flush_removed`],
    /// matching CryEngine's deferred `RemoveEntity`.
    pub fn remove(&mut self, id: EntityId) {
        if let Some(e) = self.entities.get_mut(&id) {
            if !e.removed {
                e.removed = true;
                self.events.push(WorldEvent::Removed(id));
            }
        }
    }

    pub fn flush_removed(&mut self) {
        let dead: Vec<EntityId> =
            self.entities.iter().filter(|(_, e)| e.removed).map(|(id, _)| *id).collect();
        for id in dead {
            if let Some(e) = self.entities.remove(&id) {
                let key = e.name.to_ascii_lowercase();
                if self.by_name.get(&key) == Some(&id) {
                    self.by_name.remove(&key);
                }
            }
        }
    }

    pub fn set_pos(&mut self, id: EntityId, pos: [f32; 3]) {
        if let Some(e) = self.get_mut(id) {
            e.pos = pos;
            self.events.push(WorldEvent::Moved(id));
        }
    }

    pub fn set_rot(&mut self, id: EntityId, rot: [f32; 4]) {
        if let Some(e) = self.get_mut(id) {
            e.rot = rot;
            self.events.push(WorldEvent::Moved(id));
        }
    }

    /// World position, composing parents.
    pub fn world_pos(&self, id: EntityId) -> [f32; 3] {
        let Some(e) = self.get(id) else { return [0.0; 3] };
        match e.parent.and_then(|p| self.get(p)) {
            Some(p) => {
                let off = quat_rotate(p.rot, [e.pos[0] * p.scale[0], e.pos[1] * p.scale[1], e.pos[2] * p.scale[2]]);
                let pp = self.world_pos(p.id);
                [pp[0] + off[0], pp[1] + off[1], pp[2] + off[2]]
            }
            None => e.pos,
        }
    }

    pub fn ids_in_sphere(&self, center: [f32; 3], radius: f32) -> Vec<EntityId> {
        let r2 = radius * radius;
        self.entities
            .values()
            .filter(|e| !e.removed)
            .filter(|e| dist2(e.pos, center) <= r2)
            .map(|e| e.id)
            .collect()
    }

    pub fn drain_events(&mut self) -> Vec<WorldEvent> {
        std::mem::take(&mut self.events)
    }
}

pub fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let [qx, qy, qz, qw] = q;
    let t = [
        2.0 * (qy * v[2] - qz * v[1]),
        2.0 * (qz * v[0] - qx * v[2]),
        2.0 * (qx * v[1] - qy * v[0]),
    ];
    [
        v[0] + qw * t[0] + (qy * t[2] - qz * t[1]),
        v[1] + qw * t[1] + (qz * t[0] - qx * t[2]),
        v[2] + qw * t[2] + (qx * t[1] - qy * t[0]),
    ]
}

/// CryEngine `Ang3` (radians, XYZ Euler) → quaternion `(x, y, z, w)`, as
/// `Quat::CreateRotationXYZ`.
pub fn quat_from_angles(a: [f32; 3]) -> [f32; 4] {
    let (sx, cx) = (a[0] * 0.5).sin_cos();
    let (sy, cy) = (a[1] * 0.5).sin_cos();
    let (sz, cz) = (a[2] * 0.5).sin_cos();
    [
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
        cz * cy * cx + sz * sy * sx,
    ]
}

/// Quaternion → CryEngine `Ang3` (`Ang3::GetAnglesXYZ`).
pub fn angles_from_quat(q: [f32; 4]) -> [f32; 3] {
    let [x, y, z, w] = q;
    let sy = (-2.0 * (x * z - w * y)).clamp(-1.0, 1.0);
    let ay = sy.asin();
    let ax = (2.0 * (y * z + w * x)).atan2(1.0 - 2.0 * (x * x + y * y));
    let az = (2.0 * (x * y + w * z)).atan2(1.0 - 2.0 * (y * y + z * z));
    [ax, ay, az]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_ids_are_kept_and_collisions_realloc() {
        let mut w = EntityWorld::new();
        assert_eq!(w.insert(EntityData::new(1092, "pt", "TagPoint")), 1092);
        let again = w.insert(EntityData::new(1092, "pt2", "TagPoint"));
        assert_ne!(again, 1092);
        assert_eq!(w.find_by_name("PT"), Some(1092));
    }

    #[test]
    fn removal_is_deferred() {
        let mut w = EntityWorld::new();
        let id = w.insert(EntityData::new(0, "a", "BasicEntity"));
        w.remove(id);
        assert!(w.get(id).is_none());
        assert!(w.entities.contains_key(&id));
        w.flush_removed();
        assert!(!w.entities.contains_key(&id));
        assert_eq!(w.find_by_name("a"), None);
    }

    #[test]
    fn angles_roundtrip() {
        let a = [0.3, -0.4, 1.2];
        let b = angles_from_quat(quat_from_angles(a));
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-5, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn forward_axis_of_yaw90() {
        let e = {
            let mut e = EntityData::new(1, "", "");
            e.rot = quat_from_angles([0.0, 0.0, std::f32::consts::FRAC_PI_2]);
            e
        };
        let f = e.axis(1);
        assert!((f[0] + 1.0).abs() < 1e-5 && f[1].abs() < 1e-5, "{f:?}");
    }
}

/// `Quat::CreateRotationVDir`: forward is +Y, no roll.
pub fn quat_from_dir(d: [f32; 3]) -> [f32; 4] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if len < 1e-6 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let d = [d[0] / len, d[1] / len, d[2] / len];
    quat_from_angles([d[2].clamp(-1.0, 1.0).asin(), 0.0, (-d[0]).atan2(d[1])])
}
