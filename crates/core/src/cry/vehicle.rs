//! Vehicle content data: the community/RTS definitions and the compiled
//! world-spawn list.
//!
//! State of Decay ships two independent vehicle data layers:
//!
//! * **`VehicleDef`** — the gameplay-facing record used by the RTS/community
//!   layer and the HUD, in
//!   `libs/class3/contentmanager/vehicles.xml` (root `<VehicleData>`). Each
//!   `<VehicleDef>` carries star ratings, seat count, trunk slot count, weight
//!   class and display strings. `VehicleGetItemSlotCount` (engine,
//!   `0x01861a14`) reads the same `InventorySlotCount` value.
//! * **`VehicleSpawnFile`** — the compiled `level/vehicle_spawns_<platform>.markup`
//!   list (`ULVS` container). The runtime reader is
//!   `NSVehicleManager::VehicleManager` at `0x0140b9a0` (error strings
//!   `0x01861130`..`0x01861580`), fed by the path builder `0x0140f110`
//!   (`"%s/vehicle_spawns_%s.markup"`, `0x01861908`). The editable source of
//!   the same list is the level entities of class `ULVehicleSpawnEditor`
//!   (`FindSpawnPoints`, `0x0140d630`, string `0x018617ec`) with properties
//!   `bInitialSpawn` / `sVehicleClass`.
//!
//! Per-class physical parameters (mass, engine, wheels, damage) live in the
//! separately parsed `Scripts/Entities/Vehicles/Implementations/Xml/*.xml`; see
//! [`crate::cry::vehicle_xml`].
//!
//! # ULVS layout (little-endian, recovered from `0x0140b9a0`)
//!
//! ```text
//! 0x00  u32  magic   "ULVS" (LE 0x5356_4C55)
//! 0x04  u32  version 0x0200_0000
//! 0x08  u32  endian  0
//! 0x0c  u32  initialSpawnCount  (<= 0xffff)
//! 0x10  u32  totalSpawnCount    (<= 0xffff, >= initial)
//! 0x14  u32  classNameBufSize   (<= 0xffff)
//! 0x18  ..   pad to 0x80
//!        [total] f32x3  spawn positions   , pad to 0x80
//!        [total] f32x3  spawn euler angles, pad to 0x80
//!        [total] u16    className offsets into the name buffer, pad to 0x80
//!        [classNameBufSize] u8  NUL-terminated class names
//! ```
//!
//! The runtime record stride is `0x20`: `u32` class/entity hash, then the
//! quaternion built from the euler triple, then the position. Only the
//! human-readable euler angles are needed here.
//!
//! Parsing never panics: malformed buffers yield `None`.

use std::collections::BTreeMap;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// Candidate VFS paths for the `<VehicleData>` record file.
pub const VEHICLE_DEFS_PATHS: [&str; 2] = [
    "libs/class3/contentmanager/vehicles.xml",
    "gamedata/libs/class3/contentmanager/vehicles.xml",
];

// ---------------------------------------------------------------------------
// Gameplay/RTS definition
// ---------------------------------------------------------------------------

/// One `<VehicleDef>` record (`libs/class3/contentmanager/vehicles.xml`).
///
/// Ratings are the 0.0..1.0 stars shown in the garage; the actual physics come
/// from the per-class XML ([`crate::cry::vehicle_xml::VehicleSetup`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VehicleDef {
    /// `Id` / entity class (e.g. `Hatchback_modern`).
    pub id: String,
    /// `EntityClass` (usually equal to `Id` for the shipped cars).
    pub entity_class: String,
    /// `SeatCount` (including the driver).
    pub seat_count: u8,
    /// `InventorySlotCount`: trunk size, read by `VehicleGetItemSlotCount`.
    pub inventory_slot_count: u8,
    /// `Weight` class, 0..2 (0 = lightest).
    pub weight: u8,
    /// `AccelerationRating`, 0..1.
    pub acceleration_rating: f32,
    /// `DurabilityRating`, 0..1.
    pub durability_rating: f32,
    /// `HandlingRating`, 0..1.
    pub handling_rating: f32,
    /// `TopSpeedRating`, 0..1.
    pub top_speed_rating: f32,
    /// `MapIcon` name (always `VEHICLE` in the shipped data).
    pub map_icon: String,
    /// `IconId` (HUD icon index).
    pub icon_id: u16,
    /// `<DisplayName Text=...>`.
    pub display_name: String,
    /// `<Description Text=...>`.
    pub description: String,
}

impl VehicleDef {
    /// Number of trunk slots available to the player (same accessor the engine
    /// uses).
    pub fn item_slot_count(&self) -> u8 {
        self.inventory_slot_count
    }
}

/// All `<VehicleDef>` rows, keyed case-insensitively by id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VehicleDefs {
    by_id: BTreeMap<String, VehicleDef>,
}

impl VehicleDefs {
    /// Parse a `<VehicleData>` document. Unknown/garbage input yields an empty
    /// table.
    pub fn parse(xml: &[u8]) -> VehicleDefs {
        let xml = xml.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(xml);
        let mut reader = Reader::from_reader(xml);
        let mut buf = Vec::new();
        let mut out = VehicleDefs::default();
        let mut cur: Option<VehicleDef> = None;
        let mut in_display = false;
        let mut in_description = false;

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
                        "vehicledef" => {
                            let mut d = VehicleDef::default();
                            d.id = attr(&a, "id").unwrap_or("").to_string();
                            d.entity_class = attr(&a, "entityclass").unwrap_or("").to_string();
                            if d.entity_class.is_empty() {
                                d.entity_class = d.id.clone();
                            }
                            d.seat_count = u8_of(attr(&a, "seatcount"));
                            d.inventory_slot_count =
                                u8_of(attr(&a, "inventoryslotcount"));
                            d.weight = u8_of(attr(&a, "weight"));
                            d.acceleration_rating = f32_of(attr(&a, "accelerationrating"));
                            d.durability_rating = f32_of(attr(&a, "durabilityrating"));
                            d.handling_rating = f32_of(attr(&a, "handlingrating"));
                            d.top_speed_rating = f32_of(attr(&a, "topspeedrating"));
                            d.map_icon = attr(&a, "mapicon").unwrap_or("").to_string();
                            d.icon_id = u16_of(attr(&a, "iconid"));
                            cur = Some(d);
                        }
                        "displayname" => {
                            if let Some(d) = cur.as_mut() {
                                if let Some(txt) = attr(&a, "text") {
                                    d.display_name = txt.to_string();
                                }
                            }
                            in_display =
                                !is_empty && cur.is_some() && attr(&a, "text").is_none();
                        }
                        "description" => {
                            if let Some(d) = cur.as_mut() {
                                if let Some(txt) = attr(&a, "text") {
                                    d.description = txt.to_string();
                                }
                            }
                            in_description =
                                !is_empty && cur.is_some() && attr(&a, "text").is_none();
                        }
                        _ => {}
                    }
                }
                Event::Text(t) => {
                    if let Some(d) = cur.as_mut() {
                        if let Ok(txt) = t.unescape() {
                            if in_display {
                                d.display_name = txt.into_owned();
                            } else if in_description {
                                d.description = txt.into_owned();
                            }
                        }
                    }
                }
                Event::End(e) => match local_name_raw(e.name().as_ref()).as_str() {
                    "vehicledef" => {
                        if let Some(d) = cur.take() {
                            if !d.id.is_empty() {
                                out.by_id.insert(d.id.to_ascii_lowercase(), d);
                            }
                        }
                    }
                    "displayname" => in_display = false,
                    "description" => in_description = false,
                    _ => {}
                },
                _ => {}
            }
        }
        out
    }

    /// Look up by id (case-insensitive).
    pub fn get(&self, id: &str) -> Option<&VehicleDef> {
        self.by_id.get(&id.to_ascii_lowercase())
    }

    /// Number of loaded definitions.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// True when no definitions were loaded.
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Iterate every definition in id order.
    pub fn iter(&self) -> impl Iterator<Item = &VehicleDef> {
        self.by_id.values()
    }

    /// Load all `<VehicleDef>` rows from the game VFS.
    pub fn load(vfs: &crate::crypak::Vfs) -> VehicleDefs {
        for path in VEHICLE_DEFS_PATHS {
            if let Some(bytes) = vfs.read(path) {
                let defs = VehicleDefs::parse(&bytes);
                if !defs.is_empty() {
                    return defs;
                }
            }
        }
        // Fall back to a scan (archives may keep the gamedata/ prefix only).
        for p in vfs.paths_with_ext(".xml") {
            let norm = p.replace('\\', "/").to_ascii_lowercase();
            if norm.contains("contentmanager/vehicles.xml") {
                if let Some(bytes) = vfs.read(&p) {
                    let defs = VehicleDefs::parse(&bytes);
                    if !defs.is_empty() {
                        return defs;
                    }
                }
            }
        }
        VehicleDefs::default()
    }
}

// ---------------------------------------------------------------------------
// Compiled spawn list
// ---------------------------------------------------------------------------

/// `ULVS` magic (`0x0140b9a0`).
pub const SPAWN_MAGIC: u32 = 0x5356_4C55;
/// Expected version word (`0x0200_0000`).
pub const SPAWN_VERSION: u32 = 0x0200_0000;
/// Largest accepted spawn count / name buffer (`0xffff` guard in `0x0140b9a0`).
pub const SPAWN_MAX_COUNT: usize = 0xffff;
/// Runtime record stride (`0x20` in the activation path `0x0140b180`).
pub const SPAWN_RECORD_STRIDE: usize = 0x20;

/// A single placed vehicle spawn point.
#[derive(Debug, Clone, PartialEq)]
pub struct VehicleSpawn {
    /// `sVehicleClass` / `EntityClass` of the car to place.
    pub class_name: String,
    /// World position (metres, Z-up).
    pub pos: [f32; 3],
    /// World euler angles in radians (x = pitch, y = roll, z = yaw), matching
    /// the stored rotation triple.
    pub euler: [f32; 3],
}

impl VehicleSpawn {
    /// Heading about the world up axis (`euler[2]`).
    pub fn yaw(&self) -> f32 {
        self.euler[2]
    }
}

/// Parsed `vehicle_spawns_<platform>.markup`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VehicleSpawnFile {
    /// Spawns flagged `bInitialSpawn` (the first `initial_spawn_count`).
    pub initial_spawn_count: u32,
    /// Total spawn points in the file.
    pub total_spawn_count: u32,
    /// Every spawn point, in file order.
    pub spawns: Vec<VehicleSpawn>,
}

impl VehicleSpawnFile {
    /// Parse a `ULVS` buffer. Returns `None` on any header/shape violation,
    /// matching the runtime reader's validation checks.
    pub fn parse(bytes: &[u8]) -> Option<VehicleSpawnFile> {
        let b = bytes;
        if b.len() < 24 {
            return None;
        }
        let magic = rd_u32(b, 0)?;
        let version = rd_u32(b, 4)?;
        let endian = rd_u32(b, 8)?;
        let initial = rd_u32(b, 12)?;
        let total = rd_u32(b, 16)?;
        let name_size = rd_u32(b, 20)?;

        if magic != SPAWN_MAGIC || version != SPAWN_VERSION || endian != 0 {
            return None;
        }
        let total_us = total as usize;
        let initial_us = initial as usize;
        let name_us = name_size as usize;
        if total_us > SPAWN_MAX_COUNT
            || initial_us > SPAWN_MAX_COUNT
            || name_us > SPAWN_MAX_COUNT
            || total_us < initial_us
        {
            return None;
        }

        // positions -> rotation -> class offsets -> name buffer, each block
        // 0x80-aligned exactly as the reader advances `local_4`.
        let mut off = align_up(24, 0x80);
        let positions = read_floats(b, off, total_us * 3)?;
        off = align_up(off + total_us * 12, 0x80);
        let rotations = read_floats(b, off, total_us * 3)?;
        off = align_up(off + total_us * 12, 0x80);
        if off + total_us * 2 > b.len() {
            return None;
        }
        let mut offsets = Vec::with_capacity(total_us);
        for i in 0..total_us {
            offsets.push(rd_u16(b, off + i * 2)? as usize);
        }
        off = align_up(off + total_us * 2, 0x80);
        if off + name_us != b.len() {
            // The reader requires the file to end exactly on the name buffer.
            return None;
        }
        let names = &b[off..off + name_us];

        let mut spawns = Vec::with_capacity(total_us);
        for i in 0..total_us {
            let class_name = read_cstr(names, offsets[i]).unwrap_or_default();
            spawns.push(VehicleSpawn {
                class_name,
                pos: [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]],
                euler: [rotations[i * 3], rotations[i * 3 + 1], rotations[i * 3 + 2]],
            });
        }

        Some(VehicleSpawnFile {
            initial_spawn_count: initial,
            total_spawn_count: total,
            spawns,
        })
    }

    /// Spawns marked as initial (present when the level loads).
    pub fn initial(&self) -> &[VehicleSpawn] {
        let n = (self.initial_spawn_count as usize).min(self.spawns.len());
        &self.spawns[..n]
    }

    /// Spawns the RTS layer may repopulate later (`total - initial`).
    pub fn deferred(&self) -> &[VehicleSpawn] {
        let n = (self.initial_spawn_count as usize).min(self.spawns.len());
        &self.spawns[n..]
    }
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn rd_u16(b: &[u8], o: usize) -> Option<u16> {
    b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]))
}

fn read_floats(b: &[u8], o: usize, n: usize) -> Option<Vec<f32>> {
    let end = o.checked_add(n.checked_mul(4)?)?;
    let slice = b.get(o..end)?;
    Some(
        slice
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn read_cstr(buf: &[u8], off: usize) -> Option<String> {
    let tail = buf.get(off..)?;
    let end = tail.iter().position(|&c| c == 0)?;
    Some(String::from_utf8_lossy(&tail[..end]).into_owned())
}

// ---------------------------------------------------------------------------
// Small XML helpers
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

fn u16_of(s: Option<&str>) -> u16 {
    s.and_then(|s| s.trim().parse::<u16>().ok()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFS: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<VehicleData>
  <VehicleDef AccelerationRating="0.8" DurabilityRating="0.6" EntityClass="Hatchback_modern"
      HandlingRating="0.8" Id="Hatchback_modern" InventorySlotCount="4" MapIcon="VEHICLE"
      SeatCount="4" TopSpeedRating="0.4" Weight="1" IconId="109">
    <Description Flags="" Notes="" Text="Fragile but agile." />
    <DisplayName Flags="" Notes="" Text="Hatchback" />
  </VehicleDef>
  <VehicleDef AccelerationRating="0.6" DurabilityRating="0.8" EntityClass="PickUpNew"
      HandlingRating="0.6" Id="PickUpNew" InventorySlotCount="6" MapIcon="VEHICLE"
      SeatCount="6" TopSpeedRating="0.6" Weight="2" IconId="109">
    <Description Text="Rugged farm truck." />
    <DisplayName Text="Modern Pickup" />
  </VehicleDef>
</VehicleData>"#;

    #[test]
    fn parses_vehicle_defs() {
        let defs = VehicleDefs::parse(DEFS.as_bytes());
        assert_eq!(defs.len(), 2);
        let h = defs.get("hatchback_modern").expect("hatchback");
        assert_eq!(h.entity_class, "Hatchback_modern");
        assert_eq!(h.seat_count, 4);
        assert_eq!(h.item_slot_count(), 4);
        assert_eq!(h.weight, 1);
        assert_eq!(h.icon_id, 109);
        assert_eq!(h.display_name, "Hatchback");
        assert!((h.top_speed_rating - 0.4).abs() < 1e-6);
        let p = defs.get("PickUpNew").expect("pickup");
        assert_eq!(p.seat_count, 6);
        assert_eq!(p.display_name, "Modern Pickup");
    }

    /// Build a tiny but structurally valid `ULVS` buffer.
    fn build_spawn_file(initial: u32, total: u32, entries: &[(&str, [f32; 3], [f32; 3])]) -> Vec<u8> {
        let mut names = Vec::new();
        let mut offsets = Vec::new();
        for (name, _, _) in entries {
            offsets.push(names.len() as u16);
            names.extend_from_slice(name.as_bytes());
            names.push(0);
        }
        let mut out = Vec::new();
        out.extend_from_slice(&SPAWN_MAGIC.to_le_bytes());
        out.extend_from_slice(&SPAWN_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&initial.to_le_bytes());
        out.extend_from_slice(&total.to_le_bytes());
        out.extend_from_slice(&(names.len() as u32).to_le_bytes());
        while out.len() % 0x80 != 0 {
            out.push(0);
        }
        for (_, p, _) in entries {
            for v in p {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        while out.len() % 0x80 != 0 {
            out.push(0);
        }
        for (_, _, e) in entries {
            for v in e {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        while out.len() % 0x80 != 0 {
            out.push(0);
        }
        for o in &offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        while out.len() % 0x80 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&names);
        out
    }

    #[test]
    fn parses_spawn_file() {
        let buf = build_spawn_file(
            1,
            2,
            &[
                ("Hatchback_modern", [2462.85, 1769.24, 125.5], [0.0, 0.0, -0.576]),
                ("police_car", [2646.9, 2060.5, 134.0], [0.0, 0.0, 2.356]),
            ],
        );
        let f = VehicleSpawnFile::parse(&buf).expect("parses");
        assert_eq!(f.total_spawn_count, 2);
        assert_eq!(f.initial_spawn_count, 1);
        assert_eq!(f.spawns.len(), 2);
        assert_eq!(f.spawns[0].class_name, "Hatchback_modern");
        assert!((f.spawns[0].pos[0] - 2462.85).abs() < 1e-3);
        assert!((f.spawns[1].yaw() - 2.356).abs() < 1e-6);
        assert_eq!(f.initial().len(), 1);
        assert_eq!(f.deferred().len(), 1);
    }

    #[test]
    fn rejects_bad_spawn_files() {
        assert!(VehicleSpawnFile::parse(&[0u8; 4]).is_none());

        let good = build_spawn_file(1, 1, &[("a", [0.0; 3], [0.0; 3])]);
        // Corrupt magic.
        let mut bad = good.clone();
        bad[0] ^= 0xff;
        assert!(VehicleSpawnFile::parse(&bad).is_none());
        // Truncated name buffer.
        assert!(VehicleSpawnFile::parse(&good[..good.len() - 1]).is_none());
        // initial > total.
        let mut over = Vec::new();
        over.extend_from_slice(&SPAWN_MAGIC.to_le_bytes());
        over.extend_from_slice(&SPAWN_VERSION.to_le_bytes());
        over.extend_from_slice(&0u32.to_le_bytes());
        over.extend_from_slice(&5u32.to_le_bytes());
        over.extend_from_slice(&2u32.to_le_bytes());
        over.extend_from_slice(&0u32.to_le_bytes());
        assert!(VehicleSpawnFile::parse(&over).is_none());
    }
}
