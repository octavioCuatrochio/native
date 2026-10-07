//! CryEngine 3 `lights_*.lights` level light files.
//!
//! A level's baked static lights are stored column-wise, not as one record per
//! light. The file starts with a 7-word little-endian header, padded to
//! `0x80` bytes, followed by eleven 0x80-aligned sub-tables. The header and
//! sub-table names were recovered from the loader `FUN_015ba0f0` at
//! `0x15ba0f0`; the runtime consumers are `LSUpdate` (`0x15ba8a0`),
//! `FUN_015bb8e0` (build one light, `0x15bb8e0`), `FUN_015bb580` (bounce-light
//! colour/time blend, `0x15bb580`) and `FUN_015baf20` (cubemap texture
//! resolution, `0x15baf20`).
//!
//! Header (7 x u32, `0x15ba0f0`):
//!
//! | off  | field        | Windows value | loader check                         |
//! |------|--------------|---------------|--------------------------------------|
//! | 0x00 | magic        | `ULLI`        | must equal `0x494c4c55`               |
//! | 0x04 | version      | `0x07000000`  | must equal `0x07000000` (LE)          |
//! | 0x08 | endian mark  | `0`           | must equal `0`                        |
//! | 0x0c | light count  | 2356          | `<= 0xffff`                           |
//! | 0x10 | name bytes   | 1220          | `<= 0xffff` (`s_lights.fileNames`)    |
//! | 0x14 | info count   | 52            | `<= 0xff`   (`s_lights.info`)         |
//! | 0x18 | bounce count | 91            | (`s_lights.bounceInfo`)               |
//!
//! Sub-tables, in file order (each element size in parentheses):
//!
//! 1. `positions`  (12)  - world position, 3 x f32, `0x15bb8e0` copies to
//!    `+0x100` and `+0x10c`.
//! 2. `direction`  (12)  - 3 x f32 **Euler angles in radians**; `FUN_005e4a30`
//!    (`0x5e4a30`) runs `fsincos` on each and builds a 3x3 rotation matrix.
//!    `FUN_015bb580` uses component 2 (yaw) as a sun-angle argument.
//! 3. `radius`     (4)   - f32 light radius, stored at light `+0x11c`.
//! 4. `textureIdx` (2)   - u16 **byte offset** into `fileNames`; `0xffff` means
//!    "no texture". `FUN_015baf20` validates `offset <= name_bytes` and resolves
//!    the path (projector texture or deferred cubemap).
//! 5. `geomIdx`    (2)   - u16 byte offset into `fileNames`; when flags has
//!    `0x40` it is the geometry/model name.
//! 6. `infoIdx`    (1)   - u8 index into `info` (when flags lacks `0x10`) or
//!    into `bounceInfo` (when flags has `0x10`); `LSUpdate` reads both with
//!    this byte.
//! 7. `flags`      (1)   - u8 bit field, see [`FLAG_*`].
//! 8. `clipVolGuid`(8)   - 2 x u32 entity GUID halves; `FUN_015ba000`
//!    (`0x15ba000`) resolves it and tags the light `0x4000` if the entity is a
//!    `LightBox`.
//! 9. `fileNames`  (1)   - concatenated NUL-terminated texture/geometry paths,
//!    exactly `name_bytes` long.
//! 10. `info`      (28)  - 52 x 7 x f32 light styles (colour/intensity).
//! 11. `bounceInfo`(68)  - 91 x 17 x f32 day/night bounce styles; the first
//!    seven floats share the `info` layout.
//!
//! The Windows and Xbox One variants are byte-identical; the Xbox 360 variant
//! is the same layout with big-endian fields (and an endian marker of `1`) and
//! is rejected here, matching the loader's little-endian branch.

/// Magic token `ULLI` as read on a little-endian host (`0x494c4c55`).
pub const LIGHTS_MAGIC: u32 = 0x494c_4c55;
/// Version/format word expected by the Windows loader.
pub const LIGHTS_VERSION: u32 = 0x0700_0000;
/// The header is seven u32 words; the data area starts at the next 0x80 byte.
const HEADER_SIZE: usize = 0x1c;
/// Every sub-table begins on a 0x80-byte boundary.
const DATA_ALIGN: usize = 0x80;

/// `flags` bit: light has a projector texture (`textureIdx` -> `fileNames`).
pub const FLAG_PROJECTOR: u8 = 0x01;
/// `flags` bit: deferred cubemap light (`textureIdx` is a cubemap path).
pub const FLAG_CUBEMAP: u8 = 0x02;
/// `flags` bit: selects the alternate light type word (`0x10810000`).
pub const FLAG_TYPE_04: u8 = 0x04;
/// `flags` bit: set on projector/cubemap lights; consumer not located.
pub const FLAG_UNKNOWN_08: u8 = 0x08;
/// `flags` bit: bounce/time-window light; `infoIdx` indexes `bounceInfo`.
pub const FLAG_BOUNCE: u8 = 0x10;
/// `flags` bit: runtime-only "materialised" state; never set in files.
pub const FLAG_ACTIVE: u8 = 0x20;
/// `flags` bit: `geomIdx` is a geometry/model name.
pub const FLAG_GEOMETRY: u8 = 0x40;

/// One light: the columns of a single index across tables 1-8.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelLight {
    /// World position in metres.
    pub position: [f32; 3],
    /// Orientation as Euler angles `(x, y, z)` in radians; `z` is yaw.
    pub direction: [f32; 3],
    /// Light radius in metres.
    pub radius: f32,
    /// Byte offset into [`LevelLights::file_names`] for the texture, or
    /// `0xffff` for none.
    pub texture_offset: u16,
    /// Byte offset into [`LevelLights::file_names`] for the geometry, or
    /// `0xffff` for none.
    pub geom_offset: u16,
    /// Index into `info`, or into `bounce_info` when [`FLAG_BOUNCE`] is set.
    pub info_index: u8,
    /// Light flags, see [`FLAG_PROJECTOR`] etc.
    pub flags: u8,
    /// Clip-volume entity GUID, low then high word.
    pub clip_volume: [u32; 2],
}

impl LevelLight {
    /// `true` when this light uses the extended day/night bounce style.
    pub fn is_bounce(&self) -> bool {
        self.flags & FLAG_BOUNCE != 0
    }
    /// `true` when the light has a projector texture.
    pub fn has_projector(&self) -> bool {
        self.flags & FLAG_PROJECTOR != 0
    }
    /// `true` when the light is a deferred cubemap light.
    pub fn has_cubemap(&self) -> bool {
        self.flags & FLAG_CUBEMAP != 0
    }
    /// `true` when `geomIdx` names a geometry/model.
    pub fn has_geometry(&self) -> bool {
        self.flags & FLAG_GEOMETRY != 0
    }
}

/// A compact light style: 7 floats (`s_lights.info`).
///
/// `range_scale` is multiplied by `0.5` and written to the light (`0x15bb8e0`).
/// `LSUpdate` (`0x15ba8a0`) feeds `color` plus `param_b`/`param_c` to the
/// light-colour setter. `param_a` (the fifth float) is not read on this path.
#[derive(Debug, Clone, PartialEq)]
pub struct LightInfo {
    /// Float 0: intensity/range scale (`* 0.5`).
    pub range_scale: f32,
    /// Floats 1-3: RGB colour.
    pub color: [f32; 3],
    /// Float 4: unused by the located consumers.
    pub param_a: f32,
    /// Float 5: colour blend term.
    pub param_b: f32,
    /// Float 6: specular/emissive term.
    pub param_c: f32,
}

impl LightInfo {
    fn from_slice(v: &[f32]) -> Self {
        LightInfo {
            range_scale: v[0],
            color: [v[1], v[2], v[3]],
            param_a: v[4],
            param_b: v[5],
            param_c: v[6],
        }
    }
}

/// An extended day/night bounce style: 17 floats (`s_lights.bounceInfo`).
///
/// The first seven floats share [`LightInfo`]'s layout. `FUN_015bb580`
/// (`0x15bb580`) blends `color` and `bounce_color` by a sun-angle factor and a
/// smoothstep over the `[time_start, time_mid, time_end]` window (hours), using
/// `base_intensity`, `base_ratio`, `bounce_intensity`, `specular` and
/// `param_3c`. The remaining fields (`param_34`, `param_40`) were not found in
/// the located consumers.
#[derive(Debug, Clone, PartialEq)]
pub struct BounceLightInfo {
    /// Float 0: intensity/range scale (`* 0.5`).
    pub range_scale: f32,
    /// Floats 1-3: base RGB colour.
    pub color: [f32; 3],
    /// Float 4: base intensity multiplier.
    pub base_intensity: f32,
    /// Float 5: base intensity ratio.
    pub base_ratio: f32,
    /// Float 6: specular/emissive term.
    pub specular: f32,
    /// Floats 7-9: bounce RGB colour.
    pub bounce_color: [f32; 3],
    /// Float 10: window start (hours).
    pub time_start: f32,
    /// Float 11: window midpoint (hours).
    pub time_mid: f32,
    /// Float 12: window end (hours).
    pub time_end: f32,
    /// Float 13: unused by the located consumers.
    pub param_34: f32,
    /// Float 14: bounce intensity multiplier.
    pub bounce_intensity: f32,
    /// Float 15: secondary blend term.
    pub param_3c: f32,
    /// Float 16: unused by the located consumers.
    pub param_40: f32,
}

impl BounceLightInfo {
    fn from_slice(v: &[f32]) -> Self {
        BounceLightInfo {
            range_scale: v[0],
            color: [v[1], v[2], v[3]],
            base_intensity: v[4],
            base_ratio: v[5],
            specular: v[6],
            bounce_color: [v[7], v[8], v[9]],
            time_start: v[10],
            time_mid: v[11],
            time_end: v[12],
            param_34: v[13],
            bounce_intensity: v[14],
            param_3c: v[15],
            param_40: v[16],
        }
    }
}

/// The style a light selects through its `infoIdx` byte.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightStyle<'a> {
    /// Compact style from the 52-entry `info` table.
    Info(&'a LightInfo),
    /// Extended style from the 91-entry `bounceInfo` table.
    Bounce(&'a BounceLightInfo),
}

/// A fully decoded `lights_*.lights` file.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelLights {
    /// Header version word (always [`LIGHTS_VERSION`] on success).
    pub version: u32,
    /// Header endian marker (always `0` on success).
    pub endian: u32,
    /// Per-light columns, one entry per `count`.
    pub lights: Vec<LevelLight>,
    /// Concatenated NUL-terminated texture/geometry paths.
    pub file_names: Vec<u8>,
    /// Compact light styles (52 in the shipped level).
    pub infos: Vec<LightInfo>,
    /// Extended bounce styles (91 in the shipped level).
    pub bounce_infos: Vec<BounceLightInfo>,
}

impl LevelLights {
    /// Resolve a byte offset into `file_names` as a NUL-terminated path.
    ///
    /// Returns `None` for the `0xffff` sentinel, out-of-range offsets and
    /// non-UTF-8 data.
    pub fn name_at(&self, offset: u16) -> Option<&str> {
        if offset == 0xffff {
            return None;
        }
        let start = offset as usize;
        if start >= self.file_names.len() {
            return None;
        }
        let rest = &self.file_names[start..];
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        std::str::from_utf8(&rest[..end]).ok()
    }

    /// Projector/cubemap texture path for a light, if any.
    pub fn texture_path(&self, light: &LevelLight) -> Option<&str> {
        self.name_at(light.texture_offset)
    }

    /// Geometry/model path for a light, if any.
    pub fn geometry_path(&self, light: &LevelLight) -> Option<&str> {
        self.name_at(light.geom_offset)
    }

    /// The style selected by `light.info_index`; `None` when out of range.
    pub fn style(&self, light: &LevelLight) -> Option<LightStyle<'_>> {
        let i = light.info_index as usize;
        if light.is_bounce() {
            self.bounce_infos.get(i).map(LightStyle::Bounce)
        } else {
            self.infos.get(i).map(LightStyle::Info)
        }
    }
}

/// Why a `.lights` buffer could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LightsError {
    /// Fewer than seven header words.
    TooShort,
    /// First word is not `ULLI`.
    BadMagic,
    /// Version word is not `0x07000000`.
    BadVersion,
    /// Endian marker is not zero (e.g. the big-endian Xbox 360 variant).
    BadEndian,
    /// A header count exceeds the loader's limit.
    CountTooLarge,
    /// A table runs past the end of the buffer.
    Truncated(&'static str),
    /// Bytes remain after the last table.
    TrailingData,
}

/// Parse a `lights_*.lights` buffer. See the module docs for the layout.
pub fn parse(data: &[u8]) -> Result<LevelLights, LightsError> {
    if data.len() < HEADER_SIZE {
        return Err(LightsError::TooShort);
    }
    let rd = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    if rd(0) != LIGHTS_MAGIC {
        return Err(LightsError::BadMagic);
    }
    let version = rd(4);
    if version != LIGHTS_VERSION {
        return Err(LightsError::BadVersion);
    }
    let endian = rd(8);
    if endian != 0 {
        return Err(LightsError::BadEndian);
    }
    let count = rd(0xc) as usize;
    let name_bytes = rd(0x10) as usize;
    let info_count = rd(0x14) as usize;
    let bounce_count = rd(0x18) as usize;
    if count > 0xffff || name_bytes > 0xffff || info_count > 0xff {
        return Err(LightsError::CountTooLarge);
    }

    let mut cur = DATA_ALIGN;
    let positions = take(data, &mut cur, count * 12, "positions")?;
    let direction = take(data, &mut cur, count * 12, "direction")?;
    let radius = take(data, &mut cur, count * 4, "radius")?;
    let texture = take(data, &mut cur, count * 2, "textureIdx")?;
    let geom = take(data, &mut cur, count * 2, "geomIdx")?;
    let info_idx = take(data, &mut cur, count, "infoIdx")?;
    let flags = take(data, &mut cur, count, "flags")?;
    let clip = take(data, &mut cur, count * 8, "clipVolGuid")?;
    let file_names = take(data, &mut cur, name_bytes, "fileNames")?.to_vec();
    let info_raw = take(data, &mut cur, info_count * 28, "info")?;
    let bounce_raw = take(data, &mut cur, bounce_count * 68, "bounceInfo")?;
    if cur != data.len() {
        return Err(LightsError::TrailingData);
    }

    let mut lights = Vec::with_capacity(count);
    for i in 0..count {
        lights.push(LevelLight {
            position: vec3(&positions[i * 12..]),
            direction: vec3(&direction[i * 12..]),
            radius: f32_at(&radius[i * 4..]),
            texture_offset: u16_at(&texture[i * 2..]),
            geom_offset: u16_at(&geom[i * 2..]),
            info_index: info_idx[i],
            flags: flags[i],
            clip_volume: [u32_at(&clip[i * 8..]), u32_at(&clip[i * 8 + 4..])],
        });
    }

    let infos = (0..info_count)
        .map(|i| LightInfo::from_slice(&read_floats(&info_raw[i * 28..], 7)))
        .collect();
    let bounce_infos = (0..bounce_count)
        .map(|i| BounceLightInfo::from_slice(&read_floats(&bounce_raw[i * 68..], 17)))
        .collect();

    Ok(LevelLights {
        version,
        endian,
        lights,
        file_names,
        infos,
        bounce_infos,
    })
}

/// Borrow `len` bytes at `*cur`, then advance to the next 0x80 boundary.
fn take<'a>(
    data: &'a [u8],
    cur: &mut usize,
    len: usize,
    what: &'static str,
) -> Result<&'a [u8], LightsError> {
    let end = cur.checked_add(len).ok_or(LightsError::CountTooLarge)?;
    if end > data.len() {
        return Err(LightsError::Truncated(what));
    }
    let out = &data[*cur..end];
    *cur = align_up(end);
    Ok(out)
}

fn align_up(x: usize) -> usize {
    (x + DATA_ALIGN - 1) & !(DATA_ALIGN - 1)
}

fn vec3(b: &[u8]) -> [f32; 3] {
    [f32_at(b), f32_at(&b[4..]), f32_at(&b[8..])]
}

fn f32_at(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn u32_at(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn u16_at(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn read_floats(b: &[u8], n: usize) -> Vec<f32> {
    (0..n).map(|i| f32_at(&b[i * 4..])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a synthetic file from the same column order the loader uses.
    #[allow(clippy::type_complexity)]
    fn build(
        lights: &[([f32; 3], [f32; 3], f32, u16, u16, u8, u8, [u32; 2])],
        names: &[&str],
        infos: &[[f32; 7]],
        bounce: &[[f32; 17]],
    ) -> Vec<u8> {
        let name_bytes: usize = names.iter().map(|s| s.len() + 1).sum();
        let mut out = Vec::new();
        out.extend_from_slice(b"ULLI");
        out.extend_from_slice(&LIGHTS_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(lights.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name_bytes as u32).to_le_bytes());
        out.extend_from_slice(&(infos.len() as u32).to_le_bytes());
        out.extend_from_slice(&(bounce.len() as u32).to_le_bytes());
        while out.len() % DATA_ALIGN != 0 {
            out.push(0);
        }

        let mut table = |bytes: &[u8]| {
            while out.len() % DATA_ALIGN != 0 {
                out.push(0);
            }
            out.extend_from_slice(bytes);
        };

        let mut col = Vec::new();
        for l in lights {
            for f in l.0 {
                col.extend_from_slice(&f.to_le_bytes());
            }
        }
        table(&col);

        let mut col = Vec::new();
        for l in lights {
            for f in l.1 {
                col.extend_from_slice(&f.to_le_bytes());
            }
        }
        table(&col);

        let mut col = Vec::new();
        for l in lights {
            col.extend_from_slice(&l.2.to_le_bytes());
        }
        table(&col);

        let mut col = Vec::new();
        for l in lights {
            col.extend_from_slice(&l.3.to_le_bytes());
        }
        table(&col);

        let mut col = Vec::new();
        for l in lights {
            col.extend_from_slice(&l.4.to_le_bytes());
        }
        table(&col);

        let col: Vec<u8> = lights.iter().map(|l| l.5).collect();
        table(&col);

        let col: Vec<u8> = lights.iter().map(|l| l.6).collect();
        table(&col);

        let mut col = Vec::new();
        for l in lights {
            col.extend_from_slice(&l.7[0].to_le_bytes());
            col.extend_from_slice(&l.7[1].to_le_bytes());
        }
        table(&col);

        let mut col = Vec::new();
        for s in names {
            col.extend_from_slice(s.as_bytes());
            col.push(0);
        }
        table(&col);

        let mut col = Vec::new();
        for v in infos {
            for f in v {
                col.extend_from_slice(&f.to_le_bytes());
            }
        }
        table(&col);

        let mut col = Vec::new();
        for v in bounce {
            for f in v {
                col.extend_from_slice(&f.to_le_bytes());
            }
        }
        table(&col);

        while out.len() % DATA_ALIGN != 0 {
            out.push(0);
        }
        out
    }

    fn sample_infos() -> Vec<[f32; 7]> {
        vec![[90.0, 1.0, 0.5, 0.25, 1.0, 2.0, 0.0]]
    }

    fn sample_bounce() -> Vec<[f32; 17]> {
        let mut v = [0.0f32; 17];
        v[0] = 80.0;
        v[1] = 0.1;
        v[2] = 0.2;
        v[3] = 0.3;
        v[4] = 1.0;
        v[7] = 0.9;
        v[8] = 0.8;
        v[9] = 0.7;
        v[10] = 6.0;
        v[11] = 13.0;
        v[12] = 16.0;
        v[14] = 3.0;
        vec![v]
    }

    #[test]
    fn parses_synthetic_file() {
        let names = ["textures/lights/a.tif", "objects/x.cgf"];
        let lights = [
            (
                [1.0, 2.0, 3.0],
                [0.0, 0.0, -0.785],
                4.0,
                0,
                22,
                0,
                FLAG_PROJECTOR,
                [7, 9],
            ),
            (
                [4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0],
                8.0,
                0xffff,
                0xffff,
                0,
                FLAG_BOUNCE,
                [0, 0],
            ),
        ];
        let data = build(&lights, &names, &sample_infos(), &sample_bounce());
        let f = parse(&data).unwrap();
        assert_eq!(f.lights.len(), 2);
        assert_eq!(f.infos.len(), 1);
        assert_eq!(f.bounce_infos.len(), 1);

        let a = &f.lights[0];
        assert_eq!(a.position, [1.0, 2.0, 3.0]);
        assert_eq!(a.direction, [0.0, 0.0, -0.785]);
        assert_eq!(a.radius, 4.0);
        assert_eq!(a.clip_volume, [7, 9]);
        assert!(a.has_projector());
        assert!(!a.is_bounce());
        assert_eq!(f.texture_path(a), Some("textures/lights/a.tif"));
        assert_eq!(f.geometry_path(a), Some("objects/x.cgf"));
        match f.style(a).unwrap() {
            LightStyle::Info(i) => {
                assert_eq!(i.range_scale, 90.0);
                assert_eq!(i.color, [1.0, 0.5, 0.25]);
                assert_eq!(i.param_b, 2.0);
            }
            _ => panic!("expected info style"),
        }

        let b = &f.lights[1];
        assert!(b.is_bounce());
        assert_eq!(f.texture_path(b), None);
        match f.style(b).unwrap() {
            LightStyle::Bounce(bi) => {
                assert_eq!(bi.bounce_color, [0.9, 0.8, 0.7]);
                assert_eq!(bi.time_start, 6.0);
                assert_eq!(bi.time_mid, 13.0);
                assert_eq!(bi.time_end, 16.0);
                assert_eq!(bi.bounce_intensity, 3.0);
            }
            _ => panic!("expected bounce style"),
        }
    }

    #[test]
    fn rejects_bad_headers_and_truncation() {
        assert_eq!(parse(b"short"), Err(LightsError::TooShort));
        assert_eq!(parse(&[0u8; 0x1c]), Err(LightsError::BadMagic));

        let good = build(
            &[([0.0; 3], [0.0; 3], 1.0, 0, 0, 0, 0, [0, 0])],
            &["a"],
            &sample_infos(),
            &sample_bounce(),
        );

        let mut bad = good.clone();
        bad[4..8].copy_from_slice(&0x0700_0001u32.to_le_bytes());
        assert_eq!(parse(&bad), Err(LightsError::BadVersion));

        let mut bad = good.clone();
        bad[8..0xc].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(parse(&bad), Err(LightsError::BadEndian));

        let mut bad = good.clone();
        bad.truncate(good.len() - 100);
        assert!(matches!(parse(&bad), Err(LightsError::Truncated(_))));

        let mut bad = good;
        bad.push(0);
        assert_eq!(parse(&bad), Err(LightsError::TrailingData));
    }

    #[test]
    fn name_offsets_resolve_and_sentinels_are_none() {
        let names = ["abc", "de"];
        let lights = [
            ([0.0; 3], [0.0; 3], 1.0, 0, 4, 0, 0, [0, 0]),
            ([0.0; 3], [0.0; 3], 1.0, 0xffff, 0xffff, 0, 0, [0, 0]),
            ([0.0; 3], [0.0; 3], 1.0, 999, 999, 0, 0, [0, 0]),
        ];
        let f = parse(&build(&lights, &names, &[], &[])).unwrap();
        assert_eq!(f.name_at(0), Some("abc"));
        assert_eq!(f.name_at(4), Some("de"));
        assert_eq!(f.name_at(0xffff), None);
        assert_eq!(f.name_at(999), None);
    }

    #[test]
    fn align_up_matches_loader() {
        assert_eq!(align_up(0x80), 0x80);
        assert_eq!(align_up(0x81), 0x100);
        assert_eq!(align_up(0x100), 0x100);
    }

    #[test]
    fn integration_real_level_if_present() {
        let path = std::env::var("SOD_LIGHTS").unwrap_or_else(|_| {
            "/home/battlestation/Desktop/sod-re/extracted-game/level/lights_win.lights".to_string()
        });
        let Ok(data) = std::fs::read(&path) else {
            eprintln!("skipping integration test: {path} not found");
            return;
        };
        let f = parse(&data).expect("real .lights decodes");

        // Shipped Class3 counts from the header.
        assert_eq!(f.lights.len(), 2356);
        assert_eq!(f.infos.len(), 52);
        assert_eq!(f.bounce_infos.len(), 91);
        assert_eq!(f.file_names.len(), 1220);

        // Every light decodes into the terrain box 0..4096 m with a sane radius
        // and finite orientation, and its info index is in range for the table
        // its flags select.
        for (i, l) in f.lights.iter().enumerate() {
            for c in l.position {
                assert!(
                    c.is_finite() && (0.0..=4096.0).contains(&c),
                    "light {i} pos {c}"
                );
            }
            for a in l.direction {
                assert!(a.is_finite(), "light {i} direction {a}");
            }
            assert!(
                l.radius.is_finite() && l.radius >= 0.0,
                "light {i} radius {}",
                l.radius
            );
            let n = if l.is_bounce() {
                f.bounce_infos.len()
            } else {
                f.infos.len()
            };
            assert!(
                (l.info_index as usize) < n,
                "light {i} infoIdx {} >= {n}",
                l.info_index
            );
            if l.has_projector() || l.has_cubemap() {
                assert!(
                    l.texture_offset as usize <= f.file_names.len(),
                    "light {i} texture offset {}",
                    l.texture_offset
                );
            }
        }

        // The name buffer is NUL-separated paths; spot-check a couple.
        assert!(f.file_names.contains(&b'\0'));
        assert!(f
            .file_names
            .windows(5)
            .any(|w| w == b".tif\0" || w == b".dds\0"));
    }
}
