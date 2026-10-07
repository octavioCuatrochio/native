//! `ZombieVariant.xml` — the ambient-zombie appearance table.
//!
//! The ambient Grunt is not shipped as one self-contained character. The engine
//! builds it at spawn from a weighted table (`Gender` -> `BodyType` -> per-slot
//! model lists), then assembles the chosen `Body`/`Legs`/`Skin`/`Head`/`Hair`/
//! `Hat` meshes against the gender's skeleton. Each model may carry
//! `VariantShader`/`VariantIndex` records that select cloth colour, tint and
//! pattern/decal overlays.
//!
//! # Where this comes from
//!
//! * `NSZombieManager::ZombieManager::LoadVariantData` (`0x0132cec0`) parses the
//!   table and normalises every model/gender/colour weight into a cumulative
//!   percentage array.  The file is `Libs/Class3/Zombies/ZombieVariant.xml`
//!   (string `0x018426d0`); the element/attribute names it reads are the ones
//!   parsed here (`BodyType`, `SkinVariant`, `EigenZombie`, `VariantShader`,
//!   `VariantIndex`, `ColorPrefs`, `HairTemplates`, `HairTemplate`, `weight`,
//!   `decalOdds`, `patternOdds`, `brightnessMin/Max`, `hueOffset`,
//!   `saturationMin`, `satSlope`, `satZero`, `maxDiffuse`, `minDiffuse`,
//!   `maxMelanin`).
//! * The spawn-time variant draw is in the logical-spawn routine `0x0133dc40`:
//!   for the default class (Grunt) it advances the shared zombie RNG and scans
//!   the normalised body-type cumulative array for the first entry whose
//!   percentage exceeds `roll`; all other classes use the fixed gender index of
//!   their freak group (`0x0131d110`).
//! * `FUN_01320270` composes the picked body type: one weighted model per
//!   present slot, then `FUN_0131fd00` applies the per-model shaders and
//!   `NSZombieManager::GetColor` (`0x01329980`) picks a colour preference.
//!
//! The two coupled LCGs used for every draw are recovered from the binary:
//! `a = (a & 0xffff) * 0x9069 + (a >> 16)` and
//! `b = (b & 0xffff) * 0x1_4650 + (b >> 16)`; a draw is
//! `((a << 16) + b) % 100`.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// Slots a body type may supply, in assembly order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    Body,
    Legs,
    Skin,
    Head,
    Hair,
    Hat,
}

impl Slot {
    /// Lower-case label used for logging and assembly part names.
    pub fn label(self) -> &'static str {
        match self {
            Slot::Body => "body",
            Slot::Legs => "legs",
            Slot::Skin => "skin",
            Slot::Head => "head",
            Slot::Hair => "hair",
            Slot::Hat => "hat",
        }
    }
}

/// The slot order the engine composes in (`FUN_01320270`): body first, then the
/// remaining attachments.
pub const SLOT_ORDER: [Slot; 6] =
    [Slot::Body, Slot::Legs, Slot::Skin, Slot::Head, Slot::Hair, Slot::Hat];

/// A `file` + `weight` pair (`Model`, `Pattern`, `Decal`, `HairTemplate`, ...).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WeightedPath {
    pub file: String,
    pub weight: f32,
}

/// A per-body-type skin tint range (`SkinVariant`).
#[derive(Clone, Debug, PartialEq)]
pub struct SkinVariant {
    pub max_diffuse: [f32; 3],
    pub min_diffuse: [f32; 3],
    pub max_melanin: f32,
    pub shininess: Option<f32>,
    pub blur_scale: Option<f32>,
    pub specular_strength: Option<f32>,
    pub self_shadow_softening: Option<f32>,
    pub zombify_decal_odds: f32,
}

impl Default for SkinVariant {
    fn default() -> Self {
        SkinVariant {
            max_diffuse: [0.0; 3],
            min_diffuse: [0.0; 3],
            max_melanin: 0.0,
            shininess: None,
            blur_scale: None,
            specular_strength: None,
            self_shadow_softening: None,
            zombify_decal_odds: 0.0,
        }
    }
}

/// One `VariantIndex` of a `VariantShader`: the colour/tint parameters applied
/// to a single sub-material (`index`) of the model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VariantIndex {
    /// Sub-material index inside the model this record targets.
    pub index: i32,
    pub brightness_min: f32,
    pub brightness_max: f32,
    pub sat_slope: f32,
    pub sat_zero: f32,
    pub saturation_min: f32,
    pub hue_offset: f32,
    pub decal_odds: f32,
    pub pattern_odds: f32,
    /// `ColorPrefs` name -> weight.
    pub color_prefs: Vec<(String, f32)>,
}

/// A `VariantShader` block (`type='shirt'` / `'pants'`) with its indices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VariantShader {
    pub kind: String,
    pub decal_odds: f32,
    pub pattern_odds: f32,
    pub indices: Vec<VariantIndex>,
}

/// A model choice of one slot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model {
    /// `file` attribute, without the appended `.chr` extension.
    pub file: String,
    pub weight: f32,
    /// Optional `EigenZombie` decal/material CDF.
    pub eigen: Option<String>,
    pub shaders: Vec<VariantShader>,
}

/// One `BodyType` (the unit the spawn draw picks).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BodyType {
    pub kind: String,
    pub weight: f32,
    pub body: Vec<Model>,
    pub legs: Vec<Model>,
    pub skin: Vec<Model>,
    pub head: Vec<Model>,
    pub hair: Vec<Model>,
    pub hat: Vec<Model>,
    pub skin_variant: Option<SkinVariant>,
}

impl BodyType {
    fn slot(&self, slot: Slot) -> &[Model] {
        match slot {
            Slot::Body => &self.body,
            Slot::Legs => &self.legs,
            Slot::Skin => &self.skin,
            Slot::Head => &self.head,
            Slot::Hair => &self.hair,
            Slot::Hat => &self.hat,
        }
    }

    fn slot_mut(&mut self, slot: Slot) -> &mut Vec<Model> {
        match slot {
            Slot::Body => &mut self.body,
            Slot::Legs => &mut self.legs,
            Slot::Skin => &mut self.skin,
            Slot::Head => &mut self.head,
            Slot::Hair => &mut self.hair,
            Slot::Hat => &mut self.hat,
        }
    }
}

/// A `Gender` entry: one skeleton and a weighted set of body types.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gender {
    pub kind: String,
    pub weight: f32,
    /// Skeleton path without the `.chr` extension.
    pub skeleton: String,
    pub body_types: Vec<BodyType>,
}

/// A named hue range (`Colors`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColorRange {
    pub kind: String,
    pub min: f32,
    pub max: f32,
    pub weight: f32,
}

/// A `MaterialTypes` entry (cloth model selection, parsed for completeness).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialType {
    pub kind: String,
    pub weight: f32,
    pub spec_color: [f32; 3],
}

/// The whole parsed `VariantData` document.
#[derive(Clone, Debug, Default)]
pub struct VariantData {
    pub genders: Vec<Gender>,
    pub material_types: Vec<MaterialType>,
    pub colors: Vec<ColorRange>,
    pub decals: Vec<WeightedPath>,
    pub patterns: Vec<WeightedPath>,
    pub pattern_tile: [f32; 2],
    pub hair_templates: Vec<WeightedPath>,
    pub zombify_decals: Vec<WeightedPath>,
}

/// One decoded tint produced for a `VariantIndex`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tint {
    pub material_index: i32,
    pub hue: f32,
    pub brightness: f32,
    pub saturation: f32,
}

/// One picked mesh of a spawned zombie.
#[derive(Clone, Debug, PartialEq)]
pub struct PickedPart {
    pub slot: Slot,
    /// `.chr` path (extension appended).
    pub file: String,
    pub eigen: Option<String>,
    pub tints: Vec<Tint>,
}

/// A fully picked zombie appearance: the skeleton plus one mesh per present
/// slot, and the per-model tint/pattern/decal draws.
#[derive(Clone, Debug, PartialEq)]
pub struct Appearance {
    pub gender: String,
    pub body_type: String,
    pub skeleton: String,
    pub parts: Vec<PickedPart>,
    pub pattern: Option<String>,
    pub pattern_tile: f32,
    pub decal: Option<String>,
}

impl Appearance {
    /// Stable hash of the geometry only (skeleton + part set). Two spawns with
    /// the same geometry share one assembled template even when their tints
    /// differ, because the native renderer draws the authored palettes.
    pub fn geometry_key(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |s: &str| {
            for b in s.as_bytes() {
                h ^= *b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h ^= 0xff;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        feed(&self.skeleton);
        for p in &self.parts {
            feed(p.slot.label());
            feed(&p.file);
        }
        h
    }

    /// `(slot label, .chr path)` pairs for the explicit assembler.
    pub fn chr_parts(&self) -> Vec<(String, String)> {
        self.parts
            .iter()
            .map(|p| (p.slot.label().to_string(), p.file.clone()))
            .collect()
    }
}

/// The two coupled 16-bit LCGs the zombie manager draws from.
///
/// Recovered from `0x0133dc40`, `0x01320270`, `0x0131fd00` and `0x01329980`.
#[derive(Clone, Copy, Debug)]
pub struct VariantRng {
    a: u32,
    b: u32,
}

impl VariantRng {
    /// Seed from a spawn handle. The engine seeds its globals from the manager
    /// (`FUN_01390420`); keying off the handle gives the same per-zombie
    /// stability without touching the runtime's own state.
    pub fn new(seed: u32) -> Self {
        let mut r = VariantRng {
            a: seed ^ 0x9e37_79b9,
            b: seed.rotate_left(13) ^ 0x85eb_ca6b,
        };
        r.advance();
        r
    }

    fn advance(&mut self) {
        self.a = (self.a & 0xffff).wrapping_mul(0x9069).wrapping_add(self.a >> 16);
        self.b = (self.b & 0xffff).wrapping_mul(18000).wrapping_add(self.b >> 16);
    }

    /// A draw in `0..100`, the engine's `((a << 16) + b) % 100`.
    pub fn roll100(&mut self) -> u32 {
        self.advance();
        self.a.wrapping_shl(16).wrapping_add(self.b) % 100
    }

    /// A draw in `[0, 1)`, `((a << 16) + b) * 2^-32`.
    pub fn unit(&mut self) -> f32 {
        self.advance();
        self.a.wrapping_shl(16).wrapping_add(self.b) as f32 * 2.328_306_4e-10
    }
}

/// First index whose cumulative percentage exceeds `roll`; the engine falls
/// back to the first entry when the normalised byte array underruns.
fn weighted_index(weights: &[f32], rng: &mut VariantRng) -> usize {
    let total: f32 = weights.iter().filter(|w| **w > 0.0).sum();
    if total <= 0.0 {
        return 0;
    }
    let roll = rng.roll100() as f32;
    let mut cum = 0.0f32;
    for (i, w) in weights.iter().enumerate() {
        if *w <= 0.0 {
            continue;
        }
        cum += *w * 100.0 / total;
        if roll < cum {
            return i;
        }
    }
    0
}

fn parse_f32(e: &BytesStart, name: &str, default: f32) -> f32 {
    get_attr(e, name).and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(default)
}

fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

fn parse_vec3_attr(e: &BytesStart, name: &str, default: [f32; 3]) -> [f32; 3] {
    let Some(s) = get_attr(e, name) else { return default };
    let v: Vec<f32> = s
        .replace(' ', "")
        .split(',')
        .filter_map(|p| p.parse::<f32>().ok())
        .collect();
    if v.len() >= 3 {
        [v[0], v[1], v[2]]
    } else {
        default
    }
}

fn slot_from_name(name: &[u8]) -> Option<Slot> {
    match name {
        b"Body" => Some(Slot::Body),
        b"Legs" => Some(Slot::Legs),
        b"Skin" => Some(Slot::Skin),
        b"Head" => Some(Slot::Head),
        b"Hair" => Some(Slot::Hair),
        b"Hat" => Some(Slot::Hat),
        _ => None,
    }
}

#[derive(Default)]
struct Cursor {
    gender: Option<usize>,
    body: Option<usize>,
    slot: Option<Slot>,
}

impl VariantData {
    /// Parse `ZombieVariant.xml`. Returns `None` when the document has no
    /// `<VariantData>` root; malformed content is skipped, never panics.
    pub fn parse(xml: &[u8]) -> Option<VariantData> {
        let mut reader = Reader::from_reader(xml);
        let mut buf = Vec::new();
        let mut data = VariantData { pattern_tile: [0.0, 0.0], ..Default::default() };
        let mut cur = Cursor::default();
        let mut saw_root = false;

        loop {
            let ev = match reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => break,
                Err(_) => break,
                Ok(ev) => ev,
            };
            match ev {
                Event::Start(e) | Event::Empty(e) => {
                    let name = e.local_name();
                    let name = name.as_ref().to_vec();
                    if name == b"VariantData" {
                        saw_root = true;
                    }
                    open(&mut data, &mut cur, &name, &e);
                }
                Event::End(e) => {
                    close(&mut cur, e.local_name().as_ref());
                }
                _ => {}
            }
            buf.clear();
        }

        if saw_root {
            Some(data)
        } else {
            None
        }
    }
}

fn close(cur: &mut Cursor, name: &[u8]) {
    match name {
        b"Gender" => {
            cur.gender = None;
            cur.body = None;
            cur.slot = None;
        }
        b"BodyType" => {
            cur.body = None;
            cur.slot = None;
        }
        b"Body" | b"Legs" | b"Skin" | b"Head" | b"Hair" | b"Hat" => cur.slot = None,
        _ => {}
    }
}

fn open(data: &mut VariantData, cur: &mut Cursor, name: &[u8], e: &BytesStart) {
    match name {
        b"Gender" => {
            data.genders.push(Gender {
                kind: get_attr(e, "type").unwrap_or_default(),
                weight: parse_f32(e, "weight", 1.0),
                skeleton: get_attr(e, "skeleton").unwrap_or_default(),
                body_types: Vec::new(),
            });
            cur.gender = Some(data.genders.len() - 1);
            cur.body = None;
        }
        b"BodyType" => {
            if let Some(g) = cur.gender {
                data.genders[g].body_types.push(BodyType {
                    kind: get_attr(e, "type").unwrap_or_default(),
                    weight: parse_f32(e, "weight", 1.0),
                    ..Default::default()
                });
                cur.body = Some(data.genders[g].body_types.len() - 1);
            }
        }
        b"Body" | b"Legs" | b"Skin" | b"Head" | b"Hair" | b"Hat" => {
            cur.slot = slot_from_name(name);
        }
        b"Model" => {
            if let Some(slot) = cur.slot {
                if let Some(b) = cur_body_mut(data, cur) {
                    b.slot_mut(slot).push(Model {
                        file: get_attr(e, "file").unwrap_or_default(),
                        weight: parse_f32(e, "weight", 1.0),
                        eigen: None,
                        shaders: Vec::new(),
                    });
                }
            }
        }
        b"EigenZombie" => {
            if let Some(m) = cur_model_mut(data, cur) {
                m.eigen = get_attr(e, "file");
            }
        }
        b"VariantShader" => {
            if let Some(m) = cur_model_mut(data, cur) {
                m.shaders.push(VariantShader {
                    kind: get_attr(e, "type").unwrap_or_default(),
                    decal_odds: parse_f32(e, "decalOdds", 0.0),
                    pattern_odds: parse_f32(e, "patternOdds", 0.0),
                    indices: Vec::new(),
                });
            }
        }
        b"VariantIndex" => {
            if let Some(s) = cur_shader_mut(data, cur) {
                s.indices.push(VariantIndex {
                    index: get_attr(e, "index").and_then(|v| v.parse().ok()).unwrap_or(0),
                    brightness_min: parse_f32(e, "brightnessMin", 0.0),
                    brightness_max: parse_f32(e, "brightnessMax", 0.0),
                    sat_slope: parse_f32(e, "satSlope", 0.0),
                    sat_zero: parse_f32(e, "satZero", 0.0),
                    saturation_min: parse_f32(e, "saturationMin", 0.0),
                    hue_offset: parse_f32(e, "hueOffset", 0.0),
                    decal_odds: parse_f32(e, "decalOdds", 0.0),
                    pattern_odds: parse_f32(e, "patternOdds", 0.0),
                    color_prefs: Vec::new(),
                });
            }
        }
        b"ColorPrefs" => {
            if let Some(vi) = cur_index_mut(data, cur) {
                for a in e.attributes().flatten() {
                    let Ok(val) = a.unescape_value() else { continue };
                    if let Ok(w) = val.trim().parse::<f32>() {
                        vi.color_prefs
                            .push((String::from_utf8_lossy(a.key.as_ref()).into_owned(), w));
                    }
                }
            }
        }
        b"SkinVariant" => {
            if let Some(b) = cur_body_mut(data, cur) {
                b.skin_variant = Some(SkinVariant {
                    max_diffuse: parse_vec3_attr(e, "maxDiffuse", [0.0; 3]),
                    min_diffuse: parse_vec3_attr(e, "minDiffuse", [0.0; 3]),
                    max_melanin: parse_f32(e, "maxMelanin", 0.0),
                    shininess: opt_f32(e, "Shininess"),
                    blur_scale: opt_f32(e, "BlurScale"),
                    specular_strength: opt_f32(e, "SpecularStrenght"),
                    self_shadow_softening: opt_f32(e, "SelfShadowSoftening"),
                    zombify_decal_odds: parse_f32(e, "zombifyDecalOdds", 0.0),
                });
            }
        }
        b"Material" => {
            data.material_types.push(MaterialType {
                kind: get_attr(e, "type").unwrap_or_default(),
                weight: parse_f32(e, "weight", 1.0),
                spec_color: parse_vec3_attr(e, "SpecColor1", [0.0; 3]),
            });
        }
        b"Red" | b"Orange" | b"Yellow" | b"Green" | b"Cyan" | b"Blue" | b"Purple"
        | b"Magenta" => {
            data.colors.push(ColorRange {
                kind: String::from_utf8_lossy(name).into_owned(),
                min: parse_f32(e, "min", 0.0),
                max: parse_f32(e, "max", 0.0),
                weight: parse_f32(e, "weight", 1.0),
            });
        }
        b"Decal" => data.decals.push(weighted_path(e)),
        b"Pattern" => data.patterns.push(weighted_path(e)),
        b"Patterns" => {
            data.pattern_tile = [parse_f32(e, "tileMin", 0.0), parse_f32(e, "tileMax", 0.0)];
        }
        b"HairTemplate" => data.hair_templates.push(weighted_path(e)),
        b"ZombifyDecal" => data.zombify_decals.push(weighted_path(e)),
        _ => {}
    }
}

fn weighted_path(e: &BytesStart) -> WeightedPath {
    WeightedPath {
        file: get_attr(e, "file").unwrap_or_default(),
        weight: parse_f32(e, "weight", 1.0),
    }
}

fn opt_f32(e: &BytesStart, name: &str) -> Option<f32> {
    get_attr(e, name).and_then(|v| v.trim().parse::<f32>().ok())
}

fn cur_body_mut<'a>(data: &'a mut VariantData, cur: &Cursor) -> Option<&'a mut BodyType> {
    data.genders.get_mut(cur.gender?)?.body_types.get_mut(cur.body?)
}

fn cur_model_mut<'a>(data: &'a mut VariantData, cur: &Cursor) -> Option<&'a mut Model> {
    let slot = cur.slot?;
    cur_body_mut(data, cur)?.slot_mut(slot).last_mut()
}

fn cur_shader_mut<'a>(data: &'a mut VariantData, cur: &Cursor) -> Option<&'a mut VariantShader> {
    cur_model_mut(data, cur)?.shaders.last_mut()
}

fn cur_index_mut<'a>(data: &'a mut VariantData, cur: &Cursor) -> Option<&'a mut VariantIndex> {
    cur_shader_mut(data, cur)?.indices.last_mut()
}

/// True for the classes the `ZombieVariant` system composes. Freaks carry a
/// self-contained `.cdf` from their scripts and are not built here.
pub fn is_variant_class(entity_class: &str) -> bool {
    matches!(entity_class, "Grunt" | "FreshGrunt" | "Eatter")
}

/// The gender a non-ambient variant class uses (`FreshGrunt` is the shipped
/// `FreshZombie` gender; the freak groups of other classes name their own).
fn fixed_gender(entity_class: &str) -> Option<&'static str> {
    match entity_class {
        "FreshGrunt" => Some("FreshZombie"),
        "Eatter" => Some("Eatter"),
        _ => None,
    }
}

/// Pick the appearance for `entity_class` with a deterministic seed (the spawn
/// handle). Ambient `Grunt`s roll gender and body type weighted; the fixed
/// classes use their gender and roll only the body type.
pub fn pick_for_class(data: &VariantData, entity_class: &str, seed: u32) -> Option<Appearance> {
    if !is_variant_class(entity_class) {
        return None;
    }
    let mut rng = VariantRng::new(seed);
    let (gi, bi) = match fixed_gender(entity_class) {
        Some(name) => {
            let gi = data
                .genders
                .iter()
                .position(|g| g.kind.eq_ignore_ascii_case(name))
                .or_else(|| data.genders.iter().position(|g| g.weight > 0.0))?;
            let weights: Vec<f32> = data.genders[gi].body_types.iter().map(|b| b.weight).collect();
            let bi = weighted_index(&weights, &mut rng);
            (gi, bi)
        }
        None => weighted_gender_body(data, &mut rng)?,
    };
    Some(compose(data, gi, bi, &mut rng))
}

/// Ambient draw: a flat list of `(gender, body_type)` weighted by
/// `gender.weight * body_type.weight`, exactly the normalised body-type array
/// the engine scans in `0x0133dc40`.
fn weighted_gender_body(data: &VariantData, rng: &mut VariantRng) -> Option<(usize, usize)> {
    let mut flat: Vec<(usize, usize, f32)> = Vec::new();
    for (gi, g) in data.genders.iter().enumerate() {
        if g.weight <= 0.0 {
            continue;
        }
        for (bi, b) in g.body_types.iter().enumerate() {
            if b.weight <= 0.0 {
                continue;
            }
            flat.push((gi, bi, g.weight * b.weight));
        }
    }
    if flat.is_empty() {
        return None;
    }
    let weights: Vec<f32> = flat.iter().map(|x| x.2).collect();
    let i = weighted_index(&weights, rng);
    Some((flat[i].0, flat[i].1))
}

/// Compose one body type: weighted model per present slot, then the per-model
/// shader/colour/pattern draws (`FUN_01320270` + `FUN_0131fd00`).
fn compose(data: &VariantData, gi: usize, bi: usize, rng: &mut VariantRng) -> Appearance {
    let gender = &data.genders[gi];
    let body = &gender.body_types[bi];
    let mut app = Appearance {
        gender: gender.kind.clone(),
        body_type: body.kind.clone(),
        skeleton: format!("{}.chr", gender.skeleton),
        parts: Vec::new(),
        pattern: None,
        pattern_tile: data.pattern_tile[0],
        decal: None,
    };
    for &slot in &SLOT_ORDER {
        let models = body.slot(slot);
        if models.is_empty() {
            continue;
        }
        let weights: Vec<f32> = models.iter().map(|m| m.weight).collect();
        let mi = weighted_index(&weights, rng);
        let model = &models[mi];
        let tints = pick_model_tints(data, model, rng, &mut app);
        app.parts.push(PickedPart {
            slot,
            file: format!("{}.chr", model.file),
            eigen: model.eigen.clone(),
            tints,
        });
    }
    app
}

/// Tint and pattern/decal draws for one picked model. The engine applies one
/// tint per `VariantIndex` (targeting sub-material `index`) and a pattern/decal
/// overlay when the shader odds accept.
fn pick_model_tints(
    data: &VariantData,
    model: &Model,
    rng: &mut VariantRng,
    app: &mut Appearance,
) -> Vec<Tint> {
    let mut tints = Vec::new();
    for shader in &model.shaders {
        for vi in &shader.indices {
            let prefs: Vec<f32> = vi.color_prefs.iter().map(|(_, w)| *w).collect();
            let (hue, brightness, saturation) = if prefs.iter().sum::<f32>() > 0.0 {
                let ci = weighted_index(&prefs, rng);
                let name = &vi.color_prefs[ci].0;
                let hue = match data.colors.iter().find(|c| c.kind.eq_ignore_ascii_case(name)) {
                    Some(c) => c.min + rng.unit() * (c.max - c.min) + vi.hue_offset,
                    None => vi.hue_offset,
                };
                let bright =
                    vi.brightness_min + rng.unit() * (vi.brightness_max - vi.brightness_min);
                let sat = vi.saturation_min
                    + rng.unit() * ((bright * vi.sat_slope + vi.sat_zero) - vi.saturation_min);
                (hue, bright, sat)
            } else {
                let bright =
                    vi.brightness_min + rng.unit() * (vi.brightness_max - vi.brightness_min);
                let sat = vi.saturation_min
                    + rng.unit() * ((bright * vi.sat_slope + vi.sat_zero) - vi.saturation_min);
                (vi.hue_offset, bright, sat)
            };
            tints.push(Tint {
                material_index: vi.index,
                hue,
                brightness,
                saturation,
            });
        }
        let pattern_odds = shader.pattern_odds.max(0.0);
        if pattern_odds > 0.0 && (rng.roll100() as f32) < pattern_odds * 100.0 {
            if let Some(p) = pick_path(&data.patterns, rng) {
                app.pattern = Some(p);
                app.pattern_tile = data.pattern_tile[0]
                    + rng.unit() * (data.pattern_tile[1] - data.pattern_tile[0]);
            }
        }
        let decal_odds = shader.decal_odds.max(0.0);
        if decal_odds > 0.0 && (rng.roll100() as f32) < decal_odds * 100.0 {
            if let Some(d) = pick_path(&data.decals, rng) {
                app.decal = Some(d);
            }
        }
    }
    tints
}

fn pick_path(paths: &[WeightedPath], rng: &mut VariantRng) -> Option<String> {
    if paths.is_empty() {
        return None;
    }
    let weights: Vec<f32> = paths.iter().map(|p| p.weight).collect();
    Some(paths[weighted_index(&weights, rng)].file.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small synthetic document (authored for this test, not shipped data).
    const SAMPLE: &str = r#"<VariantData>
      <Gender type="female" weight="3" skeleton="characters/female/zombie/skeleton/skeleton">
        <BodyType type="standard" weight="2">
          <Head><Model file="h/head01" weight="1"/></Head>
          <Skin><Model file="s/skin01" weight="1"/></Skin>
          <Body>
            <Model file="b/tshirt" weight="3">
              <EigenZombie file="b/low/low.cdf"/>
              <VariantShader type="shirt" patternOdds="0.8">
                <VariantIndex index="1" brightnessMin="0.1" brightnessMax="0.3"
                    satSlope="-0.4" satZero="0.6" saturationMin="0.15" hueOffset="-0.2"
                    patternOdds="0.8">
                  <ColorPrefs Orange="3" Yellow="1"/>
                </VariantIndex>
              </VariantShader>
            </Model>
            <Model file="b/blouse" weight="1"/>
          </Body>
          <Legs><Model file="l/jeans" weight="1"/></Legs>
          <Hair><Model file="hair/h01" weight="1"/></Hair>
          <SkinVariant maxDiffuse="0.25,0.3,0.45" minDiffuse="0.1,0.1,0.2"
             maxMelanin="0.6" zombifyDecalOdds="0"/>
        </BodyType>
      </Gender>
      <Gender type="male" weight="10" skeleton="characters/male/zombie/skeleton/skeleton">
        <BodyType type="noshirt" weight="1">
          <Body><Model file="b/noshirt02" weight="4"/></Body>
          <Legs><Model file="l/jeans" weight="1"/></Legs>
          <Head><Model file="h/head02" weight="1"/></Head>
        </BodyType>
      </Gender>
      <Colors><Red min="-0.16" max="0.35" weight="1"/><Blue min="2.2" max="2.62" weight="1"/></Colors>
      <Decals><Decal file="d/skull.tif" weight="2"/></Decals>
      <Patterns tileMin="0.75" tileMax="5.0"><Pattern file="p/plaid.tif" weight="6"/></Patterns>
      <HairTemplates><HairTemplate file="characters/male/human/hair/brown01" weight="12"/></HairTemplates>
    </VariantData>"#;

    #[test]
    fn parses_nested_structure() {
        let d = VariantData::parse(SAMPLE.as_bytes()).expect("root parsed");
        assert_eq!(d.genders.len(), 2);
        assert_eq!(d.genders[0].kind, "female");
        assert_eq!(d.genders[0].body_types.len(), 1);
        let bt = &d.genders[0].body_types[0];
        assert_eq!(bt.body.len(), 2);
        assert_eq!(bt.body[0].file, "b/tshirt");
        assert_eq!(bt.body[0].eigen.as_deref(), Some("b/low/low.cdf"));
        assert_eq!(bt.body[0].shaders.len(), 1);
        assert_eq!(bt.body[0].shaders[0].indices.len(), 1);
        assert_eq!(bt.body[0].shaders[0].indices[0].color_prefs.len(), 2);
        assert_eq!(bt.skin_variant.as_ref().unwrap().max_melanin, 0.6);
        assert_eq!(d.colors.len(), 2);
        assert_eq!(d.patterns.len(), 1);
        assert_eq!(d.pattern_tile, [0.75, 5.0]);
        assert_eq!(d.hair_templates.len(), 1);
    }

    #[test]
    fn rejects_non_variant_documents() {
        assert!(VariantData::parse(b"").is_none());
        assert!(VariantData::parse(b"<Other/>").is_none());
        assert!(VariantData::parse(b"\x00\xff garbage <").is_none());
    }

    #[test]
    fn grunt_rolls_gender_and_bodytype() {
        let d = VariantData::parse(SAMPLE.as_bytes()).unwrap();
        let mut seen_genders = std::collections::HashSet::new();
        for seed in 0..64u32 {
            let a = pick_for_class(&d, "Grunt", seed).expect("grunt pick");
            assert!(!a.parts.is_empty());
            assert!(a.skeleton.ends_with("skeleton.chr"));
            assert!(a.parts.iter().any(|p| p.slot == Slot::Legs));
            seen_genders.insert(a.gender.clone());
        }
        assert!(seen_genders.contains("female") && seen_genders.contains("male"));
    }

    #[test]
    fn fixed_class_uses_its_gender() {
        let d = VariantData::parse(SAMPLE.as_bytes()).unwrap();
        let a = pick_for_class(&d, "FreshGrunt", 7).expect("fresh pick");
        assert!(!a.parts.is_empty());
        // No FreshZombie in the sample, so it falls back to the first weighted
        // gender; the important part is that a fixed class does not roll gender.
        assert!(is_variant_class("FreshGrunt"));
        assert_eq!(pick_for_class(&d, "Feral", 1), None);
        assert!(!is_variant_class("Bloater"));
    }

    #[test]
    fn same_seed_is_deterministic_and_seeds_vary() {
        let d = VariantData::parse(SAMPLE.as_bytes()).unwrap();
        let a = pick_for_class(&d, "Grunt", 1234).unwrap();
        let b = pick_for_class(&d, "Grunt", 1234).unwrap();
        assert_eq!(a, b);
        let keys: std::collections::HashSet<u64> =
            (0..32).map(|s| pick_for_class(&d, "Grunt", s).unwrap().geometry_key()).collect();
        assert!(keys.len() > 1, "seeds must produce varied geometry");
    }

    // ---- Real shipped-data checks (silently skipped when absent) ----

    fn game_xml() -> Option<Vec<u8>> {
        let p = std::env::var_os("SOD_VARIANT_XML").map(std::path::PathBuf::from).unwrap_or_else(
            || {
                std::path::PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../../extracted/libs/class3/zombies/zombievariant.xml"
                ))
            },
        );
        std::fs::read(p).ok()
    }

    fn character_vfs() -> Option<crate::crypak::Vfs> {
        let root = std::env::var_os("SOD_CHARACTERS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../../extracted-game/characters"
                ))
            });
        let root = root.canonicalize().ok()?;
        let mut vfs = crate::crypak::Vfs::new();
        vfs.set_loose_root(&root);
        Some(vfs)
    }

    #[test]
    fn real_variant_file_parses() {
        let Some(bytes) = game_xml() else { return };
        let d = VariantData::parse(&bytes).expect("ZombieVariant.xml parses");
        assert!(d.genders.len() >= 10, "shipped genders: {}", d.genders.len());
        let male = d.genders.iter().find(|g| g.kind == "male").expect("male gender");
        assert!(male.weight > 0.0);
        assert!(male.body_types.iter().any(|b| b.kind == "noshirt"));
        assert!(male.body_types.iter().any(|b| b.kind == "standard"));
        assert!(!male.body_types.iter().any(|b| b.body.is_empty()), "every body type has a body");
        assert!(d.colors.len() >= 6);
        assert!(d.patterns.len() >= 6);
        assert!(!d.hair_templates.is_empty());

        // Ambient Grunts only roll the positively weighted genders.
        let mut genders = std::collections::HashSet::new();
        let mut saw_legs = false;
        for seed in 0..256u32 {
            let a = pick_for_class(&d, "Grunt", seed).expect("grunt pick");
            // Every body type carries a body; leggings are present in most but
            // not all (the `fat` body type bakes them into the body mesh).
            assert!(a.parts.iter().any(|p| p.slot == Slot::Body));
            assert!(a.parts.iter().any(|p| p.slot == Slot::Legs || p.slot == Slot::Skin));
            saw_legs |= a.parts.iter().any(|p| p.slot == Slot::Legs);
            genders.insert(a.gender.clone());
        }
        assert!(genders.contains("male") && genders.contains("female"));
        assert!(saw_legs);
    }

    /// Every model combination in every shipped body type of the two humanoid
    /// genders must decode into one continuous body (no hollow z-slab).
    #[test]
    fn every_humanoid_combination_is_continuous() {
        let (Some(bytes), Some(vfs)) = (game_xml(), character_vfs()) else { return };
        let d = VariantData::parse(&bytes).expect("parse");
        let mut checked = 0usize;
        for gender in d.genders.iter().filter(|g| g.weight > 0.0) {
            let skeleton = format!("{}.chr", gender.skeleton);
            if vfs.read(&skeleton).is_none() {
                continue;
            }
            for body in &gender.body_types {
                for combo in model_combos(body) {
                    let spec: Vec<crate::cdf::AssemblyPart> = combo
                        .iter()
                        .map(|(slot, file)| crate::cdf::AssemblyPart {
                            name: slot.label(),
                            binding: file,
                        })
                        .collect();
                    let Some((skel, parts)) = crate::cdf::assemble_explicit(&vfs, &skeleton, &spec)
                    else {
                        continue;
                    };
                    assert!(skel.bones.len() >= 40, "{gender:?}/{body:?} rig");
                    let merged = crate::cdf::merge_parts(&vfs, &parts);
                    assert!(!merged.model.meshes.is_empty(), "{gender:?}/{body:?} geometry");
                    // No empty 5 cm slab between the lowest and highest vertex.
                    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                    let mut hist = [0usize; 40];
                    for m in &merged.model.meshes {
                        for p in &m.positions {
                            mn = mn.min(p[2]);
                            mx = mx.max(p[2]);
                            hist[((p[2].max(0.0) * 20.0) as usize).min(39)] += 1;
                        }
                    }
                    assert!(mn < 0.1, "{gender:?}/{body:?} feet min z {mn}");
                    let lo = (mn.max(0.0) * 20.0) as usize;
                    let hi = ((mx * 20.0) as usize).min(39);
                    for b in lo..=hi {
                        assert!(
                            hist[b] > 0,
                            "{gender:?}/{body:?}: hollow z-slab {b} (mn {mn} mx {mx})"
                        );
                    }
                    checked += 1;
                }
            }
        }
        eprintln!("variant: checked {checked} humanoid combinations");
        assert!(checked > 100, "expected many combinations, got {checked}");
    }

    fn model_combos(body: &BodyType) -> Vec<Vec<(Slot, String)>> {
        let mut out: Vec<Vec<(Slot, String)>> = vec![Vec::new()];
        for &slot in &SLOT_ORDER {
            let models = body.slot(slot);
            if models.is_empty() {
                continue;
            }
            let mut next = Vec::new();
            for base in &out {
                for m in models {
                    let mut v = base.clone();
                    v.push((slot, format!("{}.chr", m.file)));
                    next.push(v);
                }
            }
            out = next;
        }
        out
    }
}
