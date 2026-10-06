//! CryEngine `.cdf` (Character Definition File) — assembled characters.
//!
//! A survivor is not a single `.chr`: the `.cdf` names a base skeleton `.chr`
//! plus a list of skin attachments (head, hair, body/clothes), each bound to a
//! `.chr` and optionally parented to a bone. The app used to load one `.chr`
//! directly, so the player rendered as a bare, wrong-looking body; this module
//! assembles the full part list instead.
//!
//! Example (`characters/male/human/hair/variant/hair06.cdf`):
//! ```xml
//! <CharacterDefinition>
//!  <Model File="characters/male/human/skeleton/skeleton.chr"
//!         Material="characters/male/human/skeleton/skeleton"/>
//!  <AttachmentList>
//!   <Attachment AName="head" Binding=".../heads/head14/head.chr"
//!               BoneName="" Position="0,0,0" Rotation="1,0,0,0" Type="CA_SKIN"/>
//!   <Attachment AName="hair" Binding=".../hair/hair06/hair06.chr"
//!               Material=".../hair/brown02.mtl" BoneName="Bip01 Head"
//!               Position="..." Rotation="1,0,0,0" Type="CA_SKIN"/>
//!  </AttachmentList>
//! </CharacterDefinition>
//! ```
//!
//! Robustness: `quick-xml` errors terminate parsing, malformed/missing
//! attributes fall back to defaults, and nothing here can panic.

use crate::character;
use crate::crypak::Vfs;
use crate::material::{self, MaterialSlot};
use crate::mesh;
use crate::skin;
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// One `<Attachment>` entry of a character definition.
#[derive(Debug, Clone)]
pub struct CdfAttachment {
    /// `AName` slot label, e.g. `head` / `hair` / `body`.
    pub name: String,
    /// `Binding` `.chr` path (forward-slash normalised).
    pub binding: String,
    /// Optional `Material` override (may already carry a `.mtl` extension).
    pub material: Option<String>,
    /// Optional `BoneName` the attachment is parented to.
    pub bone: Option<String>,
    /// Local translation (metres), default `(0,0,0)`.
    pub position: [f32; 3],
    /// Local rotation quaternion `(x,y,z,w)`, default identity.
    pub rotation: [f32; 4],
    /// `Type` (e.g. `CA_SKIN`).
    pub kind: Option<String>,
}

/// A parsed `.cdf`.
#[derive(Debug, Clone)]
pub struct Cdf {
    /// Base `Model File` `.chr` (skeleton).
    pub model: String,
    /// Optional base `Model Material` (`.mtl` name without extension).
    pub model_material: Option<String>,
    /// Skin attachments in document order.
    pub attachments: Vec<CdfAttachment>,
}

const ZERO_POS: [f32; 3] = [0.0, 0.0, 0.0];
const IDENTITY_ROT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Parse a `.cdf`. Returns `None` unless the document contains both a
/// `<CharacterDefinition>` and a non-empty `<Model File="...">`. Never panics.
pub fn parse(xml: &[u8]) -> Option<Cdf> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut saw_cd = false;
    let mut in_cd = false;
    let mut in_attach = false;
    let mut model: Option<String> = None;
    let mut model_material: Option<String> = None;
    let mut attachments: Vec<CdfAttachment> = Vec::new();

    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            // Treat malformed XML as end-of-input rather than panicking.
            Err(_) => break,
            Ok(ev) => ev,
        };
        match ev {
            Event::Start(e) => {
                let name = e.local_name();
                match name.as_ref() {
                    b"CharacterDefinition" => {
                        saw_cd = true;
                        in_cd = true;
                    }
                    b"AttachmentList" => in_attach = true,
                    b"Model" if in_cd => apply_model(&e, &mut model, &mut model_material),
                    b"Attachment" if in_attach => {
                        if let Some(a) = attachment_from(&e) {
                            attachments.push(a);
                        }
                    }
                    _ => {}
                }
            }
            Event::Empty(e) => {
                let name = e.local_name();
                match name.as_ref() {
                    b"CharacterDefinition" => saw_cd = true,
                    b"Model" if in_cd => apply_model(&e, &mut model, &mut model_material),
                    b"Attachment" if in_attach => {
                        if let Some(a) = attachment_from(&e) {
                            attachments.push(a);
                        }
                    }
                    _ => {}
                }
            }
            Event::End(e) => {
                let name = e.local_name();
                match name.as_ref() {
                    b"CharacterDefinition" => in_cd = false,
                    b"AttachmentList" => in_attach = false,
                    _ => {}
                }
            }
            _ => {}
        }
        buf.clear();
    }

    if !saw_cd {
        return None;
    }
    Some(Cdf { model: model?, model_material, attachments })
}

/// Fold a `<Model>` element into the accumulated base model info. A later
/// `Model` without a `File` never clears an earlier one.
fn apply_model(e: &BytesStart, model: &mut Option<String>, material: &mut Option<String>) {
    if let Some(f) = path_attr(e, "File") {
        *model = Some(f);
    }
    if let Some(m) = path_attr(e, "Material") {
        *material = Some(m);
    }
}

fn attachment_from(e: &BytesStart) -> Option<CdfAttachment> {
    let binding = path_attr(e, "Binding")?;
    Some(CdfAttachment {
        name: get_attr(e, "AName").unwrap_or_default(),
        binding,
        material: path_attr(e, "Material"),
        bone: get_attr(e, "BoneName").map(|s| s.trim().to_string()).filter(|s| !s.is_empty()),
        position: parse_vec3(get_attr(e, "Position").as_deref(), ZERO_POS),
        rotation: parse_vec4(get_attr(e, "Rotation").as_deref(), IDENTITY_ROT),
        kind: get_attr(e, "Type").map(|s| s.trim().to_string()).filter(|s| !s.is_empty()),
    })
}

/// Look up one attribute by exact case-sensitive name, unescaping XML entities.
fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

/// Like [`get_attr`], but trims and normalises path separators to `/`; empty
/// values are treated as absent.
fn path_attr(e: &BytesStart, name: &str) -> Option<String> {
    get_attr(e, name)
        .map(|s| s.trim().replace('\\', "/"))
        .filter(|s| !s.is_empty())
}

fn parse_floats(s: &str) -> Vec<f32> {
    s.replace(' ', "")
        .split(',')
        .filter(|p| !p.is_empty())
        .filter_map(|p| p.parse::<f32>().ok())
        .collect()
}

fn parse_vec3(s: Option<&str>, default: [f32; 3]) -> [f32; 3] {
    match s {
        Some(t) => {
            let v = parse_floats(t);
            if v.len() >= 3 {
                [v[0], v[1], v[2]]
            } else {
                default
            }
        }
        None => default,
    }
}

fn parse_vec4(s: Option<&str>, default: [f32; 4]) -> [f32; 4] {
    match s {
        Some(t) => {
            let v = parse_floats(t);
            if v.len() >= 4 {
                [v[0], v[1], v[2], v[3]]
            } else {
                default
            }
        }
        None => default,
    }
}

/// One resolved piece of a `.cdf`: geometry + palette + placement.
#[derive(Debug, Clone)]
pub struct CharacterPart {
    /// `AName` attachment slot (`body`, `skin`, `legs`, ...); empty for the
    /// base model.
    pub name: String,
    /// Source `.chr` path of this piece.
    pub path: String,
    /// Decoded skinned mesh (bind model space).
    pub model: mesh::DecodedModel,
    /// Material palette indices used by `model`'s submeshes.
    pub slots: Vec<MaterialSlot>,
    /// Attachment bone, when it is an attachment (base model => `None`).
    pub bone: Option<String>,
    /// Attachment local position (`(0,0,0)` for the base model).
    pub position: [f32; 3],
    /// Attachment local rotation (`(0,0,0,1)` for the base model).
    pub rotation: [f32; 4],
}

/// Resolve a material palette for `model_path` from an explicit `Material`
/// attribute first, then the model's own root material name.
///
/// The explicit name may already carry a `.mtl` extension (attachment
/// `Material=".../brown02.mtl"`), so a direct file reference is tried before
/// delegating to [`material::load_mtl`] with the extension stripped.
fn resolve_slots(
    vfs: &Vfs,
    model_path: &str,
    explicit: Option<&str>,
    own: Option<&str>,
) -> Vec<MaterialSlot> {
    if let Some(name) = explicit.map(str::trim).filter(|s| !s.is_empty()) {
        let direct = name.replace('\\', "/");
        if vfs.exists(&direct) {
            if let Some(bytes) = vfs.read(&direct) {
                return material::parse_mtl_slots(&bytes, vfs);
            }
        }
        let stem = direct
            .strip_suffix(".mtl")
            .or_else(|| direct.strip_suffix(".MTL"))
            .unwrap_or(&direct);
        let slots = material::load_mtl(vfs, model_path, stem);
        if !slots.is_empty() {
            return slots;
        }
    }
    match own {
        Some(name) if !name.is_empty() => material::load_mtl(vfs, model_path, name),
        _ => Vec::new(),
    }
}

/// The `.cdf` slot that carries the clothed torso garment.
const BODY_SLOT: &str = "body";
/// The `.cdf` slot that carries the bare/full-body base layer.
const SKIN_SLOT: &str = "skin";

/// Indices of `parts` that duplicate the body layer and should be dropped.
///
/// A shipped `.cdf` normally lists a `body` attachment (the garment:
/// `tshirt_jeans`, `jacket01`, ...) *and* a `skin` attachment. The `skin`
/// binding is a complete, already-clothed body mesh
/// (`characters/*/human/skin/*/skin.chr`, ~4k vertices spanning neck to hips)
/// while `body` is the visible garment on top; rendering both stacks two whole
/// characters at the same transform — the "two overlapping models" bug.
/// CryEngine treats `skin` as the default under-layer that a `body` slot
/// replaces, so whenever a decoded `body` part exists every `skin` part is
/// redundant and dropped.
///
/// Only parts whose geometry actually decoded are considered, so a `body`
/// attachment that failed to load never causes the surviving skin to be culled.
/// Returns an empty vector when there is no body layer.
pub fn is_redundant_skin(parts: &[CharacterPart]) -> Vec<usize> {
    let has_body = parts
        .iter()
        .any(|p| p.name.eq_ignore_ascii_case(BODY_SLOT) && !p.model.meshes.is_empty());
    if !has_body {
        return Vec::new();
    }
    parts
        .iter()
        .enumerate()
        .filter(|(_, p)| p.name.eq_ignore_ascii_case(SKIN_SLOT))
        .map(|(i, _)| i)
        .collect()
}

/// Decode the base model plus every resolvable attachment of an already-parsed
/// `cdf`, then drop redundant layers (see [`is_redundant_skin`]) so the result
/// is a single, non-overlapping character.
///
/// Missing/unreadable attachments are skipped rather than failing the whole
/// load; `None` only when the base `.chr` or its skeleton cannot be read.
fn assemble(vfs: &Vfs, cdf: &Cdf) -> Option<(skin::Skeleton, Vec<CharacterPart>)> {
    let base = vfs.read(&cdf.model)?;
    let skeleton = skin::parse_skeleton(&base)?;

    let mut parts = Vec::new();
    if let Some(model) = mesh::decode_model_with(&base, false) {
        let slots = resolve_slots(
            vfs,
            &cdf.model,
            cdf.model_material.as_deref(),
            model.material_name.as_deref(),
        );
        parts.push(CharacterPart {
            name: String::new(),
            path: cdf.model.clone(),
            model,
            slots,
            bone: None,
            position: ZERO_POS,
            rotation: IDENTITY_ROT,
        });
    }

    for att in &cdf.attachments {
        let Some(ab) = vfs.read(&att.binding) else { continue };
        let Some(model) = mesh::decode_model_with(&ab, false) else { continue };
        let slots = resolve_slots(
            vfs,
            &att.binding,
            att.material.as_deref(),
            model.material_name.as_deref(),
        );
        parts.push(CharacterPart {
            name: att.name.clone(),
            path: att.binding.clone(),
            model,
            slots,
            bone: att.bone.clone(),
            position: att.position,
            rotation: att.rotation,
        });
    }

    // Remove in reverse so earlier indices stay valid.
    for i in is_redundant_skin(&parts).into_iter().rev() {
        parts.remove(i);
    }
    Some((skeleton, parts))
}

/// Load the base model plus every resolvable attachment of `cdf_path`.
///
/// Historical entry point (the app calls it); it now delegates to
/// [`load_assembled`], so the returned parts are already de-duplicated.
pub fn load_parts(vfs: &Vfs, cdf_path: &str) -> Option<(skin::Skeleton, Vec<CharacterPart>)> {
    load_assembled(vfs, cdf_path)
}

/// Load and assemble `cdf_path` into a single, non-overlapping character.
///
/// Identical to [`load_parts`] but named for the assembly semantics: the bare
/// `skin` layer is dropped when a `body` garment is present (see
/// [`is_redundant_skin`]), so the player renders as one model rather than two
/// stacked ones. The base `Model` contributes a part only when its `.chr`
/// actually carries geometry (a bare `skeleton.chr` usually does not).
pub fn load_assembled(vfs: &Vfs, cdf_path: &str) -> Option<(skin::Skeleton, Vec<CharacterPart>)> {
    let bytes = vfs.read(cdf_path)?;
    let cdf = parse(&bytes)?;
    assemble(vfs, &cdf)
}

/// Look up a bone index by name.
///
/// `skin::Skeleton` only indexes bones by controller id (`by_controller`); it
/// exposes no name → index map, so this always returns `None`. Callers that
/// need a name lookup must scan `skeleton.bones` themselves.
pub fn attachment_bone_index(_skeleton: &skin::Skeleton, _bone: &str) -> Option<usize> {
    None
}

/// Convenience: load a full animatable character and its clip bank using the
/// CDF's base model as the rig.
pub fn load_character(
    vfs: &Vfs,
    cdf_path: &str,
    rig: Option<&str>,
    max_caf: usize,
) -> Option<(character::Character, character::ClipBank)> {
    let bytes = vfs.read(cdf_path)?;
    let cdf = parse(&bytes)?;
    let chr = character::load(vfs, &cdf.model, rig, max_caf)?;
    let bank = character::ClipBank::build(vfs, &cdf.model, rig, max_caf);
    Some((chr, bank))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    const SAMPLE: &str = r#"<?xml version="1.0" ?><CharacterDefinition>
 <Model File="characters/male/human/skeleton/skeleton.chr" Material="characters/male/human/skeleton/skeleton"/>
 <AttachmentList>
  <Attachment AName="head" Binding="characters/male/human/heads/head14/head.chr" BoneName="" Position="0,0,0" Rotation="1,0,0,0" Type="CA_SKIN"/>
  <Attachment AName="hair" Binding="characters/male/human/hair/hair06/hair06.chr" Material="characters/male/human/hair/brown02.mtl" BoneName="Bip01 Head" Position="1.9475756e-005,0.017000001,1.628" Rotation="1,0,0,0" Type="CA_SKIN"/>
 </AttachmentList>
 <ShapeDeformation COL0="0" COL1="0" COL2="0" COL3="0" COL4="0" COL5="0" COL6="0" COL7="0"/>
</CharacterDefinition>"#;

    #[test]
    fn parse_sample_model_and_attachments() {
        let cdf = parse(SAMPLE.as_bytes()).expect("sample parses");
        assert_eq!(cdf.model, "characters/male/human/skeleton/skeleton.chr");
        assert_eq!(
            cdf.model_material.as_deref(),
            Some("characters/male/human/skeleton/skeleton")
        );
        assert_eq!(cdf.attachments.len(), 2);

        let head = &cdf.attachments[0];
        assert_eq!(head.name, "head");
        assert_eq!(head.binding, "characters/male/human/heads/head14/head.chr");
        assert_eq!(head.material, None);
        assert_eq!(head.bone, None, "empty BoneName -> None");
        assert_eq!(head.position, [0.0, 0.0, 0.0]);
        // Stored verbatim; the file's identity is authored scalar-first.
        assert_eq!(head.rotation, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(head.kind.as_deref(), Some("CA_SKIN"));

        let hair = &cdf.attachments[1];
        assert_eq!(hair.name, "hair");
        assert_eq!(hair.binding, "characters/male/human/hair/hair06/hair06.chr");
        assert_eq!(
            hair.material.as_deref(),
            Some("characters/male/human/hair/brown02.mtl")
        );
        assert_eq!(hair.bone.as_deref(), Some("Bip01 Head"));
        assert_eq!(hair.kind.as_deref(), Some("CA_SKIN"));
        assert!((hair.position[0] - 1.9475756e-5).abs() < 1e-9);
        assert!((hair.position[1] - 0.017000001).abs() < 1e-9);
        assert!((hair.position[2] - 1.628).abs() < 1e-9);
        assert_eq!(hair.rotation, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn parse_lenient_on_junk_never_panics() {
        assert!(parse(b"").is_none());
        assert!(parse(b"not xml at all").is_none());
        assert!(parse(b"\x00\xff\xfe garbage <").is_none());
        // A Model without a CharacterDefinition is not enough.
        assert!(parse(br#"<Model File="a.chr"/>"#).is_none());
        // A CharacterDefinition without a Model is not enough.
        assert!(parse(br#"<CharacterDefinition/>"#).is_none());
        assert!(parse(br#"<CharacterDefinition><AttachmentList/></CharacterDefinition>"#).is_none());
        // Truncated documents must not panic (result may be Some or None).
        let _ = parse(br#"<CharacterDefinition><Model File="a.chr"#);
        let _ = parse(br#"<CharacterDefinition><Attachment AName="x""#);
        // Attributes that fail to parse fall back to defaults.
        let cdf = parse(
            br#"<CharacterDefinition><Model File="a.chr"/>
                <AttachmentList><Attachment AName="x" Binding="b.chr"
                    Position="junk" Rotation="1,2"/></AttachmentList>
               </CharacterDefinition>"#,
        )
        .expect("lenient parse");
        assert_eq!(cdf.attachments.len(), 1);
        assert_eq!(cdf.attachments[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(cdf.attachments[0].rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(cdf.attachments[0].name, "x");
    }

    #[test]
    fn attachment_without_binding_is_skipped() {
        let cdf = parse(
            br#"<CharacterDefinition><Model File="a.chr"/>
                <AttachmentList>
                  <Attachment AName="x" Position="0,0,0"/>
                  <Attachment AName="y" Binding="y.chr"/>
                </AttachmentList></CharacterDefinition>"#,
        )
        .expect("parse");
        assert_eq!(cdf.attachments.len(), 1);
        assert_eq!(cdf.attachments[0].name, "y");
    }

    fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>, limit: usize) {
        if depth > 10 || out.len() >= limit {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for ent in rd.flatten() {
            if out.len() >= limit {
                return;
            }
            let p = ent.path();
            if p.is_dir() {
                walk(&p, depth + 1, out, limit);
            } else if p.extension().map(|e| e.eq_ignore_ascii_case("cdf")).unwrap_or(false) {
                out.push(p);
            }
        }
    }

    /// Optional real-asset smoke test: if the extracted game tree is present,
    /// assemble a real `.cdf` and require a skeleton plus at least one part.
    /// Silently skipped when the tree is absent.
    #[test]
    fn real_cdf_loads_parts_when_available() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game/characters");
        let Ok(root) = root.canonicalize() else { return };
        let mut cdfs = Vec::new();
        walk(&root, 0, &mut cdfs, 120);
        if cdfs.is_empty() {
            return;
        }

        let mut loaded = None;
        for c in &cdfs {
            let Ok(rel) = c.strip_prefix(&root) else { continue };
            let rel = rel.to_string_lossy().replace('\\', "/");
            let mut vfs = Vfs::new();
            vfs.set_loose_root(&root);
            if let Some((skel, parts)) = load_parts(&vfs, &rel) {
                if !skel.bones.is_empty() && !parts.is_empty() {
                    loaded = Some((rel, skel.bones.len(), parts));
                    break;
                }
            }
        }

        let Some((rel, bones, parts)) = loaded else {
            eprintln!("real_cdf: no loadable .cdf found under {root:?}");
            return;
        };
        let slots: usize = parts.iter().map(|p| p.slots.len()).sum();
        eprintln!(
            "real_cdf: {rel} -> bones={bones} parts={} slots={slots}",
            parts.len()
        );
        assert!(bones > 0);
        assert!(!parts.is_empty());
        assert!(parts.iter().all(|p| !p.model.meshes.is_empty()));
    }

    /// Real `v01.cdf`: assembly must leave exactly one body layer.
    #[test]
    fn real_v01_assembles_single_body_layer() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game/characters");
        let Ok(root) = root.canonicalize() else { return };
        let mut vfs = Vfs::new();
        vfs.set_loose_root(&root);
        let rel = "characters/male/human/body/tshirt_jeans/variant/v01.cdf";
        if vfs.read(rel).is_none() {
            return;
        }
        let (skel, parts) = load_assembled(&vfs, rel).expect("v01 assembles");
        assert!(skel.bones.len() > 0, "skeleton present");
        assert!(parts.len() > 3, "expected several parts, got {}", parts.len());
        let bodies = parts.iter().filter(|p| p.name.eq_ignore_ascii_case("body")).count();
        let skins = parts.iter().filter(|p| p.name.eq_ignore_ascii_case("skin")).count();
        assert_eq!(bodies, 1, "exactly one body layer");
        assert_eq!(skins, 0, "bare skin layer culled");
        assert!(parts.iter().all(|p| !p.model.meshes.is_empty()));
    }

    /// Synthetic CDF listing both `body` and `skin` against real assets: the
    /// skin layer must be dropped while body/legs survive.
    #[test]
    fn synthetic_body_and_skin_drops_skin() {
        const SYNTH: &str = r#"<CharacterDefinition>
 <Model File="characters/male/human/skeleton/skeleton.chr"/>
 <AttachmentList>
  <Attachment AName="body" Binding="characters/male/human/body/tshirt_jeans/tshirt_jeans.chr" Type="CA_SKIN"/>
  <Attachment AName="skin" Binding="characters/male/human/skin/med_sleeve_thin_shirt/skin.chr" Type="CA_SKIN"/>
  <Attachment AName="legs" Binding="characters/male/human/legs/jeans/jeans.chr" Type="CA_SKIN"/>
 </AttachmentList></CharacterDefinition>"#;

        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../extracted-game/characters");
        let Ok(root) = root.canonicalize() else { return };
        let mut vfs = Vfs::new();
        vfs.set_loose_root(&root);
        let cdf = parse(SYNTH.as_bytes()).expect("synthetic cdf parses");
        // Silently skip when the referenced assets are absent.
        if vfs.read(&cdf.model).is_none()
            || vfs.read("characters/male/human/body/tshirt_jeans/tshirt_jeans.chr").is_none()
            || vfs.read("characters/male/human/skin/med_sleeve_thin_shirt/skin.chr").is_none()
        {
            return;
        }
        let (skel, parts) = assemble(&vfs, &cdf).expect("synthetic cdf assembles");
        assert!(skel.bones.len() > 0);
        assert!(parts.iter().any(|p| p.name.eq_ignore_ascii_case("body")));
        assert!(!parts.iter().any(|p| p.name.eq_ignore_ascii_case("skin")));
        assert!(parts.iter().any(|p| p.name.eq_ignore_ascii_case("legs")));
    }

    fn fake_part(name: &str, meshes: usize) -> CharacterPart {
        let mut model = crate::mesh::DecodedModel::default();
        for _ in 0..meshes {
            model.meshes.push(crate::mesh::DecodedMesh::default());
        }
        CharacterPart {
            name: name.to_string(),
            path: format!("{name}.chr"),
            model,
            slots: Vec::new(),
            bone: None,
            position: ZERO_POS,
            rotation: IDENTITY_ROT,
        }
    }

    #[test]
    fn is_redundant_skin_culls_only_with_body() {
        let with_body = [fake_part("body", 1), fake_part("skin", 1), fake_part("legs", 1)];
        assert_eq!(is_redundant_skin(&with_body), vec![1]);

        let no_body = [fake_part("skin", 1), fake_part("legs", 1)];
        assert!(is_redundant_skin(&no_body).is_empty());

        // A body attachment that decoded no geometry must not cull the skin.
        let empty_body = [fake_part("body", 0), fake_part("skin", 1)];
        assert!(is_redundant_skin(&empty_body).is_empty());

        // The base model slot (empty name) is not a body layer.
        let base_only = [fake_part("", 1), fake_part("skin", 1)];
        assert!(is_redundant_skin(&base_only).is_empty());
    }
}
