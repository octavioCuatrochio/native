//! AnimationGraph animation-name → real clip resolution (CryAnimation set).
//!
//! The AnimationGraph (`animations/graphs/*.xml`) names animations through
//! templates with placeholders — e.g. state `RelaxedWalkNW` plays
//! `_RELAXED_WALKSTRAFE_NW_01`, the generic state `X_[Stance]_Walk_[Item]`
//! plays `_[STANCE]_WALKSTRAFE_[ITEM]_01`. Those are **virtual animation
//! names**, not files. Resolving one means walking the character's animation
//! list exactly the way CryAnimation's animation set does.
//!
//! ## The chain (addresses from `StateOfDecay.unpacked.exe`)
//!
//! 1. **Graph name** — `CAnimationGraph::LoadStates` `0x68f550` /
//!    `PreprocessParameterization` `0x6924c0` expand `[STANCE]`/`[ITEM]`/
//!    `[DIR]`… macros into the concrete name the state plays. The leading `_`
//!    is part of the alias, not a path separator.
//! 2. **Character animation list** — the `.cdf` names the base `.chr`; the
//!    sibling `<chr>.chrparams` is parsed by the animation-list parser
//!    `FUN_00ad5d00` (`0xad5d00`). It understands:
//!    * `#filepath` — default directory for later relative paths;
//!    * `$Include` — recurse into another `.chrparams` (root-relative);
//!    * `$TracksDatabase` — a `.dba` of packed clips;
//!    * `$AnimEventDatabase` / `$FaceLib` — ignored here (not clips);
//!    * `<Animation name="alias" path="…"/>` — register an alias; a `*`/`?`
//!      in `path` is a directory wildcard and a `*` in `name` is replaced by
//!      the matched file stem (`Test_*` + `hero_run_forward.caf` →
//!      `Test_hero_run_forward`);
//!    * a `.lmg` target is a **LocomotionGroup** (a parametric direction/
//!      speed blend space), parsed by `FUN_00b50c70` (`0xb50c70`).
//! 3. **Alias registration** — `FUN_00afce20`/`00afd100`/`00afd2e0`/
//!    `00afd560` (`0xafce20`…) install each alias into the animation set
//!    (CAF or LMG), logging "Trying to load animation with alias".
//! 4. **Name lookup** — `FUN_00afad20` (`0xafad20`) calls the set's
//!    `GetAnimationId` (vtable `+0x14`); on a hit it resolves the CAF
//!    (`FUN_00afbf10`) or the LMG (`FUN_00afbf70`), otherwise logs
//!    `"animation-name '%s' not in CAL-file"` (`0x179e268`). The lookup is
//!    exact (case-insensitive); there is no engine-side rewriting.
//! 5. **Clip data** — a CAF is either loose (`.caf`/`.$caf`, the packed
//!    spelling) or packed in a `.dba` (`FUN_00abd0e0`, chunk `0xCCCC000D`
//!    v`0x905`); the `.img` containers (`0xAAFC0007/8`) carry compact clip
//!    headers keyed by `.caf` path.
//!
//! ## The STRAFE remap
//!
//! `playerfullbody.xml` was authored for an unshipped character (its
//! `<Character filename>` is `nk_soldier_camp_flanker_light_01.cdf`), so its
//! literal names are absent from the shipped player list
//! (`characters/male/human/skeleton/skeleton.chrparams` → `basic_player.chrparams`
//! + `unshared_male.chrparams`). The shipped list names the same locomotion
//! groups in the older scheme:
//!
//! | graph virtual name                | shipped alias | LMG file                  |
//! |-----------------------------------|---------------|---------------------------|
//! | `_RELAXED_RUNSTRAFE_NW_01`        | `_RUN_NW`     | `lmg_files/relaxed_run.lmg`|
//! | `_RELAXED_WALKSTRAFE_NW_01`       | `_WALK_NW`    | `lmg_files/relaxed_walk.lmg`|
//! | `_[STANCE]_SPRINTSTRAFE_[ITEM]_01`| `_SPRINT_NW`  | `lmg_files/relaxed_sprint.lmg`|
//!
//! i.e. `_<STANCE>_<GAIT>STRAFE_<ITEM>_<VER>` ≡ `_<GAIT>_<ITEM>` (the stance
//! is dropped and `STRAFE` removed), which is also how the girl's LMG files
//! are named (`combat_runstrafe_nw_01.lmg`). [`AnimNameTable::resolve`] tries
//! the literal alias first and only then this documented remap, so a character
//! that *does* define the literal name is unaffected.

use crate::cdf;
use crate::cryimg;
use crate::crypak::Vfs;
use crate::dba;
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::collections::HashMap;

/// Blend-space flavour of a LocomotionGroup (`<BLENDTYPE type="…">`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlendType {
    /// 1-D free-form (gait axis only).
    Flr1,
    /// 2-D free-form (direction × gait).
    Flr2,
    /// 1-D strafe.
    Stf1,
    /// 2-D strafe.
    Stf2,
    /// Idle rotate.
    Irot,
    /// Anything unrecognised, kept verbatim.
    Other(String),
}

impl BlendType {
    fn from_code(code: &str) -> BlendType {
        match code.trim().to_ascii_uppercase().as_str() {
            "FLR1" => BlendType::Flr1,
            "FLR2" => BlendType::Flr2,
            "STF1" => BlendType::Stf1,
            "STF2" => BlendType::Stf2,
            "IROT" => BlendType::Irot,
            other => BlendType::Other(other.to_string()),
        }
    }

    /// Short tag (`FLR2`, …); `Other` returns its stored text.
    pub fn as_str(&self) -> &str {
        match self {
            BlendType::Flr1 => "FLR1",
            BlendType::Flr2 => "FLR2",
            BlendType::Stf1 => "STF1",
            BlendType::Stf2 => "STF2",
            BlendType::Irot => "IROT",
            BlendType::Other(s) => s.as_str(),
        }
    }
}

/// One `<Example>` of a LocomotionGroup: an alias plus its blend position.
#[derive(Debug, Clone, PartialEq)]
pub struct LmgExample {
    /// `AName` — an animation alias (often a `Test_*` wildcard alias).
    pub alias: String,
    /// `Position` in the blend grid (`x,y,z`); `[0,0,0]` when absent.
    pub position: [f32; 3],
}

/// A parsed `<LocomotionGroup>` (`.lmg`): a parametric blend of clips.
#[derive(Debug, Clone, PartialEq)]
pub struct LocomotionGroup {
    pub blend_type: BlendType,
    /// `CAPS code` (e.g. `RUN_`, `WALK`, `IROT`).
    pub caps: String,
    pub examples: Vec<LmgExample>,
}

/// What an animation alias points at.
#[derive(Debug, Clone, PartialEq)]
pub enum AnimTarget {
    /// A single loose/packed `.caf`.
    Single(String),
    /// A LocomotionGroup blend space.
    Blend(LocomotionGroup),
}

/// Parameter values substituted into graph animation-name placeholders.
///
/// `[STANCE]`, `[ITEM]`, `[DIR]`, `[VERSION]`, … are matched
/// case-insensitively; unknown placeholders are left intact (and make the
/// name unresolvable).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnimInputs {
    entries: Vec<(String, String)>,
}

impl AnimInputs {
    pub fn new() -> Self {
        AnimInputs { entries: Vec::new() }
    }

    /// Builder-style setter.
    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.set(key, value);
        self
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        if let Some(slot) = self
            .entries
            .iter_mut()
            .find(|(k, _)| k.eq_ignore_ascii_case(&key))
        {
            slot.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    /// Replace every `[name]` with its value (case-insensitive). Names are
    /// ASCII, so byte indexing is safe.
    pub fn substitute(&self, name: &str) -> String {
        let bytes = name.as_bytes();
        let mut out = String::with_capacity(name.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'[' {
                if let Some(rel) = name[i + 1..].find(']') {
                    let key = &name[i + 1..i + 1 + rel];
                    if let Some(v) = self.get(key) {
                        out.push_str(v);
                        i = i + rel + 2;
                        continue;
                    }
                }
            }
            out.push(bytes[i] as char);
            i += 1;
        }
        out
    }
}

/// Blend metadata carried by [`ResolvedKind::Blend`].
#[derive(Debug, Clone, PartialEq)]
pub struct BlendInfo {
    pub blend_type: BlendType,
    pub caps: String,
    /// The example positions, parallel to [`Resolved::clips`].
    pub params: Vec<[f32; 3]>,
}

/// Whether a resolved name is one clip or a blend space.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedKind {
    Single,
    Blend(BlendInfo),
}

/// The result of resolving a graph animation name.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    /// `(clip_path, weight_or_param)`. For a blend the weight is the uniform
    /// `1/N` and the full blend positions live in [`BlendInfo::params`].
    pub clips: Vec<(String, f32)>,
    pub kind: ResolvedKind,
    /// The table alias (or literal clip stem) that produced this result.
    pub alias: String,
}

/// The character's animation-name table.
///
/// Built from the `.chrparams` animation list (`#filepath`, `$Include`,
/// `$TracksDatabase`, wildcards, `.lmg`) plus the `.dba`/`.img`/loose `.caf`
/// name spaces. Lookups are case-insensitive.
#[derive(Debug, Clone, Default)]
pub struct AnimNameTable {
    /// Lower-cased alias → target.
    aliases: HashMap<String, AnimTarget>,
    /// Lower-cased `.caf` stem → actual clip path (loose `.$caf`, `.caf`, or
    /// a `.dba`/`.img` path).
    clip_paths: HashMap<String, String>,
    /// Files whose lists were ingested (diagnostics).
    source: Vec<String>,
}

impl AnimNameTable {
    /// Load the animation list reachable from a character `.cdf`.
    ///
    /// Parses the `.cdf`, takes its base `.chr`, and reads the sibling
    /// `<stem>.chrparams`. Returns `None` when the `.cdf` or its `.chrparams`
    /// cannot be read.
    pub fn load_character(vfs: &Vfs, cdf_path: &str) -> Option<AnimNameTable> {
        let bytes = vfs.read(cdf_path)?;
        let cdf = cdf::parse(&bytes)?;
        let chrparams = chrparams_for(&cdf.model);
        Self::load_chrparams(vfs, &chrparams)
    }

    /// Load an animation list directly from a `.chrparams` path.
    pub fn load_chrparams(vfs: &Vfs, chrparams_path: &str) -> Option<AnimNameTable> {
        let mut loader = Loader {
            vfs,
            table: AnimNameTable::default(),
            seen: Vec::new(),
            cafs: None,
        };
        if !loader.ingest_chrparams(chrparams_path, "") {
            return None;
        }
        Some(loader.table)
    }

    /// Number of aliases registered.
    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }

    /// Number of distinct clip stems indexed.
    pub fn clip_count(&self) -> usize {
        self.clip_paths.len()
    }

    /// Files ingested, in order (diagnostics).
    pub fn sources(&self) -> &[String] {
        &self.source
    }

    /// All alias names (for tests/diagnostics).
    pub fn alias_names(&self) -> impl Iterator<Item = &str> {
        self.aliases.keys().map(|s| s.as_str())
    }

    /// Resolve a graph animation name with placeholder substitution, then the
    /// literal lookup, then the documented virtual-locomotion remap.
    pub fn resolve(&self, graph_anim_name: &str, inputs: &AnimInputs) -> Option<Resolved> {
        let substituted = inputs.substitute(graph_anim_name);
        if substituted.contains('[') {
            return None; // unknown placeholder still present
        }
        if let Some(r) = self.resolve_literal(&substituted) {
            return Some(r);
        }
        for cand in virtual_candidates(&substituted) {
            if let Some(r) = self.resolve_literal(&cand) {
                return Some(r);
            }
        }
        None
    }

    /// Resolve an already-concrete name: exact alias, then clip stem, then
    /// stem with a trailing `_<digits>` version stripped.
    pub fn resolve_literal(&self, name: &str) -> Option<Resolved> {
        let key = name.to_ascii_lowercase();
        if let Some(target) = self.aliases.get(&key) {
            return Some(self.materialise(name, target));
        }
        if let Some(path) = self.clip_paths.get(&key) {
            return Some(single(name, path.clone()));
        }
        let stripped = strip_version(&key);
        if stripped != key {
            if let Some(path) = self.clip_paths.get(stripped) {
                return Some(single(name, path.clone()));
            }
            if let Some(target) = self.aliases.get(stripped) {
                return Some(self.materialise(name, target));
            }
        }
        let bare = key.trim_start_matches('_');
        if bare != key {
            if let Some(path) = self.clip_paths.get(bare) {
                return Some(single(name, path.clone()));
            }
        }
        None
    }

    /// Turn a target into a [`Resolved`], expanding a blend's examples.
    fn materialise(&self, alias: &str, target: &AnimTarget) -> Resolved {
        match target {
            AnimTarget::Single(path) => single(alias, path.clone()),
            AnimTarget::Blend(lmg) => {
                let mut clips = Vec::new();
                let mut params = Vec::new();
                for ex in &lmg.examples {
                    // The example alias resolves through the same table (the
                    // `Test_*` wildcard entries), then by clip stem.
                    let path = self
                        .aliases
                        .get(&ex.alias.to_ascii_lowercase())
                        .and_then(|t| match t {
                            AnimTarget::Single(p) => Some(p.clone()),
                            AnimTarget::Blend(_) => None,
                        })
                        .or_else(|| {
                            self.clip_paths
                                .get(&ex.alias.to_ascii_lowercase())
                                .cloned()
                        })
                        .or_else(|| {
                            let lower = ex.alias.to_ascii_lowercase();
                            let k = strip_version(&lower);
                            self.clip_paths.get(k).cloned()
                        });
                    if let Some(p) = path {
                        clips.push((p, 1.0));
                        params.push(ex.position);
                    }
                }
                let n = clips.len().max(1) as f32;
                for c in &mut clips {
                    c.1 = 1.0 / n;
                }
                Resolved {
                    clips,
                    kind: ResolvedKind::Blend(BlendInfo {
                        blend_type: lmg.blend_type.clone(),
                        caps: lmg.caps.clone(),
                        params,
                    }),
                    alias: alias.to_string(),
                }
            }
        }
    }
}

fn single(alias: &str, path: String) -> Resolved {
    Resolved {
        clips: vec![(path, 1.0)],
        kind: ResolvedKind::Single,
        alias: alias.to_string(),
    }
}

/// Sibling `.chrparams` of a `.chr` (`…/skeleton.chr` → `…/skeleton.chrparams`).
pub fn chrparams_for(chr: &str) -> String {
    let norm = chr.replace('\\', "/");
    match norm.rsplit_once('.') {
        Some((stem, _)) => format!("{stem}.chrparams"),
        None => format!("{norm}.chrparams"),
    }
}

/// `foo_01` → `foo`, `foo_12` → `foo`; unchanged when no version suffix.
fn strip_version(name: &str) -> &str {
    let Some((head, tail)) = name.rsplit_once('_') else {
        return name;
    };
    if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) && head.len() >= 2 {
        head
    } else {
        name
    }
}

/// Candidate aliases for a virtual locomotion name.
///
/// Handles `_<STANCE>_<GAIT>STRAFE_<ITEM>_<VER>` and the transition spellings
/// (`idle2run`, `run2idle`, `idle2walk`, `walk2idle`). Candidates are
/// lower-case and are tried against the alias map in order.
fn virtual_candidates(name: &str) -> Vec<String> {
    let lower = name.to_ascii_lowercase();
    let trimmed = lower.trim_start_matches('_');
    let base = strip_version(trimmed);
    let toks: Vec<&str> = base.split('_').filter(|t| !t.is_empty()).collect();
    if toks.is_empty() {
        return Vec::new();
    }
    let stance = toks[0];
    let item = *toks.last().unwrap();

    // Which gait token? STRAFE spellings, then transition spellings.
    let mut gait: Option<&str> = None;
    for t in &toks {
        if t.len() > 6 && t.ends_with("strafe") {
            gait = Some(&t[..t.len() - 6]);
            break;
        }
    }
    if gait.is_none() {
        for t in &toks {
            for (needle, g) in [
                ("idle2run", "run"),
                ("run2idle", "run"),
                ("idle2walk", "walk"),
                ("walk2idle", "walk"),
                ("idle2sprint", "sprint"),
                ("sprint2idle", "sprint"),
            ] {
                if t.contains(needle) {
                    gait = Some(g);
                    break;
                }
            }
            if gait.is_some() {
                break;
            }
        }
    }
    let Some(gait) = gait else {
        return Vec::new();
    };

    let mut out: Vec<String> = Vec::new();
    let push = |s: String, out: &mut Vec<String>| {
        if !out.iter().any(|x| *x == s) {
            out.push(s);
        }
    };

    // 1) `_<gait>_<item>` with item fallbacks (the shipped player scheme).
    push(format!("_{gait}_{item}"), &mut out);
    for fallback in ["nw", "rifle", "injured"] {
        if fallback != item {
            push(format!("_{gait}_{fallback}"), &mut out);
        }
    }
    // 2) stance-prefixed spellings (freak/zombie chrparams: `_RELAXED_RUN_NW`).
    push(format!("_{stance}_{gait}_{item}"), &mut out);
    push(format!("_{stance}_{gait}_nw"), &mut out);
    // 3) gait-only aliases (`_CROUCH`, `_AIM_NW`).
    push(format!("_{gait}"), &mut out);
    out
}

/// Recursive `.chrparams` ingest accumulator.
struct Loader<'a> {
    vfs: &'a Vfs,
    table: AnimNameTable,
    /// Files already ingested (prevents include cycles).
    seen: Vec<String>,
    /// Lazily-built list of every `.caf`/`.$caf` path (for wildcards).
    cafs: Option<Vec<String>>,
}

impl<'a> Loader<'a> {
    /// Parse one `.chrparams` file, recursing into `$Include`s. `base` is the
    /// inherited `#filepath` (each file may override it). Returns `false`
    /// when the file could not be read.
    fn ingest_chrparams(&mut self, path: &str, inherited_base: &str) -> bool {
        let norm = path.replace('\\', "/");
        if self.seen.iter().any(|s| s.eq_ignore_ascii_case(&norm)) {
            return true;
        }
        let Some(bytes) = self.vfs.read(&norm) else {
            return false;
        };
        self.seen.push(norm.clone());
        self.table.source.push(norm.clone());

        let mut reader = Reader::from_reader(&bytes[..]);
        let mut buf = Vec::new();
        let mut in_list = false;
        let mut base = inherited_base.to_string();
        loop {
            let ev = match reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => break,
                Err(_) => break,
                Ok(ev) => ev,
            };
            match ev {
                Event::Start(e) | Event::Empty(e) => {
                    let local = e.local_name();
                    match local.as_ref() {
                        b"AnimationList" => in_list = true,
                        b"Animation" if in_list => {
                            self.ingest_animation(&e, &mut base);
                        }
                        _ => {}
                    }
                }
                Event::End(e) => {
                    if e.local_name().as_ref() == b"AnimationList" {
                        in_list = false;
                    }
                }
                _ => {}
            }
            buf.clear();
        }
        true
    }

    /// Handle one `<Animation …>` entry.
    fn ingest_animation(&mut self, e: &BytesStart, base: &mut String) {
        let Some(name) = get_attr(e, "name") else {
            return;
        };
        let Some(raw_path) = get_attr(e, "path") else {
            return;
        };
        let name = name.trim().to_string();
        let raw_path = raw_path.trim().replace('\\', "/");
        if name.is_empty() || raw_path.is_empty() {
            return;
        }

        match name.as_str() {
            "#filepath" => {
                *base = raw_path.trim_matches('/').to_string();
                return;
            }
            "$Include" => {
                // Includes are root-relative; fall back to base-relative.
                if !self.ingest_chrparams(&raw_path, base) {
                    let joined = join_base(base, &raw_path);
                    self.ingest_chrparams(&joined, base);
                }
                return;
            }
            "$TracksDatabase" => {
                let p = join_base(base, &raw_path);
                self.ingest_dba(&p);
                return;
            }
            "$AnimEventDatabase" | "$FaceLib" | "$AnimationDir" | "$AnimDir"
            | "$AnimationDirectory" | "$AnimDirectory" => return,
            _ => {}
        }

        // A regular alias. `path` may contain a wildcard.
        let full = join_base(base, &raw_path);
        if raw_path.contains('*') || raw_path.contains('?') {
            let matches = self.expand_wildcard(&full);
            for (stem, actual) in matches {
                self.table.clip_paths.insert(stem.to_ascii_lowercase(), actual.clone());
                let alias = if name.contains('*') {
                    name.replacen('*', &stem, 1)
                } else {
                    stem.clone()
                };
                self.table
                    .aliases
                    .insert(alias.to_ascii_lowercase(), AnimTarget::Single(actual));
            }
            return;
        }

        if full.to_ascii_lowercase().ends_with(".lmg") {
            if let Some(group) = self.load_lmg(&full) {
                self.table
                    .aliases
                    .insert(name.to_ascii_lowercase(), AnimTarget::Blend(group));
            }
            return;
        }

        // Single clip. Index its stem as well as the explicit alias.
        self.table
            .clip_paths
            .insert(clip_stem(&full), full.clone());
        self.table
            .aliases
            .insert(name.to_ascii_lowercase(), AnimTarget::Single(full));
    }

    /// Read a `.dba` and index every packed clip path by stem.
    fn ingest_dba(&mut self, path: &str) {
        let Some(bytes) = self.vfs.read(path) else {
            return;
        };
        let Some(dba) = dba::parse(&bytes) else {
            return;
        };
        for e in dba.entries() {
            self.table
                .clip_paths
                .insert(clip_stem(&e.name), e.name.clone());
            self.table
                .aliases
                .entry(e.name.to_ascii_lowercase())
                .or_insert_with(|| AnimTarget::Single(e.name.clone()));
        }
    }

    /// Expand a wildcard `path` against the mounted VFS (`.caf` and `.$caf`).
    /// Returns `(stem, actual_path)` pairs.
    fn expand_wildcard(&mut self, pattern: &str) -> Vec<(String, String)> {
        let pat = pattern.to_ascii_lowercase();
        let (prefix, suffix) = match pat.split_once('*') {
            Some((a, b)) => (a.to_string(), b.to_string()),
            None => (pat.clone(), String::new()),
        };
        let cafs = self
            .cafs
            .get_or_insert_with(|| collect_caf_paths(self.vfs));
        let mut out = Vec::new();
        for p in cafs.iter() {
            let norm = p.replace('\\', "/");
            let key = caf_key(&norm); // canonical `.caf` spelling, lower-case
            if !key.starts_with(&prefix) || !key.ends_with(&suffix) {
                continue;
            }
            let middle = &key[prefix.len()..key.len() - suffix.len()];
            if middle.contains('/') {
                continue;
            }
            let stem = clip_stem(&norm);
            // Prefer the `.$caf` spelling when both exist.
            if let Some(existing) = out
                .iter()
                .position(|(s, _): &(String, String)| *s == stem)
            {
                if p.to_ascii_lowercase().ends_with(".$caf") {
                    out[existing].1 = p.clone();
                }
            } else {
                out.push((stem, p.clone()));
            }
        }
        out
    }

    /// Parse a LocomotionGroup `.lmg`.
    fn load_lmg(&self, path: &str) -> Option<LocomotionGroup> {
        let bytes = self.vfs.read(path)?;
        parse_lmg(&bytes)
    }
}

/// Parse a LocomotionGroup `.lmg` document.
pub fn parse_lmg(xml: &[u8]) -> Option<LocomotionGroup> {
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut saw_root = false;
    let mut blend_type = BlendType::Other(String::new());
    let mut caps = String::new();
    let mut examples = Vec::new();
    loop {
        let ev = match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) => break,
            Err(_) => break,
            Ok(ev) => ev,
        };
        match ev {
            Event::Start(e) | Event::Empty(e) => {
                let local = e.local_name();
                match local.as_ref() {
                    b"LocomotionGroup" => saw_root = true,
                    b"BLENDTYPE" => {
                        if let Some(t) = get_attr(&e, "type") {
                            blend_type = BlendType::from_code(&t);
                        }
                    }
                    b"CAPS" => {
                        if let Some(c) = get_attr(&e, "code") {
                            caps = c.trim().to_string();
                        }
                    }
                    b"Example" | b"ExamplePseudo" | b"VExample" => {
                        if let Some(alias) = get_attr(&e, "AName") {
                            let position = get_attr(&e, "Position")
                                .map(|s| parse_vec3(&s))
                                .unwrap_or([0.0, 0.0, 0.0]);
                            examples.push(LmgExample {
                                alias: alias.trim().to_string(),
                                position,
                            });
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        buf.clear();
    }
    if !saw_root {
        return None;
    }
    Some(LocomotionGroup { blend_type, caps, examples })
}

/// `base` + `/` + `path`, unless `path` already carries the base prefix.
fn join_base(base: &str, path: &str) -> String {
    let base = base.trim_matches('/');
    let path = path.trim_matches('/');
    if base.is_empty() || path.to_ascii_lowercase().starts_with(&base.to_ascii_lowercase()) {
        path.to_string()
    } else {
        format!("{base}/{path}")
    }
}

/// Lower-case stem of a clip path (`…/hero_run_forward.$caf` →
/// `hero_run_forward`).
fn clip_stem(path: &str) -> String {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let base = base
        .strip_suffix(".$caf")
        .or_else(|| base.strip_suffix(".CAF"))
        .or_else(|| base.strip_suffix(".caf"))
        .unwrap_or(base);
    base.to_ascii_lowercase()
}

/// Canonical matching key: separators normalised, `.$caf` folded to `.caf`,
/// lower-cased.
fn caf_key(path: &str) -> String {
    path.replace('\\', "/")
        .to_ascii_lowercase()
        .replace(".$caf", ".caf")
}

/// Every `.caf`/`.$caf` path in the VFS, `.$caf` before `.caf`.
fn collect_caf_paths(vfs: &Vfs) -> Vec<String> {
    let mut out = vfs.paths_with_ext(".$caf");
    out.extend(vfs.paths_with_ext(".caf"));
    out
}

fn parse_vec3(s: &str) -> [f32; 3] {
    let v: Vec<f32> = s
        .split(',')
        .filter_map(|p| p.trim().parse::<f32>().ok())
        .collect();
    if v.len() >= 3 {
        [v[0], v[1], v[2]]
    } else if v.len() == 2 {
        [v[0], v[1], 0.0]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

/// Index the compact clip-header containers (`animations.img`,
/// `directionalblends.img`) into `table`, if present. Kept separate so tests
/// can opt in.
pub fn index_img(table: &mut AnimNameTable, vfs: &Vfs, path: &str) -> bool {
    let Some(bytes) = vfs.read(path) else {
        return false;
    };
    let Some(img) = cryimg::parse(&bytes) else {
        return false;
    };
    for e in img.entries() {
        if let Some(path) = &e.path {
            table
                .clip_paths
                .entry(clip_stem(path))
                .or_insert_with(|| path.clone());
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_is_case_insensitive_and_keeps_unknown() {
        let inputs = AnimInputs::new().with("stance", "relaxed").with("ITEM", "nw");
        assert_eq!(
            inputs.substitute("_[STANCE]_WALKSTRAFE_[ITEM]_01"),
            "_relaxed_WALKSTRAFE_nw_01"
        );
        assert_eq!(inputs.substitute("_[DIR]_"), "_[DIR]_");
    }

    #[test]
    fn strip_version_only_for_trailing_digits() {
        assert_eq!(strip_version("hero_run_forward_01"), "hero_run_forward");
        assert_eq!(strip_version("hero_run_forward"), "hero_run_forward");
        assert_eq!(strip_version("relaxed_idle_nw_12"), "relaxed_idle_nw");
        // A bare two-digit token is not stripped away.
        assert_eq!(strip_version("_01"), "_01");
    }

    #[test]
    fn virtual_candidates_cover_strafe_and_transitions() {
        let c = virtual_candidates("_RELAXED_RUNSTRAFE_NW_01");
        assert!(c.contains(&"_run_nw".to_string()), "got {c:?}");
        assert!(c.contains(&"_relaxed_run_nw".to_string()), "got {c:?}");

        let c = virtual_candidates("_RELAXED_WALKSTRAFE_NW_01");
        assert!(c.contains(&"_walk_nw".to_string()), "got {c:?}");

        let c = virtual_candidates("_[STANCE]_SPRINTSTRAFE_[ITEM]_01");
        assert!(c.contains(&"_sprint_nw".to_string()), "got {c:?}");

        let c = virtual_candidates("_RELAXED_IDLE2WALK_NW_01");
        assert!(c.contains(&"_walk_nw".to_string()), "got {c:?}");
    }

    #[test]
    fn parse_lmg_reads_blendtype_caps_and_examples() {
        let xml = br#"<LocomotionGroup>
  <BLENDTYPE type="FLR2"/>
  <CAPS code="WALK"/>
  <ExampleList>
    <Example Position="-1,1,0" AName="Test_hero_run_leanleft"/>
    <Example Position="0,-1,0" AName="Test_hero_walk_forward"/>
  </ExampleList>
</LocomotionGroup>"#;
        let g = parse_lmg(xml).expect("lmg parses");
        assert_eq!(g.blend_type, BlendType::Flr2);
        assert_eq!(g.caps, "WALK");
        assert_eq!(g.examples.len(), 2);
        assert_eq!(g.examples[1].alias, "Test_hero_walk_forward");
        assert_eq!(g.examples[1].position, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn junk_lmg_never_panics() {
        for junk in [&b""[..], b"not xml", &[0xff, 0xfe, 0x00]] {
            let _ = parse_lmg(junk);
        }
        assert!(parse_lmg(b"not xml").is_none());
    }

    #[test]
    fn chrparams_for_swaps_extension() {
        assert_eq!(
            chrparams_for("characters/male/human/skeleton/skeleton.chr"),
            "characters/male/human/skeleton/skeleton.chrparams"
        );
    }

    // -- real game data (skipped when the paks are absent) ------------------

    const DEFAULT_GAME: &str =
        "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";
    const PLAYER_CDF: &str = "characters/male/human/ul_hero_male.cdf";

    fn game_vfs() -> Option<Vfs> {
        let game = std::env::var_os("SOD_GAME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_GAME));
        if !game.is_dir() {
            return None;
        }
        let mut vfs = Vfs::new();
        let mut mounted = 0;
        for rel in ["Game/animations.pak", "Game/characters.pak"] {
            let p = game.join(rel);
            if p.is_file() && vfs.mount(&p).is_ok() {
                mounted += 1;
            }
        }
        (mounted == 2).then_some(vfs)
    }

    #[test]
    fn real_player_table_loads_aliases_and_lmgs() {
        let Some(vfs) = game_vfs() else { return };
        let table = AnimNameTable::load_character(&vfs, PLAYER_CDF).expect("player table");
        eprintln!(
            "PROBE player table: aliases={} clips={} sources={}",
            table.alias_count(),
            table.clip_count(),
            table.sources().len()
        );
        assert!(table.alias_count() > 50, "aliases={}", table.alias_count());
        assert!(table.clip_count() > 100, "clips={}", table.clip_count());

        // The locomotion-group aliases from basic_player.chrparams.
        for alias in ["_run_nw", "_walk_nw", "_sprint_nw"] {
            let r = table
                .resolve_literal(alias)
                .unwrap_or_else(|| panic!("{alias} should resolve"));
            match r.kind {
                ResolvedKind::Blend(_) => {}
                ResolvedKind::Single => panic!("{alias} should be a blend space"),
            }
            assert!(!r.clips.is_empty(), "{alias} has no example clips");
        }
    }

    #[test]
    fn real_relaxed_strafe_resolves_to_lmgs() {
        let Some(vfs) = game_vfs() else { return };
        let table = AnimNameTable::load_character(&vfs, PLAYER_CDF).expect("player table");

        let run = table
            .resolve("_RELAXED_RUNSTRAFE_NW_01", &AnimInputs::new())
            .expect("RelaxedRunNW resolves");
        let walk = table
            .resolve("_RELAXED_WALKSTRAFE_NW_01", &AnimInputs::new())
            .expect("RelaxedWalkNW resolves");

        assert!(run.alias.eq_ignore_ascii_case("_RUN_NW"), "alias {}", run.alias);
        assert!(walk.alias.eq_ignore_ascii_case("_WALK_NW"), "alias {}", walk.alias);
        assert!(matches!(run.kind, ResolvedKind::Blend(_)));
        assert!(matches!(walk.kind, ResolvedKind::Blend(_)));

        // The examples must land on the real forward clips.
        let run_paths: Vec<&str> = run.clips.iter().map(|(p, _)| p.as_str()).collect();
        let walk_paths: Vec<&str> = walk.clips.iter().map(|(p, _)| p.as_str()).collect();
        assert!(run_paths.iter().any(|p| p.contains("hero_run_forward")), "{run_paths:?}");
        assert!(walk_paths.iter().any(|p| p.contains("hero_walk_forward")), "{walk_paths:?}");
    }

    fn graph_vfs() -> Option<Vfs> {
        game_vfs()
    }

    /// Resolve every animation name referenced by `playerfullbody.xml` for a
    /// few stance/item combinations and report the resolve rate.
    #[test]
    fn real_playerfullbody_resolve_rate() {
        let Some(vfs) = graph_vfs() else { return };
        let Some(xml) = vfs.read("animations/graphs/playerfullbody.xml") else {
            return;
        };
        let graph = crate::ag_graph::parse(&xml).expect("playerfullbody parses");
        let table = AnimNameTable::load_character(&vfs, PLAYER_CDF).expect("player table");

        let combos = [
            ("relaxed", "nw"),
            ("combat", "rifle"),
            ("stealth", "nw"),
            ("crouch", "pistol"),
        ];
        for (stance, item) in combos {
            let inputs = AnimInputs::new()
                .with("stance", stance)
                .with("item", item)
                .with("dir", "fwd")
                .with("direction", "fwd")
                .with("version", "01")
                .with("side", "left")
                .with("stance2", stance)
                .with("mode", "normal")
                .with("type", "normal");
            // Three scopes: every reference; locomotion-looking references
            // (run/walk/sprint/strafe/crouch); and the STRAFE blend spaces.
            let mut all = (0usize, 0usize);
            let mut loco = (0usize, 0usize);
            let mut strafe = (0usize, 0usize);
            let mut unresolved: Vec<String> = Vec::new();
            let mut strafe_miss: Vec<String> = Vec::new();
            for st in &graph.states {
                let Some(a) = st.animation.as_deref() else { continue };
                if a.is_empty() {
                    continue;
                }
                let ok = table.resolve(a, &inputs).is_some();
                all.0 += 1;
                all.1 += ok as usize;
                let lc = a.to_ascii_lowercase();
                let is_loco = ["run", "walk", "sprint", "strafe", "crouch"]
                    .iter()
                    .any(|k| lc.contains(k));
                if is_loco {
                    loco.0 += 1;
                    loco.1 += ok as usize;
                }
                if lc.contains("strafe") {
                    strafe.0 += 1;
                    strafe.1 += ok as usize;
                    if !ok {
                        strafe_miss.push(a.to_string());
                    }
                }
                if !ok {
                    unresolved.push(a.to_string());
                }
            }
            let pct = |t: usize, r: usize| {
                if t == 0 {
                    0.0
                } else {
                    r as f64 * 100.0 / t as f64
                }
            };
            eprintln!(
                "PROBE playerfullbody {stance}/{item}: all {}/{} ({:.1}%), loco {}/{} ({:.1}%), strafe {}/{} ({:.1}%)",
                all.1, all.0, pct(all.0, all.1),
                loco.1, loco.0, pct(loco.0, loco.1),
                strafe.1, strafe.0, pct(strafe.0, strafe.1),
            );
            if !strafe_miss.is_empty() {
                eprintln!("PROBE strafe unresolved for {stance}/{item}: {strafe_miss:?}");
            }
            // Sanity: the locomotion states must always resolve.
            for state in ["RelaxedWalkNW", "RelaxedRunNW"] {
                if let Some(a) = graph.animation_for(state) {
                    assert!(
                        table.resolve(a, &inputs).is_some(),
                        "{state} ({a}) must resolve for {stance}/{item}"
                    );
                }
            }
            if stance == "relaxed" && item == "nw" {
                unresolved.sort();
                unresolved.dedup();
                eprintln!(
                    "PROBE relaxed/nw unresolved ({} distinct); first: {:?}",
                    unresolved.len(),
                    &unresolved[..unresolved.len().min(8)]
                );
            }
        }
    }

    /// Root-motion speeds of the resolved forward clips, via
    /// [`crate::locomotion`].
    #[test]
    fn real_relaxed_strafe_root_motion_speeds() {
        use crate::anim::Caf;
        use crate::locomotion;
        use crate::skin;

        let Some(vfs) = game_vfs() else { return };
        let table = AnimNameTable::load_character(&vfs, PLAYER_CDF).expect("player table");
        let skel = vfs
            .read("characters/male/human/skeleton/skeleton.chr")
            .and_then(|b| skin::parse_skeleton(&b))
            .expect("master skeleton");

        let speed_of = |path: &str| -> f32 {
            let caf = Caf::parse(&vfs.read(path).expect("read clip")).expect("parse clip");
            let rm = locomotion::root_motion(&caf, &skel, caf.start as f32, caf.end as f32);
            let d = caf.duration_secs().max(1e-6);
            (rm.translation[0] * rm.translation[0] + rm.translation[1] * rm.translation[1]).sqrt()
                / d
        };

        let run = table
            .resolve("_RELAXED_RUNSTRAFE_NW_01", &AnimInputs::new())
            .expect("run resolves");
        let walk = table
            .resolve("_RELAXED_WALKSTRAFE_NW_01", &AnimInputs::new())
            .expect("walk resolves");

        let run_fwd = run
            .clips
            .iter()
            .map(|(p, _)| p)
            .find(|p| p.contains("hero_run_forward"))
            .expect("run forward");
        let walk_fwd = walk
            .clips
            .iter()
            .map(|(p, _)| p)
            .find(|p| p.contains("hero_walk_forward"))
            .expect("walk forward");

        let rs = speed_of(run_fwd);
        let ws = speed_of(walk_fwd);
        eprintln!("PROBE root-motion speeds: walk={ws:.3} run={rs:.3}");
        assert!((ws - 1.66).abs() < 0.35, "walk speed {ws:.3} not near 1.66");
        assert!((rs - 4.0).abs() < 0.5, "run speed {rs:.3} not near 4.0");
    }
}
