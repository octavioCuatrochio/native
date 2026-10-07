//! Runtime port of the game's localisation table logic (workerC3c).
//!
//! The engine's `CLocalizeStringManager` (`0x00ee5060`) loads
//! `Game/Languages/%s.%s.btxt` (`0x014a5c63`); the `TXDB` loader is at
//! `0x014a5d40`.  Keys are stored only as CRC-32 hashes, so the label strings
//! are not recoverable from the compiled table alone.  The readable ids live in
//! `languages/embeddedstrings.xml` and `languages/text_ui_menus.xml`, and most
//! `@` design labels equal their English text up to case/space normalisation —
//! the same heuristic `tools/ui_convert/localization.py` uses offline.
//!
//! Scaleform resolves a text field whose value starts with `@` by looking the
//! remainder up in the manager; our host does the same before it hands a string
//! to the HUD movie.

use sod_core::crypak::Vfs;
use std::collections::HashMap;

fn norm(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let pat = format!("{name}=\"");
    let start = tag.find(&pat)? + pat.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

fn parse_embedded_strings(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut rest = text;
    while let Some(open) = rest.find("<EmbeddedText") {
        let after = &rest[open..];
        let Some(close) = after.find('>') else { break };
        let tag = &after[..close];
        if let (Some(id), Some(tx)) = (attr(tag, "Id"), attr(tag, "Text")) {
            out.insert(norm(id), unescape(tx));
        }
        rest = &after[close + 1..];
    }
    out
}

fn parse_ui_spreadsheet(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut rest = text;
    while let Some(open) = rest.find("<Row") {
        let after = &rest[open..];
        let Some(close) = after.find("</Row>") else { break };
        let row = &after[..close];
        let mut cells: Vec<String> = Vec::new();
        let mut inner = row;
        while let Some(c) = inner.find("<Data") {
            let a = &inner[c..];
            let Some(gt) = a.find('>') else { break };
            let Some(end) = a.find("</Data>") else { break };
            if gt < end {
                cells.push(unescape(a[gt + 1..end].trim()));
            }
            inner = &a[end + 7..];
        }
        cells.retain(|c| !c.is_empty());
        if cells.len() >= 2 {
            let key = &cells[0];
            if is_identifier(key) {
                let value = if cells.len() >= 3 { &cells[2] } else { &cells[1] };
                out.insert(norm(key), value.clone());
            }
        }
        rest = &after[close + 6..];
    }
    out
}

fn is_identifier(s: &str) -> bool {
    let mut ch = s.chars();
    match ch.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Value NUL-terminated strings from a compiled `TXDB` file.
/// Layout: `TXDB`, `u32 version`, `u32 flags`, `u32 count`, `count` hashes,
/// then `count` NUL-terminated values.
fn parse_btxt_values(data: &[u8]) -> Vec<String> {
    if data.len() < 16 || &data[..4] != b"TXDB" {
        return Vec::new();
    }
    let count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    if count > 0x8000 {
        return Vec::new();
    }
    let mut pos = 0x10 + 4 * count;
    let mut out = Vec::new();
    while pos < data.len() && out.len() < count {
        match data[pos..].iter().position(|&b| b == 0) {
            Some(nul) => {
                out.push(String::from_utf8_lossy(&data[pos..pos + nul]).into_owned());
                pos += nul + 1;
            }
            None => break,
        }
    }
    out
}

const ALIASES: &[(&str, &str)] = &[
    ("uiback", "Back"),
    ("uiapply", "Apply"),
    ("uidelete", "Delete"),
    ("uicancel", "Cancel"),
    ("uitooltip", "Tooltip"),
    ("menutitle", "Main Menu"),
    ("menusavingmessagesavingcontentdontturnconsoleoff", "Saving... don't turn off your console"),
];

/// Best-effort `@label -> text` mapping built from the mounted install.
pub struct Loc {
    exact: HashMap<String, String>,
    values: HashMap<String, String>,
}

impl Loc {
    pub fn load(vfs: &Vfs) -> Self {
        let mut exact = HashMap::new();
        if let Some(bytes) = vfs
            .read("languages/embeddedstrings.xml")
            .or_else(|| vfs.read("gamedata/languages/embeddedstrings.xml"))
        {
            let text = String::from_utf8_lossy(&bytes);
            exact.extend(parse_embedded_strings(&text));
        }
        if let Some(bytes) = vfs
            .read("languages/text_ui_menus.xml")
            .or_else(|| vfs.read("gamedata/languages/text_ui_menus.xml"))
        {
            let text = String::from_utf8_lossy(&bytes);
            for (k, v) in parse_ui_spreadsheet(&text) {
                exact.entry(k).or_insert(v);
            }
        }
        let mut values = HashMap::new();
        for name in ["languages/english.win.btxt", "gamedata/languages/english.win.btxt"] {
            if let Some(data) = vfs.read(name) {
                for v in parse_btxt_values(&data) {
                    values.entry(norm(&v)).or_insert(v);
                }
                break;
            }
        }
        Self { exact, values }
    }

    /// Resolve a bare label (no leading `@`), or return `None` unchanged.
    ///
    /// Only labels that look like localisation ids (an underscore, a space or an
    /// upper-case letter) are considered, so GFx member names (`text`,
    /// `hitbox`, …) survive untouched.
    pub fn lookup(&self, label: &str) -> Option<String> {
        let s = strip_tags(label).trim().to_string();
        if s.is_empty() || s.len() > 128 {
            return None;
        }
        if !s.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        let has_upper = s.chars().any(|c| c.is_ascii_uppercase());
        if !(has_upper || s.contains('_') || s.contains(' ')) {
            return None;
        }
        let n = norm(&s);
        let cands = if n.ends_with("tt") { vec![n.clone(), n[..n.len() - 2].to_string()] } else { vec![n] };
        for cand in cands {
            if let Some((_, v)) = ALIASES.iter().find(|(k, _)| *k == cand) {
                return Some((*v).to_string());
            }
            if let Some(value) = self.exact.get(&cand) {
                if value.contains('%') {
                    if let Some(plain) = self.values.get(&cand) {
                        return Some(plain.clone());
                    }
                }
                return Some(value.clone());
            }
            if let Some(value) = self.values.get(&cand) {
                return Some(value.clone());
            }
        }
        None
    }

    /// Resolve `@label` occurrences in a string, leaving unknown labels as-is.
    pub fn resolve(&self, text: &str) -> String {
        if !text.contains('@') {
            return text.to_string();
        }
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(at) = rest.find('@') {
            out.push_str(&rest[..at]);
            let after = &rest[at + 1..];
            let end = after
                .find(|c: char| c.is_whitespace() || c == '\n' || c == '@')
                .unwrap_or(after.len());
            let label = &after[..end];
            match self.lookup(label) {
                Some(v) => out.push_str(&v),
                None => {
                    out.push('@');
                    out.push_str(label);
                }
            }
            rest = &after[end..];
        }
        out.push_str(rest);
        out
    }
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0usize;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            c if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}
