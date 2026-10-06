//! Milestone-1 probe: mount the game's paks, run real game Lua, parse real assets.
//! No Wine, no Windows, no game executable.

use anyhow::Result;
use sod_core::{asset, crypak::Vfs, script::GameLua};
use std::path::{Path, PathBuf};
use std::rc::Rc;

const DEFAULT_GAME: &str =
    "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";

fn collect_paks(root: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_paks(&p, out);
            } else if p.extension().map(|x| x == "pak").unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

fn main() -> Result<()> {
    let game = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_GAME.to_string());
    let game = PathBuf::from(game);

    let mut paks = Vec::new();
    collect_paks(&game, &mut paks);
    paks.sort();

    let mut vfs = Vfs::new();
    for p in &paks {
        let n = vfs.mount(p)?;
        println!("mounted {:>6} entries  {}", n, p.file_name().unwrap().to_string_lossy());
    }
    println!("\ntotal: {} paks, {} entries", vfs.pak_count(), vfs.entry_count());

    let vfs = Rc::new(vfs);

    // --- Lua: run the real engine boot script -------------------------------
    println!("\n===== Lua =====");
    let lua = GameLua::new(vfs.clone()).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    lua.run("scripts/main.lua")?;
    lua.call_global0("OnInit").map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // --- Assets ------------------------------------------------------------
    // --- Mesh decode: any extra CLI args are model paths ------------------
    let models: Vec<String> = std::env::args().skip(2).collect();
    if !models.is_empty() {
        println!("\n===== Mesh decode =====");
        for p in &models {
            match vfs.read(p) {
                Some(b) => match sod_core::mesh::decode_model(&b) {
                    Some(m) => println!(
                        "{p}: {} meshes, {} verts, {} tris, bounds {:?}..{:?}",
                        m.meshes.len(),
                        m.vertex_count(),
                        m.triangle_count(),
                        m.bounds_min.map(|v| (v * 100.0).round() / 100.0),
                        m.bounds_max.map(|v| (v * 100.0).round() / 100.0),
                    ),
                    None => println!("{p}: (no decodable mesh)"),
                },
                None => println!("{p}: (not found in paks)"),
            }
        }
    }

    let samples = [
        "objects/default.cgf",
        "textures/wood/woodwendsdiffuse.dds",
        "textures/decals/shadows/square.mtl",
    ];
    println!("\n===== Assets =====");
    for p in samples {
        match vfs.read(p) {
            Some(b) => {
                println!("\n{p}  ({} bytes)", b.len());
                if let Some(d) = asset::parse_dds(&b) {
                    println!(
                        "  DDS  {}x{}  mips={}  depth={}  format={}",
                        d.width,
                        d.height,
                        d.mipmaps,
                        d.depth,
                        d.format_name()
                    );
                } else if let Some(c) = asset::parse_cry_chunk_file(&b) {
                    println!(
                        "  CryEngine chunk file: format v{:#x}, fileType {:#010x}, {} chunks",
                        c.format_version,
                        c.file_type,
                        c.num_chunks
                    );
                    for ch in c.chunks.iter().take(10) {
                        println!(
                            "    {:#010x}  ns {:#06x} kind {:#04x}  {:<16} off={:<6} size={:<6} id={}",
                            ch.chunk_type,
                            ch.namespace(),
                            ch.kind(),
                            asset::chunk_type_name(ch.kind()),
                            ch.file_offset,
                            ch.size,
                            ch.id
                        );
                    }
                } else if let Some(m) = asset::parse_mtl(&b) {
                    println!("  Material shader={} diffuse={}", m.shader, m.diffuse);
                    for (map, file) in &m.textures {
                        println!("    texture {map} -> {file}");
                    }
                } else {
                    println!("  (unrecognised format)");
                }
            }
            None => println!("\n{p}  (not found in paks)"),
        }
    }

    Ok(())
}
