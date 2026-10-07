//! sod-core: native (Wine-free) readers for State of Decay: YOSE CryEngine data.
//!
//! Milestone 1 scope:
//!   * `crypak`  - read CryEngine `.pak` archives (ZIP with backslash names + zlib).
//!   * `asset`   - recognise/parse CryEngine assets (`.cgf`, `.dds`, `.mtl`).
//!   * `script`  - run real game Lua (Lua 5.1) against a minimal engine API shim.

pub mod ag_cost;
pub mod ag_driver;
pub mod ag_graph;
pub mod ag_runtime;
pub mod ai;
pub mod anim;
pub mod anim_alias;
pub mod anim_graph;
pub mod anim_manifest;
pub mod architecture;
pub mod asset;
pub mod audio;
pub mod audio_fev;
pub mod audio_play;
pub mod cache;
pub mod camera;
pub mod cdf;
pub mod character;
pub mod class3_move;
pub mod class3_physics;
pub mod combat;
pub mod cryimg;
pub mod cry;
pub mod crypak;
pub mod dba;
pub mod entity;
pub mod game;
pub mod hints;
pub mod hud;
pub mod hud_data;
pub mod inventory;
pub mod locomotion;
pub mod mission;
pub mod nav;
pub mod npc_ai;
pub mod level;
pub mod level_lights;
pub mod material;
pub mod mesh;
pub mod parity;
pub mod physics;
pub mod player_locomotion;
pub mod script;
pub mod skin;
pub mod sm_payload;
pub mod sm_runtime;
pub mod sm_tables;
pub mod statemachine;
pub mod tuning;
pub mod surfaces;
pub mod tagpoints;
pub mod terrain;
pub mod tod;
pub mod vegetation;
pub mod view_system;
pub mod zombie_ai;

pub use crypak::{Pak, Vfs};
