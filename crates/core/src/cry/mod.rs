//! The CryEngine replacement layer: the services the game's Lua and data
//! expect from the engine, implemented natively. Game logic itself is never
//! re-written here; it runs from the original scripts loaded at runtime.

pub mod actor;
pub mod binds_audio;
pub mod ai;
pub mod ai_data;
pub mod ai_loader;
pub mod ai_runtime;
pub mod binds_entity;
pub mod binds_sound;
pub mod binds_system;
pub mod combat;
pub mod combat_perception;
pub mod content;
pub mod content_runtime;
mod cvar_table;
pub mod constants;
pub mod cvars;
pub mod events;
pub mod gamerules;
pub mod host;
pub mod mission;
pub mod navmesh;
pub mod phys_proxy;
pub mod phys_world;
pub mod render;
pub mod weapons;
pub mod weapons_stamina;
pub mod world;
pub mod zombie_actor;
pub mod zombie_runtime;
pub mod zombie_spawn;

pub use host::{Ctx, ScriptHost};
pub use world::{EntityData, EntityId, EntityWorld, WorldEvent};
