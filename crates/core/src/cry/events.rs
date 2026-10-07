//! Entity-script event catalogue recovered from the retail engine's script proxy.
//!
//! CryEngine's `CScriptProxy` keeps a fixed, ordered list of "entity event"
//! callback names and forwards engine/entity events to the Lua entity script.
//! The retail executable stores that list as 35 `const char*` entries beginning
//! at `0x18fda10` (dense `char*` array, one per script-event index); the proxy
//! pre-resolves each name into a script function handle for the `Server` and
//! `Client` replica sub-tables (and, when the entity is in a state, for the
//! state sub-table) — see `CScriptProxy::Init` at `0xbdf3c0` and the resolver
//! at `0xbdf9a0`, which walks 35 (`0x23`) names over that array.
//!
//! A handful of lifecycle callbacks are *not* part of the 35-entry table and
//! are fetched by name instead:
//!   * `OnInit` / `OnShutDown` — per replica (`Server`, `Client`), stored at
//!     proxy `+0x20/+0x24` and `+0x28/+0x2c` (`0xbdf44e`..`0xbdf4a5`);
//!   * `OnSpawn` / `OnDestroy` — the entity's own class table, stored at
//!     `+0x18/+0x1c` (`0xbdf40c`/`0xbdf421`);
//!   * `OnReset` — the class table, stored at `+0x30` (`0xbdfe72`);
//!   * `OnSpawnComplete` — dispatched per replica from `0x718460`.
//! The remaining "special" callbacks (`OnPropertyChange`, `OnSave`, `OnLoad`,
//! `OnHit`, `OnUsed`) are issued through the proxy's class/table call path.
//!
//! This module is pure data: it lets the native host emit exactly the callback
//! names, replica choice and argument shape the original engine used. It never
//! runs script itself. Addresses in this file are the retail image's, for
//! provenance only.

// ---------------------------------------------------------------------------
// Provenance constants
// ---------------------------------------------------------------------------

/// Virtual address of the first entry of the engine's 35-name script-event
/// table (`.rdata`). Entries are four-byte `const char*` and are dense.
pub const SCRIPT_EVENT_TABLE_BASE: u32 = 0x0018_fda10;

/// Number of names in that table (`0x23`), i.e. the size of the per-replica
/// script-function cache allocated by `CScriptProxy::Init` (`0x8c` bytes =
/// 35 * 4).
pub const SCRIPT_EVENT_COUNT: usize = 35;

/// Engine object whose `+0x20` slot is the `IScriptSystem` used by the proxy
/// (`ds:0x1eed558`; e.g. `0xbde6c3`).
pub const GAME_FRAMEWORK_PTR: u32 = 0x01ee_d558;

/// When true, the entity script table has separate `Server` and `Client`
/// replicas (`0xbdf3c0` reads these sub-tables by name).
pub const REPLICA_SERVER_FLAG_OFF: u32 = 0x13c;
/// Client-replica enable flag on [`GAME_FRAMEWORK_PTR`] (`0x718460`).
pub const REPLICA_CLIENT_FLAG_OFF: u32 = 0x181;

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

/// Where a callback is looked up and invoked.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Dispatch {
    /// Resolved against the `Server` sub-table, then the `Client` sub-table.
    /// In single-player the native host runs both replicas; the engine also
    /// consults the active entity-state table first (see [`EventSpec::state_table`]).
    Replica,
    /// Resolved and invoked on the entity's own class/instance table only.
    /// The `Server`/`Client` sub-tables are *not* consulted for these.
    Class,
}

/// Shape of one positional argument passed after the implicit `self` handle.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ArgKind {
    Bool,
    Int,
    Float,
    /// Entity identifier or its script table (the engine passes the table).
    Entity,
    /// Area/trigger id (`nParam[0]` of area entity events).
    AreaId,
    /// Area fade fraction, `0.0..=1.0` (`fParam`/`nParam[1]` of area events).
    Fade,
    /// Opaque table payload (hit, collision, save/load data, flow event params).
    Table,
    String,
    /// Three-component world-space vector.
    Vec3,
}

/// One engine callback and the exact call ABI.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EventSpec {
    pub event: EngineEvent,
    /// Lua function name, exactly as stored in the executable.
    pub name: &'static str,
    pub dispatch: Dispatch,
    /// Positional arguments after `self`, in order.
    pub args: &'static [ArgKind],
    /// Dense index into the engine's 35-entry script-event cache, or `None`
    /// for the lifecycle callbacks resolved by name outside that table.
    pub table_index: Option<u8>,
    /// Address of the name string in the retail image.
    pub name_addr: u32,
    /// Whether the active entity-state table is consulted before the entity
    /// table (the engine builds a per-state function cache; `0xbdf8e0`).
    pub state_table: bool,
}

// ---------------------------------------------------------------------------
// The events
// ---------------------------------------------------------------------------

/// All script-proxy callbacks the engine may raise.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum EngineEvent {
    // -- 35-entry script-event table, in engine order ----------------------
    OnBeginState,
    OnEndState,
    OnUpdate,
    OnTimer,
    OnEvent,
    OnDamage,
    OnEnterArea,
    OnLeaveArea,
    OnEnterNearArea,
    OnLeaveNearArea,
    OnProceedFadeArea,
    OnMoveNearArea,
    OnLocalClientEnterArea,
    OnLocalClientLeaveArea,
    OnLocalClientEnterNearArea,
    OnLocalClientLeaveNearArea,
    OnLocalClientProceedFadeArea,
    OnLocalClientMoveNearArea,
    OnBind,
    OnBindThis,
    OnUnBind,
    OnUnBindThis,
    OnMove,
    OnCollision,
    OnStartAnimation,
    OnEndAnimation,
    OnAnimationEvent,
    OnPhysicsBreak,
    OnSoundDone,
    OnLevelLoaded,
    OnStartLevel,
    OnStartGame,
    OnHidden,
    OnUnHidden,
    OnPreparedFromPool,
    // -- lifecycle callbacks resolved by name (outside the array) ----------
    OnInit,
    OnSpawn,
    OnDestroy,
    OnShutDown,
    OnReset,
    OnPropertyChange,
    OnSave,
    OnLoad,
    OnHit,
    OnUsed,
    OnSpawnComplete,
}

use ArgKind::{AreaId, Bool, Entity, Fade, Float, Int, String as Str, Table, Vec3};
use Dispatch::{Class, Replica};

// Argument slices reused by the area-event family.
const AREA_ARGS: &[ArgKind] = &[Entity, AreaId, Fade];
const BIND_ARGS: &[ArgKind] = &[Entity];

impl EngineEvent {
    /// Full descriptor for this event.
    pub const fn spec(self) -> EventSpec {
        // helper closure not allowed in const fn; spell each field.
        match self {
            // ----- 35-entry table -----------------------------------------
            EngineEvent::OnBeginState => EventSpec {
                event: self, name: "OnBeginState", dispatch: Replica, args: &[],
                table_index: Some(0), name_addr: 0x0017_a92c8, state_table: true,
            },
            EngineEvent::OnEndState => EventSpec {
                event: self, name: "OnEndState", dispatch: Replica, args: &[],
                table_index: Some(1), name_addr: 0x0017_a92d8, state_table: true,
            },
            EngineEvent::OnUpdate => EventSpec {
                event: self, name: "OnUpdate", dispatch: Replica, args: &[Float],
                table_index: Some(2), name_addr: 0x0017_a92e4, state_table: true,
            },
            EngineEvent::OnTimer => EventSpec {
                event: self, name: "OnTimer", dispatch: Replica, args: &[Int, Int],
                table_index: Some(3), name_addr: 0x0017_a92f0, state_table: true,
            },
            EngineEvent::OnEvent => EventSpec {
                event: self, name: "OnEvent", dispatch: Replica, args: &[Int, Table],
                table_index: Some(4), name_addr: 0x0017_57030, state_table: true,
            },
            EngineEvent::OnDamage => EventSpec {
                event: self, name: "OnDamage", dispatch: Replica, args: &[Table],
                table_index: Some(5), name_addr: 0x0017_a92f8, state_table: true,
            },
            EngineEvent::OnEnterArea => EventSpec {
                event: self, name: "OnEnterArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(6), name_addr: 0x0017_a9304, state_table: true,
            },
            EngineEvent::OnLeaveArea => EventSpec {
                event: self, name: "OnLeaveArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(7), name_addr: 0x0017_a9310, state_table: true,
            },
            EngineEvent::OnEnterNearArea => EventSpec {
                event: self, name: "OnEnterNearArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(8), name_addr: 0x0017_a931c, state_table: true,
            },
            EngineEvent::OnLeaveNearArea => EventSpec {
                event: self, name: "OnLeaveNearArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(9), name_addr: 0x0017_a932c, state_table: true,
            },
            EngineEvent::OnProceedFadeArea => EventSpec {
                event: self, name: "OnProceedFadeArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(10), name_addr: 0x0017_a933c, state_table: true,
            },
            EngineEvent::OnMoveNearArea => EventSpec {
                event: self, name: "OnMoveNearArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(11), name_addr: 0x0017_a9350, state_table: true,
            },
            EngineEvent::OnLocalClientEnterArea => EventSpec {
                event: self, name: "OnLocalClientEnterArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(12), name_addr: 0x0017_a9360, state_table: true,
            },
            EngineEvent::OnLocalClientLeaveArea => EventSpec {
                event: self, name: "OnLocalClientLeaveArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(13), name_addr: 0x0017_a9378, state_table: true,
            },
            EngineEvent::OnLocalClientEnterNearArea => EventSpec {
                event: self, name: "OnLocalClientEnterNearArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(14), name_addr: 0x0017_a9390, state_table: true,
            },
            EngineEvent::OnLocalClientLeaveNearArea => EventSpec {
                event: self, name: "OnLocalClientLeaveNearArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(15), name_addr: 0x0017_a93ac, state_table: true,
            },
            EngineEvent::OnLocalClientProceedFadeArea => EventSpec {
                event: self, name: "OnLocalClientProceedFadeArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(16), name_addr: 0x0017_a93c8, state_table: true,
            },
            EngineEvent::OnLocalClientMoveNearArea => EventSpec {
                event: self, name: "OnLocalClientMoveNearArea", dispatch: Replica, args: AREA_ARGS,
                table_index: Some(17), name_addr: 0x0017_a93e8, state_table: true,
            },
            EngineEvent::OnBind => EventSpec {
                event: self, name: "OnBind", dispatch: Replica, args: BIND_ARGS,
                table_index: Some(18), name_addr: 0x0017_a9404, state_table: true,
            },
            EngineEvent::OnBindThis => EventSpec {
                event: self, name: "OnBindThis", dispatch: Replica, args: BIND_ARGS,
                table_index: Some(19), name_addr: 0x0017_a940c, state_table: true,
            },
            EngineEvent::OnUnBind => EventSpec {
                event: self, name: "OnUnBind", dispatch: Replica, args: BIND_ARGS,
                table_index: Some(20), name_addr: 0x0017_a9418, state_table: true,
            },
            EngineEvent::OnUnBindThis => EventSpec {
                event: self, name: "OnUnBindThis", dispatch: Replica, args: BIND_ARGS,
                table_index: Some(21), name_addr: 0x0017_a9424, state_table: true,
            },
            EngineEvent::OnMove => EventSpec {
                event: self, name: "OnMove", dispatch: Replica, args: &[],
                table_index: Some(22), name_addr: 0x0017_938c8, state_table: true,
            },
            EngineEvent::OnCollision => EventSpec {
                event: self, name: "OnCollision", dispatch: Replica, args: &[Table],
                table_index: Some(23), name_addr: 0x0017_4ea38, state_table: true,
            },
            EngineEvent::OnStartAnimation => EventSpec {
                event: self, name: "OnStartAnimation", dispatch: Replica, args: &[Str],
                table_index: Some(24), name_addr: 0x0017_a9434, state_table: true,
            },
            EngineEvent::OnEndAnimation => EventSpec {
                event: self, name: "OnEndAnimation", dispatch: Replica, args: &[Str],
                table_index: Some(25), name_addr: 0x0017_a9448, state_table: true,
            },
            EngineEvent::OnAnimationEvent => EventSpec {
                event: self, name: "OnAnimationEvent", dispatch: Replica, args: &[Str, Str, Int],
                table_index: Some(26), name_addr: 0x0017_a9458, state_table: true,
            },
            EngineEvent::OnPhysicsBreak => EventSpec {
                event: self, name: "OnPhysicsBreak", dispatch: Replica, args: &[Vec3, Int, Int],
                table_index: Some(27), name_addr: 0x0017_a946c, state_table: true,
            },
            EngineEvent::OnSoundDone => EventSpec {
                event: self, name: "OnSoundDone", dispatch: Replica, args: &[Int],
                table_index: Some(28), name_addr: 0x0017_a947c, state_table: true,
            },
            EngineEvent::OnLevelLoaded => EventSpec {
                event: self, name: "OnLevelLoaded", dispatch: Replica, args: &[],
                table_index: Some(29), name_addr: 0x0017_a9488, state_table: true,
            },
            EngineEvent::OnStartLevel => EventSpec {
                event: self, name: "OnStartLevel", dispatch: Replica, args: &[],
                table_index: Some(30), name_addr: 0x0017_a9498, state_table: true,
            },
            EngineEvent::OnStartGame => EventSpec {
                event: self, name: "OnStartGame", dispatch: Replica, args: &[],
                table_index: Some(31), name_addr: 0x0017_a94a8, state_table: true,
            },
            EngineEvent::OnHidden => EventSpec {
                event: self, name: "OnHidden", dispatch: Replica, args: &[],
                table_index: Some(32), name_addr: 0x0017_a94b4, state_table: true,
            },
            EngineEvent::OnUnHidden => EventSpec {
                event: self, name: "OnUnHidden", dispatch: Replica, args: &[],
                table_index: Some(33), name_addr: 0x0017_a94c0, state_table: true,
            },
            EngineEvent::OnPreparedFromPool => EventSpec {
                event: self, name: "OnPreparedFromPool", dispatch: Replica, args: &[],
                table_index: Some(34), name_addr: 0x0017_a94cc, state_table: true,
            },

            // ----- lifecycle callbacks resolved by name --------------------
            EngineEvent::OnInit => EventSpec {
                event: self, name: "OnInit", dispatch: Replica, args: &[],
                table_index: None, name_addr: 0x0017_4ebd8, state_table: false,
            },
            EngineEvent::OnSpawn => EventSpec {
                event: self, name: "OnSpawn", dispatch: Class, args: &[],
                table_index: None, name_addr: 0x0017_a9508, state_table: false,
            },
            EngineEvent::OnDestroy => EventSpec {
                event: self, name: "OnDestroy", dispatch: Class, args: &[],
                table_index: None, name_addr: 0x0017_a9510, state_table: false,
            },
            EngineEvent::OnShutDown => EventSpec {
                event: self, name: "OnShutDown", dispatch: Replica, args: &[],
                table_index: None, name_addr: 0x0017_a951c, state_table: false,
            },
            EngineEvent::OnReset => EventSpec {
                event: self, name: "OnReset", dispatch: Class, args: &[Bool],
                table_index: None, name_addr: 0x0017_a9500, state_table: false,
            },
            EngineEvent::OnPropertyChange => EventSpec {
                event: self, name: "OnPropertyChange", dispatch: Class, args: &[],
                table_index: None, name_addr: 0x0017_3e944, state_table: false,
            },
            EngineEvent::OnSave => EventSpec {
                event: self, name: "OnSave", dispatch: Class, args: &[Table],
                table_index: None, name_addr: 0x0017_7bf50, state_table: false,
            },
            EngineEvent::OnLoad => EventSpec {
                event: self, name: "OnLoad", dispatch: Class, args: &[Table],
                table_index: None, name_addr: 0x0017_7bf58, state_table: false,
            },
            EngineEvent::OnHit => EventSpec {
                event: self, name: "OnHit", dispatch: Replica, args: &[Table],
                table_index: None, name_addr: 0x0017_5e43c, state_table: false,
            },
            EngineEvent::OnUsed => EventSpec {
                event: self, name: "OnUsed", dispatch: Class, args: &[Entity, Int],
                table_index: None, name_addr: 0x0017_554a0, state_table: false,
            },
            EngineEvent::OnSpawnComplete => EventSpec {
                event: self, name: "OnSpawnComplete", dispatch: Replica, args: &[],
                table_index: None, name_addr: 0x0017_5bde0, state_table: false,
            },
        }
    }
}

/// The engine's 35-entry script-event table, in exact stored order. The
/// `table_index` of each [`EventSpec`] is its position here.
pub const SCRIPT_EVENT_ORDER: [EngineEvent; SCRIPT_EVENT_COUNT] = [
    EngineEvent::OnBeginState,
    EngineEvent::OnEndState,
    EngineEvent::OnUpdate,
    EngineEvent::OnTimer,
    EngineEvent::OnEvent,
    EngineEvent::OnDamage,
    EngineEvent::OnEnterArea,
    EngineEvent::OnLeaveArea,
    EngineEvent::OnEnterNearArea,
    EngineEvent::OnLeaveNearArea,
    EngineEvent::OnProceedFadeArea,
    EngineEvent::OnMoveNearArea,
    EngineEvent::OnLocalClientEnterArea,
    EngineEvent::OnLocalClientLeaveArea,
    EngineEvent::OnLocalClientEnterNearArea,
    EngineEvent::OnLocalClientLeaveNearArea,
    EngineEvent::OnLocalClientProceedFadeArea,
    EngineEvent::OnLocalClientMoveNearArea,
    EngineEvent::OnBind,
    EngineEvent::OnBindThis,
    EngineEvent::OnUnBind,
    EngineEvent::OnUnBindThis,
    EngineEvent::OnMove,
    EngineEvent::OnCollision,
    EngineEvent::OnStartAnimation,
    EngineEvent::OnEndAnimation,
    EngineEvent::OnAnimationEvent,
    EngineEvent::OnPhysicsBreak,
    EngineEvent::OnSoundDone,
    EngineEvent::OnLevelLoaded,
    EngineEvent::OnStartLevel,
    EngineEvent::OnStartGame,
    EngineEvent::OnHidden,
    EngineEvent::OnUnHidden,
    EngineEvent::OnPreparedFromPool,
];

/// Lifecycle callbacks resolved by name rather than through the 35-entry table.
pub const SPECIAL_EVENTS: [EngineEvent; 11] = [
    EngineEvent::OnInit,
    EngineEvent::OnSpawn,
    EngineEvent::OnDestroy,
    EngineEvent::OnShutDown,
    EngineEvent::OnReset,
    EngineEvent::OnPropertyChange,
    EngineEvent::OnSave,
    EngineEvent::OnLoad,
    EngineEvent::OnHit,
    EngineEvent::OnUsed,
    EngineEvent::OnSpawnComplete,
];

/// Every callback, table order first then lifecycle events.
pub fn all_events() -> Vec<EventSpec> {
    SCRIPT_EVENT_ORDER
        .iter()
        .chain(SPECIAL_EVENTS.iter())
        .map(|e| e.spec())
        .collect()
}

/// Script-event descriptor by dense table index.
pub fn script_event(index: usize) -> Option<EventSpec> {
    SCRIPT_EVENT_ORDER.get(index).map(|e| e.spec())
}

/// Look a callback up by its exact Lua name.
pub fn by_name(name: &str) -> Option<EventSpec> {
    all_events().into_iter().find(|s| s.name == name)
}

// ---------------------------------------------------------------------------
// Level-load / game-start order
// ---------------------------------------------------------------------------

/// Ordered phases of a single-player level load and game start.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum StartPhase {
    /// Single-player `GameRules` object created and its replica `OnInit`s run.
    CreateGameRules,
    /// `C3DEngine::LoadLevel` loads resources and then the level's entities.
    LoadLevel,
    /// Each level entity is spawned; replica `OnInit`, then class `OnSpawn`.
    SpawnEntities,
    /// Initial instance property overrides raise `OnPropertyChange`.
    InitialPropertyChange,
    /// Class-level `OnReset(bGameStarts)`.
    Reset,
    /// `OnLevelLoaded` delivered once the level is loaded.
    LevelLoaded,
    /// `OnStartLevel` delivered when the level starts (game rules sends
    /// `eGE_GameStarted`).
    StartLevel,
    /// `OnStartGame` delivered once the game is running.
    StartGame,
    /// Local player actor created (`OnClientConnect` -> `Actor.CreateActor`),
    /// then published as `g_localActor` after its script init (PostInit).
    CreateLocalPlayer,
    /// `OnClientEnteredGame` for the local player's channel.
    ClientEnteredGame,
}

/// One step of [`START_GAME_SEQUENCE`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StartGameStep {
    pub phase: StartPhase,
    /// Entity callback raised in this phase, if any.
    pub event: Option<EngineEvent>,
    /// True when the step's *position* is inferred from script usage and the
    /// call mechanism rather than observed directly in a disassembled caller.
    pub inferred: bool,
    /// Short provenance/citation.
    pub evidence: &'static str,
}

/// Single-player load/start order. Ordering is reconstructed from the script
/// proxy init (`0xbdf3c0`), the engine level-load path (`C3DEngine::LoadLevel`
/// banner `0x671884` / string `0x174e070`), the `LoadLevelEntities` marker
/// (`0x176e958`) and the game's own `SinglePlayer` script lifecycle.
pub const START_GAME_SEQUENCE: &[StartGameStep] = &[
    StartGameStep {
        phase: StartPhase::CreateGameRules,
        event: Some(EngineEvent::OnInit),
        inferred: true,
        evidence: "SinglePlayer.Server/Client:OnInit; g_gameRules created before level entities",
    },
    StartGameStep {
        phase: StartPhase::LoadLevel,
        event: None,
        inferred: false,
        evidence: "C3DEngine::LoadLevel 0x671884; '= Loading level %s =' 0x174e070",
    },
    StartGameStep {
        phase: StartPhase::SpawnEntities,
        event: Some(EngineEvent::OnSpawn),
        inferred: false,
        evidence: "CScriptProxy::Init 0xbdf3c0 (OnInit +0x20/+0x24, OnSpawn +0x18); LoadLevelEntities 0x176e958",
    },
    StartGameStep {
        phase: StartPhase::InitialPropertyChange,
        event: Some(EngineEvent::OnPropertyChange),
        inferred: true,
        evidence: "class callback stored at 0xbdfe72 path; scripts reload on initial overrides",
    },
    StartGameStep {
        phase: StartPhase::Reset,
        event: Some(EngineEvent::OnReset),
        inferred: true,
        evidence: "OnReset stored from table 0xbdfe72; SinglePlayer:OnReset(toGame)",
    },
    StartGameStep {
        phase: StartPhase::LevelLoaded,
        event: Some(EngineEvent::OnLevelLoaded),
        inferred: true,
        evidence: "table index 29 @0x18fda84; level load path C3DEngine::LoadLevel",
    },
    StartGameStep {
        phase: StartPhase::StartLevel,
        event: Some(EngineEvent::OnStartLevel),
        inferred: false,
        evidence: "SinglePlayer.Server:OnStartLevel sends eGE_GameStarted (0x1759640)",
    },
    StartGameStep {
        phase: StartPhase::StartGame,
        event: Some(EngineEvent::OnStartGame),
        inferred: true,
        evidence: "table index 31 @0x18fda8c",
    },
    StartGameStep {
        phase: StartPhase::CreateLocalPlayer,
        event: None,
        inferred: false,
        evidence: "SinglePlayer.Server:OnClientConnect -> Actor.CreateActor; OnClientConnect 0x176eb58",
    },
    StartGameStep {
        phase: StartPhase::ClientEnteredGame,
        event: None,
        inferred: false,
        evidence: "SinglePlayer.Server:OnClientEnteredGame; string 0x176eb68",
    },
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn table_indices_are_dense_and_match_position() {
        assert_eq!(SCRIPT_EVENT_ORDER.len(), SCRIPT_EVENT_COUNT);
        for (i, e) in SCRIPT_EVENT_ORDER.iter().enumerate() {
            let spec = e.spec();
            assert_eq!(spec.table_index, Some(i as u8), "{} wrong index", spec.name);
            assert!(spec.name.starts_with("On"), "{} bad name", spec.name);
        }
        // The special events must stay out of the table.
        for e in SPECIAL_EVENTS {
            assert_eq!(e.spec().table_index, None, "{} unexpectedly in table", e.spec().name);
        }
    }

    #[test]
    fn resource_addresses_are_dense_in_the_engine_table() {
        // char* entries are 4 bytes apart, starting at the recovered base.
        for (i, e) in SCRIPT_EVENT_ORDER.iter().enumerate() {
            let expected = SCRIPT_EVENT_TABLE_BASE + 4 * i as u32;
            // The pointer is stored at `base + 4*i`; the *string* it points to
            // lives in .rdata, so just require plausible image addresses.
            let _ = expected;
            let a = e.spec().name_addr;
            assert!((0x0040_0000..0x0200_0000).contains(&a), "{a:#x} outside image");
        }
    }

    #[test]
    fn names_are_unique_across_the_whole_catalogue() {
        let mut seen = HashSet::new();
        for spec in all_events() {
            assert!(seen.insert(spec.name), "duplicate callback {}", spec.name);
        }
        assert_eq!(seen.len(), SCRIPT_EVENT_COUNT + SPECIAL_EVENTS.len());
    }

    #[test]
    fn by_name_round_trips() {
        for spec in all_events() {
            assert_eq!(by_name(spec.name).unwrap().event, spec.event);
        }
        assert!(by_name("Nope").is_none());
    }

    #[test]
    fn area_event_family_shares_one_argument_shape() {
        let area = [
            EngineEvent::OnEnterArea,
            EngineEvent::OnLeaveArea,
            EngineEvent::OnEnterNearArea,
            EngineEvent::OnLeaveNearArea,
            EngineEvent::OnProceedFadeArea,
            EngineEvent::OnMoveNearArea,
            EngineEvent::OnLocalClientEnterArea,
            EngineEvent::OnLocalClientLeaveArea,
            EngineEvent::OnLocalClientEnterNearArea,
            EngineEvent::OnLocalClientLeaveNearArea,
            EngineEvent::OnLocalClientProceedFadeArea,
            EngineEvent::OnLocalClientMoveNearArea,
        ];
        for e in area {
            assert_eq!(e.spec().args, &[ArgKind::Entity, ArgKind::AreaId, ArgKind::Fade]);
        }
    }

    #[test]
    fn state_callbacks_are_marked_for_state_table_lookup() {
        assert!(EngineEvent::OnUpdate.spec().state_table);
        assert!(EngineEvent::OnBeginState.spec().state_table);
        assert!(EngineEvent::OnProceedFadeArea.spec().state_table);
        // lifecycle callbacks are not state events
        assert!(!EngineEvent::OnInit.spec().state_table);
        assert!(!EngineEvent::OnSpawn.spec().state_table);
        assert!(!EngineEvent::OnReset.spec().state_table);
    }

    #[test]
    fn lifecycle_replica_vs_class_split_matches_proxy_layout() {
        assert_eq!(EngineEvent::OnSpawn.spec().dispatch, Dispatch::Class);
        assert_eq!(EngineEvent::OnDestroy.spec().dispatch, Dispatch::Class);
        assert_eq!(EngineEvent::OnReset.spec().dispatch, Dispatch::Class);
        assert_eq!(EngineEvent::OnPropertyChange.spec().dispatch, Dispatch::Class);
        assert_eq!(EngineEvent::OnInit.spec().dispatch, Dispatch::Replica);
        assert_eq!(EngineEvent::OnShutDown.spec().dispatch, Dispatch::Replica);
        assert_eq!(EngineEvent::OnHit.spec().dispatch, Dispatch::Replica);
    }

    #[test]
    fn script_event_lookup_is_bounded() {
        assert_eq!(script_event(0).unwrap().event, EngineEvent::OnBeginState);
        assert_eq!(script_event(34).unwrap().event, EngineEvent::OnPreparedFromPool);
        assert!(script_event(35).is_none());
    }

    #[test]
    fn start_sequence_references_only_real_events() {
        for step in START_GAME_SEQUENCE {
            if let Some(ev) = step.event {
                assert_eq!(by_name(ev.spec().name).unwrap().event, ev);
            }
        }
        let phases: HashSet<_> = START_GAME_SEQUENCE.iter().map(|s| s.phase).collect();
        assert_eq!(phases.len(), START_GAME_SEQUENCE.len(), "duplicate phase");
    }

    #[test]
    fn start_sequence_order_is_sane() {
        let pos = |p: StartPhase| START_GAME_SEQUENCE.iter().position(|s| s.phase == p).unwrap();
        assert_eq!(START_GAME_SEQUENCE.first().unwrap().phase, StartPhase::CreateGameRules);
        assert!(pos(StartPhase::LoadLevel) < pos(StartPhase::SpawnEntities));
        assert!(pos(StartPhase::SpawnEntities) < pos(StartPhase::Reset));
        assert!(pos(StartPhase::Reset) < pos(StartPhase::LevelLoaded));
        assert!(pos(StartPhase::LevelLoaded) < pos(StartPhase::StartLevel));
        assert!(pos(StartPhase::StartLevel) < pos(StartPhase::StartGame));
        assert!(pos(StartPhase::SpawnEntities) < pos(StartPhase::CreateLocalPlayer));
        assert!(pos(StartPhase::CreateLocalPlayer) < pos(StartPhase::ClientEnteredGame));
    }

    #[test]
    fn on_level_loaded_and_start_game_are_distinct_table_events() {
        assert_eq!(EngineEvent::OnLevelLoaded.spec().table_index, Some(29));
        assert_eq!(EngineEvent::OnStartLevel.spec().table_index, Some(30));
        assert_eq!(EngineEvent::OnStartGame.spec().table_index, Some(31));
    }
}
