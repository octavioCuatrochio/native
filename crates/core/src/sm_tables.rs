//! Descriptor tables for `.sm` trigger / entity / transform / filter / action
//! payload records, recovered from the engine's registration code.
//!
//! Each record type is registered by a small init routine that fills a global
//! descriptor table; the tables are zeroed at startup (`SmInt*Init`), so they
//! are not present in the image and had to be recovered from the registration
//! call sites instead:
//!
//! * triggers  - `TRIGGER_DESCS`, table `0x1ee2a40`, filler `FUN_0169e1e0`
//!   (0x169e1e0), init `FUN_0169d470` (0x169d470). Field order in the
//!   registration call is `(id, evalFn, aux, persistSize, staticSize,
//!   persistAlign, staticAlign)`.
//! * entity / transform sources - `ENTITY_DESCS` (`0x1ee2720`,
//!   `FUN_016955c0`) and `TRANSFORM_DESCS` (`0x1ee2910`, `FUN_01698980`).
//! * filters - `FILTER_DESCS`, table `0x1ee2798`, filler `FUN_01695a80`
//!   (0x1695a80).
//! * actions - `ACTION_DESCS`, table `0x1ee1018`, filler `FUN_0168a8b0`
//!   (0x168a8b0); the names are the registration strings.
//!
//! `size`/`align` describe the static operand record a payload walker consumes
//! from the matching data blob (see `sm_payload`). They were validated against
//! every block of all seven shipped machines.

/// One trigger-source descriptor (`0x1ee2a40`, 16 bytes each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmTriggerDesc {
    /// Trigger type id stored in the trigger-type byte array.
    pub id: u8,
    /// Static operand size in the data blob.
    pub static_size: u16,
    /// Static operand alignment.
    pub static_align: u8,
    /// Runtime (persistent) state size, engine-side only.
    pub persist_size: u8,
    /// Runtime (persistent) state alignment.
    pub persist_align: u8,
    /// Unused extra word (stored at descriptor `+0`).
    pub aux: u32,
    /// Address of the evaluator `(ctx, static, persist) -> bool`.
    pub eval_fn: u32,
}

/// One entity- or transform-source descriptor (12 bytes each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmSourceDesc {
    pub id: u8,
    pub size: u16,
    pub align: u8,
    pub eval_fn: u32,
}

/// One filter descriptor (`0x1ee2798`, 20 bytes each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmFilterDesc {
    pub id: u8,
    pub size: u16,
    pub align: u8,
    /// Evaluator used for collision filtering.
    pub eval_collision: u32,
    /// Evaluator used for markup filtering.
    pub eval_markup: u32,
    /// Evaluator used for transform filtering.
    pub eval_transform: u32,
}

/// One action descriptor (`0x1ee1018`, 36 bytes each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmActionDesc {
    pub id: u8,
    /// Registration name string, e.g. `Animation`, `MovementTurnTo`.
    pub name: &'static str,
    /// Static operand size consumed from the action-data blob.
    pub data_size: u16,
    /// Static operand alignment.
    pub data_align: u8,
    /// Operand size consumed from the bytecode/code stream.
    pub code_size: u16,
    pub code_align: u8,
    pub eval_fn: u32,
}

/// Look up a trigger descriptor by type id.
pub fn trigger_desc(id: u8) -> Option<&'static SmTriggerDesc> {
    TRIGGER_DESCS.binary_search_by_key(&id, |d| d.id).ok().map(|i| &TRIGGER_DESCS[i])
}

/// Look up an entity-source descriptor by type id.
pub fn entity_desc(id: u8) -> Option<&'static SmSourceDesc> {
    ENTITY_DESCS.binary_search_by_key(&id, |d| d.id).ok().map(|i| &ENTITY_DESCS[i])
}

/// Look up a transform-source descriptor by type id.
pub fn transform_desc(id: u8) -> Option<&'static SmSourceDesc> {
    TRANSFORM_DESCS.binary_search_by_key(&id, |d| d.id).ok().map(|i| &TRANSFORM_DESCS[i])
}

/// Look up a filter descriptor by type id.
pub fn filter_desc(id: u8) -> Option<&'static SmFilterDesc> {
    FILTER_DESCS.binary_search_by_key(&id, |d| d.id).ok().map(|i| &FILTER_DESCS[i])
}

/// Look up an action descriptor by descriptor id.
pub fn action_desc(id: u8) -> Option<&'static SmActionDesc> {
    ACTION_DESCS.binary_search_by_key(&id, |d| d.id).ok().map(|i| &ACTION_DESCS[i])
}

// Auto-recovered descriptor tables (see module docs).
// Trigger sources: table at 0x1ee2a40, registered by FUN_0169e1e0 0x169e1e0.
pub static TRIGGER_DESCS: &[SmTriggerDesc] = &[
    SmTriggerDesc { id: 1, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169e4d0 },
    SmTriggerDesc { id: 2, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169e590 },
    SmTriggerDesc { id: 3, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169e6c0 },
    SmTriggerDesc { id: 4, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x009e9ec0 },
    SmTriggerDesc { id: 5, static_size: 16, static_align: 4, persist_size: 8, persist_align: 4, aux: 23716736, eval_fn: 0x0169e7f0 },
    SmTriggerDesc { id: 6, static_size: 8, static_align: 4, persist_size: 12, persist_align: 4, aux: 23716704, eval_fn: 0x0169e890 },
    SmTriggerDesc { id: 7, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169e900 },
    SmTriggerDesc { id: 8, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169e910 },
    SmTriggerDesc { id: 9, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169e930 },
    SmTriggerDesc { id: 10, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169e9b0 },
    SmTriggerDesc { id: 11, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169e9e0 },
    SmTriggerDesc { id: 12, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169ea00 },
    SmTriggerDesc { id: 13, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169ea20 },
    SmTriggerDesc { id: 14, static_size: 6, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169ea80 },
    SmTriggerDesc { id: 15, static_size: 44, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169eb80 },
    SmTriggerDesc { id: 16, static_size: 102, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169ecf0 },
    SmTriggerDesc { id: 17, static_size: 18, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169eea0 },
    SmTriggerDesc { id: 18, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169ef70 },
    SmTriggerDesc { id: 19, static_size: 2, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169ef90 },
    SmTriggerDesc { id: 20, static_size: 128, static_align: 8, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169efb0 },
    SmTriggerDesc { id: 21, static_size: 128, static_align: 8, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169efd0 },
    SmTriggerDesc { id: 22, static_size: 384, static_align: 8, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169eff0 },
    SmTriggerDesc { id: 23, static_size: 256, static_align: 8, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f020 },
    SmTriggerDesc { id: 24, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f0c0 },
    SmTriggerDesc { id: 25, static_size: 8, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f050 },
    SmTriggerDesc { id: 26, static_size: 16, static_align: 4, persist_size: 8, persist_align: 4, aux: 23716736, eval_fn: 0x0169f100 },
    SmTriggerDesc { id: 27, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f190 },
    SmTriggerDesc { id: 28, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f1c0 },
    SmTriggerDesc { id: 29, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f2a0 },
    SmTriggerDesc { id: 30, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f2e0 },
    SmTriggerDesc { id: 31, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f300 },
    SmTriggerDesc { id: 32, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f320 },
    SmTriggerDesc { id: 33, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f340 },
    SmTriggerDesc { id: 34, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f2c0 },
    SmTriggerDesc { id: 35, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f360 },
    SmTriggerDesc { id: 36, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f380 },
    SmTriggerDesc { id: 37, static_size: 8, static_align: 4, persist_size: 8, persist_align: 4, aux: 23716736, eval_fn: 0x0169f3a0 },
    SmTriggerDesc { id: 38, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f420 },
    SmTriggerDesc { id: 39, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f440 },
    SmTriggerDesc { id: 40, static_size: 4, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f470 },
    SmTriggerDesc { id: 41, static_size: 4, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f4c0 },
    SmTriggerDesc { id: 42, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f530 },
    SmTriggerDesc { id: 43, static_size: 2, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f560 },
    SmTriggerDesc { id: 44, static_size: 32, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f590 },
    SmTriggerDesc { id: 45, static_size: 32, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f5d0 },
    SmTriggerDesc { id: 46, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f610 },
    SmTriggerDesc { id: 47, static_size: 32, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f630 },
    SmTriggerDesc { id: 48, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f670 },
    SmTriggerDesc { id: 49, static_size: 16, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f690 },
    SmTriggerDesc { id: 50, static_size: 8, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f720 },
    SmTriggerDesc { id: 51, static_size: 20, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f760 },
    SmTriggerDesc { id: 52, static_size: 8, static_align: 4, persist_size: 4, persist_align: 4, aux: 23716752, eval_fn: 0x0169f8e0 },
    SmTriggerDesc { id: 53, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f940 },
    SmTriggerDesc { id: 54, static_size: 2, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169f9e0 },
    SmTriggerDesc { id: 55, static_size: 33, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169fa90 },
    SmTriggerDesc { id: 56, static_size: 4, static_align: 2, persist_size: 1, persist_align: 1, aux: 23716768, eval_fn: 0x0169fb10 },
    SmTriggerDesc { id: 57, static_size: 65, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169fb80 },
    SmTriggerDesc { id: 58, static_size: 76, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169fbd0 },
    SmTriggerDesc { id: 59, static_size: 20, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169fc30 },
    SmTriggerDesc { id: 60, static_size: 4, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169fee0 },
    SmTriggerDesc { id: 61, static_size: 2, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169ff20 },
    SmTriggerDesc { id: 62, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169ff80 },
    SmTriggerDesc { id: 63, static_size: 5, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x0169ffa0 },
    SmTriggerDesc { id: 64, static_size: 8, static_align: 4, persist_size: 8, persist_align: 4, aux: 23716784, eval_fn: 0x016a0040 },
    SmTriggerDesc { id: 65, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a00b0 },
    SmTriggerDesc { id: 66, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a01c0 },
    SmTriggerDesc { id: 67, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0210 },
    SmTriggerDesc { id: 68, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0230 },
    SmTriggerDesc { id: 69, static_size: 65, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0250 },
    SmTriggerDesc { id: 70, static_size: 65, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0370 },
    SmTriggerDesc { id: 71, static_size: 36, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0490 },
    SmTriggerDesc { id: 72, static_size: 33, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0550 },
    SmTriggerDesc { id: 73, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x007fa860 },
    SmTriggerDesc { id: 74, static_size: 2, static_align: 2, persist_size: 4, persist_align: 4, aux: 23716816, eval_fn: 0x016a05f0 },
    SmTriggerDesc { id: 75, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0620 },
    SmTriggerDesc { id: 76, static_size: 20, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0640 },
    SmTriggerDesc { id: 77, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0720 },
    SmTriggerDesc { id: 78, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0740 },
    SmTriggerDesc { id: 79, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0760 },
    SmTriggerDesc { id: 80, static_size: 80, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0830 },
    SmTriggerDesc { id: 81, static_size: 12, static_align: 4, persist_size: 8, persist_align: 4, aux: 23716832, eval_fn: 0x016a0a80 },
    SmTriggerDesc { id: 82, static_size: 8, static_align: 4, persist_size: 1, persist_align: 1, aux: 23716864, eval_fn: 0x016a0bb0 },
    SmTriggerDesc { id: 83, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0c20 },
    SmTriggerDesc { id: 84, static_size: 8, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0b20 },
    SmTriggerDesc { id: 85, static_size: 16, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0b60 },
    SmTriggerDesc { id: 86, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0c50 },
    SmTriggerDesc { id: 87, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0c70 },
    SmTriggerDesc { id: 88, static_size: 32, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0c90 },
    SmTriggerDesc { id: 89, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0cc0 },
    SmTriggerDesc { id: 90, static_size: 32, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0d40 },
    SmTriggerDesc { id: 91, static_size: 1, static_align: 1, persist_size: 1, persist_align: 1, aux: 23716880, eval_fn: 0x016a0da0 },
    SmTriggerDesc { id: 92, static_size: 16, static_align: 4, persist_size: 8, persist_align: 4, aux: 23716896, eval_fn: 0x016a0dc0 },
    SmTriggerDesc { id: 93, static_size: 4, static_align: 4, persist_size: 4, persist_align: 4, aux: 23716944, eval_fn: 0x016a0e90 },
    SmTriggerDesc { id: 94, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0ec0 },
    SmTriggerDesc { id: 95, static_size: 1, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0ee0 },
    SmTriggerDesc { id: 96, static_size: 8, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0f00 },
    SmTriggerDesc { id: 97, static_size: 4, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0f50 },
    SmTriggerDesc { id: 98, static_size: 16, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a0f90 },
    SmTriggerDesc { id: 99, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1090 },
    SmTriggerDesc { id: 100, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a12d0 },
    SmTriggerDesc { id: 101, static_size: 66, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1310 },
    SmTriggerDesc { id: 102, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1380 },
    SmTriggerDesc { id: 103, static_size: 2, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1390 },
    SmTriggerDesc { id: 104, static_size: 2, static_align: 2, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a13e0 },
    SmTriggerDesc { id: 105, static_size: 20, static_align: 4, persist_size: 28, persist_align: 4, aux: 23716960, eval_fn: 0x016a1440 },
    SmTriggerDesc { id: 106, static_size: 8, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a16f0 },
    SmTriggerDesc { id: 107, static_size: 8, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1780 },
    SmTriggerDesc { id: 108, static_size: 16, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1820 },
    SmTriggerDesc { id: 109, static_size: 12, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a18d0 },
    SmTriggerDesc { id: 110, static_size: 8, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a19d0 },
    SmTriggerDesc { id: 111, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1a30 },
    SmTriggerDesc { id: 112, static_size: 4, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1a60 },
    SmTriggerDesc { id: 113, static_size: 8, static_align: 4, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1a80 },
    SmTriggerDesc { id: 114, static_size: 0, static_align: 1, persist_size: 0, persist_align: 1, aux: 0, eval_fn: 0x016a1ad0 },
];

// Entity sources: table at 0x1ee2720 (FUN_016955c0 0x16955c0). size=+8, align=+4.
pub static ENTITY_DESCS: &[SmSourceDesc] = &[
    SmSourceDesc { id: 1, size: 1, align: 1, eval_fn: 0x016954f0 },
    SmSourceDesc { id: 2, size: 0, align: 1, eval_fn: 0x01695510 },
    SmSourceDesc { id: 3, size: 0, align: 1, eval_fn: 0x01695530 },
    SmSourceDesc { id: 4, size: 0, align: 1, eval_fn: 0x01695540 },
    SmSourceDesc { id: 5, size: 0, align: 1, eval_fn: 0x01695550 },
    SmSourceDesc { id: 6, size: 1, align: 1, eval_fn: 0x01695560 },
];

// Transform sources: table at 0x1ee2910 (FUN_01698980 0x1698980). size=+4, align=+8.
pub static TRANSFORM_DESCS: &[SmSourceDesc] = &[
    SmSourceDesc { id: 1, size: 0, align: 1, eval_fn: 0x01699230 },
    SmSourceDesc { id: 2, size: 4, align: 2, eval_fn: 0x01699310 },
    SmSourceDesc { id: 3, size: 0, align: 1, eval_fn: 0x01699680 },
    SmSourceDesc { id: 4, size: 0, align: 1, eval_fn: 0x016996d0 },
    SmSourceDesc { id: 5, size: 0, align: 1, eval_fn: 0x016997b0 },
    SmSourceDesc { id: 6, size: 16, align: 4, eval_fn: 0x01699820 },
    SmSourceDesc { id: 7, size: 16, align: 4, eval_fn: 0x01699960 },
    SmSourceDesc { id: 8, size: 4, align: 2, eval_fn: 0x01699a90 },
    SmSourceDesc { id: 9, size: 0, align: 1, eval_fn: 0x01699ed0 },
    SmSourceDesc { id: 10, size: 80, align: 4, eval_fn: 0x01699f30 },
    SmSourceDesc { id: 11, size: 20, align: 4, eval_fn: 0x0169af80 },
    SmSourceDesc { id: 12, size: 80, align: 4, eval_fn: 0x0169b3b0 },
    SmSourceDesc { id: 13, size: 8, align: 4, eval_fn: 0x0169c4a0 },
    SmSourceDesc { id: 14, size: 4, align: 2, eval_fn: 0x0169c6f0 },
    SmSourceDesc { id: 15, size: 20, align: 4, eval_fn: 0x0169c810 },
    SmSourceDesc { id: 16, size: 12, align: 4, eval_fn: 0x0169cb20 },
    SmSourceDesc { id: 17, size: 4, align: 4, eval_fn: 0x0169cc10 },
    SmSourceDesc { id: 18, size: 0, align: 1, eval_fn: 0x0169ce30 },
    SmSourceDesc { id: 19, size: 16, align: 4, eval_fn: 0x0169cec0 },
    SmSourceDesc { id: 20, size: 66, align: 2, eval_fn: 0x0169d040 },
    SmSourceDesc { id: 21, size: 66, align: 2, eval_fn: 0x0169d100 },
    SmSourceDesc { id: 22, size: 0, align: 1, eval_fn: 0x0169d1b0 },
];

// Filters: table at 0x1ee2798 (FUN_01697a20 0x1697a20).
pub static FILTER_DESCS: &[SmFilterDesc] = &[
    SmFilterDesc { id: 1, size: 4, align: 2, eval_collision: 0x01695b30, eval_markup: 0x016963f0, eval_transform: 0x01696fb0 },
    SmFilterDesc { id: 2, size: 8, align: 4, eval_collision: 0x01695bd0, eval_markup: 0x01696490, eval_transform: 0x01697050 },
    SmFilterDesc { id: 3, size: 8, align: 4, eval_collision: 0x01695c90, eval_markup: 0x01696590, eval_transform: 0x01697100 },
    SmFilterDesc { id: 4, size: 12, align: 4, eval_collision: 0x01695e00, eval_markup: 0x016967b0, eval_transform: 0x01697260 },
    SmFilterDesc { id: 5, size: 4, align: 2, eval_collision: 0x01695ef0, eval_markup: 0x016967d0, eval_transform: 0x01697360 },
    SmFilterDesc { id: 6, size: 33, align: 1, eval_collision: 0x01695f90, eval_markup: 0x01696870, eval_transform: 0x01697400 },
    SmFilterDesc { id: 7, size: 2, align: 2, eval_collision: 0x01696010, eval_markup: 0x016968f0, eval_transform: 0x01697400 },
    SmFilterDesc { id: 8, size: 12, align: 4, eval_collision: 0x01696030, eval_markup: 0x01696910, eval_transform: 0x01697420 },
    SmFilterDesc { id: 9, size: 12, align: 4, eval_collision: 0x016960e0, eval_markup: 0x01696a60, eval_transform: 0x016974b0 },
    SmFilterDesc { id: 10, size: 12, align: 4, eval_collision: 0x01696180, eval_markup: 0x01696b30, eval_transform: 0x01697530 },
    SmFilterDesc { id: 11, size: 64, align: 1, eval_collision: 0x01696300, eval_markup: 0x01696b50, eval_transform: 0x016975a0 },
    // 13 and 15 are registered in the same init routine but with the argument
    // pushes separated from the call by a run of global descriptor writes.
    SmFilterDesc { id: 13, size: 4, align: 2, eval_collision: 0x01696230, eval_markup: 0x01696b90, eval_transform: 0x016975e0 },
    SmFilterDesc { id: 15, size: 1, align: 1, eval_collision: 0x01696300, eval_markup: 0x01696c50, eval_transform: 0x016976a0 },
];

// Actions: table at 0x1ee1018 (FUN_0168a8b0 0x168a8b0). Names are the
// registration strings; data_size/data_align describe the action-data operand.
pub static ACTION_DESCS: &[SmActionDesc] = &[
    SmActionDesc { id: 1, name: "AISignal", data_size: 0, data_align: 1, code_size: 64, code_align: 1, eval_fn: 0x0168a990 },
    SmActionDesc { id: 2, name: "ActorFaceTransformMode", data_size: 0, data_align: 1, code_size: 12, code_align: 4, eval_fn: 0x0168a9d0 },
    SmActionDesc { id: 3, name: "ActorKill", data_size: 0, data_align: 1, code_size: 2, code_align: 1, eval_fn: 0x0168ad00 },
    SmActionDesc { id: 4, name: "ActorStatAdd", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168af30 },
    SmActionDesc { id: 5, name: "ActorStatSub", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168afe0 },
    SmActionDesc { id: 6, name: "AimMode", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 7, name: "AllowAimMode", data_size: 4, data_align: 4, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 8, name: "AllowMovementMode", data_size: 4, data_align: 4, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 9, name: "AllowRotationMode", data_size: 4, data_align: 4, code_size: 8, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 10, name: "AnimSteeringWheelMode", data_size: 4, data_align: 4, code_size: 4, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 11, name: "AnimTurnLeanScaleMode", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 12, name: "AnimVelocityScaleMode", data_size: 4, data_align: 4, code_size: 8, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 13, name: "Animation", data_size: 0, data_align: 1, code_size: 144, code_align: 4, eval_fn: 0x0168b090 },
    SmActionDesc { id: 14, name: "AnimationControlsRotationMode", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 15, name: "AnimationGraphInput", data_size: 0, data_align: 1, code_size: 128, code_align: 1, eval_fn: 0x0168b120 },
    SmActionDesc { id: 16, name: "AnimationSlowdown", data_size: 0, data_align: 1, code_size: 20, code_align: 4, eval_fn: 0x0168b160 },
    SmActionDesc { id: 17, name: "BarricadeAcquire", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168b1e0 },
    SmActionDesc { id: 18, name: "BarricadeBuild", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168b220 },
    SmActionDesc { id: 19, name: "BarricadeRelease", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168b250 },
    SmActionDesc { id: 20, name: "BarricadeSmash", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168b280 },
    SmActionDesc { id: 21, name: "BlockTasksMode", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 22, name: "BloodAmount", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168b300 },
    SmActionDesc { id: 23, name: "CameraMode", data_size: 4, data_align: 4, code_size: 32, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 24, name: "CameraFocalPositionMode", data_size: 4, data_align: 4, code_size: 4, code_align: 2, eval_fn: 0x0168b3e0 },
    SmActionDesc { id: 25, name: "ChallengeCredit", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x0168b4f0 },
    SmActionDesc { id: 26, name: "CollisionDimensionsMode", data_size: 4, data_align: 4, code_size: 16, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 27, name: "CollisionQueryMode", data_size: 4, data_align: 4, code_size: 312, code_align: 8, eval_fn: 0x0168b550 },
    SmActionDesc { id: 28, name: "CombatCollisionMode", data_size: 4, data_align: 4, code_size: 312, code_align: 8, eval_fn: 0x0168b550 },
    SmActionDesc { id: 29, name: "CombatDamage", data_size: 0, data_align: 1, code_size: 136, code_align: 4, eval_fn: 0x0168b5e0 },
    SmActionDesc { id: 30, name: "Constrain2Mode", data_size: 84, data_align: 4, code_size: 104, code_align: 4, eval_fn: 0x0168b950 },
    SmActionDesc { id: 31, name: "ConstrainToMarkupEdgeMode", data_size: 4, data_align: 4, code_size: 92, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 32, name: "ContextTagMode", data_size: 0, data_align: 1, code_size: 2, code_align: 2, eval_fn: 0x00000000 },
    SmActionDesc { id: 33, name: "ContextTagVolumeMode", data_size: 64, data_align: 8, code_size: 64, code_align: 4, eval_fn: 0x016afc80 },
    SmActionDesc { id: 34, name: "ContextTagRemove", data_size: 0, data_align: 1, code_size: 2, code_align: 2, eval_fn: 0x0168bc00 },
    SmActionDesc { id: 35, name: "CounterCredit", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x0168bc20 },
    SmActionDesc { id: 36, name: "CounterSlotAdd", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168bc30 },
    SmActionDesc { id: 37, name: "CounterSlotSet", data_size: 0, data_align: 1, code_size: 40, code_align: 4, eval_fn: 0x0168bc50 },
    SmActionDesc { id: 38, name: "CounterSlotSetRandom", data_size: 0, data_align: 1, code_size: 44, code_align: 4, eval_fn: 0x0168bc70 },
    SmActionDesc { id: 39, name: "Damage", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x0168bcb0 },
    SmActionDesc { id: 40, name: "DamageFromExpertise", data_size: 0, data_align: 1, code_size: 72, code_align: 4, eval_fn: 0x0168bd00 },
    SmActionDesc { id: 41, name: "DamageOverTimeMode", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x0168bea0 },
    SmActionDesc { id: 42, name: "DamageOverTimeFromExpertiseMode", data_size: 0, data_align: 1, code_size: 72, code_align: 4, eval_fn: 0x0168bdd0 },
    SmActionDesc { id: 43, name: "DebugCategoryMode", data_size: 0, data_align: 1, code_size: 100, code_align: 2, eval_fn: 0x01690c80 },
    SmActionDesc { id: 44, name: "DebugLog", data_size: 0, data_align: 1, code_size: 64, code_align: 1, eval_fn: 0x00472d50 },
    SmActionDesc { id: 47, name: "DisableFootAlignmentIKMode", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 48, name: "DisableLookIKMode", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 49, name: "DisableUseButtonMode", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 50, name: "Dismember", data_size: 0, data_align: 1, code_size: 3, code_align: 1, eval_fn: 0x0168bf00 },
    SmActionDesc { id: 51, name: "DoorAcquire", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168bfd0 },
    SmActionDesc { id: 52, name: "DoorClose", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168c000 },
    SmActionDesc { id: 53, name: "DoorLock", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168c030 },
    SmActionDesc { id: 54, name: "DoorOpen", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x0168c060 },
    SmActionDesc { id: 55, name: "DoorRelease", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168c0a0 },
    SmActionDesc { id: 56, name: "DoorRepair", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168c0d0 },
    SmActionDesc { id: 57, name: "DoorSlam", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168c100 },
    SmActionDesc { id: 58, name: "EchoQueryMode", data_size: 4, data_align: 4, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 59, name: "EntityAddTrait", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168c130 },
    SmActionDesc { id: 60, name: "EntityRemoveTrait", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168c180 },
    SmActionDesc { id: 61, name: "EntitySlotAssign", data_size: 0, data_align: 1, code_size: 2, code_align: 1, eval_fn: 0x0168c1d0 },
    SmActionDesc { id: 62, name: "EntitySlotAssignMulti", data_size: 0, data_align: 1, code_size: 3, code_align: 1, eval_fn: 0x0168c230 },
    SmActionDesc { id: 63, name: "EntitySlotClear", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x0168c2c0 },
    SmActionDesc { id: 64, name: "EntitySlotClearAll", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168c2e0 },
    SmActionDesc { id: 65, name: "EntitySlotContextTagMode", data_size: 4, data_align: 4, code_size: 4, code_align: 2, eval_fn: 0x0168c310 },
    SmActionDesc { id: 66, name: "EntitySlotCopy", data_size: 0, data_align: 1, code_size: 2, code_align: 1, eval_fn: 0x0168c380 },
    SmActionDesc { id: 67, name: "EntityQuery2Mode", data_size: 32, data_align: 8, code_size: 64, code_align: 4, eval_fn: 0x016aff00 },
    SmActionDesc { id: 68, name: "EntityQueryZombieMode", data_size: 32, data_align: 8, code_size: 64, code_align: 4, eval_fn: 0x016b02a0 },
    SmActionDesc { id: 69, name: "EventTrigger", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x0168c3c0 },
    SmActionDesc { id: 70, name: "ExpertisePoints", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168c3e0 },
    SmActionDesc { id: 71, name: "FallingDamageStart", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168c430 },
    SmActionDesc { id: 72, name: "FallingDamageCancel", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168c420 },
    SmActionDesc { id: 73, name: "FallingDamageStop", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168c450 },
    SmActionDesc { id: 74, name: "ForceFeedback", data_size: 0, data_align: 1, code_size: 12, code_align: 4, eval_fn: 0x0168c4e0 },
    SmActionDesc { id: 75, name: "FxMaterialEffect", data_size: 0, data_align: 1, code_size: 97, code_align: 1, eval_fn: 0x0168c530 },
    SmActionDesc { id: 76, name: "FxPlayAudio", data_size: 0, data_align: 1, code_size: 98, code_align: 1, eval_fn: 0x0168c6c0 },
    SmActionDesc { id: 77, name: "FxPlayWeaponSound", data_size: 0, data_align: 1, code_size: 33, code_align: 1, eval_fn: 0x0168c750 },
    SmActionDesc { id: 78, name: "FxSpawnParticleEffect", data_size: 0, data_align: 1, code_size: 124, code_align: 4, eval_fn: 0x0168c790 },
    SmActionDesc { id: 79, name: "Heal", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x0168c8b0 },
    SmActionDesc { id: 80, name: "HealthRecovery", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x0168c950 },
    SmActionDesc { id: 81, name: "HealOverTimeMode", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x0168c900 },
    SmActionDesc { id: 82, name: "LaunchEntity", data_size: 0, data_align: 1, code_size: 84, code_align: 4, eval_fn: 0x0168c990 },
    SmActionDesc { id: 83, name: "LinkToEntityMode", data_size: 4, data_align: 4, code_size: 2, code_align: 2, eval_fn: 0x00000000 },
    SmActionDesc { id: 84, name: "LinkToVehicleMode", data_size: 4, data_align: 4, code_size: 72, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 85, name: "LockConsumableMode", data_size: 4, data_align: 4, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 86, name: "LookIKMode", data_size: 0, data_align: 1, code_size: 2, code_align: 2, eval_fn: 0x0168ca80 },
    SmActionDesc { id: 87, name: "MakeNoise", data_size: 0, data_align: 1, code_size: 16, code_align: 4, eval_fn: 0x0168cce0 },
    SmActionDesc { id: 88, name: "MarkupSlotAssign", data_size: 0, data_align: 1, code_size: 2, code_align: 1, eval_fn: 0x0168cd30 },
    SmActionDesc { id: 89, name: "MarkupSlotClear", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x0168cdb0 },
    SmActionDesc { id: 90, name: "MarkupSlotClearAll", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168cdd0 },
    SmActionDesc { id: 91, name: "MarkupSlotCopy", data_size: 0, data_align: 1, code_size: 2, code_align: 1, eval_fn: 0x0168ce00 },
    SmActionDesc { id: 92, name: "MarkupQuery2Mode", data_size: 4, data_align: 4, code_size: 188, code_align: 4, eval_fn: 0x016b13a0 },
    SmActionDesc { id: 93, name: "MarkupRayQuery2Mode", data_size: 4, data_align: 4, code_size: 144, code_align: 4, eval_fn: 0x016b1650 },
    SmActionDesc { id: 94, name: "MountedAnimationMode", data_size: 56, data_align: 8, code_size: 196, code_align: 4, eval_fn: 0x0168ce70 },
    SmActionDesc { id: 95, name: "MovementLaunchToPoint", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168d6e0 },
    SmActionDesc { id: 96, name: "MovementSlideToPoint", data_size: 0, data_align: 1, code_size: 28, code_align: 4, eval_fn: 0x0168d8d0 },
    SmActionDesc { id: 97, name: "MovementSlideToPointMode", data_size: 4, data_align: 4, code_size: 24, code_align: 4, eval_fn: 0x0168dba0 },
    SmActionDesc { id: 98, name: "MovementSpeedOverrideMode", data_size: 4, data_align: 4, code_size: 16, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 99, name: "MovementTurnTo", data_size: 0, data_align: 1, code_size: 20, code_align: 4, eval_fn: 0x0168dc50 },
    SmActionDesc { id: 100, name: "MountSlotClearAll", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168ce40 },
    SmActionDesc { id: 101, name: "MusicCue", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x0168e0b0 },
    SmActionDesc { id: 102, name: "PhysicsApplyImpulse", data_size: 0, data_align: 1, code_size: 8, code_align: 4, eval_fn: 0x0168e0e0 },
    SmActionDesc { id: 103, name: "PhysicsBlockVehiclesMode", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 104, name: "PhysicsColliderMode", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x0168e2f0 },
    SmActionDesc { id: 105, name: "PhysicsHit", data_size: 0, data_align: 1, code_size: 32, code_align: 4, eval_fn: 0x0168e380 },
    SmActionDesc { id: 106, name: "PhysicsIgnoreCollisionMode", data_size: 8, data_align: 4, code_size: 2, code_align: 2, eval_fn: 0x00000000 },
    SmActionDesc { id: 107, name: "PlaceEntityMode", data_size: 4, data_align: 4, code_size: 66, code_align: 2, eval_fn: 0x0168eb00 },
    SmActionDesc { id: 108, name: "PlayerDowned", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168ebc0 },
    SmActionDesc { id: 109, name: "PlayerDownedRecover", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168ebd0 },
    SmActionDesc { id: 110, name: "PlayerFaint", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168ebe0 },
    SmActionDesc { id: 111, name: "PlayerVulnerable", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168ecd0 },
    SmActionDesc { id: 112, name: "PlayerVulnerableEnd", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168ece0 },
    SmActionDesc { id: 113, name: "ProjectileReticleMode", data_size: 4, data_align: 4, code_size: 12, code_align: 4, eval_fn: 0x0168ecf0 },
    SmActionDesc { id: 114, name: "RTSResourceNodeQueryMode", data_size: 4, data_align: 4, code_size: 120, code_align: 4, eval_fn: 0x0168edb0 },
    SmActionDesc { id: 115, name: "RucksackDrop", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168ee70 },
    SmActionDesc { id: 116, name: "RucksackPickup", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168eef0 },
    SmActionDesc { id: 117, name: "SceneEventPlay", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x0168ef80 },
    SmActionDesc { id: 118, name: "ScenePlay", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168f000 },
    SmActionDesc { id: 119, name: "ScenePlaySpecific", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x0168f080 },
    SmActionDesc { id: 120, name: "ScreenShake", data_size: 0, data_align: 1, code_size: 44, code_align: 4, eval_fn: 0x0168f0f0 },
    SmActionDesc { id: 121, name: "SearchMode", data_size: 4, data_align: 4, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 122, name: "SelectItem", data_size: 0, data_align: 1, code_size: 64, code_align: 1, eval_fn: 0x0168f1c0 },
    SmActionDesc { id: 123, name: "SelectMeleeWeapon", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168f1e0 },
    SmActionDesc { id: 124, name: "SelectRangedWeapon", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168f280 },
    SmActionDesc { id: 125, name: "SetCollisionDims", data_size: 0, data_align: 1, code_size: 16, code_align: 4, eval_fn: 0x0168f290 },
    SmActionDesc { id: 126, name: "SetInteractProp", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x0168f340 },
    SmActionDesc { id: 127, name: "SetInteractZombie", data_size: 0, data_align: 1, code_size: 4, code_align: 1, eval_fn: 0x0168f360 },
    SmActionDesc { id: 128, name: "SnapFacingToTransform", data_size: 0, data_align: 1, code_size: 2, code_align: 2, eval_fn: 0x0168f3a0 },
    SmActionDesc { id: 129, name: "SocialDisplayMode", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 130, name: "SpaceTestMode", data_size: 4, data_align: 4, code_size: 80, code_align: 4, eval_fn: 0x0168fb30 },
    SmActionDesc { id: 131, name: "SpaceTestPushMode", data_size: 4, data_align: 4, code_size: 32, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 132, name: "StaminaPenaltyMode", data_size: 4, data_align: 4, code_size: 4, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 133, name: "StaminaTankModify", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x0168fd20 },
    SmActionDesc { id: 134, name: "StaminaTankRateMode", data_size: 4, data_align: 4, code_size: 4, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 135, name: "StateMachineAnimationMode", data_size: 4, data_align: 4, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 136, name: "StorePosition", data_size: 0, data_align: 1, code_size: 2, code_align: 2, eval_fn: 0x0168fd30 },
    SmActionDesc { id: 137, name: "StoredPositionClear", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x0168fdb0 },
    SmActionDesc { id: 138, name: "StrafeMode", data_size: 4, data_align: 4, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 139, name: "ThrowEntity", data_size: 0, data_align: 1, code_size: 76, code_align: 4, eval_fn: 0x0168fde0 },
    SmActionDesc { id: 140, name: "ThrowEntityAtReticle", data_size: 0, data_align: 1, code_size: 76, code_align: 4, eval_fn: 0x0168ffe0 },
    SmActionDesc { id: 141, name: "TimeScale", data_size: 0, data_align: 1, code_size: 16, code_align: 4, eval_fn: 0x016902a0 },
    SmActionDesc { id: 142, name: "TimeScaleMode", data_size: 0, data_align: 1, code_size: 12, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 143, name: "TimerSlotPause", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x016902d0 },
    SmActionDesc { id: 144, name: "TimerSlotResume", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x016902f0 },
    SmActionDesc { id: 145, name: "TimerSlotStart", data_size: 0, data_align: 1, code_size: 76, code_align: 4, eval_fn: 0x01690310 },
    SmActionDesc { id: 146, name: "TimerSlotStop", data_size: 0, data_align: 1, code_size: 1, code_align: 1, eval_fn: 0x01690390 },
    SmActionDesc { id: 147, name: "TurnToLookAtMode", data_size: 4, data_align: 4, code_size: 20, code_align: 4, eval_fn: 0x016903b0 },
    SmActionDesc { id: 148, name: "UseConsumable", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x01690760 },
    SmActionDesc { id: 149, name: "VehicleAnimation", data_size: 0, data_align: 1, code_size: 128, code_align: 1, eval_fn: 0x01690770 },
    SmActionDesc { id: 150, name: "VehicleAttachmentMode", data_size: 4, data_align: 4, code_size: 72, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 151, name: "VehicleChangeSeat", data_size: 0, data_align: 1, code_size: 4, code_align: 4, eval_fn: 0x016907a0 },
    SmActionDesc { id: 152, name: "VehicleDoorAttackMode", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x00000000 },
    SmActionDesc { id: 153, name: "VehicleExit", data_size: 0, data_align: 1, code_size: 2, code_align: 2, eval_fn: 0x01690820 },
    SmActionDesc { id: 154, name: "VehicleFinishSeating", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x01690900 },
    SmActionDesc { id: 155, name: "VehicleHit", data_size: 0, data_align: 1, code_size: 12, code_align: 4, eval_fn: 0x01690940 },
    SmActionDesc { id: 156, name: "VehicleSlowMode", data_size: 4, data_align: 4, code_size: 8, code_align: 4, eval_fn: 0x00000000 },
    SmActionDesc { id: 157, name: "VehicleStopEngine", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x01690a20 },
    SmActionDesc { id: 158, name: "VehicleUse", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x01690a50 },
    SmActionDesc { id: 159, name: "WeaponBreak", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x01690a80 },
    SmActionDesc { id: 160, name: "WeaponUseDurability", data_size: 0, data_align: 1, code_size: 0, code_align: 1, eval_fn: 0x01690ad0 },
    SmActionDesc { id: 161, name: "ZombieActivateMode", data_size: 4, data_align: 4, code_size: 2, code_align: 2, eval_fn: 0x01690b20 },
    SmActionDesc { id: 162, name: "ZombieSetType", data_size: 0, data_align: 1, code_size: 32, code_align: 1, eval_fn: 0x01690c50 },
];
