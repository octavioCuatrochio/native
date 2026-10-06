//! Inventory: item database, slot-based containers, equipment and search/loot.
//!
//! Reconstructed from the shipped data and the recovery notes:
//!  * `notes/inventory-community.md` §1 (item model, categories, `ItemSystem`),
//!    §2 (search/loot tables), §3 (resource rucksacks), §5.1 (implementable
//!    inventory model).
//!  * `notes/ghidra/rts-sim.md` §3 (resource model, rucksack/stockpile order).
//!
//! The master item DB is `libs/class3/items/items.xml`: a root `<ItemData>`
//! holding `*Definitions` sections; every real definition embeds exactly one
//! `<InventoryItemData Id=... Encumbrance=... Value=...>` record. The section
//! (Ammo/Backpack/Consumable/Gun/Rucksack/Weapon) gives the logical category and
//! the definition attributes give `EntityClass`, `MaximumStackSize`,
//! durability bounds and the rest of the physical params.
//!
//! The search side follows `libs/class3/rts/search.xml`: weighted item lists
//! (`SearchItemList`), each entry optionally yielding a `Min..Max` count, are
//! rolled with a small deterministic xorshift RNG (no external deps).
//!
//! This module is deliberately self-contained apart from `std`, `log` and
//! `quick_xml`, and defines its own RNG so the loot rolls are reproducible.

use std::collections::HashMap;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// Logical category of an item, derived from the `items.xml` section and some
/// entity-class specialisation (grenades/mines/flares are "consumables" in the
/// XML but behave as throwables; the fuel resource rucksack is fuel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemClass {
    /// Melee weapon (`WeaponDefinitions`).
    Melee,
    /// Firearm (`GunDefinitions`).
    Gun,
    /// Ammunition (`AmmoDefinitions`).
    Ammo,
    /// Usable item (`ConsumableDefinitions`), e.g. health pack, food.
    Consumable,
    /// Thrown/placed ordnance: grenades, mines, flares, firecrackers, molotovs.
    Throwable,
    /// Carryable resource pack (`RucksackDefinitions`).
    Rucksack,
    /// Worn inventory expansion (`BackpackDefinitions`).
    Backpack,
    /// The fuel resource rucksack (`FuelRucksack`).
    Fuel,
    /// Anything not otherwise classified.
    Other,
}

impl ItemClass {
    /// Parse a category name. Accepts the canonical `name()` values plus common
    /// aliases (`weapon`, `rifle`, `grenade`, `resource`, `meds`, ...).
    pub fn from_str(s: &str) -> ItemClass {
        match s.trim().to_ascii_lowercase().as_str() {
            "melee" | "weapon" | "meleeweapon" | "melee_weapon" | "blunt" | "edged" => {
                ItemClass::Melee
            }
            "gun" | "ranged" | "pistol" | "rifle" | "firearm" | "shotgun" => ItemClass::Gun,
            "ammo" | "ammunition" | "bullet" | "bullets" => ItemClass::Ammo,
            "consumable" | "use" | "usable" | "food" | "meds" | "medicine" => {
                ItemClass::Consumable
            }
            "throwable" | "grenade" | "mine" | "explosive" | "ordnance" | "flare" => {
                ItemClass::Throwable
            }
            "rucksack" | "resource" | "resourcepack" | "resource_pack" => ItemClass::Rucksack,
            "backpack" | "bag" | "pack" => ItemClass::Backpack,
            "fuel" | "gas" | "petrol" => ItemClass::Fuel,
            _ => ItemClass::Other,
        }
    }

    /// Canonical display name.
    pub fn name(self) -> &'static str {
        match self {
            ItemClass::Melee => "Melee",
            ItemClass::Gun => "Gun",
            ItemClass::Ammo => "Ammo",
            ItemClass::Consumable => "Consumable",
            ItemClass::Throwable => "Throwable",
            ItemClass::Rucksack => "Rucksack",
            ItemClass::Backpack => "Backpack",
            ItemClass::Fuel => "Fuel",
            ItemClass::Other => "Other",
        }
    }

    /// Stack size used when the definition gives none. All shipped ammo is 30;
    /// everything else is effectively unstackable (1).
    pub fn default_stack_max(self) -> u32 {
        match self {
            ItemClass::Ammo => 30,
            _ => 1,
        }
    }

    /// Which equipment slot (if any) this class may occupy.
    pub fn slot(self) -> Option<EquipSlot> {
        match self {
            ItemClass::Melee => Some(EquipSlot::Melee),
            ItemClass::Gun => Some(EquipSlot::Ranged),
            ItemClass::Consumable | ItemClass::Throwable => Some(EquipSlot::Consumable),
            ItemClass::Rucksack | ItemClass::Backpack | ItemClass::Fuel => {
                Some(EquipSlot::Rucksack)
            }
            ItemClass::Ammo | ItemClass::Other => None,
        }
    }
}

/// Equipment slots. Matches the `Goods`/`Offhand`-style carry slots the actor
/// bindings expose (melee, ranged, usable item, carried pack).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EquipSlot {
    Melee,
    Ranged,
    Consumable,
    Rucksack,
}

impl EquipSlot {
    pub fn name(self) -> &'static str {
        match self {
            EquipSlot::Melee => "melee",
            EquipSlot::Ranged => "ranged",
            EquipSlot::Consumable => "consumable",
            EquipSlot::Rucksack => "rucksack",
        }
    }
}

/// One `<InventoryItemData>` record plus the enclosing definition's params.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemDef {
    /// `InventoryItemData Id` — the primary key (e.g. `Pistol_M9`).
    pub id: String,
    /// Logical category.
    pub class: ItemClass,
    /// Carry weight per unit (`Encumbrance`).
    pub weight: f32,
    /// Maximum stack count (`MaximumStackSize`, ammo 30; else 1).
    pub stack_max: u32,
    /// Scavenge/trade worth (`Value`).
    pub value: f32,
    /// Every definition attribute plus `EntityClass`, `Name`, `Description` and
    /// the `InventoryItemData` attributes (sounds, icon, ...).
    pub params: HashMap<String, String>,
}

impl ItemDef {
    /// Look up a raw parameter.
    pub fn param(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }

    /// The spawn entity class (definition attribute), falling back to the id.
    pub fn entity_class(&self) -> &str {
        self.param("EntityClass").unwrap_or(&self.id)
    }

    /// Localised display name from `<NameText>`, falling back to the id.
    pub fn display_name(&self) -> &str {
        self.param("Name").unwrap_or(&self.id)
    }

    /// Parse a numeric param, if present and valid.
    pub fn param_f32(&self, key: &str) -> Option<f32> {
        self.param(key).and_then(|s| s.trim().parse::<f32>().ok())
    }
}

/// Parse a `<NameText>`/`<DescriptionText>` `Text` attribute into a string.
fn attr_text(e: &BytesStart) -> Option<String> {
    get_attr(e, "Text")
}

/// The master item database, loaded from `items.xml`.
#[derive(Debug, Clone, Default)]
pub struct ItemDb {
    defs: Vec<ItemDef>,
    index: HashMap<String, usize>,
}

impl ItemDb {
    pub fn new() -> Self {
        ItemDb { defs: Vec::new(), index: HashMap::new() }
    }

    /// Parse `items.xml` bytes. Definitions without an `InventoryItemData Id`
    /// (expertise/noise/reticle tables) are skipped. Duplicate ids overwrite so
    /// the DB is keyed by id.
    pub fn load_from_xml(xml: &[u8]) -> ItemDb {
        let xml = xml.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(xml);
        let mut reader = Reader::from_reader(xml);
        let mut buf = Vec::new();
        let mut db = ItemDb::new();
        // Enclosing `*Definitions` section, e.g. "WeaponDefinitions".
        let mut section = String::new();
        // The definition currently being accumulated.
        let mut build: Option<Build> = None;

        loop {
            let ev = match reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => break,
                Err(_) => break,
                Ok(ev) => ev,
            };

            match ev {
                Event::Start(e) => {
                    let name = e.name().as_ref().to_vec();
                    let text = String::from_utf8_lossy(&name).into_owned();

                    if text.ends_with("Definitions") {
                        section = text.clone();
                    } else if text.ends_with("Definition") {
                        build = Some(Build::new());
                        if let Some(b) = build.as_mut() {
                            b.attrs = attrs_map(&e);
                        }
                    } else if let Some(b) = build.as_mut() {
                        b.child(&text, &e);
                    }
                }

                Event::Empty(e) => {
                    let name = e.name().as_ref().to_vec();
                    let text = String::from_utf8_lossy(&name).into_owned();

                    if text.ends_with("Definitions") {
                        section = text;
                    } else if !text.ends_with("Definition") {
                        if let Some(b) = build.as_mut() {
                            b.child(&text, &e);
                        }
                    }
                }

                Event::End(e) => {
                    let name = e.name().as_ref().to_vec();
                    let text = String::from_utf8_lossy(&name);
                    if text.ends_with("Definition") {
                        if let Some(b) = build.take() {
                            if let Some(def) = b.finish(&section) {
                                db.insert(def);
                            }
                        }
                    }
                }

                _ => {}
            }
            buf.clear();
        }

        db
    }

    /// Insert or replace a definition by id.
    pub fn insert(&mut self, def: ItemDef) {
        match self.index.get(&def.id) {
            Some(&i) => self.defs[i] = def,
            None => {
                self.index.insert(def.id.clone(), self.defs.len());
                self.defs.push(def);
            }
        }
    }

    pub fn get(&self, id: &str) -> Option<&ItemDef> {
        self.index.get(id).map(|&i| &self.defs[i])
    }

    /// First definition whose `EntityClass` matches (or whose id matches).
    pub fn get_by_entity_class(&self, entity_class: &str) -> Option<&ItemDef> {
        self.defs
            .iter()
            .find(|d| d.entity_class() == entity_class || d.id == entity_class)
    }

    /// Stack size for an id, defaulting to 1 when unknown.
    pub fn stack_max(&self, id: &str) -> u32 {
        self.get(id).map(|d| d.stack_max).unwrap_or(1).max(1)
    }

    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ItemDef> {
        self.defs.iter()
    }
}

impl<'a> IntoIterator for &'a ItemDb {
    type Item = &'a ItemDef;
    type IntoIter = std::slice::Iter<'a, ItemDef>;
    fn into_iter(self) -> Self::IntoIter {
        self.defs.iter()
    }
}

/// Accumulator for one definition element.
struct Build {
    attrs: HashMap<String, String>,
    inv: HashMap<String, String>,
    name: Option<String>,
    description: Option<String>,
}

impl Build {
    fn new() -> Self {
        Build {
            attrs: HashMap::new(),
            inv: HashMap::new(),
            name: None,
            description: None,
        }
    }

    /// Fold one child element of a definition into the accumulator.
    fn child(&mut self, name: &str, e: &BytesStart) {
        match name {
            "InventoryItemData" => {
                for (k, v) in attrs_map(e) {
                    self.inv.insert(k, v);
                }
                if let Some(t) = attr_text(e) {
                    self.inv.insert("Text".into(), t);
                }
            }
            "NameText" => {
                if let Some(t) = attr_text(e) {
                    self.name = Some(t);
                }
            }
            "DescriptionText" => {
                if let Some(t) = attr_text(e) {
                    self.description = Some(t);
                }
            }
            _ => {}
        }
    }

    fn finish(self, section: &str) -> Option<ItemDef> {
        let id = self.inv.get("Id")?.trim().to_string();
        if id.is_empty() {
            return None;
        }

        let entity_class = self
            .attrs
            .get("EntityClass")
            .cloned()
            .or_else(|| self.inv.get("EntityClass").cloned());
        let class = classify(section, &id, entity_class.as_deref());

        let weight = parse_f32(
            self.inv
                .get("Encumbrance")
                .or_else(|| self.attrs.get("Encumbrance")),
        )
        .unwrap_or(0.0);
        let value = parse_f32(self.inv.get("Value").or_else(|| self.attrs.get("Value")))
            .unwrap_or(0.0);

        let stack_attr = self
            .attrs
            .get("MaximumStackSize")
            .or_else(|| self.attrs.get("MaxStackSize"))
            .or_else(|| self.attrs.get("StackSize"));
        let stack_max = stack_attr
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| class.default_stack_max());

        // raw params: definition attrs, then InventoryItemData attrs (which
        // override), plus the display strings.
        let mut params = self.attrs;
        for (k, v) in self.inv {
            params.insert(k, v);
        }
        if let Some(ec) = entity_class {
            params.entry("EntityClass".into()).or_insert(ec);
        }
        if let Some(n) = self.name {
            params.insert("Name".into(), n);
        }
        if let Some(d) = self.description {
            params.insert("Description".into(), d);
        }

        Some(ItemDef { id, class, weight, stack_max, value, params })
    }
}

/// Classify a definition from its section, id and entity class.
fn classify(section: &str, id: &str, entity_class: Option<&str>) -> ItemClass {
    let base = match section {
        s if s.starts_with("Ammo") => ItemClass::Ammo,
        s if s.starts_with("Backpack") => ItemClass::Backpack,
        s if s.starts_with("Consumable") => ItemClass::Consumable,
        s if s.starts_with("Gun") => ItemClass::Gun,
        s if s.starts_with("Rucksack") => ItemClass::Rucksack,
        s if s.starts_with("Weapon") => ItemClass::Melee,
        _ => ItemClass::Other,
    };

    // Consumables that are actually thrown/placed ordnance.
    if base == ItemClass::Consumable {
        let probe = entity_class.unwrap_or(id).to_ascii_lowercase();
        const THROWABLE: [&str; 7] = [
            "grenade",
            "mine",
            "flare",
            "firecracker",
            "molotov",
            "pipebomb",
            "smoke",
        ];
        if THROWABLE.iter().any(|k| probe.contains(k)) {
            return ItemClass::Throwable;
        }
    }

    // The fuel resource pack is kept distinct from the generic rucksack.
    if base == ItemClass::Rucksack && id.to_ascii_lowercase().contains("fuel") {
        return ItemClass::Fuel;
    }

    base
}

/// Parse a float, trimming whitespace. Empty/invalid -> `None`.
fn parse_f32(s: Option<&String>) -> Option<f32> {
    s.map(|s| s.trim()).and_then(|s| s.parse::<f32>().ok())
}

/// Collect all attributes of an element into a map (XML-unescaped).
fn attrs_map(e: &BytesStart) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for a in e.attributes().flatten() {
        if let (Ok(k), Ok(v)) = (std::str::from_utf8(a.key.as_ref()), a.unescape_value()) {
            m.insert(k.to_string(), v.into_owned());
        }
    }
    m
}

/// One attribute by exact name, unescaped.
fn get_attr(e: &BytesStart, name: &str) -> Option<String> {
    for a in e.attributes().flatten() {
        if a.key.as_ref() == name.as_bytes() {
            return a.unescape_value().ok().map(|c| c.into_owned());
        }
    }
    None
}

/// A counted stack of one item id.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemStack {
    pub id: String,
    pub count: u32,
    /// Durability/condition as a `0.0..=1.0` fraction (1.0 = new). Guns and melee
    /// start at their max condition; consumables/ammo keep 1.0.
    pub durability: f32,
}

impl ItemStack {
    /// A new full-durability stack.
    pub fn new(id: impl Into<String>, count: u32) -> Self {
        ItemStack { id: id.into(), count, durability: 1.0 }
    }

    /// A stack with an explicit durability fraction.
    pub fn with_durability(id: impl Into<String>, count: u32, durability: f32) -> Self {
        ItemStack { id: id.into(), count, durability: durability.clamp(0.0, 1.0) }
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Durability clamped to `0.0..=1.0`.
    pub fn durability_ratio(&self) -> f32 {
        self.durability.clamp(0.0, 1.0)
    }
}

/// Fallback stack size for an id the inventory has never seen: unstackable.
pub const DEFAULT_STACK_MAX: u32 = 1;

/// A slot-based inventory. Each slot holds one `ItemStack`; `stack_max` per id
/// controls merging. `capacity` is the number of slots.
#[derive(Debug, Clone)]
pub struct Inventory {
    slots: Vec<ItemStack>,
    capacity: usize,
    stack_max: HashMap<String, u32>,
}

impl Inventory {
    /// New empty inventory with `capacity` slots.
    pub fn new(capacity: usize) -> Self {
        Inventory { slots: Vec::new(), capacity, stack_max: HashMap::new() }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Raise the capacity, preserving existing slots.
    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        // Never truncate silently: excess slots are dropped only on explicit
        // capacity shrink.
        self.slots.truncate(capacity);
    }

    /// Teach the inventory a stack size for `id` (used by `add`).
    pub fn set_stack_max(&mut self, id: &str, max: u32) {
        self.stack_max.insert(id.to_string(), max.max(1));
    }

    fn stack_max_for(&self, id: &str) -> u32 {
        self.stack_max.get(id).copied().unwrap_or(DEFAULT_STACK_MAX).max(1)
    }

    pub fn slots(&self) -> &[ItemStack] {
        &self.slots
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.count == 0)
    }

    pub fn has(&self, id: &str) -> bool {
        self.slots.iter().any(|s| s.id == id && s.count > 0)
    }

    /// Add a stack, first topping up existing same-id stacks up to their
    /// stack_max, then opening new slots. Returns the leftover count that did
    /// not fit (0 when everything fit).
    pub fn add(&mut self, stack: ItemStack) -> u32 {
        if stack.count == 0 {
            return 0;
        }
        let max = self.stack_max_for(&stack.id);
        let mut remaining = stack.count;

        for slot in self.slots.iter_mut().filter(|s| s.id == stack.id) {
            if remaining == 0 {
                break;
            }
            let space = max.saturating_sub(slot.count);
            let take = space.min(remaining);
            slot.count = slot.count.saturating_add(take);
            remaining -= take;
        }

        while remaining > 0 && self.slots.len() < self.capacity {
            let take = remaining.min(max);
            self.slots.push(ItemStack {
                id: stack.id.clone(),
                count: take,
                durability: stack.durability,
            });
            remaining -= take;
        }

        remaining
    }

    /// Add a stack, learning its stack_max from the database first.
    pub fn add_with_db(&mut self, stack: ItemStack, db: &ItemDb) -> u32 {
        self.set_stack_max(&stack.id, db.stack_max(&stack.id));
        self.add(stack)
    }

    /// Remove up to `count` of `id`, returning how many were actually removed.
    pub fn remove(&mut self, id: &str, count: u32) -> u32 {
        let mut want = count;
        let mut removed = 0u32;
        // Drain from the back so partial stacks coalesce in iteration order.
        for slot in self.slots.iter_mut().rev() {
            if want == 0 {
                break;
            }
            if slot.id != id || slot.count == 0 {
                continue;
            }
            let take = slot.count.min(want);
            slot.count -= take;
            want -= take;
            removed += take;
        }
        self.slots.retain(|s| s.count > 0);
        removed
    }

    pub fn count_of(&self, id: &str) -> u32 {
        self.slots
            .iter()
            .filter(|s| s.id == id)
            .fold(0u32, |acc, s| acc.saturating_add(s.count))
    }

    /// Total carry weight using definition encumbrance (unknown ids weigh 0).
    pub fn total_weight(&self, db: &ItemDb) -> f32 {
        self.slots
            .iter()
            .map(|s| {
                let w = db.get(&s.id).map(|d| d.weight).unwrap_or(0.0);
                w * s.count as f32
            })
            .sum()
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

/// Compatibility failure returned by `Equipment::equip`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquipError {
    /// The item's class has no equipment slot.
    NoSlotForClass(ItemClass),
}

/// The actor's equipped items, one per slot.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Equipment {
    pub melee: Option<ItemStack>,
    pub ranged: Option<ItemStack>,
    pub consumable: Option<ItemStack>,
    pub rucksack: Option<ItemStack>,
}

impl Equipment {
    pub fn new() -> Self {
        Equipment::default()
    }

    pub fn slot(&self, slot: EquipSlot) -> Option<&ItemStack> {
        match slot {
            EquipSlot::Melee => self.melee.as_ref(),
            EquipSlot::Ranged => self.ranged.as_ref(),
            EquipSlot::Consumable => self.consumable.as_ref(),
            EquipSlot::Rucksack => self.rucksack.as_ref(),
        }
    }

    pub fn slot_mut(&mut self, slot: EquipSlot) -> &mut Option<ItemStack> {
        match slot {
            EquipSlot::Melee => &mut self.melee,
            EquipSlot::Ranged => &mut self.ranged,
            EquipSlot::Consumable => &mut self.consumable,
            EquipSlot::Rucksack => &mut self.rucksack,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.melee.is_none()
            && self.ranged.is_none()
            && self.consumable.is_none()
            && self.rucksack.is_none()
    }

    /// Equip a stack into the slot required by its class. Returns the item it
    /// replaced, if the slot was occupied. Fails for classes with no slot.
    pub fn equip(
        &mut self,
        class: ItemClass,
        stack: ItemStack,
    ) -> Result<Option<ItemStack>, EquipError> {
        let slot = class.slot().ok_or(EquipError::NoSlotForClass(class))?;
        Ok(self.slot_mut(slot).replace(stack))
    }

    /// Remove and return the item in `slot`.
    pub fn unequip(&mut self, slot: EquipSlot) -> Option<ItemStack> {
        self.slot_mut(slot).take()
    }
}

/// State machine for searching a container. `tick(dt)` advances progress and
/// returns `true` once `search_time` seconds have elapsed.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchState {
    pub searched: bool,
    pub progress: f32,
    pub search_time: f32,
    pub loot: Vec<ItemStack>,
}

impl SearchState {
    pub fn new(search_time: f32) -> Self {
        SearchState {
            searched: false,
            progress: 0.0,
            search_time: search_time.max(0.0),
            loot: Vec::new(),
        }
    }

    /// A search that is already complete (e.g. an already-looted container).
    pub fn finished() -> Self {
        SearchState { searched: true, progress: 0.0, search_time: 0.0, loot: Vec::new() }
    }

    /// Advance the search clock. Returns `true` when finished. A zero
    /// `search_time` finishes on the first call (even with `dt == 0`).
    pub fn tick(&mut self, dt: f32) -> bool {
        if self.searched {
            return true;
        }
        self.progress += dt.max(0.0);
        if self.search_time <= 0.0 || self.progress >= self.search_time {
            self.progress = self.search_time.max(0.0);
            self.searched = true;
        }
        self.searched
    }

    pub fn is_complete(&self) -> bool {
        self.searched
    }

    /// Fraction searched, `0.0..=1.0`.
    pub fn progress_ratio(&self) -> f32 {
        if self.search_time <= 0.0 {
            1.0
        } else {
            (self.progress / self.search_time).clamp(0.0, 1.0)
        }
    }

    /// Reset for another search (keeps accumulated loot).
    pub fn reset(&mut self, search_time: f32) {
        self.searched = false;
        self.progress = 0.0;
        self.search_time = search_time.max(0.0);
    }

    /// Roll one weighted draw from `table` and append the result to `loot`.
    pub fn roll(&mut self, table: &LootTable, rng: &mut XorShiftRng) {
        self.loot.extend(roll_loot(table, rng));
    }
}

/// One weighted entry in a loot list: `id` yielding a `min..=max` stack.
#[derive(Debug, Clone, PartialEq)]
pub struct LootEntry {
    pub id: String,
    pub weight: u32,
    pub min: u32,
    pub max: u32,
}

impl LootEntry {
    pub fn new(id: impl Into<String>, weight: u32, min: u32, max: u32) -> Self {
        LootEntry { id: id.into(), weight, min, max }
    }
}

/// A weighted pool (`SearchItemList`) resolved by `roll_loot`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LootTable {
    pub entries: Vec<LootEntry>,
}

impl LootTable {
    pub fn new() -> Self {
        LootTable::default()
    }

    pub fn add(&mut self, entry: LootEntry) {
        self.entries.push(entry);
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// One weighted draw.
    pub fn roll(&self, rng: &mut XorShiftRng) -> Vec<ItemStack> {
        roll_loot(self, rng)
    }
}

/// Perform one weighted draw from `table`. Zero-weight entries are never
/// selected and an empty/all-zero table yields nothing. The stack count is
/// picked uniformly in `min..=max`.
pub fn roll_loot(table: &LootTable, rng: &mut XorShiftRng) -> Vec<ItemStack> {
    if table.entries.is_empty() {
        return Vec::new();
    }
    let weights: Vec<u32> = table.entries.iter().map(|e| e.weight).collect();
    let Some(i) = rng.weighted(&weights) else {
        return Vec::new();
    };
    let e = &table.entries[i];
    let lo = e.min.min(e.max);
    let hi = e.max.max(e.min);
    let count = if hi <= lo { lo } else { lo + rng.range(hi - lo + 1) };
    if count == 0 {
        Vec::new()
    } else {
        vec![ItemStack::new(e.id.clone(), count)]
    }
}

/// Perform `n` independent weighted draws.
pub fn roll_loot_n(table: &LootTable, rng: &mut XorShiftRng, n: u32) -> Vec<ItemStack> {
    let mut out = Vec::new();
    for _ in 0..n {
        out.extend(roll_loot(table, rng));
    }
    out
}

/// Small deterministic xorshift64* PRNG. Hand-rolled so loot rolls need no
/// external dependency and are reproducible across runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    /// Seed the generator. A zero seed is remapped to a non-zero constant.
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        XorShiftRng { state }
    }

    /// Current raw state (for save/restore).
    pub fn state(&self) -> u64 {
        self.state
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform value in `0..n`. `n == 0` yields 0.
    pub fn range(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as u32
        }
    }

    /// Weighted pick: index into `weights`. Returns `None` when all zero/empty.
    pub fn weighted(&mut self, weights: &[u32]) -> Option<usize> {
        let total: u64 = weights.iter().map(|&w| w as u64).sum();
        if total == 0 {
            return None;
        }
        let mut r = self.next_u64() % total;
        for (i, &w) in weights.iter().enumerate() {
            let w = w as u64;
            if w == 0 {
                continue;
            }
            if r < w {
                return Some(i);
            }
            r -= w;
        }
        None
    }
}

impl Default for XorShiftRng {
    fn default() -> Self {
        XorShiftRng::new(0x1234_5678_9ABC_DEF0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
<ItemData>
  <AmmoDefinitions>
    <AmmoDefinition EntityClass="bullet_9mm" MaximumStackSize="30" TakeStackSize="10">
      <InventoryItemData Id="AmmoBullet9mm" Encumbrance="0.02" Value="1">
        <NameText Text="9 mm"/>
        <DescriptionText Text="Small-caliber rounds."/>
      </InventoryItemData>
    </AmmoDefinition>
  </AmmoDefinitions>
  <WeaponDefinitions>
    <WeaponDefinition EntityClass="Machete" Expertise="EdgedLightBasic" DurabilityMax="160">
      <InventoryItemData Id="Machete" Encumbrance="1.5" Value="50">
        <NameText Text="Machete"/>
      </InventoryItemData>
      <WeaponBanner Name="Edged Light" Id="3"/>
    </WeaponDefinition>
  </WeaponDefinitions>
  <GunDefinitions>
    <GunDefinition EntityClass="Pistol_M9" Expertise="Pistol Basic" DurabilityMax="275">
      <InventoryItemData Id="Pistol_M9" Encumbrance="1.1" Value="100">
        <NameText Text="M9"/>
      </InventoryItemData>
    </GunDefinition>
  </GunDefinitions>
  <ConsumableDefinitions>
    <ConsumableDefinition EntityClass="Grenade_M67">
      <InventoryItemData Id="Grenade_M67" Encumbrance="0.6" Value="50">
        <NameText Text="M67 Grenade"/>
      </InventoryItemData>
    </ConsumableDefinition>
    <ConsumableDefinition EntityClass="HealthPack">
      <InventoryItemData Id="HealthPack" Encumbrance="0.3" Value="20">
        <NameText Text="Health Pack"/>
      </InventoryItemData>
    </ConsumableDefinition>
  </ConsumableDefinitions>
  <RucksackDefinitions>
    <RucksackDefinition>
      <InventoryItemData Id="FuelRucksack" Encumbrance="0.0" Value="0.0">
        <NameText Text="Gallons of Fuel"/>
      </InventoryItemData>
    </RucksackDefinition>
  </RucksackDefinitions>
  <WeaponExpertiseDefinitions>
    <WeaponExpertiseDefinition WeaponType="Edged">
      <NameText Text="Edged"/>
    </WeaponExpertiseDefinition>
  </WeaponExpertiseDefinitions>
</ItemData>
"#;

    fn db() -> ItemDb {
        ItemDb::load_from_xml(SAMPLE.as_bytes())
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn loads_defs_and_skips_non_items() {
        let db = db();
        assert_eq!(db.len(), 6, "5 item defs + fuel rucksack; expertise skipped");
        assert!(!db.is_empty());

        let ammo = db.get("AmmoBullet9mm").expect("ammo");
        assert_eq!(ammo.class, ItemClass::Ammo);
        assert_eq!(ammo.stack_max, 30);
        assert_eq!(ammo.entity_class(), "bullet_9mm");
        assert!(approx(ammo.weight, 0.02));
        assert!(approx(ammo.value, 1.0));
        assert_eq!(ammo.display_name(), "9 mm");
        assert_eq!(ammo.param("TakeStackSize"), Some("10"));
    }

    #[test]
    fn classifies_categories() {
        let db = db();
        assert_eq!(db.get("Machete").unwrap().class, ItemClass::Melee);
        assert_eq!(db.get("Pistol_M9").unwrap().class, ItemClass::Gun);
        assert_eq!(db.get("Grenade_M67").unwrap().class, ItemClass::Throwable);
        assert_eq!(db.get("HealthPack").unwrap().class, ItemClass::Consumable);
        assert_eq!(db.get("FuelRucksack").unwrap().class, ItemClass::Fuel);
        // Non-ammo defaults to a single-item stack.
        assert_eq!(db.get("Machete").unwrap().stack_max, 1);
    }

    #[test]
    fn class_from_str_and_name() {
        assert_eq!(ItemClass::from_str("melee"), ItemClass::Melee);
        assert_eq!(ItemClass::from_str("Weapon"), ItemClass::Melee);
        assert_eq!(ItemClass::from_str("gun"), ItemClass::Gun);
        assert_eq!(ItemClass::from_str("ammo"), ItemClass::Ammo);
        assert_eq!(ItemClass::from_str("consumable"), ItemClass::Consumable);
        assert_eq!(ItemClass::from_str("grenade"), ItemClass::Throwable);
        assert_eq!(ItemClass::from_str("rucksack"), ItemClass::Rucksack);
        assert_eq!(ItemClass::from_str("backpack"), ItemClass::Backpack);
        assert_eq!(ItemClass::from_str("fuel"), ItemClass::Fuel);
        assert_eq!(ItemClass::from_str("wat"), ItemClass::Other);
        for c in [
            ItemClass::Melee,
            ItemClass::Gun,
            ItemClass::Ammo,
            ItemClass::Consumable,
            ItemClass::Throwable,
            ItemClass::Rucksack,
            ItemClass::Backpack,
            ItemClass::Fuel,
            ItemClass::Other,
        ] {
            assert_eq!(ItemClass::from_str(c.name()), c);
        }
    }

    #[test]
    fn get_by_entity_class() {
        let db = db();
        assert_eq!(db.get_by_entity_class("bullet_9mm").unwrap().id, "AmmoBullet9mm");
        assert!(db.get_by_entity_class("nope").is_none());
    }

    #[test]
    fn add_merges_up_to_stack_max() {
        let mut inv = Inventory::new(4);
        inv.set_stack_max("AmmoBullet9mm", 30);
        assert_eq!(inv.add(ItemStack::new("AmmoBullet9mm", 20)), 0);
        assert_eq!(inv.slots().len(), 1);
        // 20 + 10 fills the first slot exactly; the other 10 opens slot 2.
        assert_eq!(inv.add(ItemStack::new("AmmoBullet9mm", 20)), 0);
        assert_eq!(inv.slots().len(), 2);
        assert_eq!(inv.slots()[0].count, 30);
        assert_eq!(inv.slots()[1].count, 10);
        assert_eq!(inv.count_of("AmmoBullet9mm"), 40);
    }

    #[test]
    fn leftover_returned_when_full() {
        let mut inv = Inventory::new(1);
        inv.set_stack_max("AmmoBullet9mm", 30);
        let leftover = inv.add(ItemStack::new("AmmoBullet9mm", 40));
        assert_eq!(leftover, 10);
        assert_eq!(inv.slots().len(), 1);
        assert_eq!(inv.count_of("AmmoBullet9mm"), 30);
    }

    #[test]
    fn add_with_db_uses_stack_max() {
        let db = db();
        let mut inv = Inventory::new(3);
        assert_eq!(inv.add_with_db(ItemStack::new("AmmoBullet9mm", 40), &db), 0);
        assert_eq!(inv.slots()[0].count, 30);
        assert_eq!(inv.count_of("AmmoBullet9mm"), 40);

        // Machete stack_max 1 -> each copy is its own slot.
        assert_eq!(inv.add_with_db(ItemStack::new("Machete", 1), &db), 0);
        assert_eq!(inv.count_of("Machete"), 1);
    }

    #[test]
    fn remove_and_count() {
        let mut inv = Inventory::new(4);
        inv.set_stack_max("AmmoBullet9mm", 30);
        inv.add(ItemStack::new("AmmoBullet9mm", 40));
        assert_eq!(inv.remove("AmmoBullet9mm", 15), 15);
        assert_eq!(inv.count_of("AmmoBullet9mm"), 25);
        // Removing more than present removes only what is there.
        assert_eq!(inv.remove("AmmoBullet9mm", 100), 25);
        assert_eq!(inv.count_of("AmmoBullet9mm"), 0);
        assert!(!inv.has("AmmoBullet9mm"));
        assert!(inv.is_empty());
        assert_eq!(inv.remove("nope", 5), 0);
    }

    #[test]
    fn total_weight_sums() {
        let db = db();
        let mut inv = Inventory::new(4);
        inv.add_with_db(ItemStack::new("AmmoBullet9mm", 30), &db); // 0.60
        inv.add_with_db(ItemStack::new("Machete", 1), &db); // 1.50
        inv.add_with_db(ItemStack::new("HealthPack", 2), &db); // 0.60
        assert!(approx(inv.total_weight(&db), 2.7), "got {}", inv.total_weight(&db));
    }

    #[test]
    fn equip_enforces_slot_compatibility() {
        let mut eq = Equipment::new();
        assert!(eq.is_empty());

        let replaced = eq.equip(ItemClass::Melee, ItemStack::new("Machete", 1)).unwrap();
        assert!(replaced.is_none());
        eq.equip(ItemClass::Gun, ItemStack::new("Pistol_M9", 1)).unwrap();
        eq.equip(ItemClass::Consumable, ItemStack::new("HealthPack", 1)).unwrap();
        eq.equip(ItemClass::Throwable, ItemStack::new("Grenade_M67", 1)).unwrap();
        eq.equip(ItemClass::Fuel, ItemStack::new("FuelRucksack", 1)).unwrap();

        assert_eq!(eq.melee.as_ref().unwrap().id, "Machete");
        assert_eq!(eq.ranged.as_ref().unwrap().id, "Pistol_M9");
        // Throwable shares the consumable slot, replacing the health pack.
        assert_eq!(eq.consumable.as_ref().unwrap().id, "Grenade_M67");
        assert_eq!(eq.rucksack.as_ref().unwrap().id, "FuelRucksack");

        // Replacing returns the previous occupant.
        let old = eq.equip(ItemClass::Melee, ItemStack::new("2x4", 1)).unwrap();
        assert_eq!(old.unwrap().id, "Machete");

        // Classes with no slot fail.
        assert_eq!(
            eq.equip(ItemClass::Ammo, ItemStack::new("AmmoBullet9mm", 5)),
            Err(EquipError::NoSlotForClass(ItemClass::Ammo))
        );
        assert_eq!(
            eq.equip(ItemClass::Other, ItemStack::new("Junk", 1)),
            Err(EquipError::NoSlotForClass(ItemClass::Other))
        );

        let taken = eq.unequip(EquipSlot::Ranged).unwrap();
        assert_eq!(taken.id, "Pistol_M9");
        assert!(eq.ranged.is_none());
        assert!(eq.unequip(EquipSlot::Ranged).is_none());
    }

    #[test]
    fn search_finishes_after_search_time() {
        let mut s = SearchState::new(2.0);
        assert!(!s.tick(1.0));
        assert!(!s.is_complete());
        assert!(approx(s.progress_ratio(), 0.5));
        assert!(s.tick(1.0));
        assert!(s.is_complete());
        // Further ticks stay finished.
        assert!(s.tick(10.0));
        assert!(approx(s.progress_ratio(), 1.0));
    }

    #[test]
    fn zero_time_search_finishes_immediately() {
        let mut s = SearchState::new(0.0);
        assert!(s.tick(0.0));
        assert!(SearchState::finished().is_complete());
    }

    #[test]
    fn reset_starts_a_new_search_but_keeps_loot() {
        let mut s = SearchState::new(1.0);
        s.loot.push(ItemStack::new("HealthPack", 1));
        assert!(s.tick(1.0));
        s.reset(3.0);
        assert!(!s.is_complete());
        assert!(approx(s.progress, 0.0));
        assert_eq!(s.loot.len(), 1);
    }

    #[test]
    fn loot_roll_is_deterministic() {
        let mut table = LootTable::new();
        table.add(LootEntry::new("AmmoBullet9mm", 10, 1, 1));
        table.add(LootEntry::new("HealthPack", 90, 2, 4));

        let mut a = XorShiftRng::new(42);
        let mut b = XorShiftRng::new(42);
        assert_eq!(roll_loot(&table, &mut a), roll_loot(&table, &mut b));

        // Counts stay inside [min,max] and ids are from the table.
        let mut rng = XorShiftRng::new(7);
        for _ in 0..100 {
            for stack in roll_loot_n(&table, &mut rng, 5) {
                assert!(stack.id == "AmmoBullet9mm" || stack.id == "HealthPack");
                if stack.id == "AmmoBullet9mm" {
                    assert_eq!(stack.count, 1);
                } else {
                    assert!((2..=4).contains(&stack.count), "count {}", stack.count);
                }
            }
        }
    }

    #[test]
    fn zero_weight_entries_never_roll() {
        let mut table = LootTable::new();
        table.add(LootEntry::new("Never", 0, 1, 1));
        table.add(LootEntry::new("Always", 5, 1, 1));
        let mut rng = XorShiftRng::new(1);
        for _ in 0..50 {
            let got = roll_loot(&table, &mut rng);
            assert_eq!(got.len(), 1);
            assert_eq!(got[0].id, "Always");
        }

        // All-zero / empty tables yield nothing.
        let only_zero = LootTable { entries: vec![LootEntry::new("X", 0, 1, 1)] };
        assert!(roll_loot(&only_zero, &mut rng).is_empty());
        assert!(roll_loot(&LootTable::new(), &mut rng).is_empty());
    }

    #[test]
    fn search_rolls_loot_into_state() {
        let mut table = LootTable::new();
        table.add(LootEntry::new("Food", 1, 6, 6));
        let mut s = SearchState::new(0.5);
        let mut rng = XorShiftRng::new(99);
        assert!(!s.tick(0.4));
        s.roll(&table, &mut rng);
        assert!(s.tick(0.1));
        assert_eq!(s.loot.len(), 1);
        assert_eq!(s.loot[0].id, "Food");
        assert_eq!(s.loot[0].count, 6);
    }

    #[test]
    fn rng_range_bounds() {
        let mut rng = XorShiftRng::new(123);
        assert_eq!(rng.range(0), 0);
        for _ in 0..200 {
            assert!(rng.range(10) < 10);
        }
    }

    #[test]
    fn malformed_xml_does_not_panic() {
        assert!(ItemDb::load_from_xml(b"").is_empty());
        assert!(ItemDb::load_from_xml(b"<not xml").is_empty());
        assert!(ItemDb::load_from_xml(b"<ItemData><WeaponDefinitions></ItemData>").is_empty());
    }

    #[test]
    fn parses_shipped_items_xml_when_available() {
        // Parity check against the real extraction when it is checked out next
        // to the repo; silently skipped otherwise so tests stay self-contained.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../extracted-game/gamedata/libs/class3/items/items.xml");
        let Ok(bytes) = std::fs::read(&path) else {
            return;
        };
        let db = ItemDb::load_from_xml(&bytes);
        assert_eq!(db.len(), 438, "distinct InventoryItemData ids");
        assert_eq!(db.get("AmmoBullet9mm").unwrap().class, ItemClass::Ammo);
        assert_eq!(db.get("AmmoBullet9mm").unwrap().stack_max, 30);
        assert_eq!(db.get("Pistol_M9").unwrap().class, ItemClass::Gun);
        assert_eq!(db.get("Machete").unwrap().class, ItemClass::Melee);
        assert_eq!(db.get("HealthPack").unwrap().class, ItemClass::Consumable);
        assert_eq!(db.get("Grenade_M67").unwrap().class, ItemClass::Throwable);
        assert_eq!(db.get("FuelRucksack").unwrap().class, ItemClass::Fuel);
        assert_eq!(db.get("BackpackSmall_v01").unwrap().class, ItemClass::Backpack);
        assert_eq!(db.get("Pistol_M9").unwrap().entity_class(), "Pistol_M9");
    }
}
