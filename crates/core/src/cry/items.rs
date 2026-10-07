//! Items, loot tables and the container-search loop.
//!
//! This is the native replacement for the C++ item/search layer the game's
//! scripts talk to. It is deliberately self-contained: the item database is
//! the shipped `libs/class3/items/items.xml` (see [`crate::inventory`]), the
//! loot tables are the shipped `libs/class3/rts/search.xml`, and the two
//! script namespaces the game binds here are `ItemSystem` and the actor's
//! inventory surface.
//!
//! # Recovery
//!
//! * `NSRTSWorld::LoadSearchTable` @ `0x013ae560` (also named at `0x01858290`)
//!   parses `RTS/Search.xml` into item lists, resource definitions, resource
//!   caps, wants and per-site props — the structures below.
//! * `NSItemData::LoadItemData` @ `0x01865d2c` loads the item definitions.
//! * `NSInventory::Inventory::Load` @ `0x0189af48` loads an actor inventory.
//! * `NSRTSWorld::ScoutResourcePropTakeItem` (`0x0185963c`) and
//!   `ResourceNodeContainerMenuAction` (`0x0185953c`) are the interaction side.
//! * Search timing comes from the shipped `libs/config/meleeprototype.cfg`
//!   "Search" block: `sr_searchDurationNormal 10.0`, `sr_fastSpeedMultiplier
//!   4.0`, `sr_tapWindowDuration 0.25`, `sr_bonusStart 0.25`, `sr_bonusEnd
//!   0.75`, `sr_cooldownDuration 0.5`.
//!
//! # Loot model
//!
//! A searchable container is a `UL_Searchable` entity bound to a `Site` and a
//! `Prop` (the level/editor data supplies the pair; see [`spawn_searchable`]).
//! Rolling a prop yields:
//!   * `Items[min..=max]` physical items, each chosen by one weighted draw from
//!     one of the prop's `ItemList`s (aliases resolved recursively);
//!   * with `Resources.Odds` percent chance, `Resources[min..=max]` resource
//!     crates; each crate yields its own `Items[min..=max]` draws plus a
//!     `ResourceType` amount inside that type's [`ResourceCap`]s.
//!
//! `SearchItemList Min`/`Cap` are authored per-list yield budgets; they are
//! used here as the drawn item's stack size, clamped to the item's
//! `MaximumStackSize` (so ammo lists stack to 30, guns/melee/food to 1). This
//! keeps a single container to a handful of items, which is the observed
//! in-game behaviour, while still consuming the original weights.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use mlua::{Lua, Result as LuaResult, Table, Value};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

use super::host::{self, Ctx};
use super::world::EntityId;
use crate::crypak::Vfs;
use crate::inventory::{Equipment, Inventory, ItemDb, ItemStack, XorShiftRng};

// ---------------------------------------------------------------------------
// Search tuning (libs/config/meleeprototype.cfg, "Search" block)
// ---------------------------------------------------------------------------

/// `sr_searchDurationNormal` — seconds to search one container at normal speed.
pub const SR_SEARCH_DURATION_NORMAL: f32 = 10.0;
/// `sr_fastSpeedMultiplier` — hold sprint while searching to go this much faster.
pub const SR_FAST_SPEED_MULTIPLIER: f32 = 4.0;
/// `sr_tapWindowDuration` — window for the quick-search tap rhythm.
pub const SR_TAP_WINDOW_DURATION: f32 = 0.25;
/// `sr_bonusStart` — start of the random time-bonus window (fraction).
pub const SR_BONUS_START: f32 = 0.25;
/// `sr_bonusEnd` — end of the random time-bonus window (fraction).
pub const SR_BONUS_END: f32 = 0.75;
/// `sr_bonusMin` — smallest random time bonus (fraction of the duration).
pub const SR_BONUS_MIN: f32 = 0.5;
/// `sr_bonusMax` — largest random time bonus (fraction of the duration).
pub const SR_BONUS_MAX: f32 = 0.75;
/// `sr_cooldownDuration` — pause after a completed search before the next.
pub const SR_COOLDOWN_DURATION: f32 = 0.5;

/// Player's base backpack (the shipped small backpack, `BackpackSmall_v01`).
pub const BASE_BACKPACK_CAPACITY: usize = 8;

/// How close the player must be to a container to offer the search prompt.
pub const SEARCH_REACH: f32 = 3.5;

/// VFS path of the master item database (`NSItemData::LoadItemData`).
pub const ITEMS_XML: &str = "libs/class3/items/items.xml";
/// VFS path of the search/loot tables (`NSRTSWorld::LoadSearchTable`).
pub const SEARCH_XML: &str = "libs/class3/rts/search.xml";

// ---------------------------------------------------------------------------
// Loot-table data model (search.xml)
// ---------------------------------------------------------------------------

/// The five community resources (`ResourceCaps` order in search.xml).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Ammo,
    Food,
    Fuel,
    Materials,
    Medicine,
}

impl ResourceType {
    pub fn from_str(s: &str) -> Option<ResourceType> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ammo" => Some(ResourceType::Ammo),
            "food" => Some(ResourceType::Food),
            "fuel" => Some(ResourceType::Fuel),
            "materials" | "material" => Some(ResourceType::Materials),
            "medicine" | "meds" | "medical" => Some(ResourceType::Medicine),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ResourceType::Ammo => "Ammo",
            ResourceType::Food => "Food",
            ResourceType::Fuel => "Fuel",
            ResourceType::Materials => "Materials",
            ResourceType::Medicine => "Medicine",
        }
    }
}

/// One weighted entry of a `SearchItemList`: either a concrete item definition
/// or an alias to another list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListEntry {
    Item { id: String, weight: u32 },
    Alias { id: String, weight: u32 },
}

impl ListEntry {
    pub fn weight(&self) -> u32 {
        match self {
            ListEntry::Item { weight, .. } | ListEntry::Alias { weight, .. } => (*weight).max(1),
        }
    }
}

/// A `<SearchItemList>` weighted pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchList {
    pub id: String,
    pub min: u32,
    pub cap: u32,
    pub fallback: String,
    pub entries: Vec<ListEntry>,
}

/// Weighted resource-crate spawn on a prop (`<Resources>`).
#[derive(Debug, Clone, PartialEq)]
pub struct PropResources {
    pub odds: f32,
    pub min: u32,
    pub max: u32,
    /// `(resource definition id, weight)`.
    pub entries: Vec<(String, u32)>,
}

/// Item-list draws on a prop (`<Items>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropItems {
    pub min: u32,
    pub max: u32,
    pub lists: Vec<String>,
}

/// One `<Prop>` of a `<Site>`.
#[derive(Debug, Clone, PartialEq)]
pub struct Prop {
    pub prop_id: String,
    pub resources: Option<PropResources>,
    pub items: Option<PropItems>,
    pub wants: Vec<String>,
}

/// One `<Site>` (building archetype loot profile).
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub site_id: String,
    pub max_resources: u32,
    pub props: Vec<Prop>,
}

/// A `<ResourceDefinition>`: a resource crate and its contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceDef {
    pub id: String,
    pub resource_type: ResourceType,
    pub min: u32,
    pub max: u32,
    pub lists: Vec<String>,
}

/// Min/max yield for one resource type (the `<ResourceCaps>` table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceCap {
    pub min: u32,
    pub max: u32,
}

/// Parsed `search.xml`.
#[derive(Debug, Clone, Default)]
pub struct SearchTable {
    lists: HashMap<String, SearchList>,
    resource_defs: HashMap<String, ResourceDef>,
    caps: HashMap<ResourceType, ResourceCap>,
    sites: HashMap<String, Site>,
    want_lists: HashMap<String, Vec<String>>,
}

/// The result of rolling one container.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchYield {
    pub items: Vec<ItemStack>,
    pub resources: Vec<ResourceYield>,
}

impl SearchYield {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && self.resources.is_empty()
    }
}

/// One resource crate's contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceYield {
    pub resource: ResourceType,
    pub amount: u32,
}

// -- XML tree ---------------------------------------------------------------

struct XmlNode {
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<XmlNode>,
}

impl XmlNode {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    fn u32(&self, key: &str) -> Option<u32> {
        self.attr(key).and_then(|s| s.trim().parse::<u32>().ok())
    }

    fn f32(&self, key: &str) -> Option<f32> {
        self.attr(key).and_then(|s| s.trim().parse::<f32>().ok())
    }

    fn child(&self, name: &str) -> Option<&XmlNode> {
        self.children.iter().find(|c| c.name == name)
    }

    fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a XmlNode> {
        self.children.iter().filter(move |c| c.name == name)
    }
}

fn attrs_of(e: &BytesStart) -> Vec<(String, String)> {
    e.attributes()
        .flatten()
        .map(|a| {
            let k = String::from_utf8_lossy(a.key.as_ref()).to_string();
            let v = a
                .unescape_value()
                .map(|v| v.to_string())
                .unwrap_or_else(|_| String::from_utf8_lossy(&a.value).to_string());
            (k, v)
        })
        .collect()
}

fn name_of(e: &BytesStart) -> String {
    String::from_utf8_lossy(e.name().as_ref()).to_string()
}

/// Build a minimal element tree. Text nodes are ignored; entity scripts only
/// need tags, attributes and nesting. Malformed input yields whatever parsed
/// cleanly (never panics).
fn parse_tree(xml: &[u8]) -> Option<XmlNode> {
    let xml = xml.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(xml);
    let mut reader = Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut stack: Vec<XmlNode> = Vec::new();
    let mut root: Option<XmlNode> = None;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Eof) | Err(_) => break,
            Ok(Event::Start(e)) => {
                stack.push(XmlNode { name: name_of(&e), attrs: attrs_of(&e), children: Vec::new() });
            }
            Ok(Event::Empty(e)) => {
                let node = XmlNode { name: name_of(&e), attrs: attrs_of(&e), children: Vec::new() };
                match stack.last_mut() {
                    Some(p) => p.children.push(node),
                    None => root = Some(node),
                }
            }
            Ok(Event::End(_)) => {
                if let Some(node) = stack.pop() {
                    match stack.last_mut() {
                        Some(p) => p.children.push(node),
                        None => root = Some(node),
                    }
                }
            }
            _ => {}
        }
        buf.clear();
    }
    root
}

impl SearchTable {
    /// Parse `search.xml` bytes (the `NSRTSWorld::LoadSearchTable` equivalent).
    pub fn load_from_xml(xml: &[u8]) -> SearchTable {
        let Some(root) = parse_tree(xml) else { return SearchTable::default() };
        let mut t = SearchTable::default();

        if let Some(il) = root.child("ItemLists") {
            for l in il.children_named("SearchItemList") {
                let Some(id) = l.attr("Id") else { continue };
                let mut entries = Vec::new();
                for c in &l.children {
                    match c.name.as_str() {
                        "ItemDefinitionReference" => {
                            if let Some(eid) = c.attr("Id") {
                                entries.push(ListEntry::Item {
                                    id: eid.to_string(),
                                    weight: c.u32("Weight").unwrap_or(1).max(1),
                                });
                            }
                        }
                        "ItemListAlias" => {
                            if let Some(eid) = c.attr("Id") {
                                entries.push(ListEntry::Alias {
                                    id: eid.to_string(),
                                    weight: c.u32("Weight").unwrap_or(1).max(1),
                                });
                            }
                        }
                        _ => {}
                    }
                }
                t.lists.insert(
                    id.to_string(),
                    SearchList {
                        id: id.to_string(),
                        min: l.u32("Min").unwrap_or(0),
                        cap: l.u32("Cap").unwrap_or(0),
                        fallback: l.attr("Fallback").unwrap_or("").to_string(),
                        entries,
                    },
                );
            }
        }

        if let Some(rd) = root.child("ResourceDefinitions") {
            if let Some(caps) = rd.child("ResourceCaps") {
                for c in &caps.children {
                    let rt = match c.name.as_str() {
                        "AmmoCap" => Some(ResourceType::Ammo),
                        "FoodCap" => Some(ResourceType::Food),
                        "FuelCap" => Some(ResourceType::Fuel),
                        "MaterialsCap" => Some(ResourceType::Materials),
                        "MedicineCap" => Some(ResourceType::Medicine),
                        _ => None,
                    };
                    if let Some(rt) = rt {
                        t.caps.insert(
                            rt,
                            ResourceCap {
                                min: c.u32("Min").unwrap_or(0),
                                max: c.u32("Max").unwrap_or(0),
                            },
                        );
                    }
                }
            }
            for d in rd.children_named("ResourceDefinition") {
                let Some(id) = d.attr("Id") else { continue };
                let Some(rt) = d.attr("ResourceType").and_then(ResourceType::from_str) else { continue };
                let (min, max, lists) = d
                    .child("Items")
                    .map(|it| {
                        let min = it.u32("Min").unwrap_or(1);
                        let max = it.u32("Max").unwrap_or(min).max(min);
                        let lists = it.children_named("ItemList").filter_map(|l| l.attr("Id").map(str::to_string)).collect();
                        (min, max, lists)
                    })
                    .unwrap_or((0, 0, Vec::new()));
                t.resource_defs.insert(
                    id.to_string(),
                    ResourceDef { id: id.to_string(), resource_type: rt, min, max, lists },
                );
            }
        }

        if let Some(wl) = root.child("WantLists") {
            for w in wl.children_named("WantList") {
                let Some(id) = w.attr("Id") else { continue };
                let refs = w.children_named("WantDefinitionReference").filter_map(|r| r.attr("Id").map(str::to_string)).collect();
                t.want_lists.insert(id.to_string(), refs);
            }
        }

        if let Some(sites) = root.child("Sites") {
            for s in sites.children_named("Site") {
                let Some(site_id) = s.attr("SiteId") else { continue };
                let mut props = Vec::new();
                for p in s.children_named("Prop") {
                    let Some(prop_id) = p.attr("PropId") else { continue };
                    let resources = p.child("Resources").and_then(|r| {
                        let odds = r.f32("Odds").unwrap_or(0.0);
                        let min = r.u32("Min").unwrap_or(0);
                        let max = r.u32("Max").unwrap_or(min).max(min);
                        let entries: Vec<(String, u32)> = r
                            .children_named("Resource")
                            .filter_map(|c| c.attr("Id").map(|id| (id.to_string(), c.u32("Weight").unwrap_or(1).max(1))))
                            .collect();
                        if entries.is_empty() {
                            None
                        } else {
                            Some(PropResources { odds, min, max, entries })
                        }
                    });
                    let items = p.child("Items").and_then(|it| {
                        let lists: Vec<String> = it.children_named("ItemList").filter_map(|l| l.attr("Id").map(str::to_string)).collect();
                        if lists.is_empty() {
                            None
                        } else {
                            let min = it.u32("Min").unwrap_or(1);
                            let max = it.u32("Max").unwrap_or(min).max(min);
                            Some(PropItems { min, max, lists })
                        }
                    });
                    let wants = p
                        .child("Wants")
                        .map(|w| w.children_named("WantListReference").filter_map(|r| r.attr("Id").map(str::to_string)).collect())
                        .unwrap_or_default();
                    props.push(Prop { prop_id: prop_id.to_string(), resources, items, wants });
                }
                t.sites.insert(site_id.to_string(), Site { site_id: site_id.to_string(), max_resources: s.u32("MaxResources").unwrap_or(0), props });
            }
        }

        t
    }

    /// Load the shipped `libs/class3/rts/search.xml` from a VFS.
    pub fn load(vfs: &Vfs) -> SearchTable {
        vfs.read(SEARCH_XML).map(|b| SearchTable::load_from_xml(&b)).unwrap_or_default()
    }

    pub fn list(&self, id: &str) -> Option<&SearchList> {
        self.lists.get(id)
    }

    pub fn resource_def(&self, id: &str) -> Option<&ResourceDef> {
        self.resource_defs.get(id)
    }

    pub fn cap(&self, rt: ResourceType) -> Option<ResourceCap> {
        self.caps.get(&rt).copied()
    }

    pub fn site(&self, id: &str) -> Option<&Site> {
        self.sites.get(id)
    }

    pub fn prop(&self, site: &str, prop: &str) -> Option<&Prop> {
        self.site(site)?.props.iter().find(|p| p.prop_id == prop)
    }

    pub fn want_list(&self, id: &str) -> Option<&Vec<String>> {
        self.want_lists.get(id)
    }

    pub fn list_count(&self) -> usize {
        self.lists.len()
    }

    pub fn site_count(&self) -> usize {
        self.sites.len()
    }

    pub fn resource_def_count(&self) -> usize {
        self.resource_defs.len()
    }

    /// One weighted draw from a list, resolving aliases recursively. Returns
    /// the chosen item id (an `InventoryItemData` id or an entity class).
    pub fn pick(&self, list_id: &str, rng: &mut XorShiftRng) -> Option<String> {
        self.pick_inner(list_id, rng, 0).map(|(id, _)| id)
    }

    /// Like [`SearchTable::pick`] but also returns the authored `Cap` of the
    /// list the winning entry belongs to (the item's stack size).
    fn pick_inner(&self, list_id: &str, rng: &mut XorShiftRng, depth: usize) -> Option<(String, u32)> {
        if depth > 8 {
            return None;
        }
        let list = self.lists.get(list_id)?;
        if list.entries.is_empty() {
            return None;
        }
        let weights: Vec<u32> = list.entries.iter().map(ListEntry::weight).collect();
        let i = rng.weighted(&weights)?;
        match &list.entries[i] {
            ListEntry::Item { id, .. } => Some((id.clone(), list.cap)),
            // An alias resolves into the target list, so the target list's
            // `Cap` (not the alias's) is the yield budget.
            ListEntry::Alias { id, .. } => self.pick_inner(id, rng, depth + 1),
        }
    }

    /// Pick one of `lists` uniformly (the prop/resource `<Items>` pool) then one
    /// weighted entry inside it, returning the item id and the yielding list's
    /// authored `Cap`.
    fn pick_from_pool(&self, lists: &[String], rng: &mut XorShiftRng) -> Option<(String, u32)> {
        if lists.is_empty() {
            return None;
        }
        let idx = rng.range(lists.len() as u32) as usize;
        self.pick_inner(&lists[idx.min(lists.len() - 1)], rng, 0)
    }

    /// Roll the loot of one container: `site` + `prop` from `search.xml`.
    ///
    /// `db` canonicalises entity-class references back to `InventoryItemData`
    /// ids and supplies stack sizes.
    pub fn roll_prop(&self, site: &str, prop: &str, db: &ItemDb, rng: &mut XorShiftRng) -> SearchYield {
        let Some(p) = self.prop(site, prop) else { return SearchYield::default() };
        let mut out = SearchYield::default();

        if let Some(items) = &p.items {
            let count = roll_range(items.min, items.max, rng);
            for _ in 0..count {
                if let Some((raw, cap)) = self.pick_from_pool(&items.lists, rng) {
                    if let Some(stack) = self.make_stack(&raw, Some(cap), db, rng) {
                        out.items.push(stack);
                    }
                }
            }
        }

        if let Some(res) = &p.resources {
            if !res.entries.is_empty() && roll_odds(res.odds, rng) {
                let count = roll_range(res.min, res.max, rng);
                for _ in 0..count {
                    let weights: Vec<u32> = res.entries.iter().map(|(_, w)| *w).collect();
                    let Some(i) = rng.weighted(&weights) else { break };
                    let def_id = &res.entries[i].0;
                    self.roll_resource(def_id, db, rng, &mut out);
                }
            }
        }

        out
    }

    /// Roll one resource crate by `<ResourceDefinition>` id.
    pub fn roll_resource(&self, def_id: &str, db: &ItemDb, rng: &mut XorShiftRng, out: &mut SearchYield) {
        let Some(def) = self.resource_defs.get(def_id) else { return };
        let cap = self.caps.get(&def.resource_type).copied();
        let amount = cap.map(|c| roll_range(c.min, c.max, rng)).unwrap_or(0);
        out.resources.push(ResourceYield { resource: def.resource_type, amount });
        let count = roll_range(def.min, def.max, rng);
        for _ in 0..count {
            if let Some((raw, cap)) = self.pick_from_pool(&def.lists, rng) {
                if let Some(stack) = self.make_stack(&raw, Some(cap), db, rng) {
                    out.items.push(stack);
                }
            }
        }
    }

    /// Build an [`ItemStack`] for a search draw. Entity-class references are
    /// canonicalised to the item-definition id; the stack size is the list's
    /// authored `Cap` clamped to the item's `MaximumStackSize`.
    fn make_stack(&self, raw: &str, list_cap: Option<u32>, db: &ItemDb, _rng: &mut XorShiftRng) -> Option<ItemStack> {
        let (id, cap, stack_max) = match db.get(raw) {
            Some(d) => (d.id.clone(), list_cap.unwrap_or(1), d.stack_max.max(1)),
            None => match db.get_by_entity_class(raw) {
                Some(d) => (d.id.clone(), list_cap.unwrap_or(1), d.stack_max.max(1)),
                None => (raw.to_string(), list_cap.unwrap_or(1), 1),
            },
        };
        let count = cap.clamp(1, stack_max.max(1));
        Some(ItemStack::new(id, count))
    }

    /// [`SearchTable::make_stack`] for a caller that knows which list the pick
    /// came from (used to apply that list's `Cap`).
    pub fn make_stack_for_list(&self, list_id: &str, item: &str, db: &ItemDb, rng: &mut XorShiftRng) -> Option<ItemStack> {
        let cap = self.lists.get(list_id).map(|l| l.cap);
        self.make_stack(item, cap, db, rng)
    }
}

fn roll_range(min: u32, max: u32, rng: &mut XorShiftRng) -> u32 {
    let lo = min.min(max);
    let hi = max.max(min);
    if hi <= lo {
        lo
    } else {
        lo + rng.range(hi - lo + 1)
    }
}

/// `Odds` is authored as a percentage (`0..=100`); `<= 0` never fires.
fn roll_odds(odds: f32, rng: &mut XorShiftRng) -> bool {
    if odds <= 0.0 {
        return false;
    }
    if odds >= 100.0 {
        return true;
    }
    (rng.range(10_000) as f32) < odds * 100.0
}

// ---------------------------------------------------------------------------
// Equipment packs (libs/equipmentpacks/*.xml)
// ---------------------------------------------------------------------------

/// One `GiveItemPack` loadout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EquipPack {
    pub name: String,
    pub primary: String,
    pub items: Vec<String>,
    pub ammo: Vec<(String, u32)>,
}

impl EquipPack {
    pub fn num_items(&self) -> usize {
        self.items.len()
    }

    pub fn item(&self, index: usize) -> Option<&str> {
        self.items.get(index).map(String::as_str)
    }
}

/// Parse an `<EquipPack>` file.
pub fn parse_equip_pack(xml: &[u8]) -> Option<EquipPack> {
    let root = parse_tree(xml)?;
    if root.name != "EquipPack" {
        return None;
    }
    let mut pack = EquipPack {
        name: root.attr("name").unwrap_or("").to_string(),
        primary: root.attr("primary").unwrap_or("").to_string(),
        ..Default::default()
    };
    if let Some(items) = root.child("Items") {
        for c in &items.children {
            // Item elements are named after the item id (e.g. <Prop_Meds .../>).
            if c.attr("type").map(|t| t.eq_ignore_ascii_case("ItemGivable")).unwrap_or(true) {
                pack.items.push(c.name.clone());
            }
        }
    }
    if let Some(ammo) = root.child("Ammos") {
        for c in ammo.children_named("Ammo") {
            if let Some(name) = c.attr("name") {
                pack.ammo.push((name.to_string(), c.u32("amount").unwrap_or(0)));
            }
        }
    }
    Some(pack)
}

/// Load `libs/equipmentpacks/<name>.xml` from a VFS.
pub fn load_equip_pack(vfs: &Vfs, name: &str) -> Option<EquipPack> {
    let path = format!("libs/equipmentpacks/{}.xml", name.to_ascii_lowercase());
    vfs.read(&path).and_then(|b| parse_equip_pack(&b))
}

// ---------------------------------------------------------------------------
// Actor inventories / backpack limits
// ---------------------------------------------------------------------------

/// One actor's carry state: slot inventory plus equipment.
#[derive(Debug, Clone)]
pub struct ActorInventory {
    pub inv: Inventory,
    pub equipped: Equipment,
    pub capacity: usize,
    pub backpack: Option<String>,
}

impl ActorInventory {
    pub fn new(capacity: usize) -> Self {
        ActorInventory { inv: Inventory::new(capacity), equipped: Equipment::new(), capacity, backpack: None }
    }

    /// Equip a backpack item, adopting its authored `Capacity` (`BackpackDefinition`
    /// attribute; e.g. `BackpackSmall_v01` = 8, `BackpackLarge_v01` = 10).
    pub fn set_backpack(&mut self, id: &str, db: &ItemDb) {
        let cap = db
            .get(id)
            .and_then(|d| d.param("Capacity"))
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(BASE_BACKPACK_CAPACITY)
            .max(1);
        self.capacity = cap;
        self.inv.set_capacity(cap);
        self.backpack = Some(id.to_string());
    }

    pub fn add(&mut self, stack: ItemStack, db: &ItemDb) -> u32 {
        self.inv.add_with_db(stack, db)
    }

    /// Take an item the way the game does: weapons and packs occupy their
    /// equipment slot (freeing backpack space), everything else stacks in the
    /// backpack. Returns leftover that did not fit.
    pub fn acquire(&mut self, mut stack: ItemStack, db: &ItemDb) -> u32 {
        // Loadout / search references may be entity classes; canonicalise.
        let resolved = db
            .get(&stack.id)
            .or_else(|| db.get_by_entity_class(&stack.id))
            .map(|d| (d.id.clone(), d.class));
        if let Some((id, class)) = resolved {
            stack.id = id;
            if class.slot().is_some() {
                if let Ok(replaced) = self.equipped.equip(class, stack) {
                    return match replaced {
                        Some(old) => self.inv.add_with_db(old, db),
                        None => 0,
                    };
                }
                return 0;
            }
        }
        self.inv.add_with_db(stack, db)
    }

    pub fn used_slots(&self) -> usize {
        self.inv.slots().iter().filter(|s| !s.is_empty()).count()
    }

    pub fn total_weight(&self, db: &ItemDb) -> f32 {
        self.inv.total_weight(db)
    }

    /// HUD backpack line: `Backpack used/capacity`.
    pub fn summary(&self, db: &ItemDb) -> String {
        format!("Backpack {}/{} ({:.1} lb)", self.used_slots(), self.capacity, self.total_weight(db))
    }
}

impl Default for ActorInventory {
    fn default() -> Self {
        ActorInventory::new(BASE_BACKPACK_CAPACITY)
    }
}

/// Backpack slot count for an item id, falling back to the base backpack.
pub fn backpack_capacity(db: &ItemDb, backpack: &str) -> usize {
    db.get(backpack)
        .and_then(|d| d.param("Capacity"))
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(BASE_BACKPACK_CAPACITY)
        .max(1)
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

/// The item/search runtime: item DB + search tables + per-actor inventories.
///
/// The app holds one of these (shared with the Lua binds through `Rc`), the
/// headless test drives it directly.
pub struct ItemRuntime {
    pub db: ItemDb,
    pub search: SearchTable,
    pub rng: XorShiftRng,
    pub inventories: HashMap<u64, ActorInventory>,
    pub stockpile: HashMap<ResourceType, u32>,
    pub vfs: Rc<Vfs>,
}

impl ItemRuntime {
    pub fn new(db: ItemDb, search: SearchTable, vfs: Rc<Vfs>, seed: u64) -> Self {
        ItemRuntime {
            db,
            search,
            rng: XorShiftRng::new(seed),
            inventories: HashMap::new(),
            stockpile: HashMap::new(),
            vfs,
        }
    }

    /// Load the shipped item DB and search tables from a VFS.
    pub fn load(vfs: Rc<Vfs>) -> Self {
        let db = vfs.read(ITEMS_XML).map(|b| ItemDb::load_from_xml(&b)).unwrap_or_else(ItemDb::new);
        let search = SearchTable::load(&vfs);
        ItemRuntime::new(db, search, vfs, 0x5eed_1234_9abc_def0)
    }

    pub fn inventory(&self, actor: u64) -> Option<&ActorInventory> {
        self.inventories.get(&actor)
    }

    pub fn inventory_mut(&mut self, actor: u64) -> &mut ActorInventory {
        self.inventories.entry(actor).or_default()
    }

    pub fn ensure_inventory(&mut self, actor: u64, capacity: usize) {
        self.inventories.entry(actor).or_insert_with(|| ActorInventory::new(capacity));
    }

    /// Add one item to an actor, learning its stack size from the DB.
    pub fn give(&mut self, actor: u64, id: &str, count: u32) -> u32 {
        let db = self.db.clone();
        let inv = self.inventories.entry(actor).or_default();
        inv.acquire(ItemStack::new(id, count), &db)
    }

    /// Grant an equipment pack (`ItemSystem.GiveItemPack`). Returns what was
    /// added; leftover that did not fit is dropped, as the original does.
    pub fn grant_pack(&mut self, actor: u64, pack_name: &str) -> Vec<ItemStack> {
        let Some(pack) = load_equip_pack(&self.vfs, pack_name) else { return Vec::new() };
        let db = self.db.clone();
        let inv = self.inventories.entry(actor).or_default();
        let mut added = Vec::new();
        for id in pack.items {
            let stack = ItemStack::new(id, 1);
            inv.acquire(stack.clone(), &db);
            added.push(stack);
        }
        added
    }

    /// Roll a container and deposit the loot in the actor's inventory.
    pub fn complete_search(&mut self, actor: u64, site: &str, prop: &str) -> SearchYield {
        let db = self.db.clone();
        let yield_ = self.search.roll_prop(site, prop, &db, &mut self.rng);
        for r in &yield_.resources {
            *self.stockpile.entry(r.resource).or_insert(0) = self.stockpile.get(&r.resource).copied().unwrap_or(0) + r.amount;
        }
        let inv = self.inventories.entry(actor).or_default();
        for item in &yield_.items {
            inv.acquire(item.clone(), &db);
        }
        yield_
    }

    pub fn stockpile_of(&self, rt: ResourceType) -> u32 {
        self.stockpile.get(&rt).copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Search session (hold-to-search)
// ---------------------------------------------------------------------------

/// An in-progress or paused container search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchSession {
    /// The `UL_Searchable` entity being searched.
    pub target: u64,
    pub site: String,
    pub prop: String,
    pub duration: f32,
    pub progress: f32,
    pub fast: bool,
    pub done: bool,
}

impl SearchSession {
    pub fn begin(target: u64, site: impl Into<String>, prop: impl Into<String>) -> Self {
        SearchSession {
            target,
            site: site.into(),
            prop: prop.into(),
            duration: SR_SEARCH_DURATION_NORMAL,
            progress: 0.0,
            fast: false,
            done: false,
        }
    }

    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration.max(0.0);
        self
    }

    /// Advance the hold timer. `fast` applies `sr_fastSpeedMultiplier`.
    /// Returns `true` once the search completes.
    pub fn tick(&mut self, dt: f32, fast: bool) -> bool {
        if self.done {
            return true;
        }
        self.fast = fast;
        self.progress += dt.max(0.0) * if fast { SR_FAST_SPEED_MULTIPLIER } else { 1.0 };
        if self.progress >= self.duration {
            self.progress = self.duration;
            self.done = true;
        }
        self.done
    }

    pub fn ratio(&self) -> f32 {
        if self.duration <= 0.0 {
            1.0
        } else {
            (self.progress / self.duration).clamp(0.0, 1.0)
        }
    }

    pub fn remaining(&self) -> f32 {
        (self.duration - self.progress).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Spawning searchable containers
// ---------------------------------------------------------------------------

/// Spawn a real `UL_Searchable` entity through the script host and bind it to a
/// `search.xml` site/prop pair.
///
/// The shipped level has no authored searchables (the RTS layer generates them
/// at runtime from buildings). This is the native stand-in for that generation:
/// it uses the original `.ent` class and `ul_searchable.lua` state machine and
/// stores the site/prop binding on the instance so the search loop can roll it.
pub fn spawn_searchable(
    host: &mut host::ScriptHost,
    name: &str,
    pos: [f32; 3],
    site: &str,
    prop: &str,
) -> mlua::Result<EntityId> {
    let id = host.spawn_scripted("UL_Searchable", "scripts/entities/ul/ul_searchable.lua", name, pos)?;
    if let Some(inst) = host.ctx.instance(id) {
        let props: Table = inst.raw_get("Properties")?;
        props.raw_set("SiteId", site)?;
        props.raw_set("PropId", prop)?;
    }
    host.callback(id, "OnPropertyChange")?;
    Ok(id)
}

/// Read back the site/prop binding stored on a searchable instance.
pub fn searchable_binding(host: &host::ScriptHost, id: EntityId) -> Option<(String, String)> {
    let inst = host.ctx.instance(id)?;
    let props: Table = inst.raw_get("Properties").ok()?;
    let site: Option<String> = props.raw_get("SiteId").ok();
    let prop: Option<String> = props.raw_get("PropId").ok();
    match (site, prop) {
        (Some(s), Some(p)) if !s.is_empty() && !p.is_empty() => Some((s, p)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// ItemSystem binds
// ---------------------------------------------------------------------------

/// Install the `ItemSystem` namespace (`GiveItemPack`, `SetActorItem`, pack
/// queries). `rt` is shared with the caller so the native inventory state and
/// the scripts observe the same items.
pub fn install(lua: &Lua, ctx: &Ctx, rt: Rc<RefCell<ItemRuntime>>) -> LuaResult<()> {
    let itemsystem: Table = lua.globals().get("ItemSystem")?;

    let c = ctx.clone();
    let r = rt.clone();
    itemsystem.raw_set(
        "GiveItemPack",
        lua.create_function(move |_, (actor, pack, _drop, _clear): (Value, String, Value, Value)| {
            if let Some(actor) = host::value_to_id(&actor) {
                let _ = c; // vfs lives inside the runtime
                r.borrow_mut().grant_pack(actor as u64, &pack);
            }
            Ok(())
        })?,
    )?;

    let r = rt.clone();
    itemsystem.raw_set(
        "SetActorItem",
        lua.create_function(move |_, (actor, item, _equip): (Value, Value, Value)| {
            if let (Some(actor), Some(item)) = (host::value_to_id(&actor), value_to_string(&item)) {
                r.borrow_mut().give(actor as u64, &item, 1);
            }
            Ok(())
        })?,
    )?;

    let r = rt.clone();
    itemsystem.raw_set(
        "SetActorItemByName",
        lua.create_function(move |_, (actor, item, _equip): (Value, String, Value)| {
            if let Some(actor) = host::value_to_id(&actor) {
                r.borrow_mut().give(actor as u64, &item, 1);
            }
            Ok(())
        })?,
    )?;

    let r = rt.clone();
    itemsystem.raw_set(
        "GetPackPrimaryItem",
        lua.create_function(move |_, pack: String| {
            let primary = load_equip_pack(&r.borrow().vfs, &pack).map(|p| p.primary).unwrap_or_default();
            Ok(primary)
        })?,
    )?;

    let r = rt.clone();
    itemsystem.raw_set(
        "GetPackNumItems",
        lua.create_function(move |_, pack: String| {
            let n = load_equip_pack(&r.borrow().vfs, &pack).map(|p| p.num_items()).unwrap_or(0);
            Ok(n as i64)
        })?,
    )?;

    let r = rt.clone();
    itemsystem.raw_set(
        "GetPackItemByIndex",
        lua.create_function(move |_, (pack, index): (String, i64)| {
            let item = load_equip_pack(&r.borrow().vfs, &pack)
                .and_then(|p| p.item(index.max(0) as usize).map(str::to_string))
                .unwrap_or_default();
            Ok(item)
        })?,
    )?;

    for name in ["NextItem", "PrevItem"] {
        itemsystem.raw_set(name, lua.create_function(|_, _: mlua::Variadic<Value>| Ok(Value::Nil))?)?;
    }

    Ok(())
}

fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.to_string_lossy().to_string()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    const GAME: &str = "/home/battlestation/.local/share/Steam/steamapps/common/State of Decay YOSE";
    const EXTRACTED: &str = "/home/battlestation/Desktop/sod-re/extracted-game/gamedata";

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

    /// Real game paks when present, else the loose extracted tree.
    fn build_vfs() -> (Rc<Vfs>, &'static str) {
        let mut vfs = Vfs::new();
        let mut paks = Vec::new();
        collect_paks(Path::new(GAME), &mut paks);
        paks.sort();
        for p in &paks {
            let _ = vfs.mount(p);
        }
        if vfs.read(ITEMS_XML).is_some() {
            return (Rc::new(vfs), "game paks");
        }
        let mut loose = Vfs::new();
        if Path::new(EXTRACTED).join(ITEMS_XML).is_file() {
            loose.set_loose_root(EXTRACTED);
            return (Rc::new(loose), "loose extracted-game tree");
        }
        (Rc::new(loose), "EMPTY VFS (no source available!)")
    }

    const SAMPLE: &str = r#"
<Search>
  <ItemLists>
    <SearchItemList Cap="2" Fallback="" Id="Ammo.Tanner" Min="2">
      <ItemDefinitionReference Id="AmmoBullet9mm" Weight="64" />
      <ItemDefinitionReference Id="AmmoBullet45cal" Weight="75" />
    </SearchItemList>
    <SearchItemList Cap="1" Fallback="" Id="Melee.Tanner" Min="1">
      <ItemDefinitionReference Id="Machete" Weight="32" />
    </SearchItemList>
    <SearchItemList Cap="4" Fallback="" Id="GrabBag" Min="4">
      <ItemListAlias Id="Melee.Tanner" />
      <ItemDefinitionReference Id="HealthPack" Weight="50" />
    </SearchItemList>
  </ItemLists>
  <ResourceDefinitions>
    <ResourceCaps>
      <AmmoCap Max="60" Min="40" />
      <FoodCap Max="200" Min="180" />
      <FuelCap Max="60" Min="40" />
      <MaterialsCap Max="130" Min="110" />
      <MedicineCap Max="80" Min="60" />
    </ResourceCaps>
    <ResourceDefinition Id="Ammo.Basic" ResourceType="Ammo">
      <NameText Text="Case of Ammunition" />
      <Items Max="6" Min="2"><ItemList Id="Ammo.Tanner" /></Items>
    </ResourceDefinition>
  </ResourceDefinitions>
  <Sites>
    <Site SiteId="Any" MaxResources="2">
      <Prop PropId="Cabinet">
        <Resources Min="0" Max="1" Odds="80">
          <Resource Id="Ammo.Basic" Weight="25" />
        </Resources>
        <Items Min="1" Max="1"><ItemList Id="GrabBag" /></Items>
      </Prop>
      <Prop PropId="Bookcase">
        <Resources Min="0" Max="1" Odds="0" />
        <Items Min="1" Max="2"><ItemList Id="Melee.Tanner" /></Items>
      </Prop>
    </Site>
  </Sites>
</Search>
"#;

    #[test]
    fn parses_search_schema_with_aliases_and_resources() {
        let t = SearchTable::load_from_xml(SAMPLE.as_bytes());
        assert_eq!(t.list_count(), 3);
        assert_eq!(t.site_count(), 1);
        assert_eq!(t.resource_def_count(), 1);

        let ammo = t.list("Ammo.Tanner").expect("Ammo.Tanner");
        assert_eq!(ammo.min, 2);
        assert_eq!(ammo.cap, 2);
        assert_eq!(ammo.entries.len(), 2);

        let grab = t.list("GrabBag").unwrap();
        assert!(grab.entries.iter().any(|e| matches!(e, ListEntry::Alias { id, .. } if id == "Melee.Tanner")));

        let cab = t.prop("Any", "Cabinet").expect("Cabinet");
        assert_eq!(cab.items.as_ref().unwrap().lists, vec!["GrabBag".to_string()]);
        assert_eq!(cab.resources.as_ref().unwrap().odds, 80.0);

        let cap = t.cap(ResourceType::Ammo).unwrap();
        assert_eq!((cap.min, cap.max), (40, 60));

        let bookcase = t.prop("Any", "Bookcase").unwrap();
        assert!(bookcase.resources.is_none(), "zero-odds empty Resources must not yield crates");
    }

    #[test]
    fn malformed_input_is_empty_not_panic() {
        assert_eq!(SearchTable::load_from_xml(b"").list_count(), 0);
        assert_eq!(SearchTable::load_from_xml(b"<not xml").site_count(), 0);
        assert_eq!(SearchTable::load_from_xml(b"<Search><Sites>").list_count(), 0);
        assert!(parse_equip_pack(b"<NotAPack/>").is_none());
    }

    #[test]
    fn roll_is_deterministic_and_respects_weights() {
        let t = SearchTable::load_from_xml(SAMPLE.as_bytes());
        let db = ItemDb::new();
        let mut a = XorShiftRng::new(7);
        let mut b = XorShiftRng::new(7);
        assert_eq!(t.roll_prop("Any", "Bookcase", &db, &mut a), t.roll_prop("Any", "Bookcase", &db, &mut b));

        let mut rng = XorShiftRng::new(11);
        for _ in 0..50 {
            let y = t.roll_prop("Any", "Cabinet", &db, &mut rng);
            assert!(!y.items.is_empty(), "Cabinet always yields at least one item");
            for it in &y.items {
                // Cabinet's own list (GrabBag -> Melee.Tanner/HealthPack) plus
                // the Ammo.Basic crate's contents (Ammo.Tanner).
                assert!(
                    ["Machete", "HealthPack", "AmmoBullet9mm", "AmmoBullet45cal"].contains(&it.id.as_str()),
                    "unexpected {}",
                    it.id
                );
            }
            if let Some(r) = y.resources.first() {
                assert_eq!(r.resource, ResourceType::Ammo);
                assert!((40..=60).contains(&r.amount), "amount {}", r.amount);
            }
        }
    }

    #[test]
    fn alias_resolution_and_cycle_guard() {
        let t = SearchTable::load_from_xml(
            br#"<Search><ItemLists>
              <SearchItemList Id="A" Min="1" Cap="1"><ItemListAlias Id="B" /></SearchItemList>
              <SearchItemList Id="B" Min="1" Cap="1"><ItemListAlias Id="A" /></SearchItemList>
              <SearchItemList Id="Leaf" Min="1" Cap="1"><ItemDefinitionReference Id="X" Weight="1" /></SearchItemList>
            </ItemLists></Search>"#,
        );
        let mut rng = XorShiftRng::new(1);
        assert!(t.pick("A", &mut rng).is_none(), "cycle must terminate with None");
        assert_eq!(t.pick("Leaf", &mut rng).as_deref(), Some("X"));
    }

    #[test]
    fn equipment_pack_parses() {
        let pack = parse_equip_pack(
            br#"<EquipPack name="DefaultPlayer" primary="Fists">
              <Items><Fists type="ItemGivable"/><Hammer type="ItemGivable"/><Journal type="ItemGivable"/></Items>
              <Ammos><Ammo name="bullet" amount="0"/></Ammos>
            </EquipPack>"#,
        )
        .expect("pack");
        assert_eq!(pack.name, "DefaultPlayer");
        assert_eq!(pack.primary, "Fists");
        assert_eq!(pack.num_items(), 3);
        assert_eq!(pack.item(1), Some("Hammer"));
    }

    #[test]
    fn search_session_honours_authored_duration_and_fast_multiplier() {
        let mut slow = SearchSession::begin(1, "Any", "Cabinet");
        assert_eq!(slow.duration, SR_SEARCH_DURATION_NORMAL);
        assert!(!slow.tick(9.9, false));
        assert!(slow.tick(0.1, false));
        assert!(slow.done);
        assert!((slow.ratio() - 1.0).abs() < 1e-4);

        let mut fast = SearchSession::begin(1, "Any", "Cabinet");
        assert!(!fast.tick(2.4, true));
        assert!(fast.tick(0.1, true), "4x fast search finishes 10s in ~2.5s");
    }

    #[test]
    fn inventory_backpack_capacity_and_summary() {
        let mut db = ItemDb::new();
        let mut def = crate::inventory::ItemDef {
            id: "BackpackSmall_v01".into(),
            class: crate::inventory::ItemClass::Backpack,
            weight: 1.6,
            stack_max: 1,
            value: 10.0,
            params: HashMap::new(),
        };
        def.params.insert("Capacity".into(), "8".into());
        db.insert(def);

        let mut ai = ActorInventory::new(BASE_BACKPACK_CAPACITY);
        ai.set_backpack("BackpackSmall_v01", &db);
        assert_eq!(ai.capacity, 8);
        assert!(ai.summary(&db).starts_with("Backpack 0/8"));
        ai.add(ItemStack::new("Machete", 1), &db);
        assert_eq!(ai.used_slots(), 1);
    }

    #[test]
    fn parses_shipped_search_xml_when_available() {
        let (vfs, source) = build_vfs();
        if source.contains("EMPTY") {
            return;
        }
        let t = SearchTable::load(&vfs);
        assert!(t.list_count() > 60, "{} lists", t.list_count());
        assert!(t.site_count() > 40, "{} sites", t.site_count());
        assert!(t.resource_def_count() > 20);
        assert!(t.prop("Any", "Cabinet").is_some());
        assert!(t.prop("Any", "CoolerChest").is_some());
        assert!(t.list("Ammo.Tanner").is_some());
        assert!(t.list("Resource.Ammo.Common").is_some());
        eprintln!("[items] search.xml source={source}: {} lists, {} sites, {} resource defs", t.list_count(), t.site_count(), t.resource_def_count());
    }

    /// Level-boot style verification: run the real `UL_Searchable` script on a
    /// container placed at `RTSPoint_PlayerStart`, search a real prop from the
    /// shipped tables, flip the script state to `Searched` and print the loot.
    #[test]
    fn searches_real_container_near_player_start() {
        use crate::entity::EntityRuntime;
        use crate::tagpoints::TagPoints;

        let (vfs, source) = build_vfs();
        if source.contains("EMPTY") {
            eprintln!("[search test] no game data available; skipping");
            return;
        }

        let player_start = vfs
            .read("level/tag_points_win.markup")
            .and_then(|b| TagPoints::parse(&b))
            .and_then(|tp| tp.get("RTSPoint_PlayerStart"))
            .unwrap_or([3246.8, 3256.2, 217.1]);

        // Real entity + real script: NotSearched on spawn.
        let entities = EntityRuntime::new(vfs.clone()).expect("EntityRuntime");
        let id = entities
            .spawn("scripts/entities/ul/ul_searchable.lua", "UL_Searchable")
            .expect("spawn UL_Searchable");
        let pos = [player_start[0] + 1.5, player_start[1], player_start[2]];
        entities.set_position(id, pos).expect("set pos");
        assert_eq!(entities.get_state(id).expect("state"), "NotSearched");

        // Search a real prop (guaranteed item yield) from the shipped `Any` site.
        let mut rt = ItemRuntime::load(vfs.clone());
        let actor = 42u64;
        rt.ensure_inventory(actor, BASE_BACKPACK_CAPACITY);

        let mut session = SearchSession::begin(id as u64, "Any", "Crate");
        let mut done = false;
        for _ in 0..200 {
            done = session.tick(0.1, false);
            if done {
                break;
            }
        }
        assert!(done, "10s search should finish after 100 ticks");

        let loot = rt.complete_search(actor, "Any", "Crate");

        // The original drives this through the flowgraph `Searched` input,
        // which calls `UL_Searchable.Event_Searched` -> GotoState("Searched").
        entities.goto_state(id, "Searched").expect("goto Searched");
        assert_eq!(entities.get_state(id).expect("state"), "Searched");

        let items: Vec<String> = loot.items.iter().map(|s| format!("{}x{}", s.id, s.count)).collect();
        let resources: Vec<String> = loot.resources.iter().map(|r| format!("{}+{}", r.resource.name(), r.amount)).collect();
        eprintln!(
            "[search test] source={source} PlayerStart={player_start:?} container@{pos:?} items={items:?} resources={resources:?}",
        );
        assert!(!loot.is_empty(), "a real Crate container must yield loot");

        let inv = rt.inventory(actor).expect("inventory");
        eprintln!("[search test] {}", inv.summary(&rt.db));
        assert!(
            !inv.inv.is_empty() || !inv.equipped.is_empty(),
            "loot must land in the backpack or an equipment slot"
        );
    }

    #[test]
    fn known_entity_id_is_some() {
        // `host::EntityId` alias should be the world id type.
        let _: EntityId = 1;
    }
}
