//! Console variables: the defaults the original executable registers, with the
//! shipped config files and profile options applied in the engine's load order.

use crate::crypak::Vfs;
use std::collections::HashMap;

/// Lower-cased cvar name -> value string, as `System.GetCVar` would see it.
pub fn load(vfs: &Vfs) -> HashMap<String, String> {
    let _ = vfs;
    HashMap::new()
}
