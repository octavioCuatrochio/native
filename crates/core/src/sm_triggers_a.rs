//! Player state-machine trigger types ported from their exe evaluators
//! (batch A). See `sm_runtime::TriggerCtx` and the `SmEnv` trait.

use crate::sm_runtime::{SmEnv, TriggerCtx};

/// Evaluate trigger `type_id` if this batch implements it.
pub fn eval(type_id: u8, operand: &[u8], ctx: &mut TriggerCtx, env: &dyn SmEnv) -> Option<bool> {
    let _ = (operand, &ctx, env);
    match type_id {
        _ => None,
    }
}
