//! Summarise a decoded ULSM state machine: counts, consistency checks, and the
//! states whose names match a filter with their clips and outgoing edges.
//! Usage: `sm_dump <file.sm> [name-filter ...]`

use sod_core::statemachine::parse_sm;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: sm_dump <file.sm> [filter ...]");
    let filters: Vec<String> = args.map(|a| a.to_ascii_lowercase()).collect();
    let bytes = std::fs::read(&path).expect("read .sm");
    let sm = parse_sm(&bytes).expect("parse_sm failed");

    let n = sm.states.len() as u32;
    let bad_targets = sm.transitions.iter().filter(|t| t.target >= n || t.source >= n).count();
    let bad_frags = sm
        .states
        .iter()
        .filter(|s| s.fragment.map(|f| f as usize >= sm.fragments.len()).unwrap_or(false))
        .count();
    let start = sm.states.get(sm.start_state as usize).map(|s| s.name.as_str()).unwrap_or("?");
    println!(
        "{}: {} states, {} fragments, {} blocks, {} transitions; start = {} ({})",
        sm.name,
        n,
        sm.fragments.len(),
        sm.blocks.len(),
        sm.transitions.len(),
        sm.start_state,
        start
    );
    println!("consistency: {bad_targets} out-of-range transition endpoints, {bad_frags} bad fragment indices");

    if filters.is_empty() {
        return;
    }
    for s in &sm.states {
        let lname = s.name.to_ascii_lowercase();
        if !filters.iter().any(|f| lname.contains(f.as_str())) {
            continue;
        }
        let clip = s
            .fragment
            .and_then(|f| sm.fragments.get(f as usize))
            .map(|f| format!("{} (speed {}, flags {:#x})", f.name, f.speed, f.flags))
            .unwrap_or_else(|| "-".into());
        println!("\n[{}] {}  clip: {}", s.index, s.name, clip);
        for t in sm.transitions.iter().filter(|t| t.source == s.index).take(8) {
            let target = sm.states.get(t.target as usize).map(|x| x.name.as_str()).unwrap_or("?");
            println!(
                "    -> {target}  (cond {}, weight {}, blend in {:.2} / out {:.2}, {:?})",
                t.condition, t.weight, t.blend_in, t.blend_out, t.list
            );
        }
    }
}
