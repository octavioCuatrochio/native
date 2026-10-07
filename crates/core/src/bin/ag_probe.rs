//! Load an AnimationGraph XML and show what the runtime selects for a sweep
//! of movement speeds. Usage: `ag_probe <graph.xml> [Input=value ...]`

use sod_core::ag_runtime::{Graph, GraphInstance, InputKind};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: ag_probe <graph.xml> [Input=value ...]");
    let g = Graph::parse(&std::fs::read(&path).expect("read")).expect("parse");
    let selectable = (0..g.states.len())
        .filter(|&i| g.resolved[i].flag("allowSelect") && g.resolved[i].flag("includeInGame"))
        .count();
    println!("{} inputs, {} states ({} selectable), {} links", g.inputs.len(), g.states.len(), selectable, g.links.len());
    for name in ["Action", "Stance", "Item", "Signal", "Vehicle"] {
        if let Some(i) = g.input(name) {
            if let InputKind::Key { keys } = &i.kind {
                println!("  {name}: default {:?}, {} keys (priority {})", i.default, keys.len(), i.priority);
            }
        }
    }
    let mut v = g.default_inputs();
    for kv in args {
        if let Some((k, val)) = kv.split_once('=') {
            match val.parse::<f32>() {
                Ok(n) => v.set_number(k, n),
                Err(_) => v.set_key(k, val),
            }
        }
    }
    let mut inst = GraphInstance::default();
    for speed in [0.0f32, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0] {
        v.set_number("PseudoSpeed", speed);
        let cands = g.candidates(&v);
        let name = |i: usize| g.states[i].id.clone();
        let before = inst.current;
        inst.update(&g, &v, false);
        let route: Vec<String> = inst.queue.iter().map(|&l| name(g.links[l].to)).collect();
        while !inst.queue.is_empty() {
            inst.update(&g, &v, true);
        }
        println!(
            "PseudoSpeed {speed:>3}: target {:?} ({} candidates) path from {:?}: {:?}",
            cands.first().map(|&i| name(i)),
            cands.len(),
            before.map(name),
            route
        );
    }
}
