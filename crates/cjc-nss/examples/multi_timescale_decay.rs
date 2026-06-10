//! Phase 3b demo: multi-timescale memory under impulse + sustained
//! input.
//!
//! Demonstrates the *defining property* of multi-timescale memory:
//! the short and structural buffers respond to the same input but
//! decay at very different rates. After a single impulse, the short
//! buffer "forgets" within a few ticks while the structural buffer
//! still carries the signal 50+ ticks later.
//!
//! Run with:
//! ```text
//! cargo run --example multi_timescale_decay -p cjc-nss
//! ```

use cjc_nss::{MultiTimescaleConfig, MultiTimescaleEngine, NssSeed, Timescale};

fn main() {
    let cfg = MultiTimescaleConfig {
        per_scale_dim: 4,
        input_dim: 4,
        timescales: Timescale::ALL.to_vec(),
        init_scale: 0.5,
    };
    let engine = MultiTimescaleEngine::from_seed(cfg.clone(), NssSeed(2026)).unwrap();

    println!(
        "[config] {} timescales, per_scale_dim={}, total_state_dim={}",
        cfg.timescales.len(),
        cfg.per_scale_dim,
        cfg.total_state_dim(),
    );
    println!("[half-lives] expected exponential decay half-lives (ticks):");
    for ts in &cfg.timescales {
        println!(
            "    {:>10}  α={:.2}  half-life ≈ {:.1} ticks",
            ts.label(),
            ts.default_alpha(),
            ts.half_life_ticks(),
        );
    }

    // --- Experiment 1: unit impulse, then zero input for 60 ticks ---
    println!("\n[experiment 1] unit impulse at t=0, zero input thereafter");
    println!("    tick  | short    | medium   | long     | structural");
    println!("    ------+----------+----------+----------+-----------");
    let impulse = vec![1.0; 4];
    let zero = vec![0.0; 4];
    let mut h = engine.zero_state_concatenated();
    h = engine.step_concatenated(&h, &impulse).unwrap();
    let mags0 = engine.state_magnitudes(&h);
    print_mags(0, &mags0);
    for t in 1..=60 {
        h = engine.step_concatenated(&h, &zero).unwrap();
        if t == 1 || t == 5 || t == 10 || t == 20 || t == 40 || t == 60 {
            print_mags(t, &engine.state_magnitudes(&h));
        }
    }

    // --- Experiment 2: sustained input for 40 ticks ---
    println!("\n[experiment 2] sustained input (magnitude 0.5) for 40 ticks");
    println!("    tick  | short    | medium   | long     | structural");
    println!("    ------+----------+----------+----------+-----------");
    let sustained = vec![0.5; 4];
    let mut h = engine.zero_state_concatenated();
    for t in 1..=40 {
        h = engine.step_concatenated(&h, &sustained).unwrap();
        if t == 1 || t == 5 || t == 10 || t == 20 || t == 30 || t == 40 {
            print_mags(t, &engine.state_magnitudes(&h));
        }
    }

    // --- Experiment 3: spike + sustained (composite signal) ---
    println!("\n[experiment 3] spike at t=0, sustained low input thereafter");
    println!(
        "    interpretation: short captures the spike; structural integrates the sustained signal"
    );
    println!("    tick  | short    | medium   | long     | structural");
    println!("    ------+----------+----------+----------+-----------");
    let spike = vec![1.0; 4];
    let low = vec![0.1; 4];
    let mut h = engine.zero_state_concatenated();
    h = engine.step_concatenated(&h, &spike).unwrap();
    print_mags(0, &engine.state_magnitudes(&h));
    for t in 1..=30 {
        h = engine.step_concatenated(&h, &low).unwrap();
        if t == 1 || t == 3 || t == 5 || t == 10 || t == 20 || t == 30 {
            print_mags(t, &engine.state_magnitudes(&h));
        }
    }

    println!("\n[takeaway] short-term magnitude tracks the *recent* input;");
    println!("[takeaway] structural magnitude integrates the *cumulative* signal.");
}

fn print_mags(tick: u32, mags: &[(Timescale, f64)]) {
    println!(
        "    {:>4}  | {:.4}   | {:.4}   | {:.4}   | {:.4}",
        tick, mags[0].1, mags[1].1, mags[2].1, mags[3].1
    );
}
