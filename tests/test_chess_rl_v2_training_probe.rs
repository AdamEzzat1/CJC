//! Timing probe for the Chess RL v2.1 training driver.
//!
//! Runs a small batch of Adam training episodes in a single CJC-Lang program
//! and measures wall-clock to estimate the Phase D 500-episode budget.
//!
//! Gated behind `--ignored` so it never runs as part of the default test suite.

mod chess_rl_v2;

use chess_rl_v2::harness::{run, Backend};

#[test]
#[ignore = "timing probe, run manually with --ignored"]
fn training_probe_mir_10_episodes() {
    let body = r#"
        let w = init_weights();
        let adam = init_adam_state();
        let n_ep = 10;
        let max_moves = 20;
        let lr = 0.001;
        let ep = 0;
        while ep < n_ep {
            let result = train_one_episode_adam_full(w, adam, max_moves, lr, 1.0, 0.0 - 10.0, 999);
            w = result[0];
            adam = result[1];
            ep = ep + 1;
        }
        print(ep);
    "#;
    let t = std::time::Instant::now();
    let out = run(Backend::Mir, body, 11);
    let elapsed = t.elapsed();
    println!(
        "MIR training probe: 10 Adam episodes (max_moves=20) in {:.1}s ({:.2}s/episode)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / 10.0
    );
    assert_eq!(out[0].trim(), "10");
}

#[test]
#[ignore = "timing probe, run manually with --ignored"]
fn training_probe_20_episodes() {
    let body = r#"
        let w = init_weights();
        let adam = init_adam_state();
        let n_ep = 20;
        let max_moves = 30;
        let lr = 0.001;
        let ep = 0;
        let total_loss = 0.0;
        while ep < n_ep {
            let result = train_one_episode_adam_full(w, adam, max_moves, lr, 1.0, 0.0 - 10.0, 999);
            w = result[0];
            adam = result[1];
            let loss = result[2];
            total_loss = total_loss + loss;
            ep = ep + 1;
        }
        print(ep);
        print(total_loss);
    "#;
    let t = std::time::Instant::now();
    let out = run(Backend::Eval, body, 11);
    let elapsed = t.elapsed();
    println!(
        "training probe: 20 Adam episodes in {:.1}s ({:.2}s/episode)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / 20.0
    );
    assert_eq!(out[0].trim(), "20");
}
