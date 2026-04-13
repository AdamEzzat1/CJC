//! Parity tests: cjc-eval and cjc-mir-exec must produce byte-identical output
//! for the same program + seed.

use crate::chess_rl_v2::harness::{run, Backend};

fn assert_parity(body: &str, seed: u64, label: &str) {
    let eval_out = run(Backend::Eval, body, seed);
    let mir_out = run(Backend::Mir, body, seed);
    assert_eq!(
        eval_out, mir_out,
        "[{label}] eval vs mir-exec diverged at seed={seed}\n  eval: {eval_out:?}\n  mir:  {mir_out:?}"
    );
}

/// Parity: engine movegen.
#[test]
fn parity_engine_movegen() {
    let body = r#"
        let s = init_state();
        let m = legal_moves(s);
        print(len(m));
        let s2 = apply_move(s, 12, 28);
        let m2 = legal_moves(s2);
        print(len(m2));
        print(state_ep(s2));
    "#;
    assert_parity(body, 17, "engine movegen");
}

/// Parity: feature encoding.
#[test]
fn parity_feature_encoding() {
    let body = r#"
        let s = init_state();
        let f = encode_state(s);
        let sh = f.shape();
        print(sh[0]);
        print(sh[1]);
        // spot-check a few feature entries (piece planes)
        print(f.get([0, 0]));
        print(f.get([0, 700]));
        print(f.get([0, 772]));
    "#;
    assert_parity(body, 3, "feature encoding");
}

/// Parity: forward pass.
#[test]
fn parity_forward_pass() {
    let body = r#"
        let w = init_weights();
        let s = init_state();
        let feat = encode_state(s);
        let fwd = forward_eager(w, feat);
        let fl = fwd[0];
        let v = fwd[2];
        print(fl.get([0, 0]));
        print(fl.get([0, 7]));
        print(v);
    "#;
    assert_parity(body, 8, "forward pass");
}

/// Parity: a single training episode.
/// This is the strongest parity gate — it exercises engine, features, model,
/// rollout, GAE, and the full GradGraph-driven A2C update on both backends.
#[test]
fn parity_single_training_episode() {
    let body = r#"
        let w = init_weights();
        let r = train_one_episode(w, 12, 0.01);
        let new_w = r[0];
        let loss = r[1];
        let n = r[2];
        print(n);
        print(loss);
        print(new_w[0].get([0, 0]));
        print(new_w[0].get([1, 1]));
        print(new_w[5].get([0, 0]));
    "#;
    assert_parity(body, 21, "single training episode");
}
