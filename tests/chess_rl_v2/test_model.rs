//! Model tests: weight shapes, forward pass, action selection.

use crate::chess_rl_v2::harness::{parse_i64, run_parity};

/// Every weight tensor has the expected shape.
#[test]
fn weights_have_expected_shapes() {
    let body = r#"
        let w = init_weights();
        let sh0 = w[0].shape();
        let sh1 = w[1].shape();
        let sh2 = w[2].shape();
        let sh5 = w[5].shape();
        let sh7 = w[7].shape();
        let sh9 = w[9].shape();
        print(sh0[0]); print(sh0[1]);
        print(sh1[0]); print(sh1[1]);
        print(sh2[0]); print(sh2[1]);
        print(sh5[0]); print(sh5[1]);
        print(sh7[0]); print(sh7[1]);
        print(sh9[0]); print(sh9[1]);
    "#;
    let out = run_parity(body, 1);
    let ints: Vec<i64> = out.iter().map(|s| s.trim().parse::<i64>().unwrap()).collect();
    assert_eq!(&ints[0..2],  &[774, 48]);
    assert_eq!(&ints[2..4],  &[1, 48]);
    assert_eq!(&ints[4..6],  &[48, 48]);
    assert_eq!(&ints[6..8],  &[48, 64]);
    assert_eq!(&ints[8..10], &[48, 64]);
    assert_eq!(&ints[10..12],&[48, 1]);
}

/// `encode_state` returns a [1, 774] tensor.
#[test]
fn encoder_produces_correct_shape() {
    let body = r#"
        let s = init_state();
        let feat = encode_state(s);
        let sh = feat.shape();
        print(sh[0]);
        print(sh[1]);
    "#;
    let out = run_parity(body, 1);
    assert_eq!(parse_i64(&out), 1);
    assert_eq!(out[1].trim(), "774");
}

/// `forward_eager` produces from/to logits with shape [1, 64] and a finite value.
#[test]
fn forward_pass_shapes_and_finite() {
    let body = r#"
        let w = init_weights();
        let s = init_state();
        let feat = encode_state(s);
        let fwd = forward_eager(w, feat);
        let fl = fwd[0];
        let tl = fwd[1];
        let v = fwd[2];
        let s1 = fl.shape();
        let s2 = tl.shape();
        print(s1[0]); print(s1[1]);
        print(s2[0]); print(s2[1]);
        // Sanity: value is bounded by tanh -> |v| < 1
        let ok = 0;
        if v > 0.0 - 1.0 && v < 1.0 { ok = 1; }
        print(ok);
    "#;
    let out = run_parity(body, 7);
    let a: Vec<i64> = out[0..4].iter().map(|s| s.trim().parse().unwrap()).collect();
    assert_eq!(a, vec![1, 64, 1, 64]);
    assert_eq!(parse_i64(&out[4..]), 1, "value should satisfy -1 < v < 1");
}

/// `select_action` picks a legal action index strictly < num_legal_moves.
#[test]
fn select_action_returns_valid_index() {
    let body = r#"
        let w = init_weights();
        let s = init_state();
        let m = legal_moves(s);
        let num = len(m) / 2;
        let sel = select_action(w, s, m);
        let a = sel[0];
        let ok = 0;
        if a >= 0 && a < num { ok = 1; }
        print(ok);
        print(num);
    "#;
    let out = run_parity(body, 123);
    assert_eq!(parse_i64(&out), 1);
    assert_eq!(out[1].trim(), "20");
}

/// `select_action_greedy` is deterministic: same weights+state → same action.
#[test]
fn greedy_action_is_deterministic() {
    let body = r#"
        let w = init_weights();
        let s = init_state();
        let m = legal_moves(s);
        let a1 = select_action_greedy(w, s, m)[0];
        let a2 = select_action_greedy(w, s, m)[0];
        print(a1);
        print(a2);
    "#;
    let out = run_parity(body, 42);
    assert_eq!(out[0], out[1]);
}
