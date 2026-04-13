//! Training tests: one-episode smoke, determinism, weight movement.

use crate::chess_rl_v2::harness::{parse_i64, run, run_parity, Backend};

/// A single episode of training finishes, returns a finite loss and a move
/// count in the expected range.
#[test]
fn single_episode_smoke() {
    let body = r#"
        let w = init_weights();
        let result = train_one_episode(w, 40, 0.01);
        let w2 = result[0];
        let loss = result[1];
        let n = result[2];
        print(n);
        // Loss is a plain f64; just confirm it's finite by checking bounds.
        let finite = 0;
        if loss > 0.0 - 1.0e9 && loss < 1.0e9 { finite = 1; }
        print(finite);
        // Confirm the episode returned a full weight list (11 slots).
        print(len(w2));
    "#;
    let out = run(Backend::Eval, body, 7);
    let n = parse_i64(&out);
    assert!(n > 0 && n <= 41, "n_moves should be >0 and bounded, got {n}");
    assert_eq!(out[1].trim(), "1", "loss should be finite");
    assert_eq!(out[2].trim(), "11", "weights array should have 11 slots");
}

/// Training is deterministic: same seed + same init → identical final scalar
/// summary. We use a single episode rather than many to keep the test fast.
#[test]
fn training_is_deterministic() {
    let body = r#"
        let w1 = init_weights();
        let r1 = train_one_episode(w1, 20, 0.01);
        let new_w1 = r1[0];
        // Print one element of W1 and the episode move count.
        print(new_w1[0].get([0, 0]));
        print(r1[2]);
    "#;
    let a = run(Backend::Eval, body, 99);
    let b = run(Backend::Eval, body, 99);
    assert_eq!(a, b, "same seed should produce bit-identical training");
}

/// Weights actually change after an update (non-trivial gradient).
///
/// We probe the value-head bias `bv` which sits directly above the tanh
/// value output and always receives gradient regardless of input features.
/// (Probing e.g. `W1[0,0]` would be fragile because feature dim 0 is
/// "my pawn on a1", which is always 0 at the starting position, so that
/// weight has a structurally-zero gradient.)
#[test]
fn weights_change_after_update() {
    let body = r#"
        let w = init_weights();
        let before_bv = w[10].get([0, 0]);
        let before_b1 = w[1].get([0, 0]);
        let r = train_one_episode(w, 30, 0.05);
        let new_w = r[0];
        let after_bv = new_w[10].get([0, 0]);
        let after_b1 = new_w[1].get([0, 0]);
        let d_bv = after_bv - before_bv;
        let d_b1 = after_b1 - before_b1;
        let nonzero = 0;
        if d_bv > 1.0e-12 || d_bv < 0.0 - 1.0e-12 { nonzero = 1; }
        if d_b1 > 1.0e-12 || d_b1 < 0.0 - 1.0e-12 { nonzero = nonzero + 10; }
        print(nonzero);
    "#;
    let out = run(Backend::Eval, body, 42);
    // Expect both bv and b1 to move: 1 (bv) + 10 (b1) = 11
    let code = parse_i64(&out);
    assert!(code >= 1, "at least one weight should move, got code {code}");
    assert_eq!(code, 11, "both value-head bias and trunk bias should update");
}

/// Evaluation vs random baseline runs and returns plausible counts.
#[test]
fn eval_vs_random_runs() {
    let body = r#"
        let w = init_weights();
        let r = eval_vs_random(w, 2, 30);
        // [wins, draws, losses]
        print(r[0] + r[1] + r[2]);
    "#;
    // Only run the eval backend here; MIR runs second inside run_parity which
    // would double the cost. Keep training smoke tests eval-only; parity tests
    // live in test_parity.rs.
    let out = run(Backend::Eval, body, 3);
    assert_eq!(parse_i64(&out), 2, "total games should equal n_games");
}

// ==========================================================================
// ============== ADAM OPTIMIZER (Phase B1)
// ==========================================================================

/// Adam state initializes with the right shape: 11-slot m and v lists, step=0.
#[test]
fn adam_init_shape() {
    let body = r#"
        let st = init_adam_state();
        let m = st[0];
        let v = st[1];
        let step = st[2];
        print(len(m));
        print(len(v));
        print(step);
        // Probe a few moment shapes.
        let m_W1_sh = m[0].shape();
        print(m_W1_sh[0]);
        print(m_W1_sh[1]);
        let v_bv_sh = v[10].shape();
        print(v_bv_sh[0]);
        print(v_bv_sh[1]);
    "#;
    let out = run(Backend::Eval, body, 1);
    assert_eq!(out[0].trim(), "11", "m should have 11 slots");
    assert_eq!(out[1].trim(), "11", "v should have 11 slots");
    assert_eq!(out[2].trim(), "0", "step should start at 0");
    assert_eq!(out[3].trim(), "774", "m_W1 rows");
    assert_eq!(out[4].trim(), "48", "m_W1 cols");
    assert_eq!(out[5].trim(), "1", "v_bv rows");
    assert_eq!(out[6].trim(), "1", "v_bv cols");
}

/// Adam state buffers start at zero.
#[test]
fn adam_init_zero() {
    let body = r#"
        let st = init_adam_state();
        let m = st[0];
        let v = st[1];
        // Sample a couple of entries to confirm zero init.
        print(m[0].get([0, 0]));
        print(m[10].get([0, 0]));
        print(v[0].get([0, 0]));
        print(v[10].get([0, 0]));
    "#;
    let out = run(Backend::Eval, body, 1);
    for line in &out {
        let val: f64 = line.trim().parse().unwrap();
        assert_eq!(val, 0.0, "all moment buffers should init to 0.0, got {val}");
    }
}

/// One Adam training episode runs end-to-end and returns updated state.
#[test]
fn adam_single_episode_smoke() {
    let body = r#"
        let w = init_weights();
        let st = init_adam_state();
        let r = train_one_episode_adam(w, st, 30, 0.01);
        let new_w = r[0];
        let new_st = r[1];
        let loss = r[2];
        let n = r[3];
        print(n);
        print(len(new_w));
        // Step counter should advance to 1 after one update.
        print(new_st[2]);
        // Loss is finite.
        let finite = 0;
        if loss > 0.0 - 1.0e9 && loss < 1.0e9 { finite = 1; }
        print(finite);
    "#;
    let out = run(Backend::Eval, body, 7);
    let n = parse_i64(&out);
    assert!(n > 0, "should record at least one step, got {n}");
    assert_eq!(out[1].trim(), "11", "new_weights should be 11 slots");
    assert_eq!(out[2].trim(), "1", "step counter should advance to 1");
    assert_eq!(out[3].trim(), "1", "loss should be finite");
}

/// Adam moment buffers actually update after one episode (not all zero).
/// Probes m and v on the value-head bias `bv`, which always receives gradient.
#[test]
fn adam_state_evolves() {
    let body = r#"
        let w = init_weights();
        let st = init_adam_state();
        let r = train_one_episode_adam(w, st, 30, 0.01);
        let new_st = r[1];
        let new_m = new_st[0];
        let new_v = new_st[1];
        // bv slot is index 10.
        let m_bv = new_m[10].get([0, 0]);
        let v_bv = new_v[10].get([0, 0]);
        let m_changed = 0;
        let v_changed = 0;
        if m_bv > 1.0e-12 || m_bv < 0.0 - 1.0e-12 { m_changed = 1; }
        if v_bv > 1.0e-15 { v_changed = 10; }
        print(m_changed + v_changed);
    "#;
    let out = run(Backend::Eval, body, 42);
    // m can be ±, v is non-negative; both should be nonzero -> 1 + 10 = 11
    assert_eq!(parse_i64(&out), 11, "both m_bv and v_bv should evolve");
}

/// Adam training is deterministic with the same seed.
#[test]
fn adam_training_is_deterministic() {
    let body = r#"
        let w = init_weights();
        let st = init_adam_state();
        let r = train_one_episode_adam(w, st, 20, 0.01);
        let new_w = r[0];
        let new_st = r[1];
        print(new_w[10].get([0, 0]));
        print(new_st[0][10].get([0, 0]));
        print(new_st[1][10].get([0, 0]));
        print(new_st[2]);
    "#;
    let a = run(Backend::Eval, body, 99);
    let b = run(Backend::Eval, body, 99);
    assert_eq!(a, b, "Adam should be bit-deterministic with same seed");
}

/// Two Adam steps advance the step counter to 2 and continue updating moments.
#[test]
fn adam_two_steps_advance() {
    let body = r#"
        let w = init_weights();
        let st = init_adam_state();
        let r1 = train_one_episode_adam(w, st, 20, 0.01);
        let w1 = r1[0];
        let st1 = r1[1];
        let r2 = train_one_episode_adam(w1, st1, 20, 0.01);
        let st2 = r2[1];
        print(st1[2]);
        print(st2[2]);
    "#;
    let out = run(Backend::Eval, body, 5);
    assert_eq!(out[0].trim(), "1", "step after first episode should be 1");
    assert_eq!(out[1].trim(), "2", "step after second episode should be 2");
}

/// Adam parity: cjc-eval and cjc-mir-exec produce identical updated weights.
/// This is the strongest gate for the new optimizer code path.
#[test]
fn parity_adam_single_episode() {
    let body = r#"
        let w = init_weights();
        let st = init_adam_state();
        let r = train_one_episode_adam(w, st, 12, 0.01);
        let new_w = r[0];
        let new_st = r[1];
        let loss = r[2];
        let n = r[3];
        print(n);
        print(loss);
        print(new_w[0].get([0, 0]));
        print(new_w[10].get([0, 0]));
        print(new_st[0][10].get([0, 0]));
        print(new_st[1][10].get([0, 0]));
        print(new_st[2]);
    "#;
    let eval_out = run(Backend::Eval, body, 21);
    let mir_out = run(Backend::Mir, body, 21);
    assert_eq!(
        eval_out, mir_out,
        "Adam path should be byte-identical across executors\n  eval: {eval_out:?}\n  mir:  {mir_out:?}"
    );
}

// ==========================================================================
// ============== ADVANTAGE/RETURN NORMALIZATION (Phase B2)
// ==========================================================================

/// Whitening a constant list returns zero mean and near-zero variance (eps floor).
#[test]
fn whiten_constant_is_zero() {
    let body = r#"
        let xs = [5.0, 5.0, 5.0, 5.0];
        let out = whiten_array(xs);
        print(out[0]);
        print(out[1]);
        print(out[2]);
        print(out[3]);
    "#;
    let out = run(Backend::Eval, body, 1);
    for line in &out {
        let val: f64 = line.trim().parse().unwrap();
        assert!(val.abs() < 1e-6, "constant should whiten to ~0, got {val}");
    }
}

/// Whitening preserves the expected mean ≈ 0 and std ≈ 1 on a known list.
#[test]
fn whiten_moments() {
    let body = r#"
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = whiten_array(xs);
        // Print sum (should be ~0) and sum of squares (should be ~n=5).
        let s = out[0] + out[1] + out[2] + out[3] + out[4];
        let ss = out[0]*out[0] + out[1]*out[1] + out[2]*out[2] + out[3]*out[3] + out[4]*out[4];
        print(s);
        print(ss);
    "#;
    let out = run(Backend::Eval, body, 1);
    let s: f64 = out[0].trim().parse().unwrap();
    let ss: f64 = out[1].trim().parse().unwrap();
    assert!(s.abs() < 1e-6, "whitened sum should be ~0, got {s}");
    // Variance unbiased here is computed with /n so expected ss ≈ n = 5
    // (modulo the tiny 1e-8 eps in the std denominator).
    assert!((ss - 5.0).abs() < 1e-4, "whitened sum-of-squares should be ~5, got {ss}");
}

/// Whitening an empty list returns an empty list cleanly.
#[test]
fn whiten_empty() {
    let body = r#"
        let xs = [];
        let out = whiten_array(xs);
        print(len(out));
    "#;
    let out = run(Backend::Eval, body, 1);
    assert_eq!(out[0].trim(), "0");
}

/// Adam path with normalization is deterministic across runs.
#[test]
fn adam_with_normalization_deterministic() {
    let body = r#"
        let w = init_weights();
        let st = init_adam_state();
        let r = train_one_episode_adam(w, st, 15, 0.01);
        let new_w = r[0];
        let loss = r[2];
        print(new_w[10].get([0, 0]));
        print(loss);
    "#;
    let a = run(Backend::Eval, body, 55);
    let b = run(Backend::Eval, body, 55);
    assert_eq!(a, b, "Adam+norm should be bit-deterministic");
}

// ==========================================================================
// ============== TEMPERATURE ANNEALING (Phase B3)
// ==========================================================================

/// anneal_temp: boundary cases + midpoint.
#[test]
fn anneal_temp_schedule() {
    let body = r#"
        print(anneal_temp(0, 100, 1.0, 0.1));
        print(anneal_temp(50, 100, 1.0, 0.1));
        print(anneal_temp(100, 100, 1.0, 0.1));
        print(anneal_temp(500, 100, 1.0, 0.1));  // clamp past end
        print(anneal_temp(0 - 5, 100, 1.0, 0.1)); // clamp before start
    "#;
    let out = run(Backend::Eval, body, 1);
    let vals: Vec<f64> = out.iter().map(|s| s.trim().parse().unwrap()).collect();
    assert!((vals[0] - 1.0).abs() < 1e-9, "ep=0 should be t_start");
    assert!((vals[1] - 0.55).abs() < 1e-9, "ep=50/100 linear midpoint");
    assert!((vals[2] - 0.1).abs() < 1e-9, "ep=ep_max should be t_end");
    assert!((vals[3] - 0.1).abs() < 1e-9, "past ep_max clamps to t_end");
    assert!((vals[4] - 1.0).abs() < 1e-9, "negative clamps to t_start");
}

/// High-temperature rollout finishes; low-temperature rollout finishes.
/// Both produce a valid trajectory (smoke + stability).
#[test]
fn rollout_episode_temp_smoke() {
    let body = r#"
        let w = init_weights();
        let r_hi = rollout_episode_temp(w, 20, 2.0);
        let r_lo = rollout_episode_temp(w, 20, 0.1);
        print(r_hi[6]);
        print(r_lo[6]);
    "#;
    let out = run(Backend::Eval, body, 19);
    let n_hi: i64 = out[0].trim().parse().unwrap();
    let n_lo: i64 = out[1].trim().parse().unwrap();
    assert!(n_hi > 0 && n_hi <= 21, "high-temp rollout length out of range");
    assert!(n_lo > 0 && n_lo <= 21, "low-temp rollout length out of range");
}

/// Temperature-annealed Adam training is deterministic with same seed.
#[test]
fn adam_temp_deterministic() {
    let body = r#"
        let w = init_weights();
        let st = init_adam_state();
        let r = train_one_episode_adam_temp(w, st, 15, 0.01, 0.5);
        let new_w = r[0];
        let loss = r[2];
        print(new_w[10].get([0, 0]));
        print(loss);
    "#;
    let a = run(Backend::Eval, body, 33);
    let b = run(Backend::Eval, body, 33);
    assert_eq!(a, b, "temperature-annealed Adam should be bit-deterministic");
}

/// At temp=1.0 the temperature-annealed rollout produces the same trajectory
/// length distribution as the baseline rollout (same seed, same starting
/// state, same number of plies up to the first divergence).
/// We compare trajectory lengths — equal when the softmax distributions
/// and sampled actions line up, which they should at temp=1.0.
#[test]
fn temp_one_matches_baseline_length() {
    let body = r#"
        let w = init_weights();
        let base = rollout_episode(w, 25);
        let temp = rollout_episode_temp(w, 25, 1.0);
        print(base[6]);
        print(temp[6]);
    "#;
    let out = run(Backend::Eval, body, 77);
    let base: i64 = out[0].trim().parse().unwrap();
    let temp: i64 = out[1].trim().parse().unwrap();
    assert_eq!(
        base, temp,
        "temp=1.0 rollout should match baseline rollout length (same seed)"
    );
}

// ==========================================================================
// ============== RESIGNATION THRESHOLD (Phase B4)
// ==========================================================================

/// With resignation disabled (patience <= 0), rollout_episode_full matches
/// rollout_episode_temp at temp=1.0.
#[test]
fn resign_disabled_matches_temp_rollout() {
    let body = r#"
        let w = init_weights();
        let a = rollout_episode_temp(w, 18, 1.0);
        let b = rollout_episode_full(w, 18, 1.0, 0.0 - 0.9, 0);
        print(a[6]);
        print(b[6]);
    "#;
    let out = run(Backend::Eval, body, 61);
    assert_eq!(out[0].trim(), out[1].trim(), "disabled resign should match temp-only rollout");
}

/// A very loose resignation threshold (v < 0.999) with patience=1 forces
/// resignation on the first step (model value is bounded by tanh within [-1,1]).
/// This should terminate the rollout early with a non-zero terminal_reward
/// whose sign reflects the resigning side.
#[test]
fn resign_triggers_early() {
    let body = r#"
        let w = init_weights();
        // Use a threshold such that nearly any value triggers resignation.
        let r = rollout_episode_full(w, 30, 1.0, 0.999, 1);
        let n = r[6];
        let tr = r[5];
        print(n);
        print(tr);
    "#;
    let out = run(Backend::Eval, body, 91);
    let n: i64 = out[0].trim().parse().unwrap();
    let tr: f64 = out[1].trim().parse().unwrap();
    assert_eq!(n, 1, "extreme threshold should resign on the first ply");
    // Starting side is white (+1); white resigns → terminal_reward = -1.
    assert_eq!(tr, -1.0, "resign terminal should be -1 for white resigning");
}

/// A very strict resignation threshold (v < -10.0) never triggers, so the
/// rollout proceeds exactly as the baseline.
#[test]
fn resign_never_triggers() {
    let body = r#"
        let w = init_weights();
        let base = rollout_episode_temp(w, 20, 1.0);
        let full = rollout_episode_full(w, 20, 1.0, 0.0 - 10.0, 5);
        print(base[6]);
        print(full[6]);
    "#;
    let out = run(Backend::Eval, body, 103);
    assert_eq!(out[0].trim(), out[1].trim(), "unreachable threshold should match baseline");
}

/// Full training path is deterministic.
#[test]
fn adam_full_deterministic() {
    let body = r#"
        let w = init_weights();
        let st = init_adam_state();
        let r = train_one_episode_adam_full(w, st, 15, 0.01, 0.8, 0.0 - 0.9, 3);
        let new_w = r[0];
        let loss = r[2];
        print(new_w[10].get([0, 0]));
        print(loss);
    "#;
    let a = run(Backend::Eval, body, 44);
    let b = run(Backend::Eval, body, 44);
    assert_eq!(a, b, "full Adam training path should be bit-deterministic");
}

// ==========================================================================
// ============== MATERIAL-GREEDY BASELINE (Phase B5)
// ==========================================================================

/// Piece material values match the classic scoring.
#[test]
fn piece_material_values() {
    let body = r#"
        print(piece_material_value(1));   // pawn
        print(piece_material_value(2));   // knight
        print(piece_material_value(3));   // bishop
        print(piece_material_value(4));   // rook
        print(piece_material_value(5));   // queen
        print(piece_material_value(6));   // king
        print(piece_material_value(0));   // empty
        print(piece_material_value(0 - 5)); // black queen
    "#;
    let out = run(Backend::Eval, body, 1);
    let vals: Vec<i64> = out.iter().map(|s| s.trim().parse().unwrap()).collect();
    assert_eq!(vals, vec![1, 3, 3, 5, 9, 0, 0, 9]);
}

/// eval_vs_greedy runs and returns a well-formed outcome vector.
#[test]
fn eval_vs_greedy_runs() {
    let body = r#"
        let w = init_weights();
        let r = eval_vs_greedy(w, 2, 40);
        print(len(r));
        print(r[0] + r[1] + r[2]);
    "#;
    let out = run(Backend::Eval, body, 13);
    assert_eq!(out[0].trim(), "3", "outcome should be [wins, draws, losses]");
    assert_eq!(out[1].trim(), "2", "total games should equal n_games");
}

/// Material-greedy vs itself (agent is also material-greedy) produces
/// deterministic results across runs.
#[test]
fn eval_vs_greedy_deterministic() {
    let body = r#"
        let w = init_weights();
        let r = eval_vs_greedy(w, 2, 30);
        print(r[0]);
        print(r[1]);
        print(r[2]);
    "#;
    let a = run(Backend::Eval, body, 88);
    let b = run(Backend::Eval, body, 88);
    assert_eq!(a, b, "eval_vs_greedy should be bit-deterministic");
}

// ==========================================================================
// ============== CHECKPOINT BUNDLE (Phase C1)
// ==========================================================================

/// Checkpoint round-trip: save, load, and verify the state is bit-identical.
/// This is the strongest Phase C1 gate — any drift in weights, Adam moments,
/// or counters would surface as a diff in the printed probes.
#[test]
fn checkpoint_roundtrip() {
    let tmp = std::env::temp_dir().join(format!(
        "cjc_chess_ckpt_{}.snap",
        std::process::id()
    ));
    let tmp_str = tmp.to_string_lossy().replace('\\', "/");
    let body = format!(
        r#"
        let w = init_weights();
        let st = init_adam_state();
        // Run one Adam step so moments are non-trivial.
        let r = train_one_episode_adam(w, st, 12, 0.01);
        let w1 = r[0];
        let st1 = r[1];
        let episode = 7;

        save_checkpoint("{path}", w1, st1, episode);
        let loaded = load_checkpoint("{path}");
        let w2 = loaded[0];
        let st2 = loaded[1];
        let ep2 = loaded[2];

        // Probes: a weight, an m buffer, a v buffer, the step counter,
        // and the episode counter.
        print(w1[10].get([0, 0]));
        print(w2[10].get([0, 0]));
        print(st1[0][10].get([0, 0]));
        print(st2[0][10].get([0, 0]));
        print(st1[1][10].get([0, 0]));
        print(st2[1][10].get([0, 0]));
        print(st1[2]);
        print(st2[2]);
        print(episode);
        print(ep2);
    "#,
        path = tmp_str
    );
    let out = run(Backend::Eval, &body, 7);
    // Pairs must match:  (0,1), (2,3), (4,5), (6,7), (8,9)
    for i in (0..10).step_by(2) {
        assert_eq!(
            out[i].trim(),
            out[i + 1].trim(),
            "checkpoint roundtrip diverged at index {i}: {} vs {}",
            out[i],
            out[i + 1]
        );
    }
    let _ = std::fs::remove_file(&tmp);
}

/// weights_to_10 / weights_from_10 round-trip preserves shape.
#[test]
fn weights_pack_unpack() {
    let body = r#"
        let w = init_weights();
        let packed = weights_to_10(w);
        let unpacked = weights_from_10(packed);
        print(len(w));
        print(len(packed));
        print(len(unpacked));
        // Slot 4 was reserved in original and should be 0 in unpacked.
        // Comparing weight tensors: slot 10 should round-trip identically.
        print(w[10].get([0, 0]));
        print(unpacked[10].get([0, 0]));
    "#;
    let out = run(Backend::Eval, body, 1);
    assert_eq!(out[0].trim(), "11");
    assert_eq!(out[1].trim(), "10");
    assert_eq!(out[2].trim(), "11");
    assert_eq!(out[3].trim(), out[4].trim(), "weight tensor should round-trip");
}

/// Training resumed from a checkpoint produces the same step counter progression
/// as training run without the save/load intermediate.
#[test]
fn checkpoint_resumable() {
    let tmp = std::env::temp_dir().join(format!(
        "cjc_chess_ckpt_resume_{}.snap",
        std::process::id()
    ));
    let tmp_str = tmp.to_string_lossy().replace('\\', "/");
    let body = format!(
        r#"
        let w0 = init_weights();
        let st0 = init_adam_state();
        let r1 = train_one_episode_adam(w0, st0, 12, 0.01);
        let w1 = r1[0]; let st1 = r1[1];
        // Save + load.
        save_checkpoint("{path}", w1, st1, 1);
        let loaded = load_checkpoint("{path}");
        let w1b = loaded[0]; let st1b = loaded[1];
        // Step counters must match.
        print(st1[2]);
        print(st1b[2]);
    "#,
        path = tmp_str
    );
    let out = run(Backend::Eval, &body, 11);
    assert_eq!(out[0].trim(), out[1].trim(), "step counter should survive round-trip");
    let _ = std::fs::remove_file(&tmp);
}

// ==========================================================================
// ============== CSV TRAINING LOG (Phase C2)
// ==========================================================================

/// csv_open_log writes a header line and csv_log_episode appends rows.
/// The resulting file contains the expected header + rows.
#[test]
fn csv_log_roundtrip() {
    let tmp = std::env::temp_dir().join(format!(
        "cjc_chess_csv_{}.csv",
        std::process::id()
    ));
    let tmp_str = tmp.to_string_lossy().replace('\\', "/");
    let body = format!(
        r#"
        csv_open_log("{path}");
        csv_log_episode("{path}", 0, 0.5, 20, 1.0, 0.9, 1);
        csv_log_episode("{path}", 1, 0.4, 25, 0.0, 0.8, 2);
        csv_log_episode("{path}", 2, 0.3, 30, 0.0 - 1.0, 0.7, 3);
        let lines = file_lines("{path}");
        print(len(lines));
        print(lines[0]);
        print(lines[1]);
    "#,
        path = tmp_str
    );
    let out = run(Backend::Eval, &body, 1);
    assert_eq!(out[0].trim(), "4", "expected 4 lines (header + 3 rows)");
    assert!(
        out[1].contains("episode") && out[1].contains("loss"),
        "header should name episode and loss, got {:?}",
        out[1]
    );
    assert!(
        out[2].contains("0,") && out[2].contains("20"),
        "first row should contain episode=0 and n_moves=20, got {:?}",
        out[2]
    );
    let _ = std::fs::remove_file(&tmp);
}

/// CSV log survives a mid-training crash: rows appended up to the last
/// file_append are preserved, and a subsequent load sees them. We simulate
/// "crash recovery" by reopening the same file after partial writes.
#[test]
fn csv_log_append_recovery() {
    let tmp = std::env::temp_dir().join(format!(
        "cjc_chess_csv_recover_{}.csv",
        std::process::id()
    ));
    let tmp_str = tmp.to_string_lossy().replace('\\', "/");
    // Phase 1: open + one row.
    let body1 = format!(
        r#"
        csv_open_log("{path}");
        csv_log_episode("{path}", 0, 0.5, 10, 1.0, 0.9, 1);
    "#,
        path = tmp_str
    );
    run(Backend::Eval, &body1, 1);
    // Phase 2: append two more rows (no re-open, since csv_open_log would
    // truncate). This simulates a training restart that resumes the log.
    let body2 = format!(
        r#"
        csv_log_episode("{path}", 1, 0.4, 12, 1.0, 0.8, 2);
        csv_log_episode("{path}", 2, 0.3, 15, 0.0, 0.7, 3);
        let lines = file_lines("{path}");
        print(len(lines));
    "#,
        path = tmp_str
    );
    let out = run(Backend::Eval, &body2, 1);
    assert_eq!(out[0].trim(), "4", "header + 3 rows after recovery append");
    let _ = std::fs::remove_file(&tmp);
}

// ==========================================================================
// ============== SNAPSHOT GAUNTLET + ELO-LITE (Phase C3)
// ==========================================================================

/// `elo_expected(r, r)` must be exactly 0.5 — self-play has equal odds.
#[test]
fn elo_expected_equal_is_half() {
    let body = r#"
        let e = elo_expected(1000.0, 1000.0);
        print(e);
    "#;
    let out = run(Backend::Eval, body, 1);
    let v: f64 = out[0].trim().parse().unwrap();
    assert!((v - 0.5).abs() < 1e-12, "equal ratings -> 0.5, got {v}");
}

/// 400 Elo gap gives ~0.909 expectation for the stronger side
/// (closed form: 1/(1+10^(-1)) = 10/11 ≈ 0.90909...).
#[test]
fn elo_expected_400_gap() {
    let body = r#"
        let e = elo_expected(1400.0, 1000.0);
        print(e);
    "#;
    let out = run(Backend::Eval, body, 1);
    let v: f64 = out[0].trim().parse().unwrap();
    let expected = 10.0 / 11.0;
    assert!(
        (v - expected).abs() < 1e-9,
        "400-Elo gap closed form: got {v}, want {expected}"
    );
}

/// Expected scores for the two sides must sum to exactly 1.0 — zero-sum invariant.
#[test]
fn elo_expected_sums_to_one() {
    let body = r#"
        let a = elo_expected(1234.0, 987.0);
        let b = elo_expected(987.0, 1234.0);
        print(a + b);
    "#;
    let out = run(Backend::Eval, body, 1);
    let v: f64 = out[0].trim().parse().unwrap();
    assert!((v - 1.0).abs() < 1e-12, "expectations should sum to 1, got {v}");
}

/// elo_update: a draw between equal-rated players leaves both ratings unchanged.
#[test]
fn elo_update_draw_equal_ratings() {
    let body = r#"
        let r = elo_update(1500.0, 1500.0, 0.5, 32.0);
        print(r);
    "#;
    let out = run(Backend::Eval, body, 1);
    let v: f64 = out[0].trim().parse().unwrap();
    assert!((v - 1500.0).abs() < 1e-12, "equal draw should not move rating, got {v}");
}

/// elo_update: upset win against a stronger opponent gains more than a
/// win against a weaker one — monotonicity in opponent strength.
#[test]
fn elo_update_upset_monotone() {
    let body = r#"
        let weak   = elo_update(1000.0, 800.0,  1.0, 32.0);
        let even   = elo_update(1000.0, 1000.0, 1.0, 32.0);
        let upset  = elo_update(1000.0, 1200.0, 1.0, 32.0);
        print(weak - 1000.0);
        print(even - 1000.0);
        print(upset - 1000.0);
    "#;
    let out = run(Backend::Eval, body, 1);
    let d_weak:  f64 = out[0].trim().parse().unwrap();
    let d_even:  f64 = out[1].trim().parse().unwrap();
    let d_upset: f64 = out[2].trim().parse().unwrap();
    assert!(d_weak > 0.0 && d_even > 0.0 && d_upset > 0.0);
    assert!(
        d_upset > d_even && d_even > d_weak,
        "upset > even > weak expected, got {d_weak} {d_even} {d_upset}"
    );
}

/// Applying a full W/D/L record is order-independent in total-gain sense
/// only for draws; we just check that the running update matches a manual
/// two-step computation for a small 1W+1L record.
#[test]
fn elo_apply_record_matches_manual() {
    let body = r#"
        let r0 = 1200.0;
        let r_opp = 1200.0;
        let k = 32.0;
        let wdl = [1, 0, 1];
        let r_final = elo_apply_record(r0, r_opp, wdl, k);
        // Manual: win first, then loss.
        let after_w = elo_update(r0, r_opp, 1.0, k);
        let after_wl = elo_update(after_w, r_opp, 0.0, k);
        print(r_final);
        print(after_wl);
    "#;
    let out = run(Backend::Eval, body, 1);
    let a: f64 = out[0].trim().parse().unwrap();
    let b: f64 = out[1].trim().parse().unwrap();
    assert!((a - b).abs() < 1e-12, "apply_record should equal manual update, {a} vs {b}");
}

/// Cross-executor parity: the same Elo math in eval and mir-exec matches.
#[test]
fn elo_math_parity() {
    let body = r#"
        let e1 = elo_expected(1337.0, 1000.0);
        let r1 = elo_update(1337.0, 1000.0, 1.0, 32.0);
        let r2 = elo_update(1337.0, 1000.0, 0.0, 32.0);
        let r3 = elo_apply_record(1000.0, 1100.0, [3, 2, 5], 24.0);
        print(e1);
        print(r1);
        print(r2);
        print(r3);
    "#;
    // run_parity already asserts byte-identical eval vs mir output.
    let _ = run_parity(body, 1);
}

/// The snapshot gauntlet runs end-to-end and produces a plausible total
/// game count and a finite rating.
#[test]
fn gauntlet_runs() {
    let body = r#"
        let w_curr = init_weights();
        let snaps = [init_weights(), init_weights()];
        let ratings = [1000.0, 1000.0];
        let out = gauntlet_vs_snapshots(w_curr, snaps, ratings, 1000.0, 2, 20, 32.0);
        let r_final = out[0];
        let w = out[1];
        let d = out[2];
        let l = out[3];
        let total = w + d + l;
        print(total);
        let finite = 0;
        if r_final > 0.0 && r_final < 3000.0 { finite = 1; }
        print(finite);
    "#;
    let out = run(Backend::Eval, body, 5);
    assert_eq!(parse_i64(&out), 4, "2 snapshots * 2 games each = 4 total");
    assert_eq!(out[1].trim(), "1", "rating should stay in sane bounds");
}

/// Gauntlet is deterministic under a fixed seed.
#[test]
fn gauntlet_deterministic() {
    let body = r#"
        let w_curr = init_weights();
        let snaps = [init_weights()];
        let ratings = [1000.0];
        let out = gauntlet_vs_snapshots(w_curr, snaps, ratings, 1000.0, 4, 18, 32.0);
        print(out[0]);
        print(out[1]);
        print(out[2]);
        print(out[3]);
    "#;
    let a = run(Backend::Eval, body, 77);
    let b = run(Backend::Eval, body, 77);
    assert_eq!(a, b, "gauntlet must be deterministic under a fixed seed");
}

// ==========================================================================
// ============== PGN DUMP (Phase C4)
// ==========================================================================

/// `sq_to_uci` produces canonical file/rank notation for a handful of
/// well-known squares.
#[test]
fn sq_to_uci_known_squares() {
    let body = r#"
        print(sq_to_uci(0));   // a1
        print(sq_to_uci(7));   // h1
        print(sq_to_uci(56));  // a8
        print(sq_to_uci(63));  // h8
        print(sq_to_uci(12));  // e2  (rank=1, file=4)
        print(sq_to_uci(28));  // e4  (rank=3, file=4)
    "#;
    let out = run(Backend::Eval, body, 1);
    assert_eq!(out[0].trim(), "a1");
    assert_eq!(out[1].trim(), "h1");
    assert_eq!(out[2].trim(), "a8");
    assert_eq!(out[3].trim(), "h8");
    assert_eq!(out[4].trim(), "e2");
    assert_eq!(out[5].trim(), "e4");
}

/// `move_to_lan` formats a move as long-algebraic-notation `from-to`.
#[test]
fn move_to_lan_format() {
    let body = r#"
        print(move_to_lan(12, 28));  // e2-e4
        print(move_to_lan(62, 45));  // g8-f6 (black knight from g8 to f6)
    "#;
    let out = run(Backend::Eval, body, 1);
    assert_eq!(out[0].trim(), "e2-e4");
    assert_eq!(out[1].trim(), "g8-f6");
}

/// `pgn_result_token` maps the f64 outcome to the canonical PGN string.
#[test]
fn pgn_result_token_values() {
    let body = r#"
        print(pgn_result_token(1.0));
        print(pgn_result_token(0.0 - 1.0));
        print(pgn_result_token(0.0));
    "#;
    let out = run(Backend::Eval, body, 1);
    assert_eq!(out[0].trim(), "1-0");
    assert_eq!(out[1].trim(), "0-1");
    assert_eq!(out[2].trim(), "1/2-1/2");
}

/// `pgn_format_game` emits a string with all seven standard tag-pair
/// headers, the correct result token, and the moves in order.
#[test]
fn pgn_format_game_has_headers() {
    let body = r#"
        let moves = [12, 28, 52, 36];
        let txt = pgn_format_game(
            "Test", "local", "2026.04.09", "1",
            "white-agent", "black-agent",
            0.0, moves
        );
        print(txt);
    "#;
    let out = run(Backend::Eval, body, 1);
    let joined = out.join("\n");
    assert!(joined.contains("[Event \"Test\"]"), "missing Event tag");
    assert!(joined.contains("[Site \"local\"]"), "missing Site tag");
    assert!(joined.contains("[Date \"2026.04.09\"]"), "missing Date tag");
    assert!(joined.contains("[Round \"1\"]"), "missing Round tag");
    assert!(joined.contains("[White \"white-agent\"]"), "missing White tag");
    assert!(joined.contains("[Black \"black-agent\"]"), "missing Black tag");
    assert!(joined.contains("[Result \"1/2-1/2\"]"), "missing Result tag");
    assert!(joined.contains("e2-e4"), "missing white's first move");
    assert!(joined.contains("e7-e5"), "missing black's first move");
    assert!(joined.contains("1. e2-e4"), "move number prefix missing");
}

/// `play_recorded_game` returns a flat [from, to, ...] list whose length
/// is even and a finite outcome.
#[test]
fn play_recorded_game_smoke() {
    let body = r#"
        let w = init_weights();
        let g = play_recorded_game(w, w, 12);
        let moves = g[0];
        let result = g[1];
        let n = len(moves);
        print(n);
        // even-length: each move contributes two ints
        if n % 2 == 0 { print(1); } else { print(0); }
        // result in [-1, 1]
        let finite = 0;
        if result >= 0.0 - 1.0 && result <= 1.0 { finite = 1; }
        print(finite);
    "#;
    let out = run(Backend::Eval, body, 3);
    let n: i64 = out[0].trim().parse().unwrap();
    assert!(n > 0 && n <= 24, "should have 0 < n ≤ max_moves*2, got {n}");
    assert_eq!(out[1].trim(), "1", "moves array must have even length");
    assert_eq!(out[2].trim(), "1", "result must be in [-1, 1]");
}

/// `pgn_dump_game` writes to a file and `file_read` returns content
/// containing the expected tags.
#[test]
fn pgn_dump_game_roundtrip() {
    let tmp = std::env::temp_dir().join(format!(
        "pgn_dump_{}_{}.pgn",
        std::process::id(),
        format!("{:?}", std::thread::current().id())
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
    ));
    let tmp_str = tmp.to_string_lossy().replace('\\', "/");
    let body = format!(
        r#"
        let w = init_weights();
        let g = play_recorded_game(w, w, 10);
        let moves = g[0];
        let result = g[1];
        let n = pgn_dump_game("{path}", "CJC-Lang Chess RL v2.1", "1",
                              "cjc-agent", "cjc-agent", result, moves);
        print(n);
        let content = file_read("{path}");
        print(content);
    "#,
        path = tmp_str
    );
    let out = run(Backend::Eval, &body, 5);
    let n: i64 = out[0].trim().parse().unwrap();
    assert!(n > 0, "should have recorded at least one full move");
    let joined = out[1..].join("\n");
    assert!(joined.contains("[Event \"CJC-Lang Chess RL v2.1\"]"));
    assert!(joined.contains("[White \"cjc-agent\"]"));
    assert!(joined.contains("1. "), "must contain at least move number 1");
    let _ = std::fs::remove_file(&tmp);
}

/// Dumping three games in sequence produces a file with three
/// `[Event ...]` headers — confirms file_append semantics.
#[test]
fn pgn_dump_three_games() {
    let tmp = std::env::temp_dir().join(format!(
        "pgn_three_{}_{}.pgn",
        std::process::id(),
        format!("{:?}", std::thread::current().id())
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
    ));
    let tmp_str = tmp.to_string_lossy().replace('\\', "/");
    let body = format!(
        r#"
        let w = init_weights();
        let g1 = play_recorded_game(w, w, 8);
        let g2 = play_recorded_game(w, w, 8);
        let g3 = play_recorded_game(w, w, 8);
        pgn_dump_game("{path}", "gauntlet", "1", "white", "black", g1[1], g1[0]);
        pgn_dump_game("{path}", "gauntlet", "2", "white", "black", g2[1], g2[0]);
        pgn_dump_game("{path}", "gauntlet", "3", "white", "black", g3[1], g3[0]);
        let content = file_read("{path}");
        print(content);
    "#,
        path = tmp_str
    );
    let out = run(Backend::Eval, &body, 9);
    let joined = out.join("\n");
    let event_count = joined.matches("[Event \"gauntlet\"]").count();
    assert_eq!(event_count, 3, "expected 3 event headers, got {event_count}");
    assert!(joined.contains("[Round \"1\"]"));
    assert!(joined.contains("[Round \"2\"]"));
    assert!(joined.contains("[Round \"3\"]"));
    let _ = std::fs::remove_file(&tmp);
}

/// Recorded games are deterministic under a fixed seed and the same
/// weight set — critical for reproducible match archives.
#[test]
fn play_recorded_game_deterministic() {
    let body = r#"
        let w = init_weights();
        let g = play_recorded_game(w, w, 10);
        let moves = g[0];
        print(len(moves));
        print(g[1]);
        if len(moves) > 0 { print(moves[0]); print(moves[1]); }
    "#;
    let a = run(Backend::Eval, body, 44);
    let b = run(Backend::Eval, body, 44);
    assert_eq!(a, b, "recorded games must be deterministic under a fixed seed");
}

// ==========================================================================
// ============== VIZOR TRAINING CURVE (Phase C5)
// ==========================================================================

/// vizor_training_curve writes an SVG file whose contents parse as SVG
/// and contain a `<polyline>` or `<path>` (the geom_line render target).
#[test]
fn vizor_training_curve_writes_svg() {
    let tmp = std::env::temp_dir().join(format!(
        "cjc_train_curve_{}_{}.svg",
        std::process::id(),
        format!("{:?}", std::thread::current().id())
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
    ));
    let tmp_str = tmp.to_string_lossy().replace('\\', "/");
    let body = format!(
        r#"
        let eps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let losses = [1.0, 0.9, 0.85, 0.75, 0.6, 0.55];
        let n = vizor_training_curve("{path}", "loss curve", "loss", eps, losses);
        print(n);
    "#,
        path = tmp_str
    );
    let out = run(Backend::Eval, &body, 1);
    assert_eq!(out[0].trim(), "6", "should report 6 points");
    let content = std::fs::read_to_string(&tmp).expect("SVG file should have been written");
    assert!(content.contains("<svg"), "output should contain <svg tag");
    assert!(content.contains("</svg>"), "output should be a closed SVG document");
    let _ = std::fs::remove_file(&tmp);
}

/// vizor_training_curves emits two files into the given directory.
#[test]
fn vizor_training_curves_dual_output() {
    let base = std::env::temp_dir().join(format!(
        "cjc_train_plots_{}_{}",
        std::process::id(),
        format!("{:?}", std::thread::current().id())
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
    ));
    std::fs::create_dir_all(&base).expect("tmp plot dir");
    let base_str = base.to_string_lossy().replace('\\', "/");
    let body = format!(
        r#"
        let eps = [0.0, 1.0, 2.0, 3.0];
        let losses = [1.0, 0.8, 0.6, 0.5];
        let rewards = [0.0 - 1.0, 0.0 - 0.5, 0.0, 0.3];
        let n = vizor_training_curves("{dir}", eps, losses, rewards);
        print(n);
    "#,
        dir = base_str
    );
    let out = run(Backend::Eval, &body, 1);
    assert_eq!(out[0].trim(), "2", "should report 2 plots written");
    let loss_path = base.join("training_loss.svg");
    let reward_path = base.join("training_reward.svg");
    assert!(loss_path.exists(), "loss SVG should exist");
    assert!(reward_path.exists(), "reward SVG should exist");
    let loss_svg = std::fs::read_to_string(&loss_path).unwrap();
    let reward_svg = std::fs::read_to_string(&reward_path).unwrap();
    assert!(loss_svg.contains("<svg"));
    assert!(reward_svg.contains("<svg"));
    assert!(loss_svg.contains("loss"), "loss SVG should contain label");
    assert!(reward_svg.contains("reward"), "reward SVG should contain label");
    let _ = std::fs::remove_file(&loss_path);
    let _ = std::fs::remove_file(&reward_path);
    let _ = std::fs::remove_dir(&base);
}

/// Vizor curve is deterministic: same data → byte-identical SVG.
#[test]
fn vizor_training_curve_deterministic() {
    let tmpa = std::env::temp_dir().join(format!("cjc_curve_a_{}.svg", std::process::id()));
    let tmpb = std::env::temp_dir().join(format!("cjc_curve_b_{}.svg", std::process::id()));
    let ap = tmpa.to_string_lossy().replace('\\', "/");
    let bp = tmpb.to_string_lossy().replace('\\', "/");
    let body_a = format!(
        r#"
        let eps = [0.0, 1.0, 2.0];
        let ys = [0.5, 0.4, 0.3];
        vizor_training_curve("{p}", "det", "y", eps, ys);
        print(0);
    "#,
        p = ap
    );
    let body_b = format!(
        r#"
        let eps = [0.0, 1.0, 2.0];
        let ys = [0.5, 0.4, 0.3];
        vizor_training_curve("{p}", "det", "y", eps, ys);
        print(0);
    "#,
        p = bp
    );
    let _ = run(Backend::Eval, &body_a, 7);
    let _ = run(Backend::Eval, &body_b, 7);
    let a = std::fs::read_to_string(&tmpa).unwrap();
    let b = std::fs::read_to_string(&tmpb).unwrap();
    assert_eq!(a, b, "identical data should produce byte-identical SVG");
    let _ = std::fs::remove_file(&tmpa);
    let _ = std::fs::remove_file(&tmpb);
}

/// Rollout returns a trajectory with sensible length.
#[test]
fn rollout_trajectory_shape() {
    let body = r#"
        let w = init_weights();
        let r = rollout_episode(w, 15);
        let states = r[0];
        let moves = r[1];
        let actions = r[2];
        let values = r[3];
        let sides = r[4];
        let n = r[6];
        print(n);
        print(len(states));
        print(len(moves));
        print(len(actions));
        print(len(values));
        print(len(sides));
    "#;
    let out = run_parity(body, 11);
    let n = parse_i64(&out);
    assert!(n > 0, "should record at least one state");
    for line in &out[1..] {
        assert_eq!(line.trim().parse::<i64>().unwrap(), n, "lists should share length");
    }
}

// ============================================================================
// ===================== V2.2 tests (Tier 1 cheap ML fixes) ===================
// ============================================================================

/// T1-c: position_key_v22 is stable and differs between distinct positions.
#[test]
fn v22_position_key_distinct() {
    let body = r#"
        let s0 = init_state();
        let k0 = position_key_v22(s0);
        // Same state → same key
        let s0b = init_state();
        let k0b = position_key_v22(s0b);
        if k0 == k0b { print(1); } else { print(0); }
        // After a move, key changes
        let s1 = apply_move(s0, 12, 28);  // e2-e4
        let k1 = position_key_v22(s1);
        if k0 != k1 { print(1); } else { print(0); }
    "#;
    let out = run_parity(body, 1);
    assert_eq!(out[0].trim(), "1", "identical states should produce identical keys");
    assert_eq!(out[1].trim(), "1", "distinct states should produce distinct keys");
}

/// T1-c: rep_inc_v22 increments counter across repeated positions.
#[test]
fn v22_rep_inc_counts() {
    let body = r#"
        let m = map_new();
        let s = init_state();
        let p1 = rep_inc_v22(m, s);
        let m2 = p1[0];
        let c1 = p1[1];
        let p2 = rep_inc_v22(m2, s);
        let m3 = p2[0];
        let c2 = p2[1];
        let p3 = rep_inc_v22(m3, s);
        let c3 = p3[1];
        print(c1);
        print(c2);
        print(c3);
    "#;
    let out = run_parity(body, 2);
    assert_eq!(out[0].trim(), "1");
    assert_eq!(out[1].trim(), "2");
    assert_eq!(out[2].trim(), "3");
}

/// T1-b: move-count penalty shrinks positive rewards and clamps at +0.05.
#[test]
fn v22_move_penalty_shrinks_win() {
    let body = r#"
        // reward=1.0, n_moves=20, penalty=0.01 → 1.0 - 0.20 = 0.80
        let r1 = apply_move_penalty_v22(1.0, 20, 0.01);
        print(r1);
        // reward=1.0, n_moves=200, penalty=0.01 → would be -1.0, clamps to 0.05
        let r2 = apply_move_penalty_v22(1.0, 200, 0.01);
        print(r2);
        // draw stays 0
        let r3 = apply_move_penalty_v22(0.0, 50, 0.01);
        print(r3);
        // negative rewards shrink toward -0.05 floor
        let r4 = apply_move_penalty_v22(0.0 - 1.0, 20, 0.01);
        print(r4);
    "#;
    let out = run_parity(body, 3);
    // CJC-Lang to_string may emit 0.8 or 0.8000...
    let r1: f64 = out[0].trim().parse().unwrap();
    assert!((r1 - 0.80).abs() < 1e-9, "r1={r1}, expected 0.80");
    let r2: f64 = out[1].trim().parse().unwrap();
    assert!((r2 - 0.05).abs() < 1e-9, "r2={r2}, expected clamp 0.05");
    let r3: f64 = out[2].trim().parse().unwrap();
    assert!(r3.abs() < 1e-9, "r3={r3}, expected 0.0");
    let r4: f64 = out[3].trim().parse().unwrap();
    assert!((r4 + 0.80).abs() < 1e-9, "r4={r4}, expected -0.80");
}

/// T1-b: zero penalty is a no-op.
#[test]
fn v22_move_penalty_zero_is_noop() {
    let body = r#"
        print(apply_move_penalty_v22(1.0, 100, 0.0));
        print(apply_move_penalty_v22(0.0 - 1.0, 100, 0.0));
        print(apply_move_penalty_v22(0.0, 100, 0.0));
    "#;
    let out = run_parity(body, 4);
    let a: f64 = out[0].trim().parse().unwrap();
    let b: f64 = out[1].trim().parse().unwrap();
    let c: f64 = out[2].trim().parse().unwrap();
    assert!((a - 1.0).abs() < 1e-9);
    assert!((b + 1.0).abs() < 1e-9);
    assert!(c.abs() < 1e-9);
}

/// T1-d: select_action_eval_temp_v22 at temp=0 falls back to greedy.
#[test]
fn v22_eval_temp_zero_equals_greedy() {
    let body = r#"
        let w = init_weights();
        let s = init_state();
        let m = legal_moves(s);
        let g = select_action_greedy(w, s, m);
        let t = select_action_eval_temp_v22(w, s, m, 0.0);
        print(g[0]);
        print(t[0]);
    "#;
    let out = run_parity(body, 5);
    assert_eq!(out[0].trim(), out[1].trim(), "temp=0 should match greedy");
}

/// T1-d: stochastic eval sampler returns valid move indices for temp>0.
#[test]
fn v22_eval_temp_valid_index() {
    let body = r#"
        let w = init_weights();
        let s = init_state();
        let m = legal_moves(s);
        let num = len(m) / 2;
        let sel = select_action_eval_temp_v22(w, s, m, 0.15);
        let a = sel[0];
        let ok = 0;
        if a >= 0 && a < num { ok = 1; }
        print(ok);
    "#;
    let out = run_parity(body, 6);
    assert_eq!(out[0].trim(), "1", "action should be in range");
}

/// T1-a/b/c: v22 rollout completes, returns 8-slot layout, rep_flag is 0 or 1.
#[test]
fn v22_rollout_episode_smoke() {
    let body = r#"
        let w = init_weights();
        let r = rollout_episode_v22(w, 40, 1.0, 0.001);
        let n = r[6];
        let rep = r[7];
        print(n);
        if rep == 0 || rep == 1 { print(1); } else { print(0); }
        print(len(r));
    "#;
    let out = run_parity(body, 7);
    let n: i64 = out[0].trim().parse().unwrap();
    assert!(n > 0 && n <= 41);
    assert_eq!(out[1].trim(), "1");
    assert_eq!(out[2].trim(), "8");
}

/// T1-a/b/c: v22 rollout is deterministic across repeated calls.
#[test]
fn v22_rollout_deterministic() {
    let body = r#"
        let w = init_weights();
        let r1 = rollout_episode_v22(w, 30, 0.8, 0.001);
        let r2 = rollout_episode_v22(w, 30, 0.8, 0.001);
        print(r1[6]);
        print(r2[6]);
        print(r1[7]);
        print(r2[7]);
    "#;
    let out = run(Backend::Eval, body, 8);
    assert_eq!(out[0].trim(), out[1].trim(), "n_moves should match");
    assert_eq!(out[2].trim(), out[3].trim(), "rep_flag should match");
}

/// T1-a/b/c: full training episode via train_one_episode_adam_v22 works.
#[test]
fn v22_train_one_episode_smoke() {
    let body = r#"
        let w = init_weights();
        let adam = init_adam_state();
        let r = train_one_episode_adam_v22(w, adam, 30, 0.001, 1.0, 0.001);
        let new_w = r[0];
        let new_adam = r[1];
        let loss = r[2];
        let n = r[3];
        let rep = r[5];
        print(len(new_w));
        print(n);
        let finite = 0;
        if loss > 0.0 - 1.0e9 && loss < 1.0e9 { finite = 1; }
        print(finite);
        if rep == 0 || rep == 1 { print(1); } else { print(0); }
    "#;
    let out = run(Backend::Eval, body, 9);
    assert_eq!(out[0].trim(), "11");
    let n: i64 = out[1].trim().parse().unwrap();
    assert!(n > 0);
    assert_eq!(out[2].trim(), "1");
    assert_eq!(out[3].trim(), "1");
}

/// T1-e: csv_open_log_v22 writes the new header with 7 columns.
#[test]
fn v22_csv_header_has_seven_columns() {
    let tmp = std::env::temp_dir().join("cjc_v22_csv_header.csv");
    let _ = std::fs::remove_file(&tmp);
    let path = tmp.to_string_lossy().replace('\\', "/");
    let body = format!(
        r#"
        csv_open_log_v22("{p}");
        csv_log_episode_v22("{p}", 0, 0.5, 25, 0.0, 1.2, 1, 0);
        csv_log_episode_v22("{p}", 1, 0.4, 30, 1.0, 1.1, 2, 1);
        print(1);
    "#,
        p = path
    );
    let _ = run(Backend::Eval, &body, 10);
    let csv = std::fs::read_to_string(&tmp).expect("csv was written");
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 3, "header + 2 rows");
    assert_eq!(
        lines[0],
        "episode,loss,n_moves,terminal_reward,temp,adam_step,repetition_draw"
    );
    // Row 2 rep_flag should be 1.
    let last = lines[2];
    assert!(last.ends_with(",1"), "row 2 rep_flag column = 1, got {last}");
    let _ = std::fs::remove_file(&tmp);
}

/// T1-d: eval_vs_random_v22 returns a triple that sums to n_games.
#[test]
fn v22_eval_vs_random_triple() {
    let body = r#"
        let w = init_weights();
        let t = eval_vs_random_v22(w, 4, 20, 0.15);
        print(t[0]);
        print(t[1]);
        print(t[2]);
    "#;
    let out = run(Backend::Eval, body, 11);
    let w: i64 = out[0].trim().parse().unwrap();
    let d: i64 = out[1].trim().parse().unwrap();
    let l: i64 = out[2].trim().parse().unwrap();
    assert_eq!(w + d + l, 4);
}

/// T1-d: eval_vs_greedy_v22 returns a triple that sums to n_games.
#[test]
fn v22_eval_vs_greedy_triple() {
    let body = r#"
        let w = init_weights();
        let t = eval_vs_greedy_v22(w, 4, 20, 0.15);
        print(t[0]);
        print(t[1]);
        print(t[2]);
    "#;
    let out = run(Backend::Eval, body, 12);
    let w: i64 = out[0].trim().parse().unwrap();
    let d: i64 = out[1].trim().parse().unwrap();
    let l: i64 = out[2].trim().parse().unwrap();
    assert_eq!(w + d + l, 4);
}

/// Parity: v22 rollout reward / rep_flag / n_moves are byte-identical
/// across cjc-eval and cjc-mir-exec.
#[test]
fn v22_rollout_parity() {
    let body = r#"
        let w = init_weights();
        let r = rollout_episode_v22(w, 25, 1.0, 0.001);
        print(r[5]);  // adjusted reward
        print(r[6]);  // n_moves
        print(r[7]);  // rep_flag
    "#;
    let _ = run_parity(body, 13);
}

// ---------------------------------------------------------------------------
// Chess RL v2.3 — Tier 2 profiling determinism tests
// ---------------------------------------------------------------------------
//
// The profile counter builtins (profile_zone_start/stop/dump, see
// `crates/cjc-runtime/src/profile.rs`) are a write-only sink. Instrumenting
// the hot path of `rollout_episode_v22` must not change the weight hash or
// the trajectory. These tests lock that invariant.

/// Tier 2 determinism: training one episode with profile zones interleaved
/// produces an identical weight-list hash vs the uninstrumented run on the
/// same seed. Both runs are done on `cjc-eval`.
#[test]
fn v23_profile_instrumentation_preserves_weight_hash_eval() {
    let baseline_body = r#"
        let w = init_weights();
        let adam = init_adam_state();
        let r = train_one_episode_adam_v22(w, adam, 20, 0.001, 1.0, 0.001);
        let new_w = r[0];
        print(tensor_list_hash(weights_to_10(new_w)));
    "#;
    let scratch = std::env::temp_dir()
        .join("v23_profile_eval_reset.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let dump = std::env::temp_dir()
        .join("v23_profile_eval_dump.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let instrumented_body = format!(
        r#"
        profile_dump("{scratch}");
        let w = init_weights();
        let adam = init_adam_state();
        let outer = profile_zone_start("rollout_total");
        let r = train_one_episode_adam_v22(w, adam, 20, 0.001, 1.0, 0.001);
        profile_zone_stop(outer);
        profile_dump("{dump}");
        let new_w = r[0];
        print(tensor_list_hash(weights_to_10(new_w)));
    "#,
        scratch = scratch,
        dump = dump,
    );

    let base = run(Backend::Eval, baseline_body, 17);
    let inst = run(Backend::Eval, &instrumented_body, 17);
    assert_eq!(
        base, inst,
        "profile instrumentation changed the eval weight hash\nbase: {base:?}\ninst: {inst:?}"
    );
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(&dump);
}

/// Same invariant on the MIR executor.
#[test]
fn v23_profile_instrumentation_preserves_weight_hash_mir() {
    let baseline_body = r#"
        let w = init_weights();
        let adam = init_adam_state();
        let r = train_one_episode_adam_v22(w, adam, 20, 0.001, 1.0, 0.001);
        let new_w = r[0];
        print(tensor_list_hash(weights_to_10(new_w)));
    "#;
    let scratch = std::env::temp_dir()
        .join("v23_profile_mir_reset.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let dump = std::env::temp_dir()
        .join("v23_profile_mir_dump.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let instrumented_body = format!(
        r#"
        profile_dump("{scratch}");
        let w = init_weights();
        let adam = init_adam_state();
        let outer = profile_zone_start("rollout_total");
        let r = train_one_episode_adam_v22(w, adam, 20, 0.001, 1.0, 0.001);
        profile_zone_stop(outer);
        profile_dump("{dump}");
        let new_w = r[0];
        print(tensor_list_hash(weights_to_10(new_w)));
    "#,
        scratch = scratch,
        dump = dump,
    );

    let base = run(Backend::Mir, baseline_body, 17);
    let inst = run(Backend::Mir, &instrumented_body, 17);
    assert_eq!(
        base, inst,
        "profile instrumentation changed the mir weight hash\nbase: {base:?}\ninst: {inst:?}"
    );
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(&dump);
}

/// Cross-executor parity: an instrumented training episode run through
/// both executors produces the same weight hash and the same CSV row count
/// (one per distinct zone name).
#[test]
fn v23_profile_instrumentation_cross_executor_parity() {
    let scratch = std::env::temp_dir()
        .join("v23_profile_xparity_reset.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let dump = std::env::temp_dir()
        .join("v23_profile_xparity_dump.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let body = format!(
        r#"
        profile_dump("{scratch}");
        let w = init_weights();
        let adam = init_adam_state();
        let outer = profile_zone_start("rollout_total");
        let step = profile_zone_start("score_step");
        profile_zone_stop(step);
        let r = train_one_episode_adam_v22(w, adam, 15, 0.001, 1.0, 0.001);
        profile_zone_stop(outer);
        let rows = profile_dump("{dump}");
        print(rows);
        let new_w = r[0];
        print(tensor_list_hash(weights_to_10(new_w)));
    "#,
        scratch = scratch,
        dump = dump,
    );
    let out = run_parity(&body, 19);
    // Should have exactly 2 rows ("rollout_total" + "score_step").
    assert_eq!(out[0].trim(), "2");
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(&dump);
}

/// Tier 2: run the fully-instrumented rollout and verify the CSV has the
/// expected zone names (rollout_total, legal_moves, score_moves,
/// apply_move, rep_tracking).
#[test]
fn v23_instrumented_rollout_csv_zones() {
    let scratch = std::env::temp_dir()
        .join("v23_instr_csv_reset.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let dump = std::env::temp_dir()
        .join("v23_instr_csv_dump.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let body = format!(
        r#"
        profile_dump("{scratch}");
        let w = init_weights();
        let r = rollout_episode_v22_instrumented(w, 20, 1.0, 0.001, "{dump}");
        print(r[6]);
    "#,
        scratch = scratch,
        dump = dump,
    );
    let out = run(Backend::Eval, &body, 42);
    let n_moves: i64 = out[0].trim().parse().unwrap();
    assert!(n_moves > 0, "instrumented rollout produced 0 moves");
    let csv = std::fs::read_to_string(&dump).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    let zone_names: Vec<&str> = lines[1..]
        .iter()
        .map(|l| l.split(',').next().unwrap())
        .collect();
    assert!(zone_names.contains(&"rollout_total"), "missing rollout_total zone");
    assert!(zone_names.contains(&"legal_moves"), "missing legal_moves zone");
    assert!(zone_names.contains(&"score_moves"), "missing score_moves zone");
    assert!(zone_names.contains(&"apply_move"), "missing apply_move zone");
    assert!(zone_names.contains(&"rep_tracking"), "missing rep_tracking zone");
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(&dump);
}

/// Tier 2: run the fully-instrumented training episode and verify
/// a2c_update zone is present in the dump.
#[test]
fn v23_instrumented_train_episode_csv_has_a2c() {
    let scratch = std::env::temp_dir()
        .join("v23_instr_train_reset.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let dump = std::env::temp_dir()
        .join("v23_instr_train_dump.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let body = format!(
        r#"
        profile_dump("{scratch}");
        let w = init_weights();
        let adam = init_adam_state();
        let r = train_one_episode_adam_v22_instrumented(w, adam, 20, 0.001, 1.0, 0.001, "{dump}");
        print(r[3]);
    "#,
        scratch = scratch,
        dump = dump,
    );
    let out = run(Backend::Eval, &body, 42);
    let n_moves: i64 = out[0].trim().parse().unwrap();
    assert!(n_moves > 0, "instrumented training produced 0 moves");
    let csv = std::fs::read_to_string(&dump).unwrap();
    assert!(
        csv.contains("a2c_update"),
        "CSV missing a2c_update zone: {csv}"
    );
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(&dump);
}

/// Tier 2: instrumented vs uninstrumented rollout produces byte-identical
/// trajectory (the core determinism invariant for v2.3).
#[test]
fn v23_instrumented_rollout_trajectory_parity() {
    let dump = std::env::temp_dir()
        .join("v23_instr_parity_dump.csv")
        .to_string_lossy()
        .replace('\\', "/");

    // Uninstrumented rollout.
    let baseline = r#"
        let w = init_weights();
        let r = rollout_episode_v22(w, 25, 1.0, 0.001);
        print(r[5]);
        print(r[6]);
        print(r[7]);
    "#;
    // Instrumented rollout — same seed, same math.
    let instrumented = format!(
        r#"
        profile_dump("{dump}");
        let w = init_weights();
        let r = rollout_episode_v22_instrumented(w, 25, 1.0, 0.001, "{dump}");
        print(r[5]);
        print(r[6]);
        print(r[7]);
    "#,
        dump = dump,
    );

    let base_out = run(Backend::Eval, baseline, 13);
    let inst_out = run(Backend::Eval, &instrumented, 13);
    assert_eq!(
        base_out, inst_out,
        "instrumented rollout diverged from baseline\nbase: {base_out:?}\ninst: {inst_out:?}"
    );
    let _ = std::fs::remove_file(&dump);
}

/// Tier 2 measurement: Run 3 instrumented training episodes at max_moves=80
/// and dump the profile CSV. This is the actual profiling run that identifies
/// the hot zones for Tier 3.
///
/// Run with: cargo test --test test_chess_rl_v2 --release v23_tier2_profile_measurement -- --ignored
#[test]
#[ignore]
fn v23_tier2_profile_measurement() {
    use std::time::Instant;

    let dump = "bench_results/chess_rl_v2_3/profile_hot_zones.csv"
        .replace('\\', "/");

    let body = format!(
        r#"
        profile_dump("{dump}");
        let w = init_weights();
        let adam = init_adam_state();
        let ep = 0;
        while ep < 3 {{
            let r = train_one_episode_adam_v22_instrumented(
                w, adam, 80, 0.001, 1.0, 0.001, "{dump}"
            );
            w = r[0];
            adam = r[1];
            print("ep=" + to_string(ep) + " n_moves=" + to_string(r[3])
                  + " reward=" + to_string(r[4]));
            ep = ep + 1;
        }}
    "#,
        dump = dump,
    );

    let start = Instant::now();
    let out = run(Backend::Eval, &body, 42);
    let elapsed = start.elapsed();

    // Print output + timing
    eprintln!("=== Tier 2 Profile Measurement ===");
    for line in &out {
        eprintln!("  {line}");
    }
    eprintln!("Wall clock: {:.1}s ({:.1}s/episode)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 3.0);

    // Read the CSV and report zone percentages
    let csv = std::fs::read_to_string(&dump).unwrap_or_else(|e| panic!("read dump: {e}"));
    eprintln!("\n=== Profile CSV ===");
    eprintln!("{csv}");
}

// ---------------------------------------------------------------------------
// Chess RL v2.3 — Tier 3 native kernel rollout tests
// ---------------------------------------------------------------------------

/// v2.3 rollout smoke test: rollout_episode_v23 produces valid output.
#[test]
fn v23_rollout_episode_smoke() {
    let body = r#"
        let w = init_weights();
        let r = rollout_episode_v23(w, 25, 1.0, 0.001);
        print(r[6]);  // n_moves
        print(r[7]);  // rep_flag
    "#;
    let out = run(Backend::Eval, body, 7);
    let n: i64 = out[0].trim().parse().unwrap();
    assert!(n > 0, "v23 rollout produced 0 moves");
    let rep: i64 = out[1].trim().parse().unwrap();
    assert!(rep == 0 || rep == 1);
}

/// v2.3 rollout is deterministic (same seed → same output).
#[test]
fn v23_rollout_deterministic() {
    let body = r#"
        let w = init_weights();
        let r = rollout_episode_v23(w, 25, 1.0, 0.001);
        print(r[5]);
        print(r[6]);
    "#;
    let a = run(Backend::Eval, body, 13);
    let b = run(Backend::Eval, body, 13);
    assert_eq!(a, b, "v23 rollout not deterministic");
}

/// **CRITICAL**: v2.3 rollout produces bit-identical trajectories to v2.2.
/// This proves the native kernels compute the same values as the
/// interpreter-driven CJC-Lang code.
#[test]
fn v23_rollout_matches_v22() {
    let v22_body = r#"
        let w = init_weights();
        let r = rollout_episode_v22(w, 25, 1.0, 0.001);
        print(r[5]);
        print(r[6]);
        print(r[7]);
    "#;
    let v23_body = r#"
        let w = init_weights();
        let r = rollout_episode_v23(w, 25, 1.0, 0.001);
        print(r[5]);
        print(r[6]);
        print(r[7]);
    "#;
    let v22 = run(Backend::Eval, v22_body, 13);
    let v23 = run(Backend::Eval, v23_body, 13);
    assert_eq!(
        v22, v23,
        "v23 native kernel rollout diverged from v22 CJC-Lang rollout\nv22: {v22:?}\nv23: {v23:?}"
    );
}

/// v2.3 training episode smoke.
#[test]
fn v23_train_one_episode_smoke() {
    let body = r#"
        let w = init_weights();
        let adam = init_adam_state();
        let r = train_one_episode_adam_v23(w, adam, 25, 0.001, 1.0, 0.001);
        let new_w = r[0];
        let n = r[3];
        print(len(new_w));
        print(n);
    "#;
    let out = run(Backend::Eval, body, 9);
    assert_eq!(out[0].trim(), "11");
    let n: i64 = out[1].trim().parse().unwrap();
    assert!(n > 0);
}

/// v2.3 training episode weight hash matches v2.2.
#[test]
fn v23_train_episode_weight_hash_matches_v22() {
    let v22_body = r#"
        let w = init_weights();
        let adam = init_adam_state();
        let r = train_one_episode_adam_v22(w, adam, 20, 0.001, 1.0, 0.001);
        print(tensor_list_hash(weights_to_10(r[0])));
    "#;
    let v23_body = r#"
        let w = init_weights();
        let adam = init_adam_state();
        let r = train_one_episode_adam_v23(w, adam, 20, 0.001, 1.0, 0.001);
        print(tensor_list_hash(weights_to_10(r[0])));
    "#;
    let v22 = run(Backend::Eval, v22_body, 17);
    let v23 = run(Backend::Eval, v23_body, 17);
    assert_eq!(
        v22, v23,
        "v23 training weight hash diverged from v22\nv22: {v22:?}\nv23: {v23:?}"
    );
}

/// Cross-executor parity for v2.3 rollout.
#[test]
fn v23_rollout_cross_executor_parity() {
    let body = r#"
        let w = init_weights();
        let r = rollout_episode_v23(w, 15, 1.0, 0.001);
        print(r[5]);
        print(r[6]);
        print(r[7]);
    "#;
    let _ = run_parity(body, 11);
}

/// Speedup measurement: v23 rollout must be at least 2× faster than v22.
/// Runs both at max_moves=40 and compares wall clock.
///
/// Run with: cargo test --test test_chess_rl_v2 --release v23_rollout_speedup -- --ignored --nocapture
#[test]
#[ignore]
fn v23_rollout_speedup() {
    use std::time::Instant;

    let v22_body = r#"
        let w = init_weights();
        let r = rollout_episode_v22(w, 40, 1.0, 0.001);
        print(r[6]);
    "#;
    let v23_body = r#"
        let w = init_weights();
        let r = rollout_episode_v23(w, 40, 1.0, 0.001);
        print(r[6]);
    "#;

    let start_v22 = Instant::now();
    let _ = run(Backend::Eval, v22_body, 42);
    let v22_time = start_v22.elapsed().as_secs_f64();

    let start_v23 = Instant::now();
    let _ = run(Backend::Eval, v23_body, 42);
    let v23_time = start_v23.elapsed().as_secs_f64();

    let speedup = v22_time / v23_time;
    eprintln!("v22: {v22_time:.1}s | v23: {v23_time:.1}s | speedup: {speedup:.1}×");
    assert!(
        v23_time < v22_time * 0.5,
        "v23 is not 2× faster: v22={v22_time:.1}s v23={v23_time:.1}s speedup={speedup:.1}×"
    );
}
