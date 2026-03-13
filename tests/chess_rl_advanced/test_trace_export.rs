//! Phase 2+: Trace export tests.
//!
//! Tests that training and game data can be exported in a structured
//! format (via print output parsed by the Rust harness).

use super::helpers::*;

/// Training trace: per-episode metrics are parseable.
#[test]
fn trace_export_episode_metrics() {
    let src = multi_program(r#"
        let result = train_multi_episodes(3, 0.01, 0.99, 0.0, 8);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.len(), 9, "3 episodes * 3 lines each");

    // Build structured trace
    let mut episodes = Vec::new();
    for ep in 0..3 {
        let reward = parse_float_at(&out, ep * 3);
        let loss = parse_float_at(&out, ep * 3 + 1);
        let steps = parse_float_at(&out, ep * 3 + 2);
        episodes.push((reward, loss, steps));
    }

    // All metrics should be finite
    for (i, (r, l, s)) in episodes.iter().enumerate() {
        assert!(r.is_finite(), "episode {i} reward not finite");
        assert!(l.is_finite(), "episode {i} loss not finite");
        assert!(*s >= 1.0, "episode {i} steps should be >= 1");
    }
}

/// Training trace can be serialized as JSON-like structure.
#[test]
fn trace_export_json_format() {
    let src = multi_program(r#"
        let result = train_multi_episodes(2, 0.01, 0.99, 0.0, 8);
    "#);
    let out = run_mir(&src, 42);

    // Parse into JSON-like records
    let mut records = Vec::new();
    for ep in 0..2 {
        let reward = parse_float_at(&out, ep * 3);
        let loss = parse_float_at(&out, ep * 3 + 1);
        let steps = parse_float_at(&out, ep * 3 + 2);
        records.push(format!(
            r#"{{"episode":{},"reward":{},"loss":{},"steps":{}}}"#,
            ep, reward, loss, steps as i64
        ));
    }
    assert_eq!(records.len(), 2);

    // Verify JSON is well-formed (basic check)
    for r in &records {
        assert!(r.starts_with('{') && r.ends_with('}'));
        assert!(r.contains("episode"));
        assert!(r.contains("reward"));
    }
}

/// Win-rate trace export.
#[test]
fn trace_export_win_rate() {
    let src = multi_program(r#"
        let weights = init_weights();
        let wr = eval_win_rate(weights[0], weights[1], weights[2], 4, 8, 1);
        print(wr);
    "#);
    let out = run_mir(&src, 42);
    let wins = parse_int_at(&out, 0);
    let draws = parse_int_at(&out, 1);
    let losses = parse_int_at(&out, 2);
    let win_rate = parse_float_at(&out, 3);

    // Build trace record
    let record = format!(
        r#"{{"wins":{},"draws":{},"losses":{},"win_rate":{:.4}}}"#,
        wins, draws, losses, win_rate
    );
    assert!(record.contains("wins"));
    assert!(record.contains("win_rate"));
}

/// Full training + evaluation trace.
#[test]
fn trace_export_full_pipeline() {
    let src = multi_program(r#"
        let trained = train_multi_episodes(2, 0.01, 0.99, 0.0, 8);
        let W1 = trained[0];
        let b1 = trained[1];
        let W2 = trained[2];
        let wr = eval_win_rate(W1, b1, W2, 3, 8, 1);
        print(wr);
    "#);
    let out = run_mir(&src, 42);
    // 2 episodes * 3 lines + 4 lines (wins, draws, losses, wr) = 10 lines
    assert!(out.len() >= 10, "expected at least 10 output lines, got {}", out.len());
}

/// Trace is deterministic.
#[test]
fn trace_export_deterministic() {
    let src = multi_program(r#"
        let trained = train_multi_episodes(2, 0.01, 0.99, 0.0, 8);
        let wr = eval_win_rate(trained[0], trained[1], trained[2], 3, 8, 1);
        print(wr);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "trace export not deterministic");
}
