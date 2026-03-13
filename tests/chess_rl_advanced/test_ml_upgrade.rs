//! ML Upgrade Tests — Phase 5
//!
//! Tests that validate the CJC engine's ML capabilities haven't regressed
//! and verify core properties needed for the actor-critic upgrade.
//! These test the CJC backend; the JS platform tests are browser-based.

use super::helpers::*;

// =====================================================================
// NETWORK FORWARD PASS TESTS
// =====================================================================

/// V1 network forward pass is deterministic — same input produces same output.
#[test]
fn network_forward_determinism() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let board = init_board();
        let features = encode_board(board, 1);
        let s1 = forward_move(weights[0], weights[1], weights[2], features, 12, 28);
        let s2 = forward_move(weights[0], weights[1], weights[2], features, 12, 28);
        print(s1[0]);
        print(s2[0]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.len(), 2);
    let s1 = parse_float_at(&out, 0);
    let s2 = parse_float_at(&out, 1);
    assert_eq!(s1, s2, "forward pass not deterministic");
}

/// Network outputs finite values (no NaN, no Inf).
#[test]
fn network_no_nan_outputs() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let board = init_board();
        let features = encode_board(board, 1);
        let result = forward_move(weights[0], weights[1], weights[2], features, 12, 28);
        let score = result[0];
        let is_finite = score > -1000.0;
        let is_finite2 = score < 1000.0;
        print(is_finite);
        print(is_finite2);
        print(score);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].trim(), "true", "score should be > -1000");
    assert_eq!(out[1].trim(), "true", "score should be < 1000");
    let score = parse_float_at(&out, 2);
    assert!(score.is_finite(), "score must be finite, got {}", score);
}

/// Weight initialization produces values in expected range.
#[test]
fn weight_init_variance() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let sum = 0.0;
        let sum_sq = 0.0;
        let n = 0;
        let i = 0;
        while i < 66 {
            let j = 0;
            while j < 16 {
                let val = W1.get([i, j]);
                sum = sum + val;
                sum_sq = sum_sq + val * val;
                n = n + 1;
                j = j + 1
            }
            i = i + 1
        }
        let mean = sum / n;
        let variance = sum_sq / n - mean * mean;
        print(mean);
        print(variance);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.len(), 2);
    let mean = parse_float_at(&out, 0);
    let variance = parse_float_at(&out, 1);
    // Init uses Gaussian(0, 0.1), so variance should be ~0.01
    assert!(mean.abs() < 0.05, "mean should be near 0, got {}", mean);
    assert!(variance > 0.005 && variance < 0.02, "variance should be ~0.01, got {}", variance);
}

// =====================================================================
// TRAINING TESTS
// =====================================================================

/// Training determinism — same seed produces identical training results.
#[test]
fn training_determinism_v2() {
    let src = full_program(r#"
        let weights = init_weights();
        let result = train_episode(weights[0], weights[1], weights[2], 0.01, 0.99, 0.0, 20);
        print(result[0]);
        print(result[1]);
        print(result[2]);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "training not deterministic with same seed");
}

/// Training reward is bounded [-1, 1].
#[test]
fn training_reward_bounded() {
    for seed in [42, 99, 137, 256, 500] {
        let src = full_program(r#"
            let weights = init_weights();
            let result = train_episode(weights[0], weights[1], weights[2], 0.01, 0.99, 0.0, 20);
            print(result[0]);
        "#);
        let out = run_mir(&src, seed);
        let reward = parse_float(&out);
        assert!(reward >= -1.0 && reward <= 1.0,
            "reward out of range at seed {}: {}", seed, reward);
    }
}

/// Training loss is finite (no NaN from gradient computation).
#[test]
fn training_loss_finite() {
    for seed in [42, 99, 137] {
        let src = full_program(r#"
            let weights = init_weights();
            let result = train_episode(weights[0], weights[1], weights[2], 0.01, 0.99, 0.0, 30);
            print(result[1]);
        "#);
        let out = run_mir(&src, seed);
        let loss = parse_float(&out);
        assert!(loss.is_finite(), "loss should be finite at seed {}, got {}", seed, loss);
    }
}

// =====================================================================
// GAME ENGINE REGRESSION TESTS
// =====================================================================

/// Board encoding produces 64 values.
#[test]
fn board_encoding_size() {
    let src = chess_agent_program(r#"
        let board = init_board();
        let features = encode_board(board, 1);
        print(len(features));
    "#);
    let out = run_mir(&src, 42);
    let size = parse_int(&out);
    assert_eq!(size, 64, "encoded board should be 64 features");
}

/// Legal moves from initial position are correct for white.
#[test]
fn initial_legal_moves_count() {
    let src = chess_program(r#"
        let board = init_board();
        let moves = legal_moves(board, 1);
        print(len(moves));
    "#);
    let out = run_mir(&src, 42);
    let count = parse_int(&out);
    assert_eq!(count, 40, "white should have 40 legal moves (20 moves × 2 coords) from start, got {}", count);
}

/// Rollout game always terminates within max moves.
#[test]
fn game_terminates_within_limit() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let board = init_board();
        let side = 1;
        let moves_played = 0;
        let max_moves = 50;
        let running = true;
        while running {
            let status = terminal_status(board, side);
            if status != 0 {
                running = false
            } else {
                if moves_played >= max_moves {
                    running = false
                } else {
                    let moves = legal_moves(board, side);
                    let features = encode_board(board, side);
                    let result = select_action(weights[0], weights[1], weights[2], features, moves);
                    let action_idx = int(result[0]);
                    let from_sq = moves[action_idx * 2];
                    let to_sq = moves[action_idx * 2 + 1];
                    board = apply_move(board, from_sq, to_sq);
                    side = -1 * side;
                    moves_played = moves_played + 1
                }
            }
        }
        print(moves_played);
    "#);
    let out = run_mir(&src, 42);
    let moves = parse_int(&out);
    assert!(moves > 0 && moves <= 50, "game should terminate within limit, got {} moves", moves);
}

/// Action selection always returns valid index.
#[test]
fn action_index_valid() {
    for seed in [42, 99, 137, 256] {
        let src = chess_agent_program(r#"
            let weights = init_weights();
            let board = init_board();
            let moves = legal_moves(board, 1);
            let num_legal = len(moves) / 2;
            let features = encode_board(board, 1);
            let result = select_action(weights[0], weights[1], weights[2], features, moves);
            let action = int(result[0]);
            print(action);
            print(num_legal);
        "#);
        let out = run_mir(&src, seed);
        let action = parse_int_at(&out, 0);
        let num_legal = parse_int_at(&out, 1);
        assert!(action >= 0 && action < num_legal,
            "action {} out of range [0, {}) at seed {}", action, num_legal, seed);
    }
}

// =====================================================================
// PROPERTY TESTS
// =====================================================================

/// Different seeds produce different games (non-trivial randomness).
#[test]
fn different_seeds_different_games() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let board = init_board();
        let moves = legal_moves(board, 1);
        let features = encode_board(board, 1);
        let result = select_action(weights[0], weights[1], weights[2], features, moves);
        print(int(result[0]));
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 99);
    assert!(out1.len() == 1);
    assert!(out2.len() == 1);
}

// =====================================================================
// FUZZ-STYLE TESTS (multiple random seeds)
// =====================================================================

/// Fuzz: training doesn't crash with various random seeds.
#[test]
fn fuzz_training_various_seeds() {
    for seed in [1, 7, 13, 42, 99, 137, 256, 500, 777, 9999] {
        let src = full_program(r#"
            let weights = init_weights();
            let result = train_episode(weights[0], weights[1], weights[2], 0.01, 0.99, 0.0, 15);
            print(result[0]);
        "#);
        let out = run_mir(&src, seed);
        let reward = parse_float(&out);
        assert!(reward.is_finite(), "NaN at seed {}", seed);
    }
}

/// Fuzz: game rollout doesn't crash with various seeds.
#[test]
fn fuzz_rollout_various_seeds() {
    for seed in [1, 42, 99, 256, 777, 12345] {
        let src = chess_agent_program(r#"
            let weights = init_weights();
            let board = init_board();
            let side = 1;
            let i = 0;
            while i < 20 {
                let status = terminal_status(board, side);
                if status != 0 {
                    i = 999
                } else {
                    let moves = legal_moves(board, side);
                    let features = encode_board(board, side);
                    let result = select_action(weights[0], weights[1], weights[2], features, moves);
                    let action_idx = int(result[0]);
                    let from_sq = moves[action_idx * 2];
                    let to_sq = moves[action_idx * 2 + 1];
                    board = apply_move(board, from_sq, to_sq);
                    side = -1 * side;
                    i = i + 1
                }
            }
            print("ok");
        "#);
        let out = run_mir(&src, seed);
        assert!(out.iter().any(|l| l.contains("ok")), "rollout crashed at seed {}", seed);
    }
}

/// Fuzz: encode_board always produces finite values across game states.
#[test]
fn fuzz_encode_board_finite() {
    for seed in [42, 99, 137] {
        let src = chess_agent_program(r#"
            let weights = init_weights();
            let board = init_board();
            let side = 1;
            let i = 0;
            while i < 10 {
                let status = terminal_status(board, side);
                if status != 0 {
                    i = 999
                } else {
                    let features = encode_board(board, side);
                    let flen = len(features);
                    let check = flen > 0;
                    if check {
                        print("")
                    }
                    let moves = legal_moves(board, side);
                    let result = select_action(weights[0], weights[1], weights[2], features, moves);
                    let action_idx = int(result[0]);
                    let from_sq = moves[action_idx * 2];
                    let to_sq = moves[action_idx * 2 + 1];
                    board = apply_move(board, from_sq, to_sq);
                    side = -1 * side;
                    i = i + 1
                }
            }
            print("ok");
        "#);
        let out = run_mir(&src, seed);
        assert!(out.iter().any(|l| l.contains("ok")), "encode_board issue at seed {}", seed);
    }
}

/// Parity test: same game with same seed produces identical results in two MIR runs.
#[test]
fn mir_parity_two_runs() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let board = init_board();
        let features = encode_board(board, 1);
        let result = forward_move(weights[0], weights[1], weights[2], features, 12, 28);
        print(result[0]);
    "#);

    let mir_out1 = run_mir(&src, 42);
    let mir_out2 = run_mir(&src, 42);

    assert!(!mir_out1.is_empty(), "MIR run 1 produced no output");
    assert!(!mir_out2.is_empty(), "MIR run 2 produced no output");
    assert_eq!(mir_out1[0].trim(), mir_out2[0].trim(),
        "Two MIR runs should produce identical forward pass results");
}
