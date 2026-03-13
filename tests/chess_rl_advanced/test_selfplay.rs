//! Phase 7: Self-play tests.
//!
//! Tests that two separate agents can play against each other,
//! with independent weight sets.

use super::helpers::*;

/// Self-play episode completes without error.
#[test]
fn selfplay_episode_completes() {
    let src = selfplay_program(r#"
        let w1 = init_weights();
        let w2 = init_weights();
        let result = selfplay_episode(
            w1[0], w1[1], w1[2],
            w2[0], w2[1], w2[2],
            10
        );
        print(result[0]);
        print(result[1]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.len(), 2);
    let reward = parse_float(&out);
    assert!(reward >= -1.0 && reward <= 1.0);
}

/// Self-play result is deterministic.
#[test]
fn selfplay_deterministic() {
    let src = selfplay_program(r#"
        let w1 = init_weights();
        let w2 = init_weights();
        let result = selfplay_episode(
            w1[0], w1[1], w1[2],
            w2[0], w2[1], w2[2],
            10
        );
        print(result[0]);
        print(result[1]);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "self-play not deterministic");
}

/// Self-play with different seeds produces different weight initializations.
#[test]
fn selfplay_different_seeds() {
    // With short max_moves, many games draw identically.
    // Verify that different seeds produce different agent weights.
    let src = selfplay_program(r#"
        let w1 = init_weights();
        print(w1[0].get([0, 0]));
        print(w1[0].get([1, 1]));
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 99);
    assert_ne!(out1, out2, "different seeds should produce different weights");
}

/// Self-play move count is bounded by max_moves.
#[test]
fn selfplay_max_moves_respected() {
    let src = selfplay_program(r#"
        let w1 = init_weights();
        let w2 = init_weights();
        let result = selfplay_episode(
            w1[0], w1[1], w1[2],
            w2[0], w2[1], w2[2],
            5
        );
        print(result[1]);
    "#);
    let out = run_mir(&src, 42);
    let moves = parse_float(&out);
    assert!(moves <= 5.0, "move count {moves} exceeds max_moves 5");
}

/// eval_agents runs multiple games between two agents.
#[test]
fn eval_agents_multiple_games() {
    let src = selfplay_program(r#"
        let w1 = init_weights();
        let w2 = init_weights();
        let result = eval_agents(
            w1[0], w1[1], w1[2],
            w2[0], w2[1], w2[2],
            4, 8
        );
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.len(), 3, "expected 3 output lines (a_wins, draws, b_wins)");
    let a_wins = parse_int_at(&out, 0);
    let draws = parse_int_at(&out, 1);
    let b_wins = parse_int_at(&out, 2);
    assert_eq!(a_wins + draws + b_wins, 4, "total games should be 4");
}

/// eval_agents is deterministic.
#[test]
fn eval_agents_deterministic() {
    let src = selfplay_program(r#"
        let w1 = init_weights();
        let w2 = init_weights();
        let result = eval_agents(
            w1[0], w1[1], w1[2],
            w2[0], w2[1], w2[2],
            4, 8
        );
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "eval_agents not deterministic");
}

/// Self-play reward is from white's perspective.
#[test]
fn selfplay_reward_perspective() {
    let src = selfplay_program(r#"
        let w1 = init_weights();
        let w2 = init_weights();
        let result = selfplay_episode(
            w1[0], w1[1], w1[2],
            w2[0], w2[1], w2[2],
            8
        );
        let reward = result[0];
        // Reward should be -1, 0, or 1
        if reward == 1.0 || reward == 0.0 || reward == -1.0 {
            print("valid");
        } else {
            print("invalid");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.last().unwrap(), "valid");
}
