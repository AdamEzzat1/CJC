//! Property-based tests for the RL agent.

use proptest::prelude::*;
use crate::chess_rl_hardening::helpers::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Action index is always in valid range [0, num_moves).
    #[test]
    fn action_idx_in_range(seed in 0u64..500) {
        let src = chess_agent_program(r#"
            let w = init_weights();
            let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
            let b = init_board();
            let moves = legal_moves(b, 1);
            let feat = encode_board(b, 1);
            let result = select_action(W1, b1, W2, feat, moves);
            let idx = int(result[0]);
            let num = int(result[2]);
            if idx >= 0 && idx < num { print("OK"); } else { print("BAD"); }
        "#);
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "OK");
    }

    /// Log probability is always non-positive.
    #[test]
    fn log_prob_non_positive(seed in 0u64..500) {
        let src = chess_agent_program(r#"
            let w = init_weights();
            let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
            let b = init_board();
            let moves = legal_moves(b, 1);
            let feat = encode_board(b, 1);
            let result = select_action(W1, b1, W2, feat, moves);
            let lp = result[1];
            if lp <= 0.001 { print("OK"); } else { print("BAD"); }
        "#);
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "OK");
    }

    /// Forward score is always finite.
    #[test]
    fn forward_score_finite(seed in 0u64..500) {
        let src = chess_agent_program(r#"
            let w = init_weights();
            let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
            let b = init_board();
            let feat = encode_board(b, 1);
            let result = forward_move(W1, b1, W2, feat, 12, 28);
            let score = result[0];
            if isnan(score) || isinf(score) { print("BAD"); } else { print("OK"); }
        "#);
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "OK");
    }
}
