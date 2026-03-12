//! Property-based determinism tests.
//! Same seed → identical output for all operations.

use proptest::prelude::*;
use crate::chess_rl_hardening::helpers::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Rollout is deterministic: same seed → same result.
    #[test]
    fn rollout_deterministic(seed in 0u64..1000) {
        let src = full_program(r#"
            let w = init_weights();
            let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
            let result = play_episode(W1, b1, W2, 20);
            print(result[0]); print(result[1]);
        "#);
        let out1 = run_mir(&src, seed);
        let out2 = run_mir(&src, seed);
        prop_assert_eq!(out1, out2);
    }

    /// Legal move list is deterministic.
    #[test]
    fn legal_moves_deterministic(seed in 0u64..1000) {
        let src = chess_program(r#"
            let b = init_board();
            let m = legal_moves(b, 1);
            let i = 0;
            while i < len(m) {
                print(m[i]);
                i = i + 1;
            }
        "#);
        let out1 = run_mir(&src, seed);
        let out2 = run_mir(&src, seed);
        prop_assert_eq!(out1, out2);
    }

    /// Training is deterministic.
    #[test]
    fn training_deterministic(seed in 0u64..500) {
        let src = full_program(r#"
            let w = init_weights();
            let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
            let result = train_episode(W1, b1, W2, 0.001, 0.99, 0.0, 15);
            print(result[0]); print(result[1]); print(result[2]);
        "#);
        let out1 = run_mir(&src, seed);
        let out2 = run_mir(&src, seed);
        prop_assert_eq!(out1, out2);
    }
}
