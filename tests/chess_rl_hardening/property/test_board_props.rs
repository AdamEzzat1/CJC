//! Property-based tests for chess board invariants.

use proptest::prelude::*;
use crate::chess_rl_hardening::helpers::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Board always has exactly 64 squares regardless of seed.
    #[test]
    fn board_always_64_squares(seed in 0u64..1000) {
        let src = chess_program("let b = init_board(); print(len(b));");
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "64");
    }

    /// Apply move always preserves board length of 64.
    #[test]
    fn apply_move_preserves_length(seed in 0u64..1000) {
        let src = chess_program(r#"
            let b = init_board();
            let m = legal_moves(b, 1);
            if len(m) >= 2 {
                let b2 = apply_move(b, m[0], m[1]);
                print(len(b2));
            } else {
                print(64);
            }
        "#);
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "64");
    }

    /// Legal moves array always has even length (from/to pairs).
    #[test]
    fn legal_moves_even_length(seed in 0u64..1000) {
        let src = chess_program(r#"
            let b = init_board();
            let m = legal_moves(b, 1);
            print(len(m) % 2);
        "#);
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "0");
    }

    /// All squares in move list are in [0, 63].
    #[test]
    fn move_squares_in_range(seed in 0u64..1000) {
        let src = chess_program(r#"
            let b = init_board();
            let m = legal_moves(b, 1);
            let ok = true;
            let i = 0;
            while i < len(m) {
                if m[i] < 0 || m[i] > 63 { ok = false; }
                i = i + 1;
            }
            print(ok);
        "#);
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "true");
    }
}
