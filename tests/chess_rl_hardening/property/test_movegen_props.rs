//! Property-based tests for move generation.

use proptest::prelude::*;
use crate::chess_rl_hardening::helpers::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Legal moves from initial position always 20 for both sides.
    #[test]
    fn initial_20_moves_white(seed in 0u64..500) {
        let src = chess_program(r#"
            let b = init_board();
            print(len(legal_moves(b, 1)) / 2);
        "#);
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "20");
    }

    /// After one legal move, the other side has > 0 legal moves.
    #[test]
    fn opponent_has_moves_after_one_move(seed in 0u64..500) {
        let src = chess_program(r#"
            let b = init_board();
            let m = legal_moves(b, 1);
            let b2 = apply_move(b, m[0], m[1]);
            let m2 = legal_moves(b2, -1);
            if len(m2) > 0 { print("OK"); } else { print("BAD"); }
        "#);
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "OK");
    }

    /// Legal moves never leave own king in check (multi-step).
    #[test]
    fn no_self_check_3_moves(seed in 0u64..200) {
        let src = chess_program(r#"
            let b = init_board();
            let side = 1;
            let ok = true;
            let step = 0;
            while step < 3 {
                let m = legal_moves(b, side);
                if len(m) < 2 { break; }
                let b2 = apply_move(b, m[0], m[1]);
                if in_check(b2, side) { ok = false; }
                b = b2;
                side = -1 * side;
                step = step + 1;
            }
            print(ok);
        "#);
        let out = run_mir(&src, seed);
        prop_assert_eq!(out[0].as_str(), "true");
    }
}
