//! Engine tests for the chess RL v2 demo: board setup, movegen, rules.
//!
//! Each test builds a `fn main()` that calls into the `source::PRELUDE`
//! functions, runs it through one (or both) backend(s), and parses the
//! printed output. All tests are deterministic.

use crate::chess_rl_v2::harness::{parse_i64, parse_i64_line, run, run_parity, Backend};

/// The initial position has 32 pieces and exactly 2 kings.
#[test]
fn initial_board_piece_count() {
    let body = r#"
        let s = init_state();
        let b = state_board(s);
        let kings = 0;
        let pieces = 0;
        let i = 0;
        while i < 64 {
            let p = b[i];
            if p != 0 { pieces = pieces + 1; }
            if p == 6 || p == 0 - 6 { kings = kings + 1; }
            i = i + 1;
        }
        print(pieces);
        print(kings);
    "#;
    let out = run_parity(body, 42);
    assert_eq!(out[0].trim(), "32", "should have 32 pieces in starting position");
    assert_eq!(out[1].trim(), "2", "should have exactly 2 kings");
}

/// White has exactly 20 legal moves from the starting position.
#[test]
fn initial_legal_move_count() {
    let body = r#"
        let s = init_state();
        let m = legal_moves(s);
        print(len(m) / 2);
    "#;
    let out = run_parity(body, 1);
    assert_eq!(parse_i64(&out), 20);
}

/// After 1.e4, black also has 20 legal moves.
#[test]
fn black_after_e4_has_20_moves() {
    let body = r#"
        let s = init_state();
        // e2 = 12, e4 = 28
        let s2 = apply_move(s, 12, 28);
        let m = legal_moves(s2);
        print(len(m) / 2);
        print(state_side(s2));
    "#;
    let out = run_parity(body, 1);
    assert_eq!(parse_i64(&out), 20);
    // side-to-move flips to black (-1)
    assert_eq!(out[1].trim(), "-1");
}

/// Terminal status on the initial board is 0 (ongoing).
#[test]
fn initial_position_not_terminal() {
    let body = r#"
        let s = init_state();
        print(terminal_status(s));
    "#;
    let out = run_parity(body, 1);
    assert_eq!(parse_i64(&out), 0);
}

/// Pawn double push sets the ep_sq correctly.
/// After 1.e4, white's pawn moved from e2 (12) to e4 (28);
/// the en passant target is e3 (20).
#[test]
fn pawn_double_push_sets_ep_square() {
    let body = r#"
        let s = init_state();
        let s2 = apply_move(s, 12, 28);
        print(state_ep(s2));
    "#;
    let out = run_parity(body, 1);
    assert_eq!(parse_i64(&out), 20);
}

/// Insufficient material: King vs King is a draw.
#[test]
fn insufficient_material_king_king() {
    let body = r#"
        // Build a bare K-K board
        let empty = [
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0
        ];
        // Put white king on e1 (4), black king on e8 (60)
        let edits = [4, 6, 60, 0 - 6];
        let b = board_with_edits(empty, edits);
        let s = make_state(b, 1, [0, 0, 0, 0], 0 - 1, 0, 0);
        let status = terminal_status(s);
        print(status);
        print(insufficient_material(b));
    "#;
    let out = run_parity(body, 1);
    // 5 = insufficient material
    assert_eq!(out[0].trim(), "5");
    assert_eq!(out[1].trim(), "true");
}

/// Fool's mate: 1.f3 e5 2.g4 Qh4# — side to move (white) is in checkmate.
#[test]
fn fools_mate_checkmate_detection() {
    let body = r#"
        let s = init_state();
        // f2=13, f3=21
        let s1 = apply_move(s, 13, 21);
        // e7=52, e5=36
        let s2 = apply_move(s1, 52, 36);
        // g2=14, g4=30
        let s3 = apply_move(s2, 14, 30);
        // d8=59, h4=31
        let s4 = apply_move(s3, 59, 31);
        print(state_side(s4));
        print(in_check(s4));
        print(terminal_status(s4));
    "#;
    let out = run_parity(body, 1);
    assert_eq!(out[0].trim(), "1", "white to move after fool's mate");
    assert_eq!(out[1].trim(), "true", "white should be in check");
    assert_eq!(out[2].trim(), "2", "terminal_status = 2 (checkmate)");
}

/// 50-move rule triggers when halfmove clock reaches 100.
#[test]
fn fifty_move_rule() {
    let body = r#"
        // Build any non-mating position and bump the halfmove clock directly.
        let s = init_state();
        let b = state_board(s);
        let s2 = make_state(b, 1, state_castling(s), 0 - 1, 100, 0);
        print(terminal_status(s2));
    "#;
    let out = run_parity(body, 1);
    // 4 = 50-move rule
    assert_eq!(parse_i64(&out), 4);
}

/// Castling kingside: after clearing f1/g1 and removing the bishop/knight,
/// white should have a castling move e1→g1 in its legal list.
#[test]
fn castling_kingside_white_generated() {
    let body = r#"
        // Starting board with f1 (5), g1 (6) cleared (bishop/knight removed).
        let s = init_state();
        let b0 = state_board(s);
        let b1 = board_with_edits(b0, [5, 0, 6, 0]);
        // Keep castling rights intact.
        let s2 = make_state(b1, 1, [1, 1, 1, 1], 0 - 1, 0, 0);
        let m = legal_moves(s2);
        // Look for king move 4 -> 6 in the flat list.
        let found = 0;
        let i = 0;
        while i < len(m) {
            if m[i] == 4 {
                if m[i + 1] == 6 { found = 1; }
            }
            i = i + 2;
        }
        print(found);
    "#;
    let out = run_parity(body, 1);
    assert_eq!(parse_i64(&out), 1, "kingside castle should be in legal moves");
}

/// Pawn promotion on reaching the 8th rank auto-promotes to queen.
#[test]
fn pawn_promotion_to_queen() {
    let body = r#"
        let empty = [
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            1,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0
        ];
        // White king e1 (4), black king e8 (60), white pawn a7 (48)
        let edits = [4, 6, 60, 0 - 6];
        let b = board_with_edits(empty, edits);
        let s = make_state(b, 1, [0, 0, 0, 0], 0 - 1, 0, 0);
        // a7 (48) -> a8 (56), promoting to queen.
        let s2 = apply_move(s, 48, 56);
        let b2 = state_board(s2);
        // piece at a8 should be +5 (white queen).
        print(b2[56]);
    "#;
    let out = run_parity(body, 1);
    assert_eq!(parse_i64(&out), 5);
}

/// `run_parity` is used by every test above — make a dedicated check that
/// the backends actually agree on a generic movegen program.
#[test]
fn movegen_eval_mir_parity() {
    let body = r#"
        let s = init_state();
        // Play 1.e4 d5 2.exd5
        let s1 = apply_move(s, 12, 28);
        let s2 = apply_move(s1, 51, 35);
        let s3 = apply_move(s2, 28, 35);
        let m = legal_moves(s3);
        print(len(m) / 2);
        print(state_side(s3));
        print(state_halfmove(s3));
    "#;
    let eval_out = run(Backend::Eval, body, 7);
    let mir_out = run(Backend::Mir, body, 7);
    assert_eq!(eval_out, mir_out, "eval and mir-exec diverged on movegen");
    // Sanity: halfmove resets to 0 after a capture.
    assert_eq!(eval_out[2].trim(), "0");
    // Avoid the unused-import warning without dragging in more functions.
    let _ = parse_i64_line("1 2 3");
}
