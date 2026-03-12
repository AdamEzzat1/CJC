//! Board representation hardening tests.
//!
//! Validates board initialization, piece placement, square arithmetic,
//! and apply_move correctness.

use super::helpers::*;

// ============================================================
// Board initialization
// ============================================================

#[test]
fn board_is_64_squares() {
    let src = chess_program("let b = init_board(); print(len(b));");
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "64");
}

#[test]
fn board_white_pieces_rank0() {
    // Rank 0: rook, knight, bishop, queen, king, bishop, knight, rook
    let src = chess_program(r#"
        let b = init_board();
        print(b[0]); print(b[1]); print(b[2]); print(b[3]);
        print(b[4]); print(b[5]); print(b[6]); print(b[7]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out, vec!["4", "2", "3", "5", "6", "3", "2", "4"]);
}

#[test]
fn board_black_pieces_rank7() {
    let src = chess_program(r#"
        let b = init_board();
        print(b[56]); print(b[57]); print(b[58]); print(b[59]);
        print(b[60]); print(b[61]); print(b[62]); print(b[63]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out, vec!["-4", "-2", "-3", "-5", "-6", "-3", "-2", "-4"]);
}

#[test]
fn board_white_pawns_rank1() {
    let src = chess_program(r#"
        let b = init_board();
        let i = 8;
        let all_ones = true;
        while i < 16 {
            if b[i] != 1 { all_ones = false; }
            i = i + 1;
        }
        print(all_ones);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true");
}

#[test]
fn board_black_pawns_rank6() {
    let src = chess_program(r#"
        let b = init_board();
        let i = 48;
        let all_neg_ones = true;
        while i < 56 {
            if b[i] != -1 { all_neg_ones = false; }
            i = i + 1;
        }
        print(all_neg_ones);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true");
}

#[test]
fn board_empty_middle_ranks() {
    let src = chess_program(r#"
        let b = init_board();
        let i = 16;
        let all_empty = true;
        while i < 48 {
            if b[i] != 0 { all_empty = false; }
            i = i + 1;
        }
        print(all_empty);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true");
}

#[test]
fn board_piece_count_white_16() {
    let src = chess_program(r#"
        let b = init_board();
        let count = 0;
        let i = 0;
        while i < 64 {
            if b[i] > 0 { count = count + 1; }
            i = i + 1;
        }
        print(count);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "16");
}

#[test]
fn board_piece_count_black_16() {
    let src = chess_program(r#"
        let b = init_board();
        let count = 0;
        let i = 0;
        while i < 64 {
            if b[i] < 0 { count = count + 1; }
            i = i + 1;
        }
        print(count);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "16");
}

// ============================================================
// Square arithmetic
// ============================================================

#[test]
fn rank_of_correct() {
    let src = chess_program(r#"
        print(rank_of(0));  print(rank_of(7));
        print(rank_of(8));  print(rank_of(63));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out, vec!["0", "0", "1", "7"]);
}

#[test]
fn file_of_correct() {
    let src = chess_program(r#"
        print(file_of(0));  print(file_of(7));
        print(file_of(8));  print(file_of(63));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out, vec!["0", "7", "0", "7"]);
}

#[test]
fn sq_of_inverse() {
    // sq_of(rank_of(sq), file_of(sq)) == sq for all 64
    let src = chess_program(r#"
        let ok = true;
        let sq = 0;
        while sq < 64 {
            if sq_of(rank_of(sq), file_of(sq)) != sq { ok = false; }
            sq = sq + 1;
        }
        print(ok);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true");
}

// ============================================================
// apply_move correctness
// ============================================================

#[test]
fn apply_move_preserves_board_size() {
    let src = chess_program(r#"
        let b = init_board();
        let b2 = apply_move(b, 12, 28);
        print(len(b2));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "64");
}

#[test]
fn apply_move_clears_source_square() {
    let src = chess_program(r#"
        let b = init_board();
        let b2 = apply_move(b, 12, 28);
        print(b2[12]); print(b2[28]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "0", "source square should be empty");
    assert_eq!(out[1], "1", "target square should have pawn");
}

#[test]
fn apply_move_does_not_mutate_original() {
    let src = chess_program(r#"
        let b = init_board();
        let b2 = apply_move(b, 12, 28);
        print(b[12]); print(b[28]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "1", "original should still have pawn at source");
    assert_eq!(out[1], "0", "original should still be empty at target");
}

#[test]
fn apply_move_pawn_promotion_white() {
    // White pawn at rank 6 (sq 48+file) moving to rank 7 should promote to queen (5)
    let src = chess_program(r#"
        let b = [];
        let i = 0;
        while i < 64 {
            b = array_push(b, 0);
            i = i + 1;
        }
        // Put white pawn at e7 (sq 52) and white king at a1 (sq 0), black king at h8 (sq 63)
        b[52] = 1;
        b[0] = 6;
        b[63] = -6;
        let b2 = apply_move(b, 52, 60);
        print(b2[60]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "5", "white pawn should promote to queen");
}

// ============================================================
// King finding
// ============================================================

#[test]
fn find_king_initial_positions() {
    let src = chess_program(r#"
        let b = init_board();
        print(find_king(b, 1));
        print(find_king(b, -1));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "4", "white king at e1");
    assert_eq!(out[1], "60", "black king at e8");
}

#[test]
fn find_king_returns_negative_for_missing() {
    let src = chess_program(r#"
        let b = [];
        let i = 0;
        while i < 64 {
            b = array_push(b, 0);
            i = i + 1;
        }
        print(find_king(b, 1));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "-1", "no king should return -1");
}

// ============================================================
// Check detection
// ============================================================

#[test]
fn initial_position_not_in_check() {
    let src = chess_program(r#"
        let b = init_board();
        print(in_check(b, 1));
        print(in_check(b, -1));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "false");
    assert_eq!(out[1], "false");
}

#[test]
fn sign_function_correct() {
    let src = chess_program(r#"
        print(sign(5)); print(sign(-3)); print(sign(0));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out, vec!["1", "-1", "0"]);
}
