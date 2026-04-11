//! CJC-Lang source for the chess RL v2 demo.
//!
//! This is written as a single `pub const PRELUDE` string. Each test appends
//! its own `fn main() {...}` caller and feeds the concatenation through the
//! CJC-Lang compiler. Both executors (cjc-eval and cjc-mir-exec) consume the
//! same string, which makes parity testing a simple `assert_eq!` on output.
//!
//! # Sections
//! The source is organized into labeled sections. Search markers:
//!   - `// ============== ENGINE` — board/state/movegen/rules
//!   - `// ============== FEATURES` — state → 774-D tensor
//!   - `// ============== MODEL` — GradGraph dual-head network
//!   - `// ============== TRAINING` — A2C + GAE rollout + update
//!   - `// ============== EVAL` — random / snapshot arenas
//!
//! # Conventions
//! - `state = [board, side, castling, ep_sq, halfmove, ply]` — array-indexed
//!   "struct". `castling = [wk, wq, bk, bq]` each ∈ {0, 1}. `side ∈ {-1, 1}`.
//!   `ep_sq` is the square to which a pawn can en-passant capture, or -1.
//! - Moves are flat arrays `[from0, to0, from1, to1, ...]` — compatible with
//!   the factored policy head. Pawns auto-promote to queen on the final rank.
//! - All reward values flow from the checkmated side's perspective: the side
//!   delivering checkmate is +1, the mated side is -1, draws are 0.

pub const PRELUDE: &str = r#"
import vizor

// ==========================================================================
// ============== ENGINE
// ==========================================================================
// Board: flat 64-int array. Piece encoding:
//   0 empty
//   1 white pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king
//  -1 black pawn, -2 knight, ... -6 king
// Squares: index = rank*8 + file. Rank 0 = white back rank, rank 7 = black.

fn init_board() -> Any {
    [
        4, 2, 3, 5, 6, 3, 2, 4,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
       -1,-1,-1,-1,-1,-1,-1,-1,
       -4,-2,-3,-5,-6,-3,-2,-4
    ]
}

fn init_state() -> Any {
    // [board, side, castling, ep_sq, halfmove, ply]
    [init_board(), 1, [1, 1, 1, 1], 0 - 1, 0, 0]
}

// ---- State accessors ----
fn state_board(s: Any) -> Any { s[0] }
fn state_side(s: Any) -> i64 { s[1] }
fn state_castling(s: Any) -> Any { s[2] }
fn state_ep(s: Any) -> i64 { s[3] }
fn state_halfmove(s: Any) -> i64 { s[4] }
fn state_ply(s: Any) -> i64 { s[5] }

fn make_state(board: Any, side: i64, castling: Any, ep_sq: i64, halfmove: i64, ply: i64) -> Any {
    [board, side, castling, ep_sq, halfmove, ply]
}

// ---- Geometry ----
fn rank_of(sq: i64) -> i64 { sq / 8 }
fn file_of(sq: i64) -> i64 { sq % 8 }
fn sq_of(rank: i64, file: i64) -> i64 { rank * 8 + file }
fn on_board(rank: i64, file: i64) -> bool {
    rank >= 0 && rank < 8 && file >= 0 && file < 8
}
fn sign_of(x: i64) -> i64 {
    if x > 0 { 1 } else { if x < 0 { 0 - 1 } else { 0 } }
}
fn piece_side(p: i64) -> i64 { sign_of(p) }
fn abs_i(x: i64) -> i64 { if x < 0 { 0 - x } else { x } }

// ---- Board copy with a list of [idx, val] edits ----
// `edits` is a flat array [idx0, val0, idx1, val1, ...]; later edits override
// earlier ones. O(64 * num_edits) but num_edits is tiny per move.
fn board_with_edits(board: Any, edits: Any) -> Any {
    let out = [];
    let i = 0;
    while i < 64 {
        let v = board[i];
        let j = 0;
        while j < len(edits) {
            if edits[j] == i {
                v = edits[j + 1];
            }
            j = j + 2;
        }
        out = array_push(out, v);
        i = i + 1;
    }
    out
}

// ---- Attack detection (board-only, no state needed) ----
fn is_attacked_by(board: Any, sq: i64, by_side: i64) -> bool {
    let r = rank_of(sq);
    let f = file_of(sq);
    // Knight
    let knight = 2 * by_side;
    let kd = [[0-2,0-1],[0-2,1],[0-1,0-2],[0-1,2],[1,0-2],[1,2],[2,0-1],[2,1]];
    let ki = 0;
    while ki < 8 {
        let nr = r + kd[ki][0];
        let nf = f + kd[ki][1];
        if on_board(nr, nf) {
            if board[sq_of(nr, nf)] == knight { return true; }
        }
        ki = ki + 1;
    }
    // Pawn
    let pawn = 1 * by_side;
    let pawn_dir = 0 - by_side;
    if on_board(r + pawn_dir, f - 1) {
        if board[sq_of(r + pawn_dir, f - 1)] == pawn { return true; }
    }
    if on_board(r + pawn_dir, f + 1) {
        if board[sq_of(r + pawn_dir, f + 1)] == pawn { return true; }
    }
    // King
    let king = 6 * by_side;
    let di = 0 - 1;
    while di <= 1 {
        let dj = 0 - 1;
        while dj <= 1 {
            if di != 0 || dj != 0 {
                let kr = r + di;
                let kf = f + dj;
                if on_board(kr, kf) {
                    if board[sq_of(kr, kf)] == king { return true; }
                }
            }
            dj = dj + 1;
        }
        di = di + 1;
    }
    // Diagonals: bishop / queen
    let bishop = 3 * by_side;
    let queen = 5 * by_side;
    let diag = [[1,1],[1,0-1],[0-1,1],[0-1,0-1]];
    let dd = 0;
    while dd < 4 {
        let cr = r + diag[dd][0];
        let cf = f + diag[dd][1];
        while on_board(cr, cf) {
            let p = board[sq_of(cr, cf)];
            if p != 0 {
                if p == bishop || p == queen { return true; }
                break;
            }
            cr = cr + diag[dd][0];
            cf = cf + diag[dd][1];
        }
        dd = dd + 1;
    }
    // Straights: rook / queen
    let rook = 4 * by_side;
    let straight = [[1,0],[0-1,0],[0,1],[0,0-1]];
    let sd = 0;
    while sd < 4 {
        let cr = r + straight[sd][0];
        let cf = f + straight[sd][1];
        while on_board(cr, cf) {
            let p = board[sq_of(cr, cf)];
            if p != 0 {
                if p == rook || p == queen { return true; }
                break;
            }
            cr = cr + straight[sd][0];
            cf = cf + straight[sd][1];
        }
        sd = sd + 1;
    }
    false
}

fn find_king(board: Any, side: i64) -> i64 {
    let king = 6 * side;
    let i = 0;
    while i < 64 {
        if board[i] == king { return i; }
        i = i + 1;
    }
    0 - 1
}

fn in_check_raw(board: Any, side: i64) -> bool {
    let ksq = find_king(board, side);
    if ksq < 0 { return true; }
    is_attacked_by(board, ksq, 0 - side)
}

fn in_check(state: Any) -> bool {
    in_check_raw(state_board(state), state_side(state))
}

// ---- Apply move ----
// Handles: pawn promotions (auto-queen), pawn double push (sets ep_sq),
// en passant captures, castling (king moves 2 -> rook also moves), halfmove
// reset on pawn/capture, castling rights updates on king/rook moves/captures.
fn apply_move(state: Any, from_sq: i64, to_sq: i64) -> Any {
    let board = state_board(state);
    let side = state_side(state);
    let castling = state_castling(state);
    let ep_sq = state_ep(state);
    let halfmove = state_halfmove(state);
    let ply = state_ply(state);

    let piece = board[from_sq];
    let abs_p = abs_i(piece);
    let captured = board[to_sq];

    // Default edits: from -> 0, to -> piece.
    let edits = [];
    edits = array_push(edits, from_sq);
    edits = array_push(edits, 0);
    edits = array_push(edits, to_sq);
    edits = array_push(edits, piece);

    let new_wk = castling[0];
    let new_wq = castling[1];
    let new_bk = castling[2];
    let new_bq = castling[3];
    let new_ep = 0 - 1;
    let new_halfmove = halfmove + 1;

    // Pawn moves: reset halfmove, detect double-push / EP / promotion.
    if abs_p == 1 {
        new_halfmove = 0;
        let dr = rank_of(to_sq) - rank_of(from_sq);
        if dr == 2 || dr == 0 - 2 {
            new_ep = sq_of((rank_of(from_sq) + rank_of(to_sq)) / 2, file_of(from_sq));
        }
        // En passant capture
        if to_sq == ep_sq && ep_sq >= 0 {
            let victim_sq = sq_of(rank_of(from_sq), file_of(to_sq));
            edits = array_push(edits, victim_sq);
            edits = array_push(edits, 0);
        }
        // Promotion (auto-queen)
        if rank_of(to_sq) == 7 && side == 1 {
            edits = array_push(edits, to_sq);
            edits = array_push(edits, 5);
        }
        if rank_of(to_sq) == 0 && side == 0 - 1 {
            edits = array_push(edits, to_sq);
            edits = array_push(edits, 0 - 5);
        }
    }

    // Non-EP capture resets halfmove.
    if captured != 0 {
        new_halfmove = 0;
    }

    // King moves: handle castling, clear both castling rights for that side.
    if abs_p == 6 {
        let df = file_of(to_sq) - file_of(from_sq);
        if df == 2 || df == 0 - 2 {
            let home_r = rank_of(from_sq);
            let rook_from = 0;
            let rook_to = 0;
            if df == 2 {
                rook_from = sq_of(home_r, 7);
                rook_to = sq_of(home_r, 5);
            } else {
                rook_from = sq_of(home_r, 0);
                rook_to = sq_of(home_r, 3);
            }
            let rook_piece = 4;
            if side == 0 - 1 { rook_piece = 0 - 4; }
            edits = array_push(edits, rook_from);
            edits = array_push(edits, 0);
            edits = array_push(edits, rook_to);
            edits = array_push(edits, rook_piece);
        }
        if side == 1 {
            new_wk = 0;
            new_wq = 0;
        } else {
            new_bk = 0;
            new_bq = 0;
        }
    }

    // Rook move from corner: clear the matching castling right.
    if abs_p == 4 {
        if from_sq == 0 { new_wq = 0; }
        if from_sq == 7 { new_wk = 0; }
        if from_sq == 56 { new_bq = 0; }
        if from_sq == 63 { new_bk = 0; }
    }
    // Rook captured at corner: clear the matching right.
    if to_sq == 0 { new_wq = 0; }
    if to_sq == 7 { new_wk = 0; }
    if to_sq == 56 { new_bq = 0; }
    if to_sq == 63 { new_bk = 0; }

    let new_board = board_with_edits(board, edits);
    make_state(new_board, 0 - side, [new_wk, new_wq, new_bk, new_bq], new_ep, new_halfmove, ply + 1)
}

// ---- Pseudo-legal move generation ----
// Returns [from0, to0, from1, to1, ...]. Pawns auto-promote (encoded as a
// normal from/to; apply_move performs the promotion). Castling is generated
// here but with the full legality check (path empty, not in/through check).
fn generate_pseudo_legal(state: Any) -> Any {
    let board = state_board(state);
    let side = state_side(state);
    let ep_sq = state_ep(state);
    let castling = state_castling(state);
    let moves = [];
    let sq = 0;
    while sq < 64 {
        let p = board[sq];
        if piece_side(p) == side {
            let abs_p = abs_i(p);
            let r = rank_of(sq);
            let f = file_of(sq);
            if abs_p == 1 {
                let dir = side;
                let start_rank = 1;
                if side != 1 { start_rank = 6; }
                // Forward one
                let nr = r + dir;
                if on_board(nr, f) && board[sq_of(nr, f)] == 0 {
                    moves = array_push(moves, sq);
                    moves = array_push(moves, sq_of(nr, f));
                    // Forward two
                    let nr2 = r + 2 * dir;
                    if r == start_rank && board[sq_of(nr2, f)] == 0 {
                        moves = array_push(moves, sq);
                        moves = array_push(moves, sq_of(nr2, f));
                    }
                }
                // Captures
                if on_board(nr, f - 1) {
                    let cap = board[sq_of(nr, f - 1)];
                    if cap != 0 && piece_side(cap) == 0 - side {
                        moves = array_push(moves, sq);
                        moves = array_push(moves, sq_of(nr, f - 1));
                    }
                }
                if on_board(nr, f + 1) {
                    let cap = board[sq_of(nr, f + 1)];
                    if cap != 0 && piece_side(cap) == 0 - side {
                        moves = array_push(moves, sq);
                        moves = array_push(moves, sq_of(nr, f + 1));
                    }
                }
                // En passant: adjacent file matches ep_sq
                if ep_sq >= 0 {
                    if on_board(nr, f - 1) && sq_of(nr, f - 1) == ep_sq {
                        moves = array_push(moves, sq);
                        moves = array_push(moves, ep_sq);
                    }
                    if on_board(nr, f + 1) && sq_of(nr, f + 1) == ep_sq {
                        moves = array_push(moves, sq);
                        moves = array_push(moves, ep_sq);
                    }
                }
            }
            if abs_p == 2 {
                let kd = [[0-2,0-1],[0-2,1],[0-1,0-2],[0-1,2],[1,0-2],[1,2],[2,0-1],[2,1]];
                let ki = 0;
                while ki < 8 {
                    let nr = r + kd[ki][0];
                    let nf = f + kd[ki][1];
                    if on_board(nr, nf) {
                        let target = board[sq_of(nr, nf)];
                        if target == 0 || piece_side(target) == 0 - side {
                            moves = array_push(moves, sq);
                            moves = array_push(moves, sq_of(nr, nf));
                        }
                    }
                    ki = ki + 1;
                }
            }
            if abs_p == 3 || abs_p == 5 {
                let dirs = [[1,1],[1,0-1],[0-1,1],[0-1,0-1]];
                let d = 0;
                while d < 4 {
                    let cr = r + dirs[d][0];
                    let cf = f + dirs[d][1];
                    while on_board(cr, cf) {
                        let target = board[sq_of(cr, cf)];
                        if target == 0 {
                            moves = array_push(moves, sq);
                            moves = array_push(moves, sq_of(cr, cf));
                        } else {
                            if piece_side(target) == 0 - side {
                                moves = array_push(moves, sq);
                                moves = array_push(moves, sq_of(cr, cf));
                            }
                            break;
                        }
                        cr = cr + dirs[d][0];
                        cf = cf + dirs[d][1];
                    }
                    d = d + 1;
                }
            }
            if abs_p == 4 || abs_p == 5 {
                let dirs = [[1,0],[0-1,0],[0,1],[0,0-1]];
                let d = 0;
                while d < 4 {
                    let cr = r + dirs[d][0];
                    let cf = f + dirs[d][1];
                    while on_board(cr, cf) {
                        let target = board[sq_of(cr, cf)];
                        if target == 0 {
                            moves = array_push(moves, sq);
                            moves = array_push(moves, sq_of(cr, cf));
                        } else {
                            if piece_side(target) == 0 - side {
                                moves = array_push(moves, sq);
                                moves = array_push(moves, sq_of(cr, cf));
                            }
                            break;
                        }
                        cr = cr + dirs[d][0];
                        cf = cf + dirs[d][1];
                    }
                    d = d + 1;
                }
            }
            if abs_p == 6 {
                // Normal king moves
                let di = 0 - 1;
                while di <= 1 {
                    let dj = 0 - 1;
                    while dj <= 1 {
                        if di != 0 || dj != 0 {
                            let nr = r + di;
                            let nf = f + dj;
                            if on_board(nr, nf) {
                                let target = board[sq_of(nr, nf)];
                                if target == 0 || piece_side(target) == 0 - side {
                                    moves = array_push(moves, sq);
                                    moves = array_push(moves, sq_of(nr, nf));
                                }
                            }
                        }
                        dj = dj + 1;
                    }
                    di = di + 1;
                }
                // Castling (fully legality-checked here)
                let home_r = 0;
                if side != 1 { home_r = 7; }
                let king_home = sq_of(home_r, 4);
                if sq == king_home && !in_check_raw(board, side) {
                    let ksi = 0;
                    let qsi = 1;
                    if side != 1 { ksi = 2; qsi = 3; }
                    if castling[ksi] == 1 {
                        let f_sq = sq_of(home_r, 5);
                        let g_sq = sq_of(home_r, 6);
                        if board[f_sq] == 0 && board[g_sq] == 0 {
                            if !is_attacked_by(board, f_sq, 0 - side) {
                                if !is_attacked_by(board, g_sq, 0 - side) {
                                    moves = array_push(moves, sq);
                                    moves = array_push(moves, g_sq);
                                }
                            }
                        }
                    }
                    if castling[qsi] == 1 {
                        let b_sq = sq_of(home_r, 1);
                        let c_sq = sq_of(home_r, 2);
                        let d_sq = sq_of(home_r, 3);
                        if board[b_sq] == 0 && board[c_sq] == 0 && board[d_sq] == 0 {
                            if !is_attacked_by(board, c_sq, 0 - side) {
                                if !is_attacked_by(board, d_sq, 0 - side) {
                                    moves = array_push(moves, sq);
                                    moves = array_push(moves, c_sq);
                                }
                            }
                        }
                    }
                }
            }
        }
        sq = sq + 1;
    }
    moves
}

// ---- Legal move filter: the side to move must not be in check afterwards ----
fn legal_moves(state: Any) -> Any {
    let pseudo = generate_pseudo_legal(state);
    let legal = [];
    let side = state_side(state);
    let i = 0;
    while i < len(pseudo) {
        let from_sq = pseudo[i];
        let to_sq = pseudo[i + 1];
        let after = apply_move(state, from_sq, to_sq);
        // After apply_move, side-to-move has flipped. The mover is `side`,
        // which we need to check is NOT in check on the new board.
        if !in_check_raw(state_board(after), side) {
            legal = array_push(legal, from_sq);
            legal = array_push(legal, to_sq);
        }
        i = i + 2;
    }
    legal
}

// ---- Insufficient material ----
// Returns true for K-K, K-N, K-B, K+B vs K+B with same-colored bishops.
fn insufficient_material(board: Any) -> bool {
    let wn = 0; let wb_l = 0; let wb_d = 0; let w_other = 0;
    let bn = 0; let bb_l = 0; let bb_d = 0; let b_other = 0;
    let i = 0;
    while i < 64 {
        let p = board[i];
        let color = (rank_of(i) + file_of(i)) % 2;
        if p == 2 { wn = wn + 1; }
        if p == 3 {
            if color == 1 { wb_l = wb_l + 1; } else { wb_d = wb_d + 1; }
        }
        if p == 0 - 2 { bn = bn + 1; }
        if p == 0 - 3 {
            if color == 1 { bb_l = bb_l + 1; } else { bb_d = bb_d + 1; }
        }
        if p == 1 || p == 4 || p == 5 { w_other = w_other + 1; }
        if p == 0 - 1 || p == 0 - 4 || p == 0 - 5 { b_other = b_other + 1; }
        i = i + 1;
    }
    if w_other > 0 || b_other > 0 { return false; }
    let w_minors = wn + wb_l + wb_d;
    let b_minors = bn + bb_l + bb_d;
    if w_minors == 0 && b_minors == 0 { return true; }
    if w_minors == 1 && b_minors == 0 { return true; }
    if w_minors == 0 && b_minors == 1 { return true; }
    // K+B vs K+B, same-color bishops
    if wn == 0 && bn == 0 && wb_d == 0 && bb_d == 0 { return true; }
    if wn == 0 && bn == 0 && wb_l == 0 && bb_l == 0 { return true; }
    false
}

// ---- Terminal state ----
// 0 = ongoing; 2 = checkmate (side to move loses); 3 = stalemate;
// 4 = 50-move rule; 5 = insufficient material.
fn terminal_status(state: Any) -> i64 {
    let moves = legal_moves(state);
    if len(moves) == 0 {
        if in_check(state) { return 2; }
        return 3;
    }
    if state_halfmove(state) >= 100 { return 4; }
    if insufficient_material(state_board(state)) { return 5; }
    0
}

// ==========================================================================
// ============== FEATURES
// ==========================================================================
// Build a 774-dim float feature vector from the perspective of the side to
// move. Layout:
//   [0..768)   12 piece-type planes x 64 squares, each either 1.0 or 0.0
//              Planes 0..6   — my pawn, knight, bishop, rook, queen, king
//              Planes 6..12  — enemy pawn, knight, bishop, rook, queen, king
//              Squares are flipped so "my back rank" is always rank 0.
//   768        my kingside castling right
//   769        my queenside castling right
//   770        enemy kingside castling right
//   771        enemy queenside castling right
//   772        halfmove clock / 100.0 (50-move proximity)
//   773        has en passant target (1.0 or 0.0)

fn feat_sq(sq: i64, side: i64) -> i64 {
    // Flip the board for black so "my side" is always on the bottom.
    if side == 1 { sq } else { sq_of(7 - rank_of(sq), file_of(sq)) }
}

fn encode_state(state: Any) -> Tensor {
    let board = state_board(state);
    let side = state_side(state);
    let castling = state_castling(state);
    let ep_sq = state_ep(state);
    let halfmove = state_halfmove(state);

    let data = [];
    // Initialize 774 zeros
    let z = 0;
    while z < 774 {
        data = array_push(data, 0.0);
        z = z + 1;
    }

    // Fill in piece planes.
    let i = 0;
    while i < 64 {
        let p = board[i];
        if p != 0 {
            let abs_p = abs_i(p);
            let piece_idx = abs_p - 1;           // 0..5
            let owner = piece_side(p);
            let plane = piece_idx;
            if owner != side { plane = piece_idx + 6; }
            let mapped = feat_sq(i, side);
            let idx = plane * 64 + mapped;
            // data[idx] = 1.0 via array_set_at: we rebuild since arrays are immutable
            // Use a small helper: arr_set
            data = arr_set(data, idx, 1.0);
        }
        i = i + 1;
    }

    // Castling (flipped if black to move)
    let wk = castling[0];
    let wq = castling[1];
    let bk = castling[2];
    let bq = castling[3];
    let my_k = wk; let my_q = wq; let op_k = bk; let op_q = bq;
    if side != 1 { my_k = bk; my_q = bq; op_k = wk; op_q = wq; }
    data = arr_set(data, 768, float(my_k));
    data = arr_set(data, 769, float(my_q));
    data = arr_set(data, 770, float(op_k));
    data = arr_set(data, 771, float(op_q));

    // Halfmove clock fraction
    data = arr_set(data, 772, float(halfmove) / 100.0);
    // Has EP
    let has_ep = 0.0;
    if ep_sq >= 0 { has_ep = 1.0; }
    data = arr_set(data, 773, has_ep);

    Tensor.from_vec(data, [1, 774])
}

// Set arr[idx] := val via a copy. Linear-time but called O(32) times per
// state — still cheap compared to the matmul that follows.
fn arr_set(arr: Any, idx: i64, val: f64) -> Any {
    let out = [];
    let i = 0;
    while i < len(arr) {
        if i == idx {
            out = array_push(out, val);
        } else {
            out = array_push(out, arr[i]);
        }
        i = i + 1;
    }
    out
}

// ==========================================================================
// ============== MODEL
// ==========================================================================
// Dual-head MLP with factored policy:
//   trunk: [1,774] -> W1 -> relu -> W2 -> relu -> h[1,48]
//   from_logits: h -> Wp_from -> [1,64]  (scores per source square)
//   to_logits:   h -> Wp_to   -> [1,64]  (scores per destination square)
//   value:       h -> Wv      -> tanh    (scalar in [-1, 1])
//
// Parameters are stored as a flat array in a fixed order so the rest of the
// code can refer to them by index:
//   0: W1 [774, 48]      5: Wp_from [48, 64]
//   1: b1 [1, 48]        6: bp_from [1, 64]
//   2: W2 [48, 48]       7: Wp_to   [48, 64]
//   3: b2 [1, 48]        8: bp_to   [1, 64]
//   4: (reserved)        9: Wv      [48, 1]
//                       10: bv      [1, 1]

fn init_weights() -> Any {
    let W1 = Tensor.randn([774, 48]) * 0.05;
    let b1 = Tensor.zeros([1, 48]);
    let W2 = Tensor.randn([48, 48]) * 0.1;
    let b2 = Tensor.zeros([1, 48]);
    let Wpf = Tensor.randn([48, 64]) * 0.1;
    let bpf = Tensor.zeros([1, 64]);
    let Wpt = Tensor.randn([48, 64]) * 0.1;
    let bpt = Tensor.zeros([1, 64]);
    let Wv = Tensor.randn([48, 1]) * 0.1;
    let bv = Tensor.zeros([1, 1]);
    [W1, b1, W2, b2, 0, Wpf, bpf, Wpt, bpt, Wv, bv]
}

// ---- Forward pass (eager, no GradGraph) for action selection & evaluation ----
// Returns [from_logits_row_1x64, to_logits_row_1x64, value_scalar].
fn forward_eager(weights: Any, features: Tensor) -> Any {
    let W1 = weights[0];
    let b1 = weights[1];
    let W2 = weights[2];
    let b2 = weights[3];
    let Wpf = weights[5];
    let bpf = weights[6];
    let Wpt = weights[7];
    let bpt = weights[8];
    let Wv = weights[9];
    let bv = weights[10];
    let z1 = matmul(features, W1) + b1;
    let h1 = z1.relu();
    let z2 = matmul(h1, W2) + b2;
    let h2 = z2.relu();
    let from_logits = matmul(h2, Wpf) + bpf;
    let to_logits = matmul(h2, Wpt) + bpt;
    let v_pre = matmul(h2, Wv) + bv;
    let v = tanh(v_pre);
    [from_logits, to_logits, v.get([0, 0])]
}

// ---- Score every legal move by summing from/to logits ----
// Returns [scores_1d_tensor, value]. Scores are logits; softmax upstream.
fn score_moves(weights: Any, state: Any, moves: Any) -> Any {
    let features = encode_state(state);
    let fwd = forward_eager(weights, features);
    let from_logits = fwd[0];
    let to_logits = fwd[1];
    let v = fwd[2];
    let side = state_side(state);
    let num = len(moves) / 2;
    let scores = [];
    let i = 0;
    while i < num {
        let from_sq = moves[i * 2];
        let to_sq = moves[i * 2 + 1];
        let from_mapped = feat_sq(from_sq, side);
        let to_mapped = feat_sq(to_sq, side);
        let s = from_logits.get([0, from_mapped]) + to_logits.get([0, to_mapped]);
        scores = array_push(scores, s);
        i = i + 1;
    }
    let scores_t = Tensor.from_vec(scores, [num]);
    [scores_t, v]
}

// ---- Select one action stochastically via softmax ----
// Returns [action_idx, log_prob, value].
fn select_action(weights: Any, state: Any, moves: Any) -> Any {
    let sv = score_moves(weights, state, moves);
    let scores_t = sv[0];
    let v = sv[1];
    let probs = scores_t.softmax();
    let action = categorical_sample(probs);
    let lp = log(probs.get([action]));
    [action, lp, v]
}

// ---- Temperature-annealed action selection (Phase B3) ----
// `temp > 0` scales logits before softmax. temp=1.0 reproduces select_action;
// temp→0 collapses toward argmax (greedy); temp>1 smooths toward uniform.
// Returns [action_idx, log_prob, value].
fn select_action_temp(weights: Any, state: Any, moves: Any, temp: f64) -> Any {
    let sv = score_moves(weights, state, moves);
    let scores_t = sv[0];
    let v = sv[1];
    let inv_t = 1.0 / temp;
    let scaled = scores_t * inv_t;
    let probs = scaled.softmax();
    let action = categorical_sample(probs);
    let lp = log(probs.get([action]));
    [action, lp, v]
}

// ---- Linear annealing schedule: temp(ep) = t_start + (t_end-t_start)*ep/ep_max ----
// Clamped so values past ep_max stay at t_end. Returns an f64 in [t_end, t_start].
fn anneal_temp(ep: i64, ep_max: i64, t_start: f64, t_end: f64) -> f64 {
    if ep_max <= 0 { return t_end; }
    let e = ep;
    if e >= ep_max { e = ep_max; }
    if e < 0 { e = 0; }
    let frac = float(e) / float(ep_max);
    t_start + (t_end - t_start) * frac
}

// ---- Greedy (argmax) action for evaluation play ----
fn select_action_greedy(weights: Any, state: Any, moves: Any) -> Any {
    let sv = score_moves(weights, state, moves);
    let scores_t = sv[0];
    let v = sv[1];
    let num = len(moves) / 2;
    let best = 0;
    let best_score = scores_t.get([0]);
    let i = 1;
    while i < num {
        let s = scores_t.get([i]);
        if s > best_score {
            best_score = s;
            best = i;
        }
        i = i + 1;
    }
    [best, 0.0, v]
}

// ==========================================================================
// ============== TRAINING
// ==========================================================================
// Rollout one self-play episode, then run a single A2C + GAE update.

// ---- Rollout one self-play game, collecting trajectory data. ----
// Returns:
//   [states_list, moves_list, action_list, value_list, reward_sides_list,
//    terminal_reward, num_moves]
// where:
//   states_list[t]    — GameState at step t
//   moves_list[t]     — legal moves at step t
//   action_list[t]    — selected action index at step t
//   value_list[t]     — predicted V(s_t) scalar
//   reward_sides[t]   — side-to-move at step t (+1 or -1)
//   terminal_reward   — game outcome in white-centric sign (+1 white wins)
fn rollout_episode(weights: Any, max_moves: i64) -> Any {
    let state = init_state();
    let states_list = [];
    let moves_list = [];
    let action_list = [];
    let value_list = [];
    let side_list = [];
    let step = 0;
    let terminal_reward = 0.0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            // Side to move is checkmated -> opponent wins
            terminal_reward = float(0 - state_side(state));
            step = max_moves + 1;
        } else {
            if status != 0 {
                // Any draw: 50-move, stalemate, insufficient material
                terminal_reward = 0.0;
                step = max_moves + 1;
            } else {
                let moves = legal_moves(state);
                let sel = select_action(weights, state, moves);
                let a = sel[0];
                let v = sel[2];
                states_list = array_push(states_list, state);
                moves_list = array_push(moves_list, moves);
                action_list = array_push(action_list, a);
                value_list = array_push(value_list, v);
                side_list = array_push(side_list, state_side(state));
                let from_sq = moves[a * 2];
                let to_sq = moves[a * 2 + 1];
                state = apply_move(state, from_sq, to_sq);
                step = step + 1;
            }
        }
    }
    [states_list, moves_list, action_list, value_list, side_list, terminal_reward, len(states_list)]
}

// ---- Resignation-enabled rollout (Phase B3 + B4) ----
// Like rollout_episode_temp but also implements resignation: if the side-to-
// move's value estimate stays below `resign_thresh` for `resign_patience`
// consecutive plies, the current side resigns and the opponent wins. This
// prevents wasting compute on hopeless trajectories. `resign_patience <= 0`
// disables the check entirely (same behavior as rollout_episode_temp).
fn rollout_episode_full(weights: Any, max_moves: i64, temp: f64,
                        resign_thresh: f64, resign_patience: i64) -> Any {
    let state = init_state();
    let states_list = [];
    let moves_list = [];
    let action_list = [];
    let value_list = [];
    let side_list = [];
    let step = 0;
    let terminal_reward = 0.0;
    let bad_count = 0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            terminal_reward = float(0 - state_side(state));
            step = max_moves + 1;
        } else {
            if status != 0 {
                terminal_reward = 0.0;
                step = max_moves + 1;
            } else {
                let moves = legal_moves(state);
                let sel = select_action_temp(weights, state, moves, temp);
                let a = sel[0];
                let v = sel[2];
                // Resignation check: v is from side-to-move's perspective.
                if v < resign_thresh {
                    bad_count = bad_count + 1;
                } else {
                    bad_count = 0;
                }
                states_list = array_push(states_list, state);
                moves_list = array_push(moves_list, moves);
                action_list = array_push(action_list, a);
                value_list = array_push(value_list, v);
                side_list = array_push(side_list, state_side(state));
                if resign_patience > 0 && bad_count >= resign_patience {
                    // Side-to-move resigns — terminal in white-centric sign.
                    terminal_reward = float(0 - state_side(state));
                    step = max_moves + 1;
                } else {
                    let from_sq = moves[a * 2];
                    let to_sq = moves[a * 2 + 1];
                    state = apply_move(state, from_sq, to_sq);
                    step = step + 1;
                }
            }
        }
    }
    [states_list, moves_list, action_list, value_list, side_list, terminal_reward, len(states_list)]
}

// ---- Temperature-annealed rollout (Phase B3) ----
// Same as rollout_episode but uses select_action_temp with the given temp.
fn rollout_episode_temp(weights: Any, max_moves: i64, temp: f64) -> Any {
    let state = init_state();
    let states_list = [];
    let moves_list = [];
    let action_list = [];
    let value_list = [];
    let side_list = [];
    let step = 0;
    let terminal_reward = 0.0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            terminal_reward = float(0 - state_side(state));
            step = max_moves + 1;
        } else {
            if status != 0 {
                terminal_reward = 0.0;
                step = max_moves + 1;
            } else {
                let moves = legal_moves(state);
                let sel = select_action_temp(weights, state, moves, temp);
                let a = sel[0];
                let v = sel[2];
                states_list = array_push(states_list, state);
                moves_list = array_push(moves_list, moves);
                action_list = array_push(action_list, a);
                value_list = array_push(value_list, v);
                side_list = array_push(side_list, state_side(state));
                let from_sq = moves[a * 2];
                let to_sq = moves[a * 2 + 1];
                state = apply_move(state, from_sq, to_sq);
                step = step + 1;
            }
        }
    }
    [states_list, moves_list, action_list, value_list, side_list, terminal_reward, len(states_list)]
}

// ---- Compute GAE advantages and returns from a collected rollout. ----
// Rewards are all 0 except the terminal reward, which applies to the side
// that delivered the final move. We flip the sign per position so each side
// sees its own outcome as positive if it won.
fn compute_gae(value_list: Any, side_list: Any, terminal_reward: f64, gamma: f64, lam: f64) -> Any {
    let n = len(value_list);
    let advantages = [];
    let returns = [];
    let i = 0;
    while i < n {
        advantages = array_push(advantages, 0.0);
        returns = array_push(returns, 0.0);
        i = i + 1;
    }
    // Bellman / GAE in reverse, propagating the terminal reward back.
    let last_advantage = 0.0;
    let t = n - 1;
    while t >= 0 {
        // Reward observed when transitioning OUT of state t. Only nonzero
        // at the final step before termination.
        let r = 0.0;
        if t == n - 1 {
            // Reward is white-centric; flip for the side whose turn it is.
            let s_side = side_list[t];
            r = terminal_reward * float(s_side);
        }
        // Next value: 0 at terminal (the game is over after step n-1).
        let v_next = 0.0;
        if t < n - 1 {
            // Next state was observed by the opposing side. Because the
            // perspective flips, the sign of V flips too.
            v_next = 0.0 - value_list[t + 1];
        }
        let v_t = value_list[t];
        let delta = r + gamma * v_next - v_t;
        last_advantage = delta + gamma * lam * last_advantage;
        advantages = arr_set(advantages, t, last_advantage);
        returns = arr_set(returns, t, last_advantage + v_t);
        t = t - 1;
    }
    [advantages, returns]
}

// ---- Apply one A2C + GAE update to the weights via GradGraph ----
// Returns updated weights + loss breakdown [total, policy, value, entropy].
fn a2c_update(weights: Any, rollout: Any, lr: f64, c_value: f64, c_entropy: f64, max_grad_norm: f64) -> Any {
    let states_list = rollout[0];
    let moves_list = rollout[1];
    let action_list = rollout[2];
    let value_list = rollout[3];
    let side_list = rollout[4];
    let terminal_reward = rollout[5];
    let n = rollout[6];

    if n == 0 {
        return [weights, [0.0, 0.0, 0.0, 0.0]];
    }

    let gae = compute_gae(value_list, side_list, terminal_reward, 0.99, 0.95);
    let advantages = gae[0];
    let returns = gae[1];

    // Build a GradGraph covering every position in the episode.
    let g = GradGraph.new();
    let W1_id = g.parameter(weights[0]);
    let b1_id = g.parameter(weights[1]);
    let W2_id = g.parameter(weights[2]);
    let b2_id = g.parameter(weights[3]);
    let Wpf_id = g.parameter(weights[5]);
    let bpf_id = g.parameter(weights[6]);
    let Wpt_id = g.parameter(weights[7]);
    let bpt_id = g.parameter(weights[8]);
    let Wv_id = g.parameter(weights[9]);
    let bv_id = g.parameter(weights[10]);

    // We'll accumulate loss terms and mean them at the end.
    // Loss per step = -log π(a|s) * A  +  c_v * (V(s) - G)^2  -  c_e * H(π)
    //
    // For numerical convenience, every step contributes a scalar node which
    // we sum. We then divide by n outside (manually scaling adv/return).
    let loss_acc = 0;
    let loss_init = false;
    let t = 0;
    while t < n {
        let state = states_list[t];
        let moves = moves_list[t];
        let action = action_list[t];
        let adv = advantages[t];
        let ret = returns[t];
        let side = side_list[t];
        let num = len(moves) / 2;

        let features = encode_state(state);
        let x_id = g.input(features);

        let z1 = g.add(g.matmul(x_id, W1_id), b1_id);
        let h1 = g.relu(z1);
        let z2 = g.add(g.matmul(h1, W2_id), b2_id);
        let h2 = g.relu(z2);

        let from_logits = g.add(g.matmul(h2, Wpf_id), bpf_id);
        let to_logits = g.add(g.matmul(h2, Wpt_id), bpt_id);
        let v_pre = g.add(g.matmul(h2, Wv_id), bv_id);
        let v_node = g.tanh(v_pre);

        // Build a dense length-`num` logit vector by selecting from flat
        // heads for each legal move. We bypass the tensor op zoo and rebuild
        // per-move logits as sums of gather-equivalent picks. Since the
        // heads are small (64 wide), we implement the sum via eager
        // per-step tensor extraction + g.input(tensor) for the slice row.
        //
        // Simpler trick: compute eager scores_1xnum tensor outside the
        // graph, materialize a one-hot vector for the taken action, and
        // reconstitute the log-prob via cross_entropy_on_the_eager_scores.
        //
        // GradGraph doesn't currently support dynamic gather by action. So:
        // we compute the chosen move's log-prob via the identity
        //     log π(a) = s_a - logsumexp(s).
        // `s_a` is computed inside the graph by summing gathered from/to
        // entries through a pair of one-hot selector matrices. We build
        // those one-hot selectors as constants (input nodes) per step.
        let from_sq = moves[action * 2];
        let to_sq = moves[action * 2 + 1];
        let from_m = feat_sq(from_sq, side);
        let to_m = feat_sq(to_sq, side);

        // One-hot row selectors [64, 1] over the 64-wide head dimension.
        // score = from_logits @ onehot_from + to_logits @ onehot_to
        let oh_from_data = [];
        let oh_to_data = [];
        let kk = 0;
        while kk < 64 {
            let vf = 0.0; let vt = 0.0;
            if kk == from_m { vf = 1.0; }
            if kk == to_m { vt = 1.0; }
            oh_from_data = array_push(oh_from_data, vf);
            oh_to_data = array_push(oh_to_data, vt);
            kk = kk + 1;
        }
        let oh_from = Tensor.from_vec(oh_from_data, [64, 1]);
        let oh_to = Tensor.from_vec(oh_to_data, [64, 1]);
        let ohf_id = g.input(oh_from);
        let oht_id = g.input(oh_to);
        let s_from = g.matmul(from_logits, ohf_id);  // [1,1]
        let s_to = g.matmul(to_logits, oht_id);      // [1,1]
        let s_a_mat = g.add(s_from, s_to);           // [1,1]
        let s_a = g.sum(s_a_mat);                    // [1]  (flattened scalar)

        // Build dense [1, num] score vector via one-hot selector matrices.
        // Per-move logit = from_logit[from_m] + to_logit[to_m]. We encode the
        // gather as matmul against [64, num] one-hot selectors.
        let sel_from_data = [];
        let sel_to_data = [];
        let rr = 0;
        while rr < 64 {
            let cc = 0;
            while cc < num {
                let cc_from_sq = moves[cc * 2];
                let cc_to_sq = moves[cc * 2 + 1];
                let cc_from_m = feat_sq(cc_from_sq, side);
                let cc_to_m = feat_sq(cc_to_sq, side);
                let vf = 0.0; let vt = 0.0;
                if rr == cc_from_m { vf = 1.0; }
                if rr == cc_to_m { vt = 1.0; }
                sel_from_data = array_push(sel_from_data, vf);
                sel_to_data = array_push(sel_to_data, vt);
                cc = cc + 1;
            }
            rr = rr + 1;
        }
        let sel_from = Tensor.from_vec(sel_from_data, [64, num]);
        let sel_to = Tensor.from_vec(sel_to_data, [64, num]);
        let selF_id = g.input(sel_from);
        let selT_id = g.input(sel_to);
        let scores_from = g.matmul(from_logits, selF_id);  // [1, num]
        let scores_to = g.matmul(to_logits, selT_id);      // [1, num]
        let scores_vec = g.add(scores_from, scores_to);    // [1, num]

        // log π(a) = s_a - logsumexp(scores)
        //          = s_a - log( sum_i exp(scores_i) )
        // GradGraph has no softmax op, but exp/sum/ln are differentiable,
        // and this identity avoids the broadcast division a softmax would
        // need. It is also the numerically standard form once we offset
        // scores by max(scores) — we skip the offset here because the
        // dense scores have small magnitude in practice (init scale 0.1)
        // and gradient clipping upstream bounds any blow-up.
        let exp_scores = g.exp(scores_vec);         // [1, num]
        let sum_exp = g.sum(exp_scores);            // [1]
        let lse = g.ln(sum_exp);                    // [1]
        let log_p_a = g.sub(s_a, lse);              // [1]

        // Value scalar (flatten [1,1] → [1]).
        let v_scalar = g.sum(v_node);               // [1]

        // Policy loss: -log π(a) * adv
        let neg_adv = Tensor.from_vec([0.0 - adv], [1]);
        let neg_adv_id = g.input(neg_adv);
        let policy_term = g.mul(log_p_a, neg_adv_id);  // [1]

        // Value loss: c_v * (V(s) - G)^2
        let neg_ret = Tensor.from_vec([0.0 - ret], [1]);
        let neg_ret_id = g.input(neg_ret);
        let v_diff = g.add(v_scalar, neg_ret_id);    // V - G, [1]
        let v_sq = g.mul(v_diff, v_diff);            // [1]
        let cv = Tensor.from_vec([c_value], [1]);
        let cv_id = g.input(cv);
        let value_term = g.mul(cv_id, v_sq);         // [1]

        // Entropy regularisation is NOT routed through the graph in v2.
        // GradGraph lacks softmax, and rolling our own `H = -Σ p log p`
        // would require broadcast divisions that the current backward
        // rules don't cover. We log entropy eagerly as a diagnostic but
        // the graph loss is just policy + value. `c_entropy` is kept in
        // the signature for API stability but is currently unused.

        // Total per-step loss.
        let step_loss = g.add(policy_term, value_term);  // [1]

        if loss_init {
            loss_acc = g.add(loss_acc, step_loss);
        } else {
            loss_acc = step_loss;
            loss_init = true;
        }

        t = t + 1;
    }

    // Scale by 1/n to average.
    let inv_n = Tensor.from_vec([1.0 / float(n)], [1]);
    let inv_n_id = g.input(inv_n);
    let total_loss = g.mul(loss_acc, inv_n_id);

    g.backward(total_loss);
    let total_val = g.value(total_loss);

    // Global-norm gradient clipping before the SGD step.
    let gW1 = g.grad(W1_id);
    let gb1 = g.grad(b1_id);
    let gW2 = g.grad(W2_id);
    let gb2 = g.grad(b2_id);
    let gWpf = g.grad(Wpf_id);
    let gbpf = g.grad(bpf_id);
    let gWpt = g.grad(Wpt_id);
    let gbpt = g.grad(bpt_id);
    let gWv = g.grad(Wv_id);
    let gbv = g.grad(bv_id);

    // Compute global L2 norm. We rely on `sum` reductions over element-wise
    // squares produced eagerly — no need to route this through the graph.
    let n2 = tensor_sumsq(gW1) + tensor_sumsq(gb1) + tensor_sumsq(gW2)
           + tensor_sumsq(gb2) + tensor_sumsq(gWpf) + tensor_sumsq(gbpf)
           + tensor_sumsq(gWpt) + tensor_sumsq(gbpt) + tensor_sumsq(gWv)
           + tensor_sumsq(gbv);
    let gnorm = sqrt(n2);
    let scale = 1.0;
    if gnorm > max_grad_norm {
        scale = max_grad_norm / gnorm;
    }

    let eff_lr = lr * scale;
    let new_W1 = weights[0] - gW1 * eff_lr;
    let new_b1 = weights[1] - gb1 * eff_lr;
    let new_W2 = weights[2] - gW2 * eff_lr;
    let new_b2 = weights[3] - gb2 * eff_lr;
    let new_Wpf = weights[5] - gWpf * eff_lr;
    let new_bpf = weights[6] - gbpf * eff_lr;
    let new_Wpt = weights[7] - gWpt * eff_lr;
    let new_bpt = weights[8] - gbpt * eff_lr;
    let new_Wv = weights[9] - gWv * eff_lr;
    let new_bv = weights[10] - gbv * eff_lr;

    let new_weights = [new_W1, new_b1, new_W2, new_b2, 0, new_Wpf, new_bpf, new_Wpt, new_bpt, new_Wv, new_bv];
    let loss_info = [total_val, 0.0, 0.0, 0.0];
    let out = [new_weights, loss_info];
    return out;
}

// ---- Helper: sum of squared elements of a 2D tensor ----
fn tensor_sumsq(t: Tensor) -> f64 {
    let sh = t.shape();
    let rows = sh[0];
    let cols = sh[1];
    let acc = 0.0;
    let i = 0;
    while i < rows {
        let j = 0;
        while j < cols {
            let v = t.get([i, j]);
            acc = acc + v * v;
            j = j + 1;
        }
        i = i + 1;
    }
    acc
}

// ==========================================================================
// ============== ADAM OPTIMIZER (Phase B1)
// ==========================================================================
// Standard Adam: m=β1·m+(1-β1)·g, v=β2·v+(1-β2)·g², bias-correct, then
//   w ← w − lr · m̂ / (√v̂ + ε)
// State is held in two parallel 11-slot tensor lists mirroring the weights
// layout (slot 4 reserved as 0, like `init_weights`), plus a step counter.
// We keep `a2c_update`'s SGD path intact and ship Adam as a parallel
// `a2c_update_adam` so the byte-identical parity tests on the SGD code path
// are not perturbed.

// ---- Initialize first/second-moment buffers shaped like the weights. ----
// Returns [m_list, v_list, step] where m_list and v_list each have the same
// 11-slot layout as init_weights() (slot 4 reserved as 0).
fn init_adam_state() -> Any {
    let m_W1 = Tensor.zeros([774, 48]);
    let m_b1 = Tensor.zeros([1, 48]);
    let m_W2 = Tensor.zeros([48, 48]);
    let m_b2 = Tensor.zeros([1, 48]);
    let m_Wpf = Tensor.zeros([48, 64]);
    let m_bpf = Tensor.zeros([1, 64]);
    let m_Wpt = Tensor.zeros([48, 64]);
    let m_bpt = Tensor.zeros([1, 64]);
    let m_Wv = Tensor.zeros([48, 1]);
    let m_bv = Tensor.zeros([1, 1]);
    let m_list = [m_W1, m_b1, m_W2, m_b2, 0, m_Wpf, m_bpf, m_Wpt, m_bpt, m_Wv, m_bv];

    let v_W1 = Tensor.zeros([774, 48]);
    let v_b1 = Tensor.zeros([1, 48]);
    let v_W2 = Tensor.zeros([48, 48]);
    let v_b2 = Tensor.zeros([1, 48]);
    let v_Wpf = Tensor.zeros([48, 64]);
    let v_bpf = Tensor.zeros([1, 64]);
    let v_Wpt = Tensor.zeros([48, 64]);
    let v_bpt = Tensor.zeros([1, 64]);
    let v_Wv = Tensor.zeros([48, 1]);
    let v_bv = Tensor.zeros([1, 1]);
    let v_list = [v_W1, v_b1, v_W2, v_b2, 0, v_Wpf, v_bpf, v_Wpt, v_bpt, v_Wv, v_bv];

    [m_list, v_list, 0]
}

// ---- Apply one Adam step to a single 2D tensor. ----
// Returns [new_w, new_m, new_v]. Thin CJC-Lang wrapper around the native
// `adam_step` builtin in cjc-runtime, which operates on the flat buffer in
// Rust (shared by both executors → byte-identical by construction). Step `t`
// MUST be ≥ 1 so the bias correction denominators are non-zero.
fn adam_step_2d(w: Tensor, g: Tensor, m: Tensor, v: Tensor,
                lr: f64, b1: f64, b2: f64, eps: f64, t: i64) -> Any {
    adam_step(w, g, m, v, lr, b1, b2, eps, t)
}

// ---- High-level: run one training episode and return new weights + stats ----
// Returns [weights, total_loss, num_moves, terminal_reward].
fn train_one_episode(weights: Any, max_moves: i64, lr: f64) -> Any {
    let rollout = rollout_episode(weights, max_moves);
    let n_moves = rollout[6];
    let terminal_reward = rollout[5];
    if n_moves == 0 {
        return [weights, 0.0, 0, terminal_reward];
    }
    let result = a2c_update(weights, rollout, lr, 0.5, 0.01, 1.0);
    let new_weights = result[0];
    let losses = result[1];
    [new_weights, losses[0], n_moves, terminal_reward]
}

// ==========================================================================
// ============== ADVANTAGE + RETURN NORMALIZATION (Phase B2)
// ==========================================================================
// Standard A2C stabilization: after computing GAE advantages and returns,
// whiten each list to zero mean / unit std (with ε floor). This keeps the
// effective per-step loss scale roughly constant regardless of how many
// plies an episode lasted or how skewed the terminal reward distribution is.
//
// Only the Adam path uses these. The SGD `a2c_update` call site is
// untouched so the chess_rl_v2 parity gate on SGD remains byte-identical.

// ---- Whiten an array in place-ish: returns a new array with mean 0, std 1. ----
// Input: Any (flat array of f64). Output: Any (same length, normalized).
fn whiten_array(xs: Any) -> Any {
    let n = len(xs);
    if n == 0 { return xs; }
    // Mean.
    let sum = 0.0;
    let i = 0;
    while i < n {
        sum = sum + xs[i];
        i = i + 1;
    }
    let mean = sum / float(n);
    // Variance.
    let var_acc = 0.0;
    i = 0;
    while i < n {
        let d = xs[i] - mean;
        var_acc = var_acc + d * d;
        i = i + 1;
    }
    let var = var_acc / float(n);
    let std = sqrt(var) + 1.0e-8;
    let out = [];
    i = 0;
    while i < n {
        out = array_push(out, (xs[i] - mean) / std);
        i = i + 1;
    }
    out
}

// ==========================================================================
// ============== A2C+GAE WITH ADAM OPTIMIZER (Phase B1)
// ==========================================================================
// Mirrors `a2c_update` but applies an Adam step instead of SGD. The graph-
// building portion is duplicated verbatim from `a2c_update` so the original
// SGD path (parity-tested at tests/chess_rl_v2/test_parity.rs) is untouched.

fn a2c_update_adam(weights: Any, adam: Any, rollout: Any, lr: f64,
                   c_value: f64, c_entropy: f64, max_grad_norm: f64) -> Any {
    let states_list = rollout[0];
    let moves_list = rollout[1];
    let action_list = rollout[2];
    let value_list = rollout[3];
    let side_list = rollout[4];
    let terminal_reward = rollout[5];
    let n = rollout[6];

    if n == 0 {
        return [weights, adam, [0.0, 0.0, 0.0, 0.0]];
    }

    let gae = compute_gae(value_list, side_list, terminal_reward, 0.99, 0.95);
    let advantages = gae[0];
    let returns = gae[1];

    // Phase B2: normalize advantages and returns before feeding the graph.
    // This is only in the Adam path so the SGD parity gate stays intact.
    advantages = whiten_array(advantages);
    returns = whiten_array(returns);

    let g = GradGraph.new();
    let W1_id = g.parameter(weights[0]);
    let b1_id = g.parameter(weights[1]);
    let W2_id = g.parameter(weights[2]);
    let b2_id = g.parameter(weights[3]);
    let Wpf_id = g.parameter(weights[5]);
    let bpf_id = g.parameter(weights[6]);
    let Wpt_id = g.parameter(weights[7]);
    let bpt_id = g.parameter(weights[8]);
    let Wv_id = g.parameter(weights[9]);
    let bv_id = g.parameter(weights[10]);

    let loss_acc = 0;
    let loss_init = false;
    let t = 0;
    while t < n {
        let state = states_list[t];
        let moves = moves_list[t];
        let action = action_list[t];
        let adv = advantages[t];
        let ret = returns[t];
        let side = side_list[t];
        let num = len(moves) / 2;

        let features = encode_state(state);
        let x_id = g.input(features);

        let z1 = g.add(g.matmul(x_id, W1_id), b1_id);
        let h1 = g.relu(z1);
        let z2 = g.add(g.matmul(h1, W2_id), b2_id);
        let h2 = g.relu(z2);

        let from_logits = g.add(g.matmul(h2, Wpf_id), bpf_id);
        let to_logits = g.add(g.matmul(h2, Wpt_id), bpt_id);
        let v_pre = g.add(g.matmul(h2, Wv_id), bv_id);
        let v_node = g.tanh(v_pre);

        let from_sq = moves[action * 2];
        let to_sq = moves[action * 2 + 1];
        let from_m = feat_sq(from_sq, side);
        let to_m = feat_sq(to_sq, side);

        let oh_from_data = [];
        let oh_to_data = [];
        let kk = 0;
        while kk < 64 {
            let vf = 0.0; let vt = 0.0;
            if kk == from_m { vf = 1.0; }
            if kk == to_m { vt = 1.0; }
            oh_from_data = array_push(oh_from_data, vf);
            oh_to_data = array_push(oh_to_data, vt);
            kk = kk + 1;
        }
        let oh_from = Tensor.from_vec(oh_from_data, [64, 1]);
        let oh_to = Tensor.from_vec(oh_to_data, [64, 1]);
        let ohf_id = g.input(oh_from);
        let oht_id = g.input(oh_to);
        let s_from = g.matmul(from_logits, ohf_id);
        let s_to = g.matmul(to_logits, oht_id);
        let s_a_mat = g.add(s_from, s_to);
        let s_a = g.sum(s_a_mat);

        let sel_from_data = [];
        let sel_to_data = [];
        let rr = 0;
        while rr < 64 {
            let cc = 0;
            while cc < num {
                let cc_from_sq = moves[cc * 2];
                let cc_to_sq = moves[cc * 2 + 1];
                let cc_from_m = feat_sq(cc_from_sq, side);
                let cc_to_m = feat_sq(cc_to_sq, side);
                let vf = 0.0; let vt = 0.0;
                if rr == cc_from_m { vf = 1.0; }
                if rr == cc_to_m { vt = 1.0; }
                sel_from_data = array_push(sel_from_data, vf);
                sel_to_data = array_push(sel_to_data, vt);
                cc = cc + 1;
            }
            rr = rr + 1;
        }
        let sel_from = Tensor.from_vec(sel_from_data, [64, num]);
        let sel_to = Tensor.from_vec(sel_to_data, [64, num]);
        let selF_id = g.input(sel_from);
        let selT_id = g.input(sel_to);
        let scores_from = g.matmul(from_logits, selF_id);
        let scores_to = g.matmul(to_logits, selT_id);
        let scores_vec = g.add(scores_from, scores_to);

        let exp_scores = g.exp(scores_vec);
        let sum_exp = g.sum(exp_scores);
        let lse = g.ln(sum_exp);
        let log_p_a = g.sub(s_a, lse);

        let v_scalar = g.sum(v_node);

        let neg_adv = Tensor.from_vec([0.0 - adv], [1]);
        let neg_adv_id = g.input(neg_adv);
        let policy_term = g.mul(log_p_a, neg_adv_id);

        let neg_ret = Tensor.from_vec([0.0 - ret], [1]);
        let neg_ret_id = g.input(neg_ret);
        let v_diff = g.add(v_scalar, neg_ret_id);
        let v_sq = g.mul(v_diff, v_diff);
        let cv = Tensor.from_vec([c_value], [1]);
        let cv_id = g.input(cv);
        let value_term = g.mul(cv_id, v_sq);

        let step_loss = g.add(policy_term, value_term);

        if loss_init {
            loss_acc = g.add(loss_acc, step_loss);
        } else {
            loss_acc = step_loss;
            loss_init = true;
        }

        t = t + 1;
    }

    let inv_n = Tensor.from_vec([1.0 / float(n)], [1]);
    let inv_n_id = g.input(inv_n);
    let total_loss = g.mul(loss_acc, inv_n_id);

    g.backward(total_loss);
    let total_val = g.value(total_loss);

    let gW1 = g.grad(W1_id);
    let gb1 = g.grad(b1_id);
    let gW2 = g.grad(W2_id);
    let gb2 = g.grad(b2_id);
    let gWpf = g.grad(Wpf_id);
    let gbpf = g.grad(bpf_id);
    let gWpt = g.grad(Wpt_id);
    let gbpt = g.grad(bpt_id);
    let gWv = g.grad(Wv_id);
    let gbv = g.grad(bv_id);

    // Global L2 clip — same shape as the SGD path so the two optimizers see
    // identically scaled gradients before the per-parameter Adam step.
    let n2 = tensor_sumsq(gW1) + tensor_sumsq(gb1) + tensor_sumsq(gW2)
           + tensor_sumsq(gb2) + tensor_sumsq(gWpf) + tensor_sumsq(gbpf)
           + tensor_sumsq(gWpt) + tensor_sumsq(gbpt) + tensor_sumsq(gWv)
           + tensor_sumsq(gbv);
    let gnorm = sqrt(n2);
    let scale = 1.0;
    if gnorm > max_grad_norm {
        scale = max_grad_norm / gnorm;
    }

    // Adam state unpack and step.
    let m_list = adam[0];
    let v_list = adam[1];
    let step = adam[2] + 1;
    let b1c = 0.9;
    let b2c = 0.999;
    let eps = 1.0e-8;

    let s0 = adam_step_2d(weights[0], gW1 * scale, m_list[0], v_list[0], lr, b1c, b2c, eps, step);
    let new_W1 = s0[0]; let new_mW1 = s0[1]; let new_vW1 = s0[2];
    let s1 = adam_step_2d(weights[1], gb1 * scale, m_list[1], v_list[1], lr, b1c, b2c, eps, step);
    let new_b1 = s1[0]; let new_mb1 = s1[1]; let new_vb1 = s1[2];
    let s2 = adam_step_2d(weights[2], gW2 * scale, m_list[2], v_list[2], lr, b1c, b2c, eps, step);
    let new_W2 = s2[0]; let new_mW2 = s2[1]; let new_vW2 = s2[2];
    let s3 = adam_step_2d(weights[3], gb2 * scale, m_list[3], v_list[3], lr, b1c, b2c, eps, step);
    let new_b2 = s3[0]; let new_mb2 = s3[1]; let new_vb2 = s3[2];
    let s5 = adam_step_2d(weights[5], gWpf * scale, m_list[5], v_list[5], lr, b1c, b2c, eps, step);
    let new_Wpf = s5[0]; let new_mWpf = s5[1]; let new_vWpf = s5[2];
    let s6 = adam_step_2d(weights[6], gbpf * scale, m_list[6], v_list[6], lr, b1c, b2c, eps, step);
    let new_bpf = s6[0]; let new_mbpf = s6[1]; let new_vbpf = s6[2];
    let s7 = adam_step_2d(weights[7], gWpt * scale, m_list[7], v_list[7], lr, b1c, b2c, eps, step);
    let new_Wpt = s7[0]; let new_mWpt = s7[1]; let new_vWpt = s7[2];
    let s8 = adam_step_2d(weights[8], gbpt * scale, m_list[8], v_list[8], lr, b1c, b2c, eps, step);
    let new_bpt = s8[0]; let new_mbpt = s8[1]; let new_vbpt = s8[2];
    let s9 = adam_step_2d(weights[9], gWv * scale, m_list[9], v_list[9], lr, b1c, b2c, eps, step);
    let new_Wv = s9[0]; let new_mWv = s9[1]; let new_vWv = s9[2];
    let s10 = adam_step_2d(weights[10], gbv * scale, m_list[10], v_list[10], lr, b1c, b2c, eps, step);
    let new_bv = s10[0]; let new_mbv = s10[1]; let new_vbv = s10[2];

    let new_weights = [new_W1, new_b1, new_W2, new_b2, 0, new_Wpf, new_bpf, new_Wpt, new_bpt, new_Wv, new_bv];
    let new_m = [new_mW1, new_mb1, new_mW2, new_mb2, 0, new_mWpf, new_mbpf, new_mWpt, new_mbpt, new_mWv, new_mbv];
    let new_v = [new_vW1, new_vb1, new_vW2, new_vb2, 0, new_vWpf, new_vbpf, new_vWpt, new_vbpt, new_vWv, new_vbv];
    let new_adam = [new_m, new_v, step];
    let loss_info = [total_val, 0.0, 0.0, 0.0];
    return [new_weights, new_adam, loss_info];
}

// ---- High-level: one episode of Adam training. ----
// Returns [new_weights, new_adam_state, total_loss, num_moves, terminal_reward].
fn train_one_episode_adam(weights: Any, adam: Any, max_moves: i64, lr: f64) -> Any {
    let rollout = rollout_episode(weights, max_moves);
    let n_moves = rollout[6];
    let terminal_reward = rollout[5];
    if n_moves == 0 {
        return [weights, adam, 0.0, 0, terminal_reward];
    }
    let result = a2c_update_adam(weights, adam, rollout, lr, 0.5, 0.01, 1.0);
    let new_weights = result[0];
    let new_adam = result[1];
    let losses = result[2];
    [new_weights, new_adam, losses[0], n_moves, terminal_reward]
}

// ---- Temperature-annealed Adam training (Phase B3) ----
// Uses a temperature-aware rollout. Otherwise identical to train_one_episode_adam.
fn train_one_episode_adam_temp(weights: Any, adam: Any, max_moves: i64, lr: f64, temp: f64) -> Any {
    let rollout = rollout_episode_temp(weights, max_moves, temp);
    let n_moves = rollout[6];
    let terminal_reward = rollout[5];
    if n_moves == 0 {
        return [weights, adam, 0.0, 0, terminal_reward];
    }
    let result = a2c_update_adam(weights, adam, rollout, lr, 0.5, 0.01, 1.0);
    let new_weights = result[0];
    let new_adam = result[1];
    let losses = result[2];
    [new_weights, new_adam, losses[0], n_moves, terminal_reward]
}

// ---- Full Adam training episode: temperature + resignation (Phase B3+B4) ----
// The intended production training call used by Phase D. Combines all
// stabilizers. Returns [new_weights, new_adam, loss, n_moves, terminal_reward].
fn train_one_episode_adam_full(weights: Any, adam: Any, max_moves: i64, lr: f64,
                               temp: f64, resign_thresh: f64, resign_patience: i64) -> Any {
    let rollout = rollout_episode_full(weights, max_moves, temp, resign_thresh, resign_patience);
    let n_moves = rollout[6];
    let terminal_reward = rollout[5];
    if n_moves == 0 {
        return [weights, adam, 0.0, 0, terminal_reward];
    }
    let result = a2c_update_adam(weights, adam, rollout, lr, 0.5, 0.01, 1.0);
    let new_weights = result[0];
    let new_adam = result[1];
    let losses = result[2];
    [new_weights, new_adam, losses[0], n_moves, terminal_reward]
}

// ==========================================================================
// ============== CSV TRAINING LOG (Phase C2)
// ==========================================================================
// Streams one CSV row per episode so training runs can be analyzed offline
// (loss curves, move-count distributions, wall-clock per episode). Uses the
// `file_append` builtin so a crashed run still leaves a partial log.

// ---- Write the CSV header. Call once at training start. ----
fn csv_open_log(path: Any) -> i64 {
    let header = "episode,loss,n_moves,terminal_reward,temp,adam_step\n";
    file_write(path, header);
    0
}

// ---- Append one row. Returns 0. ----
fn csv_log_episode(path: Any, episode: i64, loss: f64, n_moves: i64,
                   terminal_reward: f64, temp: f64, adam_step: i64) -> i64 {
    let row = to_string(episode) + "," + to_string(loss) + "," + to_string(n_moves) + ","
            + to_string(terminal_reward) + "," + to_string(temp) + "," + to_string(adam_step) + "\n";
    file_append(path, row);
    0
}

// ==========================================================================
// ============== CHECKPOINT BUNDLE (Phase C1)
// ==========================================================================
// Serialize a complete training run state (weights + Adam moments + episode
// counter) into a single file using the tensor_list_save builtin from Phase A.
//
// Wire format: 32 tensors + 1 meta tensor
//   [W1, b1, W2, b2, Wpf, bpf, Wpt, bpt, Wv, bv]        (slots 0..9, skipping
//   [mW1, mb1, mW2, mb2, mWpf, mbpf, mWpt, mbpt, mWv, mbv]  the reserved slot 4)
//   [vW1, vb1, vW2, vb2, vWpf, vbpf, vWpt, vbpt, vWv, vbv]
//   [meta_tensor_shape_[1,3] = [episode, adam_step, format_version]]
// Content hash is verified on load via the tensor_snap footer.

// ---- Pack the 11-slot weights list to a 10-tensor list (drop slot 4). ----
fn weights_to_10(w: Any) -> Any {
    [w[0], w[1], w[2], w[3], w[5], w[6], w[7], w[8], w[9], w[10]]
}

// ---- Unpack a 10-tensor list back to the 11-slot layout (insert 0 at 4). ----
fn weights_from_10(t: Any) -> Any {
    [t[0], t[1], t[2], t[3], 0, t[4], t[5], t[6], t[7], t[8], t[9]]
}

// ---- Save a full training checkpoint. `episode` is the next episode index. ----
fn save_checkpoint(path: Any, weights: Any, adam: Any, episode: i64) -> i64 {
    let m_list = adam[0];
    let v_list = adam[1];
    let step = adam[2];

    let ws = weights_to_10(weights);
    let ms = weights_to_10(m_list);
    let vs = weights_to_10(v_list);

    let bundle = [];
    let i = 0;
    while i < 10 { bundle = array_push(bundle, ws[i]); i = i + 1; }
    i = 0;
    while i < 10 { bundle = array_push(bundle, ms[i]); i = i + 1; }
    i = 0;
    while i < 10 { bundle = array_push(bundle, vs[i]); i = i + 1; }

    // Meta tensor: [episode, adam_step, format_version=1]
    let meta = Tensor.from_vec([float(episode), float(step), 1.0], [1, 3]);
    bundle = array_push(bundle, meta);

    tensor_list_save(path, bundle);
    0
}

// ---- Load a training checkpoint. Returns [weights, adam, episode]. ----
fn load_checkpoint(path: Any) -> Any {
    let bundle = tensor_list_load(path);
    let n = len(bundle);
    if n != 31 {
        // Will surface as a wrong-shape downstream error; we don't raise here
        // because CJC-Lang test asserts sit on specific fields.
    }
    let ws_10 = [bundle[0], bundle[1], bundle[2], bundle[3], bundle[4],
                 bundle[5], bundle[6], bundle[7], bundle[8], bundle[9]];
    let ms_10 = [bundle[10], bundle[11], bundle[12], bundle[13], bundle[14],
                 bundle[15], bundle[16], bundle[17], bundle[18], bundle[19]];
    let vs_10 = [bundle[20], bundle[21], bundle[22], bundle[23], bundle[24],
                 bundle[25], bundle[26], bundle[27], bundle[28], bundle[29]];
    let meta = bundle[30];
    let episode = int(meta.get([0, 0]));
    let step = int(meta.get([0, 1]));

    let weights = weights_from_10(ws_10);
    let m_list = weights_from_10(ms_10);
    let v_list = weights_from_10(vs_10);
    let adam = [m_list, v_list, step];
    [weights, adam, episode]
}

// ==========================================================================
// ============== EVAL
// ==========================================================================
// Random and snapshot arenas. Returns per-game outcome in {+1, 0, -1} from
// the agent's perspective.

// ---- Play a single game: `agent_side` uses `weights` greedily,
//      the other side plays uniformly random over legal moves. ----
fn play_vs_random(weights: Any, agent_side: i64, max_moves: i64) -> f64 {
    let state = init_state();
    let step = 0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            // Side to move is mated.
            let loser = state_side(state);
            if loser == agent_side { return 0.0 - 1.0; }
            return 1.0;
        }
        if status != 0 { return 0.0; }
        let moves = legal_moves(state);
        let num = len(moves) / 2;
        let a = 0;
        if state_side(state) == agent_side {
            let sel = select_action_greedy(weights, state, moves);
            a = sel[0];
        } else {
            // Uniform pick via categorical_sample over a flat [num] vector.
            let pdata = [];
            let ii = 0;
            while ii < num {
                pdata = array_push(pdata, 1.0 / float(num));
                ii = ii + 1;
            }
            let probs = Tensor.from_vec(pdata, [num]);
            a = categorical_sample(probs);
        }
        let from_sq = moves[a * 2];
        let to_sq = moves[a * 2 + 1];
        state = apply_move(state, from_sq, to_sq);
        step = step + 1;
    }
    0.0
}

// ---- Play N games alternating colors, return [wins, draws, losses] ----
fn eval_vs_random(weights: Any, n_games: i64, max_moves: i64) -> Any {
    let wins = 0;
    let draws = 0;
    let losses = 0;
    let i = 0;
    while i < n_games {
        let side = 1;
        if i % 2 == 1 { side = 0 - 1; }
        let out = play_vs_random(weights, side, max_moves);
        if out > 0.5 { wins = wins + 1; }
        if out < 0.0 - 0.5 { losses = losses + 1; }
        if out > 0.0 - 0.5 && out < 0.5 { draws = draws + 1; }
        i = i + 1;
    }
    [wins, draws, losses]
}

// ==========================================================================
// ============== MATERIAL-GREEDY BASELINE OPPONENT (Phase B5)
// ==========================================================================
// A non-trainable scripted opponent that always plays the legal move with the
// highest immediate material delta. Used as a second acceptance gate after
// `eval_vs_random` — the trained agent should beat both random *and* greedy.
//
// Material values (absolute): pawn=1, knight=3, bishop=3, rook=5, queen=9,
// king=0 (captures ignored; illegal to capture the king anyway). Ties broken
// by move index so the opponent is deterministic for a given position.

fn piece_material_value(p: i64) -> i64 {
    let ap = if p < 0 { 0 - p } else { p };
    if ap == 1 { return 1; }
    if ap == 2 { return 3; }
    if ap == 3 { return 3; }
    if ap == 4 { return 5; }
    if ap == 5 { return 9; }
    0
}

// ---- Pick the move that captures the highest-value piece (ties: first). ----
// Returns a move index (0..num_moves-1).
fn select_action_material_greedy(state: Any, moves: Any) -> i64 {
    let board = state_board(state);
    let num = len(moves) / 2;
    let best = 0;
    let best_val = 0 - 1;
    let i = 0;
    while i < num {
        let to_sq = moves[i * 2 + 1];
        let captured = board[to_sq];
        let val = piece_material_value(captured);
        if val > best_val {
            best_val = val;
            best = i;
        }
        i = i + 1;
    }
    best
}

// ---- Play one game: `weights` plays as agent_side, baseline plays the other ----
fn play_vs_greedy(weights: Any, agent_side: i64, max_moves: i64) -> f64 {
    let state = init_state();
    let step = 0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            let loser = state_side(state);
            if loser == agent_side { return 0.0 - 1.0; }
            return 1.0;
        }
        if status != 0 { return 0.0; }
        let moves = legal_moves(state);
        let a = 0;
        if state_side(state) == agent_side {
            let sel = select_action_greedy(weights, state, moves);
            a = sel[0];
        } else {
            a = select_action_material_greedy(state, moves);
        }
        let from_sq = moves[a * 2];
        let to_sq = moves[a * 2 + 1];
        state = apply_move(state, from_sq, to_sq);
        step = step + 1;
    }
    0.0
}

// ---- Play N games alternating colors, return [wins, draws, losses] ----
fn eval_vs_greedy(weights: Any, n_games: i64, max_moves: i64) -> Any {
    let wins = 0;
    let draws = 0;
    let losses = 0;
    let i = 0;
    while i < n_games {
        let side = 1;
        if i % 2 == 1 { side = 0 - 1; }
        let out = play_vs_greedy(weights, side, max_moves);
        if out > 0.5 { wins = wins + 1; }
        if out < 0.0 - 0.5 { losses = losses + 1; }
        if out > 0.0 - 0.5 && out < 0.5 { draws = draws + 1; }
        i = i + 1;
    }
    [wins, draws, losses]
}

// ---- Snapshot arena: W_curr plays against W_prev, both greedy. ----
fn play_snapshot(w_curr: Any, w_prev: Any, curr_side: i64, max_moves: i64) -> f64 {
    let state = init_state();
    let step = 0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            let loser = state_side(state);
            if loser == curr_side { return 0.0 - 1.0; }
            return 1.0;
        }
        if status != 0 { return 0.0; }
        let moves = legal_moves(state);
        let player = w_prev;
        if state_side(state) == curr_side { player = w_curr; }
        let sel = select_action_greedy(player, state, moves);
        let a = sel[0];
        let from_sq = moves[a * 2];
        let to_sq = moves[a * 2 + 1];
        state = apply_move(state, from_sq, to_sq);
        step = step + 1;
    }
    0.0
}

fn eval_vs_snapshot(w_curr: Any, w_prev: Any, n_games: i64, max_moves: i64) -> Any {
    let wins = 0;
    let draws = 0;
    let losses = 0;
    let i = 0;
    while i < n_games {
        let side = 1;
        if i % 2 == 1 { side = 0 - 1; }
        let out = play_snapshot(w_curr, w_prev, side, max_moves);
        if out > 0.5 { wins = wins + 1; }
        if out < 0.0 - 0.5 { losses = losses + 1; }
        if out > 0.0 - 0.5 && out < 0.5 { draws = draws + 1; }
        i = i + 1;
    }
    [wins, draws, losses]
}

// ---- Phase C3: Elo-lite ratings and snapshot gauntlet. ----
//
// Standard Elo formula:
//   expected = 1 / (1 + 10^((r_opp - r_self) / 400))
//   r_new    = r_self + k * (actual - expected)
// where actual ∈ {0, 0.5, 1}, k is the update constant (we use 32).

fn elo_expected(r_self: f64, r_opp: f64) -> f64 {
    let diff = (r_opp - r_self) / 400.0;
    let denom = 1.0 + pow(10.0, diff);
    1.0 / denom
}

fn elo_update(r_self: f64, r_opp: f64, actual: f64, k: f64) -> f64 {
    let expected = elo_expected(r_self, r_opp);
    r_self + k * (actual - expected)
}

// Apply a batch of {wins, draws, losses} outcomes from a single opponent
// to a running rating. Each game updates the rating independently, in the
// canonical Elo iterative fashion: win=1.0, draw=0.5, loss=0.0.
fn elo_apply_record(r_self: f64, r_opp: f64, wdl: Any, k: f64) -> f64 {
    let wins = wdl[0];
    let draws = wdl[1];
    let losses = wdl[2];
    let r = r_self;
    let i = 0;
    while i < wins {
        r = elo_update(r, r_opp, 1.0, k);
        i = i + 1;
    }
    i = 0;
    while i < draws {
        r = elo_update(r, r_opp, 0.5, k);
        i = i + 1;
    }
    i = 0;
    while i < losses {
        r = elo_update(r, r_opp, 0.0, k);
        i = i + 1;
    }
    r
}

// Snapshot gauntlet: play `n_games_each` games against each snapshot in
// `snapshots` (a list of weight-lists), starting from rating `r_self` and
// updating it deterministically after each opponent's record is applied.
// `snapshot_ratings` must be a parallel list of f64 ratings for each
// snapshot opponent. Returns [final_rating, total_wins, total_draws, total_losses].
fn gauntlet_vs_snapshots(
    w_curr: Any,
    snapshots: Any,
    snapshot_ratings: Any,
    r_self: f64,
    n_games_each: i64,
    max_moves: i64,
    k: f64,
) -> Any {
    let n = len(snapshots);
    let r = r_self;
    let total_w = 0;
    let total_d = 0;
    let total_l = 0;
    let i = 0;
    while i < n {
        let snap = snapshots[i];
        let r_opp = snapshot_ratings[i];
        let wdl = eval_vs_snapshot(w_curr, snap, n_games_each, max_moves);
        r = elo_apply_record(r, r_opp, wdl, k);
        total_w = total_w + wdl[0];
        total_d = total_d + wdl[1];
        total_l = total_l + wdl[2];
        i = i + 1;
    }
    [r, total_w, total_d, total_l]
}

// ---- Phase C4: PGN dump helpers. ----
//
// We use long algebraic notation (`e2-e4`) rather than full SAN because
// deriving SAN requires disambiguation logic (e.g. `Nbd7`) that would
// bloat the source. LAN is accepted by most PGN parsers in lax mode and
// round-trips back to `(from, to)` unambiguously.

fn file_char(f: i64) -> Any {
    if f == 0 { return "a"; }
    if f == 1 { return "b"; }
    if f == 2 { return "c"; }
    if f == 3 { return "d"; }
    if f == 4 { return "e"; }
    if f == 5 { return "f"; }
    if f == 6 { return "g"; }
    "h"
}

fn rank_char(r: i64) -> Any {
    if r == 0 { return "1"; }
    if r == 1 { return "2"; }
    if r == 2 { return "3"; }
    if r == 3 { return "4"; }
    if r == 4 { return "5"; }
    if r == 5 { return "6"; }
    if r == 6 { return "7"; }
    "8"
}

fn sq_to_uci(sq: i64) -> Any {
    let r = rank_of(sq);
    let f = file_of(sq);
    file_char(f) + rank_char(r)
}

fn move_to_lan(from_sq: i64, to_sq: i64) -> Any {
    sq_to_uci(from_sq) + "-" + sq_to_uci(to_sq)
}

// Play a game between two weight sets (both greedy), recording the
// move sequence as a flat `[from0, to0, from1, to1, ...]` array along
// with the outcome from White's perspective:
//   +1 = white wins, -1 = black wins, 0 = draw.
// Caps at `max_moves` half-moves.
fn play_recorded_game(w_white: Any, w_black: Any, max_moves: i64) -> Any {
    let state = init_state();
    let moves_played = [];
    let step = 0;
    let result = 0.0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            // The side to move is checkmated. If side == 1 (white), black won.
            let loser = state_side(state);
            result = float(0 - loser);
            step = max_moves;
        } else {
            if status != 0 {
                // stalemate or rule draw
                result = 0.0;
                step = max_moves;
            } else {
                let legal = legal_moves(state);
                let player = w_white;
                if state_side(state) == 0 - 1 { player = w_black; }
                let sel = select_action_greedy(player, state, legal);
                let a = sel[0];
                let from_sq = legal[a * 2];
                let to_sq = legal[a * 2 + 1];
                moves_played = array_push(moves_played, from_sq);
                moves_played = array_push(moves_played, to_sq);
                state = apply_move(state, from_sq, to_sq);
                step = step + 1;
            }
        }
    }
    [moves_played, result]
}

fn pgn_result_token(result: f64) -> Any {
    if result > 0.5 { return "1-0"; }
    if result < 0.0 - 0.5 { return "0-1"; }
    "1/2-1/2"
}

// Build a PGN-formatted string for a single game. `moves` is a flat
// `[from0, to0, from1, to1, ...]` array of legal moves in play order.
fn pgn_format_game(
    event: Any,
    site: Any,
    date: Any,
    round_str: Any,
    white: Any,
    black: Any,
    result: f64,
    moves: Any,
) -> Any {
    let result_tok = pgn_result_token(result);
    let header = "[Event \"" + event + "\"]\n";
    header = header + "[Site \"" + site + "\"]\n";
    header = header + "[Date \"" + date + "\"]\n";
    header = header + "[Round \"" + round_str + "\"]\n";
    header = header + "[White \"" + white + "\"]\n";
    header = header + "[Black \"" + black + "\"]\n";
    header = header + "[Result \"" + result_tok + "\"]\n\n";

    let body = "";
    let half = len(moves) / 2;
    let i = 0;
    while i < half {
        let from_sq = moves[i * 2];
        let to_sq = moves[i * 2 + 1];
        let lan = move_to_lan(from_sq, to_sq);
        if i % 2 == 0 {
            // white move: start a move number
            let move_num = (i / 2) + 1;
            body = body + to_string(move_num) + ". " + lan;
        } else {
            body = body + " " + lan + " ";
        }
        i = i + 1;
    }
    // Ensure there's a space before the result token.
    body = body + " " + result_tok + "\n\n";
    header + body
}

// Append a PGN game to a file. Uses the Phase C2 `file_append` builtin,
// so multiple games accumulate naturally.
fn pgn_dump_game(
    path: Any,
    event: Any,
    round_str: Any,
    white: Any,
    black: Any,
    result: f64,
    moves: Any,
) -> i64 {
    let text = pgn_format_game(event, "local", "2026.04.09", round_str, white, black, result, moves);
    file_append(path, text);
    len(moves) / 2
}

// ---- Phase C5: Vizor training curve plot. ----
//
// Build an SVG plot of a training loss or reward curve and save it to disk.
// Inputs are two parallel arrays of equal length (episode index and value).
// Uses the `vizor_plot` builtin from the `import vizor;` gate at the top of
// this prelude, then calls `.geom_line().title(...).xlab(...).ylab(...).save(...)`.

fn vizor_training_curve(path: Any, title: Any, y_label: Any, episodes: Any, values: Any) -> i64 {
    let p = vizor_plot(episodes, values);
    p = p.geom_line();
    p = p.title(title);
    p = p.xlab("episode");
    p = p.ylab(y_label);
    p.save(path);
    len(episodes)
}

// Render two curves on a single figure by writing both SVG paths. We keep
// this simple: one file per series. A multi-series Vizor API exists but is
// overkill for our training diagnostics.
fn vizor_training_curves(
    dir: Any,
    episodes: Any,
    loss_values: Any,
    reward_values: Any,
) -> i64 {
    let loss_path = dir + "/training_loss.svg";
    let reward_path = dir + "/training_reward.svg";
    vizor_training_curve(loss_path, "CJC-Lang Chess RL v2.1 — loss", "loss", episodes, loss_values);
    vizor_training_curve(reward_path, "CJC-Lang Chess RL v2.1 — reward", "reward", episodes, reward_values);
    2
}

// ---- Diagnostics: average policy entropy over a rollout's states. ----
fn policy_entropy_from_rollout(weights: Any, states_list: Any, moves_list: Any) -> f64 {
    let n = len(states_list);
    if n == 0 { return 0.0; }
    let total = 0.0;
    let t = 0;
    while t < n {
        let s = states_list[t];
        let m = moves_list[t];
        let sv = score_moves(weights, s, m);
        let probs = sv[0].softmax();
        let num = len(m) / 2;
        let h = 0.0;
        let i = 0;
        while i < num {
            let p = probs.get([i]);
            if p > 1.0e-12 {
                h = h - p * log(p);
            }
            i = i + 1;
        }
        total = total + h;
        t = t + 1;
    }
    total / float(n)
}

// ==========================================================================
// ============== V2.2 UPGRADES (Tier 1, cheap ML fixes)
// ==========================================================================
// Layered improvements on top of v2.1's honest baseline. Everything is
// additive — none of these helpers replace existing v2.1 entry points, so
// the 72 existing tests remain untouched.
//
//   T1-a  max_moves raised at the call site (config, not here).
//   T1-b  Move-count penalty: terminal reward shrinks with episode length.
//   T1-c  Threefold-repetition detection via a position-string histogram.
//   T1-d  Stochastic low-temperature eval policy to break shuffling.
//   T1-e  CSV log with a repetition_draw column.

// ---- Position key: deterministic string over board + side + castling + ep.
// ~200 chars per position; slower than Zobrist but needs no RNG table and
// avoids integer overflow. Fine for the ~80-ply rollouts this demo runs.
fn position_key_v22(state: Any) -> Any {
    let board = state_board(state);
    let key = "";
    let i = 0;
    while i < 64 {
        key = key + to_string(board[i]) + ",";
        i = i + 1;
    }
    key = key + "|" + to_string(state_side(state));
    let c = state_castling(state);
    key = key + "|" + to_string(c[0]) + to_string(c[1])
        + to_string(c[2]) + to_string(c[3]);
    key = key + "|" + to_string(state_ep(state));
    key
}

// ---- Increment the repetition counter. Returns [new_map, new_count].
// Map_set is immutable; callers must rebind.
fn rep_inc_v22(rep_map: Any, state: Any) -> Any {
    let key = position_key_v22(state);
    let count = 1;
    if map_contains(rep_map, key) {
        count = map_get(rep_map, key) + 1;
    }
    let new_map = map_set(rep_map, key, count);
    [new_map, count]
}

// ---- Move-count penalty: shrink reward toward zero by penalty*n_moves.
// Draws stay 0. Wins/losses shrink but don't flip sign (floor ±0.05).
fn apply_move_penalty_v22(reward: f64, n_moves: i64, penalty_per_ply: f64) -> f64 {
    if penalty_per_ply <= 0.0 { return reward; }
    if reward == 0.0 { return 0.0; }
    let magnitude = penalty_per_ply * float(n_moves);
    if reward > 0.0 {
        let out = reward - magnitude;
        if out < 0.05 { return 0.05; }
        return out;
    }
    let out2 = reward + magnitude;
    if out2 > 0.0 - 0.05 { return 0.0 - 0.05; }
    out2
}

// ---- Stochastic low-temperature EVAL action. temp=0 ⇒ fall back to greedy. ----
fn select_action_eval_temp_v22(weights: Any, state: Any, moves: Any, temp: f64) -> Any {
    if temp <= 0.0 { return select_action_greedy(weights, state, moves); }
    let sv = score_moves(weights, state, moves);
    let scores_t = sv[0];
    let v = sv[1];
    let inv_t = 1.0 / temp;
    let scaled = scores_t * inv_t;
    let probs = scaled.softmax();
    let action = categorical_sample(probs);
    [action, 0.0, v]
}

// ---- Rollout with repetition detection + move-count penalty (T1-a/b/c). ----
// Returns:
//   [states, moves, actions, values, sides, adjusted_reward, n_moves, rep_flag]
// Slots 0..6 are layout-compatible with a2c_update_adam.
fn rollout_episode_v22(weights: Any, max_moves: i64, temp: f64,
                       penalty_per_ply: f64) -> Any {
    let state = init_state();
    let states_list = [];
    let moves_list = [];
    let action_list = [];
    let value_list = [];
    let side_list = [];
    let rep_map = map_new();
    // Seed the map with the starting position.
    let seed_pair = rep_inc_v22(rep_map, state);
    rep_map = seed_pair[0];
    let step = 0;
    let terminal_reward = 0.0;
    let rep_flag = 0;
    while step < max_moves {
        let status_base = terminal_status(state);
        if status_base == 2 {
            terminal_reward = float(0 - state_side(state));
            step = max_moves + 1;
        } else {
            if status_base != 0 {
                terminal_reward = 0.0;
                step = max_moves + 1;
            } else {
                let moves = legal_moves(state);
                let sel = select_action_temp(weights, state, moves, temp);
                let a = sel[0];
                let v = sel[2];
                states_list = array_push(states_list, state);
                moves_list = array_push(moves_list, moves);
                action_list = array_push(action_list, a);
                value_list = array_push(value_list, v);
                side_list = array_push(side_list, state_side(state));
                let from_sq = moves[a * 2];
                let to_sq = moves[a * 2 + 1];
                state = apply_move(state, from_sq, to_sq);
                let pair = rep_inc_v22(rep_map, state);
                rep_map = pair[0];
                let count = pair[1];
                if count >= 3 {
                    terminal_reward = 0.0;
                    rep_flag = 1;
                    step = max_moves + 1;
                } else {
                    step = step + 1;
                }
            }
        }
    }
    let n_moves = len(states_list);
    let adjusted = apply_move_penalty_v22(terminal_reward, n_moves, penalty_per_ply);
    [states_list, moves_list, action_list, value_list, side_list,
     adjusted, n_moves, rep_flag]
}

// ---- One full Adam training episode for v2.2.
// Returns [new_weights, new_adam, loss, n_moves, terminal_reward, rep_flag].
fn train_one_episode_adam_v22(weights: Any, adam: Any, max_moves: i64,
                              lr: f64, temp: f64, penalty: f64) -> Any {
    let rollout = rollout_episode_v22(weights, max_moves, temp, penalty);
    let n_moves = rollout[6];
    let terminal_reward = rollout[5];
    let rep_flag = rollout[7];
    if n_moves == 0 {
        return [weights, adam, 0.0, 0, terminal_reward, rep_flag];
    }
    let result = a2c_update_adam(weights, adam, rollout, lr, 0.5, 0.01, 1.0);
    let new_weights = result[0];
    let new_adam = result[1];
    let losses = result[2];
    [new_weights, new_adam, losses[0], n_moves, terminal_reward, rep_flag]
}

// ---- CSV v2.2: extra `repetition_draw` column as the 7th field. ----
fn csv_open_log_v22(path: Any) -> i64 {
    let header = "episode,loss,n_moves,terminal_reward,temp,adam_step,repetition_draw\n";
    file_write(path, header);
    0
}

fn csv_log_episode_v22(path: Any, episode: i64, loss: f64, n_moves: i64,
                       terminal_reward: f64, temp: f64, adam_step: i64,
                       rep_flag: i64) -> i64 {
    let row = to_string(episode) + "," + to_string(loss) + ","
            + to_string(n_moves) + "," + to_string(terminal_reward) + ","
            + to_string(temp) + "," + to_string(adam_step) + ","
            + to_string(rep_flag) + "\n";
    file_append(path, row);
    0
}

// ==========================================================================
// ============== V2.2 EVAL (stochastic, repetition-aware)
// ==========================================================================

fn play_vs_random_v22(weights: Any, agent_side: i64, max_moves: i64,
                      eval_temp: f64) -> f64 {
    let state = init_state();
    let rep_map = map_new();
    let pair0 = rep_inc_v22(rep_map, state);
    rep_map = pair0[0];
    let step = 0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            let loser = state_side(state);
            if loser == agent_side { return 0.0 - 1.0; }
            return 1.0;
        }
        if status != 0 { return 0.0; }
        let moves = legal_moves(state);
        let num = len(moves) / 2;
        let a = 0;
        if state_side(state) == agent_side {
            let sel = select_action_eval_temp_v22(weights, state, moves, eval_temp);
            a = sel[0];
        } else {
            let pdata = [];
            let ii = 0;
            while ii < num {
                pdata = array_push(pdata, 1.0 / float(num));
                ii = ii + 1;
            }
            let probs = Tensor.from_vec(pdata, [num]);
            a = categorical_sample(probs);
        }
        let from_sq = moves[a * 2];
        let to_sq = moves[a * 2 + 1];
        state = apply_move(state, from_sq, to_sq);
        let pair = rep_inc_v22(rep_map, state);
        rep_map = pair[0];
        let count = pair[1];
        if count >= 3 { return 0.0; }
        step = step + 1;
    }
    0.0
}

fn eval_vs_random_v22(weights: Any, n_games: i64, max_moves: i64,
                      eval_temp: f64) -> Any {
    let wins = 0;
    let draws = 0;
    let losses = 0;
    let i = 0;
    while i < n_games {
        let side = 1;
        if i % 2 == 1 { side = 0 - 1; }
        let out = play_vs_random_v22(weights, side, max_moves, eval_temp);
        if out > 0.5 { wins = wins + 1; }
        if out < 0.0 - 0.5 { losses = losses + 1; }
        if out > 0.0 - 0.5 && out < 0.5 { draws = draws + 1; }
        i = i + 1;
    }
    [wins, draws, losses]
}

fn play_vs_greedy_v22(weights: Any, agent_side: i64, max_moves: i64,
                      eval_temp: f64) -> f64 {
    let state = init_state();
    let rep_map = map_new();
    let pair0 = rep_inc_v22(rep_map, state);
    rep_map = pair0[0];
    let step = 0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            let loser = state_side(state);
            if loser == agent_side { return 0.0 - 1.0; }
            return 1.0;
        }
        if status != 0 { return 0.0; }
        let moves = legal_moves(state);
        let a = 0;
        if state_side(state) == agent_side {
            let sel = select_action_eval_temp_v22(weights, state, moves, eval_temp);
            a = sel[0];
        } else {
            a = select_action_material_greedy(state, moves);
        }
        let from_sq = moves[a * 2];
        let to_sq = moves[a * 2 + 1];
        state = apply_move(state, from_sq, to_sq);
        let pair = rep_inc_v22(rep_map, state);
        rep_map = pair[0];
        let count = pair[1];
        if count >= 3 { return 0.0; }
        step = step + 1;
    }
    0.0
}

fn eval_vs_greedy_v22(weights: Any, n_games: i64, max_moves: i64,
                      eval_temp: f64) -> Any {
    let wins = 0;
    let draws = 0;
    let losses = 0;
    let i = 0;
    while i < n_games {
        let side = 1;
        if i % 2 == 1 { side = 0 - 1; }
        let out = play_vs_greedy_v22(weights, side, max_moves, eval_temp);
        if out > 0.5 { wins = wins + 1; }
        if out < 0.0 - 0.5 { losses = losses + 1; }
        if out > 0.0 - 0.5 && out < 0.5 { draws = draws + 1; }
        i = i + 1;
    }
    [wins, draws, losses]
}

fn play_snapshot_v22(w_curr: Any, w_prev: Any, curr_side: i64, max_moves: i64,
                     eval_temp: f64) -> f64 {
    let state = init_state();
    let rep_map = map_new();
    let pair0 = rep_inc_v22(rep_map, state);
    rep_map = pair0[0];
    let step = 0;
    while step < max_moves {
        let status = terminal_status(state);
        if status == 2 {
            let loser = state_side(state);
            if loser == curr_side { return 0.0 - 1.0; }
            return 1.0;
        }
        if status != 0 { return 0.0; }
        let moves = legal_moves(state);
        let player = w_prev;
        if state_side(state) == curr_side { player = w_curr; }
        let sel = select_action_eval_temp_v22(player, state, moves, eval_temp);
        let a = sel[0];
        let from_sq = moves[a * 2];
        let to_sq = moves[a * 2 + 1];
        state = apply_move(state, from_sq, to_sq);
        let pair = rep_inc_v22(rep_map, state);
        rep_map = pair[0];
        let count = pair[1];
        if count >= 3 { return 0.0; }
        step = step + 1;
    }
    0.0
}

fn eval_vs_snapshot_v22(w_curr: Any, w_prev: Any, n_games: i64, max_moves: i64,
                        eval_temp: f64) -> Any {
    let wins = 0;
    let draws = 0;
    let losses = 0;
    let i = 0;
    while i < n_games {
        let side = 1;
        if i % 2 == 1 { side = 0 - 1; }
        let out = play_snapshot_v22(w_curr, w_prev, side, max_moves, eval_temp);
        if out > 0.5 { wins = wins + 1; }
        if out < 0.0 - 0.5 { losses = losses + 1; }
        if out > 0.0 - 0.5 && out < 0.5 { draws = draws + 1; }
        i = i + 1;
    }
    [wins, draws, losses]
}

fn gauntlet_vs_snapshots_v22(w_curr: Any, snapshots: Any, snapshot_ratings: Any,
                             r_self: f64, n_games_each: i64, max_moves: i64,
                             k: f64, eval_temp: f64) -> Any {
    let rating = r_self;
    let total_w = 0;
    let total_d = 0;
    let total_l = 0;
    let i = 0;
    while i < len(snapshots) {
        let snap = snapshots[i];
        let opp_r = snapshot_ratings[i];
        let wdl = eval_vs_snapshot_v22(w_curr, snap, n_games_each, max_moves, eval_temp);
        rating = elo_apply_record(rating, opp_r, wdl, k);
        total_w = total_w + wdl[0];
        total_d = total_d + wdl[1];
        total_l = total_l + wdl[2];
        i = i + 1;
    }
    [rating, total_w, total_d, total_l]
}

// ==========================================================================
// ============== V2.3 PROFILING — instrumented rollout
// ==========================================================================
// This is `rollout_episode_v22` with profile zones wrapped around each
// hot section. The math is identical — profile counters are write-only.
// Used by Tier 2 to measure the hot path and by the v23 parity test to
// verify that instrumentation does not perturb the weight hash.

fn rollout_episode_v22_instrumented(weights: Any, max_moves: i64, temp: f64,
                                     penalty_per_ply: f64, dump_path: Any) -> Any {
    let h_total = profile_zone_start("rollout_total");
    let state = init_state();
    let states_list = [];
    let moves_list = [];
    let action_list = [];
    let value_list = [];
    let side_list = [];
    let rep_map = map_new();
    let seed_pair = rep_inc_v22(rep_map, state);
    rep_map = seed_pair[0];
    let step = 0;
    let terminal_reward = 0.0;
    let rep_flag = 0;
    while step < max_moves {
        let status_base = terminal_status(state);
        if status_base == 2 {
            terminal_reward = float(0 - state_side(state));
            step = max_moves + 1;
        } else {
            if status_base != 0 {
                terminal_reward = 0.0;
                step = max_moves + 1;
            } else {
                let h_lm = profile_zone_start("legal_moves");
                let moves = legal_moves(state);
                profile_zone_stop(h_lm);

                let h_score = profile_zone_start("score_moves");
                let sel = select_action_temp(weights, state, moves, temp);
                profile_zone_stop(h_score);

                let a = sel[0];
                let v = sel[2];
                states_list = array_push(states_list, state);
                moves_list = array_push(moves_list, moves);
                action_list = array_push(action_list, a);
                value_list = array_push(value_list, v);
                side_list = array_push(side_list, state_side(state));
                let from_sq = moves[a * 2];
                let to_sq = moves[a * 2 + 1];

                let h_apply = profile_zone_start("apply_move");
                state = apply_move(state, from_sq, to_sq);
                profile_zone_stop(h_apply);

                let h_rep = profile_zone_start("rep_tracking");
                let pair = rep_inc_v22(rep_map, state);
                rep_map = pair[0];
                let count = pair[1];
                profile_zone_stop(h_rep);

                if count >= 3 {
                    terminal_reward = 0.0;
                    rep_flag = 1;
                    step = max_moves + 1;
                } else {
                    step = step + 1;
                }
            }
        }
    }
    let n_moves = len(states_list);
    let adjusted = apply_move_penalty_v22(terminal_reward, n_moves, penalty_per_ply);
    profile_zone_stop(h_total);
    profile_dump(dump_path);
    [states_list, moves_list, action_list, value_list, side_list,
     adjusted, n_moves, rep_flag]
}

// Instrumented training episode: same as train_one_episode_adam_v22 but
// wraps the a2c_update in a profile zone.
fn train_one_episode_adam_v22_instrumented(weights: Any, adam: Any, max_moves: i64,
                                           lr: f64, temp: f64, penalty: f64,
                                           dump_path: Any) -> Any {
    let rollout = rollout_episode_v22_instrumented(weights, max_moves, temp, penalty, dump_path);
    let n_moves = rollout[6];
    let terminal_reward = rollout[5];
    let rep_flag = rollout[7];
    if n_moves == 0 {
        return [weights, adam, 0.0, 0, terminal_reward, rep_flag];
    }
    let h_a2c = profile_zone_start("a2c_update");
    let result = a2c_update_adam(weights, adam, rollout, lr, 0.5, 0.01, 1.0);
    profile_zone_stop(h_a2c);
    let new_weights = result[0];
    let new_adam = result[1];
    let losses = result[2];
    profile_dump(dump_path);
    [new_weights, new_adam, losses[0], n_moves, terminal_reward, rep_flag]
}

// ==========================================================================
// ============== V2.3 NATIVE KERNEL ROLLOUT
// ==========================================================================
// Uses `encode_state_fast` and `score_moves_batch` native builtins to
// replace the O(n^2) CJC-Lang loops in the forward pass. The rest of the
// rollout is identical to v2.2.
//
// The weights_to_10 helper extracts the 10 actual tensors (skipping
// weights[4] placeholder) for score_moves_batch.

fn score_moves_v23(weights: Any, state: Any, moves: Any) -> Any {
    let board = state_board(state);
    let side = state_side(state);
    let castling = state_castling(state);
    let ep_sq = state_ep(state);
    let halfmove = state_halfmove(state);
    let features = encode_state_fast(board, side, castling, ep_sq, halfmove);
    score_moves_batch(weights, features, moves, side)
}

fn select_action_temp_v23(weights: Any, state: Any, moves: Any, temp: f64) -> Any {
    let sv = score_moves_v23(weights, state, moves);
    let scores_t = sv[0];
    let v = sv[1];
    let inv_t = 1.0 / temp;
    let scaled = scores_t * inv_t;
    let probs = scaled.softmax();
    let action = categorical_sample(probs);
    let lp = log(probs.get([action]));
    [action, lp, v]
}

fn rollout_episode_v23(weights: Any, max_moves: i64, temp: f64,
                       penalty_per_ply: f64) -> Any {
    let state = init_state();
    let states_list = [];
    let moves_list = [];
    let action_list = [];
    let value_list = [];
    let side_list = [];
    let rep_map = map_new();
    let seed_pair = rep_inc_v22(rep_map, state);
    rep_map = seed_pair[0];
    let step = 0;
    let terminal_reward = 0.0;
    let rep_flag = 0;
    while step < max_moves {
        let status_base = terminal_status(state);
        if status_base == 2 {
            terminal_reward = float(0 - state_side(state));
            step = max_moves + 1;
        } else {
            if status_base != 0 {
                terminal_reward = 0.0;
                step = max_moves + 1;
            } else {
                let moves = legal_moves(state);
                let sel = select_action_temp_v23(weights, state, moves, temp);
                let a = sel[0];
                let v = sel[2];
                states_list = array_push(states_list, state);
                moves_list = array_push(moves_list, moves);
                action_list = array_push(action_list, a);
                value_list = array_push(value_list, v);
                side_list = array_push(side_list, state_side(state));
                let from_sq = moves[a * 2];
                let to_sq = moves[a * 2 + 1];
                state = apply_move(state, from_sq, to_sq);
                let pair = rep_inc_v22(rep_map, state);
                rep_map = pair[0];
                let count = pair[1];
                if count >= 3 {
                    terminal_reward = 0.0;
                    rep_flag = 1;
                    step = max_moves + 1;
                } else {
                    step = step + 1;
                }
            }
        }
    }
    let n_moves = len(states_list);
    let adjusted = apply_move_penalty_v22(terminal_reward, n_moves, penalty_per_ply);
    [states_list, moves_list, action_list, value_list, side_list,
     adjusted, n_moves, rep_flag]
}

fn train_one_episode_adam_v23(weights: Any, adam: Any, max_moves: i64,
                              lr: f64, temp: f64, penalty: f64) -> Any {
    let rollout = rollout_episode_v23(weights, max_moves, temp, penalty);
    let n_moves = rollout[6];
    let terminal_reward = rollout[5];
    let rep_flag = rollout[7];
    if n_moves == 0 {
        return [weights, adam, 0.0, 0, terminal_reward, rep_flag];
    }
    let result = a2c_update_adam(weights, adam, rollout, lr, 0.5, 0.01, 1.0);
    let new_weights = result[0];
    let new_adam = result[1];
    let losses = result[2];
    [new_weights, new_adam, losses[0], n_moves, terminal_reward, rep_flag]
}

// ---- V2.3 eval helpers (reuse v22 helpers since bottleneck is training) ----
fn eval_vs_random_v23(weights: Any, n_games: i64, max_moves: i64,
                      eval_temp: f64) -> Any {
    eval_vs_random_v22(weights, n_games, max_moves, eval_temp)
}

fn eval_vs_greedy_v23(weights: Any, n_games: i64, max_moves: i64) -> Any {
    eval_vs_greedy_v22(weights, n_games, max_moves)
}
"#;
