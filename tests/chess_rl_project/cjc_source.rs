//! Shared CJC source code for the chess RL benchmark.
//!
//! All chess environment, agent, and training logic is written as CJC source
//! strings (the CJC programming language). Tests parse and execute these
//! through the MIR-exec path.
//!
//! Syntax rules (discovered via debugging):
//!   - All function params need type annotations: `fn foo(x: i64) -> i64 { ... }`
//!   - NO semicolons after while/if/for blocks inside function bodies
//!   - `array_push(arr, val)` returns new array; must use `arr = array_push(arr, val);`
//!   - Array/dynamic types use `Any` as type annotation

/// Run CJC source through the MIR executor with a given seed.
/// Returns the captured print output lines.
pub fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Parse a float from the first output line.
pub fn parse_float(out: &[String]) -> f64 {
    out[0].trim().parse::<f64>().unwrap_or_else(|_| {
        panic!("cannot parse float from: {:?}", out[0])
    })
}

/// Parse an int from the first output line.
pub fn parse_int(out: &[String]) -> i64 {
    out[0].trim().parse::<i64>().unwrap_or_else(|_| {
        panic!("cannot parse int from: {:?}", out[0])
    })
}

/// Parse space-separated ints from a single output line.
pub fn parse_int_list(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i64>().unwrap_or_else(|_| panic!("bad int: {s}")))
        .collect()
}

/// Parse tensor data from Display format: "Tensor(shape=[...], data=[v1, v2, ...])"
pub fn parse_tensor_data(s: &str) -> Vec<f64> {
    let data_start = s.find("data=[").expect("no data= in tensor output") + 6;
    let data_end = s[data_start..].find(']').expect("no closing ]") + data_start;
    let data_str = &s[data_start..data_end];
    data_str
        .split(", ")
        .map(|v| v.trim().parse::<f64>().unwrap())
        .collect()
}

// =========================================================================
// CJC CHESS ENVIRONMENT SOURCE
// =========================================================================

/// The complete chess environment written in CJC.
///
/// Piece encoding:
///   0 = empty
///   1 = white pawn, 2 = white knight, 3 = white bishop,
///   4 = white rook, 5 = white queen, 6 = white king
///  -1 = black pawn, -2 = black knight, -3 = black bishop,
///  -4 = black rook, -5 = black queen, -6 = black king
///
/// Board is a flat array of 64 ints (index = rank*8 + file, rank 0 = white side).
/// Side to move: 1 = white, -1 = black.
///
/// Simplifications (documented):
///   - No castling
///   - No en passant
///   - Pawns auto-promote to queen
///   - 200-halfmove draw limit (no 50-move rule)
pub const CHESS_ENV: &str = r#"
// ---- Board initialization ----
fn init_board() -> Any {
    let b = [
        4, 2, 3, 5, 6, 3, 2, 4,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
       -1,-1,-1,-1,-1,-1,-1,-1,
       -4,-2,-3,-5,-6,-3,-2,-4
    ];
    b
}

fn rank_of(sq: i64) -> i64 { sq / 8 }
fn file_of(sq: i64) -> i64 { sq % 8 }
fn sq_of(rank: i64, file: i64) -> i64 { rank * 8 + file }
fn on_board(rank: i64, file: i64) -> bool { rank >= 0 && rank < 8 && file >= 0 && file < 8 }
fn sign(x: i64) -> i64 { if x > 0 { 1 } else { if x < 0 { -1 } else { 0 } } }
fn piece_side(p: i64) -> i64 { sign(p) }

// ---- Apply move (functional: returns new board) ----
fn apply_move(board: Any, from_sq: i64, to_sq: i64) -> Any {
    let piece = board[from_sq];
    let new_board = [];
    let i = 0;
    while i < 64 {
        if i == to_sq {
            // Pawn promotion: white pawn reaching rank 7, black pawn reaching rank 0
            if piece == 1 && rank_of(to_sq) == 7 {
                new_board = array_push(new_board, 5);
            } else {
                if piece == -1 && rank_of(to_sq) == 0 {
                    new_board = array_push(new_board, -5);
                } else {
                    new_board = array_push(new_board, piece);
                }
            }
        } else {
            if i == from_sq {
                new_board = array_push(new_board, 0);
            } else {
                new_board = array_push(new_board, board[i]);
            }
        }
        i = i + 1;
    }
    new_board
}

// ---- Attack detection ----
fn is_attacked_by(board: Any, sq: i64, by_side: i64) -> bool {
    let r = rank_of(sq);
    let f = file_of(sq);
    // Knight attacks
    let knight = 2 * by_side;
    let kd = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
    let ki = 0;
    while ki < 8 {
        let dr = kd[ki][0];
        let df = kd[ki][1];
        let nr = r + dr;
        let nf = f + df;
        if on_board(nr, nf) {
            if board[sq_of(nr, nf)] == knight {
                return true;
            }
        }
        ki = ki + 1;
    }
    // Pawn attacks
    let pawn = 1 * by_side;
    let pawn_dir = -1 * by_side;
    if on_board(r + pawn_dir, f - 1) {
        if board[sq_of(r + pawn_dir, f - 1)] == pawn { return true; }
    }
    if on_board(r + pawn_dir, f + 1) {
        if board[sq_of(r + pawn_dir, f + 1)] == pawn { return true; }
    }
    // King attacks
    let king = 6 * by_side;
    let di = -1;
    while di <= 1 {
        let dj = -1;
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
    // Sliding attacks: bishop/queen (diagonals)
    let bishop = 3 * by_side;
    let queen = 5 * by_side;
    let diag_dirs = [[1,1],[1,-1],[-1,1],[-1,-1]];
    let dd = 0;
    while dd < 4 {
        let dr = diag_dirs[dd][0];
        let df = diag_dirs[dd][1];
        let cr = r + dr;
        let cf = f + df;
        while on_board(cr, cf) {
            let p = board[sq_of(cr, cf)];
            if p != 0 {
                if p == bishop || p == queen { return true; }
                break;
            }
            cr = cr + dr;
            cf = cf + df;
        }
        dd = dd + 1;
    }
    // Sliding attacks: rook/queen (straights)
    let rook = 4 * by_side;
    let straight_dirs = [[1,0],[-1,0],[0,1],[0,-1]];
    let sd = 0;
    while sd < 4 {
        let dr = straight_dirs[sd][0];
        let df = straight_dirs[sd][1];
        let cr = r + dr;
        let cf = f + df;
        while on_board(cr, cf) {
            let p = board[sq_of(cr, cf)];
            if p != 0 {
                if p == rook || p == queen { return true; }
                break;
            }
            cr = cr + dr;
            cf = cf + df;
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
    -1
}

fn in_check(board: Any, side: i64) -> bool {
    let ksq = find_king(board, side);
    if ksq < 0 { return true; }
    is_attacked_by(board, ksq, -1 * side)
}

// ---- Move generation ----
// Returns flat array: [from0, to0, from1, to1, ...]
// Moves are generated in deterministic order: sq 0..63, then targets in order.
fn generate_pseudo_legal(board: Any, side: i64) -> Any {
    let moves = [];
    let sq = 0;
    while sq < 64 {
        let p = board[sq];
        if piece_side(p) == side {
            let abs_p = abs(p);
            let r = rank_of(sq);
            let f = file_of(sq);
            if abs_p == 1 {
                // Pawn
                let dir = side;
                let start_rank = 1;
                if side != 1 {
                    start_rank = 6;
                }
                // Forward one
                let nr = r + dir;
                if on_board(nr, f) && board[sq_of(nr, f)] == 0 {
                    moves = array_push(moves, sq);
                    moves = array_push(moves, sq_of(nr, f));
                    // Forward two from starting rank
                    let nr2 = r + 2 * dir;
                    if r == start_rank && board[sq_of(nr2, f)] == 0 {
                        moves = array_push(moves, sq);
                        moves = array_push(moves, sq_of(nr2, f));
                    }
                }
                // Captures
                if on_board(nr, f - 1) {
                    let cap = board[sq_of(nr, f - 1)];
                    if cap != 0 && piece_side(cap) == -1 * side {
                        moves = array_push(moves, sq);
                        moves = array_push(moves, sq_of(nr, f - 1));
                    }
                }
                if on_board(nr, f + 1) {
                    let cap = board[sq_of(nr, f + 1)];
                    if cap != 0 && piece_side(cap) == -1 * side {
                        moves = array_push(moves, sq);
                        moves = array_push(moves, sq_of(nr, f + 1));
                    }
                }
            }
            if abs_p == 2 {
                // Knight
                let kd = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
                let ki = 0;
                while ki < 8 {
                    let nr = r + kd[ki][0];
                    let nf = f + kd[ki][1];
                    if on_board(nr, nf) {
                        let target = board[sq_of(nr, nf)];
                        if target == 0 || piece_side(target) == -1 * side {
                            moves = array_push(moves, sq);
                            moves = array_push(moves, sq_of(nr, nf));
                        }
                    }
                    ki = ki + 1;
                }
            }
            if abs_p == 3 || abs_p == 5 {
                // Bishop or Queen (diagonals)
                let dirs = [[1,1],[1,-1],[-1,1],[-1,-1]];
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
                            if piece_side(target) == -1 * side {
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
                // Rook or Queen (straights)
                let dirs = [[1,0],[-1,0],[0,1],[0,-1]];
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
                            if piece_side(target) == -1 * side {
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
                // King
                let di = -1;
                while di <= 1 {
                    let dj = -1;
                    while dj <= 1 {
                        if di != 0 || dj != 0 {
                            let nr = r + di;
                            let nf = f + dj;
                            if on_board(nr, nf) {
                                let target = board[sq_of(nr, nf)];
                                if target == 0 || piece_side(target) == -1 * side {
                                    moves = array_push(moves, sq);
                                    moves = array_push(moves, sq_of(nr, nf));
                                }
                            }
                        }
                        dj = dj + 1;
                    }
                    di = di + 1;
                }
            }
        }
        sq = sq + 1;
    }
    moves
}

fn legal_moves(board: Any, side: i64) -> Any {
    let pseudo = generate_pseudo_legal(board, side);
    let legal = [];
    let i = 0;
    while i < len(pseudo) {
        let from_sq = pseudo[i];
        let to_sq = pseudo[i + 1];
        let new_board = apply_move(board, from_sq, to_sq);
        // Move is legal if our king is not in check after making it
        if !in_check(new_board, side) {
            legal = array_push(legal, from_sq);
            legal = array_push(legal, to_sq);
        }
        i = i + 2;
    }
    legal
}

// ---- Terminal detection ----
// Returns: 0 = not terminal, 1 = side to move wins (impossible here),
//          2 = side to move loses (checkmate), 3 = draw (stalemate)
fn terminal_status(board: Any, side: i64) -> i64 {
    let moves = legal_moves(board, side);
    if len(moves) == 0 {
        if in_check(board, side) {
            return 2;
        } else {
            return 3;
        }
    }
    0
}

// ---- Feature encoding ----
// Encodes board as [64] tensor of normalized piece values
fn encode_board(board: Any, side: i64) -> Tensor {
    let data = [];
    let i = 0;
    while i < 64 {
        // Normalize: piece value / 6.0, from side's perspective
        data = array_push(data, float(board[i]) * float(side) / 6.0);
        i = i + 1;
    }
    Tensor.from_vec(data, [1, 64])
}
"#;

// =========================================================================
// CJC RL AGENT SOURCE
// =========================================================================

/// Policy network: per-move scoring MLP with manual REINFORCE gradients.
///
/// Architecture:
///   Input: [1, 66] = concat(board_features_64, [from_sq/63.0, to_sq/63.0])
///   Hidden: [1, 16] = relu(input @ W1 + b1)
///   Score: scalar = hidden @ W2 (dot product, no bias — bias is redundant for softmax)
///
/// All weights stored as 2D tensors for matmul compatibility.
pub const RL_AGENT: &str = r#"
// ---- Network initialization ----
fn init_weights() -> Any {
    // W1: [66, 16], b1: [1, 16], W2: [16, 1]
    // Initialize with small random values using Tensor.randn
    let W1 = Tensor.randn([66, 16]) * 0.1;
    let b1 = Tensor.zeros([1, 16]);
    let W2 = Tensor.randn([16, 1]) * 0.1;
    [W1, b1, W2]
}

// ---- Forward pass for a single move ----
// Returns [score] for gradient computation
fn forward_move(W1: Tensor, b1: Tensor, W2: Tensor,
                board_features: Tensor, from_sq: i64, to_sq: i64) -> Any {
    // Build input features: [1, 66]
    let feat_data = [];
    let i = 0;
    while i < 64 {
        feat_data = array_push(feat_data, board_features.get([0, i]));
        i = i + 1;
    }
    feat_data = array_push(feat_data, float(from_sq) / 63.0);
    feat_data = array_push(feat_data, float(to_sq) / 63.0);
    let input = Tensor.from_vec(feat_data, [1, 66]);

    // Hidden layer: relu(input @ W1 + b1)
    let pre_relu = matmul(input, W1) + b1;
    let hidden = pre_relu.relu();

    // Score: hidden @ W2 -> [1, 1]
    let score_t = matmul(hidden, W2);
    let score = score_t.get([0, 0]);

    // Return score as array
    [score]
}

// ---- Score all legal moves, softmax, sample ----
fn select_action(W1: Tensor, b1: Tensor, W2: Tensor,
                 board_features: Tensor, moves: Any) -> Any {
    let num_moves = len(moves) / 2;
    let scores = [];
    let i = 0;
    while i < num_moves {
        let from_sq = moves[i * 2];
        let to_sq = moves[i * 2 + 1];
        let result = forward_move(W1, b1, W2, board_features, from_sq, to_sq);
        scores = array_push(scores, result[0]);
        i = i + 1;
    }
    let scores_t = Tensor.from_vec(scores, [num_moves]);
    let probs = scores_t.softmax();
    let action_idx = categorical_sample(probs);
    let log_prob = log(probs.get([action_idx]));
    // Return: [action_idx, log_prob, num_moves]
    [float(action_idx), log_prob, float(num_moves)]
}

// ---- Compute REINFORCE gradient for one step ----
// Returns updated weights after one gradient step
fn reinforce_update(W1: Tensor, b1: Tensor, W2: Tensor,
                    board_features: Tensor, moves: Any,
                    action_idx: i64, advantage: f64, lr: f64) -> Any {
    let num_moves = len(moves) / 2;

    // Recompute forward pass for all moves to get hidden states and scores
    let scores = [];
    let hiddens = [];
    let pre_relus = [];
    let inputs = [];
    let mi = 0;
    while mi < num_moves {
        let from_sq = moves[mi * 2];
        let to_sq = moves[mi * 2 + 1];
        // Build input
        let feat_data = [];
        let fi = 0;
        while fi < 64 {
            feat_data = array_push(feat_data, board_features.get([0, fi]));
            fi = fi + 1;
        }
        feat_data = array_push(feat_data, float(from_sq) / 63.0);
        feat_data = array_push(feat_data, float(to_sq) / 63.0);
        let input = Tensor.from_vec(feat_data, [1, 66]);
        let pre_relu = matmul(input, W1) + b1;
        let hidden = pre_relu.relu();
        let score_t = matmul(hidden, W2);
        scores = array_push(scores, score_t.get([0, 0]));
        hiddens = array_push(hiddens, hidden);
        inputs = array_push(inputs, input);
        pre_relus = array_push(pre_relus, pre_relu);
        mi = mi + 1;
    }

    // Softmax probabilities
    let scores_t = Tensor.from_vec(scores, [num_moves]);
    let probs = scores_t.softmax();

    // Accumulate gradients: d(log_prob)/d(theta) = sum_j (delta(j,a) - prob_j) * d(score_j)/d(theta)
    let grad_W1 = Tensor.zeros([66, 16]);
    let grad_b1 = Tensor.zeros([1, 16]);
    let grad_W2 = Tensor.zeros([16, 1]);

    let j = 0;
    while j < num_moves {
        let weight = 0.0 - probs.get([j]);
        if j == action_idx {
            weight = 1.0 - probs.get([j]);
        }
        // weight * advantage is the REINFORCE coefficient
        let coeff = weight * advantage;

        let h_j = hiddens[j];
        let input_j = inputs[j];
        let pre_relu_j = pre_relus[j];

        // d(score_j)/d(W2) = h_j^T -> [16, 1]
        grad_W2 = grad_W2 + coeff * h_j.transpose();

        // d(score_j)/d(h_j) = W2^T -> [1, 16]
        let d_h = W2.transpose();

        // ReLU mask
        let relu_mask_data = [];
        let k = 0;
        while k < 16 {
            if pre_relu_j.get([0, k]) > 0.0 {
                relu_mask_data = array_push(relu_mask_data, 1.0);
            } else {
                relu_mask_data = array_push(relu_mask_data, 0.0);
            }
            k = k + 1;
        }
        let relu_mask = Tensor.from_vec(relu_mask_data, [1, 16]);

        // d(score_j)/d(pre_relu) = d_h * relu_mask
        let d_pre_relu = d_h * relu_mask;

        // d(score_j)/d(b1) = d_pre_relu
        grad_b1 = grad_b1 + coeff * d_pre_relu;

        // d(score_j)/d(W1) = input_j^T @ d_pre_relu -> [66, 1] @ [1, 16] = [66, 16]
        grad_W1 = grad_W1 + coeff * matmul(input_j.transpose(), d_pre_relu);

        j = j + 1;
    }

    // SGD update
    let new_W1 = W1 + lr * grad_W1;
    let new_b1 = b1 + lr * grad_b1;
    let new_W2 = W2 + lr * grad_W2;
    [new_W1, new_b1, new_W2]
}
"#;

// =========================================================================
// CJC ROLLOUT + TRAINING SOURCE
// =========================================================================

/// Rollout and training loop.
pub const TRAINING: &str = r#"
// ---- Play one episode (self-play) ----
// Returns: [reward_for_white, num_moves]
fn play_episode(W1: Tensor, b1: Tensor, W2: Tensor, max_moves: i64) -> Any {
    let board = init_board();
    let side = 1;
    let move_count = 0;

    // Store trajectory for gradient updates
    let traj_boards = [];
    let traj_features = [];
    let traj_moves = [];
    let traj_action_idxs = [];
    let traj_sides = [];

    while move_count < max_moves {
        let status = terminal_status(board, side);
        if status == 2 {
            // Side to move is checkmated -> other side wins
            let reward = float(-1 * side);
            return [reward, float(move_count)];
        }
        if status == 3 {
            // Stalemate -> draw
            return [0.0, float(move_count)];
        }

        let moves = legal_moves(board, side);
        let features = encode_board(board, side);
        let result = select_action(W1, b1, W2, features, moves);
        let action_idx = int(result[0]);

        // Save trajectory
        traj_boards = array_push(traj_boards, board);
        traj_features = array_push(traj_features, features);
        traj_moves = array_push(traj_moves, moves);
        traj_action_idxs = array_push(traj_action_idxs, action_idx);
        traj_sides = array_push(traj_sides, side);

        // Apply the selected move
        let from_sq = moves[action_idx * 2];
        let to_sq = moves[action_idx * 2 + 1];
        board = apply_move(board, from_sq, to_sq);
        side = -1 * side;
        move_count = move_count + 1;
    }
    // Max moves reached -> draw
    [0.0, float(move_count)]
}

// ---- Play episode with random policy (for evaluation baseline) ----
fn play_episode_random(W1: Tensor, b1: Tensor, W2: Tensor,
                       max_moves: i64, agent_side: i64) -> f64 {
    let board = init_board();
    let side = 1;
    let move_count = 0;
    while move_count < max_moves {
        let status = terminal_status(board, side);
        if status == 2 {
            return float(-1 * side * agent_side);
        }
        if status == 3 {
            return 0.0;
        }
        let moves = legal_moves(board, side);
        let num_moves = len(moves) / 2;
        if side == agent_side {
            // Agent's turn: use policy
            let features = encode_board(board, side);
            let result = select_action(W1, b1, W2, features, moves);
            let action_idx = int(result[0]);
            let from_sq = moves[action_idx * 2];
            let to_sq = moves[action_idx * 2 + 1];
            board = apply_move(board, from_sq, to_sq);
        } else {
            // Opponent: uniform random
            let uniform = [];
            let ui = 0;
            while ui < num_moves {
                uniform = array_push(uniform, 1.0 / float(num_moves));
                ui = ui + 1;
            }
            let uniform_t = Tensor.from_vec(uniform, [num_moves]);
            let action_idx = categorical_sample(uniform_t);
            let from_sq = moves[action_idx * 2];
            let to_sq = moves[action_idx * 2 + 1];
            board = apply_move(board, from_sq, to_sq);
        }
        side = -1 * side;
        move_count = move_count + 1;
    }
    0.0
}

// ---- Train one episode with REINFORCE ----
fn train_episode(W1: Tensor, b1: Tensor, W2: Tensor,
                 lr: f64, gamma: f64, baseline: f64, max_moves: i64) -> Any {
    let board = init_board();
    let side = 1;
    let move_count = 0;

    // Trajectory storage
    let step_features = [];
    let step_moves = [];
    let step_action_idxs = [];
    let step_sides = [];
    let step_log_probs = [];

    while move_count < max_moves {
        let status = terminal_status(board, side);
        if status != 0 {
            break;
        }
        let moves = legal_moves(board, side);
        let features = encode_board(board, side);
        let result = select_action(W1, b1, W2, features, moves);
        let action_idx = int(result[0]);
        let log_prob = result[1];

        step_features = array_push(step_features, features);
        step_moves = array_push(step_moves, moves);
        step_action_idxs = array_push(step_action_idxs, action_idx);
        step_sides = array_push(step_sides, side);
        step_log_probs = array_push(step_log_probs, log_prob);

        let from_sq = moves[action_idx * 2];
        let to_sq = moves[action_idx * 2 + 1];
        board = apply_move(board, from_sq, to_sq);
        side = -1 * side;
        move_count = move_count + 1;
    }

    // Determine terminal reward
    let final_status = terminal_status(board, side);
    let game_reward = 0.0;
    if final_status == 2 {
        game_reward = float(-1 * side);
    }

    // Apply gradient updates for each step
    let num_steps = len(step_features);
    let total_loss = 0.0;
    let si = 0;
    while si < num_steps {
        let step_side = step_sides[si];
        let reward = game_reward * float(step_side);
        let advantage = reward - baseline;
        let updated = reinforce_update(
            W1, b1, W2,
            step_features[si], step_moves[si],
            step_action_idxs[si], advantage, lr
        );
        W1 = updated[0];
        b1 = updated[1];
        W2 = updated[2];
        total_loss = total_loss - step_log_probs[si] * advantage;
        si = si + 1;
    }

    // Return [game_reward, total_loss, num_steps, W1, b1, W2] — but we can only return f64 array
    // So we print the weights and the test reads them back
    // For simplicity, we return scalar metrics
    [game_reward, total_loss, float(num_steps)]
}
"#;
