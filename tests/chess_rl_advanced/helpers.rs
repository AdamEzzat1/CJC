//! Shared helpers for chess RL advanced tests.
//!
//! Extends the existing chess_rl_project/hardening helpers with multi-episode
//! training, self-play, snapshot, and evaluation capabilities.

/// Run CJC source through MIR-exec with given seed, return output lines.
pub fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Parse a float from a given output line.
pub fn parse_float_at(out: &[String], idx: usize) -> f64 {
    out[idx].trim().parse::<f64>().unwrap_or_else(|_| {
        panic!("cannot parse float from line {}: {:?}", idx, out[idx])
    })
}

/// Parse a float from the first output line.
pub fn parse_float(out: &[String]) -> f64 {
    parse_float_at(out, 0)
}

/// Parse an int from a given output line.
pub fn parse_int_at(out: &[String], idx: usize) -> i64 {
    out[idx].trim().parse::<i64>().unwrap_or_else(|_| {
        panic!("cannot parse int from line {}: {:?}", idx, out[idx])
    })
}

/// Parse an int from the first output line.
pub fn parse_int(out: &[String]) -> i64 {
    parse_int_at(out, 0)
}

/// Parse space-separated floats from a single output line.
pub fn parse_float_list(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>().unwrap_or_else(|_| panic!("bad float: {s}")))
        .collect()
}

/// Parse space-separated ints from a single output line.
pub fn parse_int_list(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i64>().unwrap_or_else(|_| panic!("bad int: {s}")))
        .collect()
}

/// Chess environment CJC source.
pub fn chess_env_source() -> &'static str {
    crate::chess_rl_project::cjc_source::CHESS_ENV
}

/// RL agent CJC source.
pub fn rl_agent_source() -> &'static str {
    crate::chess_rl_project::cjc_source::RL_AGENT
}

/// Training loop CJC source.
pub fn training_source() -> &'static str {
    crate::chess_rl_project::cjc_source::TRAINING
}

/// Multi-episode training CJC source.
pub fn multi_training_source() -> &'static str {
    MULTI_TRAINING
}

/// Self-play CJC source.
pub fn selfplay_source() -> &'static str {
    SELFPLAY
}

/// Build a CJC program with chess env + custom main.
pub fn chess_program(main_body: &str) -> String {
    format!(
        "{}\nfn main() {{\n{}\n}}",
        chess_env_source(),
        main_body
    )
}

/// Build a CJC program with chess env + RL agent + custom main.
pub fn chess_agent_program(main_body: &str) -> String {
    format!(
        "{}\n{}\nfn main() {{\n{}\n}}",
        chess_env_source(),
        rl_agent_source(),
        main_body
    )
}

/// Build a CJC program with chess env + RL agent + training + custom main.
pub fn full_program(main_body: &str) -> String {
    format!(
        "{}\n{}\n{}\nfn main() {{\n{}\n}}",
        chess_env_source(),
        rl_agent_source(),
        training_source(),
        main_body
    )
}

/// Build a CJC program with all sources including multi-training.
pub fn multi_program(main_body: &str) -> String {
    format!(
        "{}\n{}\n{}\n{}\nfn main() {{\n{}\n}}",
        chess_env_source(),
        rl_agent_source(),
        training_source(),
        multi_training_source(),
        main_body
    )
}

/// Build a CJC program with all sources including self-play.
pub fn selfplay_program(main_body: &str) -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}\nfn main() {{\n{}\n}}",
        chess_env_source(),
        rl_agent_source(),
        training_source(),
        multi_training_source(),
        selfplay_source(),
        main_body
    )
}

// =========================================================================
// MULTI-EPISODE TRAINING SOURCE (Phase 2 + Phase 3)
// =========================================================================

/// Multi-episode training with per-episode metric collection and win-rate evaluation.
pub const MULTI_TRAINING: &str = r#"
// ---- Multi-episode training ----
// Trains for num_episodes episodes, printing per-episode metrics.
// Output format per episode: "EPISODE <i> <reward> <loss> <num_steps>"
fn train_multi(num_episodes: i64, lr: f64, gamma: f64,
               baseline: f64, max_moves: i64) -> Any {
    let weights = init_weights();
    let W1 = weights[0];
    let b1 = weights[1];
    let W2 = weights[2];

    let ep = 0;
    while ep < num_episodes {
        let result = train_episode(W1, b1, W2, lr, gamma, baseline, max_moves);
        let reward = result[0];
        let loss = result[1];
        let steps = result[2];

        // Re-init weights from training output
        // train_episode returns [reward, loss, steps] but updates weights internally
        // We need to re-run with updated weights. But train_episode modifies W1/b1/W2
        // locally. We need a version that returns weights too.
        print(reward);
        print(loss);
        print(steps);

        ep = ep + 1;
    }
    [W1, b1, W2]
}

// ---- Train episode returning updated weights ----
fn train_episode_returning_weights(W1: Tensor, b1: Tensor, W2: Tensor,
                                    lr: f64, gamma: f64, baseline: f64,
                                    max_moves: i64) -> Any {
    let board = init_board();
    let side = 1;
    let move_count = 0;

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

    let final_status = terminal_status(board, side);
    let game_reward = 0.0;
    if final_status == 2 {
        game_reward = float(-1 * side);
    }

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

    // Return [reward, loss, steps, W1, b1, W2]
    [game_reward, total_loss, float(num_steps), W1, b1, W2]
}

// ---- Multi-episode training with weight propagation ----
// Returns final weights after num_episodes training episodes.
// Prints per-episode metrics: one line per metric (reward, loss, steps).
fn train_multi_episodes(num_episodes: i64, lr: f64, gamma: f64,
                         baseline: f64, max_moves: i64) -> Any {
    let weights = init_weights();
    let W1 = weights[0];
    let b1 = weights[1];
    let W2 = weights[2];

    let ep = 0;
    while ep < num_episodes {
        let result = train_episode_returning_weights(W1, b1, W2, lr, gamma, baseline, max_moves);
        let reward = result[0];
        let loss = result[1];
        let steps = result[2];
        W1 = result[3];
        b1 = result[4];
        W2 = result[5];

        print(reward);
        print(loss);
        print(steps);

        ep = ep + 1;
    }
    [W1, b1, W2]
}

// ---- Evaluate win rate against random opponent ----
// Plays num_games against a random opponent, returns win rate.
// Output: prints "<wins> <draws> <losses>" on one line.
fn eval_win_rate(W1: Tensor, b1: Tensor, W2: Tensor,
                 num_games: i64, max_moves: i64, agent_side: i64) -> f64 {
    let wins = 0;
    let draws = 0;
    let losses = 0;
    let g = 0;
    while g < num_games {
        let result = play_episode_random(W1, b1, W2, max_moves, agent_side);
        if result > 0.5 {
            wins = wins + 1;
        } else {
            if result < -0.5 {
                losses = losses + 1;
            } else {
                draws = draws + 1;
            }
        }
        g = g + 1;
    }
    print(wins);
    print(draws);
    print(losses);
    float(wins) / float(num_games)
}
"#;

// =========================================================================
// SELF-PLAY SOURCE (Phase 7)
// =========================================================================

/// Self-play training: two separate weight sets playing against each other.
pub const SELFPLAY: &str = r#"
// ---- Self-play: two agents with separate weights ----
// Returns: [reward_for_white, num_moves]
fn selfplay_episode(W1_w: Tensor, b1_w: Tensor, W2_w: Tensor,
                    W1_b: Tensor, b1_b: Tensor, W2_b: Tensor,
                    max_moves: i64) -> Any {
    let board = init_board();
    let side = 1;
    let move_count = 0;

    while move_count < max_moves {
        let status = terminal_status(board, side);
        if status == 2 {
            return [float(-1 * side), float(move_count)];
        }
        if status == 3 {
            return [0.0, float(move_count)];
        }

        let moves = legal_moves(board, side);
        let features = encode_board(board, side);

        // Select which agent to use based on side
        let action_idx = 0;
        if side == 1 {
            let result = select_action(W1_w, b1_w, W2_w, features, moves);
            action_idx = int(result[0]);
        } else {
            let result = select_action(W1_b, b1_b, W2_b, features, moves);
            action_idx = int(result[0]);
        }

        let from_sq = moves[action_idx * 2];
        let to_sq = moves[action_idx * 2 + 1];
        board = apply_move(board, from_sq, to_sq);
        side = -1 * side;
        move_count = move_count + 1;
    }
    [0.0, float(move_count)]
}

// ---- Evaluate one agent vs another over multiple games ----
// Returns: win rate for agent_a (playing as both white and black)
fn eval_agents(W1_a: Tensor, b1_a: Tensor, W2_a: Tensor,
               W1_b: Tensor, b1_b: Tensor, W2_b: Tensor,
               num_games: i64, max_moves: i64) -> Any {
    let a_wins = 0;
    let draws = 0;
    let b_wins = 0;
    let g = 0;
    while g < num_games {
        // Alternate sides
        let result = [0.0, 0.0];
        if g % 2 == 0 {
            // a plays white
            result = selfplay_episode(W1_a, b1_a, W2_a, W1_b, b1_b, W2_b, max_moves);
            if result[0] > 0.5 {
                a_wins = a_wins + 1;
            } else {
                if result[0] < -0.5 {
                    b_wins = b_wins + 1;
                } else {
                    draws = draws + 1;
                }
            }
        } else {
            // a plays black
            result = selfplay_episode(W1_b, b1_b, W2_b, W1_a, b1_a, W2_a, max_moves);
            if result[0] < -0.5 {
                a_wins = a_wins + 1;
            } else {
                if result[0] > 0.5 {
                    b_wins = b_wins + 1;
                } else {
                    draws = draws + 1;
                }
            }
        }
        g = g + 1;
    }
    print(a_wins);
    print(draws);
    print(b_wins);
    [float(a_wins), float(draws), float(b_wins)]
}
"#;
