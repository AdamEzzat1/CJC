//! Phase D v2.2: Re-run of the v2.1 honest baseline with Tier 1 cheap ML fixes
//! applied. New levers:
//!
//!   T1-a  max_moves 25 → 80 (gives rollouts a chance to hit checkmate)
//!   T1-b  Move-count penalty = 0.001 per ply on the terminal reward
//!   T1-c  Threefold-repetition detection (draw_reward = 0, rep_flag in CSV)
//!   T1-d  Stochastic low-temperature eval policy (temp = 0.15)
//!   T1-e  `repetition_draw` column added to the training CSV log
//!
//! The driver is otherwise identical to `test_chess_rl_v2_phase_d.rs`. All
//! changes live in the v2.2 PRELUDE additions; the Rust harness here just
//! re-wires the training body to call the new functions.
//!
//! Gate (from UPGRADE_PROMPT_v2_2.md Tier 1):
//!   • ≥60% vs random over 20 games
//!   • ≥55% vs greedy over 10 games
//!   • Elo gain ≥ +25 over 8 gauntlet games
//!   • ≥20/60 training episodes end with non-zero terminal reward
//!   • ≤45 min wall clock
//!
//! If this gate passes, v2.2 ships. If it fails, Tier 2 profiling + Tier 3
//! native kernels come next.

mod chess_rl_v2;

use chess_rl_v2::harness::{run, Backend};

const OUT_DIR: &str = "bench_results/chess_rl_v2_2";

#[test]
#[ignore = "Phase D v2.2 training run, invoke with --ignored"]
fn phase_d_v22_training_run() {
    let out_dir = std::path::Path::new(OUT_DIR);
    std::fs::create_dir_all(out_dir).expect("create bench output dir");
    let csv_path = out_dir.join("training_log.csv");
    let ckpt0_path = out_dir.join("checkpoint_ep30.bin");
    let ckpt1_path = out_dir.join("checkpoint_ep60.bin");
    let pgn_path = out_dir.join("sample_games.pgn");
    let loss_svg = out_dir.join("training_loss.svg");
    let reward_svg = out_dir.join("training_reward.svg");
    let summary_path = out_dir.join("phase_d_v22_summary.txt");

    for p in [
        &csv_path, &ckpt0_path, &ckpt1_path, &pgn_path,
        &loss_svg, &reward_svg, &summary_path,
    ] {
        let _ = std::fs::remove_file(p);
    }

    let csv_str = csv_path.to_string_lossy().replace('\\', "/");
    let ckpt0_str = ckpt0_path.to_string_lossy().replace('\\', "/");
    let ckpt1_str = ckpt1_path.to_string_lossy().replace('\\', "/");
    let pgn_str = pgn_path.to_string_lossy().replace('\\', "/");
    let loss_str = loss_svg.to_string_lossy().replace('\\', "/");
    let reward_str = reward_svg.to_string_lossy().replace('\\', "/");

    // v2.2 driver body. Note the following differences vs v2.1:
    //   * max_moves  25 → 80
    //   * penalty    0.001 per ply
    //   * eval_temp  0.15 (stochastic low-temperature sampling)
    //   * train_one_episode_adam_v22 returns a 6-tuple with rep_flag
    //   * csv_log_episode_v22 writes the 7-column format
    //   * eval functions are the v22 variants (stochastic + repetition-aware)
    //
    // We also track a `nonzero_terminal` counter so the Rust harness can
    // assert the "≥20/60 episodes with non-zero reward" gate.
    let body = format!(
        r#"
        let w = init_weights();
        let adam = init_adam_state();
        let n_ep = 60;
        let max_moves = 80;
        let lr = 0.001;
        let temp_start = 1.2;
        let temp_end = 0.8;
        let penalty = 0.001;
        let eval_temp = 0.15;
        let ep = 0;

        csv_open_log_v22("{csv}");

        let episodes_x = [];
        let losses_y = [];
        let rewards_y = [];

        let snap_init = w;
        let snap_mid  = w;

        // Counters: non-zero terminal rewards + repetition draws seen in training.
        let nonzero_terminals = 0;
        let rep_draws = 0;

        while ep < n_ep {{
            let temp = anneal_temp(ep, n_ep, temp_start, temp_end);
            let result = train_one_episode_adam_v22(w, adam, max_moves, lr, temp, penalty);
            w = result[0];
            adam = result[1];
            let loss = result[2];
            let n_moves = result[3];
            let terminal_reward = result[4];
            let rep_flag = result[5];
            let step = adam[2];

            if terminal_reward > 1.0e-6 || terminal_reward < 0.0 - 1.0e-6 {{
                nonzero_terminals = nonzero_terminals + 1;
            }}
            if rep_flag == 1 {{
                rep_draws = rep_draws + 1;
            }}

            csv_log_episode_v22("{csv}", ep, loss, n_moves, terminal_reward, temp, step, rep_flag);
            episodes_x = array_push(episodes_x, float(ep));
            losses_y = array_push(losses_y, loss);
            rewards_y = array_push(rewards_y, terminal_reward);

            if ep == 29 {{
                save_checkpoint("{ckpt0}", w, adam, ep + 1);
                snap_mid = w;
            }}
            if ep == 59 {{
                save_checkpoint("{ckpt1}", w, adam, ep + 1);
            }}

            ep = ep + 1;
        }}

        // Evaluate vs the random baseline (stochastic eval).
        let vs_random = eval_vs_random_v22(w, 20, 80, eval_temp);
        print(vs_random[0]);
        print(vs_random[1]);
        print(vs_random[2]);

        // Evaluate vs the material-greedy baseline (stochastic eval).
        let vs_greedy = eval_vs_greedy_v22(w, 10, 80, eval_temp);
        print(vs_greedy[0]);
        print(vs_greedy[1]);
        print(vs_greedy[2]);

        // Snapshot gauntlet: 4 games per snapshot × 2 snapshots = 8 games.
        let snaps = [snap_init, snap_mid];
        let ratings = [1000.0, 1000.0];
        let gauntlet = gauntlet_vs_snapshots_v22(w, snaps, ratings, 1000.0, 4, 80, 32.0, eval_temp);
        print(gauntlet[0]);
        print(gauntlet[1]);
        print(gauntlet[2]);
        print(gauntlet[3]);

        // Dump three PGN games: untrained-vs-final, mid-vs-final, final-vs-untrained.
        let g1 = play_recorded_game(snap_init, w, 80);
        pgn_dump_game("{pgn}", "CJC-Lang Chess RL v2.2", "1", "untrained", "final",   g1[1], g1[0]);
        let g2 = play_recorded_game(snap_mid,  w, 80);
        pgn_dump_game("{pgn}", "CJC-Lang Chess RL v2.2", "2", "mid-train", "final",   g2[1], g2[0]);
        let g3 = play_recorded_game(w, snap_init, 80);
        pgn_dump_game("{pgn}", "CJC-Lang Chess RL v2.2", "3", "final",     "untrained", g3[1], g3[0]);

        // Training curves.
        vizor_training_curve("{loss}", "CJC-Lang Chess RL v2.2 — loss", "loss", episodes_x, losses_y);
        vizor_training_curve("{reward}", "CJC-Lang Chess RL v2.2 — terminal reward", "terminal reward", episodes_x, rewards_y);

        // Final weight hash for reproducibility.
        let hash = tensor_list_hash(weights_to_10(w));
        print(hash);

        // Training diagnostics (nonzero terminals, rep draws).
        print(nonzero_terminals);
        print(rep_draws);
    "#,
        csv = csv_str,
        ckpt0 = ckpt0_str,
        ckpt1 = ckpt1_str,
        pgn = pgn_str,
        loss = loss_str,
        reward = reward_str,
    );

    let t = std::time::Instant::now();
    let out = run(Backend::Eval, &body, 42);
    let elapsed = t.elapsed();

    let r_wins: i64 = out[0].trim().parse().unwrap();
    let r_draws: i64 = out[1].trim().parse().unwrap();
    let r_losses: i64 = out[2].trim().parse().unwrap();
    let g_wins: i64 = out[3].trim().parse().unwrap();
    let g_draws: i64 = out[4].trim().parse().unwrap();
    let g_losses: i64 = out[5].trim().parse().unwrap();
    let elo_final: f64 = out[6].trim().parse().unwrap();
    let gt_wins: i64 = out[7].trim().parse().unwrap();
    let gt_draws: i64 = out[8].trim().parse().unwrap();
    let gt_losses: i64 = out[9].trim().parse().unwrap();
    let weight_hash = out[10].trim().to_string();
    let nonzero_terminals: i64 = out[11].trim().parse().unwrap();
    let rep_draws: i64 = out[12].trim().parse().unwrap();

    let r_total = r_wins + r_draws + r_losses;
    let g_total = g_wins + g_draws + g_losses;
    let wr_random = (r_wins as f64 + 0.5 * r_draws as f64) / r_total as f64;
    let wr_greedy = (g_wins as f64 + 0.5 * g_draws as f64) / g_total as f64;
    let elo_gain = elo_final - 1000.0;

    // Gate evaluation (honest reporting — test does NOT fail on gate miss).
    let gate_random = wr_random >= 0.60;
    let gate_greedy = wr_greedy >= 0.55;
    let gate_elo = elo_gain >= 25.0;
    let gate_nonzero = nonzero_terminals >= 20;
    let gate_time = elapsed.as_secs_f64() <= 45.0 * 60.0;

    let summary = format!(
        "CJC-Lang Chess RL v2.2 — Phase D training summary\n\
         =====================================================\n\
         episodes:              60\n\
         max_moves:             80\n\
         lr:                    0.001\n\
         temp_start:            1.2\n\
         temp_end:              0.8\n\
         penalty_per_ply:       0.001\n\
         eval_temp:             0.15\n\
         backend:               cjc-eval (tree-walk)\n\
         wall clock:            {wall:.1}s ({wall_min:.2} min)\n\
         \n\
         vs random (20 games, stochastic eval):\n\
           wins/draws/losses: {rw}/{rd}/{rl}\n\
           win rate:          {wr_random:.3}\n\
         \n\
         vs material-greedy (10 games, stochastic eval):\n\
           wins/draws/losses: {gw}/{gd}/{gl}\n\
           win rate:          {wr_greedy:.3}\n\
         \n\
         snapshot gauntlet (2 snapshots, 4 games each, K=32):\n\
           wins/draws/losses: {gtw}/{gtd}/{gtl}\n\
           final Elo:         {elo_final:.1}\n\
           Elo gain:          {elo_gain:+.1}\n\
         \n\
         training signal:\n\
           non-zero terminals: {nzt}/60\n\
           repetition draws:   {rep}/60\n\
         \n\
         final weight hash:     {hash}\n\
         \n\
         Tier 1 gate check:\n\
           [{gr}] ≥60%% vs random     (measured: {wr_random:.3})\n\
           [{gg}] ≥55%% vs greedy     (measured: {wr_greedy:.3})\n\
           [{ge}] Elo gain ≥ +25      (measured: {elo_gain:+.1})\n\
           [{gn}] ≥20/60 non-zero     (measured: {nzt}/60)\n\
           [{gt}] ≤45 min wall clock  (measured: {wall_min:.2} min)\n",
        wall = elapsed.as_secs_f64(),
        wall_min = elapsed.as_secs_f64() / 60.0,
        rw = r_wins, rd = r_draws, rl = r_losses,
        wr_random = wr_random,
        gw = g_wins, gd = g_draws, gl = g_losses,
        wr_greedy = wr_greedy,
        gtw = gt_wins, gtd = gt_draws, gtl = gt_losses,
        elo_final = elo_final,
        elo_gain = elo_gain,
        nzt = nonzero_terminals,
        rep = rep_draws,
        hash = weight_hash,
        gr = if gate_random { "PASS" } else { "MISS" },
        gg = if gate_greedy { "PASS" } else { "MISS" },
        ge = if gate_elo { "PASS" } else { "MISS" },
        gn = if gate_nonzero { "PASS" } else { "MISS" },
        gt = if gate_time { "PASS" } else { "MISS" },
    );

    println!("\n{summary}");
    std::fs::write(&summary_path, &summary).expect("write summary");

    // Infrastructure invariants (must always hold).
    assert!(csv_path.exists(), "CSV log missing");
    assert!(ckpt0_path.exists(), "ep30 checkpoint missing");
    assert!(ckpt1_path.exists(), "ep60 checkpoint missing");
    assert!(pgn_path.exists(), "PGN dump missing");
    assert!(loss_svg.exists(), "loss plot missing");
    assert!(reward_svg.exists(), "reward plot missing");

    assert!(
        elo_final > 500.0 && elo_final < 2000.0,
        "Elo out of sane range: {elo_final}"
    );

    // CSV should have header + 60 rows = 61 lines, and the header must be the
    // v2.2 7-column format.
    let csv_content = std::fs::read_to_string(&csv_path).unwrap();
    let lines: Vec<&str> = csv_content.lines().collect();
    assert_eq!(lines.len(), 61, "expected 61 CSV lines (1 header + 60 rows)");
    assert_eq!(
        lines[0],
        "episode,loss,n_moves,terminal_reward,temp,adam_step,repetition_draw"
    );

    // PGN should contain exactly 3 event headers.
    let pgn_content = std::fs::read_to_string(&pgn_path).unwrap();
    let event_count = pgn_content
        .matches("[Event \"CJC-Lang Chess RL v2.2\"]")
        .count();
    assert_eq!(event_count, 3, "expected 3 PGN games, got {event_count}");

    // NOTE: The ML gates are reported but do NOT fail the test — this matches
    // v2.1's honest-reporting posture. Gate pass/fail is captured in the summary
    // file for post-hoc analysis.
    println!("--- Tier 1 gate summary ---");
    println!("  random  ≥60%:  {} ({:.3})", gate_random, wr_random);
    println!("  greedy  ≥55%:  {} ({:.3})", gate_greedy, wr_greedy);
    println!("  elo     ≥+25:  {} ({:+.1})", gate_elo, elo_gain);
    println!("  nonzero ≥20:   {} ({}/60)", gate_nonzero, nonzero_terminals);
    println!("  time    ≤45m:  {} ({:.2} min)", gate_time, elapsed.as_secs_f64() / 60.0);
}
