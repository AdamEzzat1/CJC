//! Phase D v2.3: Native kernel training run (Tier 3).
//!
//! Uses `train_one_episode_adam_v23` which calls `encode_state_fast` and
//! `score_moves_batch` native builtins instead of the pure CJC-Lang
//! equivalents. Measured 7.7× speedup on rollout at max_moves=40.
//!
//! Config: 120 episodes, max_moves=80, lr=0.001, temp 1.2→0.8,
//! penalty=0.001, eval_temp=0.15, seed=42, backend cjc-eval.
//!
//! Gate (from UPGRADE_PROMPT_v2_3.md Tier 4):
//!   • ≤ 45 min wall clock
//!   • ≥ 30/120 non-zero terminal rewards
//!   • ≥ 55% vs random (20 games)
//!   • ≥ 50% vs greedy (10 games)
//!   • Elo gain ≥ 0

mod chess_rl_v2;

use chess_rl_v2::harness::{run, Backend};

const OUT_DIR: &str = "bench_results/chess_rl_v2_3";

#[test]
#[ignore = "Phase D v2.3 training run, invoke with --ignored"]
fn phase_d_v23_training_run() {
    let out_dir = std::path::Path::new(OUT_DIR);
    std::fs::create_dir_all(out_dir).expect("create bench output dir");
    let csv_path = out_dir.join("training_log.csv");
    let ckpt0_path = out_dir.join("checkpoint_ep60.bin");
    let ckpt1_path = out_dir.join("checkpoint_ep120.bin");
    let pgn_path = out_dir.join("sample_games.pgn");
    let loss_svg = out_dir.join("training_loss.svg");
    let reward_svg = out_dir.join("training_reward.svg");
    let summary_path = out_dir.join("phase_d_v23_summary.txt");

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

    let body = format!(
        r#"
        let w = init_weights();
        let adam = init_adam_state();
        let n_ep = 120;
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

        let nonzero_terminals = 0;
        let rep_draws = 0;

        while ep < n_ep {{
            let temp = anneal_temp(ep, n_ep, temp_start, temp_end);
            let result = train_one_episode_adam_v23(w, adam, max_moves, lr, temp, penalty);
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

            if ep == 59 {{
                save_checkpoint("{ckpt0}", w, adam, ep + 1);
                snap_mid = w;
            }}
            if ep == 119 {{
                save_checkpoint("{ckpt1}", w, adam, ep + 1);
            }}

            ep = ep + 1;
        }}

        // Evaluate vs the random baseline.
        let vs_random = eval_vs_random_v22(w, 20, 80, eval_temp);
        print(vs_random[0]);
        print(vs_random[1]);
        print(vs_random[2]);

        // Evaluate vs the material-greedy baseline.
        let vs_greedy = eval_vs_greedy_v22(w, 10, 80, eval_temp);
        print(vs_greedy[0]);
        print(vs_greedy[1]);
        print(vs_greedy[2]);

        // Snapshot gauntlet.
        let snaps = [snap_init, snap_mid];
        let ratings = [1000.0, 1000.0];
        let gauntlet = gauntlet_vs_snapshots_v22(w, snaps, ratings, 1000.0, 4, 80, 32.0, eval_temp);
        print(gauntlet[0]);
        print(gauntlet[1]);
        print(gauntlet[2]);
        print(gauntlet[3]);

        // PGN games.
        let g1 = play_recorded_game(snap_init, w, 80);
        pgn_dump_game("{pgn}", "CJC-Lang Chess RL v2.3", "1", "untrained", "final",   g1[1], g1[0]);
        let g2 = play_recorded_game(snap_mid,  w, 80);
        pgn_dump_game("{pgn}", "CJC-Lang Chess RL v2.3", "2", "mid-train", "final",   g2[1], g2[0]);
        let g3 = play_recorded_game(w, snap_init, 80);
        pgn_dump_game("{pgn}", "CJC-Lang Chess RL v2.3", "3", "final",     "untrained", g3[1], g3[0]);

        // Training curves.
        vizor_training_curve("{loss}", "CJC-Lang Chess RL v2.3 — loss", "loss", episodes_x, losses_y);
        vizor_training_curve("{reward}", "CJC-Lang Chess RL v2.3 — terminal reward", "terminal reward", episodes_x, rewards_y);

        // Final weight hash.
        let hash = tensor_list_hash(weights_to_10(w));
        print(hash);

        // Training diagnostics.
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

    let gate_random = wr_random >= 0.55;
    let gate_greedy = wr_greedy >= 0.50;
    let gate_elo = elo_gain >= 0.0;
    let gate_nonzero = nonzero_terminals >= 30;
    let gate_time = elapsed.as_secs_f64() <= 45.0 * 60.0;

    let summary = format!(
        "CJC-Lang Chess RL v2.3 — Phase D training summary\n\
         =====================================================\n\
         episodes:              120\n\
         max_moves:             80\n\
         lr:                    0.001\n\
         temp_start:            1.2\n\
         temp_end:              0.8\n\
         penalty_per_ply:       0.001\n\
         eval_temp:             0.15\n\
         backend:               cjc-eval (tree-walk + native kernels)\n\
         wall clock:            {wall:.1}s ({wall_min:.2} min)\n\
         per-episode avg:       {per_ep:.1}s\n\
         speedup vs v2.2:       {speedup:.1}× (v2.2 was {v22_per_ep:.1}s/ep)\n\
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
           non-zero terminals: {nzt}/120\n\
           repetition draws:   {rep}/120\n\
         \n\
         final weight hash:     {hash}\n\
         \n\
         Tier 4 gate check:\n\
           [{gr}] ≥55% vs random     (measured: {wr_random:.3})\n\
           [{gg}] ≥50% vs greedy     (measured: {wr_greedy:.3})\n\
           [{ge}] Elo gain ≥ 0       (measured: {elo_gain:+.1})\n\
           [{gn}] ≥30/120 non-zero   (measured: {nzt}/120)\n\
           [{gt}] ≤45 min wall clock (measured: {wall_min:.2} min)\n",
        wall = elapsed.as_secs_f64(),
        wall_min = elapsed.as_secs_f64() / 60.0,
        per_ep = elapsed.as_secs_f64() / 120.0,
        speedup = (73.12 * 60.0 / 60.0) / (elapsed.as_secs_f64() / 120.0), // v2.2 was 73.12min / 60ep
        v22_per_ep = 73.12 * 60.0 / 60.0,
        rw = r_wins, rd = r_draws, rl = r_losses,
        gw = g_wins, gd = g_draws, gl = g_losses,
        gtw = gt_wins, gtd = gt_draws, gtl = gt_losses,
        hash = weight_hash, nzt = nonzero_terminals, rep = rep_draws,
        gr = if gate_random { "PASS" } else { "MISS" },
        gg = if gate_greedy { "PASS" } else { "MISS" },
        ge = if gate_elo { "PASS" } else { "MISS" },
        gn = if gate_nonzero { "PASS" } else { "MISS" },
        gt = if gate_time { "PASS" } else { "MISS" },
    );

    eprintln!("{summary}");
    std::fs::write(&summary_path, &summary).expect("write summary");

    // This test reports gates honestly but does NOT fail on gate miss.
    // The Phase D post-mortem (PHASE_D_v2_3.md) documents the results.
}
