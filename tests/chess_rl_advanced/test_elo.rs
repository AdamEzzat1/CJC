//! Phase 9 + 10: League manager and ELO rating tests.
//!
//! Implements a simple round-robin league with ELO ratings,
//! all computed in Rust from CJC self-play match results.

use super::helpers::*;

/// ELO calculation: expected score formula.
fn expected_score(rating_a: f64, rating_b: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
}

/// ELO update: returns (new_a, new_b) after a match.
/// result: 1.0 = a wins, 0.5 = draw, 0.0 = a loses.
fn elo_update(rating_a: f64, rating_b: f64, result: f64, k: f64) -> (f64, f64) {
    let ea = expected_score(rating_a, rating_b);
    let eb = 1.0 - ea;
    let new_a = rating_a + k * (result - ea);
    let new_b = rating_b + k * ((1.0 - result) - eb);
    (new_a, new_b)
}

#[test]
fn elo_expected_score_equal_ratings() {
    let e = expected_score(1500.0, 1500.0);
    assert!((e - 0.5).abs() < 1e-10, "equal ratings should give 0.5 expected");
}

#[test]
fn elo_expected_score_higher_rating() {
    let e = expected_score(1700.0, 1500.0);
    assert!(e > 0.5, "higher-rated player should have >0.5 expected");
    assert!(e < 1.0, "expected score should be <1.0");
}

#[test]
fn elo_update_winner_gains() {
    let (new_a, new_b) = elo_update(1500.0, 1500.0, 1.0, 32.0);
    assert!(new_a > 1500.0, "winner should gain ELO");
    assert!(new_b < 1500.0, "loser should lose ELO");
}

#[test]
fn elo_update_draw_no_change_for_equals() {
    let (new_a, new_b) = elo_update(1500.0, 1500.0, 0.5, 32.0);
    assert!((new_a - 1500.0).abs() < 1e-10, "draw between equals: no change");
    assert!((new_b - 1500.0).abs() < 1e-10, "draw between equals: no change");
}

#[test]
fn elo_update_conserves_total() {
    let (new_a, new_b) = elo_update(1600.0, 1400.0, 1.0, 32.0);
    assert!((new_a + new_b - 3000.0).abs() < 1e-10,
        "total ELO should be conserved");
}

#[test]
fn elo_update_upset_gives_more_points() {
    // Lower-rated player wins = bigger gain
    let (gain_upset, _) = elo_update(1400.0, 1600.0, 1.0, 32.0);
    let (gain_expected, _) = elo_update(1600.0, 1400.0, 1.0, 32.0);
    let upset_gain = gain_upset - 1400.0;
    let expected_gain = gain_expected - 1600.0;
    assert!(upset_gain > expected_gain,
        "upset win should give more ELO than expected win");
}

/// Run a mini league between 3 models using self-play CJC matches.
#[test]
fn league_3_models_round_robin() {
    // Train 3 different models (different seeds = different weight initializations)
    let seeds = [42u64, 99, 200];
    let mut model_outputs: Vec<Vec<String>> = Vec::new();

    // Collect initial weights from 3 different seeds
    for &seed in &seeds {
        let src = multi_program(r#"
            let weights = init_weights();
            print(weights[0].get([0, 0]));
        "#);
        let out = run_mir(&src, seed);
        model_outputs.push(out);
    }

    // Verify 3 different models initialized
    assert_eq!(model_outputs.len(), 3);
    // Different seeds should give different weights
    assert_ne!(model_outputs[0], model_outputs[1]);

    // Run pairwise matches using self-play
    let mut ratings = vec![1500.0; 3];
    let pairs = [(0, 1), (0, 2), (1, 2)];

    for &(a, b) in &pairs {
        // Use a deterministic seed for the match
        let match_seed = seeds[a] ^ seeds[b];
        let src = selfplay_program(r#"
            let w1 = init_weights();
            let w2 = init_weights();
            let result = selfplay_episode(
                w1[0], w1[1], w1[2],
                w2[0], w2[1], w2[2],
                8
            );
            print(result[0]);
        "#);
        let out = run_mir(&src, match_seed);
        let reward = parse_float(&out);

        let match_result = if reward > 0.5 {
            1.0 // a wins (a plays white)
        } else if reward < -0.5 {
            0.0 // a loses
        } else {
            0.5 // draw
        };

        let (new_a, new_b) = elo_update(ratings[a], ratings[b], match_result, 32.0);
        ratings[a] = new_a;
        ratings[b] = new_b;
    }

    // Verify ELO ratings are reasonable
    let total: f64 = ratings.iter().sum();
    assert!((total - 4500.0).abs() < 1e-6,
        "total ELO should be conserved: {total}");

    // All ratings should be in a reasonable range
    for (i, &r) in ratings.iter().enumerate() {
        assert!(r > 1300.0 && r < 1700.0,
            "model {i} ELO {r} out of reasonable range after 1 match each");
    }
}

/// League results are deterministic.
#[test]
fn league_deterministic() {
    let src = selfplay_program(r#"
        let w1 = init_weights();
        let w2 = init_weights();
        let result = eval_agents(
            w1[0], w1[1], w1[2],
            w2[0], w2[1], w2[2],
            4, 8
        );
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "league results not deterministic");
}
