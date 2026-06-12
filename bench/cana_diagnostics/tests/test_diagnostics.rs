//! Phase D diagnostics harness tests — wiring → unit → proptest → bolero
//! (the standing test-discipline contract).
//!
//! The wiring tier exercises the REAL artifacts: the committed corpus
//! (`bench_results/cana_ablation/profiles.cpdb`) and the committed
//! CPB0/CPB1 bundles. A failure here means either snapshot drift
//! (gate 1), plan-machinery drift (gate 2), an executor determinism
//! break (gate 3), or energy-formula drift (gate 4) — all of which
//! must fail BEFORE any stopwatch is trusted.

use cana_diagnostics::*;

// =============================================================================
// Wiring — subjects, gates, child workload against committed artifacts
// =============================================================================

#[test]
fn subjects_shape_matches_handoff() {
    let all = subjects();
    assert_eq!(all.len(), 23, "6 selector + 7 thermal + 9 tensor + 1 real");

    let count = |fam: &str| all.iter().filter(|s| s.family == fam).count();
    assert_eq!(count("selector"), 6);
    assert_eq!(count("thermal"), 7);
    assert_eq!(count("tensor"), 9);
    assert_eq!(count("nonsynthetic"), 1);

    // Names unique; every arm A is baseline (gate 4 normalization
    // depends on it); subject names are protocol-safe (no whitespace).
    let mut names: Vec<&str> = all.iter().map(|s| s.name.as_str()).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(before, names.len(), "duplicate subject names");
    for s in &all {
        assert_eq!(s.arm_a, "baseline", "{}: arm A must be baseline", s.name);
        assert!(
            !s.name.contains(char::is_whitespace),
            "{}: name breaks the child line protocol",
            s.name
        );
    }

    // The six named selector wins, exactly (handoff §2 arm 1).
    let wins: Vec<&str> = all
        .iter()
        .filter(|s| s.family == "selector")
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(
        wins,
        vec![
            "mem_grad_a1",
            "mem_grad_a2",
            "mem_grad_a3",
            "mem_grad_a4",
            "mem_grad_a5",
            "holdout_alloc_pulse"
        ]
    );
}

#[test]
fn gate1_all_corpus_subjects_match_committed_hashes() {
    // The strongest snapshot-drift guard: EVERY corpus-verified subject
    // source must hash to its committed corpus row. Compile-only (no
    // execution), so this stays cheap in dev profile.
    let corpus = CorpusIndex::load().expect("committed corpus must load");
    for subject in subjects().iter().filter(|s| s.corpus_verified) {
        let compiled = compile_subject(&subject.source).expect(&subject.name);
        gate1_program_hash(subject, &compiled.features, &corpus)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

#[test]
fn gates_2_3_4_pass_on_a_selector_win() {
    // mem_grad_a1 is the cheapest selector win (256-iteration loop):
    // recompute both arm plans, hold them to the committed corpus rows,
    // then the output-determinism and energy-reproduction gates.
    let corpus = CorpusIndex::load().expect("corpus");
    let heads = load_heads().expect("committed bundles");
    let subject = subjects()
        .into_iter()
        .find(|s| s.name == "mem_grad_a1")
        .unwrap();

    let compiled = compile_subject(&subject.source).unwrap();
    let recorded = record_pressures(&compiled.ast).unwrap();
    let plan_a =
        plan_for_config(subject.arm_a, &compiled.mir, &compiled.features, &recorded, &heads)
            .unwrap();
    let plan_b =
        plan_for_config(subject.arm_b, &compiled.mir, &compiled.features, &recorded, &heads)
            .unwrap();
    gate2_plan_identity(&subject, subject.arm_a, &plan_a, &corpus).unwrap_or_else(|e| panic!("{e}"));
    gate2_plan_identity(&subject, subject.arm_b, &plan_b, &corpus).unwrap_or_else(|e| panic!("{e}"));

    // A selector WIN must have a plan that actually differs.
    assert_ne!(
        plan_as_sorted_vec(&plan_a),
        plan_as_sorted_vec(&plan_b),
        "mem_grad_a1 is a named selector win; identical plans would make it a control"
    );

    let opt_a = optimize_with(&compiled.mir, &plan_a, "test/a").unwrap();
    let opt_b = optimize_with(&compiled.mir, &plan_b, "test/b").unwrap();
    let (transcript, fnv) = gate3_output_determinism(&compiled.ast, &opt_a, &opt_b).unwrap();
    assert!(!transcript.is_empty(), "subject must print something");
    assert_eq!(fnv, output_fnv(&transcript));

    let evidence = measure_modeled_energy(&compiled.ast, &opt_a, &opt_b).unwrap();
    gate4_energy_reproduction(&subject, &evidence, &corpus).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        evidence.ratio_b < 1.0,
        "mem_grad_a1 is a modeled win; ratio {} should be < 1",
        evidence.ratio_b
    );
}

#[test]
fn child_workload_is_arm_invariant_on_output() {
    // The in-process core of the child mode: both arms must produce the
    // SAME output digest, wall-clock must tick, and the RSS peak can
    // only grow from plan-application-time to loop-end. Plans travel
    // parent -> child through the file protocol, so this test pushes
    // them through serialize/parse rather than handing them over
    // directly.
    let heads = load_heads().expect("committed bundles");
    let subject = subjects()
        .into_iter()
        .find(|s| s.name == "mem_grad_a1")
        .unwrap();
    let compiled = compile_subject(&subject.source).unwrap();
    let recorded = record_pressures(&compiled.ast).unwrap();
    let plan_a = parse_plan(&serialize_plan(
        &plan_for_config(subject.arm_a, &compiled.mir, &compiled.features, &recorded, &heads)
            .unwrap(),
    ))
    .expect("arm A plan survives the file protocol");
    let plan_b = parse_plan(&serialize_plan(
        &plan_for_config(subject.arm_b, &compiled.mir, &compiled.features, &recorded, &heads)
            .unwrap(),
    ))
    .expect("arm B plan survives the file protocol");

    let a = run_child_workload(&subject, Arm::A, 2, &plan_a).unwrap();
    let b = run_child_workload(&subject, Arm::B, 2, &plan_b).unwrap();
    assert_eq!(a.output_fnv, b.output_fnv, "arms diverged on output");
    assert_eq!(a.iters, 2);
    assert!(a.wall_micros > 0);
    assert!(a.peak_rss_final_kb >= a.peak_rss_plan_kb);
    assert!(b.peak_rss_final_kb >= b.peak_rss_plan_kb);

    // Zero iterations is a protocol violation, not a free pass.
    assert!(run_child_workload(&subject, Arm::A, 0, &plan_a).is_err());
}

#[test]
fn plan_file_roundtrip_preserves_absence_and_emptiness() {
    use cjc_mir::optimize::PassPlan;

    // Present-empty ("run nothing") and absent ("run the default
    // sequence") are DIFFERENT plan semantics — the roundtrip must
    // preserve the distinction.
    let mut plan = PassPlan::empty();
    plan.per_function.insert("work".to_string(), vec![]);
    plan.per_function.insert(
        "churn".to_string(),
        vec!["dce".to_string(), "constant_fold".to_string()],
    );
    let parsed = parse_plan(&serialize_plan(&plan)).unwrap();
    assert_eq!(parsed.per_function, plan.per_function);
    assert!(!parsed.per_function.contains_key("absent_fn"));

    // Truly empty plan (no entries at all) also survives.
    let empty = PassPlan::empty();
    assert_eq!(
        parse_plan(&serialize_plan(&empty)).unwrap().per_function,
        empty.per_function
    );
}

#[test]
fn plan_file_rejects_malformed_input() {
    assert!(parse_plan("").is_none());
    assert!(parse_plan("WRONG_HEADER\nf\tdce").is_none());
    assert!(parse_plan("CANA_PLAN_V1\nno_tab_separator").is_none());
    assert!(parse_plan("CANA_PLAN_V1\n\tdce").is_none(), "empty fn name");
}

#[test]
fn nonsynthetic_subject_compiles_and_is_excluded_from_corpus_gates() {
    let subject = subjects()
        .into_iter()
        .find(|s| s.family == "nonsynthetic")
        .unwrap();
    assert!(!subject.corpus_verified);
    // The example must at least lower end-to-end (its execution is
    // covered by gate 3 during the real run).
    compile_subject(&subject.source).expect("example_08 must compile");
}

// =============================================================================
// Unit — stats, calibration, verdicts, protocol
// =============================================================================

#[test]
fn median_odd_even_and_empty() {
    assert_eq!(median(&[]), None);
    assert_eq!(median(&[3.0]), Some(3.0));
    assert_eq!(median(&[5.0, 1.0, 3.0]), Some(3.0));
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
}

#[test]
fn band_orders_min_med_max() {
    let b = band(&[5.0, 1.0, 3.0]).unwrap();
    assert_eq!((b.min, b.med, b.max), (1.0, 3.0, 5.0));
    assert!(band(&[]).is_none());
}

#[test]
fn ratio_band_is_conservative() {
    let a = Band {
        min: 90.0,
        med: 100.0,
        max: 110.0,
    };
    let b = Band {
        min: 45.0,
        med: 50.0,
        max: 55.0,
    };
    let (lo, med, hi) = ratio_band(&a, &b);
    assert!((med - 0.5).abs() < 1e-12);
    assert!((lo - 45.0 / 110.0).abs() < 1e-12);
    assert!((hi - 55.0 / 90.0).abs() < 1e-12);
    assert!(lo <= med && med <= hi);
}

#[test]
fn verdict_requires_whole_band_clear_of_one() {
    assert_eq!(verdict(0.4, 0.6), Verdict::Win);
    assert_eq!(verdict(1.1, 1.3), Verdict::Regression);
    assert_eq!(verdict(0.9, 1.1), Verdict::Inconclusive);
    assert_eq!(verdict(0.9, 1.0), Verdict::Inconclusive); // hi == 1.0 is NOT a win
}

#[test]
fn calibrate_iters_clamps_both_ends() {
    // Slow program: one run exceeds the target -> 1 iteration.
    assert_eq!(calibrate_iters(10_000_000, 5_000_000, 200_000), 1);
    // Fast program: capped at max_iters.
    assert_eq!(calibrate_iters(1, 5_000_000, 200_000), 200_000);
    // Mid: straightforward division.
    assert_eq!(calibrate_iters(1_000, 5_000_000, 200_000), 5_000);
    // Degenerate zero-duration run must not divide by zero.
    assert_eq!(calibrate_iters(0, 5_000_000, 200_000), 200_000);
}

#[test]
fn child_line_roundtrip() {
    let m = ChildMeasurement {
        iters: 42,
        wall_micros: 5_000_123,
        peak_rss_plan_kb: 18_432,
        peak_rss_final_kb: 20_480,
        output_fnv: 0xdead_beef_cafe_f00d,
    };
    let line = format_child_line("mem_grad_a3", Arm::B, &m);
    let parsed = parse_child_line(&line).expect("own format must parse");
    assert_eq!(parsed.subject, "mem_grad_a3");
    assert_eq!(parsed.arm, Arm::B);
    assert_eq!(parsed.measurement, m);
}

#[test]
fn child_line_rejects_malformed_input() {
    assert!(parse_child_line("").is_none());
    assert!(parse_child_line("unrelated output").is_none());
    // Wrong prefix.
    assert!(parse_child_line("CANA_DIAG_CHILD_V0 subject=x arm=a iters=1").is_none());
    // Missing keys.
    assert!(parse_child_line("CANA_DIAG_CHILD_V1 subject=x arm=a").is_none());
    // Bad arm / non-numeric / non-hex values.
    let good = format_child_line(
        "s",
        Arm::A,
        &ChildMeasurement {
            iters: 1,
            wall_micros: 1,
            peak_rss_plan_kb: 1,
            peak_rss_final_kb: 1,
            output_fnv: 1,
        },
    );
    assert!(parse_child_line(&good.replace("arm=a", "arm=c")).is_none());
    assert!(parse_child_line(&good.replace("iters=1", "iters=one")).is_none());
    assert!(parse_child_line(&good.replace("output_fnv=", "output_fnv=zz")).is_none());
}

#[test]
fn output_fnv_separates_line_boundaries() {
    // Length-prefixing means ["ab","c"] and ["a","bc"] must not alias.
    let one = output_fnv(&["ab".to_string(), "c".to_string()]);
    let two = output_fnv(&["a".to_string(), "bc".to_string()]);
    assert_ne!(one, two);
    assert_ne!(output_fnv(&[]), output_fnv(&[String::new()]));
}

// =============================================================================
// Proptest — stats + protocol properties
// =============================================================================

mod props {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn median_lies_within_minmax(xs in proptest::collection::vec(-1e12f64..1e12, 1..64)) {
            let m = median(&xs).unwrap();
            let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            prop_assert!(m >= lo && m <= hi);
        }

        #[test]
        fn band_is_ordered(xs in proptest::collection::vec(-1e12f64..1e12, 1..64)) {
            let b = band(&xs).unwrap();
            prop_assert!(b.min <= b.med && b.med <= b.max);
        }

        #[test]
        fn calibration_stays_in_bounds(
            single in 0u64..u64::MAX / 2,
            target in 1u64..100_000_000,
            max_iters in 1u64..1_000_000,
        ) {
            let iters = calibrate_iters(single, target, max_iters);
            prop_assert!(iters >= 1 && iters <= max_iters);
        }

        #[test]
        fn plan_roundtrips_any_wellformed_plan(
            entries in proptest::collection::btree_map(
                "[A-Za-z_][A-Za-z0-9_]{0,24}",
                proptest::collection::vec("[a-z_]{1,16}", 0..6),
                0..8,
            ),
        ) {
            let mut plan = cjc_mir::optimize::PassPlan::empty();
            for (f, p) in &entries {
                plan.per_function.insert(f.clone(), p.clone());
            }
            let parsed = parse_plan(&serialize_plan(&plan)).unwrap();
            prop_assert_eq!(parsed.per_function, plan.per_function);
        }

        #[test]
        fn protocol_roundtrips_any_measurement(
            iters in 1u64..u64::MAX,
            wall in 0u64..u64::MAX,
            rss_plan in 0u64..u64::MAX,
            rss_final in 0u64..u64::MAX,
            fnv in 0u64..u64::MAX,
            arm_b in proptest::bool::ANY,
            name in "[a-z0-9_]{1,32}",
        ) {
            let m = ChildMeasurement {
                iters,
                wall_micros: wall,
                peak_rss_plan_kb: rss_plan,
                peak_rss_final_kb: rss_final,
                output_fnv: fnv,
            };
            let arm = if arm_b { Arm::B } else { Arm::A };
            let parsed = parse_child_line(&format_child_line(&name, arm, &m)).unwrap();
            prop_assert_eq!(parsed.subject, name);
            prop_assert_eq!(parsed.arm, arm);
            prop_assert_eq!(parsed.measurement, m);
        }
    }
}

// =============================================================================
// Bolero — structural fuzz: the parser never panics
// =============================================================================

#[test]
fn fuzz_parse_child_line_never_panics() {
    bolero::check!().with_type::<String>().for_each(|s| {
        // Any byte soup must produce Some/None, never a panic.
        let _ = parse_child_line(s);
    });
}

#[test]
fn fuzz_parse_plan_never_panics() {
    bolero::check!().with_type::<String>().for_each(|s| {
        let _ = parse_plan(s);
    });
}

#[test]
fn fuzz_stats_never_panic() {
    bolero::check!().with_type::<Vec<f64>>().for_each(|xs| {
        // NaN/inf included: median's total_cmp ordering must hold.
        let _ = median(xs);
        let _ = band(xs);
    });
}
