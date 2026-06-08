//! Property tests for the CANA Phase-1 passive observer.
//!
//! These tests use proptest to assert *laws* CANA promises:
//!
//! 1. **Determinism:** `analyze(p)` is referentially transparent — every
//!    call returns the same `ProgramHash`, `FeatureHash`, and canonical bytes.
//! 2. **Stability under structurally trivial transforms:** Cloning a program
//!    must not change any feature.
//! 3. **Legality-gate totality:** the gate never panics on arbitrary
//!    pass-sequence inputs, regardless of how malformed.
//! 4. **Monotonicity:** adding statements never decreases `block_count` or
//!    `expr_count` (Phase 1 features grow monotonically with program size).
//! 5. **Float invariance under repeated extraction:** programs containing
//!    `FloatLit` (including NaN, ±0, subnormals) hash identically across runs.

use cjc_cana::{
    analyze_program, DefaultLegalityGate, LegalityGate, PassSequence, ProposedPass,
};
use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirParam, MirProgram, MirStmt};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ekind(k: MirExprKind) -> MirExpr {
    MirExpr { kind: k }
}

fn make_function(name: String, n_int_stmts: usize) -> MirFunction {
    let mut body = MirBody {
        stmts: Vec::with_capacity(n_int_stmts),
        result: None,
    };
    for i in 0..n_int_stmts {
        body.stmts
            .push(MirStmt::Expr(ekind(MirExprKind::IntLit(i as i64))));
    }
    MirFunction {
        id: MirFnId(0),
        name,
        type_params: vec![],
        params: vec![],
        return_type: None,
        body,
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Public,
        local_count: 0,
    }
}

fn make_program(fn_specs: Vec<(String, usize)>) -> MirProgram {
    let functions: Vec<MirFunction> = fn_specs
        .into_iter()
        .enumerate()
        .map(|(i, (name, n))| {
            let mut f = make_function(name, n);
            f.id = MirFnId(i as u32);
            f
        })
        .collect();
    let entry = functions.first().map(|f| f.id).unwrap_or(MirFnId(0));
    MirProgram {
        functions,
        struct_defs: vec![],
        enum_defs: vec![],
        entry,
    }
}

// ---------------------------------------------------------------------------
// Proptest generators
// ---------------------------------------------------------------------------

prop_compose! {
    fn arb_fn_name()(
        prefix in "[a-z][a-z0-9_]{0,8}",
    ) -> String {
        prefix
    }
}

prop_compose! {
    fn arb_program()(
        fns in proptest::collection::vec((arb_fn_name(), 0usize..16), 1..6),
    ) -> MirProgram {
        // Deduplicate function names so we don't generate ambiguous programs.
        let mut seen = std::collections::BTreeSet::new();
        let dedup: Vec<_> = fns.into_iter()
            .filter(|(name, _)| seen.insert(name.clone()))
            .collect();
        if dedup.is_empty() {
            make_program(vec![("__main".to_string(), 0)])
        } else {
            make_program(dedup)
        }
    }
}

prop_compose! {
    fn arb_pass_sequence()(
        entries in proptest::collection::vec(
            (arb_fn_name(), "[a-z_]{1,12}", 0u8..2),
            0..8,
        ),
    ) -> PassSequence {
        let mut seq = PassSequence::empty();
        for (fname, pass_name, action) in entries {
            let pass = if action == 0 {
                ProposedPass::Run(pass_name)
            } else {
                ProposedPass::Skip(pass_name)
            };
            seq.per_function.entry(fname).or_default().push(pass);
        }
        seq
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Law 1: analyze(p) is referentially transparent.
    #[test]
    fn analyze_is_deterministic(program in arb_program()) {
        let r1 = analyze_program(&program);
        let r2 = analyze_program(&program);
        prop_assert_eq!(r1.features.program_hash, r2.features.program_hash);
        prop_assert_eq!(r1.features.feature_hash, r2.features.feature_hash);
        prop_assert_eq!(r1.canonical_bytes(), r2.canonical_bytes());
    }

    /// Law 2: cloning preserves every feature.
    #[test]
    fn cloning_program_preserves_features(program in arb_program()) {
        let clone = program.clone();
        let r1 = analyze_program(&program);
        let r2 = analyze_program(&clone);
        prop_assert_eq!(r1.features.program_hash, r2.features.program_hash);
        prop_assert_eq!(r1.canonical_bytes(), r2.canonical_bytes());
    }

    /// Law 3: legality gate is total — never panics on any input.
    /// We don't assert what verdict it returns; we just assert it returns
    /// one (i.e., didn't panic).
    #[test]
    fn legality_gate_is_total(
        program in arb_program(),
        sequence in arb_pass_sequence(),
    ) {
        let features = analyze_program(&program).features;
        let gate = DefaultLegalityGate::new();
        let _verdict = gate.verify(&program, &sequence, &features);
    }

    /// Law 4 (a): adding a statement increases expr_count monotonically.
    #[test]
    fn adding_statement_increases_expr_count(
        n_before in 0usize..16,
        added in 1usize..16,
    ) {
        let p_before = make_program(vec![("f".to_string(), n_before)]);
        let p_after = make_program(vec![("f".to_string(), n_before + added)]);
        let r_before = analyze_program(&p_before);
        let r_after = analyze_program(&p_after);
        let ec_before = r_before.features.per_fn.get("f").map(|f| f.memory.expr_count).unwrap_or(0);
        let ec_after = r_after.features.per_fn.get("f").map(|f| f.memory.expr_count).unwrap_or(0);
        prop_assert!(ec_after >= ec_before);
    }

    /// Law 4 (b): block_count never decreases when adding statements.
    #[test]
    fn block_count_monotone_in_statements(
        n_before in 0usize..16,
        added in 1usize..16,
    ) {
        let p_before = make_program(vec![("f".to_string(), n_before)]);
        let p_after = make_program(vec![("f".to_string(), n_before + added)]);
        let r_before = analyze_program(&p_before);
        let r_after = analyze_program(&p_after);
        let bc_before = r_before.features.per_fn.get("f").map(|f| f.cfg.block_count).unwrap_or(0);
        let bc_after = r_after.features.per_fn.get("f").map(|f| f.cfg.block_count).unwrap_or(0);
        prop_assert!(bc_after >= bc_before);
    }

    /// Law 5: programs containing arbitrary float literals (including weird
    /// IEEE-754 values) still hash deterministically across repeated extraction.
    #[test]
    fn float_lit_does_not_break_determinism(
        floats in proptest::collection::vec(any::<f64>(), 0..8),
    ) {
        let mut f = make_function("f".to_string(), 0);
        for v in floats {
            f.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::FloatLit(v))));
        }
        let p = MirProgram {
            functions: vec![f],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        };
        let r1 = analyze_program(&p);
        let r2 = analyze_program(&p);
        prop_assert_eq!(r1.features.program_hash, r2.features.program_hash);
        prop_assert_eq!(r1.features.feature_hash, r2.features.feature_hash);
    }
}

// ---------------------------------------------------------------------------
// Heavy-hitter property: structurally distinct programs have distinct hashes
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Distinct function-name sets produce distinct ProgramHashes.
    /// (FeatureHash may collide in some edge cases — e.g., both programs
    /// produce zero features — so we only assert on ProgramHash here.)
    #[test]
    fn distinct_function_names_produce_distinct_program_hashes(
        names_a in proptest::collection::vec(arb_fn_name(), 1..4),
        names_b in proptest::collection::vec(arb_fn_name(), 1..4),
    ) {
        let set_a: std::collections::BTreeSet<_> = names_a.iter().cloned().collect();
        let set_b: std::collections::BTreeSet<_> = names_b.iter().cloned().collect();
        if set_a == set_b {
            return Ok(()); // Not a distinguishing case; skip.
        }
        let p_a = make_program(set_a.iter().map(|n| (n.clone(), 0)).collect());
        let p_b = make_program(set_b.iter().map(|n| (n.clone(), 0)).collect());
        let r_a = analyze_program(&p_a);
        let r_b = analyze_program(&p_b);
        prop_assert_ne!(r_a.features.program_hash, r_b.features.program_hash);
    }
}
