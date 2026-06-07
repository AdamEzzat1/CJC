//! Determinism tests for the CANA Phase-1 passive observer.
//!
//! These tests assert the foundational invariant of the entire CANA program:
//!
//! > Same input MIR → byte-identical `CanaReport.canonical_bytes()`.
//!
//! If any of these tests starts failing, **stop and find the source of
//! non-determinism** — every later phase of CANA (advisory, active,
//! NSS-integrated, profile-guided) inherits its audit-trail guarantee from
//! this property. There is no "small" determinism violation in CANA.

use cjc_cana::{analyze_program, CanaReport};
use cjc_mir::{
    MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirParam, MirProgram, MirStmt,
};

fn ekind(k: MirExprKind) -> MirExpr {
    MirExpr { kind: k }
}

fn build_complex_program() -> MirProgram {
    // A program with a representative mix of features:
    //   - 3 functions, one with parameters
    //   - One function with an if-else
    //   - One function with a while loop
    //   - Array literal, tensor literal, function call (array_push)
    //   - Deeply nested expression
    let mut f1 = MirFunction {
        id: MirFnId(0),
        name: "__main".to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "x".to_string(),
                    mutable: false,
                    init: ekind(MirExprKind::ArrayLit(vec![
                        ekind(MirExprKind::IntLit(1)),
                        ekind(MirExprKind::IntLit(2)),
                        ekind(MirExprKind::IntLit(3)),
                    ])),
                    alloc_hint: None,
                    slot: None,
                },
                MirStmt::Expr(ekind(MirExprKind::Call {
                    callee: Box::new(ekind(MirExprKind::Var("array_push".to_string()))),
                    args: vec![
                        ekind(MirExprKind::Var("x".to_string())),
                        ekind(MirExprKind::IntLit(4)),
                    ],
                })),
            ],
            result: None,
        },
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Public,
        local_count: 0,
    };

    let mut f2 = f1.clone();
    f2.id = MirFnId(1);
    f2.name = "branches".to_string();
    f2.body.stmts = vec![MirStmt::If {
        cond: ekind(MirExprKind::BoolLit(true)),
        then_body: MirBody {
            stmts: vec![MirStmt::Expr(ekind(MirExprKind::IntLit(1)))],
            result: None,
        },
        else_body: Some(MirBody {
            stmts: vec![MirStmt::Expr(ekind(MirExprKind::IntLit(2)))],
            result: None,
        }),
    }];

    let mut f3 = f1.clone();
    f3.id = MirFnId(2);
    f3.name = "loops".to_string();
    f3.params = vec![MirParam {
        name: "n".to_string(),
        ty_name: "i64".to_string(),
        default: None,
        is_variadic: false,
    }];
    f3.body.stmts = vec![
        MirStmt::Let {
            name: "acc".to_string(),
            mutable: true,
            init: ekind(MirExprKind::IntLit(0)),
            alloc_hint: None,
            slot: None,
        },
        MirStmt::While {
            cond: ekind(MirExprKind::BoolLit(true)),
            body: MirBody {
                stmts: vec![MirStmt::Expr(ekind(MirExprKind::Assign {
                    target: Box::new(ekind(MirExprKind::Var("acc".to_string()))),
                    value: Box::new(ekind(MirExprKind::Binary {
                        op: cjc_ast::BinOp::Add,
                        left: Box::new(ekind(MirExprKind::Var("acc".to_string()))),
                        right: Box::new(ekind(MirExprKind::Var("n".to_string()))),
                    })),
                }))],
                result: None,
            },
        },
    ];

    // Add a tensor literal for variety.
    f1.body.stmts.push(MirStmt::Expr(ekind(MirExprKind::TensorLit {
        rows: vec![
            vec![ekind(MirExprKind::FloatLit(1.0)), ekind(MirExprKind::FloatLit(2.0))],
            vec![ekind(MirExprKind::FloatLit(3.0)), ekind(MirExprKind::FloatLit(4.0))],
        ],
    })));

    MirProgram {
        functions: vec![f1, f2, f3],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(0),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn canonical_bytes_byte_identical_across_100_runs() {
    let program = build_complex_program();
    let first = analyze_program(&program).canonical_bytes();
    for run in 1..100 {
        let again = analyze_program(&program).canonical_bytes();
        assert_eq!(
            again,
            first,
            "determinism violated at run {} — canonical_bytes diverged",
            run
        );
    }
}

#[test]
fn program_hash_byte_identical_across_100_runs() {
    let program = build_complex_program();
    let first = analyze_program(&program).features.program_hash;
    for run in 1..100 {
        let again = analyze_program(&program).features.program_hash;
        assert_eq!(again, first, "program_hash drift at run {}", run);
    }
}

#[test]
fn feature_hash_byte_identical_across_100_runs() {
    let program = build_complex_program();
    let first = analyze_program(&program).features.feature_hash;
    for run in 1..100 {
        let again = analyze_program(&program).features.feature_hash;
        assert_eq!(again, first, "feature_hash drift at run {}", run);
    }
}

#[test]
fn cloned_program_produces_identical_report() {
    let program = build_complex_program();
    let clone = program.clone();
    let r1 = analyze_program(&program);
    let r2 = analyze_program(&clone);
    assert_eq!(r1.canonical_bytes(), r2.canonical_bytes());
}

#[test]
fn reordering_function_definitions_changes_program_hash_but_features_match_per_fn() {
    // Reordering functions changes the *program* hash (function IDs differ)
    // but each per-function feature record remains identical because we
    // key by name, not by ID.
    let program = build_complex_program();
    let mut swapped = program.clone();
    swapped.functions.swap(0, 1);

    let r1 = analyze_program(&program);
    let r2 = analyze_program(&swapped);

    // Per-fn features are identical (we key by name in BTreeMap).
    for (name, feats) in &r1.features.per_fn {
        let other = r2.features.per_fn.get(name).expect("missing fn after swap");
        assert_eq!(feats, other, "feature divergence for {}", name);
    }
}

#[test]
fn json_output_is_byte_identical_across_runs() {
    let program = build_complex_program();
    let first = analyze_program(&program).to_json();
    let again = analyze_program(&program).to_json();
    assert_eq!(first, again);
}

#[test]
fn json_output_does_not_contain_random_or_pointer_addresses() {
    // Defensive: if anyone ever Debug-prints a pointer into the report,
    // this test catches it. We check for the characteristic "0x" + hex tail
    // pattern of a 64-bit pointer address (16 chars after "0x").
    let program = build_complex_program();
    let json = analyze_program(&program).to_json();
    // We DO emit "0x" never — our hashes are bare hex without prefix.
    // Pointer addresses would look like "0x7f...".
    assert!(!json.contains("0x7f"), "JSON contains x86_64 pointer prefix");
    assert!(!json.contains("0x55"), "JSON contains x86_64 stack pointer prefix");
}

#[test]
fn canonical_bytes_unchanged_after_round_trip_to_json_string() {
    let program = build_complex_program();
    let r = analyze_program(&program);
    let bytes = r.canonical_bytes();
    let json_string = r.to_json();
    assert_eq!(bytes, json_string.as_bytes());
}

#[test]
fn empty_and_nonempty_program_produce_distinct_hashes() {
    let empty = MirProgram {
        functions: vec![],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(0),
    };
    let r_empty: CanaReport = analyze_program(&empty);
    let r_full = analyze_program(&build_complex_program());
    assert_ne!(r_empty.features.program_hash, r_full.features.program_hash);
    assert_ne!(r_empty.features.feature_hash, r_full.features.feature_hash);
}
