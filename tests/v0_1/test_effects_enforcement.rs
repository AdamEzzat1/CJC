//! v0.1 Contract Tests: Effects Enforcement
//!
//! Locks down: pure cannot allocate/IO, alloc allows tensor,
//! unannotated allows anything, broadcast effect classification,
//! NoGC rejects gc_alloc.

// ── Helpers ──────────────────────────────────────────────────────

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    program
}

fn check_effects(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut checker = cjc_types::TypeChecker::new();
    checker.check_program(&program);
    checker.diagnostics.diagnostics.iter()
        .filter(|d| d.code == "E4002")
        .map(|d| d.message.clone())
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────

#[test]
fn pure_cannot_allocate() {
    // A /pure function calling sort (ALLOC) should trigger E4002
    let src = r#"
fn sorted_data(data: Any) -> Any / pure {
    sort(data)
}
"#;
    let errors = check_effects(src);
    assert!(!errors.is_empty(),
        "pure fn + sort (alloc) should produce E4002, got no errors");
}

#[test]
fn pure_cannot_io() {
    // A /pure function calling print (IO) should trigger E4002
    let src = r#"
fn loud_add(a: i64, b: i64) -> i64 / pure {
    print(a);
    a + b
}
"#;
    let errors = check_effects(src);
    assert!(!errors.is_empty(),
        "pure fn + print should produce E4002, got no errors");
}

#[test]
fn alloc_allows_tensor() {
    // A /alloc function calling Tensor.zeros should be fine
    let src = r#"
fn make_tensor() -> Any / alloc {
    Tensor.zeros([3])
}
"#;
    let errors = check_effects(src);
    assert!(errors.is_empty(),
        "alloc fn + Tensor.zeros should be OK, but got E4002: {:?}", errors);
}

#[test]
fn unannotated_allows_anything() {
    // No effect annotation → no E4002 enforcement
    let src = r#"
fn do_everything() -> i64 {
    print("hello");
    let t = Tensor.zeros([3]);
    42
}
"#;
    let errors = check_effects(src);
    assert!(errors.is_empty(),
        "unannotated fn should have no E4002 errors, got: {:?}", errors);
}

#[test]
fn broadcast_has_alloc_effect() {
    // Verify in the effect registry that broadcast → ALLOC
    let registry = cjc_types::effect_registry::builtin_effects();
    let broadcast_effects = registry.get("broadcast")
        .expect("broadcast should be in effect registry");
    assert!(broadcast_effects.has(cjc_types::EffectSet::ALLOC),
        "broadcast should have ALLOC effect");

    let broadcast2_effects = registry.get("broadcast2")
        .expect("broadcast2 should be in effect registry");
    assert!(broadcast2_effects.has(cjc_types::EffectSet::ALLOC),
        "broadcast2 should have ALLOC effect");
}

#[test]
fn nogc_rejects_gc_builtin() {
    // A #[nogc] function that calls gc_alloc should be rejected
    // We test via the MIR-level NoGC verifier
    use cjc_mir::nogc_verify::verify_nogc;
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

    fn mk_expr(kind: MirExprKind) -> MirExpr {
        MirExpr { kind }
    }

    fn mk_call(name: &str, args: Vec<MirExpr>) -> MirExpr {
        mk_expr(MirExprKind::Call {
            callee: Box::new(mk_expr(MirExprKind::Var(name.to_string()))),
            args,
        })
    }

    let program = MirProgram {
        functions: vec![MirFunction {
            id: MirFnId(0),
            name: "bad_nogc".to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: MirBody {
                stmts: vec![MirStmt::Expr(mk_call("gc_alloc", vec![]))],
                result: None,
            },
            is_nogc: true,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        }],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(0),
    };

    let result = verify_nogc(&program);
    assert!(result.is_err(), "gc_alloc in #[nogc] function should fail verification");
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.reason.contains("gc_alloc")),
        "error should mention gc_alloc: {:?}", errors);
}
