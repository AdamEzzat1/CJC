// Milestone 2.6 — MIR Monomorphization Pass Tests
//
// Tests for the monomorphization pass that specializes generic functions
// with concrete type arguments. Follows the pattern from the monomorph
// module's own internal tests but exercises the public API.
//
// Covers:
//   - No-op pass on non-generic programs
//   - Generic function specialization with int arg
//   - Mangled name format: fn__M__type
//   - Multiple instantiations of the same generic
//   - Budget reporting
//   - Variant lit arguments

use cjc_mir::monomorph::*;
use cjc_mir::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mk_var(name: &str) -> MirExpr {
    MirExpr {
        kind: MirExprKind::Var(name.to_string()),
    }
}

fn mk_int(v: i64) -> MirExpr {
    MirExpr {
        kind: MirExprKind::IntLit(v),
    }
}

fn mk_float(v: f64) -> MirExpr {
    MirExpr {
        kind: MirExprKind::FloatLit(v),
    }
}

fn mk_bool(v: bool) -> MirExpr {
    MirExpr {
        kind: MirExprKind::BoolLit(v),
    }
}

fn mk_string(s: &str) -> MirExpr {
    MirExpr {
        kind: MirExprKind::StringLit(s.to_string()),
    }
}

fn mk_call(callee: &str, args: Vec<MirExpr>) -> MirExpr {
    MirExpr {
        kind: MirExprKind::Call {
            callee: Box::new(mk_var(callee)),
            args,
        },
    }
}

fn make_generic_id_fn(id: u32) -> MirFunction {
    MirFunction {
        id: MirFnId(id),
        name: "id".to_string(),
        type_params: vec![("T".to_string(), vec![])],
        params: vec![MirParam {
            name: "x".to_string(),
            ty_name: "T".to_string(),
            default: None,
            is_variadic: false,
        }],
        return_type: Some("T".to_string()),
        body: MirBody {
            stmts: vec![],
            result: Some(Box::new(mk_var("x"))),
        },
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Private,
    }
}

fn make_main_fn(id: u32, body_stmts: Vec<MirStmt>) -> MirFunction {
    MirFunction {
        id: MirFnId(id),
        name: "__main".to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: body_stmts,
            result: None,
        },
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Private,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn mono_noop_on_non_generic_program() {
    // A program with no generic functions should pass through unchanged.
    let program = MirProgram {
        functions: vec![MirFunction {
            id: MirFnId(0),
            name: "__main".to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: MirBody {
                stmts: vec![MirStmt::Expr(mk_int(42))],
                result: None,
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        }],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(0),
    };

    let (new_prog, report) = monomorphize_program(&program);
    assert_eq!(report.specialization_count, 0);
    assert!(!report.budget_exceeded);
    assert_eq!(new_prog.functions.len(), 1);
    assert_eq!(new_prog.functions[0].name, "__main");
}

#[test]
fn mono_generic_with_int_arg_creates_specialization() {
    // fn id<T>(x: T) -> T { x }
    // __main: id(42)
    let id_fn = make_generic_id_fn(0);
    let main_fn = make_main_fn(1, vec![MirStmt::Expr(mk_call("id", vec![mk_int(42)]))]);

    let program = MirProgram {
        functions: vec![id_fn, main_fn],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(1),
    };

    let (new_prog, report) = monomorphize_program(&program);
    assert_eq!(report.specialization_count, 1);
    assert!(!report.budget_exceeded);

    // Should have 3 functions: original id, __main, id__M__i64
    assert_eq!(new_prog.functions.len(), 3);

    let specialized = new_prog
        .functions
        .iter()
        .find(|f| f.name == "id__M__i64");
    assert!(specialized.is_some(), "expected id__M__i64 specialization");

    let spec = specialized.unwrap();
    assert!(spec.type_params.is_empty(), "specialized fn should have no type params");
    assert_eq!(spec.params[0].ty_name, "i64");
}

#[test]
fn mono_mangled_name_format() {
    // fn id<T>(x: T) -> T { x }
    // __main: id(3.14)
    let id_fn = make_generic_id_fn(0);
    let main_fn = make_main_fn(1, vec![MirStmt::Expr(mk_call("id", vec![mk_float(3.14)]))]);

    let program = MirProgram {
        functions: vec![id_fn, main_fn],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(1),
    };

    let (new_prog, _report) = monomorphize_program(&program);

    let spec = new_prog
        .functions
        .iter()
        .find(|f| f.name.starts_with("id__M__"));
    assert!(spec.is_some(), "expected mangled specialization");
    assert_eq!(
        spec.unwrap().name, "id__M__f64",
        "mangled name for f64 should be id__M__f64"
    );
}

#[test]
fn mono_multiple_instantiations_of_same_generic() {
    // fn id<T>(x: T) -> T { x }
    // __main: id(42); id(true); id("hello")
    let id_fn = make_generic_id_fn(0);
    let main_fn = make_main_fn(
        1,
        vec![
            MirStmt::Expr(mk_call("id", vec![mk_int(42)])),
            MirStmt::Expr(mk_call("id", vec![mk_bool(true)])),
            MirStmt::Expr(mk_call("id", vec![mk_string("hello")])),
        ],
    );

    let program = MirProgram {
        functions: vec![id_fn, main_fn],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(1),
    };

    let (new_prog, report) = monomorphize_program(&program);
    assert_eq!(report.specialization_count, 3);
    assert!(!report.budget_exceeded);

    // 2 original + 3 specialized = 5
    assert_eq!(new_prog.functions.len(), 5);

    let names: Vec<&str> = new_prog.functions.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"id__M__i64"), "missing id__M__i64");
    assert!(names.contains(&"id__M__bool"), "missing id__M__bool");
    assert!(names.contains(&"id__M__String"), "missing id__M__String");
}

#[test]
fn mono_budget_not_exceeded_for_small_programs() {
    let id_fn = make_generic_id_fn(0);
    let main_fn = make_main_fn(1, vec![MirStmt::Expr(mk_call("id", vec![mk_int(1)]))]);

    let program = MirProgram {
        functions: vec![id_fn, main_fn],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(1),
    };

    let (_new_prog, report) = monomorphize_program(&program);
    assert!(!report.budget_exceeded, "budget should not be exceeded for 1 instantiation");
    assert_eq!(report.specialization_count, 1);
}

#[test]
fn mono_report_top_fanout() {
    // Two different generic functions, each called once
    let id_fn = make_generic_id_fn(0);
    let wrap_fn = MirFunction {
        id: MirFnId(2),
        name: "wrap".to_string(),
        type_params: vec![("U".to_string(), vec![])],
        params: vec![MirParam {
            name: "v".to_string(),
            ty_name: "U".to_string(),
            default: None,
            is_variadic: false,
        }],
        return_type: Some("U".to_string()),
        body: MirBody {
            stmts: vec![],
            result: Some(Box::new(mk_var("v"))),
        },
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Private,
    };

    let main_fn = make_main_fn(
        1,
        vec![
            MirStmt::Expr(mk_call("id", vec![mk_int(1)])),
            MirStmt::Expr(mk_call("id", vec![mk_float(2.0)])),
            MirStmt::Expr(mk_call("wrap", vec![mk_bool(true)])),
        ],
    );

    let program = MirProgram {
        functions: vec![id_fn, wrap_fn, main_fn],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(1),
    };

    let (_new_prog, report) = monomorphize_program(&program);
    assert_eq!(report.specialization_count, 3);
    assert!(!report.top_fanout.is_empty(), "top_fanout should be populated");

    // "id" should have higher fanout (2 calls) than "wrap" (1 call)
    let id_fanout = report
        .top_fanout
        .iter()
        .find(|(name, _)| name == "id")
        .map(|(_, count)| *count);
    assert!(
        id_fanout.is_some(),
        "id should appear in top_fanout"
    );
    assert!(
        id_fanout.unwrap() >= 2,
        "id fanout should be >= 2, got {:?}",
        id_fanout
    );
}
