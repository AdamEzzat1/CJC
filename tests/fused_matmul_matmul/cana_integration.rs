//! CANA integration — fused_matmul_matmul is recognised as a native primitive.

use cjc_cana::fusion::{is_native_primitive, NATIVE_PRIMITIVES};

#[test]
fn fused_matmul_matmul_is_in_native_primitives() {
    assert!(NATIVE_PRIMITIVES.contains(&"fused_matmul_matmul"));
}

#[test]
fn fused_matmul_matmul_is_recognised_by_lookup() {
    assert!(is_native_primitive("fused_matmul_matmul"));
}

#[test]
fn matmul_chain_is_identified_as_fusion_candidate() {
    use cjc_cana::fusion::identify_fusion_candidates;
    use cjc_mir::{
        MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt,
    };

    let var = |n: &str| MirExpr { kind: MirExprKind::Var(n.to_string()) };
    let call = |callee: &str, args: Vec<MirExpr>| MirExpr {
        kind: MirExprKind::Call {
            callee: Box::new(var(callee)),
            args,
        },
    };
    let let_stmt = |name: &str, init: MirExpr| MirStmt::Let {
        name: name.to_string(),
        mutable: false,
        init,
        alloc_hint: None,
        slot: None,
    };

    let mut f = MirFunction {
        id: MirFnId(0),
        name: "main".to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: vec![
                let_stmt("h", call("matmul", vec![var("a"), var("b")])),
                let_stmt("r", call("matmul", vec![var("h"), var("c")])),
            ],
            result: None,
        },
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Public,
        local_count: 0,
    };
    f.body.stmts = f.body.stmts.clone(); // no-op, keeps clippy quiet
    let prog = MirProgram {
        functions: vec![f],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(0),
    };
    let plan = identify_fusion_candidates(&prog);
    let worthy: Vec<_> = plan.fusion_worthy().collect();
    assert!(
        !worthy.is_empty(),
        "matmul → matmul chain should be identified as fusion-worthy"
    );
    let contains = worthy.iter().any(|c| {
        let names: Vec<&str> = c.chain.iter().map(|e| e.primitive_name.as_str()).collect();
        names.windows(2).any(|w| w == ["matmul", "matmul"])
    });
    assert!(contains, "expected a [matmul, matmul] subchain");
}
