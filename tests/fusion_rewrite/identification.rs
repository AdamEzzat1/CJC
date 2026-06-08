//! CANA identification × rewriter integration.
//!
//! With `norm` now in `NATIVE_PRIMITIVES`, the Phase 3 fusion identifier
//! recognises `matmul → norm` as a fusion candidate. The rewriter consumes
//! those candidates. These tests verify the two phases agree on which
//! chains are present.

use cjc_cana::fusion::identify_fusion_candidates;
use cjc_mir::fusion_rewrite::fusion_rewrite_program;

fn parse_and_lower(src: &str) -> cjc_mir::MirProgram {
    let (ast, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(&ast);
    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mut mir = hir_to_mir.lower_program(&hir);
    cjc_mir::escape::annotate_program(&mut mir);
    mir
}

#[test]
fn cana_identifies_matmul_norm_chain() {
    let src = r#"
fn main() {
    let a = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let w = Tensor.from_vec([3.0, 0.0, 0.0, 4.0], [2, 2]);
    let h = matmul(a, w);
    let n = norm(h);
    print(n);
}
"#;
    let mir = parse_and_lower(src);
    let plan = identify_fusion_candidates(&mir);

    // The fusion identifier should find at least one fusion-worthy chain
    // of length ≥ 2 (matmul → norm). It may find additional length-1
    // "chains" for other native primitives; we only assert on the worthy ones.
    let worthy: Vec<_> = plan.fusion_worthy().collect();
    assert!(
        !worthy.is_empty(),
        "expected at least one fusion-worthy chain after Phase 3.5c flipped `norm` into NATIVE_PRIMITIVES; got plan = {:?}",
        plan,
    );

    // The chain should include matmul and norm in that order.
    let contains_matmul_norm = worthy.iter().any(|c| {
        let names: Vec<&str> = c.chain.iter().map(|e| e.primitive_name.as_str()).collect();
        names.windows(2).any(|w| w == ["matmul", "norm"])
    });
    assert!(
        contains_matmul_norm,
        "expected a [matmul, norm] subchain; got chains: {:?}",
        worthy.iter().map(|c| &c.chain).collect::<Vec<_>>()
    );
}

#[test]
fn rewriter_and_identifier_agree_on_count() {
    // The number of length-≥-2 chains identified should equal the number
    // of rewrites the rewriter applies — assuming all chains are
    // matmul→norm. This is the simplest version of the agreement contract;
    // future primitive pairs will expand it.
    let src = r#"
fn main() {
    let a1 = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let w1 = Tensor.from_vec([3.0, 0.0, 0.0, 4.0], [2, 2]);
    let h1 = matmul(a1, w1);
    let n1 = norm(h1);
    print(n1);

    let a2 = Tensor.from_vec([2.0, 0.0, 0.0, 2.0], [2, 2]);
    let w2 = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let h2 = matmul(a2, w2);
    let n2 = norm(h2);
    print(n2);
}
"#;
    let mir_for_id = parse_and_lower(src);
    let plan = identify_fusion_candidates(&mir_for_id);
    let id_count = plan.fusion_worthy().count();

    let mut mir_for_rewrite = parse_and_lower(src);
    let r = fusion_rewrite_program(&mut mir_for_rewrite);

    assert_eq!(id_count, 2, "identifier should see 2 fusion-worthy chains");
    assert_eq!(r.rewrites_applied, 2, "rewriter should apply 2 rewrites");
    assert_eq!(
        id_count, r.rewrites_applied,
        "identifier and rewriter must agree on the count"
    );
}
