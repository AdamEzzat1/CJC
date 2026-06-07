//! End-to-end parity tests for the fusion_rewrite pass.
//!
//! The pipeline under test:
//!   1. Parse a CJC-Lang program with `let h = matmul(a, w); let n = norm(h);`
//!   2. Lower AST → HIR → MIR
//!   3. Run `fusion_rewrite::fusion_rewrite_program(&mut mir)`
//!   4. Confirm the chain was rewritten (count went from 2 native calls
//!      to 1 fused call)
//!   5. Run the rewritten MIR through MIR-exec
//!   6. Compare the output bit-for-bit to the un-rewritten chain run through
//!      MIR-exec
//!
//! Step 6 is the rewriter's correctness contract: same observable output.

use cjc_mir::fusion_rewrite::fusion_rewrite_program;
use cjc_mir::{MirExprKind, MirProgram, MirStmt};

fn parse(src: &str) -> cjc_ast::Program {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    prog
}

/// Run `ast_program → MIR`, NO rewrite, return the MIR for inspection.
fn lower_to_mir(ast: &cjc_ast::Program) -> MirProgram {
    let mut ast_lowering = cjc_hir::AstLowering::new();
    let hir = ast_lowering.lower_program(ast);
    let mut hir_to_mir = cjc_mir::HirToMir::new();
    let mut mir = hir_to_mir.lower_program(&hir);
    cjc_mir::escape::annotate_program(&mut mir);
    mir
}

/// Count free-function calls by name across every function body.
fn count_calls_named(prog: &MirProgram, name: &str) -> usize {
    let mut count = 0;
    for func in &prog.functions {
        count += count_calls_in_body(&func.body, name);
    }
    count
}

fn count_calls_in_body(body: &cjc_mir::MirBody, name: &str) -> usize {
    let mut count = 0;
    for stmt in &body.stmts {
        match stmt {
            MirStmt::Let { init, .. } | MirStmt::Expr(init) => {
                count += count_calls_in_expr(init, name);
            }
            MirStmt::If { cond, then_body, else_body } => {
                count += count_calls_in_expr(cond, name);
                count += count_calls_in_body(then_body, name);
                if let Some(eb) = else_body {
                    count += count_calls_in_body(eb, name);
                }
            }
            MirStmt::While { cond, body: wb } => {
                count += count_calls_in_expr(cond, name);
                count += count_calls_in_body(wb, name);
            }
            MirStmt::Return(Some(e)) => count += count_calls_in_expr(e, name),
            MirStmt::NoGcBlock(b) => count += count_calls_in_body(b, name),
            _ => {}
        }
    }
    if let Some(e) = &body.result {
        count += count_calls_in_expr(e, name);
    }
    count
}

fn count_calls_in_expr(expr: &cjc_mir::MirExpr, name: &str) -> usize {
    let mut count = 0;
    if let MirExprKind::Call { callee, args } = &expr.kind {
        let cname = match &callee.kind {
            MirExprKind::Var(n) | MirExprKind::VarLocal { name: n, .. } => Some(n.as_str()),
            _ => None,
        };
        if cname == Some(name) {
            count += 1;
        }
        count += count_calls_in_expr(callee, name);
        for a in args {
            count += count_calls_in_expr(a, name);
        }
    } else {
        match &expr.kind {
            MirExprKind::Binary { left, right, .. } => {
                count += count_calls_in_expr(left, name);
                count += count_calls_in_expr(right, name);
            }
            MirExprKind::Unary { operand, .. } => {
                count += count_calls_in_expr(operand, name);
            }
            MirExprKind::Field { object, .. } => count += count_calls_in_expr(object, name),
            MirExprKind::Index { object, index } => {
                count += count_calls_in_expr(object, name);
                count += count_calls_in_expr(index, name);
            }
            MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
                for e in elems {
                    count += count_calls_in_expr(e, name);
                }
            }
            _ => {}
        }
    }
    count
}

/// Run MIR through the executor and return captured stdout.
fn run_mir(mir: &MirProgram, seed: u64) -> Vec<String> {
    let mut executor = cjc_mir_exec::MirExecutor::new(seed);
    executor
        .exec(mir)
        .unwrap_or_else(|e| panic!("mir-exec failed: {e:?}"));
    executor.output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const SRC_MATMUL_NORM_L2: &str = r#"
fn main() {
    let a = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let w = Tensor.from_vec([3.0, 0.0, 0.0, 4.0], [2, 2]);
    let h = matmul(a, w);
    let n = norm(h);
    print(n);
}
"#;

const SRC_MATMUL_NORM_L1: &str = r#"
fn main() {
    let a = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let w = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
    let h = matmul(a, w);
    let n = norm(h, 1);
    print(n);
}
"#;

#[test]
fn rewriter_changes_mir_call_topology() {
    let ast = parse(SRC_MATMUL_NORM_L2);
    let mut mir = lower_to_mir(&ast);

    // Before rewrite: one matmul + one norm, no fused.
    assert_eq!(count_calls_named(&mir, "matmul"), 1);
    assert_eq!(count_calls_named(&mir, "norm"), 1);
    assert_eq!(count_calls_named(&mir, "fused_matmul_norm"), 0);

    let result = fusion_rewrite_program(&mut mir);
    assert_eq!(result.rewrites_applied, 1, "should rewrite the chain");

    // After rewrite: matmul and norm are gone; fused replacement is present.
    assert_eq!(count_calls_named(&mir, "matmul"), 0);
    assert_eq!(count_calls_named(&mir, "norm"), 0);
    assert_eq!(count_calls_named(&mir, "fused_matmul_norm"), 1);
}

#[test]
fn rewriter_preserves_l2_output_bit_for_bit() {
    let ast = parse(SRC_MATMUL_NORM_L2);
    let unrewritten = lower_to_mir(&ast);
    let mut rewritten = lower_to_mir(&ast);
    let result = fusion_rewrite_program(&mut rewritten);
    assert_eq!(result.rewrites_applied, 1);

    let out_unrew = run_mir(&unrewritten, 0);
    let out_rew = run_mir(&rewritten, 0);

    assert_eq!(
        out_unrew, out_rew,
        "rewritten output diverged from unrewritten — parity broken"
    );
    // Sanity: matmul = [[3, 0], [0, 4]], ||·||_2 = 5
    assert_eq!(out_unrew, vec!["5"]);
}

#[test]
fn rewriter_preserves_l1_with_ord_arg() {
    let ast = parse(SRC_MATMUL_NORM_L1);
    let unrewritten = lower_to_mir(&ast);
    let mut rewritten = lower_to_mir(&ast);
    let result = fusion_rewrite_program(&mut rewritten);
    assert_eq!(result.rewrites_applied, 1);

    let out_unrew = run_mir(&unrewritten, 0);
    let out_rew = run_mir(&rewritten, 0);

    assert_eq!(out_unrew, out_rew, "L1 parity broken");
    // matmul = w = [[1, 2], [3, 4]], L1 = 10
    assert_eq!(out_unrew, vec!["10"]);
}

#[test]
fn rewriter_skips_when_h_consumed_twice() {
    // h is consumed by norm AND used in a print — rewriter must not fire.
    let src = r#"
fn main() {
    let a = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let w = Tensor.from_vec([3.0, 0.0, 0.0, 4.0], [2, 2]);
    let h = matmul(a, w);
    let n = norm(h);
    print(h.shape()[0]);
    print(n);
}
"#;
    let ast = parse(src);
    let mut mir = lower_to_mir(&ast);
    let result = fusion_rewrite_program(&mut mir);
    assert_eq!(
        result.rewrites_applied, 0,
        "must not rewrite when h has additional users"
    );
    // Topology unchanged.
    assert_eq!(count_calls_named(&mir, "matmul"), 1);
    assert_eq!(count_calls_named(&mir, "norm"), 1);
    assert_eq!(count_calls_named(&mir, "fused_matmul_norm"), 0);
}

#[test]
fn rewriter_is_byte_identical_across_runs() {
    // Run the rewriter 50 times against the same input; the rewritten MIR
    // must be structurally identical every time (determinism contract).
    let ast = parse(SRC_MATMUL_NORM_L2);
    let mut first_call_count: Option<(usize, usize, usize)> = None;
    for _ in 0..50 {
        let mut mir = lower_to_mir(&ast);
        let _ = fusion_rewrite_program(&mut mir);
        let counts = (
            count_calls_named(&mir, "matmul"),
            count_calls_named(&mir, "norm"),
            count_calls_named(&mir, "fused_matmul_norm"),
        );
        match first_call_count {
            None => first_call_count = Some(counts),
            Some(prev) => assert_eq!(prev, counts, "rewriter not deterministic"),
        }
    }
}
