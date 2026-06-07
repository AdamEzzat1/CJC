//! CANA Phase 3.5c — MIR fusion rewriter.
//!
//! Walks every function body looking for chains of native primitive calls
//! whose intermediate values are use-once. Each matched chain is replaced
//! with a single call to the corresponding fused primitive (registered in
//! [`cjc-runtime`]).
//!
//! ## What this pass does
//!
//! Today: the `matmul → norm` pair. A statement sequence
//!
//! ```cjcl
//! let h = matmul(a, w);    // produces [M, N] intermediate
//! let n = norm(h);          // returns Float
//! ```
//!
//! is rewritten to
//!
//! ```cjcl
//! let n = fused_matmul_norm(a, w);   // same Float, no [M, N] allocation
//! ```
//!
//! when `h` is referenced exactly once across the enclosing body — i.e.
//! by the norm call only.
//!
//! ## What this pass is NOT
//!
//! - **Not a cross-function rewriter.** A chain that crosses a function
//!   boundary stays unfused.
//! - **Not a cross-block rewriter.** A chain split by an `if`/`while`/
//!   nested body stays unfused (the recursive walk handles each body
//!   independently).
//! - **Not a control-flow analyser.** We only look at adjacent
//!   `MirStmt::Let` pairs in straight-line code.
//!
//! These restrictions exist so the pass can never produce wrong code: if
//! we can't *prove* the rewrite is sound, we don't do it.
//!
//! ## Determinism contract
//!
//! - Same MIR in → same MIR out, byte-identical across runs.
//! - Pattern matching iterates statements in their source order with no
//!   randomness, no `HashMap`, no float comparison.
//! - The rewritten primitive (`fused_matmul_norm`) is bit-identical to the
//!   unfused chain on the sequential matmul path (verified by the
//!   [`tests/fused_matmul_norm`] suite).
//!
//! ## How it's wired
//!
//! Registered as the `"fusion_rewrite"` pass in
//! [`crate::optimize::apply_pass`]. Not in [`DEFAULT_PASS_SEQUENCE`] —
//! a CANA-driven `PassPlan` opts in by name. Adding this pass to a plan
//! does not change AST/MIR parity because the fused primitive is
//! bit-identical to the unfused chain.

use crate::{MirBody, MirExpr, MirExprKind, MirFunction, MirProgram, MirStmt};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of running the fusion rewriter over a program.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FusionRewriteResult {
    /// Total number of let-pairs collapsed into fused calls across all
    /// functions in the program.
    pub rewrites_applied: usize,
}

/// Rewrite all eligible fusion candidates in a program in place.
pub fn fusion_rewrite_program(program: &mut MirProgram) -> FusionRewriteResult {
    let mut result = FusionRewriteResult::default();
    for func in &mut program.functions {
        result.rewrites_applied += fusion_rewrite_fn(func);
    }
    result
}

/// Rewrite all eligible fusion candidates in a function in place.
///
/// Returns the number of rewrites applied. Idempotent: running it twice
/// on the same function produces no additional rewrites on the second
/// pass (the fused primitive name is not itself a fusion target).
pub fn fusion_rewrite_fn(func: &mut MirFunction) -> usize {
    fusion_rewrite_body(&mut func.body)
}

// ---------------------------------------------------------------------------
// Body-level rewriter
// ---------------------------------------------------------------------------

/// Walk a body. For each adjacent (matmul-let, norm-let) pair where the
/// matmul output binding is use-once in the body, collapse to a single
/// fused_matmul_norm let. Recurses into nested bodies (if/while/nogc).
fn fusion_rewrite_body(body: &mut MirBody) -> usize {
    // Recurse into nested bodies first. Doing this before the local pass
    // means a nested rewrite never invalidates an outer pattern (an inner
    // rewrite can only reduce use counts in the inner scope).
    let mut rewrites = recurse_into_nested(body);

    // Pre-compute use counts for every variable name in this body. We
    // include the result expression and any nested scopes, but we don't
    // *re*-count for each pair — the use-count map is built once.
    let counts = body_use_counts(body);

    // Walk stmts pairwise looking for our pattern.
    let mut i = 0;
    while i + 1 < body.stmts.len() {
        // Try matmul → norm
        if let Some(matched) = match_matmul_norm_pair(&body.stmts[i], &body.stmts[i + 1]) {
            let h_count = counts.get(&matched.h_name).copied().unwrap_or(0);
            if h_count == 1 {
                let fused = build_fused_let(matched);
                body.stmts.splice(i..i + 2, std::iter::once(fused));
                rewrites += 1;
                i += 1;
                continue;
            }
        }
        // Try matmul → matmul
        if let Some(matched) = match_matmul_matmul_pair(&body.stmts[i], &body.stmts[i + 1]) {
            let h_count = counts.get(&matched.h_name).copied().unwrap_or(0);
            if h_count == 1 {
                let fused = build_fused_matmul_matmul_let(matched);
                body.stmts.splice(i..i + 2, std::iter::once(fused));
                rewrites += 1;
                i += 1;
                continue;
            }
        }
        i += 1;
    }

    rewrites
}

/// Recurse into every nested body inside a `MirStmt::If`, `While`, or
/// `NoGcBlock`. Each nested body gets its own scope so we can rewrite
/// independently inside.
fn recurse_into_nested(body: &mut MirBody) -> usize {
    let mut count = 0;
    for stmt in &mut body.stmts {
        match stmt {
            MirStmt::If { then_body, else_body, .. } => {
                count += fusion_rewrite_body(then_body);
                if let Some(eb) = else_body {
                    count += fusion_rewrite_body(eb);
                }
            }
            MirStmt::While { body: wb, .. } => {
                count += fusion_rewrite_body(wb);
            }
            MirStmt::NoGcBlock(nb) => {
                count += fusion_rewrite_body(nb);
            }
            _ => {}
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Pattern matching
// ---------------------------------------------------------------------------

/// Per-match data: names of the matmul output, the original norm output,
/// and the inputs to fold into the fused call.
struct MatmulNormMatch<'a> {
    h_name: String,
    n_name: String,
    n_mutable: bool,
    n_slot: Option<u32>,
    a_expr: &'a MirExpr,
    w_expr: &'a MirExpr,
    /// Optional ord argument (present when source code was `norm(h, p)`).
    ord_expr: Option<&'a MirExpr>,
}

/// Return Some(...) if (stmt1, stmt2) is the `let h = matmul(a, w); let n = norm(h, ?)` pattern.
fn match_matmul_norm_pair<'a>(
    stmt1: &'a MirStmt,
    stmt2: &'a MirStmt,
) -> Option<MatmulNormMatch<'a>> {
    let MirStmt::Let { name: h_name, init: init_m, .. } = stmt1 else { return None; };
    let MirStmt::Let { name: n_name, init: init_n, mutable: n_mut, slot: n_slot, .. } = stmt2 else { return None; };

    // stmt1 init must be Call(matmul, [a, w])
    let (m_callee, m_args) = match &init_m.kind {
        MirExprKind::Call { callee, args } => (callee, args),
        _ => return None,
    };
    if callee_name(m_callee) != Some("matmul") {
        return None;
    }
    if m_args.len() != 2 {
        return None;
    }

    // stmt2 init must be Call(norm, [Var(h_name), ord?])
    let (n_callee, n_args) = match &init_n.kind {
        MirExprKind::Call { callee, args } => (callee, args),
        _ => return None,
    };
    if callee_name(n_callee) != Some("norm") {
        return None;
    }
    if n_args.is_empty() || n_args.len() > 2 {
        return None;
    }
    if var_name(&n_args[0]) != Some(h_name.as_str()) {
        return None;
    }

    Some(MatmulNormMatch {
        h_name: h_name.clone(),
        n_name: n_name.clone(),
        n_mutable: *n_mut,
        n_slot: *n_slot,
        a_expr: &m_args[0],
        w_expr: &m_args[1],
        ord_expr: n_args.get(1),
    })
}

/// Extract the name from a callee expression — `Var("matmul")` or
/// `VarLocal { name: "matmul", .. }` both qualify.
fn callee_name(expr: &MirExpr) -> Option<&str> {
    match &expr.kind {
        MirExprKind::Var(n) => Some(n.as_str()),
        MirExprKind::VarLocal { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

/// Extract the binding name when an expression is a plain variable
/// reference. Returns None for anything else (literal, complex expression).
fn var_name(expr: &MirExpr) -> Option<&str> {
    match &expr.kind {
        MirExprKind::Var(n) => Some(n.as_str()),
        MirExprKind::VarLocal { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

/// Build the replacement `let n = fused_matmul_norm(a, w, ord?);` statement
/// from a matched pair, preserving the norm let's binding name, mutability,
/// and slot.
fn build_fused_let(m: MatmulNormMatch<'_>) -> MirStmt {
    let mut args = vec![m.a_expr.clone(), m.w_expr.clone()];
    if let Some(ord) = m.ord_expr {
        args.push(ord.clone());
    }
    let call = MirExpr {
        kind: MirExprKind::Call {
            callee: Box::new(MirExpr {
                kind: MirExprKind::Var("fused_matmul_norm".to_string()),
            }),
            args,
        },
    };
    MirStmt::Let {
        name: m.n_name,
        mutable: m.n_mutable,
        init: call,
        alloc_hint: None,
        slot: m.n_slot,
    }
}

/// Match for `let h = matmul(a, b); let r = matmul(h, c);` (left-associative
/// triple-product). When `h` is use-once, the rewriter collapses both lets
/// into a single `let r = fused_matmul_matmul(a, b, c);`.
struct MatmulMatmulMatch<'a> {
    h_name: String,
    r_name: String,
    r_mutable: bool,
    r_slot: Option<u32>,
    a_expr: &'a MirExpr,
    b_expr: &'a MirExpr,
    c_expr: &'a MirExpr,
}

fn match_matmul_matmul_pair<'a>(
    stmt1: &'a MirStmt,
    stmt2: &'a MirStmt,
) -> Option<MatmulMatmulMatch<'a>> {
    let MirStmt::Let { name: h_name, init: init_h, .. } = stmt1 else { return None; };
    let MirStmt::Let { name: r_name, init: init_r, mutable: r_mut, slot: r_slot, .. } = stmt2
    else {
        return None;
    };

    // stmt1: Call(matmul, [a, b])
    let (h_callee, h_args) = match &init_h.kind {
        MirExprKind::Call { callee, args } => (callee, args),
        _ => return None,
    };
    if callee_name(h_callee) != Some("matmul") || h_args.len() != 2 {
        return None;
    }

    // stmt2: Call(matmul, [Var(h_name), c])
    let (r_callee, r_args) = match &init_r.kind {
        MirExprKind::Call { callee, args } => (callee, args),
        _ => return None,
    };
    if callee_name(r_callee) != Some("matmul") || r_args.len() != 2 {
        return None;
    }
    if var_name(&r_args[0]) != Some(h_name.as_str()) {
        return None;
    }

    Some(MatmulMatmulMatch {
        h_name: h_name.clone(),
        r_name: r_name.clone(),
        r_mutable: *r_mut,
        r_slot: *r_slot,
        a_expr: &h_args[0],
        b_expr: &h_args[1],
        c_expr: &r_args[1],
    })
}

fn build_fused_matmul_matmul_let(m: MatmulMatmulMatch<'_>) -> MirStmt {
    let call = MirExpr {
        kind: MirExprKind::Call {
            callee: Box::new(MirExpr {
                kind: MirExprKind::Var("fused_matmul_matmul".to_string()),
            }),
            args: vec![m.a_expr.clone(), m.b_expr.clone(), m.c_expr.clone()],
        },
    };
    MirStmt::Let {
        name: m.r_name,
        mutable: m.r_mutable,
        init: call,
        alloc_hint: None,
        slot: m.r_slot,
    }
}

// ---------------------------------------------------------------------------
// Use-count helpers (tree-form MIR only)
// ---------------------------------------------------------------------------

/// Count occurrences of every Var/VarLocal name across an entire body,
/// including nested bodies and the trailing result expression. Returns a
/// `BTreeMap<name, count>` for deterministic ordering.
fn body_use_counts(body: &MirBody) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for stmt in &body.stmts {
        count_uses_in_stmt(stmt, &mut counts);
    }
    if let Some(e) = &body.result {
        count_uses_in_expr(e, &mut counts);
    }
    counts
}

fn count_uses_in_stmt(stmt: &MirStmt, counts: &mut BTreeMap<String, usize>) {
    match stmt {
        MirStmt::Let { init, .. } => count_uses_in_expr(init, counts),
        MirStmt::Expr(e) => count_uses_in_expr(e, counts),
        MirStmt::If { cond, then_body, else_body } => {
            count_uses_in_expr(cond, counts);
            for s in &then_body.stmts {
                count_uses_in_stmt(s, counts);
            }
            if let Some(e) = &then_body.result {
                count_uses_in_expr(e, counts);
            }
            if let Some(eb) = else_body {
                for s in &eb.stmts {
                    count_uses_in_stmt(s, counts);
                }
                if let Some(e) = &eb.result {
                    count_uses_in_expr(e, counts);
                }
            }
        }
        MirStmt::While { cond, body } => {
            count_uses_in_expr(cond, counts);
            for s in &body.stmts {
                count_uses_in_stmt(s, counts);
            }
            if let Some(e) = &body.result {
                count_uses_in_expr(e, counts);
            }
        }
        MirStmt::Return(Some(e)) => count_uses_in_expr(e, counts),
        MirStmt::Return(None) | MirStmt::Break | MirStmt::Continue => {}
        MirStmt::NoGcBlock(b) => {
            for s in &b.stmts {
                count_uses_in_stmt(s, counts);
            }
            if let Some(e) = &b.result {
                count_uses_in_expr(e, counts);
            }
        }
    }
}

fn count_uses_in_expr(expr: &MirExpr, counts: &mut BTreeMap<String, usize>) {
    match &expr.kind {
        MirExprKind::Var(n) | MirExprKind::VarLocal { name: n, .. } => {
            *counts.entry(n.clone()).or_insert(0) += 1;
        }
        MirExprKind::Binary { left, right, .. } => {
            count_uses_in_expr(left, counts);
            count_uses_in_expr(right, counts);
        }
        MirExprKind::Unary { operand, .. } => count_uses_in_expr(operand, counts),
        MirExprKind::Call { callee, args } => {
            count_uses_in_expr(callee, counts);
            for a in args {
                count_uses_in_expr(a, counts);
            }
        }
        MirExprKind::Assign { target, value } => {
            count_uses_in_expr(target, counts);
            count_uses_in_expr(value, counts);
        }
        MirExprKind::Field { object, .. } => count_uses_in_expr(object, counts),
        MirExprKind::Index { object, index } => {
            count_uses_in_expr(object, counts);
            count_uses_in_expr(index, counts);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, e) in fields {
                count_uses_in_expr(e, counts);
            }
        }
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            for e in elems {
                count_uses_in_expr(e, counts);
            }
        }
        MirExprKind::MakeClosure { captures, .. } => {
            for c in captures {
                count_uses_in_expr(c, counts);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MirFnId, MirFunction, MirProgram};

    fn ekind(k: MirExprKind) -> MirExpr {
        MirExpr { kind: k }
    }

    fn var(name: &str) -> MirExpr {
        ekind(MirExprKind::Var(name.to_string()))
    }

    fn call(callee: &str, args: Vec<MirExpr>) -> MirExpr {
        ekind(MirExprKind::Call {
            callee: Box::new(var(callee)),
            args,
        })
    }

    fn let_stmt(name: &str, init: MirExpr) -> MirStmt {
        MirStmt::Let {
            name: name.to_string(),
            mutable: false,
            init,
            alloc_hint: None,
            slot: None,
        }
    }

    fn empty_fn(name: &str) -> MirFunction {
        MirFunction {
            id: MirFnId(0),
            name: name.to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: MirBody {
                stmts: vec![],
                result: None,
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Public,
            local_count: 0,
        }
    }

    fn program(stmts: Vec<MirStmt>) -> MirProgram {
        let mut f = empty_fn("main");
        f.body.stmts = stmts;
        MirProgram {
            functions: vec![f],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    #[test]
    fn rewrites_simple_matmul_norm_chain() {
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("w")])),
            let_stmt("n", call("norm", vec![var("h")])),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 1);
        let stmts = &prog.functions[0].body.stmts;
        assert_eq!(stmts.len(), 1, "matmul let should be removed");
        match &stmts[0] {
            MirStmt::Let { name, init, .. } => {
                assert_eq!(name, "n");
                match &init.kind {
                    MirExprKind::Call { callee, args } => {
                        assert_eq!(callee_name(callee), Some("fused_matmul_norm"));
                        assert_eq!(args.len(), 2);
                        assert_eq!(var_name(&args[0]), Some("a"));
                        assert_eq!(var_name(&args[1]), Some("w"));
                    }
                    other => panic!("expected Call, got {other:?}"),
                }
            }
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn rewrites_matmul_norm_with_ord() {
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("w")])),
            let_stmt(
                "n",
                call(
                    "norm",
                    vec![var("h"), ekind(MirExprKind::IntLit(1))],
                ),
            ),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 1);
        match &prog.functions[0].body.stmts[0] {
            MirStmt::Let { init, .. } => match &init.kind {
                MirExprKind::Call { args, .. } => {
                    assert_eq!(args.len(), 3, "ord arg should be preserved");
                    assert!(matches!(args[2].kind, MirExprKind::IntLit(1)));
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn refuses_rewrite_when_h_used_twice() {
        // h is consumed by both norm and another call — rewriting would
        // delete h's binding and break the second consumer.
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("w")])),
            let_stmt("n", call("norm", vec![var("h")])),
            let_stmt("g", call("transpose", vec![var("h")])),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 0);
        assert_eq!(prog.functions[0].body.stmts.len(), 3);
    }

    #[test]
    fn refuses_rewrite_when_stmts_not_adjacent() {
        // Intervening unrelated let — we don't reorder, we don't rewrite.
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("w")])),
            let_stmt("z", call("ones", vec![var("k")])),
            let_stmt("n", call("norm", vec![var("h")])),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 0);
    }

    #[test]
    fn refuses_rewrite_when_norm_takes_non_var_first_arg() {
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("w")])),
            let_stmt(
                "n",
                call("norm", vec![call("transpose", vec![var("h")])]),
            ),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 0);
    }

    #[test]
    fn refuses_rewrite_when_norm_takes_wrong_var() {
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("w")])),
            // norm of x, not h — we don't rewrite.
            let_stmt("n", call("norm", vec![var("x")])),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 0);
    }

    #[test]
    fn rewrites_inside_nested_if_body() {
        let inner = MirBody {
            stmts: vec![
                let_stmt("h", call("matmul", vec![var("a"), var("w")])),
                let_stmt("n", call("norm", vec![var("h")])),
            ],
            result: None,
        };
        let mut prog = program(vec![MirStmt::If {
            cond: ekind(MirExprKind::BoolLit(true)),
            then_body: inner,
            else_body: None,
        }]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 1);
    }

    #[test]
    fn idempotent_second_run_yields_no_rewrites() {
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("w")])),
            let_stmt("n", call("norm", vec![var("h")])),
        ]);
        let first = fusion_rewrite_program(&mut prog);
        assert_eq!(first.rewrites_applied, 1);
        let second = fusion_rewrite_program(&mut prog);
        assert_eq!(
            second.rewrites_applied, 0,
            "second pass must be a no-op — fused_matmul_norm is not a fusion target"
        );
    }

    #[test]
    fn empty_program_no_rewrites() {
        let mut prog = program(vec![]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 0);
    }

    #[test]
    fn handles_varlocal_callee() {
        // Post slot-resolution, callees become VarLocal { name, slot }.
        let mut prog = program(vec![
            MirStmt::Let {
                name: "h".to_string(),
                mutable: false,
                init: ekind(MirExprKind::Call {
                    callee: Box::new(ekind(MirExprKind::VarLocal {
                        name: "matmul".to_string(),
                        slot: 7,
                    })),
                    args: vec![var("a"), var("w")],
                }),
                alloc_hint: None,
                slot: None,
            },
            MirStmt::Let {
                name: "n".to_string(),
                mutable: false,
                init: ekind(MirExprKind::Call {
                    callee: Box::new(ekind(MirExprKind::VarLocal {
                        name: "norm".to_string(),
                        slot: 9,
                    })),
                    args: vec![ekind(MirExprKind::VarLocal {
                        name: "h".to_string(),
                        slot: 8,
                    })],
                }),
                alloc_hint: None,
                slot: None,
            },
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 1);
    }

    #[test]
    fn rewrites_two_chains_in_one_body() {
        let mut prog = program(vec![
            let_stmt("h1", call("matmul", vec![var("a1"), var("w1")])),
            let_stmt("n1", call("norm", vec![var("h1")])),
            let_stmt("h2", call("matmul", vec![var("a2"), var("w2")])),
            let_stmt("n2", call("norm", vec![var("h2")])),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 2);
        assert_eq!(prog.functions[0].body.stmts.len(), 2);
    }

    // ------------------------------------------------------------------
    // matmul → matmul tests
    // ------------------------------------------------------------------

    #[test]
    fn rewrites_matmul_matmul_chain() {
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("b")])),
            let_stmt("r", call("matmul", vec![var("h"), var("c")])),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 1);
        let stmts = &prog.functions[0].body.stmts;
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            MirStmt::Let { name, init, .. } => {
                assert_eq!(name, "r");
                match &init.kind {
                    MirExprKind::Call { callee, args } => {
                        assert_eq!(callee_name(callee), Some("fused_matmul_matmul"));
                        assert_eq!(args.len(), 3);
                        assert_eq!(var_name(&args[0]), Some("a"));
                        assert_eq!(var_name(&args[1]), Some("b"));
                        assert_eq!(var_name(&args[2]), Some("c"));
                    }
                    other => panic!("expected Call, got {other:?}"),
                }
            }
            other => panic!("expected Let, got {other:?}"),
        }
    }

    #[test]
    fn refuses_matmul_matmul_when_h_used_twice() {
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("b")])),
            let_stmt("r", call("matmul", vec![var("h"), var("c")])),
            let_stmt("g", call("transpose", vec![var("h")])),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 0);
    }

    #[test]
    fn refuses_matmul_matmul_when_h_not_first_arg_of_second() {
        // r = matmul(c, h) — h in second position, not first. We only
        // rewrite the left-associative shape (a@b)@c.
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("b")])),
            let_stmt("r", call("matmul", vec![var("c"), var("h")])),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 0);
    }

    #[test]
    fn fused_matmul_matmul_not_re_fused() {
        // Idempotency: after rewriting, the new fused call should NOT match
        // either pattern on the next pass.
        let mut prog = program(vec![
            let_stmt("h", call("matmul", vec![var("a"), var("b")])),
            let_stmt("r", call("matmul", vec![var("h"), var("c")])),
        ]);
        let first = fusion_rewrite_program(&mut prog);
        assert_eq!(first.rewrites_applied, 1);
        let second = fusion_rewrite_program(&mut prog);
        assert_eq!(second.rewrites_applied, 0);
    }

    #[test]
    fn rewrites_mixed_chain_types() {
        // Two different fusion patterns in one body — each pair independent.
        let mut prog = program(vec![
            // pair 1: matmul → norm
            let_stmt("h1", call("matmul", vec![var("a1"), var("w1")])),
            let_stmt("n1", call("norm", vec![var("h1")])),
            // pair 2: matmul → matmul
            let_stmt("h2", call("matmul", vec![var("a2"), var("b2")])),
            let_stmt("r2", call("matmul", vec![var("h2"), var("c2")])),
        ]);
        let result = fusion_rewrite_program(&mut prog);
        assert_eq!(result.rewrites_applied, 2);
        assert_eq!(prog.functions[0].body.stmts.len(), 2);
        // Verify the second is fused_matmul_matmul
        if let MirStmt::Let { init, .. } = &prog.functions[0].body.stmts[1] {
            if let MirExprKind::Call { callee, .. } = &init.kind {
                assert_eq!(callee_name(callee), Some("fused_matmul_matmul"));
            }
        }
    }
}
