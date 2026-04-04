//! Escape analysis — intraprocedural MIR annotation pass.
//!
//! Classifies every `let`-binding in a MIR function as one of:
//!
//! - `Stack`  — primitive type, no heap allocation needed
//! - `Arena`  — non-escaping heap value, eligible for arena allocation
//! - `Rc`     — escaping value, requires reference counting
//!
//! The analysis is **conservative**: unknown = escapes (→ Rc).
//!
//! # Algorithm
//!
//! 1. Seed all let-bindings as `Arena` (optimistic).
//! 2. Walk the function body looking for escape points:
//!    - `Return(Some(expr))` — if expr references binding → `Rc`
//!    - `MakeClosure { captures }` — captured bindings → `Rc`
//!    - `ArrayLit/TupleLit/StructLit` containing binding ref → `Rc`
//!    - `Call { callee, args }` — if callee is unknown, args → `Rc`
//!    - `Assign { target: Field/Index }` — target root binding → `Rc`
//!    - Mutable let-bindings → `Rc` (conservative)
//! 3. Primitives (Int, Float, Bool, U8, Void) → always `Stack`.
//! 4. String/Array/Tuple/Struct with no escape path → `Arena`.
//!
//! # Determinism
//!
//! The analysis itself is fully deterministic: same MIR input → same
//! classification output. Iteration order is statement order in the body.
//!
//! # Integration
//!
//! After HIR→MIR lowering, call `analyze_program()` to get per-function
//! escape info. Then call `annotate_program()` to write `AllocHint` values
//! into `MirStmt::Let` nodes.

use std::collections::BTreeMap;

use crate::{MirBody, MirExpr, MirExprKind, MirFunction, MirProgram, MirStmt, MirPattern};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Allocation strategy hint for a let-binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocHint {
    /// Primitive type — lives on the stack, no heap allocation.
    Stack,
    /// Non-escaping heap value — eligible for frame-arena allocation
    /// (bulk-freed at function return).
    Arena,
    /// Escaping value — requires Rc (reference counting).
    Rc,
}

/// Why a binding was classified as escaping (or not).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EscapeReason {
    /// Value does not escape the function.
    NonEscaping,
    /// Primitive type — always stack-allocated.
    Primitive,
    /// Value is returned from the function.
    ReturnedFromFn,
    /// Value is captured by a closure.
    CapturedByClosure,
    /// Value is stored inside a container (array, tuple, struct, enum).
    StoredInContainer,
    /// Value is passed to a function whose parameter escape behavior is unknown.
    PassedToUnknownFn,
    /// Value is assigned to a field or index of another object.
    AssignedToFieldOrIndex,
    /// Binding is mutable — conservative escape (value may be aliased).
    Mutable,
    /// Init expression is a call whose return value may allocate (conservative).
    CallResult,
}

/// Per-function escape analysis results.
#[derive(Debug, Clone)]
pub struct EscapeInfo {
    /// Maps binding name → (AllocHint, EscapeReason).
    pub bindings: BTreeMap<String, (AllocHint, EscapeReason)>,
}

// ---------------------------------------------------------------------------
// Analysis entry points
// ---------------------------------------------------------------------------

/// Analyze a single MIR function, returning escape info for its let-bindings.
pub fn analyze_function(func: &MirFunction) -> EscapeInfo {
    let mut ctx = AnalysisCtx::new();

    // Phase 1: Collect all let-bindings and their init expression categories.
    collect_bindings_body(&func.body, &mut ctx);

    // Phase 2: Walk body for escape points.
    walk_body_for_escapes(&func.body, &mut ctx);

    // Phase 3: Mark mutable bindings as Rc (conservative).
    mark_mutables(&func.body, &mut ctx);

    // Phase 4: Classify primitives as Stack.
    classify_primitives(&func.body, &mut ctx);

    // Build result.
    let mut bindings = BTreeMap::new();
    for (name, info) in &ctx.bindings {
        let hint = match info.reason {
            EscapeReason::Primitive => AllocHint::Stack,
            EscapeReason::NonEscaping => {
                if is_primitive_init(&info.init_kind) {
                    AllocHint::Stack
                } else {
                    AllocHint::Arena
                }
            }
            _ => AllocHint::Rc,
        };
        bindings.insert(name.clone(), (hint, info.reason));
    }

    EscapeInfo { bindings }
}

/// Analyze all functions in a MIR program.
pub fn analyze_program(program: &MirProgram) -> BTreeMap<String, EscapeInfo> {
    let mut result = BTreeMap::new();
    for func in &program.functions {
        let info = analyze_function(func);
        result.insert(func.name.clone(), info);
    }
    result
}

/// Annotate a MIR program's `Let` statements with `AllocHint` values
/// derived from escape analysis.
pub fn annotate_program(program: &mut MirProgram) {
    let analysis = analyze_program(program);
    for func in &mut program.functions {
        if let Some(info) = analysis.get(&func.name) {
            annotate_body(&mut func.body, info);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal analysis context
// ---------------------------------------------------------------------------

/// Category of a let-binding's init expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitKind {
    /// Primitive literal (Int, Float, Bool, U8, Void).
    Primitive,
    /// String literal — heap-allocated but may be arena-eligible.
    StringLit,
    /// Container literal (Array, Tuple, Struct, Variant).
    Container,
    /// Call result — conservative, may allocate.
    Call,
    /// Variable reference — inherits from source.
    Var,
    /// Closure creation.
    Closure,
    /// Other expression.
    Other,
}

/// Per-binding info during analysis.
#[derive(Debug, Clone)]
struct BindingInfo {
    init_kind: InitKind,
    reason: EscapeReason,
    mutable: bool,
}

struct AnalysisCtx {
    bindings: BTreeMap<String, BindingInfo>,
}

impl AnalysisCtx {
    fn new() -> Self {
        AnalysisCtx {
            bindings: BTreeMap::new(),
        }
    }

    /// Mark a binding as escaping with the given reason.
    /// Only escalates (NonEscaping → escaping); never downgrades.
    fn mark_escaping(&mut self, name: &str, reason: EscapeReason) {
        if let Some(info) = self.bindings.get_mut(name) {
            if info.reason == EscapeReason::NonEscaping
                || info.reason == EscapeReason::Primitive
            {
                info.reason = reason;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Collect bindings
// ---------------------------------------------------------------------------

fn classify_init_expr(expr: &MirExpr) -> InitKind {
    match &expr.kind {
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::NaLit
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::Void => InitKind::Primitive,

        // Binary/unary ops on primitives produce primitives (e.g. `1 + 2`).
        // We check recursively: if both operands classify as Primitive, the
        // result is Primitive.  Otherwise, conservatively fall through to Other.
        MirExprKind::Binary { left, right, .. } => {
            if classify_init_expr(left) == InitKind::Primitive
                && classify_init_expr(right) == InitKind::Primitive
            {
                InitKind::Primitive
            } else {
                InitKind::Other
            }
        }
        MirExprKind::Unary { operand, .. } => {
            if classify_init_expr(operand) == InitKind::Primitive {
                InitKind::Primitive
            } else {
                InitKind::Other
            }
        }

        MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. } => InitKind::StringLit,

        MirExprKind::ArrayLit(_)
        | MirExprKind::TupleLit(_)
        | MirExprKind::StructLit { .. }
        | MirExprKind::VariantLit { .. }
        | MirExprKind::TensorLit { .. } => InitKind::Container,

        MirExprKind::Call { .. } => InitKind::Call,

        MirExprKind::Var(_) | MirExprKind::Col(_) => InitKind::Var,

        MirExprKind::MakeClosure { .. } | MirExprKind::Lambda { .. } => InitKind::Closure,

        _ => InitKind::Other,
    }
}

fn collect_bindings_body(body: &MirBody, ctx: &mut AnalysisCtx) {
    for stmt in &body.stmts {
        collect_bindings_stmt(stmt, ctx);
    }
}

fn collect_bindings_stmt(stmt: &MirStmt, ctx: &mut AnalysisCtx) {
    match stmt {
        MirStmt::Let { name, mutable, init, .. } => {
            let init_kind = classify_init_expr(init);
            let reason = if is_primitive_init(&init_kind) {
                EscapeReason::Primitive
            } else {
                EscapeReason::NonEscaping
            };
            ctx.bindings.insert(
                name.clone(),
                BindingInfo {
                    init_kind,
                    reason,
                    mutable: *mutable,
                },
            );
            // Recurse into init expression (may contain blocks with more lets).
            collect_bindings_expr(init, ctx);
        }
        MirStmt::If { then_body, else_body, cond, .. } => {
            collect_bindings_expr(cond, ctx);
            collect_bindings_body(then_body, ctx);
            if let Some(eb) = else_body {
                collect_bindings_body(eb, ctx);
            }
        }
        MirStmt::While { cond, body } => {
            collect_bindings_expr(cond, ctx);
            collect_bindings_body(body, ctx);
        }
        MirStmt::NoGcBlock(body) => {
            collect_bindings_body(body, ctx);
        }
        MirStmt::Return(opt) => {
            if let Some(e) = opt {
                collect_bindings_expr(e, ctx);
            }
        }
        MirStmt::Expr(e) => {
            collect_bindings_expr(e, ctx);
        }
        // Break/Continue have no sub-expressions.
        MirStmt::Break | MirStmt::Continue => {}
    }
}

fn collect_bindings_expr(expr: &MirExpr, ctx: &mut AnalysisCtx) {
    match &expr.kind {
        MirExprKind::Block(body) => collect_bindings_body(body, ctx),
        MirExprKind::If { then_body, else_body, cond } => {
            collect_bindings_expr(cond, ctx);
            collect_bindings_body(then_body, ctx);
            if let Some(eb) = else_body {
                collect_bindings_body(eb, ctx);
            }
        }
        MirExprKind::Match { scrutinee, arms } => {
            collect_bindings_expr(scrutinee, ctx);
            for arm in arms {
                // Pattern bindings.
                collect_pattern_bindings(&arm.pattern, ctx);
                collect_bindings_body(&arm.body, ctx);
            }
        }
        MirExprKind::Call { callee, args } => {
            collect_bindings_expr(callee, ctx);
            for a in args {
                collect_bindings_expr(a, ctx);
            }
        }
        MirExprKind::Binary { left, right, .. } => {
            collect_bindings_expr(left, ctx);
            collect_bindings_expr(right, ctx);
        }
        MirExprKind::Unary { operand, .. } => {
            collect_bindings_expr(operand, ctx);
        }
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            for e in elems {
                collect_bindings_expr(e, ctx);
            }
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, e) in fields {
                collect_bindings_expr(e, ctx);
            }
        }
        MirExprKind::VariantLit { fields, .. } => {
            for e in fields {
                collect_bindings_expr(e, ctx);
            }
        }
        MirExprKind::Field { object, .. } => collect_bindings_expr(object, ctx),
        MirExprKind::Index { object, index } => {
            collect_bindings_expr(object, ctx);
            collect_bindings_expr(index, ctx);
        }
        MirExprKind::MultiIndex { object, indices } => {
            collect_bindings_expr(object, ctx);
            for i in indices {
                collect_bindings_expr(i, ctx);
            }
        }
        MirExprKind::Assign { target, value } => {
            collect_bindings_expr(target, ctx);
            collect_bindings_expr(value, ctx);
        }
        MirExprKind::MakeClosure { captures, .. } => {
            for c in captures {
                collect_bindings_expr(c, ctx);
            }
        }
        MirExprKind::Lambda { body, .. } => {
            collect_bindings_expr(body, ctx);
        }
        MirExprKind::TensorLit { rows } => {
            for row in rows {
                for e in row {
                    collect_bindings_expr(e, ctx);
                }
            }
        }
        MirExprKind::LinalgLU { operand }
        | MirExprKind::LinalgQR { operand }
        | MirExprKind::LinalgCholesky { operand }
        | MirExprKind::LinalgInv { operand } => {
            collect_bindings_expr(operand, ctx);
        }
        MirExprKind::Broadcast { operand, target_shape } => {
            collect_bindings_expr(operand, ctx);
            for s in target_shape {
                collect_bindings_expr(s, ctx);
            }
        }
        // Leaves — no sub-expressions with bindings.
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::NaLit
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Var(_)
        | MirExprKind::Col(_)
        | MirExprKind::Void => {}
    }
}

/// Collect bindings introduced by pattern matching (match arm patterns).
fn collect_pattern_bindings(pattern: &MirPattern, ctx: &mut AnalysisCtx) {
    match pattern {
        MirPattern::Binding(name) => {
            // Pattern bindings are non-mutable (match bindings are immutable in CJC).
            // Their init_kind is "Other" since the actual value comes from the scrutinee.
            ctx.bindings.insert(
                name.clone(),
                BindingInfo {
                    init_kind: InitKind::Other,
                    reason: EscapeReason::NonEscaping,
                    mutable: false,
                },
            );
        }
        MirPattern::Tuple(pats) => {
            for p in pats {
                collect_pattern_bindings(p, ctx);
            }
        }
        MirPattern::Struct { fields, .. } => {
            for (_, p) in fields {
                collect_pattern_bindings(p, ctx);
            }
        }
        MirPattern::Variant { fields, .. } => {
            for p in fields {
                collect_pattern_bindings(p, ctx);
            }
        }
        MirPattern::Wildcard
        | MirPattern::LitInt(_)
        | MirPattern::LitFloat(_)
        | MirPattern::LitBool(_)
        | MirPattern::LitString(_) => {}
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Walk for escape points
// ---------------------------------------------------------------------------

fn walk_body_for_escapes(body: &MirBody, ctx: &mut AnalysisCtx) {
    for stmt in &body.stmts {
        walk_stmt_for_escapes(stmt, ctx);
    }
    if let Some(result) = &body.result {
        // Body result is like a return — variables here escape the block.
        let vars = collect_var_refs(result);
        for v in vars {
            ctx.mark_escaping(&v, EscapeReason::ReturnedFromFn);
        }
    }
}

fn walk_stmt_for_escapes(stmt: &MirStmt, ctx: &mut AnalysisCtx) {
    match stmt {
        MirStmt::Return(Some(expr)) => {
            // Variables referenced in return expressions escape.
            let vars = collect_var_refs(expr);
            for v in vars {
                ctx.mark_escaping(&v, EscapeReason::ReturnedFromFn);
            }
            walk_expr_for_escapes(expr, ctx);
        }
        MirStmt::Return(None) => {}
        MirStmt::Let { init, .. } => {
            walk_expr_for_escapes(init, ctx);
        }
        MirStmt::Expr(e) => {
            walk_expr_for_escapes(e, ctx);
        }
        MirStmt::If { cond, then_body, else_body } => {
            walk_expr_for_escapes(cond, ctx);
            walk_body_for_escapes(then_body, ctx);
            if let Some(eb) = else_body {
                walk_body_for_escapes(eb, ctx);
            }
        }
        MirStmt::While { cond, body } => {
            walk_expr_for_escapes(cond, ctx);
            walk_body_for_escapes(body, ctx);
        }
        MirStmt::NoGcBlock(body) => {
            walk_body_for_escapes(body, ctx);
        }
        // Break/Continue have no sub-expressions.
        MirStmt::Break | MirStmt::Continue => {}
    }
}

fn walk_expr_for_escapes(expr: &MirExpr, ctx: &mut AnalysisCtx) {
    match &expr.kind {
        // Closure captures → escaping.
        MirExprKind::MakeClosure { captures, .. } => {
            for cap in captures {
                let vars = collect_var_refs(cap);
                for v in vars {
                    ctx.mark_escaping(&v, EscapeReason::CapturedByClosure);
                }
            }
        }

        // Container literals: elements that reference bindings → StoredInContainer.
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            for e in elems {
                let vars = collect_var_refs(e);
                for v in vars {
                    ctx.mark_escaping(&v, EscapeReason::StoredInContainer);
                }
                walk_expr_for_escapes(e, ctx);
            }
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, e) in fields {
                let vars = collect_var_refs(e);
                for v in vars {
                    ctx.mark_escaping(&v, EscapeReason::StoredInContainer);
                }
                walk_expr_for_escapes(e, ctx);
            }
        }
        MirExprKind::VariantLit { fields, .. } => {
            for e in fields {
                let vars = collect_var_refs(e);
                for v in vars {
                    ctx.mark_escaping(&v, EscapeReason::StoredInContainer);
                }
                walk_expr_for_escapes(e, ctx);
            }
        }

        // Call args: if callee is not a known-safe builtin, args may escape.
        MirExprKind::Call { callee, args } => {
            // Conservative: all args to unknown callees escape.
            let callee_name = extract_callee_name(callee);
            let is_known_safe = callee_name
                .as_ref()
                .map(|n| is_non_escaping_callee(n))
                .unwrap_or(false);

            if !is_known_safe {
                for arg in args {
                    let vars = collect_var_refs(arg);
                    for v in vars {
                        ctx.mark_escaping(&v, EscapeReason::PassedToUnknownFn);
                    }
                }
            }

            walk_expr_for_escapes(callee, ctx);
            for arg in args {
                walk_expr_for_escapes(arg, ctx);
            }
        }

        // Assignment to field/index: the root object escapes
        // (its contents are mutated, may be aliased).
        MirExprKind::Assign { target, value } => {
            // Check if target is a field/index access.
            if is_field_or_index_target(target) {
                let root_vars = collect_root_var(target);
                for v in root_vars {
                    ctx.mark_escaping(&v, EscapeReason::AssignedToFieldOrIndex);
                }
            }
            walk_expr_for_escapes(target, ctx);
            walk_expr_for_escapes(value, ctx);
        }

        // Recurse into sub-expressions.
        MirExprKind::Binary { left, right, .. } => {
            walk_expr_for_escapes(left, ctx);
            walk_expr_for_escapes(right, ctx);
        }
        MirExprKind::Unary { operand, .. } => {
            walk_expr_for_escapes(operand, ctx);
        }
        MirExprKind::Field { object, .. } => {
            walk_expr_for_escapes(object, ctx);
        }
        MirExprKind::Index { object, index } => {
            walk_expr_for_escapes(object, ctx);
            walk_expr_for_escapes(index, ctx);
        }
        MirExprKind::MultiIndex { object, indices } => {
            walk_expr_for_escapes(object, ctx);
            for i in indices {
                walk_expr_for_escapes(i, ctx);
            }
        }
        MirExprKind::Block(body) => {
            walk_body_for_escapes(body, ctx);
        }
        MirExprKind::If { cond, then_body, else_body } => {
            walk_expr_for_escapes(cond, ctx);
            walk_body_for_escapes(then_body, ctx);
            if let Some(eb) = else_body {
                walk_body_for_escapes(eb, ctx);
            }
        }
        MirExprKind::Match { scrutinee, arms } => {
            walk_expr_for_escapes(scrutinee, ctx);
            for arm in arms {
                walk_body_for_escapes(&arm.body, ctx);
            }
        }
        MirExprKind::Lambda { body, .. } => {
            walk_expr_for_escapes(body, ctx);
        }
        MirExprKind::TensorLit { rows } => {
            for row in rows {
                for e in row {
                    walk_expr_for_escapes(e, ctx);
                }
            }
        }
        MirExprKind::LinalgLU { operand }
        | MirExprKind::LinalgQR { operand }
        | MirExprKind::LinalgCholesky { operand }
        | MirExprKind::LinalgInv { operand } => {
            walk_expr_for_escapes(operand, ctx);
        }
        MirExprKind::Broadcast { operand, target_shape } => {
            walk_expr_for_escapes(operand, ctx);
            for s in target_shape {
                walk_expr_for_escapes(s, ctx);
            }
        }

        // Leaves — nothing to recurse into.
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::NaLit
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Var(_)
        | MirExprKind::Col(_)
        | MirExprKind::Void => {}
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Mark mutables
// ---------------------------------------------------------------------------

fn mark_mutables(body: &MirBody, ctx: &mut AnalysisCtx) {
    mark_mutables_body(body, ctx);
}

fn mark_mutables_body(body: &MirBody, ctx: &mut AnalysisCtx) {
    for stmt in &body.stmts {
        match stmt {
            MirStmt::Let { name, mutable: true, .. } => {
                ctx.mark_escaping(name, EscapeReason::Mutable);
            }
            MirStmt::If { then_body, else_body, .. } => {
                mark_mutables_body(then_body, ctx);
                if let Some(eb) = else_body {
                    mark_mutables_body(eb, ctx);
                }
            }
            MirStmt::While { body, .. } => {
                mark_mutables_body(body, ctx);
            }
            MirStmt::NoGcBlock(body) => {
                mark_mutables_body(body, ctx);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 4: Classify primitives
// ---------------------------------------------------------------------------

fn classify_primitives(body: &MirBody, ctx: &mut AnalysisCtx) {
    classify_primitives_body(body, ctx);
}

fn classify_primitives_body(body: &MirBody, ctx: &mut AnalysisCtx) {
    for stmt in &body.stmts {
        match stmt {
            MirStmt::Let { name, init, .. } => {
                if is_primitive_init(&classify_init_expr(init)) {
                    // Override: primitives are always Stack, even if "mutable".
                    if let Some(info) = ctx.bindings.get_mut(name) {
                        if info.reason == EscapeReason::Mutable
                            && is_primitive_init(&info.init_kind)
                        {
                            info.reason = EscapeReason::Primitive;
                        }
                    }
                }
            }
            MirStmt::If { then_body, else_body, .. } => {
                classify_primitives_body(then_body, ctx);
                if let Some(eb) = else_body {
                    classify_primitives_body(eb, ctx);
                }
            }
            MirStmt::While { body, .. } => {
                classify_primitives_body(body, ctx);
            }
            MirStmt::NoGcBlock(body) => {
                classify_primitives_body(body, ctx);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_primitive_init(kind: &InitKind) -> bool {
    matches!(kind, InitKind::Primitive)
}

/// Collect all variable names referenced in an expression (non-recursive into blocks).
fn collect_var_refs(expr: &MirExpr) -> Vec<String> {
    let mut vars = Vec::new();
    collect_var_refs_inner(expr, &mut vars);
    vars
}

fn collect_var_refs_inner(expr: &MirExpr, vars: &mut Vec<String>) {
    match &expr.kind {
        MirExprKind::Var(name) => vars.push(name.clone()),
        MirExprKind::Binary { left, right, .. } => {
            collect_var_refs_inner(left, vars);
            collect_var_refs_inner(right, vars);
        }
        MirExprKind::Unary { operand, .. } => {
            collect_var_refs_inner(operand, vars);
        }
        MirExprKind::Field { object, .. } => {
            collect_var_refs_inner(object, vars);
        }
        MirExprKind::Index { object, index } => {
            collect_var_refs_inner(object, vars);
            collect_var_refs_inner(index, vars);
        }
        MirExprKind::MultiIndex { object, indices } => {
            collect_var_refs_inner(object, vars);
            for i in indices {
                collect_var_refs_inner(i, vars);
            }
        }
        MirExprKind::Call { callee, args } => {
            collect_var_refs_inner(callee, vars);
            for a in args {
                collect_var_refs_inner(a, vars);
            }
        }
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            for e in elems {
                collect_var_refs_inner(e, vars);
            }
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, e) in fields {
                collect_var_refs_inner(e, vars);
            }
        }
        MirExprKind::VariantLit { fields, .. } => {
            for e in fields {
                collect_var_refs_inner(e, vars);
            }
        }
        MirExprKind::MakeClosure { captures, .. } => {
            for c in captures {
                collect_var_refs_inner(c, vars);
            }
        }
        MirExprKind::Assign { target, value } => {
            collect_var_refs_inner(target, vars);
            collect_var_refs_inner(value, vars);
        }
        MirExprKind::TensorLit { rows } => {
            for row in rows {
                for e in row {
                    collect_var_refs_inner(e, vars);
                }
            }
        }
        MirExprKind::LinalgLU { operand }
        | MirExprKind::LinalgQR { operand }
        | MirExprKind::LinalgCholesky { operand }
        | MirExprKind::LinalgInv { operand } => {
            collect_var_refs_inner(operand, vars);
        }
        MirExprKind::Broadcast { operand, target_shape } => {
            collect_var_refs_inner(operand, vars);
            for s in target_shape {
                collect_var_refs_inner(s, vars);
            }
        }
        // Don't recurse into blocks/if/match — those have their own scope.
        MirExprKind::Block(_)
        | MirExprKind::If { .. }
        | MirExprKind::Match { .. }
        | MirExprKind::Lambda { .. } => {}
        // Leaves.
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::NaLit
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Col(_)
        | MirExprKind::Void => {}
    }
}

/// Extract the callee name from a Call expression.
fn extract_callee_name(callee: &MirExpr) -> Option<String> {
    match &callee.kind {
        MirExprKind::Var(name) => Some(name.clone()),
        MirExprKind::Field { object, name } => {
            // Method call: extract "Type.method" or just "method".
            if let Some(obj_name) = extract_callee_name(object) {
                Some(format!("{}.{}", obj_name, name))
            } else {
                Some(name.clone())
            }
        }
        _ => None,
    }
}

/// Check if a callee is known to not leak its arguments (pure/safe builtins).
///
/// Returns true if the given callee is known to NOT capture/store its arguments.
/// Backed by the effect registry (single source of truth).
/// Unknown builtins are conservatively treated as escaping.
fn is_non_escaping_callee(name: &str) -> bool {
    !cjc_types::effect_registry::may_capture(name)
}

/// Check if an assign target is a field or index access (not a simple var).
fn is_field_or_index_target(target: &MirExpr) -> bool {
    matches!(
        &target.kind,
        MirExprKind::Field { .. }
            | MirExprKind::Index { .. }
            | MirExprKind::MultiIndex { .. }
    )
}

/// Extract the root variable name from a field/index chain.
/// e.g., `a.b.c[i]` → `["a"]`
fn collect_root_var(expr: &MirExpr) -> Vec<String> {
    match &expr.kind {
        MirExprKind::Var(name) => vec![name.clone()],
        MirExprKind::Field { object, .. } => collect_root_var(object),
        MirExprKind::Index { object, .. } => collect_root_var(object),
        MirExprKind::MultiIndex { object, .. } => collect_root_var(object),
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// Annotation pass: write AllocHint into MirStmt::Let
// ---------------------------------------------------------------------------

fn annotate_body(body: &mut MirBody, info: &EscapeInfo) {
    for stmt in &mut body.stmts {
        annotate_stmt(stmt, info);
    }
}

fn annotate_stmt(stmt: &mut MirStmt, info: &EscapeInfo) {
    match stmt {
        MirStmt::Let { name, alloc_hint, .. } => {
            if let Some((hint, _)) = info.bindings.get(name.as_str()) {
                *alloc_hint = Some(*hint);
            }
        }
        MirStmt::If { then_body, else_body, .. } => {
            annotate_body(then_body, info);
            if let Some(eb) = else_body {
                annotate_body(eb, info);
            }
        }
        MirStmt::While { body, .. } => {
            annotate_body(body, info);
        }
        MirStmt::NoGcBlock(body) => {
            annotate_body(body, info);
        }
        MirStmt::Return(_) | MirStmt::Expr(_) => {}
        MirStmt::Break | MirStmt::Continue => {}
    }
}

// ---------------------------------------------------------------------------
// NoGC integration: check if any binding in a function uses Rc or Arena
// ---------------------------------------------------------------------------

/// Check if a function's escape analysis shows any heap allocation.
/// Used by the strengthened @no_gc verifier.
pub fn has_heap_alloc(info: &EscapeInfo) -> bool {
    info.bindings.values().any(|(hint, _)| {
        matches!(hint, AllocHint::Rc | AllocHint::Arena)
    })
}

/// Get all bindings that allocate in a function, with their reasons.
/// Used for @no_gc error reporting.
pub fn heap_alloc_bindings(info: &EscapeInfo) -> Vec<(String, AllocHint, EscapeReason)> {
    info.bindings
        .iter()
        .filter(|(_, (hint, _))| matches!(hint, AllocHint::Rc | AllocHint::Arena))
        .map(|(name, (hint, reason))| (name.clone(), *hint, *reason))
        .collect()
}

// ---------------------------------------------------------------------------
// Debug tracing support
// ---------------------------------------------------------------------------

/// Format escape info for --trace-alloc output.
pub fn format_trace(func_name: &str, info: &EscapeInfo) -> Vec<String> {
    let mut lines = Vec::new();
    let mut sorted: Vec<_> = info.bindings.iter().collect();
    sorted.sort_by_key(|(name, _)| (*name).clone());
    for (name, (hint, reason)) in sorted {
        lines.push(format!(
            "[alloc] {func_name}::{name}: {:?} reason={:?}",
            hint, reason
        ));
    }
    lines
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn mk_var(name: &str) -> MirExpr {
        MirExpr { kind: MirExprKind::Var(name.to_string()) }
    }

    fn mk_int(v: i64) -> MirExpr {
        MirExpr { kind: MirExprKind::IntLit(v) }
    }

    fn mk_string(s: &str) -> MirExpr {
        MirExpr { kind: MirExprKind::StringLit(s.to_string()) }
    }

    fn mk_call(callee: &str, args: Vec<MirExpr>) -> MirExpr {
        MirExpr {
            kind: MirExprKind::Call {
                callee: Box::new(mk_var(callee)),
                args,
            },
        }
    }

    fn mk_func(name: &str, stmts: Vec<MirStmt>) -> MirFunction {
        MirFunction {
            id: MirFnId(0),
            name: name.to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: MirBody {
                stmts,
                result: None,
            },
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: cjc_ast::Visibility::Private,
        }
    }

    // -- Primitive classification -------------------------------------------

    #[test]
    fn test_primitive_int_is_stack() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "x".to_string(),
                mutable: false,
                init: mk_int(42),
                alloc_hint: None,
            },
        ]);
        let info = analyze_function(&func);
        assert_eq!(info.bindings["x"], (AllocHint::Stack, EscapeReason::Primitive));
    }

    #[test]
    fn test_mutable_primitive_still_stack() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "x".to_string(),
                mutable: true,
                init: mk_int(0),
                alloc_hint: None,
            },
        ]);
        let info = analyze_function(&func);
        // Mutable primitive → still Stack (primitives override Mutable).
        assert_eq!(info.bindings["x"], (AllocHint::Stack, EscapeReason::Primitive));
    }

    // -- Non-escaping string → Arena ----------------------------------------

    #[test]
    fn test_non_escaping_string_is_arena() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: false,
                init: mk_string("hello"),
                alloc_hint: None,
            },
            MirStmt::Expr(mk_call("print", vec![mk_var("s")])),
        ]);
        let info = analyze_function(&func);
        assert_eq!(info.bindings["s"], (AllocHint::Arena, EscapeReason::NonEscaping));
    }

    // -- Return escapes → Rc ------------------------------------------------

    #[test]
    fn test_returned_string_is_rc() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: false,
                init: mk_string("hello"),
                alloc_hint: None,
            },
            MirStmt::Return(Some(mk_var("s"))),
        ]);
        let info = analyze_function(&func);
        assert_eq!(info.bindings["s"], (AllocHint::Rc, EscapeReason::ReturnedFromFn));
    }

    // -- Closure capture → Rc -----------------------------------------------

    #[test]
    fn test_closure_capture_is_rc() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "x".to_string(),
                mutable: false,
                init: mk_string("captured"),
                alloc_hint: None,
            },
            MirStmt::Expr(MirExpr {
                kind: MirExprKind::MakeClosure {
                    fn_name: "__closure_0".to_string(),
                    captures: vec![mk_var("x")],
                },
            }),
        ]);
        let info = analyze_function(&func);
        assert_eq!(info.bindings["x"], (AllocHint::Rc, EscapeReason::CapturedByClosure));
    }

    // -- Stored in container → Rc -------------------------------------------

    #[test]
    fn test_stored_in_array_is_rc() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: false,
                init: mk_string("item"),
                alloc_hint: None,
            },
            MirStmt::Expr(MirExpr {
                kind: MirExprKind::ArrayLit(vec![mk_var("s")]),
            }),
        ]);
        let info = analyze_function(&func);
        assert_eq!(info.bindings["s"], (AllocHint::Rc, EscapeReason::StoredInContainer));
    }

    // -- Unknown function call → Rc -----------------------------------------

    #[test]
    fn test_passed_to_unknown_fn_is_rc() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: false,
                init: mk_string("data"),
                alloc_hint: None,
            },
            MirStmt::Expr(mk_call("unknown_fn", vec![mk_var("s")])),
        ]);
        let info = analyze_function(&func);
        assert_eq!(info.bindings["s"], (AllocHint::Rc, EscapeReason::PassedToUnknownFn));
    }

    // -- Known-safe callee → not escaping -----------------------------------

    #[test]
    fn test_passed_to_print_stays_arena() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: false,
                init: mk_string("data"),
                alloc_hint: None,
            },
            MirStmt::Expr(mk_call("print", vec![mk_var("s")])),
        ]);
        let info = analyze_function(&func);
        assert_eq!(info.bindings["s"], (AllocHint::Arena, EscapeReason::NonEscaping));
    }

    // -- Mutable string → Rc (conservative) ---------------------------------

    #[test]
    fn test_mutable_string_is_rc() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: true,
                init: mk_string("mutable"),
                alloc_hint: None,
            },
        ]);
        let info = analyze_function(&func);
        assert_eq!(info.bindings["s"], (AllocHint::Rc, EscapeReason::Mutable));
    }

    // -- Program-level analysis ---------------------------------------------

    #[test]
    fn test_analyze_program() {
        let program = MirProgram {
            functions: vec![
                mk_func("f1", vec![
                    MirStmt::Let {
                        name: "x".to_string(),
                        mutable: false,
                        init: mk_int(1),
                        alloc_hint: None,
                    },
                ]),
                mk_func("f2", vec![
                    MirStmt::Let {
                        name: "s".to_string(),
                        mutable: false,
                        init: mk_string("hello"),
                        alloc_hint: None,
                    },
                    MirStmt::Return(Some(mk_var("s"))),
                ]),
            ],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        };
        let result = analyze_program(&program);
        assert_eq!(result["f1"].bindings["x"].0, AllocHint::Stack);
        assert_eq!(result["f2"].bindings["s"].0, AllocHint::Rc);
    }

    // -- has_heap_alloc utility ---------------------------------------------

    #[test]
    fn test_has_heap_alloc_pure_stack() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "x".to_string(),
                mutable: false,
                init: mk_int(1),
                alloc_hint: None,
            },
            MirStmt::Let {
                name: "y".to_string(),
                mutable: true,
                init: MirExpr { kind: MirExprKind::FloatLit(3.14) },
                alloc_hint: None,
            },
        ]);
        let info = analyze_function(&func);
        assert!(!has_heap_alloc(&info), "pure primitives should not heap-allocate");
    }

    #[test]
    fn test_has_heap_alloc_with_string() {
        let func = mk_func("f", vec![
            MirStmt::Let {
                name: "s".to_string(),
                mutable: false,
                init: mk_string("hello"),
                alloc_hint: None,
            },
        ]);
        let info = analyze_function(&func);
        assert!(has_heap_alloc(&info), "non-escaping string still uses Arena");
    }

    // -- format_trace -------------------------------------------------------

    #[test]
    fn test_format_trace() {
        let func = mk_func("test_fn", vec![
            MirStmt::Let {
                name: "x".to_string(),
                mutable: false,
                init: mk_int(42),
                alloc_hint: None,
            },
        ]);
        let info = analyze_function(&func);
        let lines = format_trace("test_fn", &info);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("test_fn::x"));
        assert!(lines[0].contains("Stack"));
    }
}
