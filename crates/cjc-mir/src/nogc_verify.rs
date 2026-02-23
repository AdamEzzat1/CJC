//! NoGC Static Verifier (Stage 2.4)
//!
//! Scans MIR for every `is_nogc` function and rejects:
//! - Any call to `gc_alloc` (direct GC allocation)
//! - Any call to a function that may transitively trigger GC
//! - Any call to an unknown/external/unresolved function (conservative)
//! - Any indirect call (closure/lambda) unless proven safe
//!
//! The verifier builds a conservative call graph, computes a `may_gc` effect
//! flag for each function via fixpoint iteration, and then checks that no
//! `is_nogc` function calls anything with `may_gc == true`.

use std::collections::{HashMap, HashSet};

use crate::{MirBody, MirExpr, MirExprKind, MirProgram, MirStmt};

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// A single NoGC verification error.
#[derive(Debug, Clone)]
pub struct NoGcError {
    /// Name of the `is_nogc` function that violated the constraint.
    pub function: String,
    /// Description of the offending operation.
    pub reason: String,
    /// If the violation is transitive, this is the minimal call chain.
    pub call_chain: Vec<String>,
}

impl std::fmt::Display for NoGcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nogc violation in `{}`: {}",
            self.function, self.reason
        )?;
        if !self.call_chain.is_empty() {
            write!(f, " (via {})", self.call_chain.join(" -> "))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Known GC builtins
// ---------------------------------------------------------------------------

/// Functions that are known to perform GC allocation or trigger GC.
fn is_gc_builtin(name: &str) -> bool {
    matches!(name, "gc_alloc" | "gc_collect")
}

/// Functions that are known builtins and do NOT trigger GC.
fn is_safe_builtin(name: &str) -> bool {
    matches!(
        name,
        "print"
            | "Tensor.zeros"
            | "Tensor.ones"
            | "Tensor.randn"
            | "Tensor.from_vec"
            | "matmul"
            | "Buffer.alloc"
            | "len"
            | "push"
            | "assert"
            | "assert_eq"
            | "clock"
            | "gc_live_count"
            // Linalg builtins (pure math, no GC)
            | "linalg.lu"
            | "linalg.qr"
            | "linalg.cholesky"
            | "linalg.inv"
            // Tensor views (no GC)
            | "Tensor.slice"
            | "Tensor.transpose"
            | "Tensor.broadcast_to"
            // Sparse tensor ops (no GC)
            | "SparseCsr.matvec"
            | "SparseCsr.to_dense"
            | "SparseCoo.to_csr"
            // Transformer kernels (pure math, no GC)
            | "attention"
            | "Tensor.softmax"
            | "Tensor.layer_norm"
            | "Tensor.relu"
            | "Tensor.gelu"
            | "Tensor.bmm"
            | "Tensor.linear"
            | "Tensor.transpose_last_two"
            // Phase 6: CNN Signal Processing (pure math, no GC)
            | "Tensor.conv1d"
            // Phase 7: 2D Spatial Vision (pure math, BinnedAccumulator inner loop, no GC)
            | "Tensor.conv2d"
            | "Tensor.maxpool2d"
            // Milestone 2.7: Deterministic summation (pure math, stack-only, no GC)
            | "Tensor.binned_sum"
            // Milestone 2.7 Expansion: Complex BLAS (fixed-sequence, stack-only, no GC)
            | "Complex.re" | "Complex.im" | "Complex.abs" | "Complex.conj"
            | "Complex.norm_sq" | "Complex.add" | "Complex.mul"
            // Milestone 2.7 Expansion: F16 (stack-only, no GC)
            | "F16.to_f64" | "F16.to_f32"
            // Phase 3: Zero-Copy and Multi-Head (pure math, no GC)
            | "Tensor.from_bytes"
            | "Tensor.split_heads"
            | "Tensor.merge_heads"
            | "Tensor.view_reshape"
            | "ByteSlice.as_tensor"
            // KV-Cache Scratchpad (pre-allocated, no GC)
            | "Scratchpad.new"
            | "Scratchpad.append"
            | "Scratchpad.append_tensor"
            | "Scratchpad.as_tensor"
            | "Scratchpad.len"
            | "Scratchpad.capacity"
            | "Scratchpad.dim"
            | "Scratchpad.clear"
            | "Scratchpad.is_empty"
            // Phase 4: Block-Paged KV-Cache (pre-allocated, no GC)
            | "PagedKvCache.new"
            | "PagedKvCache.append"
            | "PagedKvCache.append_tensor"
            | "PagedKvCache.as_tensor"
            | "PagedKvCache.clear"
            | "PagedKvCache.len"
            | "PagedKvCache.is_empty"
            | "PagedKvCache.max_tokens"
            | "PagedKvCache.dim"
            | "PagedKvCache.num_blocks"
            | "PagedKvCache.blocks_in_use"
            | "PagedKvCache.get_token"
            // Phase 4: Aligned ByteSlice (one-time copy fallback, no GC)
            | "AlignedByteSlice.from_bytes"
            | "AlignedByteSlice.as_tensor"
            | "AlignedByteSlice.was_realigned"
            | "AlignedByteSlice.len"
            | "AlignedByteSlice.is_empty"
            // Deterministic Map (nogc variant)
            | "Map.new"
            | "Map.insert"
            | "Map.get"
            | "Map.remove"
            | "Map.len"
            | "Map.contains_key"
            | "Map.keys"
            | "Map.values"
            // Phase 10: Tidy Primitives (view ops produce bitmask/projection, no GC heap)
            // filter and select return TidyView — O(N/64) bitmask or O(K) index alloc only.
            // materialize/to_tensor trigger column buffer allocation (allowed outside @nogc).
            | "tidy_filter"
            | "tidy_select"
            | "tidy_mask_and"
            | "tidy_nrows"
            | "tidy_ncols"
            | "tidy_column_names"
            // Phase 11: Grouping (GroupIndex = Vec<Vec<usize>>, no column alloc)
            | "tidy_group_by"
            | "tidy_ungroup"
            | "tidy_ngroups"
            // Phase 11: Slice/distinct (RowIndexMap = Vec<usize>, no column alloc)
            | "tidy_slice"
            | "tidy_slice_head"
            | "tidy_slice_tail"
            | "tidy_slice_sample"
            | "tidy_distinct"
            // Phase 12: Semi/anti join (RowIndexMap only, no column alloc)
            | "tidy_semi_join"
            | "tidy_anti_join"
            // Phase 13-16: View-only ops (ProjectionMap update only, no column alloc)
            | "tidy_relocate"
            | "tidy_drop_cols"
            // Phase 16: Group perf upgrade (BTree-accelerated GroupIndex, no column alloc)
            | "tidy_group_by_fast"
            // Phase 17: Categorical / forcats
            // fct_collapse is metadata-only: rewrites levels Vec + remaps data Vec
            // entirely on Rust heap (Vec<String>, Vec<u16>) — no GC heap involved.
            // O(L) for levels rewrite + O(N) for data remap, but no column buffer alloc.
            | "fct_collapse"
            // Materialising ops (column buffer alloc) — NOT safe inside @nogc:
            // tidy_arrange, tidy_summarise, tidy_inner_join, tidy_left_join,
            // tidy_pivot_longer, tidy_pivot_wider, tidy_bind_rows, tidy_bind_cols,
            // tidy_mutate_across, tidy_right_join, tidy_full_join,
            // tidy_inner_join_typed, tidy_left_join_typed, tidy_summarise_across,
            // tidy_rename (rebuilds base DataFrame)
            // Phase 17 materialising (intentionally absent):
            //   fct_encode  : allocates Vec<u16> + Vec<String> levels
            //   fct_lump    : allocates new levels Vec + new data Vec
            //   fct_reorder : allocates new levels Vec + new data Vec
            // are intentionally absent from this list.
    )
}

// ---------------------------------------------------------------------------
// Call graph construction
// ---------------------------------------------------------------------------

/// Collected information about calls in a function body.
struct FnCallInfo {
    /// Direct calls by name.
    direct_calls: HashSet<String>,
    /// Whether the function has any indirect calls (closures, higher-order).
    has_indirect_call: bool,
    /// Whether the function directly calls a GC builtin.
    has_gc_builtin: bool,
    /// Whether there is a call inside a NoGcBlock statement.
    nogc_block_calls: Vec<String>,
    /// Whether there is a gc builtin call inside a NoGcBlock.
    nogc_block_gc_builtins: Vec<String>,
    /// Whether there is an indirect call inside a NoGcBlock.
    nogc_block_has_indirect: bool,
}

fn collect_calls_body(body: &MirBody, in_nogc_block: bool, info: &mut FnCallInfo) {
    for stmt in &body.stmts {
        collect_calls_stmt(stmt, in_nogc_block, info);
    }
    if let Some(ref expr) = body.result {
        collect_calls_expr(expr, in_nogc_block, info);
    }
}

fn collect_calls_stmt(stmt: &MirStmt, in_nogc_block: bool, info: &mut FnCallInfo) {
    match stmt {
        MirStmt::Let { init, .. } => collect_calls_expr(init, in_nogc_block, info),
        MirStmt::Expr(expr) => collect_calls_expr(expr, in_nogc_block, info),
        MirStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            collect_calls_expr(cond, in_nogc_block, info);
            collect_calls_body(then_body, in_nogc_block, info);
            if let Some(eb) = else_body {
                collect_calls_body(eb, in_nogc_block, info);
            }
        }
        MirStmt::While { cond, body } => {
            collect_calls_expr(cond, in_nogc_block, info);
            collect_calls_body(body, in_nogc_block, info);
        }
        MirStmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                collect_calls_expr(expr, in_nogc_block, info);
            }
        }
        MirStmt::NoGcBlock(body) => {
            // Everything inside a NoGcBlock is treated as in_nogc context.
            collect_calls_body(body, true, info);
        }
    }
}

fn collect_calls_expr(expr: &MirExpr, in_nogc_block: bool, info: &mut FnCallInfo) {
    match &expr.kind {
        MirExprKind::Call { callee, args } => {
            // Determine call target
            match &callee.kind {
                MirExprKind::Var(name) => {
                    info.direct_calls.insert(name.clone());
                    if in_nogc_block {
                        info.nogc_block_calls.push(name.clone());
                        if is_gc_builtin(name) {
                            info.nogc_block_gc_builtins.push(name.clone());
                        }
                    }
                    if is_gc_builtin(name) {
                        info.has_gc_builtin = true;
                    }
                }
                MirExprKind::Field { object, name } => {
                    // Static method: Tensor.zeros etc.
                    if let MirExprKind::Var(obj_name) = &object.kind {
                        let qualified = format!("{obj_name}.{name}");
                        info.direct_calls.insert(qualified.clone());
                        if in_nogc_block {
                            info.nogc_block_calls.push(qualified);
                        }
                    } else {
                        // Method on computed object - conservative as indirect
                        info.has_indirect_call = true;
                        if in_nogc_block {
                            info.nogc_block_has_indirect = true;
                        }
                    }
                    collect_calls_expr(object, in_nogc_block, info);
                }
                _ => {
                    // Indirect call (closure, higher-order)
                    info.has_indirect_call = true;
                    if in_nogc_block {
                        info.nogc_block_has_indirect = true;
                    }
                    collect_calls_expr(callee, in_nogc_block, info);
                }
            }
            for arg in args {
                collect_calls_expr(arg, in_nogc_block, info);
            }
        }
        MirExprKind::Binary { left, right, .. } => {
            collect_calls_expr(left, in_nogc_block, info);
            collect_calls_expr(right, in_nogc_block, info);
        }
        MirExprKind::Unary { operand, .. } => {
            collect_calls_expr(operand, in_nogc_block, info);
        }
        MirExprKind::Field { object, .. } => {
            collect_calls_expr(object, in_nogc_block, info);
        }
        MirExprKind::Index { object, index } => {
            collect_calls_expr(object, in_nogc_block, info);
            collect_calls_expr(index, in_nogc_block, info);
        }
        MirExprKind::MultiIndex { object, indices } => {
            collect_calls_expr(object, in_nogc_block, info);
            for idx in indices {
                collect_calls_expr(idx, in_nogc_block, info);
            }
        }
        MirExprKind::Assign { target, value } => {
            collect_calls_expr(target, in_nogc_block, info);
            collect_calls_expr(value, in_nogc_block, info);
        }
        MirExprKind::Block(body) => {
            collect_calls_body(body, in_nogc_block, info);
        }
        MirExprKind::StructLit { fields, .. } => {
            for (_, fexpr) in fields {
                collect_calls_expr(fexpr, in_nogc_block, info);
            }
        }
        MirExprKind::ArrayLit(elems) | MirExprKind::TupleLit(elems) => {
            for e in elems {
                collect_calls_expr(e, in_nogc_block, info);
            }
        }
        MirExprKind::MakeClosure { captures, .. } => {
            for cap in captures {
                collect_calls_expr(cap, in_nogc_block, info);
            }
        }
        MirExprKind::If {
            cond,
            then_body,
            else_body,
        } => {
            collect_calls_expr(cond, in_nogc_block, info);
            collect_calls_body(then_body, in_nogc_block, info);
            if let Some(eb) = else_body {
                collect_calls_body(eb, in_nogc_block, info);
            }
        }
        MirExprKind::Match { scrutinee, arms } => {
            collect_calls_expr(scrutinee, in_nogc_block, info);
            for arm in arms {
                collect_calls_body(&arm.body, in_nogc_block, info);
            }
        }
        MirExprKind::Lambda { body, .. } => {
            collect_calls_expr(body, in_nogc_block, info);
        }
        // Linalg + broadcast: pure math, no GC
        MirExprKind::LinalgLU { operand }
        | MirExprKind::LinalgQR { operand }
        | MirExprKind::LinalgCholesky { operand }
        | MirExprKind::LinalgInv { operand } => {
            collect_calls_expr(operand, in_nogc_block, info);
        }
        MirExprKind::Broadcast { operand, target_shape } => {
            collect_calls_expr(operand, in_nogc_block, info);
            for s in target_shape {
                collect_calls_expr(s, in_nogc_block, info);
            }
        }
        // Enum variant construction: recurse into field sub-expressions (no GC)
        MirExprKind::VariantLit { fields, .. } => {
            for f in fields {
                collect_calls_expr(f, in_nogc_block, info);
            }
        }
        // Leaves: no sub-expressions
        MirExprKind::IntLit(_)
        | MirExprKind::FloatLit(_)
        | MirExprKind::BoolLit(_)
        | MirExprKind::StringLit(_)
        | MirExprKind::ByteStringLit(_)
        | MirExprKind::ByteCharLit(_)
        | MirExprKind::RawStringLit(_)
        | MirExprKind::RawByteStringLit(_)
        | MirExprKind::RegexLit { .. }
        | MirExprKind::Var(_)
        | MirExprKind::Col(_)
        | MirExprKind::Void => {}
        MirExprKind::TensorLit { rows } => {
            for row in rows {
                for elem in row {
                    collect_calls_expr(elem, in_nogc_block, info);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fixpoint: compute may_gc for all functions
// ---------------------------------------------------------------------------

/// Compute `may_gc` effect for every function in the program.
/// Returns a map from function name to may_gc flag.
fn compute_may_gc(
    program: &MirProgram,
) -> (
    HashMap<String, bool>,
    HashMap<String, FnCallInfo>,
) {
    // Step 1: Collect call info for each function.
    let mut call_infos: HashMap<String, FnCallInfo> = HashMap::new();

    for func in &program.functions {
        let mut info = FnCallInfo {
            direct_calls: HashSet::new(),
            has_indirect_call: false,
            has_gc_builtin: false,
            nogc_block_calls: Vec::new(),
            nogc_block_gc_builtins: Vec::new(),
            nogc_block_has_indirect: false,
        };
        collect_calls_body(&func.body, func.is_nogc, &mut info);
        call_infos.insert(func.name.clone(), info);
    }

    // Step 2: Seed may_gc.
    let mut may_gc: HashMap<String, bool> = HashMap::new();

    // All GC builtins are may_gc = true.
    may_gc.insert("gc_alloc".to_string(), true);
    may_gc.insert("gc_collect".to_string(), true);

    // Safe builtins are may_gc = false.
    // This list must stay in sync with `is_safe_builtin` above.
    for &name in &[
        "print",
        "Tensor.zeros",
        "Tensor.ones",
        "Tensor.randn",
        "Tensor.from_vec",
        "matmul",
        "Buffer.alloc",
        "len",
        "push",
        "assert",
        "assert_eq",
        "clock",
        "gc_live_count",
        // Linalg
        "linalg.lu", "linalg.qr", "linalg.cholesky", "linalg.inv",
        // Tensor views
        "Tensor.slice", "Tensor.transpose", "Tensor.broadcast_to",
        // Sparse
        "SparseCsr.matvec", "SparseCsr.to_dense", "SparseCoo.to_csr",
        // Transformer kernels
        "attention", "Tensor.softmax", "Tensor.layer_norm",
        "Tensor.relu", "Tensor.gelu", "Tensor.bmm", "Tensor.linear",
        "Tensor.transpose_last_two",
        // Phase 6: CNN Signal Processing
        "Tensor.conv1d",
        // Phase 7: 2D Spatial Vision
        "Tensor.conv2d", "Tensor.maxpool2d",
        // Milestone 2.7
        "Tensor.binned_sum",
        "Complex.re", "Complex.im", "Complex.abs", "Complex.conj",
        "Complex.norm_sq", "Complex.add", "Complex.mul",
        "F16.to_f64", "F16.to_f32",
        // Phase 3: Zero-Copy
        "Tensor.from_bytes", "Tensor.split_heads", "Tensor.merge_heads",
        "Tensor.view_reshape", "ByteSlice.as_tensor",
        // KV-Cache
        "Scratchpad.new", "Scratchpad.append", "Scratchpad.append_tensor",
        "Scratchpad.as_tensor", "Scratchpad.len", "Scratchpad.capacity",
    ] {
        may_gc.insert(name.to_string(), false);
    }

    // Seed user functions.
    for func in &program.functions {
        let info = &call_infos[&func.name];
        // Conservative: indirect calls => may_gc
        let initial = info.has_gc_builtin || info.has_indirect_call;
        may_gc.insert(func.name.clone(), initial);
    }

    // Step 3: Fixpoint iteration.
    let mut changed = true;
    while changed {
        changed = false;
        for func in &program.functions {
            if *may_gc.get(&func.name).unwrap_or(&false) {
                continue; // already true, can't go back
            }
            let info = &call_infos[&func.name];
            for callee in &info.direct_calls {
                // Unknown function (not a builtin, not in program) => conservative true
                let callee_may_gc = may_gc.get(callee).copied().unwrap_or(true);
                if callee_may_gc {
                    may_gc.insert(func.name.clone(), true);
                    changed = true;
                    break;
                }
            }
        }
    }

    (may_gc, call_infos)
}

// ---------------------------------------------------------------------------
// Build minimal call chain (best effort)
// ---------------------------------------------------------------------------

fn find_gc_chain(
    fn_name: &str,
    may_gc_map: &HashMap<String, bool>,
    call_infos: &HashMap<String, FnCallInfo>,
    visited: &mut HashSet<String>,
) -> Vec<String> {
    if visited.contains(fn_name) {
        return vec![];
    }
    visited.insert(fn_name.to_string());

    if is_gc_builtin(fn_name) {
        return vec![fn_name.to_string()];
    }

    if let Some(info) = call_infos.get(fn_name) {
        for callee in &info.direct_calls {
            if may_gc_map.get(callee).copied().unwrap_or(true) {
                let mut chain = vec![callee.clone()];
                let sub = find_gc_chain(callee, may_gc_map, call_infos, visited);
                chain.extend(sub);
                return chain;
            }
        }
    }

    vec![]
}

// ---------------------------------------------------------------------------
// Public API: verify
// ---------------------------------------------------------------------------

/// Run the NoGC static verifier on a MIR program.
///
/// Returns `Ok(())` if all `is_nogc` functions are clean, or a list of errors.
pub fn verify_nogc(program: &MirProgram) -> Result<(), Vec<NoGcError>> {
    let (may_gc_map, call_infos) = compute_may_gc(program);
    let mut errors = Vec::new();

    for func in &program.functions {
        if !func.is_nogc {
            // Also check NoGcBlock statements inside non-nogc functions.
            if let Some(info) = call_infos.get(&func.name) {
                // Check GC builtins inside NoGcBlock.
                for gc_call in &info.nogc_block_gc_builtins {
                    errors.push(NoGcError {
                        function: func.name.clone(),
                        reason: format!(
                            "call to GC builtin `{gc_call}` inside nogc block"
                        ),
                        call_chain: vec![],
                    });
                }
                // Check calls to may_gc functions inside NoGcBlock.
                for callee in &info.nogc_block_calls {
                    if is_gc_builtin(callee) {
                        continue; // already reported above
                    }
                    let callee_may_gc = may_gc_map.get(callee).copied().unwrap_or(true);
                    if callee_may_gc {
                        let mut visited = HashSet::new();
                        let chain = find_gc_chain(callee, &may_gc_map, &call_infos, &mut visited);
                        errors.push(NoGcError {
                            function: func.name.clone(),
                            reason: format!(
                                "call to `{callee}` (may_gc) inside nogc block"
                            ),
                            call_chain: chain,
                        });
                    }
                }
                // Indirect calls inside NoGcBlock.
                if info.nogc_block_has_indirect {
                    errors.push(NoGcError {
                        function: func.name.clone(),
                        reason: "indirect call inside nogc block (conservative rejection)"
                            .to_string(),
                        call_chain: vec![],
                    });
                }
            }
            continue;
        }

        // This is a nogc function: check everything in the body.
        let info = match call_infos.get(&func.name) {
            Some(i) => i,
            None => continue,
        };

        // Check for direct GC builtins.
        if info.has_gc_builtin {
            for callee in &info.direct_calls {
                if is_gc_builtin(callee) {
                    errors.push(NoGcError {
                        function: func.name.clone(),
                        reason: format!("direct call to GC builtin `{callee}`"),
                        call_chain: vec![],
                    });
                }
            }
        }

        // Check for indirect calls.
        if info.has_indirect_call {
            errors.push(NoGcError {
                function: func.name.clone(),
                reason: "indirect call in nogc function (conservative rejection)".to_string(),
                call_chain: vec![],
            });
        }

        // Check for calls to may_gc functions.
        for callee in &info.direct_calls {
            if is_gc_builtin(callee) {
                continue; // already reported
            }
            if is_safe_builtin(callee) {
                continue; // safe builtins never GC — don't report as may_gc
            }
            let callee_may_gc = may_gc_map.get(callee).copied().unwrap_or(true);
            if callee_may_gc {
                let mut visited = HashSet::new();
                let chain = find_gc_chain(callee, &may_gc_map, &call_infos, &mut visited);
                errors.push(NoGcError {
                    function: func.name.clone(),
                    reason: format!("call to `{callee}` which may trigger GC"),
                    call_chain: chain,
                });
            }
        }

        // Check for calls to unknown/external functions (not in program, not builtin).
        for callee in &info.direct_calls {
            if is_gc_builtin(callee) || is_safe_builtin(callee) {
                continue;
            }
            let is_user_fn = program.functions.iter().any(|f| f.name == *callee);
            if !is_user_fn && !may_gc_map.contains_key(callee) {
                errors.push(NoGcError {
                    function: func.name.clone(),
                    reason: format!("call to unknown/external function `{callee}`"),
                    call_chain: vec![],
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn mk_expr(kind: MirExprKind) -> MirExpr {
        MirExpr { kind }
    }

    fn mk_call(name: &str, args: Vec<MirExpr>) -> MirExpr {
        mk_expr(MirExprKind::Call {
            callee: Box::new(mk_expr(MirExprKind::Var(name.to_string()))),
            args,
        })
    }

    fn mk_fn(name: &str, is_nogc: bool, stmts: Vec<MirStmt>) -> MirFunction {
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
            is_nogc,
        }
    }

    fn mk_program(functions: Vec<MirFunction>) -> MirProgram {
        MirProgram {
            functions,
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    #[test]
    fn test_nogc_clean_function_passes() {
        let program = mk_program(vec![
            mk_fn("pure_add", true, vec![
                MirStmt::Expr(mk_call("print", vec![
                    mk_expr(MirExprKind::StringLit("hello".to_string())),
                ])),
            ]),
        ]);
        assert!(verify_nogc(&program).is_ok());
    }

    #[test]
    fn test_nogc_direct_gc_alloc_rejected() {
        let program = mk_program(vec![
            mk_fn("bad_fn", true, vec![
                MirStmt::Expr(mk_call("gc_alloc", vec![])),
            ]),
        ]);
        let errors = verify_nogc(&program).unwrap_err();
        assert!(!errors.is_empty());
        assert!(errors[0].reason.contains("gc_alloc"));
    }

    #[test]
    fn test_nogc_transitive_gc_rejected() {
        let program = mk_program(vec![
            mk_fn("allocator", false, vec![
                MirStmt::Expr(mk_call("gc_alloc", vec![])),
            ]),
            mk_fn("caller", true, vec![
                MirStmt::Expr(mk_call("allocator", vec![])),
            ]),
        ]);
        let errors = verify_nogc(&program).unwrap_err();
        assert!(!errors.is_empty());
        assert!(errors[0].reason.contains("allocator"));
    }

    #[test]
    fn test_nogc_unknown_function_rejected() {
        let program = mk_program(vec![
            mk_fn("caller", true, vec![
                MirStmt::Expr(mk_call("unknown_fn", vec![])),
            ]),
        ]);
        let errors = verify_nogc(&program).unwrap_err();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_nogc_indirect_call_rejected() {
        // Indirect call via a variable that's not a simple Var callee
        let program = mk_program(vec![
            mk_fn("caller", true, vec![
                MirStmt::Expr(mk_expr(MirExprKind::Call {
                    callee: Box::new(mk_expr(MirExprKind::Index {
                        object: Box::new(mk_expr(MirExprKind::Var("fns".to_string()))),
                        index: Box::new(mk_expr(MirExprKind::IntLit(0))),
                    })),
                    args: vec![],
                })),
            ]),
        ]);
        let errors = verify_nogc(&program).unwrap_err();
        assert!(errors[0].reason.contains("indirect call"));
    }

    #[test]
    fn test_nogc_block_rejects_gc_alloc() {
        // Non-nogc function with a NoGcBlock containing gc_alloc
        let program = mk_program(vec![
            mk_fn("wrapper", false, vec![
                MirStmt::NoGcBlock(MirBody {
                    stmts: vec![MirStmt::Expr(mk_call("gc_alloc", vec![]))],
                    result: None,
                }),
            ]),
        ]);
        let errors = verify_nogc(&program).unwrap_err();
        assert!(!errors.is_empty());
        assert!(errors[0].reason.contains("gc_alloc"));
    }
}
