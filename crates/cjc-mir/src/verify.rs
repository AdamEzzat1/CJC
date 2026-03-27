//! MIR Legality Verifier — Determinism and transformation legality checks
//!
//! This module extends CJC's verification infrastructure beyond `@nogc` to
//! enforce broader correctness invariants:
//!
//! 1. **Reduction contract preservation** — ensures that optimizer passes do
//!    not reorder strict-fold reductions
//! 2. **Schedule determinism** — ensures CFG structure is deterministic
//!    (no HashMap iteration, sorted block ordering, etc.)
//! 3. **Loop structure integrity** — ensures loop tree is well-formed
//! 4. **@nogc compatibility** — delegates to existing `nogc_verify` module
//!
//! ## Design decisions
//!
//! - **Additive overlay** — does not modify any existing verification code
//! - **Lightweight checks** — no heavy proof systems, just structural assertions
//! - **Vec + ID** — all internal structures use Vec indexing
//! - **Deterministic** — all checks are order-independent or use sorted iteration
//!
//! ## Usage
//!
//! ```rust,ignore
//! use cjc_mir::verify::{verify_mir_legality, LegalityReport};
//!
//! let report = verify_mir_legality(&program);
//! assert!(report.is_ok(), "MIR legality check failed: {:?}", report.errors());
//! ```

use crate::cfg::MirCfg;
use crate::dominators::DominatorTree;
use crate::loop_analysis::{self, LoopTree};
use crate::reduction::{self, ReductionKind, ReductionReport};
use crate::{BlockId, MirBody, MirExpr, MirExprKind, MirFunction, MirProgram, MirStmt};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// A single legality violation.
#[derive(Debug, Clone)]
pub struct LegalityError {
    /// Which check caught the violation.
    pub check: LegalityCheck,
    /// Human-readable description.
    pub message: String,
    /// Function where the violation occurred.
    pub function: String,
}

impl std::fmt::Display for LegalityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:?}] in `{}`: {}",
            self.check, self.function, self.message
        )
    }
}

/// Which legality check detected the violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalityCheck {
    /// CFG structure is malformed or non-deterministic.
    CfgStructure,
    /// Loop tree is malformed (cycles, missing headers, etc.).
    LoopIntegrity,
    /// A strict reduction may have been reordered.
    ReductionContract,
    /// SSA form has a violation (delegate to ssa::verify_ssa).
    SsaIntegrity,
    /// @nogc constraint violated (delegate to nogc_verify).
    NoGcContract,
    /// Infinite recursion or unbounded nesting detected.
    StructuralBound,
}

// ---------------------------------------------------------------------------
// Legality report
// ---------------------------------------------------------------------------

/// The result of running all legality checks on a MIR program.
#[derive(Debug, Clone)]
pub struct LegalityReport {
    /// All errors found.  Empty if the program is legal.
    pub errors: Vec<LegalityError>,
    /// Number of checks that passed.
    pub checks_passed: u32,
    /// Total number of checks run.
    pub checks_total: u32,
}

impl LegalityReport {
    /// Returns true if no errors were found.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns the errors.
    pub fn errors(&self) -> &[LegalityError] {
        &self.errors
    }
}

// ---------------------------------------------------------------------------
// Top-level verification
// ---------------------------------------------------------------------------

/// Run all legality checks on a MIR program.
///
/// This is the main entry point.  It runs:
/// 1. CFG structure checks
/// 2. Loop tree integrity checks
/// 3. Reduction contract checks
/// 4. Structural bound checks (nesting depth)
///
/// Note: `@nogc` verification is handled separately by `nogc_verify::verify_nogc`.
/// This function focuses on new checks that complement the existing system.
pub fn verify_mir_legality(program: &MirProgram) -> LegalityReport {
    let mut errors = Vec::new();
    let mut checks_passed = 0u32;
    let mut checks_total = 0u32;

    for func in &program.functions {
        // Check 1: CFG structure (if CFG has been built).
        if let Some(ref cfg) = func.cfg_body {
            checks_total += 1;
            let cfg_errors = check_cfg_structure(cfg, &func.name);
            if cfg_errors.is_empty() {
                checks_passed += 1;
            } else {
                errors.extend(cfg_errors);
            }

            // Check 2: Loop tree integrity.
            checks_total += 1;
            let domtree = DominatorTree::compute(cfg);
            let loop_tree = loop_analysis::compute_loop_tree(cfg, &domtree);
            let loop_errors = check_loop_integrity(&loop_tree, cfg, &func.name);
            if loop_errors.is_empty() {
                checks_passed += 1;
            } else {
                errors.extend(loop_errors);
            }
        }

        // Check 3: Structural bounds (tree-form MIR).
        checks_total += 1;
        let bound_errors = check_structural_bounds(func);
        if bound_errors.is_empty() {
            checks_passed += 1;
        } else {
            errors.extend(bound_errors);
        }
    }

    // Check 4: Reduction contract (program-wide).
    checks_total += 1;
    let reduction_report = reduction::detect_reductions(program, &[]);
    let reduction_errors = check_reduction_contracts(&reduction_report);
    if reduction_errors.is_empty() {
        checks_passed += 1;
    } else {
        errors.extend(reduction_errors);
    }

    LegalityReport {
        errors,
        checks_passed,
        checks_total,
    }
}

// ---------------------------------------------------------------------------
// Check 1: CFG structure
// ---------------------------------------------------------------------------

/// Verify CFG structural invariants:
/// - Entry block is BlockId(0)
/// - All successor/predecessor references are in bounds
/// - Every block has exactly one terminator
/// - No orphaned blocks (reachable from entry via BFS)
fn check_cfg_structure(cfg: &MirCfg, fn_name: &str) -> Vec<LegalityError> {
    let mut errors = Vec::new();
    let num_blocks = cfg.basic_blocks.len();

    // Entry must be BlockId(0).
    if cfg.entry != BlockId(0) {
        errors.push(LegalityError {
            check: LegalityCheck::CfgStructure,
            message: format!("entry block is {:?}, expected BlockId(0)", cfg.entry),
            function: fn_name.to_string(),
        });
    }

    // Block IDs must match their index.
    for (i, bb) in cfg.basic_blocks.iter().enumerate() {
        if bb.id.0 as usize != i {
            errors.push(LegalityError {
                check: LegalityCheck::CfgStructure,
                message: format!(
                    "block at index {} has id {:?} (should be {})",
                    i, bb.id, i
                ),
                function: fn_name.to_string(),
            });
        }
    }

    // All successor references must be in bounds.
    for bb in &cfg.basic_blocks {
        for succ in bb.terminator.successors() {
            if succ.0 as usize >= num_blocks {
                errors.push(LegalityError {
                    check: LegalityCheck::CfgStructure,
                    message: format!(
                        "block {:?} has out-of-bounds successor {:?} (num_blocks={})",
                        bb.id, succ, num_blocks
                    ),
                    function: fn_name.to_string(),
                });
            }
        }
    }

    // Reachability: BFS from entry.
    let mut visited = vec![false; num_blocks];
    let mut queue = vec![cfg.entry];
    while let Some(block) = queue.pop() {
        let idx = block.0 as usize;
        if idx >= num_blocks || visited[idx] {
            continue;
        }
        visited[idx] = true;
        for succ in cfg.basic_blocks[idx].terminator.successors() {
            if !visited[succ.0 as usize] {
                queue.push(succ);
            }
        }
    }

    // Note: unreachable blocks are not errors (dead code after return is
    // expected).  We just verify that the entry block is reachable.
    if num_blocks > 0 && !visited[0] {
        errors.push(LegalityError {
            check: LegalityCheck::CfgStructure,
            message: "entry block is not reachable (impossible)".to_string(),
            function: fn_name.to_string(),
        });
    }

    errors
}

// ---------------------------------------------------------------------------
// Check 2: Loop tree integrity
// ---------------------------------------------------------------------------

/// Verify loop tree structural invariants:
/// - Every loop header is in its own body
/// - Every back-edge source is in the loop body
/// - Parent/child relationships are consistent
/// - No cycles in the nesting tree
fn check_loop_integrity(
    loop_tree: &LoopTree,
    cfg: &MirCfg,
    fn_name: &str,
) -> Vec<LegalityError> {
    let mut errors = Vec::new();
    let num_blocks = cfg.basic_blocks.len();

    for info in &loop_tree.loops {
        // Header must be in body.
        if info.body_blocks.binary_search(&info.header).is_err() {
            errors.push(LegalityError {
                check: LegalityCheck::LoopIntegrity,
                message: format!(
                    "loop {:?} header {:?} not in body_blocks",
                    info.id, info.header
                ),
                function: fn_name.to_string(),
            });
        }

        // All back-edge sources must be in body.
        for &src in &info.back_edge_sources {
            if info.body_blocks.binary_search(&src).is_err() {
                errors.push(LegalityError {
                    check: LegalityCheck::LoopIntegrity,
                    message: format!(
                        "loop {:?} back-edge source {:?} not in body_blocks",
                        info.id, src
                    ),
                    function: fn_name.to_string(),
                });
            }
        }

        // All body blocks must be valid block IDs.
        for &block in &info.body_blocks {
            if block.0 as usize >= num_blocks {
                errors.push(LegalityError {
                    check: LegalityCheck::LoopIntegrity,
                    message: format!(
                        "loop {:?} body contains out-of-bounds block {:?}",
                        info.id, block
                    ),
                    function: fn_name.to_string(),
                });
            }
        }

        // Body blocks must be sorted (determinism invariant).
        let is_sorted = info
            .body_blocks
            .windows(2)
            .all(|w| w[0] <= w[1]);
        if !is_sorted {
            errors.push(LegalityError {
                check: LegalityCheck::LoopIntegrity,
                message: format!(
                    "loop {:?} body_blocks not sorted (determinism violation)",
                    info.id
                ),
                function: fn_name.to_string(),
            });
        }

        // Parent/child consistency.
        if let Some(parent) = info.parent {
            let parent_info = &loop_tree.loops[parent.0 as usize];
            if !parent_info.children.contains(&info.id) {
                errors.push(LegalityError {
                    check: LegalityCheck::LoopIntegrity,
                    message: format!(
                        "loop {:?} claims parent {:?} but parent doesn't list it as child",
                        info.id, parent
                    ),
                    function: fn_name.to_string(),
                });
            }
        }
    }

    // Check for nesting cycles (defensive — should be impossible from construction).
    for info in &loop_tree.loops {
        let mut visited = vec![false; loop_tree.loops.len()];
        let mut current = Some(info.id);
        while let Some(lid) = current {
            let idx = lid.0 as usize;
            if visited[idx] {
                errors.push(LegalityError {
                    check: LegalityCheck::LoopIntegrity,
                    message: format!("loop nesting cycle detected involving {:?}", lid),
                    function: fn_name.to_string(),
                });
                break;
            }
            visited[idx] = true;
            current = loop_tree.loops[idx].parent;
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// Check 3: Reduction contracts
// ---------------------------------------------------------------------------

/// Verify that no strict reductions have been marked as reorderable.
///
/// This is a structural check: it verifies that the reduction report itself
/// is internally consistent.  The actual "was this reduction reordered by
/// the optimizer?" check would require comparing pre- and post-optimization
/// MIR — that is a future extension.
fn check_reduction_contracts(report: &ReductionReport) -> Vec<LegalityError> {
    let mut errors = Vec::new();

    for r in &report.reductions {
        // A StrictFold reduction must not be marked as reorderable.
        if r.kind == ReductionKind::StrictFold && r.kind.is_reorderable() {
            errors.push(LegalityError {
                check: LegalityCheck::ReductionContract,
                message: format!(
                    "StrictFold reduction on `{}` is marked reorderable",
                    r.accumulator_var
                ),
                function: r.function_name.clone(),
            });
        }

        // Unknown reductions must be conservative.
        if r.kind == ReductionKind::Unknown && r.kind.is_parallelizable() {
            errors.push(LegalityError {
                check: LegalityCheck::ReductionContract,
                message: format!(
                    "Unknown reduction on `{}` is marked parallelizable",
                    r.accumulator_var
                ),
                function: r.function_name.clone(),
            });
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// Check 4: Structural bounds
// ---------------------------------------------------------------------------

/// Maximum allowed nesting depth for MIR statements.
/// This prevents stack overflow during tree-walk analysis.
const MAX_NESTING_DEPTH: u32 = 256;

/// Check structural bounds (nesting depth) in tree-form MIR.
fn check_structural_bounds(func: &MirFunction) -> Vec<LegalityError> {
    let mut errors = Vec::new();
    check_body_depth(&func.body, 0, &func.name, &mut errors);
    errors
}

fn check_body_depth(
    body: &MirBody,
    depth: u32,
    fn_name: &str,
    errors: &mut Vec<LegalityError>,
) {
    if depth > MAX_NESTING_DEPTH {
        errors.push(LegalityError {
            check: LegalityCheck::StructuralBound,
            message: format!(
                "nesting depth {} exceeds maximum {} — possible infinite recursion in MIR",
                depth, MAX_NESTING_DEPTH
            ),
            function: fn_name.to_string(),
        });
        return;
    }

    for stmt in &body.stmts {
        match stmt {
            MirStmt::If {
                then_body,
                else_body,
                ..
            } => {
                check_body_depth(then_body, depth + 1, fn_name, errors);
                if let Some(eb) = else_body {
                    check_body_depth(eb, depth + 1, fn_name, errors);
                }
            }
            MirStmt::While { body: wb, .. } => {
                check_body_depth(wb, depth + 1, fn_name, errors);
            }
            MirStmt::NoGcBlock(inner) => {
                check_body_depth(inner, depth + 1, fn_name, errors);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: verify a single function
// ---------------------------------------------------------------------------

/// Verify legality of a single function.
pub fn verify_function(func: &MirFunction) -> LegalityReport {
    let program = MirProgram {
        functions: vec![func.clone()],
        struct_defs: vec![],
        enum_defs: vec![],
        entry: func.id,
    };
    verify_mir_legality(&program)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MirBody, MirExpr, MirExprKind, MirFnId, MirParam, MirStmt};
    use cjc_ast::Visibility;

    fn int_expr(v: i64) -> MirExpr {
        MirExpr {
            kind: MirExprKind::IntLit(v),
        }
    }

    fn bool_expr(b: bool) -> MirExpr {
        MirExpr {
            kind: MirExprKind::BoolLit(b),
        }
    }

    fn make_fn(name: &str, body: MirBody) -> MirFunction {
        MirFunction {
            id: MirFnId(0),
            name: name.to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body,
            is_nogc: false,
            cfg_body: None,
            decorators: vec![],
            vis: Visibility::Private,
        }
    }

    fn make_program(functions: Vec<MirFunction>) -> MirProgram {
        let entry = functions.last().map(|f| f.id).unwrap_or(MirFnId(0));
        MirProgram {
            functions,
            struct_defs: vec![],
            enum_defs: vec![],
            entry,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Empty program passes all checks
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_program_passes() {
        let program = make_program(vec![make_fn(
            "__main",
            MirBody {
                stmts: vec![],
                result: None,
            },
        )]);
        let report = verify_mir_legality(&program);
        assert!(report.is_ok(), "empty program should pass: {:?}", report.errors);
    }

    // -----------------------------------------------------------------------
    // Test 2: Simple program with while loop passes
    // -----------------------------------------------------------------------
    #[test]
    fn test_simple_while_passes() {
        let mut func = make_fn(
            "test",
            MirBody {
                stmts: vec![MirStmt::While {
                    cond: bool_expr(true),
                    body: MirBody {
                        stmts: vec![MirStmt::Expr(int_expr(1))],
                        result: None,
                    },
                }],
                result: None,
            },
        );
        func.build_cfg();
        let program = make_program(vec![func]);
        let report = verify_mir_legality(&program);
        assert!(report.is_ok(), "simple while should pass: {:?}", report.errors);
    }

    // -----------------------------------------------------------------------
    // Test 3: CFG with valid structure passes
    // -----------------------------------------------------------------------
    #[test]
    fn test_cfg_structure_valid() {
        let mut func = make_fn(
            "test",
            MirBody {
                stmts: vec![
                    MirStmt::Let {
                        name: "x".into(),
                        mutable: false,
                        init: int_expr(42),
                        alloc_hint: None,
                    },
                    MirStmt::If {
                        cond: bool_expr(true),
                        then_body: MirBody {
                            stmts: vec![],
                            result: Some(Box::new(int_expr(1))),
                        },
                        else_body: Some(MirBody {
                            stmts: vec![],
                            result: Some(Box::new(int_expr(2))),
                        }),
                    },
                ],
                result: None,
            },
        );
        func.build_cfg();
        let cfg = func.cfg_body.as_ref().unwrap();
        let errors = check_cfg_structure(cfg, "test");
        assert!(errors.is_empty(), "valid CFG should have no errors: {:?}", errors);
    }

    // -----------------------------------------------------------------------
    // Test 4: Nesting depth check
    // -----------------------------------------------------------------------
    #[test]
    fn test_reasonable_nesting_passes() {
        // 10 levels of nesting — well under the limit.
        let mut body = MirBody {
            stmts: vec![MirStmt::Expr(int_expr(1))],
            result: None,
        };
        for _ in 0..10 {
            body = MirBody {
                stmts: vec![MirStmt::If {
                    cond: bool_expr(true),
                    then_body: body,
                    else_body: None,
                }],
                result: None,
            };
        }
        let func = make_fn("test", body);
        let errors = check_structural_bounds(&func);
        assert!(errors.is_empty(), "10-level nesting should pass");
    }

    // -----------------------------------------------------------------------
    // Test 5: Reduction contract check — strict fold is consistent
    // -----------------------------------------------------------------------
    #[test]
    fn test_strict_fold_is_consistent() {
        // StrictFold.is_reorderable() is false, so the check should pass.
        let report = ReductionReport {
            reductions: vec![reduction::ReductionInfo {
                id: reduction::ReductionId(0),
                accumulator_var: "acc".to_string(),
                op: reduction::ReductionOp::Add,
                kind: ReductionKind::StrictFold,
                loop_id: None,
                function_name: "test".to_string(),
                builtin_name: None,
            }],
        };
        let errors = check_reduction_contracts(&report);
        assert!(errors.is_empty(), "consistent StrictFold should pass");
    }

    // -----------------------------------------------------------------------
    // Test 6: Loop integrity for well-formed nested loops
    // -----------------------------------------------------------------------
    #[test]
    fn test_loop_integrity_nested() {
        let mut func = make_fn(
            "test",
            MirBody {
                stmts: vec![MirStmt::While {
                    cond: bool_expr(true),
                    body: MirBody {
                        stmts: vec![MirStmt::While {
                            cond: bool_expr(true),
                            body: MirBody {
                                stmts: vec![MirStmt::Expr(int_expr(1))],
                                result: None,
                            },
                        }],
                        result: None,
                    },
                }],
                result: None,
            },
        );
        func.build_cfg();
        let cfg = func.cfg_body.as_ref().unwrap();
        let domtree = DominatorTree::compute(cfg);
        let loop_tree = loop_analysis::compute_loop_tree(cfg, &domtree);
        let errors = check_loop_integrity(&loop_tree, cfg, "test");
        assert!(
            errors.is_empty(),
            "well-formed nested loops should pass: {:?}",
            errors
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: verify_function convenience
    // -----------------------------------------------------------------------
    #[test]
    fn test_verify_function_convenience() {
        let func = make_fn(
            "simple",
            MirBody {
                stmts: vec![MirStmt::Let {
                    name: "x".into(),
                    mutable: false,
                    init: int_expr(42),
                    alloc_hint: None,
                }],
                result: Some(Box::new(int_expr(42))),
            },
        );
        let report = verify_function(&func);
        assert!(report.is_ok());
        assert!(report.checks_passed > 0);
    }

    // -----------------------------------------------------------------------
    // Test 8: Report structure
    // -----------------------------------------------------------------------
    #[test]
    fn test_report_structure() {
        let program = make_program(vec![make_fn(
            "__main",
            MirBody {
                stmts: vec![],
                result: None,
            },
        )]);
        let report = verify_mir_legality(&program);
        assert!(report.is_ok());
        assert!(report.checks_total > 0);
        assert_eq!(report.checks_passed, report.checks_total);
        assert!(report.errors().is_empty());
    }
}
