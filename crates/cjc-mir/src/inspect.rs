//! MIR Inspect/Diagnostics — Deterministic text dumps of analysis results
//!
//! Provides human-readable, deterministic text output for loop trees,
//! reduction reports, schedule metadata, SSA overlays, and legality reports.
//!
//! All output is sorted and reproducible — same input always produces
//! identical text.  This module is intended for:
//!
//! - Debugging MIR analysis passes
//! - Test assertions (snapshot-style)
//! - CLI `--mir-inspect` output (future)
//!
//! ## Design decisions
//!
//! - **Read-only** — never modifies any analysis structure
//! - **Deterministic** — sorted iteration, no HashMap, no random order
//! - **Plain text** — no ANSI colors, no Unicode box-drawing (easy to diff)

use crate::loop_analysis::{LoopTree, SchedulePlan};
use crate::reduction::{ReductionKind, ReductionReport};
use crate::verify::LegalityReport;

// ---------------------------------------------------------------------------
// Loop tree dump
// ---------------------------------------------------------------------------

/// Produce a deterministic text dump of a loop tree.
///
/// Example output:
/// ```text
/// LoopTree (2 loops, max_depth=1):
///   Loop L0: header=B1, depth=0, body=[B1,B2,B3], exits=[B4], schedule=sequential_strict
///     countable=false, trip_count=unknown, num_exits=1
///     Loop L1: header=B2, depth=1, body=[B2,B3], exits=[B1], schedule=sequential_strict
///       countable=false, trip_count=unknown, num_exits=1
/// ```
pub fn dump_loop_tree(loop_tree: &LoopTree) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "LoopTree ({} loops, max_depth={}):\n",
        loop_tree.len(),
        loop_tree.max_depth()
    ));

    // Print root loops, then recurse for children.
    let roots = loop_tree.root_loops();
    for root_id in &roots {
        dump_loop_recursive(loop_tree, *root_id, 1, &mut out);
    }

    out
}

fn dump_loop_recursive(loop_tree: &LoopTree, loop_id: crate::loop_analysis::LoopId, indent: usize, out: &mut String) {
    let info = loop_tree.get(loop_id);
    let pad = "  ".repeat(indent);

    let body_str: String = info
        .body_blocks
        .iter()
        .map(|b| format!("B{}", b.0))
        .collect::<Vec<_>>()
        .join(",");
    let exit_str: String = info
        .exit_blocks
        .iter()
        .map(|b| format!("B{}", b.0))
        .collect::<Vec<_>>()
        .join(",");

    out.push_str(&format!(
        "{}Loop L{}: header=B{}, depth={}, body=[{}], exits=[{}], schedule={}\n",
        pad, info.id.0, info.header.0, info.depth, body_str, exit_str, info.schedule,
    ));

    let trip = match info.trip_count_hint {
        Some(n) => format!("{}", n),
        None => "unknown".to_string(),
    };
    out.push_str(&format!(
        "{}  countable={}, trip_count={}, num_exits={}\n",
        pad, info.is_countable, trip, info.num_exits,
    ));

    for &child in &info.children {
        dump_loop_recursive(loop_tree, child, indent + 1, out);
    }
}

// ---------------------------------------------------------------------------
// Reduction report dump
// ---------------------------------------------------------------------------

/// Produce a deterministic text dump of a reduction report.
///
/// Example output:
/// ```text
/// ReductionReport (3 reductions):
///   R0: acc="total", op=Add, kind=StrictFold, fn="compute", loop=L0
///       reassoc_forbidden=true, strict_order=true, semantics=plain
///   R1: builtin="sum", kind=BuiltinReduction, fn="compute"
///       reassoc_forbidden=true, strict_order=true, semantics=runtime_defined
/// ```
pub fn dump_reduction_report(report: &ReductionReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "ReductionReport ({} reductions):\n",
        report.len()
    ));

    for r in &report.reductions {
        let kind_str = match r.kind {
            ReductionKind::StrictFold => "StrictFold",
            ReductionKind::KahanFold => "KahanFold",
            ReductionKind::BinnedFold => "BinnedFold",
            ReductionKind::FixedTree => "FixedTree",
            ReductionKind::BuiltinReduction => "BuiltinReduction",
            ReductionKind::Unknown => "Unknown",
        };

        if let Some(ref builtin) = r.builtin_name {
            out.push_str(&format!(
                "  R{}: builtin=\"{}\", kind={}, fn=\"{}\"",
                r.id.0, builtin, kind_str, r.function_name,
            ));
        } else {
            let loop_str = match r.loop_id {
                Some(lid) => format!(", loop=L{}", lid.0),
                None => String::new(),
            };
            out.push_str(&format!(
                "  R{}: acc=\"{}\", op={:?}, kind={}, fn=\"{}\"{}",
                r.id.0, r.accumulator_var, r.op, kind_str, r.function_name, loop_str,
            ));
        }
        out.push('\n');

        out.push_str(&format!(
            "      reassoc_forbidden={}, strict_order={}, semantics={}\n",
            r.reassociation_forbidden, r.strict_order_required, r.accumulator_semantics,
        ));
    }

    out
}

// ---------------------------------------------------------------------------
// Legality report dump
// ---------------------------------------------------------------------------

/// Produce a deterministic text dump of a legality report.
pub fn dump_legality_report(report: &LegalityReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "LegalityReport: {}/{} checks passed",
        report.checks_passed, report.checks_total,
    ));

    if report.is_ok() {
        out.push_str(" ✓\n");
    } else {
        out.push_str(&format!(" ({} errors)\n", report.errors.len()));
        for err in &report.errors {
            out.push_str(&format!("  ERROR: {}\n", err));
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Schedule plan summary
// ---------------------------------------------------------------------------

/// Produce a one-line summary of schedule plans across all loops.
pub fn dump_schedule_summary(loop_tree: &LoopTree) -> String {
    let mut counts: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    for info in &loop_tree.loops {
        let key = match &info.schedule {
            SchedulePlan::SequentialStrict => "sequential_strict".to_string(),
            SchedulePlan::DescriptiveTiled { .. } => "descriptive_tiled".to_string(),
            SchedulePlan::DescriptiveVectorized { .. } => "descriptive_vectorized".to_string(),
            SchedulePlan::DescriptiveMaterializeBoundary => "descriptive_materialize_boundary".to_string(),
            SchedulePlan::DescriptiveStaticPartition { .. } => "descriptive_static_partition".to_string(),
        };
        *counts.entry(key).or_insert(0) += 1;
    }

    let parts: Vec<String> = counts
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect();

    if parts.is_empty() {
        "ScheduleSummary: (no loops)".to_string()
    } else {
        format!("ScheduleSummary: {}", parts.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_analysis::{LoopId, LoopInfo, LoopTree, SchedulePlan};
    use crate::reduction::{
        AccumulatorSemantics, ReductionId, ReductionInfo, ReductionKind, ReductionOp,
        ReductionReport,
    };
    use crate::BlockId;

    fn make_simple_loop_tree() -> LoopTree {
        LoopTree {
            loops: vec![LoopInfo {
                id: LoopId(0),
                header: BlockId(1),
                body_blocks: vec![BlockId(1), BlockId(2)],
                back_edge_sources: vec![BlockId(2)],
                exit_blocks: vec![BlockId(3)],
                preheader: Some(BlockId(0)),
                parent: None,
                children: Vec::new(),
                depth: 0,
                is_countable: false,
                trip_count_hint: None,
                num_exits: 1,
                schedule: SchedulePlan::default(),
            }],
            block_to_loop: vec![None, Some(LoopId(0)), Some(LoopId(0)), None],
            num_blocks: 4,
        }
    }

    #[test]
    fn test_dump_loop_tree_format() {
        let lt = make_simple_loop_tree();
        let text = dump_loop_tree(&lt);
        assert!(text.contains("LoopTree (1 loops"));
        assert!(text.contains("Loop L0"));
        assert!(text.contains("header=B1"));
        assert!(text.contains("sequential_strict"));
        assert!(text.contains("countable=false"));
    }

    #[test]
    fn test_dump_reduction_report_format() {
        let report = ReductionReport {
            reductions: vec![ReductionInfo {
                id: ReductionId(0),
                accumulator_var: "acc".to_string(),
                op: ReductionOp::Add,
                kind: ReductionKind::StrictFold,
                loop_id: Some(LoopId(0)),
                function_name: "compute".to_string(),
                builtin_name: None,
                reassociation_forbidden: true,
                strict_order_required: true,
                accumulator_semantics: AccumulatorSemantics::Plain,
            }],
        };
        let text = dump_reduction_report(&report);
        assert!(text.contains("ReductionReport (1 reductions)"));
        assert!(text.contains("R0:"));
        assert!(text.contains("StrictFold"));
        assert!(text.contains("reassoc_forbidden=true"));
    }

    #[test]
    fn test_dump_legality_ok() {
        let report = LegalityReport {
            errors: Vec::new(),
            checks_passed: 5,
            checks_total: 5,
        };
        let text = dump_legality_report(&report);
        assert!(text.contains("5/5 checks passed"));
    }

    #[test]
    fn test_dump_schedule_summary() {
        let lt = make_simple_loop_tree();
        let text = dump_schedule_summary(&lt);
        assert!(text.contains("sequential_strict=1"));
    }

    #[test]
    fn test_dump_determinism() {
        let lt = make_simple_loop_tree();
        let text1 = dump_loop_tree(&lt);
        let text2 = dump_loop_tree(&lt);
        assert_eq!(text1, text2, "dump must be deterministic");
    }
}
