//! Top-level `CanaReport` + deterministic JSON serializer.
//!
//! Phase 1 emits a JSON sidecar file (typically `<program>.cana.json`) that
//! contains every per-function feature record plus the program- and
//! feature-level hashes. Ordering is canonical: BTreeMap keys are emitted
//! in sorted order, integer fields in struct-declaration order.
//!
//! ## Why hand-write JSON
//!
//! The CJC-Lang workspace deliberately keeps the IR crates serde-free
//! (`cjc-mir/Cargo.toml` has no serde dep). Adding serde just for the CANA
//! sidecar would couple this crate's dep tree to an entire feature
//! ecosystem. The serializer here is ~80 lines and proven byte-identical by
//! the determinism tests.
//!
//! ## Format
//!
//! The schema is documented in [`CanaReport::to_json`]. It is intentionally
//! a flat structure — nested objects only where they reflect the
//! `CanaFeatures` struct hierarchy. A consumer can `jq`-pipeline it without
//! any prior schema knowledge.

use std::fmt::Write as _;

use crate::features::{CanaFeatures, FnFeatures};
use crate::legality::{LegalityVerdict, LegalityViolation, ProposedPass};

// ---------------------------------------------------------------------------
// CanaReport — the public report type
// ---------------------------------------------------------------------------

/// Top-level CANA report for a single MIR program.
///
/// Phase 1 always includes:
/// - `features` — the full [`CanaFeatures`] aggregate
/// - `baseline_verdict` — the default legality gate's verdict over the empty
///   proposed pass sequence (always [`LegalityVerdict::Approved`]; included
///   as a sanity baseline for downstream gating tests)
///
/// Phase 2+ will add `recommendations`, `cost_estimates`, etc.; the schema is
/// forward-compatible (new top-level fields can be added without breaking
/// existing consumers).
#[derive(Debug, Clone)]
pub struct CanaReport {
    pub features: CanaFeatures,
    pub baseline_verdict: LegalityVerdict,
    /// Schema version. Bumped when the JSON output shape changes
    /// incompatibly. Phase 1 = 1.
    pub schema_version: u32,
}

impl CanaReport {
    /// Construct a report from extracted features and a baseline verdict.
    pub fn new(features: CanaFeatures, baseline_verdict: LegalityVerdict) -> Self {
        Self {
            features,
            baseline_verdict,
            schema_version: 1,
        }
    }

    /// Serialize the report to canonical JSON.
    ///
    /// Canonical means:
    /// - BTreeMap keys are emitted in sorted order (deterministic by
    ///   construction).
    /// - Struct fields are emitted in declaration order, never alphabetized.
    /// - Numbers are emitted as integers (no trailing `.0`) where the source
    ///   type is integer.
    /// - No trailing whitespace, single-space separators after `:` and `,`,
    ///   2-space indent.
    ///
    /// The output is JSON-Lines safe (the entire report is one document).
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(512);
        s.push_str("{\n");
        writeln!(s, "  \"schema_version\": {},", self.schema_version).unwrap();
        writeln!(s, "  \"crate_version\": \"{}\",", env!("CARGO_PKG_VERSION")).unwrap();
        writeln!(s, "  \"phase\": \"passive_observer\",").unwrap();
        writeln!(
            s,
            "  \"program_hash\": \"{}\",",
            self.features.program_hash.to_hex()
        )
        .unwrap();
        writeln!(
            s,
            "  \"feature_hash\": \"{}\",",
            self.features.feature_hash.to_hex()
        )
        .unwrap();
        writeln!(s, "  \"function_count\": {},", self.features.function_count()).unwrap();
        writeln!(s, "  \"total_blocks\": {},", self.features.total_blocks()).unwrap();
        writeln!(
            s,
            "  \"total_strict_reductions\": {},",
            self.features.total_strict_reductions()
        )
        .unwrap();

        // Per-function records, sorted by name (BTreeMap iteration is sorted).
        s.push_str("  \"per_fn\": {\n");
        let n = self.features.per_fn.len();
        for (i, (name, feats)) in self.features.per_fn.iter().enumerate() {
            write!(s, "    \"{}\": ", json_escape_string(name)).unwrap();
            write_fn_features(&mut s, feats);
            if i + 1 < n {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  },\n");

        // Baseline legality verdict.
        s.push_str("  \"baseline_verdict\": ");
        write_verdict(&mut s, &self.baseline_verdict);
        s.push('\n');

        s.push('}');
        s.push('\n');
        s
    }

    /// Canonical bytes — the JSON output as a UTF-8 byte slice.
    ///
    /// Two reports with the same `to_json()` output have the same
    /// `canonical_bytes()`. The determinism tests assert that
    /// `extract(&p).to_json()` is byte-identical across runs.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.to_json().into_bytes()
    }
}

// ---------------------------------------------------------------------------
// Internal JSON writers
// ---------------------------------------------------------------------------

fn write_fn_features(s: &mut String, f: &FnFeatures) {
    s.push('{');
    // CFG metrics
    write!(
        s,
        " \"cfg\": {{ \"block_count\": {}, \"edge_count\": {}, \"branch_count\": {}, \
         \"return_count\": {}, \"unreachable_count\": {}, \"goto_count\": {}, \
         \"max_branch_factor\": {}, \"loop_count\": {}, \"max_loop_depth\": {}, \
         \"back_edge_count\": {}, \"countable_loop_count\": {}, \"cfg_hash\": \"{}\" }},",
        f.cfg.block_count,
        f.cfg.edge_count,
        f.cfg.branch_count,
        f.cfg.return_count,
        f.cfg.unreachable_count,
        f.cfg.goto_count,
        f.cfg.max_branch_factor,
        f.cfg.loop_count,
        f.cfg.max_loop_depth,
        f.cfg.back_edge_count,
        f.cfg.countable_loop_count,
        f.cfg.cfg_hash.to_hex()
    )
    .unwrap();
    // Memory proxy
    write!(
        s,
        " \"memory\": {{ \"alloc_sites\": {}, \"cow_write_sites\": {}, \
         \"tensor_heavy_ops\": {}, \"expr_count\": {} }},",
        f.memory.alloc_sites,
        f.memory.cow_write_sites,
        f.memory.tensor_heavy_ops,
        f.memory.expr_count
    )
    .unwrap();
    // Reduction axes
    write!(
        s,
        " \"reductions\": {{ \"strict_fold\": {}, \"kahan_fold\": {}, \
         \"binned_fold\": {}, \"fixed_tree\": {}, \"builtin_reduction\": {}, \
         \"unknown\": {}, \"strict_count\": {}, \"has_strict_reduction\": {} }} }}",
        f.reductions.strict_fold,
        f.reductions.kahan_fold,
        f.reductions.binned_fold,
        f.reductions.fixed_tree,
        f.reductions.builtin_reduction,
        f.reductions.unknown,
        f.reductions.strict_count(),
        f.reductions.has_strict_reduction()
    )
    .unwrap();
}

fn write_verdict(s: &mut String, v: &LegalityVerdict) {
    match v {
        LegalityVerdict::Approved => {
            s.push_str("{ \"status\": \"approved\", \"violations\": [] }");
        }
        LegalityVerdict::Rejected(violations) => {
            s.push_str("{ \"status\": \"rejected\", \"violations\": [");
            for (i, v) in violations.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                write_violation(s, v);
            }
            s.push_str("] }");
        }
    }
}

fn write_violation(s: &mut String, v: &LegalityViolation) {
    match v {
        LegalityViolation::StrictReductionPresent {
            function,
            proposed,
            strict_count,
        } => {
            write!(
                s,
                "{{ \"kind\": \"strict_reduction_present\", \"function\": \"{}\", \
                 \"proposed\": {}, \"strict_count\": {} }}",
                json_escape_string(function),
                proposed_to_json(proposed),
                strict_count
            )
            .unwrap();
        }
        LegalityViolation::UnknownFunction { function, proposed } => {
            write!(
                s,
                "{{ \"kind\": \"unknown_function\", \"function\": \"{}\", \
                 \"proposed\": {} }}",
                json_escape_string(function),
                proposed_to_json(proposed),
            )
            .unwrap();
        }
    }
}

fn proposed_to_json(p: &ProposedPass) -> String {
    match p {
        ProposedPass::Run(name) => format!(
            "{{ \"action\": \"run\", \"pass\": \"{}\" }}",
            json_escape_string(name)
        ),
        ProposedPass::Skip(name) => format!(
            "{{ \"action\": \"skip\", \"pass\": \"{}\" }}",
            json_escape_string(name)
        ),
    }
}

/// Minimal JSON-string escape: `\\`, `"`, control chars `< 0x20`. Sufficient
/// for function names + pass names; we don't accept arbitrary user input here.
fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).unwrap();
            }
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze_program;
    use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

    fn empty_program() -> MirProgram {
        MirProgram {
            functions: vec![MirFunction {
                id: MirFnId(0),
                name: "__main".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: MirBody {
                    stmts: vec![MirStmt::Expr(MirExpr {
                        kind: MirExprKind::IntLit(42),
                    })],
                    result: None,
                },
                is_nogc: false,
                cfg_body: None,
                decorators: vec![],
                vis: cjc_ast::Visibility::Public,
                local_count: 0,
            }],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    #[test]
    fn report_serializes_to_nonempty_json() {
        let report = analyze_program(&empty_program());
        let s = report.to_json();
        assert!(s.starts_with('{'));
        assert!(s.ends_with("}\n"));
        assert!(s.contains("\"schema_version\""));
        assert!(s.contains("\"program_hash\""));
        assert!(s.contains("\"feature_hash\""));
        assert!(s.contains("\"per_fn\""));
        assert!(s.contains("\"__main\""));
        assert!(s.contains("\"baseline_verdict\""));
    }

    #[test]
    fn report_json_is_byte_identical_across_runs() {
        let program = empty_program();
        let first = analyze_program(&program).to_json();
        for _ in 0..50 {
            let again = analyze_program(&program).to_json();
            assert_eq!(again, first);
        }
    }

    #[test]
    fn json_escape_handles_special_chars() {
        assert_eq!(json_escape_string("hello"), "hello");
        assert_eq!(json_escape_string("a\"b"), "a\\\"b");
        assert_eq!(json_escape_string("a\\b"), "a\\\\b");
        assert_eq!(json_escape_string("a\nb"), "a\\nb");
        assert_eq!(json_escape_string("a\x01b"), "a\\u0001b");
    }

    #[test]
    fn baseline_verdict_is_approved() {
        let report = analyze_program(&empty_program());
        assert!(report.baseline_verdict.is_approved());
        let s = report.to_json();
        assert!(s.contains("\"status\": \"approved\""));
    }

    #[test]
    fn canonical_bytes_matches_to_json() {
        let r = analyze_program(&empty_program());
        let bytes = r.canonical_bytes();
        let string = r.to_json();
        assert_eq!(bytes, string.as_bytes());
    }
}
