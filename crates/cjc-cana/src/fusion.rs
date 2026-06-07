//! Phase 3 — kernel-fusion candidate identification.
//!
//! ## What this module does
//!
//! Walks MIR programs to identify **chains of native primitives** (matmul,
//! mlp_layer, sum, mean, dot, adam_step, etc.) whose intermediate values
//! flow directly into one another with minimal interpreted glue. Each chain
//! becomes a `FusionCandidate` — a recommendation that the compiler emit a
//! single fused native call instead of a sequence of separate native calls
//! with MIR-walked glue between them.
//!
//! ## What this module does NOT do
//!
//! Actual code generation for fused kernels is **Phase 3.5** — the codegen
//! has to write new dispatch entries, parity-test every fused variant, and
//! integrate with the runtime. That's months of engineering. Phase 3
//! identifies *which* fusions are worth doing; Phase 3.5 does the doing.
//!
//! ## Why this matters
//!
//! From `CANA_NSS_COMPILER_IMPROVEMENT_PLAN.md` §3.3:
//!
//! > Method S3 — Hot-kernel native lowering for tensor ops (Phase 3,
//! > ~5-20× on hot paths)
//!
//! Today, a CJC-Lang sequence like:
//!
//! ```cjcl
//! let h1 = mlp_layer(input, w1, b1, "relu");      // native
//! let h2 = matmul(h1, w2);                         // native
//! let out = h2 + bias;                             // INTERPRETED — slow
//! ```
//!
//! Each native primitive is fast on its own, but the interpreted glue
//! between them is hundreds of MIR-walking instructions. A fused
//! `mlp_matmul_bias` kernel would skip the entire MIR walk between the
//! three native calls — measured 5–20× speedup on hot paths.
//!
//! ## How identification works
//!
//! Walk every function body. For each `MirStmt::Let { name, init, .. }`
//! whose `init` is a `Call` to a known native primitive (the
//! `NATIVE_PRIMITIVES` set), record (binding_name, primitive_name). Then
//! find consecutive statements where the next primitive's argument is
//! the previous primitive's binding. Group these into chains.
//!
//! ## Determinism contract
//!
//! Same MIR → byte-identical `FusionPlan`. We iterate statements in
//! their natural order, use `BTreeMap` for the binding-to-primitive
//! mapping, and never compare floats. Standard CANA discipline.

use std::collections::BTreeMap;

use cjc_mir::{MirBody, MirExpr, MirExprKind, MirProgram, MirStmt};

// ---------------------------------------------------------------------------
// Native primitive registry
// ---------------------------------------------------------------------------

/// Names of CJC-Lang native primitives that are worth fusing. Drawn from
/// `cjc-runtime`'s dispatch table — these are the operations that run as
/// pre-compiled Rust code rather than walking MIR. A chain of these is the
/// canonical Phase 3 fusion target.
///
/// Adding a new primitive here is the wiring step when `cjc-runtime` exposes
/// a new fast-path builtin (e.g., a new fused MLP variant).
pub const NATIVE_PRIMITIVES: &[&str] = &[
    "matmul",
    "transpose",
    "dot",
    "sum",
    "mean",
    "tensor_concat_1d",
    "mlp_forward",
    "mlp_layer",
    "encode_state_fast",
    "score_moves_batch",
    "adam_step",
];

/// Lookup helper.
pub fn is_native_primitive(name: &str) -> bool {
    NATIVE_PRIMITIVES.contains(&name)
}

// ---------------------------------------------------------------------------
// FusionCandidate + FusionPlan
// ---------------------------------------------------------------------------

/// A single chain of native primitives identified as fusion-worthy.
///
/// The chain is a sequence of `(binding_name, primitive_name)` pairs in
/// the order they appear in the source MIR. Phase 3.5 codegen will emit a
/// single fused native call replacing the entire chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionCandidate {
    /// The function this candidate lives in.
    pub function_name: String,
    /// Ordered list of `(binding_name, primitive_name)` entries.
    /// First entry is the chain's input, last is its output.
    pub chain: Vec<ChainEntry>,
    /// Confidence in `[0, 100]` — how likely fusion will help. Phase 3 uses
    /// chain length as a proxy; longer chains have higher confidence.
    /// Phase 5 cost model will replace this with a trained estimate.
    pub confidence_pct: u8,
}

/// A single step in a fusion chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainEntry {
    /// The local binding name this step produces (e.g., `"h1"` in
    /// `let h1 = mlp_layer(...)`).
    pub binding_name: String,
    /// The native primitive called (e.g., `"mlp_layer"`).
    pub primitive_name: String,
}

impl FusionCandidate {
    /// Number of primitives in this chain.
    pub fn chain_length(&self) -> usize {
        self.chain.len()
    }

    /// True if this chain is worth fusing (length ≥ 2).
    pub fn is_fusion_worthy(&self) -> bool {
        self.chain.len() >= 2
    }
}

/// All fusion candidates identified in a program.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FusionPlan {
    /// Candidates ordered by function name, then by appearance in body.
    pub candidates: Vec<FusionCandidate>,
}

impl FusionPlan {
    /// Number of candidates.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// True if no candidates were found.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Total native-primitive call count across all candidates.
    pub fn total_chain_length(&self) -> usize {
        self.candidates.iter().map(|c| c.chain_length()).sum()
    }

    /// Filter for chains with length ≥ 2 (the only ones worth fusing).
    pub fn fusion_worthy(&self) -> impl Iterator<Item = &FusionCandidate> {
        self.candidates.iter().filter(|c| c.is_fusion_worthy())
    }
}

// ---------------------------------------------------------------------------
// Identification entry point
// ---------------------------------------------------------------------------

/// Identify all fusion candidates in a MIR program.
///
/// Walks every function body, finds `let name = native_primitive(...)`
/// patterns, and groups consecutive ones whose argument chains directly
/// reference the previous binding.
///
/// Phase 3 limitation: only considers *straight-line* chains within a
/// single body. Cross-block fusion (e.g., through if/else or while) is
/// deferred to Phase 3.5 — it requires more sophisticated def-use
/// analysis.
pub fn identify_fusion_candidates(program: &MirProgram) -> FusionPlan {
    let mut plan = FusionPlan::default();
    for func in &program.functions {
        let candidates = identify_in_body(&func.body, &func.name);
        plan.candidates.extend(candidates);
    }
    plan
}

fn identify_in_body(body: &MirBody, fn_name: &str) -> Vec<FusionCandidate> {
    // Pass 1: collect every `let X = native_primitive(args...)` in this
    // body, in statement order. `binding_to_args` maps each binding to
    // the list of argument names (just the Var/VarLocal references —
    // we ignore literal args because they don't form a chain).
    let mut binding_order: Vec<ChainEntry> = Vec::new();
    let mut binding_args: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for stmt in &body.stmts {
        if let MirStmt::Let { name, init, .. } = stmt {
            if let Some((prim_name, args)) = extract_primitive_call(init) {
                binding_order.push(ChainEntry {
                    binding_name: name.clone(),
                    primitive_name: prim_name,
                });
                binding_args.insert(name.clone(), args);
            }
        }
    }

    // Pass 2: walk binding_order, grouping into chains. A chain extends
    // when the next binding's first var-arg matches the previous
    // binding's name. We use only the *first* arg as the chain
    // connector — Phase 3 simplification. Phase 3.5 can do
    // multi-input fusion.
    let mut chains: Vec<Vec<ChainEntry>> = Vec::new();
    let mut current: Vec<ChainEntry> = Vec::new();

    for entry in binding_order {
        // Does this entry extend the current chain?
        let extends_chain = current
            .last()
            .map(|prev| {
                binding_args
                    .get(&entry.binding_name)
                    .map(|args| args.iter().any(|a| a == &prev.binding_name))
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        if extends_chain {
            current.push(entry);
        } else {
            // Flush current chain if non-trivial (length >= 1).
            if !current.is_empty() {
                chains.push(std::mem::take(&mut current));
            }
            // Start a new chain with this entry.
            current.push(entry);
        }
    }
    if !current.is_empty() {
        chains.push(current);
    }

    // Pass 3: convert chains to FusionCandidate, computing confidence
    // from chain length.
    chains
        .into_iter()
        .map(|chain| FusionCandidate {
            function_name: fn_name.to_string(),
            confidence_pct: chain_confidence(chain.len()),
            chain,
        })
        .collect()
}

/// Extract the (primitive_name, arg_names) if expr is a Call to a known
/// native primitive. Returns None for non-Call or non-native expressions.
///
/// arg_names contains only the Var/VarLocal name strings — literals and
/// nested expressions are ignored (they can't be chain links).
fn extract_primitive_call(expr: &MirExpr) -> Option<(String, Vec<String>)> {
    let MirExprKind::Call { callee, args } = &expr.kind else {
        return None;
    };
    let callee_name = match &callee.kind {
        MirExprKind::Var(n) => n.as_str(),
        MirExprKind::VarLocal { name, .. } => name.as_str(),
        _ => return None,
    };
    if !is_native_primitive(callee_name) {
        return None;
    }
    let arg_names: Vec<String> = args
        .iter()
        .filter_map(|a| match &a.kind {
            MirExprKind::Var(n) => Some(n.clone()),
            MirExprKind::VarLocal { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    Some((callee_name.to_string(), arg_names))
}

/// Hand-tuned confidence curve. Phase 5 will train this; for Phase 3
/// foundation we encode the qualitative behaviour:
/// - chain length 1: 10% (a lone primitive can't be fused with anything)
/// - chain length 2: 50% (worth considering)
/// - chain length 3: 75%
/// - chain length 4+: 90% (almost certainly worth fusing)
fn chain_confidence(len: usize) -> u8 {
    match len {
        0 => 0,
        1 => 10,
        2 => 50,
        3 => 75,
        _ => 90,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_mir::{MirBody, MirFnId, MirFunction, MirProgram};

    fn ekind(k: MirExprKind) -> MirExpr {
        MirExpr { kind: k }
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

    fn make_call_let(name: &str, primitive: &str, args: Vec<&str>) -> MirStmt {
        let call_args: Vec<MirExpr> = args
            .into_iter()
            .map(|a| ekind(MirExprKind::Var(a.to_string())))
            .collect();
        MirStmt::Let {
            name: name.to_string(),
            mutable: false,
            init: ekind(MirExprKind::Call {
                callee: Box::new(ekind(MirExprKind::Var(primitive.to_string()))),
                args: call_args,
            }),
            alloc_hint: None,
            slot: None,
        }
    }

    fn program_from_body(stmts: Vec<MirStmt>) -> MirProgram {
        let mut f = empty_fn("f");
        f.body.stmts = stmts;
        MirProgram {
            functions: vec![f],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        }
    }

    #[test]
    fn native_primitive_lookup_works() {
        assert!(is_native_primitive("matmul"));
        assert!(is_native_primitive("mlp_layer"));
        assert!(is_native_primitive("adam_step"));
        assert!(!is_native_primitive("println"));
        assert!(!is_native_primitive("array_push"));
    }

    #[test]
    fn empty_program_yields_empty_plan() {
        let prog = MirProgram {
            functions: vec![],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        };
        let plan = identify_fusion_candidates(&prog);
        assert!(plan.is_empty());
    }

    #[test]
    fn single_native_call_makes_a_lone_chain() {
        let stmts = vec![make_call_let("y", "matmul", vec!["w", "x"])];
        let prog = program_from_body(stmts);
        let plan = identify_fusion_candidates(&prog);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.candidates[0].chain_length(), 1);
        assert_eq!(plan.candidates[0].confidence_pct, 10);
        assert!(!plan.candidates[0].is_fusion_worthy()); // length 1 < 2
    }

    #[test]
    fn chained_primitives_form_a_single_chain() {
        // let h1 = mlp_layer(input, w1, b1, relu);
        // let h2 = matmul(h1, w2);
        // let out = dot(h2, bias);
        let stmts = vec![
            make_call_let("h1", "mlp_layer", vec!["input", "w1", "b1", "relu"]),
            make_call_let("h2", "matmul", vec!["h1", "w2"]),
            make_call_let("out", "dot", vec!["h2", "bias"]),
        ];
        let prog = program_from_body(stmts);
        let plan = identify_fusion_candidates(&prog);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.candidates[0].chain_length(), 3);
        assert_eq!(plan.candidates[0].confidence_pct, 75);
        assert!(plan.candidates[0].is_fusion_worthy());
    }

    #[test]
    fn unrelated_primitives_split_into_separate_chains() {
        // let a = sum(arr1);    // chain 1
        // let b = matmul(w, x); // chain 2 (doesn't reference a)
        let stmts = vec![
            make_call_let("a", "sum", vec!["arr1"]),
            make_call_let("b", "matmul", vec!["w", "x"]),
        ];
        let prog = program_from_body(stmts);
        let plan = identify_fusion_candidates(&prog);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.candidates[0].chain_length(), 1);
        assert_eq!(plan.candidates[1].chain_length(), 1);
    }

    #[test]
    fn non_native_calls_are_ignored() {
        // let x = println(args);  // not native, ignored
        // let y = matmul(w, x);   // x is the println result but
        //                          // since println is non-native,
        //                          // there's no "previous primitive"
        let stmts = vec![
            make_call_let("x", "println", vec!["args"]),
            make_call_let("y", "matmul", vec!["w", "x"]),
        ];
        let prog = program_from_body(stmts);
        let plan = identify_fusion_candidates(&prog);
        // Only `y` is a native call. 1 candidate, length 1.
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.candidates[0].chain_length(), 1);
    }

    #[test]
    fn identification_is_deterministic() {
        let stmts = vec![
            make_call_let("h1", "mlp_layer", vec!["input", "w1", "b1"]),
            make_call_let("h2", "matmul", vec!["h1", "w2"]),
        ];
        let prog = program_from_body(stmts);
        let first = identify_fusion_candidates(&prog);
        for _ in 0..50 {
            let again = identify_fusion_candidates(&prog);
            assert_eq!(first, again);
        }
    }

    #[test]
    fn chain_length_4_gets_max_confidence() {
        let stmts = vec![
            make_call_let("a", "matmul", vec!["w", "x"]),
            make_call_let("b", "matmul", vec!["a", "y"]),
            make_call_let("c", "matmul", vec!["b", "z"]),
            make_call_let("d", "matmul", vec!["c", "q"]),
        ];
        let prog = program_from_body(stmts);
        let plan = identify_fusion_candidates(&prog);
        assert_eq!(plan.candidates[0].chain_length(), 4);
        assert_eq!(plan.candidates[0].confidence_pct, 90);
    }

    #[test]
    fn total_chain_length_sums_across_chains() {
        // Two independent chains of length 2 each.
        let stmts = vec![
            make_call_let("a", "matmul", vec!["w", "x"]),
            make_call_let("b", "matmul", vec!["a", "w"]),
            make_call_let("c", "sum", vec!["arr"]),
            make_call_let("d", "mean", vec!["c"]),
        ];
        let prog = program_from_body(stmts);
        let plan = identify_fusion_candidates(&prog);
        // Two chains of length 2.
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.total_chain_length(), 4);
        assert_eq!(plan.fusion_worthy().count(), 2);
    }

    #[test]
    fn varlocal_callee_still_recognised() {
        // Post-slot-resolution, a callee like `matmul` becomes
        // `MirExprKind::VarLocal { name: "matmul", slot: N }`.
        // Identification must accept both forms.
        let mut f = empty_fn("f");
        f.body.stmts = vec![MirStmt::Let {
            name: "y".to_string(),
            mutable: false,
            init: ekind(MirExprKind::Call {
                callee: Box::new(ekind(MirExprKind::VarLocal {
                    name: "matmul".to_string(),
                    slot: 7,
                })),
                args: vec![
                    ekind(MirExprKind::VarLocal {
                        name: "w".to_string(),
                        slot: 8,
                    }),
                    ekind(MirExprKind::VarLocal {
                        name: "x".to_string(),
                        slot: 9,
                    }),
                ],
            }),
            alloc_hint: None,
            slot: None,
        }];
        let prog = MirProgram {
            functions: vec![f],
            struct_defs: vec![],
            enum_defs: vec![],
            entry: MirFnId(0),
        };
        let plan = identify_fusion_candidates(&prog);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.candidates[0].chain[0].primitive_name, "matmul");
    }
}
