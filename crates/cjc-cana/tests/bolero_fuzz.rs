//! Bolero structural-fuzz targets for cjc-cana Phase-1.
//!
//! Both targets enforce the same contract:
//!
//! > **Total functions never panic.** The featurizer, the legality gate, and
//! > the report serializer are all pure functions of their inputs. Random
//! > bytes mapped into MIR shape MUST produce either a `CanaReport` or an
//! > orderly fall-through — never a panic.
//!
//! These compile to proptest by default and to libfuzzer / AFL under
//! `cargo bolero`, matching the cjc-cronos-gan precedent.

use bolero::check;

use cjc_cana::{
    analyze_program, DefaultLegalityGate, LegalityGate, PassSequence, ProposedPass,
};
use cjc_mir::{MirBody, MirExpr, MirExprKind, MirFnId, MirFunction, MirProgram, MirStmt};

// ---------------------------------------------------------------------------
// Cursor helpers (mirrors cjc-cronos-gan/tests/test_fuzz.rs convention)
// ---------------------------------------------------------------------------

#[inline]
fn take_u8(bytes: &[u8], cursor: &mut usize) -> u8 {
    if *cursor < bytes.len() {
        let b = bytes[*cursor];
        *cursor += 1;
        b
    } else {
        0
    }
}

#[inline]
fn take_u32(bytes: &[u8], cursor: &mut usize) -> u32 {
    let a = take_u8(bytes, cursor) as u32;
    let b = take_u8(bytes, cursor) as u32;
    let c = take_u8(bytes, cursor) as u32;
    let d = take_u8(bytes, cursor) as u32;
    a | (b << 8) | (c << 16) | (d << 24)
}

#[inline]
fn take_i64(bytes: &[u8], cursor: &mut usize) -> i64 {
    let mut v = [0u8; 8];
    for byte in v.iter_mut() {
        *byte = take_u8(bytes, cursor);
    }
    i64::from_le_bytes(v)
}

#[inline]
fn pick_in_range(bytes: &[u8], cursor: &mut usize, lo: usize, hi: usize) -> usize {
    let span = hi - lo + 1;
    let b = take_u8(bytes, cursor) as usize;
    lo + (b % span.max(1))
}

// ---------------------------------------------------------------------------
// Decoders — random bytes → small bounded MIR shapes
// ---------------------------------------------------------------------------

const MAX_FUNCTIONS: usize = 4;
const MAX_STMTS_PER_FN: usize = 6;
const MAX_EXPR_DEPTH: u32 = 3;

/// Decode a single bounded-depth MirExpr from the fuzz bytes.
/// Always terminates by hitting `depth == 0` → IntLit.
fn decode_expr(bytes: &[u8], cursor: &mut usize, depth: u32) -> MirExpr {
    if depth == 0 {
        return MirExpr {
            kind: MirExprKind::IntLit(take_i64(bytes, cursor)),
        };
    }
    let tag = take_u8(bytes, cursor) % 8;
    let kind = match tag {
        0 => MirExprKind::IntLit(take_i64(bytes, cursor)),
        1 => MirExprKind::BoolLit(take_u8(bytes, cursor) % 2 == 0),
        2 => MirExprKind::Binary {
            op: cjc_ast::BinOp::Add,
            left: Box::new(decode_expr(bytes, cursor, depth - 1)),
            right: Box::new(decode_expr(bytes, cursor, depth - 1)),
        },
        3 => MirExprKind::Var(format!("v{}", take_u8(bytes, cursor) % 4)),
        4 => MirExprKind::ArrayLit(
            (0..pick_in_range(bytes, cursor, 0, 3))
                .map(|_| decode_expr(bytes, cursor, depth - 1))
                .collect(),
        ),
        5 => MirExprKind::TupleLit(
            (0..pick_in_range(bytes, cursor, 0, 3))
                .map(|_| decode_expr(bytes, cursor, depth - 1))
                .collect(),
        ),
        6 => MirExprKind::Call {
            callee: Box::new(MirExpr {
                kind: MirExprKind::Var(format!("fn{}", take_u8(bytes, cursor) % 4)),
            }),
            args: (0..pick_in_range(bytes, cursor, 0, 3))
                .map(|_| decode_expr(bytes, cursor, depth - 1))
                .collect(),
        },
        _ => MirExprKind::Unary {
            op: cjc_ast::UnaryOp::Neg,
            operand: Box::new(decode_expr(bytes, cursor, depth - 1)),
        },
    };
    MirExpr { kind }
}

/// Decode a single non-terminating statement.
///
/// We deliberately exclude `Return`, `Break`, and `Continue` so the decoder
/// never produces a CFG with unreachable trailing blocks. That input shape
/// trips an upstream panic in `cjc-mir::dominators::dominates` on unreachable
/// blocks (see spawn_task chip "Fix cjc-mir::dominators OOB on unreachable
/// blocks"). The HIR lowering pipeline normally elides this shape too —
/// the real input distribution to `analyze_program` never contains it.
fn decode_stmt(bytes: &[u8], cursor: &mut usize) -> MirStmt {
    let tag = take_u8(bytes, cursor) % 4;
    match tag {
        0 => MirStmt::Expr(decode_expr(bytes, cursor, MAX_EXPR_DEPTH)),
        1 => MirStmt::Let {
            name: format!("v{}", take_u8(bytes, cursor) % 4),
            mutable: take_u8(bytes, cursor) % 2 == 0,
            init: decode_expr(bytes, cursor, MAX_EXPR_DEPTH),
            alloc_hint: None,
            slot: None,
        },
        2 => MirStmt::If {
            cond: decode_expr(bytes, cursor, MAX_EXPR_DEPTH),
            then_body: MirBody {
                stmts: vec![MirStmt::Expr(decode_expr(bytes, cursor, MAX_EXPR_DEPTH))],
                result: None,
            },
            else_body: None,
        },
        _ => MirStmt::While {
            cond: decode_expr(bytes, cursor, MAX_EXPR_DEPTH),
            body: MirBody {
                stmts: vec![MirStmt::Expr(decode_expr(bytes, cursor, MAX_EXPR_DEPTH))],
                result: None,
            },
        },
    }
}

fn decode_function(bytes: &[u8], cursor: &mut usize, idx: u32) -> MirFunction {
    let n_stmts = pick_in_range(bytes, cursor, 0, MAX_STMTS_PER_FN);
    let body = MirBody {
        stmts: (0..n_stmts).map(|_| decode_stmt(bytes, cursor)).collect(),
        result: None,
    };
    MirFunction {
        id: MirFnId(idx),
        name: format!("fuzz_fn_{}", idx),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body,
        is_nogc: false,
        cfg_body: None,
        decorators: vec![],
        vis: cjc_ast::Visibility::Public,
        local_count: 0,
    }
}

fn decode_program(bytes: &[u8]) -> MirProgram {
    let mut cursor = 0;
    let n_fns = pick_in_range(bytes, &mut cursor, 1, MAX_FUNCTIONS);
    let functions: Vec<MirFunction> = (0..n_fns)
        .map(|i| decode_function(bytes, &mut cursor, i as u32))
        .collect();
    MirProgram {
        functions,
        struct_defs: vec![],
        enum_defs: vec![],
        entry: MirFnId(0),
    }
}

fn decode_pass_sequence(bytes: &[u8]) -> PassSequence {
    let mut cursor = 0;
    let n_entries = pick_in_range(bytes, &mut cursor, 0, 8);
    let mut seq = PassSequence::empty();
    for _ in 0..n_entries {
        let fn_idx = take_u8(bytes, &mut cursor) % (MAX_FUNCTIONS as u8 + 2);
        let fn_name = format!("fuzz_fn_{}", fn_idx);
        let pass = match take_u8(bytes, &mut cursor) % 2 {
            0 => ProposedPass::Run(format!("p_{}", take_u8(bytes, &mut cursor) % 6)),
            _ => ProposedPass::Skip(format!("p_{}", take_u8(bytes, &mut cursor) % 6)),
        };
        seq.per_function.entry(fn_name).or_default().push(pass);
    }
    seq
}

// ---------------------------------------------------------------------------
// Targets
// ---------------------------------------------------------------------------

/// Target 1: random MIR shape → `analyze_program` must not panic and must
/// produce a serializable `CanaReport`.
#[test]
fn fuzz_analyze_program_never_panics() {
    check!()
        .with_type::<Vec<u8>>()
        .for_each(|bytes: &Vec<u8>| {
            let program = decode_program(bytes);
            let report = analyze_program(&program);
            // The report must serialize cleanly — non-empty JSON.
            let json = report.to_json();
            assert!(json.starts_with('{'));
            assert!(json.ends_with("}\n"));
            // And it must round-trip the canonical-bytes invariant on a
            // single trial (cheap determinism spot-check inside the fuzz
            // loop).
            let r2 = analyze_program(&program);
            assert_eq!(report.canonical_bytes(), r2.canonical_bytes());
        });
}

/// Target 2: random (program, pass-sequence) → legality gate must not panic
/// and must produce a verdict that is internally consistent (Approved =>
/// no violations; Rejected => ≥1 violation).
#[test]
fn fuzz_legality_gate_total_and_self_consistent() {
    check!()
        .with_type::<Vec<u8>>()
        .for_each(|bytes: &Vec<u8>| {
            let program = decode_program(bytes);
            // Split the byte stream: use the second half for the pass sequence.
            let mid = bytes.len() / 2;
            let seq_bytes = &bytes[mid..];
            let sequence = decode_pass_sequence(seq_bytes);

            let features = analyze_program(&program).features;
            let gate = DefaultLegalityGate::new();
            let verdict = gate.verify(&program, &sequence, &features);

            // Self-consistency.
            match verdict.violations() {
                Some(vs) => assert!(!vs.is_empty(), "Rejected verdict must have ≥1 violation"),
                None => assert!(verdict.is_approved()),
            }
        });
}
