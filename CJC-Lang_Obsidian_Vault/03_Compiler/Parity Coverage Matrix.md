---
title: Parity Coverage Matrix
tags: [compiler, parity, testing, reference]
status: Implemented (first pass, verified 2026-04-09)
---

# Parity Coverage Matrix

Which feature surfaces have explicit parity tests between [[cjc-eval]] (the tree-walk AST interpreter, v1) and [[cjc-mir-exec]] (the register-machine executor, v2). See [[Parity Gates]] for the underlying protocol.

**Parity test** = a test that runs the same program through both executors and asserts byte-identical (or bit-identical, for floats) output. The gate is enforced by `cargo test --workspace`; any divergence blocks merge.

## Dedicated parity files (179 tests total)

Files whose filename contains `parity` and whose purpose is explicitly dual-executor comparison:

| File | Tests | Feature surface |
|---|---|---|
| `tests/milestone_2_4/parity/mod.rs` | 15 | G-8 and G-10 roadmap gates: closures, patterns, for-loops, nested control flow |
| `tests/milestone_2_6/parity.rs` | 5 | Milestone 2.6 (records, traits, generics) |
| `tests/test_builtin_parity.rs` | 10 | Shared builtin dispatch surface |
| `tests/test_parity_stress.rs` | 11 | Stress / fuzz parity across large inputs |
| `tests/bench_v0_1/test_mir_opt_parity.rs` | 6 | **Optimizer parity** — ensures `--mir-opt` doesn't change results |
| `tests/byte_first/test_vm_runtime_parity.rs` | 14 | Byte-level VM/runtime contract |
| `tests/test_vizor_parity.rs` | 34 | Vizor deterministic rendering across executors |
| `tests/reinforcement_learning_tests/test_08_parity.rs` | 14 | REINFORCE training trajectories |
| `tests/mathematics_hardening_phase/test_10_parity.rs` | 17 | Math primitive determinism |
| `tests/beta_tests/hardening/test_dual_mode_parity.rs` | 8 | Beta-phase feature matrix |
| `tests/beta_hardening/test_phase1_parity.rs` | 7 | Hardening phase 1 regression set |
| `tests/cjc_v0_1_hardening/integration/test_wiring_parity.rs` | 23 | v0.1 wiring-pattern parity (the rule of three) |
| `tests/final_phase_hardening_before_vm/test_parity.rs` | 15 | Pre-VM-launch regression gate |
| **Total** |  | **179** |

## Non-dedicated parity checks

Many other test files exercise both executors without naming themselves "parity" — a `grep -il parity tests/` search finds 97 files that at least *mention* parity. These usually run both executors on a fixture and compare outputs inline.

High-signal examples:

- `tests/chess_rl_playability/test_pgn_import.rs` — PGN parse + legal move enumeration through both executors
- `tests/language_hardening/test_lh09_optimizer.rs` — language-level optimizer parity
- `tests/prop_tests/complex_props.rs` — property tests that both executors must satisfy
- `tests/beta_tests/prop/test_prop_determinism.rs` — determinism properties at the language level

## Coverage by feature surface

| Feature | Dedicated parity file? | Incidental coverage? | Notes |
|---|---|---|---|
| Lexer/Parser | — | ✓ | No semantic differences — AST is shared |
| Type checker | — | ✓ | Same type system for both backends |
| Closures | ✓ (milestone_2_4) | ✓ | G-8 gate |
| Pattern matching | ✓ (milestone_2_4) | ✓ | G-10 gate |
| For loops | ✓ (milestone_2_4) | ✓ | |
| `if` expression | — | ✓ (verified manually 2026-04-09) | See [[If as Expression]] |
| Records / traits | ✓ (milestone_2_6) | ✓ | |
| Builtins (core) | ✓ (test_builtin_parity) | ✓ | |
| Builtins (quantum) | **No dedicated file** | partial | Gap — worth adding |
| Tensors / linalg | — | ✓ (byte_first, mathematics_hardening) | Determinism via Kahan/Binned |
| DataFrame DSL | — | ? | **Gap**: no explicit parity file found |
| Autodiff | — | ? | **Gap**: AD runs primarily via cjc-eval, MIR integration still open |
| Vizor | ✓ (test_vizor_parity, 34 tests) | — | Strongest dedicated coverage outside milestones |
| Regex | — | ✓ (test_regex.rs) | |
| Snap / serialization | — | ✓ (lh11_snap) | |
| Module system | — | ? | **Gap**: module system runs through `run_program_with_modules`, which currently only uses the MIR executor path |
| MIR optimizer | ✓ (test_mir_opt_parity) | — | Critical — prevents optimizer from silently changing results |

## Known gaps (to schedule)

1. **DataFrame parity file** — DSL is large and complex; deserves a dedicated `test_data_parity.rs`.
2. **Quantum parity file** — 83 quantum builtins with no explicit parity coverage against a second executor.
3. **AD parity** — will naturally arrive with the MIR-level autodiff integration (see [[Open Questions]]).
4. **Module system parity** — today, multi-file programs only run through MIR-exec (`run_program_with_modules`). No second-executor variant exists yet; until it does, there can be no parity gate.

## How parity tests are written

The idiomatic shape of a parity test:

```rust
#[test]
fn parity_closures_capture_by_value() {
    let src = r#"
        fn main() -> i64 {
            let x: i64 = 10;
            let f = fn() -> i64 { x };
            f()
        }
    "#;
    let (program, _diags) = cjc_parser::parse_source(src);

    let eval_result  = cjc_eval::Interpreter::new(0).exec(&program);
    let mir_result   = cjc_mir_exec::run_program_with_executor(&program, 0).unwrap();

    assert_eq!(eval_result.value, mir_result.0, "parity divergence");
}
```

Any test that **does not call both executors on the same program** is not a parity test, even if it contains the word "parity."

## Related

- [[Parity Gates]]
- [[cjc-eval]]
- [[cjc-mir-exec]]
- [[MIR Optimizer]]
- [[Test Infrastructure]]
