# Postulate Phase 1 — the compiler side of the IDE

Phase 1 makes CJC able to tell an IDE what a training run did: which line built
each autodiff node, which layer it belongs to, which tensor ops ran off the
tape, how the loss and gradients moved, and whether the two executors agree.
Everything lives in CJC; CJC does not depend on Postulate.

All of it is opt-in and read-only with respect to the computation. Each claim
below has a gate, and each gate has a poison: a real code change that the gate
must catch, kept as a patch file with its recorded failure.

## What was built

| Piece | Where |
|---|---|
| Call spans through the pipeline: `HirExprKind::Call.span`, `MirExprKind::Call.span` (`Option<cjc_ast::Span>`; `None` for synthesized calls) | `cjc-hir`, `cjc-mir` (+ `monomorph` preserves it) |
| A span, a group and off-tape records per `GradGraph` node; `GradGraph::export()`, `GradGraph::state_bits()` | `crates/cjc-ad/src/{lib,export}.rs` |
| `Tracker`: the executor-side hooks, shared by both executors so they agree by construction | `crates/cjc-ad/src/tracker.rs` |
| `g.layer("name")` / `g.layer("")`: name the layer group for the following ops | both executors' `dispatch_method` |
| Per-step telemetry stream (JSON lines, sampled, ring-buffered, append-only file) | `crates/cjc-ad/src/telemetry.rs` |
| `echo` flag on both executors, so `print` stays off stdout under the LSP | `cjc-eval`, `cjc-mir-exec` |
| LSP: in-process `Analyzer` core; `postulate/*` requests; tensor-shape inlay hints | `crates/cjc-analyzer/src/{server,postulate}.rs` |
| Gates | `tests/postulate_phase1.rs`, `crates/cjc-analyzer/tests/replay.rs`, `bench/postulate_overhead` |
| Poisons | `docs/postulate/poison/` (`run_poisons.py` applies, tests, reverts) |

### Spans and groups

The executors set `tracker.span` when a call dispatches. That happens **after
the arguments are evaluated**, and for a method call again after its receiver
is evaluated, so a nested call never leaves its span behind. Every node a
`GradGraph` method records is stamped with that span. Spans are always stamped,
because the cost is one `Option` per node. Tracking adds three things, and
only when `ad.enabled`:

- **Groups.** A node's group is the innermost named `g.layer(...)`, or else the
  enclosing user function. `__main` and top level both count as `main`.
- **The last trained graph.** This is the graph `backward` last ran on, and it
  is what `export()` returns.
- **Off-tape ops.** A builtin or tensor method whose operand is the very buffer
  (`Buffer::ptr_eq`) handed to the current graph by `input`/`parameter`. Such
  an op is kept only if the node feeds nothing on the tape, since then it
  cannot be trained.

### `GraphExport` JSON

`GraphExport::to_json()` is deterministic. Fields come in a fixed order, and
non-finite numbers are written as strings.

```json
{"nodes":[{"id":0,"op":"Parameter","inputs":[],"shape":[2,4],
           "span":{"file":0,"start":812,"end":830},"group":0,"grad_norm":0.41}, …],
 "groups":[{"id":0,"name":"main","nodes":[0,1,2,…]}, …],
 "off_tape":[{"op":"Tensor.conv2d","span":{…},"node":0}]}
```

### Telemetry

```rust
interp.ad.telemetry = Some(Telemetry::to_file("run.jsonl", seed, DEFAULT_EVERY)); // every 10 steps
```

One `backward` is one step. A sampled step writes one line:

```text
{"step":120,"seed":42,"loss":0.0213,"grad_norms":{"n0":0.084,"n1":0.11},"non_finite":[[3,2]]}
```

`grad_norms` is keyed by parameter node. `non_finite` lists `[node, count]` for
nodes whose forward value holds NaN or Inf. Lines collect in a 64-record ring,
which is flushed when full, on `flush()` and on drop. File errors are swallowed:
telemetry must never fail a run.

### LSP

Each custom request takes `{ "textDocument": { "uri" }, "seed"?: u64 }`, with
seed 42 by default. A document runs once per (version, seed) in both executors,
with echo off, tracking on and in-memory telemetry. Panics are caught and
reported as `error`.

| Method | Result |
|---|---|
| `postulate/gradGraph` | `{seed, executor, error, graph: GraphExport, param_names: [{node, name}]}` |
| `postulate/snapshotHash` | `{seed, eval: {hash, error}, mir: {hash, error}}` (cjc-snap SHA-256 of printed output) |
| `postulate/executorAgreement` | `{seed, executors_agree, output_agree, graph_agree, eval_hash, mir_hash, eval_error, mir_error}` |
| `postulate/runReport` | `{seed, runs: 5, wall_ms_median, wall_ms_spread, measured: [...], telemetry_steps, graph_nodes, peak_bytes: null, copies: null, hottest_zone: null, error}` |
| `textDocument/inlayHint` | `: [5, 8]` at the end of each recording call, from the last trained graph |

Timing fields are **measured**, and every response lists them under
`"measured"`. The three polytrace fields are `null` until Phase 5.

## Gates

| Gate | Test | Result | Poison (live code change) | Poison result |
|---|---|---|---|---|
| **Spans** cover the exact creating call (01_mlp_xor, both executors), checked against an independent oracle that scans `g.<op>(` calls | `gate_spans_cover_the_creating_call_{eval,mir}` | pass | `01` `Tracker::call_span` (the one span constructor both executors use) shifts every start by 1 byte | spans-eval and spans-mir FAIL; parity *passes* (both moved alike), which is why spans have their own gate |
| Span of a method whose receiver is itself a call | `span_is_the_method_call_when_its_receiver_is_itself_a_call` | pass | `02` MIR skips the post-receiver span reset | FAIL: got `same(g)` |
| **Executor parity**: byte-identical `GraphExport` JSON (xor, cnn with off-tape) | `gate_executor_parity_{xor,cnn_off_tape}` | pass | `02b` MIR stamps no spans | spans-mir and both parity gates FAIL |
| **Telemetry parity**: same cjc-snap output hash **and** identical trained-graph bits with telemetry on/off. Covers xor, linreg and tanh_fit, plus cnn (no backward); every ∈ {1, 10}; both executors | `gate_telemetry_parity_three_programs` | pass | `03` the probe flips one gradient's last bit | FAIL (trained-graph bits) |
| **LSP replay**: a 13-request scripted session gives identical transcripts over 10 fresh runs, with `measured` fields masked | `gate_lsp_replay_is_identical_over_ten_runs` | pass | `04` the analyzer adds a timestamp field to `gradGraph` | FAIL on the `gradGraph` response |
| **Overhead**: telemetry at the default rate costs **≤ N = 15 %** on mlp_xor | `bench/postulate_overhead -- 101 --check 15` | pass ×2 (see `overhead_gate.txt`) | `05` telemetry builds its record every step, ignoring sampling | FAIL: +30–31 % > 15 % |

The in-test poison checks (`poison_*` in the test files) also stay in the
suite. They feed each checker a perturbed input and assert that it rejects it.

### What the poisons taught

The poisons were run before the gates were trusted. Two did not fail at first,
and both exposed a gate problem, not a poison problem:

1. **Telemetry parity hashed only printed output.** `to_string` rounds, so a
   probe that nudged a gradient by one ulp passed. The gate now also compares
   `GradGraph::state_bits()`: every tape value and gradient of the last trained
   graph, bit for bit. Its weights carry every earlier step's update.
2. **A defensive line had no test.** Resetting the span after a method's
   receiver is evaluated only matters when the receiver is itself a call.
   Removing it broke nothing until the receiver-call test was added.

### Overhead: how N was set

Measured on Windows 11 with a release build, 101 interleaved runs per mode and
two warm-up rounds, timing `exec` only. The A/A noise floor is the gap between
the medians of the `off` runs' even and odd halves.

- **N was set from the first measurements**, taken on a full-batch-bias
  variant of mlp_xor (~17–21 ms per run). There, writing to a file at the
  default rate added a near-constant **~1.2–1.5 ms per run**: the one-time file
  create/open, not per-step work. That came to +4.2 % … +13.7 % over 12
  measurements, so **N = 15 %**. In-memory collection (`mem10`) was within the
  noise floor.
- **Re-measured on the unmodified example** once the broadcast-backward fix let
  it run (~7–8 ms per run, `overhead_gate.txt`). The default rate to a file
  cost **+8.8 % … +11.9 %**, so the gate still passes. In-memory collection is
  **+4.8 … +8.9 %** here, against A/A floors of 0.6–4.2 %, so on this shorter
  program collection is measurable rather than free.
- On the ~5 ms fixtures (linreg, tanh_fit) the fixed file cost reads as
  +20–40 %. Their noise floor is 1–24 %, so they are reported, not gated.
- **Every step to a file** costs ~10–15 µs per record, +26–41 % on mlp_xor.
- **Tracking** (`ad.enabled`, which the LSP runner uses) costs +2–17 %, near
  the noise. It is not gated.
- The gate refuses to judge (exit 2) when the gated row's own A/A floor exceeds
  5 %. It never passes on noise, and one such refusal happened during the
  measurement. A failure always exits 1.
- **Not measured:** the always-on span stamp against the pre-Phase-1 tree. The
  `off` mode already includes it.

## Claim ledger

| Claim | Status | Evidence |
|---|---|---|
| Telemetry never changes results | **verified** on 4 programs × 2 rates × 2 executors, by output hash and trained bits | gate 3 + poison 03 |
| Tracking never changes results | **verified** on 2 programs × 2 executors (printed output) | `tracker_on_and_off_print_the_same_output` |
| Both executors export identical graphs | **verified** on xor, cnn, tanh_fit | gate 2 + poison 02b |
| Every exported node's span covers its creating call | **verified** on `examples/01_mlp_xor.cjcl`, unmodified | gate 1 + poisons 01, 02 |
| LSP responses are deterministic | **verified** for the scripted session, timing excluded by the response's own `measured` list | gate 4 + poison 04 |
| Telemetry overhead ≤ 15 % at the default rate | **measured**, one machine, on a 7–21 ms run; mostly a fixed file open | gate 5 + poison 05 |
| `wall_ms_*` in `runReport` | **measured**, never reproducible, labelled | — |

## Known gaps

- **The ambient `grad_graph_*` builtins** (`cjc_ad::dispatch_grad_graph`) are
  not spanned. Only the method API (`g.parameter(…)`) is.
- **`SourceSpan.file` is always 0.** Single-file only; multi-file ids need
  `cjc-module`.
- **Off-tape detection** covers builtins and tensor methods, not binary
  operators (`x * k`). It also matches on the exact buffer, so a reshaped or
  copied tensor is not recognized.
- `grad_norm` reads 0 before the first `backward`, because the gradient slots
  start at zero.
- Telemetry `grad_norms` are keyed `n<id>`. Names come from
  `postulate/gradGraph`'s `param_names`.
- `postulate/runReport` re-runs the program 5 times on the MIR executor per
  request, uncached by design: it measures.
- `peak_bytes`, `copies` and `hottest_zone` are `null` until polytrace is wired
  in (Phase 5).

## Found along the way

### Fixed: gradients through broadcasting ops (`cjc-ad`)

`examples/01_mlp_xor.cjcl` panicked at HEAD in `GradGraph::backward` with
`Tensor::add_assign shape mismatch [1,1] vs [4,1]`. Forward `add`/`sub`/`mul`/`div`
broadcast with NumPy rules, but backward handed the full broadcast-shaped
gradient to each operand. A `[1, n]` bias got an `[m, n]` gradient, and since
c4d0a83 (gradients accumulate in place) that mismatch panics. Before c4d0a83
it most likely gave a silently wrong `[m, n]` bias gradient.

- `GradGraph::accumulate_operand_grad` reduces each operand's gradient to that
  operand's shape with `cjc_ad::reduce_to_shape`. That sums over the axes the
  operand broadcast along, one `sum_axis` (binned accumulation) per axis, in
  axis order. It is applied in `backward` and `backward_with_seed`.
- A same-shape gradient takes the old path unchanged: no reduction, no copy.
  Programs that don't broadcast keep their bits.
- In the same function, `StructField`/`MapLookup` backward passed the whole
  gradient to the parent, which also panicked
  (`milestone_2_5::diff_containers`). It now gives the parent a one-hot
  gradient on the accessed chunk, the same `chunk_grad` that
  `backward_with_seed` already used.
- Tests are in `crates/cjc-ad/tests/broadcast_backward.rs`: 4 ops × 7 shape
  pairs against central finite differences, in `backward` and
  `backward_with_seed`, with bit-identical repeat runs, plus the exact bias
  case and container chunks. With the reduction disabled, 4 of the 7 fail.
- The gates now use the shipped example. `tests/postulate/mlp_xor_fullbatch.cjcl`
  is gone. `cjcl parity examples/01_mlp_xor.cjcl`: IDENTICAL.

### Not fixed

- **`examples/cnn_example.cjcl` does not run** in either executor
  (`undefined variable GradGraph` / `Tensor` for its `Tensor.conv2d(…)` and
  `GradGraph.input(g, …)` forms). `tests/postulate/cnn_off_tape.cjcl` is a
  version that runs.

## Regression gate

`cargo test --workspace --no-fail-fast` on this tree, with the broadcast and
container gradient fix: **12,180 passed, 13 failed, 126 ignored**. A clean
checkout of `HEAD` (1a0679f) failed 15. This tree fails 13 of those 15 and
nothing else, so this phase adds no failures. The 13 are:

- 9 `BlrStateHashMismatch` or canary-hash failures in the ABNG suites
  (lineage, PINN, tabular GP, chess RL v2.6)
- 2 builder assertions in `language_hardening::test_lh01_diagnostics`
- the `cjc-diag` `ErrorCode::explanation` doctest
- `qaoa_4_cycle_finds_good_cut`

The other two failures on `HEAD` were in `milestone_2_5::diff_containers`
(`diff_map_lookup_gradient_flows_to_map_node` and
`diff_reverse_mode_add_gradient_accumulation`). These were the
`StructField`/`MapLookup` backward stub, which sent the whole gradient to the
parent instead of to the accessed chunk. The same change that fixed
broadcasting replaced that stub, and both tests now pass.

## Running it

```bash
cargo test --test postulate_phase1
cargo test -p cjc-analyzer --test replay
cargo run --release -p postulate-overhead -- 101 --check 15
python -I docs/postulate/poison/run_poisons.py        # every poison must FAIL its gate
```
