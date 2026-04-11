---
title: What CJC-Lang Can Already Do
tags: [showcase]
status: Evidence-backed (grounded in README, examples, tests)
---

# What CJC-Lang Can Already Do

A grounded list of capabilities that are actually working today, with pointers to the evidence. This note is deliberately sober — it is not marketing.

## Language

- Write functions, structs, enums, classes, traits
- Closures with captured state ([[Capture Analysis]])
- Pattern matching with structural destructuring ([[Patterns and Match]])
- Control flow: `if`/`else`, `while`, `for` (the last desugared to `while`)
- Pipe expressions (`|>`) for data pipelines
- Format strings (`f"..."`) and regex literals (`/pat/flags`)
- Tensor literals (`[| ... |]`)

Evidence: examples in `examples/`, full test suite.

## Compiler

- Tokenize, parse, type-check, lower to HIR, lower to MIR, optimize (CF/DCE/CSE/SR/LICM/SCCP), execute via MIR-exec
- Alternative: skip MIR and tree-walk via cjc-eval
- Run the **same program** through both and get byte-identical output ([[Parity Gates]])
- Static verification of NoGC regions ([[NoGC Verifier]])
- Emit MIR for inspection (`cjcl emit`)

Evidence: `crates/cjc-mir/`, `tests/` parity suites, [[cjc-eval]] and [[cjc-mir-exec]].

## Numerics

- Deterministic floating-point reductions (Kahan, binned)
- Seeded RNG via [[SplitMix64]]
- Sort/argsort/rank with `total_cmp` — NaN-safe
- 24 distributions with PDF/CDF/PPF
- 24 hypothesis tests
- Linear algebra: matmul, det, solve, lstsq, eigh, svd, qr, cholesky
- FFT / RFFT / IFFT / PSD / windows
- Sparse CSR / CSC, sparse matmul, iterative sparse solvers

Evidence: `crates/cjc-runtime/src/stats.rs`, `linalg.rs`, `distributions.rs`, `hypothesis.rs`, `fft.rs`, `sparse*.rs`.

## ML

- Activation functions (relu, gelu, sigmoid, softmax, tanh, ...)
- Convolutional layers (1D and 2D) with zero-allocation fast paths
- Self-attention with KV cache
- Batch norm, layer norm, dropout (seeded)
- Adam optimizer
- Forward-mode and reverse-mode autodiff
- Physics-informed neural networks (PINN loss)

Evidence: `crates/cjc-runtime/src/ml.rs`, `crates/cjc-ad/`, examples `01–08`.

## Scientific computing

- Full quantum circuit simulator (statevector, MPS, DMRG, density, stabilizer, VQE, QAOA, QML, QEC) — [[Quantum Simulation]]
- ODE integration
- Sparse eigensolvers
- PINN for PDEs
- Clustering (k-means style)
- Interpolation
- Time series

Evidence: `crates/cjc-quantum/` (20 files), `crates/cjc-runtime/src/ode.rs`, `clustering.rs`, `interpolate.rs`, `timeseries.rs`.

## Data and visualization

- DataFrame DSL with ~73 tidy operations (filter, group_by, join, mutate, arrange, pivot, windows)
- CSV streaming (Kahan-stable)
- 80 chart types via [[Vizor]] producing deterministic SVG and BMP
- NFA regex (`cjc-regex`)
- Deterministic binary serialization with SHA-256 (`cjc-snap`)

Evidence: `crates/cjc-data/`, `crates/cjc-vizor/`, `crates/cjc-regex/`, `crates/cjc-snap/`, `gallery/`.

## CLI tooling

- ~30 subcommands including `run`, `repl`, `lex`, `parse`, `check`, `emit`, `inspect`, `trace`, `mem`, `bench`, `doctor`, `precision`, `parity`, `pack`, `explain`, `audit`, `ci`, ...

Evidence: `crates/cjc-cli/src/commands/`.

## Concrete demo: Chess RL

See [[Chess RL Demo]]. A complete chess engine + REINFORCE training loop implemented in pure CJC-Lang, running end-to-end through both executors with 216 tests including determinism and parity verification.

## Related

- [[Chess RL Demo]]
- [[Deterministic Workflow Examples]]
- [[Why CJC-Lang Matters]]
- [[Demonstrated Scientific Computing Capabilities]]
