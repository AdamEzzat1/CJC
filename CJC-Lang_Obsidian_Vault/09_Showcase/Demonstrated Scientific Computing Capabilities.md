---
title: Demonstrated Scientific Computing Capabilities
tags: [showcase, scientific]
status: Evidence-backed
---

# Demonstrated Scientific Computing Capabilities

Grounded in the `examples/` directory, `tests/` directory, and the benchmarks documented in the performance manifesto.

## ML workloads

| Example | What it exercises |
|---|---|
| `01_mlp_xor.cjcl` | MLP + XOR — the hello-world of neural networks |
| `02_rnn_sequence.cjcl` | RNN sequence modeling |
| `03_cnn1d_signal.cjcl` | 1D CNN for signal processing |
| `04_cnn2d_image.cjcl` | 2D CNN for image-like data (NCHW) |
| `05_transformer_attention.cjcl` | Self-attention with KV cache |
| `06_reinforcement_learning.cjcl` | Policy gradient RL |
| `07_physics_informed_ml.cjcl` | Physics-informed ML |
| `08_pinn_heat_equation.cjcl` | PINN solving the heat equation |
| `09_quantum_simulation.cjcl` | Quantum circuit simulation |

## NLP

| Example | What it exercises |
|---|---|
| `nlp_tokenize.cjcl` | Tokenization pipeline |
| `nlp_vocab_count.cjcl` | Vocabulary counting |

## Data engineering

| Example | What it exercises |
|---|---|
| `etl_csv_parse.cjcl` | CSV → DataFrame ETL (Kahan-stable streaming, zero-dep) |
| `transformer_forward.cjcl` | Transformer forward pass (standalone) |

## Visualization

| Example | What it exercises |
|---|---|
| `vizor_scatter.cjcl` | Scatter plot |
| `vizor_line.cjcl` | Line plot |
| `vizor_bar.cjcl` | Bar chart |
| `vizor_histogram.cjcl` | Histogram |
| `vizor_annotated.cjcl` | Annotated plot with text + shapes |

## Quantum

`09_quantum_simulation.cjcl`, plus the `cjc-quantum` test suite (`tests/bench_50q.rs`).

Advanced: VQE, QAOA, stabilizer circuits, DMRG, QEC — see [[Quantum Simulation]] for the full list.

## Performance benchmarks (from manifesto)

`docs/spec/CJC_PERFORMANCE_MANIFESTO.md` claims zero-allocation inference with:
- RNN 10K steps in ~2.5s (~3,995 steps/sec)
- Transformer ~562 tokens/sec
- Binary footprint: 1.8 MB

**Needs verification**: these numbers are from past benchmarks in the manifesto. They should be re-run before being cited externally.

## End-to-end: Chess RL

[[Chess RL Demo]] is the largest integration test: a full RL training pipeline in pure CJC-Lang with 216 dedicated tests covering determinism, parity, and correctness.

## Related

- [[What CJC-Lang Can Already Do]]
- [[Tensor and Scientific Computing]]
- [[Scientific Computing Concept Graph]]
- [[ML Primitives]]
- [[Quantum Simulation]]
