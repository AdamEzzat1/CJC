---
title: ML Primitives
tags: [runtime, ml]
status: Implemented
---

# ML Primitives

**Source**: `crates/cjc-runtime/src/ml.rs` (~69K).

## Summary

A collection of machine learning building blocks implemented directly on top of [[Tensor Runtime]]. No PyTorch, no TensorFlow, no ONNX — everything is in-house.

## Activation functions

- `relu`, `leaky_relu`, `elu`
- `sigmoid`, `tanh`
- `gelu`, `silu`, `softplus`
- `softmax` (two-pass, numerically stable)

## Layers

- `linear` (dense / fully connected) — routes through matmul
- `conv1d`, `conv1d_raw`, `conv1d_circular`
- `conv2d` — 4D NCHW kernels with `u64` stride arithmetic per performance manifesto
- `maxpool1d_raw`, `maxpool2d`
- `batch_norm`, `layer_norm`
- `dropout_mask` — deterministic with seed
- `embedding`
- `attention` — self-attention and cross-attention

## Losses

- `binary_cross_entropy`
- `mse`, `mae`
- Categorical cross-entropy

## Optimizers

- `Adam.new(lr, betas, eps)` — Adam optimizer
- SGD (via direct weight update)

## Transformer support

From performance manifesto and example `05_transformer_attention.cjcl`:
- Self-attention with proper masking
- KV caching via `paged_kv.rs` in `cjc-runtime` — vLLM-style paged block cache
- Transformer forward pass runs inside NoGC-verified regions per manifesto benchmarks

## Zero-allocation inference

The performance manifesto (`docs/spec/CJC_PERFORMANCE_MANIFESTO.md`) claims:
- RNN 10K steps in ~2.5s (~3,995 steps/sec) — **Needs verification** of current numbers
- Transformer ~562 tokens/sec
- 1.8 MB binary size
- Zero allocation once warm

This is made possible by:
- Pre-allocated weight tensors
- Reused scratch buffers via `tensor_pool.rs`
- `@nogc` verification on the forward pass
- Tiled matmul (`tensor_tiled.rs`) to stay L2-resident

## Related

- [[Tensor Runtime]]
- [[Linear Algebra]]
- [[Autodiff]]
- [[NoGC Verifier]]
- [[Memory Model]]
- [[Performance Profile]]
- [[Chess RL Demo]]
