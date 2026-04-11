---
title: Binary Serialization
tags: [data, determinism]
status: Implemented
---

# Binary Serialization

**Crate**: `cjc-snap` — `crates/cjc-snap/src/` (~29K lib plus encode, decode, hash).

## Summary

A deterministic binary format for serializing CJC-Lang runtime values — tensors, DataFrames, structs, enums, arrays — with SHA-256 integrity hashing.

## Features

- **Content-addressable blobs** — hash of the payload is the identity.
- **SHA-256** implemented in-house (zero dependencies).
- **Chunk-based tensor encoding** — large tensors are split into chunks, each independently verifiable.
- **Sparse CSR encoding** — CSR matrices have their own format for compactness.
- **Categorical dictionary encoding** — factor columns share a single level dictionary.
- **DataFrame persistence** — whole DataFrame round-trips bit-identically.

## Determinism

- NaN canonicalized to `0x7FF8_0000_0000_0000`.
- Field order from `BTreeMap` (sorted).
- No timestamp, no random UUID, no thread ID in the payload.

This is what makes snapshot-based testing possible: you can save a result to disk, commit it, and any regression produces a different hash immediately.

## API

- `snap_encode(value) -> Vec<u8>`
- `snap_decode(bytes) -> Value`
- Hash-addressable store patterns (see source).

## Used by

- Test fixtures (goldens in `tests/golden/`)
- CLI `pack` command — see [[CLI Surfaces]]
- Model weights persistence (`models/` directory at repo root)

## Related

- [[Determinism Contract]]
- [[Value Model]]
- [[Total-Cmp and NaN Ordering]]
- [[CLI Surfaces]]
