//! D-HARHT — Deterministic Hash-Array Radix-Hybrid Trie.
//!
//! Vendored verbatim from the D-HARHT reference implementation
//! (memory profile) for use as a deterministic route-memoization
//! cache in `cjc-abng` (Phase 0.10 Track Q1). Pure `std`, zero
//! external dependencies — in keeping with the CJC-Lang workspace's
//! zero-external-deps invariant.
//!
//! The engine is generic (`DHarht<V>`). `cjc-abng::route_cache` wraps
//! it as a `prefix -> RouteEvidence` memoizer behind
//! `AdaptiveBeliefGraph::descend`. The lookup pipeline is fully
//! deterministic: a fixed splitmix64-style scatter, a power-of-two
//! shard jump table, a sealed sparse front directory, and an adaptive
//! radix fallback — with a full key-equality check on every
//! successful lookup, so a hit is never a false positive.

pub mod dharht;

pub use dharht::{deterministic_permutation_scatter, DHarht, LookupProfile};
