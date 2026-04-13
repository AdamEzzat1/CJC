---
title: Binned Allocator
tags: [runtime, memory]
status: Implemented
---

# Binned Allocator

Source: `crates/cjc-runtime/src/binned_alloc.rs`.

## Summary

A size-class binned allocator for large heap objects. Per `docs/memory_model_2_0.md`, there are **13 size-class bins**; allocations round up to the next bin, which keeps fragmentation bounded and makes allocator latency predictable.

## Why

- **Predictable latency** — fixed set of free lists by size class.
- **Low fragmentation** — objects of the same size class share a common free list.
- **Pool-friendly** — buffers can be returned to the class they came from.

## Not to be confused with Binned Accumulator

This is about memory allocation. [[Binned Accumulator]] is about *numerical summation* — completely different subsystem, shared name out of coincidence.

## Related

- [[Memory Model]]
- [[Frame Arena]]
- [[COW Buffers]]
- [[Binned Accumulator]] — different thing
