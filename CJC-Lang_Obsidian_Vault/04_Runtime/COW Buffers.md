---
title: COW Buffers
tags: [runtime, memory]
status: Implemented
---

# COW Buffers

Source: `crates/cjc-runtime/src/buffer.rs`.

## Summary

Copy-on-write reference-counted buffers that back tensors, strings, and large arrays. Two values can share the same storage; on mutation, the writer gets a fresh copy.

## Why COW

- **No aliasing surprises** — user programs can't accidentally mutate a tensor they don't own.
- **Cheap functional style** — `let t2 = t1; ... t2 + 1 ...` doesn't copy until the add happens.
- **Determinism-friendly** — sharing is structural, not timing-dependent.

## Interaction with `array_push`

From CLAUDE.md: `array_push(arr, val)` returns a **new** array rather than mutating in place. Idiom:

```cjcl
arr = array_push(arr, val);
```

This is consistent with COW: if the buffer is shared, the push copies; if it's unique, it grows in place. The API elides the distinction by always returning the resulting value.

## Related

- [[Memory Model]]
- [[Tensor Runtime]]
- [[Value Model]]
