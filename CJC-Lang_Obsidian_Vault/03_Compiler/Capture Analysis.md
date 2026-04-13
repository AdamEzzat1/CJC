---
title: Capture Analysis
tags: [compiler, closures]
status: Implemented
---

# Capture Analysis

Source: `crates/cjc-hir/src/lib.rs`. Ran during AST → [[HIR]] lowering.

## Summary

For every closure in the program, determine which enclosing-scope variables it reads or writes, and how they should be captured.

## What it produces

A `HirCapture` list on each closure, with a `CaptureMode`:
- `ByRef` — captured as a reference into the enclosing frame.
- `ByClone` — captured as a deep clone at closure creation.

Both executors ([[cjc-eval]] and [[cjc-mir-exec]]) use this list at closure construction time to build the closure value without reanalyzing scopes.

## Why it's in HIR

By the time we reach [[MIR]], we want every variable access to be either a register or a known capture slot. Capture analysis has to happen before [[MIR]] lowering so that MIR can lay out captures as addressable slots.

## Relationship to Escape Analysis

Captures interact with [[Escape Analysis]] — a local that is captured `ByRef` escapes the current frame, which means its backing storage must live long enough. This can downgrade the allocation hint from `Stack` to `Arena` or `Rc`.

## Related

- [[HIR]]
- [[Closures]]
- [[Escape Analysis]]
- [[cjc-eval]]
- [[cjc-mir-exec]]
