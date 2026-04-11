---
title: Language Server
tags: [cli, tooling, lsp]
status: Experimental
---

# Language Server

**Crate**: `cjc-analyzer` — `crates/cjc-analyzer/src/`.

## Summary

An LSP-skeleton implementation for editor integration. **Experimental** — not integrated with the full type checker, limited functionality.

## Modules surveyed

| File | Role |
|---|---|
| `server.rs` | LSP protocol handler |
| `hover.rs` | Hover documentation + type info |
| `completion.rs` | Identifier and keyword completion |
| `diagnostics.rs` | Error/warning rendering with spans |
| `symbol_index.rs` | Symbol table for IDE navigation |
| `main.rs` | Standalone server binary |

## Status

- Server skeleton is present.
- Not currently the primary development path.
- Full type integration is **Needs verification** — likely partial.

## Roadmap

Better LSP integration is mentioned in `docs/spec/stage3_roadmap.md` as a P3 (future) item. See [[Roadmap]].

## Related

- [[CLI Surfaces]]
- [[Diagnostics]]
- [[Type Checker]]
- [[Roadmap]]
