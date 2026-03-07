# ADR: CJC Library Model

**Status:** Accepted
**Date:** 2026-03-07
**Context:** Design decision for how CJC integrates optional library modules.

## Decision

CJC uses an **import-gated library model** where:

1. Libraries are implemented as Rust crates in the workspace
2. Library functionality is accessed through `import <name>` declarations
3. The interpreter scans `DeclKind::Import` nodes to populate `libraries_enabled`
4. Dispatch functions check the enabled set before resolving calls

## Rationale

### Why not always-available builtins?

- Scientific computing libraries (vizor, future stats/ML libs) add many symbols
- Polluting the global namespace makes the core language harder to learn
- Import gating lets the NoGC verifier and optimizer skip library code paths

### Why not dynamic loading / plugins?

- CJC targets zero external dependencies for reproducibility
- Static linking in the Rust workspace provides compile-time safety
- The type-erased `Value` variant pattern (`Rc<dyn Any>`) gives enough flexibility

### Why `import <name>` without paths?

- CJC's library universe is small and curated
- No package manager, no version resolution needed
- Simple flat namespace avoids the complexity of nested modules

## Implementation Pattern

### Adding a new library

1. Create `crates/cjc-<name>/` with dispatch + domain logic
2. Add a `Value::<Name>` variant to `cjc-runtime` (type-erased via `Rc<dyn Any>`)
3. Register in `cjc-runtime/src/lib_registry.rs`
4. Wire dispatch into `cjc-eval` and `cjc-mir-exec` behind `libraries_enabled`
5. Add effect entries to `cjc-types/src/effect_registry.rs`
6. Provide `docs` module for LSP integration

### Type erasure contract

```rust
// In cjc-runtime:
Value::VizorPlot(Rc<dyn Any>)

// In cjc-vizor dispatch:
let spec = inner.downcast_ref::<PlotSpec>()
    .ok_or("expected VizorPlot")?;
```

The core crates never depend on library crates; the dependency flows:
```
cjc-eval ──depends──> cjc-vizor
cjc-mir-exec ──depends──> cjc-vizor
cjc-vizor ──depends──> cjc-runtime (for Value)
```

## Consequences

- **Positive:** Clean separation, no namespace pollution, easy to add libraries
- **Positive:** NoGC verifier works without knowing library internals
- **Negative:** Each library needs dispatch wiring in both eval and MIR-exec
- **Negative:** Type-erased values lose static type information at boundaries
- **Accepted:** The wiring cost is manageable for a curated library set
