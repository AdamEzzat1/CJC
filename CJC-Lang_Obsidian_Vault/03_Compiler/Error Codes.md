---
title: Error Codes
tags: [compiler, diagnostics]
status: Implemented (registry) / Documented in docs/spec/error_codes.md
---

# Error Codes

CJC-Lang has a registered error-code system, documented in `docs/spec/error_codes.md`. Codes are partitioned by phase:

| Range | Category |
|---|---|
| E0001–E0099 | Parse errors |
| E0100–E0149 | Type system |
| E0150–E0199 | Binding / mutability |
| E0200–E0249 | Numeric types |
| E0300–E0349 | Traits / generics |
| E0400–E0449 | Constants |
| E0500–E0549 | Tensor shape inference |
| ... up to E8xxx | Higher-level / runtime |

## Usage

```rust
DiagnosticBuilder::new()
    .error(ErrorCode::E0123)
    .message("mismatched types")
    .primary_span(span)
    .emit(&mut bag);
```

Each code has a stable meaning and is referenced from `cjcl explain <code>` — see [[CLI Surfaces]]. The `explain` command is one of the CLI's introspection features.

## Related

- [[Diagnostics]]
- [[CLI Surfaces]]
- [[Type Checker]]
