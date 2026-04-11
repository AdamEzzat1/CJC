---
title: REPL
tags: [cli, tooling]
status: Implemented
---

# REPL

**Source**: `crates/cjc-cli/src/line_editor.rs`, plus the standard `run` entry point reused for single-line evaluation.

## Summary

An interactive read-eval-print loop for CJC-Lang. Launch with:

```
cjcl repl
```

Each line is lexed, parsed, type-checked, and evaluated by [[cjc-eval]]. State persists across lines in a single top-level scope.

## Features observed in source

- Custom line editor (`line_editor.rs`) for history, basic editing.
- Tokenized output for pretty-printing (`highlight.rs`).
- Output formatting via `output.rs`, `table.rs`, `formats.rs`.

## Related

- [[CLI Surfaces]]
- [[cjc-eval]]
