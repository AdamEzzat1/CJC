---
title: Module System
tags: [language, modules, implemented]
status: Implemented (wired via --multi-file, verified 2026-04-09)
---

# Module System

**Crate**: `cjc-module` — `crates/cjc-module/src/lib.rs` (1,183 LOC, 17+ tests).

The multi-file module system is **fully implemented and wired into the CLI**. Programs can be split across multiple files using `import` declarations; the CLI's `--multi-file` flag runs them through the full module pipeline.

## Quick start

```cjcl
// main.cjcl
import math.linalg

fn main() -> i64 {
    print(add(1, 2));
    0
}
```

```cjcl
// math/linalg.cjcl
pub fn add(a: i64, b: i64) -> i64 { a + b }
```

Run with:

```bash
cjcl run --multi-file main.cjcl
```

Modules resolve relative to the entry file's directory. Cyclic imports are detected and reported as diagnostics.

## Name resolution and visibility

[[ADR-0047 Private Functions Are Module-Private]]. Before either executor
runs, `resolve_modules` rewrites every function reference to its qualified
name, so AST-eval and MIR-exec agree on what each name means:

- A module's own functions come first. `import m` provides `m`'s `pub`
  functions; `import m.f [as g]` provides one function. The first import
  wins, and only direct imports count.
- Local bindings shadow function names.
- Functions without `pub` are private: callable inside their module, not in
  scope elsewhere. Calling one from another module fails as "undefined",
  with a note naming the owning module.
- A name no module provides is left as is, so builtins are never shadowed
  by another module's function.
- Qualified call syntax (`m::f(x)`) does not parse; call imported
  functions by bare name.

## Pipeline

```
entry.cjcl ─► build_module_graph() ─► ModuleGraph
                                          │
                     check_visibility() ◄─┤  (`import m.Symbol` must be pub)
                                          │
                     resolve_modules() ◄──┤  (qualify every function name)
                                          │
                     merge_resolved()  ◄──┘  (MIR-exec; AST-eval runs the resolved ASTs)
                            │
                     escape::annotate_program()
                            │
                     run_program_with_modules()
                            │
                       (execution via cjc-mir-exec)
```

## What's in the crate

- `ModuleId(pub String)` — canonical identifier (`math::linalg` from `math/linalg.cjcl`)
- `ModuleInfo` — per-module metadata
- `ImportInfo` — import declarations with source spans
- `ModuleGraph` (line 200) — dependency graph with deterministic iteration
- `build_module_graph(entry_path)` (line 426) — DFS cycle detection, returns `Result<ModuleGraph>`
- `merge_programs(graph)` (line 576) — merges per-module `MirProgram`s into one
- `check_visibility(graph)` / `enforce_visibility(graph)` — `import m.Symbol` must name a `pub` symbol
- `resolve_modules(graph)` (`resolve.rs`) — module-scoped name resolution shared by both executors; `explain_undefined` adds the private-function note
- `build_import_aliases(graph)` (line 844) — resolves short names to full paths
- 17+ inline tests (lines 903–1089)

All internal maps are `BTreeMap` / `BTreeSet` so iteration order is deterministic — see [[Deterministic Ordering]].

## Integration seams (verified 2026-04-09)

- `cjc-parser/src/lib.rs:872` — parses `import` declarations
- `cjc-ast` — has `Item::Import(ImportDecl)` variant
- `cjc-cli/src/lib.rs:680-759` — `--multi-file` flag calls `cjc_module::build_module_graph` then `cjc_mir_exec::run_program_with_modules`
- `cjc-mir-exec/src/lib.rs:4063` — `pub fn run_program_with_modules(entry_path: &Path, seed: u64) -> MirExecResult`
- `cjc-mir-exec/src/lib.rs:4080` — `run_program_with_modules_executor` variant for tests that need the executor back

## Historical note

Earlier versions of `README.md` and `CLAUDE.md` described this crate as "incomplete." That was stale after the CLI wiring landed. The [[Current State of CJC-Lang]] status table now lists this as **Implemented**. Those stale strings in the root-level `README.md` should be removed on the next sweep.

## Known rough edges

- Diagnostics for malformed `import` paths are functional but not polished.
- `pub(crate)` vs `pub` distinction is not yet fine-grained.
- No namespacing scheme beyond dot-separated paths — aligned with [[ADR-0013 Package Manager]] plans.

## Related

- [[Syntax]]
- [[MIR]]
- [[CLI Surfaces]]
- [[ADR-0013 Package Manager]]
- [[Deterministic Ordering]]
