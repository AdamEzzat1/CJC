---
title: Data Systems Source Map
tags: [source-map, data]
status: Grounded in crate layout
---

# Data Systems Source Map

Where the data, CLI, visualization, and serialization layers live.

## DataFrame DSL (`cjc-data`)

| Concept | File | Hub note |
|---|---|---|
| DataFrame core | `crates/cjc-data/src/frame.rs` (or `lib.rs`) | [[DataFrame DSL]] |
| `filter`, `select`, `mutate` | `crates/cjc-data/src/verbs.rs` | [[DataFrame DSL]] |
| `group_by`, aggregations | `crates/cjc-data/src/group.rs` | [[DataFrame DSL]] |
| `join`, `pivot`, `window_*` | `crates/cjc-data/src/join.rs`, `pivot.rs`, `window.rs` | [[DataFrame DSL]] |
| Categorical / factor types | `crates/cjc-data/src/categorical.rs` | [[DataFrame DSL]] |
| CSV reader / writer | `crates/cjc-data/src/csv.rs` | [[DataFrame DSL]] |

The exact file names may vary — the set above is the target layout implied by `docs/spec/phase_*` tidy progress notes.

## Visualization (`cjc-vizor`)

| Concept | File | Hub note |
|---|---|---|
| Grammar-of-graphics pipeline | `crates/cjc-vizor/src/lib.rs` | [[Vizor]] |
| SVG renderer | `crates/cjc-vizor/src/svg.rs` | [[Vizor]] |
| BMP renderer | `crates/cjc-vizor/src/bmp.rs` | [[Vizor]] |
| Chart types (scatter, line, bar, histogram, annotated, …) | `crates/cjc-vizor/src/charts/*` | [[Vizor]] |

Example programs that drive Vizor:

- `examples/vizor_scatter.cjcl`
- `examples/vizor_line.cjcl`
- `examples/vizor_bar.cjcl`
- `examples/vizor_histogram.cjcl`
- `examples/vizor_annotated.cjcl`

Gallery fixtures live in `gallery/` and act as regression hashes (two runs must produce byte-identical SVG/BMP).

## Binary serialization (`cjc-snap`)

- `crates/cjc-snap/src/lib.rs` — serializer + SHA-256 + NaN canonicalization.
- Invoked from user code via `save_model` / `load_model` style builtins.
- See [[Binary Serialization]].

## Regex (`cjc-regex`)

- `crates/cjc-regex/src/lib.rs` — Thompson NFA, no backtracking.
- See [[Regex Engine]].

## Module system (`cjc-module`)

- `crates/cjc-module/src/lib.rs` — ~1,183 LOC, ModuleId / ModuleGraph / cycle detection / merge_programs.
- **Not yet wired** as the default CLI execution path despite the substantial infrastructure. See [[Module System]] and [[Documentation Gaps]].

## Language server (`cjc-analyzer`)

- `crates/cjc-analyzer/src/lib.rs` — experimental LSP surface.
- See [[Language Server]].

## CLI (`cjc-cli`)

| Concept | File | Notes |
|---|---|---|
| Entry point | `crates/cjc-cli/src/main.rs` | binary name `cjcl` |
| Command dispatch | `crates/cjc-cli/src/lib.rs` | routes subcommands |
| Subcommands | `crates/cjc-cli/src/commands/*` | ~30 subcommands |

See [[CLI Surfaces]] and [[REPL]].

## Example programs index

| Category | Files |
|---|---|
| ML | `examples/01_mlp_xor.cjcl` through `examples/09_quantum_simulation.cjcl` |
| NLP | `examples/nlp_tokenize.cjcl`, `examples/nlp_vocab_count.cjcl` |
| Data | `examples/etl_csv_parse.cjcl`, `examples/transformer_forward.cjcl` |
| Vizor | `examples/vizor_*.cjcl` |
| Chess RL browser demo | `examples/chess_rl_platform.html` |

See [[Demonstrated Scientific Computing Capabilities]] and [[Deterministic Workflow Examples]].

## Related

- [[Data Systems and CLI]]
- [[Runtime Source Map]]
