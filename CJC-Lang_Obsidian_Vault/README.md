# CJC-Lang Obsidian Vault

A structured knowledge base for **CJC-Lang** (Computational Jacobian Core) — a deterministic numerical programming language implemented in Rust across 21 workspace crates.

This vault is organized for Obsidian. Every note uses Obsidian wikilink syntax. Start from [[CJC-Lang Overview]] or the map of content in [[CJC-Lang Knowledge Map]].

## Structure

| Folder | Purpose |
|---|---|
| `00_Indexes/` | Hub notes and maps-of-content (MOCs) |
| `01_Foundation/` | Identity, philosophy, current state |
| `02_Language/` | Surface syntax, types, semantics |
| `03_Compiler/` | Lexer → Parser → AST → HIR → MIR → Exec |
| `04_Runtime/` | Execution backends, memory model, dispatch, builtins |
| `05_Determinism_and_Numerics/` | Reproducibility contract and numerical stability |
| `06_Tensors_ML_AD/` | Tensors, autodiff, ML primitives |
| `07_Data_and_CLI/` | DataFrames, Vizor, CLI surfaces |
| `08_Advanced_Computing/` | Quantum, PINN, solvers |
| `09_Showcase/` | Evidence-backed capability notes |
| `10_Roadmap_and_Open_Questions/` | Planned work, gaps, open questions |
| `11_Glossary/` | Term definitions and concept atoms |
| `12_Source_Maps/` | Concept-to-code location maps |

## Status Labels

Every substantive note uses one of these:
- **Implemented** — working, tested, in use
- **Partially implemented** — core works, edges missing
- **Experimental** — exists, not yet load-bearing
- **Planned** — in roadmap docs, not yet built
- **Historical / superseded** — was true once; included for context
- **Needs verification** — inferred from code, not explicitly confirmed

## Grounding

Everything in this vault is grounded in the CJC-Lang repository at `C:\Users\adame\CJC` as of 2026-04-09. Claims that could not be verified are labeled **Needs verification** rather than polished into false certainty.

Enter via [[CJC-Lang Overview]].
