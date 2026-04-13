---
title: CJC-Lang Knowledge Map
tags: [hub, moc, graph]
---

# CJC-Lang Knowledge Map

This is the top-level map of content. It links to the major clusters of the vault and is optimized for Obsidian's graph view.

## Top-Level Clusters

### Foundation
- [[CJC-Lang Overview]]
- [[What Makes CJC-Lang Distinct]]
- [[Current State of CJC-Lang]]
- [[Language Philosophy]]
- [[Version History]]

### Language Surface
- [[Syntax]] — [[Types]]
- [[Expressions and Statements]]
- [[Patterns and Match]]
- [[Closures]]
- [[Format Strings]]
- [[Operators and Precedence]]

### Compiler Pipeline
- [[Compiler Architecture]] — hub
- [[Lexer]] → [[Parser]] → [[AST]] → [[Type Checker]] → [[HIR]] → [[MIR]] → [[MIR Optimizer]] → [[cjc-mir-exec]]
- [[Capture Analysis]], [[Dispatch Layer]]
- [[SSA Form]], [[CFG]], [[Dominator Tree]], [[Loop Analysis]]
- [[Diagnostics]], [[Error Codes]]
- [[NoGC Verifier]], [[Escape Analysis]]
- [[Parity Gates]]
- [[Compiler Concept Graph]]

### Runtime
- [[Runtime Architecture]] — hub
- [[Value Model]] — [[Tensor Runtime]]
- [[Memory Model]] — [[COW Buffers]] — [[Frame Arena]] — [[Binned Allocator]]
- [[Dispatch Layer]] — [[Builtins Catalog]]
- [[cjc-eval]] — [[cjc-mir-exec]]
- [[Runtime Concept Graph]]

### Determinism and Numerics
- [[Determinism Contract]] — hub
- [[Numerical Truth]]
- [[SplitMix64]]
- [[Kahan Summation]]
- [[Binned Accumulator]]
- [[Deterministic Ordering]]
- [[Total-Cmp and NaN Ordering]]
- [[Float Reassociation Policy]]
- [[Determinism Concept Graph]]

### Tensors, ML, AD
- [[Tensor and Scientific Computing]]
- [[Tensor Runtime]]
- [[Linear Algebra]]
- [[Autodiff]] — forward dual + reverse tape
- [[ML Primitives]]
- [[Signal Processing]]
- [[Statistics and Distributions]]
- [[Hypothesis Tests]]

### Data and CLI
- [[Data Systems and CLI]] — hub
- [[DataFrame DSL]]
- [[Vizor]] — grammar of graphics
- [[Regex Engine]]
- [[Binary Serialization]]
- [[CLI Surfaces]]
- [[REPL]]

### Advanced Computing
- [[Advanced Computing in CJC-Lang]] — hub
- [[Quantum Simulation]]
- [[PINN Support]]
- [[ODE Integration]]
- [[Optimization Solvers]]
- [[Sparse Linear Algebra]]

### Showcase
- [[What CJC-Lang Can Already Do]]
- [[Chess RL Demo]]
- [[Deterministic Workflow Examples]]
- [[Why CJC-Lang Matters]]
- [[Demonstrated Scientific Computing Capabilities]]

### Roadmap and Open Questions
- [[Roadmap]]
- [[Open Questions]]
- [[Documentation Gaps]]
- [[Roadmap Dependency Graph]]

### Reference
- [[Glossary]]
- [[Compiler Source Map]]
- [[Runtime Source Map]]
- [[Data Systems Source Map]]
- [[Advanced Computing Source Map]]

## Cross-Cutting Relationships

The most important *horizontal* links — things that cross cluster boundaries:

- **[[Determinism Contract]]** is enforced by [[cjc-repro]] (primitives), observed by [[Runtime Architecture]], and verified at the leaves by [[Parity Gates]].
- **[[Memory Model]]** is specified in [[cjc-runtime]], checked statically by the [[NoGC Verifier]] in [[cjc-mir]], and materialized at runtime in [[cjc-mir-exec]].
- **[[Tensor Runtime]]** is the hot path for [[Linear Algebra]], [[ML Primitives]], [[Quantum Simulation]], [[Autodiff]], and [[DataFrame DSL]] numerics.
- **[[Dispatch Layer]]** is the single source of truth for builtin semantics; both [[cjc-eval]] and [[cjc-mir-exec]] route through it to guarantee parity.

## Graph View Notes

This vault is link-dense by design. Major hubs (this note, [[CJC-Lang Overview]], [[Compiler Architecture]], [[Runtime Architecture]], [[Determinism Contract]]) should appear as high-degree nodes in Obsidian's graph view. Concept-graph notes ([[Compiler Concept Graph]], [[Runtime Concept Graph]], [[Determinism Concept Graph]], [[Scientific Computing Concept Graph]], [[Roadmap Dependency Graph]]) explicitly describe the relationships rather than just listing them.
