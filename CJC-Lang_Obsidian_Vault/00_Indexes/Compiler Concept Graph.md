---
title: Compiler Concept Graph
tags: [concept-graph, compiler]
status: Graph hub
---

# Compiler Concept Graph

The concepts in the compiler pipeline and how they link to each other. Use this note in Obsidian graph view to see the web of relations.

## Pipeline backbone

[[Lexer]] → [[Parser]] → [[AST]] → [[Type Checker]] → [[HIR]] → [[MIR]] → [[MIR Optimizer]] → ( [[cjc-eval]] ‖ [[cjc-mir-exec]] )

## Cross-cutting

- Both [[cjc-eval]] and [[cjc-mir-exec]] consume [[AST]] (eval directly, MIR-exec via HIR→MIR).
- Both executors route operator dispatch through the [[Dispatch Layer]].
- Both must agree — enforced by [[Parity Gates]].

## Within MIR

- [[MIR]] builds a [[CFG]] of basic blocks.
- [[CFG]] enables [[SSA Form]], [[Dominator Tree]], and [[Loop Analysis]].
- [[MIR Optimizer]] uses these for CF, DCE, CSE, LICM, SCCP.
- [[NoGC Verifier]] is a call-graph fixpoint over the CFG + [[Escape Analysis]] classification.
- [[Escape Analysis]] classifies allocations as Stack / Arena / Rc, connecting the compiler to the [[Memory Model]].

## Within HIR

- [[HIR]] runs [[Capture Analysis]] to resolve which outer variables closures capture. This is how [[Closures]] become first-class in MIR.

## Diagnostics

- [[Diagnostics]] carry [[Error Codes]] (E0001–E8xxx).
- Error codes are emitted from [[Lexer]], [[Parser]], [[Type Checker]], [[HIR]], [[MIR]], and both executors.
- E0500/E0501/E0502 (shape inference) are reserved for the type checker — see [[Roadmap]].

## Registration: the Wiring Pattern

A new language feature typically touches:

- [[AST]] (new node)
- [[Type Checker]] (new rule)
- [[HIR]] (new lowering)
- [[MIR]] (new instruction)
- [[MIR Optimizer]] (awareness in passes)
- [[cjc-eval]] (interpretation)
- [[cjc-mir-exec]] (execution)
- tests/milestone_2_4/parity/ (parity gate)

New builtins use the three-place [[Wiring Pattern]] (`builtins.rs`, `cjc-eval`, `cjc-mir-exec`).

## Related

- [[Compiler Architecture]]
- [[Compiler Source Map]]
- [[CJC-Lang Knowledge Map]]
