---
title: Language Philosophy
tags: [foundation, philosophy]
status: Summarized from docs/spec/CJC_PERFORMANCE_MANIFESTO.md and README
---

# Language Philosophy

CJC-Lang has an explicit design ordering, paraphrased from its performance manifesto:

> **determinism > memory efficiency > latency > speed**

Every design decision is taken through that lens. If an optimization would save 10% runtime but introduce nondeterminism (hash randomization, FMA, thread-dependent reductions), it is rejected.

## Guiding principles

1. **Determinism is not a feature, it's a constraint.** See [[Determinism Contract]]. The compiler and runtime refuse to compile or run code that can produce different outputs on different runs of the same program with the same seed.

2. **Minimal primitives, maximal libraries.** The language core stays small; higher-level functionality belongs in libraries like [[Bastion]] (statistics) and [[Vizor]] (visualization). See `docs/bastion/CLASSIFICATION.md` which shows that ~47% of statistical functionality can be written in pure CJC once the runtime provides the right primitives.

3. **Two backends must agree.** [[cjc-eval]] and [[cjc-mir-exec]] must produce bit-identical output. This is enforced by [[Parity Gates]] and is the top-level correctness invariant of the compiler. See [[What Makes CJC-Lang Distinct]].

4. **Zero external runtime dependencies.** If a dependency could change the numerical result (libm, BLAS vendor kernels, hash randomization), it is replaced with an internal implementation. Cost: performance. Benefit: honesty about reproducibility.

5. **Numerical correctness is non-negotiable.** Reductions go through [[Kahan Summation]] or [[Binned Accumulator]]. SIMD kernels do not use FMA. Floating-point sorting uses `total_cmp`. See [[Numerical Truth]].

6. **Language core stays minimal; user surface is large.** The syntax is deliberately modest (functions, structs, enums, closures, pattern match, `for`/`while`/`if`) and the *library* surface is where the scientific computing power lives. See [[Syntax]] vs [[Builtins Catalog]].

7. **Evidence over marketing.** Design docs in `docs/` consistently distinguish proven/observed/inferred claims. The documentation culture labels things `Implemented`, `Partial`, `Planned`, `Needs verification`. This vault inherits that discipline.

## The "byte-first" strategy

A recurring theme in `docs/architecture/byte_first_type_inventory.md` and `byte_first_vm_strategy.md`: types, values, and runtime representations are designed with **canonical byte form** in mind. This supports:
- Reproducible binary serialization ([[cjc-snap]])
- Deterministic cross-platform hashing
- Parity between interpreter and register VM (both decode from the same byte-level representation)

NaN is canonicalized to `0x7FF8_0000_0000_0000` when it needs to appear in hashed/serialized form.

## The "15-primitive" model (Bastion)

The [[Bastion]] statistical library adopts a principle codified in `docs/bastion/BASTION_PRIMITIVE_ABI.md`: admit a new runtime primitive only if it unlocks 5–20 higher-level functions. Example: adding `nth_element` unlocks the entire family of robust statistics (median, MAD, IQR, quartiles) that would otherwise require O(n log n) sorts. This is how CJC-Lang keeps the core small without crippling library authors.

## Related

- [[What Makes CJC-Lang Distinct]]
- [[Determinism Contract]]
- [[Numerical Truth]]
- [[Parity Gates]]
- [[Bastion]]
