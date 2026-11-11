# Julia's language mechanisms: which ones CJC-Lang should adopt, and how

These constraints come from `CLAUDE.md`:
- language primitives stay minimal, and higher-level functionality lives in libraries (Bastion, Vizor);
- both executors must agree;
- the result must be bit-identical every time;
- NoGC paths must not allocate behind the user's back.

Each section answers three questions:
- **Buys:** what the mechanism gives Julia for accuracy or SciML.
- **Need:** whether CJC needs it.
- **Minimal deterministic version:** the smallest way to get the benefit, and where it should live.

Current CJC facts are in [CJC_INVENTORY.md §1](CJC_INVENTORY.md#1-language-and-numeric-core).

---

## 1. Multiple dispatch and parametric types → kernels that work for any dtype

**Buys.** This is Julia's single biggest accuracy lever. You can rerun an
algorithm in a more precise type (BigFloat, Double64), in an enclosing type
(Interval) or with derivatives (Dual) without writing it again. You can also
compare a Float64 result against a BigFloat run of *the same code*.

**Where CJC is today.**
- `Value` is tagged dynamically, so a generic `fn square<T>(x: T)` already runs on Int, Float and (very likely) Tensor.
- But `Tensor` is `Buffer<f64>` only. The unused `DType`/`TypedStorage` scaffolding shows this was intended, and never finished.
- `cjc-dispatch` implements real multiple dispatch with specificity ranking and a coherence check, but **no executor calls it**.

**Need.** Yes, but at the **Rust kernel level**, not as user-level multiple dispatch.

**Minimal deterministic version.**
1. **Rust side.** Add a `trait DetScalar: Copy` with `add`, `sub`, `mul`, `div`, `sqrt`, `from_f64` and `to_f64`, plus elementary functions that go through dmath.
   - Implement it for `f64`, `ComplexF64`, a new `DD` (double-double built from TwoSum and Dekker TwoProd, **no FMA**) and later `Jet<N>` (§B).
   - Make the tensor kernels generic over the trait: elementwise ops, matmul, reductions, and the generic paths of `solve`/`eigh`/`svd`.
2. **Determinism.** Every instance has a fixed operation order. Reductions become generic over an accumulator trait: Kahan for f64, plain for DD (which is already compensated).
3. **Executors.** No executor change except new `Value` variants (`Value::Tensor` gains a dtype tag, or new variants are added). Both executors call the same runtime kernel, so parity follows from that.
4. **What stays out.** User-visible overloading on argument types stays out until there is a demonstrated need. Wiring `cjc-dispatch` into the executors would change how every call resolves, which is a large blast radius for little accuracy gain.
5. **Placement.** The trait and the DD type go in `cjc-runtime`/`cjc-repro`. They are **primitives**: numerics libraries can't be built on top without them. Interval and uncertainty types belong in Bastion, built on the trait.

**Cost / risk.** L effort. Changes the `Tensor` memory layout. Needs an ADR (ROADMAP ADR-E).

## 2. Broadcast fusion (dot syntax)

**Buys.** `y .= a .* x .+ b` runs as one loop: there are no temporaries, and
the result is rounded once per element. For accuracy, fusion only matters if
it changes rounding, and Julia's broadcast does **not** contract to FMA unless
the user writes `muladd`/`@fastmath`. So the gain is speed and memory, not accuracy.

**Need.** Not for accuracy. For performance, ADR-0027 already chose explicit
fused builtins over automatic fusion.

**Minimal deterministic version.** Leave the syntax alone. If automatic fusion
is ever added, it has to be a MIR pass that keeps the exact per-element
operation order (no reassociation, no FMA). It can then be gated by a
bit-identity test against the unfused program. **Won't-do for this roadmap.**

**Related correctness gap (P0).** Broadcasting *semantics* in the AD backward
pass are broken (probe P2). That needs fixing whatever happens to the syntax.

## 3. Macros, `@generated`, symbolic → numeric code generation (ModelingToolkit)

**Buys.**
- ModelingToolkit simplifies a model symbolically before any numerics run (index reduction, tearing, symbolic Jacobians). This prevents whole classes of numerical failure, for example a DAE of index > 1 handed to an ODE solver.
- `@generated` gives code specialised by size.

**Need.**
- The *symbolic derivative* half: yes, eventually. `symbolic_derivative` is a reserved stub, and a small expression-tree differentiator with simplification is library-sized.
- A macro system: no. It would add a compile-time evaluation stage to the pipeline (Lexer → Parser → **expand** → AST …). CLAUDE.md forbids breaking the pipeline, and determinism would then also depend on macro-expansion order.

**Minimal deterministic version.**
- In Bastion: a `Sym` expression type (an enum tree with canonical ordering via BTreeMap), `diff(expr, var)`, `simplify`, and `lambdify(expr) -> closure`. These are ordinary values and closures, with no new syntax.
- In the core: nothing. Decorators already exist. If "symbolic → numeric" needs compiling, it compiles to a closure at run time, which both executors can already call.

**Placement.** Bastion. Won't-do: macros in the core.

## 4. StaticArrays-style small fixed-size types

**Buys.**
- Speed through stack allocation and unrolling, both of which come from JIT specialisation on size.
- No accuracy benefit, except that small exact-size kernels (2×2, 3×3) let you use closed-form formulas.

**Need.** No. CJC runs in an interpreter and a MIR register machine, so unboxed fixed-size arrays would not unroll anyway. `[T;N]` already exists as a type annotation.

**Minimal version.** If profiling ever shows small-matrix overhead, add
fixed-size kernels for 2×2, 3×3 and 4×4 `solve`/`det`/`inv` inside the
existing tensor builtins, chosen by shape at run time. No new type is needed.

**Won't-do** as a language feature.

## 5. Units of measure (Unitful.jl)

**Buys.** Physical correctness: adding metres to seconds becomes an error.
It does not change floating-point accuracy.

**Need.** Useful for physics-informed code, but not a numerical-accuracy gap.

**Minimal deterministic version.**
- In Bastion: a `Quantity { value: f64, dims: [i8; 7] }` record type (the SI base dimensions, exponents fixed).
- Arithmetic checks the dimensions at run time, so there is no type-level machinery. Both executors run the same `.cjcl`, so parity is automatic.
- A static check could later go into `cjc-types` once records with const generics are enforced.

**Placement.** Bastion. P2.

## 6. Composable custom derivative rules (ChainRules `rrule` / `frule`)

**Buys.**
- Numerically stable adjoints where differentiating the forward code would be unstable: `logsumexp`, `softplus`, `norm` at 0, implicit-function adjoints for `solve`/`eigh`, ODE adjoints.
- Rules can also be shared across AD systems.

**Need.** **Yes.** This is the only mechanism in this list that is both
accuracy-relevant and blocked by a *closed* design: `GradOp` is a closed enum.

**Minimal deterministic version.**
1. Add one new op, `GradOp::Custom { inputs, rule_id }`.
2. A rule is a pair of `.cjcl` closures registered on the graph: `g.custom(fwd, bwd, [inputs])`.
   - `fwd` computes the value.
   - `bwd(upstream, inputs, output)` returns one gradient per input.
3. **Determinism.**
   - The backward pass calls closures in the reverse order of the tape, which is the same order on every run.
   - Rule IDs are dense integers assigned in registration order (never hashes).
4. **Executors.** This needs the same "builtin calls back into the executor" capability as ODE solvers (ROADMAP ADR-C), so the two share one design.
5. **NoGC.** A custom rule allocates its closure environment, so mark `g.custom` as not NoGC-safe in the verifier.

**Placement.** The hook is core (`cjc-ad` plus the executors). The rules
themselves go in libraries. P1.

---

## Summary

| Mechanism | Accuracy value | Needed? | Where | Priority |
|---|---|---|---|---|
| Dtype-generic kernels (trait: f64 / DD / Complex / Jet) | high | yes | core runtime | P1 (ADR-E) |
| User-level multiple dispatch | low (for accuracy) | not now | — | Won't-do (for now) |
| Broadcast-fusion syntax | none | no | — | Won't-do |
| Broadcast-correct AD | high (correctness) | yes | `cjc-ad` | **P0** |
| Macros / `@generated` | indirect | no | — | Won't-do |
| Symbolic differentiation | medium | yes | Bastion | P2 |
| StaticArrays | none | no | — | Won't-do |
| Units | correctness, not accuracy | nice to have | Bastion | P2 |
| Custom AD rules | high | yes | core hook, rules in libraries | P1 (with ADR-C) |
</content>
</invoke>
