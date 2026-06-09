# Pass Implementation Designs — vectorize / specialize / monomorphize

**Date:** 2026-06-09
**Status:** Design only — scaffolds already shipped in [3d333d8](../../crates/cjc-mir/src/optimize.rs#L1610) and [7f2b1b5](../../crates/cjc-cana/src/legality.rs)
**Scope:** §3.2 of `HANDOFF_NEXT_SESSION_v3.md` — the three passes that
sit in `THERMALLY_AGGRESSIVE_PASSES` but had no real implementation
until this design lands

This doc covers all three because they share structural concerns —
each one is a MIR-rewriting pass that needs to preserve AST/MIR parity,
each one classifies under `NoStrictReductions` until proven safe, each
one fits the existing `apply_pass_with_diagnostics` dispatch. Where
they differ (algorithm, complexity, risk) gets a per-pass subsection.

---

## 1. Shared structure across all three

### 1.1 What they have in common

All three are MIR-rewriting passes that take a `&mut MirFunction` and
return a `usize` count of rewrites applied. They plug into the existing
`apply_pass_with_diagnostics` dispatch in `crates/cjc-mir/src/optimize.rs`
— the scaffolds shipped earlier this session already reserve the
dispatch arms and the legality classifications. Real implementations
replace the `*_noop_fn` body with actual transformation logic.

### 1.2 Test pattern

Every pass needs three test families before it ships:

1. **Unit tests in `optimize.rs`** — small MIR shapes, hand-built,
   assert that the rewrite produces the expected output. Follow the
   pattern set by `unroll_collapses_simple_short_loop` etc.

2. **AST/MIR parity tests** — the pass MUST preserve byte-identity
   between `cjcl run` (AST eval, no optimizer) and `cjcl run --mir-opt`
   (MIR exec with optimizer). The `tests/fixtures/runner.rs` gate
   exercises this automatically for any program in the fixture corpus,
   so as long as the pass is in `DEFAULT_PASS_SEQUENCE`, parity is
   tested transitively.

3. **Cost-model coefficients** — once real implementations land, add
   the pass to `bench/cana_train_cost_model/TARGET_PASSES`, add a
   handful of pass-favouring programs to the corpus
   (`bench/cana_train_cost_model/programs.rs`), run training, paste
   the new coefficients into `crates/cjc-cana/src/linear_cost_model.rs`.

### 1.3 Promotion checklist

For each pass, the promote-from-scaffold-to-real-impl change does:

- [ ] Replace `*_noop_fn` body with real logic
- [ ] Add ≥6 unit tests covering success, refusal, edge cases
- [ ] Add the pass name to `CANONICAL_PASSES` in `cjc-cana::pass_ranker`
- [ ] Add to `DEFAULT_PASS_SEQUENCE` in `cjc-mir::optimize` (post-LICM, pre-loop_unroll is the convention)
- [ ] Re-justify the legality classification OR keep `NoStrictReductions` with comment
- [ ] Add 8-10 pass-favouring corpus programs + regenerate trained coefficients
- [ ] Update CLAUDE.md if user-visible semantics change

### 1.4 Where they all sit in the pass order

Current order:
```
constant_fold → strength_reduce → dce → cse → licm → loop_unroll → cf_round_2
```

When the three new passes land, the recommended order is:

```
constant_fold → strength_reduce → dce → cse → licm → loop_unroll
  → vectorize       (after loop_unroll because unrolled bodies are
                     vectorization candidates)
  → specialize      (after vectorize because specialized variants
                     can absorb the vectorized form)
  → monomorphize    (last because it can produce many specialized
                     copies; preceding passes simplify what gets
                     duplicated)
  → cf_round_2      (still last — folds constants exposed by all
                     prior rewrites)
```

This ordering minimizes downstream-pass code bloat.

---

## 2. `monomorphize` — easiest of the three (~500-1000 LOC, 3-5 days)

### 2.1 Why first

Of the three, monomorphize has the cleanest MIR transformation:
walk the call graph, find generic call sites, clone the callee with
type substitutions. No SIMD codegen, no alignment analysis, no
arithmetic rewriting. The hardest case is recursive generic
instantiation, which has a textbook fix (worklist + memoization).

The `MirFunction.type_params: Vec<(String, Vec<String>)>` field
([crates/cjc-mir/src/lib.rs:134](../../crates/cjc-mir/src/lib.rs#L134))
already exists, so the input shape is well-defined. The output is
just more `MirFunction` entries in `MirProgram.functions` plus
rewrites at call sites.

### 2.2 Algorithm

```rust
fn monomorphize_program(program: &mut MirProgram) -> usize {
    let mut worklist: VecDeque<(MirFnId, Vec<String>)> = VecDeque::new();
    let mut instantiated: BTreeMap<(MirFnId, Vec<String>), MirFnId> = BTreeMap::new();
    let mut rewrites = 0;

    // Seed: every concrete-type call site is a worklist entry.
    for func in &program.functions {
        for_each_call(&func.body, |callee_id, type_args| {
            if !type_args.is_empty() {
                worklist.push_back((callee_id, type_args));
            }
        });
    }

    while let Some((callee_id, type_args)) = worklist.pop_front() {
        let key = (callee_id, type_args.clone());
        if instantiated.contains_key(&key) {
            continue;
        }

        // Clone the callee with type substitutions.
        let template = &program.functions[callee_id.0 as usize];
        let mut instance = template.clone();
        substitute_type_params(&mut instance, &template.type_params, &type_args);
        let new_id = MirFnId(program.functions.len() as u32);
        instance.id = new_id;
        instance.name = mangle_name(&template.name, &type_args);
        instance.type_params = vec![]; // now concrete

        // Walk the instance's calls — any generic call inside it
        // creates new worklist entries.
        for_each_call(&instance.body, |inner_id, inner_args| {
            if !inner_args.is_empty() {
                worklist.push_back((inner_id, inner_args));
            }
        });

        instantiated.insert(key, new_id);
        program.functions.push(instance);
        rewrites += 1;
    }

    // Second pass: rewrite all call sites to point at the
    // instantiated versions.
    for func in &mut program.functions {
        rewrite_calls(&mut func.body, &instantiated);
    }

    rewrites
}
```

### 2.3 Risks

| Risk | Mitigation |
|---|---|
| Recursive generic explodes worklist | Worklist memoization via `instantiated` set — same (callee, type_args) seen twice → second occurrence is a no-op. |
| Generic called with closure-captured type | The capture's concrete type is the same in every instantiation; substitute it directly. |
| Generic-over-generic (function takes a generic function as arg) | Punt — refuse to monomorphize and leave as generic call (already a "complex" pattern, no current users). |
| Code bloat (N callsites × M generics → N×M instances) | Limit instance count per template (e.g. cap at 8); above the cap, fall back to dynamic dispatch. |
| AST/MIR parity break | The transformation is observable only via execution time — the OBSERVABLE BEHAVIOR is identical. Parity gate proves this. |

### 2.4 Test pattern

- `monomorphize_collapses_one_call_site` — generic fn called once with `i64`, produce one instance
- `monomorphize_collapses_two_distinct_call_sites` — same fn called with `i64` and `f64`, produce two instances
- `monomorphize_deduplicates_repeated_call_site` — same fn called 100× with `i64`, produce ONE instance
- `monomorphize_handles_recursive_generic` — generic fn that calls itself with new type args, terminates correctly
- `monomorphize_refuses_unbounded_instance_count` — generic with N=100+ distinct call sites, observes the cap
- `monomorphize_preserves_ast_mir_parity` — fixture-based, real CJC-Lang program

---

## 3. `specialize` — harder than monomorphize (~700-1500 LOC, 4-6 days)

### 3.1 What it does

Where `monomorphize` instantiates generic functions with concrete
TYPES, `specialize` does the same with concrete VALUES. A function
called with a specific constant argument can be specialized to that
constant — equivalent to combining inlining + constant propagation +
DCE in one pass.

Example:
```cjcl
fn pow_inner(base: f64, exp: i64) -> f64 {
    let mut result: f64 = 1.0;
    let mut i: i64 = 0;
    while i < exp {
        result = result * base;
        i = i + 1;
    }
    return result;
}
// Call site:
let cube = pow_inner(x, 3);
```

Specialize produces:
```cjcl
fn pow_inner_specialized_exp3(base: f64) -> f64 {
    let mut result: f64 = 1.0;
    result = result * base;
    result = result * base;
    result = result * base;
    return result;
}
let cube = pow_inner_specialized_exp3(x);
```

`loop_unroll` (already shipped) actually does the inner unrolling
work in this example. `specialize` does the surrounding work: clone
the function, replace one parameter with a constant, let downstream
passes (CF, loop_unroll, DCE) collapse what's now collapsible.

### 3.2 Where it differs from `monomorphize`

- The substitution is at the value layer, not the type layer.
- The "concrete value" is a `MirExpr` literal, not a `String` type name.
- The instance count can blow up faster (one specialized copy per
  unique constant tuple seen at any call site).
- The benefit depends on whether downstream passes can collapse the
  specialized form — without CF + loop_unroll firing, specialize is
  pure code bloat.

### 3.3 Heuristic guard

Specialize should only fire when:

1. The argument is a literal (`MirExprKind::IntLit`, `FloatLit`,
   `BoolLit`, or `StringLit`) at the call site.
2. The argument is referenced in a loop header (the only place where
   specialization unlocks loop_unroll).
3. The function body is short (≤ 50 statements — code-bloat cap).
4. The function is not recursive (recursion + specialization = infinite
   work).

### 3.4 Risks

- **Code bloat dominates benefit on small workloads.** Trained
  coefficients will need to capture this — likely a negative
  `w_expr_count` similar to what loop_unroll learned.
- **Float-literal specialization changes rounding** in some corner
  cases — refuse to specialize float-valued arguments until per-call
  analysis can prove safety.
- **Aliasing with monomorphize** — both produce specialized variants.
  Order matters: monomorphize first (types), then specialize (values
  on already-monomorphized instances).

### 3.5 Estimated effort

7-15 days. The algorithm itself is straightforward; the heuristic guard
and the cost-model integration are where the time goes.

---

## 4. `vectorize` — hardest by far (~2000+ LOC, 2-4 weeks)

### 4.1 What it needs

SIMD vectorization requires:

1. **Loop pattern recognition** — find loops whose body operates on
   contiguous memory in lockstep (the classical inner loop of a
   matrix-vector product, a tensor element-wise op, a Kahan
   accumulator over a slice).
2. **Alignment analysis** — SIMD loads/stores require N-byte alignment
   on most architectures; the pass needs to either prove alignment
   or insert peel/scalar-tail epilogues.
3. **Lane-count selection** — AVX-512 has 8 doubles per lane, AVX2
   has 4, NEON has 2. The chosen lane count affects the IR.
4. **SIMD-aware Value variants** — the executor needs to represent
   "8 doubles in one lane" as a Value. Either add `Value::Simd(...)`
   or rely on Tensor's existing lane-aware ops.
5. **Scalar fallback** — programs running on architectures without
   the chosen SIMD instruction set need a non-SIMD code path.
6. **Determinism preservation** — float SIMD reorders operations
   within a lane. For strict reductions this is unsafe; the
   `NoStrictReductions` classification (shipped as scaffold) already
   blocks vectorize on functions with strict reductions. Without
   strict reductions, the determinism contract is preserved by
   running the same SIMD instructions in the same order each time.

### 4.2 Why this is multi-week

Item 4 alone (SIMD Value representation) touches the Value enum's
layout — and CLAUDE.md flags Value-enum layout changes as a **HARD
RULE**. Adding `Value::Simd(...)` requires updating every dispatch
arm that pattern-matches on Value, every serialization path, every
`type_name()` call. That's ~50+ touch points across `cjc-runtime`,
`cjc-eval`, `cjc-mir-exec`, `cjc-snap`, `cjc-dispatch`. Each touch
point needs its own tests.

Items 1-3 (pattern recognition, alignment, lane selection) are
standard but substantial — easily 1000 LOC of analysis + tests.

Items 5-6 (fallback, determinism) need both architecture-specific
codepaths and a test matrix across them.

### 4.3 Recommended approach: thin slice first

Rather than landing all of vectorize at once, ship a thin vertical
slice:

**Slice 1 (1 week):** SIMD-aware tensor element-wise ops only.
Detect `for i in 0..tensor.len() { c[i] = a[i] + b[i] }` shape,
rewrite to a single `tensor_add_simd(a, b, c)` builtin. NO Value
enum change — the builtin is implemented in cjc-runtime with
internal SIMD. Test on a fixture that exercises this exact pattern.

**Slice 2 (1 week):** Tensor reduction (sum/mean) with non-strict
accumulation. Add `tensor_sum_simd` builtin. Same approach — no
Value layout change, all SIMD lives in cjc-runtime.

**Slice 3 (2-3 weeks):** General loop vectorization. THIS is where
the Value enum change (or alternative representation) becomes
necessary, and where the full multi-week budget applies.

This staged approach lets PRs 1-2 land independently, each with its
own AST/MIR parity proof. PR 3 is the architecturally significant
one; landing it requires the full design pass.

### 4.4 Why scaffolding it now is still useful

The scaffold reserves the name in dispatch + legality + cost-model
infrastructure. When slice 1 lands, the change is:
- Replace `vectorize_noop_fn` body with the element-wise detector
- Add to CANONICAL_PASSES + DEFAULT_PASS_SEQUENCE
- Add corpus programs + regenerate trained coefficients

Without the scaffold, slice 1 would have to land all of the
infrastructure plumbing in the same PR, which dilutes the
implementation review.

---

## 5. Cross-cutting concerns

### 5.1 The `THERMALLY_AGGRESSIVE_PASSES` list

All three passes are already in `cjc-cana::thermal_cost_model::THERMALLY_AGGRESSIVE_PASSES`:

```rust
pub const THERMALLY_AGGRESSIVE_PASSES: &[&str] = &[
    "loop_unroll",
    "vectorize",
    "specialize",
    "monomorphize",
];
```

`loop_unroll` is the only one currently shipped, so it's the only
one the thermal-aware cost model can actually penalize. The §3.2
sweep (this design + the scaffold commits) ensures that when real
implementations land, the thermal-aware machinery applies the
penalty automatically — no further wiring needed.

### 5.2 Interaction with Option B

Once Option B (real MIR-exec instrumentation, see
[OPTION_B_DESIGN.md](OPTION_B_DESIGN.md)) lands, the cost model can
finally observe per-function thermal pressure. That's when the
§6.1 thermal penalty becomes load-bearing for these aggressive
passes — without real per-function thermal data, the penalty has
the same value on every function and effectively just rescales the
threshold.

**Order recommendation:** Option B PR 1-5 → at least one of
{vectorize slice 1, specialize, monomorphize} → measure thermal
divergence on PINN. That sequence is the only way to demonstrate
the full thermal-aware chain working end-to-end on a real workload.

### 5.3 Documentation hygiene

When each pass ships:
- Update `crates/cjc-mir/src/optimize.rs` file-level doc to list it
  in the pass-order table.
- Add an entry to the `apply_pass` doc-comment's "Recognised names"
  list (this is already there for the scaffolds).
- Update `crates/cjc-cana/src/legality.rs::PassSafetyTier` doc to
  list it under the correct tier.

---

## 6. Estimated calendar

| Pass | First-slice scope | Effort (1 dev) | Calendar (with review) |
|---|---|---|---|
| `monomorphize` | Full pass | 3-5 days | 1-2 weeks |
| `specialize` | Literal-arg + short-body only | 4-6 days | 2 weeks |
| `vectorize` slice 1 (tensor elementwise) | One detector + one builtin | 5-7 days | 1-2 weeks |
| `vectorize` slice 2 (tensor reduction) | One more detector + one more builtin | 4-5 days | 1-2 weeks |
| `vectorize` slice 3 (general loops) | Value-enum change + general detector | 10-15 days | 4-6 weeks |

**Recommended pick order:** monomorphize first (clearest algorithm,
fewest surprises), then specialize (builds on monomorphize's
infrastructure), then vectorize slices 1 → 2 → 3 (each independently
mergeable).

---

## 7. What this design intentionally defers

- **Inlining as a distinct pass.** Specialize subsumes most of the
  inlining benefits when the call is at a fixed-trip-count site.
  A general inliner is a separate design pass.
- **Auto-parallelization.** Vectorize is SIMD-within-a-core only.
  Multi-core parallelization (rayon-style work-stealing) is out of
  scope and would interact with the determinism contract in
  non-trivial ways.
- **GPU offload.** Same reason as auto-parallelization.

---

*Generated as part of the §3.2 / §3.3 / §4.1 / §3.1 sweep. Scaffolds
already shipped — this doc sets up the real-implementation work for
future sessions.*
