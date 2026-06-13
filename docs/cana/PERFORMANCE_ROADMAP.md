# Performance Roadmap — Speed & Memory, Determinism-Safe

**Date:** 2026-06-13 · **Branch:** `claude/stupefied-liskov-83b258`
**Source:** the Runtime-Systems-Engineer + Numerical-Computing-Engineer
passes of the stacked-role optimization arc, **then verified against the
actual code** (THE RULE). Every "high-leverage" lever the code-reading
panel proposed was re-checked; the verification changed the conclusions,
so this roadmap is the panel's analysis AFTER ground-truthing, not the
raw proposals.
**Gate for every item:** `docs/cana/DETERMINISM_CONTRACT.md` (the
10-invariant checklist). Nothing here may touch FP reduction order, FMA,
collection iteration order, or the RNG.

## 0. The honest headline

The clean, contained speed/memory wins the panel imagined mostly
**don't exist as contained changes**. Verification found each one is
either a *pervasive* refactor (hundreds of edit sites), a *low-magnitude*
micro-opt (<2%), or *already done*. The high-value wins are real but each
is a **dedicated-session refactor**, because landing a hundreds-of-sites
change while preserving bit-identical determinism is exactly the kind of
work that must not be rushed — the parity gate is final authority.

This roadmap scopes those refactors precisely (with measured blast
radius) so a future focused session can execute them safely, and records
the claims that did NOT survive verification so they aren't re-proposed.

## 1. Verified blast radius (why these are dedicated-session work)

| lever | panel estimate | VERIFIED reality |
|---|---|---|
| `Tensor.shape/strides` → `Rc<[usize]>` (shrinks every Tensor + Value, elides view-op Vec allocs) | "medium effort" | **172 field-read sites + 78 field-write/init sites in `tensor.rs` alone**, plus `builtins.rs` + `cjc-ad`. ~250-site pervasive refactor. |
| `Value` enum boxing (largest variants `Enum`/`Closure`/`Struct`/`Regex`/`Tensor` set the 80-byte size of every Value) | "medium-high" | Tensor is ~72 B, so boxing the struct/enum variants alone does NOT shrink Value below ~80 B — Tensor must also box. Every match arm of every boxed variant across `cjc-eval`/`cjc-mir-exec`/`cjc-runtime`/`cjc-dispatch` changes. Pervasive. |
| Closure-env clone on the call path | "10–20% win" | **Already optimal** — `cjc-mir-exec/src/lib.rs:1835` MOVES `env` (`let mut full_args = env; full_args.extend(...)`), no clone. Not a win. |
| Tensor view-method alloc elision (`transpose`/`transpose_axes`/`reshape`) | "5–15%" | The cloned shape/strides are 2–4-element Vecs; the agent's own follow-up estimate was <2%. Real but low-magnitude; no size win without the Rc change above. |
| Trace-`enabled` branch on the hot loop | "5–10%" | The branch is a well-predicted always-false in normal runs and CANNOT be compiled out (instrumentation is runtime-toggled by Option-B). Marginal. |

## 2. Prioritized plan (dedicated sessions, in value order)

### P1 — `Tensor.shape/strides` as `Rc<[usize]>` (memory + speed)
The single highest-leverage change: shrinks `Tensor` ~72 → ~40 B (and
thus every `Value` holding a tensor and every `Rc<Vec<Value>>` array of
them), AND converts the view ops (transpose/reshape/slice/broadcast) from
"clone two Vecs" to "bump two refcounts" — eliminating allocations in the
hottest tensor loops (the chess-RL / PINN inner loops).
- **Scope:** ~250 sites in `tensor.rs` + `builtins.rs` + `cjc-ad`. Field
  reads stay `&[usize]` via `Rc` deref (mostly mechanical); field writes
  build a fresh `Rc<[usize]>`; the few in-place shape mutations rebuild.
- **Determinism:** SAFE — metadata only, no FP, order unchanged. Contract
  invariants 1/2/3 (FP) untouched.
- **Measurement:** `alloc_bytes_in_window` (Phase F0 instrumentation) on
  the `tensor_*` corpus family before/after; `cana_diagnostics` wall-clock
  on `tensor_mm`/`tensor_ew`. Parity gate + double-run row-hash must stay
  green.
- **Risk:** aliasing on the in-place mutation paths — the reason this is a
  dedicated session, not a rushed edit.

### P2 — `Value` slimming via boxing the cold large variants (memory)
After P1 (Tensor ~40 B), box `Enum`/`Closure`/`Struct`/`Regex` (rare in
hot code, so few hot-path match sites) to bring `Value` toward ~24–40 B.
Halves the footprint of array/tuple element storage (the churn loops the
whole D/F arc is about).
- **Scope:** every match arm + constructor of the boxed variants across
  the executor/eval/runtime/dispatch crates.
- **Determinism:** SAFE — indirection only.
- **Measurement:** `size_of::<Value>()` assertion test; `alloc_bytes` on
  `mem_grad_*`. Cost to watch: a pointer-chase on the boxed variants —
  must verify the COMMON path (Int/Float/Tensor/Array) is untouched and
  the boxed variants aren't hot (profile first).

### P3 — Non-escaping array/tuple literal elision (memory + speed)
The churn-loop allocation the mem_grad corpus models: a per-iteration
`[a, b]` / `(x, y)` that dies immediately. If escape analysis
(`cjc-mir/src/escape.rs`) can prove the literal does not escape its
statement, back it with the frame arena instead of a fresh `Rc<Vec>`.
This is the natural continuation of Phase D (which proved these allocs
cost real wall-clock) — it speeds programs DCE can't help (the alloc is
live-but-ephemeral, not dead).
- **Determinism:** SAFE if the escape proof is conservative (when in
  doubt, `Rc` — the current behavior).
- **Risk:** correctness of the escape proof. The highest-value but
  highest-care item; needs its own design + heavy proptest/bolero on
  escape edge cases (returned, captured, stored-in-longer-lived).
- **Measurement:** `alloc_bytes` on `mem_grad_*` + `holdout_alloc_pulse`;
  `cana_diagnostics` wall-clock (these are exactly the Phase D subjects).

### P4 — Softmax/layer-norm scratchpad pooling (memory) — ACCURACY-GATED
Per-row temp-vector allocations in `tensor.rs` softmax/layer_norm. A
pooled scratchpad would cut allocations, BUT the auditor flagged this
**accuracy-DANGEROUS**: the Kahan compensation in the reduction must keep
its exact accumulation order, and a reused buffer must be same-sized and
not carry compensation state across reuses. Only attempt with a
bit-identical-output proof on the chess-RL weight hash. Lower priority
than P1–P3 precisely because of the accuracy risk.

## 3. What was REJECTED (don't re-propose)

- **Closure-env Rc<Vec>:** already moves, not a clone (§1).
- **Trace-branch removal:** can't compile out runtime-toggled
  instrumentation; branch predicts well (§1).
- **`force_default_seq` exploration config (Phase H):** duplicate of
  `force_all` — pass-count features are order-invariant.
- **Parallel/SIMD-FMA reductions:** forbidden outright by the determinism
  contract (invariants 1–3). Any "vectorize the reduction" idea is a
  determinism regression, full stop.

## 4. Methodology note

This roadmap exists because the code-reading panel's magnitude/effort
estimates did not survive verification — the closure "win" was already
done, the "medium-effort" Rc change is 250 sites, and the view micro-opt
is <2%. The lesson mirrors the whole D–H arc: **measure before claiming.**
Each P-item above ends with a concrete measurement (alloc bytes via the
Phase-F0 instrumentation, or wall-clock via the Phase-D
`cana_diagnostics` harness) so its win is proven on silicon, not
asserted — and gated by the parity + double-run determinism checks
first.
