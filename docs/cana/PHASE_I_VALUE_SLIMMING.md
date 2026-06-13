# Phase I — Memory: Slimming `Value` (the contained win) + the verified roadmap

**Date:** 2026-06-13 (follows Phase H on `claude/stupefied-liskov-83b258`)
**Source:** the Runtime + Numerical passes of the stacked-role
optimization arc, ground-truthed against the code (THE RULE) and gated
by `docs/cana/DETERMINISM_CONTRACT.md`.
**Exit criterion:** a measured, determinism-safe memory reduction landed
with full test discipline, plus a verified roadmap scoping the larger
wins that the blast-radius reality reserves for dedicated sessions.

## 0. What this phase is (and honestly is not)

The design panel proposed several "high-leverage, low-risk" speed/memory
wins. **Verification against the code changed every conclusion** (see
`PERFORMANCE_ROADMAP.md` §1): the closure-env "clone" is already a move;
the `Tensor.shape/strides` Rc change is a ~250-site refactor; `Value`
boxing is pervasive; the view-method micro-opt is <2%. So Phase I lands
the ONE win that survived verification as genuinely contained, and
documents the rest as scoped, dedicated-session work rather than forcing
a pervasive refactor late and risking the bit-identical guarantee.

## 1. The contained win: box the `SparseCsr` payload

Measured ground truth (`cargo test -p cjc-runtime --test size_probe`):

```
size_of::<Value>()  = 88   →  72   (-18%)
size_of::<Tensor>() = 64        64
size_of::<SparseCsr>() = 88  (the single largest variant — set Value's size)
```

`Value` is stored BY VALUE in arrays (`Rc<Vec<Value>>`), tuples, frame
slots, and scope maps — so `size_of::<Value>()` multiplies across nearly
all interpreter memory. A 64-element array was 64 × 88 = 5,632 B of
value storage; it is now 64 × 72 = 4,608 B — **16 B saved per element,
everywhere**, for the cost of one heap indirection on a *rare* variant.

`SparseCsr` (88 B: three `Vec`s + two `usize`) was the outlier driving
`Value`'s size, but sparse matrices are niche — **19 construction/match
sites across 6 files**. Boxing `SparseTensor(Box<SparseCsr>)` moves that
88 B off the hot path: the common variants (Int/Float/Tensor/Array/
Tuple) now set the size, and the next-largest (`Enum` at 72 B — but
`Enum` is `Option`/`Result`, common, 52 sites) stays inline. The change
is **pure indirection**: semantics, iteration order, FP, and RNG are
untouched — determinism contract invariants all hold.

### Why not box more?

- **`Enum` (72 B)** would shrink `Value` further but is `Option`/`Result`
  — common, 52 hot-path match sites. Not contained.
- **`Tensor` (64 B)** co-sets the new size, but it is the hottest variant
  (boxing adds an indirection to every tensor op) and pervasive. Wrong
  trade. (The right Tensor win is `Rc<[usize]>` shape/strides —
  roadmap P1.)

## 2. Determinism / accuracy verification (the contract)

- Invariants 1–3 (FP/FMA/reduction order): **untouched** — no arithmetic
  changed.
- Invariants 4–5 (BTreeMap / iteration order): **untouched**.
- Invariant 6 (RNG): **untouched**.
- Invariant 7 (parity AST-eval ≡ MIR-exec): **gated** — `tests/fixtures`
  green after the change.
- Boxing is a layout-only change; `Value`'s `Clone`/`Debug`/`PartialEq`
  derive through the `Box` identically.

Tests: `crates/cjc-runtime/tests/value_size_guard.rs` (2) — a `Value`
size-ceiling regression guard (catches a future variant re-bloating
every value) + a boxed-sparse roundtrip (construction / type_name /
Display reach through the Box). Plus the existing sparse-op and
`cjc-snap` serialization suites (sparse encode/decode) green unchanged.

## 3. The roadmap (dedicated sessions)

`docs/cana/PERFORMANCE_ROADMAP.md` scopes the larger wins with measured
blast radius: P1 `Tensor.shape/strides` → `Rc<[usize]>` (~250 sites,
shrinks Tensor + elides view-op allocs), P2 `Value` boxing of the cold
large variants, P3 non-escaping array/tuple literal elision (the Phase-D
allocation-churn continuation), P4 softmax/layer-norm scratchpad pooling
(accuracy-gated). Each ends with a concrete measurement (Phase-F0
`alloc_bytes` or Phase-D `cana_diagnostics` wall-clock) and the parity +
double-run determinism gate, so its win is proven on silicon, not
asserted.

## 4. Verdict

A small, honest, measured, contained memory win (`Value` −18%, every
value everywhere) landed under the determinism contract — and a
verified, prioritized roadmap for the larger wins that the blast-radius
reality reserves for focused sessions. The methodology note stands: the
panel's code-reading estimates did not survive verification, so this
phase shipped what was real and scoped what was not, rather than forcing
a pervasive refactor that would risk the project's core guarantee.
