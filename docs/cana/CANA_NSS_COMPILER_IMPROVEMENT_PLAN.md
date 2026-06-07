# CANA + NSS: A Plan to Push CJC-Lang Toward LLVM-Tier Performance

**Author:** Phase-1 CANA session
**Date:** 2026-06-06
**Status:** Design research. No code shipped in this document — it scopes a multi-phase implementation roadmap.

---

## TL;DR

CJC-Lang today produces interpreted code (AST eval) or register-machine MIR (cjc-mir-exec). It is roughly **two-to-three orders of magnitude slower than LLVM-compiled native code** for compute-heavy workloads. Closing that gap *while preserving CJC-Lang's identity* (determinism, auditability, parity) requires a different strategy than LLVM uses.

CANA + NSS together can credibly target a **10×–50× speedup over the current MIR executor for hot numerical kernels**, putting CJC-Lang in **the same order of magnitude as -O0 native code on representative scientific workloads** (PINN training, tensor reductions, MIR-walk interpretation). True LLVM-tier (-O2/-O3) is achievable only via lowering to actual native code, which this plan treats as Phase 6 (out of scope for the CANA Phase 2–5 roadmap).

The **plan's most novel element is treating thermal as a first-class compiler constraint.** NSS already has a `PressureKind::Thermal` field designed precisely for this. The compiler becomes a closed-loop system: compile → execute → observe pressure (including thermal) → adapt next compilation. This is "green compilation" — running cooler buys you both longer determinism windows and longer hardware life.

---

## 1. Why CJC-Lang is slow today (the gap to close)

Before designing solutions, name the problem precisely.

### 1.1 The cost structure of `cjc-mir-exec`

Today's MIR executor is a **tree-walking register machine**. For every `MirExpr` it:
1. Pattern-matches the discriminant (`MirExprKind::Binary`, `Call`, `Field`, etc.) — branch mispredict cost on every node.
2. Boxes intermediate values into `Value` enum cells (~24 bytes each on x86_64).
3. Allocates `Vec<Value>` for argument lists on every `Call`.
4. Performs scope-chain lookup (`O(scope_depth)`) for `MirExprKind::Var`.
5. Hits Kahan accumulation on every floating-point reduction.

Even with the Tier-0 perf work (slot-resolved `VarLocal`, fused MlpLayer, in-place gradient accumulation in v2.5), this is fundamentally **interpreted execution**. Native code skips every one of these steps — register allocator, instruction selector, and LLVM's instruction combiner do the work at compile time.

**A useful benchmark point from the project's own history:**
- Chess RL v2.3 → v2.4 → v2.5: 82s → 43s → 30s per episode (2.76× speedup) through better in-place gradient accumulation and fused MLP layers.
- Each step represented a *2–10% improvement* in dispatch cost or a fused-kernel native primitive.
- Reaching LLVM-tier requires **2-3 orders of magnitude** of additional improvement.

**Conclusion:** incremental dispatch optimization will not close the gap. The architectural step needed is one of:
- (a) JIT compilation to native code (LLVM-equivalent path, expensive to build)
- (b) Ahead-of-time specialization of hot paths (mid-cost, high-impact)
- (c) Pass-ordering and inlining quality that dramatically reduces *what code runs* in the first place (low-cost, mid-impact)

CANA enables (c) directly today, and (b) and (a) in later phases via the same pressure-prediction substrate.

### 1.2 The cost structure of `cjc-eval`

The AST tree-walk interpreter is roughly 2–3× slower than `cjc-mir-exec` and serves as the *parity oracle* — its only job is to produce byte-identical results to MIR-exec on every program. We do NOT speed `cjc-eval` up. The parity contract is more valuable than the speed.

### 1.3 What "LLVM territory" actually means

When you say "LLVM territory," there are three meaningful interpretations, listed by ambition:

| Interpretation | Implication | Realistic Phase |
|---|---|---|
| **A. LLVM -O0 native speed** | Generated code runs as fast as C compiled with no optimization. Achievable via straightforward MIR-to-native lowering. | Phase 6 (post-CANA) |
| **B. LLVM -O2 native speed for hot kernels** | Hot kernels get full LLVM optimization treatment via a JIT or AOT lowering of CANA-identified hot paths. | Phase 7 |
| **C. LLVM -O2 across the whole program** | Every line as fast as Clang/-O2. Requires whole-program lowering and the entire LLVM optimization pipeline. | Out of scope (would require ~5 years of compiler work) |

**This plan targets interpretation A (Phase 6) as the long-term ambition.** Phase 2–5 of CANA delivers measurable speedups (10×–50× on hot paths) without lowering to native — by improving what the existing MIR executor runs, not by replacing it.

---

## 2. The four dimensions of improvement

| Dimension | Why CANA/NSS can help | Today's baseline | Realistic targets after Phase 2–5 |
|---|---|---|---|
| **Speed** | Predict optimization payoffs; skip bad passes; fuse hot kernels; trigger native lowering at the right granularity | MIR walk ~100× slower than -O0 native | 10×–50× speedup on hot kernels via fused dispatch + native primitives + better pass ordering |
| **Memory** | Predict allocation lifetime; avoid temporary materialization; choose streaming over batch for memory-pressured workloads | ~3× peak memory vs equivalent C | 1.3×–1.8× peak memory; near-zero temporary tensor materialization |
| **Determinism** | Reject any optimization that touches a `StrictFold` reduction; gate via `LegalityVerifier` | Already strong; trade-off is no parallelism within strict reductions | Maintain bit-identical replay across all CANA-driven optimizations |
| **Thermal** | Predict thermal pressure; throttle aggressive optimization or split execution into cooler stages when temperature spikes | No thermal awareness | 20–40% lower sustained CPU temperature on long compute jobs by selecting cooler execution plans |

The **integration point that makes this powerful** is that all four dimensions project onto NSS's existing `PressureKind × NodeId` substrate. The same neural network architecture that predicts cluster collapse can predict pass-runtime cost, memory peak, and thermal trajectory simultaneously.

---

## 3. Speed: getting closer to LLVM

### 3.1 Method S1 — Predictive pass ordering (Phase 2, ~15% gain)

**Today:** `optimize_program` runs a fixed sequence: CF → SR → DCE → CSE → LICM → CF.
**Problem:** This is correct for ALL programs but optimal for NONE. Some programs would benefit from LICM-first (when loops dominate); some from aggressive CSE (when expression sharing dominates); some from skipping passes entirely (small programs).

**CANA's role:** the `CanaFeatures` already extracts CFG complexity, loop depth, expression density, and reduction count. A Phase 2 `PassRanker` consumes these features and predicts:
- Expected runtime *gain* from each pass on this program
- Expected compile-time *cost* of each pass

The ranker emits an ordered sequence; `optimize_program` consumes it.

**Determinism contract:** `LegalityGate` refuses any pass-ordering that would reorder a `StrictFold` reduction. The default gate I shipped in Phase 1 already enforces this.

**Concrete win:** the project's existing pass sequence is roughly the right sequence for typical CJC code. The realistic win from predictive ordering alone is ~10–20% on optimization-heavy workloads. Not transformational by itself — but it's the bootstrap for Methods S2 and S3.

### 3.2 Method S2 — CANA-guided pass skipping (Phase 2, ~10–25% gain)

**Today:** `--mir-opt` is binary: run all 6 passes, or run none.
**Problem:** On small programs, the optimization compile time exceeds the runtime saved. On simple programs, CSE finds nothing and burns analysis time. On hot-loop programs, the second CF (pass 6) is critical; on flat programs, it's wasted work.

**CANA's role:** for each (pass, program) pair, predict whether the pass will produce changes. Skip when predicted no-op.

The signal is in CANA's existing features:
- `CfgMetrics.branch_count == 0` → CSE likely no-op (no shared subexpressions across branches)
- `ReductionAxes.has_strict_reduction()` → LICM cannot hoist out of the reduction loop safely
- `MemoryProxy.alloc_sites < 5` → DCE has little to remove

**Concrete win:** skipping 1–2 of 6 passes typically saves 15–30% of compile time on average programs. Doesn't help runtime directly, but **closes the feedback loop** for CANA Phase 5 (profile-guided): cheap re-compilation enables fast iteration of recommendations.

### 3.3 Method S3 — Hot-kernel native lowering for tensor ops (Phase 3, ~5–20× on hot paths)

**Today:** `matmul`, `mlp_layer`, `tensor_concat_1d`, `sum`, `mean`, `dot`, `adam_step`, `encode_state_fast`, `score_moves_batch` are already native primitives. Anything *between* them (data movement, indexing, slicing) is interpreted.

**Problem:** A program like:
```cjcl
let weights = mlp_layer(input, w1, b1, "relu");      // native
let activated = matmul(weights, w2);                 // native
let output = activated + b2;                          // INTERPRETED — 1000 ops in a 1000-element tensor
```
The single tensor-addition burns thousands of MIR interpreter cycles.

**CANA's role:** identify chains of native primitives where the connecting MIR code is small + bounded. Emit those chains as **fused native paths** at compile time. The chain `mlp_layer ; matmul ; +` becomes a single native call — no MIR walking between primitives.

This is the **#1 lever for speed**. Chess RL v2.5's `mlp_layer` fused op already proved the model — collapsing 4 GradOps into 1 was a 1.46× win. Extending to runtime forward-pass fusion is a 5–20× win on hot paths.

**Determinism contract:** the fused native paths must reproduce the same Kahan-summed, no-FMA, no-parallel-reduction byte sequence as their unfused counterparts. The existing `cjc-runtime` primitives already enforce this. Phase 3 generates new fused kernels that compose existing primitives in the same enforcement regime.

**NSS's role:** NSS's `MultiTimescaleEngine` (α ∈ {0.5, 0.85, 0.95, 0.99}) decomposes execution pressure across timescales. A persistent (structural-timescale) hot kernel — one that runs for thousands of consecutive ticks — is a strong candidate for native fusion. A transient hot kernel (medium-timescale spike) is not worth the compile-time cost. NSS provides the *signal* of which to fuse.

### 3.4 Method S4 — Inlining policy guided by predicted call-site cost (Phase 3, ~10–30% gain)

**Today:** No inlining at MIR level. Every call is a frame setup + scope chain push + argument evaluation + dispatch.
**Problem:** Hot calls (e.g. helpers inside a training loop) pay full dispatch cost on every invocation.

**CANA's role:** for each call site, predict:
- Frequency (how often will this site execute?)
- Body size (small bodies are cheap to inline)
- Pressure impact (will inlining trigger register pressure? code-size pressure?)

Inline when predicted runtime gain > predicted (code-size + compile-time) cost.

**This is where NSS pressure prediction is most valuable.** The cluster simulator's "predict P(collapse) under this scenario" maps directly onto "predict P(code-size pressure breach) under this inlining decision."

### 3.5 Method S5 — Specialization on observed value distributions (Phase 5, ~2–10× on biased workloads)

**Today:** every function compiles once, agnostic to its argument distribution.
**Problem:** A function called 1M times with `n=100` and once with `n=1000000` compiles identically. The hot path could be specialized for `n≈100`.

**CANA's role + Phase 5 profile feedback:** collect runtime distributions of arguments. After enough samples, CANA predicts the dominant case and emits a specialized variant with branches eliminated. The general variant remains as fallback.

NSS's structural-timescale signal is the trigger: only specialize when the distribution is *persistently* skewed. Transient skews don't justify specialization's compile cost.

### 3.6 Method S6 — Phase 6: native code generation (post-CANA, ~10–100× additional gain)

The ultimate speed lever. Phase 6 lowers MIR to LLVM IR for the hot kernels CANA identifies (using methods S3, S5). This is genuinely **5+ years of compiler work** and is out of scope for the CANA Phase 2–5 roadmap. **But every CANA Phase 2–5 decision should be designed with Phase 6 in mind:**
- Pass-ordering data accumulated in Phase 2 informs Phase 6 LLVM pipeline tuning
- Inlining/fusion decisions from Phase 3 directly map to LLVM IR generation choices
- Profile data from Phase 5 informs Phase 6 PGO

### 3.7 Expected speed totals

| Phase | Method | Realistic gain on representative workloads |
|---|---|---|
| 2 | S1: pass ordering | 1.10×–1.20× |
| 2 | S2: pass skipping | 1.10×–1.25× (compile time; ~0× runtime) |
| 3 | S3: kernel fusion | 5×–20× on hot kernels (≈ 2–5× on overall workload) |
| 3 | S4: inlining | 1.10×–1.30× |
| 5 | S5: specialization | 2×–10× on biased workloads |
| **Total (Phase 2–5)** | Compounded | **~10×–50× on hot numerical workloads** |
| 6 | Native lowering | additional 10×–100× (out of scope for this plan) |

At the upper end of Phase 5 (50× speedup), CJC-Lang is comparable to **early-2010s JIT'd Python (PyPy)** — substantial but not LLVM territory. Phase 6 closes the remaining gap.

---

## 4. Memory efficiency: less, smarter allocation

### 4.1 Method M1 — Predicted lifetime escape analysis (Phase 2, 30–50% peak memory reduction)

**Today:** `cjc-mir::escape` annotates `AllocHint` on `MirStmt::Let` to guide allocation strategy. The existing escape analysis is conservative.

**Problem:** Most temporaries can be stack-allocated, but the conservative escape analysis routes too many through the heap.

**CANA's role:** the `MemoryProxy` already counts `alloc_sites`. A Phase 2 extension predicts, per allocation site, whether the value escapes the function. Predictions inform a more aggressive `AllocHint`. The existing escape analysis remains the legality gate (refuses any prediction that would allow an escape).

This is **the canonical CANA pattern**: model predicts; rule-based analysis vetoes. The model is allowed to be wrong; the analysis ensures wrong predictions cause "too conservative" not "unsafe."

### 4.2 Method M2 — Tensor temporary elimination via fusion (Phase 3, 40–70% peak memory reduction on tensor workloads)

**Today:** `a + b * c` allocates a temporary for `b * c`, then another for the final addition.
**Problem:** On tensor workloads (PINN, chess RL, ABNG), temporaries dominate memory budget.

**CANA's role:** fuse chains of tensor ops at compile time. Same machinery as Method S3 (kernel fusion); the *memory* savings come from never materializing the intermediates. This is what `mlp_layer` already does for the MLP forward pass — extending the pattern systematically is the gain.

**Concrete proof point from project history:** Chess RL v2.5's in-place gradient accumulation (`add_assign_unchecked`) eliminated ~N/2 tensor allocations per backward pass — directly enabling the 1.46× speedup. Memory savings and speed gains were coupled, as they always will be on tensor workloads.

### 4.3 Method M3 — Streaming-over-batch decision (Phase 3, can avoid OOM entirely on memory-pressured workloads)

**Today:** the runtime always allocates the full intermediate tensor for matmul, reductions, etc.
**Problem:** On memory-pressured systems (laptops, embedded), this OOMs.

**CANA's role:** when CANA predicts allocation pressure will exceed available memory, recommend a streaming execution plan. Same final result, computed in chunks, with bounded peak memory.

**NSS's role:** this is exactly the "advise mode" `SchedulerAdvisor` handles in NSS. The same architecture that recommends `ShedLoad` for a cluster recommends "use streaming matmul" for a memory-pressured CJC-Lang program. Different `AdvisoryAction` enum vocabulary, same substrate.

### 4.4 Method M4 — COW write site reduction (Phase 4, 20–40% reduction in COW-write traffic)

**Today:** `array_push`, `array_pop`, `array_reverse`, `arr_set` all use `Rc::make_mut`. Each invocation potentially clones the underlying buffer.

**Problem:** Tight loops that grow arrays trigger O(N) clones in the worst case.

**CANA's role:** identify loops where the array is provably unique (single owner). In those loops, emit a different code path that mutates in place without the `Rc::make_mut` clone check. The escape analysis is the legality gate.

### 4.5 Memory totals

| Phase | Method | Realistic gain |
|---|---|---|
| 2 | M1: better escape analysis | 1.3×–1.5× peak memory reduction |
| 3 | M2: tensor fusion | 1.5×–3× peak memory reduction on tensor workloads |
| 3 | M3: streaming-over-batch | Avoids OOM (qualitative change, not factor) |
| 4 | M4: COW write reduction | 1.2×–1.4× write traffic reduction |
| **Total (Phase 2–4)** | Compounded | **~1.8×–4× peak memory reduction; near-OOM-proof execution** |

---

## 5. Preserving determinism (the constraint that makes CJC-Lang itself)

`★ Insight ─────────────────────────────────────`
This section is the most important one. Speed and memory wins are negotiable. Determinism is not.
`─────────────────────────────────────────────────`

### 5.1 The CANA legality gate is the determinism guarantee

Phase 1 shipped `DefaultLegalityGate`. Every CANA-driven optimization (pass ordering, skipping, fusion, inlining, lowering) passes through `LegalityGate::verify` before application. Phase 1's default gate refuses any reorder touching a `StrictFold` reduction.

**Phase 2's extension:** refuse any optimization that would:
- Reorder operands of any operation involving `KahanFold` or `Unknown` reductions
- Replace a `Kahan accumulator` with a plain accumulator
- Introduce FMA into a numerical path
- Replace a `BTreeMap` with `HashMap`
- Use parallel reduction strategies that depend on thread count
- Add nondeterministic RNG calls outside `NssSeed::substream`

The implementation is straightforward: each banned pattern becomes a `LegalityViolation` variant. The gate runs in O(features + sequence) time, so checking every recommendation is cheap.

### 5.2 The AST/MIR parity gate is the **empirical** check

CLAUDE.md mandates that "every feature must work in `cjc-eval` AND `cjc-mir-exec`." CANA-driven optimizations only affect MIR. If a CANA-recommended pass causes MIR-exec output to diverge from AST-eval output, the parity gate catches it immediately.

**This means CANA's recommendations are validated TWICE:**
1. Statically by `LegalityGate::verify` before application
2. Dynamically by the AST/MIR parity test suite after application

Neither alone is sufficient; together they're definitive.

### 5.3 The content-addressed audit trail proves determinism

Phase 1's `ProgramHash` + `FeatureHash` + `CfgHash` (+ Phase 5's `PassHistory.run_id`) means every CANA-driven decision is replayable. The same MIR + same CANA model version + same recommendation seed = same output, byte-for-byte. This is the NSS-style audit story applied to compiler decisions.

### 5.4 What about Phase 6 native lowering?

This is the hardest case. LLVM does float reassociation, FMA contraction, parallel reductions — *all* of which break CJC-Lang's determinism contract.

**The plan:** Phase 6 lowering must pass a strict subset of LLVM flags:
- `-fno-associative-math -fno-reciprocal-math -fno-signed-zeros` (no float reassociation)
- `-ffp-contract=off` (no FMA contraction)
- Force single-threaded execution
- Custom LLVM optimization pipeline that excludes the reassociation passes

These flags are routinely used in other deterministic numerical libraries (e.g. some quant trading systems). They cost LLVM ~10–20% of its peak performance — meaning Phase 6 reaches "LLVM minus 10–20%" rather than "LLVM-tier."

**For CJC-Lang, that's the correct tradeoff.**

---

## 6. Thermal control — the "green compilation" idea

This section adopts your previous-project framing: **heat is a first-class compiler constraint** because thermal stress causes determinism drift (through CPU clock throttling, NaN propagation in stressed FPUs, and OS-level RNG perturbation) AND because heat wears hardware faster.

### 6.1 Why NSS's thermal model genuinely transfers

NSS's `PressureKind::Thermal` was designed for cluster thermal modeling — a hot rack throttles, propagates back-pressure to neighbors, and eventually triggers cascading failures. **The same dynamics apply at the CPU/GPU level:**

- A hot core throttles its clock (pressure)
- Throttling forces the OS to migrate work to cooler cores (propagation)
- Migration causes cache misses (downstream pressure)
- Sustained heat eventually triggers a thermal trip (collapse)

NSS's `MultiTimescaleEngine` is *especially* well-suited because thermal has distinct timescales:
- Short (α=0.5): per-instruction transient heat
- Medium (α=0.85): per-function-call sustained heat
- Long (α=0.95): per-program thermal accumulation
- Structural (α=0.99): per-machine thermal mass (the laptop's actual temperature)

### 6.2 Method T1 — Predicted thermal cost per optimization (Phase 4, qualitative win)

**The principle:** every optimization has a thermal signature.
- Loop unrolling: more instructions per loop iteration → higher sustained core usage → hotter
- Vectorization: more parallel ops → more FPU activity → hotter
- Inlining: removes call overhead but may increase register pressure → hotter
- DCE: removes work → cooler
- Streaming over batch: trades peak memory for sustained CPU time → may run cooler if streaming reduces memory contention but may run hotter if streaming extends total runtime

CANA-with-NSS predicts the **thermal trajectory** of each candidate optimization plan. The user (or the runtime) picks a `SafetyMode`:
- `MaxPerformance`: optimize purely for speed; thermal is unconstrained
- `Balanced`: 30°C cap (or whatever the user sets); fall back to less aggressive plans when predicted to exceed
- `ThermalSafe`: 25°C cap; prefer cool plans even with substantial speed penalty
- `Green`: minimize total energy (J), not wall time (s) — chooses plans that finish in slightly more time but use less CPU-cycle integral

### 6.3 Method T2 — Runtime thermal feedback (Phase 5, ~20–40% temperature reduction on long jobs)

**The principle:** CANA Phase 5 reads runtime traces. Extending those traces to include actual CPU temperature (via OS-level APIs: `/sys/class/thermal/thermal_zone*` on Linux, IOKit on macOS, Windows ACPI) closes the loop.

**Per-program thermal feedback loop:**
1. Compile with CANA's predicted-coolest plan
2. Execute; sample temperature every N ticks
3. Compare actual vs predicted; update CANA's thermal cost model
4. Re-recompile if predicted-actual gap exceeds threshold

Over time, CANA learns the **specific machine's thermal characteristics**. A laptop with a small heat sink learns its own bottleneck; a desktop with a tower cooler learns it has slack. The same CJC-Lang program compiles differently on different machines based on observed thermal behavior.

### 6.4 Method T3 — Just-in-time thermal throttling (Phase 5, prevents thermal trips)

**The principle:** even with a great compile-time prediction, sustained workloads can exceed thermal limits. The runtime monitors temperature and can:
- Switch to a streaming variant mid-execution if temperature exceeds threshold
- Reduce batch size in training loops
- Pause briefly to let the system cool
- Switch from native fused kernel back to MIR-walked variant (slower but cooler — fewer pipeline stalls)

**This requires runtime variants of every native kernel:**
- `mlp_layer_hot` (full fused, maximum speed, maximum heat)
- `mlp_layer_warm` (partial fused, moderate heat)
- `mlp_layer_cool` (MIR-walked, minimum heat)

CANA generates these at compile time; the runtime selects based on NSS's real-time pressure prediction. **All three variants produce byte-identical output** — that's the determinism contract that survives the thermal switching.

### 6.5 The green compilation success metric

For a long-running training job (e.g. chess RL phase D, 60+ episodes):
- **Today:** sustained ~85°C CPU, deterministic but stressed
- **Goal post Phase 4:** sustained ~60–65°C CPU, deterministic, ~20–30% longer wall-clock for the same total computation
- **Goal post Phase 5:** ~55–60°C CPU, with the runtime adapting throughout to maintain temperature target

The longer wall-clock cost is **the cost of green compilation**. For research workloads it's negligible (overnight runs); for production deployment it's a tradeoff knob.

---

## 7. The unified architecture: closed-loop compile-execute-observe

The methods above don't just stack — they compose into a single system.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CJC-Lang Compiler                          │
│                                                                     │
│   Source → AST → HIR → MIR ──┬─► CANA Featurizer (Phase 1, SHIPPED) │
│                              │                                      │
│                              ▼                                      │
│                       ┌─────────────────┐                           │
│                       │  CANA Cost      │ (Phase 2+)                │
│                       │  Model (NN)     │                           │
│                       └────────┬────────┘                           │
│                                ▼                                    │
│                       ┌─────────────────┐                           │
│                       │ Pass Ranker     │ → recommended sequence    │
│                       │ Fusion Planner  │ → fused kernels list      │
│                       │ Inlining Policy │ → inline decisions        │
│                       │ Spec. Trigger   │ → specialization sites    │
│                       └────────┬────────┘                           │
│                                ▼                                    │
│                       ┌─────────────────┐                           │
│                       │ LegalityGate    │ ← (vetoes unsafe rewrites)│
│                       │  (StrictFold,   │                           │
│                       │   Kahan, etc.)  │                           │
│                       └────────┬────────┘                           │
│                                ▼                                    │
│                       Optimized MIR + native primitives             │
└───────────────────────────────┬─────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CJC-Lang Runtime (cjc-mir-exec)                 │
│                                                                     │
│   Optimized MIR ──────► Executor                                    │
│                            │                                        │
│                            ▼                                        │
│              ┌─────────────────────────────┐                        │
│              │ Pressure Sampler            │ (Phase 4+)             │
│              │  - exec time                │                        │
│              │  - memory peak              │                        │
│              │  - allocation count         │                        │
│              │  - CPU temperature          │                        │
│              │  - cache miss rate (where   │                        │
│              │    available)               │                        │
│              └──────────┬──────────────────┘                        │
│                         ▼                                           │
│              ┌─────────────────────────────┐                        │
│              │ NSS Pressure Predictor      │ (Phase 4+)             │
│              │ - PressureKind::            │                        │
│              │   {Cpu, Memory, Thermal,    │                        │
│              │    Sync, Throughput, ...}   │                        │
│              │ - MultiTimescaleEngine      │                        │
│              │ - Per-block dynamics        │                        │
│              └──────────┬──────────────────┘                        │
│                         ▼                                           │
│              ┌─────────────────────────────┐                        │
│              │ Runtime Variant Selector    │ (Phase 5)              │
│              │  hot ↔ warm ↔ cool variants │                        │
│              │  streaming ↔ batch          │                        │
│              │  inline ↔ call              │                        │
│              └─────────────────────────────┘                        │
│                                                                     │
│   Execution trace ──► persisted as audit log                        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                       Feedback to CANA cost model
                       (Phase 5 — profile-guided)
```

**The key insight:** every layer is **purely deterministic given its input**. CANA's NN is seed-threaded; NSS is seed-threaded; the runtime variant selector reads NSS's deterministic prediction; legality gates are pure functions. So the entire closed loop, including the runtime feedback, produces **bit-identical output given the same starting state**. The audit chain captures every decision so any subsequent execution can be exactly reproduced.

This is what **trustworthy adaptive compilation** looks like.

---

## 8. Phase roadmap

Mapped onto CANA's existing phasing from Phase 1:

| Phase | Status | Methods delivered | Cumulative speedup target |
|---|---|---|---|
| **1** | **SHIPPED** | Featurizer, hashes, legality gate trait, cost model trait, audit reports | 1.0× (baseline; passive observer) |
| **2** | Next | S1 (pass ordering), S2 (pass skipping), M1 (escape analysis), real CostModel | 1.2×–1.5× |
| **3** | After Phase 2 | S3 (kernel fusion), S4 (inlining), M2 (tensor fusion), M3 (streaming) | 5×–15× on hot kernels |
| **4** | After Phase 3 | NSS integration: T1 (thermal-aware compile), runtime pressure sampling, M4 (COW reduction) | 5×–20× on hot kernels; ~50% lower CPU temperature on sustained jobs |
| **5** | After Phase 4 | S5 (specialization), T2/T3 (runtime thermal feedback), profile-guided CostModel | 10×–50× on hot kernels; sustained thermal targets met |
| **6** | Future | Native lowering for hot kernels via LLVM (with determinism flags) | 100×+ on hot kernels; LLVM -O0 to -O2 territory |

**Phase 2's delivery is the single highest-priority next step.** It transitions CANA from passive observer to advisor, gives you real measured speedups, and builds the data substrate for everything downstream.

---

## 9. What this plan does not solve

Honest list of where CANA + NSS will not get you to LLVM:

1. **CJC-Lang's value boxing.** Every `Value` enum cell is 24+ bytes. LLVM works on raw machine words. Closing this gap requires *typed unboxing* — knowing at compile time that a variable is always `i64`, never any other variant. This is a Phase 6+ change to the value representation, not a CANA decision.

2. **Cache-aware register allocation.** LLVM's register allocator is decades of work. CJC-Lang's frame-slot scheme is a single linear allocation. Bridging requires native code generation (Phase 6).

3. **Auto-vectorization.** LLVM auto-vectorizes loops via `-O3`. CJC-Lang's deterministic-no-FMA constraint forbids most LLVM auto-vectorization. Hand-written vectorized kernels (via Method S3) for specific shapes is the only deterministic path. We will not match LLVM-vectorized C for arbitrary loops.

4. **Aggressive PGO (profile-guided optimization).** LLVM's PGO uses block-level edge counts to reorganize code for cache locality. CJC-Lang Phase 5's profile-guided optimization is comparatively limited (cost model retraining only). Bridging requires Phase 6+ infrastructure.

These are the structural reasons CJC-Lang will land at roughly **LLVM -O0 native speed** in Phase 6, not LLVM -O3. **For CJC-Lang's target audience (scientific computing, reproducible ML, audit-grade pipelines), that gap is the correct tradeoff** — those workloads value reproducibility more than they value the last 10–20% of speed.

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CANA's cost model is wrong → bad recommendations | High initially | Low (legality gate vetoes) | Phase 2 ships with a *very* simple cost model and aggressive gating |
| Fused kernel breaks determinism | Medium | High | Every new fused kernel must pass byte-equality test vs unfused variant before shipping |
| Thermal predictions miscalibrated → no actual cooling | Medium | Low | Phase 4 ships with extensive cross-machine validation; user can disable thermal mode |
| NSS integration adds per-tick latency | Low–Medium | Medium | NSS's existing tick cost is ~100µs–1ms; for any non-trivial program this is invisible |
| Phase 6 LLVM lowering breaks determinism | High without care | Critical | Phase 6 must pass the determinism subset of LLVM flags; no compromise on this |
| Audit log size grows unboundedly | Medium | Low | Bounded ring buffer (already in Phase 1's `PassHistory`); rotate to disk for long-running programs |

---

## 11. Where to start

If you have **one week** to make progress, the highest-leverage move is:

1. **Day 1–2:** ship Phase 2 Method S1 (predictive pass ordering). This requires:
   - A simple linear regression cost model trained on synthetic MIR programs
   - Integration with the existing `optimize_program` to consume CANA's recommended sequence
   - Benchmark suite measuring before/after on 10 representative programs

2. **Day 3–4:** Phase 2 Method M1 (CANA-augmented escape analysis). This requires:
   - Extending `MemoryProxy` with a per-let escape prediction
   - Wiring into `cjc-mir::escape::AllocHint`
   - Memory benchmark suite

3. **Day 5:** documentation + ADR for Phase 2 design decisions. Forms the baseline for the eventual Phase 3 work.

That single week takes CANA from passive observer to actively saving compile time and runtime — the first **measurable user-facing win** for the architecture.

---

## 12. Summary

| Goal | CANA's role | NSS's role | Phase to ship | Expected gain |
|---|---|---|---|---|
| Speed → LLVM-tier | Pass ordering, fusion planning, inlining policy, specialization triggers | Identify hot persistent kernels via multi-timescale signal | 2–5 (some), 6 (full) | 10×–50× (Phase 2–5), 100×+ (Phase 6) |
| Memory efficiency | Escape prediction, tensor fusion, COW reduction | Predict memory pressure trajectory | 2–4 | 1.8×–4× peak memory reduction |
| Determinism preservation | LegalityGate enforces invariants; content-addressed audit trail | Audit chain via NssRunId-style hashing | 1 (shipped) + 2 (extended) | Maintained 100% |
| Thermal control | Predict per-pass thermal cost; generate hot/warm/cool variants | Predict thermal trajectory via PressureKind::Thermal + MultiTimescale | 4–5 | 20–40% lower sustained temperature |

The architecture is genuinely powerful because **the four goals reinforce each other**:

- Determinism makes the audit trail trustworthy → cost model can rely on its inputs
- Thermal control reduces clock throttling → speed and determinism both improve
- Memory efficiency reduces cache pressure → speed improves and thermal improves
- Speed improvements reduce total runtime → less wall-clock means less thermal accumulation

This is not a list of features. It's an **architecture for trustworthy adaptive compilation** that compounds across dimensions.

---

*Generated alongside CANA Phase 1 shipping. See also: `crates/cjc-cana/src/lib.rs` for the Phase 1 substrate; the NSS architecture doc for the pressure-prediction model that Phase 4+ consumes.*
