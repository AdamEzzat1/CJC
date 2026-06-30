# Future Infrastructure for CJC-Lang Compiler

Research note based on the current CJC-Lang compiler shape, the earlier infrastructure proposal, and the later CANA/NSS discoveries.

## Short Version

The older infrastructure plan still mostly holds up. I would keep roughly 75-85% of it, but the responsibility boundaries have changed.

Several ideas no longer look like brand-new standalone compiler systems. CJC-Lang already has partial infrastructure for SSA, CFGs, dominators, loop analysis, escape analysis, MIR verification, no-GC checks, arenas, tensor pools, profiling, SIMD/tiled tensor runtime support, and deterministic runtime structures.

The biggest update is that CANA and NSS should now be treated as first-class compiler intelligence layers:

- CANA should own pass ranking, profitability, compiler-aware feature interpretation, fusion guidance, compression guidance, and cost-model decisions.
- NSS should score pressure, runtime stability, thermal/memory/CPU behavior, and state-space risk once real trace data exists.
- MIR legality, alias/effect/shape analysis, and deterministic verifiers should remain the hard safety boundary.

The main missing infrastructure is not another large neural subsystem. It is the deterministic fact pipeline that lets CANA, NSS, and the future compression network all reason from the same verified compiler facts.

## What Still Works From The Older Plan

| Original Idea | Current Verdict | Updated Direction |
| --- | --- | --- |
| Region-based memory planning | Still strong, partly present | CJC already has frame arenas, pools, scratchpads, and runtime allocation support. The next step is a MIR-level lifetime planner that coordinates them. |
| Escape analysis | Still important, already started | `cjc-mir` has escape infrastructure. It should feed no-GC verification, arena placement, tensor reuse, and compression safety. |
| Lifetime-aware tensor buffer reuse | Very strong | Likely one of the highest-payoff areas. Needs tensor temp liveness, alias/effect facts, and shape facts. |
| Deterministic SSA and sparse optimizations | Still valid, partly present | SSA, CFG, dominators, loop analysis, and SSA optimization exist. The work is likely maturing these into the main optimization backbone. |
| Alias analysis | Still important | More important now because CANA may propose fusion/compression transforms that need hard legality proof. |
| Loop optimizer | Still valid | Should come after MIR/SSA canonicalization, alias/effect facts, and shape analysis. |
| Deterministic vectorization | Still valid | Runtime SIMD/tiled tensor pieces exist. Compiler vectorization should be legality-gated and deterministic. |
| Tensor fusion planner | Very relevant | This should probably be CANA-guided, verifier-approved, and NSS-scored. |
| Deterministic PGO database | Still very strong | Now looks like the missing bridge between MIR execution traces, CANA pass history, and NSS pressure scoring. |
| Pass profitability model | Valid but moved | This is CANA's job now. Avoid building a second independent profitability model. |
| Semantic legality verifier | Still essential, partly present | MIR verify/no-GC verify plus CANA legality gates should form the safety boundary. |
| Incremental compilation with content hashes | Still valid | Best if tied to existing deterministic hashes, snap/cache infrastructure, and pass-result fingerprints. |
| Hot/cold splitting | Still useful later | Needs reliable profile data first. |
| Shape-specialized compilation | Very strong | Especially for tensor ops, fusion, compression, and memory reuse. |
| Thermal-aware scheduler/runtime | Plausible but early | CANA has scaffolding. NSS needs real pressure/thermal data before this becomes trustworthy. |

## What Changed Because Of CANA

CANA changes the design from "add many compiler heuristics" to "build a verified fact pipeline and let CANA rank choices."

Based on the CANA worktree, CANA is already aimed at:

- compiler-aware feature extraction,
- cost modeling,
- pass ranking,
- pass history,
- legality gates,
- fusion candidates,
- memory proxy scoring,
- pressure-aware planning,
- trained/linear cost models,
- thermal-aware cost modeling,
- kernel variant selection,
- and a CANA-NSS bridge.

That means future infrastructure should avoid duplicating CANA. Instead, new systems should feed CANA cleaner information:

- real MIR execution profiles,
- allocation and lifetime facts,
- tensor shape facts,
- alias/effect facts,
- pass diagnostics,
- deterministic hashes,
- compression candidates,
- and verified before/after metrics.

## What Changed Because Of NSS

NSS should be treated as a pressure and state-space intelligence layer, not as the primary semantic correctness mechanism.

The current CANA-NSS bridge appears to be useful but not fully realized yet. The structural-only pressure prediction path can identify likely hot kernels from CFG structure, loop depth, and branch count, but real pressure maps appear to need deeper MIR/runtime instrumentation.

That means NSS is very promising for:

- memory pressure prediction,
- CPU pressure prediction,
- thermal pressure prediction,
- state-space risk,
- pass sequence stability,
- fusion/compression risk scoring,
- and detecting optimization choices that look locally good but globally unstable.

But NSS should not replace:

- MIR semantic verification,
- no-GC verification,
- alias analysis,
- effect analysis,
- shape legality,
- or exact reconstruction checks for compression.

## What Is Outdated In The Older Plan

The older plan is not wrong, but a few parts need reframing.

### Pass Profitability

Old framing:

> Build a deterministic pass profitability model.

Updated framing:

> CANA is the pass profitability model. Build the fact pipeline, pass diagnostics, and profile database that make CANA's rankings more trustworthy.

### NSS As Legality

Old framing implied NSS might help provide compiler legality.

Updated framing:

> NSS can score pressure, stability, and risk, but semantic legality should remain in MIR verification, no-GC verification, alias/effect analysis, shape analysis, and exact equality/reconstruction checks.

### Thermal-Aware Scheduling

Old framing made thermal-aware scheduling sound like a near-term compiler feature.

Updated framing:

> Thermal-aware scheduling is still interesting, but it should wait until NSS receives real pressure data. Otherwise the system risks making decisions from mostly structural guesses.

### Region-Based Memory Planning

Old framing sounded like CJC needed arenas from scratch.

Updated framing:

> CJC already has arena/pool/scratchpad infrastructure. The missing piece is the compiler-level lifetime plan that tells those systems what can safely share memory.

## Highest-Value Infrastructure To Add Next

### 1. Deterministic MIR Execution Profile Substrate

This is probably the biggest missing bridge.

Create a deterministic profile format that can summarize real compiler/runtime behavior without making the compiler nondeterministic.

Useful facts:

- function and MIR hashes,
- pass sequence hashes,
- basic block counts,
- branch counts,
- loop trip summaries,
- allocation-site counts,
- allocation size summaries,
- tensor temporary lifetimes,
- buffer reuse opportunities,
- deallocation/reset points,
- runtime pressure snapshots,
- pass before/after diagnostics,
- and shape-specialization keys.

This would feed:

- CANA pass ranking,
- NSS pressure prediction,
- deterministic PGO,
- hot/cold splitting,
- shape specialization,
- compression decisions,
- and regression diagnostics.

### 2. Unified Compiler Fact Database

The compiler needs a durable place to store verified facts.

This should probably combine:

- CANA pass history,
- deterministic PGO profiles,
- MIR/content hashes,
- pass diagnostics,
- shape signatures,
- alias/effect summaries,
- memory pressure summaries,
- tensor fusion outcomes,
- compression outcomes,
- and verifier results.

The key design rule: all entries should be reproducible and content-addressed.

### 3. MIR-Level Memory And Lifetime Planner

CJC already has useful runtime memory infrastructure. The missing layer is a compiler plan that decides how values should live.

A first version could track:

- allocation site,
- escape status,
- value lifetime interval,
- tensor shape,
- mutability,
- alias class,
- last use,
- safe reuse candidates,
- arena eligibility,
- tensor pool eligibility,
- and no-GC constraints.

The planner should produce conservative recommendations, not unsafe rewrites.

Example outputs:

- this temporary tensor can reuse buffer group A,
- this allocation can live in the frame arena,
- this value escapes and must not be compressed/reused aggressively,
- this intermediate can be streamed instead of materialized,
- this allocation blocks no-GC verification.

### 4. Alias, Effect, And Shape Analysis

This is the safety substrate for almost every advanced optimization.

CANA can suggest transforms, but these analyses decide whether the compiler is allowed to apply them.

Needed facts:

- may-alias / no-alias,
- read/write effects,
- mutation boundaries,
- call effects,
- tensor rank and dimension symbols,
- shape equality,
- shape compatibility,
- broadcast legality,
- layout constraints,
- and reduction-axis behavior.

These facts should feed:

- fusion,
- vectorization,
- loop transforms,
- tensor buffer reuse,
- compression,
- no-GC verification,
- and shape-specialized compilation.

### 5. SSA Optimization Maturity

Because CJC already has SSA-related infrastructure, this is less about inventing SSA and more about making it the reliable optimization backbone.

Important passes:

- SCCP,
- GVN,
- copy propagation,
- DCE,
- LICM,
- PRE,
- canonical induction variable analysis,
- bounds-check elimination,
- and reduction-aware loop transforms.

CANA can rank pass order, but MIR/SSA should provide deterministic semantics and diagnostics.

### 6. CANA-Guided Fusion And Compression

Fusion and compression should share a similar architecture.

Flow:

1. MIR/SSA/shape/alias facts identify legal candidate regions.
2. CANA ranks candidate transforms.
3. NSS scores pressure and stability impact.
4. The compiler applies only verifier-approved transforms.
5. Diagnostics record before/after size, pressure, runtime, and semantic status.

For compression specifically, the rule should be:

> compression is allowed only when reconstruction or semantic equivalence is exact enough for the compiler's contract.

Compression should reduce compiler and runtime manageability costs without silently degrading program meaning.

### 7. Shape-Specialized Compilation

Shape specialization is especially valuable for tensors and neural compiler layers.

Good candidates:

- fixed-rank tensors,
- common dimension patterns,
- known reduction axes,
- common broadcast forms,
- hot kernel shapes,
- and stable runtime shape signatures.

This should be driven by deterministic profile data and guarded by fallback generic paths.

### 8. Incremental Compilation And Content-Addressed Pass Results

Incremental compilation still sounds reasonable.

The most useful version would cache:

- parsed module hashes,
- typed/HIR/MIR hashes,
- monomorphization outputs,
- SSA form,
- verifier facts,
- analysis summaries,
- optimized MIR,
- CANA decisions,
- compression decisions,
- and generated artifacts.

The cache should invalidate based on content and compiler configuration, not timestamps alone.

### 9. Hot/Cold Splitting

This remains useful, but it should come after the profile substrate.

Hot/cold splitting can help:

- code layout,
- inlining decisions,
- specialized fast paths,
- compression decisions,
- tensor kernel selection,
- and thermal/runtime scheduling.

Without reliable profile data, this should stay conservative.

### 10. Thermal-Aware Runtime Decisions

This is still promising but should be later-stage.

The current shape suggests CANA has thermal-aware cost-model scaffolding and kernel variants, but NSS needs real pressure maps before this becomes a strong decision layer.

Best future direction:

- collect real runtime pressure summaries,
- let NSS estimate thermal/memory/CPU pressure,
- let CANA choose between byte-identical legal variants,
- and keep deterministic fallback behavior.

## Compression Network Placement

The future compression neural net should probably be CANA-adjacent rather than independent.

Possible role:

> A compression specialist that proposes exact or verifier-safe reductions of compiler state, tensor intermediates, MIR regions, pass histories, shape facts, or runtime traces.

It should not replace CANA or NSS.

Suggested split:

- CANA decides whether compression is profitable.
- NSS estimates whether compression reduces pressure or introduces instability.
- MIR/verifier/shape/equality checks decide whether compression is legal.
- The compression network proposes representations and candidate regions.

Good compression targets:

- repeated MIR subgraphs,
- repeated tensor operation patterns,
- shape signatures,
- pass history summaries,
- profile traces,
- temporary tensor materialization,
- kernel candidate sets,
- and compiler diagnostic histories.

Bad compression targets:

- semantic facts without exact reconstruction,
- verifier-critical state,
- alias/effect facts unless lossless,
- shape facts unless exact,
- user-visible behavior,
- and anything needed for deterministic replay.

## Recommended Updated Roadmap

### Phase 1: Fact Collection

- Add or mature deterministic MIR execution profiles.
- Normalize pass diagnostics.
- Store profiles using stable hashes.
- Feed profiles into CANA and NSS.

### Phase 2: Legality Foundation

- Strengthen alias analysis.
- Strengthen effect analysis.
- Strengthen shape analysis.
- Connect those facts to MIR verification and CANA legality gates.

### Phase 3: Memory Planner

- Use escape analysis, lifetimes, and tensor shapes to plan memory.
- Emit conservative arena/pool/reuse recommendations.
- Track pressure before and after each optimization.

### Phase 4: CANA/NSS Integration

- Let CANA rank passes, fusion, memory plans, and compression candidates.
- Let NSS score pressure, thermal risk, CPU pressure, and state-space stability.
- Keep compiler verifiers as the final authority.

### Phase 5: Optimization Expansion

- Mature SSA sparse optimizations.
- Add loop canonicalization and loop transforms.
- Add shape-specialized compilation.
- Add hot/cold splitting.
- Add deterministic vectorization.

### Phase 6: Compression Layer

- Introduce the compression network as a CANA-adjacent specialist.
- Start with lossless or exact-reconstruction compression.
- Apply only to low-risk compiler artifacts first, such as traces, pass histories, repeated MIR regions, and shape/profile summaries.
- Expand toward tensor/intermediate compression only when legality and reconstruction checks are strong.

## Final Opinion

The older infrastructure list still points in the right direction. The main correction is architectural:

> Do not build many separate smart compiler systems. Build deterministic compiler facts, then let CANA rank, NSS pressure-test, and verifiers approve.

The highest-value near-term infrastructure is probably:

1. deterministic MIR profile traces,
2. a unified compiler fact database,
3. alias/effect/shape analysis,
4. MIR-level memory and tensor lifetime planning,
5. CANA/NSS integration over real pressure data,
6. and only then deeper compression, vectorization, hot/cold splitting, and thermal-aware scheduling.

That keeps the compiler efficient, manageable, and intelligent without letting neural components become unsupervised semantic authorities.
