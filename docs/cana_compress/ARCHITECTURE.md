# CANA Compression Layer — Architecture Diagram

## End-to-end data flow

```text
┌──────────────────────────────────────────────────────────────────────┐
│  Compiler / Runtime emit advisory CANA artefacts:                    │
│    • pass histories (PassHistory, lossless-critical)                 │
│    • feature vectors (advisory-only, lossy ok)                       │
│    • MIR motifs (lossless-critical)                                  │
│    • runtime pressure trajectories (advisory-only)                   │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cjc_cana_compress::CompressionCandidate                             │
│  ─────────────────────────────────────────                           │
│    + id: CandidateId                                                 │
│    + kind: CompressionKind                                           │
│        LosslessTrace | MotifDictionary                               │
│        LowRankAdvisory | TensorTrainAdvisory                         │
│    + criticality: Criticality                                        │
│        SemanticCritical | AdvisoryOnly { tolerance_f }               │
│    + payload: Vec<u8>                                                │
│    + label: String                                                   │
│                                                                       │
│  ⛔ HARD RULE: SemanticCritical + lossy kind → Err at construction   │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cjc_cana_compress::CompressionPlan                                  │
│  ──────────────────────────────────                                  │
│    • sorted by (CandidateId, canonical_hash)                         │
│    • assigns deterministic slot indices                              │
│    • content-addressed plan_hash (FNV-1a over canonical bytes)       │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ execute()
┌──────────────────────────────────────────────────────────────────────┐
│  Per-entry kind dispatch                                              │
│  ───────────────────────                                              │
│                                                                       │
│  LosslessTrace ──► lossless_compress_bytes ──► RLE codec (CLT0)      │
│                   verify input_hash == reconstructed_hash             │
│                                                                       │
│  MotifDictionary ─► compress_motif_dictionary ─► LZ77-style (CMD0)   │
│                   verify input_hash == reconstructed_hash             │
│                                                                       │
│  LowRankAdvisory ─► decode (CLRP) ─► power-iteration truncated SVD   │
│                   compare frobenius_error vs declared tolerance_f     │
│                                                                       │
│  TensorTrainAdvisory ─► decode (CTTP) ─► cjc_quantum::mps SVD chain  │
│                   compare frobenius_error vs declared tolerance_f     │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cjc_cana_compress::CompressionReport                                │
│  ──────────────────────────────────                                  │
│    • per-entry status: Validated |                                   │
│                        MalformedRoundTrip |                          │
│                        ToleranceExceeded |                           │
│                        DecodeFailed                                  │
│    • canonical bytes (CCR0 magic + per-entry fields)                 │
│    • content-addressed report_hash (FNV-1a)                          │
│    • JSON sidecar for inspection                                     │
└──────────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴────────────────────┐
            │                                       │
            ▼                                       ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│  cjc_cana_compress::bridge      │  │  cjc_cana_compress::EnergyRanker│
│  ──────────────────────────     │  │  ────────────────────────────── │
│  compression_pressure_delta(    │  │  rank(Vec<(id, EnergyComponents)│
│    baseline_density,            │  │       )) → EnergyRanking        │
│    report,                      │  │                                  │
│    BridgeCoefficients,          │  │  Ising-style objective:          │
│  ) → CompressionPressureDelta { │  │    + runtime_cost                │
│    updated: PressureDensityState│  │    + memory_pressure             │
│    summary: PressureCorrelation │  │    + thermal_pressure            │
│             Summary,            │  │    + code_size_cost              │
│    delta_memory,                │  │    + reconstruction_risk         │
│    delta_thermal,               │  │    + verifier_risk_penalty       │
│    delta_throughput,            │  │    - fusion_reward               │
│    rewarded_entries,            │  │    - reuse_reward                │
│    penalised_entries,           │  │    - compression_reward          │
│  }                              │  │                                  │
│                                 │  │  Sort: (total ASC, id ASC).      │
│  Defaults:                      │  │  Stable tie-break.               │
│   memory_reward_scale = 0.6     │  │  Drop non-finite → metadata.     │
│   throughput_reward = 0.02      │  │  Decomposition exposed:          │
│   thermal_advisory = 0.1        │  │    score.components fully        │
│   memory_malformed = 0.1        │  │    auditable.                    │
│   thermal_exceeded = 0.2        │  │                                  │
└─────────────────────────────────┘  └─────────────────────────────────┘
            │                                       │
            └─────────────────┬───────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cjc_nss::PressureCorrelationSummary  ◄───────  decisions advisor    │
│  ─────────────────────────────────                                   │
│    • saturation_score ∈ [0, 1]                                       │
│    • collapse_risk ∈ [0, 1]                                          │
│    • dominant_coupling: Option<(src, dst, ρ)>                        │
│    • dominant_kind_for_risk: Option<PressureKind>                    │
│    • stable_hash (FNV-1a over canonical bytes)                       │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │   advisory only — never authoritative
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ⚖️  cjc_cana::LegalityGate / cjc_mir::verifier  ⚖️                  │
│  ───────────────────────────────────────────────                     │
│  • Final authority on MIR shape, semantic correctness, no-GC, etc.   │
│  • Reads the compression report as advice; can ignore it entirely.   │
│  • Approves or rejects compiler plan; ranker output never overrides. │
└──────────────────────────────────────────────────────────────────────┘
```

## Crate dependency graph

```text
                       ┌──────────────┐
                       │  cjc-runtime │
                       │  (complex,   │
                       │   buffer)    │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐         ┌─────────┐
                       │  cjc-quantum │         │ cjc-repro│
                       │  (MPS / SVD) │         │ (Kahan,  │
                       └──────┬───────┘         │  SplitMix)│
                              │                 └────┬─────┘
                              │                      │
       ┌────────────┐         │                      │
       │  cjc-mir   │         │                      │
       │  (MirProg) │         │                      │
       └─────┬──────┘         │                      │
             │                │                      │
       ┌─────▼──────┐         │                      │
       │  cjc-cana  │         │                      │
       │  (features,│         │                      │
       │   hasher,  │         │                      │
       │   legality)│         │                      │
       └─────┬──────┘         │                      │
             │                │                      │
       ┌─────▼──────┐         │                      │
       │  cjc-nss   │         │                      │
       │  (pressure)│◄────────│ ─────────────────────┤
       │            │         │ density module uses  │
       │            │         │ Kahan                │
       └─────┬──────┘         │                      │
             │                │                      │
             │   ┌────────────┴──────────────────────┴────┐
             │   │  cjc-cana-compress (NEW, Phase 6)       │
             └──►│  ───────────────────────────────────    │
                 │  candidate / plan / report              │
                 │  lossless_trace / motif_dictionary      │
                 │  lowrank / tensor_train                 │
                 │  energy / bridge                        │
                 └─────────────────────────────────────────┘
```

Key design choices visible in this graph:

- **`cjc-cana-compress` is a satellite crate**, not a module inside
  `cjc-cana`. This keeps the `cjc-quantum` dependency out of
  `cjc-cana`'s tree — which is consumed by the compiler driver — so
  the compression layer only loads when a caller explicitly opts in.
  Same isolation pattern as
  [`cjc-cana-nss`](../../crates/cjc-cana-nss/Cargo.toml) and
  [`cjc-ad::dispatch`](../../crates/cjc-ad/src/dispatch.rs).
- **The density module lives inside `cjc-nss`**, not in a separate
  crate, because pressure primitives are the core abstraction NSS
  already exports — splitting them out would fragment an already-
  cohesive substrate.
- **No arrow from `cjc-cana-compress` to `cjc-mir-exec`**: by
  construction, the compression layer has no path to mutate executed
  MIR. The architecture *enforces* the "advisory only" property.
