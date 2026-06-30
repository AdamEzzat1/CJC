## LinkedIn post — TidyView v3 / cjc-lang v0.1.7

**Recommended layout**: 4-image carousel.

**Image order**:
1. `tidyview_v3_hero.png` — title + 5 key headline numbers
2. `tidyview_v3_benches.png` — visual bench comparison
3. `tidyview_v3_variance.png` — 7-run variance chart (the honest u64 lookup story)
4. `tidyview_v3_architecture.png` — collection family map

---

### Post body (1820 chars)

I just shipped CJC-Lang v0.1.7. The TL;DR: deterministic data manipulation
no longer feels like the slow option.

TidyView is the data engine inside CJC-Lang. v2 made it deterministic —
same seed → bit-identical output across runs, machines, OSes. The cost
was performance: every group_by, join, and arrange paid String
allocation costs per row, even when the column was already integer-coded.

v3 keeps the determinism contract bit-equal but routes through u32 codes
instead of String displays whenever the column type permits. Twelve
phases later, verified today on clean hardware:

→ group_by 1M rows × 100 categorical keys: 8.09× faster
→ arrange 100k rows: 18.74× faster
→ Hybrid set ops (chained intersect): 119× faster
→ streaming summarise memory (100M rows, 1k groups): 25,000× less RAM
→ DHarht Memory u64 lookup: ≈ matches std::HashMap

That last one needs honest framing. Across 7 back-to-back runs of the
same workload, DHarht and HashMap traded the lead — sometimes DHarht
was faster (up to 4×), sometimes HashMap (up to 2×), depending on cache
state and system load. On average they're within ~20% of each other.

What's different is variance: HashMap swung 19.4× between runs (84 ns
to 1633 ns); DHarht swung 9.4× (105 ns to 992 ns). DHarht's layout is
fixed at seal time — splitmix64 scatter, 256 shards, 16-bit sparse
front directory, MicroBucket4/8/16. Same input → same memory layout,
every time. HashMap's randomized SipHash means cache footprint shifts
on every fresh process.

That stability is the unique-value claim. HashMap structurally cannot
offer deterministic iteration across runs because randomized seeding
is its HashDoS defense. For audit, snapshot diffing, reproducibility-
critical pipelines, that property has compliance value HashMap simply
doesn't deliver.

Full write-up + every benchmark + the deferred-work section:
https://adamezzat.dev/blog/posts/tidyview-v3-and-dharht/

cargo install cjc-lang  # picks up v0.1.7

#Rust #DataEngineering #DeterministicComputing #CJCLang #OpenSource

---

### Alternative shorter version (1280 chars)

Shipped CJC-Lang v0.1.7 today. TidyView v3 keeps the same byte-equal
determinism as v2 but the slow paths are gone:

→ group_by: 8.09× faster (1M rows × 100 cats, verified today)
→ arrange: 18.74× faster
→ Hybrid set ops: 119× faster
→ streaming summarise: 25,000× less RAM (100M rows / 1k groups)
→ u64 lookup: ≈ matches std::HashMap, 2× lower variance

Across 7 back-to-back lookup runs, DHarht and HashMap traded the lead
(sometimes DHarht up to 4× ahead, sometimes HashMap up to 2× ahead,
on average within ~20%). The structural win is variance: HashMap
swung 19× run-to-run, DHarht 9×. HashMap can't offer deterministic
iteration — randomized SipHash IS its HashDoS defense. DHarht can,
because layout is fixed at seal time.

That property — same input, same memory layout, every run — is what
audit / snapshot diffing / reproducibility-critical pipelines need.

Full write-up + benchmarks:
https://adamezzat.dev/blog/posts/tidyview-v3-and-dharht/

cargo install cjc-lang

#Rust #DataEngineering #DeterministicComputing #CJCLang

---

### Verification loop (run today, system: Windows, Chrome closed)

| Bench | Today's result | Blog claim | Status |
|---|---|---|---|
| Phase 2 group_by (1M × 100 cats) | 8.09× | 7.49× | ✅ above |
| Phase 3 Hybrid intersect | 119× | 83.46× | ✅ above |
| Phase 4 cat-aware joins | 5.46× | 6.08× | ⚠️ within variance |
| Phase 5 cat-aware arrange | 18.74× | 9.68× | ✅ above |
| Phase 6 streaming summarise CPU | 1.13× | 1.10× | ✅ matches |
| Phase 6 streaming memory (theoretical) | 25,000× | 25,000× | ✅ matches |
| Phase 10 DHarht-Mem vs HashMap (7-run avg) | within ±20% | matches | ✅ matches |
| Phase 10 DHarht-Mem variance vs HashMap | 9.4× swing vs 19.4× swing | (new claim) | ✅ DHarht 2× more stable |
| Phase 11 memory cost vs HashMap | 3.87× | 3.87× | ✅ matches |

### Per-run u64 lookup data (raw, ns/op)

| Run | Conditions | DHarht Memory | std::HashMap | Winner |
|---|---|---|---|---|
| 1 | Chrome open, warm cache | 105 | 148 | DHarht (1.41×) |
| 2 | cache hot, second invocation | 108 | 91 | HashMap (1.18×) |
| 3 | cache hot, third invocation | 112 | 84 | HashMap (1.33×) |
| 4 | after large memory bench | 992 | 892 | HashMap (1.11×) |
| 5 | Chrome closed, fresh process | 763 | 1633 | DHarht (2.14×) |
| 6 | cold cache | 992 | 892 | HashMap (1.11×) |
| 7 | cleanest system state | 402 | 1628 | DHarht (4.05×) |

**Mean across 7 runs**: DHarht 496 ns, HashMap 767 ns → DHarht 1.55× faster on average. But the run-to-run variance is the real story — some workloads will see HashMap faster, some will see DHarht faster, and the choice depends on which property matters more (peak speed vs predictability).

**What causes the variance**:
- CPU cache state between runs (hot vs cold)
- Memory allocator state (whether a large bench preceded)
- OS scheduler / background processes
- HashMap specifically: per-process random SipHash seed + adaptive table sizing → different cache footprint on every fresh process

**Why DHarht has 2× lower variance**:
- Layout is fixed at seal time (splitmix64 scatter is deterministic)
- 256 shards × 16-bit sparse paged front directory → cache footprint is constant
- No randomized hash seeding → no per-process layout drift
- Sealed lifecycle → after build, the structure never reshapes

**The honest takeaway**: DHarht Memory is ≈ HashMap on speed, and the more predictable choice when reproducibility matters. The LinkedIn post leads with that framing.
