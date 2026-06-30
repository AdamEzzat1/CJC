## LinkedIn post — ABNG Phase 0.5 demos

**Recommended layout**: text-only post, OR single image of the architecture diagram from the blog post.

---

### Post body (~2150 chars)

I just shipped 11 demos for ABNG — an experimental Adaptive Belief
Network Graph in CJC-Lang.

ABNG is a Bayesian ML primitive that combines codebook-routed BLR
posteriors with an SHA-256 audit chain. Every observation, every
weight update, every structural decision lives in a tamper-evident
hash chain — the kind of cryptographic provenance that regulated
ML training (pharma, finance, healthcare) needs but doesn't have
today.

Eleven distinct capabilities, each with a working demo:

→ Per-state Bayesian uncertainty (PINN heat equation w/ closed-form solution)
→ GP-like scaling at O(N·d²) instead of classical GP's O(N³)
→ Cryptographic dataset attestation (3 independent SHA-256 divergences on tampering)
→ OOD detection composite (density + epistemic + prefix-distance)
→ Adaptive structural triggers (graph topology adapts to workload)
→ Calibration / ECE (well-calibrated < 0.05; flipped outcomes > 0.5)
→ Distribution-drift detection
→ Log compaction with hash-chain integrity
→ Maturity observability (training-state inspection)
→ End-to-end determinism (AST↔MIR byte-equality across two executors)
→ Audit chain integrity (universal tamper-evidence)

960 tests passing across 13 test suites. Zero failures.

Now the honest framing. Every demo uses small datasets — between
3 and 200 training samples per scenario. The choice was deliberate:
prove each capability is real (not vacuous), under enough data to
make the assertion meaningful, in workloads small enough to run in
milliseconds and produce locked SHA-256 canaries.

What's NOT yet validated:

✗ Performance at n > 10⁴ (no wall-clock comparison vs sklearn / GPy)
✗ Noisy real-world data (clean synthetic only)
✗ Cross-platform determinism (canaries Windows-only)
✗ Real classifier calibration (synthetic (p, y) injection)
✗ Adaptive triggers — only Merge demonstrated in a workload (5 of 6 trigger types still unfired in demos)

So this is "proven at small scale, theoretically expected to hold
at larger scale, empirically unverified at production scale." Not
"production-ready Bayesian ML toolkit." Not yet.

Phase 0.6 closes these gaps:

- Each Phase 0.5 demo gets a `_scaled` sibling: 10⁴+ samples, additive
  Gaussian noise, real classifier outputs (not synthetic)
- GitHub Actions CI on Linux + macOS + Windows gating the canaries
- Wall-clock benchmarks vs `sklearn.gaussian_process.GaussianProcessRegressor`
  at n ∈ {10³, 10⁴, 10⁵}
- All 6 trigger types fired in CJC-Lang demos (Grow, Split, Prune,
  Compress, Freeze — Merge already done)
- Native batch_observe + bulk BLR update (forces a v13 wire-format bump)

Full technical writeup with math, code sketches, and ML-literature
comparisons (BART, Mondrian Forests, Mahalanobis OOD, in-toto attestations,
PSI/ADWIN, ECE methods, the reproducibility-crisis literature):

https://adamezzat.dev/blog/posts/abng-experimental-bayesian-graph/

#MachineLearning #BayesianML #MLProvenance #Determinism #CJCLang #ExperimentalML

---

### Alternative shorter version (~1180 chars)

I just shipped 11 demos for ABNG — an experimental Bayesian ML
primitive in CJC-Lang.

ABNG combines codebook-routed BLR posteriors with an SHA-256 audit
chain. Every observation lives in a tamper-evident hash chain —
the cryptographic provenance regulated ML training needs but doesn't
have today.

What the demos prove:

→ Per-state Bayesian uncertainty (PINN heat equation)
→ GP-like scaling without O(n³) cost
→ Cryptographic dataset attestation (3 independent SHA-256 signals)
→ OOD detection composite
→ Adaptive structural triggers
→ Calibration / drift / log compaction / maturity observability
→ End-to-end byte-equal determinism across two executors

960 tests passing. Zero failures.

Honest framing: every demo uses 3 to 200 training samples. Small
enough to run in milliseconds and lock SHA-256 canaries. NOT yet
validated at scale (n > 10⁴), under noise, or cross-platform.

This is "proven at small scale, theoretically expected to hold at
larger scale, empirically unverified at production scale."

Phase 0.6 closes the gaps: scaled demos with noise, GitHub Actions
CI on three OSes, sklearn-GP wall-clock benchmarks, all 6 trigger
types fired.

Full writeup (math + code + lit comparisons):
https://adamezzat.dev/blog/posts/abng-experimental-bayesian-graph/

#MachineLearning #BayesianML #MLProvenance #Determinism #CJCLang
