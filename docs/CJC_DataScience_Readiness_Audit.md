# CJC Data Science Readiness Audit
## "What Does CJC Need to Become a Top-Tier Data Science Language?"

**Date**: 2026-03-01
**Scope**: Full codebase audit (15 crates, ~25K LOC)
**Constraint**: All additions must preserve deterministic behavior (same input = bit-identical output)

---

## Overall Grade: **B-** (72/100)

CJC has an exceptionally strong *foundation* -- deterministic tensors, tidy data
pipelines, automatic differentiation, and a real type system -- but it's missing
the statistical and algebraic *tools* that data scientists reach for hourly.
The infrastructure is elite; the library is incomplete.

---

## Stack Role Group Assessment

We evaluate CJC from the perspective of five data science archetypes.

---

### 1. THE DATA ANALYST (pandas/R user)
**Grade: B+ (82/100)**

**What CJC does well:**
- Tidyverse-style pipeline (`filter`, `select`, `mutate`, `group_by`,
  `summarize`, `arrange`) with pipe operator `|>` -- feels native
- 7 join types (inner, left, right, full, semi, anti, cross) -- matches dplyr
- Pivot operations (longer/wider) -- present
- String manipulation (17 str_ functions) -- decent coverage
- Factor/categorical support -- present
- CSV I/O -- present
- Lazy views with bitmask (TidyView) -- memory-efficient
- Deterministic output -- huge advantage for reproducible reports
- JSON parse/emit -- present

**What's missing (Priority):**
| Gap | Impact | Difficulty |
|-----|--------|------------|
| `median()`, `quantile(p)` | P0 -- used in every EDA session | Easy |
| `variance()`, `sd()` | P0 -- basic descriptive stats | Easy |
| `n_distinct()` in grouped summarize | P1 -- count unique per group | Easy |
| `cor()` correlation matrix | P1 -- EDA essential | Medium |
| `histogram()` / `binning()` | P1 -- frequency distributions | Medium |
| `cumsum()`, `cumprod()`, `cummax()`, `cummin()` | P1 -- running aggregates | Easy |
| `lag()`, `lead()` | P1 -- time series offset | Easy |
| `rank()`, `dense_rank()`, `row_number()` | P1 -- window ranking | Medium |
| `case_when()` / vectorized if-else | P1 -- conditional column logic | Medium |
| `coalesce()` / null handling | P2 -- missing data workflows | Medium |
| Excel/Parquet I/O | P2 -- real-world data formats | Hard |

**Verdict**: CJC can already do 80% of what a data analyst needs. The missing
20% (descriptive stats, ranking, cumulative ops) blocks real EDA workflows.

---

### 2. THE STATISTICIAN (R / SAS user)
**Grade: D+ (45/100)**

**What CJC does well:**
- Numerically stable summation (Kahan + Binned) -- better than most languages
- Deterministic results -- perfect for reproducible research
- Basic reductions (sum, mean, min, max) -- correct and stable
- Window functions with Kahan summation -- present
- Seeded RNG (SplitMix64) with normal distribution -- present

**What's missing (Priority):**
| Gap | Impact | Difficulty |
|-----|--------|------------|
| `var()`, `sd()`, `se()` (standard error) | P0 -- cannot do statistics without these | Easy |
| `median()`, `quantile()`, `IQR()` | P0 -- every statistical analysis | Medium |
| `cor()`, `cov()` | P0 -- bivariate analysis | Medium |
| `t_test()`, `paired_t_test()` | P0 -- hypothesis testing | Medium |
| `p_value()`, `confidence_interval()` | P0 -- inference | Medium |
| `chi_squared_test()` | P1 -- categorical data | Medium |
| `anova()`, `f_test()` | P1 -- group comparison | Hard |
| Normal distribution CDF/PDF/PPF | P1 -- probability calculations | Medium |
| Other distributions (t, chi2, F, binomial, poisson) | P1 -- modeling | Hard |
| `lm()` -- linear regression | P0 -- the most common model | Hard |
| `glm()` -- generalized linear models | P2 -- logistic regression etc. | Hard |
| Residual analysis | P2 -- model diagnostics | Hard |
| `skewness()`, `kurtosis()` | P2 -- distribution shape | Easy |
| `z_score()`, `standardize()` | P2 -- normalization | Easy |
| Bootstrap / permutation tests | P2 -- nonparametric inference | Medium |
| Survival analysis (Kaplan-Meier) | P3 -- specialized | Hard |
| Bayesian inference primitives | P3 -- advanced | Very Hard |

**Verdict**: CJC cannot currently perform *any* classical statistical inference.
No hypothesis tests, no confidence intervals, no regression models. This is the
single largest gap for data science credibility.

---

### 3. THE ML ENGINEER (scikit-learn / PyTorch user)
**Grade: C+ (62/100)**

**What CJC does well:**
- Automatic differentiation (forward + reverse mode) -- real and working
- Dense tensor operations with matmul -- fast tiled path
- Transformer building blocks (attention, softmax, layer_norm, linear, conv1d/2d, maxpool2d) -- impressive
- LU, QR, Cholesky decompositions -- present
- Matrix inverse -- present
- Sparse tensors (CSR + COO) with matvec -- present
- Deterministic training -- massive advantage (reproducible experiments)
- bfloat16 + float16 + quantized (i8/i4) -- inference-ready
- KV-cache (Scratchpad + PagedKvCache) -- LLM inference support
- Batched matmul (bmm) -- present

**What's missing (Priority):**
| Gap | Impact | Difficulty |
|-----|--------|------------|
| **SVD** | P0 -- PCA, low-rank approx, recommender systems | Hard |
| **Eigenvalue decomposition** | P0 -- PCA, spectral methods, stability analysis | Hard |
| Determinant | P1 -- model selection, Gaussian processes | Medium |
| Solve (Ax=b) | P0 -- linear regression, optimization | Medium |
| Least squares solver | P0 -- regression, curve fitting | Medium |
| Loss functions (cross-entropy, hinge, huber) | P1 -- training | Easy |
| Optimizers (SGD, Adam, AdaGrad) | P0 -- gradient descent | Medium |
| Batch normalization | P1 -- deep learning | Easy |
| Dropout | P1 -- regularization | Easy |
| Embedding layer | P1 -- NLP | Medium |
| `argmax()`, `argmin()` | P1 -- classification output | Easy |
| `topk()` | P2 -- beam search, sampling | Medium |
| Activation functions (Sigmoid, Tanh, LeakyReLU, SiLU/Swish, Mish) | P1 -- neural nets | Easy |
| Tensor concatenation (`cat`/`stack`) | P1 -- batch assembly | Easy |
| One-hot encoding | P1 -- categorical features | Easy |
| Train/test split | P1 -- evaluation | Easy |
| Cross-validation | P2 -- model selection | Medium |
| Confusion matrix + metrics (precision, recall, F1, AUC) | P1 -- evaluation | Medium |
| Learning rate schedulers | P2 -- training | Medium |

**Verdict**: CJC has the foundation for deep learning (AD, tensor ops, transformer
blocks) but lacks the SVD/eigenvalue decompositions that underpin classical ML,
and has no optimizer or loss function library. You could *manually* build a neural
net, but it would be tedious without standard components.

---

### 4. THE NUMERICAL ANALYST (MATLAB / Julia user)
**Grade: C (58/100)**

**What CJC does well:**
- Kahan + Binned summation -- best-in-class numerical stability
- Complex arithmetic with fixed multiplication order (FMA-proof) -- excellent
- Modified Gram-Schmidt QR -- textbook-correct
- LU with partial pivoting -- correct
- Deterministic accumulation across all paths -- unique advantage
- Multiple precision types (f64, f32, f16, bf16, complex) -- good range
- Tiled matmul with L2 cache awareness -- performance-conscious

**What's missing (Priority):**
| Gap | Impact | Difficulty |
|-----|--------|------------|
| **SVD** (Golub-Kahan bidiagonalization) | P0 -- foundational decomposition | Very Hard |
| **Eigenvalues** (QR algorithm / Householder) | P0 -- foundational | Very Hard |
| **Determinant** (via LU) | P1 -- trivial to add | Easy |
| **Solve** (Ax=b via LU) | P0 -- system solving | Medium |
| **Least squares** (via QR) | P0 -- overdetermined systems | Medium |
| **Rank** (via SVD) | P1 -- depends on SVD | Easy (after SVD) |
| **Condition number** (via SVD) | P1 -- numerical diagnostics | Easy (after SVD) |
| **Schur decomposition** | P2 -- advanced | Hard |
| **Matrix exponential** | P2 -- ODEs, control theory | Hard |
| **FFT** (Fast Fourier Transform) | P1 -- signal processing, spectral analysis | Hard |
| **Sparse solvers** (CG, GMRES) | P2 -- large-scale systems | Hard |
| **ODE solver** (Runge-Kutta) | P2 -- differential equations | Medium |
| **Interpolation** (linear, cubic spline) | P2 -- data fitting | Medium |
| **Numerical integration** (quadrature) | P2 -- calculus | Medium |
| **Root finding** (Newton, bisection) | P2 -- optimization | Medium |
| **Matrix norms** (Frobenius, 1-norm, inf-norm) | P1 -- analysis | Easy |
| **Trace** | P1 -- matrix operations | Easy |
| **Kronecker product** | P2 -- tensor networks | Easy |

**Verdict**: CJC has excellent *numerics* (summation stability, complex arithmetic)
but the linear algebra library stops at QR/LU/Cholesky. Without SVD and
eigenvalues, it cannot compete with MATLAB/Julia for serious numerical work.

---

### 5. THE DATA ENGINEER (pipeline builder)
**Grade: B (78/100)**

**What CJC does well:**
- Deterministic execution -- pipelines produce identical results every run
- Module system with deterministic symbol resolution -- composable code
- File I/O builtins (read, write, exists, lines) -- basic plumbing
- JSON parse/emit with sorted keys -- interop
- CSV I/O with streaming -- data ingestion
- Effect system classifying IO/ALLOC/NONDET -- static analysis of side effects
- @nogc verification -- memory-bounded processing
- Regex engine (zero-dependency, NFA) -- text parsing
- DateTime (epoch millis, UTC, format) -- timestamps

**What's missing (Priority):**
| Gap | Impact | Difficulty |
|-----|--------|------------|
| Error handling (try/catch or Result propagation) | P1 -- pipeline robustness | Medium |
| `env()` -- environment variable access | P1 -- configuration | Easy |
| HTTP client (fetch URL) | P2 -- API ingestion | Hard |
| Process spawning / shell exec | P2 -- integration | Hard |
| Parquet/Arrow reader | P2 -- columnar data formats | Very Hard |
| TOML/YAML parsing | P2 -- config files | Medium |
| Streaming/iterator protocol | P1 -- memory-bounded ETL | Hard |
| `map_files()` / directory listing | P1 -- batch processing | Easy |
| Logging framework | P2 -- observability | Medium |
| Retry/timeout utilities | P2 -- resilience | Medium |
| Schema validation | P2 -- data quality | Medium |
| Checksumming (SHA-256, CRC32) | P2 -- data integrity | Medium |

**Verdict**: CJC can build basic deterministic pipelines today. The module system,
file I/O, JSON, and CSV cover many use cases. Missing streaming/iterator support
and richer I/O formats (Parquet) limits large-scale data engineering.

---

## Consolidated Priority Matrix

### Tier 0: "Cannot Be Taken Seriously Without These" (Must-have)

| Feature | Blocks | Effort | Determinism Risk |
|---------|--------|--------|-----------------|
| `var()`, `sd()`, `se()` | Statistician, Analyst | 1 day | None -- Kahan stable |
| `median()`, `quantile(p)` | Statistician, Analyst | 2 days | None -- sorting is deterministic |
| `cor()`, `cov()` | Statistician, Analyst, ML | 2 days | None -- Kahan stable |
| SVD decomposition | ML, Numerical, Statistician | 1-2 weeks | Medium -- iteration order matters |
| Eigenvalue decomposition | ML, Numerical, Statistician | 1-2 weeks | Medium -- same concern |
| Solve (Ax=b) | ML, Numerical | 2 days | None -- LU already exists |
| Determinant (det) | ML, Numerical | 1 day | None -- product of LU diagonal |
| `t_test()`, `p_value()` | Statistician | 3 days | Low -- needs normal CDF |
| Normal CDF/PDF/PPF | Statistician, ML | 3 days | Low -- deterministic algorithms exist |
| `lm()` linear regression | Statistician, ML | 3 days | Low -- needs Solve + QR |
| `argmax()`, `argmin()` | ML | 0.5 day | None |

### Tier 1: "Competitive Advantage" (Should-have)

| Feature | Blocks | Effort | Determinism Risk |
|---------|--------|--------|-----------------|
| Optimizers (SGD, Adam) | ML | 3 days | None -- sequential updates |
| Loss functions (CE, Huber) | ML | 1 day | None |
| `cumsum()`, `cumprod()` | Analyst | 1 day | None -- Kahan stable |
| `lag()`, `lead()` | Analyst | 1 day | None |
| `rank()`, `row_number()` | Analyst | 2 days | None -- stable sort |
| Activation functions (sigmoid, tanh, SiLU) | ML | 0.5 day | None |
| Tensor `cat()`/`stack()` | ML | 1 day | None |
| One-hot encoding | ML | 0.5 day | None |
| Confusion matrix / metrics | ML | 2 days | None |
| Matrix norms / trace | Numerical | 0.5 day | None |
| `case_when()` | Analyst | 1 day | None |
| Chi-squared test | Statistician | 2 days | Low |
| t/F/chi2 distributions | Statistician | 1 week | Low |
| FFT | Numerical | 1 week | Low -- Cooley-Tukey deterministic |
| Least squares solver | ML, Numerical | 2 days | None |
| `histogram()` / binning | Analyst | 1 day | None |

### Tier 2: "Nice to Have" (Could-have)

| Feature | Blocks | Effort |
|---------|--------|--------|
| ANOVA / F-test | Statistician | 3 days |
| GLM (logistic regression) | ML | 1 week |
| Parquet I/O | Data Engineer | 2 weeks |
| ODE solver (RK4/RK45) | Numerical | 1 week |
| Interpolation (spline) | Numerical | 1 week |
| Sparse solvers (CG, GMRES) | Numerical | 2 weeks |
| Cross-validation | ML | 3 days |
| LR schedulers | ML | 2 days |
| Bootstrap / permutation tests | Statistician | 1 week |
| Skewness / kurtosis | Statistician | 0.5 day |
| Z-score / standardize | Analyst | 0.5 day |
| Streaming iterator protocol | Data Engineer | 2 weeks |
| HTTP client | Data Engineer | 2 weeks |

---

## What CJC Is *Uniquely Good At*

These are genuine competitive advantages -- not just "has the feature" but
"does it better than alternatives":

1. **Deterministic-by-default execution**: No language in the data science
   space guarantees bit-identical results across runs. CJC does. This is
   transformative for regulated industries (finance, pharma, aviation).

2. **Binned accumulation**: Order-invariant floating-point summation that
   gives identical results regardless of thread scheduling or data order.
   No other data science language has this built in at the kernel level.

3. **@nogc static verification**: Prove at compile time that a function
   will never trigger garbage collection. Critical for latency-sensitive
   inference paths. Unique among high-level languages.

4. **Effect system**: Every builtin is classified (PURE, IO, ALLOC, GC,
   NONDET, MUTATES, ARENA_OK, CAPTURES). This enables static analysis
   that no Python/R/Julia library can match.

5. **Multi-precision pipeline**: f64 -> f32 -> bf16 -> f16 -> i8 -> i4
   all in one language with deterministic dequantization. No other data
   science language covers this range natively.

6. **Tidy + Tensor in one language**: R has tidyverse but weak tensors.
   Python has NumPy but clunky data frames. CJC has both with a unified
   type system and pipe operator.

7. **Zero-dependency regex**: Thompson NFA in the runtime, NoGC-safe,
   no backtracking. Usable in @nogc hot paths.

8. **Paged KV-Cache**: vLLM-style block-paged attention cache built into
   the runtime. No other language has this as a first-class primitive.

---

## What CJC Needs to Improve (Design/Ergonomics)

Beyond missing features, these *design* issues affect usability:

1. **Error propagation**: The `?` operator exists in the parser but error
   handling in practice is limited. Data science code needs robust
   try/recover patterns for messy real-world data.

2. **REPL / notebook experience**: No interactive evaluation mode exists.
   Data science is fundamentally interactive -- explore, visualize, iterate.
   A REPL or Jupyter kernel would transform adoption.

3. **Plotting / visualization**: Zero plotting capability. Even a simple
   text-based histogram or ASCII chart would help. A plotting DSL that
   emits SVG/PNG would be transformative.

4. **Documentation / examples**: No stdlib documentation, no tutorial,
   no cookbook. Data scientists choose tools by example quality.

5. **Package manager / ecosystem**: No way to share CJC libraries. The
   module system exists but there's no registry or dependency resolution.

6. **Debugging story**: No debugger, no profiler, no step-through.
   `print()` is the only debugging tool.

7. **Type inference verbosity**: Function signatures require full type
   annotations. Data scientists expect type inference (like Julia or
   modern Python with mypy).

8. **Broadcasting semantics**: Tensor broadcasting exists but the rules
   aren't documented. NumPy-style broadcasting rules should be explicit.

---

## Recommended Implementation Order

If building toward "top-tier data science language" while maintaining
determinism, here's the sequencing:

**Sprint 1 (1 week): Descriptive Statistics**
- `var()`, `sd()`, `se()` with Welford's online algorithm
- `median()`, `quantile()` with deterministic nth_element
- `skewness()`, `kurtosis()` (moment-based)
- `z_score()`, `standardize()`
- Wire into both executors + effect registry

**Sprint 2 (1 week): Correlation + Simple Inference**
- `cor()`, `cov()` matrices (Kahan-stable)
- Normal distribution CDF/PDF/PPF (Abramowitz & Stegun approximation)
- `t_test()`, `paired_t_test()` (needs t-distribution CDF)
- `p_value()`, `confidence_interval()`
- `chi_squared_test()` (needs chi2 CDF)

**Sprint 3 (2 weeks): Core Linear Algebra**
- SVD via Golub-Kahan bidiagonalization (deterministic iteration)
- Eigenvalues via implicit QR shifts (Wilkinson shift for determinism)
- `det()` (via LU -- trivial)
- `solve()` (Ax=b via LU -- mostly done)
- `lstsq()` (via QR -- straightforward)
- `rank()`, `cond()` (via SVD)
- Matrix norms, trace, Kronecker product

**Sprint 4 (1 week): ML Toolkit Foundation**
- Loss functions: MSE, cross-entropy, hinge, huber
- Optimizers: SGD, Adam (sequential updates -- deterministic)
- `argmax()`, `argmin()`, `topk()`
- Activation functions: sigmoid, tanh, SiLU, LeakyReLU, Mish
- Tensor `cat()`, `stack()`, `split()`
- One-hot encoding, label encoding

**Sprint 5 (1 week): Analyst Quality-of-Life**
- `cumsum()`, `cumprod()`, `cummax()`, `cummin()`
- `lag()`, `lead()` in tidy pipeline
- `rank()`, `dense_rank()`, `row_number()`
- `case_when()` vectorized conditional
- `histogram()` binning function
- `lm()` linear regression (via QR least squares)

**Sprint 6 (2 weeks): Advanced**
- FFT (Cooley-Tukey, power-of-2 + Bluestein for arbitrary)
- t/F/chi2/binomial/poisson distributions
- ANOVA, F-test
- Cross-validation utilities
- Confusion matrix, precision, recall, F1, AUC-ROC

---

## Final Scorecard

| Category | Score | Grade | Notes |
|----------|-------|-------|-------|
| Core Infrastructure | 95/100 | A | Determinism, memory model, type system |
| Tensor Operations | 85/100 | A- | Matmul, attention, conv, pool all present |
| Data Manipulation | 82/100 | B+ | Tidy pipeline excellent, missing cumulative ops |
| Automatic Differentiation | 78/100 | B | Forward+reverse present, missing higher-order |
| Linear Algebra | 45/100 | D+ | LU/QR/Cholesky only -- no SVD or eigenvalues |
| Descriptive Statistics | 30/100 | F | Only sum/mean/min/max -- no var/median/quantile |
| Inferential Statistics | 5/100 | F | Nothing implemented |
| ML Toolkit | 40/100 | D | AD exists but no optimizers, losses, or metrics |
| Probability/Distributions | 0/100 | F | Nothing implemented |
| Visualization | 0/100 | F | Nothing implemented |
| I/O & Interop | 60/100 | C | CSV+JSON+File, no Parquet/Excel/HTTP |
| Developer Experience | 50/100 | C- | No REPL, no debugger, limited docs |
| **Weighted Average** | **72/100** | **B-** | **Strong foundation, incomplete library** |

---

## The Bottom Line

CJC is an **A-tier foundation** trying to become an **A-tier language**.
The deterministic execution model, effect system, and @nogc verification
are genuinely novel -- no competitor has these. The tidy pipeline + tensor
combination is elegant. The memory model is sophisticated.

But a data scientist who opens CJC today cannot compute a standard
deviation, cannot run a t-test, cannot perform PCA, and cannot fit a
linear regression. These are table-stakes operations that every competing
language (R, Python, Julia, MATLAB) has had for decades.

The good news: CJC's architecture *supports* all of these additions cleanly.
The Kahan/Binned summation infrastructure means variance/covariance can be
implemented with best-in-class stability. The effect system means every new
builtin gets proper classification automatically. The @nogc verifier means
pure statistical functions are immediately available in GC-free hot paths.

**CJC is 6-8 weeks of focused library work away from being genuinely
competitive.** The hard part (the infrastructure) is done. What remains
is "filling in the spreadsheet" -- important but well-understood work.
