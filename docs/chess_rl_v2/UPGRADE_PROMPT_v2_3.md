---
title: CJC-Lang Chess RL v2.3 — Profiling + Native Rollout Kernel
status: DRAFT — awaiting approval before execution
scope: Tier 2 profiling infrastructure + Tier 3 native hot-path builtins
constraint: Zero external dependencies. New CJC-Lang builtins REQUIRED and welcome.
predecessor: docs/chess_rl_v2/UPGRADE_PROMPT_v2_2.md (Tier 1, all 5 gates missed)
---

# CJC-Lang Chess RL v2.3 — Profiling + Native Rollout Kernel

## CONTEXT

The v2.2 Phase D run (see `docs/chess_rl_v2/PHASE_D_v2_2.md`) missed
every Tier 1 gate:

- 73 min wall clock (gate ≤ 45 min) — **1.6× over**
- 0 W / 20 D / 0 L vs random (gate ≥ 60%) — **50% WR**
- 0 W / 9 D / 1 L vs greedy (gate ≥ 55%) — **45% WR**
- Elo gain +0 (gate ≥ +25)
- 2 / 60 non-zero terminal rewards (gate ≥ 20 / 60)

The per-episode cost scaled nearly linearly with `max_moves`, which
means the interpreter has **no constant-factor overhead to amortize** —
every ply pays the same dispatch + allocation cost. The 170× gap between
CJC-Lang and an equivalent native Rust rollout is the real bottleneck.
Tier 2 (profiling) and Tier 3 (native kernels) are the right next
investment.

---

## ROLE

You are a stacked systems team continuing Chess RL v2.x work. Your
composition:

1. **Profiling Lead** — owns Tier 2 measurement infrastructure. Picks
   the right counters, avoids profiling-induced determinism
   contamination.
2. **Runtime Systems Engineer** — owns Tier 3 native builtin
   implementation in `cjc-runtime`. Hot-path kernels.
3. **Numerical Computing Engineer** — owns tensor op correctness,
   determinism of the native forward pass, and ensuring the new
   kernels produce bit-identical outputs vs the existing pure-CJC-Lang
   path.
4. **Compiler Pipeline Engineer** — owns wiring new builtins through
   Lexer → Parser → AST → HIR → MIR → Exec, and the three-place
   registration pattern.
5. **Determinism & Parity Auditor** — enforces byte-identical
   cross-executor output on every new path, enforces that profiling
   counters never feed back into program state.
6. **QA Automation Engineer** — proptest + bolero + regression sweep
   on every new builtin. Mandatory.
7. **ML Systems Lead** — gate-keeps the re-run of Phase D once Tier 3
   lands. Still the honest-reporting regime.

---

## PRIME DIRECTIVES (non-negotiable)

1. **Zero external dependencies.** Do not add new Cargo crates. Do not
   call out to external tools or libraries. No Python, no subprocess,
   no network, no new system BLAS. New CJC-Lang builtins in
   `cjc-runtime` are explicitly permitted and expected — that is the
   whole point of this upgrade. Each new builtin must:
   - (a) Preserve determinism (same seed → bit-identical output)
   - (b) Produce byte-identical results between `cjc-eval` and
     `cjc-mir-exec` on every unit + parity test
   - (c) Ship with proptest (≥1 property) and bolero fuzz (≥1 target)
   - (d) Be wired in all three places: `cjc-runtime/src/builtins.rs`,
     `cjc-eval/src/lib.rs`, `cjc-mir-exec/src/lib.rs`
   - (e) Have a documented determinism story in its doc-comment
2. **Do not break the compiler pipeline.**
   ```
   Lexer → Parser → AST → [TypeChecker] → HIR → MIR → [Optimize] → Exec
   ```
3. **Scope discipline.** This upgrade is *only* about profiling + hot-path
   native kernels for the Chess RL v2.x demo. Do not refactor unrelated
   crates. Do not rename existing builtins. Do not touch v1, v2.1, or
   v2.2 code paths except to add *new* parallel entry points.
4. **Minimal surface area.** A new builtin earns its place only if it
   either (a) measures something unmeasurable from pure CJC-Lang
   (profile counters) or (b) delivers a measured ≥3× speedup on the
   hot path identified by Tier 2.
5. **Mandatory regression after every change.**
   - After each new builtin: run `cargo test --workspace --release`.
     Must pass at v2.2 count plus new tests. Zero regressions.
   - After each PRELUDE change: run `cargo test --test test_chess_rl_v2
     --release`. All 85 existing tests must still pass.
6. **No cherry-picking.** If Tier 3 does not deliver its target speedup
   on the first honest measurement, report the number and explain the
   gap. No tuning-until-it-looks-good.
7. **Both executors must agree.** Every new builtin must work in
   `cjc-eval` AND `cjc-mir-exec` with identical semantics.
8. **Determinism is sacred.** Profile counters are a *write-only*
   sink. They must NOT feed back into program state, weight hashes,
   RNG draws, or control flow. The weight-hash fingerprint of a Tier 2
   instrumented run must match the weight-hash of the uninstrumented
   baseline.

---

## TIER 2 — PROFILING INFRASTRUCTURE

### T2 Goals

Measure the hot path. Find where ≥ 80% of per-episode wall clock is
spent. Publish a CSV breakdown. No ML gate on Tier 2; it is a
measurement step that unlocks Tier 3.

### T2-a. New builtin: `profile_zone_start(name: String) -> i64`

Returns a zone handle (monotonically increasing i64). Captures a
`std::time::Instant` internally. Name is interned in a `BTreeMap<String,
ZoneStats>` for deterministic iteration.

**Determinism story:** the counter state is a thread-local `RefCell`
that is never observed by program logic. `profile_zone_start` returns
the handle, but the handle is an opaque i64 used only as a token for
`profile_zone_stop`. Program logic cannot observe the *value* of the
counter, so profile-instrumented runs are behaviorally identical to
uninstrumented runs at the level of tensor math and RNG.

**Parity story:** both executors delegate to the same `cjc-runtime`
implementation. Calls return the same handle sequence on both executors
(since they are deterministic calls to the same runtime state).

### T2-b. New builtin: `profile_zone_stop(handle: i64) -> f64`

Captures the elapsed duration for the zone, updates
`ZoneStats { count, total_ns, min_ns, max_ns, sum_ns, sum_sq_ns }`.
Returns the elapsed duration in seconds (f64) — but **the caller MUST
NOT use this value for program logic**. The guideline is enforced by
convention and by the test suite: a dedicated test asserts that if
a program calls `profile_zone_stop` but ignores the return value, the
final weight hash is identical to an uninstrumented run.

### T2-c. New builtin: `profile_dump(path: String) -> i64`

Writes a CSV to `path`:

```
zone_name,count,total_ns,min_ns,max_ns,mean_ns,stddev_ns
```

Sorted by `total_ns` descending so the hot zones are at the top.
Resets the counter state after writing.

### T2-d. Instrument the Chess RL v2.2 hot path

Wrap the following zones in `rollout_episode_v22`:

- `rollout_total` — whole episode
- `score_moves` — the per-step policy+value forward pass
- `compute_logsumexp` — the policy log-prob chain
- `a2c_update` — the backward pass + Adam step
- `apply_move` — the move application
- `legal_moves` — the move generator

### T2-e. Tests for profiling

- **Unit tests (≥5):** handle monotonicity, nested zones, write-only
  determinism, CSV format, reset-on-dump.
- **Proptest (≥1):** for any sequence of `profile_zone_start/stop`
  calls, the CSV output is well-formed (proper columns, no NaN, no
  negative counts).
- **Bolero fuzz (≥1):** random byte inputs as zone names do not crash
  the profiler, and produce deterministic CSV output on replay.
- **Parity test:** a program that uses profiling counters produces the
  same weight hash on both executors AND the same weight hash as an
  uninstrumented version.

### T2-f. Profile-driven analysis

Run the instrumented `rollout_episode_v22(weights, 80, 1.0, 0.001)` on
a fixed seed. Dump the CSV. Write
`docs/chess_rl_v2/PHASE_E_PROFILE.md` containing:

- The top 5 hot zones by `total_ns`
- Their share of the total episode time
- A ranked list of Tier 3 kernel candidates, ordered by expected
  speedup

### T2 Gate

Profile CSV published. `PHASE_E_PROFILE.md` identifies the ≥ 80% hot
path. Tier 2 unit tests + parity test passing. Workspace regression
clean (no existing tests broken).

---

## TIER 3 — NATIVE HOT-PATH KERNELS

### T3 Goals

Use the Tier 2 profile to pick the hottest kernel. Build a native
builtin that replaces it with a tight Rust loop. Measure the speedup.
Re-run Phase D with the new kernel enabled and report honest numbers.

### T3-a. Native `score_moves_batch` builtin (likely candidate)

**Signature:**
```
score_moves_batch(weights_list: List<Tensor>, feature: Tensor,
                  legal_move_ids: Tensor_i64) -> (Tensor_f64, f64)
```

**Semantics:**
- `weights_list`: the 10-tensor weight bundle (`[W1, b1, W2, b2, Wpf,
  bpf, Wpt, bpt, Wv, bv]`)
- `feature`: the `[1, 774]` input tensor from `encode_state`
- `legal_move_ids`: an i64 tensor of shape `[num_moves, 2]` where each
  row is `[from_sq, to_sq]`
- Returns: a tuple `(scores: [num_moves] f64, value: f64)`

Implements the full forward pass:
```
h1 = relu(feature @ W1 + b1)
h2 = relu(h1 @ W2 + b2)
from_scores = h2 @ Wpf + bpf  # [1, 64]
to_scores   = h2 @ Wpt + bpt  # [1, 64]
scores[i]   = from_scores[0, from_sq[i]] + to_scores[0, to_sq[i]]
value       = tanh(h2 @ Wv + bv)[0, 0]
```

**Parity requirement:** for every position encountered on the
instrumented Phase D run, `score_moves_batch` must produce
**bit-identical** output to the existing pure-CJC-Lang `score_moves`.
This is tested by a new `v22_score_moves_batch_parity` test that
runs both paths on 100 random states (seeded) and asserts equality
down to the last ULP on every score and value.

**Determinism story:** uses the same matmul kernels already in
`cjc-runtime` (Kahan-accumulated, no FMA, BTreeMap iteration). The
only change is eliminating the per-move Python-style loop that builds
`[64, num]` selector matrices.

**Expected speedup:** per the v2.2 profiler data (Tier 2 output),
`score_moves` should be the top zone. Estimate: ~5-10× speedup on the
full forward pass by replacing the O(num_moves) selector-matmul
pattern with a single matmul + two gather operations.

### T3-b. Native `encode_state_fast` builtin (optional, only if profile says so)

**Signature:**
```
encode_state_fast(board: List<i64>, side: i64, castling: List<i64>,
                  ep_sq: i64, halfmove: i64) -> Tensor
```

Replaces the pure-CJC-Lang `encode_state` that iterates 64 squares
through `arr_set` (O(n²) due to immutable array rebuild). Native
version uses a pre-allocated `[1, 774]` buffer filled in a single
pass.

**Parity requirement:** bit-identical to `encode_state` for every
position. Tested on 100 random boards.

**Expected speedup:** 10-30× because `arr_set` rebuilds the whole
array on every call (32 piece placements × 64-element rebuild ≈ 2048
element copies per state).

### T3-c. Persistent GradGraph reuse (optional, lowest priority)

Only if Tier 2 shows GradGraph construction is a top-3 hot zone.
Expose `graph_reset(graph) -> graph` that clears intermediate nodes
while keeping parameter tensors pinned. Must maintain topological-sort
determinism.

### T3-d. Wiring new builtins (mandatory for each)

For each new builtin:

1. Add the implementation in `cjc-runtime/src/builtins.rs` (dispatch arm)
2. Wire the arm in `cjc-eval/src/lib.rs` call handling
3. Wire the arm in `cjc-mir-exec/src/lib.rs` call handling
4. Unit tests in `tests/test_<builtin_name>.rs` (≥5 cases)
5. Proptest coverage (≥1 property)
6. Bolero fuzz target (≥1)
7. Cross-executor parity test (new `v23_<builtin>_parity` in
   `tests/chess_rl_v2/test_training.rs`)
8. Run `cargo test --workspace --release` after wiring. Zero
   regressions.

### T3-e. Opt-in use of native kernels in the v2.3 rollout

Add `rollout_episode_v23` in the PRELUDE that uses the native
kernels where available. Layer this on top of v2.2 (do not replace
v2.2 — v2.2 tests must remain untouched).

### T3-f. Tests for native kernels

Besides the per-builtin tests above:

- **Integration test:** `v23_rollout_matches_v22` asserts that the
  native-kernel rollout and the pure-CJC-Lang v22 rollout produce
  bit-identical trajectories on 10 fixed seeds.
- **Speedup test:** `v23_rollout_speedup` (marked `#[ignore]`) runs
  both rollouts and asserts `v23_time < 0.5 * v22_time` (minimum 2×
  speedup on max_moves=40). This is a measurement gate, not an ML
  gate.

### T3 Gate

- Per-episode wall clock on `rollout_episode_v23(max_moves=80)`:
  **≤ 30 s/episode** (currently ~65 s, so ≥ 2.2× speedup).
  - Stretch target: **≤ 15 s/episode** (≥ 4.3× speedup).
- Cross-executor parity: all 85 existing chess_rl_v2 tests still pass
  PLUS new v23 tests pass on both executors.
- Workspace regression: `cargo test --workspace --release` clean at
  v2.2 baseline count + new tests.
- Determinism: a v23 rollout with the same seed produces the same
  weight hash across reruns.
- Native kernels' outputs are bit-identical to their pure-CJC-Lang
  counterparts on ≥ 100 random states.

---

## TIER 4 — RE-TRAIN (only if Tier 3 gate passes)

Only attempt Tier 4 after Tier 3 hits its per-episode wall clock gate.

### T4-a. Phase D v2.3 re-run

Same driver as v2.2, but:

- Uses `rollout_episode_v23` (native kernels)
- **120 episodes** (doubled from v2.2 because the budget allows)
- max_moves=80, penalty=0.001, eval_temp=0.15
- 20 random games, 10 greedy games, 8 gauntlet games
- Backend: `cjc-eval`
- Seed: 42

### T4 Gate

- Wall clock ≤ 45 min (same as Tier 1 gate)
- Non-zero terminals ≥ 30 / 120 (doubled signal density from Tier 1)
- vs random ≥ 55% (lowered from Tier 1's 60%)
- vs greedy ≥ 50% (lowered from Tier 1's 55%)
- Elo gain ≥ 0 (must not regress)

If T4 passes → ship v2.3 with an honest LinkedIn post.
If T4 misses → document honestly, escalate to Tier 5 (wider trunk +
500-episode run).

---

## DEVELOPMENT WORKFLOW

### Step 1 — Tier 2 implementation

1.1. Design doc for profile counters (`docs/chess_rl_v2/PROFILE_DESIGN.md`)
1.2. Implement `profile_zone_start/stop/dump` in `cjc-runtime`
1.3. Wire in `cjc-eval` and `cjc-mir-exec`
1.4. Unit tests + proptest + bolero
1.5. Parity test (determinism of instrumented vs uninstrumented run)
1.6. **Workspace regression sweep. Must be clean.**

### Step 2 — Profile-driven analysis

2.1. Instrument `rollout_episode_v22` with zones
2.2. Run on seed 42, max_moves=80
2.3. Dump CSV
2.4. Write `docs/chess_rl_v2/PHASE_E_PROFILE.md` with top 5 hot zones
2.5. **If this step surfaces a surprise** (e.g., the hot path is
     *not* `score_moves`), **stop and report**. Do not proceed to
     Tier 3 on a guess.

### Step 3 — Tier 3 native kernel implementation

3.1. Pick the #1 hot zone from Tier 2
3.2. Design doc for the native builtin
3.3. Implement in `cjc-runtime`
3.4. Wire in both executors
3.5. Unit tests + proptest + bolero
3.6. Cross-executor parity test
3.7. **Bit-identical vs pure-CJC-Lang parity test** (≥100 random states)
3.8. **Workspace regression sweep.** Must be clean.
3.9. Measure speedup with the profiler
3.10. If the speedup is < 3×, reject the builtin — it has not earned
      its place. Try a different kernel.

Repeat 3.1-3.10 for up to 3 total new builtins. Stop when Tier 3 gate
passes (≤ 30 s/episode) or after 3 kernels, whichever comes first.

### Step 4 — v2.3 PRELUDE integration

4.1. Add `rollout_episode_v23` using the native kernels
4.2. Add `train_one_episode_adam_v23` that calls `rollout_episode_v23`
4.3. v23 eval variants (`eval_vs_random_v23`, etc.) — these can reuse
     v22 helpers if the bottleneck is only training-time rollout
4.4. Unit tests for `rollout_episode_v23` (smoke, determinism,
     parity, bit-identical-to-v22)
4.5. **Chess RL v2 suite regression.** All 85 existing tests plus new
     v23 tests must pass.

### Step 5 — Phase D v2.3 dry run

5.1. 10-episode timing probe at max_moves=80 to confirm ≤ 30 s/ep
5.2. If not, go back to Step 3

### Step 6 — Phase D v2.3 full run

6.1. 120 episodes, full eval, seed=42, backend cjc-eval
6.2. Collect artifacts into `bench_results/chess_rl_v2_3/`
6.3. Write `docs/chess_rl_v2/PHASE_D_v2_3.md` post-mortem with honest
     numbers

### Step 7 — Final regression sweep

7.1. `cargo test --workspace --release`
7.2. Must pass at v2.2 baseline count + all new v2.3 tests
7.3. **Zero regressions.** If any pre-existing test breaks, stop and
     fix the root cause.

### Step 8 — Documentation and vault updates

8.1. Update `docs/chess_rl_v2/README.md` with v2.3 addendum
8.2. Update `CJC-Lang_Obsidian_Vault/09_Showcase/Chess RL v2.md`
8.3. Run `python scripts/vault_audit.py` — must report "OK: all X
     wikilinks resolve"
8.4. Update `MEMORY.md` with v2.3 numbers
8.5. Update the v2.3 LinkedIn post draft (honest, leads with numbers)

---

## HARD CONSTRAINTS (repeat for emphasis)

- **No external libraries. No new crates. No subprocess calls to
  external tools. No network calls.**
- **New CJC-Lang builtins are encouraged and expected.** The whole
  point of this upgrade is to add hot-path primitives to the language
  runtime. Just follow the wiring pattern and the test requirements.
- **Every new builtin must preserve determinism.** Same seed →
  bit-identical weight hash.
- **Every new builtin must pass cross-executor parity.**
- **Every new builtin must ship with proptest + bolero coverage.**
- **Regression sweep is mandatory after every builtin, after every
  PRELUDE change, and before calling v2.3 done.**
- **No `#[ignore]` to hide failures.**
- **Scope discipline.** No touching unrelated crates, no renaming
  existing builtins, no "while we're here" cleanups.

---

## OUTPUT FORMAT

For each builtin:

```
FILE: crates/cjc-runtime/src/builtins.rs
<dispatch arm code>

FILE: crates/cjc-eval/src/lib.rs
<wiring code>

FILE: crates/cjc-mir-exec/src/lib.rs
<wiring code>

FILE: tests/test_<builtin_name>.rs
<unit tests + proptest + bolero>

FILE: tests/chess_rl_v2/test_training.rs (additions)
<parity test>
```

Then:

```
Test Summary:
  New builtins:    X
  New tests:       Y
  Existing tests:  Z (all passing)
  Workspace tests: W (no regressions)
  Parity gate:     byte-identical ✅
  Determinism:     weight hash reproduces ✅
  Speedup:         measured N× on hot zone
```

And at the end of v2.3:

```
Phase D v2.3 Results:
  Training:        120 episodes, ... min wall clock
  per-episode:     ... s (vs ~65 s in v2.2)
  speedup:         N×
  vs random:       W / D / L  (WR ...%)
  vs greedy:       W / D / L  (WR ...%)
  Elo gauntlet:    ... → ... (Δ ...)
  Non-zero rewards: ... / 120 episodes
  Gate:            PASS / MISS (honest)
```

---

## HARD RULE

If at any point a required change would break:
- Determinism (weight hash reproducibility)
- Cross-executor parity
- The compiler pipeline
- v1, v2.1, v2.2, or earlier test suites
- The NoGC boundary
- The zero-external-dependencies constraint

you must:
1. **Stop.**
2. **Report the conflict in plain language.**
3. **Propose an alternative design that preserves the invariant.**

Never force an unsafe implementation. Never reach for an external
library or tool — new in-language builtins are the escape hatch, and
they must earn their place by measurement.
