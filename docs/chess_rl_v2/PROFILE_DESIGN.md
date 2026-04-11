---
title: CJC-Lang Chess RL v2.3 — Profile Counter Design
status: Step 1.1 of UPGRADE_PROMPT_v2_3.md (design-before-code)
scope: Tier 2 profiling builtins (profile_zone_start / stop / dump)
constraint: Zero external dependencies. Must preserve determinism and cross-executor parity.
---

# Profile Counter Design (Tier 2)

## Goal

Surface the hot path inside `rollout_episode_v22` **without** perturbing
program state, RNG draws, tensor math, or the final weight hash. The
Tier 2 gate is a measurement gate, not an ML gate — so the entire
profiler surface must be a write-only sink.

## Three builtins

| Name                   | Args                    | Returns   | Purpose                                                    |
|------------------------|-------------------------|-----------|------------------------------------------------------------|
| `profile_zone_start`   | `name: String`          | `i64`     | Start timer, return opaque handle                          |
| `profile_zone_stop`    | `handle: i64`           | `f64`     | Stop timer, accumulate stats; return elapsed seconds       |
| `profile_dump`         | `path: String`          | `i64`     | Write sorted CSV; reset counters; return row count         |

`profile_zone_stop`'s return value exists for ergonomics (so the user
*can* inspect a zone's elapsed time in an ad-hoc print), but the parity
test enforces that the program's weight hash is identical whether or
not that return value is used.

## State layout

Internal to `cjc-runtime::profile` (new module), we keep a
**thread-local** `RefCell<ProfileState>`:

```rust
struct ZoneStats {
    count: u64,
    total_ns: u128,
    min_ns: u128,
    max_ns: u128,
    sum_sq_ns: u128,   // for stddev without FMA
}

struct ProfileState {
    zones: BTreeMap<String, ZoneStats>,           // deterministic iteration
    active: BTreeMap<i64, (String, Instant)>,     // handle → (name, start)
    next_handle: i64,                             // monotonically increasing
}
```

- `BTreeMap` — deterministic ordered iteration, per project rules. No
  `HashMap`.
- Thread-local — each test gets its own clean counter state, no cross-
  test interference. The Chess RL training driver runs on one thread, so
  this is the right granularity.
- `u128` for nanoseconds — `Instant::elapsed().as_nanos()` returns
  `u128`, and squaring nanoseconds for stddev can overflow a `u64` after
  a few hundred billion ns.

## Handle scheme

`next_handle` starts at 0 and increments by 1 on every
`profile_zone_start`. The handle is the token the user passes back to
`profile_zone_stop` so we can close the right zone under nested zone
patterns like:

```
let h_outer = profile_zone_start("rollout_total");
  let h_inner = profile_zone_start("score_moves");
  profile_zone_stop(h_inner);
profile_zone_stop(h_outer);
```

The program logic can observe the *identity* of the handle (pair it
with a later `stop` call), but the program must not use the *integer
value* of the handle for control flow, arithmetic, or RNG — because
the value depends on how many zones have been opened previously, which
is a profiling-only concept. The test suite does not enforce this
statically; it's a convention. The parity test covers the one thing
that matters: **ignoring the handle's value produces a bit-identical
weight hash**.

## CSV format

`profile_dump(path)` writes a CSV with the header:

```
zone_name,count,total_ns,min_ns,max_ns,mean_ns,stddev_ns
```

Rows are sorted by `total_ns` descending so the hot zones naturally
appear at the top. Mean is computed as `total_ns / count` (integer
division, deterministic). Stddev uses the two-pass nanosecond sums
formula without FMA:

```
var_ns = sum_sq_ns / count - (total_ns / count)^2
stddev_ns = sqrt(var_ns as f64)
```

For `count == 0` (should not happen after a zone has been stopped),
we emit `0,0,0,0,0,0`.

After writing, the counters are **reset** (zones cleared, next_handle
reset to 0). This gives the v2.3 driver a clean slate for each
instrumented episode and makes the dump function idempotent.

## Determinism story

1. **Program-visible state never depends on profile counters.** The
   only value flowing back into program land is the return value of
   `profile_zone_stop` (f64 elapsed seconds). The parity test enforces
   that the Chess RL v2.2 weight hash reproduces whether or not
   instrumentation is enabled.

2. **No RNG touched.** The profiler never calls `rand`, never draws
   from SplitMix64, never observes the executor RNG state.

3. **No tensor math touched.** The profiler stores only u128
   nanosecond counters; no floats until CSV rendering at `profile_dump`
   time, at which point the counters have already been committed.

4. **Ordered iteration.** `BTreeMap` everywhere so CSV rows appear in
   a reproducible order (we additionally sort by total_ns descending,
   but BTreeMap ordering is the deterministic tiebreaker).

5. **Thread-local isolation.** Tests run serially in the default cargo
   configuration for this crate; one thread, one counter state. Even
   if a future test runs in parallel, each thread keeps its own
   counters — so there's no cross-test coupling on the profiler.

## Parity story

Both `cjc-eval` and `cjc-mir-exec` delegate all builtin calls through
`cjc_runtime::builtins::dispatch_builtin`, so adding a dispatch arm in
one place covers both executors automatically. No changes to
`cjc-eval/src/lib.rs` or `cjc-mir-exec/src/lib.rs` beyond the fact that
they already route unknown calls through runtime dispatch.

The cross-executor parity test runs an instrumented rollout on both
executors and asserts the weight hash is identical.

## Failure modes

- **Stop called on unknown handle:** return -1.0 as elapsed time and do
  not update `zones`. Does not panic (so a typo can't bring down a
  training run).
- **Dump with no zones:** write header only (still a valid CSV), return
  0 as row count.
- **File I/O error:** propagate the error through the dispatch Err
  path, same pattern as `file_append`.
- **Zone name collision with an already-closed zone:** appends to the
  existing `ZoneStats` — this is the normal case for a repeated zone
  (e.g., `score_moves` called 80 times in one episode).

## What this design deliberately avoids

- **No sampling profiler.** We don't need statistical sampling; the
  Chess RL hot path is known to be a few named zones (rollout,
  score_moves, a2c_update) and exact counts are more useful than
  samples.
- **No flame graph output.** CSV is sufficient for ranking and the
  v2.3 post-mortem writes `PHASE_E_PROFILE.md` by hand from the top 5
  rows.
- **No atomic counters.** Thread-local `RefCell` is cheaper and
  sufficient for a single-threaded training driver. Atomics would be a
  ~10× overhead on every `profile_zone_stop`, which would itself
  distort the hot-path measurement.
- **No new crates.** Uses `std::time::Instant`, `std::collections::
  BTreeMap`, `std::cell::RefCell`, `thread_local!`. All already in the
  standard library.

## Expected overhead

Each `profile_zone_start` is:
- One `Instant::now()` call
- One `BTreeMap::insert` at the `active` map (~log n where n is the
  number of currently-open nested zones, typically 1-2)
- One `next_handle += 1`

Each `profile_zone_stop` is:
- One `Instant::now()` call
- One `BTreeMap::remove` at the `active` map
- One `BTreeMap::get_or_insert_with` at the `zones` map
- ~5 u128 additions to update the stats

Estimated per-call overhead: ~500 ns to 2 µs on a modern CPU. With 6
zones instrumented per-episode-step and 80 steps, that's
`6 × 80 × 2 × 2 µs ≈ 2 ms` of overhead per episode at the *worst*
case. At current ~65 s/episode, that's a 0.003% perturbation — well
below the noise floor.

## Out of scope for this doc

- The actual instrumentation of `rollout_episode_v22` (Step 2 of the
  workflow, lives in PRELUDE additions)
- The analysis and Tier 3 kernel selection (Step 2.4, lives in
  PHASE_E_PROFILE.md after measurement)
- Anything about Tier 3 native kernels
