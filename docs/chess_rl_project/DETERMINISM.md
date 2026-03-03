# Chess RL Benchmark: Determinism Proof

## Determinism Guarantee

Given the same RNG seed, the chess RL benchmark produces bit-identical results across runs. This is verified by:

1. **Internal determinism tests** (test suites 02, 04, 06): Same-seed runs produce identical move sequences, rollout outcomes, and training trajectories
2. **Double-run gate**: Full test suite run twice with sorted output comparison -- identical results confirmed

## Sources of Determinism

### RNG Seeding
- CJC uses `cjc_repro::Rng` with explicit seed initialization
- `categorical_sample` uses the executor's seeded RNG (`self.rng.next_f64()`)
- Each test specifies its seed via `run_mir(src, seed)`

### Deterministic Move Generation
- Board scanned left-to-right (sq 0..63)
- For each piece, targets generated in fixed directional order
- Legal move filtering preserves ordering (sequential check after pseudo-legal generation)

### Deterministic Network Forward Pass
- Tensor operations (matmul, add, relu, softmax) are deterministic for identical inputs
- No parallel computation or non-deterministic scheduling
- All floating-point operations follow IEEE 754 semantics

### Deterministic Training
- Trajectory stored in arrays, processed sequentially
- Gradient accumulation in fixed move order
- SGD weight updates are deterministic given identical gradients

## Verification Tests

| Test | What it verifies |
|------|-----------------|
| `test_02::movegen_same_seed_deterministic` | Same seed -> identical move lists |
| `test_02::double_run_identical_moves` | Two separate parse+execute cycles -> identical |
| `test_04::rollout_same_seed_identical` | Same seed -> identical game outcome |
| `test_04::rollout_different_seed_differs` | Different seeds -> different outcomes |
| `test_04::multiple_rollouts_deterministic` | Batch of rollouts reproducible |
| `test_06::single_episode_training_deterministic` | Training step reproducible |
| `test_06::multi_episode_training_deterministic` | Multi-episode training reproducible |
| `test_06::different_seeds_produce_different_training` | Different seeds diverge |
| `test_06::eval_vs_random_deterministic` | Evaluation games reproducible |

## Double-Run Gate Result

```
Run 1: 49 passed, 0 failed
Run 2: 49 passed, 0 failed
Sorted comparison: PASS (identical test results, only timing differs)
```

## Known Non-Determinism Boundaries

- **Test execution order**: Rust's parallel test runner may execute tests in different orders, but each individual test is deterministic
- **Timing**: Wall-clock execution time varies between runs (irrelevant to output determinism)
- **Cross-platform**: Determinism is verified on the same platform (Windows 11, x86_64). IEEE 754 guarantees should extend to other platforms for basic operations, but edge cases (e.g., FMA fusion) could theoretically differ
