# Chess RL Benchmark: Results

## What This Benchmark Proves

The chess RL benchmark demonstrates that CJC can:

1. **Express complex game logic** -- A complete chess environment (board init, move generation, legality checking, check detection, terminal state detection) in ~280 lines of CJC
2. **Perform neural network inference** -- Matrix multiplication, ReLU activation, softmax normalization with tensors of arbitrary shape
3. **Compute analytical gradients** -- Manual REINFORCE policy gradients with chain rule through matmul and ReLU
4. **Run RL training loops** -- Self-play, trajectory collection, per-step gradient updates, multi-episode training
5. **Maintain strict determinism** -- Identical seeds produce bit-identical results across runs

## Primitives Added

| Builtin | Type | Lines of Rust | Justification |
|---------|------|--------------|---------------|
| `log` | Pure | ~8 | Natural log for log-probability (standard math) |
| `exp` | Pure | ~8 | Exponential for softmax (standard math) |
| `categorical_sample` | Nondet+Alloc | ~15 | Sample from discrete distribution (general RL/stats) |

Additionally wired in MIR-exec (already existed in eval):
- `.transpose()` for 2D tensors
- Tensor-scalar arithmetic (Add, Sub, Mul, Div) for all combinations (Tensor*Float, Float*Tensor, Tensor*Int, Int*Tensor)

**All additions are general-purpose** -- no chess-specific or RL-specific naming. `log` and `exp` are standard mathematical functions. `categorical_sample` is a fundamental statistical operation used across RL, Bayesian inference, NLP, and generative modeling.

## Regression Status

| Metric | Value |
|--------|-------|
| Total workspace tests passed | 2186 |
| Total workspace tests failed | 0 |
| Total workspace tests ignored | 16 (13 pre-existing + 3 diagnostic) |
| Chess RL tests passed | 49 |
| Chess RL tests failed | 0 |

## Test Coverage Breakdown

| Suite | Tests | Coverage |
|-------|-------|----------|
| 01: Board Invariants | 11 | Board init, piece encoding, apply_move, king positions, check detection |
| 02: Movegen Determinism | 7 | Move count, determinism, validity, re-parse determinism |
| 03: Legal Move Sanity | 10 | Pawn rules, knight patterns, king safety, stalemate, encoding |
| 04: Rollout Determinism | 5 | Game length, reward range, seed determinism, batch determinism |
| 05: Training Smoke | 9 | Builtins (log, exp, categorical), forward pass, training metrics |
| 06: Training Determinism | 4 | Single/multi-episode determinism, seed variation |
| 07: Eval vs Random | 5 | Evaluation games, win rate, full pipeline |

## Determinism Proof

- Same-seed double-run gate: **PASS**
- Internal determinism tests: 15 dedicated tests across suites 02, 04, 06
- See [DETERMINISM.md](DETERMINISM.md) for full details

## Limitations

1. **Performance**: The benchmark is not optimized for speed. Move generation rebuilds arrays from scratch, gradient computation recomputes all forward passes. A production chess engine would use bitboards and incremental updates.

2. **Network size**: The 66->16->1 MLP is tiny by RL standards. This was intentional to keep test execution under 60 seconds.

3. **Training effectiveness**: The REINFORCE algorithm with this network and training budget does not learn meaningful chess strategy. The benchmark proves CJC can express and execute the algorithm, not that the algorithm converges.

4. **Parser limitations discovered**:
   - `if` is a statement, not an expression (cannot use `let x = if cond { a } else { b }`)
   - No semicolons after `while {}`/`if {}`/`for {}` blocks inside function bodies
   - Function parameters require type annotations (`fn f(x: i64)` not `fn f(x)`)

5. **Array semantics**: `array_push(arr, val)` returns a new array; must use `arr = array_push(arr, val)`. This is consistent with CJC's immutable-first design but requires explicit reassignment.

## CJC Viability Assessment

**CJC is viable for expressing RL workloads.** The benchmark demonstrates:

- **Tensor operations**: matmul, transpose, element-wise ops, softmax, relu all work correctly through the MIR-exec pipeline
- **Control flow**: while loops, conditionals, function calls, early returns, break statements all work inside function bodies (with correct syntax)
- **Data structures**: Arrays of arrays, arrays of tensors, mixed-type arrays all work for trajectory storage
- **Numerical computing**: IEEE 754 arithmetic, logarithms, exponentials, normalization -- all produce correct results
- **Deterministic execution**: Seeded RNG + deterministic tensor ops = reproducible experiments

The main gaps for production RL use would be:
- Autograd integration (GradGraph exists but wasn't used here)
- Higher-level RL abstractions (replay buffers, advantage estimation, etc.)
- Performance optimization (vectorized operations, GPU support)
- A module system for organizing larger codebases
