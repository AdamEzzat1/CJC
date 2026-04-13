> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [../REBRAND_NOTICE.md](../REBRAND_NOTICE.md) for the full mapping.

# Primitive Coverage Map

## Overview

The `bench_primitive_coverage.cjc` benchmark tests 40+ CJC builtins and produces
a single master SHA-256 hash. If any builtin changes behavior, the master hash
changes and the golden-hash test (`primitive_master_hash_golden`) fails.

## Covered Primitives

### Tensor Constructors (3)

| Primitive | Test Expression |
|-----------|----------------|
| `Tensor.zeros` | `Tensor.zeros([4, 4])` |
| `Tensor.ones` | `Tensor.ones([4, 4])` |
| `Tensor.randn` | `Tensor.randn([4, 4])` (seed-dependent) |

### Tensor Operations (7)

| Primitive | Test Expression |
|-----------|----------------|
| `matmul` | `matmul(a, b)` (3x3 matrices) |
| `norm` | `norm(a)` (L2 norm) |
| `argmax` | `argmax(a)` |
| `.sum()` | `a.sum()` (tensor method) |
| `.mean()` | `a.mean()` (tensor method) |
| `.transpose()` | `a.transpose()` (tensor method) |
| `dot` | `dot(v1, v2)` (1D vectors) |

### Broadcast Unary (11)

| Function | Input |
|----------|-------|
| `sqrt` | `[0.0, 1.0, 4.0, 9.0]` |
| `sin` | `[0.0, 1.0, 4.0, 9.0]` |
| `cos` | `[0.0, 1.0, 4.0, 9.0]` |
| `exp` | `[0.0, 1.0, 4.0, 9.0]` |
| `abs` | `[0.0, 1.0, 4.0, 9.0]` |
| `relu` | `[0.0, 1.0, 4.0, 9.0]` |
| `sigmoid` | `[0.0, 1.0, 4.0, 9.0]` |
| `tanh` | `[0.0, 1.0, 4.0, 9.0]` |
| `floor` | `[0.0, 1.0, 4.0, 9.0]` |
| `ceil` | `[0.0, 1.0, 4.0, 9.0]` |
| `neg` | `[0.0, 1.0, 4.0, 9.0]` |

### Broadcast Binary (8)

| Function | Inputs |
|----------|--------|
| `add` | `[2,3,4]` + `[3,2,1]` |
| `sub` | `[2,3,4]` - `[3,2,1]` |
| `mul` | `[2,3,4]` * `[3,2,1]` |
| `div` | `[2,3,4]` / `[3,2,1]` |
| `pow` | `[2,3,4]` ^ `[3,2,1]` |
| `min` | element-wise min |
| `max` | element-wise max |
| `atan2` | element-wise atan2 |

### Math Builtins (9)

| Primitive | Test Expression |
|-----------|----------------|
| `sin` | `sin(1.0)` |
| `cos` | `cos(1.0)` |
| `sqrt` | `sqrt(2.0)` |
| `exp` | `exp(1.0)` |
| `log` | `log(2.718281828)` |
| `abs` | `abs(-5.0)` |
| `floor` | `floor(3.7)` |
| `ceil` | `ceil(3.2)` |
| `round` | `round(3.5)` |

### Statistics Builtins (5)

| Primitive | Test Expression |
|-----------|----------------|
| `median` | `median([1..8])` |
| `sd` | `sd([1..8])` |
| `variance` | `variance([1..8])` |
| `quantile` | `quantile([1..8], 0.25)` |
| `iqr` | `iqr([1..8])` |

### Additional Operations (3)

| Primitive | Test Expression |
|-----------|----------------|
| `matmul` (identity) | `matmul(eye3, a)` |
| `broadcast("sign")` | `broadcast("sign", t)` |
| `broadcast2("atan2")` | `broadcast2("atan2", p, q)` |

### String Operations (5)

| Primitive | Test Expression |
|-----------|----------------|
| `str_len` | `str_len("hello world")` |
| `str_to_upper` | `str_to_upper("hello")` |
| `str_replace` | `str_replace("foo bar foo", "foo", "baz")` |
| `str_trim` | `str_trim("  hello  ")` |
| `str_to_lower` | `str_to_lower("WORLD")` |

## Total: 51 primitive operations

## How to Extend Coverage

1. Add new `snap_hash(...)` lines to `bench_primitive_coverage.cjc`
2. Run `cargo test --test test_bench_v0_1 primitive_master_hash_golden -- --nocapture`
3. Copy the new master hash from stderr output
4. Update `GOLDEN_MASTER` in `test_primitive_abi.rs`
5. Verify: `cargo test --test test_bench_v0_1`

## Golden Hash

Current golden master: `7c2e0248d8e50c7a58d9e8c64afb9b8c31db43e81835673eac2ca46a077275d5`

This hash locks down the combined behavior of all 51 primitives (46 original +
5 string ops). Any change to any builtin's output will produce a different hash
and fail the test.
