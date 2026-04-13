---
title: Deterministic Workflow Examples
tags: [showcase, determinism]
status: Grounded in examples/ and tests/
---

# Deterministic Workflow Examples

Concrete workflows in CJC-Lang where determinism is the point, not a side effect.

## 1. Reproducible ML training

```cjcl
fn main() -> i64 {
    let X = load_training_data();
    let model = init_model(seed=42);
    for epoch in 0..100 {
        let loss = train_step(model, X);
        print(f"epoch {epoch} loss {loss}");
    }
    save_model(model, "model.snap");
    0
}
```

- `init_model(seed=42)` uses [[SplitMix64]] — same seed, same initial weights.
- `train_step` uses Kahan-stable reductions and deterministic AD.
- `save_model` uses [[Binary Serialization]] — two runs produce **bit-identical** `.snap` files.

The [[Chess RL Demo]] exercises this whole loop with 216 tests.

## 2. Statistical analysis that anyone can reproduce

```cjcl
fn main() -> i64 {
    let data = csv_read("survey.csv");
    let ages = data["age"];
    let mean_age = mean(ages);        // Kahan-stable
    let sd_age = sd(ages);            // Kahan-stable
    let test = t_test(ages, 30.0);    // deterministic
    print(f"t = {test.statistic}, p = {test.p_value}");
    0
}
```

- Two different people running this on different machines get **identical p-values to the last bit**.
- No dependency on BLAS vendor, libm version, or hash seed.

## 3. Deterministic quantum experiments

```cjcl
fn main() -> i64 {
    let circuit = qubits(5);
    circuit.h(0);
    for i in 0..4 {
        circuit.cnot(i, i+1);
    }
    let samples = circuit.q_sample(1000, seed=7);
    print(f"samples: {samples}");
    0
}
```

- The measurement outcomes are a pure function of the seed.
- Running the same program on two machines gives the same 1000 sample strings.
- See [[Quantum Simulation]].

## 4. Deterministic visualization

Any call into [[Vizor]] produces byte-identical SVG or BMP output. Two runs of the same plot generate the same file hash — which is how the `gallery/` directory is used as a regression fixture.

## 5. Deterministic data pipelines

[[DataFrame DSL]] operations (`filter`, `group_by`, `join`, `pivot`, `window_sum`) produce identical outputs on identical inputs. Combined with [[Binary Serialization]], you can snapshot intermediate DataFrames, commit them, and detect any semantic drift in future runs via a hash diff.

## Why this matters

Most numerical computing languages say "use a seed and you'll get the same results" and then the seeded RNG interacts with thread-pool scheduling, FMA, `HashMap` iteration order, and vendor BLAS — so you don't. CJC-Lang removes those failure modes by construction.

## Related

- [[Determinism Contract]]
- [[Numerical Truth]]
- [[Parity Gates]]
- [[Chess RL Demo]]
- [[Binary Serialization]]
