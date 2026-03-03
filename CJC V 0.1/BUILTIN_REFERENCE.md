# CJC Built-in Function Reference (v0.1)

Complete catalog of all built-in functions, constants, and methods available in CJC.

---

## Table of Contents

1. [Mathematical Constants](#mathematical-constants)
2. [Core Mathematics](#core-mathematics)
3. [Trigonometry](#trigonometry)
4. [Type Conversion & Inspection](#type-conversion--inspection)
5. [Statistics -- Descriptive](#statistics----descriptive)
6. [Statistics -- Distributions](#statistics----distributions)
7. [Statistics -- Hypothesis Testing](#statistics----hypothesis-testing)
8. [Statistics -- Regression](#statistics----regression)
9. [Linear Algebra](#linear-algebra)
10. [Machine Learning -- Loss Functions](#machine-learning----loss-functions)
11. [Machine Learning -- Regularization & Learning Rate](#machine-learning----regularization--learning-rate)
12. [Machine Learning -- Layers & Evaluation](#machine-learning----layers--evaluation)
13. [Machine Learning -- Activations](#machine-learning----activations)
14. [Machine Learning -- Optimizers](#machine-learning----optimizers)
15. [Machine Learning -- Gradient Utilities](#machine-learning----gradient-utilities)
16. [Tensor Constructors](#tensor-constructors)
17. [Tensor Shape & Query](#tensor-shape--query)
18. [Tensor Arithmetic](#tensor-arithmetic)
19. [Tensor Activations](#tensor-activations)
20. [Tensor Matrix Operations](#tensor-matrix-operations)
21. [Tensor Convolution & Pooling](#tensor-convolution--pooling)
22. [Tensor View Operations](#tensor-view-operations)
23. [Automatic Differentiation -- Forward Mode](#automatic-differentiation----forward-mode)
24. [Automatic Differentiation -- GradGraph (Reverse Mode)](#automatic-differentiation----gradgraph-reverse-mode)
25. [Array & Collection Operations](#array--collection-operations)
26. [Map Operations](#map-operations)
27. [Set Operations](#set-operations)
28. [String Operations](#string-operations)
29. [DateTime](#datetime)
30. [I/O](#io)
31. [CSV](#csv)
32. [JSON](#json)
33. [Snap Serialization](#snap-serialization)
34. [Bitwise Operations](#bitwise-operations)
35. [Cumulative & Window Functions](#cumulative--window-functions)
36. [Complex Numbers](#complex-numbers)
37. [FFT & Signal Processing](#fft--signal-processing)
38. [Sparse Tensors](#sparse-tensors)
39. [Data DSL / Tidy](#data-dsl--tidy)
40. [Assertions & Miscellaneous](#assertions--miscellaneous)
41. [Scratchpad & KV-Cache](#scratchpad--kv-cache)
42. [Float16 / BFloat16](#float16--bfloat16)
43. [Buffer & Memory](#buffer--memory)
44. [Broadcasting Builtins](#broadcasting-builtins)

---

## Mathematical Constants

| Name       | Usage      | Description                              |
|------------|------------|------------------------------------------|
| `PI`       | `PI`       | The constant pi (3.14159...)             |
| `E`        | `E`        | Euler's number (2.71828...)              |
| `TAU`      | `TAU`      | Tau, equal to 2 * pi (6.28318...)        |
| `INF`      | `INF`      | Positive infinity                        |
| `NAN_VAL`  | `NAN_VAL`  | IEEE 754 Not-a-Number value              |

---

## Core Mathematics

| Name    | Usage                   | Description                                      |
|---------|-------------------------|--------------------------------------------------|
| `sqrt`  | `sqrt(x)`               | Square root of x                                 |
| `pow`   | `pow(x, n)`             | Raise x to the power n                           |
| `exp`   | `exp(x)`                | Exponential function e^x                         |
| `log`   | `log(x)`                | Natural logarithm (ln)                           |
| `log2`  | `log2(x)`               | Base-2 logarithm                                 |
| `log10` | `log10(x)`              | Base-10 logarithm                                |
| `log1p` | `log1p(x)`              | Natural log of (1 + x), accurate for small x     |
| `expm1` | `expm1(x)`              | e^x - 1, accurate for small x                    |
| `abs`   | `abs(x)`                | Absolute value                                   |
| `sign`  | `sign(x)`               | Sign of x: -1, 0, or 1                           |
| `floor` | `floor(x)`              | Round down to nearest integer                    |
| `ceil`  | `ceil(x)`               | Round up to nearest integer                      |
| `round` | `round(x)`              | Round to nearest integer                         |
| `hypot` | `hypot(x, y)`           | Euclidean distance: sqrt(x^2 + y^2)              |
| `clamp` | `clamp(x, min, max)`    | Constrain x to the range [min, max]              |
| `min`   | `min(x, y)`             | Minimum of two values                            |
| `max`   | `max(x, y)`             | Maximum of two values                            |

```cjc
// Core math example
let x = 2.0
let y = sqrt(x)           // 1.4142...
let z = pow(x, 3.0)       // 8.0
let c = clamp(15.0, 0.0, 10.0)  // 10.0
let h = hypot(3.0, 4.0)   // 5.0
print(log(E))              // 1.0
```

---

## Trigonometry

| Name          | Usage              | Description                                 |
|---------------|--------------------|---------------------------------------------|
| `sin`         | `sin(x)`           | Sine (radians)                              |
| `cos`         | `cos(x)`           | Cosine (radians)                            |
| `tan`         | `tan(x)`           | Tangent (radians)                           |
| `asin`        | `asin(x)`          | Inverse sine (arc sine)                     |
| `acos`        | `acos(x)`          | Inverse cosine (arc cosine)                 |
| `atan`        | `atan(x)`          | Inverse tangent (arc tangent)               |
| `atan2`       | `atan2(y, x)`      | Two-argument arc tangent                    |
| `sinh`        | `sinh(x)`          | Hyperbolic sine                             |
| `cosh`        | `cosh(x)`          | Hyperbolic cosine                           |
| `tanh_scalar` | `tanh_scalar(x)`   | Hyperbolic tangent (scalar)                 |

```cjc
// Trigonometry example
let angle = PI / 4.0
let s = sin(angle)         // 0.7071...
let c = cos(angle)         // 0.7071...
let heading = atan2(1.0, 1.0)  // PI/4
print(sinh(1.0))           // 1.1752...
```

---

## Type Conversion & Inspection

| Name        | Usage           | Description                                  |
|-------------|-----------------|----------------------------------------------|
| `int`       | `int(x)`        | Convert to integer (truncating)              |
| `float`     | `float(x)`      | Convert to floating-point                    |
| `to_string` | `to_string(x)`  | Convert any value to its string form         |
| `isnan`     | `isnan(x)`      | Returns true if x is NaN                     |
| `isinf`     | `isinf(x)`      | Returns true if x is positive or negative infinity |

```cjc
// Type conversion example
let n = int(3.7)           // 3
let f = float(42)          // 42.0
let s = to_string(123)     // "123"
print(isnan(NAN_VAL))      // true
print(isinf(INF))          // true
```

---

## Statistics -- Descriptive

| Name              | Usage                       | Description                                      |
|-------------------|-----------------------------|--------------------------------------------------|
| `mean`            | `mean(arr)`                 | Arithmetic mean                                  |
| `variance`        | `variance(arr)`             | Population variance                              |
| `sd`              | `sd(arr)`                   | Population standard deviation                    |
| `se`              | `se(arr)`                   | Standard error of the mean                       |
| `median`          | `median(arr)`               | Median value                                     |
| `quantile`        | `quantile(arr, q)`          | q-th quantile (q in [0,1])                       |
| `iqr`             | `iqr(arr)`                  | Interquartile range (Q3 - Q1)                    |
| `skewness`        | `skewness(arr)`             | Sample skewness                                  |
| `kurtosis`        | `kurtosis(arr)`             | Sample kurtosis                                  |
| `z_score`         | `z_score(arr)`              | Z-scores for each element                        |
| `standardize`     | `standardize(arr)`          | Standardize to zero mean, unit variance          |
| `n_distinct`      | `n_distinct(arr)`           | Count of distinct elements                       |
| `mad`             | `mad(arr)`                  | Median absolute deviation                        |
| `mode`            | `mode(arr)`                 | Most frequently occurring value                  |
| `percentile_rank` | `percentile_rank(arr, val)` | Percentile rank of val within arr                |
| `trimmed_mean`    | `trimmed_mean(arr, pct)`    | Mean after trimming pct% from each tail          |
| `weighted_mean`   | `weighted_mean(arr, w)`     | Weighted arithmetic mean                         |
| `weighted_var`    | `weighted_var(arr, w)`      | Weighted variance                                |
| `winsorize`       | `winsorize(arr, pct)`       | Winsorize by clamping extreme values at pct%     |
| `sample_variance` | `sample_variance(arr)`      | Sample variance (Bessel-corrected)               |
| `sample_sd`       | `sample_sd(arr)`            | Sample standard deviation (Bessel-corrected)     |
| `sample_cov`      | `sample_cov(x, y)`         | Sample covariance of two arrays                  |
| `cor`             | `cor(x, y)`                 | Pearson correlation coefficient                  |
| `cov`             | `cov(x, y)`                 | Population covariance                            |
| `spearman_cor`    | `spearman_cor(x, y)`       | Spearman rank correlation                        |
| `kendall_cor`     | `kendall_cor(x, y)`        | Kendall tau rank correlation                     |
| `partial_cor`     | `partial_cor(x, y, z)`     | Partial correlation controlling for z            |
| `cor_ci`          | `cor_ci(x, y, conf)`       | Confidence interval for Pearson correlation      |

```cjc
// Descriptive statistics example
let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
print(mean(data))          // 5.0
print(median(data))        // 4.5
print(sd(data))            // ~2.0
print(quantile(data, 0.75)) // 5.75 (Q3)
print(iqr(data))           // Q3 - Q1
let x = [1.0, 2.0, 3.0]
let y = [2.0, 4.0, 6.0]
print(cor(x, y))           // 1.0
```

---

## Statistics -- Distributions

| Name           | Usage                          | Description                                  |
|----------------|--------------------------------|----------------------------------------------|
| `normal_cdf`   | `normal_cdf(x)`               | Standard normal cumulative distribution      |
| `normal_pdf`   | `normal_pdf(x)`               | Standard normal probability density          |
| `normal_ppf`   | `normal_ppf(p)`               | Standard normal percent-point (inverse CDF)  |
| `t_cdf`        | `t_cdf(x, df)`                | Student's t cumulative distribution          |
| `t_ppf`        | `t_ppf(p, df)`                | Student's t inverse CDF                      |
| `chi2_cdf`     | `chi2_cdf(x, df)`             | Chi-squared cumulative distribution          |
| `chi2_ppf`     | `chi2_ppf(p, df)`             | Chi-squared inverse CDF                      |
| `f_cdf`        | `f_cdf(x, df1, df2)`          | F-distribution cumulative distribution       |
| `f_ppf`        | `f_ppf(p, df1, df2)`          | F-distribution inverse CDF                   |
| `beta_pdf`     | `beta_pdf(x, a, b)`           | Beta probability density                     |
| `beta_cdf`     | `beta_cdf(x, a, b)`           | Beta cumulative distribution                 |
| `gamma_pdf`    | `gamma_pdf(x, shape, rate)`   | Gamma probability density                    |
| `gamma_cdf`    | `gamma_cdf(x, shape, rate)`   | Gamma cumulative distribution                |
| `exp_pdf`      | `exp_pdf(x, rate)`            | Exponential probability density              |
| `exp_cdf`      | `exp_cdf(x, rate)`            | Exponential cumulative distribution          |
| `weibull_pdf`  | `weibull_pdf(x, shape, scale)`| Weibull probability density                  |
| `weibull_cdf`  | `weibull_cdf(x, shape, scale)`| Weibull cumulative distribution              |
| `binomial_pmf` | `binomial_pmf(k, n, p)`       | Binomial probability mass function           |
| `binomial_cdf` | `binomial_cdf(k, n, p)`       | Binomial cumulative distribution             |
| `poisson_pmf`  | `poisson_pmf(k, lambda)`      | Poisson probability mass function            |
| `poisson_cdf`  | `poisson_cdf(k, lambda)`      | Poisson cumulative distribution              |

```cjc
// Distribution functions example
let p = normal_cdf(1.96)      // ~0.975
let z = normal_ppf(0.975)     // ~1.96
let t_val = t_ppf(0.975, 10.0)  // critical t for df=10
print(binomial_pmf(3, 10, 0.5))  // P(X=3) for Bin(10,0.5)
print(poisson_pmf(5, 3.0))       // P(X=5) for Poisson(3)
```

---

## Statistics -- Hypothesis Testing

| Name                   | Usage                          | Description                                        |
|------------------------|--------------------------------|----------------------------------------------------|
| `t_test`               | `t_test(arr, mu)`              | One-sample t-test against hypothesized mean mu     |
| `t_test_two_sample`    | `t_test_two_sample(a, b)`      | Independent two-sample t-test                      |
| `t_test_paired`        | `t_test_paired(a, b)`          | Paired-sample t-test                               |
| `chi_squared_test`     | `chi_squared_test(obs, exp)`   | Chi-squared goodness-of-fit test                   |
| `f_test`               | `f_test(v1, v2)`               | F-test for equality of variances                   |
| `anova_oneway`         | `anova_oneway(groups)`         | One-way ANOVA across groups                        |
| `mann_whitney`         | `mann_whitney(a, b)`           | Mann-Whitney U test (nonparametric)                |
| `wilcoxon_signed_rank` | `wilcoxon_signed_rank(a, b)`   | Wilcoxon signed-rank test (nonparametric)          |
| `kruskal_wallis`       | `kruskal_wallis(groups)`       | Kruskal-Wallis H test (nonparametric ANOVA)        |
| `tukey_hsd`            | `tukey_hsd(groups, treat)`     | Tukey HSD post-hoc pairwise comparisons            |
| `bonferroni`           | `bonferroni(pvals, alpha)`     | Bonferroni correction for multiple comparisons     |
| `fdr_bh`               | `fdr_bh(pvals, alpha)`         | Benjamini-Hochberg false discovery rate correction |

```cjc
// Hypothesis testing example
let control = [5.1, 4.9, 5.0, 5.2, 4.8]
let treatment = [5.5, 5.7, 5.6, 5.8, 5.4]
let result = t_test_two_sample(control, treatment)
print(result)   // { t_stat, p_value, df }

let pvals = [0.01, 0.04, 0.03, 0.20]
let adjusted = bonferroni(pvals, 0.05)
print(adjusted)
```

---

## Statistics -- Regression

| Name                  | Usage                     | Description                              |
|-----------------------|---------------------------|------------------------------------------|
| `lm`                  | `lm(X, y)`               | Ordinary least squares linear regression |
| `wls`                 | `wls(X, y, w)`            | Weighted least squares regression        |
| `logistic_regression` | `logistic_regression(X, y)` | Logistic regression (binary classifier) |

```cjc
// Regression example
let X = [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]]
let y = [3.0, 5.0, 8.0]
let model = lm(X, y)
print(model)   // { coefficients, r_squared, ... }
```

---

## Linear Algebra

| Name              | Usage                     | Description                                  |
|-------------------|---------------------------|----------------------------------------------|
| `matmul`          | `matmul(A, B)`            | Matrix multiplication                        |
| `dot`             | `dot(a, b)`               | Dot product of two vectors                   |
| `outer`           | `outer(a, b)`             | Outer product of two vectors                 |
| `cross`           | `cross(a, b)`             | Cross product of two 3D vectors              |
| `norm`            | `norm(v)`                 | Euclidean (L2) norm of a vector              |
| `det`             | `det(A)`                  | Matrix determinant                           |
| `solve`           | `solve(A, b)`             | Solve linear system Ax = b                   |
| `lstsq`           | `lstsq(A, b)`             | Least-squares solution to Ax ~ b             |
| `trace`           | `trace(A)`                | Sum of diagonal elements                     |
| `kron`            | `kron(A, B)`              | Kronecker product                            |
| `eigh`            | `eigh(A)`                 | Eigenvalues/vectors of symmetric matrix      |
| `schur`           | `schur(A)`                | Schur decomposition                          |
| `matrix_exp`      | `matrix_exp(A)`           | Matrix exponential                           |
| `matrix_rank`     | `matrix_rank(A)`          | Rank of a matrix                             |
| `cond`            | `cond(A)`                 | Condition number                             |
| `norm_frobenius`  | `norm_frobenius(A)`       | Frobenius norm                               |
| `norm_1`          | `norm_1(A)`               | 1-norm (max column sum)                      |
| `norm_inf`        | `norm_inf(A)`             | Infinity norm (max row sum)                  |
| `linalg.lu`       | `linalg.lu(A)`            | LU decomposition                             |
| `linalg.qr`       | `linalg.qr(A)`           | QR decomposition                             |
| `linalg.cholesky` | `linalg.cholesky(A)`     | Cholesky decomposition (symmetric pos-def)   |
| `linalg.inv`      | `linalg.inv(A)`          | Matrix inverse                               |

```cjc
// Linear algebra example
let A = [[1.0, 2.0], [3.0, 4.0]]
let b = [5.0, 11.0]
let x = solve(A, b)       // [1.0, 2.0]
print(det(A))              // -2.0
print(trace(A))            // 5.0
let v = [3.0, 4.0]
print(norm(v))             // 5.0
```

---

## Machine Learning -- Loss Functions

| Name                   | Usage                                 | Description                                |
|------------------------|---------------------------------------|--------------------------------------------|
| `mse_loss`             | `mse_loss(pred, actual)`              | Mean squared error loss                    |
| `cross_entropy_loss`   | `cross_entropy_loss(logits, labels)`  | Cross-entropy loss for multi-class         |
| `binary_cross_entropy` | `binary_cross_entropy(pred, labels)`  | Binary cross-entropy loss                  |
| `huber_loss`           | `huber_loss(pred, actual, delta)`     | Huber loss (smooth L1)                     |
| `hinge_loss`           | `hinge_loss(pred, labels)`            | Hinge loss for SVM-style classification    |

```cjc
// Loss function example
let pred = Tensor.from_vec([0.9, 0.1, 0.8], [3])
let actual = Tensor.from_vec([1.0, 0.0, 1.0], [3])
let loss = mse_loss(pred, actual)
print(loss)
```

---

## Machine Learning -- Regularization & Learning Rate

| Name              | Usage                                  | Description                                    |
|-------------------|----------------------------------------|------------------------------------------------|
| `l1_penalty`      | `l1_penalty(w)`                        | L1 (lasso) regularization penalty              |
| `l2_penalty`      | `l2_penalty(w)`                        | L2 (ridge) regularization penalty              |
| `lr_step_decay`   | `lr_step_decay(lr, step, rate, steps)` | Step decay learning rate schedule              |
| `lr_cosine`       | `lr_cosine(lr, step, total)`           | Cosine annealing learning rate schedule        |
| `lr_linear_warmup`| `lr_linear_warmup(lr, step, warmup)`   | Linear warmup learning rate schedule           |

```cjc
// Learning rate schedule example
let base_lr = 0.01
let step = 50
let total_steps = 100
let lr = lr_cosine(base_lr, step, total_steps)
print(lr)   // cosine-annealed LR at step 50
```

---

## Machine Learning -- Layers & Evaluation

| Name                  | Usage                                    | Description                                   |
|-----------------------|------------------------------------------|-----------------------------------------------|
| `attention`           | `attention(Q, K, V, mask)`               | Scaled dot-product attention                  |
| `batch_norm`          | `batch_norm(x, gamma, beta, mean, var, eps)` | Batch normalization                       |
| `dropout_mask`        | `dropout_mask(shape, prob, seed)`        | Generate a dropout mask                       |
| `confusion_matrix`    | `confusion_matrix(pred, actual)`         | Build confusion matrix from predictions       |
| `auc_roc`             | `auc_roc(pred, actual)`                  | Area under the ROC curve                      |
| `one_hot`             | `one_hot(indices, classes)`              | One-hot encode indices into class vectors     |
| `logistic_regression` | `logistic_regression(X, y)`              | Logistic regression classifier                |
| `lm`                  | `lm(X, y)`                              | Linear model (OLS)                            |

```cjc
// Attention layer example
let Q = Tensor.randn([2, 4])
let K = Tensor.randn([2, 4])
let V = Tensor.randn([2, 4])
let out = attention(Q, K, V, Tensor.ones([2, 2]))
print(Tensor.shape(out))
```

---

## Machine Learning -- Activations

Standalone activation functions:

| Name         | Usage                 | Description                                   |
|--------------|-----------------------|-----------------------------------------------|
| `leaky_relu` | `leaky_relu(x, alpha)`| Leaky ReLU with configurable negative slope   |
| `silu`       | `silu(x)`             | SiLU / Swish activation (x * sigmoid(x))     |
| `mish`       | `mish(x)`             | Mish activation (x * tanh(softplus(x)))       |

Tensor activation methods (see [Tensor Activations](#tensor-activations) for full list):
`.relu()`, `.sigmoid()`, `.gelu()`, `.softmax()`, `.tanh_activation()`

```cjc
// Activations example
print(leaky_relu(-2.0, 0.01))  // -0.02
print(silu(1.0))                // ~0.7311
print(mish(1.0))                // ~0.8651
```

---

## Machine Learning -- Optimizers

| Name                  | Usage                              | Description                               |
|-----------------------|------------------------------------|-------------------------------------------|
| `Adam.new`            | `Adam.new(lr, beta1, beta2, eps)`  | Create Adam optimizer                     |
| `Sgd.new`             | `Sgd.new(lr)`                      | Create SGD optimizer                      |
| `OptimizerState.step` | `OptimizerState.step(state, params, grads)` | Perform one optimization step    |

```cjc
// Optimizer example
let opt = Adam.new(0.001, 0.9, 0.999, 1e-8)
let params = Tensor.randn([10])
let grads = Tensor.randn([10])
let opt2 = OptimizerState.step(opt, params, grads)
```

---

## Machine Learning -- Gradient Utilities

| Name              | Usage                    | Description                                    |
|-------------------|--------------------------|------------------------------------------------|
| `stop_gradient`   | `stop_gradient(x)`       | Detach value from gradient computation         |
| `grad_checkpoint` | `grad_checkpoint(x)`     | Checkpoint value for memory-efficient backprop |
| `clip_grad`       | `clip_grad(grad, max)`   | Clip gradient norm to max value                |
| `grad_scale`      | `grad_scale(grad, s)`    | Scale gradient by factor s                     |

```cjc
// Gradient utilities example
let g = Tensor.randn([5])
let clipped = clip_grad(g, 1.0)   // norm <= 1.0
let scaled = grad_scale(g, 0.5)   // halve the gradient
```

---

## Tensor Constructors

| Name              | Usage                                  | Description                                  |
|-------------------|----------------------------------------|----------------------------------------------|
| `Tensor.zeros`    | `Tensor.zeros(shape)`                  | Tensor filled with zeros                     |
| `Tensor.ones`     | `Tensor.ones(shape)`                   | Tensor filled with ones                      |
| `Tensor.eye`      | `Tensor.eye(n)`                        | n x n identity matrix                        |
| `Tensor.randn`    | `Tensor.randn(shape)`                  | Tensor with standard normal random values    |
| `Tensor.uniform`  | `Tensor.uniform(shape)`               | Tensor with uniform random values in [0,1)   |
| `Tensor.full`     | `Tensor.full(shape, val)`              | Tensor filled with a specific value          |
| `Tensor.linspace` | `Tensor.linspace(start, end, steps)`   | Evenly spaced values from start to end       |
| `Tensor.arange`   | `Tensor.arange(start, end, step)`      | Values from start to end with given step     |
| `Tensor.from_vec` | `Tensor.from_vec(data, shape)`         | Create tensor from flat data and shape       |
| `Tensor.diag`     | `Tensor.diag(vec)`                     | Diagonal matrix from a vector                |
| `Tensor.from_bytes`| `Tensor.from_bytes(bytes)`            | Deserialize tensor from byte buffer          |

```cjc
// Tensor constructors example
let z = Tensor.zeros([3, 3])
let I = Tensor.eye(4)
let r = Tensor.randn([2, 5])
let v = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2])
let ls = Tensor.linspace(0.0, 1.0, 5)  // [0.0, 0.25, 0.5, 0.75, 1.0]
```

---

## Tensor Shape & Query

| Name                | Usage                           | Description                              |
|---------------------|---------------------------------|------------------------------------------|
| `Tensor.shape`      | `Tensor.shape(t)`               | Return the shape as an array             |
| `Tensor.len`        | `Tensor.len(t)`                 | Total number of elements                 |
| `Tensor.get`        | `Tensor.get(t, idx)`            | Get element at flat or multi-dim index   |
| `Tensor.set`        | `Tensor.set(t, idx, val)`       | Set element at flat or multi-dim index   |
| `Tensor.reshape`    | `Tensor.reshape(t, shape)`      | Reshape to new dimensions                |
| `Tensor.transpose`  | `Tensor.transpose(t)`           | Transpose the tensor                     |
| `Tensor.broadcast_to` | `Tensor.broadcast_to(t, shape)` | Broadcast tensor to target shape       |
| `Tensor.to_vec`     | `Tensor.to_vec(t)`              | Convert tensor to a flat array           |

```cjc
// Shape & query example
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
print(Tensor.shape(t))    // [2, 3]
print(Tensor.len(t))      // 6
print(Tensor.get(t, [0, 1]))  // 2.0
let flat = Tensor.reshape(t, [6])
```

---

## Tensor Arithmetic

| Name                | Usage                        | Description                             |
|---------------------|------------------------------|-----------------------------------------|
| `Tensor.add`        | `Tensor.add(a, b)`          | Element-wise addition                   |
| `Tensor.sub`        | `Tensor.sub(a, b)`          | Element-wise subtraction                |
| `Tensor.mul`        | `Tensor.mul(a, b)`          | Element-wise multiplication             |
| `Tensor.scalar_mul` | `Tensor.scalar_mul(t, s)`   | Multiply every element by scalar s      |
| `Tensor.neg`        | `Tensor.neg(t)`             | Negate every element                    |
| `Tensor.sum_axis`   | `Tensor.sum_axis(t, ax)`    | Sum along a given axis                  |
| `Tensor.binned_sum` | `Tensor.binned_sum(t)`      | Binned summation for numerical stability|
| `sum`               | `sum(t)`                    | Sum all elements                        |
| `mean`              | `mean(t)`                   | Mean of all elements                    |
| `argmax`            | `argmax(t)`                 | Index of the maximum element            |
| `argmin`            | `argmin(t)`                 | Index of the minimum element            |

```cjc
// Tensor arithmetic example
let a = Tensor.from_vec([1.0, 2.0, 3.0], [3])
let b = Tensor.from_vec([4.0, 5.0, 6.0], [3])
let c = Tensor.add(a, b)      // [5.0, 7.0, 9.0]
let s = Tensor.scalar_mul(a, 10.0)  // [10.0, 20.0, 30.0]
print(sum(a))                  // 6.0
print(argmax(b))               // 2
```

---

## Tensor Activations

| Name                     | Usage                                 | Description                              |
|--------------------------|---------------------------------------|------------------------------------------|
| `Tensor.relu`            | `Tensor.relu(t)`                      | ReLU activation (max(0, x))             |
| `Tensor.sigmoid`         | `Tensor.sigmoid(t)`                   | Sigmoid activation (1/(1+e^-x))         |
| `Tensor.softmax`         | `Tensor.softmax(t)`                   | Softmax normalization                    |
| `Tensor.gelu`            | `Tensor.gelu(t)`                      | GELU activation                          |
| `Tensor.tanh_activation` | `Tensor.tanh_activation(t)`           | Tanh activation                          |
| `Tensor.layer_norm`      | `Tensor.layer_norm(t, gamma, beta, eps)` | Layer normalization                  |

```cjc
// Tensor activations example
let logits = Tensor.from_vec([2.0, 1.0, 0.1], [3])
let probs = Tensor.softmax(logits)
print(probs)               // softmax probabilities
let activated = Tensor.relu(Tensor.from_vec([-1.0, 0.0, 1.0], [3]))
print(activated)           // [0.0, 0.0, 1.0]
```

---

## Tensor Matrix Operations

| Name                 | Usage                         | Description                                 |
|----------------------|-------------------------------|---------------------------------------------|
| `Tensor.matmul`      | `Tensor.matmul(a, b)`       | Matrix multiplication                       |
| `Tensor.bmm`         | `Tensor.bmm(a, b)`          | Batched matrix multiplication               |
| `Tensor.linear`      | `Tensor.linear(x, W, b)`    | Linear layer: x @ W^T + b                  |
| `Tensor.split_heads` | `Tensor.split_heads(t, h)`  | Split last dim into h attention heads       |
| `Tensor.merge_heads` | `Tensor.merge_heads(t)`     | Merge attention heads back into single dim  |

```cjc
// Matrix operations example
let A = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2])
let B = Tensor.eye(2)
let C = Tensor.matmul(A, B)   // A * I = A
let x = Tensor.randn([4, 8])
let W = Tensor.randn([16, 8])
let b = Tensor.zeros([16])
let out = Tensor.linear(x, W, b)  // [4, 16]
```

---

## Tensor Convolution & Pooling

| Name              | Usage                                 | Description                              |
|-------------------|---------------------------------------|------------------------------------------|
| `Tensor.conv1d`   | `Tensor.conv1d(x, k, stride, pad)`   | 1D convolution                           |
| `Tensor.conv2d`   | `Tensor.conv2d(x, k, stride, pad)`   | 2D convolution                           |
| `Tensor.maxpool2d`| `Tensor.maxpool2d(x, k, stride)`     | 2D max pooling                           |

```cjc
// Convolution example
let signal = Tensor.randn([1, 1, 16])
let kernel = Tensor.randn([1, 1, 3])
let conv_out = Tensor.conv1d(signal, kernel, 1, 0)
print(Tensor.shape(conv_out))
```

---

## Tensor View Operations

| Name                         | Usage                                | Description                              |
|------------------------------|--------------------------------------|------------------------------------------|
| `Tensor.slice`               | `Tensor.slice(t, ranges)`           | Slice tensor along each dimension        |
| `Tensor.view_reshape`        | `Tensor.view_reshape(t, shape)`     | Reshape as a view (no copy if possible)  |
| `Tensor.transpose_last_two`  | `Tensor.transpose_last_two(t)`      | Transpose the last two dimensions        |
| `Tensor.to_contiguous`       | `Tensor.to_contiguous(t)`           | Ensure tensor is contiguous in memory    |

```cjc
// View operations example
let t = Tensor.arange(0.0, 12.0, 1.0)
let m = Tensor.view_reshape(t, [3, 4])
let s = Tensor.slice(m, [[0, 2], [1, 3]])  // rows 0-1, cols 1-2
let mt = Tensor.transpose_last_two(Tensor.view_reshape(t, [2, 2, 3]))
```

---

## Automatic Differentiation -- Forward Mode

| Name       | Usage                    | Description                                     |
|------------|--------------------------|-------------------------------------------------|
| `Dual.new` | `Dual.new(val, deriv)`   | Create a dual number for forward-mode AD        |
| `grad`     | `grad(fn, arg_index)`    | Compute gradient of fn w.r.t. the given argument|

```cjc
// Forward-mode AD example
fn f(x: f64) -> f64 {
    x * x + 2.0 * x
}
let df = grad(f, 0)
print(df(3.0))   // derivative at x=3: 2*3 + 2 = 8
```

---

## Automatic Differentiation -- GradGraph (Reverse Mode)

| Name                   | Usage                           | Description                                  |
|------------------------|---------------------------------|----------------------------------------------|
| `GradGraph.new`        | `GradGraph.new()`               | Create a new computation graph               |
| `GradGraph.parameter`  | `GradGraph.parameter(g, v)`     | Register a trainable parameter               |
| `GradGraph.input`      | `GradGraph.input(g, v)`         | Register an input (non-trainable)            |
| `GradGraph.backward`   | `GradGraph.backward(g)`         | Run backpropagation                          |
| `GradGraph.value`      | `GradGraph.value(n)`            | Get scalar value of a node                   |
| `GradGraph.tensor`     | `GradGraph.tensor(n)`           | Get tensor value of a node                   |
| `GradGraph.grad`       | `GradGraph.grad(n)`             | Get accumulated gradient of a node           |
| `GradGraph.zero_grad`  | `GradGraph.zero_grad(n)`        | Reset gradient to zero                       |
| `GradGraph.set_tensor` | `GradGraph.set_tensor(n, t)`    | Overwrite tensor value of a node             |
| `GradGraph.add`        | `GradGraph.add(g, a, b)`        | Add two nodes                                |
| `GradGraph.sub`        | `GradGraph.sub(g, a, b)`        | Subtract two nodes                           |
| `GradGraph.mul`        | `GradGraph.mul(g, a, b)`        | Multiply two nodes                           |
| `GradGraph.div`        | `GradGraph.div(g, a, b)`        | Divide two nodes                             |
| `GradGraph.neg`        | `GradGraph.neg(g, x)`           | Negate a node                                |
| `GradGraph.matmul`     | `GradGraph.matmul(g, a, b)`     | Matrix multiplication node                   |
| `GradGraph.sum`        | `GradGraph.sum(g, x)`           | Sum-reduction node                           |
| `GradGraph.mean`       | `GradGraph.mean(g, x)`          | Mean-reduction node                          |
| `GradGraph.relu`       | `GradGraph.relu(g, x)`          | ReLU activation node                         |
| `GradGraph.sigmoid`    | `GradGraph.sigmoid(g, x)`       | Sigmoid activation node                      |
| `GradGraph.tanh`       | `GradGraph.tanh(g, x)`          | Tanh activation node                         |
| `GradGraph.sin`        | `GradGraph.sin(g, x)`           | Sine node                                    |
| `GradGraph.cos`        | `GradGraph.cos(g, x)`           | Cosine node                                  |
| `GradGraph.sqrt`       | `GradGraph.sqrt(g, x)`          | Square root node                             |
| `GradGraph.pow`        | `GradGraph.pow(g, x, n)`        | Power node                                   |
| `GradGraph.exp`        | `GradGraph.exp(g, x)`           | Exponential node                             |
| `GradGraph.ln`         | `GradGraph.ln(g, x)`            | Natural logarithm node                       |
| `GradGraph.scalar_mul` | `GradGraph.scalar_mul(g, x, s)` | Scalar multiplication node                   |

```cjc
// Reverse-mode AD example
let g = GradGraph.new()
let x = GradGraph.parameter(g, Tensor.from_vec([3.0], [1]))
let y = GradGraph.parameter(g, Tensor.from_vec([4.0], [1]))
let z = GradGraph.add(g, x, y)
let loss = GradGraph.sum(g, GradGraph.mul(g, z, z))
GradGraph.backward(g)
print(GradGraph.grad(x))  // d(loss)/dx
print(GradGraph.grad(y))  // d(loss)/dy
```

---

## Array & Collection Operations

| Name              | Usage                         | Description                                  |
|-------------------|-------------------------------|----------------------------------------------|
| `len`             | `len(x)`                     | Length of array, string, or collection        |
| `push`            | `push(arr, val)`             | Append value (returns new array)             |
| `sort`            | `sort(arr)`                  | Sort array in ascending order                |
| `array_push`      | `array_push(arr, val)`       | Append value (returns new array)             |
| `array_pop`       | `array_pop(arr)`             | Remove and return last element               |
| `array_contains`  | `array_contains(arr, val)`   | Check if array contains value                |
| `array_reverse`   | `array_reverse(arr)`         | Reverse the array                            |
| `array_flatten`   | `array_flatten(arr)`         | Flatten nested arrays by one level           |
| `array_len`       | `array_len(arr)`             | Length of the array                          |
| `array_slice`     | `array_slice(arr, start, end)` | Extract sub-array [start, end)            |
| `argsort`         | `argsort(arr)`               | Indices that would sort the array            |
| `gather`          | `gather(t, idx)`             | Gather elements by index array               |
| `scatter`         | `scatter(t, idx, v)`         | Scatter values into tensor at indices        |
| `index_select`    | `index_select(t, idx)`       | Select elements by index                     |
| `cat`             | `cat(tensors, ax)`           | Concatenate tensors along axis               |
| `stack`           | `stack(tensors)`             | Stack tensors along a new axis               |
| `topk`            | `topk(t, k)`                 | Return top-k values and indices              |

```cjc
// Array operations example
let arr = [3, 1, 4, 1, 5]
let sorted = sort(arr)           // [1, 1, 3, 4, 5]
let idx = argsort(arr)           // indices for sorted order
let arr2 = array_push(arr, 9)   // [3, 1, 4, 1, 5, 9]
let sub = array_slice(arr, 1, 3) // [1, 4]
print(array_contains(arr, 4))   // true
```

---

## Map Operations

| Name               | Usage                     | Description                            |
|--------------------|---------------------------|----------------------------------------|
| `Map.new`          | `Map.new()`               | Create an empty map                    |
| `Map.insert`       | `Map.insert(m, k, v)`    | Insert key-value pair (returns new map)|
| `Map.get`          | `Map.get(m, k)`          | Get value by key                       |
| `Map.remove`       | `Map.remove(m, k)`       | Remove key (returns new map)           |
| `Map.contains_key` | `Map.contains_key(m, k)` | Check if key exists                    |
| `Map.len`          | `Map.len(m)`             | Number of entries                      |
| `Map.keys`         | `Map.keys(m)`            | Array of all keys                      |
| `Map.values`       | `Map.values(m)`          | Array of all values                    |

```cjc
// Map example
let m = Map.new()
let m = Map.insert(m, "name", "CJC")
let m = Map.insert(m, "version", "0.1")
print(Map.get(m, "name"))         // "CJC"
print(Map.contains_key(m, "version"))  // true
print(Map.keys(m))                // ["name", "version"]
```

---

## Set Operations

| Name            | Usage                  | Description                          |
|-----------------|------------------------|--------------------------------------|
| `Set.new`       | `Set.new()`            | Create an empty set                  |
| `Set.add`       | `Set.add(s, v)`        | Add a value (returns new set)        |
| `Set.contains`  | `Set.contains(s, v)`   | Check if value is in the set         |
| `Set.remove`    | `Set.remove(s, v)`     | Remove a value (returns new set)     |
| `Set.len`       | `Set.len(s)`           | Number of elements                   |
| `Set.to_array`  | `Set.to_array(s)`      | Convert set to array                 |

```cjc
// Set example
let s = Set.new()
let s = Set.add(s, 1)
let s = Set.add(s, 2)
let s = Set.add(s, 1)   // duplicate ignored
print(Set.len(s))        // 2
print(Set.contains(s, 2))  // true
```

---

## String Operations

| Name               | Usage                            | Description                              |
|--------------------|----------------------------------|------------------------------------------|
| `str_detect`       | `str_detect(s, pat)`             | Returns true if pattern is found         |
| `str_extract`      | `str_extract(s, pat)`            | Extract first regex match                |
| `str_extract_all`  | `str_extract_all(s, pat)`        | Extract all regex matches                |
| `str_replace`      | `str_replace(s, pat, rep)`       | Replace first occurrence                 |
| `str_replace_all`  | `str_replace_all(s, pat, rep)`   | Replace all occurrences                  |
| `str_split`        | `str_split(s, pat)`              | Split string by pattern                  |
| `str_count`        | `str_count(s, pat)`              | Count pattern occurrences                |
| `str_trim`         | `str_trim(s)`                    | Trim leading and trailing whitespace     |
| `str_to_upper`     | `str_to_upper(s)`                | Convert to uppercase                     |
| `str_to_lower`     | `str_to_lower(s)`                | Convert to lowercase                     |
| `str_starts`       | `str_starts(s, pre)`             | Check if string starts with prefix       |
| `str_ends`         | `str_ends(s, suf)`               | Check if string ends with suffix         |
| `str_sub`          | `str_sub(s, start, end)`         | Extract substring [start, end)           |
| `str_len`          | `str_len(s)`                     | Length of string in characters            |

```cjc
// String operations example
let s = "  Hello, World!  "
let trimmed = str_trim(s)            // "Hello, World!"
let upper = str_to_upper(trimmed)    // "HELLO, WORLD!"
let parts = str_split(trimmed, ", ") // ["Hello", "World!"]
print(str_detect(trimmed, "World"))  // true
print(str_replace(trimmed, "World", "CJC"))  // "Hello, CJC!"
print(str_len(trimmed))             // 13
```

---

## DateTime

| Name                   | Usage                                    | Description                                |
|------------------------|------------------------------------------|--------------------------------------------|
| `datetime_now`         | `datetime_now()`                         | Current date-time (UTC milliseconds)       |
| `datetime_from_epoch`  | `datetime_from_epoch(ms)`                | DateTime from epoch milliseconds           |
| `datetime_from_parts`  | `datetime_from_parts(y, m, d, h, min, s)` | DateTime from components                |
| `datetime_year`        | `datetime_year(dt)`                      | Extract year                               |
| `datetime_month`       | `datetime_month(dt)`                     | Extract month (1-12)                       |
| `datetime_day`         | `datetime_day(dt)`                       | Extract day of month                       |
| `datetime_hour`        | `datetime_hour(dt)`                      | Extract hour (0-23)                        |
| `datetime_minute`      | `datetime_minute(dt)`                    | Extract minute (0-59)                      |
| `datetime_second`      | `datetime_second(dt)`                    | Extract second (0-59)                      |
| `datetime_diff`        | `datetime_diff(a, b)`                    | Difference in milliseconds (a - b)        |
| `datetime_add_millis`  | `datetime_add_millis(dt, ms)`            | Add milliseconds to a DateTime             |
| `datetime_format`      | `datetime_format(dt, fmt)`               | Format DateTime as string                  |

```cjc
// DateTime example
let now = datetime_now()
let dt = datetime_from_parts(2026, 3, 3, 12, 0, 0)
print(datetime_year(dt))      // 2026
print(datetime_month(dt))     // 3
let later = datetime_add_millis(dt, 3600000)  // +1 hour
print(datetime_format(later, "%Y-%m-%d %H:%M"))
```

---

## I/O

| Name          | Usage                        | Description                              |
|---------------|------------------------------|------------------------------------------|
| `print`       | `print(val)`                 | Print value to stdout                    |
| `file_read`   | `file_read(path)`            | Read entire file as string               |
| `file_write`  | `file_write(path, content)`  | Write string to file                     |
| `file_exists` | `file_exists(path)`          | Check if file exists                     |
| `file_lines`  | `file_lines(path)`           | Read file as array of lines              |
| `read_line`   | `read_line()`                | Read one line from stdin                 |

```cjc
// I/O example
file_write("output.txt", "Hello from CJC!")
let exists = file_exists("output.txt")  // true
let content = file_read("output.txt")   // "Hello from CJC!"
let lines = file_lines("output.txt")    // ["Hello from CJC!"]
print(content)
```

---

## CSV

| Name                | Usage                         | Description                              |
|---------------------|-------------------------------|------------------------------------------|
| `Csv.parse`         | `Csv.parse(path)`             | Parse CSV file into table structure      |
| `Csv.parse_tsv`     | `Csv.parse_tsv(path)`         | Parse TSV (tab-separated) file           |
| `Csv.stream_sum`    | `Csv.stream_sum(path, col)`   | Stream-sum a column without full load    |
| `Csv.stream_minmax` | `Csv.stream_minmax(path, col)`| Stream min/max of a column               |

```cjc
// CSV example
let data = Csv.parse("data.csv")
let total = Csv.stream_sum("data.csv", "revenue")
let bounds = Csv.stream_minmax("data.csv", "temperature")
print(total)
print(bounds)
```

---

## JSON

| Name             | Usage                | Description                              |
|------------------|----------------------|------------------------------------------|
| `json_parse`     | `json_parse(s)`      | Parse JSON string into CJC value         |
| `json_stringify`  | `json_stringify(v)` | Serialize CJC value to JSON string       |

```cjc
// JSON example
let obj = json_parse("{\"name\": \"CJC\", \"version\": 0.1}")
print(obj)
let s = json_stringify(obj)
print(s)
```

---

## Snap Serialization

| Name            | Usage                    | Description                                     |
|-----------------|--------------------------|-------------------------------------------------|
| `snap`          | `snap(val)`              | Serialize any value to a binary blob             |
| `restore`       | `restore(blob)`          | Deserialize blob back to original value          |
| `snap_hash`     | `snap_hash(blob)`        | Compute hash of a snap blob                      |
| `snap_save`     | `snap_save(blob, path)`  | Save snap blob to file                           |
| `snap_load`     | `snap_load(path)`        | Load snap blob from file                         |
| `snap_to_json`  | `snap_to_json(blob)`     | Convert snap blob to JSON representation         |
| `memo_call`     | `memo_call(fn, args)`    | Memoized function call (cache by snap of args)   |

```cjc
// Snap serialization example
let data = [1, 2, 3, 4, 5]
let blob = snap(data)
snap_save(blob, "checkpoint.snap")
let loaded = snap_load("checkpoint.snap")
let recovered = restore(loaded)
print(recovered)  // [1, 2, 3, 4, 5]
```

---

## Bitwise Operations

| Name       | Usage              | Description                          |
|------------|--------------------|--------------------------------------|
| `bit_and`  | `bit_and(x, y)`    | Bitwise AND                          |
| `bit_or`   | `bit_or(x, y)`     | Bitwise OR                           |
| `bit_xor`  | `bit_xor(x, y)`    | Bitwise XOR                          |
| `bit_not`  | `bit_not(x)`       | Bitwise NOT                          |
| `bit_shl`  | `bit_shl(x, n)`    | Shift left by n bits                 |
| `bit_shr`  | `bit_shr(x, n)`    | Shift right by n bits                |
| `popcount` | `popcount(x)`      | Count number of set bits             |

```cjc
// Bitwise operations example
let a = 0b1100
let b = 0b1010
print(bit_and(a, b))   // 0b1000 = 8
print(bit_or(a, b))    // 0b1110 = 14
print(bit_xor(a, b))   // 0b0110 = 6
print(popcount(0b1111)) // 4
print(bit_shl(1, 4))   // 16
```

---

## Cumulative & Window Functions

| Name            | Usage                     | Description                                  |
|-----------------|---------------------------|----------------------------------------------|
| `cumsum`        | `cumsum(arr)`             | Cumulative sum                               |
| `cumprod`       | `cumprod(arr)`            | Cumulative product                           |
| `cummax`        | `cummax(arr)`             | Cumulative maximum                           |
| `cummin`        | `cummin(arr)`             | Cumulative minimum                           |
| `lag`           | `lag(arr, n)`             | Lag values by n positions                    |
| `lead`          | `lead(arr, n)`            | Lead values by n positions                   |
| `rank`          | `rank(arr)`               | Ranks with average ties                      |
| `dense_rank`    | `dense_rank(arr)`         | Dense ranks (no gaps)                        |
| `row_number`    | `row_number(arr)`         | Sequential row numbers                       |
| `percent_rank`  | `percent_rank(arr)`       | Percentile-based rank [0, 1]                 |
| `cume_dist`     | `cume_dist(arr)`          | Cumulative distribution function             |
| `histogram`     | `histogram(arr, bins)`    | Frequency counts across bins                 |
| `ntile`         | `ntile(arr, n)`           | Divide into n equal-sized groups             |
| `window_sum`    | `window_sum(arr, w)`      | Rolling sum with window size w               |
| `window_mean`   | `window_mean(arr, w)`     | Rolling mean with window size w              |
| `window_min`    | `window_min(arr, w)`      | Rolling minimum with window size w           |
| `window_max`    | `window_max(arr, w)`      | Rolling maximum with window size w           |
| `case_when`     | `case_when(conds, vals)`  | SQL-style CASE WHEN conditional mapping      |

```cjc
// Cumulative & window functions example
let data = [1.0, 3.0, 2.0, 5.0, 4.0]
print(cumsum(data))         // [1.0, 4.0, 6.0, 11.0, 15.0]
print(cummax(data))         // [1.0, 3.0, 3.0, 5.0, 5.0]
print(window_mean(data, 3)) // rolling 3-period mean
print(rank(data))           // [1.0, 3.0, 2.0, 5.0, 4.0]
let bins = histogram(data, 3)
print(bins)
```

---

## Complex Numbers

| Name            | Usage                    | Description                              |
|-----------------|--------------------------|------------------------------------------|
| `Complex`       | `Complex(re, im)`        | Create a complex number                  |
| `Complex.re`    | `Complex.re(z)`          | Real part                                |
| `Complex.im`    | `Complex.im(z)`          | Imaginary part                           |
| `Complex.abs`   | `Complex.abs(z)`         | Magnitude (modulus)                      |
| `Complex.norm_sq`| `Complex.norm_sq(z)`    | Squared magnitude                        |
| `Complex.conj`  | `Complex.conj(z)`        | Complex conjugate                        |
| `Complex.is_nan`| `Complex.is_nan(z)`      | True if either part is NaN               |
| `Complex.is_finite` | `Complex.is_finite(z)` | True if both parts are finite          |
| `Complex.add`   | `Complex.add(a, b)`      | Complex addition                         |
| `Complex.sub`   | `Complex.sub(a, b)`      | Complex subtraction                      |
| `Complex.mul`   | `Complex.mul(a, b)`      | Complex multiplication                   |
| `Complex.div`   | `Complex.div(a, b)`      | Complex division                         |
| `Complex.neg`   | `Complex.neg(z)`         | Negate complex number                    |
| `Complex.scale` | `Complex.scale(z, s)`    | Scale by real factor s                   |

```cjc
// Complex numbers example
let z1 = Complex(3.0, 4.0)
let z2 = Complex(1.0, -2.0)
let z3 = Complex.mul(z1, z2)
print(Complex.abs(z1))     // 5.0 (magnitude)
print(Complex.re(z3))      // real part of product
print(Complex.conj(z1))    // 3 - 4i
```

---

## FFT & Signal Processing

| Name             | Usage                 | Description                                   |
|------------------|-----------------------|-----------------------------------------------|
| `rfft`           | `rfft(signal)`        | Real-valued FFT                               |
| `psd`            | `psd(signal)`         | Power spectral density                        |
| `fft_arbitrary`  | `fft_arbitrary(signal)` | FFT for arbitrary-length signals            |
| `fft_2d`         | `fft_2d(signal)`      | 2D FFT                                        |
| `ifft_2d`        | `ifft_2d(signal)`     | Inverse 2D FFT                                |
| `hann`           | `hann(n)`             | Hann window of length n                       |
| `hamming`        | `hamming(n)`          | Hamming window of length n                    |
| `blackman`       | `blackman(n)`         | Blackman window of length n                   |

```cjc
// FFT & signal processing example
let signal = Tensor.from_vec([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0], [8])
let spectrum = rfft(signal)
let power = psd(signal)
let w = hann(8)
print(spectrum)
print(power)
```

---

## Sparse Tensors

| Name                  | Usage                    | Description                              |
|-----------------------|--------------------------|------------------------------------------|
| `SparseCsr.to_dense`  | `SparseCsr.to_dense(s)` | Convert CSR sparse matrix to dense       |
| `SparseCsr.matvec`    | `SparseCsr.matvec(s, v)`| Sparse matrix-vector multiplication      |
| `SparseCoo.to_csr`    | `SparseCoo.to_csr(s)`   | Convert COO sparse format to CSR         |

```cjc
// Sparse tensor example (CSR format)
// Sparse representations for large, mostly-zero matrices
let dense = SparseCsr.to_dense(sparse_mat)
let result = SparseCsr.matvec(sparse_mat, vec)
```

---

## Data DSL / Tidy

| Name                  | Usage                            | Description                                 |
|-----------------------|----------------------------------|---------------------------------------------|
| `col`                 | `col(name)`                      | Reference a column by name                  |
| `desc`                | `desc(name)`                     | Descending sort specification               |
| `asc`                 | `asc(name)`                      | Ascending sort specification                |
| `dexpr_binop`         | `dexpr_binop(op, l, r)`         | Binary operation on column expressions      |
| `tidy_filter`         | `tidy_filter(v, e)`             | Filter rows by expression                   |
| `tidy_select`         | `tidy_select(v, cols)`          | Select specific columns                     |
| `tidy_mask_and`       | `tidy_mask_and(a, b)`           | Combine two boolean masks with AND          |
| `tidy_distinct`       | `tidy_distinct(v, cols)`        | Distinct rows by columns                    |
| `tidy_group_by`       | `tidy_group_by(v, cols)`        | Group by columns                            |
| `tidy_group_by_fast`  | `tidy_group_by_fast(v, cols)`   | Optimized group-by for large datasets       |
| `tidy_ungroup`        | `tidy_ungroup(v)`               | Remove grouping                             |
| `tidy_slice`          | `tidy_slice(v, n)`              | Take first n rows                           |
| `tidy_slice_head`     | `tidy_slice_head(v, n)`         | Take first n rows (alias)                   |
| `tidy_slice_tail`     | `tidy_slice_tail(v, n)`         | Take last n rows                            |
| `tidy_slice_sample`   | `tidy_slice_sample(v, n)`       | Random sample of n rows                     |
| `tidy_semi_join`      | `tidy_semi_join(l, r, cols)`    | Semi-join: keep rows in l matching r        |
| `tidy_anti_join`      | `tidy_anti_join(l, r, cols)`    | Anti-join: keep rows in l not matching r    |
| `tidy_relocate`       | `tidy_relocate(v, cols)`        | Reorder columns                             |
| `tidy_drop_cols`      | `tidy_drop_cols(v, cols)`       | Drop specified columns                      |
| `tidy_nrows`          | `tidy_nrows(v)`                 | Number of rows                              |
| `tidy_ncols`          | `tidy_ncols(v)`                 | Number of columns                           |
| `tidy_column_names`   | `tidy_column_names(v)`          | Array of column names                       |
| `tidy_ngroups`        | `tidy_ngroups(v)`               | Number of groups                            |
| `tidy_count`          | `tidy_count(g)`                 | Count rows per group                        |
| `tidy_sum`            | `tidy_sum(col)`                 | Sum aggregation expression                  |
| `tidy_mean`           | `tidy_mean(col)`                | Mean aggregation expression                 |
| `tidy_min`            | `tidy_min(col)`                 | Min aggregation expression                  |
| `tidy_max`            | `tidy_max(col)`                 | Max aggregation expression                  |
| `tidy_first`          | `tidy_first(col)`               | First value aggregation                     |
| `tidy_last`           | `tidy_last(col)`                | Last value aggregation                      |
| `fct_collapse`        | `fct_collapse(f, m)`            | Collapse factor levels via mapping          |

```cjc
// Tidy data DSL example
let data = Csv.parse("sales.csv")
let filtered = tidy_filter(data, dexpr_binop(">", col("revenue"), 1000.0))
let selected = tidy_select(filtered, ["region", "revenue"])
let grouped = tidy_group_by(selected, ["region"])
let summary = tidy_sum("revenue")
print(tidy_nrows(filtered))
print(tidy_column_names(data))
```

---

## Assertions & Miscellaneous

| Name                 | Usage                           | Description                                 |
|----------------------|---------------------------------|---------------------------------------------|
| `assert`             | `assert(cond, msg)`             | Assert condition is true, fail with msg     |
| `assert_eq`          | `assert_eq(a, b, msg)`          | Assert two values are equal                 |
| `gc_live_count`      | `gc_live_count()`               | Number of live GC-tracked objects           |
| `clock`              | `clock()`                       | Current time in seconds (monotonic)         |
| `categorical_sample` | `categorical_sample(logits)`    | Sample index from categorical distribution  |

```cjc
// Assertions & misc example
assert(1 + 1 == 2, "math is broken")
assert_eq(abs(-5), 5, "abs failed")
let t0 = clock()
// ... some computation ...
let elapsed = clock() - t0
print(elapsed)

let logits = [0.1, 0.7, 0.2]
let idx = categorical_sample(logits)  // likely 1
```

---

## Scratchpad & KV-Cache

### Scratchpad

| Name                      | Usage                         | Description                               |
|---------------------------|-------------------------------|-------------------------------------------|
| `Scratchpad.new`          | `Scratchpad.new()`            | Create a new scratchpad buffer            |
| `Scratchpad.append`       | `Scratchpad.append(sp, s)`    | Append a scalar value                     |
| `Scratchpad.append_tensor`| `Scratchpad.append_tensor(sp, t)` | Append a tensor's data               |
| `Scratchpad.as_tensor`    | `Scratchpad.as_tensor(sp)`    | View scratchpad contents as a tensor      |
| `Scratchpad.len`          | `Scratchpad.len(sp)`          | Number of elements stored                 |
| `Scratchpad.capacity`     | `Scratchpad.capacity(sp)`     | Current buffer capacity                   |
| `Scratchpad.dim`          | `Scratchpad.dim(sp)`          | Dimensionality of stored data             |
| `Scratchpad.clear`        | `Scratchpad.clear(sp)`        | Clear all stored data                     |
| `Scratchpad.is_empty`     | `Scratchpad.is_empty(sp)`     | True if scratchpad has no data            |

### PagedKvCache

| Name                       | Usage                                  | Description                                |
|----------------------------|----------------------------------------|--------------------------------------------|
| `PagedKvCache.new`         | `PagedKvCache.new(bs, nb, dim)`        | Create paged KV cache                      |
| `PagedKvCache.append`      | `PagedKvCache.append(c, tok, kv)`      | Append token with key-value pair           |
| `PagedKvCache.append_tensor`| `PagedKvCache.append_tensor(c, t)`    | Append tensor data to cache                |
| `PagedKvCache.as_tensor`   | `PagedKvCache.as_tensor(c)`            | View cache contents as tensor              |
| `PagedKvCache.clear`       | `PagedKvCache.clear(c)`               | Clear the cache                            |
| `PagedKvCache.len`         | `PagedKvCache.len(c)`                 | Number of cached tokens                    |
| `PagedKvCache.is_empty`    | `PagedKvCache.is_empty(c)`            | True if cache is empty                     |
| `PagedKvCache.max_tokens`  | `PagedKvCache.max_tokens(c)`          | Maximum token capacity                     |
| `PagedKvCache.dim`         | `PagedKvCache.dim(c)`                 | Dimensionality of cached vectors           |
| `PagedKvCache.num_blocks`  | `PagedKvCache.num_blocks(c)`          | Total number of memory blocks              |
| `PagedKvCache.blocks_in_use`| `PagedKvCache.blocks_in_use(c)`      | Number of blocks currently in use          |
| `PagedKvCache.get_token`   | `PagedKvCache.get_token(c, i)`        | Get cached data for token at index i       |

```cjc
// Scratchpad example
let sp = Scratchpad.new()
Scratchpad.append(sp, 1.0)
Scratchpad.append(sp, 2.0)
Scratchpad.append(sp, 3.0)
let t = Scratchpad.as_tensor(sp)
print(Scratchpad.len(sp))   // 3

// KV-Cache example
let cache = PagedKvCache.new(1, 4, 64)
print(PagedKvCache.max_tokens(cache))
print(PagedKvCache.is_empty(cache))  // true
```

---

## Float16 / BFloat16

| Name           | Usage               | Description                                  |
|----------------|---------------------|----------------------------------------------|
| `f16_to_f64`   | `f16_to_f64(x)`    | Convert float16 to f64                       |
| `f16_to_f32`   | `f16_to_f32(x)`    | Convert float16 to f32                       |
| `f32_to_f16`   | `f32_to_f16(x)`    | Convert f32 to float16                       |
| `f64_to_f16`   | `f64_to_f16(x)`    | Convert f64 to float16                       |
| `F16.to_f64`   | `F16.to_f64(x)`    | Method-style float16 to f64 conversion       |
| `F16.to_f32`   | `F16.to_f32(x)`    | Method-style float16 to f32 conversion       |
| `bf16_to_f32`  | `bf16_to_f32(x)`   | Convert bfloat16 to f32                      |
| `f32_to_bf16`  | `f32_to_bf16(x)`   | Convert f32 to bfloat16                      |

```cjc
// Float16 conversion example
let val = 3.14
let half = f64_to_f16(val)
let back = f16_to_f64(half)
print(back)   // ~3.14 (with float16 precision loss)

let bf = f32_to_bf16(1.5)
let restored = bf16_to_f32(bf)
print(restored)  // 1.5
```

---

## Buffer & Memory

| Name                           | Usage                              | Description                                   |
|--------------------------------|------------------------------------|-----------------------------------------------|
| `Buffer.alloc`                 | `Buffer.alloc(size)`               | Allocate a raw byte buffer of given size      |
| `AlignedByteSlice.from_bytes`  | `AlignedByteSlice.from_bytes(b)`   | Create aligned byte slice from raw bytes      |
| `AlignedByteSlice.as_tensor`   | `AlignedByteSlice.as_tensor(s)`    | Reinterpret aligned bytes as a tensor         |
| `AlignedByteSlice.was_realigned`| `AlignedByteSlice.was_realigned(s)`| True if data was copied to achieve alignment |
| `AlignedByteSlice.len`         | `AlignedByteSlice.len(s)`          | Length in bytes                               |
| `AlignedByteSlice.is_empty`    | `AlignedByteSlice.is_empty(s)`     | True if slice is empty                        |
| `ByteSlice.as_tensor`          | `ByteSlice.as_tensor(bs)`          | Reinterpret byte slice as a tensor            |

```cjc
// Buffer example
let buf = Buffer.alloc(1024)
let aligned = AlignedByteSlice.from_bytes(buf)
print(AlignedByteSlice.len(aligned))          // 1024
print(AlignedByteSlice.was_realigned(aligned)) // false or true
```

---

## Broadcasting Builtins

Explicit element-wise function application over tensors. These complement the
implicit broadcasting in tensor arithmetic (`t1 + t2`, `t * 2.0`).

### `broadcast(fn_name, tensor)` -- Unary

Applies a named unary function element-wise to every element of a tensor.
Returns a new tensor of the same shape.

| `fn_name` | Operation | Notes |
|-----------|-----------|-------|
| `"sin"` | sine | radians |
| `"cos"` | cosine | radians |
| `"tan"` | tangent | radians |
| `"asin"` | arcsine | returns radians |
| `"acos"` | arccosine | returns radians |
| `"atan"` | arctangent | returns radians |
| `"exp"` | e^x | |
| `"ln"` | natural log | alias: `"log"` |
| `"log"` | natural log | alias for `"ln"` |
| `"log2"` | log base 2 | |
| `"log10"` | log base 10 | |
| `"log1p"` | ln(1+x) | accurate near 0 |
| `"expm1"` | e^x - 1 | accurate near 0 |
| `"sqrt"` | square root | |
| `"abs"` | absolute value | |
| `"floor"` | floor | |
| `"ceil"` | ceiling | |
| `"round"` | round to nearest | |
| `"sigmoid"` | 1/(1+e^(-x)) | logistic sigmoid |
| `"relu"` | max(0, x) | rectified linear unit |
| `"tanh"` | hyperbolic tangent | |
| `"neg"` | -x | negation |
| `"sign"` | sign function | -1, 0, or 1 |

```cjc
let t = Tensor.from_vec([0.0, 1.5708, 3.1416], [3]);
print(broadcast("sin", t));   // [0, 1, ~0]
print(broadcast("relu", Tensor.from_vec([-1.0, 0.0, 1.0, 2.0], [4])));  // [0, 0, 1, 2]
```

### `broadcast2(fn_name, t1, t2)` -- Binary

Applies a named binary function element-wise to two tensors (with NumPy-style
shape broadcasting). Returns a new tensor.

| `fn_name` | Operation |
|-----------|-----------|
| `"add"` | t1 + t2 |
| `"sub"` | t1 - t2 |
| `"mul"` | t1 * t2 |
| `"div"` | t1 / t2 |
| `"pow"` | t1 ^ t2 |
| `"min"` | element-wise min |
| `"max"` | element-wise max |
| `"atan2"` | atan2(t1, t2) |
| `"hypot"` | sqrt(t1^2 + t2^2) |

```cjc
let a = Tensor.from_vec([2.0, 3.0, 4.0], [3]);
let b = Tensor.from_vec([2.0, 2.0, 2.0], [3]);
print(broadcast2("pow", a, b));    // [4, 9, 16]
print(broadcast2("hypot", a, b));  // [~2.83, ~3.61, ~4.47]
```

**Effect:** Both `broadcast` and `broadcast2` have `ALLOC` effect (they allocate
new tensors). They are allowed in `/ alloc` functions but not `/ pure`.

---

*CJC v0.1 -- Built-in Function Reference*
*Total: 300+ functions across 30+ domains*
