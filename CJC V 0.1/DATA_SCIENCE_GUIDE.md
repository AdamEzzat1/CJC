# Data Science Guide

## Overview

CJC provides a comprehensive data science toolkit: tidy-style data wrangling,
statistics, string operations, regex, CSV/JSON I/O, and datetime handling.

## Statistics

### Descriptive Statistics

```
let data = [23.1, 25.4, 22.8, 24.5, 26.1, 23.9, 24.2, 25.0];

print(mean(data));           // arithmetic mean
print(median(data));         // median
print(sd(data));             // standard deviation
print(variance(data));       // variance
print(se(data));             // standard error
print(quantile(data, 0.25)); // first quartile
print(iqr(data));            // interquartile range
print(skewness(data));       // skewness
print(kurtosis(data));       // kurtosis
print(mad(data));            // median absolute deviation
```

### Correlations

```
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [2.1, 3.9, 6.2, 7.8, 10.1];

print(cor(x, y));            // Pearson correlation
print(cov(x, y));            // covariance
print(spearman_cor(x, y));   // Spearman rank correlation
print(kendall_cor(x, y));    // Kendall tau
```

### Distribution Functions

```
// Standard normal
print(normal_cdf(1.96));    // ~0.975
print(normal_pdf(0.0));     // ~0.399
print(normal_ppf(0.975));   // ~1.96

// Student's t
print(t_cdf(2.0, 10));      // t-distribution CDF, df=10
print(t_ppf(0.975, 10));    // quantile

// Chi-squared, F, Beta, Gamma, Exponential, Weibull, Binomial, Poisson
// all follow the same pattern: _cdf(x, params), _pdf(x, params), _ppf(p, params)
```

### Hypothesis Testing

```
let sample = [23.1, 25.4, 22.8, 24.5, 26.1, 23.9, 24.2, 25.0];

// One-sample t-test: is the mean different from 24?
print(t_test(sample, 24.0));

// Two-sample t-test
let group_a = [23.1, 25.4, 22.8, 24.5];
let group_b = [26.1, 23.9, 24.2, 25.0];
print(t_test_two_sample(group_a, group_b));

// Paired t-test
print(t_test_paired(group_a, group_b));

// Chi-squared goodness-of-fit
let observed = [20.0, 30.0, 50.0];
let expected = [33.3, 33.3, 33.4];
print(chi_squared_test(observed, expected));

// One-way ANOVA
print(anova_oneway([group_a, group_b]));

// Multiple comparison correction
let pvals = [0.01, 0.04, 0.03, 0.20];
print(bonferroni(pvals, 0.05));
print(fdr_bh(pvals, 0.05));
```

### Regression

```
// Linear regression: lm(X_flat, y, n, p)
// X_flat is the predictor matrix in row-major order (n rows, p columns)
// lm() AUTO-ADDS an intercept column — output has p+1 coefficients
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = [2.1, 4.0, 5.8, 8.1, 9.9, 12.2, 13.8, 16.1, 18.0, 20.1];
let result = lm(x, y, 10, 1);  // n=10 observations, p=1 predictor

// Result is a struct with fields:
print(result.coefficients);    // [intercept, slope] — p+1 values
print(result.r_squared);       // R² goodness of fit (0 to 1)
print(result.residuals);       // residuals (length n)
print(result.std_errors);      // standard errors
print(result.t_values);        // t-statistics
print(result.p_values);        // p-values
print(result.f_statistic);     // F-statistic

// Requirements: n > p+1 (otherwise rank-deficiency error)
// Uses QR decomposition internally for numerical stability
```

### Broadcasting for Element-wise Math

```
// Apply a function element-wise to every element of a tensor
let t = Tensor.from_vec([0.0, 1.0, 4.0, 9.0], [4]);
print(broadcast("sqrt", t));    // [0, 1, 2, 3]
print(broadcast("exp", t));     // element-wise e^x

// Binary broadcast: apply a binary op to two tensors
let a = Tensor.from_vec([2.0, 3.0, 4.0], [3]);
let b = Tensor.from_vec([2.0, 2.0, 2.0], [3]);
print(broadcast2("pow", a, b)); // [4, 9, 16]

// Supports 23 unary functions (sin, cos, sqrt, relu, sigmoid, ...)
// and 9 binary functions (add, sub, mul, div, pow, min, max, atan2, hypot)
```

## Cumulative and Window Functions

```
let arr = [1.0, 2.0, 3.0, 4.0, 5.0];

// Cumulative operations
print(cumsum(arr));          // [1, 3, 6, 10, 15]
print(cumprod(arr));         // [1, 2, 6, 24, 120]
print(cummax(arr));          // [1, 2, 3, 4, 5]
print(cummin(arr));          // [1, 1, 1, 1, 1]

// Lag and lead
print(lag(arr, 1));          // [NaN, 1, 2, 3, 4]
print(lead(arr, 1));         // [2, 3, 4, 5, NaN]

// Ranking
print(rank(arr));            // [1, 2, 3, 4, 5]
print(dense_rank(arr));      // [1, 2, 3, 4, 5]
print(row_number(arr));      // [1, 2, 3, 4, 5]

// Rolling/window operations
print(window_sum(arr, 3));   // rolling sum with window=3
print(window_mean(arr, 3));  // rolling mean
print(window_min(arr, 3));   // rolling min
print(window_max(arr, 3));   // rolling max

// Binning
print(ntile(arr, 3));        // assign to 3 equal groups
print(histogram(arr, 5));    // 5-bin histogram
```

## String Operations

CJC provides 14 string functions inspired by R's stringr package:

```
let s = "Hello, World! Hello again.";

// Queries (pure)
print(str_detect(s, "World"));       // true
print(str_count(s, "Hello"));        // 2
print(str_starts(s, "Hello"));       // true
print(str_ends(s, "again."));        // true
print(str_len(s));                   // 26

// Extraction
print(str_extract(s, "[A-Z][a-z]+"));      // "Hello" (first match)
print(str_extract_all(s, "[A-Z][a-z]+")); // ["Hello", "World", "Hello"]

// Transformation
print(str_replace(s, "Hello", "Hi"));       // "Hi, World! Hello again."
print(str_replace_all(s, "Hello", "Hi"));   // "Hi, World! Hi again."
print(str_split(s, " "));                   // ["Hello,", "World!", ...]
print(str_trim("  spaces  "));              // "spaces"
print(str_to_upper("hello"));               // "HELLO"
print(str_to_lower("HELLO"));               // "hello"
print(str_sub(s, 0, 5));                    // "Hello"
```

## Regex

CJC supports regex literals and match operators:

```
// Regex literal syntax
let pattern = /\d+/;
let email_pattern = /[\w.]+@[\w.]+/i;    // case-insensitive

// Match operators
let text = "Order #12345";
if text ~= /\d+/ {
    print("contains numbers");
}

if text !~ /^[A-Z]+$/ {
    print("not all uppercase");
}

// Use with string functions
print(str_detect("abc123", /\d+/));     // true
print(str_extract("abc123", /\d+/));    // "123"
```

Regex flags: `i` (case-insensitive), `g` (global), `m` (multiline),
`s` (dotall), `x` (extended/verbose).

## DateTime Operations

```
// Current time
let now = datetime_now();

// From epoch (milliseconds)
let dt = datetime_from_epoch(1700000000000);

// From parts
let dt2 = datetime_from_parts(2024, 1, 15, 10, 30, 0);

// Extract components
print(datetime_year(dt2));     // 2024
print(datetime_month(dt2));    // 1
print(datetime_day(dt2));      // 15
print(datetime_hour(dt2));     // 10
print(datetime_minute(dt2));   // 30
print(datetime_second(dt2));   // 0

// Arithmetic
let diff = datetime_diff(now, dt2);    // milliseconds difference
let later = datetime_add_millis(dt2, 3600000); // add 1 hour

// Formatting
print(datetime_format(dt2, "%Y-%m-%d %H:%M:%S"));
```

## CSV I/O

```
// Parse a CSV file
let df = Csv.parse("data.csv");

// Parse a TSV file
let tsv_df = Csv.parse_tsv("data.tsv");

// Streaming operations (for large files)
let total = Csv.stream_sum("data.csv", "revenue");
let bounds = Csv.stream_minmax("data.csv", "temperature");
```

## JSON

```
// Parse JSON string to value
let data = json_parse("{\"name\": \"Alice\", \"age\": 30}");

// Convert value to JSON string
let json_str = json_stringify(data);
print(json_str);
```

## Tidy Data DSL

CJC provides R/dplyr-inspired tidy data operations. These work on
view-based projections (TidyView) that are allocation-efficient.

### Column References

```
let name_col = col("name");
let desc_score = desc("score");    // descending order
let asc_age = asc("age");         // ascending order
```

### Core Verbs

```
// Filter rows
let filtered = tidy_filter(view, predicate);

// Select columns
let selected = tidy_select(view, cols);

// Distinct rows
let unique = tidy_distinct(view, cols);

// Drop columns
let trimmed = tidy_drop_cols(view, cols);

// Reorder columns
let reordered = tidy_relocate(view, cols);
```

### Grouping and Aggregation

```
// Group by column
let grouped = tidy_group_by(view, [col("category")]);

// Aggregation functions
let counts = tidy_count(grouped);
let totals = tidy_sum(col("revenue"));
let averages = tidy_mean(col("price"));
let mins = tidy_min(col("temperature"));
let maxes = tidy_max(col("temperature"));
let firsts = tidy_first(col("name"));
let lasts = tidy_last(col("name"));

// Ungroup
let ungrouped = tidy_ungroup(grouped);
```

### Joins

```
// Semi join: keep rows in left that match right
let semi = tidy_semi_join(left, right, [col("id")]);

// Anti join: keep rows in left that DON'T match right
let anti = tidy_anti_join(left, right, [col("id")]);
```

### Slicing

```
let first5 = tidy_slice_head(view, 5);
let last5 = tidy_slice_tail(view, 5);
let sample = tidy_slice_sample(view, 10);
```

### View Queries

```
let nrows = tidy_nrows(view);
let ncols = tidy_ncols(view);
let names = tidy_column_names(view);
let ngroups = tidy_ngroups(grouped_view);
```

## File I/O

```
// Read entire file
let content = file_read("data.txt");

// Write to file
file_write("output.txt", "Hello, World!");

// Check existence
if file_exists("data.txt") {
    print("file found");
}

// Read lines as array
let lines = file_lines("data.txt");
for i in 0..len(lines) {
    print(lines[i]);
}
```
