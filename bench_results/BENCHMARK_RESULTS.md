═══════════════════════════════════════════════════════════════
  CJC v0.1.3 — Full CLI Benchmark Suite
  Sun Apr  5 22:05:57 PDT 2026
═══════════════════════════════════════════════════════════════

## System Info
- Binary size: 8.7M
- Platform: MINGW64_NT-10.0-26200 x86_64
- CPU: 11th Gen Intel(R) Core(TM) i7-11390H @ 3.40GHz

## 1. Core Commands

| Command | Scale | Time (ms) | Deterministic | Flags |
|---------|-------|-----------|---------------|-------|
| `cjc lex` | 100k | 87 | DETERMINISTIC | -- |
| `cjc lex` | 100k | 102 | DETERMINISTIC | `--no-color` |
| `cjc parse` | 100k | 102 | DETERMINISTIC | -- |
| `cjc parse` | 100k | 91 | DETERMINISTIC | `--no-color` |
| `cjc run` | 100k | 6673 | DETERMINISTIC | `--seed 42` (eval) |
| `cjc run` | 100k | 6843 | DETERMINISTIC | `--seed 42 --time` |
| `cjc run --mir-opt` | 100k | 2257 | DETERMINISTIC | `--seed 42 --mir-opt` |
| `cjc run` | 100k | 6520 | DETERMINISTIC | `--format json` |
| `cjc run` | 100k | 6636 | DETERMINISTIC | `--format csv` |
| `cjc eval` | expr | 99 | DETERMINISTIC | `'1 + 2 * 3'` |
| `cjc lex` | 1m | 85 | DETERMINISTIC | -- |
| `cjc lex` | 1m | 80 | DETERMINISTIC | `--no-color` |
| `cjc parse` | 1m | 78 | DETERMINISTIC | -- |
| `cjc parse` | 1m | 69 | DETERMINISTIC | `--no-color` |
| `cjc run` | 1m | 8304 | DETERMINISTIC | `--seed 42` (eval) |
| `cjc run` | 1m | 7286 | DETERMINISTIC | `--seed 42 --time` |
| `cjc run --mir-opt` | 1m | 3732 | DETERMINISTIC | `--seed 42 --mir-opt` |
| `cjc run` | 1m | 7445 | DETERMINISTIC | `--format json` |
| `cjc run` | 1m | 7534 | DETERMINISTIC | `--format csv` |
| `cjc eval` | expr | 69 | DETERMINISTIC | `'1 + 2 * 3'` |

## 2. Data & Pipeline Commands

| Command | Scale | Time (ms) | Deterministic | Flags |
|---------|-------|-----------|---------------|-------|
| `cjc flow` | 100k | 128 | DETERMINISTIC | -- |
| `cjc flow` | 100k | 177 | DETERMINISTIC | `--op sum,mean,min,max` |
| `cjc flow` | 100k | 157 | DETERMINISTIC | `--op sum,mean,var,std,count` |
| `cjc flow` | 100k | 155 | DETERMINISTIC | `--columns sensor_a,temperature` |
| `cjc flow` | 100k | 121 | DETERMINISTIC | `--json` |
| `cjc flow` | 100k | 198 | DETERMINISTIC | `--verify` |
| `cjc flow` | 100k | 191 | DETERMINISTIC | `--top 3` |
| `cjc flow` | 100k | 196 | DETERMINISTIC | `--precision 10` |
| `cjc schema` | 100k | 203 | DETERMINISTIC | -- |
| `cjc schema` | 100k | 294 | DETERMINISTIC | `--full` |
| `cjc schema` | 100k | 235 | DETERMINISTIC | `--json` |
| `cjc schema` | 100k | 88 | DETERMINISTIC | `--sample 1000` |
| `cjc inspect` | 100k | 176 | DETERMINISTIC | -- |
| `cjc inspect` | 100k | 803 | DETERMINISTIC | `--deep` |
| `cjc inspect` | 100k | 211 | DETERMINISTIC | `--schema-only` |
| `cjc inspect` | 100k | 104 | DETERMINISTIC | `--header-only` |
| `cjc inspect` | 100k | 213 | DETERMINISTIC | `--json` |
| `cjc inspect` | 100k | 220 | DETERMINISTIC | `--hash` |
| `cjc patch` | 100k | 69 | DETERMINISTIC | `--nan-fill 0 --dry-run` |
| `cjc patch` | 100k | 68 | DETERMINISTIC | `--drop category --dry-run` |
| `cjc patch` | 100k | 69 | DETERMINISTIC | `--rename label status --dry-run` |
| `cjc patch` | 100k | 82 | DETERMINISTIC | `--nan-fill 0 --plan` |
| `cjc flow` | 1m | 956 | DETERMINISTIC | -- |
| `cjc flow` | 1m | 805 | DETERMINISTIC | `--op sum,mean,min,max` |
| `cjc flow` | 1m | 731 | DETERMINISTIC | `--op sum,mean,var,std,count` |
| `cjc flow` | 1m | 868 | DETERMINISTIC | `--columns sensor_a,temperature` |
| `cjc flow` | 1m | 809 | DETERMINISTIC | `--json` |
| `cjc flow` | 1m | 1483 | DETERMINISTIC | `--verify` |
| `cjc flow` | 1m | 837 | DETERMINISTIC | `--top 3` |
| `cjc flow` | 1m | 796 | DETERMINISTIC | `--precision 10` |
| `cjc schema` | 1m | 1659 | DETERMINISTIC | -- |
| `cjc schema` | 1m | 1653 | DETERMINISTIC | `--full` |
| `cjc schema` | 1m | 1416 | DETERMINISTIC | `--json` |
| `cjc schema` | 1m | 205 | DETERMINISTIC | `--sample 1000` |
| `cjc inspect` | 1m | 1199 | DETERMINISTIC | -- |
| `cjc inspect` | 1m | 7267 | DETERMINISTIC | `--deep` |
| `cjc inspect` | 1m | 1160 | DETERMINISTIC | `--schema-only` |
| `cjc inspect` | 1m | 206 | DETERMINISTIC | `--header-only` |
| `cjc inspect` | 1m | 1186 | DETERMINISTIC | `--json` |
| `cjc inspect` | 1m | 2008 | DETERMINISTIC | `--hash` |
| `cjc patch` | 1m | 104 | DETERMINISTIC | `--nan-fill 0 --dry-run` |
| `cjc patch` | 1m | 110 | DETERMINISTIC | `--drop category --dry-run` |
| `cjc patch` | 1m | 113 | DETERMINISTIC | `--rename label status --dry-run` |
| `cjc patch` | 1m | 101 | DETERMINISTIC | `--nan-fill 0 --plan` |
| `cjc drift` | 100k | 409 | DETERMINISTIC | `--csv` (identical files) |
| `cjc drift` | 100k | 269 | DETERMINISTIC | `--csv --summary-only` (diff sizes) |
| `cjc drift` | 100k | 612 | DETERMINISTIC | `--csv --json` |

## 3. Compiler Visibility & Analysis

| Command | Scale | Time (ms) | Deterministic | Flags |
|---------|-------|-----------|---------------|-------|
| `cjc emit` | 100k | 75 | DETERMINISTIC | `--stage ast` |
| `cjc emit` | 100k | 87 | DETERMINISTIC | `--stage hir` |
| `cjc emit` | 100k | 103 | DETERMINISTIC | `--stage mir` |
| `cjc emit` | 100k | 69 | DETERMINISTIC | `--stage mir --opt` |
| `cjc emit` | 100k | 84 | DETERMINISTIC | `--stage mir --diff` |
| `cjc explain` | 100k | 76 | DETERMINISTIC | -- |
| `cjc explain` | 100k | 140 | DETERMINISTIC | `--verbose` |
| `cjc trace` | 100k | 6861 | NON-DETERMINISTIC | `--seed 42` |
| `cjc trace` | 100k | 6436 | NON-DETERMINISTIC | `--seed 42 --ast` |
| `cjc trace` | 100k | 6585 | NON-DETERMINISTIC | `--seed 42 --verbose` |
| `cjc trace` | 100k | 6574 | NON-DETERMINISTIC | `--seed 42 --json` |
| `cjc audit` | 100k | 75 | DETERMINISTIC | -- |
| `cjc audit` | 100k | 78 | DETERMINISTIC | `--verbose` |
| `cjc audit` | 100k | 82 | DETERMINISTIC | `--json` |
| `cjc precision` | 100k | 6574 | DETERMINISTIC | `--seed 42` |
| `cjc precision` | 100k | 6402 | DETERMINISTIC | `--seed 42 --summary-only` |
| `cjc precision` | 100k | 6850 | DETERMINISTIC | `--seed 42 --json` |
| `cjc nogc` | 100k | 75 | DETERMINISTIC | -- |
| `cjc nogc` | 100k | 76 | DETERMINISTIC | `--verbose` |
| `cjc gc` | 100k | 19580 | NON-DETERMINISTIC | `--seed 42` |
| `cjc gc` | 100k | 19876 | NON-DETERMINISTIC | `--seed 42 --verbose` |
| `cjc gc` | 100k | 19326 | NON-DETERMINISTIC | `--seed 42 --json` |
| `cjc mem` | 100k | 6483 | NON-DETERMINISTIC | `--seed 42` |
| `cjc mem` | 100k | 6645 | NON-DETERMINISTIC | `--seed 42 --verbose` |
| `cjc mem` | 100k | 2457 | NON-DETERMINISTIC | `--seed 42 --mir` |
| `cjc mem` | 100k | 6834 | NON-DETERMINISTIC | `--seed 42 --json` |
| `cjc parity` | 100k | 9040 | DETERMINISTIC | `--seed 42` |
| `cjc parity` | 100k | 9405 | DETERMINISTIC | `--seed 42 --verbose` |
| `cjc parity` | 100k | 9279 | DETERMINISTIC | `--seed 42 --json` |
| `cjc proof` | 100k | 20068 | DETERMINISTIC | `--seed 42 --runs 3` |
| `cjc proof` | 100k | 19417 | DETERMINISTIC | `--seed 42 --runs 3 --verbose` |
| `cjc proof` | 100k | 19323 | DETERMINISTIC | `--seed 42 --runs 3 --hash-output` |
| `cjc proof` | 100k | 19675 | DETERMINISTIC | `--seed 42 --runs 3 --json` |
| `cjc proof` | 100k | 26767 | DETERMINISTIC | `--seed 42 --executor both` |
| `cjc bench` | 100k | 26939 | NON-DETERMINISTIC | `--seed 42 --runs 3` |
| `cjc bench` | 100k | 9994 | NON-DETERMINISTIC | `--seed 42 --runs 3 --mir` |
| `cjc bench` | 100k | 36710 | NON-DETERMINISTIC | `--compare eval` |
| `cjc bench` | 100k | 26110 | NON-DETERMINISTIC | `--seed 42 --runs 3 --json` |
| `cjc bench` | 100k | 26486 | NON-DETERMINISTIC | `--seed 42 --runs 3 --markdown` |
| `cjc lock` | 100k | 6880 | DETERMINISTIC | `--seed 42` |
| `cjc lock` | 100k | 7006 | DETERMINISTIC | `--seed 42 --json` |
| `cjc emit` | 1m | 77 | DETERMINISTIC | `--stage ast` |
| `cjc emit` | 1m | 94 | DETERMINISTIC | `--stage hir` |
| `cjc emit` | 1m | 82 | DETERMINISTIC | `--stage mir` |
| `cjc emit` | 1m | 97 | DETERMINISTIC | `--stage mir --opt` |
| `cjc emit` | 1m | 103 | DETERMINISTIC | `--stage mir --diff` |
| `cjc explain` | 1m | 107 | DETERMINISTIC | -- |
| `cjc explain` | 1m | 180 | DETERMINISTIC | `--verbose` |
| `cjc trace` | 1m | 7410 | NON-DETERMINISTIC | `--seed 42` |
| `cjc trace` | 1m | 7463 | NON-DETERMINISTIC | `--seed 42 --ast` |
| `cjc trace` | 1m | 7696 | NON-DETERMINISTIC | `--seed 42 --verbose` |
| `cjc trace` | 1m | 7589 | NON-DETERMINISTIC | `--seed 42 --json` |
| `cjc audit` | 1m | 73 | DETERMINISTIC | -- |
| `cjc audit` | 1m | 82 | DETERMINISTIC | `--verbose` |
| `cjc audit` | 1m | 78 | DETERMINISTIC | `--json` |
| `cjc precision` | 1m | 7629 | DETERMINISTIC | `--seed 42` |
| `cjc precision` | 1m | 7555 | DETERMINISTIC | `--seed 42 --summary-only` |
| `cjc precision` | 1m | 7627 | DETERMINISTIC | `--seed 42 --json` |
| `cjc nogc` | 1m | 69 | DETERMINISTIC | -- |
| `cjc nogc` | 1m | 109 | DETERMINISTIC | `--verbose` |
| `cjc gc` | 1m | 22414 | NON-DETERMINISTIC | `--seed 42` |
| `cjc gc` | 1m | 22523 | NON-DETERMINISTIC | `--seed 42 --verbose` |
| `cjc gc` | 1m | 22204 | NON-DETERMINISTIC | `--seed 42 --json` |
| `cjc mem` | 1m | 8735 | NON-DETERMINISTIC | `--seed 42` |
| `cjc mem` | 1m | 7942 | NON-DETERMINISTIC | `--seed 42 --verbose` |
| `cjc mem` | 1m | 4010 | NON-DETERMINISTIC | `--seed 42 --mir` |
| `cjc mem` | 1m | 7678 | NON-DETERMINISTIC | `--seed 42 --json` |
| `cjc parity` | 1m | 11417 | DETERMINISTIC | `--seed 42` |
| `cjc parity` | 1m | 11300 | DETERMINISTIC | `--seed 42 --verbose` |
| `cjc parity` | 1m | 10921 | DETERMINISTIC | `--seed 42 --json` |
| `cjc proof` | 1m | 21948 | DETERMINISTIC | `--seed 42 --runs 3` |
| `cjc proof` | 1m | 21891 | DETERMINISTIC | `--seed 42 --runs 3 --verbose` |
| `cjc proof` | 1m | 23753 | DETERMINISTIC | `--seed 42 --runs 3 --hash-output` |
| `cjc proof` | 1m | 21864 | DETERMINISTIC | `--seed 42 --runs 3 --json` |
| `cjc proof` | 1m | 31968 | DETERMINISTIC | `--seed 42 --executor both` |
| `cjc bench` | 1m | 29239 | NON-DETERMINISTIC | `--seed 42 --runs 3` |
| `cjc bench` | 1m | 14763 | NON-DETERMINISTIC | `--seed 42 --runs 3 --mir` |
| `cjc bench` | 1m | 44018 | NON-DETERMINISTIC | `--compare eval` |
| `cjc bench` | 1m | 29122 | NON-DETERMINISTIC | `--seed 42 --runs 3 --json` |
| `cjc bench` | 1m | 28938 | NON-DETERMINISTIC | `--seed 42 --runs 3 --markdown` |
| `cjc lock` | 1m | 7891 | DETERMINISTIC | `--seed 42` |
| `cjc lock` | 1m | 7082 | DETERMINISTIC | `--seed 42 --json` |

## 4. Filesystem & Project Commands

| Command | Target | Time (ms) | Deterministic | Flags |
|---------|--------|-----------|---------------|-------|
| `cjc view` | bench_results/ | 83 | DETERMINISTIC | -- |
| `cjc view` | bench_results/ | 95 | DETERMINISTIC | `--json` |
| `cjc view` | examples/ | 83 | DETERMINISTIC | `--recursive` |
| `cjc seek` | bench_results/ | 78 | DETERMINISTIC | `*.csv` |
| `cjc seek` | project root | 161 | DETERMINISTIC | `--type cjc --max-depth 2` |
| `cjc seek` | bench_results/ | 85 | DETERMINISTIC | `*.cjc --hash` |
| `cjc seek` | bench_results/ | 73 | DETERMINISTIC | `*.cjc --json` |
| `cjc seek` | bench_results/ | 80 | DETERMINISTIC | `*.cjc --manifest` |
| `cjc doctor` | bench_results/ | 371 | DETERMINISTIC | -- |
| `cjc doctor` | bench_results/ | 357 | DETERMINISTIC | `--verbose` |
| `cjc doctor` | bench_results/ | 331 | DETERMINISTIC | `--json` |
| `cjc pack` | 100k program | 80 | DETERMINISTIC | `--dry-run` |
| `cjc pack` | 100k program | 71 | DETERMINISTIC | `--manifest-only` |

## 5. Determinism Deep Dive — Multi-Seed Verification

Running each program 5 times across 5 different seeds, comparing SHA-256 of output.

| Program | Seed | Run 1 Hash | Run 2 Hash | Match |
|---------|------|------------|------------|-------|
| bench_100k | 1 | `08d4ccd024b3b95b...` | `08d4ccd024b3b95b...` | MATCH |
| bench_100k (mir-opt) | 1 | `08d4ccd024b3b95b...` | `08d4ccd024b3b95b...` | MATCH |
| bench_1m | 1 | `20e7987f6a14e9f7...` | `20e7987f6a14e9f7...` | MATCH |
| bench_1m (mir-opt) | 1 | `20e7987f6a14e9f7...` | `20e7987f6a14e9f7...` | MATCH |
| bench_100k | 42 | `93e279c7085a1cf6...` | `93e279c7085a1cf6...` | MATCH |
| bench_100k (mir-opt) | 42 | `93e279c7085a1cf6...` | `93e279c7085a1cf6...` | MATCH |
| bench_1m | 42 | `aeb48bad62219338...` | `aeb48bad62219338...` | MATCH |
| bench_1m (mir-opt) | 42 | `aeb48bad62219338...` | `aeb48bad62219338...` | MATCH |
| bench_100k | 100 | `4a52508e0fdfbc9b...` | `4a52508e0fdfbc9b...` | MATCH |
| bench_100k (mir-opt) | 100 | `4a52508e0fdfbc9b...` | `4a52508e0fdfbc9b...` | MATCH |
| bench_1m | 100 | `e41c6b48deffef92...` | `e41c6b48deffef92...` | MATCH |
| bench_1m (mir-opt) | 100 | `e41c6b48deffef92...` | `e41c6b48deffef92...` | MATCH |
| bench_100k | 999 | `3a8f0eef05031375...` | `3a8f0eef05031375...` | MATCH |
| bench_100k (mir-opt) | 999 | `3a8f0eef05031375...` | `3a8f0eef05031375...` | MATCH |
| bench_1m | 999 | `28c442cfd15bac61...` | `28c442cfd15bac61...` | MATCH |
| bench_1m (mir-opt) | 999 | `28c442cfd15bac61...` | `28c442cfd15bac61...` | MATCH |
| bench_100k | 12345 | `c01eadef8a18d4c5...` | `c01eadef8a18d4c5...` | MATCH |
| bench_100k (mir-opt) | 12345 | `c01eadef8a18d4c5...` | `c01eadef8a18d4c5...` | MATCH |
| bench_1m | 12345 | `0e5fb74a8533aae9...` | `0e5fb74a8533aae9...` | MATCH |
| bench_1m (mir-opt) | 12345 | `0e5fb74a8533aae9...` | `0e5fb74a8533aae9...` | MATCH |

## 6. Eval vs MIR-Opt Parity

| Program | Seed | Eval Hash | MIR-Opt Hash | Parity |
|---------|------|-----------|--------------|--------|
| bench_100k | 1 | `08d4ccd024b3b95b...` | `08d4ccd024b3b95b...` | IDENTICAL |
| bench_1m | 1 | `20e7987f6a14e9f7...` | `20e7987f6a14e9f7...` | IDENTICAL |
| bench_100k | 42 | `93e279c7085a1cf6...` | `93e279c7085a1cf6...` | IDENTICAL |
| bench_1m | 42 | `aeb48bad62219338...` | `aeb48bad62219338...` | IDENTICAL |
| bench_100k | 100 | `4a52508e0fdfbc9b...` | `4a52508e0fdfbc9b...` | IDENTICAL |
| bench_1m | 100 | `e41c6b48deffef92...` | `e41c6b48deffef92...` | IDENTICAL |
| bench_100k | 999 | `3a8f0eef05031375...` | `3a8f0eef05031375...` | IDENTICAL |
| bench_1m | 999 | `28c442cfd15bac61...` | `28c442cfd15bac61...` | IDENTICAL |
| bench_100k | 12345 | `c01eadef8a18d4c5...` | `c01eadef8a18d4c5...` | IDENTICAL |
| bench_1m | 12345 | `0e5fb74a8533aae9...` | `0e5fb74a8533aae9...` | IDENTICAL |

## Summary

- **Total benchmarks:** ~204
- **Determinism checks passed:** 192
- **Determinism checks failed:** 32
- **Binary size:** 8.7M
- **Zero external dependencies**

Generated by CJC v0.1.3 benchmark suite on Sun Apr  5 23:33:23 PDT 2026
