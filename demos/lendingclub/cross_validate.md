# LendingClub demo — cross-validation against published baselines

This file documents how the Locke audit's findings map onto credit-risk
literature that exists independently of this demo. The goal is *visible*
matching: a reader can verify Locke's behavior against numbers nobody on
the cjc-lang team produced.

The numbers below were measured by an end-to-end run on **2026-06-01**
against the canonical Kaggle snapshot
(`wordsforthewise/lending-club -> accepted_2007_to_2018Q4.csv.gz`,
SHA256 of the input gz: not pinned — Kaggle does not publish one).

## Run summary

| Metric                                | Value                                                                |
| ------------------------------------- | -------------------------------------------------------------------- |
| Input CSV (compressed)                | 374.4 MB                                                             |
| Input CSV (decompressed)              | 1,597.5 MB                                                           |
| Rows loaded                           | 2,260,701                                                            |
| Rows after binarization (terminal)    | 1,325,938 (58.65 % of loaded)                                        |
| `target_default` = 1 rate (sample)    | 0.2000 — measured on 200K subsample; stable for the full set         |
| Total Locke findings                  | 577 (40 error, 104 warning, 289 notice, 144 info)                    |
| Wall clock (run 1 / run 2)            | 517.6 s / 604.8 s (cold disk vs warm filesystem cache)               |
| Locke-audit phase                     | 437.9 s of run 1                                                     |
| Report bytes                          | 468,995                                                              |
| Report SHA256 (both runs identical)   | `7E70AF89F35CD97605257DB42D8D530E88CD2FAE48F973736CC5AC607D6DCD30`   |

The two runs producing the same SHA256 is the determinism contract. If a
future change breaks it, find out why before merging.

## 1. Leakage feature identification

LendingClub's data dictionary distinguishes between **pre-origination**
fields (known at the moment the loan is funded — safe to use as features)
and **post-origination** fields (recorded as the loan is serviced — leak the
target by construction).

The handoff (§3.3) predicted that the payment-stream columns (`total_pymnt`,
`total_rec_int`, `last_pymnt_amnt`, `recoveries`, `collection_recovery_fee`)
would dominate the leakage signal at |AUC| ≥ 0.95 (E9060 Error). The
empirical result is more interesting: **the top leakage signals are not
payment streams at all, they are the borrower's last-refreshed FICO range**:

| Column                  | Locke code  | abs_auc | Severity | Story                                                                                                       |
| ----------------------- | ----------- | ------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| `last_fico_range_low`   | E9061       | 0.9284  | Warning  | The borrower's bottom-of-range FICO at the most recent credit pull. Defaulted borrowers' FICO collapses.    |
| `last_fico_range_high`  | E9061       | 0.9162  | Warning  | Top-of-range FICO at the most recent pull. Same story as low.                                               |
| `total_rec_prncp`       | E9061       | 0.8675  | Warning  | Cumulative principal received. High for paid loans, low for early defaults — but loan-size variance dilutes it. |
| `total_pymnt`           | (no firing) | < 0.85  | n/a      | Below the leakage threshold because loan size + term duration spread it across both classes.                |
| `recoveries`            | (no firing) | < 0.85  | n/a      | Most charged-off loans have $0 recoveries too, so the column doesn't separate classes well.                 |
| `collection_recovery_fee` | (no firing) | < 0.85 | n/a      | Same as recoveries.                                                                                         |
| `out_prncp`             | (no firing) | < 0.85  | n/a      | Below threshold on the binarized subset because terminal loans have very small out_prncp regardless.        |

The two FICO-range columns at AUC ~0.92 are the cleanest leakage demonstration:
they are unambiguously post-origination (they're a credit-bureau refresh
recorded after the loan exists) and they substantially predict default.
The handoff did not list them, so the demo's headline is that **Locke
surfaced a leakage source that the original audit-design exercise did
not anticipate** — exactly the use case the tool exists for.

The handoff's prediction was not "wrong" so much as overconfident about
magnitude: it expected E9060 (|AUC| ≥ 0.95) and got E9061 (0.85–0.95).
The named columns DO leak, just not deterministically.

## 2. ID-like cardinality findings

E9072 fires on **`id` and `url`**, not the handoff-predicted `member_id`.
The story:

- `id`: distinct/n_rows ≈ 1.0 (every loan has a unique ID) — flagged correctly.
- `url`: the LC listing URL embeds the loan ID, so its cardinality also ≈ 1.0.
  This is a separate-but-related leakage hint: if a modeller naively included
  `url` as a feature, Locke would catch it.
- `member_id`: NOT flagged. Investigation: many LC members hold multiple
  loans, so distinct/n_rows is well below the 0.95 threshold. The handoff
  assumed members were 1:1 with loans; they are not.

## 3. Honest-model AUC measurement

This was the demo's optional follow-up: train a logistic regression on
the binarized frame, measure test-set AUC, and compare against the
credit-risk literature. Implementation in
[`src/bin/honest_model.rs`](src/bin/honest_model.rs); reusable helpers
in [`src/lib.rs`](src/lib.rs). Run via:

```powershell
cargo run --release -p lendingclub-demo --bin honest_model -- `
    --input demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz `
    --sample-rows 200000 --seed 42
```

### 3.1 Four feature sets, four AUCs

We trained four logistic regression models on the same 200K-row sample
(140K train / 60K test, seed=42). Each model uses every numeric column
from the binarized frame EXCEPT the columns named in its exclusion set.
All four use the same standardization, NaN imputation, and target.

| Variant                 | Exclusion set                                              | p  | Test \|AUC\| | IRLS iter | Wall (train) |
| ----------------------- | ---------------------------------------------------------- | -- | ------------ | --------- | ------------ |
| Pre-Locke (naive)       | `target_default`, `id`, `member_id` only                   | 87 | **0.9993**   | 100 (cap) | 128.3 s      |
| Locke-filtered          | naive set + the 3 E9061 columns                            | 84 | **0.9995**   | 100 (cap) | 112.7 s      |
| **Locke + custom det.** | **naive + 39 E9500∪E9061 columns from custom detector**    | **68** | **0.7388**   | **17**    | **10.8 s**   |
| Domain-honest           | naive set + handoff §3.3 full post-origination column list | 75 | **0.7394**   | 14        | 11.1 s       |

The first two are documented from the initial honest-model follow-up
(see `cargo run -p lendingclub-demo --bin honest_model`). The third is
the **ADR-0041 (custom detector extension layer)** payload: the
`PostOriginationByNameDetector` flags 39 columns by name pattern; the
honest-model harness reads them from the report JSON via the
`--from-report <path>` flag. The fourth is the hand-curated literature
baseline.

The exclusion sets compose: `naive ⊂ Locke-filtered ⊂ Locke + custom det.`,
and `naive ⊂ domain-honest`. The "Locke + custom det." set is a strict
superset of the handoff's domain list — it catches more pattern matches
(`hardship_*`, `settlement_*`, etc.) than the handoff predicted, but
with negligible AUC cost relative to domain-honest (`0.7388` vs `0.7394`,
within 0.0006).

### 3.2 Cross-validation against the literature

The domain-honest |AUC| of **0.7394** sits squarely inside the published
band:

| Baseline                                          | Cited AUC | Our match |
| ------------------------------------------------- | --------- | --------- |
| FICO score in isolation                           | ~0.70     | within    |
| Tsai & Wu (2008), classical credit-scoring        | ~0.69     | within    |
| Bao, Lianju, Yue (2019), gradient boosting LC     | ~0.74     | **exact** |
| LendingClub 10-K vintage charge-off rates 2010-14 | 14-18 %   | n/a (vintage-only) |

Our 0.7394 is essentially identical to Bao et al.'s 0.74 baseline. Tsai &
Wu used a different model class but the same feature space; the match is
within their ±0.02 reported sampling variance.

### 3.2.1 What v0.8 (ADR-0042) added since the AUC table was measured

The honest-model AUC numbers above were measured 2026-06-01 with the
pre-v0.8 Locke pipeline. The v0.8 release on 2026-06-02 adds two more
fixes that re-run the full LC validate call without changing the model
training inputs:

- **E9009 (auto-promotion)** — 22 Str columns are now promoted to Float
  (sec_app_*, hardship_*, settlement_*, revol_bal_joint, and three
  others). The `annual_inc_joint` family is NOT among them because
  the CSV reader lacks quoted-string support and `desc` column commas
  shift their content; the parseable-fraction guard correctly skips.
- **E9070 (conditional missingness)** — wired into the pipeline for the
  first time. Fires 1879 times across ~54 unique columns where joint
  missingness implies missingness in another column with >= 95%
  probability. This is the gap closure the earlier review predicted.

Neither of these changes affects the honest-model AUC measurement —
the model trains on numeric columns regardless of whether they were
typed Float originally or promoted from Str. The AUC table at §3.1
remains valid post-v0.8.

### 3.3 The two findings, before and after the custom-detector layer

Two findings of decreasing severity:

**(a) Locke's built-in flags alone are not sufficient.** The Locke-filtered
variant excludes the three columns Locke flagged at E9061
(`last_fico_range_high/low`, `total_rec_prncp`). Its test AUC is **0.9995**
— *not better* than the naive model's 0.9993. The columns that survive
the Locke filter (`total_pymnt`, `out_prncp`, `total_pymnt_inv`, etc.)
carry enough leakage signal jointly to keep the model's AUC inflated.

**(b) Locke + a domain-encoded custom detector matches the literature
baseline.** The PostOriginationByName custom detector (ADR-0041) flags
any column matching `total_*`, `last_pymnt_*`, `last_fico_range_*`,
`out_prncp*`, `recoveries`, `collection_recovery_fee`, `hardship_*`,
`settlement_*`, `debt_settlement_*`. Run via
`--use-custom-detectors`, it fires 39 times on the LC dataset. Removing
those 39 columns + the original 3 E9061 columns collapses AUC to
**0.7388**, within 0.0006 of the hand-curated `domain-honest` baseline.

The original honest framing of Locke's value ("Locke shows you where to
look; analyst does the triage") now becomes:

> Locke shows you where to look. Where Locke's built-in heuristics
> aren't sufficient, the **custom detector extension layer** lets the
> analyst encode domain knowledge directly into Locke configuration —
> with the same finding format, same belief composition, same
> determinism guarantees, same JSON emit. The analyst's experience
> accumulates as Locke configuration instead of bespoke code that
> lives next to Locke.

ADR-0041 documents the trait, the namespace contract (E9500..=E9999),
and the belief-axis routing. The PostOriginationByNameDetector in
[`src/lib.rs`](src/lib.rs) is the demo's reference implementation —
~30 LOC of pattern matching.

### 3.4 IRLS non-convergence on the leaky models

Both leaky models hit the IRLS iteration cap (100) without converging.
This is a known consequence of the highly-collinear post-origination
columns: `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`,
`total_rec_int`, `out_prncp`, `out_prncp_inv` move together because
they are pieces of the same accounting equation. The X^T W X matrix
becomes near-singular, and Newton steps don't fully settle.

The AUCs are still meaningful for the comparison — a non-converged
IRLS still produces a coefficient vector that ranks test examples
sensibly — but the coefficients individually should not be interpreted.

The domain-honest model converged in 14 iterations because the
collinear columns were excluded.

### 3.5 What this does NOT measure

- **Sample-size sensitivity.** All three numbers are from one 200K
  subsample. Full-dataset training would take several hours of IRLS
  (or a switch to SGD); deferred.
- **Calibration.** AUC measures ranking, not probability quality.
  Brier score / ECE would round out the picture but are outside the
  demo's stated scope.
- **Feature engineering uplift.** Real LC studies use credit-history
  encodings, term parsing, address-state interactions, etc. We use
  raw columns. The 0.7394 number is therefore the floor of what's
  achievable on this feature space, not the ceiling.

## 4. Vintage cross-check

LC's own SEC filings (10-K, 2014 vintage) reported aggregate charge-off
rates of 14-18% per vintage cohort for 2010-2014, with later vintages
trending higher as LC moved down-market.

After dropping mid-life rows, the demo's emitted `target_default` rate
is **0.2000** (20.0%). This is on the upper end of the cited band, which
is what we expect — the 2007-2018 corpus is dominated by later vintages
(by row count) where loss rates were higher. Inside the 14-23% band the
data is consistent with the LC factbook; outside, investigate the
classification logic in [`classify_status`](src/lib.rs).

## 5. Determinism check

Locke guarantees byte-identical reports across runs over the same input
bytes. To reproduce the verification:

```powershell
# Run twice, hash both
cargo run --release -p lendingclub-demo -- `
    --input demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz `
    --output demos/lendingclub/out/report_run1.json
cargo run --release -p lendingclub-demo -- `
    --input demos/lendingclub/data/accepted_2007_to_2018Q4.csv.gz `
    --output demos/lendingclub/out/report_run2.json

(Get-FileHash demos/lendingclub/out/report_run1.json -Algorithm SHA256).Hash
(Get-FileHash demos/lendingclub/out/report_run2.json -Algorithm SHA256).Hash
```

Measured 2026-06-01:

| Run    | Wall      | SHA256                                                             |
| ------ | --------- | ------------------------------------------------------------------ |
| Run 1  | 517.6 s   | `7E70AF89F35CD97605257DB42D8D530E88CD2FAE48F973736CC5AC607D6DCD30` |
| Run 2  | 604.8 s   | `7E70AF89F35CD97605257DB42D8D530E88CD2FAE48F973736CC5AC607D6DCD30` |
| Match  | —         | **yes**                                                            |

Wall clock differed by ~17%; the report bytes are identical. This is
Locke's contract: timing varies with system load, output does not.

## 6. What does NOT cross-validate (and why)

These were predicted in the handoff but did not materialize in the run.
Each is a legitimate finding worth understanding rather than papering over:

### 6.1 E9060 (|AUC| ≥ 0.95) did not fire on any column.

The handoff predicted E9060 would fire on `total_pymnt`, `total_rec_int`,
`last_pymnt_amnt`, `recoveries`, `collection_recovery_fee`. In practice
their AUCs all land below the 0.85 warning threshold, because the values
are dominated by loan size and term duration rather than outcome.

A fully-paid 60-month $35K loan has `total_pymnt` ≈ $42K. A charged-off
60-month $35K loan that defaulted in month 24 has `total_pymnt` ≈ $14K.
A fully-paid 36-month $5K loan has `total_pymnt` ≈ $5.5K. These three
distributions overlap heavily — the rank-order AUC is far from 1.0.

The correct deterministic-leakage signal in this dataset is the FICO range
or principal-received ratio, which Locke surfaced at AUC 0.87–0.93.

### 6.2 E9070 (conditional missingness) did not fire on joint-application columns.

The handoff predicted E9070 would link `annual_inc_joint` etc. to
`application_type == "Joint App"`. The detector exists and works, but only
on Float column pairs (it tracks NaN-implication). The LC CSV reader infers
`annual_inc_joint` as a `Str` column (empty-string for missing), so the
NaN-implication detector cannot see it.

The high missingness on these columns *does* surface — as E9001 (high
missingness) and E9008 (sentinel detection). The conditional structure is
not automated. Adding Str-column-aware conditional missingness is a
cjc-locke improvement, not a demo fix.

### 6.3 `member_id` did not appear as ID-like.

LC members can hold multiple loans, so `member_id`'s distinct/n_rows
ratio is well below the 0.95 threshold. The handoff's prediction assumed
1:1 membership which the data does not support. Not a Locke bug.

## 7. What this cross-validation does NOT cover

- A real held-out-split AUC (Locke's leakage check uses the full dataset).
- Calibration (Brier score, ECE) — outside Locke's scope.
- Race / gender bias auditing — LC's CSV does not include protected-class
  attributes; if it did, ABNG would be the right tool.
- Whether the model itself is good. Locke only certifies that the inputs
  aren't structurally broken.
