"""Compare Locke v0.5 and Great Expectations 1.17 findings on the same
synthetic churn dataset. Both tools were run against
bench/locke_vs_gx/churn.csv with equivalent validation suites.
"""

import json
import pathlib


def load_locke():
    return json.loads(
        pathlib.Path("bench/locke_vs_gx/locke_results.json").read_text()
    )


def load_gx():
    return json.loads(pathlib.Path("bench/locke_vs_gx/gx_results.json").read_text())


def locke_count(report, column, code, label):
    """Walk Locke findings for (code, column, evidence label)."""
    for f in report.get("findings", []):
        if f.get("code") != code or f.get("column") != column:
            continue
        for e in f.get("evidence", []):
            if e.get("kind") == "count" and e.get("label") == label:
                return e.get("value"), f.get("severity")
    return None, None


def gx_failure(gx_findings, name_prefix):
    for f in gx_findings:
        if f["name"].startswith(name_prefix) and not f["success"]:
            d = f.get("details", {})
            return d.get("unexpected_count"), f["severity"].lower()
    return None, None


def main():
    locke = load_locke()
    gx = load_gx()

    print("=" * 78)
    print("Locke v0.5 vs Great Expectations 1.17 -- side-by-side on synthetic churn")
    print("=" * 78)
    print()
    print(
        "Dataset: 7023 rows (7000 base + 23 appended duplicates), 8 columns. "
        "Seeded ground truth properties below."
    )
    print()
    print(f"{'Fact':<38} | {'GT':>6} | {'Locke (code, count, sev)':>30} | {'GX (count, sev)':>20}")
    print("-" * 105)

    # Fact 1: missing values in total_charges
    # Locke routes empty strings -> E9007 sentinel (not E9001 NaN) because
    # the column was CSV-inferred as Str.
    locke_v, locke_s = locke_count(locke, "total_charges", "E9007", "occurrences")
    locke_emit = (
        f"E9007 ({locke_v}, {locke_s})" if locke_v is not None else "not detected"
    )
    gx_v, gx_s = gx_failure(gx, "not_null(total_charges)")
    print(
        f"{'NaN in total_charges':<38} | {'100':>6} | "
        f"{locke_emit:>30} | {f'{gx_v} ({gx_s})':>20}"
    )

    # Fact 2: impossible monthly_charges (-9999)
    # Locke routes this through E9041 (extreme outlier) because the rate
    # is below sentinel-detection threshold but the values are far below
    # the IQR Q1.
    locke_v, locke_s = locke_count(locke, "monthly_charges", "E9041", "n_extreme")
    locke_emit = (
        f"E9041 ({locke_v}, {locke_s})" if locke_v is not None else "not detected"
    )
    gx_v, gx_s = gx_failure(gx, "monthly_charges_in_range[-100,500]")
    print(
        f"{'Impossible monthly_charges (-9999)':<38} | {'5':>6} | "
        f"{locke_emit:>30} | {f'{gx_v} ({gx_s})':>20}"
    )

    # Fact 3: outlier tenure_months
    locke_v, locke_s = locke_count(locke, "tenure_months", "E9041", "n_extreme")
    locke_emit = (
        f"E9041 ({locke_v}, {locke_s})" if locke_v is not None else "not detected"
    )
    gx_v, gx_s = gx_failure(gx, "tenure_months_in_range[0,200]")
    print(
        f"{'Outlier tenure_months (> 200)':<38} | {'17':>6} | "
        f"{locke_emit:>30} | {f'{gx_v} ({gx_s})':>20}"
    )

    # Fact 4: full-row duplicates
    locke_v, locke_s = locke_count(locke, None, "E9003", "duplicate_rows")
    locke_emit = (
        f"E9003 ({locke_v}, {locke_s})" if locke_v is not None else "not detected"
    )
    gx_v, gx_s = gx_failure(gx, "full_row_unique")
    # GX reports all rows in any non-unique group; halve to compare semantically.
    gx_extras = (gx_v // 2) if gx_v is not None else None
    gx_emit = f"{gx_extras} = {gx_v}/2 ({gx_s})" if gx_v is not None else "n/a"
    print(
        f"{'Full-row duplicates (extras only)':<38} | {'23':>6} | "
        f"{locke_emit:>30} | {gx_emit:>20}"
    )

    # Fact 5: sortedness
    locke_v, locke_s = locke_count(
        locke, "last_login_unix", "E9050", "n_unsorted_pairs"
    )
    locke_emit = (
        f"E9050 ({locke_v}, {locke_s})" if locke_v is not None else "not detected"
    )
    gx_v, gx_s = gx_failure(gx, "last_login_unix_increasing")
    print(
        f"{'last_login_unix not sorted':<38} | {'>=1':>6} | "
        f"{locke_emit:>30} | {f'{gx_v} ({gx_s})':>20}"
    )

    # Fact 6: duplicate primary key
    locke_v, locke_s = locke_count(locke, "customer_id", "E9004", "duplicate_keys")
    locke_emit = (
        f"E9004 ({locke_v}, {locke_s})" if locke_v is not None else "not detected"
    )
    print(
        f"{'Duplicate customer_id keys':<38} | {'23':>6} | "
        f"{locke_emit:>30} | {'(no equivalent expectation set)':>20}"
    )

    print()
    print("=" * 78)
    print("Built-in checks Locke runs that GX does NOT (out of the box)")
    print("=" * 78)
    extras = {
        "E9070": "Conditional missingness (missing(A) implies missing(B))",
        "E9071": "Imbalanced-class warning (caller-declared target)",
        "E9072": "ID-like cardinality hint",
        "E9073": "Duplicate-key conditioning (which columns disagree in dup groups)",
        "E9060": "Target leakage E9060 (perfect |AUC|>=0.95)",
        "E9061": "Target leakage E9061 (suspicious |AUC|>=0.85)",
        "E9051": "Train/test temporal overlap",
        "E9052": "Future-leakage cutoff (caller-supplied max_timestamp)",
        "E9004": "Duplicate primary-key detection",
    }
    for code, label in extras.items():
        n = sum(1 for f in locke.get("findings", []) if f.get("code") == code)
        marker = f"fired {n}x" if n > 0 else "not triggered on this dataset"
        print(f"  {code}: {label:<60s} [{marker}]")

    print()
    print("=" * 78)
    print("Built-in checks GX runs that Locke does NOT (out of the box)")
    print("=" * 78)
    print(
        "  - ExpectColumnDistinctValuesToBeInSet for plan / country\n"
        "    (Locke has E9020-E9022 schema mismatch but no value-set membership check)\n"
        "  - Strict-monotonic sortedness option (Locke only checks non-decreasing)\n"
        "  - User-declared mean / quantile range expectations\n"
        "    (Locke surfaces the values but doesn't accept caller-declared ranges)\n"
        "  - ~180 other expectations covering regex, length, JSON-schema, etc."
    )

    print()
    print("=" * 78)
    print("Ground-truth match summary (5 quantitative facts)")
    print("=" * 78)
    matches = []
    # 100 missing
    l1, _ = locke_count(locke, "total_charges", "E9007", "occurrences")
    g1, _ = gx_failure(gx, "not_null(total_charges)")
    matches.append(("100 NaN in total_charges", l1 == 100, g1 == 100))
    # 5 impossible
    l2, _ = locke_count(locke, "monthly_charges", "E9041", "n_extreme")
    g2, _ = gx_failure(gx, "monthly_charges_in_range[-100,500]")
    matches.append(("5 impossible monthly_charges", l2 == 5, g2 == 5))
    # 17 outliers
    l3, _ = locke_count(locke, "tenure_months", "E9041", "n_extreme")
    g3, _ = gx_failure(gx, "tenure_months_in_range[0,200]")
    matches.append(("17 outlier tenure_months", l3 == 17, g3 == 17))
    # 23 duplicates
    l4, _ = locke_count(locke, None, "E9003", "duplicate_rows")
    g4, _ = gx_failure(gx, "full_row_unique")
    g4_extras = (g4 // 2) if g4 else None
    matches.append(("23 full-row duplicates", l4 == 23, g4_extras == 23))
    # Sortedness
    l5, _ = locke_count(locke, "last_login_unix", "E9050", "n_unsorted_pairs")
    g5, _ = gx_failure(gx, "last_login_unix_increasing")
    matches.append((">=1 sortedness violation", (l5 or 0) >= 1, (g5 or 0) >= 1))

    n_both = sum(1 for _, l, g in matches if l and g)
    n_locke = sum(1 for _, l, _ in matches if l)
    n_gx = sum(1 for _, _, g in matches if g)
    print(
        f"\nBoth match ground truth on: {n_both}/5"
        f"   Locke matches: {n_locke}/5   GX matches: {n_gx}/5\n"
    )
    for label, l_ok, g_ok in matches:
        print(
            f"  {label:<42s} Locke={'PASS' if l_ok else 'MISS'}   "
            f"GX={'PASS' if g_ok else 'MISS'}"
        )


if __name__ == "__main__":
    main()
