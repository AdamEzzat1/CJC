"""Run Great Expectations 1.17 over bench/locke_vs_gx/churn.csv and
emit a normalized JSON of the findings so we can diff against Locke.

Expectations applied (chosen to mirror what Locke v0.5 fires by default):
- ExpectColumnValuesToNotBeNull on every numeric column → catches NaN
- ExpectColumnValuesToBeBetween for monthly_charges (-100, 500) → catches -9999
- ExpectColumnValuesToBeBetween for tenure_months (0, 200) → catches outliers
- ExpectCompoundColumnsToBeUnique on (customer_id, last_login_unix) → would catch dups
- ExpectColumnValueLengthsToBeBetween for country / plan (sanity)
- ExpectColumnDistinctValuesToBeInSet for plan
- ExpectColumnMean / Stdev / Quantile sanity

Output: bench/locke_vs_gx/gx_results.json
"""

import json
import pathlib

import great_expectations as gx
import pandas as pd


def numeric_or_none(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    df = pd.read_csv("bench/locke_vs_gx/churn.csv")
    # The CSV's total_charges contains empty strings to represent
    # missing values; pandas defaults that to NaN, which matches GX's
    # expectation of null semantics.

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="churn")
    batch_def = data_asset.add_batch_definition_whole_dataframe("whole")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    findings = []

    def record(name, severity, result):
        # GX `result` has `.success` and `.result` (dict with metrics).
        try:
            res_dict = result.to_json_dict()
        except Exception:
            res_dict = {"raw": str(result)}
        findings.append(
            {
                "name": name,
                "severity": severity,
                "success": bool(res_dict.get("success", False)),
                "details": res_dict.get("result", {}),
            }
        )

    # ── Missingness on numeric columns ──
    for col in [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "last_login_unix",
        "churned",
    ]:
        exp = gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        record(f"not_null({col})", "Error", batch.validate(exp))

    # ── Impossible values: monthly_charges in [-100, 500] ──
    exp = gx.expectations.ExpectColumnValuesToBeBetween(
        column="monthly_charges", min_value=-100, max_value=500
    )
    record("monthly_charges_in_range[-100,500]", "Error", batch.validate(exp))

    # ── Outliers: tenure_months in [0, 200] ──
    exp = gx.expectations.ExpectColumnValuesToBeBetween(
        column="tenure_months", min_value=0, max_value=200
    )
    record("tenure_months_in_range[0,200]", "Warning", batch.validate(exp))

    # ── Duplicate-row detection (GX uses compound key) ──
    exp = gx.expectations.ExpectCompoundColumnsToBeUnique(
        column_list=[
            "customer_id",
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "country",
            "plan",
            "last_login_unix",
            "churned",
        ]
    )
    record("full_row_unique", "Error", batch.validate(exp))

    # ── Plan must be one of the known set ──
    exp = gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="plan", value_set=["basic", "standard", "premium"]
    )
    record("plan_in_set", "Error", batch.validate(exp))

    # ── Country must be one of the known set ──
    exp = gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="country", value_set=["US", "UK", "DE", "FR", "JP", "CA", "AU", "BR"]
    )
    record("country_in_set", "Error", batch.validate(exp))

    # ── Basic numeric summaries (Locke also emits these implicitly) ──
    exp = gx.expectations.ExpectColumnMeanToBeBetween(
        column="tenure_months", min_value=0, max_value=300
    )
    record("tenure_mean_in_range", "Info", batch.validate(exp))

    # ── Sortedness of last_login_unix ──
    exp = gx.expectations.ExpectColumnValuesToBeIncreasing(
        column="last_login_unix", strictly=False
    )
    record("last_login_unix_increasing", "Warning", batch.validate(exp))

    # ── Churn class balance ──
    exp = gx.expectations.ExpectColumnValuesToBeInSet(
        column="churned", value_set=[0, 1]
    )
    record("churned_binary", "Info", batch.validate(exp))

    pathlib.Path("bench/locke_vs_gx/gx_results.json").write_text(
        json.dumps(findings, indent=2)
    )

    print(f"=== GX run summary ===")
    print(f"expectations evaluated: {len(findings)}")
    print(f"passed:  {sum(1 for f in findings if f['success'])}")
    print(f"failed:  {sum(1 for f in findings if not f['success'])}")

    # Inline summary of the failures we care about
    print("\n=== Failures (each = a finding GX would emit) ===")
    for f in findings:
        if not f["success"]:
            d = f["details"]
            print(
                f"  - {f['name']} (severity={f['severity']}) "
                f"unexpected_count={d.get('unexpected_count')} "
                f"unexpected_percent={d.get('unexpected_percent')}"
            )


if __name__ == "__main__":
    main()
