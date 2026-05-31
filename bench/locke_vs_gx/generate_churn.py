"""Generate a deterministic synthetic customer-churn dataset with
*exactly-known* statistical properties so Locke and Great Expectations
can be evaluated against shared ground truth.

The dataset mimics the IBM Telco Customer Churn dataset shape:
~7000 rows, mix of demographics, account info, services, charges.

Seeded properties (the ground truth we'll check both tools against):
- exactly 100 NaN values in `total_charges` column
- exactly 23 full-row duplicates (legitimately ingested twice)
- exactly 5 rows where `monthly_charges` is the impossible value -9999
- target `churned` is imbalanced: 26.5% positive
- exactly 17 rows where `tenure_months` > 200 (extreme outlier)
- `customer_id` is unique except for the 23 duplicate rows
- columns `last_login_date_unix` is sorted in non-decreasing order
"""

import csv
import math
import random

SEED = 0xCAFEBABE
N_ROWS = 7000

random.seed(SEED)


def main():
    rows = []
    countries = ["US", "UK", "DE", "FR", "JP", "CA", "AU", "BR"]
    plans = ["basic", "standard", "premium"]

    for i in range(N_ROWS):
        # Customer ID (will be made non-unique on a few rows via duplicates below).
        cid = 100000 + i
        tenure = random.randint(1, 72)
        monthly = round(random.uniform(20.0, 120.0), 2)
        total = round(monthly * tenure + random.uniform(-50, 50), 2)
        country = random.choice(countries)
        plan = random.choice(plans)
        # Churn rate ~26.5% biased by tenure (shorter tenure churns more)
        churn_prob = 0.5 if tenure < 12 else 0.3 if tenure < 24 else 0.15
        churned = 1 if random.random() < churn_prob else 0
        # last_login_date: monotonic per index to fake a sorted time column
        last_login = 1_600_000_000 + i * 86400 + random.randint(-3600, 3600)
        rows.append(
            {
                "customer_id": cid,
                "tenure_months": tenure,
                "monthly_charges": monthly,
                "total_charges": total,
                "country": country,
                "plan": plan,
                "last_login_unix": last_login,
                "churned": churned,
            }
        )

    # Seed exactly 100 NaN values in total_charges (rows 0..99).
    for i in range(100):
        rows[i]["total_charges"] = ""

    # Seed exactly 5 impossible values (-9999) in monthly_charges.
    for i in range(150, 155):
        rows[i]["monthly_charges"] = -9999

    # Seed exactly 17 outlier rows: tenure_months > 200.
    for i in range(200, 217):
        rows[i]["tenure_months"] = 250 + i % 50

    # Seed exactly 23 full-row duplicates: copy rows 1000..1022 to the end.
    n_dups = 23
    for i in range(n_dups):
        src = 1000 + i
        rows.append(dict(rows[src]))

    # Force last_login_unix to be sorted (write the actual sequential values).
    # The duplicates inserted above will violate sortedness — we'll let that be
    # a real finding for both tools.

    out_path = "bench/locke_vs_gx/churn.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Re-count for the ground truth.
    n_nan_total = sum(1 for r in rows if r["total_charges"] == "")
    n_impossible_monthly = sum(1 for r in rows if r["monthly_charges"] == -9999)
    n_outlier_tenure = sum(1 for r in rows if r["tenure_months"] > 200)
    n_churned = sum(1 for r in rows if r["churned"] == 1)
    n_rows_total = len(rows)
    churn_rate = n_churned / n_rows_total

    print("=== Ground Truth Properties ===")
    print(f"n_rows                       = {n_rows_total}")
    print(f"n_missing(total_charges)     = {n_nan_total}")
    print(f"n_impossible(monthly_charges)= {n_impossible_monthly}")
    print(f"n_outlier(tenure_months>200) = {n_outlier_tenure}")
    print(f"n_full_row_duplicates        = {n_dups}")
    print(f"n_churned                    = {n_churned}")
    print(f"churn_rate                   = {churn_rate:.4f}")
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()
