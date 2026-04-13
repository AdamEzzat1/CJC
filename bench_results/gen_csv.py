"""Generate benchmark CSV files at 100K and 1M rows."""
import random, math, os

random.seed(42)

def gen_csv(path, n_rows):
    with open(path, 'w') as f:
        f.write("id,sensor_a,sensor_b,temperature,pressure,category,timestamp,label\n")
        categories = ["alpha","beta","gamma","delta","epsilon"]
        labels = ["normal","anomaly","warning","critical"]
        for i in range(n_rows):
            sid = i + 1
            a = round(random.gauss(50.0, 15.0), 6)
            b = round(random.gauss(100.0, 25.0), 6)
            temp = round(20.0 + 10.0 * math.sin(i / 1000.0) + random.gauss(0, 2), 4)
            pres = round(101.325 + random.gauss(0, 5), 4)
            cat = categories[i % 5]
            ts = f"2026-01-{1 + (i % 28):02d}T{(i % 24):02d}:{(i*7)%60:02d}:00"
            label = labels[i % 4]
            # Inject some NaN/empty values
            if i % 1000 == 0:
                a = ""
            if i % 2000 == 0:
                b = "NaN"
            f.write(f"{sid},{a},{b},{temp},{pres},{cat},{ts},{label}\n")
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"Generated {path}: {n_rows:,} rows, {size_mb:.1f} MB")

gen_csv("bench_results/data_100k.csv", 100_000)
gen_csv("bench_results/data_1m.csv", 1_000_000)
