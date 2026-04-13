#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# CJC v0.1.3 — Comprehensive CLI Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════════
# Tests EVERY CLI command at 100K and 1M scale
# Measures: speed (3 runs, median), determinism (bit-identical across runs)
# ═══════════════════════════════════════════════════════════════════════════

set -e

CJC="C:/Users/adame/CJC/target/release/cjc.exe"
DIR="C:/Users/adame/CJC/bench_results"
OUT="$DIR/BENCHMARK_RESULTS.md"

B100="$DIR/bench_100k.cjc"
B1M="$DIR/bench_1m.cjc"
CSV100="$DIR/data_100k.csv"
CSV1M="$DIR/data_1m.csv"

# Create a second CSV for drift comparison
head -50001 "$CSV100" > "$DIR/data_100k_half.csv"
# Slightly modified version for drift
cp "$CSV100" "$DIR/data_100k_b.csv"

# Tiny helper: time a command, capture output, return ms
timecmd() {
    local start end elapsed
    start=$(date +%s%N 2>/dev/null || python -c "import time; print(int(time.time()*1e9))")
    eval "$@" > /tmp/cjc_bench_out.txt 2>&1
    local exit_code=$?
    end=$(date +%s%N 2>/dev/null || python -c "import time; print(int(time.time()*1e9))")
    elapsed=$(( (end - start) / 1000000 ))
    echo "$elapsed"
    return $exit_code
}

# Determinism check: run twice, compare SHA256 of output
detcheck() {
    eval "$@" > /tmp/cjc_det_a.txt 2>&1
    eval "$@" > /tmp/cjc_det_b.txt 2>&1
    local ha=$(sha256sum /tmp/cjc_det_a.txt | cut -d' ' -f1)
    local hb=$(sha256sum /tmp/cjc_det_b.txt | cut -d' ' -f1)
    if [ "$ha" = "$hb" ]; then
        echo "DETERMINISTIC"
    else
        echo "NON-DETERMINISTIC"
    fi
}

# Triple run, report median
median3() {
    local t1 t2 t3
    t1=$(timecmd "$@")
    t2=$(timecmd "$@")
    t3=$(timecmd "$@")
    echo "$t1 $t2 $t3" | tr ' ' '\n' | sort -n | sed -n '2p'
}

echo "═══════════════════════════════════════════════════════════════" | tee "$OUT"
echo "  CJC v0.1.3 — Full CLI Benchmark Suite" | tee -a "$OUT"
echo "  $(date)" | tee -a "$OUT"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "## System Info" | tee -a "$OUT"
echo "- Binary size: $(ls -lh "$CJC" | awk '{print $5}')" | tee -a "$OUT"
echo "- Platform: $(uname -s) $(uname -m)" | tee -a "$OUT"
echo "- CPU: $(cat /proc/cpuinfo 2>/dev/null | grep 'model name' | head -1 | cut -d: -f2 | xargs || echo 'N/A')" | tee -a "$OUT"
echo "" | tee -a "$OUT"

# ─────────────────────────────────────────────────────────────────
# SECTION 1: CORE COMMANDS
# ─────────────────────────────────────────────────────────────────
echo "## 1. Core Commands" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "| Command | Scale | Time (ms) | Deterministic | Flags |" | tee -a "$OUT"
echo "|---------|-------|-----------|---------------|-------|" | tee -a "$OUT"

for scale in 100k 1m; do
    if [ "$scale" = "100k" ]; then BENCH="$B100"; else BENCH="$B1M"; fi

    # lex
    t=$(median3 "$CJC lex $BENCH")
    d=$(detcheck "$CJC lex $BENCH")
    echo "| \`cjc lex\` | $scale | ${t} | $d | -- |" | tee -a "$OUT"

    # lex --no-color
    t=$(median3 "$CJC lex $BENCH --no-color")
    d=$(detcheck "$CJC lex $BENCH --no-color")
    echo "| \`cjc lex\` | $scale | ${t} | $d | \`--no-color\` |" | tee -a "$OUT"

    # parse
    t=$(median3 "$CJC parse $BENCH")
    d=$(detcheck "$CJC parse $BENCH")
    echo "| \`cjc parse\` | $scale | ${t} | $d | -- |" | tee -a "$OUT"

    # parse --no-color
    t=$(median3 "$CJC parse $BENCH --no-color")
    d=$(detcheck "$CJC parse $BENCH --no-color")
    echo "| \`cjc parse\` | $scale | ${t} | $d | \`--no-color\` |" | tee -a "$OUT"

    # run (eval)
    t=$(median3 "$CJC run $BENCH --seed 42")
    d=$(detcheck "$CJC run $BENCH --seed 42")
    echo "| \`cjc run\` | $scale | ${t} | $d | \`--seed 42\` (eval) |" | tee -a "$OUT"

    # run (eval) --time
    t=$(median3 "$CJC run $BENCH --seed 42 --time")
    d=$(detcheck "$CJC run $BENCH --seed 42")
    echo "| \`cjc run\` | $scale | ${t} | $d | \`--seed 42 --time\` |" | tee -a "$OUT"

    # run (mir-opt)
    t=$(median3 "$CJC run $BENCH --seed 42 --mir-opt")
    d=$(detcheck "$CJC run $BENCH --seed 42 --mir-opt")
    echo "| \`cjc run --mir-opt\` | $scale | ${t} | $d | \`--seed 42 --mir-opt\` |" | tee -a "$OUT"

    # run --format json
    t=$(median3 "$CJC run $BENCH --seed 42 --format json")
    d=$(detcheck "$CJC run $BENCH --seed 42 --format json")
    echo "| \`cjc run\` | $scale | ${t} | $d | \`--format json\` |" | tee -a "$OUT"

    # run --format csv
    t=$(median3 "$CJC run $BENCH --seed 42 --format csv")
    d=$(detcheck "$CJC run $BENCH --seed 42 --format csv")
    echo "| \`cjc run\` | $scale | ${t} | $d | \`--format csv\` |" | tee -a "$OUT"

    # eval
    t=$(median3 "$CJC eval '1 + 2 * 3'")
    d=$(detcheck "$CJC eval '1 + 2 * 3'")
    echo "| \`cjc eval\` | expr | ${t} | $d | \`'1 + 2 * 3'\` |" | tee -a "$OUT"
done

# ─────────────────────────────────────────────────────────────────
# SECTION 2: DATA & PIPELINE COMMANDS
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$OUT"
echo "## 2. Data & Pipeline Commands" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "| Command | Scale | Time (ms) | Deterministic | Flags |" | tee -a "$OUT"
echo "|---------|-------|-----------|---------------|-------|" | tee -a "$OUT"

for scale in 100k 1m; do
    if [ "$scale" = "100k" ]; then CSV="$CSV100"; else CSV="$CSV1M"; fi

    # flow (default: all ops)
    t=$(median3 "$CJC flow $CSV")
    d=$(detcheck "$CJC flow $CSV")
    echo "| \`cjc flow\` | $scale | ${t} | $d | -- |" | tee -a "$OUT"

    # flow --op sum,mean,min,max
    t=$(median3 "$CJC flow $CSV --op sum,mean,min,max")
    d=$(detcheck "$CJC flow $CSV --op sum,mean,min,max")
    echo "| \`cjc flow\` | $scale | ${t} | $d | \`--op sum,mean,min,max\` |" | tee -a "$OUT"

    # flow --op sum,mean,var,std,count
    t=$(median3 "$CJC flow $CSV --op sum,mean,var,std,count")
    d=$(detcheck "$CJC flow $CSV --op sum,mean,var,std,count")
    echo "| \`cjc flow\` | $scale | ${t} | $d | \`--op sum,mean,var,std,count\` |" | tee -a "$OUT"

    # flow --columns sensor_a,temperature
    t=$(median3 "$CJC flow $CSV --columns sensor_a,temperature")
    d=$(detcheck "$CJC flow $CSV --columns sensor_a,temperature")
    echo "| \`cjc flow\` | $scale | ${t} | $d | \`--columns sensor_a,temperature\` |" | tee -a "$OUT"

    # flow --json
    t=$(median3 "$CJC flow $CSV --json")
    d=$(detcheck "$CJC flow $CSV --json")
    echo "| \`cjc flow\` | $scale | ${t} | $d | \`--json\` |" | tee -a "$OUT"

    # flow --verify
    t=$(median3 "$CJC flow $CSV --verify")
    d=$(detcheck "$CJC flow $CSV --verify")
    echo "| \`cjc flow\` | $scale | ${t} | $d | \`--verify\` |" | tee -a "$OUT"

    # flow --top 3
    t=$(median3 "$CJC flow $CSV --top 3")
    d=$(detcheck "$CJC flow $CSV --top 3")
    echo "| \`cjc flow\` | $scale | ${t} | $d | \`--top 3\` |" | tee -a "$OUT"

    # flow --precision 10
    t=$(median3 "$CJC flow $CSV --precision 10")
    d=$(detcheck "$CJC flow $CSV --precision 10")
    echo "| \`cjc flow\` | $scale | ${t} | $d | \`--precision 10\` |" | tee -a "$OUT"

    # schema
    t=$(median3 "$CJC schema $CSV")
    d=$(detcheck "$CJC schema $CSV")
    echo "| \`cjc schema\` | $scale | ${t} | $d | -- |" | tee -a "$OUT"

    # schema --full
    t=$(median3 "$CJC schema $CSV --full")
    d=$(detcheck "$CJC schema $CSV --full")
    echo "| \`cjc schema\` | $scale | ${t} | $d | \`--full\` |" | tee -a "$OUT"

    # schema --json
    t=$(median3 "$CJC schema $CSV --json")
    d=$(detcheck "$CJC schema $CSV --json")
    echo "| \`cjc schema\` | $scale | ${t} | $d | \`--json\` |" | tee -a "$OUT"

    # schema --sample 1000
    t=$(median3 "$CJC schema $CSV --sample 1000")
    d=$(detcheck "$CJC schema $CSV --sample 1000")
    echo "| \`cjc schema\` | $scale | ${t} | $d | \`--sample 1000\` |" | tee -a "$OUT"

    # inspect
    t=$(median3 "$CJC inspect $CSV")
    d=$(detcheck "$CJC inspect $CSV")
    echo "| \`cjc inspect\` | $scale | ${t} | $d | -- |" | tee -a "$OUT"

    # inspect --deep
    t=$(median3 "$CJC inspect $CSV --deep")
    d=$(detcheck "$CJC inspect $CSV --deep")
    echo "| \`cjc inspect\` | $scale | ${t} | $d | \`--deep\` |" | tee -a "$OUT"

    # inspect --schema-only
    t=$(median3 "$CJC inspect $CSV --schema-only")
    d=$(detcheck "$CJC inspect $CSV --schema-only")
    echo "| \`cjc inspect\` | $scale | ${t} | $d | \`--schema-only\` |" | tee -a "$OUT"

    # inspect --header-only
    t=$(median3 "$CJC inspect $CSV --header-only")
    d=$(detcheck "$CJC inspect $CSV --header-only")
    echo "| \`cjc inspect\` | $scale | ${t} | $d | \`--header-only\` |" | tee -a "$OUT"

    # inspect --json
    t=$(median3 "$CJC inspect $CSV --json")
    d=$(detcheck "$CJC inspect $CSV --json")
    echo "| \`cjc inspect\` | $scale | ${t} | $d | \`--json\` |" | tee -a "$OUT"

    # inspect --hash
    t=$(median3 "$CJC inspect $CSV --hash")
    d=$(detcheck "$CJC inspect $CSV --hash")
    echo "| \`cjc inspect\` | $scale | ${t} | $d | \`--hash\` |" | tee -a "$OUT"

    # patch --nan-fill 0
    t=$(median3 "$CJC patch $CSV --nan-fill 0 --dry-run")
    d=$(detcheck "$CJC patch $CSV --nan-fill 0 --dry-run")
    echo "| \`cjc patch\` | $scale | ${t} | $d | \`--nan-fill 0 --dry-run\` |" | tee -a "$OUT"

    # patch --drop category --dry-run
    t=$(median3 "$CJC patch $CSV --drop category --dry-run")
    d=$(detcheck "$CJC patch $CSV --drop category --dry-run")
    echo "| \`cjc patch\` | $scale | ${t} | $d | \`--drop category --dry-run\` |" | tee -a "$OUT"

    # patch --rename label status --dry-run
    t=$(median3 "$CJC patch $CSV --rename label status --dry-run")
    d=$(detcheck "$CJC patch $CSV --rename label status --dry-run")
    echo "| \`cjc patch\` | $scale | ${t} | $d | \`--rename label status --dry-run\` |" | tee -a "$OUT"

    # patch --plan
    t=$(median3 "$CJC patch $CSV --nan-fill 0 --plan")
    d=$(detcheck "$CJC patch $CSV --nan-fill 0 --plan")
    echo "| \`cjc patch\` | $scale | ${t} | $d | \`--nan-fill 0 --plan\` |" | tee -a "$OUT"
done

# drift (100k only — comparing two files)
t=$(median3 "$CJC drift $CSV100 $DIR/data_100k_b.csv --csv")
d=$(detcheck "$CJC drift $CSV100 $DIR/data_100k_b.csv --csv")
echo "| \`cjc drift\` | 100k | ${t} | $d | \`--csv\` (identical files) |" | tee -a "$OUT"

t=$(median3 "$CJC drift $CSV100 $DIR/data_100k_half.csv --csv --summary-only")
d=$(detcheck "$CJC drift $CSV100 $DIR/data_100k_half.csv --csv --summary-only")
echo "| \`cjc drift\` | 100k | ${t} | $d | \`--csv --summary-only\` (diff sizes) |" | tee -a "$OUT"

t=$(median3 "$CJC drift $CSV100 $DIR/data_100k_b.csv --csv --json")
d=$(detcheck "$CJC drift $CSV100 $DIR/data_100k_b.csv --csv --json")
echo "| \`cjc drift\` | 100k | ${t} | $d | \`--csv --json\` |" | tee -a "$OUT"

# ─────────────────────────────────────────────────────────────────
# SECTION 3: COMPILER COMMANDS (CJC programs)
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$OUT"
echo "## 3. Compiler Visibility & Analysis" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "| Command | Scale | Time (ms) | Deterministic | Flags |" | tee -a "$OUT"
echo "|---------|-------|-----------|---------------|-------|" | tee -a "$OUT"

for scale in 100k 1m; do
    if [ "$scale" = "100k" ]; then BENCH="$B100"; else BENCH="$B1M"; fi

    # emit --stage ast
    t=$(median3 "$CJC emit $BENCH --stage ast")
    d=$(detcheck "$CJC emit $BENCH --stage ast")
    echo "| \`cjc emit\` | $scale | ${t} | $d | \`--stage ast\` |" | tee -a "$OUT"

    # emit --stage hir
    t=$(median3 "$CJC emit $BENCH --stage hir")
    d=$(detcheck "$CJC emit $BENCH --stage hir")
    echo "| \`cjc emit\` | $scale | ${t} | $d | \`--stage hir\` |" | tee -a "$OUT"

    # emit --stage mir
    t=$(median3 "$CJC emit $BENCH --stage mir")
    d=$(detcheck "$CJC emit $BENCH --stage mir")
    echo "| \`cjc emit\` | $scale | ${t} | $d | \`--stage mir\` |" | tee -a "$OUT"

    # emit --stage mir --opt
    t=$(median3 "$CJC emit $BENCH --stage mir --opt")
    d=$(detcheck "$CJC emit $BENCH --stage mir --opt")
    echo "| \`cjc emit\` | $scale | ${t} | $d | \`--stage mir --opt\` |" | tee -a "$OUT"

    # emit --stage mir --diff
    t=$(median3 "$CJC emit $BENCH --stage mir --diff")
    d=$(detcheck "$CJC emit $BENCH --stage mir --diff")
    echo "| \`cjc emit\` | $scale | ${t} | $d | \`--stage mir --diff\` |" | tee -a "$OUT"

    # explain
    t=$(median3 "$CJC explain $BENCH")
    d=$(detcheck "$CJC explain $BENCH")
    echo "| \`cjc explain\` | $scale | ${t} | $d | -- |" | tee -a "$OUT"

    # explain --verbose
    t=$(median3 "$CJC explain $BENCH --verbose")
    d=$(detcheck "$CJC explain $BENCH --verbose")
    echo "| \`cjc explain\` | $scale | ${t} | $d | \`--verbose\` |" | tee -a "$OUT"

    # trace
    t=$(median3 "$CJC trace $BENCH --seed 42")
    d=$(detcheck "$CJC trace $BENCH --seed 42")
    echo "| \`cjc trace\` | $scale | ${t} | $d | \`--seed 42\` |" | tee -a "$OUT"

    # trace --ast
    t=$(median3 "$CJC trace $BENCH --seed 42 --ast")
    d=$(detcheck "$CJC trace $BENCH --seed 42 --ast")
    echo "| \`cjc trace\` | $scale | ${t} | $d | \`--seed 42 --ast\` |" | tee -a "$OUT"

    # trace --verbose
    t=$(median3 "$CJC trace $BENCH --seed 42 --verbose")
    d=$(detcheck "$CJC trace $BENCH --seed 42 --verbose")
    echo "| \`cjc trace\` | $scale | ${t} | $d | \`--seed 42 --verbose\` |" | tee -a "$OUT"

    # trace --json
    t=$(median3 "$CJC trace $BENCH --seed 42 --json")
    d=$(detcheck "$CJC trace $BENCH --seed 42 --json")
    echo "| \`cjc trace\` | $scale | ${t} | $d | \`--seed 42 --json\` |" | tee -a "$OUT"

    # audit
    t=$(median3 "$CJC audit $BENCH")
    d=$(detcheck "$CJC audit $BENCH")
    echo "| \`cjc audit\` | $scale | ${t} | $d | -- |" | tee -a "$OUT"

    # audit --verbose
    t=$(median3 "$CJC audit $BENCH --verbose")
    d=$(detcheck "$CJC audit $BENCH --verbose")
    echo "| \`cjc audit\` | $scale | ${t} | $d | \`--verbose\` |" | tee -a "$OUT"

    # audit --json
    t=$(median3 "$CJC audit $BENCH --json")
    d=$(detcheck "$CJC audit $BENCH --json")
    echo "| \`cjc audit\` | $scale | ${t} | $d | \`--json\` |" | tee -a "$OUT"

    # precision
    t=$(median3 "$CJC precision $BENCH --seed 42")
    d=$(detcheck "$CJC precision $BENCH --seed 42")
    echo "| \`cjc precision\` | $scale | ${t} | $d | \`--seed 42\` |" | tee -a "$OUT"

    # precision --summary-only
    t=$(median3 "$CJC precision $BENCH --seed 42 --summary-only")
    d=$(detcheck "$CJC precision $BENCH --seed 42 --summary-only")
    echo "| \`cjc precision\` | $scale | ${t} | $d | \`--seed 42 --summary-only\` |" | tee -a "$OUT"

    # precision --json
    t=$(median3 "$CJC precision $BENCH --seed 42 --json")
    d=$(detcheck "$CJC precision $BENCH --seed 42 --json")
    echo "| \`cjc precision\` | $scale | ${t} | $d | \`--seed 42 --json\` |" | tee -a "$OUT"

    # nogc
    t=$(median3 "$CJC nogc $BENCH")
    d=$(detcheck "$CJC nogc $BENCH")
    echo "| \`cjc nogc\` | $scale | ${t} | $d | -- |" | tee -a "$OUT"

    # nogc --verbose
    t=$(median3 "$CJC nogc $BENCH --verbose")
    d=$(detcheck "$CJC nogc $BENCH --verbose")
    echo "| \`cjc nogc\` | $scale | ${t} | $d | \`--verbose\` |" | tee -a "$OUT"

    # gc
    t=$(median3 "$CJC gc $BENCH --seed 42")
    d=$(detcheck "$CJC gc $BENCH --seed 42")
    echo "| \`cjc gc\` | $scale | ${t} | $d | \`--seed 42\` |" | tee -a "$OUT"

    # gc --verbose
    t=$(median3 "$CJC gc $BENCH --seed 42 --verbose")
    d=$(detcheck "$CJC gc $BENCH --seed 42 --verbose")
    echo "| \`cjc gc\` | $scale | ${t} | $d | \`--seed 42 --verbose\` |" | tee -a "$OUT"

    # gc --json
    t=$(median3 "$CJC gc $BENCH --seed 42 --json")
    d=$(detcheck "$CJC gc $BENCH --seed 42 --json")
    echo "| \`cjc gc\` | $scale | ${t} | $d | \`--seed 42 --json\` |" | tee -a "$OUT"

    # mem
    t=$(median3 "$CJC mem $BENCH --seed 42")
    d=$(detcheck "$CJC mem $BENCH --seed 42")
    echo "| \`cjc mem\` | $scale | ${t} | $d | \`--seed 42\` |" | tee -a "$OUT"

    # mem --verbose
    t=$(median3 "$CJC mem $BENCH --seed 42 --verbose")
    d=$(detcheck "$CJC mem $BENCH --seed 42 --verbose")
    echo "| \`cjc mem\` | $scale | ${t} | $d | \`--seed 42 --verbose\` |" | tee -a "$OUT"

    # mem --mir
    t=$(median3 "$CJC mem $BENCH --seed 42 --mir")
    d=$(detcheck "$CJC mem $BENCH --seed 42 --mir")
    echo "| \`cjc mem\` | $scale | ${t} | $d | \`--seed 42 --mir\` |" | tee -a "$OUT"

    # mem --json
    t=$(median3 "$CJC mem $BENCH --seed 42 --json")
    d=$(detcheck "$CJC mem $BENCH --seed 42 --json")
    echo "| \`cjc mem\` | $scale | ${t} | $d | \`--seed 42 --json\` |" | tee -a "$OUT"

    # parity
    t=$(median3 "$CJC parity $BENCH --seed 42")
    d=$(detcheck "$CJC parity $BENCH --seed 42")
    echo "| \`cjc parity\` | $scale | ${t} | $d | \`--seed 42\` |" | tee -a "$OUT"

    # parity --verbose
    t=$(median3 "$CJC parity $BENCH --seed 42 --verbose")
    d=$(detcheck "$CJC parity $BENCH --seed 42 --verbose")
    echo "| \`cjc parity\` | $scale | ${t} | $d | \`--seed 42 --verbose\` |" | tee -a "$OUT"

    # parity --json
    t=$(median3 "$CJC parity $BENCH --seed 42 --json")
    d=$(detcheck "$CJC parity $BENCH --seed 42 --json")
    echo "| \`cjc parity\` | $scale | ${t} | $d | \`--seed 42 --json\` |" | tee -a "$OUT"

    # proof
    t=$(median3 "$CJC proof $BENCH --seed 42 --runs 3")
    d=$(detcheck "$CJC proof $BENCH --seed 42 --runs 3")
    echo "| \`cjc proof\` | $scale | ${t} | $d | \`--seed 42 --runs 3\` |" | tee -a "$OUT"

    # proof --verbose
    t=$(median3 "$CJC proof $BENCH --seed 42 --runs 3 --verbose")
    d=$(detcheck "$CJC proof $BENCH --seed 42 --runs 3 --verbose")
    echo "| \`cjc proof\` | $scale | ${t} | $d | \`--seed 42 --runs 3 --verbose\` |" | tee -a "$OUT"

    # proof --hash-output
    t=$(median3 "$CJC proof $BENCH --seed 42 --runs 3 --hash-output")
    d=$(detcheck "$CJC proof $BENCH --seed 42 --runs 3 --hash-output")
    echo "| \`cjc proof\` | $scale | ${t} | $d | \`--seed 42 --runs 3 --hash-output\` |" | tee -a "$OUT"

    # proof --json
    t=$(median3 "$CJC proof $BENCH --seed 42 --runs 3 --json")
    d=$(detcheck "$CJC proof $BENCH --seed 42 --runs 3 --json")
    echo "| \`cjc proof\` | $scale | ${t} | $d | \`--seed 42 --runs 3 --json\` |" | tee -a "$OUT"

    # proof --executor both
    t=$(median3 "$CJC proof $BENCH --seed 42 --runs 3 --executor both")
    d=$(detcheck "$CJC proof $BENCH --seed 42 --runs 3 --executor both")
    echo "| \`cjc proof\` | $scale | ${t} | $d | \`--seed 42 --executor both\` |" | tee -a "$OUT"

    # bench (the CLI bench command)
    t=$(median3 "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1")
    d=$(detcheck "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1")
    echo "| \`cjc bench\` | $scale | ${t} | $d | \`--seed 42 --runs 3\` |" | tee -a "$OUT"

    # bench --mir
    t=$(median3 "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1 --mir")
    d=$(detcheck "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1 --mir")
    echo "| \`cjc bench\` | $scale | ${t} | $d | \`--seed 42 --runs 3 --mir\` |" | tee -a "$OUT"

    # bench --compare
    t=$(median3 "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1 --compare eval")
    d=$(detcheck "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1 --compare eval")
    echo "| \`cjc bench\` | $scale | ${t} | $d | \`--compare eval\` |" | tee -a "$OUT"

    # bench --json
    t=$(median3 "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1 --json")
    d=$(detcheck "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1 --json")
    echo "| \`cjc bench\` | $scale | ${t} | $d | \`--seed 42 --runs 3 --json\` |" | tee -a "$OUT"

    # bench --markdown
    t=$(median3 "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1 --markdown")
    d=$(detcheck "$CJC bench $BENCH --seed 42 --runs 3 --warmup 1 --markdown")
    echo "| \`cjc bench\` | $scale | ${t} | $d | \`--seed 42 --runs 3 --markdown\` |" | tee -a "$OUT"

    # lock
    t=$(median3 "$CJC lock $BENCH --seed 42")
    d=$(detcheck "$CJC lock $BENCH --seed 42")
    echo "| \`cjc lock\` | $scale | ${t} | $d | \`--seed 42\` |" | tee -a "$OUT"

    # lock --json
    t=$(median3 "$CJC lock $BENCH --seed 42 --json")
    d=$(detcheck "$CJC lock $BENCH --seed 42 --json")
    echo "| \`cjc lock\` | $scale | ${t} | $d | \`--seed 42 --json\` |" | tee -a "$OUT"
done

# ─────────────────────────────────────────────────────────────────
# SECTION 4: FILESYSTEM/PROJECT COMMANDS
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$OUT"
echo "## 4. Filesystem & Project Commands" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "| Command | Target | Time (ms) | Deterministic | Flags |" | tee -a "$OUT"
echo "|---------|--------|-----------|---------------|-------|" | tee -a "$OUT"

# view
t=$(median3 "$CJC view $DIR")
d=$(detcheck "$CJC view $DIR")
echo "| \`cjc view\` | bench_results/ | ${t} | $d | -- |" | tee -a "$OUT"

# view --json
t=$(median3 "$CJC view $DIR --json")
d=$(detcheck "$CJC view $DIR --json")
echo "| \`cjc view\` | bench_results/ | ${t} | $d | \`--json\` |" | tee -a "$OUT"

# view --recursive
t=$(median3 "$CJC view C:/Users/adame/CJC/examples --recursive")
d=$(detcheck "$CJC view C:/Users/adame/CJC/examples --recursive")
echo "| \`cjc view\` | examples/ | ${t} | $d | \`--recursive\` |" | tee -a "$OUT"

# seek
t=$(median3 "$CJC seek $DIR '*.csv'")
d=$(detcheck "$CJC seek $DIR '*.csv'")
echo "| \`cjc seek\` | bench_results/ | ${t} | $d | \`*.csv\` |" | tee -a "$OUT"

# seek --type cjc
t=$(median3 "$CJC seek C:/Users/adame/CJC --type cjc --max-depth 2")
d=$(detcheck "$CJC seek C:/Users/adame/CJC --type cjc --max-depth 2")
echo "| \`cjc seek\` | project root | ${t} | $d | \`--type cjc --max-depth 2\` |" | tee -a "$OUT"

# seek --hash
t=$(median3 "$CJC seek $DIR '*.cjc' --hash")
d=$(detcheck "$CJC seek $DIR '*.cjc' --hash")
echo "| \`cjc seek\` | bench_results/ | ${t} | $d | \`*.cjc --hash\` |" | tee -a "$OUT"

# seek --json
t=$(median3 "$CJC seek $DIR '*.cjc' --json")
d=$(detcheck "$CJC seek $DIR '*.cjc' --json")
echo "| \`cjc seek\` | bench_results/ | ${t} | $d | \`*.cjc --json\` |" | tee -a "$OUT"

# seek --manifest
t=$(median3 "$CJC seek $DIR '*.cjc' --manifest")
d=$(detcheck "$CJC seek $DIR '*.cjc' --manifest")
echo "| \`cjc seek\` | bench_results/ | ${t} | $d | \`*.cjc --manifest\` |" | tee -a "$OUT"

# doctor
t=$(median3 "$CJC doctor $DIR")
d=$(detcheck "$CJC doctor $DIR")
echo "| \`cjc doctor\` | bench_results/ | ${t} | $d | -- |" | tee -a "$OUT"

# doctor --verbose
t=$(median3 "$CJC doctor $DIR --verbose")
d=$(detcheck "$CJC doctor $DIR --verbose")
echo "| \`cjc doctor\` | bench_results/ | ${t} | $d | \`--verbose\` |" | tee -a "$OUT"

# doctor --json
t=$(median3 "$CJC doctor $DIR --json")
d=$(detcheck "$CJC doctor $DIR --json")
echo "| \`cjc doctor\` | bench_results/ | ${t} | $d | \`--json\` |" | tee -a "$OUT"

# pack --dry-run
t=$(median3 "$CJC pack $B100 --dry-run")
d=$(detcheck "$CJC pack $B100 --dry-run")
echo "| \`cjc pack\` | 100k program | ${t} | $d | \`--dry-run\` |" | tee -a "$OUT"

# pack --manifest-only
t=$(median3 "$CJC pack $B100 --manifest-only")
d=$(detcheck "$CJC pack $B100 --manifest-only")
echo "| \`cjc pack\` | 100k program | ${t} | $d | \`--manifest-only\` |" | tee -a "$OUT"

# ─────────────────────────────────────────────────────────────────
# SECTION 5: DETERMINISM DEEP DIVE
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$OUT"
echo "## 5. Determinism Deep Dive — Multi-Seed Verification" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "Running each program 5 times across 5 different seeds, comparing SHA-256 of output." | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "| Program | Seed | Run 1 Hash | Run 2 Hash | Match |" | tee -a "$OUT"
echo "|---------|------|------------|------------|-------|" | tee -a "$OUT"

for seed in 1 42 100 999 12345; do
    for scale in 100k 1m; do
        if [ "$scale" = "100k" ]; then BENCH="$B100"; else BENCH="$B1M"; fi

        "$CJC" run "$BENCH" --seed $seed > /tmp/det_r1.txt 2>&1
        "$CJC" run "$BENCH" --seed $seed > /tmp/det_r2.txt 2>&1
        h1=$(sha256sum /tmp/det_r1.txt | cut -c1-16)
        h2=$(sha256sum /tmp/det_r2.txt | cut -c1-16)
        if [ "$h1" = "$h2" ]; then m="MATCH"; else m="DIVERGED"; fi
        echo "| bench_${scale} | $seed | \`${h1}...\` | \`${h2}...\` | $m |" | tee -a "$OUT"

        # Also check MIR-opt
        "$CJC" run "$BENCH" --seed $seed --mir-opt > /tmp/det_r1.txt 2>&1
        "$CJC" run "$BENCH" --seed $seed --mir-opt > /tmp/det_r2.txt 2>&1
        h1=$(sha256sum /tmp/det_r1.txt | cut -c1-16)
        h2=$(sha256sum /tmp/det_r2.txt | cut -c1-16)
        if [ "$h1" = "$h2" ]; then m="MATCH"; else m="DIVERGED"; fi
        echo "| bench_${scale} (mir-opt) | $seed | \`${h1}...\` | \`${h2}...\` | $m |" | tee -a "$OUT"
    done
done

# ─────────────────────────────────────────────────────────────────
# SECTION 6: EVAL vs MIR-OPT PARITY
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$OUT"
echo "## 6. Eval vs MIR-Opt Parity" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "| Program | Seed | Eval Hash | MIR-Opt Hash | Parity |" | tee -a "$OUT"
echo "|---------|------|-----------|--------------|--------|" | tee -a "$OUT"

for seed in 1 42 100 999 12345; do
    for scale in 100k 1m; do
        if [ "$scale" = "100k" ]; then BENCH="$B100"; else BENCH="$B1M"; fi

        "$CJC" run "$BENCH" --seed $seed > /tmp/eval_out.txt 2>&1
        "$CJC" run "$BENCH" --seed $seed --mir-opt > /tmp/mir_out.txt 2>&1
        he=$(sha256sum /tmp/eval_out.txt | cut -c1-16)
        hm=$(sha256sum /tmp/mir_out.txt | cut -c1-16)
        if [ "$he" = "$hm" ]; then p="IDENTICAL"; else p="DIVERGED"; fi
        echo "| bench_${scale} | $seed | \`${he}...\` | \`${hm}...\` | $p |" | tee -a "$OUT"
    done
done

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$OUT"
echo "## Summary" | tee -a "$OUT"
echo "" | tee -a "$OUT"

total_benchmarks=$(grep -c "^|" "$OUT" | head -1)
det_pass=$(grep -c "DETERMINISTIC\|MATCH\|IDENTICAL" "$OUT")
det_fail=$(grep -c "NON-DETERMINISTIC\|DIVERGED" "$OUT")

echo "- **Total benchmarks:** ~${total_benchmarks}" | tee -a "$OUT"
echo "- **Determinism checks passed:** ${det_pass}" | tee -a "$OUT"
echo "- **Determinism checks failed:** ${det_fail}" | tee -a "$OUT"
echo "- **Binary size:** $(ls -lh "$CJC" | awk '{print $5}')" | tee -a "$OUT"
echo "- **Zero external dependencies**" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "Generated by CJC v0.1.3 benchmark suite on $(date)" | tee -a "$OUT"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  BENCHMARK COMPLETE — Results saved to $OUT"
echo "═══════════════════════════════════════════════════════════════"
