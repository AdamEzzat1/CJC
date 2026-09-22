# bench/bruchion_kernels_bench/run.ps1 -- the gated launcher for the CJC-side Bruchion
# kernel record. It refuses to measure on a loaded machine (the run that produced a 1.7x
# slowdown on identical bits was taken under a runaway service host), checks the boot time
# rather than believing "rebooted" (Fast Startup turns a shut-down into a resume), archives
# the previous record before overwriting it, and stamps the gate's readings into the
# provenance the runner writes.
#
#   .\bench\bruchion_kernels_bench\run.ps1 -KernelsDir <bruchionc build-kernel --emit-archive dir>
#
# Windows PowerShell 5.1.
param(
    [string]$KernelsDir = $env:BRUCHION_KERNELS_DIR,
    [int]$N = 65536,
    [int]$Phases = 5,
    [int]$PhaseMicros = 500000,
    [string]$Out = "bench_results/bruchion_kernels",
    [double]$GateAvg = 15.0,
    [double]$GateMax = 25.0,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $root

if (-not $KernelsDir -or -not (Test-Path (Join-Path $KernelsDir 'libkernels_f64.a'))) {
    Write-Error "BRUCHION_KERNELS_DIR / -KernelsDir must name a directory holding libkernels_f64.a (bruchionc build-kernel --emit-archive)"
}
$env:BRUCHION_KERNELS_DIR = (Resolve-Path $KernelsDir).Path

# The boot time, printed so a stale "rebooted" cannot pass unnoticed.
$os = Get-CimInstance Win32_OperatingSystem
$boot = '{0} (uptime {1:N1} h)' -f $os.LastBootUpTime, ((Get-Date) - $os.LastBootUpTime).TotalHours
Write-Host "last boot: $boot"

# The load gate: total CPU over 20 s, and the busiest processes for the record.
$c = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 10
$v = $c.CounterSamples | ForEach-Object { $_.CookedValue }
$avg = ($v | Measure-Object -Average).Average
$max = ($v | Measure-Object -Maximum).Maximum
$pc = Get-Counter '\Process(*)\% Processor Time' -SampleInterval 2 -MaxSamples 2 -ErrorAction SilentlyContinue
$acc = @{}
foreach ($s in $pc) { foreach ($x in $s.CounterSamples) { $inst = $x.InstanceName; if ($inst -ne '_total' -and $inst -ne 'idle') { if (-not $acc.ContainsKey($inst)) { $acc[$inst] = 0.0 }; $acc[$inst] += $x.CookedValue / 2 } } }
$top = ($acc.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 3 | ForEach-Object { '{0} {1:N0}%' -f $_.Key, $_.Value }) -join ', '
$gate = 'total CPU avg {0:N1}% max {1:N1}% over 20 s (thresholds {2}% / {3}%); busiest processes (% of one core): {4}' -f $avg, $max, $GateAvg, $GateMax, $top
Write-Host "gate: $gate"
if (($avg -ge $GateAvg -or $max -ge $GateMax) -and -not $Force) {
    Write-Host 'GATE CLOSED: the machine is not quiet; no record taken (use -Force to take a probe anyway, which the provenance will say)'
    exit 2
}
if ($Force -and ($avg -ge $GateAvg -or $max -ge $GateMax)) { $gate = "FORCED PAST A CLOSED GATE (a probe, not a record): $gate" }

# Archive the previous record before overwriting it.
if (Test-Path (Join-Path $Out 'REPORT.md')) {
    $prev = Get-Content (Join-Path $Out 'provenance.txt') -ErrorAction SilentlyContinue | Where-Object { $_ -like 'commit:*' } | ForEach-Object { $_.Substring(8, 7) }
    if (-not $prev) { $prev = 'unknown' }
    $stamp = '{0:yyyyMMdd-HHmmss}-{1}' -f (Get-Date), $prev
    $hist = Join-Path $Out (Join-Path 'history' $stamp)
    New-Item -ItemType Directory -Force $hist | Out-Null
    foreach ($f in 'REPORT.md', 'rows.jsonl', 'phases.csv', 'provenance.txt') {
        if (Test-Path (Join-Path $Out $f)) { Move-Item (Join-Path $Out $f) (Join-Path $hist $f) -Force }
    }
    Write-Host "archived the previous record to $hist"
}

$env:BRUCHION_BENCH_GATE = $gate
$env:BRUCHION_BENCH_BOOT = $boot
$env:BRUCHION_BENCH_LAUNCHER = 'bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)'
# cargo reports progress on stderr; under 'Stop' PowerShell 5.1 would turn that into an error.
$ErrorActionPreference = 'Continue'
cargo run -p bruchion-kernels-bench --release --features bruchion-kernels -- --out $Out --n $N --phases $Phases --phase-micros $PhaseMicros
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($code -ne 0) { Write-Host "runner exited $code"; exit $code }
Get-Content (Join-Path $Out 'REPORT.md') | Select-String '^\|' | ForEach-Object { $_.Line }
