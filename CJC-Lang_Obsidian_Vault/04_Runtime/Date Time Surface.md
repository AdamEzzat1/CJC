---
title: Date Time Surface
tags: [runtime, builtins, datetime]
status: Implemented (v0.1.7, 2026-04-19)
---

# Date Time Surface

**Source**: `crates/cjc-runtime/src/datetime.rs`.

A minimal, deterministic date/time surface exposed as builtins. Timestamps are represented as **epoch milliseconds (i64)** — not seconds — so that sub-second resolution is available without introducing a floating-point time type.

## Representation

- **Timestamp**: `i64`, milliseconds since the Unix epoch (1970-01-01T00:00:00Z).
- **Time zone**: all builtins operate in UTC. No local-time conversion is performed (local time is non-deterministic across machines and therefore outside the [[Determinism Contract]]).

## Builtins

### Construction and parsing
- `now()` — current wall-clock time as epoch ms. **Non-deterministic**; do not use inside reproducible pipelines. Reserved for CLI and logging paths.
- `parse_date(s: str, fmt: str) -> i64` — parse a formatted string to epoch ms. Returns `0` on malformed input (not an error). Callers should validate upstream when strictness matters.
- `date_format(ts: i64, fmt: str) -> str` — format an epoch-ms timestamp with the supported specifiers.

### Component extraction (all take an i64 epoch-ms timestamp, return i64)
- `year(ts)`
- `month(ts)`
- `day(ts)`
- `hour(ts)`
- `minute(ts)`
- `second(ts)`

### Arithmetic
- `date_add(ts: i64, amount: i64, unit: str) -> i64` — returns a new timestamp.
- `date_diff(ts_a: i64, ts_b: i64, unit: str) -> i64` — returns `ts_a - ts_b` in the requested unit.

## Unit strings

`date_add` and `date_diff` accept:

| Unit | Meaning |
|---|---|
| `"ms"` | milliseconds |
| `"s"` | seconds |
| `"min"` | minutes |
| `"h"` | hours |
| `"d"` | days |

No `"mo"` or `"y"` units — months and years are not fixed durations, so they are omitted from the deterministic arithmetic surface. Use component extraction + reconstruction to move across months or years.

## Format specifiers

`date_format` and `parse_date` recognize:

| Specifier | Meaning |
|---|---|
| `%Y` | 4-digit year |
| `%m` | 2-digit month (01-12) |
| `%d` | 2-digit day (01-31) |
| `%H` | 2-digit hour (00-23) |
| `%M` | 2-digit minute (00-59) |
| `%S` | 2-digit second (00-59) |

Example:

```cjcl
let ts: i64 = parse_date("2026-04-19 12:34:56", "%Y-%m-%d %H:%M:%S");
print(date_format(ts, "%Y/%m/%d"));   // "2026/04/19"
print(year(ts));                       // 2026

let tomorrow: i64 = date_add(ts, 1, "d");
print(date_diff(tomorrow, ts, "h"));   // 24
```

## Determinism notes

- All builtins except `now()` are pure functions of their inputs and are thus trivially deterministic.
- `parse_date` on malformed input returning `0` means callers never observe non-deterministic panic paths; the trade-off is that `0` is a valid epoch (1970-01-01T00:00:00Z). Check upstream if ambiguity matters.
- Component extraction uses integer arithmetic — no floating-point rounding.

## Test coverage

`tests/test_datetime.rs` — **19 tests, all passing**. Covers round-trip (parse → format), component extraction across month/year boundaries, unit-string handling for both `date_add` and `date_diff`, and malformed-input behaviour.

## Related

- [[Builtins Catalog]]
- [[Determinism Contract]]
- [[Wiring Pattern]]
- [[DataFrame DSL]] — `DateTime` column type
- [[Version History]]
