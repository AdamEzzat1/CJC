---
title: Regex Engine
tags: [data, runtime, regex]
status: Implemented
updated: 2026-04-18
version: v0.1.6
---

# Regex Engine

**Crate**: `cjc-regex` — `crates/cjc-regex/src/lib.rs` (~2,430 lines, single file).

## Summary

An NFA-based regex engine using Thompson's construction and epsilon-closure simulation. **No backtracking** — which means O(n·m) worst-case matching in input length × pattern length, with no catastrophic regex cases possible. Fully deterministic: identical pattern + flags + haystack always produces bit-identical results.

## Why NFA and not backtracking

- **Performance predictability** — O(n·m) worst case is guaranteed; backtracking PCRE can be exponential.
- **Determinism** — the dual-list simulation is order-independent, no hidden state.
- **NoGC safety** — no per-match heap allocations in the hot path; safe in NoGC-verified code paths.
- **Simplicity** — no hash maps, no random iteration, no FMA in the inner loop.

## Architecture

The engine uses a **fragment-based compilation** model:

1. **Compile**: `Compiler` translates a pattern string into an `Nfa` (a `Vec<NfaNode>`) using Pratt-style recursive descent. Each sub-expression returns a `(start, end)` fragment pair; the `end` is an `Epsilon(usize::MAX)` placeholder that gets `patch()`-ed to the next fragment's start as the chain is assembled.
2. **Execute (fast path)**: `nfa_match_at()` runs a dual-list Thompson simulation — a `current` state set and a `next` state set, swapped each byte. Zero-width assertions (anchors, `\b`, `\B`) are resolved during the epsilon-closure step (`add_state()`), not during the main scan. `Save` nodes are treated as epsilon in this path — no allocation.
3. **Execute (capture path)**: `pike_match_at()` runs a **Pike VM** — each thread carries its own `slots: Vec<Option<usize>>` alongside the state id. At `Split` nodes the slot array is cloned for each branch; at `Save(n)` nodes `slots[n] = Some(pos)` is written. The first thread to reach a state ID "wins" (leftmost-longest priority), preserving O(n·m).

### NfaNode variants (15 total)

| Variant | Purpose |
|---------|---------|
| `Byte(u8)` | Match a single byte |
| `Class(ByteClass)` | Match a 256-bit bitmap character class |
| `AnyByte` | Match any byte (dotall mode) |
| `AnyByteNoNl` | Match any byte except `\n` (default `.`) |
| `Split(usize, usize)` | Epsilon fork — greedy or lazy depending on arm order |
| `Epsilon(usize)` | Unconditional epsilon transition |
| `Accept` | Accepting state |
| `AnchorStart` | `^` — start of input or line (multiline-aware) |
| `AnchorEnd` | `$` — end of input or line (multiline-aware) |
| `WordBoundary` | `\b` |
| `NonWordBoundary` | `\B` |
| `AbsoluteStart` | `\A` — always absolute start, ignores multiline |
| `AbsoluteEnd` | `\z` — always absolute end |
| `AbsoluteEndBeforeNewline` | `\Z` — end or before final newline |
| `Save(usize)` | Store current byte position into capture slot `n` (Pike VM only) |

### Lazy quantifiers

Lazy quantifiers (`*?`, `+?`, `??`, `{n,m}?`) reverse the arm order of the `Split` node and set `has_lazy = true` on the NFA. The executor short-circuits on the **first** `Accept` reached rather than waiting for the longest match.

### Counted repetition — node cloning

`{n}`, `{n,}`, `{n,m}` are compiled by **unrolling** the atom fragment:
- The atom's NFA nodes are snapshotted before any patching.
- Each required copy clones the snapshot (remapping internal indices by delta).
- Optional copies use `Split` nodes between the atom clone and an `Epsilon(MAX)` out-edge.
- This preserves the O(n·m) guarantee at the cost of a larger NFA (bounded by `MAX_NODES = 65536`).

## Supported Syntax

```text
.            any byte (or any byte except \n without s flag)
\d \w \s     ASCII shorthand classes
\D \W \S     negated shorthand classes
[abc]        literal character class
[^abc]       negated character class
[a-z]        range inside class
[:alpha:]    POSIX character class inside [...] (double-bracket form in patterns)
a|b          alternation
(...)        capturing group (tracked by Pike VM; group 0 = full match, 1+ = groups)
(?:...)      non-capturing group (no slot allocated)
(?P<name>...)  named capturing group (Python syntax)
(?<name>...)   named capturing group (PCRE syntax)
(?i)(?m)     standalone inline flags (i, m, s, x)
(?i:...)     flag-scoped group (flags apply only within)
*  +  ?      greedy quantifiers
*? +? ??     lazy (non-greedy) quantifiers
{n}          exactly n repetitions
{n,}         n or more repetitions
{n,m}        between n and m repetitions
{n}?         lazy counted
^  $         start/end of input (or line in m mode)
\A           absolute start of input (ignores m flag)
\z           absolute end of input
\Z           end or before final newline
\b  \B       word / non-word boundary
\\           literal backslash
\xNN         hex byte escape
\uNNNN       Unicode codepoint (UTF-8 encoded)
\u{NNNN}     Unicode codepoint (braced form, up to 6 hex digits)
```

### POSIX character classes

Available inside `[...]` using double-bracket syntax, e.g. `[:alpha:]` written as `[` + `[:alpha:]` + `]`:

| Class | Matches |
|-------|---------|
| `[:alpha:]` | `[A-Za-z]` |
| `[:digit:]` | `[0-9]` |
| `[:alnum:]` | `[A-Za-z0-9]` |
| `[:space:]` | space, tab, newline, CR, FF, VT |
| `[:blank:]` | space, tab |
| `[:upper:]` | `[A-Z]` |
| `[:lower:]` | `[a-z]` |
| `[:punct:]` | printable non-alphanumeric ASCII |
| `[:print:]` | `[ -~]` |
| `[:graph:]` | `[!-~]` |
| `[:cntrl:]` | control characters (0x00–0x1F, 0x7F) |
| `[:xdigit:]` | `[0-9a-fA-F]` |

## Flags

Pass flags as a string (e.g. `"im"` for case-insensitive multiline):

| Flag | Meaning |
|------|---------|
| `i` | Case-insensitive (ASCII only) |
| `m` | Multiline (`^`/`$` match line boundaries) |
| `s` | Dotall (`.` matches `\n`) |
| `x` | Extended (whitespace ignored, `#` comments) |
| `g` | Global (for `find_all`/`split`) |

Inline flags inside the pattern (`(?i)`, `(?m:...)`) are also supported and are merged with the flags string at runtime.

## Public API

```rust
// Boolean match
is_match(pattern: &str, flags: &str, haystack: &[u8]) -> bool

// First match span
find(pattern: &str, flags: &str, haystack: &[u8]) -> Option<(usize, usize)>

// All non-overlapping match spans
find_all(pattern: &str, flags: &str, haystack: &[u8]) -> Vec<(usize, usize)>

// Split on pattern, returning segment spans
split(pattern: &str, flags: &str, haystack: &[u8]) -> Vec<(usize, usize)>

// Structured match result
find_match(pattern: &str, flags: &str, haystack: &[u8]) -> Option<MatchResult>
find_all_matches(pattern: &str, flags: &str, haystack: &[u8]) -> Vec<MatchResult>

// Human-readable NFA description (debugging)
regex_explain(pattern: &str, flags: &str) -> Result<String, String>

// Capture groups — first match
find_captures(pattern: &str, flags: &str, haystack: &[u8]) -> Option<CaptureResult>

// Capture groups — all non-overlapping matches
find_all_captures(pattern: &str, flags: &str, haystack: &[u8]) -> Vec<CaptureResult>

// Number of capturing groups in the pattern
capture_count(pattern: &str, flags: &str) -> usize
```

### MatchResult

```rust
pub struct MatchResult {
    pub start: usize,  // inclusive byte offset
    pub end: usize,    // exclusive byte offset
}

impl MatchResult {
    fn extract<'a>(&self, haystack: &'a [u8]) -> &'a [u8]
    fn extract_str<'a>(&self, haystack: &'a [u8]) -> Option<&'a str>
    fn len(&self) -> usize
    fn is_empty(&self) -> bool
}
```

### CaptureResult

```rust
pub struct CaptureResult {
    pub full: MatchResult,               // group 0 — the full match
    pub groups: Vec<Option<Capture>>,    // index 0 = full match, 1+ = capture groups
    pub names: BTreeMap<String, usize>,  // name → group index (1-based)
}

impl CaptureResult {
    fn get(&self, idx: usize) -> Option<&Capture>
    fn get_named(&self, name: &str) -> Option<&Capture>
}

pub struct Capture {
    pub start: usize,
    pub end: usize,
    fn extract<'a>(&self, haystack: &'a [u8]) -> &'a [u8]
    fn extract_str<'a>(&self, haystack: &'a [u8]) -> Option<&'a str>
}
```

The `get(0)` / `full` field always holds the overall match span. `get(1)`, `get(2)`, … return individual groups (or `None` if the group did not participate in the match, e.g., the unmatched arm of an alternation). Named groups are accessible via `get_named("year")` in addition to their numeric index.

## Regex Composition Builtins

Three builtins are wired via the standard [[Wiring Pattern]] in all three locations (`cjc-runtime/builtins.rs`, `cjc-eval`, `cjc-mir-exec`):

| Builtin | Signature | Returns |
|---------|-----------|---------|
| `regex_or(p1, p2, ...)` | One or more pattern strings | `"(?:p1)|(?:p2)|..."` |
| `regex_seq(p1, p2, ...)` | One or more pattern strings | `"(?:p1)(?:p2)..."` |
| `regex_explain(pattern, flags?)` | Pattern + optional flags | Human-readable NFA description string |
| `regex_captures(pattern, flags, text)` | Pattern + flags + text string | Array of strings: `[full_match, group1, group2, ...]`, empty array on no match |
| `regex_named_capture(pattern, flags, text, name)` | Pattern + flags + text + group name | Captured string for the named group, or `""` on no match |
| `regex_capture_count(pattern, flags)` | Pattern + flags | Integer — number of capturing groups (not counting group 0) |

Example usage in CJC-Lang:

```cjcl
let digits_or_alpha: str = regex_or("\\d+", "[a-zA-Z]+");
let matched: bool = "hello123" ~= /\d+|[a-zA-Z]+/;

let year_pattern: str = regex_seq("\\d{2}", "\\d{2}");
let year: str = str_extract("born 1994", year_pattern, "");

// Capture groups
let parts: Any = regex_captures("(\\d{4})-(\\d{2})-(\\d{2})", "", "2026-04-18");
// parts[0] = "2026-04-18", parts[1] = "2026", parts[2] = "04", parts[3] = "18"

// Named captures
let yr: str = regex_named_capture("(?P<year>\\d{4})-(?P<month>\\d{2})", "", "2026-04", "year");
// yr = "2026"

// Count groups
let n: Any = regex_capture_count("(a)(b)(c)", "");
// n = 3
```

## Safety Limits

| Constant | Value | Enforced at |
|----------|-------|-------------|
| `MAX_PATTERN_LEN` | 4,096 bytes | `Compiler::compile()` |
| `MAX_NODES` | 65,536 nodes | `Compiler::compile()` after build |

Both return `Err(String)` — no panics, no silent truncation.

## Surface Syntax Integration

CJC-Lang has dedicated regex literal syntax: `/pattern/flags`. Plus operators `~=` (match) and `!~` (not-match):

```cjcl
if name ~= /^[A-Z][a-z]+$/ { ... }
if email !~ /[@]/ { print("invalid email") }

// With flags
if text ~= /hello/i { print("case-insensitive match") }

// In TidyView / stringr functions
let found: bool = str_detect(name, "[:alpha:]+", "");  // double-bracket in real pattern
let year: str  = str_extract(date_str, "\\d{4}", "");
```

See [[Syntax]] and [[Operators and Precedence]].

## Explicitly Not Supported (by design)

These features require backtracking and are fundamentally incompatible with the Thompson NFA guarantee:

| Feature | Reason |
|---------|--------|
| Lookahead / lookbehind `(?=...)` | Requires bounded-width constraint or backtracking |
| Backreferences `\1` | NP-hard in general; cannot be modelled by NFA |
| Possessive quantifiers `*+` | Requires backtracking stack |
| Atomic groups `(?>...)` | Requires backtracking stack |
| Conditional patterns `(?(cond)...)` | PCRE-only; requires call stack |
| Unicode property classes `\p{L}` | Would require ~100 KB Unicode tables |

Capture groups `(...)`, named captures `(?P<name>...)` / `(?<name>...)`, and the Pike VM that powers them are **fully implemented** (v0.1.6).

## Test Coverage

| Suite | File | Count |
|-------|------|-------|
| Engine internal | `crates/cjc-regex/src/lib.rs` | 77 |
| Engine doc-tests | `crates/cjc-regex/src/lib.rs` | 7 |
| Integration (Lexer/Parser/Eval/MIR) | `tests/test_regex.rs` | 77 |
| Feature engine tests | `tests/regex/engine.rs` | 75 |
| Property tests (proptest) | `tests/regex/property.rs` | 8 |
| TidyView integration + parity | `tests/regex/tidyview.rs` | 31 |
| Fuzz targets (bolero) | `tests/regex/fuzz.rs` | 4 |
| Capture groups + Pike VM | `tests/regex/captures.rs` | 41 |
| **Total** | | **320** |

## Related

- [[Builtins Catalog]]
- [[TidyView Architecture]]
- [[DataFrame DSL]]
- [[Syntax]]
- [[Operators and Precedence]]
- [[Wiring Pattern]]
- [[Determinism Contract]]
