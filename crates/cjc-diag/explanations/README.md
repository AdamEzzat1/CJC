# Error-Code Explanations

This directory holds the long-form, Elm-style explanations of CJC-Lang error
codes. Each file is named after its error code (e.g., `E1003.md`, `W0001.md`)
and is embedded into the compiler binary at build time via `include_str!`
from `crates/cjc-diag/src/error_codes.rs`.

To view an explanation from the command line:

```
cjcl explain E1003
```

## Adding a New Explanation

1. Create `EXXXX.md` (or `WXXXX.md`) in this directory.
2. Open `crates/cjc-diag/src/error_codes.rs` and register the file in
   `ErrorCode::explanation()`:

   ```rust
   ErrorCode::EXXXX => Some(include_str!("../explanations/EXXXX.md")),
   ```

3. Run `cargo test -p cjc-diag` — the drift check
   `test_documented_codes_have_explanations` will surface any registration
   you missed.

## Style Guide

Each explanation should follow this skeleton, with **plain prose**, not bullet
soup:

```markdown
# <Title — one short sentence describing the situation>

## What happened
A plain-English description of what the compiler/runtime saw in your code,
written so a reader who has never used CJC-Lang before can follow.

## Why it matters
Connect the error back to a CJC-Lang value: determinism, numerical accuracy,
NoGC discipline, reproducibility, audit-safety. Explain *why this is enforced*
— not just *what* the rule is. This is the section that mentors the user.

## How to fix it
A "before" / "after" code example showing the minimal change. If the fix has
multiple options (most do), show 2-3 with one-line rationale each.

## Common pitfall
The one thing that catches people by surprise about this error. Often this is
a habit carried over from another language (Python, JavaScript, Rust, C++).

## Learn more
Pointers to relevant docs, ADRs, or design notes. Optional but encouraged.
```

The goal is for someone hitting this error to come away knowing not just
*how to fix it* but *why CJC-Lang is shaped this way*. Avoid jargon when a
plain word works. Show code examples that are short enough to read in one
glance.

## Pedagogy Layer Notes

The Elm-style pedagogy is the *second* layer of CJC-Lang's two-fold error
system. The first layer is the Rust-style precision diagnostic (with spans,
labels, fix suggestions) emitted at error time. The pedagogy layer is opt-in
via `cjcl explain` — most users see the precise diagnostic first and reach
for the explanation only when the diagnostic alone isn't enough.

This separation means:

- The precise diagnostic stays terse and machine-readable.
- The explanation can be longer, prose-heavy, and teaching-oriented.
- Experts who don't need the explanation never see it.
- Beginners who need scaffolding always have it one command away.
