# LinkedIn post — Two-fold error system

**Image:** `error_system_linkedin.png` (1200x630, attach as link preview / single image)

---

## Post body

Just shipped a two-fold error system for CJC-Lang.

Most languages make you choose: Rust-grade precision (spans, carets, exact diagnostic codes) OR Elm-grade pedagogy (long-form explanations that teach you what the language is *for*).

CJC-Lang now does both — automatically.

The shape:
→ The inline diagnostic gives Rust-style precision (file:line:col + error code + label + caret underline)
→ A one-line `hint: cjcl explain EXXXX` footer is auto-attached at the universal emit chokepoint, on every error
→ `cjcl explain EXXXX` prints the long-form Elm-style explanation: what happened, *why CJC-Lang enforces this*, how to fix it, common pitfalls, related codes

The numbers:
• 82 / 82 error codes documented (100% coverage)
• Auto-discovery hint fires on every diagnostic — compile-time AND runtime
• 10 runtime emit sites migrated to a typed `Coded(ErrorCode, String)` variant — no more lossy strings
• 3 drift tests lock the 100% invariant — adding a new error code without an explanation now breaks the build
• 2,505 / 2,505 workspace tests pass; eval ↔ MIR-exec parity preserved on both backends

What makes it distinctive:
Every explanation connects the error to a *language-level value choice*. E1003 doesn't just say "add a type annotation" — it explains why CJC-Lang requires it (determinism contract, kernel selection). E8002 doesn't just say "don't divide by zero" — it explains why the language halts deterministically instead of producing silent ±Inf the way IEEE 754 does.

Compiler errors as a teaching surface. The error message isn't a dead end — it's the most-used learning resource the language has.

10 commits, all on master, ready for the observability and performance work that comes next.

`#compilers` `#programminglanguages` `#rust` `#languagedesign` `#developertools` `#cjclang`

---

## Post variants

### Shorter (under 1300 chars, fits the LinkedIn algorithm sweet spot)

Just shipped a two-fold error system for CJC-Lang.

Most languages make you choose: Rust-grade precision OR Elm-grade pedagogy. CJC-Lang now does both — automatically.

→ Inline diagnostic: file:line:col + error code + caret underline
→ Auto-attached hint: `cjcl explain EXXXX` (fires on every error, no opt-in)
→ Long-form explanation: what happened, *why CJC-Lang enforces this*, how to fix, common pitfalls

The numbers:
• 82/82 error codes documented (100% coverage)
• Auto-hint on compile-time AND runtime errors
• Typed runtime errors — no more lossy strings
• Drift tests lock the invariant: new codes must come with a written explanation or the build breaks
• 2,505/2,505 tests pass; eval ↔ MIR-exec parity preserved

What makes it distinctive: every explanation connects the error to a *language-level value choice*. E1003 doesn't just say "add a type annotation" — it explains why CJC-Lang requires it (determinism contract). E8002 doesn't just say "don't divide by zero" — it explains why the language halts deterministically instead of producing silent ±Inf.

Compiler errors as a teaching surface.

`#compilers` `#languagedesign` `#rust` `#cjclang`

---

## Notes for posting

- LinkedIn preview behaviour: when you attach a single image, it appears above the post text. Use the 1200x630 PNG.
- Best engagement window: Tue/Wed/Thu morning, US time.
- The phrase "Compiler errors as a teaching surface" is the load-bearing tagline — keep it.
- If anyone asks for the deep dive, the blog post you're planning is the right follow-up. Until that lands, point them at `docs/T0_INTERPRETER_PERF_HANDOFF.md` (no, wait — that's a different topic) — point them at the GitHub repo + `crates/cjc-diag/explanations/`.
