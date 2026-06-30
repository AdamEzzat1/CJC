# Instagram post — Two-fold error system

**Image:** `error_system_instagram.png` (1080x1080, single-image post)

---

## Caption

Compiler errors that teach.

Most languages make you choose between Rust-grade precision (spans, carets, exact codes) and Elm-grade pedagogy (long-form explanations that teach you what the language is *for*).

CJC-Lang now does both — automatically, on every error the compiler or runtime can emit.

→ 82/82 error codes documented (100%)
→ Auto-hint fires on compile-time AND runtime errors
→ Typed runtime errors, no more lossy strings
→ Drift tests lock the invariant: new codes must come with a written explanation or the build breaks

Every explanation connects the error to a *why*. E1003 doesn't just say "add a type annotation" — it explains why CJC-Lang requires it (the determinism contract, kernel selection). E8002 doesn't just say "don't divide by zero" — it explains why the language halts deterministically instead of producing silent ±Inf.

The error message isn't a dead end. It's the most-used learning resource the language has.

10 commits, 2,505/2,505 tests passing, parity preserved across both executors.

. . .

#compilers #programminglanguages #languagedesign #rust #developertools #softwareengineering #devtools #programmer #coding #opensource #cjclang

---

## Caption variants

### Shorter (Stories / Reels caption)

Compiler errors that teach.

82/82 error codes in CJC-Lang now ship with long-form explanations. Every error tells you what happened, *why the language enforces this*, and how to fix it.

The error message isn't a dead end. It's the most-used learning resource the language has.

#compilers #languagedesign #rust #cjclang

---

## Notes for posting

- Instagram crops to square for the feed by default. The image is already 1080x1080 so no cropping needed.
- First sentence "Compiler errors that teach." is the hook — keep it as line 1.
- Hashtags: place them as a separate paragraph at the end (or in a first comment) so they don't visually clutter the caption. Instagram caps hashtags at 30 per post; we're at 11.
- Consider a carousel follow-up: slide 1 = this image, slides 2-5 = before/after of specific error codes (E1003, E4001, E8001, E8002). Each slide can be a screenshot of `cjcl explain EXXXX` output.
