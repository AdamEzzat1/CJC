# Error-Code Explanations

The long-form explanations of CJC-Lang error codes (the Elm-style
pedagogy layer of the two-fold error system) live next to the
diagnostic source for build-time embedding:

> [`crates/cjc-diag/explanations/`](../../crates/cjc-diag/explanations/)

## Reading an Explanation

```
cjcl explain E1003
```

The CLI dispatches on argument shape: anything matching `EXXXX` or
`WXXXX` is treated as an error-code lookup; anything else is treated as
a `.cjcl` filename for HIR lowering inspection (the original behaviour
of `cjcl explain`).

## Adding an Explanation

See the style guide at
[`crates/cjc-diag/explanations/README.md`](../../crates/cjc-diag/explanations/README.md).
New explanations are checked at test time — adding the markdown file
without registering it in `ErrorCode::explanation()` will surface in the
drift-check test.

## Why Not Here?

This directory used to be the proposed location. It moved next to the
crate source because Rust's `include_str!` cannot embed assets from
outside the crate boundary at publish time — `cargo publish` would
succeed but downstream `cargo install` of `cjc-lang` would fail with a
missing-file error. Keeping the explanations inside `cjc-diag` makes
them automatically ship with the published crate tarball.
