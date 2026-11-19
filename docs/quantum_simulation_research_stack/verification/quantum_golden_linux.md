# Cross-platform check of `crates/cjc-quantum/tests/cross_platform_golden.rs`

Run 2026-09-25.

| Platform | Result |
|---|---|
| Windows 11 x86-64 (UCRT), `cargo test --release -p cjc-quantum --test cross_platform_golden` | 2 passed. The golden hash `0x65eead6855ab878d` was recorded here |
| Debian 12 x86-64 (glibc 2.36), Docker `rust:1-bookworm`, `--network none` | 2 passed (same hash) |
| macOS | Not run locally. The test runs in the ubuntu/windows/macos CI matrix (`cargo test --workspace`) |

## How the Linux run was done

The container had no network access (DNS to crates.io failed), so it used a
minimal copy of the workspace with no external dependencies:
- **Crates:** `cjc-quantum`, `cjc-runtime`, `cjc-repro`, and `cjc-regex`, with a root manifest that supplies the `[workspace.package]` fields.
- **`rayon` removed from the copied `cjc-runtime` manifest only.** Its `parallel` feature became an empty list. `rayon` backs that optional feature, and `cjc-quantum` compiled and ran without it.

The repository's manifests were not changed.

For comparison, the platform libm made Windows and Linux disagree on 6.0% of
gate angles (`libm_bits.*`). With the libm this hash would be expected to
differ between the two platforms; that counterfactual was not run.
