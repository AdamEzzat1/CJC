---
title: Signal Processing
tags: [runtime, signal]
status: Implemented
---

# Signal Processing

**Source**: `crates/cjc-runtime/src/fft.rs`, `window.rs`, `differentiate.rs`, `integrate.rs`, `interpolate.rs`, `timeseries.rs`, `stationarity.rs`.

## FFT family

- `fft(x)` — complex-to-complex FFT
- `rfft(x)` — real-to-complex FFT
- `ifft(X)` — inverse FFT
- `psd(x)` — power spectral density

## Window functions

Standard windows: Hann, Hamming, Blackman, Tukey, Kaiser, etc. (see `window.rs` for the exact list).

## Rolling / streaming operations

From `crates/cjc-runtime/src/window.rs` and `timeseries.rs`:
- `window_sum`, `window_mean`, `window_var`
- Rolling aggregations (see also [[DataFrame DSL]])
- Kahan-stable across window shifts

## Numerical calculus

- `diff` — numerical differentiation
- `trapz` — trapezoidal integration
- `simps` — Simpson's rule
- `integrate` — higher-order

## Time series

- Stationarity tests (ADF — also in [[Hypothesis Tests]])
- Autocorrelation

## Related

- [[Tensor Runtime]]
- [[Statistics and Distributions]]
- [[Hypothesis Tests]]
- [[DataFrame DSL]]
