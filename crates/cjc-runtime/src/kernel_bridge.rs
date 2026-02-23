

// ---------------------------------------------------------------------------
// 2d. Raw-Pointer Kernel Bridge — bypass interpreter overhead for hot loops
// ---------------------------------------------------------------------------

/// Raw-pointer kernel functions that operate directly on f64 slices.
///
/// These bypass the `Value::Tensor` wrapper, operating on contiguous `&[f64]`
/// data. The interpreter resolves tensor pointers once at call entry, then
/// dispatches to these zero-overhead kernels.
///
/// All functions here are safe Rust — they accept slices, not raw pointers.
/// The "raw pointer" concept means: the interpreter does one `to_vec()` or
/// `buffer.borrow()` at entry, then passes the contiguous slice through.
pub mod kernel {
    use cjc_repro::{kahan_sum_f64, KahanAccumulatorF64};

    /// Matrix multiply: C[m,n] = A[m,k] × B[k,n] with Kahan-summed dots.
    ///
    /// `a`, `b` are row-major contiguous slices; `c` is the output buffer
    /// (must be pre-allocated to `m * n`).
    ///
    /// Uses in-place `KahanAccumulatorF64` — zero heap allocation per dot product.
    #[inline]
    pub fn matmul_raw(
        a: &[f64], b: &[f64], c: &mut [f64],
        m: usize, k: usize, n: usize,
    ) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(c.len(), m * n);
        for i in 0..m {
            for j in 0..n {
                let mut acc = KahanAccumulatorF64::new();
                for p in 0..k {
                    acc.add(a[i * k + p] * b[p * n + j]);
                }
                c[i * n + j] = acc.finalize();
            }
        }
    }

    /// Softmax over the last dimension of a contiguous buffer.
    ///
    /// `data` is the input (length = `outer * n`), `out` is the output.
    /// Applies two-pass stable softmax per row of length `n`.
    #[inline]
    pub fn softmax_raw(data: &[f64], out: &mut [f64], outer: usize, n: usize) {
        debug_assert_eq!(data.len(), outer * n);
        debug_assert_eq!(out.len(), outer * n);
        for row in 0..outer {
            let start = row * n;
            let slice = &data[start..start + n];

            // Pass 1: max
            let mut max_val = f64::NEG_INFINITY;
            for &v in slice {
                if v > max_val { max_val = v; }
            }

            // Pass 2: exp + Kahan sum
            let mut sum = 0.0f64;
            let mut comp = 0.0f64;
            for i in 0..n {
                let e = (slice[i] - max_val).exp();
                out[start + i] = e;
                let y = e - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }

            // Normalize
            if sum == 0.0 {
                let uniform = 1.0 / n as f64;
                for i in 0..n {
                    out[start + i] = uniform;
                }
            } else {
                for i in 0..n {
                    out[start + i] /= sum;
                }
            }
        }
    }

    /// Linear projection: Y[outer, out_f] = X[outer, in_f] @ W^T[out_f, in_f] + bias[out_f]
    #[inline]
    pub fn linear_raw(
        x: &[f64], w: &[f64], bias: &[f64], out: &mut [f64],
        outer: usize, in_f: usize, out_f: usize,
    ) {
        debug_assert_eq!(x.len(), outer * in_f);
        debug_assert_eq!(w.len(), out_f * in_f);
        debug_assert_eq!(bias.len(), out_f);
        debug_assert_eq!(out.len(), outer * out_f);
        for row in 0..outer {
            let x_start = row * in_f;
            let x_slice = &x[x_start..x_start + in_f];
            let y_start = row * out_f;
            for j in 0..out_f {
                let w_start = j * in_f;
                let mut acc = KahanAccumulatorF64::new();
                for p in 0..in_f {
                    acc.add(x_slice[p] * w[w_start + p]);
                }
                out[y_start + j] = acc.finalize() + bias[j];
            }
        }
    }

    /// Layer normalization over the last dimension.
    ///
    /// For each row of length `n`: normalize to mean=0, var=1, then
    /// scale by gamma and shift by beta.
    #[inline]
    pub fn layer_norm_raw(
        data: &[f64], gamma: &[f64], beta: &[f64], out: &mut [f64],
        outer: usize, n: usize, eps: f64,
    ) {
        debug_assert_eq!(data.len(), outer * n);
        debug_assert_eq!(gamma.len(), n);
        debug_assert_eq!(beta.len(), n);
        debug_assert_eq!(out.len(), outer * n);
        for row in 0..outer {
            let start = row * n;
            let slice = &data[start..start + n];

            // Mean (Kahan)
            let mean = kahan_sum_f64(slice) / n as f64;

            // Variance (Kahan)
            let diffs: Vec<f64> = slice.iter().map(|&x| (x - mean) * (x - mean)).collect();
            let var = kahan_sum_f64(&diffs) / n as f64;
            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..n {
                out[start + i] = (slice[i] - mean) * inv_std * gamma[i] + beta[i];
            }
        }
    }

    /// ReLU: max(0, x) element-wise.
    #[inline]
    pub fn relu_raw(data: &[f64], out: &mut [f64]) {
        debug_assert_eq!(data.len(), out.len());
        for (o, &x) in out.iter_mut().zip(data.iter()) {
            *o = if x > 0.0 { x } else { 0.0 };
        }
    }

    /// Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #[inline]
    pub fn gelu_raw(data: &[f64], out: &mut [f64]) {
        debug_assert_eq!(data.len(), out.len());
        let sqrt_2_over_pi: f64 = (2.0 / std::f64::consts::PI).sqrt();
        for (o, &x) in out.iter_mut().zip(data.iter()) {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            *o = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    /// 1D convolution with stride=1, no padding ("valid" mode).
    ///
    /// `signal`: input signal of length `signal_len`.
    /// `filters`: `[out_channels, kernel_size]` row-major.
    /// `bias`: per-channel bias of length `out_channels`.
    /// `out`: output buffer `[out_channels, out_len]`, `out_len = signal_len - kernel_size + 1`.
    /// Uses Kahan summation for deterministic dot products.
    pub fn conv1d_raw(
        signal: &[f64], filters: &[f64], bias: &[f64], out: &mut [f64],
        signal_len: usize, out_channels: usize, kernel_size: usize,
    ) {
        debug_assert!(signal_len >= kernel_size);
        let out_len = signal_len - kernel_size + 1;
        debug_assert_eq!(signal.len(), signal_len);
        debug_assert_eq!(filters.len(), out_channels * kernel_size);
        debug_assert_eq!(bias.len(), out_channels);
        debug_assert_eq!(out.len(), out_channels * out_len);

        for ch in 0..out_channels {
            let filter_start = ch * kernel_size;
            let filter_slice = &filters[filter_start..filter_start + kernel_size];
            let out_row_start = ch * out_len;
            for pos in 0..out_len {
                let products: Vec<f64> = (0..kernel_size)
                    .map(|k| signal[pos + k] * filter_slice[k])
                    .collect();
                out[out_row_start + pos] = kahan_sum_f64(&products) + bias[ch];
            }
        }
    }

    /// 1D convolution on a sliding window of a circular buffer.
    ///
    /// Extracts the most recent `window_size` samples from `buffer`
    /// (handling wrap-around at `write_pos`) into `window`, then
    /// delegates to `conv1d_raw`.
    pub fn conv1d_circular(
        buffer: &[f64], write_pos: usize, window_size: usize,
        window: &mut [f64],
        filters: &[f64], bias: &[f64], out: &mut [f64],
        out_channels: usize, kernel_size: usize,
    ) {
        let buf_len = buffer.len();
        debug_assert!(window_size <= buf_len);
        debug_assert_eq!(window.len(), window_size);

        let start = if write_pos >= window_size {
            write_pos - window_size
        } else {
            buf_len - (window_size - write_pos)
        };
        for i in 0..window_size {
            window[i] = buffer[(start + i) % buf_len];
        }

        conv1d_raw(window, filters, bias, out, window_size, out_channels, kernel_size);
    }

    // -- Phase 7: 2D Spatial Kernels ------------------------------------------

    /// 2D convolution — NCHW layout, valid mode (no padding), configurable stride.
    ///
    /// # Layout
    /// - `input`:   `[N, C_in, H_in, W_in]`  row-major contiguous
    /// - `filters`: `[C_out, C_in, kH, kW]`  row-major contiguous
    /// - `bias`:    `[C_out]`
    /// - `out`:     `[N, C_out, H_out, W_out]`  pre-allocated by caller
    ///
    /// where `H_out = (H_in - kH) / stride + 1` and `W_out = (W_in - kW) / stride + 1`.
    ///
    /// # Numerical contract
    /// Every kernel-to-patch dot product uses `BinnedAccumulatorF64`, guaranteeing
    /// bit-identical results regardless of stride, batch size, or channel count.
    ///
    /// # NoGC guarantee
    /// All index arithmetic uses `u64` before narrowing to `usize`, preventing
    /// overflow for high-resolution inputs (e.g., 8192×8192). The output buffer
    /// is caller-allocated; this function performs zero heap allocations.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_raw(
        input:   &[f64],
        filters: &[f64],
        bias:    &[f64],
        out:     &mut [f64],
        n: usize, c_in: usize, h_in: usize, w_in: usize,
        c_out: usize, kh: usize, kw: usize,
        stride: usize,
    ) {
        use crate::accumulator::BinnedAccumulatorF64;

        let h_out: u64 = ((h_in - kh) / stride + 1) as u64;
        let w_out: u64 = ((w_in - kw) / stride + 1) as u64;

        // Strides in the input tensor (NCHW, row-major).
        let s_n:   u64 = (c_in  * h_in * w_in) as u64;
        let s_cin: u64 = (h_in  * w_in) as u64;
        let s_hin: u64 = w_in as u64;

        // Strides in the filter tensor [C_out, C_in, kH, kW].
        let f_cout: u64 = (c_in * kh * kw) as u64;
        let f_cin:  u64 = (kh * kw) as u64;
        let f_kh:   u64 = kw as u64;

        // Strides in the output tensor (NCHW).
        let o_n:    u64 = c_out as u64 * h_out * w_out;
        let o_cout: u64 = h_out * w_out;

        debug_assert_eq!(input.len(),   n * c_in  * h_in * w_in);
        debug_assert_eq!(filters.len(), c_out * c_in * kh * kw);
        debug_assert_eq!(bias.len(),    c_out);
        debug_assert_eq!(out.len(),     n * c_out * h_out as usize * w_out as usize);

        for bn in 0..n as u64 {
            for co in 0..c_out as u64 {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = BinnedAccumulatorF64::new();

                        // Sum over input channels and kernel spatial extent.
                        for ci in 0..c_in as u64 {
                            for ki in 0..kh as u64 {
                                for kj in 0..kw as u64 {
                                    let ih: u64 = oh * stride as u64 + ki;
                                    let iw: u64 = ow * stride as u64 + kj;

                                    let inp_idx = (bn  * s_n
                                                 + ci  * s_cin
                                                 + ih  * s_hin
                                                 + iw) as usize;
                                    let flt_idx = (co  * f_cout
                                                 + ci  * f_cin
                                                 + ki  * f_kh
                                                 + kj) as usize;

                                    acc.add(input[inp_idx] * filters[flt_idx]);
                                }
                            }
                        }

                        let out_idx = (bn * o_n
                                     + co * o_cout
                                     + oh * w_out
                                     + ow) as usize;
                        out[out_idx] = acc.finalize() + bias[co as usize];
                    }
                }
            }
        }
    }

    /// 2D convolution using dispatched summation strategy.
    ///
    /// Identical to `conv2d_raw` but selects Kahan or Binned based on the
    /// reduction context.  Useful when callers want runtime-configurable
    /// accumulation precision.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_dispatched(
        input:   &[f64],
        filters: &[f64],
        bias:    &[f64],
        out:     &mut [f64],
        n: usize, c_in: usize, h_in: usize, w_in: usize,
        c_out: usize, kh: usize, kw: usize,
        stride: usize,
        ctx: &crate::dispatch::ReductionContext,
    ) {
        let h_out = (h_in - kh) / stride + 1;
        let w_out = (w_in - kw) / stride + 1;

        let s_n   = c_in  * h_in * w_in;
        let s_cin = h_in  * w_in;
        let s_hin = w_in;

        let f_cout = c_in * kh * kw;
        let f_cin  = kh * kw;
        let f_kh   = kw;

        let o_n    = c_out * h_out * w_out;
        let o_cout = h_out * w_out;

        for bn in 0..n {
            for co in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut terms = Vec::with_capacity(c_in * kh * kw);
                        for ci in 0..c_in {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih = oh * stride + ki;
                                    let iw = ow * stride + kj;
                                    let inp_idx = bn * s_n  + ci * s_cin + ih * s_hin + iw;
                                    let flt_idx = co * f_cout + ci * f_cin + ki * f_kh + kj;
                                    terms.push(input[inp_idx] * filters[flt_idx]);
                                }
                            }
                        }
                        let out_idx = bn * o_n + co * o_cout + oh * w_out + ow;
                        out[out_idx] =
                            crate::dispatch::dispatch_sum_f64(&terms, ctx) + bias[co];
                    }
                }
            }
        }
    }

    /// 2D max-pooling — NCHW layout, stride = pool_size (non-overlapping).
    ///
    /// - `input`: `[N, C, H_in, W_in]`
    /// - `out`:   `[N, C, H_in/ph, W_in/pw]`  (floor division, pre-allocated)
    ///
    /// All index arithmetic uses `u64` to support large spatial extents.
    pub fn maxpool2d_raw(
        input:  &[f64],
        out:    &mut [f64],
        n: usize, c: usize, h_in: usize, w_in: usize,
        ph: usize, pw: usize,
    ) {
        let h_out: u64 = (h_in / ph) as u64;
        let w_out: u64 = (w_in / pw) as u64;

        let s_n:   u64 = (c * h_in * w_in) as u64;
        let s_c:   u64 = (h_in * w_in) as u64;
        let s_hin: u64 = w_in as u64;

        let o_n:   u64 = (c as u64) * h_out * w_out;
        let o_c:   u64 = h_out * w_out;

        debug_assert_eq!(input.len(), n * c * h_in * w_in);
        debug_assert_eq!(out.len(),   n * c * h_out as usize * w_out as usize);

        for bn in 0..n as u64 {
            for ch in 0..c as u64 {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f64::NEG_INFINITY;
                        for pi in 0..ph as u64 {
                            for pj in 0..pw as u64 {
                                let ih: u64 = oh * ph as u64 + pi;
                                let iw: u64 = ow * pw as u64 + pj;
                                let idx = (bn * s_n + ch * s_c + ih * s_hin + iw) as usize;
                                let v = input[idx];
                                if v > max_val { max_val = v; }
                            }
                        }
                        let o_idx = (bn * o_n + ch * o_c + oh * w_out + ow) as usize;
                        out[o_idx] = max_val;
                    }
                }
            }
        }
    }

    /// Max-pooling over 1D signal, stride = pool_size.
    pub fn maxpool1d_raw(data: &[f64], out: &mut [f64], data_len: usize, pool_size: usize) {
        debug_assert_eq!(data.len(), data_len);
        let out_len = data_len / pool_size;
        debug_assert_eq!(out.len(), out_len);
        for i in 0..out_len {
            let start = i * pool_size;
            let mut max_val = data[start];
            for j in 1..pool_size {
                let v = data[start + j];
                if v > max_val { max_val = v; }
            }
            out[i] = max_val;
        }
    }

    // -- Dispatched kernel variants (Milestone 2.7) ---------------------------

    /// Matrix multiply using dispatched summation strategy.
    ///
    /// Identical to `matmul_raw` but uses `dispatch_dot_f64` for each dot product,
    /// selecting Kahan or Binned based on the reduction context.
    #[inline]
    pub fn matmul_dispatched(
        a: &[f64], b: &[f64], c: &mut [f64],
        m: usize, k: usize, n: usize,
        ctx: &crate::dispatch::ReductionContext,
    ) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(c.len(), m * n);
        for i in 0..m {
            for j in 0..n {
                // Collect column from B for the dot product.
                let a_row = &a[i * k..(i + 1) * k];
                let b_col: Vec<f64> = (0..k).map(|p| b[p * n + j]).collect();
                c[i * n + j] = crate::dispatch::dispatch_dot_f64(a_row, &b_col, ctx);
            }
        }
    }

    /// Linear projection using dispatched summation.
    #[inline]
    pub fn linear_dispatched(
        x: &[f64], w: &[f64], bias: &[f64], out: &mut [f64],
        outer: usize, in_f: usize, out_f: usize,
        ctx: &crate::dispatch::ReductionContext,
    ) {
        debug_assert_eq!(x.len(), outer * in_f);
        debug_assert_eq!(w.len(), out_f * in_f);
        debug_assert_eq!(bias.len(), out_f);
        debug_assert_eq!(out.len(), outer * out_f);
        for row in 0..outer {
            let x_start = row * in_f;
            let x_slice = &x[x_start..x_start + in_f];
            let y_start = row * out_f;
            for j in 0..out_f {
                let w_start = j * in_f;
                let w_slice = &w[w_start..w_start + in_f];
                out[y_start + j] = crate::dispatch::dispatch_dot_f64(x_slice, w_slice, ctx) + bias[j];
            }
        }
    }

    /// Layer normalization using dispatched summation for mean/variance.
    #[inline]
    pub fn layer_norm_dispatched(
        data: &[f64], gamma: &[f64], beta: &[f64], out: &mut [f64],
        outer: usize, n: usize, eps: f64,
        ctx: &crate::dispatch::ReductionContext,
    ) {
        debug_assert_eq!(data.len(), outer * n);
        debug_assert_eq!(gamma.len(), n);
        debug_assert_eq!(beta.len(), n);
        debug_assert_eq!(out.len(), outer * n);
        for row in 0..outer {
            let start = row * n;
            let slice = &data[start..start + n];

            let mean = crate::dispatch::dispatch_sum_f64(slice, ctx) / n as f64;

            let diffs: Vec<f64> = slice.iter().map(|&x| (x - mean) * (x - mean)).collect();
            let var = crate::dispatch::dispatch_sum_f64(&diffs, ctx) / n as f64;
            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..n {
                out[start + i] = (slice[i] - mean) * inv_std * gamma[i] + beta[i];
            }
        }
    }

    /// 1D convolution using dispatched summation.
    pub fn conv1d_dispatched(
        signal: &[f64], filters: &[f64], bias: &[f64], out: &mut [f64],
        signal_len: usize, out_channels: usize, kernel_size: usize,
        ctx: &crate::dispatch::ReductionContext,
    ) {
        debug_assert!(signal_len >= kernel_size);
        let out_len = signal_len - kernel_size + 1;
        for ch in 0..out_channels {
            let filter_start = ch * kernel_size;
            let filter_slice = &filters[filter_start..filter_start + kernel_size];
            let out_row_start = ch * out_len;
            for pos in 0..out_len {
                let sig_slice = &signal[pos..pos + kernel_size];
                out[out_row_start + pos] =
                    crate::dispatch::dispatch_dot_f64(sig_slice, filter_slice, ctx) + bias[ch];
            }
        }
    }
}

