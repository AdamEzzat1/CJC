
use cjc_repro::{kahan_sum_f64, KahanAccumulatorF64, Rng};

use crate::accumulator;
use crate::buffer::Buffer;
use crate::dispatch;
use crate::error::RuntimeError;
use crate::kernel as kernel_fns;
use crate::tensor_tiled::TiledMatmul;

// ---------------------------------------------------------------------------
// 2. Tensor Runtime
// ---------------------------------------------------------------------------

/// An N-dimensional tensor backed by a `Buffer<f64>`.
///
/// Supports element-wise arithmetic, matrix multiplication (2-D), and
/// numerically-stable reductions via Kahan summation.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub buffer: Buffer<f64>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

impl Tensor {
    // -- Construction -------------------------------------------------------

    /// Compute row-major strides for a given shape.
    pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Total number of elements implied by `shape`.
    fn shape_numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel = Self::shape_numel(shape);
        Tensor {
            buffer: Buffer::alloc(numel, 0.0),
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
            offset: 0,
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let numel = Self::shape_numel(shape);
        Tensor {
            buffer: Buffer::alloc(numel, 1.0),
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
            offset: 0,
        }
    }

    /// Create a tensor filled with samples from the standard normal
    /// distribution, drawn deterministically from `rng`.
    pub fn randn(shape: &[usize], rng: &mut Rng) -> Self {
        let numel = Self::shape_numel(shape);
        let data: Vec<f64> = (0..numel).map(|_| rng.next_normal_f64()).collect();
        Tensor {
            buffer: Buffer::from_vec(data),
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
            offset: 0,
        }
    }

    /// Create a tensor from raw data and a shape. Returns an error if the
    /// number of elements does not match the shape.
    pub fn from_vec(data: Vec<f64>, shape: &[usize]) -> Result<Self, RuntimeError> {
        let numel = Self::shape_numel(shape);
        if data.len() != numel {
            return Err(RuntimeError::ShapeMismatch {
                expected: numel,
                got: data.len(),
            });
        }
        Ok(Tensor {
            buffer: Buffer::from_vec(data),
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
            offset: 0,
        })
    }

    // -- Accessors ----------------------------------------------------------

    /// The shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        Self::shape_numel(&self.shape)
    }

    /// Whether the tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flatten a multi-dimensional index to a linear offset in the buffer.
    fn linear_index(&self, indices: &[usize]) -> Result<usize, RuntimeError> {
        if indices.len() != self.shape.len() {
            return Err(RuntimeError::DimensionMismatch {
                expected: self.shape.len(),
                got: indices.len(),
            });
        }
        let mut off = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(RuntimeError::IndexOutOfBounds {
                    index: idx,
                    length: self.shape[i],
                });
            }
            off += idx * self.strides[i];
        }
        Ok(off)
    }

    /// Whether this tensor is contiguous in memory (row-major, no offset).
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let expected = Self::compute_strides(&self.shape);
        self.strides == expected
    }

    /// Create a zero-copy slice (view) of this tensor.
    /// `ranges` contains `(start, end)` for each dimension.
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<Tensor, RuntimeError> {
        if ranges.len() != self.shape.len() {
            return Err(RuntimeError::DimensionMismatch {
                expected: self.shape.len(),
                got: ranges.len(),
            });
        }
        let mut new_offset = self.offset;
        let mut new_shape = Vec::with_capacity(ranges.len());
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if end > self.shape[i] || start > end {
                return Err(RuntimeError::IndexOutOfBounds {
                    index: end,
                    length: self.shape[i],
                });
            }
            new_offset += start * self.strides[i];
            new_shape.push(end - start);
        }
        Ok(Tensor {
            buffer: self.buffer.clone(), // shared — zero copy
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
        })
    }

    /// Materialize a contiguous copy if this tensor is non-contiguous.
    pub fn to_contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        let data = self.to_vec();
        Tensor {
            buffer: Buffer::from_vec(data),
            shape: self.shape.clone(),
            strides: Self::compute_strides(&self.shape),
            offset: 0,
        }
    }

    /// Create a broadcast view of this tensor to `target_shape`.
    /// Uses stride=0 for dimensions that need broadcasting (size 1 -> target size).
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Tensor, RuntimeError> {
        let src_ndim = self.shape.len();
        let tgt_ndim = target_shape.len();
        if tgt_ndim < src_ndim {
            return Err(RuntimeError::InvalidOperation(
                "cannot broadcast to a smaller rank".to_string(),
            ));
        }
        let pad = tgt_ndim - src_ndim;
        let mut new_strides = vec![0usize; tgt_ndim];
        for i in 0..tgt_ndim {
            if i < pad {
                // Padded dimension: stride = 0 (broadcast)
                new_strides[i] = 0;
            } else {
                let src_i = i - pad;
                if self.shape[src_i] == target_shape[i] {
                    new_strides[i] = self.strides[src_i];
                } else if self.shape[src_i] == 1 {
                    new_strides[i] = 0; // broadcast
                } else {
                    return Err(RuntimeError::ShapeMismatch {
                        expected: target_shape[i],
                        got: self.shape[src_i],
                    });
                }
            }
        }
        Ok(Tensor {
            buffer: self.buffer.clone(),
            shape: target_shape.to_vec(),
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Read the element at the given multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> Result<f64, RuntimeError> {
        let offset = self.linear_index(indices)?;
        self.buffer
            .get(offset)
            .ok_or(RuntimeError::IndexOutOfBounds {
                index: offset,
                length: self.buffer.len(),
            })
    }

    /// Write the element at the given multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], val: f64) -> Result<(), RuntimeError> {
        let offset = self.linear_index(indices)?;
        self.buffer.set(offset, val)
    }

    /// Extract the raw data as a `Vec<f64>`, respecting strides and offset.
    pub fn to_vec(&self) -> Vec<f64> {
        if self.is_contiguous() {
            let full = self.buffer.as_slice();
            let numel = self.len();
            if full.len() == numel {
                return full;
            }
            // Buffer may be larger than the tensor's logical size
            // (e.g. Scratchpad pre-allocates extra capacity)
            return full[..numel].to_vec();
        }
        // Non-contiguous: iterate via strided access
        let numel = self.len();
        let mut result = Vec::with_capacity(numel);
        let ndim = self.shape.len();
        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            let mut off = self.offset;
            for d in 0..ndim {
                off += indices[d] * self.strides[d];
            }
            result.push(self.buffer.get(off).unwrap_or(0.0));
            // Increment multi-index (row-major order)
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < self.shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        result
    }

    // -- Reshape ------------------------------------------------------------

    /// Reshape to `new_shape`. The new shape must have the same total number
    /// of elements. The returned tensor **shares** the underlying buffer.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor, RuntimeError> {
        let new_numel = Self::shape_numel(new_shape);
        if new_numel != self.len() {
            return Err(RuntimeError::ShapeMismatch {
                expected: self.len(),
                got: new_numel,
            });
        }
        // Reshape requires contiguous data; materialize if needed
        let tensor = if self.is_contiguous() { self.clone() } else { self.to_contiguous() };
        Ok(Tensor {
            buffer: tensor.buffer,
            shape: new_shape.to_vec(),
            strides: Self::compute_strides(new_shape),
            offset: 0,
        })
    }

    // -- Element-wise operations --------------------------------------------

    /// Apply a binary operation element-wise with broadcasting support.
    fn elementwise_binop(
        &self,
        other: &Tensor,
        op: impl Fn(f64, f64) -> f64,
    ) -> Result<Tensor, RuntimeError> {
        if self.shape == other.shape && self.is_contiguous() && other.is_contiguous() {
            // Fast path: same shape, both contiguous
            let a = self.buffer.as_slice();
            let b = other.buffer.as_slice();
            let data: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect();
            return Ok(Tensor {
                buffer: Buffer::from_vec(data),
                shape: self.shape.clone(),
                strides: Self::compute_strides(&self.shape),
                offset: 0,
            });
        }

        // Broadcasting path: compute result shape
        let result_shape = Self::broadcast_result_shape(&self.shape, &other.shape)?;
        let a_broadcast = self.broadcast_to(&result_shape)?;
        let b_broadcast = other.broadcast_to(&result_shape)?;

        let numel = Self::shape_numel(&result_shape);
        let ndim = result_shape.len();
        let mut data = Vec::with_capacity(numel);
        let mut indices = vec![0usize; ndim];

        for _ in 0..numel {
            let mut off_a = a_broadcast.offset;
            let mut off_b = b_broadcast.offset;
            for d in 0..ndim {
                off_a += indices[d] * a_broadcast.strides[d];
                off_b += indices[d] * b_broadcast.strides[d];
            }
            let va = a_broadcast.buffer.get(off_a).unwrap_or(0.0);
            let vb = b_broadcast.buffer.get(off_b).unwrap_or(0.0);
            data.push(op(va, vb));

            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < result_shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        Ok(Tensor {
            buffer: Buffer::from_vec(data),
            shape: result_shape.clone(),
            strides: Self::compute_strides(&result_shape),
            offset: 0,
        })
    }

    /// Compute the broadcast result shape for two shapes (NumPy rules).
    fn broadcast_result_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, RuntimeError> {
        let max_ndim = a.len().max(b.len());
        let mut result = Vec::with_capacity(max_ndim);
        for i in 0..max_ndim {
            let da = if i < max_ndim - a.len() { 1 } else { a[i - (max_ndim - a.len())] };
            let db = if i < max_ndim - b.len() { 1 } else { b[i - (max_ndim - b.len())] };
            if da == db {
                result.push(da);
            } else if da == 1 {
                result.push(db);
            } else if db == 1 {
                result.push(da);
            } else {
                return Err(RuntimeError::ShapeMismatch {
                    expected: da,
                    got: db,
                });
            }
        }
        Ok(result)
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise_binop(other, |a, b| a + b)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise_binop(other, |a, b| a - b)
    }

    /// Element-wise (Hadamard) multiplication.
    pub fn mul_elem(&self, other: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise_binop(other, |a, b| a * b)
    }

    /// Element-wise division.
    pub fn div_elem(&self, other: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise_binop(other, |a, b| a / b)
    }

    /// Apply a unary function to every element, returning a new contiguous tensor.
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        let data: Vec<f64> = self.to_vec().iter().map(|&x| f(x)).collect();
        Tensor {
            buffer: Buffer::from_vec(data),
            shape: self.shape.clone(),
            strides: Self::compute_strides(&self.shape),
            offset: 0,
        }
    }

    // -- Reductions (using Kahan summation) ---------------------------------

    /// Sum of all elements (Kahan-compensated).
    pub fn sum(&self) -> f64 {
        kahan_sum_f64(&self.buffer.as_slice())
    }

    /// Sum of all elements using BinnedAccumulator (order-invariant, deterministic).
    ///
    /// Bit-identical results regardless of element ordering or reduction schedule.
    pub fn binned_sum(&self) -> f64 {
        accumulator::binned_sum_f64(&self.buffer.as_slice())
    }

    /// Sum with dispatched strategy based on execution context.
    ///
    /// Uses Kahan in serial mode, Binned in parallel/@nogc/strict/linalg mode.
    pub fn dispatched_sum(&self, ctx: &dispatch::ReductionContext) -> f64 {
        dispatch::dispatch_sum_f64(&self.buffer.as_slice(), ctx)
    }

    /// Mean of all elements (Kahan-compensated sum / count).
    pub fn mean(&self) -> f64 {
        let n = self.len();
        if n == 0 {
            return 0.0;
        }
        self.sum() / n as f64
    }

    /// Mean with dispatched strategy based on execution context.
    pub fn dispatched_mean(&self, ctx: &dispatch::ReductionContext) -> f64 {
        let n = self.len();
        if n == 0 {
            return 0.0;
        }
        self.dispatched_sum(ctx) / n as f64
    }

    /// Sum along a specific axis, returning a tensor with that dimension reduced.
    /// For a 2-D tensor with axis=0, sums each column (result shape [1, N]).
    /// For axis=1, sums each row (result shape [M, 1]).
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "sum_axis currently requires a 2-D tensor".to_string(),
            ));
        }
        if axis >= 2 {
            return Err(RuntimeError::IndexOutOfBounds {
                index: axis,
                length: 2,
            });
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let data = self.to_vec();

        if axis == 0 {
            // Sum columns: result shape [1, N]
            let mut result = vec![0.0f64; n];
            for j in 0..n {
                let col: Vec<f64> = (0..m).map(|i| data[i * n + j]).collect();
                result[j] = kahan_sum_f64(&col);
            }
            Tensor::from_vec(result, &[1, n])
        } else {
            // Sum rows: result shape [M, 1]
            // But for bias gradient we actually want shape [1, N] from axis=0
            // For convenience, sum_axis(1) returns [M, 1]
            let mut result = vec![0.0f64; m];
            for i in 0..m {
                let row: Vec<f64> = (0..n).map(|j| data[i * n + j]).collect();
                result[i] = kahan_sum_f64(&row);
            }
            Tensor::from_vec(result, &[m, 1])
        }
    }

    // -- Matrix multiplication (2-D only) -----------------------------------

    /// Negate every element, returning a new tensor.
    pub fn neg(&self) -> Tensor {
        self.map(|x| -x)
    }

    /// Transpose a 2-D tensor as a zero-copy view (swap strides and shape).
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "transpose requires a 2-D tensor");
        let m = self.shape[0];
        let n = self.shape[1];
        Tensor {
            buffer: self.buffer.clone(), // shared — zero copy
            shape: vec![n, m],
            strides: vec![self.strides[1], self.strides[0]],
            offset: self.offset,
        }
    }

    /// Multiply every element by a scalar, returning a new tensor.
    pub fn scalar_mul(&self, s: f64) -> Tensor {
        self.map(|x| x * s)
    }

    // ── Panicking convenience constructors (used by AD engine) --------

    /// Create a tensor from raw data and shape.
    /// **Panics** if `data.len()` does not match the shape.
    pub fn from_vec_unchecked(data: Vec<f64>, shape: &[usize]) -> Tensor {
        Self::from_vec(data, shape).expect("Tensor::from_vec_unchecked: shape mismatch")
    }

    /// Element-wise addition. **Panics** on shape mismatch.
    pub fn add_unchecked(&self, other: &Tensor) -> Tensor {
        self.add(other).expect("Tensor::add shape mismatch")
    }

    /// Element-wise subtraction. **Panics** on shape mismatch.
    pub fn sub_unchecked(&self, other: &Tensor) -> Tensor {
        self.sub(other).expect("Tensor::sub shape mismatch")
    }

    /// Element-wise multiplication. **Panics** on shape mismatch.
    pub fn mul_elem_unchecked(&self, other: &Tensor) -> Tensor {
        self.mul_elem(other).expect("Tensor::mul_elem shape mismatch")
    }

    /// Element-wise division. **Panics** on shape mismatch.
    pub fn div_elem_unchecked(&self, other: &Tensor) -> Tensor {
        self.div_elem(other).expect("Tensor::div_elem shape mismatch")
    }

    /// Matrix multiplication. **Panics** on dimension mismatch.
    pub fn matmul_unchecked(&self, other: &Tensor) -> Tensor {
        self.matmul(other).expect("Tensor::matmul dimension mismatch")
    }

    /// Matrix multiplication for 2-D tensors.
    ///
    /// `self` is (M, K), `other` is (K, N) => result is (M, N).
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "matmul requires 2-D tensors".to_string(),
            ));
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];
        if k != k2 {
            return Err(RuntimeError::DimensionMismatch {
                expected: k,
                got: k2,
            });
        }

        let a = self.to_vec();
        let b = other.to_vec();

        // Parallel path (Mode A): parallelize over output rows when the parallel
        // feature is enabled and the matrix is large enough (>= 256 in any dim).
        #[cfg(feature = "parallel")]
        {
            if m >= 256 || n >= 256 || k >= 256 {
                return Self::matmul_parallel_mode_a(&a, &b, m, n, k);
            }
        }

        // Tiled path: use L2-friendly tiled matmul for medium-to-large matrices.
        // Threshold: any dimension >= 64 (the default tile size).
        // NOTE: tiled path uses naive accumulation (not Kahan) — different
        // numerical path for large matrices, but better cache locality.
        if m >= 64 || n >= 64 || k >= 64 {
            return Self::matmul_tiled(&a, &b, m, n, k);
        }

        // Sequential path: single-threaded with Kahan summation.
        Self::matmul_sequential(&a, &b, m, n, k)
    }

    /// Sequential matmul (always available, deterministic reference).
    fn matmul_sequential(
        a: &[f64], b: &[f64], m: usize, n: usize, k: usize,
    ) -> Result<Tensor, RuntimeError> {
        let mut result = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = KahanAccumulatorF64::new();
                for p in 0..k {
                    acc.add(a[i * k + p] * b[p * n + j]);
                }
                result[i * n + j] = acc.finalize();
            }
        }
        Tensor::from_vec(result, &[m, n])
    }

    /// Tiled matmul: delegates to `TiledMatmul` for L2-cache-friendly tiling.
    ///
    /// Used for medium matrices (any dimension >= 64) where cache locality
    /// matters but parallel overhead isn't justified. The tiled path uses
    /// naive accumulation (not Kahan summation), trading a small amount of
    /// floating-point precision for better cache behavior.
    fn matmul_tiled(
        a: &[f64], b: &[f64], m: usize, n: usize, k: usize,
    ) -> Result<Tensor, RuntimeError> {
        let engine = TiledMatmul::new();
        let result = engine.matmul(a, m, k, b, n);
        Tensor::from_vec(result, &[m, n])
    }

    /// Parallel matmul Mode A: parallelize over output rows, sequential k-reduction.
    ///
    /// Deterministic because:
    /// - Each output element C[i,j] is computed by exactly one thread.
    /// - The k-reduction within each element uses sequential Kahan summation
    ///   in the same fixed order (p = 0..k-1).
    /// - No cross-thread reduction or merge of partial sums.
    ///
    /// This is the mandatory baseline for reproducibility mode.
    #[cfg(feature = "parallel")]
    fn matmul_parallel_mode_a(
        a: &[f64], b: &[f64], m: usize, n: usize, k: usize,
    ) -> Result<Tensor, RuntimeError> {
        use rayon::prelude::*;

        let mut result = vec![0.0f64; m * n];

        // Parallelize over rows: each thread computes one or more full rows.
        result
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                for j in 0..n {
                    let mut acc = KahanAccumulatorF64::new();
                    for p in 0..k {
                        acc.add(a[i * k + p] * b[p * n + j]);
                    }
                    row[j] = acc.finalize();
                }
            });

        Tensor::from_vec(result, &[m, n])
    }

    // -- Transformer Kernels ------------------------------------------------

    /// Batched matrix multiplication.
    ///
    /// `self` is `[..., M, K]`, `other` is `[..., K, N]` => result is `[..., M, N]`.
    /// The batch dimensions must be identical (no broadcast).
    /// For 2-D inputs, delegates to `matmul`.
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(RuntimeError::InvalidOperation(
                "bmm requires at least 2-D tensors".to_string(),
            ));
        }
        if self.ndim() == 2 && other.ndim() == 2 {
            return self.matmul(other);
        }
        if self.ndim() != other.ndim() {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "bmm requires same number of dimensions, got {} and {}",
                    self.ndim(),
                    other.ndim()
                ),
            ));
        }
        let nd = self.ndim();
        let batch_dims_a = &self.shape[..nd - 2];
        let batch_dims_b = &other.shape[..nd - 2];
        if batch_dims_a != batch_dims_b {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "bmm batch dimensions mismatch: {:?} vs {:?}",
                    batch_dims_a, batch_dims_b
                ),
            ));
        }
        let m = self.shape[nd - 2];
        let k = self.shape[nd - 1];
        let k2 = other.shape[nd - 2];
        let n = other.shape[nd - 1];
        if k != k2 {
            return Err(RuntimeError::DimensionMismatch {
                expected: k,
                got: k2,
            });
        }

        let batch_size: usize = batch_dims_a.iter().product();
        let a = self.to_vec();
        let b = other.to_vec();
        let mat_a_stride = m * k;
        let mat_b_stride = k * n;
        let mat_c_stride = m * n;
        let mut result = vec![0.0f64; batch_size * mat_c_stride];

        for batch in 0..batch_size {
            let a_off = batch * mat_a_stride;
            let b_off = batch * mat_b_stride;
            let c_off = batch * mat_c_stride;
            for i in 0..m {
                for j in 0..n {
                    // In-place accumulation — zero heap allocation per dot product.
                    let mut acc = KahanAccumulatorF64::new();
                    for p in 0..k {
                        acc.add(a[a_off + i * k + p] * b[b_off + p * n + j]);
                    }
                    result[c_off + i * n + j] = acc.finalize();
                }
            }
        }

        let mut out_shape = batch_dims_a.to_vec();
        out_shape.push(m);
        out_shape.push(n);
        Tensor::from_vec(result, &out_shape)
    }

    /// Softmax along the last dimension (two-pass stable algorithm).
    ///
    /// Pass 1: find max per row (prevents overflow in exp)
    /// Pass 2: compute exp(x - max), accumulate sum, normalize
    ///
    /// For a tensor of shape `[..., N]`, softmax is applied independently
    /// to each length-N slice along the last axis.
    pub fn softmax(&self) -> Result<Tensor, RuntimeError> {
        if self.ndim() == 0 {
            return Err(RuntimeError::InvalidOperation(
                "softmax requires at least 1-D tensor".to_string(),
            ));
        }
        let data = self.to_vec();
        let n = *self.shape.last().unwrap(); // last dimension size
        let outer: usize = data.len() / n;  // product of all dims except last
        let mut result = vec![0.0f64; data.len()];

        for row in 0..outer {
            let start = row * n;
            let end = start + n;
            let slice = &data[start..end];

            // Pass 1: find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for &v in slice {
                if v > max_val {
                    max_val = v;
                }
            }

            // Pass 2: exp(x - max) and accumulate sum
            let mut exp_vals = vec![0.0f64; n];
            let mut sum = 0.0f64;
            let mut comp = 0.0f64; // Kahan compensation
            for i in 0..n {
                let e = (slice[i] - max_val).exp();
                exp_vals[i] = e;
                // Kahan summation for the denominator
                let y = e - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }

            // Normalize
            if sum == 0.0 {
                // Degenerate case: all -inf inputs → uniform
                let uniform = 1.0 / n as f64;
                for i in 0..n {
                    result[start + i] = uniform;
                }
            } else {
                for i in 0..n {
                    result[start + i] = exp_vals[i] / sum;
                }
            }
        }

        Tensor::from_vec(result, &self.shape)
    }

    /// Layer normalization over the last dimension.
    ///
    /// For each length-D slice along the last axis:
    ///   1. mean = Σx / D  (Kahan)
    ///   2. var  = Σ(x - mean)² / D  (Kahan)
    ///   3. normalized = (x - mean) / √(var + eps)
    ///   4. output = gamma * normalized + beta
    ///
    /// `gamma` and `beta` are 1-D tensors of shape `[D]`.
    /// `eps` is a small constant (typically 1e-5).
    pub fn layer_norm(
        &self,
        gamma: &Tensor,
        beta: &Tensor,
        eps: f64,
    ) -> Result<Tensor, RuntimeError> {
        if self.ndim() == 0 {
            return Err(RuntimeError::InvalidOperation(
                "layer_norm requires at least 1-D tensor".to_string(),
            ));
        }
        let d = *self.shape.last().unwrap();
        if gamma.len() != d || beta.len() != d {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "layer_norm: gamma/beta length {} must match last dim {}",
                    gamma.len(),
                    d
                ),
            ));
        }

        let data = self.to_vec();
        let gamma_data = gamma.to_vec();
        let beta_data = beta.to_vec();
        let outer = data.len() / d;
        let mut result = vec![0.0f64; data.len()];

        for row in 0..outer {
            let start = row * d;
            let slice = &data[start..start + d];

            // Pass 1: compute mean via Kahan
            let mean = kahan_sum_f64(slice) / d as f64;

            // Pass 2: compute variance via Kahan
            let diffs: Vec<f64> = slice.iter().map(|&x| {
                let diff = x - mean;
                diff * diff
            }).collect();
            let variance = kahan_sum_f64(&diffs) / d as f64;

            // Normalize, scale, shift
            let inv_std = 1.0 / (variance + eps).sqrt();
            for i in 0..d {
                let normalized = (slice[i] - mean) * inv_std;
                result[start + i] = gamma_data[i] * normalized + beta_data[i];
            }
        }

        Tensor::from_vec(result, &self.shape)
    }

    /// ReLU activation: max(0, x) element-wise.
    pub fn relu(&self) -> Tensor {
        let data = self.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        Tensor::from_vec(result, &self.shape).unwrap()
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)) element-wise.
    pub fn sigmoid(&self) -> Tensor {
        let data = self.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::from_vec(result, &self.shape).unwrap()
    }

    /// Tanh activation element-wise.
    pub fn tanh_activation(&self) -> Tensor {
        let data = self.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.tanh()).collect();
        Tensor::from_vec(result, &self.shape).unwrap()
    }

    /// Leaky ReLU activation: max(alpha*x, x) element-wise.
    pub fn leaky_relu(&self, alpha: f64) -> Tensor {
        let data = self.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| if x > 0.0 { x } else { alpha * x }).collect();
        Tensor::from_vec(result, &self.shape).unwrap()
    }

    /// SiLU (Swish) activation: x * sigmoid(x) element-wise.
    pub fn silu(&self) -> Tensor {
        let data = self.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        Tensor::from_vec(result, &self.shape).unwrap()
    }

    /// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    pub fn mish(&self) -> Tensor {
        let data = self.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| {
            let sp = (1.0 + x.exp()).ln();
            x * sp.tanh()
        }).collect();
        Tensor::from_vec(result, &self.shape).unwrap()
    }

    /// Argmax: index of the maximum element (first occurrence, deterministic).
    pub fn argmax(&self) -> usize {
        let data = self.to_vec();
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for (i, &v) in data.iter().enumerate() {
            if v > best_val || (v == best_val && i < best_idx) {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Argmin: index of the minimum element (first occurrence, deterministic).
    pub fn argmin(&self) -> usize {
        let data = self.to_vec();
        let mut best_idx = 0;
        let mut best_val = f64::INFINITY;
        for (i, &v) in data.iter().enumerate() {
            if v < best_val || (v == best_val && i < best_idx) {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Clamp all elements to [min, max].
    pub fn clamp(&self, min: f64, max: f64) -> Tensor {
        let data = self.to_vec();
        let result: Vec<f64> = data.iter().map(|&x| x.max(min).min(max)).collect();
        Tensor::from_vec(result, &self.shape).unwrap()
    }

    /// One-hot encoding: given a 1D tensor of integer indices and a depth,
    /// returns a 2D tensor of shape [len, depth].
    pub fn one_hot(indices: &[usize], depth: usize) -> Result<Tensor, RuntimeError> {
        let n = indices.len();
        let mut data = vec![0.0; n * depth];
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= depth {
                return Err(RuntimeError::InvalidOperation(format!(
                    "one_hot: index {idx} >= depth {depth}"
                )));
            }
            data[i * depth + idx] = 1.0;
        }
        Tensor::from_vec(data, &[n, depth])
    }

    /// GELU activation (approximate): x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    pub fn gelu(&self) -> Tensor {
        let data = self.to_vec();
        let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
        let result: Vec<f64> = data.iter().map(|&x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }).collect();
        Tensor::from_vec(result, &self.shape).unwrap()
    }

    /// Linear layer: output = input @ weight^T + bias
    ///
    /// `self` is `[..., in_features]`, `weight` is `[out_features, in_features]`,
    /// `bias` is `[out_features]`.
    /// Result is `[..., out_features]`.
    pub fn linear(
        &self,
        weight: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor, RuntimeError> {
        if weight.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "linear: weight must be 2-D [out_features, in_features]".to_string(),
            ));
        }
        let out_features = weight.shape[0];
        let in_features = weight.shape[1];
        let last_dim = *self.shape.last().ok_or_else(|| {
            RuntimeError::InvalidOperation("linear: input must be at least 1-D".to_string())
        })?;
        if last_dim != in_features {
            return Err(RuntimeError::DimensionMismatch {
                expected: in_features,
                got: last_dim,
            });
        }
        if bias.len() != out_features {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "linear: bias length {} must match out_features {}",
                    bias.len(),
                    out_features
                ),
            ));
        }

        let data = self.to_vec();
        let w = weight.to_vec();
        let b = bias.to_vec();
        let outer = data.len() / in_features;
        let mut result = vec![0.0f64; outer * out_features];

        for row in 0..outer {
            let x_start = row * in_features;
            let x_slice = &data[x_start..x_start + in_features];
            let y_start = row * out_features;
            for j in 0..out_features {
                let w_start = j * in_features;
                let mut acc = KahanAccumulatorF64::new();
                for p in 0..in_features {
                    acc.add(x_slice[p] * w[w_start + p]);
                }
                result[y_start + j] = acc.finalize() + b[j];
            }
        }

        let mut out_shape = self.shape[..self.shape.len() - 1].to_vec();
        out_shape.push(out_features);
        Tensor::from_vec(result, &out_shape)
    }

    /// 1D convolution: signal `[signal_len]` * filters `[out_ch, kernel_size]` + bias
    ///
    /// Returns `[out_ch, signal_len - kernel_size + 1]` (valid mode, stride=1).
    pub fn conv1d(
        &self,
        filters: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 1 {
            return Err(RuntimeError::InvalidOperation(
                "conv1d: input must be 1-D [signal_len]".to_string(),
            ));
        }
        if filters.ndim() != 2 {
            return Err(RuntimeError::InvalidOperation(
                "conv1d: filters must be 2-D [out_channels, kernel_size]".to_string(),
            ));
        }
        let signal_len = self.shape[0];
        let out_channels = filters.shape[0];
        let kernel_size = filters.shape[1];
        if signal_len < kernel_size {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "conv1d: signal_len {} < kernel_size {}",
                    signal_len, kernel_size
                ),
            ));
        }
        if bias.len() != out_channels {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "conv1d: bias length {} must match out_channels {}",
                    bias.len(), out_channels
                ),
            ));
        }
        let out_len = signal_len - kernel_size + 1;
        let s = self.to_vec();
        let f = filters.to_vec();
        let b = bias.to_vec();
        let mut result = vec![0.0; out_channels * out_len];
        kernel_fns::conv1d_raw(&s, &f, &b, &mut result, signal_len, out_channels, kernel_size);
        Tensor::from_vec(result, &[out_channels, out_len])
    }

    /// 2D convolution — NCHW layout, valid mode, configurable stride.
    ///
    /// # Arguments
    /// - `self`:    `[N, C_in, H, W]` input tensor
    /// - `filters`: `[C_out, C_in, kH, kW]`
    /// - `bias`:    `[C_out]`
    /// - `stride`:  spatial stride (default 1)
    ///
    /// # Returns
    /// `[N, C_out, H_out, W_out]` where `H_out = (H - kH) / stride + 1`.
    ///
    /// Uses `BinnedAccumulatorF64` for every dot product — bit-identical results
    /// across all runs and hardware configurations.
    pub fn conv2d(
        &self,
        filters: &Tensor,
        bias: &Tensor,
        stride: usize,
    ) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 4 {
            return Err(RuntimeError::InvalidOperation(
                "conv2d: input must be 4-D [N, C_in, H, W]".to_string(),
            ));
        }
        if filters.ndim() != 4 {
            return Err(RuntimeError::InvalidOperation(
                "conv2d: filters must be 4-D [C_out, C_in, kH, kW]".to_string(),
            ));
        }
        if stride == 0 {
            return Err(RuntimeError::InvalidOperation(
                "conv2d: stride must be >= 1".to_string(),
            ));
        }

        let n    = self.shape[0];
        let c_in = self.shape[1];
        let h_in = self.shape[2];
        let w_in = self.shape[3];

        let c_out      = filters.shape[0];
        let c_in_check = filters.shape[1];
        let kh         = filters.shape[2];
        let kw         = filters.shape[3];

        if c_in != c_in_check {
            return Err(RuntimeError::InvalidOperation(format!(
                "conv2d: input C_in={} does not match filter C_in={}",
                c_in, c_in_check
            )));
        }
        if h_in < kh || w_in < kw {
            return Err(RuntimeError::InvalidOperation(format!(
                "conv2d: input spatial [{}, {}] is smaller than kernel [{}, {}]",
                h_in, w_in, kh, kw
            )));
        }
        if bias.len() != c_out {
            return Err(RuntimeError::InvalidOperation(format!(
                "conv2d: bias length {} must match C_out={}",
                bias.len(), c_out
            )));
        }

        let h_out = (h_in - kh) / stride + 1;
        let w_out = (w_in - kw) / stride + 1;

        let inp = self.to_vec();
        let flt = filters.to_vec();
        let b   = bias.to_vec();
        let mut result = vec![0.0f64; n * c_out * h_out * w_out];

        kernel_fns::conv2d_raw(&inp, &flt, &b, &mut result,
                           n, c_in, h_in, w_in, c_out, kh, kw, stride);

        Tensor::from_vec(result, &[n, c_out, h_out, w_out])
    }

    /// 2D max-pooling — NCHW layout, non-overlapping windows.
    ///
    /// - `self`: `[N, C, H, W]`
    /// - `ph`, `pw`: pool height/width (stride = window size)
    ///
    /// Returns `[N, C, H/ph, W/pw]`.
    pub fn maxpool2d(&self, ph: usize, pw: usize) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 4 {
            return Err(RuntimeError::InvalidOperation(
                "maxpool2d: input must be 4-D [N, C, H, W]".to_string(),
            ));
        }
        if ph == 0 || pw == 0 {
            return Err(RuntimeError::InvalidOperation(
                "maxpool2d: pool size must be >= 1".to_string(),
            ));
        }

        let n    = self.shape[0];
        let c    = self.shape[1];
        let h_in = self.shape[2];
        let w_in = self.shape[3];

        if h_in < ph || w_in < pw {
            return Err(RuntimeError::InvalidOperation(format!(
                "maxpool2d: input [{}, {}] smaller than pool [{}, {}]",
                h_in, w_in, ph, pw
            )));
        }

        let h_out = h_in / ph;
        let w_out = w_in / pw;

        let inp = self.to_vec();
        let mut result = vec![0.0f64; n * c * h_out * w_out];

        kernel_fns::maxpool2d_raw(&inp, &mut result, n, c, h_in, w_in, ph, pw);

        Tensor::from_vec(result, &[n, c, h_out, w_out])
    }

    /// Scaled dot-product attention (single head).
    ///
    /// `queries` is `[..., T, d_k]`
    /// `keys`    is `[..., S, d_k]`
    /// `values`  is `[..., S, d_v]`
    ///
    /// Computes: softmax(Q × Kᵀ / √d_k) × V
    /// Returns `[..., T, d_v]`.
    pub fn scaled_dot_product_attention(
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<Tensor, RuntimeError> {
        if queries.ndim() < 2 || keys.ndim() < 2 || values.ndim() < 2 {
            return Err(RuntimeError::InvalidOperation(
                "attention: Q, K, V must be at least 2-D".to_string(),
            ));
        }
        let nd = queries.ndim();
        let d_k = queries.shape[nd - 1];
        let scale = 1.0 / (d_k as f64).sqrt();

        // Transpose keys: swap last two dims
        let keys_t = keys.transpose_last_two()?;

        // Q × K^T → [... T, S]
        let scores = queries.bmm(&keys_t)?;

        // Scale
        let scores_scaled = scores.scalar_mul(scale);

        // Softmax along last dim
        let attn_weights = scores_scaled.softmax()?;

        // Attn × V → [... T, d_v]
        attn_weights.bmm(values)
    }

    /// Transpose the last two dimensions of a tensor.
    ///
    /// `[..., A, B]` → `[..., B, A]`
    pub fn transpose_last_two(&self) -> Result<Tensor, RuntimeError> {
        if self.ndim() < 2 {
            return Err(RuntimeError::InvalidOperation(
                "transpose_last_two requires at least 2-D tensor".to_string(),
            ));
        }
        let nd = self.ndim();
        let rows = self.shape[nd - 2];
        let cols = self.shape[nd - 1];
        let data = self.to_vec();
        let batch_size: usize = self.shape[..nd - 2].iter().product::<usize>().max(1);
        let mat_size = rows * cols;
        let mut result = vec![0.0f64; data.len()];

        for b in 0..batch_size {
            let off = b * mat_size;
            for i in 0..rows {
                for j in 0..cols {
                    result[off + j * rows + i] = data[off + i * cols + j];
                }
            }
        }

        let mut out_shape = self.shape.clone();
        out_shape[nd - 2] = cols;
        out_shape[nd - 1] = rows;
        Tensor::from_vec(result, &out_shape)
    }

    // -- Zero-Copy Weight Mapping -------------------------------------------

    /// Create a tensor view from raw bytes — **zero allocation**.
    ///
    /// Interprets `bytes` as a contiguous block of `f64` (8 bytes each) or
    /// `f32` (4 bytes each, promoted to f64) values and maps them into a
    /// `Tensor` with the given shape.
    ///
    /// `dtype` must be `"f64"` or `"f32"`.
    ///
    /// For f64: bytes.len() must equal shape_numel * 8.
    /// For f32: bytes.len() must equal shape_numel * 4.
    ///
    /// The returned tensor **owns** its buffer (copied from the raw bytes)
    /// but performs exactly one allocation for the data vector.
    pub fn from_bytes(bytes: &[u8], shape: &[usize], dtype: &str) -> Result<Tensor, RuntimeError> {
        let numel = Self::shape_numel(shape);
        match dtype {
            "f64" => {
                let expected = numel * 8;
                if bytes.len() != expected {
                    return Err(RuntimeError::ShapeMismatch {
                        expected,
                        got: bytes.len(),
                    });
                }
                let mut data = Vec::with_capacity(numel);
                for i in 0..numel {
                    let off = i * 8;
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&bytes[off..off + 8]);
                    data.push(f64::from_le_bytes(buf));
                }
                Ok(Tensor {
                    buffer: Buffer::from_vec(data),
                    shape: shape.to_vec(),
                    strides: Self::compute_strides(shape),
                    offset: 0,
                })
            }
            "f32" => {
                let expected = numel * 4;
                if bytes.len() != expected {
                    return Err(RuntimeError::ShapeMismatch {
                        expected,
                        got: bytes.len(),
                    });
                }
                let mut data = Vec::with_capacity(numel);
                for i in 0..numel {
                    let off = i * 4;
                    let mut buf = [0u8; 4];
                    buf.copy_from_slice(&bytes[off..off + 4]);
                    data.push(f32::from_le_bytes(buf) as f64);
                }
                Ok(Tensor {
                    buffer: Buffer::from_vec(data),
                    shape: shape.to_vec(),
                    strides: Self::compute_strides(shape),
                    offset: 0,
                })
            }
            _ => Err(RuntimeError::InvalidOperation(
                format!("from_bytes: unsupported dtype '{}', expected 'f32' or 'f64'", dtype),
            )),
        }
    }

    // -- Multi-Head Attention Splitting -------------------------------------

    /// Reshape a 3D tensor `[batch, seq, model_dim]` into 4D
    /// `[batch, num_heads, seq, head_dim]` by splitting the last dimension.
    ///
    /// This is a **zero-copy view** — it only changes shape/strides metadata.
    /// `model_dim` must be divisible by `num_heads`.
    pub fn split_heads(&self, num_heads: usize) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 3 {
            return Err(RuntimeError::DimensionMismatch {
                expected: 3,
                got: self.ndim(),
            });
        }
        let batch = self.shape[0];
        let seq = self.shape[1];
        let model_dim = self.shape[2];
        if model_dim % num_heads != 0 {
            return Err(RuntimeError::InvalidOperation(
                format!(
                    "split_heads: model_dim {} not divisible by num_heads {}",
                    model_dim, num_heads
                ),
            ));
        }
        let head_dim = model_dim / num_heads;
        // Need contiguous data for the reshape
        let tensor = if self.is_contiguous() { self.clone() } else { self.to_contiguous() };
        // Reshape [B, S, H*D] -> [B, S, H, D] then transpose to [B, H, S, D]
        let reshaped = Tensor {
            buffer: tensor.buffer.clone(),
            shape: vec![batch, seq, num_heads, head_dim],
            strides: Self::compute_strides(&[batch, seq, num_heads, head_dim]),
            offset: 0,
        };
        // Transpose dims 1 and 2: [B, S, H, D] -> [B, H, S, D]
        // New strides: swap strides[1] and strides[2]
        Ok(Tensor {
            buffer: reshaped.buffer,
            shape: vec![batch, num_heads, seq, head_dim],
            strides: vec![
                reshaped.strides[0], // batch stride unchanged
                reshaped.strides[2], // head stride (was dim 2)
                reshaped.strides[1], // seq stride (was dim 1)
                reshaped.strides[3], // head_dim stride unchanged
            ],
            offset: 0,
        })
    }

    /// Merge heads back: reshape 4D `[batch, num_heads, seq, head_dim]` into
    /// 3D `[batch, seq, model_dim]`. Materializes if non-contiguous.
    pub fn merge_heads(&self) -> Result<Tensor, RuntimeError> {
        if self.ndim() != 4 {
            return Err(RuntimeError::DimensionMismatch {
                expected: 4,
                got: self.ndim(),
            });
        }
        let batch = self.shape[0];
        let num_heads = self.shape[1];
        let seq = self.shape[2];
        let head_dim = self.shape[3];
        // Need [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        // Transpose dims 1 and 2 first
        let transposed = Tensor {
            buffer: self.buffer.clone(),
            shape: vec![batch, seq, num_heads, head_dim],
            strides: vec![
                self.strides[0],
                self.strides[2], // seq stride
                self.strides[1], // head stride
                self.strides[3],
            ],
            offset: self.offset,
        };
        // Materialize contiguous then reshape
        let contig = transposed.to_contiguous();
        let model_dim = num_heads * head_dim;
        Ok(Tensor {
            buffer: contig.buffer,
            shape: vec![batch, seq, model_dim],
            strides: Self::compute_strides(&[batch, seq, model_dim]),
            offset: 0,
        })
    }

    /// View-only reshape: reinterpret shape without copying.
    /// Only works on contiguous tensors. Falls back to copy if non-contiguous.
    pub fn view_reshape(&self, new_shape: &[usize]) -> Result<Tensor, RuntimeError> {
        self.reshape(new_shape)
    }
}

