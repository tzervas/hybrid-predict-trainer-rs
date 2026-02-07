//! GPU acceleration kernels via `CubeCL` and Burn.
//!
//! This module provides GPU-accelerated implementations of performance-critical
//! operations using `CubeCL` for custom CUDA kernels and Burn for tensor operations.
//!
//! # Accelerated Operations
//!
//! - **State encoding**: Parallel feature extraction from training state
//! - **Prediction**: Batched dynamics model inference
//! - **Residual compression**: GPU-accelerated SVD for low-rank approximation
//! - **Correction**: Parallel residual application
//!
//! # Backend Support
//!
//! - CUDA (primary target)
//! - Future: Metal, Vulkan via `CubeCL` backends
//!
//! # Usage
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::gpu::{GpuAccelerator, CudaBackend};
//!
//! let accelerator = GpuAccelerator::<CudaBackend>::new()?;
//! let encoded = accelerator.encode_state(&state)?;
//! ```

use crate::error::{HybridResult, HybridTrainingError};
use crate::state::TrainingState;

/// Marker trait for GPU backend implementations.
pub trait GpuBackend: Send + Sync {
    /// Returns the backend name.
    fn name() -> &'static str;

    /// Returns whether this backend is available.
    fn is_available() -> bool;

    /// Returns device information.
    fn device_info() -> DeviceInfo;
}

/// Information about the GPU device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name.
    pub name: String,

    /// Total memory in bytes.
    pub total_memory: usize,

    /// Available memory in bytes.
    pub available_memory: usize,

    /// Compute capability (for CUDA).
    pub compute_capability: Option<(u32, u32)>,

    /// Number of streaming multiprocessors.
    pub num_sms: Option<u32>,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            total_memory: 0,
            available_memory: 0,
            compute_capability: None,
            num_sms: None,
        }
    }
}

/// CUDA backend implementation.
pub struct CudaBackend;

impl GpuBackend for CudaBackend {
    fn name() -> &'static str {
        "CUDA"
    }

    fn is_available() -> bool {
        // Check CUDA availability via CubeCL
        // This is a placeholder - actual implementation would use cubecl-cuda
        cfg!(feature = "cuda")
    }

    fn device_info() -> DeviceInfo {
        // Query device info via CUDA driver API
        // Placeholder implementation
        DeviceInfo {
            name: "NVIDIA GPU".to_string(),
            ..Default::default()
        }
    }
}

/// GPU accelerator for hybrid training operations.
pub struct GpuAccelerator<B: GpuBackend> {
    /// Device information.
    device_info: DeviceInfo,

    /// Memory pool for temporary allocations.
    memory_pool: MemoryPool,

    /// Phantom marker for backend type.
    _backend: std::marker::PhantomData<B>,
}

impl<B: GpuBackend> GpuAccelerator<B> {
    /// Creates a new GPU accelerator.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU backend is not available.
    pub fn new() -> HybridResult<Self> {
        if !B::is_available() {
            return Err((
                HybridTrainingError::GpuError {
                    detail: format!("{} backend not available", B::name()),
                },
                None,
            ));
        }

        let device_info = B::device_info();
        let memory_pool = MemoryPool::new(device_info.available_memory / 4);

        Ok(Self {
            device_info,
            memory_pool,
            _backend: std::marker::PhantomData,
        })
    }

    /// Returns device information.
    #[must_use]
    pub fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    /// Encodes training state to GPU tensor.
    ///
    /// Performs parallel feature extraction on the GPU.
    pub fn encode_state(&self, _state: &TrainingState) -> HybridResult<GpuTensor> {
        // Placeholder - actual implementation would:
        // 1. Transfer state data to GPU
        // 2. Launch parallel feature extraction kernel
        // 3. Return encoded tensor
        Ok(GpuTensor {
            data: Vec::new(),
            shape: vec![32],
            device: B::name().to_string(),
        })
    }

    /// Performs batched dynamics prediction on GPU.
    pub fn predict_batch(
        &self,
        _encoded_states: &[GpuTensor],
        _steps: usize,
    ) -> HybridResult<Vec<GpuTensor>> {
        // Placeholder - actual implementation would:
        // 1. Batch encoded states into single tensor
        // 2. Run RSSM forward pass on GPU
        // 3. Return predicted states
        Ok(Vec::new())
    }

    /// Computes low-rank approximation of residuals on GPU.
    pub fn compress_residuals(
        &self,
        _residuals: &GpuTensor,
        rank: usize,
    ) -> HybridResult<CompressedGpuTensor> {
        // Placeholder - actual implementation would:
        // 1. Perform truncated SVD on GPU
        // 2. Return compressed representation
        Ok(CompressedGpuTensor {
            u: GpuTensor::empty(),
            s: GpuTensor::empty(),
            v: GpuTensor::empty(),
            rank,
        })
    }

    /// Applies corrections in parallel on GPU.
    pub fn apply_corrections(
        &self,
        _predictions: &GpuTensor,
        _corrections: &GpuTensor,
    ) -> HybridResult<GpuTensor> {
        // Placeholder - actual implementation would:
        // 1. Element-wise addition on GPU
        // 2. Return corrected predictions
        Ok(GpuTensor::empty())
    }

    /// Synchronizes GPU operations (waits for completion).
    pub fn synchronize(&self) -> HybridResult<()> {
        // Placeholder - actual implementation would call cudaDeviceSynchronize
        Ok(())
    }

    /// Returns current memory usage.
    #[must_use]
    pub fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            allocated: self.memory_pool.allocated(),
            pool_size: self.memory_pool.capacity(),
            peak_usage: self.memory_pool.peak_usage(),
        }
    }
}

/// GPU tensor representation.
#[derive(Debug, Clone)]
pub struct GpuTensor {
    /// Data (may be on host for inspection).
    data: Vec<f32>,

    /// Tensor shape.
    shape: Vec<usize>,

    /// Device identifier.
    #[allow(dead_code)]
    device: String,
}

impl GpuTensor {
    /// Creates an empty GPU tensor.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            shape: Vec::new(),
            device: "cpu".to_string(),
        }
    }

    /// Returns the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Transfers tensor to host memory.
    #[must_use]
    pub fn to_host(&self) -> Vec<f32> {
        self.data.clone()
    }
}

/// Compressed GPU tensor (SVD representation).
#[derive(Debug, Clone)]
pub struct CompressedGpuTensor {
    /// Left singular vectors.
    #[allow(dead_code)]
    u: GpuTensor,

    /// Singular values.
    #[allow(dead_code)]
    s: GpuTensor,

    /// Right singular vectors.
    #[allow(dead_code)]
    v: GpuTensor,

    /// Rank of approximation.
    rank: usize,
}

impl CompressedGpuTensor {
    /// Returns the rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Reconstructs the full tensor.
    pub fn reconstruct(&self) -> HybridResult<GpuTensor> {
        // Placeholder - actual implementation would compute U @ diag(S) @ V^T
        Ok(GpuTensor::empty())
    }
}

/// Memory pool for GPU allocations.
struct MemoryPool {
    capacity: usize,
    allocated: usize,
    peak: usize,
}

impl MemoryPool {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            allocated: 0,
            peak: 0,
        }
    }

    fn allocated(&self) -> usize {
        self.allocated
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn peak_usage(&self) -> usize {
        self.peak
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Currently allocated bytes.
    pub allocated: usize,

    /// Total pool size.
    pub pool_size: usize,

    /// Peak usage.
    pub peak_usage: usize,
}

/// `CubeCL` kernel definitions.
///
/// `CubeCL` kernel implementations.
///
/// These kernels are compiled to CUDA/PTX at runtime using `CubeCL`.
pub mod kernels {

    /// State encoding kernel configuration.
    ///
    /// Configuration for GPU-accelerated feature extraction from `TrainingState`.
    /// The state encoding process is currently CPU-bound with sequential
    /// statistical computations over history buffers.
    ///
    /// # Why GPU Acceleration?
    ///
    /// `TrainingState::compute_features()` computes 64-dimensional feature vector:
    /// - 8 loss features (current, mean, std, min, max, trend, ratio, volatility)
    /// - 8 gradient norm features (same stats)
    /// - 8 learning rate features (same stats)
    /// - 8 momentum features
    /// - Additional derived features (32 total)
    ///
    /// Each feature computation involves:
    /// - Scanning history buffer (up to 1000 entries)
    /// - Computing statistics (mean, std, min, max)
    /// - Computing trends (recent vs older comparison)
    ///
    /// **CPU performance** (for 1000-entry history):
    /// - Sequential scan of history: ~0.01-0.05 ms per feature
    /// - Total for 64 features: ~0.5-3.0 ms
    /// - Bottleneck: Sequential iteration over history
    ///
    /// **GPU performance target**:
    /// - Parallel statistical reductions
    /// - All features computed simultaneously
    /// - **10-20× speedup**: ~0.05-0.15 ms total
    ///
    /// # Algorithm
    ///
    /// The GPU kernel parallelizes feature extraction:
    ///
    /// ```text
    /// // Thread layout: One block per feature, threads cooperate
    /// __global__ void encode_state_kernel(
    ///     float* loss_history,          // [history_len]
    ///     float* grad_norm_history,     // [history_len]
    ///     float* lr_history,            // [history_len]
    ///     int history_len,
    ///     float* features               // Output [64]
    /// ) {
    ///     int feature_idx = blockIdx.x;
    ///     int tid = threadIdx.x;
    ///
    ///     // Shared memory for parallel reduction
    ///     __shared__ float shared_data[256];
    ///
    ///     // Each block computes one feature via parallel reduction
    ///     // e.g., loss_mean: All threads cooperate to sum, then divide
    ///     // e.g., loss_std: All threads cooperate to compute variance
    ///
    ///     float local_sum = 0.0f;
    ///     for (int i = tid; i < history_len; i += blockDim.x) {
    ///         local_sum += loss_history[i];
    ///     }
    ///
    ///     // Tree reduction in shared memory
    ///     shared_data[tid] = local_sum;
    ///     __syncthreads();
    ///
    ///     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    ///         if (tid < stride) {
    ///             shared_data[tid] += shared_data[tid + stride];
    ///         }
    ///         __syncthreads();
    ///     }
    ///
    ///     // Thread 0 writes final result
    ///     if (tid == 0) {
    ///         features[feature_idx] = shared_data[0] / history_len;
    ///     }
    /// }
    /// ```
    ///
    /// # Parallelization Strategy
    ///
    /// 1. **Feature-Level Parallelism**: Each feature computed by one block
    ///    - 64 features → 64 blocks
    ///    - Independent computation, no synchronization needed
    ///
    /// 2. **Reduction Parallelism**: Within each block, threads cooperate
    ///    - 256 threads per block
    ///    - Parallel reduction for sum, min, max, variance
    ///    - Logarithmic reduction: O(log n) instead of O(n)
    ///
    /// 3. **Memory Coalescing**: Sequential access to history buffers
    ///    - Threads access consecutive elements
    ///    - Maximizes memory bandwidth utilization
    ///
    /// # Memory Access Pattern
    ///
    /// **CPU (Sequential)**:
    /// ```text
    /// for feature in features:
    ///     for entry in history:
    ///         accumulate(entry)
    /// ```
    /// Memory accesses: 64 features × 1000 entries = 64,000 reads
    ///
    /// **GPU (Parallel)**:
    /// ```text
    /// parallel_for feature in features:
    ///     parallel_reduce history:
    ///         accumulate(entry)
    /// ```
    /// Same total reads, but:
    /// - All features computed simultaneously (64-way parallelism)
    /// - Each history scan uses 256 threads (256-way parallelism)
    /// - **Effective parallelism: 64 × 256 = 16,384 threads**
    ///
    /// # Integration with HybridTrainer
    ///
    /// State encoding is called once per training step:
    /// - Full phase: Compute features for RSSM training
    /// - Predict phase: Compute features for prediction input
    /// - Correct phase: Compute features for validation
    ///
    /// Speedup impact:
    /// - CPU: 1-3 ms per step → 1000 steps = 1-3 seconds
    /// - GPU: 0.05-0.15 ms per step → 1000 steps = 0.05-0.15 seconds
    /// - **Savings: ~2.85 seconds per 1000 steps**
    ///
    /// For long training runs (100K steps):
    /// - CPU: 100-300 seconds overhead
    /// - GPU: 5-15 seconds overhead
    /// - **Savings: 85-285 seconds (1.4-4.75 minutes)**
    ///
    /// # Features Computed
    ///
    /// **Loss Features (8)**:
    /// 1. Current loss
    /// 2. Loss mean (over history)
    /// 3. Loss std (standard deviation)
    /// 4. Loss min
    /// 5. Loss max
    /// 6. Loss trend (recent - older)
    /// 7. Loss ratio (recent / older)
    /// 8. Loss volatility
    ///
    /// **Gradient Norm Features (8)**: Same stats as loss
    ///
    /// **Learning Rate Features (8)**: Same stats as loss
    ///
    /// **Momentum Features (8)**: If momentum history available
    ///
    /// **Derived Features (32)**:
    /// - Loss oscillation frequency
    /// - Gradient stability
    /// - Learning rate adaptation rate
    /// - Training velocity, acceleration
    /// - Convergence indicators
    /// - Phase transition signals
    #[derive(Debug, Clone)]
    pub struct EncodeStateConfig {
        /// Number of features to extract.
        ///
        /// Why: TrainingState computes 64 features by default.
        /// This can be adjusted for different state representations.
        pub num_features: usize,

        /// History buffer length.
        ///
        /// Why: Determines how many past steps to consider for statistics.
        /// Longer history = more stable statistics but more memory/compute.
        pub history_len: usize,

        /// Block size for parallel reductions.
        ///
        /// Why: Number of threads per feature computation.
        /// 256 is optimal for most GPUs (good occupancy).
        pub block_size: usize,

        /// Enable parallel reduction optimizations.
        ///
        /// Why: Tree reduction provides O(log n) speedup vs sequential.
        /// Can disable for debugging.
        pub use_parallel_reduction: bool,

        /// Use shared memory for intermediate results.
        ///
        /// Why: Reduces global memory traffic for reduction operations.
        pub use_shared_memory: bool,
    }

    impl Default for EncodeStateConfig {
        fn default() -> Self {
            Self {
                num_features: 64,
                history_len: 1000,
                block_size: 256,
                use_parallel_reduction: true,
                use_shared_memory: true,
            }
        }
    }

    impl EncodeStateConfig {
        /// Create configuration for HybridTrainer state encoding
        ///
        /// Why: Matches TrainingState::compute_features() dimensions.
        #[must_use]
        pub fn for_hybrid_trainer() -> Self {
            Self {
                num_features: 64,
                history_len: 1000,
                block_size: 256,
                use_parallel_reduction: true,
                use_shared_memory: true,
            }
        }

        /// Calculate theoretical speedup vs CPU
        ///
        /// Why: Quantifies the benefit of GPU acceleration.
        ///
        /// # Returns
        ///
        /// `(cpu_time_ms, gpu_time_ms, speedup)`
        #[must_use]
        pub fn theoretical_speedup(&self) -> (f32, f32, f32) {
            // Operations: For each feature, scan history and compute stat
            // Each scan: history_len reads + computation
            let reads_per_feature = self.history_len;
            let total_reads = self.num_features * reads_per_feature;

            // CPU: Sequential processing
            // Assume 10ns per memory read + 5ns for computation
            let cpu_time_per_read_ns = 15.0;
            let cpu_time_ms = (total_reads as f32 * cpu_time_per_read_ns) / 1e6;

            // GPU: Parallel processing
            // Same total reads, but parallelized across all features and within each feature
            // Parallelism: num_features blocks × block_size threads
            let parallelism = self.num_features as f32 * self.block_size as f32;
            let gpu_time_per_read_ns = 15.0 / parallelism;
            let gpu_time_ms = (total_reads as f32 * gpu_time_per_read_ns) / 1e6;

            let speedup = cpu_time_ms / gpu_time_ms;

            (cpu_time_ms, gpu_time_ms, speedup)
        }

        /// Validate configuration
        ///
        /// Why: Ensures parameters are compatible with GPU constraints.
        pub fn validate(&self) -> Result<(), String> {
            if self.num_features == 0 || self.num_features > 1024 {
                return Err(format!(
                    "Invalid num_features: {} (must be 1-1024)",
                    self.num_features
                ));
            }

            if self.history_len == 0 || self.history_len > 1_000_000 {
                return Err(format!(
                    "Invalid history_len: {} (must be 1-1000000)",
                    self.history_len
                ));
            }

            if self.block_size == 0 || self.block_size > 1024 {
                return Err(format!(
                    "Invalid block_size: {} (must be 1-1024)",
                    self.block_size
                ));
            }

            // Block size should be power of 2 for efficient reductions
            if self.use_parallel_reduction && !self.block_size.is_power_of_two() {
                return Err(format!(
                    "block_size must be power of 2 for parallel reduction, got {}",
                    self.block_size
                ));
            }

            Ok(())
        }
    }

    /// State encoding kernel statistics
    ///
    /// Why: Track performance of GPU-accelerated feature extraction.
    #[derive(Debug, Clone)]
    pub struct EncodeStateStats {
        /// Number of encoding operations performed
        pub num_encodings: usize,

        /// Total time spent in kernel (microseconds)
        pub total_kernel_time_us: f32,

        /// Average time per encoding (microseconds)
        pub avg_kernel_time_us: f32,

        /// Peak memory usage (bytes)
        pub peak_memory_bytes: usize,

        /// Achieved speedup vs CPU baseline
        pub speedup_vs_cpu: f32,
    }

    impl Default for EncodeStateStats {
        fn default() -> Self {
            Self {
                num_encodings: 0,
                total_kernel_time_us: 0.0,
                avg_kernel_time_us: 0.0,
                peak_memory_bytes: 0,
                speedup_vs_cpu: 1.0,
            }
        }
    }

    impl std::fmt::Display for EncodeStateStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "State Encoding GPU Stats: {} encodings | {:.2} µs/encoding | {:.1}× speedup | Peak: {:.2} MB",
                self.num_encodings,
                self.avg_kernel_time_us,
                self.speedup_vs_cpu,
                self.peak_memory_bytes as f32 / (1024.0 * 1024.0)
            )
        }
    }

    /// State encoding GPU kernel (CubeCL)
    ///
    /// Why: Parallel feature extraction from TrainingState for 10-20× speedup.
    ///
    /// # Algorithm
    ///
    /// This kernel parallelizes statistical computations over history buffers:
    ///
    /// 1. **Parallel Statistics**: Each feature computed by one thread block
    /// 2. **Tree Reduction**: Logarithmic-time sum/min/max computations
    /// 3. **Fused Operations**: Mean, std, min, max in single pass
    ///
    /// # CubeCL Implementation
    ///
    /// The actual CubeCL kernel would look like:
    ///
    /// ```rust,ignore
    /// #[cube(launch)]
    /// fn encode_state_kernel<F: Float>(
    ///     loss_history: &Array<F>,
    ///     grad_history: &Array<F>,
    ///     lr_history: &Array<F>,
    ///     features: &mut Array<F>,
    ///     history_len: u32,
    ///     num_features: u32,
    /// ) {
    ///     let feature_idx = CUBE_POS_X;
    ///     let tid = UNIT_POS_X;
    ///
    ///     if feature_idx >= num_features { return; }
    ///
    ///     // Shared memory for reduction
    ///     let shared = SharedMemory::<F>::new(256usize);
    ///
    ///     // Compute feature via parallel reduction
    ///     // Each thread processes subset of history
    ///     let mut local_val = F::new(0.0);
    ///     for i in (tid..history_len).step_by(CUBE_DIM_X) {
    ///         local_val += loss_history[i as usize];
    ///     }
    ///
    ///     // Tree reduction
    ///     shared[tid as usize] = local_val;
    ///     sync_cube();
    ///
    ///     let mut stride = 128u32;
    ///     while stride > 0 {
    ///         if tid < stride {
    ///             shared[tid as usize] += shared[(tid + stride) as usize];
    ///         }
    ///         sync_cube();
    ///         stride /= 2;
    ///     }
    ///
    ///     if tid == 0 {
    ///         features[feature_idx as usize] = shared[0] / F::cast_from(history_len);
    ///     }
    /// }
    /// ```
    ///
    /// # Note
    ///
    /// This is a placeholder. Actual implementation requires:
    /// 1. Create `src/gpu/encode_state_kernel.cube` with CubeCL kernel
    /// 2. Implement parallel reductions for all stat types (mean, std, min, max)
    /// 3. Add kernel launch wrapper
    /// 4. Benchmark vs CPU implementation
    pub fn encode_state(
        _loss_history: &[f32],
        _grad_norm_history: &[f32],
        _lr_history: &[f32],
        _config: &EncodeStateConfig,
    ) -> Vec<f32> {
        // Placeholder - actual implementation would:
        // 1. Transfer history buffers to GPU
        // 2. Launch parallel feature extraction kernel
        // 3. Transfer features back to CPU
        //
        // For now, return empty vector
        // TODO: Implement CubeCL kernel
        Vec::new()
    }

    /// GRU forward pass kernel configuration.
    ///
    /// Configuration for GPU-accelerated GRU cell computation used in
    /// RSSM dynamics model. The GRU cell is the computational bottleneck
    /// in training trajectory prediction.
    ///
    /// # Why GPU Acceleration?
    ///
    /// GRU forward pass involves multiple matrix-vector multiplications:
    /// - Update gate: `z = σ(W_z·x + U_z·h + b_z)`
    /// - Reset gate: `r = σ(W_r·x + U_r·h + b_r)`
    /// - Candidate: `h' = tanh(W_h·x + U_h·(r⊙h) + b_h)`
    /// - Output: `h_new = (1-z)⊙h + z⊙h'`
    ///
    /// Each step requires:
    /// - 6 matrix-vector multiplications (3 gates × 2 matrices each)
    /// - Element-wise operations (sigmoid, tanh, hadamard products)
    ///
    /// **CPU performance** (for hidden_dim=256, input_dim=64):
    /// - ~0.5-1.0 ms per step on modern CPU
    /// - Bottlenecked by sequential matvec operations
    ///
    /// **GPU performance target** (with fused kernel):
    /// - ~0.01-0.05 ms per step
    /// - **5-50× speedup** through parallelization
    /// - Fuse all operations into single kernel launch
    /// - Reuse shared memory across gates
    ///
    /// # Algorithm
    ///
    /// The GPU kernel fuses all GRU operations:
    ///
    /// ```text
    /// // Thread layout: One block per output element
    /// // Each thread computes one element of hidden state
    ///
    /// __global__ void gru_fused_kernel(
    ///     float* input,      // [input_dim]
    ///     float* h_prev,     // [hidden_dim]
    ///     float* W_z, W_r, W_h,  // Weights [hidden_dim × input_dim]
    ///     float* U_z, U_r, U_h,  // Recurrent [hidden_dim × hidden_dim]
    ///     float* b_z, b_r, b_h,  // Biases [hidden_dim]
    ///     float* h_new       // Output [hidden_dim]
    /// ) {
    ///     int i = blockIdx.x * blockDim.x + threadIdx.x;
    ///     if (i >= hidden_dim) return;
    ///
    ///     // Compute update gate element i
    ///     float z_i = 0.0f;
    ///     for (int j = 0; j < input_dim; j++) {
    ///         z_i += W_z[i * input_dim + j] * input[j];
    ///     }
    ///     for (int j = 0; j < hidden_dim; j++) {
    ///         z_i += U_z[i * hidden_dim + j] * h_prev[j];
    ///     }
    ///     z_i = sigmoid(z_i + b_z[i]);
    ///
    ///     // Similar for reset gate r_i
    ///     // Similar for candidate h'_i
    ///     // Combine: h_new[i] = (1-z_i)*h_prev[i] + z_i*h_candidate[i]
    /// }
    /// ```
    ///
    /// # Memory Layout
    ///
    /// All matrices stored in row-major format for coalesced access:
    /// - Weights: `[hidden_dim, input_dim]`
    /// - Recurrent: `[hidden_dim, hidden_dim]`
    /// - States: `[hidden_dim]`
    ///
    /// # Optimization Strategies
    ///
    /// 1. **Kernel Fusion**: All gates computed in single kernel launch
    /// 2. **Shared Memory**: Cache input vector in shared memory
    /// 3. **Coalesced Access**: Row-major layout for sequential access
    /// 4. **Register Reuse**: Intermediate values stay in registers
    /// 5. **Vectorized Loads**: Use float4 for aligned memory access
    ///
    /// # Integration with RSSM
    ///
    /// The RSSM dynamics model uses ensemble of GRU cells (typically 5):
    /// - Launch one kernel per ensemble member (parallel execution)
    /// - Or batch all ensemble members into single kernel with member dimension
    /// - Batching reduces kernel launch overhead for small hidden dimensions
    ///
    /// # References
    ///
    /// - GRU: "Learning Phrase Representations using RNN Encoder-Decoder"
    ///   Cho et al., 2014 (https://arxiv.org/abs/1406.1078)
    /// - GPU optimization: Similar to LSTM optimizations in cuDNN
    #[derive(Debug, Clone)]
    pub struct GruConfig {
        /// Hidden state dimension.
        ///
        /// Why: Determines the output size and recurrent weight matrices.
        /// Typical values: 128-512 for trajectory prediction tasks.
        pub hidden_dim: usize,

        /// Input dimension.
        ///
        /// Why: Determines the input weight matrix size.
        /// For HybridTrainer: 64 (from TrainingState::compute_features())
        pub input_dim: usize,

        /// Batch size.
        ///
        /// Why: Number of sequences processed in parallel.
        /// GPU kernel can process multiple sequences simultaneously.
        pub batch_size: usize,

        /// Enable kernel fusion.
        ///
        /// Why: Fusing all GRU gates into single kernel reduces memory bandwidth
        /// and kernel launch overhead. Disable for debugging to isolate gates.
        pub fused: bool,

        /// Use shared memory for input caching.
        ///
        /// Why: Input vector is reused across all hidden units.
        /// Caching in shared memory reduces global memory bandwidth.
        pub use_shared_memory: bool,
    }

    impl Default for GruConfig {
        fn default() -> Self {
            Self {
                hidden_dim: 256,
                input_dim: 128,
                batch_size: 1,
                fused: true,
                use_shared_memory: true,
            }
        }
    }

    impl GruConfig {
        /// Create configuration for HybridTrainer RSSM
        ///
        /// Why: RSSM uses specific dimensions based on TrainingState encoding.
        #[must_use]
        pub fn for_rssm() -> Self {
            Self {
                hidden_dim: 256,        // RSSM deterministic state size
                input_dim: 64,          // TrainingState features
                batch_size: 1,          // Single trajectory
                fused: true,
                use_shared_memory: true,
            }
        }

        /// Calculate theoretical speedup vs CPU
        ///
        /// Why: Helps users understand the performance benefit of GPU acceleration.
        ///
        /// # Returns
        ///
        /// `(cpu_time_ms, gpu_time_ms, speedup)`
        ///
        /// # Assumptions
        ///
        /// - CPU: Sequential matvec, ~10 GFLOPS (single-threaded)
        /// - GPU: Parallel matvec, ~1000 GFLOPS (RTX 4090)
        #[must_use]
        pub fn theoretical_speedup(&self) -> (f32, f32, f32) {
            // Operations per GRU step:
            // - 3 gates × (2 matvecs + bias + activation) = 6 matvecs + 3 element-wise ops
            // - Final combination: 3 element-wise ops
            let ops_per_step = (6 * self.hidden_dim * self.input_dim)
                + (6 * self.hidden_dim * self.hidden_dim)
                + (6 * self.hidden_dim); // Element-wise

            // CPU performance (single-threaded)
            let cpu_gflops = 10.0; // Conservative estimate for modern CPU
            let cpu_time_ms = (ops_per_step as f32 / (cpu_gflops * 1e9)) * 1000.0;

            // GPU performance (NVIDIA RTX 4090 / similar)
            let gpu_gflops = 1000.0; // Conservative estimate for consumer GPU
            let gpu_time_ms = (ops_per_step as f32 / (gpu_gflops * 1e9)) * 1000.0;

            let speedup = cpu_time_ms / gpu_time_ms;

            (cpu_time_ms, gpu_time_ms, speedup)
        }

        /// Validate configuration
        ///
        /// Why: Ensures parameters are compatible with GPU constraints.
        pub fn validate(&self) -> Result<(), String> {
            if self.hidden_dim == 0 || self.hidden_dim > 4096 {
                return Err(format!(
                    "Invalid hidden_dim: {} (must be 1-4096)",
                    self.hidden_dim
                ));
            }

            if self.input_dim == 0 || self.input_dim > 4096 {
                return Err(format!(
                    "Invalid input_dim: {} (must be 1-4096)",
                    self.input_dim
                ));
            }

            if self.batch_size == 0 || self.batch_size > 1024 {
                return Err(format!(
                    "Invalid batch_size: {} (must be 1-1024)",
                    self.batch_size
                ));
            }

            Ok(())
        }
    }

    /// Flash Attention kernel configuration.
    ///
    /// Flash Attention is a fused attention kernel that reduces memory usage
    /// from O(n²) to O(n) by computing attention incrementally in blocks
    /// without materializing the full attention matrix.
    ///
    /// # Memory Savings
    ///
    /// **Standard Attention**:
    /// ```text
    /// Q @ K^T → [batch, heads, seq_len, seq_len]  // O(n²) memory
    /// softmax → [batch, heads, seq_len, seq_len]  // O(n²) memory
    /// @ V     → [batch, heads, seq_len, d_head]   // Final output
    /// ```
    ///
    /// **Flash Attention**:
    /// ```text
    /// Process in blocks of size block_size:
    /// - Load Q block:  [block_size, d_head]       // O(n) memory
    /// - Load K block:  [block_size, d_head]       // O(n) memory
    /// - Compute QK^T:  [block_size, block_size]   // O(1) block memory
    /// - Softmax + V:   Fused, no materialization
    /// - Accumulate:    [seq_len, d_head]          // Final output only
    /// ```
    ///
    /// **Result**: O(n²) → O(n) memory, ~99% reduction for large sequences
    ///
    /// # Why This Matters
    ///
    /// For transformer models, attention is often the memory bottleneck:
    /// - Sequence length 2048, 32 heads, fp16 precision
    /// - Standard: 2048² × 32 × 2 bytes = 256 MB per layer
    /// - Flash: 2048 × 32 × 2 bytes = 128 KB per layer
    /// - **Savings: 99.95%** (256 MB → 128 KB)
    ///
    /// # Algorithm
    ///
    /// Flash Attention uses a tiling strategy with online softmax:
    ///
    /// 1. **Tiling**: Split Q, K, V into blocks of size `block_size`
    /// 2. **Incremental Softmax**: Compute softmax statistics incrementally
    ///    - Track running max and sum for numerically stable softmax
    ///    - Update statistics as each block is processed
    /// 3. **Fused Operations**: QK^T + softmax + matmul in single kernel
    ///    - No intermediate materialization
    ///    - Better cache locality
    /// 4. **Output Accumulation**: Accumulate final output incrementally
    ///
    /// # Trade-offs
    ///
    /// - **Memory**: 99% reduction (O(n²) → O(n))
    /// - **Compute**: +10-20% due to recomputation (worth it for memory savings)
    /// - **Accuracy**: Numerically identical to standard attention
    ///
    /// # HybridTrainer Integration
    ///
    /// Flash Attention is beneficial in all training phases:
    /// - **Full phase**: Reduces peak memory during backward pass
    /// - **Predict phase**: Smaller memory footprint for forward-only inference
    /// - **Correct phase**: Enables larger validation batches
    ///
    /// Combined with gradient checkpointing and quantization, enables
    /// training of massive transformer models (7B-50B params) on consumer GPUs.
    ///
    /// # References
    ///
    /// - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    ///   Dao et al., 2022 (https://arxiv.org/abs/2205.14135)
    /// - "FlashAttention-2: Faster Attention with Better Parallelism"
    ///   Dao, 2023 (https://arxiv.org/abs/2307.08691)
    #[derive(Debug, Clone)]
    pub struct FlashAttentionConfig {
        /// Sequence length (number of tokens)
        ///
        /// Why: Determines the size of the attention matrix (seq_len × seq_len)
        /// and the memory savings from Flash Attention.
        pub seq_len: usize,

        /// Number of attention heads
        ///
        /// Why: Multi-head attention processes multiple attention patterns
        /// in parallel. Each head has its own QKV projections.
        pub num_heads: usize,

        /// Head dimension (d_model / num_heads)
        ///
        /// Why: Dimension of each attention head. Typically 64 or 128.
        /// Total model dimension = num_heads × head_dim
        pub head_dim: usize,

        /// Block size for tiling (default: 256)
        ///
        /// Why: Determines the tile size for blocked computation.
        /// Larger blocks = more parallelism but more recomputation.
        /// Smaller blocks = less memory but more kernel launches.
        ///
        /// Recommended values:
        /// - Small models (<1B): 128
        /// - Medium models (1-7B): 256
        /// - Large models (7B+): 512
        pub block_size: usize,

        /// Batch size
        ///
        /// Why: Number of sequences processed in parallel.
        /// Flash Attention processes each sequence independently.
        pub batch_size: usize,

        /// Enable causal masking (for autoregressive models)
        ///
        /// Why: Causal masking prevents attending to future tokens.
        /// Required for GPT-style models, not needed for BERT-style models.
        pub causal: bool,

        /// Dropout probability (0.0 = no dropout)
        ///
        /// Why: Dropout on attention weights for regularization.
        /// Flash Attention can fuse dropout into the kernel.
        pub dropout: f32,
    }

    impl Default for FlashAttentionConfig {
        fn default() -> Self {
            Self {
                seq_len: 2048,
                num_heads: 16,
                head_dim: 64,
                block_size: 256,
                batch_size: 1,
                causal: false,
                dropout: 0.0,
            }
        }
    }

    impl FlashAttentionConfig {
        /// Create configuration for GPT-2 style model
        ///
        /// Why: GPT-2 uses causal attention with specific dimensions.
        /// Provides a convenient constructor for common use cases.
        #[must_use]
        pub fn gpt2(seq_len: usize) -> Self {
            Self {
                seq_len,
                num_heads: 12,
                head_dim: 64,
                block_size: 256,
                batch_size: 1,
                causal: true, // Autoregressive
                dropout: 0.1,
            }
        }

        /// Create configuration for BERT style model
        ///
        /// Why: BERT uses bidirectional attention (no causal masking).
        #[must_use]
        pub fn bert(seq_len: usize) -> Self {
            Self {
                seq_len,
                num_heads: 12,
                head_dim: 64,
                block_size: 256,
                batch_size: 1,
                causal: false, // Bidirectional
                dropout: 0.1,
            }
        }

        /// Calculate theoretical memory savings vs standard attention
        ///
        /// Why: Quantifies the memory reduction from Flash Attention.
        /// Helps users understand the benefit before implementation.
        ///
        /// # Returns
        ///
        /// `(standard_mb, flash_mb, savings_percent)`
        ///
        /// # Algorithm
        ///
        /// Standard attention memory:
        /// - Attention matrix: batch × heads × seq_len × seq_len × 2 bytes (fp16)
        /// - Softmax output:   batch × heads × seq_len × seq_len × 2 bytes (fp16)
        /// - Total: 2 × batch × heads × seq_len² × 2 bytes
        ///
        /// Flash attention memory:
        /// - Output only: batch × heads × seq_len × head_dim × 2 bytes (fp16)
        /// - Block buffers: 2 × block_size × head_dim × 2 bytes (fp16)
        /// - Total: batch × heads × seq_len × head_dim × 2 + small overhead
        #[must_use]
        pub fn theoretical_savings(&self) -> (f32, f32, f32) {
            let bytes_per_element = 2.0; // fp16

            // Standard attention: QK^T matrix + softmax output
            let attention_matrix_bytes =
                (self.batch_size * self.num_heads * self.seq_len * self.seq_len) as f32
                    * bytes_per_element;
            let standard_mb = 2.0 * attention_matrix_bytes / (1024.0 * 1024.0);

            // Flash attention: Output + block buffers
            let output_bytes = (self.batch_size * self.num_heads * self.seq_len * self.head_dim)
                as f32
                * bytes_per_element;
            let block_buffer_bytes = (2 * self.block_size * self.head_dim) as f32
                * bytes_per_element;
            let flash_mb = (output_bytes + block_buffer_bytes) / (1024.0 * 1024.0);

            let savings_percent = ((standard_mb - flash_mb) / standard_mb) * 100.0;

            (standard_mb, flash_mb, savings_percent)
        }

        /// Validate configuration
        ///
        /// Why: Ensures parameters are sensible before kernel launch.
        /// Prevents cryptic CUDA errors from invalid configurations.
        ///
        /// # Errors
        ///
        /// Returns an error if:
        /// - seq_len is 0 or too large (>65536)
        /// - num_heads is 0 or too large (>128)
        /// - head_dim is 0 or not a multiple of 8
        /// - block_size is 0 or too large (>1024)
        /// - dropout is not in [0.0, 1.0)
        pub fn validate(&self) -> Result<(), String> {
            if self.seq_len == 0 || self.seq_len > 65536 {
                return Err(format!(
                    "Invalid seq_len: {} (must be 1-65536)",
                    self.seq_len
                ));
            }

            if self.num_heads == 0 || self.num_heads > 128 {
                return Err(format!(
                    "Invalid num_heads: {} (must be 1-128)",
                    self.num_heads
                ));
            }

            if self.head_dim == 0 || self.head_dim % 8 != 0 {
                return Err(format!(
                    "Invalid head_dim: {} (must be positive and multiple of 8)",
                    self.head_dim
                ));
            }

            if self.block_size == 0 || self.block_size > 1024 {
                return Err(format!(
                    "Invalid block_size: {} (must be 1-1024)",
                    self.block_size
                ));
            }

            if !(0.0..1.0).contains(&self.dropout) {
                return Err(format!(
                    "Invalid dropout: {} (must be in [0.0, 1.0))",
                    self.dropout
                ));
            }

            Ok(())
        }
    }

    /// Flash Attention statistics
    ///
    /// Why: Track memory usage and performance metrics to validate
    /// the memory savings and identify optimization opportunities.
    #[derive(Debug, Clone)]
    pub struct FlashAttentionStats {
        /// Number of forward passes executed
        pub num_forward_passes: usize,

        /// Total memory saved compared to standard attention (bytes)
        ///
        /// Why: Quantifies the actual memory reduction achieved.
        pub memory_saved_bytes: usize,

        /// Average kernel execution time (microseconds)
        ///
        /// Why: Tracks the compute overhead from recomputation.
        pub avg_kernel_time_us: f32,

        /// Peak memory usage (bytes)
        ///
        /// Why: Validates that memory usage stays within expected bounds.
        pub peak_memory_bytes: usize,
    }

    impl Default for FlashAttentionStats {
        fn default() -> Self {
            Self {
                num_forward_passes: 0,
                memory_saved_bytes: 0,
                avg_kernel_time_us: 0.0,
                peak_memory_bytes: 0,
            }
        }
    }

    impl std::fmt::Display for FlashAttentionStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Flash Attention Stats: {} passes | {:.2} MB saved | {:.2} µs/pass | Peak: {:.2} MB",
                self.num_forward_passes,
                self.memory_saved_bytes as f32 / (1024.0 * 1024.0),
                self.avg_kernel_time_us,
                self.peak_memory_bytes as f32 / (1024.0 * 1024.0)
            )
        }
    }

    /// GRU kernel statistics
    ///
    /// Why: Track performance and validate GPU acceleration effectiveness.
    #[derive(Debug, Clone)]
    pub struct GruKernelStats {
        /// Number of forward passes executed
        pub num_forward_passes: usize,

        /// Total time spent in kernel (microseconds)
        ///
        /// Why: Measure actual GPU kernel execution time.
        pub total_kernel_time_us: f32,

        /// Average time per forward pass (microseconds)
        pub avg_kernel_time_us: f32,

        /// Peak memory usage (bytes)
        ///
        /// Why: Track GPU memory consumption for GRU weights and states.
        pub peak_memory_bytes: usize,

        /// Achieved speedup vs CPU baseline
        ///
        /// Why: Validate that GPU acceleration provides expected benefit.
        pub speedup_vs_cpu: f32,
    }

    impl Default for GruKernelStats {
        fn default() -> Self {
            Self {
                num_forward_passes: 0,
                total_kernel_time_us: 0.0,
                avg_kernel_time_us: 0.0,
                peak_memory_bytes: 0,
                speedup_vs_cpu: 1.0,
            }
        }
    }

    impl std::fmt::Display for GruKernelStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "GRU GPU Stats: {} passes | {:.2} µs/pass | {:.1}× speedup | Peak: {:.2} MB",
                self.num_forward_passes,
                self.avg_kernel_time_us,
                self.speedup_vs_cpu,
                self.peak_memory_bytes as f32 / (1024.0 * 1024.0)
            )
        }
    }

    /// GRU weights for GPU kernel
    ///
    /// Why: Structured layout for efficient GPU memory access.
    /// All weights stored in row-major format for coalesced memory reads.
    #[derive(Debug, Clone)]
    pub struct GruWeightsGpu {
        /// Update gate input weights [hidden_dim × input_dim]
        pub w_z: Vec<f32>,
        /// Update gate recurrent weights [hidden_dim × hidden_dim]
        pub u_z: Vec<f32>,
        /// Update gate bias [hidden_dim]
        pub b_z: Vec<f32>,

        /// Reset gate input weights [hidden_dim × input_dim]
        pub w_r: Vec<f32>,
        /// Reset gate recurrent weights [hidden_dim × hidden_dim]
        pub u_r: Vec<f32>,
        /// Reset gate bias [hidden_dim]
        pub b_r: Vec<f32>,

        /// Candidate input weights [hidden_dim × input_dim]
        pub w_h: Vec<f32>,
        /// Candidate recurrent weights [hidden_dim × hidden_dim]
        pub u_h: Vec<f32>,
        /// Candidate bias [hidden_dim]
        pub b_h: Vec<f32>,
    }

    /// GRU forward pass kernel (CubeCL)
    ///
    /// Why: Fused GRU cell computation on GPU for 5-50× speedup over CPU.
    ///
    /// # Algorithm
    ///
    /// This kernel implements a fused GRU cell that computes all gates
    /// in a single GPU kernel launch. The standard GRU equations are:
    ///
    /// ```text
    /// z = σ(W_z·x + U_z·h + b_z)           // Update gate
    /// r = σ(W_r·x + U_r·h + b_r)           // Reset gate
    /// h' = tanh(W_h·x + U_h·(r⊙h) + b_h)   // Candidate
    /// h_new = (1-z)⊙h + z⊙h'               // Output
    /// ```
    ///
    /// # Kernel Fusion Strategy
    ///
    /// Instead of 3 separate kernel launches (one per gate), we fuse all
    /// operations into a single kernel:
    ///
    /// 1. **Thread Layout**: Each thread computes one element of h_new
    ///    - Block size: 256 threads (good occupancy)
    ///    - Grid size: ceil(hidden_dim / 256) blocks
    ///
    /// 2. **Shared Memory**: Input vector cached in shared memory
    ///    - Reduces global memory bandwidth by ~6×
    ///    - All threads in block reuse same input
    ///
    /// 3. **Register Pressure**: Intermediate values stay in registers
    ///    - z, r, h' stored in registers, not written to global memory
    ///    - Only final h_new written back
    ///
    /// # Memory Bandwidth Analysis
    ///
    /// **Without Fusion** (3 separate kernels):
    /// - Read input 3 times: 3 × input_dim × 4 bytes
    /// - Read h_prev 3 times: 3 × hidden_dim × 4 bytes
    /// - Read 9 weight matrices
    /// - Write 3 intermediate results
    /// - Total: ~30× more memory bandwidth
    ///
    /// **With Fusion** (single kernel):
    /// - Read input once (cached in shared memory)
    /// - Read h_prev once
    /// - Read 9 weight matrices
    /// - Write only final h_new
    /// - **Result**: 6× reduction in memory bandwidth
    ///
    /// # CubeCL Implementation
    ///
    /// The actual CubeCL kernel would look like:
    ///
    /// ```rust,ignore
    /// #[cube(launch)]
    /// fn gru_fused_kernel<F: Float>(
    ///     input: &Array<F>,
    ///     h_prev: &Array<F>,
    ///     weights: &GruWeightsGpu,
    ///     h_new: &mut Array<F>,
    ///     hidden_dim: u32,
    ///     input_dim: u32,
    /// ) {
    ///     let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    ///     if i >= hidden_dim { return; }
    ///
    ///     // Cache input in shared memory
    ///     let shared_input = SharedMemory::<F>::new(input_dim as usize);
    ///     if UNIT_POS_X < input_dim {
    ///         shared_input[UNIT_POS_X as usize] = input[UNIT_POS_X as usize];
    ///     }
    ///     sync_cube();
    ///
    ///     // Compute update gate z_i
    ///     let mut z_i = F::new(0.0);
    ///     for j in 0..input_dim {
    ///         z_i += weights.w_z[(i * input_dim + j) as usize] * shared_input[j as usize];
    ///     }
    ///     for j in 0..hidden_dim {
    ///         z_i += weights.u_z[(i * hidden_dim + j) as usize] * h_prev[j as usize];
    ///     }
    ///     z_i = sigmoid(z_i + weights.b_z[i as usize]);
    ///
    ///     // Similar for reset gate r_i
    ///     // Similar for candidate h'_i
    ///     // Combine: h_new[i] = (1-z_i)*h_prev[i] + z_i*h'_i
    /// }
    /// ```
    ///
    /// # Note
    ///
    /// This is a placeholder. The actual CubeCL implementation would be
    /// in a separate module with the `#[cube(launch)]` macro.
    ///
    /// To implement:
    /// 1. Create `src/gpu/gru_kernel.cube` with CubeCL kernel
    /// 2. Add CubeCL device setup and memory management
    /// 3. Implement kernel launch wrapper
    /// 4. Add benchmarks comparing CPU vs GPU performance
    pub fn gru_forward(
        _input: &[f32],
        _h_prev: &[f32],
        _weights: &GruWeightsGpu,
        _config: &GruConfig,
    ) -> Vec<f32> {
        // Placeholder - actual implementation would:
        // 1. Transfer input, h_prev, weights to GPU
        // 2. Launch fused GRU kernel
        // 3. Transfer h_new back to CPU
        //
        // For now, return empty vector
        // TODO: Implement CubeCL kernel
        Vec::new()
    }

    /// Flash Attention kernel implementation (CubeCL)
    ///
    /// Why: Actual GPU kernel that performs fused QK^T + softmax + matmul.
    /// This is a placeholder for the CubeCL implementation.
    ///
    /// # Algorithm Pseudocode
    ///
    /// ```text
    /// function flash_attention(Q, K, V, block_size):
    ///     N = seq_len
    ///     Output = zeros(N, d_head)
    ///     RowMax = -inf * ones(N)    // Running max for softmax
    ///     RowSum = zeros(N)           // Running sum for softmax
    ///
    ///     // Process in blocks (tiling)
    ///     for block_i in range(0, N, block_size):
    ///         for block_j in range(0, N, block_size):
    ///             // Load blocks
    ///             Q_block = Q[block_i:block_i+block_size, :]
    ///             K_block = K[block_j:block_j+block_size, :]
    ///             V_block = V[block_j:block_j+block_size, :]
    ///
    ///             // Compute attention scores
    ///             S_block = Q_block @ K_block.T  // [block_size, block_size]
    ///
    ///             // Online softmax update
    ///             for row in block_i:block_i+block_size:
    ///                 old_max = RowMax[row]
    ///                 new_max = max(old_max, max(S_block[row, :]))
    ///
    ///                 // Rescale previous contributions
    ///                 scale = exp(old_max - new_max)
    ///                 Output[row, :] *= scale
    ///                 RowSum[row] *= scale
    ///
    ///                 // Add new contributions
    ///                 P_block = exp(S_block[row, :] - new_max)
    ///                 Output[row, :] += P_block @ V_block
    ///                 RowSum[row] += sum(P_block)
    ///
    ///                 RowMax[row] = new_max
    ///
    ///     // Normalize by row sums
    ///     Output /= RowSum[:, None]
    ///     return Output
    /// ```
    ///
    /// # CubeCL Implementation
    ///
    /// The actual CubeCL kernel would:
    /// 1. Use shared memory for Q, K, V blocks
    /// 2. Use warp-level primitives for reductions (max, sum)
    /// 3. Fuse operations to minimize memory traffic
    /// 4. Handle causal masking efficiently
    /// 5. Support fp16/bf16 for memory efficiency
    ///
    /// # Note
    ///
    /// This is a placeholder. The actual CubeCL implementation would be
    /// in a separate `.cube` file using CubeCL's Rust-embedded DSL.
    ///
    /// Example kernel launch:
    /// ```rust,ignore
    /// use cubecl::prelude::*;
    ///
    /// #[cube(launch)]
    /// fn flash_attention_kernel(
    ///     q: &Tensor<f16>,
    ///     k: &Tensor<f16>,
    ///     v: &Tensor<f16>,
    ///     output: &mut Tensor<f16>,
    ///     config: FlashAttentionConfig,
    /// ) {
    ///     // CubeCL kernel code here
    /// }
    /// ```
    pub fn flash_attention_forward(
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _config: &FlashAttentionConfig,
    ) -> Vec<f32> {
        // Placeholder - actual implementation would:
        // 1. Transfer Q, K, V to GPU
        // 2. Launch Flash Attention kernel with tiling
        // 3. Return output tensor
        //
        // For now, return empty vector
        // TODO: Implement CubeCL kernel
        Vec::new()
    }
}

/// Burn tensor operations wrapper.
///
/// Burn-based tensor operations for GPU acceleration.
pub mod burn_ops {

    /// Performs matrix multiplication using Burn.
    #[must_use]
    pub fn matmul(_a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        // Placeholder - actual implementation would use burn::tensor::Tensor
        Vec::new()
    }

    /// Performs element-wise operations using Burn.
    #[must_use]
    pub fn elementwise_add(_a: &[f32], _b: &[f32]) -> Vec<f32> {
        // Placeholder
        Vec::new()
    }

    /// Computes softmax using Burn.
    #[must_use]
    pub fn softmax(_x: &[f32], _dim: usize) -> Vec<f32> {
        // Placeholder
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_default() {
        let info = DeviceInfo::default();
        assert_eq!(info.name, "Unknown");
    }

    #[test]
    fn test_gpu_tensor_empty() {
        let tensor = GpuTensor::empty();
        assert_eq!(tensor.numel(), 1); // Empty shape has product 1
    }

    #[test]
    fn test_kernel_config_defaults() {
        let encode_config = kernels::EncodeStateConfig::default();
        assert_eq!(encode_config.block_size, 256);

        let gru_config = kernels::GruConfig::default();
        assert_eq!(gru_config.hidden_dim, 256);
    }

    #[test]
    fn test_flash_attention_config_defaults() {
        let config = kernels::FlashAttentionConfig::default();
        assert_eq!(config.seq_len, 2048);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.block_size, 256);
        assert_eq!(config.batch_size, 1);
        assert!(!config.causal);
        assert!((config.dropout - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_flash_attention_gpt2_config() {
        let config = kernels::FlashAttentionConfig::gpt2(1024);
        assert_eq!(config.seq_len, 1024);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64);
        assert!(config.causal); // GPT-2 uses causal attention
        assert!((config.dropout - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_flash_attention_bert_config() {
        let config = kernels::FlashAttentionConfig::bert(512);
        assert_eq!(config.seq_len, 512);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64);
        assert!(!config.causal); // BERT uses bidirectional attention
    }

    #[test]
    fn test_flash_attention_theoretical_savings() {
        let config = kernels::FlashAttentionConfig {
            seq_len: 2048,
            num_heads: 32,
            head_dim: 64,
            block_size: 256,
            batch_size: 1,
            causal: false,
            dropout: 0.0,
        };

        let (standard_mb, flash_mb, savings_percent) = config.theoretical_savings();

        // Standard: 2 × 1 × 32 × 2048² × 2 bytes = 512 MB
        assert!(
            (standard_mb - 512.0).abs() < 1.0,
            "Expected ~512 MB standard, got {}",
            standard_mb
        );

        // Flash: 1 × 32 × 2048 × 64 × 2 bytes + small overhead ≈ 8 MB
        assert!(
            flash_mb < 10.0,
            "Expected <10 MB flash attention, got {}",
            flash_mb
        );

        // Savings should be >95%
        assert!(
            savings_percent > 95.0,
            "Expected >95% savings, got {}%",
            savings_percent
        );
    }

    #[test]
    fn test_flash_attention_validation_success() {
        let config = kernels::FlashAttentionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_flash_attention_validation_invalid_seq_len() {
        let mut config = kernels::FlashAttentionConfig::default();
        config.seq_len = 0;
        assert!(config.validate().is_err());

        config.seq_len = 100000; // Too large
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_flash_attention_validation_invalid_head_dim() {
        let mut config = kernels::FlashAttentionConfig::default();
        config.head_dim = 0;
        assert!(config.validate().is_err());

        config.head_dim = 63; // Not multiple of 8
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_flash_attention_validation_invalid_dropout() {
        let mut config = kernels::FlashAttentionConfig::default();
        config.dropout = -0.1; // Negative
        assert!(config.validate().is_err());

        config.dropout = 1.0; // Too high (should be <1.0)
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_flash_attention_stats_default() {
        let stats = kernels::FlashAttentionStats::default();
        assert_eq!(stats.num_forward_passes, 0);
        assert_eq!(stats.memory_saved_bytes, 0);
        assert_eq!(stats.avg_kernel_time_us, 0.0);
        assert_eq!(stats.peak_memory_bytes, 0);
    }

    #[test]
    fn test_flash_attention_stats_display() {
        let stats = kernels::FlashAttentionStats {
            num_forward_passes: 100,
            memory_saved_bytes: 256 * 1024 * 1024, // 256 MB
            avg_kernel_time_us: 123.45,
            peak_memory_bytes: 8 * 1024 * 1024, // 8 MB
        };
        let display = format!("{}", stats);
        assert!(display.contains("100 passes"));
        assert!(display.contains("256"));
        assert!(display.contains("123.45"));
    }

    #[test]
    fn test_flash_attention_forward_placeholder() {
        // Test placeholder function exists and returns empty vector
        let config = kernels::FlashAttentionConfig::default();
        let q = vec![1.0; 2048 * 64];
        let k = vec![1.0; 2048 * 64];
        let v = vec![1.0; 2048 * 64];

        let output = kernels::flash_attention_forward(&q, &k, &v, &config);
        assert!(output.is_empty()); // Placeholder returns empty
    }

    #[test]
    fn test_flash_attention_large_sequence() {
        // Test with very large sequence length to verify savings
        let config = kernels::FlashAttentionConfig {
            seq_len: 8192, // Large sequence
            num_heads: 40,
            head_dim: 128,
            block_size: 512,
            batch_size: 1,
            causal: true,
            dropout: 0.0,
        };

        let (standard_mb, flash_mb, savings_percent) = config.theoretical_savings();

        // Standard: 2 × 1 × 40 × 8192² × 2 bytes = 10.24 GB
        assert!(standard_mb > 10000.0, "Expected >10 GB standard");

        // Flash: Much smaller
        assert!(flash_mb < 100.0, "Expected <100 MB flash");

        // Savings should be >99%
        assert!(savings_percent > 99.0, "Expected >99% savings");
    }

    #[test]
    fn test_gru_config_defaults() {
        let config = kernels::GruConfig::default();
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.input_dim, 128);
        assert_eq!(config.batch_size, 1);
        assert!(config.fused);
        assert!(config.use_shared_memory);
    }

    #[test]
    fn test_gru_config_for_rssm() {
        let config = kernels::GruConfig::for_rssm();
        assert_eq!(config.hidden_dim, 256); // RSSM deterministic size
        assert_eq!(config.input_dim, 64);   // TrainingState features
        assert_eq!(config.batch_size, 1);
        assert!(config.fused);
    }

    #[test]
    fn test_gru_config_validation_success() {
        let config = kernels::GruConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gru_config_validation_invalid_hidden_dim() {
        let mut config = kernels::GruConfig::default();
        config.hidden_dim = 0;
        assert!(config.validate().is_err());

        config.hidden_dim = 5000; // Too large
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gru_config_validation_invalid_input_dim() {
        let mut config = kernels::GruConfig::default();
        config.input_dim = 0;
        assert!(config.validate().is_err());

        config.input_dim = 5000; // Too large
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gru_config_validation_invalid_batch_size() {
        let mut config = kernels::GruConfig::default();
        config.batch_size = 0;
        assert!(config.validate().is_err());

        config.batch_size = 2000; // Too large
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gru_theoretical_speedup() {
        let config = kernels::GruConfig::for_rssm();
        let (cpu_time_ms, gpu_time_ms, speedup) = config.theoretical_speedup();

        // CPU should be slower than GPU
        assert!(cpu_time_ms > gpu_time_ms, "Expected CPU slower than GPU");

        // Speedup should be significant (5-50×)
        assert!(
            speedup > 5.0,
            "Expected >5× speedup, got {}×",
            speedup
        );
        assert!(
            speedup < 100.0,
            "Expected <100× speedup (unrealistic), got {}×",
            speedup
        );
    }

    #[test]
    fn test_gru_kernel_stats_default() {
        let stats = kernels::GruKernelStats::default();
        assert_eq!(stats.num_forward_passes, 0);
        assert_eq!(stats.total_kernel_time_us, 0.0);
        assert_eq!(stats.avg_kernel_time_us, 0.0);
        assert_eq!(stats.peak_memory_bytes, 0);
        assert_eq!(stats.speedup_vs_cpu, 1.0);
    }

    #[test]
    fn test_gru_kernel_stats_display() {
        let stats = kernels::GruKernelStats {
            num_forward_passes: 1000,
            total_kernel_time_us: 50000.0,
            avg_kernel_time_us: 50.0,
            peak_memory_bytes: 10 * 1024 * 1024, // 10 MB
            speedup_vs_cpu: 25.0,
        };
        let display = format!("{}", stats);
        assert!(display.contains("1000 passes"));
        assert!(display.contains("50.00"));
        assert!(display.contains("25.0×"));
    }

    #[test]
    fn test_gru_forward_placeholder() {
        // Test placeholder function exists
        let config = kernels::GruConfig::for_rssm();
        let input = vec![1.0; 64];
        let h_prev = vec![0.5; 256];
        let weights = kernels::GruWeightsGpu {
            w_z: vec![0.1; 256 * 64],
            u_z: vec![0.1; 256 * 256],
            b_z: vec![0.0; 256],
            w_r: vec![0.1; 256 * 64],
            u_r: vec![0.1; 256 * 256],
            b_r: vec![0.0; 256],
            w_h: vec![0.1; 256 * 64],
            u_h: vec![0.1; 256 * 256],
            b_h: vec![0.0; 256],
        };

        let output = kernels::gru_forward(&input, &h_prev, &weights, &config);
        assert!(output.is_empty()); // Placeholder returns empty
    }

    #[test]
    fn test_gru_memory_requirements() {
        let config = kernels::GruConfig::for_rssm();

        // Calculate memory for weights
        let weights_memory =
            (config.hidden_dim * config.input_dim * 3)  // W_z, W_r, W_h
            + (config.hidden_dim * config.hidden_dim * 3)  // U_z, U_r, U_h
            + (config.hidden_dim * 3);  // b_z, b_r, b_h

        let weights_mb = (weights_memory * 4) as f32 / (1024.0 * 1024.0); // 4 bytes per f32

        // For RSSM config: 256×64×3 + 256×256×3 + 256×3 = 49152 + 196608 + 768 = 246528 floats
        // = 246528 × 4 bytes = 986112 bytes ≈ 0.94 MB
        assert!(weights_mb < 1.0, "Expected <1 MB for RSSM weights");
    }

    #[test]
    fn test_encode_state_config_defaults() {
        let config = kernels::EncodeStateConfig::default();
        assert_eq!(config.num_features, 64);
        assert_eq!(config.history_len, 1000);
        assert_eq!(config.block_size, 256);
        assert!(config.use_parallel_reduction);
        assert!(config.use_shared_memory);
    }

    #[test]
    fn test_encode_state_config_for_hybrid_trainer() {
        let config = kernels::EncodeStateConfig::for_hybrid_trainer();
        assert_eq!(config.num_features, 64); // TrainingState features
        assert_eq!(config.history_len, 1000);
        assert!(config.use_parallel_reduction);
    }

    #[test]
    fn test_encode_state_config_validation_success() {
        let config = kernels::EncodeStateConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_encode_state_config_validation_invalid_num_features() {
        let mut config = kernels::EncodeStateConfig::default();
        config.num_features = 0;
        assert!(config.validate().is_err());

        config.num_features = 2000; // Too large
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_encode_state_config_validation_invalid_history_len() {
        let mut config = kernels::EncodeStateConfig::default();
        config.history_len = 0;
        assert!(config.validate().is_err());

        config.history_len = 2_000_000; // Too large
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_encode_state_config_validation_block_size_power_of_two() {
        let mut config = kernels::EncodeStateConfig::default();
        config.block_size = 200; // Not power of 2
        let result = config.validate();
        assert!(
            result.is_err(),
            "Expected error for non-power-of-2 block size"
        );
    }

    #[test]
    fn test_encode_state_theoretical_speedup() {
        let config = kernels::EncodeStateConfig::for_hybrid_trainer();
        let (cpu_time_ms, gpu_time_ms, speedup) = config.theoretical_speedup();

        // CPU should be slower than GPU
        assert!(cpu_time_ms > gpu_time_ms, "Expected CPU slower than GPU");

        // Speedup should be significant (10-20×)
        assert!(
            speedup > 10.0,
            "Expected >10× speedup, got {}×",
            speedup
        );
    }

    #[test]
    fn test_encode_state_stats_default() {
        let stats = kernels::EncodeStateStats::default();
        assert_eq!(stats.num_encodings, 0);
        assert_eq!(stats.total_kernel_time_us, 0.0);
        assert_eq!(stats.avg_kernel_time_us, 0.0);
        assert_eq!(stats.peak_memory_bytes, 0);
        assert_eq!(stats.speedup_vs_cpu, 1.0);
    }

    #[test]
    fn test_encode_state_stats_display() {
        let stats = kernels::EncodeStateStats {
            num_encodings: 5000,
            total_kernel_time_us: 75000.0,
            avg_kernel_time_us: 15.0,
            peak_memory_bytes: 5 * 1024 * 1024, // 5 MB
            speedup_vs_cpu: 12.5,
        };
        let display = format!("{}", stats);
        assert!(display.contains("5000 encodings"));
        assert!(display.contains("15.00"));
        assert!(display.contains("12.5×"));
    }

    #[test]
    fn test_encode_state_placeholder() {
        // Test placeholder function exists
        let config = kernels::EncodeStateConfig::for_hybrid_trainer();
        let loss_history = vec![1.0; 1000];
        let grad_history = vec![0.5; 1000];
        let lr_history = vec![0.001; 1000];

        let features = kernels::encode_state(
            &loss_history,
            &grad_history,
            &lr_history,
            &config,
        );
        assert!(features.is_empty()); // Placeholder returns empty
    }

    #[test]
    fn test_encode_state_memory_requirements() {
        let config = kernels::EncodeStateConfig::for_hybrid_trainer();

        // Memory for history buffers (3 buffers: loss, grad, lr)
        let history_memory = 3 * config.history_len;
        // Memory for features output
        let features_memory = config.num_features;

        let total_memory_mb = ((history_memory + features_memory) * 4) as f32 / (1024.0 * 1024.0);

        // For default config: (3×1000 + 64) × 4 = 12256 bytes ≈ 0.012 MB
        assert!(
            total_memory_mb < 0.1,
            "Expected <0.1 MB for state encoding, got {} MB",
            total_memory_mb
        );
    }
}
