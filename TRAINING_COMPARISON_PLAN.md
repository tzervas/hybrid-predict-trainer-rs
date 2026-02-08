# Training Comparison & Evaluation Plan

**Goal:** Compare hybrid predictive training against traditional training with rigorous quality evaluation.

---

## Phase 7: HybridTrainer GPU Integration

### 7.1 Integrate GPU Rollout into RSSMLite

**File:** `src/dynamics.rs`

```rust
impl RSSMLite {
    /// GPU-accelerated predict_y_steps (when CUDA available)
    #[cfg(all(feature = "cuda", feature = "candle"))]
    fn predict_y_steps_gpu(&self, state: &TrainingState, y_steps: usize)
        -> (PhasePrediction, PredictionUncertainty) {
        // Convert weights to GPU format
        let gpu_weights = self.to_gpu_weights();
        let config = RssmRolloutConfig {
            ensemble_size: self.config.ensemble_size,
            hidden_dim: self.config.deterministic_dim,
            stochastic_dim: self.config.stochastic_dim,
            feature_dim: 64,
            y_steps,
        };

        // GPU rollout
        let results = rssm_rollout_gpu(&config, &gpu_weights,
            &self.latent_states, &features, state.loss)?;

        // Convert results back to prediction format
        self.aggregate_ensemble_predictions(results, y_steps)
    }

    /// CPU fallback
    fn predict_y_steps_cpu(&self, state: &TrainingState, y_steps: usize)
        -> (PhasePrediction, PredictionUncertainty) {
        // Existing CPU implementation
    }

    /// Auto-select GPU or CPU based on availability
    pub fn predict_y_steps(&self, state: &TrainingState, y_steps: usize)
        -> (PhasePrediction, PredictionUncertainty) {
        #[cfg(all(feature = "cuda", feature = "candle"))]
        {
            if self.use_gpu {
                return self.predict_y_steps_gpu(state, y_steps);
            }
        }

        self.predict_y_steps_cpu(state, y_steps)
    }
}
```

**Status:** ⏳ In Progress

---

## Phase 8: GPU Optimizations

### 8.1 Ensemble Batching (5× Speedup)

**Challenge:** Each ensemble member has different GRU weights.

**Solution:** Batch-aware GRU kernel

```rust
// New kernel: gru_forward_multi_weight
pub fn gru_forward_multi_weight<B: Backend>(
    configs: &[GruKernelConfig],
    weights: &[GpuGruWeights],  // Different weights per batch element
    hiddens: &[Vec<f32>],
    inputs: &[Vec<f32>],
    device: &B::Device,
) -> HybridResult<Vec<Vec<f32>>> {
    // Process each (hidden, input, weight) tuple in parallel
    // Use tensor stacking to process all 5 ensemble members at once
}
```

**Expected:** 5× speedup (1 GPU call instead of 5)

### 8.2 Persistent GPU State (2× Speedup)

**Current:** Transfer CPU ↔ GPU every step

**Optimized:** Keep state on GPU

```rust
pub struct GpuRssmState<B: Backend> {
    latent_states: Tensor<B, 2>,  // [ensemble_size, hidden_dim] on GPU
    features: Tensor<B, 1>,        // [feature_dim] on GPU
    device: B::Device,
}

impl<B: Backend> GpuRssmState<B> {
    pub fn rollout_in_place(&mut self, y_steps: usize) -> HybridResult<Vec<f32>> {
        // All ops happen on GPU, no transfers until final result
    }
}
```

**Expected:** 2× speedup (eliminate transfer overhead)

### 8.3 Kernel Fusion (1.5× Speedup)

**Current:** 3 separate kernels per step
- GRU forward
- Loss decode
- Feature update

**Fused:** Single kernel for all ops

```rust
#[cube]
fn fused_rollout_step<F: Float>(
    hidden: &Tensor<F>,
    features: &Tensor<F>,
    weights: &GruWeights<F>,
    loss_head: &Tensor<F>,
) -> (Tensor<F>, F) {  // (new_hidden, loss)
    // GRU forward
    let new_hidden = gru_forward_fused(hidden, features, weights);

    // Decode loss (inline)
    let combined = concat(new_hidden, sample_stochastic(new_hidden));
    let loss = dot(combined, loss_head);

    // Update features (inline)
    features[0] = loss;

    (new_hidden, loss)
}
```

**Expected:** 1.5× speedup (reduce kernel launches)

**Combined Optimization Target:** 5 × 2 × 1.5 = **15× faster**

**Status:** ⏳ Planned

---

## Phase 9: Training Comparison Framework

### 9.1 Dataset Management

**Location:** `/data/datasets/` (homelab volume)

**Structure:**
```
/data/datasets/
├── mnist/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── cifar10/
│   ├── train/
│   └── test/
├── wikitext/
│   ├── train.txt
│   ├── valid.txt
│   └── test.txt
└── openwebtext/
    └── ...
```

**Dataset Loader:**
```rust
pub struct DatasetConfig {
    pub root: PathBuf,  // Default: /data/datasets
    pub name: String,   // mnist, cifar10, wikitext, etc.
    pub split: DatasetSplit,
}

pub enum DatasetSplit {
    Train,
    Validation,
    Test,
}

impl DatasetConfig {
    pub fn mnist_train() -> Self {
        Self {
            root: PathBuf::from("/data/datasets"),
            name: "mnist".to_string(),
            split: DatasetSplit::Train,
        }
    }
}
```

### 9.2 Training Parity Verification

**Ensure identical training conditions:**

```rust
pub struct TrainingRun {
    pub model_arch: ModelArchitecture,
    pub dataset: DatasetConfig,
    pub hyperparams: Hyperparameters,
    pub seed: u64,
    pub device: Device,
}

pub struct Hyperparameters {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub optimizer: OptimizerType,
    pub epochs: usize,
    pub weight_decay: f32,
}
```

**Verification:**
- ✅ Same model architecture
- ✅ Same dataset (exact same data points)
- ✅ Same hyperparameters
- ✅ Same random seed
- ✅ Same device (GPU/CPU)

**Only difference:** Training strategy (hybrid vs traditional)

### 9.3 Comparison Experiments

#### Experiment 1: MNIST Classification

**Model:** SimpleCNN (~100K params)
```
Conv2d(1, 16, 5x5) -> ReLU -> MaxPool
Conv2d(16, 32, 5x5) -> ReLU -> MaxPool
Linear(32*5*5, 128) -> ReLU
Linear(128, 10)
```

**Hyperparameters:**
- Batch size: 128
- Learning rate: 0.001
- Optimizer: Adam (β1=0.9, β2=0.999)
- Epochs: 10
- Dataset: MNIST (60K train, 10K test)

**Comparison:**
```rust
// Traditional training
let traditional_metrics = train_traditional(
    model.clone(),
    dataset,
    hyperparams,
    seed=42
);

// Hybrid training
let hybrid_config = HybridTrainerConfig {
    warmup_steps: 100,
    min_full_steps: 50,
    prediction_horizon: 75,
    correction_interval: Some(15),
    confidence_threshold: 0.60,
    divergence_sigma: 2.2,
    ..Default::default()
};

let hybrid_metrics = train_hybrid(
    model.clone(),
    dataset,
    hyperparams,
    hybrid_config,
    seed=42
);

// Compare
compare_training_runs(traditional_metrics, hybrid_metrics);
```

**Metrics Tracked:**
- **Quality:**
  - Final test accuracy
  - Training loss curve
  - Test loss curve
  - Per-epoch accuracy

- **Efficiency:**
  - Total training time
  - Time per epoch
  - Backward passes executed
  - GPU utilization

- **Stability:**
  - Loss variance
  - Gradient norm variance
  - Divergence count
  - Recovery count

#### Experiment 2: GPT-2 Small Language Modeling

**Model:** GPT-2 Small (124M params)
- 12 layers
- 768 hidden dim
- 12 attention heads

**Hyperparameters:**
- Batch size: 8
- Sequence length: 512
- Learning rate: 6e-4
- Optimizer: AdamW
- Warmup steps: 2000
- Dataset: WikiText-103

**Comparison Metrics:**
- Perplexity (train/val/test)
- Training throughput (tokens/sec)
- Memory usage (peak VRAM)
- Total training time
- Final model quality (BLEU, ROUGE)

### 9.4 Quality Evaluation Framework

```rust
pub struct TrainingMetrics {
    pub experiment_id: String,
    pub training_type: TrainingType,

    // Quality metrics
    pub final_train_loss: f32,
    pub final_test_loss: f32,
    pub final_test_accuracy: f32,
    pub loss_curve: Vec<f32>,
    pub accuracy_curve: Vec<f32>,

    // Efficiency metrics
    pub total_time_sec: f64,
    pub time_per_epoch_sec: f64,
    pub backward_passes: usize,
    pub forward_passes: usize,
    pub speedup_vs_baseline: f32,

    // Stability metrics
    pub loss_variance: f32,
    pub gradient_norm_mean: f32,
    pub gradient_norm_std: f32,
    pub divergence_count: usize,
    pub recovery_count: usize,

    // Resource usage
    pub peak_memory_mb: f64,
    pub avg_gpu_utilization: f32,
}

pub enum TrainingType {
    Traditional,
    Hybrid {
        prediction_horizon: usize,
        correction_interval: Option<usize>,
    },
}

impl TrainingMetrics {
    pub fn save_to_json(&self, path: &Path) -> Result<()> {
        // Save metrics for analysis
    }

    pub fn compare(&self, other: &Self) -> ComparisonReport {
        ComparisonReport {
            quality_delta: self.final_test_accuracy - other.final_test_accuracy,
            speedup: other.total_time_sec / self.total_time_sec,
            efficiency_gain: (other.backward_passes - self.backward_passes) as f32
                / other.backward_passes as f32,
        }
    }
}
```

### 9.5 Automated Comparison Pipeline

```rust
pub async fn run_comparison_experiment(
    experiment: ExperimentConfig,
) -> Result<ComparisonReport> {
    println!("🔬 Running comparison experiment: {}", experiment.name);

    // 1. Prepare dataset
    let dataset = load_dataset(&experiment.dataset_config).await?;
    println!("✅ Dataset loaded: {} samples", dataset.len());

    // 2. Run traditional training
    println!("🏃 Running traditional training...");
    let traditional_start = Instant::now();
    let traditional_metrics = train_traditional(
        &experiment.model_arch,
        &dataset,
        &experiment.hyperparams,
        experiment.seed,
    ).await?;
    let traditional_time = traditional_start.elapsed();
    println!("✅ Traditional training complete: {:.2}s", traditional_time.as_secs_f64());

    // 3. Run hybrid training
    println!("🏃 Running hybrid training...");
    let hybrid_start = Instant::now();
    let hybrid_metrics = train_hybrid(
        &experiment.model_arch,
        &dataset,
        &experiment.hyperparams,
        &experiment.hybrid_config,
        experiment.seed,
    ).await?;
    let hybrid_time = hybrid_start.elapsed();
    println!("✅ Hybrid training complete: {:.2}s", hybrid_time.as_secs_f64());

    // 4. Compare results
    let report = traditional_metrics.compare(&hybrid_metrics);

    // 5. Generate visualization
    generate_comparison_plots(&traditional_metrics, &hybrid_metrics, &experiment)?;

    // 6. Save results
    traditional_metrics.save_to_json(&format!("results/{}_traditional.json", experiment.name))?;
    hybrid_metrics.save_to_json(&format!("results/{}_hybrid.json", experiment.name))?;
    report.save_to_json(&format!("results/{}_comparison.json", experiment.name))?;

    println!("\n📊 Comparison Results:");
    println!("  Quality delta: {:+.4}%", report.quality_delta * 100.0);
    println!("  Speedup: {:.2}×", report.speedup);
    println!("  Efficiency gain: {:.1}%", report.efficiency_gain * 100.0);

    Ok(report)
}
```

---

## Implementation Checklist

### Phase 7: Integration ⏳
- [ ] Add GPU path to `RSSMLite::predict_y_steps()`
- [ ] Add weight conversion: `RSSMLite::to_gpu_weights()`
- [ ] Add ensemble aggregation helper
- [ ] Test GPU vs CPU equivalence
- [ ] Add `use_gpu` flag to config

### Phase 8: Optimizations ⏳
- [ ] Implement ensemble batching (5×)
- [ ] Implement persistent GPU state (2×)
- [ ] Implement kernel fusion (1.5×)
- [ ] Benchmark optimized version
- [ ] Validate 15× combined speedup

### Phase 9: Training Comparison ⏳
- [ ] Set up dataset management (`/data/datasets/`)
- [ ] Implement `DatasetConfig` and loaders
- [ ] Create `TrainingMetrics` framework
- [ ] Implement MNIST comparison experiment
- [ ] Implement GPT-2 comparison experiment
- [ ] Create automated comparison pipeline
- [ ] Generate comparison visualizations
- [ ] Write comparison report

---

## Success Criteria

**Quality Parity:**
- ✅ Hybrid training achieves ≥99.9% of traditional quality
- ✅ Test accuracy within 0.1% of baseline
- ✅ Loss curves converge to same final value

**Efficiency Gains:**
- ✅ Hybrid training ≥2× faster than traditional
- ✅ 50-80% reduction in backward passes
- ✅ GPU utilization ≥70%

**Stability:**
- ✅ Zero catastrophic divergences
- ✅ Automatic recovery from minor divergences
- ✅ Variance ≤ traditional training

**Reproducibility:**
- ✅ Same seed → same results (within float tolerance)
- ✅ Dataset checksum verification
- ✅ All hyperparameters logged

---

## Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 7 | GPU integration | 2-3 hours |
| Phase 8 | Optimizations | 4-6 hours |
| Phase 9 | Comparison framework | 3-4 hours |
| **Total** | **End-to-end** | **9-13 hours** |

---

## Next Steps

1. Complete Phase 7 GPU integration
2. Implement Phase 8 optimizations
3. Set up `/data/datasets/` structure
4. Run MNIST comparison experiment
5. Analyze results and iterate

**Ready to proceed?** ✅
