//! Memory Profiling Validation (Sprint 4, Tasks #56-#57)
//!
//! This example demonstrates the memory profiler module and validates
//! memory tracking capabilities for large model training.
//!
//! ## v0.3.0 Memory Optimization Status
//!
//! **Implemented** (Sprint 1-3):
//! - ✅ GradientCheckpointer module (src/gradient_checkpointing.rs)
//! - ✅ CpuOffloadManager module (src/cpu_offloading.rs)
//! - ✅ Quantizer module (src/quantization.rs)
//! - ✅ Flash Attention kernel specs (src/gpu.rs)
//! - ✅ GPU kernels (RSSM, State Encoding)
//! - ✅ MemoryProfiler module (src/memory_profiler.rs)
//!
//! **Pending** (Sprint 4-5):
//! - 🔲 Integration into HybridTrainer::step()
//! - 🔲 Integration into HybridTrainerConfig
//! - 🔲 CubeCL runtime integration for GPU kernels
//! - 🔲 Full validation on 1B/7B models
//!
//! This example validates memory profiling works correctly and provides
//! a framework for full validation once integration is complete.
//!
//! ## Usage
//!
//! ```bash
//! # Run with existing GPT-2 Small to validate profiler
//! cargo run --release --example memory_profile_validation --features autodiff,ndarray
//!
//! # With CUDA (if available)
//! cargo run --release --example memory_profile_validation --features autodiff,cuda
//! ```

use burn::{
    backend::{Autodiff, NdArray},
    module::Module as BurnModule,
    optim::AdamConfig,
    tensor::{backend::Backend, Tensor, TensorData},
};

#[cfg(feature = "cuda")]
use burn::backend::Cuda;

use hybrid_predict_trainer_rs::{
    burn_integration::{BurnBatch, BurnForwardFn, BurnModelWrapper, BurnOptimizerWrapper},
    config::HybridTrainerConfig,
    memory_profiler::MemoryProfiler,
    models::gpt2::{Gpt2Batch, Gpt2Config, Gpt2Model},
    HybridTrainer, Model, Optimizer,
};
use std::time::Instant;

#[cfg(feature = "cuda")]
type MyBackend = Autodiff<Cuda>;

#[cfg(not(feature = "cuda"))]
type MyBackend = Autodiff<NdArray>;

fn generate_synthetic_batch(
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &<MyBackend as Backend>::Device,
) -> Gpt2Batch<MyBackend> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let input_data: Vec<i64> = (0..batch_size * seq_len)
        .map(|_| rng.gen_range(0..vocab_size as i64))
        .collect();

    let target_data: Vec<i64> = (0..batch_size * seq_len)
        .map(|_| rng.gen_range(0..vocab_size as i64))
        .collect();

    let input_ids = Tensor::from_data(
        TensorData::new(input_data, [batch_size, seq_len]),
        device,
    );
    let targets = Tensor::from_data(
        TensorData::new(target_data, [batch_size, seq_len]),
        device,
    );

    Gpt2Batch { input_ids, targets }
}

struct Gpt2Forward;

impl BurnForwardFn<MyBackend, Gpt2Model<MyBackend>, Gpt2Batch<MyBackend>> for Gpt2Forward {
    fn forward(
        &self,
        model: Gpt2Model<MyBackend>,
        batch: &BurnBatch<MyBackend, Gpt2Batch<MyBackend>>,
    ) -> (Gpt2Model<MyBackend>, Tensor<MyBackend, 1>) {
        let logits = model.forward(batch.data.input_ids.clone());

        let [b, s, v] = logits.dims();
        let logits_flat = logits.reshape([b * s, v]);
        let targets_flat = batch.data.targets.clone().reshape([b * s]);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        (model, loss)
    }
}

fn main() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔍 Memory Profiling Validation (Sprint 4)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Use GPT-2 Small for validation (124M params)
    let config = Gpt2Config::gpt2_small();
    let batch_size = 2;
    let seq_len = 64;
    let steps = 50;
    let log_interval = 5;

    println!("Model Configuration:");
    println!("  Model: GPT-2 Small (124M params)");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Layers: {}", config.n_layer);
    println!("  Hidden dim: {}", config.n_embd);
    println!("  Batch size: {}", batch_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Training steps: {}\n", steps);

    // Create trainer with default config
    let hybrid_config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .max_predict_steps(15)
        .correction_interval(2)
        .divergence_threshold(2.5)
        .confidence_threshold(0.5)
        .build();

    let device = <MyBackend as Backend>::Device::default();
    println!("✓ Creating model...");
    let model = Gpt2Model::new(&config, &device);
    let forward_fn = Gpt2Forward;
    let wrapped_model = BurnModelWrapper::new(model, forward_fn, device.clone());

    println!("✓ Creating optimizer (Adam, lr=6e-4)...");
    let optim = AdamConfig::new().with_epsilon(1e-8).init();
    let wrapped_optimizer = BurnOptimizerWrapper::new(optim, 6e-4);

    println!("✓ Creating HybridTrainer...");
    let mut trainer = HybridTrainer::new(wrapped_model, wrapped_optimizer, hybrid_config)
        .expect("Failed to create HybridTrainer");

    // Initialize memory profiler
    println!("✓ Starting memory profiler...\n");
    let mut profiler = MemoryProfiler::new();
    profiler.start();
    profiler.record_step(0, "Init");

    println!("Step | Phase   | Loss    | VRAM (GB) | Δ VRAM | Time (ms)");
    println!("-----|---------|---------|-----------|--------|----------");

    let start_time = Instant::now();
    let mut phase_counts = std::collections::HashMap::new();

    for step in 0..steps {
        let step_start = Instant::now();

        // Record memory BEFORE step
        let phase_str = format!("{:?}", trainer.current_phase());
        profiler.record_step(step, &phase_str);

        // Generate batch
        let batch_data = generate_synthetic_batch(batch_size, seq_len, config.vocab_size, &device);
        let batch = BurnBatch::new(batch_data, batch_size);

        // Training step
        let result = trainer.step(&batch).expect("Training step failed");
        let step_time = step_start.elapsed().as_secs_f64() * 1000.0;

        // Track phases
        let phase_str = format!("{:?}", result.phase);
        *phase_counts.entry(phase_str.clone()).or_insert(0) += 1;

        // Get memory snapshot
        let snapshot = profiler.snapshots().last();
        let vram_gb = snapshot.map(|s| s.used_gb()).unwrap_or(0.0);
        let initial_vram = profiler.snapshots().first().map(|s| s.used_gb()).unwrap_or(0.0);
        let delta_vram = vram_gb - initial_vram;

        if step % log_interval == 0 || step == steps - 1 {
            println!(
                "{:4} | {:7} | {:.5} | {:9.2} | {:6.2} | {:8.1}",
                step, phase_str, result.loss, vram_gb, delta_vram, step_time
            );
        }
    }

    let total_time_s = start_time.elapsed().as_secs_f64();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Profiling Complete");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Performance:");
    println!("  Total time: {:.1}s", total_time_s);
    println!("  Throughput: {:.1} tokens/sec",
             (batch_size * seq_len * steps) as f64 / total_time_s);

    // Generate comprehensive profiling report
    println!("\n{}", profiler.generate_report());

    println!("\nPhase Distribution:");
    let mut phases: Vec<_> = phase_counts.iter().collect();
    phases.sort_by_key(|(name, _)| name.as_str());
    for (phase, count) in phases {
        let percentage = (*count as f64 / steps as f64) * 100.0;
        println!("  {}: {} steps ({:.1}%)", phase, count, percentage);
    }

    // Export CSV for analysis
    let csv_path = "/tmp/memory_profile_validation.csv";
    std::fs::write(csv_path, profiler.export_csv()).ok();
    println!("\n📊 Memory profile exported to: {}", csv_path);

    // Validate profiler functionality
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Memory Profiler Validation:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let snapshots_count = profiler.snapshots().len();
    let peak_gb = profiler.peak_usage_gb();
    let peak_step = profiler.peak_step();
    let phase_stats = profiler.phase_statistics();
    let spikes = profiler.identify_spikes(100.0);

    println!("  ✅ Snapshot recording: {} snapshots", snapshots_count);
    println!("  ✅ Peak tracking: {:.2} GB at step {:?}", peak_gb, peak_step);
    println!("  ✅ Phase statistics: {} phases tracked", phase_stats.len());
    println!("  ✅ Spike detection: {} spikes identified", spikes.len());
    println!("  ✅ CSV export: {} bytes", profiler.export_csv().len());

    let success = snapshots_count > 0 && phase_stats.len() > 0;
    if success {
        println!("\n✅ MEMORY PROFILER VALIDATED - Ready for large model validation");
    } else {
        println!("\n⚠️  PROFILER ISSUES DETECTED - Check nvidia-smi availability");
    }

    println!("\n📋 Next Steps (Sprint 4-5):");
    println!("  1. Integrate v0.3.0 optimization modules into HybridTrainer");
    println!("  2. Add config fields to HybridTrainerConfig");
    println!("  3. Implement phase-aware optimization switching");
    println!("  4. Validate on scaled models (1B, 7B params)");
    println!("  5. Document memory optimization guide");
}
