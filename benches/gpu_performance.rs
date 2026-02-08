//! Comprehensive GPU performance benchmarks.
//!
//! Measures GPU vs CPU speedup for all critical operations:
//! - GRU forward (single & batched)
//! - RSSM multi-step rollout
//! - State encoding
//! - End-to-end prediction phase

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(all(feature = "cuda", feature = "candle"))]
use burn_cuda::{Cuda, CudaDevice};

use hybrid_predict_trainer_rs::gpu::kernels::gru::{gru_forward_cpu, GpuGruWeights, GruKernelConfig};

#[cfg(all(feature = "cuda", feature = "candle"))]
use hybrid_predict_trainer_rs::gpu::kernels::gru_batched::gru_forward_batched_gpu;

use hybrid_predict_trainer_rs::gpu::kernels::rssm_rollout::{
    rssm_rollout_cpu, RssmEnsembleWeights, RssmRolloutConfig,
};

#[cfg(all(feature = "cuda", feature = "candle"))]
use hybrid_predict_trainer_rs::gpu::kernels::rssm_rollout::rssm_rollout_gpu;

use hybrid_predict_trainer_rs::gpu::kernels::state_encode::encode_state_cpu;

#[cfg(all(feature = "cuda", feature = "candle"))]
use hybrid_predict_trainer_rs::gpu::kernels::state_encode::{encode_state_gpu, StateDataFlat, StateEncodeConfig};

use hybrid_predict_trainer_rs::state::TrainingState;

// ============================================================================
// Benchmark 1: GRU Forward Pass (Single Sample)
// ============================================================================

fn create_gru_weights(hidden_dim: usize, input_dim: usize) -> GpuGruWeights {
    GpuGruWeights {
        w_z: vec![0.01; hidden_dim * input_dim],
        u_z: vec![0.01; hidden_dim * hidden_dim],
        b_z: vec![0.0; hidden_dim],
        w_r: vec![0.01; hidden_dim * input_dim],
        u_r: vec![0.01; hidden_dim * hidden_dim],
        b_r: vec![0.0; hidden_dim],
        w_h: vec![0.01; hidden_dim * input_dim],
        u_h: vec![0.01; hidden_dim * hidden_dim],
        b_h: vec![0.0; hidden_dim],
        hidden_dim,
        input_dim,
    }
}

fn bench_gru_forward_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gru_forward_cpu");

    for &hidden_dim in &[64, 128, 256] {
        let input_dim = 64;
        let weights = create_gru_weights(hidden_dim, input_dim);
        let hidden = vec![0.5; hidden_dim];
        let input = vec![0.3; input_dim];

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    black_box(gru_forward_cpu(&weights, &hidden, &input));
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 2: GRU Forward Pass (Batched GPU)
// ============================================================================

#[cfg(all(feature = "cuda", feature = "candle"))]
fn bench_gru_forward_batched_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gru_forward_batched_gpu");
    let device = CudaDevice::default();

    for &batch_size in &[8, 16, 32, 64] {
        let hidden_dim = 256;
        let input_dim = 64;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size,
        };

        let weights = create_gru_weights(hidden_dim, input_dim);
        let hiddens: Vec<Vec<f32>> = (0..batch_size).map(|_| vec![0.5; hidden_dim]).collect();
        let inputs: Vec<Vec<f32>> = (0..batch_size).map(|_| vec![0.3; input_dim]).collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        gru_forward_batched_gpu::<Cuda>(&config, &weights, &hiddens, &inputs, &device)
                            .unwrap(),
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 3: RSSM Rollout (CPU)
// ============================================================================

fn create_rssm_weights(ensemble_size: usize, hidden_dim: usize, feature_dim: usize) -> RssmEnsembleWeights {
    let mut gru_weights = Vec::new();
    for _ in 0..ensemble_size {
        gru_weights.push(create_gru_weights(hidden_dim, feature_dim));
    }

    RssmEnsembleWeights {
        gru_weights,
        loss_head_weights: vec![0.01; hidden_dim * 2],
        delta_head_weights: vec![0.01; hidden_dim * 2 * 10],
    }
}

fn bench_rssm_rollout_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("rssm_rollout_cpu");

    for &y_steps in &[10, 25, 50, 75] {
        let config = RssmRolloutConfig {
            ensemble_size: 5,
            hidden_dim: 256,
            stochastic_dim: 256,
            feature_dim: 64,
            y_steps,
        };

        let weights = create_rssm_weights(config.ensemble_size, config.hidden_dim, config.feature_dim);
        let initial_latents: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|_| vec![0.5; config.hidden_dim])
            .collect();
        let features = vec![1.0; config.feature_dim];
        let initial_loss = 2.5;

        group.throughput(Throughput::Elements((y_steps * config.ensemble_size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(y_steps), &y_steps, |b, _| {
            b.iter(|| {
                black_box(rssm_rollout_cpu(
                    &config,
                    &weights,
                    &initial_latents,
                    &features,
                    initial_loss,
                ));
            });
        });
    }

    group.finish();
}

// ============================================================================
// Benchmark 4: RSSM Rollout (GPU)
// ============================================================================

#[cfg(all(feature = "cuda", feature = "candle"))]
fn bench_rssm_rollout_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("rssm_rollout_gpu");

    for &y_steps in &[10, 25, 50, 75] {
        let config = RssmRolloutConfig {
            ensemble_size: 5,
            hidden_dim: 256,
            stochastic_dim: 256,
            feature_dim: 64,
            y_steps,
        };

        let weights = create_rssm_weights(config.ensemble_size, config.hidden_dim, config.feature_dim);
        let initial_latents: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|_| vec![0.5; config.hidden_dim])
            .collect();
        let features = vec![1.0; config.feature_dim];
        let initial_loss = 2.5;

        group.throughput(Throughput::Elements((y_steps * config.ensemble_size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(y_steps), &y_steps, |b, _| {
            b.iter(|| {
                black_box(
                    rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss)
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

// ============================================================================
// Benchmark 5: State Encoding
// ============================================================================

fn bench_state_encoding_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_encoding_cpu");

    let state = TrainingState::default();

    group.bench_function("encode_64_dim", |b| {
        b.iter(|| {
            black_box(encode_state_cpu(&state));
        });
    });

    group.finish();
}

#[cfg(all(feature = "cuda", feature = "candle"))]
fn bench_state_encoding_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_encoding_gpu");

    let state = TrainingState::default();
    let flat = StateDataFlat::from_state(&state);
    let config = StateEncodeConfig::default();
    let device = CudaDevice::default();

    group.bench_function("encode_64_dim", |b| {
        b.iter(|| {
            black_box(encode_state_gpu::<Cuda>(&config, &flat, &device).unwrap());
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

#[cfg(all(feature = "cuda", feature = "candle"))]
criterion_group!(
    gpu_benches,
    bench_gru_forward_cpu,
    bench_gru_forward_batched_gpu,
    bench_rssm_rollout_cpu,
    bench_rssm_rollout_gpu,
    bench_state_encoding_cpu,
    bench_state_encoding_gpu,
);

#[cfg(not(all(feature = "cuda", feature = "candle")))]
criterion_group!(
    gpu_benches,
    bench_gru_forward_cpu,
    bench_rssm_rollout_cpu,
    bench_state_encoding_cpu,
);

criterion_main!(gpu_benches);
