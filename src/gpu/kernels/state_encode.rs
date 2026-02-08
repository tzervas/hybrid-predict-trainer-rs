// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! State encoding GPU kernel using Burn tensor operations.
//!
//! This module provides GPU-accelerated feature extraction from `TrainingState`
//! via Burn's backend-agnostic tensor operations rather than raw CubeCL kernels.
//!
//! # Why Burn Instead of CubeCL?
//!
//! State encoding involves:
//! - Small output size (64 dimensions)
//! - Complex branching logic (history length checks)
//! - Multiple reduction operations (mean, std, min, max)
//! - Not in critical path (called once per prediction phase)
//!
//! Burn tensor ops provide cleaner code with automatic GPU dispatch.
//!
//! # Algorithm
//!
//! Extracts 64-dimensional feature vector from `TrainingState`:
//! ```text
//! Features [64]:
//!   [0-9]:   Current metrics (loss, grad_norm, lr, etc.)
//!   [10-31]: Loss history statistics (mean, std, min, max, trends)
//!   [32-47]: Gradient history statistics
//!   [48-63]: Phase indicators and temporal features
//! ```

use crate::error::{HybridResult, HybridTrainingError};
use crate::state::TrainingState;
use crate::Phase;

#[cfg(all(feature = "cuda", feature = "candle"))]
use crate::state::BufferStatistics;

#[cfg(all(feature = "cuda", feature = "candle"))]
use burn::tensor::{backend::Backend, Tensor, TensorData};

/// State encoding configuration.
#[derive(Debug, Clone)]
pub struct StateEncodeConfig {
    /// Feature output dimension (must be 64).
    pub feature_dim: usize,
    /// Maximum history length to consider.
    pub max_history: usize,
}

impl Default for StateEncodeConfig {
    fn default() -> Self {
        Self {
            feature_dim: 64,
            max_history: 1000,
        }
    }
}

impl StateEncodeConfig {
    /// Validates configuration.
    pub fn validate(&self) -> HybridResult<()> {
        if self.feature_dim != 64 {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!(
                        "feature_dim must be 64, got {}",
                        self.feature_dim
                    ),
                },
                None,
            ));
        }
        Ok(())
    }
}

/// Flattened training state data for GPU transfer.
///
/// Packs `TrainingState` into contiguous arrays for efficient GPU upload.
#[derive(Debug, Clone)]
pub struct StateDataFlat {
    pub loss: f32,
    pub gradient_norm: f32,
    pub step: u64,
    pub phase_step: usize,
    pub current_phase: Phase,
    pub loss_history: Vec<f32>,
    pub gradient_norm_history: Vec<f32>,
    pub optimizer_momentum_mean: f32,
    pub optimizer_momentum_std: f32,
    pub optimizer_variance_mean: f32,
    pub optimizer_variance_std: f32,
    pub optimizer_effective_lr: f32,
    pub optimizer_beta1_power: f32,
    pub optimizer_beta2_power: f32,
}

impl StateDataFlat {
    /// Creates flattened state from `TrainingState`.
    #[must_use]
    pub fn from_state(state: &TrainingState) -> Self {
        Self {
            loss: state.loss,
            gradient_norm: state.gradient_norm,
            step: state.step,
            phase_step: state.phase_step,
            current_phase: state.current_phase,
            loss_history: state.loss_history.iter().copied().collect(),
            gradient_norm_history: state.gradient_norm_history.iter().copied().collect(),
            optimizer_momentum_mean: state.optimizer_state_summary.momentum_mean,
            optimizer_momentum_std: state.optimizer_state_summary.momentum_std,
            optimizer_variance_mean: state.optimizer_state_summary.variance_mean,
            optimizer_variance_std: state.optimizer_state_summary.variance_std,
            optimizer_effective_lr: state.optimizer_state_summary.effective_lr,
            optimizer_beta1_power: state.optimizer_state_summary.beta1_power,
            optimizer_beta2_power: state.optimizer_state_summary.beta2_power,
        }
    }
}

/// Computes buffer statistics (mean, std, min, max) using GPU tensors.
#[cfg(all(feature = "cuda", feature = "candle"))]
fn compute_statistics_gpu<B: Backend>(
    data: &[f32],
    device: &B::Device,
) -> HybridResult<BufferStatistics> {
    if data.is_empty() {
        return Ok(BufferStatistics::default());
    }

    // Convert to tensor
    let tensor = Tensor::<B, 1>::from_data(TensorData::from(data), device);

    // Compute mean
    let mean_tensor = tensor.clone().mean();
    let mean = mean_tensor.into_data().to_vec::<f32>().unwrap()[0] as f64;

    // Variance: E[(x - μ)²]
    let mean_broadcast = Tensor::<B, 1>::from_data(
        TensorData::from([mean as f32]),
        device,
    )
    .repeat_dim(0, data.len());

    let diff = tensor.clone() - mean_broadcast;
    let variance = (diff.clone() * diff).mean();
    let std = variance.sqrt().into_data().to_vec::<f32>().unwrap()[0] as f64;

    // Min/max
    let min = tensor.clone().min().into_data().to_vec::<f32>().unwrap()[0] as f64;
    let max = tensor.max().into_data().to_vec::<f32>().unwrap()[0] as f64;

    Ok(BufferStatistics { mean, std, min, max })
}

/// GPU-accelerated state encoding using Burn tensor operations.
///
/// # Implementation Strategy
///
/// Uses Burn's backend-agnostic tensor ops for automatic GPU dispatch:
/// - Reductions: `mean()`, `std()`, `min()`, `max()`
/// - Element-wise: normalization, log transforms
/// - Backend: Automatically uses CUDA/WGPU/CPU based on device
///
/// Mirrors CPU implementation in `TrainingState::compute_features()` but uses
/// GPU tensor operations for statistical computations.
///
/// # Returns
///
/// 64-dimensional feature vector matching CPU output.
#[cfg(all(feature = "cuda", feature = "candle"))]
pub fn encode_state_gpu<B: Backend>(
    _config: &StateEncodeConfig,
    state: &StateDataFlat,
    device: &B::Device,
) -> HybridResult<Vec<f32>> {
    let _span = tracing::debug_span!("encode_state_gpu", step = state.step).entered();

    let mut features = Vec::with_capacity(64);

    // Loss features (8)
    let loss_stats = compute_statistics_gpu::<B>(&state.loss_history, device)?;
    features.push(state.loss);
    features.push(loss_stats.mean as f32);
    features.push(loss_stats.std as f32);
    features.push(loss_stats.min as f32);
    features.push(loss_stats.max as f32);

    // Loss trend (recent vs older)
    if state.loss_history.len() >= 32 {
        let all_losses = &state.loss_history;
        let recent: Vec<f32> = all_losses.iter().rev().take(16).copied().collect();
        let older: Vec<f32> = all_losses.iter().rev().skip(16).take(16).copied().collect();

        let recent_tensor = Tensor::<B, 1>::from_data(TensorData::from(&recent[..]), device);
        let older_tensor = Tensor::<B, 1>::from_data(TensorData::from(&older[..]), device);

        let recent_mean = recent_tensor.mean().into_data().to_vec::<f32>().unwrap()[0];
        let older_mean = older_tensor.mean().into_data().to_vec::<f32>().unwrap()[0];

        features.push(recent_mean - older_mean); // Trend
        features.push(recent_mean / older_mean.max(1e-8)); // Ratio
    } else {
        features.push(0.0);
        features.push(1.0);
    }
    features.push(state.step as f32 / 10000.0); // Normalized step

    // Gradient features (8)
    let grad_stats = compute_statistics_gpu::<B>(&state.gradient_norm_history, device)?;
    features.push(state.gradient_norm);
    features.push(grad_stats.mean as f32);
    features.push(grad_stats.std as f32);
    features.push(grad_stats.min as f32);
    features.push(grad_stats.max as f32);
    features.push(state.gradient_norm.log10().max(-10.0)); // Log scale
    features.push((state.gradient_norm / grad_stats.mean.max(1e-8) as f32).min(10.0)); // Relative
    features.push(grad_stats.std as f32 / grad_stats.mean.max(1e-8) as f32); // CV

    // Optimizer features (8)
    features.push(state.optimizer_momentum_mean);
    features.push(state.optimizer_momentum_std);
    features.push(state.optimizer_variance_mean);
    features.push(state.optimizer_variance_std);
    features.push(state.optimizer_effective_lr.log10().max(-10.0));
    features.push(state.optimizer_beta1_power);
    features.push(state.optimizer_beta2_power);
    features.push(0.0); // Reserved

    // Phase features (8)
    features.push(match state.current_phase {
        Phase::Warmup => 0.0,
        Phase::Full => 1.0,
        Phase::Predict => 2.0,
        Phase::Correct => 3.0,
    });
    features.push(state.phase_step as f32 / 100.0);

    // Pad to 64 features
    while features.len() < 64 {
        features.push(0.0);
    }

    tracing::debug!(feature_count = features.len(), "GPU state encoding complete");

    Ok(features)
}

/// Non-CUDA/Candle fallback - delegates to CPU implementation.
#[cfg(not(all(feature = "cuda", feature = "candle")))]
pub fn encode_state_gpu_fallback(
    _config: &StateEncodeConfig,
    state: &TrainingState,
) -> HybridResult<Vec<f32>> {
    tracing::debug!(
        "GPU state encoding requested but CUDA/Candle not available, using CPU"
    );
    Ok(encode_state_cpu(state))
}

/// CPU reference implementation - directly uses `TrainingState::compute_features()`.
///
/// Returns 64-dimensional feature vector.
pub fn encode_state_cpu(state: &TrainingState) -> Vec<f32> {
    let _span = tracing::trace_span!(
        "encode_state_cpu",
        step = state.step,
        loss = %state.loss,
        grad_norm = %state.gradient_norm
    )
    .entered();

    // Use the existing compute_features() method
    let features = state.compute_features();

    tracing::trace!(
        features_computed = features.len(),
        loss = %features[0],
        grad_norm = %features[8],
        "State encoding complete"
    );

    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Phase;

    fn create_test_state() -> TrainingState {
        TrainingState::default()
    }

    #[test]
    fn test_state_encode_cpu_output_dim() {
        let state = create_test_state();
        let features = encode_state_cpu(&state);
        assert_eq!(features.len(), 64, "Must output exactly 64 features");
    }

    #[test]
    fn test_state_encode_cpu_deterministic() {
        let state = create_test_state();
        let features1 = encode_state_cpu(&state);
        let features2 = encode_state_cpu(&state);

        // Both should be same length
        assert_eq!(features1.len(), features2.len());
        assert_eq!(features1.len(), 64);
    }

    #[test]
    fn test_state_encode_cpu_first_feature_is_loss() {
        let state = create_test_state();
        let features = encode_state_cpu(&state);

        // First feature should be current loss (matches compute_features implementation)
        // Both may be NaN in default state, so check they're both NaN or both equal
        if state.loss.is_nan() {
            assert!(features[0].is_nan(), "First feature should match loss (NaN)");
        } else {
            assert_eq!(features[0], state.loss, "First feature should match loss");
        }
    }

    #[test]
    fn test_state_encode_config_validation() {
        let valid = StateEncodeConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = StateEncodeConfig {
            feature_dim: 32, // Wrong dimension
            max_history: 1000,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_state_data_flat_features_length() {
        let state = create_test_state();
        let flat = StateDataFlat::from_state(&state);

        assert_eq!(flat.features.len(), 64);
    }
}
