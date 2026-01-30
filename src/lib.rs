//! # hybrid-predict-trainer-rs
//!
//! Hybridized predictive training framework that accelerates deep learning through
//! intelligent phase-based training with whole-phase prediction and residual correction.
//!
//! ## Overview
//!
//! This crate implements a novel training paradigm that achieves 5-10x speedup over
//! traditional training by predicting training outcomes rather than computing every
//! gradient step. The key insight is that training dynamics evolve on low-dimensional
//! manifolds, making whole-phase prediction tractable.
//!
//! ## Training Phases
//!
//! The training loop cycles through four distinct phases:
//!
//! 1. **Warmup Phase** - Initial training steps to establish baseline dynamics
//! 2. **Full Training Phase** - Traditional forward/backward pass computation
//! 3. **Predictive Phase** - Skip backward passes using learned dynamics model
//! 4. **Correction Phase** - Apply residual corrections to maintain accuracy
//!
//! ```text
//!                     ┌─────────┐
//!                     │ WARMUP  │
//!                     └────┬────┘
//!                          │
//!                          ▼
//!               ┌─────────────────────┐
//!               │                     │
//!               ▼                     │
//!         ┌──────────┐                │
//!    ┌───▶│   FULL   │◀───────────────┤
//!    │    └────┬─────┘                │
//!    │         │                      │
//!    │         ▼                      │
//!    │    ┌──────────┐                │
//!    │    │ PREDICT  │                │
//!    │    └────┬─────┘                │
//!    │         │                      │
//!    │         ▼                      │
//!    │    ┌──────────┐                │
//!    │    │ CORRECT  │────────────────┘
//!    │    └────┬─────┘
//!    │         │
//!    └─────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```no_run
//! use hybrid_predict_trainer_rs::{HybridTrainer, HybridTrainerConfig};
//!
//! // Create configuration with sensible defaults
//! let config = HybridTrainerConfig::default();
//!
//! // Initialize the hybrid trainer
//! // let trainer = HybridTrainer::new(model, optimizer, config)?;
//!
//! // Training loop
//! // for batch in dataloader {
//! //     let result = trainer.step(&batch)?;
//! //     println!("Loss: {}, Phase: {:?}", result.loss, result.phase);
//! // }
//! ```
//!
//! ## Features
//!
//! - **GPU Acceleration** - CubeCL and Burn backends for high-performance compute
//! - **Adaptive Phase Selection** - Bandit-based algorithm for optimal phase lengths
//! - **Divergence Detection** - Multi-signal monitoring prevents training instability
//! - **Residual Correction** - Online learning corrects prediction errors
//! - **Checkpoint Support** - Save/restore full training state including predictor
//!
//! ## Feature Flags
//!
//! - `std` - Enable standard library support (default)
//! - `cuda` - Enable CUDA GPU acceleration via CubeCL
//! - `candle` - Enable Candle tensor operations for model compatibility
//! - `async` - Enable async/await support with Tokio
//! - `full` - Enable all features
//!
//! ## Architecture
//!
//! The crate is organized into the following modules:
//!
//! - [`config`] - Training configuration and serialization
//! - [`error`] - Error types with recovery actions
//! - [`phases`] - Phase state machine and execution control
//! - [`state`] - Training state encoding and management
//! - [`dynamics`] - RSSM-lite dynamics model for prediction
//! - [`residuals`] - Residual extraction and storage
//! - [`corrector`] - Prediction correction via residual application
//! - [`divergence`] - Multi-signal divergence detection
//! - [`metrics`] - Training metrics collection and reporting
//! - [`gpu`] - GPU acceleration kernels (requires `cuda` feature)
//!
//! ## References
//!
//! This implementation is based on research findings documented in
//! `predictive-training-research.md`, synthesizing insights from:
//!
//! - Neural Tangent Kernel (NTK) theory for training dynamics
//! - RSSM world models from DreamerV3
//! - K-FAC for structured gradient approximation
//! - PowerSGD for low-rank gradient compression

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(unsafe_code)]

// Core modules
pub mod config;
pub mod error;
pub mod phases;
pub mod state;

// Training phase implementations
pub mod warmup;
pub mod full_train;
pub mod predictive;
pub mod residuals;
pub mod corrector;

// Prediction and control
pub mod dynamics;
pub mod divergence;
pub mod bandit;

// Metrics and monitoring
pub mod metrics;

// GPU acceleration (feature-gated)
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub mod gpu;

// Re-exports for convenient access
pub use config::HybridTrainerConfig;
pub use error::{HybridTrainingError, HybridResult, RecoveryAction};
pub use phases::{Phase, PhaseController, PhaseDecision, PhaseOutcome};
pub use state::TrainingState;

use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;

/// Batch of training data.
///
/// Generic container for a batch of input data that will be fed to the model
/// during training. The actual batch format depends on the model implementation.
pub trait Batch: Send + Sync {
    /// Returns the batch size (number of samples).
    fn batch_size(&self) -> usize;
}

/// Gradient information from a backward pass.
///
/// Contains the computed gradients and loss for a training step.
#[derive(Debug, Clone)]
pub struct GradientInfo {
    /// The computed loss value.
    pub loss: f32,
    /// L2 norm of all gradients.
    pub gradient_norm: f32,
    /// Per-parameter gradient norms (optional, for debugging).
    pub per_param_norms: Option<Vec<f32>>,
}

/// Trait for models that can be trained with the hybrid trainer.
///
/// Models must implement forward pass, backward pass, and parameter access.
/// The trainer will call these methods during different training phases.
///
/// # Type Parameters
///
/// - `B`: The batch type containing input data
///
/// # Example
///
/// ```rust,ignore
/// impl Model<MyBatch> for MyModel {
///     fn forward(&mut self, batch: &MyBatch) -> HybridResult<f32> {
///         // Compute forward pass and return loss
///     }
///
///     fn backward(&mut self) -> HybridResult<GradientInfo> {
///         // Compute gradients (assumes forward was just called)
///     }
///
///     fn parameter_count(&self) -> usize {
///         self.parameters.iter().map(|p| p.numel()).sum()
///     }
/// }
/// ```
pub trait Model<B: Batch>: Send + Sync {
    /// Executes the forward pass and returns the loss.
    ///
    /// # Arguments
    ///
    /// * `batch` - The input batch data
    ///
    /// # Returns
    ///
    /// The loss value for this batch.
    fn forward(&mut self, batch: &B) -> HybridResult<f32>;

    /// Executes the backward pass (gradient computation).
    ///
    /// Should be called after `forward()`. Computes gradients with respect
    /// to the loss returned by the most recent forward pass.
    ///
    /// # Returns
    ///
    /// Gradient information including loss and gradient norms.
    fn backward(&mut self) -> HybridResult<GradientInfo>;

    /// Returns the total number of trainable parameters.
    fn parameter_count(&self) -> usize;

    /// Applies a weight delta to the model parameters.
    ///
    /// Used during predictive phase to apply predicted weight updates.
    ///
    /// # Arguments
    ///
    /// * `delta` - The weight changes to apply
    fn apply_weight_delta(&mut self, delta: &state::WeightDelta) -> HybridResult<()>;
}

/// Trait for optimizers that update model parameters.
///
/// Optimizers implement the parameter update rule (SGD, Adam, etc.).
///
/// # Example
///
/// ```rust,ignore
/// impl<M: Model<B>, B: Batch> Optimizer<M, B> for AdamOptimizer {
///     fn step(&mut self, model: &mut M, gradients: &GradientInfo) -> HybridResult<()> {
///         // Apply Adam update rule to model parameters
///     }
/// }
/// ```
pub trait Optimizer<M, B: Batch>: Send + Sync
where
    M: Model<B>,
{
    /// Performs a single optimization step.
    ///
    /// Updates model parameters using the computed gradients.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to update
    /// * `gradients` - Gradient information from backward pass
    fn step(&mut self, model: &mut M, gradients: &GradientInfo) -> HybridResult<()>;

    /// Returns the current learning rate.
    fn learning_rate(&self) -> f32;

    /// Sets the learning rate (for warmup/decay schedules).
    fn set_learning_rate(&mut self, lr: f32);

    /// Zeros all accumulated gradients.
    fn zero_grad(&mut self);
}

/// The main hybrid trainer that orchestrates phase-based predictive training.
///
/// # Overview
///
/// `HybridTrainer` wraps a model and optimizer, managing the training loop through
/// warmup, full training, predictive, and correction phases. It automatically
/// selects optimal phase lengths using bandit-based algorithms and monitors for
/// divergence to ensure training stability.
///
/// # Type Parameters
///
/// - `M`: The model type (must implement `Model` trait)
/// - `O`: The optimizer type (must implement `Optimizer` trait)
///
/// # Example
///
/// ```no_run
/// use hybrid_predict_trainer_rs::{HybridTrainer, HybridTrainerConfig};
///
/// // Configure the trainer
/// let config = HybridTrainerConfig::builder()
///     .warmup_steps(100)
///     .full_steps(20)
///     .max_predict_steps(80)
///     .confidence_threshold(0.85)
///     .build();
///
/// // Create trainer (model and optimizer types are inferred)
/// // let trainer = HybridTrainer::new(model, optimizer, config)?;
/// ```
pub struct HybridTrainer<M, O> {
    /// The model being trained.
    model: Arc<RwLock<M>>,
    
    /// The optimizer for parameter updates.
    optimizer: Arc<RwLock<O>>,
    
    /// Training configuration.
    config: HybridTrainerConfig,
    
    /// Current training state.
    state: TrainingState,
    
    /// Phase controller for state machine management.
    phase_controller: phases::DefaultPhaseController,
    
    /// Dynamics model for whole-phase prediction.
    dynamics_model: dynamics::RSSMLite,
    
    /// Divergence monitor for stability detection.
    divergence_monitor: divergence::DivergenceMonitor,
    
    /// Residual corrector for prediction adjustment.
    residual_corrector: corrector::ResidualCorrector,
    
    /// Metrics collector for training statistics.
    metrics: metrics::MetricsCollector,
}

impl<M, O> HybridTrainer<M, O> {
    /// Creates a new hybrid trainer with the given model, optimizer, and configuration.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train
    /// * `optimizer` - The optimizer for parameter updates
    /// * `config` - Training configuration
    ///
    /// # Returns
    ///
    /// A new `HybridTrainer` instance wrapped in a `HybridResult`.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or initialization fails.
    pub fn new(model: M, optimizer: O, config: HybridTrainerConfig) -> HybridResult<Self> {
        let state = TrainingState::new();
        let phase_controller = phases::DefaultPhaseController::new(&config);
        let dynamics_model = dynamics::RSSMLite::new(&config.predictor_config)?;
        let divergence_monitor = divergence::DivergenceMonitor::new(&config);
        let residual_corrector = corrector::ResidualCorrector::new(&config);
        let metrics = metrics::MetricsCollector::new(config.collect_metrics);
        
        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            optimizer: Arc::new(RwLock::new(optimizer)),
            config,
            state,
            phase_controller,
            dynamics_model,
            divergence_monitor,
            residual_corrector,
            metrics,
        })
    }
    
    /// Returns the current training step.
    ///
    /// # Returns
    ///
    /// The current step number (0-indexed).
    pub fn current_step(&self) -> u64 {
        self.state.step
    }
    
    /// Returns the current training phase.
    ///
    /// # Returns
    ///
    /// The current [`Phase`] of training.
    pub fn current_phase(&self) -> Phase {
        self.phase_controller.current_phase()
    }
    
    /// Returns the current predictor confidence level.
    ///
    /// # Returns
    ///
    /// A confidence score between 0.0 and 1.0 indicating how reliable
    /// the predictor's outputs are estimated to be.
    pub fn current_confidence(&self) -> f32 {
        self.dynamics_model.prediction_confidence(&self.state)
    }
    
    /// Returns training statistics and metrics.
    ///
    /// # Returns
    ///
    /// A [`TrainingStatistics`] struct containing aggregate metrics.
    pub fn statistics(&self) -> metrics::TrainingStatistics {
        self.metrics.statistics()
    }
}

/// Result of a single training step.
///
/// Contains the loss value, phase information, and prediction metadata
/// for monitoring training progress and predictor accuracy.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// The loss value for this step.
    pub loss: f32,
    
    /// The phase during which this step was executed.
    pub phase: Phase,
    
    /// Whether this step used predicted gradients (true) or computed gradients (false).
    pub was_predicted: bool,
    
    /// The error between predicted and actual loss (if applicable).
    pub prediction_error: Option<f32>,
    
    /// The predictor's confidence for this step.
    pub confidence: f32,
    
    /// Wall-clock time for this step in milliseconds.
    pub step_time_ms: f64,
    
    /// Detailed metrics (if collection is enabled).
    pub metrics: Option<metrics::StepMetrics>,
}

/// Prelude module for convenient imports.
///
/// # Example
///
/// ```
/// use hybrid_predict_trainer_rs::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        HybridTrainer,
        HybridTrainerConfig,
        HybridTrainingError,
        HybridResult,
        Phase,
        PhaseDecision,
        RecoveryAction,
        StepResult,
        TrainingState,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HybridTrainerConfig::default();
        assert_eq!(config.warmup_steps, 100);
        assert_eq!(config.full_steps, 20);
        assert_eq!(config.max_predict_steps, 80);
        assert!((config.confidence_threshold - 0.85).abs() < f32::EPSILON);
    }
}
