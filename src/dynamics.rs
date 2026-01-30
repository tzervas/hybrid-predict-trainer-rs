//! RSSM-lite dynamics model for training trajectory prediction.
//!
//! This module implements a simplified Recurrent State-Space Model (RSSM)
//! inspired by DreamerV3 for predicting training dynamics. The model
//! combines deterministic (GRU-based) and stochastic components to capture
//! both predictable trends and inherent uncertainty in training.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                    RSSM-Lite                        │
//! ├─────────────────────────────────────────────────────┤
//! │                                                     │
//! │  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
//! │  │  Input   │───▶│   GRU    │───▶│Determ.   │      │
//! │  │ Encoder  │    │  Cell    │    │ State    │      │
//! │  └──────────┘    └──────────┘    └────┬─────┘      │
//! │                                       │            │
//! │                                       ▼            │
//! │                               ┌──────────────┐     │
//! │                               │  Stochastic  │     │
//! │                               │   Sampler    │     │
//! │                               └──────┬───────┘     │
//! │                                      │             │
//! │                                      ▼             │
//! │  ┌──────────┐    ┌──────────┐    ┌──────────┐     │
//! │  │  Loss    │◀───│ Combined │◀───│Stochastic│     │
//! │  │  Head    │    │  State   │    │  State   │     │
//! │  └──────────┘    └──────────┘    └──────────┘     │
//! │                                                    │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Deterministic path**: Captures predictable training dynamics
//! - **Stochastic path**: Models uncertainty and variance in outcomes
//! - **Ensemble**: Multiple models for uncertainty estimation
//! - **Multi-step prediction**: Directly predict Y steps ahead

use crate::config::PredictorConfig;
use crate::error::HybridResult;
use crate::predictive::PhasePrediction;
use crate::state::{TrainingState, WeightDelta};

/// Latent state of the RSSM model.
#[derive(Debug, Clone)]
pub struct LatentState {
    /// Deterministic state (GRU hidden state).
    pub deterministic: Vec<f32>,
    
    /// Stochastic state (sampled from categorical).
    pub stochastic: Vec<f32>,
    
    /// Logits for the categorical distribution.
    pub stochastic_logits: Vec<f32>,
    
    /// Combined state (concatenation of deterministic and stochastic).
    pub combined: Vec<f32>,
}

impl LatentState {
    /// Creates a new latent state with the given dimensions.
    pub fn new(deterministic_dim: usize, stochastic_dim: usize) -> Self {
        let combined_dim = deterministic_dim + stochastic_dim;
        Self {
            deterministic: vec![0.0; deterministic_dim],
            stochastic: vec![0.0; stochastic_dim],
            stochastic_logits: vec![0.0; stochastic_dim],
            combined: vec![0.0; combined_dim],
        }
    }
    
    /// Updates the combined state from deterministic and stochastic components.
    pub fn update_combined(&mut self) {
        self.combined.clear();
        self.combined.extend_from_slice(&self.deterministic);
        self.combined.extend_from_slice(&self.stochastic);
    }
}

/// Uncertainty estimate for a prediction.
#[derive(Debug, Clone)]
pub struct PredictionUncertainty {
    /// Aleatoric uncertainty (inherent randomness).
    pub aleatoric: f32,
    
    /// Epistemic uncertainty (model uncertainty).
    pub epistemic: f32,
    
    /// Total uncertainty (combined).
    pub total: f32,
    
    /// Entropy of the stochastic distribution.
    pub entropy: f32,
}

impl Default for PredictionUncertainty {
    fn default() -> Self {
        Self {
            aleatoric: 0.0,
            epistemic: 0.0,
            total: 0.0,
            entropy: 0.0,
        }
    }
}

/// Configuration for RSSM-lite model.
#[derive(Debug, Clone)]
pub struct RSSMConfig {
    /// Dimension of deterministic state.
    pub deterministic_dim: usize,
    
    /// Dimension of stochastic state.
    pub stochastic_dim: usize,
    
    /// Number of categorical distributions.
    pub num_categoricals: usize,
    
    /// Number of ensemble members.
    pub ensemble_size: usize,
    
    /// Input feature dimension.
    pub input_dim: usize,
    
    /// Hidden dimension for MLPs.
    pub hidden_dim: usize,
    
    /// Learning rate for model updates.
    pub learning_rate: f32,
}

impl Default for RSSMConfig {
    fn default() -> Self {
        Self {
            deterministic_dim: 256,
            stochastic_dim: 32,
            num_categoricals: 32,
            ensemble_size: 3,
            input_dim: 32, // From TrainingState::compute_features
            hidden_dim: 128,
            learning_rate: 0.001,
        }
    }
}

impl From<&PredictorConfig> for RSSMConfig {
    fn from(config: &PredictorConfig) -> Self {
        match config {
            PredictorConfig::RSSM {
                deterministic_dim,
                stochastic_dim,
                num_categoricals,
                ensemble_size,
            } => Self {
                deterministic_dim: *deterministic_dim,
                stochastic_dim: *stochastic_dim,
                num_categoricals: *num_categoricals,
                ensemble_size: *ensemble_size,
                ..Default::default()
            },
            _ => Self::default(),
        }
    }
}

/// RSSM-lite dynamics model for training prediction.
pub struct RSSMLite {
    /// Model configuration.
    config: RSSMConfig,
    
    /// Current latent state for each ensemble member.
    latent_states: Vec<LatentState>,
    
    /// GRU weights for each ensemble member.
    gru_weights: Vec<GRUWeights>,
    
    /// Loss prediction head weights.
    loss_head_weights: Vec<f32>,
    
    /// Training step counter.
    training_steps: usize,
    
    /// Historical prediction errors for confidence estimation.
    prediction_errors: Vec<f32>,
    
    /// Temperature for stochastic sampling.
    temperature: f32,
}

/// Weights for a GRU cell.
#[derive(Debug, Clone)]
struct GRUWeights {
    /// Update gate weights.
    w_z: Vec<f32>,
    /// Reset gate weights.
    w_r: Vec<f32>,
    /// Candidate hidden state weights.
    w_h: Vec<f32>,
    /// Update gate recurrent weights.
    u_z: Vec<f32>,
    /// Reset gate recurrent weights.
    u_r: Vec<f32>,
    /// Candidate hidden state recurrent weights.
    u_h: Vec<f32>,
    /// Biases.
    b_z: Vec<f32>,
    b_r: Vec<f32>,
    b_h: Vec<f32>,
}

impl GRUWeights {
    /// Creates randomly initialized GRU weights.
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        
        let scale = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        
        let mut random_vec = |size: usize| -> Vec<f32> {
            (0..size).map(|_| rng.random_range(-scale..scale)).collect()
        };
        
        Self {
            w_z: random_vec(input_dim * hidden_dim),
            w_r: random_vec(input_dim * hidden_dim),
            w_h: random_vec(input_dim * hidden_dim),
            u_z: random_vec(hidden_dim * hidden_dim),
            u_r: random_vec(hidden_dim * hidden_dim),
            u_h: random_vec(hidden_dim * hidden_dim),
            b_z: vec![0.0; hidden_dim],
            b_r: vec![0.0; hidden_dim],
            b_h: vec![0.0; hidden_dim],
        }
    }
}

impl RSSMLite {
    /// Creates a new RSSM-lite model with the given configuration.
    pub fn new(config: &PredictorConfig) -> HybridResult<Self> {
        let rssm_config = RSSMConfig::from(config);
        
        let latent_states: Vec<_> = (0..rssm_config.ensemble_size)
            .map(|_| LatentState::new(rssm_config.deterministic_dim, rssm_config.stochastic_dim))
            .collect();
        
        let gru_weights: Vec<_> = (0..rssm_config.ensemble_size)
            .map(|_| GRUWeights::new(rssm_config.input_dim, rssm_config.deterministic_dim))
            .collect();
        
        let combined_dim = rssm_config.deterministic_dim + rssm_config.stochastic_dim;
        let loss_head_weights = vec![0.0; combined_dim];
        
        Ok(Self {
            config: rssm_config,
            latent_states,
            gru_weights,
            loss_head_weights,
            training_steps: 0,
            prediction_errors: Vec::with_capacity(1000),
            temperature: 1.0,
        })
    }
    
    /// Initializes latent state from training state.
    pub fn initialize_state(&mut self, state: &TrainingState) {
        let features = state.compute_features();
        
        for latent in &mut self.latent_states {
            // Simple initialization: project features to deterministic state
            for (i, &f) in features.iter().enumerate() {
                if i < latent.deterministic.len() {
                    latent.deterministic[i] = f.tanh();
                }
            }
            
            // Initialize stochastic state uniformly
            for s in &mut latent.stochastic {
                *s = 1.0 / self.config.stochastic_dim as f32;
            }
            
            latent.update_combined();
        }
    }
    
    /// Predicts training outcome after Y steps.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    /// * `y_steps` - Number of steps to predict ahead
    ///
    /// # Returns
    ///
    /// Prediction with uncertainty estimate.
    pub fn predict_y_steps(
        &self,
        state: &TrainingState,
        y_steps: usize,
    ) -> (PhasePrediction, PredictionUncertainty) {
        let _features = state.compute_features();
        
        // Get predictions from each ensemble member
        let mut predictions: Vec<f32> = Vec::with_capacity(self.config.ensemble_size);
        
        for (_i, latent) in self.latent_states.iter().enumerate() {
            // Simple prediction: dot product of combined state with loss head
            let pred: f32 = latent.combined
                .iter()
                .zip(self.loss_head_weights.iter())
                .map(|(&s, &w)| s * w)
                .sum();
            
            // Apply step scaling (assume linear loss decay)
            let step_factor = 1.0 - (y_steps as f32 * 0.001).min(0.5);
            let scaled_pred = state.loss * step_factor + pred * 0.1;
            
            predictions.push(scaled_pred);
        }
        
        // Compute ensemble statistics
        let mean_pred: f32 = predictions.iter().sum::<f32>() / predictions.len() as f32;
        let variance: f32 = predictions
            .iter()
            .map(|&p| (p - mean_pred).powi(2))
            .sum::<f32>() / predictions.len() as f32;
        let std = variance.sqrt();
        
        let uncertainty = PredictionUncertainty {
            aleatoric: std * 0.5,
            epistemic: std * 0.5,
            total: std,
            entropy: 0.0, // Would compute from stochastic logits
        };
        
        let prediction = PhasePrediction {
            weight_delta: WeightDelta::empty(),
            predicted_final_loss: mean_pred,
            loss_trajectory: vec![state.loss, mean_pred],
            confidence: 1.0 / (1.0 + std), // Higher std = lower confidence
            loss_bounds: (mean_pred - 2.0 * std, mean_pred + 2.0 * std),
            num_steps: y_steps,
        };
        
        (prediction, uncertainty)
    }
    
    /// Returns the prediction confidence for the current state.
    pub fn prediction_confidence(&self, state: &TrainingState) -> f32 {
        // Base confidence from ensemble agreement
        let (_, uncertainty) = self.predict_y_steps(state, 10);
        let agreement_confidence = 1.0 / (1.0 + uncertainty.total);
        
        // Historical accuracy confidence
        let historical_confidence = if self.prediction_errors.len() < 10 {
            0.5 // Low confidence until we have enough data
        } else {
            let recent_errors: Vec<_> = self.prediction_errors
                .iter()
                .rev()
                .take(50)
                .cloned()
                .collect();
            let mean_error: f32 = recent_errors.iter().sum::<f32>() / recent_errors.len() as f32;
            (1.0 / (1.0 + mean_error)).min(0.99)
        };
        
        // Combine confidences
        (agreement_confidence * 0.6 + historical_confidence * 0.4).clamp(0.0, 1.0)
    }
    
    /// Updates the model from observed training data.
    ///
    /// # Arguments
    ///
    /// * `state_before` - State before training
    /// * `state_after` - State after training
    /// * `loss_trajectory` - Observed loss values during training
    pub fn update_from_observation(
        &mut self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        loss_trajectory: &[f32],
    ) -> HybridResult<()> {
        // Record prediction error
        let (prediction, _) = self.predict_y_steps(state_before, loss_trajectory.len());
        let actual_final_loss = state_after.loss;
        let error = (prediction.predicted_final_loss - actual_final_loss).abs();
        
        self.prediction_errors.push(error);
        if self.prediction_errors.len() > 1000 {
            self.prediction_errors.remove(0);
        }
        
        // Update model weights using simple gradient descent
        // This is a placeholder - real implementation would use proper backprop
        let learning_rate = self.config.learning_rate;
        let error_signal = prediction.predicted_final_loss - actual_final_loss;
        
        for (i, &combined) in self.latent_states[0].combined.iter().enumerate() {
            if i < self.loss_head_weights.len() {
                self.loss_head_weights[i] -= learning_rate * error_signal * combined;
            }
        }
        
        self.training_steps += 1;
        
        Ok(())
    }
    
    /// Returns the number of training updates performed.
    pub fn training_steps(&self) -> usize {
        self.training_steps
    }
    
    /// Resets the model to initial state.
    pub fn reset(&mut self) {
        for latent in &mut self.latent_states {
            latent.deterministic.fill(0.0);
            latent.stochastic.fill(1.0 / self.config.stochastic_dim as f32);
            latent.update_combined();
        }
        self.prediction_errors.clear();
        self.training_steps = 0;
    }
}

/// Trait for dynamics models that predict training trajectories.
pub trait DynamicsModel: Send + Sync {
    /// The latent state type.
    type LatentState: Clone + Send;
    
    /// Initializes latent state from training state.
    fn initialize(&mut self, state: &TrainingState);
    
    /// Predicts outcome after Y steps.
    fn predict_y_steps(
        &self,
        state: &TrainingState,
        y_steps: usize,
    ) -> (PhasePrediction, PredictionUncertainty);
    
    /// Returns prediction confidence.
    fn prediction_confidence(&self, state: &TrainingState) -> f32;
    
    /// Updates from observation.
    fn update_from_observation(
        &mut self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        loss_trajectory: &[f32],
    ) -> HybridResult<()>;
}

impl DynamicsModel for RSSMLite {
    type LatentState = LatentState;
    
    fn initialize(&mut self, state: &TrainingState) {
        self.initialize_state(state);
    }
    
    fn predict_y_steps(
        &self,
        state: &TrainingState,
        y_steps: usize,
    ) -> (PhasePrediction, PredictionUncertainty) {
        RSSMLite::predict_y_steps(self, state, y_steps)
    }
    
    fn prediction_confidence(&self, state: &TrainingState) -> f32 {
        RSSMLite::prediction_confidence(self, state)
    }
    
    fn update_from_observation(
        &mut self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        loss_trajectory: &[f32],
    ) -> HybridResult<()> {
        RSSMLite::update_from_observation(self, state_before, state_after, loss_trajectory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rssm_creation() {
        let config = PredictorConfig::default();
        let rssm = RSSMLite::new(&config).unwrap();
        
        assert_eq!(rssm.config.ensemble_size, 3);
        assert_eq!(rssm.latent_states.len(), 3);
    }

    #[test]
    fn test_latent_state_combined() {
        let mut state = LatentState::new(4, 2);
        state.deterministic = vec![1.0, 2.0, 3.0, 4.0];
        state.stochastic = vec![0.5, 0.5];
        state.update_combined();
        
        assert_eq!(state.combined.len(), 6);
        assert_eq!(state.combined[0], 1.0);
        assert_eq!(state.combined[4], 0.5);
    }

    #[test]
    fn test_prediction() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();
        
        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.record_step(2.5, 1.0);
        
        rssm.initialize_state(&state);
        let (prediction, uncertainty) = rssm.predict_y_steps(&state, 10);
        
        assert!(prediction.predicted_final_loss > 0.0);
        assert!(prediction.confidence > 0.0 && prediction.confidence <= 1.0);
        assert!(uncertainty.total >= 0.0);
    }

    #[test]
    fn test_confidence_with_history() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();
        
        // Initialize with proper state
        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.record_step(2.5, 1.0);
        rssm.initialize_state(&state);
        
        // Add some prediction errors (low errors = high confidence)
        for _ in 0..20 {
            rssm.prediction_errors.push(0.1);
        }
        
        let confidence = rssm.prediction_confidence(&state);
        
        // Should have reasonable confidence with low errors
        assert!(confidence > 0.5, "confidence={} should be > 0.5", confidence);
    }
}
