//! Training metrics collection and reporting.
//!
//! Provides comprehensive metrics collection for monitoring training
//! progress, debugging issues, and evaluating the effectiveness of
//! predictive training.
//!
//! # Collected Metrics
//!
//! - **Step-level**: Loss, gradient norm, prediction error, phase
//! - **Phase-level**: Duration, steps, average metrics
//! - **Aggregate**: Total speedup, backward reduction, loss quality
//!
//! # Output Formats
//!
//! Metrics can be exported as:
//! - JSON for programmatic analysis
//! - Console summary for monitoring
//! - Parquet for efficient storage (future)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Phase;

/// Metrics for a single training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Training step number.
    pub step: u64,
    
    /// Loss value.
    pub loss: f32,
    
    /// Gradient norm.
    pub gradient_norm: f32,
    
    /// Current phase.
    pub phase: Phase,
    
    /// Whether this step used predictions.
    pub was_predicted: bool,
    
    /// Prediction error (if applicable).
    pub prediction_error: Option<f32>,
    
    /// Predictor confidence.
    pub confidence: f32,
    
    /// Wall-clock time in milliseconds.
    pub time_ms: f64,
    
    /// Learning rate (if available).
    pub learning_rate: Option<f32>,
}

/// Metrics for a completed training phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    /// Phase type.
    pub phase: Phase,
    
    /// Starting step.
    pub start_step: u64,
    
    /// Ending step.
    pub end_step: u64,
    
    /// Number of steps executed.
    pub steps_executed: usize,
    
    /// Average loss during phase.
    pub average_loss: f32,
    
    /// Final loss at phase end.
    pub final_loss: f32,
    
    /// Average gradient norm.
    pub average_gradient_norm: f32,
    
    /// Total phase duration in milliseconds.
    pub duration_ms: f64,
    
    /// Whether phase completed normally.
    pub completed_normally: bool,
    
    /// Prediction error (for predict phase).
    pub prediction_error: Option<f32>,
}

/// Aggregate training statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingStatistics {
    /// Total training steps.
    pub total_steps: u64,
    
    /// Steps spent in warmup.
    pub warmup_steps: usize,
    
    /// Steps spent in full training.
    pub full_steps: usize,
    
    /// Steps spent in prediction.
    pub predict_steps: usize,
    
    /// Steps spent in correction.
    pub correct_steps: usize,
    
    /// Percentage of backward passes avoided.
    pub backward_reduction_pct: f32,
    
    /// Wall-clock speedup factor vs traditional training.
    pub wall_clock_speedup: f32,
    
    /// Final training loss.
    pub final_loss: f32,
    
    /// Estimated baseline loss (traditional training).
    pub baseline_loss_estimate: f32,
    
    /// Loss gap percentage.
    pub loss_gap_pct: f32,
    
    /// Average prediction length.
    pub avg_predict_length: f32,
    
    /// Maximum prediction length achieved.
    pub max_predict_length: usize,
    
    /// Average predictor confidence.
    pub avg_confidence: f32,
    
    /// Prediction accuracy statistics.
    pub prediction_accuracy: PredictionAccuracy,
    
    /// Number of divergence events.
    pub divergence_events: usize,
    
    /// Predictor overhead statistics.
    pub predictor_overhead: PredictorOverhead,
}

/// Prediction accuracy statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionAccuracy {
    /// Mean absolute error of loss predictions.
    pub loss_mae: f32,
    
    /// Correlation between predicted and actual loss.
    pub loss_correlation: f32,
    
    /// Cosine similarity of predicted vs actual weight updates.
    pub weight_cosine_similarity: f32,
}

/// Predictor overhead statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictorOverhead {
    /// Average encoding time in milliseconds.
    pub encode_time_ms_avg: f64,
    
    /// Average prediction time in milliseconds.
    pub predict_time_ms_avg: f64,
    
    /// Average update time in milliseconds.
    pub update_time_ms_avg: f64,
    
    /// Memory used by predictor in MB.
    pub memory_used_mb: f32,
    
    /// Percentage of step time spent on prediction.
    pub pct_of_step_time: f32,
}

/// Divergence event record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceEvent {
    /// Step where divergence was detected.
    pub step: u64,
    
    /// Severity level.
    pub severity: String,
    
    /// Action taken.
    pub action: String,
    
    /// Whether recovery was successful.
    pub recovery_successful: bool,
}

/// Collector for training metrics.
pub struct MetricsCollector {
    /// Whether collection is enabled.
    enabled: bool,
    
    /// Step-level metrics (limited buffer).
    step_metrics: Vec<StepMetrics>,
    
    /// Phase-level metrics.
    phase_metrics: Vec<PhaseMetrics>,
    
    /// Divergence events.
    divergence_events: Vec<DivergenceEvent>,
    
    /// Running statistics.
    statistics: TrainingStatistics,
    
    /// Maximum step metrics to keep in memory.
    max_step_metrics: usize,
    
    /// Per-phase step counters.
    phase_step_counts: HashMap<Phase, usize>,
    
    /// Total time in each phase.
    phase_times: HashMap<Phase, f64>,
    
    /// Prediction errors for accuracy tracking.
    prediction_errors: Vec<f32>,
}

impl MetricsCollector {
    /// Creates a new metrics collector.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            step_metrics: Vec::with_capacity(10000),
            phase_metrics: Vec::new(),
            divergence_events: Vec::new(),
            statistics: TrainingStatistics::default(),
            max_step_metrics: 10000,
            phase_step_counts: HashMap::new(),
            phase_times: HashMap::new(),
            prediction_errors: Vec::new(),
        }
    }
    
    /// Records metrics for a training step.
    pub fn record_step(&mut self, metrics: StepMetrics) {
        if !self.enabled {
            return;
        }

        // Update per-phase counters
        *self.phase_step_counts.entry(metrics.phase).or_insert(0) += 1;
        *self.phase_times.entry(metrics.phase).or_insert(0.0) += metrics.time_ms;

        // Track prediction errors
        if let Some(error) = metrics.prediction_error {
            self.prediction_errors.push(error);
        }

        // Update statistics
        self.statistics.total_steps = metrics.step;
        self.statistics.final_loss = metrics.loss;

        // Store step metrics (with eviction if needed)
        if self.step_metrics.len() >= self.max_step_metrics {
            self.step_metrics.remove(0);
        }
        self.step_metrics.push(metrics);
    }

    /// Records metrics for a training step from individual values.
    ///
    /// Convenience method that creates a StepMetrics and records it.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `loss` - Loss value
    /// * `phase` - Current phase
    /// * `was_predicted` - Whether predictions were used
    /// * `prediction_error` - Error between predicted and actual (if applicable)
    /// * `confidence` - Predictor confidence level
    ///
    /// # Returns
    ///
    /// The created StepMetrics struct.
    pub fn record_step_data(
        &mut self,
        step: u64,
        loss: f32,
        phase: Phase,
        was_predicted: bool,
        prediction_error: Option<f32>,
        confidence: f32,
    ) -> StepMetrics {
        let metrics = StepMetrics {
            step,
            loss,
            gradient_norm: 0.0, // Updated separately if available
            phase,
            was_predicted,
            prediction_error,
            confidence,
            time_ms: 0.0, // Will be updated by caller
            learning_rate: None,
        };

        self.record_step(metrics.clone());
        metrics
    }
    
    /// Records metrics for a completed phase.
    pub fn record_phase(&mut self, metrics: PhaseMetrics) {
        if !self.enabled {
            return;
        }
        
        self.phase_metrics.push(metrics);
    }
    
    /// Records a divergence event.
    pub fn record_divergence(&mut self, event: DivergenceEvent) {
        if !self.enabled {
            return;
        }
        
        self.divergence_events.push(event);
        self.statistics.divergence_events += 1;
    }
    
    /// Finalizes statistics computation.
    pub fn finalize(&mut self) {
        // Update phase step counts
        self.statistics.warmup_steps = *self.phase_step_counts.get(&Phase::Warmup).unwrap_or(&0);
        self.statistics.full_steps = *self.phase_step_counts.get(&Phase::Full).unwrap_or(&0);
        self.statistics.predict_steps = *self.phase_step_counts.get(&Phase::Predict).unwrap_or(&0);
        self.statistics.correct_steps = *self.phase_step_counts.get(&Phase::Correct).unwrap_or(&0);
        
        // Compute backward reduction
        let total = self.statistics.total_steps as f32;
        let backward_steps = (self.statistics.warmup_steps + self.statistics.full_steps) as f32;
        if total > 0.0 {
            self.statistics.backward_reduction_pct = 
                100.0 * (1.0 - backward_steps / total);
        }
        
        // Compute prediction accuracy
        if !self.prediction_errors.is_empty() {
            let sum: f32 = self.prediction_errors.iter().sum();
            self.statistics.prediction_accuracy.loss_mae = 
                sum / self.prediction_errors.len() as f32;
        }
        
        // Compute average prediction length
        let predict_phases: Vec<_> = self.phase_metrics
            .iter()
            .filter(|p| p.phase == Phase::Predict)
            .collect();
        
        if !predict_phases.is_empty() {
            let total_predict_steps: usize = predict_phases
                .iter()
                .map(|p| p.steps_executed)
                .sum();
            self.statistics.avg_predict_length = 
                total_predict_steps as f32 / predict_phases.len() as f32;
            self.statistics.max_predict_length = predict_phases
                .iter()
                .map(|p| p.steps_executed)
                .max()
                .unwrap_or(0);
        }
        
        // Compute wall-clock speedup (estimate based on times)
        let full_time = self.phase_times.get(&Phase::Full).unwrap_or(&0.0);
        let predict_time = self.phase_times.get(&Phase::Predict).unwrap_or(&0.0);
        
        if *predict_time > 0.0 && self.statistics.predict_steps > 0 {
            let full_time_per_step = if self.statistics.full_steps > 0 {
                full_time / self.statistics.full_steps as f64
            } else {
                1.0
            };
            let predict_time_per_step = predict_time / self.statistics.predict_steps as f64;
            
            if predict_time_per_step > 0.0 {
                let speedup_factor = full_time_per_step / predict_time_per_step;
                self.statistics.wall_clock_speedup = speedup_factor as f32;
            }
        }
    }
    
    /// Returns the current statistics.
    pub fn statistics(&self) -> TrainingStatistics {
        self.statistics.clone()
    }
    
    /// Exports metrics to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let export = MetricsExport {
            training_summary: self.statistics.clone(),
            phase_history: self.phase_metrics.clone(),
            divergence_events: self.divergence_events.clone(),
        };
        serde_json::to_string_pretty(&export)
    }
    
    /// Returns a console-friendly summary.
    pub fn summary(&self) -> String {
        let stats = &self.statistics;
        format!(
            "Training Summary:\n\
             ├─ Total Steps: {}\n\
             ├─ Phases: W={}, F={}, P={}, C={}\n\
             ├─ Backward Reduction: {:.1}%\n\
             ├─ Wall-Clock Speedup: {:.1}x\n\
             ├─ Final Loss: {:.4}\n\
             ├─ Avg Predict Length: {:.1}\n\
             ├─ Prediction MAE: {:.4}\n\
             └─ Divergence Events: {}",
            stats.total_steps,
            stats.warmup_steps, stats.full_steps, stats.predict_steps, stats.correct_steps,
            stats.backward_reduction_pct,
            stats.wall_clock_speedup,
            stats.final_loss,
            stats.avg_predict_length,
            stats.prediction_accuracy.loss_mae,
            stats.divergence_events
        )
    }
    
    /// Resets all collected metrics.
    pub fn reset(&mut self) {
        self.step_metrics.clear();
        self.phase_metrics.clear();
        self.divergence_events.clear();
        self.statistics = TrainingStatistics::default();
        self.phase_step_counts.clear();
        self.phase_times.clear();
        self.prediction_errors.clear();
    }
}

/// Export structure for metrics serialization.
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsExport {
    /// Training summary statistics.
    pub training_summary: TrainingStatistics,
    
    /// Phase-level metrics history.
    pub phase_history: Vec<PhaseMetrics>,
    
    /// Divergence events.
    pub divergence_events: Vec<DivergenceEvent>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_disabled() {
        let mut collector = MetricsCollector::new(false);
        
        collector.record_step(StepMetrics {
            step: 1,
            loss: 2.5,
            gradient_norm: 1.0,
            phase: Phase::Warmup,
            was_predicted: false,
            prediction_error: None,
            confidence: 0.9,
            time_ms: 10.0,
            learning_rate: Some(0.001),
        });
        
        // Should not record when disabled
        assert!(collector.step_metrics.is_empty());
    }

    #[test]
    fn test_collector_enabled() {
        let mut collector = MetricsCollector::new(true);
        
        collector.record_step(StepMetrics {
            step: 1,
            loss: 2.5,
            gradient_norm: 1.0,
            phase: Phase::Warmup,
            was_predicted: false,
            prediction_error: None,
            confidence: 0.9,
            time_ms: 10.0,
            learning_rate: Some(0.001),
        });
        
        assert_eq!(collector.step_metrics.len(), 1);
    }

    #[test]
    fn test_finalize_statistics() {
        let mut collector = MetricsCollector::new(true);
        
        // Record some warmup steps
        for i in 0..10 {
            collector.record_step(StepMetrics {
                step: i,
                loss: 3.0 - i as f32 * 0.1,
                gradient_norm: 1.0,
                phase: Phase::Warmup,
                was_predicted: false,
                prediction_error: None,
                confidence: 0.5,
                time_ms: 10.0,
                learning_rate: Some(0.001),
            });
        }
        
        // Record some predict steps
        for i in 10..30 {
            collector.record_step(StepMetrics {
                step: i,
                loss: 2.0 - i as f32 * 0.01,
                gradient_norm: 0.8,
                phase: Phase::Predict,
                was_predicted: true,
                prediction_error: Some(0.05),
                confidence: 0.9,
                time_ms: 5.0,
                learning_rate: Some(0.001),
            });
        }
        
        collector.finalize();
        
        let stats = collector.statistics();
        assert_eq!(stats.warmup_steps, 10);
        assert_eq!(stats.predict_steps, 20);
        assert!(stats.backward_reduction_pct > 0.0);
    }

    #[test]
    fn test_json_export() {
        let collector = MetricsCollector::new(true);
        let json = collector.to_json().unwrap();
        
        assert!(json.contains("training_summary"));
        assert!(json.contains("phase_history"));
    }
}
