//! Training evaluation metrics for hybrid predictive training.
//!
//! Tracks loss curves, perplexity, and throughput during training.

use crate::phases::Phase;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

/// Snapshot of training metrics at a given step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training step number.
    pub step: u64,
    /// Training loss value.
    pub train_loss: f32,
    /// Validation loss (if available).
    pub val_loss: Option<f32>,
    /// Perplexity (exp of loss).
    pub perplexity: f32,
    /// Tokens processed per second.
    pub tokens_per_second: f32,
    /// Current training phase.
    pub phase: String,
    /// VRAM usage in megabytes.
    pub vram_used_mb: u64,
    /// Unix timestamp in seconds.
    pub timestamp_secs: u64,
}

impl TrainingMetrics {
    /// Create new training metrics snapshot.
    ///
    /// # Arguments
    ///
    /// * `step` - Training step number
    /// * `train_loss` - Training loss value
    /// * `tokens_per_second` - Throughput in tokens/sec
    /// * `phase` - Current training phase
    pub fn new(step: u64, train_loss: f32, tokens_per_second: f32, phase: &Phase) -> Self {
        let perplexity = train_loss.exp();
        Self {
            step,
            train_loss,
            val_loss: None,
            perplexity,
            tokens_per_second,
            phase: format!("{phase:?}"),
            vram_used_mb: 0,
            timestamp_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Compute perplexity from loss value.
///
/// Perplexity = exp(loss)
#[must_use]
pub fn perplexity_from_loss(loss: f32) -> f32 {
    loss.exp()
}

/// Running average of training loss with history tracking.
#[derive(Debug)]
pub struct LossTracker {
    /// Sliding window of recent losses.
    window: VecDeque<f32>,
    /// Window size for averaging.
    window_size: usize,
    /// Full history of metrics.
    history: Vec<TrainingMetrics>,
    /// Optional log file path.
    log_path: Option<std::path::PathBuf>,
}

impl LossTracker {
    /// Create new loss tracker with specified window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of recent steps to average
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            history: Vec::new(),
            log_path: None,
        }
    }

    /// Set the log file path for saving history.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save JSON metrics
    pub fn with_log_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.log_path = Some(path.into());
        self
    }

    /// Record a training step with metrics.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Training metrics to record
    pub fn record(&mut self, metrics: TrainingMetrics) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(metrics.train_loss);
        self.history.push(metrics);
    }

    /// Compute current average loss over window.
    ///
    /// # Returns
    ///
    /// Average of recent losses, or infinity if no losses recorded.
    #[must_use]
    pub fn avg_loss(&self) -> f32 {
        if self.window.is_empty() {
            return f32::INFINITY;
        }
        self.window.iter().sum::<f32>() / self.window.len() as f32
    }

    /// Compute current perplexity from average loss.
    ///
    /// # Returns
    ///
    /// Perplexity = exp(average_loss)
    #[must_use]
    pub fn perplexity(&self) -> f32 {
        perplexity_from_loss(self.avg_loss())
    }

    /// Save metrics history to JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save to
    ///
    /// # Errors
    ///
    /// Returns error if file write fails.
    pub fn save_json(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.history)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Get final metrics summary for model card generation.
    ///
    /// # Returns
    ///
    /// Tuple of (final_loss, final_perplexity)
    #[must_use]
    pub fn summary(&self) -> (f32, f32) {
        let last_loss = self.history.last().map_or(f32::INFINITY, |m| m.train_loss);
        let last_ppl = perplexity_from_loss(last_loss);
        (last_loss, last_ppl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity_from_loss() {
        // ln(2) ≈ 0.693, so loss=0.693 → perplexity=2.0
        let ppl = perplexity_from_loss(0.6931472);
        assert!((ppl - 2.0).abs() < 0.01, "Expected perplexity ~2.0, got {ppl}");
    }

    #[test]
    fn test_loss_tracker() {
        let mut tracker = LossTracker::new(10);
        for i in 0..5 {
            let m = TrainingMetrics::new(i as u64, 2.0 - i as f32 * 0.1, 1000.0, &Phase::Full);
            tracker.record(m);
        }
        assert!(tracker.avg_loss() > 1.5);
        assert!(tracker.avg_loss() < 2.1);
    }

    #[test]
    fn test_loss_tracker_summary() {
        let mut tracker = LossTracker::new(5);
        let m1 = TrainingMetrics::new(0, 2.0, 1000.0, &Phase::Full);
        tracker.record(m1);
        let m2 = TrainingMetrics::new(1, 1.5, 1000.0, &Phase::Full);
        tracker.record(m2);

        let (loss, ppl) = tracker.summary();
        assert_eq!(loss, 1.5);
        assert!((ppl - 1.5_f32.exp()).abs() < 0.001);
    }

    #[test]
    fn test_loss_tracker_window() {
        let mut tracker = LossTracker::new(3);
        tracker.record(TrainingMetrics::new(0, 2.0, 1000.0, &Phase::Full));
        tracker.record(TrainingMetrics::new(1, 2.0, 1000.0, &Phase::Full));
        tracker.record(TrainingMetrics::new(2, 2.0, 1000.0, &Phase::Full));
        tracker.record(TrainingMetrics::new(3, 1.0, 1000.0, &Phase::Full));

        // Window should contain [2.0, 2.0, 1.0], average = 5.0/3 ≈ 1.667
        let avg = tracker.avg_loss();
        assert!((avg - 5.0 / 3.0).abs() < 0.01);
    }
}
