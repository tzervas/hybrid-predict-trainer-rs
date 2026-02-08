//! Memory profiling utilities for tracking VRAM usage during training.
//!
//! This module provides utilities for monitoring GPU memory usage in real-time,
//! tracking memory spikes, and generating profiling reports.
//!
//! ## Why This Module
//!
//! The v0.3.0 memory optimization stack (gradient checkpointing, CPU offloading,
//! quantization, Flash Attention) aims to enable training of massive models
//! (1B-50B parameters) on consumer GPUs (16-24 GB VRAM). This module provides
//! the tools to validate these optimizations work as intended.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use hybrid_predict_trainer_rs::memory_profiler::{MemoryProfiler, MemorySnapshot};
//!
//! let mut profiler = MemoryProfiler::new();
//! profiler.start();
//!
//! // Training loop
//! for step in 0..1000 {
//!     profiler.record_step(step, "Full");
//!     // ... training ...
//! }
//!
//! let report = profiler.generate_report();
//! println!("{}", report);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A single memory usage snapshot.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Training step number.
    pub step: usize,
    /// Training phase ("Warmup", "Full", "Predict", "Correct").
    pub phase: String,
    /// Total GPU memory used in bytes.
    pub used_bytes: u64,
    /// Total GPU memory available in bytes.
    pub total_bytes: u64,
    /// Timestamp when snapshot was taken.
    pub timestamp: Instant,
}

impl MemorySnapshot {
    /// Memory used in MB.
    pub fn used_mb(&self) -> f64 {
        self.used_bytes as f64 / 1_048_576.0
    }

    /// Memory used in GB.
    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / 1_073_741_824.0
    }

    /// Memory utilization percentage.
    pub fn utilization_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }
}

/// Memory profiler for tracking GPU VRAM usage over time.
///
/// ## Why This Type
///
/// Provides comprehensive memory tracking with:
/// - Per-step memory snapshots
/// - Phase-aware memory analysis
/// - Peak memory detection
/// - Memory spike identification
/// - Profiling report generation
///
/// Critical for validating v0.3.0 memory optimizations work as intended.
#[derive(Debug)]
pub struct MemoryProfiler {
    /// All memory snapshots recorded.
    snapshots: Vec<MemorySnapshot>,
    /// Start time of profiling session.
    start_time: Option<Instant>,
    /// Peak memory usage seen (bytes).
    peak_usage: u64,
    /// Step where peak occurred.
    peak_step: Option<usize>,
    /// Phase where peak occurred.
    peak_phase: Option<String>,
}

impl MemoryProfiler {
    /// Create a new memory profiler.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            start_time: None,
            peak_usage: 0,
            peak_step: None,
            peak_phase: None,
        }
    }

    /// Start profiling session.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.snapshots.clear();
        self.peak_usage = 0;
        self.peak_step = None;
        self.peak_phase = None;
    }

    /// Record a memory snapshot for the current training step.
    ///
    /// ## Why This Method
    ///
    /// Call this at the beginning of each training step to track memory usage
    /// over time. The profiler will query nvidia-smi for current GPU memory
    /// usage and record it along with phase information.
    pub fn record_step(&mut self, step: usize, phase: &str) {
        if let Some(snapshot) = self.query_memory(step, phase) {
            // Track peak
            if snapshot.used_bytes > self.peak_usage {
                self.peak_usage = snapshot.used_bytes;
                self.peak_step = Some(step);
                self.peak_phase = Some(phase.to_string());
            }

            self.snapshots.push(snapshot);
        }
    }

    /// Query current GPU memory usage via nvidia-smi.
    ///
    /// Returns None if nvidia-smi is unavailable or fails.
    fn query_memory(&self, step: usize, phase: &str) -> Option<MemorySnapshot> {
        // Query used memory
        let used_output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;

        let used_mb = String::from_utf8(used_output.stdout)
            .ok()?
            .trim()
            .parse::<f64>()
            .ok()?;

        // Query total memory
        let total_output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;

        let total_mb = String::from_utf8(total_output.stdout)
            .ok()?
            .trim()
            .parse::<f64>()
            .ok()?;

        Some(MemorySnapshot {
            step,
            phase: phase.to_string(),
            used_bytes: (used_mb * 1_048_576.0) as u64,
            total_bytes: (total_mb * 1_048_576.0) as u64,
            timestamp: Instant::now(),
        })
    }

    /// Get all recorded snapshots.
    pub fn snapshots(&self) -> &[MemorySnapshot] {
        &self.snapshots
    }

    /// Get peak memory usage in bytes.
    pub fn peak_usage_bytes(&self) -> u64 {
        self.peak_usage
    }

    /// Get peak memory usage in MB.
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_usage as f64 / 1_048_576.0
    }

    /// Get peak memory usage in GB.
    pub fn peak_usage_gb(&self) -> f64 {
        self.peak_usage as f64 / 1_073_741_824.0
    }

    /// Get step where peak occurred.
    pub fn peak_step(&self) -> Option<usize> {
        self.peak_step
    }

    /// Get phase where peak occurred.
    pub fn peak_phase(&self) -> Option<&str> {
        self.peak_phase.as_deref()
    }

    /// Calculate memory statistics by phase.
    ///
    /// Returns map of phase -> (mean_mb, max_mb, min_mb).
    pub fn phase_statistics(&self) -> HashMap<String, (f64, f64, f64)> {
        let mut stats: HashMap<String, Vec<f64>> = HashMap::new();

        for snapshot in &self.snapshots {
            stats
                .entry(snapshot.phase.clone())
                .or_default()
                .push(snapshot.used_mb());
        }

        stats
            .into_iter()
            .map(|(phase, values)| {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                (phase, (mean, max, min))
            })
            .collect()
    }

    /// Identify memory spikes (sudden increases > threshold).
    ///
    /// Returns list of (step, delta_mb, phase).
    pub fn identify_spikes(&self, threshold_mb: f64) -> Vec<(usize, f64, String)> {
        let mut spikes = Vec::new();

        for i in 1..self.snapshots.len() {
            let prev = &self.snapshots[i - 1];
            let curr = &self.snapshots[i];

            let delta = curr.used_mb() - prev.used_mb();

            if delta > threshold_mb {
                spikes.push((curr.step, delta, curr.phase.clone()));
            }
        }

        spikes
    }

    /// Generate a comprehensive profiling report.
    ///
    /// ## Why This Method
    ///
    /// Provides human-readable summary of memory usage patterns, phase-specific
    /// statistics, and identifies potential issues (spikes, high utilization).
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        report.push_str("Memory Profiling Report\n");
        report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

        if self.snapshots.is_empty() {
            report.push_str("No memory snapshots recorded.\n");
            return report;
        }

        // Overall statistics
        report.push_str("Overall Statistics:\n");
        report.push_str(&format!(
            "  Peak usage: {:.2} GB (step {}, phase {})\n",
            self.peak_usage_gb(),
            self.peak_step.unwrap_or(0),
            self.peak_phase.as_deref().unwrap_or("unknown")
        ));

        let first = &self.snapshots[0];
        let last = &self.snapshots[self.snapshots.len() - 1];
        report.push_str(&format!(
            "  Initial usage: {:.2} GB\n",
            first.used_gb()
        ));
        report.push_str(&format!("  Final usage: {:.2} GB\n", last.used_gb()));
        report.push_str(&format!(
            "  GPU capacity: {:.2} GB\n",
            first.total_bytes as f64 / 1_073_741_824.0
        ));
        report.push_str(&format!(
            "  Peak utilization: {:.1}%\n",
            (self.peak_usage as f64 / first.total_bytes as f64) * 100.0
        ));

        // Phase statistics
        report.push_str("\nPhase Statistics:\n");
        let phase_stats = self.phase_statistics();
        let mut phases: Vec<_> = phase_stats.keys().collect();
        phases.sort();

        for phase in phases {
            let (mean, max, min) = phase_stats[phase];
            report.push_str(&format!(
                "  {:<10} mean: {:.2} GB, max: {:.2} GB, min: {:.2} GB\n",
                phase,
                mean / 1024.0,
                max / 1024.0,
                min / 1024.0
            ));
        }

        // Memory spikes
        report.push_str("\nMemory Spikes (>100 MB increase):\n");
        let spikes = self.identify_spikes(100.0);
        if spikes.is_empty() {
            report.push_str("  No significant spikes detected.\n");
        } else {
            for (step, delta, phase) in spikes.iter().take(10) {
                report.push_str(&format!(
                    "  Step {:4} ({:7}): +{:.1} MB\n",
                    step, phase, delta
                ));
            }
            if spikes.len() > 10 {
                report.push_str(&format!("  ... and {} more spikes\n", spikes.len() - 10));
            }
        }

        // Duration
        if let Some(start) = self.start_time {
            let duration = last.timestamp.duration_since(start);
            report.push_str(&format!(
                "\nProfiling duration: {:.1}s ({} steps)\n",
                duration.as_secs_f64(),
                self.snapshots.len()
            ));
        }

        report.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        report
    }

    /// Export snapshots to CSV format.
    ///
    /// Returns CSV string with columns: step,phase,used_mb,total_mb,utilization_percent
    pub fn export_csv(&self) -> String {
        let mut csv = String::from("step,phase,used_mb,total_mb,utilization_percent\n");

        for snapshot in &self.snapshots {
            csv.push_str(&format!(
                "{},{},{:.2},{:.2},{:.2}\n",
                snapshot.step,
                snapshot.phase,
                snapshot.used_mb(),
                snapshot.total_bytes as f64 / 1_048_576.0,
                snapshot.utilization_percent()
            ));
        }

        csv
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_snapshot_conversions() {
        let snapshot = MemorySnapshot {
            step: 42,
            phase: "Full".to_string(),
            used_bytes: 2_147_483_648, // 2 GB
            total_bytes: 17_179_869_184, // 16 GB
            timestamp: Instant::now(),
        };

        assert_eq!(snapshot.used_mb(), 2048.0);
        assert_eq!(snapshot.used_gb(), 2.0);
        assert!((snapshot.utilization_percent() - 12.5).abs() < 0.01);
    }

    #[test]
    fn test_profiler_creation() {
        let profiler = MemoryProfiler::new();
        assert_eq!(profiler.snapshots().len(), 0);
        assert_eq!(profiler.peak_usage_bytes(), 0);
        assert_eq!(profiler.peak_step(), None);
        assert_eq!(profiler.peak_phase(), None);
    }

    #[test]
    fn test_profiler_start() {
        let mut profiler = MemoryProfiler::new();
        profiler.start();
        assert!(profiler.start_time.is_some());
    }

    #[test]
    fn test_phase_statistics_empty() {
        let profiler = MemoryProfiler::new();
        let stats = profiler.phase_statistics();
        assert!(stats.is_empty());
    }

    #[test]
    fn test_identify_spikes_empty() {
        let profiler = MemoryProfiler::new();
        let spikes = profiler.identify_spikes(100.0);
        assert!(spikes.is_empty());
    }

    #[test]
    fn test_generate_report_empty() {
        let profiler = MemoryProfiler::new();
        let report = profiler.generate_report();
        assert!(report.contains("No memory snapshots recorded"));
    }

    #[test]
    fn test_export_csv_empty() {
        let profiler = MemoryProfiler::new();
        let csv = profiler.export_csv();
        assert_eq!(csv, "step,phase,used_mb,total_mb,utilization_percent\n");
    }

    #[test]
    fn test_peak_tracking() {
        let mut profiler = MemoryProfiler::new();
        profiler.start();

        // Manually insert snapshots (since nvidia-smi may not be available)
        profiler.snapshots.push(MemorySnapshot {
            step: 0,
            phase: "Warmup".to_string(),
            used_bytes: 1_073_741_824, // 1 GB
            total_bytes: 17_179_869_184, // 16 GB
            timestamp: Instant::now(),
        });

        profiler.snapshots.push(MemorySnapshot {
            step: 1,
            phase: "Full".to_string(),
            used_bytes: 3_221_225_472, // 3 GB
            total_bytes: 17_179_869_184,
            timestamp: Instant::now(),
        });

        profiler.snapshots.push(MemorySnapshot {
            step: 2,
            phase: "Predict".to_string(),
            used_bytes: 2_147_483_648, // 2 GB
            total_bytes: 17_179_869_184,
            timestamp: Instant::now(),
        });

        // Manually update peak (normally done in record_step)
        profiler.peak_usage = 3_221_225_472;
        profiler.peak_step = Some(1);
        profiler.peak_phase = Some("Full".to_string());

        assert_eq!(profiler.peak_usage_gb(), 3.0);
        assert_eq!(profiler.peak_step(), Some(1));
        assert_eq!(profiler.peak_phase(), Some("Full"));
    }

    #[test]
    fn test_spike_identification() {
        let mut profiler = MemoryProfiler::new();
        profiler.start();

        // Create snapshots with a spike
        profiler.snapshots.push(MemorySnapshot {
            step: 0,
            phase: "Warmup".to_string(),
            used_bytes: 1_073_741_824, // 1 GB
            total_bytes: 17_179_869_184,
            timestamp: Instant::now(),
        });

        profiler.snapshots.push(MemorySnapshot {
            step: 1,
            phase: "Full".to_string(),
            used_bytes: 1_178_599_424, // ~1.1 GB (+100 MB)
            total_bytes: 17_179_869_184,
            timestamp: Instant::now(),
        });

        profiler.snapshots.push(MemorySnapshot {
            step: 2,
            phase: "Full".to_string(),
            used_bytes: 2_415_919_104, // ~2.25 GB (+1.15 GB = spike!)
            total_bytes: 17_179_869_184,
            timestamp: Instant::now(),
        });

        let spikes = profiler.identify_spikes(500.0); // 500 MB threshold
        assert_eq!(spikes.len(), 1);
        assert_eq!(spikes[0].0, 2); // step 2
        assert_eq!(spikes[0].2, "Full"); // Full phase
    }

    #[test]
    fn test_csv_export() {
        let mut profiler = MemoryProfiler::new();
        profiler.start();

        profiler.snapshots.push(MemorySnapshot {
            step: 0,
            phase: "Warmup".to_string(),
            used_bytes: 1_073_741_824, // 1 GB
            total_bytes: 17_179_869_184, // 16 GB
            timestamp: Instant::now(),
        });

        let csv = profiler.export_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 data row
        assert!(lines[0].contains("step,phase,used_mb"));
        assert!(lines[1].contains("0,Warmup,1024.00"));
    }
}
