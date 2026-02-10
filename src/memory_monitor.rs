//! Memory monitoring and management for training safety.
//!
//! Ensures RAM usage stays within configured limits and provides warnings
//! when approaching thresholds.

use std::fs;
use std::time::{Duration, Instant};

/// Memory limit configuration
#[derive(Debug, Clone, Copy)]
pub struct MemoryLimit {
    /// Maximum RAM usage in GB
    pub max_ram_gb: f64,
    /// Warning threshold (% of max)
    pub warning_threshold: f64,
    /// Check interval
    pub check_interval: Duration,
}

impl Default for MemoryLimit {
    fn default() -> Self {
        Self {
            max_ram_gb: 30.0,
            warning_threshold: 0.85,
            check_interval: Duration::from_secs(5),
        }
    }
}

/// Memory monitor for training safety
pub struct MemoryMonitor {
    config: MemoryLimit,
    last_check: Instant,
    warnings_issued: usize,
    peak_usage: f64,
}

impl MemoryMonitor {
    /// Creates a new memory monitor with default limits (30GB max)
    pub fn new() -> Self {
        Self {
            config: MemoryLimit::default(),
            last_check: Instant::now(),
            warnings_issued: 0,
            peak_usage: 0.0,
        }
    }

    /// Creates a monitor with custom RAM limit in GB
    pub fn with_limit(max_ram_gb: f64) -> Self {
        Self {
            config: MemoryLimit {
                max_ram_gb,
                warning_threshold: 0.85,
                check_interval: Duration::from_secs(5),
            },
            last_check: Instant::now(),
            warnings_issued: 0,
            peak_usage: 0.0,
        }
    }

    /// Gets current system memory usage in GB
    pub fn current_usage_gb() -> f64 {
        Self::parse_meminfo("MemTotal", "MemAvailable")
    }

    /// Gets memory available in GB
    pub fn available_memory_gb() -> f64 {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb) = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        return kb / (1024.0 * 1024.0);
                    }
                }
            }
        }
        0.0
    }

    /// Gets current resident set size (actual process memory)
    pub fn process_rss_mb() -> f64 {
        if let Ok(content) = fs::read_to_string("/proc/self/status") {
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb) = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        return kb / 1024.0;
                    }
                }
            }
        }
        0.0
    }

    /// Checks memory usage and returns status
    pub fn check(&mut self) -> MemoryStatus {
        if self.last_check.elapsed() < self.config.check_interval {
            return MemoryStatus::Ok;
        }

        let total_gb = Self::total_ram_gb();
        let available_gb = Self::available_memory_gb();
        let used_gb = total_gb - available_gb;
        let process_mb = Self::process_rss_mb();

        self.peak_usage = self.peak_usage.max(used_gb);
        self.last_check = Instant::now();

        // Check against limit
        if used_gb > self.config.max_ram_gb {
            MemoryStatus::ExceededLimit {
                used_gb,
                limit_gb: self.config.max_ram_gb,
                process_mb,
            }
        } else if used_gb > (self.config.max_ram_gb * self.config.warning_threshold) {
            self.warnings_issued += 1;
            MemoryStatus::Warning {
                used_gb,
                limit_gb: self.config.max_ram_gb,
                process_mb,
            }
        } else {
            MemoryStatus::Ok
        }
    }

    /// Gets peak memory usage observed
    pub fn peak_usage_gb(&self) -> f64 {
        self.peak_usage
    }

    /// Gets total system RAM in GB
    pub fn total_ram_gb() -> f64 {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        return kb / (1024.0 * 1024.0);
                    }
                }
            }
        }
        0.0
    }

    /// Parses /proc/meminfo for memory calculations
    fn parse_meminfo(total_field: &str, available_field: &str) -> f64 {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            let total = content
                .lines()
                .find(|l| l.starts_with(total_field))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            let available = content
                .lines()
                .find(|l| l.starts_with(available_field))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);

            (total - available) / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory status from monitoring
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryStatus {
    /// All good, under limits
    Ok,
    /// Warning: approaching limit
    Warning {
        used_gb: f64,
        limit_gb: f64,
        process_mb: f64,
    },
    /// Error: exceeded limit
    ExceededLimit {
        used_gb: f64,
        limit_gb: f64,
        process_mb: f64,
    },
}

impl MemoryStatus {
    /// Checks if status indicates a problem
    pub fn is_ok(&self) -> bool {
        matches!(self, MemoryStatus::Ok)
    }

    /// Checks if status is a warning
    pub fn is_warning(&self) -> bool {
        matches!(self, MemoryStatus::Warning { .. })
    }

    /// Checks if limit exceeded
    pub fn exceeded(&self) -> bool {
        matches!(self, MemoryStatus::ExceededLimit { .. })
    }

    /// Gets human-readable message
    pub fn message(&self) -> String {
        match self {
            MemoryStatus::Ok => "✅ Memory OK".to_string(),
            MemoryStatus::Warning {
                used_gb,
                limit_gb,
                process_mb,
            } => {
                format!(
                    "⚠️  Memory Warning: {:.1}GB / {:.1}GB (Process: {:.0}MB)",
                    used_gb, limit_gb, process_mb
                )
            }
            MemoryStatus::ExceededLimit {
                used_gb,
                limit_gb,
                process_mb,
            } => {
                format!(
                    "❌ Memory Exceeded: {:.1}GB / {:.1}GB (Process: {:.0}MB)",
                    used_gb, limit_gb, process_mb
                )
            }
        }
    }
}

/// Memory-aware batch size calculator
pub struct MemoryAwareBatchSize {
    initial_batch_size: usize,
    min_batch_size: usize,
    max_batch_size: usize,
    memory_monitor: MemoryMonitor,
}

impl MemoryAwareBatchSize {
    /// Creates new memory-aware batch size calculator
    pub fn new(initial_batch_size: usize) -> Self {
        Self {
            initial_batch_size,
            min_batch_size: 4,
            max_batch_size: initial_batch_size,
            memory_monitor: MemoryMonitor::new(),
        }
    }

    /// Gets recommended batch size based on current memory usage
    pub fn get_batch_size(&mut self) -> usize {
        let status = self.memory_monitor.check();

        match status {
            MemoryStatus::ExceededLimit { .. } => {
                // Reduce aggressively
                tracing::warn!("{}", status.message());
                (self.initial_batch_size / 4).max(self.min_batch_size)
            }
            MemoryStatus::Warning { .. } => {
                // Reduce moderately
                tracing::warn!("{}", status.message());
                (self.initial_batch_size / 2).max(self.min_batch_size)
            }
            MemoryStatus::Ok => self.initial_batch_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_monitor_creation() {
        let monitor = MemoryMonitor::new();
        assert_eq!(monitor.config.max_ram_gb, 30.0);
    }

    #[test]
    fn test_custom_limit() {
        let monitor = MemoryMonitor::with_limit(16.0);
        assert_eq!(monitor.config.max_ram_gb, 16.0);
    }

    #[test]
    fn test_memory_reading() {
        let total = MemoryMonitor::total_ram_gb();
        assert!(total > 0.0, "Should read total RAM");

        let available = MemoryMonitor::available_memory_gb();
        assert!(available > 0.0, "Should read available RAM");
        assert!(available <= total, "Available should be <= total");
    }

    #[test]
    fn test_memory_status_messages() {
        let ok = MemoryStatus::Ok;
        assert!(ok.is_ok());
        assert!(!ok.is_warning());
        assert!(!ok.exceeded());

        let warning = MemoryStatus::Warning {
            used_gb: 25.0,
            limit_gb: 30.0,
            process_mb: 2048.0,
        };
        assert!(!warning.is_ok());
        assert!(warning.is_warning());
        assert!(!warning.exceeded());

        let exceeded = MemoryStatus::ExceededLimit {
            used_gb: 31.0,
            limit_gb: 30.0,
            process_mb: 3072.0,
        };
        assert!(!exceeded.is_ok());
        assert!(!exceeded.is_warning());
        assert!(exceeded.exceeded());
    }

    #[test]
    fn test_batch_size_adjustment() {
        let mut calc = MemoryAwareBatchSize::new(64);
        let batch = calc.get_batch_size();
        assert!(batch <= 64);
        assert!(batch >= 4);
    }
}
