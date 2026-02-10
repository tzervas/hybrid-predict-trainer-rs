//! GPU memory monitoring and CPU-GPU coordination.
//!
//! Monitors GPU VRAM to prevent collisions, timeouts, and memory pressure.
//! Coordinates CPU and GPU memory allocation to avoid bottlenecks.

use std::time::{Duration, Instant};

/// GPU memory limits and configuration
#[derive(Debug, Clone, Copy)]
pub struct GpuMemoryConfig {
    /// Maximum GPU memory in GB (typically 8-80 depending on device)
    pub max_gpu_gb: f64,
    /// Warning threshold (% of max)
    pub warning_threshold: f64,
    /// CPU side memory to keep free for coordination
    pub cpu_reserve_gb: f64,
    /// Check interval
    pub check_interval: Duration,
}

impl GpuMemoryConfig {
    /// Default config for typical GPU (24GB VRAM)
    pub fn default_24gb() -> Self {
        Self {
            max_gpu_gb: 24.0,
            warning_threshold: 0.85,
            cpu_reserve_gb: 5.0,
            check_interval: Duration::from_secs(1),
        }
    }

    /// Config for high-end GPU (80GB A100)
    pub fn high_end() -> Self {
        Self {
            max_gpu_gb: 80.0,
            warning_threshold: 0.85,
            cpu_reserve_gb: 10.0,
            check_interval: Duration::from_secs(1),
        }
    }

    /// Config for mobile GPU (8GB)
    pub fn mobile() -> Self {
        Self {
            max_gpu_gb: 8.0,
            warning_threshold: 0.80,
            cpu_reserve_gb: 2.0,
            check_interval: Duration::from_secs(2),
        }
    }
}

impl Default for GpuMemoryConfig {
    fn default() -> Self {
        Self::default_24gb()
    }
}

/// GPU memory monitor
pub struct GpuMemoryMonitor {
    config: GpuMemoryConfig,
    last_check: Instant,
    peak_usage: f64,
    warnings_issued: usize,
    critical_events: usize,
    last_successful_allocation: Instant,
}

impl GpuMemoryMonitor {
    /// Creates new GPU monitor
    pub fn new(config: GpuMemoryConfig) -> Self {
        Self {
            config,
            last_check: Instant::now(),
            peak_usage: 0.0,
            warnings_issued: 0,
            critical_events: 0,
            last_successful_allocation: Instant::now(),
        }
    }

    /// Gets estimated GPU memory used (requires nvidia-smi on CUDA systems)
    pub fn current_gpu_usage_gb() -> Option<f64> {
        #[cfg(feature = "cuda")]
        {
            // Try to get GPU memory via nvidia-smi
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&[
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    if let Ok(mb) = text.trim().parse::<f64>() {
                        return Some(mb / 1024.0);
                    }
                }
            }
        }
        None
    }

    /// Gets total GPU memory (requires nvidia-smi)
    pub fn total_gpu_memory_gb() -> Option<f64> {
        #[cfg(feature = "cuda")]
        {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&[
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    if let Ok(mb) = text.trim().parse::<f64>() {
                        return Some(mb / 1024.0);
                    }
                }
            }
        }
        None
    }

    /// Checks GPU memory status
    pub fn check(&mut self) -> GpuMemoryStatus {
        if self.last_check.elapsed() < self.config.check_interval {
            return GpuMemoryStatus::Ok;
        }

        self.last_check = Instant::now();

        // Try to get actual GPU memory
        if let Some(used_gb) = Self::current_gpu_usage_gb() {
            self.peak_usage = self.peak_usage.max(used_gb);

            // Check against limit
            if used_gb > self.config.max_gpu_gb {
                self.critical_events += 1;
                GpuMemoryStatus::ExceededLimit {
                    used_gb,
                    limit_gb: self.config.max_gpu_gb,
                }
            } else if used_gb > (self.config.max_gpu_gb * self.config.warning_threshold) {
                self.warnings_issued += 1;
                GpuMemoryStatus::Warning {
                    used_gb,
                    limit_gb: self.config.max_gpu_gb,
                }
            } else {
                self.last_successful_allocation = Instant::now();
                GpuMemoryStatus::Ok
            }
        } else {
            // GPU not available or nvidia-smi not accessible
            GpuMemoryStatus::Unavailable
        }
    }

    /// Checks if timeslicing might be needed (high memory pressure)
    pub fn needs_timeslicing(&self) -> bool {
        self.warnings_issued > 2 || self.critical_events > 0
    }

    /// Checks if MLP (Multi-Level Parallelism) strategy needed
    pub fn needs_mlp_strategy(&self) -> bool {
        self.last_successful_allocation.elapsed() > Duration::from_secs(10)
    }

    /// Gets diagnostic info
    pub fn diagnostics(&self) -> String {
        format!(
            "GPU Monitor: peak={:.1}GB, warnings={}, critical={}",
            self.peak_usage, self.warnings_issued, self.critical_events
        )
    }
}

impl Default for GpuMemoryMonitor {
    fn default() -> Self {
        Self::new(GpuMemoryConfig::default())
    }
}

/// GPU memory status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuMemoryStatus {
    /// All good
    Ok,
    /// Approaching limit
    Warning {
        used_gb: f64,
        limit_gb: f64,
    },
    /// Exceeded limit
    ExceededLimit {
        used_gb: f64,
        limit_gb: f64,
    },
    /// GPU not available or not monitored
    Unavailable,
}

impl GpuMemoryStatus {
    /// Checks if safe
    pub fn is_ok(&self) -> bool {
        matches!(self, GpuMemoryStatus::Ok | GpuMemoryStatus::Unavailable)
    }

    /// Gets message
    pub fn message(&self) -> String {
        match self {
            GpuMemoryStatus::Ok => "✅ GPU Memory OK".to_string(),
            GpuMemoryStatus::Warning { used_gb, limit_gb } => {
                format!(
                    "⚠️  GPU Memory Warning: {:.1}GB / {:.1}GB",
                    used_gb, limit_gb
                )
            }
            GpuMemoryStatus::ExceededLimit { used_gb, limit_gb } => {
                format!(
                    "❌ GPU Memory Exceeded: {:.1}GB / {:.1}GB",
                    used_gb, limit_gb
                )
            }
            GpuMemoryStatus::Unavailable => "⚠️  GPU Memory Unavailable (CPU-only or no nvidia-smi)".to_string(),
        }
    }
}

/// CPU-GPU coordination strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinationStrategy {
    /// Normal operation
    Normal,
    /// Reduce GPU load (use timeslicing)
    Timeslicing,
    /// Use Multi-Level Parallelism (CPU + GPU)
    MLP,
    /// Fallback to CPU-only
    CpuFallback,
}

/// Coordinator for CPU-GPU memory
pub struct CpuGpuCoordinator {
    cpu_monitor: super::memory_monitor::MemoryMonitor,
    gpu_monitor: GpuMemoryMonitor,
    last_strategy: CoordinationStrategy,
}

impl CpuGpuCoordinator {
    /// Creates new coordinator
    pub fn new(gpu_config: GpuMemoryConfig) -> Self {
        Self {
            cpu_monitor: super::memory_monitor::MemoryMonitor::new(),
            gpu_monitor: GpuMemoryMonitor::new(gpu_config),
            last_strategy: CoordinationStrategy::Normal,
        }
    }

    /// Recommends strategy based on current memory state
    pub fn recommend_strategy(&mut self) -> CoordinationStrategy {
        let gpu_status = self.gpu_monitor.check();
        let cpu_used = super::memory_monitor::MemoryMonitor::total_ram_gb()
            - super::memory_monitor::MemoryMonitor::available_memory_gb();

        // Decide strategy
        let strategy = match (gpu_status, cpu_used > 27.0) {
            (GpuMemoryStatus::ExceededLimit { .. }, _) => CoordinationStrategy::CpuFallback,
            (GpuMemoryStatus::Warning { .. }, true) => CoordinationStrategy::MLP,
            (GpuMemoryStatus::Warning { .. }, false) => CoordinationStrategy::Timeslicing,
            (GpuMemoryStatus::Ok | GpuMemoryStatus::Unavailable, true) => CoordinationStrategy::MLP,
            _ => CoordinationStrategy::Normal,
        };

        self.last_strategy = strategy;
        strategy
    }

    /// Gets diagnostic info
    pub fn diagnostics(&mut self) -> String {
        format!(
            "CPU-GPU: CPU={}GB, GPU={}, Strategy={:?}",
            super::memory_monitor::MemoryMonitor::total_ram_gb()
                - super::memory_monitor::MemoryMonitor::available_memory_gb(),
            gpu_memory_status_short(&self.gpu_monitor.check()),
            self.last_strategy
        )
    }
}

fn gpu_memory_status_short(status: &GpuMemoryStatus) -> String {
    match status {
        GpuMemoryStatus::Ok => "OK".to_string(),
        GpuMemoryStatus::Warning { .. } => "WARN".to_string(),
        GpuMemoryStatus::ExceededLimit { .. } => "EXCEEDED".to_string(),
        GpuMemoryStatus::Unavailable => "N/A".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_presets() {
        let default = GpuMemoryConfig::default();
        assert_eq!(default.max_gpu_gb, 24.0);

        let high_end = GpuMemoryConfig::high_end();
        assert_eq!(high_end.max_gpu_gb, 80.0);

        let mobile = GpuMemoryConfig::mobile();
        assert_eq!(mobile.max_gpu_gb, 8.0);
    }

    #[test]
    fn test_gpu_monitor_creation() {
        let monitor = GpuMemoryMonitor::new(GpuMemoryConfig::default());
        assert_eq!(monitor.peak_usage, 0.0);
        assert!(!monitor.needs_timeslicing());
    }

    #[test]
    fn test_gpu_memory_status() {
        let ok = GpuMemoryStatus::Ok;
        assert!(ok.is_ok());

        let warn = GpuMemoryStatus::Warning {
            used_gb: 20.0,
            limit_gb: 24.0,
        };
        assert!(!warn.is_ok());

        let unavail = GpuMemoryStatus::Unavailable;
        assert!(unavail.is_ok());
    }

    #[test]
    fn test_coordinator_creation() {
        let coordinator = CpuGpuCoordinator::new(GpuMemoryConfig::default());
        assert_eq!(coordinator.last_strategy, CoordinationStrategy::Normal);
    }
}
