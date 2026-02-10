# Memory Management Guide

Complete guide for managing CPU RAM and GPU VRAM during training.

---

## Overview

The system includes comprehensive memory monitoring to ensure safe, efficient training:
- **CPU RAM**: Capped at 30GB with adaptive batch sizing
- **GPU VRAM**: Real-time monitoring with coordination strategies
- **CPU-GPU Coordination**: Prevents memory collisions and deadlocks

---

## Quick Start

### CPU Memory Monitoring (30GB Limit)

```bash
# Run training with automatic memory management
cargo run --release --example memory_aware_training --features "autodiff,datasets"
```

**Features:**
- ✅ Monitors RAM every 5 batches
- ✅ Auto-reduces batch size if approaching 30GB
- ✅ Reports memory status per epoch
- ✅ Peak memory tracking

**Example Output:**
```
📊 System Memory:
   Total RAM: 46.8GB
   Available: 26.7GB
   Process RSS: 3MB
   Memory Limit: 30.0GB

🏃 Training with Memory Monitoring
   Batch 10/50: loss=2.4915, mem=20.2GB, batch_sz=32
   ...
✅ Epoch 1/3: loss=2.3428, time=22.8s, peak_mem=20.2GB

📊 Final System Memory:
   Used: 19.9GB / 30.0GB limit ✅
```

---

## CPU Memory Management

### MemoryMonitor API

```rust
use hybrid_predict_trainer_rs::memory_monitor::*;

// Create monitor with default 30GB limit
let mut monitor = MemoryMonitor::new();

// Check memory status
match monitor.check() {
    MemoryStatus::Ok => { /* continue training */ },
    MemoryStatus::Warning { used_gb, limit_gb, process_mb } => {
        eprintln!("⚠️  {:.1}GB/{:.1}GB used", used_gb, limit_gb);
        // Reduce batch size
    },
    MemoryStatus::ExceededLimit { .. } => {
        eprintln!("❌ Memory limit exceeded!");
        // Emergency reduction or halt
    }
}

// Get peak usage
let peak_gb = monitor.peak_usage_gb();
```

### Dynamic Batch Sizing

```rust
use hybrid_predict_trainer_rs::memory_monitor::MemoryAwareBatchSize;

let mut batch_calc = MemoryAwareBatchSize::new(64);

loop {
    // Get recommended batch size based on current memory
    let batch_size = batch_calc.get_batch_size();

    // Train with adaptive batch
    train_batch(batch_size)?;
}
```

### Memory API Reference

```rust
// Get system info
MemoryMonitor::total_ram_gb()           // Total system RAM
MemoryMonitor::available_memory_gb()    // Free RAM
MemoryMonitor::current_usage_gb()       // Used RAM
MemoryMonitor::process_rss_mb()         // Process memory

// Create monitor with custom limit
MemoryMonitor::with_limit(16.0)  // 16GB limit

// Check status
monitor.check()                  // Returns MemoryStatus
monitor.peak_usage_gb()          // Peak observed

// Status checking
status.is_ok()                   // All good
status.is_warning()              // Near limit
status.exceeded()                // Over limit
status.message()                 // Human-readable message
```

---

## GPU Memory Management

### GpuMemoryMonitor API

```bash
# Requires nvidia-smi for GPU monitoring
# Install CUDA Toolkit to get nvidia-smi
```

```rust
use hybrid_predict_trainer_rs::gpu_memory_monitor::*;

// Create monitor for your GPU
let config = GpuMemoryConfig::default_24gb();  // 24GB VRAM
let mut gpu_monitor = GpuMemoryMonitor::new(config);

// Check GPU memory
match gpu_monitor.check() {
    GpuMemoryStatus::Ok => { /* continue */ },
    GpuMemoryStatus::Warning { used_gb, limit_gb } => {
        println!("⚠️  GPU: {:.1}GB/{:.1}GB", used_gb, limit_gb);
        // Reduce GPU load or enable timeslicing
    },
    GpuMemoryStatus::ExceededLimit { .. } => {
        println!("❌ GPU memory exceeded!");
        // Use CPU fallback
    },
    GpuMemoryStatus::Unavailable => {
        println!("GPU not available - CPU training");
    }
}

// Check if strategies needed
if gpu_monitor.needs_timeslicing() {
    println!("⏱️  Enable GPU timeslicing");
}
if gpu_monitor.needs_mlp_strategy() {
    println!("🔀 Use Multi-Level Parallelism");
}
```

### GPU Presets

```rust
// Typical consumer GPU (RTX 3080, 4090)
let config = GpuMemoryConfig::default_24gb();

// High-end enterprise GPU (A100)
let config = GpuMemoryConfig::high_end();  // 80GB

// Mobile GPU
let config = GpuMemoryConfig::mobile();   // 8GB
```

### CPU-GPU Coordination

```rust
use hybrid_predict_trainer_rs::gpu_memory_monitor::*;

let mut coordinator = CpuGpuCoordinator::new(GpuMemoryConfig::default());

loop {
    // Get recommended strategy
    let strategy = coordinator.recommend_strategy();

    match strategy {
        CoordinationStrategy::Normal => {
            // Full speed, no constraints
            full_speed_training()?;
        },
        CoordinationStrategy::Timeslicing => {
            // GPU: Reduce batch, use time slicing
            timesliced_training()?;
        },
        CoordinationStrategy::MLP => {
            // Both CPU and GPU: Multi-level parallelism
            mlp_training()?;
        },
        CoordinationStrategy::CpuFallback => {
            // GPU overloaded: Use CPU only
            cpu_only_training()?;
        }
    }

    // Get diagnostics
    println!("{}", coordinator.diagnostics());
}
```

---

## Coordination Strategies

### 1. Normal Operation (Default)

**When:** CPU < 25GB, GPU < 85% capacity
**What:** Full speed training, all available resources

```
CPU: =========          (20GB / 30GB)
GPU: ==============     (20GB / 24GB)
Strategy: Normal ✅
```

### 2. Timeslicing

**When:** GPU approaching 85% capacity
**What:** Reduce GPU load by increasing kernel launch intervals
**Benefit:** Prevents GPU memory exhaustion, maintains speed

```
CPU: ===========        (22GB / 30GB)
GPU: =================== (21GB / 24GB - WARNING)
Strategy: Timeslicing ⏱️
- Reduce batch size
- Increase gradient accumulation steps
- Enable GPU timeslicing if available
```

### 3. Multi-Level Parallelism (MLP)

**When:** Both CPU and GPU memory tight
**What:** Distribute work across CPU and GPU
**Benefit:** Better resource utilization

```
CPU: =================== (25GB / 30GB)
GPU: =================== (22GB / 24GB)
Strategy: MLP 🔀
- CPU: Data loading, preprocessing
- GPU: Model forward/backward
- Careful memory handoff
```

### 4. CPU Fallback

**When:** GPU memory exceeded
**What:** Move computation to CPU
**Benefit:** Avoid OOM crashes, continue training

```
CPU: ======================= (28GB / 30GB)
GPU: ======================== (24GB / 24GB - EXCEEDED!)
Strategy: CPU Fallback 🔴
- Transfer GPU tensors to CPU
- Slower but stable
- Monitor for OOM on CPU side
```

---

## System Configuration Examples

### 46GB System (Your System)

```
Total RAM: 46GB
Training allocation: 30GB
Reserved: 16GB (OS, other processes)

CPU Monitor:
├─ Limit: 30GB
├─ Warning: 25.5GB (85%)
└─ Peak observed: 20.2GB

Recommended:
- Batch size: 32-64
- Epochs: 3-5
- Gradient accumulation: 2-4
```

### High-Memory Server (256GB)

```
Total RAM: 256GB
Training allocation: 200GB
Reserved: 56GB (OS, services)

CPU Monitor:
├─ Limit: 200GB
├─ Warning: 170GB (85%)
└─ Good headroom for large models

Recommended:
- Batch size: 512-1024
- Large models: GPT-2 Large (360M)
- Gradient accumulation: 1-2
```

### Memory-Constrained (16GB)

```
Total RAM: 16GB
Training allocation: 12GB
Reserved: 4GB (OS, required)

CPU Monitor:
├─ Limit: 12GB
├─ Warning: 10.2GB (85%)
└─ Very tight!

Recommended:
- Batch size: 4-8
- Gradient accumulation: 8-16
- Enable CPU offloading
- Consider mixed precision (fp16)
```

---

## Best Practices

### 1. Monitor During Development

```bash
# Open new terminal
watch -n 1 free -h
```

```bash
# Monitor GPU (if available)
watch -n 1 nvidia-smi
```

### 2. Start Conservative

```rust
// Start with small batch size
let initial_batch_size = 16;

// Monitor for 1 epoch
// If memory stable, increase to 32
// If memory increasing, stay at 16
```

### 3. Use Gradient Accumulation

Instead of larger batch sizes (more memory):
```rust
// ❌ Large batch = more memory
train_with_batch_size(128)?;  // High memory usage

// ✅ Gradient accumulation = same effective batch, less memory
for chunk in [32, 32, 32, 32] {
    train_with_batch_size(chunk)?;  // Lower memory usage
    accumulate_gradients();
}
```

### 4. Enable Mixed Precision

```rust
// ✅ FP16 training uses ~50% less memory
// ✅ Similar accuracy
// ✅ Faster on modern GPUs
enable_mixed_precision();
```

### 5. Checkpointing

```rust
// Save state regularly
if epoch % 10 == 0 {
    save_checkpoint(&model, &optimizer, epoch)?;
}

// If OOM, can resume from last checkpoint
load_checkpoint(last_epoch)?;
```

---

## Troubleshooting

### "Memory Exceeded Limit"

**Problem:** RAM usage > 30GB
**Solutions:**
1. Reduce batch size
2. Enable gradient accumulation
3. Use gradient checkpointing
4. Enable mixed precision (fp16)
5. Reduce number of layers

### "GPU Memory Warning"

**Problem:** GPU VRAM near 85%
**Solutions:**
1. Reduce GPU batch size
2. Enable timeslicing: `CoordinationStrategy::Timeslicing`
3. Move some computation to CPU
4. Use activation checkpointing
5. Reduce model size

### "CPU-GPU Collision"

**Problem:** Both CPU and GPU memory tight
**Solutions:**
1. Use Multi-Level Parallelism strategy
2. Reduce batch size
3. Increase gradient accumulation steps
4. Enable CPU offloading
5. Wait for background processes

### "Training Gets Slower Over Time"

**Problem:** Memory fragmentation or resource exhaustion
**Solutions:**
1. Check `monitor.peak_usage_gb()` - if increasing, memory leak
2. Monitor GPU separately
3. Restart training to defragment memory
4. Check for background processes
5. Use memory profiler to find leaks

---

## Memory Profiling

### CPU Memory

```rust
use hybrid_predict_trainer_rs::memory_monitor::*;

println!("Before training:");
println!("  Used: {:.1}GB", total_ram() - available_ram());
println!("  Process RSS: {:.0}MB", MemoryMonitor::process_rss_mb());

// Run training
train()?;

println!("After training:");
println!("  Used: {:.1}GB", total_ram() - available_ram());
println!("  Peak observed: {:.1}GB", monitor.peak_usage_gb());
```

### GPU Memory

```bash
# Monitor during training
nvidia-smi -l 1  # Update every 1 second
```

```bash
# Get detailed GPU memory profile
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader
```

---

## Reference

### Key Constants

```
System limits:
├─ CPU RAM: 30GB hard limit
├─ CPU Warning: 25.5GB (85%)
└─ CPU OK: < 25.5GB

GPU (typical 24GB):
├─ GPU Limit: 24GB hard limit
├─ GPU Warning: 20.4GB (85%)
└─ GPU OK: < 20.4GB
```

### File Locations

- CPU monitor: `src/memory_monitor.rs`
- GPU monitor: `src/gpu_memory_monitor.rs`
- Example: `examples/memory_aware_training.rs`
- Tests: See `#[cfg(test)]` in monitor modules

### Related Features

- `gradient_accumulation.rs` - Memory-efficient gradients
- `delta_accumulator.rs` - Batch weight updates
- `vram_manager.rs` - GPU memory tracking
- `predict_aware_memory.rs` - Prediction-phase optimization

---

## Performance Impact

### Memory Monitoring Overhead

- **CPU check**: ~1ms (one `/proc/meminfo` read)
- **GPU check**: ~10ms (nvidia-smi query)
- **Recommendation**: Check every 10-50 batches

### Batch Size Reduction Impact

- **Reduced batch**: Training slower (more iterations)
- **Example**: 32→16 batch ~2× more iterations, ~10% slower overall
- **Trade-off**: Stability worth the slowdown

---

## Next Steps

1. **Run example:** `cargo run --release --example memory_aware_training --features "autodiff,datasets"`
2. **Monitor system:** `watch -n 1 free -h` in another terminal
3. **Review output:** Check peak memory and adjustments made
4. **Integrate:** Use `MemoryMonitor` in your training loop
5. **Scale:** Gradually increase batch size and model size

---

## Support

Questions? Check:
- `src/memory_monitor.rs` - CPU monitoring docs
- `src/gpu_memory_monitor.rs` - GPU monitoring docs
- `examples/memory_aware_training.rs` - Working example
- Tests in module files - Usage patterns

---

**Memory-Safe Training: Enabled ✅**
