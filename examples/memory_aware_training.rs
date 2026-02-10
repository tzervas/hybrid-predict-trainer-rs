//! Memory-Aware Training Example
//!
//! Demonstrates safe training with automatic memory monitoring
//! and batch size adjustment to keep RAM usage under 30GB.
//!
//! Usage:
//! ```bash
//! cargo run --release --example memory_aware_training --features "autodiff,datasets"
//! ```

use hybrid_predict_trainer_rs::{
    memory_monitor::{MemoryMonitor, MemoryStatus},
    models::mnist_cnn::{MnistBatch, MnistCnn, MnistCnnConfig},
};

use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{ElementConversion, Shape, Tensor, TensorData};
use rand::Rng;
use std::time::Instant;

type Backend = Autodiff<NdArray>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!("\n{}", "=".repeat(70));
    println!("🛡️  Memory-Aware Training with 30GB Limit");
    println!("{}", "=".repeat(70));

    // Print system memory status
    let total_ram = MemoryMonitor::total_ram_gb();
    let available_ram = MemoryMonitor::available_memory_gb();
    let process_rss = MemoryMonitor::process_rss_mb();

    println!("\n📊 System Memory:");
    println!("   Total RAM: {:.1}GB", total_ram);
    println!("   Available: {:.1}GB", available_ram);
    println!("   Process RSS: {:.0}MB", process_rss);
    println!("   Memory Limit: 30.0GB");

    // Create memory monitor with 30GB limit
    let mut monitor = MemoryMonitor::new();

    // Configuration
    let num_epochs = 3;
    let mut batch_size = 32;
    let num_batches_per_epoch = 50;
    let learning_rate = 0.01;

    println!("\n🔧 Training Configuration:");
    println!("   Epochs: {}", num_epochs);
    println!("   Initial Batch Size: {}", batch_size);
    println!("   Batches/Epoch: {}", num_batches_per_epoch);
    println!("   Learning Rate: {}", learning_rate);

    // Initialize model and optimizer
    let device = Default::default();
    let config = MnistCnnConfig::default();
    let mut model = MnistCnn::<Backend>::new(&config, &device);
    let mut optim = AdamConfig::new().init();

    println!("\n{}", "=".repeat(70));
    println!("🏃 Training with Memory Monitoring");
    println!("{}", "=".repeat(70));

    let start_time = Instant::now();
    let mut losses = Vec::new();
    let mut total_backward_passes = 0;
    let mut memory_adjustments = 0;

    for epoch in 0..num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;

        for batch_idx in 0..num_batches_per_epoch {
            // Check memory status before each batch
            let status = monitor.check();

            // Adjust batch size based on memory
            match status {
                MemoryStatus::Ok => {
                    if batch_size < 32 {
                        batch_size = 32;
                    }
                }
                MemoryStatus::Warning {
                    used_gb,
                    limit_gb,
                    process_mb,
                } => {
                    if batch_size > 16 {
                        batch_size = 16;
                        memory_adjustments += 1;
                        tracing::warn!(
                            "⚠️  Memory Warning: {:.1}GB/{:.1}GB - Reducing batch to {}",
                            used_gb,
                            limit_gb,
                            batch_size
                        );
                    }
                }
                MemoryStatus::ExceededLimit {
                    used_gb,
                    limit_gb,
                    process_mb,
                } => {
                    if batch_size > 4 {
                        batch_size = 4;
                        memory_adjustments += 1;
                        tracing::error!(
                            "❌ Memory Critical: {:.1}GB/{:.1}GB - Reducing batch to {}",
                            used_gb,
                            limit_gb,
                            batch_size
                        );
                    }
                }
            }

            // Generate synthetic batch
            let batch = generate_synthetic_batch(batch_size, &device);
            let (_logits, loss) = model.forward_with_loss(batch.images, batch.targets);

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads_params);

            epoch_loss += loss.into_scalar().elem::<f32>() as f64;
            total_backward_passes += 1;

            // Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 {
                let avg_loss = epoch_loss / (batch_idx + 1) as f64;
                let memory_gb = MemoryMonitor::available_memory_gb();
                let used_gb = MemoryMonitor::total_ram_gb() - memory_gb;
                println!(
                    "   Batch {}/{}: loss={:.4}, mem={:.1}GB, batch_sz={}",
                    batch_idx + 1,
                    num_batches_per_epoch,
                    avg_loss,
                    used_gb,
                    batch_size
                );
            }
        }

        let avg_loss = epoch_loss / num_batches_per_epoch as f64;
        losses.push(avg_loss);

        let epoch_time = epoch_start.elapsed().as_secs_f64();
        let peak_mem = monitor.peak_usage_gb();

        println!(
            "✅ Epoch {}/{}: loss={:.4}, time={:.1}s, peak_mem={:.1}GB",
            epoch + 1,
            num_epochs,
            avg_loss,
            epoch_time,
            peak_mem
        );
    }

    let total_time = start_time.elapsed().as_secs_f64();
    let final_loss = *losses.last().unwrap();
    let peak_memory = monitor.peak_usage_gb();

    println!("\n{}", "=".repeat(70));
    println!("📊 Training Complete");
    println!("{}", "=".repeat(70));

    println!("\n🎯 Results:");
    println!("   Final Loss: {:.4}", final_loss);
    println!("   Total Time: {:.1}s", total_time);
    println!("   Backward Passes: {}", total_backward_passes);
    println!("   Peak Memory: {:.1}GB", peak_memory);
    println!("   Memory Adjustments: {}", memory_adjustments);

    // Final system status
    let final_available = MemoryMonitor::available_memory_gb();
    let final_used = MemoryMonitor::total_ram_gb() - final_available;

    println!("\n📊 Final System Memory:");
    println!("   Used: {:.1}GB / 30.0GB limit", final_used);
    println!("   Available: {:.1}GB", final_available);
    println!("   Process RSS: {:.0}MB", MemoryMonitor::process_rss_mb());

    if final_used < 30.0 {
        println!("\n✅ SUCCESS: Memory usage stayed under 30GB limit!");
    } else {
        println!("\n⚠️  WARNING: Memory usage exceeded 30GB limit!");
    }

    println!("\n{}", "=".repeat(70));

    Ok(())
}

fn generate_synthetic_batch(
    batch_size: usize,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> MnistBatch<Backend> {
    let mut rng = rand::rng();

    let image_data: Vec<f32> = (0..batch_size * 784).map(|_| rng.random()).collect();
    let images = Tensor::from_data(
        TensorData::new(image_data, Shape::new([batch_size, 1, 28, 28])),
        device,
    );

    let label_data: Vec<i64> = (0..batch_size).map(|_| rng.random_range(0..10)).collect();
    let targets = Tensor::from_data(TensorData::new(label_data, Shape::new([batch_size])), device);

    MnistBatch { images, targets }
}
