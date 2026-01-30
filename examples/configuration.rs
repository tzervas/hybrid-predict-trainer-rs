//! Configuration examples.
//!
//! Demonstrates different configuration patterns for the hybrid trainer.

use hybrid_predict_trainer_rs::config::{
    DivergenceConfig, HybridTrainerConfig, PredictorConfig,
};

fn main() {
    println!("=== Hybrid Trainer Configuration Examples ===\n");
    
    // Default configuration
    println!("1. Default configuration:");
    let default_config = HybridTrainerConfig::default();
    print_config(&default_config);
    
    // Conservative configuration (safer, less speedup)
    println!("\n2. Conservative configuration:");
    let conservative = HybridTrainerConfig::builder()
        .warmup_steps(500)
        .min_full_steps(50)
        .max_predict_length(20)
        .confidence_threshold(0.95)
        .max_loss_gap(0.01) // 1% max gap
        .build()
        .expect("Failed to build config");
    print_config(&conservative);
    
    // Aggressive configuration (more speedup, higher risk)
    println!("\n3. Aggressive configuration:");
    let aggressive = HybridTrainerConfig::builder()
        .warmup_steps(100)
        .min_full_steps(10)
        .max_predict_length(100)
        .confidence_threshold(0.75)
        .max_loss_gap(0.05) // 5% max gap
        .build()
        .expect("Failed to build config");
    print_config(&aggressive);
    
    // RSSM predictor configuration
    println!("\n4. RSSM predictor configuration:");
    let rssm_config = HybridTrainerConfig::builder()
        .predictor_config(PredictorConfig::RSSM {
            deterministic_dim: 512,
            stochastic_dim: 64,
            num_categoricals: 32,
            ensemble_size: 5,
        })
        .build()
        .expect("Failed to build config");
    println!("  Predictor: {:?}", rssm_config.predictor_config);
    
    // Custom divergence thresholds
    println!("\n5. Custom divergence thresholds:");
    let divergence_config = DivergenceConfig {
        loss_sigma_threshold: 2.0,
        gradient_norm_multiplier: 5.0,
        vanishing_gradient_threshold: 0.001,
    };
    let custom_divergence = HybridTrainerConfig::builder()
        .divergence_config(divergence_config)
        .build()
        .expect("Failed to build config");
    println!(
        "  Loss sigma threshold: {}",
        custom_divergence.divergence_config.loss_sigma_threshold
    );
    println!(
        "  Gradient norm multiplier: {}",
        custom_divergence.divergence_config.gradient_norm_multiplier
    );
}

fn print_config(config: &HybridTrainerConfig) {
    println!("  warmup_steps: {}", config.warmup_steps);
    println!("  min_full_steps: {}", config.min_full_steps);
    println!("  max_predict_length: {}", config.max_predict_length);
    println!("  confidence_threshold: {}", config.confidence_threshold);
    println!("  max_loss_gap: {}", config.max_loss_gap);
}
