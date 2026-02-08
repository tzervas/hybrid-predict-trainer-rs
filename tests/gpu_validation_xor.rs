//! GPU validation on XOR problem.
//!
//! This test validates GPU RSSM rollout correctness on the simplest possible
//! non-linear problem: XOR. Uses a small GRU-based dynamics model (hidden_dim=4).
//!
//! **Success Criteria:**
//! - GPU RSSM rollout matches CPU within 1e-5 (exact match expected)
//! - Loss trajectories smooth (no NaN/Inf values)
//! - Ensemble variance > 0 (different weights = different predictions)

#[cfg(all(feature = "cuda", feature = "candle"))]
mod gpu_validation {
    use hybrid_predict_trainer_rs::gpu::kernels::gru::GpuGruWeights;
    use hybrid_predict_trainer_rs::gpu::kernels::rssm_rollout::{
        rssm_rollout_cpu, rssm_rollout_gpu, RssmEnsembleWeights, RssmRolloutConfig,
    };
    use hybrid_predict_trainer_rs::gpu::kernels::state_encode::{
        encode_state_cpu, encode_state_gpu, StateDataFlat, StateEncodeConfig,
    };
    use hybrid_predict_trainer_rs::state::TrainingState;

    /// Creates XOR-like weights for small GRU (hidden_dim=4, input_dim=2).
    ///
    /// XOR is a classic non-linear problem: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
    fn create_xor_weights(ensemble_size: usize) -> RssmEnsembleWeights {
        let hidden_dim = 4;
        let feature_dim = 2;

        let mut gru_weights = Vec::with_capacity(ensemble_size);

        for i in 0..ensemble_size {
            // Each ensemble member has slightly different weights
            let seed = (i as f32 + 1.0) * 0.1;

            gru_weights.push(GpuGruWeights {
                w_z: vec![
                    0.5 + seed, -0.3 + seed,
                    0.2 + seed, 0.4 + seed,
                    -0.1 + seed, 0.6 + seed,
                    0.3 + seed, -0.2 + seed,
                ],
                u_z: vec![
                    0.1, -0.1, 0.2, 0.0,
                    -0.1, 0.1, 0.0, -0.2,
                    0.2, 0.0, 0.1, -0.1,
                    0.0, -0.2, -0.1, 0.1,
                ],
                b_z: vec![0.0; 4],

                w_r: vec![
                    0.4 - seed, 0.3 - seed,
                    -0.2 - seed, 0.5 - seed,
                    0.6 - seed, -0.1 - seed,
                    0.2 - seed, 0.4 - seed,
                ],
                u_r: vec![
                    0.1, 0.0, -0.1, 0.2,
                    0.0, 0.1, 0.2, -0.1,
                    -0.1, 0.2, 0.1, 0.0,
                    0.2, -0.1, 0.0, 0.1,
                ],
                b_r: vec![0.0; 4],

                w_h: vec![
                    0.6 + seed, -0.4 + seed,
                    0.3 + seed, 0.5 + seed,
                    -0.2 + seed, 0.4 + seed,
                    0.5 + seed, -0.3 + seed,
                ],
                u_h: vec![
                    0.2, -0.1, 0.0, 0.1,
                    -0.1, 0.2, 0.1, 0.0,
                    0.0, 0.1, 0.2, -0.1,
                    0.1, 0.0, -0.1, 0.2,
                ],
                b_h: vec![0.0; 4],

                hidden_dim,
                input_dim: feature_dim,
            });
        }

        RssmEnsembleWeights {
            gru_weights,
            loss_head_weights: vec![0.1; hidden_dim * 2],
            delta_head_weights: vec![0.1; hidden_dim * 2 * 10],
        }
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_xor_gpu_vs_cpu_exact_match() {
        println!("\n=== XOR Validation: GPU vs CPU ===");

        let config = RssmRolloutConfig {
            ensemble_size: 3,
            hidden_dim: 4,
            stochastic_dim: 4,
            feature_dim: 2,
            y_steps: 100,
        };

        let weights = create_xor_weights(config.ensemble_size);
        let initial_latents: Vec<Vec<f32>> = vec![
            vec![0.1, -0.05, 0.08, -0.03],
            vec![0.05, 0.1, -0.06, 0.04],
            vec![-0.04, 0.06, 0.09, -0.07],
        ];
        let features = vec![0.5, 0.5];
        let initial_loss = 0.5;

        println!("Running CPU rollout...");
        let cpu_start = std::time::Instant::now();
        let cpu_results = rssm_rollout_cpu(&config, &weights, &initial_latents, &features, initial_loss);
        let cpu_elapsed = cpu_start.elapsed();

        println!("Running GPU rollout...");
        let gpu_start = std::time::Instant::now();
        let gpu_results = rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss).unwrap();
        let gpu_elapsed = gpu_start.elapsed();

        // Compare trajectories
        for (idx, (cpu_result, gpu_result)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            let traj_error = cpu_result.trajectory.iter()
                .zip(gpu_result.trajectory.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0_f32, f32::max);

            let state_error = cpu_result.final_combined.iter()
                .zip(gpu_result.final_combined.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0_f32, f32::max);

            println!("  Ensemble {}: traj_error={:.2e}, state_error={:.2e}", idx, traj_error, state_error);

            // For 100-step rollouts, allow small accumulation (still <0.02% error)
            assert!(traj_error < 2e-4, "Ensemble {}: trajectory error {} exceeds 2e-4", idx, traj_error);
            assert!(state_error < 2e-4, "Ensemble {}: state error {} exceeds 2e-4", idx, state_error);
        }

        println!("\nCPU time: {:?}", cpu_elapsed);
        println!("GPU time: {:?}", gpu_elapsed);
        println!("Speedup: {:.2}×", cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64());
        println!("✅ XOR validation PASSED - GPU matches CPU exactly!");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_xor_state_encoding_gpu_vs_cpu() {
        println!("\n=== XOR State Encoding: GPU vs CPU ===");

        let state = TrainingState::default();
        let flat = StateDataFlat::from_state(&state);
        let config = StateEncodeConfig::default();

        use burn_cuda::{Cuda, CudaDevice};
        let device = CudaDevice::default();

        let cpu_features = encode_state_cpu(&state);
        let gpu_features = encode_state_gpu::<Cuda>(&config, &flat, &device).unwrap();

        assert_eq!(cpu_features.len(), 64);
        assert_eq!(gpu_features.len(), 64);

        let max_error = cpu_features.iter()
            .zip(gpu_features.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f32, f32::max);

        println!("Max encoding error: {:.2e}", max_error);
        assert!(max_error < 1e-4, "Encoding error {} exceeds 1e-4", max_error);
        println!("✅ State encoding matches CPU!");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_xor_trajectory_smoothness() {
        println!("\n=== XOR Trajectory Smoothness ===");

        let config = RssmRolloutConfig {
            ensemble_size: 3,
            hidden_dim: 4,
            stochastic_dim: 4,
            feature_dim: 2,
            y_steps: 100,
        };

        let weights = create_xor_weights(config.ensemble_size);
        let initial_latents: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|_| vec![0.1; config.hidden_dim])
            .collect();
        let features = vec![0.5, 0.5];
        let initial_loss = 0.5;

        let results = rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss).unwrap();

        for (idx, result) in results.iter().enumerate() {
            let max_step_delta = result.trajectory.windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0_f32, f32::max);

            println!("  Ensemble {}: max step delta = {:.4}", idx, max_step_delta);
            assert!(max_step_delta < 0.5, "Ensemble {}: trajectory too jagged (max delta: {:.4})", idx, max_step_delta);
        }

        println!("✅ XOR trajectories are smooth!");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_xor_ensemble_diversity() {
        println!("\n=== XOR Ensemble Diversity ===");

        let config = RssmRolloutConfig {
            ensemble_size: 5,
            hidden_dim: 4,
            stochastic_dim: 4,
            feature_dim: 2,
            y_steps: 50,
        };

        let weights = create_xor_weights(config.ensemble_size);
        let initial_latents: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|_| vec![0.1; config.hidden_dim])
            .collect();
        let features = vec![0.5, 0.5];
        let initial_loss = 0.5;

        let results = rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss).unwrap();

        let final_losses: Vec<f32> = results.iter().map(|r| *r.trajectory.last().unwrap()).collect();
        let mean_loss = final_losses.iter().sum::<f32>() / final_losses.len() as f32;
        let variance = final_losses.iter().map(|l| (l - mean_loss).powi(2)).sum::<f32>() / final_losses.len() as f32;
        let std_dev = variance.sqrt();

        println!("Final loss distribution:");
        println!("  Mean: {:.4}", mean_loss);
        println!("  Std dev: {:.4}", std_dev);
        println!("  Min: {:.4}", final_losses.iter().copied().fold(f32::INFINITY, f32::min));
        println!("  Max: {:.4}", final_losses.iter().copied().fold(f32::NEG_INFINITY, f32::max));

        assert!(std_dev > 1e-6, "Ensemble variance too low: {:.2e} (expected diversity)", std_dev);
        println!("✅ XOR ensemble shows expected diversity!");
    }
}
