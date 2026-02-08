//! GPU RSSM rollout validation tests.
//!
//! Validates that GPU RSSM rollout produces correct multi-step predictions.

#[cfg(all(feature = "cuda", feature = "candle"))]
mod rssm_tests {
    use hybrid_predict_trainer_rs::gpu::kernels::gru::GpuGruWeights;
    use hybrid_predict_trainer_rs::gpu::kernels::rssm_rollout::{
        rssm_rollout_cpu, rssm_rollout_gpu, RssmEnsembleWeights, RssmRolloutConfig,
    };

    fn create_test_weights(ensemble_size: usize, hidden_dim: usize, feature_dim: usize) -> RssmEnsembleWeights {
        let mut gru_weights = Vec::with_capacity(ensemble_size);

        for i in 0..ensemble_size {
            let scale = 0.01 * (i as f32 + 1.0);
            gru_weights.push(GpuGruWeights {
                w_z: (0..hidden_dim * feature_dim)
                    .map(|j| scale * ((j as f32 * 0.1).sin()))
                    .collect(),
                u_z: (0..hidden_dim * hidden_dim)
                    .map(|j| scale * ((j as f32 * 0.1).cos()))
                    .collect(),
                b_z: vec![0.0; hidden_dim],
                w_r: (0..hidden_dim * feature_dim)
                    .map(|j| scale * ((j as f32 * 0.2).sin()))
                    .collect(),
                u_r: (0..hidden_dim * hidden_dim)
                    .map(|j| scale * ((j as f32 * 0.2).cos()))
                    .collect(),
                b_r: vec![0.0; hidden_dim],
                w_h: (0..hidden_dim * feature_dim)
                    .map(|j| scale * ((j as f32 * 0.3).sin()))
                    .collect(),
                u_h: (0..hidden_dim * hidden_dim)
                    .map(|j| scale * ((j as f32 * 0.3).cos()))
                    .collect(),
                b_h: vec![0.0; hidden_dim],
                hidden_dim,
                input_dim: feature_dim,
            });
        }

        RssmEnsembleWeights {
            gru_weights,
            loss_head_weights: vec![0.01; hidden_dim * 2], // combined_dim = 2 × hidden_dim
            delta_head_weights: vec![0.01; hidden_dim * 2 * 10],
        }
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_rssm_rollout_gpu_vs_cpu() {
        let config = RssmRolloutConfig {
            ensemble_size: 3,
            hidden_dim: 64,
            stochastic_dim: 64,
            feature_dim: 16,
            y_steps: 10,
        };

        let weights = create_test_weights(config.ensemble_size, config.hidden_dim, config.feature_dim);
        let initial_latents: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|_| vec![0.5; config.hidden_dim])
            .collect();
        let features = vec![1.0; config.feature_dim];
        let initial_loss = 2.5;

        // CPU rollout
        let cpu_results = rssm_rollout_cpu(&config, &weights, &initial_latents, &features, initial_loss);

        // GPU rollout
        let gpu_results =
            rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss).unwrap();

        assert_eq!(cpu_results.len(), gpu_results.len());

        // Compare trajectories for each ensemble member
        for (idx, (cpu_result, gpu_result)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            assert_eq!(
                cpu_result.trajectory.len(),
                gpu_result.trajectory.len(),
                "Ensemble {}: trajectory length mismatch",
                idx
            );

            // Compare loss trajectories
            let max_traj_error = cpu_result
                .trajectory
                .iter()
                .zip(gpu_result.trajectory.iter())
                .map(|(cpu, gpu)| (cpu - gpu).abs())
                .fold(0.0_f32, f32::max);

            assert!(
                max_traj_error < 1e-3,
                "Ensemble {}: trajectory max error {} exceeds 1e-3",
                idx,
                max_traj_error
            );

            // Compare final combined states
            assert_eq!(
                cpu_result.final_combined.len(),
                gpu_result.final_combined.len(),
                "Ensemble {}: combined state length mismatch",
                idx
            );

            let max_state_error = cpu_result
                .final_combined
                .iter()
                .zip(gpu_result.final_combined.iter())
                .map(|(cpu, gpu)| (cpu - gpu).abs())
                .fold(0.0_f32, f32::max);

            assert!(
                max_state_error < 1e-3,
                "Ensemble {}: final state max error {} exceeds 1e-3",
                idx,
                max_state_error
            );

            println!(
                "✅ Ensemble {}: traj_error={:.2e}, state_error={:.2e}",
                idx, max_traj_error, max_state_error
            );
        }

        println!("\n✅ GPU RSSM rollout matches CPU for {} ensemble members × {} steps",
            config.ensemble_size, config.y_steps);
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_rssm_rollout_gpu_output_shapes() {
        let config = RssmRolloutConfig {
            ensemble_size: 5,
            hidden_dim: 256,
            stochastic_dim: 256,
            feature_dim: 64,
            y_steps: 50,
        };

        let weights = create_test_weights(config.ensemble_size, config.hidden_dim, config.feature_dim);
        let initial_latents: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|_| vec![0.5; config.hidden_dim])
            .collect();
        let features = vec![1.0; config.feature_dim];
        let initial_loss = 2.5;

        let results =
            rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss).unwrap();

        assert_eq!(results.len(), config.ensemble_size);

        for (idx, result) in results.iter().enumerate() {
            assert_eq!(
                result.trajectory.len(),
                config.y_steps + 1,
                "Ensemble {}: trajectory should have y_steps + 1 elements",
                idx
            );

            assert_eq!(
                result.final_combined.len(),
                config.combined_dim(),
                "Ensemble {}: final combined state dimension mismatch",
                idx
            );

            assert_eq!(
                result.trajectory[0], initial_loss,
                "Ensemble {}: first trajectory element should be initial loss",
                idx
            );
        }

        println!("✅ GPU RSSM rollout produces correct output shapes");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_rssm_rollout_gpu_deterministic() {
        let config = RssmRolloutConfig {
            ensemble_size: 3,
            hidden_dim: 128,
            stochastic_dim: 128,
            feature_dim: 32,
            y_steps: 20,
        };

        let weights = create_test_weights(config.ensemble_size, config.hidden_dim, config.feature_dim);
        let initial_latents: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|_| vec![0.5; config.hidden_dim])
            .collect();
        let features = vec![1.0; config.feature_dim];
        let initial_loss = 2.5;

        // Run twice with same inputs
        let results1 =
            rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss).unwrap();
        let results2 =
            rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss).unwrap();

        // Should be deterministic
        for (idx, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            for (step, (t1, t2)) in r1.trajectory.iter().zip(r2.trajectory.iter()).enumerate() {
                assert!(
                    (t1 - t2).abs() < 1e-6,
                    "Ensemble {}, step {}: non-deterministic result",
                    idx,
                    step
                );
            }
        }

        println!("✅ GPU RSSM rollout is deterministic");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_rssm_rollout_gpu_small_config() {
        // Test with small dimensions to verify edge cases
        let config = RssmRolloutConfig {
            ensemble_size: 2,
            hidden_dim: 16,
            stochastic_dim: 16,
            feature_dim: 8,
            y_steps: 5,
        };

        let weights = create_test_weights(config.ensemble_size, config.hidden_dim, config.feature_dim);
        let initial_latents: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|_| vec![0.5; config.hidden_dim])
            .collect();
        let features = vec![1.0; config.feature_dim];
        let initial_loss = 1.5;

        let results =
            rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].trajectory.len(), 6); // y_steps + 1

        println!("✅ GPU RSSM rollout handles small configurations");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_rssm_rollout_gpu_single_step() {
        // Test y_steps=1 edge case
        let config = RssmRolloutConfig {
            ensemble_size: 3,
            hidden_dim: 64,
            stochastic_dim: 64,
            feature_dim: 16,
            y_steps: 1,
        };

        let weights = create_test_weights(config.ensemble_size, config.hidden_dim, config.feature_dim);
        let initial_latents: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|_| vec![0.5; config.hidden_dim])
            .collect();
        let features = vec![1.0; config.feature_dim];
        let initial_loss = 2.0;

        let results =
            rssm_rollout_gpu(&config, &weights, &initial_latents, &features, initial_loss).unwrap();

        for result in &results {
            assert_eq!(result.trajectory.len(), 2); // initial + 1 step
            assert_eq!(result.trajectory[0], initial_loss);
        }

        println!("✅ GPU RSSM rollout handles single-step correctly");
    }
}
