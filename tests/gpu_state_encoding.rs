//! GPU state encoding validation tests.
//!
//! Validates that GPU state encoding produces identical results to CPU implementation.

#[cfg(all(feature = "cuda", feature = "candle"))]
mod state_encoding_tests {
    use burn_cuda::{Cuda, CudaDevice};
    use hybrid_predict_trainer_rs::gpu::kernels::state_encode::{
        encode_state_cpu, encode_state_gpu, StateDataFlat, StateEncodeConfig,
    };
    use hybrid_predict_trainer_rs::state::{OptimizerStateSummary, TrainingState};
    use hybrid_predict_trainer_rs::Phase;

    fn create_realistic_state() -> TrainingState {
        let mut state = TrainingState::default();

        // Fill with realistic training data
        state.loss = 2.5;
        state.gradient_norm = 0.15;
        state.step = 1000;
        state.phase_step = 50;
        state.current_phase = Phase::Predict;

        // Add loss history
        for i in 0..100 {
            state.loss_history.push(3.0 - (i as f32 * 0.005));
        }

        // Add gradient history
        for i in 0..100 {
            state
                .gradient_norm_history
                .push(0.2 - (i as f32 * 0.0001));
        }

        // Set optimizer state
        state.optimizer_state_summary = OptimizerStateSummary {
            momentum_mean: 0.01,
            momentum_std: 0.005,
            variance_mean: 0.001,
            variance_std: 0.0005,
            effective_lr: 0.001,
            beta1_power: 0.9_f32.powi(1000),
            beta2_power: 0.999_f32.powi(1000),
        };

        state
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_gpu_state_encode_output_dim() {
        let state = create_realistic_state();
        let flat = StateDataFlat::from_state(&state);
        let config = StateEncodeConfig::default();
        let device = CudaDevice::default();

        let features = encode_state_gpu::<Cuda>(&config, &flat, &device).unwrap();

        assert_eq!(features.len(), 64, "Must output exactly 64 features");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_gpu_vs_cpu_correctness() {
        let state = create_realistic_state();
        let flat = StateDataFlat::from_state(&state);
        let config = StateEncodeConfig::default();
        let device = CudaDevice::default();

        // CPU encoding
        let cpu_features = encode_state_cpu(&state);

        // GPU encoding
        let gpu_features = encode_state_gpu::<Cuda>(&config, &flat, &device).unwrap();

        // Both should be 64-dim
        assert_eq!(cpu_features.len(), 64);
        assert_eq!(gpu_features.len(), 64);

        // Compare element-wise
        let max_error = cpu_features
            .iter()
            .zip(gpu_features.iter())
            .enumerate()
            .map(|(idx, (cpu, gpu))| {
                let err = (cpu - gpu).abs();
                if err > 1e-4 {
                    println!(
                        "Feature {}: CPU={:.6}, GPU={:.6}, diff={:.6}",
                        idx, cpu, gpu, err
                    );
                }
                err
            })
            .fold(0.0_f32, f32::max);

        assert!(
            max_error < 1e-4,
            "GPU vs CPU max error {} exceeds 1e-4",
            max_error
        );

        println!("✅ GPU state encoding matches CPU (max error: {:.2e})", max_error);
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_gpu_state_encode_deterministic() {
        let state = create_realistic_state();
        let flat = StateDataFlat::from_state(&state);
        let config = StateEncodeConfig::default();
        let device = CudaDevice::default();

        let features1 = encode_state_gpu::<Cuda>(&config, &flat, &device).unwrap();
        let features2 = encode_state_gpu::<Cuda>(&config, &flat, &device).unwrap();

        assert_eq!(features1.len(), features2.len());

        for (idx, (f1, f2)) in features1.iter().zip(features2.iter()).enumerate() {
            assert_eq!(
                f1, f2,
                "Feature {} differs between runs: {} vs {}",
                idx, f1, f2
            );
        }

        println!("✅ GPU state encoding is deterministic");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_gpu_state_encode_empty_history() {
        let state = TrainingState::default();
        let flat = StateDataFlat::from_state(&state);
        let config = StateEncodeConfig::default();
        let device = CudaDevice::default();

        let features = encode_state_gpu::<Cuda>(&config, &flat, &device).unwrap();

        assert_eq!(features.len(), 64);
        println!("✅ GPU state encoding handles empty history");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_gpu_state_encode_first_feature_is_loss() {
        let state = create_realistic_state();
        let flat = StateDataFlat::from_state(&state);
        let config = StateEncodeConfig::default();
        let device = CudaDevice::default();

        let features = encode_state_gpu::<Cuda>(&config, &flat, &device).unwrap();

        assert_eq!(
            features[0], state.loss,
            "First feature should be current loss"
        );
        println!("✅ First feature correctly set to current loss");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_gpu_state_encode_all_phases() {
        let config = StateEncodeConfig::default();
        let device = CudaDevice::default();

        for (phase_idx, phase) in [
            Phase::Warmup,
            Phase::Full,
            Phase::Predict,
            Phase::Correct,
        ]
        .iter()
        .enumerate()
        {
            let mut state = create_realistic_state();
            state.current_phase = *phase;

            let flat = StateDataFlat::from_state(&state);

            // Get both CPU and GPU features for comparison
            let cpu_features = encode_state_cpu(&state);
            let gpu_features = encode_state_gpu::<Cuda>(&config, &flat, &device).unwrap();

            assert_eq!(gpu_features.len(), 64);

            // Phase encoding is at index 24 (after 8 loss + 8 gradient + 8 optimizer features)
            let cpu_phase_feature = cpu_features[24];
            let gpu_phase_feature = gpu_features[24];

            assert_eq!(
                cpu_phase_feature, phase_idx as f32,
                "CPU: Phase {:?} should encode to {}",
                phase, phase_idx
            );
            assert_eq!(
                gpu_phase_feature, phase_idx as f32,
                "GPU: Phase {:?} should encode to {}",
                phase, phase_idx
            );
            assert_eq!(
                gpu_phase_feature, cpu_phase_feature,
                "GPU should match CPU for phase {:?}",
                phase
            );
        }

        println!("✅ GPU state encoding handles all phases correctly");
    }

    #[test]
    fn test_state_encode_config_validation() {
        let valid = StateEncodeConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = StateEncodeConfig {
            feature_dim: 32, // Wrong dimension
            max_history: 1000,
        };
        assert!(invalid.validate().is_err());
    }
}
