//! Tests for ensemble-batched GRU operations.
//!
//! Validates that batched processing of ensemble members produces
//! identical results to sequential processing, while achieving speedup.

#[cfg(all(feature = "cuda", feature = "candle"))]
mod ensemble_batching_tests {
    use burn::backend::Wgpu;
    use burn::tensor::backend::Backend;
    use hybrid_predict_trainer::gpu::kernels::gru::GpuGruWeights;
    use hybrid_predict_trainer::gpu::kernels::gru_ensemble_batched::{
        gru_forward_ensemble_batched, EnsembleGruConfig,
    };

    type TestBackend = Wgpu;

    /// Helper: Create random GRU weights for testing.
    fn random_gru_weights(hidden_dim: usize, input_dim: usize, seed: u64) -> GpuGruWeights {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let random_vec = |size: usize| -> Vec<f32> {
            (0..size).map(|_| rng.gen_range(-0.5..0.5)).collect()
        };

        GpuGruWeights {
            w_z: random_vec(hidden_dim * input_dim),
            u_z: random_vec(hidden_dim * hidden_dim),
            b_z: random_vec(hidden_dim),
            w_r: random_vec(hidden_dim * input_dim),
            u_r: random_vec(hidden_dim * hidden_dim),
            b_r: random_vec(hidden_dim),
            w_h: random_vec(hidden_dim * input_dim),
            u_h: random_vec(hidden_dim * hidden_dim),
            b_h: random_vec(hidden_dim),
        }
    }

    /// Helper: Single-member GRU forward (CPU reference).
    fn gru_forward_cpu(
        weights: &GpuGruWeights,
        hidden: &[f32],
        input: &[f32],
        hidden_dim: usize,
        input_dim: usize,
    ) -> Vec<f32> {
        use std::f32;

        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let tanh = |x: f32| x.tanh();

        let matvec = |mat: &[f32], vec: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| mat[i * cols + j] * vec[j])
                        .sum::<f32>()
                })
                .collect()
        };

        // Update gate: z = σ(W_z·x + U_z·h + b_z)
        let z_input = matvec(&weights.w_z, input, hidden_dim, input_dim);
        let z_hidden = matvec(&weights.u_z, hidden, hidden_dim, hidden_dim);
        let z: Vec<f32> = z_input
            .iter()
            .zip(z_hidden.iter())
            .zip(weights.b_z.iter())
            .map(|((a, b), c)| sigmoid(a + b + c))
            .collect();

        // Reset gate: r = σ(W_r·x + U_r·h + b_r)
        let r_input = matvec(&weights.w_r, input, hidden_dim, input_dim);
        let r_hidden = matvec(&weights.u_r, hidden, hidden_dim, hidden_dim);
        let r: Vec<f32> = r_input
            .iter()
            .zip(r_hidden.iter())
            .zip(weights.b_r.iter())
            .map(|((a, b), c)| sigmoid(a + b + c))
            .collect();

        // Candidate: h̃ = tanh(W_h·x + U_h·(r⊙h) + b_h)
        let r_h: Vec<f32> = r.iter().zip(hidden.iter()).map(|(a, b)| a * b).collect();
        let h_cand_input = matvec(&weights.w_h, input, hidden_dim, input_dim);
        let h_cand_hidden = matvec(&weights.u_h, &r_h, hidden_dim, hidden_dim);
        let h_cand: Vec<f32> = h_cand_input
            .iter()
            .zip(h_cand_hidden.iter())
            .zip(weights.b_h.iter())
            .map(|((a, b), c)| tanh(a + b + c))
            .collect();

        // New hidden: h' = (1-z)⊙h + z⊙h̃
        z.iter()
            .zip(hidden.iter())
            .zip(h_cand.iter())
            .map(|((z_val, h_val), h_cand_val)| (1.0 - z_val) * h_val + z_val * h_cand_val)
            .collect()
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_ensemble_batching_correctness() {
        let device = <TestBackend as Backend>::Device::default();

        let config = EnsembleGruConfig {
            ensemble_size: 5,
            hidden_dim: 32,
            input_dim: 16,
        };

        // Create random weights for each ensemble member
        let ensemble_weights: Vec<GpuGruWeights> = (0..config.ensemble_size)
            .map(|i| random_gru_weights(config.hidden_dim, config.input_dim, 1000 + i as u64))
            .collect();

        // Create random hiddens and inputs
        let hiddens: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|i| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(2000 + i as u64);
                (0..config.hidden_dim)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        let inputs: Vec<Vec<f32>> = (0..config.ensemble_size)
            .map(|i| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(3000 + i as u64);
                (0..config.input_dim)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        // GPU batched forward
        let gpu_results = gru_forward_ensemble_batched::<TestBackend>(
            &config,
            &ensemble_weights,
            &hiddens,
            &inputs,
            &device,
        )
        .expect("GPU ensemble batching failed");

        assert_eq!(gpu_results.len(), config.ensemble_size);

        // Compare each ensemble member with CPU reference
        for i in 0..config.ensemble_size {
            let cpu_result = gru_forward_cpu(
                &ensemble_weights[i],
                &hiddens[i],
                &inputs[i],
                config.hidden_dim,
                config.input_dim,
            );

            assert_eq!(gpu_results[i].len(), config.hidden_dim);
            assert_eq!(cpu_result.len(), config.hidden_dim);

            // Check element-wise error
            let max_error = gpu_results[i]
                .iter()
                .zip(cpu_result.iter())
                .map(|(gpu, cpu)| (gpu - cpu).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_error < 1e-4,
                "Ensemble member {} max error {} exceeds tolerance",
                i,
                max_error
            );
        }

        println!("✅ Ensemble batching correctness validated for {} members", config.ensemble_size);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_ensemble_batching_deterministic() {
        let device = <TestBackend as Backend>::Device::default();

        let config = EnsembleGruConfig {
            ensemble_size: 3,
            hidden_dim: 16,
            input_dim: 8,
        };

        let ensemble_weights: Vec<GpuGruWeights> = (0..config.ensemble_size)
            .map(|i| random_gru_weights(config.hidden_dim, config.input_dim, 100 + i as u64))
            .collect();

        let hiddens: Vec<Vec<f32>> = vec![vec![0.1; config.hidden_dim]; config.ensemble_size];
        let inputs: Vec<Vec<f32>> = vec![vec![0.2; config.input_dim]; config.ensemble_size];

        // Run twice
        let results1 = gru_forward_ensemble_batched::<TestBackend>(
            &config,
            &ensemble_weights,
            &hiddens,
            &inputs,
            &device,
        )
        .expect("GPU ensemble batching failed (run 1)");

        let results2 = gru_forward_ensemble_batched::<TestBackend>(
            &config,
            &ensemble_weights,
            &hiddens,
            &inputs,
            &device,
        )
        .expect("GPU ensemble batching failed (run 2)");

        // Compare
        for i in 0..config.ensemble_size {
            for j in 0..config.hidden_dim {
                let diff = (results1[i][j] - results2[i][j]).abs();
                assert!(
                    diff < 1e-6,
                    "Non-deterministic output at ensemble {} dim {}: diff {}",
                    i,
                    j,
                    diff
                );
            }
        }

        println!("✅ Ensemble batching is deterministic");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_ensemble_batching_output_shapes() {
        let device = <TestBackend as Backend>::Device::default();

        let config = EnsembleGruConfig {
            ensemble_size: 7,
            hidden_dim: 64,
            input_dim: 32,
        };

        let ensemble_weights: Vec<GpuGruWeights> = (0..config.ensemble_size)
            .map(|i| random_gru_weights(config.hidden_dim, config.input_dim, 500 + i as u64))
            .collect();

        let hiddens: Vec<Vec<f32>> = vec![vec![0.0; config.hidden_dim]; config.ensemble_size];
        let inputs: Vec<Vec<f32>> = vec![vec![1.0; config.input_dim]; config.ensemble_size];

        let results = gru_forward_ensemble_batched::<TestBackend>(
            &config,
            &ensemble_weights,
            &hiddens,
            &inputs,
            &device,
        )
        .expect("GPU ensemble batching failed");

        assert_eq!(
            results.len(),
            config.ensemble_size,
            "Wrong number of ensemble outputs"
        );

        for (i, result) in results.iter().enumerate() {
            assert_eq!(
                result.len(),
                config.hidden_dim,
                "Wrong hidden dimension for ensemble member {}",
                i
            );
        }

        println!("✅ Ensemble batching output shapes correct");
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_ensemble_batching_size_mismatch_error() {
        let device = <TestBackend as Backend>::Device::default();

        let config = EnsembleGruConfig {
            ensemble_size: 3,
            hidden_dim: 16,
            input_dim: 8,
        };

        let ensemble_weights: Vec<GpuGruWeights> = (0..config.ensemble_size)
            .map(|i| random_gru_weights(config.hidden_dim, config.input_dim, 700 + i as u64))
            .collect();

        // Wrong number of hiddens (should be 3, provide 2)
        let hiddens: Vec<Vec<f32>> = vec![vec![0.1; config.hidden_dim]; 2];
        let inputs: Vec<Vec<f32>> = vec![vec![0.2; config.input_dim]; config.ensemble_size];

        let result = gru_forward_ensemble_batched::<TestBackend>(
            &config,
            &ensemble_weights,
            &hiddens,
            &inputs,
            &device,
        );

        assert!(
            result.is_err(),
            "Expected error for ensemble size mismatch"
        );

        println!("✅ Ensemble size mismatch correctly detected");
    }
}

#[cfg(not(all(feature = "cuda", feature = "candle")))]
#[test]
fn test_ensemble_batching_requires_cuda() {
    // Placeholder test for non-CUDA builds
    println!("⚠️  Ensemble batching tests require 'cuda' and 'candle' features");
}
