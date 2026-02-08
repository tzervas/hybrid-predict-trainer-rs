//! Hugging Face Hub integration for model checkpointing.
//!
//! Provides utilities for uploading trained models and checkpoints to Hugging Face Hub,
//! enabling reproducibility and sharing of hybrid training results.
//!
//! # Features
//!
//! - Model repository creation
//! - Checkpoint upload with metadata
//! - Training metrics export
//! - Model card generation
//!
//! # Usage
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::hf_integration::*;
//!
//! let uploader = HfUploader::new("username/model-name", token)?;
//! uploader.create_repo(RepoVisibility::Public)?;
//! uploader.upload_checkpoint("checkpoint.safetensors", metadata)?;
//! uploader.upload_training_results(&comparison_results)?;
//! ```

use crate::error::{HybridResult, HybridTrainingError};
use crate::training_comparison::ComparisonResults;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Hugging Face repository visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepoVisibility {
    /// Public repository.
    Public,
    /// Private repository.
    Private,
}

/// Hugging Face model uploader.
#[derive(Debug, Clone)]
pub struct HfUploader {
    /// Repository ID (format: "username/repo-name").
    repo_id: String,

    /// Authentication token.
    token: String,

    /// Base API URL.
    api_url: String,
}

impl HfUploader {
    /// Creates a new Hugging Face uploader.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID in format "username/repo-name"
    /// * `token` - HF authentication token
    pub fn new(repo_id: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            token: token.into(),
            api_url: "https://huggingface.co".to_string(),
        }
    }

    /// Creates a new repository on Hugging Face Hub.
    ///
    /// # Arguments
    ///
    /// * `visibility` - Repository visibility (public/private)
    ///
    /// # Errors
    ///
    /// Returns error if repository creation fails.
    #[cfg(feature = "datasets")]
    pub fn create_repo(&self, visibility: RepoVisibility) -> HybridResult<()> {
        let url = format!("{}/api/repos/create", self.api_url);
        let private = matches!(visibility, RepoVisibility::Private);

        let json_body = serde_json::json!({
            "type": "model",
            "name": self.repo_id.split('/').last().unwrap(),
            "private": private,
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.token))
            .set("Content-Type", "application/json")
            .send_string(&json_body.to_string())
            .map_err(|e| {
                (
                    HybridTrainingError::ConfigError {
                        detail: format!("Failed to create HF repository: {}", e),
                    },
                    None,
                )
            })?;

        if response.status() == 201 || response.status() == 200 {
            tracing::info!(repo_id = %self.repo_id, "Created Hugging Face repository");
            Ok(())
        } else if response.status() == 409 {
            tracing::info!(repo_id = %self.repo_id, "Repository already exists");
            Ok(())
        } else {
            Err((
                HybridTrainingError::ConfigError {
                    detail: format!("Unexpected status: {}", response.status()),
                },
                None,
            ))
        }
    }

    /// Uploads a file to the repository.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Local file path
    /// * `repo_path` - Path within repository
    ///
    /// # Errors
    ///
    /// Returns error if upload fails.
    #[cfg(feature = "datasets")]
    pub fn upload_file(&self, file_path: &Path, repo_path: &str) -> HybridResult<()> {
        let url = format!(
            "{}/api/models/{}/uploadfile/main",
            self.api_url, self.repo_id
        );

        let file_content = std::fs::read(file_path).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to read file: {}", e),
                },
                None,
            )
        })?;

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.token))
            .set("Content-Type", "application/octet-stream")
            .query("path", repo_path)
            .send_bytes(&file_content)
            .map_err(|e| {
                (
                    HybridTrainingError::ConfigError {
                        detail: format!("Failed to upload file: {}", e),
                    },
                    None,
                )
            })?;

        if response.status() == 200 || response.status() == 201 {
            tracing::info!(
                repo_id = %self.repo_id,
                path = %repo_path,
                "Uploaded file to Hugging Face"
            );
            Ok(())
        } else {
            Err((
                HybridTrainingError::ConfigError {
                    detail: format!("Upload failed with status: {}", response.status()),
                },
                None,
            ))
        }
    }

    /// Uploads training comparison results.
    ///
    /// # Arguments
    ///
    /// * `results` - Comparison results to upload
    ///
    /// # Errors
    ///
    /// Returns error if upload fails.
    pub fn upload_training_results(&self, results: &ComparisonResults) -> HybridResult<()> {
        // Save results to temporary file
        let temp_dir = std::env::temp_dir();
        let results_path = temp_dir.join("training_results.json");

        results.save(&results_path)?;

        // Upload to HF
        #[cfg(feature = "datasets")]
        self.upload_file(&results_path, "training_results.json")?;

        #[cfg(not(feature = "datasets"))]
        tracing::warn!("datasets feature not enabled, skipping HF upload");

        Ok(())
    }

    /// Generates and uploads a model card.
    ///
    /// # Arguments
    ///
    /// * `results` - Training results for card generation
    ///
    /// # Errors
    ///
    /// Returns error if upload fails.
    pub fn upload_model_card(&self, results: &ComparisonResults) -> HybridResult<()> {
        let model_card = self.generate_model_card(results);

        let temp_dir = std::env::temp_dir();
        let card_path = temp_dir.join("README.md");
        std::fs::write(&card_path, model_card).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to write model card: {}", e),
                },
                None,
            )
        })?;

        #[cfg(feature = "datasets")]
        self.upload_file(&card_path, "README.md")?;

        #[cfg(not(feature = "datasets"))]
        tracing::warn!("datasets feature not enabled, skipping HF upload");

        Ok(())
    }

    /// Generates a model card from training results.
    fn generate_model_card(&self, results: &ComparisonResults) -> String {
        format!(
            r#"---
tags:
- hybrid-training
- predictive-training
- {dataset}
library_name: burn
license: mit
---

# {experiment_name}

Hybrid predictive training results for {dataset} using hybrid-predict-trainer-rs.

## Model Description

This model was trained using hybrid predictive training, which achieves significant speedups by intelligently predicting training steps instead of computing full forward/backward passes.

## Training Results

### Traditional Training (Baseline)

- **Final Validation Loss:** {trad_loss:.4}
- **Final Validation Accuracy:** {trad_acc:.2}%
- **Training Time:** {trad_time:.1}s
- **Backward Passes:** {trad_backward}

### Hybrid Predictive Training

- **Final Validation Loss:** {hybrid_loss:.4}
- **Final Validation Accuracy:** {hybrid_acc:.2}%
- **Training Time:** {hybrid_time:.1}s
- **Backward Passes:** {hybrid_backward}

### Performance Metrics

- **Speedup:** {speedup:.2}× faster
- **Quality Retention:** {quality:.2}%
- **Backward Pass Reduction:** {backward_reduction:.1}%

## Training Configuration

- **Dataset:** {dataset}
- **Experiment:** {experiment_name}
- **Framework:** hybrid-predict-trainer-rs
- **Backend:** Burn (Rust deep learning)

## Usage

This model can be loaded and used with the Burn framework:

```rust
use burn::prelude::*;
use hybrid_predict_trainer_rs::models::*;

let device = Default::default();
let model = load_checkpoint("path/to/checkpoint.safetensors", &device)?;
```

## Citation

If you use this model or hybrid predictive training in your research, please cite:

```bibtex
@software{{hybrid_predict_trainer,
  title = {{Hybrid Predictive Training}},
  author = {{Zervas, Tyler}},
  year = {{2026}},
  url = {{https://github.com/tzervas/hybrid-predict-trainer-rs}}
}}
```

## Repository

- **Source Code:** https://github.com/tzervas/hybrid-predict-trainer-rs
- **Documentation:** https://docs.rs/hybrid-predict-trainer-rs

## License

MIT License
"#,
            dataset = results.dataset,
            experiment_name = results.experiment_name,
            trad_loss = results.traditional.final_val_loss,
            trad_acc = results.traditional.final_val_accuracy * 100.0,
            trad_time = results.traditional.total_time.as_secs_f64(),
            trad_backward = results.traditional.backward_pass_count,
            hybrid_loss = results.hybrid.final_val_loss,
            hybrid_acc = results.hybrid.final_val_accuracy * 100.0,
            hybrid_time = results.hybrid.total_time.as_secs_f64(),
            hybrid_backward = results.hybrid.backward_pass_count,
            speedup = results.speedup_ratio(),
            quality = results.quality_retention() * 100.0,
            backward_reduction = results.backward_reduction(),
        )
    }
}

/// Checkpoint metadata for Hugging Face upload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Model architecture.
    pub architecture: String,

    /// Number of parameters.
    pub num_parameters: usize,

    /// Training method (traditional/hybrid).
    pub training_method: String,

    /// Training epoch.
    pub epoch: usize,

    /// Validation loss.
    pub val_loss: f64,

    /// Validation accuracy.
    pub val_accuracy: f64,

    /// Training timestamp.
    pub timestamp: String,
}

impl CheckpointMetadata {
    /// Creates metadata from training results.
    pub fn from_results(
        architecture: impl Into<String>,
        num_parameters: usize,
        method: impl Into<String>,
        epoch: usize,
        val_loss: f64,
        val_accuracy: f64,
    ) -> Self {
        Self {
            architecture: architecture.into(),
            num_parameters,
            training_method: method.into(),
            epoch,
            val_loss,
            val_accuracy,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Saves metadata to JSON file.
    pub fn save(&self, path: &Path) -> HybridResult<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to serialize metadata: {}", e),
                },
                None,
            )
        })?;

        std::fs::write(path, json).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to write metadata: {}", e),
                },
                None,
            )
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_uploader_creation() {
        let uploader = HfUploader::new("test-user/test-model", "test-token");
        assert_eq!(uploader.repo_id, "test-user/test-model");
    }

    #[test]
    fn test_checkpoint_metadata_creation() {
        let metadata = CheckpointMetadata::from_results(
            "MnistCnn",
            225_000,
            "hybrid",
            5,
            0.1234,
            0.99,
        );

        assert_eq!(metadata.architecture, "MnistCnn");
        assert_eq!(metadata.num_parameters, 225_000);
        assert_eq!(metadata.training_method, "hybrid");
        assert_eq!(metadata.epoch, 5);
    }

    #[test]
    fn test_model_card_generation() {
        use crate::training_comparison::{TrainingMethod, TrainingResults};
        use std::time::Duration;

        let trad_result = TrainingResults {
            method: TrainingMethod::Traditional,
            config: Default::default(),
            final_val_loss: 0.05,
            final_test_loss: 0.05,
            final_val_accuracy: 0.99,
            total_time: Duration::from_secs(60),
            compute_time: Duration::from_secs(60),
            backward_pass_count: 100,
            peak_memory: None,
            avg_memory: None,
            train_loss_history: vec![],
            val_loss_history: vec![],
            val_acc_history: vec![],
            phase_stats: None,
            completed_at: chrono::Utc::now().to_rfc3339(),
        };

        let hybrid_result = TrainingResults {
            method: TrainingMethod::Hybrid,
            backward_pass_count: 25,
            total_time: Duration::from_secs(15),
            compute_time: Duration::from_secs(15),
            ..trad_result.clone()
        };

        let comparison = ComparisonResults {
            traditional: trad_result,
            hybrid: hybrid_result,
            experiment_name: "test_experiment".to_string(),
            dataset: "MNIST".to_string(),
        };

        let uploader = HfUploader::new("test/test", "token");
        let card = uploader.generate_model_card(&comparison);

        assert!(card.contains("MNIST"));
        assert!(card.contains("test_experiment"));
        assert!(card.contains("4.00× faster")); // 60/15 = 4x
    }
}
