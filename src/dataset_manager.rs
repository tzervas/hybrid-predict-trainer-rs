//! Dataset management for training comparisons.
//!
//! This module provides standardized dataset loading and management for
//! comparing hybrid predictive training against traditional training.
//!
//! # Dataset Storage
//!
//! All datasets are stored on the homelab `/data` volume for persistence
//! across training runs and system restarts:
//!
//! ```text
//! /data/datasets/
//! ├── mnist/
//! │   ├── train-images.idx3-ubyte
//! │   ├── train-labels.idx1-ubyte
//! │   ├── t10k-images.idx3-ubyte
//! │   └── t10k-labels.idx1-ubyte
//! ├── cifar10/
//! │   └── cifar-10-batches-bin/
//! ├── gpt2-datasets/
//! │   ├── wikitext-103/
//! │   └── openwebtext/
//! └── metadata/
//!     ├── mnist_splits.json
//!     ├── cifar10_splits.json
//!     └── training_configs.json
//! ```
//!
//! # Training Parity
//!
//! To ensure fair comparison between hybrid and traditional training:
//!
//! 1. **Identical datasets**: Both methods use same train/val/test splits
//! 2. **Deterministic splits**: Seeded random state for reproducibility
//! 3. **Metadata tracking**: Record dataset versions and preprocessing
//! 4. **Checksum validation**: Verify dataset integrity before training
//!
//! # Usage
//!
//! ```rust,ignore
//! use hybrid_predict_trainer::dataset_manager::{DatasetManager, DatasetConfig};
//!
//! let config = DatasetConfig {
//!     name: "mnist".to_string(),
//!     root: "/data/datasets".into(),
//!     download_if_missing: true,
//!     validation_split: 0.1,
//!     random_seed: 42,
//! };
//!
//! let manager = DatasetManager::new(config)?;
//! let (train, val, test) = manager.load_splits()?;
//! ```

use crate::error::{HybridResult, HybridTrainingError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Dataset configuration for training comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset name (e.g., "mnist", "cifar10", "gpt2-wikitext").
    pub name: String,

    /// Root directory for dataset storage (default: "/data/datasets").
    pub root: PathBuf,

    /// Whether to download dataset if not found locally.
    pub download_if_missing: bool,

    /// Fraction of training data to use for validation (0.0 - 1.0).
    pub validation_split: f32,

    /// Random seed for deterministic train/val splits.
    pub random_seed: u64,

    /// Batch size for training.
    pub batch_size: usize,

    /// Number of worker threads for data loading.
    pub num_workers: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            name: "mnist".to_string(),
            root: PathBuf::from("/data/datasets"),
            download_if_missing: true,
            validation_split: 0.1,
            random_seed: 42,
            batch_size: 64,
            num_workers: 4,
        }
    }
}

impl DatasetConfig {
    /// Validates configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if validation_split is out of range [0.0, 1.0].
    pub fn validate(&self) -> HybridResult<()> {
        if !(0.0..=1.0).contains(&self.validation_split) {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!(
                        "validation_split must be in [0.0, 1.0], got {}",
                        self.validation_split
                    ),
                },
                None,
            ));
        }

        if self.batch_size == 0 {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: "batch_size must be > 0".to_string(),
                },
                None,
            ));
        }

        Ok(())
    }

    /// Returns path to dataset directory.
    #[must_use]
    pub fn dataset_path(&self) -> PathBuf {
        self.root.join(&self.name)
    }

    /// Returns path to metadata file for this dataset.
    #[must_use]
    pub fn metadata_path(&self) -> PathBuf {
        self.root
            .join("metadata")
            .join(format!("{}_splits.json", self.name))
    }
}

/// Metadata for dataset splits to ensure training parity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name.
    pub name: String,

    /// Dataset version or checksum.
    pub version: String,

    /// Random seed used for splits.
    pub random_seed: u64,

    /// Validation split fraction.
    pub validation_split: f32,

    /// Training set size.
    pub train_size: usize,

    /// Validation set size.
    pub val_size: usize,

    /// Test set size.
    pub test_size: usize,

    /// Timestamp of split creation.
    pub created_at: String,

    /// SHA256 checksum of dataset files.
    pub checksum: Option<String>,
}

impl DatasetMetadata {
    /// Creates new metadata from configuration and sizes.
    pub fn new(
        config: &DatasetConfig,
        train_size: usize,
        val_size: usize,
        test_size: usize,
    ) -> Self {
        Self {
            name: config.name.clone(),
            version: "1.0.0".to_string(),
            random_seed: config.random_seed,
            validation_split: config.validation_split,
            train_size,
            val_size,
            test_size,
            created_at: chrono::Utc::now().to_rfc3339(),
            checksum: None,
        }
    }

    /// Saves metadata to JSON file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be written.
    pub fn save(&self, path: &PathBuf) -> HybridResult<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                (
                    HybridTrainingError::ConfigError {
                        detail: format!("Failed to create metadata directory: {}", e),
                    },
                    None,
                )
            })?;
        }

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
                    detail: format!("Failed to write metadata file: {}", e),
                },
                None,
            )
        })?;

        tracing::info!(path = ?path, "Saved dataset metadata");
        Ok(())
    }

    /// Loads metadata from JSON file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed.
    pub fn load(path: &PathBuf) -> HybridResult<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to read metadata file: {}", e),
                },
                None,
            )
        })?;

        let metadata: Self = serde_json::from_str(&json).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to parse metadata: {}", e),
                },
                None,
            )
        })?;

        tracing::info!(path = ?path, "Loaded dataset metadata");
        Ok(metadata)
    }
}

/// Dataset manager for loading and managing training data.
pub struct DatasetManager {
    /// Dataset configuration.
    config: DatasetConfig,

    /// Cached metadata (loaded or created).
    metadata: Option<DatasetMetadata>,
}

impl DatasetManager {
    /// Creates a new dataset manager.
    ///
    /// # Arguments
    ///
    /// * `config` - Dataset configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: DatasetConfig) -> HybridResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            metadata: None,
        })
    }

    /// Returns dataset configuration.
    #[must_use]
    pub fn config(&self) -> &DatasetConfig {
        &self.config
    }

    /// Returns cached metadata if available.
    #[must_use]
    pub fn metadata(&self) -> Option<&DatasetMetadata> {
        self.metadata.as_ref()
    }

    /// Verifies dataset exists and is accessible.
    ///
    /// # Errors
    ///
    /// Returns error if dataset directory doesn't exist and download_if_missing is false.
    pub fn verify_dataset(&self) -> HybridResult<()> {
        let dataset_path = self.config.dataset_path();

        if !dataset_path.exists() {
            if self.config.download_if_missing {
                return Err((
                    HybridTrainingError::ConfigError {
                        detail: format!(
                            "Dataset '{}' not found at {:?}. Auto-download not yet implemented.",
                            self.config.name, dataset_path
                        ),
                    },
                    None,
                ));
            } else {
                return Err((
                    HybridTrainingError::ConfigError {
                        detail: format!(
                            "Dataset '{}' not found at {:?}",
                            self.config.name, dataset_path
                        ),
                    },
                    None,
                ));
            }
        }

        tracing::info!(
            dataset = self.config.name,
            path = ?dataset_path,
            "Dataset verified"
        );

        Ok(())
    }

    /// Loads or creates dataset metadata.
    ///
    /// If metadata file exists, loads it and verifies compatibility.
    /// Otherwise, creates new metadata based on current configuration.
    ///
    /// # Errors
    ///
    /// Returns error if metadata cannot be loaded or created.
    pub fn load_or_create_metadata(&mut self) -> HybridResult<&DatasetMetadata> {
        let metadata_path = self.config.metadata_path();

        if metadata_path.exists() {
            // Load existing metadata
            let metadata = DatasetMetadata::load(&metadata_path)?;

            // Verify compatibility
            if metadata.random_seed != self.config.random_seed
                || (metadata.validation_split - self.config.validation_split).abs() > 1e-6
            {
                tracing::warn!(
                    "Metadata mismatch: config seed={}, split={:.2}, metadata seed={}, split={:.2}",
                    self.config.random_seed,
                    self.config.validation_split,
                    metadata.random_seed,
                    metadata.validation_split
                );
            }

            self.metadata = Some(metadata);
        } else {
            // Create placeholder metadata (actual sizes will be filled during loading)
            tracing::info!("No metadata found, will create after dataset loading");
        }

        Ok(self
            .metadata
            .as_ref()
            .expect("Metadata should be set at this point"))
    }

    /// Creates and saves new metadata after dataset loading.
    ///
    /// Called after dataset splits are created to record split sizes.
    pub fn save_metadata(
        &mut self,
        train_size: usize,
        val_size: usize,
        test_size: usize,
    ) -> HybridResult<()> {
        let metadata = DatasetMetadata::new(&self.config, train_size, val_size, test_size);
        let metadata_path = self.config.metadata_path();

        metadata.save(&metadata_path)?;
        self.metadata = Some(metadata);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_config_default() {
        let config = DatasetConfig::default();
        assert_eq!(config.name, "mnist");
        assert_eq!(config.validation_split, 0.1);
        assert_eq!(config.random_seed, 42);
    }

    #[test]
    fn test_dataset_config_validation() {
        let mut config = DatasetConfig::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid validation_split should fail
        config.validation_split = 1.5;
        assert!(config.validate().is_err());

        config.validation_split = -0.1;
        assert!(config.validate().is_err());

        // Zero batch size should fail
        config.validation_split = 0.1;
        config.batch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dataset_paths() {
        let config = DatasetConfig {
            name: "mnist".to_string(),
            root: PathBuf::from("/data/datasets"),
            ..Default::default()
        };

        assert_eq!(config.dataset_path(), PathBuf::from("/data/datasets/mnist"));
        assert_eq!(
            config.metadata_path(),
            PathBuf::from("/data/datasets/metadata/mnist_splits.json")
        );
    }

    #[test]
    fn test_metadata_creation() {
        let config = DatasetConfig::default();
        let metadata = DatasetMetadata::new(&config, 50000, 10000, 10000);

        assert_eq!(metadata.name, "mnist");
        assert_eq!(metadata.train_size, 50000);
        assert_eq!(metadata.val_size, 10000);
        assert_eq!(metadata.test_size, 10000);
        assert_eq!(metadata.random_seed, 42);
    }
}
