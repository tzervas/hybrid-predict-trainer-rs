//! Dataset loading and streaming for training comparisons.
//!
//! This module provides dataset loaders that support both:
//! 1. **Local datasets** from homelab `/data` volume
//! 2. **Remote datasets** streamed from external sources
//!
//! # Supported Datasets
//!
//! - **MNIST** - Handwritten digit classification (28×28 grayscale images)
//! - **CIFAR-10** - Object classification (32×32 RGB images)
//! - **WikiText-103** - Language modeling corpus
//! - **OpenWebText** - GPT-2 training data subset
//!
//! # Storage Strategy
//!
//! Datasets are cached in `/data/datasets/` for reuse:
//! ```text
//! /data/datasets/
//! ├── mnist/
//! │   ├── train-images-idx3-ubyte
//! │   ├── train-labels-idx1-ubyte
//! │   ├── t10k-images-idx3-ubyte
//! │   └── t10k-labels-idx1-ubyte
//! ├── cifar10/
//! └── wikitext-103/
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::datasets::{MnistDataset, DatasetSource};
//!
//! // Load from local /data volume
//! let dataset = MnistDataset::load(DatasetSource::Local("/data/datasets".into()))?;
//!
//! // Or download from remote source
//! let dataset = MnistDataset::load(DatasetSource::Remote {
//!     url: "http://yann.lecun.com/exdb/mnist/".to_string(),
//!     cache_dir: "/data/datasets".into(),
//! })?;
//!
//! // Access data
//! let (train_images, train_labels) = dataset.train_data();
//! let (test_images, test_labels) = dataset.test_data();
//! ```

pub mod mnist;

#[cfg(feature = "datasets")]
pub mod parquet_stream;

use crate::error::{HybridResult, HybridTrainingError};
use std::path::PathBuf;

/// Dataset source configuration.
#[derive(Debug, Clone)]
pub enum DatasetSource {
    /// Load from local filesystem.
    Local(PathBuf),

    /// Download from remote URL and cache locally.
    Remote {
        /// Base URL for dataset files.
        url: String,
        /// Local directory to cache downloaded files.
        cache_dir: PathBuf,
    },
}

impl Default for DatasetSource {
    fn default() -> Self {
        Self::Local(PathBuf::from("/data/datasets"))
    }
}

impl DatasetSource {
    /// Returns the local path where dataset should be stored.
    #[must_use]
    pub fn local_path(&self, dataset_name: &str) -> PathBuf {
        match self {
            Self::Local(path) => path.join(dataset_name),
            Self::Remote { cache_dir, .. } => cache_dir.join(dataset_name),
        }
    }

    /// Checks if dataset exists locally.
    pub fn exists(&self, dataset_name: &str) -> bool {
        self.local_path(dataset_name).exists()
    }
}

/// Common trait for all datasets.
pub trait Dataset {
    /// Returns dataset name.
    fn name(&self) -> &str;

    /// Returns number of training samples.
    fn train_size(&self) -> usize;

    /// Returns number of test samples.
    fn test_size(&self) -> usize;

    /// Returns input dimension (e.g., 784 for MNIST).
    fn input_dim(&self) -> usize;

    /// Returns number of classes (for classification tasks).
    fn num_classes(&self) -> Option<usize>;
}

/// Helper function to download file from URL.
///
/// # Arguments
///
/// * `url` - Full URL to download from
/// * `dest_path` - Local path to save file
///
/// # Errors
///
/// Returns error if download fails or file cannot be written.
#[cfg(feature = "async")]
pub async fn download_file(url: &str, dest_path: &PathBuf) -> HybridResult<()> {
    use tokio::io::AsyncWriteExt;

    tracing::info!(url, path = ?dest_path, "Downloading file");

    // Create parent directory if needed
    if let Some(parent) = dest_path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to create directory: {}", e),
                },
                None,
            )
        })?;
    }

    // Download file
    let response = reqwest::get(url).await.map_err(|e| {
        (
            HybridTrainingError::ConfigError {
                detail: format!("Failed to download file: {}", e),
            },
            None,
        )
    })?;

    if !response.status().is_success() {
        return Err((
            HybridTrainingError::ConfigError {
                detail: format!("HTTP error: {}", response.status()),
            },
            None,
        ));
    }

    let bytes = response.bytes().await.map_err(|e| {
        (
            HybridTrainingError::ConfigError {
                detail: format!("Failed to read response: {}", e),
            },
            None,
        )
    })?;

    // Write to file
    let mut file = tokio::fs::File::create(dest_path).await.map_err(|e| {
        (
            HybridTrainingError::ConfigError {
                detail: format!("Failed to create file: {}", e),
            },
            None,
        )
    })?;

    file.write_all(&bytes).await.map_err(|e| {
        (
            HybridTrainingError::ConfigError {
                detail: format!("Failed to write file: {}", e),
            },
            None,
        )
    })?;

    tracing::info!(path = ?dest_path, size_bytes = bytes.len(), "File downloaded");
    Ok(())
}

/// Synchronous version of download_file for non-async contexts.
#[cfg(not(feature = "async"))]
pub fn download_file_sync(url: &str, dest_path: &PathBuf) -> HybridResult<()> {
    use std::io::Write;

    tracing::info!(url, path = ?dest_path, "Downloading file");

    // Create parent directory if needed
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to create directory: {}", e),
                },
                None,
            )
        })?;
    }

    // Download file using ureq (synchronous HTTP client)
    let response = ureq::get(url).call().map_err(|e| {
        (
            HybridTrainingError::ConfigError {
                detail: format!("Failed to download file: {}", e),
            },
            None,
        )
    })?;

    if response.status() != 200 {
        return Err((
            HybridTrainingError::ConfigError {
                detail: format!("HTTP error: {}", response.status()),
            },
            None,
        ));
    }

    // Read response body
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to read response: {}", e),
                },
                None,
            )
        })?;

    // Write to file
    let mut file = std::fs::File::create(dest_path).map_err(|e| {
        (
            HybridTrainingError::ConfigError {
                detail: format!("Failed to create file: {}", e),
            },
            None,
        )
    })?;

    file.write_all(&bytes).map_err(|e| {
        (
            HybridTrainingError::ConfigError {
                detail: format!("Failed to write file: {}", e),
            },
            None,
        )
    })?;

    tracing::info!(path = ?dest_path, size_bytes = bytes.len(), "File downloaded");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_source_local_path() {
        let source = DatasetSource::Local(PathBuf::from("/data/datasets"));
        assert_eq!(
            source.local_path("mnist"),
            PathBuf::from("/data/datasets/mnist")
        );
    }

    #[test]
    fn test_dataset_source_remote_path() {
        let source = DatasetSource::Remote {
            url: "http://example.com/data".to_string(),
            cache_dir: PathBuf::from("/data/datasets"),
        };
        assert_eq!(
            source.local_path("mnist"),
            PathBuf::from("/data/datasets/mnist")
        );
    }
}
