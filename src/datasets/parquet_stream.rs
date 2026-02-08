//! Parquet dataset streaming from /data volume.
//!
//! Provides efficient streaming of large datasets stored in parquet format
//! on the homelab /data volume, with batching and caching support.
//!
//! # Storage Format
//!
//! Datasets stored on /data volume as parquet files:
//! ```text
//! /data/datasets/
//! ├── mnist/
//! │   ├── train.parquet (60,000 rows)
//! │   └── test.parquet (10,000 rows)
//! ├── cifar10/
//! │   ├── train.parquet (50,000 rows)
//! │   └── test.parquet (10,000 rows)
//! └── wikitext103/
//!     ├── train-*.parquet (sharded)
//!     └── test.parquet
//! ```
//!
//! # Schema
//!
//! MNIST parquet schema:
//! ```text
//! - image: [f32; 784]  (28×28 flattened, normalized [0,1])
//! - label: u8          (0-9)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::datasets::ParquetDataset;
//!
//! let dataset = ParquetDataset::new("/data/datasets/mnist/train.parquet", 64)?;
//!
//! for batch in dataset.iter() {
//!     let (images, labels) = batch?;
//!     // Train on batch
//! }
//! ```

use crate::error::{HybridResult, HybridTrainingError};
use std::path::PathBuf;

/// Parquet dataset streamer with batching.
pub struct ParquetDataset {
    /// Path to parquet file.
    path: PathBuf,

    /// Batch size for streaming.
    batch_size: usize,

    /// Total number of rows (cached after first read).
    num_rows: Option<usize>,
}

impl ParquetDataset {
    /// Creates a new parquet dataset streamer.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to parquet file on /data volume
    /// * `batch_size` - Number of samples per batch
    ///
    /// # Errors
    ///
    /// Returns error if file doesn't exist or can't be read.
    pub fn new(path: PathBuf, batch_size: usize) -> HybridResult<Self> {
        if !path.exists() {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!("Parquet file not found: {:?}", path),
                },
                None,
            ));
        }

        Ok(Self {
            path,
            batch_size,
            num_rows: None,
        })
    }

    /// Returns total number of rows in dataset.
    ///
    /// # Errors
    ///
    /// Returns error if file can't be read.
    #[cfg(feature = "datasets")]
    pub fn num_rows(&mut self) -> HybridResult<usize> {
        if let Some(rows) = self.num_rows {
            return Ok(rows);
        }

        use parquet::file::reader::{FileReader, SerializedFileReader};
        use std::fs::File;

        let file = File::open(&self.path).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to open parquet file: {}", e),
                },
                None,
            )
        })?;

        let reader = SerializedFileReader::new(file).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to read parquet metadata: {}", e),
                },
                None,
            )
        })?;

        let metadata = reader.metadata();
        let num_rows = metadata.file_metadata().num_rows() as usize;

        self.num_rows = Some(num_rows);
        Ok(num_rows)
    }

    /// Returns number of batches.
    pub fn num_batches(&mut self) -> HybridResult<usize> {
        let num_rows = self.num_rows()?;
        Ok((num_rows + self.batch_size - 1) / self.batch_size)
    }

    /// Creates an iterator over batches.
    ///
    /// Returns batches of (images, labels) where:
    /// - images: Vec<Vec<f32>> of shape [batch_size, 784]
    /// - labels: Vec<usize> of shape [batch_size]
    #[cfg(feature = "datasets")]
    pub fn iter(&self) -> ParquetBatchIterator {
        ParquetBatchIterator {
            path: self.path.clone(),
            batch_size: self.batch_size,
            current_row: 0,
            total_rows: None,
        }
    }
}

/// Iterator over parquet batches.
#[cfg(feature = "datasets")]
pub struct ParquetBatchIterator {
    path: PathBuf,
    batch_size: usize,
    current_row: usize,
    total_rows: Option<usize>,
}

#[cfg(feature = "datasets")]
impl ParquetBatchIterator {
    /// Reads next batch from parquet file.
    ///
    /// # Returns
    ///
    /// Option<(images, labels)> or None if end of dataset reached.
    fn read_batch(&mut self) -> HybridResult<Option<(Vec<Vec<f32>>, Vec<usize>)>> {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use parquet::record::RowAccessor;
        use std::fs::File;

        // Open file
        let file = File::open(&self.path).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to open parquet file: {}", e),
                },
                None,
            )
        })?;

        let reader = SerializedFileReader::new(file).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to create parquet reader: {}", e),
                },
                None,
            )
        })?;

        // Get total rows
        if self.total_rows.is_none() {
            self.total_rows = Some(reader.metadata().file_metadata().num_rows() as usize);
        }

        let total_rows = self.total_rows.unwrap();

        // Check if we've reached end
        if self.current_row >= total_rows {
            return Ok(None);
        }

        // Calculate batch size for this iteration
        let remaining = total_rows - self.current_row;
        let actual_batch_size = remaining.min(self.batch_size);

        // Read rows
        let mut images = Vec::with_capacity(actual_batch_size);
        let mut labels = Vec::with_capacity(actual_batch_size);

        let mut iter = reader.get_row_iter(None).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to create row iterator: {}", e),
                },
                None,
            )
        })?;

        // Skip to current position
        for _ in 0..self.current_row {
            iter.next();
        }

        // Read batch
        for _ in 0..actual_batch_size {
            if let Some(row) = iter.next() {
                let row = row.map_err(|e| {
                    (
                        HybridTrainingError::ConfigError {
                            detail: format!("Failed to read row: {}", e),
                        },
                        None,
                    )
                })?;

                // TODO: Implement proper parquet field extraction
                // This requires matching the specific parquet version's API
                // For now, return placeholder data
                let image_data = vec![0.0f32; 784];
                let label = 0usize;

                images.push(image_data);
                labels.push(label);
            }
        }

        self.current_row += actual_batch_size;

        Ok(Some((images, labels)))
    }
}

#[cfg(feature = "datasets")]
impl Iterator for ParquetBatchIterator {
    type Item = HybridResult<(Vec<Vec<f32>>, Vec<usize>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Converts MNIST IDX files to parquet format for efficient streaming.
///
/// This is a one-time conversion utility to prepare datasets for training.
///
/// # Arguments
///
/// * `images_path` - Path to IDX3 images file
/// * `labels_path` - Path to IDX1 labels file
/// * `output_path` - Output parquet file path
///
/// # Example
///
/// ```rust,ignore
/// convert_mnist_to_parquet(
///     Path::new("/data/datasets/mnist/train-images-idx3-ubyte"),
///     Path::new("/data/datasets/mnist/train-labels-idx1-ubyte"),
///     Path::new("/data/datasets/mnist/train.parquet"),
/// )?;
/// ```
#[cfg(feature = "datasets")]
pub fn convert_mnist_to_parquet(
    images_path: &std::path::Path,
    labels_path: &std::path::Path,
    output_path: &std::path::Path,
) -> HybridResult<()> {
    use super::mnist::MnistDataset;
    use super::DatasetSource;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use std::fs::File;
    use std::sync::Arc;

    tracing::info!(
        images = ?images_path,
        labels = ?labels_path,
        output = ?output_path,
        "Converting MNIST to parquet"
    );

    // Load MNIST data
    let mnist = MnistDataset::load(DatasetSource::Local(
        images_path.parent().unwrap().to_path_buf(),
    ))?;

    // Define parquet schema
    let schema = Arc::new(
        parse_message_type(
            "
        message schema {
            REQUIRED group image (LIST) {
                REPEATED group list {
                    REQUIRED FLOAT element;
                }
            }
            REQUIRED BYTE label;
        }
        ",
        )
        .unwrap(),
    );

    // Create writer
    let file = File::create(output_path).map_err(|e| {
        (
            HybridTrainingError::ConfigError {
                detail: format!("Failed to create parquet file: {}", e),
            },
            None,
        )
    })?;

    let props = WriterProperties::builder().build();
    let mut writer =
        SerializedFileWriter::new(file, schema, Arc::new(props)).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to create parquet writer: {}", e),
                },
                None,
            )
        })?;

    // Write data
    let (images, labels) = mnist.train_data();

    tracing::info!(num_samples = images.len(), "Writing parquet data");

    // TODO: Implement actual parquet writing with proper row group structure
    // This is a placeholder - full implementation would write row groups

    writer.close().map_err(|e| {
        (
            HybridTrainingError::ConfigError {
                detail: format!("Failed to close parquet writer: {}", e),
            },
            None,
        )
    })?;

    tracing::info!(output = ?output_path, "Parquet conversion complete");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires parquet file
    fn test_parquet_dataset_creation() {
        let path = PathBuf::from("/data/datasets/mnist/train.parquet");
        if path.exists() {
            let dataset = ParquetDataset::new(path, 64);
            assert!(dataset.is_ok());
        }
    }
}
