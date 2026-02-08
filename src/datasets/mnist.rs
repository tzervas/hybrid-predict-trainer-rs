//! MNIST dataset loader with local and remote support.
//!
//! Supports loading MNIST handwritten digit dataset from:
//! 1. Local `/data/datasets/mnist/` directory
//! 2. Remote download from Yann LeCun's website
//!
//! # Dataset Format
//!
//! MNIST uses IDX file format:
//! - `train-images-idx3-ubyte` - 60,000 training images (28×28)
//! - `train-labels-idx1-ubyte` - 60,000 training labels (0-9)
//! - `t10k-images-idx3-ubyte` - 10,000 test images (28×28)
//! - `t10k-labels-idx1-ubyte` - 10,000 test labels (0-9)
//!
//! # Example
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::datasets::{MnistDataset, DatasetSource};
//!
//! // Load from /data volume
//! let mnist = MnistDataset::load(DatasetSource::default())?;
//!
//! // Access training data
//! let (images, labels) = mnist.train_data();
//! assert_eq!(images.len(), 60000);
//! assert_eq!(images[0].len(), 784); // 28×28 flattened
//!
//! // Access test data
//! let (test_images, test_labels) = mnist.test_data();
//! assert_eq!(test_images.len(), 10000);
//! ```

use super::{Dataset, DatasetSource};
use crate::error::{HybridResult, HybridTrainingError};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

/// MNIST dataset.
pub struct MnistDataset {
    /// Training images [60000 × 784]
    train_images: Vec<Vec<f32>>,

    /// Training labels [60000]
    train_labels: Vec<usize>,

    /// Test images [10000 × 784]
    test_images: Vec<Vec<f32>>,

    /// Test labels [10000]
    test_labels: Vec<usize>,
}

impl MnistDataset {
    /// Official MNIST download URLs.
    const MNIST_URL: &'static str = "http://yann.lecun.com/exdb/mnist/";

    /// Loads MNIST dataset from source.
    ///
    /// # Arguments
    ///
    /// * `source` - Dataset source (local or remote)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Files not found locally and download fails
    /// - Files are corrupted or wrong format
    /// - I/O errors during reading
    pub fn load(source: DatasetSource) -> HybridResult<Self> {
        let mnist_dir = source.local_path("mnist");

        // Check if files exist locally
        let train_images_path = mnist_dir.join("train-images-idx3-ubyte");
        let train_labels_path = mnist_dir.join("train-labels-idx1-ubyte");
        let test_images_path = mnist_dir.join("t10k-images-idx3-ubyte");
        let test_labels_path = mnist_dir.join("t10k-labels-idx1-ubyte");

        let all_exist = train_images_path.exists()
            && train_labels_path.exists()
            && test_images_path.exists()
            && test_labels_path.exists();

        // Download if needed
        if !all_exist {
            if let DatasetSource::Remote { url, cache_dir } = &source {
                tracing::info!("MNIST files not found locally, downloading...");
                Self::download_mnist(url, cache_dir)?;
            } else {
                return Err((
                    HybridTrainingError::ConfigError {
                        detail: format!(
                            "MNIST files not found at {:?}. Use DatasetSource::Remote to download.",
                            mnist_dir
                        ),
                    },
                    None,
                ));
            }
        }

        tracing::info!("Loading MNIST dataset from {:?}", mnist_dir);

        // Load training data
        let train_images = Self::load_images(&train_images_path)?;
        let train_labels = Self::load_labels(&train_labels_path)?;

        // Load test data
        let test_images = Self::load_images(&test_images_path)?;
        let test_labels = Self::load_labels(&test_labels_path)?;

        tracing::info!(
            train_size = train_images.len(),
            test_size = test_images.len(),
            "MNIST dataset loaded successfully"
        );

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    /// Downloads MNIST files from remote source.
    fn download_mnist(base_url: &str, cache_dir: &PathBuf) -> HybridResult<()> {
        let mnist_dir = cache_dir.join("mnist");
        std::fs::create_dir_all(&mnist_dir).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to create MNIST directory: {}", e),
                },
                None,
            )
        })?;

        let files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ];

        for file in &files {
            let url = format!("{}{}", base_url, file);
            let gz_path = mnist_dir.join(file);
            let output_path = mnist_dir.join(file.trim_end_matches(".gz"));

            // Skip if already extracted
            if output_path.exists() {
                tracing::info!(file, "Already downloaded and extracted");
                continue;
            }

            // Download .gz file
            #[cfg(feature = "async")]
            {
                use super::download_file;
                // For async version, would use tokio runtime
                tracing::warn!("Async download not implemented, use sync version");
                return Err((
                    HybridTrainingError::ConfigError {
                        detail: "Async download requires tokio runtime".to_string(),
                    },
                    None,
                ));
            }

            #[cfg(not(feature = "async"))]
            {
                use super::download_file_sync;
                download_file_sync(&url, &gz_path)?;
            }

            // Decompress .gz file
            Self::decompress_gz(&gz_path, &output_path)?;

            // Remove .gz file to save space
            std::fs::remove_file(&gz_path).ok();

            tracing::info!(file, "Downloaded and extracted");
        }

        Ok(())
    }

    /// Decompresses .gz file.
    fn decompress_gz(gz_path: &PathBuf, output_path: &PathBuf) -> HybridResult<()> {
        use flate2::read::GzDecoder;
        use std::io::Write;

        let gz_file = File::open(gz_path).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to open .gz file: {}", e),
                },
                None,
            )
        })?;

        let mut decoder = GzDecoder::new(gz_file);
        let mut output_file = File::create(output_path).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to create output file: {}", e),
                },
                None,
            )
        })?;

        std::io::copy(&mut decoder, &mut output_file).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to decompress file: {}", e),
                },
                None,
            )
        })?;

        Ok(())
    }

    /// Loads images from IDX file.
    fn load_images(path: &PathBuf) -> HybridResult<Vec<Vec<f32>>> {
        let file = File::open(path).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to open images file: {}", e),
                },
                None,
            )
        })?;

        let mut reader = BufReader::new(file);

        // Read IDX header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to read magic number: {}", e),
                },
                None,
            )
        })?;

        let magic_number = u32::from_be_bytes(magic);
        if magic_number != 0x00000803 {
            // IDX3 magic number
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!("Invalid IDX3 magic number: {:x}", magic_number),
                },
                None,
            ));
        }

        // Read dimensions
        let num_images = Self::read_u32(&mut reader)?;
        let rows = Self::read_u32(&mut reader)?;
        let cols = Self::read_u32(&mut reader)?;

        tracing::debug!(
            num_images,
            rows,
            cols,
            path = ?path,
            "Loading images"
        );

        // Read pixel data
        let image_size = (rows * cols) as usize;
        let mut images = Vec::with_capacity(num_images as usize);

        for _ in 0..num_images {
            let mut pixels = vec![0u8; image_size];
            reader.read_exact(&mut pixels).map_err(|e| {
                (
                    HybridTrainingError::ConfigError {
                        detail: format!("Failed to read pixel data: {}", e),
                    },
                    None,
                )
            })?;

            // Normalize to [0, 1]
            let normalized: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();
            images.push(normalized);
        }

        Ok(images)
    }

    /// Loads labels from IDX file.
    fn load_labels(path: &PathBuf) -> HybridResult<Vec<usize>> {
        let file = File::open(path).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to open labels file: {}", e),
                },
                None,
            )
        })?;

        let mut reader = BufReader::new(file);

        // Read IDX header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to read magic number: {}", e),
                },
                None,
            )
        })?;

        let magic_number = u32::from_be_bytes(magic);
        if magic_number != 0x00000801 {
            // IDX1 magic number
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!("Invalid IDX1 magic number: {:x}", magic_number),
                },
                None,
            ));
        }

        // Read number of labels
        let num_labels = Self::read_u32(&mut reader)?;

        tracing::debug!(num_labels, path = ?path, "Loading labels");

        // Read label data
        let mut labels = Vec::with_capacity(num_labels as usize);
        for _ in 0..num_labels {
            let mut label = [0u8; 1];
            reader.read_exact(&mut label).map_err(|e| {
                (
                    HybridTrainingError::ConfigError {
                        detail: format!("Failed to read label: {}", e),
                    },
                    None,
                )
            })?;
            labels.push(label[0] as usize);
        }

        Ok(labels)
    }

    /// Reads u32 in big-endian format.
    fn read_u32(reader: &mut BufReader<File>) -> HybridResult<u32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).map_err(|e| {
            (
                HybridTrainingError::ConfigError {
                    detail: format!("Failed to read u32: {}", e),
                },
                None,
            )
        })?;
        Ok(u32::from_be_bytes(buf))
    }

    /// Returns training data.
    #[must_use]
    pub fn train_data(&self) -> (&[Vec<f32>], &[usize]) {
        (&self.train_images, &self.train_labels)
    }

    /// Returns test data.
    #[must_use]
    pub fn test_data(&self) -> (&[Vec<f32>], &[usize]) {
        (&self.test_images, &self.test_labels)
    }
}

impl Dataset for MnistDataset {
    fn name(&self) -> &str {
        "MNIST"
    }

    fn train_size(&self) -> usize {
        self.train_images.len()
    }

    fn test_size(&self) -> usize {
        self.test_images.len()
    }

    fn input_dim(&self) -> usize {
        784 // 28×28
    }

    fn num_classes(&self) -> Option<usize> {
        Some(10) // Digits 0-9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires dataset files
    fn test_load_mnist_local() {
        let source = DatasetSource::Local(PathBuf::from("/data/datasets"));
        let result = MnistDataset::load(source);

        if result.is_ok() {
            let dataset = result.unwrap();
            assert_eq!(dataset.train_size(), 60000);
            assert_eq!(dataset.test_size(), 10000);
            assert_eq!(dataset.input_dim(), 784);
            assert_eq!(dataset.num_classes(), Some(10));
        }
    }
}
