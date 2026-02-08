//! Reference model implementations for validation.
//!
//! This module contains model architectures used for validating the HybridTrainer
//! on real-world tasks. These are not part of the core library functionality.

#[cfg(feature = "autodiff")]
pub mod gpt2;

#[cfg(feature = "autodiff")]
pub mod mnist_cnn;
