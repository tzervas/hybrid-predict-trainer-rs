# Hugging Face Hub Integration Guide

Complete guide for uploading trained models to Hugging Face Hub with hybrid-predict-trainer-rs.

---

## Overview

The hybrid-predict-trainer-rs framework includes first-class Hugging Face Hub integration for:
- Automatic repository creation
- Model checkpoint uploads
- Training results and metrics export
- Auto-generated model cards with performance stats
- Version tracking and metadata

---

## Quick Start

### 1. Get Your Hugging Face Token

1. Visit https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "hybrid-trainer-upload")
4. Select "Write" permissions
5. Copy the token

### 2. Set Environment Variable

```bash
export HF_TOKEN="hf_your_token_here"
```

### 3. Create Repositories

```bash
./scripts/create_hf_repos.sh
```

This creates repositories for:
- `tzervas/mnist-cnn-traditional` - Baseline traditional training
- `tzervas/mnist-cnn-hybrid` - Hybrid predictive training
- `tzervas/mnist-cnn-hybrid-gpu` - GPU-accelerated hybrid
- `tzervas/gpt2-small-traditional` - GPT-2 Small baseline
- `tzervas/gpt2-small-hybrid` - GPT-2 Small hybrid

### 4. Train and Upload

```bash
cargo run --release --example mnist_with_hf_upload --features "autodiff,datasets"
```

Models are automatically uploaded upon completion!

---

## Repository Structure

Each model repository contains:

```
tzervas/model-name/
├── README.md              # Auto-generated model card
├── training_results.json  # Complete training metrics
├── checkpoint.safetensors # Model weights (when available)
└── metadata.json          # Checkpoint metadata
```

### Model Card Example

The auto-generated model cards include:
- Training method (traditional vs hybrid)
- Performance metrics (speedup, quality retention)
- Training configuration
- Usage examples
- Citation information

---

## Using the API Directly

### Creating a Repository

```rust
use hybrid_predict_trainer_rs::hf_integration::{HfUploader, RepoVisibility};

let uploader = HfUploader::new("tzervas/my-model", "hf_token_here");
uploader.create_repo(RepoVisibility::Public)?;
```

### Uploading Training Results

```rust
use hybrid_predict_trainer_rs::training_comparison::ComparisonResults;

// After training...
uploader.upload_training_results(&comparison_results)?;
uploader.upload_model_card(&comparison_results)?;
```

### Uploading Checkpoints

```rust
use hybrid_predict_trainer_rs::hf_integration::CheckpointMetadata;
use std::path::Path;

let metadata = CheckpointMetadata::from_results(
    "MnistCnn",
    225_000,  // num parameters
    "hybrid",
    5,        // epoch
    0.05,     // val_loss
    0.99,     // val_accuracy
);

metadata.save(Path::new("metadata.json"))?;
uploader.upload_file(Path::new("checkpoint.safetensors"), "checkpoint.safetensors")?;
```

---

## Model Variants

### MNIST CNN Variants

**Traditional Training**
- Repository: `tzervas/mnist-cnn-traditional`
- Method: Standard SGD with full backward passes
- Accuracy: ~99.0%
- Training time: ~60-90s (CPU)

**Hybrid Training**
- Repository: `tzervas/mnist-cnn-hybrid`
- Method: Hybrid predictive with 75% backward reduction
- Accuracy: ~98.8-99.0% (99.9% retention)
- Training time: ~15-25s (CPU)
- Speedup: **4-4.2× faster**

**GPU-Accelerated Hybrid**
- Repository: `tzervas/mnist-cnn-hybrid-gpu`
- Method: Hybrid + ensemble batching + persistent GPU state + kernel fusion
- Target speedup: **15× faster** (5× × 2× × 1.5×)
- Requires: CUDA-capable GPU

### GPT-2 Small Variants

**Traditional Training**
- Repository: `tzervas/gpt2-small-traditional`
- Model: GPT-2 Small (124M parameters)
- Dataset: WikiText-103
- Baseline for comparison

**Hybrid Training**
- Repository: `tzervas/gpt2-small-hybrid`
- Model: GPT-2 Small (124M parameters)
- Expected speedup: **1.74-4× faster** (memory-bound baseline)
- Memory optimization: Delta accumulator + VRAM manager

---

## Performance Metrics Uploaded

Each upload includes comprehensive metrics:

```json
{
  "traditional": {
    "final_val_loss": 0.05,
    "final_val_accuracy": 0.99,
    "total_time": "60s",
    "backward_pass_count": 100
  },
  "hybrid": {
    "final_val_loss": 0.051,
    "final_val_accuracy": 0.988,
    "total_time": "15s",
    "backward_pass_count": 25
  },
  "speedup_ratio": 4.0,
  "quality_retention": 0.998,
  "backward_reduction": 75.0
}
```

---

## Advanced Configuration

### Custom Repository Names

```rust
let uploader = HfUploader::new("your-username/custom-model-name", token);
```

### Private Repositories

```rust
uploader.create_repo(RepoVisibility::Private)?;
```

### Batch Uploads

```rust
// Upload multiple files
uploader.upload_file(Path::new("checkpoint_epoch1.safetensors"), "checkpoints/epoch1.safetensors")?;
uploader.upload_file(Path::new("checkpoint_epoch2.safetensors"), "checkpoints/epoch2.safetensors")?;
uploader.upload_file(Path::new("checkpoint_epoch3.safetensors"), "checkpoints/epoch3.safetensors")?;
```

---

## Troubleshooting

### "Failed to create repository"

**Problem:** HTTP 409 (Conflict)
**Solution:** Repository already exists. This is normal - the code will proceed with upload.

### "Authentication failed"

**Problem:** Invalid or expired token
**Solution:**
1. Check your token is correct: `echo $HF_TOKEN`
2. Verify it has "Write" permissions
3. Generate a new token if needed

### "Upload failed"

**Problem:** Network or file errors
**Solution:**
1. Check internet connection
2. Verify file exists and is readable
3. Check disk space for temporary files

### "datasets feature not enabled"

**Problem:** Trying to upload without `datasets` feature
**Solution:**
```bash
cargo run --release --features "autodiff,datasets" --example mnist_with_hf_upload
```

---

## Integration with Training Pipeline

### Automatic Upload After Training

```rust
use hybrid_predict_trainer_rs::{
    HybridTrainer,
    hf_integration::HfUploader,
    training_comparison::ComparisonResults,
};

// Train both variants
let trad_results = train_traditional()?;
let hybrid_results = train_hybrid()?;

// Create comparison
let comparison = ComparisonResults {
    traditional: trad_results,
    hybrid: hybrid_results,
    experiment_name: "mnist_experiment".into(),
    dataset: "MNIST".into(),
};

// Upload to HF
if let Ok(token) = std::env::var("HF_TOKEN") {
    let uploader = HfUploader::new("tzervas/my-model", token);
    uploader.create_repo(RepoVisibility::Public)?;
    uploader.upload_training_results(&comparison)?;
    uploader.upload_model_card(&comparison)?;
}
```

---

## Benchmark Integration

The benchmarking script (`scripts/run_full_benchmarks.sh`) includes HF upload prompts:

```bash
./scripts/run_full_benchmarks.sh

# After benchmarks complete:
export HF_TOKEN="your_token"
cargo run --release --example mnist_with_hf_upload --features "autodiff,datasets"
```

---

## Model Card Format

Auto-generated model cards follow this structure:

```markdown
---
tags:
- hybrid-training
- predictive-training
- mnist
library_name: burn
license: mit
---

# Model Name

Hybrid predictive training results for MNIST using hybrid-predict-trainer-rs.

## Model Description
[Training method details...]

## Training Results
### Traditional Training
- Final Validation Loss: 0.05
- Final Validation Accuracy: 99.0%
- Training Time: 60s
- Backward Passes: 100

### Hybrid Predictive Training
- Final Validation Loss: 0.051
- Final Validation Accuracy: 98.8%
- Training Time: 15s
- Backward Passes: 25

### Performance Metrics
- Speedup: 4.0× faster
- Quality Retention: 99.8%
- Backward Pass Reduction: 75%

## Usage
[Code examples...]

## Citation
[BibTeX citation...]
```

---

## Repository Management Scripts

### Create All Repositories

```bash
./scripts/create_hf_repos.sh
```

### List Your Repositories

```bash
curl -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models?author=tzervas"
```

### Delete a Repository

```bash
curl -X DELETE \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/repos/delete" \
  -d '{"type": "model", "name": "tzervas/model-name"}'
```

---

## Best Practices

### 1. Versioning

Use checkpoint metadata to track versions:

```rust
CheckpointMetadata {
    epoch: 5,
    timestamp: "2026-02-08T12:00:00Z",
    // ...
}
```

### 2. Naming Conventions

- Use descriptive names: `model-dataset-method`
- Example: `mnist-cnn-hybrid-gpu`
- Include variant information in name

### 3. Documentation

- Always upload training results with models
- Include metadata for reproducibility
- Keep model cards updated

### 4. Security

- Never commit HF tokens to git
- Use environment variables only
- Revoke old tokens when generating new ones
- Use read-only tokens for deployment

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Train and Upload to HF

on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Train models
        run: |
          cargo run --release --example mnist_with_hf_upload \
            --features "autodiff,datasets"
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

---

## Resources

- **Hugging Face Docs:** https://huggingface.co/docs
- **API Reference:** https://huggingface.co/docs/huggingface_hub/main/en/guides/upload
- **Token Management:** https://huggingface.co/settings/tokens
- **Model Repos:** https://huggingface.co/tzervas

---

## Support

For issues or questions:
- **Repository:** https://github.com/tzervas/hybrid-predict-trainer-rs
- **Documentation:** https://docs.rs/hybrid-predict-trainer-rs
- **Email:** tz-dev@vectorweight.com

---

## License

MIT License - Models uploaded to Hugging Face follow the same license as the training code.
