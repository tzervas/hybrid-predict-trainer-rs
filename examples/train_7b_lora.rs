//! Fine-tune a 7B LLM using LoRA + Hybrid Predictive Training.
//!
//! ## Architecture
//!
//! Unlike pretraining (train_2_8b.rs), this example:
//! 1. Loads a pre-trained 7B base model from HuggingFace (safetensors format)
//! 2. Freezes base model weights (no gradients for 7B parameters)
//! 3. Injects LoRA adapters (r=16) into attention Q/V projections
//! 4. Trains ONLY the LoRA adapters (~0.4% of params) with HybridTrainer
//!
//! ## VRAM Budget (16GB RTX 5080)
//!
//! ```text
//! Frozen base (bf16 loaded):        ~14GB   (7B × 2 bytes)
//! LoRA adapters (fp32, r=16):       ~0.4GB  (2 × 32 × 2 × 4096 × 16 × 32 layers)
//! Optimizer states (LoRA only):     ~0.8GB
//! Activations:                      ~1.0GB
//! ─────────────────────────────────────────
//! Total peak:                      ~16.2GB  ← tight, needs activation checkpointing
//! ```
//!
//! ## Data
//!
//! Reads parquet shards from:
//! - /data/datasets/tritter/pretrain/fineweb-edu-100B/  (100B tokens)
//! - /data/datasets/tritter/pretrain/wikipedia/         (67GB)
//! - /data/datasets/tritter/pretrain/openwebmath/       (26GB)
//!
//! ## Usage
//!
//! ```bash
//! # Download base model first (requires ~14GB disk for bf16 safetensors):
//! python3 -c "
//! from huggingface_hub import snapshot_download
//! snapshot_download('meta-llama/Llama-3.1-8B', local_dir='/data/models/llama-3.1-8b')
//! "
//!
//! # Run LoRA fine-tuning
//! cargo run --example train_7b_lora --features "autodiff,cuda,datasets" --release
//! ```
//!
//! ## Status
//!
//! TODO: This example is a planned skeleton. Requires:
//! - [ ] Burn bf16 backend configuration (Autodiff<Cuda, BF16>)
//! - [ ] Safetensors weight loader → Burn tensor conversion
//! - [ ] LoRA adapter injection using peft-rs crate (workspace sibling)
//! - [ ] LLaMA/Mistral model architecture in Burn (or candle bridge)
//! - [ ] Activation checkpointing to fit 16GB
//! - [ ] HybridTrainer wiring for LoRA-only gradient updates

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  7B LoRA Fine-tuning: Planned Architecture");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("VRAM Budget (16GB RTX 5080):");
    println!("  Frozen base (bf16):     ~14.0GB  (7B × 2 bytes)");
    println!("  LoRA adapters (fp32):    ~0.4GB");
    println!("  Optimizer (LoRA only):   ~0.8GB");
    println!("  Activations:             ~1.0GB");
    println!("  ─────────────────────────────────");
    println!("  Total peak:             ~16.2GB  [needs activation checkpointing]");

    println!("\nRequired components (TODO):");
    println!("  1. bf16 backend: Autodiff<Cuda, BF16>");
    println!("  2. Safetensors loader: load 7B weights into Burn bf16 tensors");
    println!("  3. LoRA injection: peft-rs for adapter layers (r=16, alpha=32)");
    println!("  4. LLaMA architecture: Burn impl with RoPE + SwiGLU + GQA");
    println!("  5. Activation checkpointing: recompute activations to save VRAM");

    println!("\nBase model options (HF hub):");
    println!("  - meta-llama/Llama-3.1-8B     (bf16 safetensors, ~14.8GB)");
    println!("  - mistralai/Mistral-7B-v0.1   (bf16 safetensors, ~13.5GB)");
    println!("  - Qwen/Qwen2.5-7B             (bf16 safetensors, ~13.8GB)");

    println!("\nDatasets (available on /data):");
    println!("  - fineweb-edu-100B:  100B tokens (high quality educational)");
    println!("  - wikipedia:         67GB (encyclopedic knowledge)");
    println!("  - openwebmath:       26GB (mathematical reasoning)");
    println!("  - finemath-4plus:    18GB (high-quality math)");

    println!("\nNext steps:");
    println!("  1. Implement Burn bf16 backend switch in Gpt2Config");
    println!("  2. Add safetensors loader to src/models/");
    println!("  3. Wire peft-rs LoRA adapters to HybridTrainer");
    println!("  4. Test with gpt2_1_3b_bf16() before scaling to 7B");

    println!("\nSee CLAUDE.md for implementation roadmap.");
}
