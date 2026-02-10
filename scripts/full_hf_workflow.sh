#!/bin/bash
# Complete Hugging Face workflow: Create repos + Train + Upload
# Requires HF_TOKEN environment variable

set -e

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║            🚀 Complete Hugging Face Workflow - Train & Upload                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN not set"
    echo ""
    echo "To proceed, export your token:"
    echo "  export HF_TOKEN='your_hf_token_here'"
    echo ""
    echo "Or run this script with the token:"
    echo "  HF_TOKEN='token' ./scripts/full_hf_workflow.sh"
    echo ""
    exit 1
fi

echo "✅ HF_TOKEN detected (${#HF_TOKEN} characters)"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Step 1: Create repositories
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "STEP 1: Creating Hugging Face Repositories"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

./scripts/create_hf_repos.sh

echo ""
echo "✅ Repositories created successfully!"
echo ""

# Step 2: Build release
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "STEP 2: Building Project"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

cargo build --release --features "autodiff,datasets" 2>&1 | tail -5

echo ""
echo "✅ Build complete!"
echo ""

# Step 3: Run proof-of-concept first
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "STEP 3: Running Proof-of-Concept Validation"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

cargo run --release --example proof_of_concept --features "autodiff,datasets" | tail -30

echo ""
echo "✅ Infrastructure validated!"
echo ""

# Step 4: Train and upload to HF
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "STEP 4: Training Models & Uploading to Hugging Face"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

cargo run --release --example mnist_with_hf_upload --features "autodiff,datasets"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "✅ COMPLETE WORKFLOW FINISHED"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Your models are now available on Hugging Face!"
echo ""
echo "🔗 View your models at:"
echo "   https://huggingface.co/tzervas"
echo ""
echo "📦 Repositories created:"
echo "   - https://huggingface.co/tzervas/mnist-cnn-traditional"
echo "   - https://huggingface.co/tzervas/mnist-cnn-hybrid"
echo "   - https://huggingface.co/tzervas/mnist-cnn-hybrid-gpu"
echo "   - https://huggingface.co/tzervas/gpt2-small-traditional"
echo "   - https://huggingface.co/tzervas/gpt2-small-hybrid"
echo ""
echo "✨ Each repository includes:"
echo "   - Auto-generated model card with metrics"
echo "   - Training results (JSON)"
echo "   - Checkpoint metadata"
echo "   - Performance comparisons"
echo ""
echo "🎉 Next steps:"
echo "   1. Visit your HF profile to verify uploads"
echo "   2. Share your models with the community"
echo "   3. Run benchmarks: ./scripts/run_full_benchmarks.sh"
echo ""
