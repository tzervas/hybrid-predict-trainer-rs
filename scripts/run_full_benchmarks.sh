#!/bin/bash
# Comprehensive training and benchmarking script
# Trains multiple variants, runs benchmarks, generates reports

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RESULTS_DIR/$TIMESTAMP"

echo "======================================================================="
echo "🚀 Hybrid Predict Trainer - Full Benchmarking Suite"
echo "======================================================================="
echo ""
echo "Results will be saved to: $RUN_DIR"
echo ""

# Create results directory
mkdir -p "$RUN_DIR"

# Save system info
echo "📊 Collecting system information..."
cat > "$RUN_DIR/system_info.txt" <<EOF
Benchmark Run: $TIMESTAMP
System: $(uname -a)
Rust Version: $(rustc --version)
Cargo Version: $(cargo --version)
Hostname: $(hostname)
CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
Memory: $(free -h | grep Mem | awk '{print $2}')
EOF

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> "$RUN_DIR/system_info.txt"
    nvidia-smi >> "$RUN_DIR/gpu_info.txt" 2>&1
    HAS_CUDA=true
else
    echo "GPU: None (CPU only)" >> "$RUN_DIR/system_info.txt"
    HAS_CUDA=false
fi

cat "$RUN_DIR/system_info.txt"
echo ""

# Build all examples and benchmarks
echo "======================================================================="
echo "🔧 Building project..."
echo "======================================================================="
echo ""

if [ "$HAS_CUDA" = true ]; then
    cargo build --release --all-features
else
    cargo build --release --features "autodiff,datasets"
fi

echo ""
echo "✅ Build complete"
echo ""

# Run proof-of-concept
echo "======================================================================="
echo "1️⃣  Proof-of-Concept Validation"
echo "======================================================================="
echo ""

cargo run --release --example proof_of_concept --features "autodiff,datasets" \
    | tee "$RUN_DIR/01_proof_of_concept.log"

echo ""
echo "✅ Proof-of-concept complete"
echo ""

# Run MNIST with HF upload (without token for local test)
echo "======================================================================="
echo "2️⃣  MNIST Training Comparison"
echo "======================================================================="
echo ""

cargo run --release --example mnist_with_hf_upload --features "autodiff,datasets" \
    | tee "$RUN_DIR/02_mnist_comparison.log"

echo ""
echo "✅ MNIST comparison complete"
echo ""

# Run CPU benchmarks
echo "======================================================================="
echo "3️⃣  CPU Performance Benchmarks"
echo "======================================================================="
echo ""

echo "Running phase transition benchmarks..."
cargo bench --bench phase_transitions | tee "$RUN_DIR/03_bench_phase_transitions.log"

echo ""
echo "Running predictor performance benchmarks..."
cargo bench --bench predictor_performance | tee "$RUN_DIR/04_bench_predictor_performance.log"

echo ""
echo "Running residual extraction benchmarks..."
cargo bench --bench residual_extraction | tee "$RUN_DIR/05_bench_residual_extraction.log"

echo ""
echo "Running hybrid trainer benchmarks..."
cargo bench --bench hybrid_trainer_benchmarks | tee "$RUN_DIR/06_bench_hybrid_trainer.log"

echo ""
echo "✅ CPU benchmarks complete"
echo ""

# Run GPU benchmarks if available
if [ "$HAS_CUDA" = true ]; then
    echo "======================================================================="
    echo "4️⃣  GPU Performance Benchmarks"
    echo "======================================================================="
    echo ""

    echo "Running GPU kernel benchmarks..."
    cargo bench --bench gpu_kernels --features cuda | tee "$RUN_DIR/07_bench_gpu_kernels.log"

    echo ""
    echo "Running GPU memory profiling..."
    cargo bench --bench gpu_memory_profile --features cuda | tee "$RUN_DIR/08_bench_gpu_memory.log"

    echo ""
    echo "Running GPU performance benchmarks..."
    cargo bench --bench gpu_performance --features cuda | tee "$RUN_DIR/09_bench_gpu_performance.log"

    echo ""
    echo "✅ GPU benchmarks complete"
    echo ""
else
    echo "⚠️  Skipping GPU benchmarks (no CUDA detected)"
    echo ""
fi

# Generate summary report
echo "======================================================================="
echo "📊 Generating Summary Report"
echo "======================================================================="
echo ""

cat > "$RUN_DIR/SUMMARY.md" <<EOF
# Benchmark Results Summary

**Run Date:** $TIMESTAMP
**System:** $(hostname)

## System Configuration

$(cat "$RUN_DIR/system_info.txt")

## Training Results

### Proof-of-Concept
$(grep -A 10 "🎯 Key Metrics" "$RUN_DIR/01_proof_of_concept.log" || echo "See log for details")

### MNIST Comparison
$(grep -A 10 "🎯 Key Metrics" "$RUN_DIR/02_mnist_comparison.log" || echo "See log for details")

## Performance Benchmarks

### CPU Benchmarks
- Phase Transitions: See \`03_bench_phase_transitions.log\`
- Predictor Performance: See \`04_bench_predictor_performance.log\`
- Residual Extraction: See \`05_bench_residual_extraction.log\`
- Hybrid Trainer: See \`06_bench_hybrid_trainer.log\`

EOF

if [ "$HAS_CUDA" = true ]; then
    cat >> "$RUN_DIR/SUMMARY.md" <<EOF
### GPU Benchmarks
- GPU Kernels: See \`07_bench_gpu_kernels.log\`
- GPU Memory: See \`08_bench_gpu_memory.log\`
- GPU Performance: See \`09_bench_gpu_performance.log\`

EOF
fi

cat >> "$RUN_DIR/SUMMARY.md" <<EOF
## Files Generated

\`\`\`
$(ls -lh "$RUN_DIR" | tail -n +2)
\`\`\`

## Next Steps

1. Review benchmark results in individual log files
2. Set HF_TOKEN to upload models: \`export HF_TOKEN="your_token"\`
3. Re-run with HF upload: \`cargo run --release --example mnist_with_hf_upload --features "autodiff,datasets"\`
4. Analyze Criterion HTML reports in \`target/criterion/\`

## Benchmark Visualizations

Criterion has generated detailed HTML reports:
- Open \`target/criterion/report/index.html\` in your browser
- Interactive charts for all benchmarks
- Performance regression detection

EOF

cat "$RUN_DIR/SUMMARY.md"

echo ""
echo "======================================================================="
echo "✅ BENCHMARKING COMPLETE"
echo "======================================================================="
echo ""
echo "Results saved to: $RUN_DIR"
echo ""
echo "📊 Summary: $RUN_DIR/SUMMARY.md"
echo "📈 Criterion reports: file://$PROJECT_ROOT/target/criterion/report/index.html"
echo ""
echo "To upload models to Hugging Face:"
echo "  export HF_TOKEN='your_token_here'"
echo "  cargo run --release --example mnist_with_hf_upload --features 'autodiff,datasets'"
echo ""
