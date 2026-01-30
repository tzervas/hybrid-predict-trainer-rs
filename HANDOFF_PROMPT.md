# Hybrid Predict Trainer RS - Handoff Prompt for Claude Code

## Quick Start Command

To continue development of this crate with Claude Code:

```bash
cd /home/kang/Documents/projects/rust-ai/hybrid-predict-trainer-rs
claude -p "$(cat HANDOFF_PROMPT.md)"
```

Or for interactive mode:

```bash
cd /home/kang/Documents/projects/rust-ai/hybrid-predict-trainer-rs
claude
```

Then paste this prompt to begin.

---

## Development Handoff Prompt

You are continuing development of **hybrid-predict-trainer-rs**, a Rust crate implementing hybridized predictive training for deep learning.

### Project Context

**Location**: `/home/kang/Documents/projects/rust-ai/hybrid-predict-trainer-rs`
**Part of**: rust-ai workspace
**Version**: 0.0.1
**License**: MIT
**Author**: Tyler Zervas (tzervas) tz-dev@vectorweight.com
**MSRV**: 1.92

### What's Been Done

The crate boilerplate and structure is complete:

```
hybrid-predict-trainer-rs/
├── Cargo.toml           ✅ Complete with all dependencies
├── LICENSE-MIT          ✅ Complete
├── CHANGELOG.md         ✅ Complete
├── README.md            ✅ Complete
├── CLAUDE.md            ✅ Development context
└── src/
    ├── lib.rs           ✅ HybridTrainer struct with module exports
    ├── config.rs        ✅ Configuration with builder pattern
    ├── error.rs         ✅ Error types with recovery actions
    ├── state.rs         ✅ TrainingState, RingBuffer, StateEncoder
    ├── phases.rs        ✅ Phase enum, PhaseController trait
    ├── warmup.rs        ✅ WarmupExecutor, WarmupStatistics
    ├── full_train.rs    ✅ FullTrainExecutor, GradientObservation
    ├── predictive.rs    ✅ PredictiveExecutor, PhasePrediction
    ├── residuals.rs     ✅ Residual, ResidualStore, compression
    ├── corrector.rs     ✅ ResidualCorrector, CorrectionExecutor
    ├── dynamics.rs      ✅ RSSMLite, DynamicsModel trait
    ├── divergence.rs    ✅ DivergenceMonitor, multi-signal detection
    ├── bandit.rs        ✅ BanditSelector, LinUCB algorithm
    ├── metrics.rs       ✅ MetricsCollector, TrainingStatistics
    └── gpu.rs           ✅ GPU module stub (feature-gated)
```

### Core Architecture

The training loop cycles through 4 phases:

1. **Warmup**: Collect baseline statistics
2. **Full Train**: Standard training + train dynamics model
3. **Predict**: Skip backward passes using predictions
4. **Correct**: Apply residual corrections

Target: **5-10x training speedup** with <2% loss quality degradation.

### Key Dependencies

- `burn` 0.20.1 - Deep learning framework
- `cubecl` 0.9.0 - GPU compute
- `serde` 1.0.228 - Serialization
- `thiserror` 2.0.18 - Error handling
- `rand` (latest) - Random numbers
- `half` 2.7.1 - f16/bf16 support

### What Needs to Be Done (Priority Order)

#### Phase 1: Wire Up the Training Loop
1. Implement `HybridTrainer::step()` to actually execute phases
2. Connect phase executors to the state machine
3. Wire up the dynamics model to make real predictions
4. Implement checkpointing/restore

#### Phase 2: GPU Acceleration
1. Implement CubeCL CUDA kernels in `gpu.rs`
2. Add Burn tensor operations for model inference
3. GPU state encoding and prediction

#### Phase 3: Integration & Testing
1. Integration with Burn models/optimizers
2. End-to-end tests with simple models
3. Benchmark suite with Criterion

#### Phase 4: Examples & Documentation
1. Example training scripts
2. Integration examples (candle, tch-rs)
3. Benchmarking on standard tasks

### Development Commands

```bash
# Build
cargo build

# Build with CUDA
cargo build --features cuda

# Test
cargo test

# Test specific module
cargo test divergence::tests

# Check compilation
cargo check --all-features

# Lint
cargo clippy --all-features

# Documentation
cargo doc --open
```

### Code Quality Guidelines

1. All public items need doc comments with examples
2. Use `HybridResult<T>` for error handling with recovery hints
3. Define traits for extensibility (new predictors, backends)
4. GPU code behind `cuda` feature flag
5. Unit tests in each module

### Research Background

This implementation is based on:
- **DreamerV3** (Hafner 2023): RSSM architecture
- **PowerSGD** (Vogels 2019): Gradient compression
- **LinUCB**: Bandit algorithm for adaptive selection

### Start Here

Read `CLAUDE.md` for full development context, then:

1. First, verify the crate compiles:
   ```bash
   cargo check
   ```

2. Then run existing tests:
   ```bash
   cargo test
   ```

3. Start implementing `HybridTrainer::step()` in `src/lib.rs`

### Git Repository

After completing a development session:

```bash
git add -A
git commit -m "feat: <description of changes>"
git push origin dev
```

The repository should be at: `github.com/tzervas/hybrid-predict-trainer-rs`

---

## Claude Code Reference

### Useful Options

```bash
# Interactive mode
claude

# Print mode (for scripts)
claude -p "prompt"

# Continue previous session
claude -c

# Resume specific session
claude -r

# Debug mode
claude -d

# Specific model
claude --model opus
```

### Available Built-in Tools

- `Bash` - Run shell commands
- `Edit` - Edit files
- `Read` - Read files
- `WebSearch` - Search the web
- `WebFetch` - Fetch web content

### Permission Modes

- `default` - Ask for permissions
- `acceptEdits` - Auto-accept file edits
- `plan` - Planning mode, no edits

---

*Generated for hybrid-predict-trainer-rs v0.0.1*
