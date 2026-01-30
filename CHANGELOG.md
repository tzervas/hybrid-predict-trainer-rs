# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffolding and boilerplate
- Core trait definitions for hybrid predictive training
- Phase state machine infrastructure (Warmup, Full, Predict, Correct)
- Residual extraction and correction framework stubs
- GPU acceleration support via CubeCL and Burn
- Comprehensive documentation and development guidelines

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.0.1] - 2026-01-30

### Added
- Initial release with project structure and boilerplate
- Core module stubs for all major components:
  - `config`: Training configuration and serialization
  - `error`: Comprehensive error types with recovery actions
  - `phases`: Phase state machine and execution control
  - `warmup`: Warmup phase implementation
  - `full_train`: Full training phase implementation
  - `predictive`: Forward/backward predictive training
  - `residuals`: Residual extraction and storage
  - `corrector`: Prediction correction via residual application
  - `state`: Training state encoding and management
  - `dynamics`: RSSM-lite dynamics model for prediction
  - `divergence`: Multi-signal divergence detection
  - `metrics`: Training metrics collection and reporting
  - `gpu`: CubeCL and Burn GPU acceleration kernels
- MIT license
- README with architecture overview
- CLAUDE.md development context document
- Benchmark scaffolding
- Example programs

[Unreleased]: https://github.com/tzervas/hybrid-predict-trainer-rs/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/tzervas/hybrid-predict-trainer-rs/releases/tag/v0.0.1
