# Contributing to Rusty

Thank you for your interest in contributing to Rusty! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Rust 1.75 or later (stable channel)
- Git
- A GPU with WebGPU support is recommended for most training, tests, and benchmarks

### Setup

```bash
# Clone the repository
git clone https://github.com/puranikyashaswin/rusty.git
cd rusty

# Build the project
cargo build --workspace

# Run tests
cargo test --workspace

# Run clippy for linting
cargo clippy --workspace -- -D warnings

# Format code
cargo fmt --all
```

## Code Style

We follow standard Rust formatting conventions. Before submitting a PR:

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

### Guidelines

- Use descriptive variable names (avoid single-letter names except in iterators)
- Document all public items with doc comments
- Write tests for new functionality
- Handle errors gracefully (avoid `.unwrap()` in user-facing code)
- Follow the existing code structure and patterns

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`cargo test --workspace`)
6. Run clippy and fix any warnings (`cargo clippy --workspace -- -D warnings`)
7. Format your code (`cargo fmt --all`)
8. Commit your changes with clear commit messages
9. Push to your fork
10. Open a Pull Request

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add tensor broadcasting support
fix: prevent NaN panic in argmax
docs: add installation guide
refactor: simplify gradient computation
```

## Architecture Overview

Rusty is organized as a Cargo workspace with the following crates:

- `rusty` - Core library (tensors, autograd, neural network modules)
- `rusty-backend` - GPU compute engine and WGSL kernels
- `rusty-autograd` - Automatic differentiation engine
- `rusty-cli` - Command-line interface
- `rusty-trainer` - Training utilities and abstractions
- `rusty-graph` - Model loading and graph construction
- `rusty-loader` - Model file loading (safetensors, tokenizer)
- `rusty-monitor` - Training metrics and monitoring TUI
- `rusty-hub` - Model hub helpers
- `benchmarks` - GPU performance benchmarks (`rusty-benchmarks`)

## Testing

### Training Scope

The main user-facing priority is lightweight fine-tuning on consumer hardware. Contributions that improve the following are especially valuable:

- LoRA fine-tuning correctness and resume safety
- Small-model memory use and predictable CLI defaults
- Packed dataset handling and lightweight evaluation workflows
- Honest monitoring and reproducible benchmark reporting

### Unit Tests

Add unit tests alongside your code:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        // ...
    }
}
```

### Integration Tests

Add integration tests in the `tests/` directory of each crate.

### Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p rusty

# Specific test
cargo test test_name
```

## Reporting Issues

- **Bug reports**: Include Rust version, OS, and steps to reproduce
- **Feature requests**: Describe the use case and proposed solution
- **Security issues**: Report privately via security email

## License

By contributing, you agree that your contributions will be licensed under the project's license.
