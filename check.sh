#!/usr/bin/env bash
# Runs the CI checks locally.
set -eux

cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo test --doc
