//! Command-line entry point: `cargo run -p rusty-benchmarks --release [-- --quick] [-- --filter matmul]`.

fn main() {
    rusty_benchmarks::run(&rusty_benchmarks::Config::from_args());
}
