//! Command-line entry point: `cargo run -p ferrotensor-benchmarks --release [-- --quick] [-- --filter matmul]`.

fn main() {
    ferrotensor_benchmarks::run(&ferrotensor_benchmarks::Config::from_args());
}
