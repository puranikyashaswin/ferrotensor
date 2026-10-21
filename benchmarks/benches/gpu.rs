//! `cargo bench -p ferrotensor-benchmarks` runs the same suite as the `bench` binary.

fn main() {
    ferrotensor_benchmarks::run(&ferrotensor_benchmarks::Config::from_args());
}
