//! `cargo bench -p rusty-benchmarks` runs the same suite as the `bench` binary.

fn main() {
    rusty_benchmarks::run(&rusty_benchmarks::Config::from_args());
}
