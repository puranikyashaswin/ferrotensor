//! Lets a plain `cargo build` link the extension on macOS, where Python's
//! symbols are only resolved when the interpreter loads the module.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
