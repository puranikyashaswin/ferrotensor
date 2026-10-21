//! Ensures the Phase 0 reference fixture is available for parity tests.
//! Generates it via the pure-Python helper when missing.

use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/reference/phase0_linear_rmsnorm.json")
}

#[test]
fn phase0_fixture_exists_or_is_generated() {
    let path = fixture_path();
    if !path.exists() {
        let generator = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/reference/generate_phase0_pure.py");
        let status = std::process::Command::new("python3")
            .arg(&generator)
            .status()
            .expect("failed to spawn python3 for phase0 fixture generation");
        assert!(
            status.success(),
            "generate_phase0_pure.py failed; cannot run Phase 0 parity tests"
        );
    }
    assert!(path.exists(), "expected fixture at {}", path.display());
    let content = std::fs::read_to_string(&path).expect("read fixture");
    assert!(content.contains("linear_weight_hf_layout"));
    assert!(content.contains("rms_output"));
}
