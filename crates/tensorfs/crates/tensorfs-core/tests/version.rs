//! tfs-051: `tfs version` is the skew observable tensorhub and cozy-creator refuse on.

use std::process::Command;

#[test]
fn version_prints_release_and_own_digest() {
    let out = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .arg("version")
        .output()
        .expect("run tfs version");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let said = String::from_utf8(out.stdout).expect("utf8");
    let fields: Vec<&str> = said.split_whitespace().collect();
    assert_eq!(
        fields.len(),
        3,
        "line is `tfs <version> sha256:<hex>`, got {said:?}"
    );
    assert_eq!(fields[0], "tfs");
    assert_eq!(fields[1], env!("CARGO_PKG_VERSION"));
    let digest = fields[2]
        .strip_prefix("sha256:")
        .expect("sha256-prefixed digest");
    let bytes = std::fs::read(env!("CARGO_BIN_EXE_tfs")).expect("read the binary");
    assert_eq!(
        digest,
        tensorfs_core::sha256::hex_digest(&bytes),
        "digest is of its own bytes"
    );
}
