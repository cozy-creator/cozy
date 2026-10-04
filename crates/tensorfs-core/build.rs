//! Capture actual native source/dependency/build inputs, including same-version edits.
use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};
fn collect(path: &Path, files: &mut Vec<std::path::PathBuf>) {
    if path.is_dir() {
        for entry in fs::read_dir(path).unwrap() {
            collect(&entry.unwrap().path(), files);
        }
    } else {
        files.push(path.to_path_buf());
    }
}
fn main() {
    let mut files = Vec::new();
    for path in [
        "src",
        "../../vectors",
        "Cargo.toml",
        "../../Cargo.toml",
        "../../Cargo.lock",
        "build.rs",
    ] {
        println!("cargo:rerun-if-changed={path}");
        collect(Path::new(path), &mut files);
    }
    files.sort();
    let mut hash = Sha256::new();
    for file in files {
        let name = file.to_string_lossy();
        let bytes = fs::read(&file).unwrap();
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
    }
    for key in [
        "TARGET",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        hash.update(key.as_bytes());
        hash.update(env::var(key).unwrap_or_default().as_bytes());
        hash.update([0]);
    }
    let rustc = std::process::Command::new(env::var("RUSTC").unwrap())
        .arg("--version")
        .output()
        .unwrap();
    hash.update(&rustc.stdout);
    let digest: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    println!("cargo:rustc-env=TENSORFS_IMPLEMENTATION_SHA256={digest}");
}
