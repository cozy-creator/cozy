use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(require_aot)");
    println!("cargo:rerun-if-env-changed=TENSORFS_REQUIRE_AOT");
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let aot = root.join("kernels/aot");
    let catalog = aot.join("catalog.rs");
    // Cargo treats a missing watched path as changed on every invocation. Watch the
    // nearest existing parent so creating/removing the optional catalog still invalidates it.
    let mut watch = aot.as_path();
    while !watch.exists() {
        watch = watch.parent().expect("the crate directory exists");
    }
    println!("cargo:rerun-if-changed={}", watch.display());
    println!("cargo:rerun-if-changed=../tensorfs-py/kernels");
    if env::var("TENSORFS_REQUIRE_AOT").is_ok_and(|v| v == "1") {
        assert!(catalog.exists(), "CUDA wheels require prebuilt TensorFS kernels; run crates/tensorfs-py/kernels/build.py first");
        println!("cargo:rustc-cfg=require_aot");
    }
    let source = if catalog.exists() {
        for name in ["kernels.py", "build.py", "abi.py"] {
            assert_eq!(
                fs::read(root.join("../tensorfs-py/kernels").join(name)).unwrap(),
                fs::read(root.join("kernels/aot").join(format!("source-{name}")))
                    .expect("AOT source receipt"),
                "prebuilt TensorFS kernels are stale; rebuild the AOT catalog"
            );
        }
        fs::read_to_string(catalog).expect("read the prebuilt CUDA kernel catalog")
    } else {
        "pub static BINARIES: &[Binary] = &[];\n".into()
    };
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("kernels.rs");
    if fs::read(&output).ok().as_deref() != Some(source.as_bytes()) {
        fs::write(output, source).unwrap();
    }
}
