use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(require_aot)");
    println!("cargo:rerun-if-env-changed=TENSORFS_REQUIRE_AOT");
    let catalog =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("kernels/aot/catalog.rs");
    println!("cargo:rerun-if-changed=kernels/aot");
    println!("cargo:rerun-if-changed=../tensorfs/kernels");
    if env::var("TENSORFS_REQUIRE_AOT").is_ok_and(|v| v == "1") {
        assert!(catalog.exists(), "CUDA wheels require prebuilt TensorFS kernels; run crates/tensorfs/kernels/build.py first");
        println!("cargo:rustc-cfg=require_aot");
    }
    let source = if catalog.exists() {
        for name in ["kernels.py", "build.py"] {
            let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
            assert_eq!(
                fs::read(root.join("../tensorfs/kernels").join(name)).unwrap(),
                fs::read(root.join("kernels/aot").join(format!("source-{name}")))
                    .expect("AOT source receipt"),
                "prebuilt TensorFS kernels are stale; rebuild the AOT catalog"
            );
        }
        fs::read_to_string(catalog).expect("read the prebuilt CUDA kernel catalog")
    } else {
        "pub static BINARIES: &[Binary] = &[];\n".into()
    };
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("kernels.rs"),
        source,
    )
    .unwrap();
}
