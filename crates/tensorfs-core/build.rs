use std::{env, fs, path::PathBuf};

fn main() {
    let catalog =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("kernels/aot/catalog.rs");
    println!("cargo:rerun-if-changed=kernels/aot");
    let source = if catalog.exists() {
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
