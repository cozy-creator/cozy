fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = "proto/cozy/machine/v1/machine.proto";
    let mut config = tonic_prost_build::Config::new();
    config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);
    tonic_prost_build::configure()
        .build_transport(false)
        .compile_with_config(config, &[proto], &["proto"])?;
    println!("cargo:rerun-if-changed={proto}");
    Ok(())
}
