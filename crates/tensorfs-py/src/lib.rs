//! tensorfs-py: the `tensorfs` wheel's compiled half. It marshals; every byte-level fact comes
//! from `tensorfs` and `varena`. Torch stays outside: tensors cross as DLPack.

mod ceiling;
mod cleanup;
mod errors;
#[cfg(feature = "store")]
mod classref;
#[cfg(feature = "store")]
mod file;
#[cfg(feature = "store")]
mod store;
#[cfg(feature = "store")]
mod tensors;
#[cfg(feature = "store")]
mod credentials;
#[cfg(feature = "store")]
mod dlpack_source;
#[cfg(feature = "store")]
mod verification;
#[cfg(feature = "store")]
mod resident_decode;
#[cfg(feature = "store")]
mod varena;
#[cfg(feature = "store")]
mod allocator;

use pyo3::prelude::*;

fn python_distribution_version() -> String {
    let cargo_version = env!("CARGO_PKG_VERSION");
    let Some((release, development)) = cargo_version.split_once("-dev.") else {
        return cargo_version.to_string();
    };
    format!("{release}.dev{development}")
}

#[pymodule]
fn _ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The library's warnings reach the process's stderr. SPIKE DIAG: RUST_LOG=debug adds the
    // mechanism's own ops (target varena_diag).
    let level = std::env::var("RUST_LOG").ok().and_then(|l| l.parse().ok()).unwrap_or(tracing::level_filters::LevelFilter::INFO);
    let _ = tracing_subscriber::fmt().with_max_level(level).with_writer(std::io::stderr).try_init();
    errors::install(m)?;
    cleanup::install(m)?;
    ceiling::install(m)?;
    m.add("__version__", python_distribution_version())?;
    // The store format every process on the machine shares.
    m.add("STORE_FORMAT", tensorfs::STORE_FORMAT)?;
    #[cfg(feature = "store")]
    store::install(m)?;
    #[cfg(feature = "store")]
    credentials::install(m)?;
    #[cfg(feature = "store")]
    varena::install(m)?;
    Ok(())
}
