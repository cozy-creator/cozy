//! CPU-only process fixture compiling the actual private filesystem guard, not a copy.

#[cfg(target_os = "linux")]
#[path = "../src/filesystem_identity.rs"]
mod filesystem_identity;
#[cfg(target_os = "linux")]
use tensorfs_core::store;

#[cfg(target_os = "linux")]
fn main() -> std::io::Result<()> {
    use std::fs;
    use std::io::{Read, Write};
    use std::os::unix::fs::MetadataExt;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::PathBuf;

    use rustix::process::{
        getpid, parent_process_death_signal, set_parent_process_death_signal, Signal,
    };

    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(&args[2]);
    if args[1] == "setup" {
        store::Store::init(&root).expect("real native Store init");
        return Ok(());
    }
    let signal = match args[3].as_str() {
        "0" => None,
        "9" => Some(Signal::KILL),
        "15" => Some(Signal::TERM),
        _ => panic!("fixture supports disabled, SIGKILL and SIGTERM"),
    };
    set_parent_process_death_signal(signal)?;
    let before = parent_process_death_signal()?.map_or(0, Signal::as_raw);
    let mut in_scope = before;
    let mut file_uid = 0;
    if args[1] == "store" {
        store::Store::open(&root).expect("real native foreign-owner Store open");
    } else {
        let mut operation = || -> std::io::Result<()> {
            let _guard = filesystem_identity::FilesystemIdentity::for_store_root(&root)?;
            in_scope = parent_process_death_signal()?.map_or(0, Signal::as_raw);
            match args[1].as_str() {
                "success" => {
                    let path = root.join("guard-fixture-data");
                    fs::write(&path, b"real scoped native I/O")?;
                    assert_eq!(fs::read(&path)?, b"real scoped native I/O");
                    file_uid = fs::metadata(&path)?.uid();
                }
                "error" => {
                    assert_eq!(
                        fs::read(root.join("absent-fixture-file"))
                            .unwrap_err()
                            .kind(),
                        std::io::ErrorKind::NotFound
                    );
                    return Err(std::io::Error::other("owned I/O failure"));
                }
                "unwind" => panic!("owned scope unwind"),
                "wait" => {
                    println!(
                        "{{\"ready\":true,\"guard\":{in_scope},\"pid\":{}}}",
                        getpid().as_raw_nonzero()
                    );
                    std::io::stdout().flush()?;
                    std::io::stdin().read_exact(&mut [0])?;
                }
                _ => panic!("unknown fixture operation"),
            }
            Ok(())
        };
        match args[1].as_str() {
            "unwind" => assert!(catch_unwind(AssertUnwindSafe(operation)).is_err()),
            "error" => assert!(operation().is_err()),
            _ => operation()?,
        }
    }
    let after = parent_process_death_signal()?.map_or(0, Signal::as_raw);
    let maps = fs::read_to_string("/proc/self/maps")?;
    let cuda_mapped = maps.contains("libcuda") || maps.contains("libnvidia");
    println!("{{\"before\":{before},\"in_scope\":{in_scope},\"after\":{after},\"file_uid\":{file_uid},\"cuda_mapped\":{cuda_mapped}}}");
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Linux-only parent-death credential fixture");
    std::process::exit(2);
}
