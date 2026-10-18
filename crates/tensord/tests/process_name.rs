//! The actual CPU service keeps its name when launched through update/proc executable aliases.
//! Linux: the name and executable are read from `/proc`.
#![cfg(target_os = "linux")]
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
};

struct OwnedService {
    child: Child,
    root: PathBuf,
}

impl OwnedService {
    fn start(executable: &Path, root: PathBuf) -> Self {
        fs::create_dir_all(&root).unwrap();
        let child = Command::new(executable)
            .arg("serve").arg("--state").arg(root.join("state"))
            .stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
        let mut service = Self { child, root };
        let process = PathBuf::from(format!("/proc/{}", service.child.id()));
        // Started once it owns its state: the lock it takes after naming itself.
        while !service.root.join("state/owner.lock").exists() {
            assert!(service.child.try_wait().unwrap().is_none(), "service did not start");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(fs::read_to_string(process.join("comm")).unwrap().trim(), "tensord");
        let maps = fs::read_to_string(process.join("maps")).unwrap();
        assert!(!maps.contains("libcuda") && !maps.contains("libnvidia-ml"));
        assert!(service.child.try_wait().unwrap().is_none());
        service
    }
}

impl Drop for OwnedService {
    fn drop(&mut self) {
        // Explicit teardown of these test-owned services, never a product timeout or sweep.
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn cpu_service_name_survives_current_and_proc_executable_aliases() {
    let directory = std::env::temp_dir().join(format!("tensord-name-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let current = directory.join("current");
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_tensord"), &current).unwrap();
    let first = OwnedService::start(&current, directory.join("current-state"));
    // /proc/self/exe from inside TensorD has this same executable alias. From this test,
    // `self` would be the test binary, so use the live service's exact PID instead.
    let executable = PathBuf::from(format!("/proc/{}/exe", first.child.id()));
    let second = OwnedService::start(&executable, directory.join("proc-state"));
    drop(second);
    drop(first);
    fs::remove_file(current).unwrap();
    fs::remove_dir(directory).unwrap();
}
