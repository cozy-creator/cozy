//! A development rental's SSH maintenance: the granted key for root and the image's sshd
//! config. Started by the stable parent, so it outlives service restarts.
//!
//! The key is kept outside `/root/.ssh`: a provider may write its own key there after the
//! container starts (vast.ai replaces `authorized_keys` about a second in). SSH is optional: a
//! machine without sshd, or not root, serves without it and says so once.
use std::io;
#[cfg(target_os = "linux")]
use std::{fs, io::Write, os::unix::fs::OpenOptionsExt, path::PathBuf, process::Command};

#[cfg(target_os = "linux")]
const KEY_DIR: &str = "/run/cozy-dev";
#[cfg(target_os = "linux")]
const KEYS: &str = "/run/cozy-dev/authorized_keys";

/// SSH into a dev machine is a Linux rental's.
#[cfg(not(target_os = "linux"))]
pub fn start(_public_key: &str) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "no SSH into a machine on this OS; see `cozy doctor`"))
}

/// Starts sshd for `public_key`; the error is the one warning a machine without SSH logs.
#[cfg(target_os = "linux")]
pub fn start(public_key: &str) -> io::Result<()> {
    let (Some(keygen), Some(sshd)) = (tool("ssh-keygen"), tool("sshd")) else {
        return Err(io::Error::other("sshd not found: no SSH into this dev machine; see `cozy doctor`"));
    };
    if !nix::unistd::geteuid().is_root() {
        return Err(io::Error::other("sshd needs root: no SSH into this dev machine; see `cozy doctor`"));
    }
    launch(public_key, keygen, sshd).map_err(|error| {
        io::Error::other(format!("sshd did not start ({error}): no SSH into this dev machine; see `cozy doctor`"))
    })
}

#[cfg(target_os = "linux")]
fn launch(public_key: &str, keygen: PathBuf, sshd: PathBuf) -> io::Result<()> {
    if public_key.len() > 8192 || public_key.contains(['\0', '\r', '\n']) {
        return Err(io::Error::other(
            "the SSH key must be one public key of at most 8192 bytes",
        ));
    }
    fs::create_dir_all(KEY_DIR)?;
    platform::durable::set_mode(std::path::Path::new(KEY_DIR), 0o700).map_err(std::io::Error::from)?;
    fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(KEYS)?
        .write_all(format!("{public_key}\n").as_bytes())?;
    fs::create_dir_all("/run/sshd")?;
    let keys = Command::new(keygen).arg("-A").output()?;
    if !keys.status.success() {
        return Err(io::Error::other(format!(
            "ssh host keys: {}",
            String::from_utf8_lossy(&keys.stderr)
        )));
    }
    Command::new(sshd)
        .args(["-D", "-e", "-f", "/opt/cozy/dev/sshd_config", "-o"])
        .arg(format!("AuthorizedKeysFile={KEYS}"))
        .spawn()?;
    Ok(())
}

/// `name` on this machine's PATH.
#[cfg(target_os = "linux")]
fn tool(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join(name)).find(|file| file.is_file())
}
