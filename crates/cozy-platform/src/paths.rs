//! Where things live (design §7). Locations only: the caller's config resolves the user's home
//! and hands it in, and nothing here reads the environment.
use crate::{Result, durable};
use std::path::{Path, PathBuf};

/// The `~/.cozy` home: `%USERPROFILE%\.cozy` on Windows, the same layout on every OS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Home {
    root: PathBuf,
}

impl Home {
    pub fn under(user_home: &Path) -> Self {
        Self::at(user_home.join(".cozy"))
    }

    /// A root the config names instead.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Creates the root, private to this user: it holds keys, tokens and run payloads.
    pub fn open(&self) -> Result<()> {
        durable::private_dir(&self.root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn bin(&self) -> PathBuf {
        self.root.join("bin")
    }

    pub fn binary(&self) -> PathBuf {
        self.bin().join(exe("cozy"))
    }

    pub fn config(&self) -> PathBuf {
        self.root.join("config.toml")
    }

    pub fn keys(&self) -> PathBuf {
        self.root.join("keys")
    }

    /// This controller's Ed25519 signer for the local machine and added machines.
    pub fn owner_key(&self) -> PathBuf {
        self.keys().join("owner.pem")
    }

    /// One rental's key (PKCS#8 PEM, 0600), named by the name it was asked under: a single path
    /// component.
    pub fn rental_key(&self, name: &str) -> PathBuf {
        self.keys().join(format!("rental-{name}.pem"))
    }

    pub fn calcifer(&self) -> PathBuf {
        self.root.join("calcifer")
    }

    pub fn calcifer_db(&self) -> PathBuf {
        self.calcifer().join("state.db")
    }

    pub fn calcifer_log(&self) -> PathBuf {
        self.calcifer().join("calcifer.log")
    }

    pub fn calcifer_lock(&self) -> PathBuf {
        self.calcifer().join("calcifer.lock")
    }

    /// TensorD's state, identity and logs when this computer is a machine.
    pub fn machine(&self) -> PathBuf {
        self.root.join("machine")
    }

    pub fn outputs(&self) -> PathBuf {
        self.root.join("outputs")
    }

    /// Service definitions this home installed, and on Windows the birth of each one it started.
    pub fn services(&self) -> PathBuf {
        self.root.join("services")
    }

    /// Tools cozy installed for this computer's own use (`cozy tools install`), never bundled.
    pub fn tools(&self) -> PathBuf {
        self.root.join("tools")
    }

    /// The owner's backups, and the one-shot importer's archive of the Go CLI's state.
    pub fn backups(&self) -> PathBuf {
        self.root.join("backups")
    }
}

/// A program's file name on this OS.
pub fn exe(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

/// The interpreter inside a venv: `bin/python`, or `Scripts\python.exe` on Windows.
pub fn venv_python(venv: &Path) -> PathBuf {
    venv_tool(venv, "python")
}

/// A console script inside a venv, spelled for this OS.
pub fn venv_tool(venv: &Path, name: &str) -> PathBuf {
    let dir = if cfg!(windows) { "Scripts" } else { "bin" };
    venv.join(dir).join(exe(name))
}

/// This OS user's own home, from the account database (never the environment): `pw_dir` on
/// Unix, the profile directory on Windows. `None` when it cannot be read.
pub fn login_home() -> Option<PathBuf> {
    login::home()
}

/// Whether `user_home` is this OS user's own home.
pub fn is_login_home(user_home: &Path) -> bool {
    let canonical =
        |path: &Path| std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    login_home().is_some_and(|login| canonical(&login) == canonical(user_home))
}

#[cfg(unix)]
mod login {
    use std::{ffi::CStr, os::unix::ffi::OsStrExt, path::PathBuf};

    pub(super) fn home() -> Option<PathBuf> {
        let mut buffer = vec![0u8; 16 << 10];
        // SAFETY: plain-old-data the call fills.
        let mut entry: libc::passwd = unsafe { std::mem::zeroed() };
        let mut found: *mut libc::passwd = std::ptr::null_mut();
        // SAFETY: geteuid has no arguments; the entry, buffer and result outlive the call.
        let read = unsafe {
            libc::getpwuid_r(
                libc::geteuid(),
                &mut entry,
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                &mut found,
            )
        };
        if read != 0 || found.is_null() || entry.pw_dir.is_null() {
            return None;
        }
        // SAFETY: a NUL-terminated string inside the buffer the call filled.
        let dir = unsafe { CStr::from_ptr(entry.pw_dir) };
        Some(PathBuf::from(std::ffi::OsStr::from_bytes(dir.to_bytes())))
    }
}

#[cfg(windows)]
mod login {
    use std::{
        os::windows::{
            ffi::OsStringExt,
            io::{FromRawHandle, OwnedHandle},
        },
        path::PathBuf,
    };
    use windows_sys::Win32::{
        Security::TOKEN_QUERY,
        System::Threading::{GetCurrentProcess, OpenProcessToken},
        UI::Shell::GetUserProfileDirectoryW,
    };

    pub(super) fn home() -> Option<PathBuf> {
        let mut token = std::ptr::null_mut();
        // SAFETY: this process's pseudo-handle and a writable token handle.
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
            return None;
        }
        // SAFETY: a new handle that is now ours.
        let token = unsafe { OwnedHandle::from_raw_handle(token) };
        let mut size = 0u32;
        // SAFETY: a size query: no buffer.
        unsafe {
            GetUserProfileDirectoryW(
                std::os::windows::io::AsRawHandle::as_raw_handle(&token),
                std::ptr::null_mut(),
                &mut size,
            )
        };
        let mut wide = vec![0u16; size as usize];
        // SAFETY: a buffer of `size` wide characters.
        if unsafe {
            GetUserProfileDirectoryW(
                std::os::windows::io::AsRawHandle::as_raw_handle(&token),
                wide.as_mut_ptr(),
                &mut size,
            )
        } == 0
        {
            return None;
        }
        let len = wide.iter().position(|&c| c == 0).unwrap_or(wide.len());
        Some(PathBuf::from(std::ffi::OsString::from_wide(&wide[..len])))
    }
}

#[cfg(test)]
mod login_tests {
    #[test]
    fn this_users_home_is_their_login_home_and_another_is_not() {
        let login = super::login_home().expect("the account database names a home");
        assert!(super::is_login_home(&login));
        let other = tempfile::tempdir().unwrap();
        assert!(!super::is_login_home(other.path()));
    }
}
