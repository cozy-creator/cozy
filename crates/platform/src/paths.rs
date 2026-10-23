//! Where things live (design §7). Locations only: the caller's config resolves the user's home
//! and hands it in, and nothing here reads the environment.
#![allow(unsafe_code)]
use crate::{Result, durable};
use std::path::{Path, PathBuf};
#[cfg(windows)]
use windows_sys::Win32::{
    Foundation::ERROR_FILE_NOT_FOUND,
    System::Registry::{
        HKEY_CURRENT_USER, RRF_NOEXPAND, RRF_RT_REG_EXPAND_SZ, RRF_RT_REG_SZ, RegGetValueW,
        RegSetKeyValueW,
    },
};

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

    /// Where the installer puts `cozy`.
    pub fn binary(&self) -> PathBuf {
        self.bin().join(exe("cozy"))
    }

    /// The path of the installed `cozy`, which Calcifer and the machine run, wherever its
    /// installer put it.
    pub fn installed(&self) -> PathBuf {
        self.root.join("installed")
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

/// A venv's site-packages: `lib/python3.X/site-packages`, or `Lib\site-packages` on Windows.
pub fn venv_site_packages(venv: &Path) -> Option<PathBuf> {
    if cfg!(windows) {
        return Some(venv.join("Lib").join("site-packages"));
    }
    std::fs::read_dir(venv.join("lib"))
        .ok()?
        .flatten()
        .map(|python| python.path().join("site-packages"))
        .find(|site| site.is_dir())
}

/// This OS user's own home, from the account database (never the environment): `pw_dir` on
/// Unix, the profile directory on Windows. `None` when it cannot be read.
pub fn login_home() -> Option<PathBuf> {
    login::home()
}

/// Removes `dir` from this user's PATH (`HKCU\Environment`), where the PowerShell installer put
/// it: whether it was there.
#[cfg(windows)]
pub fn remove_from_user_path(dir: &Path) -> Result<bool> {
    let wide = |text: &str| text.encode_utf16().chain([0]).collect::<Vec<u16>>();
    let (key, value) = (wide("Environment"), wide("Path"));
    let failed = |op, status: u32| crate::Error::Os {
        op,
        source: std::io::Error::from_raw_os_error(status as i32),
    };
    // An environment variable holds at most 32767 characters.
    let (mut kind, mut data) = (0u32, vec![0u16; 32768]);
    let mut bytes = (data.len() * 2) as u32;
    let flags = RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND;
    // SAFETY: NUL-terminated names; the buffer holds `bytes` bytes.
    let status = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            key.as_ptr(),
            value.as_ptr(),
            flags,
            &mut kind,
            data.as_mut_ptr().cast(),
            &mut bytes,
        )
    };
    match status {
        0 => {}
        ERROR_FILE_NOT_FOUND => return Ok(false),
        _ => return Err(failed("read the user PATH", status)),
    }
    let path = String::from_utf16_lossy(&data[..bytes as usize / 2]);
    let path = path.trim_end_matches('\0');
    let ours = dir.to_string_lossy();
    let same = |entry: &str| {
        entry
            .trim_end_matches('\\')
            .eq_ignore_ascii_case(ours.trim_end_matches('\\'))
    };
    let kept: Vec<&str> = path.split(';').filter(|entry| !same(entry)).collect();
    if kept.len() == path.split(';').count() {
        return Ok(false);
    }
    let kept = wide(&kept.join(";"));
    // SAFETY: NUL-terminated names and the data's exact byte length.
    let status = unsafe {
        RegSetKeyValueW(
            HKEY_CURRENT_USER,
            key.as_ptr(),
            value.as_ptr(),
            kind,
            kept.as_ptr().cast(),
            (kept.len() * 2) as u32,
        )
    };
    match status {
        0 => Ok(true),
        _ => Err(failed("write the user PATH", status)),
    }
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
