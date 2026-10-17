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
