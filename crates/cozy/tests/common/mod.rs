//! A fixture for product tests: the real `cozy` binary, run with its own temp `HOME`, so its home
//! is `<temp>/.cozy`. They run only on hosted runners: on any other computer `Home::new()` answers
//! `None` and the test skips, so `cargo test` there never reaches the user's `~/.cozy`, their
//! services or their Calcifer.
#![allow(dead_code)]
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub struct Home {
    dir: tempfile::TempDir,
}

/// One finished `cozy` process.
#[derive(Debug)]
pub struct Out {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Home {
    /// A fresh home on a hosted runner; `None` anywhere else.
    pub fn new() -> Option<Self> {
        if std::env::var_os("GITHUB_ACTIONS").is_none() {
            eprintln!(
                "skipped: product tests run only on hosted runners (GITHUB_ACTIONS is unset)"
            );
            return None;
        }
        Some(Self {
            dir: tempfile::tempdir().expect("a temp dir"),
        })
    }

    /// The cozy home: `~/.cozy` under the child's `HOME`.
    pub fn root(&self) -> PathBuf {
        self.dir.path().join(".cozy")
    }

    pub fn config_path(&self) -> PathBuf {
        self.root().join("config.toml")
    }

    pub fn write_config(&self, text: &str) {
        std::fs::create_dir_all(self.root()).unwrap();
        std::fs::write(self.config_path(), text).unwrap();
    }

    pub fn config(&self) -> String {
        std::fs::read_to_string(self.config_path()).unwrap_or_default()
    }

    /// `cozy` with this home, no inherited credentials and no user service manager: a Calcifer
    /// it starts is its own detached process.
    pub fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cozy"));
        command
            .env("HOME", self.dir.path())
            .env("USERPROFILE", self.dir.path())
            .env("XDG_CONFIG_HOME", self.dir.path().join(".config"))
            .env("NO_COLOR", "1");
        for var in [
            "XDG_RUNTIME_DIR",
            "DBUS_SESSION_BUS_ADDRESS",
            "HF_TOKEN",
            "CIVITAI_TOKEN",
            "COZY_DOWNLOAD_URL",
            "COMPLETE",
        ] {
            command.env_remove(var);
        }
        command
    }

    pub fn cozy<I, S>(&self, args: I) -> Out
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        out(self.command().args(args).output().expect("cozy runs"))
    }

    /// Runs a command that must succeed and parses its `--json` document.
    pub fn json<I, S>(&self, args: I) -> serde_json::Value
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let out = self.cozy(args);
        assert_eq!(out.code, 0, "{out:?}");
        serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }
}

pub fn out(output: Output) -> Out {
    Out {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

impl Out {
    /// The refusal envelope a `--json` command printed.
    pub fn refusal(&self) -> serde_json::Value {
        let doc: serde_json::Value =
            serde_json::from_str(&self.stdout).unwrap_or_else(|e| panic!("{e}: {self:?}"));
        doc["error"].clone()
    }
}
