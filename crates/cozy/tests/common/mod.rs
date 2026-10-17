//! A fixture for product tests: the real `cozy` binary, run with its own temp `HOME`, so its home
//! is `<temp>/.cozy`. They run only on hosted runners: on any other computer `Home::new()` answers
//! `None` and the test skips, so `cargo test` there never reaches the user's `~/.cozy`, their
//! services or their Calcifer.
#![allow(dead_code)]
#[cfg(unix)]
pub mod https;
pub mod package;
#[cfg(unix)]
pub mod tensord;

use std::{
    ffi::OsStr,
    io::Write as _,
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub struct Home {
    dir: tempfile::TempDir,
    /// The `cozy` it runs: this build's or COZY_TEST_BINARY, unless a test chose another
    /// (`use_binary`).
    binary: std::cell::RefCell<PathBuf>,
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
        let hosted = std::env::var_os("GITHUB_ACTIONS").is_some();
        if !hosted {
            // Past libtest's capture, so a skip shows in every log.
            let _ = writeln!(std::io::stderr(), "skipped: not on a hosted runner");
            return None;
        }
        // COZY_TEST_BINARY: a release's own `cozy` (owner-day.yml, called by release.yml).
        let binary = std::env::var_os("COZY_TEST_BINARY").map_or_else(
            || PathBuf::from(env!("CARGO_BIN_EXE_cozy")),
            PathBuf::from,
        );
        Some(Self {
            dir: tempfile::tempdir().expect("a temp dir"),
            binary: binary.into(),
        })
    }

    /// A fresh home whose current Hub is the job's Tensorhub at its public origin
    /// (`TENSORHUB_ORIGIN`, Tensorhub's dev-build `cozy_ref` job, trusted through the job's
    /// `SSL_CERT_FILE`), as a person's names tensorhub.com; `None` without one.
    pub fn with_hub() -> Option<Self> {
        let Some(url) = std::env::var_os("TENSORHUB_ORIGIN") else {
            let _ = writeln!(
                std::io::stderr(),
                "skipped: no Tensorhub (TENSORHUB_ORIGIN is unset)"
            );
            return None;
        };
        let home = Self::new()?;
        home.write_config(&format!(
            "hub = \"ci\"\n[hubs]\nci = \"{}\"\n",
            url.to_string_lossy()
        ));
        Some(home)
    }

    /// Signs `email` up at the home's Hub and keeps its device key where `cozy` reads it, as
    /// `cozy login` would; registers the account name before the `@`. Answers that name.
    pub fn sign_in(&self, email: &str) -> String {
        let url = std::env::var("TENSORHUB_ORIGIN").expect("a Tensorhub");
        let log = std::env::var("TENSORHUB_LOG").expect("the Tensorhub's log");
        let hub = cozy_proto::config::Hub {
            name: Some("ci".into()),
            origin: cozy_proto::config::origin(&url).unwrap(),
        };
        let home = cozy_platform::paths::Home::at(self.root());
        let name = email.split('@').next().unwrap().to_owned();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let http = reqwest::Client::new();
            let anonymous =
                hub_client::Client::connect(&url, http.clone(), hub_client::Credential::Anonymous);
            let (key, bearer) = hub_client::testing::sign_up(&anonymous, email, log.as_ref())
                .await
                .expect("sign up");
            std::fs::create_dir_all(home.keys()).unwrap();
            key.save(&hub.device_key(&home)).unwrap();
            let user = hub_client::Client::connect(
                &url,
                http,
                hub_client::Credential::Bearer(bearer.token),
            );
            user.register_account(&name, &hub_client::types::Empty(Default::default()))
                .await
                .expect("an account name");
        });
        name
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
        let mut command = Command::new(&*self.binary.borrow());
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

    /// Runs `binary` (another release's `cozy`) from now on.
    pub fn use_binary(&self, binary: impl Into<PathBuf>) {
        *self.binary.borrow_mut() = binary.into();
    }
}

impl Drop for Home {
    /// No Calcifer outlives its test.
    fn drop(&mut self) {
        if self
            .root()
            .join("calcifer")
            .join("calcifer.lock.holder")
            .exists()
        {
            let _ = self.command().arg("down").output();
        }
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
