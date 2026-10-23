//! This computer's machine, `local`: always recorded, never enabled. Anything that uses it starts
//! its TensorD (`cozy machine serve`) when none runs, detached from Calcifer and from Calcifer's
//! service, so a Calcifer restart or `cozy down` never ends it. TensorD exits by itself after 15
//! minutes idle; its executors give their memory back after 5. One start at a time here, and
//! TensorD's lock on its root keeps one machine per root whoever starts it.
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime},
};

use platform::{
    lock::{Kind as LockKind, LockFile},
    paths::Home,
    service::{ServiceSpec, Services},
};
use protocol::{
    config,
    error::{Error, Kind},
};
use tracing::info;

use crate::store::Endpoint;

/// How often a start is looked at, and how often its log says it still starts.
const SAMPLE: Duration = Duration::from_millis(250);
const REPORT: Duration = Duration::from_secs(10);

pub(crate) struct Local {
    root: PathBuf,
    home: Home,
    services: Services,
    pub(crate) starting: tokio::sync::Mutex<()>,
}

impl Local {
    pub(crate) fn new(home: &Home, user_home: &Path, config: &config::Config) -> Self {
        Self {
            root: config.machine.root(home),
            home: home.clone(),
            services: Services::new(home, user_home),
            starting: tokio::sync::Mutex::new(()),
        }
    }

    fn lock(&self) -> PathBuf {
        self.root.join("var/lib/cozy/machine/agent.lock")
    }

    fn ready(&self) -> PathBuf {
        self.root.join("var/lib/cozy/tensord/api-ready.json")
    }

    fn start_log(&self) -> PathBuf {
        self.root.join("var/lib/cozy/machine/start.log")
    }

    /// Whether a TensorD holds this computer's machine root now.
    pub(crate) fn running(&self) -> bool {
        let lock = self.lock();
        lock.exists() && matches!(LockFile::new(lock).try_lock(LockKind::Exclusive), Ok(None))
    }

    /// Where the running TensorD serves, as its `api-ready.json` says.
    pub(crate) fn endpoint(&self) -> Result<Endpoint, Error> {
        crate::machines::endpoint_of(&self.ready())
    }

    /// Where the running TensorD serves, once it does (one still starting is waited for); none
    /// when none runs.
    pub(crate) async fn serving(&self) -> Option<Endpoint> {
        while self.running() {
            if let Ok(endpoint) = self.endpoint() {
                return Some(endpoint);
            }
            tokio::time::sleep(SAMPLE).await;
        }
        None
    }

    /// Another computer's controller is authorized to use this machine (`cozy machine
    /// authorize`): Calcifer starts it with itself, at login, and it stays up.
    pub(crate) fn authorized(&self) -> bool {
        let text = std::fs::read_to_string(self.home.keys().join("authorized")).unwrap_or_default();
        text.lines().any(|line| !line.split('#').next().unwrap_or_default().trim().is_empty())
    }

    /// Starts TensorD from the installed `cozy` and waits until it serves: its `api-ready.json`
    /// written after the start while it holds its root. A start that exits first is refused with
    /// its log; nothing ends the wait but TensorD's own answer.
    pub(crate) async fn start(&self) -> Result<Endpoint, Error> {
        let log = self.start_log();
        let failed = |why: String| {
            let message = format!("this computer's machine did not start: {why} (its log: {})", log.display());
            Error::new(Kind::Unavailable, "machine_start_failed", message)
        };
        let program = self.home.launcher().ok_or_else(|| {
            failed(format!("no cozy is installed in {}", self.home.root().display()))
        })?;
        let dir = log.parent().expect("a log has a directory");
        std::fs::create_dir_all(dir).map_err(|e| failed(format!("{}: {e}", dir.display())))?;
        let before = served_at(&self.ready());
        let spec = ServiceSpec {
            name: "tensord".into(),
            program,
            args: vec!["machine".into(), "serve".into()],
            log: Some(log.clone()),
        };
        let mut child = self.services.detach(&spec).map_err(|e| failed(e.to_string()))?;
        info!(pid = child.pid(), "starting this computer's machine");
        let (began, mut reported) = (Instant::now(), Instant::now());
        loop {
            if served_at(&self.ready()) > before && self.running() {
                info!(ms = began.elapsed().as_millis() as u64, "this computer's machine serves");
                return self.endpoint();
            }
            match child.try_wait() {
                // Another start holds the root: it serves soon, or lets go.
                Ok(Some(status)) if status.success() && self.running() => {}
                Ok(Some(status)) => {
                    return Err(failed(format!("the machine exited ({status}) before it served")));
                }
                Ok(None) => {}
                Err(e) => return Err(failed(e.to_string())),
            }
            if reported.elapsed() >= REPORT {
                reported = Instant::now();
                info!(s = began.elapsed().as_secs(), log = %log.display(), "this computer's machine still starts");
            }
            tokio::time::sleep(SAMPLE).await;
        }
    }

    /// `cozy machine enable`'s service from before the machine started itself: removed, and the
    /// machine it ran ended (the next use starts this build's). True when there was one. Only a
    /// login home had one; another home's (a test's) never touches the user's manager.
    pub(crate) fn retire_service(&self) -> Result<bool, Error> {
        let failed = |e: platform::Error| Error::new(Kind::Internal, "machine_service", e.to_string());
        if !self.services.available() || !self.services.uninstall("machine").map_err(failed)? {
            return Ok(false);
        }
        platform::process::end_holder(&self.lock()).map_err(failed)?;
        Ok(true)
    }
}

/// When the machine last wrote that it serves: its `api-ready.json`'s mtime.
fn served_at(ready: &Path) -> Option<SystemTime> {
    std::fs::metadata(ready).and_then(|m| m.modified()).ok()
}
