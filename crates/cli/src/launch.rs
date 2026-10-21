//! Starting Calcifer or this computer's machine: as its user service when there is one, else
//! detached, then waiting until it serves. One way for both: a service its manager gave up on or
//! restarted, or a detached process that exited, is the start's answer, never a wait.
use std::time::{Duration, Instant};

use platform::{
    process::Detached,
    service::{ServiceSpec, Services},
};
use protocol::error::Error;

/// How often a start is looked at.
const SAMPLE: Duration = Duration::from_millis(50);
/// How often a start that still waits says so.
const REPORT: Duration = Duration::from_secs(5);

/// A program started as its user service, else detached.
pub(crate) struct Launch<'a> {
    services: Option<&'a Services>,
    spec: &'a ServiceSpec,
    child: Option<Detached>,
}

/// How a started program ended before it served.
pub(crate) enum Ended {
    /// It failed: why.
    Failed(String),
    /// A detached one exited cleanly: another instance held its lock.
    Yielded,
}

impl<'a> Launch<'a> {
    pub(crate) fn new(services: Option<&'a Services>, spec: &'a ServiceSpec) -> Self {
        Self {
            services,
            spec,
            child: None,
        }
    }

    /// Starts it: its service (one already running is left as it is), else a detached process
    /// when this launch runs none.
    pub(crate) fn start(&mut self) -> platform::Result<()> {
        let spec = self.spec;
        match self.services {
            Some(services) => services.start(spec),
            None if self.child.is_none() => {
                let child =
                    platform::process::detach(&spec.program, &spec.args, spec.log.as_deref())?;
                self.child = Some(child);
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Whether it ended before it served, and how. A detached process that ended is forgotten,
    /// so the next `start` runs another.
    pub(crate) fn ended(&mut self) -> Option<Ended> {
        if let Some(services) = self.services {
            let failed = services.failed(&self.spec.name);
            return failed.then(|| {
                Ended::Failed("its service failed or was restarted before it served".into())
            });
        }
        let status = self.child.as_mut()?.try_wait().ok()??;
        self.child = None;
        Some(match status.success() {
            true => Ended::Yielded,
            false => Ended::Failed(format!("it exited ({status}) before it served")),
        })
    }
}

/// Polls `serves` until it answers, saying every few seconds that `what` still starts and where
/// `log` tells what it is doing. It ends on the program's answer or refusal, never on a clock.
pub(crate) async fn until<T>(
    json: bool,
    what: &str,
    log: &str,
    mut serves: impl AsyncFnMut() -> Result<Option<T>, Error>,
) -> Result<T, Error> {
    let began = Instant::now();
    let mut reported = began;
    loop {
        if let Some(answer) = serves().await? {
            return Ok(answer);
        }
        tokio::time::sleep(SAMPLE).await;
        if reported.elapsed() >= REPORT && !json {
            reported = Instant::now();
            let waited = began.elapsed().as_secs();
            anstream::eprintln!(
                "waiting for {what} to start ({waited}s so far); {log} says what it is doing"
            );
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn sh(script: &str) -> ServiceSpec {
        ServiceSpec {
            name: "launch-test".into(),
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
            log: None,
        }
    }

    /// A detached program that fails before it serves is the start's refusal, at once; one that
    /// exits cleanly yielded to another instance and is forgotten, so the next start runs again.
    #[tokio::test]
    async fn a_start_ends_on_the_programs_end_never_a_wait() {
        let failing = sh("exit 3");
        let mut launch = Launch::new(None, &failing);
        launch.start().unwrap();
        let refused = until(true, "it", "its log", async || match launch.ended() {
            Some(Ended::Failed(why)) => Err(Error::new(
                protocol::error::Kind::Unavailable,
                "failed",
                why,
            )),
            Some(Ended::Yielded) => panic!("a failure read as a yield"),
            None => Ok(None::<()>),
        });
        assert!(refused.await.unwrap_err().message.contains("exited"));

        let yielding = sh("exit 0");
        let mut launch = Launch::new(None, &yielding);
        launch.start().unwrap();
        let yielded = until(true, "it", "its log", async || {
            Ok(launch.ended().map(|ended| matches!(ended, Ended::Yielded)))
        });
        assert!(yielded.await.unwrap());
        assert!(launch.child.is_none(), "a yielded start is forgotten");
    }
}
