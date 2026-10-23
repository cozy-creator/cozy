//! Each remote machine's logs, kept by Calcifer as the machine writes them, so a machine that is
//! gone (a rental released, a pod lost mid-run) can still be read: `cozy machine logs <m> <log>`
//! answers the kept copy once the machine cannot. This computer's machine keeps its own.
use std::{
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};

use futures_util::StreamExt as _;
use protocol::calcifer::{MachineKind, MachineLog, MachineState};

use crate::{link::LinkError, machines::Machines};

const LOGS: [MachineLog; 3] = [MachineLog::Tensord, MachineLog::Memory, MachineLog::Tensorfs];
/// The newest bytes kept of one log (and as many again before them, rotated once).
const KEEP: u64 = 16 << 20;
/// What a first follow reads back of a log written before Calcifer watched it.
const FIRST: u64 = 10_000;
/// How long the copy of an ended machine's logs is kept.
const KEPT_FOR: Duration = Duration::from_secs(7 * 24 * 3600);

impl Machines {
    /// The directory the machine's kept logs live in.
    fn kept(&self, name: &str) -> PathBuf {
        self.home.calcifer().join("machine-logs").join(name)
    }

    /// Follows the machine's logs for as long as it lives, appending each batch to its copy;
    /// a broken stream opens again from the place after the last batch kept.
    pub(crate) async fn keep_logs(self: Arc<Self>, name: String, kind: MachineKind) {
        if kind == MachineKind::Local {
            return;
        }
        let keepers = LOGS.map(|log| self.clone().keep_log(name.clone(), log));
        futures_util::future::join_all(keepers).await;
    }

    async fn keep_log(self: Arc<Self>, name: String, log: MachineLog) {
        let mut pause = Duration::ZERO;
        loop {
            match self.store().machine(&name) {
                Ok(Some(row)) if row.state != MachineState::Ended => {}
                _ => return,
            }
            if self.offline(&name).is_none() {
                // A log not written yet (or not kept by this machine) is asked again, paced; a
                // place kept that this machine does not know (it restarted) is read anew.
                if let Err(LinkError::Refused { .. }) = self.keep_some(&name, log, &mut pause).await {
                    let _ = fs::remove_file(self.kept(&name).join(format!("{}.after", file(log))));
                }
            }
            pause = (pause * 2).clamp(Duration::from_secs(1), Duration::from_secs(30));
            tokio::time::sleep(pause).await;
        }
    }

    /// Appends what the machine writes to the log until its stream ends.
    async fn keep_some(&self, name: &str, log: MachineLog, pause: &mut Duration) -> Result<(), LinkError> {
        let Ok(Some(link)) = self.calcifer.link(name) else {
            return Ok(());
        };
        let dir = self.kept(name);
        let after_file = dir.join(format!("{}.after", file(log)));
        let after = fs::read_to_string(&after_file).ok().filter(|a| !a.is_empty());
        let lines = if after.is_some() { 0 } else { FIRST };
        let mut batches = link.follow_log(crate::machines::log_name(log), lines, after.as_deref()).await?;
        while let Some(batch) = batches.next().await {
            let batch = batch?;
            *pause = Duration::ZERO;
            if let Err(error) = append(&dir, log, &batch.text).and_then(|()| fs::write(&after_file, &batch.after)) {
                tracing::warn!(machine = name, "keeping its {} log: {error}", file(log));
            }
        }
        Ok(())
    }

    /// The newest `lines` lines (0: all) of the copy Calcifer kept of the machine's log, if any.
    pub(crate) fn kept_log(&self, name: &str, log: MachineLog, lines: u64) -> Option<Vec<u8>> {
        let dir = self.kept(name);
        let path = dir.join(format!("{}.log", file(log)));
        let mut text = fs::read(path.with_extension("log.1")).unwrap_or_default();
        text.extend(fs::read(&path).ok()?);
        if lines == 0 {
            return Some(text);
        }
        let cut = text.iter().enumerate().rev().filter(|(_, b)| **b == b'\n').nth(lines as usize).map_or(0, |(i, _)| i + 1);
        Some(text.split_off(cut))
    }

    /// Removes the copies of machines that ended (or are no longer recorded) and have not been
    /// written for `KEPT_FOR`.
    pub(crate) fn prune_kept(&self) {
        let Ok(entries) = fs::read_dir(self.home.calcifer().join("machine-logs")) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let live = matches!(self.store().machine(&name), Ok(Some(row)) if row.state != MachineState::Ended);
            let old = (entry.metadata().and_then(|m| m.modified()).ok())
                .and_then(|at| SystemTime::now().duration_since(at).ok())
                .is_some_and(|age| age > KEPT_FOR);
            if !live && old {
                let _ = fs::remove_dir_all(entry.path());
            }
        }
    }
}

/// The name of the log's copy.
fn file(log: MachineLog) -> &'static str {
    match log {
        MachineLog::Tensord => "tensord",
        MachineLog::Tensorfs => "tensorfs",
        MachineLog::Memory => "memory",
    }
}

/// Appends `text` to the log's copy; past `KEEP` the copy so far becomes the older one.
fn append(dir: &Path, log: MachineLog, text: &str) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.log", file(log)));
    if fs::metadata(&path).is_ok_and(|m| m.len() >= KEEP) {
        fs::rename(&path, path.with_extension("log.1"))?;
    }
    let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
    file.write_all(text.as_bytes())?;
    if !text.is_empty() && !text.ends_with('\n') {
        file.write_all(b"\n")?;
    }
    Ok(())
}
