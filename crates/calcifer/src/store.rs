//! Calcifer's store: one SQLite file only Calcifer writes. A run's state lives in its row's
//! columns; what the CLI's event stream replays is derived from them (`events_after`). Durable
//! here: transitions, outputs at their newest revision, and log lines. Never progress.
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use protocol::{
    calcifer::{
        CHECKPOINT, Call, Event, ItemStatus, Kind, Machine, MachineKind, MachineState, OutputItem,
        Reason, Run, Status, Warning,
    },
    config::Origin,
    ids::{Digest, RunId},
    machine as v1,
};
use rusqlite::{Connection, OptionalExtension, Row, Transaction, TransactionBehavior, params};

/// The version at which runs' specs and outcomes are JSON.
/// The store's version (`PRAGMA user_version`), and the schema a new store is made with. A store
/// at an older version is from before cozy 0.9.4, which this cozy does not read.
const VERSION: i64 = 5;
const SCHEMA: &str = "
    CREATE TABLE machines (
        name TEXT PRIMARY KEY,
        kind TEXT NOT NULL CHECK (kind IN ('local', 'rental', 'added')),
        hub TEXT NOT NULL DEFAULT '',
        rental_id TEXT,
        worker_id TEXT,
        address TEXT,
        pin BLOB,
        key TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('booting', 'ready', 'ended')),
        served INTEGER NOT NULL DEFAULT 0,
        reason TEXT,
        created_at INTEGER NOT NULL,
        alias TEXT
    ) STRICT;
    CREATE UNIQUE INDEX machines_rental ON machines (rental_id) WHERE rental_id IS NOT NULL;
    CREATE UNIQUE INDEX machines_alias ON machines (alias) WHERE alias IS NOT NULL;
    CREATE TABLE runs (
        id TEXT PRIMARY KEY,
        number INTEGER NOT NULL UNIQUE,
        idempotency_key TEXT NOT NULL UNIQUE,
        digest TEXT NOT NULL,
        hub TEXT NOT NULL,
        kind TEXT NOT NULL,
        target TEXT NOT NULL,
        machine TEXT REFERENCES machines (name),
        pinned INTEGER NOT NULL,
        spec BLOB NOT NULL,
        out_dir TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('queued', 'sent', 'accepted', 'settled')),
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        cursor INTEGER NOT NULL DEFAULT 0,
        machine_number INTEGER,
        machine_state TEXT,
        status TEXT CHECK (status IN ('completed', 'failed', 'canceled')),
        reason_code TEXT,
        reason TEXT,
        outcome BLOB,
        seq INTEGER NOT NULL DEFAULT 1,
        state_seq INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL,
        accepted_at INTEGER,
        settled_at INTEGER,
        warnings TEXT NOT NULL DEFAULT '[]',
        inputs TEXT NOT NULL DEFAULT '[]',
        wants TEXT
    ) STRICT;
    CREATE INDEX runs_owed ON runs (machine) WHERE state <> 'settled';
    CREATE TABLE outputs (
        run TEXT NOT NULL REFERENCES runs (id),
        name TEXT NOT NULL,
        idx INTEGER NOT NULL,
        rev INTEGER NOT NULL,
        length INTEGER NOT NULL,
        digest TEXT NOT NULL,
        media_type TEXT NOT NULL,
        label TEXT NOT NULL DEFAULT '',
        duration_us INTEGER NOT NULL DEFAULT 0,
        done INTEGER NOT NULL DEFAULT 0,
        path TEXT,
        exported_rev INTEGER,
        created_seq INTEGER NOT NULL,
        seq INTEGER NOT NULL,
        failed_rev INTEGER,
        export_error TEXT,
        PRIMARY KEY (run, name, idx)
    ) STRICT, WITHOUT ROWID;
    CREATE INDEX outputs_unexported ON outputs (run) WHERE exported_rev IS NOT rev;
    CREATE TABLE run_logs (
        run TEXT NOT NULL REFERENCES runs (id),
        seq INTEGER NOT NULL,
        at_ms INTEGER NOT NULL,
        level TEXT NOT NULL,
        text TEXT NOT NULL,
        PRIMARY KEY (run, seq)
    ) STRICT, WITHOUT ROWID;
    CREATE TABLE asks (
        key TEXT PRIMARY KEY,
        hub TEXT NOT NULL,
        digest TEXT NOT NULL,
        body BLOB NOT NULL,
        machine TEXT,
        state TEXT NOT NULL CHECK (state IN ('pending', 'done', 'failed')),
        reason TEXT,
        created_at INTEGER NOT NULL,
        kind TEXT NOT NULL DEFAULT 'rent' CHECK (kind IN ('rent', 'end')),
        run TEXT REFERENCES runs (id)
    ) STRICT;";

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Calcifer's store: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error(
        "Calcifer's store is at version {0}: this cozy reads version 5, the stores of cozy 0.9.4 on"
    )]
    Version(i64),
    #[error("idempotency key {0:?} already names another submission")]
    Conflict(String),
    #[error("no machine named {0:?}")]
    UnknownMachine(String),
    #[error("Calcifer's store holds a row it cannot read: {0}")]
    Corrupt(String),
}

type Result<T> = std::result::Result<T, StoreError>;

/// Where a run is between its submission and its end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// In the outbox: never sent to its current machine.
    Queued,
    /// Crossed the send boundary: its machine may hold it.
    Sent,
    /// Its machine's first `RunState` was recorded.
    Accepted,
    Settled,
}

#[derive(Clone, Debug)]
pub struct RunRow {
    pub id: RunId,
    pub number: u64,
    pub kind: Kind,
    pub target: String,
    /// The run's Hub; none for local code.
    pub hub: Option<Origin>,
    pub machine: Option<String>,
    /// The caller named the machine: the run is never placed elsewhere.
    pub pinned: bool,
    /// The spec as it is sent.
    pub spec: v1::RunSpec,
    pub out_dir: PathBuf,
    pub state: State,
    pub cancel_requested: bool,
    /// The last machine sequence whose durable effect is recorded here.
    pub cursor: u64,
    pub machine_state: Option<String>,
    pub status: Option<Status>,
    pub reason: Option<Reason>,
    pub state_seq: u64,
    pub created_at_ms: i64,
    pub accepted_at_ms: Option<i64>,
    pub settled_at_ms: Option<i64>,
    pub warnings: Vec<Warning>,
    /// The staged objects its machine is written before the send.
    pub inputs: Vec<(Digest, u64)>,
    /// Its machine's outcome, once settled by one.
    pub outcome: Option<v1::Outcome>,
    /// Where an unplaced run goes: `any`, or `new[:sku]` to rent one.
    pub wants: String,
}

/// Which runs a listing answers, newest first.
#[derive(Debug, Default)]
pub struct RunFilter {
    pub status: Option<Status>,
    /// A prefix of the run's target (`org/pkg`).
    pub target: Option<String>,
    pub machine: Option<String>,
    /// Only runs numbered below this.
    pub before: Option<u64>,
    pub limit: u64,
}

/// A new run, recorded under its idempotency key.
pub struct NewRun {
    pub id: RunId,
    pub kind: Kind,
    pub target: String,
    pub hub: Option<Origin>,
    pub machine: Option<String>,
    pub pinned: bool,
    pub spec: v1::RunSpec,
    pub out_dir: PathBuf,
    pub warnings: Vec<Warning>,
    pub inputs: Vec<(Digest, u64)>,
    /// Where an unpinned run may go: `any`, `new` or `new:<sku>`.
    pub wants: Option<String>,
}

/// How to reach a machine: the id its caps name, its address and its TLS leaf's sha256.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub worker_id: String,
    pub address: String,
    pub pin: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineRow {
    pub name: String,
    pub kind: MachineKind,
    pub hub: String,
    pub rental_id: Option<String>,
    pub endpoint: Option<Endpoint>,
    /// The signer key file; the machine scopes runs to the key that sent them.
    pub key: PathBuf,
    pub state: MachineState,
    /// It had an endpoint once: runs it never confirmed may be placed elsewhere.
    pub served: bool,
    pub reason: Option<String>,
}

/// Durable effects of a run of machine frames, recorded in one transaction.
#[derive(Debug, Default)]
pub struct Batch {
    /// The highest durable sequence among the frames.
    pub cursor: u64,
    pub logs: Vec<(i64, String, String)>,
    pub products: Vec<v1::Product>,
    pub machine_state: Option<String>,
}

impl Batch {
    pub fn is_empty(&self) -> bool {
        self.cursor == 0
            && self.logs.is_empty()
            && self.products.is_empty()
            && self.machine_state.is_none()
    }
}

/// What recording a machine's acceptance found.
#[derive(Debug, PartialEq)]
pub enum Acceptance {
    /// Recorded now.
    Fresh(Vec<Event>),
    /// Recorded before (a reattach).
    Again,
    /// The run has moved on from this machine (or ended): the machine must not run it.
    Stale,
}

/// An output whose newest revision has not landed on this computer, which Calcifer still owes.
#[derive(Clone, Debug)]
pub struct OwedExport {
    pub run: RunId,
    pub number: u64,
    pub name: String,
    pub index: u32,
    pub length: u64,
}

/// One output item at its newest revision, as its export sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputRow {
    pub name: String,
    pub index: u32,
    pub rev: u64,
    pub length: u64,
    /// `sha256:<hex>` of this revision's bytes.
    pub digest: String,
    pub media_type: String,
    pub done: bool,
    /// Its file, holding revision `exported_rev`.
    pub path: Option<PathBuf>,
    pub exported_rev: Option<u64>,
    /// Why this revision will not be exported.
    pub export_error: Option<String>,
}

/// The store: writes on one connection, one at a time, each commit durable; reads on their own,
/// which WAL lets go on while a commit waits on the disk, so a slow flush never holds up a read.
pub struct Store {
    writer: Mutex<Connection>,
    reader: Mutex<Connection>,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| StoreError::Corrupt(e.to_string()))?;
        }
        let mut writer = Connection::open(path)?;
        writer.pragma_update(None, "journal_mode", "WAL")?;
        writer.pragma_update(None, "synchronous", "FULL")?;
        writer.pragma_update(None, "foreign_keys", "ON")?;
        match writer.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0))? {
            0 => {
                let tx = writer.transaction()?;
                tx.execute_batch(SCHEMA)?;
                tx.pragma_update(None, "user_version", VERSION)?;
                tx.commit()?;
            }
            VERSION => {}
            other => return Err(StoreError::Version(other)),
        }
        let reader = Connection::open(path)?;
        reader.pragma_update(None, "query_only", "ON")?;
        Ok(Self {
            writer: Mutex::new(writer),
            reader: Mutex::new(reader),
        })
    }

    fn read<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        f(&self.reader.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// One transaction. A commit can wait seconds on a busy disk's write-back: on a multi-thread
    /// runtime the worker says so first, so its other tasks (the API's among them) move on.
    fn write<T>(&self, f: impl FnOnce(&Transaction) -> Result<T>) -> Result<T> {
        let commit = || {
            let mut conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let value = f(&tx)?;
            tx.commit()?;
            Ok(value)
        };
        match tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()) {
            Ok(tokio::runtime::RuntimeFlavor::MultiThread) => tokio::task::block_in_place(commit),
            _ => commit(),
        }
    }

    pub fn put_machine(&self, machine: &MachineRow) -> Result<()> {
        self.write(|tx| put_machine_in(tx, machine))
    }

    pub fn machine(&self, name: &str) -> Result<Option<MachineRow>> {
        self.read(|conn| {
            conn.query_row(
                &format!("SELECT {MACHINE_COLUMNS} FROM machines WHERE name = ?1"),
                [name],
                machine_row,
            )
            .optional()?
            .transpose()
        })
    }

    pub fn machines(&self) -> Result<Vec<MachineRow>> {
        self.read(|conn| {
            let mut statement = conn.prepare(&format!(
                "SELECT {MACHINE_COLUMNS} FROM machines ORDER BY name"
            ))?;
            let rows = statement
                .query_map([], machine_row)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows.into_iter().collect()
        })
    }

    /// The machine a name, a former name (alias) or a rental id names.
    pub fn find_machine(&self, text: &str) -> Result<Option<MachineRow>> {
        self.read(|conn| {
            conn.query_row(
                &format!(
                    "SELECT {MACHINE_COLUMNS} FROM machines WHERE name = ?1 OR alias = ?1 OR rental_id = ?1
                     ORDER BY name = ?1 DESC LIMIT 1"
                ),
                [text],
                machine_row,
            )
            .optional()?
            .transpose()
        })
    }

    /// The name a renamed machine had before.
    pub fn alias(&self, name: &str) -> Result<Option<String>> {
        self.read(|conn| {
            Ok(conn
                .query_row("SELECT alias FROM machines WHERE name = ?1", [name], |r| {
                    r.get(0)
                })
                .optional()?
                .flatten())
        })
    }

    /// Renames a machine that has not served (its Hub named it): its runs move with it, and the
    /// old name stays its alias. An ended machine that held the name gives it up: its history
    /// keeps it as `<name>@<when it was recorded>`.
    pub fn rename_machine(&self, old: &str, new: &str) -> Result<bool> {
        self.write(|tx| {
            let renamable: bool = tx.query_row(
                "SELECT EXISTS (SELECT 1 FROM machines WHERE name = ?1 AND served = 0 AND state <> 'ended')
                     AND NOT EXISTS (SELECT 1 FROM machines WHERE alias = ?2 AND name <> ?1)",
                params![old, new],
                |r| r.get(0),
            )?;
            let holder: Option<(String, i64)> = tx
                .query_row("SELECT state, created_at FROM machines WHERE name = ?1", [new], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })
                .optional()?;
            match holder {
                _ if !renamable => return Ok(false),
                Some((state, _)) if state != "ended" => return Ok(false),
                Some((_, at)) => move_machine(tx, new, &format!("{new}@{at}"), None)?,
                None => {}
            }
            move_machine(tx, old, new, Some(old))?;
            Ok(true)
        })
    }

    /// Unsettled runs by machine: how many it accepted, and how many wait to be sent or accepted.
    pub fn machine_load(&self) -> Result<std::collections::HashMap<String, (u32, u32)>> {
        self.read(|conn| {
            let mut statement = conn.prepare(
                "SELECT machine, SUM(state = 'accepted'), SUM(state <> 'accepted') FROM runs
                 WHERE state <> 'settled' AND machine IS NOT NULL GROUP BY machine",
            )?;
            let rows = statement.query_map([], |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?))))?;
            Ok(rows.collect::<rusqlite::Result<_>>()?)
        })
    }

    /// Unsettled runs no machine was chosen for, oldest first, with where each may go.
    pub fn unplaced(&self) -> Result<Vec<RunRow>> {
        self.read(|conn| {
            let mut statement = conn.prepare(&format!(
                "SELECT {RUN_COLUMNS} FROM runs WHERE state = 'queued' AND machine IS NULL ORDER BY number"
            ))?;
            let rows = statement
                .query_map([], run_row)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows.into_iter().collect()
        })
    }

    /// Records an ask of a Hub under its key before it is made; the same key again answers the
    /// recorded ask (`false`), and with another intent (`digest`) is a conflict.
    pub fn begin_ask(&self, ask: &Ask) -> Result<(Ask, bool)> {
        self.write(|tx| {
            if let Some(recorded) = ask_in(tx, &ask.key)? {
                return if recorded.digest == ask.digest {
                    Ok((recorded, false))
                } else {
                    Err(StoreError::Conflict(ask.key.clone()))
                };
            }
            tx.execute(
                "INSERT INTO asks (key, hub, digest, body, machine, state, reason, created_at, kind, run)
                 VALUES (?1, ?2, ?3, ?4, ?5, 'pending', NULL, ?6, ?7, ?8)",
                params![
                    ask.key,
                    ask.hub,
                    ask.digest,
                    ask.body,
                    ask.machine,
                    now_ms(),
                    ask.kind.as_str(),
                    ask.run.as_ref().map(RunId::as_str),
                ],
            )?;
            Ok((ask.clone(), true))
        })
    }

    /// The rent ask that made the machine.
    pub fn rent_ask(&self, machine: &str) -> Result<Option<Ask>> {
        self.read(|conn| {
            conn.query_row(
                &format!("SELECT {ASK_COLUMNS} FROM asks WHERE machine = ?1 AND kind = 'rent' AND state = 'done'"),
                [machine],
                ask_row,
            )
            .optional()?
            .transpose()
        })
    }

    /// Asks a failed ask again.
    pub fn reopen_ask(&self, key: &str) -> Result<()> {
        self.write(|tx| {
            tx.execute("UPDATE asks SET state = 'pending', reason = NULL WHERE key = ?1 AND state = 'failed'", [key])?;
            Ok(())
        })
    }

    pub fn ask(&self, key: &str) -> Result<Option<Ask>> {
        self.read(|conn| ask_in(conn, key))
    }

    /// Asks not yet answered: what a starting Calcifer asks again.
    pub fn pending_asks(&self) -> Result<Vec<Ask>> {
        self.read(|conn| {
            let mut statement = conn.prepare(&format!(
                "SELECT {ASK_COLUMNS} FROM asks WHERE state = 'pending' ORDER BY created_at"
            ))?;
            let rows = statement
                .query_map([], ask_row)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows.into_iter().collect()
        })
    }

    /// Ends an ask: `done` with the machine it made, or `failed` with why. A rent ask that is
    /// done records its machine in the same transaction.
    pub fn settle_ask(
        &self,
        key: &str,
        outcome: std::result::Result<Option<&MachineRow>, &str>,
    ) -> Result<()> {
        self.write(|tx| {
            match outcome {
                Ok(machine) => {
                    if let Some(machine) = machine {
                        put_machine_in(tx, machine)?;
                    }
                    tx.execute(
                        "UPDATE asks SET state = 'done', machine = COALESCE(?2, machine) WHERE key = ?1",
                        params![key, machine.map(|m| &m.name)],
                    )?;
                }
                Err(reason) => {
                    tx.execute("UPDATE asks SET state = 'failed', reason = ?2 WHERE key = ?1", params![key, reason])?;
                }
            }
            Ok(())
        })
    }

    /// Records `new` under `key`, once: the same key again answers the recorded run (`false`), and
    /// with another submission (`digest`) is a conflict.
    /// Whether a run is recorded under `key`.
    pub fn recorded(&self, key: &str) -> Result<bool> {
        self.read(|db| {
            let found = db.query_row("SELECT 1 FROM runs WHERE idempotency_key = ?1", [key], |_| Ok(()));
            Ok(found.optional()?.is_some())
        })
    }

    pub fn submit(&self, key: &str, digest: &str, new: NewRun) -> Result<(RunRow, bool)> {
        let (id, fresh) = self.write(|tx| {
            let recorded: Option<(String, String)> = tx
                .query_row("SELECT id, digest FROM runs WHERE idempotency_key = ?1", [key], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })
                .optional()?;
            if let Some((id, recorded)) = recorded {
                return if recorded == digest { Ok((id, false)) } else { Err(StoreError::Conflict(key.into())) };
            }
            // A machine is named by its name, the name it had before its Hub named it, or its rental id.
            let machine = match &new.machine {
                None => None,
                Some(text) => Some(
                    tx.query_row(
                        "SELECT name FROM machines WHERE name = ?1 OR alias = ?1 OR rental_id = ?1
                         ORDER BY name = ?1 DESC LIMIT 1",
                        [text],
                        |r| r.get::<_, String>(0),
                    )
                    .optional()?
                    .ok_or_else(|| StoreError::UnknownMachine(text.clone()))?,
                ),
            };
            tx.execute(
                "INSERT INTO runs (id, number, idempotency_key, digest, hub, kind, target, machine, pinned, spec,
                     out_dir, state, created_at, warnings, inputs, wants)
                 VALUES (?1, (SELECT COALESCE(MAX(number), 0) + 1 FROM runs), ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                     ?10, 'queued', ?11, ?12, ?13, ?14)",
                params![
                    new.id.as_str(),
                    key,
                    digest,
                    new.hub.as_deref().unwrap_or_default(),
                    new.kind.as_str(),
                    new.target,
                    machine,
                    new.pinned,
                    serde_json::to_vec(&new.spec).expect("serializes"),
                    new.out_dir.to_string_lossy(),
                    now_ms(),
                    serde_json::to_string(&new.warnings).expect("serializes"),
                    serde_json::to_string(&new.inputs).expect("serializes"),
                    new.wants,
                ],
            )?;
            Ok((new.id.to_string(), true))
        })?;
        let row = self
            .run(&id)?
            .ok_or_else(|| StoreError::Corrupt(format!("run {id} vanished")))?;
        Ok((row, fresh))
    }

    pub fn run(&self, id: &str) -> Result<Option<RunRow>> {
        self.read(|conn| run_in(conn, id))
    }

    pub fn run_by_number(&self, number: u64) -> Result<Option<RunRow>> {
        self.read(|conn| {
            conn.query_row(
                &format!("SELECT {RUN_COLUMNS} FROM runs WHERE number = ?1"),
                [number as i64],
                run_row,
            )
            .optional()?
            .transpose()
        })
    }

    /// The runs `filter` names, newest first, and how many more match beyond its limit.
    pub fn runs(&self, filter: &RunFilter) -> Result<(Vec<RunRow>, u64)> {
        const MATCHING: &str = "FROM runs WHERE number < ?1
            AND (?2 IS NULL OR CASE state WHEN 'accepted' THEN 'in_progress' WHEN 'settled' THEN status
                ELSE 'queued' END = ?2)
            AND (?3 IS NULL OR substr(target, 1, length(?3)) = ?3)
            AND (?4 IS NULL OR machine = ?4)";
        self.read(|conn| {
            let status = filter.status.map(|status| match status {
                Status::Queued => "queued",
                Status::InProgress => "in_progress",
                Status::Completed => "completed",
                Status::Failed => "failed",
                Status::Canceled => "canceled",
            });
            let before = filter.before.map_or(i64::MAX, |n| n as i64);
            let matching = params![before, status, filter.target, filter.machine];
            let total: i64 =
                conn.query_row(&format!("SELECT COUNT(*) {MATCHING}"), matching, |r| {
                    r.get(0)
                })?;
            let mut statement = conn.prepare(&format!(
                "SELECT {RUN_COLUMNS} {MATCHING} ORDER BY number DESC LIMIT ?5"
            ))?;
            let rows = statement
                .query_map(
                    params![
                        before,
                        status,
                        filter.target,
                        filter.machine,
                        filter.limit as i64
                    ],
                    run_row,
                )?
                .collect::<rusqlite::Result<Vec<_>>>()?
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let older = (total as u64).saturating_sub(rows.len() as u64);
            Ok((rows, older))
        })
    }

    /// The runs not yet settled, oldest first: what a starting Calcifer drives.
    pub fn unsettled(&self) -> Result<Vec<RunId>> {
        self.read(|conn| {
            let mut statement =
                conn.prepare("SELECT id FROM runs WHERE state <> 'settled' ORDER BY number")?;
            let ids = statement
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            ids.into_iter()
                .map(|id| RunId::try_from(id).map_err(|e| StoreError::Corrupt(e.into())))
                .collect()
        })
    }

    /// Whether any unsettled run is written the staged object `digest`.
    pub fn input_in_use(&self, digest: &Digest) -> Result<bool> {
        self.read(|conn| {
            Ok(conn.query_row(
                "SELECT EXISTS (SELECT 1 FROM runs, json_each(runs.inputs)
                 WHERE runs.state <> 'settled' AND json_extract(json_each.value, '$[0]') = ?1)",
                [digest.to_string()],
                |r| r.get(0),
            )?)
        })
    }

    /// The run with what `?detail=full` adds: its measurements and its settled calls.
    pub fn detail(&self, row: &RunRow) -> Result<Run> {
        self.read(|conn| {
            let mut run = record(conn, row)?;
            run.measurements = row
                .outcome
                .as_ref()
                .and_then(|outcome| outcome.measurements.as_ref())
                .and_then(|measurements| serde_json::from_str(measurements.get()).ok());
            let mut statement = conn.prepare(
                "SELECT text FROM run_logs WHERE run = ?1 AND level = 'call' ORDER BY seq",
            )?;
            let calls = statement.query_map([row.id.as_str()], |r| r.get::<_, String>(0))?;
            for call in calls {
                run.calls.extend(serde_json::from_str::<Call>(&call?).ok());
            }
            Ok(run)
        })
    }

    /// Places an unplaced run on `machine`.
    pub fn place(&self, id: &str, machine: &str) -> Result<bool> {
        self.write(|tx| {
            Ok(tx.execute(
                "UPDATE runs SET machine = ?2 WHERE id = ?1 AND machine IS NULL AND state = 'queued'",
                params![id, machine],
            )? == 1)
        })
    }

    /// The send boundary: the run may reach `machine` only if this wins against a cancel.
    pub fn mark_sent(&self, id: &str, machine: &str) -> Result<bool> {
        self.write(|tx| {
            Ok(tx.execute(
                "UPDATE runs SET state = 'sent'
                 WHERE id = ?1 AND state = 'queued' AND machine = ?2 AND cancel_requested = 0",
                params![id, machine],
            )? == 1)
        })
    }

    /// Records `machine`'s first `RunState` of the run.
    pub fn accept(&self, id: &str, machine: &str, state: &v1::RunState) -> Result<Acceptance> {
        self.write(|tx| {
            let row = run_in(tx, id)?.ok_or_else(|| StoreError::Corrupt(format!("no run {id}")))?;
            if row.machine.as_deref() != Some(machine) || row.state == State::Queued {
                return Ok(Acceptance::Stale);
            }
            if row.state != State::Sent {
                return Ok(if row.state == State::Accepted { Acceptance::Again } else { Acceptance::Stale });
            }
            tx.execute(
                "UPDATE runs SET state = 'accepted', machine_number = ?2, machine_state = ?3, accepted_at = ?4
                 WHERE id = ?1",
                params![id, state.number as i64, state.state, now_ms()],
            )?;
            Ok(Acceptance::Fresh(vec![lifecycle(tx, id)?]))
        })
    }

    /// Records a batch of an accepted run's frames and moves its cursor past them.
    pub fn observe(&self, id: &str, batch: Batch) -> Result<Vec<Event>> {
        self.write(|tx| {
            let Some(row) = run_in(tx, id)?.filter(|row| row.state == State::Accepted) else {
                return Ok(vec![]);
            };
            let mut events = vec![];
            for (at_ms, level, text) in batch.logs {
                let seq = next_seq(tx, id)?;
                tx.execute(
                    "INSERT INTO run_logs (run, seq, at_ms, level, text) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![id, seq as i64, at_ms, level, text],
                )?;
                events.push(log_event(seq, at_ms, level, text));
            }
            for product in &batch.products {
                events.extend(put_output(tx, &row, product)?);
            }
            if let Some(state) = batch.machine_state.filter(|s| row.machine_state.as_ref() != Some(s)) {
                tx.execute("UPDATE runs SET machine_state = ?2 WHERE id = ?1", params![id, state])?;
                events.push(lifecycle(tx, id)?);
            }
            tx.execute("UPDATE runs SET cursor = MAX(cursor, ?2) WHERE id = ?1", params![id, batch.cursor as i64])?;
            events.sort_by_key(Event::sequence_number);
            Ok(events)
        })
    }

    /// A line of Calcifer's own in the run's log, whatever its machine has said yet (the inputs it
    /// reused); none once it settled.
    pub fn note(&self, id: &str, level: &str, text: &str) -> Result<Vec<Event>> {
        self.write(|tx| {
            if run_in(tx, id)?.is_none_or(|row| row.state == State::Settled) {
                return Ok(vec![]);
            }
            let (seq, at_ms) = (next_seq(tx, id)?, now_ms());
            tx.execute(
                "INSERT INTO run_logs (run, seq, at_ms, level, text) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![id, seq as i64, at_ms, level, text],
            )?;
            Ok(vec![log_event(seq, at_ms, level.to_owned(), text.to_owned())])
        })
    }

    /// Ends an accepted run with its machine's outcome.
    pub fn settle(&self, id: &str, outcome: &v1::Outcome, cursor: u64) -> Result<Vec<Event>> {
        self.write(|tx| {
            let Some(row) = run_in(tx, id)?.filter(|row| row.state == State::Accepted) else {
                return Ok(vec![]);
            };
            let mut events = vec![];
            for product in &outcome.outputs {
                events.extend(put_output(tx, &row, product)?);
            }
            let status = match outcome.status.as_str() {
                "succeeded" => Status::Completed,
                "canceled" => Status::Canceled,
                _ => Status::Failed,
            };
            let reason = outcome
                .reason
                .as_ref()
                .filter(|_| status != Status::Completed)
                .map(|r| Reason {
                    code: r.code.clone(),
                    message: r.message.clone(),
                });
            tx.execute(
                "UPDATE runs SET cursor = MAX(cursor, ?2), outcome = ?3 WHERE id = ?1",
                params![
                    id,
                    cursor as i64,
                    serde_json::to_vec(outcome).expect("serializes")
                ],
            )?;
            events.extend(end(tx, id, status, reason)?);
            Ok(events)
        })
    }

    /// Ends a run its machine never accepted, with `reason`.
    pub fn fail(&self, id: &str, reason: Reason) -> Result<Vec<Event>> {
        self.write(|tx| match run_in(tx, id)? {
            Some(row) if matches!(row.state, State::Queued | State::Sent) => {
                end(tx, id, Status::Failed, Some(reason))
            }
            _ => Ok(vec![]),
        })
    }

    /// Adds a warning to an unsettled run, answered as its lifecycle event.
    pub fn warn(&self, id: &str, warning: &Warning) -> Result<Vec<Event>> {
        self.write(|tx| {
            let Some(mut row) = run_in(tx, id)?.filter(|row| row.state != State::Settled) else {
                return Ok(vec![]);
            };
            row.warnings.push(warning.clone());
            let warnings = serde_json::to_string(&row.warnings).expect("serializes");
            tx.execute(
                "UPDATE runs SET warnings = ?2 WHERE id = ?1",
                params![id, warnings],
            )?;
            Ok(vec![lifecycle(tx, id)?])
        })
    }

    /// Records the intent to cancel. Unsent work ends canceled here, in the transaction that the
    /// send boundary races; sent work keeps the intent until its machine's outcome arrives.
    pub fn request_cancel(&self, id: &str) -> Result<Vec<Event>> {
        self.write(|tx| {
            let Some(row) = run_in(tx, id)? else {
                return Ok(vec![]);
            };
            match row.state {
                State::Settled => Ok(vec![]),
                _ if row.cancel_requested => Ok(vec![]),
                State::Queued => {
                    tx.execute("UPDATE runs SET cancel_requested = 1 WHERE id = ?1", [id])?;
                    let reason = Reason {
                        code: "canceled".into(),
                        message: "canceled before it was sent".into(),
                    };
                    end(tx, id, Status::Canceled, Some(reason))
                }
                State::Sent | State::Accepted => {
                    tx.execute("UPDATE runs SET cancel_requested = 1 WHERE id = ?1", [id])?;
                    Ok(vec![lifecycle(tx, id)?])
                }
            }
        })
    }

    /// Records `machine` as gone for good and settles what it owed: a run it accepted fails (or
    /// ends canceled when that was asked) and is never run again; one it never confirmed goes back
    /// to the outbox under the same id when Calcifer placed it and the machine had served, and
    /// otherwise fails. Answers the events, by run.
    pub fn machine_ended(&self, machine: &str, reason: &str) -> Result<Vec<(RunId, Event)>> {
        self.write(|tx| {
            let served: Option<bool> = tx
                .query_row("SELECT served FROM machines WHERE name = ?1", [machine], |r| r.get(0))
                .optional()?;
            let Some(served) = served else { return Err(StoreError::UnknownMachine(machine.into())) };
            tx.execute("UPDATE machines SET state = 'ended', reason = ?2 WHERE name = ?1", params![machine, reason])?;
            let mut statement = tx.prepare(&format!(
                "SELECT {RUN_COLUMNS} FROM runs WHERE machine = ?1 AND state <> 'settled' ORDER BY number"
            ))?;
            let rows = statement.query_map([machine], run_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
            let mut events = vec![];
            for row in rows {
                let row = row?;
                let id = row.id.to_string();
                let ended = if row.cancel_requested {
                    let reason = Reason { code: "canceled".into(), message: format!("canceled; {reason}") };
                    end(tx, &id, Status::Canceled, Some(reason))?
                } else if row.state != State::Accepted && !row.pinned && served {
                    tx.execute(
                        "UPDATE runs SET machine = NULL, state = 'queued', cursor = 0, machine_state = NULL
                         WHERE id = ?1",
                        [&id],
                    )?;
                    vec![lifecycle(tx, &id)?]
                } else {
                    let message = match row.state {
                        State::Accepted => format!("its machine {machine} ended mid-run, and the run was not retried: {reason}"),
                        _ => format!("its machine {machine} ended before it accepted the run: {reason}"),
                    };
                    end(tx, &id, Status::Failed, Some(Reason { code: "machine_lost".into(), message }))?
                };
                events.extend(ended.into_iter().map(|event| (row.id.clone(), event)));
            }
            Ok(events)
        })
    }

    /// The run as the API answers it.
    pub fn record(&self, row: &RunRow) -> Result<Run> {
        self.read(|conn| record(conn, row))
    }

    /// The run's durable events after `after`, as a reconnecting stream gets them: its current
    /// lifecycle event, each output item changed since at its newest revision, and its log lines.
    pub fn events_after(&self, id: &str, after: u64) -> Result<Vec<Event>> {
        self.read(|conn| {
            let Some(row) = run_in(conn, id)? else { return Ok(vec![]) };
            let run = record(conn, &row)?;
            let mut events = vec![];
            for (index, item) in run.output.iter().enumerate() {
                let (created, seq): (i64, i64) = conn.query_row(
                    "SELECT created_seq, seq FROM outputs WHERE run = ?1 AND name = ?2 AND idx = ?3",
                    params![id, item.name, item.index],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?;
                let (created, seq, output_index) = (created as u64, seq as u64, index as u32);
                if seq <= after {
                    continue;
                }
                events.push(if item.path.is_some() {
                    Event::OutputLanded { sequence_number: seq, output_index, item: item.clone() }
                } else if item.status != ItemStatus::InProgress {
                    Event::OutputDone { sequence_number: seq, output_index, item: item.clone() }
                } else if created > after {
                    Event::OutputAdded { sequence_number: seq, output_index, item: item.clone() }
                } else {
                    Event::OutputDelta {
                        sequence_number: seq,
                        output_index,
                        item_id: item.id.clone(),
                        rev: item.rev,
                        length: item.length,
                        appended_from: None,
                    }
                });
            }
            let mut statement =
                conn.prepare("SELECT seq, at_ms, level, text FROM run_logs WHERE run = ?1 AND seq > ?2 ORDER BY seq")?;
            let logs = statement.query_map(params![id, after as i64], |r| {
                Ok(log_event(r.get::<_, i64>(0)? as u64, r.get(1)?, r.get(2)?, r.get(3)?))
            })?;
            for log in logs {
                events.push(log?);
            }
            if row.state_seq > after {
                events.push(Event::lifecycle(row.state_seq, run, row.state_seq == 1));
            }
            events.sort_by_key(Event::sequence_number);
            Ok(events)
        })
    }

    /// The runs, by id and number, with an output whose file lacks its newest revision that
    /// Calcifer has not given up.
    pub fn exports_owed(&self) -> Result<Vec<(RunId, u64)>> {
        self.read(|conn| {
            let mut statement = conn.prepare(
                "SELECT DISTINCT runs.id, runs.number FROM outputs JOIN runs ON runs.id = outputs.run
                 WHERE outputs.exported_rev IS NOT outputs.rev AND outputs.failed_rev IS NOT outputs.rev
                   AND outputs.media_type IS NOT ?1
                 ORDER BY runs.number",
            )?;
            let rows = statement
                .query_map([CHECKPOINT], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as u64)))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows.into_iter()
                .map(|(id, number)| Ok((RunId::try_from(id).map_err(|e| StoreError::Corrupt(e.into()))?, number)))
                .collect()
        })
    }

    /// The outputs of runs on `machine` whose file lacks its newest revision and that Calcifer has
    /// not given up, by run and appearance.
    pub fn exports_owed_on(&self, machine: &str) -> Result<Vec<OwedExport>> {
        self.read(|conn| {
            let mut statement = conn.prepare(
                "SELECT runs.id, runs.number, outputs.name, outputs.idx, outputs.length
                 FROM outputs JOIN runs ON runs.id = outputs.run
                 WHERE runs.machine = ?1 AND outputs.exported_rev IS NOT outputs.rev
                   AND outputs.failed_rev IS NOT outputs.rev AND outputs.media_type IS NOT ?2
                 ORDER BY runs.number, outputs.created_seq",
            )?;
            let rows = statement
                .query_map([machine, CHECKPOINT], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)? as u64,
                        r.get::<_, String>(2)?,
                        r.get::<_, i64>(3)? as u32,
                        r.get::<_, i64>(4)? as u64,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows.into_iter()
                .map(|(id, number, name, index, length)| {
                    let run = RunId::try_from(id).map_err(|e| StoreError::Corrupt(e.into()))?;
                    Ok(OwedExport {
                        run,
                        number,
                        name,
                        index,
                        length,
                    })
                })
                .collect()
        })
    }

    /// The run's outputs to export, in the order they first appeared. A checkpoint is never
    /// exported: it stays on its machine (`OutputItem::is_checkpoint`), so it is owed nowhere.
    pub fn outputs_to_export(&self, run: &str) -> Result<Vec<OutputRow>> {
        self.read(|conn| {
            let mut statement = conn.prepare(&format!(
                "SELECT {OUTPUT_COLUMNS} FROM outputs WHERE run = ?1 AND exported_rev IS NOT rev
                 AND failed_rev IS NOT rev AND media_type IS NOT ?2 ORDER BY created_seq"
            ))?;
            Ok(statement
                .query_map([run, CHECKPOINT], output_row)?
                .collect::<rusqlite::Result<_>>()?)
        })
    }

    pub fn output(&self, run: &str, name: &str, index: u32) -> Result<Option<OutputRow>> {
        self.read(|conn| {
            Ok(conn
                .query_row(
                    &format!("SELECT {OUTPUT_COLUMNS} FROM outputs WHERE run = ?1 AND name = ?2 AND idx = ?3"),
                    params![run, name, index],
                    output_row,
                )
                .optional()?)
        })
    }

    /// Records the output's file at `path` as holding revision `rev`; while that is its current
    /// revision, the run's next event says it landed.
    pub fn exported(
        &self,
        run: &str,
        name: &str,
        index: u32,
        rev: u64,
        path: &Path,
    ) -> Result<Vec<Event>> {
        self.write(|tx| {
            tx.execute(
                "UPDATE outputs SET exported_rev = ?4, path = ?5 WHERE run = ?1 AND name = ?2 AND idx = ?3",
                params![run, name, index, rev as i64, path.to_string_lossy()],
            )?;
            export_settled(tx, run, name, index, rev, true)
        })
    }

    /// Records why revision `rev` of the output will not be exported; while that is its current
    /// revision, the run's next event is the item done again, carrying why.
    pub fn export_failed(
        &self,
        run: &str,
        name: &str,
        index: u32,
        rev: u64,
        reason: &str,
    ) -> Result<Vec<Event>> {
        self.write(|tx| {
            tx.execute(
                "UPDATE outputs SET failed_rev = ?4, export_error = ?5 WHERE run = ?1 AND name = ?2 AND idx = ?3",
                params![run, name, index, rev as i64, reason],
            )?;
            export_settled(tx, run, name, index, rev, false)
        })
    }
}

/// The event of an output's export at `rev` settling, if that is its current revision.
fn export_settled(
    tx: &Transaction,
    run: &str,
    name: &str,
    index: u32,
    rev: u64,
    landed: bool,
) -> Result<Vec<Event>> {
    let current: Option<i64> = tx
        .query_row(
            "SELECT rev FROM outputs WHERE run = ?1 AND name = ?2 AND idx = ?3",
            params![run, name, index],
            |r| r.get(0),
        )
        .optional()?;
    if current != Some(rev as i64) {
        return Ok(vec![]);
    }
    let seq = next_seq(tx, run)?;
    tx.execute(
        "UPDATE outputs SET seq = ?4 WHERE run = ?1 AND name = ?2 AND idx = ?3",
        params![run, name, index, seq as i64],
    )?;
    let row = run_in(tx, run)?.ok_or_else(|| StoreError::Corrupt(format!("no run {run}")))?;
    let items = record(tx, &row)?.output.into_iter().enumerate();
    let Some((output_index, item)) = items
        .into_iter()
        .find(|(_, i)| i.name == name && i.index == index)
    else {
        return Ok(vec![]);
    };
    let (sequence_number, output_index) = (seq, output_index as u32);
    Ok(vec![match landed {
        true => Event::OutputLanded {
            sequence_number,
            output_index,
            item,
        },
        false => Event::OutputDone {
            sequence_number,
            output_index,
            item,
        },
    }])
}

const OUTPUT_COLUMNS: &str = "name, idx, rev, length, digest, media_type, done, path, exported_rev,
    CASE WHEN failed_rev = rev THEN export_error END";

fn output_row(r: &Row) -> rusqlite::Result<OutputRow> {
    Ok(OutputRow {
        name: r.get(0)?,
        index: r.get::<_, i64>(1)? as u32,
        rev: r.get::<_, i64>(2)? as u64,
        length: r.get::<_, i64>(3)? as u64,
        digest: r.get(4)?,
        media_type: r.get(5)?,
        done: r.get(6)?,
        path: r.get::<_, Option<String>>(7)?.map(PathBuf::from),
        exported_rev: r.get::<_, Option<i64>>(8)?.map(|rev| rev as u64),
        export_error: r.get(9)?,
    })
}

fn put_machine_in(tx: &Transaction, machine: &MachineRow) -> Result<()> {
    let (worker, address, pin) = match &machine.endpoint {
        Some(e) => (Some(&e.worker_id), Some(&e.address), Some(e.pin.to_vec())),
        None => (None, None, None),
    };
    tx.execute(
        "INSERT INTO machines (name, kind, hub, rental_id, worker_id, address, pin, key, state,
             served, reason, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
         ON CONFLICT (name) DO UPDATE SET kind = ?2, hub = ?3, rental_id = ?4, worker_id = ?5,
             address = ?6, pin = ?7, key = ?8, state = ?9, served = served OR ?10, reason = ?11",
        params![
            machine.name,
            machine_kind(machine.kind),
            machine.hub,
            machine.rental_id,
            worker,
            address,
            pin,
            machine.key.to_string_lossy(),
            machine_state(machine.state),
            machine.served || machine.endpoint.is_some(),
            machine.reason,
            now_ms(),
        ],
    )?;
    Ok(())
}

const RUN_COLUMNS: &str = "id, number, kind, target, hub, machine, pinned, spec, out_dir, state, cancel_requested,
    cursor, machine_state, status, reason_code, reason, state_seq, created_at, accepted_at, settled_at, warnings,
    inputs, outcome, wants";

fn run_row(r: &Row) -> rusqlite::Result<Result<RunRow>> {
    let id: String = r.get(0)?;
    let kind: String = r.get(2)?;
    let hub: String = r.get(4)?;
    let state: String = r.get(9)?;
    let status: Option<String> = r.get(13)?;
    let reason_code: Option<String> = r.get(14)?;
    let reason: Option<String> = r.get(15)?;
    let out_dir: String = r.get(8)?;
    let warnings: String = r.get(20)?;
    let inputs: String = r.get(21)?;
    let spec: Vec<u8> = r.get(7)?;
    let outcome: Option<Vec<u8>> = r.get(22)?;
    let wants: Option<String> = r.get(23)?;
    Ok((|| {
        Ok(RunRow {
            id: RunId::try_from(id).map_err(|_| corrupt("a run id"))?,
            number: r.get::<_, i64>(1)? as u64,
            kind: Kind::parse(&kind).ok_or_else(|| corrupt("a run kind"))?,
            target: r.get(3)?,
            // One spelling, whatever an older Calcifer recorded (`https://tensorhub.com:443`).
            hub: (!hub.is_empty())
                .then(|| Origin::parse(&hub))
                .transpose()
                .map_err(|_| corrupt("a run's Hub"))?,
            machine: r.get(5)?,
            pinned: r.get(6)?,
            spec: serde_json::from_slice(&spec).map_err(|_| corrupt("a run's spec"))?,
            out_dir: PathBuf::from(out_dir),
            state: match state.as_str() {
                "queued" => State::Queued,
                "sent" => State::Sent,
                "accepted" => State::Accepted,
                "settled" => State::Settled,
                _ => return Err(corrupt("a run state")),
            },
            cancel_requested: r.get(10)?,
            cursor: r.get::<_, i64>(11)? as u64,
            machine_state: r.get(12)?,
            status: match status.as_deref() {
                None => None,
                Some("completed") => Some(Status::Completed),
                Some("failed") => Some(Status::Failed),
                Some("canceled") => Some(Status::Canceled),
                Some(_) => return Err(corrupt("a run status")),
            },
            reason: reason_code
                .zip(reason)
                .map(|(code, message)| Reason { code, message }),
            state_seq: r.get::<_, i64>(16)? as u64,
            created_at_ms: r.get(17)?,
            accepted_at_ms: r.get(18)?,
            settled_at_ms: r.get(19)?,
            warnings: serde_json::from_str(&warnings).map_err(|_| corrupt("a run's warnings"))?,
            inputs: serde_json::from_str(&inputs).map_err(|_| corrupt("a run's inputs"))?,
            outcome: match outcome {
                Some(bytes) => {
                    Some(serde_json::from_slice(&bytes).map_err(|_| corrupt("an outcome"))?)
                }
                None => None,
            },
            wants: wants.unwrap_or_else(|| "any".into()),
        })
    })())
}

fn run_in(conn: &Connection, id: &str) -> Result<Option<RunRow>> {
    conn.query_row(
        &format!("SELECT {RUN_COLUMNS} FROM runs WHERE id = ?1"),
        [id],
        run_row,
    )
    .optional()?
    .transpose()
}

const MACHINE_COLUMNS: &str =
    "name, kind, hub, rental_id, worker_id, address, pin, key, state, served, reason";

fn machine_row(r: &Row) -> rusqlite::Result<Result<MachineRow>> {
    let kind: String = r.get(1)?;
    let worker: Option<String> = r.get(4)?;
    let address: Option<String> = r.get(5)?;
    let pin: Option<Vec<u8>> = r.get(6)?;
    let key: String = r.get(7)?;
    let state: String = r.get(8)?;
    let endpoint = match (worker, address, pin.map(<[u8; 32]>::try_from)) {
        (Some(worker_id), Some(address), Some(Ok(pin))) => Some(Endpoint {
            worker_id,
            address,
            pin,
        }),
        (None, None, None) => None,
        _ => return Ok(Err(StoreError::Corrupt("a machine endpoint".into()))),
    };
    Ok(Ok(MachineRow {
        name: r.get(0)?,
        kind: match kind.as_str() {
            "local" => MachineKind::Local,
            "rental" => MachineKind::Rental,
            _ => MachineKind::Added,
        },
        hub: r.get(2)?,
        rental_id: r.get(3)?,
        endpoint,
        key: PathBuf::from(key),
        state: match state.as_str() {
            "booting" => MachineState::Booting,
            "ready" => MachineState::Ready,
            _ => MachineState::Ended,
        },
        served: r.get(9)?,
        reason: r.get(10)?,
    }))
}

/// Moves a machine to another name, with its runs and asks; `alias` is its former name.
fn move_machine(tx: &Transaction, from: &str, to: &str, alias: Option<&str>) -> Result<()> {
    // Its unique rental id and alias move with it.
    let (rental, held): (Option<String>, Option<String>) = tx.query_row(
        "SELECT rental_id, alias FROM machines WHERE name = ?1",
        [from],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    tx.execute(
        "UPDATE machines SET rental_id = NULL, alias = NULL WHERE name = ?1",
        [from],
    )?;
    tx.execute(
        "INSERT INTO machines (name, kind, hub, rental_id, worker_id, address, pin, key, state, served,
             reason, created_at, alias)
         SELECT ?2, kind, hub, ?3, worker_id, address, pin, key, state, served, reason, created_at, ?4
         FROM machines WHERE name = ?1",
        params![from, to, rental, held.as_deref().or(alias)],
    )?;
    tx.execute(
        "UPDATE runs SET machine = ?2 WHERE machine = ?1",
        params![from, to],
    )?;
    tx.execute(
        "UPDATE asks SET machine = ?2 WHERE machine = ?1",
        params![from, to],
    )?;
    tx.execute("DELETE FROM machines WHERE name = ?1", [from])?;
    Ok(())
}

/// What a Hub is asked to do, recorded before it is asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AskKind {
    Rent,
    End,
}

impl AskKind {
    fn as_str(self) -> &'static str {
        match self {
            AskKind::Rent => "rent",
            AskKind::End => "end",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AskState {
    Pending,
    Done,
    Failed,
}

/// One ask of a Hub: its key is the idempotency key the Hub gets, and `body` the exact bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ask {
    pub key: String,
    pub kind: AskKind,
    pub hub: String,
    /// What was asked for, which the same key must ask again.
    pub digest: String,
    pub body: Vec<u8>,
    pub machine: Option<String>,
    /// The run it was made for.
    pub run: Option<RunId>,
    pub state: AskState,
    pub reason: Option<String>,
}

const ASK_COLUMNS: &str = "key, kind, hub, digest, body, machine, run, state, reason";

fn ask_row(r: &Row) -> rusqlite::Result<Result<Ask>> {
    let kind: String = r.get(1)?;
    let run: Option<String> = r.get(6)?;
    let state: String = r.get(7)?;
    Ok((|| {
        Ok(Ask {
            key: r.get(0)?,
            kind: if kind == "end" {
                AskKind::End
            } else {
                AskKind::Rent
            },
            hub: r.get(2)?,
            digest: r.get(3)?,
            body: r.get(4)?,
            machine: r.get(5)?,
            run: run
                .map(RunId::try_from)
                .transpose()
                .map_err(|_| StoreError::Corrupt("an ask's run".into()))?,
            state: match state.as_str() {
                "pending" => AskState::Pending,
                "done" => AskState::Done,
                _ => AskState::Failed,
            },
            reason: r.get(8)?,
        })
    })())
}

fn ask_in(conn: &Connection, key: &str) -> Result<Option<Ask>> {
    conn.query_row(
        &format!("SELECT {ASK_COLUMNS} FROM asks WHERE key = ?1"),
        [key],
        ask_row,
    )
    .optional()?
    .transpose()
}

fn machine_kind(kind: MachineKind) -> &'static str {
    match kind {
        MachineKind::Local => "local",
        MachineKind::Rental => "rental",
        MachineKind::Added => "added",
    }
}

fn machine_state(state: MachineState) -> &'static str {
    match state {
        MachineState::Booting => "booting",
        // Ending is the API's word for a machine whose end its Hub has not affirmed.
        MachineState::Ready | MachineState::Ending => "ready",
        MachineState::Ended => "ended",
    }
}

impl From<&MachineRow> for Machine {
    fn from(row: &MachineRow) -> Self {
        Machine {
            name: row.name.clone(),
            kind: row.kind,
            state: row.state,
            hub: row.hub.clone(),
            rental_id: row.rental_id.clone(),
            alias: None,
            address: row.endpoint.as_ref().map(|e| e.address.clone()),
            fingerprint: row.endpoint.as_ref().map(|e| Digest::from_bytes(e.pin)),
            reason: row.reason.clone(),
            rental: None,
            status: None,
            unreachable: None,
            boot: None,
            running: 0,
            queued: 0,
            recorded: true,
            unverified: false,
        }
    }
}

/// A recorded log line as its event: a settled call is `run.call`, the rest `run.log`.
fn log_event(sequence_number: u64, at_ms: i64, level: String, text: String) -> Event {
    match serde_json::from_str::<Call>(&text) {
        Ok(call) if level == "call" => Event::Call {
            sequence_number,
            call,
        },
        _ => Event::Log {
            sequence_number,
            at_ms,
            level,
            text,
        },
    }
}

/// The run's next event sequence number.
fn next_seq(tx: &Transaction, id: &str) -> Result<u64> {
    Ok(tx.query_row(
        "UPDATE runs SET seq = seq + 1 WHERE id = ?1 RETURNING seq",
        [id],
        |r| r.get::<_, i64>(0),
    )? as u64)
}

/// A new lifecycle event for the run as it now stands.
fn lifecycle(tx: &Transaction, id: &str) -> Result<Event> {
    let seq = next_seq(tx, id)?;
    tx.execute(
        "UPDATE runs SET state_seq = ?2 WHERE id = ?1",
        params![id, seq as i64],
    )?;
    let row = run_in(tx, id)?.ok_or_else(|| StoreError::Corrupt(format!("no run {id}")))?;
    Ok(Event::lifecycle(seq, record(tx, &row)?, false))
}

/// Settles the run: its outputs can no longer change, and its terminal event is the last.
fn end(tx: &Transaction, id: &str, status: Status, reason: Option<Reason>) -> Result<Vec<Event>> {
    let mut events = vec![];
    let open: Vec<(String, i64)> = {
        let mut statement = tx.prepare(
            "SELECT name, idx FROM outputs WHERE run = ?1 AND done = 0 ORDER BY created_seq",
        )?;
        statement
            .query_map([id], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?
    };
    tx.execute(
        "UPDATE runs SET state = 'settled', status = ?2, reason_code = ?3, reason = ?4, settled_at = ?5 WHERE id = ?1",
        params![
            id,
            match status {
                Status::Completed => "completed",
                Status::Canceled => "canceled",
                _ => "failed",
            },
            reason.as_ref().map(|r| &r.code),
            reason.as_ref().map(|r| &r.message),
            now_ms(),
        ],
    )?;
    for (name, idx) in open {
        let seq = next_seq(tx, id)?;
        tx.execute(
            "UPDATE outputs SET done = 1, seq = ?4 WHERE run = ?1 AND name = ?2 AND idx = ?3",
            params![id, name, idx, seq as i64],
        )?;
        let row = run_in(tx, id)?.ok_or_else(|| StoreError::Corrupt(format!("no run {id}")))?;
        let run = record(tx, &row)?;
        if let Some((output_index, item)) = run
            .output
            .into_iter()
            .enumerate()
            .find(|(_, item)| item.name == name && i64::from(item.index) == idx)
        {
            events.push(Event::OutputDone {
                sequence_number: seq,
                output_index: output_index as u32,
                item,
            });
        }
    }
    events.push(lifecycle(tx, id)?);
    Ok(events)
}

/// Records one revision of an output, unless a newer one is recorded.
fn put_output(tx: &Transaction, row: &RunRow, product: &v1::Product) -> Result<Vec<Event>> {
    let id = row.id.as_str();
    let index = i64::from(product.index);
    let recorded: Option<(i64, bool)> = tx
        .query_row(
            "SELECT rev, done FROM outputs WHERE run = ?1 AND name = ?2 AND idx = ?3",
            params![id, product.output, index],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    if recorded.is_some_and(|(rev, done)| done || rev as u64 >= product.rev) {
        return Ok(vec![]);
    }
    let seq = next_seq(tx, id)?;
    tx.execute(
        "INSERT INTO outputs (run, name, idx, rev, length, digest, media_type, label, duration_us, created_seq, seq)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)
         ON CONFLICT (run, name, idx) DO UPDATE SET rev = ?4, length = ?5, digest = ?6, media_type = ?7,
             label = ?8, duration_us = ?9, seq = ?10",
        params![
            id,
            product.output,
            index,
            product.rev as i64,
            product.length as i64,
            product.digest,
            product.media_type,
            product.label,
            product.duration_us as i64,
            seq as i64,
        ],
    )?;
    let run = record(tx, row)?;
    let Some((output_index, item)) = run
        .output
        .into_iter()
        .enumerate()
        .find(|(_, item)| item.name == product.output && item.index == product.index)
    else {
        return Ok(vec![]);
    };
    let output_index = output_index as u32;
    Ok(vec![match recorded {
        None => Event::OutputAdded {
            sequence_number: seq,
            output_index,
            item,
        },
        Some(_) => Event::OutputDelta {
            sequence_number: seq,
            output_index,
            item_id: item.id,
            rev: product.rev,
            length: product.length,
            appended_from: product.appended_from,
        },
    }])
}

/// A run row as the API answers it, with its output items in the order they first appeared.
fn record(conn: &Connection, row: &RunRow) -> Result<Run> {
    let mut statement = conn.prepare(
        "SELECT name, idx, rev, length, digest, media_type, label, done,
             CASE WHEN exported_rev = rev THEN path END, CASE WHEN failed_rev = rev THEN export_error END
         FROM outputs WHERE run = ?1 ORDER BY created_seq",
    )?;
    let settled_ok = row.status == Some(Status::Completed);
    let items = statement.query_map([row.id.as_str()], |r| {
        let name: String = r.get(0)?;
        let index = r.get::<_, i64>(1)? as u32;
        let digest: String = r.get(4)?;
        let done: bool = r.get(7)?;
        let path: Option<String> = r.get(8)?;
        let id = match index {
            0 => format!("{}/{name}", row.number),
            i => format!("{}/{name}/{i}", row.number),
        };
        let status = match (done, settled_ok) {
            (false, _) => ItemStatus::InProgress,
            (true, true) => ItemStatus::Completed,
            (true, false) => ItemStatus::Incomplete,
        };
        Ok(OutputItem {
            id,
            name,
            index,
            rev: r.get::<_, i64>(2)? as u64,
            length: r.get::<_, i64>(3)? as u64,
            media_type: r.get(5)?,
            label: r.get(6)?,
            sha256: (status == ItemStatus::Completed)
                .then(|| digest.parse().ok())
                .flatten(),
            status,
            path: path.map(PathBuf::from),
            export_error: r.get(9)?,
        })
    })?;
    Ok(Run {
        id: row.id.clone(),
        number: row.number,
        kind: row.kind,
        target: row.target.clone(),
        hub: row.hub.clone().map(String::from).unwrap_or_default(),
        machine: row.machine.clone(),
        status: match (row.state, row.status) {
            (State::Settled, Some(status)) => status,
            (State::Accepted, _) => Status::InProgress,
            _ => Status::Queued,
        },
        cancel_requested: row.cancel_requested,
        machine_state: row.machine_state.clone(),
        reason: row.reason.clone(),
        output: items.collect::<rusqlite::Result<_>>()?,
        created_at_ms: row.created_at_ms,
        accepted_at_ms: row.accepted_at_ms,
        settled_at_ms: row.settled_at_ms,
        waiting: None,
        progress: None,
        warnings: row.warnings.clone(),
        adjustments: row
            .outcome
            .iter()
            .flat_map(|outcome| outcome.adjustments.iter().cloned())
            .collect(),
        result: row
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.result.as_ref())
            .and_then(|result| serde_json::from_str(result.get()).ok()),
        execution_ms: row
            .outcome
            .as_ref()
            .map(|outcome| outcome.execution_ms)
            .filter(|ms| *ms > 0),
        measurements: None,
        calls: vec![],
    })
}

fn corrupt(what: &str) -> StoreError {
    StoreError::Corrupt(what.into())
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}
