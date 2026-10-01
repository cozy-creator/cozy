//! SQLite is a rebuildable projection and operation coordinator. Repository/manifests/blobs
//! remain the authority; SQL is never consulted to decide GC liveness.

use std::collections::HashMap;
use std::fs::{self, File};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

use fs2::FileExt;
use rusqlite::{params, Connection};

use crate::err::{refuse, Code, Refusal, Result};
use crate::storage::Projection;

mod lock;

/// The catalog this build writes. A newer build may only ADD tables and columns (an
/// incompatible change renames the table), so any catalog at this version or later that
/// holds every table and column below opens.
const SCHEMA_VERSION: u64 = 2;

const SCHEMA: &[(&str, &[&str])] = &[
    ("tensorfs_derived_transactions", &["id", "bytes"]),
    ("tensorfs_holds", &["operation_id", "key", "length"]),
    (
        "tensorfs_operations",
        &["id", "org", "name", "writer_marker"],
    ),
    (
        "tensorfs_released_manifests",
        &["org", "name", "manifest_sha256"],
    ),
    (
        "tensorfs_source_preparations",
        &["operation_id", "request_digest", "state"],
    ),
    (
        "tensorfs_source_results",
        &[
            "operation_id",
            "slot",
            "profile",
            "manifest_sha256",
            "manifest_length",
        ],
    ),
    (
        "tensorfs_verified_blobs",
        &["sha256", "length", "dev", "ino", "mtime_s", "mtime_ns"],
    ),
];

fn sql(what: impl AsRef<str>, error: rusqlite::Error) -> Refusal {
    crate::store::classify_sql(what, error)
}

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    crate::store::classify_io(what, error)
}

/// Every table and column this build uses is present. Additions by a newer build are its own.
fn verify_schema(connection: &Connection) -> Result<()> {
    for (table, expected_columns) in SCHEMA {
        let mut columns = connection
            .prepare("SELECT name FROM pragma_table_info(?1)")
            .map_err(|error| sql("prepare catalog column census", error))?;
        let actual: Vec<String> = columns
            .query_map([table], |row| row.get(0))
            .map_err(|error| sql("read catalog column census", error))?
            .collect::<std::result::Result<_, _>>()
            .map_err(|error| sql("collect catalog column census", error))?;
        if let Some(missing) = expected_columns
            .iter()
            .find(|column| !actual.iter().any(|name| name == *column))
        {
            return refuse(
                Code::UNKNOWN_FORMAT,
                format!(
                    "tensorfs.sqlite table {table} has no column {missing}; remove the catalog and rebuild from repository files"
                ),
            );
        }
    }
    Ok(())
}

fn opened() -> &'static Mutex<HashMap<PathBuf, u64>> {
    static OPENED: OnceLock<Mutex<HashMap<PathBuf, u64>>> = OnceLock::new();
    OPENED.get_or_init(Default::default)
}

/// Connections this process has opened to the catalog under `root`. The trust path's
/// contract is one per catalog file per process; this makes it observable.
pub fn connections_opened(root: &Path) -> u64 {
    opened()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&Catalog::path(root))
        .copied()
        .unwrap_or(0)
}

pub(crate) fn connect(path: &Path) -> Result<Connection> {
    // SQLite creates WAL/SHM files before assigning their database owner. Opening
    // them under root exposes a transient root-owned inode to concurrent owner
    // connections, which can permanently fall back to read-only WAL access.
    let _identity = crate::filesystem_identity::FilesystemIdentity::for_store_root(
        path.parent().unwrap_or(path),
    )
    .map_err(|error| io("enter catalog filesystem identity", error))?;
    *opened()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .entry(path.to_path_buf())
        .or_default() += 1;
    crate::stats::catalog_opened();
    let connection = Connection::open(path).map_err(|error| {
        // SQLite reports a full descriptor table only as "unable to open"; the table
        // itself answers whether that is what happened.
        if crate::descriptors::available().is_some_and(|free| free == 0) {
            crate::descriptors::exhausted(&std::io::Error::from_raw_os_error(
                rustix::io::Errno::MFILE.raw_os_error(),
            ));
            return Refusal {
                code: Code::FD_HEADROOM,
                detail: format!("open catalog: no file descriptor ({error})"),
            };
        }
        sql("open catalog", error)
    })?;
    configure(&connection, path)?;
    Ok(connection)
}

/// Journal and sync by what a crash can take from the disk the catalog lives on
/// (`disk::DiskClass`): an ephemeral disk dies with the machine, so no sync buys anything;
/// a persistent one keeps a WAL synced at checkpoints; a network filesystem keeps the
/// rollback journal and a full sync, because WAL's shared memory is not coherent across
/// hosts. The catalog is only ever opened by processes on one host otherwise.
pub(crate) fn configure(connection: &Connection, path: &Path) -> Result<()> {
    lock::wait_for_live_holders(connection, path)?;
    let (journal, synchronous) = match crate::disk::DiskClass::of(path.parent().unwrap_or(path)) {
        crate::disk::DiskClass::Ephemeral => ("wal", "OFF"),
        crate::disk::DiskClass::Persistent => ("wal", "NORMAL"),
        crate::disk::DiskClass::Network => ("delete", "FULL"),
    };
    connection
        .query_row(&format!("PRAGMA journal_mode={journal}"), [], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| sql("set catalog journal", error))?;
    connection
        .execute_batch(&format!(
            "PRAGMA foreign_keys=ON; PRAGMA synchronous={synchronous};"
        ))
        .map_err(|error| sql("configure catalog", error))
}

/// A connection to an existing catalog of exactly this schema.
fn validated(path: &Path) -> Result<Connection> {
    if !path.is_file() {
        return refuse(
            Code::DURABILITY_UNPROVEN,
            format!(
                "{} is absent; store is recovery-only until `tfs store rebuild`",
                path.display()
            ),
        );
    }
    let connection = connect(path)?;
    let version: u64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|error| sql("read catalog version", error))?;
    if version < SCHEMA_VERSION {
        return refuse(
            Code::UNKNOWN_FORMAT,
            format!(
                "tensorfs.sqlite user_version {version} predates {SCHEMA_VERSION}; remove it and rebuild from repository files"
            ),
        );
    }
    verify_schema(&connection)?;
    Ok(connection)
}

/// THE PROCESS'S TRUST INDEX: every `tensorfs_verified_blobs` row of one catalog, loaded
/// with one query the first time any handle asks, then answered from memory.
///
/// A row is a cache fact bound to `(dev, ino, length, mtime)`; the caller still compares it
/// with the object's `fstat`. A remembered row therefore stays sound after another process
/// drops it: unchanged bytes under the same binding are the bytes that were hashed, and
/// anything that changed them fails the comparison and pays its rehash. A miss asks the one
/// kept connection, so rows written by other processes are seen; a replaced catalog file
/// (rebuild) is reopened and reloaded. In-process writes and drops update the index directly.
pub(crate) struct Trust {
    path: PathBuf,
    rows: RwLock<HashMap<String, crate::store::VRecord>>,
    kept: Mutex<Option<Kept>>,
}

struct Kept {
    identity: (u64, u64),
    connection: Connection,
}

fn trust_index() -> &'static Mutex<HashMap<PathBuf, Weak<Trust>>> {
    static INDEX: OnceLock<Mutex<HashMap<PathBuf, Weak<Trust>>>> = OnceLock::new();
    INDEX.get_or_init(Default::default)
}

impl Trust {
    /// The one index for the catalog under `root`, shared by every live handle in the
    /// process. It, and its kept connection, end with the last handle.
    pub(crate) fn of(root: &Path) -> Arc<Trust> {
        let path = Catalog::path(root);
        let mut index = trust_index()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(trust) = index.get(&path).and_then(Weak::upgrade) {
            return trust;
        }
        index.retain(|_, trust| trust.strong_count() > 0);
        let trust = Arc::new(Trust {
            path: path.clone(),
            rows: RwLock::default(),
            kept: Mutex::default(),
        });
        index.insert(path, Arc::downgrade(&trust));
        trust
    }

    fn live(path: &Path) -> Option<Arc<Trust>> {
        trust_index()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(path)
            .and_then(Weak::upgrade)
    }

    pub(crate) fn get(&self, sha256: &str) -> Option<crate::store::VRecord> {
        if let Some(row) = self
            .rows
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .get(sha256)
        {
            return Some(row.clone());
        }
        self.miss(sha256)
    }

    /// Validate the catalog on the kept connection: the handle's one open, which also loads
    /// the index. A live connection to the same catalog file is already validated.
    pub(crate) fn validate(&self) -> Result<()> {
        let mut kept = self.kept.lock().unwrap_or_else(|error| error.into_inner());
        self.load(&mut kept).map(|_| ())
    }

    /// Finish setup on the already-validated reader connection. The callback is
    /// synchronous under its mutex; no second setup connection or credential
    /// lifetime escapes into the returned value.
    pub(crate) fn with_connection<T>(
        &self,
        use_connection: impl FnOnce(&mut Connection) -> Result<T>,
    ) -> Result<T> {
        let mut kept = self.kept.lock().unwrap_or_else(|error| error.into_inner());
        self.load(&mut kept)?;
        let connection = &mut kept.as_mut().expect("validated connection").connection;
        // A kept connection can be used by another Runtime thread later. Install
        // the lock context on the thread actually executing this operation.
        lock::wait_for_live_holders(connection, &self.path)?;
        let _identity = crate::filesystem_identity::FilesystemIdentity::for_store_root(
            self.path.parent().unwrap_or(&self.path),
        )
        .map_err(|error| io("enter catalog filesystem identity", error))?;
        use_connection(connection)
    }

    fn miss(&self, sha256: &str) -> Option<crate::store::VRecord> {
        let mut kept = self.kept.lock().unwrap_or_else(|error| error.into_inner());
        if self.load(&mut kept).ok()? {
            return self
                .rows
                .read()
                .unwrap_or_else(|error| error.into_inner())
                .get(sha256)
                .cloned();
        }
        let open = kept.as_ref()?;
        let row = verification_on(&open.connection, sha256).ok().flatten()?;
        self.remember(&row);
        Some(row)
    }

    /// Keep a validated connection to the current catalog file, loading every row when it is
    /// newly opened. Returns whether it was (re)opened.
    fn load(&self, kept: &mut Option<Kept>) -> Result<bool> {
        let identity = fs::symlink_metadata(&self.path)
            .map(|metadata| (metadata.dev(), metadata.ino()))
            .ok();
        if kept
            .as_ref()
            .is_some_and(|open| Some(open.identity) == identity)
        {
            return Ok(false);
        }
        let connection = validated(&self.path)?;
        let Some(identity) = identity else {
            return refuse(
                Code::DURABILITY_UNPROVEN,
                format!("{} changed while it was opened", self.path.display()),
            );
        };
        let rows = verifications_on(&connection)?;
        *kept = Some(Kept {
            identity,
            connection,
        });
        self.rows
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .extend(rows.into_iter().map(|row| (row.sha256.clone(), row)));
        Ok(true)
    }

    fn remember(&self, row: &crate::store::VRecord) {
        self.rows
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .insert(row.sha256.clone(), row.clone());
    }

    fn forget(&self, sha256: &str) {
        self.rows
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .remove(sha256);
    }
}

#[derive(Debug, Clone)]
pub struct Catalog {
    path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePreparationResult {
    pub slot: String,
    pub profile: String,
    pub manifest_sha256: String,
    pub manifest_length: u64,
}

pub enum SourcePreparation {
    Open(WriterGuard),
    Prepared(Vec<SourcePreparationResult>),
}

pub(crate) fn source_operation_id(operation: &str) -> Result<()> {
    crate::ids::ascii_name(
        "model-source operation",
        operation,
        crate::limits::MAX_NAME_BYTES,
    )?;
    if operation.contains('/') || matches!(operation, "." | "..") {
        return refuse(
            Code::KEY_GRAMMAR,
            "model-source operation is one ordinary path segment",
        );
    }
    Ok(())
}

const SOURCE_OPERATION_SELECT: &str = "SELECT p.operation_id,p.request_digest,p.state,o.org,o.name
     FROM tensorfs_source_preparations p
     LEFT JOIN tensorfs_operations o ON o.id=p.operation_id";

struct SourceOperationRow {
    operation: String,
    digest: String,
    state: String,
    org: Option<String>,
    name: Option<String>,
}

impl SourceOperationRow {
    fn read(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            operation: row.get(0)?,
            digest: row.get(1)?,
            state: row.get(2)?,
            org: row.get(3)?,
            name: row.get(4)?,
        })
    }

    fn validated(self) -> Result<String> {
        source_operation_id(&self.operation)?;
        crate::ids::prefixed("source preparation request", &self.digest)?;
        if !matches!(self.state.as_str(), "open" | "prepared")
            || self.org.as_deref() != Some("_tensorfs")
            || self.name.as_deref() != Some("model-source-preparation")
        {
            return refuse(
                Code::DURABILITY_UNPROVEN,
                "source preparation has missing or conflicting operation identity/state",
            );
        }
        Ok(self.operation)
    }
}

impl Catalog {
    pub fn path(root: &Path) -> PathBuf {
        root.join("tensorfs.sqlite")
    }

    pub fn initialize(root: &Path) -> Result<Self> {
        let catalog = Self {
            path: Self::path(root),
        };
        catalog.initialize_schema()?;
        Ok(catalog)
    }

    pub fn open(root: &Path) -> Result<Self> {
        let path = Self::path(root);
        validated(&path)?;
        Ok(Self { path })
    }

    fn connection(&self) -> Result<Connection> {
        connect(&self.path)
    }

    fn initialize_schema(&self) -> Result<()> {
        let connection = self.connection()?;
        connection
            .execute_batch("BEGIN IMMEDIATE")
            .map_err(|error| sql("lock catalog initialization", error))?;
        let version: u64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(|error| sql("read catalog version", error))?;
        if version >= SCHEMA_VERSION {
            connection
                .execute_batch("COMMIT")
                .map_err(|error| sql("commit catalog open", error))?;
            return verify_schema(&connection);
        }
        if version != 0 {
            let _ = connection.execute_batch("ROLLBACK");
            return refuse(
                Code::UNKNOWN_FORMAT,
                format!(
                    "tensorfs.sqlite user_version {version} predates {SCHEMA_VERSION}; remove it and rebuild from repository files"
                ),
            );
        }
        let existing: u64 = connection
            .query_row(
                "SELECT count(*) FROM sqlite_master
                 WHERE type='table' AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|error| sql("inspect unversioned catalog", error))?;
        if existing != 0 {
            let _ = connection.execute_batch("ROLLBACK");
            return refuse(
                Code::UNKNOWN_FORMAT,
                "unversioned tensorfs.sqlite is not an empty new catalog; remove it and rebuild from repository files",
            );
        }
        let result = connection.execute_batch(
            "
                 CREATE TABLE tensorfs_released_manifests (
                   org TEXT NOT NULL,
                   name TEXT NOT NULL,
                   manifest_sha256 TEXT NOT NULL,
                   PRIMARY KEY (org,name,manifest_sha256)
                 ) STRICT;
                 CREATE TABLE tensorfs_operations (
                   id TEXT PRIMARY KEY,
                   org TEXT NOT NULL,
                   name TEXT NOT NULL,
                   writer_marker TEXT NOT NULL
                 ) STRICT;
                 CREATE TABLE tensorfs_holds (
                   operation_id TEXT NOT NULL,
                   key TEXT NOT NULL,
                   length INTEGER NOT NULL,
                   PRIMARY KEY (operation_id,key),
                   FOREIGN KEY (operation_id) REFERENCES tensorfs_operations(id) ON DELETE CASCADE
                 ) STRICT;
                 CREATE TABLE tensorfs_derived_transactions (
                   id TEXT PRIMARY KEY,
                   bytes BLOB NOT NULL
                 ) STRICT;
                 CREATE TABLE tensorfs_source_preparations (
                   operation_id TEXT PRIMARY KEY REFERENCES tensorfs_operations(id) ON DELETE CASCADE,
                   request_digest TEXT NOT NULL,
                   state TEXT NOT NULL CHECK (state IN ('open','prepared'))
                 ) STRICT;
                 CREATE TABLE tensorfs_source_results (
                   operation_id TEXT NOT NULL REFERENCES tensorfs_source_preparations(operation_id) ON DELETE CASCADE,
                   slot TEXT NOT NULL,
                   profile TEXT NOT NULL,
                   manifest_sha256 TEXT NOT NULL,
                   manifest_length INTEGER NOT NULL,
                   PRIMARY KEY (operation_id,slot)
                 ) STRICT;
                 CREATE TABLE tensorfs_verified_blobs (
                   sha256 TEXT PRIMARY KEY,
                   length INTEGER NOT NULL,
                   dev TEXT NOT NULL,
                   ino TEXT NOT NULL,
                   mtime_s INTEGER NOT NULL,
                   mtime_ns INTEGER NOT NULL
                 ) STRICT;
                 PRAGMA user_version=2;
                 COMMIT;",
        );
        match result {
            Ok(()) => verify_schema(&connection),
            Err(error) => {
                let _ = connection.execute_batch("ROLLBACK");
                Err(sql("create catalog", error))
            }
        }
    }

    pub fn replace_all(&self, projection: &Projection) -> Result<()> {
        let mut connection = self.connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| sql("begin projection rebuild", error))?;
        transaction
            .execute_batch("DELETE FROM tensorfs_released_manifests;")
            .map_err(|error| sql("clear projection", error))?;
        insert_projection(&transaction, projection)?;
        transaction
            .commit()
            .map_err(|error| sql("commit projection rebuild", error))
    }

    pub fn replace_repo(&self, org: &str, name: &str, projection: &Projection) -> Result<()> {
        let mut connection = self.connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| sql("begin repository projection", error))?;
        transaction
            .execute(
                "DELETE FROM tensorfs_released_manifests WHERE org=?1 AND name=?2",
                params![org, name],
            )
            .map_err(|error| sql("delete old repository projection", error))?;
        insert_projection(&transaction, projection)?;
        transaction
            .commit()
            .map_err(|error| sql("commit repository projection", error))
    }

    pub fn begin_operation(&self, id: &str, org: &str, name: &str) -> Result<WriterGuard> {
        let root = self.path.parent().ok_or_else(|| Refusal {
            code: Code::IO_FAILED,
            detail: "catalog has no store parent".into(),
        })?;
        let guard = WriterGuard::acquire(root)?;
        self.connection()?
            .execute(
                "INSERT INTO tensorfs_operations
                 (id,org,name,writer_marker) VALUES (?1,?2,?3,?4)",
                params![id, org, name, guard.path.to_string_lossy()],
            )
            .map_err(|error| sql("begin operation", error))?;
        Ok(guard)
    }

    /// Claim an existing open operation for this process, or recreate its coordination row
    /// after an explicit catalog rebuild. The returned advisory-lock guard must live for the
    /// whole resumed writer/install process.
    pub fn resume_operation(&self, id: &str, org: &str, name: &str) -> Result<WriterGuard> {
        self.resume_operation_inner(id, org, name, true)
    }

    /// Borrow an existing operation without recreating a released predecessor.
    pub(crate) fn resume_retained_operation(
        &self,
        id: &str,
        org: &str,
        name: &str,
    ) -> Result<WriterGuard> {
        self.resume_operation_inner(id, org, name, false)
    }

    fn resume_operation_inner(
        &self,
        id: &str,
        org: &str,
        name: &str,
        recreate: bool,
    ) -> Result<WriterGuard> {
        use rusqlite::OptionalExtension;

        let root = self.path.parent().ok_or_else(|| Refusal {
            code: Code::IO_FAILED,
            detail: "catalog has no store parent".into(),
        })?;
        let guard = WriterGuard::acquire(root)?;
        let mut connection = self.connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| sql("begin operation resume", error))?;
        let existing: Option<(String, String, String)> = transaction
            .query_row(
                "SELECT writer_marker,org,name FROM tensorfs_operations WHERE id=?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .map_err(|error| sql("read operation resume", error))?;
        match existing {
            None if !recreate => {
                return refuse(
                    Code::ROOT_ABSENT,
                    "predecessor operation is no longer retained",
                )
            }
            None => {
                transaction
                    .execute(
                        "INSERT INTO tensorfs_operations
                         (id,org,name,writer_marker) VALUES (?1,?2,?3,?4)",
                        params![id, org, name, guard.path.to_string_lossy()],
                    )
                    .map_err(|error| sql("recreate operation after rebuild", error))?;
            }
            Some((marker, existing_org, existing_name)) => {
                if existing_org != org || existing_name != name {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        "same operation id was opened for a different repository",
                    );
                }
                let marker = PathBuf::from(marker);
                if marker_is_live(&marker)? {
                    return refuse(Code::STORE_BUSY, "operation already has a live writer");
                }
                if marker != guard.path {
                    let _ = fs::remove_file(marker);
                }
                transaction
                    .execute(
                        "UPDATE tensorfs_operations
                         SET writer_marker=?2
                         WHERE id=?1",
                        params![id, guard.path.to_string_lossy()],
                    )
                    .map_err(|error| sql("claim resumed operation", error))?;
            }
        }
        transaction
            .commit()
            .map_err(|error| sql("commit operation resume", error))?;
        Ok(guard)
    }

    /// Open or replay one model-source preparation. Its semantic request is represented by
    /// one digest only; typed callers retain the fields and re-derive that digest on retry.
    /// URL/capability facts therefore have no column to leak into.
    pub fn begin_source_preparation(
        &self,
        operation: &str,
        request_digest: &str,
    ) -> Result<SourcePreparation> {
        self.begin_source_preparation_inner(operation, request_digest, true)
    }

    pub(crate) fn begin_retained_source_preparation(
        &self,
        operation: &str,
        request_digest: &str,
    ) -> Result<SourcePreparation> {
        self.begin_source_preparation_inner(operation, request_digest, false)
    }

    fn begin_source_preparation_inner(
        &self,
        operation: &str,
        request_digest: &str,
        recreate: bool,
    ) -> Result<SourcePreparation> {
        source_operation_id(operation)?;
        let request_digest = crate::ids::prefixed("source preparation request", request_digest)?;
        let guard = self.resume_operation_inner(
            operation,
            "_tensorfs",
            "model-source-preparation",
            recreate,
        )?;
        let connection = self.connection()?;
        let existing = connection.query_row(
            "SELECT request_digest,state FROM tensorfs_source_preparations WHERE operation_id=?1",
            [operation],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        );
        let state = match existing {
            Ok((digest, state)) => {
                if digest != request_digest {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        "same model-source operation id binds a different request",
                    );
                }
                state
            }
            Err(rusqlite::Error::QueryReturnedNoRows) if !recreate => {
                return refuse(
                    Code::ROOT_ABSENT,
                    "source preparation is no longer retained",
                )
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                connection
                    .execute(
                        "INSERT INTO tensorfs_source_preparations(operation_id,request_digest,state)
                         VALUES (?1,?2,'open')",
                        params![operation, request_digest],
                    )
                    .map_err(|error| sql("record source preparation", error))?;
                "open".to_string()
            }
            Err(error) => return Err(sql("read source preparation", error)),
        };
        if state == "prepared" {
            drop(guard);
            Ok(SourcePreparation::Prepared(
                self.source_preparation_results(operation)?,
            ))
        } else {
            Ok(SourcePreparation::Open(guard))
        }
    }

    fn source_preparation_results(&self, operation: &str) -> Result<Vec<SourcePreparationResult>> {
        let connection = self.connection()?;
        let mut statement = connection
            .prepare(
                "SELECT slot,profile,manifest_sha256,manifest_length
                 FROM tensorfs_source_results WHERE operation_id=?1 ORDER BY slot",
            )
            .map_err(|error| sql("prepare source results", error))?;
        let rows = statement
            .query_map([operation], |row| {
                Ok(SourcePreparationResult {
                    slot: row.get(0)?,
                    profile: row.get(1)?,
                    manifest_sha256: row.get(2)?,
                    manifest_length: row.get(3)?,
                })
            })
            .map_err(|error| sql("read source results", error))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| sql("collect source results", error))?;
        Ok(rows)
    }

    /// Registered source operations needing explicit lifecycle release. This is the
    /// existing operation journal, not GC liveness or a filesystem census. Ordinary
    /// process replacement and projection rebuild preserve it; after catalog loss,
    /// replaying the exact source request recreates this coordination as before.
    pub fn model_source_operations(&self) -> Result<Vec<String>> {
        let connection = self.connection()?;
        let mut statement = connection
            .prepare(&format!(
                "{SOURCE_OPERATION_SELECT} ORDER BY p.operation_id"
            ))
            .map_err(|error| sql("prepare source operation roster", error))?;
        let rows = statement
            .query_map([], SourceOperationRow::read)
            .map_err(|error| sql("read source operation roster", error))?;
        let mut operations = Vec::new();
        for row in rows {
            operations.push(
                row.map_err(|error| sql("read source operation", error))?
                    .validated()?,
            );
        }
        Ok(operations)
    }

    /// Publish all prepared refs to replay readers in one SQLite commit. CAS bytes and exact
    /// holds may precede this transaction; no partial row set can become visible.
    pub fn commit_source_preparation(
        &self,
        operation: &str,
        request_digest: &str,
        results: &[SourcePreparationResult],
    ) -> Result<()> {
        if results.is_empty() {
            return refuse(
                Code::MISSING_FIELD,
                "source preparation produced no profiles",
            );
        }
        if results.windows(2).any(|pair| pair[0].slot >= pair[1].slot) {
            return refuse(
                Code::SORT_ORDER,
                "source preparation results must be sorted uniquely by slot",
            );
        }
        for result in results {
            crate::ids::ascii_name(
                "source preparation slot",
                &result.slot,
                crate::limits::MAX_NAME_BYTES,
            )?;
            crate::ids::ascii_name(
                "source preparation profile",
                &result.profile,
                crate::limits::MAX_KEY_BYTES,
            )?;
            crate::ids::hex64("source preparation manifest", &result.manifest_sha256)?;
            if result.manifest_length == 0 {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    "source preparation Manifest has zero length",
                );
            }
        }
        let request_digest = crate::ids::prefixed("source preparation request", request_digest)?;
        let mut connection = self.connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| sql("begin source preparation commit", error))?;
        let (stored, state): (String, String) = transaction
            .query_row(
                "SELECT request_digest,state FROM tensorfs_source_preparations WHERE operation_id=?1",
                [operation],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|error| sql("read source preparation before commit", error))?;
        if stored != request_digest || state != "open" {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "source preparation changed before result commit",
            );
        }
        for result in results {
            transaction
                .execute(
                    "INSERT INTO tensorfs_source_results
                     (operation_id,slot,profile,manifest_sha256,manifest_length)
                     VALUES (?1,?2,?3,?4,?5)",
                    params![
                        operation,
                        result.slot,
                        result.profile,
                        result.manifest_sha256,
                        result.manifest_length,
                    ],
                )
                .map_err(|error| sql("insert source result", error))?;
        }
        transaction
            .execute(
                "UPDATE tensorfs_source_preparations SET state='prepared' WHERE operation_id=?1",
                [operation],
            )
            .map_err(|error| sql("mark source preparation complete", error))?;
        transaction
            .commit()
            .map_err(|error| sql("commit source preparation", error))
    }

    /// The caller quiesces source lifecycle changes before enumerating/releasing.
    /// Remove its filesystem journal while the existing writer guard excludes GC and
    /// preparation, then delete the roster row. Failed cleanup remains discoverable.
    /// An absent source entry is a no-op and never authorizes filesystem cleanup.
    pub fn release_source_preparation<F>(&self, operation: &str, cleanup: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        use rusqlite::OptionalExtension;
        source_operation_id(operation)?;
        let found = self
            .connection()?
            .query_row(
                &format!("{SOURCE_OPERATION_SELECT} WHERE p.operation_id=?1"),
                [operation],
                SourceOperationRow::read,
            )
            .optional()
            .map_err(|error| sql("find source operation for release", error))?;
        if found
            .map(SourceOperationRow::validated)
            .transpose()?
            .is_none()
        {
            return Ok(());
        }
        let _guard = self.resume_operation(operation, "_tensorfs", "model-source-preparation")?;
        cleanup()?;
        self.complete_operation(operation)
    }

    /// Hold SQLite's short writer transaction across the filesystem repository replacement.
    /// If projection insertion fails after rename, storage remains ahead of SQL and rebuild
    /// repairs it; SQL is never allowed to lead the file.
    pub fn commit_repository<F>(&self, org: &str, name: &str, write: F) -> Result<()>
    where
        F: FnOnce() -> Result<Projection>,
    {
        let connection = self.connection()?;
        connection
            .execute_batch("BEGIN IMMEDIATE")
            .map_err(|error| sql("begin repository commit", error))?;
        let result = (|| {
            let projection = write()?;
            connection
                .execute(
                    "DELETE FROM tensorfs_released_manifests WHERE org=?1 AND name=?2",
                    params![org, name],
                )
                .map_err(|error| sql("delete old repository projection", error))?;
            insert_projection(&connection, &projection)?;
            connection
                .execute_batch("COMMIT")
                .map_err(|error| sql("commit repository projection", error))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = connection.execute_batch("ROLLBACK");
        }
        result
    }

    /// Register an exact key against an operation.
    ///
    /// **A HOLD IS NOT GC PROTECTION, and never was.** `gc::collect` computes liveness from
    /// `repos/`, `manifests/` and `blobs/` plus the FILESYSTEM hold protocol — open ingest
    /// session roots and the conversion journals inside them — and `tensorfs.sqlite` is
    /// never consulted (gc.rs, owner ruling 2026-09-02). `held_keys()` below has no
    /// production caller; it exists for tests and for `tfs gc plan --holds`, which is fed a
    /// file, not this table. What these rows are FOR is the operation's own bookkeeping:
    /// `complete_operation`/`abandon` cascade them away, so an operation can state exactly
    /// which keys it put down without walking the store again.
    ///
    /// A writer that needs its in-flight output to survive a `tfs gc` pass must name those
    /// objects on the filesystem — `transaction::name_candidates`, or the conversion journal
    /// that `ingest::journal` appends to as each op lands. Writing a row here and expecting
    /// the collector to honour it is the mistake this note exists to prevent.
    pub fn hold(&self, operation: &str, key: &str, length: u64) -> Result<()> {
        hold_on(&self.connection()?, operation, key, length)
    }

    /// A HELD admission: the hold is written before the object is published, so an
    /// operation that cannot hold it publishes nothing. Unheld admissions, which is every
    /// pull, take `record_grouped` instead.
    pub(crate) fn admit_blob(
        &self,
        operation: Option<&str>,
        object: &crate::ids::ObjectRef,
        publish: impl FnOnce() -> Result<Option<crate::store::VRecord>>,
    ) -> Result<bool> {
        // One admission at a time per catalog in this process, holding the write lock from
        // before publication: streams landing objects together queue here instead of in
        // SQLite's busy timeout, where a lost race refused bytes already verified.
        let turn = held_admissions(&self.path);
        let _turn = turn.lock().unwrap_or_else(|error| error.into_inner());
        let mut connection = self.connection()?;
        let transaction = connection
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| sql("begin blob admission", error))?;
        if let Some(operation) = operation {
            hold_on(
                &transaction,
                operation,
                &crate::storage::blob_key(&object.sha256)?,
                object.length,
            )?;
        }
        let verified = publish()?;
        if let Some(row) = &verified {
            write_verification_on(&transaction, row)?;
        }
        transaction
            .commit()
            .map_err(|error| sql("commit blob admission", error))?;
        if let (Some(row), Some(trust)) = (&verified, Trust::live(&self.path)) {
            trust.remember(row);
        }
        Ok(verified.is_some())
    }

    /// The object's existing hold and verification become durable together. Filesystem
    /// publication has already happened, on the caller's thread, and precedes this record,
    /// which is written with FULL synchronization.
    ///
    /// Records queue per catalog in this process, and whichever arrives while no batch is
    /// committing leads the next one: one immediate transaction and one commit make every
    /// waiting record durable. One object alone costs what it always did. Before this,
    /// directory creation, link, directory fsync and commit ran for one object at a time
    /// under one lock, several fsyncs each, and on a disk busy flushing those same objects'
    /// bodies that queue, not the network, set a pull's rate.
    pub(crate) fn record_grouped(&self, recording: Recording) -> Result<()> {
        let group = admission_group(&self.path);
        let mut state = group
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let ticket = state.next;
        state.next += 1;
        state.queue.push((ticket, recording));
        loop {
            if let Some(result) = state.done.remove(&ticket) {
                return result;
            }
            if state.leading {
                state = group
                    .turned
                    .wait(state)
                    .unwrap_or_else(|error| error.into_inner());
                continue;
            }
            state.leading = true;
            let batch = std::mem::take(&mut state.queue);
            let mut lead = Lead {
                group: &group,
                tickets: batch.iter().map(|(ticket, _)| *ticket).collect(),
                results: Vec::new(),
            };
            drop(state);
            let started = std::time::Instant::now();
            let objects = batch.len();
            lead.results = self.record_batch(batch);
            crate::stats::admission_batch(objects, started.elapsed());
            drop(lead);
            state = group
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    fn record_batch(&self, batch: Vec<(u64, Recording)>) -> Vec<(u64, Result<()>)> {
        let fail_all = |batch: Vec<(u64, Recording)>, error: Refusal| {
            batch
                .into_iter()
                .map(|(ticket, _)| (ticket, Err(error.clone())))
                .collect::<Vec<_>>()
        };
        let mut connection = match self.connection() {
            Ok(connection) => connection,
            Err(error) => return fail_all(batch, error),
        };
        let mut transaction =
            match connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate) {
                Ok(transaction) => transaction,
                Err(error) => return fail_all(batch, sql("begin blob admission", error)),
            };
        let mut results = Vec::with_capacity(batch.len());
        let mut written = Vec::with_capacity(batch.len());
        // Each object's row sits behind its own savepoint: one refused row refuses that
        // object, as it did when every object had a transaction of its own.
        for (ticket, recording) in batch {
            let outcome = transaction
                .savepoint()
                .map_err(|error| sql("begin object admission", error))
                .and_then(|savepoint| {
                    if let Some(row) = &recording.record {
                        write_verification_on(&savepoint, row)?;
                    }
                    savepoint
                        .commit()
                        .map_err(|error| sql("record object admission", error))
                });
            match outcome {
                Ok(()) => written.push((ticket, recording.record)),
                Err(error) => results.push((ticket, Err(error))),
            }
        }
        match transaction
            .commit()
            .map_err(|error| sql("commit blob admission", error))
        {
            Ok(()) => {
                let trust = Trust::live(&self.path);
                for (ticket, record) in written {
                    if let (Some(row), Some(trust)) = (&record, &trust) {
                        trust.remember(row);
                    }
                    results.push((ticket, Ok(())));
                }
            }
            Err(error) => {
                results.extend(
                    written
                        .into_iter()
                        .map(|(ticket, _)| (ticket, Err(error.clone()))),
                );
            }
        }
        results
    }

    pub fn held_keys(&self) -> Result<Vec<crate::storage::HeldKey>> {
        let connection = self.connection()?;
        let mut statement = connection
            .prepare(
                "SELECT h.key,h.length FROM tensorfs_holds h
                 JOIN tensorfs_operations o ON o.id=h.operation_id
                 ORDER BY h.key",
            )
            .map_err(|error| sql("prepare hold census", error))?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?))
            })
            .map_err(|error| sql("read hold census", error))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| sql("collect hold census", error))?;
        let mut result: Vec<crate::storage::HeldKey> = Vec::new();
        for (key, length) in rows {
            if let Some(existing) = result.last() {
                if existing.key == key {
                    if existing.length != length {
                        return refuse(Code::LENGTH_MISMATCH, "holds disagree on key length");
                    }
                    continue;
                }
            }
            let kind = if key.starts_with("blobs/") {
                "blob"
            } else if key.starts_with("manifests/") {
                "manifest"
            } else {
                return refuse(Code::KEY_GRAMMAR, "stored hold has a foreign key");
            };
            let line = crate::canon::write(&crate::canon::Value::obj(vec![
                ("key", crate::canon::Value::str(key)),
                ("kind", crate::canon::Value::str(kind)),
                ("length", crate::canon::Value::uint(length)),
            ]));
            result.push(crate::storage::HeldKey::parse_line(&line)?);
        }
        Ok(result)
    }

    /// Explicit abandonment after every writer grant is revoked and the writer is gone.
    pub fn abandon(&self, operation: &str, grants_revoked: bool) -> Result<()> {
        use rusqlite::OptionalExtension;
        if !grants_revoked {
            return refuse(Code::STORE_BUSY, "abandon requires revoked grants");
        }
        let mut connection = self.connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| sql("begin abandon", error))?;
        // A session root with no operation row (the catalog was rebuilt under exclusivity,
        // so no writer of it was live) has nothing left to arbitrate against.
        let writer_marker: Option<String> = transaction
            .query_row(
                "SELECT writer_marker FROM tensorfs_operations WHERE id=?1",
                [operation],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| sql("read abandon candidate", error))?;
        if let Some(writer_marker) = writer_marker {
            if marker_is_live(Path::new(&writer_marker))? {
                return refuse(Code::STORE_BUSY, "operation writer is still live");
            }
        }
        transaction
            .execute("DELETE FROM tensorfs_operations WHERE id=?1", [operation])
            .map_err(|error| sql("delete abandoned operation and holds", error))?;
        transaction
            .commit()
            .map_err(|error| sql("commit abandonment", error))
    }

    pub fn complete_operation(&self, operation: &str) -> Result<()> {
        let mut connection = self.connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| sql("begin operation completion", error))?;
        let changed = transaction
            .execute("DELETE FROM tensorfs_operations WHERE id=?1", [operation])
            .map_err(|error| sql("delete completed operation and holds", error))?;
        if changed != 1 {
            return refuse(Code::TRANSACTION_CLOSED, "operation is not open");
        }
        transaction
            .commit()
            .map_err(|error| sql("commit operation completion", error))
    }

    pub(crate) fn write_verification(&self, row: &crate::store::VRecord) -> Result<()> {
        write_verification_on(&self.connection()?, row)?;
        if let Some(trust) = Trust::live(&self.path) {
            trust.remember(row);
        }
        Ok(())
    }

    pub(crate) fn drop_verification(&self, sha256: &str) {
        if let Some(trust) = Trust::live(&self.path) {
            trust.forget(sha256);
        }
        if let Ok(connection) = self.connection() {
            let _ = connection.execute(
                "DELETE FROM tensorfs_verified_blobs WHERE sha256=?1",
                [sha256],
            );
        }
    }

    /// Drop every verification row whose object `present` no longer finds: a row dies with
    /// its object, after it, and a pass interrupted between the two converges here.
    pub(crate) fn prune_verifications(&self, present: impl Fn(&str) -> bool) -> Result<u64> {
        let connection = self.connection()?;
        let mut statement = connection
            .prepare("SELECT sha256 FROM tensorfs_verified_blobs")
            .map_err(|error| sql("prepare verification census", error))?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| sql("read verification census", error))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| sql("collect verification census", error))?;
        let mut pruned = 0;
        for sha256 in rows {
            if !present(&sha256) {
                connection
                    .execute(
                        "DELETE FROM tensorfs_verified_blobs WHERE sha256=?1",
                        [&sha256],
                    )
                    .map_err(|error| sql("prune verification row", error))?;
                if let Some(trust) = Trust::live(&self.path) {
                    trust.forget(&sha256);
                }
                pruned += 1;
            }
        }
        Ok(pruned)
    }
}

fn held_admissions(path: &Path) -> Arc<Mutex<()>> {
    static TURNS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    TURNS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .entry(path.to_path_buf())
        .or_default()
        .clone()
}

/// One published object's catalog rows, handed to the admission group.
pub(crate) struct Recording {
    /// The verification row, when this call installed the object.
    pub record: Option<crate::store::VRecord>,
}

#[derive(Default)]
struct AdmissionGroup {
    state: Mutex<GroupState>,
    turned: std::sync::Condvar,
}

#[derive(Default)]
struct GroupState {
    next: u64,
    leading: bool,
    queue: Vec<(u64, Recording)>,
    done: HashMap<u64, Result<()>>,
}

fn admission_group(path: &Path) -> Arc<AdmissionGroup> {
    static GROUPS: OnceLock<Mutex<HashMap<PathBuf, Arc<AdmissionGroup>>>> = OnceLock::new();
    GROUPS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .entry(path.to_path_buf())
        .or_default()
        .clone()
}

/// A leader hands its batch's results back however it leaves, so a panic inside one batch
/// cannot strand the admissions queued behind it.
struct Lead<'a> {
    group: &'a AdmissionGroup,
    tickets: Vec<u64>,
    results: Vec<(u64, Result<()>)>,
}

impl Drop for Lead<'_> {
    fn drop(&mut self) {
        let mut state = self
            .group
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.leading = false;
        let mut answered = std::collections::HashSet::new();
        for (ticket, result) in self.results.drain(..) {
            answered.insert(ticket);
            state.done.insert(ticket, result);
        }
        for ticket in &self.tickets {
            if !answered.contains(ticket) {
                state.done.insert(
                    *ticket,
                    Err(Refusal {
                        code: Code::IO_FAILED,
                        detail: "the admission batch holding this object did not finish".into(),
                    }),
                );
            }
        }
        self.group.turned.notify_all();
    }
}

fn hold_on(connection: &Connection, operation: &str, key: &str, length: u64) -> Result<()> {
    let kind = if key.starts_with("blobs/") {
        "blob"
    } else if key.starts_with("manifests/") {
        "manifest"
    } else {
        return refuse(Code::KEY_GRAMMAR, "hold key must be a blob or manifest key");
    };
    let line = crate::canon::write(&crate::canon::Value::obj(vec![
        ("key", crate::canon::Value::str(key)),
        ("kind", crate::canon::Value::str(kind)),
        ("length", crate::canon::Value::uint(length)),
    ]));
    crate::storage::HeldKey::parse_line(&line)?;
    let changed = connection
        .execute(
            "INSERT INTO tensorfs_holds(operation_id,key,length) VALUES (?1,?2,?3)
                 ON CONFLICT(operation_id,key) DO NOTHING",
            params![operation, key, length],
        )
        .map_err(|error| sql("register exact hold", error))?;
    if changed == 1 {
        return Ok(());
    }
    let existing: u64 = connection
        .query_row(
            "SELECT length FROM tensorfs_holds WHERE operation_id=?1 AND key=?2",
            params![operation, key],
            |row| row.get(0),
        )
        .map_err(|error| sql("read existing hold", error))?;
    if existing == length {
        Ok(())
    } else {
        refuse(
            Code::LENGTH_MISMATCH,
            "the operation already holds this key at another length",
        )
    }
}

type VerificationRow = (String, u64, String, String, u64, u64);

fn verification_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<VerificationRow> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
    ))
}

fn vrecord(
    (sha256, length, dev, ino, mtime_s, mtime_ns): VerificationRow,
) -> Result<crate::store::VRecord> {
    let invalid = |what: &str| Refusal {
        code: Code::IO_FAILED,
        detail: format!("invalid verification {what}"),
    };
    Ok(crate::store::VRecord {
        sha256,
        length,
        dev: dev.parse().map_err(|_| invalid("dev"))?,
        ino: ino.parse().map_err(|_| invalid("ino"))?,
        mtime_s,
        mtime_ns,
    })
}

fn verification_on(connection: &Connection, sha256: &str) -> Result<Option<crate::store::VRecord>> {
    use rusqlite::OptionalExtension;
    connection
        .prepare_cached(
            "SELECT sha256,length,dev,ino,mtime_s,mtime_ns
             FROM tensorfs_verified_blobs WHERE sha256=?1",
        )
        .and_then(|mut statement| statement.query_row([sha256], verification_row).optional())
        .map_err(|error| sql("read blob verification", error))?
        .map(vrecord)
        .transpose()
}

fn verifications_on(connection: &Connection) -> Result<Vec<crate::store::VRecord>> {
    let mut statement = connection
        .prepare("SELECT sha256,length,dev,ino,mtime_s,mtime_ns FROM tensorfs_verified_blobs")
        .map_err(|error| sql("prepare verification index", error))?;
    let rows = statement
        .query_map([], verification_row)
        .map_err(|error| sql("read verification index", error))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| sql("collect verification index", error))?;
    rows.into_iter().map(vrecord).collect()
}

fn write_verification_on(connection: &Connection, row: &crate::store::VRecord) -> Result<()> {
    connection
        .execute(
            "INSERT INTO tensorfs_verified_blobs
                 (sha256,length,dev,ino,mtime_s,mtime_ns)
                 VALUES (?1,?2,?3,?4,?5,?6)
                 ON CONFLICT(sha256) DO UPDATE SET
                   length=excluded.length,dev=excluded.dev,ino=excluded.ino,
                   mtime_s=excluded.mtime_s,mtime_ns=excluded.mtime_ns",
            params![
                row.sha256,
                row.length,
                row.dev.to_string(),
                row.ino.to_string(),
                row.mtime_s,
                row.mtime_ns,
            ],
        )
        .map_err(|error| sql("write blob verification", error))?;
    Ok(())
}

fn insert_projection(transaction: &Connection, projection: &Projection) -> Result<()> {
    for row in projection.checkpoints.iter().filter(|row| row.published) {
        transaction
            .execute(
                "INSERT INTO tensorfs_released_manifests(org,name,manifest_sha256)
                 VALUES (?1,?2,?3) ON CONFLICT DO NOTHING",
                params![row.org, row.name, row.manifest_sha256],
            )
            .map_err(|error| sql("insert released manifest", error))?;
    }
    Ok(())
}

fn marker_is_live(path: &Path) -> Result<bool> {
    let marker = match File::options().read(true).write(true).open(path) {
        Ok(marker) => marker,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(io("open operation marker", error)),
    };
    Ok(marker.try_lock_exclusive().is_err())
}

/// Process-liveness marker used while the database may be absent. The retained advisory
/// file lock is the proof; rebuild removes a marker only after it can acquire that lock.
#[derive(Debug)]
pub struct WriterGuard {
    path: PathBuf,
    _file: File,
    _recovery: File,
}

/// `tmp/writers` inside a root that already exists.
///
/// A writer marker is never the thing that brings a root into being: pointed at an absent path
/// this used to lay down a directory tree wherever the caller happened to be (tfs-053).
fn writer_markers(root: &Path) -> Result<PathBuf> {
    match fs::symlink_metadata(root) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
        Ok(_) => {
            return refuse(
                Code::STORE_ERA,
                format!("{} is not a real Store directory", root.display()),
            )
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return refuse(
                Code::STORE_ROOT_ABSENT,
                format!(
                    "{} does not exist; create the root before writing into it",
                    root.display()
                ),
            )
        }
        Err(error) => return Err(io(format!("stat root {}", root.display()), error)),
    }
    let directory = root.join("tmp/writers");
    fs::create_dir_all(&directory).map_err(|error| io("mkdir writer markers", error))?;
    Ok(directory)
}

impl WriterGuard {
    pub fn acquire(root: &Path) -> Result<Self> {
        let directory = writer_markers(root)?;
        let recovery = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.join("recovery.lock"))
            .map_err(|error| io("open recovery lock", error))?;
        FileExt::lock_shared(&recovery).map_err(|error| io("share recovery lock", error))?;
        let path = directory.join(format!(
            "{}-{}.writer",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ));
        let file = File::options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|error| io("create writer marker", error))?;
        file.try_lock_exclusive()
            .map_err(|error| io("lock writer marker", error))?;
        Ok(Self {
            path,
            _file: file,
            _recovery: recovery,
        })
    }

    pub fn lock_rebuild(root: &Path) -> Result<RebuildGuard> {
        let directory = writer_markers(root)?;
        let recovery = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.join("recovery.lock"))
            .map_err(|error| io("open recovery lock", error))?;
        if recovery.try_lock_exclusive().is_err() {
            return refuse(
                Code::STORE_BUSY,
                format!("the store is held by {}", live_holders(root)?),
            );
        }
        for entry in fs::read_dir(&directory).map_err(|error| io("read writer markers", error))? {
            let path = entry
                .map_err(|error| io("read writer marker", error))?
                .path();
            if path.extension().and_then(|value| value.to_str()) != Some("writer") {
                continue;
            }
            let marker = File::options()
                .read(true)
                .write(true)
                .open(&path)
                .map_err(|error| io("open writer marker", error))?;
            if marker.try_lock_exclusive().is_err() {
                return refuse(
                    Code::STORE_BUSY,
                    format!("the store is held by {}", live_holders(root)?),
                );
            }
            fs::remove_file(path).map_err(|error| io("reap dead writer marker", error))?;
        }
        Ok(RebuildGuard { _file: recovery })
    }
}

/// Who holds the store right now, by marker: `writer <pid>` for an open operation
/// (`tmp/writers/<pid>-<nanos>.writer`) and `<kind> <pid>` for a lease
/// (`tmp/leases/<kind>-<pid>-<nanos>.lease`). Liveness is the kernel lock on the marker;
/// a marker whose lock can be taken belongs to nobody and is not named.
fn live_holders(root: &Path) -> Result<String> {
    let mut holders = Vec::new();
    for (directory, extension) in [("tmp/writers", "writer"), ("tmp/leases", "lease")] {
        let Ok(entries) = fs::read_dir(root.join(directory)) else {
            continue;
        };
        for entry in entries {
            let path = entry.map_err(|error| io("read marker", error))?.path();
            if path.extension().and_then(|value| value.to_str()) != Some(extension) {
                continue;
            }
            let Ok(marker) = File::options().read(true).write(true).open(&path) else {
                continue;
            };
            if marker.try_lock_exclusive().is_ok() {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            let mut parts = stem.rsplitn(2, '-');
            let _nanos = parts.next();
            let head = parts.next().unwrap_or(stem);
            holders.push(match extension {
                "writer" => format!("writer pid {head}"),
                _ => head.rsplit_once('-').map_or_else(
                    || head.to_string(),
                    |(kind, pid)| format!("{kind} lease pid {pid}"),
                ),
            });
        }
    }
    holders.sort();
    holders.dedup();
    Ok(if holders.is_empty() {
        "another process (recovery lock shared)".to_string()
    } else {
        holders.join(", ")
    })
}

/// Exclusive against every writer from census through catalog replacement.
#[derive(Debug)]
pub struct RebuildGuard {
    _file: File,
}

impl Drop for WriterGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_database_is_a_capacity_verdict_not_an_io_failure() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(
            "PRAGMA page_size = 512; CREATE TABLE t (x BLOB); PRAGMA max_page_count = 3;",
        )
        .unwrap();
        let full = db
            .execute("INSERT INTO t VALUES (zeroblob(65536))", [])
            .unwrap_err();
        assert_eq!(sql("record", full).code, Code::CAPACITY_EXHAUSTED);
        let other = db
            .execute("INSERT INTO missing VALUES (1)", [])
            .unwrap_err();
        assert_eq!(sql("record", other).code, Code::IO_FAILED);
    }

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-catalog-{name}-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ))
    }

    /// Another process holding the catalog's write lock: this test binary, re-run as
    /// `lock_holder_child`, until it has taken the lock.
    fn holder(catalog: &Path, millis: u64) -> std::process::Child {
        use std::io::BufRead;
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "catalog::tests::lock_holder_child", "--ignored"])
            .args(["--nocapture", "--test-threads=1"])
            .env("TENSORFS_TEST_HOLD", catalog)
            .env("TENSORFS_TEST_HOLD_MS", millis.to_string())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
        let held = lines
            .by_ref()
            .map_while(std::result::Result::ok)
            .any(|line| line.ends_with("held"));
        assert!(held, "the holder never took the catalog lock");
        std::thread::spawn(move || lines.for_each(drop));
        child
    }

    #[test]
    #[ignore = "run by holder() in a child process"]
    fn lock_holder_child() {
        let path = std::env::var("TENSORFS_TEST_HOLD").unwrap();
        let millis: u64 = std::env::var("TENSORFS_TEST_HOLD_MS")
            .unwrap()
            .parse()
            .unwrap();
        let connection = Connection::open(path).unwrap();
        connection.execute_batch("BEGIN IMMEDIATE").unwrap();
        connection
            .execute(
                "INSERT INTO tensorfs_released_manifests VALUES ('held','by',?1)",
                [std::process::id().to_string()],
            )
            .unwrap();
        println!("held");
        std::thread::sleep(std::time::Duration::from_millis(millis));
        connection.execute_batch("COMMIT").unwrap();
    }

    fn write(catalog: &Catalog, name: &str) -> Result<usize> {
        catalog
            .connection()?
            .execute(
                "INSERT INTO tensorfs_released_manifests VALUES ('org',?1,'digest')",
                [name],
            )
            .map_err(|error| sql("test write", error))
    }

    #[test]
    fn a_live_lock_holder_is_waited_for_past_any_fixed_deadline() {
        let root = temporary("live-holder");
        fs::create_dir_all(&root).unwrap();
        let catalog = Catalog::initialize(&root).unwrap();
        let mut child = holder(&catalog.path, 7_000);
        let started = std::time::Instant::now();
        assert_eq!(write(&catalog, "after-holder").unwrap(), 1);
        assert!(started.elapsed() >= std::time::Duration::from_secs(6));
        assert!(child.wait().unwrap().success());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_stopped_lock_holder_is_refused() {
        use rustix::process::{kill_process, Pid, Signal};
        let root = temporary("stopped-holder");
        fs::create_dir_all(&root).unwrap();
        let catalog = Catalog::initialize(&root).unwrap();
        let mut child = holder(&catalog.path, 60_000);
        let pid = Pid::from_raw(child.id() as i32).unwrap();
        kill_process(pid, Signal::STOP).unwrap();
        let refused = write(&catalog, "behind-a-stopped-holder").unwrap_err();
        assert!(refused.detail.contains("locked"), "{refused:?}");
        kill_process(pid, Signal::KILL).unwrap();
        child.wait().unwrap();
        assert_eq!(write(&catalog, "after-the-holder-died").unwrap(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn catalog_has_seven_tables_and_opens_additive_future_versions() {
        let root = temporary("schema");
        fs::create_dir_all(&root).unwrap();
        let catalog = Catalog::initialize(&root).unwrap();
        let connection = catalog.connection().unwrap();
        let tables: Vec<String> = {
            let mut statement = connection
                .prepare(
                    "SELECT name FROM sqlite_master
                     WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
                )
                .unwrap();
            statement
                .query_map([], |row| row.get(0))
                .unwrap()
                .collect::<std::result::Result<_, _>>()
                .unwrap()
        };
        assert_eq!(
            tables,
            vec![
                "tensorfs_derived_transactions",
                "tensorfs_holds",
                "tensorfs_operations",
                "tensorfs_released_manifests",
                "tensorfs_source_preparations",
                "tensorfs_source_results",
                "tensorfs_verified_blobs",
            ]
        );
        // A newer build added a table and a column: this build still opens and uses it.
        connection
            .execute_batch(
                "CREATE TABLE tensorfs_future(value TEXT);
                 ALTER TABLE tensorfs_holds ADD COLUMN note TEXT;
                 PRAGMA user_version=3;",
            )
            .unwrap();
        drop(connection);
        let reopened = Catalog::initialize(&root).unwrap();
        let guard = reopened.begin_operation("op", "org", "repo").unwrap();
        reopened
            .hold(
                "op",
                &crate::storage::blob_key(&"ab".repeat(32)).unwrap(),
                7,
            )
            .unwrap();
        drop(guard);
        drop(reopened);
        Catalog::open(&root).unwrap();
        let connection = rusqlite::Connection::open(Catalog::path(&root)).unwrap();
        let version: u64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 3);
        // A catalog without a column this build uses is genuinely unreadable.
        connection
            .execute_batch("ALTER TABLE tensorfs_verified_blobs DROP COLUMN mtime_ns")
            .unwrap();
        drop(connection);
        assert_eq!(
            Catalog::initialize(&root).unwrap_err().code,
            Code::UNKNOWN_FORMAT
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn old_or_wrong_v1_catalogs_refuse_without_mutation() {
        let root = temporary("v1-hardcut");
        fs::create_dir_all(&root).unwrap();
        let connection = rusqlite::Connection::open(Catalog::path(&root)).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE legacy(value TEXT);
                 INSERT INTO legacy VALUES ('keep-me');
                 PRAGMA user_version=1;",
            )
            .unwrap();
        drop(connection);
        assert_eq!(
            Catalog::initialize(&root).unwrap_err().code,
            Code::UNKNOWN_FORMAT
        );
        let connection = rusqlite::Connection::open(Catalog::path(&root)).unwrap();
        let value: String = connection
            .query_row("SELECT value FROM legacy", [], |row| row.get(0))
            .unwrap();
        assert_eq!(value, "keep-me");
        drop(connection);

        fs::remove_file(Catalog::path(&root)).unwrap();
        let connection = rusqlite::Connection::open(Catalog::path(&root)).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE legacy(value TEXT); INSERT INTO legacy VALUES ('still-here');",
            )
            .unwrap();
        drop(connection);
        assert_eq!(
            Catalog::initialize(&root).unwrap_err().code,
            Code::UNKNOWN_FORMAT
        );
        let connection = rusqlite::Connection::open(Catalog::path(&root)).unwrap();
        let value: String = connection
            .query_row("SELECT value FROM legacy", [], |row| row.get(0))
            .unwrap();
        assert_eq!(value, "still-here");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rebuild_refuses_live_writer_and_reaps_dead_marker() {
        let root = temporary("writer");
        fs::create_dir_all(&root).unwrap();
        let guard = WriterGuard::acquire(&root).unwrap();
        assert_eq!(
            WriterGuard::lock_rebuild(&root).unwrap_err().code,
            Code::STORE_BUSY
        );
        drop(guard);
        let directory = root.join("tmp/writers");
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("dead.writer"), b"").unwrap();
        let rebuild = WriterGuard::lock_rebuild(&root).unwrap();
        assert_eq!(
            fs::read_dir(directory)
                .unwrap()
                .filter(|entry| entry
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .and_then(|x| x.to_str())
                    == Some("writer"))
                .count(),
            0
        );
        drop(rebuild);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn inherited_recovery_descriptor_keeps_gc_fenced_until_child_exit() {
        use std::process::{Command, Stdio};
        let root = temporary("inherited-recovery");
        crate::store::Store::init(&root).unwrap();
        let writer = WriterGuard::acquire(&root).unwrap();
        // An actual child retains the same open-file description. Concurrent
        // spawn/exec can briefly inherit such descriptors even with CLOEXEC.
        let mut child = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::from(writer._recovery.try_clone().unwrap()))
            .spawn()
            .unwrap();
        drop(writer);
        let refused = WriterGuard::lock_rebuild(&root);
        drop(child.stdin.take());
        assert!(child.wait().unwrap().success());
        assert_eq!(refused.unwrap_err().code, Code::STORE_BUSY);
        WriterGuard::lock_rebuild(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gc_recovery_lock_refuses_live_read_or_derived_hold() {
        let root = temporary("hold-fence");
        let store = crate::store::Store::init(&root).unwrap();
        let meta = crate::meta::Meta::open(&store).unwrap();
        let hold = meta.acquire_hold("derived-writer").unwrap();
        assert_eq!(
            WriterGuard::lock_rebuild(&root).unwrap_err().code,
            Code::STORE_BUSY
        );
        hold.release(&meta).unwrap();
        WriterGuard::lock_rebuild(&root).unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn exact_holds_and_explicit_abandonment_are_closed() {
        let root = temporary("operation");
        fs::create_dir_all(&root).unwrap();
        let catalog = Catalog::initialize(&root).unwrap();
        let guard = catalog.begin_operation("op", "org", "repo").unwrap();
        let digest = "11".repeat(32);
        catalog
            .hold("op", &crate::storage::blob_key(&digest).unwrap(), 7)
            .unwrap();
        catalog
            .hold("op", &crate::storage::blob_key(&digest).unwrap(), 7)
            .unwrap();
        assert_eq!(
            catalog
                .hold("op", "public/blobs/not-this-domain", 7)
                .unwrap_err()
                .code,
            Code::KEY_GRAMMAR
        );
        assert_eq!(
            catalog.abandon("op", true).unwrap_err().code,
            Code::STORE_BUSY
        );
        drop(guard);
        catalog.abandon("op", true).unwrap();
        let connection = catalog.connection().unwrap();
        let holds: u64 = connection
            .query_row("SELECT count(*) FROM tensorfs_holds", [], |row| row.get(0))
            .unwrap();
        assert_eq!(holds, 0);
        let operations: u64 = connection
            .query_row("SELECT count(*) FROM tensorfs_operations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(
            operations, 0,
            "closed operations are not retained as tombstones"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resumed_operation_keeps_live_identity_and_recreates_coordination_after_db_loss() {
        let root = temporary("operation-resume");
        fs::create_dir_all(&root).unwrap();
        let catalog = Catalog::initialize(&root).unwrap();
        let first = catalog.begin_operation("op", "org", "repo").unwrap();
        assert_eq!(
            catalog
                .resume_operation("op", "org", "repo")
                .unwrap_err()
                .code,
            Code::STORE_BUSY
        );
        drop(first);
        assert_eq!(
            catalog
                .resume_operation("op", "org", "different")
                .unwrap_err()
                .code,
            Code::TRANSACTION_CONFLICT
        );
        let resumed = catalog.resume_operation("op", "org", "repo").unwrap();
        assert_eq!(
            catalog
                .resume_operation("op", "org", "repo")
                .unwrap_err()
                .code,
            Code::STORE_BUSY
        );
        drop(resumed);

        fs::remove_file(Catalog::path(&root)).unwrap();
        let rebuilt = Catalog::initialize(&root).unwrap();
        let recovered = rebuilt.resume_operation("op", "org", "repo").unwrap();
        rebuilt
            .hold(
                "op",
                &crate::storage::blob_key(&"44".repeat(32)).unwrap(),
                9,
            )
            .unwrap();
        assert_eq!(rebuilt.held_keys().unwrap().len(), 1);
        drop(recovered);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn source_operation_roster_is_typed_read_only_and_survives_process_reopen() {
        let root = temporary("source-roster");
        fs::create_dir_all(&root).unwrap();
        let catalog = Catalog::initialize(&root).unwrap();
        let digest = format!("sha256:{}", "31".repeat(32));
        let source = catalog
            .begin_source_preparation("z-prepared", &digest)
            .unwrap();
        catalog
            .commit_source_preparation(
                "z-prepared",
                &digest,
                &[SourcePreparationResult {
                    slot: "model".into(),
                    profile: "proof".into(),
                    manifest_sha256: "32".repeat(32),
                    manifest_length: 161,
                }],
            )
            .unwrap();
        drop(source);
        let incomplete = catalog
            .begin_source_preparation("a-incomplete", &digest)
            .unwrap();
        // Open work is discoverable even before any filesystem session root exists.
        assert!(!root.join("tmp/ingest").exists());
        let generic = catalog
            .begin_operation("generic-ingest", "_tensorfs", "ingest")
            .unwrap();
        let derived = catalog
            .begin_operation("derived-operation", "_tensorfs", "derived")
            .unwrap();
        let lookalike = catalog
            .begin_operation(
                "unregistered-source",
                "_tensorfs",
                "model-source-preparation",
            )
            .unwrap();
        let before = fs::read(Catalog::path(&root)).unwrap();
        let reopened = Catalog::open(&root).unwrap();
        assert_eq!(
            reopened.model_source_operations().unwrap(),
            ["a-incomplete", "z-prepared"]
        );
        assert_eq!(fs::read(Catalog::path(&root)).unwrap(), before);
        reopened.replace_all(&Projection::default()).unwrap();
        assert_eq!(
            reopened.model_source_operations().unwrap(),
            ["a-incomplete", "z-prepared"]
        );
        assert_eq!(
            reopened
                .release_source_preparation("a-incomplete", || panic!("live writer cleaned"))
                .unwrap_err()
                .code,
            Code::STORE_BUSY
        );
        drop(incomplete);
        drop(generic);
        drop(derived);
        drop(lookalike);
        for name in [
            "generic-ingest",
            "derived-operation",
            "unregistered-source",
            "absent",
        ] {
            reopened
                .release_source_preparation(name, || panic!("non-source cleanup invoked"))
                .unwrap();
        }
        let failed = reopened.release_source_preparation("a-incomplete", || {
            refuse(Code::IO_FAILED, "injected cleanup refusal")
        });
        assert_eq!(failed.unwrap_err().code, Code::IO_FAILED);
        assert_eq!(
            reopened.model_source_operations().unwrap(),
            ["a-incomplete", "z-prepared"]
        );
        reopened
            .release_source_preparation("a-incomplete", || Ok(()))
            .unwrap();
        reopened
            .release_source_preparation("a-incomplete", || panic!("repeated cleanup invoked"))
            .unwrap();
        assert_eq!(reopened.model_source_operations().unwrap(), ["z-prepared"]);
        reopened
            .release_source_preparation("z-prepared", || Ok(()))
            .unwrap();
        assert!(reopened.model_source_operations().unwrap().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn corrupt_source_operation_rows_refuse_before_any_cleanup() {
        for mutation in [
            "UPDATE tensorfs_source_preparations SET request_digest='bad-digest'",
            "UPDATE tensorfs_operations SET name='derived'",
            "UPDATE tensorfs_operations SET org='another-owner'",
            "PRAGMA foreign_keys=OFF; DELETE FROM tensorfs_operations",
            "PRAGMA ignore_check_constraints=ON; UPDATE tensorfs_source_preparations SET state='closed'",
            "PRAGMA foreign_keys=OFF; UPDATE tensorfs_operations SET id='../outside'; UPDATE tensorfs_source_preparations SET operation_id='../outside'",
        ] {
            let root = temporary("bad-source-roster");
        fs::create_dir_all(&root).unwrap();
            let catalog = Catalog::initialize(&root).unwrap();
            let source = catalog.begin_source_preparation("source", &format!("sha256:{}", "42".repeat(32))).unwrap();
            drop(source);
            catalog.connection().unwrap().execute_batch(mutation).unwrap();
            let before = fs::read(Catalog::path(&root)).unwrap();
            assert!(catalog.model_source_operations().is_err(), "admitted {mutation}");
            let requested = if mutation.contains("../outside") { "../outside" } else { "source" };
            assert!(catalog.release_source_preparation(requested, || panic!("corrupt operation cleaned")).is_err());
            assert_eq!(fs::read(Catalog::path(&root)).unwrap(), before);
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn source_operation_identifiers_cannot_address_parent_or_current_directory() {
        let root = temporary("source-roster-paths");
        fs::create_dir_all(&root).unwrap();
        let catalog = Catalog::initialize(&root).unwrap();
        for operation in ["", ".", "..", "a/b", "../outside", "bad name"] {
            assert_eq!(
                catalog
                    .begin_source_preparation(operation, &format!("sha256:{}", "42".repeat(32)))
                    .err()
                    .unwrap()
                    .code,
                Code::KEY_GRAMMAR
            );
            assert_eq!(
                catalog
                    .release_source_preparation(operation, || panic!("invalid path cleaned"))
                    .unwrap_err()
                    .code,
                Code::KEY_GRAMMAR
            );
        }
        assert!(catalog.model_source_operations().unwrap().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn releasing_one_source_does_not_read_unrelated_corrupt_roster_entries() {
        let root = temporary("targeted-source-release");
        fs::create_dir_all(&root).unwrap();
        let catalog = Catalog::initialize(&root).unwrap();
        let digest = format!("sha256:{}", "42".repeat(32));
        drop(catalog.begin_source_preparation("good", &digest).unwrap());
        drop(
            catalog
                .begin_source_preparation("unrelated", &digest)
                .unwrap(),
        );
        catalog.connection().unwrap().execute(
            "UPDATE tensorfs_source_preparations SET request_digest='corrupt' WHERE operation_id='unrelated'", [],
        ).unwrap();
        assert!(catalog.model_source_operations().is_err());
        let cleaned = std::cell::Cell::new(false);
        catalog
            .release_source_preparation("good", || {
                cleaned.set(true);
                Ok(())
            })
            .unwrap();
        assert!(cleaned.get());
        catalog
            .release_source_preparation("good", || panic!("repeated cleanup"))
            .unwrap();
        assert!(catalog
            .release_source_preparation("unrelated", || panic!("corrupt cleanup"))
            .is_err());
        let _ = fs::remove_dir_all(root);
    }
}
