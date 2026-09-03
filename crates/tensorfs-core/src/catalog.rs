//! SQLite is a rebuildable projection and operation coordinator. Repository/manifests/blobs
//! remain the authority; SQL is never consulted to decide GC liveness.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Duration;

use fs2::FileExt;
use rusqlite::{params, Connection};

use crate::err::{refuse, Code, Refusal, Result};
use crate::storage::Projection;

/// The longest one catalog call may SLEEP on another writer's lock before it refuses. It is
/// the only wait a store operation can legitimately spend blocked and idle, so it is also
/// the floor under any patience a supervisor grants a worker that has gone quiet.
pub const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

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
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {error}", what.as_ref()),
    }
}

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {error}", what.as_ref()),
    }
}

fn verify_schema(connection: &Connection) -> Result<()> {
    let mut statement = connection
        .prepare(
            "SELECT name FROM sqlite_master
             WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        .map_err(|error| sql("prepare catalog schema census", error))?;
    let tables: Vec<String> = statement
        .query_map([], |row| row.get(0))
        .map_err(|error| sql("read catalog schema census", error))?
        .collect::<std::result::Result<_, _>>()
        .map_err(|error| sql("collect catalog schema census", error))?;
    let expected: Vec<String> = SCHEMA.iter().map(|(name, _)| (*name).to_string()).collect();
    if tables != expected {
        return refuse(
            Code::UNKNOWN_FORMAT,
            "tensorfs.sqlite is not the exact v1 catalog; remove it and rebuild from repository files",
        );
    }
    for (table, expected_columns) in SCHEMA {
        let mut columns = connection
            .prepare("SELECT name FROM pragma_table_info(?1) ORDER BY cid")
            .map_err(|error| sql("prepare catalog column census", error))?;
        let actual: Vec<String> = columns
            .query_map([table], |row| row.get(0))
            .map_err(|error| sql("read catalog column census", error))?
            .collect::<std::result::Result<_, _>>()
            .map_err(|error| sql("collect catalog column census", error))?;
        let expected: Vec<String> = expected_columns
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        let strict: u64 = connection
            .query_row(
                "SELECT strict FROM pragma_table_list WHERE schema='main' AND name=?1",
                [table],
                |row| row.get(0),
            )
            .map_err(|error| sql("read catalog strict-table census", error))?;
        if actual != expected || strict != 1 {
            return refuse(
                Code::UNKNOWN_FORMAT,
                format!(
                    "tensorfs.sqlite table {table} is not the exact v1 shape; remove the catalog and rebuild from repository files"
                ),
            );
        }
    }
    Ok(())
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
        if !path.is_file() {
            return refuse(
                Code::DURABILITY_UNPROVEN,
                format!(
                    "{} is absent; store is recovery-only until `tfs store rebuild`",
                    path.display()
                ),
            );
        }
        let catalog = Self { path };
        let connection = catalog.connection()?;
        let version: u64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(|error| sql("read catalog version", error))?;
        if version != SCHEMA_VERSION {
            return refuse(
                Code::UNKNOWN_FORMAT,
                format!("tensorfs.sqlite user_version {version} != {SCHEMA_VERSION}"),
            );
        }
        verify_schema(&connection)?;
        Ok(catalog)
    }

    fn connection(&self) -> Result<Connection> {
        let connection =
            Connection::open(&self.path).map_err(|error| sql("open catalog", error))?;
        connection
            .busy_timeout(BUSY_TIMEOUT)
            .map_err(|error| sql("set catalog busy timeout", error))?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")
            .map_err(|error| sql("configure catalog", error))?;
        Ok(connection)
    }

    fn initialize_schema(&self) -> Result<()> {
        let connection = self.connection()?;
        connection
            .execute_batch("BEGIN IMMEDIATE")
            .map_err(|error| sql("lock catalog initialization", error))?;
        let version: u64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(|error| sql("read catalog version", error))?;
        if version == SCHEMA_VERSION {
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
                    "tensorfs.sqlite user_version {version} is not supported; remove it and rebuild from repository files"
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
        crate::ids::ascii_name(
            "source preparation operation",
            operation,
            crate::limits::MAX_NAME_BYTES,
        )?;
        let request_digest = crate::ids::prefixed("source preparation request", request_digest)?;
        let guard = self.resume_operation(operation, "_tensorfs", "model-source-preparation")?;
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

    /// Explicit lifecycle release. Acquiring the operation writer lock first prevents a
    /// caller from dropping holds under an in-flight preparation.
    pub fn release_source_preparation(&self, operation: &str) -> Result<()> {
        let count: u64 = self
            .connection()?
            .query_row(
                "SELECT count(*) FROM tensorfs_source_preparations WHERE operation_id=?1",
                [operation],
                |row| row.get(0),
            )
            .map_err(|error| sql("find source preparation for release", error))?;
        if count != 1 {
            return refuse(Code::ROOT_ABSENT, "model-source preparation is absent");
        }
        let _guard = self.resume_operation(operation, "_tensorfs", "model-source-preparation")?;
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

    pub fn hold(&self, operation: &str, key: &str, length: u64) -> Result<()> {
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
        let connection = self.connection()?;
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
        self.connection()?
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

    pub(crate) fn verification(&self, sha256: &str) -> Result<Option<crate::store::VRecord>> {
        use rusqlite::OptionalExtension;
        let row: Option<(u64, String, String, u64, u64)> = self
            .connection()?
            .query_row(
                "SELECT length,dev,ino,mtime_s,mtime_ns
                 FROM tensorfs_verified_blobs WHERE sha256=?1",
                [sha256],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| sql("read blob verification", error))?;
        row.map(|(length, dev, ino, mtime_s, mtime_ns)| {
            Ok(crate::store::VRecord {
                sha256: sha256.to_string(),
                length,
                dev: dev.parse().map_err(|_| Refusal {
                    code: Code::IO_FAILED,
                    detail: "invalid verification dev".into(),
                })?,
                ino: ino.parse().map_err(|_| Refusal {
                    code: Code::IO_FAILED,
                    detail: "invalid verification ino".into(),
                })?,
                mtime_s,
                mtime_ns,
            })
        })
        .transpose()
    }

    pub(crate) fn drop_verification(&self, sha256: &str) {
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
                pruned += 1;
            }
        }
        Ok(pruned)
    }
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

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-catalog-{name}-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ))
    }

    #[test]
    fn catalog_has_seven_tables_and_refuses_future_versions_without_mutation() {
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
        connection.execute_batch("PRAGMA user_version=3").unwrap();
        drop(connection);
        assert_eq!(
            Catalog::initialize(&root).unwrap_err().code,
            Code::UNKNOWN_FORMAT
        );
        let connection = rusqlite::Connection::open(Catalog::path(&root)).unwrap();
        let version: u64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 3);
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
}
