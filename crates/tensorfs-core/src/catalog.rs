//! SQLite is a rebuildable projection and operation coordinator. Repository/manifests/blobs
//! remain the authority; SQL is never consulted to decide GC liveness.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Duration;

use fs2::FileExt;
use rusqlite::{params, Connection};

use crate::err::{refuse, Code, Refusal, Result};
use crate::storage::Projection;

const ABANDON_AFTER_SECS: u64 = 7 * 24 * 60 * 60;
const SCHEMA_VERSION: u64 = 2;

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

#[derive(Debug, Clone)]
pub struct Catalog {
    path: PathBuf,
}

impl Catalog {
    pub fn path(root: &Path) -> PathBuf {
        root.join("tensorfs.sqlite")
    }

    pub fn initialize(root: &Path) -> Result<Self> {
        let catalog = Self {
            path: Self::path(root),
        };
        catalog.migrate()?;
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
        Ok(catalog)
    }

    fn connection(&self) -> Result<Connection> {
        let connection =
            Connection::open(&self.path).map_err(|error| sql("open catalog", error))?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|error| sql("set catalog busy timeout", error))?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")
            .map_err(|error| sql("configure catalog", error))?;
        Ok(connection)
    }

    fn migrate(&self) -> Result<()> {
        let connection = self.connection()?;
        let version: u64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(|error| sql("read catalog version", error))?;
        match version {
            0 => connection
                .execute_batch(
                "BEGIN IMMEDIATE;
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
                   writer_marker TEXT NOT NULL,
                   last_progress_at INTEGER NOT NULL
                 ) STRICT;
                 CREATE TABLE IF NOT EXISTS tensorfs_holds (
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
            )
            .map_err(|error| sql("create catalog", error)),
            1 => connection
                .execute_batch(
                "BEGIN IMMEDIATE;
                 CREATE TABLE tensorfs_released_manifests (
                   org TEXT NOT NULL,
                   name TEXT NOT NULL,
                   manifest_sha256 TEXT NOT NULL,
                   PRIMARY KEY (org,name,manifest_sha256)
                 ) STRICT;
                 INSERT INTO tensorfs_released_manifests(org,name,manifest_sha256)
                   SELECT DISTINCT org,name,manifest_sha256 FROM tensorfs_releases;
                 CREATE TABLE tensorfs_operations_v2 (
                   id TEXT PRIMARY KEY,
                   org TEXT NOT NULL,
                   name TEXT NOT NULL,
                   writer_marker TEXT NOT NULL,
                   last_progress_at INTEGER NOT NULL
                 ) STRICT;
                 INSERT INTO tensorfs_operations_v2(id,org,name,writer_marker,last_progress_at)
                   SELECT id,org,name,writer_marker,last_progress_at
                   FROM tensorfs_operations WHERE state='open';
                 CREATE TABLE tensorfs_holds_v2 (
                   operation_id TEXT NOT NULL,
                   key TEXT NOT NULL,
                   length INTEGER NOT NULL,
                   PRIMARY KEY (operation_id,key),
                   FOREIGN KEY (operation_id) REFERENCES tensorfs_operations_v2(id) ON DELETE CASCADE
                 ) STRICT;
                 INSERT INTO tensorfs_holds_v2(operation_id,key,length)
                   SELECT h.operation_id,h.key,h.length FROM tensorfs_holds h
                   JOIN tensorfs_operations o ON o.id=h.operation_id WHERE o.state='open';
                 DROP TABLE tensorfs_read_lease_objects;
                 DROP TABLE tensorfs_read_leases;
                 DROP TABLE tensorfs_release_encodings;
                 DROP TABLE tensorfs_release_components;
                 DROP TABLE tensorfs_release_configs;
                 DROP TABLE tensorfs_release_objects;
                 DROP TABLE tensorfs_releases;
                 DROP TABLE tensorfs_repos;
                 DROP TABLE tensorfs_holds;
                 DROP TABLE tensorfs_operations;
                 ALTER TABLE tensorfs_operations_v2 RENAME TO tensorfs_operations;
                 ALTER TABLE tensorfs_holds_v2 RENAME TO tensorfs_holds;
                 PRAGMA user_version=2;
                 COMMIT;",
            )
            .map_err(|error| sql("migrate catalog v1 to v2", error)),
            SCHEMA_VERSION => Ok(()),
            _ => refuse(
                Code::UNKNOWN_FORMAT,
                format!("tensorfs.sqlite user_version {version} is newer than {SCHEMA_VERSION}"),
            ),
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

    pub fn begin_operation(
        &self,
        id: &str,
        org: &str,
        name: &str,
        now: u64,
    ) -> Result<WriterGuard> {
        let root = self.path.parent().ok_or_else(|| Refusal {
            code: Code::IO_FAILED,
            detail: "catalog has no store parent".into(),
        })?;
        let guard = WriterGuard::acquire(root)?;
        self.connection()?
            .execute(
                "INSERT INTO tensorfs_operations
                 (id,org,name,writer_marker,last_progress_at) VALUES (?1,?2,?3,?4,?5)",
                params![id, org, name, guard.path.to_string_lossy(), now],
            )
            .map_err(|error| sql("begin operation", error))?;
        Ok(guard)
    }

    /// Claim an existing open operation for this process, or recreate its coordination row
    /// after an explicit catalog rebuild. The returned advisory-lock guard must live for the
    /// whole resumed writer/install process.
    pub fn resume_operation(
        &self,
        id: &str,
        org: &str,
        name: &str,
        now: u64,
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
            None => {
                transaction
                    .execute(
                        "INSERT INTO tensorfs_operations
                         (id,org,name,writer_marker,last_progress_at) VALUES (?1,?2,?3,?4,?5)",
                        params![id, org, name, guard.path.to_string_lossy(), now],
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
                         SET writer_marker=?2,last_progress_at=?3
                         WHERE id=?1",
                        params![id, guard.path.to_string_lossy(), now],
                    )
                    .map_err(|error| sql("claim resumed operation", error))?;
            }
        }
        transaction
            .commit()
            .map_err(|error| sql("commit operation resume", error))?;
        Ok(guard)
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

    pub fn progress(&self, operation: &str, now: u64, objects: u64, bytes: u64) -> Result<()> {
        let changed = self
            .connection()?
            .execute(
                "UPDATE tensorfs_operations SET last_progress_at=?2
                 WHERE id=?1 AND (?3>0 OR ?4>0)",
                params![operation, now, objects, bytes],
            )
            .map_err(|error| sql("record operation progress", error))?;
        if changed != 1 {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "progress requires an open operation and positive accepted progress",
            );
        }
        Ok(())
    }

    /// Explicit abandonment. Seven progress-free days only makes this transition eligible;
    /// it never fires the transition itself.
    pub fn abandon(&self, operation: &str, now: u64, grants_revoked: bool) -> Result<()> {
        if !grants_revoked {
            return refuse(Code::STORE_BUSY, "abandon requires revoked/expired grants");
        }
        let mut connection = self.connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| sql("begin abandon", error))?;
        let (writer_marker, last_progress): (String, u64) = transaction
            .query_row(
                "SELECT writer_marker,last_progress_at FROM tensorfs_operations WHERE id=?1",
                [operation],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|error| sql("read abandon candidate", error))?;
        if now.saturating_sub(last_progress) < ABANDON_AFTER_SECS {
            return refuse(
                Code::ROOT_EXPIRED,
                "operation has not been progress-free for seven days",
            );
        }
        if marker_is_live(Path::new(&writer_marker))? {
            return refuse(Code::STORE_BUSY, "operation writer is still live");
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
}

fn insert_projection(transaction: &Connection, projection: &Projection) -> Result<()> {
    for row in &projection.releases {
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

impl WriterGuard {
    pub fn acquire(root: &Path) -> Result<Self> {
        let directory = root.join("tmp/writers");
        fs::create_dir_all(&directory).map_err(|error| io("mkdir writer markers", error))?;
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
        let directory = root.join("tmp/writers");
        fs::create_dir_all(&directory).map_err(|error| io("mkdir writer markers", error))?;
        let recovery = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.join("recovery.lock"))
            .map_err(|error| io("open recovery lock", error))?;
        recovery.try_lock_exclusive().map_err(|_| Refusal {
            code: Code::STORE_BUSY,
            detail: "a live writer blocks rebuild".into(),
        })?;
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
                return refuse(Code::STORE_BUSY, "a live writer blocks rebuild");
            }
            fs::remove_file(path).map_err(|error| io("reap dead writer marker", error))?;
        }
        Ok(RebuildGuard { _file: recovery })
    }
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
    fn catalog_has_five_tables_and_refuses_future_versions_without_mutation() {
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
    fn version_one_migrates_only_live_coordination_and_released_manifests() {
        let root = temporary("v1-migration");
        fs::create_dir_all(&root).unwrap();
        let connection = rusqlite::Connection::open(Catalog::path(&root)).unwrap();
        connection
            .execute_batch(
                "PRAGMA foreign_keys=OFF;
                 CREATE TABLE tensorfs_repos(org TEXT,name TEXT,document_sha256 TEXT);
                 CREATE TABLE tensorfs_releases(org TEXT,name TEXT,manifest_sha256 TEXT);
                 CREATE TABLE tensorfs_release_objects(value TEXT);
                 CREATE TABLE tensorfs_release_encodings(value TEXT);
                 CREATE TABLE tensorfs_release_components(value TEXT);
                 CREATE TABLE tensorfs_release_configs(value TEXT);
                 CREATE TABLE tensorfs_operations(
                   id TEXT PRIMARY KEY,org TEXT,name TEXT,state TEXT,writer_marker TEXT,
                   last_progress_at INTEGER,accepted_objects INTEGER,accepted_bytes INTEGER);
                 CREATE TABLE tensorfs_holds(operation_id TEXT,key TEXT,length INTEGER);
                 CREATE TABLE tensorfs_read_leases(id TEXT);
                 CREATE TABLE tensorfs_read_lease_objects(lease_id TEXT);
                 CREATE TABLE tensorfs_derived_transactions(id TEXT PRIMARY KEY,bytes BLOB);
                 CREATE TABLE tensorfs_verified_blobs(
                   sha256 TEXT PRIMARY KEY,length INTEGER,dev TEXT,ino TEXT,
                   mtime_s INTEGER,mtime_ns INTEGER);
                 INSERT INTO tensorfs_releases VALUES ('org','repo','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
                 INSERT INTO tensorfs_operations VALUES ('open','org','repo','open','dead',7,1,9);
                 INSERT INTO tensorfs_operations VALUES ('closed','org','repo','committed','dead',8,2,10);
                 INSERT INTO tensorfs_holds VALUES ('open','blobs/aa',9);
                 INSERT INTO tensorfs_holds VALUES ('closed','blobs/bb',10);
                 INSERT INTO tensorfs_derived_transactions VALUES ('derived',x'7b7d');
                 INSERT INTO tensorfs_verified_blobs VALUES ('bb',1,'1','2',3,4);
                 PRAGMA user_version=1;",
            )
            .unwrap();
        drop(connection);

        let catalog = Catalog::initialize(&root).unwrap();
        let connection = catalog.connection().unwrap();
        let released: u64 = connection
            .query_row(
                "SELECT count(*) FROM tensorfs_released_manifests",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let operations: Vec<String> = connection
            .prepare("SELECT id FROM tensorfs_operations ORDER BY id")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        let holds: u64 = connection
            .query_row("SELECT count(*) FROM tensorfs_holds", [], |row| row.get(0))
            .unwrap();
        let derived: u64 = connection
            .query_row(
                "SELECT count(*) FROM tensorfs_derived_transactions",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let verified: u64 = connection
            .query_row("SELECT count(*) FROM tensorfs_verified_blobs", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(released, 1);
        assert_eq!(operations, vec!["open"]);
        assert_eq!(holds, 1);
        assert_eq!(derived, 1);
        assert_eq!(verified, 1);
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
    fn exact_holds_progress_and_explicit_abandonment_are_closed() {
        let root = temporary("operation");
        fs::create_dir_all(&root).unwrap();
        let catalog = Catalog::initialize(&root).unwrap();
        let guard = catalog.begin_operation("op", "org", "repo", 10).unwrap();
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
        catalog.progress("op", 20, 1, 7).unwrap();
        assert_eq!(
            catalog
                .abandon("op", 20 + ABANDON_AFTER_SECS - 1, true)
                .unwrap_err()
                .code,
            Code::ROOT_EXPIRED
        );
        assert_eq!(
            catalog
                .abandon("op", 20 + ABANDON_AFTER_SECS, true)
                .unwrap_err()
                .code,
            Code::STORE_BUSY
        );
        drop(guard);
        catalog
            .abandon("op", 20 + ABANDON_AFTER_SECS, true)
            .unwrap();
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
        let first = catalog.begin_operation("op", "org", "repo", 10).unwrap();
        assert_eq!(
            catalog
                .resume_operation("op", "org", "repo", 11)
                .unwrap_err()
                .code,
            Code::STORE_BUSY
        );
        drop(first);
        assert_eq!(
            catalog
                .resume_operation("op", "org", "different", 12)
                .unwrap_err()
                .code,
            Code::TRANSACTION_CONFLICT
        );
        let resumed = catalog.resume_operation("op", "org", "repo", 12).unwrap();
        assert_eq!(
            catalog
                .resume_operation("op", "org", "repo", 13)
                .unwrap_err()
                .code,
            Code::STORE_BUSY
        );
        drop(resumed);

        fs::remove_file(Catalog::path(&root)).unwrap();
        let rebuilt = Catalog::initialize(&root).unwrap();
        let recovered = rebuilt.resume_operation("op", "org", "repo", 14).unwrap();
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
