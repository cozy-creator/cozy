//! Transient local coordination in SQLite.
//!
//! Repository files are the only durable catalog authority. Read leases and derived writes
//! coordinate local processes; none of these rows makes content durable or adds a stored format.

use std::fs::{self, File};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use rusqlite::{params, Connection};

use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{ascii_name, hex64};
use crate::limits;
use crate::store::Store;

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

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub fn now_nanos_unique() -> u128 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    nanos + NEXT.fetch_add(1, Ordering::Relaxed) as u128
}

#[derive(Debug, Clone)]
pub struct Meta {
    db: PathBuf,
    lease_dir: PathBuf,
}

impl Meta {
    pub fn open(store: &Store) -> Result<Self> {
        crate::catalog::Catalog::open(store.root())?;
        let lease_dir = store.root().join("tmp/leases");
        fs::create_dir_all(&lease_dir).map_err(|error| io("mkdir lease markers", error))?;
        Ok(Self {
            db: crate::catalog::Catalog::path(store.root()),
            lease_dir,
        })
    }

    fn connection(&self) -> Result<Connection> {
        let connection = Connection::open(&self.db).map_err(|error| sql("open catalog", error))?;
        connection
            .busy_timeout(std::time::Duration::from_secs(5))
            .map_err(|error| sql("set catalog busy timeout", error))?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")
            .map_err(|error| sql("configure catalog", error))?;
        Ok(connection)
    }

    fn lease_path(&self, id: &str) -> PathBuf {
        self.lease_dir.join(format!("{id}.lease"))
    }

    pub fn manifest_is_released(&self, digest: &str) -> Result<bool> {
        hex64("manifest", digest)?;
        let count: u64 = self
            .connection()?
            .query_row(
                "SELECT count(*) FROM tensorfs_releases WHERE manifest_sha256=?1",
                [digest],
                |row| row.get(0),
            )
            .map_err(|error| sql("query released manifest", error))?;
        Ok(count != 0)
    }

    pub(crate) fn txn(&self) -> Result<Txn> {
        let connection = self.connection()?;
        connection
            .execute_batch("BEGIN IMMEDIATE")
            .map_err(|error| sql("begin derived transaction", error))?;
        let rows = read_derived(&connection)?;
        Ok(Txn { connection, rows })
    }

    pub(crate) fn derived_rows(&self) -> Result<Vec<crate::derived::TransactionRow>> {
        read_derived(&self.connection()?)
    }

    pub fn acquire_hold(&self, kind: &str, manifest: &str, objects: Vec<String>) -> Result<Hold> {
        ascii_name("lease kind", kind, limits::MAX_NAME_BYTES)?;
        if !manifest.is_empty() {
            hex64("lease manifest", manifest)?;
        }
        if objects.len() > limits::MAX_ENTRIES {
            return refuse(Code::COUNT_CAP, "read lease object count exceeds cap");
        }
        let mut objects = objects;
        for digest in &objects {
            hex64("lease object", digest)?;
        }
        objects.sort();
        objects.dedup();

        let id = format!("{kind}-{}-{}", std::process::id(), now_nanos_unique());
        let path = self.lease_path(&id);
        let file = File::options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|error| io("create lease marker", error))?;
        file.try_lock_exclusive()
            .map_err(|error| io("lock lease marker", error))?;

        let mut connection = self.connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| sql("begin read lease", error))?;
        transaction
            .execute(
                "INSERT INTO tensorfs_read_leases(id,manifest_sha256,kind,owner_marker)
                 VALUES (?1,?2,?3,?4)",
                params![id, manifest, kind, path.to_string_lossy()],
            )
            .map_err(|error| sql("insert read lease", error))?;
        for digest in &objects {
            transaction
                .execute(
                    "INSERT INTO tensorfs_read_lease_objects(lease_id,sha256) VALUES (?1,?2)",
                    params![id, digest],
                )
                .map_err(|error| sql("insert read lease object", error))?;
        }
        transaction
            .commit()
            .map_err(|error| sql("commit read lease", error))?;
        Ok(Hold {
            id,
            kind: kind.to_string(),
            file: Some(file),
            path,
            released: false,
        })
    }

    pub fn reap_holds(&self) -> Result<Vec<String>> {
        let connection = self.connection()?;
        let mut statement = connection
            .prepare("SELECT id,owner_marker FROM tensorfs_read_leases ORDER BY id")
            .map_err(|error| sql("prepare lease census", error))?;
        let candidates = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| sql("read lease census", error))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| sql("collect lease census", error))?;
        drop(statement);

        let mut reaped = Vec::new();
        for (id, marker) in candidates {
            let path = PathBuf::from(marker);
            let dead = match File::options().read(true).write(true).open(&path) {
                Ok(file) => file.try_lock_exclusive().is_ok(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
                Err(error) => return Err(io("open lease marker", error)),
            };
            if dead {
                connection
                    .execute("DELETE FROM tensorfs_read_leases WHERE id=?1", [&id])
                    .map_err(|error| sql("reap read lease", error))?;
                let _ = fs::remove_file(path);
                reaped.push(id);
            }
        }
        Ok(reaped)
    }
}

pub(crate) struct Txn {
    connection: Connection,
    pub(crate) rows: Vec<crate::derived::TransactionRow>,
}

impl Txn {
    pub fn commit(mut self) -> Result<()> {
        for row in &mut self.rows {
            row.normalize();
        }
        self.rows.sort_by(|left, right| left.id.cmp(&right.id));
        if self.rows.windows(2).any(|pair| pair[0].id == pair[1].id) {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "duplicate derived transaction id",
            );
        }
        self.connection
            .execute("DELETE FROM tensorfs_derived_transactions", [])
            .map_err(|error| sql("clear derived transactions", error))?;
        for row in &self.rows {
            let bytes = crate::canon::write(&row.to_value());
            self.connection
                .execute(
                    "INSERT INTO tensorfs_derived_transactions(id,bytes) VALUES (?1,?2)",
                    params![row.id, bytes],
                )
                .map_err(|error| sql("write derived transaction", error))?;
        }
        self.connection
            .execute_batch("COMMIT")
            .map_err(|error| sql("commit derived transactions", error))
    }
}

fn read_derived(connection: &Connection) -> Result<Vec<crate::derived::TransactionRow>> {
    let mut statement = connection
        .prepare("SELECT bytes FROM tensorfs_derived_transactions ORDER BY id")
        .map_err(|error| sql("prepare derived rows", error))?;
    let values = statement
        .query_map([], |row| row.get::<_, Vec<u8>>(0))
        .map_err(|error| sql("read derived rows", error))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| sql("collect derived rows", error))?;
    values
        .into_iter()
        .map(|bytes| {
            let value = crate::canon::parse_canonical(&bytes, limits::DOC_MAX_BYTES)?;
            crate::derived::TransactionRow::from_value(&value)
        })
        .collect()
}

#[derive(Debug)]
pub struct Hold {
    id: String,
    kind: String,
    file: Option<File>,
    path: PathBuf,
    released: bool,
}

impl Hold {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn release(mut self, meta: &Meta) -> Result<()> {
        let changed = meta
            .connection()?
            .execute("DELETE FROM tensorfs_read_leases WHERE id=?1", [&self.id])
            .map_err(|error| sql("release read lease", error))?;
        if changed != 1 {
            return refuse(
                Code::LEASE_REVOKED,
                format!("lease {:?} is absent", self.id),
            );
        }
        self.file = None;
        let _ = fs::remove_file(&self.path);
        self.released = true;
        Ok(())
    }

    pub fn still_registered(&self, meta: &Meta) -> Result<()> {
        let count: u64 = meta
            .connection()?
            .query_row(
                "SELECT count(*) FROM tensorfs_read_leases WHERE id=?1",
                [&self.id],
                |row| row.get(0),
            )
            .map_err(|error| sql("check read lease", error))?;
        if count == 1 {
            Ok(())
        } else {
            refuse(
                Code::LEASE_REVOKED,
                format!("lease {:?} is absent", self.id),
            )
        }
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        if !self.released {
            eprintln!(
                "DEGRADED lease-dropped-unreleased hold={} kind={} — row retained until owner-death reap",
                self.id, self.kind
            );
        }
    }
}
