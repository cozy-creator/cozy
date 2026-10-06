//! Transient local coordination in SQLite.
//!
//! Repository files are the only durable catalog authority. Advisory marker locks protect local
//! readers; SQLite retains resumable derived writes.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use rusqlite::{params, Connection, OptionalExtension};

use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{ascii_name, ObjectRef};
use crate::limits;
use crate::store::Store;

fn sql(what: impl AsRef<str>, error: rusqlite::Error) -> Refusal {
    crate::store::classify_sql(what, error)
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
    use std::sync::Mutex;
    static LAST: Mutex<u128> = Mutex::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    // Reserve the entire name together. Adding a separately sampled clock and
    // counter can collide when threads observe those two values in reverse order.
    let mut last = LAST.lock().unwrap();
    *last = nanos.max(*last + 1);
    *last
}

/// The store's owner process (the machine). Its read leases pin their objects here instead
/// of holding `recovery.lock` shared, so its own GC keeps them and runs while layouts are
/// served. Leases taken by any other process still exclude GC.
#[derive(Debug)]
pub(crate) struct Owner {
    /// Every owned Store clone and read pin retains this same exclusive claim.
    _lock: File,
    /// Held shared while a lease admits its objects, exclusive while GC plans and deletes.
    pub(crate) gate: RwLock<()>,
    pins: Mutex<(u64, HashMap<u64, Vec<ObjectRef>>)>,
}

impl Owner {
    pub(crate) fn pinned(&self) -> Vec<ObjectRef> {
        let pins = self.pins.lock().unwrap();
        pins.1.values().flatten().cloned().collect()
    }
}

static OWNERS: Mutex<Vec<(PathBuf, Weak<Owner>)>> = Mutex::new(Vec::new());

fn canonical(root: &Path) -> PathBuf {
    fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf())
}

fn owner_file(root: &Path) -> Result<File> {
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(crate::store::O_NOFOLLOW | rustix::fs::OFlags::NONBLOCK.bits() as i32)
        .open(root.join("owner.lock"))
        .map_err(|error| io("open Store owner lock", error))?;
    if !lock
        .metadata()
        .map_err(|error| io("stat Store owner lock", error))?
        .is_file()
    {
        return refuse(
            Code::NOT_REGULAR_FILE,
            "Store owner lock is not a regular file",
        );
    }
    Ok(lock)
}

fn owner_lock_error(error: std::io::Error) -> Refusal {
    if error.kind() == std::io::ErrorKind::WouldBlock {
        Refusal {
            code: Code::STORE_BUSY,
            detail: "this TensorFS store already has an owner or active collection".into(),
        }
    } else {
        io("lock Store owner", error)
    }
}

/// Claim the owner role before opening the catalog. The caller holds the Store creation
/// lock and has established a valid layout, so adding owner.lock cannot poison a new root.
pub(crate) fn claim(root: &Path) -> Result<Arc<Owner>> {
    let lock = owner_file(root)?;
    lock.try_lock_exclusive().map_err(owner_lock_error)?;
    let owner = Arc::new(Owner {
        _lock: lock,
        gate: RwLock::default(),
        pins: Mutex::default(),
    });
    let root = canonical(root);
    let mut owners = OWNERS.lock().unwrap();
    owners.retain(|(_, owner)| owner.strong_count() != 0);
    owners.push((root, Arc::downgrade(&owner)));
    Ok(owner)
}

pub(crate) fn owner(root: &Path) -> Option<Arc<Owner>> {
    let root = canonical(root);
    let owners = OWNERS.lock().unwrap();
    owners
        .iter()
        .find(|(owned, _)| *owned == root)
        .and_then(|(_, owner)| owner.upgrade())
}

/// A destructive pass must see the owner's pins, or exclude an owner for its whole life.
/// External passes take shared locks so reclamation can call ordinary GC while retaining
/// its exclusion across root removal. Their recovery locks still serialize deletion.
#[derive(Debug)]
pub(crate) enum OwnerGuard {
    Local { _owner: Arc<Owner> },
    Shared { _file: File },
}

impl OwnerGuard {
    pub(crate) fn acquire(root: &Path) -> Result<Self> {
        if let Some(owner) = owner(root) {
            return Ok(Self::Local { _owner: owner });
        }
        let file = owner_file(root)?;
        FileExt::try_lock_shared(&file).map_err(owner_lock_error)?;
        Ok(Self::Shared { _file: file })
    }
}

#[derive(Debug, Clone)]
pub struct Meta {
    db: PathBuf,
    lease_dir: PathBuf,
    recovery_lock: PathBuf,
}

impl Meta {
    pub fn open(store: &Store) -> Result<Self> {
        store.trust().validate()?;
        let lease_dir = store.root().join("tmp/leases");
        fs::create_dir_all(&lease_dir).map_err(|error| io("mkdir lease markers", error))?;
        let writer_dir = store.root().join("tmp/writers");
        fs::create_dir_all(&writer_dir).map_err(|error| io("mkdir writer markers", error))?;
        let meta = Self {
            db: crate::catalog::Catalog::path(store.root()),
            lease_dir,
            recovery_lock: writer_dir.join("recovery.lock"),
        };
        store.trust().with_connection(Self::ensure_part_rows)?;
        Ok(meta)
    }

    /// Accepted parts are rows of their own, so accepting one costs that one part. Older
    /// builds ignore both additions; a row one of them rewrites has no `open_session`,
    /// which sends this build back to its full-row check.
    fn ensure_part_rows(connection: &mut Connection) -> Result<()> {
        let transaction = connection
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| sql("lock derived part schema", error))?;
        let has_session: bool = transaction
            .query_row(
                "SELECT count(*) FROM pragma_table_info('tensorfs_derived_transactions')
                 WHERE name='open_session'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|error| sql("inspect derived transaction columns", error))?
            > 0;
        if !has_session {
            transaction
                .execute_batch(
                    "ALTER TABLE tensorfs_derived_transactions ADD COLUMN open_session INTEGER",
                )
                .map_err(|error| sql("add derived open session", error))?;
        }
        transaction
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS tensorfs_derived_parts (
                   id TEXT NOT NULL,
                   component TEXT NOT NULL,
                   key TEXT NOT NULL,
                   role TEXT NOT NULL,
                   part BLOB NOT NULL,
                   PRIMARY KEY (id,component,key,role)
                 ) STRICT, WITHOUT ROWID;",
            )
            .map_err(|error| sql("create derived parts", error))?;
        transaction
            .commit()
            .map_err(|error| sql("commit derived part schema", error))
    }

    /// Refuse unless `writer_session` is the transaction's current writer. One indexed read
    /// when the row names its open session; otherwise the full row gives the exact refusal.
    pub(crate) fn check_session(&self, id: &str, writer_session: u64) -> Result<()> {
        let connection = self.connection()?;
        let open: Option<Option<i64>> = connection
            .query_row(
                "SELECT open_session FROM tensorfs_derived_transactions WHERE id=?1",
                [id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| sql("read writer session", error))?;
        if matches!(open, Some(Some(session)) if session as u64 == writer_session) {
            return Ok(());
        }
        let mut rows: Vec<_> = read_derived_row(&connection, id)?.into_iter().collect();
        crate::derived::open_row(&mut rows, id, writer_session).map(|_| ())
    }

    pub(crate) fn part_count(&self, id: &str) -> Result<usize> {
        self.connection()?
            .query_row(
                "SELECT count(*) FROM tensorfs_derived_parts WHERE id=?1",
                [id],
                |row| row.get::<_, i64>(0),
            )
            .map(|count| count as usize)
            .map_err(|error| sql("count derived parts", error))
    }

    /// Record one accepted part for the current writer: one indexed read of the writer's
    /// session and of that part. `Ok(false)` is the same part accepted before.
    pub(crate) fn accept_part(
        &self,
        id: &str,
        writer_session: u64,
        added: &crate::derived::AddedPart,
    ) -> Result<bool> {
        let connection = self.connection()?;
        connection
            .execute_batch("BEGIN IMMEDIATE")
            .map_err(|error| sql("begin part acceptance", error))?;
        let open: Option<Option<i64>> = connection
            .query_row(
                "SELECT open_session FROM tensorfs_derived_transactions WHERE id=?1",
                [id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| sql("read writer session", error))?;
        let legacy = match open {
            Some(Some(session)) if session as u64 == writer_session => None,
            _ => {
                // Absent, closed, fenced, or last written by a build without the column:
                // the full row names the exact refusal, or admits this writer.
                let mut rows: Vec<_> = read_derived_row(&connection, id)?.into_iter().collect();
                let row = crate::derived::open_row(&mut rows, id, writer_session)?;
                Some(row.added_parts.clone())
            }
        };
        let bytes = crate::derived::part_bytes(&added.part);
        let existing: Option<Vec<u8>> = connection
            .query_row(
                "SELECT part FROM tensorfs_derived_parts
                 WHERE id=?1 AND component=?2 AND key=?3 AND role=?4",
                params![id, added.component, added.key, added.role],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| sql("read accepted part", error))?
            .or_else(|| {
                legacy
                    .iter()
                    .flatten()
                    .find(|old| old.same_value(added))
                    .map(|old| crate::derived::part_bytes(&old.part))
            });
        match existing {
            Some(existing) if existing == bytes => {
                let _ = connection.execute_batch("ROLLBACK");
                Ok(false)
            }
            Some(_) => {
                let _ = connection.execute_batch("ROLLBACK");
                refuse(
                    Code::TRANSACTION_CONFLICT,
                    format!(
                        "added value {}/{}#{} differs from accepted bytes",
                        added.component, added.key, added.role
                    ),
                )
            }
            None => {
                connection
                    .execute(
                        "INSERT INTO tensorfs_derived_parts(id,component,key,role,part)
                         VALUES (?1,?2,?3,?4,?5)",
                        params![id, added.component, added.key, added.role, bytes],
                    )
                    .map_err(|error| sql("write accepted part", error))?;
                connection
                    .execute_batch("COMMIT")
                    .map_err(|error| sql("commit accepted part", error))?;
                Ok(true)
            }
        }
    }

    fn connection(&self) -> Result<Connection> {
        crate::catalog::connect(&self.db)
    }

    fn lease_path(&self, id: &str) -> PathBuf {
        self.lease_dir.join(format!("{id}.lease"))
    }

    pub(crate) fn txn(&self) -> Result<Txn> {
        let connection = self.connection()?;
        connection
            .execute_batch("BEGIN IMMEDIATE")
            .map_err(|error| sql("begin derived transaction", error))?;
        let rows = read_derived(&connection)?;
        Ok(Txn {
            connection,
            rows,
            selected: None,
        })
    }

    /// A writer mutates its own primary-key row; unrelated writers and retained history
    /// must neither be decoded nor rewritten. The existing write lock still serializes
    /// epoch changes, roots and accepted parts against global lifecycle operations.
    pub(crate) fn txn_for(&self, id: &str) -> Result<Txn> {
        let connection = self.connection()?;
        connection
            .execute_batch("BEGIN IMMEDIATE")
            .map_err(|error| sql("begin derived transaction", error))?;
        let rows = read_derived_row(&connection, id)?.into_iter().collect();
        Ok(Txn {
            connection,
            rows,
            selected: Some(id.to_string()),
        })
    }

    /// Observations do not acquire SQLite's writer lock. Source reads check this both
    /// before and after I/O, so a concurrent fence still invalidates the read.
    pub(crate) fn derived_row(&self, id: &str) -> Result<Option<crate::derived::TransactionRow>> {
        read_derived_row(&self.connection()?, id)
    }

    pub(crate) fn derived_rows(&self) -> Result<Vec<crate::derived::TransactionRow>> {
        read_derived(&self.connection()?)
    }

    pub fn acquire_hold(&self, kind: &str) -> Result<Hold> {
        ascii_name("lease kind", kind, limits::MAX_NAME_BYTES)?;
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
        let recovery = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&self.recovery_lock)
            .map_err(|error| io("open recovery lock", error))?;
        FileExt::lock_shared(&recovery).map_err(|error| io("share recovery lock", error))?;

        Ok(Hold {
            id,
            kind: kind.to_string(),
            file: Some(file),
            _recovery: Some(recovery),
            path,
            released: false,
            pin: None,
        })
    }

    /// A read hold over `objects`: an in-process pin when this process owns the store, else
    /// a marker that excludes GC.
    pub(crate) fn read_hold(&self, owner: Option<&Arc<Owner>>, objects: &[ObjectRef]) -> Result<Hold> {
        let Some(owner) = owner else {
            return self.acquire_hold("read");
        };
        let mut pins = owner.pins.lock().unwrap();
        pins.0 += 1;
        let pin = pins.0;
        pins.1.insert(pin, objects.to_vec());
        Ok(Hold {
            id: format!("read-pin-{}-{pin}", std::process::id()),
            kind: "read".into(),
            file: None,
            _recovery: None,
            path: PathBuf::new(),
            released: false,
            pin: Some((owner.clone(), pin)),
        })
    }

    pub fn reap_holds(&self) -> Result<Vec<String>> {
        let mut reaped = Vec::new();
        for entry in
            fs::read_dir(&self.lease_dir).map_err(|error| io("read lease markers", error))?
        {
            let path = entry
                .map_err(|error| io("read lease marker", error))?
                .path();
            let Some(id) = path
                .file_stem()
                .and_then(|value| value.to_str())
                .map(str::to_owned)
            else {
                continue;
            };
            let dead = match File::options().read(true).write(true).open(&path) {
                Ok(file) => file.try_lock_exclusive().is_ok(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
                Err(error) => return Err(io("open lease marker", error)),
            };
            if dead {
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
    selected: Option<String>,
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
        if let Some(id) = &self.selected {
            if self.rows.len() > 1 || self.rows.iter().any(|row| &row.id != id) {
                return refuse(
                    Code::TRANSACTION_CONFLICT,
                    "scoped transaction changed another writer",
                );
            }
            self.connection
                .execute(
                    "DELETE FROM tensorfs_derived_transactions WHERE id=?1",
                    [id],
                )
                .map_err(|error| sql("replace derived transaction", error))?;
        } else {
            self.connection
                .execute("DELETE FROM tensorfs_derived_transactions", [])
                .map_err(|error| sql("clear derived transactions", error))?;
        }
        match &self.selected {
            Some(id) => self
                .connection
                .execute("DELETE FROM tensorfs_derived_parts WHERE id=?1", [id]),
            None => self
                .connection
                .execute("DELETE FROM tensorfs_derived_parts", []),
        }
        .map_err(|error| sql("replace derived parts", error))?;
        for row in &self.rows {
            let (bytes, open) = row.stored();
            self.connection
                .execute(
                    "INSERT INTO tensorfs_derived_transactions(id,bytes,open_session)
                     VALUES (?1,?2,?3)",
                    params![row.id, bytes, open.map(|session| session as i64)],
                )
                .map_err(|error| sql("write derived transaction", error))?;
            for added in &row.added_parts {
                self.connection
                    .execute(
                        "INSERT INTO tensorfs_derived_parts(id,component,key,role,part)
                         VALUES (?1,?2,?3,?4,?5)",
                        params![
                            row.id,
                            added.component,
                            added.key,
                            added.role,
                            crate::derived::part_bytes(&added.part)
                        ],
                    )
                    .map_err(|error| sql("write derived part", error))?;
            }
        }
        self.connection
            .execute_batch("COMMIT")
            .map_err(|error| sql("commit derived transactions", error))
    }
}

fn read_derived_row(
    connection: &Connection,
    id: &str,
) -> Result<Option<crate::derived::TransactionRow>> {
    let bytes = connection
        .query_row(
            "SELECT bytes FROM tensorfs_derived_transactions WHERE id=?1",
            [id],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .optional()
        .map_err(|error| sql("read derived transaction", error))?;
    bytes
        .map(|bytes| {
            let value = crate::canon::parse_canonical(&bytes, limits::DOC_MAX_BYTES)?;
            let mut row = crate::derived::TransactionRow::from_value(&value)?;
            row.accept_stored(read_parts(connection, id)?)?;
            if row.id != id {
                return refuse(
                    Code::TRANSACTION_CONFLICT,
                    "derived row differs from its primary key",
                );
            }
            Ok(row)
        })
        .transpose()
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
            let mut row = crate::derived::TransactionRow::from_value(&value)?;
            row.accept_stored(read_parts(connection, &row.id.clone())?)?;
            Ok(row)
        })
        .collect()
}

fn read_parts(connection: &Connection, id: &str) -> Result<Vec<crate::derived::AddedPart>> {
    let mut statement = connection
        .prepare(
            "SELECT component,key,role,part FROM tensorfs_derived_parts
             WHERE id=?1 ORDER BY component,key,role",
        )
        .map_err(|error| sql("prepare derived parts", error))?;
    let rows = statement
        .query_map([id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Vec<u8>>(3)?,
            ))
        })
        .map_err(|error| sql("read derived parts", error))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| sql("collect derived parts", error))?;
    rows.into_iter()
        .map(|(component, key, role, bytes)| {
            crate::derived::AddedPart::stored(component, key, role, &bytes)
        })
        .collect()
}

#[derive(Debug)]
pub struct Hold {
    id: String,
    kind: String,
    file: Option<File>,
    _recovery: Option<File>,
    path: PathBuf,
    released: bool,
    pin: Option<(Arc<Owner>, u64)>,
}

impl Hold {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn release(mut self, meta: &Meta) -> Result<()> {
        let _ = meta;
        if let Some((owner, pin)) = &self.pin {
            owner.pins.lock().unwrap().1.remove(pin);
            self.released = true;
            return Ok(());
        }
        fs::remove_file(&self.path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Refusal {
                    code: Code::LEASE_REVOKED,
                    detail: format!("lease {:?} is absent", self.id),
                }
            } else {
                io("remove lease marker", error)
            }
        })?;
        self.file = None;
        self.released = true;
        Ok(())
    }

    pub fn still_registered(&self, meta: &Meta) -> Result<()> {
        let _ = meta;
        let registered = match &self.pin {
            Some((owner, pin)) => owner.pins.lock().unwrap().1.contains_key(pin),
            None => self.file.is_some() && self.path.is_file(),
        };
        if registered {
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

#[cfg(test)]
mod setup_tests {
    use super::*;

    #[test]
    fn unique_nanos_never_repeat_across_threads() {
        let minted: Vec<u128> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..16)
                .map(|_| scope.spawn(|| (0..20_000).map(|_| now_nanos_unique()).collect::<Vec<_>>()))
                .collect();
            workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
        });
        let distinct: std::collections::HashSet<_> = minted.iter().collect();
        assert_eq!(distinct.len(), minted.len());
    }

    #[test]
    fn failed_additive_setup_rolls_back_and_reuses_the_validated_connection() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-meta-setup-{}-{}",
            std::process::id(),
            now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let opened = crate::catalog::connections_opened(&root);
        store
            .trust()
            .with_connection(|connection| {
                // Existing readers use the original two-column table. An index
                // collision makes the later CREATE fail after ALTER has run.
                connection
                    .execute_batch(
                        "CREATE INDEX tensorfs_derived_parts ON tensorfs_derived_transactions(id)",
                    )
                    .map_err(|error| sql("create conflicting fixture index", error))
            })
            .unwrap();
        assert!(Meta::open(&store).is_err());
        store
            .trust()
            .with_connection(|connection| {
                assert!(
                    connection.is_autocommit(),
                    "failed setup kept its write lock"
                );
                let added: u64 = connection
                    .query_row(
                        "SELECT count(*) FROM pragma_table_info('tensorfs_derived_transactions')
                         WHERE name='open_session'",
                        [],
                        |row| row.get(0),
                    )
                    .map_err(|error| sql("inspect rolled-back schema", error))?;
                assert_eq!(added, 0, "failed setup published a partial schema");
                connection
                    .execute_batch("DROP INDEX tensorfs_derived_parts")
                    .map_err(|error| sql("remove fixture index", error))
            })
            .unwrap();
        Meta::open(&store).unwrap();
        assert_eq!(crate::catalog::connections_opened(&root) - opened, 1);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }
}
