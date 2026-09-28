//! Independent retained ownership of an existing result; its execution receipt stays immutable.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

use super::{
    checkpoint, receipt_from, require_payload_record, roots, Disposition, TransactionState,
};
use crate::canon::{Fields, Value};
use crate::catalog::WriterGuard;
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{object_id, prefixed, ObjectRef, Plain};
use crate::meta::Meta;
use crate::storage::HeldKey;
use crate::store::Store;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedResult {
    pub retention_id: String,
    pub transaction_id: String,
    pub receipt_digest: String,
    pub manifest: Option<ObjectRef>,
    pub released: bool,
    /// The newest TensorFS that wrote this root.
    pub tensorfs: Option<String>,
}

impl Plain for RetainedResult {
    const MAX_BYTES: usize = 4096;

    fn from_value(value: &Value) -> Result<Self> {
        let mut f = Fields::new("RetainedDerivedResult", value)?;
        let retention_id = prefixed("retention", f.req_str("retention_id")?)?;
        let transaction_id = prefixed("transaction", f.req_str("transaction_id")?)?;
        let receipt_digest = prefixed("receipt", f.req_str("receipt_digest")?)?;
        let manifest = f
            .opt("manifest")
            .map(|v| ObjectRef::from_value("retained manifest", v))
            .transpose()?;
        let released = match f.req("released")? {
            Value::Bool(value) => *value,
            _ => return refuse(Code::WRONG_TYPE, "retained result release must be boolean"),
        };
        let tensorfs = f
            .opt("tensorfs")
            .map(|value| {
                crate::canon::as_str("RetainedResult", "tensorfs", value).map(str::to_string)
            })
            .transpose()?;
        f.done_written_by(tensorfs.as_deref())?;
        if !released && manifest.is_none() {
            return refuse(Code::MISSING_FIELD, "held result has no manifest");
        }
        Ok(Self {
            tensorfs,
            retention_id,
            transaction_id,
            receipt_digest,
            manifest,
            released,
        })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("retention_id", Value::str(self.retention_id.clone())),
            ("transaction_id", Value::str(self.transaction_id.clone())),
            ("receipt_digest", Value::str(self.receipt_digest.clone())),
            ("released", Value::Bool(self.released)),
        ];
        if let Some(manifest) = &self.manifest {
            fields.push(("manifest", manifest.to_value()));
        }
        if let Some(tensorfs) = &self.tensorfs {
            fields.push(("tensorfs", Value::str(tensorfs.clone())));
        }
        Value::obj(fields)
    }
}

fn io(error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("retained derived result: {error}"),
    }
}

fn directory(store: &Store) -> PathBuf {
    store.root().join("roots/retained-derived")
}

fn path(store: &Store, id: &str) -> Result<PathBuf> {
    prefixed("retention", id)?;
    Ok(directory(store).join(format!("{}.json", &id[7..])))
}

fn read(store: &Store, id: &str) -> Result<Option<RetainedResult>> {
    let file = match File::open(path(store, id)?) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(e)),
    };
    let mut bytes = Vec::new();
    file.take(RetainedResult::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    let root = RetainedResult::parse(&bytes)?;
    if root.retention_id != id {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "retention path names another owner",
        );
    }
    Ok(Some(root))
}

fn write(store: &Store, root: &RetainedResult) -> Result<()> {
    let root = &RetainedResult {
        tensorfs: Some(crate::VERSION.into()),
        ..root.clone()
    };
    let path = path(store, &root.retention_id)?;
    fs::create_dir_all(directory(store)).map_err(io)?;
    let temp = path.with_extension(format!("tmp-{}", crate::meta::now_nanos_unique()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(io)?;
    file.write_all(&root.canonical_bytes())
        .and_then(|()| file.sync_all())
        .map_err(io)?;
    fs::rename(&temp, &path).map_err(io)?;
    for dir in [
        directory(store),
        store.root().join("roots"),
        store.root().to_path_buf(),
    ] {
        File::open(dir).and_then(|f| f.sync_all()).map_err(io)?;
    }
    Ok(())
}

fn identity(transaction: &str, receipt: &str, id: &str) -> Result<RetainedResult> {
    Ok(RetainedResult {
        tensorfs: Some(crate::VERSION.into()),
        retention_id: prefixed("retention", id)?,
        transaction_id: prefixed("transaction", transaction)?,
        receipt_digest: prefixed("receipt", receipt)?,
        manifest: None,
        released: false,
    })
}

fn check_identity(root: &RetainedResult, wanted: &RetainedResult) -> Result<()> {
    if root.transaction_id != wanted.transaction_id || root.receipt_digest != wanted.receipt_digest
    {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "retention already names a different result",
        );
    }
    Ok(())
}

fn verify_manifest(store: &Store, reference: &ObjectRef) -> Result<()> {
    let manifest = checkpoint::load_manifest(store, reference)?;
    let walk = checkpoint::walk_cozytensors(store, &manifest)?;
    walk.require_resident(store)?;
    for object in &walk.objects {
        if object.kind == "part" || object.kind == "config" {
            require_payload_record(store, &object.obj, object.kind)?;
        }
    }
    Ok(())
}

/// Acquire another durable root while the original exact result is still retained.
/// Verification consults native records, never reads or hashes tensor payloads.
pub fn retain_result(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    receipt: &str,
    id: &str,
) -> Result<RetainedResult> {
    let mut wanted = identity(transaction, receipt, id)?;
    let _gc = WriterGuard::acquire(store.root())?;
    let txn = meta.txn()?; // Serializes result disposal, release, and competing adoption.
    if let Some(root) = read(store, id)? {
        check_identity(&root, &wanted)?;
        if root.released {
            return refuse(
                Code::TRANSACTION_CLOSED,
                "retention was permanently released",
            );
        }
        verify_manifest(
            store,
            root.manifest.as_ref().expect("held root has manifest"),
        )?;
        return Ok(root);
    }
    let row = txn
        .rows
        .iter()
        .find(|row| row.id == transaction)
        .ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: "derived result is absent".into(),
        })?;
    if !matches!(row.state, TransactionState::Committed(_)) {
        return refuse(
            Code::TRANSACTION_CLOSED,
            "only a retained committed result can acquire another owner",
        );
    }
    let facts = receipt_from(store, row)?;
    if object_id(&crate::canon::write(&facts.to_value())) != receipt {
        return refuse(
            Code::OBJECT_ID_MISMATCH,
            "retained result receipt differs from committed computation",
        );
    }
    let source = roots::read(store, transaction)?;
    let original_held = matches!(
        row.state,
        TransactionState::Committed(Disposition::Pending | Disposition::Adopted(_))
    ) && source.as_ref().and_then(|root| root.manifest.as_ref())
        == Some(&facts.manifest);
    let independently_held = !original_held
        && all(store)?.iter().any(|root| {
            !root.released
                && root.transaction_id == transaction
                && root.receipt_digest == receipt
                && root.manifest.as_ref() == Some(&facts.manifest)
        });
    if !original_held && !independently_held {
        return refuse(
            if matches!(
                row.state,
                TransactionState::Committed(Disposition::Released)
            ) {
                Code::TRANSACTION_CLOSED
            } else {
                Code::DURABILITY_UNPROVEN
            },
            "result has no original or independent retained owner",
        );
    }
    verify_manifest(store, &facts.manifest)?;
    wanted.manifest = Some(facts.manifest);
    write(store, &wanted)?;
    Ok(wanted)
}

/// Release one owner. The durable tombstone fences an adoption arriving afterwards.
pub fn release_retention(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    receipt: &str,
    id: &str,
) -> Result<RetainedResult> {
    let wanted = identity(transaction, receipt, id)?;
    let _gc = WriterGuard::acquire(store.root())?;
    let _txn = meta.txn()?;
    let mut root = read(store, id)?.unwrap_or_else(|| wanted.clone());
    check_identity(&root, &wanted)?;
    if !root.released {
        root.released = true;
        write(store, &root)?;
    }
    Ok(root)
}

fn all(store: &Store) -> Result<Vec<RetainedResult>> {
    let entries = match fs::read_dir(directory(store)) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io(e)),
    };
    let mut roots = Vec::new();
    for entry in entries {
        let name = entry.map_err(io)?.file_name();
        let name = name.to_string_lossy();
        let Some(hex) = name.strip_suffix(".json") else {
            continue;
        };
        if let Some(root) = read(store, &format!("sha256:{hex}"))? {
            roots.push(root);
        }
    }
    Ok(roots)
}

pub(super) fn held_objects(store: &Store) -> Result<Vec<HeldKey>> {
    let mut held = Vec::new();
    for root in all(store)? {
        if !root.released {
            let manifest = root.manifest.expect("held root has manifest");
            held.push(HeldKey {
                key: crate::storage::manifest_key(&manifest.sha256)?,
                kind: "manifest".into(),
                length: manifest.length,
                sha256: manifest.sha256,
            });
        }
    }
    Ok(held)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::{self, tests::created_declaration};

    struct Fixture {
        root: PathBuf,
        store: Store,
        meta: Meta,
        transaction: String,
        receipt: String,
        manifest: ObjectRef,
        payload: ObjectRef,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn id(byte: &str) -> String {
        format!("sha256:{}", byte.repeat(32))
    }

    fn fixture() -> Fixture {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-retain-result-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let transaction = id("51");
        let writer =
            derived::begin(&store, &meta, &transaction, 1, created_declaration(), None).unwrap();
        let part = derived::add_part(
            &store,
            &meta,
            &transaction,
            1,
            ("model", "weight", "value"),
            &mut vec![0x11; 2048].as_slice(),
        )
        .unwrap();
        let result = derived::commit(&store, &meta, &transaction, 1).unwrap();
        writer.writer_hold.release(&meta).unwrap();
        let receipt = object_id(&crate::canon::write(&result.to_value()));
        Fixture {
            root,
            store,
            meta,
            transaction,
            receipt,
            manifest: result.manifest,
            payload: part.part.segments()[0].clone(),
        }
    }

    #[test]
    fn a_result_is_retained_under_its_receipt_and_nothing_else() {
        let f = fixture();
        retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("b1")).unwrap();
        assert_eq!(
            retain_result(&f.store, &f.meta, &f.transaction, &id("99"), &id("b3"))
                .unwrap_err()
                .code,
            Code::OBJECT_ID_MISMATCH
        );
    }

    #[test]
    fn independent_owners_keep_original_bytes_and_receipt_through_disposal_and_gc() {
        let f = fixture();
        let first =
            retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a1")).unwrap();
        let second =
            retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a2")).unwrap();
        assert_eq!(first.manifest, Some(f.manifest.clone()));
        assert_eq!(first.receipt_digest, f.receipt);
        let original = derived::dispose(&f.store, &f.meta, &f.transaction).unwrap();
        assert_eq!(
            object_id(&crate::canon::write(&original.receipt().to_value())),
            f.receipt
        );
        crate::gc::collect(&f.root, false).unwrap();
        assert!(f.store.contains(&f.payload.sha256));
        let reopened = Meta::open(&f.store).unwrap();
        assert_eq!(
            retain_result(&f.store, &reopened, &f.transaction, &f.receipt, &id("a1")).unwrap(),
            first
        );
        let released =
            release_retention(&f.store, &reopened, &f.transaction, &f.receipt, &id("a1")).unwrap();
        assert!(released.released);
        assert_eq!(
            release_retention(&f.store, &reopened, &f.transaction, &f.receipt, &id("a1")).unwrap(),
            released
        );
        crate::gc::collect(&f.root, false).unwrap();
        assert!(f.store.contains(&f.payload.sha256));
        assert_eq!(
            retain_result(&f.store, &reopened, &f.transaction, &f.receipt, &id("a1"))
                .unwrap_err()
                .code,
            Code::TRANSACTION_CLOSED
        );
        assert_eq!(
            retain_result(&f.store, &reopened, &f.transaction, &f.receipt, &id("a2")).unwrap(),
            second
        );
        release_retention(&f.store, &reopened, &f.transaction, &f.receipt, &id("a2")).unwrap();
        crate::gc::collect(&f.root, false).unwrap();
        assert!(!f.store.contains(&f.payload.sha256));
        assert!(!f.store.manifest_path(&f.manifest.sha256).exists());
        assert_eq!(
            retain_result(&f.store, &reopened, &f.transaction, &f.receipt, &id("a3"))
                .unwrap_err()
                .code,
            Code::TRANSACTION_CLOSED
        );
    }

    #[test]
    fn wrong_provenance_and_release_before_acquire_cannot_retain_a_result() {
        let f = fixture();
        assert_eq!(
            retain_result(&f.store, &f.meta, &f.transaction, &id("fe"), &id("a1"))
                .unwrap_err()
                .code,
            Code::OBJECT_ID_MISMATCH
        );
        let hold = retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a1")).unwrap();
        assert_eq!(
            release_retention(&f.store, &f.meta, &id("fe"), &f.receipt, &id("a1"))
                .unwrap_err()
                .code,
            Code::CROSS_SUBJECT_REPLAY
        );
        assert_eq!(read(&f.store, &id("a1")).unwrap(), Some(hold));
        let stopped =
            release_retention(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a2")).unwrap();
        assert!(stopped.released && stopped.manifest.is_none());
        let reopened = Meta::open(&f.store).unwrap();
        assert_eq!(
            retain_result(&f.store, &reopened, &f.transaction, &f.receipt, &id("a2"))
                .unwrap_err()
                .code,
            Code::TRANSACTION_CLOSED
        );
    }

    #[test]
    fn a_later_retry_acquires_from_an_independent_owner_after_original_cancellation() {
        let f = fixture();
        retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a1")).unwrap();
        derived::dispose(&f.store, &f.meta, &f.transaction).unwrap();
        crate::gc::collect(&f.root, false).unwrap();
        let second =
            retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a2")).unwrap();
        assert_eq!(second.manifest.as_ref(), Some(&f.manifest));
        release_retention(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a1")).unwrap();
        let reopened = Meta::open(&f.store).unwrap();
        let third =
            retain_result(&f.store, &reopened, &f.transaction, &f.receipt, &id("a3")).unwrap();
        assert_eq!(third.receipt_digest, f.receipt);
        release_retention(&f.store, &reopened, &f.transaction, &f.receipt, &id("a2")).unwrap();
        crate::gc::collect(&f.root, false).unwrap();
        assert!(f.store.contains(&f.payload.sha256));
        release_retention(&f.store, &reopened, &f.transaction, &f.receipt, &id("a3")).unwrap();
        assert_eq!(
            retain_result(&f.store, &reopened, &f.transaction, &f.receipt, &id("a4"))
                .unwrap_err()
                .code,
            Code::TRANSACTION_CLOSED
        );
        crate::gc::collect(&f.root, false).unwrap();
        assert!(!f.store.contains(&f.payload.sha256));
    }

    #[test]
    fn missing_or_corrupt_custody_is_never_claimed_as_reusable() {
        let f = fixture();
        retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a1")).unwrap();
        fs::remove_file(f.store.blob_path(&f.payload.sha256)).unwrap();
        fs::write(f.store.blob_path(&f.payload.sha256), vec![0x12; 2048]).unwrap();
        assert!(retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a1")).is_err());
        assert!(retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a2")).is_err());
        assert!(read(&f.store, &id("a2")).unwrap().is_none());
    }

    #[test]
    fn gc_discovers_retention_without_transaction_metadata() {
        let f = fixture();
        retain_result(&f.store, &f.meta, &f.transaction, &f.receipt, &id("a1")).unwrap();
        derived::dispose(&f.store, &f.meta, &f.transaction).unwrap();
        let database = crate::catalog::Catalog::path(&f.root);
        let connection = rusqlite::Connection::open(&database).unwrap();
        connection
            .execute(
                "UPDATE tensorfs_derived_transactions SET bytes=?1 WHERE id=?2",
                rusqlite::params![b"unreadable".as_slice(), &f.transaction],
            )
            .unwrap();
        drop(connection);
        crate::gc::collect(&f.root, false).unwrap();
        assert!(f.store.contains(&f.payload.sha256));
        f.store.read_manifest(&f.manifest).unwrap();
    }

    #[test]
    fn concurrent_original_disposal_cannot_acknowledge_unrooted_reuse() {
        for _ in 0..8 {
            let f = fixture();
            let start = std::sync::Arc::new(std::sync::Barrier::new(2));
            let root = f.root.clone();
            let transaction = f.transaction.clone();
            let receipt = f.receipt.clone();
            let begin = start.clone();
            let adopter = std::thread::spawn(move || {
                let store = Store::open(&root).unwrap();
                let meta = Meta::open(&store).unwrap();
                begin.wait();
                retain_result(&store, &meta, &transaction, &receipt, &id("a1"))
            });
            start.wait();
            derived::dispose(&f.store, &f.meta, &f.transaction).unwrap();
            let acquired = match adopter.join().unwrap() {
                Ok(root) => {
                    assert!(!root.released);
                    true
                }
                Err(error) => {
                    assert_eq!(error.code, Code::TRANSACTION_CLOSED);
                    false
                }
            };
            crate::gc::collect(&f.root, false).unwrap();
            assert_eq!(f.store.contains(&f.payload.sha256), acquired);
        }
    }
}
