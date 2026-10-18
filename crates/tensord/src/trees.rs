//! Retained file trees: a run's native inputs, a result's bytes, an uploaded package file. A
//! tree's document is the machine wire's (`{"entries": [{"kind": "file", "path", "blob":
//! {"sha256", "length"}}]}`, keys sorted, entries by path); its receipt is the document itself.
//! One pin, `tree-<owner>`, keeps the document and every file until the owner releases it.
use std::{collections::BTreeMap, fs::File, io, path::Path};

use serde::Deserialize;
use serde_json::json;

use crate::storage::{self, pin, Store};
use tensorfs_core::format::ObjectRef;

/// A pinned tree: its pin, its document and the bytes of its files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tree {
    pub pin: String,
    pub manifest: ObjectRef,
    pub content: u64,
}

/// A tree's document for `files`.
pub fn document(files: &BTreeMap<String, ObjectRef>) -> Vec<u8> {
    let entries: Vec<_> = (files.iter())
        .map(|(path, o)| json!({"blob": {"length": o.length, "sha256": o.sha256}, "kind": "file", "path": path}))
        .collect();
    crate::boundary_json::exact(&json!({ "entries": entries }))
}

/// A tree document's files, by path: ordinary files only, each path once.
pub fn parse(bytes: &[u8]) -> io::Result<BTreeMap<String, ObjectRef>> {
    #[derive(Deserialize)]
    struct Document {
        entries: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        kind: String,
        path: String,
        blob: ObjectRef,
    }
    let invalid = |why: String| io::Error::new(io::ErrorKind::InvalidInput, why);
    let document: Document = serde_json::from_slice(bytes).map_err(|e| invalid(format!("a tree document: {e}")))?;
    let mut files = BTreeMap::new();
    for entry in document.entries {
        if entry.kind != "file" {
            return Err(invalid(format!("a tree holds files, not {:?}", entry.kind)));
        }
        if files.insert(entry.path.clone(), entry.blob).is_some() {
            return Err(invalid(format!("a tree names {} twice", entry.path)));
        }
    }
    Ok(files)
}

/// Pins the tree of `files` (each already in the store) for `owner`. Idempotent per owner.
pub fn put(store: &Store, owner: &str, files: BTreeMap<String, ObjectRef>) -> io::Result<Tree> {
    let manifest = store.put_bytes(&document(&files)).map_err(storage::io_error)?;
    hold(store, owner, manifest, &files)
}

/// Pins `manifest`, a tree of `files` already in the store, for `owner`.
pub fn hold(store: &Store, owner: &str, manifest: ObjectRef, files: &BTreeMap<String, ObjectRef>) -> io::Result<Tree> {
    let id = pin::tree(owner);
    let mut roots = vec![manifest.clone()];
    for file in files.values() {
        if !roots.contains(file) {
            roots.push(file.clone());
        }
    }
    storage::hold(store, &id, roots, Some(owner.to_string()))?;
    Ok(Tree { pin: id, manifest, content: files.values().map(|f| f.length).sum() })
}

/// Admits each `(path, local file, object)` not yet held, then pins their tree for `owner`.
pub fn import(store: &Store, owner: &str, files: &[(String, &Path, ObjectRef)]) -> io::Result<Tree> {
    admit(store, files)?;
    put(store, owner, files.iter().map(|(path, _, object)| (path.clone(), object.clone())).collect())
}

/// Admits each `(path, local file, object)` the store does not hold yet.
pub fn admit(store: &Store, files: &[(String, &Path, ObjectRef)]) -> io::Result<()> {
    for (_, local, object) in files {
        if !store.reuse(object).map_err(storage::io_error)? {
            store.put(File::open(local)?, Some(object)).map_err(storage::io_error)?;
        }
    }
    Ok(())
}

/// `owner`'s pinned tree, if any: its document is the pin's first root.
pub fn read(store: &Store, owner: &str) -> io::Result<Option<Tree>> {
    let held = store.get_pin(&pin::tree(owner)).map_err(storage::io_error)?;
    let Some(manifest) = held.and_then(|pin| pin.roots.into_iter().next()) else {
        return Ok(None);
    };
    let files = parse(&store.read(&manifest).map_err(storage::io_error)?)?;
    Ok(Some(Tree { pin: pin::tree(owner), manifest, content: files.values().map(|f| f.length).sum() }))
}

/// Drops `owner`'s pin: its files become the GC's once nothing else holds them.
pub fn release(store: &Store, owner: &str) -> io::Result<bool> {
    store.unpin(&pin::tree(owner)).map_err(storage::io_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tree_is_pinned_until_released() {
        let root = std::env::temp_dir().join(format!("td-trees-{}", uuid::Uuid::new_v4()));
        let store = storage::open(&root).unwrap().as_ref().clone().with_grace(std::time::Duration::ZERO);
        let local = root.join("payload.bin");
        std::fs::write(&local, b"a result's bytes").unwrap();
        let object = ObjectRef::of(b"a result's bytes");
        let tree = import(&store, "run-1/out", &[("payload".into(), local.as_path(), object.clone())]).unwrap();
        assert_eq!(read(&store, "run-1/out").unwrap(), Some(tree.clone()));
        assert_eq!(parse(&store.read(&tree.manifest).unwrap()).unwrap()["payload"], object);
        store.gc().unwrap();
        assert!(store.reuse(&object).unwrap(), "a pinned tree keeps its files");
        assert!(release(&store, "run-1/out").unwrap());
        store.gc().unwrap();
        assert!(!store.reuse(&object).unwrap() && !store.reuse(&tree.manifest).unwrap());
        let _ = std::fs::remove_dir_all(root);
    }
}
