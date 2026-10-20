//! Retained file trees: a run's native inputs, a result's bytes, an uploaded package file. A
//! tree's document is the machine wire's (`{"entries": [{"kind": "file", "path", "blob":
//! {"sha256", "length"}}]}`, keys sorted, entries by path); its receipt is the document itself.
//! One hold in the run objects' store (`cas`), `tree-<owner>`, keeps the document and every
//! file until the owner releases it.
use std::{
    collections::BTreeMap,
    fs::File,
    io,
    path::{Component, Path},
};

use serde::Deserialize;
use serde_json::json;

use crate::cas::{Cas, Hold};
use tensorfs::format::ObjectRef;

/// A held tree: its hold's id, its document and the bytes of its files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tree {
    pub hold: String,
    pub manifest: ObjectRef,
    pub content: u64,
}

/// The hold that keeps `owner`'s tree.
fn id(owner: &str) -> String {
    format!("tree-{}", tensorfs::sha256::hex_digest(owner.as_bytes()))
}

/// A tree's document for `files`.
pub fn document(files: &BTreeMap<String, ObjectRef>) -> Vec<u8> {
    let entries: Vec<_> = (files.iter())
        .map(|(path, o)| json!({"blob": {"length": o.length, "sha256": o.sha256}, "kind": "file", "path": path}))
        .collect();
    crate::boundary_json::exact(&json!({ "entries": entries }))
}

/// A tree document's files, by path: ordinary files only, each path once and inside the tree.
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
        let inside = Path::new(&entry.path).components().all(|c| matches!(c, Component::Normal(_)));
        if !inside || entry.path.is_empty() || entry.path.contains(['\\', '\0']) {
            return Err(invalid(format!("tree path {:?} leaves its tree", entry.path)));
        }
        if files.insert(entry.path.clone(), entry.blob).is_some() {
            return Err(invalid(format!("a tree names {} twice", entry.path)));
        }
    }
    Ok(files)
}

/// Holds the tree of `files` (each already held) for `owner`. Idempotent per owner.
pub fn put(cas: &Cas, owner: &str, files: BTreeMap<String, ObjectRef>) -> io::Result<Tree> {
    let manifest = cas.put_bytes(&document(&files))?;
    hold(cas, owner, manifest, &files)
}

/// Holds `manifest`, a tree of `files` already held, for `owner`.
pub fn hold(cas: &Cas, owner: &str, manifest: ObjectRef, files: &BTreeMap<String, ObjectRef>) -> io::Result<Tree> {
    let mut roots = vec![manifest.clone()];
    for file in files.values() {
        if !roots.contains(file) {
            roots.push(file.clone());
        }
    }
    cas.hold(&id(owner), &Hold { owner: owner.to_string(), roots })?;
    Ok(Tree { hold: id(owner), manifest, content: files.values().map(|f| f.length).sum() })
}

/// Admits each `(path, local file, object)` not yet held, then holds their tree for `owner`.
pub fn import(cas: &Cas, owner: &str, files: &[(String, &Path, ObjectRef)]) -> io::Result<Tree> {
    admit(cas, files)?;
    put(cas, owner, files.iter().map(|(path, _, object)| (path.clone(), object.clone())).collect())
}

/// Admits each `(path, local file, object)` not held yet.
pub fn admit(cas: &Cas, files: &[(String, &Path, ObjectRef)]) -> io::Result<()> {
    for (_, local, object) in files {
        if !cas.reuse(object)? {
            cas.put(File::open(local)?, Some(object))?;
        }
    }
    Ok(())
}

/// `owner`'s held tree, if any: its document is the hold's first root.
pub fn read(cas: &Cas, owner: &str) -> io::Result<Option<Tree>> {
    let Some(manifest) = cas.held(&id(owner))?.and_then(|hold| hold.roots.into_iter().next()) else {
        return Ok(None);
    };
    let files = parse(&cas.read(&manifest)?)?;
    Ok(Some(Tree { hold: id(owner), manifest, content: files.values().map(|f| f.length).sum() }))
}

/// Drops `owner`'s hold: its files age from their last use.
pub fn release(cas: &Cas, owner: &str) -> io::Result<bool> {
    cas.release(&id(owner))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tree_is_held_until_released() {
        let root = std::env::temp_dir().join(format!("td-trees-{}", uuid::Uuid::new_v4()));
        let cas = Cas::new(root.join("objects"));
        std::fs::create_dir_all(&root).unwrap();
        let local = root.join("payload.bin");
        std::fs::write(&local, b"a result's bytes").unwrap();
        let object = ObjectRef::of(b"a result's bytes");
        let tree = import(&cas, "run-1/out", &[("payload".into(), local.as_path(), object.clone())]).unwrap();
        assert_eq!(read(&cas, "run-1/out").unwrap(), Some(tree.clone()));
        assert_eq!(parse(&cas.read(&tree.manifest).unwrap()).unwrap()["payload"], object);
        assert!(cas.sweep(std::time::Duration::ZERO, &mut |_| Ok(true)).unwrap().is_empty(), "a held tree keeps its files");
        assert!(release(&cas, "run-1/out").unwrap());
        assert_eq!(cas.sweep(std::time::Duration::ZERO, &mut |_| Ok(true)).unwrap().len(), 2);
        assert!(!cas.reuse(&object).unwrap() && !cas.reuse(&tree.manifest).unwrap());
        let _ = std::fs::remove_dir_all(root);
    }
}
