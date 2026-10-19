//! Plain folders in and out of the store. A folder is a tree on disk: repo files at their paths,
//! `model.<variant>.cozytensors` headers, and `chunks/<sha256>` for the chunks and assets they
//! cite (thin: some may be in the store instead).

use std::fs::{self, File};
use std::io;
use std::path::Path;

use super::{fail, Error, Result, Store};
use crate::format::{check_file, variant_of, Header, ObjectRef, Tree, CHUNKS_DIR};

impl Store {
    /// Puts the tree document; its files need not be here.
    pub fn put_tree(&self, tree: &Tree) -> Result<ObjectRef> {
        self.put_bytes(&tree.to_json()?)
    }

    /// Admits the plain folder `dir`: every object a header cites (from `chunks/`, or already
    /// here), then the headers and repo files, then the tree, which it returns. Every byte is
    /// hashed once, here. A cited object in neither place refuses the folder.
    pub fn admit_folder(&self, dir: &Path) -> Result<ObjectRef> {
        let mut tree = Tree::default();
        for path in files(dir, "")? {
            let at = dir.join(&path);
            if variant_of(&path).is_some() {
                let bytes = fs::read(&at).map_err(fail("read", &at))?;
                for object in Header::from_cbor(&bytes)?.objects() {
                    let chunk = dir.join(CHUNKS_DIR).join(&object.sha256);
                    match File::open(&chunk) {
                        Ok(file) => drop(self.put(file, Some(object))?),
                        Err(e) if e.kind() == io::ErrorKind::NotFound => {
                            if !self.reuse(object)? {
                                return Err(Error::Absent(format!(
                                    "{} (cited by {path}): not in the folder or the store",
                                    object.sha256
                                )));
                            }
                        }
                        Err(e) => return Err(fail("open", &chunk)(e)),
                    }
                }
                tree.files.insert(path, self.put_bytes(&bytes)?);
            } else {
                let file = File::open(&at).map_err(fail("open", &at))?;
                tree.files.insert(path, self.put(file, None)?);
            }
        }
        self.put_tree(&tree)
    }

    /// The tree `tree` as a plain folder in `dir`: its repo files, the headers of `variants`
    /// (all when `None`), and those headers' objects this store holds, hard-linked when it can.
    /// Headers go last.
    pub fn snapshot(&self, tree: &ObjectRef, dir: &Path, variants: Option<&[String]>) -> Result<()> {
        let tree = self.tree(tree)?;
        let chunks = dir.join(CHUNKS_DIR);
        fs::create_dir_all(&chunks).map_err(fail("create", &chunks))?;
        for (path, object) in tree.repo_files() {
            self.materialize(object, &dir.join(path))?;
        }
        let chosen: Vec<(&str, &ObjectRef)> = (tree.variants())
            .filter(|(v, _)| variants.is_none_or(|want| want.iter().any(|w| w == v)))
            .collect();
        for (_, header) in &chosen {
            for object in Header::from_cbor(&self.read(header)?)?.objects() {
                if self.reuse(object)? {
                    self.materialize(object, &chunks.join(&object.sha256))?;
                }
            }
        }
        for (variant, header) in chosen {
            self.materialize(header, &dir.join(crate::format::variant_path(variant)))?;
        }
        Ok(())
    }

    /// `object` at `dest`, unless a file of its length is there.
    pub fn materialize(&self, object: &ObjectRef, dest: &Path) -> Result<()> {
        if fs::metadata(dest).is_ok_and(|m| m.len() == object.length) {
            return Ok(());
        }
        let parent = dest.parent().expect("a file has a directory");
        fs::create_dir_all(parent).map_err(fail("create", parent))?;
        let temp = parent.join(format!(".{}.tmp-{}", object.sha256, std::process::id()));
        let _ = fs::remove_file(&temp);
        let src = self.path(object)?;
        if fs::hard_link(&src, &temp).is_err() {
            io::copy(&mut self.file(object)?, &mut File::create(&temp).map_err(fail("create", &temp))?)
                .map_err(fail("copy", &temp))?;
        }
        fs::rename(&temp, dest).map_err(fail("rename into", dest))
    }
}

/// Every regular file under `dir`, as tree paths, but `chunks/` at the root.
fn files(dir: &Path, prefix: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(fail("list", dir))? {
        let entry = entry.map_err(fail("list", dir))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = match prefix {
            "" if name == CHUNKS_DIR => continue,
            "" => name,
            p => format!("{p}/{name}"),
        };
        let kind = entry.file_type().map_err(fail("stat", entry.path()))?;
        if kind.is_dir() {
            out.extend(files(&entry.path(), &path)?);
        } else if kind.is_file() {
            check_file(&path)?;
            out.push(path);
        }
    }
    Ok(out)
}
