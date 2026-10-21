//! The repo documents: `cozytensors.tree/1`, a commit's file tree, and `cozytensors.repo/1`, a
//! model's tags naming commits. Both parse through a `Wire` twin that carries the `format` tag.

use std::collections::{BTreeMap, HashSet};
use std::str::FromStr;

use pep440_rs::Version;

use serde::{Deserialize, Serialize};

use super::doc::{is_digest, path, segment};
use super::{check_format, invalid, json, variant_of, variant_path, Error, ObjectRef, Result};
use super::{REPO_FORMAT, TREE_FORMAT};

/// The folder that holds a plain folder's chunks and assets: no repo file lives under it.
pub const CHUNKS_DIR: &str = "chunks";

/// `cozytensors.tree/1`: a commit, a repo state as files, as on HF. Each `model.<variant>.cozytensors` is
/// a variant's header; every other file (README, LICENSE, images) is a blob shared by them all.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(try_from = "TreeWire", into = "TreeWire")]
pub struct Tree {
    pub files: BTreeMap<String, ObjectRef>,
}

#[derive(Serialize, Deserialize)]
struct TreeWire {
    format: String,
    files: BTreeMap<String, ObjectRef>,
}

impl TryFrom<TreeWire> for Tree {
    type Error = Error;
    fn try_from(w: TreeWire) -> Result<Self> {
        check_format(&w.format, TREE_FORMAT)?;
        for p in w.files.keys() {
            check_file(p)?;
        }
        Ok(Tree { files: w.files })
    }
}

impl From<Tree> for TreeWire {
    fn from(t: Tree) -> Self {
        TreeWire {
            format: TREE_FORMAT.into(),
            files: t.files,
        }
    }
}

/// A repo file path: plain segments, not under [`CHUNKS_DIR`].
pub(crate) fn check_file(p: &str) -> Result<()> {
    if !path(p) || p.split('/').next() == Some(CHUNKS_DIR) {
        return invalid(format!("file path {p:?} is not a repo file path"));
    }
    Ok(())
}

impl Tree {
    pub fn from_json(bytes: &[u8]) -> Result<Tree> {
        json::read(bytes, TREE_FORMAT)
    }
    pub fn to_json(&self) -> Result<Vec<u8>> {
        json::write(self)
    }

    /// Each variant (`""` for `model.cozytensors`) and its header, by variant name.
    pub fn variants(&self) -> impl Iterator<Item = (&str, &ObjectRef)> {
        (self.files.iter()).filter_map(|(p, r)| Some((variant_of(p)?, r)))
    }

    /// The header of `variant`.
    pub fn header(&self, variant: &str) -> Option<&ObjectRef> {
        self.files.get(&variant_path(variant))
    }

    /// `variant`, or with `None` the tree's only one: (its name, its header).
    pub fn variant(&self, variant: Option<&str>) -> Result<(&str, &ObjectRef)> {
        let found = match variant {
            Some(v) => self.files.get_key_value(&variant_path(v)).map(|(p, r)| (variant_of(p), r)),
            None => match self.variants().collect::<Vec<_>>()[..] {
                [(v, r)] => Some((Some(v), r)),
                _ => None,
            },
        };
        match found {
            Some((Some(v), r)) => Ok((v, r)),
            _ => {
                let has: Vec<&str> = self.variants().map(|(v, _)| v).collect();
                let want = variant.map_or("its only variant".into(), |v| format!("variant {v:?}"));
                invalid(format!("the tree has no {want} (has {has:?})"))
            }
        }
    }

    /// The repo files: everything but the variants' headers.
    pub fn repo_files(&self) -> impl Iterator<Item = (&str, &ObjectRef)> {
        (self.files.iter())
            .filter(|(p, _)| variant_of(p).is_none())
            .map(|(p, r)| (p.as_str(), r))
    }

    /// Sets `variant`'s header, replacing the one it had.
    pub fn set_variant(&mut self, variant: &str, header: ObjectRef) {
        self.files.insert(variant_path(variant), header);
    }

    /// A commit on top of this tree, as Hugging Face's `upload_folder`: the paths a `delete`
    /// pattern matches ([`fnmatch`]) removed, then `files` added or replaced.
    pub fn commit(&self, files: &BTreeMap<String, ObjectRef>, delete: &[&str]) -> Tree {
        let mut files_kept = self.files.clone();
        files_kept.retain(|path, _| !delete.iter().any(|pattern| fnmatch(pattern, path)));
        files_kept.extend(files.iter().map(|(p, r)| (p.clone(), r.clone())));
        Tree { files: files_kept }
    }
}

/// Python's `fnmatch` (Hugging Face's `delete_patterns`): `*` any run of characters, `/`
/// included; `?` one; `[set]`, `[!set]` and ranges one of a set; a pattern ending in `/` matches
/// everything under it.
pub fn fnmatch(pattern: &str, path: &str) -> bool {
    let mut p: Vec<char> = pattern.chars().collect();
    if pattern.ends_with('/') {
        p.push('*');
    }
    let s: Vec<char> = path.chars().collect();
    let (mut pi, mut si, mut star) = (0, 0, None);
    while si < s.len() {
        if p.get(pi) == Some(&'*') {
            star = Some((pi, si));
            pi += 1;
        } else if let Some(next) = (pi < p.len()).then(|| one(&p, pi, s[si])).flatten() {
            (pi, si) = (next, si + 1);
        } else if let Some((at, from)) = star {
            (pi, si) = (at + 1, from + 1);
            star = Some((at, from + 1));
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

/// Where the token of `p` at `at` ends when it matches `c`.
fn one(p: &[char], at: usize, c: char) -> Option<usize> {
    let set = (p[at] == '[').then(|| {
        let negate = p.get(at + 1) == Some(&'!');
        let first = at + 1 + negate as usize;
        let close = (first + 1..p.len()).find(|&i| p[i] == ']')?;
        let mut hit = false;
        let mut i = first;
        while i < close {
            if i + 2 < close && p[i + 1] == '-' {
                hit |= p[i] <= c && c <= p[i + 2];
                i += 3;
            } else {
                hit |= p[i] == c;
                i += 1;
            }
        }
        Some((hit != negate, close + 1))
    });
    match set.flatten() {
        Some((true, next)) => Some(next),
        Some((false, _)) => None,
        None => (p[at] == '?' || p[at] == c).then_some(at + 1),
    }
}

/// `refs/<org>/<name>.json`, ASCII-lowercased: repo names are case-insensitive, one identity.
pub fn repo_key(name: &str) -> String {
    format!("refs/{}.json", name.to_ascii_lowercase())
}

/// `cozytensors.repo/1`: a model's tags, each naming a commit (a tree), in creation order. Kept at
/// [`repo_key`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "RepoWire", into = "RepoWire")]
pub struct Repo {
    pub name: String,
    pub tags: Vec<Tag>,
}

/// A movable name for a commit. A public tag is a release named by a PEP 440 version; a private
/// one is the owner's. A yanked tag is skipped by an unversioned lookup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tag {
    pub name: String,
    pub commit: ObjectRef,
    #[serde(default, skip_serializing_if = "is_false")]
    pub public: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub yanked: bool,
}

fn is_false(b: &bool) -> bool {
    !b
}

#[derive(Serialize, Deserialize)]
struct RepoWire {
    format: String,
    name: String,
    tags: Vec<Tag>,
}

impl TryFrom<RepoWire> for Repo {
    type Error = Error;
    fn try_from(w: RepoWire) -> Result<Self> {
        check_format(&w.format, REPO_FORMAT)?;
        let repo = Repo { name: w.name, tags: w.tags };
        repo.check()?;
        Ok(repo)
    }
}

impl From<Repo> for RepoWire {
    fn from(r: Repo) -> Self {
        RepoWire { format: REPO_FORMAT.into(), name: r.name, tags: r.tags }
    }
}

/// The one "newest" ordering (spec §7): of `items`, the one whose `version` is the highest final
/// PEP 440 version (a post-release is final), else the highest pre-release (`a`, `b`, `rc`,
/// `.dev`); of equal versions (`1.0`, `1.0.0`), the last given. Items whose version does not parse
/// are skipped; callers filter anything else first.
pub fn newest_by<T>(items: impl IntoIterator<Item = T>, version: impl Fn(&T) -> &str) -> Option<T> {
    let versioned = items.into_iter().filter_map(|item| Some((Version::from_str(version(&item)).ok()?, item)));
    let order = |(a, _): &(Version, T), (b, _): &(Version, T)| a.is_stable().cmp(&b.is_stable()).then_with(|| a.cmp(b));
    versioned.max_by(order).map(|(_, item)| item)
}

/// The unversioned default among `tags`: [`newest_by`] of the public, non-yanked ones, in
/// creation order.
pub fn newest<'a>(tags: impl IntoIterator<Item = &'a Tag>) -> Option<&'a Tag> {
    newest_by(tags.into_iter().filter(|t| t.public && !t.yanked), |t| t.name.as_str())
}

/// A repository name: `<org>/<name>`, two segments.
pub(crate) fn repo_name(name: &str) -> bool {
    name.split_once('/').is_some_and(|(org, name)| segment(org) && segment(name))
}

/// A tag name: a segment that cannot be read as a digest.
pub fn tag_name(name: &str) -> bool {
    segment(name) && !name.starts_with("sha256:")
}

/// What a revision names (§7), before any tree is read.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution<'r> {
    /// A tag, or the commit of one: resolved.
    Tag(&'r Tag),
    /// A variant header (its SHA-256): the commit is the first of these tags' whose tree cites it.
    Header(String, Vec<&'r Tag>),
}

impl<'r> Resolution<'r> {
    /// The tag and, for a header, the variant that names it, reading each candidate's tree with
    /// `tree`. `None` when no candidate cites the header.
    pub fn find<E>(self, mut tree: impl FnMut(&ObjectRef) -> std::result::Result<Tree, E>) -> std::result::Result<Option<(&'r Tag, Option<String>)>, E> {
        let (sha256, tags) = match self {
            Resolution::Tag(tag) => return Ok(Some((tag, None))),
            Resolution::Header(sha256, tags) => (sha256, tags),
        };
        for tag in tags {
            if let Some(variant) = tree(&tag.commit)?.cites(&sha256) {
                return Ok(Some((tag, Some(variant.to_owned()))));
            }
        }
        Ok(None)
    }
}

impl Tree {
    /// The variant whose header has SHA-256 `sha256`.
    pub fn cites(&self, sha256: &str) -> Option<&str> {
        self.variants().find(|(_, h)| h.sha256 == sha256).map(|(v, _)| v)
    }
}

impl Repo {
    pub fn new(name: &str) -> Repo {
        Repo { name: name.into(), tags: vec![] }
    }
    pub fn from_json(bytes: &[u8]) -> Result<Repo> {
        json::read(bytes, REPO_FORMAT)
    }
    /// The canonical bytes; refuses a document a reader would refuse.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        self.check()?;
        json::write(self)
    }
    /// The repo's key in a store or static origin: [`repo_key`] of its name.
    pub fn key(&self) -> String {
        repo_key(&self.name)
    }

    fn check(&self) -> Result<()> {
        if !repo_name(&self.name) {
            return invalid(format!("repo name {:?} is not <org>/<name>", self.name));
        }
        let mut seen = HashSet::new();
        for t in &self.tags {
            if !tag_name(&t.name) || !seen.insert(&t.name) {
                return invalid(format!("tag {:?} is not a segment, starts with sha256: or repeats", t.name));
            }
            if t.public && Version::from_str(&t.name).is_err() {
                return invalid(format!("public tag {:?} is not a PEP 440 version", t.name));
            }
        }
        Ok(())
    }

    /// Tag `name`, yanked or not.
    pub fn tag(&self, name: &str) -> Option<&Tag> {
        self.tags.iter().find(|t| t.name == name)
    }

    pub fn tag_mut(&mut self, name: &str) -> Option<&mut Tag> {
        self.tags.iter_mut().find(|t| t.name == name)
    }

    /// The unversioned default: [`newest`] of the tags.
    pub fn default_tag(&self) -> Option<&Tag> {
        newest(&self.tags)
    }

    /// Resolves `revision` once (§7): none is the default; `sha256:<hex>` a tag's commit, else a
    /// variant header; anything else a tag name. `None` when it names nothing here.
    pub fn resolve(&self, revision: Option<&str>) -> Option<Resolution<'_>> {
        let Some(revision) = revision else {
            return self.default_tag().map(Resolution::Tag);
        };
        let Some(sha256) = revision.strip_prefix("sha256:") else {
            return self.tag(revision).map(Resolution::Tag);
        };
        if !is_digest(sha256) {
            return None;
        }
        // The default's commit first, then the last created.
        let mut order: Vec<&Tag> = self.default_tag().into_iter().collect();
        order.extend(self.tags.iter().rev());
        if let Some(tag) = order.iter().find(|t| t.commit.sha256 == sha256) {
            return Some(Resolution::Tag(tag));
        }
        let mut commits = HashSet::new();
        order.retain(|t| commits.insert(&t.commit));
        Some(Resolution::Header(sha256.to_owned(), order))
    }

    /// Moves tag `name` to `commit`, keeping its place and flags; a new tag is private and last.
    pub fn set_tag(&mut self, name: &str, commit: ObjectRef) {
        match self.tag_mut(name) {
            Some(t) => t.commit = commit,
            None => self.tags.push(Tag { name: name.into(), commit, public: false, yanked: false }),
        }
    }

    /// Every commit the repo keeps alive: each tag's, yanked or private too.
    pub fn roots(&self) -> impl Iterator<Item = &ObjectRef> {
        self.tags.iter().map(|t| &t.commit)
    }
}
