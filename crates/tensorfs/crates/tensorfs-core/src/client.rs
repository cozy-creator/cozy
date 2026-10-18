//! The v2 library's front door, shared by the `tfs` CLI and Python: references, import into a
//! named ref, pull from a static origin, and plane regions.
//!
//! A reference is `org/name[@release]` (with a lane, else the release's only one) or a
//! manifest digest, `sha256:<hex>` or bare hex.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::failure::{fail, Kind, Result};
use crate::fetch::{self, policy, Client, Config, Fetcher, HttpOrigin, ObjectStore, Origin};
use crate::fetch::{SourcePolicy, VolumeOrigin, Want};
use crate::format::{repo_key, Body, Header, ObjectRef, Release, Repo};
use crate::import::{self, Access};
use crate::store2::Store;

/// The release an import refs.
pub const IMPORT_RELEASE: &str = "1";
/// The most bytes a refs document may have.
const REPO_BYTES: u64 = 16 << 20;

/// A checkpoint and the ref, release and lane that name it, when one does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Named {
    pub manifest: ObjectRef,
    pub name: Option<String>,
    pub release: Option<String>,
    pub lane: Option<String>,
}

enum Reference<'a> {
    Name(&'a str, Option<&'a str>),
    Digest(&'a str),
}

fn parse(reference: &str) -> Reference<'_> {
    let hex = reference.strip_prefix("sha256:").unwrap_or(reference);
    if hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Reference::Digest(hex);
    }
    match reference.split_once('@') {
        Some((name, release)) => Reference::Name(name, Some(release)),
        None => Reference::Name(reference, None),
    }
}

/// `repo` with `lane` of release `version` naming `manifest`. Releases keep `order`'s order
/// where it has them, so the latest stays last.
fn with_lane(
    repo: Option<Repo>,
    name: &str,
    version: &str,
    lane: &str,
    manifest: &ObjectRef,
    order: &[Release],
) -> Repo {
    let mut repo = repo.unwrap_or_else(|| Repo {
        name: name.to_owned(),
        releases: vec![],
    });
    match repo.releases.iter_mut().find(|r| r.version == version) {
        Some(release) => {
            release.lanes.insert(lane.to_owned(), manifest.clone());
        }
        None => repo.releases.push(Release {
            version: version.to_owned(),
            lanes: BTreeMap::from([(lane.to_owned(), manifest.clone())]),
        }),
    }
    let rank = |v: &str| {
        order
            .iter()
            .position(|r| r.version == v)
            .map_or(0, |p| p + 1)
    };
    repo.releases.sort_by_key(|r| rank(&r.version));
    repo
}

/// What [`Store::import_named`] does besides the sources.
pub struct ImportOptions<'a> {
    /// `org/name`, or a bare name under `local/`; by default the first source's name.
    pub name: Option<&'a str>,
    pub converter: Option<&'a str>,
    pub variant: Option<&'a str>,
    pub progress: &'a (dyn Fn(&str, u64, u64) + Sync),
}

/// Provider credentials from `HF_TOKEN` and `CIVITAI_TOKEN`.
pub fn access() -> Access {
    let bearer = |var: &str| {
        std::env::var(var)
            .ok()
            .filter(|t| !t.trim().is_empty())
            .map_or_else(String::new, |t| format!("bearer {}", t.trim()))
    };
    Access {
        huggingface_token: bearer("HF_TOKEN"),
        civitai_token: bearer("CIVITAI_TOKEN"),
        ..Access::default()
    }
}

/// `local/<the source's own name>`, in ref grammar.
fn default_name(source: &str) -> String {
    let source = source.trim().trim_end_matches('/');
    let stem = if let Some(rest) = source.strip_prefix("hf://") {
        rest.split('/')
            .nth(1)
            .unwrap_or(rest)
            .split('@')
            .next()
            .unwrap_or(rest)
            .to_owned()
    } else if let Some(rest) = source.strip_prefix("civitai://") {
        format!("civitai-{}", rest.replace('/', "-"))
    } else {
        let last = source
            .split(['/', '?'])
            .rfind(|s| !s.is_empty() && !s.contains('='));
        let last = last.unwrap_or("model");
        last.strip_suffix(".safetensors").unwrap_or(last).to_owned()
    };
    let clean: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '-'
            }
        })
        .collect();
    format!("local/{}", clean.trim_start_matches('.'))
}

/// The lowercase dtype holding most of the header's bytes (`f32`, `bf16`), or the scheme of
/// most encoded bytes.
fn main_lane(header: &Header) -> String {
    let mut bytes: BTreeMap<String, u64> = BTreeMap::new();
    for component in header.components.values() {
        for t in component.tensors.values() {
            let (name, length) = match &t.body {
                Body::Data(at) => (t.dtype.name().to_ascii_lowercase(), at.length),
                Body::Encoded { encoding, parts } => (
                    encoding.scheme.clone(),
                    parts.values().map(|p| p.length).sum(),
                ),
            };
            *bytes.entry(name).or_default() += length;
        }
    }
    bytes
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map_or_else(|| "files".to_owned(), |(name, _)| name)
}

impl Store {
    /// The checkpoint `reference` names here.
    pub fn resolve(&self, reference: &str, lane: Option<&str>) -> Result<Named> {
        match parse(reference) {
            Reference::Digest(sha256) => {
                let path = self.path(&ObjectRef {
                    sha256: sha256.to_owned(),
                    length: 0,
                })?;
                let length = match fs::metadata(&path) {
                    Ok(m) => m.len(),
                    Err(_) => return fail(Kind::NotFound, format!("no object {sha256} here")),
                };
                let manifest = ObjectRef {
                    sha256: sha256.to_owned(),
                    length,
                };
                for repo in self.refs()? {
                    for release in &repo.releases {
                        for (l, m) in &release.lanes {
                            if *m == manifest && lane.is_none_or(|want| want == l) {
                                return Ok(Named {
                                    manifest,
                                    name: Some(repo.name.clone()),
                                    release: Some(release.version.clone()),
                                    lane: Some(l.clone()),
                                });
                            }
                        }
                    }
                }
                Ok(Named {
                    manifest,
                    name: None,
                    release: None,
                    lane: None,
                })
            }
            Reference::Name(name, release) => {
                let Some(repo) = self.get_ref(name)? else {
                    return fail(
                        Kind::NotFound,
                        format!("no ref {name} in {}", self.root().display()),
                    );
                };
                let (version, lane, manifest) = fetch::choose(&repo, release, lane)?;
                Ok(Named {
                    manifest,
                    name: Some(repo.name),
                    release: Some(version),
                    lane: Some(lane),
                })
            }
        }
    }

    /// Imports `sources` into one checkpoint and refs it as release [`IMPORT_RELEASE`], lane
    /// its main dtype. Re-importing a name replaces that lane; an interrupted import resumes.
    pub fn import_named(&self, sources: &[String], opts: &ImportOptions) -> Result<Named> {
        let Some(first) = sources.first() else {
            return fail(Kind::Invalid, "import needs a source");
        };
        let name = match opts.name {
            Some(name) if name.contains('/') => name.to_owned(),
            Some(name) => format!("local/{name}"),
            None => default_name(first),
        };
        // A bad name fails here, before any download.
        self.get_ref(&name)?;
        let access = access();
        let req = import::Request {
            sources,
            converter: opts.converter,
            variant: opts.variant,
            access: &access,
            tmp: &self.root().join("tmp"),
            progress: opts.progress,
        };
        let key = format!("import.{}", name.replace('/', ".").to_ascii_lowercase());
        let manifest = self.import(&req, Some(&key))?;
        let named = self.set_lane(&name, IMPORT_RELEASE, None, &manifest)?;
        self.unpin(&key)?;
        Ok(named)
    }

    /// Refs `manifest` as `lane` (by default its main dtype) of `name`'s release `version`.
    pub fn set_lane(
        &self,
        name: &str,
        version: &str,
        lane: Option<&str>,
        manifest: &ObjectRef,
    ) -> Result<Named> {
        let lane = match lane {
            Some(lane) => lane.to_owned(),
            None => main_lane(self.checkpoint(manifest)?.header()),
        };
        let repo = self.update_ref(name, |old| {
            Some(with_lane(old, name, version, &lane, manifest, &[]))
        })?;
        Ok(Named {
            manifest: manifest.clone(),
            name: repo.map(|r| r.name),
            release: Some(version.to_owned()),
            lane: Some(lane),
        })
    }

    /// Pulls `reference` (`org/name[@release]`) from `origin`, an http(s) URL of a static origin
    /// or a directory in that layout, then refs it here as the origin does.
    pub fn pull(&self, origin: &str, reference: &str, lane: Option<&str>) -> Result<Pulled> {
        let Reference::Name(name, release) = parse(reference) else {
            return fail(
                Kind::Invalid,
                "a pull names org/name[@release], not a digest",
            );
        };
        let store: Arc<dyn ObjectStore> = Arc::new(self.clone());
        let (version, lane, manifest, repo, pulled) = fetch::runtime()?.block_on(async {
            let (origin, client, repo) = open(origin, name).await?;
            let (version, lane, manifest) = fetch::choose(&repo, release, lane)?;
            let fetcher = Fetcher::new(store, client, None, Config::default());
            let pulled = fetcher
                .pull_manifest(&origin, &manifest, &Want::default())
                .await?;
            fetcher.settled().await;
            Ok::<_, crate::failure::Failure>((version, lane, manifest, repo, pulled))
        })?;
        let local = self.update_ref(&repo.name, |old| {
            Some(with_lane(
                old,
                &repo.name,
                &version,
                &lane,
                &manifest,
                &repo.releases,
            ))
        })?;
        Ok(Pulled {
            named: Named {
                manifest,
                name: local.map(|r| r.name),
                release: Some(version),
                lane: Some(lane),
            },
            objects: pulled.objects,
            fetched: pulled.landed.origin + pulled.landed.volume,
            held: pulled.landed.local,
        })
    }
}

/// What a pull named, and its objects: those fetched (bytes) and those already here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pulled {
    pub named: Named,
    pub objects: usize,
    pub fetched: u64,
    pub held: u64,
}

/// The origin `spec` names and its refs document for `name`.
async fn open(spec: &str, name: &str) -> Result<(Arc<dyn Origin>, Client, Repo)> {
    let key = repo_key(name);
    let (origin, client, bytes): (Arc<dyn Origin>, Client, Vec<u8>) =
        if spec.starts_with("https://") || spec.starts_with("http://") {
            // The origin is the one the caller named; its objects are held to their digests
            // wherever a redirect sends them. Plaintext is for loopback only.
            let client = Client::new(SourcePolicy {
                any_host: true,
                allow_local: policy::is_loopback_origin(spec),
                max_redirects: 5,
                ..SourcePolicy::default()
            });
            let origin = HttpOrigin::new(&client, spec)?;
            let url = format!("{}/{key}", spec.trim_end_matches('/'));
            let bytes = client.document(&url, &[], REPO_BYTES).await?;
            (Arc::new(origin), client, bytes)
        } else {
            let dir = Path::new(spec.strip_prefix("file://").unwrap_or(spec));
            let bytes = match fs::read(dir.join(&key)) {
                Ok(bytes) => bytes,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return fail(
                        Kind::NotFound,
                        format!("no ref {name} at {}", dir.display()),
                    )
                }
                Err(e) => return fail(Kind::Io, format!("{}: {e}", dir.join(&key).display())),
            };
            let origin = VolumeOrigin::new(dir, Duration::from_secs(7 * 86400));
            (
                Arc::new(origin),
                Client::new(SourcePolicy::default()),
                bytes,
            )
        };
    Ok((origin, client, Repo::from_json(&bytes)?))
}

/// The origin layout `HttpOrigin` reads: `name`'s refs document gains `named`'s lane, and
/// `blobs/` every object it reaches. Objects are hard-linked when they can be.
pub fn export_origin(store: &Store, named: &Named, dir: &Path) -> Result<Vec<String>> {
    let (Some(name), Some(version), Some(lane)) = (&named.name, &named.release, &named.lane) else {
        return fail(
            Kind::NotFound,
            format!(
                "no ref names {}: an origin serves refs, export a named one",
                named.manifest.sha256
            ),
        );
    };
    let mut written = Vec::new();
    for object in store.closure(&named.manifest)? {
        let key = object.key();
        let dest = dir.join(&key);
        if fs::metadata(&dest).is_ok_and(|m| m.len() == object.length) {
            continue;
        }
        let src = store.path(&object)?;
        if !src.exists() {
            return fail(
                Kind::NotFound,
                format!("object {} is not here", object.sha256),
            );
        }
        let parent = dest.parent().expect("a blob has a shard");
        fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
        let temp = parent.join(format!(".{}.tmp-{}", object.sha256, std::process::id()));
        let _ = fs::remove_file(&temp);
        if fs::hard_link(&src, &temp).is_err() {
            fs::copy(&src, &temp).map_err(|e| io(&temp, e))?;
        }
        fs::rename(&temp, &dest).map_err(|e| io(&dest, e))?;
        written.push(key);
    }
    let path = dir.join(repo_key(name));
    let old = match fs::read(&path) {
        Ok(bytes) => Some(Repo::from_json(&bytes)?),
        Err(_) => None,
    };
    let order = store.get_ref(name)?.map(|r| r.releases).unwrap_or_default();
    let repo = with_lane(old, name, version, lane, &named.manifest, &order);
    let parent = path.parent().expect("a ref has a directory");
    fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    fs::write(&path, repo.to_json()?).map_err(|e| io(&path, e))?;
    written.push(repo_key(name));
    Ok(written)
}

pub(crate) fn io(path: &Path, e: std::io::Error) -> crate::failure::Failure {
    let mut failure = crate::failure::Failure::from(e);
    failure.detail = format!("{}: {}", path.display(), failure.detail);
    failure
}

/// `header`'s tensors grouped into regions for a weight plane, as `<component>/<key>` names in
/// chunk order. Tensors sharing a chunk stay together; neighbouring groups merge while their
/// chunks total at most `max_bytes`, and a group larger than that is a region alone. A region
/// never spans components.
pub fn regions(header: &Header, max_bytes: u64) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for (component, c) in &header.components {
        let name = |key: &str| format!("{component}/{key}");
        // Each tensor's first (chunk, offset) and last chunk; an empty tensor has neither.
        let (mut placed, mut empty) = (Vec::new(), Vec::new());
        for (key, t) in &c.tensors {
            let at: Vec<_> = match &t.body {
                Body::Data(at) => vec![*at],
                Body::Encoded { parts, .. } => parts.values().map(|p| p.at()).collect(),
            };
            let spans: Vec<_> = at.iter().flat_map(|a| header.spans(a)).collect();
            let first = spans.iter().map(|s| (s.chunk, s.start)).min();
            match (first, spans.iter().map(|s| s.chunk).max()) {
                (Some(first), Some(last)) => placed.push((first, last, key.as_str())),
                _ => empty.push(name(key)),
            }
        }
        placed.sort();
        // Runs of tensors over overlapping chunks: (keys, first chunk, last chunk).
        let mut groups: Vec<(Vec<&str>, usize, usize)> = Vec::new();
        for ((chunk, _), last, key) in placed {
            match groups.last_mut() {
                Some(g) if chunk <= g.2 => {
                    g.0.push(key);
                    g.2 = g.2.max(last);
                }
                _ => groups.push((vec![key], chunk, last)),
            }
        }
        let mut regions: Vec<(Vec<String>, u64)> = Vec::new();
        for (keys, lo, hi) in groups {
            let n: u64 = header.chunks[lo..=hi].iter().map(|c| c.length).sum();
            let keys = keys.into_iter().map(name);
            match regions.last_mut() {
                Some((r, bytes)) if *bytes + n <= max_bytes => {
                    r.extend(keys);
                    *bytes += n;
                }
                _ => regions.push((keys.collect(), n)),
            }
        }
        match regions.first_mut() {
            Some((r, _)) => r.extend(empty),
            None if !empty.is_empty() => regions.push((empty, 0)),
            None => {}
        }
        out.extend(regions.into_iter().map(|(r, _)| r));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::default_name;

    #[test]
    fn default_names() {
        assert_eq!(
            default_name("hf://acme/tiny-sdxl@abc/unet"),
            "local/tiny-sdxl"
        );
        assert_eq!(default_name("civitai://123/45"), "local/civitai-123-45");
        assert_eq!(default_name("/data/x/model.safetensors"), "local/model");
        assert_eq!(
            default_name("https://e.com/a/B c.safetensors?x=1"),
            "local/B-c"
        );
    }
}
