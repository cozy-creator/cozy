//! Sources: parse, pin, list and select their members, and download remote ones.
//!
//! `hf://<org>/<repo>[@<revision>][/<path>]`, `civitai://<version>[/<file id>]`, a
//! huggingface.co or civitai.com URL, any other https URL (one file), or a local file or
//! directory. A revision pins to its full commit; members carry the provider's sha256.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::err::{refuse, Code, Refusal, Result};
use crate::transport::{self, Deadline, Ledger, SourcePolicy};

/// Provider origins and credentials (`bearer <token>` or empty), each presented only to
/// its own host. Tests point the origins at a loopback server and set `allow_local`.
#[derive(Debug, Clone)]
pub struct Access {
    pub huggingface: String,
    pub civitai: String,
    pub huggingface_token: String,
    pub civitai_token: String,
    pub allow_local: bool,
}

impl Default for Access {
    fn default() -> Access {
        Access {
            huggingface: "https://huggingface.co".into(),
            civitai: "https://civitai.com".into(),
            huggingface_token: String::new(),
            civitai_token: String::new(),
            allow_local: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Spec {
    Hf {
        org: String,
        repo: String,
        revision: String,
        path: String,
    },
    Civitai {
        version: u64,
        file: Option<u64>,
    },
    Url(String),
    Local(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Hf,
    Civitai,
    Web,
}

#[derive(Debug, Clone)]
pub struct Member {
    /// Path in the source tree.
    pub path: String,
    pub at: At,
    pub length: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub enum At {
    Local(PathBuf),
    Remote { url: String, provider: Provider },
}

fn not<T>(why: impl Into<String>) -> Result<T> {
    refuse(Code::SOURCE_NOT_ALLOWED, why)
}

fn hex40(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

pub fn parse(raw: &str) -> Result<Spec> {
    let raw = raw.trim();
    if let Some(rest) = raw.strip_prefix("hf://") {
        let mut parts = rest.splitn(3, '/');
        let (Some(org), Some(named)) = (parts.next(), parts.next()) else {
            return not(format!("{raw}: hf://<org>/<repo>[@<revision>][/<path>]"));
        };
        let (repo, revision) = named.split_once('@').unwrap_or((named, "main"));
        let path = parts.next().unwrap_or("").trim_matches('/').to_string();
        return Ok(Spec::Hf {
            org: org.into(),
            repo: repo.into(),
            revision: revision.into(),
            path,
        });
    }
    if let Some(rest) = raw.strip_prefix("civitai://") {
        let (version, file) = rest.split_once('/').unwrap_or((rest, ""));
        let number = |s: &str| {
            s.parse::<u64>().map_err(|_| Refusal {
                code: Code::SOURCE_NOT_ALLOWED,
                detail: format!("{raw}: civitai://<version>[/<file id>]"),
            })
        };
        let file = if file.is_empty() {
            None
        } else {
            Some(number(file)?)
        };
        return Ok(Spec::Civitai {
            version: number(version)?,
            file,
        });
    }
    let Some(rest) = raw
        .strip_prefix("https://")
        .or_else(|| raw.strip_prefix("http://"))
    else {
        return Ok(Spec::Local(PathBuf::from(raw)));
    };
    let (host, rest) = rest.split_once('/').unwrap_or((rest, ""));
    let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    match host.to_ascii_lowercase().as_str() {
        "huggingface.co" | "hf.co" => match parts.as_slice() {
            [org, repo] => parse(&format!("hf://{org}/{repo}")),
            [org, repo, "tree" | "blob" | "resolve", revision, path @ ..] => {
                parse(&format!("hf://{org}/{repo}@{revision}/{}", path.join("/")))
            }
            _ => not(format!(
                "{raw}: a Hugging Face URL names a repository, a tree or a file"
            )),
        },
        "civitai.com" => {
            let version = query
                .split('&')
                .find_map(|q| q.strip_prefix("modelVersionId="));
            match (parts.as_slice(), version) {
                (["api", "download", "models", v], _) => parse(&format!("civitai://{v}")),
                (["models", ..], Some(v)) => parse(&format!("civitai://{v}")),
                _ => not(format!("{raw}: a Civitai URL names one model version")),
            }
        }
        _ => Ok(Spec::Url(raw.to_string())),
    }
}

impl Access {
    fn base(&self, p: Provider) -> &str {
        match p {
            Provider::Hf => self.huggingface.trim_end_matches('/'),
            Provider::Civitai => self.civitai.trim_end_matches('/'),
            Provider::Web => "",
        }
    }

    fn policy(&self, p: Provider) -> SourcePolicy {
        let mut hosts: Vec<String> = match p {
            Provider::Hf => vec!["huggingface.co".into(), ".huggingface.co".into(), "hf.co".into(), ".hf.co".into()],
            Provider::Civitai => vec![
                "civitai.com".into(),
                ".civitai.com".into(),
                "civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com".into(),
            ],
            Provider::Web => vec![],
        };
        if let Ok(host) = transport::base_host(self.base(p)) {
            hosts.push(host);
        }
        SourcePolicy {
            any_host: p == Provider::Web,
            allowed_hosts: hosts,
            allow_local: self.allow_local,
            max_redirects: 5,
            ..Default::default()
        }
    }

    fn credential(&self, p: Provider) -> Result<transport::ScopedHeaders> {
        let (token, hosts) = match p {
            Provider::Hf => (
                &self.huggingface_token,
                vec![transport::base_host(self.base(p))?],
            ),
            Provider::Civitai => (
                &self.civitai_token,
                vec![transport::base_host(self.base(p))?],
            ),
            Provider::Web => return transport::credential_from_spec("", vec![]),
        };
        transport::credential_from_spec(token, hosts)
    }

    fn get_json<T: for<'d> Deserialize<'d>>(
        &self,
        p: Provider,
        url: &str,
    ) -> Result<(T, Option<String>)> {
        let (body, next) = transport::api_get(
            url,
            &self.policy(p),
            &self.credential(p)?,
            32 << 20,
            Deadline::none(),
            &Ledger::new(),
        )?;
        let doc = serde_json::from_slice(&body).map_err(|e| Refusal {
            code: Code::MALFORMED_JSON,
            detail: format!("{url}: {e}"),
        })?;
        Ok((doc, next))
    }
}

#[derive(Deserialize)]
struct Revision {
    sha: String,
}

#[derive(Deserialize)]
struct TreeRow {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: Option<u64>,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    oid: String,
    size: u64,
}

#[derive(Deserialize)]
struct CivitaiVersion {
    files: Vec<CivitaiFile>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CivitaiFile {
    id: u64,
    name: String,
    #[serde(default)]
    primary: bool,
    download_url: Option<String>,
    #[serde(default)]
    hashes: BTreeMap<String, String>,
}

/// The source pinned (`hf://…@<commit>/<path>`), and every member it may select from.
/// `exact` is true when the source names one file.
pub struct Listing {
    pub pinned: String,
    pub root: String,
    pub exact: bool,
    pub members: Vec<Member>,
}

pub fn list(spec: &Spec, access: &Access) -> Result<Listing> {
    match spec {
        Spec::Hf {
            org,
            repo,
            revision,
            path,
        } => {
            let base = access.base(Provider::Hf);
            let commit = if hex40(revision) {
                revision.to_ascii_lowercase()
            } else {
                let url = format!("{base}/api/models/{org}/{repo}/revision/{revision}");
                let (r, _): (Revision, _) = access.get_json(Provider::Hf, &url)?;
                if !hex40(&r.sha) {
                    return refuse(
                        Code::MISSING_FIELD,
                        format!("hf://{org}/{repo}@{revision} names no commit"),
                    );
                }
                r.sha
            };
            let mut page = Some(format!(
                "{base}/api/models/{org}/{repo}/tree/{commit}?recursive=true"
            ));
            let mut members = Vec::new();
            while let Some(url) = page {
                let (rows, next): (Vec<TreeRow>, _) = access.get_json(Provider::Hf, &url)?;
                for row in rows.into_iter().filter(|r| r.kind == "file") {
                    if let Some(lfs) = &row.lfs {
                        if lfs.oid.bytes().all(|b| b == b'*') {
                            return refuse(
                                Code::CREDENTIAL_REQUIRED,
                                format!("{}: gated; supply a Hugging Face token", row.path),
                            );
                        }
                    }
                    members.push(Member {
                        at: At::Remote {
                            url: format!(
                                "{base}/{org}/{repo}/resolve/{commit}/{}",
                                row.path.replace(' ', "%20")
                            ),
                            provider: Provider::Hf,
                        },
                        length: row.lfs.as_ref().map(|l| l.size).or(row.size),
                        sha256: row.lfs.map(|l| l.oid),
                        path: row.path,
                    });
                }
                page = next;
            }
            let exact = members.iter().any(|m| &m.path == path);
            members.retain(|m| {
                path.is_empty() || &m.path == path || m.path.starts_with(&format!("{path}/"))
            });
            if members.is_empty() {
                return refuse(
                    Code::MISSING_TENSOR,
                    format!("hf://{org}/{repo}@{commit}/{path} holds no file"),
                );
            }
            let pinned = format!(
                "hf://{org}/{repo}@{commit}{}",
                if path.is_empty() {
                    String::new()
                } else {
                    format!("/{path}")
                }
            );
            Ok(Listing {
                pinned,
                root: if exact { String::new() } else { path.clone() },
                exact,
                members,
            })
        }
        Spec::Civitai { version, file } => {
            let base = access.base(Provider::Civitai);
            let (doc, _): (CivitaiVersion, _) = access.get_json(
                Provider::Civitai,
                &format!("{base}/api/v1/model-versions/{version}"),
            )?;
            let chosen: Vec<&CivitaiFile> = doc
                .files
                .iter()
                .filter(|f| match file {
                    Some(id) => f.id == *id,
                    None => f.primary && f.name.to_ascii_lowercase().ends_with(".safetensors"),
                })
                .collect();
            let [f] = chosen.as_slice() else {
                return refuse(
                    Code::MISSING_TENSOR,
                    format!(
                        "civitai://{version}: no single {} safetensors file",
                        if file.is_some() { "named" } else { "primary" }
                    ),
                );
            };
            if f.name.contains(['/', '\\']) || f.name.starts_with('.') {
                return not(format!("Civitai file name {:?}", f.name));
            }
            let url = f
                .download_url
                .clone()
                .unwrap_or_else(|| format!("{base}/api/download/models/{version}?fileId={}", f.id));
            let member = Member {
                path: f.name.clone(),
                at: At::Remote {
                    url,
                    provider: Provider::Civitai,
                },
                length: None,
                sha256: f.hashes.get("SHA256").map(|h| h.to_ascii_lowercase()),
            };
            Ok(Listing {
                pinned: format!("civitai://{version}/{}", f.id),
                root: String::new(),
                exact: true,
                members: vec![member],
            })
        }
        Spec::Url(url) => {
            let name = url
                .split(['?', '#'])
                .next()
                .unwrap_or("")
                .rsplit('/')
                .next()
                .unwrap_or("");
            if name.is_empty() || name.starts_with('.') {
                return not(format!("{url}: names no file"));
            }
            let member = Member {
                path: name.to_string(),
                at: At::Remote {
                    url: url.clone(),
                    provider: Provider::Web,
                },
                length: None,
                sha256: None,
            };
            Ok(Listing {
                pinned: url.clone(),
                root: String::new(),
                exact: true,
                members: vec![member],
            })
        }
        Spec::Local(path) => {
            let path = path
                .canonicalize()
                .map_err(|e| crate::store::classify_io(format!("open {}", path.display()), e))?;
            let mut members = Vec::new();
            let exact = path.is_file();
            if exact {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("source")
                    .to_string();
                members.push(Member {
                    path: name,
                    at: At::Local(path.clone()),
                    length: None,
                    sha256: None,
                });
            } else {
                walk(&path, &path, &mut members)?;
            }
            Ok(Listing {
                pinned: path.display().to_string(),
                root: String::new(),
                exact,
                members,
            })
        }
    }
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<Member>) -> Result<()> {
    let io = |e| crate::store::classify_io(format!("list {}", dir.display()), e);
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(io)?
        .collect::<std::io::Result<_>>()
        .map_err(io)?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            walk(root, &path, out)?;
        } else if let Ok(relative) = path.strip_prefix(root) {
            out.push(Member {
                path: relative.to_string_lossy().replace('\\', "/"),
                at: At::Local(path.clone()),
                length: None,
                sha256: None,
            });
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------ selection

/// Weights, and the documents a model is built from.
fn wanted(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".safetensors")
        || [".json", ".txt", ".model", ".jinja", ".tiktoken"]
            .iter()
            .any(|e| lower.ends_with(e))
        || lower.starts_with("license")
}

/// `Some(variant)` of a Diffusers/Transformers weights file or index:
/// `diffusion_pytorch_model.fp16-00001-of-00002.safetensors` and
/// `model.safetensors.index.fp16.json` are `Some(Some("fp16"))`; other files are `None`.
fn variant(path: &str) -> Option<Option<&str>> {
    let name = path.rsplit('/').next().unwrap_or(path);
    let (stem, variant) = match name.split_once(".safetensors.index") {
        Some((stem, rest)) => (stem, rest.strip_suffix(".json")?.strip_prefix('.')),
        None => {
            let stem = name.strip_suffix(".safetensors")?;
            let stem = match stem
                .rsplit_once("-of-")
                .and_then(|(head, _)| head.rsplit_once('-'))
            {
                Some((head, part)) if part.bytes().all(|b| b.is_ascii_digit()) => head,
                _ => stem,
            };
            match stem.split_once('.') {
                Some((stem, variant)) => (stem, Some(variant)),
                None => (stem, None),
            }
        }
    };
    ["model", "diffusion_pytorch_model", "pytorch_model"]
        .contains(&stem)
        .then_some(variant)
}

/// Keeps one weights variant per folder: `variant` if the folder has it, else the plain one,
/// else the folder's only variant.
fn one_variant(members: Vec<Member>, want: Option<&str>) -> Vec<Member> {
    let mut by_folder: BTreeMap<String, Vec<Option<String>>> = BTreeMap::new();
    for m in &members {
        if let Some(v) = variant(&m.path) {
            by_folder
                .entry(folder(&m.path).into())
                .or_default()
                .push(v.map(str::to_string));
        }
    }
    let pick: BTreeMap<String, Option<String>> = by_folder
        .into_iter()
        .map(|(f, mut vs)| {
            vs.sort();
            vs.dedup();
            let chosen = match want {
                Some(w) if vs.iter().any(|v| v.as_deref() == Some(w)) => Some(w.to_string()),
                _ if vs.contains(&None) => None,
                _ if vs.len() == 1 => vs[0].clone(),
                _ => None,
            };
            (f, chosen)
        })
        .collect();
    members
        .into_iter()
        .filter(|m| match variant(&m.path) {
            Some(v) => pick.get(folder(&m.path)).is_some_and(|p| p.as_deref() == v),
            None => true,
        })
        .collect()
}

fn folder(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(f, _)| f)
}

#[derive(Deserialize)]
struct ModularSpec {
    subfolder: Option<String>,
}

/// The component folders a Diffusers index names.
fn index_folders(doc: &[u8]) -> Result<Vec<String>> {
    let doc: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(doc).map_err(|e| Refusal {
            code: Code::MALFORMED_JSON,
            detail: format!("pipeline index: {e}"),
        })?;
    let mut folders = Vec::new();
    for (name, entry) in doc {
        let Some(row) = entry
            .as_array()
            .filter(|r| r.len() >= 2 && !name.starts_with('_'))
        else {
            continue;
        };
        let sub = row
            .get(2)
            .and_then(|s| serde_json::from_value::<ModularSpec>(s.clone()).ok())
            .and_then(|s| s.subfolder);
        folders.push(sub.unwrap_or(name));
    }
    Ok(folders)
}

/// What the import takes: the named file, else the layout under the root (a Diffusers
/// pipeline's indexed components, a Transformers model's root files, or every weights file
/// and document), one weights variant per folder.
pub fn select(
    listing: Listing,
    variant: Option<&str>,
    read: &dyn Fn(&Member) -> Result<Vec<u8>>,
) -> Result<Vec<Member>> {
    if listing.exact {
        return Ok(listing.members);
    }
    let root = listing.root.as_str();
    let under = |p: &str| {
        if root.is_empty() {
            p.to_string()
        } else {
            format!("{root}/{p}")
        }
    };
    let find = |p: &str| listing.members.iter().find(|m| m.path == under(p));
    let index = find("model_index.json")
        .or_else(|| find("modular_model_index.json"))
        .cloned();
    let keep: Box<dyn Fn(&str) -> bool> = match index {
        Some(index) => {
            let folders: Vec<String> = index_folders(&read(&index)?)?
                .into_iter()
                .map(|f| under(&f))
                .collect();
            Box::new(move |p: &str| {
                p == index.path
                    || folders
                        .iter()
                        .any(|f| folder(p) == f || p.starts_with(&format!("{f}/")))
            })
        }
        None if find("config.json").is_some()
            && listing
                .members
                .iter()
                .any(|m| folder(&m.path) == root && m.path.ends_with(".safetensors")) =>
        {
            let root = root.to_string();
            Box::new(move |p: &str| folder(p) == root)
        }
        None => Box::new(|_: &str| true),
    };
    let members = listing
        .members
        .into_iter()
        .filter(|m| wanted(&m.path) && keep(&m.path))
        .collect();
    Ok(one_variant(members, variant))
}

/// A document the model carries as a file: configs, tokenizers, templates, licenses.
pub fn document(path: &str) -> bool {
    wanted(path) && !path.contains(".safetensors")
}

// ------------------------------------------------------------------------------ download

/// Lands `m` at `dest`: resumes a partial file, then checks length and sha256 when the
/// provider states them. A local member is read where it is.
pub fn land(m: &Member, dest: &Path, access: &Access) -> Result<PathBuf> {
    let (url, provider) = match &m.at {
        At::Local(path) => return Ok(path.clone()),
        At::Remote { url, provider } => (url, *provider),
    };
    let io = |e| crate::store::classify_io(format!("write {}", dest.display()), e);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    let have = std::fs::metadata(dest).map(|md| md.len()).unwrap_or(0);
    if m.length != Some(have) || m.sha256.is_none() {
        let from = if m.length.is_some_and(|l| have < l) {
            have
        } else {
            0
        };
        let mut out: File = OpenOptions::new()
            .create(true)
            .write(true)
            .append(from > 0)
            .truncate(from == 0)
            .open(dest)
            .map_err(io)?;
        transport::get_into(
            url,
            &access.policy(provider),
            &access.credential(provider)?,
            from,
            None,
            &mut out,
            Deadline::none(),
            &Ledger::new(),
        )?;
        out.sync_all().map_err(io)?;
    }
    let len = std::fs::metadata(dest).map_err(io)?.len();
    if m.length.is_some_and(|l| l != len) {
        return refuse(
            Code::LENGTH_MISMATCH,
            format!(
                "{}: {len} bytes, the provider states {:?}",
                m.path, m.length
            ),
        );
    }
    if let Some(want) = &m.sha256 {
        let got = super::source::file_sha256(dest)?;
        if &got != want {
            std::fs::remove_file(dest).map_err(io)?;
            return refuse(
                Code::OBJECT_CORRUPT,
                format!("{}: sha256 {got}, the provider states {want}", m.path),
            );
        }
    }
    Ok(dest.to_path_buf())
}

/// Bytes `from..to` of a remote member.
pub fn read_range(m: &Member, from: u64, to: u64, access: &Access) -> Result<Vec<u8>> {
    let At::Remote { url, provider } = &m.at else {
        return refuse(
            Code::SOURCE_NOT_ALLOWED,
            format!("{} is not remote", m.path),
        );
    };
    let mut out = Vec::new();
    transport::get_into(
        url,
        &access.policy(*provider),
        &access.credential(*provider)?,
        from,
        Some(to),
        &mut out,
        Deadline::none(),
        &Ledger::new(),
    )?;
    Ok(out)
}

/// A document of `m`, whole (an index or a config, never weights).
pub fn read_small(m: &Member, access: &Access) -> Result<Vec<u8>> {
    match &m.at {
        At::Local(path) => std::fs::read(path)
            .map_err(|e| crate::store::classify_io(format!("read {}", path.display()), e)),
        At::Remote { url, provider } => Ok(transport::api_get(
            url,
            &access.policy(*provider),
            &access.credential(*provider)?,
            32 << 20,
            Deadline::none(),
            &Ledger::new(),
        )?
        .0),
    }
}
