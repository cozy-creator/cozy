//! `tfs conform endpoint`: a black-box check of an endpoint against `spec/distribution.md`, over
//! a fixture repo ([`FIXTURE`], release `1`) built the same way every time. With `write` it
//! uploads the fixture and checks the write side first; without, the fixture must be there (a
//! `write` run over the same storage seeds it). A case is a MUST or a SHOULD; buckets and static
//! servers fail some SHOULDs (no `428`, no closure check, no Range) by design.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use bytes::Bytes;

use base64::engine::{general_purpose::STANDARD, Engine as _};
use super::endpoint::{upload_tree, with_tree, Endpoint, Fresh, Payload};
use super::{Client, Config, Fetcher, ObjectStore, Origin, Want};
use crate::err::{Code, Refusal, Result};
use crate::format::{repo_key, Config as HeaderConfig, Dtype, ObjectRef, Repo, Tree};
use crate::sha256::hex_digest;
use crate::store::Store;
use crate::writer::{Data, Source};

pub const FIXTURE: &str = "conformance/endpoint";

/// One case's outcome.
#[derive(Debug, Clone)]
pub struct Case {
    pub name: &'static str,
    pub must: bool,
    /// `None` when the case does not apply (a directory has no HTTP statuses).
    pub ok: Option<bool>,
    pub detail: String,
}

struct Cases(Vec<Case>);

impl Cases {
    fn record(&mut self, name: &'static str, must: bool, outcome: Option<std::result::Result<(), String>>) -> bool {
        let (ok, detail) = match outcome {
            None => (None, "n/a".to_owned()),
            Some(Ok(())) => (Some(true), String::new()),
            Some(Err(why)) => (Some(false), why),
        };
        self.0.push(Case { name, must, ok, detail });
        ok != Some(false)
    }
}

fn expect(cond: bool, why: impl FnOnce() -> String) -> std::result::Result<(), String> {
    if cond {
        Ok(())
    } else {
        Err(why())
    }
}

/// The fixture in a store under `scratch`: a one-component cozytensors as variant `int8` of a
/// tree with a README. The tree.
fn fixture(scratch: &Path) -> Result<(Store, ObjectRef)> {
    let failed = crate::store::Error::refusal;
    let store = Store::open(scratch.join("fixture")).map_err(failed)?;
    let mut w = store.writer(None, None).map_err(failed)?;
    let bytes: Vec<u8> = (0..300_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8).collect();
    let config: HeaderConfig = serde_json::from_str(r#"{"conformance":1}"#).unwrap();
    let mut c = w.component("model", Default::default(), config, None).map_err(failed)?;
    c.put("w", &Dtype::U8, &[bytes.len() as u64], Data::Plain(Source::Read(&mut &bytes[..]))).map_err(failed)?;
    let header = w.commit().map_err(failed)?;
    let mut tree = Tree::default();
    tree.set_variant("int8", header);
    tree.files.insert("README.md".into(), store.put_bytes(b"# cozytensors distribution conformance\n").map_err(failed)?);
    let tree = store.put_tree(&tree).map_err(failed)?;
    Ok((store, tree))
}

/// Runs the suite at `endpoint`, building the fixture under `scratch`.
pub async fn endpoint(endpoint: Arc<Endpoint>, client: &Client, write: bool, scratch: &Path) -> Result<Vec<Case>> {
    let (store, tree) = fixture(scratch)?;
    let mut cases = Cases(Vec::new());
    let nonce = hex_digest(format!("{:?}", SystemTime::now()).as_bytes());
    if write && !write_side(&endpoint, &store, &tree, &nonce, &mut cases).await? {
        return Ok(cases.0);
    }
    read_side(&endpoint, client, &store, &tree, &nonce, scratch, &mut cases).await?;
    Ok(cases.0)
}

async fn write_side(endpoint: &Arc<Endpoint>, store: &Store, tree: &ObjectRef, nonce: &str, cases: &mut Cases) -> Result<bool> {
    let up = upload_tree(endpoint, FIXTURE, tree, store, Some("1")).await;
    let up = up.map(drop).map_err(|e| e.to_string());
    if !cases.record("upload: HEAD 404 → PUT → HEAD 200, then the ref", true, Some(up)) {
        return Ok(false);
    }

    let claimed = ObjectRef::of(format!("conformance {nonce}").as_bytes());
    let wrong = Bytes::from(vec![b'x'; claimed.length as usize]);
    let sent = endpoint.upload(&claimed, &move || Ok(Payload::Bytes(wrong.clone()))).await;
    let held = endpoint.has(&claimed).await?;
    let outcome = expect(sent.is_err() && !held, || format!("answered {sent:?}, holds it: {held}"));
    cases.record("upload: wrong bytes are refused and not stored", true, Some(outcome));

    let readme = Tree::from_json(&store.read(tree).map_err(crate::store::Error::refusal)?)
        .map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: e.to_string() })?
        .files["README.md"]
        .clone();
    let bytes = Bytes::from(store.read(&readme).map_err(crate::store::Error::refusal)?);
    let checksum = STANDARD.encode(crate::sha256::digest(&bytes));
    let headers = vec![
        ("content-length".to_owned(), bytes.len().to_string()),
        ("x-amz-checksum-sha256".to_owned(), checksum),
        ("if-none-match".to_owned(), "*".to_owned()),
    ];
    let again = endpoint.raw("PUT", &readme.key(), headers, Some(bytes)).await?;
    let outcome = again.map(|(status, ..)| expect(matches!(status, 200 | 201 | 204 | 412), || format!("answered {status}")));
    cases.record("upload: a repeated PUT is idempotent", true, outcome);

    let Fresh::Doc(doc, etag) = endpoint.repo_bytes(FIXTURE, None).await? else {
        cases.record("refs: the uploaded document reads back", true, Some(Err("absent".into())));
        return Ok(false);
    };
    let created = endpoint.put_repo_bytes(FIXTURE, doc.clone(), None).await;
    cases.record("refs: a second create is 412", true, Some(expect(matches!(created, Ok(false)), || format!("{created:?}"))));
    let stale = endpoint.put_repo_bytes(FIXTURE, doc.clone(), Some("\"conformance-stale\"")).await;
    cases.record("refs: a stale If-Match is 412", true, Some(expect(matches!(stale, Ok(false)), || format!("{stale:?}"))));
    let json = vec![("content-type".to_owned(), "application/json".to_owned())];
    let bare = endpoint.raw("PUT", &repo_key(FIXTURE), json.clone(), Some(Bytes::from(doc.clone()))).await?;
    let outcome = bare.map(|(status, ..)| expect(status == 428, || format!("answered {status}")));
    cases.record("refs: no precondition is 428 (servers)", false, outcome);

    // A document citing a tree nobody uploaded; restored at once wherever it was taken.
    let missing = ObjectRef::of(format!("absent tree {nonce}").as_bytes());
    let mut cited = Repo::from_json(&doc).map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: e.to_string() })?;
    cited.checkpoints.push(missing.clone());
    let cited = cited.to_json().map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: e.to_string() })?;
    let mut matched = json.clone();
    matched.push(("if-match".to_owned(), etag.clone()));
    let refused = endpoint.raw("PUT", &repo_key(FIXTURE), matched, Some(Bytes::from(cited))).await?;
    let outcome = refused.map(|(status, _, body)| {
        let body = String::from_utf8_lossy(&body);
        expect(status == 409 && body.contains("closure_incomplete") && body.contains(&missing.sha256), || format!("answered {status} {body}"))
    });
    cases.record("refs: an incomplete closure is 409 closure_incomplete (servers)", false, outcome);
    endpoint.update_repo(FIXTURE, |old| Ok(old.map(|mut r| (r.checkpoints.retain(|c| *c != missing), r).1))).await?;

    let mut trees = Vec::new();
    for i in 0..8 {
        let readme = store.put_bytes(format!("writer {i} {nonce}").as_bytes()).map_err(crate::store::Error::refusal)?;
        let t = store.put_tree(&Tree { files: BTreeMap::from([("README.md".to_owned(), readme.clone())]) });
        let t = t.map_err(crate::store::Error::refusal)?;
        for object in [&readme, &t] {
            let (from, at) = (store.clone(), object.clone());
            endpoint.upload(object, &move || from.file(&at).map(Payload::File).map_err(crate::store::Error::refusal)).await?;
        }
        trees.push(t);
    }
    let writers = trees.iter().cloned().map(|t| {
        let endpoint = Arc::clone(endpoint);
        tokio::spawn(async move { endpoint.update_repo(FIXTURE, |old| Ok(Some(with_tree(old, FIXTURE, &t, None)))).await })
    });
    let landed = futures::future::join_all(writers).await;
    let repo = endpoint.repo(FIXTURE).await?.map(|(r, _)| r).unwrap_or_default();
    let all = trees.iter().all(|t| repo.checkpoints.contains(t));
    let outcome = expect(all && landed.iter().all(|l| matches!(l, Ok(Ok(_)))), || format!("{landed:?}"));
    cases.record("refs: 8 concurrent writers all land through the 412 loop", true, Some(outcome));
    endpoint.update_repo(FIXTURE, |old| Ok(old.map(|mut r| (r.checkpoints.retain(|c| !trees.contains(c)), r).1))).await?;
    Ok(true)
}

async fn read_side(
    endpoint: &Arc<Endpoint>,
    client: &Client,
    store: &Store,
    tree: &ObjectRef,
    nonce: &str,
    scratch: &Path,
    cases: &mut Cases,
) -> Result<()> {
    let fixture = endpoint.repo(FIXTURE).await?;
    let read = match &fixture {
        Some((repo, _)) => expect(repo.release(Some("1")).is_some_and(|r| r.tree == *tree), || format!("release 1 is not the fixture: {repo:?}")),
        None => Err(format!("{FIXTURE} is absent: seed it with a --write run over the same storage")),
    };
    if !cases.record("refs: GET is 200 with the document", true, Some(read)) {
        return Ok(());
    }
    let got = endpoint.raw("GET", &repo_key(FIXTURE), vec![], None).await?;
    let etag = got.as_ref().and_then(|(_, headers, _)| headers.get("etag")?.to_str().ok().map(str::to_owned));
    cases.record("refs: an ETag", false, got.as_ref().map(|_| expect(etag.is_some(), || "none sent".into())));
    let fresh = match &etag {
        Some(etag) => endpoint.raw("GET", &repo_key(FIXTURE), vec![("if-none-match".into(), etag.clone())], None).await?,
        None => None,
    };
    let outcome = fresh.map(|(status, ..)| expect(status == 304, || format!("answered {status}")));
    let unasked = (got.is_some() && etag.is_none()).then(|| Err("no ETag to ask with".into()));
    cases.record("refs: If-None-Match is 304", false, outcome.or(unasked));
    let absent = endpoint.repo_bytes(&format!("conformance/absent-{}", &nonce[..12]), None).await?;
    cases.record("refs: an absent repo is 404", true, Some(expect(absent == Fresh::Absent, || format!("{absent:?}"))));

    // Down through the Fetcher: redirects followed, every object checked.
    let into = Store::open(scratch.join("pulled")).map_err(crate::store::Error::refusal)?;
    let fetcher = Fetcher::new(Arc::new(into.clone()) as Arc<dyn ObjectStore>, client.clone(), None, Config::default());
    let origin: Arc<dyn Origin> = endpoint.clone();
    let pulled = fetcher.pull(&origin, FIXTURE, Some("1"), Some("int8"), &Want::default()).await;
    fetcher.settled().await;
    let outcome = pulled.map(|p| p.objects).map_err(|e| e.to_string()).and_then(|n| expect(n >= 4, || format!("{n} objects")));
    cases.record("blobs: GET returns each object, verified", true, Some(outcome));

    let header = Tree::from_json(&store.read(tree).map_err(crate::store::Error::refusal)?)
        .map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: e.to_string() })?
        .header("int8")
        .cloned()
        .expect("the fixture's variant");
    let head = endpoint.raw("HEAD", &header.key(), vec![], None).await?;
    let outcome = head.map(|(status, headers, _)| {
        let length = headers.get("content-length").and_then(|v| v.to_str().ok()?.parse::<u64>().ok());
        expect(status == 200 && length == Some(header.length), || format!("answered {status}, length {length:?}"))
    });
    cases.record("blobs: HEAD is 200 with the length", true, outcome);
    let last = header.length - 1;
    let ranged = endpoint.raw("GET", &header.key(), vec![("range".into(), format!("bytes={last}-{last}"))], None).await?;
    let expected = store.read(&header).map_err(crate::store::Error::refusal)?[last as usize];
    let outcome = ranged.map(|(status, _, body)| match status {
        300..=399 => Ok(()),
        _ => expect(status == 206 && body == [expected], || format!("answered {status} with {} bytes", body.len())),
    });
    cases.record("blobs: Range gives 206 with that byte (or redirects)", false, outcome);
    let unknown = ObjectRef { sha256: hex_digest(nonce.as_bytes()), length: 1 };
    let held = endpoint.has(&unknown).await;
    cases.record("blobs: an unknown object is 404", true, Some(expect(matches!(held, Ok(false)), || format!("{held:?}"))));
    Ok(())
}
