//! A commit sets files on top of its tag's tree (git's rule, as Hugging Face's `upload_folder`):
//! `Writer(repo_id, revision=)` and `upload_folder` share that one path, over a real directory
//! endpoint and the local store's own.
#![cfg(feature = "store")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tensorfs::format::{fnmatch, repo_key, ClassRef, Dtype, ObjectRef, Repo, Tree};
use tensorfs::{Client, Config, Download, Endpoint, Format, Kind, Open, Ref, WriterOptions};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("commits-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn client(cache: &Path, endpoint: &Path) -> Client {
    let mut config = Config::from_env();
    config.cache_dir = cache.to_path_buf();
    config.endpoint = endpoint.to_str().unwrap().to_owned();
    config.mirror = None;
    config.token = None;
    Client::new(config).unwrap()
}

/// Writes `weight` as tensor `w` of `variant`, committed onto `repo@tag`.
async fn write(client: &Client, repo: &str, tag: &str, variant: &str, weight: &[u8]) -> ObjectRef {
    let options = WriterOptions { repo_id: Some(repo), revision: Some(tag), variant: Some(variant), ..WriterOptions::default() };
    let mut writer = client.writer(options).unwrap();
    (writer.component("model", ClassRef::new("test", "1.0", "Model"), Default::default(), None).unwrap())
        .put("w", Dtype::U8, &[weight.len() as u64], &mut std::io::Cursor::new(weight))
        .unwrap();
    writer.commit().await.unwrap().tree().unwrap().clone()
}

/// A plain folder holding `model.<variant>.cozytensors` of `weight` (none: no variant) and the
/// repo `files`.
async fn folder(client: &Client, at: &Path, variant: Option<&str>, weight: &[u8], files: &[(&str, &[u8])]) -> PathBuf {
    if let Some(variant) = variant {
        let mut writer = client.writer(WriterOptions::default()).unwrap();
        (writer.component("model", ClassRef::new("test", "1.0", "Model"), Default::default(), None).unwrap())
            .put("w", Dtype::U8, &[weight.len() as u64], &mut std::io::Cursor::new(weight))
            .unwrap();
        writer.commit().await.unwrap().materialize(at, Format::Cozytensors, false).await.unwrap();
        fs::rename(at.join("model.cozytensors"), at.join(format!("model.{variant}.cozytensors"))).unwrap();
    }
    fs::create_dir_all(at).unwrap();
    for (path, bytes) in files {
        fs::write(at.join(path), bytes).unwrap();
    }
    at.to_path_buf()
}

/// The tree tag `tag` of `repo` names in the store or endpoint at `root`.
fn tree(root: &Path, repo: &str, tag: &str) -> Tree {
    let repo = Repo::from_json(&fs::read(root.join(repo_key(repo))).unwrap()).unwrap();
    let commit = repo.tag(tag).unwrap().commit.clone();
    Tree::from_json(&fs::read(root.join(commit.key())).unwrap()).unwrap()
}

fn paths(tree: &Tree) -> Vec<&str> {
    tree.files.keys().map(String::as_str).collect()
}

/// The bytes of tensor `w` at `reference`, from a fresh store.
async fn read(cache: &Path, endpoint: &Path, reference: &str) -> Vec<u8> {
    let checkpoint = client(cache, endpoint).open(reference, &Open::default()).await.unwrap();
    checkpoint.download(&Download::default()).await.unwrap();
    checkpoint.component("model").unwrap().read_decoded("w").unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn writer_variants_accumulate_on_a_tag_and_publish_as_one_release() {
    let scratch = Scratch::new();
    let endpoint = scratch.0.join("endpoint");
    let client = client(&scratch.0.join("cache"), &endpoint);
    let bf16 = write(&client, "acme/two", "main", "bf16", b"bf16").await;
    let main = write(&client, "acme/two", "main", "int8", b"int8").await;
    assert_ne!(bf16, main);
    assert_eq!(paths(&tree(&endpoint, "acme/two", "main")), ["model.bf16.cozytensors", "model.int8.cozytensors"]);

    // Publishing main as 0.2.0 names that commit with a public tag.
    let remote = Endpoint::new(endpoint.to_str().unwrap(), None).unwrap();
    let published = remote.update_repo("acme/two", |old| {
        let mut repo = old.unwrap();
        repo.set_tag("0.2.0", repo.tag("main").unwrap().commit.clone());
        repo.tag_mut("0.2.0").unwrap().public = true;
        Ok(Some(repo))
    });
    published.await.unwrap();
    for (variant, weight) in [("bf16", b"bf16"), ("int8", b"int8")] {
        assert_eq!(read(&scratch.0.join(format!("r-{variant}")), &endpoint, &format!("acme/two@0.2.0/{variant}")).await, weight);
    }
    let latest = client.open("acme/two", &Open { variant: Some("int8".into()), ..Open::default() }).await.unwrap();
    assert_eq!(latest.tree(), Some(&main), "the default is 0.2.0, main's commit");
}

#[tokio::test(flavor = "current_thread")]
async fn an_upload_keeps_the_tags_other_files_and_deletes_only_by_pattern() {
    let scratch = Scratch::new();
    let endpoint = scratch.0.join("endpoint");
    let cache = scratch.0.join("cache");
    let client = client(&cache, &endpoint);
    let a = folder(&client, &scratch.0.join("a"), Some("bf16"), b"bf16", &[("README.md", b"a")]).await;
    let b = folder(&client, &scratch.0.join("b"), Some("int8"), b"int8", &[]).await;
    let c = folder(&client, &scratch.0.join("c"), None, b"", &[("README.md", b"c"), ("LICENSE", b"MIT")]).await;
    for repo in ["acme/up", "local/up"] {
        let root = if repo.starts_with("local/") { &cache } else { &endpoint };
        let main: Ref = format!("{repo}@main").parse().unwrap();
        client.upload_folder(&main, &a, &[], None).await.unwrap();
        client.upload_folder(&main, &b, &[], None).await.unwrap();
        let both = tree(root, repo, "main");
        assert_eq!(paths(&both), ["README.md", "model.bf16.cozytensors", "model.int8.cozytensors"], "{repo}");

        // Deletes come only from patterns; a file this commit sets stays, replaced.
        client.upload_folder(&main, &c, &["*.int8.cozytensors", "README*"], None).await.unwrap();
        let after = tree(root, repo, "main");
        assert_eq!(paths(&after), ["LICENSE", "README.md", "model.bf16.cozytensors"], "{repo}");
        assert_eq!(after.files["README.md"], ObjectRef::of(b"c"));
        assert_eq!(after.files["model.bf16.cozytensors"], both.files["model.bf16.cozytensors"]);
    }
    let bf16 = read(&scratch.0.join("reader"), &endpoint, "acme/up@main/bf16").await;
    assert_eq!(bf16, b"bf16");
    let gone = client.open("acme/up@main/int8", &Open::default()).await.unwrap_err();
    assert_eq!(gone.kind, Kind::NotFound, "{}", gone.detail);
}

/// `delete_patterns` match as Python's `fnmatch` does, the way Hugging Face matches them.
#[test]
fn delete_patterns_match_as_fnmatch() {
    for (pattern, path, hit) in [
        ("*.int8.cozytensors", "model.int8.cozytensors", true),
        ("*.int8.cozytensors", "sub/model.int8.cozytensors", true),
        ("*.int8.cozytensors", "model.bf16.cozytensors", false),
        ("model.?.cozytensors", "model.a.cozytensors", true),
        ("README", "README.md", false),
        ("README*", "README.md", true),
        ("[a-c]*", "b.txt", true),
        ("[!m]*", "model.cozytensors", false),
        ("docs/", "docs/a.md", true),
        ("[", "[", true),
        ("*", "", true),
        ("?", "", false),
    ] {
        assert_eq!(fnmatch(pattern, path), hit, "{pattern} {path}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_commits_to_one_tag_all_land() {
    let scratch = Scratch::new();
    let endpoint = scratch.0.join("endpoint");
    let make = client(&scratch.0.join("maker"), &endpoint);
    let mut folders = Vec::new();
    for i in 0..6 {
        let at = scratch.0.join(format!("f{i}"));
        folders.push(folder(&make, &at, Some(&format!("u{i}")), format!("up{i}").as_bytes(), &[]).await);
    }
    let mut work = Vec::new();
    for (i, at) in folders.into_iter().enumerate() {
        let (cache, endpoint) = (scratch.0.join(format!("c{i}")), endpoint.clone());
        work.push(tokio::spawn(async move {
            let main: Ref = "acme/many@main".parse().unwrap();
            client(&cache, &endpoint).upload_folder(&main, &at, &[], None).await.unwrap();
        }));
    }
    for i in 0..2 {
        let (cache, endpoint) = (scratch.0.join(format!("w{i}")), endpoint.clone());
        work.push(tokio::spawn(async move {
            write(&client(&cache, &endpoint), "acme/many", "main", &format!("w{i}"), b"ww").await;
        }));
    }
    for job in work {
        job.await.unwrap();
    }
    let want: Vec<String> = (0..6).map(|i| format!("model.u{i}.cozytensors")).chain((0..2).map(|i| format!("model.w{i}.cozytensors"))).collect();
    let mut want: Vec<&str> = want.iter().map(String::as_str).collect();
    want.sort();
    assert_eq!(paths(&tree(&endpoint, "acme/many", "main")), want);
}
