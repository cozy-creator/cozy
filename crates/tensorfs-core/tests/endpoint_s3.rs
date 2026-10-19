//! An S3 bucket as an endpoint (`s3://`, SigV4): the MinIO that tensorhub's dev-build
//! `cozy_ref` job (`scripts/live-hub-client.sh`) runs on 127.0.0.1:9000 with bucket `repo`.
//! Skipped where that job's TENSORHUB_ORIGIN is unset. One test: it sets the `AWS_*` locations
//! and credentials an `s3://` endpoint reads.

use std::collections::BTreeMap;
use std::sync::Arc;

use tensorfs_core::err::Code;
use tensorfs_core::fetch::{
    upload_tree, with_tree, Client, Config, Endpoint, Fetcher, Fresh, ObjectStore, Origin, Payload, SourcePolicy, Want,
};
use tensorfs_core::format::{Config, Dtype, ObjectRef, Tree};
use tensorfs_core::store2::Store;
use tensorfs_core::writer::{Data, Source};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bucket_serves_the_protocol_under_sigv4() {
    if std::env::var("TENSORHUB_ORIGIN").is_err() {
        eprintln!("skipped: no MinIO (tensorhub's cozy_ref job runs one)");
        return;
    }
    for (var, value) in [
        ("AWS_ENDPOINT_URL", "http://127.0.0.1:9000"),
        ("AWS_ACCESS_KEY_ID", "proof"),
        ("AWS_SECRET_ACCESS_KEY", "proof-secret"),
        ("AWS_REGION", "us-east-1"),
    ] {
        std::env::set_var(var, value);
    }
    let scratch = std::env::temp_dir().join(format!("tfs-endpoint-s3-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let client = Client::new(SourcePolicy { any_host: true, allow_local: true, max_redirects: 5, ..Default::default() });
    let endpoint = Arc::new(Endpoint::new(&client, &format!("s3://repo/tfs-{}", std::process::id()), None).unwrap());

    let source = Store::open(scratch.join("source")).unwrap();
    let mut w = source.writer(None, None).unwrap();
    let big: Vec<u8> = (0..20u32 << 20).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
    let config: Config = serde_json::from_str(r#"{"layers":1}"#).unwrap();
    let mut unet = w.component("unet", "diffusers.Thing", config, None).unwrap();
    unet.put("w", &Dtype::U8, &[big.len() as u64], Data::Plain(Source::Read(&mut &big[..]))).unwrap();
    unet.put("b", &Dtype::U8, &[3], Data::Plain(Source::Read(&mut &b"abc"[..]))).unwrap();
    let header = w.commit().unwrap();
    let mut tree = Tree::default();
    tree.set_variant("int8", header.clone());
    tree.files.insert("README.md".into(), source.put_bytes(b"# s3").unwrap());
    let tree = source.put_tree(&tree).unwrap();

    let up = upload_tree(&endpoint, "acme/s3", &tree, &source, Some("1.0")).await.unwrap();
    assert_eq!(up.sent, up.objects);
    assert_eq!(upload_tree(&endpoint, "acme/s3", &tree, &source, Some("1.0")).await.unwrap().sent, 0);
    let (_, etag) = endpoint.repo("acme/s3").await.unwrap().unwrap();
    assert_eq!(endpoint.repo_since("acme/s3", Some(&etag)).await.unwrap(), Fresh::Unchanged, "a 304");

    // The bucket checks the digest: wrong bytes are refused and nothing is stored.
    let claimed = ObjectRef::of(b"what was promised");
    let wrong = endpoint.upload(&claimed, &|| Ok(Payload::Bytes(b"something else!!!"[..].into()))).await;
    assert!(wrong.is_err() && !endpoint.has(&claimed).await.unwrap(), "{wrong:?}");

    // Eight writers race conditional PUTs (If-Match): every checkpoint lands.
    let mut trees = Vec::new();
    for i in 0..8u8 {
        let readme = source.put_bytes(&[b's', i]).unwrap();
        let t = source.put_tree(&Tree { files: BTreeMap::from([("README.md".to_owned(), readme)]) }).unwrap();
        upload_tree(&endpoint, "acme/s3-scratch", &t, &source, None).await.unwrap();
        trees.push(t);
    }
    let writers = trees.iter().cloned().map(|t| {
        let endpoint = Arc::clone(&endpoint);
        tokio::spawn(async move { endpoint.update_repo("acme/s3", |old| Ok(Some(with_tree(old, "acme/s3", &t, None)))).await })
    });
    for writer in futures::future::join_all(writers).await {
        writer.unwrap().unwrap();
    }
    let (repo, _) = endpoint.repo("acme/s3").await.unwrap().unwrap();
    let mut landed = repo.checkpoints.clone();
    (landed.sort(), trees.sort());
    assert_eq!(landed, trees);
    let stale = endpoint.put_repo_bytes("acme/s3", b"{}".to_vec(), Some("\"0123\"")).await.unwrap();
    assert!(!stale, "a stale If-Match is a 412");

    // And down again, through the Fetcher, into a fresh store.
    let into = Store::open(scratch.join("into")).unwrap();
    let fetcher = Fetcher::new(Arc::new(into.clone()) as Arc<dyn ObjectStore>, client.clone(), None, Config::default());
    let origin: Arc<dyn Origin> = endpoint.clone();
    let pulled = fetcher.pull(&origin, "acme/s3", Some("1.0"), Some("int8"), &Want::default()).await.unwrap();
    fetcher.settled().await;
    assert_eq!(pulled.header, header);
    assert!(into.checkpoint(&header).unwrap().read("unet", "w").unwrap() == big);
    let refused = Endpoint::new(&client, "s3://", None).err().map(|e| e.code);
    assert_eq!(refused, Some(Code::SOURCE_NOT_ALLOWED));
    let _ = std::fs::remove_dir_all(&scratch);

    // The conformance suite, through the CLI: a bucket fails only the server-side SHOULDs.
    let url = format!("s3://repo/tfs-conform-{}", std::process::id());
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_tfs")).args(["conform", "endpoint", &url, "--write", "--json"]).output().unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(out.status.success() && doc["failed"] == 0, "{doc:#}");
    let warned: Vec<&str> = (doc["cases"].as_array().unwrap().iter()).filter(|c| c["ok"] == false).filter_map(|c| c["case"].as_str()).collect();
    assert_eq!(warned, ["refs: no precondition is 428 (servers)", "refs: an incomplete closure is 409 closure_incomplete (servers)"]);
}
