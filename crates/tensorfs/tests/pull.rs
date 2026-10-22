//! Pulls through the real Fetcher, client and downloader from real loopback origins:
//! `python -m http.server`, and a server that injects faults.

mod pull_support;

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use pull_support::*;
use tensorfs::err::Code;
use tensorfs::fetch::{
    Config, Fetcher, Endpoint, ObjectStore, Origin, Progress, VolumeOrigin, Want,
};
use tensorfs::store::Store;

const SAMPLE: Duration = Duration::from_millis(10);

fn config() -> Config {
    Config {
        tick: SAMPLE,
        ..Config::default()
    }
}

fn fetcher(store: &Arc<Store>, volume: Option<VolumeOrigin>) -> Fetcher {
    Fetcher::new(
        Arc::clone(store) as Arc<dyn ObjectStore>,
        client(),
        volume,
        config(),
    )
}

fn http(base: &str) -> Arc<dyn Origin> {
    Arc::new(Endpoint::new(&client(), base, None).unwrap().anonymous())
}

fn all() -> Want<'static> {
    Want::default()
}

/// `tfs pull` from a static origin served by `python -m http.server` (HTTP/1.0, no ranges):
/// the store holds exactly the origin's objects, a component pull holds only its chunks, and a
/// warm pull moves nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pull_from_python_http_server_is_bit_identical() {
    let origin_dir = temporary("python-origin");
    let model = model(
        &origin_dir,
        "acme/tiny",
        &[
            ("unet", &[3 * MIB, 3 * MIB, 2 * MIB + 17]),
            ("vae", &[300_000, 5 * MIB]),
        ],
        1,
    );
    let python = Python::serve(&origin_dir);
    let origin = http(&python.base);

    let root = temporary("python-store");
    let store = open_store(&root);
    let fetcher = fetcher(&store, None);
    let progress = Progress::default();
    let want = Want {
        progress: Some(&progress),
        ..Want::default()
    };
    let pulled = fetcher
        .pull(&origin, "acme/tiny", Some("1.0"), Some("int8"), &want)
        .await
        .unwrap();
    assert_eq!((pulled.tree.as_ref(), &pulled.header), (Some(&model.tree), &model.header));
    assert_eq!(pulled.variant, "int8");
    assert_eq!(pulled.landed.origin, model.bytes());
    assert_eq!(
        progress.held.load(std::sync::atomic::Ordering::Relaxed),
        model.bytes()
    );
    assert_eq!(
        blobs(&root),
        blobs(&origin_dir),
        "the store is the origin's blobs, byte for byte"
    );
    let requests = fetcher.stats().requests;

    let warm = fetcher
        .pull(&origin, "acme/tiny", None, None, &all())
        .await
        .unwrap();
    assert_eq!((warm.landed.local, warm.landed.origin), (model.bytes(), 0));
    assert_eq!(
        fetcher.stats().requests,
        requests,
        "a warm pull asks for no object"
    );

    let digest_root = temporary("python-header-digest");
    let digest_fetcher = crate::fetcher(&open_store(&digest_root), None);
    let digest_progress = Progress::default();
    let revision = format!("sha256:{}", model.header.sha256);
    let by_header = digest_fetcher
        .pull(&origin, "acme/tiny", Some(&revision), None,
              &Want { progress: Some(&digest_progress), ..Want::default() })
        .await
        .unwrap();
    assert_eq!(by_header.landed.origin, model.bytes(), "resolution retains the tree's first source");
    assert_eq!(by_header.landed.local, 0);
    assert_eq!(digest_progress.total.load(std::sync::atomic::Ordering::Relaxed), model.bytes());
    assert_eq!(digest_progress.held.load(std::sync::atomic::Ordering::Relaxed), model.bytes());

    let vae_root = temporary("python-vae");
    let vae_store = open_store(&vae_root);
    let vae = ["vae".to_string()];
    let partial = crate::fetcher(&vae_store, None)
        .pull(
            &origin,
            "acme/tiny",
            None,
            Some("int8"),
            &Want {
                components: Some(&vae),
                ..Want::default()
            },
        )
        .await
        .unwrap();
    let held: BTreeSet<_> = model
        .objects
        .iter()
        .filter(|o| vae_store.has(o))
        .cloned()
        .collect();
    let unet: BTreeSet<_> = model.chunks["unet"].iter().cloned().collect();
    assert!(held.is_disjoint(&unet), "a vae pull fetched unet chunks");
    assert!(model.chunks["vae"].iter().all(|c| held.contains(c)));
    assert!(model.assets.iter().all(|a| !held.contains(a)), "nor the tokenizer's asset");
    assert_eq!(held.len() + unet.len() + model.assets.len(), model.objects.len());
    assert_eq!(
        partial.landed.origin,
        held.iter().map(|o| o.length).sum::<u64>()
    );

    let refused = fetcher
        .pull(&origin, "acme/tiny", Some("9.9"), Some("int8"), &all())
        .await
        .unwrap_err();
    assert_eq!(refused.code, Code::REVISION_ABSENT, "{refused}");
    let refused = fetcher
        .pull(&origin, "acme/absent", None, Some("int8"), &all())
        .await
        .unwrap_err();
    assert_eq!(refused.code, Code::REF_NOT_FOUND, "{refused}");
}

/// A corrupted answer is refused at admission and the object asked again; the same wrong bytes
/// twice end it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_corrupted_object_is_refused_and_refetched() {
    let origin_dir = temporary("corrupt-origin");
    let model = model(
        &origin_dir,
        "acme/corrupt",
        &[("unet", &[2 * MIB, 3 * MIB])],
        2,
    );
    let once = format!("/{}", model.chunks["unet"][0].key());
    let always = format!("/{}", model.chunks["unet"][1].key());
    let (corrupt_once, corrupt_always) = (once.clone(), always.clone());
    let server = Server::serve(
        &origin_dir,
        Arc::new(move |ask, body, socket| {
            if (ask.path == corrupt_once && ask.n == 1) || ask.path == corrupt_always {
                let mut wrong = body.to_vec();
                wrong[body.len() / 2] ^= 0xff;
                let _ = answer(socket, &wrong, ask.first);
                return true;
            }
            false
        }),
    );
    let root = temporary("corrupt-store");
    let store = open_store(&root);
    let fetcher = fetcher(&store, None);
    let origin = http(&server.base);
    let refused = fetcher
        .pull(&origin, "acme/corrupt", None, Some("int8"), &all())
        .await
        .unwrap_err();
    assert_eq!(refused.code, Code::OBJECT_ID_MISMATCH, "{refused}");
    // The refusal ends the pull; the other object's second answer may still be landing.
    fetcher.settled().await;
    let asks = |path: &str| server.blob_asks().iter().filter(|a| a.path == path).count();
    assert_eq!(
        asks(&always),
        2,
        "the same wrong bytes twice end the object"
    );
    assert_eq!(asks(&once), 2, "a wrong answer is asked again");
    assert!(store.has(&model.chunks["unet"][0]) && !store.has(&model.chunks["unet"][1]));
    assert_eq!(
        std::fs::read_dir(root.join("tmp")).unwrap().count(),
        0,
        "no temp is left behind"
    );
}

/// One machine's pull fills the volume; the next machine reads everything from it and asks the
/// origin for no object; a corrupt volume copy is refused, fetched from the origin and replaced.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_volume_is_written_back_and_serves_the_next_machine() {
    let origin_dir = temporary("volume-origin");
    let model = model(
        &origin_dir,
        "acme/shared",
        &[("unet", &[2 * MIB, 2 * MIB, MIB]), ("vae", &[400_000])],
        3,
    );
    let server = Server::serve(&origin_dir, plain());
    let origin = http(&server.base);
    let volume_dir = temporary("volume");
    let volume = VolumeOrigin::new(volume_dir.to_path_buf(), Duration::from_secs(7 * 86400));

    let (a_root, b_root, c_root) = (
        temporary("volume-a"),
        temporary("volume-b"),
        temporary("volume-c"),
    );
    let a = fetcher(&open_store(&a_root), Some(volume.clone()));
    let pulled = a
        .pull(&origin, "acme/shared", None, Some("int8"), &all())
        .await
        .unwrap();
    assert_eq!(pulled.landed.origin, model.bytes());
    a.settled().await;
    assert_eq!(
        blobs(&volume_dir),
        blobs(&origin_dir),
        "every object written back"
    );
    assert_eq!(a.stats().volume_written, model.objects.len() as u64);

    let asked = server.blob_asks().len();
    let b = fetcher(&open_store(&b_root), Some(volume.clone()));
    let pulled = b
        .pull(&origin, "acme/shared", None, Some("int8"), &all())
        .await
        .unwrap();
    assert_eq!(
        (pulled.landed.volume, pulled.landed.origin),
        (model.bytes(), 0)
    );
    assert_eq!(
        server.blob_asks().len(),
        asked,
        "the second machine asked the origin for no object"
    );
    assert_eq!(blobs(&b_root), blobs(&origin_dir));

    let bad = &model.chunks["unet"][1];
    let mut bytes = std::fs::read(volume_dir.join(bad.key())).unwrap();
    bytes[7] ^= 1;
    let path = volume_dir.join(bad.key());
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let c = fetcher(&open_store(&c_root), Some(volume.clone()));
    let pulled = c
        .pull(&origin, "acme/shared", None, Some("int8"), &all())
        .await
        .unwrap();
    assert_eq!(
        pulled.landed.origin, bad.length,
        "only the corrupt copy came from the origin"
    );
    c.settled().await;
    assert_eq!(blobs(&c_root), blobs(&origin_dir));
    assert_eq!(
        blobs(&volume_dir),
        blobs(&origin_dir),
        "the corrupt copy was replaced"
    );
}

/// Over TLS, a private CA is trusted only by a client that names it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn https_trusts_a_private_ca_only_when_named() {
    let origin_dir = temporary("tls-origin");
    let model = model(&origin_dir, "acme/tls", &[("unet", &[MIB + 3])], 8);
    let (server, pem) = Server::serve_tls(&origin_dir);
    let policy = tensorfs::fetch::SourcePolicy {
        allowed_hosts: vec!["127.0.0.1".into()],
        allow_local: true,
        ..Default::default()
    };
    let trusting =
        tensorfs::fetch::Client::with_roots(policy.clone(), pem.as_bytes()).unwrap();
    let root = temporary("tls-store");
    let fetcher = Fetcher::new(
        open_store(&root) as Arc<dyn ObjectStore>,
        trusting.clone(),
        None,
        config(),
    );
    let origin: Arc<dyn Origin> = Arc::new(Endpoint::new(&trusting, &server.base, None).unwrap().anonymous());
    let pulled = fetcher
        .pull(&origin, "acme/tls", None, Some("int8"), &all())
        .await
        .unwrap();
    assert_eq!(pulled.landed.origin, model.bytes());
    assert_eq!(blobs(&root), blobs(&origin_dir));

    let webpki = tensorfs::fetch::Client::new(policy);
    let untrusting: Arc<dyn Origin> = Arc::new(Endpoint::new(&webpki, &server.base, None).unwrap().anonymous());
    let refused = fetcher
        .pull(&untrusting, "acme/tls", None, Some("int8"), &all())
        .await
        .unwrap_err();
    assert_eq!(refused.code, Code::TLS_UNTRUSTED, "{refused}");
}

/// Objects from a private origin (a bearer) are never written to the shared volume, and the
/// bearer reaches the origin on every ask.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_private_origin_writes_nothing_to_the_volume() {
    let origin_dir = temporary("private-origin");
    let model = model(&origin_dir, "acme/private", &[("unet", &[MIB, 70_000])], 4);
    let server = Server::serve(
        &origin_dir,
        Arc::new(|ask, _, socket| {
            if ask.authorization.as_deref() != Some("Bearer s3cret") {
                let _ = std::io::Write::write_all(
                    socket,
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\n\r\n",
                );
                return true;
            }
            false
        }),
    );
    let origin: Arc<dyn Origin> = Arc::new(
        Endpoint::new(&client(), &server.base, Some(tensorfs::fetch::bearer("s3cret")))
            .unwrap(),
    );
    let (volume_dir, root) = (temporary("private-volume"), temporary("private-store"));
    let fetcher = fetcher(
        &open_store(&root),
        Some(VolumeOrigin::new(
            volume_dir.to_path_buf(),
            Duration::from_secs(60),
        )),
    );
    let pulled = fetcher
        .pull(&origin, "acme/private", None, Some("int8"), &all())
        .await
        .unwrap();
    assert_eq!(pulled.landed.origin, model.bytes());
    fetcher.settled().await;
    assert!(
        blobs(&volume_dir).is_empty(),
        "a private object reached the volume"
    );

    let anonymous = http(&server.base);
    let refused = fetcher
        .pull(&anonymous, "acme/private", None, Some("int8"), &all())
        .await
        .unwrap_err();
    assert_eq!(refused.code, Code::CREDENTIAL_REQUIRED, "{refused}");
}

/// A volume read that blocks in the kernel (a FIFO standing in for a hung mount) is abandoned
/// on its own measured silence: the object comes from the origin, the volume is left alone
/// while the read is out, and serves again once it returns.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hung_volume_is_left_alone_until_it_answers() {
    struct ReleaseFifo(std::path::PathBuf);
    impl Drop for ReleaseFifo {
        fn drop(&mut self) {
            use std::os::unix::fs::OpenOptionsExt;
            // An assertion failure must release the blocking reader before Tokio joins
            // its blocking pool. O_RDWR is nonblocking even if the reader never started.
            let _ = std::fs::OpenOptions::new().read(true).write(true)
                .custom_flags(libc::O_NONBLOCK).open(&self.0);
        }
    }
    let origin_dir = temporary("hung-origin");
    let model = model(&origin_dir, "acme/hung", &[("unet", &[MIB, MIB])], 5);
    let server = Server::serve(&origin_dir, plain());
    let origin = http(&server.base);
    let volume_dir = temporary("hung-volume");
    let fifo = volume_dir.join(model.tree.key());
    std::fs::create_dir_all(fifo.parent().unwrap()).unwrap();
    assert!(std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap()
        .success());
    let _release = ReleaseFifo(fifo.clone());
    let volume = VolumeOrigin::new(volume_dir.to_path_buf(), Duration::from_secs(60));

    let root = temporary("hung-store");
    let fetcher = fetcher(&open_store(&root), Some(volume.clone()));
    let pulled = fetcher
        .pull(&origin, "acme/hung", None, Some("int8"), &all())
        .await
        .unwrap();
    assert_eq!(
        (pulled.landed.volume, pulled.landed.origin),
        (0, model.bytes())
    );
    fetcher.settled().await;
    assert_eq!(
        fetcher.stats().volume_written,
        0,
        "nothing is written while a read hangs"
    );

    // The blocked open returns once a writer opens the FIFO.
    let writer = std::fs::OpenOptions::new().write(true).open(&fifo).unwrap();
    drop(writer);
    std::fs::remove_file(&fifo).unwrap();
    for object in &model.objects {
        let path = volume_dir.join(object.key());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::copy(origin_dir.join(object.key()), path).unwrap();
    }
    let mut from_volume = 0;
    for attempt in 0..100 {
        let fresh = temporary(&format!("hung-after-{attempt}"));
        let next = Fetcher::new(
            open_store(&fresh) as Arc<dyn ObjectStore>,
            client(),
            Some(volume.clone()),
            config(),
        );
        from_volume = next
            .pull(&origin, "acme/hung", None, Some("int8"), &all())
            .await
            .unwrap()
            .landed
            .volume;
        if from_volume == model.bytes() {
            break;
        }
        tokio::time::sleep(SAMPLE).await;
    }
    assert_eq!(
        from_volume,
        model.bytes(),
        "the volume never answered again"
    );
}

/// An object wanted by two callers is fetched once, and a smaller priority goes first: while
/// the one worker holds the first priority-10 object, a priority-1 batch is queued, and runs
/// before the rest of the first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_object_is_fetched_once_and_priority_orders_the_queue() {
    let origin_dir = temporary("order-origin");
    let model = model(
        &origin_dir,
        "acme/order",
        &[("a", &[300_000; 6]), ("b", &[200_000; 6])],
        6,
    );
    let (a, b) = (model.chunks["a"].clone(), model.chunks["b"].clone());
    let held = format!("/{}", a[0].key());
    let server = Server::serve(
        &origin_dir,
        Arc::new(move |ask, _, _| {
            if ask.path == held {
                std::thread::sleep(Duration::from_millis(300));
            }
            false
        }),
    );
    let origin = http(&server.base);
    let root = temporary("order-store");
    let store = open_store(&root);
    // A coarse tick: the held answer must not be judged a stall (that is the tail tests' job).
    let config = Config {
        tick: Duration::from_millis(100),
        streams: 1,
        max_streams: 1,
    };
    let fetcher = Fetcher::new(
        Arc::clone(&store) as Arc<dyn ObjectStore>,
        client(),
        None,
        config,
    );
    let mut urgent = b.clone();
    urgent.push(a[5].clone());
    let (later, sooner) = (
        Want {
            priority: 10,
            ..Want::default()
        },
        Want {
            priority: 1,
            ..Want::default()
        },
    );
    let (first, second) = tokio::join!(fetcher.objects(&origin, &a, &later), async {
        while server.blob_asks().is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        fetcher.objects(&origin, &urgent, &sooner).await
    });
    let total = |objects: &[tensorfs::format::ObjectRef]| {
        objects.iter().map(|o| o.length).sum::<u64>()
    };
    // The shared object counts for both callers.
    assert_eq!(
        first.unwrap().origin + second.unwrap().origin,
        total(&a) + total(&b) + a[5].length
    );
    let asks: Vec<String> = server.blob_asks().into_iter().map(|ask| ask.path).collect();
    let expected: Vec<String> = [&a[..1], &b[..], &a[5..], &a[1..5]]
        .concat()
        .iter()
        .map(|o| format!("/{}", o.key()))
        .collect();
    assert_eq!(
        asks, expected,
        "each object once, priority 1 before priority 10"
    );
}

/// An origin whose every answer dies half way and that honours no range: nothing new ever
/// lands, so the object ends on the fetch's own measured patience and quotes the last answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pull_where_nothing_lands_ends_and_quotes_the_last_answer() {
    let origin_dir = temporary("never-origin");
    let model = model(&origin_dir, "acme/never", &[("unet", &[MIB])], 7);
    let never = format!("/{}", model.chunks["unet"][0].key());
    let server = Server::serve(
        &origin_dir,
        Arc::new(move |ask, body, socket| {
            if ask.path != never {
                return false;
            }
            let _ = head(socket, body.len(), 0);
            let _ = std::io::Write::write_all(socket, &body[..body.len() / 2]);
            true
        }),
    );
    let root = temporary("never-store");
    let fetcher = fetcher(&open_store(&root), None);
    let started = std::time::Instant::now();
    let refused = fetcher
        .pull(
            &http(&server.base),
            "acme/never",
            None,
            Some("int8"),
            &all(),
        )
        .await
        .unwrap_err();
    assert_eq!(refused.code, Code::STALLED, "{refused}");
    assert!(
        refused.detail.contains("the last answer") && refused.detail.contains("of 1048576 B"),
        "{refused}"
    );
    let ticks = started.elapsed().as_secs_f64() / SAMPLE.as_secs_f64();
    let asks = server.blob_asks().len() as f64;
    assert!(
        asks < ticks * 1.5,
        "{asks} asks in {ticks:.0} ticks: a hot loop"
    );
}
