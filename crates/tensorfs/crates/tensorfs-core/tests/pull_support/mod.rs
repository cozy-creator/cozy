//! What the pull tests share: the v2 store, a checkpoint written as a static origin, and two real loopback origins: `python -m http.server`, and a small
//! server that injects faults on chosen asks.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tensorfs_core::fetch::{Client, SourcePolicy};
use tensorfs_core::format::{
    Body, Component, Dtype, Header, Location, Manifest, ObjectRef, Release, Repo, Tensor,
};
use tensorfs_core::store2::Store;

pub const MIB: usize = 1 << 20;

/// A scratch directory, removed when dropped (a failed test included).
pub struct Scratch(PathBuf);

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn temporary(name: &str) -> Scratch {
    static UNIQUE: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "tfs-pull-{name}-{}-{}",
        std::process::id(),
        UNIQUE.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

/// A client for loopback origins.
pub fn client() -> Client {
    Client::new(SourcePolicy {
        allowed_hosts: vec!["127.0.0.1".into()],
        allow_local: true,
        max_redirects: 2,
        ..Default::default()
    })
}

// ------------------------------------------------------------------ the store

/// The v2 store at `root`, which the fetcher admits through.
pub fn open_store(root: &Path) -> Arc<Store> {
    Arc::new(Store::open(root).unwrap())
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every file under `root/blobs`, by relative path.
pub fn blobs(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(dir: &Path, root: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            match path.is_dir() {
                true => walk(&path, root, out),
                false => {
                    out.insert(
                        path.strip_prefix(root).unwrap().display().to_string(),
                        fs::read(&path).unwrap(),
                    );
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(&root.join("blobs"), root, &mut out);
    out
}

// ------------------------------------------------------------------ a checkpoint

pub struct Model {
    pub manifest: ObjectRef,
    pub header: ObjectRef,
    /// Chunk refs by component.
    pub chunks: BTreeMap<String, Vec<ObjectRef>>,
    /// Every object of the closure.
    pub objects: Vec<ObjectRef>,
}

impl Model {
    pub fn bytes(&self) -> u64 {
        self.objects.iter().map(|o| o.length).sum()
    }
}

fn noise(seed: u64, length: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn put(origin: &Path, bytes: &[u8]) -> ObjectRef {
    let object = ObjectRef::of(bytes);
    let path = origin.join(object.key());
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
    object
}

/// `name` at release 1.0, lane `int8`, written into `origin` as a static origin: per component,
/// chunks of the given sizes, each holding one U8 tensor (the first component's first tensor
/// spans its first two chunks), plus a config per component and a tokenizer file.
pub fn model(origin: &Path, name: &str, components: &[(&str, &[usize])], seed: u64) -> Model {
    let mut header = Header::default();
    let mut chunks = BTreeMap::new();
    let mut files = BTreeMap::new();
    for (c, (component, sizes)) in components.iter().enumerate() {
        let mut tensors = BTreeMap::new();
        let first = header.chunks.len();
        for (i, size) in sizes.iter().enumerate() {
            let chunk = put(origin, &noise(seed * 1000 + (c * 100 + i) as u64, *size));
            header.chunks.push(chunk);
            if c == 0 && i == 1 {
                continue;
            }
            let length = match (c, i) {
                (0, 0) if sizes.len() > 1 => (sizes[0] + sizes[1]) as u64,
                _ => *size as u64,
            };
            let at = Location {
                chunk: (first + i) as u64,
                offset: 0,
                length,
            };
            tensors.insert(
                format!("blocks.{i}.weight"),
                Tensor {
                    dtype: Dtype::U8,
                    shape: vec![length],
                    body: Body::Data(at),
                },
            );
        }
        chunks.insert(component.to_string(), header.chunks[first..].to_vec());
        header
            .components
            .insert(component.to_string(), Component { tensors });
        files.insert(
            format!("{component}/config.json"),
            put(
                origin,
                format!("{{\"component\":\"{component}\",\"seed\":{seed}}}").as_bytes(),
            ),
        );
    }
    files.insert(
        "tokenizer/vocab.json".into(),
        put(origin, &noise(seed + 7, 3000)),
    );
    let header_ref = put(origin, &header.to_json().unwrap());
    files.insert("model.cozytensors".into(), header_ref.clone());
    let manifest = Manifest {
        files,
        ..Default::default()
    };
    let manifest_ref = put(origin, &manifest.to_json().unwrap());
    let repo = Repo {
        name: name.into(),
        releases: vec![Release {
            version: "1.0".into(),
            lanes: [("int8".to_string(), manifest_ref.clone())].into(),
        }],
    };
    let path = origin.join(repo.key());
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, repo.to_json().unwrap()).unwrap();
    let mut objects: Vec<ObjectRef> = manifest.files.values().cloned().collect();
    objects.extend(header.chunks.iter().cloned());
    objects.push(manifest_ref.clone());
    objects.sort();
    objects.dedup();
    Model {
        manifest: manifest_ref,
        header: header_ref,
        chunks,
        objects,
    }
}

// ------------------------------------------------------------------ python -m http.server

/// `python3 -m http.server` over `root` on loopback; killed on drop.
pub struct Python {
    child: Child,
    pub base: String,
}

impl Python {
    pub fn serve(root: &Path) -> Python {
        let mut child = Command::new("python3")
            .args([
                "-u",
                "-m",
                "http.server",
                "0",
                "--bind",
                "127.0.0.1",
                "--directory",
            ])
            .arg(root)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("python3");
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        // "Serving HTTP on 127.0.0.1 port 41234 (http://127.0.0.1:41234/) ..."
        let port = line
            .split(" port ")
            .nth(1)
            .and_then(|rest| rest.split(' ').next())
            .expect(&line)
            .to_string();
        Python {
            child,
            base: format!("http://127.0.0.1:{port}"),
        }
    }
}

impl Drop for Python {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ------------------------------------------------------------------ a fault-injecting origin

/// One GET as the server saw it.
#[derive(Debug, Clone)]
pub struct Ask {
    pub path: String,
    /// The first byte asked for.
    pub first: usize,
    /// This path's ask number, from 1.
    pub n: usize,
    pub authorization: Option<String>,
}

/// Answers an ask itself (and the connection then closes), or returns false for the plain answer.
pub type Fault = Arc<dyn Fn(&Ask, &[u8], &mut TcpStream) -> bool + Send + Sync>;

pub fn plain() -> Fault {
    Arc::new(|_, _, _| false)
}

/// A loopback HTTP/1.1 origin over a static-origin directory, honouring ranges.
pub struct Server {
    pub base: String,
    pub asks: Arc<Mutex<Vec<Ask>>>,
}

impl Server {
    pub fn serve(root: &Path, fault: Fault) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let asks: Arc<Mutex<Vec<Ask>>> = Arc::default();
        let bodies: Arc<Mutex<HashMap<String, Arc<Vec<u8>>>>> = Arc::default();
        let (root, log) = (root.to_path_buf(), Arc::clone(&asks));
        std::thread::spawn(move || {
            for socket in listener.incoming() {
                let Ok(socket) = socket else { return };
                let (root, log, bodies, fault) = (
                    root.clone(),
                    Arc::clone(&log),
                    Arc::clone(&bodies),
                    Arc::clone(&fault),
                );
                std::thread::spawn(move || connection(socket, &root, &log, &bodies, &*fault));
            }
        });
        Server { base, asks }
    }

    /// The same origin over TLS, under a self-signed certificate for 127.0.0.1, whose PEM
    /// is returned.
    pub fn serve_tls(root: &Path) -> (Server, String) {
        let key = rcgen::generate_simple_self_signed(vec!["127.0.0.1".into()]).unwrap();
        let cert = rustls::pki_types::CertificateDer::from(key.cert.der().to_vec());
        let private = rustls::pki_types::PrivateKeyDer::try_from(key.signing_key.serialize_der())
            .unwrap()
            .clone_key();
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![cert], private)
        .unwrap();
        let config = Arc::new(config);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!(
            "https://127.0.0.1:{}",
            listener.local_addr().unwrap().port()
        );
        let asks: Arc<Mutex<Vec<Ask>>> = Arc::default();
        let bodies: Arc<Mutex<HashMap<String, Arc<Vec<u8>>>>> = Arc::default();
        let (root, log) = (root.to_path_buf(), Arc::clone(&asks));
        std::thread::spawn(move || {
            for socket in listener.incoming() {
                let Ok(socket) = socket else { return };
                let (root, log, bodies) = (root.clone(), Arc::clone(&log), Arc::clone(&bodies));
                let tls = rustls::StreamOwned::new(
                    rustls::ServerConnection::new(Arc::clone(&config)).unwrap(),
                    socket,
                );
                std::thread::spawn(move || connection(tls, &root, &log, &bodies, &|_, _, _| false));
            }
        });
        (Server { base, asks }, key.cert.pem())
    }

    /// Asks for blobs, in order.
    pub fn blob_asks(&self) -> Vec<Ask> {
        self.asks
            .lock()
            .unwrap()
            .iter()
            .filter(|a| a.path.starts_with("/blobs/"))
            .cloned()
            .collect()
    }
}

fn connection<S: Read + Write>(
    mut socket: S,
    root: &Path,
    log: &Mutex<Vec<Ask>>,
    bodies: &Mutex<HashMap<String, Arc<Vec<u8>>>>,
    fault: &(dyn Fn(&Ask, &[u8], &mut S) -> bool + Send + Sync),
) {
    loop {
        let mut head = Vec::new();
        let mut byte = [0];
        while !head.ends_with(b"\r\n\r\n") {
            match socket.read(&mut byte) {
                Ok(1) => head.push(byte[0]),
                _ => return,
            }
        }
        let head = String::from_utf8_lossy(&head).to_string();
        let path = head.split(' ').nth(1).unwrap_or("/").to_string();
        let header = |name: &str| {
            head.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                key.trim()
                    .eq_ignore_ascii_case(name)
                    .then(|| value.trim().to_string())
            })
        };
        let first = header("range")
            .and_then(|r| {
                r.strip_prefix("bytes=")
                    .and_then(|r| r.split('-').next())
                    .and_then(|f| f.parse().ok())
            })
            .unwrap_or(0);
        let body = {
            let mut bodies = bodies.lock().unwrap();
            match bodies.get(&path) {
                Some(body) => Some(Arc::clone(body)),
                None => fs::read(root.join(path.trim_start_matches('/')))
                    .ok()
                    .map(|bytes| {
                        let body = Arc::new(bytes);
                        bodies.insert(path.clone(), Arc::clone(&body));
                        body
                    }),
            }
        };
        let ask = {
            let mut log = log.lock().unwrap();
            let n = log.iter().filter(|a| a.path == path).count() + 1;
            let ask = Ask {
                path,
                first,
                n,
                authorization: header("authorization"),
            };
            log.push(ask.clone());
            ask
        };
        let Some(body) = body else {
            let _ = socket.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n");
            continue;
        };
        if fault(&ask, &body, &mut socket) {
            return;
        }
        if answer(&mut socket, &body, first).is_err() {
            return;
        }
    }
}

/// The status line and headers for `body` from byte `first`.
pub fn head(socket: &mut impl Write, total: usize, first: usize) -> std::io::Result<()> {
    let head = match first {
        0 => format!("HTTP/1.1 200 OK\r\ncontent-length: {total}\r\n\r\n"),
        _ => format!("HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\ncontent-range: bytes {first}-{}/{total}\r\n\r\n", total - first, total - 1),
    };
    socket.write_all(head.as_bytes())
}

pub fn answer(socket: &mut impl Write, body: &[u8], first: usize) -> std::io::Result<()> {
    head(socket, body.len(), first)?;
    socket.write_all(&body[first..])
}

/// Holds the connection open, sending nothing, until the client goes.
pub fn hold(socket: &mut TcpStream) {
    socket
        .set_read_timeout(Some(Duration::from_millis(20)))
        .unwrap();
    let mut byte = [0];
    loop {
        match socket.read(&mut byte) {
            Ok(0) => return,
            Ok(_) => {}
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => return,
        }
    }
}

/// Writes `bytes` at `rate` bytes per second, in small drips.
pub fn drip(socket: &mut TcpStream, bytes: &[u8], rate: u64) -> std::io::Result<()> {
    const DRIP: usize = 4096;
    for piece in bytes.chunks(DRIP) {
        std::thread::sleep(Duration::from_secs_f64(piece.len() as f64 / rate as f64));
        socket.write_all(piece)?;
    }
    Ok(())
}

/// Every MiB of `slow` paths whose index is 1 mod 4 crawls at `rate` for the first ask that
/// reaches it and comes at once for every later one, like a cold R2 object.
pub fn cold(slow: HashSet<String>, rate: u64) -> Fault {
    let touched: Arc<Mutex<HashSet<(String, usize)>>> = Arc::default();
    Arc::new(move |ask, body, socket| {
        if !slow.contains(&ask.path) {
            return false;
        }
        if head(socket, body.len(), ask.first).is_err() {
            return true;
        }
        let mut at = ask.first;
        while at < body.len() {
            let mib = at / MIB;
            let end = ((mib + 1) * MIB).min(body.len());
            let first_ask = touched.lock().unwrap().insert((ask.path.clone(), mib));
            let sent = match mib % 4 == 1 && first_ask {
                true => drip(socket, &body[at..end], rate),
                false => socket.write_all(&body[at..end]),
            };
            if sent.is_err() {
                return true;
            }
            at = end;
        }
        true
    })
}
