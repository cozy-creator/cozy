//! A real loopback hub (plain or TLS): the closure, presign and object GET routes over
//! sockets the test starts, serving real checkpoints. Nothing about the transport is mocked.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use tensorfs_core::canon::{self, Value};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::{Draft, Entry};
use tensorfs_core::registry;
use tensorfs_core::store::Store;

pub fn temporary(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "tfs-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn store(dir: &Path) -> PathBuf {
    let root = dir.join("store");
    Store::init(&root).unwrap();
    root
}

/// A real checkpoint as a hub serves it: `model@release` lane `lane`.
pub struct Checkpoint {
    pub model: String,
    pub release: String,
    pub lane: String,
    pub bodies: HashMap<String, Vec<u8>>,
    pub manifest: ObjectRef,
    pub header: ObjectRef,
    /// Header and blobs, manifest excluded: what the hub's closure lists.
    pub objects: Vec<ObjectRef>,
    /// Each component's blobs, in component order.
    pub components: Vec<Vec<ObjectRef>>,
    pub blobs: Vec<ObjectRef>,
}

impl Checkpoint {
    pub fn refspec(&self) -> String {
        format!("{}@{}", self.model, self.release)
    }
    pub fn total(&self) -> u64 {
        self.bodies.values().map(|b| b.len() as u64).sum()
    }
}

/// One blob's bytes, distinct per (`salt`, component, index).
pub fn body(size: usize, salt: usize, component: usize, index: usize) -> Vec<u8> {
    let mut body: Vec<u8> = (0..size)
        .map(|at| ((at * 31 + index * 7 + component * 13 + salt) % 251) as u8)
        .collect();
    for (at, byte) in ((salt * 1_000_003 + component * 1009 + index) as u64)
        .to_le_bytes()
        .iter()
        .enumerate()
    {
        body[at] ^= byte;
    }
    body
}

/// `acme/model@r1` lane `public` with one `model` component of `blobs` blobs.
pub fn checkpoint(blobs: usize, size: usize, salt: usize) -> Checkpoint {
    components("acme/model", &[blobs], size, salt, &[])
}

/// `model@r1` lane `public`, its components `c0`, `c1`, ... of `counts` blobs each, the first
/// component also reaching `shared` (another checkpoint's blobs, with their bodies).
pub fn components(
    model: &str,
    counts: &[usize],
    size: usize,
    salt: usize,
    shared: &[(ObjectRef, Vec<u8>)],
) -> Checkpoint {
    let spec = registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let mut bodies = HashMap::new();
    let (mut header_components, mut reached) = (vec![], vec![]);
    for (component, count) in counts.iter().enumerate() {
        let (mut tensors, mut refs) = (vec![], vec![]);
        let extra = if component == 0 { shared.to_vec() } else { vec![] };
        let made = (0..*count).map(|index| {
            let body = body(size, salt, component, index);
            (ObjectRef::of(&body), body)
        });
        for (index, (object, body)) in made.chain(extra).enumerate() {
            tensors.push((
                format!("blocks.{index}.weight"),
                Tensor {
                    dtype: Dtype::F32,
                    shape: vec![(body.len() / 4) as u64],
                    encoding: spec.object_id(),
                    parts: vec![(
                        "value".into(),
                        Part {
                            dtype: Dtype::F32,
                            shape: vec![(body.len() / 4) as u64],
                            body: Body::Segments(vec![object.clone()]),
                        },
                    )],
                },
            ));
            refs.push(object.clone());
            bodies.insert(object.sha256.clone(), body);
        }
        header_components.push((format!("c{component}"), tensors));
        reached.push(refs);
    }
    let header = Header {
        configs: Vec::new(),
        assets: Vec::new(),
        encodings: vec![spec],
        components: header_components,
    };
    header.validate(&Closure::default()).unwrap();
    let header_bytes = header.canonical_bytes().unwrap();
    let header_ref = ObjectRef::of(&header_bytes);
    bodies.insert(header_ref.sha256.clone(), header_bytes);
    let manifest = Draft {
        entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header_ref.clone()))],
    }
    .seal()
    .unwrap()
    .canonical_bytes();
    let manifest_ref = ObjectRef::of(&manifest);
    bodies.insert(manifest_ref.sha256.clone(), manifest);
    let mut objects: Vec<ObjectRef> = (bodies.iter())
        .filter(|(hex, _)| **hex != manifest_ref.sha256)
        .map(|(hex, body)| ObjectRef {
            sha256: hex.clone(),
            length: body.len() as u64,
        })
        .collect();
    objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    Checkpoint {
        model: model.into(),
        release: "r1".into(),
        lane: "public".into(),
        bodies,
        manifest: manifest_ref,
        header: header_ref,
        objects,
        blobs: reached.iter().flatten().cloned().collect(),
        components: reached,
    }
}

fn row(object: &ObjectRef) -> Value {
    Value::obj(vec![
        ("length", Value::uint(object.length)),
        ("sha256", Value::str(object.sha256.clone())),
    ])
}

/// How one object GET is answered: its body, `Truncate` (declare it, send half, close), or
/// `Hold` (wait until [`Hub::open`], then its body).
pub enum Answer {
    Body,
    Truncate,
    Hold,
}

pub type Rule = Arc<dyn Fn(&str, usize) -> Answer + Send + Sync>;

pub fn serve_all() -> Rule {
    Arc::new(|_, _| Answer::Body)
}

pub struct Hub {
    pub base: String,
    pub gets: Arc<Mutex<HashMap<String, usize>>>,
    /// Object GETs in the order they were asked.
    pub log: Arc<Mutex<Vec<String>>>,
    gate: Arc<(Mutex<bool>, Condvar)>,
}

impl Hub {
    pub fn gets(&self, hex: &str) -> usize {
        *self.gets.lock().unwrap().get(hex).unwrap_or(&0)
    }
    /// Object GETs, closures and presigns not counted.
    pub fn total_gets(&self) -> usize {
        let gets = self.gets.lock().unwrap();
        gets.iter()
            .filter(|(key, _)| *key != "closure" && *key != "presign")
            .map(|(_, n)| n)
            .sum()
    }
    pub fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
    /// Every held GET is answered.
    pub fn open(&self) {
        *self.gate.0.lock().unwrap() = true;
        self.gate.1.notify_all();
    }
}

/// A hub serving `checkpoints`. `rule` answers each object GET by (digest, how many times it
/// was asked); `gap` paces every object body.
pub fn hub(
    checkpoints: Vec<Arc<Checkpoint>>,
    tls: Option<Arc<rustls::ServerConfig>>,
    gap: Duration,
    rule: Rule,
) -> Hub {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let scheme = if tls.is_some() { "https" } else { "http" };
    let base = format!("{scheme}://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let hub = Hub {
        base,
        gets: Arc::default(),
        log: Arc::default(),
        gate: Arc::new((Mutex::new(false), Condvar::new())),
    };
    let served = Served {
        checkpoints: Arc::new(checkpoints),
        base: hub.base.clone(),
        gets: Arc::clone(&hub.gets),
        log: Arc::clone(&hub.log),
        gate: Arc::clone(&hub.gate),
        gap,
        rule,
    };
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(socket) = socket else { return };
            let (served, tls) = (served.clone(), tls.clone());
            std::thread::spawn(move || match tls {
                Some(config) => {
                    let conn = rustls::ServerConnection::new(config).unwrap();
                    let mut stream = rustls::StreamOwned::new(conn, socket);
                    served.serve(&mut stream);
                    stream.conn.send_close_notify();
                    let _ = stream.flush();
                }
                None => served.serve(&mut { socket }),
            });
        }
    });
    hub
}

#[derive(Clone)]
struct Served {
    checkpoints: Arc<Vec<Arc<Checkpoint>>>,
    base: String,
    gets: Arc<Mutex<HashMap<String, usize>>>,
    log: Arc<Mutex<Vec<String>>>,
    gate: Arc<(Mutex<bool>, Condvar)>,
    gap: Duration,
    rule: Rule,
}

impl Served {
    fn count(&self, key: &str) -> usize {
        let mut gets = self.gets.lock().unwrap();
        let count = gets.entry(key.to_string()).or_insert(0);
        *count += 1;
        *count
    }

    fn serve(&self, stream: &mut (impl Read + Write)) {
        let Some((method, path, body)) = request(stream) else {
            return;
        };
        let json = |value: Value| (200, canon::write(&value));
        let field = |name: &str| -> Option<String> {
            let Ok(Value::Obj(fields)) = canon::parse(&body, 1 << 20) else {
                return None;
            };
            fields
                .into_iter()
                .find(|(k, _)| k == name)
                .and_then(|(_, v)| match v {
                    Value::Str(s) => Some(s),
                    _ => None,
                })
        };
        let (status, reply, truncate) = match (method.as_str(), path.as_str()) {
            ("POST", "/v1/tensorfs/closure") => {
                self.count("closure");
                let asked = field("ref").unwrap_or_default();
                match self
                    .checkpoints
                    .iter()
                    .find(|c| c.refspec() == asked || c.model == asked)
                {
                    Some(c) => {
                        let (status, reply) = json(Value::obj(vec![
                            ("complete", Value::Bool(true)),
                            ("lane", Value::str(c.lane.clone())),
                            ("manifest", row(&c.manifest)),
                            ("model", Value::str(c.model.clone())),
                            ("objects", Value::arr(c.objects.iter().map(row).collect())),
                            ("presign_max_digests", Value::uint(1024)),
                            ("release", Value::str(c.release.clone())),
                            ("scope", Value::str("runtime")),
                        ]));
                        (status, reply, false)
                    }
                    None => (
                        404,
                        b"{\"error\":{\"code\":\"ref_not_found\",\"message\":\"no such ref\"}}".to_vec(),
                        false,
                    ),
                }
            }
            ("POST", "/v1/tensorfs/presign") => {
                self.count("presign");
                let Value::Obj(fields) = canon::parse(&body, 1 << 20).unwrap() else {
                    panic!("presign body")
                };
                let Some((_, Value::Arr(digests))) = fields.into_iter().find(|(k, _)| k == "digests") else {
                    panic!("presign digests")
                };
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let urls = digests
                    .into_iter()
                    .map(|d| {
                        let Value::Str(hex) = d else { panic!() };
                        let url = Value::str(format!("{}/obj/{hex}", self.base));
                        (hex, url)
                    })
                    .collect();
                let (status, reply) = json(Value::obj(vec![
                    ("expires_at_unix", Value::uint(now + 3600)),
                    ("server_time_unix", Value::uint(now)),
                    ("urls", Value::map(urls)),
                ]));
                (status, reply, false)
            }
            ("GET", path) if path.starts_with("/obj/") => {
                let hex = path["/obj/".len()..].to_string();
                let asked = self.count(&hex);
                self.log.lock().unwrap().push(hex.clone());
                let answer = (self.rule)(&hex, asked);
                if matches!(answer, Answer::Hold) {
                    let (open, opened) = &*self.gate;
                    drop(opened.wait_while(open.lock().unwrap(), |open| !*open).unwrap());
                }
                match self.checkpoints.iter().find_map(|c| c.bodies.get(&hex)) {
                    Some(body) => (200, body.clone(), matches!(answer, Answer::Truncate)),
                    None => (404, b"no such object".to_vec(), false),
                }
            }
            _ => (404, b"no such route".to_vec(), false),
        };
        let head = format!(
            "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            reply.len()
        );
        let _ = stream.write_all(head.as_bytes());
        let sent = if truncate {
            &reply[..reply.len() / 2]
        } else {
            &reply[..]
        };
        if self.gap.is_zero() || !path.starts_with("/obj/") {
            let _ = stream.write_all(sent);
            return;
        }
        for piece in sent.chunks(sent.len().div_ceil(4).max(1)) {
            std::thread::sleep(self.gap);
            if stream.write_all(piece).and_then(|()| stream.flush()).is_err() {
                return;
            }
        }
    }
}

fn request(stream: &mut impl Read) -> Option<(String, String, Vec<u8>)> {
    let (mut collected, mut buf) = (Vec::new(), [0u8; 4096]);
    let split = loop {
        if let Some(at) = collected.windows(4).position(|w| w == b"\r\n\r\n") {
            break at;
        }
        let n = stream.read(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        collected.extend_from_slice(&buf[..n]);
    };
    let head = String::from_utf8_lossy(&collected[..split]).to_string();
    let mut body = collected[split + 4..].to_vec();
    let mut line = head.lines().next()?.split_whitespace();
    let (method, target) = (line.next()?.to_string(), line.next()?.to_string());
    let length: usize = head
        .lines()
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.trim().parse().ok())
        .unwrap_or(0);
    while body.len() < length {
        let n = stream.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }
    Some((method, target.split('?').next()?.to_string(), body))
}

/// A loopback TLS server config under a fresh self-signed authority, and that authority.
pub fn tls_config() -> (Arc<rustls::ServerConfig>, String) {
    let key = rcgen::generate_simple_self_signed(vec!["127.0.0.1".into()]).unwrap();
    let cert = rustls::pki_types::CertificateDer::from(key.cert.der().to_vec());
    let private = rustls::pki_types::PrivateKeyDer::try_from(key.key_pair.serialize_der())
        .unwrap()
        .clone_key();
    let config =
        rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert], private)
            .unwrap();
    (Arc::new(config), key.cert.pem())
}
