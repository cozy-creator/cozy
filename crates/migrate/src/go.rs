//! The Go CLI's home, read once and never written: its `creator.sqlite` (opened read-only), its
//! config, device keys, sessions and per-rental files.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use rusqlite::{Connection, OpenFlags, Row};
use serde::Deserialize;

pub type Result<T> = std::result::Result<T, String>;

/// One Go request, the row `cozy run list` showed, with what its machine execution recorded.
pub struct Request {
    pub id: String,
    pub kind: String,
    pub package: String,
    pub entrypoint: String,
    pub release: String,
    pub state: String,
    pub hub: String,
    pub created_at: String,
    /// The human word of the rental it was bound to.
    pub machine: String,
    /// The rental id it was assigned to.
    pub worker: String,
    /// A model download's or upload's JSON body.
    pub operation: Vec<u8>,
    pub execution: Option<Execution>,
    pub accepted_at: Option<String>,
    pub settled_at: Option<String>,
    /// The newest `run.failed` or `run.canceled` payload.
    pub ending: Option<String>,
    pub out_dir: Option<String>,
    pub published: Vec<String>,
}

/// Its `machine_executions` row: the run on a machine over `cozy.machine.v1`.
pub struct Execution {
    /// `local`, a rental id, or a dev endpoint's name.
    pub machine: String,
    /// The machine's first `RunState`, once it accepted the run over `cozy.machine.v1`.
    pub receipt: Vec<u8>,
    pub cursor: u64,
    pub outcome: Vec<u8>,
    pub cancel_requested: bool,
    pub accepted: bool,
    pub sent: bool,
    /// The Go CLI stopped following it: abandoned, or its machine was lost.
    pub dropped: bool,
}

/// One revision of an output item (`output_item.delta`).
#[derive(Clone, Deserialize)]
pub struct Product {
    pub output: String,
    #[serde(default)]
    pub index: u32,
    pub rev: u64,
    #[serde(default)]
    pub length: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub media_type: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub duration_us: u64,
    #[serde(default)]
    pub path: Option<String>,
}

pub struct Rental {
    pub id: String,
    /// The name it was asked under.
    pub asked: String,
    /// The Hub's name for the pod.
    pub hub_name: String,
    pub state: String,
    pub hub: String,
    pub address: String,
    pub worker_id: String,
    /// The rental's key: `<id>.creator.pem`, or its pending one while it boots.
    pub key: PathBuf,
    pub cert: PathBuf,
}

/// `auth/<sha256(origin)>.session.json`.
#[derive(Deserialize)]
pub struct Session {
    pub access_token: String,
    pub expires_at: String,
    pub device_key_id: String,
}

/// `machine/agent.json`: the TensorD the Go CLI launched on this computer.
#[derive(Deserialize)]
pub struct Agent {
    pub worker_id: String,
    pub worker_port: u16,
}

pub struct Records {
    pub requests: Vec<Request>,
    /// By request id: each item's newest revision.
    pub products: HashMap<String, Vec<Product>>,
    pub rentals: Vec<Rental>,
}

pub fn records(home: &Path) -> Result<Records> {
    let path = home.join("creator.sqlite");
    let db = Connection::open_with_flags(
        &path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    let sql = |e: rusqlite::Error| format!("{}: {e}", path.display());
    let mut statement = db.prepare(REQUESTS).map_err(sql)?;
    let requests = statement
        .query_map([], request)
        .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
        .map_err(sql)?;
    let mut statement = db
        .prepare("SELECT request_id, payload FROM request_events WHERE type = 'output_item.delta' ORDER BY seq")
        .map_err(sql)?;
    let mut newest: HashMap<String, Vec<Product>> = HashMap::new();
    let deltas = statement
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
        .map_err(sql)?;
    for (id, payload) in deltas {
        let Ok(product) = serde_json::from_str::<Product>(&payload) else {
            continue;
        };
        let items = newest.entry(id).or_default();
        match items
            .iter_mut()
            .find(|p| p.output == product.output && p.index == product.index)
        {
            Some(held) if held.rev <= product.rev => *held = product,
            Some(_) => {}
            None => items.push(product),
        }
    }
    let mut statement = db
        .prepare(
            "SELECT r.id, r.machine_name, r.stable_machine_name, r.state, r.hub, r.address, r.expected_worker_id,
                 (SELECT operation_key FROM rental_operations o WHERE o.rental_id = r.id LIMIT 1)
             FROM rentals r ORDER BY r.rented_at",
        )
        .map_err(sql)?;
    let rentals = statement
        .query_map([], |r| {
            let id: String = r.get(0)?;
            let operation: Option<String> = r.get(7)?;
            let mut key = home.join("rentals").join(format!("{id}.creator.pem"));
            if let (false, Some(operation)) = (key.exists(), operation) {
                let digest =
                    hex::encode(<sha2::Sha256 as sha2::Digest>::digest(operation.as_bytes()));
                key = home
                    .join("rentals")
                    .join(format!("pending-{digest}.creator.pem"));
            }
            Ok(Rental {
                cert: home.join("rentals").join(format!("{id}.pem")),
                id,
                asked: r.get(1)?,
                hub_name: r.get(2)?,
                state: r.get(3)?,
                hub: r.get(4)?,
                address: r.get(5)?,
                worker_id: r.get(6)?,
                key,
            })
        })
        .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
        .map_err(sql)?;
    Ok(Records {
        requests,
        products: newest,
        rentals,
    })
}

/// Every request in Go's numbering order: a run's number is its position by (created_at, id).
const REQUESTS: &str = "SELECT r.id, r.kind, r.package, r.entrypoint, r.package_release, r.state, r.hub, r.created_at,
    r.machine, r.worker,
    CASE WHEN r.kind IN ('download', 'upload') THEN r.payload END,
    e.machine_id, e.receipt, e.remote_cursor, e.outcome, e.cancel_requested,
    EXISTS (SELECT 1 FROM request_events WHERE request_id = r.id AND type = 'machine.api_v1'),
    EXISTS (SELECT 1 FROM request_events WHERE request_id = r.id AND type = 'machine.api_v1_sent'
        AND json_extract(payload, '$.machine') = e.machine_id),
    EXISTS (SELECT 1 FROM request_events WHERE request_id = r.id AND (type = 'client.machine_abandoned'
        OR type = 'client.machine_lost' AND json_extract(payload, '$.machine_id') = e.machine_id)),
    (SELECT at FROM request_events WHERE request_id = r.id AND type IN ('machine.api_v1', 'run.in_progress')
        ORDER BY seq LIMIT 1),
    (SELECT at FROM request_events WHERE request_id = r.id AND type IN ('run.completed', 'run.failed', 'run.canceled')
        ORDER BY seq DESC LIMIT 1),
    (SELECT payload FROM request_events WHERE request_id = r.id AND type IN ('run.failed', 'run.canceled')
        ORDER BY seq DESC LIMIT 1),
    x.directory, x.published_paths
    FROM requests r LEFT JOIN machine_executions e ON e.request_id = r.id
    LEFT JOIN request_output_exports x ON x.request_id = r.id
    ORDER BY r.created_at, r.id";

fn request(r: &Row) -> rusqlite::Result<Request> {
    let execution = match r.get::<_, Option<String>>(11)? {
        None => None,
        Some(machine) => Some(Execution {
            machine,
            receipt: bytes(r, 12)?,
            cursor: r.get::<_, i64>(13)?.max(0) as u64,
            outcome: bytes(r, 14)?,
            cancel_requested: r.get(15)?,
            accepted: r.get(16)?,
            sent: r.get(17)?,
            dropped: r.get(18)?,
        }),
    };
    let published: Option<String> = r.get(23)?;
    Ok(Request {
        id: r.get(0)?,
        kind: r.get(1)?,
        package: r.get(2)?,
        entrypoint: r.get(3)?,
        release: r.get(4)?,
        state: r.get(5)?,
        hub: r.get(6)?,
        created_at: r.get(7)?,
        machine: r.get(8)?,
        worker: r.get(9)?,
        operation: bytes(r, 10)?,
        execution,
        accepted_at: r.get(19)?,
        settled_at: r.get(20)?,
        ending: r.get(21)?,
        out_dir: r.get(22)?,
        published: published
            .and_then(|text| serde_json::from_str::<Vec<Option<String>>>(&text).ok())
            .unwrap_or_default()
            .into_iter()
            .flatten()
            .collect(),
    })
}

/// A text or blob column's bytes; NULL is none.
fn bytes(r: &Row, index: usize) -> rusqlite::Result<Vec<u8>> {
    let value = r.get_ref(index)?;
    Ok(match value {
        rusqlite::types::ValueRef::Null => vec![],
        value => value.as_bytes()?.to_vec(),
    })
}

/// A file read as JSON, `None` when it is absent.
pub fn json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// The DER of a PEM file's first certificate.
pub fn certificate(path: &Path) -> Option<Vec<u8>> {
    use base64::Engine as _;
    let text = std::fs::read_to_string(path).ok()?;
    let body = text
        .split("-----BEGIN CERTIFICATE-----")
        .nth(1)?
        .split("-----END CERTIFICATE-----")
        .next()?;
    let body: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD.decode(body).ok()
}
