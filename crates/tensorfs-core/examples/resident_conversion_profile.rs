//! Explicit local profiling harness; no remote IO, publication, or model residency claim.
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};
use tensorfs_core::{
    canon::{self, Value},
    catalog::Catalog,
    ingest::{carrier, convert, transaction},
    registry,
    store::{Progress, Store},
};

struct Probe {
    moved: AtomicU64,
    admitted: AtomicU64,
    pause: u64,
    marker: String,
}
impl Progress for Probe {
    fn moved(&self, n: u64) {
        let total = self.moved.fetch_add(n, Ordering::Relaxed) + n;
        if self.pause > 0 && total >= self.pause {
            fs::write(&self.marker, format!("{total}\n")).unwrap();
            loop {
                std::thread::park();
            }
        }
    }
    fn admitted(&self) {
        self.admitted.fetch_add(1, Ordering::Relaxed);
    }
}
fn counters(path: &str) -> BTreeMap<String, u64> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|line| {
            let (k, v) = line.split_once(':')?;
            Some((k.into(), v.split_whitespace().next()?.parse().ok()?))
        })
        .collect()
}
fn emit(value: Value) {
    println!("{}", String::from_utf8(canon::write(&value)).unwrap())
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args[0] == "span" {
        let raw = fs::read(&args[1]).unwrap();
        let header =
            carrier::parse_header(&raw[8..], raw.len() as u64, Some(args[2].parse().unwrap()))
                .unwrap();
        let tensor = header.get(&args[3]).unwrap();
        let (_, offset) = carrier::resolve(Path::new(&args[1]), &header, tensor).unwrap();
        emit(Value::obj(vec![
            ("dtype", Value::str(tensor.dtype.name())),
            (
                "shape",
                Value::arr(tensor.shape.iter().copied().map(Value::uint).collect()),
            ),
            ("offset", Value::uint(offset)),
            ("length", Value::uint(tensor.nbytes())),
        ]));
        return;
    }
    assert_eq!(args[0], "convert");
    let root = Path::new(&args[1]);
    let operation = &args[2];
    let mode = &args[3];
    let conv = convert::converter(&args[4]).unwrap();
    let mut names = Vec::new();
    let mut files = Vec::new();
    for arg in &args[5..] {
        let (component, path) = arg.split_once('=').unwrap();
        names.push(component.to_string());
        files.push(transaction::SourceFile {
            path: path.into(),
            header: carrier::read_header(Path::new(path)).unwrap(),
        });
    }
    let sources = files
        .iter()
        .enumerate()
        .map(|(file, source)| convert::Source::Carrier {
            component: names[file].clone(),
            file,
            header: &source.header,
        })
        .collect::<Vec<_>>();
    let target = convert::Target {
        components: names
            .iter()
            .map(|name| (name.clone(), "plain/1".into()))
            .collect(),
    };
    let specs = registry::seeds()
        .into_iter()
        .filter(|s| s.alias == "plain/1")
        .map(|s| (s.alias.to_string(), s.spec))
        .collect::<Vec<_>>();
    let mut plan = convert::plan(conv, &sources, &target, &specs, &Default::default()).unwrap();
    let order = plan
        .ops
        .iter()
        .map(|op| (op.component.clone(), op.out_key.clone()))
        .collect::<Vec<_>>();
    plan.apply_order(&order).unwrap();
    let base = if root.exists() {
        Store::open(root)
    } else {
        Store::init(root)
    }
    .unwrap();
    base.reap().unwrap();
    let (pause, marker) = mode
        .strip_prefix("pause:")
        .map(|s| {
            let (n, path) = s.split_once(':').unwrap();
            (n.parse().unwrap(), path.to_string())
        })
        .unwrap_or_default();
    let probe = Arc::new(Probe {
        moved: AtomicU64::new(0),
        admitted: AtomicU64::new(0),
        pause,
        marker,
    });
    let store = base.observed(probe.clone());
    let catalog = Catalog::open(root).unwrap();
    let _writer = catalog
        .resume_operation(operation, "profile", "resident")
        .unwrap();
    let before = counters("/proc/self/io");
    let began = Instant::now();
    let landed = files
        .iter()
        .enumerate()
        .map(|(i, _)| i == 0)
        .collect::<Vec<_>>();
    let carriers = if mode == "first" {
        transaction::Carriers::Landed(&landed)
    } else {
        transaction::Carriers::All
    };
    let (progress, outcome) = transaction::advance(
        &store,
        &plan,
        &files,
        &specs,
        &[],
        operation,
        carriers,
        None,
    )
    .unwrap();
    let elapsed = began.elapsed().as_nanos() as u64;
    let after = counters("/proc/self/io");
    let status = counters("/proc/self/status");
    let mut fields = vec![
        ("wall_ns", Value::uint(elapsed)),
        ("peak_rss_bytes", Value::uint(status["VmHWM"] * 1024)),
        (
            "input_bytes",
            Value::uint(files.iter().map(|f| f.header.declared_bytes()).sum()),
        ),
        (
            "roles_converted",
            Value::uint(progress.converted_roles as u64),
        ),
        ("roles_resumed", Value::uint(progress.resumed_roles as u64)),
        ("deferred_ops", Value::uint(progress.deferred_ops as u64)),
        ("payload_written", Value::uint(progress.written)),
        ("payload_hashed", Value::uint(progress.hashed)),
        ("deduped", Value::uint(progress.deduped)),
        (
            "store_moved",
            Value::uint(probe.moved.load(Ordering::Relaxed)),
        ),
        (
            "objects_admitted",
            Value::uint(probe.admitted.load(Ordering::Relaxed)),
        ),
    ];
    let mut fields = fields
        .drain(..)
        .map(|(key, value)| (key.to_string(), value))
        .collect::<Vec<_>>();
    for (key, value) in after {
        let old = *before.get(&key).unwrap_or(&0);
        fields.push((key, Value::uint(value.saturating_sub(old))));
    }
    if let Some(outcome) = outcome {
        fields.push(("manifest".into(), Value::str(outcome.manifest_ref.sha256)));
    }
    emit(Value::map(fields));
}
