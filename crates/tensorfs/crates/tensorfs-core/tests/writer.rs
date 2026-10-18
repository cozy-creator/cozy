//! The v2 writer and checkpoint reads through real files and processes: a tiny HF SDXL imported
//! and read back bit-identical; lane-to-lane sharing by `copy` and a base; SIGKILL mid-write then
//! resume; children forked while writers take the fence; the layout against the planner; and a
//! merged read plan filled with O_DIRECT.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use tensorfs_core::format::layout::{self, Item, ALIGN, STANDALONE};
use tensorfs_core::format::{decode, Body, Dtype, Encoding, Location, ObjectRef, PartBytes, Value};
use tensorfs_core::import::{self, source::SourceSet, Access, Request};
use tensorfs_core::store2::{fault, Checkpoint, Store};
use tensorfs_core::writer::{Data, PartData, Source, Writer};

const MIB: usize = 1 << 20;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("writer-{name}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }

    fn store(&self, name: &str) -> Store {
        Store::open(self.0.join(name)).unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Deterministic noise: equal only where the seed is.
fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(n);
    out
}

fn plain(w: &mut Writer, component: &str, key: &str, bytes: &[u8]) {
    let shape = [bytes.len() as u64 / 2];
    w.put(
        component,
        key,
        &Dtype::BF16,
        &shape,
        Data::Plain(Source::Read(&mut &bytes[..])),
    )
    .unwrap();
}

fn blobs(store: &Store) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for top in fs::read_dir(store.root().join("blobs")).unwrap().flatten() {
        for shard in fs::read_dir(top.path()).unwrap().flatten() {
            for blob in fs::read_dir(shard.path()).unwrap().flatten() {
                found.insert(blob.file_name().to_string_lossy().into_owned());
            }
        }
    }
    found
}

/// The checkpoint's layout is what the planner makes of its parts in header order.
fn assert_layout(ck: &Checkpoint) {
    let mut parts: Vec<(&str, &str, &str, Location)> = Vec::new();
    for (component, t) in ck.tensors() {
        for (role, at) in t.locations() {
            parts.push((component, &t.key, role, at));
        }
    }
    let items: Vec<Item> = parts
        .iter()
        .map(|(component, key, role, at)| Item {
            component,
            key,
            role,
            length: at.length,
        })
        .collect();
    let plan = layout::plan(&items);
    let lengths: Vec<u64> = ck.header().chunks.iter().map(|c| c.length).collect();
    assert_eq!(plan.chunks, lengths, "chunk lengths");
    let written: Vec<Location> = parts.iter().map(|p| p.3).collect();
    assert_eq!(plan.places, written, "part locations");
}

/// The checkpoint's (component, key) pairs in header order.
fn header_order(ck: &Checkpoint) -> Vec<(String, String)> {
    ck.tensors()
        .map(|(c, t)| (c.to_owned(), t.key.clone()))
        .collect()
}

// --------------------------------------------------------------- a tiny HF SDXL, end to end

/// network: Hugging Face.
#[test]
fn a_tiny_hf_sdxl_imports_and_reads_back_bit_identical() {
    let scratch = Scratch::new("tiny-sdxl");
    let store = scratch.store("store");
    let sources = ["hf://hf-internal-testing/tiny-sdxl-pipe".to_string()];
    let access = Access::default();
    let request = Request {
        sources: &sources,
        converter: None,
        variant: None,
        access: &access,
        tmp: &store.root().join("tmp"),
        progress: &|_, _, _| {},
        stop: None,
    };
    let id = store.import(&request, Some("import-tiny-sdxl")).unwrap();
    assert_eq!(
        store.get_pin("import-tiny-sdxl").unwrap().unwrap().roots,
        std::slice::from_ref(&id)
    );
    assert!(!store.root().join("tmp/writers/import-tiny-sdxl").exists());
    let ck = store.checkpoint(&id).unwrap();
    let meta = &ck.header().metadata;
    assert_eq!(meta["normalized"], Value::Bool(true));
    let Value::List(pinned) = &meta["sources"] else {
        panic!("{meta:?}")
    };
    let pinned = pinned[0].as_str().unwrap();
    assert!(pinned.starts_with("hf://hf-internal-testing/tiny-sdxl-pipe@"));

    // The repository's own files, landed apart from the import.
    let listing = import::remote::list(&import::remote::parse(pinned).unwrap(), &access).unwrap();
    let members =
        import::remote::select(listing, None, &|m| import::remote::read_small(m, &access)).unwrap();
    let landed: Vec<(String, PathBuf)> = members
        .iter()
        .map(|m| {
            let dest = scratch.0.join("check").join(&m.path);
            (
                m.path.clone(),
                import::remote::land(m, &dest, &access).unwrap(),
            )
        })
        .collect();
    let set = SourceSet::scan(&landed).unwrap();
    let mut checked = 0;
    for g in &set.groups {
        for (key, t) in &g.tensors {
            let mut want = vec![0; t.len as usize];
            let file = File::open(&set.files[t.file]).unwrap();
            file.read_exact_at(&mut want, t.start).unwrap();
            let tensor = ck.tensor(&g.folder, key).unwrap();
            assert_eq!(
                (&tensor.dtype, &tensor.shape),
                (&t.dtype, &t.shape),
                "{key}"
            );
            assert!(
                ck.read(&g.folder, key).unwrap() == want,
                "{}/{key}",
                g.folder
            );
            checked += 1;
        }
    }
    assert_eq!(checked, ck.tensors().count());
    assert!(checked > 100, "{checked} tensors");
    for p in &set.plain {
        if let Ok(bytes) = ck.file(&p.path) {
            assert_eq!(bytes, set.read_plain(p).unwrap(), "{}", p.path);
        }
    }
    for file in [
        "model_index.json",
        "unet/config.json",
        "tokenizer/vocab.json",
    ] {
        assert!(ck.manifest().files.contains_key(file), "{file}");
    }
    assert_layout(&ck);
    // Reimporting resumes nothing and lands on the same checkpoint.
    assert_eq!(store.import(&request, None).unwrap(), id);
}

// --------------------------------------------------------------- lanes share large parts

/// Lane A as `(component, key, bytes)`.
fn lane_a() -> Vec<(&'static str, &'static str, Vec<u8>)> {
    vec![
        ("transformer", "blocks.0.attn.weight", noise(1, 70 * MIB)),
        ("transformer", "blocks.0.norm.weight", noise(2, 3000)),
        (
            "transformer",
            "blocks.1.attn.weight",
            noise(3, STANDALONE as usize),
        ),
        ("transformer", "blocks.1.mlp.weight", noise(4, 20 * MIB)),
        ("transformer", "blocks.1.norm.weight", noise(5, 5 * MIB)),
        ("transformer", "blocks.2.norm.weight", noise(6, 9 * MIB)),
        ("transformer", "blocks.10.norm.weight", noise(7, 12 * MIB)),
        ("vae", "decoder.conv.weight", noise(8, 2 * MIB)),
        ("vae", "encoder.conv.weight", noise(9, 17 * MIB)),
    ]
}

fn chunk_set(ck: &Checkpoint) -> BTreeSet<ObjectRef> {
    ck.header().chunks.iter().cloned().collect()
}

#[test]
fn copy_and_a_base_share_the_large_parts_objects() {
    let scratch = Scratch::new("lanes");
    let store = scratch.store("store");
    let a_tensors = lane_a();
    let mut w = store.writer(None, None).unwrap();
    for (component, key, bytes) in &a_tensors {
        plain(&mut w, component, key, bytes);
    }
    w.file("transformer/config.json", b"{\"layers\":3}")
        .unwrap();
    let a = store.checkpoint(&w.commit().unwrap()).unwrap();
    assert_layout(&a);
    for (component, key, bytes) in &a_tensors {
        assert!(
            a.read(component, key).unwrap() == *bytes,
            "{component}/{key}"
        );
    }

    // Lane B: three large parts and two small ones copied, one tensor encoded, one re-put.
    let before = blobs(&store);
    let mut w = store.writer(None, None).unwrap();
    let copied_large = [
        ("transformer", "blocks.0.attn.weight"),
        ("transformer", "blocks.1.attn.weight"),
        ("vae", "encoder.conv.weight"),
    ];
    let copied_small = [
        ("transformer", "blocks.0.norm.weight"),
        ("vae", "decoder.conv.weight"),
    ];
    for (component, key) in copied_large.iter().chain(&copied_small) {
        w.copy(component, key, &a).unwrap();
    }
    let (rows, cols) = (4096u64, 5120u64);
    let weight = noise(10, (rows * cols) as usize);
    let scale: Vec<u8> = (0..rows)
        .flat_map(|r| (1.0f32 / (r + 1) as f32).to_le_bytes())
        .collect();
    let encoding = Encoding::fp8_rowwise(0);
    let (mut weight_src, mut scale_src) = (&weight[..], &scale[..]);
    let parts = vec![
        PartData {
            role: "weight".into(),
            dtype: Dtype::F8_E4M3,
            shape: vec![rows, cols],
            data: Source::Read(&mut weight_src),
        },
        PartData {
            role: "scale".into(),
            dtype: Dtype::F32,
            shape: vec![rows],
            data: Source::Read(&mut scale_src),
        },
    ];
    let encoded = Data::Encoded {
        encoding: encoding.clone(),
        parts,
    };
    w.put(
        "transformer",
        "blocks.1.mlp.weight",
        &Dtype::BF16,
        &[rows, cols],
        encoded,
    )
    .unwrap();
    let renorm = noise(11, 5 * MIB);
    plain(&mut w, "transformer", "blocks.1.norm.weight", &renorm);
    w.file("transformer/config.json", b"{\"layers\":3}")
        .unwrap();
    let b = store.checkpoint(&w.commit().unwrap()).unwrap();
    assert_layout(&b);

    // Design §3's accounting: the share of B's chunk bytes that are objects of A.
    let (a_chunks, b_chunks) = (chunk_set(&a), chunk_set(&b));
    let shared: BTreeSet<&ObjectRef> = b_chunks.intersection(&a_chunks).collect();
    let shared_bytes: u64 = shared.iter().map(|o| o.length).sum();
    let large_bytes: u64 = copied_large
        .iter()
        .map(|(c, k)| match &a.tensor(c, k).unwrap().body {
            Body::Data(at) => at.length,
            _ => unreachable!(),
        })
        .sum();
    // The vae pack holds `decoder.conv.weight` alone in both lanes: re-packed, it is the same
    // object. Every other shared object is a copied large part's.
    let vae_pack = 2 * MIB as u64;
    assert_eq!(shared_bytes, large_bytes + vae_pack);
    assert_eq!(
        shared.len(),
        2 + 1 + 1 + 1,
        "70 MiB on the grid is two objects"
    );
    let b_bytes: u64 = b_chunks.iter().map(|o| o.length).sum();
    println!("lane B shares {shared_bytes} of {b_bytes} bytes with A");
    // B wrote only its own new chunks, its header and its manifest (the config is A's object).
    let fresh = blobs(&store).difference(&before).count();
    assert_eq!(fresh, b_chunks.difference(&a_chunks).count() + 2);
    for (component, key) in copied_large.iter().chain(&copied_small) {
        let want = &a_tensors
            .iter()
            .find(|t| (t.0, t.1) == (*component, *key))
            .unwrap()
            .2;
        assert!(
            b.read(component, key).unwrap() == *want,
            "{component}/{key}"
        );
    }
    let (matrix, column) = ([rows, cols], [rows]);
    let views = BTreeMap::from([
        (
            "weight",
            PartBytes {
                dtype: &Dtype::F8_E4M3,
                shape: &matrix,
                bytes: &weight,
            },
        ),
        (
            "scale",
            PartBytes {
                dtype: &Dtype::F32,
                shape: &column,
                bytes: &scale,
            },
        ),
    ]);
    let logical = decode(&Dtype::BF16, &[rows, cols], &encoding, &views).unwrap();
    assert!(b.read("transformer", "blocks.1.mlp.weight").unwrap() == logical);
    let mut buf = vec![0; logical.len()];
    b.read_into("transformer", "blocks.1.mlp.weight", &mut buf)
        .unwrap();
    assert!(buf == logical);
    assert!(b.read("transformer", "blocks.1.norm.weight").unwrap() == renorm);

    // Lane C from base A: one tensor replaced, everything else carried; the unchanged large
    // parts keep A's objects.
    let replaced = noise(12, 20 * MIB);
    let mut w = store.writer(Some(a.id()), Some("lane-c")).unwrap();
    plain(&mut w, "transformer", "blocks.1.mlp.weight", &replaced);
    assert!(w.has("transformer", "blocks.1.mlp.weight"));
    assert!(
        !w.has("transformer", "blocks.0.attn.weight"),
        "base tensors are not written"
    );
    let c = store.checkpoint(&w.commit().unwrap()).unwrap();
    assert_layout(&c);
    assert_eq!(
        c.manifest().files["transformer/config.json"],
        a.manifest().files["transformer/config.json"]
    );
    for (component, key, bytes) in &a_tensors {
        let want = if *key == "blocks.1.mlp.weight" {
            &replaced
        } else {
            bytes
        };
        assert!(
            c.read(component, key).unwrap() == *want,
            "{component}/{key}"
        );
    }
    // Only the replaced large part's object differs: every pack and other large part is A's.
    let c_chunks = chunk_set(&c);
    let (only_c, only_a): (Vec<_>, Vec<_>) = (
        c_chunks.difference(&a_chunks).collect(),
        a_chunks.difference(&c_chunks).collect(),
    );
    assert_eq!((only_c.len(), only_a.len()), (1, 1));
    assert_eq!(only_c[0].length, 20 * MIB as u64);
    assert_eq!(
        store.get_pin("lane-c").unwrap().unwrap().roots,
        [c.id().clone()]
    );

    // An explicit execution order is the planner's given order.
    let order = [
        ("transformer", "blocks.10.norm.weight"),
        ("transformer", "blocks.1.norm.weight"),
    ];
    let mut w = store.writer(Some(a.id()), None).unwrap();
    w.set_order(&order);
    let d = store.checkpoint(&w.commit().unwrap()).unwrap();
    assert_layout(&d);
    let head: Vec<(String, String)> = (order.iter())
        .map(|(c, k)| (c.to_string(), k.to_string()))
        .collect();
    let d_order = header_order(&d);
    assert_eq!(
        d_order[..2],
        head[..],
        "set_order's tensors come first, in its order"
    );
    let others: Vec<_> = (header_order(&a).into_iter())
        .filter(|t| !head.contains(t))
        .collect();
    assert_eq!(d_order[2..], others[..], "the rest keep the base's order");
    assert_ne!(d.header().chunks, a.header().chunks);
}

// --------------------------------------------------------------- the read plan, O_DIRECT

/// Objects stored elsewhere (another store, an origin) are referenced, never copied or required:
/// a large part on the grid by `Source::Reuse`, a file by `reuse_file`. A keyed writer resumes
/// them, and once the objects arrive the checkpoint reads back bit-identical.
#[test]
fn reused_parts_and_files_are_referenced_not_copied() {
    let scratch = Scratch::new("reuse");
    let (a, b) = (scratch.store("a"), scratch.store("b"));
    let big = noise(7, 80 * MIB);
    let mut w = a.writer(None, None).unwrap();
    plain(&mut w, "unet", "big.weight", &big);
    let vocab = a.put_bytes(b"{\"a\":0}").unwrap();
    let from = a.checkpoint(&w.commit().unwrap()).unwrap();
    let Body::Data(at) = &from.tensor("unet", "big.weight").unwrap().body else {
        panic!("plain");
    };
    let objects: Vec<ObjectRef> = (from.header().spans(at).iter())
        .map(|s| from.header().chunks[s.chunk].clone())
        .collect();
    assert_eq!(objects.len(), 2);

    let small = noise(8, MIB);
    let shape = [big.len() as u64 / 2];
    let mut w = b.writer(None, Some("migrate")).unwrap();
    let wrong = Source::Reuse(vec![objects[1].clone()]);
    assert!(w
        .put(
            "unet",
            "short",
            &Dtype::BF16,
            &[MIB as u64 / 2],
            Data::Plain(wrong)
        )
        .is_err());
    let reuse = Data::Plain(Source::Reuse(objects.clone()));
    w.put("unet", "big.weight", &Dtype::BF16, &shape, reuse)
        .unwrap();
    plain(&mut w, "unet", "small.weight", &small);
    w.reuse_file("tokenizer/vocab.json", &vocab).unwrap();
    drop(w);
    let mut w = b.writer(None, Some("migrate")).unwrap();
    assert!(w.has("unet", "big.weight") && w.has("unet", "small.weight"));
    let ck = b.checkpoint(&w.commit().unwrap()).unwrap();

    let here = blobs(&b);
    assert!(objects
        .iter()
        .chain([&vocab])
        .all(|o| !here.contains(&o.sha256)));
    assert!(objects.iter().all(|o| ck.header().chunks.contains(o)));
    assert_eq!(ck.manifest().files["tokenizer/vocab.json"], vocab);
    assert_eq!(ck.read("unet", "small.weight").unwrap(), small);
    for o in objects.iter().chain([&vocab]) {
        b.put(&a.read(o).unwrap()[..], Some(o)).unwrap();
    }
    assert!(ck.read("unet", "big.weight").unwrap() == big);
    assert_eq!(ck.file("tokenizer/vocab.json").unwrap(), b"{\"a\":0}");
}

#[test]
fn a_merged_plan_fills_a_buffer_with_o_direct() {
    let scratch = Scratch::new("plan");
    let store = scratch.store("store");
    let mut w = store.writer(None, None).unwrap();
    for (component, key, bytes) in lane_a() {
        plain(&mut w, component, key, &bytes);
    }
    let ck = store.checkpoint(&w.commit().unwrap()).unwrap();
    let wanted: Vec<(&str, &str)> = ck
        .tensors()
        .filter(|t| t.0 == "transformer")
        .map(|(c, t)| (c, t.key.as_str()))
        .collect();
    let plan = ck.plan(&wanted).unwrap();
    let touched: BTreeSet<usize> = plan.reads.iter().map(|r| r.chunk).collect();
    assert_eq!(touched.len(), plan.reads.len(), "one merged read per chunk");
    // Aligned memory for O_DIRECT.
    let mut memory = vec![0u8; plan.bytes as usize + ALIGN as usize];
    let skew = memory.as_ptr().align_offset(ALIGN as usize);
    let buffer = &mut memory[skew..skew + plan.bytes as usize];
    for read in &plan.reads {
        assert_eq!(
            (read.offset % ALIGN, read.length % ALIGN, read.dest % ALIGN),
            (0, 0, 0)
        );
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&ck.chunk_paths()[read.chunk])
            .unwrap();
        let want = read.length.min(read.object.length - read.offset) as usize;
        let at = read.dest as usize;
        let mut got = 0;
        while got < want {
            let window = &mut buffer[at + got..at + read.length as usize];
            match file.read_at(window, read.offset + got as u64).unwrap() {
                0 => break,
                n => got += n,
            }
        }
        assert_eq!(got, want);
    }
    for placed in &plan.parts {
        assert_eq!(placed.pieces.len(), 1, "a rule-written part is one piece");
        let piece = placed.pieces[0];
        let bytes = ck.read(&placed.component, &placed.key).unwrap();
        let got = &buffer[piece.dest as usize..(piece.dest + piece.length) as usize];
        assert!(got == bytes, "{}/{}", placed.component, placed.key);
    }
}

// --------------------------------------------------------------- crash, resume, fence

/// 20 small tensors (a journal flush after the 16th), one large one, three more and a file.
fn kill_model() -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = (0..20)
        .map(|i| (format!("blocks.{i}.norm.weight"), noise(100 + i, MIB)))
        .collect();
    out.push(("blocks.20.proj.weight".into(), noise(200, 150 * MIB)));
    out.extend((21..24).map(|i| (format!("blocks.{i}.norm.weight"), noise(100 + i, MIB))));
    out
}

/// Writes the kill model, skipping what the writer already has; returns how many it skipped.
fn write_kill_model(w: &mut Writer) -> usize {
    let mut skipped = 0;
    for (key, bytes) in kill_model() {
        if w.has("unet", &key) {
            skipped += 1;
            continue;
        }
        plain(w, "unet", &key, &bytes);
    }
    w.file("unet/config.json", b"{\"blocks\":24}").unwrap();
    skipped
}

#[derive(Serialize, Deserialize)]
struct Job {
    root: PathBuf,
    arm: Option<String>,
    work: Work,
}

#[derive(Serialize, Deserialize)]
enum Work {
    /// Writes the kill model under key "resume", until killed.
    Kill,
    /// `rounds` times: takes the fence of "fenced", proves it alone holds it, commits or drops.
    Fence { marker: PathBuf, rounds: usize },
}

#[test]
#[ignore = "a child process of the tests in this file"]
fn child() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let job: Job = serde_json::from_str(&line).unwrap();
    if let Some(point) = &job.arm {
        fault::arm(point);
    }
    let store = Store::open(&job.root).unwrap();
    match job.work {
        Work::Kill => {
            let mut w = store.writer(None, Some("resume")).unwrap();
            write_kill_model(&mut w);
            w.commit().unwrap();
        }
        Work::Fence { marker, rounds } => fenced(&store, &marker, rounds, "child"),
    }
    println!("DONE");
}

fn fenced(store: &Store, marker: &Path, rounds: usize, who: &str) {
    for i in 0..rounds {
        let mut w = store.writer(None, Some("fenced")).unwrap();
        // Created only by a writer holding the fence, and removed before it lets go.
        File::create_new(marker).unwrap_or_else(|e| panic!("{who} round {i}: {e}"));
        plain(
            &mut w,
            "unet",
            &format!("{who}.{i}"),
            &noise(i as u64, 4096),
        );
        fs::remove_file(marker).unwrap();
        match i % 2 {
            0 => drop(w.commit().unwrap()),
            _ => drop(w),
        }
    }
}

struct Child {
    process: std::process::Child,
    output: BufReader<ChildStdout>,
}

impl Child {
    fn spawn(job: &Job) -> Child {
        let mut process = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "child",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut input = process.stdin.take().unwrap();
        writeln!(input, "{}", serde_json::to_string(job).unwrap()).unwrap();
        // The pipe stays open, so an armed child waits at its kill point.
        std::mem::forget(input);
        let output = BufReader::new(process.stdout.take().unwrap());
        Child { process, output }
    }

    fn wait_for(&mut self, prefix: &str) -> String {
        loop {
            let mut line = String::new();
            assert_ne!(
                self.output.read_line(&mut line).unwrap(),
                0,
                "the child ended before {prefix}"
            );
            if let Some(at) = line.find(prefix) {
                return line[at..].trim().to_owned();
            }
        }
    }
}

impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

#[test]
fn a_writer_killed_mid_write_resumes_to_the_uninterrupted_checkpoint() {
    let scratch = Scratch::new("kill");
    let root = scratch.0.join("store");
    Store::open(&root).unwrap();
    let job = Job {
        root: root.clone(),
        arm: Some("writer.grid".into()),
        work: Work::Kill,
    };
    let mut child = Child::spawn(&job);
    // SIGKILL after the large tensor's first object, its predecessors partly journaled.
    assert_eq!(child.wait_for("FAULT"), "FAULT writer.grid");
    drop(child);

    let store = Store::open(&root).unwrap();
    let mut w = store.writer(None, Some("resume")).unwrap();
    let skipped = write_kill_model(&mut w);
    assert_eq!(
        skipped, 16,
        "the tensors journaled before the kill are skipped"
    );
    let resumed = w.commit().unwrap();
    assert!(!root.join("tmp/writers/resume").exists());
    assert_eq!(
        store.get_pin("resume").unwrap().unwrap().roots,
        std::slice::from_ref(&resumed)
    );

    let other = scratch.store("uninterrupted");
    let mut w = other.writer(None, None).unwrap();
    assert_eq!(write_kill_model(&mut w), 0);
    assert_eq!(
        w.commit().unwrap(),
        resumed,
        "the same checkpoint, byte for byte"
    );
    let ck = store.checkpoint(&resumed).unwrap();
    assert_layout(&ck);
    for (key, bytes) in kill_model() {
        assert!(ck.read("unet", &key).unwrap() == bytes, "{key}");
    }
}

/// TensorD forks executors while writers run. A child forked meanwhile holds a fence at most
/// until it execs: a writer may wait for it, never fails, and never shares the fence.
#[test]
fn children_forked_meanwhile_never_misfire_the_fence() {
    let scratch = Scratch::new("fence");
    let root = scratch.0.join("store");
    let store = Store::open(&root).unwrap();
    let marker = scratch.0.join("inside");
    let stop = std::sync::atomic::AtomicBool::new(false);
    let rounds = 40;
    /// Stops the forks however the scope ends, a panic included.
    struct Stop<'a>(&'a std::sync::atomic::AtomicBool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    std::thread::scope(|scope| {
        let _stop = Stop(&stop);
        let forks = scope.spawn(|| {
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                assert!(Command::new("true").status().unwrap().success());
                n += 1;
            }
            n
        });
        let job = Job {
            root: root.clone(),
            arm: None,
            work: Work::Fence {
                marker: marker.clone(),
                rounds,
            },
        };
        let mut child = Child::spawn(&job);
        let threads: Vec<_> = ["a", "b"]
            .map(|who| {
                let (store, marker) = (&store, &marker);
                scope.spawn(move || fenced(store, marker, rounds, who))
            })
            .into_iter()
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        child.wait_for("DONE");
        stop.store(true, Ordering::Relaxed);
        assert!(forks.join().unwrap() > 0);
    });
    // The last holder committed or left its journal; either way the fence is free.
    let w = store.writer(None, Some("fenced")).unwrap();
    w.abandon();
    assert!(!root.join("tmp/writers/fenced").exists());
}

// --------------------------------------------------------------- construction order (#624)

/// (component, key, roles) of a checkpoint, in header order.
fn layout_names(ck: &Checkpoint) -> Vec<(String, String, Vec<String>)> {
    (ck.tensors())
        .map(|(c, t)| {
            let roles = t
                .locations()
                .into_iter()
                .map(|(r, _)| r.to_owned())
                .collect();
            (c.to_owned(), t.key.clone(), roles)
        })
        .collect()
}

fn encoded(w: &mut Writer, component: &str, key: &str, roles: &[&str]) {
    let bytes: Vec<Vec<u8>> = roles
        .iter()
        .enumerate()
        .map(|(i, _)| noise(i as u64, 4))
        .collect();
    let mut readers: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    let parts = (roles.iter().zip(readers.iter_mut()))
        .map(|(role, data)| PartData {
            role: role.to_string(),
            dtype: Dtype::U8,
            shape: vec![4],
            data: Source::Read(data),
        })
        .collect();
    let encoding = Encoding {
        scheme: "acme-q".into(),
        params: Default::default(),
    };
    w.put(
        component,
        key,
        &Dtype::F32,
        &[2],
        Data::Encoded { encoding, parts },
    )
    .unwrap();
}

/// The header keeps the producer's order (no sorting, components by first appearance, parts as
/// put), a base's order survives a derive, and reordering changes the checkpoint's identity.
#[test]
fn the_header_keeps_the_producers_order() {
    let scratch = Scratch::new("order");
    let store = scratch.store("store");
    let names = |list: &[(&str, &str, &[&str])]| -> Vec<(String, String, Vec<String>)> {
        (list.iter())
            .map(|(c, k, r)| {
                (
                    c.to_string(),
                    k.to_string(),
                    r.iter().map(|r| r.to_string()).collect(),
                )
            })
            .collect()
    };
    let mut w = store.writer(None, None).unwrap();
    plain(&mut w, "vae", "z.weight", &noise(1, 8));
    encoded(&mut w, "unet", "up.10.weight", &["weight", "scale"]);
    plain(&mut w, "unet", "up.9.weight", &noise(2, 8));
    plain(&mut w, "vae", "a.weight", &noise(3, 8));
    let a = store.checkpoint(&w.commit().unwrap()).unwrap();
    let want: &[(&str, &str, &[&str])] = &[
        ("vae", "z.weight", &[""]),
        ("vae", "a.weight", &[""]),
        ("unet", "up.10.weight", &["weight", "scale"]),
        ("unet", "up.9.weight", &[""]),
    ];
    assert_eq!(layout_names(&a), names(want));
    assert_layout(&a);
    let reread = tensorfs_core::format::Header::from_cbor(&a.file("model.cozytensors").unwrap());
    assert_eq!(reread.unwrap(), *a.header());

    // The same tensors in another order: another header, another checkpoint.
    let mut w = store.writer(None, None).unwrap();
    plain(&mut w, "vae", "a.weight", &noise(3, 8));
    plain(&mut w, "vae", "z.weight", &noise(1, 8));
    encoded(&mut w, "unet", "up.10.weight", &["weight", "scale"]);
    plain(&mut w, "unet", "up.9.weight", &noise(2, 8));
    let b = store.checkpoint(&w.commit().unwrap()).unwrap();
    assert_ne!(
        b.manifest().header(),
        a.manifest().header(),
        "reordering two tensors changes the header digest"
    );
    assert_ne!(b.id(), a.id());

    // A derive keeps the base's order: a tensor written again keeps its place, a new one follows.
    let mut w = store.writer(Some(a.id()), None).unwrap();
    plain(&mut w, "extra", "x.weight", &noise(9, 8));
    plain(&mut w, "unet", "up.10.weight", &noise(4, 8));
    let c = store.checkpoint(&w.commit().unwrap()).unwrap();
    let want: &[(&str, &str, &[&str])] = &[
        ("vae", "z.weight", &[""]),
        ("vae", "a.weight", &[""]),
        ("unet", "up.10.weight", &[""]),
        ("unet", "up.9.weight", &[""]),
        ("extra", "x.weight", &[""]),
    ];
    assert_eq!(layout_names(&c), names(want));
    assert_layout(&c);
}

/// The owner's real pre-cut lanes (read-only): the reader refuses their headers as pre-cut; a
/// checkpoint the writer makes from them, tensor by tensor in their order, has their order, and
/// migrate's old decoder refuses its header. The bytes are one-byte stand-ins of an unknown dtype; the
/// order is under test, and the chunk counts the rule gives the real part lengths are reported.
#[test]
#[ignore = "reads the owner's ~/.tensorfs v1 lanes"]
fn real_v1_lanes_keep_their_order() {
    use tensorfs_core::header::Header as V1;
    let home = PathBuf::from(std::env::var_os("HOME").unwrap()).join(".tensorfs");
    let blob = |sha: &str| {
        fs::read(
            home.join("blobs")
                .join(&sha[..2])
                .join(&sha[2..4])
                .join(sha),
        )
        .unwrap()
    };
    let scratch = Scratch::new("v1");
    let store = scratch.store("store");
    // (lane, manifest prefix, chunk count design §3 measured by the rule)
    for (lane, prefix, chunks) in [
        ("sdxl-int8", "3eb5674846e8", 203),
        ("sdxl-fp8", "d33267e269ae", 201),
        ("qwen-int8", "e7c79ee942b8", 544),
        ("anima-fp8", "9bbe72e3b132", 173),
        ("h3-fp8", "302eec1e11fd", 2138),
    ] {
        let dir = home
            .join("manifests")
            .join(&prefix[..2])
            .join(&prefix[2..4]);
        let manifest = (fs::read_dir(&dir).unwrap().flatten())
            .find(|e| e.file_name().to_string_lossy().starts_with(prefix))
            .unwrap()
            .path();
        let doc: serde_json::Value = serde_json::from_slice(&fs::read(manifest).unwrap()).unwrap();
        let entry = (doc["entries"].as_array().unwrap().iter())
            .find(|e| e["kind"] == "cozytensors")
            .unwrap();
        let old = blob(entry["blob"]["sha256"].as_str().unwrap());
        let err = tensorfs_core::format::Header::from_cbor(&old).unwrap_err();
        assert_eq!(err.code(), "pre_cut", "{lane}: {err}");
        let v1 = V1::parse(&old).unwrap();
        let mut w = store.writer(None, None).unwrap();
        let mut want = Vec::new();
        let mut items = Vec::new();
        for (component, tensors) in &v1.components {
            for (key, t) in tensors {
                let roles: Vec<&str> = t.parts.iter().map(|(r, _)| r.as_str()).collect();
                let stand_in = Dtype::Other("V1".into());
                let data: Vec<Vec<u8>> = roles.iter().map(|_| vec![7u8]).collect();
                let mut readers: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
                let parts = (roles.iter().zip(readers.iter_mut()))
                    .map(|(role, data)| PartData {
                        role: role.to_string(),
                        dtype: stand_in.clone(),
                        shape: vec![],
                        data: Source::Read(data),
                    })
                    .collect();
                let encoding = Encoding {
                    scheme: "v1".into(),
                    params: Default::default(),
                };
                w.put(
                    component,
                    key,
                    &stand_in,
                    &[],
                    Data::Encoded { encoding, parts },
                )
                .unwrap();
                want.push((
                    component.clone(),
                    key.clone(),
                    roles.iter().map(|r| r.to_string()).collect::<Vec<_>>(),
                ));
                for (role, p) in &t.parts {
                    items.push((
                        component.clone(),
                        key.clone(),
                        role.clone(),
                        p.nbytes(key).unwrap(),
                    ));
                }
            }
        }
        let ck = store.checkpoint(&w.commit().unwrap()).unwrap();
        assert_eq!(layout_names(&ck), want, "{lane}: v2 order is v1 order");
        let new = ck.file("model.cozytensors").unwrap();
        assert!(tensorfs_core::migrate::v1::header(&new).is_err(), "{lane}");
        let items: Vec<Item> = (items.iter())
            .map(|(c, k, r, n)| Item {
                component: c,
                key: k,
                role: r,
                length: *n,
            })
            .collect();
        // The design measured its counts in natural key order; v1 order closes packs at other
        // block boundaries, so the count is reported, not asserted.
        let planned = layout::plan(&items).chunks.len();
        println!("{lane}: {} tensors, v2 order = v1 order; {planned} chunks in v1 order (design §3, natural order: {chunks})", want.len());
    }
}
