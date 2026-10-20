//! The writer and checkpoint reads through real files and processes: a tiny HF SDXL imported
//! and read back bit-identical; variant-to-variant sharing by `copy_as` and a base; a
//! two-variant repo's tree, closure and GC; plain folders written, opened, admitted and
//! snapshotted; SIGKILL mid-write then resume; two writers and a collector at once; children
//! forked while writers take the fence; the layout against the planner; and a merged read plan
//! filled with O_DIRECT.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tensorfs::format::layout::{self, Item, ALIGN, STANDALONE};
use tensorfs::format::{
    decode, Body, Config, ConfigValue, Dtype, Encoding, Location, ObjectRef, PartBytes, Repo, Tree,
    Value, CHUNKS_DIR,
};
use tensorfs::import::{self, source::SourceSet, Access, Request};
use tensorfs::store::{fault, Checkpoint, Store};
use tensorfs::writer::{Data, PartData, Source, Writer};

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
    let data = Data::Plain(Source::Read(&mut &bytes[..]));
    w.get(component).put(key, &Dtype::BF16, &shape, data).unwrap();
}

fn config(json: &str) -> Config {
    serde_json::from_str(json).unwrap()
}

#[test]
fn logical_config_edits_preserve_imported_filename_for_explicit_export() {
    let scratch = Scratch::new("config-filename");
    let store = scratch.store("store");
    let mut writer = store.writer(None, None).unwrap();
    writer.component("tokenizer", "transformers.Tokenizer", config("{\"version\":1}"), Some("tokenizer_config.json")).unwrap();
    writer.get("tokenizer").asset("vocab.txt", b"hello\n").unwrap();
    let base = store.checkpoint(&writer.commit().unwrap()).unwrap();

    let mut edited = store.writer(Some(&base), None).unwrap();
    edited.component("tokenizer", "transformers.Tokenizer", config("{\"version\":2}"), None).unwrap();
    edited.component("tokenizer", "transformers.Tokenizer", config("{\"version\":3}"), None).unwrap();
    let checkpoint = store.checkpoint(&edited.commit().unwrap()).unwrap();
    let exported = checkpoint.export_component_files("tokenizer", &scratch.0.join("support")).unwrap();
    assert_eq!(fs::read(exported.join("vocab.txt")).unwrap(), b"hello\n");
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&fs::read(exported.join("tokenizer_config.json")).unwrap()).unwrap(), serde_json::json!({"version": 3}));
    assert!(!exported.join("config.json").exists());
    assert!(!exported.join(CHUNKS_DIR).exists());

    // The raw importer retains an explicit way to replace the known source filename.
    let mut reset = store.writer(Some(&checkpoint), None).unwrap();
    reset.component("tokenizer", "transformers.Tokenizer", Config::new(), Some("config.json")).unwrap();
    let checkpoint = store.checkpoint(&reset.commit().unwrap()).unwrap();
    assert_eq!(checkpoint.component("tokenizer").unwrap().config_file, None);
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

/// GC reports allocated storage only when unlinking the last, unopened inode name.
/// Capture these facts before collection, without keeping a descriptor to the candidate.
fn gc_allocation(store: &Store, digest: &str) -> u64 {
    let object = ObjectRef { sha256: digest.to_owned(), length: 0 };
    let path = store.path(&object).unwrap();
    let metadata = fs::metadata(&path).unwrap();
    if platform::durable::links(&path).unwrap() != 1 { return 0; }
    #[cfg(target_os = "linux")]
    {
        let id = platform::durable::FileId::at(&path).unwrap();
        if fs::read_dir("/proc/self/fd").unwrap().flatten()
            .any(|fd| platform::durable::FileId::at(&fd.path()).ok() == Some(id)) { return 0; }
    }
    platform::durable::allocated(&metadata)
}

/// Account only for the normal Trees written by native retention, including superseded
/// roots not yet swept. Every raw entry must name a known fixture header/payload exactly.
fn retention_trees(store: &Store, digests: &BTreeSet<String>, known: &BTreeSet<ObjectRef>) -> BTreeMap<String, u64> {
    digests.iter().map(|digest| {
        let object = ObjectRef { sha256: digest.clone(), length: 0 };
        let bytes = fs::read(store.path(&object).unwrap()).unwrap();
        assert_eq!(&ObjectRef::of(&bytes).sha256, digest, "metadata digest");
        let tree = Tree::from_json(&bytes).expect("only known retention metadata may accompany the payload");
        assert_eq!(bytes, tree.to_json().unwrap(), "canonical retention Tree");
        for (path, object) in &tree.files {
            assert_eq!(path, &format!("held/{}", object.sha256), "retention raw-entry name");
            assert!(known.contains(object), "retention Tree names an unexpected object: {object:?}");
        }
        (digest.clone(), gc_allocation(store, digest))
    }).collect()
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
    let (id, files) = store.import(&request, Some("import-tiny-sdxl")).unwrap();
    assert!(store.get_pin("import-tiny-sdxl").unwrap().unwrap().roots.contains(&id));
    assert!(!store.root().join("tmp/writers/import-tiny-sdxl").exists());
    let ck = store.checkpoint(&id).unwrap();
    let meta = &ck.header().metadata;
    let pinned = match meta.get("source").or(meta.get("sources")) {
        Some(Value::Str(one)) => one.as_str(),
        Some(Value::List(list)) => list[0].as_str().unwrap(),
        _ => panic!("{meta:?}"),
    };
    assert!(pinned.starts_with("hf://hf-internal-testing/tiny-sdxl-pipe@"));

    // The repository's own files, landed apart from the import.
    let listing = import::remote::list(&import::remote::parse(pinned).unwrap(), &access).unwrap();
    let members = import::remote::select(listing, None, import::remote::Files::All, &|m| {
        import::remote::read_small(m, &access)
    })
    .unwrap();
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
    // Every source file is a repo file, the pipeline, a component's config or its asset; repo
    // files and assets are byte-exact.
    let asset = |path: &str| {
        let (c, name) = path.split_once('/')?;
        ck.component(c).ok()?.assets.get(name)?;
        ck.asset(c, name).ok()
    };
    for p in &set.plain {
        let kept = files.get(&p.path).map(|o| store.read(o).unwrap()).or_else(|| asset(&p.path));
        if let Some(bytes) = kept {
            assert_eq!(bytes, set.read_plain(p).unwrap(), "{}", p.path);
        }
    }
    let config_of = |c: &str| ck.component(c).is_ok_and(|c| !c.config.is_empty());
    assert!(files.contains_key("model_index.json") || ck.pipeline().is_some());
    assert!(files.contains_key("unet/config.json") || config_of("unet"));
    assert!(files.contains_key("tokenizer/vocab.json") || asset("tokenizer/vocab.json").is_some());
    assert_layout(&ck);
    // Reimporting resumes nothing and lands on the same checkpoint.
    assert_eq!(store.import(&request, None).unwrap().0, id);
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
    w.component("transformer", "diffusers.Transformer", config(r#"{"layers":3}"#), None)
        .unwrap();
    for (component, key, bytes) in &a_tensors {
        plain(&mut w, component, key, bytes);
    }
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
        w.get(component).copy_as(key, &a, component, key).unwrap();
    }
    let (rows, cols) = (4096u64, 5120u64);
    let weight = noise(10, (rows * cols) as usize);
    let scale: Vec<u8> = (0..rows)
        .flat_map(|r| (1.0f32 / (r + 1) as f32).to_le_bytes())
        .collect();
    let encoding = Encoding::fp8_rowwise(0, &Dtype::F32);
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
    (w.get("transformer").put("blocks.1.mlp.weight", &Dtype::BF16, &[rows, cols], encoded))
        .unwrap();
    let renorm = noise(11, 5 * MIB);
    plain(&mut w, "transformer", "blocks.1.norm.weight", &renorm);
    w.component("transformer", "diffusers.Transformer", config(r#"{"layers":3}"#), None)
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
    // B wrote its own new chunks and header, plus native hold Trees. Preserve exact
    // payload/header identities and reject every extra object not of that known metadata form.
    let fresh: BTreeSet<String> = blobs(&store).difference(&before).cloned().collect();
    let written: BTreeSet<String> = b_chunks.difference(&a_chunks).map(|o| o.sha256.clone())
        .chain(std::iter::once(b.id().sha256.clone())).collect();
    assert_eq!(written.len(), b_chunks.difference(&a_chunks).count() + 1);
    assert!(written.is_subset(&fresh), "every expected new payload/header was admitted");
    let extra: BTreeSet<String> = fresh.difference(&written).cloned().collect();
    let known: BTreeSet<ObjectRef> = [a.id().clone(), b.id().clone()].into_iter()
        .chain(a.header().objects().cloned()).chain(b.header().objects().cloned()).collect();
    let metadata = retention_trees(&store, &extra, &known);
    assert_eq!(fresh.len(), written.len() + metadata.len());
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
    assert!(b.read("transformer", "blocks.1.norm.weight").unwrap() == renorm);

    // Lane C from base A: one tensor replaced, everything else carried; the unchanged large
    // parts keep A's objects.
    let replaced = noise(12, 20 * MIB);
    let mut w = store.writer(Some(&a), Some("lane-c")).unwrap();
    plain(&mut w, "transformer", "blocks.1.mlp.weight", &replaced);
    assert!(w.has("transformer", "blocks.1.mlp.weight"));
    assert!(
        !w.has("transformer", "blocks.0.attn.weight"),
        "base tensors are not written"
    );
    let c = store.checkpoint(&w.commit().unwrap()).unwrap();
    assert_layout(&c);
    let (ct, at) = (c.component("transformer").unwrap(), a.component("transformer").unwrap());
    assert_eq!((&ct.class, &ct.config), (&at.class, &at.config), "the base's config carries");
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
    let mut w = store.writer(Some(&a), None).unwrap();
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
/// a large part on the grid by `Source::Reuse`, an asset by `reuse_asset`. A keyed writer resumes
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
    let short = w.get("unet").put("short", &Dtype::BF16, &[MIB as u64 / 2], Data::Plain(wrong));
    assert!(short.is_err());
    let reuse = Data::Plain(Source::Reuse(objects.clone()));
    w.get("unet").put("big.weight", &Dtype::BF16, &shape, reuse).unwrap();
    plain(&mut w, "unet", "small.weight", &small);
    w.get("tokenizer").reuse_asset("vocab.json", &vocab).unwrap();
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
    assert_eq!(ck.component("tokenizer").unwrap().assets["vocab.json"], vocab);
    assert_eq!(ck.read("unet", "small.weight").unwrap(), small);
    for o in objects.iter().chain([&vocab]) {
        b.put(&a.read(o).unwrap()[..], Some(o)).unwrap();
    }
    assert!(ck.read("unet", "big.weight").unwrap() == big);
    assert_eq!(ck.asset("tokenizer", "vocab.json").unwrap(), b"{\"a\":0}");
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
            .open(ck.chunk_path(read.chunk))
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
    w.component("unet", "diffusers.UNet2DConditionModel", config(r#"{"blocks":24}"#), None)
        .unwrap();
    let mut skipped = 0;
    for (key, bytes) in kill_model() {
        if w.has("unet", &key) {
            skipped += 1;
            continue;
        }
        plain(w, "unet", &key, &bytes);
    }
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
    /// Slowly writes [`overlap`]`(seed)` under a short grace; prints `HEADER <sha256> <length>`.
    Overlap { seed: u64 },
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
        Work::Overlap { seed } => {
            let mut w = store.with_grace(GRACE).writer(None, None).unwrap();
            for (key, bytes) in overlap(seed) {
                plain(&mut w, "unet", &key, &bytes);
                std::thread::sleep(GRACE / 2);
            }
            let header = w.commit().unwrap();
            println!("HEADER {} {}", header.sha256, header.length);
        }
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
    let data = Data::Encoded { encoding, parts };
    w.get(component).put(key, &Dtype::F32, &[2], data).unwrap();
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
    let reread = tensorfs::format::Header::from_cbor(&store.read(a.id()).unwrap());
    assert_eq!(reread.unwrap(), *a.header());

    // The same tensors in another order: another header, another checkpoint.
    let mut w = store.writer(None, None).unwrap();
    plain(&mut w, "vae", "a.weight", &noise(3, 8));
    plain(&mut w, "vae", "z.weight", &noise(1, 8));
    encoded(&mut w, "unet", "up.10.weight", &["weight", "scale"]);
    plain(&mut w, "unet", "up.9.weight", &noise(2, 8));
    let b = store.checkpoint(&w.commit().unwrap()).unwrap();
    assert_ne!(b.id(), a.id(), "reordering two tensors changes the header digest");

    // A derive keeps the base's order: a tensor written again keeps its place, a new one follows.
    let mut w = store.writer(Some(&a), None).unwrap();
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

// --------------------------------------------------------------- variants, trees and GC

/// A repo of two variants: `bf16` with a pipeline, configs (floats and nulls among them), an
/// asset and a config-only scheduler; `int8` from it as a base, one large part encoded. The tree
/// adds README and LICENSE. Its closure is the tree, its files and both headers' chunks and
/// assets, shared ones once; GC with no grace keeps all of it and sweeps a dropped checkpoint.
#[test]
fn a_two_variant_repo_shares_chunks_and_files_and_gc_keeps_its_closure() {
    let scratch = Scratch::new("variants");
    let store = scratch.store("store");
    let mut w = store.writer(None, None).unwrap();
    let index = r#"{"_class_name":"StableDiffusionXLPipeline","unet":["diffusers","UNet2DConditionModel"]}"#;
    w.pipeline("diffusers.StableDiffusionXLPipeline", config(index));
    let unet = r#"{"in_channels":4,"scaling_factor":0.13025,"act":null,"blocks":[1,2.5]}"#;
    w.component("unet", "diffusers.UNet2DConditionModel", config(unet), None).unwrap();
    let tensors = [("down.weight", noise(1, 20 * MIB)), ("mid.weight", noise(2, 18 * MIB)), ("norm.weight", noise(3, 4096))];
    for (key, bytes) in &tensors {
        plain(&mut w, "unet", key, bytes);
    }
    let vocab = br#"{"<pad>":0}"#;
    let tokenizer = r#"{"model_max_length":77}"#;
    (w.component("tokenizer", "transformers.CLIPTokenizer", config(tokenizer), Some("tokenizer_config.json")))
        .unwrap()
        .asset("vocab.json", vocab)
        .unwrap();
    let scheduler = r#"{"beta_end":0.012,"_class_name":"EulerDiscreteScheduler"}"#;
    w.component("scheduler", "diffusers.EulerDiscreteScheduler", config(scheduler), Some("scheduler_config.json"))
        .unwrap();
    let bf16 = store.checkpoint(&w.commit().unwrap()).unwrap();

    let mut w = store.writer(Some(&bf16), None).unwrap();
    let (rows, cols) = (2048u64, 4096u64);
    let (q, scale) = (noise(4, (rows * cols) as usize), noise(5, rows as usize * 4));
    let (mut q_src, mut scale_src) = (&q[..], &scale[..]);
    let parts = vec![
        PartData { role: "weight".into(), dtype: Dtype::I8, shape: vec![rows, cols], data: Source::Read(&mut q_src) },
        PartData { role: "scale".into(), dtype: Dtype::F32, shape: vec![rows], data: Source::Read(&mut scale_src) },
    ];
    let encoding = Encoding { scheme: "int8.rowwise".into(), params: Default::default() };
    let data = Data::Encoded { encoding, parts };
    w.get("unet").put("down.weight", &Dtype::BF16, &[rows, cols], data).unwrap();
    let int8 = store.checkpoint(&w.commit().unwrap()).unwrap();

    // Both variants carry everything but the re-put tensor; the other large part is one object.
    assert_eq!(int8.pipeline(), bf16.pipeline());
    assert_eq!(int8.components().len(), 3);
    assert_eq!(int8.component("scheduler").unwrap().config_file.as_deref(), Some("scheduler_config.json"));
    assert_eq!(int8.asset("tokenizer", "vocab.json").unwrap(), vocab);
    let c = bf16.component("unet").unwrap();
    assert_eq!(c.config["scaling_factor"], serde_json::json!(0.13025));
    assert_eq!(c.config["act"], ConfigValue::Null);
    assert_eq!(c.config["blocks"], serde_json::json!([1, 2.5]));
    let (a, b): (BTreeSet<ObjectRef>, BTreeSet<ObjectRef>) = (bf16.header().objects().cloned().collect(), int8.header().objects().cloned().collect());
    let shared: Vec<&ObjectRef> = a.intersection(&b).collect();
    assert!(shared.iter().any(|o| o.length == 18 * MIB as u64), "mid.weight's object is shared");
    assert!(shared.contains(&&bf16.component("tokenizer").unwrap().assets["vocab.json"]));

    let mut tree = Tree::default();
    tree.files.insert("README.md".into(), store.put_bytes(b"# sdxl\n").unwrap());
    tree.files.insert("LICENSE".into(), store.put_bytes(b"MIT\n").unwrap());
    tree.set_variant("bf16", bf16.id().clone());
    tree.set_variant("int8", int8.id().clone());
    let at = store.put_tree(&tree).unwrap();
    let named = |at: ObjectRef| move |_: Option<Repo>| {
        let mut repo = Repo::new("Acme/SDXL");
        repo.set_release("1.0", at.clone());
        Some(repo)
    };
    store.update_ref("acme/sdxl", named(at.clone())).unwrap();
    for ck in [&bf16, &int8] {
        assert!(store.unpin(&ck.id().sha256).unwrap(), "a keyless writer pins its header");
    }
    let mut w = store.writer(None, None).unwrap();
    plain(&mut w, "unet", "dropped.weight", &noise(9, 17 * MIB));
    let dropped = store.checkpoint(&w.commit().unwrap()).unwrap();
    store.unpin(&dropped.id().sha256).unwrap();

    let mut want: BTreeSet<ObjectRef> = tree.files.values().cloned().collect();
    want.insert(at.clone());
    want.extend(a.iter().chain(&b).cloned());
    let closure: BTreeSet<ObjectRef> = store.closure(&at).unwrap().into_iter().collect();
    assert_eq!(closure, want);
    let bytes: u64 = closure.iter().map(|o| o.length).sum();
    assert_eq!(store.usage().unwrap().refs, bytes, "shared objects count once");

    let gone: BTreeSet<ObjectRef> = std::iter::once(dropped.id().clone()).chain(dropped.header().objects().cloned()).collect();
    drop(dropped); // A live native owner must keep its payload even after the ordinary pin goes.
    let before = blobs(&store);
    let known: BTreeSet<ObjectRef> = closure.union(&gone).cloned().collect();
    let known_digests: BTreeSet<String> = known.iter().map(|o| o.sha256.clone()).collect();
    let extra: BTreeSet<String> = before.difference(&known_digests).cloned().collect();
    let metadata = retention_trees(&store, &extra, &known);
    let mut allocated = metadata.clone();
    allocated.extend(gone.iter().map(|o| (o.sha256.clone(), gc_allocation(&store, &o.sha256))));
    let collected = store.clone().with_grace(Duration::ZERO).gc().unwrap();
    let removed: BTreeSet<String> = before.difference(&blobs(&store)).cloned().collect();
    let gone_digests: BTreeSet<String> = gone.iter().map(|o| o.sha256.clone()).collect();
    assert!(gone_digests.is_subset(&removed), "the released checkpoint's exact payload is swept");
    let removed_metadata: BTreeSet<String> = removed.difference(&gone_digests).cloned().collect();
    assert!(removed_metadata.iter().all(|digest| metadata.contains_key(digest)), "GC removed an unexpected object");
    assert_eq!(collected.objects, gone.len() as u64 + removed_metadata.len() as u64);
    let physical: u64 = removed.iter().map(|digest| allocated[digest]).sum();
    assert_eq!(collected.bytes, physical, "physical allocation captured before unlink");
    for o in &gone {
        assert!(!store.path(o).unwrap().exists(), "{} is swept", o.sha256);
    }
    for o in &closure {
        assert!(store.verify(o).unwrap(), "{} is kept", o.sha256);
    }

    assert_eq!(store.variant(&at, Some("int8")).unwrap().id(), int8.id());
    let ck = store.variant(&at, Some("bf16")).unwrap();
    for (key, bytes) in &tensors {
        assert!(ck.read("unet", key).unwrap() == *bytes, "{key}");
    }
    assert!(store.variant(&at, Some("int8")).unwrap().missing(None).unwrap().is_empty());
}

/// Every file under `dir`, by relative path.
fn files_under(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in fs::read_dir(&at).unwrap().flatten() {
            let path = entry.path();
            match path.is_dir() {
                true => stack.push(path),
                false => {
                    let rel = path.strip_prefix(dir).unwrap().to_string_lossy().into_owned();
                    out.insert(rel, fs::read(&path).unwrap());
                }
            }
        }
    }
    out
}

/// A folder writer lays chunks and assets in `chunks/` and the header last; the folder opens as
/// a checkpoint, enters a store and comes back out byte for byte. A folder written on a stored
/// base is thin: it holds only its own new objects, reads the rest from the store, and enters
/// only a store that has them. A keyed folder writer resumes.
#[test]
fn a_plain_folder_round_trips_through_the_store() {
    let scratch = Scratch::new("folder");
    let dir = scratch.0.join("sdxl");
    let tensors = [("big.weight", noise(21, 70 * MIB)), ("small.weight", noise(22, 3 * MIB))];
    let mut w = Writer::folder(&dir, "bf16", None, false).unwrap();
    w.component("unet", "diffusers.UNet2DConditionModel", config(r#"{"in_channels":4}"#), None).unwrap();
    for (key, bytes) in &tensors {
        plain(&mut w, "unet", key, bytes);
    }
    let vocab = b"a\nb\n";
    (w.component("tokenizer", "transformers.CLIPTokenizer", Config::new(), None)).unwrap().asset("vocab.txt", vocab).unwrap();
    w.repo_file("README.md", b"# sdxl\n").unwrap();
    assert!(!dir.join("model.bf16.cozytensors").exists(), "the header comes last");
    let header = w.commit().unwrap();
    assert_eq!(ObjectRef::of(&fs::read(dir.join("model.bf16.cozytensors")).unwrap()), header);
    assert!(!dir.join(CHUNKS_DIR).join(".writing").exists());

    let ck = Checkpoint::folder(&dir, None, None).unwrap();
    assert_eq!(ck.id(), &header);
    for o in ck.header().objects() {
        assert_eq!(ObjectRef::of(&fs::read(dir.join(CHUNKS_DIR).join(&o.sha256)).unwrap()), *o);
    }
    for (key, bytes) in &tensors {
        assert!(ck.read("unet", key).unwrap() == *bytes, "{key}");
    }
    assert_eq!(ck.asset("tokenizer", "vocab.txt").unwrap(), vocab);

    let store = scratch.store("store");
    let tree = store.admit_folder(&dir).unwrap();
    let files: Vec<String> = store.tree(&tree).unwrap().files.into_keys().collect();
    assert_eq!(files, ["README.md", "model.bf16.cozytensors"]);
    let out = scratch.0.join("out");
    store.snapshot(&tree, &out, None).unwrap();
    assert!(files_under(&out) == files_under(&dir), "the snapshot is the folder, byte for byte");

    let base = store.variant(&tree, None).unwrap();
    let thin = scratch.0.join("thin");
    let mut w = Writer::folder(&thin, "int8", Some(&base), false).unwrap();
    plain(&mut w, "unet", "small.weight", &noise(23, 3 * MIB));
    let int8 = w.commit().unwrap();
    let own: Vec<String> = (fs::read_dir(thin.join(CHUNKS_DIR)).unwrap().flatten())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(own.len(), 1, "only the new pack: {own:?}");
    let alone = Checkpoint::folder(&thin, Some("int8"), None).unwrap();
    assert_eq!(alone.id(), &int8);
    assert_eq!(alone.missing(None).unwrap().len(), 3, "big.weight's two objects and the asset are the store's");
    let cached = Checkpoint::folder(&thin, Some("int8"), Some(&store)).unwrap();
    assert!(cached.read("unet", "big.weight").unwrap() == tensors[0].1);
    assert!(scratch.store("empty").admit_folder(&thin).is_err(), "a thin folder needs its objects somewhere");
    let thin_tree = store.admit_folder(&thin).unwrap();
    assert!(store.variant(&thin_tree, Some("int8")).unwrap().missing(None).unwrap().is_empty());

    let resumed = scratch.0.join("resumed");
    let mut w = Writer::folder(&resumed, "", None, true).unwrap();
    plain(&mut w, "unet", "big.weight", &tensors[0].1);
    drop(w);
    let mut w = Writer::folder(&resumed, "", None, true).unwrap();
    assert!(w.has("unet", "big.weight"));
    plain(&mut w, "unet", "small.weight", &tensors[1].1);
    w.commit().unwrap();
    let ck = Checkpoint::folder(&resumed, None, None).unwrap();
    assert!(ck.read("unet", "big.weight").unwrap() == tensors[0].1);
}

/// The grace two concurrent writers run under while a collector sweeps.
const GRACE: Duration = Duration::from_secs(6);

/// Ten tensors, slowly: a large part and a small one every seed shares, and its own.
fn overlap(seed: u64) -> Vec<(String, Vec<u8>)> {
    let mut out = vec![("shared.big".to_owned(), noise(500, 20 * MIB)), ("shared.small".to_owned(), noise(501, MIB))];
    out.extend((0..8).map(|i| (format!("own.{i}"), noise(seed * 100 + i, if i % 3 == 0 { 17 * MIB } else { MIB }))));
    out
}

/// Any process reads and writes the store: two writers in two processes write checkpoints that
/// share objects, each over several graces, while this one collects with that grace in a loop.
/// Nothing either writer refers to is taken, and both read back whole.
#[test]
fn two_writers_and_a_collector_lose_nothing() {
    let scratch = Scratch::new("overlap");
    let root = scratch.0.join("store");
    let store = Store::open(&root).unwrap().with_grace(GRACE);
    let (stop, collected) = (std::sync::atomic::AtomicBool::new(false), AtomicU64::new(0));
    /// Stops the collector however the scope ends, a panic included.
    struct Stop<'a>(&'a std::sync::atomic::AtomicBool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    let ready = std::sync::Barrier::new(2);
    let headers: Vec<ObjectRef> = std::thread::scope(|scope| {
        let _stop = Stop(&stop);
        let collector = scope.spawn(|| {
            let started = Instant::now();
            ready.wait();
            while !stop.load(Ordering::Relaxed) {
                collected.fetch_add(store.gc().unwrap().objects, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(50));
            }
            started.elapsed()
        });
        ready.wait();
        let mut children: Vec<Child> = [1, 2]
            .map(|seed| Child::spawn(&Job { root: root.clone(), arm: None, work: Work::Overlap { seed } }))
            .into_iter()
            .collect();
        let headers = (children.iter_mut())
            .map(|c| {
                let line = c.wait_for("HEADER");
                let mut words = line.split(' ').skip(1);
                ObjectRef { sha256: words.next().unwrap().into(), length: words.next().unwrap().parse().unwrap() }
            })
            .collect();
        stop.store(true, Ordering::Relaxed);
        assert!(collector.join().unwrap() > GRACE * 3, "the collector ran for several graces");
        headers
    });
    println!("the collector swept {} objects meanwhile", collected.load(Ordering::Relaxed));
    for (seed, header) in [1, 2].into_iter().zip(&headers) {
        let ck = store.checkpoint(header).unwrap();
        assert!(ck.missing(None).unwrap().is_empty());
        for (key, bytes) in overlap(seed) {
            assert!(ck.read("unet", &key).unwrap() == bytes, "{seed}: {key}");
        }
    }
    let shared: BTreeSet<_> = store.checkpoint(&headers[0]).unwrap().header().objects().cloned().collect();
    let other: BTreeSet<_> = store.checkpoint(&headers[1]).unwrap().header().objects().cloned().collect();
    assert!(shared.intersection(&other).any(|o| o.length == 20 * MIB as u64), "the shared large part is one object");
}
