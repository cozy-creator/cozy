//! `tfs checkpoint` — the live end-to-end checkpoint plane (tfs-015).
//!
//! A real multi-component synthetic tree is generated, objectized through the real CAS,
//! declared by one canonical `model.cozytensors`, and read back from the header alone.
//! Tensor bytes are a deterministic keyed stream, so the reader can regenerate exactly what
//! the writer was handed and compare digests — an end-to-end byte-identity claim, not a
//! round-trip of whatever happened to be stored.

use std::io::{Read, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::checkpoint::{self as ck, build_manifest};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Asset, Body, Closure, Header, Located, Part, Tensor};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::limits;
use tensorfs_core::registry;
use tensorfs_core::relation::Carrier;
use tensorfs_core::sha256;
use tensorfs_core::spec::EncodingSpec;
use tensorfs_core::store::Store;

use crate::{bail, flag, flag_num, flag_on, rss_kb, Flags};

// ------------------------------------------------------------------ tensor bytes

/// splitmix64 keyed by a role's full name: reproducible "random" tensor bytes that never
/// have to be stored to be compared against.
pub struct Noise {
    s: u64,
    left: u64,
}

impl Noise {
    pub fn new(seed: &str, len: u64) -> Noise {
        let d = sha256::digest(seed.as_bytes());
        Noise {
            s: u64::from_le_bytes(d[0..8].try_into().unwrap()),
            left: len,
        }
    }
    fn next(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn digest(seed: &str, len: u64) -> String {
        let mut n = Noise::new(seed, len);
        let mut h = sha256::Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let k = n.read(&mut buf).unwrap();
            if k == 0 {
                return sha256::hex(&h.finish());
            }
            h.update(&buf[..k]);
        }
    }
}

impl Read for Noise {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = (buf.len() as u64).min(self.left) as usize;
        let mut i = 0;
        while i < n {
            let w = self.next().to_le_bytes();
            let k = (n - i).min(8);
            buf[i..i + k].copy_from_slice(&w[..k]);
            i += k;
        }
        self.left -= n as u64;
        Ok(n)
    }
}

// ------------------------------------------------------------------ the tree

/// One logical tensor. The ENCODING SPEC decides the role set and every storage shape —
/// the plan states only the logical truth plus which spec the tensor is encoded under.
struct Item {
    comp: &'static str,
    key: String,
    alias: &'static str,
    dtype: Dtype,
    shape: Vec<u64>,
    /// Overridden only for the dedup arm: two keys, byte-identical runs.
    seed: Option<String>,
}

/// The synthetic corpus has one normal, mixed-encoding form. `Plain` is a closed fixture
/// variant for hardware that has only qualified `plain/1`: it changes the encoding choice
/// and presents the same logical set in PyTorch traversal order; it never changes a logical
/// component/key/dtype/shape fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlanEncoding {
    Mixed,
    Plain,
}

impl PlanEncoding {
    fn alias(self, mixed: &'static str) -> &'static str {
        match self {
            PlanEncoding::Mixed => mixed,
            PlanEncoding::Plain => "plain/1",
        }
    }
}

fn item(
    comp: &'static str,
    key: String,
    alias: &'static str,
    dtype: Dtype,
    shape: Vec<u64>,
) -> Item {
    Item {
        comp,
        key,
        alias,
        dtype,
        shape,
        seed: None,
    }
}

/// SDXL-shaped: three components, mixed encodings, parts on both sides of the inline
/// threshold, one part over the 64 MiB grid, one rank-0 scalar, one byte-identical pair.
/// `scale` divides every dimension (floor 2), so the same SHAPE of corpus can be written
/// at a size a repository can carry: the committed wheel-smoke store is this plan at
/// scale 32, ~40 KB, and it exercises the same mixed encoding choices and path law as the
/// full one. `PlanEncoding::Plain` retains every logical fact and only replaces aliases.
fn plan(blocks: u64, scale: u64, encoding: PlanEncoding) -> Vec<Item> {
    let d = |dims: Vec<u64>| -> Vec<u64> { dims.iter().map(|x| (x / scale).max(2)).collect() };
    let mut v = Vec::new();
    for i in 0..blocks {
        let k = |s: &str| format!("blocks.{i}.{s}");
        v.push(item(
            "unet",
            k("attn1.to_q.weight"),
            encoding.alias("fp8-rowwise/1"),
            Dtype::Bf16,
            d(vec![2048, 2048]),
        ));
        v.push(item(
            "unet",
            k("attn1.to_out.0.weight"),
            encoding.alias("plain/1"),
            Dtype::Bf16,
            d(vec![2048, 2048]),
        ));
        let ff = item(
            "unet",
            k("ff.net.0.proj.weight"),
            encoding.alias("nvfp4-w4a4/1"),
            Dtype::Bf16,
            d(vec![2048, 8192]),
        );
        let norm1 = item(
            "unet",
            k("norm1.weight"),
            encoding.alias("plain/1"),
            Dtype::Bf16,
            d(vec![2048]),
        );
        let norm_q = item(
            "unet",
            k("attn1.norm_q.weight"),
            encoding.alias("plain/1"),
            Dtype::Bf16,
            d(vec![64]),
        );
        match encoding {
            // Existing fixture identity is production evidence: its default order is frozen.
            PlanEncoding::Mixed => v.extend([ff, norm1, norm_q]),
            // `torch.nn.Module.named_parameters()` is depth-first: finish attn1 before
            // visiting sibling modules ff and norm1. This is the executable fixture order.
            PlanEncoding::Plain => v.extend([norm_q, ff, norm1]),
        }
    }
    v.push(item(
        "text_encoder",
        "embeddings.weight".into(),
        encoding.alias("plain/1"),
        Dtype::Bf16,
        d(vec![8192, 8192]),
    ));
    v.push(item(
        "text_encoder",
        "final_norm.weight".into(),
        encoding.alias("plain/1"),
        Dtype::F32,
        d(vec![1024]),
    ));
    v.push(item(
        "vae",
        "decoder.conv_in.weight".into(),
        encoding.alias("plain/1"),
        Dtype::F32,
        d(vec![512, 512]),
    ));
    let mut twin = item(
        "vae",
        "decoder.conv_out.weight".into(),
        encoding.alias("plain/1"),
        Dtype::F32,
        d(vec![512, 512]),
    );
    twin.seed = Some("vae/decoder.conv_in.weight".to_string());
    v.push(twin);
    v
}

fn seed_of(it: &Item, role: &str) -> String {
    match &it.seed {
        Some(s) => format!("{s}#{role}"),
        None => format!("{}/{}#{role}", it.comp, it.key),
    }
}

fn registry_seed(alias: &str) -> registry::Seed {
    registry::seeds()
        .into_iter()
        .find(|s| s.alias == alias)
        .expect("seed alias")
}

fn spec_of(alias: &str) -> EncodingSpec {
    registry_seed(alias).spec
}

fn carrier_dtype(c: &Carrier, logical: Dtype) -> Dtype {
    match c {
        Carrier::SameAsLogical => logical,
        Carrier::Set(v) => v[0],
    }
}

fn configs() -> Vec<(&'static str, &'static [u8])> {
    vec![
        (
            "unet",
            br#"{"block_out_channels":[320,640,1280],"in_channels":4}"# as &[u8],
        ),
        ("text_encoder", br#"{"hidden_size":2048,"num_layers":32}"#),
    ]
}

// ------------------------------------------------------------------ write

pub fn cmd_write(root: &Path, f: &Flags) -> ExitCode {
    let blocks: u64 = flag_num(f, "blocks", 12);
    let scale: u64 = flag_num::<u64>(f, "scale", 1).max(1);
    let store = ok!(Store::init(root));
    let encoding = if flag_on(f, "plain") {
        PlanEncoding::Plain
    } else {
        PlanEncoding::Mixed
    };
    let items = plan(blocks, scale, encoding);

    // The encoding closure first. Construction configs are canonical bytes inside the header.
    let mut aliases: Vec<&'static str> = items.iter().map(|i| i.alias).collect();
    aliases.sort_unstable();
    aliases.dedup();
    let mut closure = Closure::default();
    let mut encodings = Vec::new();
    for a in &aliases {
        let sd = registry_seed(a);
        encodings.push(sd.spec.clone());
        closure.insert(sd.spec);
    }
    encodings.sort_by_key(EncodingSpec::object_id);

    // --no-configs is the TENSORS-ONLY shape: one header plus its closure, zero sibling
    // assets. The manifest id derives once from the virtual tree.
    let mut inline_configs = Vec::new();
    let cfgs = if flag_on(f, "no-configs") {
        Vec::new()
    } else {
        configs()
    };
    for (name, bytes) in cfgs {
        inline_configs.push((
            name.to_string(),
            ok!(tensorfs_core::header::canonical_config("config", bytes)),
        ));
    }
    inline_configs.sort_by(|a, b| a.0.cmp(&b.0));
    let asset_bytes = b"<pad>0</pad>\n<pad>1</pad>\n";
    let asset_object = ok!(store.put_stream(
        &mut &asset_bytes[..],
        Some(&ObjectRef::of(asset_bytes)),
        &Default::default()
    ))
    .obj;
    let assets = vec![(
        "tokenizer/vocab.txt".to_string(),
        Asset {
            logical_sha256: sha256::hex_digest(asset_bytes),
            logical_length: asset_bytes.len() as u64,
            media_type: "text/plain".to_string(),
            segments: vec![asset_object],
        },
    )];

    let t0 = Instant::now();
    let (mut bytes, mut deduped, mut segments, mut inlines) = (0u64, 0u64, 0usize, 0usize);
    let mut comps: Vec<(String, Vec<(String, Tensor)>)> = Vec::new();
    for it in &items {
        let spec = spec_of(it.alias);
        let elements = ok!(tensorfs_core::dtype::checked_elements(&it.key, &it.shape));
        let mut parts = Vec::new();
        for (role, r) in &spec.roles {
            let shape = ok!(r.shape.eval(&it.shape, elements));
            let dt = carrier_dtype(&r.carrier, it.dtype);
            let n = ok!(tensorfs_core::dtype::checked_bytes(role, &shape, dt));
            let mut src = Noise::new(&seed_of(it, role), n);
            let w = ok!(ck::objectize(
                &store,
                &format!("{}/{}#{role}", it.comp, it.key),
                dt,
                shape,
                &mut src
            ));
            bytes += w.bytes;
            deduped += w.deduped;
            segments += w.segments;
            if w.segments == 0 {
                inlines += 1;
            }
            parts.push((role.clone(), w.part));
        }
        let tensor = Tensor {
            dtype: it.dtype,
            shape: it.shape.clone(),
            encoding: spec.object_id(),
            parts,
        };
        match comps.iter_mut().find(|(c, _)| c == it.comp) {
            Some((_, ts)) => ts.push((it.key.clone(), tensor)),
            None => comps.push((it.comp.to_string(), vec![(it.key.clone(), tensor)])),
        }
    }
    let write_s = t0.elapsed().as_secs_f64();

    let header = Header {
        configs: inline_configs,
        assets,
        encodings,
        components: comps,
    };
    let t = Instant::now();
    let hbytes = ok!(header.canonical_bytes());
    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
    ok!(header.validate(&closure));
    let href = ok!(ck::put_doc(&store, &header));

    // The manifest is unconditional: manifest_id is its ObjectId, one derivation always.
    let snap = ok!(build_manifest(&header, &href, &closure));
    let sref = ok!(store.put_manifest(&snap)).obj;

    let mb = bytes as f64 / (1024.0 * 1024.0);
    println!("checkpoint written to {}", root.display());
    println!("  header_digest   {}", href.id());
    println!("  manifest_id     {}", sref.id());
    println!("  tensor_schema_digest {}", header.tensor_schema_digest());
    println!(
        "  components {}  tensors {}  parts {}  segments {}  inline {}",
        header.components.len(),
        header.tensors().count(),
        header
            .tensors()
            .map(|(_, _, t)| t.parts.len())
            .sum::<usize>(),
        segments,
        inlines
    );
    println!(
        "  encodings {}  configs {}  assets {}  header {} B canonical (encode {encode_ms:.1} ms)",
        header.encodings.len(),
        header.configs.len(),
        header.assets.len(),
        hbytes.len()
    );
    println!(
        "  tensor bytes {bytes} ({mb:.1} MiB) in {write_s:.2}s = {:.1} MiB/s",
        mb / write_s
    );
    println!(
        "  byte-identical dedup: {deduped} B ({:.1} MiB) already installed",
        deduped as f64 / (1024.0 * 1024.0)
    );
    println!("  peak RSS {:.1} MiB", rss_kb("VmHWM") as f64 / 1024.0);
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ read

struct Sink(u64);
impl Write for Sink {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0 += b.len() as u64;
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn header_ref(root: &Path, hex: &str) -> Result<(Store, ObjectRef), ExitCode> {
    let store = match Store::open(root) {
        Ok(s) => s,
        Err(e) => return Err(bail(e)),
    };
    let len = match std::fs::metadata(store.object_path(hex)) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("REFUSED OBJECT_ABSENT: {hex}: {e}");
            return Err(ExitCode::FAILURE);
        }
    };
    let r = ObjectRef {
        sha256: hex.to_string(),
        length: len,
    };
    Ok((store, r))
}

macro_rules! opened {
    ($root:expr, $hex:expr) => {
        match header_ref($root, $hex) {
            Ok(v) => v,
            Err(c) => return c,
        }
    };
}

pub fn cmd_read(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let (store, href) = opened!(root, hex);
    let blocks: u64 = flag_num(f, "blocks", 12);
    let scale: u64 = flag_num::<u64>(f, "scale", 1).max(1);

    let t = Instant::now();
    let header = ok!(ck::load_header(&store, &href));
    let parse_ms = t.elapsed().as_secs_f64() * 1000.0;
    let closure = ok!(ck::load_closure(&store, &header));
    let t = Instant::now();
    ok!(header.validate(&closure));
    let validate_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Expected whole-part digests, regenerated from the plan — never read from the store.
    let encoding = if flag_on(f, "plain") {
        PlanEncoding::Plain
    } else {
        PlanEncoding::Mixed
    };
    let items = plan(blocks, scale, encoding);
    let mut expect: Vec<(String, String)> = Vec::new();
    for it in &items {
        let spec = spec_of(it.alias);
        let elements = ok!(tensorfs_core::dtype::checked_elements(&it.key, &it.shape));
        for (role, r) in &spec.roles {
            let shape = ok!(r.shape.eval(&it.shape, elements));
            let dt = carrier_dtype(&r.carrier, it.dtype);
            let n = ok!(tensorfs_core::dtype::checked_bytes(role, &shape, dt));
            expect.push((
                format!("{}/{}#{role}", it.comp, it.key),
                Noise::digest(&seed_of(it, role), n),
            ));
        }
    }

    let t = Instant::now();
    let (mut bytes, mut checked, mut mismatched) = (0u64, 0usize, 0usize);
    for (comp, key, tensor) in header.tensors() {
        for (role, part) in &tensor.parts {
            let what = format!("{comp}/{key}#{role}");
            let mut sink = Sink(0);
            let got = ok!(ck::materialize(&store, &what, part, &mut sink));
            bytes += sink.0;
            match expect.iter().find(|(w, _)| *w == what) {
                Some((_, want)) if *want == got => checked += 1,
                Some((_, want)) => {
                    mismatched += 1;
                    eprintln!("BYTE MISMATCH {what}: want sha256:{want} got sha256:{got}");
                }
                None => {
                    mismatched += 1;
                    eprintln!("UNEXPECTED PART {what} — not in the regenerated plan");
                }
            }
        }
    }
    let mut checked_assets = 0usize;
    for (name, asset) in &header.assets {
        let mut sink = Sink(0);
        ok!(ck::read_asset(&store, name, asset, &mut sink));
        bytes = match bytes.checked_add(sink.0) {
            Some(bytes) => bytes,
            None => {
                eprintln!("REFUSED ARITH_OVERFLOW: checkpoint read byte total overflows");
                return ExitCode::FAILURE;
            }
        };
        checked_assets += 1;
    }
    let read_s = t.elapsed().as_secs_f64();
    let mb = bytes as f64 / (1024.0 * 1024.0);

    println!("checkpoint read back from {}", root.display());
    println!(
        "  header {} B, strict parse {parse_ms:.1} ms, validate {validate_ms:.1} ms",
        href.length
    );
    println!(
        "  {} tensors, {} parts, {checked_assets} assets, {bytes} B ({mb:.1} MiB) reconstructed in {read_s:.2}s = {:.1} MiB/s",
        header.tensors().count(),
        expect.len(),
        mb / read_s
    );
    println!(
        "  byte-identical parts {checked}/{}, mismatched {mismatched}",
        expect.len()
    );
    println!("  peak RSS {:.1} MiB", rss_kb("VmHWM") as f64 / 1024.0);
    if mismatched > 0 {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ info / locate

pub fn cmd_info(root: &Path, hex: &str) -> ExitCode {
    let (store, href) = opened!(root, hex);
    let header = ok!(ck::load_header(&store, &href));
    println!("header {} ({} B)", href.id(), href.length);
    for (name, c) in &header.configs {
        println!("config {name:<14} {} inline B", c.len());
    }
    for r in &header.encodings {
        println!("encoding {} (inlined)", r.object_id());
    }
    for (comp, ts) in &header.components {
        let (mut bytes, mut parts, mut segs) = (0u64, 0usize, 0usize);
        for (k, t) in ts {
            for (role, p) in &t.parts {
                parts += 1;
                segs += p.segments().len();
                bytes += ok!(p.nbytes(&format!("{comp}/{k}#{role}")));
            }
        }
        println!(
            "component {comp:<14} tensors {:<5} parts {parts:<5} segments {segs:<5} {:.1} MiB",
            ts.len(),
            bytes as f64 / (1024.0 * 1024.0)
        );
    }
    ExitCode::SUCCESS
}

pub fn cmd_locate(
    root: &Path,
    hex: &str,
    sel: [&str; 3],
    off: u64,
    len: u64,
    f: &Flags,
) -> ExitCode {
    let (store, href) = opened!(root, hex);
    let header = ok!(ck::load_header(&store, &href));
    let [comp, key, role] = sel;
    let part: &Part = match header
        .tensors()
        .find(|(c, k, _)| *c == comp && *k == key)
        .and_then(|(_, _, t)| t.parts.iter().find(|(n, _)| n == role))
    {
        Some((_, p)) => p,
        None => {
            eprintln!("REFUSED MISSING_TENSOR: no part {comp}/{key}#{role} in this header");
            return ExitCode::FAILURE;
        }
    };
    let what = format!("{comp}/{key}#{role}");
    println!(
        "part {what}: dtype {} shape {:?} nbytes {} segments {}",
        part.dtype.name(),
        part.shape,
        ok!(part.nbytes(&what)),
        part.segments().len()
    );
    let t = Instant::now();
    let located = ok!(part.locate(&what, off, len));
    let derive_us = t.elapsed().as_secs_f64() * 1e6;
    match &located {
        Located::Inline { off, len } => println!("  inline slice [{off}, {})", off + len),
        Located::Segments(spans) => {
            for s in spans {
                println!(
                    "  segment {:<4} sha256:{}  +{} .. +{}",
                    s.index,
                    &s.obj.sha256[..16],
                    s.off,
                    s.off + s.len
                );
            }
        }
    }
    println!("  offsets derived in {derive_us:.1} us with zero fetches");
    let got = ok!(ck::read_span(&store, &what, part, off, len));
    println!(
        "  fetched {} B, sha256:{}",
        got.len(),
        sha256::hex_digest(&got)
    );
    if let Some(seed) = flag(f, "seed") {
        let mut n = Noise::new(seed, off + len);
        let mut all = vec![0u8; (off + len) as usize];
        n.read_exact(&mut all).unwrap();
        let want = &all[off as usize..];
        println!(
            "  against the regenerated source: {}",
            if want == got.as_slice() {
                "IDENTICAL"
            } else {
                "DIFFERENT"
            }
        );
        if want != got.as_slice() {
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ caps / size

/// The whole-document totals are red-armed in memory: the over-cap documents are far too
/// large to commit as frozen vectors, so they are BUILT and refused live.
pub fn cmd_caps() -> ExitCode {
    let plain = spec_of("plain/1");
    let mut closure = Closure::default();
    closure.insert(plain.clone());

    let seg = |n: usize| ObjectRef {
        sha256: sha256::hex_digest(format!("seg{n}").as_bytes()),
        length: 1024,
    };
    let over = limits::MAX_TOTAL_REFS + 1;
    let mut tensors = Vec::new();
    for i in 0..(over / limits::MAX_SEGMENTS + 1) {
        let n = limits::MAX_SEGMENTS.min(over - i * limits::MAX_SEGMENTS);
        tensors.push((
            format!("t{i:04}.weight"),
            Tensor {
                dtype: Dtype::U8,
                shape: vec![n as u64 * 1024],
                encoding: plain.object_id(),
                parts: vec![(
                    "value".to_string(),
                    Part {
                        dtype: Dtype::U8,
                        shape: vec![n as u64 * 1024],
                        body: Body::Segments(
                            (0..n).map(|k| seg(i * limits::MAX_SEGMENTS + k)).collect(),
                        ),
                    },
                )],
            },
        ));
    }
    let refs_doc = Header {
        configs: vec![],
        assets: vec![],
        encodings: vec![plain.clone()],
        components: vec![("transformer".to_string(), tensors)],
    };
    println!(
        "whole-document reference total: {} refs against a cap of {}",
        refs_doc
            .tensors()
            .map(|(_, _, t)| t
                .parts
                .iter()
                .map(|(_, p)| p.segments().len())
                .sum::<usize>())
            .sum::<usize>(),
        limits::MAX_TOTAL_REFS
    );
    report(refs_doc.validate(&closure));

    // One tensor whose DECLARED bytes pass the whole-document byte total.
    let big = 1u64 << 48;
    let bytes_doc = Header {
        configs: vec![],
        assets: vec![],
        encodings: vec![plain.clone()],
        components: vec![(
            "transformer".to_string(),
            vec![(
                "huge.weight".to_string(),
                Tensor {
                    dtype: Dtype::F64,
                    shape: vec![big],
                    encoding: plain.object_id(),
                    parts: vec![(
                        "value".to_string(),
                        Part {
                            dtype: Dtype::F64,
                            shape: vec![big],
                            body: Body::Segments(vec![ObjectRef {
                                sha256: sha256::hex_digest(b"huge"),
                                length: big * 8,
                            }]),
                        },
                    )],
                },
            )],
        )],
    };
    println!(
        "whole-document byte total: {} B declared against a cap of {} B",
        big * 8,
        limits::MAX_TOTAL_BYTES
    );
    report(bytes_doc.validate(&closure));
    ExitCode::SUCCESS
}

fn report(r: tensorfs_core::err::Result<()>) {
    match r {
        Ok(()) => println!("  ACCEPTED — the cap did not bind (arm is not red)"),
        Err(e) => println!("  REFUSED {}: {}", e.code.as_str(), e.detail),
    }
}

/// Header-size cost of the ONE uniform tensor form at the real selected-H3 cardinality
/// (~3,699 tensors), over a realistic mixed-encoding tree rather than one repeated shape.
pub fn cmd_size(n: usize) -> ExitCode {
    let aliases = ["plain/1", "fp8-rowwise/1", "nvfp4-w4a4/1"];
    let specs: Vec<EncodingSpec> = aliases.iter().map(|a| spec_of(a)).collect();
    let mut closure = Closure::default();
    for s in &specs {
        closure.insert(s.clone());
    }
    let mut comps: Vec<(String, Vec<(String, Tensor)>)> = Vec::new();
    for i in 0..n {
        let comp = ["unet", "text_encoder", "vae"][i % 3];
        let which = i % 3;
        let spec = &specs[which];
        let shape: Vec<u64> = match which {
            0 => vec![4096, 4096],
            1 => vec![4096, 4096],
            _ => vec![4096, 8192],
        };
        let elements = shape.iter().product::<u64>();
        // Key LENGTH drives header size, so the keys are shaped like the real ones:
        // `examples/safetensors-headers/minimax_h3_fl2va_fp8_scaled.header.json` carries 943
        // logical DiT keys averaging 37.7 bytes (max 43). Synthetic short keys understate
        // the cost by ~11 B/tensor.
        const SUFFIX: [&str; 6] = [
            "attn.to_q.weight",
            "attn.to_k.weight",
            "attn.to_v.weight",
            "attn.to_out.0.weight",
            "ff.net.0.proj.weight",
            "norm1.linear.weight",
        ];
        let key = format!("transformer_blocks.{}.{}", i / 6, SUFFIX[i % 6]);
        let mut parts = Vec::new();
        for (role, r) in &spec.roles {
            let sshape = ok!(r.shape.eval(&shape, elements));
            let dt = carrier_dtype(&r.carrier, Dtype::Bf16);
            let nb = ok!(tensorfs_core::dtype::checked_bytes(role, &sshape, dt));
            let body = if Part::is_inline(nb) {
                Body::Inline(vec![0u8; nb as usize])
            } else {
                Body::Segments(
                    (0..nb.div_ceil(limits::GRID_BYTES))
                        .map(|s| ObjectRef {
                            sha256: sha256::hex_digest(format!("{i}:{role}:{s}").as_bytes()),
                            length: if s + 1 == nb.div_ceil(limits::GRID_BYTES) {
                                nb - s * limits::GRID_BYTES
                            } else {
                                limits::GRID_BYTES
                            },
                        })
                        .collect(),
                )
            };
            parts.push((
                role.clone(),
                Part {
                    dtype: dt,
                    shape: sshape,
                    body,
                },
            ));
        }
        let t = Tensor {
            dtype: Dtype::Bf16,
            shape,
            encoding: spec.object_id(),
            parts,
        };
        match comps.iter_mut().find(|(c, _)| c == comp) {
            Some((_, ts)) => ts.push((key, t)),
            None => comps.push((comp.to_string(), vec![(key, t)])),
        }
    }
    let mut enc: Vec<EncodingSpec> = specs.clone();
    enc.sort_by_key(EncodingSpec::object_id);
    let h = Header {
        configs: vec![],
        assets: vec![],
        encodings: enc,
        components: comps,
    };
    let count = h.tensors().count();
    let parts: usize = h.tensors().map(|(_, _, t)| t.parts.len()).sum();
    let t = Instant::now();
    let bytes = ok!(h.canonical_bytes());
    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let parsed = ok!(Header::parse(&bytes));
    let parse_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    ok!(parsed.validate(&closure));
    let validate_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "uniform-form header, mixed tree: {count} tensors, {parts} parts, {} components",
        h.components.len()
    );
    println!(
        "  {} B canonical ({:.2} MiB) = {} B/tensor",
        bytes.len(),
        bytes.len() as f64 / (1024.0 * 1024.0),
        bytes.len() / count
    );
    println!(
        "  encode {encode_ms:.1} ms   strict parse {parse_ms:.1} ms   validate {validate_ms:.1} ms"
    );
    println!("  tensor_schema_digest {}", parsed.tensor_schema_digest());
    println!("  peak RSS {:.1} MiB", rss_kb("VmHWM") as f64 / 1024.0);
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);
    const SCALED_SCHEMA: &str =
        "sha256:9a73b8829dcc6a29ce4b8bdd12672b788078e0f32d9e5fbd806a3cbb13f7646a";
    const FULL_SCHEMA: &str =
        "sha256:fd58d2dec532f72f8744f3369ebf6b35c489c58f42647c17f950f9aae23f0afc";
    const MIXED_BLOCK_ORDER: [&str; 5] = [
        "blocks.0.attn1.to_q.weight",
        "blocks.0.attn1.to_out.0.weight",
        "blocks.0.ff.net.0.proj.weight",
        "blocks.0.norm1.weight",
        "blocks.0.attn1.norm_q.weight",
    ];
    const PLAIN_BLOCK_ORDER: [&str; 5] = [
        "blocks.0.attn1.to_q.weight",
        "blocks.0.attn1.to_out.0.weight",
        "blocks.0.attn1.norm_q.weight",
        "blocks.0.ff.net.0.proj.weight",
        "blocks.0.norm1.weight",
    ];

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new(name: &str) -> Self {
            let unique = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "tensorfs-{name}-{}-{nanos}-{unique}",
                std::process::id()
            ));
            assert!(!root.exists(), "fresh test root already exists");
            TestRoot(root)
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            if self.0.exists() {
                fs::remove_dir_all(&self.0).expect("remove owned TensorFS test root");
            }
        }
    }

    fn write_and_load(name: &str, plain: bool) -> (TestRoot, ObjectRef, Header, Closure, Flags) {
        let root = TestRoot::new(name);
        let mut flags = vec![
            ("blocks".to_string(), "1".to_string()),
            ("scale".to_string(), "32".to_string()),
        ];
        if plain {
            flags.push(("plain".to_string(), "1".to_string()));
        }
        assert_eq!(cmd_write(&root.0, &flags), ExitCode::SUCCESS);

        let store = Store::open(&root.0).expect("open written test store");
        let mut headers = Vec::new();
        for hex in store.objects().expect("list written objects") {
            let path = store.object_path(&hex);
            let length = fs::metadata(&path).expect("stat written object").len();
            if length > Header::MAX_BYTES as u64 {
                continue;
            }
            let bytes = fs::read(&path).expect("read bounded written object");
            if let Ok(header) = Header::parse(&bytes) {
                headers.push((
                    ObjectRef {
                        sha256: hex,
                        length,
                    },
                    header,
                ));
            }
        }
        assert_eq!(headers.len(), 1, "store must contain exactly one header");
        let (href, header) = headers.pop().expect("one header");
        let closure = ck::load_closure(&store, &header).expect("load emitted header closure");
        (root, href, header, closure, flags)
    }

    fn encoding_ids(header: &Header) -> BTreeSet<String> {
        header
            .tensors()
            .map(|(_, _, tensor)| tensor.encoding.clone())
            .collect()
    }

    fn component_keys<'a>(header: &'a Header, component: &str) -> Vec<&'a str> {
        header
            .components
            .iter()
            .find(|(name, _)| name == component)
            .expect("component exists")
            .1
            .iter()
            .map(|(key, _)| key.as_str())
            .collect()
    }

    #[test]
    fn full_plain_plan_preserves_every_logical_fact_and_volume() {
        let logical = |encoding| {
            plan(12, 1, encoding)
                .into_iter()
                .map(|item| {
                    let bytes = tensorfs_core::dtype::checked_bytes(
                        &format!("{}/{}", item.comp, item.key),
                        &item.shape,
                        item.dtype,
                    )
                    .expect("valid synthetic logical tensor");
                    (item.comp, item.key, item.dtype.name(), item.shape, bytes)
                })
                .collect::<Vec<_>>()
        };
        let mut mixed = logical(PlanEncoding::Mixed);
        let mut plain = logical(PlanEncoding::Plain);
        mixed.sort_by(|left, right| left.0.cmp(right.0).then(left.1.cmp(&right.1)));
        plain.sort_by(|left, right| left.0.cmp(right.0).then(left.1.cmp(&right.1)));
        assert_eq!(
            plain, mixed,
            "--plain may change only encoding aliases and traversal order"
        );
        assert_eq!(mixed.len(), 64);
        assert_eq!(
            mixed.iter().map(|item| item.4).sum::<u64>(),
            740_349_440,
            "the full synthetic corpus logical byte volume is fixed"
        );
        assert_eq!(
            mixed.iter().map(|item| item.0).collect::<BTreeSet<_>>(),
            BTreeSet::from(["text_encoder", "unet", "vae"])
        );
        assert!(plan(12, 1, PlanEncoding::Plain)
            .iter()
            .all(|item| item.alias == "plain/1"));
        let full_plain = plan(12, 1, PlanEncoding::Plain);
        let schema = tensorfs_core::header::tensor_schema_value(full_plain.iter().map(|item| {
            (
                item.comp,
                item.key.as_str(),
                item.dtype,
                item.shape.as_slice(),
            )
        }));
        assert_eq!(
            tensorfs_core::header::tensor_schema_digest_of(&schema),
            FULL_SCHEMA
        );
    }

    #[test]
    fn fixture_plans_lock_default_and_pytorch_traversal_orders() {
        let keys = |encoding| {
            plan(1, 32, encoding)
                .into_iter()
                .filter(|item| item.comp == "unet")
                .map(|item| item.key)
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(PlanEncoding::Mixed), MIXED_BLOCK_ORDER);
        assert_eq!(keys(PlanEncoding::Plain), PLAIN_BLOCK_ORDER);
    }

    #[test]
    fn default_writer_emits_a_mixed_header_and_closure() {
        let (_root, _href, header, closure, _flags) = write_and_load("mixed", false);
        let ids = encoding_ids(&header);
        let header_ids: BTreeSet<String> = header
            .encodings
            .iter()
            .map(EncodingSpec::object_id)
            .collect();
        let closure_ids: BTreeSet<String> =
            closure.specs.iter().map(|(id, _)| id.clone()).collect();
        assert!(
            ids.len() > 1,
            "default checkpoint unexpectedly stopped being mixed"
        );
        assert_eq!(ids, header_ids);
        assert_eq!(ids, closure_ids);
        assert!(ids.contains(&registry_seed("plain/1").spec.object_id()));
        assert_eq!(component_keys(&header, "unet"), MIXED_BLOCK_ORDER);
        assert_eq!(header.tensor_schema_digest(), SCALED_SCHEMA);
    }

    #[test]
    fn plain_writer_emits_only_plain_in_header_closure_and_tensors() {
        let (root, href, header, closure, flags) = write_and_load("plain", true);
        let plain_id = registry_seed("plain/1").spec.object_id();
        assert_eq!(
            encoding_ids(&header),
            BTreeSet::from([plain_id.clone()]),
            "every emitted tensor must cite plain/1"
        );
        assert_eq!(
            header
                .encodings
                .iter()
                .map(EncodingSpec::object_id)
                .collect::<Vec<_>>(),
            vec![plain_id.clone()],
            "the header closure must cite only plain/1"
        );
        assert_eq!(closure.specs.len(), 1);
        assert_eq!(closure.specs[0].0, plain_id);
        assert!(header
            .tensors()
            .all(|(_, _, tensor)| tensor.parts.len() == 1 && tensor.parts[0].0 == "value"));
        assert_eq!(component_keys(&header, "unet"), PLAIN_BLOCK_ORDER);
        assert_eq!(header.tensor_schema_digest(), SCALED_SCHEMA);
        assert_eq!(cmd_read(&root.0, &href.sha256, &flags), ExitCode::SUCCESS);
    }
}
