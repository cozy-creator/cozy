//! `tfs checkpoint` — the live end-to-end checkpoint plane (tfs-015).
//!
//! A real multi-component synthetic tree is generated, objectized through the real CAS,
//! declared by one canonical `model.cozytensors`, and read back from the header alone.
//! Tensor bytes are a deterministic keyed stream, so the reader can regenerate exactly what
//! the writer was handed and compare digests — an end-to-end byte-identity claim, not a
//! round-trip of whatever happened to be stored.

use std::io::Read;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::checkpoint::{self as ck, build_manifest};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Asset, Closure, Header, Tensor};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::registry;
use tensorfs_core::relation::Carrier;
use tensorfs_core::sha256;
use tensorfs_core::spec::EncodingSpec;
use tensorfs_core::store::Store;

use crate::{bail, flag_num, flag_on, rss_kb, Flags};

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
        let store = Store::open(&root.0).unwrap();
        for (_, _, tensor) in header.tensors() {
            for (_, part) in &tensor.parts {
                if let tensorfs_core::header::Body::Segments(objects) = &part.body {
                    objects.iter().for_each(|o| drop(store.open_verified(&o.sha256).unwrap()));
                }
            }
        }
        let _ = (href, flags);
    }
}
