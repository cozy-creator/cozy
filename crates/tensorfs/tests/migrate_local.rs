//! The laptop's one-time conversion (xs-045 U12, `cozy model migrate --local`): a store from
//! before the format cut and the machine's store on one filesystem. A local model migrates,
//! and the old objects the new checkpoint reuses move in by hard link: no byte is copied, so the
//! conversion costs only its new bytes. Across filesystems they are copied through the store's
//! door. Every tensor reads back bit-identical, and the links outlive the old store.
#![cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use tensorfs::format::{Body, ObjectRef};
use tensorfs::migrate::{self, v1};
use tensorfs::sha256::hex_digest;
use tensorfs::store2::Store;

/// plain/1, as real pre-cut headers carry it (its digest is its name).
const PLAIN: &str = "848c030b0201000405080706090a8081836576616c756581008100818258204766f931bbb70ed431e34b34e4f9fc5dda8174e96f43a8acd79d8799efcd43d2190204";
const BF16: u64 = 3;

fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

fn head(major: u8, n: u64, out: &mut Vec<u8>) {
    let m = major << 5;
    match n {
        0..=23 => out.push(m | n as u8),
        24..=0xff => out.extend([m | 24, n as u8]),
        0x100..=0xffff => {
            out.push(m | 25);
            out.extend((n as u16).to_be_bytes());
        }
        _ => {
            out.push(m | 26);
            out.extend((n as u32).to_be_bytes());
        }
    }
}

fn text(t: &str, out: &mut Vec<u8>) {
    head(3, t.len() as u64, out);
    out.extend(t.as_bytes());
}

/// `(key, shape, bytes, inline)`: plain bf16 tensors of one component.
type Tensor = (&'static str, Vec<u64>, Vec<u8>, bool);

/// A pre-cut store at `root` holding `local/tiny`: a 16 MiB tensor on the old grid (which the
/// new checkpoint reuses), a small one in its own object and one inline. Answers the tensors.
fn precut(root: &Path) -> Vec<Tensor> {
    let tensors: Vec<Tensor> = vec![
        ("blocks.0.big.weight", vec![4096, 2048], noise(1, 16 << 20), false),
        ("blocks.0.small.weight", vec![64, 64], noise(2, 8192), false),
        ("blocks.0.small.bias", vec![64], noise(3, 128), true),
    ];
    let put = |dir: &str, name: String, bytes: &[u8]| {
        let digest = hex_digest(bytes);
        let path = root.join(dir).join(&digest[..2]).join(&digest[2..4]).join(name.replace('*', &digest));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
        digest
    };
    let mut header = vec![0x85];
    text("cozytensors/1", &mut header);
    header.extend([0x80, 0x80, 0x81]);
    header.extend(hex_digest_bytes(PLAIN));
    head(4, 1, &mut header);
    head(4, 2, &mut header);
    text("migrate_test", &mut header);
    head(4, tensors.len() as u64, &mut header);
    for (key, shape, bytes, inline) in &tensors {
        head(4, 5, &mut header);
        text(key, &mut header);
        head(0, BF16, &mut header);
        let dims = |out: &mut Vec<u8>| {
            head(4, shape.len() as u64, out);
            shape.iter().for_each(|n| head(0, *n, out));
        };
        dims(&mut header);
        head(0, 0, &mut header);
        head(4, 1, &mut header);
        head(4, 4, &mut header);
        text("value", &mut header);
        head(0, BF16, &mut header);
        dims(&mut header);
        match inline {
            true => {
                head(2, bytes.len() as u64, &mut header);
                header.extend(bytes);
            }
            false => {
                let digest = put("blobs", "*".into(), bytes);
                header.extend([0x81, 0x82]);
                head(2, 32, &mut header);
                header.extend(hex_digest_bytes(&digest));
                head(0, bytes.len() as u64, &mut header);
            }
        }
    }
    let header_digest = put("blobs", "*".into(), &header);
    let manifest = format!(
        r#"{{"entries":[{{"blob":{{"length":{},"sha256":"{header_digest}"}},"kind":"cozytensors","path":"model.cozytensors"}}]}}"#,
        header.len()
    );
    let digest = put("manifests", "*.json".into(), manifest.as_bytes());
    let repo = format!(
        r#"{{"checkpoints":[{{"manifest":{{"length":{},"sha256":"{digest}"}},"published":false}}],"releases":[],"repo":{{"name":"tiny","org":"local"}}}}"#,
        manifest.len()
    );
    std::fs::create_dir_all(root.join("repos/local")).unwrap();
    std::fs::write(root.join("repos/local/tiny.json"), repo).unwrap();
    tensors
}

fn hex_digest_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(parent: &Path, name: &str) -> Scratch {
    let path = parent.join(format!("migrate-local-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    Scratch(path)
}

/// `local/tiny` of the pre-cut store at `old` into a new store at `new`: what moved in, and
/// its reused 16 MiB object's path in each store.
fn convert(old: &Path, new: &Path, tensors: &[Tensor]) -> (migrate::Adopted, PathBuf, PathBuf) {
    let (src, store) = (v1::Source::new(old), Store::open(new).unwrap());
    let repo = src.repos().unwrap().into_iter().find(|r| r.name == "local/tiny").unwrap();
    let lane = v1::Lane { manifest: repo.checkpoints[0].clone(), names: vec![repo.name.clone()] };
    let report = migrate::lane(&src, Some(&store), &lane, false);
    assert!(report.pass, "{:?}", report.doc);
    let made = report.manifests.1.unwrap();
    let closure = store.closure(&made).unwrap();
    let adopted = migrate::adopt(&src, &store, &closure).unwrap();
    assert_eq!(migrate::adopt(&src, &store, &closure).unwrap(), migrate::Adopted::default(), "a rerun moves nothing");
    // Every tensor reads back from the new store alone.
    let ck = store.checkpoint(&made).unwrap();
    for (key, _, bytes, _) in tensors {
        let Body::Data(at) = &ck.tensor("migrate_test", key).unwrap().body else { panic!("{key} is plain") };
        let mut read = vec![0; at.length as usize];
        ck.bytes_into(at, &mut read).unwrap();
        assert!(read == *bytes, "{key} differs from the old bytes");
    }
    let big = ObjectRef { sha256: hex_digest(&tensors[0].2), length: 16 << 20 };
    (adopted, src.path(&big).unwrap(), store.path(&big).unwrap())
}

#[test]
fn a_local_model_moves_in_by_hard_links_on_one_filesystem() {
    let dir = scratch(Path::new(env!("CARGO_TARGET_TMPDIR")), "one-fs");
    let (old, new) = (dir.0.join("precut"), dir.0.join("store"));
    let tensors = precut(&old);
    let (adopted, from, to) = convert(&old, &new, &tensors);
    assert_eq!(adopted, migrate::Adopted { linked: (1, 16 << 20), copied: (0, 0) });
    let (a, b) = (std::fs::metadata(&from).unwrap(), std::fs::metadata(&to).unwrap());
    assert_eq!((a.ino(), a.dev(), b.nlink()), (b.ino(), b.dev(), 2), "the reused object is one inode in both stores");
    // The old store goes; the new one keeps its bytes.
    std::fs::remove_dir_all(&old).unwrap();
    assert_eq!(std::fs::metadata(&to).unwrap().nlink(), 1);
    let store = Store::open(&new).unwrap();
    assert!(store.verify(&ObjectRef { sha256: hex_digest(&tensors[0].2), length: 16 << 20 }).unwrap());
}

#[test]
fn across_filesystems_the_reused_objects_are_copied() {
    let shm = Path::new("/dev/shm");
    let here = Path::new(env!("CARGO_TARGET_TMPDIR"));
    if !shm.is_dir() || std::fs::metadata(shm).unwrap().dev() == std::fs::metadata(here).unwrap().dev() {
        eprintln!("skipped: no second filesystem at /dev/shm");
        return;
    }
    let (old, new) = (scratch(shm, "precut"), scratch(here, "store"));
    let tensors = precut(&old.0);
    let (adopted, from, to) = convert(&old.0, &new.0, &tensors);
    assert_eq!(adopted, migrate::Adopted { linked: (0, 0), copied: (1, 16 << 20) });
    assert_ne!(std::fs::metadata(&from).unwrap().dev(), std::fs::metadata(&to).unwrap().dev());
}

/// nvfp4-w4a4/1, as real pre-cut headers carry it: `weight`, `weight_scale`, `weight_scale_2`.
const NVFP4: &str = "8482030281028383667765696768748201810a82018282000083010102836c7765696768745f7363616c658201810482018282000083010110836e7765696768745f7363616c655f328201810182018081825820ffc364c5657cd5620662a11aa454706f3a942906b1f1e83344f7eb3febfd4c5d191da3";

/// A v1 part role the new encoding does not map (AWQ's `pre_quant_scale`) refuses its lane,
/// naming the role: migrate never drops a part.
#[test]
fn a_part_role_the_scheme_does_not_map_refuses_the_lane() {
    let dir = scratch(Path::new(env!("CARGO_TARGET_TMPDIR")), "unmapped-role");
    let part = |role: &str, dtype: u64, shape: &[u64], n: usize, out: &mut Vec<u8>| {
        head(4, 4, out);
        text(role, out);
        head(0, dtype, out);
        head(4, shape.len() as u64, out);
        shape.iter().for_each(|d| head(0, *d, out));
        head(2, n as u64, out);
        out.extend(noise(n as u64, n));
    };
    let mut header = vec![0x85];
    text("cozytensors/1", &mut header);
    header.extend([0x80, 0x80, 0x81]);
    header.extend(hex_digest_bytes(NVFP4));
    header.extend([0x81, 0x82]);
    text("transformer", &mut header);
    header.extend([0x81, 0x85]);
    text("blocks.0.proj.weight", &mut header);
    head(0, BF16, &mut header);
    header.extend([0x82, 0x10, 0x18, 0x20, 0x00, 0x84]);
    part("weight", 10, &[16, 16], 256, &mut header);
    part("weight_scale", 4, &[16, 2], 32, &mut header);
    part("weight_scale_2", 1, &[], 4, &mut header);
    part("pre_quant_scale", 1, &[32], 128, &mut header);
    let header_digest = hex_digest(&header);
    std::fs::write(dir.0.join(&header_digest), &header).unwrap();
    let manifest = format!(
        r#"{{"entries":[{{"blob":{{"length":{},"sha256":"{header_digest}"}},"kind":"cozytensors","path":"model.cozytensors"}}]}}"#,
        header.len()
    );
    let digest = hex_digest(manifest.as_bytes());
    std::fs::write(dir.0.join(&digest), &manifest).unwrap();
    let lane = v1::Lane { manifest: ObjectRef { sha256: digest, length: manifest.len() as u64 }, names: vec![] };
    let report = migrate::lane(&v1::Source::new(&dir.0), None, &lane, false);
    assert!(!report.pass, "{:?}", report.doc);
    let error = report.doc["error"]["detail"].as_str().unwrap_or_default();
    assert!(error.contains("pre_quant_scale"), "{:?}", report.doc);
}
