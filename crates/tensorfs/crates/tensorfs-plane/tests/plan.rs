//! `layout::plan` over a real cozytensors/2 checkpoint: every part's bytes, in header order.
#![cfg(feature = "store")]
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};

use tensorfs_core::format::{Dtype, Encoding, Metadata};
use tensorfs_core::read::Source;
use tensorfs_core::store2::Store;
use tensorfs_core::writer::{Data, PartData};
use tensorfs_plane::layout::{self, Layout};

fn bytes(seed: u8, n: usize) -> Vec<u8> {
    (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
}

#[test]
fn a_plan_reads_every_part_in_header_order() {
    let dir = std::env::temp_dir().join(format!("tensorfs-plan-{}", std::process::id()));
    let store = Store::open(&dir).unwrap();
    let mut w = store.writer(None, None).unwrap();
    let mut expect: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    // A large part (its own chunks, past the 64 MiB grid) and small ones packed together.
    let big = bytes(1, (64 << 20) + 4096 + 3);
    w.put("unet", "a.weight", &Dtype::U8, &[big.len() as u64], Data::Plain(&mut big.as_slice())).unwrap();
    expect.insert("unet/a.weight#value".into(), big);
    let small = bytes(2, 1000);
    w.put("unet", "b.bias", &Dtype::U8, &[1000], Data::Plain(&mut small.as_slice())).unwrap();
    expect.insert("unet/b.bias#value".into(), small);
    let (weight, scale) = (bytes(3, 64), bytes(4, 16));
    let (mut w_in, mut s_in) = (weight.as_slice(), scale.as_slice());
    let parts = vec![
        PartData { role: "weight".into(), dtype: Dtype::F8_E4M3, shape: vec![4, 16], data: &mut w_in },
        PartData { role: "scale".into(), dtype: Dtype::F32, shape: vec![4], data: &mut s_in },
    ];
    let mut params = Metadata::new();
    params.insert("axis".into(), tensorfs_core::format::Value::Int(0.into()));
    let encoding = Encoding { scheme: "fp8-rowwise".into(), params };
    w.put("vae", "c.weight", &Dtype::BF16, &[4, 16], Data::Encoded { encoding, parts }).unwrap();
    expect.insert("vae/c.weight#weight".into(), weight);
    expect.insert("vae/c.weight#scale".into(), scale);
    let ck = store.checkpoint(&w.commit().unwrap()).unwrap();
    let header = ck.header();

    let plan = layout::plan(header, &["unet".into(), "vae".into()], 1 << 20).unwrap();
    // Destinations are contiguous, and the parts come in the header's own order.
    let mut dest = 0;
    let mut order: Vec<String> = Vec::new();
    let mut got: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for item in &plan.items {
        assert_eq!(item.dest_off, dest);
        assert!(item.len <= 1 << 20);
        dest += item.len;
        if order.last() != Some(&item.what) {
            order.push(item.what.clone());
        }
        let Source::Object(range) = &item.source else { panic!("a v2 plan reads chunks") };
        let path = header.chunks.iter().zip(ck.chunk_paths()).find(|(c, _)| c.sha256 == range.obj.sha256).unwrap().1;
        let mut file = std::fs::File::open(path).unwrap();
        let mut buf = vec![0; range.len as usize];
        file.seek(SeekFrom::Start(range.off)).unwrap();
        file.read_exact(&mut buf).unwrap();
        got.entry(item.what.clone()).or_default().extend(buf);
    }
    assert_eq!(plan.bytes, dest);
    assert_eq!(got, expect);
    let header_order: Vec<String> = header
        .components
        .iter()
        .flat_map(|(c, comp)| {
            comp.tensors.iter().flat_map(move |(k, t)| match &t.body {
                tensorfs_core::format::Body::Data(_) => vec![format!("{c}/{k}#value")],
                tensorfs_core::format::Body::Encoded { parts, .. } => parts.keys().map(|r| format!("{c}/{k}#{r}")).collect(),
            })
        })
        .collect();
    assert_eq!(order, header_order);

    // One component alone, as a weight set registers it; regions name its tensors.
    let unet = layout::plan(header, &["unet".into()], 0).unwrap();
    assert!(unet.items.iter().all(|i| i.what.starts_with("unet/")));
    Layout::build(&unet, &[vec!["unet/a.weight".into()], vec!["unet/b.bias".into()]]).unwrap();
    assert!(layout::plan(header, &["text_encoder".into()], 0).is_err());
    assert!(layout::plan(header, &[], 0).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
