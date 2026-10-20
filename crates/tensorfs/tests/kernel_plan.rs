use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use base64::Engine;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tensorfs::format::decode::plan::{Core, Elems, Plan, Post};
use tensorfs::format::decode::{self, gguf, Geometry};
use tensorfs::format::{Dtype, Encoding, PartBytes};
use tensorfs::kernels::{self, Prepared, BINARIES, CORE_KINDS, FAMILIES};

#[test]
fn every_encoded_green_vector_prepares_a_kernel_without_host_fallback() {
    #[derive(Deserialize)]
    struct Cases {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        component: String,
        key: String,
        dtype: Dtype,
        shape: Vec<u64>,
        encoding: Option<Encoding>,
        #[serde(default)]
        parts: Vec<Part>,
        expect: Option<String>,
    }
    #[derive(Deserialize)]
    struct Part {
        role: String,
        dtype: Dtype,
        shape: Vec<u64>,
        bits: String,
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/vectors/cases");
    let mut files: Vec<_> = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    let mut corpus = Sha256::new();
    let mut checked = 0;
    let mut schemes = BTreeSet::new();
    for file in files {
        let bytes = fs::read(&file).unwrap();
        corpus.update(file.file_name().unwrap().to_str().unwrap().as_bytes());
        corpus.update([0]);
        corpus.update(&bytes);
        let cases: Cases = serde_json::from_slice(&bytes).unwrap();
        for case in cases.cases {
            let (Some(encoding), Some(expected)) = (&case.encoding, &case.expect) else {
                continue;
            };
            let label = format!("{}/{}/{}", file.display(), case.component, case.key);
            let buffers: Vec<Vec<u8>> = case
                .parts
                .iter()
                .map(|part| {
                    base64::engine::general_purpose::STANDARD
                        .decode(&part.bits)
                        .unwrap()
                })
                .collect();
            let parts: BTreeMap<_, _> = case
                .parts
                .iter()
                .zip(&buffers)
                .map(|(part, bytes)| {
                    (
                        part.role.as_str(),
                        PartBytes {
                            dtype: &part.dtype,
                            shape: &part.shape,
                            bytes,
                        },
                    )
                })
                .collect();
            let expected = base64::engine::general_purpose::STANDARD
                .decode(expected)
                .unwrap();
            let width = case.dtype.byte_len(&[1]).unwrap();
            let total = expected.len() as u64 / width;
            assert!(expected.len() > 1, "{label}");
            let window_bytes = 13.min(expected.len() - 1) as u64;
            // The installed GPU gate reads an odd byte window starting at offset 1.
            for count in [total, (1 + window_bytes).div_ceil(width)] {
                decode::planned(
                    &case.dtype,
                    &case.shape,
                    encoding,
                    &parts,
                    true,
                    |plan, geometry| {
                        assert!(
                            matches!(
                                kernels::prepare(plan, geometry, &case.dtype, 0, count)?,
                                Prepared::Kernel(_)
                            ),
                            "{label}: CPU decode plus upload is not kernel evidence"
                        );
                        Ok(())
                    },
                )
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            }
            checked += 1;
            schemes.insert(encoding.scheme.clone());
        }
    }
    let digest = format!("{:x}", corpus.finalize());
    assert_eq!(
        digest,
        "475acde968a69d84d1748e26279cedacc8193c3fade193b56f124d424fe27c1b"
    );
    assert_eq!(checked, 227);
    assert_eq!(schemes.len(), 42);
    println!("kernel routing: {checked} full + {checked} odd-byte windows, {} schemes, 0 host fallbacks; corpus {digest}", schemes.len());
}

#[test]
fn host_only_plans_preserve_logical_range_transposition_and_rounding() {
    let g = Geometry {
        rows: 2,
        cols: 3,
        height: 2,
        swap: true,
    };
    let plan = Plan {
        core: Core::Values(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        post: Post::None,
    };
    let Prepared::Host(bytes) = kernels::prepare(plan, g, &Dtype::F32, 1, 3).unwrap() else {
        panic!("host-only plan")
    };
    let expected: Vec<u8> = [3.0f32, 1.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(bytes, expected);

    let g = Geometry {
        rows: 1,
        cols: 3,
        height: 1,
        swap: false,
    };
    let plan = Plan {
        core: Core::Values(vec![1.5, 2.5, f32::NAN]),
        post: Post::One(Elems::f32(2.0)),
    };
    let Prepared::Host(bytes) = kernels::prepare(plan, g, &Dtype::BF16, 1, 2).unwrap() else {
        panic!("host-only plan")
    };
    assert_eq!(bytes, [0xa0, 0x40, 0xc0, 0x7f]);
}

#[test]
fn a_decode_window_is_checked_before_preparing_or_allocating_output() {
    let g = Geometry {
        rows: 1,
        cols: 4,
        height: 1,
        swap: false,
    };
    for (begin, count) in [(4, 1), (u64::MAX, 1), (3, u64::MAX)] {
        let plan = Plan {
            core: Core::Values(vec![]),
            post: Post::None,
        };
        assert!(kernels::prepare(plan, g, &Dtype::F32, begin, count).is_err());
    }
}

// Inspect the emitted CUDA ELF, not the source JIT signature: Triton adds two
// parameters absent from that signature. This check runs on EVERY actual AOT image.
fn check_parameter_abi(image: &[u8], symbol: &str) {
    fn word<const N: usize>(bytes: &[u8], at: usize) -> [u8; N] {
        bytes
            .get(at..at.checked_add(N).unwrap())
            .expect("ELF field outside image")
            .try_into()
            .unwrap()
    }
    fn section(image: &[u8], at: usize) -> &[u8] {
        let offset = usize::try_from(u64::from_le_bytes(word(image, at + 24))).unwrap();
        let size = usize::try_from(u64::from_le_bytes(word(image, at + 32))).unwrap();
        image
            .get(offset..offset.checked_add(size).unwrap())
            .expect("ELF section outside image")
    }
    assert_eq!(&image[..6], b"\x7fELF\x02\x01");
    assert_eq!(u16::from_le_bytes(word(image, 18)), 190, "not CUDA ELF");
    let table = usize::try_from(u64::from_le_bytes(word(image, 40))).unwrap();
    let size = usize::from(u16::from_le_bytes(word(image, 58)));
    let count = usize::from(u16::from_le_bytes(word(image, 60)));
    let strings = usize::from(u16::from_le_bytes(word(image, 62)));
    assert_eq!(size, 64);
    assert!(strings < count);
    let names = section(image, table + size * strings);
    let wanted = format!(".nv.info.{symbol}");
    let entry = (0..count)
        .map(|i| table + i * size)
        .find(|&at| {
            let begin = usize::try_from(u32::from_le_bytes(word(image, at))).unwrap();
            let name = names.get(begin..).expect("ELF name outside string table");
            let end = name
                .iter()
                .position(|b| *b == 0)
                .expect("unterminated ELF name");
            &name[..end] == wanted.as_bytes()
        })
        .expect("kernel parameter attributes missing");
    let info = section(image, entry);
    let (mut at, mut parameters, mut bank) = (0, Vec::new(), None);
    while at < info.len() {
        let [form, attribute, lo, hi] = word(info, at);
        let size = usize::from(u16::from_le_bytes([lo, hi]));
        assert!((1..=4).contains(&form), "unknown CUDA attribute format");
        let end = at
            .checked_add(4 + if form == 4 { size } else { 0 })
            .unwrap();
        assert!(end <= info.len());
        if form == 3 && attribute == 0x19 {
            bank = Some(size);
        }
        if form == 4 && attribute == 0x17 {
            assert_eq!(size, 12);
            let ordinal = u16::from_le_bytes(word(info, at + 8));
            let offset = u16::from_le_bytes(word(info, at + 10));
            let width = u32::from_le_bytes(word(info, at + 12)) >> 18;
            parameters.push((ordinal, offset, width));
        }
        at = end;
    }
    parameters.sort_unstable();
    assert_eq!(
        parameters,
        (0u16..13).map(|i| (i, i * 8, 8u32)).collect::<Vec<_>>(),
        "{symbol}: compiled parameter ABI differs from the thirteen-slot native frame"
    );
    assert_eq!(bank, Some(13 * 8));
}

#[test]
#[ignore = "requires the build-only AOT compile, never a GPU or Triton at runtime"]
fn prebuilt_images_cover_every_native_core_and_family() {
    let kinds: Vec<&str> = CORE_KINDS
        .into_iter()
        .chain(gguf::STRUCTS.iter().map(|s| s.name))
        .collect();
    assert_eq!(BINARIES.len(), FAMILIES.len() * kinds.len() * 3);
    for family in FAMILIES {
        for kind in &kinds {
            for dtype in ["F32", "BF16", "F16"] {
                let images: Vec<_> = BINARIES
                    .iter()
                    .filter(|b| b.family == family && b.kind == *kind && b.dtype == dtype)
                    .collect();
                assert_eq!(images.len(), 1, "sm{family}/{kind}/{dtype}");
                assert!(images[0].image.starts_with(b"\x7fELF"));
                check_parameter_abi(images[0].image, images[0].symbol);
                assert!(!images[0].symbol.is_empty());
                assert_eq!(images[0].warps, 4);
            }
        }
    }
}
