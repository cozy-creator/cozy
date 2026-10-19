use tensorfs_core::format::decode::plan::{Core, Elems, Plan, Post};
use tensorfs_core::format::decode::{gguf, Geometry};
use tensorfs_core::format::Dtype;
use tensorfs_core::kernels::{self, Prepared, BINARIES, FAMILIES};

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

#[test]
#[ignore = "requires the build-only AOT compile, never a GPU or Triton at runtime"]
fn prebuilt_images_cover_every_native_core_and_family() {
    let kinds: Vec<&str> = ["codes", "unpack", "hadamard", "rotated"]
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
                assert!(!images[0].symbol.is_empty());
                assert_eq!(images[0].warps, 4);
            }
        }
    }
}
