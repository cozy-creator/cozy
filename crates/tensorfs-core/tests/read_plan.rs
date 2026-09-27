use tensorfs_core::dtype::Dtype;
use tensorfs_core::err::Code;
use tensorfs_core::header::{Header, Part, Tensor};
use tensorfs_core::read::{plan_for_traversal, Source};
use tensorfs_core::registry;

fn header() -> Header {
    let plain = registry::seeds()
        .into_iter()
        .find(|s| s.alias == "plain/1")
        .unwrap()
        .spec;
    let tensors = [("bias", 4), ("weight", 2048), ("tail", 8)]
        .into_iter()
        .map(|(name, bytes)| {
            (
                name.into(),
                Tensor {
                    dtype: Dtype::F32,
                    shape: vec![bytes / 4],
                    encoding: plain.object_id(),
                    parts: vec![(
                        "value".into(),
                        Part::plan(Dtype::F32, vec![bytes / 4], &vec![0x31; bytes as usize]),
                    )],
                },
            )
        })
        .collect();
    let header = Header {
        configs: Vec::new(),
        assets: Vec::new(),
        encodings: vec![plain],
        components: vec![("model".into(), tensors)],
    };
    Header::parse(&header.canonical_bytes().unwrap()).unwrap()
}

#[test]
fn selecting_parts_preserves_native_windows_and_rebases_only_destinations() {
    let header = header();
    let traversal = ["bias", "weight", "tail"].map(|key| ("model".into(), key.into()));
    let full = plan_for_traversal(&header, &traversal, &["model".into()], 128).unwrap();
    let selected = full
        .select(&["model/tail#value".into(), "model/weight#value".into()])
        .unwrap();
    assert_eq!(
        (selected.bytes, selected.io_bytes, selected.inline_parts),
        (2056, 2048, 1)
    );
    assert_eq!(selected.order, full.order);
    assert_eq!(selected.items.len(), 17);
    let expected = &full.items[1..];
    let mut offset = 0;
    for (selected, original) in selected.items.iter().zip(expected) {
        assert_eq!(
            (&selected.what, selected.len),
            (&original.what, original.len)
        );
        assert_eq!(selected.dest_off, offset);
        match (&selected.source, &original.source) {
            (Source::Object(left), Source::Object(right)) => assert_eq!(left, right),
            (Source::Inline(left), Source::Inline(right)) => assert_eq!(left, right),
            _ => panic!("selection changed an item's source kind"),
        }
        offset += selected.len;
    }
    assert_eq!(full.items[1].dest_off, 4, "the original plan is unchanged");
    assert_eq!(
        selected.batches(512).iter().map(|p| p.bytes).sum::<u64>(),
        selected.bytes
    );
}

#[test]
fn selection_does_not_weaken_the_complete_plan_gate() {
    let header = header();
    let traversal = ["bias", "weight", "tail"].map(|key| ("model".into(), key.into()));
    assert_eq!(
        plan_for_traversal(&header, &traversal[1..], &["model".into()], 128)
            .unwrap_err()
            .code,
        Code::TRAVERSAL_INCOMPLETE
    );
    let full = plan_for_traversal(&header, &traversal, &["model".into()], 128).unwrap();
    assert_eq!(
        full.select(&["model/weight#value".into(), "model/weight#value".into()])
            .unwrap_err()
            .code,
        Code::DUPLICATE_KEY
    );
    assert_eq!(
        full.select(&["model/weight#scale".into()])
            .unwrap_err()
            .code,
        Code::MISSING_FIELD
    );
    let empty = full.select(&[]).unwrap();
    assert!(empty.items.is_empty());
    assert_eq!((empty.bytes, empty.io_bytes, empty.inline_parts), (0, 0, 0));
    assert!(empty.batches(512).is_empty());
    assert_eq!(
        empty
            .select(&["model/weight#value".into()])
            .unwrap_err()
            .code,
        Code::MISSING_FIELD
    );
}
