//! CPU artifact identity and shared-host admission; no GPU qualification.
#![cfg(target_os = "linux")] // The host tier (memfd, seals, OFD locks) is Linux-only.
use std::os::fd::AsRawFd;
use tensorfs_core::{
    dtype::Dtype,
    header::{Header, Part, Tensor},
    read,
};
use varena::{
    host::{self, HostMem},
    io::{IoTally, Source},
    layout::Layout,
};

fn layout(bytes: &[u8]) -> Layout {
    let plain = tensorfs_core::registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let header = Header {
        configs: vec![],
        assets: vec![],
        encodings: vec![plain.clone()],
        components: vec![(
            "model".into(),
            vec![(
                "weight".into(),
                Tensor {
                    dtype: Dtype::U8,
                    shape: vec![bytes.len() as u64],
                    encoding: plain.object_id(),
                    parts: vec![(
                        "value".into(),
                        Part::plan(Dtype::U8, vec![bytes.len() as u64], bytes),
                    )],
                },
            )],
        )],
    };
    let plan = read::plan_for_traversal(
        &header,
        &[("model".into(), "weight".into())],
        &["model".into()],
        4 << 20,
    )
    .unwrap();
    Layout::build(&plan, &[vec!["model/weight".into()]]).unwrap()
}

#[test]
fn inline_bytes_distinguish_layouts_at_the_same_offsets() {
    let first = layout(&[1]);
    let second = layout(&[2]);
    assert_eq!(first.nbytes, second.nbytes);
    assert_ne!(first.digest, second.digest);
    assert_eq!(first.digest, layout(&[1]).digest);
}

#[test]
fn object_backed_layout_identity_remains_stable() {
    // Recorded pre-fix identity: object sha/off/len and geometry are unchanged.
    assert_eq!(
        layout(&vec![0x31; 1024]).digest,
        "40561c9382f6af130587487a480fbf2773b3c32342a4810e07f132e94d0b91eb"
    );
}

#[test]
fn host_adoption_rejects_same_shape_with_different_inline_bytes() {
    let first = layout(&[1]);
    let second = layout(&[2]);
    let (fd, fill) = host::open_sealed("inline-identity", &first, None, None).unwrap();
    fill.fill(
        &first,
        &Source::files(vec![], false, 0),
        1,
        &IoTally::default(),
    )
    .unwrap();
    assert!(HostMem::adopt_sealed(fd.as_raw_fd(), &first).is_ok());
    assert!(
        HostMem::adopt_sealed(fd.as_raw_fd(), &second).is_err(),
        "a cached layout cannot supply another inline value"
    );
}
