use tensorfs_core::dtype::Dtype;
use tensorfs_core::fit::{self, Custody, Fit, TensorRequirements};
use tensorfs_core::header::Header;

fn header() -> Header {
    Header::parse(include_bytes!(
        "../../../vectors/cases/header/plain-fp8.cbor"
    ))
    .unwrap()
}

fn requirements(header: &Header) -> TensorRequirements {
    TensorRequirements::new(header.tensors().map(|(component, key, tensor)| {
        (
            component.clone(),
            key.clone(),
            tensor.shape.clone(),
            Some(tensor.dtype),
        )
    }))
    .unwrap()
}

#[test]
fn exact_request_time_fit_needs_no_requirements_document() {
    let header = header();
    let requirements = requirements(&header);
    let answer = fit::fit(&requirements, &header, Custody::Canonical, true, None, None).unwrap();
    assert!(matches!(answer, Fit::Ok { .. }));
}

#[test]
fn shape_and_optional_logical_dtype_are_exact() {
    let header = header();
    let (component, key, tensor) = header.tensors().next().unwrap();
    let wrong_shape =
        TensorRequirements::new([(component.clone(), key.clone(), vec![7], None)]).unwrap();
    assert!(matches!(
        fit::fit(&wrong_shape, &header, Custody::Local, true, None, None).unwrap(),
        Fit::ShapeMismatch { .. }
    ));

    let wrong_dtype = TensorRequirements::new([(
        component.clone(),
        key.clone(),
        tensor.shape.clone(),
        Some(if tensor.dtype == Dtype::F32 {
            Dtype::F16
        } else {
            Dtype::F32
        }),
    )])
    .unwrap();
    assert!(matches!(
        fit::fit(&wrong_dtype, &header, Custody::Local, true, None, None).unwrap(),
        Fit::DtypeMismatch { .. }
    ));
}

#[test]
fn encoded_leaf_consent_is_one_request_flag() {
    let header = header();
    let requirements = requirements(&header);
    assert!(matches!(
        fit::fit(
            &requirements,
            &header,
            Custody::Canonical,
            false,
            None,
            None
        )
        .unwrap(),
        Fit::EncodingUnsupported { .. }
    ));
}

#[test]
fn duplicate_rows_refuse_before_fit() {
    let row = ("model".to_string(), "weight".to_string(), vec![2], None);
    let error = TensorRequirements::new([row.clone(), row]).unwrap_err();
    assert_eq!(error.code, tensorfs_core::err::Code::DUPLICATE_KEY);
}

/// A stored tensor the code does not build is skipped on every custody, and the verdict
/// carries one warning: how many, how many bytes, and the first few keys.
#[test]
fn stored_tensors_the_code_does_not_build_are_skipped_with_one_warning() {
    let plain = tensorfs_core::registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    // Eight tensors: seven small inline ones and one 1.5 MiB one stored as a segment.
    let tensors = (0..8u64)
        .map(|index| {
            let bytes = if index == 0 { 3 << 19 } else { 4 * (index + 1) };
            let part = tensorfs_core::header::Part::plan(
                Dtype::F32,
                vec![bytes / 4],
                &vec![index as u8; bytes as usize],
            );
            let tensor = tensorfs_core::header::Tensor {
                dtype: Dtype::F32,
                shape: vec![bytes / 4],
                encoding: plain.object_id(),
                parts: vec![("value".into(), part)],
            };
            (format!("block.{index}.weight"), tensor)
        })
        .collect();
    let header = Header {
        configs: Vec::new(),
        assets: Vec::new(),
        encodings: vec![plain],
        components: vec![("model".into(), tensors)],
    };
    let header = Header::parse(&header.canonical_bytes().unwrap()).unwrap();
    let stored: Vec<_> = header.tensors().collect();
    let bytes = |tensor: &tensorfs_core::header::Tensor| -> u64 {
        tensor
            .parts
            .iter()
            .map(|(_, part)| match &part.body {
                tensorfs_core::header::Body::Segments(objects) => {
                    objects.iter().map(|object| object.length).sum()
                }
                tensorfs_core::header::Body::Inline(inline) => inline.len() as u64,
            })
            .sum()
    };
    for skipped in [1, 7] {
        let built = TensorRequirements::new(
            stored[skipped..]
                .iter()
                .map(|(c, k, t)| ((*c).clone(), (*k).clone(), t.shape.clone(), Some(t.dtype))),
        )
        .unwrap();
        let mut names: Vec<String> = stored[..skipped]
            .iter()
            .map(|(c, k, _)| format!("{c}/{k}"))
            .collect();
        names.sort();
        let total: u64 = stored[..skipped].iter().map(|(_, _, t)| bytes(t)).sum();
        for custody in [Custody::Canonical, Custody::Local] {
            let answer = fit::fit(&built, &header, custody, true, None, None).unwrap();
            let Fit::Ok {
                ignored,
                ignored_bytes,
                ..
            } = &answer
            else {
                panic!("{}", answer.text());
            };
            assert_eq!((ignored, *ignored_bytes), (&names, total));
            let warning = answer.warning().unwrap();
            assert!(
                warning.starts_with(&format!(
                    "{skipped} stored tensor(s) the code does not build were skipped, not \
                     loaded: {total} B ({}",
                    names[0]
                )),
                "{warning}"
            );
            assert_eq!(warning.contains("and 2 more"), skipped == 7, "{warning}");
            assert!(answer.text().ends_with(&warning));
        }
    }
    // Nothing skipped, nothing to say.
    let all = requirements(&header);
    assert_eq!(all.count(), 8);
    let answer = fit::fit(&all, &header, Custody::Canonical, true, None, None).unwrap();
    assert_eq!(answer.warning(), None);
}

/// The other half is unchanged: a key the code builds must be stored.
#[test]
fn a_built_key_the_checkpoint_lacks_still_refuses() {
    let header = header();
    let (component, _, tensor) = header.tensors().next().unwrap();
    let built = TensorRequirements::new([(
        component.clone(),
        "never.stored".to_string(),
        tensor.shape.clone(),
        None,
    )])
    .unwrap();
    assert!(matches!(
        fit::fit(&built, &header, Custody::Local, true, None, None).unwrap(),
        Fit::TensorMissing { .. }
    ));
}
