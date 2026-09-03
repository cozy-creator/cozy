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
