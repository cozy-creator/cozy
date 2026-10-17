use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{tensor_schema_digest_of, tensor_schema_value};

#[test]
fn tensor_schema_is_exact_and_order_independent() {
    let ordered = tensor_schema_value(
        [
            ("model", "bias", Dtype::F32, [2].as_slice()),
            ("model", "weight", Dtype::Bf16, [2, 3].as_slice()),
        ]
        .into_iter(),
    );
    let reversed = tensor_schema_value(
        [
            ("model", "weight", Dtype::Bf16, [2, 3].as_slice()),
            ("model", "bias", Dtype::F32, [2].as_slice()),
        ]
        .into_iter(),
    );
    assert_eq!(ordered, reversed);
    assert_ne!(
        tensor_schema_digest_of(&ordered),
        tensor_schema_digest_of(&tensor_schema_value(
            [("model", "bias", Dtype::F16, [2].as_slice())].into_iter(),
        ))
    );
    assert_ne!(
        tensor_schema_digest_of(&ordered),
        tensor_schema_digest_of(&tensor_schema_value(
            [("model", "bias", Dtype::F32, [3].as_slice())].into_iter(),
        ))
    );
}

#[test]
fn tensor_schema_names_logical_dtype_explicitly() {
    let schema = tensor_schema_value([("model", "weight", Dtype::F32, [2].as_slice())].into_iter());
    let json = String::from_utf8(tensorfs_core::canon::write(&schema)).unwrap();
    assert!(json.contains(r#""logical_dtype":"f32""#), "{json}");
    assert!(!json.contains(r#""dtype":"#), "{json}");
}
