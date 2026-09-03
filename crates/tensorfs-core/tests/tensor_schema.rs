use tensorfs_core::dtype::Dtype;
use tensorfs_core::fit::{Accepts, TensorRequirement, TensorRequirements};
use tensorfs_core::header::{tensor_schema_digest_of, tensor_schema_value, Header, Tensor};

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
fn checkpoint_and_code_requirements_share_one_tensor_schema() {
    let tensor = Tensor {
        dtype: Dtype::F32,
        shape: vec![2],
        encoding: String::new(),
        parts: Vec::new(),
    };
    let header = Header {
        configs: Vec::new(),
        assets: Vec::new(),
        encodings: Vec::new(),
        components: vec![("model".into(), vec![("weight".into(), tensor)])],
    };
    let requirements = TensorRequirements {
        contract: "sha256:".to_string() + &"11".repeat(32),
        components: vec![(
            "model".into(),
            vec![(
                "weight".into(),
                TensorRequirement {
                    dtype: Dtype::F32,
                    shape: vec![2],
                    accepts: Accepts::Any,
                    pinned: false,
                },
            )],
        )],
        constructed_components: vec!["model".into()],
    };
    assert_eq!(
        header.tensor_schema_digest(),
        requirements.tensor_schema_digest()
    );
}
