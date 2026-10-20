#![cfg(feature = "store")]

use tensorfs::err::{Code, Refusal};
use tensorfs::failure::{Failure, Kind};
use tensorfs::store2::Error;

#[test]
fn transport_causes_keep_their_public_family_through_a_writer_error() {
    for (code, expected) in [(Code::CREDENTIAL_REQUIRED, Kind::Denied), (Code::TRANSFER_FAILED, Kind::Network), (Code::REPOSITORY_CONFLICT, Kind::Conflict)] {
        let error = Error::Io { op: "send", path: "the endpoint".into(), source: std::io::Error::other(Refusal { code, detail: "refused by the endpoint".into() }) };
        let failure = Failure::from(error);
        assert_eq!(failure.kind, expected);
        assert!(failure.detail.contains(code.as_str()));
    }
}

#[test]
fn a_wrapped_cuda_allocation_failure_keeps_its_family_and_driver_cause() {
    let cuda = tensorfs::io::cuda::Error::Cuda {
        op: "cuMemHostAlloc", code: 2, name: "CUDA_ERROR_OUT_OF_MEMORY".into(),
    };
    let error = Error::Io { op: "read", path: "tensor".into(), source: std::io::Error::other(cuda) };
    let failure = Failure::from(error);
    assert_eq!(failure.kind, Kind::Full);
    assert!(failure.detail.contains("cuMemHostAlloc") && failure.detail.contains("2") && failure.detail.contains("CUDA_ERROR_OUT_OF_MEMORY"));
}
