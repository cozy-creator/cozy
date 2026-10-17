//! A machine's leaf thumbprint, against one computed independently (openssl's point, RFC 7638).
use hub_client::leaf_thumbprint;

#[test]
fn a_leaf_names_its_p256_key() {
    let leaf = include_bytes!("leaf/p256.der");
    assert_eq!(
        leaf_thumbprint(leaf).unwrap(),
        "BxNyPIXgGDh5iljLlYlFflIXGiBjI9AeI1MRjCLVYxU"
    );
    assert!(leaf_thumbprint(include_bytes!("leaf/ed25519.der")).is_err());
    assert!(leaf_thumbprint(&leaf[1..]).is_err());
}
