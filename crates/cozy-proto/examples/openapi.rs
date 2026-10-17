//! Prints the machine API's OpenAPI spec: `cargo run -p cozy-proto --example openapi >
//! crates/cozy-proto/openapi/machine.json`.
fn main() {
    print!("{}", cozy_proto::machine::api::spec());
}
