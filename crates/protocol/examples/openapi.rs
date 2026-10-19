//! Prints the machine API's OpenAPI spec: `cargo run -p protocol --example openapi >
//! crates/protocol/openapi/machine.json`.
fn main() {
    print!("{}", protocol::machine::api::spec());
}
