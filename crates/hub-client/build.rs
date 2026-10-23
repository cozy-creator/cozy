//! Generates the client from the vendored API document. Operations marked `x-raw-body` whose
//! only success is a redirect (the package index's doors) are left to uv, and those that take
//! exact bytes are hand-written (`raw.rs`): a generated body would be re-encoded. Timestamps are read
//! as strings: one malformed optional field (an older Hub's `"cut_at": ""`) must never reject a
//! whole answer, while the Hub's own conformance tests still hold it to `date-time`.
use saphyr::LoadableYamlNode;

fn main() {
    let source = "openapi.yaml";
    println!("cargo:rerun-if-changed={source}");
    let text = std::fs::read_to_string(source).expect("openapi.yaml");
    let documents = saphyr::Yaml::load_from_str(&text).expect("openapi.yaml parses");
    let mut document = json(&documents[0]);
    plain_timestamps(&mut document);
    let mut spec: openapiv3::OpenAPI =
        serde_json::from_value(document).expect("openapi.yaml is OpenAPI");
    for item in spec.paths.paths.values_mut() {
        let openapiv3::ReferenceOr::Item(item) = item else {
            continue;
        };
        for op in [
            &mut item.get,
            &mut item.put,
            &mut item.post,
            &mut item.delete,
            &mut item.patch,
        ] {
            let skipped = op.as_ref().is_some_and(|op| {
                op.extensions.get("x-raw-body").and_then(|v| v.as_bool()) == Some(true)
                    && (op.request_body.is_some()
                        || !op
                            .responses
                            .responses
                            .keys()
                            .any(|status| status.to_string().starts_with('2')))
            });
            if skipped {
                *op = None;
            }
        }
    }
    let mut settings = progenitor::GenerationSettings::default();
    settings
        .with_interface(progenitor::InterfaceStyle::Positional)
        .with_tag(progenitor::TagStyle::Merged)
        .with_inner_type(quote::quote!(crate::Auth))
        .with_hooks(progenitor::HooksMode::Expected);
    let tokens = progenitor::Generator::new(&settings)
        .generate_tokens(&spec)
        .expect("the client generates");
    let file = syn::parse2(tokens).expect("generated Rust parses");
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("hub.rs");
    std::fs::write(out, prettyplease::unparse(&file)).expect("write the client");
}

/// Drops every `format: date-time`, so the generated field is a `String`.
fn plain_timestamps(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            if map.get("format").and_then(|f| f.as_str()) == Some("date-time") {
                map.remove("format");
            }
            map.values_mut().for_each(plain_timestamps);
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(plain_timestamps),
        _ => {}
    }
}

/// The document as JSON (1.2 core schema, aliases resolved; a key is its scalar's text).
fn json(node: &saphyr::Yaml) -> serde_json::Value {
    use saphyr::{Scalar, Yaml};
    use serde_json::Value;
    match node {
        Yaml::Value(Scalar::Null) => Value::Null,
        Yaml::Value(Scalar::Boolean(b)) => Value::Bool(*b),
        Yaml::Value(Scalar::Integer(i)) => Value::from(*i),
        Yaml::Value(Scalar::FloatingPoint(f)) => Value::from(f.into_inner()),
        Yaml::Value(Scalar::String(s)) | Yaml::Representation(s, ..) => Value::String(s.to_string()),
        Yaml::Sequence(items) => Value::Array(items.iter().map(json).collect()),
        Yaml::Mapping(map) => Value::Object(
            map.iter()
                .map(|(k, v)| match json(k) {
                    Value::String(key) => (key, json(v)),
                    key => (key.to_string(), json(v)),
                })
                .collect(),
        ),
        Yaml::Tagged(_, inner) => json(inner),
        Yaml::Alias(_) | Yaml::BadValue => panic!("openapi.yaml has an alias to no anchor"),
    }
}
