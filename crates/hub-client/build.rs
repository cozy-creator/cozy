//! Generates the client from the Hub's API document (`scripts/hub-api.sh` vendors it unedited).
//! Operations marked `x-raw-body` that take exact bytes (a generated body would be re-encoded) or
//! whose only success is a redirect (the package index's doors, which uv follows) are left out.
//! Timestamps are read as strings.
//!
//! The Hub and this client are released apart, so an answer is read leniently (owner, 2026-10-23):
//! a field the reader does not know is ignored, a missing string, number, bool, list or map is its
//! empty value, a `null` list or map is empty, and a wrong shape names its field (`wire.rs`). A
//! missing object still refuses, by name. Structs an untagged enum picks between keep their
//! required fields: those tell its variants apart.
use std::collections::HashSet;

use saphyr::LoadableYamlNode;
use syn::punctuated::Punctuated;
use syn::visit_mut::{self, VisitMut};

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
    let mut file = syn::parse2(tokens).expect("generated Rust parses");
    let mut untagged = Untagged::default();
    untagged.visit_file_mut(&mut file);
    Lenient(untagged.0).visit_file_mut(&mut file);
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("hub.rs");
    std::fs::write(out, prettyplease::unparse(&file)).expect("write the client");
}

/// The types an untagged enum's variants hold.
#[derive(Default)]
struct Untagged(HashSet<String>);

impl VisitMut for Untagged {
    fn visit_item_enum_mut(&mut self, item: &mut syn::ItemEnum) {
        if item.attrs.iter().any(|a| serde_has(a, "untagged")) {
            let held = item
                .variants
                .iter()
                .flat_map(|v| v.fields.iter())
                .map(|f| last(&f.ty));
            self.0.extend(held);
        }
    }
}

/// Makes every answer type lenient and routes every answer through `wire::from_response`.
struct Lenient(HashSet<String>);

impl VisitMut for Lenient {
    fn visit_item_struct_mut(&mut self, item: &mut syn::ItemStruct) {
        let read = item.attrs.iter().any(|a| {
            a.path().is_ident("derive") && quote::quote!(#a).to_string().contains("Deserialize")
        });
        let syn::Fields::Named(fields) = &mut item.fields else {
            return;
        };
        if !read || self.0.contains(&item.ident.to_string()) {
            return;
        }
        item.attrs.retain(|a| !serde_has(a, "deny_unknown_fields"));
        for field in fields.named.iter_mut() {
            let many = match last(&field.ty).as_str() {
                "Vec" | "HashMap" | "BTreeMap" | "Map" => true,
                "String" | "bool" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64"
                | "f32" | "f64" | "Value" | "Option" => false,
                _ => continue,
            };
            let mut metas: Vec<syn::Meta> = Vec::new();
            if let Some(at) = field.attrs.iter().position(|a| a.path().is_ident("serde")) {
                let attr = field.attrs.remove(at);
                let parsed =
                    attr.parse_args_with(Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated);
                metas.extend(parsed.expect("a serde attribute"));
            }
            let has = |metas: &[syn::Meta], name| metas.iter().any(|m| m.path().is_ident(name));
            if !has(&metas, "default") {
                metas.push(syn::parse_quote!(default));
            }
            if many && !has(&metas, "deserialize_with") {
                metas.push(syn::parse_quote!(
                    deserialize_with = "crate::wire::null_is_empty"
                ));
            }
            field.attrs.push(syn::parse_quote!(#[serde(#(#metas),*)]));
        }
    }

    fn visit_path_mut(&mut self, path: &mut syn::Path) {
        let names: Vec<_> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        if names == ["ResponseValue", "from_response"] {
            *path = syn::parse_quote!(crate::wire::from_response);
        }
        visit_mut::visit_path_mut(self, path);
    }
}

fn serde_has(attr: &syn::Attribute, word: &str) -> bool {
    attr.path().is_ident("serde") && quote::quote!(#attr).to_string().contains(word)
}

/// A type's last path segment: `Vec` for `::std::vec::Vec<T>`.
fn last(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(p) => p
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default(),
        _ => String::new(),
    }
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
