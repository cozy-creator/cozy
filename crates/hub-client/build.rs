//! Generates the client from the vendored API document. Operations marked `x-raw-body` whose
//! only success is a redirect (the package index's doors) are left to uv.
fn main() {
    let source = "openapi.yaml";
    println!("cargo:rerun-if-changed={source}");
    let text = std::fs::read_to_string(source).expect("openapi.yaml");
    let mut spec: openapiv3::OpenAPI = serde_yaml::from_str(&text).expect("openapi.yaml parses");
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
            let redirect_only = op.as_ref().is_some_and(|op| {
                op.extensions.get("x-raw-body").and_then(|v| v.as_bool()) == Some(true)
                    && !op
                        .responses
                        .responses
                        .keys()
                        .any(|status| status.to_string().starts_with('2'))
            });
            if redirect_only {
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
