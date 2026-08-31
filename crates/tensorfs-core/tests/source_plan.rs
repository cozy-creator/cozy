use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-source-plan-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn carrier(path: &Path, keys: &[&str]) {
    let mut fields = Vec::new();
    for (index, key) in keys.iter().enumerate() {
        let begin = index * 4;
        fields.push(format!(
            "{key:?}:{{\"data_offsets\":[{begin},{}],\"dtype\":\"F32\",\"shape\":[1]}}",
            begin + 4
        ));
    }
    let header = format!("{{{}}}", fields.join(","));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(bytes.len() + keys.len() * 4, 0);
    fs::write(path, bytes).unwrap();
}

fn order(path: &Path, rows: &[(&str, &str)]) {
    let body = format!(
        "[{}]",
        rows.iter()
            .map(|(component, key)| format!("[{component:?},{key:?}]"))
            .collect::<Vec<_>>()
            .join(",")
    );
    fs::write(path, body).unwrap();
}

fn bank(
    registry: &Path,
    profile: &str,
    converter: &str,
    dialect: &str,
    sources: &[(&str, &Path)],
    order_path: &Path,
) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tfs"));
    command.args([
        "ingest",
        "bank",
        dialect,
        "source-plan-test",
        profile,
        converter,
        "--registry",
    ]);
    command.arg(registry);
    command.args(["--source-profile", profile, "--order"]);
    command.arg(order_path);
    for (component, path) in sources {
        command.args(["--source", &format!("{component}={}", path.display())]);
        command.args(["--target", &format!("{component}=plain/1")]);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "bank failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn source_plan(registry: &Path, carriers: &[&Path]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tfs"));
    command.args(["ingest", "source-plan", "--registry"]);
    command.arg(registry);
    for carrier in carriers {
        command.args(["--carrier", carrier.to_str().unwrap()]);
    }
    command.output().unwrap()
}

#[test]
fn source_plan_maps_reviewed_single_file_components_without_its_filename() {
    let root = temporary("single");
    fs::create_dir_all(&root).unwrap();
    let oddly_named = root.join("nothing-about-components.bin");
    let registry = root.join("registry.json");
    let order_path = root.join("order.json");
    carrier(
        &oddly_named,
        &["model.diffusion_model.weight", "first_stage_model.weight"],
    );
    order(&order_path, &[("unet", "weight"), ("vae", "weight")]);
    bank(
        &registry,
        "single",
        "single_file.identity/1",
        "safetensors.single_file",
        &[("unet", &oddly_named), ("vae", &oddly_named)],
        &order_path,
    );

    let output = source_plan(&registry, &[&oddly_named]);
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains(r#""profile":"single""#));
    assert!(text.contains(r#""component":"unet""#));
    assert!(text.contains(r#""component":"vae""#));
    assert_eq!(text.matches(r#""projected":true"#).count(), 2);
    assert!(text.contains(r#""target":"unet=plain/1,vae=plain/1""#));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn source_plan_maps_reviewed_multi_file_headers_independent_of_argument_order() {
    let root = temporary("multi");
    fs::create_dir_all(&root).unwrap();
    let first = root.join("z.bin");
    let second = root.join("a.bin");
    let registry = root.join("registry.json");
    let order_path = root.join("order.json");
    carrier(&first, &["weight"]);
    carrier(&second, &["bias"]);
    order(&order_path, &[("encoder", "weight"), ("decoder", "bias")]);
    bank(
        &registry,
        "multi",
        "diffusers.identity/1",
        "safetensors.diffusers",
        &[("encoder", &first), ("decoder", &second)],
        &order_path,
    );

    let output = source_plan(&registry, &[&second, &first]);
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains(&format!(
        r#""component":"encoder","path":{:?}"#,
        first.display().to_string()
    )));
    assert!(text.contains(&format!(
        r#""component":"decoder","path":{:?}"#,
        second.display().to_string()
    )));
    assert_eq!(text.matches(r#""projected":false"#).count(), 2);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn source_plan_refuses_identical_header_component_permutations() {
    let root = temporary("ambiguous");
    fs::create_dir_all(&root).unwrap();
    let first = root.join("first.bin");
    let second = root.join("second.bin");
    let registry = root.join("registry.json");
    let order_path = root.join("order.json");
    carrier(&first, &["weight"]);
    carrier(&second, &["weight"]);
    order(&order_path, &[("left", "weight"), ("right", "weight")]);
    bank(
        &registry,
        "ambiguous",
        "diffusers.identity/1",
        "safetensors.diffusers",
        &[("left", &first), ("right", &second)],
        &order_path,
    );

    let output = source_plan(&registry, &[&first, &second]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("AMBIGUOUS_CLASSIFICATION"), "{error}");
    assert!(
        error.contains("provider member/slot review is required"),
        "{error}"
    );
    let _ = fs::remove_dir_all(root);
}
