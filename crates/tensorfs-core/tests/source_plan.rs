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
    members: &[(&str, &str)],
    order_path: &Path,
) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tfs"));
    command.args(["ingest", "bank", dialect, converter, "--registry"]);
    command.arg(registry);
    command.args(["--source-profile", profile, "--order"]);
    command.arg(order_path);
    for (component, path) in sources {
        command.args(["--source", &format!("{component}={}", path.display())]);
        command.args(["--target", &format!("{component}=plain/1")]);
    }
    for (component, member) in members {
        command.args(["--member", &format!("{component}={member}")]);
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

fn write_source_plan(registry: &Path, carriers: &[&Path], output: &Path) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tfs"));
    command.args(["ingest", "source-plan", "--registry"]);
    command.arg(registry);
    command.args(["--out", output.to_str().unwrap()]);
    for carrier in carriers {
        command.args(["--carrier", carrier.to_str().unwrap()]);
    }
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "source-plan --out failed:\n{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

fn string_field(path: &Path, field: &str) -> String {
    let body = fs::read_to_string(path).unwrap();
    let marker = format!(r#""{field}":""#);
    let tail = body.split_once(&marker).unwrap().1;
    tail.split_once('"').unwrap().0.to_string()
}

#[test]
fn h3_profiles_expose_only_the_five_reviewed_carriers() {
    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "ingest",
            "source-members",
            "hf/minimax-h3/native-dual-bf16/1",
            "hf/minimax-h3/shared-bf16/1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "source-members failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let members: Vec<&str> = std::str::from_utf8(&output.stdout)
        .unwrap()
        .lines()
        .collect();
    assert_eq!(
        members,
        vec![
            "FL2VA/transformer/model.safetensors.index.json",
            "Ref2VA/transformer/model.safetensors.index.json",
            "audio_vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors.index.json",
            "vae/diffusion_pytorch_model.safetensors.index.json",
        ]
    );
    assert!(!members.iter().any(|member| {
        member.starts_with("transformer/") || member.starts_with("transformer_ref/")
    }));
}

#[test]
fn source_plan_maps_reviewed_single_file_components_without_its_filename() {
    let root = temporary("single");
    fs::create_dir_all(&root).unwrap();
    let oddly_named = root.join("nothing-about-components.bin");
    let registry = root.join("registry.json");
    let order_path = root.join("order.json");
    let plan_path = root.join("source-plan.json");
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
        &[],
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
    write_source_plan(&registry, &[&oddly_named], &plan_path);
    let preview = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ingest", "plan", "--source-plan"])
        .arg(&plan_path)
        .output()
        .unwrap();
    assert!(
        preview.status.success(),
        "closed preview failed: {}",
        String::from_utf8_lossy(&preview.stderr)
    );
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
    let plan_path = root.join("source-plan.json");
    let store = root.join("store");
    carrier(&first, &["weight"]);
    carrier(&second, &["bias"]);
    order(&order_path, &[("encoder", "weight"), ("decoder", "bias")]);
    bank(
        &registry,
        "multi",
        "diffusers.identity/1",
        "safetensors.diffusers",
        &[("encoder", &first), ("decoder", &second)],
        &[],
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
    write_source_plan(&registry, &[&second, &first], &plan_path);
    assert!(Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "init"])
        .arg(&store)
        .status()
        .unwrap()
        .success());
    let run = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ingest", "run"])
        .arg(&store)
        .args(["--source-plan"])
        .arg(&plan_path)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "closed run failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("candidate    manifest sha256:"));
    let session = string_field(&plan_path, "session");
    let install = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ingest", "install"])
        .arg(&store)
        .args([
            &session,
            "local",
            "planned-model",
            &"11".repeat(32),
            "local",
            "--observed",
            "absent",
            "--source-uri",
            "hf://reviewed/source@1111111111111111111111111111111111111111",
            "--declared-license",
            "test-only",
        ])
        .output()
        .unwrap();
    assert!(
        install.status.success(),
        "closed install failed:\n{}{}",
        String::from_utf8_lossy(&install.stdout),
        String::from_utf8_lossy(&install.stderr)
    );
    assert!(String::from_utf8_lossy(&install.stdout).contains("installed    sha256:"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn source_plan_selects_a_unique_reviewed_subset_from_provider_candidates() {
    let root = temporary("candidate-subset");
    fs::create_dir_all(&root).unwrap();
    let first = root.join("first.bin");
    let second = root.join("second.bin");
    let unrelated = root.join("unrelated.bin");
    let registry = root.join("registry.json");
    let order_path = root.join("order.json");
    carrier(&first, &["weight"]);
    carrier(&second, &["bias"]);
    carrier(&unrelated, &["other"]);
    order(&order_path, &[("encoder", "weight"), ("decoder", "bias")]);
    bank(
        &registry,
        "candidate-subset",
        "diffusers.identity/1",
        "safetensors.diffusers",
        &[("encoder", &first), ("decoder", &second)],
        &[],
        &order_path,
    );

    let output = source_plan(&registry, &[&unrelated, &second, &first]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.matches(first.to_str().unwrap()).count(), 2);
    assert_eq!(text.matches(second.to_str().unwrap()).count(), 2);
    assert_eq!(text.matches(unrelated.to_str().unwrap()).count(), 1);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn reviewed_production_selects_one_named_profile_from_a_shared_source() {
    let root = temporary("named-profile");
    fs::create_dir_all(&root).unwrap();
    let first = root.join("first.bin");
    let second = root.join("second.bin");
    let registry = root.join("registry.json");
    let first_order = root.join("first-order.json");
    let second_order = root.join("second-order.json");
    carrier(&first, &["first"]);
    carrier(&second, &["second"]);
    order(&first_order, &[("first", "first")]);
    order(&second_order, &[("second", "second")]);
    bank(
        &registry,
        "profile-a",
        "diffusers.identity/1",
        "safetensors.diffusers",
        &[("first", &first)],
        &[("first", "provider/first")],
        &first_order,
    );
    bank(
        &registry,
        "profile-b",
        "diffusers.identity/1",
        "safetensors.diffusers",
        &[("second", &second)],
        &[("second", "provider/second")],
        &second_order,
    );

    let mut command = Command::new(env!("CARGO_BIN_EXE_tfs"));
    let selected = command
        .args(["ingest", "source-plan", "--registry"])
        .arg(&registry)
        .args(["--source-profile", "profile-b"])
        .args(["--carrier", &format!("provider/first={}", first.display())])
        .args([
            "--carrier",
            &format!("provider/second={}", second.display()),
        ])
        .output()
        .unwrap();
    assert!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let text = String::from_utf8(selected.stdout).unwrap();
    assert!(text.contains(r#""profile":"profile-b""#));
    assert!(text.contains(second.to_str().unwrap()));
    assert!(!text.contains(r#""component":"first""#));

    let absent = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ingest", "source-plan", "--registry"])
        .arg(&registry)
        .args(["--source-profile", "profile-c"])
        .args([
            "--carrier",
            &format!("provider/second={}", second.display()),
        ])
        .output()
        .unwrap();
    assert!(!absent.status.success());
    assert!(String::from_utf8_lossy(&absent.stderr).contains("reviewed source profile"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn source_plan_uses_the_compiled_product_registry_by_default() {
    let root = temporary("compiled-registry");
    fs::create_dir_all(&root).unwrap();
    let unknown = root.join("unknown.bin");
    carrier(&unknown, &["not.a.reviewed.model.weight"]);
    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ingest", "source-plan", "--carrier"])
        .arg(&unknown)
        .output()
        .unwrap();
    // An unrecognized carrier is a hint miss, not a refusal: it plans, previews and runs as-is.
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "{error}");
    let plan = String::from_utf8(output.stdout).unwrap();
    assert!(plan.contains(r#""profile":"as-is/1""#), "{plan}");
    assert!(plan.contains(r#""converter":"identity/1""#), "{plan}");
    let (plan_path, store) = (root.join("source-plan.json"), root.join("store"));
    fs::write(&plan_path, plan.trim_end()).unwrap();
    let tfs = |args: &[&std::ffi::OsStr]| {
        let run = Command::new(env!("CARGO_BIN_EXE_tfs"))
            .args(args)
            .output()
            .unwrap();
        let said = String::from_utf8_lossy(&run.stderr).to_string();
        assert!(run.status.success(), "{args:?}: {said}");
        String::from_utf8_lossy(&run.stdout).to_string()
    };
    tfs(&[
        "ingest".as_ref(),
        "plan".as_ref(),
        "--source-plan".as_ref(),
        plan_path.as_ref(),
    ]);
    tfs(&["store".as_ref(), "init".as_ref(), store.as_ref()]);
    let ran = tfs(&[
        "ingest".as_ref(),
        "run".as_ref(),
        store.as_ref(),
        "--source-plan".as_ref(),
        plan_path.as_ref(),
    ]);
    assert!(ran.contains("candidate    manifest sha256:"), "{ran}");
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
        &[],
        &order_path,
    );

    let output = source_plan(&registry, &[&first, &second]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("AMBIGUOUS_CLASSIFICATION"), "{error}");
    assert!(
        error.contains("Candidate members:")
            && error.contains("first.bin")
            && error.contains("second.bin"),
        "{error}"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn reviewed_members_resolve_identical_headers_and_missing_members_refuse() {
    let root = temporary("reviewed-members");
    fs::create_dir_all(&root).unwrap();
    let first = root.join("one.bin");
    let second = root.join("two.bin");
    let registry = root.join("registry.json");
    let order_path = root.join("order.json");
    carrier(&first, &["weight"]);
    carrier(&second, &["weight"]);
    order(&order_path, &[("left", "weight"), ("right", "weight")]);
    bank(
        &registry,
        "member-bound",
        "diffusers.identity/1",
        "safetensors.diffusers",
        &[("left", &first), ("right", &second)],
        &[("left", "provider/fl2va"), ("right", "provider/ref2va")],
        &order_path,
    );

    let planned = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ingest", "source-plan", "--registry"])
        .arg(&registry)
        .args([
            "--carrier",
            &format!("provider/ref2va={}", second.display()),
        ])
        .args(["--carrier", &format!("provider/fl2va={}", first.display())])
        .output()
        .unwrap();
    assert!(
        planned.status.success(),
        "member-bound plan failed: {}",
        String::from_utf8_lossy(&planned.stderr)
    );
    let body = String::from_utf8(planned.stdout).unwrap();
    assert!(body.contains(r#""source_member":"provider/fl2va""#));
    assert!(body.contains(r#""source_member":"provider/ref2va""#));

    // Unbound or foreign members never match the member-bound profile: they are stored as-is.
    let missing = source_plan(&registry, &[&first, &second]);
    assert!(String::from_utf8_lossy(&missing.stdout).contains(r#""profile":"as-is/1""#));

    let unreviewed = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ingest", "source-plan", "--registry"])
        .arg(&registry)
        .args(["--carrier", &format!("provider/other={}", first.display())])
        .args([
            "--carrier",
            &format!("provider/ref2va={}", second.display()),
        ])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&unreviewed.stdout).contains(r#""profile":"as-is/1""#));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn provider_bound_and_local_profiles_do_not_steal_each_others_candidates() {
    let root = temporary("provider-local-split");
    fs::create_dir_all(&root).unwrap();
    let first = root.join("first.bin");
    let second = root.join("second.bin");
    let registry = root.join("registry.json");
    let order_path = root.join("order.json");
    carrier(&first, &["weight"]);
    carrier(&second, &["bias"]);
    order(&order_path, &[("encoder", "weight"), ("decoder", "bias")]);
    bank(
        &registry,
        "provider-bound",
        "diffusers.identity/1",
        "safetensors.diffusers",
        &[("encoder", &first), ("decoder", &second)],
        &[
            ("encoder", "provider/encoder"),
            ("decoder", "provider/decoder"),
        ],
        &order_path,
    );
    bank(
        &registry,
        "local-unlabelled",
        "diffusers.identity/1",
        "safetensors.diffusers",
        &[("encoder", &first), ("decoder", &second)],
        &[],
        &order_path,
    );

    let provider = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ingest", "source-plan", "--registry"])
        .arg(&registry)
        .args([
            "--carrier",
            &format!("provider/encoder={}", first.display()),
        ])
        .args([
            "--carrier",
            &format!("provider/decoder={}", second.display()),
        ])
        .output()
        .unwrap();
    assert!(provider.status.success());
    assert!(String::from_utf8_lossy(&provider.stdout).contains(r#""profile":"provider-bound""#));

    let local = source_plan(&registry, &[&first, &second]);
    assert!(local.status.success());
    assert!(String::from_utf8_lossy(&local.stdout).contains(r#""profile":"local-unlabelled""#));

    let foreign = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ingest", "source-plan", "--registry"])
        .arg(&registry)
        .args(["--carrier", &format!("other/encoder={}", first.display())])
        .args(["--carrier", &format!("other/decoder={}", second.display())])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&foreign.stdout).contains(r#""profile":"as-is/1""#));
    let _ = fs::remove_dir_all(root);
}
