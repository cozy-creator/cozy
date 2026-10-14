//! Actual CPU executor/socket/result/receipt path; no machine-global SDK attribution.
use super::*;
use crate::{device_executor::Baseline, journal::{Invocation, Journal}};
use std::process::Command;

#[test]
#[ignore = "actual CPU provenance gate requires an explicitly installed cpu_software SDK fixture"]
fn a_successful_cpu_attempt_keeps_its_actual_software_after_reload() {
    let python = PathBuf::from(std::env::var("TENSORD_SOFTWARE_TEST_PYTHON")
        .expect("explicit interpreter with the cpu_software fixture installed"));
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_software");
    let root = std::env::temp_dir().join(format!("cm-software-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let described = Command::new(python.with_file_name("cozy-runtime")).args(["--json", "--dir"])
        .arg(fixture).arg("describe").output().unwrap();
    assert!(described.status.success(), "{}", String::from_utf8_lossy(&described.stderr));
    let interface = root.join("interface.json");
    fs::write(&interface, &described.stdout).unwrap();
    let engine = Engine::open(&root.join("state")).unwrap();
    let run = engine.submit("software", Invocation { job: true, ..Default::default() }).unwrap();
    engine.with_journal(|journal| journal.claim(&run.id)).unwrap();
    let mut executor = DeviceExecutor::spawn_observed(ExecutorConfig {
        python, root: root.join("executor"), socket: root.join("e.sock"),
        environment: Default::default(), seal: Seal::prepare(&root, None, "software", "fixture", "").unwrap(),
        generation_hold: None, identity: None, cgroup_namespace: None,
    }, |birth, cancel| {
        let cancel = cancel.clone();
        let id = run.id.clone();
        engine.register_managed(&run.id, birth.clone(), Arc::new(move || cancel.cancel(&id)))
    }).unwrap();
    let facts = ExecutorFacts::observed(executor.birth.pid, executor.hello.runtime_version.clone(),
        executor.hello.tensorfs_version.clone());
    let expected = facts.software(1);
    assert!(engine.authorize_managed(&run.id, Some(facts)).unwrap());
    let maps = fs::read_to_string(format!("/proc/{}/maps", executor.birth.pid)).unwrap();
    assert!(!maps.contains("libcuda") && !maps.contains("libnvidia-ml"));
    let spool = engine.staging(&run.id).unwrap();
    let reply = executor.command(&DeviceCommand::RunJob {
        max_output_bytes: crate::output_capacity::DEFAULT_BYTES, request_id: run.id.clone(),
        job: "compute".into(), payload: json!({"value": 21}), application: "cpu_software:app".into(),
        package_interface: interface, spool: spool.clone(), scratch: root.join("scratch"), deadline_s: None,
        inputs: Default::default(), trees: Default::default(), call_interfaces: vec![],
        models: Default::default(), weights: false, budget: None,
    }, &mut Baseline).unwrap();
    assert_eq!(reply.outcome.as_ref().unwrap().terminal, "succeeded", "{reply:?}");
    conclude(&engine, &run.id, &executor, &spool, reply).unwrap();
    executor.shutdown().unwrap();
    let completed = engine.get(&run.id).unwrap();
    assert_eq!(completed.state, State::Completed);
    assert_eq!(completed.result.as_ref().unwrap().value["doubled"], 42);
    assert_eq!(completed.result.as_ref().unwrap().value["runtime"], expected.runtime);
    let measured: Value = serde_json::from_slice(&engine.with_journal(|j| j.measurements(&run.id))
        .unwrap().unwrap()).unwrap();
    assert_eq!(measured["software"], serde_json::to_value(&expected).unwrap());
    assert_eq!(expected.tensord, env!("CARGO_PKG_VERSION"));
    assert_eq!(expected.tensord_commit.as_deref(), option_env!("TENSORD_COMMIT"));
    drop(engine);
    let journal = Journal::open(&root.join("state")).unwrap();
    let reopened: Value = serde_json::from_slice(&journal.measurements(&run.id).unwrap().unwrap()).unwrap();
    assert_eq!(reopened["software"], measured["software"]);
    eprintln!("CPU attempt software: {}", measured["software"]);
    drop(journal);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn legacy_executor_facts_do_not_acquire_the_current_tensord_identity() {
    let facts: ExecutorFacts = serde_json::from_str(r#"{"pid":123,"runtime_version":"old-sdk","tensorfs_version":"old-tfs"}"#).unwrap();
    let software = serde_json::to_value(facts.software(2)).unwrap();
    assert!(software.get("tensord").is_none() && software.get("tensord_commit").is_none());
    assert_eq!(software["runtime"], "old-sdk");
    assert_eq!(software["attempt_ordinal"], 2);
}
