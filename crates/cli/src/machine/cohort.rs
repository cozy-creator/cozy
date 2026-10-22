//! The binary/SDK memory-protocol boundary. These are independent release floors, not a
//! patch-version lockstep; package metadata still resolves the actual SDK dependencies.
use protocol::{calcifer::MachineStatus, error::{Error, Kind}};
use serde_json::{Value, json};

const RUNTIME: &str = "0.31.0";
const TENSORFS: &str = "0.14.0";

fn at_least(version: &str, floor: (u32, u32)) -> bool {
    let mut parts = version.split('.');
    let band = parts.next().and_then(|v| v.parse().ok())
        .zip(parts.next().and_then(|v| v.parse().ok()));
    band.is_some_and(|v| v >= floor)
}

fn compatible(runtime: &str, tensorfs: &str) -> bool {
    at_least(runtime, (0, 31)) && at_least(tensorfs, (0, 14))
}

/// A binary activation must carry both SDK choices when the selected pair predates its ABI.
/// Already-compatible pairs stay selected, even when the Hub's target is older.
pub(super) fn needs_pair(version: &str, status: Option<&MachineStatus>) -> bool {
    at_least(version, (0, 12)) && !status.is_some_and(|s| compatible(&s.runtime, &s.tensorfs))
}

pub(super) fn adoption(version: &str, status: Option<&MachineStatus>, target: Option<(&str, &str)>) -> Value {
    let mut payload = json!({"cozy": version});
    if needs_pair(version, status) {
        let (runtime, tensorfs) = target.filter(|(r, t)| compatible(r, t)).unwrap_or((RUNTIME, TENSORFS));
        payload["runtime"] = runtime.into();
        payload["tensorfs"] = tensorfs.into();
    }
    payload
}

/// SDK-only updates cannot move an old (or unreported) TensorD across the protocol boundary.
/// An explicit dev binary is checked and activated by the existing coherent-update path.
pub(super) fn require_binary(
    name: &str,
    status: Option<&MachineStatus>,
    runtime: &str,
    tensorfs: &str,
    binary: bool,
) -> Result<(), Error> {
    if binary || status.is_some_and(|s| at_least(&s.version, (0, 12)))
        || (!at_least(runtime, (0, 31)) && !at_least(tensorfs, (0, 14)))
    { return Ok(()); }
    let version = status.map(|s| s.version.as_str()).filter(|v| !v.is_empty()).unwrap_or("unknown");
    Err(Error::new(
        Kind::Conflict,
        "memory_protocol_incompatible",
        format!("{name} runs cozy {version}; Runtime >=0.31 or TensorFS >=0.14 requires cozy >=0.12 and both SDKs in the same upgrade"),
    ).next(format!("run `cozy update` on {name}'s host using cozy >=0.11.5, or stage --tensord with both SDK wheels in one `cozy machine update`")))
}

/// Before replacing the CLI, a stopped machine with a selected old SDK must first run the
/// bridge that can activate both choices. Never start a service or rewrite its selection here.
pub(super) fn stopped_selection(version: &str, root: &std::path::Path, stopped: bool) -> Result<(), Error> {
    if !stopped || !at_least(version, (0, 12)) { return Ok(()); }
    let selected = root.join("var/lib/cozy/tensord/sdk/current");
    if matches!(std::fs::symlink_metadata(&selected), Err(e) if e.kind() == std::io::ErrorKind::NotFound) {
        return Ok(()); // No persisted selection: fresh startup owns its SDK resolution.
    }
    let pair = || -> Option<(String, String)> {
        let (mut runtime, mut tensorfs) = (None, None);
        for entry in std::fs::read_dir(&selected).ok()? {
            let entry = entry.ok()?;
            let file = entry.file_name();
            let Some(file) = file.to_str().and_then(|v| v.strip_suffix(".whl")) else { continue };
            let mut fields = file.split('-');
            let distribution = fields.next()?;
            let destination = match distribution {
                "cozy_runtime" => &mut runtime,
                "tensorfs" => &mut tensorfs,
                _ => continue,
            };
            // A selected pair has one wheel of each distribution. Ambiguous/missing bytes
            // cannot prove compatibility; leave that selection untouched for its old daemon.
            if destination.is_some() || !entry.path().is_file() { return None; }
            *destination = Some(fields.next()?.to_owned());
        }
        Some((runtime?, tensorfs?))
    };
    if pair().is_some_and(|(r, t)| compatible(&r, &t)) { return Ok(()); }
    Err(Error::new(Kind::Conflict, "memory_protocol_incompatible",
        format!("the stopped local machine has an old or unreadable SDK selection at {}; cozy {version} cannot activate its replacement while the machine is stopped", selected.display()))
        .next("with cozy 0.11.5, run `cozy machine enable`, then retry `cozy update`; this refusal leaves the binary and stopped machine unchanged"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(cozy: &str, runtime: &str, tensorfs: &str) -> MachineStatus {
        MachineStatus { version: cozy.into(), runtime: runtime.into(), tensorfs: tensorfs.into(), ..Default::default() }
    }

    #[test]
    fn a_binary_upgrade_carries_both_published_floors_when_the_hub_is_old_or_unavailable() {
        let old = status("0.11.4", "0.30.3", "0.13.4");
        for target in [None, Some(("0.30.3", "0.13.4")), Some(("0.31.5", "0.13.4")), Some(("0.30.3", "0.14.2"))] {
            assert_eq!(adoption("0.12.0", Some(&old), target), json!({"cozy":"0.12.0","runtime":"0.31.0","tensorfs":"0.14.0"}));
        }
    }

    #[test]
    fn a_compatible_hub_pair_is_selected_together() {
        let old = status("0.11.4", "0.30.3", "0.13.4");
        assert_eq!(adoption("0.12.0", Some(&old), Some(("0.31.5", "0.14.2"))),
            json!({"cozy":"0.12.0","runtime":"0.31.5","tensorfs":"0.14.2"}));
    }

    #[test]
    fn partial_or_unknown_current_pairs_are_never_carried_across_the_boundary() {
        for current in [None, Some(status("0.11.4", "0.31.5", "0.13.4")), Some(status("0.11.4", "0.30.3", "0.14.2")), Some(status("0.11.4", "", ""))] {
            assert!(needs_pair("0.12.0", current.as_ref()));
            assert_eq!(adoption("0.12.0", current.as_ref(), None), json!({"cozy":"0.12.0","runtime":"0.31.0","tensorfs":"0.14.0"}));
        }
    }

    #[test]
    fn already_compatible_newer_pairs_and_legacy_binary_updates_stay_selected() {
        let newer = status("0.12.3", "0.32.1", "0.15.2");
        assert!(!needs_pair("0.12.4", Some(&newer)));
        assert_eq!(adoption("0.12.4", Some(&newer), Some(("0.31.0", "0.14.0"))), json!({"cozy":"0.12.4"}));
        assert_eq!(adoption("0.11.4", None, Some(("0.31.0", "0.14.0"))), json!({"cozy":"0.11.4"}));
    }

    #[test]
    fn sdk_only_crossings_refuse_old_and_unknown_binaries_before_submission() {
        for current in [None, Some(status("", "0.30.3", "0.13.4")), Some(status("0.11.4", "0.30.3", "0.13.4"))] {
            for (runtime, tensorfs) in [("0.31.0", "0.14.0"), ("0.31.0", ""), ("", "0.14.0")] {
                let error = require_binary("local", current.as_ref(), runtime, tensorfs, false).unwrap_err();
                assert_eq!((error.kind, error.code.as_str()), (Kind::Conflict, "memory_protocol_incompatible"));
                assert!(error.message.contains("both SDKs"));
            }
        }
    }

    #[test]
    fn existing_new_binaries_explicit_dev_updates_and_legacy_targets_remain_supported() {
        let old = status("0.11.4", "0.30.3", "0.13.4");
        let new = status("0.12.1+dev.g123", "0.31.0", "0.14.0");
        require_binary("local", Some(&old), "0.30.4", "0.13.5", false).unwrap();
        require_binary("local", Some(&old), "0.31.0", "0.14.0", true).unwrap();
        require_binary("local", Some(&new), "0.31.2", "0.14.1", false).unwrap();
    }

    fn selected(root: &std::path::Path, runtime: &str, tensorfs: &str) -> std::path::PathBuf {
        let dir = root.join("var/lib/cozy/tensord/sdk/current");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, version) in [("cozy_runtime", runtime), ("tensorfs", tensorfs)] {
            std::fs::write(dir.join(format!("{name}-{version}-py3-none-any.whl")), b"fixture").unwrap();
        }
        dir
    }

    #[test]
    fn a_stopped_old_selection_refuses_before_replacement_without_changing_state() {
        let root = tempfile::tempdir().unwrap();
        let dir = selected(root.path(), "0.30.3", "0.13.4");
        let binary = root.path().join("cozy");
        std::fs::write(&binary, b"old binary").unwrap();
        let error = stopped_selection("0.12.0", root.path(), true).unwrap_err();
        assert_eq!(error.code, "memory_protocol_incompatible");
        assert_eq!(std::fs::read(&binary).unwrap(), b"old binary");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        assert!(!root.path().join("var/lib/cozy/machine/agent.lock").exists());
        stopped_selection("0.12.0", root.path(), false).unwrap(); // Running bridge uses coherent activation.
        stopped_selection("0.11.4", root.path(), true).unwrap();
    }

    #[test]
    fn stopped_compatible_and_unselected_machines_pass_the_read_only_preflight() {
        let root = tempfile::tempdir().unwrap();
        stopped_selection("0.12.0", root.path(), true).unwrap();
        assert!(!root.path().join("var").exists());
        selected(root.path(), "0.31.2", "0.14.3");
        stopped_selection("0.12.0", root.path(), true).unwrap();
    }

    #[test]
    fn incomplete_or_ambiguous_selected_pairs_refuse() {
        let root = tempfile::tempdir().unwrap();
        let dir = selected(root.path(), "0.31.2", "0.14.3");
        std::fs::remove_file(dir.join("tensorfs-0.14.3-py3-none-any.whl")).unwrap();
        assert!(stopped_selection("0.12.0", root.path(), true).is_err());
        selected(root.path(), "0.31.3", "0.14.3");
        assert!(stopped_selection("0.12.0", root.path(), true).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn a_broken_selected_symlink_is_not_a_fresh_install() {
        let root = tempfile::tempdir().unwrap();
        let sdk = root.path().join("var/lib/cozy/tensord/sdk");
        std::fs::create_dir_all(&sdk).unwrap();
        std::os::unix::fs::symlink("missing", sdk.join("current")).unwrap();
        assert!(stopped_selection("0.12.0", root.path(), true).is_err());
        assert_eq!(std::fs::read_link(sdk.join("current")).unwrap(), std::path::Path::new("missing"));
    }

}
