package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/machines"
)

// The ordinary local install command must use the actual machine transaction,
// preserve the unrelated environment and run packages after update and refusal.
func TestLocalInstallUsesMachineTransaction(t *testing.T) {
	if *machineRuntimeWheel == "" || *machineTensorFSWheel == "" {
		t.Skip("requires the audited Runtime/TensorFS pair")
	}
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel, Pinned: true}
	_, root, layout, _ := parityMachinesOn(t, source)
	prefix := filepath.Join(layout.Machine, "root/opt/cozy/python")
	python := filepath.Join(prefix, "bin/python")
	installCommand(t, "uv", "pip", "install", "--no-config", "--python", python, installWheel(t, "machine_base_framework", "2.14.0+cu130", ""))
	marker := filepath.Join(prefix, "operator-settings")
	must(t, os.WriteFile(marker, []byte("keep"), 0600))
	if code, out := runCozy(t, root, "package", "install", parityProjectOn(t, source), "--editable"); code != 0 {
		t.Fatalf("package install [%d]: %s", code, out)
	}
	run := func() {
		t.Helper()
		if code, out := runCozy(t, root, "run", parityPackage+"/add", "value=41", "--await", "--json"); code != 0 || !strings.Contains(out, `"value":42`) {
			t.Fatalf("package [%d]: %s", code, out)
		}
	}
	run()
	recordPath := filepath.Join(layout.Machine, "agent.json")
	before, err := os.ReadFile(recordPath)
	must(t, err)
	metadataPath := filepath.Join(layout.Machine, "installed.json")
	metadataBefore, err := os.ReadFile(metadataPath)
	must(t, err)
	agentBefore := fileSHA(t, filepath.Join(prefix, "bin/cozy-machine"))
	candidate := localBuild(t, source.RuntimeWheel, "localupdated")
	if *machineUpdateWheel != "" {
		candidate = *machineUpdateWheel
	}
	update := func(wheel string) (int, string) {
		return runCozy(t, root, "machine", "install", "--runtime-wheel", wheel, "--tensorfs-wheel", source.TensorFSWheel, "--json")
	}
	if code, out := update(candidate); code != 0 {
		t.Fatalf("local transaction [%d]: %s", code, out)
	}
	after, err := os.ReadFile(recordPath)
	must(t, err)
	if string(before) != string(after) {
		t.Fatal("local update replaced the stable bootstrap process")
	}
	metadata, err := os.ReadFile(metadataPath)
	must(t, err)
	var installed machines.Installed
	must(t, json.Unmarshal(metadata, &installed))
	agentAfter := fileSHA(t, filepath.Join(prefix, "bin/cozy-machine"))
	if installed.Host.Module != machines.AgentModule || installed.Host.SHA256 != agentAfter || installed.HostPinned {
		t.Fatalf("updated agent provenance cannot authorize subsequent Hub access: %+v", installed.Host)
	}
	if *machineUpdateWheel != "" && agentBefore == agentAfter {
		t.Fatal("distinct-agent qualification fixture did not replace the application")
	}
	version := strings.SplitN(filepath.Base(candidate), "-", 3)[1]
	// Startup updates occur without Creator rewriting its installation record.
	// Show must read the live authority, even with that historical record present.
	must(t, os.WriteFile(metadataPath, metadataBefore, 0600))
	defer func() { must(t, os.WriteFile(metadataPath, metadata, 0600)) }()
	code, shown := runCozy(t, root, "machine", "show", "--json")
	var live struct {
		Runtime   string                  `json:"runtime"`
		Agent     struct{ SHA256 string } `json:"agent"`
		Bootstrap struct{ ABI string }    `json:"bootstrap"`
	}
	if code != 0 || json.Unmarshal([]byte(shown), &live) != nil || live.Runtime != version || live.Agent.SHA256 != agentAfter || live.Bootstrap.ABI != machines.BootstrapCapability {
		t.Fatalf("machine show used historical installation provenance [%d]: %s", code, shown)
	}
	must(t, os.WriteFile(metadataPath, metadata, 0600))
	assertInstallVersion(t, python, hostruntime.Distribution, version)
	assertInstallVersion(t, python, "machine-base-framework", "2.14.0+cu130")
	if body, err := os.ReadFile(marker); err != nil || string(body) != "keep" {
		t.Fatal("local update discarded machine settings")
	}
	run()
	if *machineUpdateWheel != "" {
		if code, out := runCozy(t, root, "machine", "install", "--host", source.Host, "--runtime-wheel", candidate, "--tensorfs-wheel", source.TensorFSWheel, "--json"); code == 0 || !strings.Contains(out, "machine.agent_bundle_required") {
			t.Fatalf("a different explicit agent bypassed the bundled transaction [%d]: %s", code, out)
		}
	}
	observed, err := os.ReadFile(recordPath)
	must(t, err)
	if string(before) != string(observed) {
		t.Fatal("package invocation after update replaced the live agent")
	}
	if code, out := update(brokenBuild(t, candidate)); code == 0 || !strings.Contains(out, "machine.update_failed") {
		t.Fatalf("broken candidate admitted [%d]: %s", code, out)
	}
	assertInstallVersion(t, python, hostruntime.Distribution, version)
	assertInstallVersion(t, python, "machine-base-framework", "2.14.0+cu130")
	run()
}
