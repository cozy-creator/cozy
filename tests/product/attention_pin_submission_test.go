package producttest

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// Real CLI -> HTTP daemon -> SQLite, without a GPU. A parser-only test missed two
// copies dropping the kernel pin before a paid H3 request reached its worker.
func TestAttentionPinSurvivesSubmissionAndConflictingReplayRefuses(t *testing.T) {
	root := t.TempDir()
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t), "--editable"); code != 0 {
		t.Fatalf("install %d: %s", code, out)
	}
	args := []string{"--json", "run", localWeightlessRef + "/echo", "why=attention-pin", "--attention-kernel=flash-attn3-fp8", "--idempotency-key=attention-pin"}
	// This fixture needs the real admission path, not a GPU executor.
	_, first := runCozy(t, root, args...)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey("attention-pin")
	fatal(t, problem)
	if request == nil || request.AttentionKernel != "flash-attn3-fp8" {
		t.Fatalf("durable request lost pin: %+v; CLI %s", request, first)
	}
	t.Logf("recorded release=%q install=%q local_revision=%q", request.Release, request.InstallID, request.LocalInstallationID)
	// Same request is idempotent; changing only the kernel must never replay it.
	_, replay := runCozy(t, root, args...)
	if strings.Contains(replay, "different body") {
		t.Fatalf("same pin conflicts: %s", replay)
	}
	// The same pin given by flag and by payload term is one pin, not a conflict.
	twice := append(append([]string(nil), args...), "kernel.attention=flash-attn3-fp8")
	if _, out := runCozy(t, root, twice...); strings.Contains(out, "pinned") || strings.Contains(out, "different body") {
		t.Fatalf("an identical repeated pin was refused: %s", out)
	}
	args[4] = "--attention-kernel=flash-attn3"
	if code, out := runCozy(t, root, args...); code == 0 || !strings.Contains(out, "different body") {
		t.Fatalf("changed pin replayed: %s", out)
	}
	again, problem := store.RequestByIdempotencyKey("attention-pin")
	fatal(t, problem)
	if again.ID != request.ID || again.AttentionKernel != request.AttentionKernel {
		t.Fatalf("conflict mutated original: %+v", again)
	}
}

func TestScopedAttentionPinSurvivesCLIAndLegacySubmission(t *testing.T) {
	root := t.TempDir()
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t), "--editable"); code != 0 {
		t.Fatalf("install %d: %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, pin := range []string{"fl2va_dit=kitchen-int8", "model/fl2va_dit=flashinfer-bf16-fp8"} {
		key := "scoped-" + pin
		args := []string{"--json", "run", localWeightlessRef + "/echo", "why=scoped-attention", "--attention-kernel=" + pin, "--idempotency-key=" + key}
		_, first := runCozy(t, root, args...)
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if request == nil || request.AttentionKernel != pin {
			t.Fatalf("durable scoped pin lost: %+v; CLI %s", request, first)
		}
		args[4] = "kernel.attention=" + pin
		if _, out := runCozy(t, root, args...); strings.Contains(out, "different body") {
			t.Fatalf("equivalent legacy spelling changed identity: %s", out)
		}
		args[4] = "--attention-kernel=other_component=kitchen-int8"
		if code, out := runCozy(t, root, args...); code == 0 || !strings.Contains(out, "different body") {
			t.Fatalf("changing only scope replayed the request: %s", out)
		}
	}
}

func TestAttentionOverrideMalformedHTTPRefusesBeforeAdmission(t *testing.T) {
	root := t.TempDir()
	svc := startDaemonProcess(t, root)
	for _, pin := range []string{"=kitchen-int8", "dit=", "dit=a=b", "dit =a"} {
		reply := svc.call(t, "POST", "/v1/requests", map[string]any{
			"package": "local/not-installed", "function": "generate", "input": map[string]any{}, "attention_kernel": pin,
		}, "Idempotency-Key", "malformed-"+pin)
		if reply.Status != http.StatusBadRequest || reply.code() != "attention_override_invalid" {
			t.Fatalf("malformed override reached package lookup/admission: %s", reply.brief())
		}
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	if request, problem := store.RequestByIdempotencyKey("malformed-dit=a=b"); problem != nil || request != nil {
		t.Fatalf("malformed override was durably admitted: request=%+v problem=%v", request, problem)
	}
}

func TestJobAttentionOverrideCannotBeSilentlyIgnored(t *testing.T) {
	root := t.TempDir()
	iface := []byte(`{"application":"q:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"models":[],"name":"prepare","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)
	parsed, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	if len(parsed.Raw) == 0 {
		t.Fatal("fixture lost its canonical interface")
	}
	dir := filepath.Join(root, "installs", "attention-job")
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(dir)), 0700))
	must(t, os.WriteFile(launch.PackageInterfacePath(dir), iface, 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	_, problem = store.Activate(records.PackageInstall{ID: "attention-job", Package: "proof/attention-job", Major: 1, Version: "1.0.0", SourceKind: "tensorhub", Dir: dir, Platform: "linux-x86"})
	fatal(t, problem)
	for _, term := range []string{"--attention-kernel=dit=kitchen-int8", "kernel.attention=dit=kitchen-int8"} {
		if code, out := runCozy(t, root, "run", "proof/attention-job/prepare", term, "--idempotency-key=job-pin"); code == 0 || !strings.Contains(out, "applies only to serving callables") {
			t.Fatalf("job override was not refused: %s", out)
		}
	}
	if request, problem := store.RequestByIdempotencyKey("job-pin"); problem != nil || request != nil {
		t.Fatalf("unsupported job pin was submitted: request=%+v problem=%v", request, problem)
	}
}

// A version bump in editable metadata is a new revision of the same package: the next
// run relocks and refreshes onto it instead of refusing the stale lock or a manual reinstall.
func TestEditableVersionBumpRefreshesTheInstall(t *testing.T) {
	root := t.TempDir()
	project := weightlessProject(t)
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("install %d: %s", code, out)
	}
	pyproject := filepath.Join(project, "pyproject.toml")
	metadata, err := os.ReadFile(pyproject)
	must(t, err)
	bumped := strings.Replace(string(metadata), `version = "1.0.0"`, `version = "1.0.1"`, 1)
	if bumped == string(metadata) {
		t.Fatal("editable fixture declares no 1.0.0 version")
	}
	must(t, os.WriteFile(pyproject, []byte(bumped), 0o644))
	_, out := runCozy(t, root, "--json", "run", localWeightlessRef+"/echo", "why=version-bump", "--idempotency-key=version-bump")
	if strings.Contains(out, "editable_refresh_failed") || strings.Contains(out, "editable_identity_changed") {
		t.Fatalf("editable version bump refused: %s", out)
	}
	if active := activePackageInstall(t, root); active.Version != "1.0.1" {
		t.Fatalf("editable version bump left install version %s: %s", active.Version, out)
	}
}
