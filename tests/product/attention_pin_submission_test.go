package producttest

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

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

// --attention-kernel is the one spelling: a `kernel.attention=` term is an ordinary
// undeclared payload field, never a pin, and a malformed flag refuses before admission.
func TestScopedAttentionPinSurvivesSubmission(t *testing.T) {
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
		args[4] = "--attention-kernel=other_component=kitchen-int8"
		if code, out := runCozy(t, root, args...); code == 0 || !strings.Contains(out, "different body") {
			t.Fatalf("changing only scope replayed the request: %s", out)
		}
	}
	_, out := runCozy(t, root, "--json", "run", localWeightlessRef+"/echo", "why=term", "kernel.attention=sdpa", "--idempotency-key=term")
	request, problem := store.RequestByIdempotencyKey("term")
	fatal(t, problem)
	if request == nil || request.AttentionKernel != "" {
		t.Fatalf("kernel.attention= still pins: %+v; CLI %s", request, out)
	}
	code, out := runCozy(t, root, "--json", "run", localWeightlessRef+"/echo", "why=bad", "--attention-kernel=dit=a=b", "--idempotency-key=bad")
	if code == 0 || !strings.Contains(out, "attention_override_invalid") {
		t.Fatalf("malformed --attention-kernel was admitted (%d): %s", code, out)
	}
	if request, problem := store.RequestByIdempotencyKey("bad"); problem != nil || request != nil {
		t.Fatalf("malformed --attention-kernel was durably admitted: %+v %v", request, problem)
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
