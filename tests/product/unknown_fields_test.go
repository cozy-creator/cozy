package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestUnknownFieldsWarnAndRun is the owner's ruling (2026-09-28) as behaviour: an undeclared
// request field, top-level or nested, never refuses a run. The CLI drops it before
// submitting, the run succeeds, and one warning names it in human output, in --json and in
// `run show`. Types and required fields stay strict.
func TestUnknownFieldsWarnAndRun(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: this computer's machine runs the call")
	}
	root, err := os.MkdirTemp("", "czu")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	provisionMachine(t, root)
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t)); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	target := localWeightlessRef + "/labels"
	warning := records.Warning{
		Code:    "request_fields_ignored",
		Message: "ignored unknown fields labels[1].foo, style — not in " + target + "'s interface",
		Fields:  []string{"labels[1].foo", "style"},
	}
	nested := `labels:=[{"text":"a"},{"text":"b","foo":1}]`

	code, stdout, stderr := runCozyStreams(t, root, "run", target, nested, "style=noir", "--await")
	t.Logf("cozy run:\n%s%s", stderr, stdout)
	if code != 0 || !strings.Contains(stdout, "a,b") ||
		strings.Count(stderr, "warning: "+warning.Message) != 1 || strings.Count(stderr, "warning") != 1 {
		t.Fatalf("human run did not succeed with one warning [exit %d]\nstdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}

	input := filepath.Join(root, "input.json")
	must(t, os.WriteFile(input, []byte(`{"style":"noir","labels":[{"text":"a"},{"text":"b","foo":1}]}`), 0o600))
	code, out := runCozy(t, root, "--json", "run", target, "--input", input, "--await", "--idempotency-key", "unknown-fields")
	var run struct {
		Result   map[string]any    `json:"result"`
		Warnings []records.Warning `json:"warnings"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &run) != nil || run.Result["text"] != "a,b" ||
		len(run.Warnings) != 1 || !sameWarning(run.Warnings[0], warning) {
		t.Fatalf("--json run lost its result or warning [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	request, problem := store.RequestByIdempotencyKey("unknown-fields")
	store.Close()
	fatal(t, problem)
	if payload := string(request.Payload); strings.Contains(payload, "style") || strings.Contains(payload, "foo") {
		t.Fatalf("the submitted payload kept an undeclared field: %s", payload)
	}

	if code, out := runCozy(t, root, "run", "show", request.ID); code != 0 ||
		strings.Count(out, "warning request_fields_ignored: "+warning.Message) != 1 {
		t.Fatalf("run show did not list the warning once [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "--json", "run", "show", request.ID)
	var shown struct {
		Warnings []records.Warning `json:"warnings"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || len(shown.Warnings) != 1 ||
		!sameWarning(shown.Warnings[0], warning) {
		t.Fatalf("run show --json did not carry the warning [exit %d]\n%s", code, out)
	}

	// A missing required field still refuses, and still names what it would have dropped.
	code, out = runCozy(t, root, "--json", "run", target, "style=noir")
	var refused struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
			Details struct {
				Warnings []records.Warning `json:"warnings"`
			} `json:"details"`
		} `json:"error"`
	}
	if code != 1 || json.Unmarshal([]byte(out), &refused) != nil || refused.Error.Code != "request_payload_invalid" ||
		!strings.Contains(refused.Error.Message, "labels") || len(refused.Error.Details.Warnings) != 1 ||
		refused.Error.Details.Warnings[0].Fields[0] != "style" {
		t.Fatalf("a missing required field did not refuse [exit %d]\n%s", code, out)
	}
	// A wrong type still refuses, at any depth.
	for _, term := range []string{`labels:=[{"text":7}]`, `labels:="a"`} {
		code, out = runCozy(t, root, "--json", "run", target, term, "style=noir")
		if code != 1 || refusalOf(t, out).Code != "request_payload_invalid" {
			t.Fatalf("%s did not refuse [exit %d]\n%s", term, code, out)
		}
	}
}

func sameWarning(got, want records.Warning) bool {
	return got.Code == want.Code && got.Message == want.Message && strings.Join(got.Fields, ",") == strings.Join(want.Fields, ",")
}
