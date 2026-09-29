package producttest

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestSubmitSchemaValidation is cl-105/cl-106 as behaviour. The defect: `cozy run
// paul/sdxl/generate` with no prompt queued run 98 instead of refusing — the installed
// PackageInterface knew the whole request schema and submit never consulted it usefully for the
// person typing. Now the daemon-submit seam refuses a payload the PackageInterface refutes in
// ONE typed `request_payload_invalid` naming every offending field, with the callable's
// usage line as the remedy, BEFORE a request row exists or an idempotency key is burned —
// and `--describe` renders the same contract from the same PackageInterface. The worker's own
// author-surface check still stands behind it for a payload injected past this seam.
func TestSubmitSchemaValidation(t *testing.T) {
	root, err := os.MkdirTemp(os.TempDir(), "cozy-schema-")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})
	project := weightlessProject(t)
	code, out := runCozy(t, root, "package", "install", project, "--editable")
	if code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	video := localWeightlessRef + "/video_transport"
	usage := "cozy run " + video + " prompt=<str> --asset first_frame=<file>"

	// cl-106: --describe renders the callable's contract from the installed PackageInterface —
	// the exact facts submit validates against — without dialing anything.
	code, out = runCozy(t, root, "run", video, "--describe")
	for _, expected := range []string{video, "prompt: str", "first_frame: image asset", "output:", "usage: " + usage} {
		if code != 0 || !strings.Contains(out, expected) {
			t.Fatalf("--describe omitted %q [exit %d]\n%s", expected, code, out)
		}
	}
	code, out = runCozy(t, root, "run", localWeightlessRef+"/tile_job", "--describe")
	// A Runtime that publishes each job field's own default (cozy-runtime#778) shows it.
	size := "size: int (>=8, <=256)"
	var surface struct {
		Jobs []struct {
			Name    string `json:"name"`
			Request struct {
				Fields []struct {
					Name    string          `json:"name"`
					Default json.RawMessage `json:"default"`
				} `json:"fields"`
			} `json:"request"`
		} `json:"jobs"`
	}
	raw, err := os.ReadFile(filepath.Join(project, "metadata", "package-interface.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &surface))
	for _, job := range surface.Jobs {
		for _, field := range job.Request.Fields {
			if job.Name == "tile_job" && field.Name == "size" && len(field.Default) > 0 {
				size += " (default = " + string(field.Default) + ")"
			}
		}
	}
	for _, expected := range []string{"(job)", size} {
		if code != 0 || !strings.Contains(out, expected) {
			t.Fatalf("a job's --describe omitted %q [exit %d]\n%s", expected, code, out)
		}
	}
	// --json is the raw request struct, verbatim from the PackageInterface: field order is the
	// author's, and a required field carries NO wire member rather than a synthesized one.
	code, out = runCozy(t, root, "--json", "run", video, "--describe")
	var request struct {
		Fields []struct {
			Name string `json:"name"`
		} `json:"fields"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &request) != nil || len(request.Fields) != 2 ||
		request.Fields[0].Name != "prompt" || request.Fields[1].Name != "first_frame" ||
		strings.Contains(out, `"wire"`) {
		t.Fatalf("--describe --json is not the raw request struct [exit %d]\n%s", code, out)
	}

	// cl-105: the promptless submit refuses AT SUBMIT — one typed refusal naming BOTH
	// omitted required fields, remedy the usage line — and the human typing gets it too.
	code, out = runCozy(t, root, "--json", "run", video)
	refused := refusalOf(t, out)
	if code != 1 || refused.Code != "request_payload_invalid" ||
		!strings.Contains(refused.Message, "prompt: str") ||
		!strings.Contains(refused.Message, "first_frame: image asset") ||
		refused.Remedy != usage {
		t.Fatalf("promptless submit did not refuse typed with the usage line [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", video); code != 1 ||
		!strings.Contains(out, "provide required arguments: [") || !strings.Contains(out, "Arguments:") || !strings.Contains(out, usage) {
		t.Fatalf("the human refusal lost its fields or usage line [exit %d]\n%s", code, out)
	}

	// Field NAMES are case-insensitive at the CLI composition seam (Paul, 2026-09-02):
	// a typed key folds onto the PackageInterface's spelling and the wire carries ONLY the
	// canonical name — PROMPT= composes exactly what prompt= composes, so the daemon's
	// strict validator misses first_frame alone.
	for _, spelling := range []string{"PROMPT=fold", "Prompt=fold"} {
		code, out = runCozy(t, root, "--json", "run", video, spelling)
		folded := refusalOf(t, out)
		if code != 1 || folded.Code != "request_payload_invalid" ||
			!strings.Contains(folded.Message, "provide required arguments: [first_frame: image asset") ||
			strings.Contains(folded.Message, "prompt:") {
			t.Fatalf("%s did not fold onto prompt [exit %d]\n%s", spelling, code, out)
		}
	}
	// The same fold reaches --asset field paths: PROMPT folds to prompt, and the walk's
	// canonical refusal names the PackageInterface's spelling.
	code, out = runCozy(t, root, "run", video, "prompt=x", "--asset", "PROMPT=missing.png")
	if code != 1 || !strings.Contains(out, "video_transport.prompt is not an asset field") {
		t.Fatalf("--asset PROMPT did not fold onto prompt [exit %d]\n%s", code, out)
	}
	// A genuinely unknown name is never guessed at: it is dropped with a warning, and the
	// field it may have meant is still required.
	code, out = runCozy(t, root, "run", video, "prompts=x")
	if code != 1 || !strings.Contains(out, "warning: ignored unknown field prompts — not in "+video+"'s interface") ||
		!strings.Contains(out, "provide required arguments: [prompt: str") {
		t.Fatalf("an unknown name did not warn beside the required fields [exit %d]\n%s", code, out)
	}

	// The same law for a WIRE-DRIVEN client: no CLI composed this payload, and the daemon
	// still refuses a mistyped field by name before anything is recorded. An undeclared
	// field beside it is not one of the problems.
	daemon := attachDaemon(t, root)
	wire := daemon.call(t, "POST", "/v1/requests", map[string]any{
		"package": localWeightlessRef, "function": "tile",
		"input": map[string]any{"bogus": 1, "size": "big"},
	}, "Idempotency-Key", "schema-wire-1")
	if wire.Status != http.StatusBadRequest || wire.code() != "request_payload_invalid" ||
		!strings.Contains(string(wire.Body), `size does not match declared int`) || strings.Contains(string(wire.Body), "bogus") {
		t.Fatalf("a wire-driven mistyped field was not refused typed: %s", wire.brief())
	}

	// NOTHING was recorded by any refusal: zero request rows, and the refused submissions
	// burned no idempotency key — the same key now queues a valid payload.
	if listed := daemon.call(t, "GET", "/v1/requests?limit=10", nil); listed.Status != http.StatusOK ||
		!strings.Contains(string(listed.Body), `"count":0`) {
		t.Fatalf("a refused submission left a request row: %s", listed.brief())
	}
	accepted := daemon.call(t, "POST", "/v1/requests", map[string]any{
		"package": localWeightlessRef, "function": "tile",
		"input": map[string]any{"size": 24, "seed": 7},
	}, "Idempotency-Key", "schema-wire-1")
	if accepted.Status != http.StatusAccepted || strings.Contains(string(accepted.Body), `"idempotent_replay":true`) {
		t.Fatalf("the refused key was burned; a valid payload under it did not queue: %s", accepted.brief())
	}
	var handle struct {
		RequestID string `json:"request_id"`
	}
	must(t, json.Unmarshal(accepted.Body, &handle))
	waitRequestStatus(t, daemon, handle.RequestID, "completed")

	// An OPTIONAL-field omission still queues and runs: every tile field has a default.
	if code, out := runCozy(t, root, "run", localWeightlessRef+"/tile", "--await"); code != 0 ||
		!strings.Contains(out, "pixels:") {
		t.Fatalf("optional-field omission no longer queues [exit %d]\n%s", code, out)
	}

	// THE RED ARM: worker-side enforcement stands untouched behind this seam. A payload
	// injected PAST the client API — straight into the orchestrator, as no product client
	// can — still dies at execution on the runtime's own author-surface check.
	if code, out := runCozy(t, root, "down", "--all"); code != 0 {
		t.Fatalf("down before the red arm [exit %d]\n%s", code, out)
	}
	o := ownerAtRoot(t, root)
	// The prior successful request holds the binding returned by its worker. An
	// install now carries static metadata until preparation, so it has no plan to
	// inject here. Reuse the real retained binding while bypassing only validation.
	prepared, problem := o.store.RequestRow(handle.RequestID)
	fatal(t, problem)
	if prepared.PlanID == "" || prepared.LocalInstallationID == "" {
		t.Fatal("the completed request retained no worker-prepared local binding")
	}
	requestID, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "schema-past-client", Package: localWeightlessRef, Entrypoint: "tile",
		PlanID: prepared.PlanID, Payload: []byte(`{"size":"big"}`),
		Outputs:   strings.FieldsFunc(prepared.Outputs, func(r rune) bool { return r == ',' }),
		InstallID: prepared.InstallID, Release: prepared.Release,
		LocalInstallationID: prepared.LocalInstallationID,
	})
	fatal(t, problem)
	deadline := time.Now().Add(180 * time.Second)
	for {
		row, problem := o.store.RequestRow(requestID)
		fatal(t, problem)
		if row.State == "failed" || row.State == "refused" {
			attempts, problem := o.store.Attempts(requestID)
			fatal(t, problem)
			last := attempts[len(attempts)-1]
			if last.TerminalCause != "INVALID_REQUEST" ||
				!strings.Contains(last.SafeMessage, "invalid_request") {
				t.Fatalf("the worker-side check answered %q (%s), not INVALID_REQUEST",
					last.TerminalCause, last.SafeMessage)
			}
			break
		}
		if row.State == "succeeded" {
			t.Fatal("a payload injected past the client EXECUTED")
		}
		if time.Now().After(deadline) {
			t.Fatalf("no worker-side verdict; request stayed %q\n%s", row.State, productWorkerLogs(root))
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// attachDaemon joins the hidden daemon a `cozy run` already started: the suite stays a
// CLIENT, reading the same address and credential every product client reads.
func attachDaemon(t *testing.T, root string) *daemonProcess {
	t.Helper()
	s := &daemonProcess{root: root}
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		lock, lockErr := os.ReadFile(filepath.Join(root, "daemon.lock"))
		if lockErr == nil {
			for _, line := range strings.Split(string(lock), "\n") {
				if value, ok := strings.CutPrefix(line, "addr="); ok {
					s.addr = strings.TrimSpace(value)
				}
				if value, ok := strings.CutPrefix(line, "token="); ok {
					s.token = strings.TrimSpace(value)
				}
			}
			if s.addr != "" && s.token != "" {
				return s
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("no hidden daemon published an address under %s", root)
	return nil
}

func waitRequestStatus(t *testing.T, daemon *daemonProcess, requestID, wanted string) {
	t.Helper()
	deadline := time.Now().Add(180 * time.Second)
	status := ""
	for time.Now().Before(deadline) {
		var life struct {
			Status string `json:"status"`
		}
		answer := daemon.call(t, "GET", "/v1/requests/"+requestID, nil)
		must(t, json.Unmarshal(answer.Body, &life))
		status = life.Status
		if status == wanted {
			return
		}
		if status == "failed" || status == "canceled" {
			skipWithoutMachine(t, 1, string(answer.Body))
			t.Fatalf("request %s settled %q, wanted %q: %s", requestID, status, wanted, answer.brief())
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("request %s stayed %q, wanted %q\n%s", requestID, status, wanted, productWorkerLogs(daemon.root))
}

// ownerAtRoot is hostOwner on an EXISTING root: the real orchestrator over the records
// and installs a product flow already created, wired to the real package resolver so a
// dispatch launches the install's own runtime worker.
func ownerAtRoot(t *testing.T, root string) *owner {
	t.Helper()
	must(t, os.Setenv("COZY_HOME", root))
	cfg, e := config.Load()
	fatal(t, e)
	cfg.Home = root
	l, e := home.Open(cfg.Home)
	fatal(t, e)
	st, e := records.Open(l.DB)
	fatal(t, e)
	log, err := os.Create(filepath.Join(root, "owner.log"))
	must(t, err)
	options := orchestrator.Options{
		Cfg: cfg, Layout: l, Store: st, Log: log,
	}
	c, e := orchestrator.Open(options)
	fatal(t, e)
	go func() { _ = c.Serve() }()
	o := &owner{root: root, cfg: cfg, l: l, store: st, c: c}
	o.closer = func() {
		c.Close(20 * time.Second)
		st.Close()
		log.Close()
	}
	t.Cleanup(o.close)
	return o
}
