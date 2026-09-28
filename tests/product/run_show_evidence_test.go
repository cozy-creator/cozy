package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A bundle written before execution records existed still shows its real stages and
// steps (run 1183, MiniMax H3 on four H100s); nothing is invented for what it lacks.
func TestRunShowReadsAPreRecordBundleTolerantly(t *testing.T) {
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "h3-run-1183-attribution.json"))
	must(t, err)
	o := hostOwner(t, "run-show-legacy")
	id := succeededWithTriage(t, o, bundle)
	defer publicationControlAPI(t, o)()
	code, human := runCozy(t, o.root, "run", "show", id)
	t.Logf("cozy run show:\n%s", human)
	for _, want := range []string{"denoise", "decode_video", "51.1s",
		"steps denoise: 8 in 51.1s; first 11.6s, then mean 5.7s (min 5.6s, max 11.6s)"} {
		if code != 0 || !strings.Contains(human, want) {
			t.Fatalf("run show [%d] lacks %q:\n%s", code, want, human)
		}
	}
	if strings.Contains(human, "ranks") {
		t.Fatalf("a bundle with no execution record showed ranks:\n%s", human)
	}
}

// Each rank's attention kernels are readable without a shell on the pod: the one that
// served, and for every other kernel of its chains why not (still compiling with its
// progress, unsupported on this card) and what compiling it cost. The bundle is a
// constructed two-rank Blackwell run in the shape Runtime's execution evidence carries.
func TestRunShowPrintsEachRanksAttentionKernels(t *testing.T) {
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "kernel-ranks.json"))
	must(t, err)
	o := hostOwner(t, "run-show-kernels")
	id := succeededWithTriage(t, o, bundle)
	defer publicationControlAPI(t, o)()
	code, human := runCozy(t, o.root, "run", "show", id)
	t.Logf("cozy run show:\n%s", human)
	at := strings.Index(human, "attention kernels")
	if code != 0 || at < 0 || !strings.Contains(human, "sm_100") {
		t.Fatalf("run show [%d] lacks the ranks' arch or kernels:\n%s", code, human)
	}
	golden, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "kernel-ranks.golden"))
	must(t, err)
	if got := human[at:]; got != string(golden) {
		t.Fatalf("attention kernels differ from the golden:\n%s\nwant:\n%s", got, golden)
	}

	code, out := runCozy(t, o.root, "run", "show", id, "--json")
	var report struct {
		Ranks []struct {
			Arch      string `json:"arch"`
			Attention struct {
				Kernels []map[string]any `json:"kernels"`
			} `json:"attention"`
		} `json:"ranks"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &report) != nil || len(report.Ranks) != 2 {
		t.Fatalf("run show --json [%d]:\n%s", code, out)
	}
	for _, rank := range report.Ranks {
		kernels := rank.Attention.Kernels
		if rank.Arch != "sm_100" || len(kernels) != 4 || kernels[1]["served"] != true ||
			kernels[3]["state"] != "compiling" || kernels[3]["progress"] != 0.42 {
			t.Fatalf("run show --json lost a rank's kernel evidence: %+v", rank)
		}
	}
}

// A GPU call's kernels are its own: a long_form run's H3 segments are child calls, whose ranks
// reach Creator only in their GPU releases, never in the root's triage. Run 1525 served SDPA
// there with nothing on screen saying why. Each call prints its kernels, and the warm's phase
// prints what it started compiling.
func TestRunShowPrintsEachGPUCallsKernelsAndTheWarmsCompiles(t *testing.T) {
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "h3-run-1183-attribution.json"))
	must(t, err)
	o := hostOwner(t, "run-show-call-kernels")
	id := succeededWithTriage(t, o, bundle)
	defer publicationControlAPI(t, o)()
	warm := "rank 0: sol-attn compiling, sageattention compiling (3%), flash-attn3 unsupported, sdpa ready"
	fatal(t, o.store.AppendEvent(id, "request.log", 1, map[string]any{"name": "Starting kernel compiles",
		"value": "info", "fields": map[string]any{"phase": "Starting kernel compiles", "completed": true,
			"started_unix_ms": 1790592484035, "elapsed_ms": 969.9, "detail": warm}}))
	call := "call-748e50c5d0661d690230326fbad94dbeb43dd235#1"
	fatal(t, o.store.AppendEvent(id, "machine.gpu.grant", 1, map[string]any{"key": call, "ordinals": []any{0}}))
	kernels := []any{
		map[string]any{"kernel": "sageattention", "state": "compiling", "progress": 0.61, "served": false},
		map[string]any{"kernel": "sdpa", "state": "ready", "served": false},
		map[string]any{"kernel": "sol-attn", "state": "ready", "served": true, "compile_ms": 11586.9},
	}
	fatal(t, o.store.AppendEvent(id, "machine.gpu.release", 1, map[string]any{"key": call,
		"ordinals": []any{0}, "cause": "exited", "ranks": []any{map[string]any{"rank": 0,
			"ordinal": 0, "arch": "sm_120", "attention": map[string]any{"observed": "sol-attn",
				"kernels": kernels}}}}))
	code, human := runCozy(t, o.root, "run", "show", id)
	t.Logf("cozy run show:\n%s", human)
	at := strings.Index(human, "attention kernels, GPU "+call)
	if code != 0 || at < 0 || !strings.Contains(human, warm) {
		t.Fatalf("run show [%d] lacks the call's kernels or the warm's compiles:\n%s", code, human)
	}
	for _, want := range []string{"sol-attn       served", "11.6s", "sageattention  compiling (61%)"} {
		if !strings.Contains(human[at:], want) {
			t.Fatalf("the call's kernels lack %q:\n%s", want, human[at:])
		}
	}
}

// succeededWithTriage records one settled attempt that kept `bundle`, the way the
// orchestrator's terminal transaction does.
func succeededWithTriage(t *testing.T, o *owner, bundle []byte) string {
	t.Helper()
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-evidence",
		Package: "paul/minimax-h3", WorkerID: "local", Devices: []string{"0", "1", "2", "3"}}))
	id := "req-evidence"
	if _, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: "idem-" + id,
		BodyDigest: "sha256:" + sixtyFour("9"), Package: "paul/minimax-h3", Entrypoint: "ref2va_turbo",
		Payload: []byte("{}"), Outputs: "video"}); problem != nil {
		t.Fatal(problem)
	}
	session, digest := "session-evidence", "sha256:"+sixtyFour("a")
	attempt, problem := o.store.Dispatch(records.Attempt{RequestID: id, SessionID: session,
		InstanceID: "ins-evidence", InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, o.store.OfferDispatch(id, attempt, session))
	fatal(t, o.store.Accepted(id, attempt, session))
	if _, problem := o.store.AcceptTerminal(records.Terminal{RequestID: id, Attempt: attempt,
		SessionID: session, InvocationDigest: digest, TerminalID: "out-evidence",
		TerminalDigest: "sha256:" + sixtyFour("f"), Status: "SUCCEEDED", TriageSubject: "trb-evidence",
		TriageDigest: "sha256:" + sixtyFour("b"), TriageLength: int64(len(bundle)),
		TriageBundle: bundle, EventType: "request.completed", EventPayload: map[string]any{},
		RequestState: "succeeded"}); problem != nil {
		t.Fatal(problem)
	}
	return id
}
