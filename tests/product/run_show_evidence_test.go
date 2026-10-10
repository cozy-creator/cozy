package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
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
	if strings.Contains(human, "GPUs (") {
		t.Fatalf("a bundle with no execution record showed GPUs:\n%s", human)
	}
}

// Each GPU's attention kernels are readable without a shell on the pod: the one that
// served, and for every other kernel of its chains why not (still compiling with its
// progress, unsupported on this card) and what compiling it cost. Each bundle is a
// constructed Blackwell run on GPUs 2 and 3: the current Runtime names them in `gpus`
// (sealed by UUID, its `ranks` rows hold no number), and an older one only as the `ordinal`
// of its `ranks` rows. Both read as the GPUs nvidia-smi shows, never as ranks.
func TestRunShowPrintsEachGPUsAttentionKernels(t *testing.T) {
	golden, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "kernel-gpus.golden"))
	must(t, err)
	for _, fixture := range []string{"kernel-gpus.json", "kernel-ranks.json"} {
		bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", fixture))
		must(t, err)
		o := hostOwner(t, "run-show-"+strings.TrimSuffix(fixture, ".json"))
		id := succeededWithTriage(t, o, bundle)
		stop := publicationControlAPI(t, o)
		code, human := runCozy(t, o.root, "run", "show", id)
		t.Logf("cozy run show (%s):\n%s", fixture, human)
		at := strings.Index(human, "attention kernels")
		table := regexp.MustCompile(`(?m)^GPUs \(2\)\nGPU +ARCH +UUID +PID +START +TIME +ATTENTION\n2 +sm_100 +GPU-6f1c2a9e\S* +41021 .*\n3 +sm_100 +GPU-0a7e5d13\S* +41022 `)
		if code != 0 || at < 0 || !table.MatchString(human) || strings.Contains(human, "rank") {
			t.Fatalf("run show [%d] of %s lacks GPUs 2 and 3, or says rank:\n%s", code, fixture, human)
		}
		if got := human[at:]; got != string(golden) {
			t.Fatalf("attention kernels of %s differ from the golden:\n%s\nwant:\n%s", fixture, got, golden)
		}

		code, out := runCozy(t, o.root, "run", "show", id, "--json")
		stop()
		var report struct {
			GPUs []struct {
				GPU       int    `json:"gpu"`
				Arch      string `json:"arch"`
				Attention struct {
					Kernels []map[string]any `json:"kernels"`
				} `json:"attention"`
			} `json:"gpus"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &report) != nil || len(report.GPUs) != 2 {
			t.Fatalf("run show --json [%d] of %s:\n%s", code, fixture, out)
		}
		for index, gpu := range report.GPUs {
			kernels := gpu.Attention.Kernels
			if gpu.GPU != 2+index || gpu.Arch != "sm_100" || len(kernels) != 4 || kernels[1]["served"] != true ||
				kernels[3]["state"] != "compiling" || kernels[3]["progress"] != 0.42 {
				t.Fatalf("run show --json lost a GPU's kernel evidence from %s: %+v", fixture, gpu)
			}
		}
	}
}

// A kernel that finishes compiling mid-run serves the run's later steps, and run show names
// which steps served which kernels, for the run and its call, while --json keeps the record.
// Run 5168's recorded measurements: a fresh RTX 5090 took Sage3 FP4 from denoise step 3.
func TestRunShowNamesWhichStepsServedWhichKernels(t *testing.T) {
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "h3-run-5168-segments.json"))
	must(t, err)
	o := hostOwner(t, "run-show-segments")
	id := succeededWithTriage(t, o, bundle)
	fatal(t, o.store.AppendEvent(id, "run.in_progress", 1, map[string]any{"started_unix_ms": 1791588625585}))
	defer publicationControlAPI(t, o)()
	want := "\nGPU 0 attention by step: denoise steps 0–2 fl2va_dit=sol-attn, dense sageattention · 3–7 " +
		"fl2va_dit=kitchen-sol-producer-fp4-lowmem-shared-qkv, dense sageattention3-fp4-global-lowmem; sparse-prefix=kitchen-int8\n\nattention kernels\n"
	for _, args := range [][]string{{"run", "show", id}, {"run", "show", id, "--call", "0"}} {
		code, human := runCozy(t, o.root, args...)
		if code != 0 || !strings.Contains(human, want) {
			t.Fatalf("%v [%d] does not say which steps served which kernels:\n%s", args, code, human)
		}
	}
	code, out := runCozy(t, o.root, "run", "show", id, "--json")
	var report struct {
		GPUs []struct {
			Attention struct {
				Segments []struct {
					Stage string `json:"stage"`
					Step  int    `json:"step"`
				} `json:"segments"`
			} `json:"attention"`
		} `json:"gpus"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &report) != nil || len(report.GPUs) != 1 ||
		fmt.Sprint(report.GPUs[0].Attention.Segments) != "[{ 0} {denoise 3}]" {
		t.Fatalf("run show --json [%d] lost the attention segments: %+v\n%s", code, report, out)
	}
}

// A GPU call's kernels are its own: a long_form run's H3 segments are child calls, whose GPUs
// reach Creator only in their GPU releases, never in the root's triage. Run 1525 served SDPA
// there with nothing on screen saying why. Each call's row names the kernel that served it,
// the call prints all its kernels, and the warm's phase prints what it started compiling.
func TestRunShowPrintsEachGPUCallsKernelsAndTheWarmsCompiles(t *testing.T) {
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "h3-run-1183-attribution.json"))
	must(t, err)
	o := hostOwner(t, "run-show-call-kernels")
	id := succeededWithTriage(t, o, bundle)
	defer publicationControlAPI(t, o)()
	warm := "GPU 0: sol-attn compiling, sageattention compiling (3%), flash-attn3 unsupported, sdpa ready"
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
			"ordinal": 0, "pid": 4100, "arch": "sm_120", "attention": map[string]any{"observed": "sol-attn",
				"kernels": kernels}}}}))
	code, human := runCozy(t, o.root, "run", "show", id)
	t.Logf("cozy run show:\n%s", human)
	row := regexp.MustCompile(`(?m)^1 +call-748e50c5… .* sol-attn$`)
	if code != 0 || !row.MatchString(human) || !strings.Contains(human, warm) {
		t.Fatalf("run show [%d] lacks the call's served kernel or the warm's compiles:\n%s", code, human)
	}
	code, human = runCozy(t, o.root, "run", "show", id, "--call", "call-748e50c5")
	t.Logf("cozy run show --call call-748e50c5:\n%s", human)
	at := strings.Index(human, "attention kernels")
	if code != 0 || at < 0 {
		t.Fatalf("run show --call [%d] lacks the call's kernels:\n%s", code, human)
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
		TriageBundle: bundle, EventType: "run.completed", EventPayload: map[string]any{},
		RequestState: "succeeded"}); problem != nil {
		t.Fatal(problem)
	}
	return id
}
