package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A job's calls show the GPUs each ran on, as a run does. The machine sends each settled
// child's measurements on its call (here run 5243's: MiniMax H3 on four H100s, once with a
// transport record); the store keeps their execution record and run show prints each call's
// GPUs, attention and transport. A call from an older machine, without them, still lists.
func TestRunShowPrintsEachCallsGPUs(t *testing.T) {
	measured, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "h3-run-5243-measurements.json"))
	must(t, err)
	var doc map[string]json.RawMessage
	var execution map[string]any
	must(t, json.Unmarshal(measured, &doc))
	must(t, json.Unmarshal(doc["execution"], &execution))
	execution["transport"] = map[string]any{"degree": 4, "gpus": []int{0, 1, 2, 3}, "route": "peer-ce",
		"nccl": map[string]any{"NCCL_P2P_LEVEL": "SYS", "NCCL_P2P_USE_CUDA_MEMCPY": 1},
		"pairs": []map[string]any{
			{"a": 0, "b": 1, "peer": true, "direct_gbps": 52.4, "staged_gbps": 11.9, "route": "peer-ce"},
			{"a": 0, "b": 2, "peer": true, "direct_gbps": 28.0, "staged_gbps": 11.3, "route": "peer-ce"},
		},
		"reason": "peer copies beat host staging", "future": "kept"}
	doc["execution"], err = json.Marshal(execution)
	must(t, err)
	transported, err := json.Marshal(doc)
	must(t, err)

	o := hostOwner(t, "run-show-call-gpus")
	id := "job-call-gpus"
	_, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: id, Package: "paul/minimax-h3",
		Entrypoint: "long_form", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("c"),
		MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, o.store.LinkMachineExecution(id, "pr-unreachable"))
	fatal(t, o.store.AcceptRunV1(id, "pr-unreachable", &v1.RunState{Id: id, Number: 1, State: "running", Attempt: 1}))
	stamp := time.Now().UnixMilli()
	for i, call := range []struct {
		label        string
		measurements []byte
	}{{"Segment 1 of 3", transported}, {"Segment 2 of 3", measured}, {"Segment 3 of 3", nil}} {
		fatal(t, o.store.ObserveRunV1(id, &v1.RunEvent{Sequence: uint64(i + 1), AtMs: stamp + int64(i),
			Event: &v1.RunEvent_Call{Call: &v1.Call{Run: fmt.Sprintf("%s/%d", id, i), Index: uint32(i),
				Function: "motion_segment_turbo", Label: call.label, Status: "succeeded",
				CalledAtMs: stamp - 1000 + int64(i), FinishedAtMs: stamp + int64(i), Measurements: call.measurements}}}, nil))
	}
	fatal(t, o.store.RecordRunOutcomeV1(id, records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded"}}))
	defer publicationControlAPI(t, o)()

	transport := "GPUs 0-3 · route peer-ce (NCCL P2P_LEVEL=SYS, P2P_USE_CUDA_MEMCPY=1) · " +
		"GPU 0↔2 28.0 GB/s direct vs 11.3 staged, slowest of 2 · peer copies beat host staging"
	code, human := runCozy(t, o.root, "run", "show", id)
	t.Logf("cozy run show:\n%s", human)
	for _, row := range []string{`1 +Segment 1 of 3 .* 0-3 .* flash-attn3,sdpa,sol-attn\n`,
		`2 +Segment 2 of 3 .* 0-3 .* flash-attn3,sdpa,sol-attn\n`, `3 +Segment 3 of 3 +motion_segment_turbo +succeeded +- `} {
		if code != 0 || !regexp.MustCompile(`(?m)^`+row).MatchString(human) {
			t.Fatalf("run show [%d] lacks the call row %q:\n%s", code, row, human)
		}
	}
	if !strings.Contains(human, "  call 1 transport: "+transport+"\n") || strings.Contains(human, "call 2 transport") {
		t.Fatalf("run show lacks call 1's transport alone:\n%s", human)
	}

	code, one := runCozy(t, o.root, "run", "show", id, "--call", "1")
	t.Logf("cozy run show --call 1:\n%s", one)
	table := regexp.MustCompile(`(?m)^GPUs \(4\)\nGPU +ARCH +UUID +PID +START +TIME +ATTENTION\n0 +sm_90 +GPU-6ff9540a\S* +12442 .*Sol calls: 192 sparse, 416 dense\n1 +sm_90 .* 12447 .*\n2 +sm_90 .* 12449 .*\n3 +sm_90 .* 12451 .*\ntransport: ` + regexp.QuoteMeta(transport) + `\n\nattention kernels\n`)
	if code != 0 || !table.MatchString(one) || !strings.Contains(one, "· GPUs 0-3") || strings.Contains(one, "rank") {
		t.Fatalf("run show --call 1 [%d] lacks its GPU table, attention and transport, or says rank:\n%s", code, one)
	}
	code, full := runCozy(t, o.root, "run", "show", id, "--call", "1", "--full")
	if code != 0 || !strings.Contains(full, "transport: GPUs 0-3 · route peer-ce (NCCL P2P_LEVEL=SYS, P2P_USE_CUDA_MEMCPY=1) · "+
		"GPU 0↔2 28.0 GB/s direct vs 11.3 staged · GPU 0↔1 52.4 GB/s direct vs 11.9 staged · peer copies beat host staging\n") {
		t.Fatalf("run show --call 1 --full [%d] lacks every pair:\n%s", code, full)
	}
	code, older := runCozy(t, o.root, "run", "show", id, "--call", "3")
	if code != 0 || !strings.Contains(older, "call 3 of 3  Segment 3 of 3") || strings.Contains(older, "GPUs") || strings.Contains(older, "transport") {
		t.Fatalf("run show --call 3 [%d] of an older machine's call:\n%s", code, older)
	}

	code, out := runCozy(t, o.root, "run", "show", id, "--json")
	var report struct {
		Calls []struct {
			Label string `json:"label"`
			GPUs  []struct {
				GPU int `json:"gpu"`
				PID int `json:"pid"`
			} `json:"gpus"`
			Transport map[string]any `json:"transport"`
		} `json:"calls"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &report) != nil || len(report.Calls) != 4 {
		t.Fatalf("run show --json [%d]:\n%.3000s", code, out)
	}
	for _, call := range report.Calls[1:3] {
		if len(call.GPUs) != 4 || call.GPUs[3].GPU != 3 || call.GPUs[3].PID != 12451 {
			t.Fatalf("run show --json lost %s's GPUs: %+v", call.Label, call)
		}
	}
	if report.Calls[1].Transport["future"] != "kept" || report.Calls[2].Transport != nil || len(report.Calls[3].GPUs) != 0 {
		t.Fatalf("run show --json calls: %+v", report.Calls)
	}
}
