package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// The real run5020 timestamps and step timing: authored phase weights are not
// elapsed-time weights. Exercise the persisted Runtime lane through ordinary CLI.
func TestRunListETAKeepsMeasuredStageSeparateFromUnknownFutureWork(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0600))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, _, problem := store.Submit(records.Request{ID: "stage-eta", IdemKey: "stage-eta", Package: "local/proof",
		Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("b"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "pr-unreachable"))
	fatal(t, store.AppendEvent(request.ID, records.RunV1Sent, 0, map[string]any{"machine": "pr-unreachable"}))
	fatal(t, store.AcceptRunV1(request.ID, "pr-unreachable", &v1.RunState{Id: request.ID, Number: 1, State: "running", Attempt: 1}))
	sequence := uint64(1)
	at := time.Date(2026, 10, 9, 1, 39, 43, 224000000, time.UTC)
	report := func(ms int64, progress *v1.Progress) {
		t.Helper()
		stamp := at.Add(time.Duration(ms) * time.Millisecond).UnixMilli()
		if ms < 0 {
			stamp = 0 // Older source omitted its timestamp; receipt time is not a measurement.
		}
		sequence++
		fatal(t, store.ObserveRunV1(request.ID, &v1.RunEvent{Sequence: sequence, AtMs: stamp,
			Event: &v1.RunEvent_Progress{Progress: progress}}, nil))
		sequence++
		fatal(t, store.ObserveRunV1(request.ID, &v1.RunEvent{Sequence: sequence, AtMs: at.Add(time.Duration(ms) * time.Millisecond).UnixMilli(),
			Event: &v1.RunEvent_State{State: &v1.RunState{Id: request.ID, Number: 1, State: "running", Attempt: 1}}}, nil))
	}
	type progressRow struct {
		Stage            string   `json:"progress_stage"`
		RemainingMS      *int64   `json:"remaining_ms"`
		StageRemainingMS *int64   `json:"stage_remaining_ms"`
		OverallFraction  *float64 `json:"overall_fraction"`
	}
	list := func() progressRow {
		t.Helper()
		code, out := runCozy(t, root, "run", "list", "--json")
		var document struct{ Invocations []progressRow }
		if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Invocations) != 1 {
			t.Fatalf("ordinary run list: %d %s", code, out)
		}
		return document.Invocations[0]
	}
	report(0, &v1.Progress{Stage: "prepare", Fraction: 0})
	report(1, &v1.Progress{Stage: "prepare", Fraction: .03})
	report(8743, &v1.Progress{Stage: "condition_text", Fraction: .08})
	report(8744, &v1.Progress{Stage: "denoise", Fraction: .15})
	report(46875, &v1.Progress{Stage: "denoise", Fraction: .173333, Completed: 1, Total: 30, StepMs: 35541.02015681565})
	row := list()
	if row.RemainingMS != nil {
		t.Fatalf("authored progress weights invented whole-run ETA %dms; later decode work is unknown", *row.RemainingMS)
	}
	if row.Stage != "denoise" || row.StageRemainingMS == nil || *row.StageRemainingMS != 1030689 {
		t.Fatalf("first35.541s step with29left lost its stage estimate: %+v", row)
	}
	// A changed measured pace is allowed to change only the current stage estimate.
	report(300000, &v1.Progress{Stage: "denoise", Fraction: .40, Completed: 11, Total: 30, StepMs: 20000})
	report(320000, &v1.Progress{Stage: "denoise", Fraction: .43, Completed: 12, Total: 30, StepMs: 20000})
	if row = list(); row.RemainingMS != nil || row.StageRemainingMS == nil || *row.StageRemainingMS != 360000 {
		t.Fatalf("new step pace did not stay stage-local: %+v", row)
	}
	report(700000, &v1.Progress{Stage: "denoise", Fraction: .85, Completed: 30, Total: 30, StepMs: 20000})
	if row = list(); row.RemainingMS != nil || row.StageRemainingMS == nil || *row.StageRemainingMS != 0 {
		t.Fatalf("finished stage passed for finished job: %+v", row)
	}
	report(700001, &v1.Progress{Stage: "decode_video", Fraction: .85})
	if row = list(); row.Stage != "decode_video" || row.RemainingMS != nil || row.StageRemainingMS != nil {
		t.Fatalf("untimed decode inherited denoise ETA: %+v", row)
	}
	// Decode updates count frames, while step_ms times an entire chunk.
	report(720000, &v1.Progress{Stage: "decode_video", Fraction: .90, Completed: 170, Total: 362, StepMs: 1879.7})
	if row = list(); row.StageRemainingMS != nil {
		t.Fatalf("one multi-frame sample invented a unit rate: %+v", row)
	}
	report(721880, &v1.Progress{Stage: "decode_video", Fraction: .91, Completed: 187, Total: 362, StepMs: 1879.7})
	if row = list(); row.RemainingMS != nil || row.StageRemainingMS == nil || *row.StageRemainingMS != 19352 {
		t.Fatalf("decoded chunk duration was multiplied by remaining frames: %+v", row)
	}
	// An omitted intermediate chunk: the source interval covers both17-frame chunks.
	report(725640, &v1.Progress{Stage: "decode_video", Fraction: .92, Completed: 221, Total: 362, StepMs: 1879.7})
	if row = list(); row.StageRemainingMS == nil || *row.StageRemainingMS != 15592 {
		t.Fatalf("coalesced chunk rate used only the final chunk's duration: %+v", row)
	}
	report(-1, &v1.Progress{Stage: "decode_video", Fraction: .93, Completed: 238, Total: 362, StepMs: 1879.7})
	if row = list(); row.StageRemainingMS != nil {
		t.Fatalf("client receipt time passed for a source measurement: %+v", row)
	}

}
