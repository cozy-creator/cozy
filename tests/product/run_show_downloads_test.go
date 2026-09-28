package producttest

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Run 1560's stage table kept one "Downloading model weights" row with no bytes, and its
// 100 GB prefetch none at all. A Runtime that records each model's pull has every download
// in `run show` with its bytes, time and rate, the phase it replaces is gone, and a call
// held for its callee's weights has the wait among its own stages.
func TestRunShowListsEachModelDownloadAndTheWaitItHeld(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := newReleaseMachine()
	machine.changed = make(chan struct{})
	record := func(name string, fields map[string]any) []byte {
		body, err := json.Marshal(map[string]any{"type": "log", "payload": map[string]any{
			"name": name, "value": "info", "at_unix_ms": 1790625718426, "fields": fields}})
		must(t, err)
		return body
	}
	machine.logs = [][]byte{
		record("Downloading model weights", map[string]any{"child_request": "model-preparation.1",
			"phase": "Downloading model weights", "completed": true, "elapsed_ms": 890686.5}),
		record("model fetch", map[string]any{"event": "end", "model": "proof/motion", "entrypoint": "motion_segment_turbo",
			"prefetch": true, "step": "", "started_unix_ms": 1790624830627, "completed": true, "elapsed_ms": 887594.0,
			"bytes": 99894471997, "total_bytes": 99894471997, "moved_bytes": 99894471997, "rate_bytes_per_second": 112545000.0}),
		record("model fetch", map[string]any{"event": "end", "model": "proof/image", "entrypoint": "generate_image",
			"prefetch": false, "step": "Creating reference Background", "started_unix_ms": 1790624848880, "completed": true,
			"elapsed_ms": 273447.0, "bytes": 33118370151, "total_bytes": 33118370151, "moved_bytes": 33118370151,
			"rate_bytes_per_second": 121113000.0}),
		record("model wait", map[string]any{"event": "end", "step": "Segment 1 of 9", "call": "call-segment-1",
			"entrypoint": "motion_segment_turbo", "waited_ms": 697000.0}),
	}
	root := startRentedPod(t, h, &fakePod{}, func(string) machineExecutionPeer { return machine },
		"--extra-index-url https://public.example/v1/index/proof/simple/\n")
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json",
		"--idempotency-key", "downloads"); code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("downloads")
	fatal(t, problem)
	waitFor(t, root, "the run's acceptance", func() bool { return machine.accepted(row.ID) })
	machine.begin(row.ID)
	var report struct {
		Stages []struct {
			Name, Kind, Detail string
			Bytes              int64
			MS                 float64
		}
		Calls []struct {
			Request string
			Stages  []struct{ Name, Kind, Detail string }
		}
	}
	shown := ""
	waitFor(t, root, "the downloads in run show", func() bool {
		_, shown = runCozy(t, root, "run", "show", row.ID, "--json")
		return json.Unmarshal([]byte(shown), &report) == nil && len(report.Calls) > 0
	})
	downloads := map[string]string{}
	for _, stage := range report.Stages {
		if stage.Name == "Downloading model weights" {
			t.Fatalf("the phase a recorded pull replaces is still a row: %s", shown)
		}
		if stage.Kind == "download" {
			downloads[stage.Name] = stage.Detail
			if stage.Bytes <= 0 || stage.MS <= 0 {
				t.Fatalf("a download row lacks its bytes or time: %+v", stage)
			}
		}
	}
	if downloads["download proof/motion"] != "93.0GiB at 107.3MiB/s; prefetch for motion_segment_turbo" ||
		downloads["download proof/image"] != "30.8GiB at 115.5MiB/s; for generate_image, held Creating reference Background" {
		t.Fatalf("run show downloads %v:\n%s", downloads, shown)
	}
	if len(report.Calls) != 1 || report.Calls[0].Request != "call-segment-1" || len(report.Calls[0].Stages) != 1 ||
		report.Calls[0].Stages[0].Kind != "wait" ||
		!strings.Contains(report.Calls[0].Stages[0].Detail, "motion_segment_turbo weights, in Segment 1 of 9") {
		t.Fatalf("the held call's wait is not among its stages: %+v", report.Calls)
	}
	if _, human := runCozy(t, root, "run", "show", row.ID); !strings.Contains(human, "download proof/motion") ||
		!strings.Contains(human, "14m47.6s") {
		t.Fatalf("the human stage table lacks the prefetch's download:\n%s", human)
	}
}
