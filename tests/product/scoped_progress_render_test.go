package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/output"
)

func TestScopedProgressKeepsChildTimingSeparateFromOverallWork(t *testing.T) {
	p, _ := progressSink(output.Mode{Human: true, Live: true}, false)
	t.Cleanup(p.Done)
	for _, position := range []int{2, 3} {
		p.On(liveEvent("progress", map[string]any{
			"stage": "Shot 2 of 7 / denoise", "position": position, "total": 8,
			"stage_fraction": float64(position) / 8, "overall_fraction": .1 + float64(position)/30,
			"step_ms": 1000,
		}))
	}
	child := liveFrame(p, time.Now())
	p.On(liveEvent("progress", map[string]any{"stage": "Assembling video", "overall_fraction": .95}))
	got := child + "\n" + liveFrame(p, time.Now())
	for _, want := range []string{"Shot 2 of 7 · Generating video", "step 3/8", "1.00s/step avg", " 20%", " 95%"} {
		if !strings.Contains(got, want) {
			t.Fatalf("scoped progress lacks %q: %q", want, got)
		}
	}
	if strings.Contains(got, "20% · ETA") || strings.Contains(got, "95% · ETA") {
		t.Fatalf("child duration became a prediction for later work: %q", got)
	}
}

// This is a built CLI subprocess consuming representative daemon SSE frames.
// It proves presentation and protocol consumption, not live model execution.
func TestRunWatchRendersScopedChildProgress(t *testing.T) {
	frames := []api.Envelope{
		liveEvent("phase", map[string]any{"phase": "materialize_model_defaults"}),
		liveEvent("phase", map[string]any{"phase": "downloading", "moved_bytes": 1024, "total_bytes": 2048}),
		liveEvent("accepted", nil),
		liveEvent("progress", map[string]any{"stage": "prep_model_binding"}),
		liveEvent("progress", map[string]any{"stage": "checkpoint_input_admission"}),
		liveEvent("progress", map[string]any{"stage": "Shot 2 of 7 / denoise", "position": 3, "total": 8,
			"stage_fraction": .375, "overall_fraction": .2, "step_ms": 1000}),
		liveEvent("completed", nil),
	}
	for i := range frames {
		frames[i].RequestID = "req-scoped-progress"
	}
	life := api.Lifecycle{Number: 1, RequestID: "req-scoped-progress", Kind: "invocation", Status: "completed",
		Package: "proof/video", Function: "long_form", CreatedAt: time.Now().UTC().Format(time.RFC3339Nano)}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/":
			w.WriteHeader(http.StatusOK)
		case "/v1/requests/1", "/v1/requests/req-scoped-progress":
			_ = json.NewEncoder(w).Encode(life)
		case "/v1/requests/req-scoped-progress/events":
			w.Header().Set("Content-Type", "text/event-stream")
			for _, frame := range frames {
				encoded, err := json.Marshal(frame)
				must(t, err)
				_, _ = fmt.Fprintf(w, "data: %s\n\n", encoded)
			}
		default:
			t.Errorf("unexpected daemon route: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	held, problem := daemon.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
	fatal(t, problem)
	defer held.Release()
	_, problem = api.Mint(layout)
	fatal(t, problem)
	code, stdout, stderr := runCozyStreams(t, root, "run", "watch", "1")
	if code != 0 {
		t.Fatalf("CLI watch failed [%d]: %s\n%s", code, stdout, stderr)
	}
	for _, want := range []string{"Preparing model files", "downloading", "Waiting for a progress update",
		"Checking model compatibility", "Checking model inputs", "Shot 2 of 7 · Generating video · step 3/8", "20% overall"} {
		if !strings.Contains(stderr, want) {
			t.Fatalf("CLI progress lacks %q: %s", want, stderr)
		}
	}
	for _, leak := range []string{"materialize_model_defaults", "prep_model_binding", "checkpoint_input_admission", "running · waiting"} {
		if strings.Contains(stderr, leak) {
			t.Fatalf("CLI progress exposes %q: %s", leak, stderr)
		}
	}
	code, _, stderr = runCozyStreams(t, root, "run", "watch", "1", "--json")
	if code != 0 || !strings.Contains(stderr, `"stage":"Shot 2 of 7 / denoise"`) ||
		!strings.Contains(stderr, `"overall_fraction":0.2`) {
		t.Fatalf("raw progress changed: exit=%d %s", code, stderr)
	}
}
