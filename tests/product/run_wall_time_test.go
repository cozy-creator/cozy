package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
)

// The built CLI reattaches years after the durable terminal, twice. Both run
// kinds must report recorded request time, not time since creation at viewing.
func TestCompletedRunWatchUsesRecordedWallTime(t *testing.T) {
	for _, kind := range []string{"job", "invocation"} {
		for _, stamp := range []struct {
			name, start, end string
			known            bool
		}{
			{"delayed", "2020-01-02T03:04:05.100Z", "2020-01-02T03:05:07.445Z", true},
			{"missing", "2020-01-02T03:04:05.100Z", "", false},
			{"missing-start", "", "2020-01-02T03:05:07.445Z", false},
			{"malformed", "2020-01-02T03:04:05.100Z", "not-a-timestamp", false},
			{"reversed", "2020-01-02T03:05:07.445Z", "2020-01-02T03:04:05.100Z", false},
		} {
			t.Run(kind+"/"+stamp.name, func(t *testing.T) {
				id := "req-recorded-wall"
				if kind == "job" {
					id = "job-recorded-wall"
				}
				life := api.Lifecycle{Number: 1, Kind: kind, RequestID: id, Status: "completed", Package: "proof/wall", Function: "run", CreatedAt: stamp.start, QueuedMS: 1234, ExecutionMS: 60000}
				job := api.JobState{Number: 1, JobID: id, Status: "completed", Package: life.Package, Function: life.Function, CreatedAt: stamp.start, QueuedMS: life.QueuedMS, ExecutionMS: life.ExecutionMS, MachineExecution: &api.MachineExecutionView{Accepted: true, Collected: true}}
				event := api.Envelope{Type: "request.completed", RequestID: id, EventID: 7, At: stamp.end}
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					switch r.URL.Path {
					case "/":
						w.WriteHeader(http.StatusOK)
					case "/v1/requests/1", "/v1/requests/" + id:
						_ = json.NewEncoder(w).Encode(life)
					case "/v1/local/jobs/" + id, "/v1/local/jobs/1":
						_ = json.NewEncoder(w).Encode(job)
					case "/v1/requests/" + id + "/events":
						w.Header().Set("Content-Type", "text/event-stream")
						raw, err := json.Marshal(event)
						must(t, err)
						_, _ = fmt.Fprintf(w, "id: 7\ndata: %s\n\n", raw)
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
				for repeat := 0; repeat < 2; repeat++ {
					code, out, errout := runCozyStreams(t, root, "run", "watch", "1", "--json", "--full")
					var result map[string]any
					if code != 0 || json.Unmarshal([]byte(out), &result) != nil {
						t.Fatalf("watch [%d]: %s %s", code, out, errout)
					}
					wall, present := result["wall_ms"]
					if present != stamp.known || present && wall != float64(62345) {
						t.Fatalf("recorded wall changed on replay %d: %s", repeat, out)
					}
					if result["queued"] != "1.2s" || result["execution"] != "60.0s" {
						t.Fatalf("wall duration changed independent timing facts: %s", out)
					}
				}
			})
		}
	}
}
