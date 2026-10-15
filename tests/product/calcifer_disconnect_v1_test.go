package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/calcifer"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A handoff its machine has not confirmed does not hold `cozy down`: Calcifer stops, names it
// in flight, and leaves it exactly as it was, neither canceled nor accepted, for the next
// Calcifer to confirm. A second down is a no-op.
func TestCalciferDownDoesNotCancelAnUnconfirmedHandoff(t *testing.T) {
	root := t.TempDir()
	startCalciferProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, _, problem := store.Submit(records.Request{ID: "job-handoff", IdemKey: "handoff", Package: "local/example", Entrypoint: "main", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "pr-unreachable-machine"))
	if sent, problem := store.MarkRunV1Sent(request.ID); problem != nil || !sent {
		t.Fatalf("the fixture's dispatch was not marked: %v", problem)
	}
	before, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	code, out := runCozy(t, root, "down", "--json")
	if code != 0 || !strings.Contains(out, `"calcifer":"stopped"`) || !strings.Contains(out, request.ID) {
		t.Fatalf("down did not stop and name the unconfirmed handoff [%d]: %s", code, out)
	}
	after, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	current, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	sent, problem := store.RunV1Marked(request.ID, records.RunV1Sent)
	fatal(t, problem)
	if current.State != before.State || after.CancelRequested || after.Abandoned || len(after.Receipt) != 0 || !sent {
		t.Fatalf("down changed the handoff: %+v %+v", current, after)
	}
	code, out = runCozy(t, root, "down", "--json")
	if code != 0 || !strings.Contains(out, `"changed":false`) {
		t.Fatalf("offline down restarted or refused [%d]: %s", code, out)
	}
	if calcifer.Probe(config.Config{Home: root}).Up {
		t.Fatal("offline down restarted Calcifer")
	}
}

type shutdownRaceResolver struct {
	publishedRouteResolver
	entered chan struct{}
	release chan struct{}
}

func (r shutdownRaceResolver) ResolveRemoteRelease(hub, pkg, release, function string, models []orchestrator.ModelRef) (orchestrator.LogicalPackage, *launch.Entrypoint, *exit.Error) {
	if r.entered != nil {
		close(r.entered)
		<-r.release
	}
	return r.publishedRouteResolver.ResolveRemoteRelease(hub, pkg, release, function, models)
}

func TestCalciferDownFencesConcurrentMachineIntake(t *testing.T) {
	for _, beforeCommit := range []bool{true, false} {
		t.Run(fmt.Sprintf("before_commit=%v", beforeCommit), func(t *testing.T) {
			entered, release := make(chan struct{}), make(chan struct{})
			defer func() {
				select {
				case <-release:
				default:
					close(release)
				}
			}()
			var starts atomic.Int32
			o := hostOwner(t, "shutdown-intake", func(options *orchestrator.Options) {
				options.StartMachineExecution = func(records.Request) *exit.Error {
					starts.Add(1)
					if !beforeCommit {
						close(entered)
						<-release
					}
					return nil
				}
			})
			fatal(t, o.store.RecordRental(records.Rental{ID: "pr-shutdown-race", MachineName: "heron", AcceleratorModel: "CPU", AcceleratorCount: 1, State: "ready", HourlyRateUSDMicros: 1, Hub: "http://127.0.0.1:1", Address: "127.0.0.1:1"}))
			const token = "shutdown-intake-fixture"
			resolver := shutdownRaceResolver{}
			if beforeCommit {
				resolver.entered, resolver.release = entered, release
			}
			stopped := make(chan struct{})
			handler, problem := api.New(api.Options{Orchestrator: o.c, Cfg: o.cfg, Creds: api.Credentials{CLI: secret.New(token)}, Addr: "127.0.0.1:11111", Web: http.NotFoundHandler(), Packages: resolver, MachineExecutions: publishedRouteObserver{}, Shutdown: func() { close(stopped) }}).Handler()
			fatal(t, problem)
			call := func(route string, body any) *httptest.ResponseRecorder {
				raw, err := json.Marshal(body)
				must(t, err)
				request := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:11111"+route, bytes.NewReader(raw))
				request.RemoteAddr = "127.0.0.1:12345"
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Authorization", "Bearer "+token)
				request.Header.Set("Idempotency-Key", "shutdown-race")
				response := httptest.NewRecorder()
				handler.ServeHTTP(response, request)
				return response
			}
			submission := map[string]any{"package": "alice/ops", "release": "1.0.0", "function": "main", "input": map[string]any{}, "rental": true, "requested_rental": "pr-shutdown-race"}
			replied := make(chan *httptest.ResponseRecorder, 1)
			go func() { replied <- call("/v1/requests", submission) }()
			if beforeCommit {
				select {
				case <-entered:
				case response := <-replied:
					t.Fatalf("intake ended before resolution boundary: %d %s", response.Code, response.Body.String())
				}
			} else {
				response := <-replied
				if response.Code != http.StatusAccepted {
					t.Fatalf("intake ended before activation boundary: %d %s", response.Code, response.Body.String())
				}
				<-entered
				replied <- response
			}
			down := call("/v1/local/calcifer/down", map[string]bool{})
			if beforeCommit {
				if down.Code != http.StatusAccepted {
					t.Fatalf("pre-commit down %d %s", down.Code, down.Body.String())
				}
				<-stopped
				close(release)
				result := <-replied
				if result.Code != http.StatusServiceUnavailable || !strings.Contains(result.Body.String(), "calcifer_shutting_down") {
					t.Fatalf("late intake crossed shutdown: %d %s", result.Code, result.Body.String())
				}
				row, problem := o.store.RequestByIdempotencyKey("shutdown-race")
				fatal(t, problem)
				if row != nil || starts.Load() != 0 {
					t.Fatal("shutdown admitted hidden work")
				}
			} else {
				// Committed intake is durable: down stops at once and names it in flight.
				result := <-replied
				committed, problem := o.store.RequestByIdempotencyKey("shutdown-race")
				fatal(t, problem)
				if result.Code != http.StatusAccepted || committed == nil {
					t.Fatalf("initial intake %d %s", result.Code, result.Body.String())
				}
				if down.Code != http.StatusAccepted || !strings.Contains(down.Body.String(), committed.ID) {
					t.Fatalf("down over committed intake: %d %s", down.Code, down.Body.String())
				}
				<-stopped
				close(release)
				row, problem := o.store.RequestByIdempotencyKey("shutdown-race")
				fatal(t, problem)
				if row == nil || row.State != committed.State || row.BodyDigest != committed.BodyDigest {
					t.Fatalf("down changed the committed request: before=%+v after=%+v", committed, row)
				}
				if _, problem := o.c.ActivateRecordedRequest(*row); problem == nil || starts.Load() != 1 {
					t.Fatal("post-shutdown machine activation crossed the closing fence")
				}
			}
		})
	}
}
