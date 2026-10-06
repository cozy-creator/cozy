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
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// v1RetainedRun records a cozy.machine.v1 run on rental, kept for its owner.
func v1RetainedRun(t *testing.T, store *records.Store, label, rental string) records.Request {
	t.Helper()
	request, _, problem := store.Submit(records.Request{ID: "req-retained-" + label, IdemKey: "retained-" + label,
		BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "local/retained-proof", Entrypoint: "prepare", Kind: "job",
		Payload: []byte(`{"seed":17}`), Worker: rental, Rental: true, RentalRequired: true, RetainWork: true, MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, rental))
	return request
}

// `cozy down` disconnects; it never ends work kept for its owner. A paused run, a blocked one
// and a finished one whose result is still on its machine stay exactly as they were across
// two disconnects, and so does their rental.
func TestDaemonDownPreservesInactiveRetainedWork(t *testing.T) {
	root := t.TempDir()
	startDaemonProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const rental = "pr-retained-disconnect"
	fatal(t, store.RecordRental(records.Rental{ID: rental, MachineName: "heron", AcceleratorModel: "CPU", AcceleratorCount: 1, State: "ready", Hub: "http://127.0.0.1:1", HourlyRateUSDMicros: 1}))
	paused := v1RetainedRun(t, store, "paused", rental)
	_, problem = store.MarkRunV1Sent(paused.ID)
	fatal(t, problem)
	fatal(t, store.AcceptRunV1(paused.ID, rental, &v1.RunState{Id: paused.ID, Number: 1, State: "running", Attempt: 1}))
	fatal(t, store.ObserveRunV1(paused.ID, &v1.RunEvent{Sequence: 1, Event: &v1.RunEvent_State{State: &v1.RunState{Id: paused.ID, Number: 1, State: "paused", Sequence: 1, Attempt: 1}}}, nil))
	blocked := v1RetainedRun(t, store, "blocked", rental)
	_, problem = store.BlockRetainedWork(blocked.ID, "fixture.blocked", "retained inputs")
	fatal(t, problem)
	finished := v1RetainedRun(t, store, "finished", rental)
	_, problem = store.MarkRunV1Sent(finished.ID)
	fatal(t, problem)
	fatal(t, store.AcceptRunV1(finished.ID, rental, &v1.RunState{Id: finished.ID, Number: 2, State: "running", Attempt: 1}))
	fatal(t, store.RecordRunOutcomeV1(finished.ID, records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded", Result: []byte(`{}`)},
		Refused: exit.Named(exit.Unavailable, "output.write_failed", "this computer could not write the run's output")}))
	kept := filepath.Join(root, "retained-output.txt")
	must(t, os.WriteFile(kept, []byte("retained result"), 0o600))
	before := map[string]*records.Request{}
	links := map[string]*records.MachineExecution{}
	for _, id := range []string{paused.ID, blocked.ID, finished.ID} {
		before[id], problem = store.RequestRow(id)
		fatal(t, problem)
		links[id], problem = store.MachineExecution(id)
		fatal(t, problem)
	}
	if !before[paused.ID].RetainWork || !before[blocked.ID].RetainWork || before[paused.ID].State != "paused" || before[blocked.ID].State != "blocked" {
		t.Fatal("the fixture's paused and blocked runs are not kept for their owner")
	}
	for cycle := range 2 {
		code, out := runCozy(t, root, "down", "--json")
		if code != 0 || !strings.Contains(out, `"daemon":"stopped"`) {
			t.Fatalf("retained down [%d]: %s", code, out)
		}
		if daemon.Probe(config.Config{Home: root}).Up {
			t.Fatal("daemon remained online")
		}
		for id, prior := range before {
			current, problem := store.RequestRow(id)
			fatal(t, problem)
			link, problem := store.MachineExecution(id)
			fatal(t, problem)
			if current == nil || current.State != prior.State || current.BodyDigest != prior.BodyDigest || current.Worker != prior.Worker || current.RetainWork != prior.RetainWork ||
				link.CancelRequested || link.Collected != links[id].Collected || !bytes.Equal(link.Receipt, links[id].Receipt) || !bytes.Equal(link.Outcome, links[id].Outcome) {
				t.Fatalf("retained run %s changed: before=%+v after=%+v", id, prior, current)
			}
		}
		row, problem := store.RentalRow(rental)
		fatal(t, problem)
		if row == nil || row.State != "ready" {
			t.Fatal("disconnect ended or forgot the rental")
		}
		if raw, err := os.ReadFile(kept); err != nil || string(raw) != "retained result" {
			t.Fatal("disconnect changed output bytes")
		}
		if cycle == 0 {
			if code, out = runCozy(t, root, "up", "--json"); code != 0 {
				t.Fatalf("up [%d]: %s", code, out)
			}
		}
	}
}

// A handoff its machine has not confirmed does not hold `cozy down`: the daemon stops, names it
// in flight, and leaves it exactly as it was, neither canceled nor accepted, for the next
// daemon to confirm. A second down is a no-op.
func TestDaemonDownDoesNotCancelAnUnconfirmedHandoff(t *testing.T) {
	root := t.TempDir()
	startDaemonProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, _, problem := store.Submit(records.Request{ID: "job-handoff", IdemKey: "handoff", Package: "local/example", Entrypoint: "main",
		Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "pr-unreachable-machine"))
	if sent, problem := store.MarkRunV1Sent(request.ID); problem != nil || !sent {
		t.Fatalf("the fixture's dispatch was not marked: %v", problem)
	}
	before, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	code, out := runCozy(t, root, "down", "--json")
	if code != 0 || !strings.Contains(out, `"daemon":"stopped"`) || !strings.Contains(out, request.ID) {
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
	if daemon.Probe(config.Config{Home: root}).Up {
		t.Fatal("offline down restarted the daemon")
	}
}

type shutdownRaceResolver struct {
	publishedRouteResolver
	entered chan struct{}
	release chan struct{}
}

func (r shutdownRaceResolver) ResolveRemoteJob(hub, pkg, release, function string, models []orchestrator.ModelRef, deferred bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
	if r.entered != nil {
		close(r.entered)
		<-r.release
	}
	return r.publishedRouteResolver.ResolveRemoteJob(hub, pkg, release, function, models, deferred)
}

func TestDaemonDownFencesConcurrentMachineIntake(t *testing.T) {
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
			go func() { replied <- call("/v1/local/jobs", submission) }()
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
			down := call("/v1/local/daemon/down", map[string]bool{})
			if beforeCommit {
				if down.Code != http.StatusAccepted {
					t.Fatalf("pre-commit down %d %s", down.Code, down.Body.String())
				}
				<-stopped
				close(release)
				result := <-replied
				if result.Code != http.StatusServiceUnavailable || !strings.Contains(result.Body.String(), "daemon_shutting_down") {
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
