package producttest

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestDaemonDownPreservesInactiveRetainedWork(t *testing.T) {
	root := t.TempDir()
	startDaemonProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{ID: "pr-retained-disconnect", MachineName: "heron", AcceleratorModel: "CPU", AcceleratorCount: 1, State: "ready", Hub: "http://127.0.0.1:1", HourlyRateUSDMicros: 1}))
	paused := recordPrivateTransaction(t, store, "disconnect-paused", "pr-retained-disconnect")
	_, problem = store.RequestPause(paused.ID, "fixture")
	fatal(t, problem)
	_, problem = store.CompleteRequestPause(paused.ID)
	fatal(t, problem)
	blocked := recordPrivateTransaction(t, store, "disconnect-blocked", "pr-retained-disconnect")
	_, problem = store.BlockRetainedWork(blocked.ID, "fixture.blocked", "retained inputs")
	fatal(t, problem)
	completed := recordPrivateTransaction(t, store, "disconnect-completed", "pr-retained-disconnect")
	// This is a historical retained result, with no live executor. Seed its prior
	// terminal state so the actual daemon must preserve it across down/up.
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite"))
	must(t, err)
	_, err = db.Exec("UPDATE requests SET state='succeeded',child_artifacts=1 WHERE id=?", completed.ID)
	must(t, err)
	_, err = db.Exec("INSERT INTO request_output_exports(request_id,directory,outputs,state,updated_at) VALUES(?,?,'[]','pending',?)", paused.ID, filepath.Join(root, "paused-output"), time.Now().UTC().Format(time.RFC3339Nano))
	must(t, err)
	db.Close()
	kept := filepath.Join(root, "retained-output.txt")
	must(t, os.WriteFile(kept, []byte("retained result"), 0600))
	before := map[string]*records.Request{}
	for _, id := range []string{paused.ID, blocked.ID, completed.ID} {
		before[id], problem = store.RequestRow(id)
		fatal(t, problem)
	}
	for cycle := 0; cycle < 2; cycle++ {
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
			if current == nil || current.State != prior.State || current.BodyDigest != prior.BodyDigest || current.Worker != prior.Worker || !current.RetainWork {
				t.Fatalf("retained request changed: before=%+v after=%+v", prior, current)
			}
		}
		rental, problem := store.RentalRow("pr-retained-disconnect")
		fatal(t, problem)
		if rental == nil || rental.State != "ready" {
			t.Fatal("disconnect ended or forgot the rental")
		}
		raw, err := os.ReadFile(kept)
		must(t, err)
		if string(raw) != "retained result" {
			t.Fatal("disconnect changed output bytes")
		}
		if cycle == 0 {
			code, out = runCozy(t, root, "up", "--json")
			if code != 0 {
				t.Fatalf("up [%d]: %s", code, out)
			}
		}
	}
}

func TestDaemonForceDisconnectDoesNotCancelPreparation(t *testing.T) {
	root := t.TempDir()
	startDaemonProcess(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, _ := machineObserverRecord(t, store)
	before, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	code, out := runCozy(t, root, "down", "--json")
	if code == 0 || !strings.Contains(out, "active_work") || !strings.Contains(out, request.ID) {
		t.Fatalf("unaccepted handoff did not hold normal down [%d]: %s", code, out)
	}
	if code, out = runCozy(t, root, "down", "--all", "--force", "--json"); code == 0 {
		t.Fatal("destructive and disconnect modes combined")
	}
	code, out = runCozy(t, root, "down", "--force", "--json")
	if code != 0 {
		t.Fatalf("force disconnect [%d]: %s", code, out)
	}
	after, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	current, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if current.State != request.State || after.CancelRequested || !bytes.Equal(before.Submission, after.Submission) || len(after.Receipt) != 0 {
		t.Fatalf("force changed execution intent: %+v %+v", current, after)
	}
	// A stopped daemon is an idempotent no-op even with incomplete durable work.
	code, out = runCozy(t, root, "down", "--json")
	if code != 0 || !strings.Contains(out, `"changed":false`) {
		t.Fatalf("offline down restarted/refused [%d]: %s", code, out)
	}
	if daemon.Probe(config.Config{Home: root}).Up {
		t.Fatal("offline down restarted the daemon")
	}
}

// The sentinel is a stand-in executable only; the daemon, records, startup Resume,
// CLI shutdown, and process restart are real. A fresh queued canary proves that
// the fixture detects an attempted Runtime start rather than passing unarmed.
func TestDaemonRestartDoesNotSubmitStoppedMachineExecutions(t *testing.T) {
	bin := t.TempDir()
	marker := filepath.Join(bin, "runtime-started")
	script := fmt.Sprintf(`#!/bin/sh
if [ "$1" = "--json" ] && [ "$2" = "version" ]; then
 printf '%%s\n' '{"distribution":"0.18.14","wire_protocol":"cozy.worker.v1+minor.%d"}'
 exit 0
fi
if [ "$1" = "serve" ]; then
 printf '%%s\n' 'started' >> %q
 exit 1
fi
exit 2
`, pb.WireMinor, marker)
	must(t, os.WriteFile(filepath.Join(bin, "cozy-runtime"), []byte(script), 0700)) //cozy:allow sentinel executable verifies that startup Resume does not start stopped work
	must(t, os.WriteFile(filepath.Join(bin, "python"), []byte("#!/bin/sh\nexit 1\n"), 0700))
	seed := func(root, state string) (*records.Store, records.Request) {
		store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		request, _, problem := store.Submit(records.Request{ID: "job-restart-" + state, IdemKey: state, Package: "local/stopped", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true, RetainWork: state == "paused" || state == "blocked"})
		fatal(t, problem)
		fatal(t, store.LinkMachineExecution(request.ID, "local"))
		if state == "failed" {
			// Retained intake still owes custody management, but must not grant
			// permission to execute a request that failed before submission.
			_, problem = store.BeginMachineInput(request.ID, records.AssetBinding{FieldPath: "files", Snapshot: &records.ByteInputSnapshot{Manifest: records.ArtifactObjectRef{Digest: childDigest("4"), Length: 128}, ContentBytes: 1024}})
			fatal(t, problem)
		}
		switch state {
		case "failed", "blocked":
			_, problem = store.FailQueuedRequest(request.ID, map[string]any{"error_type": "fixture.preparation_failed", "error": "failed before Submit"})
			fatal(t, problem)
		case "paused":
			_, problem = store.RequestPause(request.ID, "fixture")
			fatal(t, problem)
			_, problem = store.CompleteRequestPause(request.ID)
			fatal(t, problem)
		case "canceled":
			_, problem = store.CancelQueuedRequest(request.ID, map[string]any{"reason": "fixture"})
			fatal(t, problem)
		case "refused":
			fatal(t, store.RefuseMachineSubmission(request.ID, "fixture.refused", "not accepted"))
		}
		return store, request
	}
	canary := t.TempDir()
	canaryStore, _ := seed(canary, "queued")
	defer canaryStore.Close()
	if code, out := runCozyPath(t, canary, bin, "up", "--json"); code != 0 {
		t.Fatalf("canary up [%d]: %s", code, out)
	}
	waitUntil(t, "fresh queued execution reaches Runtime", func() bool { _, err := os.Stat(marker); return err == nil })
	if code, out := runCozyPath(t, canary, bin, "down", "--force", "--json"); code != 0 {
		t.Fatalf("canary down [%d]: %s", code, out)
	}
	must(t, os.Remove(marker))
	for _, state := range []string{"failed", "paused", "blocked", "canceled", "refused"} {
		t.Run(state, func(t *testing.T) {
			root := t.TempDir()
			store, request := seed(root, state)
			defer store.Close()
			for cycle := 0; cycle < 2; cycle++ {
				if code, out := runCozyPath(t, root, bin, "up", "--json"); code != 0 {
					t.Fatalf("up [%d]: %s", code, out)
				}
				// Resume starts before the API is published; a read round-trip gives its
				// observer goroutines a turn without creating another execution request.
				if code, out := runCozyPath(t, root, bin, "run", "list", "--json"); code != 0 {
					t.Fatalf("list [%d]: %s", code, out)
				}
				time.Sleep(100 * time.Millisecond)
				if code, out := runCozyPath(t, root, bin, "down", "--json"); code != 0 {
					t.Fatalf("stopped work held down [%d]: %s", code, out)
				}
				if _, err := os.Stat(marker); !os.IsNotExist(err) {
					t.Fatal("restart attempted a new Runtime execution for stopped work")
				}
				current, problem := store.RequestRow(request.ID)
				fatal(t, problem)
				link, problem := store.MachineExecution(request.ID)
				fatal(t, problem)
				if current.State != state || current.Ordinal != 0 || len(link.Submission) != 0 || len(link.Receipt) != 0 || link.CancelRequested {
					t.Fatalf("startup revived or canceled stopped work: %+v %+v", current, link)
				}
			}
		})
	}
}

func TestDaemonDownReattachesAcceptedLocalRuntime(t *testing.T) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires the exact Runtime wheel used by CI")
	}
	// Runtime appends its executor socket below this root. Keep the fixture
	// independent of the test function's long name and the kernel sun_path cap.
	root, err := os.MkdirTemp("", "cozy-down-")
	must(t, err)
	claimScratch(root)
	t.Cleanup(func() {
		reapDaemonRoot(root)
		if t.Failed() {
			t.Log("disconnect evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	script := filepath.Join(t.TempDir(), "disconnect.py")
	code := fmt.Sprintf(`# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=0.18.14"]
# [tool.uv.sources]
# cozy-runtime = {path = %q}
# ///
import time
def main():
    time.sleep(5)
`, wheel)
	must(t, os.WriteFile(script, []byte(code), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	status, out := runCozy(t, root, "run", script, "--json", "--idempotency-key", "disconnect-accepted")
	if status != 0 || !strings.Contains(out, `"machine_accepted":true`) {
		t.Fatalf("Runtime acceptance [%d]: %s", status, out)
	}
	request, problem := store.RequestByIdempotencyKey("disconnect-accepted")
	fatal(t, problem)
	before, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if before == nil || len(before.Receipt) == 0 || before.Collected {
		t.Fatal("execution did not remain live through the receipt boundary")
	}
	runtimeProcess, err := os.ReadFile(filepath.Join(root, "runtime", "process.json"))
	must(t, err)
	status, out = runCozy(t, root, "down", "--json")
	if status != 0 {
		t.Fatalf("accepted Runtime blocked down [%d]: %s", status, out)
	}
	if daemon.Probe(config.Config{Home: root}).Up {
		t.Fatal("client stayed online")
	}
	currentProcess, err := os.ReadFile(filepath.Join(root, "runtime", "process.json"))
	must(t, err)
	if !bytes.Equal(runtimeProcess, currentProcess) {
		t.Fatal("disconnect replaced the Runtime process identity")
	}
	status, out = runCozy(t, root, "up", "--json")
	if status != 0 {
		t.Fatalf("reattach up [%d]: %s", status, out)
	}
	status, out = runCozy(t, root, "run", "watch", request.ID, "--json")
	if status != 0 {
		t.Fatalf("reattach collection [%d]: %s", status, out)
	}
	currentProcess, err = os.ReadFile(filepath.Join(root, "runtime", "process.json"))
	must(t, err)
	if !bytes.Equal(runtimeProcess, currentProcess) {
		t.Fatal("reattachment replaced the independently running Runtime")
	}
	after, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	current, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if current.State != "succeeded" || current.Ordinal != 1 || !after.Collected || after.CancelRequested || len(attempts) != 0 || !bytes.Equal(before.Submission, after.Submission) || !bytes.Equal(before.Receipt, after.Receipt) {
		t.Fatalf("reattach resubmitted/canceled/replaced execution: %+v %+v", current, after)
	}
	status, out = runCozy(t, root, "down", "--json")
	if status != 0 {
		t.Fatalf("completed down [%d]: %s", status, out)
	}
}

type shutdownRaceResolver struct {
	publishedRouteResolver
	entered chan struct{}
	release chan struct{}
}

func (r shutdownRaceResolver) ResolveRemoteJob(pkg, release, function string, models []orchestrator.ModelRef, deferred bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
	if r.entered != nil {
		close(r.entered)
		<-r.release
	}
	return r.publishedRouteResolver.ResolveRemoteJob(pkg, release, function, models, deferred)
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
				options.Cfg.RentalsMaxHourlySpendUSDMicros = 1_000_000
				options.StartMachineExecution = func(records.Request) *exit.Error {
					starts.Add(1)
					if !beforeCommit {
						close(entered)
						<-release
					}
					return nil
				}
			})
			o.cfg.RentalsMaxHourlySpendUSDMicros = 1_000_000
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
				if down.Code != http.StatusConflict || !strings.Contains(down.Body.String(), "active_work") {
					t.Fatalf("committed intake was not protected: %d %s", down.Code, down.Body.String())
				}
				select {
				case <-stopped:
					t.Fatal("refused down stopped daemon")
				default:
				}
				result := <-replied
				if result.Code != http.StatusAccepted {
					t.Fatalf("initial intake %d %s", result.Code, result.Body.String())
				}
				committed, problem := o.store.RequestByIdempotencyKey("shutdown-race")
				fatal(t, problem)
				if committed == nil {
					t.Fatal("initial intake has no durable request")
				}
				forced := call("/v1/local/daemon/down", map[string]bool{"force": true})
				if forced.Code != http.StatusAccepted {
					t.Fatalf("force %d %s", forced.Code, forced.Body.String())
				}
				<-stopped
				close(release)
				row, problem := o.store.RequestByIdempotencyKey("shutdown-race")
				fatal(t, problem)
				if row == nil || row.State != committed.State || row.BodyDigest != committed.BodyDigest {
					t.Fatalf("force changed the committed request: before=%+v after=%+v", committed, row)
				}
				if _, problem := o.c.ActivateRecordedRequest(*row); problem == nil || starts.Load() != 1 {
					t.Fatal("post-shutdown machine activation crossed the closing fence")
				}
			}
		})
	}
}

// The SQL owed predicate must not turn an unsent failed request into runnable
// work. Accepted receipts still authorize observation of a real later outcome.
func TestMachineRestartOwedPolicySeparatesFailedIntentFromAcceptedOutcome(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	// An accepted workspace may report completion after its observer disconnected.
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	_, problem := store.FailQueuedRequest(request.ID, map[string]any{"error_type": "fixture.observer_interrupted"})
	fatal(t, problem)
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "succeeded"}
	owed, problem := store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	if !owed {
		t.Fatal("accepted outcome lost its observer")
	}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	observed, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if observed.State != "succeeded" {
		t.Fatal("real accepted outcome could not reconcile")
	}
}
