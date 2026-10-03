package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// `cozy down` mid-run stops the daemon and nothing else. This computer's machine and the
// rental — the same Host binary, each with a real Runtime — keep running the work, and
// `cozy run watch` brings the daemon back, reattaches, and delivers the run's file, sha
// verified. Delivered cancellation survives handoff; undelivered intent is reconciled
// without rewriting a successful outcome that won before cancellation arrived.
func TestDownMidRunLeavesTheWorkRunning(t *testing.T) {
	h, root, layout, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", downProject(t), "--editable"); code != 0 {
		t.Fatalf("installing the down package [exit %d]\n%s", code, out)
	}
	// Observe the authority directly while the personal daemon is down. This uses
	// the same authenticated machine client as the parity inventory proof.
	found := &machines.Resolver{Host: machines.NewHost(layout.Machine, "", nil), HubOrigin: h.server.URL,
		Rentals: rental.Resolver(layout, store), UseRental: func(string, string) (func(), *exit.Error) { return func() {}, nil },
		RentalKey: func(id string) (rental.CreatorIdentity, *exit.Error) { return rental.CreatorIdentityFor(layout, id) }}
	defer found.Forget(machines.Local)
	defer found.Forget(parityRental)
	for _, venue := range []struct {
		name string
		args []string
	}{{"local", nil}, {"rental", []string{"--rental=tessa"}}} {
		t.Run(venue.name, func(t *testing.T) {
			gates := t.TempDir()
			name := map[string]string{"local": machines.Local, "rental": parityRental}[venue.name]
			var started []string
			var observer *machines.Machine
			t.Cleanup(func() {
				if t.Failed() {
					downFailureDiagnostics(t, root, venue.name, h.provider, store, observer, started)
				}
			})
			// start runs `make` behind its own gate and returns once it runs on its machine.
			start := func(key string) *records.Request {
				args := append([]string{"run", "local/down-proof/make", "gate=" + filepath.Join(gates, key), "size=200000",
					"--json", "--idempotency-key", venue.name + "-" + key}, venue.args...)
				if code, out := runCozy(t, root, args...); code != 0 {
					t.Fatalf("submitting %s [exit %d]\n%s", key, code, out)
				}
				var row *records.Request
				eventually(t, root, key+" running on its machine", func() bool {
					row, _ = store.RequestByIdempotencyKey(venue.name + "-" + key)
					if row != nil && (len(started) == 0 || started[len(started)-1] != row.ID) {
						started = append(started, row.ID)
					}
					return row != nil && observed(store, row.ID) == "running"
				})
				return row
			}
			down := func(want *records.Request, state string) {
				host, err := os.ReadFile(filepath.Join(root, "machine", "agent.json"))
				must(t, err)
				code, out := runCozy(t, root, "down", "--json")
				if code != 0 || !strings.Contains(out, `"daemon":"stopped"`) || !strings.Contains(out, want.ID+" ("+state) {
					t.Fatalf("down did not stop and name %s (%s) [exit %d]\n%s", want.ID, state, code, out)
				}
				if daemon.Probe(config.Config{Home: root}).Up {
					t.Fatal("down left the daemon running")
				}
				t.Logf("cozy down --json: %s", out)
				var record struct{ PID int }
				now, err := os.ReadFile(filepath.Join(root, "machine", "agent.json"))
				if err != nil || !bytes.Equal(now, host) || json.Unmarshal(now, &record) != nil || syscall.Kill(record.PID, 0) != nil {
					t.Fatalf("down touched this computer's machine Host: %s", now)
				}
			}
			open := func(key string) { must(t, os.WriteFile(filepath.Join(gates, key), nil, 0o600)) }
			authorityState := func(row *records.Request) string {
				link, problem := store.MachineExecution(row.ID)
				fatal(t, problem)
				var receipt pb.MachineExecutionReceipt
				must(t, proto.Unmarshal(link.Receipt, &receipt))
				machine, problem := found.Dial(t.Context(), name, "observe outcome while personal daemon is down")
				fatal(t, problem)
				observer = machine
				defer machine.Close()
				state, err := machine.Host.GetMachineExecution(t.Context(), &pb.MachineExecutionQuery{
					Claim: machine.Claim, RequestId: row.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId})
				must(t, err)
				return state.State
			}
			watchCompleted := func(row *records.Request) {
				code, out := runCozy(t, root, "run", "watch", row.ID, "--json")
				var result struct {
					Status string
					Saved  []struct{ Output, Path, Digest string }
				}
				if code != 0 || json.Unmarshal([]byte(out), &result) != nil || result.Status != "completed" || len(result.Saved) != 1 {
					t.Fatalf("watch did not deliver the run's output [exit %d]\n%s", code, out)
				}
				data, err := os.ReadFile(result.Saved[0].Path)
				want := digestOf(downProofBytes(200000))
				if err != nil || digestOf(data) != want || result.Saved[0].Digest != want {
					t.Fatalf("the output file at %s is not the run's bytes (%v): %s", result.Saved[0].Path, err, out)
				}
			}
			watchCanceled := func(request *records.Request) {
				runCozy(t, root, "run", "watch", request.ID, "--json")
				eventually(t, root, "the canceled run to settle", func() bool {
					row, _ := store.RequestRow(request.ID)
					link, _ := store.MachineExecution(request.ID)
					return row != nil && row.State == "canceled" && link != nil && !link.CancelRequested && len(link.PendingControl) == 0
				})
			}

			done := start("done")
			found.Held = func(string) bool { return true }
			var problem *exit.Error
			observer, problem = found.Dial(t.Context(), name, "read-only cancellation diagnostics")
			fatal(t, problem)
			down(done, "")
			open("done")
			watchCompleted(done)

			canceled := start("canceled")
			if code, out := runCozy(t, root, "run", "cancel", canceled.ID, "--json"); code != 0 {
				t.Fatalf("cancel [exit %d]\n%s", code, out)
			}
			eventually(t, root, "Runtime acknowledging cancellation", func() bool {
				row, _ := store.RequestRow(canceled.ID)
				return row != nil && row.State == "canceling" && observed(store, canceled.ID) == "canceling"
			})
			down(canceled, "canceling")
			open("canceled")
			watchCanceled(canceled)

			// Recreate the handoff boundary deterministically: persist intent with
			// no daemon present to deliver it. Natural completion must keep its
			// successful terminal and bytes; waiting work must receive the intent
			// when observation restarts.
			for _, natural := range []bool{true, false} {
				key := map[bool]string{true: "completion-before-delivery", false: "intent-before-restart"}[natural]
				pending := start(key)
				down(pending, "")
				accepted, problem := store.RequestMachineCancellation(pending.ID, "cozy job cancel")
				fatal(t, problem)
				link, problem := store.MachineExecution(pending.ID)
				fatal(t, problem)
				if !accepted || !link.CancelRequested || len(link.PendingControl) != 0 || authorityState(pending) != "running" {
					t.Fatal("fixture did not retain an undelivered cancellation for a running execution")
				}
				if natural {
					open(key)
					eventually(t, root, "natural completion while the daemon is down", func() bool { return authorityState(pending) == "succeeded" })
					if daemon.Probe(config.Config{Home: root}).Up {
						t.Fatal("authority observation restarted the personal daemon")
					}
					found.Forget(name)
					watchCompleted(pending)
					eventually(t, root, "late cancellation reconciled without changing success", func() bool {
						row, _ := store.RequestRow(pending.ID)
						link, _ := store.MachineExecution(pending.ID)
						return row != nil && row.State == "succeeded" && link != nil && !link.CancelRequested && len(link.PendingControl) == 0
					})
				} else {
					found.Forget(name)
					if code, out := runCozy(t, root, "up", "--json"); code != 0 {
						t.Fatalf("restart observer [%d]: %s", code, out)
					}
					eventually(t, root, "delivery of retained cancellation after restart", func() bool { return observed(store, pending.ID) == "canceling" })
					open(key)
					watchCanceled(pending)
				}
			}
			if link, _ := store.MachineExecution(done.ID); link == nil || link.MachineID != map[string]string{"local": machines.Local, "rental": parityRental}[venue.name] {
				t.Fatalf("the run did not execute on its %s machine: %+v", venue.name, link)
			}
		})
	}
}

// Failure snapshots deliberately select status/control facts. Neither database
// copies, execution offers, event bodies nor credential-bearing records are kept.
func downFailureDiagnostics(t *testing.T, root, venue, provider string, store *records.Store, observer *machines.Machine, ids []string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	facts := map[string]any{"venue": venue}
	if observer != nil {
		facts["observer_connection"] = observer.Conn.GetState().String()
	}
	rows := []map[string]any{}
	for _, id := range ids {
		row := map[string]any{"request": id}
		request, problem := store.RequestRow(id)
		if problem == nil && request != nil {
			row["creator_state"] = request.State
		}
		link, problem := store.MachineExecution(id)
		if problem == nil && link != nil {
			row["cancel_requested"], row["collected"], row["cursor"] = link.CancelRequested, link.Collected, link.RemoteCursor
			row["pending_control"] = len(link.PendingControl) != 0
			row["pending_control_bytes"] = len(link.PendingControl)
			var pending pb.MachineExecutionControl
			if proto.Unmarshal(link.PendingControl, &pending) == nil {
				row["pending_command"], row["pending_action"] = pending.CommandId, pending.Action.String()
			}
			var receipt pb.MachineExecutionReceipt
			if observer != nil && proto.Unmarshal(link.Receipt, &receipt) == nil {
				query := &pb.MachineExecutionQuery{Claim: observer.Claim, RequestId: id, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}
				state, err := observer.Host.GetMachineExecution(ctx, query)
				if err != nil {
					row["runtime_read_code"] = status.Code(err).String()
				} else {
					row["runtime_state"], row["runtime_sequence"], row["runtime_collected"] = state.State, state.Sequence, state.Collected
					row["runtime_generation"], row["runtime_attempt"] = state.Generation, state.AttemptOrdinal
					row["worker_boot"], row["workspace"] = state.WorkerBootId, state.ExecutionWorkspaceId
				}
				page, err := observer.Host.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{Execution: query, After: uint64(max(link.RemoteCursor, 0)), Limit: 16})
				if err != nil {
					row["runtime_event_code"] = status.Code(err).String()
				} else {
					kinds := []string{}
					for _, event := range page.Events {
						kinds = append(kinds, event.Kind)
					}
					row["event_head"], row["event_kinds"] = page.HeadSequence, kinds
				}
			}
		}
		rows = append(rows, row)
	}
	facts["requests"] = rows
	journal := machineJournal(root)
	if venue == "rental" {
		journal = filepath.Join(provider, "var/lib/tensorfs/.cozy-workspace/journal.sqlite3")
	}
	python := filepath.Join(root, "machine/root/opt/cozy/python/bin/python")
	command := exec.CommandContext(ctx, python, append([]string{"-I", "-c", `import json,sqlite3,sys
from pathlib import Path
path=Path(sys.argv[1]); ids=sys.argv[2:]
if not path.is_file(): print('{"journal":"absent"}'); raise SystemExit()
c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True); c.row_factory=sqlite3.Row
out={}; marks=','.join('?' for _ in ids)
for table,fields in [('executions','request,state,desired,retention_waived,collected,sequence,generation'),('attempts','request,ordinal,state,length(outcome) AS outcome_bytes')]:
    out[table]=[dict(row) for row in c.execute('SELECT '+fields+' FROM '+table+' WHERE request IN ('+marks+') LIMIT 32',ids)] if ids else []
out['commands']=[]
if ids:
    for row in c.execute('SELECT request,command,intent,length(response) AS response_bytes FROM execution_commands WHERE request IN ('+marks+') LIMIT 32',ids):
        command={'request':row['request'],'command':row['command'],'response_bytes':row['response_bytes']}
        try: command['action']=json.loads(row['intent']).get('action','')
        except (ValueError,TypeError): command['action']='unreadable'
        out['commands'].append(command)
print(json.dumps(out,sort_keys=True))
`, journal}, ids...)...)
	if output, err := command.Output(); err == nil {
		var data map[string]json.RawMessage
		if json.Unmarshal(output, &data) == nil {
			facts["journal"] = data
		}
	} else {
		facts["journal_read"] = "unavailable"
	}
	encoded, err := json.MarshalIndent(facts, "", "  ")
	if err != nil {
		t.Log("cancellation snapshot unavailable")
		return
	}
	t.Logf("cancellation status snapshot: %s", encoded)
	if directory := *failureDiagnosticsDirectory; directory != "" {
		if os.MkdirAll(directory, 0700) == nil {
			if err := os.WriteFile(filepath.Join(directory, "down-mid-run-"+venue+".json"), encoded, 0600); err != nil {
				t.Log("could not save cancellation snapshot")
			}
		}
	}
}

// observed is the state the daemon last read from the run's machine.
func observed(store *records.Store, id string) string {
	link, _ := store.MachineExecution(id)
	var state pb.MachineExecutionState
	if link == nil || proto.Unmarshal(link.ObservedState, &state) != nil {
		return ""
	}
	return state.State
}

func downProofBytes(size int) []byte {
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i % 251)
	}
	return data
}

// downProject is local/down-proof: `make` waits for its gate file without looking at a
// cancel, so a canceled run stays canceling until the gate opens, then saves `size`
// deterministic bytes as its one file.
func downProject(t *testing.T) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "down-proof")
	must(t, os.MkdirAll(project, 0o700))
	sources := ""
	if *machineRuntimeWheel != "" {
		sources = fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="down-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=`+runtimeFloor+`", "msgspec>=0.19"]
[project.entry-points."cozy.application"]
default="down_proof:app"
`+sources+`[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["down_proof.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"down_proof:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "down_proof.py"), []byte(`import asyncio
import os
from typing import Annotated

import msgspec
from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs, invocable

Blob = Annotated[FileAsset, AssetBound(max_bytes=1 << 20, media_types=("application/octet-stream",))]


class Result(msgspec.Struct):
    blob: Blob


@invocable(memoize=False)
async def make(ctx: Context, out: Outputs, *, gate: str, size: int) -> Result:
    while not os.path.exists(gate):
        await asyncio.sleep(0.1)
    ctx.raise_if_cancelled()
    data = bytes(i % 251 for i in range(size))
    return Result(out.save_bytes(data, media_type="application/octet-stream"))


app = App()
app.job(make)
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking the down package: %v\n%s", err, out)
	}
	return project
}

// A run whose editable package is uploading when `cozy down` stops the daemon is not
// lost: the next daemon uploads it again, and it reaches Runtime once.
func TestDownMidUploadResumesTheUpload(t *testing.T) {
	machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	pod := newEditablePod(machines)
	pod.block, pod.released = make(chan struct{}), make(chan struct{})
	root, store := rentedEditable(t, pod)
	if code, out := runCozy(t, root, "run", "local/upload-proof/main", "steps=1", "--rental=tessa", "--json", "--idempotency-key", "down"); code != 0 {
		t.Fatalf("the rented run was refused [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the editable package's upload", closed(pod.block))
	row, problem := store.RequestByIdempotencyKey("down")
	fatal(t, problem)
	code, out := runCozy(t, root, "down")
	if code != 0 || !strings.Contains(out, row.ID) || !strings.Contains(out, "the next cozy command reattaches") {
		t.Fatalf("down did not stop and name the uploading run [exit %d]: %s", code, out)
	}
	t.Logf("cozy down:\n%s", out)
	waitFor(t, root, "the upload's stream to end with the daemon", closed(pod.released))
	pod.unblock()
	if code, out = runCozy(t, root, "run", "watch", row.ID, "--json"); code != 0 || !strings.Contains(out, `"status":"completed"`) {
		t.Fatalf("the run did not complete after the daemon came back [exit %d]: %s", code, out)
	}
	if calls, prepared := pod.counts(); calls < 2 || prepared != 1 || len(machines.submitted()) != 1 {
		t.Fatalf("%d upload call(s), %d preparation(s), %d submission(s); want a resumed upload that reaches Runtime once",
			calls, prepared, len(machines.submitted()))
	}
}
