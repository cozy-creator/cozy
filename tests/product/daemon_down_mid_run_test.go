package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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
			name := map[string]string{"local": machines.Local, "rental": parityRental}[venue.name]
			authorityState := func(row *records.Request) string {
				link, problem := store.MachineExecution(row.ID)
				fatal(t, problem)
				var receipt pb.MachineExecutionReceipt
				must(t, proto.Unmarshal(link.Receipt, &receipt))
				machine, problem := found.Dial(t.Context(), name, "observe outcome while personal daemon is down")
				fatal(t, problem)
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
dependencies=["cozy-runtime>=`+machines.RuntimeFloor+`", "msgspec>=0.19"]
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
