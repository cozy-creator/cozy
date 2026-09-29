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
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// `cozy down` mid-run stops the daemon and nothing else. This computer's machine and the
// rental — the same Host binary, each with a real Runtime — keep running the work, and
// `cozy run watch` brings the daemon back, reattaches, and delivers the run's file, sha
// verified. A run canceling when the daemon stops settles canceled after it comes back.
func TestDownMidRunLeavesTheWorkRunning(t *testing.T) {
	_, root, _, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", downProject(t), "--editable"); code != 0 {
		t.Fatalf("installing the down package [exit %d]\n%s", code, out)
	}
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
				host, err := os.ReadFile(filepath.Join(root, "machine", "host.json"))
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
				now, err := os.ReadFile(filepath.Join(root, "machine", "host.json"))
				if err != nil || !bytes.Equal(now, host) || json.Unmarshal(now, &record) != nil || syscall.Kill(record.PID, 0) != nil {
					t.Fatalf("down touched this computer's machine Host: %s", now)
				}
			}
			open := func(key string) { must(t, os.WriteFile(filepath.Join(gates, key), nil, 0o600)) }

			done := start("done")
			down(done, "")
			open("done")
			code, out := runCozy(t, root, "run", "watch", done.ID, "--json")
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

			canceled := start("canceled")
			if code, out := runCozy(t, root, "run", "cancel", canceled.ID, "--json"); code != 0 {
				t.Fatalf("cancel [exit %d]\n%s", code, out)
			}
			eventually(t, root, "the run canceling", func() bool {
				row, _ := store.RequestRow(canceled.ID)
				return row != nil && row.State == "canceling"
			})
			down(canceled, "canceling")
			open("canceled")
			runCozy(t, root, "run", "watch", canceled.ID, "--json")
			eventually(t, root, "the canceled run to settle", func() bool {
				row, _ := store.RequestRow(canceled.ID)
				link, _ := store.MachineExecution(canceled.ID)
				return row != nil && row.State == "canceled" && link != nil && len(link.PendingControl) == 0
			})
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
