package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// This is an ordinary CLI/controller and a real native machine in a test-owned
// home. The authored byte job has a file gate; there are no product test modes.
func nativeLifecycleHome(t *testing.T) (string, *records.Store, *machines.V1) {
	t.Helper()
	integration(t)
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the actual native machine")
	}
	root, err := os.MkdirTemp(scratchBase, "czvl")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("native lifecycle evidence retained at %s", root)
		} else {
			_ = removeAllForce(root)
		}
	})
	project := t.TempDir()
	sources, floor := "", "0.18.102"
	if *machineRuntimeWheel != "" {
		floor = runtimeFixtureVersion(t, *machineRuntimeWheel)
		sources = fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="native-lifecycle"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=`+floor+`", "msgspec>=0.19"]
[project.entry-points."cozy.application"]
default="native_lifecycle:app"
`+sources+`[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["native_lifecycle.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"native_lifecycle:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "native_lifecycle.py"), []byte(`import asyncio
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
        await asyncio.sleep(0.05)
    ctx.raise_if_cancelled()
    return Result(out.save_bytes(bytes(i % 251 for i in range(size)), media_type="application/octet-stream"))

app = App()
app.job(make)
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking authored lifecycle fixture: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("installing authored lifecycle fixture [%d]: %s", code, out)
	}
	warm := filepath.Join(root, "warm-gate")
	must(t, os.WriteFile(warm, nil, 0o600))
	if code, out := runCozy(t, root, "run", "local/native-lifecycle/make", "gate="+warm, "size=64", "--idempotency-key=native-warm", "--await", "--json"); code != 0 {
		t.Fatalf("warming native fixture [%d]: %s", code, out)
	}
	store, problem := records.Open(home.Paths(root).DB)
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	resolver := &machines.Resolver{Host: machines.NewHost(filepath.Join(root, "machine"), "", nil)}
	machine, problem := resolver.DialV1(machines.AttachOnly(t.Context()), machines.Local, "native lifecycle test")
	fatal(t, problem)
	t.Cleanup(machine.Close)
	return root, store, machine
}

func nativeLifecycleRequest(t *testing.T, store *records.Store, key string) *records.Request {
	t.Helper()
	var row *records.Request
	landed(t, "native run accepted and running", func() bool {
		row, _ = store.RequestByIdempotencyKey(key)
		if row == nil {
			return false
		}
		accepted, problem := store.RunV1(row.ID)
		return problem == nil && accepted && row.State == "dispatching"
	})
	return row
}

func nativeLifecycleBytes(t *testing.T, directory string, size int) {
	t.Helper()
	want := make([]byte, size)
	for i := range want {
		want[i] = byte(i % 251)
	}
	var found int
	must(t, filepath.WalkDir(directory, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		data, err := os.ReadFile(path)
		if err == nil && bytes.Equal(data, want) {
			found++
		}
		return err
	}))
	if found != 1 {
		t.Fatalf("native output export has %d exact authored byte files; want one in %s", found, directory)
	}
}

func TestV1CancelFencesDelayedNativeAcceptance(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("uses Linux's observable flock wait to hold the real acceptance boundary")
	}
	root, store, machine := nativeLifecycleHome(t)
	frame, err := machine.Status(t.Context())
	must(t, err)
	var installation string
	for _, environment := range frame.Environments {
		if environment.Package == "local/native-lifecycle" {
			installation = environment.Installation
		}
	}
	if installation == "" {
		t.Fatal("native Status omitted the authored fixture installation")
	}
	lock, err := os.OpenFile(filepath.Join(machineStore(root), "tmp", "writers", "recovery.lock"), os.O_RDWR, 0)
	must(t, err)
	defer lock.Close()
	must(t, flock.Exclusive(lock))
	defer flock.Release(lock)
	fdinfo, err := os.ReadFile(fmt.Sprintf("/proc/self/fdinfo/%d", lock.Fd()))
	must(t, err)
	var inode string
	for _, line := range strings.Split(string(fdinfo), "\n") {
		if strings.HasPrefix(line, "ino:") {
			inode = strings.TrimSpace(strings.TrimPrefix(line, "ino:"))
		}
	}
	if inode == "" {
		t.Fatal("cannot identify the real writer-lock inode")
	}
	payload, err := json.Marshal(map[string]any{"gate": filepath.Join(root, "delayed-gate"), "size": 64})
	must(t, err)
	request, _, problem := store.Submit(records.Request{ID: "native-delayed-acceptance", IdemKey: "native-delayed-acceptance",
		Package: "local/native-lifecycle", Entrypoint: "make", Kind: "job", Payload: payload,
		BodyDigest: childDigest("d"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, machines.Local))
	fatal(t, store.AppendEvent(request.ID, records.RunV1Sent, 0, map[string]any{"machine": machines.Local}))
	stream, err := machine.Run(t.Context(), request.ID, 0, &v1.RunSpec{Kind: v1.RunKind_RUN_KIND_JOB, Entrypoint: "make", Payload: payload,
		Source: &v1.RunSpec_Installation{Installation: installation}})
	must(t, err)
	landed(t, "native acceptance blocked on the real writer lock", func() bool {
		locks, err := os.ReadFile("/proc/locks")
		if err != nil {
			return false
		}
		for _, line := range strings.Split(string(locks), "\n") {
			if strings.Contains(line, "-> FLOCK") {
				for _, field := range strings.Fields(line) {
					if strings.HasSuffix(field, ":"+inode) {
						return true
					}
				}
			}
		}
		return false
	})
	probe, err := machine.Run(t.Context(), request.ID, 0, nil)
	must(t, err)
	_, err = probe.Recv()
	if status.Code(err) != codes.NotFound {
		t.Fatalf("fixture was already accepted before explicit cancellation: %v", err)
	}
	if code, out := runCozy(t, root, "run", "cancel", request.ID, "--json"); code != 0 {
		t.Fatalf("explicit native cancel [%d]: %s", code, out)
	}
	landed(t, "durable cancellation resolved before delayed acceptance", func() bool {
		row, _ := store.RequestRow(request.ID)
		return row != nil && row.State == "canceled"
	})
	canceled, err := machine.Run(t.Context(), request.ID, 0, nil)
	must(t, err)
	state, err := canceled.Recv()
	must(t, err)
	if state.GetState().GetState() != "canceled" {
		t.Fatalf("native machine did not reserve the canceled id: %v", state)
	}
	flock.Release(lock)
	for {
		event, err := stream.Recv()
		must(t, err)
		if event.GetState().GetState() == "running" {
			t.Fatal("delayed native acceptance started work after its explicit cancel")
		}
		if outcome := event.GetOutcome(); outcome != nil {
			if outcome.Status != "canceled" {
				t.Fatalf("delayed native run ended %q", outcome.Status)
			}
			break
		}
	}
}
