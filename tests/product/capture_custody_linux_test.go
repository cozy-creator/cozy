package producttest

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestCapturePublicationExcludesTerminalCleanupAcrossProcesses(t *testing.T) {
	o := hostOwner(t, fmt.Sprintf("capture-publication-%d", time.Now().UnixNano()))
	digest := func(b byte) []byte { return bytes.Repeat([]byte{b}, 32) }
	raw, hash, err := canonical.Identity(&pb.LocalPackageRevision{Package: "local/restored", Release: "1.0.0",
		SourceDigest: digest(1), PackageInterface: &pb.Ref{Digest: digest(2), Length: 1},
		Files: []*pb.LocalPackageFileRef{{Digest: digest(3), Filename: "restored-1.0.0-py3-none-any.whl", Length: 1}}})
	must(t, err)
	revision, err := canonical.Spell(hash)
	must(t, err)
	root := filepath.Join(o.l.LocalPackages, strings.TrimPrefix(revision, "sha256:"))
	must(t, os.MkdirAll(root, 0700))
	must(t, os.WriteFile(filepath.Join(root, "revision.json"), raw, 0600))
	_, _, problem := o.store.Submit(records.Request{ID: "finishing-old", IdemKey: "finishing-old", BodyDigest: revision,
		Package: "local/restored", Entrypoint: "main", Payload: []byte(`{}`), LocalPackageDigest: revision})
	fatal(t, problem)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	// A different process holds the same kernel writer lock as CLI publication.
	child := exec.CommandContext(ctx, "flock", "--nonblock", o.l.Lock, "cat")
	input, err := child.StdinPipe()
	must(t, err)
	output, err := child.StdoutPipe()
	must(t, err)
	must(t, child.Start())
	t.Cleanup(func() { input.Close(); cancel(); _ = child.Wait() })
	_, err = input.Write([]byte("writer held\n"))
	must(t, err)
	line, err := bufio.NewReader(output).ReadString('\n')
	must(t, err)
	if line != "writer held\n" {
		t.Fatal("publication process did not acquire the writer")
	}
	fatal(t, o.c.CancelQueued("finishing-old", "publication race control"))
	if _, err := os.Stat(root); err != nil {
		t.Fatal("terminal cleanup deleted a revision held by another publication process", err)
	}
	_, _, problem = o.c.Reconcile()
	fatal(t, problem)
	if _, err := os.Stat(root); err != nil {
		t.Fatal("boot sweep deleted an in-progress publication", err)
	}
	inst := records.PackageInstall{ID: "restored-install", Package: "local/restored", Dir: o.l.InstallDir("restored-install"), Version: "1.0.0"}
	fatal(t, o.store.RecordInstall(inst))
	_, _, problem = o.store.Submit(records.Request{ID: "accepted-new", IdemKey: "accepted-new", BodyDigest: revision,
		Package: "local/restored", InstallID: inst.ID, Entrypoint: "main", Payload: []byte(`{}`), LocalPackageDigest: revision})
	fatal(t, problem)
	must(t, input.Close())
	must(t, child.Wait())
	fatal(t, localpackage.DropDigestUnowned(o.l, o.store, revision))
	if _, err := os.Stat(root); err != nil {
		t.Fatal("accepted request failed to take custody after writer release", err)
	}
}

func TestCaptureReaderHandoffReleasesWriterBeforeAwait(t *testing.T) {
	integration(t)
	if *privateScriptRuntimeWheel == "" {
		t.Skip("requires the exact candidate Runtime wheel")
	}
	root, err := os.MkdirTemp("", "cozy-capture-reader-")
	must(t, err)
	gate, entered := filepath.Join(root, "gate"), filepath.Join(root, "entered")
	t.Cleanup(func() {
		_ = os.Remove(gate)
		path := ""
		for _, value := range childEnv(t, root) {
			if strings.HasPrefix(value, "PATH=") {
				path = strings.TrimPrefix(value, "PATH=")
			}
		}
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("capture reader evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	script := filepath.Join(t.TempDir(), "reader.py")
	must(t, os.WriteFile(script, []byte(fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=%s"]
# [tool.uv.sources]
# cozy-runtime={path=%q}
# ///
from pathlib import Path
import time

def main():
    gate = Path(%q)
    if gate.exists():
        Path(%q).write_text("entered")
        while gate.exists():
            time.sleep(0.02)
`, runtimeFixtureVersion(t, *privateScriptRuntimeWheel), *privateScriptRuntimeWheel, gate, entered)), 0600))
	if code, stdout, stderr := runCozyStreams(t, root, "run", script, "--await", "--json"); code != 0 {
		t.Fatalf("initial capture failed [%d]: %s %s", code, stdout, stderr)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	bad := filepath.Join(root, "bad.json")
	must(t, os.WriteFile(bad, []byte(`[]`), 0600))
	if code, _, _ := runCozyStreams(t, root, "run", script, "--in", bad, "--json"); code == 0 {
		t.Fatal("invalid payload was accepted")
	}
	writer, problem := install.Lock(layout)
	fatal(t, problem)
	writer.Unlock() // validation failure must release the selected capture's reader
	fifo := filepath.Join(root, "payload.fifo")
	must(t, syscall.Mkfifo(fifo, 0600))
	must(t, os.WriteFile(gate, []byte("wait"), 0600))
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, cozyBin, "run", script, "--in", fifo, "--await", "--json")
	command.Env = childEnv(t, root)
	var log bytes.Buffer
	command.Stdout, command.Stderr = &log, &log
	must(t, command.Start())
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	var pipe *os.File
	for pipe == nil {
		pipe, err = os.OpenFile(fifo, os.O_WRONLY|syscall.O_NONBLOCK, 0600)
		if err == nil {
			break
		}
		select {
		case err := <-done:
			t.Fatalf("reader exited before opening payload barrier: %v %s", err, log.String())
		case <-ctx.Done():
			t.Fatal("reader did not reach payload barrier")
		case <-time.After(10 * time.Millisecond):
		}
	}
	defer pipe.Close()
	if other, problem := install.Lock(layout); problem == nil {
		other.Unlock()
		t.Fatal("invocation target released its writer before the request owned it")
	}
	_, err = pipe.Write([]byte(`{}`))
	must(t, err)
	must(t, pipe.Close())
	for {
		if _, err := os.Stat(entered); err == nil {
			break
		}
		select {
		case err := <-done:
			t.Fatalf("request exited before execution barrier: %v %s", err, log.String())
		case <-ctx.Done():
			t.Fatal("request did not enter its execution gate")
		case <-time.After(20 * time.Millisecond):
		}
	}
	writer, problem = install.Lock(layout)
	fatal(t, problem) // --await must no longer own the writer
	active, problem := store.ActiveRequests()
	fatal(t, problem)
	if len(active) != 1 || active[0].InstallID == "" {
		t.Fatal("expected one accepted invocation with its install")
	}
	installID := active[0].InstallID
	_, problem = install.Reclaim(layout, store, installID)
	fatal(t, problem)
	if _, err := os.Stat(layout.InstallDir(installID)); err != nil {
		t.Fatal("request failed to retain its install after reader release", err)
	}
	writer.Unlock()
	must(t, os.Remove(gate))
	if err := <-done; err != nil {
		t.Fatalf("awaited reader did not finish: %v %s", err, log.String())
	}
	t.Log("separate CLI process held writer through payload barrier; durable request owned install while --await released writer")
}
