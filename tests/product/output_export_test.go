package producttest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
)

func TestAwaitedRunReportsAttemptZeroFailureWithoutWaitingForOutputExport(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "await-attempt-zero-output")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})

	project := weightlessProject(t)
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("local package install [exit %d]\n%s", code, out)
	}
	install := activePackageInstall(t, root)
	runtime := launch.Binary(install) //cozy:allow product red arm makes the selected worker executable unstartable
	info, err := os.Stat(runtime)
	must(t, err)
	// The selected generation stays structurally valid, but its worker process cannot
	// start. This is the live failure class: the request has an --out intent, reaches a
	// request.failed terminal while still at attempt zero, and has no bytes to publish.
	must(t, os.Chmod(runtime, info.Mode().Perm()&^0o111))

	destination := filepath.Join(root, "never-published")
	commandCtx, stop := context.WithTimeout(context.Background(), 10*time.Second)
	defer stop()
	cmd := exec.CommandContext(commandCtx, "/usr/bin/nice", "-n", "19", cozyBin,
		"run", localWeightlessRef+"/tile", "size=32", "--out", destination, "--await")
	cmd.Env = childEnv(t, root)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	began := time.Now()
	_ = cmd.Run()
	if commandCtx.Err() != nil {
		t.Fatalf("--await outlived its request.failed terminal:\nstdout:\n%s\nstderr:\n%s",
			stdout.String(), stderr.String())
	}
	if code := cmd.ProcessState.ExitCode(); code != 1 {
		t.Fatalf("attempt-zero failure exited %d, want operational 1\nstdout:\n%s\nstderr:\n%s",
			code, stdout.String(), stderr.String())
	}
	if took := time.Since(began); took > 5*time.Second {
		t.Fatalf("attempt-zero terminal took %s to reach --await", took)
	}
	answer := stdout.String() + stderr.String()
	if !strings.Contains(answer, "ended failed") ||
		!strings.Contains(answer, "cannot start the package worker") {
		t.Fatalf("attempt-zero failure was not explained\nstdout:\n%s\nstderr:\n%s",
			stdout.String(), stderr.String())
	}
	for _, noise := range []string{
		"cancel requested", "the attempt's own terminal still settles it", "second interrupt",
	} {
		if strings.Contains(answer, noise) {
			t.Fatalf("ordinary failure emitted cancellation noise %q\n%s", noise, answer)
		}
	}

	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	defer store.Close()
	requests, problem := store.RequestsOfKind("serving", "", 10)
	fatal(t, problem)
	if len(requests) != 1 || requests[0].State != "failed" {
		t.Fatalf("attempt-zero request state = %#v", requests)
	}
	request := requests[0]
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatalf("worker-start failure minted attempts: %#v", attempts)
	}
	events, problem := store.EventsAfter(request.ID, 0, 100)
	fatal(t, problem)
	if len(events) == 0 || events[len(events)-1].Type != "request.failed" {
		t.Fatalf("attempt-zero stream has no failed terminal: %#v", events)
	}
	export, problem := store.OutputExportOf(request.ID)
	fatal(t, problem)
	if export == nil || export.State != "skipped" ||
		!strings.Contains(export.SafeError, "published no successful result") {
		t.Fatalf("failed request left its output export unsettled: %#v", export)
	}
}

func TestOutputExportResumesFromTerminalAfterOwnerRestart(t *testing.T) {
	root := t.TempDir()
	database := filepath.Join(root, "records.db")
	destination := filepath.Join(root, "results")
	source := filepath.Join(root, "accepted.webp")
	content := []byte("RIFF\x10\x00\x00\x00WEBPdaemon-owned-test")
	must(t, os.WriteFile(source, content, 0o600))
	contentDigest := sha256.Sum256(content)
	payloadDigest := sha256.Sum256([]byte(`{"prompt":"restart proof"}`))
	payloadHash := hex.EncodeToString(payloadDigest[:])
	filename := payloadHash + ".webp"

	store, problem := records.Open(database)
	fatal(t, problem)
	request := records.Request{
		ID: "req-output-restart", IdemKey: "output-restart", BodyDigest: "body",
		Package: "cozy/export-proof", Entrypoint: "render", Payload: []byte(`{}`),
		Outputs: "image",
		OutputExport: &records.OutputExportIntent{
			Directory: destination, PayloadHash: payloadHash,
			Outputs: []records.OutputExportEntry{{
				OutputID: "image", MediaType: "image/webp", Filename: filename,
			}},
		},
	}
	_, fresh, problem := store.Submit(request)
	if problem != nil || !fresh {
		t.Fatalf("recording export request: fresh=%v, problem=%v", fresh, problem)
	}
	fatal(t, store.AttachWorker(records.WorkerProcess{
		InstanceID: "worker-output-restart", Package: request.Package, WorkerID: "worker",
	}))
	attempt, problem := store.Dispatch(records.Attempt{
		RequestID: request.ID, InstanceID: "worker-output-restart", SessionID: "boot",
		InvocationDigest: "invocation", InvocationCanonical: []byte(`{}`),
	})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(request.ID, attempt, "boot"))
	applied, problem := store.AcceptTerminal(records.Terminal{
		RequestID: request.ID, Attempt: attempt, SessionID: "boot",
		InvocationDigest: "invocation", TerminalID: "outcome", TerminalDigest: "terminal",
		Status: "SUCCEEDED", RequestState: "succeeded",
		Outputs: []records.Output{{
			OutputID: "image", MediaID: "media-output-restart", Path: source,
			Digest: "sha256:" + hex.EncodeToString(contentDigest[:]),
			Length: int64(len(content)), MimeType: "image/webp",
		}},
	})
	if problem != nil || !applied {
		t.Fatalf("recording terminal: applied=%v, problem=%v", applied, problem)
	}
	store.Close() // terminal committed; publication has not started

	store, problem = records.Open(database)
	fatal(t, problem)
	owner, problem := orchestrator.Open(orchestrator.Options{Store: store})
	fatal(t, problem)
	fatal(t, owner.ResumeOutputExports())
	owner.Close(0)

	published, err := os.ReadFile(filepath.Join(destination, filename))
	must(t, err)
	if !bytes.Equal(published, content) {
		t.Fatalf("published bytes changed across owner restart: %x", published)
	}
	export, problem := store.OutputExportOf(request.ID)
	fatal(t, problem)
	if export == nil || export.State != "published" || export.Attempts != 1 ||
		len(export.PublishedPaths) != 1 || export.PublishedPaths[0] != filepath.Join(destination, filename) {
		t.Fatalf("settled export = %#v", export)
	}
	store.Close()

	store, problem = records.Open(database)
	fatal(t, problem)
	defer store.Close()
	owner, problem = orchestrator.Open(orchestrator.Options{Store: store})
	fatal(t, problem)
	defer owner.Close(0)
	fatal(t, owner.ResumeOutputExports())
	export, problem = store.OutputExportOf(request.ID)
	fatal(t, problem)
	if export.Attempts != 1 {
		t.Fatalf("published export replayed after another restart: %#v", export)
	}
}

func TestOutputExportNeverOverwritesDifferentBytes(t *testing.T) {
	root := t.TempDir()
	destination := filepath.Join(root, "results")
	filename := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.webp"
	publish := func(name string, content []byte) *records.Output {
		t.Helper()
		path := filepath.Join(root, name)
		must(t, os.WriteFile(path, content, 0o600))
		digest := sha256.Sum256(content)
		return &records.Output{
			OutputID: "image", Path: path, Digest: "sha256:" + hex.EncodeToString(digest[:]),
			Length: int64(len(content)), MimeType: "image/webp",
		}
	}
	first := publish("first.webp", []byte("first"))
	paths, problem := resultfiles.Publish(destination, []resultfiles.Entry{{
		OutputID: first.OutputID, MediaType: first.MimeType, Filename: filename,
		Source: first.Path, Digest: first.Digest, Length: first.Length,
	}})
	if problem != nil || len(paths) != 1 {
		t.Fatalf("first publish = %v, %v", paths, problem)
	}
	second := publish("second.webp", []byte("second"))
	_, problem = resultfiles.Publish(destination, []resultfiles.Entry{{
		OutputID: second.OutputID, MediaType: second.MimeType, Filename: filename,
		Source: second.Path, Digest: second.Digest, Length: second.Length,
	}})
	if problem == nil || problem.ErrName() != "output_export_destination_conflict" {
		t.Fatalf("different destination bytes were not refused: %v", problem)
	}
	got, err := os.ReadFile(paths[0])
	must(t, err)
	if !bytes.Equal(got, []byte("first")) {
		t.Fatalf("conflict replaced existing output: %q", got)
	}
}
