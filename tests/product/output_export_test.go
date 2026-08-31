package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
)

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
