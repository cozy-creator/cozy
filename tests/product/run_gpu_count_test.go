package producttest

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// The ordinary CLI must carry the flag through the API and durable request row.
// This is admission/custody coverage, not GPU inference qualification.
func TestRunGPUCountSurvivesCLIAdmissionAndConflictingReplay(t *testing.T) {
	root := t.TempDir()
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t), "--editable"); code != 0 {
		t.Fatalf("install %d: %s", code, out)
	}
	args := []string{"--json", "run", localWeightlessRef + "/echo", "why=gpu-count", "--gpus=2", "--idempotency-key=two-gpus"}
	_, out := runCozy(t, root, args...)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey("two-gpus")
	fatal(t, problem)
	if request == nil || request.GPUs != 2 {
		t.Fatalf("lost GPU count: %+v; %s", request, out)
	}
	args[4] = "--gpus=1"
	if code, changed := runCozy(t, root, args...); code == 0 || !strings.Contains(changed, "different body") {
		t.Fatalf("changed GPU count replayed: %d %s", code, changed)
	}
	again, problem := store.RequestByIdempotencyKey("two-gpus")
	fatal(t, problem)
	if again.ID != request.ID || again.GPUs != 2 {
		t.Fatalf("conflict changed request: %+v", again)
	}
}

func TestRunGPUCountRejectsInvalidFlagsBeforeAdmission(t *testing.T) {
	root := t.TempDir()
	for _, invalid := range []string{"0", "-1", "1.5", "4294967296"} {
		code, out := runCozy(t, root, "run", "unused/package/call", "--gpus="+invalid, "--json")
		if code == 0 || !strings.Contains(out, "gpus") {
			t.Fatalf("invalid %q accepted: %d %s", invalid, code, out)
		}
	}
}

func TestGPUCountColumnIsAddedWithoutChangingRecordedRuns(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	request := records.Request{ID: "old", IdemKey: "old", BodyDigest: childDigest("a"), Package: "proof/pkg", Entrypoint: "call", Payload: []byte(`{}`)}
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = db.Exec(`ALTER TABLE requests DROP COLUMN gpus`)
	must(t, err)
	must(t, db.Close())
	store, problem = records.Open(path)
	fatal(t, problem)
	old, problem := store.RequestRow("old")
	fatal(t, problem)
	if old == nil || old.GPUs != 0 || old.BodyDigest != request.BodyDigest {
		t.Fatalf("old row changed: %+v", old)
	}
	request.ID, request.IdemKey, request.GPUs = "new", "new", 2
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	reloaded, problem := store.RequestRow("new")
	fatal(t, problem)
	if reloaded.GPUs != 2 {
		t.Fatalf("GPU count lost on restart: %+v", reloaded)
	}
}

func TestGPUCountParticipatesInOperationIdentity(t *testing.T) {
	request := records.Request{ChildTargetDigest: childDigest("a"), Payload: []byte(`{"seed":17}`)}
	keys := map[string]bool{}
	for _, count := range []uint32{0, 1, 2} {
		request.GPUs = count
		key, problem := records.OperationKey(request)
		fatal(t, problem)
		if keys[key] {
			t.Fatalf("GPU count %d reused another execution configuration", count)
		}
		keys[key] = true
	}
}
