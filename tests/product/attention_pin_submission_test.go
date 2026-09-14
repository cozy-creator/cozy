package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Real CLI -> HTTP daemon -> SQLite, without a GPU. A parser-only test missed two
// copies dropping the kernel pin before a paid H3 request reached its worker.
func TestAttentionPinSurvivesSubmissionAndConflictingReplayRefuses(t *testing.T) {
	root := t.TempDir()
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t), "--editable"); code != 0 {
		t.Fatalf("install %d: %s", code, out)
	}
	args := []string{"--json", "run", localWeightlessRef + "/echo", "why=attention-pin", "--attention-kernel=flash-attn3-fp8", "--idempotency-key=attention-pin"}
	// This fixture needs the real admission path, not a GPU executor.
	_, first := runCozy(t, root, args...)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey("attention-pin")
	fatal(t, problem)
	if request == nil || request.AttentionKernel != "flash-attn3-fp8" {
		t.Fatalf("durable request lost pin: %+v; CLI %s", request, first)
	}
	t.Logf("recorded release=%q install=%q local_revision=%q", request.Release, request.InstallID, request.LocalPackageDigest)
	// Same request is idempotent; changing only the kernel must never replay it.
	_, replay := runCozy(t, root, args...)
	if strings.Contains(replay, "different body") {
		t.Fatalf("same pin conflicts: %s", replay)
	}
	args[4] = "--attention-kernel=flash-attn3"
	if code, out := runCozy(t, root, args...); code == 0 || !strings.Contains(out, "different body") {
		t.Fatalf("changed pin replayed: %s", out)
	}
	again, problem := store.RequestByIdempotencyKey("attention-pin")
	fatal(t, problem)
	if again.ID != request.ID || again.AttentionKernel != request.AttentionKernel {
		t.Fatalf("conflict mutated original: %+v", again)
	}
}
