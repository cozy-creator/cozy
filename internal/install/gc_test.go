package install

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

const gcDigest = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"

func TestCollectDoesNotDeleteNewlyWorkflowPinnedGeneration(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	_, problem = store.Activate(records.EndpointInstall{
		ID: "generation-race", Endpoint: "org/ep", Major: 1, Version: "1.0.0",
		SourceKind: "dir", SourceRef: ".", SourceDigest: gcDigest,
		Dir: layout.GenerationDir("generation-race"), Python: "python", UV: "uv",
		LockDigest: gcDigest, Platform: "test", LinkMode: "copy", Closure: "none",
		Descriptor: gcDigest, BytesExcl: 4,
	})
	if problem != nil {
		t.Fatal(problem)
	}
	if err := os.MkdirAll(layout.GenerationDir("generation-race"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(layout.GenerationDir("generation-race"), "kept"),
		[]byte("kept"), 0o600); err != nil {
		t.Fatal(err)
	}
	if problem := store.Unpin("org/ep", 1); problem != nil {
		t.Fatal(problem)
	}
	plan, problem := Plan(layout, store)
	if problem != nil || len(plan) != 1 {
		t.Fatalf("plan=%#v problem=%v", plan, problem)
	}
	_, _, problem = store.CreateWorkflow(records.WorkflowExecution{
		ID: "wfl-race", IdemKey: "gc-race", BodyDigest: gcDigest,
		ExecutionDigest: gcDigest, CreativePlanDigest: gcDigest, Plan: []byte(`{}`),
	}, nil, map[int]string{1: "generation-race"}, nil, 1)
	if problem != nil {
		t.Fatal(problem)
	}
	freed, problem := Collect(layout, store, plan)
	if problem != nil || freed != 0 {
		t.Fatalf("freed=%d problem=%v", freed, problem)
	}
	if _, err := os.Stat(filepath.Join(layout.GenerationDir("generation-race"), "kept")); err != nil {
		t.Fatalf("workflow-pinned bytes were removed: %v", err)
	}
	if generation, problem := store.Install("generation-race"); problem != nil || generation == nil {
		t.Fatalf("workflow-pinned row missing: %#v %v", generation, problem)
	}
}
