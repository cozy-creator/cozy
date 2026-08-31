package producttest

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestModelProductionOperationSurvivesRestartAndReplaysExactly(t *testing.T) {
	plan := modelproduction.Plan{
		Destination: "tensorhub/minimax-h3", Release: "1.0.0",
		Source:          "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("a", 40),
		SourceSelection: "sha256:" + strings.Repeat("b", 64),
		SourceFiles: []modelproduction.SourceFile{{
			Member: "transformer/model-00001-of-00002.safetensors",
			SHA256: strings.Repeat("c", 64), Length: 4096,
		}},
		Producer: "tensorhub/h3/four-lane", ProducerRelease: "1.0.0",
		ProducerDigest:   "sha256:" + strings.Repeat("d", 64),
		DescriptorDigest: "sha256:" + strings.Repeat("e", 64),
		Jobs: []modelproduction.JobPin{{Node: "full", Callable: "tensorhub/h3/assemble",
			Release: "1.0.0", ReleaseDigest: "sha256:" + strings.Repeat("f", 64),
			Profile: "torch2.13-cuda13"}},
	}
	data, err := plan.Bytes()
	must(t, err)
	digest, err := plan.Digest()
	must(t, err)
	if bytes.Contains(data, []byte("https://")) || bytes.Contains(data, []byte("token")) ||
		bytes.Contains(data, []byte("upload_grant")) {
		t.Fatalf("restart plan contains transient authority: %s", data)
	}
	path := filepath.Join(t.TempDir(), "records.db")
	store, problem := records.Open(path)
	fatal(t, problem)
	created, replay, problem := store.BeginModelProduction(records.ModelProductionOperation{
		ID: plan.ID(), PlanDigest: digest, Plan: data,
	})
	fatal(t, problem)
	if replay || created.State != "accepted" || created.NodeIndex != 0 {
		t.Fatalf("created operation = %+v replay=%t", created, replay)
	}
	store.Close()

	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	replayed, replay, problem := store.BeginModelProduction(records.ModelProductionOperation{
		ID: plan.ID(), PlanDigest: digest, Plan: data,
	})
	fatal(t, problem)
	if !replay || replayed.CreatedAt != created.CreatedAt || !bytes.Equal(replayed.Plan, data) {
		t.Fatalf("restarted replay = %+v replay=%t", replayed, replay)
	}
	fatal(t, store.AdvanceModelProduction(plan.ID(), "accepted", "source_preparing", 0, "rental-1"))
	fatal(t, store.AdvanceModelProduction(plan.ID(), "source_preparing", "source_prepared", 0, "rental-1"))
	fatal(t, store.AdvanceModelProduction(plan.ID(), "source_prepared", "node_running", 0, "rental-1"))
	fatal(t, store.AdvanceModelProduction(plan.ID(), "node_running", "node_running", 1, "rental-1"))
	current, problem := store.ModelProduction(plan.ID())
	fatal(t, problem)
	if current == nil || current.State != "node_running" || current.NodeIndex != 1 ||
		current.RentalID != "rental-1" {
		t.Fatalf("advanced operation = %+v", current)
	}
	if _, _, problem := store.BeginModelProduction(records.ModelProductionOperation{
		ID: plan.ID(), PlanDigest: digest, Plan: append(data, '\n'),
	}); problem == nil || problem.Name != "model_production.identity_conflict" {
		t.Fatalf("changed replay bytes = %v", problem)
	}
	if problem := store.AdvanceModelProduction(plan.ID(), "node_running", "completed", 1, "rental-1"); problem == nil || problem.Name != "model_production.transition_invalid" {
		t.Fatalf("skipped release/cleanup states = %v", problem)
	}
}
