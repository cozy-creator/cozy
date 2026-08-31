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
		}},
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

func TestModelProductionJoinsNodeArtifactAndTransferBeforeReplay(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	defer store.Close()
	plan := modelproduction.Plan{Destination: "acme/model", Release: "1.0.0",
		Source:          "hf://acme/model@" + strings.Repeat("a", 40),
		SourceSelection: "sha256:" + strings.Repeat("b", 64)}
	data, err := plan.Bytes()
	must(t, err)
	digest, err := plan.Digest()
	must(t, err)
	_, _, problem = store.BeginModelProduction(records.ModelProductionOperation{
		ID: plan.ID(), PlanDigest: digest, Plan: data,
	})
	fatal(t, problem)
	_, problem = store.BeginModelProductionNode(records.ModelProductionNode{
		OperationID: plan.ID(), NodeIndex: 0, NodeName: "derive", State: "pending",
	})
	fatal(t, problem)
	fatal(t, store.SetModelProductionNodeRequest(plan.ID(), 0, "derive", "job-1", "submitted"))
	node, problem := store.ModelProductionNodeByRequest("job-1")
	fatal(t, problem)
	if node == nil || node.OperationID != plan.ID() || node.NodeName != "derive" {
		t.Fatalf("node request join = %+v", node)
	}
	receipt := []byte(`{"format":"proof/1"}`)
	evidence := []byte(`{"format":"evidence/1"}`)
	artifact := records.ModelProductionArtifact{OperationID: plan.ID(), NodeName: "derive",
		OutputSlot: "model", RequestID: "job-1", Attempt: 1,
		InvocationDigest: "sha256:" + strings.Repeat("c", 64), TransactionID: "txn-1",
		WriterGeneration: 1, ReceiptDigest: "sha256:" + strings.Repeat("d", 64),
		Receipt: receipt, ManifestID: "sha256:" + strings.Repeat("e", 64),
		ManifestLength: 128, ReleaseEvidence: evidence}
	object := records.ModelProductionObject{OperationID: plan.ID(), NodeName: "derive",
		OutputSlot: "model", ObjectID: "sha256:" + strings.Repeat("f", 64),
		Length: 256, SourceRef: "opaque-source"}
	fatal(t, store.RecordModelProductionArtifact(artifact, []records.ModelProductionObject{object}))
	changedObject := object
	changedObject.SourceRef = "other-source"
	if problem := store.RecordModelProductionArtifact(artifact,
		[]records.ModelProductionObject{changedObject}); problem == nil ||
		problem.Name != "model_production.artifact_conflict" {
		t.Fatalf("changed object inventory replay = %v", problem)
	}
	fatal(t, store.RecordModelProductionObjectStatus(records.ModelProductionObject{
		OperationID: plan.ID(), NodeName: "derive", OutputSlot: "model",
		ObjectID: object.ObjectID, Length: object.Length, TransferOperationID: "pub-1",
		GrantRevision: 1, UpdateSequence: 1, State: "uploaded", TransferredBytes: object.Length,
	}))
	fatal(t, store.MarkModelProductionArtifactPublished(plan.ID(), "derive", "model", "publish-1"))
	artifacts, problem := store.ModelProductionArtifacts(plan.ID())
	fatal(t, problem)
	objects, problem := store.ModelProductionObjects(plan.ID(), "derive", "model")
	fatal(t, problem)
	if len(artifacts) != 1 || artifacts[0].PublicationID != "publish-1" ||
		artifacts[0].State != "prepared" || len(objects) != 1 || objects[0].State != "uploaded" ||
		objects[0].TransferOperationID != "pub-1" {
		t.Fatalf("durable artifact join = artifacts=%+v objects=%+v", artifacts, objects)
	}
	if problem := store.RecordModelProductionObjectStatus(records.ModelProductionObject{
		OperationID: plan.ID(), NodeName: "derive", OutputSlot: "model",
		ObjectID: object.ObjectID, Length: object.Length, TransferOperationID: "other",
		GrantRevision: 1, UpdateSequence: 2, State: "uploaded", TransferredBytes: object.Length,
	}); problem == nil || problem.Name != "model_production.artifact_status_conflict" {
		t.Fatalf("changed transfer operation replay = %v", problem)
	}
}
