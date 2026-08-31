package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRemoteModelProductionResolvesHubMetadataWithoutLocalInstall(t *testing.T) {
	producerBytes := []byte(`{"application":"remote_producer:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[],"model_productions":[{"name":"build","nodes":[{"callable":"proof/remote-job/derive","models":{"source":"source"},"name":"derive","outputs":["model"],"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"}}],"outputs":[{"lane_key":"bf16","name":"bf16","required_contract":{"encodings":["sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],"topology_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"source":"derive.model"}],"sources":{"source":"proof/source/1"}}]}`)
	producer, problem := launch.DecodeDescriptor(producerBytes)
	fatal(t, problem)
	jobBytes := []byte(`{"application":"remote_job:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[{"artifact_outputs":[{"max_bytes":4096,"mime_type":"application/vnd.cozy.model-manifest","output_id":"model"}],"models":[{"class":"ProofModel","component_use":{},"path":"derive.models.source","stamps":{}}],"name":"derive","publishes":false,"request":{"fields":[]},"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"},"result":{"fields":[]}}],"model_productions":[]}`)
	job, problem := launch.DecodeDescriptor(jobBytes)
	fatal(t, problem)
	type fixture struct {
		release, digest string
		document        json.RawMessage
		descriptor      *launch.PackageDescriptor
	}
	newFixture := func(release string, descriptor *launch.PackageDescriptor) fixture {
		document := json.RawMessage(`{"format":"PackageRelease/1","release":"` + release + `"}`)
		return fixture{release: release, digest: "sha256:" + strings.Repeat(release[:1], 64), document: document,
			descriptor: descriptor}
	}
	producerRelease, jobRelease := newFixture("2.0.0", producer), newFixture("3.0.0", job)
	writeRelease := func(w http.ResponseWriter, selected fixture) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"release": map[string]any{
				"release": selected.release, "release_digest": selected.digest,
				"package_descriptor_digest": selected.descriptor.Digest,
				"package_descriptor_length": len(selected.descriptor.Raw),
				"created_at":                "2026-08-31T00:00:00Z",
			},
			"document":           selected.document,
			"package_descriptor": json.RawMessage(selected.descriptor.Raw),
		})
	}
	requests := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, r.Method+" "+r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models/resolve":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model": "acme/input", "release": "1.0.0", "lane": "bf16",
				"manifest_id":   "sha256:" + strings.Repeat("d", 64),
				"header_digest": "sha256:" + strings.Repeat("e", 64),
				"objects":       1, "bytes": 4096,
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/remote-producer":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"package": map[string]any{"org": "proof", "name": "remote-producer",
					"created_at": "2026-08-31T00:00:00Z"},
				"releases": []map[string]any{
					{"release": "1.0.0", "cut_at": "2026-08-31T00:00:00Z"},
					{"release": "9.0.0", "cut_at": "2026-08-31T00:00:00Z", "yanked": true},
					{"release": "2.0.0", "cut_at": "2026-08-31T00:00:00Z"},
				},
			})
		case r.Method == http.MethodGet &&
			r.URL.Path == "/v1/packages/proof/remote-producer/releases/2.0.0":
			writeRelease(w, producerRelease)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/remote-job":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"package": map[string]any{"org": "proof", "name": "remote-job",
					"created_at": "2026-08-31T00:00:00Z"},
				"releases": []map[string]any{
					{"release": "2.0.0", "cut_at": "2026-08-31T00:00:00Z"},
					{"release": "3.0.0", "cut_at": "2026-08-31T00:00:00Z"},
				},
			})
		case r.Method == http.MethodGet &&
			r.URL.Path == "/v1/packages/proof/remote-job/releases/3.0.0":
			writeRelease(w, jobRelease)
		default:
			http.Error(w, "unexpected route", http.StatusNotFound)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte(
		"tensorhub_url: "+server.URL+"\nrentals:\n  max_hourly_spend_usd: 10\n"), 0o600))
	code, out := runCozyDir(t, root, "", []string{"PATH=/usr/bin:/bin"},
		"--json", "--full", "model", "publish", "acme/output", "acme/input@1.0.0",
		"--release", "1.0.0", "--producer", "proof/remote-producer/build",
		"--rental", "--dry-run")
	if code != 0 || !strings.Contains(out, `"status":"planned"`) ||
		!strings.Contains(out, `"producer":"proof/remote-producer/build@2.0.0"`) {
		t.Fatalf("metadata-only remote production [exit %d]\n%s", code, out)
	}
	if got := strings.Join(requests, "\n"); strings.Contains(got, "/download") ||
		got != "GET /v1/models/resolve\nGET /v1/packages/proof/remote-producer\n"+
			"GET /v1/packages/proof/remote-producer/releases/2.0.0\n"+
			"GET /v1/packages/proof/remote-job\n"+
			"GET /v1/packages/proof/remote-job/releases/3.0.0" {
		t.Fatalf("remote production metadata routes =\n%s", got)
	}
	requests = nil
	code, out = runCozyDir(t, root, "", []string{"PATH=/usr/bin:/bin"},
		"model", "publish", "acme/output", "acme/input@1.0.0", "--release", "1.0.0",
		"--producer", "proof/remote-producer/build", "--dry-run")
	if code == 0 || !strings.Contains(out, "not installed") {
		t.Fatalf("local production stopped requiring a local install [exit %d]\n%s", code, out)
	}
	if got := strings.Join(requests, "\n"); got != "GET /v1/models/resolve" {
		t.Fatalf("local production unexpectedly resolved remote package metadata:\n%s", got)
	}
}

func TestModelProductionOperationSurvivesRestartAndReplaysExactly(t *testing.T) {
	plan := modelproduction.Plan{
		Destination: "tensorhub/minimax-h3", Release: "1.0.0",
		Source:          "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("a", 40),
		SourceSelection: "sha256:" + strings.Repeat("b", 64),
		SourceFiles: []modelproduction.SourceFile{{
			Member: "transformer/model-00001-of-00002.safetensors",
			SHA256: strings.Repeat("c", 64), Length: 4096,
		}},
		Producer: "tensorhub/minimax-h3-tools/four-lane", ProducerRelease: "1.0.0",
		ProducerDigest:   "sha256:" + strings.Repeat("d", 64),
		DescriptorDigest: "sha256:" + strings.Repeat("e", 64),
		Jobs: []modelproduction.JobPin{{Node: "full", Callable: "tensorhub/minimax-h3-tools/assemble",
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
	fatal(t, store.CancelModelProduction(plan.ID(), "node_running", "user interrupted"))
	canceled, problem := store.ModelProduction(plan.ID())
	fatal(t, problem)
	if canceled == nil || canceled.State != "canceled" || !canceled.CancelRequested {
		t.Fatalf("canceled production = %+v", canceled)
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
