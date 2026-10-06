package producttest

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
)

func servingPlacementFixture(t *testing.T) ([]byte, []byte) {
	t.Helper()
	entry := map[string]canonical.Value{"name": "generate", "slots": []canonical.Value{}}
	identity, err := canonical.Write(entry)
	must(t, err)
	binding := assessmentDigest(identity)
	entry["entrypoint_binding_digest"] = binding
	models := []canonical.Value{map[string]canonical.Value{"id": "candidate", "manifest": map[string]canonical.Value{"digest": "sha256:" + strings.Repeat("c", 64), "length": int64(10)}}}
	entries := []canonical.Value{entry}
	identity, err = canonical.Write(map[string]canonical.Value{"entrypoints": entries, "models": models})
	must(t, err)
	bindings := assessmentDigest(identity)
	set, err := canonical.Write(map[string]canonical.Value{
		"format": "cozy.worker.v1.PlacementSet/1",
		"placements": []canonical.Value{map[string]canonical.Value{
			"placement_id": "prepared", "package": map[string]canonical.Value{"package": "proof/model", "release": "1.0.0"},
			"bindings_digest": bindings, "entrypoints": entries, "models": models,
		}},
	})
	must(t, err)
	invocation := assessmentDocument(t, map[string]any{"serving": map[string]any{"entrypoint_binding_digest": binding, "attempt_binding_id": binding, "bindings_digest": bindings, "_format": "cozy.worker.v1.ServingInvocationSpec/1"}, "_format": "cozy.worker.v1.InvocationSpec/1"})
	return set, invocation
}

func TestServingPlacementEvidenceIsAtomicAndSurvivesReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "instance", Package: "proof/model", WorkerID: "worker", Devices: []string{"cpu"}}))
	set, invocation := servingPlacementFixture(t)
	request, _, problem := store.Submit(records.Request{ID: "render", IdemKey: "render", BodyDigest: assessmentDigest([]byte("render")), Package: "proof/model", Entrypoint: "generate", Payload: []byte(`{}`)})
	fatal(t, problem)
	a := records.Attempt{RequestID: request.ID, InstanceID: "instance", SessionID: "boot", InvocationDigest: assessmentDigest(invocation), InvocationCanonical: invocation}
	if _, problem := store.Dispatch(a); problem == nil {
		t.Fatal("new serving binding accepted without exact prepared bytes")
	}
	before, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if before.State != "submitted" || before.Ordinal != 0 {
		t.Fatal("refusal advanced the request or minted an attempt")
	}
	for _, changed := range [][]byte{
		append([]byte(" "), set...),
		bytes.Replace(set, []byte(strings.Repeat("c", 64)), []byte(strings.Repeat("d", 64)), 1),
		bytes.Replace(set, []byte("proof/model"), []byte("other/model"), 1),
		bytes.Replace(set, []byte("generate"), []byte("unbound"), 1),
	} {
		a.ServingPlacementSet = changed
		if _, problem := store.Dispatch(a); problem == nil {
			t.Fatal("changed prepared binding accepted")
		}
	}
	a.ServingPlacementSet = set
	ordinal, problem := store.Dispatch(a)
	fatal(t, problem)
	if ordinal != 1 {
		t.Fatal("refused evidence consumed an ordinal")
	}
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	retained, problem := store.AttemptRow(request.ID, ordinal)
	fatal(t, problem)
	if retained == nil || !bytes.Equal(retained.ServingPlacementSet, set) || !bytes.Equal(retained.InvocationCanonical, invocation) {
		t.Fatal("exact placement bytes did not survive reopen")
	}
	placement, problem := records.BoundServingPlacement(request, *retained)
	fatal(t, problem)
	if placement.Str("placement_id") != "prepared" || placement.List("models")[0].Sub("manifest").Str("digest") != "sha256:"+strings.Repeat("c", 64) {
		t.Fatal("retained placement changed model bindings")
	}
	request.Package = "other/model"
	if _, problem := records.BoundServingPlacement(request, *retained); problem == nil {
		t.Fatal("another package borrowed the retained set")
	}
}
