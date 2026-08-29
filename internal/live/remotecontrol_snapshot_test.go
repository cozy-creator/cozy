package live

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/remotecontrol"
)

type rentalFixture struct {
	EndpointRef     string                   `json:"endpoint_ref"`
	ControlSnapshot hub.ExactControlDocument `json:"control_snapshot"`
}

func currentRental(t *testing.T) rentalFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/tensorhub-current-rental.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture rentalFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func exact(raw []byte) hub.ExactControlDocument {
	digest := sha256.Sum256(raw)
	return hub.ExactControlDocument{
		CanonicalBytes: raw,
		Digest:         "sha256:" + hex.EncodeToString(digest[:]),
		Length:         int64(len(raw)),
	}
}

func TestDecodeCurrentTensorhubRentalSnapshot(t *testing.T) {
	fixture := currentRental(t)
	facts, problem := remotecontrol.Decode(fixture.ControlSnapshot, fixture.EndpointRef)
	if problem != nil {
		t.Fatal(problem)
	}
	if facts.EndpointExecutionDigest != "sha256:a7b7a2395f5f6759a256631b2692c95479d7bd1012e9d16ba470af012bfc9c8f" ||
		facts.Placement.Endpoint != "cozy/marco-polo-derived" ||
		facts.Placement.ReleaseID != "cozy/marco-polo-derived@derived-portable-1" ||
		facts.Placement.EnvironmentSpecDigest == "" || len(facts.Placement.Bindings) != 1 ||
		facts.Placement.Bindings[0].Entrypoint != "marco" {
		t.Fatalf("decoded current Tensorhub snapshot incompletely: %+v", facts)
	}
}

func TestDecodeRefusesRetiredEvaluatedConfigFields(t *testing.T) {
	fixture := currentRental(t)
	var snapshot map[string]any
	if err := json.Unmarshal(fixture.ControlSnapshot.CanonicalBytes, &snapshot); err != nil {
		t.Fatal(err)
	}
	snapshot["evaluated_config"] = snapshot["descriptor"]
	raw, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, problem := remotecontrol.Decode(exact(raw), fixture.EndpointRef); problem == nil ||
		problem.ErrName() != "rental.control_snapshot_invalid" {
		t.Fatalf("retired snapshot evaluated_config was not refused: %v", problem)
	}

	delete(snapshot, "evaluated_config")
	bundleRaw, err := json.Marshal(snapshot["endpoint_bundle"])
	if err != nil {
		t.Fatal(err)
	}
	var bundleDocument hub.ExactControlDocument
	if err := json.Unmarshal(bundleRaw, &bundleDocument); err != nil {
		t.Fatal(err)
	}
	var bundle map[string]any
	if err := json.Unmarshal(bundleDocument.CanonicalBytes, &bundle); err != nil {
		t.Fatal(err)
	}
	bundle["evaluated_config"] = bundle["descriptor"]
	changedBundle, err := json.Marshal(bundle)
	if err != nil {
		t.Fatal(err)
	}
	changedExact := exact(changedBundle)
	snapshot["endpoint_bundle"] = map[string]any{
		"canonical_bytes": changedExact.CanonicalBytes,
		"digest":          changedExact.Digest,
		"length":          changedExact.Length,
	}
	raw, err = json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, problem := remotecontrol.Decode(exact(raw), fixture.EndpointRef); problem == nil ||
		problem.ErrName() != "rental.control_snapshot_invalid" {
		t.Fatalf("retired bundle evaluated_config was not refused: %v", problem)
	}
}
