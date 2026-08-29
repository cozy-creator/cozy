package live

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/remotecontrol"
)

type rentalFixture struct {
	PackageRef      string                   `json:"package_ref"`
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
	facts, problem := remotecontrol.Decode(fixture.ControlSnapshot, fixture.PackageRef)
	if problem != nil {
		t.Fatal(problem)
	}
	if facts.PackageExecutionDigest == "" || facts.PackageDescriptor == nil ||
		facts.Placement.Package != "proof/package" ||
		facts.Placement.PackageReleaseID != "proof/package@v1" ||
		facts.Placement.PackageDescriptorDigest != facts.PackageDescriptor.Digest ||
		facts.Placement.EnvironmentSpecDigest == "" || len(facts.Placement.Bindings) != 1 ||
		facts.Placement.Bindings[0].Entrypoint != "marco" {
		t.Fatalf("decoded current Tensorhub snapshot incompletely: %+v", facts)
	}
}

func TestDecodeRefusesRetiredSnapshotAndBundleShapes(t *testing.T) {
	fixture := currentRental(t)
	var snapshot map[string]any
	if err := json.Unmarshal(fixture.ControlSnapshot.CanonicalBytes, &snapshot); err != nil {
		t.Fatal(err)
	}
	snapshot["evaluated_config"] = snapshot["package_descriptor"]
	raw, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, problem := remotecontrol.Decode(exact(raw), fixture.PackageRef); problem == nil ||
		problem.ErrName() != "rental.control_snapshot_invalid" {
		t.Fatalf("retired snapshot evaluated_config was not refused: %v", problem)
	}

	delete(snapshot, "evaluated_config")
	bundleRaw, err := json.Marshal(snapshot["package_bundle"])
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
	bundle["evaluated_config"] = bundle["package_descriptor"]
	changedBundle, err := json.Marshal(bundle)
	if err != nil {
		t.Fatal(err)
	}
	changedExact := exact(changedBundle)
	snapshot["package_bundle"] = map[string]any{
		"canonical_bytes": changedExact.CanonicalBytes,
		"digest":          changedExact.Digest,
		"length":          changedExact.Length,
	}
	raw, err = json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, problem := remotecontrol.Decode(exact(raw), fixture.PackageRef); problem == nil ||
		problem.ErrName() != "rental.control_snapshot_invalid" {
		t.Fatalf("retired bundle evaluated_config was not refused: %v", problem)
	}

	if err := json.Unmarshal(fixture.ControlSnapshot.CanonicalBytes, &snapshot); err != nil {
		t.Fatal(err)
	}
	bundleRaw, err = json.Marshal(snapshot["package_bundle"])
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(bundleRaw, &bundleDocument); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(bundleDocument.CanonicalBytes, &bundle); err != nil {
		t.Fatal(err)
	}
	sourceTree := bundle["source_tree"]
	delete(bundle, "source_tree")
	bundle["format"] = "tensorhub.package_bundle/2"
	bundle["source_archive"] = sourceTree
	bundle["source_lock"] = map[string]any{"digest": "sha256:" + strings.Repeat("0", 64), "length": 1}
	changedBundle, err = json.Marshal(bundle)
	if err != nil {
		t.Fatal(err)
	}
	changedExact = exact(changedBundle)
	snapshot["package_bundle"] = map[string]any{
		"canonical_bytes": changedExact.CanonicalBytes,
		"digest":          changedExact.Digest,
		"length":          changedExact.Length,
	}
	raw, err = json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, problem := remotecontrol.Decode(exact(raw), fixture.PackageRef); problem == nil ||
		problem.ErrName() != "rental.control_snapshot_invalid" {
		t.Fatalf("retired PackageBundle/2 was not refused: %v", problem)
	}
}
