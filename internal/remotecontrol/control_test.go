package remotecontrol

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/hub"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

func TestExactAttemptControlSurvivesWithoutALocalInstall(t *testing.T) {
	control := controlFixture(t)
	facts, e := Decode(control, "acme/h3/v1/generate")
	if e != nil {
		t.Fatal(e)
	}
	p := facts.Placement
	if p.EnvironmentSpecDigest == "" || p.ConfigDigest == "" ||
		p.InstalledEnvironmentReceiptDigest == "" || p.PlacementID() != "ra-test" ||
		len(p.Bindings) != 1 {
		t.Fatalf("incomplete exact projection: %#v", p)
	}
	id, raw, e := p.Bindings[0].Staged()
	if e != nil || id != p.Bindings[0].RuntimePlan.Digest ||
		string(raw) != string(p.Bindings[0].RuntimePlan.CanonicalBytes) {
		t.Fatalf("plan was not relayed byte-for-byte: id=%q bytes=%q err=%v", id, raw, e)
	}
	entrypoint, descriptorErr := facts.Descriptor.Function("generate")
	if string(p.ExactPlacementSetBytes) == "" || entrypoint == nil || descriptorErr != nil {
		t.Fatal("exact PlacementSet or remote descriptor was lost")
	}
}

func TestAReRenderedPlanCannotEnterThePersistedSnapshot(t *testing.T) {
	control := controlFixture(t)
	var s snapshot
	if err := json.Unmarshal(control.CanonicalBytes, &s); err != nil {
		t.Fatal(err)
	}
	// A locally rendered byte string with a self-consistent new identity still
	// disagrees with the attempt's frozen PlacementSet and binding release.
	s.BindingDocuments[0].CanonicalBytes = []byte(`{"bindings":[],"descriptor":{"digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","length":1},"entrypoint":"generate","format":"cozy.endpoint.EntrypointBindingPlan/1"}`)
	changed := exact(s.BindingDocuments[0].CanonicalBytes)
	s.BindingDocuments[0].Digest, s.BindingDocuments[0].Length = changed.Digest, changed.Length
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if _, e := Decode(exact(raw), "acme/h3/v1/generate"); e == nil {
		t.Fatal("changed locally rendered plan entered the attempt-bound control closure")
	}
}

func controlFixture(t *testing.T) hub.ExactControlDocument {
	t.Helper()
	descriptor := exact(mustPrettyJSON(t, map[string]any{"entrypoints": []any{map[string]any{
		"hidden": false, "kind": "entrypoint", "models": []any{}, "name": "generate",
		"request": map[string]any{"fields": []any{}, "struct": "Request"},
		"result": map[string]any{"fields": []any{map[string]any{
			"name": "video", "type": map[string]any{"asset": "video"}, "wire": "asset",
		}}, "struct": "Result"},
	}}}))
	evaluatedConfig := exact([]byte(`{}`))
	plan := exact(mustJSON(t, map[string]any{
		"bindings": []any{}, "descriptor": refMap(descriptor), "entrypoint": "generate", "format": planFormat,
	}))
	bindingRelease := exact(mustJSON(t, map[string]any{
		"descriptor":                    refMap(descriptor),
		"documents":                     []any{map[string]any{"kind": "entrypoint_binding_plan", "ref": refMap(plan)}},
		"endpoint_execution_digest":     digestOf([]byte("execution")),
		"environment_spec":              map[string]any{}, // filled below
		"format":                        bindingFormat,
		"installed_environment_receipt": map[string]any{}, // filled below
		"object_set_digest":             digestOf([]byte("objects")),
	}))
	_ = bindingRelease

	endpointBundle := exact(mustJSON(t, map[string]any{
		"descriptor": refMap(descriptor), "endpoint_release_id": "acme/h3@v1",
		"evaluated_config": refMap(evaluatedConfig), "format": bundleFormat,
		"project_wheel": map[string]any{}, "resolved_wheel_set": refMap(exact([]byte(`{"resolved":true}`))),
		"wheelhouse_manifest": refMap(exact([]byte(`{"wheelhouse":true}`))),
	}))
	projectDigest, wheelhouseDigest := digestOf([]byte("project-wheel")), digestOf([]byte("wheelhouse"))
	environmentBytes, _, err := canonical.Identity(&pb.EndpointEnvironmentSpec{
		PlatformTarget: &pb.PlatformTarget{OsArch: "linux/amd64", Libc: "glibc2.39",
			PythonAbi: "cp312", AcceleratorBackend: "cuda", AcceleratorAbi: "cu129"},
		EndpointBundleDigest:     mustRaw(t, endpointBundle.Digest),
		ProjectWheelDigest:       mustRaw(t, projectDigest),
		WheelhouseManifestDigest: mustRaw(t, wheelhouseDigest),
	})
	if err != nil {
		t.Fatal(err)
	}
	environment := exact(environmentBytes)
	receipt := exact(mustJSON(t, map[string]any{
		"distributions": []any{}, "endpoint_bundle_digest": endpointBundle.Digest,
		"environment_spec_digest": environment.Digest,
		"format":                  "cozy.worker.v1.InstalledEnvironmentReceipt/1",
		"platform_target": map[string]any{"accelerator_abi": "cu129", "accelerator_backend": "cuda",
			"libc": "glibc2.39", "os_arch": "linux/amd64", "python_abi": "cp312"},
		"project_wheel_digest": projectDigest, "wheelhouse_manifest_digest": wheelhouseDigest,
	}))

	bindingRelease = exact(mustJSON(t, map[string]any{
		"descriptor":                refMap(descriptor),
		"documents":                 []any{map[string]any{"kind": "entrypoint_binding_plan", "ref": refMap(plan)}},
		"endpoint_execution_digest": digestOf([]byte("execution")),
		"environment_spec":          refMap(environment), "format": bindingFormat,
		"installed_environment_receipt": refMap(receipt),
		"object_set_digest":             digestOf([]byte("objects")),
	}))
	placementBytes, _, err := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{
		{
			PlacementId: "ra-test", Spec: &pb.PlacementSpec{
				EndpointReleaseId: "acme/h3@v1", EnvironmentSpecDigest: mustRaw(t, environment.Digest),
				InstalledEnvironmentReceiptDigest: mustRaw(t, receipt.Digest),
				DescriptorDigest:                  mustRaw(t, descriptor.Digest),
				BindingPlans: []*pb.ArtifactSubject{{Digest: mustRaw(t, plan.Digest),
					SubjectId: plan.Digest, Kind: "plan", Length: uint64(plan.Length)}},
			},
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	placement := exact(placementBytes)
	s := snapshot{
		AcquisitionAttemptID: "ra-test",
		BindingDocuments: []bindingDocument{{CanonicalBytes: plan.CanonicalBytes,
			Digest: plan.Digest, Kind: "entrypoint_binding_plan", Length: plan.Length}},
		BindingRelease: bindingRelease, Descriptor: descriptor, EndpointBundle: endpointBundle,
		EndpointExecutionDigest: digestOf([]byte("execution")), EnvironmentSpec: environment,
		EvaluatedConfig: evaluatedConfig, Format: format,
		InstalledEnvironmentReceipt: receipt, PlacementSet: placement,
	}
	return exact(mustJSON(t, s))
}

func exact(raw []byte) hub.ExactControlDocument {
	return hub.ExactControlDocument{CanonicalBytes: raw, Digest: digestOf(raw), Length: int64(len(raw))}
}

func refMap(document hub.ExactControlDocument) map[string]any {
	return map[string]any{"digest": document.Digest, "length": document.Length}
}

func digestOf(raw []byte) string {
	return "sha256:" + hex.EncodeToString(sha256Sum(raw))
}

func sha256Sum(raw []byte) []byte {
	sum := sha256.Sum256(raw)
	return sum[:]
}

func mustRaw(t *testing.T, digest string) []byte {
	t.Helper()
	raw, err := canonical.Raw(digest)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func mustPrettyJSON(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(raw, '\n')
}
