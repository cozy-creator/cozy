package live

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/endpointprofile"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/wheel"

	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

func TestQualifiedManagedLocalInstall(t *testing.T) {
	cozyHome := t.TempDir()
	layout, problem := home.Open(cozyHome)
	fatal(t, problem)
	projectTree := t.TempDir()
	mustWrite(t, filepath.Join(projectTree, "pyproject.toml"), "[project]\nname=\"marco\"\nversion=\"1.0.0\"\n")
	mustWrite(t, filepath.Join(projectTree, "endpoint.toml"), "[application]\nobject=\"marco:app\"\n")
	mustWrite(t, filepath.Join(projectTree, "marco.py"), "app=object()\n")
	mustWrite(t, filepath.Join(projectTree, "endpoint.descriptor.json"), `{"application":"marco:app","entrypoints":[],"format":"cozy.endpoint.descriptor/1","jobs":[]}`)
	packed, problem := wheel.Pack(wheel.Request{Tree: projectTree, OutDir: t.TempDir()})
	fatal(t, problem)
	wheelBytes := mustRead(t, packed.Path)
	customPath := filepath.Join(t.TempDir(), "custom_op-1.2.3-cp312-cp312-manylinux_2_28_x86_64.whl")
	writeTestWheel(t, customPath, "custom_op", "1.2.3", false,
		[]string{"cp312-cp312-manylinux_2_28_x86_64"}, map[string][]byte{
			"custom_op/__init__.py": []byte("from ._native import run\n"),
			"custom_op/_native.so":  []byte("prebuilt-native-fixture"),
		})
	customFact, problem := wheel.Inspect(customPath, wheel.CustomWheel)
	fatal(t, problem)
	customBytes := mustRead(t, customPath)

	bundle := jcs(t, map[string]any{"format": "tensorhub.endpoint_bundle/2", "project_wheel": packed.Fact})
	resolved := jcs(t, map[string]any{"format": "ResolvedWheelSet/3", "wheels": []any{customFact}})
	lock := jcs(t, map[string]any{"format": "tensorhub.resolution_lock/1"})
	baseDistributions := []map[string]string{{"distribution": "torch", "version": "2.13.0+cu130"}}
	wheelhouse := jcs(t, map[string]any{
		"base_distributions":    baseDistributions,
		"base_import_roots":     []map[string]any{{"providers": []string{"torch"}, "root": "torch"}},
		"compatibility_profile": map[string]string{"accelerator_build": "cu130", "os_cpu": "linux-x86", "python_abi": "cp312", "torch_release": "2.13.0"},
		"format":                "WheelhouseManifest/3", "minimum_cuda_version": "13.0", "minimum_driver_version": "580.00.00",
		"platform_target": map[string]string{"accelerator_abi": "cu130", "accelerator_backend": "cuda", "libc": "glibc2.39", "os_arch": "linux/amd64", "python_abi": "cp312"},
		"wheels": []map[string]any{{"digest": "sha256:" + strings.Repeat("a", 64), "distribution": "torch",
			"filename": "torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl", "import_roots": []string{"torch"},
			"length": 1, "tags": []string{"cp312-cp312-manylinux_2_28_x86_64"}, "version": "2.13.0+cu130"}},
	})
	descriptor := jcs(t, map[string]any{"application": "marco:app", "entrypoints": []any{},
		"format": "cozy.endpoint.descriptor/1", "jobs": []any{}})
	evaluated := jcs(t, map[string]any{})
	eesValue, err := canonical.Document(&pb.EndpointEnvironmentSpec{
		PlatformTarget: &pb.PlatformTarget{OsArch: "linux/amd64", Libc: "glibc2.39",
			PythonAbi: "cp312", AcceleratorBackend: "cuda", AcceleratorAbi: "cu130"},
		CompatibilityProfile: &pb.CompatibilityProfile{TorchRelease: "2.13.0",
			AcceleratorBuild: "cu130", PythonAbi: "cp312", OsCpu: "linux-x86"},
		EndpointBundleDigest: digestRaw(t, bundle), ProjectWheelDigest: digestRaw(t, wheelBytes),
		WheelhouseManifestDigest: digestRaw(t, wheelhouse),
	})
	must(t, err)
	ees, err := canonical.Write(eesValue)
	must(t, err)
	runtimeReceipt := jcs(t, map[string]any{
		"base_family_digest": digestText(wheelhouse), "environment_spec_digest": digestText(ees),
		"format": "cozy.runtime.EndpointOverlayReceipt/1",
		"overlay_wheels": []map[string]string{
			{"digest": customFact.Digest, "distribution": customFact.Distribution, "owner": "custom", "version": customFact.Version},
			{"digest": packed.Fact.Digest, "distribution": packed.Fact.Distribution, "owner": "project", "version": packed.Fact.Version},
		},
		"project_wheel_digest": packed.Fact.Digest,
	})
	wheelDigests := []string{customFact.Digest, packed.Fact.Digest}
	sort.Strings(wheelDigests)
	overlayContentDigest := digestText(jcs(t, map[string]any{
		"base_family_digest": digestText(wheelhouse), "format": "cozy.runtime.EndpointOverlayContent/1",
		"wheel_digests": wheelDigests,
	}))
	emptySymbols := digestText(jcs(t, []string{}))
	staticInspection := map[string]any{"format": "cozy.runtime.NativeWheelInspection/1", "members": []map[string]any{{
		"cubin_sms": []int{89}, "cuda_sections": []string{".nv_fatbin"},
		"defined_symbols_digest": emptySymbols, "member": "custom_op/_native.so",
		"needed": []string{}, "ptx_compute": []int{}, "required_symbols_digest": emptySymbols,
		"runpaths": []string{}, "soname": "_native.so",
	}}}
	staticInspectionBytes := jcs(t, staticInspection)
	expectedResultDigest := "sha256:" + strings.Repeat("3", 64)
	nativeEvidence := jcs(t, map[string]any{
		"base_worker_profile":              endpointprofile.CU130,
		"endpoint_environment_spec_digest": digestText(ees), "format": "cozy.runtime.NativeWheelQualification/1",
		"host_capability": map[string]any{
			"gpu": map[string]any{"device_index": 0, "driver_version": "580.1.2", "name": "NVIDIA GeForce RTX 4090", "sm": 89},
			"seat": map[string]any{"cuda_available": true, "distributions": baseDistributions, "implementation": "cpython",
				"libc_name": "glibc", "libc_version": "2.39", "machine": "x86_64", "python_abi": "cp312",
				"system": "linux", "torch_cuda": "13.0", "torch_version": "2.13.0+cu130"},
		},
		"host_qualification": map[string]any{"format": "cozy.runtime.NativeHostQualification/1", "gpu_sm": 89,
			"host_library_count": 1, "inspection_digest": digestText(staticInspectionBytes), "qualification": "static-host-compatible"},
		"installed_environment_receipt_digest": digestText(runtimeReceipt),
		"operator_observation": map[string]any{"expected_result_digest": expectedResultDigest, "fixture": "custom_op:run",
			"format": "cozy.runtime.NativeOperatorObservation/1", "result_digest": expectedResultDigest},
		"overlay_content_digest": overlayContentDigest, "static_inspection": staticInspection,
		"wheelhouse_manifest_digest": digestText(wheelhouse),
	})

	baseReceipt := jcs(t, map[string]any{
		"environment_proof":   "bin/cozy-environment-proof",
		"format":              "cozy.local.ManagedBaseReceipt/1",
		"generation_identity": "sha256:" + strings.Repeat("9", 64),
		"profile":             endpointprofile.CU130, "python": "bin/python", "python_abi": "cp312",
		"wheelhouse_manifest_digest": digestText(wheelhouse),
	})
	baseDigest := digestText(baseReceipt)
	base := layout.ManagedBase(baseDigest)
	proofOrder := filepath.Join(cozyHome, "proof-order")
	must(t, os.MkdirAll(filepath.Join(base, "bin"), 0o755))
	must(t, os.WriteFile(filepath.Join(base, "ManagedBaseReceipt.json"), baseReceipt, 0o600))
	must(t, os.WriteFile(filepath.Join(base, "WheelhouseManifest.json"), wheelhouse, 0o600))
	writeExecutable(t, filepath.Join(base, "bin", "python"), "#!/bin/sh\nexit 0\n")
	environmentProof := func(receipt []byte) string {
		return `#!/usr/bin/python3
import base64,hashlib,json,pathlib,sys
request=json.loads(pathlib.Path(sys.argv[1]).read_text())
pathlib.Path("` + proofOrder + `").open("a").write("materialize\n")
generation=pathlib.Path(request["environment_root"])/"contents"/"` + strings.TrimPrefix(overlayContentDigest, "sha256:") + `"
(generation/"site-packages").mkdir(parents=True)
receipt=base64.b64decode("` + base64.StdEncoding.EncodeToString(receipt) + `")
pathlib.Path(sys.argv[2]).write_bytes(receipt)
print(json.dumps({"digest":"sha256:"+hashlib.sha256(receipt).hexdigest(),"generation":str(generation),"length":len(receipt),"reused":False},separators=(",",":"),sort_keys=True))
`
	}
	writeExecutable(t, filepath.Join(base, "bin", "cozy-environment-proof"), environmentProof(runtimeReceipt))
	writeExecutable(t, filepath.Join(base, "bin", "cozy-runtime"), //cozy:allow independent Runtime CLI fixture, not a product execution door
		`#!/usr/bin/python3
import hashlib,json,pathlib,sys
if "describe" in sys.argv:
 pathlib.Path("`+proofOrder+`").open("a").write("descriptor\n")
 print(json.dumps({"descriptor_digest":"`+digestText(descriptor)+`"}))
elif "doctor" in sys.argv:
 pathlib.Path("`+proofOrder+`").open("a").write("host\n")
 print(json.dumps({"device":{"name":"NVIDIA GeForce RTX 4090","state":"present","sm":89,"vram_total_bytes":1,"driver_version":"580.1.2","cuda_version":"13.0"},"host":{"platform":"fixture","ram_total_bytes":1,"vcpu_count":1},"cas":{"root":"/tmp","present":True,"artifacts":0},"credentials":[],"unreadable":[]}))
else: raise SystemExit(2)
`)
	nativeProof := func(evidence []byte) string {
		return `#!/usr/bin/python3
import base64,hashlib,json,pathlib,sys
request=json.loads(pathlib.Path(sys.argv[1]).read_text())
pathlib.Path("` + proofOrder + `").open("a").write("native\n")
assert request["format"]=="cozy.runtime.NativeWheelProofRequest/1"
assert set(request)=={"device_index","endpoint_environment_spec_digest","environment_root","expected_result_digest","fixture","format","installed_environment_receipt_digest","python","wheelhouse_manifest"}
assert request["installed_environment_receipt_digest"]=="` + digestText(runtimeReceipt) + `"
evidence=base64.b64decode("` + base64.StdEncoding.EncodeToString(evidence) + `")
pathlib.Path(sys.argv[2]).write_bytes(evidence)
print(json.dumps({"digest":"sha256:"+hashlib.sha256(evidence).hexdigest(),"length":len(evidence)},separators=(",",":"),sort_keys=True))
`
	}
	writeExecutable(t, filepath.Join(base, "bin", "cozy-native-wheel-proof"), nativeProof(nativeEvidence))

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet && r.URL.Path == "/wheel" {
			_, _ = w.Write(wheelBytes)
			return
		}
		if r.Method == http.MethodGet && r.URL.Path == "/custom" {
			_, _ = w.Write(customBytes)
			return
		}
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/local-qualification-materials") {
			if r.Header.Get("Authorization") != "Bearer admin" {
				writeHubError(w, http.StatusUnauthorized, "auth.token_invalid", "bad token")
				return
			}
			expiresAt := time.Now().Add(time.Hour).UTC().Format(time.RFC3339)
			realization := map[string]string{"kind": "managed-local", "digest": baseDigest}
			if strings.Contains(r.Header.Get("X-Tensorhub-Reason"), "OCI substitution") {
				realization = map[string]string{"kind": "oci", "digest": "registry.invalid/worker@sha256:" + strings.Repeat("5", 64)}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"candidate_id": "candidate-local", "profile": endpointprofile.CU130,
				"lease_id": "lease-local", "lease_expires_at": expiresAt,
				"base_realization":          realization,
				"endpoint_environment_spec": exactDoc(ees), "endpoint_bundle": exactDoc(bundle),
				"descriptor": exactDoc(descriptor), "evaluated_config": exactDoc(evaluated),
				"resolved_wheel_set": exactDoc(resolved), "wheelhouse_manifest": exactDoc(wheelhouse),
				"resolution_lock": exactDoc(lock),
				"native_wheel_proof": map[string]any{
					"expected_result_digest": "sha256:" + strings.Repeat("3", 64), "fixture": "custom_op:run"},
				"downloads": []map[string]any{
					{"role": "project_wheel", "ref": map[string]any{"digest": packed.Digest, "length": len(wheelBytes)}, "url": server.URL + "/wheel", "expires_at": expiresAt},
					{"role": "custom_wheel:custom-op:" + strings.TrimPrefix(customFact.Digest, "sha256:"), "ref": map[string]any{"digest": customFact.Digest, "length": len(customBytes)}, "url": server.URL + "/custom", "expires_at": expiresAt},
				},
			})
			return
		}
		writeHubError(w, http.StatusNotFound, "route.not_found", r.URL.Path)
	}))
	defer server.Close()

	run := func(args ...string) (int, string) {
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
		cmd.Env = childEnv(t, cozyHome, "TENSORHUB_URL="+server.URL, "TENSORHUB_TOKEN=admin")
		body, _ := cmd.CombinedOutput()
		return cmd.ProcessState.ExitCode(), string(body)
	}
	code, output := run("install", "cozy/marco@1.0.0", "--profile", endpointprofile.CU130,
		"--major", "v1", "--reason", "managed fixture")
	if code != 0 || !strings.Contains(output, "candidate-local") ||
		!strings.Contains(output, "no dependency resolution") {
		t.Fatalf("managed install [exit %d]\n%s", code, output)
	}
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	installed, problem := store.Installed()
	fatal(t, problem)
	if len(installed) != 1 || installed[0].SourceKind != "published-profile" ||
		installed[0].Runtime != filepath.Join(base, "bin", "cozy-runtime") || //cozy:allow assertion over the independent Runtime CLI fixture
		!strings.HasSuffix(installed[0].ProjectDir, "/site-packages") {
		t.Fatalf("managed control install is incomplete: %+v", installed)
	}
	facts, problem := store.ManagedInstall(installed[0].ID)
	fatal(t, problem)
	if facts == nil || facts.CandidateID != "candidate-local" || facts.LeaseID != "lease-local" ||
		facts.HostEvidenceDigest == "" || facts.NativeEvidenceDigest == "" ||
		facts.HostEvidenceDigest == facts.NativeEvidenceDigest || facts.InstalledReceiptDigest == "" {
		t.Fatalf("managed evidence/lease facts are incomplete: %+v", facts)
	}
	if facts.HostEvidenceDigest != digestText(mustRead(t, filepath.Join(installed[0].Dir, "host-evidence.json"))) ||
		facts.NativeEvidenceDigest != digestText(mustRead(t, filepath.Join(installed[0].Dir, "native-wheel-qualification.json"))) {
		t.Fatalf("managed host/native evidence digests do not identify their separate stored bytes: %+v", facts)
	}
	if order := strings.TrimSpace(string(mustRead(t, proofOrder))); order != "materialize\nhost\nnative\ndescriptor" {
		t.Fatalf("endpoint code ran before local hardware proof: %q", order)
	}
	if code, output := run("install", "cozy/marco@1.0.0", "--profile", endpointprofile.CU130,
		"--major", "v1", "--force", "--reason", "OCI substitution fixture"); code != 6 ||
		!strings.Contains(output, "managed_install_grant_invalid") {
		t.Fatalf("OCI realization entered managed-local qualification [exit %d]\n%s", code, output)
	}
	_, active, problem := store.ActivePin("cozy/marco", 1)
	fatal(t, problem)
	if active == nil || active.ID != installed[0].ID {
		t.Fatalf("OCI substitution disturbed active managed-local install: %+v", active)
	}
	if code, output := run("install", "cozy/marco@1.0.0", "--profile", endpointprofile.CU130,
		"--major", "v1", "--reason", "managed fixture replay"); code != 0 ||
		!strings.Contains(output, "already pinned") {
		t.Fatalf("managed install replay [exit %d]\n%s", code, output)
	}
	wrongDigest := "sha256:" + strings.Repeat("e", 64)
	badNativeEvidence := bytes.Replace(nativeEvidence, []byte(overlayContentDigest), []byte(wrongDigest), 1)
	writeExecutable(t, filepath.Join(base, "bin", "cozy-native-wheel-proof"), nativeProof(badNativeEvidence))
	if code, output := run("install", "cozy/marco@1.0.0", "--profile", endpointprofile.CU130,
		"--major", "v1", "--force", "--reason", "native evidence mismatch fixture"); code != 6 ||
		!strings.Contains(output, "managed_native_evidence_invalid") {
		t.Fatalf("managed native evidence mismatch [exit %d]\n%s", code, output)
	}
	_, active, problem = store.ActivePin("cozy/marco", 1)
	fatal(t, problem)
	if active == nil || active.ID != installed[0].ID {
		t.Fatalf("failed native-evidence replacement disturbed active install: %+v", active)
	}
	writeExecutable(t, filepath.Join(base, "bin", "cozy-native-wheel-proof"), nativeProof(nativeEvidence))
	badReceipt := bytes.Replace(runtimeReceipt, []byte(packed.Fact.Digest), []byte(wrongDigest), 1)
	writeExecutable(t, filepath.Join(base, "bin", "cozy-environment-proof"), environmentProof(badReceipt))
	if code, output := run("install", "cozy/marco@1.0.0", "--profile", endpointprofile.CU130,
		"--major", "v1", "--force", "--reason", "receipt mismatch fixture"); code != 6 ||
		!strings.Contains(output, "managed_environment_receipt_invalid") {
		t.Fatalf("managed receipt mismatch [exit %d]\n%s", code, output)
	}
	_, active, problem = store.ActivePin("cozy/marco", 1)
	fatal(t, problem)
	if active == nil || active.ID != installed[0].ID {
		t.Fatalf("failed receipt replacement disturbed active install: %+v", active)
	}
	writeExecutable(t, filepath.Join(base, "bin", "cozy-environment-proof"), environmentProof(runtimeReceipt))
	writeExecutable(t, filepath.Join(base, "bin", "cozy-runtime"), //cozy:allow independent Runtime CLI mismatch fixture
		`#!/usr/bin/python3
import json,sys
if "describe" in sys.argv: print(json.dumps({"descriptor_digest":"sha256:`+strings.Repeat("d", 64)+`"}))
elif "doctor" in sys.argv: print(json.dumps({"device":{"name":"NVIDIA GeForce RTX 4090","state":"present","cuda_version":"13.0"}}))
else: raise SystemExit(2)
`)
	if code, output := run("install", "cozy/marco@1.0.0", "--profile", endpointprofile.CU130,
		"--major", "v1", "--force", "--reason", "descriptor mismatch fixture"); code != 6 ||
		!strings.Contains(output, "managed_descriptor_mismatch") {
		t.Fatalf("managed descriptor mismatch [exit %d]\n%s", code, output)
	}
	_, active, problem = store.ActivePin("cozy/marco", 1)
	fatal(t, problem)
	if active == nil || active.ID != installed[0].ID {
		t.Fatalf("failed replacement disturbed active install: %+v", active)
	}
	fatal(t, store.Unpin("cozy/marco", 1))
	forgotten, problem := store.ForgetIfUnreferenced(installed[0].ID)
	fatal(t, problem)
	remaining, problem := store.ManagedInstall(installed[0].ID)
	fatal(t, problem)
	if !forgotten || remaining != nil {
		t.Fatalf("managed profile facts did not cascade with reclaimed control install: %t %+v", forgotten, remaining)
	}
}

func jcs(t *testing.T, value any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	must(t, err)
	body, err = canonical.NormalizeJCS(body)
	must(t, err)
	return body
}

func exactDoc(body []byte) map[string]any {
	return map[string]any{"digest": digestText(body), "length": len(body), "canonical_bytes_base64": body}
}

func digestText(body []byte) string {
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func digestRaw(t *testing.T, body []byte) []byte {
	t.Helper()
	raw, err := hex.DecodeString(strings.TrimPrefix(digestText(body), "sha256:"))
	must(t, err)
	return raw
}

func writeExecutable(t *testing.T, path, body string) {
	t.Helper()
	must(t, os.WriteFile(path, []byte(body), 0o700))
}
