package live

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

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
	wheelhouse := jcs(t, map[string]any{"format": "WheelhouseManifest/3"})
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

	baseReceipt := jcs(t, map[string]any{
		"environment_proof":   "bin/cozy-environment-proof",
		"format":              "cozy.local.ManagedBaseReceipt/1",
		"generation_identity": "sha256:" + strings.Repeat("9", 64),
		"profile":             endpointprofile.CU130, "python": "bin/python", "python_abi": "cp312",
		"wheelhouse_manifest_digest": digestText(wheelhouse),
	})
	baseDigest := digestText(baseReceipt)
	base := layout.ManagedBase(baseDigest)
	must(t, os.MkdirAll(filepath.Join(base, "bin"), 0o755))
	must(t, os.WriteFile(filepath.Join(base, "ManagedBaseReceipt.json"), baseReceipt, 0o600))
	must(t, os.WriteFile(filepath.Join(base, "WheelhouseManifest.json"), wheelhouse, 0o600))
	writeExecutable(t, filepath.Join(base, "bin", "python"), "#!/bin/sh\nexit 0\n")
	writeExecutable(t, filepath.Join(base, "bin", "cozy-environment-proof"), `#!/usr/bin/python3
import hashlib,json,pathlib,sys
request=json.loads(pathlib.Path(sys.argv[1]).read_text())
generation=pathlib.Path(request["environment_root"])/"contents"/"fixture"
(generation/"site-packages").mkdir(parents=True)
receipt=b'{"base_family_digest":"sha256:`+strings.Repeat("7", 64)+`","environment_spec_digest":"sha256:`+strings.Repeat("8", 64)+`","format":"cozy.runtime.EndpointOverlayReceipt/1","overlay_wheels":[],"project_wheel_digest":"sha256:`+strings.Repeat("6", 64)+`"}'
pathlib.Path(sys.argv[2]).write_bytes(receipt)
print(json.dumps({"digest":"sha256:"+hashlib.sha256(receipt).hexdigest(),"generation":str(generation),"length":len(receipt),"reused":False},separators=(",",":"),sort_keys=True))
`)
	writeExecutable(t, filepath.Join(base, "bin", "cozy-runtime"), //cozy:allow independent Runtime CLI fixture, not a product execution door
		`#!/usr/bin/python3
import hashlib,json,sys
if "describe" in sys.argv:
 print(json.dumps({"descriptor_digest":"sha256:"+"d"*64}))
elif "doctor" in sys.argv:
 print(json.dumps({"device":{"name":"NVIDIA GeForce RTX 4090","state":"present","sm":89,"vram_total_bytes":1,"driver_version":"580.1.2","cuda_version":"13.0"},"host":{"platform":"fixture","ram_total_bytes":1,"vcpu_count":1},"cas":{"root":"/tmp","present":True,"artifacts":0},"credentials":[],"unreadable":[]}))
else: raise SystemExit(2)
`)
	writeExecutable(t, filepath.Join(base, "bin", "cozy-native-wheel-proof"), `#!/usr/bin/python3
import hashlib,json,pathlib,sys
request=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert request["format"]=="cozy.runtime.NativeWheelProofRequest/1"
assert request["base_realization_kind"]=="managed-local"
evidence=b'{"base_realization_digest":"`+baseDigest+`","base_realization_kind":"managed-local","base_worker_profile":"`+endpointprofile.CU130+`","endpoint_environment_spec_digest":"'+request["endpoint_environment_spec_digest"].encode()+b'","format":"cozy.runtime.NativeWheelQualification/1","host_capability":{},"host_qualification":{},"installed_environment_receipt_digest":"'+request["installed_environment_receipt_digest"].encode()+b'","operator_observation":{},"overlay_content_digest":"sha256:`+strings.Repeat("4", 64)+`","static_inspection":{},"wheelhouse_manifest_digest":"`+digestText(wheelhouse)+`"}'
pathlib.Path(sys.argv[2]).write_bytes(evidence)
print(json.dumps({"digest":"sha256:"+hashlib.sha256(evidence).hexdigest(),"length":len(evidence)},separators=(",",":"),sort_keys=True))
`)

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
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/local-execution") {
			if r.Header.Get("Authorization") != "Bearer admin" {
				writeHubError(w, http.StatusUnauthorized, "auth.token_invalid", "bad token")
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"candidate_id": "candidate-local", "profile": endpointprofile.CU130,
				"lease_id": "lease-local", "lease_expires_at": "2026-08-29T00:00:00Z",
				"base_worker_image_digest":  "sha256:" + strings.Repeat("5", 64),
				"base_realization":          map[string]string{"kind": "managed-local", "digest": baseDigest},
				"endpoint_environment_spec": exactDoc(ees), "endpoint_bundle": exactDoc(bundle),
				"descriptor": exactDoc(descriptor), "evaluated_config": exactDoc(evaluated),
				"resolved_wheel_set": exactDoc(resolved), "wheelhouse_manifest": exactDoc(wheelhouse),
				"resolution_lock": exactDoc(lock),
				"native_wheel_proof": map[string]any{
					"expected_result_digest": "sha256:" + strings.Repeat("3", 64), "fixture": "custom_op:run"},
				"downloads": []map[string]any{
					{"role": "project_wheel", "ref": map[string]any{"digest": packed.Digest, "length": len(wheelBytes)}, "url": server.URL + "/wheel", "expires_at": "2026-08-29T00:00:00Z"},
					{"role": "custom_wheel:custom-op:" + strings.TrimPrefix(customFact.Digest, "sha256:"), "ref": map[string]any{"digest": customFact.Digest, "length": len(customBytes)}, "url": server.URL + "/custom", "expires_at": "2026-08-29T00:00:00Z"},
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
		facts.HostEvidenceDigest == "" || facts.InstalledReceiptDigest == "" {
		t.Fatalf("managed evidence/lease facts are incomplete: %+v", facts)
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
