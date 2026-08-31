package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/managedbase"
)

func managedBaseInput(t *testing.T) managedbase.Input {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"format": "WheelhouseManifest/1",
		"compatibility_profile": map[string]any{
			"accelerator_build": "cu130", "os_cpu": "linux-x86",
			"python_abi": "cp312", "torch_release": "2.13.0",
		},
		"base_distributions": []map[string]any{
			{"distribution": "cozy-runtime", "version": "0.0.11"}, //cozy:allow exact managed-base fixture
			{"distribution": "tensorfs", "version": "0.0.3"},
			{"distribution": "torch", "version": "2.13.0+cu130"},
		},
		"base_import_roots": []map[string]any{{"providers": []string{"cozy-runtime"}, "root": "cozy_runtime"}}, //cozy:allow exact managed-base fixture
		"wheels": []map[string]any{
			{"digest": "sha256:" + strings.Repeat("1", 64), "distribution": "cozy-runtime", "filename": "cozy_runtime-0.0.11-py3-none-any.whl", "import_roots": []string{"cozy_runtime"}, "length": 101, "tags": []string{"py3-none-any"}, "version": "0.0.11"}, //cozy:allow exact managed-base fixture
			{"digest": "sha256:" + strings.Repeat("2", 64), "distribution": "tensorfs", "filename": "tensorfs-0.0.3-cp312-abi3-manylinux.whl", "import_roots": []string{"tensorfs"}, "length": 202, "tags": []string{"cp312-abi3-manylinux"}, "version": "0.0.3"},
		},
	})
	must(t, err)
	raw, err = canonical.NormalizeJCS(raw)
	must(t, err)
	sum := sha256.Sum256(raw)
	return managedbase.Input{Profile: managedbase.SupportedProfile, Manifest: managedbase.ExactDocument{
		Bytes: raw, Digest: "sha256:" + hex.EncodeToString(sum[:]), Length: int64(len(raw)),
	}}
}

func TestManagedBaseExactProductionAndSelection(t *testing.T) {
	input := managedBaseInput(t)
	changed := input
	changed.Profile = "torch2.13.0-cu132-cp312-linux-x86"
	if _, problem := managedbase.Ensure(filepath.Join(t.TempDir(), "managed-bases"), changed); problem == nil ||
		problem.ErrName() != "managed_base.profile_unsupported" {
		t.Fatalf("unsupported profile = %v", problem)
	}
	changed = input
	changed.Manifest.Length++
	if _, problem := managedbase.Ensure(filepath.Join(t.TempDir(), "managed-bases"), changed); problem == nil ||
		problem.ErrName() != "managed_base.manifest_identity_mismatch" {
		t.Fatalf("changed manifest identity = %v", problem)
	}

	imageDigest := "sha256:" + strings.Repeat("3", 64)
	configDigest := "sha256:" + strings.Repeat("4", 64)
	labels := map[string]string{
		"cozy.base_worker.profile":          input.Profile,
		"cozy.wheelhouse_manifest_digest":   input.Manifest.Digest,
		"cozy.python.version":               "3.12.3",
		"cozy.torch.version":                "2.13.0+cu130",
		"cozy.cuda.version":                 "13.0",
		"cozy.control_runtime_digest":       "sha256:" + strings.Repeat("1", 64),
		"cozy.control_runtime_length":       "101",
		"cozy.runtime_worker.entrypoint":    "/opt/cozy/bin/cozy-runtime-worker",
		"cozy.base_image.source":            "docker.io/pytorch/pytorch:2.13.0@sha256:" + strings.Repeat("5", 64),
		"org.opencontainers.image.source":   "https://github.com/cozy-creator/base-worker-image",
		"org.opencontainers.image.revision": strings.Repeat("6", 40),
	}
	inspect, _ := json.Marshal([]any{map[string]any{
		"Id": configDigest, "RepoDigests": []string{"tensorhub/worker@" + imageDigest},
		"Config": map[string]any{"Labels": labels},
	}})
	probe, _ := json.Marshal(map[string]any{
		"manifest_sha256": strings.TrimPrefix(input.Manifest.Digest, "sha256:"),
		"python":          "3.12.3", "runtime": "0.0.11", "tensorfs": "0.0.3",
		"torch": "2.13.0+cu130", "torch_cuda": "13.0", "worker": true,
	})
	tools := t.TempDir()
	docker := filepath.Join(tools, "docker")
	script := "#!/bin/sh\n" +
		"if [ \"$1\" = image ]; then printf '%s' '" + string(inspect) + "'; exit 0; fi\n" +
		"if [ \"$1\" = run ]; then printf '%s' '" + string(probe) + "'; exit 0; fi\n" +
		"if [ \"$1\" = pull ]; then exit 0; fi\nexit 91\n"
	must(t, os.WriteFile(docker, []byte(script), 0o755))
	t.Setenv("PATH", tools)

	cozyHome := filepath.Join(t.TempDir(), "cozy-home")
	baseRoot := filepath.Join(cozyHome, "managed-bases")
	record, problem := managedbase.Ensure(baseRoot, input)
	fatal(t, problem)
	if record.Identity.Image != "docker.io/tensorhub/worker@"+imageDigest ||
		record.Identity.Runtime.Version != "0.0.11" || record.Identity.TensorFS.Version != "0.0.3" {
		t.Fatalf("record = %+v", record)
	}
	active, err := os.ReadFile(filepath.Join(baseRoot, "active", input.Profile+".json"))
	if err != nil || !strings.Contains(string(active), record.BaseID) {
		t.Fatalf("active = %q, %v", active, err)
	}
	// A different manifest cannot borrow the mutable profile tag, and the failed
	// refresh cannot move the already active exact generation.
	drifted := input
	drifted.Manifest.Bytes = []byte(strings.ReplaceAll(string(input.Manifest.Bytes), "cozy_runtime", "cozy_runtimf"))
	driftSum := sha256.Sum256(drifted.Manifest.Bytes)
	drifted.Manifest.Digest = "sha256:" + hex.EncodeToString(driftSum[:])
	drifted.Manifest.Length = int64(len(drifted.Manifest.Bytes))
	if _, driftProblem := managedbase.Ensure(baseRoot, drifted); driftProblem == nil ||
		driftProblem.ErrName() != "managed_base.image_mismatch" {
		t.Fatalf("drifted profile tag = %v", driftProblem)
	}
	stillActive, err := os.ReadFile(filepath.Join(baseRoot, "active", input.Profile+".json"))
	if err != nil || string(stillActive) != string(active) {
		t.Fatalf("failed refresh moved active base: before=%q after=%q err=%v", active, stillActive, err)
	}

	command, launchProblem := managedbase.RuntimeCommand(baseRoot, input.Profile,
		input.Manifest.Digest, filepath.Join(cozyHome, "generations", "g1"), true)
	fatal(t, launchProblem)
	joined := strings.Join(command.Args, " ")
	if command.Bin != docker || !strings.Contains(joined, record.Identity.Image) ||
		!strings.Contains(joined, "-m cozy_runtime.cli.main") || strings.Contains(joined, "cozy-runtime serve") {
		t.Fatalf("managed command = %s %s", command.Bin, joined)
	}
}
