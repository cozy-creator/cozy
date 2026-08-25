package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/coord"
)

const (
	mib = 1 << 20
	gib = 1 << 30
	// The release cr-005's real SDXL UNet is published under. The coordinator pins it and
	// refuses a worker that registers under any other.
	release = "cozy/sdxl-unet@cr-005"
)

// sdxlSpec is the REAL endpoint cl-001's end-to-end proof runs: cr-005's SDXL UNet,
// 1,680 destinations and 4.782 GiB out of ~/cozy_v2/tensorfs-bench's real CAS, served by
// the REAL cozy-runtime supervisor and executor from a read-only checkout.
//
// The binding RECORD is cozy-creator's own: it is the local pinned-binding record this
// coordinator owns, staged into the worker's COZY_HOME, and the plan id on the wire is
// the digest of its canonical bytes. th-004 replaces the document later; the identity
// rule does not move.
func sdxlSpec(entrypoints ...string) coord.EndpointSpec {
	runtime := flag("runtime", "/home/fidika/cozy_v2/cozy-runtime")
	bench := flag("bench", "/home/fidika/cozy_v2/tensorfs-bench")
	if len(entrypoints) == 0 {
		entrypoints = []string{"denoise"}
	}

	snapshot := ""
	data, err := os.ReadFile(filepath.Join(bench, "results", "checkpoint.txt"))
	must("reading the bench checkpoint", err)
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if name, value, ok := strings.Cut(line, "\t"); ok && name == "checkpoint" {
			snapshot = value
		}
	}
	if snapshot == "" {
		must("the bench checkpoint", fmt.Errorf("no checkpoint digest in results/checkpoint.txt"))
	}

	var bindings []*coord.Binding
	for _, entrypoint := range entrypoints {
		bindings = append(bindings, &coord.Binding{
			Entrypoint: entrypoint,
			Record: map[string]any{
				"project":                   filepath.Join(runtime, "corpus", "endpoint"),
				"model_class":               "SdxlUnetModel",
				"binding_path":              entrypoint + ".models.model",
				"param":                     "model",
				"component":                 "unet",
				"store":                     filepath.Join(bench, "store"),
				"config":                    filepath.Join(bench, "hf", "sdxl", "unet", "config.json"),
				"snapshot":                  snapshot,
				"release":                   release,
				"variant":                   "sm89",
				"vram_bytes":                int64(6 * gib),
				"host_bytes":                int64(2 * gib),
				"pinned_bytes":              int64(64 * mib),
				"entrypoint":                entrypoint,
				"model_construction_digest": "",
			},
		})
	}

	return coord.EndpointSpec{
		Endpoint:   "cozy/sdxl-unet",
		ReleaseID:  release,
		Generation: "", // an uninstalled dev tree: cl-009 generations pin an installed one
		// nice(1) is this DRIVER's resource discipline on a shared box, imposed on the
		// launch rather than baked into the coordinator's policy.
		Python: "/usr/bin/nice",
		Args: []string{"-n", "19", filepath.Join(runtime, "corpus", ".venv", "bin", "python"), "-c",
			"import sys; from cozy_runtime.internal.worker.session import main; " +
				"raise SystemExit(main(sys.argv[1:]))"},
		Dir:      runtime,
		Imposed:  []string{"PYTHONPATH=" + runtime + ":" + filepath.Join(runtime, "src")},
		Devices:  []string{"0"},
		Bindings: bindings,
		GraceSec: 3,
	}
}

// planIDOf resolves the wire id of one staged binding.
func planIDOf(spec coord.EndpointSpec, entrypoint string) string {
	for _, b := range spec.Bindings {
		if b.Entrypoint == entrypoint {
			id, e := b.PlanID()
			if e != nil {
				must("plan id", e)
			}
			return id
		}
	}
	must("plan id", fmt.Errorf("no binding for %s", entrypoint))
	return ""
}

func payload(v map[string]any) []byte {
	data, err := json.Marshal(v)
	must("rendering the request payload", err)
	return data
}

func fileDigest(path string) (string, int64) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", 0
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), int64(len(data))
}

// requireFreeGPU refuses to start when someone else already holds the card. This box is
// shared with other agents' live runs, and two 6 GiB reservations on an 8 GiB card is not
// a verification, it is an OOM.
func requireFreeGPU() int {
	used := gpuUsedMiB()
	if used > 900 {
		must("the GPU", fmt.Errorf("%d MiB is already in use on device 0 — wait for the card", used))
	}
	return used
}

func gpuUsedMiB() int {
	out, err := runOut("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
	if err != nil {
		return -1
	}
	n := 0
	fmt.Sscanf(strings.TrimSpace(strings.Split(out, "\n")[0]), "%d", &n)
	return n
}
