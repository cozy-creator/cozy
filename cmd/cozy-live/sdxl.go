package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
)

const (
	mib = 1 << 20
	gib = 1 << 30
	// The release cr-005's real SDXL UNet is published under. The orchestrator pins it and
	// refuses a worker that registers under any other.
	release = "cozy/sdxl-unet@cr-005"
)

// sdxlSpec is the REAL endpoint cl-001's end-to-end proof runs: cr-005's SDXL UNet,
// 1,680 destinations and 4.782 GiB out of ~/cozy_v2/tensorfs-bench's real CAS, served by
// the REAL cozy-runtime supervisor and executor from a read-only checkout.
//
// This MODELED harness preserves the older explicit binding record: it is staged into the
// worker's COZY_HOME, and its canonical identity is the plan id on the wire. Weightless
// installed releases instead consume Runtime-authored plan subjects and launch through
// Runtime's explicit weightless-endpoint mode; this corpus is not that path.
// PIN THE RUNTIME. `--runtime` may name a `git archive` of a pinned cozy-runtime commit
// rather than the live checkout, and `--venv` then supplies the interpreter (a venv is
// not in a git archive). Found by being bitten, mid-run: another agent's UNCOMMITTED
// cr-009 edit added `job_capacity=` beside `serving_capacity=` in the supervisor's
// Report, and `capacity` is a proto ONEOF — so every Report advertised zero ready
// serving plans and no orchestrator could ever dispatch. A verification whose peer moves
// underneath it measures nothing, and the pin is what makes a number readable.
func sdxlSpec(entrypoints ...string) orchestrator.WorkerLaunchSpec {
	runtime := flag("runtime", "/home/fidika/cozy_v2/cozy-runtime")
	venv := flag("venv", filepath.Join(runtime, "corpus", ".venv"))
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

	var bindings []*orchestrator.Binding
	for _, entrypoint := range entrypoints {
		bindings = append(bindings, &orchestrator.Binding{
			Entrypoint: entrypoint,
			Record: map[string]any{
				"project":              filepath.Join(runtime, "corpus", "endpoint"),
				"endpoint_release":     release,
				"model_class":          "SdxlUnetModel",
				"model_binding_path":   entrypoint + ".models.model",
				"model_parameter_name": "model",
				"components":           []string{"unet"},
				"store":                filepath.Join(bench, "store"),
				"config":               filepath.Join(bench, "hf", "sdxl", "unet", "config.json"),
				"snapshot":             snapshot,
				"release":              release,
				"variant":              "sm89",
				"vram_bytes":           int64(6 * gib),
				"host_bytes":           int64(2 * gib),
				// tfs-007's measured staging shape: 16 slots x 16 MiB. The record's
				// VALUES are as much the runtime's contract as its key set is.
				"pinned_bytes":              int64(256 * mib),
				"entrypoint":                entrypoint,
				"model_construction_digest": "",
			},
		})
	}

	return orchestrator.WorkerLaunchSpec{
		Placement: orchestrator.DesiredPlacement{
			Endpoint:  "cozy/sdxl-unet",
			ReleaseID: release,
			InstallID: "", // an uninstalled dev tree: cl-009 installs pin a real one
			Bindings:  bindings,
		},
		// nice(1) is this DRIVER's resource discipline on a shared box, imposed on the
		// launch rather than baked into the orchestrator's policy.
		Python: "/usr/bin/nice",
		// `launch.Binary` names the runtime, so the ONE site that spells the binary stays the
		// one site (the `runtime` fence). This dev spec points it at the corpus venv rather
		// than a generation, which is the only difference from what `cozy start` launches.
		Args:     []string{"-n", "19", launch.Binary(filepath.Dir(venv)), "serve"}, //cozy:allow the DRIVER resolves the real runtime for the orchestrator's own spawn — the one execution path, driven live
		Dir:      runtime,
		Imposed:  []string{"PYTHONPATH=" + runtime + ":" + filepath.Join(runtime, "src")},
		Devices:  []string{"0"},
		GraceSec: 3,
	}
}

// planIDOf resolves the wire id of one staged binding.
func planIDOf(spec orchestrator.WorkerLaunchSpec, entrypoint string) string {
	for _, b := range spec.Placement.Bindings {
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

// gpuReleased polls until the card is back at its baseline. A fixed sleep was wrong:
// the CUDA context is torn down by the DRIVER after the process exits, and on a shared
// box that can take longer than a guess. Waiting on the fact rather than on a clock is
// the same discipline the rest of this repository uses for liveness.
func gpuReleased(idle int, timeout time.Duration) int {
	deadline := time.Now().Add(timeout)
	used := gpuUsedMiB()
	for time.Now().Before(deadline) {
		if used <= idle+40 {
			return used
		}
		time.Sleep(500 * time.Millisecond)
		used = gpuUsedMiB()
	}
	return used
}
