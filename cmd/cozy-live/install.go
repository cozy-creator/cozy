package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
)

// installEndpoint puts the SDXL endpoint on a service root THE WAY A USER DOES: one
// `cozy install --from <release archive> --digest <sha256>`, then the local artifact index
// row its binding selects.
//
// This is what replaced `--dev-endpoint` (cl-010). The document that flag read was a
// hand-written EndpointSpec — an interpreter, an argv, a device envelope and a binding
// record, all supplied by the driver. None of it is supplied now: the generation's venv is
// the interpreter, its committed descriptor is the surface, and its endpoint.toml is the
// selection. A driver that has to hand the product its launch facts is not driving the
// product.
//
// The ARTIFACT ROW is the one thing still placed by the harness, and it is named rather
// than hidden: `cozy-runtime pull`/`ingest` are cr-016's owed verbs (they refuse typed
// naming tfs-002/tfs-003 today), so nothing yet WRITES the local artifact index. The row
// points at cr-005's real 4.782 GiB CAS, and the runtime's own `artifacts.install` writes
// it — the harness does not compose the index's layout.
func installEndpoint(root string) string {
	release := flag("release", defaultRelease())
	if _, err := os.Stat(release); err != nil {
		must("the SDXL release archive", fmt.Errorf(
			"%s: %w — build it with scripts/sdxl-release.sh", release, err))
	}
	bench := flag("bench", "/home/fidika/cozy_v2/tensorfs-bench")

	// cozyRun, not a bare exec: the install must land on THIS root, and the root reaches
	// a child only through the product's own env allowlist.
	code, out := cozyRun(root, "install", "cozy/sdxl-unet", "--from", release,
		"--digest", digestOf(release))
	if code != 0 {
		fmt.Println(out)
		must("installing the endpoint", fmt.Errorf("cozy install exited %d", code))
	}
	generation := ""
	for _, line := range strings.Split(out, "\n") {
		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "generation" {
			generation = strings.TrimSpace(value)
		}
	}
	if generation == "" {
		fmt.Println(out)
		must("the install", fmt.Errorf("printed no generation"))
	}

	snapshot := benchSnapshot(bench)
	python := home.VenvPython(filepath.Join(root, "generations", generation, "venv"))
	script := `
import sys
from pathlib import Path
from cozy_runtime.cli import artifacts
home, store, config, snap = sys.argv[1:5]
artifacts.install(Path(home), artifacts.Artifact(
    ref="cozy/sdxl-unet@cr-005", store=store, config=config,
    snapshots={"unet": snap}, lane="plain-fp16",
    bytes=5_134_927_368, tensors=1680, variant="sm89"))
print("artifact row installed")
`
	cmd := exec.Command("/usr/bin/nice", "-n", "19", python, "-c", script, root,
		filepath.Join(bench, "store"),
		filepath.Join(bench, "hf", "sdxl", "unet", "config.json"), snapshot)
	cmd.Env = childEnv(root)
	data, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(data))
		must("installing the artifact index row", err)
	}
	return "cozy/sdxl-unet"
}

// childEnv is the PRODUCT's own allowlist, pointed at one service root. The driver has no
// business inventing a second child-env mechanism, and the env fence says there is only
// one reader of the environment in this repository.
func childEnv(root string) []string {
	must("setting COZY_HOME for the one env reader", os.Setenv("COZY_HOME", root))
	cfg, e := config.Load()
	must("config", errOf(e))
	return cfg.Child("COZY_HOME=" + root)
}

// defaultRelease is where scripts/sdxl-release.sh puts the archive.
func defaultRelease() string {
	home, err := os.UserHomeDir()
	must("the home directory", err)
	return filepath.Join(home, ".cache", "cozy", "cl-010", "sdxl-unet-1.0.0.tar.gz")
}

func cozyBinary() string {
	abs, err := filepath.Abs(flag("cozy", defaultCozy()))
	must("resolving the cozy binary", err)
	return abs
}

// defaultCozy is the built product binary's default name — an .exe where executables
// carry one.
func defaultCozy() string {
	if runtime.GOOS == "windows" {
		return "./cozy.exe"
	}
	return "./cozy"
}

// digestOf reads the archive's sha256 the way `cozy install --digest` wants it. The
// release builder prints it too; recomputing it here keeps the driver runnable against any
// archive it is pointed at.
func digestOf(path string) string {
	out, err := runOut("sha256sum", path)
	must("hashing the release archive", err)
	return "sha256:" + strings.Fields(out)[0]
}

// benchSnapshot reads cr-005's checkpoint id out of the bench's own results file.
func benchSnapshot(bench string) string {
	data, err := os.ReadFile(filepath.Join(bench, "results", "checkpoint.txt"))
	must("reading the bench checkpoint", err)
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if name, value, ok := strings.Cut(line, "\t"); ok && name == "checkpoint" {
			return value
		}
	}
	must("the bench checkpoint", fmt.Errorf("no checkpoint digest in results/checkpoint.txt"))
	return ""
}

// cozyRun runs the product binary as a user would type it, against one service root.
func cozyRun(root string, args ...string) (int, string) {
	cmd := niceCmd(cozyBinary(), args...)
	cmd.Env = childEnv(root)
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return code, string(data)
}

// cozyJSON runs the product binary under --json and decodes its ONE document.
func cozyJSON(root string, args ...string) (int, map[string]any, string) {
	code, out := cozyRun(root, append(args, "--json")...)
	var doc map[string]any
	_ = json.Unmarshal([]byte(lastJSONLine(out)), &doc)
	return code, doc, out
}

// lastJSONLine picks the document out of mixed output: progress narration goes to stderr
// but CombinedOutput merges the two, and stdout carries exactly one document.
func lastJSONLine(out string) string {
	for _, line := range reverse(strings.Split(strings.TrimSpace(out), "\n")) {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "{") && strings.HasSuffix(line, "}") {
			return line
		}
	}
	return ""
}

func reverse(in []string) []string {
	out := make([]string, 0, len(in))
	for i := len(in) - 1; i >= 0; i-- {
		out = append(out, in[i])
	}
	return out
}

// niceCmd builds a `nice -n 19` child — this DRIVER's resource discipline on a shared
// box, imposed on the launch rather than baked into the product. Windows has no nice(1);
// the CI runner is not the shared box, so the child simply runs.
func niceCmd(name string, args ...string) *exec.Cmd {
	if runtime.GOOS == "windows" {
		return exec.Command(name, args...)
	}
	return exec.Command("/usr/bin/nice", append([]string{"-n", "19", name}, args...)...)
}

// interruptSignal is what a person's ctrl-C sends.
func interruptSignal() os.Signal { return syscall.SIGINT }
