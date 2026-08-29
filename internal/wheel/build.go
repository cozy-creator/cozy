// Package wheel builds and inspects the endpoint project's installable wheel.
// uv is the build frontend; the project owns its PEP 517 backend and file layout.
package wheel

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

const Tag = "py3-none-any"

type Request struct {
	Tree   string
	OutDir string
}

type Result struct {
	Path string
	Fact Fact
}

// Build asks uv to build the current working tree, then validates the exact
// wheel emitted by the project's declared build backend. Git is not involved.
func Build(req Request) (*Result, *exit.Error) {
	root, err := filepath.Abs(req.Tree)
	if err != nil {
		return nil, exit.Usagef("project directory %q is not resolvable: %s", req.Tree, err)
	}
	info, err := os.Stat(root)
	if err != nil || !info.IsDir() {
		return nil, exit.Named(exit.NotFound, "project_tree_absent", "%s is not a directory", root)
	}
	out, err := filepath.Abs(req.OutDir)
	if err != nil || strings.TrimSpace(req.OutDir) == "" {
		return nil, exit.Usagef("wheel output directory %q is not resolvable", req.OutDir)
	}
	if err := os.MkdirAll(out, 0o700); err != nil {
		return nil, exit.Named(exit.Structural, "project_wheel_output_unwritable", "%s: %v", out, err)
	}
	if held, _ := filepath.Glob(filepath.Join(out, "*.whl")); len(held) != 0 {
		return nil, exit.Named(exit.Conflict, "project_wheel_output_not_empty",
			"private wheel staging already contains %d wheel(s)", len(held))
	}

	cmd := exec.Command("uv", "build", "--wheel", "--out-dir", out,
		"--no-build-logs", "--no-progress", root)
	body, runErr := cmd.CombinedOutput()
	if cmd.ProcessState == nil {
		return nil, exit.Named(exit.Structural, "project_wheel_builder_missing",
			"cannot run uv build: %v", runErr).
			WithRemedy("install uv; Creator delegates standard PEP 517 wheel construction to `uv build --wheel`")
	}
	if cmd.ProcessState.ExitCode() != 0 {
		detail := strings.Join(strings.Fields(string(body)), " ")
		if detail == "" {
			detail = runErr.Error()
		}
		return nil, exit.Named(exit.Validation, "project_wheel_build_refused", "uv build refused: %s", detail).
			WithRemedy("fix the project's pyproject.toml, build backend, or package layout, then run `uv build --wheel` locally")
	}
	wheels, err := filepath.Glob(filepath.Join(out, "*.whl"))
	if err != nil || len(wheels) != 1 {
		return nil, exit.Named(exit.Validation, "project_wheel_build_result_invalid",
			"uv build emitted %d wheels; endpoint publication requires exactly one", len(wheels))
	}
	fact, problem := Inspect(wheels[0], ProjectWheel)
	if problem != nil {
		return nil, problem
	}
	return &Result{Path: wheels[0], Fact: fact}, nil
}
