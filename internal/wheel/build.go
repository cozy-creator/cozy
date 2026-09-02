// Package wheel builds and inspects the package project's installable wheel.
// uv is the build frontend; the project owns its PEP 517 backend and file layout.
package wheel

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/processtree"
)

type Request struct {
	Context context.Context
	Tree    string
	OutDir  string
}

type Result struct {
	Path string
}

const (
	maxBuildLogBytes = 1 << 20
)

type buildLog struct {
	mu     sync.Mutex
	buffer bytes.Buffer
}

func (w *buildLog) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if room := maxBuildLogBytes - w.buffer.Len(); room > 0 {
		_, _ = w.buffer.Write(p[:min(len(p), room)])
	}
	return len(p), nil
}

func (w *buildLog) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buffer.String()
}

// Build asks uv to build the current working tree, then validates the exact
// wheel emitted by the project's declared build backend. Git is not involved.
func Build(req Request) (*Result, *exit.Error) {
	parent := req.Context
	if parent == nil {
		parent = context.Background()
	}
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
			"local wheel staging already contains %d wheel(s)", len(held))
	}

	ctx, cancel := context.WithCancelCause(parent)
	defer cancel(nil)
	cmd := exec.CommandContext(ctx, "uv", "build", "--wheel", "--out-dir", out,
		"--no-progress", root)
	cmd.Env = config.Frozen().Tool()
	cmd.WaitDelay = 250 * time.Millisecond
	processtree.Prepare(cmd)
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return os.ErrProcessDone
		}
		err := processtree.Kill(cmd.Process.Pid, syscall.SIGKILL)
		if err != nil && !processtree.Alive(cmd.Process.Pid) {
			return os.ErrProcessDone
		}
		return err
	}
	log := &buildLog{}
	cmd.Stdout, cmd.Stderr = log, log
	runErr := cmd.Start()
	var containmentErr error
	if runErr == nil {
		pid := cmd.Process.Pid
		if containmentErr = processtree.Adopt(cmd); containmentErr != nil {
			_ = cmd.Wait()
			runErr = containmentErr
		} else {
			runErr = cmd.Wait()
		}
		// A backend may have spawned descendants that outlived uv itself. The
		// build transaction ends the entire contained tree on every exit path.
		_ = processtree.Kill(pid, syscall.SIGKILL)
		processtree.Release(pid)
	}
	body := log.String()
	if parent.Err() != nil {
		return nil, exit.Named(exit.Canceled, "project_wheel_build_canceled", "uv build was canceled")
	}
	if containmentErr != nil {
		return nil, exit.Named(exit.Structural, "project_wheel_builder_uncontained",
			"cannot contain the uv build process tree: %v", containmentErr)
	}
	if cmd.ProcessState == nil {
		return nil, exit.Named(exit.Structural, "project_wheel_builder_missing",
			"cannot run uv build: %v", runErr).
			WithRemedy("install uv; Cozy delegates standard PEP 517 wheel construction to `uv build --wheel`")
	}
	if cmd.ProcessState.ExitCode() != 0 {
		detail := strings.Join(strings.Fields(body), " ")
		if detail == "" {
			detail = runErr.Error()
		}
		return nil, exit.Named(exit.Validation, "project_wheel_build_refused", "uv build refused: %s", detail).
			WithRemedy("fix the project's pyproject.toml, build backend, or package layout, then run `uv build --wheel` locally")
	}
	wheels, err := filepath.Glob(filepath.Join(out, "*.whl"))
	if err != nil || len(wheels) != 1 {
		return nil, exit.Named(exit.Validation, "project_wheel_build_result_invalid",
			"uv build emitted %d wheels; package publication requires exactly one", len(wheels))
	}
	info, err = os.Stat(wheels[0])
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > MaxWheelBytes {
		return nil, exit.Named(exit.Validation, "project_wheel_build_result_invalid",
			"uv build output is not one regular wheel at or below %d B", MaxWheelBytes)
	}
	return &Result{Path: wheels[0]}, nil
}
