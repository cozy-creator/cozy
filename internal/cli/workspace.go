package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"os"
	"path/filepath"
)

// ResolveWorkspace launches only the installed control Runtime. It imports no
// package, allocates no accelerator and uses the same store as local jobs.
func (r *Resolver) ResolveWorkspace() (orchestrator.WorkerLaunchSpec, *exit.Error) {
	runtime, problem := hostruntime.Path(r.cfg.Child())
	if problem != nil {
		return orchestrator.WorkerLaunchSpec{}, problem
	}
	artifacts := filepath.Join(r.cfg.Home, "runtime-workspace", "artifacts")
	if err := os.MkdirAll(artifacts, 0o700); err != nil {
		return orchestrator.WorkerLaunchSpec{}, exit.Internalf("cannot prepare the Runtime workspace artifact directory: %s", err)
	}
	return orchestrator.WorkerLaunchSpec{Python: runtime, Args: []string{"serve"},
		Dir: r.cfg.Home, TensorFSRoot: r.cfg.TensorFSRoot, ArtifactCache: artifacts, Warmup: orchestrator.WarmupNone}, nil
}
