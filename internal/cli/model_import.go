package cli

import (
	"os"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/modelsource"
)

func handleModelImport(ctx *Context) *exit.Error {
	name := strings.TrimSpace(ctx.Inv.Value("--name"))
	if problem := modelsource.LocalName(name); problem != nil {
		return problem
	}
	cwd, err := os.Getwd()
	if err != nil {
		return exit.Internalf("cannot resolve the current directory: %s", err)
	}
	if _, problem := modelsource.Parse(ctx.Inv.Args[0], cwd); problem != nil {
		return problem
	}
	if ctx.Cfg.TensorFSRegistry == "" {
		return exit.Named(exit.Structural, "tensorfs_registry_unavailable",
			"model import requires TensorFS's reviewed source registry, but no registry path is configured").
			WithRemedy("set tensorfs_registry in %s to the registry shipped and reviewed with TensorFS", ctx.Cfg.Home+"/config.yaml")
	}
	return exit.Named(exit.Structural, "model_import_execution_unavailable",
		"model import execution is not yet wired in this build")
}
