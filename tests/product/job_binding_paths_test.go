package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestPublishedJobPreparationCarriesExactInterfaceModelPaths(t *testing.T) {
	if *modeledPackageHub == "" {
		t.Skip("explicit published-package metadata readback required")
	}
	st, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer st.Close()
	resolver := cli.NewResolver(st, config.Config{HubURL: *modeledPackageHub, Home: t.TempDir()}, nil)
	const pkg = "paul/minimax-h3-tools"
	models := []orchestrator.ModelRef{
		{Package: pkg, Slot: "dits", Model: "paul/minimax-h3", Release: "1.0.0-rc.1", Lane: "bf16-full", Manifest: "sha256:3f6c224a010fbff36adce0f19a8b866f4f1739a1d30e2202c140ba994dc0b4df", ManifestLength: 164},
		{Package: pkg, Slot: "shared", Model: "paul/minimax-h3", Release: "1.0.0-rc.1", Lane: "bf16-full", Manifest: "sha256:3f6c224a010fbff36adce0f19a8b866f4f1739a1d30e2202c140ba994dc0b4df", ManifestLength: 164},
	}
	logical, job, problem := resolver.ResolveRemoteJob(pkg, "2.2.9", "assemble_full", models, false)
	fatal(t, problem)
	for _, model := range logical.Models {
		matched := false
		for _, declared := range job.Models {
			if model.Slot == declared.Param {
				matched = model.BindingPath == declared.Path
			}
		}
		if !matched {
			t.Fatal("prepared path did not come from the selected interface")
		}
	}
	if models[0].BindingPath != "" || models[1].BindingPath != "" {
		t.Fatal("resolver mutated caller model bindings")
	}
	logical.Models[0].BindingPath = "four-lane.models.dits"
	if _, _, problem := resolver.ResolveRemoteJob(pkg, "2.2.9", "assemble_full", logical.Models, false); problem == nil {
		t.Fatal("another declared callable's model path was accepted")
	}
}
