package producttest

import (
	"flag"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

var modeledPackageHub = flag.String("modeled-package-hub", "", "real Tensorhub for read-only published H3 interface validation")

// The published package interface, rather than a copied test schema, supplies
// both callable/slot declarations. This reads metadata only and claims no model custody.
func TestPublishedH3ModeledEntrypointSelection(t *testing.T) {
	if *modeledPackageHub == "" {
		t.Skip("requires -modeled-package-hub for the published H3 1.1.2 metadata")
	}
	root := t.TempDir()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	resolver := cli.NewResolver(store, config.Config{HubURL: *modeledPackageHub, Home: root}, nil)
	for _, function := range []string{"first_last_frame_to_video", "reference_media_to_video"} {
		models := []orchestrator.ModelRef{{Package: "paul/minimax-h3", Slot: function + ".models.model",
			Model: "paul/minimax-h3", Release: "1.0.0", Lane: "bf16-full",
			Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}}
		logical, entrypoint, problem := resolver.ResolveRemoteRelease("", "paul/minimax-h3", "1.1.2", function, models)
		fatal(t, problem)
		if entrypoint.Name != function || logical.Function != function || logical.PlanID != "" ||
			len(logical.Models) != 1 || logical.Models[0].Slot != models[0].Slot ||
			logical.Models[0].Manifest != models[0].Manifest {
			t.Fatal("remote metadata resolution changed the callable or its selected model")
		}
		models[0].Slot = "unselected.models.model"
		if _, _, problem := resolver.ResolveRemoteRelease("", "paul/minimax-h3", "1.1.2", function, models); problem == nil {
			t.Fatal("another callable's model selection was accepted")
		}
	}
}
