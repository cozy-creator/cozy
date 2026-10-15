package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

// Future calls may override an unavailable default or never run. Capture records
// the unavailable fact; the actual omitted-argument call must refuse, as covered
// by TestCapturedModelDefaultsDoNotAcquireUnusedOrInaccessibleModels.
func TestPrivateModelPreflightDefersAbsentCapturedInputs(t *testing.T) {
	for _, mode := range []string{"weights-unbound", "entrypoint-unbound", "weights-default"} {
		t.Run(mode, func(t *testing.T) {
			catalog := newLadderHub(t)
			layout, problem := home.Open(t.TempDir())
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			parent := cleanupTestInstall(layout, "1111111111111111", "1.0.0")
			child := cleanupTestInstall(layout, "2222222222222222", "1.0.0")
			child.Package = "local/private_ops"
			slot := map[string]any{"class": "Source", "path": "compute.models.source", "component_use": map[string]any{}}
			if mode != "entrypoint-unbound" {
				slot["weights"] = true
			}
			if mode == "weights-default" {
				slot["default_ladder"] = authoredH3(ladderLane)
			}
			call := map[string]any{"name": "compute", "models": []any{slot},
				"request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}},
				"invocable": map[string]any{"context": "ctx", "module": "private_ops", "export": "compute",
					"parameters": []any{"source"}, "defaults": map[string]any{}, "type_names": map[string]any{},
					"enum_members": map[string]any{}, "memoize": true, "capabilities": []any{}}}
			doc := map[string]any{"format": "cozy.package.interface/1", "application": "private_ops:app",
				"entrypoints": []any{call}}
			raw := assessmentJSON(t, doc)

			must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(child.Dir)), 0o700))
			must(t, os.WriteFile(launch.PackageInterfacePath(child.Dir), raw, 0o444))
			fatal(t, store.RecordInstall(parent))
			fatal(t, store.RecordInstall(child))
			fatal(t, store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: parent.ID,
				ChildInstallID: child.ID,
				Module:         "private_ops", Export: "compute", Entrypoint: "compute"}}))
			resolver := cli.NewResolver(store, config.Config{Home: layout.Root, HubURL: catalog.server.URL, HubToken: secret.New("ladder-test")})
			models, problem := resolver.UnpublishedChildModels(records.Request{InstallID: parent.ID})
			switch mode {
			case "weights-unbound", "entrypoint-unbound":
				fatal(t, problem)
				if len(models) != 0 {
					t.Fatal("future retained artifact acquired a fabricated static selection")
				}
			case "weights-default":
				fatal(t, problem)
				if len(models) != 1 || models[0].Model != ladderModel || len(models[0].Ladder) != 1 || models[0].Ladder[0].Lane != ladderLane {
					t.Fatal("deferral discarded an authored model ladder")
				}
			}
		})
	}
}
