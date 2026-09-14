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

func TestPrivateModelPreflightDefersAbsentCapturedInputs(t *testing.T) {
	for _, mode := range []string{"job-unbound", "entrypoint-unbound", "job-default", "job-owner-unreadable"} {
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
			if mode == "job-default" || mode == "job-owner-unreadable" {
				slot["default_ladder"] = authoredH3(ladderLane)
			}
			if mode == "job-owner-unreadable" {
				child.Package = ladderPackage
				catalog.bindingsUnavailable = true
			}
			call := map[string]any{"name": "compute", "models": []any{slot},
				"request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}},
				"invocable": map[string]any{"context": "ctx", "module": "private_ops", "export": "compute",
					"parameters": []any{"source"}, "defaults": map[string]any{}, "type_names": map[string]any{},
					"enum_members": map[string]any{}, "memoize": true, "capabilities": []any{}}}
			doc := map[string]any{"format": "cozy.package.interface/1", "application": "private_ops:app",
				"entrypoints": []any{}, "jobs": []any{}}
			if mode == "entrypoint-unbound" {
				doc["entrypoints"] = []any{call}
			} else {
				call["publishes"], call["weights_outputs"] = false, []any{}
				doc["jobs"] = []any{call}
			}
			raw := assessmentJSON(t, doc)
			child.PackageInterface = assessmentDigest(raw)
			must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(child.Dir)), 0o700))
			must(t, os.WriteFile(launch.PackageInterfacePath(child.Dir), raw, 0o444))
			fatal(t, store.RecordInstall(parent))
			fatal(t, store.RecordInstall(child))
			fatal(t, store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: parent.ID,
				ChildInstallID: child.ID, InterfaceDigest: child.PackageInterface, LocalRevisionDigest: childDigest("b"),
				Module: "private_ops", Export: "compute", Entrypoint: "compute"}}))
			resolver := cli.NewResolver(store, config.Config{Home: layout.Root, HubURL: catalog.server.URL, HubToken: secret.New("ladder-test")}, nil)
			models, problem := resolver.UnpublishedChildModels(records.Request{InstallID: parent.ID, Kind: "job"})
			switch mode {
			case "job-unbound", "entrypoint-unbound", "job-owner-unreadable":
				fatal(t, problem)
				if len(models) != 0 {
					t.Fatal("future retained artifact acquired a fabricated static selection")
				}
			case "job-default":
				fatal(t, problem)
				if len(models) != 1 || models[0].Model != ladderModel || len(models[0].Ladder) != 1 || models[0].Ladder[0].Lane != ladderLane {
					t.Fatal("deferral discarded an authored model ladder")
				}
			}
		})
	}
}

func TestInvokedCapturedEntrypointStillRequiresItsActualModel(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	installed := cleanupTestInstall(layout, "3333333333333333", "1.0.0")
	installed.Package = "local/captured"
	raw := assessmentJSON(t, map[string]any{
		"format": "cozy.package.interface/1", "application": "captured:app",
		"entrypoints": []any{map[string]any{
			"name": "judge", "models": []any{map[string]any{"class": "Judge", "path": "judge.models.model", "component_use": map[string]any{}}},
			"request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}},
			"invocable": map[string]any{"context": "ctx", "module": "captured", "export": "judge", "parameters": []any{},
				"defaults": map[string]any{}, "type_names": map[string]any{}, "enum_members": map[string]any{}, "memoize": false, "capabilities": []any{}},
		}},
		"jobs": []any{map[string]any{"name": "parent", "publishes": false, "models": []any{},
			"request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}}, "weights_outputs": []any{}}},
	})
	installed.PackageInterface = assessmentDigest(raw)
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0o700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installed.Dir), raw, 0o444))
	fatal(t, store.RecordInstall(installed))
	fatal(t, store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: installed.ID, ChildInstallID: installed.ID,
		InterfaceDigest: installed.PackageInterface, Module: "captured", Export: "judge", Entrypoint: "judge"}}))
	resolver := cli.NewResolver(store, config.Config{Home: layout.Root}, nil)
	parent := records.Request{ID: "parent", InstallID: installed.ID, Entrypoint: "parent", Kind: "job"}
	models, problem := resolver.UnpublishedChildModels(parent)
	fatal(t, problem)
	if len(models) != 0 {
		t.Fatal("preflight invented a model")
	}
	_, _, problem = resolver.ResolveUnpublishedChild(parent, installed.PackageInterface, "captured", "judge",
		assessmentJSON(t, map[string]any{"payload": map[string]any{}, "models": map[string]any{"model": nil}}))
	if problem == nil || problem.ErrName() != "child.model_unbound" {
		t.Fatal("invoking an unbound entrypoint bypassed actual model admission", problem)
	}
}
