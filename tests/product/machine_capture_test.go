package producttest

import (
	"bytes"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestMachineCaptureIncludesInvocableServingSelfBindings(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	root := machineCaptureRevision(t, "local/model-tools", 20)
	inst := records.PackageInstall{ID: root.ID, Dir: t.TempDir(), Package: root.Package, Version: root.Release, SourceKind: "local"}
	fatal(t, store.RecordInstall(inst))
	surface := &launch.PackageInterface{
		Jobs: []launch.Entrypoint{{Name: "long_form", Invocable: &launch.Invocable{Module: "model_tools", Export: "long_form"}}},
		Entrypoints: []launch.Entrypoint{
			{Name: "segment", Invocable: &launch.Invocable{Module: "model_tools", Export: "segment"}},
			{Name: "unmanaged"},
		},
	}
	for range 2 { // Existing immutable installs must capture the same rows on reuse.
		fatal(t, install.CaptureSelfBindings(store, inst, surface))
	}
	capture, problem := localpackage.CaptureExecution(inst.ID, root, store.ChildBindings,
		func(string, string) (localpackage.Installation, *exit.Error) {
			t.Fatal("self calls must use the already frozen revision")
			return localpackage.Installation{}, nil
		})
	fatal(t, problem)
	var doc pb.MachineExecutionCapture
	must(t, canonical.Unmarshal(capture.Canonical, &doc))
	if len(doc.Bindings) != 2 || len(doc.InstalledPackages) != 1 {
		t.Fatalf("job and serving self bindings were not closed: %v", &doc)
	}
	for _, binding := range doc.Bindings {
		if binding.Export != "long_form" && binding.Export != "segment" {
			t.Fatalf("unmanaged function captured: %v", binding)
		}
		if binding.CallerInstallationId != binding.CalleeInstallationId {
			t.Fatal("self binding changed the captured revision")
		}
	}
}

func machineCaptureRevision(t *testing.T, name string, code byte) localpackage.Installation {
	t.Helper()
	id := fmt.Sprintf("install-%d", code)
	return localpackage.Installation{ID: id, Package: name, Release: "1.0.0", PackageInterface: fixturePackageInterface,
		Files: []localpackage.File{{Filename: "source.tar", Length: 512, Kind: "source", Path: "/private/laptop/source.tar"}}, SourceArchive: "source.tar"}
}

func TestMachineCapturePreservesChildIdentityAcrossCallerEditsWithoutClientPaths(t *testing.T) {
	child := machineCaptureRevision(t, "local/ops", 10)
	var previous []byte
	for _, code := range []byte{20, 30} {
		root := machineCaptureRevision(t, "local/script", code)
		binding := records.ChildBinding{ParentInstallID: root.ID, ChildInstallID: child.ID, Module: "ops", Export: "prepare", Entrypoint: "prepare"}
		capture, problem := localpackage.CaptureExecution(root.ID, root, func(install string) ([]records.ChildBinding, *exit.Error) {
			if install == root.ID {
				return []records.ChildBinding{binding}, nil
			}
			return []records.ChildBinding{{ParentInstallID: child.ID, ChildInstallID: child.ID, Module: "ops", Export: "prepare", Entrypoint: "prepare"}}, nil
		}, func(install, digest string) (localpackage.Installation, *exit.Error) {
			if install != child.ID || digest != child.ID {
				t.Fatalf("mutable dependency lookup: %q %q", install, digest)
			}
			return child, nil
		})
		fatal(t, problem)
		if bytes.Contains(capture.Canonical, []byte("/private/laptop")) {
			t.Fatal("client identity leaked into execution capture")
		}
		if previous != nil && bytes.Equal(previous, capture.Digest) {
			t.Fatal("caller source edit did not change capture")
		}
		previous = capture.Digest
		var doc pb.MachineExecutionCapture
		must(t, canonical.Unmarshal(capture.Canonical, &doc))
		if len(doc.InstalledPackages) != 2 || len(doc.Bindings) != 2 {
			t.Fatalf("self binding did not close: %v", &doc)
		}
		for _, row := range doc.Bindings {
			got := row.CalleeInstallationId
			if got != child.ID {
				t.Fatal("caller edit changed child revision")
			}
		}
	}
}

func TestMachineCaptureRejectsForeignOwnershipAndConflictingBindings(t *testing.T) {
	root := machineCaptureRevision(t, "local/script", 20)
	child := machineCaptureRevision(t, "local/ops", 10)
	_, problem := localpackage.CaptureExecution("foreign", root, func(string) ([]records.ChildBinding, *exit.Error) { return nil, nil }, func(string, string) (localpackage.Installation, *exit.Error) { return child, nil })
	if problem == nil {
		t.Fatal("foreign root installation accepted")
	}
	_, problem = localpackage.CaptureExecution(root.ID, root, func(string) ([]records.ChildBinding, *exit.Error) {
		return []records.ChildBinding{{ParentInstallID: "foreign", ChildInstallID: child.ID, Module: "ops", Export: "prepare", Entrypoint: "prepare"}}, nil
	}, func(string, string) (localpackage.Installation, *exit.Error) { return child, nil })
	if problem == nil {
		t.Fatal("foreign binding owner accepted")
	}
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, installation := range []localpackage.Installation{root, child} {
		fatal(t, store.RecordInstall(records.PackageInstall{ID: installation.ID, Package: installation.Package, Version: installation.Release}))
	}
	binding := records.ChildBinding{ParentInstallID: root.ID, ChildInstallID: child.ID, Module: "ops", Export: "prepare", Entrypoint: "prepare"}
	fatal(t, store.RecordChildBindings([]records.ChildBinding{binding}))
	for _, field := range []string{"callee", "entrypoint"} {
		changed := binding
		if field == "callee" {
			changed.ChildInstallID = root.ID
		} else {
			changed.Entrypoint = "changed"
		}
		if problem := store.RecordChildBindings([]records.ChildBinding{changed}); problem == nil {
			t.Fatal("changed immutable binding accepted", field)
		}
	}
}
