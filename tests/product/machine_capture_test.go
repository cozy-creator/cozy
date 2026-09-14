package producttest

import (
	"bytes"
	"path/filepath"
	"strings"
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
	inst := records.PackageInstall{ID: "model-tools", Dir: t.TempDir(), Package: root.Package, Version: root.Release, SourceKind: "local"}
	fatal(t, store.RecordInstall(inst))
	surface := &launch.PackageInterface{Digest: root.PackageInterfaceDigest,
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
		func(string, string) (localpackage.Revision, *exit.Error) {
			t.Fatal("self calls must use the already frozen revision")
			return localpackage.Revision{}, nil
		})
	fatal(t, problem)
	var doc pb.MachineExecutionCapture
	must(t, canonical.Unmarshal(capture.Canonical, &doc))
	if len(doc.Bindings) != 2 || len(doc.Revisions) != 1 {
		t.Fatalf("job and serving self bindings were not closed: %v", &doc)
	}
	for _, binding := range doc.Bindings {
		if binding.Export != "long_form" && binding.Export != "segment" {
			t.Fatalf("unmanaged function captured: %v", binding)
		}
		if !bytes.Equal(binding.CallerRevisionDigest, binding.CalleeRevisionDigest) {
			t.Fatal("self binding changed the captured revision")
		}
	}
}

func machineCaptureRevision(t *testing.T, name string, code byte) localpackage.Revision {
	t.Helper()
	source := bytes.Repeat([]byte{code}, 32)
	iface := bytes.Repeat([]byte{code + 1}, 32)
	file := bytes.Repeat([]byte{code + 2}, 32)
	_, digest, err := canonical.Identity(&pb.LocalPackageRevision{
		Package: name, Release: "1.0.0", SourceDigest: source,
		PackageInterface: &pb.Ref{Digest: iface, Length: 100},
		Files:            []*pb.LocalPackageFileRef{{Digest: file, Filename: "fixture-1.0.0-py3-none-any.whl", Length: 512}},
	})
	must(t, err)
	spell := func(v []byte) string { result, err := canonical.Spell(v); must(t, err); return result }
	return localpackage.Revision{Package: name, Release: "1.0.0", SourceDigest: spell(source), Digest: spell(digest), PackageInterfaceDigest: spell(iface), PackageInterfaceLength: 100,
		Files: []localpackage.File{{Digest: spell(file), Filename: "fixture-1.0.0-py3-none-any.whl", Length: 512, Kind: "project", Path: "/private/laptop/fixture.whl"}}}
}

func TestMachineCapturePreservesChildIdentityAcrossCallerEditsWithoutClientPaths(t *testing.T) {
	child := machineCaptureRevision(t, "local/ops", 10)
	var previous []byte
	for _, code := range []byte{20, 30} {
		root := machineCaptureRevision(t, "local/script", code)
		binding := records.ChildBinding{ParentInstallID: "root", ChildInstallID: "ops", LocalRevisionDigest: child.Digest, InterfaceDigest: child.PackageInterfaceDigest, Module: "ops", Export: "prepare", Entrypoint: "prepare"}
		capture, problem := localpackage.CaptureExecution("root", root, func(install string) ([]records.ChildBinding, *exit.Error) {
			if install == "root" {
				return []records.ChildBinding{binding}, nil
			}
			return []records.ChildBinding{{ParentInstallID: "ops", ChildInstallID: "ops", InterfaceDigest: child.PackageInterfaceDigest, Module: "ops", Export: "prepare", Entrypoint: "prepare"}}, nil
		}, func(install, digest string) (localpackage.Revision, *exit.Error) {
			if install != "ops" || digest != child.Digest {
				t.Fatalf("mutable dependency lookup: %q %q", install, digest)
			}
			return child, nil
		})
		fatal(t, problem)
		if bytes.Contains(capture.Canonical, []byte("/private/laptop")) || bytes.Contains(capture.Canonical, []byte("install_id")) {
			t.Fatal("client identity leaked into execution capture")
		}
		if previous != nil && bytes.Equal(previous, capture.Digest) {
			t.Fatal("caller source edit did not change capture")
		}
		previous = capture.Digest
		var doc pb.MachineExecutionCapture
		must(t, canonical.Unmarshal(capture.Canonical, &doc))
		if len(doc.Revisions) != 2 || len(doc.Bindings) != 2 {
			t.Fatalf("self binding did not close: %v", &doc)
		}
		for _, row := range doc.Bindings {
			got, _ := canonical.Spell(row.CalleeRevisionDigest)
			if got != child.Digest {
				t.Fatal("caller edit changed child revision")
			}
		}
	}
}

func TestMachineCaptureRejectsChangedInventoryAndConflictingBindings(t *testing.T) {
	root := machineCaptureRevision(t, "local/script", 20)
	child := machineCaptureRevision(t, "local/ops", 10)
	for _, arm := range []string{"inventory", "interface", "target"} {
		t.Run(arm, func(t *testing.T) {
			selected := child
			selected.Files = append([]localpackage.File(nil), child.Files...)
			rows := []records.ChildBinding{{ParentInstallID: "root", ChildInstallID: "ops", LocalRevisionDigest: child.Digest, InterfaceDigest: child.PackageInterfaceDigest, Module: "ops", Export: "prepare", Entrypoint: "prepare"}}
			switch arm {
			case "inventory":
				selected.Files[0].Length++
			case "interface":
				rows[0].InterfaceDigest = "sha256:" + strings.Repeat("f", 64)
			case "target":
				rows = append(rows, rows[0])
				rows[1].Entrypoint = "changed"
			}
			_, problem := localpackage.CaptureExecution("root", root, func(install string) ([]records.ChildBinding, *exit.Error) {
				if install == "root" {
					return rows, nil
				}
				return nil, nil
			}, func(string, string) (localpackage.Revision, *exit.Error) { return selected, nil })
			if problem == nil {
				t.Fatal("changed frozen identity accepted")
			}
		})
	}
}
