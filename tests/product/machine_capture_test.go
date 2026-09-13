package producttest

import (
	"bytes"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

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
