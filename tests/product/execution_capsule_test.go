package producttest

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/executionowner"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func capsuleWheel(t *testing.T, name string) []byte {
	t.Helper()
	var raw bytes.Buffer
	archive := zip.NewWriter(&raw)
	metadata, err := archive.Create(name + "-1.0.0.dist-info/METADATA")
	must(t, err)
	_, err = metadata.Write([]byte("Metadata-Version: 2.4\nName: " + name + "\nVersion: 1.0.0\n"))
	must(t, err)
	must(t, archive.Close())
	return raw.Bytes()
}

func executionCapsule(t *testing.T) executionowner.Capsule {
	t.Helper()
	makePackage := func(name, function string) (executionowner.Package, string, string) {
		iface, err := canonical.NormalizeJCS([]byte(`{"format":"cozy.package.interface/1","application":"` + name + `:app","entrypoints":[],"jobs":[{"name":"` + function + `","models":[],"request":{"fields":[]},"result":{"fields":[]},"publishes":false,"weights_outputs":[],"invocable":{"context":"ctx","module":"` + name + `","export":"` + function + `","parameters":[],"defaults":{},"type_names":{},"enum_members":{},"memoize":false,"capabilities":[]}}]}`))
		must(t, err)
		ifaceID, _ := canonical.Spell(canonical.Digest(iface))
		wheel := capsuleWheel(t, name)
		revision, digest, err := canonical.Identity(&pb.LocalPackageRevision{
			Package: "local/" + name, Release: "1.0.0", SourceDigest: canonical.Digest([]byte(name)),
			PackageInterface: &pb.Ref{Digest: canonical.Digest(iface), Length: uint64(len(iface))},
			Files:            []*pb.LocalPackageFileRef{{Digest: canonical.Digest(wheel), Filename: name + "-1.0.0-py3-none-any.whl", Length: uint64(len(wheel))}},
		})
		must(t, err)
		id, _ := canonical.Spell(digest)
		return executionowner.Package{Revision: revision, Interface: iface}, id, ifaceID
	}
	parent, parentID, _ := makePackage("parent", "main")
	child, childID, iface := makePackage("step", "advance")
	return executionowner.Capsule{Format: executionowner.CapsuleFormat,
		Root:     executionowner.Root{Revision: parentID, Entrypoint: "main", Input: json.RawMessage(`{}`), IdempotencyKey: "first-admission"},
		Packages: []executionowner.Package{parent, child},
		Bindings: []executionowner.Binding{{ParentRevision: parentID, ChildRevision: childID, InterfaceDigest: iface, Module: "step", Export: "advance", Entrypoint: "advance"}},
	}
}

func TestExecutionCapsuleBindsTheExactCapturedInventory(t *testing.T) {
	c := executionCapsule(t)
	raw, problem := executionowner.Encode(c)
	fatal(t, problem)
	accepted, problem := executionowner.Decode(raw)
	fatal(t, problem)
	want, _ := canonical.Spell(canonical.Digest(raw))
	if accepted.Digest != want || len(accepted.Packages) != 2 || accepted.Capsule.Root.Revision != c.Root.Revision || !bytes.Equal(accepted.Raw, raw) {
		t.Fatal("capsule did not preserve exact admission and package identities")
	}
	for name, change := range map[string]func(*executionowner.Capsule){
		"foreign root":                 func(c *executionowner.Capsule) { c.Root.Revision = childDigest("f") },
		"unknown root callable":        func(c *executionowner.Capsule) { c.Root.Entrypoint = "other" },
		"foreign child":                func(c *executionowner.Capsule) { c.Bindings[0].ChildRevision = childDigest("f") },
		"changed interface binding":    func(c *executionowner.Capsule) { c.Bindings[0].InterfaceDigest = childDigest("f") },
		"unreachable captured package": func(c *executionowner.Capsule) { c.Bindings = nil },
		"duplicate binding":            func(c *executionowner.Capsule) { c.Bindings = append(c.Bindings, c.Bindings[0]) },
		"duplicate revision":           func(c *executionowner.Capsule) { c.Packages = append(c.Packages, c.Packages[0]) },
		"callable path escape":         func(c *executionowner.Capsule) { c.Bindings[0].Module = "../step" },
		"changed interface bytes":      func(c *executionowner.Capsule) { c.Packages[1].Interface = c.Packages[0].Interface },
		"null root input":              func(c *executionowner.Capsule) { c.Root.Input = json.RawMessage(`null`) },
	} {
		t.Run(name, func(t *testing.T) {
			changed := executionCapsule(t)
			change(&changed)
			if _, problem := executionowner.Encode(changed); problem == nil {
				t.Fatal("altered admission escaped the capture boundary")
			}
		})
	}
	for _, altered := range [][]byte{
		append(append([]byte(nil), raw...), '\n'),
		bytes.Replace(raw, []byte(`"root":{`), []byte(`"root":{"command":"python evil.py",`), 1),
		[]byte(strings.Repeat(" ", pb.MaxInlineControlBytes+1)),
	} {
		if _, problem := executionowner.Decode(altered); problem == nil {
			t.Fatal("capsule accepted noncanonical, executable, or oversized metadata")
		}
	}
}
