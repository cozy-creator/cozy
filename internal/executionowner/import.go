package executionowner

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
)

// Import retains source snapshots in the ordinary Creator install/revision store.
// Captured Python/closure fields remain source facts. Native package preparation
// subsequently materializes and verifies each actual execution environment.
// No active pins, laptop paths, request rows, or attempt journals are imported.
func (b *Bootstrap) Import(layout home.Layout, store *records.Store) (*records.PackageInstall, *exit.Error) {
	writer, problem := install.Lock(layout)
	if problem != nil {
		return nil, problem
	}
	defer writer.Unlock()
	installs := map[string]records.PackageInstall{}
	for _, pkg := range b.Capsule.Capsule.Packages {
		id, _ := canonical.Spell(canonical.Digest(pkg.Revision))
		doc := b.Capsule.Packages[id]
		if doc == nil || len(b.Wheels[id]) == 0 {
			return nil, invalid("validated capture lost its uploaded wheels")
		}
		// Source environment observations may differ while the portable revision
		// is identical. Keep those captured histories in distinct install rows;
		// operation memoization continues to use its ordinary callee identity.
		capture, _ := json.Marshal(pkg)
		installID := "ins-execution-" + hex.EncodeToString(canonical.Digest(capture))[:32]
		dir := layout.InstallDir(installID)
		major, problem := install.MajorOf(doc.Str("release"))
		if problem != nil {
			return nil, problem
		}
		inst := records.PackageInstall{ID: installID, Package: doc.Str("package"), Major: major, Version: doc.Str("release"),
			SourceKind: "wheel", SourceRef: dir, SourceDigest: doc.Str("source_digest"), Dir: dir, ProjectDir: dir,
			Python: pkg.Capture.Python, Platform: pkg.Capture.Platform, Closure: pkg.Capture.Closure, Extra: pkg.Capture.Extra,
			PackageInterface: b.Capsule.Surfaces[id].Digest, LockDigest: id, Packages: len(strings.Split(pkg.Capture.Closure, "\n"))}
		paths := append([]string(nil), b.Wheels[id]...)
		for index, path := range paths {
			identity, problem := wheel.InspectIdentity(path)
			if problem != nil {
				return nil, problem
			}
			if identity.Distribution == strings.TrimPrefix(inst.Package, "local/") && identity.Version == inst.Version {
				paths[0], paths[index] = paths[index], paths[0]
				break
			}
		}
		revision, problem := localpackage.StageWheels(layout, inst, pkg.Interface, paths)
		if problem != nil {
			return nil, problem
		}
		if revision.Digest != id {
			return nil, invalid("imported package differs from its captured revision")
		}
		if problem := writeImmutable(launch.PackageInterfacePath(dir), pkg.Interface); problem != nil {
			return nil, problem
		}
		if problem := writeImmutable(filepath.Join(dir, "private-revision"), []byte(id)); problem != nil {
			return nil, problem
		}
		prior, problem := store.Install(installID)
		if problem != nil {
			return nil, problem
		}
		if prior != nil {
			prior.CreatedAt = ""
			if !reflect.DeepEqual(*prior, inst) {
				return nil, invalid("an imported capture changed its existing install facts")
			}
		} else if problem := store.RecordInstall(inst); problem != nil {
			return nil, problem
		}
		installs[id] = inst
	}
	bindings := make([]records.ChildBinding, 0, len(b.Capsule.Capsule.Bindings))
	for _, binding := range b.Capsule.Capsule.Bindings {
		bindings = append(bindings, records.ChildBinding{ParentInstallID: installs[binding.ParentRevision].ID,
			ChildInstallID: installs[binding.ChildRevision].ID, LocalRevisionDigest: binding.ChildRevision,
			InterfaceDigest: binding.InterfaceDigest, Module: binding.Module, Export: binding.Export, Entrypoint: binding.Entrypoint})
	}
	if problem := store.RecordChildBindings(bindings); problem != nil {
		return nil, problem
	}
	root := installs[b.Capsule.Capsule.Root.Revision]
	return &root, nil
}

func writeImmutable(path string, data []byte) *exit.Error {
	if old, err := os.ReadFile(path); err == nil {
		if bytes.Equal(old, data) {
			return nil
		}
		return invalid("imported metadata conflicts with an existing capture")
	} else if !os.IsNotExist(err) {
		return invalid("existing capture metadata is unavailable")
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return invalid("cannot create capture metadata directory")
	}
	file, err := os.CreateTemp(dir, ".capture-")
	if err != nil {
		return invalid("cannot stage capture metadata")
	}
	defer os.Remove(file.Name())
	_, writeErr := file.Write(data)
	modeErr, syncErr, closeErr := file.Chmod(0400), file.Sync(), file.Close()
	if writeErr != nil || modeErr != nil || syncErr != nil || closeErr != nil || os.Rename(file.Name(), path) != nil {
		return invalid("cannot commit capture metadata")
	}
	directory, err := os.Open(dir)
	if err != nil {
		return invalid("cannot open capture metadata directory")
	}
	defer directory.Close()
	if directory.Sync() != nil {
		return invalid("cannot sync capture metadata directory")
	}
	return nil
}

// RecordAdmission follows the ordinary durable request insert. Host acceptance
// and process startup alone never create this receipt.
func (b *Bootstrap) RecordAdmission(requestID string) *exit.Error {
	if requestID == "" {
		return invalid("execution root receipt is incomplete")
	}
	raw, err := json.Marshal(map[string]any{"request_id": requestID, "capsule_digest": b.Capsule.Digest,
		"record_owner_id": b.Grant.Grant.RecordOwnerId, "record_owner_epoch": b.Grant.Grant.RecordOwnerEpoch})
	if err != nil {
		return invalid("execution root receipt cannot be encoded")
	}
	return writeImmutable(filepath.Join(b.Authority.CreatorHome, "initial-root.json"), raw)
}

// PinGeneration precedes opening/reconciling the Creator journal. A different
// grant may not adopt requests merely by pointing at an existing coordinator home.
func (b *Bootstrap) PinGeneration() *exit.Error {
	path := filepath.Join(b.Authority.CreatorHome, "execution-grant.json")
	if _, err := os.Lstat(path); os.IsNotExist(err) {
		if _, err := os.Lstat(filepath.Join(b.Authority.CreatorHome, "creator.sqlite")); err == nil || !os.IsNotExist(err) {
			return invalid("private execution home contains an unbound records database")
		}
	}
	raw, err := canonical.Bytes(b.Grant.Grant)
	if err != nil {
		return invalid("execution generation cannot be identified")
	}
	return writeImmutable(path, raw)
}
