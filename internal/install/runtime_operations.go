package install

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/runtimeoperation"
)

// RuntimeOperations captures a base-owned App without copying a caller's code or
// promoting its observed dependency versions into the builtin's requirements.
// Older SDKs simply have no builtin; their existing private functions still work.
func RuntimeOperations(ctx context.Context, layout home.Layout, store *records.Store) (*records.ChildBinding, string, *exit.Error) {
	env := config.Frozen().Tool("COZY_HOME=" + runtimeScratchHome())
	tool, problem := launch.BuiltinOperationsTool(layout.Root, runtimeScratchHome(), env)
	if problem != nil {
		return nil, "", problem
	}
	version, problem := tool.RuntimeVersion(ctx)
	if problem != nil {
		return nil, "", problem
	}
	parsed, err := pep440.Parse(version)
	if err != nil {
		return nil, "", exit.New(exit.Conflict, "Runtime returned an invalid builtin version")
	}
	if parsed.LessThan(pep440.MustParse(runtimeoperation.Floor)) {
		return nil, "", nil
	}
	surface, problem := tool.BuiltinOperations(ctx)
	if problem != nil {
		return nil, "", problem
	}
	filename, wheel, problem := runtimeoperation.Carrier(version, surface.Digest)
	if problem != nil {
		return nil, "", problem
	}
	id, problem := newInstallID()
	if problem != nil {
		return nil, "", problem
	}
	dir := layout.InstallDir(id)
	if err := os.MkdirAll(filepath.Join(dir, "wheels"), 0700); err != nil {
		return nil, "", exit.Internalf("cannot stage Runtime operations: %s", err)
	}
	fail := func(problem *exit.Error) (*records.ChildBinding, string, *exit.Error) {
		_ = os.RemoveAll(dir)
		return nil, "", problem
	}
	carrier := filepath.Join(dir, "wheels", filename)
	if err := os.WriteFile(carrier, wheel, 0400); err != nil {
		return fail(exit.Internalf("cannot retain Runtime carrier: %s", err))
	}
	prepared, problem := tool.PrepareBuiltin(ctx, carrier, filepath.Join(dir, "builtin"))
	if problem != nil {
		return fail(problem)
	}
	actual, problem := launch.DecodePackageInterface(prepared.PackageInterface)
	if problem != nil {
		return fail(problem)
	}
	if actual.Digest != surface.Digest || !bytes.Equal(actual.Raw, surface.Raw) || prepared.RuntimeVersion != version {
		return fail(exit.New(exit.Conflict, "Runtime changed its builtin while preparing it"))
	}
	for _, digest := range []string{prepared.EnvironmentDigest, prepared.ContentDigest, prepared.ReceiptDigest, prepared.ImplementationDigest} {
		if _, err := canonical.Raw(digest); err != nil {
			return fail(exit.New(exit.Conflict, "Runtime builtin preparation omitted exact identities"))
		}
	}
	carrierDigest, _ := canonical.Spell(canonical.Digest(wheel))
	identity, _ := json.Marshal(map[string]string{"carrier_digest": carrierDigest, "interface_digest": surface.Digest, "implementation_digest": prepared.ImplementationDigest})
	identity, err = canonical.NormalizeJCS(identity)
	if err != nil {
		return fail(exit.Internalf("cannot identify Runtime builtin: %s", err))
	}
	sourceDigest, _ := canonical.Spell(canonical.Digest(identity))
	metadata := runtimeoperation.Environment{Python: prepared.EnvironmentPython, EnvironmentDigest: prepared.EnvironmentDigest, InterfaceDigest: surface.Digest, SourceDigest: sourceDigest}
	raw, _ := json.Marshal(metadata)
	if err := os.WriteFile(filepath.Join(dir, runtimeoperation.EnvironmentFile), raw, 0400); err != nil {
		return fail(exit.Internalf("cannot retain builtin environment: %s", err))
	}
	if _, problem := runtimeoperation.ReadEnvironment(dir, prepared.EnvironmentDigest, surface.Digest, sourceDigest); problem != nil {
		return fail(problem)
	}
	interfacePath := launch.PackageInterfacePath(dir)
	if err := os.MkdirAll(filepath.Dir(interfacePath), 0700); err != nil {
		return fail(exit.Internalf("cannot retain builtin interface: %s", err))
	}
	if err := os.WriteFile(interfacePath, surface.Raw, 0400); err != nil {
		return fail(exit.Internalf("cannot retain builtin interface: %s", err))
	}
	if prepared.Closure == "" || !strings.Contains("\n"+prepared.Closure+"\n", "\ncozy-runtime=="+version+"\n") {
		return fail(exit.New(exit.Conflict, "builtin environment omitted actual Runtime distribution"))
	}
	python := pythonVersion(filepath.Dir(filepath.Dir(prepared.EnvironmentPython)))
	if python == "" {
		return fail(exit.New(exit.Conflict, "builtin interpreter identity is unobserved"))
	}
	inst := records.PackageInstall{ID: id, Package: "local/" + runtimeoperation.Name, Version: version, SourceKind: "wheel", SourceRef: dir, SourceDigest: sourceDigest, Dir: dir, ProjectDir: dir, Runtime: tool.Bin, Python: python, UV: toolVersion("uv", "--version"), LockDigest: prepared.EnvironmentDigest, Platform: runtime.GOOS + "/" + runtime.GOARCH, Closure: prepared.Closure, Packages: len(strings.Split(strings.TrimSpace(prepared.Closure), "\n")), PackageInterface: surface.Digest}
	inst.BytesExcl, inst.BytesShared = Disk(dir)
	if problem := store.RecordInstall(inst); problem != nil {
		return fail(problem)
	}
	revision, problem := localpackage.StageWheels(layout, inst, surface.Raw, []string{carrier})
	if problem != nil {
		return nil, id, problem
	}
	if err := os.WriteFile(filepath.Join(dir, "private-revision"), []byte(revision.Digest), 0400); err != nil {
		return nil, id, exit.Internalf("cannot retain builtin revision: %s", err)
	}
	return &records.ChildBinding{InterfaceDigest: surface.Digest, Module: runtimeoperation.Module, Export: "quantize", ChildInstallID: id, LocalRevisionDigest: revision.Digest, Entrypoint: "quantize"}, id, nil
}
