package install

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// LockedRequirementsFile is the release's exact export — index directives plus
// hash-pinned rows — written at environment materialization and re-consumed by Runtime
// preparation, including a later model selection. It replaces the retired
// package-preparation.json wheel inventory (wire 30: wheel facts stop being install
// inputs; the export's hashes are the authority).
const LockedRequirementsFile = "locked-requirements.txt"

func hasServingModelSlots(packageInterface *launch.PackageInterface) bool {
	for i := range packageInterface.Entrypoints {
		if len(packageInterface.Entrypoints[i].Models) > 0 {
			return true
		}
	}
	return false
}

func hasWeightlessCallable(packageInterface *launch.PackageInterface) bool {
	for i := range packageInterface.Entrypoints {
		if len(packageInterface.Entrypoints[i].Models) == 0 {
			return true
		}
	}
	for i := range packageInterface.Jobs {
		if len(packageInterface.Jobs[i].Models) == 0 {
			return true
		}
	}
	return false
}

// preparePublished materializes the release's complete frozen uv environment and asks THIS
// host's Runtime to admit it and author its resident placement. The surface is READ from the
// release's own committed package interface; nothing here re-derives it.
func preparePublished(l home.Layout, installDir string, published *PublishedSource) (
	*launch.PackageInterface, ExactDocument, string, *EnvironmentReceipt, *exit.Error,
) {
	var empty ExactDocument
	sourceDir := filepath.Join(installDir, "source")
	if err := os.MkdirAll(sourceDir, 0o700); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot create package metadata directory: %s", err)
	}
	for _, item := range []struct {
		name     string
		document ExactDocument
	}{{"package.toml", published.PackageConfig}, {"pyproject.toml", published.Pyproject},
		{"uv.lock", published.UVLock}} {
		name, document := item.name, item.document
		if err := os.WriteFile(filepath.Join(sourceDir, name), document.Bytes, 0o400); err != nil {
			return nil, empty, "", nil, exit.Internalf("cannot retain exact %s: %s", name, err)
		}
	}
	packageInterfacePath := launch.PackageInterfacePath(installDir)
	if err := os.MkdirAll(filepath.Dir(packageInterfacePath), 0o700); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot create package interface directory: %s", err)
	}
	if err := os.WriteFile(packageInterfacePath,
		published.Selection.PackageInterface.Bytes, 0o600); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot retain exact package interface: %s", err)
	}
	cache := filepath.Join(installDir, "artifact-cache")
	if err := os.MkdirAll(cache, 0o700); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot create package artifact cache: %s", err)
	}
	venvDir := filepath.Join(installDir, "venv")
	environment, problem := MaterializePublishedEnvironment(sourceDir, venvDir, published)
	if problem != nil {
		return nil, empty, "", nil, problem
	}
	runtimeBin := home.VenvTool(venvDir, "cozy-runtime")
	if info, err := os.Stat(runtimeBin); err != nil || !info.Mode().IsRegular() {
		return nil, empty, "", nil, exit.Named(exit.Structural, "runtime_missing",
			"the published package environment provides no cozy-runtime").
			WithRemedy("declare cozy-runtime in pyproject.toml and refresh uv.lock")
	}
	// DECISION #713: the publisher derives the interface and the committed document is the
	// truth. An install has no module tree to re-derive it from — `source/` holds
	// package.toml, pyproject.toml and uv.lock, nothing else — and re-deriving buys nothing:
	// the interface is not a security boundary (Runtime re-enforces every declared bound,
	// slot and component use at execution), and `cozy package publish` already refuses a
	// release whose own static describe disagrees with the document it commits. Asking the
	// environment's Runtime to describe here is what made every 0.5.2+ install fail
	// ("static_module: module 'h3_tables.job' is not a file under .../source"), and before
	// 0.5.2 it answered by IMPORTING the package, which #713 forbids outright.
	packageInterface, problem := launch.DecodePackageInterface(
		published.Selection.PackageInterface.Bytes)
	if problem != nil || packageInterface.Digest != published.Selection.PackageInterface.Digest {
		return nil, empty, "", nil, exit.Named(exit.Structural, "package_interface_invalid",
			"the release commits an invalid package interface")
	}
	// Installing code is not permission to run its imports or constructors.
	// Serving selections are prepared by the claimed worker before activation.
	return packageInterface, empty, runtimeBin, environment, nil
}

// PreparePublishedSelection turns one installed code/environment tree plus exact
// invocation-selected model Manifests into Runtime's immutable PlacementSet. It never
// resolves a human model ref and never downloads bytes; those are Creator's preceding
// control/transfer steps. Repeating it with the same inputs returns the same digest.
func PreparePublishedSelection(l home.Layout, inst records.PackageInstall,
	models []PublishedModel,
) (ExactDocument, *exit.Error) {
	var empty ExactDocument
	if inst.SourceKind != "tensorhub" || inst.Runtime == "" || len(models) == 0 {
		return empty, exit.Named(exit.Structural, "package_model_selection_incomplete",
			"install %s has no complete published model selection", inst.ID).
			WithRemedy("select one exact model for every callable slot")
	}
	if _, err := os.Stat(filepath.Join(inst.Dir, LockedRequirementsFile)); err != nil {
		return empty, exit.Named(exit.Conflict, "package_code_preparation_missing",
			"%s was installed without its locked-requirements export: %s", inst.Package, err).
			WithRemedy("reinstall the package; model weights are not required for reinstall")
	}
	packageInterfaceBytes, err := os.ReadFile(launch.PackageInterfacePath(inst.Dir))
	if err != nil {
		return empty, exit.New(exit.NotFound, "cannot read installed package interface: %s", err)
	}
	packageInterfaceDigest, _ := canonical.Spell(canonical.Digest(packageInterfaceBytes))
	if packageInterfaceDigest != inst.PackageInterface {
		return empty, exit.Named(exit.Conflict, "package_interface_changed",
			"installed package interface does not match %s", inst.PackageInterface)
	}
	published := &PublishedSource{Package: inst.Package, Release: inst.Version,
		Models: append([]PublishedModel(nil), models...), Selection: Selection{
			PackageInterface: ExactDocument{Bytes: packageInterfaceBytes, Digest: packageInterfaceDigest,
				Length: int64(len(packageInterfaceBytes))},
		}}
	return runPublishedSelection(l, inst, published)
}

func runPublishedSelection(l home.Layout, inst records.PackageInstall,
	published *PublishedSource,
) (ExactDocument, *exit.Error) {
	var empty ExactDocument
	answer, problem := preparePackageSet(l, inst.Dir, published)
	if problem != nil {
		// The runtime's refusal is the run's error, unrelabelled: nothing on this path
		// evaluates the checkpoint, so nothing here may call it a fit (model-code-fit §3).
		return empty, problem
	}
	if answer.Package != inst.Package || answer.Release != inst.Version ||
		!bytes.Equal(answer.PackageInterface.Bytes, published.Selection.PackageInterface.Bytes) {
		return empty, exit.Named(exit.Structural, "package_prepare_invalid",
			"cozy-runtime returned an invalid selected package preparation")
	}
	set, readErr := canonical.Read(answer.PlacementSet.Bytes, &pb.PlacementSet{})
	if readErr != nil || len(set.List("placements")) != 1 ||
		len(set.List("placements")[0].List("entrypoints")) == 0 {
		return empty, exit.Named(exit.Structural, "package_model_selection_empty",
			"cozy-runtime selected no runnable entrypoint for the requested model slots")
	}
	path := filepath.Join(inst.Dir, "artifact-cache",
		strings.TrimPrefix(answer.PlacementSet.Digest, "sha256:"))
	if err := os.WriteFile(path, answer.PlacementSet.Bytes, 0o600); err != nil {
		return empty, exit.Internalf("cannot retain selected package placement: %s", err)
	}
	return answer.PlacementSet, nil
}

type packagePreparation struct {
	Package          string        `json:"package"`
	Release          string        `json:"release"`
	PackageInterface ExactDocument `json:"package_interface"`
	PlacementSet     ExactDocument `json:"placement_set"`
}

// preparePackageSet runs THIS host's Runtime over the install's venv: it admits every exact
// wheel against this machine (interpreter seat, loader namespace, GPU), has the venv's own
// pinned Runtime describe the package in an executor child, and authors the PlacementSet.
// Admission is the installer's judgement, not the package's: the same host Runtime re-admits
// the venv when it serves it (launch.Spec), and an admission fix reaches every installed
// package at once instead of waiting for each release to re-lock a newer cozy-runtime. The
// venv's own Runtime running this verb judged its own site-packages as "the base" and refused
// every native dependency wheel it had just installed (hf_xet.abi3.so "already belongs to the
// selected base"). The Runtime's typed refusal is returned as itself (launch.RuntimeExit);
// this machine's own problems (no host Runtime, an unreadable answer) carry their own names.
func preparePackageSet(l home.Layout, installDir string, published *PublishedSource,
) (answer packagePreparation, problem *exit.Error) {
	var empty packagePreparation
	runtimeBin, problem := hostruntime.Path(config.Frozen().Tool())
	if problem != nil {
		return empty, problem
	}
	args := []string{"--json", "prepare-package",
		"--tensorfs-root", config.Frozen().TensorFSRoot,
		"--package", published.Package,
		"--release", published.Release,
		"--locked-requirements", filepath.Join(installDir, LockedRequirementsFile),
		"--artifact-cache", filepath.Join(installDir, "artifact-cache"),
		"--environment-python", home.VenvPython(filepath.Join(installDir, "venv")),
	}
	for _, model := range published.Models {
		raw, err := json.Marshal(model)
		if err != nil {
			return empty, exit.Internalf("cannot encode selected package model: %s", err)
		}
		args = append(args, "--model", string(raw))
	}
	cmd := exec.Command(runtimeBin, args...)
	cmd.Env = config.Frozen().Tool("COZY_HOME=" + runtimeScratchHome())
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return empty, exit.Internalf("cannot run %s: %s", runtimeBin, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return empty, hostruntime.RuntimeExit(code, "prepare-package", "runtime_preparation_failed",
			stdout.String(), stderr.String())
	}
	decoder := json.NewDecoder(strings.NewReader(stdout.String()))
	decoder.DisallowUnknownFields()
	decodeErr := decoder.Decode(&answer)
	var trailing any
	if decodeErr == nil {
		decodeErr = decoder.Decode(&trailing)
	}
	if decodeErr != io.EOF || answer.Package != published.Package ||
		answer.Release != published.Release ||
		!validRuntimeExact(answer.PackageInterface) || !validRuntimeExact(answer.PlacementSet) {
		return empty, exit.Named(exit.Structural, "package_prepare_invalid",
			"cozy-runtime returned an invalid package preparation result")
	}
	return answer, nil
}

func validRuntimeExact(value ExactDocument) bool {
	digest, err := canonical.Raw(value.Digest)
	return err == nil && value.Length == int64(len(value.Bytes)) &&
		bytes.Equal(canonical.Digest(value.Bytes), digest)
}
