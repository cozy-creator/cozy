package install

import (
	"bytes"
	"crypto/sha256"
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
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// PublishedPreparationFile retains the bounded exact wheel inventory already verified
// during code installation. It is not an identity document or executable authority:
// Runtime re-verifies every wheel and authors the exact PlacementSet only after an
// invocation selects model Manifests. A fixed file avoids rediscovering install files.
const PublishedPreparationFile = "package-preparation.json"

type publishedPreparation struct {
	Package      string           `json:"package"`
	Release      string           `json:"release"`
	ProjectWheel PublishedWheel   `json:"project_wheel"`
	Wheels       []PublishedWheel `json:"wheels"`
}

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

// preparePublished materializes the release's complete frozen uv environment, asks that
// environment's Runtime to describe the surface the release itself pinned, and asks THIS
// host's Runtime to admit the environment and author its resident placement. Creator
// compares the derived package interface with the committed publication interface.
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
	setDir := filepath.Join(cache, "sets", "package")
	if err := os.MkdirAll(setDir, 0o700); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot create package wheel cache: %s", err)
	}
	// The set-rooted interface copy: prepare-package admits the PACKAGE_INTERFACE row only
	// at `<set>/metadata/package-interface.json`; the documents/ copy above serves launch.
	staged := launch.StagedPackageInterfacePath(installDir)
	if err := os.MkdirAll(filepath.Dir(staged), 0o700); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot create the staged interface directory: %s", err)
	}
	if err := os.WriteFile(staged, published.Selection.PackageInterface.Bytes, 0o600); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot stage the exact package interface: %s", err)
	}
	if problem := stagePublishedWheel(cache, setDir, &published.ProjectWheel); problem != nil {
		return nil, empty, "", nil, problem
	}
	for index := range published.Wheels {
		if problem := stagePublishedWheel(cache, setDir, &published.Wheels[index]); problem != nil {
			return nil, empty, "", nil, problem
		}
	}
	for index := range published.LocalWheels {
		if problem := stagePublishedWheel(cache, setDir, &published.LocalWheels[index]); problem != nil {
			return nil, empty, "", nil, problem
		}
	}
	venvDir := filepath.Join(installDir, "venv")
	environment, problem := MaterializePublishedEnvironment(sourceDir, venvDir, l.Companions,
		published.ProjectWheel, published.Wheels, published.LocalWheels)
	if problem != nil {
		return nil, empty, "", nil, problem
	}
	runtimeBin := home.VenvTool(venvDir, "cozy-runtime")
	if info, err := os.Stat(runtimeBin); err != nil || !info.Mode().IsRegular() {
		return nil, empty, "", nil, exit.Named(exit.Structural, "runtime_missing",
			"the published package environment provides no cozy-runtime").
			WithRemedy("declare cozy-runtime in pyproject.toml and refresh uv.lock")
	}
	packageInterface, problem := describePublished(runtimeBin, sourceDir,
		published.Selection.PackageInterface)
	if problem != nil {
		if published.ReportDefect != nil && problem.Name == "package_interface_mismatch" {
			published.ReportDefect("package_prepare_interface_disagrees",
				"local describe derived a different package interface than the committed release")
		}
		return nil, empty, "", nil, problem
	}
	deferredModels := len(published.Models) == 0 && hasServingModelSlots(packageInterface)
	if deferredModels {
		inventory, err := json.Marshal(publishedPreparation{Package: published.Package,
			Release:      published.Release,
			ProjectWheel: published.ProjectWheel, Wheels: published.Wheels})
		if err != nil {
			return nil, empty, "", nil, exit.Internalf("cannot encode package preparation: %s", err)
		}
		if err := os.WriteFile(filepath.Join(installDir, PublishedPreparationFile), inventory, 0o400); err != nil {
			return nil, empty, "", nil, exit.Internalf(
				"cannot retain code-only package preparation: %s", err)
		}
		if !hasWeightlessCallable(packageInterface) {
			return packageInterface, empty, runtimeBin, environment, nil
		}
	}

	for _, wheel := range published.LocalWheels {
		if err := os.Remove(wheel.Path); err != nil {
			return nil, empty, "", nil, exit.Internalf(
				"cannot remove local materialization-wheel view: %s", err)
		}
	}
	answer, problem := preparePackageSet(l, installDir, published)
	if problem != nil {
		return nil, empty, "", nil, problem
	}
	if !bytes.Equal(answer.PackageInterface.Bytes, published.Selection.PackageInterface.Bytes) {
		if published.ReportDefect != nil {
			published.ReportDefect("package_prepare_interface_disagrees",
				"local preparation derived a different package interface than the committed release")
		}
		return nil, empty, "", nil, exit.Named(exit.Conflict, "package_interface_mismatch",
			"the installed package describes a different callable surface than its committed release")
	}
	packageInterface, problem = launch.DecodePackageInterface(answer.PackageInterface.Bytes)
	if problem != nil || packageInterface.Digest != answer.PackageInterface.Digest {
		return nil, empty, "", nil, exit.Named(exit.Structural, "package_interface_invalid",
			"cozy-runtime returned an invalid package interface")
	}
	set, readErr := canonical.Read(answer.PlacementSet.Bytes, &pb.PlacementSet{})
	if readErr != nil || len(set.List("placements")) != 1 {
		return nil, empty, "", nil, exit.Named(exit.Structural, "package_placement_invalid",
			"cozy-runtime returned an invalid package PlacementSet")
	}
	prepared := set.List("placements")[0]
	fact := prepared.Sub("package")
	if fact.Str("package") != published.Package || fact.Str("release") != published.Release ||
		fact.Sub("project_wheel").Sub("ref").Str("digest") != published.ProjectWheel.Digest {
		return nil, empty, "", nil, exit.Named(exit.Conflict, "package_placement_mismatch",
			"cozy-runtime prepared different package bytes than Creator downloaded")
	}
	placementPath := filepath.Join(cache, strings.TrimPrefix(answer.PlacementSet.Digest, "sha256:"))
	if err := os.WriteFile(placementPath, answer.PlacementSet.Bytes, 0o600); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot store prepared package placement: %s", err)
	}
	if deferredModels {
		// Keep the bounded wheel views for a modeled invocation. This package also has a
		// weightless callable, so its code-only PlacementSet remains immediately runnable.
		return packageInterface, answer.PlacementSet, runtimeBin, environment, nil
	}
	selectedDependencies := map[string]bool{}
	for _, wheel := range prepared.Sub("environment").List("wheels") {
		selectedDependencies[wheel.Sub("ref").Str("digest")] = true
	}
	if err := os.Remove(published.ProjectWheel.Path); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot remove prepared project-wheel view: %s", err)
	}
	for _, wheel := range published.Wheels {
		if runtimePreparationDependency(wheel) && !selectedDependencies[wheel.Digest] {
			return nil, empty, "", nil, exit.Named(exit.Structural,
				"package_placement_incomplete",
				"cozy-runtime omitted published dependency %s from the local package placement",
				wheel.Filename)
		}
		if err := os.Remove(wheel.Path); err != nil {
			return nil, empty, "", nil, exit.Internalf(
				"cannot remove prepared dependency-wheel view: %s", err)
		}
	}
	// The placement records the exact wheel subset used by a rental base. The local
	// environment receipt above records the complete frozen closure this machine runs.
	return packageInterface, answer.PlacementSet, runtimeBin, environment, nil
}

func describePublished(runtimeBin, sourceDir string, committed ExactDocument) (
	*launch.PackageInterface, *exit.Error,
) {
	cmd := exec.Command(runtimeBin, "--json", "--dir", sourceDir, "describe")
	cmd.Env = config.Frozen().Tool()
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return nil, exit.Internalf("cannot run %s: %s", runtimeBin, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return nil, launch.RuntimeExit(code, "describe", "runtime_query_failed",
			stdout.String(), stderr.String())
	}
	raw := bytes.TrimSuffix([]byte(stdout.String()), []byte("\n"))
	if !bytes.Equal(raw, committed.Bytes) {
		return nil, exit.Named(exit.Conflict, "package_interface_mismatch",
			"the installed package describes a different callable surface than its committed release")
	}
	packageInterface, problem := launch.DecodePackageInterface(raw)
	if problem != nil || packageInterface.Digest != committed.Digest {
		return nil, exit.Named(exit.Structural, "package_interface_invalid",
			"cozy-runtime returned an invalid installed package interface")
	}
	return packageInterface, nil
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
	seedBytes, err := os.ReadFile(filepath.Join(inst.Dir, PublishedPreparationFile))
	if err != nil {
		return empty, exit.Named(exit.Conflict, "package_code_preparation_missing",
			"%s was installed without reusable code-only preparation: %s", inst.Package, err).
			WithRemedy("reinstall the package; model weights are not required for reinstall")
	}
	var inventory publishedPreparation
	decoder := json.NewDecoder(bytes.NewReader(seedBytes))
	decoder.DisallowUnknownFields()
	decodeErr := decoder.Decode(&inventory)
	var trailing any
	if decodeErr == nil {
		decodeErr = decoder.Decode(&trailing)
	}
	if decodeErr != io.EOF || inventory.Package != inst.Package ||
		inventory.Release != inst.Version ||
		len(inventory.Wheels) > 128 {
		return empty, exit.Named(exit.Conflict, "package_code_preparation_changed",
			"%s code-only preparation does not match install %s", inst.Package, inst.ID)
	}
	cache := filepath.Join(inst.Dir, "artifact-cache")
	setDir := filepath.Join(cache, "sets", "package")
	wheelFrom := func(wheel PublishedWheel) (PublishedWheel, *exit.Error) {
		digest, digestErr := canonical.Raw(wheel.Digest)
		path := filepath.Join(setDir, wheel.Filename)
		if digestErr != nil || len(digest) != 32 || wheel.Filename == "" ||
			filepath.Base(wheel.Filename) != wheel.Filename || wheel.Length <= 0 ||
			wheel.Path != path {
			return PublishedWheel{}, exit.Named(exit.Conflict,
				"package_code_preparation_invalid", "prepared package wheel is incomplete")
		}
		info, statErr := os.Stat(path)
		if statErr != nil || !info.Mode().IsRegular() || info.Size() != wheel.Length {
			return PublishedWheel{}, exit.Named(exit.Conflict,
				"package_code_wheel_missing", "prepared wheel %s is absent or changed", wheel.Filename).
				WithRemedy("reinstall the package code; no model download is required")
		}
		return wheel, nil
	}
	project, problem := wheelFrom(inventory.ProjectWheel)
	if problem != nil {
		return empty, problem
	}
	dependencies := make([]PublishedWheel, 0, len(inventory.Wheels))
	for _, value := range inventory.Wheels {
		wheel, problem := wheelFrom(value)
		if problem != nil {
			return empty, problem
		}
		dependencies = append(dependencies, wheel)
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
		ProjectWheel: project, Wheels: dependencies,
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
	runtimeBin, problem := launch.HostRuntime(config.Frozen().Tool())
	if problem != nil {
		return empty, problem
	}
	args := []string{"--json", "prepare-package",
		"--artifact-store", config.Frozen().TensorFSRoot,
		"--package", published.Package,
		"--release", published.Release,
		"--package-interface", launch.StagedPackageInterfacePath(installDir),
		"--project-wheel", published.ProjectWheel.Path,
		"--artifact-cache", filepath.Join(installDir, "artifact-cache"),
		"--environment-python", home.VenvPython(filepath.Join(installDir, "venv")),
	}
	for _, wheel := range published.Wheels {
		if runtimePreparationDependency(wheel) {
			args = append(args, "--dependency-wheel", wheel.Path)
		}
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
		return empty, launch.RuntimeExit(code, "prepare-package", "runtime_preparation_failed",
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

// The package venv runs the exact cozy-runtime wheel selected by uv.lock. Older releases may
// still carry it as a dependency wheel; passing it back to its own prepare-package command as
// package-owned would shadow the selected worker image. New releases carry it only in the local
// materialization lane. All ordinary dependency wheels remain explicit package inputs.
func runtimePreparationDependency(wheel PublishedWheel) bool {
	return wheel.Distribution != "cozy-runtime"
}

func stagePublishedWheel(cache, setDir string, wheel *PublishedWheel) *exit.Error {
	digest, err := canonical.Raw(wheel.Digest)
	if err != nil || len(digest) != 32 || wheel.Length <= 0 ||
		filepath.Base(wheel.Filename) != wheel.Filename || filepath.Base(wheel.Path) != wheel.Filename {
		return exit.Named(exit.Structural, "package_wheel_invalid",
			"downloaded package wheel metadata is invalid")
	}
	target := filepath.Join(cache, strings.TrimPrefix(wheel.Digest, "sha256:"))
	if problem := copyPublishedWheel(wheel.Path, target, digest, wheel.Length); problem != nil {
		return problem
	}
	view := filepath.Join(setDir, wheel.Filename)
	if err := os.Link(target, view); err != nil {
		if problem := copyPublishedWheel(target, view, digest, wheel.Length); problem != nil {
			return problem
		}
	}
	if err := os.Chmod(target, 0o400); err != nil {
		return exit.Internalf("cannot protect downloaded package wheel %s: %s", wheel.Filename, err)
	}
	if err := os.Chmod(view, 0o400); err != nil {
		return exit.Internalf("cannot protect staged package wheel %s: %s", wheel.Filename, err)
	}
	wheel.Path = view
	return nil
}

func copyPublishedWheel(source, target string, digest []byte, length int64) *exit.Error {
	input, err := os.Open(source)
	if err != nil {
		return exit.Named(exit.Conflict, "package_wheel_changed",
			"downloaded package wheel changed before installation")
	}
	defer input.Close()
	info, err := input.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() != length {
		return exit.Named(exit.Conflict, "package_wheel_changed",
			"downloaded package wheel changed before installation")
	}
	output, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return exit.Internalf("cannot retain downloaded package wheel: %s", err)
	}
	measured := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(output, measured), input)
	syncErr := output.Sync()
	closeErr := output.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil || written != length ||
		!bytes.Equal(measured.Sum(nil), digest) {
		_ = os.Remove(target)
		return exit.Named(exit.Conflict, "package_wheel_changed",
			"downloaded package wheel changed before installation")
	}
	return nil
}

func validRuntimeExact(value ExactDocument) bool {
	digest, err := canonical.Raw(value.Digest)
	return err == nil && value.Length == int64(len(value.Bytes)) &&
		bytes.Equal(canonical.Digest(value.Bytes), digest)
}
