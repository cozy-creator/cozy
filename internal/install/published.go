package install

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// preparePublished asks the one trusted Runtime to install/import the exact small
// wheel overlay and author its resident routing facts. Creator only stages bytes and
// compares Runtime's descriptor with the committed publication descriptor.
func preparePublished(l home.Layout, genDir, runtimeBin string, published *PublishedSource) (
	*launch.PackageDescriptor, ExactDocument, string, *EnvironmentReceipt, *exit.Error,
) {
	var empty ExactDocument
	if runtimeBin == "" {
		return nil, empty, "", nil, exit.Internalf("published install has no trusted host Runtime")
	}
	sourceDir := filepath.Join(genDir, "source")
	if err := os.MkdirAll(sourceDir, 0o700); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot create package metadata directory: %s", err)
	}
	if err := os.WriteFile(filepath.Join(sourceDir, "package.toml"),
		published.PackageConfig.Bytes, 0o400); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot retain exact package.toml: %s", err)
	}
	cache := filepath.Join(genDir, "artifact-cache")
	setDir := filepath.Join(cache, "sets", "package")
	if err := os.MkdirAll(setDir, 0o700); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot create package wheel cache: %s", err)
	}
	if problem := stagePublishedWheel(cache, setDir, &published.ProjectWheel); problem != nil {
		return nil, empty, "", nil, problem
	}
	for index := range published.Wheels {
		if problem := stagePublishedWheel(cache, setDir, &published.Wheels[index]); problem != nil {
			return nil, empty, "", nil, problem
		}
	}

	args := []string{"--json", "prepare-package",
		"--artifact-store", l.CAS,
		"--package", published.Package,
		"--release", published.Release,
		"--release-digest", published.SourceDigest,
		"--project-wheel", published.ProjectWheel.Path,
		"--artifact-cache", cache,
		"--environment-root", filepath.Join(genDir, "environment"),
		"--base-manifest", l.LocalBase,
	}
	for _, wheel := range published.Wheels {
		args = append(args, "--dependency-wheel", wheel.Path)
	}
	for _, model := range published.Models {
		raw, err := json.Marshal(model)
		if err != nil {
			return nil, empty, "", nil, exit.Internalf("cannot encode selected package model: %s", err)
		}
		args = append(args, "--model", string(raw))
	}
	cmd := exec.Command(runtimeBin, args...)
	cmd.Env = config.Frozen().Tool("COZY_HOME=" + l.Root)
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if cmd.ProcessState == nil {
		return nil, empty, "", nil, exit.Internalf("cannot run %s: %s", runtimeBin, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return nil, empty, "", nil, metadataRefusal(code, "prepare-package", stderr.String())
	}
	var answer struct {
		Package           string        `json:"package"`
		Release           string        `json:"release"`
		PackageDescriptor ExactDocument `json:"package_descriptor"`
		PlacementSet      ExactDocument `json:"placement_set"`
	}
	decoder := json.NewDecoder(strings.NewReader(stdout.String()))
	decoder.DisallowUnknownFields()
	decodeErr := decoder.Decode(&answer)
	var trailing any
	if decodeErr == nil {
		decodeErr = decoder.Decode(&trailing)
	}
	if decodeErr != io.EOF || answer.Package != published.Package || answer.Release != published.Release ||
		!validRuntimeExact(answer.PackageDescriptor) || !validRuntimeExact(answer.PlacementSet) {
		return nil, empty, "", nil, exit.Named(exit.Structural, "package_prepare_invalid",
			"cozy-runtime returned an invalid package preparation result")
	}
	if !bytes.Equal(answer.PackageDescriptor.Bytes, published.Selection.PackageDescriptor.Bytes) {
		return nil, empty, "", nil, exit.Named(exit.Conflict, "descriptor_mismatch",
			"the installed package describes a different callable surface than its committed release")
	}
	descriptor, problem := launch.DecodeDescriptor(answer.PackageDescriptor.Bytes)
	if problem != nil || descriptor.Digest != answer.PackageDescriptor.Digest {
		return nil, empty, "", nil, exit.Named(exit.Structural, "package_descriptor_invalid",
			"cozy-runtime returned an invalid package descriptor")
	}
	set, readErr := canonical.Read(answer.PlacementSet.Bytes, &pb.PlacementSet{})
	if readErr != nil || len(set.List("placements")) != 1 {
		return nil, empty, "", nil, exit.Named(exit.Structural, "package_placement_invalid",
			"cozy-runtime returned an invalid package PlacementSet")
	}
	prepared := set.List("placements")[0]
	fact := prepared.Sub("package")
	if fact.Str("package") != published.Package || fact.Str("release") != published.Release ||
		fact.Str("release_digest") != published.SourceDigest ||
		fact.Sub("project_wheel").Sub("ref").Str("digest") != published.ProjectWheel.Digest {
		return nil, empty, "", nil, exit.Named(exit.Conflict, "package_placement_mismatch",
			"cozy-runtime prepared different package bytes than Creator downloaded")
	}
	placementPath := filepath.Join(cache, strings.TrimPrefix(answer.PlacementSet.Digest, "sha256:"))
	if err := os.WriteFile(placementPath, answer.PlacementSet.Bytes, 0o600); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot store prepared package placement: %s", err)
	}
	closure := []string{fact.Sub("project_wheel").Str("distribution") + "==" +
		fact.Sub("project_wheel").Str("version")}
	selectedDependencies := map[string]bool{}
	for _, wheel := range prepared.Sub("environment").List("wheels") {
		selectedDependencies[wheel.Sub("ref").Str("digest")] = true
		closure = append(closure, wheel.Str("distribution")+"=="+wheel.Str("version"))
	}
	if err := os.Remove(published.ProjectWheel.Path); err != nil {
		return nil, empty, "", nil, exit.Internalf("cannot remove prepared project-wheel view: %s", err)
	}
	for _, wheel := range published.Wheels {
		if selectedDependencies[wheel.Digest] {
			if err := os.Remove(wheel.Path); err != nil {
				return nil, empty, "", nil, exit.Internalf(
					"cannot remove prepared dependency-wheel view: %s", err)
			}
			continue
		}
		// Runtime proved this dependency is already owned by the observed base.
		// Its arriving carrier has no execution consumer and does not survive install.
		for _, path := range []string{wheel.Path,
			filepath.Join(cache, strings.TrimPrefix(wheel.Digest, "sha256:"))} {
			if err := os.Remove(path); err != nil {
				return nil, empty, "", nil, exit.Internalf(
					"cannot discard redundant base dependency %s: %s", wheel.Filename, err)
			}
		}
	}
	sort.Strings(closure)
	environment := &EnvironmentReceipt{
		Python: "CPython 3.12", Platform: "linux/amd64", LinkMode: "overlay",
		Packages: len(closure), Closure: strings.Join(closure, "\n"),
	}
	return descriptor, answer.PlacementSet, runtimeBin, environment, nil
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
