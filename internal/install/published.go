package install

import (
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/launch"
)

// LockedRequirementsFile is the release's exact export — index directives plus
// hash-pinned rows — written at environment materialization and re-consumed by Runtime
// preparation, including a later model selection. It replaces the retired
// package-preparation.json wheel inventory (wire 30: wheel facts stop being install
// inputs; the export's hashes are the authority).
const LockedRequirementsFile = "locked-requirements.txt"

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
	// The publisher's patch is provenance; its wheel closure needs only the minor ABI.
	if minor := hostruntime.PythonMinor(published.PythonVersion); minor != "" {
		if err := os.WriteFile(filepath.Join(sourceDir, ".python-version"), []byte(minor+"\n"), 0o400); err != nil {
			return nil, empty, "", nil, exit.Internalf("cannot retain published Python selection: %s", err)
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
	if problem != nil {
		return nil, empty, "", nil, exit.Named(exit.Structural, "package_interface_invalid",
			"the release commits an invalid package interface: %s", problem.Message)
	}
	// Installing code is not permission to run its imports or constructors.
	// Serving selections are prepared by the claimed worker before activation.
	return packageInterface, empty, runtimeBin, environment, nil
}
