package cli

import (
	"bytes"
	"path/filepath"
	"slices"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
)

func handleRegistryInstall(ctx *Context) *exit.Error {
	result, cleanup, problem := installRegistryPackage(ctx, "")
	if problem != nil {
		return problem
	}
	return emitInstallResult(ctx, result, cleanup...)
}

// installRegistryPackage is the ordinary installer. A bulk update additionally
// pins the observed active install so a concurrent edit or removal wins safely.
func installRegistryPackage(ctx *Context, expectedInstallID string) (*install.Result, []output.Field, *exit.Error) {
	ref, plan, problem := resolveRegistryPackage(ctx, ctx.Inv.Args[0], ctx.Inv.Value("--version"))
	if problem != nil {
		return nil, nil, problem
	}
	packageConfig, problem := exactPackageInstallDocument("package.toml", plan.PackageConfig)
	if problem != nil {
		return nil, nil, problem
	}
	packageInterface, problem := exactPackageInstallDocument(
		"package interface", plan.PackageInterface)
	if problem != nil {
		return nil, nil, problem
	}
	pyproject, problem := exactPackageInstallDocument("pyproject.toml", plan.Pyproject)
	if problem != nil {
		return nil, nil, problem
	}
	uvLock, problem := exactPackageInstallDocument("uv.lock", plan.UVLock)
	if problem != nil {
		return nil, nil, problem
	}
	release := plan.Release
	_, existing, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return nil, nil, problem
	}
	_, existingInstall, problem := existing.ActivePackage(ctx.Cfg.HubURL, ref.String())
	if problem != nil {
		existing.Close()
		return nil, nil, problem
	}
	if problem := requirePackageUpdatePin(existing, ctx.Cfg.HubURL, ref.String(), expectedInstallID); problem != nil {
		existing.Close()
		return nil, nil, problem
	}
	if existingInstall != nil && existingInstall.SourceKind == "tensorhub" && existingInstall.Version == release {
		defer existing.Close()
		result := &install.Result{Install: *existingInstall, Idempotent: true}
		return result, nil, nil
	}
	existing.Close()
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, nil, problem
	}
	work, problem := scratch.Temp(layout.Tmp, "package-install-")
	if problem != nil {
		return nil, nil, problem
	}
	defer work.Release()
	published, problem := packageInstallPlanFacts(ctx, ref, release,
		plan, packageConfig, packageInterface, pyproject, uvLock)
	if problem != nil {
		return nil, nil, problem
	}
	_, st, writer, problem := open(ctx.Cfg, true)
	if problem != nil {
		return nil, nil, problem
	}
	if problem := requirePackageUpdatePin(st, ctx.Cfg.HubURL, ref.String(), expectedInstallID); problem != nil {
		st.Close()
		writer.Unlock()
		return nil, nil, problem
	}
	var result *install.Result
	problem = packagePublishStage(ctx, "Creating local package environment", func() *exit.Error {
		var installProblem *exit.Error
		result, installProblem = install.Run(layout, st, install.Request{
			Ref: install.Ref{Package: ref.String()}, Force: true, Published: published,
		})
		return installProblem
	})
	if problem != nil {
		st.Close()
		writer.Unlock()
		return nil, nil, problem
	}
	cleanup := reclaimInstallResult(layout, st, result)
	st.Close()
	writer.Unlock()
	return result, cleanup, nil
}

// resolveRegistryPackage is shared by local and rental installation. An empty
// release lets Tensorhub select the newest eligible published release; neither
// path consults installed editable sources or package model defaults.
func resolveRegistryPackage(ctx *Context, value, version string) (hub.Ref, hub.PackageDownloadPlan, *exit.Error) {
	ref, release, problem := registryPackageRef(value, version)
	if problem != nil {
		return ref, hub.PackageDownloadPlan{}, problem
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	packagePublishStatus(ctx, "Resolving %s...", ref.String())
	plan, problem := client(ctx).PackageDownloads(hctx, ref, release)
	if problem == nil && (plan.Release == "" || release != "" && !sameRelease(plan.Release, release)) {
		problem = exit.Internalf("Tensorhub answered release %q for requested %q", plan.Release, release)
	}
	return ref, plan, packageHubProblem(ctx, ref.String(), problem)
}

func requirePackageUpdatePin(st *records.Store, hub, pkg, expected string) *exit.Error {
	if expected == "" {
		return nil
	}
	_, installed, problem := st.ActivePackage(hub, pkg)
	if problem != nil {
		return problem
	}
	if installed == nil || installed.ID != expected || installed.SourceKind != "tensorhub" {
		return exit.Named(exit.Conflict, "package.update_changed", "%s changed after the bulk update began; its current selection was preserved", pkg)
	}
	return nil
}

func registryPackageRef(value, release string) (hub.Ref, string, *exit.Error) {
	name := strings.TrimSpace(value)
	if strings.Contains(name, "@") {
		return hub.Ref{}, "", exit.Usagef("a version does not belong in the package name").
			WithRemedy("use cozy package install org/package --version 1.2.3")
	}
	release = strings.TrimSpace(release)
	if strings.ContainsAny(release, `@/\`) {
		return hub.Ref{}, "", exit.Usagef("--version %q is not a release name", release)
	}
	ref, problem := hub.ParseRef(name)
	return ref, release, problem
}

func packageInstallPlanFacts(cli *Context, ref hub.Ref,
	release string, plan hub.PackageDownloadPlan, packageConfig,
	packageInterface, pyproject, uvLock install.ExactDocument,
) (*install.PublishedSource, *exit.Error) {
	if len(plan.Downloads) == 0 {
		return nil, exit.Internalf("Tensorhub returned an empty package install plan")
	}
	published := &install.PublishedSource{
		PythonVersion: plan.PythonVersion, Package: ref.String(), Release: release,
		PackageConfig: packageConfig,
		Pyproject:     pyproject,
		UVLock:        uvLock,
		Selection:     install.Selection{PackageInterface: packageInterface},
		IndexURL: strings.TrimRight(cli.Cfg.HubURL, "/") +
			"/v1/index/" + ref.Org + "/simple/",
		Hub: cli.Cfg.HubURL,
	}
	// Wire 30: the plan's rows are release wheel FACTS only. No wheel byte is downloaded;
	// the environment materializes from the locked-requirements export, whose hashes pin
	// every artifact against PyPI plus the org index.
	seen := map[string]hub.PackageInstallDownload{}
	for _, download := range plan.Downloads {
		switch download.Kind {
		case "project_wheel", "dependency_wheel", "local_materialization_wheel":
			if filepath.Base(download.Path) != download.Path ||
				!strings.HasSuffix(download.Path, ".whl") {
				return nil, exit.Internalf("Tensorhub returned unsafe package wheel path %q",
					download.Path)
			}
			if prior, ok := seen[download.Path]; ok {
				if prior.Digest != download.Digest || prior.Length != download.Length || prior.Kind != download.Kind {
					return nil, exit.Internalf("Tensorhub returned two different wheels named %s", download.Path)
				}
				continue
			}
			seen[download.Path] = download
			wheel := install.PublishedWheel{Digest: download.Digest,
				Distribution: download.Distribution, Filename: download.Path,
				ImportRoots: slices.Clone(download.ImportRoots),
				Length:      download.Length,
				Tags:        append([]string(nil), download.Tags...), Version: download.Version}
			switch download.Kind {
			case "project_wheel":
				if published.ProjectWheel.Digest != "" {
					return nil, exit.Internalf("Tensorhub returned more than one project wheel")
				}
				published.ProjectWheel = wheel
			case "dependency_wheel":
				published.Wheels = append(published.Wheels, wheel)
			default:
				published.LocalWheels = append(published.LocalWheels, wheel)
			}
		}
	}
	if published.ProjectWheel.Digest == "" {
		return nil, exit.Internalf("Tensorhub package install plan has no project wheel")
	}
	return published, nil
}

func exactPackageInstallDocument(name string, document hub.ExactDocument) (install.ExactDocument, *exit.Error) {
	digest, err := canonical.Raw(document.Digest)
	if err != nil || len(digest) != 32 || document.Length != int64(len(document.CanonicalBytes)) ||
		document.Length == 0 ||
		!bytes.Equal(canonical.Digest(document.CanonicalBytes), digest) {
		return install.ExactDocument{}, exit.Named(exit.Structural,
			"hub.package_install_document_invalid",
			"Tensorhub returned %s bytes that do not match their digest and length", name)
	}
	return install.ExactDocument{Bytes: append([]byte(nil), document.CanonicalBytes...),
		Digest: document.Digest, Length: document.Length}, nil
}

func requireCheckpointComponents(model string, slot launch.Slot, available []string) *exit.Error {
	missing := launch.MissingComponents(slot, available)
	if len(missing) == 0 {
		return nil
	}
	return exit.Named(exit.Validation, "checkpoint_component_missing",
		"model %s cannot satisfy %s; checkpoint is missing component(s): %s",
		model, slot.Path, strings.Join(missing, ", ")).
		WithRemedy("select a checkpoint whose component set includes %s", strings.Join(missing, ", "))
}

// sameRelease compares release labels by PEP 440 value, so 1.0.0rc1 and 1.0.0-rc1 agree.
func sameRelease(a, b string) bool {
	left, errA := pep440.Parse(a)
	right, errB := pep440.Parse(b)
	if errA != nil || errB != nil {
		return a == b
	}
	return left.Equal(right)
}
