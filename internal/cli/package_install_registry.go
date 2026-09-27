package cli

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
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
	_, existingInstall, problem := existing.ActivePackage(ref.String())
	if problem != nil {
		existing.Close()
		return nil, nil, problem
	}
	if problem := requirePackageUpdatePin(existing, ref.String(), expectedInstallID); problem != nil {
		existing.Close()
		return nil, nil, problem
	}
	if existingInstall != nil && existingInstall.SourceKind == "tensorhub" &&
		existingInstall.Package == ref.String() && existingInstall.Version == release {
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
	if problem := requirePackageUpdatePin(st, ref.String(), expectedInstallID); problem != nil {
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
	if problem == nil && (plan.Release == "" || release != "" && plan.Release != release) {
		problem = exit.Internalf("Tensorhub returned a changed or absent package release")
	}
	return ref, plan, problem
}

func requirePackageUpdatePin(st *records.Store, pkg, expected string) *exit.Error {
	if expected == "" {
		return nil
	}
	_, installed, problem := st.ActivePackage(pkg)
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

// acquirePublishedModel is LOCAL-FIRST. An exact TensorFS release row is sufficient
// authority to invoke an already-installed Manifest even if Tensorhub's catalog was
// reset or is offline. Only a local miss asks Tensorhub to resolve/download.
func acquirePublishedModel(ctx context.Context, cli *Context, tool *tfs.Tool,
	hubClient *hub.Client, spec, lane, packageName string, slot launch.Slot, work string,
) (install.PublishedModel, *exit.Error) {
	var empty install.PublishedModel
	if local, ok, problem := exactLocalModel(tool, spec, lane, work); problem != nil {
		return empty, problem
	} else if ok {
		return install.PublishedModel{Package: packageName, Slot: slot.Path, Model: local.Model,
			Release: local.Release, Lane: local.Lane, Manifest: local.Manifest,
			ManifestLength: local.ManifestLength, Reused: true}, nil
	}
	layout, problem := home.Open(cli.Cfg.Home)
	if problem != nil {
		return empty, problem
	}
	modelName, release, selectedLane, manifestPin, problem := hub.ParseModelRef(spec)
	if problem != nil {
		return empty, problem
	}
	if selectedLane != "" {
		if lane != "" && lane != selectedLane {
			return empty, exit.Usagef("model reference and selected lane disagree")
		}
		lane = selectedLane
	}
	refspec := modelName
	if release != "" {
		refspec += "@" + release
	} else if manifestPin != "" {
		refspec += "@" + manifestPin
	}
	fetch := &transfer.Fetch{Tool: tool, Hub: hubClient, Spec: refspec, Lane: lane,
		Progress: progress(cli), Scratch: work, Locks: layout.AcquisitionLocks()}
	resolved, problem := fetch.Resolve(ctx)
	if problem != nil {
		return empty, exit.Named(problem.Code, "model_resolution_unavailable",
			"cannot resolve model %s for slot %s: %s", spec, slot.Path, problem.Message).
			WithRemedy("download another compatible model or override this slot with model.%s=org/model@release", slot.Path[strings.LastIndex(slot.Path, ".")+1:])
	}
	if problem := requireCheckpointComponents(spec, slot, resolved.Components); problem != nil {
		return empty, problem
	}
	if manifestPin != "" && resolved.ManifestID != manifestPin {
		return empty, exit.Named(exit.Conflict, "model_resolution_changed", "resolved model differs from its checkpoint pin")
	}
	if cli.Inv.Bool("--dry-run") {
		var root []byte
		if resolved.Release == "" {
			root, problem = hubClient.CheckpointManifest(ctx, fetch.Ref, resolved.ManifestID)
		} else {
			root, problem = hubClient.ReleaseManifest(ctx, fetch.Ref, resolved.Release, resolved.Lane)
		}
		if problem != nil {
			return empty, problem
		}
		digest, err := canonical.Spell(canonical.Digest(root))
		if err != nil || len(root) == 0 || digest != resolved.ManifestID {
			return empty, exit.Named(exit.Conflict, "job.model_manifest_changed", "published model root differs from its selected checkpoint")
		}
		return install.PublishedModel{Package: packageName, Slot: slot.Path, Model: fetch.Ref.String(), Release: resolved.Release, Lane: resolved.Lane, Manifest: resolved.ManifestID, ManifestLength: int64(len(root))}, nil
	}
	fetched, problem := fetch.Acquire(ctx, resolved)
	if problem != nil {
		return empty, problem
	}
	manifest, err := canonical.Raw(fetched.ManifestID)
	if err != nil || len(manifest) != 32 || fetched.ManifestLength <= 0 ||
		(fetched.Release == "") != (fetched.Lane == "") || fetch.Ref.String() == "/" {
		return empty, exit.Named(exit.Structural, "model_download_result_invalid",
			"model acquisition returned an incomplete exact selection for package slot %s", slot.Path)
	}
	return install.PublishedModel{Package: packageName, Slot: slot.Path, Model: fetch.Ref.String(),
		Release: fetched.Release, Lane: fetched.Lane, Manifest: fetched.ManifestID,
		ManifestLength: fetched.ManifestLength, Reused: fetched.Moved == 0}, nil
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

type localModelSelection struct {
	Model, Release, Lane, Manifest string
	ManifestLength                 int64
}

func exactLocalModel(tool *tfs.Tool, spec, lane, work string) (
	localModelSelection, bool, *exit.Error,
) {
	var empty localModelSelection
	modelRelease, manifest, hasManifest := strings.Cut(strings.TrimSpace(spec), "#")
	modelName, release, hasRelease := strings.Cut(modelRelease, "@")
	if !hasRelease && hasManifest && lane == "" {
		ref, problem := hub.ParseRef(modelName)
		if problem != nil {
			return empty, false, problem
		}
		if raw, err := canonical.Raw(manifest); err != nil || len(raw) != 32 {
			return empty, false, exit.Usagef("%q is not an exact model Manifest", manifest)
		}
		if err := os.MkdirAll(work, 0o700); err != nil {
			return empty, false, exit.Internalf("cannot create model lookup scratch: %s", err)
		}
		length, problem := tool.RetainedCheckpoint(ref.Org, ref.Name, manifest, work)
		if problem != nil || length == 0 {
			return empty, false, problem
		}
		return localModelSelection{Model: ref.String(), Manifest: manifest, ManifestLength: length}, true, nil
	}
	if !hasRelease || release == "" || hasManifest && manifest == "" {
		return empty, false, nil
	}
	ref, problem := hub.ParseRef(modelName)
	if problem != nil {
		return empty, false, problem
	}
	if manifest != "" {
		if raw, err := canonical.Raw(manifest); err != nil || len(raw) != 32 {
			return empty, false, exit.Usagef("%q is not an exact model Manifest", manifest)
		}
	}
	if err := os.MkdirAll(work, 0o700); err != nil {
		return empty, false, exit.Internalf("cannot create model lookup scratch: %s", err)
	}
	rows, problem := tool.Releases(filepath.Join(work, "local-releases.jsonl"))
	if problem != nil {
		return empty, false, problem
	}
	matches := make([]localModelSelection, 0, 1)
	for _, row := range rows {
		rowManifest := "sha256:" + row.ManifestSHA256
		if row.Org == ref.Org && row.Name == ref.Name && row.Version == release &&
			(lane == "" || row.Lane == lane) && (manifest == "" || rowManifest == manifest) {
			matches = append(matches, localModelSelection{Model: ref.String(), Release: release,
				Lane: row.Lane, Manifest: rowManifest, ManifestLength: row.ManifestLength})
		}
	}
	if len(matches) == 1 {
		return matches[0], true, nil
	}
	// Zero or ambiguous local rows defer to Tensorhub's catalog resolver. In particular,
	// never guess between two lanes merely because both are on disk.
	return empty, false, nil
}
