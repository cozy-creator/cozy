package cli

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/scratch"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
)

func handleRegistryInstall(ctx *Context) *exit.Error {
	ref, release, problem := registryPackageRef(ctx.Inv.Args[0], ctx.Inv.Value("--version"))
	if problem != nil {
		return problem
	}
	c := client(ctx)
	hctx, cancel := hub.LongContext()
	defer cancel()
	packagePublishStatus(ctx, "Resolving %s...", ref.String())
	plan, problem := c.PackageDownloads(hctx, ref, release)
	if problem != nil {
		return problem
	}
	if plan.Release == "" || release != "" && plan.Release != release {
		return exit.Internalf("Tensorhub returned a changed or absent package release")
	}
	packageConfig, problem := exactPackageInstallDocument("package.toml", plan.PackageConfig)
	if problem != nil {
		return problem
	}
	packageInterface, problem := exactPackageInstallDocument(
		"package interface", plan.PackageInterface)
	if problem != nil {
		return problem
	}
	pyproject, problem := exactPackageInstallDocument("pyproject.toml", plan.Pyproject)
	if problem != nil {
		return problem
	}
	uvLock, problem := exactPackageInstallDocument("uv.lock", plan.UVLock)
	if problem != nil {
		return problem
	}
	release = plan.Release
	existingLayout, existing, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return problem
	}
	_, existingInstall, problem := existing.ActivePackage(ref.String())
	if problem != nil {
		existing.Close()
		return problem
	}
	if existingInstall != nil && existingInstall.SourceKind == "tensorhub" &&
		existingInstall.Package == ref.String() && existingInstall.Version == release &&
		!install.CompanionsStale(existingLayout.Companions,
			filepath.Join(existingInstall.Dir, "venv")) {
		defer existing.Close()
		result := &install.Result{Install: *existingInstall, Idempotent: true}
		modelScratch, problem := scratch.Temp(existingLayout.Tmp, "package-model-prefetch-")
		if problem != nil {
			return problem
		}
		defer modelScratch.Release()
		bestEffortDefaultModels(hctx, ctx, modelScratch.Path,
			&install.PublishedSource{Package: ref.String(), Release: release,
				PackageConfig: packageConfig,
				Selection:     install.Selection{PackageInterface: packageInterface}}, result)
		return emitInstallResult(ctx, existingLayout, existing, result)
	}
	existing.Close()
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return problem
	}
	work, problem := scratch.Temp(layout.Tmp, "package-install-")
	if problem != nil {
		return problem
	}
	defer work.Release()
	published, problem := downloadPackageInstallPlan(hctx, ctx, work.Path, ref, release,
		plan, packageConfig, packageInterface, pyproject, uvLock)
	if problem != nil {
		return problem
	}
	_, st, writer, problem := open(ctx.Cfg, true)
	if problem != nil {
		return problem
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
		return problem
	}
	st.Close()
	writer.Unlock()
	bestEffortDefaultModels(hctx, ctx, work.Path, published, result)
	_, outputStore, outputWriter, problem := open(ctx.Cfg, true)
	if problem != nil {
		return problem
	}
	defer outputStore.Close()
	defer outputWriter.Unlock()
	return emitInstallResult(ctx, layout, outputStore, result)
}

func bestEffortDefaultModels(hctx context.Context, ctx *Context, root string,
	published *install.PublishedSource, result *install.Result,
) {
	if ctx.Inv.Bool("--no-model-download") {
		result.ModelStatus = "skipped"
		return
	}
	if problem := downloadPublishedPackageModels(hctx, ctx, root, published); problem != nil {
		result.ModelStatus = "failed"
		result.ModelError = problem.ErrName() + ": " + problem.Message
		result.Warnings = append(result.Warnings,
			"package code is installed; default model prefetch failed: "+problem.Message)
		return
	}
	result.ModelStatus = install.ModelPrefetchStatus(published.Models)
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

func downloadPackageInstallPlan(ctx context.Context, cli *Context, scratch string, ref hub.Ref,
	release string, plan hub.PackageDownloadPlan, packageConfig,
	packageInterface, pyproject, uvLock install.ExactDocument,
) (*install.PublishedSource, *exit.Error) {
	if len(plan.Downloads) == 0 || len(plan.Downloads) > hub.MaxPackageInstallDownloads {
		return nil, exit.Internalf("Tensorhub returned an invalid package install plan")
	}
	published := &install.PublishedSource{
		Package: ref.String(), Release: release,
		PackageConfig: packageConfig,
		Pyproject:     pyproject,
		UVLock:        uvLock,
		Selection:     install.Selection{PackageInterface: packageInterface},
		ReportDefect:  localDefectReporter(cli, ref, release),
	}
	seen := map[string]bool{}
	type job struct {
		download hub.PackageInstallDownload
		dst      string
	}
	jobs := make([]job, 0, len(plan.Downloads))
	for _, download := range plan.Downloads {
		var dst string
		switch download.Kind {
		case "project_wheel", "dependency_wheel", "local_materialization_wheel":
			if filepath.Base(download.Path) != download.Path || !strings.HasSuffix(download.Path, ".whl") {
				return nil, exit.Internalf("Tensorhub returned unsafe package wheel path %q", download.Path)
			}
			dst = filepath.Join(scratch, "wheels", download.Path)
			if download.Kind == "project_wheel" {
				if published.ProjectWheel.Path != "" {
					return nil, exit.Internalf("Tensorhub returned more than one project wheel")
				}
				published.ProjectWheel = install.PublishedWheel{Digest: download.Digest,
					Distribution: download.Distribution, Filename: download.Path,
					ImportRoots: append([]string(nil), download.ImportRoots...), Length: download.Length,
					Path: dst, Tags: append([]string(nil), download.Tags...), Version: download.Version}
			} else if download.Kind == "dependency_wheel" {
				published.Wheels = append(published.Wheels, install.PublishedWheel{Digest: download.Digest,
					Distribution: download.Distribution, Filename: download.Path,
					ImportRoots: append([]string(nil), download.ImportRoots...), Length: download.Length,
					Path: dst, Tags: append([]string(nil), download.Tags...), Version: download.Version})
			} else {
				published.LocalWheels = append(published.LocalWheels, install.PublishedWheel{Digest: download.Digest,
					Distribution: download.Distribution, Filename: download.Path,
					ImportRoots: append([]string(nil), download.ImportRoots...), Length: download.Length,
					Path: dst, Tags: append([]string(nil), download.Tags...), Version: download.Version})
			}
		default:
			return nil, exit.Internalf("Tensorhub returned unknown package file kind %q", download.Kind)
		}
		if seen[dst] || download.URL == "" {
			return nil, exit.Internalf("Tensorhub returned a duplicate or unreadable package file")
		}
		seen[dst] = true
		jobs = append(jobs, job{download: download, dst: dst})
	}
	if published.ProjectWheel.Path == "" {
		return nil, exit.Internalf("Tensorhub package install plan has no project wheel")
	}
	queue := make(chan job, len(jobs))
	results := make(chan *exit.Error, len(jobs))
	for _, item := range jobs {
		queue <- item
	}
	close(queue)
	var group sync.WaitGroup
	for range min(16, len(jobs)) {
		group.Add(1)
		go func() {
			defer group.Done()
			for item := range queue {
				results <- transfer.DownloadExact(ctx, item.download.Path, item.download.URL, item.dst,
					item.download.Digest, item.download.Length)
			}
		}()
	}
	go func() {
		group.Wait()
		close(results)
	}()
	progress := packageFileCounter(cli, "Downloading files")
	progress(0, len(jobs))
	completed := 0
	var firstProblem *exit.Error
	for problem := range results {
		completed++
		progress(completed, len(jobs))
		if firstProblem == nil && problem != nil {
			firstProblem = problem
		}
	}
	if firstProblem != nil {
		return nil, firstProblem
	}
	for _, item := range jobs {
		published.Files++
		published.Bytes += item.download.Length
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

// downloadPublishedPackageModels prefetches the models the package's CURRENT
// hub bindings select (th-116): the mutable rows seeded from the shipped
// package.toml at release commit and owner-retargetable afterwards. The
// installed toml is never consulted; only rows naming a slot this release's
// PackageInterface declares are prefetched.
func downloadPublishedPackageModels(ctx context.Context, cli *Context, root string,
	published *install.PublishedSource,
) *exit.Error {
	packageInterface, problem := launch.DecodePackageInterface(published.Selection.PackageInterface.Bytes)
	if problem != nil {
		return problem
	}
	declared := map[string]launch.Slot{}
	for _, callables := range [][]launch.Entrypoint{packageInterface.Entrypoints, packageInterface.Jobs} {
		for i := range callables {
			for _, slot := range callables[i].Models {
				declared[slot.Path] = slot
			}
		}
	}
	ref, problem := hub.ParseRef(published.Package)
	if problem != nil {
		return problem
	}
	rows, problem := client(cli).PackageBindings(ctx, ref)
	if problem != nil {
		return problem
	}
	bindings := make([]hub.PackageBindingRow, 0, len(rows))
	for _, row := range rows {
		if _, ok := declared[row.Slot]; ok {
			bindings = append(bindings, row)
		}
	}
	if len(bindings) == 0 {
		return nil
	}
	if err := raiseOpenFileLimit(); err != nil {
		return exit.Named(exit.Structural, "open_file_limit_unavailable",
			"cannot raise the open-file limit for model leases: %s", err).
			WithRemedy("allow Cozy to raise RLIMIT_NOFILE to this account's hard limit")
	}
	tool, _, problem := localTensorFS(cli)
	if problem != nil {
		return problem
	}
	hubClient := client(cli)
	for index, binding := range bindings {
		packagePublishStatus(cli, "Resolving model %s...", binding.Ref())
		selected, problem := acquirePublishedModel(ctx, cli, tool, hubClient,
			binding.Ref(), binding.Lane, published.Package, declared[binding.Slot],
			filepath.Join(root, "models", fmt.Sprintf("%03d", index)))
		if problem != nil {
			return problem
		}
		published.Models = append(published.Models, selected)
	}
	return nil
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
	fetch := &transfer.Fetch{Tool: tool, Hub: hubClient, Spec: spec, Lane: lane,
		Progress: progress(cli), Scratch: work}
	resolved, problem := fetch.Resolve(ctx)
	if problem != nil {
		return empty, exit.Named(problem.Code, "model_resolution_unavailable",
			"cannot resolve model %s for slot %s: %s", spec, slot.Path, problem.Message).
			WithRemedy("download another compatible model or override this slot with model.%s=org/model@release", slot.Path[strings.LastIndex(slot.Path, ".")+1:])
	}
	if problem := requireCheckpointComponents(spec, slot, resolved.Components); problem != nil {
		return empty, problem
	}
	fetched, problem := fetch.Acquire(ctx, resolved)
	if problem != nil {
		return empty, problem
	}
	manifest, err := canonical.Raw(fetched.ManifestID)
	if err != nil || len(manifest) != 32 || fetched.ManifestLength <= 0 ||
		fetched.Release == "" || fetched.Lane == "" || fetch.Ref.String() == "/" {
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
