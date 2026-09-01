package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
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
	packageDescriptor, problem := exactPackageInstallDocument(
		"package descriptor", plan.PackageDescriptor)
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
	if _, err := canonical.Raw(plan.ReleaseDigest); err != nil {
		return exit.Internalf("Tensorhub returned an invalid release digest")
	}
	releaseDigest := plan.ReleaseDigest
	existingLayout, existing, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return problem
	}
	_, existingInstall, problem := existing.ActivePackage(ref.String())
	if problem != nil {
		existing.Close()
		return problem
	}
	if existingInstall != nil && existingInstall.SourceDigest == releaseDigest {
		defer existing.Close()
		result := &install.Result{Install: *existingInstall, Idempotent: true}
		if err := os.MkdirAll(existingLayout.Transfer, 0o700); err != nil {
			return exit.Internalf("cannot create model prefetch scratch: %s", err)
		}
		modelScratch, err := os.MkdirTemp(existingLayout.Transfer, "package-model-prefetch-")
		if err != nil {
			return exit.Internalf("cannot create model prefetch scratch: %s", err)
		}
		defer os.RemoveAll(modelScratch)
		bestEffortDefaultModels(hctx, ctx, modelScratch, existingInstall.Runtime,
			&install.PublishedSource{Package: ref.String(), Release: release,
				SourceDigest: releaseDigest, PackageConfig: packageConfig,
				Selection: install.Selection{PackageDescriptor: packageDescriptor}}, result)
		return emitInstallResult(ctx, existingLayout, existing, result)
	}
	existing.Close()
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return problem
	}
	if err := os.MkdirAll(layout.Transfer, 0o700); err != nil {
		return exit.Internalf("cannot create package download scratch: %s", err)
	}
	scratch, err := os.MkdirTemp(layout.Transfer, "package-install-")
	if err != nil {
		return exit.Internalf("cannot create package download scratch: %s", err)
	}
	defer os.RemoveAll(scratch)
	published, problem := downloadPackageInstallPlan(hctx, ctx, scratch, ref, release, releaseDigest,
		plan, packageConfig, packageDescriptor, pyproject, uvLock)
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
	bestEffortDefaultModels(hctx, ctx, scratch, result.Install.Runtime, published, result)
	_, outputStore, outputWriter, problem := open(ctx.Cfg, true)
	if problem != nil {
		return problem
	}
	defer outputStore.Close()
	defer outputWriter.Unlock()
	return emitInstallResult(ctx, layout, outputStore, result)
}

func bestEffortDefaultModels(hctx context.Context, ctx *Context, root, runtimeBin string,
	published *install.PublishedSource, result *install.Result,
) {
	if ctx.Inv.Bool("--no-model-download") {
		result.ModelStatus = "skipped"
		return
	}
	if runtimeBin == "" {
		result.ModelStatus = "failed"
		result.ModelError = "installed Runtime path is absent"
		result.Warnings = append(result.Warnings,
			"package code is installed; default model prefetch was skipped because its Runtime path is absent")
		return
	}
	if problem := downloadPublishedPackageModels(hctx, ctx, root, runtimeBin, published); problem != nil {
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
	release, releaseDigest string, plan hub.PackageDownloadPlan, packageConfig,
	packageDescriptor, pyproject, uvLock install.ExactDocument,
) (*install.PublishedSource, *exit.Error) {
	if len(plan.Downloads) == 0 || len(plan.Downloads) > hub.MaxPackageInstallDownloads {
		return nil, exit.Internalf("Tensorhub returned an invalid package install plan")
	}
	published := &install.PublishedSource{
		Package: ref.String(), Release: release, SourceDigest: releaseDigest,
		PackageConfig: packageConfig,
		Pyproject:     pyproject,
		UVLock:        uvLock,
		Selection:     install.Selection{PackageDescriptor: packageDescriptor},
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

type publishedDefaultBinding struct {
	ModelBindingPath string `json:"model_binding_path"`
	ModelParameter   string `json:"model_parameter_name"`
	ModelClass       string `json:"model_class"`
	Ref              string `json:"ref"`
	Lane             string `json:"lane"`
	Source           string `json:"source"`
}

func downloadPublishedPackageModels(ctx context.Context, cli *Context, root, runtimeBin string,
	published *install.PublishedSource,
) *exit.Error {
	bindings, problem := publishedDefaultBindings(ctx, cli.Cfg, root, runtimeBin,
		published.PackageConfig, published.Selection.PackageDescriptor)
	if problem != nil || len(bindings) == 0 {
		return problem
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
		packagePublishStatus(cli, "Resolving model %s...", binding.Ref)
		selected, problem := acquirePublishedModel(ctx, cli, tool, hubClient,
			binding.Ref, binding.Lane, published.Package, binding.ModelBindingPath,
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
	hubClient *hub.Client, spec, lane, packageName, slot, work string,
) (install.PublishedModel, *exit.Error) {
	var empty install.PublishedModel
	if local, ok, problem := exactLocalModel(tool, spec, lane, work); problem != nil {
		return empty, problem
	} else if ok {
		return install.PublishedModel{Package: packageName, Slot: slot, Model: local.Model,
			Release: local.Release, Lane: local.Lane, Manifest: local.Manifest,
			ManifestLength: local.ManifestLength, Reused: true}, nil
	}
	fetch := &transfer.Fetch{Tool: tool, Hub: hubClient, Spec: spec, Lane: lane,
		Progress: progress(cli), Scratch: work}
	resolved, problem := fetch.Resolve(ctx)
	if problem != nil {
		return empty, exit.Named(problem.Code, "model_resolution_unavailable",
			"cannot resolve model %s for slot %s: %s", spec, slot, problem.Message).
			WithRemedy("download another compatible model or override this slot with --model %s=org/model@release", slot)
	}
	fetched, problem := fetch.Acquire(ctx, resolved)
	if problem != nil {
		return empty, problem
	}
	manifest, err := canonical.Raw(fetched.ManifestID)
	if err != nil || len(manifest) != 32 || fetched.ManifestLength <= 0 ||
		fetched.Release == "" || fetched.Lane == "" || fetch.Ref.String() == "/" {
		return empty, exit.Named(exit.Structural, "model_download_result_invalid",
			"model acquisition returned an incomplete exact selection for package slot %s", slot)
	}
	return install.PublishedModel{Package: packageName, Slot: slot, Model: fetch.Ref.String(),
		Release: fetched.Release, Lane: fetched.Lane, Manifest: fetched.ManifestID,
		ManifestLength: fetched.ManifestLength, Reused: fetched.Moved == 0}, nil
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

func publishedDefaultBindings(parent context.Context, cfg config.Config, root, runtimeBin string,
	packageConfig, descriptor install.ExactDocument,
) ([]publishedDefaultBinding, *exit.Error) {
	metadataRoot := filepath.Join(root, "selection")
	if err := os.MkdirAll(metadataRoot, 0o700); err != nil {
		return nil, exit.Internalf("cannot create package selection scratch: %s", err)
	}
	packageConfigPath := filepath.Join(metadataRoot, "package.toml")
	descriptorPath := filepath.Join(metadataRoot, "descriptor.json")
	if err := os.WriteFile(packageConfigPath, packageConfig.Bytes, 0o600); err != nil {
		return nil, exit.Internalf("cannot stage exact package.toml: %s", err)
	}
	if err := os.WriteFile(descriptorPath, descriptor.Bytes, 0o600); err != nil {
		return nil, exit.Internalf("cannot stage exact package descriptor: %s", err)
	}

	queryCtx, cancel := context.WithTimeout(parent, launch.DefaultRuntimeQueryTimeout)
	defer cancel()
	cmd := exec.CommandContext(queryCtx, runtimeBin, "--json", "--dir", metadataRoot,
		"--descriptor", descriptorPath, "bindings")
	cmd.WaitDelay = 250 * time.Millisecond
	cmd.Env = cfg.Tool("COZY_HOME=" + cfg.Home)
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if queryCtx.Err() == context.DeadlineExceeded {
		return nil, exit.Named(exit.Deadline, "runtime_query_stalled",
			"`cozy-runtime bindings` did not answer its metadata query within %s",
			launch.DefaultRuntimeQueryTimeout).
			WithRemedy("bindings must resolve package.toml against the exact descriptor without importing package code")
	}
	if cmd.ProcessState == nil {
		return nil, exit.Named(exit.Structural, "runtime_missing",
			"cannot run the trusted cozy-runtime at %s: %s", runtimeBin, err)
	}
	if code := cmd.ProcessState.ExitCode(); code != 0 {
		return nil, publishedBindingsRefusal(code, stdout.String(), stderr.String())
	}
	var answer struct {
		Bindings        []publishedDefaultBinding `json:"bindings"`
		WeightlessPlans []json.RawMessage         `json:"weightless_plans,omitempty"`
	}
	decoder := json.NewDecoder(strings.NewReader(stdout.String()))
	decoder.DisallowUnknownFields()
	decodeErr := decoder.Decode(&answer)
	var trailing any
	if decodeErr == nil {
		decodeErr = decoder.Decode(&trailing)
	}
	if decodeErr != io.EOF {
		return nil, exit.Named(exit.Structural, "runtime_bindings_invalid",
			"cozy-runtime returned an invalid package binding document")
	}
	seen := make(map[string]bool, len(answer.Bindings))
	for _, binding := range answer.Bindings {
		if binding.ModelBindingPath == "" || binding.ModelParameter == "" ||
			binding.ModelClass == "" || binding.Ref == "" || binding.Lane == "" ||
			!strings.HasPrefix(binding.Source, "package.toml:") ||
			seen[binding.ModelBindingPath] ||
			strings.TrimSpace(binding.Ref) != binding.Ref || strings.TrimSpace(binding.Lane) != binding.Lane {
			return nil, exit.Named(exit.Structural, "runtime_bindings_invalid",
				"cozy-runtime returned an incomplete or duplicate package model binding")
		}
		seen[binding.ModelBindingPath] = true
	}
	return answer.Bindings, nil
}

func publishedBindingsRefusal(code int, stdout, stderr string) *exit.Error {
	said := strings.TrimSpace(stderr)
	if said == "" {
		said = strings.TrimSpace(stdout)
	}
	var document struct {
		Error struct {
			Name    string `json:"name"`
			Message string `json:"message"`
			Remedy  string `json:"remedy"`
		} `json:"error"`
	}
	name, remedy := "runtime_bindings_refused", ""
	if json.Unmarshal([]byte(stderr), &document) == nil && document.Error.Message != "" {
		said, remedy = document.Error.Message, document.Error.Remedy
		if document.Error.Name != "" {
			name = document.Error.Name
		}
	}
	exitCode := exit.Code(code)
	if !exitCode.Valid() {
		exitCode = exit.Internal
	}
	if said == "" {
		said = "the Runtime gave no diagnostic"
	}
	problem := exit.Named(exitCode, name, "`cozy-runtime bindings`: %s",
		strings.Join(strings.Fields(said), " "))
	if remedy != "" {
		problem.WithRemedy("%s", remedy)
	}
	return problem
}
