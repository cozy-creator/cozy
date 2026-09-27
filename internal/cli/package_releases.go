package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/transfer"
)

var immutablePackageVersion = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

func handlePackagePublish(ctx *Context) *exit.Error {
	publishStarted := time.Now()
	timings := newPackagePublishTimings()
	pack, problem := packagepublish.Prepare()
	if problem != nil {
		return problem
	}
	defer pack.Close()
	c, account, problem := publicationAccount(ctx)
	if problem != nil {
		return problem
	}
	ref, problem := hub.ParseRef(account.Name + "/" + pack.Name)
	if problem != nil {
		return problem.WithRemedy("fix [project] name in pyproject.toml")
	}
	release := pack.Release
	reason := "cozy package publish " + ref.String() + "@" + release
	packagePublishStatus(ctx, "Checking %s@%s...", ref.String(), release)

	hctx, cancel := hub.LongContext()
	defer cancel()
	lookupStarted := time.Now()
	detail, lookupProblem := c.PackageRelease(hctx, ref, release)
	timings.measure("release_lookup", lookupStarted)
	if lookupProblem == nil {
		if detail.Release.Release != release {
			return exit.Internalf("package lookup returned release %q, want %q",
				detail.Release.Release, release)
		}
		packagePublishStatus(ctx, "Release already published; no build or upload needed.")
		return emit(ctx, compactRecord([]output.Field{
			{K: "package", V: ref.String()}, {K: "release", V: release},
			{K: "status", V: "already published"}, {K: "changed", V: false},
			{K: "uploaded", V: output.Bytes(0)}, {K: "hub", V: c.Base()},
		}, "package", "release", "status"))
	}
	if lookupProblem.Code != exit.NotFound {
		return lookupProblem
	}
	if problem := timings.stage(ctx, "Building package wheel and local dependencies", func() *exit.Error {
		if problem := pack.BuildForPublish(hctx, ctx.Inv.Values["--wheel"]...); problem != nil {
			return problem
		}
		return nil
	}); problem != nil {
		return problem
	}
	filesStarted := time.Now()
	declared, registry, locals, problem := packageFiles(pack)
	timings.measure("file_digesting", filesStarted)
	if problem != nil {
		return problem
	}
	packagePublishStatus(ctx, "Declaring %d files and %d registry rows...",
		len(declared), len(registry))
	declareStarted := time.Now()
	draft, problem := c.DeclarePackageRelease(hctx, ref, release, declared, reason)
	timings.measure("declaration", declareStarted)
	if problem != nil {
		return problem
	}
	if draft.PublicationID == "" {
		return exit.Internalf("package declaration returned no publication id")
	}
	uploadStarted := time.Now()
	var moved int64
	moved, problem = uploadPackageFiles(hctx, declared, locals, draft.Files,
		packageUploadCounter(ctx))
	timings.measure("object_upload", uploadStarted)
	if problem != nil {
		return problem
	}
	var done hub.PackageReleaseCommit
	problem = timings.stage(ctx, "Committing exact package release", func() *exit.Error {
		var finalProblem *exit.Error
		done, finalProblem = c.CommitPackageRelease(hctx, ref, release,
			draft.PublicationID, registry, reason, pack.PythonVersion)
		if finalProblem == nil && done.State != "committed" {
			done, finalProblem = c.WaitPackageRelease(hctx, ref, release, done,
				func(status hub.PackageReleaseCommit) {
					packagePublishStatus(ctx, "Package finalization: %s", status.State)
				})
		}
		return finalProblem
	})
	if problem != nil {
		return problem
	}
	if done.State != "committed" {
		return exit.Internalf("package commit did not reach committed state")
	}
	if done.PublicationID != draft.PublicationID {
		return exit.Internalf("package commit returned publication id %q, want %q",
			done.PublicationID, draft.PublicationID)
	}
	timings.finish(publishStarted)
	fields := []output.Field{
		{K: "package", V: ref.String()}, {K: "release", V: release},
		{K: "status", V: "published"}, {K: "changed", V: true},
		{K: "uploaded", V: output.Bytes(moved)}, {K: "hub", V: c.Base()},
		{K: "timings", V: *timings},
	}
	record := compactRecord(fields, "package", "release", "status", "timings")
	for _, warning := range done.Warnings {
		record.Notes = append(record.Notes, "warning: "+warning)
	}
	for _, dependency := range pack.Vendored {
		record.Notes = append(record.Notes, packagepublish.VendoredNote(account.Name, dependency))
	}
	return emit(ctx, record)
}

func handlePackageYank(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	release := strings.TrimSpace(ctx.Inv.Value("--version"))
	if _, err := pep440.Parse(release); err != nil {
		return exit.Usagef("--version %q is not a package release", release).
			WithRemedy("use a release such as 1.2.3")
	}
	reason := "cozy package yank " + ref.String() + "@" + release
	hctx, cancel := hub.LongContext()
	defer cancel()
	yanked, problem := client(ctx).YankPackageRelease(hctx, ref, release, reason)
	if problem != nil {
		return problem
	}
	if yanked.State != "yanked" {
		return exit.Internalf("package yank answered state %q", yanked.State)
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "package", V: ref.String()}, {K: "release", V: release},
		{K: "status", V: "yanked"}, {K: "changed", V: yanked.Changed},
		{K: "yanked_at", V: yanked.YankedAt},
	}, "package", "release", "status"))
}

// packageFiles measures the PyPI-shaped release: one or more project wheels, one source
// archive, exact install metadata, and any locally vendored dependency wheels.
// Registry dependencies remain external lock facts and are never uploaded.
func packageFiles(pack *packagepublish.Package) (
	[]hub.PackageDeclaredFile, []hub.PackageRegistryRow, map[string]string, *exit.Error,
) {
	locals := map[string]string{}
	add := func(path, local string) *exit.Error {
		if _, exists := locals[path]; exists {
			return exit.Named(exit.Conflict, "package_file_path_conflict",
				"more than one package file maps to %s", path)
		}
		locals[path] = local
		return nil
	}
	projectWheels := pack.ProjectWheels
	if len(projectWheels) == 0 && pack.Wheel != "" {
		projectWheels = []string{pack.Wheel}
	}
	if len(projectWheels) == 0 {
		return nil, nil, nil, exit.Internalf("package build produced no project wheels")
	}
	for _, wheel := range projectWheels {
		projectPath := "artifacts/project/" + filepath.Base(wheel)
		if problem := add(projectPath, wheel); problem != nil {
			return nil, nil, nil, problem
		}
	}
	if pack.SourceArchive == "" {
		return nil, nil, nil, exit.Internalf("package build produced no source archive")
	}
	sourcePath := "artifacts/source/" + filepath.Base(pack.SourceArchive)
	if problem := add(sourcePath, pack.SourceArchive); problem != nil {
		return nil, nil, nil, problem
	}
	if problem := add("metadata/package-interface.json", pack.PackageInterface); problem != nil {
		return nil, nil, nil, problem
	}
	for _, name := range []string{"package.toml", "pyproject.toml", "uv.lock"} {
		local, ok := pack.Files[name]
		if !ok {
			return nil, nil, nil, exit.Named(exit.Structural, "package_source_required_file_missing",
				"prepared package source has no %s", name)
		}
		if problem := add(name, local); problem != nil {
			return nil, nil, nil, problem
		}
	}
	for _, dependency := range pack.DependencyWheels {
		path := "artifacts/dependencies/" + dependency.Filename
		if problem := add(path, dependency.Path); problem != nil {
			return nil, nil, nil, problem
		}
	}
	declared := make([]hub.PackageDeclaredFile, 0, len(locals))
	for path, local := range locals {
		raw, err := os.ReadFile(local)
		if err != nil {
			return nil, nil, nil, exit.Named(exit.Structural, "package_source_unreadable",
				"%s is unreadable: %v", path, err)
		}
		sum := sha256.Sum256(raw)
		hexDigest := hex.EncodeToString(sum[:])
		declared = append(declared, hub.PackageDeclaredFile{
			Digest: "sha256:" + hexDigest,
			Length: int64(len(raw)), Path: path})
	}
	sort.Slice(declared, func(i, j int) bool { return declared[i].Path < declared[j].Path })
	registry := make([]hub.PackageRegistryRow, 0, len(pack.Registry))
	for _, row := range pack.Registry {
		registry = append(registry, hub.PackageRegistryRow{Name: row.Name, SHA256: row.SHA256,
			Size: row.Size, URL: row.URL, Version: row.Version})
	}
	return declared, registry, locals, nil
}

type packageFile struct {
	subject string
	path    string
	upload  hub.PackagePresignedUpload
}

const maxConcurrentPackageUploads = 16

// uploadPackageFiles PUTs every absent subject under its checksum-pinned grant.
// A 412 stands as success (the content-addressed key already holds these exact
// bytes); the hub's finalize re-hashes everything regardless.
func uploadPackageFiles(ctx context.Context, declared []hub.PackageDeclaredFile,
	locals map[string]string, grants []hub.PackageFileGrant,
	progress func(completed, total int),
) (int64, *exit.Error) {
	want := make(map[string]hub.PackageDeclaredFile, len(declared))
	for _, file := range declared {
		want[file.Path] = file
	}
	pending := make([]packageFile, 0, len(grants))
	for _, grant := range grants {
		if _, ok := want[grant.Path]; !ok {
			return 0, exit.Internalf("package grants answered an undeclared subject %q", grant.Path)
		}
		delete(want, grant.Path)
		if grant.Present {
			continue
		}
		if grant.Upload == nil || grant.Upload.URL == "" {
			return 0, exit.Internalf("package grant for %s is neither present nor uploadable", grant.Path)
		}
		local, ok := locals[grant.Path]
		if !ok {
			return 0, exit.Internalf("package grant names unknown local subject %q", grant.Path)
		}
		pending = append(pending, packageFile{subject: grant.Path, path: local, upload: *grant.Upload})
	}
	if len(want) != 0 {
		return 0, exit.Internalf("package grants omitted %d declared subjects", len(want))
	}
	if progress != nil {
		progress(0, len(pending))
	}
	type outcome struct {
		moved int64
		err   *exit.Error
	}
	jobs := make(chan packageFile, len(pending))
	results := make(chan outcome, len(pending))
	for _, file := range pending {
		jobs <- file
	}
	close(jobs)
	var group sync.WaitGroup
	for range min(len(pending), maxConcurrentPackageUploads) {
		group.Add(1)
		go func() {
			defer group.Done()
			for file := range jobs {
				bytes, problem := transfer.UploadPresigned(ctx, file.subject, file.path,
					file.upload.URL, file.upload.RequiredHeaders)
				results <- outcome{moved: bytes, err: problem}
			}
		}()
	}
	go func() {
		group.Wait()
		close(results)
	}()
	var moved int64
	var firstProblem *exit.Error
	completed := 0
	for result := range results {
		completed++
		if progress != nil {
			progress(completed, len(pending))
		}
		moved += result.moved
		if firstProblem == nil && result.err != nil {
			firstProblem = result.err
		}
	}
	return moved, firstProblem
}

const packagePublishHeartbeat = 15 * time.Second

func packagePublishStatus(ctx *Context, format string, args ...any) {
	if ctx.Mode().JSON {
		return
	}
	_ = output.Progress(ctx.Err, fmt.Sprintf(format, args...))
}

func packagePublishStage(ctx *Context, label string, run func() *exit.Error) *exit.Error {
	packagePublishStatus(ctx, "%s...", label)
	done := make(chan struct{})
	var group sync.WaitGroup
	group.Add(1)
	go func() {
		defer group.Done()
		started := time.Now()
		ticker := time.NewTicker(packagePublishHeartbeat)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				packagePublishStatus(ctx, "%s... %s elapsed", label,
					time.Since(started).Round(time.Second))
			}
		}
	}()
	problem := run()
	close(done)
	group.Wait()
	return problem
}

func packageUploadCounter(ctx *Context) func(completed, total int) {
	return packageFileCounter(ctx, "Uploading files")
}

func packageFileCounter(ctx *Context, label string) func(completed, total int) {
	last := -1
	return func(completed, total int) {
		if total == 0 {
			if last < 0 {
				packagePublishStatus(ctx, "%s: all already present", label)
				last = 0
			}
			return
		}
		step := max(1, (total+19)/20)
		if completed != 0 && completed != total && completed-last < step {
			return
		}
		packagePublishStatus(ctx, "%s: %d/%d", label, completed, total)
		last = completed
	}
}
