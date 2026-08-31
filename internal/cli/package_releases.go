package cli

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/transfer"
)

var immutablePackageVersion = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

func handlePackagePublish(ctx *Context) *exit.Error {
	pack, problem := packagepublish.Prepare()
	if problem != nil {
		return problem
	}
	defer pack.Close()
	ref, problem := hub.ParseRef(pack.Organization + "/" + pack.Name)
	if problem != nil {
		return problem.WithRemedy("fix [tool.cozy] organization or [project] name in pyproject.toml")
	}
	release := pack.Release
	reason := "cozy package publish " + ref.String() + "@" + release
	packagePublishStatus(ctx, "Checking %s@%s...", ref.String(), release)

	c := client(ctx)
	hctx, cancel := hub.LongContext()
	defer cancel()
	if problem := packagePublishStage(ctx, "Building package wheel and local dependencies", func() *exit.Error {
		return pack.Build(hctx)
	}); problem != nil {
		return problem
	}
	paths := packagepublish.Paths(pack.Files)
	dependencyWheels := packagepublish.WheelFilenames(pack.DependencyWheels)
	packagePublishStatus(ctx, "Declaring %d source files and %d dependency wheels...",
		len(paths), len(dependencyWheels))
	draft, problem := c.DeclarePackageRelease(hctx, ref, release, paths, dependencyWheels, reason)
	if problem != nil {
		return problem
	}
	if draft.State != "pending" && draft.State != "committed" {
		return exit.Internalf("package declaration returned invalid state %q", draft.State)
	}
	var moved int64
	if draft.State == "pending" {
		moved, problem = uploadPackageFiles(hctx, pack, draft.Uploads,
			packageUploadCounter(ctx))
		if problem != nil {
			return problem
		}
	} else {
		packagePublishStatus(ctx, "Release already published; refreshing status...")
	}
	var done hub.PackageReleaseCommit
	problem = packagePublishStage(ctx, "Committing exact package release", func() *exit.Error {
		var finalProblem *exit.Error
		done, finalProblem = c.CommitPackageRelease(hctx, ref, release, reason)
		return finalProblem
	})
	if problem != nil {
		return problem
	}
	if done.State != "committed" {
		return exit.Internalf("package commit did not reach committed state")
	}
	if _, err := canonical.Raw(done.ReleaseDigest); err != nil {
		return exit.Internalf("package commit returned no exact release digest")
	}
	replay := draft.State == "committed"
	status := "published"
	if replay {
		status = "already published"
	}
	fields := []output.Field{
		{K: "package", V: ref.String()}, {K: "release", V: release},
		{K: "status", V: status}, {K: "changed", V: !replay},
		{K: "release_digest", V: done.ReleaseDigest},
		{K: "uploaded", V: output.Bytes(moved)}, {K: "hub", V: c.Base()},
	}
	return emit(ctx, compactRecord(fields, "package", "release", "status"))
}

func handlePackageYank(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	release := strings.TrimSpace(ctx.Inv.Value("--version"))
	if !immutablePackageVersion.MatchString(release) {
		return exit.Usagef("--version %q is not an immutable N.M.P package release", release).
			WithRemedy("use a release such as 1.2.3")
	}
	reason := "cozy package yank " + ref.String() + "@" + release
	hctx, cancel := hub.LongContext()
	defer cancel()
	yanked, problem := client(ctx).YankPackageRelease(hctx, ref, release, reason)
	if problem != nil {
		return problem
	}
	if yanked.State != "yanked" || yanked.Release != release || yanked.YankedAt == "" {
		return exit.Internalf("package yank returned an invalid release tombstone")
	}
	if _, err := time.Parse(time.RFC3339Nano, yanked.YankedAt); err != nil {
		return exit.Internalf("package yank returned invalid yanked_at %q", yanked.YankedAt)
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "package", V: ref.String()}, {K: "release", V: release},
		{K: "status", V: "yanked"}, {K: "changed", V: yanked.Changed},
		{K: "yanked_at", V: yanked.YankedAt},
	}, "package", "release", "status"))
}

func shorten(value string, limit int) string {
	value = strings.Join(strings.Fields(value), " ")
	runes := []rune(value)
	if len(runes) > limit {
		return string(runes[:limit]) + "…"
	}
	return value
}

type packageFile struct {
	subject string
	path    string
	upload  hub.PackageUpload
}

const maxConcurrentPackageUploads = 16

func uploadPackageFiles(ctx context.Context, pack *packagepublish.Package,
	uploads []hub.PackageUpload, progress func(completed, total int),
) (int64, *exit.Error) {
	files := make([]packageFile, 0, len(uploads))
	wantSources := make(map[string]string, len(pack.Files))
	for path, local := range pack.Files {
		wantSources[path] = local
	}
	wantDependencies := make(map[string]string, len(pack.DependencyWheels))
	for _, dependency := range pack.DependencyWheels {
		wantDependencies[dependency.Filename] = dependency.Path
	}
	wantProject := true
	wantDescriptor := true
	for _, upload := range uploads {
		var local string
		var ok bool
		switch upload.Kind {
		case "project_wheel":
			if !wantProject || upload.Path != "project.whl" {
				return 0, exit.Internalf("package uploads returned unknown project wheel %q", upload.Path)
			}
			local, ok = pack.Wheel, true
			wantProject = false
		case "descriptor":
			if !wantDescriptor || upload.Path != "descriptor.json" || pack.Descriptor == "" {
				return 0, exit.Internalf("package uploads returned unknown descriptor %q", upload.Path)
			}
			local, ok = pack.Descriptor, true
			wantDescriptor = false
		case "source":
			local, ok = wantSources[upload.Path]
			delete(wantSources, upload.Path)
		case "dependency_wheel":
			local, ok = wantDependencies[upload.Path]
			delete(wantDependencies, upload.Path)
		default:
			return 0, exit.Internalf("package uploads returned invalid kind %q for %q", upload.Kind, upload.Path)
		}
		if !ok {
			return 0, exit.Internalf("package uploads returned unknown or duplicate %s path %q", upload.Kind, upload.Path)
		}
		files = append(files, packageFile{subject: upload.Path, path: local, upload: upload})
	}
	if wantProject || wantDescriptor || len(wantSources) != 0 || len(wantDependencies) != 0 {
		return 0, exit.Internalf("package uploads omitted %d source files and %d dependency wheels",
			len(wantSources), len(wantDependencies))
	}

	pending := make([]packageFile, 0, len(files))
	for _, file := range files {
		if file.upload.AlreadyUploaded {
			continue
		}
		if file.upload.URL == "" {
			return 0, exit.Internalf("package upload returned no URL for %s", file.subject)
		}
		pending = append(pending, file)
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
