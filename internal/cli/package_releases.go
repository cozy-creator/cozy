package cli

import (
	"context"
	"sort"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/transfer"
)

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

	c := client(ctx)
	hctx, cancel := hub.LongContext()
	defer cancel()
	begun, problem := c.BeginPackageRelease(hctx, ref, release, reason)
	if problem != nil {
		return problem
	}
	if begun.State != "pending" && begun.State != "committed" {
		return exit.Internalf("package begin returned invalid state %q", begun.State)
	}
	var moved int64
	if begun.State == "pending" {
		if problem := pack.Build(hctx); problem != nil {
			return problem
		}
		paths := packagepublish.Paths(pack.Files)
		dependencyWheels := packagepublish.WheelFilenames(pack.DependencyWheels)
		var uploads []hub.PackageUpload
		for len(paths) > 0 || dependencyWheels != nil {
			n := min(len(paths), 1000)
			batch, problem := c.PackageReleaseUploads(hctx, ref, release, paths[:n], dependencyWheels, reason)
			if problem != nil {
				return problem
			}
			uploads = append(uploads, batch.Uploads...)
			paths = paths[n:]
			dependencyWheels = nil
		}
		moved, problem = uploadPackageFiles(hctx, pack, begun.ProjectWheelUpload, uploads)
		if problem != nil {
			return problem
		}
	}
	done, problem := c.FinalizePackageRelease(hctx, ref, release, reason)
	if problem != nil {
		return problem
	}
	if strings.TrimSpace(done.QualificationState) == "" {
		return exit.Internalf("package finalize returned no qualification state")
	}
	profileRows := make([]string, 0, len(done.Profiles))
	var candidateRows, executionRows, refusalRows []string
	for _, profile := range done.Profiles {
		profileRows = append(profileRows, profile.Profile+":"+profile.State+":"+profile.BaseRealizationKind+"@"+
			profile.BaseRealizationDigest)
		if profile.State == "qualified" {
			candidateRows = append(candidateRows, profile.Profile+"/"+profile.BaseRealizationKind+"="+profile.CandidateID)
		} else {
			refusal := profile.Profile + "/" + profile.BaseRealizationKind + " " + profile.RefusalCode
			if profile.RefusalDetail != "" {
				refusal += ": " + shorten(profile.RefusalDetail, 240)
			}
			refusalRows = append(refusalRows, refusal)
		}
	}
	for _, execution := range done.PackageExecutions {
		executionRows = append(executionRows, execution.Function+"@"+execution.Profile+"="+
			execution.State+" ("+execution.Digest+")")
	}
	sort.Strings(profileRows)
	sort.Strings(candidateRows)
	sort.Strings(executionRows)
	sort.Strings(refusalRows)
	fields := []output.Field{
		{K: "package", V: ref.String()}, {K: "release", V: release},
		{K: "status", V: "published"}, {K: "changed", V: begun.State != "committed"},
		{K: "qualification", V: done.QualificationState},
		{K: "qualification_error", V: done.QualificationError},
		{K: "compatible_profiles", V: done.CompatibleProfiles}, {K: "profiles", V: profileRows},
		{K: "package_executions", V: executionRows},
		{K: "requires_python", V: done.RequiresPython}, {K: "requirements", V: done.Requirements},
		{K: "uploaded", V: output.Bytes(moved)}, {K: "hub", V: c.Base()},
	}
	if len(candidateRows) > 0 {
		fields = append(fields, output.Field{K: "candidates", V: candidateRows})
	}
	if len(refusalRows) > 0 {
		fields = append(fields, output.Field{K: "profile_refusals", V: refusalRows})
	}
	defaults := []string{"package", "release", "status", "qualification"}
	if done.QualificationState != "qualified" {
		defaults = append(defaults, "requires_python", "requirements", "compatible_profiles")
	}
	if done.QualificationState == "refused" && len(refusalRows) > 0 {
		defaults = append(defaults, "profile_refusals")
	}
	if done.QualificationError != "" {
		defaults = append(defaults, "qualification_error")
	}
	defaults = append(defaults, "package_executions", "uploaded", "changed")
	record := compactRecord(fields, defaults...)
	switch done.QualificationState {
	case "pending":
		record.Notes = append(record.Notes,
			"published successfully; Tensorhub will qualify it when a compatible base worker image becomes active")
	case "refused", "unsupported":
		record.Notes = append(record.Notes,
			"published successfully, but no execution is currently eligible for placement")
	}
	return emit(ctx, record)
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

func uploadPackageFiles(ctx context.Context, pack *packagepublish.Package, wheel hub.PackageUpload,
	uploads []hub.PackageUpload,
) (int64, *exit.Error) {
	files := make([]packageFile, 0, len(uploads)+1)
	files = append(files, packageFile{subject: "project_wheel", path: pack.Wheel, upload: wheel})
	wantSources := make(map[string]string, len(pack.Files))
	for path, local := range pack.Files {
		wantSources[path] = local
	}
	wantDependencies := make(map[string]string, len(pack.DependencyWheels))
	for _, dependency := range pack.DependencyWheels {
		wantDependencies[dependency.Filename] = dependency.Path
	}
	for _, upload := range uploads {
		var local string
		var ok bool
		switch upload.Kind {
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
	if len(wantSources) != 0 || len(wantDependencies) != 0 {
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
				uploaded, bytes, problem := transfer.UploadPresigned(ctx, file.subject, file.path,
					file.upload.URL, file.upload.RequiredHeaders)
				if !uploaded {
					bytes = 0
				}
				results <- outcome{moved: bytes, err: problem}
			}
		}()
	}
	group.Wait()
	close(results)
	var moved int64
	for result := range results {
		moved += result.moved
		if result.err != nil {
			return moved, result.err
		}
	}
	return moved, nil
}
