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
	if !ctx.Cfg.HubToken.Present() {
		return exit.Named(exit.Credential, "hub.publish_credential_missing",
			"publishing to %s requires a configured Tensorhub token", ctx.Cfg.HubURL).
			WithRemedy("set tensorhub_token in %s/config.yaml or set TENSORHUB_TOKEN", ctx.Cfg.Home)
	}
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
	if _, problem := c.Package(hctx, ref); problem != nil {
		if problem.Code != exit.NotFound {
			return problem
		}
		if _, problem := c.CreatePackage(hctx, ref.Org, ref.Name, reason); problem != nil && problem.Code != exit.Conflict {
			return problem
		}
	}
	begun, problem := c.BeginPackageRelease(hctx, ref, release, reason)
	if problem != nil {
		return problem
	}
	if begun.Release != release || (begun.State != "pending" && begun.State != "committed") {
		return exit.Internalf("package begin returned another release or invalid state %q", begun.State)
	}
	var moved int64
	if begun.State == "pending" {
		paths := packagepublish.Paths(pack.Files)
		var uploads []hub.PackageUpload
		for len(paths) > 0 {
			n := min(len(paths), 1000)
			batch, problem := c.PackageReleaseSourceUploads(hctx, ref, release, paths[:n], reason)
			if problem != nil {
				return problem
			}
			uploads = append(uploads, batch.Uploads...)
			paths = paths[n:]
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
	if done.Release != release {
		return exit.Internalf("package finalize returned another release %q", done.Release)
	}
	switch done.QualificationState {
	case "pending", "qualified", "refused", "unsupported":
	default:
		return exit.Internalf("package finalize returned invalid qualification state %q", done.QualificationState)
	}

	profileRows := make([]string, 0, len(done.Profiles))
	var candidateRows, refusalRows []string
	for _, profile := range done.Profiles {
		profileRows = append(profileRows, profile.Profile+":"+profile.State+":"+profile.BaseRealizationKind)
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
	sort.Strings(profileRows)
	sort.Strings(candidateRows)
	sort.Strings(refusalRows)
	fields := []output.Field{
		{K: "package", V: ref.String()}, {K: "release", V: done.Release},
		{K: "status", V: "published"}, {K: "changed", V: begun.State != "committed"},
		{K: "created", V: done.Created}, {K: "qualification", V: done.QualificationState},
		{K: "qualification_error", V: done.QualificationError},
		{K: "compatible_profiles", V: done.CompatibleProfiles}, {K: "profiles", V: profileRows},
		{K: "package_executions", V: len(done.PackageExecutions)},
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

func uploadPackageFiles(ctx context.Context, pack *packagepublish.Package, wheel hub.PackageUpload,
	sources []hub.PackageUpload,
) (int64, *exit.Error) {
	files := make([]packageFile, 0, len(sources)+1)
	files = append(files, packageFile{subject: "project_wheel", path: pack.Wheel, upload: wheel})
	want := make(map[string]string, len(pack.Files))
	for path, local := range pack.Files {
		want[path] = local
	}
	for _, upload := range sources {
		local, ok := want[upload.Path]
		if !ok {
			return 0, exit.Internalf("package uploads returned unknown or duplicate source path %q", upload.Path)
		}
		delete(want, upload.Path)
		files = append(files, packageFile{subject: upload.Path, path: local, upload: upload})
	}
	if len(want) != 0 {
		return 0, exit.Internalf("package uploads omitted %d source files", len(want))
	}

	type outcome struct {
		moved int64
		err   *exit.Error
	}
	results := make(chan outcome, len(files))
	var group sync.WaitGroup
	for _, file := range files {
		if file.upload.AlreadyUploaded {
			continue
		}
		if file.upload.URL == "" {
			return 0, exit.Internalf("package upload returned no URL for %s", file.subject)
		}
		group.Add(1)
		go func(file packageFile) {
			defer group.Done()
			uploaded, bytes, problem := transfer.UploadPresigned(ctx, file.subject, file.path,
				file.upload.URL, file.upload.RequiredHeaders)
			if !uploaded {
				bytes = 0
			}
			results <- outcome{moved: bytes, err: problem}
		}(file)
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

func packageReleaseRef(value string) (hub.Ref, string, *exit.Error) {
	name, release, ok := strings.Cut(strings.TrimSpace(value), "@")
	if !ok || release == "" || strings.Contains(release, "@") || strings.ContainsAny(release, `/\\`) {
		return hub.Ref{}, "", exit.Usagef("%q is not <org/package>@<release>", value)
	}
	ref, problem := hub.ParseRef(name)
	return ref, release, problem
}
