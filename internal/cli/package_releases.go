package cli

import (
	"context"
	"sort"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/output"
	"github.com/cozy-creator/cozy-creator/internal/packagepublish"
	"github.com/cozy-creator/cozy-creator/internal/transfer"
)

func handlePackagePublish(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	release := strings.TrimSpace(ctx.Inv.Value("--release"))
	reason, problem := mutationReason(ctx, "package publish")
	if problem != nil {
		return problem
	}
	pack, problem := packagepublish.Prepare(packagepublish.Request{
		Tree: ctx.Inv.Value("--dir"), Release: release,
	})
	if problem != nil {
		return problem
	}
	defer pack.Close()

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
		sources, problem := c.PackageReleaseSourceUploads(hctx, ref, release,
			packagepublish.Paths(pack.Files), reason)
		if problem != nil {
			return problem
		}
		moved, problem = uploadPackageFiles(hctx, pack, begun.ProjectWheelUpload, sources.Uploads)
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
		{K: "created", V: done.Created}, {K: "profiles", V: profileRows},
		{K: "package_executions", V: len(done.PackageExecutions)},
		{K: "uploaded", V: output.Bytes(moved)}, {K: "hub", V: c.Base()},
	}
	if len(candidateRows) > 0 {
		fields = append(fields, output.Field{K: "candidates", V: candidateRows})
	}
	if len(refusalRows) > 0 {
		fields = append(fields, output.Field{K: "profile_refusals", V: refusalRows})
	}
	return emit(ctx, compactRecord(fields,
		"package", "release", "status", "profiles", "uploaded", "changed"))
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

func mutationReason(ctx *Context, command string) (string, *exit.Error) {
	reason := strings.TrimSpace(ctx.Inv.Value("--reason"))
	if reason == "" {
		return "", exit.Usagef("`cozy %s` needs --reason <why>", command).
			WithRemedy("Tensorhub records why every mutation or paid acquisition happened before it acts")
	}
	return reason, nil
}
