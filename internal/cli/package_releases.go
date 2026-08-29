package cli

import (
	"context"
	"sort"
	"strings"
	"sync"
	"time"

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
	declaration, problem := pack.Declaration.CanonicalBytes()
	if problem != nil {
		return problem
	}

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
	begun, problem := c.BeginPackageRelease(hctx, ref, release, declaration, reason)
	if problem != nil {
		return problem
	}
	if problem := validateBegin(pack, begun); problem != nil {
		return problem
	}
	moved, held, problem := uploadPackageRoles(hctx, pack, begun.Uploads)
	if problem != nil {
		return problem
	}
	done, problem := c.FinalizePackageRelease(hctx, ref, release, declaration, reason)
	if problem != nil {
		return problem
	}
	if problem := validateFinalize(pack, release, done); problem != nil {
		return problem
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
		{K: "declaration", V: done.DeclarationDigest}, {K: "created", V: done.Created},
		{K: "profiles", V: profileRows}, {K: "package_executions", V: len(done.PackageExecutions)},
		{K: "uploaded", V: output.Bytes(moved)}, {K: "held", V: output.Bytes(held)},
		{K: "hub", V: c.Base()},
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

func validateBegin(pack *packagepublish.Package, begun hub.PackageReleaseBegin) *exit.Error {
	if begun.DeclarationDigest != pack.Declaration.Digest() ||
		(begun.State != "pending" && begun.State != "committed") {
		return exit.Internalf("package begin returned another declaration or invalid state %q", begun.State)
	}
	want := declarationRoles(pack)
	seen := map[string]bool{}
	for _, upload := range begun.Uploads {
		local, ok := want[upload.Role]
		if !ok || seen[upload.Role] {
			return exit.Internalf("package begin returned unknown or duplicate upload role %q", upload.Role)
		}
		seen[upload.Role] = true
		if upload.Ref.Digest != local.ref.Digest || upload.Ref.Length != local.ref.Length {
			return exit.Internalf("package begin changed the identity of role %s", upload.Role)
		}
		if !upload.AlreadyHeld && upload.URL == "" {
			return exit.Internalf("package begin returned no URL for missing role %s", upload.Role)
		}
		if !upload.AlreadyHeld {
			expires, err := time.Parse(time.RFC3339, upload.ExpiresAt)
			if err != nil || !expires.After(time.Now()) {
				return exit.Internalf("package begin returned an absent, malformed, or expired upload grant for %s", upload.Role)
			}
		}
	}
	if len(seen) != len(want) {
		return exit.Internalf("package begin returned %d of %d declared upload roles", len(seen), len(want))
	}
	return nil
}

func validateFinalize(pack *packagepublish.Package, release string,
	done hub.PackageReleaseFinalize,
) *exit.Error {
	if done.Release != release || done.DeclarationDigest != pack.Declaration.Digest() {
		return exit.Internalf("package finalize returned another release or declaration digest")
	}
	seenProfiles := map[string]bool{}
	seenRealizations := map[string]bool{}
	qualifiedProfiles := map[string]bool{}
	for _, row := range done.Profiles {
		if row.Profile == "" || seenProfiles[row.Profile] {
			return exit.Internalf("package finalize returned an empty or duplicate profile %q", row.Profile)
		}
		seenProfiles[row.Profile] = true
		key := row.Profile + "\x00" + row.BaseRealizationKind
		if seenRealizations[key] || (row.BaseRealizationKind != "oci" && row.BaseRealizationKind != "managed-local") ||
			row.CandidateID == "" || !baseRealization(row.BaseRealizationKind, row.BaseRealizationDigest) ||
			!objectRef(row.PackageEnvironmentSpec) || !objectRef(row.ResolvedWheelSet) ||
			!objectRef(row.ResolutionLock) {
			return exit.Internalf("package finalize returned malformed or duplicate realization %q/%q",
				row.Profile, row.BaseRealizationKind)
		}
		seenRealizations[key] = true
		if row.State != "qualified" && row.State != "refused" {
			return exit.Internalf("package finalize returned unknown profile state %q", row.State)
		}
		if row.State == "refused" && row.RefusalCode == "" {
			return exit.Internalf("package finalize returned a refusal without a typed code")
		}
		if row.State == "qualified" {
			if row.RefusalCode != "" || row.RefusalDetail != "" {
				return exit.Internalf("package finalize returned refusal detail on a passing candidate")
			}
			qualifiedProfiles[row.Profile] = true
		}
	}
	seenExecutions := map[string]bool{}
	for _, execution := range done.PackageExecutions {
		key := execution.Profile + "\x00" + execution.Function
		if !qualifiedProfiles[execution.Profile] || strings.TrimSpace(execution.Function) == "" ||
			seenExecutions[key] || !sha256Digest(execution.Digest) ||
			execution.State != "qualified" {
			return exit.Internalf("package finalize returned malformed or duplicate execution %q/%q",
				execution.Profile, execution.Function)
		}
		seenExecutions[key] = true
	}
	return nil
}

func objectRef(ref hub.ObjectRef) bool { return sha256Digest(ref.Digest) && ref.Length > 0 }

func baseRealization(kind, value string) bool {
	if kind == "managed-local" {
		return sha256Digest(value)
	}
	repository, manifest, ok := strings.Cut(value, "@")
	return kind == "oci" && ok && repository != "" && !strings.ContainsAny(repository, "@ \t\r\n") &&
		sha256Digest(manifest)
}

func sha256Digest(value string) bool {
	if len(value) != 71 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	for _, r := range strings.TrimPrefix(value, "sha256:") {
		if !strings.ContainsRune("0123456789abcdef", r) {
			return false
		}
	}
	return true
}

type localRole struct {
	ref  packagepublish.ObjectRef
	path string
}

func declarationRoles(pack *packagepublish.Package) map[string]localRole {
	d := pack.Declaration
	out := map[string]localRole{
		"source_archive": {ref: d.SourceArchive, path: pack.Files["source_archive"]},
		"source_lock":    {ref: d.SourceLock, path: pack.Files["source_lock"]},
		"project_wheel":  {ref: packagepublish.ObjectRef{Digest: d.ProjectWheel.Digest, Length: d.ProjectWheel.Length}, path: pack.Files["project_wheel"]},
		"descriptor":     {ref: d.Descriptor, path: pack.Files["descriptor"]},
	}
	return out
}

func uploadPackageRoles(ctx context.Context, pack *packagepublish.Package,
	uploads []hub.PackageUpload,
) (int64, int64, *exit.Error) {
	want := declarationRoles(pack)
	type outcome struct {
		role  string
		moved int64
		held  int64
		err   *exit.Error
	}
	results := make(chan outcome, len(uploads))
	var group sync.WaitGroup
	for _, upload := range uploads {
		local := want[upload.Role]
		if upload.AlreadyHeld {
			results <- outcome{role: upload.Role, held: local.ref.Length}
			continue
		}
		group.Add(1)
		go func(upload hub.PackageUpload, local localRole) {
			defer group.Done()
			moved, problem := transfer.UploadPresigned(ctx, upload.Role, local.path,
				upload.URL, local.ref.Digest, local.ref.Length, upload.RequiredHeaders)
			result := outcome{role: upload.Role, err: problem}
			if moved {
				result.moved = local.ref.Length
			} else if problem == nil {
				result.held = local.ref.Length
			}
			results <- result
		}(upload, local)
	}
	group.Wait()
	close(results)
	var moved, held int64
	var failures []outcome
	for result := range results {
		moved += result.moved
		held += result.held
		if result.err != nil {
			failures = append(failures, result)
		}
	}
	if len(failures) > 0 {
		sort.Slice(failures, func(i, j int) bool { return failures[i].role < failures[j].role })
		return moved, held, failures[0].err
	}
	return moved, held, nil
}

func packageReleaseRef(value string) (hub.Ref, string, *exit.Error) {
	name, release, ok := strings.Cut(strings.TrimSpace(value), "@")
	if !ok || release == "" || strings.Contains(release, "@") || strings.ContainsAny(release, `/\`) {
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
