package cli

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/endpointpublish"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/output"
	"github.com/cozy-creator/cozy-creator/internal/transfer"
)

func handleEndpointPublish(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	release := strings.TrimSpace(ctx.Inv.Value("--release"))
	reason, problem := mutationReason(ctx, "endpoint publish")
	if problem != nil {
		return problem
	}
	pack, problem := endpointpublish.Prepare(endpointpublish.Request{
		Tree: ctx.Inv.Value("--dir"), Release: release,
		Profiles:     append([]string(nil), ctx.Inv.Values["--profile"]...),
		CustomWheels: append([]string(nil), ctx.Inv.Values["--custom-wheel"]...),
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
	if _, problem := c.Endpoint(hctx, ref); problem != nil {
		if problem.Code != exit.NotFound {
			return problem
		}
		if _, problem := c.CreateEndpoint(hctx, ref.Org, ref.Name, reason); problem != nil && problem.Code != exit.Conflict {
			return problem
		}
	}
	begun, problem := c.BeginEndpointRelease(hctx, ref, release, declaration, reason)
	if problem != nil {
		return problem
	}
	if problem := validateBegin(pack, begun); problem != nil {
		return problem
	}
	moved, held, problem := uploadEndpointRoles(hctx, pack, begun.Uploads)
	if problem != nil {
		return problem
	}
	done, problem := c.FinalizeEndpointRelease(hctx, ref, release, declaration, reason)
	if problem != nil {
		return problem
	}
	if problem := validateFinalize(pack, release, done); problem != nil {
		return problem
	}
	profileRows := make([]string, 0, len(done.Profiles))
	var candidateIDRows, refusalRows []string
	for _, profile := range done.Profiles {
		profileRows = append(profileRows, profile.Profile+":"+profile.State+":"+profile.BaseRealizationKind)
		if profile.State != "refused" {
			candidateIDRows = append(candidateIDRows, profile.Profile+"/"+profile.BaseRealizationKind+"="+profile.CandidateID)
		} else {
			refusal := profile.Profile + "/" + profile.BaseRealizationKind + " " + profile.RefusalCode
			if profile.RefusalDetail != "" {
				refusal += ": " + shorten(profile.RefusalDetail, 240)
			}
			refusalRows = append(refusalRows, refusal)
		}
	}
	sort.Strings(profileRows)
	sort.Strings(candidateIDRows)
	sort.Strings(refusalRows)
	fields := []output.Field{
		{K: "endpoint", V: ref.String()}, {K: "release", V: done.Release},
		{K: "declaration", V: done.DeclarationDigest}, {K: "created", V: done.Created},
		{K: "profiles", V: profileRows}, {K: "endpoint_executions", V: len(done.EndpointExecutions)},
		{K: "uploaded", V: output.Bytes(moved)}, {K: "held", V: output.Bytes(held)},
		{K: "hub", V: c.Base()},
	}
	if len(candidateIDRows) > 0 {
		fields = append(fields, output.Field{K: "candidates", V: candidateIDRows})
	}
	if len(refusalRows) > 0 {
		fields = append(fields, output.Field{K: "profile_refusals", V: refusalRows})
	}
	return emit(ctx, output.Record{Kind: "endpoint publication", Fields: fields, Notes: []string{
		"the endpoint name was created idempotently when absent",
		"source/project bytes were published once; profile candidates are non-serving until Tensorhub qualification",
		"no endpoint image, Dockerfile, dependency resolver, native build, or serving-pointer move ran",
	}, Next: []string{"cozy endpoint install " + ref.String() + "@" + release}})
}

func shorten(value string, limit int) string {
	value = strings.Join(strings.Fields(value), " ")
	runes := []rune(value)
	if len(runes) > limit {
		return string(runes[:limit]) + "…"
	}
	return value
}

func validateBegin(pack *endpointpublish.Package, begun hub.EndpointReleaseBegin) *exit.Error {
	if begun.DeclarationDigest != pack.Declaration.Digest() ||
		(begun.State != "pending" && begun.State != "committed") {
		return exit.Internalf("endpoint begin returned another declaration or invalid state %q", begun.State)
	}
	wantProfiles := pack.Declaration.Profiles
	gotProfiles := make([]string, 0, len(begun.Profiles))
	for _, row := range begun.Profiles {
		if row.State != "candidate_pending" || row.CandidateID != "" ||
			row.BaseRealizationKind != "" || row.BaseRealizationDigest != "" ||
			row.RefusalCode != "" || row.RefusalDetail != "" {
			return exit.Internalf("endpoint begin returned a non-pending profile row for %q", row.Profile)
		}
		gotProfiles = append(gotProfiles, row.Profile)
	}
	sort.Strings(gotProfiles)
	if !sameStrings(wantProfiles, gotProfiles) {
		return exit.Internalf("endpoint begin profile set differs from the frozen declaration")
	}
	want := declarationRoles(pack)
	seen := map[string]bool{}
	for _, upload := range begun.Uploads {
		local, ok := want[upload.Role]
		if !ok || seen[upload.Role] {
			return exit.Internalf("endpoint begin returned unknown or duplicate upload role %q", upload.Role)
		}
		seen[upload.Role] = true
		if upload.Ref.Digest != local.ref.Digest || upload.Ref.Length != local.ref.Length {
			return exit.Internalf("endpoint begin changed the identity of role %s", upload.Role)
		}
		if !upload.AlreadyHeld && upload.URL == "" {
			return exit.Internalf("endpoint begin returned no URL for missing role %s", upload.Role)
		}
		if !upload.AlreadyHeld {
			expires, err := time.Parse(time.RFC3339, upload.ExpiresAt)
			if err != nil || !expires.After(time.Now()) {
				return exit.Internalf("endpoint begin returned an absent, malformed, or expired upload grant for %s", upload.Role)
			}
		}
	}
	if len(seen) != len(want) {
		return exit.Internalf("endpoint begin returned %d of %d declared upload roles", len(seen), len(want))
	}
	return nil
}

func validateFinalize(pack *endpointpublish.Package, release string,
	done hub.EndpointReleaseFinalize,
) *exit.Error {
	if done.Release != release || done.DeclarationDigest != pack.Declaration.Digest() {
		return exit.Internalf("endpoint finalize returned another release or declaration digest")
	}
	declared := map[string]bool{}
	for _, profile := range pack.Declaration.Profiles {
		declared[profile] = true
	}
	seenProfiles := map[string]bool{}
	seenRealizations := map[string]bool{}
	executionProfileStates := map[string]string{}
	for _, row := range done.Profiles {
		if !declared[row.Profile] {
			return exit.Internalf("endpoint finalize returned unknown profile %q", row.Profile)
		}
		seenProfiles[row.Profile] = true
		key := row.Profile + "\x00" + row.BaseRealizationKind
		if seenRealizations[key] || (row.BaseRealizationKind != "oci" && row.BaseRealizationKind != "managed-local") ||
			row.CandidateID == "" || !baseRealization(row.BaseRealizationKind, row.BaseRealizationDigest) ||
			!objectRef(row.EndpointEnvironmentSpec) || !objectRef(row.ResolvedWheelSet) ||
			!objectRef(row.ResolutionLock) {
			return exit.Internalf("endpoint finalize returned malformed or duplicate realization %q/%q",
				row.Profile, row.BaseRealizationKind)
		}
		seenRealizations[key] = true
		if !endpointProfileState(row.State) {
			return exit.Internalf("endpoint finalize returned unknown profile state %q", row.State)
		}
		if row.State == "refused" && row.RefusalCode == "" {
			return exit.Internalf("endpoint finalize returned a refusal without a typed code")
		}
		if row.State != "refused" {
			if row.RefusalCode != "" || row.RefusalDetail != "" {
				return exit.Internalf("endpoint finalize returned refusal detail on a non-refused profile")
			}
			if row.BaseRealizationKind == "oci" {
				executionProfileStates[row.Profile] = row.State
			}
		}
	}
	if len(seenProfiles) != len(declared) {
		return exit.Internalf("endpoint finalize returned %d of %d declared profiles", len(seenProfiles), len(declared))
	}
	seenExecutions := map[string]bool{}
	for _, execution := range done.EndpointExecutions {
		key := execution.Profile + "\x00" + execution.Function
		if executionProfileStates[execution.Profile] == "" || strings.TrimSpace(execution.Function) == "" ||
			seenExecutions[key] || !sha256Digest(execution.Digest) ||
			execution.State != executionProfileStates[execution.Profile] {
			return exit.Internalf("endpoint finalize returned malformed or duplicate execution %q/%q",
				execution.Profile, execution.Function)
		}
		seenExecutions[key] = true
	}
	return nil
}

func endpointProfileState(state string) bool {
	return state == "candidate" || state == "qualified" || state == "refused"
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
	ref  endpointpublish.ObjectRef
	path string
}

func declarationRoles(pack *endpointpublish.Package) map[string]localRole {
	d := pack.Declaration
	out := map[string]localRole{
		"source_archive":   {ref: d.SourceArchive, path: pack.Files["source_archive"]},
		"source_lock":      {ref: d.SourceLock, path: pack.Files["source_lock"]},
		"project_wheel":    {ref: endpointpublish.ObjectRef{Digest: d.ProjectWheel.Digest, Length: d.ProjectWheel.Length}, path: pack.Files["project_wheel"]},
		"descriptor":       {ref: d.Descriptor, path: pack.Files["descriptor"]},
		"evaluated_config": {ref: d.EvaluatedConfig, path: pack.Files["evaluated_config"]},
	}
	for _, custom := range d.CustomWheels {
		role := "custom_wheel:" + custom.Wheel.Distribution + ":" + strings.TrimPrefix(custom.Wheel.Digest, "sha256:")
		out[role] = localRole{ref: endpointpublish.ObjectRef{Digest: custom.Wheel.Digest,
			Length: custom.Wheel.Length}, path: pack.Files[role]}
	}
	return out
}

func uploadEndpointRoles(ctx context.Context, pack *endpointpublish.Package,
	uploads []hub.EndpointUpload,
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
		go func(upload hub.EndpointUpload, local localRole) {
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

func handleEndpointPromote(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	release := strings.TrimSpace(ctx.Inv.Args[1])
	if release == "" {
		return exit.Usagef("endpoint release id is required")
	}
	serving, problem := servingTargets(ctx.Inv.Values["--serve"])
	if problem != nil {
		return problem
	}
	reason, problem := mutationReason(ctx, "endpoint promote")
	if problem != nil {
		return problem
	}
	c := client(ctx)
	hctx, cancel := hub.LongContext()
	defer cancel()
	promoted, problem := c.PromoteEndpointRelease(hctx, ref, release, serving, reason)
	if problem != nil {
		return problem
	}
	var rows []string
	for _, row := range promoted.Serving {
		rows = append(rows, row.EndpointRef+":"+strings.Join(row.EndpointExecutionDigests, ","))
	}
	sort.Strings(rows)
	return emit(ctx, output.Record{Kind: "endpoint promotion", Fields: []output.Field{
		{K: "endpoint", V: ref.String()}, {K: "release", V: promoted.Release},
		{K: "serving", V: rows}, {K: "hub", V: c.Base()},
	}, Notes: []string{"the serving pointers moved in one Tensorhub transaction; endpoint publication never moves them implicitly"}})
}

var functionName = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_-]{0,127}$`)

func servingTargets(values []string) ([]hub.ServingTarget, *exit.Error) {
	seen := map[string]bool{}
	var out []hub.ServingTarget
	for _, value := range values {
		major, function, ok := strings.Cut(strings.TrimSpace(value), "/")
		if !ok || !regexp.MustCompile(`^v[1-9][0-9]*$`).MatchString(major) ||
			!functionName.MatchString(function) || strings.Contains(function, "/") {
			return nil, exit.Usagef("--serve %q is not vN/function; a bare major has no atomic function set", value)
		}
		key := major + "/" + function
		if !seen[key] {
			seen[key] = true
			out = append(out, hub.ServingTarget{Major: major, Function: function})
		}
	}
	if len(out) == 0 {
		return nil, exit.Usagef("`cozy endpoint promote` needs at least one --serve <vN/function>")
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Major != out[j].Major {
			return out[i].Major < out[j].Major
		}
		return out[i].Function < out[j].Function
	})
	return out, nil
}

func endpointReleaseRef(value string) (hub.Ref, string, *exit.Error) {
	name, release, ok := strings.Cut(strings.TrimSpace(value), "@")
	if !ok || release == "" || strings.Contains(release, "@") || strings.ContainsAny(release, `/\`) {
		return hub.Ref{}, "", exit.Usagef("%q is not <org/endpoint>@<release>", value)
	}
	ref, problem := hub.ParseRef(name)
	return ref, release, problem
}

func mutationReason(ctx *Context, command string) (string, *exit.Error) {
	reason := strings.TrimSpace(ctx.Inv.Value("--reason"))
	if reason == "" {
		return "", exit.Usagef("`cozy %s` needs --reason <why>", command).
			WithRemedy("Tensorhub records why every mutation happened before it acts")
	}
	return reason, nil
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func fmtProfileState(row hub.EndpointProfileState) string {
	return fmt.Sprintf("%s:%s", row.Profile, row.State)
}
