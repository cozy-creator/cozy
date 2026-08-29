package app

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/endpointprofile"
	"github.com/cozy-creator/cozy-creator/internal/endpointpublish"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/render"
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
	if ctx.Inv.Bool("--create") {
		if _, problem := c.CreateEndpoint(hctx, ref.Org, ref.Name, reason); problem != nil {
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
	if done.Release != release || done.DeclarationDigest != pack.Declaration.Digest() {
		return exit.Internalf("endpoint finalize returned another release or declaration digest")
	}
	profileRows := make([]string, 0, len(done.Profiles))
	for _, profile := range done.Profiles {
		profileRows = append(profileRows, profile.Profile+":"+profile.State)
	}
	sort.Strings(profileRows)
	return emit(ctx, render.Record{Kind: "endpoint publication", Fields: []render.Field{
		{K: "endpoint", V: ref.String()}, {K: "release", V: done.Release},
		{K: "declaration", V: done.DeclarationDigest}, {K: "created", V: done.Created},
		{K: "profiles", V: profileRows}, {K: "endpoint_executions", V: len(done.EndpointExecutions)},
		{K: "uploaded", V: render.Bytes(moved)}, {K: "held", V: render.Bytes(held)},
		{K: "hub", V: c.Base()},
	}, Notes: []string{
		"source/project bytes were published once; profile candidates are non-serving until explicit hardware qualification",
		"no endpoint image, Dockerfile, dependency resolver, native build, or serving-pointer move ran",
	}, Next: []string{
		"cozy endpoint qualify " + ref.String() + "@" + release + " --profile <profile> --gpu <model> --max-cost <usd> --reason <why>",
	}})
}

func validateBegin(pack *endpointpublish.Package, begun hub.EndpointReleaseBegin) *exit.Error {
	if begun.DeclarationDigest != pack.Declaration.Digest() ||
		(begun.State != "pending" && begun.State != "committed") {
		return exit.Internalf("endpoint begin returned another declaration or invalid state %q", begun.State)
	}
	wantProfiles := pack.Declaration.Profiles
	gotProfiles := make([]string, 0, len(begun.Profiles))
	for _, row := range begun.Profiles {
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
	}
	if len(seen) != len(want) {
		return exit.Internalf("endpoint begin returned %d of %d declared upload roles", len(seen), len(want))
	}
	return nil
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

func handleEndpointQualify(ctx *Context) *exit.Error {
	ref, release, problem := endpointReleaseRef(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	profiles, problem := endpointprofile.NormalizeSet(ctx.Inv.Values["--profile"])
	if problem != nil {
		return problem
	}
	if len(profiles) != 1 {
		return exit.Usagef("`cozy endpoint qualify` takes exactly one --profile per paid operation")
	}
	gpu := strings.TrimSpace(ctx.Inv.Value("--gpu"))
	if gpu == "" {
		return exit.Usagef("`cozy endpoint qualify` needs --gpu <provider-neutral-model>")
	}
	maxCost, problem := parseUSDMicros(ctx.Inv.Value("--max-cost"))
	if problem != nil {
		return problem
	}
	durationText := strings.TrimSpace(ctx.Inv.Value("--duration"))
	if durationText == "" {
		durationText = "15m"
	}
	duration, err := time.ParseDuration(durationText)
	if err != nil || duration <= 0 || duration > 24*time.Hour {
		return exit.Usagef("--duration %q is not a positive duration at or below 24h", durationText)
	}
	reason, problem := mutationReason(ctx, "endpoint qualify")
	if problem != nil {
		return problem
	}
	c := client(ctx)
	hctx, cancel := hub.LongContext()
	if timeoutText := strings.TrimSpace(ctx.Inv.Value("--timeout")); timeoutText != "" {
		timeout, err := time.ParseDuration(timeoutText)
		if err != nil || timeout <= 0 {
			cancel()
			return exit.Usagef("--timeout %q is not a positive duration", timeoutText)
		}
		cancel()
		hctx, cancel = context.WithTimeout(context.Background(), timeout)
	}
	defer cancel()
	qualified, problem := c.QualifyEndpointProfile(hctx, ref, release, profiles[0],
		hub.EndpointQualificationRequest{AcceleratorModel: gpu,
			ProviderExposureLimitUSDMicros: maxCost, DurationCapSeconds: int64(duration / time.Second)}, reason)
	if problem != nil {
		return problem
	}
	if qualified.CandidateID == "" || qualified.QualificationID == "" || qualified.State == "" {
		return exit.Internalf("endpoint qualification returned no candidate, qualification id, or state")
	}
	qualified, problem = waitQualification(ctx, hctx, c, ref, release, profiles[0], qualified)
	if problem != nil {
		return problem
	}
	return emit(ctx, qualificationRecord(c.Base(), ref, release, profiles[0], qualified))
}

func waitQualification(ctx *Context, hctx context.Context, c *hub.Client, ref hub.Ref,
	release, profile string, current hub.EndpointQualification,
) (hub.EndpointQualification, *exit.Error) {
	priorState := ""
	for {
		terminal := false
		switch current.State {
		case "pending", "acquiring", "booting", "running":
			if current.ReclaimProven {
				return current, exit.Internalf("transient qualification state %q incorrectly claims provider reclaim proof", current.State)
			}
		case "qualified", "refused", "failed", "canceled":
			if !current.ReclaimProven {
				return current, exit.Internalf("terminal qualification state %q arrived before provider reclaim proof", current.State)
			}
			terminal = true
		default:
			return current, exit.Internalf("Tensorhub returned unknown qualification state %q", current.State)
		}
		if terminal {
			break
		}
		if current.State != priorState {
			fmt.Fprintf(ctx.Err, "  qualification %s: %s (reclaimed=%t)\n",
				current.QualificationID, current.State, current.ReclaimProven)
			priorState = current.State
		}
		select {
		case <-hctx.Done():
			return current, exit.Named(exit.Deadline, "endpoint.qualification_wait_deadline",
				"qualification %s is %s; the caller wait ended before terminal reclaim proof",
				current.QualificationID, current.State).
				WithRemedy("the paid operation remains Tensorhub-owned; re-run the exact qualify command to continue observing it")
		case <-time.After(2 * time.Second):
		}
		next, problem := c.EndpointProfileQualification(hctx, ref, release, profile)
		if problem != nil {
			return current, problem
		}
		if current.QualificationID != "" && (next.QualificationID != current.QualificationID ||
			next.CandidateID != current.CandidateID) {
			return current, exit.Internalf("qualification read changed operation or candidate identity")
		}
		current = next
	}
	if current.State != "qualified" {
		return current, exit.Named(exit.Failed, "endpoint.qualification_"+current.State,
			"qualification %s ended %s after %s; cost %s; reclaimed=%t",
			current.QualificationID, current.State, current.AcceleratorModel,
			microUSD(current.ObservedCostUSDMicros), current.ReclaimProven).
			WithRemedy("inspect the banked admission/observation digests, repair the release/profile, and submit a new explicit qualification")
	}
	return current, nil
}

func qualificationRecord(base string, ref hub.Ref, release, profile string,
	q hub.EndpointQualification,
) render.Record {
	rec := render.Record{Kind: "endpoint qualification", Fields: []render.Field{
		{K: "endpoint", V: ref.String()}, {K: "release", V: release},
		{K: "profile", V: profile}, {K: "qualification", V: q.QualificationID},
		{K: "candidate", V: q.CandidateID}, {K: "state", V: q.State},
		{K: "gpu", V: q.AcceleratorModel}, {K: "provider_resource", V: q.ProviderResourceID},
		{K: "exposure", V: microUSD(q.ProviderExposureLimitUSDMicros)},
		{K: "cost", V: microUSD(q.ObservedCostUSDMicros)}, {K: "reclaimed", V: q.ReclaimProven},
		{K: "qualification_spec", V: q.ModelQualificationSpecDigest},
		{K: "execution_observation", V: q.ExecutionObservationDigest},
		{K: "admission_decision", V: q.ModelAdmissionDecisionDigest},
		{K: "hub", V: base},
	}}
	if q.State == "qualified" {
		rec.Next = []string{"cozy endpoint promote " + ref.String() + " " + release + " --serve <vN/function> --reason <why>"}
	}
	return rec
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
	return emit(ctx, render.Record{Kind: "endpoint promotion", Fields: []render.Field{
		{K: "endpoint", V: ref.String()}, {K: "release", V: promoted.Release},
		{K: "serving", V: rows}, {K: "hub", V: c.Base()},
	}, Notes: []string{"the serving pointers moved in one Tensorhub transaction; endpoint publish and qualify never call this route"}})
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
			WithRemedy("Tensorhub records why every mutation or paid acquisition happened before it acts")
	}
	return reason, nil
}

func parseUSDMicros(raw string) (int64, *exit.Error) {
	value := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(raw), "$"))
	whole, fraction, found := strings.Cut(value, ".")
	if !found {
		fraction = ""
	}
	if whole == "" {
		whole = "0"
	}
	if len(fraction) > 6 {
		return 0, exit.Usagef("--max-cost %q has more than six decimal USD places", raw)
	}
	for len(fraction) < 6 {
		fraction += "0"
	}
	if strings.HasPrefix(whole, "+") || strings.HasPrefix(whole, "-") {
		return 0, exit.Usagef("--max-cost %q is not a positive USD amount", raw)
	}
	dollars, err1 := strconv.ParseInt(whole, 10, 64)
	micros, err2 := strconv.ParseInt(fraction, 10, 64)
	if err1 != nil || err2 != nil || dollars > (1<<53-1)/1_000_000 {
		return 0, exit.Usagef("--max-cost %q is not an exact bounded USD amount", raw)
	}
	total := dollars*1_000_000 + micros
	if total <= 0 {
		return 0, exit.Usagef("--max-cost must be greater than zero")
	}
	return total, nil
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
