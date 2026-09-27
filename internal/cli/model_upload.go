package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/records"
)

type publishSource struct {
	Canonical string
	Selection string
	License   string
	Lane      string
	Files     int
	Bytes     int64
	Exact     []modeltransfer.SourceFile
	Access    []sourceCapability
	// The provider resolution this source came from, ALREADY NARROWED to the reviewed
	// carriers, and the resolver that answered it. Carried so a header-first conversion
	// preflight (tfs-076) can range-read those carriers' headers without re-asking the
	// provider everything it has just answered. Nil for a local file, a `local/` alias,
	// or a Tensorhub checkpoint — none of which has a provider header to read.
	Resolver   *modelsource.Resolver
	Resolution modelsource.Plan
}

type sourceCapability struct {
	Member, ObjectID, Provider, URL string
	Length                          int64
	ExpiresAtUnix                   uint64
}

// sourceInvocation is the already resolved ordinary job. Source acquisition does
// not select another package or construct another payload.
type sourceInvocation struct {
	Target   Target
	Job      *launch.Entrypoint
	Profiles map[string]string
}

func sourceProfileNames(profiles map[string]string) []string {
	names := make([]string, 0, len(profiles))
	for _, name := range profiles {
		names = append(names, name)
	}
	sort.Strings(names)
	return slices.Compact(names)
}

func handleModelUpload(ctx *Context) *exit.Error {
	adoptRentalHub(ctx, ctx.Inv.Value("--rental"))
	if handled, problem := nativeModelUpload(ctx); handled {
		return problem
	}
	return handleModelTransfer(ctx, "model-upload")
}

func handleModelDownload(ctx *Context) *exit.Error {
	if ctx.Inv.Value("--rental") != "" {
		return handleRentalModelDownload(ctx)
	}
	if len(ctx.Inv.Args) < 2 || strings.TrimSpace(ctx.Inv.Args[1]) == "" {
		return exit.Usagef("local model download requires a local/name destination; use --rental=NAME for a remote download")
	}
	return handleModelTransfer(ctx, "model-download")
}

func handleModelTransfer(ctx *Context, kind string) *exit.Error {
	return submitSourceTransfer(ctx, kind, ctx.Inv.Args[0], ctx.Inv.Args[1], nil,
		api.JobSubmission{Input: []byte("{}")})
}

// submitSourceTransfer is shared by ordinary modeled jobs and standalone ingest.
// It freezes the existing source inventory and headers before submitting the same
// JobSubmission and ModelTransferIntent consumed by the resumable transfer owner.
func submitSourceTransfer(ctx *Context, kind, sourceArg, destinationArg string,
	invocation *sourceInvocation, submission api.JobSubmission,
) *exit.Error {
	destinationArg = strings.TrimSpace(destinationArg)
	destination := destinationArg
	privateOutputs := invocation != nil && destinationArg == ""
	if privateOutputs {
		submission.RetainWork = true
	} else if kind == "model-upload" {
		ref, problem := hub.ParseRef(destinationArg)
		if problem != nil {
			return problem
		}
		if ref.Org == "local" {
			return exit.Usagef("local/ is reserved for private aliases and cannot be a Tensorhub destination").
				WithRemedy("upload under your Tensorhub account, for example alice/%s", ref.Name)
		}
		destination = ref.String()
	} else {
		name, local, problem := modelsource.LocalAlias(destinationArg)
		if !local {
			return exit.Usagef("model download destination %q is not local/name", destinationArg)
		}
		if problem != nil {
			return problem
		}
		destination = "local/" + name
		if ctx.Inv.Bool("--rental-only") {
			return exit.Named(exit.Unavailable, "model_download.rented_return_unavailable",
				"rented model download cannot yet return an output to local TensorFS").
				WithRemedy("run without --rental or --rental-only; tracked remote-return support is not landed")
		}
	}
	preflightLocalOnly := localOnlyModelSource(sourceArg)
	if !(preflightLocalOnly && ctx.Inv.Bool("--rental") && !ctx.Inv.Bool("--rental-only")) {
		if problem := validateRunPlacement(ctx); problem != nil {
			return problem
		}
	}
	var suppliedProfiles map[string]string
	if invocation != nil {
		suppliedProfiles = invocation.Profiles
	}
	localOnly := preflightLocalOnly
	if localOnly && (ctx.Inv.Bool("--rental-only") || ctx.Inv.Value("--rental") != "") {
		return exit.Usagef("a local model source cannot run under --rental-only").
			WithRemedy("omit --rental-only or use an addressable provider/Tensorhub source")
	}
	effectiveRental := rentalRequested(ctx) && !localOnly
	if kind == "model-download" && effectiveRental {
		return exit.Named(exit.Unavailable, "model_download.rented_return_unavailable",
			"rented model download cannot yet return an output to local TensorFS").
			WithRemedy("run locally until the negotiated weights-read return plane is active")
	}
	if invocation == nil && effectiveRental {
		return exit.Named(exit.Unavailable, "model_transfer.rented_source_unavailable",
			"a rented worker ingests only Hugging Face and Civitai sources").
			WithRemedy("upload this source without --rental")
	}
	if invocation == nil && len(ctx.Inv.Values["--source-profile"]) > 0 {
		return exit.Usagef("--source-profile selects the profiles a rented ingest converts").
			WithRemedy("add --rental=<name>, or omit --source-profile to use the one profile the headers match")
	}
	if localOnly {
		ctx.Inv.Bools["--rental"] = false
		ctx.Inv.Bools["--rental-only"] = false
	}
	sourceProfiles := sourceProfileNames(suppliedProfiles)
	source, problem := resolvePublishSource(ctx, sourceArg, sourceProfiles)
	if problem != nil {
		return problem
	}
	if effectiveRental && catalogModelSpelling(source.Canonical) {
		return exit.Named(exit.Unavailable, "model_transfer.rented_catalog_source_unavailable",
			"rented model transfer cannot yet bind a Tensorhub checkpoint through the worker download set").
			WithRemedy("run locally or use the original pinned provider source until the tracked catalog binding lands")
	}
	plan := modeltransfer.Plan{
		Kind: kind, Destination: destination,
		Source: source.Canonical, SourceSelection: source.Selection,
		SourceLicense: source.License, InputLane: source.Lane,
		SourceFiles: source.Exact,
	}
	if invocation != nil {
		plan.SourceProfiles = suppliedProfiles
		for _, output := range invocation.Job.WeightsOutputs {
			plan.Outputs = append(plan.Outputs, modeltransfer.OutputPin{Name: output.OutputID})
		}
	} else {
		plan.Outputs = []modeltransfer.OutputPin{{Name: "model"}}
	}
	// THE GUARD (tfs-076). A conversion plan is a function of the source HEADERS, so it is
	// decidable HERE — owner-side, before a rental is requested, before a byte of payload
	// moves, and before this process submits anything. Runs 290, 294 and 309 each moved
	// 210.3 GB and then refused on facts that were in the first few kilobytes of each
	// member.
	//
	// Placed on the RENTED transfer, which is the one that spends money to find out. A
	// local transfer already plans from headers itself, in prepareLocalTransferSources,
	// and paying for a second header read here would be the same work twice.
	conversion := conversionPreflight{Undecided: "not run on this path"}
	if effectiveRental {
		pctx, cancel := hub.LongContext()
		decided, problem := preflightConversionPlan(pctx, ctx, source, plan.SourceProfiles)
		cancel()
		if problem != nil {
			return problem
		}
		conversion = decided
		if effectiveRental && !conversion.decided() {
			return exit.Named(exit.Unavailable, "model_source.preflight_unavailable",
				"source headers must be inspected before renting: %s", conversion.Undecided).
				WithRemedy("retry after the provider can serve every selected source header")
		}
	}

	if kind == "model-upload" && !privateOutputs {
		ref, _ := hub.ParseRef(destination)
		if _, problem := ownedPublication(ctx, ref); problem != nil {
			return problem
		}
	}
	intent := modelTransferIntent(plan)
	for i := range intent.SourceFiles {
		intent.SourceFiles[i].Header = conversion.Headers[intent.SourceFiles[i].Member]
	}
	intent.LocalOnly = localOnly
	daemonState, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	ctx.Daemon = daemonState
	local, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	submission.ModelTransfer = &intent
	submission.Rental, submission.RentalRequired = effectiveRental, ctx.Inv.Bool("--rental-only") || submission.RequestedRental != ""
	if invocation != nil && kind == "model-upload" && !privateOutputs {
		org := strings.Split(destination, "/")[0]
		if submission.Org != "" && submission.Org != org {
			return exit.Usagef("--org must match the --publish-to organization")
		}
		submission.Org = org
	}
	handle, problem := local.SubmitJob(submission, requestKey(ctx.Inv.Value("--idempotency-key")))
	if invocation != nil {
		releaseSnapshotReader(invocation.Target)
	}
	if problem != nil {
		return problem
	}
	state, problem := local.Job(handle.JobID)
	if problem != nil {
		return problem
	}
	if ctx.Inv.Bool("--await") {
		return watchJob(ctx, local, state)
	}
	return renderSubmittedJob(ctx, state, !handle.Replay)
}

func localOnlyModelSource(source string) bool {
	if strings.HasPrefix(source, "local/") {
		return true
	}
	cwd, err := os.Getwd()
	if err != nil {
		return false
	}
	parsed, problem := modelsource.Parse(source, cwd)
	return problem == nil && parsed.Kind == modelsource.LocalFile
}

func modelTransferIntent(plan modeltransfer.Plan) records.ModelTransferIntent {
	files := make([]records.ModelTransferSourceFile, 0, len(plan.SourceFiles))
	for _, file := range plan.SourceFiles {
		files = append(files, records.ModelTransferSourceFile{Member: file.Member,
			SHA256: file.SHA256, Length: file.Length})
	}
	outputs := make([]records.ModelTransferOutput, 0, len(plan.Outputs))
	for _, output := range plan.Outputs {
		outputs = append(outputs, records.ModelTransferOutput{Name: output.Name})
	}
	return records.ModelTransferIntent{Kind: plan.Kind, Destination: plan.Destination,
		Source: plan.Source, SourceSelection: plan.SourceSelection, SourceLicense: plan.SourceLicense,
		SourceFiles: files, InputLane: plan.InputLane, SourceProfiles: plan.SourceProfiles,
		Outputs: outputs}
}

func resolvePublishSource(ctx *Context, raw string, sourceProfiles []string) (publishSource, *exit.Error) {
	raw = strings.TrimSpace(raw)
	lane := strings.TrimSpace(ctx.Inv.Value("--lane"))
	if name, local, problem := modelsource.LocalAlias(raw); local {
		if problem != nil {
			return publishSource{}, problem
		}
		if ctx.Inv.Bool("--rental") {
			return publishSource{}, exit.Usagef("a local/%s alias cannot be read by a rented worker", name).
				WithRemedy("use its addressable Tensorhub release or original pinned foreign source")
		}
		if lane != "" {
			return publishSource{}, exit.Usagef("--lane selects only a Tensorhub model release")
		}
		tool, _, problem := localTensorFS(ctx)
		if problem != nil {
			return publishSource{}, problem
		}
		alias, problem := tool.ResolveLocal(name)
		if problem != nil {
			return publishSource{}, problem
		}
		return publishSource{Canonical: "local/" + name, Selection: alias.ManifestDigest}, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return publishSource{}, exit.Internalf("cannot resolve the current directory: %s", err)
	}
	parsed, parseProblem := modelsource.Parse(raw, cwd)
	if parseProblem == nil {
		if lane != "" {
			return publishSource{}, exit.Usagef("--lane selects only a Tensorhub model release")
		}
		if parsed.Kind == modelsource.LocalFile {
			if ctx.Inv.Bool("--rental") {
				return publishSource{}, exit.Usagef("a local model file cannot be read by a rented worker").
					WithRemedy("import it locally or use an addressable pinned foreign source")
			}
			file, err := os.Open(parsed.Path)
			if err != nil {
				return publishSource{}, exit.New(exit.NotFound, "cannot open local model source: %s", err)
			}
			defer file.Close()
			hash := sha256.New()
			length, err := io.Copy(hash, file)
			if err != nil || length != parsed.Bytes {
				return publishSource{}, exit.Named(exit.Conflict, "model_source.local_changed",
					"local model source changed while measuring its exact identity")
			}
			digest := hex.EncodeToString(hash.Sum(nil))
			return publishSource{Canonical: parsed.Canonical, Selection: "sha256:" + digest,
				Files: 1, Bytes: parsed.Bytes, Exact: []modeltransfer.SourceFile{{
					Member: filepath.Base(parsed.Path), SHA256: digest, Length: parsed.Bytes}}}, nil
		}
		var token = ctx.Cfg.HuggingFaceToken
		if parsed.Kind == modelsource.Civitai {
			token = ctx.Cfg.CivitaiToken
		}
		resolver, problem := modelsource.NewResolver(parsed.Kind, token)
		if problem != nil {
			return publishSource{}, problem
		}
		hctx, cancel := hub.LongContext()
		defer cancel()
		resolved, problem := resolver.Resolve(hctx, parsed)
		if problem != nil {
			return publishSource{}, problem
		}
		if parsed.Kind == modelsource.HuggingFace && parsed.Revision == "" {
			fmt.Fprintf(ctx.Err, "pinned %s\n", resolved.Canonical)
		}
		// A single carrier is already exact (for example Civitai's primary
		// checkpoint). Multi-carrier provider repositories must be narrowed by
		// TensorFS's reviewed profiles before any body is persisted or granted.
		if len(resolved.Files) > 1 && len(sourceProfiles) > 0 {
			tool, _, problem := localTensorFS(ctx)
			if problem != nil {
				return publishSource{}, problem
			}
			members, problem := tool.SourceProfileMembers(ctx.Cfg.TensorFSRegistry, sourceProfiles)
			if problem != nil {
				return publishSource{}, problem
			}
			resolved, problem = resolved.Select(members)
			if problem != nil {
				return publishSource{}, problem
			}
		}
		exact := make([]modeltransfer.SourceFile, 0, len(resolved.Files))
		access := make([]sourceCapability, 0, len(resolved.Files))
		provider := string(parsed.Kind)
		for _, file := range resolved.Files {
			exact = append(exact, modeltransfer.SourceFile{
				Member: file.Member, SHA256: file.SHA256, Length: file.Length,
			})
			access = append(access, sourceCapability{
				Member: file.Member, ObjectID: "sha256:" + file.SHA256,
				Length: file.Length, Provider: provider, URL: file.URL,
			})
		}
		return publishSource{Canonical: resolved.Canonical,
			Selection: "sha256:" + resolved.SelectionSHA256,
			License:   resolved.License, Files: len(resolved.Files), Bytes: resolved.Bytes,
			Exact: exact, Access: access,
			Resolver: resolver, Resolution: resolved}, nil
	}
	if !catalogModelSpelling(raw) {
		return publishSource{}, parseProblem
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	refspec := raw
	if strings.Contains(raw, "#") {
		model, release, selectedLane, manifest, problem := hub.ParseModelRef(raw)
		if problem != nil {
			return publishSource{}, problem
		}
		if selectedLane != "" {
			if lane != "" && lane != selectedLane {
				return publishSource{}, exit.Usagef("model reference and selected lane disagree")
			}
			lane = selectedLane
		}
		if manifest == "" {
			resolvedModel, problem := resolveRemoteModel(ctx, "", launch.Slot{}, raw, lane, nil)
			if problem != nil {
				return publishSource{}, problem
			}
			refspec = model + "@" + resolvedModel.Release
			lane = resolvedModel.Lane
		} else {
			refspec = model + "@" + manifest
			if release != "" {
				refspec = model + "@" + release + "@" + manifest
			}
		}
	}
	resolved, problem := client(ctx).ResolveModel(hctx, refspec, lane)
	if problem != nil {
		return publishSource{}, problem
	}
	if (resolved.Release == "") != (resolved.Lane == "") || resolved.ManifestID == "" {
		return publishSource{}, exit.Named(exit.Structural, "model_source_resolution_incomplete",
			"Tensorhub did not resolve one exact checkpoint")
	}
	selectedRef := resolved.Model + "@" + resolved.ManifestID
	if resolved.Release != "" {
		selectedRef = resolved.Model + "@" + resolved.Release
	}
	return publishSource{Canonical: selectedRef,
		Selection: resolved.ManifestID, Lane: resolved.Lane,
		Files: resolved.Objects, Bytes: resolved.Bytes}, nil
}

// narrowPublishSource applies a reviewed TensorFS source profile to a provider
// resolution already used for header preflight. Keeping this in-memory avoids a
// second provider resolution between the preflight and durable submission while
// making the accepted source inventory match RefreshRemoteSource exactly.
func narrowPublishSource(ctx *Context, source publishSource, profiles []string) (publishSource, *exit.Error) {
	if source.Resolver == nil || len(profiles) == 0 {
		return source, nil
	}
	tool, _, problem := localTensorFS(ctx)
	if problem != nil {
		return publishSource{}, problem
	}
	members, problem := tool.SourceProfileMembers(ctx.Cfg.TensorFSRegistry, profiles)
	if problem != nil {
		return publishSource{}, problem
	}
	selected, problem := source.Resolution.Select(members)
	if problem != nil {
		return publishSource{}, problem
	}
	accessByMember := make(map[string]sourceCapability, len(source.Access))
	for _, access := range source.Access {
		accessByMember[access.Member] = access
	}
	exact := make([]modeltransfer.SourceFile, 0, len(selected.Files))
	access := make([]sourceCapability, 0, len(selected.Files))
	for _, file := range selected.Files {
		exact = append(exact, modeltransfer.SourceFile{Member: file.Member,
			SHA256: file.SHA256, Length: file.Length})
		capability, ok := accessByMember[file.Member]
		if !ok {
			return publishSource{}, exit.Named(exit.Conflict, "model_source.capability_missing",
				"reviewed source profile selected %s without a provider capability", file.Member)
		}
		access = append(access, capability)
	}
	return publishSource{Canonical: selected.Canonical, Selection: "sha256:" + selected.SelectionSHA256,
		License: selected.License, Lane: source.Lane, Files: len(selected.Files), Bytes: selected.Bytes,
		Exact: exact, Access: access, Resolver: source.Resolver, Resolution: selected}, nil
}

func catalogModelSpelling(value string) bool {
	if strings.Contains(value, "://") || filepath.IsAbs(value) || strings.HasPrefix(value, ".") {
		return false
	}
	name, _, _ := strings.Cut(value, "@")
	parts := strings.Split(name, "/")
	return len(parts) == 2 && parts[0] != "" && parts[1] != ""
}

// parseSourceProfileFlags reads exact caller-owned slot=profile narrowing.
func parseSourceProfileFlags(ctx *Context) (map[string]string, *exit.Error) {
	values := ctx.Inv.Values["--source-profile"]
	supplied := make(map[string]string, len(values))
	for _, value := range values {
		slot, profile, ok := strings.Cut(value, "=")
		slot, profile = strings.TrimSpace(slot), strings.TrimSpace(profile)
		if !ok || slot == "" || profile == "" {
			return nil, exit.Usagef("--source-profile %q is not slot=profile", value)
		}
		if _, duplicate := supplied[slot]; duplicate {
			return nil, exit.Usagef("--source-profile names slot %s twice", slot)
		}
		supplied[slot] = profile
	}
	return supplied, nil
}
