package cli

import (
	"context"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/scratch"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// The CONVERSION plan, decided from HEADERS, before a rental is requested and before a
// byte of payload moves (tfs-076).
//
//	"why does conversion have to download all 200GBs of files and then refuse?
//	 shouldn't it refuse right away?"
//
// It should. Three MiniMax-H3 runs on 2026-09-03/04 each moved 210.3 GB and then refused
// on facts totalling under 300 KB — twice the member list, and once the first EIGHT BYTES
// of one file, read as a header length. None of those refusals needed a tensor byte, and
// the ordering that put them behind the transfer was an accident of call order rather than
// a decision.
//
// **Nothing here is new machinery.** `Resolver.Stage(..., headersOnly)` and
// `planLocalSourceProfiles` are the LOCAL transfer's own header-first path, unchanged: two
// exact ranged GETs per carrier, a sparse file sized to the member's full length so the
// declared geometry is checked against something true, and `tfs ingest source-plan` over
// the result. That path has existed and worked for local transfers all along. All that was
// missing was a caller in front of the money.
//
// Two outcomes, and they are NOT the same outcome:
//
//   - the plan REFUSED. This selection cannot convert, and transferring it does not help.
//     The refusal is returned and nothing is rented.
//   - the headers could not be read — an origin that will not serve a range, a network
//     fault, a source with no provider resolution at all. UNDECIDED. "I could not look" is
//     never "this is bad". A rented invocation waits for a retry before acquisition;
//     an undecided header read cannot authorize paying for a conversion gamble.
type conversionPreflight struct {
	// Plans by producer slot, when the headers decided, and the source members they use.
	Plans   map[string]tfs.SourcePlan
	Members []string
	// Exact inspected prefixes survive the temporary sparse files and enter the durable request.
	Headers map[string][]byte
	// Why the headers could not be read, when they could not. Non-empty means UNDECIDED,
	// and Plans is then empty.
	Undecided string
}

func (p conversionPreflight) decided() bool { return len(p.Plans) > 0 }

// preflightConversionPlan reads each reviewed carrier's header by ranged GET and plans the
// conversion from it. No rental, no pod, no destination store, and no tensor byte: on
// MiniMax-H3 that is a few hundred KB of header standing in for 210.3 GB of payload.
//
// The resolution it stages from is the one `resolvePublishSource` already narrowed to the
// reviewed carriers — a repository is not a model, and narrowing first is what makes this
// 48 header reads instead of 112.
func preflightConversionPlan(runCtx context.Context, ctx *Context, source publishSource,
	slots map[string]string,
) (conversionPreflight, *exit.Error) {
	if len(slots) == 0 {
		// Standalone imports use the same automatic header classification as
		// their real transfer.
		slots = map[string]string{"model": ""}
	}
	if source.Resolver == nil {
		// A local file, a `local/` alias, or a Tensorhub checkpoint. None of those has a
		// provider header to range-read, and none of them is the 210 GB case.
		return conversionPreflight{Undecided: "the source has no provider resolution"}, nil
	}
	tool, layout, problem := localTensorFS(ctx)
	if problem != nil {
		return conversionPreflight{}, problem
	}
	work, problem := scratch.Temp(layout.Tmp, "source-preflight-")
	if problem != nil {
		return conversionPreflight{}, problem
	}
	defer work.Release()

	// Reading the headers is the part that can be legitimately impossible. A refusal from
	// here is a statement about the ORIGIN, never about the model, so it degrades to
	// undecided; only a missing credential is the caller's to fix. The caller requires a
	// decided plan before renting.
	staged, problem := source.Resolver.Stage(runCtx, source.Resolution,
		filepath.Join(work.Path, "headers"), true, progress(ctx))
	if problem != nil && problem.Code == exit.Credential {
		return conversionPreflight{}, problem
	}
	if problem != nil {
		return conversionPreflight{Undecided: problem.Message}, nil
	}

	// From here the answer is about the MODEL, and a refusal is the answer.
	plans, members, problem := planLocalSourceProfiles(tool, ctx, slots, staged, true,
		filepath.Join(work.Path, "preflight-plans"))
	if problem != nil {
		return conversionPreflight{}, problem
	}
	headers := make(map[string][]byte, len(staged))
	for _, file := range staged {
		headers[file.Member] = file.Header
	}
	return conversionPreflight{Plans: plans, Members: members, Headers: headers}, nil
}
