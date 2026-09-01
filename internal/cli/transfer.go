package cli

import (
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
)

// The model transfer verbs (cl-012). `model publish` is th-002's declare-first protocol
// driven from this side; `model download` is its inverse into the local canonical store. Neither owns a
// byte or a protocol: the byte plane is TensorFS's (internal/tfs) and the protocol is
// the hub's (internal/hub). What these own is the argument surface, the progress
// accounting, and the exit code.

// tooling opens the byte plane and the hub together, with the credential this
// invocation carries. A verb that could not get either says so before it moves.
func tooling(ctx *Context) (*tfs.Tool, *hub.Client, home.Layout, *exit.Error) {
	layout, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return nil, nil, layout, e
	}
	tool, e := tfs.Open(ctx.Cfg, layout)
	if e != nil {
		return nil, nil, layout, e
	}
	c := client(ctx)
	if ctx.Inv.Bool("--token-stdin") {
		v, e := readToken(ctx)
		if e != nil {
			return nil, nil, layout, e
		}
		c = c.WithToken(v, "stdin")
	}
	return tool, c, layout, nil
}

// readToken takes the invocation's credential off stdin. It is a VALUE read, never a
// prompt: nothing is printed, nothing is waited on interactively, and an empty stdin
// is a typed refusal rather than a hang. argv is world-readable, which is why the
// flag has no argument at all (cl-011's secret fence).
func readToken(ctx *Context) (secret.Value, *exit.Error) {
	raw, err := io.ReadAll(io.LimitReader(os.Stdin, 8<<10)) //cozy:stdin-value the credential arrives as a value on stdin
	if err != nil {
		return secret.Value{}, exit.Internalf("reading the credential from stdin failed: %s", err)
	}
	v := secret.New(string(raw))
	if !v.Present() {
		return secret.Value{}, exit.Named(exit.Credential, "token.empty_stdin",
			"--token-stdin was given and stdin carried no credential").
			WithRemedy("pipe the token in: printf %%s \"$TOKEN\" | cozy … --token-stdin").
			WithNext("cozy package search")
	}
	return v, nil
}

// scratch is this transfer's own staging directory, named by its subject so two
// transfers never stage over each other.
func scratch(layout home.Layout, subject string) string {
	return filepath.Join(layout.Transfer, strings.TrimPrefix(subject, "sha256:")[:16])
}

func progress(ctx *Context) func(string) {
	return func(line string) { _ = output.Progress(ctx.Err, line) }
}

func handleDirectModelPublish(ctx *Context) *exit.Error {
	ref, e := hub.ParseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	if ref.Org == "local" {
		return exit.Usagef("local/ is reserved for private aliases and cannot be a Tensorhub destination").
			WithRemedy("publish under your Tensorhub account, for example alice/%s", ref.Name)
	}
	release, lane := strings.TrimSpace(ctx.Inv.Value("--release")), strings.TrimSpace(ctx.Inv.Value("--lane"))
	if release == "" || lane == "" {
		return exit.Usagef("model publish requires --release and --lane")
	}
	publicationClient, e := ownedPublication(ctx, ref)
	if e != nil {
		return e
	}
	subject := ctx.Inv.Args[1]
	var evidenceRef hub.Ref
	var manifestID string
	if strings.HasPrefix(subject, "local/") {
		name := strings.TrimPrefix(subject, "local/")
		if problem := modelsource.LocalName(name); problem != nil {
			return problem
		}
		tool, _, problem := localTensorFS(ctx)
		if problem != nil {
			return problem
		}
		row, problem := tool.ResolveLocal(name)
		if problem != nil {
			return problem
		}
		manifestID = row.ManifestDigest
		evidenceRef = hub.Ref{Org: "local", Name: name}
	} else {
		manifestID, e = tfs.ManifestID(subject)
		if e != nil {
			cwd, err := os.Getwd()
			if err != nil {
				return exit.Internalf("cannot resolve the current directory: %s", err)
			}
			if source, sourceProblem := modelsource.Parse(subject, cwd); sourceProblem == nil {
				return exit.Named(exit.Structural, "model_source_planner_unavailable",
					"%s is a valid model source, but this TensorFS build cannot derive its closed ingest plan", source.Canonical).
					WithRemedy("the byte plane must supply reviewed component, encoding, and construction-order facts; Creator will not infer them from filenames")
			}
			return e
		}
	}
	reason := "cozy model publish " + ref.String() + " " + manifestID + " --release " + release + " --lane " + lane
	// The versioned operation binds the complete named-lane intent without
	// colliding with publications opened under earlier request semantics.
	session := transfer.PublicationOperationID(ref, release, lane, manifestID)
	tool, _, layout, e := tooling(ctx)
	if e != nil {
		return e
	}
	p := &transfer.Publish{
		Tool: tool, Hub: publicationClient, Ref: ref, EvidenceRef: evidenceRef,
		ManifestID: manifestID, Release: release, Lane: lane, Session: session,
		Reason: reason, DryRun: ctx.Inv.Bool("--dry-run"),
		Progress: progress(ctx), Scratch: scratch(layout, manifestID),
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	res, e := p.Run(hctx)
	if e != nil {
		return e
	}

	status, changed := "published", !res.Dup
	if p.DryRun {
		status, changed = "planned", false
	}
	fields := []output.Field{
		{K: "model", V: ref.String()},
		{K: "manifest_id", V: manifestID},
		{K: "release", V: release},
		{K: "lane", V: lane},
		{K: "status", V: status},
		{K: "changed", V: changed},
		{K: "publish_id", V: res.PublishID},
		{K: "session", V: res.Session},
		{K: "objects", V: res.Totals.DeclaredObjects},
		{K: "bytes", V: output.Bytes(res.Totals.DeclaredBytes)},
		{K: "moved", V: output.Bytes(res.Moved)},
		{K: "deduped", V: output.Bytes(res.Deduped)},
	}
	if p.DryRun {
		fields = append(fields,
			output.Field{K: "missing", V: res.Totals.MissingObjects},
			output.Field{K: "held", V: res.Totals.HeldObjects})
		rec := compactRecord(fields, "model", "release", "lane", "manifest_id", "status", "missing", "changed")
		rec.Next = []string{reason}
		return emit(ctx, rec)
	}
	fields = append(fields,
		output.Field{K: "uploaded", V: res.Uploaded},
		output.Field{K: "verified", V: res.Verified},
		output.Field{K: "manifest_length", V: res.Manifest.Length},
		output.Field{K: "topology", V: res.TopologyDigest},
		output.Field{K: "release_operation", V: res.CutOperation},
		output.Field{K: "repository_sha256", V: res.RepositorySHA},
		output.Field{K: "duplicate", V: res.Dup},
	)
	return emit(ctx, compactRecord(fields,
		"model", "release", "lane", "manifest_id", "status", "moved", "deduped", "changed"))
}

func handleModelDownload(ctx *Context) *exit.Error {
	spec := ctx.Inv.Args[0]
	tool, c, layout, e := tooling(ctx)
	if e != nil {
		return e
	}
	f := &transfer.Fetch{
		Tool: tool, Hub: c, Spec: spec, Lane: strings.TrimSpace(ctx.Inv.Value("--lane")),
		DryRun: ctx.Inv.Bool("--dry-run"), Progress: progress(ctx),
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	row, e := f.Resolve(hctx)
	if e != nil {
		return e
	}
	ref := f.Ref
	f.Scratch = scratch(layout, row.ManifestID)
	res, e := f.Acquire(hctx, row)
	if e != nil {
		return e
	}

	status, changed := "downloaded", res.Moved > 0
	if f.DryRun {
		status, changed = "planned", false
	}
	fields := []output.Field{
		{K: "model", V: ref.String()},
		{K: "manifest_id", V: res.ManifestID},
		{K: "release", V: res.Release},
		{K: "lane", V: res.Lane},
		{K: "status", V: status},
		{K: "changed", V: changed},
		{K: "objects", V: res.Objects},
		{K: "bytes", V: output.Bytes(res.Bytes)},
		{K: "moved", V: output.Bytes(res.Moved)},
		{K: "deduped", V: output.Bytes(res.Held)},
	}
	if f.DryRun {
		rec := compactRecord(fields, "model", "release", "lane", "manifest_id", "status", "bytes", "changed")
		rec.Next = []string{"cozy model download " + ref.String() + "@" + res.Release + " --lane " + res.Lane}
		return emit(ctx, rec)
	}
	fields = append(fields,
		output.Field{K: "header", V: res.HeaderID},
		output.Field{K: "admitted", V: res.Admitted},
		output.Field{K: "skipped", V: res.Skipped},
		output.Field{K: "store", V: tool.Root},
	)
	return emit(ctx, compactRecord(fields,
		"model", "release", "lane", "manifest_id", "status", "moved", "deduped", "changed"))
}

func localTensorFS(ctx *Context) (*tfs.Tool, home.Layout, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, layout, problem
	}
	tool, problem := tfs.Open(ctx.Cfg, layout)
	return tool, layout, problem
}

func handleModelList(ctx *Context) *exit.Error {
	tool, layout, problem := localTensorFS(ctx)
	if problem != nil {
		return problem
	}
	if err := os.MkdirAll(layout.Transfer, 0o700); err != nil {
		return exit.Internalf("cannot create transfer scratch: %s", err)
	}
	scratch, err := os.MkdirTemp(layout.Transfer, "repo-list-")
	if err != nil {
		return exit.Internalf("cannot create repository-list scratch: %s", err)
	}
	defer os.RemoveAll(scratch)
	releases, problem := tool.Releases(filepath.Join(scratch, "rows.jsonl"))
	if problem != nil {
		return problem
	}
	list := output.List{
		Name: "models", Fields: []string{"model", "release", "lane", "manifest_id"},
		AllFields: []string{"model", "kind", "release", "lane", "manifest_id"},
	}
	for _, release := range releases {
		row := map[string]string{"model": release.Org + "/" + release.Name,
			"kind": "catalog", "release": release.Version, "lane": release.Lane,
			"manifest_id": "sha256:" + release.ManifestSHA256}
		if release.Org == "local" {
			row["kind"], row["release"], row["lane"] = "local", "", ""
		}
		list.Rows = append(list.Rows, row)
	}
	return emit(ctx, list)
}

func handleModelRemove(ctx *Context) *exit.Error {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return problem
	}
	live, problem := store.LiveWorkers()
	if problem != nil {
		store.Close()
		return problem
	}
	for _, worker := range live {
		if worker.WorkerID != "remote" {
			store.Close()
			return exit.New(exit.Conflict, "local worker %s may still hold model residency", worker.InstanceID).
				WithRemedy("run `cozy unload`, then remove the model repository").
				WithNext("cozy unload")
		}
	}
	active, problem := store.ActiveRequests()
	store.Close()
	if problem != nil {
		return problem
	}
	for _, request := range active {
		if request.Worker == "" {
			return exit.New(exit.Conflict, "active invocation %s may still need local model bytes", request.ID).
				WithRemedy("cancel active local work before removing a model repository").
				WithNext("cozy run cancel " + request.ID)
		}
	}
	tool, _, problem := localTensorFS(ctx)
	if problem != nil {
		return problem
	}
	if err := os.MkdirAll(layout.Transfer, 0o700); err != nil {
		return exit.Internalf("cannot create transfer scratch: %s", err)
	}
	scratchDir, err := os.MkdirTemp(layout.Transfer, "repo-remove-")
	if err != nil {
		return exit.Internalf("cannot create repository-remove scratch: %s", err)
	}
	defer os.RemoveAll(scratchDir)
	releases, problem := tool.Releases(filepath.Join(scratchDir, "rows.jsonl"))
	if problem != nil {
		return problem
	}
	held := make(map[string]bool, len(releases))
	for _, release := range releases {
		held[release.Org+"/"+release.Name] = true
	}
	removed := output.List{
		Name: "models", Fields: []string{"model"}, AllFields: []string{"model"},
	}
	for _, name := range ctx.Inv.Args {
		ref, parseProblem := hub.ParseRef(name)
		if parseProblem != nil {
			return parseProblem
		}
		if !held[ref.String()] {
			continue
		}
		if ref.Org == "local" {
			if problem := modelsource.LocalName(ref.Name); problem != nil {
				return problem
			}
			alias, problem := tool.ResolveLocal(ref.Name)
			if problem != nil {
				return problem
			}
			if problem := tool.RemoveLocal(ref.Name, alias.RepositoryDigest); problem != nil {
				return problem
			}
		} else {
			repoScratch := filepath.Join(scratchDir, strings.ReplaceAll(ref.String(), "/", "-"))
			if err := os.MkdirAll(repoScratch, 0o700); err != nil {
				return exit.Internalf("cannot create repository-remove scratch: %s", err)
			}
			if problem := tool.DeleteRepository(ref.Org, ref.Name, repoScratch); problem != nil {
				return problem
			}
		}
		removed.Rows = append(removed.Rows, map[string]string{"model": ref.String()})
		delete(held, ref.String())
	}
	removed.Aggregates = []output.Field{{K: "changed", V: len(removed.Rows) > 0}}
	removed.Notes = []string{"local repositories were deleted; TensorFS garbage collection decides later byte reclamation"}
	return emit(ctx, removed)
}
