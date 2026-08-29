package cli

import (
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/output"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/secret"
	"github.com/cozy-creator/cozy-creator/internal/tfs"
	"github.com/cozy-creator/cozy-creator/internal/transfer"
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
			WithNext("cozy endpoint search")
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

func handleModelPublish(ctx *Context) *exit.Error {
	ref, e := hub.ParseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	subject := ctx.Inv.Args[1]
	if strings.ContainsRune(subject, os.PathSeparator) || strings.HasPrefix(subject, ".") {
		// A path is an INGEST subject, not a publish subject (decisions #58's two-step
		// model): the border runs where the bytes are, and only a canonical snapshot
		// is publishable. Refusing by name beats growing a second border here.
		return exit.Usagef("%q is a path, and a path is not publishable", subject).
			WithRemedy("ingest it first — `tfs ingest run <store> <component=alias> --source …` then `tfs ingest install` — and publish the snapshot id it prints").
			WithNext("cozy help model publish")
	}
	snapshot, e := tfs.Snapshot(subject)
	if e != nil {
		return e
	}
	reason := strings.TrimSpace(ctx.Inv.Value("--reason"))
	if reason == "" {
		return exit.Usagef("`cozy model publish` needs --reason <why>").
			WithRemedy("the hub records why every first-party write happened, before it acts").
			WithNext("cozy help model publish")
	}
	// The operation is bound to the complete immutable snapshot identity. A user-
	// supplied operation id could be replayed with different bytes and is therefore
	// not part of the product surface.
	session := "snapshot-" + strings.TrimPrefix(snapshot, "sha256:")
	failAfter, e := devKill(ctx)
	if e != nil {
		return e
	}

	tool, c, layout, e := tooling(ctx)
	if e != nil {
		return e
	}
	p := &transfer.Publish{
		Tool: tool, Hub: c, Ref: ref, Snapshot: snapshot, Session: session,
		Reason: reason, DryRun: ctx.Inv.Bool("--dry-run"),
		Progress: progress(ctx), Scratch: scratch(layout, snapshot), FailAfter: failAfter,
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	if _, e := c.Model(hctx, ref); e != nil {
		if e.Code != exit.NotFound {
			return e
		}
		if _, e := c.CreateModel(hctx, ref.Org, ref.Name, reason); e != nil && e.Code != exit.Conflict {
			return e
		}
	}
	res, e := p.Run(hctx)
	if e != nil {
		return e
	}

	fields := []output.Field{
		{K: "model", V: ref.String()},
		{K: "snapshot", V: snapshot},
		{K: "status", V: "published"},
		{K: "changed", V: res.Moved > 0},
		{K: "publish_id", V: res.PublishID},
		{K: "session", V: res.Session},
		{K: "objects", V: res.Totals.DeclaredObjects},
		{K: "bytes", V: output.Bytes(res.Totals.DeclaredBytes)},
		{K: "moved", V: output.Bytes(res.Moved)},
		{K: "deduped", V: output.Bytes(res.Deduped)},
	}
	if p.DryRun {
		fields[2].V = "planned"
		fields[3].V = false
		fields = append(fields,
			output.Field{K: "missing", V: res.Totals.MissingObjects},
			output.Field{K: "held", V: res.Totals.HeldObjects})
		rec := compactRecord(fields, "model", "snapshot", "status", "missing", "changed")
		rec.Next = []string{"cozy model publish " + ref.String() + " " + snapshot + " --reason <why>"}
		return emit(ctx, rec)
	}
	fields = append(fields,
		output.Field{K: "uploaded", V: res.Uploaded},
		output.Field{K: "verified", V: res.Verified},
		output.Field{K: "checksum_source", V: res.Sources},
		output.Field{K: "header", V: res.Root.HeaderID},
		output.Field{K: "topology", V: res.Root.TopologyDigest},
		output.Field{K: "catalog_root", V: res.Root.CatalogRootID},
		output.Field{K: "grade", V: res.Grade},
		output.Field{K: "satisfaction", V: res.Verdict},
		output.Field{K: "verifier", V: res.Root.VerifierBuild},
		output.Field{K: "reingested", V: res.Reingest},
		output.Field{K: "duplicate", V: res.Dup},
	)
	return emit(ctx, compactRecord(fields,
		"model", "snapshot", "status", "moved", "deduped", "changed"))
}

func handleModelDownload(ctx *Context) *exit.Error {
	spec := ctx.Inv.Args[0]
	failAfter, e := devKill(ctx)
	if e != nil {
		return e
	}

	tool, c, layout, e := tooling(ctx)
	if e != nil {
		return e
	}
	f := &transfer.Fetch{
		Tool: tool, Hub: c, Spec: spec, Lane: strings.TrimSpace(ctx.Inv.Value("--lane")),
		DryRun: ctx.Inv.Bool("--dry-run"), Progress: progress(ctx), FailAfter: failAfter,
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	row, e := f.Resolve(hctx)
	if e != nil {
		return e
	}
	ref := f.Ref
	f.Scratch = scratch(layout, row.SnapshotID)
	res, e := f.Run(hctx, row)
	if e != nil {
		return e
	}

	fields := []output.Field{
		{K: "model", V: ref.String()},
		{K: "snapshot", V: res.Snapshot},
		{K: "status", V: "downloaded"},
		{K: "changed", V: res.Moved > 0},
		{K: "objects", V: res.Objects},
		{K: "bytes", V: output.Bytes(res.Bytes)},
		{K: "moved", V: output.Bytes(res.Moved)},
		{K: "deduped", V: output.Bytes(res.Held)},
	}
	if f.DryRun {
		fields[2].V = "planned"
		fields[3].V = false
		rec := compactRecord(fields, "model", "snapshot", "status", "bytes", "changed")
		rec.Next = []string{"cozy model download " + ref.String() + "@" + res.Snapshot}
		return emit(ctx, rec)
	}
	fields = append(fields,
		output.Field{K: "header", V: res.HeaderID},
		output.Field{K: "admitted", V: res.Admitted},
		output.Field{K: "skipped", V: res.Skipped},
		output.Field{K: "tensors", V: res.Tensors},
		output.Field{K: "parts", V: res.Parts},
		output.Field{K: "grade", V: res.Grade},
		output.Field{K: "root", V: tool.Root},
	)
	return emit(ctx, compactRecord(fields,
		"model", "snapshot", "status", "moved", "deduped", "changed"))
}

func localTensorFS(ctx *Context) (*tfs.Tool, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, problem
	}
	return tfs.Open(ctx.Cfg, layout)
}

func handleModelList(ctx *Context) *exit.Error {
	tool, problem := localTensorFS(ctx)
	if problem != nil {
		return problem
	}
	roots, problem := tool.Roots()
	if problem != nil {
		return problem
	}
	list := output.List{
		Name: "models", Fields: []string{"model", "snapshot"},
		AllFields: []string{"model", "snapshot", "kind"},
	}
	for _, root := range roots {
		list.Rows = append(list.Rows, map[string]string{
			"model": root.Name, "snapshot": root.Snapshot, "kind": root.Kind,
		})
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
				WithRemedy("run `cozy unload`, then remove the model root").
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
				WithRemedy("cancel active local work before removing a model root").
				WithNext("cozy invoke cancel " + request.ID)
		}
	}
	tool, problem := localTensorFS(ctx)
	if problem != nil {
		return problem
	}
	roots, problem := tool.Roots()
	if problem != nil {
		return problem
	}
	held := make(map[string]tfs.Root, len(roots))
	for _, root := range roots {
		held[root.Name] = root
	}
	removed := output.List{
		Name: "models", Fields: []string{"model", "snapshot"},
		AllFields: []string{"model", "snapshot"},
	}
	for _, name := range ctx.Inv.Args {
		root, ok := held[name]
		if !ok {
			continue
		}
		if problem := tool.ReleaseRoot(name); problem != nil {
			return problem
		}
		removed.Rows = append(removed.Rows, map[string]string{
			"model": root.Name, "snapshot": root.Snapshot,
		})
		delete(held, name)
	}
	removed.Aggregates = []output.Field{{K: "changed", V: len(removed.Rows) > 0}}
	removed.Notes = []string{"local names were released; TensorFS garbage collection decides later byte reclamation"}
	return emit(ctx, removed)
}

// devKill parses the development kill point both transfer verbs share. It exists to
// prove convergence: an interrupted transfer must resume, and the only honest way to
// show that is to actually interrupt one.
func devKill(ctx *Context) (int, *exit.Error) {
	v := strings.TrimSpace(ctx.Inv.Value("--crash-after"))
	if v == "" {
		return 0, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return 0, exit.Usagef("--crash-after takes a positive object count, not %q", v).
			WithRemedy("--crash-after 3 stops after the third object has moved")
	}
	return n, nil
}
