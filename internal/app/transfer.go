package app

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/render"
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
			WithNext("cozy hub status")
	}
	return v, nil
}

// scratch is this transfer's own staging directory, named by its subject so two
// transfers never stage over each other.
func scratch(layout home.Layout, subject string) string {
	return filepath.Join(layout.Transfer, strings.TrimPrefix(subject, "sha256:")[:16])
}

func progress(ctx *Context) func(string) {
	if ctx.Mode().JSON {
		// Under --json stdout carries exactly ONE document. Progress goes to stderr
		// or nowhere; a machine consumer reads the result, not the narration.
		return func(line string) { fmt.Fprintln(ctx.Err, line) }
	}
	return func(line string) { fmt.Fprintln(ctx.Out, line) }
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
	res, e := p.Run(hctx)
	if e != nil {
		return e
	}

	fields := []render.Field{
		{K: "model", V: ref.String()},
		{K: "publish_id", V: res.PublishID},
		{K: "session", V: res.Session},
		{K: "objects", V: res.Totals.DeclaredObjects},
		{K: "bytes", V: render.Bytes(res.Totals.DeclaredBytes)},
		{K: "moved", V: render.Bytes(res.Moved)},
		{K: "deduped", V: render.Bytes(res.Deduped)},
	}
	if p.DryRun {
		return emit(ctx, render.Record{
			Kind: "model publish plan", Fields: append(fields,
				render.Field{K: "missing", V: res.Totals.MissingObjects},
				render.Field{K: "held", V: res.Totals.HeldObjects},
				render.Field{K: "hub", V: c.Base()}),
			Notes: []string{"--dry-run declared and stopped: the plan is the HUB's answer, not a local guess"},
			Next:  []string{"cozy model publish " + ref.String() + " " + snapshot + " --reason <why>"},
		})
	}
	fields = append(fields,
		render.Field{K: "uploaded", V: res.Uploaded},
		render.Field{K: "verified", V: res.Verified},
		render.Field{K: "checksum_source", V: res.Sources},
		render.Field{K: "snapshot", V: res.Root.SnapshotID},
		render.Field{K: "header", V: res.Root.HeaderID},
		render.Field{K: "topology", V: res.Root.TopologyDigest},
		render.Field{K: "catalog_root", V: res.Root.CatalogRootID},
		render.Field{K: "grade", V: res.Grade},
		render.Field{K: "satisfaction", V: res.Verdict},
		render.Field{K: "verifier", V: res.Root.VerifierBuild},
		render.Field{K: "reingested", V: res.Reingest},
		render.Field{K: "duplicate", V: res.Dup},
		render.Field{K: "hub", V: c.Base()},
	)
	notes := []string{
		fmt.Sprintf("the hub re-verified %d already-held objects hermetically: a client receipt substitutes for nothing (law 18)", res.Reingest),
	}
	if res.Multipart > 0 {
		notes = append(notes, fmt.Sprintf("%d objects went as ranged uploads; R2 signs no digest on those, so the hub's own streaming hash discharged it", res.Multipart))
	}
	return emit(ctx, render.Record{
		Kind: "model publish", Fields: fields, Notes: notes,
		Next: []string{"cozy model download " + ref.String() + "@" + res.Root.SnapshotID},
	})
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

	fields := []render.Field{
		{K: "model", V: ref.String()},
		{K: "snapshot", V: res.Snapshot},
		{K: "objects", V: res.Objects},
		{K: "bytes", V: render.Bytes(res.Bytes)},
		{K: "moved", V: render.Bytes(res.Moved)},
		{K: "deduped", V: render.Bytes(res.Held)},
	}
	if f.DryRun {
		return emit(ctx, render.Record{
			Kind: "model download plan", Fields: append(fields, render.Field{K: "hub", V: c.Base()}),
			Notes: []string{"--dry-run moved nothing; the tensor set is computed from the checkpoint's own documents once they land"},
			Next:  []string{"cozy model download " + ref.String() + "@" + res.Snapshot},
		})
	}
	fields = append(fields,
		render.Field{K: "header", V: res.HeaderID},
		render.Field{K: "admitted", V: res.Admitted},
		render.Field{K: "skipped", V: res.Skipped},
		render.Field{K: "tensors", V: res.Tensors},
		render.Field{K: "parts", V: res.Parts},
		render.Field{K: "grade", V: res.Grade},
		render.Field{K: "root", V: tool.Root},
		render.Field{K: "hub", V: c.Base()},
	)
	return emit(ctx, render.Record{
		Kind: "model download", Fields: fields,
		Notes: []string{
			fmt.Sprintf("admitted counts the snapshot manifest too: the closure names what a transfer MOVES (%d objects) and the manifest separately", res.Objects),
			"every declared byte was verified before this snapshot became a named local root",
			"the local root is noted hub_published: durability without a pin — `tfs gc` may still reclaim it",
		},
		Next: []string{"cozy model download " + ref.String() + "@" + res.Snapshot},
	})
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
