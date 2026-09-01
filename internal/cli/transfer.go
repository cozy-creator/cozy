package cli

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// The model transfer verbs (cl-012). `model upload` is th-002's declare-first protocol
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
	return tool, client(ctx), layout, nil
}

// scratch is this transfer's own staging directory, named by its subject so two
// transfers never stage over each other.
func scratch(layout home.Layout, subject string) string {
	return filepath.Join(layout.Transfer, strings.TrimPrefix(subject, "sha256:")[:16])
}

func progress(ctx *Context) func(string) {
	return func(line string) { _ = output.Progress(ctx.Err, line) }
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
		Name: "models", Fields: []string{"model", "release", "lane"},
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
