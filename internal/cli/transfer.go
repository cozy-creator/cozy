package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
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
	tool, e := tfs.Open(ctx.Cfg)
	if e != nil {
		return nil, nil, layout, e
	}
	return tool, client(ctx), layout, nil
}

func progress(ctx *Context) func(string) {
	return func(line string) { _ = output.Progress(ctx.Err, line) }
}

func localTensorFS(ctx *Context) (*tfs.Tool, home.Layout, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, layout, problem
	}
	tool, problem := tfs.Open(ctx.Cfg)
	return tool, layout, problem
}

func handleModelList(ctx *Context) *exit.Error {
	tool, layout, problem := localTensorFS(ctx)
	if problem != nil {
		return problem
	}
	work, problem := scratch.Temp(layout.Tmp, "repo-list-")
	if problem != nil {
		return problem
	}
	defer work.Release()
	releases, problem := tool.Releases(filepath.Join(work.Path, "rows.jsonl"))
	if problem != nil {
		return problem
	}
	// The list stands without its byte columns: a store the byte plane refuses to
	// measure is still a store with names in it, and the refusal is said, not hidden.
	usage, usageProblem := tool.Usage(filepath.Join(work.Path, "usage.jsonl"))
	bytesOf := make(map[string]tfs.RepositoryUsage, len(usage.Repos))
	for _, repo := range usage.Repos {
		bytesOf[repo.Org+"/"+repo.Name] = repo
	}
	list := output.List{
		Name: "models", Fields: []string{"model", "release", "lane", "size", "shared"},
		AllFields: []string{"model", "kind", "release", "lane", "manifest_id", "size", "shared"},
		Bytes:     []string{"size", "shared", "unique"},
		// unique is the fact tfs computes; a program gets it, a table does not widen for it.
		Machine: []string{"unique"},
	}
	for _, release := range releases {
		row := map[string]string{"model": release.Org + "/" + release.Name,
			"kind": "catalog", "release": release.Version, "lane": release.Lane,
			"manifest_id": "sha256:" + release.ManifestSHA256}
		if release.Kind == "local" {
			row["kind"] = "local"
		}
		if repo, ok := bytesOf[row["model"]]; ok {
			row["size"] = output.Int(repo.Total)
			row["shared"] = output.Int(repo.Total - repo.Unique)
			row["unique"] = output.Int(repo.Unique)
		}
		list.Rows = append(list.Rows, row)
	}
	if usageProblem != nil {
		list.Notes = []string{"disk usage is unavailable: " + usageProblem.Message}
		return emit(ctx, list)
	}
	list.Aggregates = []output.Field{{K: "store", V: storeUsage{Size: usage.Total,
		Unique: usage.UniqueSum, Unreferenced: usage.Unreferenced, Models: len(usage.Repos)}}}
	return emit(ctx, list)
}

// storeUsage is the model list's footer: the store's byte plane for a program, one
// sentence for a person — and only a sentence once there is something to reclaim.
type storeUsage struct {
	Size         int64 `json:"size"`   // the union every model reaches
	Unique       int64 `json:"unique"` // the sum of every model's unique part
	Unreferenced int64 `json:"unreferenced"`
	Models       int   `json:"models"`
}

func (s storeUsage) Human() string {
	if s.Unreferenced == 0 {
		return ""
	}
	noun := "models"
	if s.Models == 1 {
		noun = "model"
	}
	return fmt.Sprintf("%s in %d %s · %s unreferenced",
		output.Bytes(s.Size), s.Models, noun, output.Bytes(s.Unreferenced))
}

func handleModelRemove(ctx *Context) *exit.Error {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return problem
	}
	if problem := modelRemovalRefusal(layout, ctx.Inv.Args); problem != nil {
		return problem
	}
	fence, problem := readReclamationFence(layout)
	if problem != nil {
		return problem
	}
	tool, _, problem := localTensorFS(ctx)
	if problem != nil {
		return problem
	}
	return removeModels(ctx, tool, layout, fence)
}

// modelRemovalRefusal refuses only for what still holds a named model: a local worker's
// residency, or a live local request that reads or writes it.
func modelRemovalRefusal(layout home.Layout, names []string) *exit.Error {
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return problem
	}
	defer store.Close()
	live, problem := store.LiveWorkers()
	if problem != nil {
		return problem
	}
	for _, worker := range live {
		if worker.WorkerID != "remote" {
			return exit.New(exit.Conflict, "local worker %s may still hold model residency", worker.InstanceID).
				WithRemedy("run `cozy unload`, then remove the model repository").
				WithNext("cozy unload")
		}
	}
	for _, name := range names {
		ref, problem := hub.ParseRef(name)
		if problem != nil {
			return problem
		}
		users, problem := store.LocalModelUsers(ref.String())
		if problem != nil {
			return problem
		}
		if len(users) > 0 {
			first := users[0]
			return exit.New(exit.Conflict, "%s still uses %s", runsPhrase(users), ref.String()).
				WithRemedy("let it settle, or cancel it, then remove the model repository").
				WithNext("cozy run cancel " + runReference(first.Number, first.ID))
		}
	}
	return nil
}

func removeModels(ctx *Context, tool *tfs.Tool, layout home.Layout, fence reclamationFence) *exit.Error {
	work, problem := scratch.Temp(layout.Tmp, "repo-remove-")
	if problem != nil {
		return problem
	}
	defer work.Release()
	releases, problem := tool.Releases(filepath.Join(work.Path, "rows.jsonl"))
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
			if _, problem := modelsource.LocalName(ref.Name); problem != nil {
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
			repoScratch := filepath.Join(work.Path, strings.ReplaceAll(ref.String(), "/", "-"))
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
	// Reclamation is part of the act: the name is gone, so its bytes go now, unless a local
	// request may still be moving bytes into the store (a download's admitted objects are
	// unnamed until its commit) or the byte plane names a live holder. Either keeps the name
	// removed and says what deferred it; `cozy model gc` finishes the job.
	if problem := fence.busy(); problem != nil {
		removed.Notes = []string{"reclamation deferred: " + problem.Message}
		removed.Next = []string{"cozy model gc"}
		return emit(ctx, removed)
	}
	report, notes, problem := fence.collect(tool)
	if problem != nil {
		removed.Notes = []string{"reclamation deferred: " + problem.Message}
		removed.Next = []string{"cozy model gc"}
		return emit(ctx, removed)
	}
	removed.Notes = notes
	removed.Aggregates = append(removed.Aggregates, reclaimFields(report)...)
	return emit(ctx, removed)
}

// handleMachineModelDownload freezes a catalog selection before durable acceptance and
// downloads it into a machine's own store: the named rental's, or without --rental this
// computer's machine, the one local runs read.
func handleMachineModelDownload(ctx *Context) *exit.Error {
	machine := strings.TrimSpace(ctx.Inv.Value("--rental"))
	if machine != "" {
		adoptRentalHub(ctx, machine)
	} else {
		machine = machines.Local
	}
	if len(ctx.Inv.Args) > 1 && strings.TrimSpace(ctx.Inv.Args[1]) != "" {
		return exit.Usagef("--rental downloads into the worker store and takes no local destination")
	}
	if ctx.Inv.Bool("--rental-only") || ctx.Inv.Bool("--await") || ctx.Inv.Value("--idempotency-key") != "" {
		return exit.Usagef("machine model download does not accept --rental-only, --await, or --idempotency-key")
	}
	model, problem := resolveRemoteModel(ctx, "", launch.Slot{}, ctx.Inv.Args[0], ctx.Inv.Value("--lane"), nil)
	if problem != nil {
		return problem
	}
	selection := records.RentalInstallSelection{Models: []records.ModelRef{model}}
	return enqueueRentalInstall(ctx, machine, selection)
}
