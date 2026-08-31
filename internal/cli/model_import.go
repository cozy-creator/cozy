package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/tfs"
)

func handleModelImport(ctx *Context) *exit.Error {
	name := strings.TrimSpace(ctx.Inv.Value("--name"))
	if problem := modelsource.LocalName(name); problem != nil {
		return problem
	}
	cwd, err := os.Getwd()
	if err != nil {
		return exit.Internalf("cannot resolve the current directory: %s", err)
	}
	source, problem := modelsource.Parse(ctx.Inv.Args[0], cwd)
	if problem != nil {
		return problem
	}
	var token secret.Value
	if ctx.Inv.Bool("--token-stdin") {
		if source.Kind == modelsource.LocalFile {
			return exit.Usagef("--token-stdin applies only to a Hugging Face or Civitai source")
		}
		token, problem = readToken(ctx)
		if problem != nil {
			return problem
		}
	} else if source.Kind == modelsource.HuggingFace {
		token = ctx.Cfg.HuggingFaceToken
	} else if source.Kind == modelsource.Civitai {
		token = ctx.Cfg.CivitaiToken
	}

	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return problem
	}
	tool, problem := tfs.Open(ctx.Cfg, layout)
	if problem != nil {
		return problem
	}
	runctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	var resolved modelsource.Plan
	var resolver *modelsource.Resolver
	if source.Kind == modelsource.LocalFile {
		resolved, problem = modelsource.ResolveLocal(source)
	} else {
		resolver, problem = modelsource.NewResolver(source.Kind, token)
		if problem == nil {
			resolved, problem = resolver.Resolve(runctx, source)
		}
	}
	if problem != nil {
		return importCanceled(runctx, problem)
	}

	root := filepath.Join(layout.Transfer, "model-import-"+importOperation(name, resolved.Canonical))
	if err := os.MkdirAll(root, 0o700); err != nil {
		return exit.Internalf("cannot create model import operation: %s", err)
	}
	headerRoot := filepath.Join(root, "headers")
	var headerFiles []modelsource.StagedFile
	if source.Kind == modelsource.LocalFile {
		headerFiles = []modelsource.StagedFile{{Path: source.Path, Carrier: true}}
	} else {
		headerFiles, problem = resolver.Stage(runctx, resolved, headerRoot, true, progress(ctx))
		if problem != nil {
			return importCanceled(runctx, problem)
		}
	}
	headerPlanPath := filepath.Join(root, "source-plan.json")
	headerPlan, problem := tool.PlanSource(importCarriers(headerFiles,
		source.Kind != modelsource.LocalFile), headerPlanPath)
	if problem != nil {
		return problem
	}
	if problem := tool.PreviewSource(headerPlanPath); problem != nil {
		return problem
	}

	selected := resolved
	if source.Kind != modelsource.LocalFile {
		selected, problem = resolved.Select(planMembers(headerPlan, headerFiles))
		if problem != nil {
			return problem
		}
	}
	if ctx.Inv.Bool("--dry-run") {
		_ = os.RemoveAll(root)
		fields := []output.Field{
			{K: "model", V: "local/" + name}, {K: "source", V: selected.Canonical},
			{K: "status", V: "planned"}, {K: "files", V: len(selected.Files)},
			{K: "bytes", V: output.Bytes(selected.Bytes)}, {K: "target", V: headerPlan.Target},
			{K: "changed", V: false},
		}
		return emit(ctx, compactRecord(fields, "model", "source", "status", "bytes", "changed"))
	}

	observed, problem := tool.ObserveLocal(name, filepath.Join(root, "local-rows.jsonl"))
	if problem != nil {
		return problem
	}
	fullFiles := headerFiles
	if source.Kind != modelsource.LocalFile {
		fullFiles, problem = resolver.Stage(runctx, selected, filepath.Join(root, "files"), false, progress(ctx))
		if problem != nil {
			return importCanceled(runctx, problem)
		}
	}
	fullPlanPath := filepath.Join(root, "source-plan-full.json")
	fullPlan, problem := tool.PlanSource(importCarriers(fullFiles,
		source.Kind != modelsource.LocalFile), fullPlanPath)
	if problem != nil {
		return problem
	}
	manifest, problem := tool.RunSource(fullPlanPath)
	if problem != nil {
		return importCanceled(runctx, problem)
	}
	if problem := tool.InstallLocal(fullPlan.Session, name, "sha256:"+selected.SelectionSHA256,
		observed, selected.Canonical, selected.License); problem != nil {
		return problem
	}
	alias, problem := tool.ResolveLocal(name)
	if problem != nil {
		return problem
	}
	if alias.ManifestDigest != manifest || alias.SourceSelection != "sha256:"+selected.SelectionSHA256 {
		return exit.Internalf("TensorFS installed local/%s under different exact identities", name)
	}
	if problem := tool.VerifyManifest(alias.ManifestDigest); problem != nil {
		return problem
	}
	if err := os.RemoveAll(root); err != nil {
		return exit.Internalf("model imported, but transient source staging could not be deleted: %s", err)
	}
	fields := []output.Field{
		{K: "model", V: "local/" + name}, {K: "manifest_id", V: alias.ManifestDigest},
		{K: "source", V: selected.Canonical}, {K: "status", V: "imported"},
		{K: "bytes", V: output.Bytes(selected.Bytes)}, {K: "changed", V: observed != alias.RepositoryDigest},
	}
	return emit(ctx, compactRecord(fields, "model", "manifest_id", "source", "status", "changed"))
}

func importOperation(name, source string) string {
	sum := sha256.Sum256([]byte(name + "\x00" + source))
	return hex.EncodeToString(sum[:8])
}

func importCarriers(files []modelsource.StagedFile, labelled bool) []tfs.SourceCarrier {
	carriers := make([]tfs.SourceCarrier, 0)
	for _, file := range files {
		if !file.Carrier {
			continue
		}
		carrier := tfs.SourceCarrier{Path: file.Path}
		if labelled {
			carrier.Member = file.Member
		}
		carriers = append(carriers, carrier)
	}
	return carriers
}

func planMembers(plan tfs.SourcePlan, staged []modelsource.StagedFile) []string {
	byPath := make(map[string]string, len(staged))
	for _, file := range staged {
		byPath[file.Path] = file.Member
	}
	members := make([]string, 0, len(plan.Sources))
	for _, source := range plan.Sources {
		member := source.SourceMember
		if member == "" {
			member = byPath[source.Path]
		}
		members = append(members, member)
	}
	return members
}

func importCanceled(ctx context.Context, problem *exit.Error) *exit.Error {
	if ctx.Err() != nil {
		return exit.New(exit.Canceled, "model import canceled; verified partial downloads remain resumable")
	}
	return problem
}
