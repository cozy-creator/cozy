package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
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
	if len(ctx.Inv.Args) < 2 || strings.TrimSpace(ctx.Inv.Args[1]) == "" {
		return handleMachineModelDownload(ctx)
	}
	adoptRentalHub(ctx, ctx.Inv.Value("--rental"))
	return handleModelTransfer(ctx, "model-download")
}

// handleModelTransfer puts a local file, a local/ alias or a Tensorhub checkpoint in a
// Tensorhub destination (upload), or a Tensorhub model under a local/ alias (download): one
// warm run on the machine that makes and holds it, this computer's or --rental's.
func handleModelTransfer(ctx *Context, kind string) *exit.Error {
	sourceArg, destinationArg := strings.TrimSpace(ctx.Inv.Args[0]), strings.TrimSpace(ctx.Inv.Args[1])
	machine := ctx.Inv.Value("--rental")
	if machine == "" && rentalRequested(ctx) {
		return exit.Usagef("a model transfer runs on this computer's machine or on --rental=NAME")
	}
	if len(ctx.Inv.Values["--source-profile"]) > 0 {
		return exit.Usagef("--source-profile selects the profiles a provider ingest converts").
			WithRemedy("omit --source-profile to use the one profile the headers match")
	}
	var selection records.RentalInstallSelection
	if kind == "model-upload" {
		ref, problem := hub.ParseRef(destinationArg)
		if problem != nil {
			return problem
		}
		if ref.Org == "local" {
			return exit.Usagef("local/ is reserved for private aliases and cannot be a Tensorhub destination").
				WithRemedy("upload under your Tensorhub account, for example alice/%s", ref.Name)
		}
		if _, problem := ownedPublication(ctx, ref); problem != nil {
			return problem
		}
		selection.Destination = ref.String()
	} else {
		name, local, problem := modelsource.LocalAlias(destinationArg)
		if !local {
			return exit.Usagef("model download destination %q is not local/name", destinationArg)
		}
		if problem != nil {
			return problem
		}
		selection.Destination = "local/" + name
	}
	model, file, problem := transferSource(ctx, sourceArg, kind == "model-upload")
	if problem != nil {
		return problem
	}
	selection.Models, selection.Write = []records.ModelRef{model}, file
	return enqueueRentalInstall(ctx, either(machine, machines.Local), selection, ctx.Inv.Bool("--await"))
}

// transferSource is the model a transfer moves: a local file (written to the machine, then
// made like a provider's single file), a local/ alias the machine holds, or a Tensorhub
// checkpoint. file is the local file to write.
func transferSource(ctx *Context, raw string, upload bool) (records.ModelRef, string, *exit.Error) {
	lane := strings.TrimSpace(ctx.Inv.Value("--lane"))
	if name, local, problem := modelsource.LocalAlias(raw); local {
		if problem != nil {
			return records.ModelRef{}, "", problem
		}
		if !upload || lane != "" {
			return records.ModelRef{}, "", exit.Usagef("a local/%s alias is uploaded as it is, to a Tensorhub repository", name)
		}
		return records.ModelRef{Slot: "model", Model: "local/" + name}, "", nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return records.ModelRef{}, "", exit.Internalf("cannot resolve the current directory: %s", err)
	}
	parsed, problem := modelsource.Parse(raw, cwd)
	if problem == nil && parsed.Kind == modelsource.LocalFile {
		if !upload || lane != "" {
			return records.ModelRef{}, "", exit.Usagef("a local file is uploaded as it is, to a Tensorhub repository")
		}
		digest, problem := fileDigest(parsed.Path, parsed.Bytes)
		if problem != nil {
			return records.ModelRef{}, "", problem
		}
		source := "object://sha256:" + digest + "/" + url.PathEscape(filepath.Base(parsed.Path))
		return records.ModelRef{Slot: "model", Source: source}, parsed.Path, nil
	}
	if problem == nil {
		// A provider source kept under a local alias: made on the machine, as an upload makes it.
		if lane != "" {
			return records.ModelRef{}, "", exit.Usagef("--lane selects only a Tensorhub model release")
		}
		source, problem := pinnedProviderSource(ctx, parsed.Canonical)
		return records.ModelRef{Slot: "model", Source: source}, "", problem
	}
	model, problem := resolveRemoteModel(ctx, "", launch.Slot{}, raw, lane, nil)
	model.Slot = "model"
	return model, "", problem
}

// fileDigest is a local file's sha256, refused if its length changed since it was measured.
func fileDigest(path string, length int64) (string, *exit.Error) {
	file, err := os.Open(path)
	if err != nil {
		return "", exit.New(exit.NotFound, "cannot open local model source: %s", err)
	}
	defer file.Close()
	hash := sha256.New()
	if read, err := io.Copy(hash, file); err != nil || read != length {
		return "", exit.Named(exit.Conflict, "model_source.local_changed",
			"local model source changed while measuring its exact identity")
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
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
		// An as-is plan narrows itself from the headers, as an unnamed selection does.
		// A Civitai primary with its companions is narrowed by its header plan instead.
		if len(resolved.Files) > 1 && len(sourceProfiles) > 0 && !slices.Contains(sourceProfiles, tfs.AsIsProfile) &&
			!slices.ContainsFunc(resolved.Files, func(file modelsource.File) bool { return file.Companion }) {
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
	return selectPublishMembers(source, members)
}

// selectPublishMembers narrows a provider resolution to exact carrier members and their shards.
func selectPublishMembers(source publishSource, members []string) (publishSource, *exit.Error) {
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
