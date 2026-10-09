package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

func handlePackageUpdateAll(ctx *Context) *exit.Error {
	_, store, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return problem
	}
	installed, problem := store.Installed()
	store.Close()
	if problem != nil {
		return problem
	}
	list := output.List{Name: "packages",
		Fields:    []string{"package", "from", "to", "status", "detail"},
		AllFields: []string{"package", "from", "to", "status", "detail", "error_code"},
		Machine:   []string{"error_code"},
	}
	counts := map[string]int{"updated": 0, "current": 0, "failed": 0, "skipped": 0}
	for _, prior := range installed {
		if !inHubScope(ctx, prior) {
			continue
		}
		// Each installation updates from the hub it came from, with that hub's credential.
		row := updateInstalledPackage(ctx.forHub(prior.Hub), prior)
		if prior.SourceKind == "tensorhub" {
			row["hub"] = ctx.Cfg.HubLabel(either(prior.Hub, ctx.Cfg.HubURL))
		}
		list.Rows = append(list.Rows, row)
		counts[row["status"]]++
	}
	list.Fields = append(list.Fields, "hub")
	list.AllFields = append(list.AllFields, "hub")
	if !everyHub(ctx) {
		list.Notes = []string{"Only installations from hub " + ctx.Cfg.HubText(ctx.Cfg.HubURL) + "; omit --tensorhub to update every hub's."}
	}
	for _, status := range []string{"updated", "current", "failed", "skipped"} {
		list.Aggregates = append(list.Aggregates, output.Field{K: status, V: counts[status]})
	}
	list.Trail = []string{"Model downloads: skipped."}
	if counts["failed"] > 0 {
		ctx.exitCode = 1
	}
	// A bulk mutation reports every package, including failures beyond the usual list cap.
	ctx.Inv.Mode.Full = true
	return emit(ctx, list)
}

func updateInstalledPackage(ctx *Context, prior records.PackageInstall) map[string]string {
	row := map[string]string{"package": prior.Package, "from": prior.Version, "to": prior.Version, "status": "skipped"}
	fail := func(problem *exit.Error) map[string]string {
		row["status"], row["error_code"], row["detail"] = "failed", problem.ErrName(), problem.Message
		if problem.Remedy != "" {
			row["detail"] += "; " + problem.Remedy
		}
		return row
	}
	if prior.SourceKind != "tensorhub" {
		row["detail"] = "local or unpublished install; no registry update"
		return row
	}
	if prior.Hub == "" {
		row["detail"] = "no recorded hub; reinstall it to update it"
		return row
	}
	if !immutablePackageVersion.MatchString(prior.Version) {
		row["detail"] = "development or prerelease selection preserved; use package install --version to change it"
		return row
	}
	ref, problem := hub.ParseRef(prior.Package)
	if problem != nil {
		return fail(problem)
	}
	hctx, cancel := hub.Context()
	card, problem := client(ctx).PackageCard(hctx, ref)
	cancel()
	if problem != nil {
		return fail(problem)
	}
	latest, problem := newestPackageRelease(card.Releases)
	if problem != nil {
		return fail(problem)
	}
	if latest == prior.Version {
		row["status"] = "current"
		return row
	}
	newest, problem := newestPackageRelease([]hub.ReleaseSummary{{Release: prior.Version}, {Release: latest}})
	if problem != nil {
		return fail(problem)
	}
	if newest != latest {
		row["detail"] = "installed version is newer than the registry; preserved"
		return row
	}
	row["to"] = latest
	perPackage := *ctx
	perPackage.Inv = &Invocation{Args: []string{prior.Package},
		Values: values("--version", latest), Mode: ctx.Mode()}
	result, _, problem := installRegistryPackage(&perPackage, prior.ID)
	if problem != nil {
		return fail(problem)
	}
	row["status"], row["to"] = "updated", result.Install.Version
	if result.Idempotent {
		row["status"] = "current"
	}
	row["detail"] = strings.Join(result.Warnings, "; ")
	return row
}
