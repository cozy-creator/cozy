package cli

import (
	"fmt"
	"maps"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

func selectedHubText(ctx *Context) string {
	label := ctx.Cfg.HubLabel(ctx.Cfg.HubURL)
	if label == ctx.Cfg.HubURL {
		return label
	}
	return fmt.Sprintf("%s (%s)", label, ctx.Cfg.HubURL)
}

// Inventory provenance is not a catalog availability check. Listing never contacts
// another Hub and never claims an installed release still exists at its source.
func installHubScope(ctx *Context, inst records.PackageInstall) string {
	if inst.SourceKind != "tensorhub" {
		return "local"
	}
	if inst.Hub == "" {
		return "unknown hub"
	}
	if inst.PublishedAt(ctx.Cfg.HubURL) {
		return "selected hub"
	}
	return "other hub"
}

// A missing package stays missing at the selected Hub. Local provenance can explain
// the mismatch, but cannot supply a package, change the Hub, or trigger a fallback.
func packageHubProblem(ctx *Context, pkg string, problem *exit.Error) *exit.Error {
	if problem == nil || problem.ErrName() != "package.not_found" || strings.HasPrefix(pkg, "local/") {
		return problem
	}
	copy := *problem
	copy.Message = fmt.Sprintf("no package %s at selected Hub %s", pkg, selectedHubText(ctx))
	copy.Details = maps.Clone(problem.Details)
	if copy.Details == nil {
		copy.Details = map[string]any{}
	}
	copy.Details["package"], copy.Details["selected_hub"] = pkg, ctx.Cfg.HubURL
	copy.Next = []string{"cozy package search --tensorhub=" + ctx.Cfg.HubLabel(ctx.Cfg.HubURL), "cozy hub list"}
	// This is a read-only hint from an existing record, not another resolution step.
	store, openProblem := records.OpenReadOnly(home.Paths(ctx.Cfg.Home).DB)
	if openProblem != nil {
		return &copy
	}
	defer store.Close()
	_, installed, readProblem := store.ActivePackage(pkg)
	if readProblem != nil || installed == nil || installHubScope(ctx, *installed) != "other hub" {
		return &copy
	}
	origin := ctx.forHub(installed.Hub).Cfg.HubURL
	label := ctx.Cfg.HubLabel(origin)
	copy.Details["installed_hub"] = origin
	copy.Details["installed_version"] = installed.Version
	copy.Remedy = fmt.Sprintf("the installed %s@%s came from %s; add --tensorhub=%s to explicitly use that Hub for this command", pkg, installed.Version, origin, label)
	copy.Next = []string{"cozy package info " + pkg + " --tensorhub=" + label, "cozy hub list"}
	return &copy
}

// Local source captures have no catalog owner. Published identities need a known,
// matching origin even if an older install has the same package name/version.
func installedInScope(ctx *Context, inst records.PackageInstall) bool {
	return inst.SourceKind != "tensorhub" || inst.PublishedAt(ctx.Cfg.HubURL)
}

func foreignPackageRemoval(ctx *Context, inst records.PackageInstall) *exit.Error {
	problem := exit.Named(exit.Conflict, "package.other_hub", "%s is not installed from selected Hub %s; its installation was preserved", inst.Package, selectedHubText(ctx))
	if inst.Hub == "" {
		return problem.WithRemedy("this installation has no recorded Hub; inspect it with cozy package list --all-hubs --full")
	}
	return problem.WithRemedy("the installation came from %s; explicitly use --tensorhub=%s to remove it", inst.Hub, ctx.Cfg.HubLabel(inst.Hub))
}
