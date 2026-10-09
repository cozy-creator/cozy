package cli

import (
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// An org/name reference resolves at this command's hub (the current one, or --tensorhub's)
// and nowhere else: the same org/name at two hubs is two packages. Work on a record (an
// installation, a rental, a run) uses that record's own hub. Inventory spans every hub
// unless --tensorhub narrows it to that hub and local sources.

func inHubScope(ctx *Context, inst records.PackageInstall) bool {
	return everyHub(ctx) || inst.SourceKind != "tensorhub" || inst.PublishedAt(ctx.Cfg.HubURL)
}

// installHubScope is how a reference in this command reaches an installation. It is
// provenance, not a catalog check: listing never contacts any Hub.
func installHubScope(ctx *Context, inst records.PackageInstall) string {
	switch {
	case inst.SourceKind != "tensorhub":
		return "local"
	case inst.Hub == "":
		return "unknown hub"
	case inst.PublishedAt(ctx.Cfg.HubURL):
		return "current hub"
	}
	return "other hub"
}

// packageHubProblem explains a package this command's hub does not have, from this computer's
// records alone; no other hub is asked and nothing falls back. It names the hub searched,
// another hub this computer installed or ran the package from, and the same name under another
// org at this hub, each with the command that works. The hub's own remedy addresses a
// publisher, not a reader.
func packageHubProblem(ctx *Context, pkg string, problem *exit.Error) *exit.Error {
	if problem == nil || problem.ErrName() != "package.not_found" || strings.HasPrefix(pkg, "local/") {
		return problem
	}
	here := ctx.Cfg.HubURL
	miss := *problem
	miss.Message = fmt.Sprintf("no package %s on hub %s", pkg, ctx.Cfg.HubText(here))
	miss.Details = maps.Clone(problem.Details)
	if miss.Details == nil {
		miss.Details = map[string]any{}
	}
	miss.Details["package"], miss.Details["tensorhub"] = pkg, here
	_, name, _ := strings.Cut(pkg, "/")
	search := "cozy package search " + name
	if ctx.Cfg.HubURLSource == "flag" {
		search += " --tensorhub=" + ctx.Cfg.HubLabel(here)
	}
	miss.Remedy, miss.Next = "check the name, and the current hub with `cozy hub list`", []string{search, "cozy hub list"}
	store, opened := records.OpenReadOnly(home.Paths(ctx.Cfg.Home).DB)
	if opened != nil {
		return &miss
	}
	defer store.Close()
	var seen, next []string
	sighted := func(fact, command string) {
		seen = append(seen, fact)
		if command != "" {
			next = append(next, command)
		}
	}
	pins, _ := store.Pins(pkg)
	if i := slices.IndexFunc(pins, func(p records.Pin) bool { return p.Hub != "" && p.Hub != here }); i >= 0 {
		version := ""
		if inst, _ := store.Install(pins[i].InstallID); inst != nil {
			version = "@" + inst.Version
		}
		sighted(fmt.Sprintf("%s%s is installed from hub %s", pkg, version, ctx.Cfg.HubText(pins[i].Hub)),
			retyped(ctx, "", "", ctx.Cfg.HubLabel(pins[i].Hub)))
	} else if runs, _ := store.Requests("", pkg, 20); len(runs) > 0 {
		if i := slices.IndexFunc(runs, func(r records.Request) bool { return r.Hub != "" && strings.TrimRight(r.Hub, "/") != here }); i >= 0 {
			sighted(fmt.Sprintf("%s ran from hub %s", pkg, ctx.Cfg.HubText(runs[i].Hub)), retyped(ctx, "", "", ctx.Cfg.HubLabel(runs[i].Hub)))
		}
	}
	installed, _ := store.Installed()
	if i := slices.IndexFunc(installed, func(inst records.PackageInstall) bool {
		_, other, _ := strings.Cut(inst.Package, "/")
		return other == name && inst.Package != pkg && inst.PublishedAt(here)
	}); i >= 0 {
		sibling := installed[i]
		sighted(fmt.Sprintf("%s@%s is installed from hub %s", sibling.Package, sibling.Version, ctx.Cfg.HubText(here)),
			retyped(ctx, pkg, sibling.Package, ""))
	}
	if len(seen) > 0 {
		miss.Remedy, miss.Next = strings.Join(seen, "; "), append(next, "cozy hub list")
	}
	return &miss
}

var plainWord = regexp.MustCompile(`^[A-Za-z0-9_@%+=:,./-]+$`)

// retyped is this command as typed with the package reference from renamed to, and with
// --tensorhub=hub when hub is named; "" when the typed command is unknown.
func retyped(ctx *Context, from, to, hub string) string {
	if len(ctx.argv) == 0 {
		return ""
	}
	words, skip := []string{"cozy"}, false
	for _, arg := range ctx.argv {
		switch {
		case skip:
			skip = false
			continue
		case hub != "" && arg == "--tensorhub":
			skip = true
			continue
		case hub != "" && strings.HasPrefix(arg, "--tensorhub="):
			continue
		case from != "" && (strings.EqualFold(arg, from) || len(arg) > len(from) && strings.EqualFold(arg[:len(from)], from) && strings.ContainsRune("/@", rune(arg[len(from)]))):
			arg = to + arg[len(from):]
		}
		if !plainWord.MatchString(arg) {
			arg = "'" + strings.ReplaceAll(arg, "'", `'\''`) + "'"
		}
		words = append(words, arg)
	}
	if hub != "" {
		words = append(words, "--tensorhub="+hub)
	}
	return strings.Join(words, " ")
}

func foreignPackageRemoval(ctx *Context, origin, pkg string) *exit.Error {
	from := "an unrecorded hub"
	if origin != "" {
		from = "hub " + ctx.Cfg.HubText(origin)
	}
	return exit.Named(exit.Conflict, "package.other_hub", "%s is installed from %s, not hub %s; it was kept",
		pkg, from, ctx.Cfg.HubText(ctx.Cfg.HubURL)).
		WithRemedy("omit --tensorhub to remove every hub's installation of %s", pkg)
}
