package cli

import (
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// Named hubs work like kubectl contexts: `cozy hub use` picks the hub later commands
// address, `--tensorhub` picks one for a single command, and the one daemon serves every
// hub at once. Records keep the hub they were created on, so switching strands nothing.

func handleHubList(ctx *Context) *exit.Error {
	rentals, counted := localRentalsByHub(ctx)
	origins := map[string]bool{}
	names := make([]string, 0, len(ctx.Cfg.Hubs))
	for name, origin := range ctx.Cfg.Hubs {
		names = append(names, name)
		origins[origin] = true
	}
	sort.Strings(names)
	type entry struct{ name, origin string }
	entries := make([]entry, 0, len(names)+len(rentals)+1)
	for _, name := range names {
		entries = append(entries, entry{name, ctx.Cfg.Hubs[name]})
	}
	// Unnamed origins still hold work or are selected: list them too.
	extra := []string{ctx.Cfg.HubURL}
	for origin := range rentals {
		extra = append(extra, origin)
	}
	sort.Strings(extra)
	for _, origin := range extra {
		if !origins[origin] {
			origins[origin] = true
			entries = append(entries, entry{"", origin})
		}
	}
	list := output.List{Name: "hubs",
		Fields:         []string{"current", "name", "url", "login", "rentals"},
		TypedFields:    []string{"name", "url", "current", "logged_in", "rentals"},
		TypedAllFields: []string{"name", "url", "current", "logged_in", "rentals"},
		Next:           []string{"cozy hub use <name>", "cozy hub add <name> <url>"},
	}
	for _, each := range entries {
		current := each.origin == ctx.Cfg.HubURL && (each.name == "" || each.name == ctx.Cfg.HubName)
		login := accountauth.New(ctx.Cfg.ForHub(each.origin)).CredentialPresent()
		row := map[string]string{"current": "", "name": orNone(each.name), "url": each.origin,
			"login": "no", "rentals": "unknown"}
		typed := map[string]any{"url": each.origin, "current": current, "logged_in": login}
		if current {
			row["current"] = "*"
		}
		if login {
			row["login"] = "yes"
		}
		if each.name != "" {
			typed["name"] = each.name
		}
		if counted {
			row["rentals"] = strconv.Itoa(rentals[each.origin])
			typed["rentals"] = rentals[each.origin]
		}
		list.Rows = append(list.Rows, row)
		list.TypedRows = append(list.TypedRows, typed)
	}
	if ctx.Cfg.HubURLSource == "flag" {
		list.Trail = append(list.Trail, "--tensorhub selects this command's hub only; `cozy hub use` changes the current one.")
	}
	return emit(ctx, list)
}

// localRentalsByHub counts this host's live rental records per hub, without the daemon.
func localRentalsByHub(ctx *Context) (map[string]int, bool) {
	store, recorded := existingRecords(ctx)
	if store == nil {
		// A root with no records holds no rentals; unreadable records are unknown.
		return map[string]int{}, !recorded
	}
	defer store.Close()
	rows, problem := store.Rentals()
	if problem != nil {
		return nil, false
	}
	counts := map[string]int{}
	for _, row := range rows {
		if !hub.RentalAbsent(row.State) {
			counts[ctx.Cfg.ForHub(row.Hub).HubURL]++
		}
	}
	return counts, true
}

func handleHubUse(ctx *Context) *exit.Error {
	if selected, _, problem := config.ResolveHub(ctx.Inv.Args[0], ctx.Cfg.Hubs); problem == nil &&
		ctx.Cfg.StaticToken() && selected != ctx.Cfg.ConfiguredHubURL {
		// The operator token names no hub of its own: switching would carry it along.
		target := ctx.Inv.Args[0]
		return exit.Named(exit.Conflict, "hub.static_token_bound",
			"the operator token set by %s belongs to %s; switching the current hub would send it to %s",
			strings.Join(ctx.Cfg.StaticTokenSettings(), " and "), ctx.Cfg.ConfiguredHubURL, target).
			WithRemedy("remove %s, then run `cozy hub use %s` again; each hub then uses its own machine login. "+
				"To address %s for one command instead, pass --tensorhub=%s, which never sends the token",
				strings.Join(ctx.Cfg.StaticTokenSettings(), " and "), target, target, target).
			WithNext("cozy auth login <email> --tensorhub="+target, "cozy hub use "+target)
	}
	origin, problem := config.UseHub(ctx.Cfg.Home, ctx.Inv.Args[0], ctx.Cfg.Hubs)
	if problem != nil {
		return problem
	}
	selected := ctx.Cfg.ForHub(origin)
	record := compactRecord([]output.Field{
		{K: "hub", V: selected.HubLabel(origin)}, {K: "url", V: origin}, {K: "current", V: true},
	}, "hub", "url")
	record.Summary = []string{"Commands now use " + selected.HubLabel(origin) + " (" + origin + ")."}
	record.Notes = []string{"existing runs and rentals keep their own hub; the daemon serves every hub"}
	if !accountauth.New(selected).CredentialPresent() {
		record.Next = []string{"cozy auth login <email>"}
	}
	return emit(ctx, record)
}

func handleHubAdd(ctx *Context) *exit.Error {
	name := ctx.Inv.Args[0]
	origin, problem := config.AddHub(ctx.Cfg.Home, name, ctx.Inv.Args[1])
	if problem != nil {
		return problem
	}
	record := compactRecord([]output.Field{{K: "hub", V: name}, {K: "url", V: origin}}, "hub", "url")
	record.Next = []string{"cozy hub use " + name, "cozy auth login <email> --tensorhub=" + name}
	return emit(ctx, record)
}

func handleHubRemove(ctx *Context) *exit.Error {
	name := ctx.Inv.Args[0]
	if problem := config.RemoveHub(ctx.Cfg.Home, name); problem != nil {
		return problem
	}
	return emit(ctx, compactRecord([]output.Field{{K: "hub", V: name}, {K: "removed", V: true}}, "hub", "removed"))
}

// adoptRentalHub points a command that names an existing rental at that rental's hub,
// with that hub's credential: work on a record uses the record's hub. An explicit
// --tensorhub is kept, and the daemon refuses the mismatch for this one command.
func adoptRentalHub(ctx *Context, name string) {
	name = strings.TrimSpace(name)
	if name == "" || ctx.Cfg.HubURLSource == "flag" {
		return
	}
	store, _ := existingRecords(ctx)
	if store == nil {
		return
	}
	defer store.Close()
	known, problem := rental.Resolve(store, name)
	if problem != nil {
		return
	}
	origin, invalid := config.HubOrigin(known.Hub)
	if invalid != nil {
		return
	}
	scoped := ctx.forHub(origin)
	ctx.Cfg, ctx.AccountAuth = scoped.Cfg, scoped.AccountAuth
}

// existingRecords opens this root's records only when they exist: a read never
// creates a database or a root. recorded reports whether a database exists.
func existingRecords(ctx *Context) (*records.Store, bool) {
	path := home.Paths(ctx.Cfg.Home).DB
	if _, err := os.Stat(path); err != nil {
		return nil, false
	}
	store, problem := records.Open(path)
	if problem != nil {
		return nil, true
	}
	return store, true
}

// adoptInstallHub points a run of a published install at the hub it came from, so its
// model bindings are that hub's. An explicit --tensorhub is kept.
func adoptInstallHub(ctx *Context, install records.PackageInstall) {
	if install.SourceKind != "tensorhub" || install.Hub == "" || ctx.Cfg.HubURLSource == "flag" {
		return
	}
	origin, invalid := config.HubOrigin(install.Hub)
	if invalid != nil {
		return
	}
	scoped := ctx.forHub(origin)
	ctx.Cfg, ctx.AccountAuth = scoped.Cfg, scoped.AccountAuth
}

// everyHub is whether a list shows every hub's rows: always, unless --tensorhub names one.
// --all-hubs is the default and kept for scripts.
func everyHub(ctx *Context) bool {
	return ctx.Cfg.HubURLSource != "flag" || ctx.Inv.Bool("--all-hubs")
}
