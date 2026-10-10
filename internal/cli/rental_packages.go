package cli

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

// rentalEnvironments are the installations a rental's machine holds for this computer's key,
// each with the hub it came from as this computer names it, within --tensorhub's scope.
func rentalEnvironments(ctx *Context) (string, []api.MachineEnvironment, *exit.Error) {
	rentalID := machines.Local // this computer's machine, as --rental=local installs on it
	if ctx.Inv.Value("--rental") != machines.Local {
		var problem *exit.Error
		if rentalID, problem = requestedRental(ctx); problem != nil {
			return "", nil, problem
		}
	}
	client, problem := dial(ctx)
	if problem != nil {
		return "", nil, problem
	}
	status, problem := client.MachineStatus(rentalID)
	if problem != nil {
		return "", nil, problem
	}
	// A machine that names no hub (older software) keeps every installation in scope.
	envs := slices.DeleteFunc(status.Environments, func(env api.MachineEnvironment) bool {
		return !everyHub(ctx) && !strings.HasPrefix(env.Package, "local/") && env.Hub != "" && env.Hub != ctx.Cfg.HubURL
	})
	return rentalID, envs, nil
}

// rentalPackageUpdate is `cozy package update [<package>…] --rental`: each published package the
// rental holds goes to its hub's newest release, read there by the machine (one it holds moves
// nothing), and the package's other releases from that hub leave the machine.
func rentalPackageUpdate(ctx *Context) *exit.Error {
	rental := ctx.Inv.Value("--rental")
	_, envs, problem := rentalEnvironments(ctx)
	if problem != nil {
		return problem
	}
	if envs, problem = namedInstalls(ctx.Inv.Args, envs, func(env api.MachineEnvironment) string { return env.Package }, "on "+rental); problem != nil {
		return problem
	}
	list := output.List{Name: "packages", Fields: []string{"package", "from", "to", "status", "detail", "hub"},
		AllFields: []string{"package", "from", "to", "status", "detail", "hub", "error_code"}, Machine: []string{"error_code"}}
	counts := map[string]int{"updated": 0, "current": 0, "failed": 0, "skipped": 0}
	type key struct{ pkg, hub string }
	var order []key
	held := map[key][]api.MachineEnvironment{}
	for _, env := range envs {
		k := key{env.Package, env.Hub}
		if held[k] == nil {
			order = append(order, k)
		}
		held[k] = append(held[k], env)
	}
	for _, k := range order {
		releases := make([]string, 0, len(held[k]))
		for _, env := range held[k] {
			releases = append(releases, env.Release)
		}
		row := map[string]string{"package": k.pkg, "from": strings.Join(releases, ", "), "to": "", "hub": ctx.Cfg.HubLabel(k.hub)}
		switch {
		case strings.HasPrefix(k.pkg, "local/"):
			row["status"], row["detail"] = "skipped", "local code; install its directory again to change it"
		default:
			row["status"], row["to"] = updateRentalPackage(ctx, rental, k.pkg, k.hub, held[k], row)
		}
		counts[row["status"]]++
		list.Rows = append(list.Rows, row)
	}
	for _, status := range []string{"updated", "current", "failed", "skipped"} {
		list.Aggregates = append(list.Aggregates, output.Field{K: status, V: counts[status]})
	}
	list.Trail = []string{"Model downloads: skipped."}
	if counts["failed"] > 0 {
		ctx.exitCode = 1
	}
	ctx.Inv.Mode.Full = true
	return emit(ctx, list)
}

// updateRentalPackage installs pkg's newest release from hub on the rental, then removes its
// other releases from that hub. Answers the row's status and the release it holds now.
func updateRentalPackage(ctx *Context, rental, pkg, hub string, held []api.MachineEnvironment, row map[string]string) (string, string) {
	fail := func(problem *exit.Error) (string, string) {
		row["error_code"], row["detail"] = problem.ErrName(), problem.Message
		return "failed", ""
	}
	if hub == "" {
		hub = ctx.Cfg.HubURL // a machine that names no hub: this command's
		row["hub"] = ctx.Cfg.HubLabel(hub)
	}
	installed, problem := settleRentalInstall(ctx, rental, records.RentalInstallSelection{Package: pkg, Hub: hub})
	if problem != nil {
		return fail(problem)
	}
	var result struct{ Release string }
	_ = json.Unmarshal(installed.Result, &result)
	status := "current"
	for _, env := range held {
		if env.Release == result.Release {
			continue
		}
		status = "updated"
		removal := records.RentalInstallSelection{Package: pkg, Release: env.Release, Hub: hub, Remove: env.Installation}
		if _, problem := settleRentalInstall(ctx, rental, removal); problem != nil {
			row["detail"] = fmt.Sprintf("%s@%s stays: %s", pkg, env.Release, problem.Message)
		}
	}
	return status, result.Release
}

// rentalPackageRemove is `cozy package remove <package>… --rental`: every installation of each
// name on the rental (one hub's with --tensorhub) leaves the machine, its environment with it.
func rentalPackageRemove(ctx *Context) *exit.Error {
	rental := ctx.Inv.Value("--rental")
	_, envs, problem := rentalEnvironments(ctx)
	if problem != nil {
		return problem
	}
	if envs, problem = namedInstalls(ctx.Inv.Args, envs, func(env api.MachineEnvironment) string { return env.Package }, "on "+rental); problem != nil {
		return problem
	}
	removed := output.List{Name: "packages", Fields: []string{"package", "release", "hub", "reclaimed"},
		AllFields: []string{"package", "release", "hub", "reclaimed", "installation"}}
	var freed int64
	for _, env := range envs {
		selection := records.RentalInstallSelection{Package: env.Package, Release: env.Release, Hub: env.Hub, Remove: env.Installation}
		settled, problem := settleRentalInstall(ctx, rental, selection)
		if problem != nil {
			return problem
		}
		var result struct {
			Reclaimed int64    `json:"reclaimed_bytes"`
			Retained  []string `json:"retained"`
		}
		_ = json.Unmarshal(settled.Result, &result)
		freed += result.Reclaimed
		if len(result.Retained) > 0 {
			removed.Notes = append(removed.Notes, env.Package+"@"+env.Release+"'s files stay until its running executor ends")
		}
		removed.Rows = append(removed.Rows, map[string]string{"package": env.Package, "release": env.Release,
			"hub": ctx.Cfg.HubLabel(env.Hub), "reclaimed": output.Bytes(result.Reclaimed), "installation": env.Installation})
	}
	removed.Aggregates = []output.Field{{K: "changed", V: len(removed.Rows) > 0}, {K: "reclaimed", V: output.Bytes(freed)}}
	removed.Next = []string{"cozy package list --rental=" + rental}
	return emit(ctx, removed)
}
