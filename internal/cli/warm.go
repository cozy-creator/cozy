package cli

import (
	"cmp"
	"slices"
	"strings"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
)

// handleWarm is `cozy run <target> --warm=<level>`: instead of running the function, the
// machine keeps it ready up to that level, a member of its warm set; `off` takes it out. Its
// `model.<param>=` choices and --lora are the member's models.
func handleWarm(ctx *Context, target Target, ep *launch.Entrypoint, level string) *exit.Error {
	if level != "off" && !slices.Contains(records.WarmLevels, level) {
		return exit.Usagef("--warm=%s is not one of %s or off", level, strings.Join(records.WarmLevels, ", "))
	}
	if ep.Kind == "job" {
		return exit.Usagef("--warm keeps a serving function ready; %s is a job", target.Function)
	}
	if strings.HasPrefix(target.Package, "local/") || target.Snapshot {
		return exit.Usagef("--warm keeps a published package release ready; %s is local code", target.Package)
	}
	if ctx.endpoint != nil || ctx.Inv.Bool("--rent-new") || ctx.Inv.Bool("--rental-only") || ctx.Inv.Bool("--rental") {
		return exit.Usagef("--warm names its machine: this computer's, or --rental=<name>")
	}
	selection := records.RentalInstallSelection{Package: target.Package, Release: target.Release,
		Entrypoint: target.Function, Warm: level}
	if level != "off" {
		_, overrides, problem := launch.ParsePayload(ep, ctx.Inv.Args[1:], ctx.Inv.Value("--in"))
		if problem != nil {
			return problem
		}
		models, _, problem := modelChoices(ctx, target, ep, overrides.Models)
		if problem != nil {
			return problem
		}
		if selection.Models, problem = applyModelAdapters(ctx, target, ep, models, overrides.Overlays); problem != nil {
			return problem
		}
	}
	machine := either(ctx.Inv.Value("--rental"), machines.Local)
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	client, problem := localapi.Open(ctx.Cfg, state)
	if problem != nil {
		return problem
	}
	if caps, problem := client.Capabilities(); problem != nil || caps.WarmSet {
		return cmp.Or(problem, enqueueRentalInstall(ctx, machine, selection, true))
	}
	// A daemon from an older cozy would drop the member: this command sends it to the rental.
	if machine == machines.Local {
		return exit.Named(exit.Unavailable, "daemon.warm_set_unsupported", "the running cozy daemon predates warm sets").
			WithRemedy("run `cozy down`; the next command starts this cozy's daemon")
	}
	pinned, problem := rentalMachineEndpoint(ctx, machine)
	if problem != nil {
		return problem
	}
	if taken, problem := foregroundInstall(ctx, pinned, machine, selection); taken || problem != nil {
		return problem
	}
	return exit.Named(exit.Structural, "machine.warm_set_unsupported", "%s keeps no warm set; %s", machine, machines.RuntimeUpdate(machine))
}
