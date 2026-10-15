package cli

import (
	"slices"
	"strings"

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
	if ctx.endpoint != nil || ctx.Inv.Bool("--rent-new") || ctx.Inv.Bool("--rental-only") || ctx.Inv.Bool("--rental") {
		return exit.Usagef("--warm names its machine: this computer's, or --rental=<name>")
	}
	op := records.Operation{Package: target.Package, Release: target.Release, Entrypoint: target.Function, Warm: level}
	if level != "off" {
		_, overrides, problem := launch.ParsePayload(ep, ctx.Inv.Args[1:], ctx.Inv.Value("--in"))
		if problem != nil {
			return problem
		}
		models, _, problem := modelChoices(ctx, target, ep, overrides.Models)
		if problem != nil {
			return problem
		}
		// The member's every slot is resolved here (th-241): the machine reads no binding.
		if models, problem = exactRunModels(ctx, target, ep, overrides.Models, models, resolveInvocationModels); problem != nil {
			return problem
		}
		if op.Models, problem = applyModelAdapters(ctx, target, ep, models, overrides.Overlays); problem != nil {
			return problem
		}
	}
	if _, _, problem := ensureCalcifer(ctx); problem != nil {
		return problem
	}
	machine := either(ctx.Inv.Value("--rental"), machines.Local)
	if target.InstallID != "" && level != "off" {
		// Local code is installed on the machine first, as `cozy package install <dir>` does;
		// the warm set then names that installation.
		install := records.Operation{Package: target.Package, Release: target.Release, Hub: ctx.Cfg.HubURL, Local: target.InstallID}
		if _, problem := settleOperation(ctx, machine, install); problem != nil {
			return problem
		}
	}
	return runOperation(ctx, machine, op, true)
}
