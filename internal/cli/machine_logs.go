package cli

import (
	"encoding/json"
	"fmt"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
)

// handleMachineLogs is `cozy machine logs --tensorfs|--memory`: a log this computer's machine
// keeps, read through the same machine call a rental answers.
func handleMachineLogs(ctx *Context) *exit.Error {
	name, problem := machineLogName(ctx)
	if problem != nil {
		return problem
	}
	return printMachineLog(ctx, machines.Local, name, "this computer's machine")
}

// machineLogName is the log a `--tensorfs` or `--memory` flag names.
func machineLogName(ctx *Context) (string, *exit.Error) {
	switch {
	case ctx.Inv.Bool("--tensorfs"):
		return "tensorfs", nil
	case ctx.Inv.Bool("--memory"):
		return "memory", nil
	}
	return "", exit.Usagef("name the log to print: --tensorfs or --memory")
}

// printMachineLog prints a log a machine keeps, oldest line first. A machine whose agent
// predates the read, or that has logged nothing there, prints a note and succeeds.
func printMachineLog(ctx *Context, machine, name, label string) *exit.Error {
	calcifer, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	log, problem := calcifer.MachineLog(machine, name, 0)
	if problem != nil {
		return problem
	}
	switch {
	case ctx.Mode().JSON:
		if err := json.NewEncoder(ctx.Out).Encode(log); err != nil {
			return exit.Internalf("cannot write the log: %s", err)
		}
	case log.Unavailable != "":
		fmt.Fprintf(ctx.Err, "%s: %s\n", label, log.Unavailable)
	case log.Text == "":
		fmt.Fprintf(ctx.Err, "%s has logged nothing in its %s log\n", label, name)
	default:
		fmt.Fprint(ctx.Out, log.Text)
	}
	return nil
}
