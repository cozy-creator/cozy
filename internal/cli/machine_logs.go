package cli

import (
	"encoding/json"
	"fmt"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
)

// handleMachineLogs is `cozy machine logs --tensorfs`: the TensorFS transport decisions this
// computer's machine logged, read through the same machine call a rental answers.
func handleMachineLogs(ctx *Context) *exit.Error {
	if !ctx.Inv.Bool("--tensorfs") {
		return exit.Usagef("name the log to print: --tensorfs")
	}
	return printMachineLog(ctx, machines.Local, "this computer's machine")
}

// printMachineLog prints the TensorFS log a machine keeps, oldest line first. A machine whose
// agent predates the read, or whose TensorFS has logged nothing, prints a note and succeeds.
func printMachineLog(ctx *Context, machine, label string) *exit.Error {
	daemon, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	log, problem := daemon.MachineLog(machine, "tensorfs", 0)
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
		fmt.Fprintf(ctx.Err, "%s has logged no TensorFS transport decisions (TensorFS 0.3.82 and later log them)\n", label)
	default:
		fmt.Fprint(ctx.Out, log.Text)
	}
	return nil
}
