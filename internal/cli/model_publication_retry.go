package cli

import "github.com/cozy-creator/cozy/internal/exit"

func handleRunRetryPublication(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	state, problem := client.RetryJobPublication(ctx.Inv.Args[0], "cozy run retry-publication")
	if problem != nil {
		return problem
	}
	return emit(ctx, compactRecord(jobFields(ctx.Mode(), state, true), "job", "status"))
}
