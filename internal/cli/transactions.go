package cli

import "github.com/cozy-creator/cozy/internal/exit"

func handleRunPause(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	state, problem := client.PauseJob(ctx.Inv.Args[0], "cozy run pause")
	if problem != nil {
		return problem
	}
	return renderSubmittedJob(ctx, state, true)
}

func handleRunResume(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	state, problem := client.ResumeJob(ctx.Inv.Args[0], "cozy run resume")
	if problem != nil {
		return problem
	}
	return renderSubmittedJob(ctx, state, true)
}
