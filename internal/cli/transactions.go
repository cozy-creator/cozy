package cli

import "github.com/cozy-creator/cozy/internal/exit"

func handleRunPause(ctx *Context) *exit.Error {
	return controlJob(ctx, "pause")
}

func handleRunResume(ctx *Context) *exit.Error {
	return controlJob(ctx, "resume")
}

// controlJob pauses or resumes a job, through its recorded explicit endpoint when it has one.
func controlJob(ctx *Context, action string) *exit.Error {
	close, problem := endpointForRecordedRun(ctx, ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	if close != nil {
		defer close()
	}
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	control := client.PauseJob
	if action == "resume" {
		control = client.ResumeJob
	}
	state, problem := control(ctx.Inv.Args[0], "cozy run "+action)
	if problem != nil {
		return problem
	}
	return renderSubmittedJob(ctx, state, true)
}
