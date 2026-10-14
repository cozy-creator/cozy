package launch

import "github.com/cozy-creator/cozy/internal/exit"

// NeedsAccelerator is one callable's own machine class, as its package declares it: an
// entrypoint with model slots is served on a GPU, one without runs deviceless, and a job
// runs where its required `accelerator` says. Nothing infers it from dependencies, so an
// undeclared job refuses before anything is rented or downloaded.
func (e *Entrypoint) NeedsAccelerator(pkg string) (bool, *exit.Error) {
	if e.Kind != "job" {
		return len(e.Models) > 0, nil
	}
	if e.Accelerator == nil {
		return false, exit.Named(exit.Validation, "package.job_accelerator_undeclared",
			"%s job %s declares no accelerator", pkg, e.Name).
			WithRemedy("republish it with app.job(..., accelerator=True) if its body computes on a GPU, accelerator=False if on CPU")
	}
	return *e.Accelerator, nil
}

// RunsOn renders a job's declared machine class.
func (e *Entrypoint) RunsOn() string {
	switch {
	case e.Accelerator == nil:
		return "undeclared (refused until the package declares app.job(accelerator=...))"
	case *e.Accelerator:
		return "GPU"
	}
	return "CPU"
}
