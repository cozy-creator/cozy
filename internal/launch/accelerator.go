package launch

// NeedsAccelerator is one callable's own machine class, as its package declares it: an
// entrypoint with model slots is served on a GPU, one without runs deviceless, and a job runs
// on CPU only when it declares `accelerator=False`; an undeclared job wants a GPU. Nothing
// infers it from dependencies.
func (e *Entrypoint) NeedsAccelerator() bool {
	if e.Kind != "job" {
		return len(e.Models) > 0
	}
	return e.Accelerator == nil || *e.Accelerator
}

// RunsOn renders a job's declared machine class.
func (e *Entrypoint) RunsOn() string {
	switch {
	case e.Accelerator == nil:
		return "GPU, else CPU on a machine without one (undeclared)"
	case *e.Accelerator:
		return "GPU"
	}
	return "CPU"
}
