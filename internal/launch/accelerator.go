package launch

// NeedsAccelerator is one function's machine class, as its package declares it: CPU only when
// it declares `accelerator=False`, else the machine's GPU when the machine has one. Nothing
// infers it from model slots or dependencies.
func (e *Entrypoint) NeedsAccelerator() bool {
	return e.Accelerator == nil || *e.Accelerator
}
