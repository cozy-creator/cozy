//go:build !windows

package coord

// bootstrapRequired: the unix transport carries SO_PEERCRED, so the kernel already
// attests who dialled — no handed credential is needed.
//
// What used to be here — `listenLocal`, `cleanupLocal` and the SO_PEERCRED credentials —
// died with the dial-direction flip (#436): the WORKER hosts WorkerControl and this side
// dials out, so there is no socket for a kernel to attest a peer on. Authority is the
// per-spawn bootstrap credential presented as `Claim.proof` (#463), on every platform.
const bootstrapRequired = false
