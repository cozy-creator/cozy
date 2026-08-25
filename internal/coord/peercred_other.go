//go:build !linux && !darwin

package coord

// No kernel-attested peer identity on this platform. peerPID already documents 0 as "the
// platform did not answer", and every caller reads it that way — the adoption guard the
// pid backs is a Linux/macOS property, and claiming one here would be worse than none.
func peerCred(uintptr) (peerIdentity, bool) { return peerIdentity{}, false }
