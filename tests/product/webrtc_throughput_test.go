package producttest

import "testing"

// The former throughput harness served numeric runs through the retired supervisor
// boundary. Until this benchmark creates a real machine-v1 output, a timing from it
// would not describe the shipped player path. The Pion and browser product tests
// exercise that path, but do not claim transport throughput qualification.
func BenchmarkWebRTCGet(b *testing.B) {
	b.Skip("requires a machine-v1 throughput fixture; retired numeric-run harness is unsupported")
}
