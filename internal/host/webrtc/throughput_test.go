package webrtc_test

import (
	"crypto/rand"
	"fmt"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/host/webrtc/webrtctest"
)

// BenchmarkGet reads a finished 64 MiB output over loopback and through a delay relay, with a
// browser-like credit window of 8 MiB beyond what the client has received.
//
//	go test -run '^$' -bench Get -benchtime 3x ./internal/host/webrtc/
func BenchmarkGet(b *testing.B) {
	const size, window = 64 << 20, 8 << 20
	film := make([]byte, size)
	rand.Read(film)
	for _, rtt := range []time.Duration{0, 50 * time.Millisecond, 100 * time.Millisecond, 200 * time.Millisecond} {
		for _, rwnd := range []uint32{0, 8 << 20} {
			b.Run(fmt.Sprintf("rtt=%s/client-rwnd=%d", rtt, rwnd), func(b *testing.B) {
				h := newHarness(b)
				h.m.Append(7, "film", -1, film, 1_000_000)
				h.m.End(7, "completed")
				addr := h.srv.Addr
				if rtt > 0 {
					addr = newRelay(b, addr, rtt/2).addr()
				}
				c, err := webrtctest.Dial(h.ctx, addr, h.srv.Fingerprint, webrtctest.Options{ReceiveBuffer: rwnd})
				if err != nil {
					b.Fatal(err)
				}
				defer c.Close()
				h.send(c, map[string]any{"t": "hello", "v": 1, "cap": h.grant(nil)})
				h.expect(c, "welcome")
				b.SetBytes(size)
				cpu := cpuTime()
				b.ResetTimer()
				var granted uint64
				for i := range b.N {
					h.send(c, map[string]any{"t": "get", "id": i, "run": "7", "output": "film"})
					for received := uint64(0); ; {
						if want := uint64(i)*size + received + window; want >= granted+window/8 {
							granted = want
							h.send(c, map[string]any{"t": "credit", "bytes": granted})
						}
						m, _ := c.Recv(h.ctx)
						if m.T == "end" {
							break
						}
						received += uint64(len(m.Data))
					}
				}
				b.StopTimer()
				b.ReportMetric((cpuTime()-cpu).Seconds()*1000/float64(b.N*size>>20), "cpu-ms/MB")
			})
		}
	}
}

// cpuTime is this process's user and system time: the machine's and the client's together.
func cpuTime() time.Duration {
	var u syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &u)
	return time.Duration(u.Utime.Nano() + u.Stime.Nano())
}
