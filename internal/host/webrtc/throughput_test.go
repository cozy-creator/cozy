package webrtc_test

import (
	"crypto/rand"
	"fmt"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/host/webrtc/webrtctest"
)

// BenchmarkGet reads a finished 64 MiB output over loopback, through a delay relay, and
// through a bottleneck, with a browser-like credit window of 8 MiB beyond what the client has
// received. queue-ms is the longest any byte waited at the bottleneck: the machine's window
// must keep it far below SCTP's retransmission timeout (1 s at least).
//
//	go test -run '^$' -bench Get -benchtime 3x ./internal/host/webrtc/
func BenchmarkGet(b *testing.B) {
	const size, window = 64 << 20, 8 << 20
	film := make([]byte, size)
	rand.Read(film)
	for _, path := range []struct {
		rtt  time.Duration
		rate float64 // bytes/s at the bottleneck; 0: none
		rwnd uint32  // the client's SCTP receive window; 0: pion's default
	}{
		{0, 0, 0}, {50 * time.Millisecond, 0, 0}, {50 * time.Millisecond, 0, 8 << 20},
		{100 * time.Millisecond, 0, 0}, {100 * time.Millisecond, 0, 8 << 20},
		{200 * time.Millisecond, 0, 0}, {200 * time.Millisecond, 0, 8 << 20},
		{100 * time.Millisecond, 4e6, 8 << 20}, {150 * time.Millisecond, 1.5e6, 8 << 20},
	} {
		b.Run(fmt.Sprintf("rtt=%s/link=%gMBps/client-rwnd=%d", path.rtt, path.rate/1e6, path.rwnd), func(b *testing.B) {
			h := newHarness(b)
			h.m.Append(7, "film", -1, film, 1_000_000)
			h.m.End(7, "completed")
			var link *relay
			addr := h.srv.Addr
			if path.rtt > 0 {
				link = newLink(b, addr, path.rtt/2, path.rate)
				addr = link.addr()
			}
			c, err := webrtctest.Dial(h.ctx, addr, h.srv.Fingerprint, webrtctest.Options{ReceiveBuffer: path.rwnd})
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
			if link != nil && path.rate > 0 {
				b.ReportMetric(float64(link.maxQueue.Milliseconds()), "queue-ms")
			}
		})
	}
}

// cpuTime is this process's user and system time: the machine's and the client's together.
func cpuTime() time.Duration {
	var u syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &u)
	return time.Duration(u.Utime.Nano() + u.Stime.Nano())
}
