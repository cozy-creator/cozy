package transfer

import (
	"context"
	"io"
	"sync/atomic"
	"time"
)

// stillSamples is how many consecutive samples may find a transfer's byte counter
// unmoved before it is called STALLED. A COUNT, not a duration — the same shape the
// runtime's silence detector uses, and what makes the decision clock-free: it is not
// "this took too long", it is "nothing arrived, repeatedly".
//
// Eight is the fleet's still-factor (xs-007 row 29): the figure `orchestrator`'s media
// stall budget and the runtime's own liveness rule spend before calling a meter dead. The
// point of a factor above one is that a single slow read, a GC pause, or a retry inside the
// transport cannot be mistaken for a stall — only silence that repeats can.
const stillSamples = 8

// movingSample is how often the counter is READ. A sampling cadence is the resolution
// of an observation, never a bound on one: a transfer moving one byte per sample is
// never stalled, however long it takes.
//
// The resolution has to be coarse enough that a healthy-but-slow transport still moves a
// byte between two reads. This codebase's figure for "an HTTP transfer that has moved
// nothing for this long is not moving" is the media plane's own per-operation silence
// budget (16s), so the counter is read just inside it: anything the transport itself still
// considers alive registers here as movement.
const movingSample = 15 * time.Second

// StallBudget is how long a meter may stay still before the work it watches is stalled.
const StallBudget = stillSamples * movingSample

// mover watches BYTES, so a transfer is bounded by whether it is working rather than
// by how long it has been working.
//
// It replaces a 30-minute wall clock on both the whole transfer and each object at the
// storage edge. That constant was self-described as "generous because it is bounded by
// bytes on a link, not by anyone's latency" — which is the admission that bytes were
// the right meter and a clock was standing in for one. A 400 GB checkpoint on a
// residential uplink exceeds it while making perfect progress, and got killed for it.
type mover struct {
	moved atomic.Int64
}

// context lives while bytes are moving. It ends when the parent ends, when the caller
// cancels, or when `stillSamples` consecutive reads find the counter where they left it.
func (m *mover) context(parent context.Context) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(parent)
	go func() {
		tick := time.NewTicker(movingSample)
		defer tick.Stop()
		still, seen := 0, int64(-1)
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				at := m.moved.Load()
				if at > seen {
					still, seen = 0, at
					continue
				}
				still++
				if still >= stillSamples {
					cancel()
					return
				}
			}
		}
	}()
	return ctx, cancel
}

// reader counts every byte that actually arrives. Wrapping the body is what makes the
// meter fine-grained: a per-object or per-part total only moves when a whole object or
// part has landed, so a single multi-gigabyte part would read as no progress at all.
func (m *mover) reader(r io.Reader) io.Reader { return &counted{r: r, m: m} }

func (m *mover) readCloser(r io.ReadCloser) io.ReadCloser {
	return &countedReadCloser{counted: counted{r: r, m: m}, closer: r}
}

type counted struct {
	r io.Reader
	m *mover
}

func (c *counted) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	if n > 0 {
		c.m.moved.Add(int64(n))
	}
	return n, err
}

type countedReadCloser struct {
	counted counted
	closer  io.Closer
}

func (c *countedReadCloser) Read(p []byte) (int, error) { return c.counted.Read(p) }
func (c *countedReadCloser) Close() error               { return c.closer.Close() }

// Progress bounds a multi-step operation by observed movement rather than a clock: its
// context ends only after `stillSamples` consecutive samples see no step and no byte.
type Progress struct{ m mover }

func (p *Progress) Context(parent context.Context) (context.Context, context.CancelFunc) {
	return p.m.context(parent)
}

// Advance records completed work: bytes received, or one finished step.
func (p *Progress) Advance(n int64) {
	if p != nil {
		p.m.moved.Add(n)
	}
}
