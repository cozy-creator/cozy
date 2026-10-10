package machinev1

import (
	"context"
	"errors"
	"io"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// A big output is read the way an S3 client reads an object: cut into ranges on one queue,
// Lanes connections each reading one range at a time into its place in the file. A range that
// fails, or moves bytes far slower than its peers, is dropped and its rest asked again on a
// fresh connection; a lane with nothing left to ask takes the far half of the longest range
// still in flight. One TCP connection on a long path carries a fraction of the link (its
// sender's buffer bounds it, and each lost packet halves its window); several side by side
// fill it.
const (
	Lanes = 8
	// A range's bytes: an output's share of the lanes, at least rangeLeast (a smaller output
	// is one read) and at most rangeMost.
	rangeLeast = 1 << 20
	rangeMost  = 8 << 20
	// A range whose bytes come this many times slower than its peers' is asked again.
	slowFactor = 8
	// How often ranges in flight are judged. A range's bytes are judged over a ramp: two
	// samples, or rampWaits round trips when that is longer, since a connection's first round
	// trips carry little however good the link.
	sample    = time.Second
	rampWaits = 16
)

// Whether a machine ends a read at the range asked for: unknown until its first answer. One
// that predates ranges sends to the output's end, so each lane is asked for its whole share
// once and its read is cut here.
const (
	rangesUnknown int32 = iota
	rangesRead
	rangesIgnored
)

type span struct{ first, end int64 }

// ReadRanges writes the target's bytes [from, length) at their offsets in w and tells held how
// many bytes from offset 0 are in hand as they land. It returns how far from `from` the bytes
// are whole in order: what a cut read keeps.
func (c *Client) ReadRanges(ctx context.Context, target *pb.OutputTarget, rev uint64, from, length int64, w io.WriterAt, held func(int64)) (int64, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	share := max((length-from+Lanes-1)/Lanes, rangeLeast)
	r := &rangeRead{c: c, ctx: ctx, target: target, rev: rev, from: from, length: length, w: w, held: held,
		share: share, landed: []span{{from, from}}}
	r.wake = sync.NewCond(&r.mu)
	switch c.ranges.Load() {
	case rangesRead:
		r.queue = cut(from, length, min(share, rangeMost))
	case rangesIgnored:
		r.queue = cut(from, length, share)
	default:
		// The first range goes alone: its answer says how the rest is asked.
		r.queue, r.lead = []span{{from, min(from+min(share, rangeMost), length)}}, true
	}
	var lanes sync.WaitGroup
	for lane := range Lanes {
		lanes.Go(func() { r.work(lane) })
	}
	over, watched := make(chan struct{}), make(chan struct{})
	go func() { defer close(watched); r.watch(over) }()
	lanes.Wait()
	close(over)
	<-watched
	whole := r.landed[0].end - from
	switch {
	case r.failure != nil:
		return whole, r.failure
	case whole < length-from && ctx.Err() != nil:
		return whole, ctx.Err()
	case whole < length-from:
		return whole, io.ErrUnexpectedEOF
	}
	return whole, nil
}

type rangeRead struct {
	c       *Client
	ctx     context.Context
	target  *pb.OutputTarget
	rev     uint64
	from    int64
	length  int64
	share   int64 // a lane's share of the bytes
	w       io.WriterAt
	held    func(int64)
	moved   atomic.Int64
	reports sync.Mutex

	mu      sync.Mutex
	wake    *sync.Cond
	lead    bool // one range is out and the rest waits for its answer
	queue   []span
	flights []*flight
	landed  []span // merged, in order; landed[0] starts at `from`
	waits   []float64
	rates   []float64
	failed  int   // reads failed since bytes last landed
	since   int64 // r.moved when that count began
	failure error
}

// flight is one range being read. at and end move under rangeRead.mu: the reader claims bytes
// before it writes them, and a lane with nothing to ask takes the far half by moving end.
type flight struct {
	lane    int
	first   int64
	at, end int64
	began   time.Time
	byte1   time.Time // its first byte
	judged  time.Time
	seen    int64
	dropped bool
	cancel  context.CancelFunc
}

// cut splits [first, end) into ranges of size bytes.
func cut(first, end, size int64) []span {
	var spans []span
	for ; first < end; first += size {
		spans = append(spans, span{first, min(first+size, end)})
	}
	return spans
}

// next is a range to read: the queue's first, else the far half of the longest in flight. It
// waits while the lead is out or nothing is worth halving; false when the read is over.
func (r *rangeRead) next(lane int) (*flight, context.Context, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for {
		if r.failure != nil || r.ctx.Err() != nil {
			return nil, nil, false
		}
		if len(r.queue) == 0 && !r.lead {
			// The range that needs longest, when that is longer than a new ask's ramp.
			var longest *flight
			needs, now := r.ramp().Seconds(), time.Now()
			for _, f := range r.flights {
				if f.at == f.first || f.end-f.at < 2*rangeLeast {
					continue
				}
				rate := float64(f.at-f.first) / now.Sub(f.byte1).Seconds()
				if left := float64(f.end-f.at) / rate; left > needs {
					longest, needs = f, left
				}
			}
			if longest != nil {
				half := longest.at + (longest.end-longest.at)/2
				r.queue, longest.end = []span{{half, longest.end}}, half
			}
		}
		if len(r.queue) > 0 {
			s := r.queue[0]
			r.queue = r.queue[1:]
			ctx, cancel := context.WithCancel(r.ctx)
			f := &flight{lane: lane, first: s.first, at: s.first, end: s.end, began: time.Now(), cancel: cancel}
			r.flights = append(r.flights, f)
			return f, ctx, true
		}
		if len(r.flights) == 0 {
			return nil, nil, false
		}
		r.wake.Wait()
	}
}

func (r *rangeRead) work(lane int) {
	for {
		f, ctx, ok := r.next(lane)
		if !ok {
			return
		}
		err := r.read(ctx, f)
		f.cancel()
		r.settle(f, err)
	}
}

// read writes one range's bytes into place, up to wherever its end has moved.
func (r *rangeRead) read(ctx context.Context, f *flight) error {
	r.mu.Lock()
	first, end := f.first, f.end
	r.mu.Unlock()
	stream, err := r.c.lane(f.lane).Read(ctx, &pb.ReadRequest{Target: &pb.ReadRequest_Output{Output: r.target},
		Offset: uint64(first), Length: uint64(end - first), IfRev: r.rev})
	if err != nil {
		return err
	}
	meta, err := stream.Recv()
	if err != nil {
		return err
	}
	if int64(meta.Length) != r.length {
		return status.Errorf(codes.FailedPrecondition, "the output holds %d bytes, not the %d asked for", meta.Length, r.length)
	}
	r.answered(f, meta.End != 0)
	for {
		r.mu.Lock()
		whole := f.at >= f.end
		r.mu.Unlock()
		if whole {
			return nil
		}
		frame, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return io.ErrUnexpectedEOF
		}
		if err != nil {
			return err
		}
		data := frame.GetData()
		r.mu.Lock()
		at := f.at
		data = data[:max(min(int64(len(data)), f.end-at), 0)]
		f.at += int64(len(data))
		if f.byte1.IsZero() && len(data) > 0 {
			f.byte1 = time.Now()
			r.waits = append(r.waits, f.byte1.Sub(f.began).Seconds())
		}
		r.mu.Unlock()
		if len(data) == 0 {
			continue
		}
		if _, err := r.w.WriteAt(data, at); err != nil {
			return &writeError{err}
		}
		r.report(r.moved.Add(int64(len(data))))
	}
}

// answered takes the lead's answer: a machine that reads ranges is asked for the rest in
// ranges; one that sends to the end is asked for each lane's share, the lead's included.
func (r *rangeRead) answered(f *flight, ranged bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.lead {
		return
	}
	r.lead = false
	size := min(r.share, rangeMost)
	switch {
	case ranged:
		r.c.ranges.Store(rangesRead)
	case f.end < r.length:
		r.c.ranges.Store(rangesIgnored)
		size, f.end = r.share, min(r.from+r.share, r.length)
	}
	r.queue = cut(f.end, r.length, size)
	r.wake.Broadcast()
}

func (r *rangeRead) report(moved int64) {
	if r.held == nil {
		return
	}
	r.reports.Lock()
	defer r.reports.Unlock()
	r.held(r.from + moved)
}

// settle records what a range landed and puts its rest back on the queue when it was dropped
// or failed for a reason asking again can change.
func (r *rangeRead) settle(f *flight, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	defer r.wake.Broadcast()
	r.flights = slices.DeleteFunc(r.flights, func(other *flight) bool { return other == f })
	r.land(span{f.first, f.at})
	if err == nil {
		if got := f.at - f.first; got >= rangeLeast {
			r.rates = append(r.rates, float64(got)/time.Since(f.byte1).Seconds())
		}
		return
	}
	if r.ctx.Err() != nil || r.failure != nil {
		return
	}
	if !f.dropped {
		if !retryable(err) {
			r.failure = err
			return
		}
		if moved := r.moved.Load(); moved > r.since {
			r.failed, r.since = 0, moved
		}
		// Every lane asked again since a byte last landed, and none moved one.
		if r.failed++; r.failed > Lanes {
			r.failure = err
			return
		}
	}
	r.c.redial(f.lane)
	r.queue = append([]span{{f.at, f.end}}, r.queue...)
}

// land merges a landed span into the list.
func (r *rangeRead) land(s span) {
	if s.end <= s.first {
		return
	}
	r.landed = append(r.landed, s)
	sort.Slice(r.landed, func(i, j int) bool { return r.landed[i].first < r.landed[j].first })
	merged := r.landed[:1]
	for _, next := range r.landed[1:] {
		if last := &merged[len(merged)-1]; next.first <= last.end {
			last.end = max(last.end, next.end)
		} else {
			merged = append(merged, next)
		}
	}
	r.landed = merged
}

// watch judges every range in flight once a sample: one that has waited far longer for its
// first byte than its peers did, or whose bytes come far slower than theirs, is dropped.
func (r *rangeRead) watch(over <-chan struct{}) {
	tick := time.NewTicker(sample)
	defer tick.Stop()
	for {
		select {
		case <-over:
			return
		case <-tick.C:
		}
		r.mu.Lock()
		now := time.Now()
		wait, ramp := median(r.waits), r.ramp()
		rates := slices.Clone(r.rates)
		for _, f := range r.flights {
			if !f.byte1.IsZero() && now.Sub(f.byte1) >= ramp {
				rates = append(rates, float64(f.at-f.first)/now.Sub(f.byte1).Seconds())
			}
		}
		typical := median(rates)
		for _, f := range r.flights {
			slow := false
			switch {
			case f.dropped:
			case f.byte1.IsZero():
				slow = wait > 0 && now.Sub(f.began).Seconds() > max(wait*slowFactor, (2*sample).Seconds())
			default:
				if f.judged.IsZero() {
					f.judged, f.seen = f.byte1, f.first
				}
				if window := now.Sub(f.judged); window >= ramp {
					rate := float64(f.at-f.seen) / window.Seconds()
					f.judged, f.seen = now, f.at
					slow = f.at < f.end && rate*slowFactor < typical
				}
			}
			if slow {
				f.dropped = true
				f.cancel()
			}
		}
		r.mu.Unlock()
	}
}

// ramp is how long a new ask takes to carry what its link can: rampWaits of the shortest
// first-byte wait seen (a round trip), and never under two samples.
func (r *rangeRead) ramp() time.Duration {
	if len(r.waits) == 0 {
		return 2 * sample
	}
	return max(2*sample, time.Duration(rampWaits*slices.Min(r.waits)*float64(time.Second)))
}

// median is what peers typically measured; 0 until half the lanes have a measurement.
func median(values []float64) float64 {
	if len(values) == 0 || len(values) < Lanes/2 {
		return 0
	}
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	return sorted[len(sorted)/2]
}

// writeError is the destination refusing bytes: asking the machine again cannot change it.
type writeError struct{ err error }

func (e *writeError) Error() string { return e.err.Error() }
func (e *writeError) Unwrap() error { return e.err }

// retryable is a read that may land on a fresh connection: the connection's failure, not the
// machine's answer.
func retryable(err error) bool {
	if refused := (*writeError)(nil); errors.As(err, &refused) {
		return false
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	switch status.Code(err) {
	case codes.Unavailable, codes.Canceled, codes.DeadlineExceeded, codes.Internal, codes.Unknown, codes.ResourceExhausted, codes.Aborted:
		return true
	}
	return false
}

// lane is the connection lane i reads on: the client's own for lane 0, else one of its own,
// dialed when first used and again after redial drops it.
func (c *Client) lane(i int) pb.MachineClient {
	if i == 0 || c.dial == nil {
		return c.Machine
	}
	c.lanesMu.Lock()
	defer c.lanesMu.Unlock()
	if c.lanes == nil {
		c.lanes = make([]*grpc.ClientConn, Lanes)
	}
	if c.lanes[i] == nil {
		conn, err := c.dial()
		if err != nil {
			return c.Machine
		}
		c.lanes[i] = conn
	}
	return pb.NewMachineClient(c.lanes[i])
}

// redial drops lane i's connection so its next range goes on a fresh one. Lane 0's is the
// client's own, which other calls ride, and is kept.
func (c *Client) redial(i int) {
	c.lanesMu.Lock()
	defer c.lanesMu.Unlock()
	if i > 0 && i < len(c.lanes) && c.lanes[i] != nil {
		_ = c.lanes[i].Close()
		c.lanes[i] = nil
	}
}
