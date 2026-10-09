package records

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"time"

	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/protobuf/proto"
)

// Observing a v1 run never waits on fsync. The machine's journal is the authority for the
// run's log, and every transaction here advances the run's cursor with the rows it writes, so
// a crash loses both together and the machine sends those entries again on reattach.
// Progress samples and log lines are batched in memory and written together by one writer; a
// state or a product is written at once with the run's pending rows ahead of it, so the log
// keeps its order. All of these commit with synchronous=NORMAL under WAL. What the machine
// cannot replay (the request, its acceptance, its outcome and collection) commits FULL, and
// that sync covers the unsynced rows before it.
type telemetryV1 struct {
	write   sync.Mutex // held by whoever writes pending rows, from taking them to dropping them
	mu      sync.Mutex
	pending map[string][]pendingV1
	bytes   int
	wake    chan struct{}
	stop    chan struct{}
	done    chan struct{}
	closed  bool
}

type pendingV1 struct {
	event *v1.RunEvent
	at    string
	size  int
}

// telemetryBound is the pending bytes past which an observer writes the batch itself.
const telemetryBound = 8 << 20

// telemetryTypes are the AppendEvent kinds written without sync.
var telemetryTypes = map[string]bool{"machine.progress": true, "request.preparing": true, "request.log": true}

func telemetryEventV1(event *v1.RunEvent) bool {
	return event.GetProgress() != nil || event.GetLog() != nil
}

// queueTelemetryV1 holds one progress or log entry of a run for the next batch.
func (s *Store) queueTelemetryV1(id string, event *v1.RunEvent) {
	entry := pendingV1{event: event, size: proto.Size(event)}
	if event.AtMs == 0 {
		entry.at = now()
	}
	t := &s.telemetry
	t.mu.Lock()
	if t.pending == nil {
		t.pending = map[string][]pendingV1{}
	}
	t.pending[id] = append(t.pending[id], entry)
	t.bytes += entry.size
	inline := t.bytes > telemetryBound || t.closed
	if t.stop == nil && !t.closed {
		t.wake, t.stop, t.done = make(chan struct{}, 1), make(chan struct{}), make(chan struct{})
		go s.writeTelemetry(t.wake, t.stop, t.done)
	}
	wake := t.wake
	t.mu.Unlock()
	if inline {
		_ = s.flushTelemetry()
		return
	}
	select {
	case wake <- struct{}{}:
	default:
	}
}

// writeTelemetry writes whatever is pending each time it is woken; entries that arrive
// while a batch is being written go in the next one.
func (s *Store) writeTelemetry(wake, stop, done chan struct{}) {
	defer close(done)
	for {
		select {
		case <-stop:
			return
		case <-wake:
			_ = s.flushTelemetry() // a failed batch stays pending for the next one or the run's next transition
		}
	}
}

// stopTelemetry ends the writer and writes what is left.
func (s *Store) stopTelemetry() {
	t := &s.telemetry
	t.mu.Lock()
	stop, done := t.stop, t.done
	t.stop, t.closed = nil, true
	t.mu.Unlock()
	if stop != nil {
		close(stop)
		<-done
	}
	_ = s.flushTelemetry()
}

// flushTelemetry writes every run's pending rows in one unsynced transaction.
func (s *Store) flushTelemetry() error {
	s.telemetry.write.Lock()
	defer s.telemetry.write.Unlock()
	s.telemetry.mu.Lock()
	ids := make([]string, 0, len(s.telemetry.pending))
	for id := range s.telemetry.pending {
		ids = append(ids, id)
	}
	s.telemetry.mu.Unlock()
	if len(ids) == 0 {
		return nil
	}
	taken := map[string]int{}
	err := s.unsynced(func(tx *sql.Tx) error {
		for _, id := range ids {
			n, err := s.telemetryTx(tx, id)
			if err != nil {
				return err
			}
			taken[id] = n
		}
		return nil
	})
	if err == nil {
		for id, n := range taken {
			s.dropTelemetry(id, n)
		}
	}
	return err
}

// telemetryTx writes the run's pending rows past its recorded cursor and moves the cursor
// past them. The caller holds telemetry.write and, once tx commits, drops the n entries.
func (s *Store) telemetryTx(tx *sql.Tx, id string) (n int, err error) {
	s.telemetry.mu.Lock()
	entries := append([]pendingV1(nil), s.telemetry.pending[id]...)
	s.telemetry.mu.Unlock()
	if len(entries) == 0 {
		return 0, nil
	}
	var cursor, ordinal int64
	var current string
	err = tx.QueryRow(`SELECT e.remote_cursor, r.ordinal, r.state FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE e.request_id=?`, id).Scan(&cursor, &ordinal, &current)
	if err == sql.ErrNoRows {
		return len(entries), nil // the run is gone; so are its rows
	}
	if err != nil {
		return 0, err
	}
	moved := false
	for _, entry := range entries {
		if entry.event.Sequence != 0 && int64(entry.event.Sequence) <= cursor {
			continue // written before, or sent again by a reattach
		}
		kind, payload := telemetryRowV1(entry.event, current)
		raw, err := json.Marshal(payload)
		if err != nil {
			return 0, err
		}
		at := entry.at
		if at == "" {
			at = time.UnixMilli(entry.event.AtMs).UTC().Format(time.RFC3339Nano)
		}
		if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`, id, kind, max(ordinal, 1), string(raw), at); err != nil {
			return 0, err
		}
		if entry.event.Sequence != 0 {
			cursor, moved = int64(entry.event.Sequence), true
		}
	}
	if moved {
		if _, err := tx.Exec(`UPDATE machine_executions SET remote_cursor=? WHERE request_id=?`, cursor, id); err != nil {
			return 0, err
		}
	}
	return len(entries), nil
}

// dropTelemetry forgets the first n pending entries of the run: the ones a committed
// transaction took. Entries are only appended behind them meanwhile.
func (s *Store) dropTelemetry(id string, n int) {
	if n == 0 {
		return
	}
	t := &s.telemetry
	t.mu.Lock()
	defer t.mu.Unlock()
	entries := t.pending[id]
	for _, entry := range entries[:n] {
		t.bytes -= entry.size
	}
	if len(entries) == n {
		delete(t.pending, id)
		return
	}
	t.pending[id] = entries[n:]
}

// telemetryRowV1 is the request event an entry becomes, in the shapes a worker.v1 import
// writes: progress as a `machine.progress` sample (a queued run's as `request.preparing`),
// a log line as `request.log`.
func telemetryRowV1(event *v1.RunEvent, current string) (string, map[string]any) {
	if log := event.GetLog(); log != nil {
		return "request.log", map[string]any{"level": log.Level, "message": log.Text}
	}
	p := event.GetProgress()
	if current == "queued" {
		detail := p.Stage
		if p.BytesTotal > 0 {
			detail = strings.TrimSpace(detail + " " + humanBytes(p.BytesDone) + " of " + humanBytes(p.BytesTotal))
		} else if p.BytesDone > 0 {
			detail = strings.TrimSpace(detail + " · " + humanBytes(p.BytesDone) + " read")
		}
		preparing := map[string]any{"stage": "machine", "step": p.Stage, "detail": detail}
		if p.BytesTotal > 0 || p.BytesDone > 0 {
			preparing["bytes_done"], preparing["bytes_total"] = p.BytesDone, p.BytesTotal
		}
		return "request.preparing", preparing
	}
	sample := map[string]any{"stage": p.Stage}
	if event.AtMs > 0 {
		// Keep source time distinguishable from the event row's receipt-time fallback.
		// Replayed/coalesced progress may arrive together and cannot be timed by arrival.
		sample["sample_unix_ms"] = event.AtMs
	}
	if p.Fraction >= 0 {
		sample["overall_fraction"] = p.Fraction
	}
	if p.StageFraction != nil {
		sample["stage_fraction"] = *p.StageFraction
	}
	if p.Total > 0 {
		sample["position"], sample["total"] = p.Completed, p.Total
	}
	if p.BytesTotal > 0 {
		sample["bytes_done"], sample["bytes_total"] = p.BytesDone, p.BytesTotal
	}
	if p.StepMs > 0 {
		sample["step_ms"] = p.StepMs
	}
	return "machine.progress", map[string]any{"type": "progress", "payload": sample}
}

// unsynced runs fn in one transaction whose commit does not wait on fsync; the next synced
// commit makes it durable along with its own. The connection's level is restored after, and
// a connection that cannot be restored is discarded rather than handed on.
func (s *Store) unsynced(fn func(*sql.Tx) error) error {
	ctx := context.Background()
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return err
	}
	defer conn.Close()
	var level int
	if err := conn.QueryRowContext(ctx, `PRAGMA synchronous`).Scan(&level); err != nil {
		return err
	}
	if _, err := conn.ExecContext(ctx, `PRAGMA synchronous=NORMAL`); err != nil {
		return err
	}
	defer func() {
		if _, err := conn.ExecContext(ctx, `PRAGMA synchronous=`+strconv.Itoa(level)); err != nil {
			_ = conn.Raw(func(any) error { return driver.ErrBadConn })
		}
	}()
	tx, err := conn.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit()
}
