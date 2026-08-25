// Package coord is the LocalCoordinator: the scheduling role of the ONE long-lived
// LocalService `cozy up` starts (cl-001). It is the worker protocol's SERVER — the
// cozy-runtime supervisor dials IT, per the dial-toward-a-stable-address law — and the
// authority for everything the runtime deliberately is not: which request runs, which
// attempt ordinal exists, which devices a process may see, and which terminal/result
// becomes visible.
//
// What it does NOT do, structurally rather than by policy: it never executes a model,
// never chooses a placement or a plan, never authorizes a reader from inside the
// runtime, and never mints a credential. A local grant is a CAS root plus an output
// directory — there is no token field set anywhere in this package, which is what makes
// "no fake cloud tokens" a fact a reader can check rather than a promise.
//
// The one law this package is written around (worker-protocol/02 §6.2): the
// `recovered_attempts` a new session reports on Register are OPEN OBLIGATIONS, and no
// next ordinal for those request ids may be minted until each is closed by its own
// journaled terminal. A new session_id never manufactures absence.
package coord

import (
	"fmt"
	"io"
	"net"
	"os"
	"sync"
	"time"

	"google.golang.org/grpc"

	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// Options is the frozen input to one LocalService. Every field is decided by the
// entrypoint; nothing in this package reads the environment.
type Options struct {
	Cfg    config.Config
	Layout home.Layout
	Store  *records.Store
	// Socket is the worker-protocol unix socket the supervisor dials. A stable address
	// the worker dials TOWARD; the coordinator never dials a worker.
	Socket string
	// Yield is the GPU yield policy: smart | always | never.
	Yield string
	Log   io.Writer
}

// Coordinator is the LocalService's scheduling role.
type Coordinator struct {
	opt Options

	grpc     *grpc.Server
	listener net.Listener

	mu       sync.Mutex
	sessions map[string]*session // by session_id
	workers  map[string]*worker  // by instance_id
	waits    map[string]*wait    // by request#attempt
	revision uint64              // hub-owned, monotonic; every Directive bumps it
	events   []string
}

type wait struct {
	accepted chan struct{}
	closed   chan struct{}
	once     sync.Once
	onceA    sync.Once
	err      *exit.Error
}

// Open builds the coordinator and binds its unix socket. Binding is where a second
// LocalService on one root would fail, and it happens before any worker exists.
func Open(opt Options) (*Coordinator, *exit.Error) {
	if opt.Log == nil {
		opt.Log = io.Discard
	}
	if opt.Yield == "" {
		opt.Yield = "smart"
	}
	c := &Coordinator{
		opt:      opt,
		sessions: map[string]*session{},
		workers:  map[string]*worker{},
		waits:    map[string]*wait{},
	}
	// A stale socket file is a leftover, never evidence: the service lock already proved
	// no live owner exists on this root, so removing it is safe and required.
	_ = os.Remove(opt.Socket)
	ln, err := net.Listen("unix", opt.Socket)
	if err != nil {
		return nil, exit.New(exit.Conflict, "cannot bind the worker socket %s: %s", opt.Socket, err).
			WithRemedy("another LocalService may own this root; `cozy status`")
	}
	c.listener = ln
	c.grpc = grpc.NewServer()
	pb.RegisterWorkerServer(c.grpc, &hub{c: c})
	return c, nil
}

// Serve runs until Close. The coordinator is the SERVER; it never dials a worker.
func (c *Coordinator) Serve() error { return c.grpc.Serve(c.listener) }

// Close stops every worker this service owns, then the server. A worker outliving its
// launcher is exactly the class of bug the birth identity exists to catch, so `down`
// stops what it started rather than orphaning it.
func (c *Coordinator) Close(grace time.Duration) {
	for _, w := range c.workerList() {
		c.StopWorker(w.instanceID, grace)
	}
	c.grpc.Stop()
	_ = os.Remove(c.opt.Socket)
}

func (c *Coordinator) workerList() []*worker {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]*worker, 0, len(c.workers))
	for _, w := range c.workers {
		out = append(out, w)
	}
	return out
}

func (c *Coordinator) logf(format string, args ...any) {
	line := fmt.Sprintf("[coord %s] ", time.Now().UTC().Format("15:04:05.000")) +
		fmt.Sprintf(format, args...)
	fmt.Fprintln(c.opt.Log, line)
	c.mu.Lock()
	c.events = append(c.events, line)
	if len(c.events) > 512 {
		c.events = c.events[len(c.events)-512:]
	}
	c.mu.Unlock()
}

// Events is the coordinator's own recent activity, for status rendering. Runtime events
// reach a client through here and never bypass the coordinator.
func (c *Coordinator) Events() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.events...)
}

// nextRevision mints the Directive revision. The hub owns it; it is monotonic, and a
// changed body always carries a new one.
func (c *Coordinator) nextRevision() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.revision++
	return c.revision
}

func key(requestID string, attempt uint64) string {
	return fmt.Sprintf("%s#%d", requestID, attempt)
}

func (c *Coordinator) waitFor(k string) *wait {
	c.mu.Lock()
	defer c.mu.Unlock()
	w, ok := c.waits[k]
	if !ok {
		w = &wait{accepted: make(chan struct{}), closed: make(chan struct{})}
		c.waits[k] = w
	}
	return w
}

func (w *wait) markAccepted() { w.onceA.Do(func() { close(w.accepted) }) }
func (w *wait) markClosed(e *exit.Error) {
	w.once.Do(func() {
		w.err = e
		close(w.closed)
	})
}
