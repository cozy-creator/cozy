package main

// THE SUPERVISION HALF OF THE POD, AND THE ONLY PLACE IT LIVES.
//
// cl-036 merged the media plane into this process, so PID 1 now shares an address space
// with an HTTP parser. This file is the boundary that makes that acceptable: every
// privilege the supervisor holds and the request path must not — exec, signal, reap — is
// spelled HERE and nowhere else. `internal/podmedia` is fenced against all three, and this
// is package main, so a handler cannot reach these functions even by name. The pod's one
// child is the control runtime's adapter; the media plane runs in-process as a leg beside
// it, and either one ending is fatal to the pod, exactly as when they were two processes.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/podmedia"
)

const adapterPath = "/opt/cozy/bin/cozy-materialize-launch"

// leg is one thing whose death is the pod's death. There are two — the media plane
// serving in this process, and the adapter this process exec'd — and `supervise` cannot
// tell them apart on purpose: there are no restart semantics to lose, and a pod whose
// byte plane or whose worker launcher has stopped is a pod, not a degraded mode.
type leg struct {
	name string
	done chan struct{}
	err  error

	// terminate asks this leg to stop; kill ends one that would not. The media plane has
	// no kill — closing its listener ends Serve — and that asymmetry is the whole
	// difference between an in-process leg and an exec'd one.
	terminate func()
	kill      func()
}

func (l *leg) exited() bool {
	select {
	case <-l.done:
		return true
	default:
		return false
	}
}

// run performs the generic pod boot. It dials nothing: the whole launch surface is the
// injected environment, and the readiness payload the adapter drops is opaque bytes this
// process only authenticates (cr-048/th-067 retired the two provision documents).
func run(parent context.Context) error {
	cfg, err := parseConfig()
	if err != nil {
		return err
	}
	readinessCtx, cancelReadiness := context.WithDeadline(parent, cfg.leaseExpiry)
	defer cancelReadiness()
	if err := prepareState(cfg); err != nil {
		return err
	}
	// The plane BINDS before the adapter exists. A grant this pod cannot authenticate,
	// or a port it cannot take, is a boot failure with nothing started behind it — which
	// is strictly stronger than the two-binary shape, where the same failure arrived as
	// a child that had already been exec'd and then died.
	media, err := startMedia(cfg)
	if err != nil {
		return err
	}
	legs := []*leg{media}
	defer stopLegs(legs)
	adapter, err := startAdapter(cfg)
	if err != nil {
		return err
	}
	legs = append(legs, adapter)
	if err := publishReceipt(readinessCtx, cfg.receipt, legs); err != nil {
		return classifyContext(parent, readinessCtx, err)
	}
	cancelReadiness()
	return supervise(parent, legs)
}

// startMedia binds the pod's byte plane and serves it in this process. Its grant is the
// validated renter token DIGESTS, handed over IN MEMORY: they were an argv value while
// this was a second binary, and a merged pod has no argv to put them on.
func startMedia(cfg config) (*leg, error) {
	plane, err := podmedia.Bind(podmedia.Options{
		Listen:           net.JoinHostPort("0.0.0.0", strconv.Itoa(int(cfg.mediaPort))),
		Root:             mediaRoot,
		TokenHashes:      mediaGrant(cfg.tokenHashes),
		Cert:             certificatePath,
		Key:              privateKeyPath,
		BootstrapReceipt: receiptEnvelopePath,
	})
	if err != nil {
		return nil, fmt.Errorf("the pod's media plane refused to bind: %w", err)
	}
	l := &leg{name: "media plane", done: make(chan struct{}), terminate: func() { _ = plane.Close() }}
	go func() {
		l.err = plane.Serve()
		close(l.done)
	}()
	return l, nil
}

// mediaGrant spells the validated hash set the way the media plane takes it. These are
// DIGESTS, not credentials — the raw token exists only on the renter's host — and since
// cl-036 they never leave this address space, so no process list carries them either.
func mediaGrant(hashes []string) []string {
	lines := make([]string, len(hashes))
	for i, hash := range hashes {
		lines[i] = "sha256:" + hash
	}
	return lines
}

func startAdapter(cfg config) (*leg, error) {
	cmd := exec.Command(adapterPath)
	env := childEnvironment()
	hashes, _ := json.Marshal(cfg.tokenHashes)
	env = append(env,
		"COZY_ADAPTER_ACQUISITION_ATTEMPT_ID="+cfg.attemptID,
		"COZY_ADAPTER_POD_BOOT_ID_PATH="+podBootIDPath,
		"COZY_ADAPTER_TLS_CERT_PATH="+certificatePath,
		"COZY_ADAPTER_TLS_KEY_PATH="+privateKeyPath,
		"COZY_ADAPTER_RECEIPT_PAYLOAD_PATH="+receiptPayloadPath,
		"COZY_ADAPTER_WORKER_INTERNAL_PORT="+strconv.Itoa(int(cfg.workerPort)),
		"COZY_ADAPTER_MEDIA_INTERNAL_PORT="+strconv.Itoa(int(cfg.mediaPort)),
		"COZY_ADAPTER_RENTER_TOKEN_SHA256_JSON="+string(hashes),
		// The lease expiry, spent as the adapter's boot deadline: readiness has to be
		// proven inside the rental it is being proven for. Shortening THIS costs a boot;
		// shortening the lease expiry itself expires the serving certificate (state.go).
		"COZY_ADAPTER_READINESS_DEADLINE_UNIX="+strconv.FormatInt(cfg.leaseExpiry.Unix(), 10),
	)
	cmd.Env = env
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start materialize/launch adapter: %w", err)
	}
	l := &leg{
		name:      "materialize/launch adapter",
		done:      make(chan struct{}),
		terminate: func() { _ = cmd.Process.Signal(syscall.SIGTERM) },
		kill:      func() { _ = cmd.Process.Kill() },
	}
	go func() {
		l.err = cmd.Wait()
		close(l.done)
	}()
	return l, nil
}

// supervise is the pod's whole liveness rule: any leg ending ends the pod. It is a
// polling loop rather than a signal handler because PID 1 in this container has exactly
// two things to watch and reaping is `cmd.Wait`'s job.
func supervise(ctx context.Context, legs []*leg) error {
	for {
		for _, l := range legs {
			if l.exited() {
				return legExit(l)
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}

func legExit(l *leg) error {
	if l.err == nil {
		return fmt.Errorf("%s exited unexpectedly", l.name)
	}
	return fmt.Errorf("%s failed: %w", l.name, l.err)
}

func stopLegs(legs []*leg) {
	for _, l := range legs {
		if l != nil && l.terminate != nil {
			l.terminate()
		}
	}
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	for _, l := range legs {
		if l == nil {
			continue
		}
		select {
		case <-l.done:
		case <-deadline.C:
			for _, remaining := range legs {
				if remaining != nil && !remaining.exited() && remaining.kill != nil {
					remaining.kill()
				}
			}
			return
		}
	}
}

func classifyContext(parent, readiness context.Context, fallback error) error {
	if parent.Err() != nil {
		return parent.Err()
	}
	if errors.Is(readiness.Err(), context.DeadlineExceeded) {
		return errors.New("frozen readiness deadline elapsed")
	}
	return fallback
}
