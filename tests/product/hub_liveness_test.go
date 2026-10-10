package producttest

import (
	"bytes"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// slowAnswer is longer than any clock a Hub call ever had.
const slowAnswer = 25 * time.Second

const searchAnswer = `{"packages":[{"org":"slow","name":"hub","created_at":"2026-10-10T00:00:00Z"}],` +
	`"search":{"total":1,"limit":20,"capped":false,"q":"slow"}}`

// livenessHome is an empty Creator home; addressHub points it at a Hub.
func livenessHome(t *testing.T, name string) string {
	t.Helper()
	must(t, os.MkdirAll(scratchBase, 0o755))
	root, err := os.MkdirTemp(scratchBase, name+"-")
	must(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	return root
}

// addressHub makes hubURL root's Tensorhub, with a liveness window of seconds.
func addressHub(t *testing.T, root, hubURL string, seconds int) {
	t.Helper()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(fmt.Sprintf(
		"tensorhub_url: %s\ntensorhub_liveness_s: %d\ndaemon:\n  idle_shutdown_s: 0\n", hubURL, seconds)), 0o600))
}

// tlsHub serves handler over HTTP/2 with TLS and returns the SSL_CERT_FILE setting that
// makes a cozy child trust it.
func tlsHub(t *testing.T, root string, handler http.Handler) (*httptest.Server, string) {
	t.Helper()
	server := httptest.NewUnstartedServer(handler)
	server.EnableHTTP2 = true
	server.StartTLS()
	t.Cleanup(server.Close)
	ca := filepath.Join(root, "hub-ca.pem")
	must(t, os.WriteFile(ca, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw}), 0o600))
	return server, "SSL_CERT_FILE=" + ca
}

type cozyRun struct {
	code           int
	stdout, stderr string
	took           time.Duration
}

// startCozy runs the real binary in the background, so several slow calls share one wait.
func startCozy(t *testing.T, root string, imposed []string, args ...string) <-chan cozyRun {
	t.Helper()
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root, imposed...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	began := time.Now()
	must(t, cmd.Start())
	done := make(chan cozyRun, 1)
	go func() {
		_ = cmd.Wait()
		done <- cozyRun{cmd.ProcessState.ExitCode(), stdout.String(), stderr.String(), time.Since(began)}
	}()
	return done
}

// A Hub that is alive but answers after 25 s is waited on and its answer used, over HTTP/1
// (a local Hub) and over HTTP/2, whose liveness PINGs it keeps answering meanwhile.
func TestSlowHubAnswerIsAwaited(t *testing.T) {
	slow := func(protocol *atomic.Int32) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			protocol.Store(int32(r.ProtoMajor))
			select {
			case <-time.After(slowAnswer):
			case <-r.Context().Done():
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, searchAnswer)
		})
	}
	var plainProtocol, tlsProtocol atomic.Int32
	plain := httptest.NewServer(slow(&plainProtocol))
	t.Cleanup(plain.Close)
	plainRoot := livenessHome(t, "hub-slow-http1")
	addressHub(t, plainRoot, plain.URL, 2)
	tlsRoot := livenessHome(t, "hub-slow-http2")
	secure, trust := tlsHub(t, tlsRoot, slow(&tlsProtocol))
	addressHub(t, tlsRoot, secure.URL, 2)

	runs := map[string]<-chan cozyRun{
		plain.URL:  startCozy(t, plainRoot, nil, "--json", "package", "search", "slow"),
		secure.URL: startCozy(t, tlsRoot, []string{trust}, "--json", "package", "search", "slow"),
	}
	for hubURL, done := range runs {
		run := <-done
		if run.code != 0 || !strings.Contains(run.stdout, "slow/hub") {
			t.Fatalf("a Hub answering after %s failed the call (exit %d)\n%s\n%s", slowAnswer, run.code, run.stdout, run.stderr)
		}
		if run.took < slowAnswer {
			t.Fatalf("the call returned in %s, before the Hub answered", run.took)
		}
		if !strings.Contains(run.stderr, "waiting on Tensorhub at "+hubURL+" (") {
			t.Fatalf("a long wait on %s showed no progress\n%s", hubURL, run.stderr)
		}
	}
	if plainProtocol.Load() != 1 || tlsProtocol.Load() != 2 {
		t.Fatalf("served over HTTP/%d and HTTP/%d, want HTTP/1 and HTTP/2", plainProtocol.Load(), tlsProtocol.Load())
	}
}

// A connection that goes dead mid-call (the blackhole swallows every byte, PINGs included,
// while its kernel still ACKs TCP keepalives) fails within the liveness window with an
// error that says so.
func TestDeadHubConnectionFailsWithinLiveness(t *testing.T) {
	const liveness = 2 * time.Second
	var path *blackhole
	released := make(chan struct{})
	root := livenessHome(t, "hub-dead-connection")
	upstream, trust := tlsHub(t, root, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path.cut()
		select {
		case <-released:
		case <-r.Context().Done():
		}
	}))
	path = newBlackhole(t, upstream.Listener.Addr().String())
	t.Cleanup(func() { close(released) })
	hubURL := "https://" + path.addr
	addressHub(t, root, hubURL, int(liveness/time.Second))

	run := <-startCozy(t, root, []string{trust}, "--json", "package", "search", "slow")
	if run.code == 0 {
		t.Fatalf("a dead connection answered\n%s", run.stdout)
	}
	name, message := errorOf(t, run.stdout)
	want := "the connection to Tensorhub at " + hubURL + " went dead: it stopped answering liveness probes"
	if name != "hub.unreachable" || message != want {
		t.Fatalf("got %s %q, want hub.unreachable %q\n%s", name, message, want, run.stderr)
	}
	path.mu.Lock()
	cut := path.dark
	path.mu.Unlock()
	if !cut {
		t.Fatal("the call failed before its connection went dark")
	}
	if run.took > liveness+10*time.Second {
		t.Fatalf("the dead connection took %s to fail, past its %s liveness window", run.took, liveness)
	}
}

// A refused connection fails at once, whatever the liveness window.
func TestRefusedHubFailsFast(t *testing.T) {
	hubURL := closedHub(t)
	root := livenessHome(t, "hub-refused")
	addressHub(t, root, hubURL, 120)
	run := <-startCozy(t, root, nil, "--json", "package", "search", "slow")
	name, message := errorOf(t, run.stdout)
	want := "Tensorhub unreachable at " + hubURL + " (connection refused)"
	if run.code == 0 || name != "hub.unreachable" || message != want {
		t.Fatalf("got exit %d %s %q, want hub.unreachable %q", run.code, name, message, want)
	}
	if run.took > 10*time.Second {
		t.Fatalf("a refused connection took %s to fail", run.took)
	}
}
