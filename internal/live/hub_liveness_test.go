package live

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/secret"
)

// TestHubLargeResponsesRefuseSilentHeadersAndBodies drives the paid response class that
// drops the ordinary total request timeout. It retains bounded headers and then requires
// continuing body progress.
func TestHubLargeResponsesRefuseSilentHeadersAndBodies(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "body") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("{"))
			w.(http.Flusher).Flush()
		}
		<-r.Context().Done()
	}))
	defer server.Close()

	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("admin")}, "live-test")
	type result struct {
		name    string
		problem *exit.Error
	}
	results := make(chan result, 2)
	var started sync.WaitGroup
	started.Add(2)
	call := func(name string, run func() *exit.Error) {
		go func() {
			started.Done()
			results <- result{name: name, problem: run()}
		}()
	}
	call("rental headers", func() *exit.Error {
		_, problem := client.Rental(context.Background(), "rental-header")
		return problem
	})
	call("rental body", func() *exit.Error {
		_, problem := client.Rental(context.Background(), "rental-body")
		return problem
	})
	started.Wait()

	deadline := time.After(15 * time.Second)
	for range 2 {
		select {
		case got := <-results:
			if got.problem == nil || got.problem.Code != exit.Deadline {
				t.Errorf("%s = %v, want deadline", got.name, got.problem)
			}
			if strings.Contains(got.name, "body") && got.problem != nil &&
				got.problem.ErrName() != "hub.response_stalled" {
				t.Errorf("%s = %s, want hub.response_stalled", got.name, got.problem.ErrName())
			}
		case <-deadline:
			t.Fatal("large Hub responses outlived their header/body progress bounds")
		}
	}
}

// TestRentTimeoutInterruptsTricklingHubResponses proves the CLI's one paid-flow deadline
// reaches both the initial POST and subsequent polling GET. The peer moves a byte often
// enough to keep the Hub body-progress guard satisfied; only the caller's --timeout can
// end these responses.
func TestRentTimeoutInterruptsTricklingHubResponses(t *testing.T) {
	for _, phase := range []string{"post", "poll"} {
		t.Run(phase, func(t *testing.T) {
			var sawPost, sawPoll bool
			var mu sync.Mutex
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				if r.Method == http.MethodPost {
					sawPost = true
				} else if r.Method == http.MethodGet {
					sawPoll = true
				}
				mu.Unlock()
				if phase == "poll" && r.Method == http.MethodPost {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusAccepted)
					_, _ = w.Write([]byte(`{"rental_id":"rental-1","state":"acquiring"}`))
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("{"))
				w.(http.Flusher).Flush()
				ticker := time.NewTicker(25 * time.Millisecond)
				defer ticker.Stop()
				for {
					select {
					case <-r.Context().Done():
						return
					case <-ticker.C:
						_, _ = w.Write([]byte(" "))
						w.(http.Flusher).Flush()
					}
				}
			}))
			defer server.Close()

			root := filepath.Join(t.TempDir(), "cozy-home")
			t.Cleanup(func() { terminateTestDaemon(t, root) })
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			args := []string{"-n", "19", cozyBin, "rental", "new", "cozy/package/v1/generate",
				"--accelerator", "NVIDIA H200", "--reason", "liveness proof",
				"--timeout", "300ms", "--idempotency-key", "rent-timeout-" + phase}
			cmd := exec.CommandContext(ctx, "/usr/bin/nice", args...)
			cmd.Env = childEnv(t, root,
				"TENSORHUB_URL="+server.URL, "TENSORHUB_TOKEN=admin")
			started := time.Now()
			data, _ := cmd.CombinedOutput()
			elapsed := time.Since(started)
			if ctx.Err() != nil {
				t.Fatalf("cozy rental new ignored --timeout and required harness termination:\n%s", data)
			}
			if cmd.ProcessState == nil || cmd.ProcessState.ExitCode() != 1 ||
				(!strings.Contains(string(data), "did not answer within") &&
					!strings.Contains(string(data), "at the --timeout you set")) {
				t.Fatalf("cozy rental new %s exit = %v, want operational 1 with deadline detail:\n%s",
					phase, cmd.ProcessState, data)
			}
			if elapsed > 2*time.Second {
				t.Errorf("cozy rental new %s returned after %s, far past its 300ms deadline", phase, elapsed)
			}
			mu.Lock()
			post, poll := sawPost, sawPoll
			mu.Unlock()
			if !post || phase == "poll" && !poll {
				t.Errorf("cozy rental new %s reached post=%t poll=%t", phase, post, poll)
			}
		})
	}
}
