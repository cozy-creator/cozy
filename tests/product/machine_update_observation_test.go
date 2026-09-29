package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
)

func updateObservationPeer(t *testing.T, answer func(http.ResponseWriter, *http.Request, int32)) (*machines.Maintenance, *atomic.Int32) {
	t.Helper()
	public, key, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	calls := &atomic.Int32{}
	var posts atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Cozy-Cap ")
		grant, err := capability.Verify(token, "observer-machine", []ed25519.PublicKey{public}, time.Now(), "")
		if err != nil || !grant.Permits(capability.Maintenance) {
			t.Errorf("observation lost its signed maintenance authority: %v", err)
			http.Error(w, "unauthorized", 403)
			return
		}
		if r.Method == http.MethodPost && r.URL.Path == "/v1/machine/runtime/update" {
			if posts.Add(1) != 1 {
				t.Error("observer resubmitted an accepted update")
			}
			var request struct{ Operation string }
			if json.NewDecoder(r.Body).Decode(&request) != nil || request.Operation != "accepted-operation" {
				t.Error("initial update did not retain operation identity")
			}
			w.WriteHeader(http.StatusAccepted)
			return
		}
		if r.Method != http.MethodGet || r.URL.Path != "/v1/machine/runtime" {
			t.Errorf("observer issued mutation or changed route: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected mutation", 400)
			return
		}
		answer(w, r, calls.Add(1))
	}))
	t.Cleanup(server.Close)
	client := server.Client()
	client.Transport.(*http.Transport).DisableKeepAlives = true
	return &machines.Maintenance{Base: server.URL, Client: client, Machine: "observer-machine", Public: public, Sign: func(raw []byte) []byte { return ed25519.Sign(key, raw) }}, calls
}
func updateObservationState(w http.ResponseWriter, operation, state string) {
	_ = json.NewEncoder(w).Encode(map[string]any{"phase": "ready", "capabilities": []string{machines.RuntimeUpdateCapability}, "runtime": "0.18.88", "tensorfs": "0.3.78", "agent": map[string]any{"version": "0.18.88", "sha256": strings.Repeat("a", 64), "selection": "bundled"}, "bootstrap": map[string]any{"abi": machines.BootstrapCapability}, "update": map[string]any{"operation": operation, "state": state, "error": "candidate refusal"}})
}
func TestAcceptedMachineUpdateObservationSurvivesApplicationHandoff(t *testing.T) {
	client, calls := updateObservationPeer(t, func(w http.ResponseWriter, r *http.Request, n int32) {
		switch n {
		case 1:
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"code":"runtime_starting","message":"replacement is initializing"}`))
		case 2:
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			conn.Close()
		case 3:
			w.Header().Set("Content-Length", "1000")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"phase":`))
			w.(http.Flusher).Flush()
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			conn.Close()
		case 4:
			updateObservationState(w, "accepted-operation", "installing")
		default:
			updateObservationState(w, "accepted-operation", "succeeded")
		}
	})
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	code, problem := client.Do(ctx, http.MethodPost, "/v1/machine/runtime/update", bytes.NewBufferString(`{"operation":"accepted-operation"}`), nil)
	if problem != nil || code != http.StatusAccepted {
		t.Fatalf("accept update: HTTP %d %v", code, problem)
	}
	state, problem := client.AwaitUpdate(ctx, "accepted-operation")
	if problem != nil || state == nil || state.Update == nil || state.Update.State != "succeeded" || calls.Load() != 5 {
		t.Fatalf("accepted operation lost through handoff: calls=%d state=%+v problem=%v", calls.Load(), state, problem)
	}
}
func TestAcceptedMachineUpdateObservationKeepsTerminalFailure(t *testing.T) {
	for _, terminal := range []string{"rolled_back", "failed"} {
		t.Run(terminal, func(t *testing.T) {
			client, calls := updateObservationPeer(t, func(w http.ResponseWriter, _ *http.Request, _ int32) {
				updateObservationState(w, "accepted-operation", terminal)
			})
			state, problem := client.AwaitUpdate(t.Context(), "accepted-operation")
			if problem != nil || state.Update.State != terminal || state.Update.Error != "candidate refusal" || calls.Load() != 1 {
				t.Fatalf("terminal update changed: %+v %v", state, problem)
			}
		})
	}
}

func TestAcceptedMachineUpdateObservationReturnsPendingActivation(t *testing.T) {
	client, calls := updateObservationPeer(t, func(w http.ResponseWriter, _ *http.Request, _ int32) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"phase": "ready", "capabilities": []string{machines.RuntimeUpdateCapability},
			"runtime": "0.18.88", "tensorfs": "0.3.78",
			"agent":     map[string]any{"version": "0.18.88", "sha256": strings.Repeat("a", 64), "selection": "bundled"},
			"bootstrap": map[string]any{"abi": machines.BootstrapCapability},
			"update": map[string]any{
				"operation": "accepted-operation", "state": "waiting_activation",
				"from": map[string]string{"runtime": "0.18.88", "tensorfs": "0.3.78"},
				"to":   map[string]string{"runtime": "0.18.89", "tensorfs": "0.3.78"},
			},
		})
	})
	state, problem := client.AwaitUpdateOrPending(t.Context(), "accepted-operation")
	if problem != nil || state == nil || state.Update == nil || !state.Update.PendingActivation() ||
		state.Update.To.Runtime != "0.18.89" || calls.Load() != 1 {
		t.Fatalf("pending activation was not returned as a durable candidate: calls=%d state=%+v problem=%v", calls.Load(), state, problem)
	}
}

func TestAcceptedMachineUpdateObservationRejectsPermanentLoss(t *testing.T) {
	for _, arm := range []string{"auth", "invalid JSON", "another operation", "missing operation", "wrong certificate"} {
		t.Run(arm, func(t *testing.T) {
			client, calls := updateObservationPeer(t, func(w http.ResponseWriter, _ *http.Request, _ int32) {
				switch arm {
				case "auth":
					http.Error(w, "owner authority revoked", http.StatusForbidden)
				case "invalid JSON":
					_, _ = w.Write([]byte(`{"unfinished"`))
				case "another operation":
					updateObservationState(w, "replacement-operation", "succeeded")
				case "missing operation":
					updateObservationState(w, "", "succeeded")
				default:
					updateObservationState(w, "accepted-operation", "succeeded")
				}
			})
			if arm == "wrong certificate" {
				client.Client.Transport = &http.Transport{TLSClientConfig: &tls.Config{RootCAs: x509.NewCertPool(), MinVersion: tls.VersionTLS12}}
			}
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			defer cancel()
			state, problem := client.AwaitUpdate(ctx, "accepted-operation")
			if state != nil || problem == nil || ctx.Err() != nil || calls.Load() > 1 {
				t.Fatalf("permanent loss was retried or hidden: state=%+v error=%v calls=%d context=%v", state, problem, calls.Load(), ctx.Err())
			}
			if arm == "wrong certificate" && problem.Code != exit.Credential {
				t.Fatalf("certificate refusal lost its credential classification: %v", problem)
			}
		})
	}
}
func TestAcceptedMachineUpdateObservationCancellationOnlyDetaches(t *testing.T) {
	entered := make(chan struct{}, 1)
	client, calls := updateObservationPeer(t, func(w http.ResponseWriter, _ *http.Request, _ int32) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"code":"runtime_starting","message":"replacement is initializing"}`))
		entered <- struct{}{}
	})
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan *exit.Error, 1)
	go func() { _, problem := client.AwaitUpdate(ctx, "accepted-operation"); done <- problem }()
	<-entered
	cancel()
	problem := <-done
	if problem == nil || problem.Code != exit.Canceled || !strings.Contains(problem.Message, "accepted-operation") || calls.Load() != 1 {
		t.Fatalf("cancel changed accepted operation or failed to detach: %v calls=%d", problem, calls.Load())
	}
}
