package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/secret"
)

// inputMediaHealth may carry the retired `attempt_scoped_inputs` flag; the client must ignore it.
func inputMediaHealth(w io.Writer, scoped *bool) {
	revision := mediawire.ContractRev
	_ = json.NewEncoder(w).Encode(struct {
		mediawire.Health
		Scoped *bool `json:"attempt_scoped_inputs,omitempty"`
	}{Health: mediawire.Health{Service: mediawire.Service, ContractRev: &revision}, Scoped: scoped})
}

func inputMediaClient(t *testing.T, address string, budget time.Duration) *media.Client {
	t.Helper()
	client, problem := media.Dial(media.Spec{Addr: address, Token: secret.New("media-input-test")}, budget, 1024)
	fatal(t, problem)
	return client
}

// The pod's media server ships in its image, independently of this host. A plane at the
// current revision or newer is dialled and moves bytes; an older plane is refused before
// any byte moves.
func TestMediaPlaneRevisionIsAFloor(t *testing.T) {
	for _, arm := range []struct {
		name     string
		rev      *int
		accepted bool
	}{
		{"absent", nil, false},
		{"older", new(mediawire.ContractRev - 1), false},
		{"current", new(mediawire.ContractRev), true},
		{"newer", new(mediawire.ContractRev + 1), true},
	} {
		t.Run(arm.name, func(t *testing.T) {
			var writes atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/health" {
					_ = json.NewEncoder(w).Encode(mediawire.Health{Service: mediawire.Service, ContractRev: arm.rev})
					return
				}
				writes.Add(1)
				if strings.HasPrefix(r.URL.Path, "/v1/triage/") {
					_, _ = w.Write([]byte("{}"))
					return
				}
				body, _ := io.ReadAll(r.Body)
				_ = json.NewEncoder(w).Encode(map[string]any{"path": "/tmp/cozy/input", "digest": inputDigest(body), "length": len(body)})
			}))
			defer server.Close()
			client := inputMediaClient(t, strings.TrimPrefix(server.URL, "http://"), time.Second)
			problem := client.Health(t.Context())
			if !arm.accepted {
				if problem == nil || problem.ErrName() != "media_contract_mismatch" {
					t.Fatalf("a plane below the floor was not refused: %v", problem)
				}
				return
			}
			fatal(t, problem)
			_, problem = client.PutInput("attempt-7", "payload", []byte("payload"))
			fatal(t, problem)
			_, problem = client.GetTriage("subject-1", inputDigest([]byte("{}")), 2)
			fatal(t, problem)
		})
	}
}

func TestScopedMediaFailureNeverFallsBackToUnscopedUpload(t *testing.T) {
	var scoped, legacy atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/health" {
			scoped := true
			inputMediaHealth(w, &scoped)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/v1/inputs/") {
			legacy.Add(1)
		} else {
			scoped.Add(1)
		}
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusConflict)
		_, _ = io.WriteString(w, `{"error":{"code":"media.input_conflict","message":"changed input"}}`)
	}))
	defer server.Close()
	client := inputMediaClient(t, strings.TrimPrefix(server.URL, "http://"), time.Second)
	fatal(t, client.Health(t.Context()))
	if path, problem := client.PutInput("attempt-7", "payload", []byte("payload")); problem == nil || path != "" || problem.ErrName() != "media.input_conflict" {
		t.Fatalf("scoped refusal changed: %s %v", path, problem)
	}
	if scoped.Load() != 1 || legacy.Load() != 0 {
		t.Fatalf("scoped failure retried legacy route: scoped=%d legacy=%d", scoped.Load(), legacy.Load())
	}
}

// A zero-output attempt still reserves an explicit count; the receiver must
// distinguish zero from an old client which omitted inode obligations entirely.
func TestMediaOutputReservationCarriesExactCount(t *testing.T) {
	for _, count := range []int{0, 3} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			bound := int64(count) * 512
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/outputs/attempt-7" ||
					r.URL.Query().Get("output_count") != fmt.Sprint(count) ||
					r.URL.Query().Get("max_bytes") != fmt.Sprint(bound) {
					t.Errorf("output byte/count reservation changed: %s %s", r.Method, r.URL)
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"dir": "/outputs/attempt-7"})
			}))
			defer server.Close()
			client := inputMediaClient(t, strings.TrimPrefix(server.URL, "http://"), time.Second)
			dir, problem := client.ReserveOutputs("attempt-7", bound, count)
			fatal(t, problem)
			if dir != "/outputs/attempt-7" {
				t.Fatalf("reservation lost its directory: %q", dir)
			}
		})
	}
}

// Exercise the common bytes/file transport with a response lost after the peer
// consumes the body. Retrying preserves both path components and the original file.
func TestMediaInputRetryKeepsAttemptAndOriginalFile(t *testing.T) {
	body := []byte("shared input")
	original := filepath.Join(t.TempDir(), "original")
	must(t, os.WriteFile(original, body, 0600))
	before, err := os.Stat(original)
	must(t, err)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut || r.URL.EscapedPath() != "/v1/attempts/attempt-7/inputs/input-2" {
			t.Errorf("wrong scoped upload: %s %s", r.Method, r.URL.EscapedPath())
		}
		if r.Header.Get("Authorization") != "Bearer media-input-test" || r.ContentLength != int64(len(body)) {
			t.Error("upload omitted credential or exact length")
		}
		got, err := io.ReadAll(r.Body)
		if err != nil || !bytes.Equal(got, body) {
			t.Errorf("upload body changed: %q, %v", got, err)
		}
		if requests.Add(1) == 1 {
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			_ = conn.Close()
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"path": "/tmp/cozy/shared.bin", "digest": inputDigest(got), "length": len(got)})
	}))
	defer server.Close()
	client := inputMediaClient(t, strings.TrimPrefix(server.URL, "http://"), time.Second)
	if _, problem := client.PutInput("attempt-7", "input-2", body); problem == nil {
		t.Fatal("lost response incorrectly produced a usable path")
	}
	path, problem := client.PutInputFile("attempt-7", "input-2", original, inputDigest(body), int64(len(body)))
	fatal(t, problem)
	if path != "/tmp/cozy/shared.bin" || requests.Load() != 2 {
		t.Fatalf("retry changed destination or replayed unexpectedly: %q, %d", path, requests.Load())
	}
	after, err := os.Stat(original)
	must(t, err)
	if !os.SameFile(before, after) {
		t.Fatal("upload replaced the borrowed original")
	}
	got, err := os.ReadFile(original)
	must(t, err)
	if !bytes.Equal(got, body) {
		t.Fatal("upload modified the borrowed original")
	}
}

// The pod's receipt echo is optional (the worker re-verifies the digest it consumes);
// an echo it does send must agree with the bytes streamed.
func TestMediaInputReceiptEchoMustAgree(t *testing.T) {
	for response, accepted := range map[string]bool{
		`{"path":"/tmp/cozy/input","length":3}`:                             true,
		`{"path":"/tmp/cozy/input","length":2,"digest":"sha256:incorrect"}`: false,
		`{"path":"/tmp/cozy/input","length":3,"digest":"sha256:incorrect"}`: false,
	} {
		t.Run(response, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/health" {
					inputMediaHealth(w, nil)
					return
				}
				_, _ = io.Copy(io.Discard, r.Body)
				_, _ = io.WriteString(w, response)
			}))
			defer server.Close()
			client := inputMediaClient(t, strings.TrimPrefix(server.URL, "http://"), time.Second)
			fatal(t, client.Health(t.Context()))
			path, problem := client.PutInput("attempt-1", "payload", []byte("abc"))
			if accepted != (problem == nil && path != "") {
				t.Fatalf("receipt echo admission = %q, %v; want accepted=%v", path, problem, accepted)
			}
		})
	}
}

func inputDigest(data []byte) string { return fmt.Sprintf("sha256:%x", sha256.Sum256(data)) }
