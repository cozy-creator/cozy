package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
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

var podMediaProofAddress = flag.String("pod-media-proof-addr", "", "disposable Tensorhub media receiver for cross-repository proof")

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

// A pod publishing the retired `attempt_scoped_inputs` flag as true, false, or not at
// all gets the same scoped upload route and explicit `output_count`.
func TestMediaHealthFlagsNeverChangeTheRequestShape(t *testing.T) {
	no, yes := false, true
	for _, capability := range []*bool{nil, &no, &yes} {
		name := "absent"
		if capability != nil {
			name = fmt.Sprint(*capability)
		}
		t.Run(name, func(t *testing.T) {
			var writes atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodGet && r.URL.Path == "/v1/health" {
					inputMediaHealth(w, capability)
					return
				}
				writes.Add(1)
				if r.Method == http.MethodPost {
					if r.URL.Path != "/v1/outputs/attempt-7" ||
						r.URL.Query().Get("output_count") != "3" ||
						r.URL.Query().Get("max_bytes") != "1024" {
						t.Errorf("health flag changed the reservation: %s", r.URL)
					}
					_ = json.NewEncoder(w).Encode(map[string]any{"dir": "/outputs/attempt-7"})
					return
				}
				if r.Method != http.MethodPut || r.URL.Path != "/v1/attempts/attempt-7/inputs/payload" {
					t.Errorf("health flag changed the upload: %s %s", r.Method, r.URL.Path)
				}
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"path": "/tmp/cozy/input", "digest": inputDigest(body), "length": len(body)})
			}))
			defer server.Close()
			client := inputMediaClient(t, strings.TrimPrefix(server.URL, "http://"), time.Second)
			fatal(t, client.Health(t.Context()))
			if writes.Load() != 0 {
				t.Fatal("health moved bytes before the handshake settled")
			}
			_, problem := client.ReserveOutputs("attempt-7", 1024, 3)
			fatal(t, problem)
			_, problem = client.PutInput("attempt-7", "payload", []byte("payload"))
			fatal(t, problem)
			original := filepath.Join(t.TempDir(), "original")
			must(t, os.WriteFile(original, []byte("payload"), 0600))
			_, problem = client.PutInputFile("attempt-7", "payload", original, inputDigest([]byte("payload")), 7)
			fatal(t, problem)
			if writes.Load() != 3 {
				t.Fatalf("unexpected transport replay count: %d", writes.Load())
			}
		})
	}
}

// The pod's media server ships in its image, independently of this host. A plane at the
// floor or newer is dialled and moves bytes; a route an older plane lacks fails only that
// operation, and a plane below the floor is refused before any byte moves.
func TestMediaPlaneRevisionIsAFloor(t *testing.T) {
	for _, arm := range []struct {
		name     string
		rev      *int
		accepted bool
		triage   bool
	}{
		{"absent", nil, false, false},
		{"rev1", new(1), false, false},
		{"rev2", new(2), true, false},
		{"current", new(mediawire.ContractRev), true, true},
		{"newer", new(mediawire.ContractRev + 1), true, true},
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
			if arm.triage {
				fatal(t, problem)
			} else if problem == nil || problem.ErrName() != "triage_unsupported" {
				t.Fatalf("an older plane's missing triage route was not a typed operation failure: %v", problem)
			}
		})
	}
}

// An unprobed client still sends the exact output count.
func TestMediaReservationCarriesCountWithoutHealth(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("output_count") != "0" {
			t.Errorf("unprobed client omitted the output count: %s", r.URL)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"dir": "/outputs/attempt-9"})
	}))
	defer server.Close()
	client := inputMediaClient(t, strings.TrimPrefix(server.URL, "http://"), time.Second)
	_, problem := client.ReserveOutputs("attempt-9", 0, 0)
	fatal(t, problem)
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

// This optional cross-repository proof dials an actual Tensorhub podmedia server
// with a disposable ledger/cache. The address never points at a paid rental.
func TestAttemptScopedInputsAgainstPodMedia(t *testing.T) {
	address := *podMediaProofAddress
	if address == "" {
		t.Skip("requires disposable Tensorhub podmedia receiver at -pod-media-proof-addr")
	}
	client := inputMediaClient(t, address, 2*time.Second)
	fatal(t, client.Health(t.Context()))
	outputSlot := media.Slot("output-count-proof", 1)
	defer client.DropAttempt(outputSlot)
	_, problem := client.ReserveOutputs(outputSlot, 4096, 3)
	fatal(t, problem)
	_, problem = client.ReserveOutputs(outputSlot, 4096, 3)
	fatal(t, problem)
	if _, problem = client.ReserveOutputs(outputSlot, 4096, 2); problem == nil {
		t.Fatal("receiver allowed an existing output count to change")
	}
	body := []byte("same cached bytes across attempts")
	slotA := media.Slot("input-proof-a", 1)
	slotB := media.Slot("input-proof-b", 1)
	defer client.DropAttempt(slotA)
	defer client.DropAttempt(slotB)
	if _, problem := client.PutInput(slotA, "payload", body); problem == nil {
		t.Fatal("receiver accepted bytes before reservation")
	}
	_, problem = client.ReserveOutputs(slotA, 0, 0)
	fatal(t, problem)
	// Let the actual receiver commit the input, then drop its response before
	// Creator can obtain a path. The next upload retries the same binding.
	proxy := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: "http", Host: address})
	var committed atomic.Bool
	proxy.ModifyResponse = func(response *http.Response) error {
		if response.StatusCode >= 400 {
			return nil
		}
		_, err := io.Copy(io.Discard, response.Body)
		_ = response.Body.Close()
		if err != nil {
			return err
		}
		committed.Store(true)
		return fmt.Errorf("discard committed upload response")
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, _ error) {
		connection, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			_ = connection.Close()
		}
	}
	lostReply := httptest.NewServer(proxy)
	defer lostReply.Close()
	interrupted := inputMediaClient(t, strings.TrimPrefix(lostReply.URL, "http://"), time.Second)
	if path, problem := interrupted.PutInput(slotA, "payload", body); problem == nil || path != "" || !committed.Load() {
		t.Fatalf("lost committed response yielded a grant: %q, %v, committed=%t", path, problem, committed.Load())
	}
	path, problem := client.PutInput(slotA, "payload", body)
	fatal(t, problem)
	if _, problem := client.PutInput(slotA, "payload", []byte("changed binding")); problem == nil {
		t.Fatal("receiver rebound an existing input to changed bytes")
	}
	_, problem = client.ReserveOutputs(slotB, 0, 0)
	fatal(t, problem)
	original := filepath.Join(t.TempDir(), "original")
	must(t, os.WriteFile(original, body, 0600))
	shared, problem := client.PutInputFile(slotB, "input-0", original, inputDigest(body), int64(len(body)))
	fatal(t, problem)
	if shared != path {
		t.Fatalf("same bytes copied into per-attempt paths: %q != %q", shared, path)
	}
	fatal(t, client.DropAttempt(slotA))
	fatal(t, client.DropAttempt(slotA))
	got, err := os.ReadFile(shared)
	must(t, err)
	if !bytes.Equal(got, body) {
		t.Fatal("pre-accept cancellation removed another attempt's input")
	}
	got, err = os.ReadFile(original)
	must(t, err)
	if !bytes.Equal(got, body) {
		t.Fatal("attempt cleanup changed the borrowed original")
	}
	for _, boundary := range []string{"reserved", "payload", "file"} {
		t.Run("cancel_"+boundary, func(t *testing.T) {
			slot := media.Slot("cancel-"+boundary, 1)
			_, problem := client.ReserveOutputs(slot, 0, 0)
			fatal(t, problem)
			if boundary != "reserved" {
				_, problem = client.PutInput(slot, "payload", body)
				fatal(t, problem)
			}
			if boundary == "file" {
				_, problem = client.PutInputFile(slot, "input-0", original, inputDigest(body), int64(len(body)))
				fatal(t, problem)
			}
			fatal(t, client.DropAttempt(slot))
			fatal(t, client.DropAttempt(slot))
			if _, problem = client.PutInput(slot, "payload", body); problem == nil {
				t.Fatal("released reservation still accepted an input")
			}
			got, err := os.ReadFile(shared)
			must(t, err)
			if !bytes.Equal(got, body) {
				t.Fatal("cancellation removed another attempt's input")
			}
		})
	}
}

func inputDigest(data []byte) string { return fmt.Sprintf("sha256:%x", sha256.Sum256(data)) }
