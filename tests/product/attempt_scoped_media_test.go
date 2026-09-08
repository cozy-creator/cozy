package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
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

func inputMediaClient(t *testing.T, address string, budget time.Duration) *media.Client {
	t.Helper()
	client, problem := media.Dial(media.Spec{Addr: address, Token: secret.New("media-input-test")}, budget, 1024)
	fatal(t, problem)
	return client
}

func TestMediaRequiresScopedInputSupportBeforeBytes(t *testing.T) {
	for _, capability := range []string{"", `,"attempt_scoped_inputs":false`, `,"attempt_scoped_inputs":true`} {
		t.Run(fmt.Sprint(capability), func(t *testing.T) {
			var writes atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || r.URL.Path != "/v1/health" {
					writes.Add(1)
				}
				fmt.Fprintf(w, `{"service":"cozy-media","contract_rev":%d%s}`, mediawire.ContractRev, capability)
			}))
			defer server.Close()
			client := inputMediaClient(t, strings.TrimPrefix(server.URL, "http://"), time.Second)
			problem := client.Health()
			if strings.HasSuffix(capability, "true") {
				fatal(t, problem)
			} else if problem == nil || problem.ErrName() != "media_input_scope_unsupported" {
				t.Fatalf("old receiver was not refused by capability: %v", problem)
			}
			if writes.Load() != 0 {
				t.Fatal("health negotiation moved bytes before support was established")
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

func TestMediaInputRequiresExactReceipt(t *testing.T) {
	for _, response := range []string{
		`{"path":"/tmp/cozy/input","length":3}`,
		`{"path":"/tmp/cozy/input","length":2,"digest":"sha256:incorrect"}`,
	} {
		t.Run(response, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				_, _ = io.WriteString(w, response)
			}))
			defer server.Close()
			client := inputMediaClient(t, strings.TrimPrefix(server.URL, "http://"), time.Second)
			if path, problem := client.PutInput("attempt-1", "payload", []byte("abc")); problem == nil || path != "" {
				t.Fatalf("inexact receipt became input grant: %q, %v", path, problem)
			}
		})
	}
}

// This optional cross-repository proof dials an actual Tensorhub podmedia server
// with a disposable ledger/cache. The address never points at a paid rental.
func TestAttemptScopedInputsAgainstPodMedia(t *testing.T) {
	address := os.Getenv("COZY_TEST_POD_MEDIA_ADDR")
	if address == "" {
		t.Skip("requires disposable Tensorhub podmedia receiver at COZY_TEST_POD_MEDIA_ADDR")
	}
	client := inputMediaClient(t, address, 2*time.Second)
	fatal(t, client.Health())
	body := []byte("same cached bytes across attempts")
	slotA := media.Slot("input-proof-a", 1)
	slotB := media.Slot("input-proof-b", 1)
	defer client.DropAttempt(slotA)
	defer client.DropAttempt(slotB)
	if _, problem := client.PutInput(slotA, "payload", body); problem == nil {
		t.Fatal("receiver accepted bytes before reservation")
	}
	_, problem := client.ReserveOutputs(slotA, 0)
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
	_, problem = client.ReserveOutputs(slotB, 0)
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
			_, problem := client.ReserveOutputs(slot, 0)
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
