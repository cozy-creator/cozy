package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/providerrelease"
	"github.com/cozy-creator/cozy/internal/secret"
)

func runExternalCleanup(t *testing.T, root, origin, input string, flags ...string) (int, string) {
	t.Helper()
	args := []string{"rental", "end-external", "--provider", "vast", "--resource-id", "42", "--expected-label", "owned-proof", "--provider-url", origin, "--json"}
	args = append(args, flags...)
	cmd := exec.Command(cozyBin, args...)
	cmd.Env = childEnv(t, root)
	cmd.Stdin = strings.NewReader(input) //cozy:stdin-value explicitly supplied test provider credential
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	err := cmd.Run()
	if err == nil {
		return 0, out.String()
	}
	if status, ok := err.(*exec.ExitError); ok {
		return status.ExitCode(), out.String()
	}
	t.Fatal(err)
	return -1, out.String()
}

// This is the actual CLI and HTTP boundary, not provider account discovery or a
// live destructive test. Only GET/DELETE of the supplied ID may be sent.
func TestExternalProviderCleanupChecksIdentityAndConfirmsAbsence(t *testing.T) {
	const token = "explicit-provider-token"
	var mu sync.Mutex
	var requests []string
	deleted := false
	hubCalls := 0
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		hubCalls++
		mu.Unlock()
		w.WriteHeader(500)
	}))
	defer hub.Close()
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		requests = append(requests, r.Method+" "+r.URL.RequestURI())
		if r.URL.Path != "/api/v0/instances/42/" || r.Header.Get("Authorization") != "Bearer "+token {
			w.WriteHeader(403)
			return
		}
		switch r.Method {
		case "GET":
			if deleted {
				w.WriteHeader(404)
				return
			}
			fmt.Fprint(w, `{"instances":{"id":42,"label":"owned-proof","actual_status":"stopped"}}`)
		case "DELETE":
			deleted = true
			fmt.Fprint(w, `{"success":true}`)
		default:
			w.WriteHeader(405)
		}
	}))
	defer provider.Close()
	root := t.TempDir()
	original := []byte("tensorhub_url: " + hub.URL + "\n")
	must(t, os.WriteFile(filepath.Join(root, config.FileName), original, 0600))
	for attempt := 0; attempt < 2; attempt++ {
		code, out := runExternalCleanup(t, root, provider.URL, token+"\n", "--token-stdin")
		var got providerrelease.Result
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &got) != nil || !got.ProviderAbsent || got.State != "absent" || got.ResourceID != 42 || got.Changed != (attempt == 0) {
			t.Fatalf("cleanup attempt%d: exit%d %s", attempt, code, out)
		}
		if strings.Contains(out, token) {
			t.Fatal("credential was printed")
		}
	}
	current, err := os.ReadFile(filepath.Join(root, config.FileName))
	must(t, err)
	if !bytes.Equal(current, original) {
		t.Fatal("selected Hub configuration changed")
	}
	mu.Lock()
	defer mu.Unlock()
	if hubCalls != 0 {
		t.Fatalf("external cleanup contacted Hub %d times", hubCalls)
	}
	want := []string{"GET /api/v0/instances/42/", "DELETE /api/v0/instances/42/", "GET /api/v0/instances/42/", "GET /api/v0/instances/42/"}
	if !reflect.DeepEqual(requests, want) {
		t.Fatalf("provider requests: %v", requests)
	}
}

func TestExternalProviderCleanupRefusesUnprovenIdentityAndFailureResponses(t *testing.T) {
	const token = "do-not-echo-this-token"
	cases := []struct {
		name         string
		status       int
		body         string
		deleteStatus int
		deleteBody   string
		wantDeletes  int
	}{
		{"wrong-label", 200, `{"instances":{"id":42,"label":"somebody-else"}}`, 200, "", 0},
		{"wrong-id", 200, `{"instances":{"id":43,"label":"owned-proof"}}`, 200, "", 0},
		{"missing-instances-is-not-absence", 200, `{}`, 200, "", 0},
		{"null-envelope-is-not-absence", 200, `null`, 200, "", 0},
		{"array-is-not-single-instance", 200, `{"instances":[]}`, 200, "", 0},
		{"null-with-failure-is-not-absence", 200, `{"success":false,"instances":null}`, 200, "", 0},
		{"unauthorized", 401, token, 200, "", 0},
		{"forbidden", 403, token, 200, "", 0},
		{"read-failed", 503, token, 200, "", 0},
		{"false-success", 200, `{"success":false,"message":"` + token + `"}`, 200, "", 0},
		{"delete-refused", 200, `{"instances":{"id":42,"label":"owned-proof"}}`, 200, `{"success":false,"message":"` + token + `"}`, 1},
		{"delete-unauthorized", 200, `{"instances":{"id":42,"label":"owned-proof"}}`, 403, token, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			deletes := 0
			provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				defer mu.Unlock()
				if r.Method == "DELETE" {
					deletes++
					w.WriteHeader(tc.deleteStatus)
					fmt.Fprint(w, tc.deleteBody)
					return
				}
				w.WriteHeader(tc.status)
				fmt.Fprint(w, tc.body)
			}))
			defer provider.Close()
			code, out := runExternalCleanup(t, t.TempDir(), provider.URL, token, "--token-stdin")
			if code == 0 || strings.Contains(out, token) || strings.Contains(out, `"provider_absent":true`) {
				t.Fatalf("failure reported incorrectly: %d %s", code, out)
			}
			mu.Lock()
			defer mu.Unlock()
			if deletes != tc.wantDeletes {
				t.Fatalf("DELETE count%d; want%d", deletes, tc.wantDeletes)
			}
		})
	}
}

func TestExternalProviderCleanupDoesNotRedirectOrReadAmbientCredentials(t *testing.T) {
	var mu sync.Mutex
	calls, redirected := 0, 0
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { mu.Lock(); redirected++; mu.Unlock(); w.WriteHeader(404) }))
	defer destination.Close()
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		mu.Unlock()
		http.Redirect(w, r, destination.URL, 307)
	}))
	defer provider.Close()
	for _, tc := range []struct {
		input string
		flags []string
	}{
		{"explicit-token", nil}, {"", []string{"--token-stdin"}}, {"first\nsecond", []string{"--token-stdin"}}, {strings.Repeat("x", 8193), []string{"--token-stdin"}},
	} {
		if code, out := runExternalCleanup(t, t.TempDir(), provider.URL, tc.input, tc.flags...); code == 0 {
			t.Fatalf("invalid credential accepted: %s", out)
		}
	}
	mu.Lock()
	before := calls
	mu.Unlock()
	if before != 0 {
		t.Fatalf("invalid credentials made%d requests", before)
	}
	if code, out := runExternalCleanup(t, t.TempDir(), provider.URL, "explicit-token", "--token-stdin"); code == 0 {
		t.Fatalf("redirect accepted: %s", out)
	}
	mu.Lock()
	defer mu.Unlock()
	if calls != 1 || redirected != 0 {
		t.Fatalf("credential redirect: calls%d redirected%d", calls, redirected)
	}
}

func TestExternalProviderCleanupInterruptedReadbackDoesNotClaimRelease(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var mu sync.Mutex
	deletes := 0
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "DELETE" {
			mu.Lock()
			deletes++
			mu.Unlock()
			cancel()
			fmt.Fprint(w, `{"success":true}`)
			return
		}
		fmt.Fprint(w, `{"instances":{"id":42,"label":"owned-proof"}}`)
	}))
	defer provider.Close()
	got, problem := providerrelease.Vast(ctx, provider.URL, 42, "owned-proof", secret.New("explicit-token"))
	if problem == nil || got.ProviderAbsent || got.State == "absent" {
		t.Fatalf("interrupted readback claimed absence: %+v %v", got, problem)
	}
	mu.Lock()
	defer mu.Unlock()
	if deletes != 1 {
		t.Fatalf("sent%d deletes", deletes)
	}
}

// Live Vast readback and its own CLI use HTTP200 instances:null for an absent
// resource, both before deletion and after a previously accepted deletion.
func TestExternalProviderCleanupAcceptsExplicitNullAbsence(t *testing.T) {
	for _, present := range []bool{false, true} {
		t.Run(fmt.Sprintf("initially-present-%t", present), func(t *testing.T) {
			var mu sync.Mutex
			var methods []string
			exists := present
			provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				defer mu.Unlock()
				methods = append(methods, r.Method)
				if r.Method == "DELETE" {
					exists = false
					fmt.Fprint(w, `{"success":true}`)
					return
				}
				if exists {
					fmt.Fprint(w, `{"instances":{"id":42,"label":"owned-proof"}}`)
					return
				}
				fmt.Fprint(w, `{"instances": null}`)
			}))
			defer provider.Close()
			code, out := runExternalCleanup(t, t.TempDir(), provider.URL, "explicit-token", "--token-stdin")
			var got providerrelease.Result
			if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &got) != nil || !got.ProviderAbsent || got.State != "absent" || got.Changed != present {
				t.Fatalf("null absence: exit%d %s", code, out)
			}
			mu.Lock()
			defer mu.Unlock()
			want := []string{"GET"}
			if present {
				want = []string{"GET", "DELETE", "GET"}
			}
			if !reflect.DeepEqual(methods, want) {
				t.Fatalf("unexpected provider calls: %v", methods)
			}
		})
	}
}
