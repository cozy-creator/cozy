package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

func TestScopedLogoutPreservesOtherHubAndNeverStartsStoppedAgent(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	origin := server.URL
	server.Close()
	root := t.TempDir()
	other := "https://other.example.test"
	agentOrigin := "https://agent-origin.example.test"
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+origin+"\n"), 0600))
	firstKey := writeMachineCredential(t, root, origin)
	otherKey := writeMachineCredential(t, root, other)
	path := filepath.Join(root, "machine", "execution-access.json")
	must(t, os.MkdirAll(filepath.Dir(path), 0700))
	cache := map[string]any{origin: map[string]any{"origin": agentOrigin, "token": "first-scoped-secret"}, other: map[string]any{"origin": other, "token": "other-scoped-secret", "future_binding": "keep"}}
	raw, err := json.Marshal(cache)
	must(t, err)
	must(t, os.WriteFile(path, raw, 0600))
	witness := filepath.Join(root, "agent-started")
	binary := filepath.Join(root, "machine/root/usr/local/bin/cozy-machine")
	must(t, os.MkdirAll(filepath.Dir(binary), 0755))
	must(t, os.WriteFile(binary, []byte("#!/bin/sh\ntouch "+witness+"\n"), 0755))
	code, out := runCozy(t, root, "auth", "logout", "--json")
	if code != 0 || !strings.Contains(out, "logged out") || !strings.Contains(out, "queued") || strings.Contains(out, "scoped-secret") {
		t.Fatal(code, out)
	}
	if _, err := os.Stat(firstKey); !os.IsNotExist(err) {
		t.Fatal("selected device key remains", err)
	}
	if _, err := os.Stat(otherKey); err != nil {
		t.Fatal("other Hub key was erased", err)
	}
	raw, err = os.ReadFile(path)
	must(t, err)
	var kept map[string]json.RawMessage
	must(t, json.Unmarshal(raw, &kept))
	if len(kept) != 1 || !strings.Contains(string(kept[other]), "future_binding") {
		t.Fatal("logout changed another Hub's cache")
	}
	pending, err := os.ReadFile(filepath.Join(root, "machine/execution-access-resets.json"))
	must(t, err)
	if !strings.Contains(string(pending), agentOrigin) || !strings.Contains(string(pending), `"pending":true`) {
		t.Fatal("actual agent origin was not queued")
	}
	if _, err := os.Stat(witness); !os.IsNotExist(err) {
		t.Fatal("logout started a stopped agent")
	}
}
