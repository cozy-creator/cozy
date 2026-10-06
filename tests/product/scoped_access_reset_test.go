package producttest

import (
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
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+origin+"\n"), 0600))
	firstKey := writeMachineCredential(t, root, origin)
	otherKey := writeMachineCredential(t, root, other)
	witness := filepath.Join(root, "agent-started")
	binary := filepath.Join(root, "machine/root/usr/local/bin/cozy-machine")
	must(t, os.MkdirAll(filepath.Dir(binary), 0755))
	must(t, os.WriteFile(binary, []byte("#!/bin/sh\ntouch "+witness+"\n"), 0755))
	code, out := runCozy(t, root, "auth", "logout", "--json")
	if code != 0 || !strings.Contains(out, "logged out") {
		t.Fatal(code, out)
	}
	if _, err := os.Stat(firstKey); !os.IsNotExist(err) {
		t.Fatal("selected device key remains", err)
	}
	if _, err := os.Stat(otherKey); err != nil {
		t.Fatal("other Hub key was erased", err)
	}
	if _, err := os.Stat(witness); !os.IsNotExist(err) {
		t.Fatal("logout started a stopped agent")
	}
}
