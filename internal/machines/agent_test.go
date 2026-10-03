package machines

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// An agent is accepted by what it says it implements, whatever language built it.
func TestAgentIsAcceptedByIdentityNotImplementation(t *testing.T) {
	dir := t.TempDir()
	agent := func(name, body string) string {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("#!/bin/sh\n[ \"$1\" = version ] || exit 2\necho '"+body+"'\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	ctx := context.Background()
	full := `{"name":"cozy-machine","implementation":"rust","wire_minor":72,"minimum_wire_minor":64,"capabilities":["hub-access/1","runtime-update/1","machine-bootstrap/1"]}`
	if !compatibleAgent(ctx, agent("rust", full)) {
		t.Fatal("a non-Go agent with the required identity and capabilities was refused")
	}
	for name, body := range map[string]string{
		"no-update":   `{"name":"cozy-machine","wire_minor":72,"minimum_wire_minor":64,"capabilities":["hub-access/1","machine-bootstrap/1"]}`,
		"other-name":  `{"name":"something-else","wire_minor":72,"minimum_wire_minor":64,"capabilities":["hub-access/1","runtime-update/1","machine-bootstrap/1"]}`,
		"newer-floor": `{"name":"cozy-machine","wire_minor":99,"minimum_wire_minor":98,"capabilities":["hub-access/1","runtime-update/1","machine-bootstrap/1"]}`,
	} {
		if compatibleAgent(ctx, agent(name, body)) {
			t.Fatalf("%s was accepted", name)
		}
	}
}
