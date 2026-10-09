package devcomfy

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSSHUsesStrictPinAndPayloadIsStdinOnly(t *testing.T) {
	ssh := SSH{Host: "127.0.0.1", Port: "2200", Key: "/private/key", KnownHosts: "/verified/hosts", Python: "/python with 'quote"}
	marker := "$(touch /must-not-run) `echo unsafe`"
	cmd := ssh.command(context.Background(), map[string]any{"graph": marker})
	args := strings.Join(cmd.Args, "\n")
	if !strings.Contains(args, "StrictHostKeyChecking=yes") || !strings.Contains(args, `UserKnownHostsFile="/verified/hosts"`) {
		t.Fatal(args)
	}
	if strings.Contains(args, marker) {
		t.Fatal("graph entered SSH argv")
	}
	raw, err := io.ReadAll(cmd.Stdin)
	if err != nil {
		t.Fatal(err)
	}
	line, _, ok := strings.Cut(string(raw), "\n")
	if !ok {
		t.Fatal("missing framing")
	}
	var payload map[string]string
	if err = json.Unmarshal([]byte(line), &payload); err != nil {
		t.Fatal(err)
	}
	if payload["graph"] != marker {
		t.Fatal(payload)
	}
}

func TestVerifiedArtifactPublicationNeverReplacesRacingOutput(t *testing.T) {
	root := t.TempDir()
	partial := filepath.Join(root, "video.partial")
	target := filepath.Join(root, "video.mp4")
	if _, err := os.Lstat(target); !os.IsNotExist(err) {
		t.Fatal(err)
	} // fetch's initial destination check
	if err := os.WriteFile(partial, []byte("verified original"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("another operation's video"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := publishVerified(partial, target); !errors.Is(err, os.ErrExist) {
		t.Fatal("racing output replaced", err)
	}
	got, _ := os.ReadFile(target)
	held, _ := os.ReadFile(partial)
	if string(got) != "another operation's video" || string(held) != "verified original" {
		t.Fatal("collision destroyed evidence")
	}
}

func TestInputDefaultsAndRefusals(t *testing.T) {
	input, err := ParseInput([]byte(`{"graph_json":"{\"1\":{}}","output_root":"/outputs"}`))
	if err != nil || input.Port != 8188 || input.TimeoutS != 3600 || input.ExpectedSteps != 8 {
		t.Fatalf("%+v %v", input, err)
	}
	for _, raw := range []string{`{"graph_json":"{}","output_root":"/out"}`, `{"graph_json":"{\"1\":{}}","output_root":"relative"}`, `{"graph_json":"{\"1\":{}}","output_root":"/out","port":0}`, `{"graph_json":"{\"1\":{}}","output_root":"/out","expected_steps":7}`, `{"graph_json":"{\"1\":{}}","output_root":"/out","network":true}`} {
		if _, err = ParseInput([]byte(raw)); err == nil {
			t.Fatal("accepted", raw)
		}
	}
}
