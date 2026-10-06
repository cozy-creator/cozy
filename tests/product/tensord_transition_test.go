package producttest

import (
	"bytes"
	"context"
	"flag"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/machines"
)

var previousNativeHost = flag.String("previous-native-host", "", "previously named native executable for the tensord transition refusal proof")

func TestTensordRenameNeverReplacesAnExistingNativeJournal(t *testing.T) {
	if *previousNativeHost == "" {
		t.Skip("requires -previous-native-host pointing to an actual older-named native machine")
	}
	dir := t.TempDir()
	h := machines.NewHost(dir, "", nil)
	program := filepath.Join(h.Root(), "usr/local/bin/cozy-machine")
	binary, err := os.ReadFile(*previousNativeHost)
	must(t, err)
	must(t, os.MkdirAll(filepath.Dir(program), 0o700))
	must(t, os.WriteFile(program, binary, 0o700))
	saved := map[string][]byte{
		filepath.Join(dir, "installed.json"):                                              []byte(`{"host":{"name":"cozy-machine"}}`),
		filepath.Join(dir, "agent.json"):                                                  []byte(`{"pid":12345,"worker_id":"kept-native"}`),
		filepath.Join(h.Root(), "var/lib/cozy/rust-machine/execution/executions.sqlite3"): []byte("kept journal bytes"),
	}
	for path, raw := range saved {
		must(t, os.MkdirAll(filepath.Dir(path), 0o700))
		must(t, os.WriteFile(path, raw, 0o600))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, problem := h.Install(ctx, machines.Source{}, "uv")
	if problem == nil || problem.ErrName() != "machine.rename_required" {
		t.Fatalf("rename entered legacy replacement: %v", problem)
	}
	if _, problem = h.Start(ctx, nil); problem == nil || problem.ErrName() != "machine.rename_required" {
		t.Fatalf("new controller adopted an old-named native process: %v", problem)
	}
	if problem = h.Stop(ctx); problem == nil || problem.ErrName() != "machine.rename_required" {
		t.Fatalf("new controller discarded the old native launch record: %v", problem)
	}
	for path, before := range saved {
		after, err := os.ReadFile(path)
		must(t, err)
		if !bytes.Equal(before, after) {
			t.Fatalf("rename refusal altered %s", path)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "root.replaced")); !os.IsNotExist(err) {
		t.Fatalf("rename refusal created a legacy replacement: %v", err)
	}
}
