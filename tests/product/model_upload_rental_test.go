package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Exercise the actual command parser and placement refusal without provider I/O,
// a daemon or an allocation. A named rental must never degrade to a boolean.
func TestModelUploadNamedRentalGrammar(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source.safetensors")
	must(t, os.WriteFile(source, []byte("local source"), 0600))
	for _, row := range []struct {
		flags []string
		want  string
	}{
		{[]string{"--rental=otter"}, "a local model source"},
		{[]string{"--rental=otter", "--rental-only"}, "named --rental"},
		{[]string{"--rental="}, "requires an existing rental name"},
	} {
		args := append([]string{"model", "upload", source, "proof/output"}, row.flags...)
		code, out := runCozy(t, root, args...)
		if code != 2 || !strings.Contains(out, row.want) {
			t.Fatalf("%v [%d]: %s", args, code, out)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatal("placement refusal started a daemon")
	}
}
