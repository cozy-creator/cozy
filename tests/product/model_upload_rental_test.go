package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A named rental this host does not hold is refused, and a named rental must never degrade
// to a boolean; both refuse without Hub or provider I/O, a daemon or an allocation.
func TestModelUploadNamedRentalGrammar(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source.safetensors")
	must(t, os.WriteFile(source, []byte("local source"), 0600))
	for _, row := range []struct {
		flags []string
		code  int
		want  string
	}{
		{[]string{"--rental=otter"}, 1, `no rental \"otter\" on this host`},
		{[]string{"--rental="}, 2, "requires an existing rental name"},
	} {
		args := append([]string{"model", "upload", source, "proof/output", "--json"}, row.flags...)
		code, out := runCozy(t, root, args...)
		if code != row.code || !strings.Contains(out, row.want) {
			t.Fatalf("%v [%d]: %s", args, code, out)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatal("placement refusal started a daemon")
	}
}
