package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// A config file written for another Cozy version keeps every command working: a retired
// or newer key is named once on stderr and ignored, and an identical repeated key is one
// setting. Only a key repeated with different values is ambiguous.
func TestConfigFileFromAnotherVersionKeepsCommandsWorking(t *testing.T) {
	root := t.TempDir()
	write := func(body string) {
		must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(body), 0600))
	}
	write("port: 7433\nport: 7433\nfuture_setting: on\nrentals:\n  max_hourly_spend_usd: 10\n  development: false\n")
	code, stdout, stderr := runCozyStreams(t, root, "package", "list", "--json")
	if code != 0 || !strings.Contains(stdout, `"packages"`) {
		t.Fatalf("a stale config key disabled the command [exit %d]: %s %s", code, stdout, stderr)
	}
	for _, key := range []string{"future_setting", "rentals.max_hourly_spend_usd"} {
		if !strings.Contains(stderr, key) {
			t.Fatalf("ignored key %s was not named: %s", key, stderr)
		}
	}
	write("port: 7433\nport: 7434\n")
	if code, stdout, stderr := runCozyStreams(t, root, "package", "list", "--json"); code == 0 || !strings.Contains(stdout+stderr, "different values") {
		t.Fatalf("a key repeated with different values was accepted [exit %d]: %s %s", code, stdout, stderr)
	}
}
