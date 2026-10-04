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
	write("tensorhub_url: http://127.0.0.1:1\ntensorhub_url: http://127.0.0.1:2\n")
	if code, stdout, stderr := runCozyStreams(t, root, "package", "list", "--json"); code == 0 || !strings.Contains(stdout+stderr, "different values") {
		t.Fatalf("a location repeated with different values was accepted [exit %d]: %s %s", code, stdout, stderr)
	}
}

// An unusable value for a behaviour setting never disables a command: it is named once
// on stderr with the default that applies instead.
func TestUnusableBehaviourSettingsFallBackToTheirDefaults(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("port: 7433\nport: 7434\nyield: sometimes\n"+
		"local_rate_micro_usd_per_hour: -3\nrentals:\n  development: maybe\ndaemon:\n  idle_shutdown_s: soon\n"+
		"maintenance:\n  gc_cron: nightly\nplacement:\n  prefer: fastest\n"), 0600))
	code, stdout, stderr := runCozyStreams(t, root, "package", "list", "--json")
	if code != 0 || !strings.Contains(stdout, `"packages"`) {
		t.Fatalf("an unusable behaviour value disabled the command [exit %d]: %s %s", code, stdout, stderr)
	}
	for _, line := range []string{
		`port "7433" is set more than once with different values; using "8818"`,
		`yield "sometimes" is not smart, always or never; using "smart"`,
		`local_rate_micro_usd_per_hour "-3" is not a non-negative integer; using "0"`,
		`rentals.development "maybe" is not true or false; using "true"`,
		`daemon.idle_shutdown_s "soon" is not a non-negative integer; using "900"`,
		`this version does not use maintenance; ignored`, // the store GC schedule, removed
		`placement.prefer "fastest" is not fast, balanced or cheap; using "balanced"`,
	} {
		if strings.Count(stderr, line) != 1 {
			t.Fatalf("stderr does not name %q exactly once:\n%s", line, stderr)
		}
	}
}
