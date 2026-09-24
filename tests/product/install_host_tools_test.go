package producttest

import (
	"os"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// The installers, the README and every host-tool remedy name ONE command: the scripts
// cannot import Go, so this is where their copies are held to hostruntime.InstallCommand.
func TestInstallersRunTheHostToolCommand(t *testing.T) {
	words := strings.Fields(strings.ReplaceAll(hostruntime.InstallCommand, "'", ""))
	quoted := make([]string, 0, len(words)-1)
	for _, w := range words[1:] {
		quoted = append(quoted, "'"+w+"'")
	}
	want := map[string]string{
		"../../scripts/install.sh":  "HOST_TOOLS=(" + hostruntime.InstallCommand + ")",
		"../../scripts/install.ps1": "$hostTools = @(" + strings.Join(quoted, ", ") + ")",
		"../../README.md":           hostruntime.InstallCommand,
	}
	for path, line := range want {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(body), line) {
			t.Errorf("%s does not carry the host-tool command\nwant: %s", path, line)
		}
	}
}
