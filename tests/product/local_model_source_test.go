package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/modelsource"
)

// A local file kept under a local alias is an installation on this computer's machine that
// names the file by its content (object://sha256:<hex>/<name>) and the exact path written to
// the machine; noncanonical file spellings are refused.
func TestLocalModelSourceStaysLocalAfterCanonicalization(t *testing.T) {
	root, _, _, _, _ := runModelCatalog(t)
	source := filepath.Join(root, "fixture.safetensors")
	body := []byte("small local source")
	must(t, os.WriteFile(source, body, 0o600))
	startDaemonProcess(t, root)
	accepted := queuedTransfer(t, root, "model", "download", source, "local/classification-proof", "--json", "--full")
	sum := sha256.Sum256(body)
	if len(accepted.Models) != 1 || accepted.Models[0].Source != "object://sha256:"+hex.EncodeToString(sum[:])+"/fixture.safetensors" ||
		accepted.Write != source || accepted.Destination != "local/classification-proof" {
		t.Fatalf("the local file lost its exact identity at submission: %+v", accepted)
	}
	for _, spelling := range []string{"file:./fixture.safetensors", "file://" + source} {
		if _, problem := modelsource.Parse(spelling, root); problem == nil {
			t.Fatalf("noncanonical file identity accepted: %q", spelling)
		}
	}
}
