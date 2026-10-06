package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// A gated source without a configured provider token is refused before renting with the
// credential it needs, never as an origin that cannot serve header ranges.
func TestGatedProviderSourceNamesTheMissingToken(t *testing.T) {
	root := filepath.Join(scratchBase, "gated-source")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = removeAllForce(root) })
	standIn := httptest.NewServer(http.NotFoundHandler())
	defer standIn.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+standIn.URL+"\n"), 0o600))
	for _, row := range []struct{ probe, source, token string }{
		{"https://civitai.com/api/download/models/889818", "civitai://889818", "civitai_token"},
		{"https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
			"hf://black-forest-labs/FLUX.1-dev@3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/flux1-dev.safetensors", "huggingface_token"},
	} {
		response, err := http.Get(row.probe)
		if err != nil {
			t.Skipf("provider unreachable from this runner: %v", err)
		}
		response.Body.Close()
		if response.StatusCode != http.StatusUnauthorized {
			t.Fatalf("%s is no longer gated: HTTP %d", row.probe, response.StatusCode)
		}
		cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "model", "upload", row.source, "proof/gated", "--rental-only", "--json")
		cmd.Env = childEnv(t, root, "TENSORHUB_TOKEN=operator-proof", "HF_TOKEN=", "CIVITAI_TOKEN=")
		out, _ := cmd.Output()
		var document struct {
			Error struct{ Code, Remedy string } `json:"error"`
		}
		if json.Unmarshal(out, &document) != nil || document.Error.Code != "model_source.auth_required" ||
			!strings.Contains(document.Error.Remedy, row.token) {
			t.Fatalf("%s must name the missing %s [exit %d]: %s", row.source, row.token, cmd.ProcessState.ExitCode(), out)
		}
	}
}
