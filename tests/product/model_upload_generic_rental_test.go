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
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// TestRentedIngestNeedsNoRecipe ingests real provider sources that have no Runtime recipe:
// the owner-side header preflight picks the reviewed TensorFS profiles, an ambiguous source
// names them, and several profiles of one source compose one model. The stand-in hub serves
// nothing, so a source that passes the preflight goes on to submission and nothing is rented.
func TestRentedIngestNeedsNoRecipe(t *testing.T) {
	fullRun(t, "plans ingests against live HuggingFace and Civitai")
	probe := &http.Client{Timeout: 10 * time.Second}
	for _, url := range []string{"https://huggingface.co/api/models/alibaba-pai/MiniMax-H3-Acc-LoRAs",
		"https://civitai.com/api/v1/model-versions/128078"} {
		response, err := probe.Get(url)
		if err != nil {
			t.Skipf("provider unreachable from this runner: %v", err)
		}
		response.Body.Close()
	}
	root := filepath.Join(scratchBase, "generic-rental-ingest")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = removeAllForce(root) })
	standIn := httptest.NewServer(http.NotFoundHandler())
	defer standIn.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+standIn.URL+"\n"), 0o600))
	const pdd = "hf://alibaba-pai/MiniMax-H3-Acc-LoRAs@335001fb9e5455d68a0caa18ec2e319072150328"
	plan := func(args ...string) (int, map[string]any) {
		t.Helper()
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin, "model", "upload"},
			append(args, "--rental-only", "--json")...)...)
		// An operator identity publishes to its named org without a Hub account read.
		cmd.Env = childEnv(t, root, "TENSORHUB_TOKEN=operator-proof")
		out, _ := cmd.Output()
		var document map[string]any
		if err := json.Unmarshal(out, &document); err != nil {
			t.Fatalf("model upload %v printed no JSON document: %v\n%s", args, err, out)
		}
		return cmd.ProcessState.ExitCode(), document
	}

	code, document := plan(pdd, "proof/pdd")
	message, _ := json.Marshal(document["error"])
	if code == 0 || !strings.Contains(string(message), "AMBIGUOUS_CLASSIFICATION") ||
		!strings.Contains(string(message), "--source-profile") {
		t.Fatalf("an ambiguous source must name its profiles and the flag [exit %d]: %s", code, message)
	}
	// Past the preflight the ingest is submitted; whatever the stand-in hub then refuses,
	// it is no longer a source-profile or header refusal.
	planned := func(code int, document map[string]any) bool {
		failure, _ := json.Marshal(document["error"])
		return code == 0 || !strings.Contains(string(failure), "AMBIGUOUS_CLASSIFICATION") &&
			!strings.Contains(string(failure), "model_source") && !strings.Contains(string(failure), "source_profile") &&
			!strings.Contains(string(failure), "slot=profile") && !strings.Contains(string(failure), "UNREGISTERED_FINGERPRINT")
	}
	code, document = plan(pdd, "proof/pdd", "--source-profile", "hf/minimax-h3/pdd-fl2va-bf16/1",
		"--source-profile", "hf/minimax-h3/pdd-ref2va-bf16/1")
	if !planned(code, document) {
		t.Fatalf("two reviewed profiles of one source must plan one composed ingest [exit %d]: %v", code, document)
	}
	code, document = plan("hf://MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe", "proof/h3",
		"--source-profile", "hf/minimax-h3/native-dual-bf16/1", "--source-profile", "hf/minimax-h3/shared-bf16/1")
	if !planned(code, document) {
		t.Fatalf("H3's two reviewed profiles must plan one composed ingest [exit %d]: %v", code, document)
	}
	code, document = plan("civitai://128078", "proof/sdxl", "--source-profile", "civitai/sdxl/single-file/1")
	if !planned(code, document) {
		t.Fatalf("a Civitai source must plan a rented ingest [exit %d]: %v", code, document)
	}
}
