package producttest

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestRentedIngestNeedsNoRecipe plans rented ingests of real provider sources that have no
// Runtime recipe: the owner-side header preflight picks the reviewed TensorFS profiles, an
// ambiguous source names them, and several profiles of one source compose one model. It
// reads only provider metadata and headers; nothing is rented or uploaded.
func TestRentedIngestNeedsNoRecipe(t *testing.T) {
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
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	const pdd = "hf://alibaba-pai/MiniMax-H3-Acc-LoRAs@335001fb9e5455d68a0caa18ec2e319072150328"
	plan := func(args ...string) (int, map[string]any) {
		t.Helper()
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin, "model", "upload"},
			append(args, "--rental-only", "--dry-run", "--json")...)...)
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
	code, document = plan(pdd, "proof/pdd", "--source-profile", "hf/minimax-h3/pdd-fl2va-bf16/1",
		"--source-profile", "hf/minimax-h3/pdd-ref2va-bf16/1")
	profiles, _ := json.Marshal(document["source_profiles"])
	if code != 0 || document["status"] != "planned" ||
		string(profiles) != `["hf/minimax-h3/pdd-fl2va-bf16/1","hf/minimax-h3/pdd-ref2va-bf16/1"]` {
		t.Fatalf("two reviewed profiles of one source must plan one composed ingest [exit %d]: %v", code, document)
	}
	code, document = plan("civitai://128078", "proof/sdxl", "--source-profile", "civitai/sdxl/single-file/1")
	if code != 0 || document["status"] != "planned" {
		t.Fatalf("a Civitai source must plan a rented ingest [exit %d]: %v", code, document)
	}
}
