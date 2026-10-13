package producttest

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"
)

// TestRentedIngestNeedsNoRecipe plans ingests of real provider sources that have no Runtime
// recipe, as `cozy rental new --model` sizes a rental for them: the owner-side header preflight
// picks the reviewed TensorFS profiles, an ambiguous source names them, and several profiles of
// one source compose one model. The stand-in hub refuses the paid request; nothing is rented.
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
	root, _ := rentalCapture(t)
	const pdd = "hf://alibaba-pai/MiniMax-H3-Acc-LoRAs@335001fb9e5455d68a0caa18ec2e319072150328"
	plan := func(source string, args ...string) (int, map[string]any) {
		t.Helper()
		out, code := rentalForSource(t, root, nil, source, args...)
		var document map[string]any
		if err := json.Unmarshal([]byte(lastJSONLine(string(out))), &document); err != nil {
			t.Fatalf("rental new --model %s printed no JSON document: %v\n%s", source, err, out)
		}
		return code, document
	}

	code, document := plan(pdd)
	message, _ := json.Marshal(document["error"])
	if code == 0 || !strings.Contains(string(message), "AMBIGUOUS_CLASSIFICATION") ||
		!strings.Contains(string(message), "--source-profile") {
		t.Fatalf("an ambiguous source must name its profiles and the flag [exit %d]: %s", code, message)
	}
	// Past the preflight the rental is asked for; whatever the stand-in hub then refuses,
	// it is no longer a source-profile or header refusal.
	planned := func(code int, document map[string]any) bool {
		failure, _ := json.Marshal(document["error"])
		return code == 0 || !strings.Contains(string(failure), "AMBIGUOUS_CLASSIFICATION") &&
			!strings.Contains(string(failure), "model_source") && !strings.Contains(string(failure), "source_profile") &&
			!strings.Contains(string(failure), "slot=profile") && !strings.Contains(string(failure), "UNREGISTERED_FINGERPRINT")
	}
	code, document = plan(pdd, "--source-profile", "hf/minimax-h3/pdd-fl2va-bf16/1",
		"--source-profile", "hf/minimax-h3/pdd-ref2va-bf16/1")
	if !planned(code, document) {
		t.Fatalf("two reviewed profiles of one source must plan one composed ingest [exit %d]: %v", code, document)
	}
	code, document = plan("hf://MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe",
		"--source-profile", "hf/minimax-h3/native-dual-bf16/1", "--source-profile", "hf/minimax-h3/shared-bf16/1")
	if !planned(code, document) {
		t.Fatalf("H3's two reviewed profiles must plan one composed ingest [exit %d]: %v", code, document)
	}
	code, document = plan("civitai://128078", "--source-profile", "civitai/sdxl/single-file/1")
	if !planned(code, document) {
		t.Fatalf("a Civitai source must plan a rented ingest [exit %d]: %v", code, document)
	}
}
