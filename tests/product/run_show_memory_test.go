package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A run's device memory is readable from the CLI alone, on a local machine or a rental: its
// torch peaks and each stage's, the executor process at the end, what the weight plane paged
// in and evicted since the executor's previous call, and a refused stage's shortfall. The
// bundle is run 5886's (SDXL on an RTX 4000 Ada held to an 8 GiB pool); its refusal is run
// 5686's (qwen-image-2's decode on an RTX 3070).
func TestRunShowPrintsItsMemory(t *testing.T) {
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "sdxl-run-5886-memory.json"))
	must(t, err)
	var evidence map[string]map[string]json.RawMessage
	must(t, json.Unmarshal(bundle, &evidence))
	var memory map[string]any
	must(t, json.Unmarshal(evidence["measurements"]["memory"], &memory))
	memory["shortfall"] = map[string]any{"resource": "vram", "needed_bytes": 1288490188,
		"available_bytes": 1234803097, "request_shape": "vae.decode"}
	evidence["measurements"]["memory"], err = json.Marshal(memory)
	must(t, err)
	bundle, err = json.Marshal(evidence)
	must(t, err)
	o := hostOwner(t, "run-show-memory")
	id := succeededWithTriage(t, o, bundle)
	defer publicationControlAPI(t, o)()
	code, human := runCozy(t, o.root, "run", "show", id)
	t.Logf("cozy run show:\n%s", human)
	for _, want := range []string{
		"\nmemory\n",
		"  peak       1.2GiB allocated · 1.2GiB reserved · stage activations decode 1.2GiB, denoise 464.0MiB, encode 185.9MiB\n",
		"  at end     process 6.4GiB · context 232.0MiB · torch 194.0MiB · weights 6.0GiB · budget 6.0GiB · cap 7.5GiB\n",
		"  weights    paged in 798.3MiB · from disk 798.3MiB · evicted 42\n",
		"  allocator  cache releases 1 (1.0GiB)\n",
		"  refused    vram for vae.decode: needed 1.2GiB, 1.1GiB available\n",
		"  mode       vae.decode chunked 32\n",
	} {
		if code != 0 || !strings.Contains(human, want) {
			t.Fatalf("run show [%d] lacks %q:\n%s", code, want, human)
		}
	}
	code, out := runCozy(t, o.root, "run", "show", id, "--json")
	var report struct {
		Memory struct {
			PagedInBytes int64            `json:"paged_in_bytes"`
			Evictions    int64            `json:"evictions"`
			StagePeaks   map[string]int64 `json:"stage_peaks"`
			Shortfall    struct {
				NeededBytes int64 `json:"needed_bytes"`
			} `json:"shortfall"`
		} `json:"memory"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &report) != nil || report.Memory.PagedInBytes != 837099784 ||
		report.Memory.Evictions != 42 || report.Memory.StagePeaks["decode"] != 1304070656 ||
		report.Memory.Shortfall.NeededBytes != 1288490188 {
		t.Fatalf("run show --json [%d] lacks the memory record: %+v\n%s", code, report.Memory, out)
	}
}
