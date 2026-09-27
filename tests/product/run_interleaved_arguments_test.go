package producttest

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
)

func TestRunDescribeAcceptsPayloadAfterOptions(t *testing.T) {
	h := newLadderHub(t)
	root := ladderRoot(t, h)
	// This is the user's ordering: a prompt, two reference flags, more payload,
	// and placement/wait flags. Describe proves parsing without starting a job.
	args := []string{"run", ladderPackage + "/generate", "prompt=Picture-1.",
		"--asset=/missing/a=b.png", "--asset=/missing/c=d.png", "duration_s=14", "steps=30",
		"--rental-only", "--await", "--describe", "--json"}
	code, out := runCozy(t, root, args...)
	if code != 0 || !strings.Contains(out, `"name":"steps"`) {
		t.Fatalf("interleaved run did not reach describe: %d %s", code, out)
	}
	for _, invalid := range [][]string{
		append(append([]string{}, args...), "--unknown-option"),
		append(append([]string{}, args...), "--asset"),
		{"--json", "run", "list", "payload=not-a-run-input", "--limit", "1"},
		{"--json", "run", "watch", "1", "--asset", "a.png", "steps=30"},
	} {
		code, out := runCozy(t, root, invalid...)
		if code == 0 || !strings.Contains(out, `"code":"cli.usage"`) {
			t.Fatalf("normalization reinterpreted a non-run argument: %d %s", code, out)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatal("describe or parser refusal started a daemon")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.posts) != 0 {
		t.Fatal("describe attempted to buy a rental")
	}
}

// The real CLI records the exact composed payload with the admitted run, making order and
// literal-value preservation observable at admission.
func interleavedAssetsRoot(t *testing.T, extra ...map[string]any) string {
	t.Helper()
	var doc map[string]any
	must(t, json.Unmarshal([]byte(declaredAssetsInterface), &doc))
	job := doc["entrypoints"].([]any)[0].(map[string]any)
	job["name"], job["publishes"], job["weights_outputs"] = "prepare", false, []any{}
	fields := job["request"].(map[string]any)["fields"].([]any)
	for _, name := range []string{"steps", "seed"} {
		fields = append(fields, map[string]any{"name": name, "type": "int", "wire": "optional"})
	}
	for _, field := range extra {
		fields = append(fields, field)
	}
	job["request"].(map[string]any)["fields"] = fields
	doc["jobs"], doc["entrypoints"] = []any{job}, []any{}
	raw, err := json.Marshal(doc)
	must(t, err)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = raw
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = assessmentDigest(iface.Raw)
	detail.Release.PackageInterfaceLength = int64(len(raw))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/assets", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "assets"}, Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
	})
	mux.HandleFunc("GET /v1/packages/proof/assets/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(detail)
	})
	// Admitted runs are canceled once read; anything later in their lifecycle finds nothing.
	mux.HandleFunc("/", http.NotFound)
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	root, err := os.MkdirTemp(scratchBase, "submitted-run-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: proof\n"), 0600))
	return root
}

func TestInterleavedRunPreservesAssetsPayloadAndLiteralTail(t *testing.T) {
	root := interleavedAssetsRoot(t)
	images := make([]string, 2)
	for i, name := range []string{"a=b.png", "c=d.png"} {
		images[i] = filepath.Join(root, name)
		frame := image.NewRGBA(image.Rect(0, 0, 2, 2))
		frame.SetRGBA(0, 0, color.RGBA{R: uint8(i * 255), A: 255})
		file, err := os.Create(images[i])
		must(t, err)
		must(t, png.Encode(file, frame))
		must(t, file.Close())
	}
	prompt := "Picture=1, with literal --asset and --model.model=words"
	args := []string{"--json", "run", "proof/assets/prepare", "prompt=" + prompt,
		"--asset", "first=" + images[0], "steps=30", "--asset=second=" + images[1],
		"seed=24680", "--asset", "again=" + images[0], "--rental-only"}
	request, _, out := submitRun(t, root, "interleaved", args...)
	if request == nil {
		t.Fatalf("interleaved run was not admitted: %s", out)
	}
	var input struct {
		Prompt string
		Steps  int
		Seed   int
		Assets []struct{ Asset, Label string }
	}
	must(t, json.Unmarshal(request.Payload, &input))
	if input.Prompt != prompt || input.Steps != 30 || input.Seed != 24680 || len(input.Assets) != 3 {
		t.Fatalf("payload changed: %s", request.Payload)
	}
	labels := []string{input.Assets[0].Label, input.Assets[1].Label, input.Assets[2].Label}
	if !reflect.DeepEqual(labels, []string{"first", "second", "again"}) || input.Assets[0].Asset != input.Assets[2].Asset || input.Assets[0].Asset == input.Assets[1].Asset {
		t.Fatalf("asset occurrences reordered, collapsed, or changed: %s", request.Payload)
	}
	// --in consumes exactly its following value; '=' in that filename stays a
	// filename, while later payload terms still override its scalar values.
	infile := filepath.Join(root, "request=seed.json")
	must(t, os.WriteFile(infile, []byte(`{"prompt":"from file","steps":2}`), 0600))
	request, _, out = submitRun(t, root, "interleaved-in", "run", "proof/assets/prepare", "--in", infile, "steps=7",
		"--asset", images[0], "--rental-only", "--json")
	if request == nil || submittedPayload(t, request)["prompt"] != "from file" || submittedPayload(t, request)["steps"] != float64(7) {
		t.Fatalf("flag value or scalar override changed: %s", out)
	}
	base := []string{"--json", "run", "proof/assets/prepare", "--asset", images[0], "--rental-only"}
	request, _, out = submitRun(t, root, "interleaved-tail", append(append([]string{}, base...), "--", "--looks-like-a-flag", "steps=9")...)
	if request == nil || submittedPayload(t, request)["prompt"] != "--looks-like-a-flag" || submittedPayload(t, request)["steps"] != float64(9) {
		t.Fatalf("literal tail became options: %s", out)
	}
	code, out := runCozy(t, root, append(append([]string{}, base...), "prompt=literal", "--", "--model.model=literal")...)
	if code == 0 || !strings.Contains(out, `no request field`) || !strings.Contains(out, `--model.model`) {
		t.Fatalf("literal dashed model argument became an override: %d %s", code, out)
	}
}

func TestInterleavedRunKeepsExplicitModelOverrides(t *testing.T) {
	root, _, _, _, _ := runModelCatalog(t)
	for i, model := range []string{"model.dits=proof/source@1.0.0/bf16", "--model.dits=proof/source@1.0.0/bf16"} {
		request, _, out := submitRun(t, root, fmt.Sprintf("interleaved-model-%d", i), "--json", "run", "proof/quantize/quantize", "steps=7",
			"--rental-only", model, "--publish-to", "proof/output", "model.shared=proof/source@1.0.0/bf16")
		if request == nil || submittedPayload(t, request)["steps"] != float64(7) || len(request.Models) != 2 ||
			request.Models[0].Slot != "dits" || request.Models[1].Slot != "shared" {
			t.Fatalf("interleaved model override changed: %+v %s", request, out)
		}
	}
}
