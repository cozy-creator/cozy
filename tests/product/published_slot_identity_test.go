package producttest

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/hub"
)

// workflowInterface is the default workflow release: a serving `generate` and a CPU job.
var workflowInterface = []byte(`{"application":"h3:app","entrypoints":[{"invocable":{"context":"ctx","defaults":{},"enum_members":{},"export":"generate","module":"h3","parameters":["steps"],"type_names":{}},"models":[{"class":"H3","component_use":{"condition_text":["text_encoder"],"decode_video":["video_vae"],"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[{"models":[{"class":"Source","component_use":{},"path":"long_form.models.source"}],"name":"long_form","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)

// publishWorkflowRelease replaces proof/h3@1.0.0 with a torch-free release whose serving
// entrypoint is invocable by a CPU `long_form` job (or with h.workflow's interface), and
// publishes its exact install plan and the bf16 checkpoint's exact-digest resolution.
// `degrees` are the sequence-parallel degrees its first serving model slot declares.
func publishWorkflowRelease(t *testing.T, h *ladderHub, degrees ...int) {
	t.Helper()
	raw := h.workflow
	if raw == nil {
		raw = workflowInterface
	}
	if len(degrees) > 0 {
		var doc map[string]any
		must(t, json.Unmarshal(raw, &doc))
		slot := doc["entrypoints"].([]any)[0].(map[string]any)["models"].([]any)[0].(map[string]any)
		slot["sequence_parallel"] = map[string]any{"degrees": degrees}
		var err error
		raw, err = json.Marshal(doc)
		must(t, err)
	}
	iface, err := canonical.NormalizeJCS(raw)
	must(t, err)
	exact := func(raw []byte) hub.ExactDocument {
		return hub.ExactDocument{CanonicalBytes: raw, Digest: mustSpell(raw), Length: int64(len(raw))}
	}
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = iface
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = mustSpell(iface)
	detail.Release.PackageInterfaceLength = int64(len(iface))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
	wheel := []byte("h3 wheel")
	plan := hub.PackageDownloadPlan{Release: "1.0.0",
		PackageConfig:    exact([]byte("[application]\nobject = \"h3:app\"\n")),
		PackageInterface: exact(iface),
		Pyproject:        exact([]byte("[project]\nname = \"h3\"\nversion = \"1.0.0\"\nrequires-python = \">=3.12\"\ndependencies = []\n")),
		UVLock:           exact([]byte("version = 1\nrequires-python = \">=3.12\"\n\n[[package]]\nname = \"h3\"\nversion = \"1.0.0\"\nsource = { editable = \".\" }\n")),
		Downloads: []hub.PackageInstallDownload{{Kind: "project_wheel", Path: "h3-1.0.0-py3-none-any.whl",
			Distribution: "h3", Version: "1.0.0", Digest: mustSpell(wheel), Length: int64(len(wheel)),
			Tags: []string{"py3-none-any"}, ImportRoots: []string{"h3"}}}}
	resolved := hub.ModelResolution{Model: ladderModel, ManifestID: bf16Manifest, ManifestLength: 161,
		Components:     []string{"audio_vae", "fl2va_dit", "ref2va_dit", "text_encoder", "video_vae"},
		ComponentBytes: h3BF16Components(), Bytes: 130 * gib}
	fallback := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body any
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/proof/h3/releases/1.0.0":
			body = detail
		case r.Method == http.MethodPost && r.URL.Path == "/v1/packages/proof/h3/download" && r.URL.Query().Get("release") == "1.0.0":
			body = plan
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models/resolve" && r.URL.Query().Get("ref") == ladderModel+"@"+bf16Manifest:
			body = resolved
		default:
			fallback.ServeHTTP(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(body)
	})
}
