package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Runs 1170/1171 and 1173 of paul/qwen-image-2/generate_image (2026-09-26). The release's
// requirements name no torch, so the published child walk ran for an accelerator package
// and re-contributed its own slot as a child of itself: `generate_image/model` beside
// `generate_image.models.model`. The download set refused the identical pair, and an exact
// digest override rode beside the default ladder until the pod refused both for one slot.
//
// Each arm drives the REAL binary and daemon against the stand-in hub through the paid ask,
// then hands the pinned selection to a real orchestrator and a stand-in pod, so what the
// pod receives is exactly what the daemon's dispatch would converge.
func TestPublishedRentalSelectsOneModelPerSlot(t *testing.T) {
	for _, arm := range []struct {
		name, target, override string
		// want is each selected slot's manifest, and callable the slot a callee contributes.
		want     map[string]string
		callable string
	}{
		{name: "default", target: "generate", want: map[string]string{"generate.models.model": fp8Manifest}},
		{name: "exact override", target: "generate", override: "model.model=" + ladderModel + "#" + bf16Manifest,
			want: map[string]string{"generate.models.model": bf16Manifest}},
		// A CPU workflow whose own Model input is derive-only still holds capacity for its
		// published callee's default, chosen for the machine it buys.
		{name: "cpu workflow", target: "long_form", override: "model.source=" + ladderModel + "@" + ladderRelease + "/" + ladderLane,
			want: map[string]string{"long_form.models.source": fp8Manifest, "generate.models.model": fp8Manifest}, callable: "generate.models.model"},
	} {
		t.Run(arm.name, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			publishWorkflowRelease(t, h)
			root := ladderRoot(t, h)
			startDaemonProcess(t, root)
			key := "slot-identity-" + strings.ReplaceAll(arm.name, " ", "-")
			args := []string{"run", ladderPackage + "/" + arm.target}
			if arm.override != "" {
				args = append(args, arm.override)
			}
			if arm.target == "generate" {
				args = append(args, "steps=1")
			}
			_, out := runCozy(t, root, append(args, "--rental-only", "--json", "--idempotency-key", key)...)
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			row, problem := store.RequestByIdempotencyKey(key)
			fatal(t, problem)
			if row == nil {
				t.Fatalf("the run was not submitted: %s", out)
			}
			waitFor(t, root, "the first paid ask", func() bool { return len(h.postedSKUs()) > 0 })
			row, problem = store.RequestByIdempotencyKey(key)
			fatal(t, problem)
			if row.NeedsAccelerator {
				t.Fatal("the fixture release names torch; the arm must reproduce a torch-free release")
			}
			if sku := strings.Split(h.postedSKUs()[0], "/")[0]; sku == "cpu" || sku == "rtx-4090" || sku == "rtx-5090" {
				t.Fatalf("bought %s for a selection that needs a 51.5 GiB text encoder on a GPU", sku)
			}
			got := map[string]string{}
			for _, model := range row.Models {
				if _, repeated := got[model.BindingSlot()]; repeated {
					t.Fatalf("slot %s is selected twice: %+v", model.BindingSlot(), row.Models)
				}
				got[model.BindingSlot()] = model.Manifest
				if (model.BindingSlot() == arm.callable) != (model.Callable != "") {
					t.Fatalf("slot %s callable %q; only a callee's default is contributed", model.BindingSlot(), model.Callable)
				}
			}
			if fmt.Sprint(got) != fmt.Sprint(arm.want) {
				t.Fatalf("selected %v; want %v", got, arm.want)
			}
			if log := tail(filepath.Join(root, "daemon.log")); strings.Contains(log, "/model=") {
				t.Fatalf("the placement names a second spelling of one slot:\n%s", log)
			}

			pod := &standInPod{}
			o, instance := attachStandInRental(t, key, pod)
			fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: ladderPackage, Release: "1.0.0"}},
				orchestrator.DownloadModelRefs(row.Models)))
			var delivered pb.DownloadDelegation
			must(t, canonical.Unmarshal(pod.await(t, 1, 30*time.Second)[0].document, &delivered))
			received := map[string]string{}
			for _, model := range delivered.Models {
				if _, repeated := received[model.Slot]; repeated {
					t.Fatalf("the pod was handed slot %s twice", model.Slot)
				}
				received[model.Slot] = model.Manifest
			}
			if fmt.Sprint(received) != fmt.Sprint(arm.want) {
				t.Fatalf("the pod received %v; want %v", received, arm.want)
			}
		})
	}
}

// publishWorkflowRelease replaces proof/h3@1.0.0 with a torch-free release whose serving
// entrypoint is invocable by a CPU `long_form` job, and publishes its exact install plan
// and the bf16 checkpoint's exact-digest resolution.
func publishWorkflowRelease(t *testing.T, h *ladderHub) {
	t.Helper()
	iface, err := canonical.NormalizeJCS([]byte(`{"application":"h3:app","entrypoints":[{"invocable":{"context":"ctx","defaults":{},"enum_members":{},"export":"generate","module":"h3","parameters":["steps"],"type_names":{}},"models":[{"class":"H3","component_use":{"condition_text":["text_encoder"],"decode_video":["video_vae"],"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[{"models":[{"class":"Source","component_use":{},"path":"long_form.models.source"}],"name":"long_form","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`))
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
