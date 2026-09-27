package producttest

import (
	"bytes"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// The real CLI/API/retained-request boundary against an isolated catalog that
// serves exact package metadata, offers no rental SKU and refuses every paid POST.
func TestH3ReferenceSubmissionRetainsDeclaredOutputTypes(t *testing.T) {
	submitH3Reference(t, "1.1.3")
}

// 1.1.2 declared bare VideoAsset/ImageAsset results; export needs an exact MIME.
func TestH3ReferenceSubmissionRefusesUntypedOutputsBeforeQueueing(t *testing.T) {
	submitH3Reference(t, "1.1.2")
}

func submitH3Reference(t *testing.T, release string) {
	typedOutputs := release != "1.1.2"
	fixtureRoot := t.TempDir()
	payloadPath := filepath.Join(fixtureRoot, "request.json")
	must(t, os.WriteFile(payloadPath, []byte(`{"prompt":"A red fox in the snow","seed":12345,"mute":true,"references":[{"type":"image"},{"type":"image"}]}`), 0600))
	createImage := func(name string) string {
		path := filepath.Join(fixtureRoot, name)
		f, err := os.Create(path)
		must(t, err)
		must(t, png.Encode(f, image.NewRGBA(image.Rect(0, 0, 64, 64))))
		must(t, f.Close())
		return path
	}
	image0, image1 := createImage("first.png"), createImage("second.png")
	raw, err := os.ReadFile("testdata/h3/package-interface-" + release + ".json")
	must(t, err)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	for _, function := range []string{"first_last_frame_to_video", "reference_media_to_video"} {
		ep, problem := iface.Function(function)
		fatal(t, problem)
		for output, mime := range map[string]string{"video": "video/mp4", "continuation_frame": "image/png"} {
			spec, ok := launch.ResultAssetSpec(ep, output)
			if typedOutputs && (!ok || len(spec.MediaTypes) != 1 || spec.MediaTypes[0] != mime) {
				t.Fatalf("%s/%s does not declare %s: %+v", function, output, mime, spec)
			}
		}
	}
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = iface.Raw
	detail.Release.Release = release
	detail.Release.PackageInterfaceDigest = assessmentDigest(iface.Raw)
	detail.Release.PackageInterfaceLength = int64(len(iface.Raw))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.28", "torch>=2.13,<2.14"}
	detail.RequiresPython = ">=3.12,<3.13"
	var paidPosts atomic.Int64
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/paul/minimax-h3", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "paul", Name: "minimax-h3"}, Releases: []hub.ReleaseSummary{{Release: release}}})
	})
	mux.HandleFunc("GET /v1/packages/paul/minimax-h3/releases/"+release, func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(detail) })
	mux.HandleFunc("GET /v1/packages/paul/minimax-h3/bindings", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"bindings":[]}`)) })
	const manifest = "sha256:c3b72104aff017e77fc9f5b427c8cd0c5f8dc7c4cbe604584e89f9fc24f1025a"
	mux.HandleFunc("GET /v1/models/paul/minimax-h3", func(w http.ResponseWriter, _ *http.Request) {
		card := hub.ModelCard{
			Model: hub.Resource{Org: "paul", Name: "minimax-h3"},
			Releases: []hub.ModelReleaseSummary{{
				ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0-rc.1"},
				Lanes: []hub.ModelLaneSummary{{Lane: "bf16-full", ManifestID: manifest,
					Bytes:      195021215719,
					Components: []string{"fl2va_dit", "ref2va_dit", "text_encoder", "video_vae", "audio_vae"},
				}},
			}},
		}
		_ = json.NewEncoder(w).Encode(card)
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"code":"proof.capacity_held","message":"capacity held by fixture"}}`))
	})
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"rentals":[]}`)) })
	mux.HandleFunc("POST /v1/rentals", func(w http.ResponseWriter, _ *http.Request) {
		paidPosts.Add(1)
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":{"code":"proof.no_paid_create","message":"no rental authorized"}}`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: h3-contract-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	startDaemonProcess(t, root)
	args := []string{"run", "paul/minimax-h3/reference_media_to_video", "model.model=paul/minimax-h3@1.0.0-rc.1/bf16-full",
		"--in", payloadPath, "--asset", "references.0.image=" + image0, "--asset", "references.1.image=" + image1,
		"--rental-only", "--out", filepath.Join(root, "output"), "--idempotency-key", "h3-output-contract-proof", "--json", "--full"}
	code, output := runCozy(t, root, args...)
	// Capacity is deliberately unavailable; the front door still accepts and
	// retains the request while the existing scheduler waits for a later retry.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("h3-output-contract-proof")
	fatal(t, problem)
	if !typedOutputs {
		if row != nil || code == 0 || !strings.Contains(output, "output_export_media_type_ambiguous") || paidPosts.Load() != 0 {
			t.Fatalf("untyped outputs did not refuse before queueing: exit=%d row=%v output=%s", code, row, output)
		}
		return
	}
	if code != 0 || row == nil {
		t.Fatalf("real CLI/API refused before recording the request: exit=%d %s", code, output)
	}
	if row.State != "submitted" || row.Package != "paul/minimax-h3" || row.Entrypoint != "reference_media_to_video" || row.Release != release || len(row.Assets) != 2 || len(row.Models) != 1 || row.Models[0].Manifest != manifest {
		t.Fatalf("retained request identities differ: %+v", row)
	}
	for i, path := range []string{image0, image1} {
		facts, problem := inputasset.Fingerprint(path, 64<<20)
		fatal(t, problem)
		asset := row.Assets[i]
		if asset.Digest != facts.Digest || asset.Length != facts.Length || asset.MediaType != facts.MediaType || asset.LocalPath != path || asset.Order != uint32(i) || asset.ModTime != facts.ModTime {
			t.Fatalf("reference image was not borrowed under its exact identity: %+v", asset)
		}
		fatal(t, inputasset.Verify(asset, 64<<20))
	}
	export, problem := store.OutputExportOf(row.ID)
	fatal(t, problem)
	if export == nil || len(export.Outputs) != 2 {
		t.Fatalf("result export obligation absent: %+v", export)
	}
	got := map[string]string{}
	for _, entry := range export.Outputs {
		got[entry.OutputID] = entry.MediaType
	}
	if got["video"] != "video/mp4" || got["continuation_frame"] != "image/png" {
		t.Fatalf("wrong output contract: %+v", got)
	}
	entrypoint, problem := iface.Function("reference_media_to_video")
	fatal(t, problem)
	expected, _, problem := launch.ParsePayload(entrypoint, nil, payloadPath)
	fatal(t, problem)
	expected, _, problem = launch.ParseAssets(entrypoint, expected, []string{"references.0.image=" + image0, "references.1.image=" + image1}, nil, nil)
	fatal(t, problem)
	if !bytes.Equal(expected, row.Payload) {
		t.Fatalf("payload changed: %s != %s", row.Payload, expected)
	}
	attempts, problem := store.Attempts(row.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("metadata admission invented an execution attempt")
	}
	if paidPosts.Load() != 0 {
		t.Fatal("the isolated proof reached paid acquisition")
	}
	t.Logf("retained request=%s interface=%s images=2 video=video/mp4 continuation=image/png; no attempt or rental", row.ID, assessmentDigest(iface.Raw))
}
