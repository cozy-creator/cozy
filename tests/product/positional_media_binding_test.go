package producttest

import (
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// positionalMedia writes one sniffable file of each media kind a package input declares.
func positionalMedia(t *testing.T) (dir, picture, clip, sound string) {
	t.Helper()
	dir = t.TempDir()
	picture, clip, sound = filepath.Join(dir, "picture.png"), filepath.Join(dir, "clip.mp4"), filepath.Join(dir, "sound.wav")
	f, err := os.Create(picture)
	must(t, err)
	must(t, png.Encode(f, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	must(t, f.Close())
	must(t, os.WriteFile(clip, append([]byte("\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"), make([]byte, 64)...), 0600))
	must(t, os.WriteFile(sound, append([]byte("RIFF\x24\x00\x00\x00WAVEfmt "), make([]byte, 64)...), 0600))
	return dir, picture, clip, sound
}

func positionalCallable(t *testing.T, fixture, function string) *launch.Entrypoint {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata/positional", fixture+".json"))
	must(t, err)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	ep, problem := iface.Function(function)
	fatal(t, problem)
	return ep
}

// bindPositionals is the CLI's own payload and asset composition over terms: the payload, each
// bound field path's file name, or the refusal.
func bindPositionals(ep *launch.Entrypoint, terms ...string) (map[string]any, map[string]string, string) {
	payload, keys, problem := launch.ParsePayload(ep, terms, "")
	var assets []records.AssetBinding
	if problem == nil {
		payload, assets, problem = launch.ParseAssets(ep, payload, nil, keys.Files, nil, nil)
	}
	if problem != nil {
		return nil, nil, problem.ErrName() + ": " + problem.Message + " " + problem.Remedy
	}
	var document map[string]any
	_ = json.Unmarshal(payload, &document)
	bound := map[string]string{}
	for _, asset := range assets {
		bound[asset.FieldPath] = filepath.Base(asset.LocalPath)
	}
	return document, bound, ""
}

// A positional file fills the one input that declares its media kind, over the real describe
// output of each package; it never fills a string or enum field, and an ambiguous or
// unmatched file is refused with the inputs named.
func TestPositionalFileBindsTheInputOfItsMediaKind(t *testing.T) {
	dir, picture, clip, sound := positionalMedia(t)
	upscale, ref2va, fl2va := positionalCallable(t, "seedvr2", "upscale"), positionalCallable(t, "minimax-h3", "ref2va"), positionalCallable(t, "minimax-h3", "fl2va")
	qwen, describer := positionalCallable(t, "qwen-image-2", "generate_image"), positionalCallable(t, "character-describer", "caption_images")
	document, bound, refused := bindPositionals(upscale, clip)
	if refused != "" || bound["video"] != "clip.mp4" || len(bound) != 1 || document["color_correction"] != nil {
		t.Fatalf("seedvr2 video did not fill its video input: %v %v %s", document, bound, refused)
	}
	// A bare name in the working directory is a file too; text stays the first text field.
	t.Chdir(dir)
	if document, bound, refused = bindPositionals(upscale, "clip.mp4", "wavelet"); refused != "" || bound["video"] != "clip.mp4" || document["color_correction"] != "wavelet" {
		t.Fatalf("relative video or enum text misbound: %v %v %s", document, bound, refused)
	}
	for _, args := range [][]string{{picture}, {"./missing.mp4"}, {clip, clip}} {
		if _, _, refused = bindPositionals(upscale, args...); refused == "" || strings.Contains(refused, "color_correction") {
			t.Fatalf("%v reached a non-media field or was accepted: %s", args, refused)
		}
	}
	if _, _, refused = bindPositionals(upscale, picture); !strings.Contains(refused, "positional_file_unbound") || !strings.Contains(refused, "video (video)") {
		t.Fatalf("an image for a video-only callable did not name its media input: %s", refused)
	}

	// minimax-h3's Assets collection takes each kind it declares, in order; fl2va takes images only.
	document, bound, refused = bindPositionals(ref2va, "A fox runs.", picture, sound, clip, "duration_s=15")
	assets, _ := document["assets"].([]any)
	if refused != "" || document["prompt"] != "A fox runs." || len(assets) != 3 ||
		bound["assets.0.asset"] != "picture.png" || bound["assets.1.asset"] != "sound.wav" || bound["assets.2.asset"] != "clip.mp4" {
		t.Fatalf("h3 ref2va media did not fill its Assets collection: %v %v %s", document, bound, refused)
	}
	if _, bound, refused = bindPositionals(fl2va, "A fox.", picture, picture, "duration_s=15"); refused != "" || len(bound) != 2 {
		t.Fatalf("h3 fl2va first/last frames misbound: %v %s", bound, refused)
	}
	if _, _, refused = bindPositionals(fl2va, "A fox.", clip, "duration_s=15"); !strings.Contains(refused, "positional_file_unbound") || !strings.Contains(refused, "assets (image)") {
		t.Fatalf("h3 fl2va accepted a video: %s", refused)
	}

	// Lists of one media kind grow by one item per file.
	document, bound, refused = bindPositionals(qwen, "A lighthouse.", picture, picture)
	if refused != "" || document["prompt"] != "A lighthouse." || bound["reference_images.0"] != "picture.png" || bound["reference_images.1"] != "picture.png" {
		t.Fatalf("qwen reference images misbound: %v %v %s", document, bound, refused)
	}
	if _, bound, refused = bindPositionals(describer, picture, picture, picture); refused != "" || len(bound) != 3 || bound["images.2"] != "picture.png" {
		t.Fatalf("describer images misbound: %v %s", bound, refused)
	}

	// Two inputs that take images equally: refused with both named, unless one is already bound.
	declared := assetsCallable(t)
	if _, _, refused = bindPositionals(declared, "prompt", picture); !strings.Contains(refused, "positional_file_ambiguous") || !strings.Contains(refused, "assets and poster") || !strings.Contains(refused, "--asset poster="+picture) {
		t.Fatalf("an ambiguous image did not name its candidates: %s", refused)
	}
	payload, keys, problem := launch.ParsePayload(declared, []string{"prompt", picture}, "")
	fatal(t, problem)
	_, explicit, problem := launch.ParseAssets(declared, payload, []string{"poster=" + picture}, keys.Files, nil, nil)
	fatal(t, problem)
	if len(explicit) != 2 || explicit[0].FieldPath != "assets.0.asset" || explicit[1].FieldPath != "poster" {
		t.Fatalf("the remaining free image input was not chosen: %+v", explicit)
	}
}

// The real CLI submits published packages with positional media: SeedVR2's video reaches its
// video input (not color_correction) and minimax-h3's job takes its images and refuses a video.
func TestRunPositionalMediaThroughTheCLI(t *testing.T) {
	_, picture, clip, _ := positionalMedia(t)
	const manifest = "sha256:c3b72104aff017e77fc9f5b427c8cd0c5f8dc7c4cbe604584e89f9fc24f1025a"
	mux := http.NewServeMux()
	for _, name := range []string{"seedvr2", "minimax-h3"} {
		raw, err := os.ReadFile(filepath.Join("testdata/positional", name+".json"))
		must(t, err)
		iface, problem := launch.DecodePackageInterface(raw)
		fatal(t, problem)
		var detail hub.PackageReleaseDetail
		detail.PackageInterface = iface.Raw
		detail.Release.Release = "1.0.0"
		detail.Release.PackageInterfaceDigest = assessmentDigest(iface.Raw)
		detail.Release.PackageInterfaceLength = int64(len(iface.Raw))
		detail.ExecutionRequirements = []string{"cozy-runtime>=0.23.0"}
		detail.RequiresPython = ">=3.12,<3.13"
		card := hub.PackageCard{Package: hub.Resource{Name: name}, Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}}
		mux.HandleFunc("GET /v1/packages/proof/"+name, func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(card) })
		mux.HandleFunc("GET /v1/packages/proof/"+name+"/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(detail) })
		mux.HandleFunc("GET /v1/packages/proof/"+name+"/bindings", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"bindings":[]}`)) })
	}
	mux.HandleFunc("GET /v1/models/proof/seedvr2-7b", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.ModelCard{Model: hub.Resource{Name: "seedvr2-7b"},
			Releases: []hub.ModelReleaseSummary{{ReleaseSummary: hub.ReleaseSummary{Release: "0.1.0"},
				Lanes: []hub.ModelLaneSummary{{Lane: "original", ManifestID: manifest, Bytes: 1 << 30, Components: []string{"conditioning", "dit", "vae"}}}}}})
	})
	mux.HandleFunc("/", http.NotFound)
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: positional\ncalcifer:\n  idle_shutdown_s: 0\n"), 0600))

	code, out := runCozy(t, root, "run", "proof/seedvr2/upscale", picture, "--rental-only", "--json")
	if code == 0 || !strings.Contains(out, "positional_file_unbound") || !strings.Contains(out, "video (video)") {
		t.Fatalf("an image for SeedVR2 was not refused naming its video input: %d %s", code, out)
	}
	startCalciferProcess(t, root)
	request, _, out := submitRun(t, root, "positional-seedvr2", "run", "proof/seedvr2/upscale", clip,
		"model.model=proof/seedvr2-7b@0.1.0/original", "--rental-only", "--json")
	if request == nil || len(request.Assets) != 1 || request.Assets[0].FieldPath != "video" || request.Assets[0].MediaType != "video/mp4" ||
		submittedPayload(t, request)["video"] != request.Assets[0].Digest || submittedPayload(t, request)["color_correction"] != nil {
		t.Fatalf("SeedVR2's positional video was not admitted as its video input: %s", out)
	}
	request, _, out = submitRun(t, root, "positional-h3", "run", "proof/minimax-h3/fl2va", "A fox crosses a frozen river.",
		picture, picture, "duration_s=15", "--rental-only", "--json")
	if request == nil || len(request.Assets) != 2 || request.Assets[1].FieldPath != "assets.1.asset" ||
		submittedPayload(t, request)["prompt"] != "A fox crosses a frozen river." {
		t.Fatalf("minimax-h3's positional frames were not admitted into its Assets: %s", out)
	}
	code, out = runCozy(t, root, "run", "proof/minimax-h3/fl2va", "A fox.", clip, "duration_s=15", "--rental-only", "--json")
	if code == 0 || !strings.Contains(out, "assets (image)") {
		t.Fatalf("minimax-h3 fl2va took a video: %d %s", code, out)
	}
}
