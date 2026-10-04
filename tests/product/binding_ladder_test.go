package producttest

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// cl-166, the run that filed it: the hub binding said release "1.0.0" lane
// "profile=fp8-adaln-pruned" while the card carried "1.0.0-rc.1" / "fp8-adaln-pruned",
// and the mismatch surfaced only at run time as a bare "no lane". These arms drive the
// REAL binary and the REAL daemon against a stand-in hub that answers the catalog routes
// the way Tensorhub does and refuses every paid create typed, so no pod is ever bought.

// ladderHub is the stand-in: one package with one modeled entrypoint, one model with one
// unyanked release of two lanes, the 2026-09-07 GPU market, and a mutable binding row.
type ladderHub struct {
	mu                  sync.Mutex
	server              *httptest.Server
	mux                 *http.ServeMux
	iface               []byte // the h3 1.0.0 release interface; nil answers not published
	workflow            []byte // the interface publishWorkflowRelease publishes; nil is workflowInterface
	bindings            []hub.PackageBindingRow
	puts                [][]byte
	deletes             [][]byte
	resetConflict       bool
	bindingsUnavailable bool
	posts               [][]byte
	// soldOut names the SKUs whose paid ask the hub refuses for inventory; rentals holds
	// what a GET on a rental answers; throughput is the model's published table, and a
	// hub holding none answers the route as an older build would — not at all.
	soldOut    map[string]bool
	rentals    map[string]map[string]any
	throughput []hub.ModelThroughput
	// provisions makes a bought pod come up READY with the triple the ask pinned, so the
	// buy completes and the placement is recorded; otherwise the fixture pod fails.
	provisions bool
	// unsized makes the card publish no component bytes for any lane (the card as it
	// stood on 2026-09-07 22:27Z, cl-170). later adds the sized 1.0.0-rc.2 release a
	// later package binds while a run still names rc.1 explicitly (cl-174).
	unsized bool
	later   bool
	// market is what GET /v1/rental-skus answers. It is MUTABLE because a catalog that
	// momentarily offers no product of the request's class is a real hub state and one
	// half of what killed run 412 (cl-185); a fixed market cannot express it.
	market          []hub.RentalSKU
	runtimeVersions map[string]string
	// spendCap stands in for Tensorhub's owner fleet cap: a paid ask whose SKU
	// total would take the live rentals' burn past it is refused, buying nothing.
	spendCap int64
}

const (
	ladderPackage = "proof/h3"
	ladderSlot    = "generate.models.model"
	ladderModel   = "proof/minimax"
	ladderRelease = "1.0.0-rc.1"
	ladderLane    = "fp8-adaln-pruned"
)

func newLadderHub(t *testing.T, authored ...[]launch.ModelDefaultRung) *ladderHub {
	t.Helper()
	return newOrgLadderHub(t, "proof", authored...)
}

// newOrgLadderHub is the same stand-in for another account's h3 package and minimax model.
func newOrgLadderHub(t *testing.T, org string, authored ...[]launch.ModelDefaultRung) *ladderHub {
	t.Helper()
	// `generate` is the serving entrypoint whose methods stage components. `lane` is the
	// se-037 shape: a JOB whose model input is a derive-only source it reads the header of
	// and inherits by reference — its class declares no component_use, and it writes one
	// 64 KiB config per output.
	iface := []byte(`{"application":"h3:app","entrypoints":[{"name":"generate","models":[{"class":"H3","component_use":{"condition_fl2va_media":["video_vae"],"condition_ref2va_media":["audio_vae","video_vae"],"condition_text":["text_encoder"],"decode_audio":["audio_vae"],"decode_video":["video_vae"],"sample_fl2va":["fl2va_dit"],"sample_ref2va":["ref2va_dit"]},"path":"generate.models.model"}],"request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[{"name":"lane","models":[{"class":"Source","component_use":{},"path":"lane.models.pruned"}],"publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[{"max_bytes":65536,"mime_type":"application/vnd.cozy.model-manifest","output_id":"attn8"}]}]}`)
	if len(authored) > 0 {
		var doc map[string]any
		must(t, json.Unmarshal(iface, &doc))
		for index, group := range []string{"entrypoints", "jobs"} {
			if index >= len(authored) {
				break
			}
			call := doc[group].([]any)[0].(map[string]any)
			slot := call["models"].([]any)[0].(map[string]any)
			slot["default_ladder"] = authored[index]
		}
		var err error
		iface, err = json.Marshal(doc)
		must(t, err)
	}
	h := &ladderHub{soldOut: map[string]bool{}, rentals: map[string]map[string]any{},
		market: market20260907(), iface: iface}
	mux := http.NewServeMux()
	h.mux = mux
	mux.HandleFunc("GET /v1/packages/"+org+"/h3", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: org, Name: "h3"},
			Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
	})
	mux.HandleFunc("GET /v1/packages/"+org+"/h3/releases/1.0.0", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		iface := h.iface
		h.mu.Unlock()
		if iface == nil {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"not_found","message":"not published"}}`))
			return
		}
		normalized, err := canonical.NormalizeJCS(iface)
		must(t, err)
		digest, err := canonical.Spell(canonical.Digest(normalized))
		must(t, err)
		var detail hub.PackageReleaseDetail
		detail.PackageInterface = iface
		detail.Release.Release = "1.0.0"
		detail.Release.PackageInterfaceDigest = digest
		detail.Release.PackageInterfaceLength = int64(len(iface))
		detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25", "torch<3,>=2.13"}
		_ = json.NewEncoder(w).Encode(detail)
	})
	mux.HandleFunc("GET /v1/packages/"+org+"/h3/bindings", func(w http.ResponseWriter, _ *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.bindingsUnavailable {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":{"code":"bindings_unavailable","message":"temporarily unavailable"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"bindings": append([]hub.PackageBindingRow{}, h.bindings...)})
	})
	mux.HandleFunc("DELETE /v1/packages/"+org+"/h3/bindings/{slot}", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer ladder-test" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		body, err := io.ReadAll(r.Body)
		must(t, err)
		var request struct {
			ExpectedRevision *int64 `json:"expected_revision"`
		}
		must(t, json.Unmarshal(body, &request))
		if request.ExpectedRevision == nil {
			t.Error("reset omitted revision")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		h.mu.Lock()
		defer h.mu.Unlock()
		h.deletes = append(h.deletes, body)
		for index, row := range h.bindings {
			if row.Slot != r.PathValue("slot") {
				continue
			}
			if h.resetConflict || row.Revision != *request.ExpectedRevision {
				w.WriteHeader(http.StatusConflict)
				_, _ = w.Write([]byte(`{"error":{"code":"binding.revision_conflict","message":"owner changed binding"}}`))
				return
			}
			h.bindings = append(h.bindings[:index], h.bindings[index+1:]...)
			_ = json.NewEncoder(w).Encode(hub.PackageBindingReset{Slot: row.Slot, Changed: true})
			return
		}
		_ = json.NewEncoder(w).Encode(hub.PackageBindingReset{Slot: r.PathValue("slot"), Changed: false})
	})
	mux.HandleFunc("PUT /v1/packages/"+org+"/h3/bindings/{slot}", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		must(t, err)
		if r.Header.Get("Authorization") != "Bearer ladder-test" {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"code":"auth.required","message":"owner token required"}}`))
			return
		}
		var body struct {
			Model            string            `json:"model"`
			Release          string            `json:"release"`
			Ladder           []hub.BindingRung `json:"ladder"`
			ExpectedRevision int64             `json:"expected_revision"`
		}
		must(t, json.Unmarshal(raw, &body))
		h.mu.Lock()
		defer h.mu.Unlock()
		h.puts = append(h.puts, raw)
		row := hub.PackageBindingRow{Slot: r.PathValue("slot"), Model: body.Model, Release: body.Release,
			Ladder: body.Ladder, Revision: body.ExpectedRevision + 1, UpdatedAt: "2026-09-07T00:00:00Z"}
		h.bindings = []hub.PackageBindingRow{row}
		_ = json.NewEncoder(w).Encode(hub.PackageBindingWrite{Binding: row, Changed: true})
	})
	mux.HandleFunc("GET /v1/models/"+org+"/minimax", func(w http.ResponseWriter, _ *http.Request) {
		h.mu.Lock()
		unsized, later := h.unsized, h.later
		h.mu.Unlock()
		card := hub.ModelCard{Model: hub.Resource{Org: org, Name: "minimax"},
			Releases: []hub.ModelReleaseSummary{
				{ReleaseSummary: hub.ReleaseSummary{Release: "0.9.0", Yanked: true}, Lanes: []hub.ModelLaneSummary{
					{Lane: "bf16-full", ManifestID: bf16Manifest, Bytes: 130 * gib}}},
				{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0-rc.1"}, Lanes: []hub.ModelLaneSummary{
					{Lane: "fp8-adaln-pruned", ManifestID: fp8Manifest, Bytes: 103 * gib,
						Components:     []string{"audio_vae", "fl2va_dit", "ref2va_dit", "text_encoder", "video_vae"},
						ComponentBytes: h3Components()},
					{Lane: "mxfp8-adaln-pruned", ManifestID: mxfpManifest, Bytes: 55 * gib,
						Components: []string{"audio_vae", "fl2va_dit", "ref2va_dit", "text_encoder", "video_vae"}},
					{Lane: "bf16-full", ManifestID: bf16Manifest, Bytes: 130 * gib,
						Components:     []string{"audio_vae", "fl2va_dit", "ref2va_dit", "text_encoder", "video_vae"},
						ComponentBytes: h3BF16Components()}}},
			}}
		if unsized {
			for _, release := range card.Releases {
				for i := range release.Lanes {
					release.Lanes[i].ComponentBytes = nil
				}
			}
		}
		if later {
			card.Releases = append(card.Releases, hub.ModelReleaseSummary{
				ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0-rc.2"}, Lanes: []hub.ModelLaneSummary{
					{Lane: "fp8-adaln-pruned", ManifestID: fp8LaterManifest, Bytes: 104 * gib,
						Components:     []string{"audio_vae", "fl2va_dit", "ref2va_dit", "text_encoder", "video_vae"},
						ComponentBytes: h3Components()},
					{Lane: "mxfp8-adaln-pruned", ManifestID: mxfpManifest, Bytes: 55 * gib},
					{Lane: "bf16-full", ManifestID: bf16Manifest, Bytes: 130 * gib, ComponentBytes: h3BF16Components()}}})
		}
		_ = json.NewEncoder(w).Encode(card)
	})
	mux.HandleFunc("GET /v1/models/"+org+"/minimax/releases/{release}/lanes/{lane}/manifest",
		func(w http.ResponseWriter, r *http.Request) {
			if r.PathValue("release") != ladderRelease || r.PathValue("lane") != ladderLane {
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write(fp8ManifestBody)
		})
	mux.HandleFunc("GET /v1/models/"+org+"/minimax/throughput", func(w http.ResponseWriter, _ *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.throughput == nil {
			http.NotFound(w, nil)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"model": org + "/minimax", "throughput": h.throughput})
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		_ = json.NewEncoder(w).Encode(hub.RentalProducts(h.market))
	})
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"rentals":[]}`)) })
	mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		row, ok := h.rentals[r.PathValue("id")]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"rental.not_found","message":"absent"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(row)
	})
	mux.HandleFunc("POST /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		must(t, err)
		request, problem := hub.ParseRentalRequestBytes(raw)
		fatal(t, problem)
		h.mu.Lock()
		defer h.mu.Unlock()
		h.posts = append(h.posts, raw)
		if h.spendCap > 0 && h.liveBurnLocked()+h.skuTotalLocked(request.SKU) > h.spendCap {
			w.WriteHeader(http.StatusPaymentRequired)
			_, _ = w.Write([]byte(`{"error":{"code":"rental.fleet_spend_cap","message":"` + request.SKU + ` would exceed the owner hourly spend cap"}}`))
			return
		}
		if h.soldOut[request.SKU] {
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"error":{"code":"rental.sku_out_of_stock","message":"` + request.SKU + ` has no provider inventory right now"}}`))
			return
		}
		id := "pr-ladder-" + request.SKU
		row := map[string]any{"rental_id": id, "name": request.Name, "state": "failed",
			"requested_accelerator_model": "NVIDIA H200", "accelerator_count": 1, "hourly_rate_usd_micros": 3_590_000,
			"failure": map[string]any{"code": "fixture_finished"}}
		if h.provisions {
			row = map[string]any{"rental_id": id, "name": request.Name, "state": "ready",
				"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1, "hourly_rate_usd_micros": 2_790_000,
				"worker_address": "127.0.0.1:1", "media_address": "127.0.0.1:2", "cert_pem": "fixture",
				"worker_id": "fixture-worker", "worker_boot_id": "fixture-boot",
				"creator_public_key": request.CreatorPublicKey, "media_token_sha256": []string{request.MediaTokenSHA256}}
		}
		h.rentals[id] = row
		accepted := map[string]any{}
		for k, v := range row {
			accepted[k] = v
		}
		accepted["state"] = "acquiring"
		delete(accepted, "failure")
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(accepted)
	})
	mux.HandleFunc("POST /v1/rental-quotes", listedRentalQuote(mux))
	h.server = httptest.NewServer(mux)
	t.Cleanup(h.server.Close)
	return h
}

func (h *ladderHub) bind(row hub.PackageBindingRow) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.bindings = []hub.PackageBindingRow{row}
}

func (h *ladderHub) liveBurnLocked() int64 {
	var burn int64
	for _, row := range h.rentals {
		if state := row["state"]; state == "released" || state == "failed" {
			continue
		}
		switch rate := row["hourly_rate_usd_micros"].(type) {
		case int:
			burn += int64(rate)
		case int64:
			burn += rate
		}
	}
	return burn
}

func (h *ladderHub) skuTotalLocked(name string) int64 {
	for _, sku := range h.market {
		if sku.Name == name {
			return sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour
		}
	}
	return 0
}

func (h *ladderHub) postedSKUs() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]string, 0, len(h.posts))
	for _, raw := range h.posts {
		request, _ := hub.ParseRentalRequestBytes(raw)
		lane := ""
		if len(request.ServingModels) == 1 {
			lane = request.ServingModels[0].Lane
		}
		out = append(out, request.SKU+"/"+lane)
	}
	return out
}

// sell replaces what the catalog offers; an empty list is a hub with nothing on offer.
func (h *ladderHub) sell(skus ...hub.RentalSKU) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.market = skus
}

// addState is a rental the hub reports in one exact lifecycle word, with no worker triple:
// `acquiring` for a pod still being provisioned, `failed` for one that never will be.
func (h *ladderHub) addState(id, machine, accelerator, state string, rate int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.rentals[id] = map[string]any{"rental_id": id, "name": machine, "state": state,
		"requested_accelerator_model": accelerator, "accelerator_count": 1,
		"hourly_rate_usd_micros": rate}
}

func (h *ladderHub) addReady(id, machine, accelerator string, rate int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.rentals[id] = map[string]any{"rental_id": id, "name": machine, "state": "ready",
		"requested_accelerator_model": accelerator, "accelerator_count": 1, "hourly_rate_usd_micros": rate,
		"worker_address": "127.0.0.1:1", "media_address": "127.0.0.1:2"}
}

func ladderRoot(t *testing.T, h *ladderHub) string {
	t.Helper()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: ladder-test\n"+
		"daemon:\n  idle_shutdown_s: 0\n"), 0600))
	return root
}

func TestBindVerifiesTheLadderAgainstTheCardBeforeWriting(t *testing.T) {
	h := newLadderHub(t)
	root := ladderRoot(t, h)
	bind := func(args ...string) (int, string) {
		return runCozy(t, root, append([]string{"package", "bind", ladderPackage}, append(args, "--json")...)...)
	}
	for _, test := range []struct {
		name string
		args []string
		want []string
	}{
		{"release absent from the card", []string{ladderSlot, "proof/minimax@1.0.0", "--gpu", "H100=fp8-adaln-pruned"},
			[]string{"model.release_not_found", "releases: 1.0.0-rc.1"}},
		{"lane absent from the release", []string{ladderSlot, "proof/minimax@1.0.0-rc.1", "--gpu", "H100=profile=fp8-adaln-pruned"},
			[]string{"model.lane_not_found", "lanes: bf16-full, fp8-adaln-pruned, mxfp8-adaln-pruned"}},
		{"catch-all not last", []string{ladderSlot, "proof/minimax@1.0.0-rc.1", "--gpu", "*=bf16-full", "--gpu", "H100=fp8-adaln-pruned"},
			[]string{"catch-all rung '*' must be the last rung"}},
		{"no rung", []string{ladderSlot, "proof/minimax@1.0.0-rc.1"}, []string{"at least one --gpu"}},
		{"lane on the target", []string{ladderSlot, "proof/minimax@1.0.0-rc.1/fp8-adaln-pruned", "--gpu", "H100=fp8-adaln-pruned"},
			[]string{"ride its ladder"}},
		{"no release on the target", []string{ladderSlot, "proof/minimax", "--gpu", "H100=fp8-adaln-pruned"},
			[]string{"names no release"}},
		{"slot the interface does not declare", []string{"generate.models.other", "proof/minimax@1.0.0-rc.1", "--gpu", "H100=fp8-adaln-pruned"},
			[]string{"binding.slot_not_found", "slots: generate.models.model"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			code, out := bind(test.args...)
			if code == 0 {
				t.Fatalf("bind was accepted: %s", out)
			}
			for _, want := range test.want {
				if !strings.Contains(out, want) {
					t.Fatalf("refusal does not say %q: %s", want, out)
				}
			}
		})
	}
	h.mu.Lock()
	writes := len(h.puts)
	h.mu.Unlock()
	if writes != 0 {
		t.Fatalf("%d refused bind(s) still reached the hub", writes)
	}

	code, out := bind(ladderSlot, "proof/minimax@1.0.0-rc.1", "--gpu", "H100=fp8-adaln-pruned", "--gpu", "*=bf16-full")
	if code != 0 || !strings.Contains(out, "H100=fp8-adaln-pruned, *=bf16-full") || !strings.Contains(out, `"bound"`) {
		t.Fatalf("a verified bind was refused [exit %d]: %s", code, out)
	}
	h.mu.Lock()
	written := append([]byte(nil), h.puts[0]...)
	h.mu.Unlock()
	if string(written) != `{"expected_revision":0,"ladder":[{"gpu":"H100","lane":"fp8-adaln-pruned"},{"gpu":"*","lane":"bf16-full"}],"model":"proof/minimax","release":"1.0.0-rc.1"}` {
		t.Fatalf("the wire body is not the agreed contract: %s", written)
	}
	code, out = runCozy(t, root, "package", "bindings", ladderPackage)
	if code != 0 || !strings.Contains(out, "H100=fp8-adaln-pruned, *=bf16-full") || !strings.Contains(out, "1.0.0-rc.1") {
		t.Fatalf("bindings does not print the ladder [exit %d]: %s", code, out)
	}
	// A bind changes the binding revision every later run carries to its machine.
	revision := func() string {
		store, problem := records.Open(home.Paths(root).DB)
		if problem != nil {
			t.Fatal(problem)
		}
		defer store.Close()
		held, problem := store.BindingRevision()
		if problem != nil {
			t.Fatal(problem)
		}
		return held
	}
	first := revision()
	if first == "" {
		t.Fatal("a bind left no binding revision")
	}

	// A yanked release named explicitly is the owner's exact choice: honoured, with a warning.
	code, out = runCozy(t, root, "package", "bind", ladderPackage, ladderSlot, "proof/minimax@0.9.0", "--gpu", "*=bf16-full")
	if code != 0 || !strings.Contains(out, "warning: proof/minimax@0.9.0 is yanked; using it because it was named explicitly") {
		t.Fatalf("an explicitly named yanked release was not honoured with a warning [exit %d]: %s", code, out)
	}
	h.mu.Lock()
	written = append([]byte(nil), h.puts[len(h.puts)-1]...)
	h.mu.Unlock()
	if !strings.Contains(string(written), `"release":"0.9.0"`) {
		t.Fatalf("the yanked release was not bound: %s", written)
	}
	if revision() == first {
		t.Fatal("a second bind kept the binding revision")
	}
}

func TestRunRefusesEarlyWithWhatTheCardOffers(t *testing.T) {
	h := newLadderHub(t)
	root := ladderRoot(t, h)
	run := func() (int, string) {
		return runCozy(t, root, "run", "proof/h3/generate", "steps=1", "--rental-only", "--json",
			"--idempotency-key", "ladder-refusal")
	}
	// No binding at all: the remedy is the exact bind command, never a repo fallback.
	code, out := run()
	if code == 0 || !strings.Contains(out, "package_default_model_unavailable") ||
		!strings.Contains(out, "cozy package bind proof/h3 generate.models.model org/model@release --gpu") {
		t.Fatalf("an unbound slot did not refuse with the bind command [exit %d]: %s", code, out)
	}
	// The binding as it stood on 2026-09-07: a release the card does not carry.
	h.bind(hub.PackageBindingRow{Slot: ladderSlot, Model: ladderModel, Release: "1.0.0",
		Ladder: []hub.BindingRung{{GPU: "H100", Lane: "fp8-adaln-pruned"}}, Revision: 1})
	code, out = run()
	if code == 0 || !strings.Contains(out, "model.release_not_found") ||
		!strings.Contains(out, "releases: 1.0.0-rc.1") || !strings.Contains(out, "rebind: cozy package bind proof/h3 generate.models.model") {
		t.Fatalf("a stale binding did not refuse with the card's releases [exit %d]: %s", code, out)
	}
	h.bind(hub.PackageBindingRow{Slot: ladderSlot, Model: ladderModel, Release: "1.0.0-rc.1",
		Ladder: []hub.BindingRung{{GPU: "H100", Lane: "profile=fp8-adaln-pruned"}}, Revision: 2})
	code, out = run()
	if code == 0 || !strings.Contains(out, "model.lane_not_found") || !strings.Contains(out, "lanes: bf16-full, fp8-adaln-pruned, mxfp8-adaln-pruned") {
		t.Fatalf("a stale lane did not refuse with the release's lanes [exit %d]: %s", code, out)
	}
	// Early means before submission: no request row exists on this root, so no daemon
	// ever had work, and nothing reached a paid ask.
	if store, problem := records.Open(filepath.Join(root, "creator.sqlite")); problem == nil {
		defer store.Close()
		if row, problem := store.RequestByIdempotencyKey("ladder-refusal"); problem != nil || row != nil {
			t.Fatalf("a refused run was submitted: %+v (%v)", row, problem)
		}
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.posts) != 0 {
		t.Fatal("a refused run reached a paid ask")
	}
}

// goodLadder is the owner's fit map: fp8 on H100-class cards, mxfp8 (a lane the card
// sizes no components for) on B200, bf16 anywhere else that can hold it.
func goodLadder() hub.PackageBindingRow {
	return hub.PackageBindingRow{Slot: ladderSlot, Model: ladderModel, Release: "1.0.0-rc.1",
		Ladder: []hub.BindingRung{{GPU: "H100", Lane: "fp8-adaln-pruned"}, {GPU: "B200", Lane: "mxfp8-adaln-pruned"},
			{GPU: "*", Lane: "bf16-full"}}, Revision: 3}
}

func waitFor(t *testing.T, root, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatalf("%s did not happen: %s", what, tail(filepath.Join(root, "daemon.log")))
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestAutoRentWalksTheLadderAndNeverBuysAShortCard(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.soldOut["h100-80"], h.soldOut["h100-nvl"], h.soldOut["b200"] = true, true, true
	root := ladderRoot(t, h)
	startDaemonProcess(t, root)
	// The fixture pod fails to provision after the walk, so the short observation the
	// client keeps may already see that terminal; what is asserted is the walk itself.
	_, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", "--rental-only", "--json",
		"--idempotency-key", "ladder-walk")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	queued, problem := store.RequestByIdempotencyKey("ladder-walk")
	fatal(t, problem)
	if queued == nil {
		t.Fatalf("the laddered run was not submitted: %s", out)
	}
	waitFor(t, root, "the walk reaching the H200", func() bool { return len(h.postedSKUs()) >= 4 })
	posted := h.postedSKUs()
	// The 103 GB fp8 lane is asked of the 80 GB H100 first: its largest resident group
	// is the 51.5 GiB text encoder (cl-168). The unsized mxfp8 lane rides the owner's
	// B200 rung by assertion. The 24 GB and 32 GB cards never hold that text encoder.
	if strings.Join(posted[:4], " ") != "h100-80/fp8-adaln-pruned h100-nvl/fp8-adaln-pruned b200/mxfp8-adaln-pruned h200/bf16-full" {
		t.Fatalf("paid asks %v; want the H100 rung cheapest first, the B200 rung, then the catch-all on the H200", posted)
	}
	for _, ask := range posted {
		if strings.HasPrefix(ask, "rtx-") {
			t.Fatalf("a card the lane cannot fit was asked for: %v", posted)
		}
	}
	waitFor(t, root, "the lane pinned with the buy", func() bool {
		row, problem := store.RequestByIdempotencyKey("ladder-walk")
		return problem == nil && row.Models[0].Lane == "bf16-full"
	})
	row, problem := store.RequestByIdempotencyKey("ladder-walk")
	fatal(t, problem)
	if row.Models[0].Manifest != bf16Manifest || len(row.Models[0].Ladder) != 3 {
		t.Fatalf("the buy did not pin the H200's rung exactly, ladder kept: %+v", row.Models[0])
	}
	log := tail(filepath.Join(root, "daemon.log"))
	for _, want := range []string{"renting h100-80", "(rung 1, lane fp8-adaln-pruned, fit components 51.5 GiB (total memory unmeasured for this exact workload/SKU) of 80 GB)",
		"h100-80 has no inventory; choosing again without it",
		"renting b200", "(rung 2, lane mxfp8-adaln-pruned, fit rung_asserted)",
		"renting h200", "(rung 3, lane bf16-full, fit components 51.5 GiB (total memory unmeasured for this exact workload/SKU) of 141 GB)"} {
		if !strings.Contains(log, want) {
			t.Fatalf("daemon.log does not say %q:\n%s", want, log)
		}
	}
}

func TestAutoRentReusesTheFittingRentalBeforeBuying(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	root := ladderRoot(t, h)
	cert := filepath.Join(root, "zack.pem")
	must(t, os.WriteFile(cert, []byte("fixture"), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	// Three live rentals the user already has up. Rung order ranks no machine that is
	// already paid for. The 80 GB H100 holds the 103 GB fp8 lane — no method holds more
	// than its 51.5 GiB text encoder — and wins on the fewest bytes to download; the H200
	// on the bf16 rung fits too; the 5090 fits a rung but not that text encoder, so it is
	// passed over with the need recorded (the 2026-09-07 production case, cl-168).
	for _, seed := range []records.Rental{
		{AcceleratorCount: 1, ID: "pr-zack", MachineName: "zack", SKU: "h200", AcceleratorModel: "NVIDIA H200",
			HourlyRateUSDMicros: 3_590_000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: h.server.URL},
		{AcceleratorCount: 1, ID: "pr-cheap", MachineName: "cheap", SKU: "rtx-5090", AcceleratorModel: "NVIDIA GeForce RTX 5090",
			HourlyRateUSDMicros: 990_000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: h.server.URL},
		{AcceleratorCount: 1, ID: "pr-morgiana", MachineName: "morgiana", SKU: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3",
			HourlyRateUSDMicros: 2_490_000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: h.server.URL},
	} {
		fatal(t, store.RecordRental(seed))
		h.addReady(seed.ID, seed.MachineName, seed.AcceleratorModel, seed.HourlyRateUSDMicros)
	}
	store.Close()
	startDaemonProcess(t, root)
	// The seeded rental has no media bearer on this host, so the request fails after the
	// pin; the pin and the lane it carries are what this proves.
	_, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", "--rental-only", "--json",
		"--idempotency-key", "ladder-reuse")
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	waitFor(t, root, "the request pinned to the live H100", func() bool {
		row, problem := store.RequestByIdempotencyKey("ladder-reuse")
		return problem == nil && row != nil && row.Worker != ""
	})
	row, problem := store.RequestByIdempotencyKey("ladder-reuse")
	fatal(t, problem)
	if row == nil {
		t.Fatalf("the laddered run was not submitted: %s", out)
	}
	if row.Worker != "pr-morgiana" || row.Models[0].Lane != "fp8-adaln-pruned" || row.Models[0].Manifest != fp8Manifest ||
		row.Models[0].ComponentBytes["text_encoder"] != textEncoderNeed {
		t.Fatalf("pinned to %q with %+v; want pr-morgiana on fp8-adaln-pruned with its component bytes", row.Worker, row.Models[0])
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("a fitting live rental was passed over for a buy: %v", asks)
	}
	log := tail(filepath.Join(root, "daemon.log"))
	if !strings.Contains(log, "cheap excluded:vram_short: needs 51.5 GiB resident (condition_text: text_encoder), rtx-5090 has 32 GB") ||
		!strings.Contains(log, "placement: reuse morgiana (h100-80, fp8-adaln-pruned) — balanced, unmeasured (rung 1)") {
		t.Fatalf("daemon.log does not record the passed-over 5090's need and the placement:\n%s", log)
	}
	// A hub without the throughput route measured nothing: the record says every open
	// candidate was unmeasured and cites no row, and the live H100 won by the ladder's
	// own order — attached first — over a later-rung H200 and every buy.
	placement := placementEvent(t, store, row.ID)
	if placement["tier"] != "balanced" || len(placement["throughput"].([]any)) != 0 {
		t.Fatalf("an unmeasured placement recorded %v", placement)
	}
	verdicts := candidateVerdicts(placement)
	if verdicts["morgiana"] != "chosen" || verdicts["zack"] != "unmeasured" || verdicts["h100-80"] != "unmeasured" ||
		verdicts["b200"] != "unmeasured" || !strings.HasPrefix(verdicts["cheap"], "excluded:vram_short") {
		t.Fatalf("unmeasured verdicts %v", verdicts)
	}
}

// placementEvent is the request's one durable `request.placement` record.
func placementEvent(t *testing.T, store *records.Store, requestID string) map[string]any {
	t.Helper()
	events, problem := store.EventsAfter(requestID, 0, 200)
	fatal(t, problem)
	for _, event := range events {
		if event.Type == "request.placement" {
			return event.Payload
		}
	}
	t.Fatalf("no request.placement event among %d events", len(events))
	return nil
}

// candidateVerdicts reads the record's candidates as name -> verdict.
func candidateVerdicts(placement map[string]any) map[string]string {
	out := map[string]string{}
	for _, raw := range placement["candidates"].([]any) {
		c := raw.(map[string]any)
		name, _ := c["machine"].(string)
		if name == "" {
			name, _ = c["sku"].(string)
		}
		out[name], _ = c["verdict"].(string)
	}
	return out
}

// cl-180. On 2026-09-08 `cozy run paul/minimax-h3-tools/attention-lane
// --model.pruned=paul/minimax-h3@1.0.0-rc.2/fp8-adaln-pruned --rental-only` was sized
// "fit components 48.0 GiB (working memory unmeasured) of 48 GB": the job reads the source HEADER, inherits 3,858
// tensors by reference and writes 550 bytes, but the sizer held the card to the
// components a SERVING construction of that lane would stage. The 24 GB card was
// excluded, two rtx-6000-ada pods sat in `acquiring` for 22 and 24 minutes, and ~32 s of
// work cost 849 s of wall. cozy-runtime gives a job a derive-only view of the Manifest
// and refuses load and component access on it, so a job's model is never on the device:
// the cheap card is admissible and the record says `derive_only` rather than a figure.
func TestAJobIsNotSizedByTheComponentsItNeverStages(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	// The cheapest card is refused for stock, so the walk itself is the evidence: the
	// 24 GB product is asked for FIRST, which the component figure made impossible.
	h.soldOut["rtx-4090"] = true
	root := ladderRoot(t, h)
	startDaemonProcess(t, root)
	_, out := runCozy(t, root, "run", ladderPackage+"/lane",
		"--model.pruned="+ladderModel+"@"+ladderRelease+"/"+ladderLane,
		"--rental-only", "--json", "--idempotency-key", "job-derive-fit")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	queued, problem := store.RequestByIdempotencyKey("job-derive-fit")
	fatal(t, problem)
	if queued == nil || queued.Kind != "job" {
		t.Fatalf("the by-reference job was not submitted: %s", out)
	}
	if len(queued.Models) != 1 || queued.Models[0].Manifest != fp8Manifest ||
		queued.Models[0].ComponentBytes["text_encoder"] != textEncoderNeed {
		t.Fatalf("the job did not carry the sized lane it reads: %+v", queued.Models)
	}
	waitFor(t, root, "the walk passing the cheapest card", func() bool { return len(h.postedSKUs()) >= 2 })
	posted := h.postedSKUs()
	if !strings.HasPrefix(posted[0], "rtx-4090/") || !strings.HasPrefix(posted[1], "rtx-5090/") {
		t.Fatalf("paid asks %v; want the cheapest cards first for a job that stages nothing", posted)
	}
	log := tail(filepath.Join(root, "daemon.log"))
	for _, want := range []string{"renting rtx-4090", "(lane fp8-adaln-pruned, fit derive_only)",
		"rtx-4090 has no inventory; choosing again without it", "renting rtx-5090"} {
		if !strings.Contains(log, want) {
			t.Fatalf("daemon.log does not say %q:\n%s", want, log)
		}
	}
	if strings.Contains(log, "vram_short") {
		t.Fatalf("a job's derive-only model excluded a card for memory:\n%s", log)
	}
}

// A rung naming a lane the release no longer carries is skipped; the rest of the owner's
// ladder still places the run instead of the stale rung refusing it.
func TestRunSkipsAStaleRungAndKeepsTheLadder(t *testing.T) {
	h := newLadderHub(t)
	h.bind(hub.PackageBindingRow{Slot: ladderSlot, Model: ladderModel, Release: "1.0.0-rc.1",
		Ladder: []hub.BindingRung{{GPU: "H100", Lane: "retired-lane"}, {GPU: "*", Lane: "bf16-full"}}, Revision: 1})
	root := ladderRoot(t, h)
	startDaemonProcess(t, root)
	_, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", "--rental-only", "--json",
		"--idempotency-key", "ladder-stale-rung")
	if strings.Contains(out, "model.lane_not_found") {
		t.Fatalf("one stale rung refused the whole ladder: %s", out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	queued, problem := store.RequestByIdempotencyKey("ladder-stale-rung")
	fatal(t, problem)
	if queued == nil || len(queued.Models) != 1 || len(queued.Models[0].Ladder) != 1 ||
		queued.Models[0].Ladder[0].Lane != "bf16-full" {
		t.Fatalf("the usable rung was not submitted: %+v\n%s", queued, out)
	}
}
