package producttest

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// The stand-in Hub the rental placement arms drive the REAL binary and the REAL Calcifer
// against: it answers the catalog routes the way Tensorhub does and refuses every paid
// create typed unless a test provisions it, so no pod is ever bought.

const (
	gib           = int64(1) << 30
	h100SXM       = "NVIDIA H100 80GB HBM3"
	nvfp4Manifest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	bf16Manifest  = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	// fp8LaterManifest is the fp8 lane of the sized 1.0.0-rc.2 release (cl-174).
	fp8LaterManifest = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
	// textEncoderNeed is the largest group any H3 method holds resident: the text
	// conditioning's 51.5 GiB encoder. Every DiT sample holds one ~20 GiB DiT.
	textEncoderNeed    = 103 * gib / 2
	textEncoderVerdict = "excluded:vram_short: needs 51.5 GiB resident (condition_text: text_encoder)"
)

// fp8ManifestBody is the fp8 lane's manifest document, and fp8Manifest its real digest: a
// JOB binds EXACT manifest bytes and verifies their digest before it is submitted
// (jobManifestInputs), so the stand-in hub has to be able to serve them.
var (
	fp8ManifestBody = []byte(`{"cozytensors":"proof","lane":"fp8"}`)
	fp8Manifest     = mustSpell(fp8ManifestBody)
)

func mustSpell(raw []byte) string {
	id, err := canonical.Spell(canonical.Digest(raw))
	if err != nil {
		panic(err)
	}
	return id
}

// h3Components is the fp8 lane as the card publishes it: 103 GB in five components.
// h3BF16Components is the bf16 lane, 130 GiB, same text encoder.
func h3Components() map[string]int64 {
	return map[string]int64{"text_encoder": textEncoderNeed, "fl2va_dit": 20 * gib,
		"ref2va_dit": 20 * gib, "video_vae": 52 * gib / 5, "audio_vae": 3 * gib / 5}
}

func h3BF16Components() map[string]int64 {
	return map[string]int64{"text_encoder": textEncoderNeed, "fl2va_dit": 135 * gib / 4,
		"ref2va_dit": 135 * gib / 4, "video_vae": 52 * gib / 5, "audio_vae": 3 * gib / 5}
}

// market20260907 is the catalog shape on the day of the defect, with the memory figures
// the hub publishes for each product.
func market20260907() []hub.RentalSKU {
	gpu := func(name, model string, vram int64, price int64) hub.RentalSKU {
		return hub.RentalSKU{Name: name, AcceleratorModel: model, AcceleratorCount: 1, VRAMGB: vram,
			ComputeCapability:     "9.0",
			PriceUSDMicrosPerHour: price, StorageUSDMicrosPerHour: 213_504,
			BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"}
	}
	return []hub.RentalSKU{
		gpu("rtx-4090", "NVIDIA GeForce RTX 4090", 24, 740_000),
		gpu("rtx-5090", "NVIDIA GeForce RTX 5090", 32, 990_000),
		gpu("h100-80", "NVIDIA H100 80GB HBM3", 80, 2_490_000),
		gpu("h100-nvl", "NVIDIA H100 NVL", 94, 2_790_000),
		gpu("h200", "NVIDIA H200", 141, 3_590_000),
		gpu("b200", "NVIDIA B200", 180, 5_990_000),
		{Name: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, PriceUSDMicrosPerHour: 70_000,
			BaseWorkerProfile: "python3.12-cpu-linux-x86"},
	}
}

// walk is the order the choice would buy in if the hub refused every product for stock in
// turn: each pick is marked no_stock and the choice repeats on a copy of the candidates.
func walk(candidates []orchestrator.PlacementCandidate) []string {
	rows := append([]orchestrator.PlacementCandidate(nil), candidates...)
	var out []string
	for i := rental.Place("balanced", rows); i >= 0; i = rental.Place("balanced", rows) {
		out = append(out, rows[i].Name())
		rows[i].Verdict = orchestrator.VerdictNoStock
	}
	return out
}

// standInHub is the stand-in: one package with one modeled entrypoint, one model with one
// unyanked release of three lanes, and the 2026-09-07 GPU market.
type standInHub struct {
	mu       sync.Mutex
	server   *httptest.Server
	mux      *http.ServeMux
	iface    []byte // the h3 1.0.0 release interface; nil answers not published
	workflow []byte // the interface publishWorkflowRelease publishes; nil is workflowInterface
	posts    [][]byte
	// soldOut names the SKUs whose paid ask the hub refuses for inventory; rentals holds
	// what a GET on a rental answers; throughput is the model's published table, and a
	// hub holding none answers the route as an older build would — not at all.
	soldOut    map[string]bool
	rentals    map[string]map[string]any
	throughput []hub.ModelThroughput
	// throughputReads and skuReads count the placement reads of each kind.
	throughputReads, skuReads int
	// provisions makes a bought pod come up READY with the triple the ask pinned, so the
	// buy completes and the placement is recorded; otherwise the fixture pod fails.
	provisions bool
	// unsized makes the card publish no component bytes for any lane (the card as it
	// stood on 2026-09-07 22:27Z, cl-170). later adds the sized 1.0.0-rc.2 release a
	// later package defaults to while a run still names rc.1 explicitly (cl-174).
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
	h3Package = "proof/h3"
	h3Slot    = "generate.models.model"
	h3Model   = "proof/minimax"
	h3Release = "1.0.0-rc.1"
	h3Lane    = "fp8"
	// h3Default is the generate slot's authored default: the release, never a lane.
	h3Default = h3Model + "@" + h3Release
)

// newStandInHub is the stand-in whose entrypoints' slots declare defaults, in order.
func newStandInHub(t *testing.T, defaults ...string) *standInHub {
	t.Helper()
	return newOrgStandInHub(t, "proof", defaults...)
}

// newOrgStandInHub is the same stand-in for another account's h3 package and minimax model.
func newOrgStandInHub(t *testing.T, org string, defaults ...string) *standInHub {
	t.Helper()
	// `generate` is the function whose methods stage components. `lane` is the se-037 shape:
	// its model input is a `Weights` source it reads the header of and inherits by reference —
	// it declares no component_use, and it writes one 64 KiB config per output.
	iface := []byte(`{"application":"h3:app","entrypoints":[{"name":"generate","models":[{"class":"H3","component_use":{"condition_fl2va_media":["video_vae"],"condition_ref2va_media":["audio_vae","video_vae"],"condition_text":["text_encoder"],"decode_audio":["audio_vae"],"decode_video":["video_vae"],"sample_fl2va":["fl2va_dit"],"sample_ref2va":["ref2va_dit"]},"path":"generate.models.model"}],"request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}},{"name":"lane","models":[{"class":"Model","component_use":{},"path":"lane.models.pruned","weights":true}],"publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[{"max_bytes":65536,"mime_type":"application/vnd.cozy.model-manifest","output_id":"attn8"}]}],"format":"cozy.package.interface/1"}`)
	if len(defaults) > 0 {
		var doc map[string]any
		must(t, json.Unmarshal(iface, &doc))
		for index, def := range defaults {
			call := doc["entrypoints"].([]any)[index].(map[string]any)
			slot := call["models"].([]any)[0].(map[string]any)
			slot["default"] = def
		}
		var err error
		iface, err = json.Marshal(doc)
		must(t, err)
	}
	h := &standInHub{soldOut: map[string]bool{}, rentals: map[string]map[string]any{},
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
	mux.HandleFunc("GET /v1/models/"+org+"/minimax", func(w http.ResponseWriter, _ *http.Request) {
		h.mu.Lock()
		unsized, later := h.unsized, h.later
		h.mu.Unlock()
		card := hub.ModelCard{Model: hub.Resource{Org: org, Name: "minimax"},
			Releases: []hub.ModelReleaseSummary{
				{ReleaseSummary: hub.ReleaseSummary{Release: "0.9.0", Yanked: true}, Lanes: []hub.ModelLaneSummary{
					{Lane: "bf16", ManifestID: bf16Manifest, Bytes: 130 * gib}}},
				{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0-rc.1"}, Lanes: []hub.ModelLaneSummary{
					{Lane: "fp8", ManifestID: fp8Manifest, Bytes: 103 * gib,
						Components:     []string{"audio_vae", "fl2va_dit", "ref2va_dit", "text_encoder", "video_vae"},
						ComponentBytes: h3Components()},
					{Lane: "nvfp4", ManifestID: nvfp4Manifest, Bytes: 55 * gib,
						Components: []string{"audio_vae", "fl2va_dit", "ref2va_dit", "text_encoder", "video_vae"}},
					{Lane: "bf16", ManifestID: bf16Manifest, Bytes: 130 * gib,
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
					{Lane: "fp8", ManifestID: fp8LaterManifest, Bytes: 104 * gib,
						Components:     []string{"audio_vae", "fl2va_dit", "ref2va_dit", "text_encoder", "video_vae"},
						ComponentBytes: h3Components()},
					{Lane: "nvfp4", ManifestID: nvfp4Manifest, Bytes: 55 * gib},
					{Lane: "bf16", ManifestID: bf16Manifest, Bytes: 130 * gib, ComponentBytes: h3BF16Components()}}})
		}
		_ = json.NewEncoder(w).Encode(card)
	})
	mux.HandleFunc("GET /v1/models/"+org+"/minimax/releases/{release}/lanes/{lane}/manifest",
		func(w http.ResponseWriter, r *http.Request) {
			if r.PathValue("release") != h3Release || r.PathValue("lane") != h3Lane {
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write(fp8ManifestBody)
		})
	mux.HandleFunc("GET /v1/models/"+org+"/minimax/throughput", func(w http.ResponseWriter, _ *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.throughputReads++
		if h.throughput == nil {
			http.NotFound(w, nil)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"model": org + "/minimax", "throughput": h.throughput})
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.skuReads++
		_ = json.NewEncoder(w).Encode(hub.RentalProducts(h.market))
	})
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, _ *http.Request) {
		h.mu.Lock()
		defer h.mu.Unlock()
		rows := make([]map[string]any, 0, len(h.rentals))
		for _, row := range h.rentals {
			rows = append(rows, row)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"rentals": rows})
	})
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
	// A signed-in run on a rental reads where its machine reaches this Hub (#1127), as on Tensorhub.
	mux.HandleFunc("GET /v1/execution-environment", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer ladder-test" {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"code":"auth.required","message":"account required"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"environment": map[string]string{
			"TENSORHUB_ORIGIN": h.server.URL, "TENSORHUB_PUBLIC_ORIGIN": h.server.URL}})
	})
	h.server = httptest.NewServer(mux)
	t.Cleanup(h.server.Close)
	return h
}

func (h *standInHub) liveBurnLocked() int64 {
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

func (h *standInHub) skuTotalLocked(name string) int64 {
	for _, sku := range h.market {
		if sku.Name == name {
			return sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour
		}
	}
	return 0
}

func (h *standInHub) postedSKUs() []string {
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
func (h *standInHub) sell(skus ...hub.RentalSKU) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.market = skus
}

// addState is a rental the hub reports in one exact lifecycle word, with no worker triple:
// `acquiring` for a pod still being provisioned, `failed` for one that never will be.
func (h *standInHub) addState(id, machine, accelerator, state string, rate int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.rentals[id] = map[string]any{"rental_id": id, "name": machine, "state": state,
		"requested_accelerator_model": accelerator, "accelerator_count": 1,
		"hourly_rate_usd_micros": rate}
}

func (h *standInHub) addReady(id, machine, accelerator string, rate int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.rentals[id] = map[string]any{"rental_id": id, "name": machine, "state": "ready",
		"requested_accelerator_model": accelerator, "accelerator_count": 1, "hourly_rate_usd_micros": rate,
		"worker_address": "127.0.0.1:1", "media_address": "127.0.0.1:2", "machine_name": machine,
		"cert_pem": "fixture", "worker_id": "fixture", "worker_boot_id": "fixture", "creator_public_key": "fixture"}
}

func standInRoot(t *testing.T, h *standInHub) string {
	t.Helper()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: ladder-test\n"+
		"calcifer:\n  idle_shutdown_s: 0\n"), 0600))
	return root
}

func waitFor(t *testing.T, root, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatalf("%s did not happen: %s", what, tail(filepath.Join(root, "calcifer.log")))
		}
		time.Sleep(20 * time.Millisecond)
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
