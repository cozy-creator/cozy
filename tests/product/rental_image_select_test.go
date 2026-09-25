//go:build !windows

package producttest

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// `--image` reaches the hub as the rental's one named image, is pinned to the
// paid operation, and the fleet listing shows the image each rental booted.
func TestRentalNewNamesARegisteredImageAndListShowsIt(t *testing.T) {
	root, hubURL, stand := rentalEndRoot(t, "rental-image-select")
	stand.publishListing()
	stand.setSKUs(map[string]any{
		"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
		"base_worker_profile": "torch2.14.0-cu130-cp312-linux-x86", "compute_capability": "9.0", "vram_gb": 94,
		"minimum_ram_per_gpu_gb": 64, "price_usd_micros_per_hour": 3_190_000,
	})
	const candidate = "sha256:2932afaa81a73a19ea04c132bf447adf15b894adf6b2a77d1667cf750e2517ad"
	const tag = "rt026dev-tfs054-host81bd55b7-cu130-20260925"
	var posted []map[string]any
	stand.mu.Lock()
	stand.rent = func(request map[string]any) map[string]any {
		posted = append(posted, request)
		// It never becomes ready: --timeout ends the wait and the operation stays open.
		return map[string]any{"rental_id": "pr-image-select", "name": request["name"], "state": "pending_acquisition",
			"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
			"hourly_rate_usd_micros": 3_190_000, "base_worker_image_digest": candidate, "base_worker_image_tag": tag}
	}
	stand.mu.Unlock()

	args := []string{"rental", "new", "h100-nvl", "--image", tag, "--idempotency-key", "image-select", "--timeout=2s", "--json"}
	if code, out := runCozy(t, root, args...); code == 0 {
		t.Fatalf("a rental that never became ready succeeded: %s", out)
	}
	stand.mu.Lock()
	if len(posted) != 1 || posted[0]["image"] != tag {
		stand.mu.Unlock()
		t.Fatalf("hub did not receive the named image: %+v", posted)
	}
	stand.mu.Unlock()

	// The operation's image is immutable: another --image under the same key
	// is refused before any second paid ask.
	args[4] = "cpu"
	code, out := runCozy(t, root, args...)
	if code == 0 || !strings.Contains(out, "rental.idempotency_conflict") {
		t.Fatalf("changed image under one operation was not refused: %d %s", code, out)
	}
	stand.mu.Lock()
	asks := len(posted)
	stand.mu.Unlock()
	if asks != 1 {
		t.Fatalf("a changed image reached the hub: %d asks", asks)
	}

	// Listing: one rental this host recorded, one it did not; both show their image.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	const recorded, other = "pr-image-recorded", "pr-image-other"
	stand.add(recorded, "wren")
	stand.set(recorded, "base_worker_image_digest", candidate)
	stand.set(recorded, "base_worker_image_tag", tag)
	stand.add(other, "finch")
	stand.set(other, "base_worker_image_digest", "sha256:"+strings.Repeat("5", 64))
	stand.set(other, "base_worker_image_tag", "cpu-torch-current")
	fatal(t, store.RecordRental(records.Rental{ID: recorded, MachineName: "wren", SKU: "cpu",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL}))
	store.Close()
	startDaemonProcess(t, root)
	code, out = runCozy(t, root, "rental", "list", "--json")
	if code != 0 {
		t.Fatalf("rental list failed: %d %s", code, out)
	}
	var doc struct {
		Rows []map[string]any `json:"rentals"`
	}
	must(t, json.Unmarshal([]byte(out), &doc))
	images := map[string][2]any{}
	for _, row := range doc.Rows {
		images[row["rental_id"].(string)] = [2]any{row["base_worker_image_tag"], row["base_worker_image_digest"]}
	}
	if images[recorded] != [2]any{tag, candidate} || images[other] != [2]any{"cpu-torch-current", "sha256:" + strings.Repeat("5", 64)} {
		t.Fatalf("rental list omitted images: %s", out)
	}
}
