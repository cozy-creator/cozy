//go:build !windows

package producttest

import (
	"fmt"
	"strings"
	"testing"
)

// `--disk-gb` reaches the Hub as the rental's requested container disk, is pinned to the
// paid operation, and a changed size under the same operation is refused before any ask.
func TestRentalNewRequestsTheContainerDisk(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "rental-disk-gb")
	stand.publishListing()
	stand.setSKUs(map[string]any{
		"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
		"base_worker_profile": "torch2.14.0-cu130-cp312-linux-x86", "compute_capability": "9.0", "vram_gb": 94,
		"price_usd_micros_per_hour": 3_190_000,
	})
	var posted []map[string]any
	stand.mu.Lock()
	stand.rent = func(request map[string]any) map[string]any {
		posted = append(posted, request)
		// It never becomes ready: an interrupted wait leaves the operation open.
		view := map[string]any{"rental_id": fmt.Sprintf("pr-disk-gb-%d", len(posted)), "name": request["name"], "state": "pending_acquisition",
			"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1, "hourly_rate_usd_micros": 3_190_000}
		if disk, ok := request["container_disk_gb"].(float64); ok && disk == 700 {
			view["container_disk_gb"] = 350 // the minimum this Hub would buy
		}
		return view
	}
	stand.mu.Unlock()
	asks := func() []map[string]any {
		stand.mu.Lock()
		defer stand.mu.Unlock()
		return append([]map[string]any(nil), posted...)
	}

	if code, out := rentUntilRecorded(t, root, "pr-disk-gb-1", "h100-nvl", "--disk-gb=600", "--idempotency-key", "disk-gb", "--json"); code == 0 {
		t.Fatalf("a rental that never became ready succeeded: %s", out)
	}
	if sent := asks(); len(sent) != 1 || sent[0]["container_disk_gb"] != float64(600) {
		t.Fatalf("the Hub did not receive the requested disk: %+v", sent)
	}
	// Resuming the operation without the flag replays the same request.
	if code, out := rentUntilRecorded(t, root, "pr-disk-gb-2", "h100-nvl", "--idempotency-key", "disk-gb", "--json"); code == 0 ||
		strings.Contains(out, "idempotency_conflict") {
		t.Fatalf("resuming the operation without --disk-gb was refused or completed: %d %s", code, out)
	}
	if sent := asks(); len(sent) != 2 || sent[1]["container_disk_gb"] != float64(600) {
		t.Fatalf("the resumed request changed its disk: %+v", sent)
	}
	code, out := runCozy(t, root, "rental", "new", "h100-nvl", "--disk-gb=900", "--idempotency-key", "disk-gb", "--timeout=60s", "--json")
	if code == 0 || !strings.Contains(out, "rental.idempotency_conflict") {
		t.Fatalf("a changed disk under one operation was not refused: %d %s", code, out)
	}
	for _, bad := range [][]string{{"rental", "new", "h100-nvl", "--disk-gb=-5"}, {"rental", "new", "--disk-gb=600"}} {
		if code, out := runCozy(t, root, append(bad, "--json")...); code == 0 {
			t.Fatalf("%v was accepted: %s", bad, out)
		}
	}
	if sent := asks(); len(sent) != 2 {
		t.Fatalf("a refused disk request reached the Hub: %d asks", len(sent))
	}
	// Before purchase the view's disk is the minimum the Hub will buy; below the request, say so.
	if code, out := runCozy(t, root, "rental", "new", "h100-nvl", "--disk-gb=700", "--idempotency-key", "short-disk", "--timeout=2s"); code == 0 ||
		!strings.Contains(out, "warning: Tensorhub plans a 350 GB disk, below the requested 700 GB") {
		t.Fatalf("a planned disk below the request was not named [exit %d]: %s", code, out)
	}
	// Undeclared stays off the wire.
	if code, out := rentUntilRecorded(t, root, "pr-disk-gb-4", "h100-nvl", "--idempotency-key", "no-disk", "--json"); code == 0 {
		t.Fatalf("a rental that never became ready succeeded: %s", out)
	}
	if sent := asks(); len(sent) != 4 {
		t.Fatalf("paid asks = %d, want 4", len(sent))
	} else if _, present := sent[3]["container_disk_gb"]; present {
		t.Fatalf("a rental without --disk-gb declared a disk: %+v", sent[3])
	}
}
