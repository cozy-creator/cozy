//go:build !windows

package producttest

import (
	"strings"
	"testing"
	"time"
)

// Drive the ordinary CLI through real polling against a Hub fixture. The fixture
// fails after boot so the proof rents no hardware and needs no worker substitute.
func TestManualRentShowsSharedAcquisitionProgress(t *testing.T) {
	for _, terminal := range []bool{false, true} {
		t.Run(map[bool]string{false: "redirected", true: "terminal"}[terminal], func(t *testing.T) {
			root, _, stand := rentalEndRoot(t, "manual-rent-progress")
			stand.publishListing()
			stand.setSKUs(map[string]any{
				"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
				"base_worker_profile": "proof", "compute_capability": "9.0", "vram_gb": 94,
				"minimum_ram_per_gpu_gb": 64, "price_usd_micros_per_hour": 3_190_000,
			})
			const id = "pr-manual-progress"
			const image = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
			created := make(chan struct{})
			stand.mu.Lock()
			stand.rent = func(request map[string]any) map[string]any {
				close(created)
				return map[string]any{"rental_id": id, "name": request["name"], "state": "pending_acquisition",
					"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
					"hourly_rate_usd_micros": 3_190_000, "base_worker_image_digest": image,
				}
			}
			stand.mu.Unlock()
			finished := make(chan struct{})
			go func() {
				defer close(finished)
				<-created
				time.Sleep(300 * time.Millisecond)
				stand.set(id, "provider_state", "RUNNING")
				stand.set(id, "container_state", "PULLING")
				time.Sleep(2 * time.Second)
				stand.set(id, "container_state", "RUNNING")
				time.Sleep(2 * time.Second)
				stand.setState(id, "failed", "fixture boot completed without a worker")
			}()
			var code int
			var log string
			if terminal {
				code, log = ptyRun(t, root, "rent", "h100-nvl", "--timeout=8s")
			} else {
				code, _, log = runCozyStreams(t, root, "rent", "h100-nvl", "--timeout=8s")
			}
			<-finished
			if code == 0 {
				t.Fatal("the terminal fixture failure unexpectedly succeeded")
			}
			for _, want := range []string{"acquiring", "NVIDIA H100 NVL · $3.19/hour", "image: sha256:0123456789abcdef", "pulling image on", "booting on"} {
				if !strings.Contains(log, want) {
					t.Fatalf("manual rent omitted %q: %q", want, log)
				}
			}
			if strings.Contains(log, "ETA") || strings.Contains(log, "%") {
				t.Fatalf("boot acquired a fabricated estimate or percentage: %q", log)
			}
			if terminal {
				if !strings.Contains(log, "\r\033[K") || !strings.Contains(log, " · done") || !strings.Contains(log, " · failed") {
					t.Fatalf("manual rent did not retain completed stages and final failure: %q", log)
				}
			} else if strings.ContainsAny(log, "\r\033") || strings.Count(log, "pulling image on") != 1 {
				t.Fatalf("redirected boot was not sparse plain text: %q", log)
			}
		})
	}
}
