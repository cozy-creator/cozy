//go:build !windows

package producttest

import (
	"bytes"
	"encoding/json"
	"os/exec"
	"strings"
	"syscall"
	"testing"
	"time"
)

// `cozy rental new` stopped while the Hub still acquires (an interrupt, a SIGTERM from a
// caller's timeout) does not cancel the order, and says so: the rental goes on, and the
// words name how to end it. The Hub is asked to release nothing.
func TestRentalNewStoppedWhileAcquiringSaysTheOrderContinues(t *testing.T) {
	for _, mode := range []string{"redirected", "json"} {
		t.Run(mode, func(t *testing.T) {
			root, _, stand := rentalEndRoot(t, "rental-new-interrupt")
			stand.publishListing()
			stand.setSKUs(map[string]any{
				"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
				"base_worker_profile": "proof", "compute_capability": "9.0", "vram_gb": 94,
				"minimum_ram_per_gpu_gb": 64, "price_usd_micros_per_hour": 3_190_000,
			})
			const id = "pr-interrupted-order"
			created := make(chan struct{})
			stand.mu.Lock()
			stand.rent = func(request map[string]any) map[string]any {
				close(created)
				return map[string]any{"rental_id": id, "name": request["name"], "state": "pending_acquisition",
					"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
					"hourly_rate_usd_micros": 3_190_000}
			}
			stand.mu.Unlock()
			args := []string{"rental", "new", "h100-nvl"}
			if mode == "json" {
				args = append(args, "--json")
			}
			cmd := exec.Command(cozyBin, args...)
			cmd.Env = childEnv(t, root)
			var stdout, stderr bytes.Buffer
			cmd.Stdout, cmd.Stderr = &stdout, &stderr
			must(t, cmd.Start())
			t.Cleanup(func() { _ = cmd.Process.Kill() })
			select {
			case <-created:
			case <-time.After(time.Minute): // harness bound on a broken CLI
				t.Fatalf("the CLI did not order the rental\n%s", stderr.String())
			}
			time.Sleep(500 * time.Millisecond) // into the acquisition wait
			must(t, cmd.Process.Signal(syscall.SIGTERM))
			_ = cmd.Wait()
			code, out := cmd.ProcessState.ExitCode(), stdout.String()+stderr.String()
			if strings.Contains(out, "cancelled") && !strings.Contains(out, "not cancelled") ||
				!strings.Contains(out, "the order was not cancelled") || !strings.Contains(out, "cozy rental end "+id) {
				t.Fatalf("a stopped acquisition did not say the order continues and how to end it [exit %d]\n%s", code, out)
			}
			if mode == "json" {
				var document struct {
					Error struct {
						Code string `json:"code"`
					} `json:"error"`
					Next []string `json:"next"`
				}
				if code == 0 || json.Unmarshal(stdout.Bytes(), &document) != nil || document.Error.Code != "rental.wait_stopped" ||
					len(document.Next) == 0 || document.Next[0] != "cozy rental end "+id {
					t.Fatalf("--json did not end on the typed stop [exit %d]\n%s", code, out)
				}
			}
			if released := stand.releases(id); released != 0 {
				t.Fatalf("stopping the wait released the rental %d times", released)
			}
		})
	}
}
