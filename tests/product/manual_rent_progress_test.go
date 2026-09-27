//go:build !windows

package producttest

import (
	"bytes"
	"encoding/json"
	"io"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// Drive the ordinary CLI through real polling against a Hub fixture. The fixture
// fails after boot so the proof rents no hardware and needs no worker substitute.
func TestManualRentShowsSharedAcquisitionProgress(t *testing.T) {
	for _, mode := range []string{"redirected", "terminal", "json"} {
		t.Run(mode, func(t *testing.T) {
			root, _, stand := rentalEndRoot(t, "manual-rent-progress")
			stand.publishListing()
			stand.setSKUs(map[string]any{
				"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
				"base_worker_profile": "proof", "compute_capability": "9.0", "vram_gb": 94,
				"minimum_ram_per_gpu_gb": 64, "price_usd_micros_per_hour": 3_190_000,
			})
			const id = "pr-manual-progress"
			const tag = "torch2.13.0-cu130-cp312-linux-x86-runtime0.17.1"
			const image = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
			created := make(chan struct{})
			stand.mu.Lock()
			stand.rent = func(request map[string]any) map[string]any {
				close(created)
				row := map[string]any{"rental_id": id, "name": request["name"], "state": "pending_acquisition",
					"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
					"hourly_rate_usd_micros": 3_190_000, "base_worker_image_digest": image, "base_worker_image_tag": tag,
					"base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86",
				}
				if mode == "json" {
					row["state"] = "failed"
					row["detail"] = "fixture boot completed without a worker"
				}
				return row
			}
			stand.mu.Unlock()
			// The fixture's failure ends the acquisition; the timeout only bounds a hang.
			args := []string{"rental", "new", "h100-nvl", "--timeout=120s"}
			var cmd *exec.Cmd
			var reader io.ReadCloser
			var stdout bytes.Buffer
			if mode == "terminal" {
				reader, cmd = startPTY(t, root, 24, ptyColumns, args...)
			} else {
				if mode == "json" {
					args = append(args, "--json")
				}
				cmd = exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
				cmd.Env = childEnv(t, root)
				cmd.Stdout = &stdout
				var err error
				reader, err = cmd.StderrPipe()
				must(t, err)
				must(t, cmd.Start())
				t.Cleanup(func() { _ = cmd.Process.Kill() })
			}
			defer reader.Close()
			// Keep each provider phase until the actual CLI renders it. Timed phase
			// changes can occur between polls, especially on a busy CI runner.
			var progress strings.Builder
			stage := 0
			chunk := make([]byte, 4096)
			for {
				n, err := reader.Read(chunk)
				progress.Write(chunk[:n])
				text := progress.String()
				switch {
				case stage == 0 && strings.Contains(text, "acquiring"):
					select {
					case <-created:
					case <-time.After(10 * time.Second):
						t.Fatal("CLI did not submit its rental request")
					}
					stand.mu.Lock()
					stand.rentals[id]["provider_state"] = "RUNNING"
					stand.rentals[id]["container_state"] = "PULLING"
					stand.mu.Unlock()
					stage++
				case stage == 1 && strings.Contains(text, "pulling image on"):
					stand.set(id, "container_state", "RUNNING")
					stage++
				case stage == 2 && strings.Contains(text, "booting on"):
					stand.setState(id, "failed", "fixture boot completed without a worker")
					stage++
				}
				if err != nil {
					break
				}
			}
			_ = cmd.Wait()
			code, log := cmd.ProcessState.ExitCode(), progress.String()
			if code == 0 {
				t.Fatal("the terminal fixture failure unexpectedly succeeded")
			}
			if mode == "json" {
				var result map[string]any
				if err := json.Unmarshal(stdout.Bytes(), &result); err != nil || result["error"] == nil || log != "" {
					t.Fatalf("manual rental JSON is not one clean error document: stdout=%q stderr=%q", stdout.String(), log)
				}
				return
			}
			for _, want := range []string{"acquiring", "NVIDIA H100 NVL · $3.19/hour", "image: " + tag, "pulling image on", "booting on"} {
				if !strings.Contains(log, want) {
					t.Fatalf("manual rent omitted %q: %q", want, log)
				}
			}
			if strings.Contains(log, "image: sha256:") {
				t.Fatalf("acquisition exposed an opaque image digest: %q", log)
			}
			if strings.Contains(log, "ETA") || strings.Contains(log, "%") {
				t.Fatalf("boot acquired a fabricated estimate or percentage: %q", log)
			}
			if mode == "terminal" {
				if !strings.Contains(log, "\r\033[K") || !strings.Contains(log, " · done") || !strings.Contains(log, " · failed") {
					t.Fatalf("manual rent did not retain completed stages and final failure: %q", log)
				}
			} else if strings.ContainsAny(log, "\r\033") || strings.Count(log, "pulling image on") != 1 {
				t.Fatalf("redirected boot was not sparse plain text: %q", log)
			}
		})
	}
}
