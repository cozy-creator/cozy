package producttest

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"testing"
	"time"
)

// Expose only this synthetic wheel, never a directory, API, or caller source.
// This uses the bounded temporary transport pattern from Hub PR616.
func publishedFixtureWheel(t *testing.T, wheel []byte, filename string) string {
	t.Helper()
	if len(wheel) == 0 || len(wheel) > 1<<20 || filepath.Base(filename) != filename {
		t.Fatal("public fixture wheel exceeds its exact bound")
	}
	var capability [32]byte
	_, err := rand.Read(capability[:])
	must(t, err)
	prefix := "/" + hex.EncodeToString(capability[:]) + "/"
	slots := make(chan struct{}, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.RawQuery != "" || (r.URL.Path != prefix+filename && r.URL.Path != prefix+"health") {
			http.NotFound(w, r)
			return
		}
		if r.ContentLength > 0 || len(r.TransferEncoding) != 0 {
			w.WriteHeader(http.StatusRequestEntityTooLarge)
			return
		}
		if r.Method != "GET" && r.Method != "HEAD" {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		select {
		case slots <- struct{}{}:
			defer func() { <-slots }()
		default:
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.Header().Set("X-Cozy-Test-Wheel", "1")
		w.Header().Set("Cache-Control", "no-store, no-transform")
		if r.URL.Path == prefix+"health" {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		http.ServeContent(w, r, filename, time.Time{}, bytes.NewReader(wheel))
	}))
	t.Cleanup(server.Close)
	area := t.TempDir()
	config, logPath := filepath.Join(area, "cloudflared.yml"), filepath.Join(area, "tunnel.log")
	must(t, os.WriteFile(config, []byte("{}\n"), 0600))
	log, err := os.OpenFile(logPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	must(t, err)
	command := exec.CommandContext(t.Context(), "/usr/local/bin/cloudflared", "tunnel", "--config", config, "--no-autoupdate", "--url", server.URL, "--metrics", "127.0.0.1:0", "--grace-period", "2s")
	command.Env = []string{"PATH=/usr/local/bin:/usr/bin:/bin"}
	command.Stdout, command.Stderr = log, log
	bindPublishedTunnelParent(t, command)
	must(t, command.Start())
	t.Cleanup(func() { _ = command.Process.Signal(os.Interrupt); _ = command.Wait(); _ = log.Close() })
	pattern := regexp.MustCompile(`https://[a-z0-9-]+\.trycloudflare\.com`)
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	client := &http.Client{Timeout: 3 * time.Second}
	for {
		raw, err := os.ReadFile(logPath)
		must(t, err)
		if endpoint := string(pattern.Find(raw)); endpoint != "" {
			request, err := http.NewRequestWithContext(ctx, "HEAD", endpoint+prefix+"health", nil)
			must(t, err)
			response, err := client.Do(request)
			if err == nil {
				_, _ = io.Copy(io.Discard, response.Body)
				_ = response.Body.Close()
				if response.StatusCode == http.StatusNoContent && response.Header.Get("X-Cozy-Test-Wheel") == "1" {
					t.Logf("bounded public wheel transport ready, owned pid=%d", command.Process.Pid)
					return endpoint + prefix + filename
				}
			}
		}
		select {
		case <-ctx.Done():
			t.Fatal("public fixture transport did not become ready")
		case <-ticker.C:
		}
	}
}
