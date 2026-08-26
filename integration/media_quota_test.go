package integration

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

func TestMediaQuotaAcrossWorkerAndHTTPProcesses(t *testing.T) {
	_, here, _, _ := runtime.Caller(0)
	repo := filepath.Dir(filepath.Dir(here))
	base := t.TempDir()
	root, handoff := filepath.Join(base, "media"), filepath.Join(base, "handoff")
	tokens := filepath.Join(base, "tokens")
	if err := os.WriteFile(tokens, []byte(secret.HashLine(secret.New("test-token"))+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(base, "cozy-media")
	build := exec.Command("go", "build", "-o", binary, "./cmd/cozy-media")
	build.Dir = repo
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build cozy-media: %v: %s", err, output)
	}
	process := exec.Command(binary,
		"--listen", "127.0.0.1:0", "--root", root, "--tokens", tokens,
		"--out", handoff, "--quota", "10", "--max-body", "10")
	process.Dir = repo
	var processLog bytes.Buffer
	process.Stdout, process.Stderr = &processLog, &processLog
	if err := process.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = process.Process.Kill()
		_ = process.Wait()
	})

	addressFile := filepath.Join(handoff, "media.addr")
	deadline := time.Now().Add(15 * time.Second)
	var address string
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(addressFile); err == nil {
			address = string(bytes.TrimSpace(data))
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if address == "" {
		t.Fatalf("cozy-media did not publish its address: %s", processLog.String())
	}
	call := func(method, path string, body io.Reader) *http.Response {
		t.Helper()
		req, err := http.NewRequest(method, "http://"+address+path, body)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer test-token")
		response, err := (&http.Client{Timeout: 3 * time.Second}).Do(req)
		if err != nil {
			t.Fatalf("media call: %v; process log: %s", err, processLog.String())
		}
		return response
	}
	reserved := call(http.MethodPost, "/v1/outputs/a111?max_bytes=6", nil)
	var reservation struct {
		Dir string `json:"dir"`
	}
	if err := json.NewDecoder(reserved.Body).Decode(&reservation); err != nil {
		reserved.Body.Close()
		t.Fatal(err)
	}
	reserved.Body.Close()
	if reserved.StatusCode != http.StatusCreated || reservation.Dir == "" {
		t.Fatalf("reserve = %d, %#v", reserved.StatusCode, reservation)
	}

	started, finished := make(chan struct{}), make(chan error, 1)
	go func() {
		path := filepath.Join(reservation.Dir, "image")
		for i := 0; i < 100; i++ {
			if err := os.WriteFile(path, bytes.Repeat([]byte("x"), i%6+1), 0o644); err != nil {
				finished <- err
				return
			}
			if i == 0 {
				close(started)
			}
			time.Sleep(time.Millisecond)
		}
		finished <- nil
	}()
	<-started
	// The media server is a separate process, and this direct writer is active while its
	// HTTP admission runs. No in-process mutex can serialize these two filesystem users.
	upload := call(http.MethodPut, "/v1/inputs/blob", bytes.NewReader([]byte("12345")))
	_, _ = io.Copy(io.Discard, upload.Body)
	upload.Body.Close()
	if err := <-finished; err != nil {
		t.Fatal(err)
	}
	if upload.StatusCode != http.StatusRequestEntityTooLarge && upload.StatusCode != http.StatusInsufficientStorage {
		t.Fatalf("HTTP upload racing direct worker publication = %d, want quota refusal", upload.StatusCode)
	}
	if _, err := os.Stat(filepath.Join(root, "inputs", "blob")); !os.IsNotExist(err) {
		t.Fatalf("refused HTTP bytes landed: %v", err)
	}
}
