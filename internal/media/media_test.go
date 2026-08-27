package media

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

const testTransferBudget = 20 * time.Millisecond

func testClient(t *testing.T, server *httptest.Server, maxObject int64) *Client {
	t.Helper()
	client, problem := Dial(Spec{Addr: strings.TrimPrefix(server.URL, "http://"),
		Token: secret.New("test-token")}, testTransferBudget, maxObject)
	if problem != nil {
		t.Fatal(problem)
	}
	return client
}

func TestLengthDerivedDeadlineDoesNotRequireGigabitMedia(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()
	client := testClient(t, server, 512<<20)
	deadline := client.transferDeadline(2 << 30)
	if deadline <= 16*time.Second || deadline < 34*time.Minute {
		t.Fatalf("2 GiB deadline = %s", deadline)
	}
}

func TestThrottledOutputUsesDeclaredTransferAllowance(t *testing.T) {
	data := bytes.Repeat([]byte("video"), 24<<10)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(data)))
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		const writes = 8
		for index := 0; index < writes; index++ {
			start, stop := index*len(data)/writes, (index+1)*len(data)/writes
			_, _ = w.Write(data[start:stop])
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer server.Close()
	client := testClient(t, server, int64(len(data)))
	destination := filepath.Join(t.TempDir(), "video.mp4")
	started := time.Now()
	written, problem := client.GetOutputTo("attempt", "video", destination,
		digestOf(data), int64(len(data)))
	if problem != nil || written != int64(len(data)) || time.Since(started) <= testTransferBudget {
		t.Fatalf("download written=%d elapsed=%s problem=%v", written, time.Since(started), problem)
	}
	stored, err := os.ReadFile(destination)
	if err != nil || !bytes.Equal(stored, data) {
		t.Fatalf("stored output length=%d error=%v", len(stored), err)
	}
}

func TestThrottledInputUsesDeclaredTransferAllowance(t *testing.T) {
	data := bytes.Repeat([]byte("input"), 100<<10)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hashData := make([]byte, 0, len(data))
		buffer := make([]byte, 32<<10)
		for {
			n, err := r.Body.Read(buffer)
			if n > 0 {
				hashData = append(hashData, buffer[:n]...)
				time.Sleep(3 * time.Millisecond)
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"path": "/pod/input", "digest": digestOf(hashData), "length": len(hashData),
		})
	}))
	defer server.Close()
	client := testClient(t, server, int64(len(data)))
	source := filepath.Join(t.TempDir(), "reference.mp4")
	if err := os.WriteFile(source, data, 0o600); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	path, problem := client.PutInputFile("blob", source, digestOf(data), int64(len(data)))
	if problem != nil || path != "/pod/input" || time.Since(started) <= testTransferBudget {
		t.Fatalf("upload path=%s elapsed=%s problem=%v", path, time.Since(started), problem)
	}
	if client.transferDeadline(int64(len(data))) <= testTransferBudget {
		t.Fatal("declared bytes added no transfer allowance")
	}
}
