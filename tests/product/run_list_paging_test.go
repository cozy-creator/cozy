//go:build !windows

package producttest

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/calcifer"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRunListReadsBoundedHistoryPages(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	for index := 1; index <= 151; index++ {
		id := fmt.Sprintf("history-%03d", index)
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: strings.Repeat("1", 64), Package: "proof/history", Entrypoint: "run", Payload: []byte("{}")})
		fatal(t, problem)
	}
	var before int64
	seen := map[int64]bool{}
	for _, count := range []int{50, 50, 50, 1} {
		page, problem := store.RequestsBefore("", "proof/history", 50, before)
		fatal(t, problem)
		if len(page) != count {
			t.Fatalf("page length=%d want=%d", len(page), count)
		}
		for _, row := range page {
			if seen[row.Number] || row.Number < 1 || before > 0 && row.Number >= before {
				t.Fatalf("unstable history page: %+v", row)
			}
			seen[row.Number] = true
		}
		before = page[len(page)-1].Number
	}
	if len(seen) != 151 || !seen[1] || !seen[151] {
		t.Fatal("history is not fully reachable")
	}
}

func TestRunListInteractivePagingCachesRowsAndRefreshesVisibleHistory(t *testing.T) {
	var head atomic.Int64
	head.Store(151)
	var newerProgress atomic.Bool
	var mu sync.Mutex
	var requests [][2]int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			return
		}
		if r.URL.Path != "/v1/requests" {
			t.Errorf("per-row or unexpected status request: %s", r.URL.Path)
			http.Error(w, "unexpected", 500)
			return
		}
		limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
		before, _ := strconv.ParseInt(r.URL.Query().Get("before"), 10, 64)
		mu.Lock()
		requests = append(requests, [2]int64{int64(limit), before})
		mu.Unlock()
		start := head.Load()
		if before > 0 {
			start = min(start, before-1)
		}
		rows := make([]api.Lifecycle, 0, limit)
		for number := start; number > 0 && len(rows) < limit; number-- {
			row := api.Lifecycle{Number: number, RequestID: fmt.Sprintf("request-%d", number), Package: "p", Function: "run", Status: "completed"}
			if number == 90 {
				row.Status = "in_progress"
				row.ProgressStage = "older-stage"
				if newerProgress.Load() {
					row.ProgressStage = "older-updated"
				}
			}
			rows = append(rows, row)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"requests": rows})
	}))
	defer server.Close()
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	held, problem := calcifer.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
	fatal(t, problem)
	defer held.Release()
	_, problem = api.Mint(layout)
	fatal(t, problem)
	master, cmd := startPTY(t, layout.Root, 24, 120, "run", "list", "--fields=number,progress")
	chunks := make(chan string, 32)
	go func() {
		defer close(chunks)
		data := make([]byte, 8192)
		for {
			n, err := master.Read(data)
			if n > 0 {
				chunks <- string(data[:n])
			}
			if err != nil {
				return
			}
		}
	}()
	var output string
	await := func(want string) string {
		t.Helper()
		deadline := time.NewTimer(10 * time.Second)
		defer deadline.Stop()
		for {
			select {
			case chunk, ok := <-chunks:
				if !ok {
					t.Fatalf("terminal closed before %q: %s", want, output)
				}
				output += chunk
				if strings.Contains(output, want) {
					frame := output
					output = ""
					return frame
				}
			case <-deadline.C:
				t.Fatalf("terminal did not show %q: %s", want, output)
			}
		}
	}
	await("/50+")
	mu.Lock()
	initial := append([][2]int64(nil), requests...)
	mu.Unlock()
	for _, request := range initial {
		if request != [2]int64{50, 0} {
			t.Fatalf("initial board eagerly read history: %v", initial)
		}
	}
	_, _ = master.Write([]byte("\x1b[6~\x1b[6~"))
	await("/100+")
	_, _ = master.Write([]byte("\x1b[6~"))
	await("older-stage")
	newerProgress.Store(true)
	await("older-updated")
	head.Store(152)
	frame := await("/101+")
	if !strings.Contains(frame, "rows 52-") {
		t.Fatalf("new head moved the selected history anchor: %s", frame)
	}
	_, _ = master.Write([]byte("\x1b[F"))
	await("/151+")
	_, _ = master.Write([]byte("\x1b[F"))
	await("/152")
	_, _ = master.Write([]byte("q"))
	for range chunks {
	}
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	for _, request := range requests {
		if request[0] > 100 {
			t.Fatalf("board fetched unbounded history: %v", requests)
		}
	}
	mu.Unlock()
	// Piped output is bounded by default; all history remains explicit.
	for _, test := range []struct {
		args  []string
		count int
	}{{[]string{"--json", "--full"}, 50}, {[]string{"--json", "--full", "--limit=0"}, 152}} {
		code, out := runCozy(t, layout.Root, append([]string{"run", "list"}, test.args...)...)
		var document struct {
			Invocations []any `json:"invocations"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Invocations) != test.count {
			t.Fatalf("snapshot [%d] wanted%d: %s", code, test.count, out)
		}
	}
}

func TestRunListCanExitWhilePageFetchIsPending(t *testing.T) {
	started := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			return
		}
		started <- struct{}{}
		<-r.Context().Done()
	}))
	defer server.Close()
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	held, problem := calcifer.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
	fatal(t, problem)
	defer held.Release()
	_, problem = api.Mint(layout)
	fatal(t, problem)
	master, cmd := startPTY(t, layout.Root, 24, 120, "run", "list")
	done := make(chan error, 1)
	go func() { _, _ = io.Copy(io.Discard, master); done <- cmd.Wait() }()
	select {
	case <-started:
	case <-time.After(10 * time.Second):
		t.Fatal("page fetch did not start")
	}
	_, _ = master.Write([]byte("q"))
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("exit blocked behind page fetch")
	}
}
