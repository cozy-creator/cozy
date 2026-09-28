package producttest

import (
	"context"
	"errors"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
)

func TestDaemonHTTPDrainFlushesReplyAndEndsStream(t *testing.T) {
	buffered, canceled, finish := make(chan struct{}), make(chan struct{}), make(chan struct{})
	streamStarted, streamStopped := make(chan struct{}), make(chan struct{})
	const reply = `{"shutting_down":true}`
	server := daemon.NewHTTPServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/events" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: ready\n\n")
			w.(http.Flusher).Flush()
			close(streamStarted)
			<-r.Context().Done()
			close(streamStopped)
			return
		}
		_, _ = io.WriteString(w, reply) // remains in net/http's response buffer
		close(buffered)
		<-r.Context().Done()
		close(canceled)
		<-finish
	}))
	v4, v6, address, problem := api.Listeners(0)
	fatal(t, problem)
	go func() { _ = server.Serve(v4) }()
	if v6 != nil {
		go func() { _ = server.Serve(v6) }()
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	})
	client := &http.Client{Timeout: 5 * time.Second}
	stream, err := client.Get("http://" + address + "/events")
	must(t, err)
	defer stream.Body.Close()
	<-streamStarted
	response := make(chan error, 1)
	go func() {
		r, err := client.Post("http://"+address+"/down", "application/json", strings.NewReader(`{}`))
		if err == nil {
			var raw []byte
			raw, err = io.ReadAll(r.Body)
			_ = r.Body.Close()
			if err == nil && string(raw) != reply {
				err = errors.New("shutdown response was truncated")
			}
		}
		response <- err
	}()
	<-buffered
	drained := make(chan error, 1)
	go func() { drained <- server.Shutdown(t.Context()) }()
	<-canceled
	<-streamStopped
	select {
	case err := <-drained:
		close(finish)
		t.Fatalf("shutdown returned before the buffered response handler finished: %v", err)
	default:
	}
	close(finish)
	must(t, <-response)
	must(t, <-drained)
}

// The drain is proven deterministically above; these real up/down cycles prove the CLI
// receives the final response.
func TestRepeatedCLIUpDownReceivesTheFinalResponse(t *testing.T) {
	root := t.TempDir()
	for i := range 3 {
		if code, out := runCozy(t, root, "up", "--json"); code != 0 {
			t.Fatalf("up iteration%d: %d %s", i, code, out)
		}
		if code, out := runCozy(t, root, "down", "--json"); code != 0 ||
			!strings.Contains(out, `"daemon":"stopped"`) {
			t.Fatalf("down iteration%d: %d %s", i, code, out)
		}
		if state := daemon.Probe(config.Config{Home: root}); state.Up {
			t.Fatalf("down left its daemon alive: %s\n%s", state.Details, tail(filepath.Join(root, "daemon.log")))
		}
	}
}

func TestDaemonHTTPDrainHonorsItsCancellationBound(t *testing.T) {
	started, canceled, finish := make(chan struct{}), make(chan struct{}), make(chan struct{})
	server := daemon.NewHTTPServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-r.Context().Done()
		close(canceled)
		<-finish
	}))
	v4, v6, address, problem := api.Listeners(0)
	fatal(t, problem)
	if v6 != nil {
		_ = v6.Close()
	}
	go func() { _ = server.Serve(v4) }()
	done := make(chan error, 1)
	go func() {
		r, err := (&http.Client{Timeout: 5 * time.Second}).Get("http://" + address)
		if r != nil {
			_ = r.Body.Close()
		}
		done <- err
	}()
	<-started
	ctx, cancel := context.WithCancel(t.Context())
	drained := make(chan error, 1)
	go func() { drained <- server.Shutdown(ctx) }()
	<-canceled
	cancel()
	if err := <-drained; !errors.Is(err, context.Canceled) {
		t.Fatal("shutdown ignored its bound", err)
	}
	if err := <-done; err == nil {
		t.Fatal("the uncooperative connection remained open")
	}
	close(finish)
}
