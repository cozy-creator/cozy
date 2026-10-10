package daemon

import (
	"context"
	"io"
	"net"
	"net/http"
	"sync/atomic"
)

// HTTPServer owns both loopback listeners and their active handlers. It answers from the
// moment it serves: until Open hands it the API, every request is told the daemon is
// starting, so a bound port never holds connections nobody accepts. Canceling its request
// context ends event streams; Shutdown then waits for buffered replies.
type HTTPServer struct {
	server *http.Server
	cancel context.CancelFunc
	api    atomic.Pointer[http.Handler]
}

func NewHTTPServer() *HTTPServer {
	ctx, cancel := context.WithCancel(context.Background())
	s := &HTTPServer{cancel: cancel}
	s.server = &http.Server{Handler: http.HandlerFunc(s.serveHTTP),
		BaseContext: func(net.Listener) context.Context { return ctx }}
	return s
}

// Open starts answering with the API.
func (s *HTTPServer) Open(api http.Handler) { s.api.Store(&api) }

func (s *HTTPServer) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if api := s.api.Load(); api != nil {
		(*api).ServeHTTP(w, r)
		return
	}
	h := w.Header()
	h.Set("Content-Type", "application/json")
	h.Set("X-Content-Type-Options", "nosniff")
	h.Set("Cache-Control", "no-store")
	h.Set("Retry-After", "1")
	w.WriteHeader(http.StatusServiceUnavailable)
	_, _ = io.WriteString(w, `{"error":{"code":"daemon_starting","message":"the Cozy daemon is opening its records","remedy":"retry in a moment"}}`+"\n")
}

func (s *HTTPServer) Serve(listener net.Listener) error { return s.server.Serve(listener) }

func (s *HTTPServer) Shutdown(ctx context.Context) error {
	s.cancel()
	if err := s.server.Shutdown(ctx); err != nil {
		// An uncooperative handler cannot prevent bounded daemon teardown.
		_ = s.server.Close()
		return err
	}
	return nil
}
