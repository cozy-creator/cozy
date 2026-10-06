package calcifer

import (
	"context"
	"net"
	"net/http"
)

// HTTPServer owns both loopback listeners and their active handlers. Canceling its
// request context ends event streams; Shutdown then waits for buffered replies.
type HTTPServer struct {
	server *http.Server
	cancel context.CancelFunc
}

func NewHTTPServer(handler http.Handler) *HTTPServer {
	ctx, cancel := context.WithCancel(context.Background())
	return &HTTPServer{server: &http.Server{Handler: handler,
		BaseContext: func(net.Listener) context.Context { return ctx }}, cancel: cancel}
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
