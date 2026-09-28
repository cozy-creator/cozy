package host

import (
	"crypto/tls"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"google.golang.org/grpc"
)

// The one machine endpoint: gRPC and HTTPS on one TLS listener with the machine's leaf,
// told apart by content type. A Hub older than M6 also probes the receipt on the media port,
// and the Runtime proves that port refuses a foreign bearer, so that port serves only those.

const maxMessageBytes = 16 << 20

type listeners struct {
	servers []*http.Server
	failed  chan error
}

func (l *listeners) close() {
	for _, s := range l.servers {
		_ = s.Close()
	}
}

func (m *Machine) listen() (*listeners, error) {
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{m.id.Leaf}}
	endpoint, err := net.Listen("tcp", net.JoinHostPort(m.grant.ListenHost, strconv.Itoa(m.grant.WorkerPort)))
	if err != nil {
		return nil, fmt.Errorf("bind the machine endpoint: %w", err)
	}
	mediaHost, mediaPort := m.grant.ListenHost, m.grant.MediaPort
	if mediaPort == 0 {
		mediaHost = "127.0.0.1"
	}
	media, err := net.Listen("tcp", net.JoinHostPort(mediaHost, strconv.Itoa(mediaPort)))
	if err != nil {
		endpoint.Close()
		return nil, fmt.Errorf("bind the receipt listener: %w", err)
	}
	m.mediaAddr = net.JoinHostPort("127.0.0.1", strconv.Itoa(media.Addr().(*net.TCPAddr).Port))
	grpcServer := grpc.NewServer(grpc.MaxRecvMsgSize(maxMessageBytes), grpc.MaxSendMsgSize(maxMessageBytes),
		grpc.UnknownServiceHandler(m.serveRPC))
	routes := http.NewServeMux()
	routes.HandleFunc("GET /v1/bootstrap/receipt", m.serveReceipt)
	routes.HandleFunc("GET /v1/health", serveHealth)
	routes.HandleFunc("GET /v1/objects/{digest}", m.serveObject)
	main := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor == 2 && strings.HasPrefix(r.Header.Get("Content-Type"), "application/grpc") {
			grpcServer.ServeHTTP(w, r)
			return
		}
		routes.ServeHTTP(w, r)
	})
	receiptOnly := http.NewServeMux()
	receiptOnly.HandleFunc("GET /v1/bootstrap/receipt", m.serveReceipt)
	receiptOnly.HandleFunc("GET /v1/health", serveHealth)
	errorLog := log.New(m.log, "cozy machine: ", 0)
	l := &listeners{failed: make(chan error, 2)}
	for _, s := range []struct {
		listener net.Listener
		handler  http.Handler
	}{{endpoint, main}, {media, receiptOnly}} {
		server := &http.Server{Handler: s.handler, TLSConfig: tlsConfig.Clone(), ErrorLog: errorLog, ReadHeaderTimeout: time.Minute}
		l.servers = append(l.servers, server)
		go func(listener net.Listener) {
			if err := server.ServeTLS(listener, "", ""); err != nil && !errors.Is(err, http.ErrServerClosed) {
				l.failed <- fmt.Errorf("the machine endpoint stopped serving: %w", err)
			}
		}(s.listener)
	}
	return l, nil
}

// serveReceipt answers 425 until the envelope is sealed. Any answer tells the Hub the
// machine is up; the sealed envelope tells it the Runtime is ready.
func (m *Machine) serveReceipt(w http.ResponseWriter, _ *http.Request) {
	envelope := m.receipt.Envelope()
	w.Header().Set("Cache-Control", "no-store")
	if envelope == nil {
		http.Error(w, `{"error":{"code":"media.bootstrap_pending","message":"the machine is booting"}}`, http.StatusTooEarly)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(envelope)
}

// serveHealth carries no data. A request presenting a credential is refused: this machine
// has no bearer, and the Runtime's readiness proves exactly that.
func serveHealth(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "" {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

var sha256Digest = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

// serveObject streams one object of the machine's TensorFS store with full Range semantics:
// the digest is its ETag, so If-Range holds exactly. A capability in the Authorization header
// (`Cozy-Cap <token>`) grants it.
func (m *Machine) serveObject(w http.ResponseWriter, r *http.Request) {
	digest := r.PathValue("digest")
	token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Cozy-Cap ")
	grant, err := capability.Verify(token, m.grant.WorkerID, m.claims.authorized, time.Now())
	switch {
	case err != nil:
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	case !sha256Digest.MatchString(digest) || !grant.AllowsObject(digest):
		http.Error(w, capability.ErrScope.Error(), http.StatusForbidden)
		return
	}
	hex := strings.TrimPrefix(digest, "sha256:")
	file, err := os.OpenFile(filepath.Join(m.layout.Store, "blobs", hex[:2], hex[2:4], hex), os.O_RDONLY|syscall.O_NOFOLLOW, 0)
	if err != nil {
		http.Error(w, "this machine holds no such object", http.StatusNotFound)
		return
	}
	defer file.Close()
	w.Header().Set("ETag", `"`+digest+`"`)
	w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
	w.Header().Set("Content-Type", contentType(r.URL.Query().Get("type")))
	http.ServeContent(w, r, "", time.Time{}, file)
}

// contentType is the media type a client asks for, from a small closed set.
func contentType(asked string) string {
	switch asked {
	case "video/mp4", "audio/mp4", "audio/mpeg", "audio/wav", "image/png", "image/jpeg", "image/webp",
		"video/mp2t", "application/json", "text/plain":
		return asked
	}
	return "application/octet-stream"
}
