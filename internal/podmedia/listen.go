package podmedia

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
)

// THE MEDIA PLANE'S ONE BIND SITE, and the reason it is a file of its own.
//
// `internal/api/listen.go` is the OWNER's one bind site and it refuses a non-loopback
// address, because a loopback bind is the whole boundary of a local client API. This is a
// different surface with a different boundary: it runs INSIDE a rented pod, it is the leg
// an off-machine owner reaches, and off-loopback is what it is for. The two rules are both
// real and they are not the same rule, so they live in two files and the fence names each
// one by the program part it belongs to (#506b: the api fence's one-bind-site rule STANDS —
// this is not a second bind in the owner's binary, it is the pod's own).
//
// What replaces "loopback only" here is the pair below it: a non-loopback listener REQUIRES
// TLS, exactly as the worker's own does (`serve --tls-cert/--tls-key`), and there is no
// unauthenticated RENTER route. The bootstrap receipt contains no capability and is
// authenticated by its attempt HMAC before Tensorhub trusts the serving certificate. A
// plaintext media plane reachable off-loopback would still put owner tokens in the clear.
//
// BIND IS SEPARATE FROM SERVE (cl-036). The pod supervisor now runs this plane in its own
// process, so a plane that cannot bind must be a BOOT failure and not a child that dies
// later: `Bind` validates the grant, lays out the subtree and takes the socket, and only a
// bound plane is ever served. The supervisor execs the adapter after Bind returns, which
// is also why the media listener is up before anything else in the pod exists.

// Plane is a bound, validated media plane. It exists only if the grant authenticated and
// the socket was taken.
type Plane struct {
	srv *server
	ln  net.Listener
}

// Bind validates the grant, prepares the subtree and takes the listener.
func Bind(opt Options) (*Plane, error) {
	if err := opt.prepare(); err != nil {
		return nil, err
	}
	if !loopback(opt.Listen) && (opt.Cert == "" || opt.Key == "") {
		return nil, fmt.Errorf("refusing to bind %s without TLS: this plane is reached from "+
			"off the pod and the bearer it checks would travel in the clear", opt.Listen)
	}
	ln, err := net.Listen("tcp", opt.Listen) //cozy:allow the POD's media plane (cl-014) — a pod-side surface, not the owner's local API; its own bind rule is TLS-or-loopback, enforced above
	if err != nil {
		return nil, fmt.Errorf("cannot bind %s: %w", opt.Listen, err)
	}
	return &Plane{srv: &server{opt: opt}, ln: ln}, nil
}

// Addr is the address actually bound, so `:0` is a usable request and nobody has to agree
// a port in advance.
func (p *Plane) Addr() string { return p.ln.Addr().String() }

// Close drops the listener without serving it.
func (p *Plane) Close() error { return p.ln.Close() }

// Serve runs until the listener fails or is closed.
func (p *Plane) Serve() error {
	srv := &http.Server{Handler: baseline(p.srv.routes())}
	fmt.Printf("[media] serving %s root=%s quota=%d B\n",
		p.Addr(), p.srv.opt.Root, p.srv.opt.Quota)
	_ = os.Stdout.Sync()
	var err error
	if p.srv.opt.Cert != "" {
		err = srv.ServeTLS(p.ln, p.srv.opt.Cert, p.srv.opt.Key)
	} else {
		err = srv.Serve(p.ln)
	}
	// A closed listener is how the supervisor stops this leg; it is not a failure. Any
	// other end IS one, and the supervisor treats it as fatal to the pod.
	if err != nil && !errors.Is(err, http.ErrServerClosed) && !errors.Is(err, net.ErrClosed) {
		return fmt.Errorf("the media plane stopped serving: %w", err)
	}
	return nil
}

// baseline is what EVERY response carries, refusals included — the same posture the
// owner API's baseline() takes (internal/api/api.go), because this is the one
// off-loopback listener and it was the one door sending none of it (cl-026).
func baseline(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		// No response is cacheable: inputs and outputs are attempt-scoped and live.
		h.Set("Cache-Control", "no-store")
		h.Set("Referrer-Policy", "no-referrer")
		// Nothing this server renders may load, connect, or execute anything.
		h.Set("Content-Security-Policy",
			"default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
		// NO Access-Control-Allow-* header is set anywhere in this package, deliberately.
		next.ServeHTTP(w, r)
	})
}

// loopback answers whether an address is one this machine alone can reach. It is the ONLY
// case in which this plane may run without TLS, and it exists for the harness that runs
// the pod and the owner on one box.
func loopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
