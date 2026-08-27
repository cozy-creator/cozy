package main

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
)

// THE MEDIA SERVER'S ONE BIND SITE, and the reason it is a file of its own.
//
// `internal/api/listen.go` is the OWNER's one bind site and it refuses a non-loopback
// address, because a loopback bind is the whole boundary of a local client API. This is a
// different program with a different boundary: it runs INSIDE a rented pod, it is the leg
// an off-machine owner reaches, and off-loopback is what it is for. The two rules are both
// real and they are not the same rule, so they live in two files and the fence names each
// one by the program it belongs to (#506b: the api fence's one-bind-site rule STANDS —
// this is not a second bind in the owner's binary, it is the pod's own).
//
// What replaces "loopback only" here is the pair below it: a non-loopback listener REQUIRES
// TLS, exactly as the worker's own does (`serve --tls-cert/--tls-key`), and there is no
// unauthenticated RENTER route. The bootstrap receipt contains no capability and is
// authenticated by its attempt HMAC before Tensorhub trusts the serving certificate. A
// plaintext media server reachable off-loopback would still put owner tokens in the clear.
func (s *server) serve() int {
	if !loopback(s.opt.listen) && (s.opt.cert == "" || s.opt.key == "") {
		return fatal("refusing to bind %s without TLS: this server is reached from off the "+
			"pod and the bearer it checks would travel in the clear", s.opt.listen)
	}
	ln, err := net.Listen("tcp", s.opt.listen) //cozy:allow the POD's media server (cl-014) — a pod-side program, not the owner's local API; its own bind rule is TLS-or-loopback, enforced above
	if err != nil {
		return fatal("cannot bind %s: %v", s.opt.listen, err)
	}
	bound := ln.Addr().String()
	// THE FILE-HANDOFF DISCOVERY CONTRACT, the same one the worker keeps for `control.addr`
	// (#436): the process that binds publishes the address it actually got, so `:0` is a
	// usable request and nobody has to agree a port in advance.
	if s.opt.out != "" {
		if err := os.MkdirAll(s.opt.out, 0o755); err == nil {
			staging := filepath.Join(s.opt.out, "media.addr.staging")
			if os.WriteFile(staging, []byte(bound+"\n"), 0o644) == nil {
				_ = os.Rename(staging, filepath.Join(s.opt.out, "media.addr"))
			}
		}
	}
	srv := &http.Server{Handler: s.routes()}
	fmt.Printf("[media] serving %s root=%s quota=%d B plans=%v\n",
		bound, s.opt.root, s.opt.quota, s.opt.plans != "")
	os.Stdout.Sync()
	if s.opt.cert != "" {
		err = srv.ServeTLS(ln, s.opt.cert, s.opt.key)
	} else {
		err = srv.Serve(ln)
	}
	if err != nil && err != http.ErrServerClosed {
		return fatal("serving ended: %v", err)
	}
	return 0
}

// loopback answers whether an address is one this machine alone can reach. It is the ONLY
// case in which this server may run without TLS, and it exists for the harness that runs
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
