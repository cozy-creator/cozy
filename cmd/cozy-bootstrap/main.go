// cozy-bootstrap is THE POD'S SUPERVISOR: PID 1 of a rented pod's single container, and
// the process that turns a set of injected environment grants into a running pod.
//
// It does four things and nothing else. It mints the pod's TLS key pair and the
// token-hash file that `cozy-media` authenticates renters against — so no long-lived
// credential ships inside the image. It fetches the two provision documents by EXACT
// grant: one credential-free HTTPS GET each, at a declared byte length, verified against
// a declared sha256 before the bytes are published, and it treats their contents as
// opaque. It execs the two children — `cozy-media` (this repo's byte plane) and the
// control runtime's materialize/launch adapter — with a five-name environment, by
// absolute path, with no PATH. And it publishes the readiness receipt: the adapter drops
// an opaque payload on the filesystem, this process HMACs it under the attempt key and
// writes the envelope `cozy-media` serves at `GET /v1/bootstrap/receipt`.
//
// It lives beside `cmd/cozy-media` because they are the two Go processes of one pod image
// and one of them starts the other. The envelope ceiling both ends must agree on is
// `mediawire.MaxReceiptBytes` — read, never restated: this file wrote it out as a second
// `64 << 10` once, and two silent copies of one bound is exactly the defect cl-031 closed.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	if len(os.Args) != 1 {
		fmt.Fprintln(os.Stderr, "cozy-bootstrap: this entrypoint takes no arguments")
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintln(os.Stderr, "cozy-bootstrap:", err)
		os.Exit(1)
	}
}
