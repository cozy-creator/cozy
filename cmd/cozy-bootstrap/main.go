// cozy-bootstrap is THE POD'S SUPERVISOR: PID 1 of a rented pod's single container, and
// the process that turns a set of injected environment grants into a running pod.
//
// It does three things and nothing else, and IT DIALS NOTHING — cl-036 deleted its one
// outbound door with the two provision documents it fetched, so the fence now holds the
// pod supervisor to the same absolute no-egress rule as `cozy-media` beside it. It mints
// the pod's TLS key pair — so no long-lived credential ships inside the image. It execs
// the two children — `cozy-media` (this repo's byte plane) and the control runtime's
// materialize/launch adapter — by absolute path, with no PATH, each with a closed
// environment, handing each the validated renter token DIGESTS it needs. And it
// publishes the readiness receipt: the adapter drops an opaque payload on the filesystem,
// this process HMACs it under the attempt key and writes the envelope `cozy-media` serves
// at `GET /v1/bootstrap/receipt`.
//
// Its whole launch surface is eight ALLOWLISTED `COZY_*` variables; a pod boots
// ready-but-empty and its endpoint closure arrives after the RecordOwner connects.
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
