// pod-supervisor is the rented pod's process supervisor: PID 1 of its single container and
// the whole of this repo's guest side. It was two binaries until cl-036 — `cozy-bootstrap`
// supervised and `cozy-media` served — and they are now one image artifact, one pinned
// commit, and one process.
//
// IT DIALS NOTHING. The fence holds this binary WHOLE — `cmd/pod-supervisor` and
// `internal/podmedia` alike — to an absolute no-egress rule with no exception of any kind.
// That was the merge's precondition and it is why the merge is admissible: cl-036's diet
// deleted the supervisor's one outbound door along with the two provision documents it
// fetched, so there is no file left to carve out.
//
// It does four things and nothing else:
//
//   - It PARSES ITS GRANT. The whole launch surface is six ALLOWLISTED `COZY_*`
//     environment variables; it takes no arguments and reads no configuration file, and an
//     unrecognized `COZY_*` name is a boot failure rather than an ignored default.
//   - It MINTS the pod's TLS leaf and boot id, so no long-lived credential ships in the
//     image.
//   - It SERVES THE MEDIA PLANE in this process (`internal/podmedia`), bound before any
//     child exists — a grant it cannot authenticate is a pod that does not boot.
//   - It EXECS ONE CHILD, the control runtime's materialize/launch adapter, by absolute
//     path with no PATH and a closed five-name environment, then HMACs the opaque
//     readiness payload the adapter drops into the envelope the media plane serves at
//     `GET /v1/bootstrap/receipt`, and supervises until something exits.
//
// THE PRICE OF THE MERGE, AND WHAT PAYS IT. This process is PID 1, the parser of
// attacker-influenced request bytes, and the holder of the readiness HMAC key. Two
// boundaries buy that concentration down and both are fenced:
//
//   - SUPERVISION IS NOT REACHABLE FROM A REQUEST PATH. Everything that execs, signals or
//     reaps lives in `supervise.go` of THIS package, which is package main and therefore
//     cannot be imported at all; the handlers live in `internal/podmedia`, which the fence
//     forbids `os/exec`, signals and process handles outright.
//   - THE HMAC KEY IS WIPED. Its environment name is unset the instant it is decoded, and
//     the bytes are zeroed by the one-shot `attemptKey` the moment the envelope is sealed
//     (`receipt.go`). A second read is an error, not a second signature.
//
// What did NOT change with the merge: media↔worker decoupling is still BY THE FILESYSTEM
// ALONE — the worker is a separately exec'd process either way — and the envelope ceiling
// both ends agree on is `mediawire.MaxReceiptBytes`, read and never restated.
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
		fmt.Fprintln(os.Stderr, "pod-supervisor: this entrypoint takes no arguments")
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintln(os.Stderr, "pod-supervisor:", err)
		os.Exit(1)
	}
}
