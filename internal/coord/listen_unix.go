//go:build !windows

package coord

import (
	"net"
	"os"

	"google.golang.org/grpc/credentials"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// bootstrapRequired: the unix transport carries SO_PEERCRED, so the kernel already
// attests who dialled — no handed credential is needed.
const bootstrapRequired = false

// listenLocal binds the worker-protocol unix socket and returns it with the peer
// credentials and the address workers dial (the path itself).
func listenLocal(socket string) (net.Listener, credentials.TransportCredentials, string, *exit.Error) {
	// A stale socket file is a leftover, never evidence: the service lock already proved
	// no live owner exists on this root, so removing it is safe and required.
	_ = os.Remove(socket)
	ln, err := net.Listen("unix", socket)
	if err != nil {
		return nil, nil, "", exit.New(exit.Conflict, "cannot bind the worker socket %s: %s", socket, err).
			WithRemedy("another LocalService may own this root; `cozy status`")
	}
	return ln, unixPeer{}, socket, nil
}

func cleanupLocal(socket string) { _ = os.Remove(socket) }
