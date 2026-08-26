//go:build windows

package coord

import (
	"context"
	"errors"
	"net"

	"google.golang.org/grpc/credentials"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// bootstrapRequired: a loopback TCP port has no SO_PEERCRED, so the kernel cannot say
// which process dialled. The substitute (#449) is a PER-SPAWN bootstrap credential: the
// launcher mints one, hands it to the child through its environment, and Register must
// present it — a fixed port with no credential is exactly the door this refuses to open.
const bootstrapRequired = true

// listenLocal binds an EPHEMERAL loopback port — chosen by the kernel, never configured —
// and returns the bound address for workers to dial. Windows' AF_UNIX exists, but the
// Python side of this protocol cannot dial it (asyncio has no unix sockets there), so
// loopback is the honest local transport.
func listenLocal(string) (net.Listener, credentials.TransportCredentials, string, *exit.Error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, nil, "", exit.New(exit.Conflict,
			"cannot bind a loopback port for the worker protocol: %s", err)
	}
	return ln, loopbackPeer{}, ln.Addr().String(), nil
}

// cleanupLocal has nothing to remove: a TCP port dies with its listener.
func cleanupLocal(string) {}

// loopbackPeer is the no-kernel-identity counterpart of unixPeer: it records nothing, so
// `peerPID` answers 0 ("the platform did not answer") and the pid adoption guard yields to
// the bootstrap credential check in Register.
type loopbackPeer struct{}

func (loopbackPeer) ServerHandshake(raw net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return raw, peerIdentity{}, nil
}

func (loopbackPeer) ClientHandshake(context.Context, string, net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return nil, nil, errors.New("the coordinator is the server; it never dials a worker")
}

func (loopbackPeer) Info() credentials.ProtocolInfo {
	return credentials.ProtocolInfo{SecurityProtocol: "loopback-peer"}
}

func (l loopbackPeer) Clone() credentials.TransportCredentials { return l }

func (loopbackPeer) OverrideServerName(string) error { return nil }
