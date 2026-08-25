package coord

import (
	"context"
	"errors"
	"net"
	"syscall"

	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/peer"
)

// The OS process-birth identity, applied to the wire. A worker's protocol identities
// (`worker_id`, `instance_id`, `session_id`, the two ordinals) say WHO it claims to be;
// SO_PEERCRED says which process is actually on the other end of this socket.
//
// This exists because it was needed. Observed live: a supervisor from a PREVIOUS
// LocalService, still reconnecting on its backoff, dialled the socket the next
// LocalService created at the same path, registered under the same (stable) instance_id
// with a fresh session, and was adopted — clobbering the real worker's readiness with its
// own ERROR reports. Nothing in the protocol's own vocabulary catches that: both peers
// present a valid session and a valid instance. The kernel does.

type peerIdentity struct {
	PID int32
	UID uint32
}

func (peerIdentity) AuthType() string { return "unix-peer" }

// unixPeer is a TransportCredentials that does no cryptography and one thing: it records
// the connecting process's kernel-attested pid. A unix socket is a local boundary, and
// this is the only identity on it that a peer cannot claim for itself.
type unixPeer struct{}

func (unixPeer) ServerHandshake(raw net.Conn) (net.Conn, credentials.AuthInfo, error) {
	uc, ok := raw.(*net.UnixConn)
	if !ok {
		return raw, peerIdentity{}, nil
	}
	rc, err := uc.SyscallConn()
	if err != nil {
		return raw, peerIdentity{}, nil
	}
	var cred *syscall.Ucred
	var inner error
	if err := rc.Control(func(fd uintptr) {
		cred, inner = syscall.GetsockoptUcred(int(fd), syscall.SOL_SOCKET, syscall.SO_PEERCRED)
	}); err != nil || inner != nil || cred == nil {
		return raw, peerIdentity{}, nil
	}
	return raw, peerIdentity{PID: cred.Pid, UID: cred.Uid}, nil
}

func (unixPeer) ClientHandshake(context.Context, string, net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return nil, nil, errors.New("the coordinator is the server; it never dials a worker")
}

func (unixPeer) Info() credentials.ProtocolInfo {
	return credentials.ProtocolInfo{SecurityProtocol: "unix-peer"}
}

func (u unixPeer) Clone() credentials.TransportCredentials { return u }

func (unixPeer) OverrideServerName(string) error { return nil }

// peerPID is the kernel-attested pid of whoever is on this stream, or 0 when the
// platform did not answer.
func peerPID(ctx context.Context) int32 {
	p, ok := peer.FromContext(ctx)
	if !ok || p.AuthInfo == nil {
		return 0
	}
	id, ok := p.AuthInfo.(peerIdentity)
	if !ok {
		return 0
	}
	return id.PID
}
