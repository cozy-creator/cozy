package machinev1

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"math/big"
	"net"
	"testing"
	"time"

	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
)

// A socket whose SYN is never answered (a fresh pod's port mapping changing under it) does not
// hold the connection: a new socket joins after synWait and reaches the machine. gRPC's own
// dialer waited on the one socket for its whole 20 s connect.
func TestAnUnansweredSYNDoesNotHoldTheConnection(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	must(t, err)
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(serverTLS(t))))
	pb.RegisterMachineServer(server, pb.UnimplementedMachineServer{})
	go server.Serve(listener)
	t.Cleanup(server.Stop)

	real, attempts := dialSocket, 0
	dialSocket = func(ctx context.Context, addr string) (net.Conn, error) {
		attempts++
		if attempts == 1 {
			addr = "192.0.2.1:443" // TEST-NET-1: nothing answers
		}
		return real(ctx, addr)
	}
	t.Cleanup(func() { dialSocket = real })

	conn, err := grpc.NewClient(listener.Addr().String(), dialing(&tls.Config{InsecureSkipVerify: true})...)
	must(t, err)
	t.Cleanup(func() { conn.Close() })
	began := time.Now()
	conn.Connect()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for s := conn.GetState(); s != connectivity.Ready; s = conn.GetState() {
		if !conn.WaitForStateChange(ctx, s) {
			t.Fatalf("not connected after %s", time.Since(began))
		}
	}
	if took := time.Since(began); attempts < 2 || took > 4*time.Second {
		t.Fatalf("connected after %s and %d attempts", took, attempts)
	}
}

func serverTLS(t *testing.T) *tls.Config {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour)}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	must(t, err)
	return &tls.Config{Certificates: []tls.Certificate{{Certificate: [][]byte{der}, PrivateKey: key}}}
}
