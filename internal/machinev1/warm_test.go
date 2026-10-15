package machinev1

import (
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

// An open upload connection to a remote machine carries a call more often than TCP's
// retransmission timeout while no write uses it, so the next run's inputs do not restart from
// slow start. One to this computer's machine has no path to keep warm and carries none.
func TestAnIdleUploadConnectionIsKeptWarm(t *testing.T) {
	var calls atomic.Int64
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(serverTLS(t))),
		grpc.UnknownServiceHandler(func(any, grpc.ServerStream) error {
			calls.Add(1)
			return status.Error(codes.Unimplemented, "no such method")
		}))
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	must(t, err)
	go func() { _ = server.Serve(listener) }()
	defer server.Stop()
	public, private, _ := ed25519.GenerateKey(nil)
	client, err := Dial(listener.Addr().String(), &tls.Config{InsecureSkipVerify: true}, "worker",
		Signer{Public: public, Sign: func(b []byte) []byte { return ed25519.Sign(private, b) }})
	must(t, err)
	defer client.Close()
	upload, _ := client.uploads()
	upload.Connect()
	for upload.GetState() != connectivity.Ready {
		upload.WaitForStateChange(t.Context(), upload.GetState())
	}
	time.Sleep(5 * warmEvery)
	if calls.Load() != 0 {
		t.Fatal("a connection to this computer's machine was kept warm")
	}
	warming, stop := context.WithCancel(t.Context())
	go client.warm(warming)
	time.Sleep(10 * warmEvery)
	if warmed := calls.Load(); warmed < 5 {
		t.Fatalf("an idle upload connection carried %d calls in %v", warmed, 10*warmEvery)
	}
	stop()
	time.Sleep(warmEvery)
	stopped := calls.Load()
	time.Sleep(3 * warmEvery)
	if calls.Load() != stopped {
		t.Fatal("a stopped client kept calling its machine")
	}
}
