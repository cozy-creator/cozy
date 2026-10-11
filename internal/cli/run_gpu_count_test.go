package cli

import (
	"context"
	"net"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
)

type gpuCountPeer struct {
	v1.UnimplementedMachineServer
	supported bool
	statuses  int
}

func (p *gpuCountPeer) Status(_ *v1.StatusRequest, stream grpc.ServerStreamingServer[v1.StatusFrame]) error {
	p.statuses++
	var capabilities []string
	if p.supported {
		capabilities = []string{"run-gpus/1"}
	}
	return stream.Send(&v1.StatusFrame{Capabilities: capabilities})
}

func TestRunGPUCountReachesNativeSpecOrRefusesBeforePreparation(t *testing.T) {
	for _, count := range []uint32{0, 1, 2} {
		for _, supported := range []bool{false, true} {
			peer := &gpuCountPeer{supported: supported}
			listener := bufconn.Listen(1 << 20)
			server := grpc.NewServer()
			v1.RegisterMachineServer(server, peer)
			go server.Serve(listener)
			conn, err := grpc.NewClient("passthrough:///gpu-count", grpc.WithTransportCredentials(insecure.NewCredentials()),
				grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
			if err != nil {
				t.Fatal(err)
			}
			machine := &machines.V1{Name: "fixture", Client: &machinev1.Client{Machine: v1.NewMachineClient(conn)}}
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			if problem != nil {
				t.Fatal(problem)
			}
			m := &machineRuns{store: store, context: &Context{}, resolver: &Resolver{}}
			for _, kind := range []string{"serving", "job"} {
				spec, problem := m.specV1(context.Background(), records.Request{Package: "proof/package", Entrypoint: "call", Kind: kind, GPUs: count, Payload: []byte(`{}`)}, machine)
				if count > 0 && !supported {
					if problem == nil || problem.ErrName() != "machine.gpu_count_unsupported" || spec != nil {
						t.Fatalf("count silently ignored: count=%d spec=%+v problem=%v", count, spec, problem)
					}
					continue
				}
				if problem != nil {
					t.Fatal(problem)
				}
				wire, err := proto.Marshal(spec)
				if err != nil {
					t.Fatal(err)
				}
				decoded := &v1.RunSpec{}
				if err := proto.Unmarshal(wire, decoded); err != nil {
					t.Fatal(err)
				}
				if decoded.Gpus != count {
					t.Fatalf("lost count in protobuf: %+v", decoded)
				}
			}
			if count == 0 && peer.statuses != 0 {
				t.Fatal("automatic run unnecessarily requires new peer capability")
			}
			store.Close()
			conn.Close()
			server.Stop()
		}
	}
}
