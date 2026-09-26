// Probe real Host/Runtime protocol negotiation and signed Claim with this Creator's bindings.
package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
	"io"
	"os"
	"strings"
	"time"
)

func must(err error) {
	if err != nil {
		panic(err)
	}
}
func main() {
	keyBytes, err := io.ReadAll(os.Stdin)
	must(err)
	block, _ := pem.Decode(keyBytes)
	if block == nil {
		panic("missing PKCS8 identity")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	must(err)
	key, ok := parsed.(ed25519.PrivateKey)
	if !ok {
		panic("expected Ed25519 identity")
	}
	raw, err := os.ReadFile("/run/cozy/bootstrap/readiness-envelope.json")
	must(err)
	var envelope struct{ Payload string }
	must(json.Unmarshal(raw, &envelope))
	payload, err := base64.StdEncoding.DecodeString(envelope.Payload)
	must(err)
	var ready struct {
		PodBootID string `json:"pod_boot_id"`
	}
	must(json.Unmarshal(payload, &ready))
	pin, err := workertls.LoadPin("/run/cozy/bootstrap/tls.crt")
	must(err)
	runtimeAddr, err := os.ReadFile("/run/cozy/worker/control.addr")
	must(err)
	observed := map[string]any{}
	for _, peer := range []struct{ name, address string }{{"host", "127.0.0.1:18441"}, {"runtime", strings.TrimSpace(string(runtimeAddr))}} {
		conn, err := grpc.NewClient(peer.address, grpc.WithTransportCredentials(credentials.NewTLS(pin.TLSConfig())))
		must(err)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		var info *pb.ProtocolInfoResult
		if peer.name == "host" {
			info, err = pb.NewPodHostClient(conn).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
		} else {
			info, err = pb.NewRuntimePreparationClient(conn).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
		}
		must(err)
		if problem := orchestrator.ValidateWorkerProtocol(info, peer.name == "host"); problem != nil {
			panic(problem)
		}
		observed[peer.name] = map[string]any{"current": info.WireMinor, "minimum": info.MinimumWireMinor, "supports_rental_keepalive": info.SupportsRentalKeepalive}
		cancel()
		conn.Close()
	}
	proof, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: 1, WorkerId: os.Args[1], WorkerBootId: ready.PodBootID, WorkerTlsCertificateDigest: pin.Digest()})
	must(err)
	claim := &pb.Claim{RecordOwnerEpoch: 1, RecordOwnerId: "cozy-local-client", WorkerId: os.Args[1], WorkerBootId: ready.PodBootID, WireMinor: pb.WireMinor, Proof: ed25519.Sign(key, proof)}
	conn, err := grpc.NewClient("127.0.0.1:18441", grpc.WithTransportCredentials(credentials.NewTLS(pin.TLSConfig())))
	must(err)
	defer conn.Close()
	accepted := false
	var ackWire uint32
	for attempt := 0; attempt < 2; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		stream, err := pb.NewWorkerControlClient(conn).Control(ctx)
		must(err)
		must(stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: claim}}))
		ack, err := stream.Recv()
		if err != nil {
			cancel()
			if attempt == 0 && status.Code(err) == codes.Unavailable && status.Convert(err).Message() == "workspace authority transferred; reconnect for the recovered snapshot" {
				continue
			}
			must(err)
		}
		ackWire = ack.GetClaimAck().GetWireMinor()
		if !ack.GetClaimAck().GetAccepted() {
			panic(fmt.Sprintf("claim rejected: %v", ack))
		}
		snapshot, err := stream.Recv()
		if err != nil {
			cancel()
			if attempt == 0 && status.Code(err) == codes.Unavailable && status.Convert(err).Message() == "workspace authority transferred; reconnect for the recovered snapshot" {
				continue
			}
			must(err)
		}
		if snapshot.GetSnapshot().GetSnapshotId() == "" {
			panic("missing actual worker snapshot")
		}
		if !bytes.Equal(canonical.Digest(snapshot.GetSnapshot().SnapshotCanonicalBytes), snapshot.GetSnapshot().SnapshotDigest) {
			panic("actual snapshot digest differs")
		}
		actual, err := canonical.Read(snapshot.GetSnapshot().SnapshotCanonicalBytes, &pb.WorkerSnapshotBody{})
		must(err)
		_ = actual
		accepted = true
		cancel()
		break
	}
	if !accepted {
		panic("claim not accepted")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	workspace, err := pb.NewPodHostClient(conn).GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: claim})
	must(err)
	if workspace.ExecutionWorkspaceId == "" || workspace.WorkerId != claim.WorkerId {
		panic("normal unary call did not resolve exact workspace")
	}
	result := map[string]any{"creator_current": pb.WireMinor, "creator_minimum": pb.MinCompatibleWireMinor, "claim_wire": claim.WireMinor, "claim_ack_wire": ackWire, "negotiated_wire": min(claim.WireMinor, ackWire), "protocol": observed, "signed_claim_accepted": true, "normal_workspace_query": true}
	body, err := json.Marshal(result)
	must(err)
	fmt.Println(string(body))
}
