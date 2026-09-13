package producttest

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/executionowner"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/protobuf/proto"
)

// This independently qualifies the fixed image-owned Creator bootstrap. Capture
// uses the ordinary local CLI with no provider capacity; execution is performed
// only by the actual Creator CLI launched by Host. The end-user routing proof is
// separate and must not be inferred from this transport/bootstrap fixture.
func TestExecutionOwnerActualBootstrap(t *testing.T) {
	layout, store, host, path, _ := startActualChildHostMode(t, false)
	identity, problem := rental.PendingCreatorIdentity(layout, "child-host-proof")
	fatal(t, problem)
	t.Cleanup(func() {
		logs, _ := exec.Command("docker", "logs", host.Container).CombinedOutput()
		_ = os.WriteFile(filepath.Join(layout.Root, "host-final.log"), logs, 0600)
		_, _ = exec.Command("docker", "cp", host.Container+":/var/lib/cozy/execution-owner", filepath.Join(layout.Root, "remote-creator")).CombinedOutput()
		compositionDown(t, layout.Root, path)
		label, err := exec.Command("docker", "inspect", "--format", `{{index .Config.Labels "com.cozy.task"}}`, host.Container).Output()
		if err != nil || strings.TrimSpace(string(label)) != "proto051-disconnected-owner" {
			t.Error("refused cleanup of unowned bootstrap container")
			return
		}
		for _, args := range [][]string{{"stop", host.Container}, {"rm", host.Container}} {
			if output, err := exec.Command("docker", args...).CombinedOutput(); err != nil {
				t.Errorf("owned container cleanup: %v %s", err, output)
			}
		}
	})
	script, err := filepath.Abs("testdata/disconnected_execution/first.py")
	must(t, err)
	_, output := runCozyPath(t, layout.Root, path, "run", script, "--rental-only", "--json", "--idempotency-key=bootstrap-capture")
	must(t, os.WriteFile(filepath.Join(layout.Root, "capture-cli.json"), []byte(output), 0600))
	request, problem := store.RequestByIdempotencyKey("bootstrap-capture")
	fatal(t, problem)
	if request == nil || request.Ordinal != 0 || request.LocalPackageDigest == "" {
		t.Fatalf("CLI did not retain an unexecuted capture: %+v %s", request, output)
	}
	// This is a separate initial execution, not migration of a laptop attempt.
	remote := *request
	remote.ID, remote.IdemKey = records.NewID("job"), "remote-bootstrap-root"
	capture, problem := executionowner.Export(layout, store, remote)
	fatal(t, problem)
	must(t, os.WriteFile(filepath.Join(layout.Root, "capsule.json"), capture.Raw, 0600))
	readiness, err := os.ReadFile(filepath.Join(layout.Root, "host-readiness.json"))
	must(t, err)
	var ready struct {
		Boot string `json:"pod_boot_id"`
	}
	must(t, json.Unmarshal(readiness, &ready))
	pin, err := workertls.LoadPin(filepath.Join(layout.Root, "host-certificate.pem"))
	must(t, err)
	proof, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: 1, WorkerId: "private-child-host", WorkerBootId: ready.Boot, WorkerTlsCertificateDigest: pin.Digest()})
	must(t, err)
	claim := &pb.Claim{RecordOwnerEpoch: 1, RecordOwnerId: "cozy-local-client", WorkerId: "private-child-host", WorkerBootId: ready.Boot, WireMinor: pb.WireMinor, Proof: identity.Sign(proof)}
	connection, err := grpc.NewClient(host.Control, grpc.WithTransportCredentials(credentials.NewTLS(pin.TLSConfig())))
	must(t, err)
	defer connection.Close()
	peer := pb.NewPodHostClient(connection)
	for digest, revision := range capture.Packages {
		source, err := canonical.Raw(revision.SourceDigest)
		must(t, err)
		for _, file := range revision.Files {
			id, err := canonical.Raw(file.Digest)
			must(t, err)
			stream, err := peer.LocalPackageUpload(context.Background())
			must(t, err)
			must(t, stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Header{Header: &pb.LocalPackageUploadHeader{Claim: claim, OperationId: executionowner.UploadOperation(digest), SourceDigest: source, File: &pb.LocalPackageFileRef{Digest: id, Filename: file.Filename, Length: uint64(file.Length)}}}}))
			status, err := stream.Recv()
			must(t, err)
			input, err := os.Open(file.Path)
			must(t, err)
			_, err = input.Seek(int64(status.ReceivedBytes), io.SeekStart)
			must(t, err)
			buffer := make([]byte, 1<<20)
			for status.ReceivedBytes < uint64(file.Length) {
				n, err := input.Read(buffer)
				if err != nil && err != io.EOF {
					t.Fatal(err)
				}
				if n == 0 {
					t.Fatal("captured upload ended before its exact length")
				}
				must(t, stream.Send(&pb.LocalPackageUploadFrame{Body: &pb.LocalPackageUploadFrame_Chunk{Chunk: &pb.LocalPackageUploadChunk{Offset: status.ReceivedBytes, Data: buffer[:n]}}}))
				status, err = stream.Recv()
				must(t, err)
			}
			input.Close()
			must(t, stream.CloseSend())
		}
	}
	// End only this unexecuted capture's local bookkeeping. No laptop worker
	// claim ever crossed to the actual private Host, and no local daemon remains.
	compositionDown(t, layout.Root, path)
	prepared, err := peer.PrepareExecutionOwner(context.Background(), &pb.PrepareExecutionOwnerCall{Claim: claim})
	must(t, err)
	grant := &pb.ExecutionOwnerGrant{RecordOwnerEpoch: 1, RecordOwnerId: "private-bootstrap-owner", WorkerId: prepared.WorkerId,
		WorkerBootId: prepared.WorkerBootId, WorkerTlsCertificateDigest: pin.Digest(), ExecutionPublicKey: prepared.ExecutionPublicKey, InitialCapsuleDigest: canonical.Digest(capture.Raw)}
	grantBytes, err := canonical.Bytes(grant)
	must(t, err)
	authorization := &pb.SignedExecutionOwnerGrant{Grant: grant, Signature: identity.Sign(grantBytes)}
	accepted, err := peer.AcceptExecutionOwner(context.Background(), &pb.AcceptExecutionOwnerCall{Authorization: authorization, InitialCapsule: capture.Raw})
	if err != nil {
		signed, encodeErr := proto.Marshal(authorization)
		must(t, encodeErr)
		authority, encodeErr := json.Marshal(executionowner.Authority{Format: executionowner.AuthorityFormat, SignedGrant: signed,
			CreatorHome: "/var/lib/cozy/execution-owner", PackageStageRoot: "/var/lib/cozy/installs/.stage",
			WorkerAddress: "127.0.0.1:19781", WorkerTLSCertificatePath: "/run/cozy/bootstrap/tls.crt", TensorFSRoot: "/var/lib/tensorfs", MediaAddress: "https://127.0.0.1:19782"})
		must(t, encodeErr)
		authorityPath := filepath.Join(layout.Root, "validate-authority.json")
		must(t, os.WriteFile(authorityPath, authority, 0600))
		for _, name := range []string{"capsule.json", "validate-authority.json"} {
			if data, copyErr := exec.Command("docker", "cp", filepath.Join(layout.Root, name), host.Container+":/tmp/"+name).CombinedOutput(); copyErr != nil {
				t.Logf("validation diagnostic copy: %v %s", copyErr, data)
			}
		}
		program := `import os,subprocess
a=os.open('/tmp/capsule.json',os.O_RDONLY); b=os.open('/tmp/validate-authority.json',os.O_RDONLY)
p=subprocess.run(['/opt/cozy/bin/cozy','execution-owner','validate','--capsule-fd',str(a),'--authority-fd',str(b),'--json'],pass_fds=(a,b))
raise SystemExit(p.returncode)`
		data, diagnosticErr := exec.Command("docker", "exec", host.Container, "python3", "-c", program).CombinedOutput()
		_ = os.WriteFile(filepath.Join(layout.Root, "validation-diagnostic.log"), data, 0600)
		t.Logf("exact fixed Creator validation diagnostic: %v %s", diagnosticErr, data)
	}
	must(t, err)
	acceptance, err := json.Marshal(accepted)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(layout.Root, "host-acceptance.json"), acceptance, 0600))
	waitFile := func(name, want string) {
		t.Helper()
		for deadline := time.Now().Add(3 * time.Minute); ; {
			value, err := exec.Command("docker", "exec", host.Container, "cat", "/tmp/cozy-offline-proto051/"+name).Output()
			if err == nil && string(value) == want {
				return
			}
			if time.Now().After(deadline) {
				logs, _ := exec.Command("docker", "logs", "--tail", "80", host.Container).CombinedOutput()
				t.Fatalf("actual bootstrap did not reach %s: %s", name, logs)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}
	waitFile("waiting", "49")
	_, err = exec.Command("docker", "exec", host.Container, "touch", "/tmp/cozy-offline-proto051/continue").CombinedOutput()
	must(t, err)
	waitFile("continued", "50")
	waitFile("finished", "51")
	command := exec.Command("docker", "exec", "--env", "COZY_HOME=/var/lib/cozy/execution-owner", host.Container, "/opt/cozy/bin/cozy", "run", "watch", remote.ID, "--json")
	result, err := command.CombinedOutput()
	must(t, os.WriteFile(filepath.Join(layout.Root, "remote-result.json"), result, 0600))
	if err != nil {
		t.Fatalf("actual remote Creator CLI watch: %v %s", err, result)
	}
}
