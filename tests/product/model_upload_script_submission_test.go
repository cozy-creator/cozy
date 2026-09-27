package producttest

import (
	"archive/tar"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// civitaiIngestScript is the one bounded-disk upload call a rented Civitai ingest submits.
const civitaiIngestScript = `# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.18.51,<1", "tensorfs>=0.3.60,<0.4"]
# ///
from cozy_runtime.author.sources import upload_civitai, upload_huggingface

async def main() -> dict[str, str]:
    checkpoint = await upload_civitai(128078, profiles=("civitai/sdxl/single-file/1",), destination="proof/sdxl")
    return {"destination": checkpoint.destination, "checkpoint": checkpoint.checkpoint}
`

// `cozy model upload <civitai> <org/model> --source-profile P --rental=NAME` reaches the pod
// as exactly the single-call script, through the real capture, upload and submission path.
// The consumed --source-profile never reaches the run, where it would be reread as a
// job's slot=profile binding (#762).
func TestRentedCivitaiIngestSubmitsTheSingleCallScript(t *testing.T) {
	base, err := os.MkdirTemp(scratchBase, "ingest-script-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(base) })
	root, tmp := filepath.Join(base, "home"), filepath.Join(base, "tmp")
	must(t, os.MkdirAll(tmp, 0o700))
	must(t, os.MkdirAll(root, 0o700))
	response, err := http.Get("https://civitai.com/api/v1/model-versions/128078")
	if err != nil {
		t.Skipf("provider unreachable from this runner: %v", err)
	}
	response.Body.Close()

	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "ingest-script")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	var mu sync.Mutex
	received := map[string][]byte{}
	var prepared []*pb.PrepareLocalPackageCall
	machine := &runtimeMachine{blocker: "gpu-holder"}
	pod := &fakePod{controlKey: public, machine: machine}
	// The pod keeps each uploaded file's bytes: this is where the script crosses.
	pod.localUpload = func(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
		frame, err := stream.Recv()
		if err != nil {
			return err
		}
		header := frame.GetHeader()
		if header == nil || header.File == nil {
			return status.Error(codes.InvalidArgument, "header required")
		}
		if err := pod.verifyClaim(header.Claim, false); err != nil {
			return err
		}
		file := header.File
		state := &pb.LocalPackageFileStatus{OperationId: header.OperationId, Digest: file.Digest, Filename: file.Filename,
			Length: file.Length, State: pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING}
		if err := stream.Send(state); err != nil {
			return err
		}
		var body []byte
		for uint64(len(body)) < file.Length {
			frame, err := stream.Recv()
			if err != nil {
				return err
			}
			chunk := frame.GetChunk()
			if chunk == nil || chunk.Offset != uint64(len(body)) || uint64(len(body)+len(chunk.Data)) > file.Length {
				return status.Error(codes.InvalidArgument, "chunk offset or size")
			}
			body = append(body, chunk.Data...)
			state.ReceivedBytes = uint64(len(body))
			if state.ReceivedBytes == file.Length {
				state.State = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED
			}
			if err := stream.Send(state); err != nil {
				return err
			}
		}
		mu.Lock()
		received[file.Filename] = body
		mu.Unlock()
		return nil
	}
	// Runtime answers preparation with the installed interface; the owner captured the same one.
	pod.localPrepare = func(call *pb.PrepareLocalPackageCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
		if err := pod.verifyClaim(call.Claim, false); err != nil {
			return err
		}
		selected := call.GetLocalPackageSet().GetPackage()
		install, problem := store.Install(selected.GetInstallationId())
		if problem != nil || install == nil {
			return status.Error(codes.NotFound, "no captured installation")
		}
		surface, err := os.ReadFile(launch.PackageInterfacePath(install.Dir))
		if err != nil {
			return status.Error(codes.FailedPrecondition, err.Error())
		}
		mu.Lock()
		prepared = append(prepared, call)
		mu.Unlock()
		return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED,
			InstalledPackage: &pb.InstalledPackage{InstallationId: selected.InstallationId, Package: selected.Package,
				Release: selected.Release, PackageInterface: surface}})
	}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)

	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "ingester")
	for key, value := range map[string]any{"requested_accelerator_model": "fake-4090", "worker_id": podWorkerID,
		"worker_boot_id": podBootID, "cert_pem": string(cert), "worker_address": connection.Addr,
		"media_address": connection.Media.Addr} {
		hub.set(podRental, key, value)
	}
	hub.mu.Lock()
	hub.inventories = map[string]json.RawMessage{podRental: json.RawMessage(`{"format":"tensorhub.image_inventory/1",` +
		`"profile":"python3.12-cpu-linux-x86","python":"3.12.12","distributions":[{"name":"` + runtimeDistribution + `","version":"0.18.51"}]}`)}
	hub.mu.Unlock()
	served := hub.server.Config.Handler
	hub.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/v1/machine-authorizations" {
			_ = json.NewEncoder(w).Encode(map[string]any{"token": "ingest-grant", "expires_at": time.Now().Add(time.Hour)})
			return
		}
		served.ServeHTTP(w, r)
	})
	row := records.Rental{ID: podRental, MachineName: "ingester", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: connection.Addr,
		MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	fatal(t, rental.Attach(layout, store, row, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	startDaemonProcess(t, root, "TMPDIR="+tmp)

	cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "model", "upload", "civitai://128078", "proof/sdxl",
		"--source-profile", "civitai/sdxl/single-file/1", "--rental=ingester", "--json")
	cmd.Env = childEnv(t, root, "TMPDIR="+tmp)
	out, _ := cmd.CombinedOutput()
	if code := cmd.ProcessState.ExitCode(); code != 0 || strings.Contains(string(out), "slot=profile") {
		t.Fatalf("the rented ingest was refused [exit %d]:\n%s", code, out)
	}
	waitFor(t, root, "the ingest's machine submission", func() bool { return machine.submitted() != nil })

	var capture pb.MachineExecutionCapture
	must(t, canonical.Unmarshal(machine.submitted().CaptureCanonicalBytes, &capture))
	mu.Lock()
	defer mu.Unlock()
	if len(prepared) != 1 || capture.RootInstallationId != prepared[0].LocalPackageSet.Package.InstallationId {
		t.Fatalf("the submission does not run the uploaded installation: prepared %d, root %q", len(prepared), capture.RootInstallationId)
	}
	script, err := tarMember(received["source.tar"], "cozy_script.py")
	must(t, err)
	if string(script) != civitaiIngestScript {
		t.Fatalf("the pod received another ingest script:\n%s\nwant:\n%s", script, civitaiIngestScript)
	}
}

func tarMember(archive []byte, name string) ([]byte, error) {
	reader := tar.NewReader(bytes.NewReader(archive))
	for {
		header, err := reader.Next()
		if err != nil {
			if err == io.EOF {
				return nil, errors.New(name + " is not in the uploaded source")
			}
			return nil, err
		}
		if header.Name == name {
			return io.ReadAll(reader)
		}
	}
}
