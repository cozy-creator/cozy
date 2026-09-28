package producttest

import (
	"archive/tar"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
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
	pod := newRentedIngestPod(t, "https://civitai.com/api/v1/model-versions/128078")
	pod.machine.memoLookup = true // the owner answers this Runtime's memo lookups
	code, out := pod.upload("civitai://128078", "proof/sdxl", "--source-profile", "civitai/sdxl/single-file/1")
	if code != 0 || strings.Contains(out, "slot=profile") {
		t.Fatalf("the rented ingest was refused [exit %d]:\n%s", code, out)
	}
	if script := pod.submittedScript(t); script != civitaiIngestScript {
		t.Fatalf("the pod received another ingest script:\n%s\nwant:\n%s", script, civitaiIngestScript)
	}
	if !pod.machine.submitted().OwnerMemo {
		t.Fatal("the submission to a memo_lookup Runtime does not say its owner answers memo lookups")
	}
}

// A moving Hugging Face ref is pinned to its current commit when the ingest is submitted:
// the owner sees the pin, and the script (so its resume and memo key) names the commit.
func TestRentedIngestPinsAMovingHuggingFaceRef(t *testing.T) {
	const repo = "alibaba-pai/MiniMax-H3-Acc-LoRAs"
	pod := newRentedIngestPod(t, "https://huggingface.co/api/models/"+repo)
	var current struct {
		SHA string `json:"sha"`
	}
	response, err := http.Get("https://huggingface.co/api/models/" + repo + "/revision/main")
	must(t, err)
	must(t, json.NewDecoder(response.Body).Decode(&current))
	response.Body.Close()
	code, out := pod.upload("hf://"+repo, "proof/pdd", "--source-profile", "hf/minimax-h3/pdd-fl2va-bf16/1")
	if code != 0 || !strings.Contains(out, "pinned hf://"+repo+"@"+current.SHA+"\n") {
		t.Fatalf("the moving ref was not pinned to %s [exit %d]:\n%s", current.SHA, code, out)
	}
	if script := pod.submittedScript(t); !strings.Contains(script, `upload_huggingface("`+repo+`", revision="`+current.SHA+`"`) {
		t.Fatalf("the submitted script does not name the pinned commit %s:\n%s", current.SHA, script)
	}
}

// Several uploads to one pod are submitted together: a capture shares the install root with
// other captures instead of refusing them as a competing writer.
func TestConcurrentRentedIngestsAreAllSubmitted(t *testing.T) {
	pod := newRentedIngestPod(t, "https://civitai.com/api/v1/model-versions/128078")
	// The owner's local TensorFS store already exists, as it does after any first command.
	if code, out := runCozy(t, pod.root, "model", "list", "--json"); code != 0 {
		t.Fatalf("model list [exit %d]: %s", code, out)
	}
	outputs := make([]string, 2)
	codes := make([]int, 2)
	var group sync.WaitGroup
	for i := range outputs {
		group.Add(1)
		go func() {
			defer group.Done()
			codes[i], outputs[i] = pod.upload("civitai://128078", fmt.Sprintf("proof/sdxl-%d", i),
				"--source-profile", "civitai/sdxl/single-file/1")
		}()
	}
	group.Wait()
	for i := range outputs {
		if codes[i] != 0 {
			t.Fatalf("concurrent upload %d was refused [exit %d]:\n%s", i, codes[i], outputs[i])
		}
	}
	runs, problem := pod.store.Requests("", "", 10)
	fatal(t, problem)
	if len(runs) != 2 || runs[0].ID == runs[1].ID {
		t.Fatalf("want both uploads recorded as runs, got %d", len(runs))
	}
}

// A gated source without a configured provider token is refused before renting with the
// credential it needs, never as an origin that cannot serve header ranges.
func TestGatedProviderSourceNamesTheMissingToken(t *testing.T) {
	root := filepath.Join(scratchBase, "gated-source")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = removeAllForce(root) })
	standIn := httptest.NewServer(http.NotFoundHandler())
	defer standIn.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+standIn.URL+"\n"), 0o600))
	for _, row := range []struct{ probe, source, token string }{
		{"https://civitai.com/api/download/models/889818", "civitai://889818", "civitai_token"},
		{"https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
			"hf://black-forest-labs/FLUX.1-dev@3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/flux1-dev.safetensors", "huggingface_token"},
	} {
		response, err := http.Get(row.probe)
		if err != nil {
			t.Skipf("provider unreachable from this runner: %v", err)
		}
		response.Body.Close()
		if response.StatusCode != http.StatusUnauthorized {
			t.Fatalf("%s is no longer gated: HTTP %d", row.probe, response.StatusCode)
		}
		cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "model", "upload", row.source, "proof/gated", "--rental-only", "--json")
		cmd.Env = childEnv(t, root, "TENSORHUB_TOKEN=operator-proof", "HF_TOKEN=", "CIVITAI_TOKEN=")
		out, _ := cmd.Output()
		var document struct {
			Error struct{ Code, Remedy string } `json:"error"`
		}
		if json.Unmarshal(out, &document) != nil || document.Error.Code != "model_source.auth_required" ||
			!strings.Contains(document.Error.Remedy, row.token) {
			t.Fatalf("%s must name the missing %s [exit %d]: %s", row.source, row.token, cmd.ProcessState.ExitCode(), out)
		}
	}
}

// Re-running a completed ingest submits it again to the pod: its Runtime, whose version
// and the source's configs are in its own memo key, decides whether the result is still
// the same; the unchanged script is no evidence of that (Runtime 0.18.57 added Diffusers
// configs to an Anima upload the old checkpoint lacked). A re-run of a live ingest is still
// the same run, and nothing is submitted twice.
func TestARerunOfACompletedIngestReachesThePod(t *testing.T) {
	pod := newRentedIngestPod(t, "https://civitai.com/api/v1/model-versions/128078")
	args := []string{"civitai://128078", "proof/sdxl", "--source-profile", "civitai/sdxl/single-file/1"}
	// The pod counts runs by request; a live run's resubmission is the same run.
	submissions := func() int {
		pod.machine.mu.Lock()
		defer pod.machine.mu.Unlock()
		requests := map[string]bool{}
		for _, submit := range pod.machine.submissions {
			requests[submit.Offer.RequestId] = true
		}
		return len(requests)
	}
	for range 2 {
		if code, out := pod.upload(args...); code != 0 {
			t.Fatalf("the ingest was refused [exit %d]:\n%s", code, out)
		}
	}
	waitFor(t, pod.root, "the ingest's machine submission", func() bool { return submissions() > 0 })
	runs, problem := pod.store.Requests("", "", 10)
	fatal(t, problem)
	if len(runs) != 1 || submissions() != 1 {
		t.Fatalf("a re-run of a live ingest was not the same run: %d runs, %d submitted", len(runs), submissions())
	}
	// The ingest completed under the pod's previous Runtime.
	fatal(t, pod.store.SettleRequest(runs[0].ID, "succeeded"))
	if code, out := pod.upload(args...); code != 0 {
		t.Fatalf("the re-run was refused [exit %d]:\n%s", code, out)
	}
	waitFor(t, pod.root, "the re-run's machine submission", func() bool { return submissions() == 2 })
	if runs, problem = pod.store.Requests("", "", 10); problem != nil || len(runs) != 2 {
		t.Fatalf("the re-run replayed the completed ingest instead of a new run: %d runs %v", len(runs), problem)
	}
}

// rentedIngestPod is a daemon with one attached fake pod, "ingester", that keeps the bytes of
// every uploaded file and answers preparation with the interface the owner captured.
type rentedIngestPod struct {
	root     string
	env      []string
	store    *records.Store
	machine  *runtimeMachine
	mu       sync.Mutex
	received map[string][]byte
	prepared []*pb.PrepareLocalPackageCall
}

// newRentedIngestPod attaches a fake rented pod and starts the daemon with extraConfig appended to
// its home config. An empty providerProbe skips reaching a real provider.
func newRentedIngestPod(t *testing.T, providerProbe string, extraConfig ...string) *rentedIngestPod {
	t.Helper()
	base, err := os.MkdirTemp(scratchBase, "ingest-script-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(base) })
	root, tmp := filepath.Join(base, "home"), filepath.Join(base, "tmp")
	must(t, os.MkdirAll(tmp, 0o700))
	must(t, os.MkdirAll(root, 0o700))
	if providerProbe != "" {
		response, err := http.Get(providerProbe)
		if err != nil {
			t.Skipf("provider unreachable from this runner: %v", err)
		}
		response.Body.Close()
	}

	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	identity, problem := rental.PendingCreatorIdentity(layout, "ingest-script")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	fixture := &rentedIngestPod{root: root, store: store, machine: &runtimeMachine{blocker: "gpu-holder"},
		received: map[string][]byte{}}
	pod := &fakePod{controlKey: public, machine: fixture.machine}
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
		fixture.mu.Lock()
		fixture.received[file.Filename] = body
		fixture.mu.Unlock()
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
		fixture.mu.Lock()
		fixture.prepared = append(fixture.prepared, call)
		fixture.mu.Unlock()
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
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"+strings.Join(extraConfig, "")), 0o600))
	startDaemonProcess(t, root, "TMPDIR="+tmp)
	fixture.env = childEnv(t, root, "TMPDIR="+tmp)
	return fixture
}

// upload runs one rented `cozy model upload` against the pod and answers its exit and output.
func (p *rentedIngestPod) upload(args ...string) (int, string) {
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin, "model", "upload"},
		append(args, "--rental=ingester", "--json")...)...)
	cmd.Env = p.env
	out, _ := cmd.CombinedOutput()
	return cmd.ProcessState.ExitCode(), string(out)
}

// submittedScript is the one ingest script the pod was asked to run, from the installation
// it prepared and the machine submission that names it.
func (p *rentedIngestPod) submittedScript(t *testing.T) string {
	t.Helper()
	waitFor(t, p.root, "the ingest's machine submission", func() bool { return p.machine.submitted() != nil })
	var capture pb.MachineExecutionCapture
	must(t, canonical.Unmarshal(p.machine.submitted().CaptureCanonicalBytes, &capture))
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.prepared) != 1 || capture.RootInstallationId != p.prepared[0].LocalPackageSet.Package.InstallationId {
		t.Fatalf("the submission does not run the uploaded installation: prepared %d, root %q", len(p.prepared), capture.RootInstallationId)
	}
	script, err := tarMember(p.received["source.tar"], "cozy_script.py")
	must(t, err)
	return string(script)
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
