package producttest

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const gatedToken = "fake-civitai-token-3d9e21"

// A rented run of a gated Civitai source carries the owner's configured token to the pod on
// the wire and nowhere else. The pod's native fetch presents it to the provider, which
// refuses without it. When the pod reports the value lost (a Runtime restart), the daemon
// supplies it again. A provider refusal reads as model_source.auth_required and names
// `cozy rental update`.
func TestRentedGatedSourceCarriesTheOwnerCredential(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+gatedToken {
			w.WriteHeader(http.StatusUnauthorized)
		}
	}))
	defer provider.Close()
	pod := newRentedIngestPod(t, "", "civitai_token: "+gatedToken+"\n")
	pod.machine.mu.Lock()
	pod.machine.sourceCredentials = true
	pod.machine.mu.Unlock()

	script := filepath.Join(filepath.Dir(pod.root), "tmp", "gated.py")
	must(t, os.WriteFile(script, []byte(`# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.18.51,<1", "tensorfs>=0.3.60,<0.4"]
# ///
from cozy_runtime.author.sources import download_civitai

async def main() -> dict[str, str]:
    source = await download_civitai(777)
    return {"manifest": source.manifest.digest}
`), 0o600))
	if code, out := pod.cozy("run", script, "--rental=ingester", "--json"); code != 0 {
		t.Fatalf("the rented run was refused [exit %d]:\n%s", code, out)
	}
	submission := pod.machine.submitted()
	waitFor(t, pod.root, "the machine submission", func() bool { submission = pod.machine.submitted(); return submission != nil })
	want := []*pb.SourceCredential{{Provider: pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_CIVITAI, Credential: "bearer " + gatedToken}}
	if !credentialsEqual(submission.SourceCredentials, want) {
		t.Fatalf("the submission carried %v, want the configured Civitai credential", submission.SourceCredentials)
	}
	// The pod's native fetch presents the carrier as TensorFS does; the provider needs it.
	for carrier, status := range map[string]int{submission.SourceCredentials[0].Credential: 200, "": 401} {
		request, err := http.NewRequest(http.MethodGet, provider.URL, nil)
		must(t, err)
		if carrier != "" {
			request.Header.Set("Authorization", "Bearer "+strings.TrimPrefix(carrier, "bearer "))
		}
		response, err := http.DefaultClient.Do(request)
		must(t, err)
		response.Body.Close()
		if response.StatusCode != status {
			t.Fatalf("the provider answered %d to carrier %q, want %d", response.StatusCode, carrier, status)
		}
	}

	// A restart lost the value on the pod: the daemon's observation supplies it again with the
	// identical submission.
	request := submission.Offer.RequestId
	pod.machine.mu.Lock()
	pod.machine.state.AwaitingSourceCredentials = []pb.NativeSourceOperation{pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_CIVITAI}
	pod.machine.mu.Unlock()
	var again *pb.MachineExecutionSubmit
	waitFor(t, pod.root, "the credential supplied again", func() bool {
		pod.cozy("run", "show", request, "--json")
		pod.machine.mu.Lock()
		defer pod.machine.mu.Unlock()
		if len(pod.machine.submissions) > 1 {
			again = pod.machine.submissions[len(pod.machine.submissions)-1]
			pod.machine.state.AwaitingSourceCredentials = nil
		}
		return again != nil
	})
	resent := proto.Clone(again).(*pb.MachineExecutionSubmit)
	resent.SourceCredentials, resent.Claim, resent.Offer.WorkerBootId, resent.Offer.RecordOwnerEpoch = nil, nil, "", 0
	first := proto.Clone(submission).(*pb.MachineExecutionSubmit)
	first.SourceCredentials, first.Claim, first.Offer.WorkerBootId, first.Offer.RecordOwnerEpoch = nil, nil, "", 0
	if !credentialsEqual(again.SourceCredentials, want) || !proto.Equal(resent, first) {
		t.Fatalf("the credential was not supplied with the identical submission: %v", again.SourceCredentials)
	}

	// The provider refuses the fetch: typed, naming the token.
	pod.machine.mu.Lock()
	pod.machine.failure = "TRANSFER_FAILED: source worker ended without a bounded result; stderr: tensorfs.errors.TransferFailed: TRANSFER_FAILED: origin answered HTTP 401 to a length probe"
	pod.machine.mu.Unlock()
	pod.machine.finish()
	var shown string
	waitFor(t, pod.root, "the typed provider refusal", func() bool {
		_, shown = pod.cozy("run", "watch", request, "--json")
		return strings.Contains(shown, "model_source.auth_required")
	})
	if !strings.Contains(shown, "civitai_token") {
		t.Fatalf("the refusal does not name the token:\n%s", shown)
	}

	// The token is on the wire only: not in the frozen submission, the store, any log or file
	// under the home other than the owner's own config, or what the CLI shows.
	link, problem := pod.store.MachineExecution(request)
	fatal(t, problem)
	if bytes.Contains(link.Submission, []byte(gatedToken)) {
		t.Fatal("the frozen submission holds the credential")
	}
	must(t, filepath.Walk(filepath.Dir(pod.root), func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || info.Name() == "config.yaml" || info.Mode()&os.ModeType != 0 {
			return err
		}
		if body, readErr := os.ReadFile(path); readErr == nil && bytes.Contains(body, []byte(gatedToken)) {
			t.Errorf("the credential was written to %s", path)
		}
		return nil
	}))
	if strings.Contains(shown, gatedToken) {
		t.Fatal("the CLI showed the credential")
	}
}

func credentialsEqual(got, want []*pb.SourceCredential) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if !proto.Equal(got[i], want[i]) {
			return false
		}
	}
	return true
}

// cozy runs one CLI command against the pod's home and answers its exit and output.
func (p *rentedIngestPod) cozy(args ...string) (int, string) {
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = p.env
	out, _ := cmd.CombinedOutput()
	return cmd.ProcessState.ExitCode(), string(out)
}
