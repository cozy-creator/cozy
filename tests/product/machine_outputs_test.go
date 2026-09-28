package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// machineServesOutput reads an output of this computer's machine's newest run over the
// machine endpoint, as any holder of a capability does: the current bytes with their
// revision as ETag, byte ranges, revalidation, and the digest once final. A machine whose
// Runtime predates run numbers answers 501 for the one route.
func machineServesOutput(t *testing.T, root, output, digest string, revisions int) {
	t.Helper()
	layout, problem := home.Open(root)
	fatal(t, problem)
	host := machines.NewHost(layout.Machine, "", nil)
	state, problem := host.Status()
	fatal(t, problem)
	pin, problem := host.Pin()
	fatal(t, problem)
	owner, problem := host.Owner()
	fatal(t, problem)
	var record struct {
		WorkerPort int `json:"worker_port"`
	}
	raw, err := os.ReadFile(filepath.Join(layout.Machine, "host.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &record))
	found := &machines.Resolver{Host: host, HubOrigin: state.Hub,
		UseRental: func(string, string) (func(), *exit.Error) { return func() {}, nil },
		RentalKey: func(string) (rental.CreatorIdentity, *exit.Error) { return rental.CreatorIdentity{}, nil }}
	machine, problem := found.Dial(context.Background(), machines.Local, "run outputs")
	fatal(t, problem)
	list, err := machine.Host.ListMachineExecutions(context.Background(), &pb.MachineExecutionListQuery{
		Claim: machine.Claim, NewestFirst: true, Limit: 1})
	machine.Close()
	run, numbered := "1", status.Code(err) != codes.Unimplemented
	if numbered {
		must(t, err)
		if len(list.Executions) == 0 || list.Executions[0].Number == 0 {
			t.Fatalf("the machine lists no numbered run: %+v", list)
		}
		run = strconv.FormatUint(list.Executions[0].Number, 10)
	}
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	must(t, err)
	token, err := capability.MintSigned(ed25519.PublicKey(public), owner.Sign,
		capability.Grant{Machine: state.MachineID, Run: run, Expires: time.Now().Add(10 * time.Minute).Unix()})
	must(t, err)
	client := &http.Client{Transport: &http.Transport{TLSClientConfig: pin.TLSConfig()}}
	get := func(headers ...string) (*http.Response, []byte) {
		t.Helper()
		request, err := http.NewRequest(http.MethodGet, fmt.Sprintf("https://127.0.0.1:%d/v1/runs/%s/outputs/%s", record.WorkerPort, run, output), nil)
		must(t, err)
		request.Header.Set("Authorization", "Cozy-Cap "+token)
		for i := 0; i+1 < len(headers); i += 2 {
			request.Header.Set(headers[i], headers[i+1])
		}
		response, err := client.Do(request)
		must(t, err)
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		must(t, err)
		return response, body
	}
	whole, body := get()
	if !numbered {
		if whole.StatusCode != http.StatusNotImplemented {
			t.Fatalf("a machine without run numbers answered %d for its outputs: %s", whole.StatusCode, body)
		}
		return
	}
	sum := sha256.Sum256(body)
	etag := fmt.Sprintf(`"r%d"`, revisions)
	if whole.StatusCode != http.StatusOK || "sha256:"+hex.EncodeToString(sum[:]) != digest || whole.Header.Get("ETag") != etag ||
		whole.Header.Get("Repr-Digest") != "sha-256=:"+base64.StdEncoding.EncodeToString(sum[:])+":" {
		t.Fatalf("the output answered %d %v with %d bytes: %.300s", whole.StatusCode, whole.Header, len(body), body)
	}
	part, piece := get("Range", "bytes=0-9")
	if part.StatusCode != http.StatusPartialContent || string(piece) != string(body[:10]) ||
		part.Header.Get("Content-Range") != fmt.Sprintf("bytes 0-9/%d", len(body)) {
		t.Fatalf("a range answered %d %v", part.StatusCode, part.Header)
	}
	if same, _ := get("If-None-Match", etag); same.StatusCode != http.StatusNotModified {
		t.Fatalf("revalidating the current revision answered %d", same.StatusCode)
	}
	if stale, again := get("Range", "bytes=0-9", "If-Range", `"r1"`); stale.StatusCode != http.StatusOK || len(again) != len(body) {
		t.Fatalf("a range of an older revision answered %d with %d bytes", stale.StatusCode, len(again))
	}
	request, err := http.NewRequest(http.MethodGet, fmt.Sprintf("https://127.0.0.1:%d/v1/runs/%s/outputs/%s", record.WorkerPort, run, output), nil)
	must(t, err)
	if anonymous, err := client.Do(request); err != nil || anonymous.StatusCode != http.StatusForbidden {
		t.Fatalf("an output without a capability answered %v %v", anonymous, err)
	}
}
