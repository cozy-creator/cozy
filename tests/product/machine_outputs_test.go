package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
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

// outputReader reads one output of this computer's machine over its endpoint, as any holder
// of a capability for the run does.
type outputReader struct {
	get       func(headers ...string) (*http.Response, []byte)
	anonymous func() int
}

func machineOutputs(t *testing.T, root, run, output string) (outputReader, string) {
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
	raw, err := os.ReadFile(filepath.Join(layout.Machine, "agent.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &record))
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	must(t, err)
	token, err := capability.MintSigned(ed25519.PublicKey(public), owner.Sign,
		capability.Grant{Machine: state.MachineID, Run: run, Expires: time.Now().Add(10 * time.Minute).Unix()})
	must(t, err)
	client := &http.Client{Transport: &http.Transport{TLSClientConfig: pin.TLSConfig()}}
	url := fmt.Sprintf("https://127.0.0.1:%d/v1/runs/%s/outputs/%s", record.WorkerPort, run, output)
	send := func(authorize bool, headers ...string) (*http.Response, []byte) {
		t.Helper()
		request, err := http.NewRequest(http.MethodGet, url, nil)
		must(t, err)
		if authorize {
			request.Header.Set("Authorization", "Cozy-Cap "+token)
		}
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
	return outputReader{
		get:       func(headers ...string) (*http.Response, []byte) { return send(true, headers...) },
		anonymous: func() int { response, _ := send(false); return response.StatusCode },
	}, state.MachineID
}

// checkOutput proves the output's current bytes with their revision as ETag, byte ranges,
// revalidation, and the digest once final.
func checkOutput(t *testing.T, r outputReader, digest string, revisions int) {
	t.Helper()
	whole, body := r.get()
	sum := sha256.Sum256(body)
	etag := fmt.Sprintf(`"r%d"`, revisions)
	if whole.StatusCode != http.StatusOK || "sha256:"+hex.EncodeToString(sum[:]) != digest || whole.Header.Get("ETag") != etag ||
		whole.Header.Get("Repr-Digest") != "sha-256=:"+base64.StdEncoding.EncodeToString(sum[:])+":" {
		t.Fatalf("the output answered %d %v with %d bytes: %.300s", whole.StatusCode, whole.Header, len(body), body)
	}
	middle := len(body) / 2
	part, piece := r.get("Range", fmt.Sprintf("bytes=%d-%d", middle, middle+9))
	if part.StatusCode != http.StatusPartialContent || string(piece) != string(body[middle:middle+10]) ||
		part.Header.Get("Content-Range") != fmt.Sprintf("bytes %d-%d/%d", middle, middle+9, len(body)) {
		t.Fatalf("a range answered %d %v", part.StatusCode, part.Header)
	}
	if same, _ := r.get("If-None-Match", etag); same.StatusCode != http.StatusNotModified {
		t.Fatalf("revalidating the current revision answered %d", same.StatusCode)
	}
	if stale, again := r.get("Range", "bytes=0-9", "If-Range", `"r1"`); stale.StatusCode != http.StatusOK || len(again) != len(body) {
		t.Fatalf("a range of an older revision answered %d with %d bytes", stale.StatusCode, len(again))
	}
	if code := r.anonymous(); code != http.StatusForbidden {
		t.Fatalf("an output without a capability answered %d", code)
	}
}

// machineServesOutput reads an output of the machine's newest run and returns the run's
// number. A machine whose Runtime predates run numbers answers 501 for the one route.
func machineServesOutput(t *testing.T, root, output, digest string, revisions int) string {
	t.Helper()
	layout, problem := home.Open(root)
	fatal(t, problem)
	host := machines.NewHost(layout.Machine, "", nil)
	found := &machines.Resolver{Host: host,
		UseRental: func(string, orchestrator.Holder) (func(), *exit.Error) { return func() {}, nil },
		RentalKey: func(string) (rental.CreatorIdentity, *exit.Error) { return rental.CreatorIdentity{}, nil }}
	machine, problem := found.Dial(context.Background(), machines.Local, orchestrator.Holder{What: "run outputs"})
	fatal(t, problem)
	list, err := machine.Host.ListMachineExecutions(context.Background(), &pb.MachineExecutionListQuery{
		Claim: machine.Claim, NewestFirst: true, Limit: 1})
	machine.Close()
	if status.Code(err) == codes.Unimplemented {
		reader, _ := machineOutputs(t, root, "1", output)
		if whole, body := reader.get(); whole.StatusCode != http.StatusNotImplemented {
			t.Fatalf("a machine without run numbers answered %d for its outputs: %s", whole.StatusCode, body)
		}
		return ""
	}
	must(t, err)
	if len(list.Executions) == 0 || list.Executions[0].Number == 0 {
		t.Fatalf("the machine lists no numbered run: %+v", list)
	}
	run := strconv.FormatUint(list.Executions[0].Number, 10)
	reader, _ := machineOutputs(t, root, run, output)
	checkOutput(t, reader, digest, revisions)
	return run
}

// machineServesOutputAsleep makes this fixture's Runtime structurally unavailable
// while its machine remains up. Finished output reads still serve their bytes/ranges
// from retained state and neither relaunch Runtime nor renew the idle ledger.
func machineServesOutputAsleep(t *testing.T, root, run, output, digest string, revisions int) {
	t.Helper()
	if run == "" {
		t.Log("the machine has no run numbers; its outputs are not served")
		return
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	var record struct {
		PID        int `json:"pid"`
		WorkerPort int `json:"worker_port"`
	}
	raw, err := os.ReadFile(filepath.Join(layout.Machine, "agent.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &record))
	host := machines.NewHost(layout.Machine, "", nil)
	state, problem := host.Status()
	fatal(t, problem)
	pin, problem := host.Pin()
	fatal(t, problem)
	owner, problem := host.Owner()
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	must(t, err)
	transport := &http.Transport{TLSClientConfig: pin.TLSConfig()}
	defer transport.CloseIdleConnections()
	maintenance := &machines.Maintenance{Base: fmt.Sprintf("https://127.0.0.1:%d", record.WorkerPort), Machine: state.MachineID, Public: ed25519.PublicKey(public), Sign: owner.Sign, Client: &http.Client{Transport: transport, Timeout: 5 * time.Second}}
	// Ordinary SIGKILL is recoverable. Instead, arrange the documented launch
	// refusal while keeping version/capability probes and read helpers functional.
	entry := filepath.Join(host.Root(), "opt/cozy/bin/cozy-runtime-worker")
	original := entry + ".output-proof-original"
	witness := filepath.Join(host.Root(), "tmp/output-proof-runtime-refused")
	must(t, os.Rename(entry, original))
	t.Cleanup(func() { _ = os.Remove(entry); must(t, os.Rename(original, entry)) })
	quote := func(value string) string { return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'" }
	wrapper := "#!/bin/sh\nif [ \"$#\" -eq 0 ]; then\n : > " + quote(witness) + "\n exit 6\nfi\nexec " + quote(original) + " \"$@\"\n"
	must(t, os.WriteFile(entry, []byte(wrapper), 0755))
	var pid int
	landed(t, "the fixture Runtime to stop", func() bool { pid = runtimeChild(record.PID); return pid != 0 })
	command, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "cmdline"))
	must(t, err)
	if !strings.Contains(string(command), filepath.Join(host.Root(), "opt/cozy")+"/") {
		t.Fatal("selected Runtime does not belong to this fixture")
	}
	must(t, syscall.Kill(pid, syscall.SIGKILL))
	landed(t, "structural Runtime refusal with the authenticated machine still available", func() bool {
		observed, problem := maintenance.State(t.Context())
		_, witnessErr := os.Stat(witness)
		return problem == nil && observed.Phase == "failed" && runtimeChild(record.PID) == 0 && witnessErr == nil
	})
	idle := filepath.Join(host.Root(), "var/lib/cozy/machine/idle.json")
	before := readText(idle)
	reader, _ := machineOutputs(t, root, run, output)
	checkOutput(t, reader, digest, revisions)
	if pid := runtimeChild(record.PID); pid != 0 {
		t.Fatalf("reading a finished run's output started Runtime %d", pid)
	}
	if after := readText(idle); after != before {
		t.Fatalf("reading a finished run's output moved the idle ledger:\n%s\n%s", before, after)
	}
	t.Logf("run %s's %s served while Runtime was unavailable for structural repair", run, output)
}

// runtimeChild is the Runtime the machine process runs, or 0.
func runtimeChild(parent int) int {
	entries, _ := os.ReadDir("/proc")
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		stat, _ := os.ReadFile(filepath.Join("/proc", entry.Name(), "stat"))
		fields := strings.Fields(string(stat[strings.LastIndexByte(string(stat), ')')+1:]))
		cmdline, _ := os.ReadFile(filepath.Join("/proc", entry.Name(), "cmdline"))
		if len(fields) > 1 && fields[1] == strconv.Itoa(parent) && strings.Contains(string(cmdline), "cozy-runtime-worker") {
			return pid
		}
	}
	return 0
}

func readText(path string) string {
	raw, _ := os.ReadFile(path)
	return string(raw)
}
