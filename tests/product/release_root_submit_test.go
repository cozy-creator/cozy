package producttest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// releaseMachine is a Runtime that takes roots by their release: it installs what it lacks
// from its own Hub, resolves and prepares on its own and mints each receipt. It keeps every
// input object it was sent, per owner, and counts everything that reaches it.
type releaseMachine struct {
	acceptingMachines
	mu        sync.Mutex
	installed map[string]bool
	objects   map[string]bool
	roots     []*pb.ReleaseRoot
	submits   atomic.Int32
	blobBytes atomic.Int64
	imports   atomic.Int32
	// acceptedAt is when each request's receipt was minted.
	acceptedAt sync.Map
	// changed wakes held event reads; waits counts the reads held open; running marks the
	// runs whose machine recorded that execution began.
	changed chan struct{}
	waits   atomic.Int32
	running sync.Map
	// ended marks the runs the machine canceled; stale, the one whose next state read
	// predates that terminal (it was journaled between the observer's two reads).
	ended sync.Map
	stale sync.Map
}

const resolvedBody = `{"installation_id":"inst-h3","models":[{"gpus":2,"lane":"fp8-adaln-pruned","manifest":{"digest":"sha256:` +
	`1111111111111111111111111111111111111111111111111111111111111111","length":161},"parameter":"model",` +
	`"release":"1.0.0-rc.1","repository":"proof/minimax"}],"package":"proof/h3","release":"1.0.0"}`

// events are what this machine journals for an accepted run: the identities it resolved,
// then, once it runs, that it began.
func (m *releaseMachine) events(request string) []*pb.MachineExecutionEvent {
	if !m.accepted(request) {
		return nil
	}
	out := []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: uint64(time.Now().UnixMilli()), Kind: "resolved", BodyCanonicalBytes: []byte(resolvedBody)}}
	if _, ok := m.running.Load(request); ok {
		out = append(out, &pb.MachineExecutionEvent{Sequence: 2, AttemptOrdinal: 1, AtMs: uint64(time.Now().UnixMilli()), Kind: "running", BodyCanonicalBytes: []byte(`{"generation":1}`)})
	}
	if _, ok := m.ended.Load(request); ok {
		out = append(out, &pb.MachineExecutionEvent{Sequence: uint64(len(out) + 1), AttemptOrdinal: 1, AtMs: uint64(time.Now().UnixMilli()), Kind: "canceled", BodyCanonicalBytes: []byte(`{"generation":1}`)})
	}
	return out
}

func (m *releaseMachine) GetMachineExecution(ctx context.Context, query *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	state, err := m.acceptingMachines.GetMachineExecution(ctx, query)
	if state != nil {
		state.Sequence = uint64(len(m.events(query.RequestId)))
		if _, ended := m.ended.Load(query.RequestId); ended {
			state.State = "canceled"
		}
		if _, stale := m.stale.LoadAndDelete(query.RequestId); stale {
			state.State, state.Sequence = "running", state.Sequence-1
		}
	}
	return state, err
}

func (m *releaseMachine) ListMachineExecutionEvents(ctx context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	for {
		m.mu.Lock()
		changed := m.changed
		m.mu.Unlock()
		all := m.events(query.Execution.RequestId)
		var page pb.MachineExecutionEventPage
		for _, event := range all {
			if event.Sequence > query.After {
				page.Events = append(page.Events, event)
			}
		}
		page.HeadSequence, page.NextAfter = uint64(len(all)), max(query.After, uint64(len(all)))
		if len(page.Events) > 0 || !query.Wait || changed == nil {
			return &page, nil
		}
		m.waits.Add(1)
		select {
		case <-changed:
		case <-ctx.Done():
			m.waits.Add(-1)
			return nil, status.FromContextError(ctx.Err()).Err()
		}
		m.waits.Add(-1)
	}
}

// begin records that a run started executing and wakes the reads held open for it.
func (m *releaseMachine) begin(request string) {
	m.running.Store(request, true)
	m.wake()
}

func (m *releaseMachine) wake() {
	m.mu.Lock()
	close(m.changed)
	m.changed = make(chan struct{})
	m.mu.Unlock()
}

func newReleaseMachine() *releaseMachine {
	return &releaseMachine{acceptingMachines: acceptingMachines{states: map[string]*pb.MachineExecutionState{}},
		installed: map[string]bool{}, objects: map[string]bool{}}
}

func (m *releaseMachine) GetMachineExecutionWorkspace(ctx context.Context, query *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	workspace, err := m.acceptingMachines.GetMachineExecutionWorkspace(ctx, query)
	return workspace, err
}

func (m *releaseMachine) SubmitMachineExecution(ctx context.Context, submit *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.submits.Add(1)
	root := submit.ReleaseRoot
	if root == nil || len(submit.CaptureCanonicalBytes) > 0 || submit.PreparedState != nil || len(submit.Offer.InvocationSpecCanonicalBytes) > 0 {
		return nil, status.Error(codes.InvalidArgument, "a release root names only its request")
	}
	m.mu.Lock()
	m.installed[root.Package+"@"+root.Release] = true // installed from its own Hub
	m.roots = append(m.roots, proto.Clone(root).(*pb.ReleaseRoot))
	m.mu.Unlock()
	receipt, err := m.acceptingMachines.SubmitMachineExecution(ctx, submit)
	m.acceptedAt.Store(submit.Offer.RequestId, time.Now())
	if receipt != nil {
		minted := sha256.Sum256([]byte(submit.SubmissionId))
		receipt.CaptureDigest, receipt.InvocationSpecDigest = minted[:], minted[:]
	}
	return receipt, err
}

func (m *releaseMachine) ImportInputTree(stream grpc.ClientStreamingServer[pb.InputTreeImportFrame, pb.NativeByteRetentionResult]) error {
	m.imports.Add(1)
	first, err := stream.Recv()
	if err != nil || first.GetHeader() == nil {
		return status.Error(codes.InvalidArgument, "input tree needs its header")
	}
	header := first.GetHeader()
	var members struct {
		Entries []struct {
			Blob struct{ Sha256 string } `json:"blob"`
		} `json:"entries"`
	}
	if json.Unmarshal(header.ManifestCanonicalBytes, &members) != nil {
		return status.Error(codes.InvalidArgument, "manifest")
	}
	sent := map[string]bool{}
	for {
		frame, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if blob := frame.GetBlob(); blob != nil {
			m.blobBytes.Add(int64(len(blob.Data)))
			sent[string(blob.Object.Digest)] = true
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, entry := range members.Entries {
		digest, _ := canonical.Raw("sha256:" + entry.Blob.Sha256)
		if !sent[string(digest)] && !m.objects[entry.Blob.Sha256] {
			return status.Error(codes.FailedPrecondition, "input_objects_absent: 1 object(s) must be sent")
		}
		m.objects[entry.Blob.Sha256] = true
	}
	receipt := sha256.Sum256([]byte(header.RequestId + header.InputId))
	retention, _ := canonical.Spell(receipt[:])
	return stream.SendAndClose(&pb.NativeByteRetentionResult{RetentionId: retention, Source: &pb.NativeByteTreeRef{
		ProducerRootId: retention, ReceiptDigest: receipt[:], Manifest: header.Manifest, ContentBytes: header.ContentBytes}})
}

// hostRuntimeRecorder is a PATH whose first cozy-runtime identifies itself, answers every
// image-prepare "raw" and records each question; asked returns (and clears) what it was asked.
func hostRuntimeRecorder(t *testing.T, root string) (path string, asked func() []string) {
	t.Helper()
	dir := t.TempDir()
	log := filepath.Join(dir, "asked")
	script := fmt.Sprintf(`#!/bin/sh
case "$*" in
*version*) echo version >> %[1]q; echo '{"distribution":"%[2]s","wire_protocol":"cozy.worker.v1+minor.%[3]d"}' ;;
*image-prepare*) source=$(sed -n 's/.*"source_path":"\([^"]*\)".*/\1/p'); echo "image-prepare $source" >> %[1]q
  echo "{\"status\":\"raw\",\"path\":\"$source\",\"media_type\":\"image/png\",\"profile\":\"image-fit/1\"}" ;;
*) echo "$*" >> %[1]q; exit 3 ;;
esac
`, log, hostruntime.ToolFloor, pb.WireMinor)
	must(t, os.WriteFile(filepath.Join(dir, hostruntime.Distribution), []byte(script), 0o755))
	path = dir
	for _, item := range childEnv(t, root) {
		if inherited, ok := strings.CutPrefix(item, "PATH="); ok {
			path += string(os.PathListSeparator) + inherited
		}
	}
	return path, func() []string {
		raw, _ := os.ReadFile(log)
		_ = os.Remove(log)
		if len(raw) == 0 {
			return nil
		}
		return strings.Split(strings.TrimSpace(string(raw)), "\n")
	}
}

// A published serving call to a named rental is one message the machine takes by release:
// no hub read, no preparation, no fit, no ladder, cold or warm. The machine installs from its
// own Hub. Warm, no input byte it already holds is sent again, and no new connection is made.
func TestWarmReleaseRootSubmitMakesNoHubCallsAndOneMachineRoundTrip(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	// Its references are an Assets slot that asks for image-fit/1 preparation, as H3's does.
	h.workflow = []byte(strings.Replace(string(workflowInterface), `"name":"generate","request":{"fields":[{"name":"steps","type":"int"}]}`,
		`"assets":{"kinds":[{"kind":"image","max_count":2,"media_types":["image/png"],"prepare":{"max_edge":64,"profile":"image-fit/1"}}],"parameter":"assets","view":"decoded"},`+
			`"name":"generate","request":{"fields":[{"name":"steps","type":"int"},{"constraints":{"min_length":1,"max_length":2},"name":"assets",`+
			`"type":{"list":{"fields":[{"name":"asset","type":{"asset":"file"}},{"name":"label","type":"str","wire":"optional"},`+
			`{"name":"fidelity","type":{"literal":["auto","high","low","medium"]},"wire":"optional"}]}}}]}`, 1))
	machine := newReleaseMachine()
	pod := &fakePod{}
	root := startRentedPod(t, h, pod, func(string) machineExecutionPeer { return machine },
		"--extra-index-url https://public.example/v1/index/proof/simple/\n")
	var hubCalls atomic.Int32
	var hubPaths sync.Map
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The daemon's fleet reconciliation reads rentals on its own cadence, run or no run.
		if !strings.HasPrefix(r.URL.Path, "/v1/rentals") || strings.Contains(r.URL.Path, "prepare-facts") || strings.Contains(r.URL.Path, "image-inventory") {
			hubCalls.Add(1)
			hubPaths.Store(r.Method+" "+r.URL.Path, true)
		}
		served.ServeHTTP(w, r)
	})
	// One reference already fits the policy; the other is larger and is the host Runtime's
	// to prepare. The Runtime on PATH records what it is asked and answers "raw".
	pictures := t.TempDir()
	picture, large := filepath.Join(pictures, "mara.png"), filepath.Join(pictures, "veyra.png")
	for path, edge := range map[string]int{picture: 48, large: 96} {
		file, err := os.Create(path)
		must(t, err)
		must(t, png.Encode(file, image.NewRGBA(image.Rect(0, 0, edge, edge))))
		must(t, file.Close())
	}
	path, hostRuntime := hostRuntimeRecorder(t, root)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	// The package is installed here, as `cozy package install` leaves it: its release is known.
	layout, problem := home.Open(root)
	fatal(t, problem)
	installed := records.PackageInstall{ID: "5eed5eed5eed5eed", Package: ladderPackage, Major: 1, Version: "1.0.0",
		SourceKind: "tensorhub", SourceRef: ladderPackage + "@1.0.0", Verified: true, Dir: layout.InstallDir("5eed5eed5eed5eed")}
	iface, err := canonical.NormalizeJCS(h.workflow)
	must(t, err)
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0o700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installed.Dir), iface, 0o444))
	_, problem = store.Activate(installed)
	fatal(t, problem)
	run := func(key string, extra ...string) time.Duration {
		began := time.Now()
		if code, out := runCozyPath(t, root, path, append([]string{"run", ladderPackage + "/generate", "steps=1", "--asset", picture, "--asset", large,
			"--rental=tessa", "--json", "--idempotency-key", key}, extra...)...); code != 0 {
			t.Fatalf("the %s run was refused [exit %d]: %s", key, code, out)
		}
		var id string
		waitFor(t, root, "the "+key+" acceptance", func() bool {
			row, problem := store.RequestByIdempotencyKey(key)
			if problem == nil && row != nil {
				id = row.ID
			}
			return id != "" && machine.accepted(id)
		})
		at, _ := machine.acceptedAt.Load(id)
		return at.(time.Time).Sub(began)
	}

	// Only the image larger than its policy starts a Runtime process: one admission and one
	// preparation. A reference that already fits costs no process at all.
	prepared := func(key string) {
		if asked := hostRuntime(); !slices.Equal(asked, []string{"version", "image-prepare " + large}) {
			t.Fatalf("the %s CLI asked the host Runtime %q; want only the larger image prepared", key, asked)
		}
	}
	cold := run("cold")
	prepared("cold")
	if calls := hubCalls.Load(); calls != 0 {
		var paths []string
		hubPaths.Range(func(key, _ any) bool { paths = append(paths, key.(string)); return true })
		t.Fatalf("a cold run made %d hub calls; want none: %v", calls, paths)
	}
	if machine.submits.Load() != 1 || !machine.installed[ladderPackage+"@1.0.0"] {
		t.Fatalf("a cold machine installs by itself on the one submit; %d submits", machine.submits.Load())
	}
	coldBytes := machine.blobBytes.Load()
	if coldBytes == 0 {
		t.Fatal("the cold input's bytes never reached the machine")
	}
	pod.mu.Lock()
	prepares, dials := len(pod.prepares), pod.protocolReads
	pod.mu.Unlock()
	hubCalls.Store(0)
	hubPaths.Clear()
	machine.submits.Store(0)

	warm := run("warm")
	prepared("warm")
	if calls := hubCalls.Load(); calls != 0 {
		var paths []string
		hubPaths.Range(func(key, _ any) bool { paths = append(paths, key.(string)); return true })
		t.Fatalf("a warm resubmission made %d hub calls; want none: %v", calls, paths)
	}
	if machine.submits.Load() != 1 {
		t.Fatalf("a warm resubmission took %d submits; want one", machine.submits.Load())
	}
	if machine.blobBytes.Load() != coldBytes {
		t.Fatalf("the warm run sent %d input bytes the machine already held", machine.blobBytes.Load()-coldBytes)
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.prepares) != prepares || prepares != 0 {
		t.Fatalf("release roots made %d preparation calls; want none", len(pod.prepares))
	}
	if pod.protocolReads != dials {
		t.Fatalf("the warm run dialed the machine again (%d probes)", pod.protocolReads-dials)
	}
	last := machine.roots[len(machine.roots)-1]
	if last.Entrypoint != "generate" || len(last.Models) != 0 || len(last.InputAccess) != 2 {
		t.Fatalf("the warm root is not the one message the machine resolves: %+v", last)
	}
	// The machine's own record of what it installed and resolved is the run's identity.
	waitFor(t, root, "the resolved identities", func() bool {
		_, out := runCozy(t, root, "run", "show", "3", "--json")
		var report struct {
			Resolved struct {
				InstallationID string `json:"installation_id"`
				Models         []struct {
					Repository, Lane string
					Manifest         struct{ Digest string }
				}
			}
		}
		return json.Unmarshal([]byte(out), &report) == nil && report.Resolved.InstallationID == "inst-h3" &&
			len(report.Resolved.Models) == 1 && report.Resolved.Models[0].Lane == "fp8-adaln-pruned" &&
			strings.HasPrefix(report.Resolved.Models[0].Manifest.Digest, "sha256:1111")
	})
	t.Logf("CLI start to machine acceptance: cold %s, warm %s", cold.Round(time.Millisecond), warm.Round(time.Millisecond))

	// A provider source named at its commit is the machine's to resolve, narrow and convert:
	// no Hub call, no provider call here, one submission, and the identical root again.
	source := "hf://example/diffusers@" + strings.Repeat("c", 40)
	for _, key := range []string{"source-cold", "source-warm"} {
		hubCalls.Store(0)
		hubPaths.Clear()
		machine.submits.Store(0)
		run(key, "model.model="+source, "--source-profile", "model=fixture/diffusers/1")
		if calls := hubCalls.Load(); calls != 0 {
			var paths []string
			hubPaths.Range(func(key, _ any) bool { paths = append(paths, key.(string)); return true })
			t.Fatalf("the %s source run made %d hub calls; want none: %v", key, calls, paths)
		}
		if machine.submits.Load() != 1 {
			t.Fatalf("the %s source run took %d submits; want one", key, machine.submits.Load())
		}
		last := machine.roots[len(machine.roots)-1]
		if len(last.Models) != 1 || last.Models[0].Parameter != "model" || last.Models[0].Source != source ||
			!slices.Equal(last.Models[0].Profiles, []string{"fixture/diffusers/1"}) || last.Models[0].Repository != "" {
			t.Fatalf("the %s root does not name the source for the machine: %+v", key, last.Models)
		}
	}

	// A published job is a release root too: the machine resolves its Model and mints the
	// job. Warm, it costs no Hub call and one submission, and names the owner's grant.
	for _, key := range []string{"job-cold", "job-warm"} {
		hubCalls.Store(0)
		hubPaths.Clear()
		machine.submits.Store(0)
		if code, out := runCozy(t, root, "run", ladderPackage+"/long_form", "--rental=tessa", "--json",
			"--idempotency-key", key); code != 0 {
			t.Fatalf("the %s job was refused [exit %d]: %s", key, code, out)
		}
		var id string
		waitFor(t, root, "the "+key+" acceptance", func() bool {
			row, problem := store.RequestByIdempotencyKey(key)
			if problem == nil && row != nil {
				id = row.ID
			}
			return id != "" && machine.accepted(id)
		})
		if calls := hubCalls.Load(); key == "job-warm" && calls != 0 {
			var paths []string
			hubPaths.Range(func(key, _ any) bool { paths = append(paths, key.(string)); return true })
			t.Fatalf("a warm job made %d hub calls; want none: %v", calls, paths)
		}
		if machine.submits.Load() != 1 {
			t.Fatalf("the %s job took %d submits; want one", key, machine.submits.Load())
		}
		last := machine.roots[len(machine.roots)-1]
		if !last.Job || last.Entrypoint != "long_form" || last.PublicationGrant != "local/_job-"+id ||
			len(last.Models) != 0 || len(last.InputAccess) != 0 {
			t.Fatalf("the %s job is not the one message the machine mints: %+v", key, last)
		}
	}
}

// The observer holds one events read open on its kept connection: a run's next machine
// event reaches its record at once, with no clock and no new dial.
func TestObserverTakesTheMachinesNextEventWithoutPolling(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := newReleaseMachine()
	machine.changed = make(chan struct{})
	pod := &fakePod{}
	root := startRentedPod(t, h, pod, func(string) machineExecutionPeer { return machine },
		"--extra-index-url https://public.example/v1/index/proof/simple/\n")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", "held"); code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	var id string
	waitFor(t, root, "an events read held open", func() bool {
		if row, problem := store.RequestByIdempotencyKey("held"); problem == nil && row != nil {
			id = row.ID
		}
		return id != "" && machine.waits.Load() > 0
	})
	pod.mu.Lock()
	dials := pod.protocolReads
	pod.mu.Unlock()
	began := time.Now()
	machine.begin(id)
	waitFor(t, root, "the observed start", func() bool {
		events, problem := store.EventsAfter(id, 0, 256)
		for _, event := range events {
			if problem == nil && event.Type == "request.accepted" {
				return true
			}
		}
		return false
	})
	took := time.Since(began)
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if pod.protocolReads != dials {
		t.Fatalf("observing the run dialed the machine %d more times", pod.protocolReads-dials)
	}
	if took > time.Second {
		t.Fatalf("the machine's event reached the run's record after %s", took)
	}
	t.Logf("machine event to recorded: %s", took.Round(time.Millisecond))
}

// A checkpoint named by digest alone (a lane no release names yet) reaches the machine as
// the release root's exact choice (runs 1406/1408/1411: it was never asked).
func TestCheckpointOverrideGoesByReleaseRoot(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &runtimeMachine{blocker: "none"}
	pod := &fakePod{machine: machine, deviceCount: 4}
	root, layout := rentedLadderMachine(t, h, pod, nil)
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "model.model="+ladderModel+"#"+bf16Manifest,
		"--rental=tessa", "--json", "--idempotency-key", "checkpoint"); code != 0 {
		t.Fatalf("the checkpoint override was refused [exit %d]: %s", code, out)
	}
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	waitFor(t, root, "the submission or a settled run", func() bool {
		row, problem := store.RequestByIdempotencyKey("checkpoint")
		return problem == nil && row != nil && (machine.submitted() != nil || records.Settled(row.State))
	})
	submission := machine.submitted()
	if submission == nil {
		_, show := runCozy(t, root, "run", "show", "2", "--json")
		t.Fatalf("the checkpoint override never reached the machine: %s", show)
	}
	digest, err := canonical.Raw(bf16Manifest)
	must(t, err)
	choices := submission.GetReleaseRoot().GetModels()
	if len(choices) != 1 || choices[0].Repository != ladderModel || !bytes.Equal(choices[0].Manifest.GetDigest(), digest) {
		t.Fatalf("the release root does not carry the exact checkpoint: %+v", choices)
	}
}

// sourceJobHome is a rented pod whose Runtime takes jobs with provider sources by release root.
func sourceJobHome(t *testing.T) (string, *releaseMachine, *records.Store, *ladderHub) {
	t.Helper()
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := newReleaseMachine()
	root := startRentedPod(t, h, &fakePod{}, func(string) machineExecutionPeer { return machine },
		"--extra-index-url https://public.example/v1/index/proof/simple/\n")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	return root, machine, store, h
}

const sourceJobModel = "hf://example/diffusers@cccccccccccccccccccccccccccccccccccccccc"

func sourceJob(t *testing.T, root, key string) (int, string) {
	return runCozy(t, root, "run", ladderPackage+"/long_form", "model.source="+sourceJobModel,
		"--source-profile", "source=fixture/diffusers/1", "--rental=tessa", "--json", "--idempotency-key", key)
}

// A job's provider-source Model on a machine whose Runtime makes it goes by release root:
// the machine narrows the source; the host reads no provider and records no transfer.
func TestAJobsProviderSourceGoesByReleaseRootWhereTheMachineMakesIt(t *testing.T) {
	root, machine, store, _ := sourceJobHome(t)
	if code, out := sourceJob(t, root, "source-root"); code != 0 {
		t.Fatalf("the source job was refused [exit %d]: %s", code, out)
	}
	var id string
	waitFor(t, root, "the source job's acceptance", func() bool {
		row, problem := store.RequestByIdempotencyKey("source-root")
		if problem == nil && row != nil {
			id = row.ID
		}
		return id != "" && machine.accepted(id)
	})
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	machine.mu.Lock()
	last := machine.roots[len(machine.roots)-1]
	machine.mu.Unlock()
	if row.ModelTransfer != nil || !last.Job || len(last.Models) != 1 || last.Models[0].Source != sourceJobModel ||
		!slices.Equal(last.Models[0].Profiles, []string{"fixture/diffusers/1"}) {
		t.Fatalf("the source job did not go by release root: transfer %+v, root %+v", row.ModelTransfer, last)
	}
}

// A provider-source job with no machine named goes by release root too: the source's
// reviewed carriers size the rental that placement picks (provider reads only; no transfer is
// recorded), and the machine it lands on makes the source.
func TestAnAutoRentedProviderSourceJobGoesByReleaseRoot(t *testing.T) {
	const source = "hf://alibaba-pai/MiniMax-H3-Acc-LoRAs@335001fb9e5455d68a0caa18ec2e319072150328"
	probe, err := (&http.Client{Timeout: 20 * time.Second}).Get("https://huggingface.co/api/models/alibaba-pai/MiniMax-H3-Acc-LoRAs")
	if err != nil {
		t.Skipf("provider unreachable from this runner: %v", err)
	}
	probe.Body.Close()
	root, machine, store, _ := sourceJobHome(t)
	code, out := runCozy(t, root, "run", ladderPackage+"/long_form", "model.source="+source,
		"--source-profile", "source=hf/minimax-h3/pdd-fl2va-bf16/1", "--rental-only", "--json", "--idempotency-key", "source-auto")
	if code != 0 {
		t.Fatalf("the auto-rented source job was refused [exit %d]: %s", code, out)
	}
	var id string
	waitFor(t, root, "the auto-rented source job's acceptance", func() bool {
		row, problem := store.RequestByIdempotencyKey("source-auto")
		if problem == nil && row != nil {
			id = row.ID
		}
		return id != "" && machine.accepted(id)
	})
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	planned, problem := store.PlannedSourceBytes(id)
	fatal(t, problem)
	machine.mu.Lock()
	last := machine.roots[len(machine.roots)-1]
	machine.mu.Unlock()
	sourced := slices.IndexFunc(last.Models, func(choice *pb.ModelChoice) bool { return choice.Parameter == "source" })
	if row.ModelTransfer != nil || row.Worker != "pr-test-0001" || planned <= 0 || !last.Job || sourced < 0 ||
		last.Models[sourced].Source != source || !slices.Equal(last.Models[sourced].Profiles, []string{"hf/minimax-h3/pdd-fl2va-bf16/1"}) {
		t.Fatalf("the auto-rented source job did not go by release root: worker %q, transfer %+v, planned %d, root %+v",
			row.Worker, row.ModelTransfer, planned, last)
	}
}

// A terminal the machine journals between the observer's state read and its event read is
// taken at once: the observer reads the state again instead of holding a read open for an
// event after the machine's last one.
func TestATerminalBetweenTheStateAndEventReadsIsNotWaitedFor(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := newReleaseMachine()
	machine.changed = make(chan struct{})
	root := startRentedPod(t, h, &fakePod{}, func(string) machineExecutionPeer { return machine },
		"--extra-index-url https://public.example/v1/index/proof/simple/\n")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", "raced"); code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	var id string
	waitFor(t, root, "an events read held open", func() bool {
		if row, problem := store.RequestByIdempotencyKey("raced"); problem == nil && row != nil {
			id = row.ID
		}
		return id != "" && machine.waits.Load() > 0
	})
	machine.stale.Store(id, true)
	machine.ended.Store(id, true)
	machine.wake()
	waitFor(t, root, "the observed terminal", func() bool {
		row, problem := store.RequestRow(id)
		return problem == nil && row != nil && row.State == "canceled"
	})
}

// A release this computer never installed is described by the machine it runs on: the run
// reads the interface from the machine, which reads its own Hub, and the client makes no
// Hub request at all.
func TestAReleaseNotInstalledHereIsDescribedByItsMachine(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := newReleaseMachine()
	root := startRentedPod(t, h, &fakePod{}, func(string) machineExecutionPeer { return machine }, "")
	var hubCalls atomic.Int32
	var hubPaths sync.Map
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1/rentals") { // the fleet's own reconciliation
			hubCalls.Add(1)
			hubPaths.Store(r.Method+" "+r.URL.Path, true)
		}
		served.ServeHTTP(w, r)
	})
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json",
		"--idempotency-key", "described"); code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	var id string
	waitFor(t, root, "the acceptance", func() bool {
		row, problem := store.RequestByIdempotencyKey("described")
		if problem == nil && row != nil {
			id = row.ID
		}
		return id != "" && machine.accepted(id)
	})
	if calls := hubCalls.Load(); calls != 0 {
		var paths []string
		hubPaths.Range(func(key, _ any) bool { paths = append(paths, key.(string)); return true })
		t.Fatalf("a run on a known machine made %d hub calls; want none: %v", calls, paths)
	}
	machine.mu.Lock()
	defer machine.mu.Unlock()
	if last := machine.roots[len(machine.roots)-1]; last.Release != "1.0.0" || last.Entrypoint != "generate" {
		t.Fatalf("the root does not name the release its machine described: %+v", last)
	}
}

// installedHere records a published release as `cozy package install` leaves it, so a run
// naming a rental these tests cannot reach reads the release's interface here.
func installedHere(t *testing.T, root, hubURL, pkg, release string) {
	t.Helper()
	response, err := http.Get(hubURL + "/v1/packages/" + pkg + "/releases/" + release)
	must(t, err)
	var detail struct {
		PackageInterface json.RawMessage `json:"package_interface"`
	}
	must(t, json.NewDecoder(response.Body).Decode(&detail))
	response.Body.Close()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	sum := sha256.Sum256([]byte(pkg + "@" + release))
	id := fmt.Sprintf("%x", sum[:8])
	installed := records.PackageInstall{ID: id, Package: pkg, Major: 1, Version: release,
		SourceKind: "tensorhub", SourceRef: pkg + "@" + release, Verified: true, Dir: layout.InstallDir(id)}
	iface, err := canonical.NormalizeJCS(detail.PackageInterface)
	must(t, err)
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0o700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installed.Dir), iface, 0o444))
	_, problem = store.Activate(installed)
	fatal(t, problem)
}
