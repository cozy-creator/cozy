package producttest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"image"
	"image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// releaseMachine is a Runtime that takes roots by their release: it names the releases it
// holds no installation of, installs from the facts that follow, resolves and prepares on
// its own and mints each receipt. It keeps every input object it was sent, per owner, and
// counts everything that reaches it.
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
	return out
}

func (m *releaseMachine) GetMachineExecution(ctx context.Context, query *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	state, err := m.acceptingMachines.GetMachineExecution(ctx, query)
	if state != nil {
		state.Sequence = uint64(len(m.events(query.RequestId)))
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
	if workspace != nil {
		workspace.ReleaseRoots, workspace.ResolvesModelDefaults, workspace.InputObjectReuse = true, true, true
		workspace.EventWait = m.changed != nil
	}
	return workspace, err
}

func (m *releaseMachine) SubmitMachineExecution(ctx context.Context, submit *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.submits.Add(1)
	root := submit.ReleaseRoot
	if root == nil || len(submit.CaptureCanonicalBytes) > 0 || submit.PreparedState != nil || len(submit.Offer.InvocationSpecCanonicalBytes) > 0 {
		return nil, status.Error(codes.InvalidArgument, "a release root names only its request")
	}
	key := root.Package + "@" + root.Release
	m.mu.Lock()
	for _, row := range root.Installations {
		if row.Key == key && row.Preparation.GetApplication() != "" && len(row.Preparation.LockedRequirements) > 0 {
			m.installed[key] = true
		}
	}
	installed := m.installed[key]
	m.roots = append(m.roots, proto.Clone(root).(*pb.ReleaseRoot))
	m.mu.Unlock()
	if !installed {
		_ = grpc.SetTrailer(ctx, metadata.Pairs("cozy-error-code", "release_root_installations_absent", "cozy-absent-release", key))
		return nil, status.Error(codes.FailedPrecondition, "install facts are absent for "+key)
	}
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

// A warm resubmission of a published serving call to a named rental is one message the
// machine takes by release: no hub read, no preparation, no fit, no ladder, no input byte
// the machine already holds, and no new connection. A cold machine asks for install facts.
func TestWarmReleaseRootSubmitMakesNoHubCallsAndOneMachineRoundTrip(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.workflow = []byte(strings.Replace(string(workflowInterface), `"request":{"fields":[{"name":"steps","type":"int"}]}`,
		`"request":{"fields":[{"asset_bound":{"max_bytes":67108864},"name":"image","type":{"asset":"image"}},{"name":"steps","type":"int"}]}`, 1))
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
	picture := filepath.Join(t.TempDir(), "mara.png")
	file, err := os.Create(picture)
	must(t, err)
	must(t, png.Encode(file, image.NewRGBA(image.Rect(0, 0, 48, 48))))
	must(t, file.Close())
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
	run := func(key string) time.Duration {
		began := time.Now()
		if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--asset", "image="+picture,
			"--rental=tessa", "--json", "--idempotency-key", key); code != 0 {
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

	cold := run("cold")
	if machine.submits.Load() != 2 || !machine.installed[ladderPackage+"@1.0.0"] {
		t.Fatalf("a cold machine takes the facts it named on its second submit; %d submits", machine.submits.Load())
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
	if last.Entrypoint != "generate" || len(last.Models) != 0 || len(last.Installations) != 0 || len(last.InputAccess) != 1 || last.CatalogOrigin != "https://public.example" {
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

// A machine whose Runtime predates release roots is not refused: the client resolves the
// call's slots and prepares it as before, and the run says how to reach the fast path.
func TestOlderRuntimeTakesTheCompatibilityPath(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &runtimeMachine{older: true}
	pod := &fakePod{machine: machine, deviceCount: 4}
	root, layout := rentedLadderMachine(t, h, pod, func(_ home.Layout, store *records.Store) {
		holder, _, problem := store.Submit(records.Request{ID: "req-gpu-holder", IdemKey: "gpu-holder", BodyDigest: childDigest("7"),
			Package: ladderPackage, Entrypoint: "generate", Payload: []byte(`{}`)})
		fatal(t, problem)
		_, problem = store.FailQueuedRequest(holder.ID, map[string]any{"error_type": "proof", "error": "held elsewhere"})
		fatal(t, problem)
		machine.blocker = holder.ID
	})
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", "older"); code != 0 {
		t.Fatalf("the run was refused on an older Runtime [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the Runtime submission", func() bool { return machine.submitted() != nil })
	machine.finish()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	var row *records.Request
	deadline := time.Now().Add(30 * time.Second)
	for {
		row, problem = store.RequestByIdempotencyKey("older")
		if problem == nil && row.State == "succeeded" {
			break
		}
		if time.Now().After(deadline) {
			_, show := runCozy(t, root, "run", "show", "2", "--json")
			t.Fatalf("the run did not complete (%s): %s\n%s", row.State, show, tail(filepath.Join(root, "daemon.log")))
		}
		time.Sleep(100 * time.Millisecond)
	}
	submission := machine.submitted()
	if submission.ReleaseRoot != nil || submission.PreparedState.GetPlacementSet() == nil {
		t.Fatalf("an older Runtime was sent a release root: %+v", submission)
	}
	pod.mu.Lock()
	prepares := len(pod.prepares)
	pod.mu.Unlock()
	if prepares == 0 || len(row.Models) != 1 || row.Models[0].Manifest == "" {
		t.Fatalf("the compatibility path did not prepare the pinned model: %d preparations, %+v", prepares, row.Models)
	}
	_, show := runCozy(t, root, "run", "show", "2", "--json")
	if !strings.Contains(show, "lacks release roots; used the compatibility path") || !strings.Contains(show, "cozy rental update") {
		t.Fatalf("the run does not say it took the compatibility path: %s", show)
	}
}

// A checkpoint named by digest alone (a lane no release names yet) reaches the machine on
// both paths: a release root carries it as the exact choice, and an older Runtime is asked
// to download and bind exactly that checkpoint (runs 1406/1408/1411: it was never asked).
func TestCheckpointOverrideRunsOnBothPaths(t *testing.T) {
	for _, older := range []bool{false, true} {
		t.Run(map[bool]string{false: "release root", true: "compatibility"}[older], func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := &runtimeMachine{older: older, blocker: "none"}
			pod := &fakePod{machine: machine, deviceCount: 4}
			root, layout := rentedLadderMachine(t, h, pod, nil)
			if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "model.model="+ladderModel+"#"+bf16Manifest,
				"--rental=tessa", "--json", "--idempotency-key", "checkpoint"); code != 0 {
				t.Fatalf("the checkpoint override was refused [exit %d]: %s", code, out)
			}
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			var row *records.Request
			waitFor(t, root, "the submission or a settled run", func() bool {
				row, problem = store.RequestByIdempotencyKey("checkpoint")
				return problem == nil && row != nil && (machine.submitted() != nil || records.Settled(row.State))
			})
			submission := machine.submitted()
			if submission == nil {
				_, show := runCozy(t, root, "run", "show", "2", "--json")
				t.Fatalf("the checkpoint override never reached the machine: %s", show)
			}
			digest, err := canonical.Raw(bf16Manifest)
			must(t, err)
			if !older {
				choices := submission.GetReleaseRoot().GetModels()
				if len(choices) != 1 || choices[0].Repository != ladderModel || !bytes.Equal(choices[0].Manifest.GetDigest(), digest) {
					t.Fatalf("the release root does not carry the exact checkpoint: %+v", choices)
				}
				return
			}
			pod.mu.Lock()
			prepares := append([]*pb.PreparePackageSetCall(nil), pod.prepares...)
			pod.mu.Unlock()
			asked := false
			for _, call := range prepares {
				var set pb.DownloadDelegation
				must(t, canonical.Unmarshal(call.PackageSet.DownloadDelegation, &set))
				for _, model := range set.Models {
					asked = asked || model.Model == ladderModel && model.Manifest == bf16Manifest && model.Slot == ladderSlot
				}
			}
			if !asked {
				t.Fatalf("the older Runtime was never asked for the checkpoint: %d preparation(s)", len(prepares))
			}
			machine.finish()
			waitFor(t, root, "the checkpoint run to succeed", func() bool {
				row, problem = store.RequestByIdempotencyKey("checkpoint")
				return problem == nil && row.State == "succeeded"
			})
		})
	}
}
