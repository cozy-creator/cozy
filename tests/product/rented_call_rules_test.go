package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// Owner rules of rented calls, restated for Runtime-owned machine execution (#758).

// A failed rented call shows Runtime's diagnosis of that call: two runs on one pod each
// read their own, and a terminal Runtime hands back under another run's identity is
// refused rather than shown as this run's.
func TestRentedRunShowsOnlyItsOwnDiagnosis(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	headroom := "device_group_infeasible: ordinal 0 has 84460830720 B measured headroom for 103012383680 B of declared weights"
	oom := "author_exception: CUDA out of memory in sample_fl2va"
	var first string
	machines := newTerminalMachines(func(payload map[string]any) *pb.AttemptOutcomeBody {
		switch payload["steps"] {
		case 1.0:
			return outcome(pb.OutcomeStatus_OUTCOME_STATUS_FAILED, headroom, nil)
		case 2.0:
			return outcome(pb.OutcomeStatus_OUTCOME_STATUS_FAILED, oom, nil)
		}
		// Runtime answers this run with the first run's terminal.
		misattributed := outcome(pb.OutcomeStatus_OUTCOME_STATUS_FAILED, headroom, nil)
		misattributed.RequestId = first
		return misattributed
	})
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machines}, nil)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()

	row, out := rentedRun(t, root, store, "headroom", "generate", "steps=1")
	first = row.ID
	_, again := rentedRun(t, root, store, "oom", "generate", "steps=2")
	if !strings.Contains(out, headroom) || strings.Contains(out, oom) || !strings.Contains(again, oom) || strings.Contains(again, headroom) {
		t.Fatalf("a run printed another run's diagnosis:\n%s\n%s", out, again)
	}
	if _, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=3", "--rental=tessa", "--json", "--idempotency-key", "stranger"); strings.Contains(out, headroom) {
		t.Fatalf("run 3 printed run 1's diagnosis: %s", out)
	}
	waitFor(t, root, "the refusal of another run's terminal", func() bool {
		third, _ := store.RequestByIdempotencyKey("stranger")
		return third != nil && strings.Contains(tail(filepath.Join(root, "daemon.log")),
			"machine execution "+third.ID+": machine outcome differs from its current execution")
	})
	var page struct {
		Invocations []struct {
			Number int64  `json:"number"`
			Status string `json:"status"`
			Error  string `json:"error"`
		}
	}
	_, list := runCozy(t, root, "run", "list", "--json", "--full")
	must(t, json.Unmarshal([]byte(list), &page))
	errors := map[int64]string{}
	for _, run := range page.Invocations {
		errors[run.Number] = run.Status + ": " + run.Error
	}
	if errors[1] != "failed: "+headroom || errors[2] != "failed: "+oom || strings.Contains(errors[3], headroom) {
		t.Fatalf("run list attributes a diagnosis to another run: %v", errors)
	}
}

// A rented call whose result is text alone creates no output directory and owes no export.
func TestRentedScalarResultCreatesNoOutputDirectory(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.workflow = []byte(`{"application":"h3:app","entrypoints":[{"models":[{"class":"H3","component_use":{"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[{"name":"text","type":"str"}]}}],"format":"cozy.package.interface/1","jobs":[]}`)
	machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		body := outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
		body.Result.InlineResult = []byte(`{"text":"Polo"}`)
		return body
	})
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machines}, nil)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	row, out := rentedRun(t, root, store, "scalar", "generate", "steps=1")
	if row.State != "succeeded" {
		t.Fatalf("the scalar run did not succeed: %s\n%s", row.State, out)
	}
	for _, directory := range []string{layout.PackageOutputs(ladderPackage), layout.PublicationRoot(row.Org, row.ID)} {
		if _, err := os.Stat(directory); !os.IsNotExist(err) {
			t.Fatalf("a text result created %s: %v", directory, err)
		}
	}
	if export, problem := store.OutputExportOf(row.ID); problem != nil || export != nil {
		t.Fatalf("a text result owes a file export: %+v %v", export, problem)
	}
}

// A transport loss while the pod prepares a rented call leaves the run queued, saying
// why; the next attempt prepares again and the call is submitted once.
func TestRentedPreparationTransportLossKeepsTheRunQueued(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	pod := &fakePod{machine: machines, prepareUnavailable: 1}
	root, layout := rentedLadderMachine(t, h, pod, nil)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	row, out := rentedRun(t, root, store, "outage", "generate", "steps=1")
	if row.State != "succeeded" {
		t.Fatalf("the run did not survive the transport loss: %s\n%s", row.State, out)
	}
	events, problem := store.EventsAfter(row.ID, 0, 200)
	fatal(t, problem)
	parked := false
	for _, event := range events {
		reason, _ := event.Payload["reason"].(string)
		parked = parked || event.Type == "request.parked" && strings.Contains(reason, "Tensorhub is restarting")
		if event.Type == "request.failed" {
			t.Fatalf("the transport loss failed the run: %v", event.Payload)
		}
	}
	pod.mu.Lock()
	prepares := len(pod.prepares)
	pod.mu.Unlock()
	if !parked || prepares != 2 || len(machines.submitted()) != 1 {
		t.Fatalf("parked %v after the loss, %d preparation(s), %d submission(s); want parked, 2, 1", parked, prepares, len(machines.submitted()))
	}
}

// A rented call's LoRA stack is part of what the pod must prepare. A preparation that
// drops the stack, changes a strength or the order, or adds one nobody asked for is
// refused before anything reaches Runtime; the exact echo is submitted.
func TestRentedLoRAStackMustBeEchoedBeforeSubmission(t *testing.T) {
	style := "sha256:" + strings.Repeat("2", 64)
	for _, arm := range []string{"omitted", "changed_scale", "changed_order", "unexpected", "matched"} {
		t.Run(arm, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			served := h.server.Config.Handler
			h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/models/resolve" || r.URL.Query().Get("ref") != "proof/style@"+style {
					served.ServeHTTP(w, r)
					return
				}
				_ = json.NewEncoder(w).Encode(hub.ModelResolution{Model: "proof/style", ManifestID: style,
					ManifestLength: 128, Bytes: 1 << 20, Components: []string{"adapter"}})
			})
			machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
				return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
			})
			pod := &fakePod{machine: machines, preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
				placement := modelBearingPlacement(t)(download, pkg, release)
				var set pb.DownloadDelegation
				must(t, canonical.Unmarshal(download, &set))
				adapters := set.Models[0].Adapters
				if arm == "unexpected" {
					adapters = []*pb.DownloadAdapterRef{{Component: "fl2va_dit", Model: "proof/style", Manifest: style, SourceComponent: "adapter", Scale: "1"}}
				}
				slot := placement.Entrypoints[0].Slots[0]
				for index, adapter := range adapters {
					digest, err := canonical.Raw(adapter.Manifest)
					must(t, err)
					id := fmt.Sprintf("adapter-%d", index)
					placement.Models = append(placement.Models, &pb.Model{Id: id, Repo: adapter.Model, Version: adapter.Release,
						Lane: adapter.Lane, Manifest: &pb.Ref{Digest: digest, Length: 128}})
					slot.Adapters = append(slot.Adapters, &pb.ModelAdapter{Component: adapter.Component, ModelId: id,
						SourceComponent: adapter.SourceComponent, Scale: adapter.Scale})
				}
				switch arm {
				case "omitted":
					slot.Adapters = nil
				case "changed_scale":
					slot.Adapters[0].Scale = "1"
				case "changed_order":
					slot.Adapters[0], slot.Adapters[1] = slot.Adapters[1], slot.Adapters[0]
				}
				return placement
			}}
			root, layout := rentedLadderMachine(t, h, pod, nil)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			var loras []string
			if arm != "unexpected" {
				loras = []string{"--lora", "model:fl2va_dit=proof/style#" + style + ",0", "--lora", "model:fl2va_dit=proof/style#" + style + ",-0.25"}
			}
			row, out := rentedRun(t, root, store, arm, "generate", append([]string{"steps=1"}, loras...)...)
			if arm == "matched" {
				if row.State != "succeeded" || len(machines.submitted()) != 1 {
					t.Fatalf("the echoed stack was not submitted: %s\n%s", row.State, out)
				}
				return
			}
			var page struct {
				Invocations []struct {
					Status    string `json:"status"`
					ErrorType string `json:"error_type"`
				}
			}
			_, list := runCozy(t, root, "run", "list", "--json", "--full")
			must(t, json.Unmarshal([]byte(list), &page))
			if len(page.Invocations) != 1 || page.Invocations[0].Status != "failed" ||
				page.Invocations[0].ErrorType != "model_adapters_preparation_mismatch" || len(machines.submitted()) != 0 {
				t.Fatalf("a preparation that %s the LoRA stack was not refused before submission (%d submitted): %s",
					arm, len(machines.submitted()), list)
			}
		})
	}
}

// constructionInterface is H3's shape: two serving entrypoints over one construction.
const constructionInterface = `{"application":"h3:app","entrypoints":[` +
	`{"models":[{"class":"H3","component_use":{"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}},` +
	`{"models":[{"class":"H3","component_use":{"sample_ref2va":["ref2va_dit"]},"path":"extend.models.model"}],"name":"extend","request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}}` +
	`],"format":"cozy.package.interface/1","jobs":[]}`

const tenantPackage = "proof/tenant"

// publishTenant publishes proof/tenant@0.1.0, a second package whose serving `generate`
// binds the ladder's model, and returns its interface.
func publishTenant(t *testing.T, h *ladderHub) []byte {
	t.Helper()
	iface, err := canonical.NormalizeJCS([]byte(`{"application":"tenant:app","entrypoints":[{"models":[{"class":"H3","component_use":{"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`))
	must(t, err)
	exact := func(raw []byte) hub.ExactDocument {
		return hub.ExactDocument{CanonicalBytes: raw, Digest: mustSpell(raw), Length: int64(len(raw))}
	}
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = iface
	detail.Release.Release = "0.1.0"
	detail.Release.PackageInterfaceDigest = mustSpell(iface)
	detail.Release.PackageInterfaceLength = int64(len(iface))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
	wheel := []byte("tenant wheel")
	plan := hub.PackageDownloadPlan{Release: "0.1.0",
		PackageConfig:    exact([]byte("[application]\nobject = \"tenant:app\"\n")),
		PackageInterface: exact(iface),
		Pyproject:        exact([]byte("[project]\nname = \"tenant\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\ndependencies = []\n")),
		UVLock:           exact([]byte("version = 1\nrequires-python = \">=3.12\"\n\n[[package]]\nname = \"tenant\"\nversion = \"0.1.0\"\nsource = { editable = \".\" }\n")),
		Downloads: []hub.PackageInstallDownload{{Kind: "project_wheel", Path: "tenant-0.1.0-py3-none-any.whl",
			Distribution: "tenant", Version: "0.1.0", Digest: mustSpell(wheel), Length: int64(len(wheel)),
			Tags: []string{"py3-none-any"}, ImportRoots: []string{"tenant"}}}}
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body any
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/"+tenantPackage:
			body = hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "tenant"}, Releases: []hub.ReleaseSummary{{Release: "0.1.0"}}}
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/"+tenantPackage+"/releases/0.1.0":
			body = detail
		case r.Method == http.MethodPost && r.URL.Path == "/v1/packages/"+tenantPackage+"/download":
			body = plan
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/"+tenantPackage+"/bindings":
			body = map[string]any{"bindings": []hub.PackageBindingRow{goodLadder()}}
		default:
			served.ServeHTTP(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(body)
	})
	return iface
}

// A warm pod serves what it already prepared (run 864: every alternation was a ~100 GB
// prepare for zero new bytes) and stacks packages without limit. The binding is the one
// the pod authored for the requested function — `generate` is the second entrypoint the
// placement lists — and the sibling entrypoint, a repeat, and a repeat after another
// package joined the pod each send the byte-identical preparation the pod already holds.
func TestRentedPodServesNamedSiblingsRepeatsAndCoTenants(t *testing.T) {
	h := newLadderHub(t)
	h.workflow = []byte(constructionInterface)
	sibling := goodLadder()
	sibling.Slot = "extend.models.model"
	h.mu.Lock()
	h.bindings = []hub.PackageBindingRow{goodLadder(), sibling}
	h.mu.Unlock()
	tenant := publishTenant(t, h)
	machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	pod := &fakePod{machine: machines, retainPrepared: true, preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
		placement := modelBearingPlacement(t)(download, pkg, release)
		if pkg == tenantPackage {
			placement.PackageInterface = tenant
		}
		return placement
	}}
	root, layout := rentedLadderMachine(t, h, pod, nil)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()

	runs := []struct{ key, target string }{{"generate", "generate"}, {"extend", "extend"},
		{"tenant", tenantPackage + "/generate"}, {"generate-again", "generate"}}
	for _, run := range runs {
		argv := []string{"run", ladderPackage + "/" + run.target, "steps=1"}
		if strings.Contains(run.target, "/") {
			argv[1] = run.target
		}
		if code, out := runCozy(t, root, append(argv, "--rental=tessa", "--json", "--idempotency-key", run.key)...); code != 0 {
			t.Fatalf("rented run %s was refused [exit %d]: %s", run.key, code, out)
		}
	}
	submitted := machines.submitted()
	if len(submitted) != len(runs) {
		t.Fatalf("the pod accepted %d of %d calls", len(submitted), len(runs))
	}
	placement := ""
	for index, run := range runs {
		row, problem := store.RequestByIdempotencyKey(run.key)
		fatal(t, problem)
		submission := submitted[index]
		if submission.Offer.RequestId != row.ID {
			t.Fatalf("call %d reached the pod out of order", index)
		}
		if row.Package != ladderPackage {
			continue
		}
		set, err := canonical.Read(submission.PreparedState.GetPlacementSet().PlacementSetCanonicalBytes, &pb.PlacementSet{})
		must(t, err)
		authored := map[string]string{}
		var names []string
		for _, entrypoint := range set.List("placements")[0].List("entrypoints") {
			authored[entrypoint.Str("name")] = entrypoint.Str("entrypoint_binding_digest")
			names = append(names, entrypoint.Str("name"))
		}
		var spec pb.InvocationSpec
		must(t, canonical.Unmarshal(submission.Offer.InvocationSpecCanonicalBytes, &spec))
		binding := spec.GetServing().GetEntrypointBindingDigest()
		if names[0] != "extend" || authored["generate"] == authored["extend"] ||
			binding != authored[row.Entrypoint] || row.PlanID != binding {
			t.Fatalf("%s was submitted under binding %s; the pod authored %v in order %v", row.Entrypoint, binding, authored, names)
		}
		if placement == "" {
			placement = submission.Offer.PlacementId
		}
		if submission.Offer.PlacementId != placement {
			t.Fatalf("%s left the warm placement %s for %s", run.key, placement, submission.Offer.PlacementId)
		}
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	var h3 []*pb.PreparePackageSetCall
	for _, call := range pod.prepares {
		var set pb.DownloadDelegation
		must(t, canonical.Unmarshal(call.PackageSet.DownloadDelegation, &set))
		if len(set.Packages) != 1 {
			t.Fatalf("one preparation named %d packages", len(set.Packages))
		}
		if set.Packages[0].Package == ladderPackage {
			h3 = append(h3, call)
		}
	}
	if len(h3) != 3 || len(pod.prepares) != 4 {
		t.Fatalf("%d preparations, %d of them proof/h3; want 4 and 3", len(pod.prepares), len(h3))
	}
	for _, call := range h3[1:] {
		if !proto.Equal(call, h3[0]) {
			t.Fatal("a sibling or repeated call sent a preparation the pod does not hold; it restages the construction")
		}
	}
}

// A package whose preparation the pod refuses fails its own run with the pod's code; a
// co-tenant package on the same pod runs before it and again after it.
func TestRentedRefusedPackageFailsAloneOnASharedPod(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	publishTenant(t, h)
	machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	pod := &fakePod{machine: machines, refusePrepare: map[string]*pb.PrepareEvent{
		tenantPackage: {SafeCode: "wheel_download_failed", SafeDetail: "tenant-0.1.0-py3-none-any.whl: transfer interrupted"}}}
	root, layout := rentedLadderMachine(t, h, pod, nil)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	for _, run := range []struct{ key, target, state string }{{"before", ladderPackage + "/generate", "succeeded"},
		{"refused", tenantPackage + "/generate", "failed"}, {"after", ladderPackage + "/generate", "succeeded"}} {
		_, out := runCozy(t, root, "run", run.target, "steps=1", "--rental=tessa", "--json", "--idempotency-key", run.key)
		var row *records.Request
		waitFor(t, root, "run "+run.key+" to settle ("+out+")", func() bool {
			row, problem = store.RequestByIdempotencyKey(run.key)
			return problem == nil && row != nil && records.Settled(row.State)
		})
		if row.State != run.state {
			t.Fatalf("run %s settled %s, want %s: %s", run.key, row.State, run.state, out)
		}
		if run.state == "failed" {
			errType, errCode, message, problem := store.SettledFailure(row.ID)
			fatal(t, problem)
			if errType != "machine_execution.prepare_refused" || errCode != "wheel_download_failed" || !strings.Contains(message, "transfer interrupted") {
				t.Fatalf("the refused package's run lost the pod's refusal: %s %s %s", errType, errCode, message)
			}
		}
	}
	for _, submitted := range machines.submitted() {
		if row, _ := store.RequestRow(submitted.Offer.RequestId); row == nil || row.Package != ladderPackage {
			t.Fatalf("a refused package's call reached Runtime: %+v", row)
		}
	}
	if n := len(machines.submitted()); n != 2 {
		t.Fatalf("Runtime took %d calls; want the co-tenant's two", n)
	}
}

// intakeMachines is terminalMachines that also takes each root input tree the way
// Runtime's intake does, keeping every committed input's bytes by field. A later abort
// of a committed intake releases it and answers that commit's source.
type intakeMachines struct {
	*terminalMachines
	received  sync.Map // input id -> []byte
	committed sync.Map // retention id -> *pb.NativeByteTreeRef
}

func (m *intakeMachines) ImportInputTree(stream grpc.ClientStreamingServer[pb.InputTreeImportFrame, pb.NativeByteRetentionResult]) error {
	frame, err := stream.Recv()
	if err != nil {
		return err
	}
	header := frame.GetHeader()
	if header == nil || !bytes.Equal(canonical.Digest(header.ManifestCanonicalBytes), header.Manifest.GetDigest()) {
		return status.Error(codes.InvalidArgument, "input header")
	}
	identity := sha256.Sum256([]byte(header.RequestId + "\x00" + header.InputId))
	retention, _ := canonical.Spell(identity[:])
	var body []byte
	for {
		frame, err := stream.Recv()
		if err != nil {
			return err
		}
		if blob := frame.GetBlob(); blob != nil {
			if blob.Offset != uint64(len(body)) {
				return status.Error(codes.InvalidArgument, "blob offset")
			}
			body = append(body, blob.Data...)
			continue
		}
		result := &pb.NativeByteRetentionResult{RetentionId: retention, Released: frame.GetCommit().GetAbort()}
		if !result.Released {
			m.received.Store(header.InputId, body)
			receipt := sha256.Sum256(body)
			m.committed.Store(retention, &pb.NativeByteTreeRef{ProducerRootId: retention, ReceiptDigest: receipt[:],
				Manifest: header.Manifest, ContentBytes: header.ContentBytes})
		}
		if source, ok := m.committed.Load(retention); ok {
			result.Source = source.(*pb.NativeByteTreeRef)
		}
		return stream.SendAndClose(result)
	}
}

// Declared Assets reach a rented call as the caller gave them: each occurrence's label
// and order in the payload, each file's exact binding, and the bytes it was submitted with
// staged on the pod, though the caller's original changed and then vanished while the pod
// was still preparing, before any input was staged.
func TestRentedCallReceivesDeclaredAssetsInOrder(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.workflow = []byte(`{"application":"h3:app","entrypoints":[{"assets":{"kinds":[{"kind":"image","max_bytes":1024,"max_decoded_bytes":4096,"media_types":["image/png"]}],"parameter":"assets"},"models":[{"class":"H3","component_use":{"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"prompt","type":"str"},{"constraints":{"max_length":3,"min_length":1},"name":"assets","type":{"list":{"fields":[{"name":"asset","type":{"asset":"file"}},{"name":"label","type":"str","wire":"optional"}]}}}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`)
	machines := &intakeMachines{terminalMachines: newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})}
	entered, held := make(chan struct{}, 1), make(chan struct{})
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machines, prepareEvent: func(*pb.PrepareEvent) {
		select {
		case entered <- struct{}{}:
		default:
		}
		<-held
	}}, nil)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	photo := filepath.Join(t.TempDir(), "photo.png")
	must(t, os.WriteFile(photo, encoded.Bytes(), 0o600))

	printed := make(chan string, 1)
	go func() {
		_, out := runCozy(t, root, "run", ladderPackage+"/generate", "prompt=<Picture1> beside <Picture2>",
			"--asset", "alice="+photo, "--asset", photo, "--rental=tessa", "--json", "--idempotency-key", "assets")
		printed <- out
	}()
	waitFor(t, root, "the pod to begin preparing the run", func() bool {
		select {
		case <-entered:
			return true
		default:
			return false
		}
	})
	must(t, os.WriteFile(photo, bytes.Repeat([]byte("x"), encoded.Len()), 0o600))
	must(t, os.Remove(photo))
	close(held)
	out := <-printed
	var row *records.Request
	waitFor(t, root, "the rented call with Assets to settle", func() bool {
		row, problem = store.RequestByIdempotencyKey("assets")
		return problem == nil && row != nil && records.Settled(row.State)
	})
	if row.State != "succeeded" || len(machines.submitted()) != 1 {
		t.Fatalf("the rented call with Assets did not run: %s\n%s", row.State, out)
	}
	submission := machines.submitted()[0]
	var spec pb.InvocationSpec
	must(t, canonical.Unmarshal(submission.Offer.InvocationSpecCanonicalBytes, &spec))
	digest := mustSpell(encoded.Bytes())
	bound := map[string]*pb.InputBinding{}
	for _, input := range spec.Inputs {
		bound[input.InputId] = input
	}
	for order, field := range []string{"assets.0.asset", "assets.1.asset"} {
		input := bound[field]
		received, _ := machines.received.Load(field)
		if input == nil || input.Digest != digest || input.Length != uint64(encoded.Len()) || input.KindMime != "image/png" ||
			input.Order != uint32(order) || !bytes.Equal(received.([]byte), encoded.Bytes()) {
			t.Fatalf("%s did not reach the pod as the caller's file at position %d: %+v", field, order, input)
		}
	}
	var payload struct {
		Prompt string `json:"prompt"`
		Assets []struct{ Asset, Label string }
	}
	must(t, json.Unmarshal(submission.PayloadCanonicalBytes, &payload))
	if payload.Prompt != "<Picture1> beside <Picture2>" || len(payload.Assets) != 2 || payload.Assets[0].Label != "alice" ||
		payload.Assets[1].Label != "" || payload.Assets[0].Asset != digest || payload.Assets[1].Asset != digest {
		t.Fatalf("the rented payload lost the Assets' labels or order: %s", submission.PayloadCanonicalBytes)
	}
}

// Runtime's measured memory sizes the next selection. Only a succeeded, measured rented
// call counts, at its maximum over runs; its exact measured total sizes only the same
// request on the same SKU and width; another entrypoint inherits nothing.
func TestRentedMeasuredMemorySizesTheNextSelection(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machines := newTerminalMachines(func(payload map[string]any) *pb.AttemptOutcomeBody {
		switch payload["steps"] {
		case 1.0:
			return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", &pb.AttemptMetrics{RuntimeMs: 1,
				WorkingPeakDeviceBytes: 30 << 30, PeakDeviceMemoryBytes: 44 << 30})
		case 2.0:
			return outcome(pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "author_exception", &pb.AttemptMetrics{WorkingPeakDeviceBytes: 70 << 30})
		case 3.0:
			return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
		}
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", &pb.AttemptMetrics{WorkingPeakDeviceBytes: 20 << 30})
	})
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machines}, nil)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	var rows []*records.Request
	for steps := 1; steps <= 4; steps++ {
		row, _ := rentedRun(t, root, store, fmt.Sprint("measured-", steps), "generate", fmt.Sprint("steps=", steps))
		rows = append(rows, row)
	}
	measured, later := rows[0], rows[3]
	peaks, problem := store.WorkingPeaks(*measured)
	fatal(t, problem)
	if peak := peaks.For(measured.Models, "fake-x", 1); peak != (records.WorkingPeak{Bytes: 30 << 30, Runs: 2, TotalBytes: 44 << 30, TotalRuns: 1}) {
		t.Fatalf("the ledger kept %+v; want the two succeeded measured runs at their maximum and the one exact total", peak)
	}
	for _, other := range []records.WorkingPeak{peaks.For(measured.Models, "rtx-4090", 1), peaks.For(measured.Models, "fake-x", 2)} {
		if other.TotalRuns != 0 {
			t.Fatalf("another SKU or width inherited the measured total: %+v", other)
		}
	}
	if peaks, problem := store.WorkingPeaks(*later); problem != nil || peaks.For(later.Models, "fake-x", 1).TotalRuns != 0 {
		t.Fatalf("another request inherited the measured total: %+v %v", peaks, problem)
	}
	sibling := *measured
	sibling.Entrypoint = "extend"
	if peaks, problem := store.WorkingPeaks(sibling); problem != nil || len(peaks) != 0 {
		t.Fatalf("another entrypoint inherited the measurement: %+v %v", peaks, problem)
	}
	skus := []hub.RentalSKU{{Name: "fake-x", AcceleratorModel: "fake-4090", AcceleratorCount: 1, VRAMGB: 48},
		{Name: "rtx-4090", AcceleratorModel: "fake-4090", AcceleratorCount: 1, VRAMGB: 48}}
	sized := rental.Purchases(skus, measured.Models, true, false, rental.Constraints{Working: peaks})
	if !strings.Contains(sized[0].Fit, "measured total 44.0 GiB from 1 exact runs") || strings.Contains(sized[1].Fit, "measured total") {
		t.Fatalf("the measured total did not size its own SKU alone: %q / %q", sized[0].Fit, sized[1].Fit)
	}
}
