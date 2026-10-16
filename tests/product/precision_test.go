package producttest

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// xs-042: a machine picks every lane from its GPUs' family. A run carries its bits and the
// caller's own choices, nothing else; a rental the fleet buys for it is sized by the lanes
// the machine's rules would serve.

// recordingMachine is a cozy.machine.v1 machine on the Hub's target software that keeps every
// RunSpec it is sent and succeeds each run.
type recordingMachine struct {
	v1.UnimplementedMachineServer
	mu    sync.Mutex
	specs []*v1.RunSpec
	runs  map[string]bool
}

func (m *recordingMachine) Status(_ *v1.StatusRequest, stream grpc.ServerStreamingServer[v1.StatusFrame]) error {
	return stream.Send(&v1.StatusFrame{WorkerId: olderWorkerID, BootId: olderBootID, Version: "0.1.0", Phase: "ready",
		Runtime: "0.18.102", Tensorfs: "0.3.93", Capabilities: []string{"status/1", "run/1", "warm/1", "warm/2"}})
}

func (m *recordingMachine) Run(request *v1.RunRequest, stream grpc.ServerStreamingServer[v1.RunEvent]) error {
	m.mu.Lock()
	if spec := request.GetSpec(); spec != nil {
		m.specs, m.runs[request.GetId()] = append(m.specs, spec), true
	} else if !m.runs[request.GetId()] {
		m.mu.Unlock()
		return status.Error(codes.NotFound, "no such run")
	}
	m.mu.Unlock()
	now := time.Now().UnixMilli()
	if request.GetAfter() < 1 {
		if err := stream.Send(&v1.RunEvent{Sequence: 1, AtMs: now, Event: &v1.RunEvent_State{State: &v1.RunState{
			Id: request.GetId(), Number: 1, State: "running", Sequence: 1}}}); err != nil {
			return err
		}
	}
	return stream.Send(&v1.RunEvent{Sequence: 2, AtMs: now, Event: &v1.RunEvent_Outcome{Outcome: &v1.Outcome{Status: "succeeded", Result: []byte(`{}`)}}})
}

func (m *recordingMachine) Control(_ context.Context, request *v1.ControlRequest) (*v1.RunState, error) {
	return &v1.RunState{Id: request.GetId(), Number: 1, State: "succeeded", Sequence: 2}, nil
}

// spec is the first RunSpec of kind the machine was sent.
func (m *recordingMachine) spec(kind v1.RunKind) *v1.RunSpec {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, spec := range m.specs {
		if spec.GetKind() == kind {
			return spec
		}
	}
	return nil
}

// precisionInterface is a describer: a compute slot, and an LM decode slot that allows 4 bits.
const precisionInterface = `{"application":"describer:app","entrypoints":[{"name":"describe","models":[` +
	`{"class":"Vision","component_use":{},"path":"describe.models.model","default":"proof/vision@1.0.0"},` +
	`{"class":"Decoder","component_use":{},"path":"describe.models.decoder","default":"proof/decoder@1.0.0","min_bits":4,"phase":"decode"}],` +
	`"request":{"fields":[]},"result":{"fields":[]}}],"format":"cozy.package.interface/1"}`

// servePrecisionRelease publishes proof/describer@1.0.0 on mux, as the Hub answers it.
func servePrecisionRelease(t *testing.T, mux *http.ServeMux) {
	t.Helper()
	iface := []byte(precisionInterface)
	contract, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = iface
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = assessmentDigest(contract.Raw)
	detail.Release.PackageInterfaceLength = int64(len(iface))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25", "torch<3,>=2.13"}
	mux.HandleFunc("GET /v1/packages/proof/describer", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "describer"},
			Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
	})
	mux.HandleFunc("GET /v1/packages/proof/describer/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(detail)
	})
	// Each model publishes the vocabulary's lanes, unsized: whole-lane bytes are the need.
	lanes := map[string][3]int64{"vision": {18, 9, 36}, "decoder": {10, 5, 20}}
	for model, bytes := range lanes {
		card := hub.ModelCard{Model: hub.Resource{Org: "proof", Name: model}, Releases: []hub.ModelReleaseSummary{{
			ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0"}, Lanes: []hub.ModelLaneSummary{
				{Lane: "int8", ManifestID: precisionManifest(model, "int8"), Bytes: bytes[0] * gib},
				{Lane: "nvfp4", ManifestID: precisionManifest(model, "nvfp4"), Bytes: bytes[1] * gib},
				{Lane: "bf16", ManifestID: precisionManifest(model, "bf16"), Bytes: bytes[2] * gib}}}}}
		mux.HandleFunc("GET /v1/models/proof/"+model, func(w http.ResponseWriter, _ *http.Request) {
			_ = json.NewEncoder(w).Encode(card)
		})
	}
}

func precisionManifest(model, lane string) string {
	return mustSpell([]byte(model + "/" + lane))
}

// sizedLanes is each slot's sized lane as the request records it: facts no machine is sent.
func sizedLanes(t *testing.T, row *records.Request) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, model := range row.Models {
		if model.Choice {
			t.Fatalf("a default reached the request as the caller's choice: %+v", model)
		}
		out[model.BindingSlot()] = model.Lane
	}
	return out
}

// A rental the fleet buys is sized by the lane each slot's machine would serve, from lane
// bytes: 8 bits (int8) for the compute slot, nvfp4 for the decode slot that allows 4 bits.
// --bits holds every slot to that width. Every product gets the same lane-bytes check.
func TestARentalIsSizedByTheLanesTheMachineWouldServe(t *testing.T) {
	for _, arm := range []struct {
		bits           uint32
		model, decoder string
		chosen, short  string
		fit            string
	}{
		{0, "int8", "nvfp4", "rtx-4090", "", "lane_bytes 23.0 GiB (working memory unmeasured) of 24 GB"},
		{4, "nvfp4", "nvfp4", "rtx-4090", "", "lane_bytes 14.0 GiB (working memory unmeasured) of 24 GB"},
		{16, "bf16", "bf16", "h100-80", "rtx-5090", "lane_bytes 56.0 GiB (working memory unmeasured) of 80 GB"},
	} {
		t.Run(fmt.Sprintf("bits=%d", arm.bits), func(t *testing.T) {
			h := newStandInHub(t)
			h.provisions = true // the buy completes, so the placement is recorded
			servePrecisionRelease(t, h.mux)
			root := standInRoot(t, h)
			startCalciferProcess(t, root)
			args := []string{"run", "proof/describer/describe", "--rental-only", "--json", "--idempotency-key", "sized"}
			if arm.bits != 0 {
				args = append(args, fmt.Sprintf("--bits=%d", arm.bits))
			}
			_, out := runCozy(t, root, args...)
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			row, problem := store.RequestByIdempotencyKey("sized")
			fatal(t, problem)
			if row == nil {
				t.Fatalf("the run was not submitted: %s", out)
			}
			if lanes := sizedLanes(t, row); lanes["describe.models.model"] != arm.model ||
				lanes["describe.models.decoder"] != arm.decoder || row.Bits != arm.bits {
				t.Fatalf("sized %v at bits %d; want model=%s decoder=%s", lanes, row.Bits, arm.model, arm.decoder)
			}
			waitFor(t, root, "the placement record of "+out, func() bool { return len(placementEvents(t, store, row.ID)) > 0 })
			placement := placementEvent(t, store, row.ID)
			verdicts, fits := candidateVerdicts(placement), candidateFits(placement)
			if verdicts[arm.chosen] != "chosen" || fits[arm.chosen] != arm.fit ||
				arm.short != "" && !strings.HasPrefix(verdicts[arm.short], "excluded:vram_short") {
				t.Fatalf("verdicts %v fits %v; want %s chosen at %q", verdicts, fits, arm.chosen, arm.fit)
			}
			if placement["override"] != nil {
				t.Fatalf("a run naming no lane recorded an override: %v", placement["override"])
			}
		})
	}
}

// --bits and the caller's own choices are all a machine is sent, for a run and for a warm set
// member: the slot named with no lane reaches it with none, a named lane is pinned as named,
// and a slot with no choice reaches it with nothing (its machine loads the declared default).
// A named rental is never sized, so no model is read at any Hub.
func TestBitsAndOnlyTheCallersChoicesReachTheMachine(t *testing.T) {
	machine := &recordingMachine{runs: map[string]bool{}}
	root, store, h := attachedMachineHome(t, machine)
	servePrecisionRelease(t, h.mux)
	var mu sync.Mutex
	var read []string
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		read = append(read, r.Method+" "+r.URL.Path)
		mu.Unlock()
		served.ServeHTTP(w, r)
	})
	h.mux.HandleFunc("GET /v1/execution-environment", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"environment": map[string]string{
			"TENSORHUB_ORIGIN": h.server.URL, "TENSORHUB_PUBLIC_ORIGIN": h.server.URL}})
	})
	startCalciferProcess(t, root)

	code, out := runCozy(t, root, "run", "proof/describer/describe", "model.model=proof/vision@2.0.0",
		"model.decoder=proof/decoder@1.0.0/bf16", "--bits=4", "--rental=elder", "--json", "--idempotency-key", "bits-run")
	if code != 0 {
		t.Fatalf("the run was not submitted [exit %d]\n%s", code, out)
	}
	row, problem := store.RequestByIdempotencyKey("bits-run")
	fatal(t, problem)
	if row == nil || row.Bits != 4 {
		t.Fatalf("the run did not record its bits: %+v", row)
	}
	eventually(t, root, "the machine taking the run", func() bool { return machine.spec(v1.RunKind_RUN_KIND_CALL) != nil })
	call := machine.spec(v1.RunKind_RUN_KIND_CALL)
	choices := map[string]*v1.ModelChoice{}
	for _, choice := range call.GetModels() {
		choices[choice.GetParameter()] = choice
	}
	vision, decoder := choices["model"], choices["decoder"]
	if call.GetBits() != 4 || len(choices) != 2 || vision == nil || decoder == nil ||
		vision.GetRepository() != "proof/vision" || vision.GetRelease() != "2.0.0" || vision.GetLane() != "" || vision.GetManifest() != "" ||
		decoder.GetRelease() != "1.0.0" || decoder.GetLane() != "bf16" {
		t.Fatalf("the machine was sent bits %d and choices %v; want 4 and the two named, the vision lane its own to pick", call.GetBits(), call.GetModels())
	}

	if code, out := runCozy(t, root, "run", "proof/describer/describe", "--warm=gpu", "--bits=8", "--rental=elder", "--json"); code != 0 {
		t.Fatalf("warm [exit %d]\n%s", code, out)
	}
	eventually(t, root, "the machine taking the warm set", func() bool { return machine.spec(v1.RunKind_RUN_KIND_WARM) != nil })
	set := machine.spec(v1.RunKind_RUN_KIND_WARM).GetSet().GetItems()
	if len(set) != 1 || set[0].GetEntrypoint() != "describe" || set[0].GetBits() != 8 || len(set[0].GetModels()) != 0 {
		t.Fatalf("the warm member was sent as %v; want describe at 8 bits with no choice", set)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, request := range read {
		if strings.Contains(request, "/v1/models/") {
			t.Fatalf("a run on a named rental read a model at the Hub: %q", read)
		}
	}
}

// cl-180. On 2026-09-08 `cozy run paul/minimax-h3-tools/attention-lane
// --model.pruned=paul/minimax-h3@1.0.0-rc.2/fp8-adaln-pruned --rental-only` was sized
// "fit components 48.0 GiB (working memory unmeasured) of 48 GB": the job reads the source HEADER, inherits 3,858
// tensors by reference and writes 550 bytes, but the sizer held the card to the
// components a SERVING construction of that lane would stage. The 24 GB card was
// excluded, two rtx-6000-ada pods sat in `acquiring` for 22 and 24 minutes, and ~32 s of
// work cost 849 s of wall. A `Weights[M]` argument is read raw and never constructed, so
// it is never on the device: the cheap card is admissible and the record says `weights`
// rather than a figure.
func TestAWeightsArgumentIsNotSizedByTheComponentsItNeverStages(t *testing.T) {
	h := newStandInHub(t, h3Default)
	// The cheapest card is refused for stock, so the walk itself is the evidence: the
	// 24 GB product is asked for FIRST, which the component figure made impossible.
	h.soldOut["rtx-4090"] = true
	root := standInRoot(t, h)
	startCalciferProcess(t, root)
	_, out := runCozy(t, root, "run", h3Package+"/lane",
		"--model.pruned="+h3Model+"@"+h3Release+"/"+h3Lane,
		"--rental-only", "--json", "--idempotency-key", "weights-fit")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	queued, problem := store.RequestByIdempotencyKey("weights-fit")
	fatal(t, problem)
	if queued == nil {
		t.Fatalf("the by-reference call was not submitted: %s", out)
	}
	if len(queued.Models) != 1 || queued.Models[0].Manifest != fp8Manifest || !queued.Models[0].Weights ||
		queued.Models[0].ComponentBytes["text_encoder"] != textEncoderNeed {
		t.Fatalf("the call did not carry the sized lane it reads as Weights: %+v", queued.Models)
	}
	waitFor(t, root, "the walk passing the cheapest card", func() bool { return len(h.postedSKUs()) >= 2 })
	posted := h.postedSKUs()
	if !strings.HasPrefix(posted[0], "rtx-4090/") || !strings.HasPrefix(posted[1], "rtx-5090/") {
		t.Fatalf("paid asks %v; want the cheapest cards first for a call that stages nothing", posted)
	}
	log := tail(filepath.Join(root, "calcifer.log"))
	for _, want := range []string{"renting rtx-4090", "(lane fp8, fit weights)",
		"rtx-4090 has no inventory; choosing again without it", "renting rtx-5090"} {
		if !strings.Contains(log, want) {
			t.Fatalf("calcifer.log does not say %q:\n%s", want, log)
		}
	}
	if strings.Contains(log, "vram_short") {
		t.Fatalf("a Weights argument excluded a card for memory:\n%s", log)
	}
}

// Publishing writes the publisher's account into each org-less slot default, exactly as the
// published interface carries it; a default naming its org, in any case, is kept as written.
func TestPublishingWritesTheAccountIntoOrgLessDefaults(t *testing.T) {
	raw := []byte(`{"application":"h3:app","entrypoints":[{"name":"generate","models":[` +
		`{"class":"H3","component_use":{},"path":"generate.models.model","default":"minimax@1.0.0-rc.1","min_bits":4,"phase":"decode"},` +
		`{"class":"Vae","component_use":{},"path":"generate.models.vae","default":"Paul/Vae@2.0"}],` +
		`"request":{"fields":[]},"result":{"fields":[]}}],"format":"cozy.package.interface/1"}`)
	qualified, problem := packagepublish.QualifyInterface(raw, "fidika")
	fatal(t, problem)
	if _, err := canonical.NormalizeJCS(qualified); err != nil {
		t.Fatalf("the published interface is not canonical: %v", err)
	}
	published, problem := launch.DecodePackageInterface(qualified)
	fatal(t, problem)
	slots := published.Entrypoints[0].Models
	if slots[0].Default != "fidika/minimax@1.0.0-rc.1" || slots[0].MinBits != 4 || slots[0].Phase != "decode" ||
		slots[0].Bits() != 4 || slots[1].Default != "Paul/Vae@2.0" || slots[1].Bits() != 8 {
		t.Fatalf("published slots %+v", slots)
	}
	for _, invalid := range []string{"minimax@1.0.0/fp8", "minimax", "minimax@" + strings.Repeat("a", 64)} {
		decoded, problem := launch.DecodePackageInterface([]byte(strings.Replace(string(raw), "minimax@1.0.0-rc.1", invalid, 1)))
		fatal(t, problem)
		if decoded.Unavailable["generate"] == nil {
			t.Fatalf("default %q was admitted; a default names a release, never a lane or a digest", invalid)
		}
	}
}
