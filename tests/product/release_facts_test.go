package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const releaseFactsInterface = `{"application":"h3:app","entrypoints":[{"name":"generate","models":[{"class":"H3","component_use":{"condition_text":["text_encoder"],"sample_fl2va":["fl2va_dit"]},"path":"generate.models.model"}],"request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[{"name":"lane","models":[{"class":"Source","component_use":{},"path":"lane.models.pruned"}],"publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[{"max_bytes":65536,"mime_type":"application/vnd.cozy.model-manifest","output_id":"attn8"}]}]}`

// releaseFactsHub answers the two routes the real PrepareFactsSource reads: the
// rental-scoped facts and the public release record carrying the interface.
func releaseFactsHub(t *testing.T, application string) (*httptest.Server, []byte) {
	t.Helper()
	normalized, err := canonical.NormalizeJCS([]byte(releaseFactsInterface))
	must(t, err)
	digest, err := canonical.Spell(canonical.Digest(normalized))
	must(t, err)
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = []byte(releaseFactsInterface)
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = digest
	detail.Release.PackageInterfaceLength = int64(len(normalized))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.18.21"}
	facts := testPrepareFacts("proof/h3", "1.0.0")
	inventory, err := json.Marshal(map[string]any{"format": "tensorhub.image_inventory/1",
		"profile": facts.ImageInventory.Profile, "python": facts.ImageInventory.Python,
		"distributions": []map[string]string{{"name": "numpy", "version": "2.1.0"}}})
	must(t, err)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/rentals/"+podRental+"/prepare-facts", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("package") != "proof/h3" || r.URL.Query().Get("release") != "1.0.0" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(hub.PrepareFactsView{Application: application,
			ModelSlotPaths: []string{"generate.models.model", "lane.models.pruned"},
			ImageInventory: inventory, LockedRequirements: string(facts.LockedRequirements)})
	})
	mux.HandleFunc("GET /v1/packages/proof/h3/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(detail)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server, normalized
}

// The owner's PodHost prepare carries the hub release's exact PackageInterface bytes
// (wire 61 field 10), fetched by the production facts source. Agreement between the
// interface and the assembled facts is the preparing Runtime's check, not Creator's:
// a fact the hub spells differently still reaches the pod that owns the decision.
func TestPrepareFactsCarryReleaseInterfaceToPodHost(t *testing.T) {
	for _, application := range []string{"h3:app", "other:app"} {
		t.Run(application, func(t *testing.T) {
			server, normalized := releaseFactsHub(t, application)
			client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("fixture")}, "")
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "release-facts-"+strings.ReplaceAll(application, ":", "-"), rentalWiring(connection, private),
				func(options *orchestrator.Options) { options.RentalPrepareFacts = rental.PrepareFactsSource(client) })
			instance, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: "proof/h3", Release: "1.0.0"}}, nil))
			waitUntil(t, "the PodHost prepare", func() bool {
				pod.mu.Lock()
				defer pod.mu.Unlock()
				return len(pod.prepares) > 0
			})
			pod.mu.Lock()
			defer pod.mu.Unlock()
			if got := pod.prepares[0].PackageInterface; !bytes.Equal(got, normalized) {
				t.Fatalf("PreparePackageSetCall.package_interface = %q, want the release's canonical bytes %q", got, normalized)
			}
		})
	}
}

func TestWorkingMemoryLedgerSizesTheNextSelection(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	offered := make(chan func(*pb.WorkerFrame) error, 1)
	offers := make(chan *pb.AttemptOffer, 1)
	acked := make(chan struct{}, 4)
	pod := &fakePod{controlKey: public, serve: true, slots: 8}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if offer := frame.GetAttemptOffer(); offer != nil {
			err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: &pb.AttemptAccepted{
				RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
				WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
				InvocationSpecDigest: offer.InvocationSpecDigest, PlacementId: offer.PlacementId,
			}}})
			offers <- offer
			offered <- send
			return true, err
		}
		if frame.GetOutcomeAck() != nil {
			acked <- struct{}{}
			return true, nil
		}
		return false, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "working-memory-ledger", rentalWiring(connection, private))
	fatal(t, o.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "otter", State: "ready",
		SKU: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3", HourlyRateUSDMicros: 100_000,
		Address: connection.Addr, CertPath: connection.CACert, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	const pkg = "proof/working"
	run := func(key string, status pb.OutcomeStatus, metrics *pb.AttemptMetrics) records.Request {
		t.Helper()
		id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: key, Package: pkg, Release: "1.0.0",
			Entrypoint: "tile", PlanID: podPlanID(pkg), Payload: []byte(`{}`), Worker: podRental,
			Rental: true, RentalRequired: true})
		fatal(t, problem)
		var offer *pb.AttemptOffer
		var send func(*pb.WorkerFrame) error
		select {
		case offer = <-offers:
			send = <-offered
		case <-time.After(10 * time.Second):
			t.Fatal("worker did not receive the request")
		}
		cause, origin := pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME
		if status == pb.OutcomeStatus_OUTCOME_STATUS_FAILED {
			cause, origin = pb.CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION, pb.CauseOrigin_CAUSE_ORIGIN_AUTHOR
		}
		frame := privateAttemptOutcome(offer, status, cause, origin)
		outcome := frame.GetAttemptOutcome()
		var body pb.AttemptOutcomeBody
		must(t, canonical.Unmarshal(outcome.OutcomeCanonicalBytes, &body))
		body.Metrics = metrics
		outcome.OutcomeCanonicalBytes, outcome.OutcomeDigest, err = canonical.Identity(&body)
		must(t, err)
		must(t, send(frame))
		select {
		case <-acked:
		case <-time.After(10 * time.Second):
			t.Fatal("outcome was not acknowledged")
		}
		req, problem := o.store.RequestRow(id)
		fatal(t, problem)
		return *req
	}
	measured := &pb.AttemptMetrics{RuntimeMs: 1, WorkingPeakDeviceBytes: 30 << 30, ShapeCell: "768x1344x121"}
	req := run("measured", pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, measured)
	run("failed", pb.OutcomeStatus_OUTCOME_STATUS_FAILED, &pb.AttemptMetrics{WorkingPeakDeviceBytes: 70 << 30})
	run("pre-61", pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, &pb.AttemptMetrics{RuntimeMs: 1})
	run("smaller", pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, &pb.AttemptMetrics{WorkingPeakDeviceBytes: 20 << 30, ShapeCell: "768x800x121"})

	peaks, problem := o.store.WorkingPeaks(req)
	fatal(t, problem)
	want := records.WorkingPeak{Bytes: 30 << 30, Runs: 2}
	if len(peaks) != 1 || peaks[records.ModelsDigest(req.Models)] != want {
		t.Fatalf("ledger = %+v, want the two succeeded measured runs at their max %+v", peaks, want)
	}
	// A measured total device peak for the same published request, SKU and width sizes
	// that exact selection (Runtime #685's measured-total path).
	run("total", pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, &pb.AttemptMetrics{RuntimeMs: 1, WorkingPeakDeviceBytes: 30 << 30, PeakDeviceMemoryBytes: 44 << 30})
	totals, problem := o.store.WorkingPeaks(req)
	fatal(t, problem)
	if exact := totals.For(req.Models, "h100-80", 1); exact.TotalBytes != 44<<30 || exact.TotalRuns != 1 {
		t.Fatalf("the published request's measured total was not recorded for its SKU and width: %+v", exact)
	}
	if other := totals.For(req.Models, "rtx-4090", 1); other.TotalBytes != 0 {
		t.Fatalf("another SKU inherited the measured total: %+v", other)
	}
	other := req
	other.Entrypoint = "other"
	if peaks, problem := o.store.WorkingPeaks(other); problem != nil || len(peaks) != 0 {
		t.Fatalf("another entrypoint inherited the measurement: %+v %v", peaks, problem)
	}

	skus := []hub.RentalSKU{{Name: "rtx-4090", AcceleratorModel: "NVIDIA GeForce RTX 4090", AcceleratorCount: 1, VRAMGB: 24},
		{Name: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3", AcceleratorCount: 1, VRAMGB: 80}}
	candidates := rental.Purchases(skus, req.Models, true, false, rental.Constraints{Working: peaks})
	if !strings.HasPrefix(candidates[0].Verdict, orchestrator.VerdictExcluded+orchestrator.ExcludedVRAMShort) ||
		!strings.Contains(candidates[0].Verdict, "measured working 30.0 GiB") {
		t.Fatalf("rtx-4090 verdict %q; want vram_short by the measured working peak", candidates[0].Verdict)
	}
	if candidates[1].Verdict != "" || candidates[1].Fit != "measured working 30.0 GiB (2 runs) of 80 GB" {
		t.Fatalf("h100-80 verdict %q fit %q", candidates[1].Verdict, candidates[1].Fit)
	}

	// A staged selection is sized by its declared residency and says its total is
	// unmeasured. A legacy working peak has no co-resident scope, so it cannot enlarge
	// a staged execution.
	models := []records.ModelRef{{Slot: "generate.models.model", Model: "proof/minimax", Release: "1.0.0",
		Lane: "fp8", Manifest: "sha256:" + strings.Repeat("ab", 32),
		ComponentBytes: map[string]int64{"text_encoder": 51 << 30, "fl2va_dit": 21 << 30},
		ComponentUse:   map[string][]string{"condition_text": {"text_encoder"}, "sample_fl2va": {"fl2va_dit"}}}}
	working := records.WorkingPeaks{records.ModelsDigest(models): {Bytes: 30 << 30, Runs: 3}}
	for _, peaks := range []records.WorkingPeaks{nil, working} {
		sized := rental.Purchases(skus[1:], models, true, false, rental.Constraints{Working: peaks})
		if sized[0].Fit != "components 51.0 GiB (total memory unmeasured for this exact workload/SKU) of 80 GB" || sized[0].Verdict != "" {
			t.Fatalf("staged fit %q verdict %q", sized[0].Fit, sized[0].Verdict)
		}
	}

	// Unstaged weights and working memory add: the fit names both, and an unmeasured
	// selection says so rather than reading as "needs nothing more".
	lane := []records.ModelRef{{Slot: "generate.models.model", Model: "proof/minimax", Release: "1.0.0",
		Lane: "fp8", Manifest: "sha256:" + strings.Repeat("ab", 32), Bytes: 51 << 30}}
	sized := rental.Purchases(skus[1:], lane, true, false, rental.Constraints{})
	if sized[0].Fit != "lane_bytes 51.0 GiB (working memory unmeasured) of 80 GB" || sized[0].Verdict != "" {
		t.Fatalf("unmeasured fit %q verdict %q", sized[0].Fit, sized[0].Verdict)
	}
	sized = rental.Purchases(skus[1:], lane, true, false, rental.Constraints{Working: working})
	if !strings.Contains(sized[0].Verdict, "needs 81.0 GiB resident (lane fp8; measured working 30.0 GiB)") {
		t.Fatalf("measured verdict %q fit %q", sized[0].Verdict, sized[0].Fit)
	}
	if sized[0].Fit != "lane_bytes 51.0 GiB + measured working 30.0 GiB (3 runs) of 80 GB" {
		t.Fatalf("measured fit %q", sized[0].Fit)
	}
}

// `cozy rental update` reads the pinned PodHost's protocol range over the real TLS
// probe and refuses a Runtime whose declared wire minimum that host does not reach. A
// Runtime that declares no readable range is attempted; each operation gates itself.
func TestRuntimeUpdateRefusesRuntimeAheadOfHost(t *testing.T) {
	for _, test := range []struct {
		host   uint32
		target *rental.RuntimeWire
		code   string
	}{
		{60, &rental.RuntimeWire{WireMinor: 61, MinimumWireMinor: 61}, "rental.runtime_update_host_too_old"},
		{60, &rental.RuntimeWire{WireMinor: 61, MinimumWireMinor: 60}, ""},
		{61, &rental.RuntimeWire{WireMinor: 61, MinimumWireMinor: 61}, ""},
		{61, &rental.RuntimeWire{WireMinor: 62, MinimumWireMinor: 62}, "rental.runtime_update_host_too_old"},
		{61, nil, ""},
		{61, &rental.RuntimeWire{WireMinor: 60, MinimumWireMinor: 61}, ""},
	} {
		public, _, err := ed25519.GenerateKey(rand.Reader)
		must(t, err)
		connection, _ := startFakePod(t, t.TempDir(), &fakePod{controlKey: public, wireMinor: test.host})
		info, problem := orchestrator.RentalProtocolInfo(context.Background(), connection)
		fatal(t, problem)
		problem = rental.RuntimeUpdateHost("0.18.99", test.target, info.WireMinor)
		got := ""
		if problem != nil {
			got = problem.ErrName()
		}
		if got != test.code {
			t.Fatalf("host %d target %+v: %v, want %q", test.host, test.target, problem, test.code)
		}
	}
}
