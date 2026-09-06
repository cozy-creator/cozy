package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func outputPublicationSubmission() orchestrator.Submission {
	return orchestrator.Submission{
		IdemKey: "output-publication", Package: "cozy/h3-package", Entrypoint: "four-lane",
		PlanID: "sha256:" + strings.Repeat("35", 32), Release: "1.0.7",
		Kind: "job", Org: "paul", Payload: []byte(`{"steps":4}`), Outputs: []string{"model"},
		WeightsOutputs: []orchestrator.WeightsOutput{{OutputID: "model",
			MimeType: orchestrator.WeightsManifestMime, MaxBytes: 1 << 30}},
		Worker: podRental, Rental: true, RentalRequired: true, ProducerParams: []string{"dits", "shared"},
		Models: []orchestrator.ModelRef{
			{Package: "cozy/h3-package", Slot: "dits", Model: "source/dits", Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164},
			{Package: "cozy/h3-package", Slot: "shared", Model: "source/shared", Manifest: "sha256:" + strings.Repeat("2", 64), ManifestLength: 164},
		},
		ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: "paul/output-publication",
			Outputs: []records.ModelTransferOutput{{Name: "model"}}},
	}
}

// The real owner is wired to its production I/O implementation. An accidental
// source-acquisition call would refuse the absent source instead of producing an offer.
func TestOutputPublicationKeepsSelectedInputs(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "output-publication-inputs", rentalWiring(connection, private),
		func(options *orchestrator.Options) {
			options.ModelTransfers = cli.NewModelTransferOwner(options.Cfg, options.Store, options.Log, nil)
		})
	sub := outputPublicationSubmission()
	fatal(t, modeltransfer.ValidateSubmission(sub))
	fatal(t, records.NormalizeModelTransferIntent(sub.ModelTransfer))
	requestID, _, problem := o.c.Submit(sub)
	fatal(t, problem)
	waitUntil(t, "output publication offer without source acquisition", func() bool {
		request, problem := o.store.RequestRow(requestID)
		fatal(t, problem)
		if request.State == "failed" {
			t.Fatalf("output-only job failed before its offer: %v", o.c.Events())
		}
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) == 1
	})
	request, problem := o.store.RequestRow(requestID)
	fatal(t, problem)
	if len(request.Models) != 2 || request.Models[0].Manifest != sub.Models[0].Manifest ||
		request.Models[1].Manifest != sub.Models[1].Manifest {
		t.Fatalf("publication changed the selected generic inputs: %+v", request.Models)
	}
	transfer, problem := o.store.ModelTransferOf(requestID)
	fatal(t, problem)
	if transfer.State != "pending" || len(transfer.Models) != 0 || transfer.HasAcquisition() {
		t.Fatalf("publication invented materialized source custody: %+v", transfer)
	}
}

func TestOutputPublicationRefusesPartialAcquisitionAndUnsizedRental(t *testing.T) {
	for name, change := range map[string]func(*records.ModelTransferIntent){
		"source":    func(i *records.ModelTransferIntent) { i.Source = "hf://example/model" },
		"selection": func(i *records.ModelTransferIntent) { i.SourceSelection = "sha256:" + strings.Repeat("1", 64) },
		"license":   func(i *records.ModelTransferIntent) { i.SourceLicense = "MIT" },
		"lane":      func(i *records.ModelTransferIntent) { i.InputLane = "bf16" },
		"profiles":  func(i *records.ModelTransferIntent) { i.SourceProfiles = map[string]string{"dits": "example/1"} },
		"files": func(i *records.ModelTransferIntent) {
			i.SourceFiles = []records.ModelTransferSourceFile{{Member: "source"}}
		},
		"local authority": func(i *records.ModelTransferIntent) { i.LocalOnly = true },
	} {
		t.Run(name, func(t *testing.T) {
			intent := outputPublicationSubmission().ModelTransfer
			change(intent)
			if problem := records.NormalizeModelTransferIntent(intent); problem == nil {
				t.Fatal("partial acquisition was accepted as output-only publication")
			}
		})
	}
	sub := outputPublicationSubmission()
	sub.Worker = ""
	// Neither small Manifest metadata nor an unverified caller byte count sizes
	// a paid rental for this operation's input closure and output working set.
	sub.Models[0].Bytes = 210_000_000_000
	if problem := modeltransfer.ValidateSubmission(sub); problem == nil ||
		problem.ErrName() != "model_transfer.prepared_rental_required" {
		t.Fatalf("unplanned rental was not refused: %v", problem)
	}
	sub = outputPublicationSubmission()
	sub.ModelTransfer.Outputs[0].Name = "undeclared"
	if problem := modeltransfer.ValidateSubmission(sub); problem == nil {
		t.Fatal("publication accepted an output absent from the job interface")
	}
}
