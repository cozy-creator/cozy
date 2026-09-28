package producttest

// th-152: the ingest half of the container-disk train. A rental bought FOR a
// model transfer must declare that transfer's source bytes, so the hub can size
// the pod's container disk to the job.
//
// Why this matters more than a billing rounding error: an ingest holds its
// source objects AND the canonical CAS output built from them in one TensorFS
// Store on the pod's CONTAINER DISK (/var/lib/tensorfs, which the pod
// supervisor refuses to relocate onto the /workspace volume). The H3 mirror is
// 210.3 GB of source, so that pod peaks near 442 GB. Bought at the 200 GB
// serving default it does not error — a full filesystem BLOCKS writes, which is
// indistinguishable from a stall, hours into a paid run.
//
// The arm asserts on the exact bytes that go on the wire, because the sum being
// available in the store proves nothing about what was sent.

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// h3SourceBytes is the measured full both-partition MiniMax-H3 mirror.
const h3SourceBytes int64 = 210_331_470_098

func submitH3Transfer(t *testing.T, store *records.Store, requestID string) {
	t.Helper()
	// Two members that sum to the measured mirror, sorted by member name the
	// way the recorder requires.
	intent := &records.ModelTransferIntent{
		Kind: "model-upload", Destination: "paul/minimax-h3",
		Source:          "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("4", 40),
		SourceSelection: "sha256:" + strings.Repeat("2", 64),
		SourceFiles: []records.ModelTransferSourceFile{
			{Member: "a.safetensors", SHA256: strings.Repeat("a", 64), Length: 110_331_470_098},
			{Member: "b.safetensors", SHA256: strings.Repeat("b", 64), Length: 100_000_000_000},
		},
		Outputs: []records.ModelTransferOutput{{Name: "full"}},
	}
	_, created, problem := store.Submit(records.Request{
		ID: requestID, IdemKey: requestID, BodyDigest: "sha256:" + strings.Repeat("c", 64),
		Package: "paul/minimax-h3-tools", Entrypoint: "four-lane", State: "queued",
		Kind: "job", Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]",
		ModelTransfer: intent,
	})
	fatal(t, problem)
	if !created {
		t.Fatal("the H3 transfer request was not created; the arm proves nothing")
	}
}

func TestPlannedSourceBytesTotalsTheDeclaredIngest(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	submitH3Transfer(t, store, "req-h3")
	total, problem := store.PlannedSourceBytes("req-h3")
	fatal(t, problem)
	if total != h3SourceBytes {
		t.Fatalf("the declared H3 ingest totals %d bytes, want %d", total, h3SourceBytes)
	}
	// A request that moves no model bytes declares nothing, so an ordinary
	// serving rental keeps the image's default.
	none, problem := store.PlannedSourceBytes("req-absent")
	fatal(t, problem)
	if none != 0 {
		t.Fatalf("a request with no model transfer totalled %d bytes, want 0", none)
	}
}

// The wire arm: the declared workload must survive into the exact canonical
// bytes the hub is sent and replays.
func TestRentalRequestCarriesTheDeclaredWorkload(t *testing.T) {
	body, problem := hub.RentalRequestBytes("twine", "h200", 1, strings.Repeat("ab", 32),
		"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
		hub.DeclaredWorkload{SourceBytes: h3SourceBytes}, nil, "")
	fatal(t, problem)

	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	got, ok := wire["planned_source_bytes"]
	if !ok {
		t.Fatal("the rental request the hub receives carries no planned_source_bytes; " +
			"the hub would size this ingest pod at the serving default and it would wedge")
	}
	if int64(got.(float64)) != h3SourceBytes {
		t.Fatalf("the rental request declares %v source bytes, want %d", got, h3SourceBytes)
	}

	// It must survive the canonical replay path unchanged, or the paid intent
	// and the disk it was sized for could drift apart.
	reopened, problem := hub.ParseRentalRequestBytes(body)
	fatal(t, problem)
	if reopened.PlannedSourceBytes != h3SourceBytes {
		t.Fatalf("the replayed paid intent declares %d source bytes, want %d",
			reopened.PlannedSourceBytes, h3SourceBytes)
	}

	// A serving rental declares nothing, and the field must stay OFF the wire so
	// an undeclared rental is byte-identical to one authored before th-152.
	serving, problem := hub.RentalRequestBytes("twine", "h200", 1, strings.Repeat("ab", 32),
		"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", hub.DeclaredWorkload{}, nil, "")
	fatal(t, problem)
	if strings.Contains(string(serving), "planned_source_bytes") {
		t.Fatalf("an undeclared serving rental put planned_source_bytes on the wire: %s", serving)
	}
}

// ─── cl-130: the serving half ────────────────────────────────────────────────

// h3ServingModels is what a request bought to SERVE H3 resolves to before any
// pod exists: two slots of one model, plus one release-less operation-local ref
// that is NOT a published model and must never reach the wire.
func h3ServingModels() []records.ModelRef {
	return []records.ModelRef{
		{Package: "paul/h3-video", Slot: "dit", Model: "paul/minimax-h3", Release: "1.0.0",
			Lane: "bf16", Manifest: "sha256:" + strings.Repeat("1", 64)},
		{Package: "paul/h3-video", Slot: "vae", Model: "paul/minimax-h3-vae", Release: "1.0.0",
			Lane: "bf16", Manifest: "sha256:" + strings.Repeat("2", 64)},
		{Package: "paul/h3-video", Slot: "scratch", Model: "local/step-output",
			Manifest: "sha256:" + strings.Repeat("3", 64)},
	}
}

// The store half: a request's resolved models are available to the rental
// BEFORE the pod is bought, and an operation-local ref is dropped.
func TestDeclaredServingModelsReadsTheResolvedSelection(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	_, created, problem := store.Submit(records.Request{
		ID: "req-serve", IdemKey: "req-serve", BodyDigest: "sha256:" + strings.Repeat("c", 64),
		Package: "paul/h3-video", Entrypoint: "generate", State: "queued",
		Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]",
		Release: "1.0.0", Models: h3ServingModels(),
	})
	fatal(t, problem)
	if !created {
		t.Fatal("the serving request was not created; the arm proves nothing")
	}
	declared, problem := store.DeclaredServingModels("req-serve")
	fatal(t, problem)
	if len(declared) != 2 {
		t.Fatalf("the request declared %d serving models, want 2 — the release-less "+
			"operation-local ref must be dropped and the two published ones kept: %+v",
			len(declared), declared)
	}
	for _, model := range declared {
		if model.Release == "" || model.Manifest == "" {
			t.Fatalf("a declared serving model is missing its pins: %+v", model)
		}
		if model.Model == "local/step-output" {
			t.Fatal("an operation-local manifest reached the declared serving set; the hub " +
				"cannot resolve it and would refuse the whole rental")
		}
	}
	// A request that names no models declares none, so an unmanaged rental
	// stays byte-identical.
	none, problem := store.DeclaredServingModels("req-absent")
	fatal(t, problem)
	if len(none) != 0 {
		t.Fatalf("an absent request declared %d serving models, want 0", len(none))
	}
}

// The wire arm: the declared serving set must survive into the exact canonical
// bytes the hub is sent and replays. The store holding the answer proves nothing
// about what was sent — that gap is the whole reason th-155 exists.
func TestRentalRequestCarriesTheDeclaredServingSet(t *testing.T) {
	declared := []hub.ServingModel{
		{Lane: "bf16", Manifest: "sha256:" + strings.Repeat("2", 64),
			Model: "paul/minimax-h3-vae", Release: "1.0.0"},
		{Lane: "bf16", Manifest: "sha256:" + strings.Repeat("1", 64),
			Model: "paul/minimax-h3", Release: "1.0.0"},
	}
	body, problem := hub.RentalRequestBytes("twine", "h200", 1, strings.Repeat("ab", 32),
		"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
		hub.DeclaredWorkload{ServingModels: declared}, nil, "")
	fatal(t, problem)

	var wire struct {
		ServingModels []hub.ServingModel `json:"serving_models"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.ServingModels) != 2 {
		t.Fatalf("the rental request the hub receives carries %d serving models, want 2; "+
			"the hub would size this pod at the serving default, which holds the H3 serve "+
			"set with 35 GB to spare and no room for a second model", len(wire.ServingModels))
	}
	// SORTED, so the same selection in any order authors the same bytes and a
	// replay cannot re-author a different request digest.
	if wire.ServingModels[0].Model != "paul/minimax-h3" ||
		wire.ServingModels[1].Model != "paul/minimax-h3-vae" {
		t.Fatalf("the declared serving set is not canonically ordered: %+v", wire.ServingModels)
	}
	// Every pin must survive: the hub resolves model@release@manifest and a lost
	// pin is a lane it has to guess between, which it refuses.
	if wire.ServingModels[0].Manifest != "sha256:"+strings.Repeat("1", 64) ||
		wire.ServingModels[0].Lane != "bf16" || wire.ServingModels[0].Release != "1.0.0" {
		t.Fatalf("a declared serving model lost a pin on the wire: %+v", wire.ServingModels[0])
	}

	reopened, problem := hub.ParseRentalRequestBytes(body)
	fatal(t, problem)
	if len(reopened.ServingModels) != 2 ||
		reopened.ServingModels[0].Model != "paul/minimax-h3" {
		t.Fatalf("the replayed paid intent declares %+v", reopened.ServingModels)
	}

	// Undeclared stays OFF the wire, so deploying this cannot change the request
	// digest of a rental authored before th-155 and invalidate its idempotency key.
	serving, problem := hub.RentalRequestBytes("twine", "h200", 1, strings.Repeat("ab", 32),
		"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", hub.DeclaredWorkload{}, nil, "")
	fatal(t, problem)
	if strings.Contains(string(serving), "serving_models") {
		t.Fatalf("an undeclared rental put serving_models on the wire: %s", serving)
	}
}

// A serving model missing a pin is refused at authoring time rather than sent
// half-stated: the hub cannot resolve it and would refuse the whole rental, so
// the useful place to say so is here, before anything is persisted.
func TestAnIncompleteServingModelIsRefusedBeforeItIsSent(t *testing.T) {
	for _, arm := range []struct {
		what  string
		model hub.ServingModel
	}{
		{"no manifest", hub.ServingModel{Lane: "bf16", Model: "paul/m", Release: "1.0.0"}},
		{"lane without release", hub.ServingModel{Lane: "bf16", Model: "paul/m",
			Manifest: "sha256:" + strings.Repeat("1", 64)}},
		{"no model", hub.ServingModel{Lane: "bf16", Release: "1.0.0",
			Manifest: "sha256:" + strings.Repeat("1", 64)}},
	} {
		_, problem := hub.RentalRequestBytes("twine", "h200", 1, strings.Repeat("ab", 32),
			"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
			hub.DeclaredWorkload{ServingModels: []hub.ServingModel{arm.model}}, nil, "")
		if problem == nil {
			t.Fatalf("a serving model with %s was authored onto the wire", arm.what)
		}
	}
}
