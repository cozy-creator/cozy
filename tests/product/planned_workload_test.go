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
	body, problem := hub.RentalRequestBytes("twine", "h200", strings.Repeat("ab", 32),
		"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", h3SourceBytes)
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
	serving, problem := hub.RentalRequestBytes("twine", "h200", strings.Repeat("ab", 32),
		"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 0)
	fatal(t, problem)
	if strings.Contains(string(serving), "planned_source_bytes") {
		t.Fatalf("an undeclared serving rental put planned_source_bytes on the wire: %s", serving)
	}
}
