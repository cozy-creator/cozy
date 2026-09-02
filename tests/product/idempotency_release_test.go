package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestCanceledRunReleasesItsIdempotencyKey: a cancellation withdraws the
// intent; an identical resubmission must get a fresh run, while every other
// state keeps replaying (the live incident: the H3 four-lane upload's durable
// run was canceled to unwedge a dead daemon's queue, and the identical
// resubmission replayed the cancellation forever).
func TestCanceledRunReleasesItsIdempotencyKey(t *testing.T) {
	l, e := home.Open(t.TempDir())
	fatal(t, e)
	st, e := records.Open(l.DB)
	fatal(t, e)
	defer st.Close()

	recorded, fresh, e := st.Submit(records.Request{
		Kind: "job", Package: "paul/minimax-h3-tools", Entrypoint: "four-lane",
		Org: "paul", IdemKey: "model-transfer-test", Payload: []byte("{}"),
	})
	fatal(t, e)
	if !fresh {
		t.Fatal("first submission must be fresh")
	}

	existing, e := st.RequestByIdempotencyKey("model-transfer-test")
	fatal(t, e)
	if existing == nil || existing.ID != recorded.ID {
		t.Fatal("key must replay while the run is live")
	}
	released, e := st.ReleaseCanceledIdempotencyKey("model-transfer-test")
	fatal(t, e)
	if released {
		t.Fatal("a live run's key must not release")
	}

	fatal(t, st.SettleRequest(recorded.ID, "canceled"))
	released, e = st.ReleaseCanceledIdempotencyKey("model-transfer-test")
	fatal(t, e)
	if !released {
		t.Fatal("a canceled run's key must release")
	}
	existing, e = st.RequestByIdempotencyKey("model-transfer-test")
	fatal(t, e)
	if existing != nil {
		t.Fatal("released key must read as absent")
	}
	row, e := st.RequestRow(recorded.ID)
	fatal(t, e)
	if row == nil || row.State != "canceled" {
		t.Fatalf("history must survive under the derived key: %+v", row)
	}
}
