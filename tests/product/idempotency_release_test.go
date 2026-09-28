package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
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
		ID: "req-idem-1", Kind: "job", Package: "paul/minimax-h3-tools", Entrypoint: "four-lane",
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

	second, fresh2, e := st.Submit(records.Request{
		ID: "req-idem-2", Kind: "job", Package: "paul/minimax-h3-tools", Entrypoint: "four-lane",
		Org: "paul", IdemKey: "model-transfer-test", Payload: []byte("{}"),
	})
	fatal(t, e)
	if !fresh2 || second.ID == recorded.ID {
		t.Fatal("released key must admit a fresh run")
	}
	fatal(t, st.SettleRequest(second.ID, "failed"))
	released, e = st.ReleaseCanceledIdempotencyKey("model-transfer-test")
	fatal(t, e)
	if !released {
		t.Fatal("a failed run's key must release once its cause is fixable")
	}
	fatal(t, func() *exit.Error {
		_, _, e := st.Submit(records.Request{
			Kind: "job", Package: "paul/minimax-h3-tools", Entrypoint: "four-lane",
			Org: "paul", ID: "req-idem-3", IdemKey: "keep-succeeded", Payload: []byte("{}")})
		return e
	}())
	if got, e := st.RequestByIdempotencyKey("keep-succeeded"); e != nil || got == nil {
		t.Fatal("live key still replays")
	}
	rows, e := st.RequestByIdempotencyKey("keep-succeeded")
	fatal(t, e)
	fatal(t, st.SettleRequest(rows.ID, "succeeded"))
	released, e = st.ReleaseCanceledIdempotencyKey("keep-succeeded")
	fatal(t, e)
	if released {
		t.Fatal("a succeeded run's key must never release")
	}
}
