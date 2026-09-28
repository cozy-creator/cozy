package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Exercise the real durable transition's wake decision, including the stale
// failure rule: keep the verdict, but never roll byte/revision counters back.
func TestWeightsStatusCommittedStateChange(t *testing.T) {
	store, db, id := publicationRetryFixture(t)
	rows, problem := store.AllModelTransferWeights(id, 1)
	fatal(t, problem)
	object := rows[0].Objects[0]
	_, err := db.Exec(`UPDATE request_model_transfer_objects SET operation_id='',grant_revision=0,update_sequence=0,state='pending',transferred=0,safe_code='',safe_detail='' WHERE request_id=?`, id)
	must(t, err)
	object.OperationID, object.State = "proof", "accepted"
	object.GrantRevision, object.UpdateSequence, object.Transferred = 1, 1, 0
	cases := []struct {
		name    string
		mutate  func(*records.ModelTransferObject)
		changed bool
		code    string
	}{
		{"accepted", func(*records.ModelTransferObject) {}, true, ""},
		{"exact replay", func(*records.ModelTransferObject) {}, false, ""},
		{"same state progress", func(o *records.ModelTransferObject) { o.UpdateSequence++; o.Transferred++ }, false, ""},
		{"stale sequence", func(o *records.ModelTransferObject) { o.UpdateSequence-- }, false, "model_transfer.object_status_superseded"},
		{"new grant same state", func(o *records.ModelTransferObject) { o.GrantRevision = 4; o.UpdateSequence = 1 }, false, ""},
		{"stale failure retains verdict", func(o *records.ModelTransferObject) {
			o.State = "failed"
			o.GrantRevision = 1
			o.Transferred = 0
			o.SafeCode = "origin_refused"
		}, true, ""},
		{"same grant cannot undo failure", func(o *records.ModelTransferObject) { o.State = "accepted"; o.GrantRevision = 4; o.UpdateSequence = 2 }, false, "model_transfer.object_status_superseded"},
		{"new grant retries failure", func(o *records.ModelTransferObject) {
			o.State = "accepted"
			o.GrantRevision = 5
			o.UpdateSequence = 1
			o.SafeCode = ""
		}, true, ""},
		{"held", func(o *records.ModelTransferObject) { o.State = "held"; o.UpdateSequence = 2; o.Transferred = o.Length }, true, ""},
		{"held replay", func(*records.ModelTransferObject) {}, false, ""},
		{"held absorbs newer failure", func(o *records.ModelTransferObject) { o.State = "failed"; o.GrantRevision++ }, false, "model_transfer.object_status_superseded"},
		{"changed identity", func(o *records.ModelTransferObject) { o.Length++ }, false, "model_transfer.object_status_changed"},
		{"unknown object", func(o *records.ModelTransferObject) { o.ObjectID = "absent" }, false, "*"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			next := object
			tc.mutate(&next)
			changed, problem := store.RecordModelTransferObjectStatus(next)
			code := ""
			if problem != nil {
				code = problem.ErrName()
			}
			if changed != tc.changed || tc.code == "*" && problem == nil || tc.code != "*" && code != tc.code {
				t.Fatalf("changed=%t code=%q; want %t/%q", changed, code, tc.changed, tc.code)
			}
			if problem == nil {
				rows, problem := store.ModelTransferObjects(id, object.Attempt, object.OutputSlot)
				fatal(t, problem)
				for _, row := range rows {
					if row.ObjectID == object.ObjectID {
						object = row
					}
				}
				if tc.name == "stale failure retains verdict" && (object.GrantRevision != 4 || object.Transferred != 1 || object.SafeCode != "origin_refused") {
					t.Fatalf("stale verdict changed counters: %+v", object)
				}
			}
		})
	}
}
