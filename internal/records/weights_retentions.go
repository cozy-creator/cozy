package records

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/canonical"
)

func ArtifactRetentionID(request, kind, slot string, artifact ModelArtifact) string {
	raw, _ := json.Marshal(map[string]any{"request_id": request, "kind": kind, "slot": slot, "producer_request_id": artifact.ProducerRequestID, "output_slot": artifact.OutputSlot, "tensorfs_receipt_digest": artifact.TensorFSReceiptDigest})
	raw, _ = canonical.NormalizeJCS(raw)
	id, _ := canonical.Spell(canonical.Digest(raw))
	return id
}

const weightsRetentionsDDL = `
CREATE TABLE IF NOT EXISTS request_weights_retentions (
 request_id TEXT NOT NULL REFERENCES requests(id),
 kind TEXT NOT NULL CHECK(kind IN ('input','result')),
 slot TEXT NOT NULL,
 producer_request_id TEXT NOT NULL,
 producer_attempt INTEGER NOT NULL,
 producer_output_slot TEXT NOT NULL,
 retention_id TEXT NOT NULL UNIQUE,
 instance_id TEXT NOT NULL DEFAULT '',
 worker_boot_id TEXT NOT NULL DEFAULT '',
 state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','held','releasing','released')),
 PRIMARY KEY(request_id,kind,slot),
 FOREIGN KEY(producer_request_id,producer_attempt,producer_output_slot) REFERENCES request_model_transfer_outputs(request_id,attempt,output_slot)
)`

type WeightsRetention struct {
	RequestID          string
	Kind               string
	Slot               string
	ProducerRequestID  string
	ProducerAttempt    int64
	ProducerOutputSlot string
	RetentionID        string
	InstanceID         string
	WorkerBootID       string
	State              string
}

const weightsRetentionCols = `request_id,kind,slot,producer_request_id,producer_attempt,producer_output_slot,retention_id,instance_id,worker_boot_id,state`

func scanWeightsRetention(row interface{ Scan(...any) error }) (WeightsRetention, error) {
	var r WeightsRetention
	err := row.Scan(&r.RequestID, &r.Kind, &r.Slot, &r.ProducerRequestID, &r.ProducerAttempt, &r.ProducerOutputSlot, &r.RetentionID, &r.InstanceID, &r.WorkerBootID, &r.State)
	return r, err
}
