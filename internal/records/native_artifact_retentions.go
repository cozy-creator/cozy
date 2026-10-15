package records

const nativeArtifactRetentionsDDL = `CREATE TABLE IF NOT EXISTS native_artifact_retentions (
 artifact_kind TEXT NOT NULL DEFAULT 'derived' CHECK(artifact_kind IN ('derived','tree')),
 producer_attempt INTEGER NOT NULL DEFAULT 0, producer_output_id TEXT NOT NULL DEFAULT '',content_bytes INTEGER NOT NULL DEFAULT 0,
 consumer_id TEXT NOT NULL, parent_request_id TEXT NOT NULL REFERENCES requests(id),
 kind TEXT NOT NULL CHECK(kind IN ('input','result','effect')),slot TEXT NOT NULL,
 producer_id TEXT NOT NULL, manifest_id TEXT NOT NULL, manifest_length INTEGER NOT NULL,
 receipt_digest TEXT NOT NULL, transaction_id TEXT NOT NULL, owner_request_id TEXT NOT NULL, owner_worker TEXT NOT NULL,
 retention_id TEXT NOT NULL UNIQUE, instance_id TEXT NOT NULL DEFAULT '',worker_boot_id TEXT NOT NULL DEFAULT '',
 state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','held','releasing','released')),
 PRIMARY KEY(consumer_id,kind,slot)
)`

type NativeArtifactRetention struct {
	ArtifactKind, ProducerOutputID                                                                          string
	ProducerAttempt, ContentBytes                                                                           int64
	ConsumerID, ParentRequestID, Kind, Slot                                                                 string
	ProducerID, ManifestID                                                                                  string
	ManifestLength                                                                                          int64
	ReceiptDigest, TransactionID, OwnerRequestID, OwnerWorker, RetentionID, InstanceID, WorkerBootID, State string
}

const nativeArtifactCols = `artifact_kind,producer_attempt,producer_output_id,content_bytes,consumer_id,parent_request_id,kind,slot,producer_id,manifest_id,manifest_length,receipt_digest,transaction_id,owner_request_id,owner_worker,retention_id,instance_id,worker_boot_id,state`

func scanNativeArtifact(row interface{ Scan(...any) error }) (NativeArtifactRetention, error) {
	var h NativeArtifactRetention
	err := row.Scan(&h.ArtifactKind, &h.ProducerAttempt, &h.ProducerOutputID, &h.ContentBytes, &h.ConsumerID, &h.ParentRequestID, &h.Kind, &h.Slot, &h.ProducerID, &h.ManifestID, &h.ManifestLength, &h.ReceiptDigest, &h.TransactionID, &h.OwnerRequestID, &h.OwnerWorker, &h.RetentionID, &h.InstanceID, &h.WorkerBootID, &h.State)
	return h, err
}
