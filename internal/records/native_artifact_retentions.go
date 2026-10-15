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
