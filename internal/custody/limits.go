// Package custody owns this controller's bounded artifact and receipt metadata.
package custody

const (
	MaxWeightsReceipts              = 16
	MaxWeightsReceiptBytes          = 1 << 20
	MaxWeightsReceiptAggregateBytes = 4 << 20
	MaxMetadataBytes                = 4 << 20
	MaxChildArtifacts               = 32
	MaxSourceHeaderBytes            = MaxMetadataBytes
)

type ObjectRef struct {
	Digest []byte
	Length uint64
}
type TreeRef struct {
	ProducerRootID string
	ReceiptDigest  []byte
	Manifest       *ObjectRef
	ContentBytes   uint64
}
