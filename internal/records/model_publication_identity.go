package records

import (
	"crypto/sha256"
	"encoding/hex"
)

// ModelTransferOutputOperation is the single publication operation identity for
// one retained job output. Hub verification binds this operation to the exact
// immutable checkpoint before CompleteModelTransferOutput records it.
func ModelTransferOutputOperation(requestID, slot string) string {
	sum := sha256.Sum256([]byte(requestID + "\x00" + slot))
	return "model-artifact-" + hex.EncodeToString(sum[:])
}
