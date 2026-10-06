package launch

import (
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type RetainedModelResult struct {
	Pointer   string
	Canonical []byte
	Artifact  records.ModelArtifact
	Retention *pb.DerivedRetentionRequest
}
