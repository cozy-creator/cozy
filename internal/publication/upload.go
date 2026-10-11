package publication

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

type UploadIntent struct {
	Request       UploadRequest `json:"request"`
	Objects       []hub.Object  `json:"objects"`
	ClosureDigest string        `json:"closure_digest"`
}
type ObjectUploader func(context.Context, hub.Grant, int64) *exit.Error
