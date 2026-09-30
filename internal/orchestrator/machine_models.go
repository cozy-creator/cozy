package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// PrivateRevisionModelRefs selects only exact inputs for this captured package.
// An imported child's unselected default is acquired when that child is invoked.
func PrivateRevisionModelRefs(request records.Request, pkg string) []*pb.DownloadModelRef {
	if pkg != request.Package {
		return nil
	}
	var models []ModelRef
	for _, model := range request.Models {
		if model.Choice && model.Callable != "" || model.Source != "" {
			continue // machine-owned choices travel separately from materialized downloads
		}
		if model.Package == pkg || model.Package == "" && request.Package == pkg {
			if model.Package == "" {
				model.Package = pkg
			}
			model.Adapters = nil // Runtime resolves the requested stack before returning the binding
			models = append(models, model)
		}
	}
	return downloadModelRefs(models)
}
