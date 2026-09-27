package cli

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func (r *Resolver) capturedResultInterface(request records.Request) (*launch.PackageInterface, *exit.Error) {
	link, problem := r.store.MachineExecution(request.ID)
	if problem != nil {
		return nil, problem
	}
	if (link == nil || len(link.Submission) == 0) && request.InstallID != "" {
		_, surface, problem := r.installPackageInterface(request.InstallID)
		return surface, problem
	}
	var submission pb.MachineExecutionSubmit
	var capture pb.MachineExecutionCapture
	if link == nil || proto.Unmarshal(link.Submission, &submission) != nil || canonical.Unmarshal(submission.CaptureCanonicalBytes, &capture) != nil {
		return nil, exit.New(exit.Conflict, "accepted execution metadata is unavailable")
	}
	for _, installed := range capture.InstalledPackages {
		if installed.InstallationId == capture.RootInstallationId {
			return launch.DecodePackageInterface(installed.PackageInterface)
		}
	}
	return nil, exit.New(exit.NotFound, "worker interface is absent from accepted execution")
}
